"""Run one SAA64 GA priority policy against the unchanged daily environment.

Only this experiment adapter holds scenario truth, to evaluate legal executions
and deduplicate physically identical schedules. The policy receives candidate
identities only. SQLite stores compressed full trajectories and candidate scores;
the unchanged GA can replay its deterministic search after a process failure.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import zipfile
import zlib

import numpy as np
import pandas as pd

from src.experiment.daily import ROOT, prepare, save_problem, _source_digest, _source_files
from src.experiment.layout import publish_experiment_gallery
from src.environment.daily import DailyTraffic, DailyEpisode
from src.methods.daily_ga import PriorityPolicy, possible_roads, initial_orders, search
from src.methods.metaheuristic import GA_PARAMS, BUDGET_CAP

_SIMULATOR = None


def write_json(path, value):
    """Atomic replace with retry for transient Windows file locks."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    until = time.monotonic() + 120
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= until:
                raise
            time.sleep(1)


@contextmanager
def single_runner(path):
    """OS releases the lock on process death; stale PID text is not a lock."""
    import msvcrt
    with Path(path).open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def physical_schedule(scenario, order):
    """Environment-side cache key only; this is NEVER supplied to the policy.

    With one nonpreemptive crew and no access constraints, fixed-priority start
    times depend only on realized durations and discovery days, not on traffic.
    The real simulator verifies this key for every newly evaluated schedule.
    """
    policy = PriorityPolicy(order)
    pending = {r.edge_id: r for r in scenario.roads}
    day, schedule = 1, []
    while pending:
        candidates = tuple(e for e, r in pending.items() if r.discovery_day <= day)
        if not candidates:
            day = max(day, min(r.discovery_day for r in pending.values()))
            continue
        road = policy.choose(candidates)
        duration = pending.pop(road).true_duration
        schedule.append((road, day, day + duration))
        day += duration
    return tuple(schedule)


def schedule_key(scenario, order):
    return json.dumps((scenario.signature, physical_schedule(scenario, order)), separators=(",", ":"))


def initialize_worker(problem):
    global _SIMULATOR
    # Per-worker bounded memory cache only: do not grow the RL traffic database.
    _SIMULATOR = DailyTraffic(problem, cache_dir=None)


def evaluate_priority(request):
    scenario, order = request
    simulator = _SIMULATOR
    started = time.perf_counter()
    calls, hits = simulator.solve_calls, simulator.cache_hits
    policy = PriorityPolicy(order)
    episode = DailyEpisode(simulator, scenario, "response7d", disconnected_flow="unserved")
    actions = []
    while not episode.finished:
        observation = episode.observation()
        road = policy.choose(observation.candidates)
        _, reward, _ = episode.step(road)
        actions.append(dict(day=observation.day, road=road,
                            candidates=list(observation.candidates), reward=reward))
    result = episode.result()
    actual = tuple((e, result["starts"][e], result["completions"][e]) for e in result["order"])
    if actual != physical_schedule(scenario, order):
        raise AssertionError("cache schedule differs from the real daily environment")
    if abs(sum(a["reward"] for a in actions) + result["objective"] - result["initial_loss"]) > 1e-8:
        raise AssertionError("GA reward/score mismatch")
    return dict(result=result, actions=actions, runtime=dict(seconds=time.perf_counter()-started,
                ue_solves=simulator.solve_calls-calls, cached_days=simulator.cache_hits-hits))


class EvaluationStore:
    def __init__(self, run, executor, scenarios):
        if len(scenarios) != 64 or any(not s.scenario_id.startswith("train_") for s in scenarios):
            raise ValueError("GA search requires exactly the 64 training scenarios")
        self.run, self.executor, self.scenarios = Path(run), executor, scenarios
        self.db = sqlite3.connect(self.run / "results/evaluations.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS worlds (key TEXT PRIMARY KEY, payload BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS candidates (key TEXT PRIMARY KEY, score TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value REAL)")
        row = self.db.execute("SELECT value FROM metadata WHERE key='search_seconds'").fetchone()
        self.elapsed_before = row[0] if row else 0.0
        self.started = time.perf_counter()
        self.generation = 0

    def elapsed(self):
        return self.elapsed_before + time.perf_counter() - self.started

    def progress(self, **extra):
        seconds = self.elapsed()
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('search_seconds', ?)", (seconds,))
        self.db.commit()
        status = dict(status="searching", pid=os.getpid(), generation=self.generation,
                      training_scenarios=64, search_seconds=seconds,
                      evaluated_orders=self.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
                      distinct_world_schedules=self.db.execute("SELECT COUNT(*) FROM worlds").fetchone()[0],
                      updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        write_json(self.run / "log/status.json", status)

    def world(self, key):
        row = self.db.execute("SELECT payload FROM worlds WHERE key=?", (key,)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def score_batch(self, orders):
        scores, todo, keys = {}, {}, {}
        for order in orders:
            encoded = json.dumps(order)
            old = self.db.execute("SELECT score FROM candidates WHERE key=?", (encoded,)).fetchone()
            if old:
                scores[order] = tuple(json.loads(old[0]))
                continue
            keys[order] = []
            for scenario in self.scenarios:
                key = schedule_key(scenario, order)
                keys[order].append(key)
                if not self.db.execute("SELECT 1 FROM worlds WHERE key=?", (key,)).fetchone():
                    todo.setdefault(key, (scenario, order))
        self.progress(pending_world_schedules=len(todo))
        futures = {self.executor.submit(evaluate_priority, request): key for key, request in todo.items()}
        remaining = len(futures)
        for future in as_completed(futures):
            try:
                trajectory = future.result()  # numerical failures are explicit, never penalty scores
            except BaseException as error:
                key = futures[future]
                world, order = todo[key]
                write_json(self.run / "log/evaluation_failure.json", dict(scenario=world.scenario_id,
                           priority_order=order, error_type=type(error).__name__, message=str(error)))
                for pending in futures:
                    pending.cancel()
                # Python 3.14: do not wait for a whole unfinished generation
                # after a confirmed worker error. Already committed scores remain.
                if hasattr(self.executor, "terminate_workers"):
                    self.executor.terminate_workers()
                raise
            key = futures[future]
            blob = zlib.compress(json.dumps(trajectory, allow_nan=False).encode(), level=3)
            self.db.execute("INSERT INTO worlds VALUES (?, ?)", (key, blob))
            remaining -= 1
            self.progress(pending_world_schedules=remaining)
        for order, world_keys in keys.items():
            results = [self.world(key)["result"] for key in world_keys]
            if len(results) != 64:
                raise AssertionError("missing training scenario in GA score")
            score = tuple(float(np.mean([r[k] for r in results])) for k in ("objective", "time_loss", "flow_loss"))
            scores[order] = score
            self.db.execute("INSERT INTO candidates VALUES (?, ?)", (json.dumps(order), json.dumps(score)))
        self.progress(pending_world_schedules=0)
        return scores


def run(n=11, seed=42, workers=4):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    problem, splits, simulator, encoder, folder, identity = prepare(n)
    normalized = json.loads(json.dumps(identity))
    # Execution order from one historical experiment is not a problem rule.
    # GA and RL freeze/verify the same current scenarios, independently of which
    # method ran first or whether its policy includes the road-class ablation.
    save_problem(folder, problem, splits, identity, encoder)
    run_dir = folder / "data/methods" / f"ga_saa64_fixed_priority_seed{seed}"
    for subdir in ("config", "results", "log"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    roads = possible_roads(problem)
    seeds = initial_orders(problem, simulator.ctx, splits["train"])
    digest, hashes = _source_digest()
    contract = dict(identity=dict(experiment=normalized, training_behavior="ga", code_sha256=digest),
                    seed=seed, hyperparameters=GA_PARAMS, budget=BUDGET_CAP,
                    training_scenarios=64, evaluation_environment="response7d", workers=workers,
                    policy="one priority order; choose highest-ranked currently discovered unrepaired road",
                    possible_roads=list(roads), initial_orders=[list(p) for p in seeds],
                    initialization="legacy flow/exposure/exposure-per-duration seeds; estimated labels averaged over training occurrences",
                    resume="replay unchanged deterministic GA; persistent scores revealed only on original query",
                    delivery="lowest mean training objective; test never used to select order")
    with single_runner(run_dir / "log/runner.lock"):
        prior = run_dir / "config/training.json"
        if prior.exists() and json.loads(prior.read_text()) != contract:
            raise ValueError("GA source or contract changed; preserve old results before a new run")
        write_json(prior, contract)
        write_json(run_dir / "config/source_hashes.json", hashes)
        archive = run_dir / "config/source_snapshot.zip"
        if not archive.exists():
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                for path in _source_files():
                    bundle.write(path, path.relative_to(ROOT))
        if (run_dir / "log/status.json").exists():
            previous_status = json.loads((run_dir / "log/status.json").read_text())
            if previous_status.get("status") == "complete":
                return folder
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                     initargs=(problem,)) as executor:
                store = EvaluationStore(run_dir, executor, splits["train"])

                def on_generation(generation, trace):
                    store.generation = generation
                    pd.DataFrame(trace).to_csv(run_dir / "log/search.csv", index=False)
                    store.progress(best_training_objective=trace[-1]["best_F"])
                    print(json.dumps(dict(generation=generation, best=trace[-1]["best_F"],
                                          evaluations=trace[-1]["cum_evals"], seconds=store.elapsed())), flush=True)
                    if generation % 3 == 0:
                        from src.analysis.viz.daily_viz import make_ga_search
                        make_ga_search(pd.DataFrame(trace), run_dir)
                        publish_experiment_gallery(directory=folder)

                committed = run_dir / "results/solution.json"
                if not committed.exists():
                    (order, score), stop, trace = search(roads, seeds, store.score_batch, seed, on_generation)
                    training = dict(training_seconds=store.elapsed(), search_seconds=store.elapsed(),
                                    generations=stop.gen, evaluated_orders=trace[-1]["cum_evals"],
                                    outcome="plateau" if stop.done() else "budget_cap",
                                    best_training_objective=score[0], training_scenarios=64)
                    pd.DataFrame(trace).to_csv(run_dir / "log/search.csv", index=False)
                    write_json(run_dir / "results/training_summary.json", training)
                    write_json(committed, dict(order=order, mean_training_score=score[0], training=training))
                else:
                    solution = json.loads(committed.read_text())
                    order, training = tuple(solution["order"]), solution["training"]
                store.db.close()
                started = time.perf_counter()
                trajectories = []
                for trajectory in executor.map(evaluate_priority, [(s, order) for s in splits["test"]]):
                    trajectories.append(trajectory)
                    write_json(run_dir / "log/status.json", dict(status="testing", completed=len(trajectories), total=50))
                elapsed = time.perf_counter() - started
            rows, daily, actions = [], [], []
            for trajectory in trajectories:
                result = trajectory["result"]
                rows.append(dict(scenario=result["scenario"], F=result["objective"],
                                 time_loss=result["time_loss"], flow_loss=result["flow_loss"],
                                 recovery_day=result["recovery_day"], repair_completion_day=result["repair_completion_day"],
                                 order="-".join(map(str, result["order"])), **trajectory["runtime"]))
                daily.extend(dict(scenario=result["scenario"], **day) for day in result["daily"])
                actions.extend(dict(scenario=result["scenario"], decision=i, **a) for i, a in enumerate(trajectory["actions"]))
            pd.DataFrame(rows).to_csv(run_dir / "results/test_results.csv", index=False)
            pd.DataFrame(daily).to_csv(run_dir / "results/test_daily.csv", index=False)
            pd.DataFrame(actions).to_csv(run_dir / "results/test_actions.csv", index=False)
            write_json(run_dir / "results/test_schedules.json", [dict(scenario=t["result"]["scenario"],
                       starts=t["result"]["starts"], completions=t["result"]["completions"]) for t in trajectories])
            evaluation = dict(evaluation_environment="response7d", test_scenarios=len(rows),
                              mean_objective=float(np.mean([r["F"] for r in rows])),
                              median_objective=float(np.median([r["F"] for r in rows])),
                              unique_solutions=len({r["order"] for r in rows}), committed_priority_policies=1,
                              evaluation_wall_seconds=elapsed,
                              simulator_worker_seconds=sum(r["seconds"] for r in rows))
            write_json(run_dir / "results/evaluation_summary.json", evaluation)
            from src.analysis.viz.daily_viz import make_ga_search
            from src.analysis.daily import refresh_comparison
            make_ga_search(pd.read_csv(run_dir / "log/search.csv"), run_dir)
            refresh_comparison(folder)
            publish_experiment_gallery(directory=folder)
            write_json(run_dir / "log/status.json", dict(status="complete", **training, **evaluation))
        except BaseException as error:
            write_json(run_dir / "log/status.json", dict(status="interrupted_or_failed", error_type=type(error).__name__,
                       message=str(error), pid=os.getpid(), updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            raise
        finally:
            simulator.close()
    return folder
