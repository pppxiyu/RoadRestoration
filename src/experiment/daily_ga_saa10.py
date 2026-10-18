"""A ten-training-world GA with flow insertion for previously unseen roads.

The existing genetic operators and stopping rule are unchanged. Each candidate
priority order is scored on the same frozen ten complete training scenarios.
The test policy sees candidate identities and public normal-period flows only.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import zipfile

import numpy as np
import pandas as pd

from src.experiment.daily import (ROOT, prepare, save_problem, _source_digest,
                                  _source_files, _write_json)
from src.experiment.daily_ga import physical_schedule, schedule_key, single_runner
from src.experiment.daily_ga_nominal import _initialize, _evaluate, _save_test
from src.methods.daily_ga import initial_orders, possible_roads, search
from src.methods.metaheuristic import GA_PARAMS, BUDGET_CAP


METHOD = "ga_saa10_flow_insert"
FULL_RL_PREFIX = "rl_s2v_saa64_adaptive_train_response7d_roadclass"


def training_subset(scenarios, seed):
    """Select 10 of the frozen 64 training worlds once, without replacement."""
    if len(scenarios) != 64 or any(not s.scenario_id.startswith("train_") for s in scenarios):
        raise ValueError("expected the frozen 64 training scenarios")
    indices = sorted(np.random.default_rng(int(seed)).choice(64, 10, replace=False).tolist())
    return indices, [scenarios[i] for i in indices]


class TenScenarioScores:
    """Persistent schedule and candidate scores for deterministic GA replay."""
    def __init__(self, run_dir, executor, scenarios, flows):
        self.run_dir, self.executor = Path(run_dir), executor
        self.scenarios, self.flows = scenarios, flows
        self.db = sqlite3.connect(self.run_dir / "results/evaluations.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS worlds (key TEXT PRIMARY KEY, score TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS candidates (priority TEXT PRIMARY KEY, score TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value REAL NOT NULL)")
        row = self.db.execute("SELECT value FROM metadata WHERE key='search_seconds'").fetchone()
        self.elapsed_before = float(row[0]) if row else 0.0
        self.started = time.perf_counter()
        self.generation = 0

    def elapsed(self):
        return self.elapsed_before + time.perf_counter() - self.started

    def progress(self, **extra):
        elapsed = self.elapsed()
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('search_seconds', ?)", (elapsed,))
        self.db.commit()
        _write_json(self.run_dir / "log/status.json", dict(
            status="searching", generation=self.generation,
            evaluated_orders=self.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
            distinct_world_schedules=self.db.execute("SELECT COUNT(*) FROM worlds").fetchone()[0],
            search_seconds=elapsed, **extra))

    def score_batch(self, orders):
        scores, keys_by_order, todo = {}, {}, {}
        for order in orders:
            row = self.db.execute("SELECT score FROM candidates WHERE priority=?",
                                  (json.dumps(order),)).fetchone()
            if row is not None:
                scores[order] = json.loads(row[0])
                continue
            keys = []
            for scenario in self.scenarios:
                key = schedule_key(scenario, order)
                keys.append(key)
                if self.db.execute("SELECT 1 FROM worlds WHERE key=?", (key,)).fetchone() is None:
                    todo.setdefault(key, (scenario, order))
            keys_by_order[order] = keys
        self.progress(pending_world_schedules=len(todo))
        futures = {self.executor.submit(_evaluate, (scenario, order, self.flows, False)): key
                   for key, (scenario, order) in todo.items()}
        for future in as_completed(futures):
            key = futures[future]
            scenario, order = todo[key]
            trajectory = future.result()
            result = trajectory["result"]
            actual = tuple((edge, result["starts"][edge], result["completions"][edge])
                           for edge in result["order"])
            if actual != physical_schedule(scenario, order):
                raise AssertionError("cached physical schedule differs from the real environment")
            score = [float(result[name]) for name in ("objective", "time_loss", "flow_loss")]
            if not np.isfinite(score).all():
                raise ValueError("nonfinite scenario score")
            self.db.execute("INSERT INTO worlds VALUES (?, ?)", (key, json.dumps(score)))
            self.progress(pending_world_schedules=len(futures) -
                          self.db.execute("SELECT COUNT(*) FROM worlds WHERE key IN (" +
                                          ",".join("?" for _ in futures) + ")",
                                          tuple(futures.values())).fetchone()[0])
        for order, keys in keys_by_order.items():
            world_scores = [json.loads(self.db.execute(
                "SELECT score FROM worlds WHERE key=?", (key,)).fetchone()[0]) for key in keys]
            if len(world_scores) != 10:
                raise AssertionError("a candidate was not evaluated on all ten scenarios")
            score = np.mean(world_scores, axis=0).tolist()
            scores[order] = score
            self.db.execute("INSERT INTO candidates VALUES (?, ?)", (json.dumps(order), json.dumps(score)))
        self.progress(pending_world_schedules=0)
        return scores


def compare(folder, run_dir, full_rl, *, output_stem="ga_saa10_vs_full_rl", rl_label="Full RL"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.viz.daily_viz import STYLE

    manifest = json.loads((folder / "data/manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((full_rl / "config/training.json").read_text(encoding="utf-8"))
    if contract["identity"]["experiment"] != manifest["identity"]:
        raise ValueError("full RL has a different experiment identity")
    ga_contract = json.loads((run_dir / "config/training.json").read_text(encoding="utf-8"))
    if ga_contract["identity"]["experiment"] != manifest["identity"]:
        raise ValueError("GA has a different experiment identity")
    expected = [row["scenario_id"] for row in json.loads(
        (folder / "data/scenarios/test.json").read_text(encoding="utf-8"))]
    frames = [pd.read_csv(path / "results/test_results.csv").set_index("scenario")
              for path in (run_dir, full_rl)]
    if any(len(frame) != 50 or frame.index.has_duplicates or set(frame.index) != set(expected)
           or not np.isfinite(frame.F).all() for frame in frames):
        raise ValueError("both methods must have the same 50 finite test results")
    ga, rl = (frame.loc[expected, "F"].to_numpy() for frame in frames)
    delta = ga - rl
    draws = np.random.default_rng(42).choice(delta, (5000, 50), replace=True).mean(axis=1)
    analysis = folder / "data/analysis"
    (analysis / "results").mkdir(parents=True, exist_ok=True)
    result = dict(first=run_dir.name, second=full_rl.name,
        difference="ten-scenario GA minus full RL; positive favors full RL",
        test_scenarios=50, ga_mean=float(ga.mean()), rl_mean=float(rl.mean()),
        ga_median=float(np.median(ga)), rl_median=float(np.median(rl)),
        paired_mean_difference=float(delta.mean()),
        relative_improvement_of_rl=float(delta.mean() / ga.mean()),
        rl_wins=int((delta > 1e-9).sum()), ties=int((abs(delta) <= 1e-9).sum()),
        bootstrap_mean_95_interval=np.quantile(draws, [.025, .975]).tolist(),
        scope="scenario variation conditional on one GA search seed and one RL training seed")
    _write_json(analysis / "results" / (output_stem + ".json"), result)
    pd.DataFrame(dict(scenario=expected, ga_saa10=ga, full_rl=rl, ga_minus_rl=delta)).to_csv(
        analysis / "results" / (output_stem + "_paired.csv"), index=False)
    y_max = max(float(max(ga.max(), rl.max())) * 1.08, 1.0)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 6.4), layout="constrained")
        boxes = ax.boxplot((ga, rl), patch_artist=True, widths=.46, showfliers=False,
                           medianprops=dict(color="#252525", linewidth=2),
                           whiskerprops=dict(linewidth=1.6), capprops=dict(linewidth=1.6))
        jitter = np.random.default_rng(0).uniform(-.08, .08, 50)
        for position, (box, values, color) in enumerate(
                zip(boxes["boxes"], (ga, rl), ("#418C8B", "#8265A3")), 1):
            box.set_facecolor(color)
            box.set_alpha(.55)
            ax.scatter(position + jitter, values, s=24, color=color, alpha=.6, linewidths=0)
            ax.text(position, y_max * .985,
                    f"Mean {values.mean():.2f}   Median {np.median(values):.2f}",
                    ha="center", va="top", fontsize=15, color="#252525")
        ax.set_xticks([1, 2], ["10-scenario GA\n+ flow insertion", rl_label])
        ax.set_ylabel("Cumulative transport loss\n(equivalent days)")
        ax.set_xlim(.5, 2.5)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=.16)
        target = analysis / ("performance_distribution_" + output_stem + ".png")
        fig.savefig(target, dpi=300, bbox_inches="tight")
        plt.close(fig)
    shutil.copy2(target, folder / target.name)
    return result


def run(n=11, seed=42, workers=4, *, problem_variant="current", full_rl_method=None):
    if n != 11:
        raise ValueError("the reviewed ten-scenario comparison is defined for n11")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    problem, splits, simulator, encoder, folder, identity = prepare(n, problem_variant=problem_variant)
    try:
        save_problem(folder, problem, splits, identity, encoder)
        indices, train = training_subset(splits["train"], seed)
        roads = tuple(sorted({road.edge_id for scenario in train for road in scenario.roads}))
        possible = set(possible_roads(problem))
        if not set(roads) <= possible or not set(problem["public_roads"]) <= set(roads):
            raise ValueError("training road union is invalid")
        seeds = [tuple(road for road in order if road in roads)
                 for order in initial_orders(problem, simulator.ctx, train)]
        run_dir = folder / "data/methods" / f"{METHOD}_seed{seed}"
        full_rl = folder / "data/methods" / (full_rl_method or f"{FULL_RL_PREFIX}_seed{seed}")
        if not (full_rl / "results/test_results.csv").is_file():
            raise FileNotFoundError("completed full RL test results are required")
        if full_rl_method is not None:
            rl_config = json.loads((full_rl / "config/training.json").read_text(encoding="utf-8"))
            rl_status = json.loads((full_rl / "log/status.json").read_text(encoding="utf-8"))
            if (rl_status["status"] != "complete" or
                    rl_config["identity"]["experiment"] != json.loads(json.dumps(identity))):
                raise ValueError("requested comparison RL is incomplete or uses a different experiment")
        for part in ("config", "log", "results"):
            (run_dir / part).mkdir(parents=True, exist_ok=True)
        source, hashes = _source_digest()
        contract = dict(identity=dict(experiment=identity, training_behavior="ga_saa10",
                                      code_sha256=source), seed=seed,
            training_scenarios=10, selected_training_indices=indices,
            selected_training_ids=[world.scenario_id for world in train],
            train_road_union=list(roads), unseen_possible_roads=sorted(possible - set(roads)),
            hyperparameters=GA_PARAMS, budget=BUDGET_CAP, workers=workers,
            initial_orders=[list(order) for order in seeds],
            evaluation_environment="response7d", search_environment="response7d",
            policy="one priority order for roads seen in ten training worlds; unseen discovered roads inserted by normal-period flow among currently available known roads",
            delivery="lowest ten-world mean training objective; test never used to choose order")
        with single_runner(run_dir / "log/runner.lock"):
            config_path = run_dir / "config/training.json"
            if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != json.loads(json.dumps(contract)):
                raise ValueError("existing run has a different source or scientific contract")
            _write_json(config_path, contract)
            _write_json(run_dir / "config/source_hashes.json", hashes)
            archive = run_dir / "config/source_snapshot.zip"
            if not archive.exists():
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                    for path in _source_files():
                        bundle.write(path, path.relative_to(ROOT))
            status_path = run_dir / "log/status.json"
            if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete":
                return folder
            try:
                with ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                                         initargs=(problem,)) as executor:
                    solution_path = run_dir / "results/solution.json"
                    if not solution_path.exists():
                        store = TenScenarioScores(run_dir, executor, train, problem["baseline_flow"])
                        def on_generation(generation, trace):
                            store.generation = generation
                            pd.DataFrame(trace).to_csv(run_dir / "log/search.csv", index=False)
                            store.progress(best_training_objective=trace[-1]["best_F"])
                            print(json.dumps(dict(generation=generation,
                                best=trace[-1]["best_F"], evaluated=trace[-1]["cum_evals"],
                                seconds=store.elapsed())), flush=True)
                        (order, score), stop, trace = search(roads, seeds, store.score_batch,
                                                              seed, on_generation)
                        elapsed = store.elapsed()
                        store.db.close()
                        training = dict(training_seconds=elapsed, search_seconds=elapsed,
                            generations=stop.gen, evaluated_orders=trace[-1]["cum_evals"],
                            outcome="plateau" if stop.done() else "budget_cap",
                            best_training_objective=score[0], training_scenarios=10)
                        pd.DataFrame(trace).to_csv(run_dir / "log/search.csv", index=False)
                        _write_json(run_dir / "results/training_summary.json", training)
                        _write_json(solution_path, dict(order=order, mean_training_score=score,
                                                        training=training))
                    else:
                        solution = json.loads(solution_path.read_text(encoding="utf-8"))
                        order, training = tuple(solution["order"]), solution["training"]
                    started = time.perf_counter()
                    trajectories = []
                    _write_json(status_path, dict(status="testing", completed=0, total=50))
                    for trajectory in executor.map(_evaluate, [(world, order, problem["baseline_flow"], True)
                                                               for world in splits["test"]]):
                        trajectories.append(trajectory)
                        _write_json(status_path, dict(status="testing", completed=len(trajectories), total=50))
                    evaluation = _save_test(run_dir, trajectories, time.perf_counter() - started)
                comparison = compare(folder, run_dir, full_rl)
                _write_json(status_path, dict(status="complete", **training, **evaluation,
                                              paired_mean_difference=comparison["paired_mean_difference"]))
            except BaseException as error:
                _write_json(status_path, dict(status="interrupted_or_failed",
                    error_type=type(error).__name__, message=str(error), pid=os.getpid()))
                raise
        return folder
    finally:
        simulator.close()
