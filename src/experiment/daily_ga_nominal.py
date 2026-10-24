"""GA on one training-derived eight-road nominal world; flow insertion at test time.

The nominal world is an approximation used only for search. The delivered policy
never reads hidden damage truth: discovered hidden roads are placed by public
normal-period flow among the still-unstarted public roads.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd

from src.experiment.daily import prepare, save_problem, _source_digest, _write_json
from src.experiment.daily_ga import single_runner
from src.environment.daily import DailyTraffic, DailyEpisode
from src.methods.daily_ga import initial_orders, search
from src.methods.metaheuristic import GA_PARAMS, BUDGET_CAP
from src.problem.daily import Scenario, RoadDamage


METHOD = "ga_nominal8_flow_insert"
FULL_RL = "rl_s2v_saa64_adaptive_train_response7d_roadclass_seed42"
_SIMULATOR = None


def nearest_day(value):
    return max(1, int(np.floor(float(value) + 0.5)))


def nominal_world(problem, training):
    """Train-only means; no validation/test truth and no hidden roads."""
    public = tuple(problem["public_roads"])
    if len(training) != 64 or len(public) != 8:
        raise ValueError("this comparison requires 64 training worlds and eight public roads")
    by_world = [{r.edge_id: r for r in scenario.roads} for scenario in training]
    if any(set(public) - set(world) for world in by_world):
        raise ValueError("a training world is missing a public road")
    rows = []
    for edge in public:
        observed = [world[edge] for world in by_world]
        rows.append(RoadDamage(edge_id=edge,
            estimated_severity=min(3, nearest_day(np.mean([r.estimated_severity for r in observed]))),
            estimated_duration=nearest_day(np.mean([r.estimated_duration for r in observed])),
            true_severity=min(3, nearest_day(np.mean([r.true_severity for r in observed]))),
            true_duration=nearest_day(np.mean([r.true_duration for r in observed])),
            discovery_day=0))
    return Scenario("nominal_public8_train64", tuple(rows))


def hybrid_order(candidates, public_order, baseline_flow):
    """Preserve GA public order; sort hidden roads by normal flow within gaps.

    Each hidden road goes before the first still-available public road with a
    strictly lower normal flow, otherwise after all public roads. Equal flows
    stay after that public road, with hidden ties ordered by road identity.
    """
    remaining = set(candidates)
    public = [e for e in public_order if e in remaining]
    if not remaining or len(public) + len(remaining - set(public_order)) != len(remaining):
        raise ValueError("unknown or empty candidate set")
    bins = [[] for _ in range(len(public) + 1)]
    for edge in remaining - set(public_order):
        if edge not in baseline_flow:
            raise ValueError("hidden road has no public normal-period flow")
        position = next((i for i, known in enumerate(public)
                         if baseline_flow[known] < baseline_flow[edge]), len(public))
        bins[position].append(edge)
    result = []
    for i, known in enumerate(public):
        result.extend(sorted(bins[i], key=lambda e: (-baseline_flow[e], e)))
        result.append(known)
    result.extend(sorted(bins[-1], key=lambda e: (-baseline_flow[e], e)))
    return tuple(result)


def _initialize(problem):
    global _SIMULATOR
    _SIMULATOR = DailyTraffic(problem, cache_dir=None)


def _evaluate(request):
    scenario, order, flow, hybrid = request
    start = time.perf_counter()
    previous_solves = _SIMULATOR.solve_calls
    episode = DailyEpisode(_SIMULATOR, scenario, "response7d", disconnected_flow="unserved")
    actions = []
    while not episode.finished:
        observed = episode.observation()
        ranked = hybrid_order(observed.candidates, order, flow) if hybrid else tuple(
            edge for edge in order if edge in observed.candidates)
        if not ranked:
            raise AssertionError("policy has no legal candidate")
        chosen = ranked[0]
        _, reward, _ = episode.step(chosen)
        actions.append(dict(day=observed.day, road=chosen,
                            candidates=list(observed.candidates), reward=reward))
    result = episode.result()
    if abs(sum(a["reward"] for a in actions) + result["objective"] - result["initial_loss"]) > 1e-8:
        raise AssertionError("daily reward and objective differ")
    return dict(result=result, actions=actions,
                seconds=time.perf_counter() - start,
                ue_solves=_SIMULATOR.solve_calls - previous_solves)


class NominalScores:
    """Reveal persisted candidate scores only in the unchanged GA query order."""
    def __init__(self, run_dir, executor, world, flow):
        self.run_dir, self.executor, self.world, self.flow = run_dir, executor, world, flow
        self.db = sqlite3.connect(run_dir / "results" / "nominal_scores.sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (priority TEXT PRIMARY KEY, score TEXT NOT NULL)")
        self.db.commit()
        self.started = time.perf_counter()
        self.generation = 0

    def score_batch(self, orders):
        results, pending = {}, []
        for order in orders:
            row = self.db.execute("SELECT score FROM scores WHERE priority=?", (json.dumps(order),)).fetchone()
            if row:
                results[order] = json.loads(row[0])
            else:
                pending.append(order)
        for order, trajectory in zip(pending, self.executor.map(
                _evaluate, [(self.world, order, self.flow, False) for order in pending])):
            result = trajectory["result"]
            score = [float(result[k]) for k in ("objective", "time_loss", "flow_loss")]
            if not np.isfinite(score).all():
                raise ValueError("nominal GA score is nonfinite")
            self.db.execute("INSERT INTO scores VALUES (?, ?)", (json.dumps(order), json.dumps(score)))
            self.db.commit()
            results[order] = score
            _write_json(self.run_dir / "log" / "status.json", dict(status="searching",
                generation=self.generation, evaluated_orders=self.db.execute(
                    "SELECT COUNT(*) FROM scores").fetchone()[0],
                elapsed_seconds=time.perf_counter() - self.started))
        return results


def _save_test(run_dir, trajectories, elapsed):
    rows, daily, actions = [], [], []
    for item in trajectories:
        result = item["result"]
        rows.append(dict(scenario=result["scenario"], F=result["objective"],
            time_loss=result["time_loss"], flow_loss=result["flow_loss"],
            recovery_day=result["recovery_day"],
            repair_completion_day=result["repair_completion_day"],
            order="-".join(map(str, result["order"])),
            seconds=item["seconds"], ue_solves=item["ue_solves"]))
        daily.extend(dict(scenario=result["scenario"], **day) for day in result["daily"])
        actions.extend(dict(scenario=result["scenario"], decision=i, **action)
                       for i, action in enumerate(item["actions"]))
    pd.DataFrame(rows).to_csv(run_dir / "results/test_results.csv", index=False)
    pd.DataFrame(daily).to_csv(run_dir / "results/test_daily.csv", index=False)
    pd.DataFrame(actions).to_csv(run_dir / "results/test_actions.csv", index=False)
    summary = dict(evaluation_environment="response7d", test_scenarios=len(rows),
        mean_objective=float(np.mean([r["F"] for r in rows])),
        median_objective=float(np.median([r["F"] for r in rows])),
        unique_solutions=len({r["order"] for r in rows}),
        evaluation_wall_seconds=elapsed,
        simulator_worker_seconds=sum(r["seconds"] for r in rows))
    _write_json(run_dir / "results/evaluation_summary.json", summary)
    return summary


def _compare(folder, run_dir, full_rl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.viz.daily_viz import STYLE

    manifest = json.loads((folder / "data/manifest.json").read_text(encoding="utf-8"))
    full_contract = json.loads((full_rl / "config/training.json").read_text(encoding="utf-8"))
    if full_contract["identity"]["experiment"] != manifest["identity"]:
        raise ValueError("full RL uses a different problem/scoring contract")
    expected = [s["scenario_id"] for s in json.loads(
        (folder / "data/scenarios/test.json").read_text(encoding="utf-8"))]
    frames = [pd.read_csv(path / "results/test_results.csv").set_index("scenario")
              for path in (run_dir, full_rl)]
    if any(len(frame) != 50 or frame.index.has_duplicates or set(frame.index) != set(expected)
           or not np.isfinite(frame.F).all() for frame in frames):
        raise ValueError("the comparison requires the same 50 finite test scenario scores")
    ga, rl = (frame.loc[expected, "F"].to_numpy() for frame in frames)
    gap = ga - rl
    draws = np.random.default_rng(42).choice(gap, (5000, 50), replace=True).mean(axis=1)
    comparison = dict(first=METHOD, second=full_rl.name,
        difference="GA minus full RL; positive favors full RL", test_scenarios=50,
        ga_mean=float(ga.mean()), rl_mean=float(rl.mean()),
        ga_median=float(np.median(ga)), rl_median=float(np.median(rl)),
        paired_mean_difference=float(gap.mean()),
        relative_improvement_of_rl=float(gap.mean() / ga.mean()),
        rl_wins=int((gap > 1e-9).sum()), ties=int((abs(gap) <= 1e-9).sum()),
        bootstrap_mean_95_interval=np.quantile(draws, [.025, .975]).tolist(),
        scope="scenario variation conditional on one search seed and one RL training seed")
    analysis = folder / "data/analysis"
    (analysis / "results").mkdir(parents=True, exist_ok=True)
    _write_json(analysis / "results/nominal_ga_vs_full_rl.json", comparison)
    pd.DataFrame(dict(scenario=expected, nominal_ga=ga, full_rl=rl,
                      ga_minus_rl=gap)).to_csv(
        analysis / "results/nominal_ga_vs_full_rl_paired.csv", index=False)
    maximum = max(float(max(ga.max(), rl.max())) * 1.08, 1.0)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 6.4), layout="constrained")
        boxes = ax.boxplot((ga, rl), patch_artist=True, widths=.46, showfliers=False,
                           medianprops=dict(color="#252525", linewidth=2),
                           whiskerprops=dict(linewidth=1.6), capprops=dict(linewidth=1.6))
        jitter = np.random.default_rng(0).uniform(-.08, .08, 50)
        for i, (box, values, color) in enumerate(zip(boxes["boxes"], (ga, rl),
                                                     ("#BA8855", "#8265A3")), 1):
            box.set_facecolor(color)
            box.set_alpha(.55)
            ax.scatter(i + jitter, values, s=24, color=color, alpha=.6, linewidths=0)
            ax.text(i, maximum * .985,
                    f"Mean {values.mean():.2f}   Median {np.median(values):.2f}",
                    ha="center", va="top", fontsize=15, color="#252525")
        ax.set_xticks([1, 2], ["Nominal GA + flow insertion", "Full RL"])
        ax.set_ylabel("Cumulative transport loss\n(equivalent days)")
        ax.set_xlim(.5, 2.5)
        ax.set_ylim(0, maximum)
        ax.grid(axis="y", alpha=.16)
        target = analysis / "performance_distribution_nominal_ga_vs_full_rl.png"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        plt.close(fig)
    shutil.copy2(target, folder / target.name)
    return comparison


from src.experiment.retired import retired_problem


@retired_problem
def run(n=11, seed=42, workers=4):
    if n != 11:
        raise ValueError("this reviewed nominal eight-road comparison is defined only for n11")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    problem, splits, simulator, encoder, folder, identity = prepare(n)
    try:
        save_problem(folder, problem, splits, identity, encoder)
        run_dir = folder / "data/methods" / f"{METHOD}_seed{seed}"
        full_rl = folder / "data/methods" / f"rl_s2v_saa64_adaptive_train_response7d_roadclass_seed{seed}"
        if not (full_rl / "results/test_results.csv").is_file():
            raise FileNotFoundError("completed Full RL results are required")
        for part in ("config", "results", "log"):
            (run_dir / part).mkdir(parents=True, exist_ok=True)
        world = nominal_world(problem, splits["train"])
        public = tuple(problem["public_roads"])
        seeds = [tuple(e for e in order if e in public)
                 for order in initial_orders(problem, simulator.ctx, splits["train"])]
        source, hashes = _source_digest()
        contract = dict(identity=dict(experiment=identity, training_behavior="ga_nominal_public",
                                      code_sha256=source),
            seed=seed, training_scenarios=64, validation_scenarios_used=0,
            evaluation_environment="response7d", search_environment="response7d",
            nominal_scenario=world.record(), hyperparameters=GA_PARAMS, budget=BUDGET_CAP,
            initial_orders=[list(order) for order in seeds], workers=workers,
            policy="GA public relative order; discovered hidden roads inserted by normal-period flow",
            averaging="per-public-road mean true severity and duration across 64 training worlds; half-up integer rounding",
            delivery="best nominal objective; test never selects GA order")
        config_path = run_dir / "config/training.json"
        with single_runner(run_dir / "log/runner.lock"):
            if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != json.loads(json.dumps(contract)):
                raise ValueError("existing nominal GA run has a different scientific/source contract")
            _write_json(config_path, contract)
            _write_json(run_dir / "config/source_hashes.json", hashes)
            if (run_dir / "log/status.json").exists():
                status = json.loads((run_dir / "log/status.json").read_text(encoding="utf-8"))
                if status.get("status") == "complete":
                    return folder
            try:
                with ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                                         initargs=(problem,)) as executor:
                    solution_path = run_dir / "results/solution.json"
                    if not solution_path.exists():
                        scores = NominalScores(run_dir, executor, world, problem["baseline_flow"])
                        def on_generation(generation, trace):
                            scores.generation = generation
                            pd.DataFrame(trace).to_csv(run_dir / "log/search.csv", index=False)
                            _write_json(run_dir / "log/status.json", dict(status="searching",
                                generation=generation, evaluated_orders=trace[-1]["cum_evals"],
                                best_nominal_objective=trace[-1]["best_F"],
                                elapsed_seconds=time.perf_counter() - scores.started))
                            print(json.dumps(dict(generation=generation, best=trace[-1]["best_F"],
                                                  evaluated=trace[-1]["cum_evals"])), flush=True)
                        (order, score), stop, trace = search(public, seeds, scores.score_batch,
                                                              seed, on_generation)
                        elapsed = time.perf_counter() - scores.started
                        scores.db.close()
                        training = dict(training_seconds=elapsed, search_seconds=elapsed,
                            generations=stop.gen, evaluated_orders=trace[-1]["cum_evals"],
                            outcome="plateau" if stop.done() else "budget_cap",
                            best_nominal_objective=score[0], training_scenarios=64)
                        pd.DataFrame(trace).to_csv(run_dir / "log/search.csv", index=False)
                        _write_json(run_dir / "results/training_summary.json", training)
                        _write_json(solution_path, dict(order=order, nominal_score=score, training=training))
                    else:
                        solution = json.loads(solution_path.read_text(encoding="utf-8"))
                        order, training = tuple(solution["order"]), solution["training"]
                    _write_json(run_dir / "log/status.json", dict(status="testing", completed=0, total=50))
                    started = time.perf_counter()
                    trajectories = []
                    for trajectory in executor.map(_evaluate, [(s, order, problem["baseline_flow"], True)
                                                               for s in splits["test"]]):
                        trajectories.append(trajectory)
                        _write_json(run_dir / "log/status.json", dict(
                            status="testing", completed=len(trajectories), total=50))
                    evaluation = _save_test(run_dir, trajectories, time.perf_counter() - started)
                comparison = _compare(folder, run_dir, full_rl)
                _write_json(run_dir / "log/status.json", dict(status="complete", **training, **evaluation,
                                                               paired_mean_difference=comparison["paired_mean_difference"]))
            except BaseException as error:
                _write_json(run_dir / "log/status.json", dict(status="interrupted_or_failed",
                    error_type=type(error).__name__, message=str(error), pid=os.getpid()))
                raise
        return folder
    finally:
        simulator.close()
