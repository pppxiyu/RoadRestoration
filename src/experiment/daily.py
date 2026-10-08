"""Run the reviewed daily experiment through the five existing module boundaries.

Problem supplies public data and frozen scenarios; environment owns scenario
truth and traffic; methods see observations only; analysis reads saved results.
The primary entry is main.py; retired problem runners are not active alternatives.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd

from src import config as P
from src.environment.daily import DailyEpisode, DailyTraffic
from src.environment.behavior_models import select_behavior_model
from src.environment.gravity import DEFAULT_MODEL
from src.experiment.layout import ROOT, daily_experiment_identity, publish_experiment_gallery
from src.methods.daily_s2v import DailyEncoder, DailyQ, rollout, train
from src.problem.daily import build_problem, make_splits, public_spec


_WORKER = None


def _initialize_worker(problem, cache_dir, hp, road_class_input=False, traffic_input=True):
    import torch
    torch.set_num_threads(1)
    global _WORKER
    simulator = DailyTraffic(problem, cache_dir)
    encoder = DailyEncoder(problem, simulator.ctx, road_class_input=road_class_input, traffic_input=traffic_input)
    _WORKER = (simulator, encoder, DailyQ(encoder, hp))


def _run_world(request):
    scenario, behavior, weights, epsilon, seed = request
    simulator, encoder, q_function = _WORKER
    q_function.net.load_state_dict(weights)
    q_function.net.eval()
    start_calls, start_hits = simulator.solve_calls, simulator.cache_hits
    started = time.perf_counter()
    episode = DailyEpisode(simulator, scenario, behavior, disconnected_flow="unserved")
    result = rollout(episode, encoder, q_function, epsilon=epsilon, rng=np.random.RandomState(seed))
    result["runtime"] = dict(seconds=time.perf_counter() - started,
                             ue_solves=simulator.solve_calls - start_calls,
                             cached_days=simulator.cache_hits - start_hits)
    return result


def _source_files():
    return sorted([ROOT / "main.py", ROOT / "requirements.txt", *ROOT.glob("src/**/*.py")])


def _source_digest():
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in _source_files()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def _write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    deadline = time.monotonic() + 120
    while True:
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(1)


def prepare(n=11, scenario_seed=P.SEED, *, road_class_input=True, traffic_input=True):
    problem = build_problem(n, scenario_seed)
    splits = make_splits(problem)
    simulator = DailyTraffic(problem)
    encoder = DailyEncoder(problem, simulator.ctx, road_class_input=road_class_input, traffic_input=traffic_input)
    folder, identity = daily_experiment_identity(problem, simulator, splits)
    return problem, splits, simulator, encoder, folder, identity


def save_problem(folder, problem, splits, identity, encoder):
    """Freeze a new problem, or verify existing evidence without rewriting it."""
    data = folder / "data"
    manifest_path = data / "manifest.json"
    normalized = json.loads(json.dumps(identity))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["identity"] != normalized:
            raise ValueError("existing daily experiment has different scoring rules")
        for name, scenarios in splits.items():
            path = data / "scenarios" / (name + ".json")
            if (not path.is_file() or json.loads(path.read_text(encoding="utf-8"))
                    != [s.record() for s in scenarios]):
                raise ValueError(f"frozen {name} scenarios differ or are missing")
        return
    # Also protect evidence left by an interruption before the manifest was saved.
    for name, scenarios in splits.items():
        path = data / "scenarios" / (name + ".json")
        if (path.exists() and json.loads(path.read_text(encoding="utf-8"))
                != [s.record() for s in scenarios]):
            raise ValueError(f"existing {name} scenarios differ")
    for name, scenarios in splits.items():
        _write_json(data / "scenarios" / (name + ".json"), [s.record() for s in scenarios])
    basis = data / "problem"
    basis.mkdir(parents=True, exist_ok=True)
    problem["edges"].to_csv(basis / "edges.csv", index=False)
    problem["od"].to_csv(basis / "od_pairs.csv", index=False)
    problem["baseline_links"].to_csv(basis / "normal_traffic.csv", index=False)
    pd.DataFrame([dict(rank=i + 1, edge_id=e, normal_two_way_flow=problem["baseline_flow"][e],
                       flow_bin="high" if i < 8 else "middle" if i < 29 else "low",
                       initially_public=e in problem["public_roads"])
                  for i, e in enumerate(problem["ranked"])]).to_csv(basis / "road_selection.csv", index=False)
    _write_json(basis / "public_problem.json", public_spec(problem))
    _write_json(basis / "gravity_model.json", json.loads(DEFAULT_MODEL.read_text(encoding="utf-8")))
    _write_json(data / "manifest.json", dict(identity=identity, feature_schema=encoder.specification(),
                training_samples=len(splits["train"]), validation_samples=len(splits["validation"]),
                testing_samples=len(splits["test"]),
                validation_environment="same as each model's training environment",
                common_testing_environment="response7d", delivery="final policy",
                fixed_total_horizon=None, voluntary_waiting=False, crew_accessibility_constraint=False))


def _save_evaluation(run_dir, trajectories, seconds):
    results = run_dir / "results"
    rows, daily_rows, actions = [], [], []
    feature_arrays = {}
    for i, trajectory in enumerate(trajectories):
        result = trajectory["result"]
        rows.append(dict(scenario=result["scenario"], F=result["objective"],
                         time_loss=result["time_loss"], flow_loss=result["flow_loss"],
                         recovery_day=result["recovery_day"], repair_completion_day=result["repair_completion_day"],
                         order="-".join(map(str, result["order"])),
                         **trajectory["runtime"]))
        daily_rows.extend(dict(scenario=result["scenario"], **row) for row in result["daily"])
        for step, (state, pick, reward) in enumerate(zip(trajectory["states"], trajectory["picks"], trajectory["rewards"])):
            actions.append(dict(scenario=result["scenario"], decision=step, day=state["day"],
                                chosen_road=state["roads"][pick], candidates=json.dumps(state["roads"]), reward=reward))
        feature_arrays[f"scenario_{i}_x"] = np.stack([s["x"] for s in trajectory["states"]])
        feature_arrays[f"scenario_{i}_g"] = np.stack([s["g"] for s in trajectory["states"]])
        feature_arrays[f"scenario_{i}_picks"] = np.asarray(trajectory["picks"], dtype=int)
    pd.DataFrame(rows).to_csv(results / "test_results.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(results / "test_daily.csv", index=False)
    pd.DataFrame(actions).to_csv(results / "test_actions.csv", index=False)
    np.savez_compressed(results / "test_policy_inputs.npz", **feature_arrays)
    _write_json(results / "test_schedules.json", [dict(scenario=t["result"]["scenario"],
                starts=t["result"]["starts"], completions=t["result"]["completions"]) for t in trajectories])
    summary = dict(evaluation_environment="response7d", test_scenarios=len(rows),
                   mean_objective=float(np.mean([r["F"] for r in rows])),
                   median_objective=float(np.median([r["F"] for r in rows])),
                   unique_solutions=len({r["order"] for r in rows}), evaluation_wall_seconds=seconds,
                   simulator_worker_seconds=sum(r["seconds"] for r in rows))
    _write_json(results / "evaluation_summary.json", summary)
    return summary


def run(n=11, training_behavior="both", seed=P.SEED, workers=4, *, scenario_seed=P.SEED,
        road_class_input=True, traffic_input=True):
    """Run either/both training environments with public road class by default.

    Explicit road_class_input=False retains the historical 14-column policy.
    Completed compatible runs are reused without overwriting their provenance.
    """
    import torch
    from src.analysis.viz.daily_viz import make_training_figures
    from src.analysis.daily import refresh_comparison
    if training_behavior not in ("both", "natural", "response7d"):
        raise ValueError("choose natural, response7d, or both")
    workers = max(1, min(int(workers), os.cpu_count() or 1))
    problem, splits, simulator, encoder, folder, identity = prepare(
        n, scenario_seed, road_class_input=road_class_input, traffic_input=traffic_input)
    save_problem(folder, problem, splits, identity, encoder)
    source_digest, source_hashes = _source_digest()
    cache = ROOT / ".cache" / "daily_traffic"
    behaviors = ("natural", "response7d") if training_behavior == "both" else (training_behavior,)
    print(f"daily experiment: {folder}; workers={workers}; training={behaviors}", flush=True)
    for behavior in behaviors:
        hp, stop_params, ep_cap = None, None, None
        if road_class_input:
            baseline = folder / "data/methods" / f"rl_s2v_saa64_adaptive_train_{behavior}_seed{seed}"
            contract_path = baseline / "config/training.json"
            if contract_path.exists():
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                if (contract["identity"]["experiment"] != json.loads(json.dumps(identity))
                        or contract["identity"]["training_behavior"] != behavior or contract["seed"] != seed):
                    raise ValueError("baseline training/scoring contract differs")
                hp, stop_params, ep_cap = contract["hyperparameters"], contract["stop"], contract["ep_cap"]
        # Each worker pool receives this behavior's matching training parameters.
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker,
                                 initargs=(problem, cache, hp, road_class_input, traffic_input)) as executor:
            select_behavior_model(behavior + "_daily", 24, for_methods=True)
            method = f"rl_s2v_saa64_adaptive_train_{behavior}"
            if road_class_input:
                method += "_roadclass"
            if not traffic_input:
                method += "_no_traffic"
            run_dir = folder / "data" / "methods" / f"{method}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_identity = dict(experiment=identity, training_behavior=behavior, code_sha256=source_digest)
            if road_class_input:
                run_identity["input_variant"] = "roadclass"
            if not traffic_input:
                run_identity["input_variant"] = "roadclass_no_traffic" if road_class_input else "no_traffic"
                run_identity["traffic_input"] = False
            status_path = run_dir / "log/status.json"
            if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete":
                completed = json.loads((run_dir / "config/training.json").read_text(encoding="utf-8"))
                expected_identity = {k: v for k, v in run_identity.items() if k != "code_sha256"}
                saved_identity = {k: v for k, v in completed["identity"].items() if k != "code_sha256"}
                if (saved_identity != json.loads(json.dumps(expected_identity))
                        or completed["seed"] != seed
                        or completed["feature_schema"] != json.loads(json.dumps(encoder.specification()))
                        or (hp is not None and completed["hyperparameters"] != hp)
                        or (stop_params is not None and completed["stop"] != stop_params)
                        or (ep_cap is not None and completed["ep_cap"] != ep_cap)):
                    raise ValueError(f"completed run has incompatible settings: {run_dir}")
                required = ("model_final.pt", "training_summary.json", "evaluation_summary.json", "test_results.csv")
                if not all((run_dir / "results" / name).is_file() for name in required):
                    raise ValueError(f"completed run is missing deliverables: {run_dir}")
                print(f"Reusing completed run: {run_dir.name}", flush=True)
                continue
            # Source provenance must agree with any checkpoint BEFORE overwriting
            # run config. The trainer performs the full contract check as well.
            previous = run_dir / "config" / "source_hashes.json"
            if previous.exists() and json.loads(previous.read_text()) != source_hashes:
                raise ValueError(f"source changed since this run began; preserve {run_dir} before rerunning")
            _write_json(previous, source_hashes)
            archive = run_dir / "config" / "source_snapshot.zip"
            if not archive.exists():
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                    for path in _source_files():
                        bundle.write(path, path.relative_to(ROOT))
            git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
            git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
            _write_json(run_dir / "config" / "provenance.json", dict(git_head=git_head,
                        dirty_worktree=bool(git_status.strip()), exact_source_archive="source_snapshot.zip",
                        source_sha256=source_digest, workers=workers,
                        python=platform.python_version(), platform=platform.platform(),
                        numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__))
            _write_json(run_dir / "log" / "status.json", dict(status="training", behavior=behavior))

            def batch(split, indices, weights, epsilon, seeds):
                if split not in ("train", "validation"):
                    raise ValueError("training may not request test scenarios")
                requests = [(splits[split][i], behavior, weights, epsilon, rollout_seed)
                            for i, rollout_seed in zip(indices, seeds)]
                return list(executor.map(_run_world, requests))

            def redraw(directory):
                make_training_figures(directory)
                publish_experiment_gallery(directory=folder)

            try:
                training_options = dict(hp=hp, stop_params=stop_params)
                if ep_cap is not None:
                    training_options["ep_cap"] = ep_cap
                policy, training = train(encoder, batch, train_count=len(splits["train"]),
                                         validation_count=len(splits["validation"]), seed=seed,
                                         run_dir=run_dir, identity=run_identity, redraw=redraw,
                                         **training_options)
                _write_json(run_dir / "log" / "status.json", dict(status="testing", **training))
                eval_started = time.perf_counter()
                requests = [(s, "response7d", policy.net.state_dict(), 0.0, 0) for s in splits["test"]]
                trajectories = list(executor.map(_run_world, requests))
                evaluation = _save_evaluation(run_dir, trajectories, time.perf_counter() - eval_started)
                _write_json(run_dir / "log" / "status.json", dict(status="complete", **training, **evaluation))
                refresh_comparison(folder)
                publish_experiment_gallery(directory=folder)
                print(method, training, evaluation, flush=True)
            except BaseException as exc:
                _write_json(run_dir / "log" / "status.json", dict(status="interrupted_or_failed",
                            error_type=type(exc).__name__, message=str(exc)))
                raise
    simulator.close()
    return folder


def compare(n=11):
    from src.analysis.daily import refresh_comparison
    *_, folder, identity = prepare(n)
    refresh_comparison(folder)
    publish_experiment_gallery(directory=folder)
    return folder


def ablation_plan(seed=42):
    """Prespecified contrasts; the full model is trained once and used in both."""
    base = "rl_s2v_saa64_adaptive_train_"
    natural = f"{base}natural_roadclass_seed{seed}"
    full = f"{base}response7d_roadclass_seed{seed}"
    no_traffic = f"{base}response7d_roadclass_no_traffic_seed{seed}"
    return dict(
        version=1, training_seed=seed, test_scenarios=50,
        evaluation_environment="response7d", validation_environment="same as training",
        road_class_input=True, traffic_ablation="zero four node and two global observed-traffic channels in every split",
        unchanged="network dimensions, optimizer, learning hyperparameters, stopping and final-policy delivery",
        methods=[dict(name=natural, training_behavior="natural", traffic_input=True),
                 dict(name=full, training_behavior="response7d", traffic_input=True),
                 dict(name=no_traffic, training_behavior="response7d", traffic_input=False)],
        comparisons=[dict(name="human_response", methods=[natural, full],
                          labels=["RL trained without\nhuman response", "Full RL"],
                          colors=["#527DA8", "#8265A3"],
                          filename="performance_distribution_human_response.png",
                          question="Does training with human response improve complete-environment performance?"),
                     dict(name="traffic_information", methods=[no_traffic, full],
                          labels=["RL without real-time\ntraffic inputs", "Full RL"],
                          colors=["#BA8855", "#8265A3"],
                          filename="performance_distribution_traffic_information.png",
                          question="Does observing live traffic improve performance under the same complete environment?")],
        figure_contract=dict(backend="python", format="PNG", archetype="quantitative comparison",
                             center="median", box="25th to 75th percentiles", whiskers="1.5 IQR",
                             points="all 50 paired test scenarios; not training replicates",
                             method_order="ablation, full model", shared_y_scale=True,
                             conclusion="measure each controlled difference without assuming the full model wins",
                             annotations=False, limitation="one training seed per policy"))


def prepare_ablation_experiment(n=11, seed=42, *, scenario_seed=P.SEED):
    problem, splits, simulator, encoder, folder, identity = prepare(n, scenario_seed)
    try:
        save_problem(folder, problem, splits, identity, encoder)
        plan = ablation_plan(seed)
        plan["experiment_identity"] = identity
        plan["source_sha256"] = _source_digest()[0]
        path = folder / "data/analysis/config/rl_ablation_plan.json"
        normalized = json.loads(json.dumps(plan))
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != normalized:
                raise ValueError("ablation plan/source changed; preserve this experiment before continuing")
        else:
            _write_json(path, plan)
        (folder / "data/log").mkdir(parents=True, exist_ok=True)
        return folder, plan
    finally:
        simulator.close()


def run_rl_ablations(n=11, seed=42, workers=4):
    """Sequential, resumable three-model experiment; no GA or historical reuse."""
    from src.experiment.daily_ga import single_runner
    folder, plan = prepare_ablation_experiment(n, seed)
    status_path = folder / "data/log/rl_ablation_status.json"
    started = time.perf_counter()
    completed = []
    with single_runner(folder / "data/log/rl_ablation_runner.lock"):
        try:
            for method in plan["methods"]:
                _write_json(status_path, dict(status="running", current_method=method["name"],
                            completed_methods=completed, pid=os.getpid(), workers=workers))
                result = run(n, method["training_behavior"], seed, workers,
                             road_class_input=True, traffic_input=method["traffic_input"])
                if result != folder:
                    raise AssertionError("ablation methods did not share the same experiment")
                path = folder / "data/methods" / method["name"] / "log/status.json"
                if json.loads(path.read_text(encoding="utf-8"))["status"] != "complete":
                    raise AssertionError("method returned without completing training and testing")
                completed.append(method["name"])
            compare(n)
            for pair in plan["comparisons"]:
                if not (folder / pair["filename"]).is_file():
                    raise AssertionError("requested comparison figure is missing")
            _write_json(status_path, dict(status="complete", completed_methods=completed,
                        attempt_wall_seconds=time.perf_counter() - started,
                        comparisons=[p["filename"] for p in plan["comparisons"]]))
        except BaseException as error:
            _write_json(status_path, dict(status="interrupted_or_failed", completed_methods=completed,
                        error_type=type(error).__name__, message=str(error), pid=os.getpid()))
            raise
    return folder
