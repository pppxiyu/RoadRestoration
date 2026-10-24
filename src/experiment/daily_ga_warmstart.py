"""GA demonstration initialization, validation-only screening, then standard RL.

No plots are produced. Historical methods are read but never overwritten.
Run with python -X utf8 -u -m src.experiment.daily_ga_warmstart.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import zipfile

import numpy as np
import pandas as pd

from src.experiment import daily
from src.experiment.daily_ga import single_runner, physical_schedule, schedule_key
from src.experiment.daily_ga_nominal import hybrid_order
from src.methods.daily_s2v import train
from src.methods.daily_imitation import pretrain
from src.environment.daily import DailyEpisode

FULL = "rl_s2v_saa64_adaptive_train_response7d_roadclass_seed42"
TEACHER = "ga_saa10_flow_insert_seed42"
CAMPAIGN = "ga10_initialized_full_rl_seed42"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_torch(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    deadline = time.monotonic() + 120
    while True:
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(1)


def teacher_rollout(request):
    scenario, priority, flow = request
    simulator, encoder, _ = daily._WORKER
    start = time.perf_counter()
    calls, hits = simulator.solve_calls, simulator.cache_hits
    episode = DailyEpisode(simulator, scenario, "response7d", disconnected_flow="unserved")
    states, picks, rewards = [], [], []
    while not episode.finished:
        state = encoder.encode(episode.observation())
        chosen = hybrid_order(state["roads"], priority, flow)[0]
        pick = state["roads"].index(chosen)
        _, reward, _ = episode.step(chosen)
        states.append(state)
        picks.append(pick)
        rewards.append(float(reward))
    result = episode.result()
    if abs(sum(rewards) + result["objective"] - result["initial_loss"]) > 1e-8:
        raise AssertionError("teacher return and environment objective differ")
    return dict(states=states, picks=picks, rewards=rewards, result=result,
                runtime=dict(seconds=time.perf_counter() - start,
                             ue_solves=simulator.solve_calls - calls, cached_days=simulator.cache_hits - hits))


def screen_score(run_dir):
    trace = pd.read_csv(run_dir / "log/training.csv")
    probes = trace.dropna(subset=["validation_objective"]).tail(3)
    return float(probes.validation_objective.mean())


def paired_report(current, reference):
    merged = current.merge(reference, on="scenario", suffixes=("_new", "_reference"), validate="one_to_one")
    if len(merged) != 50 or len(current) != len(reference):
        raise AssertionError("test scenario mismatch")
    improvement = (merged.F_reference - merged.F_new).to_numpy()
    rng = np.random.default_rng(4208)
    boot = improvement[rng.integers(0, len(improvement), (10000, len(improvement)))].mean(axis=1)
    return dict(mean_reduction=float(improvement.mean()),
                percentage_reduction=100 * float(improvement.mean() / merged.F_reference.mean()),
                wins=int((improvement > 1e-9).sum()), ties=int((abs(improvement) <= 1e-9).sum()),
                mean_reduction_bootstrap95=np.quantile(boot, [.025, .975]).tolist())


from src.experiment.retired import retired_problem


@retired_problem
def run(workers=4):
    import torch
    torch.set_num_threads(1)
    problem, splits, simulator, encoder, folder, identity = daily.prepare(11, 42)
    daily.save_problem(folder, problem, splits, identity, encoder)
    if folder.name != "n11_discovery_response7d_1day_lhs50_s42_368acec5":
        raise ValueError("problem identity changed")
    methods = folder / "data/methods"
    root = methods / CAMPAIGN
    for name in ("config", "results", "log"):
        (root / name).mkdir(parents=True, exist_ok=True)
    status_path = root / "log/status.json"
    full_contract = json.loads((methods / FULL / "config/training.json").read_text())
    teacher_path = methods / TEACHER / "results/solution.json"
    priority = tuple(json.loads(teacher_path.read_text())["order"])
    hp = full_contract["hyperparameters"]
    if full_contract["feature_schema"] != json.loads(json.dumps(encoder.specification())):
        raise AssertionError("network information schema differs from Full RL")
    if full_contract["identity"]["experiment"] != json.loads(json.dumps(identity)):
        raise AssertionError("Full RL used a different experiment")
    teacher_contract = json.loads((methods / TEACHER / "config/training.json").read_text())
    if teacher_contract["identity"]["experiment"] != json.loads(json.dumps(identity)):
        raise AssertionError("GA teacher used a different experiment")
    source_hash, source_files = daily._source_digest()
    plan = dict(version=1, experiment=identity, teacher=TEACHER,
        teacher_solution_sha256=digest(teacher_path), source_sha256=source_hash,
        train_scenario_sha256=digest(folder / "data/scenarios/train.json"),
        validation_scenario_sha256=digest(folder / "data/scenarios/validation.json"),
        test_scenario_sha256=digest(folder / "data/scenarios/test.json"),
        training_demonstrations=64, validation_demonstrations_used_for_gradient=False,
        pretraining=dict(seed=42, updates=[200, 600, 1200], batch=32, lr=hp["lr"],
                         loss="legal-action cross entropy(Q/0.2) + 0.1 smooth-L1(chosen Q, teacher discounted return)",
                         gradient_clip=10, selection="smallest validation mean among three fixed checkpoints"),
        initialization="same pretrained weights plus training demo replay and per-world incumbent exemplars",
        online_variants=dict(original_exploration={}, reduced_exploration=dict(eps0=0.2)),
        screening=dict(completed_episodes=76, selection="minimum average of last three validation probes; ties use original exploration"),
        continuation="only selected variant resumes, original plateau rule and cap5000, best validation policy delivery",
        final_test="only selected variant best-validation policy, same 50 full-environment scenarios; no selection by test",
        teacher_replay_check=dict(schedule="exact", absolute_per_world_tolerance=0.001,
                                  absolute_mean_tolerance=0.0001),
        plots=False, workers=workers, cache="isolated ga10_warmstart; historical traffic cache is not accessed")
    plan = json.loads(json.dumps(plan))
    plan_path = root / "config/plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("campaign settings/source changed; do not overwrite its provenance")
    daily._write_json(plan_path, plan)
    daily._write_json(root / "config/source_hashes.json", source_files)
    archive = root / "config/source_snapshot.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in daily._source_files():
                bundle.write(path, path.relative_to(daily.ROOT))
    with single_runner(root / "log/runner.lock"):
        if (root / "results/comparison.json").exists():
            print((root / "results/comparison.json").read_text(), flush=True)
            return
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=daily._initialize_worker,
                                     initargs=(problem, daily.ROOT / ".cache/ga10_warmstart", hp, True, True)) as pool:
                def status(stage, **details):
                    daily._write_json(status_path, dict(status=stage, pid=os.getpid(), updated=time.time(), **details))

                def batch(split, indices, weights, epsilon, seeds):
                    if split not in ("train", "validation"):
                        raise ValueError("test set cannot be accessed during training")
                    return list(pool.map(daily._run_world, [(splits[split][i], "response7d", weights, epsilon, seed)
                                                            for i, seed in zip(indices, seeds)]))

                demo_path = root / "results/teacher_trajectories.pt"
                if not demo_path.exists():
                    status("collecting_training_demonstrations")
                    started = time.perf_counter()
                    demos = []
                    for trajectory in pool.map(teacher_rollout, [(s, priority, problem["baseline_flow"]) for s in splits["train"]]):
                        demos.append(trajectory)
                        status("collecting_training_demonstrations", completed=len(demos), total=64)
                    demo_seconds = time.perf_counter() - started
                    # Validation teacher outcomes are reference measurements, never training samples.
                    val_started = time.perf_counter()
                    teacher_val = list(pool.map(teacher_rollout, [(s, priority, problem["baseline_flow"]) for s in splits["validation"]]))
                    payload = dict(train=demos, teacher_validation=teacher_val,
                                   demonstration_seconds=demo_seconds,
                                   teacher_validation_seconds=time.perf_counter() - val_started)
                    save_torch(demo_path, payload)
                payload = torch.load(demo_path, weights_only=False)
                demos = payload["train"]
                if len(demos) != 64 or [t["result"]["scenario"] for t in demos] != [s.scenario_id for s in splits["train"]]:
                    raise AssertionError("training demonstrations do not match frozen train order")
                teacher_indices = sorted(np.random.default_rng(42).choice(64, 10, replace=False).tolist())
                teacher_score = float(np.mean([demos[i]["result"]["objective"] for i in teacher_indices]))
                original_teacher_score = json.loads(teacher_path.read_text())["mean_training_score"][0]
                audit = []
                database = methods / TEACHER / "results/evaluations.sqlite3"
                connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
                try:
                    for i in teacher_indices:
                        result = demos[i]["result"]
                        actual = tuple((e, result["starts"][e], result["completions"][e]) for e in result["order"])
                        if actual != physical_schedule(splits["train"][i], priority):
                            raise AssertionError("teacher replay changed a repair schedule")
                        previous = json.loads(connection.execute("SELECT score FROM worlds WHERE key=?",
                            (schedule_key(splits["train"][i], priority),)).fetchone()[0])[0]
                        difference = result["objective"] - previous
                        audit.append(dict(scenario=result["scenario"], saved=previous,
                                          replayed=result["objective"], difference=difference, schedule_identical=True))
                        if abs(difference) > plan["teacher_replay_check"]["absolute_per_world_tolerance"]:
                            raise AssertionError("teacher replay changed a scenario score materially")
                finally:
                    connection.close()
                daily._write_json(root / "results/teacher_replay_check.json", dict(worlds=audit,
                    mean_difference=teacher_score-original_teacher_score,
                    note="same scientific solver tolerances; small numerical differences across executions are retained"))
                if abs(teacher_score - original_teacher_score) > plan["teacher_replay_check"]["absolute_mean_tolerance"]:
                    raise AssertionError("teacher replay does not reproduce the saved GA training score")
                teacher_validation = float(np.mean([t["result"]["objective"] for t in payload["teacher_validation"]]))
                initial_path = root / "results/initial_weights.pt"
                if not initial_path.exists():
                    status("pretraining", teacher_validation=teacher_validation)
                    def evaluate(weights):
                        trajectories = batch("validation", list(range(11)), weights, 0.0, [0] * 11)
                        return np.mean([t["result"]["objective"] for t in trajectories])
                    def probe(item, weights):
                        save_torch(root / "results" / f"imitation_{item['updates']}.pt", weights)
                        daily._write_json(root / "log" / f"imitation_{item['updates']}.json", item)
                        status("pretraining", latest_probe=item, teacher_validation=teacher_validation)
                        print("imitation validation", item, flush=True)
                    weights, pretraining = pretrain(encoder, demos, hp, seed=42, evaluate=evaluate, record_probe=probe)
                    daily._write_json(root / "results/pretraining_summary.json", pretraining)
                    save_torch(initial_path, weights)
                pretraining = json.loads((root / "results/pretraining_summary.json").read_text())
                initialization = dict(weights=torch.load(initial_path, weights_only=True),
                    demonstrations=dict(enumerate(demos)), metadata=dict(
                        teacher=TEACHER, weights_sha256=digest(initial_path), demonstrations_sha256=digest(demo_path),
                        training_scenarios=64, seeded_replay=True, seeded_exemplars=True,
                        pretraining_updates=pretraining["selected_updates"]))
                screening_path = root / "results/screening.json"
                if not screening_path.exists():
                    screening = {}
                    for variant, changes in plan["online_variants"].items():
                        destination = methods / f"rl_s2v_ga10_init_{variant}_seed42"
                        status("screening", variant=variant, completed_variants=screening)
                        policy, outcome = train(encoder, batch, train_count=64, validation_count=11,
                            seed=42, run_dir=destination,
                            identity=dict(experiment=identity, training_behavior="response7d", input_variant="roadclass",
                                          code_sha256=source_hash, campaign=CAMPAIGN, variant=variant),
                            hp=dict(hp, **changes), stop_params=full_contract["stop"], ep_cap=5000,
                            initialization=initialization, pause_after=76)
                        screening[variant] = dict(mean_last_three_validation=screen_score(destination), **outcome)
                        daily._write_json(destination / "log/status.json", screening[variant])
                    selected = min(screening, key=lambda name: screening[name]["mean_last_three_validation"])
                    daily._write_json(screening_path, dict(selected=selected, variants=screening,
                        decision_basis="validation only; no new test evaluation has occurred"))
                screening = json.loads(screening_path.read_text())
                selected = screening["selected"]
                destination = methods / f"rl_s2v_ga10_init_{selected}_seed42"
                status("training_selected_variant", selected=selected, screening=screening)
                policy, outcome = train(encoder, batch, train_count=64, validation_count=11,
                    seed=42, run_dir=destination,
                    identity=dict(experiment=identity, training_behavior="response7d", input_variant="roadclass",
                                  code_sha256=source_hash, campaign=CAMPAIGN, variant=selected),
                    hp=dict(hp, **plan["online_variants"][selected]), stop_params=full_contract["stop"],
                    ep_cap=5000, initialization=initialization)
                daily._write_json(destination / "log/status.json", dict(status="testing", **outcome))
                final_trace = pd.read_csv(destination / "log/training.csv").dropna(subset=["validation_objective"])
                status("testing_selected_final_policy", selected=selected, **outcome)
                started = time.perf_counter()
                tests = list(pool.map(daily._run_world, [(s, "response7d", policy.net.state_dict(), 0.0, 0) for s in splits["test"]]))
                evaluation = daily._save_evaluation(destination, tests, time.perf_counter() - started)
                daily._write_json(destination / "log/status.json", dict(status="complete", **outcome, **evaluation))
                current = pd.read_csv(destination / "results/test_results.csv")
                report = dict(selected_variant=selected, training=outcome, testing=evaluation,
                    final_validation_objective=float(final_trace.iloc[-1].validation_objective),
                    pretraining=pretraining, teacher_validation_objective=teacher_validation,
                    screening=screening, demonstration_seconds=payload["demonstration_seconds"],
                    teacher_validation_seconds=payload["teacher_validation_seconds"],
                    additional_screening_seconds=sum(v["training_seconds"] for k, v in screening["variants"].items() if k != selected),
                    versus_full_rl=paired_report(current, pd.read_csv(methods / FULL / "results/test_results.csv")),
                    versus_teacher=paired_report(current, pd.read_csv(methods / TEACHER / "results/test_results.csv")),
                    limitations="one seed; validation used for three imitation checkpoints and two early online variants; final RL policy tested once")
                daily._write_json(root / "results/comparison.json", report)
                status("complete", selected=selected, **evaluation)
                print(json.dumps(report, indent=2), flush=True)
        except BaseException as error:
            daily._write_json(status_path, dict(status="failed", pid=os.getpid(), error_type=type(error).__name__,
                                               message=str(error), updated=time.time()))
            raise
        finally:
            simulator.close()


if __name__ == "__main__":
    run()
