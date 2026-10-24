"""Initialize the frozen no-live-traffic S2V policy from GA10 demonstrations.

Run: python -X utf8 -u -m src.experiment.daily_ga_warmstart_no_traffic
The only training change from the no-traffic baseline is the initial network,
training-only replay, and exemplar states. No test set is read before delivery.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd

from src.experiment import daily
from src.experiment.daily_ga import single_runner
from src.experiment.daily_ga_warmstart import digest, paired_report, save_torch
from src.methods.daily_imitation import pretrain
from src.methods.daily_s2v import TRAFFIC_GLOBAL_COLUMNS, TRAFFIC_NODE_COLUMNS, train
from src.environment.daily import DailyEpisode, DailyTraffic


METHOD = "rl_s2v_ga10_init_no_traffic_seed42"
CAMPAIGN = "ga10_initialized_no_traffic_rl_seed42"
BASELINE = "rl_s2v_saa64_adaptive_train_response7d_roadclass_no_traffic_seed42"
TEACHER = "ga_saa10_flow_insert_seed42"
DONOR = "ga10_initialized_full_rl_seed42"
FULL = "rl_s2v_saa64_adaptive_train_response7d_roadclass_seed42"


def mask_traffic(trajectory):
    masked = copy.copy(trajectory)
    masked["states"] = []
    for state in trajectory["states"]:
        recoded = dict(state)
        recoded["x"] = state["x"].copy()
        recoded["g"] = state["g"].copy()
        recoded["x"][:, TRAFFIC_NODE_COLUMNS] = 0
        recoded["g"][list(TRAFFIC_GLOBAL_COLUMNS)] = 0
        masked["states"].append(recoded)
    return masked


def compare_encoded_states(saved, fresh):
    if saved["picks"] != fresh["picks"] or len(saved["states"]) != len(fresh["states"]):
        raise AssertionError("traffic masking changed the teacher's legal actions")
    for old, new in zip(saved["states"], fresh["states"]):
        if old["roads"] != new["roads"] or old["day"] != new["day"]:
            raise AssertionError("traffic masking changed candidate identities or decision day")
        if not np.allclose(old["x"], new["x"], atol=1e-7) or not np.allclose(old["g"], new["g"], atol=1e-7):
            raise AssertionError("masked donor inputs do not equal the actual no-traffic encoder")


from src.experiment.retired import retired_problem


@retired_problem
def run(workers=4):
    import torch
    from src.experiment.daily_ga_warmstart import teacher_rollout

    torch.set_num_threads(1)
    problem, splits, simulator, encoder, folder, identity = daily.prepare(11, 42, traffic_input=False)
    daily.save_problem(folder, problem, splits, identity, encoder)
    if folder.name != "n11_discovery_response7d_1day_lhs50_s42_368acec5":
        raise ValueError("problem identity changed")
    methods = folder / "data/methods"
    root, run_dir = methods / CAMPAIGN, methods / METHOD
    for child in ("config", "results", "log"):
        (root / child).mkdir(parents=True, exist_ok=True)
    status_path = root / "log/status.json"
    base_contract = json.loads((methods / BASELINE / "config/training.json").read_text())
    teacher_contract = json.loads((methods / TEACHER / "config/training.json").read_text())
    if base_contract["identity"]["experiment"] != json.loads(json.dumps(identity)) or \
            teacher_contract["identity"]["experiment"] != json.loads(json.dumps(identity)):
        raise AssertionError("baseline, teacher, and new run do not share the same problem")
    if base_contract["feature_schema"] != json.loads(json.dumps(encoder.specification())):
        raise AssertionError("new policy input schema differs from the existing no-traffic RL")
    if base_contract["identity"]["traffic_input"] is not False or base_contract["ep_cap"] != 5000:
        raise AssertionError("wrong no-traffic baseline")
    hp = base_contract["hyperparameters"]
    if hp["pool_n"] != 64 or len(splits["train"]) != 64 or len(splits["validation"]) != 11 or len(splits["test"]) != 50:
        raise AssertionError("scenario split or RL pool differs")
    donor_path = methods / DONOR / "results/teacher_trajectories.pt"
    if json.loads((methods / DONOR / "log/status.json").read_text())["status"] != "complete":
        raise AssertionError("the donor's teacher trajectories are not from a completed campaign")
    teacher_solution = methods / TEACHER / "results/solution.json"
    source_hash, source_files = daily._source_digest()
    plan = dict(version=1, experiment=identity, source_sha256=source_hash,
                donor_teacher_trajectories_sha256=digest(donor_path),
                teacher_solution_sha256=digest(teacher_solution),
                train_scenarios_sha256=digest(folder / "data/scenarios/train.json"),
                validation_scenarios_sha256=digest(folder / "data/scenarios/validation.json"),
                test_scenarios_sha256=digest(folder / "data/scenarios/test.json"),
                baseline=BASELINE, teacher=TEACHER, final_method=METHOD, seed=42,
                demonstration_rule="reuse frozen GA10 training trajectories; zero road columns 10-13 and global columns 4-5 exactly as no-traffic encoder does",
                verification="compare donor-masked inputs against direct no-traffic encoding on two complete frozen training scenarios",
                pretraining=dict(checkpoints=[200, 600, 1200], criterion="lowest 11-scenario validation mean",
                                  loss="legal-action cross entropy(Q/0.2) + 0.1 smooth-L1(chosen Q, teacher discounted return)"),
                online_training=dict(hyperparameters_unchanged=True, stop_unchanged=True,
                                     eps0=hp["eps0"], delivery="best validation policy", training_worlds=64),
                final_test=dict(worlds=50, behavior="response7d", selected_by="validation only"),
                workers=workers, figures=False)
    plan = json.loads(json.dumps(plan))
    plan_path = root / "config/plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("campaign contract or source changed; do not overwrite this run")
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
        def status(stage, **fields):
            daily._write_json(status_path, dict(status=stage, pid=os.getpid(), updated=time.time(), **fields))
        try:
            # The historical cache has a damaged SQLite file. The prior GA10
            # warm-start cache is isolated and has the same problem/traffic rules.
            cache = daily.ROOT / ".cache/ga10_warmstart"
            with ProcessPoolExecutor(max_workers=workers, initializer=daily._initialize_worker,
                                     initargs=(problem, cache, hp, True, False)) as executor:
                def batch(split, indices, weights, epsilon, seeds):
                    if split not in ("train", "validation"):
                        raise ValueError("training may not access test scenarios")
                    return list(executor.map(daily._run_world, [(splits[split][i], "response7d", weights, epsilon, seed)
                                                                for i, seed in zip(indices, seeds)]))

                demo_path = root / "results/teacher_trajectories_no_traffic.pt"
                if not demo_path.exists():
                    status("preparing_demonstrations")
                    donor = torch.load(donor_path, weights_only=False)
                    if len(donor["train"]) != 64 or [x["result"]["scenario"] for x in donor["train"]] != \
                            [s.scenario_id for s in splits["train"]]:
                        raise AssertionError("donor training scenarios differ")
                    if len(donor["teacher_validation"]) != 11 or \
                            [x["result"]["scenario"] for x in donor["teacher_validation"]] != \
                            [s.scenario_id for s in splits["validation"]]:
                        raise AssertionError("donor validation scenarios differ")
                    demos = [mask_traffic(x) for x in donor["train"]]
                    if any(np.any(s["x"][:, TRAFFIC_NODE_COLUMNS]) or np.any(s["g"][list(TRAFFIC_GLOBAL_COLUMNS)])
                           for t in demos for s in t["states"]):
                        raise AssertionError("traffic information survived masking")
                    priority = tuple(json.loads(teacher_solution.read_text())["order"])
                    start = time.perf_counter()
                    for index in (0, 63):
                        fresh = list(executor.map(teacher_rollout, [(splits["train"][index], priority,
                                                                      problem["baseline_flow"])]))[0]
                        compare_encoded_states(demos[index], fresh)
                    verification_seconds = time.perf_counter() - start
                    save_torch(demo_path, dict(train=demos, verification_seconds=verification_seconds,
                                              teacher_validation_mean=float(np.mean([
                                                  x["result"]["objective"] for x in donor["teacher_validation"]]))))
                payload = torch.load(demo_path, weights_only=False)
                demos = payload["train"]
                if len(demos) != 64 or [x["result"]["scenario"] for x in demos] != \
                        [s.scenario_id for s in splits["train"]]:
                    raise AssertionError("saved no-traffic demonstrations differ")
                init_path = root / "results/initial_weights.pt"
                if not init_path.exists():
                    status("pretraining")
                    def evaluate(weights):
                        trajectories = batch("validation", list(range(11)), weights, 0.0, [0] * 11)
                        return np.mean([x["result"]["objective"] for x in trajectories])
                    def probe(item, weights):
                        save_torch(root / "results" / f"imitation_{item['updates']}.pt", weights)
                        daily._write_json(root / "log" / f"imitation_{item['updates']}.json", item)
                        status("pretraining", latest_probe=item)
                        print("imitation validation", item, flush=True)
                    initial_weights, pretraining = pretrain(encoder, demos, hp, seed=42,
                                                            evaluate=evaluate, record_probe=probe)
                    daily._write_json(root / "results/pretraining_summary.json", pretraining)
                    save_torch(init_path, initial_weights)
                pretraining = json.loads((root / "results/pretraining_summary.json").read_text())
                initialization = dict(weights=torch.load(init_path, weights_only=True),
                    demonstrations=dict(enumerate(demos)), metadata=dict(teacher=TEACHER,
                        weights_sha256=digest(init_path), demonstrations_sha256=digest(demo_path),
                        pretraining_updates=pretraining["selected_updates"],
                        traffic_input=False, seeded_replay=True, seeded_exemplars=True))
                run_identity = dict(experiment=identity, training_behavior="response7d",
                                    input_variant="roadclass_no_traffic", traffic_input=False,
                                    code_sha256=source_hash, campaign=CAMPAIGN)
                status("training", pretraining=pretraining)
                policy, training = train(encoder, batch, train_count=64, validation_count=11,
                    seed=42, run_dir=run_dir, identity=run_identity, hp=hp,
                    stop_params=base_contract["stop"], ep_cap=5000,
                    initialization=initialization)
                status("testing", training=training)
                test_start = time.perf_counter()
                tests = list(executor.map(daily._run_world,
                    [(s, "response7d", policy.net.state_dict(), 0.0, 0) for s in splits["test"]]))
                testing = daily._save_evaluation(run_dir, tests, time.perf_counter() - test_start)
                inputs = np.load(run_dir / "results/test_policy_inputs.npz")
                if any(np.any(inputs[f"scenario_{i}_x"][:, :, TRAFFIC_NODE_COLUMNS]) or
                       np.any(inputs[f"scenario_{i}_g"][:, list(TRAFFIC_GLOBAL_COLUMNS)]) for i in range(50)):
                    raise AssertionError("test policy received a forbidden traffic observation")
                trace = pd.read_csv(run_dir / "log/training.csv").dropna(subset=["validation_objective"])
                frame = pd.read_csv(run_dir / "results/test_results.csv")
                references = {name: pd.read_csv(methods / name / "results/test_results.csv")
                              for name in (BASELINE, TEACHER, FULL)}
                report = dict(training=training, testing=testing,
                    final_validation_objective=float(trace.iloc[-1].validation_objective),
                    pretraining=pretraining, demonstration_verification_seconds=payload["verification_seconds"],
                    versus_no_traffic_rl=paired_report(frame, references[BASELINE]),
                    versus_ga10=paired_report(frame, references[TEACHER]),
                    versus_full_rl=paired_report(frame, references[FULL]),
                    limitation="one seed; three pretraining checkpoints selected only by validation; no test result selected this model")
                daily._write_json(run_dir / "log/status.json", dict(status="complete", **training, **testing))
                daily._write_json(root / "results/comparison.json", report)
                status("complete", **testing)
                print(json.dumps(report, indent=2), flush=True)
        except BaseException as error:
            status("failed", error_type=type(error).__name__, message=str(error))
            raise
        finally:
            simulator.close()


if __name__ == "__main__":
    run()
