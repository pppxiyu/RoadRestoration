"""Sequential origin/destination-encoding comparisons at 7/4 and 5/6 discovery.

Reuse the saved GA only in the original setting. Train from scratch in each
setting; the second setting then runs its own unchanged ten-scenario GA.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from src.experiment.daily import (prepare, save_problem, _write_json, _source_digest,
                                  _source_files, _save_evaluation)
from src.experiment.daily_ga import single_runner
from src.experiment.daily_ga_saa10 import compare, run as run_ga
from src.experiment.daily_junction import (initialize_worker, run_world,
                                          verify_saved_inputs, METHOD as OLD_JUNCTION,
                                          MEMORY_METHOD)
from src.experiment.daily_od_history import baseline_hashes, check_contract, BASE_RL, GA
from src.experiment.layout import ROOT, publish_experiment_gallery
from src.methods.daily_junction import JunctionEncoder
from src.methods.daily_s2v import DailyQ, train, POLICY_DELIVERY
from src.problem.daily import public_spec


METHOD = "rl_s2v_dual_od16_memory_seed42"
STEM = "ga_saa10_vs_dual_junction_full_rl"


def evidence_hashes(folder):
    hashes = baseline_hashes(folder)
    for method in (OLD_JUNCTION, MEMORY_METHOD):
        root = folder / "data/methods" / method
        for relative in ("config/training.json", "results/model_final.pt",
                         "results/test_results.csv", "results/training_summary.json",
                         "results/evaluation_summary.json", "log/status.json"):
            path = root / relative
            hashes[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_discovery_only(first, second):
    a, b = json.loads(json.dumps(public_spec(first))), json.loads(json.dumps(public_spec(second)))
    assert len(a.pop("public_roads")) == 7 and len(b.pop("public_roads")) == 5
    assert a.pop("hidden_quotas") == [1, 3] and b.pop("hidden_quotas") == [1, 5]
    a["rules"].pop("version")
    b["rules"].pop("version")
    if a != b:
        raise ValueError("second experiment differs beyond public/hidden quotas and version")


def freeze(path, value):
    normalized = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != normalized:
            raise ValueError(f"frozen run contract changed: {path}")
    else:
        _write_json(path, value)


def train_one(prepared, old, source, source_hashes):
    problem, splits, simulator, plain, folder, identity = prepared
    encoder = JunctionEncoder(problem, simulator.ctx, simulator.gamma,
                              demand_memory=True, bidirectional=True)
    simulator.close()
    save_problem(folder, problem, splits, identity, plain)
    root = folder / "data/methods" / METHOD
    for part in ("config", "log", "results"):
        (root / part).mkdir(parents=True, exist_ok=True)
    status = root / "log/status.json"
    plan = dict(source_sha256=source, experiment_identity=identity, method=METHOD,
        feature_schema=encoder.specification(), hyperparameters=old["hyperparameters"],
        stop=old["stop"], ep_cap=old["ep_cap"], seed=42, workers=4,
        training_scenarios=64, validation_scenarios=11, test_scenarios=50,
        initialization="random from scratch; no GA demonstrations",
        delivery=POLICY_DELIVERY, test_selection=False,
        change="both outgoing and incoming fixed OD slots; concatenate role embeddings then fuse; no traffic-conditioned OD encoder",
        figure=dict(backend="Python/matplotlib", format="PNG", panels=1,
            purpose="compare same-setting saved GA and new RL without presuming improvement",
            evidence="50 paired test scenarios; boxplots and all individual points",
            box="quartiles", whiskers="1.5 IQR", annotations="requested mean and median",
            title=None))
    with single_runner(root / "log/runner.lock"):
        freeze(folder / "data/analysis/config/dual_junction_plan.json", plan)
        freeze(root / "config/source_hashes.json", source_hashes)
        archive = root / "config/source_snapshot.zip"
        if not archive.exists():
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                for path in _source_files():
                    bundle.write(path, path.relative_to(ROOT))
        complete = status.exists() and json.loads(status.read_text())["status"] == "complete"
        if not complete:
            _write_json(status, dict(status="training", pid=os.getpid()))
            from src.analysis.viz.daily_viz import make_training_figures
            with ProcessPoolExecutor(max_workers=4, initializer=initialize_worker,
                    initargs=(problem, ROOT / ".cache/daily_traffic", old["hyperparameters"], True, True)) as pool:
                def batch(split, indices, weights, epsilon, seeds):
                    if split not in ("train", "validation"):
                        raise ValueError("training may not request test data")
                    return list(pool.map(run_world, [(splits[split][i], weights, epsilon, seed)
                                                    for i, seed in zip(indices, seeds)]))
                policy, training = train(encoder, batch, train_count=64, validation_count=11,
                    seed=42, run_dir=root, hp=old["hyperparameters"], stop_params=old["stop"],
                    ep_cap=old["ep_cap"], redraw=make_training_figures,
                    identity=dict(experiment=identity, training_behavior="response7d",
                                  input_variant="dual_junction_od16_demand_memory", code_sha256=source))
                _write_json(status, dict(status="testing", pid=os.getpid(), **training))
                started = time.perf_counter()
                trajectories = list(pool.map(run_world, [(s, policy.net.state_dict(), 0., 0)
                                                         for s in splits["test"]]))
                seconds = time.perf_counter() - started
                evaluation = _save_evaluation(root, trajectories, seconds)
                np.savez_compressed(root / "results/test_junction_inputs.npz",
                    **{f"scenario_{i}_od": np.stack([s["od"] for s in t["states"]])
                       for i, t in enumerate(trajectories)})
                verify_saved_inputs(folder, root, encoder)
                _write_json(status, dict(status="complete", **training, **evaluation))
        current = json.loads((root / "config/training.json").read_text())
        for key in ("hyperparameters", "stop", "ep_cap", "seed", "reward", "discount"):
            if current[key] != old[key]:
                raise ValueError(f"learning contract changed: {key}")
        if current["delivery"] != POLICY_DELIVERY:
            raise ValueError("this entry requires validation-selected test policy; preserve historical final-policy results")
        if current["identity"]["experiment"] != json.loads(json.dumps(identity)):
            raise ValueError("saved RL experiment differs")
        verify_saved_inputs(folder, root, encoder)
    return encoder, root


def finish(folder, encoder, root):
    result = compare(folder, folder / "data/methods" / GA, root,
                     output_stem=STEM, rl_label="Full RL\nwith origin/destination encoding")
    history = pd.read_csv(root / "log/training.csv")
    result.update(saved_inputs_verified=True,
        training=json.loads((root / "results/training_summary.json").read_text()),
        evaluation=json.loads((root / "results/evaluation_summary.json").read_text()),
        ga_training=json.loads((folder / "data/methods" / GA / "results/training_summary.json").read_text()),
        ga_evaluation=json.loads((folder / "data/methods" / GA / "results/evaluation_summary.json").read_text()),
        final_validation_objective=float(history.validation_objective.dropna().iloc[-1]),
        best_validation_objective=float(history.best_validation_objective.iloc[-1]),
        parameter_count=sum(p.numel() for p in DailyQ(encoder).net.parameters()),
        limitation="one training/search seed; differences between discovery settings are not paired identical-truth causal effects")
    _write_json(folder / "data/analysis/results/dual_junction_comparison.json", result)
    publish_experiment_gallery(directory=folder)
    print(json.dumps(dict(stage_complete=str(folder), **result), indent=2), flush=True)
    return result


def main():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    first = prepare(problem_variant="poi_unmet")
    problem, splits, simulator, plain, folder, identity = first
    old = check_contract(folder, plain, identity)
    source, source_hashes = _source_digest()
    plan_path = folder / "data/analysis/config/dual_junction_sequence.json"
    hashes = evidence_hashes(folder)
    plan = dict(source_sha256=source, baseline_hashes=hashes,
        method=METHOD, seed=42, workers=4,
        stages=[dict(variant="poi_unmet", public=7, hidden=4, reuse_ga=True),
                dict(variant="poi_hidden6", public=5, hidden=6, reuse_ga=False)],
        execution="finish and verify first comparison before starting second RL, then second GA")
    overall = folder / "data/log/dual_junction_sequence_status.json"
    with single_runner(folder / "data/log/dual_junction_sequence.lock"):
        freeze(plan_path, plan)
        try:
            _write_json(overall, dict(status="running", stage="public7_hidden4_rl", pid=os.getpid()))
            encoder, root = train_one(first, old, source, source_hashes)
            if evidence_hashes(folder) != hashes:
                raise ValueError("baseline evidence changed")
            first_result = finish(folder, encoder, root)
            if _source_digest()[0] != source:
                raise ValueError("source changed during the sequence")
            second = prepare(problem_variant="poi_hidden6")
            validate_discovery_only(problem, second[0])
            _write_json(overall, dict(status="running", stage="public5_hidden6_rl", pid=os.getpid(),
                                     first_result=first_result, second_experiment=str(second[4])))
            encoder, root = train_one(second, old, source, source_hashes)
            _write_json(overall, dict(status="running", stage="public5_hidden6_ga", pid=os.getpid(),
                                     first_result=first_result, second_experiment=str(second[4])))
            if _source_digest()[0] != source:
                raise ValueError("source changed before GA search")
            run_ga(problem_variant="poi_hidden6", full_rl_method=METHOD)
            second_result = finish(second[4], encoder, root)
            if evidence_hashes(folder) != hashes:
                raise ValueError("baseline evidence changed")
            _write_json(overall, dict(status="complete", baseline_files_unchanged=True,
                first_result=first_result, second_result=second_result, second_experiment=str(second[4])))
        except BaseException as error:
            _write_json(overall, dict(status="interrupted_or_failed", error_type=type(error).__name__,
                                    message=str(error), pid=os.getpid()))
            raise


if __name__ == "__main__":
    main()
