"""Train one new Full RL input variant; reuse completed GA and old RL unchanged.

Entry: python -X utf8 -u -m src.experiment.daily_od_history
Only policy inputs and their input-layer width change. Test data never enter
training or stopping. Both baselines are verified from already saved evidence.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment.daily import prepare, save_problem, run, _write_json, _source_digest
from src.experiment.daily_ga import single_runner
from src.experiment.daily_ga_saa10 import compare as compare_saved_ga
from src.methods.daily_s2v import DailyQ, POLICY_DELIVERY
from src.methods.rl_rank import EP_CAP, STOP_PARAMS


BASE_RL = "rl_s2v_saa64_adaptive_train_response7d_roadclass_seed42"
NEW_RL = "rl_s2v_saa64_adaptive_train_response7d_roadclass_od_history_poi_counts_seed42"
GA = "ga_saa10_flow_insert_seed42"


def baseline_hashes(folder):
    """Freeze the actual baseline evidence, without reading/writing search caches."""
    paths = []
    for name in (BASE_RL, GA):
        root = folder / "data/methods" / name
        status = json.loads((root / "log/status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError(f"baseline is not complete: {name}")
        paths.extend(root / relative for relative in (
            "config/training.json", "results/test_results.csv", "results/training_summary.json",
            "results/evaluation_summary.json", "log/status.json"))
        paths.append(root / "results" / ("solution.json" if name == GA else "model_final.pt"))
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def check_contract(folder, encoder, identity):
    old = json.loads((folder / "data/methods" / BASE_RL / "config/training.json").read_text(encoding="utf-8"))
    if old["identity"]["experiment"] != json.loads(json.dumps(identity)):
        raise ValueError("environment or scenarios changed relative to the completed experiment")
    if (DailyQ(encoder).hp != old["hyperparameters"] or STOP_PARAMS != old["stop"]
            or EP_CAP != old["ep_cap"] or old["seed"] != 42):
        raise ValueError("learning settings changed relative to the original Full RL")
    return old


def summarize(folder, encoder, baseline):
    methods = folder / "data/methods"
    new_dir = methods / NEW_RL
    original = methods / BASE_RL
    current = json.loads((new_dir / "config/training.json").read_text(encoding="utf-8"))
    old = json.loads((original / "config/training.json").read_text(encoding="utf-8"))
    for key in ("hyperparameters", "stop", "ep_cap", "seed", "reward", "discount"):
        if current[key] != old[key]:
            raise ValueError(f"unapproved algorithm difference: {key}")
    if current["delivery"] != POLICY_DELIVERY:
        raise ValueError("preserve historical final-policy results; new evaluations require validation selection")
    if baseline_hashes(folder) != baseline:
        raise ValueError("baseline results changed during this input-only experiment")
    expected = [s["scenario_id"] for s in json.loads((folder / "data/scenarios/test.json").read_text())]
    frames = [pd.read_csv(methods / name / "results/test_results.csv").set_index("scenario")
              for name in (BASE_RL, NEW_RL)]
    for frame in frames:
        if len(frame) != 50 or frame.index.has_duplicates or set(frame.index) != set(expected):
            raise ValueError("RL test scenarios differ")
    values = [frame.loc[expected, "F"].to_numpy() for frame in frames]
    delta = values[0] - values[1]
    draws = np.random.default_rng(42).choice(delta, (5000, 50), replace=True).mean(axis=1)
    with np.load(new_dir / "results/test_policy_inputs.npz") as inputs:
        for i in range(50):
            g = inputs[f"scenario_{i}_g"]
            if g.shape[1] != encoder.global_dimension or not np.isfinite(g).all():
                raise ValueError("incorrect saved test input shape or values")
            blocks = g[:, 6:].reshape(len(g), len(encoder.od_pairs), 6)
            np.testing.assert_allclose(blocks[:, :, 1:] * encoder.poi_count_scale,
                np.broadcast_to(encoder.od_poi_counts, blocks[:, :, 1:].shape), rtol=1e-6, atol=1e-5)
            np.testing.assert_array_equal(blocks[0, :, 0], 0)
            if (np.diff(blocks[:, :, 0], axis=0) < -1e-6).any():
                raise ValueError("cumulative unmet demand cannot decrease")
    ga_pair = compare_saved_ga(folder, methods / GA, new_dir,
        output_stem="ga_saa10_vs_od_history_full_rl",
        rl_label="Full RL with OD history\nand destination facility counts")
    result = dict(original_rl_mean=float(values[0].mean()), new_rl_mean=float(values[1].mean()),
        original_rl_median=float(np.median(values[0])), new_rl_median=float(np.median(values[1])),
        paired_original_minus_new_mean=float(delta.mean()),
        relative_improvement_over_original=float(delta.mean() / values[0].mean()),
        new_rl_wins=int((delta > 1e-9).sum()), ties=int((abs(delta) <= 1e-9).sum()),
        bootstrap_mean_95_interval=np.quantile(draws, [.025, .975]).tolist(), ga_comparison=ga_pair,
        training=json.loads((new_dir / "results/training_summary.json").read_text()),
        evaluation=json.loads((new_dir / "results/evaluation_summary.json").read_text()),
        baseline_files_unchanged=True, saved_inputs_verified=True,
        limitation="one training seed; both input additions and input-layer width change together")
    _write_json(folder / "data/analysis/results/od_history_rl_comparison.json", result)
    pd.DataFrame(dict(scenario=expected, original_full_rl=values[0],
                     od_history_full_rl=values[1], original_minus_new=delta)).to_csv(
        folder / "data/analysis/results/od_history_rl_paired.csv", index=False)
    return result


from src.experiment.retired import retired_problem


@retired_problem
def main():
    problem, splits, simulator, encoder, folder, identity = prepare(
        problem_variant="poi_unmet", od_history_input=True)
    simulator.close()
    save_problem(folder, problem, splits, identity, encoder)
    check_contract(folder, encoder, identity)
    plan_path = folder / "data/analysis/config/od_history_input_plan.json"
    plan = dict(source_sha256=_source_digest()[0], experiment_identity=identity,
        method=NEW_RL, baseline_methods=[BASE_RL, GA], baseline_hashes=baseline_hashes(folder),
        features=encoder.specification(), training_scenarios=64, validation_scenarios=11, test_scenarios=50,
        training_seed=42, workers=4, initialization="random; no GA demonstrations",
        changes="append each OD's past unmet demand and destination facility counts to global input",
        unchanged="road inputs, S2V propagation/readout rules, optimizer, hyperparameters, stop, all environment/scenario rules",
        delivery=POLICY_DELIVERY,
        ga_recomputed=False, test_selection=False,
        figure_contract=dict(backend="python", format="PNG", purpose="compare new input variant with saved GA without presuming improvement",
            archetype="quantitative comparison", points="same 50 test scenarios", center="median", box="quartiles",
            whiskers="1.5 IQR", annotations="previously requested mean and median", title=None))
    normalized = json.loads(json.dumps(plan))
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != normalized:
        raise ValueError("input experiment plan/source/baseline changed; do not overwrite")
    _write_json(plan_path, plan)
    status = folder / "data/log/od_history_rl_status.json"
    with single_runner(folder / "data/log/od_history_rl_runner.lock"):
        try:
            _write_json(status, dict(status="running", method=NEW_RL, pid=os.getpid()))
            result_folder = run(11, "response7d", 42, 4, problem_variant="poi_unmet", od_history_input=True)
            if result_folder != folder:
                raise ValueError("new RL was saved to a different experiment")
            result = summarize(folder, encoder, plan["baseline_hashes"])
            _write_json(status, dict(status="complete", **result))
            print(json.dumps(result, indent=2), flush=True)
        except BaseException as error:
            _write_json(status, dict(status="interrupted_or_failed", error_type=type(error).__name__,
                                    message=str(error), pid=os.getpid()))
            raise


if __name__ == "__main__":
    main()
