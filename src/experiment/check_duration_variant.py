"""Audit and freeze the isolated n11 75%-public, narrow-duration experiment.

Run before training: python -m src.experiment.check_duration_variant
All other random labels are checked against the existing 75%-public experiment.
This performs no policy optimization or test-set policy selection.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import time

import numpy as np
from scipy.stats import norm

from src import config as P
from src.experiment.daily import ROOT, prepare, save_problem, _write_json, _source_digest
from src.experiment.daily_ga_saa10 import training_subset
from src.environment.daily import DailyEpisode
from src.methods.daily_s2v import DailyQ, rollout
from src.methods.metaheuristic import GA_PARAMS, BUDGET_CAP
from src.methods.rl_rank import EP_CAP, STOP_PARAMS
from src.problem.daily import (NARROW_DURATION_CV, duration_draw, duration_expectation,
                               estimated_duration, public_spec)


VARIANT = "public75_duration25"
BASELINE = ROOT / "outputs/experiments/n11_discovery_response7d_1day_lhs50_s42_368acec5"
CURRENT = ROOT / "outputs/experiments/n11_discovery_response7d_1day_lhs50_s42_d35c7028"


def audit_and_freeze():
    # Protect the already completed default experiment from an accidental change
    # in sampling, traffic identity or output path caused by adding the variant.
    current = prepare(11)
    try:
        current_manifest = json.loads((CURRENT / "data/manifest.json").read_text(encoding="utf-8"))
        if current[4] != CURRENT or json.loads(json.dumps(current[5])) != current_manifest["identity"]:
            raise AssertionError("default 60/40 experiment changed")
    finally:
        current[2].close()

    problem, splits, simulator, encoder, folder, identity = prepare(11, problem_variant=VARIANT)
    try:
        previous_manifest = json.loads((BASELINE / "data/manifest.json").read_text(encoding="utf-8"))
        old = previous_manifest["identity"]["problem"]
        new = json.loads(json.dumps(public_spec(problem)))
        allowed = {"rules", "duration_sd", "estimated_duration_cell_expectation_days"}
        if {k: v for k, v in old.items() if k not in allowed} != {
                k: v for k, v in new.items() if k not in allowed}:
            raise AssertionError("public problem differs beyond the duration distribution")
        if {k: v for k, v in old["rules"].items() if k != "version"} != {
                k: v for k, v in new["rules"].items() if k != "version"}:
            raise AssertionError("another daily rule changed")
        for key in ("evaluation_behavior", "disconnected_flow", "objective"):
            if previous_manifest["identity"][key] != identity[key]:
                raise AssertionError(f"scoring contract changed: {key}")

        changed = {}
        for name, scenarios in splits.items():
            old_records = json.loads((BASELINE / f"data/scenarios/{name}.json").read_text(encoding="utf-8"))
            if len(old_records) != len(scenarios):
                raise AssertionError(f"sample size differs: {name}")
            differences = 0
            for before, after in zip(old_records, scenarios):
                if before["scenario_id"] != after.scenario_id or len(before["roads"]) != len(after.roads):
                    raise AssertionError("scenario identities differ")
                for a, road in zip(before["roads"], after.roads):
                    b = asdict(road)
                    if {k: v for k, v in a.items() if k != "true_duration"} != {
                            k: v for k, v in b.items() if k != "true_duration"}:
                        raise AssertionError("road identity, severity, estimate or discovery changed")
                    differences += a["true_duration"] != b["true_duration"]
            changed[name] = dict(scenarios=len(scenarios), changed_true_durations=differences)
            if not differences:
                raise AssertionError("narrow-duration variant did not change durations")

        cells = []
        sd_table = problem["duration_sd"]
        for cell, mean in P.DUR_MEAN.items():
            sd = sd_table[cell]
            variance = np.log1p((sd / mean) ** 2)
            sigma, mu, upper = np.sqrt(variance), np.log(mean) - variance / 2, mean * P.DUR_TRUNC_MULT
            days = np.arange(1, int(np.floor(upper + .5)) + 1)
            cdf = norm.cdf((np.log(np.minimum(days + .5, upper)) - mu) / sigma)
            cdf /= norm.cdf((np.log(upper) - mu) / sigma)
            pmf = np.diff(np.r_[0., cdf])
            coverage = float(pmf[(days >= .75 * mean) & (days <= 1.25 * mean)].sum())
            integer_mean = float(days @ pmf)
            if coverage < .95 or sd >= P.DUR_SD[cell]:
                raise AssertionError(f"duration coverage/spread failed: {cell}")
            if abs(integer_mean - duration_expectation(*cell, sd_by_cell=sd_table)) > 1e-12:
                raise AssertionError("public estimate uses the wrong distribution")
            for day, low, high in zip(days, np.r_[0., cdf[:-1]], cdf):
                if high - low > 1e-9 and duration_draw(*cell, float((low + high) / 2),
                                                      sd_by_cell=sd_table) != day:
                    raise AssertionError("inverse-CDF sampling differs from integer probabilities")
            cells.append(dict(road_class=cell[0], severity=cell[1], target_mean=mean,
                              old_sd=P.DUR_SD[cell], new_sd=sd,
                              integer_mean=integer_mean,
                              integer_sd=float(np.sqrt((days - integer_mean) ** 2 @ pmf)),
                              public_estimate=estimated_duration(*cell, sd_by_cell=sd_table),
                              probability_within_mean_plus_minus_25_percent=coverage))
        indices, _ = training_subset(splits["train"], 42)
        old_ga = json.loads((BASELINE / "data/methods/ga_saa10_flow_insert_seed42/config/training.json")
                           .read_text(encoding="utf-8"))
        if indices != old_ga["selected_training_indices"]:
            raise AssertionError("GA training subset changed")
        if GA_PARAMS != old_ga["hyperparameters"] or BUDGET_CAP != old_ga["budget"]:
            raise AssertionError("GA search settings changed")
        if json.loads(json.dumps(encoder.specification())) != previous_manifest["feature_schema"]:
            raise AssertionError("RL input specification changed")
        old_rl = json.loads((BASELINE / "data/methods/rl_s2v_saa64_adaptive_train_response7d_roadclass_seed42/config/training.json")
                           .read_text(encoding="utf-8"))
        policy = DailyQ(encoder)
        if (json.loads(json.dumps(policy.hp)) != old_rl["hyperparameters"] or
                STOP_PARAMS != old_rl["stop"] or EP_CAP != old_rl["ep_cap"]):
            raise AssertionError("RL learning/stopping settings changed")
        started = time.perf_counter()
        trajectory = rollout(DailyEpisode(simulator, splits["train"][0], "response7d",
                                           disconnected_flow="unserved"), encoder, policy)
        if not np.isfinite(trajectory["result"]["objective"]):
            raise AssertionError("new problem failed the full-environment training-world smoke check")
        smoke = dict(scenario=splits["train"][0].scenario_id,
                     decisions=len(trajectory["picks"]),
                     objective=trajectory["result"]["objective"],
                     seconds=time.perf_counter() - started)
        save_problem(folder, problem, splits, identity, encoder)
        audit = dict(variant=VARIANT, experiment=str(folder), baseline=str(BASELINE),
                     approved_probability=.95, fractional_half_width=.25,
                     public_roads=8, hidden_roads=3, duration_cv=NARROW_DURATION_CV,
                     probability_definition="per road-class/true-severity cell, after truncation and integer rounding",
                     unchanged_labels="road identities, estimated/true severity, estimated duration, discovery day",
                     default_experiment_identity_unchanged=True,
                     learning_and_search_settings_unchanged=True, training_world_smoke=smoke,
                     samples=changed, ga_training_indices=indices, duration_cells=cells,
                     source_sha256=_source_digest()[0])
        _write_json(folder / "data/analysis/config/duration_variant_audit.json", audit)
        return audit
    finally:
        simulator.close()


if __name__ == "__main__":
    print(json.dumps(audit_and_freeze(), indent=2))
