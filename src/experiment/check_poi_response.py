"""Numerical and isolation checks before the POI-response GA/RL comparison.

The saved flow-priority trajectory is an implementation reference, not a model
selection criterion. No learned policy is scored or selected by this check.
"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment.daily import DailyEpisode, DailyTraffic
from src.environment.evaluate import build_damaged_edges
from src.environment.poi_response import sensitivity
from src.environment.shifted_demand import solve_shifted
from src.experiment.daily import prepare, save_problem, _write_json, _source_digest, ROOT
from src.experiment.daily_ga_saa10 import training_subset
from src.methods.daily_s2v import DailyQ
from src.methods.rl_rank import EP_CAP, STOP_PARAMS
from src.methods.metaheuristic import GA_PARAMS, BUDGET_CAP
from src.problem.daily import build_problem, make_splits, public_spec


def flow_rollout(simulator, scenario):
    episode = DailyEpisode(simulator, scenario, "response7d", disconnected_flow="unserved")
    states = [episode.traffic]
    while not episode.finished:
        obs = episode.observation()
        selected = min(obs.candidates, key=lambda e: (-simulator.problem["baseline_flow"][e], e))
        episode.step(selected)
        states.append(episode.traffic)
    return episode.result(), states


def audit_and_freeze():
    started = time.perf_counter()
    problem, splits, sim, encoder, folder, identity = prepare(problem_variant="poi_unmet")
    try:
        baseline_problem = build_problem(11, 42)
        baseline_splits = make_splits(baseline_problem)
        spec = public_spec(problem)
        del spec["human_response"]
        assert spec == public_spec(baseline_problem)
        assert all([s.record() for s in splits[k]] == [s.record() for s in baseline_splits[k]]
                   for k in splits)
        old = ROOT / "outputs/experiments/n11_discovery_response7d_1day_lhs50_s42_d35c7028"
        for split in splits:
            assert [s.record() for s in splits[split]] == json.loads(
                (old / f"data/scenarios/{split}.json").read_text(encoding="utf-8"))
        old_rl = json.loads((old / "data/methods/rl_s2v_saa64_adaptive_train_response7d_roadclass_seed42/config/training.json").read_text(encoding="utf-8"))
        old_ga = json.loads((old / "data/methods/ga_saa10_flow_insert_seed42/config/training.json").read_text(encoding="utf-8"))
        assert json.loads(json.dumps(encoder.specification())) == old_rl["feature_schema"]
        assert json.loads(json.dumps(DailyQ(encoder).hp)) == old_rl["hyperparameters"]
        assert STOP_PARAMS == old_rl["stop"] and EP_CAP == old_rl["ep_cap"]
        indices, _ = training_subset(splits["train"], 42)
        assert indices == old_ga["selected_training_indices"]
        assert GA_PARAMS == old_ga["hyperparameters"] and BUDGET_CAP == old_ga["budget"]
        counts = pd.read_csv(ROOT / ".codex-build/gamma_latest_daily/destination_poi_counts.csv")
        np.testing.assert_array_equal(counts.iloc[:, 1:], problem["human_response"]["counts"])
        # The scalar and constant-vector branches must solve the same equations.
        forecast, _, reference = sim.reference(1)
        damaged = {r.edge_id: r.true_severity for r in splits["train"][0].roads}
        edges = build_damaged_edges(sim.ctx, damaged)
        args = ((1-sim.weight)*forecast, sim.weight*forecast)
        kwargs = dict(tolerance=sim.rules.fixed_point_tolerance, ue_tolerance=sim.rules.response_ue_gap)
        scalar = solve_shifted(sim.ctx, edges, forecast, reference, sim.gamma, *args, **kwargs)
        vector = solve_shifted(sim.ctx, edges, forecast, reference, np.full_like(forecast, sim.gamma), *args, **kwargs)
        np.testing.assert_allclose(scalar[0], vector[0], rtol=0, atol=1e-10)
        # Confirm the approved rate law, correct date of history, and two distinct
        # histories for two OD pairs sharing the same destination.
        zero = np.zeros_like(forecast)
        initial = sensitivity(sim.gamma, zero, sim.od_function_weights, sim.response_spec)
        np.testing.assert_allclose(initial, sim.gamma, rtol=1e-15)
        large = sensitivity(sim.gamma, zero+1e5, sim.od_function_weights, sim.response_spec)
        np.testing.assert_allclose(large, .5*sim.gamma, rtol=1e-15)
        first = sim.solve(1, damaged, "response7d")
        second = sim.solve(2, damaged, "response7d", first)
        expected_unmet = np.maximum(0., 1-first.served_demand/sim.ctx["H0"])
        np.testing.assert_array_equal(first.cumulative_unmet, expected_unmet)
        np.testing.assert_array_equal(second.gamma_od, sensitivity(
            sim.gamma, expected_unmet, sim.od_function_weights, sim.response_spec))
        packed = sim._unpack(first.key, first.day, sim._pack(first))
        np.testing.assert_array_equal(packed.cumulative_unmet, first.cumulative_unmet)
        np.testing.assert_array_equal(packed.gamma_od, first.gamma_od)
        # Reproduce the saved sandbox schedule and cumulative loss within the
        # existing numerical tolerances. Never relax a solver tolerance here.
        reference_meta = json.loads((ROOT / ".codex-build/gamma_latest_daily/run_meta.json").read_text())
        assert splits["test"][0].record() == reference_meta["scenario"]
        result, states = flow_rollout(sim, splits["test"][0])
        prior = reference_meta["result"]
        assert result["order"] == prior["order"]
        assert result["recovery_day"] == prior["recovery_day"]
        assert result["repair_completion_day"] == prior["repair_completion_day"]
        assert abs(result["objective"]-prior["objective"]) < 1e-4
        arrays = np.load(ROOT / ".codex-build/gamma_latest_daily/od_gamma_data.npz")
        for state in states:
            i = state.day-1
            np.testing.assert_allclose(state.gamma_od, arrays["gamma_od"][i], atol=1e-7, rtol=0)
            assert np.max(abs(state.served_demand-arrays["adjusted_od"][i])/sim.ctx["H0"]) < 1e-4
        # Reusing the same worker for another world must not carry any history
        # from a preceding episode. Disk hits must keep history and warm state.
        with tempfile.TemporaryDirectory() as temporary:
            disk = DailyTraffic(problem, temporary)
            try:
                train_result, _ = flow_rollout(disk, splits["train"][0])
                other_result, _ = flow_rollout(disk, splits["train"][1])
                repeat, _ = flow_rollout(disk, splits["train"][0])
                assert repeat["objective"] == train_result["objective"]
                disk.memo.clear()
                from_disk, _ = flow_rollout(disk, splits["train"][0])
                assert from_disk["objective"] == train_result["objective"]
                assert disk.cache_hits > 0
                day1 = disk.solve(1, damaged, "response7d")
                np.testing.assert_allclose(day1.gamma_od, sim.gamma, atol=1e-15, rtol=0)
                disk.cache_limit_bytes = 1
                rows_before = disk.connection.execute("SELECT COUNT(*) FROM traffic").fetchone()[0]
                # A capped cache remains readable and missing states are solved.
                third_damage = {r.edge_id: r.true_severity for r in splits["train"][2].roads}
                third = disk.solve(1, third_damage, "response7d")
                assert np.isfinite(third.demand).all()
                assert disk.connection.execute("SELECT COUNT(*) FROM traffic").fetchone()[0] == rows_before
            finally:
                disk.close()
        save_problem(folder, problem, splits, identity, encoder)
        audit = dict(experiment=str(folder), baseline=str(old),
                     changed="only OD-specific sensitivity, from yesterday's unmet demand and frozen POI weights",
                     unchanged="all scenario truths, repair rules, loss, numerical tolerances, RL inputs and hyperparameters, GA operators and training subset",
                     counts_match_behavior_study=True, scenarios_identical=True,
                     scalar_vector_equivalent=True, history_and_disk_cache_checks_passed=True,
                     public_roads=len(problem["public_roads"]), hidden_roads=sum(problem["hidden_quotas"]),
                     ga_training_indices=indices, source_sha256=_source_digest()[0],
                     disk_cache_limit_bytes=sim.cache_limit_bytes,
                     sandbox_reproduction=dict(objective=result["objective"], prior=prior["objective"],
                         recovery_day=result["recovery_day"], repair_completion_day=result["repair_completion_day"]),
                     training_smoke_objectives=[train_result["objective"], other_result["objective"]],
                     check_seconds=time.perf_counter()-started,
                     figure_contract=dict(backend="python", format="PNG", archetype="quantitative comparison",
                         purpose="compare the test distributions without presuming a winner", methods=2,
                         box="quartiles and median", whiskers="1.5 IQR", points="50 paired test worlds",
                         annotations=["mean", "median"], colors=["#418C8B", "#8265A3"],
                         source_data="saved per-scenario test results", training_replicates=1))
        _write_json(folder / "data/analysis/config/poi_response_audit.json", audit)
        return audit
    finally:
        sim.close()


if __name__ == "__main__":
    print(json.dumps(audit_and_freeze(), indent=2))
