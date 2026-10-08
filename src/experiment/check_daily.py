"""Fast contract checks and an optional real-traffic end-to-end check.

Run python -m src.experiment.check_daily; add --traffic for numerical checks.
No training, official results, or historical files are changed by these checks.
"""
from dataclasses import replace
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
import json

import numpy as np

from src.environment.daily import (DailyEpisode, DailyTraffic, TrafficState, Observation,
                                   context_from_problem, daily_losses, recovery_fraction,
                                   _feasible_daily_seed)
from src.methods.daily_s2v import DailyEncoder, DailyQ, train
from src.problem.daily import (DailyRules, RoadDamage, Scenario, build_problem, make_splits,
                               duration_expectation, estimated_duration, public_spec)


class DurationChecks(unittest.TestCase):
    def test_mean_is_of_actual_integer_distribution(self):
        from src import config as P
        from src.problem.scenarios import _cell_pmf
        for cell in P.DUR_MEAN:
            with self.subTest(cell=cell):
                expected = sum(day * probability for day, probability in _cell_pmf(cell))
                self.assertAlmostEqual(duration_expectation(*cell), expected, places=12)
                self.assertEqual(estimated_duration(*cell), max(1, int(np.floor(expected + 0.5))))
                self.assertIsInstance(estimated_duration(*cell), int)
        self.assertAlmostEqual(duration_expectation("major", 1), 3.999326, places=6)

    def test_estimate_rounds_half_up_and_has_one_day_floor(self):
        for value, expected in ((0.1, 1), (2.49, 2), (2.5, 3), (3.5, 4)):
            with patch("src.problem.daily.duration_expectation", return_value=value):
                self.assertEqual(estimated_duration("local", 1), expected)

    def test_estimate_uses_estimated_cell_not_severity_mixture(self):
        from src.problem.daily import CONFUSION
        mixed = sum(p * duration_expectation("local", severity)
                    for severity, p in enumerate(CONFUSION[1], 1))
        self.assertNotEqual(estimated_duration("local", 1), int(np.floor(mixed + 0.5)))

    def test_estimate_recomputes_actual_distribution_when_parameters_change(self):
        from src import config as P
        # With a much longer tail, truncation changes even the rounded mean.
        # This prevents replacing the expectation calculation with DUR_MEAN.
        with patch.dict(P.DUR_SD, {("major", 1): 8.0}):
            self.assertNotEqual(estimated_duration("major", 1), int(P.DUR_MEAN["major", 1]))


class ConstantTraffic:
    """Traffic stub for isolating scheduling and information rules."""
    rules = DailyRules()
    ctx = {"H0": np.array([10.0]), "baseline_u": np.array([1.0])}

    def __init__(self):
        self.observed_damage = []

    def solve(self, day, damaged, behavior, previous=None):
        self.observed_damage.append((day, dict(damaged)))
        q = np.array([5.0 if damaged else 10.0])
        return TrafficState(str(day), day, q, q, np.ones(1), np.ones(1, dtype=bool),
                            np.zeros(38), np.zeros(38), None, {})


class ContractChecks(unittest.TestCase):
    def test_recovery(self):
        curve = recovery_fraction(np.arange(80), DailyRules())
        self.assertTrue(np.all(np.diff(curve) >= 0))
        self.assertEqual(float(curve[35]), 1.0)
        self.assertTrue(np.all(curve[35:] == 1))

    def test_discovery_and_rewards(self):
        scenario = Scenario("schedule_test", (
            RoadDamage(1, 1, 20, 3, 2, 0),
            RoadDamage(2, 2, 1, 2, 1, 5)))
        sim = ConstantTraffic()
        episode = DailyEpisode(sim, scenario, "natural", disconnected_flow="unserved")
        observation = episode.observation()
        self.assertEqual(observation.candidates, (1,))
        self.assertFalse(hasattr(observation.visible[0], "true_severity"))
        self.assertFalse(hasattr(observation.visible[0], "true_duration"))
        self.assertEqual(observation.visible[0].estimated_duration, 20)
        with self.assertRaises(ValueError):
            episode.step(2)
        with self.assertRaises(ValueError):
            episode.step(None)
        next_observation, reward1, done = episode.step(1)
        self.assertFalse(done)
        self.assertEqual(next_observation.day, 5)
        self.assertEqual(next_observation.candidates, (2,))
        self.assertEqual(episode.completed[1], 3)
        self.assertIn(2, sim.observed_damage[0][1])
        _, reward2, done = episode.step(2)
        self.assertTrue(done)
        result = episode.result()
        self.assertEqual(result["recovery_day"], 7)
        self.assertEqual(result["repair_completion_day"], 6)
        self.assertAlmostEqual(reward1 + reward2, -result["objective"])
        self.assertEqual([r["day"] for r in result["daily"]], list(range(1, 8)))

    def test_disconnected_trips(self):
        state = TrafficState("disconnected", 1, np.array([10.0, 10.0]), np.array([10.0, 10.0]),
                             np.array([100.0, 1.0]), np.array([False, True]),
                             np.zeros(38), np.zeros(38), None, {})
        result = daily_losses(state, dict(H0=np.array([10.0, 10.0]), baseline_u=np.ones(2)),
                              disconnected_flow="unserved")
        self.assertEqual(result["flow_loss"], 0.5)
        self.assertEqual(result["time_loss"], 0.0)
        state.reachable[:] = False
        result = daily_losses(state, dict(H0=np.array([10.0, 10.0]), baseline_u=np.ones(2)),
                              disconnected_flow="unserved")
        self.assertEqual(result["flow_loss"], 1.0)
        self.assertEqual(result["time_loss"], 0.0)


class PolicyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = build_problem()
        cls.context = context_from_problem(cls.problem)
        cls.encoder = DailyEncoder(cls.problem, cls.context)
        cls.scenario = make_splits(cls.problem)["train"][0]

    def observation(self, day=1):
        c = self.context
        visible = self.scenario.visible(1)
        return Observation(day, visible, (), tuple(r.edge_id for r in visible),
                           c["H0"].copy() * 0.5, c["H0"].copy() * 0.75, c["baseline_u"].copy(),
                           np.ones(len(c["H0"]), dtype=bool),
                           np.zeros(len(self.problem["edges"])), np.zeros(len(self.problem["edges"])))

    def test_public_features(self):
        observation = self.observation(70)
        state = self.encoder.encode(observation)
        self.assertEqual(state["x"].shape, (38, 14))
        self.assertEqual(state["g"].shape, (6,))
        self.assertEqual(state["g"][0], 2.0)  # no hidden horizon or clipping
        self.assertAlmostEqual(float(state["g"][2]), 0.25)
        self.assertAlmostEqual(float(state["g"][4]), 0.5)
        invisible = set(self.encoder.ids) - {r.edge_id for r in observation.visible}
        columns = self.encoder.specification()["discovered_only_columns"]
        self.assertTrue(np.all(state["x"][[self.encoder.idx[e] for e in invisible]][:, columns] == 0))
        self.assertTrue(np.all(state["x"][:, 9] == 0))
        road = observation.candidates[0]
        later = replace(observation, completed=((road, 67),), candidates=observation.candidates[1:])
        self.assertAlmostEqual(float(self.encoder.encode(later)["x"][self.encoder.idx[road], 8]), 0.7**3)

    def test_new_duration_rule_in_all_splits_and_policy_inputs(self):
        classes = self.problem["edges"].set_index("edge_id").road_class.to_dict()
        for scenarios in make_splits(self.problem).values():
            for scenario in scenarios:
                for road in scenario.roads:
                    self.assertEqual(road.estimated_duration,
                                     estimated_duration(classes[road.edge_id], road.estimated_severity))
        observation = self.observation()
        for with_class in (False, True):
            encoder = DailyEncoder(self.problem, self.context, road_class_input=with_class)
            state = encoder.encode(observation)
            for road in observation.visible:
                self.assertAlmostEqual(float(state["x"][encoder.idx[road.edge_id], 5]),
                                       road.estimated_duration / 35, places=6)
        spec = public_spec(self.problem)
        self.assertEqual(spec["rules"]["version"], "daily_discovery_v3")
        self.assertEqual(spec["estimated_duration_days"][str(("major", 1))], 4)

    def test_estimate_calculation_does_not_change_other_random_labels(self):
        original = make_splits(self.problem)
        with patch("src.problem.daily.estimated_duration", return_value=99):
            modified = make_splits(self.problem)
        for split in original:
            for first, second in zip(original[split], modified[split]):
                self.assertEqual(first.scenario_id, second.scenario_id)
                self.assertNotEqual(first.signature, second.signature)
                for a, b in zip(first.roads, second.roads):
                    self.assertEqual(replace(a, estimated_duration=99), b)
                for day in (0, 1, 7, 14):
                    self.assertEqual([r.edge_id for r in first.visible(day)],
                                     [r.edge_id for r in second.visible(day)])

    def test_mixed_demand_warm_start(self):
        from src.environment.ue import _Network, solve_ue
        from src.environment.evaluate import _matrix_from_H, od_travel_times
        c = self.context
        demand = c["H0"] * np.linspace(0.85, 1.1, len(c["H0"]))
        seed = _feasible_daily_seed(c["edges"], demand, c, (self.problem["baseline_links"], c["H0"]))
        network = _Network(c["edges"], _matrix_from_H(demand, c), c["zone_ids"])
        outflow = np.bincount(network.ti, weights=seed, minlength=network.n)
        inflow = np.bincount(network.hi, weights=seed, minlength=network.n)
        expected = np.zeros(network.n)
        for (origin, destination), q in zip(c["od_pairs"], demand):
            expected[network.pos[origin]] += q
            expected[network.pos[destination]] -= q
        self.assertTrue(np.allclose(outflow-inflow, expected, atol=1e-7))
        cold, _ = solve_ue(c["edges"], _matrix_from_H(demand, c), c["zone_ids"],
                           rgap=1e-5, max_iter=2000, quiet=True)
        warm, _ = solve_ue(c["edges"], _matrix_from_H(demand, c), c["zone_ids"], x0=seed,
                           rgap=1e-5, max_iter=2000, quiet=True)
        a, b = od_travel_times(cold, c), od_travel_times(warm, c)
        self.assertLess(float(np.max(np.abs(a-b)/a)), 0.005)

    def test_shifted_objective_gradient(self):
        from types import SimpleNamespace
        from scipy.sparse import csc_matrix
        from src.environment.shifted_demand import potential_factory
        network = SimpleNamespace(t0=np.array([2., 3., 4.]), alpha=np.full(3, .15),
                                  beta=np.full(3, 4.), cap=np.array([9., 13., 7.]))
        network.cost = lambda x: network.t0*(1+network.alpha*(x/network.cap)**network.beta)
        incidence = csc_matrix([[1.,0.,1.,0.], [0.,1.,0.,1.], [1.,1.,0.,0.]])
        a, b = np.array([5.,8.]), np.array([2.,3.])
        objective = potential_factory(network, incidence, np.array([0,0,1,1]), a, b,
            np.array([5.,7.]), .027, np.array([True,True]), b*.01)
        for f in (np.array([4.,4.,6.,6.]), np.array([1.,1.,2.,2.])):
            _, gradient = objective(f)
            numeric = []
            for j in range(4):
                step = np.zeros(4); step[j] = 1e-5
                numeric.append((objective(f+step)[0]-objective(f-step)[0])/2e-5)
            self.assertLess(float(np.max(np.abs(gradient-numeric)/np.maximum(1,np.abs(gradient)))), 1e-7)

    def test_network_and_resumable_training(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(42)
        state = self.encoder.encode(self.observation())
        model = DailyQ(self.encoder)
        self.assertEqual(tuple(model(state).shape), (len(state["cand"]),))
        self.assertTrue(torch.equal(model(state), model(state)))
        gradient = model(state).sum()
        gradient.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.net.parameters() if p.grad is not None))

        def dummy_batch(split, indices, weights, epsilon, seeds):
            self.assertIn(split, ("train", "validation"))
            return [dict(states=[state], picks=[0], rewards=[-1.0],
                         result=dict(objective=1.0, order=[state["roads"][0]])) for _ in indices]

        with tempfile.TemporaryDirectory(prefix="road_daily_training_check_") as temporary:
            arguments = dict(train_count=2, validation_count=1, seed=42, run_dir=temporary,
                             identity={"test_only": True}, hp=dict(pool_n=2, batch_worlds=1, updates_per_ep=1, batch=1),
                             stop_params=dict(n_val=1, ep_min=0, patience_P=1, probe_every=1), ep_cap=5)
            trained, first = train(self.encoder, dummy_batch, **arguments)
            resumed, second = train(self.encoder, dummy_batch, **arguments)
            self.assertEqual(first["outcome"], "plateau_no_improvement")
            self.assertEqual(first["episodes"], 2)
            self.assertEqual(second["episodes"], 2)
            self.assertTrue(all(torch.equal(v, resumed.net.state_dict()[k]) for k, v in trained.net.state_dict().items()))


def traffic_checks():
    started = time.perf_counter()
    problem = build_problem()
    splits = make_splits(problem)
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [64, 11, 50]
    public = set(problem["public_roads"])
    for sample in splits.values():
        for scenario in sample:
            assert len(scenario.roads) == 11
            assert {r.edge_id for r in scenario.roads if r.discovery_day == 0} == public
            assert all(1 <= r.discovery_day <= 14 for r in scenario.roads if r.edge_id not in public)
    simulator = DailyTraffic(problem)
    scenario = splits["train"][0]
    for behavior in ("natural", "response7d"):
        ep_start = time.perf_counter()
        env = DailyEpisode(simulator, scenario, behavior, disconnected_flow="unserved")
        rewards = []
        while not env.finished:
            observation = env.observation()
            road = max(observation.candidates, key=problem["baseline_flow"].get)
            _, reward, _ = env.step(road)
            rewards.append(reward)
        result = env.result()
        assert abs(sum(rewards) + result["objective"] - result["initial_loss"]) < 1e-9
        print(behavior, {k: result[k] for k in ("objective", "recovery_day", "repair_completion_day")},
              "seconds", round(time.perf_counter() - ep_start, 2), flush=True)
    # Previous daily demand is part of the seven-day state, not just current damage.
    damage = {r.edge_id: r.true_severity for r in scenario.roads}
    first = simulator.solve(1, damage, "response7d")
    altered = replace(first, key="different_history", demand=first.demand * 0.9)
    original_next = simulator.solve(2, damage, "response7d", first)
    altered_next = simulator.solve(2, damage, "response7d", altered)
    assert original_next.key != altered_next.key
    assert not np.allclose(original_next.demand, altered_next.demand)
    print("daily traffic checks passed; seconds", round(time.perf_counter() - started, 2), flush=True)


def parallel_checks():
    """Real traffic and current 17-column policies in the production worker path."""
    from concurrent.futures import ProcessPoolExecutor
    from src.experiment.daily import _initialize_worker, _run_world
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(42)
    problem = build_problem()
    context = context_from_problem(problem)
    encoder = DailyEncoder(problem, context, road_class_input=True)
    policy = DailyQ(encoder)
    sample = make_splits(problem)["train"][:3]
    requests = [(s, b, policy.net.state_dict(), 0.0, 0)
                for b in ("natural", "response7d") for s in sample]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="road_daily_cache_check_") as cache:
        with ProcessPoolExecutor(max_workers=3, initializer=_initialize_worker,
                                 initargs=(problem, cache, None, True)) as pool:
            results = list(pool.map(_run_world, requests))
            # Cache round trip must reproduce the complete observed trajectory.
            repeated = list(pool.map(_run_world, requests))
    for request, first, second in zip(requests, results, repeated):
        assert first["result"]["order"] == second["result"]["order"]
        assert abs(first["result"]["objective"] - second["result"]["objective"]) < 1e-10
        assert len(first["states"]) == 11
        assert all(s["x"].shape == (38, 17) for s in first["states"])
        assert all(np.array_equal(a["x"], b["x"]) and np.array_equal(a["g"], b["g"])
                   for a, b in zip(first["states"], second["states"]))
        print(request[0].scenario_id, request[1], "days", first["result"]["recovery_day"],
              "objective", first["result"]["objective"], "runtime", first["runtime"], flush=True)
    print("parallel daily checks passed; seconds", round(time.perf_counter() - started, 2), flush=True)


def preflight_checks():
    """All training/validation worlds, held-out tests untouched, random legal orders.

    This is a numerical/information-contract check, NOT policy selection. A
    deterministic random-order policy is used in every world. Cache is reusable
    only under the simulator's exact source/settings fingerprint.
    """
    from concurrent.futures import ProcessPoolExecutor
    from src.experiment.daily import _initialize_worker, _run_world
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(42)
    problem = build_problem()
    simulator = DailyTraffic(problem)
    encoder = DailyEncoder(problem, simulator.ctx)
    policy = DailyQ(encoder)
    splits = make_splits(problem)
    sample = splits["train"] + splits["validation"]
    requests = [(s, "response7d", policy.net.state_dict(), 1.0, 701+i) for i,s in enumerate(sample)]
    root = Path(__file__).resolve().parents[2]
    cache = root / ".cache" / "daily_traffic"
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, initializer=_initialize_worker,
                             initargs=(problem, cache, None)) as pool:
        for i, trajectory in enumerate(pool.map(_run_world, requests)):
            result = trajectory["result"]
            daily = result["daily"]
            assert len(result["order"]) == 11
            assert np.isfinite(result["objective"])
            assert max(r["fixed_point_residual"] for r in daily) <= problem["rules"].fixed_point_tolerance
            assert max(r["ue_relative_gap"] for r in daily) <= problem["rules"].response_ue_gap
            assert all(r["fixed_point_residual"] <= r["fixed_point_tolerance"]
                       and r["ue_relative_gap"] <= r["ue_tolerance"] for r in daily)
            assert all(r["refined_after_repairs"] == (r["repaired"] == 11) for r in daily)
            rows.append(dict(scenario=result["scenario"], objective=result["objective"],
                             recovery_day=result["recovery_day"],
                             fallback_days=sum(r.get("solver_fallback",False) for r in daily),
                             max_residual=max(r["fixed_point_residual"] for r in daily),
                             **trajectory["runtime"]))
            if (i+1)%5 == 0:
                print("preflight",i+1,"/",len(sample),"seconds",round(time.perf_counter()-started,1),flush=True)
    report = dict(simulator_fingerprint=simulator.fingerprint, held_out_test_used=False,
                  checked_worlds=len(rows), elapsed_seconds=time.perf_counter()-started,
                  fallback_days=sum(r["fallback_days"] for r in rows), rows=rows)
    path = root / ".cache" / "daily_preflight" / (simulator.fingerprint[:16]+".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf8")
    print("preflight passed",path,"fallback days",report["fallback_days"],flush=True)


def equivalence_checks():
    """Same world and public flow-priority schedule, two numerical solvers."""
    problem = build_problem()
    scenario = make_splits(problem)["train"][0]
    results = {}
    for solver in ("joint", "nested"):
        configured = dict(problem, rules=replace(problem["rules"], response_solver=solver))
        simulator = DailyTraffic(configured)
        started = time.perf_counter()
        episode = DailyEpisode(simulator, scenario, "response7d", disconnected_flow="unserved")
        while not episode.finished:
            road = max(episode.observation().candidates, key=problem["baseline_flow"].get)
            episode.step(road)
        results[solver] = episode.result()
        print("equivalence",solver,"objective",results[solver]["objective"],
              "recovery day",results[solver]["recovery_day"],
              "repair completion day",results[solver]["repair_completion_day"],
              "seconds",round(time.perf_counter()-started,2),flush=True)
    report = Path(__file__).resolve().parents[2] / ".cache" / "daily_preflight" / "solver_equivalence.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results,indent=2),encoding="utf8")
    assert results["joint"]["order"] == results["nested"]["order"]
    assert results["joint"]["completions"] == results["nested"]["completions"]
    assert results["joint"]["recovery_day"] == results["nested"]["recovery_day"]
    difference = abs(results["joint"]["objective"]-results["nested"]["objective"])/results["nested"]["objective"]
    assert difference < .005, difference
    print("joint/nested objective relative difference",difference,flush=True)


if __name__ == "__main__":
    if "--traffic" in sys.argv:
        traffic_checks()
    elif "--parallel" in sys.argv:
        parallel_checks()
    elif "--preflight" in sys.argv:
        preflight_checks()
    elif "--equivalence" in sys.argv:
        equivalence_checks()
    else:
        unittest.main()
