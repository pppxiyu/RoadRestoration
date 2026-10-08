"""Regression checks for the finalized logic-map rules; no official experiment writes.

Run with python -m unittest src.experiment.check_problem_definition.
"""
from dataclasses import asdict, replace
from contextlib import redirect_stderr
import importlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config as P
from src.problem.daily import (CONFUSION, ESTIMATE_PROBABILITIES, QUOTAS, DailyRules,
                               RoadDamage, Scenario, build_problem, make_splits,
                               duration_draw, duration_expectation, public_spec)
from src.environment.daily import (DailyEpisode, DailyTraffic, TrafficState,
                                   context_from_problem, daily_losses, recovery_fraction)
from src.environment.evaluate import build_damaged_edges
from src.experiment.check_daily import ConstantTraffic
from src.experiment.daily import prepare, save_problem
from src.experiment.layout import daily_experiment_identity
from src.methods.daily_s2v import DailyEncoder


ROOT = Path(__file__).resolve().parents[2]
OLD_SD = {("local", 1): 1., ("local", 2): 1.8, ("local", 3): 2.4,
          ("major", 1): 2.2, ("major", 2): 3.2, ("major", 3): 3.8,
          ("highway", 1): 3.2, ("highway", 2): 4.5, ("highway", 3): 5.5}


class ProblemDefinitionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = {n: build_problem(n) for n in QUOTAS}
        cls.samples = {n: make_splits(p) for n, p in cls.problems.items()}
        cls.problem = cls.problems[11]
        cls.context = context_from_problem(cls.problem)

    def test_01_recovery_matches_approved_rescaling(self):
        initial, plateau = P.RECOVERY_INITIAL_LEVEL, P.RECOVERY_PLATEAU_LEVEL
        ratio = plateau / initial - 1
        rate = np.log(ratio / (1 / .99 - 1)) / 35
        old = lambda t: plateau / (1 + ratio * np.exp(-rate * t))
        days = np.arange(36)
        expected = initial + (1 - initial) * (old(days) - old(0)) / (old(35) - old(0))
        np.testing.assert_allclose(recovery_fraction(days, DailyRules()), expected)
        np.testing.assert_array_equal(recovery_fraction(np.arange(35, 501), DailyRules()), 1.)

    def test_02_all_scales_public_hidden_and_flow_quotas(self):
        self.assertEqual(QUOTAS, {6: (1, 4, 0, 1), 11: (2, 6, 0, 3),
                                 17: (2, 11, 1, 3), 23: (4, 13, 1, 5)})
        for n, problem in self.problems.items():
            with self.subTest(n=n):
                high, middle, low = map(set, problem["bins"])
                self.assertEqual(tuple(map(len, problem["bins"])), (8, 21, 9))
                ranked = problem["ranked"]
                self.assertEqual(high, set(ranked[:8]))
                self.assertEqual(middle, set(ranked[8:29]))
                self.assertEqual(ranked, sorted(ranked, key=lambda e: (-problem["baseline_flow"][e], e)))
                public = set(problem["public_roads"])
                ph, pm, hh, hm = QUOTAS[n]
                self.assertEqual(len(public), int(np.floor(.75 * n + .5)))
                self.assertEqual((len(public & high), len(public & middle)), (ph, pm))
                hidden_sets = set()
                for sample in self.samples[n].values():
                    for scenario in sample:
                        ids = {r.edge_id for r in scenario.roads}
                        hidden = ids - public
                        self.assertEqual(len(ids), n)
                        self.assertTrue(public <= ids)
                        self.assertFalse(ids & low)
                        self.assertEqual((len(hidden & high), len(hidden & middle)), (hh, hm))
                        self.assertEqual({r.edge_id for r in scenario.visible(0)}, public)
                        self.assertEqual({r.edge_id for r in scenario.visible(14)}, ids)
                        hidden_sets.add(tuple(sorted(hidden)))
                self.assertGreater(len(hidden_sets), 1)

    def test_03_severity_probabilities_and_physical_state(self):
        self.assertEqual(ESTIMATE_PROBABILITIES, (.25, .5, .25))
        self.assertEqual(CONFUSION, {1: (.6, 1/3, 1/15), 2: (.2, .6, .2), 3: (1/15, 1/3, .6)})
        road = self.problem["edges"].iloc[0]
        for severity, cap, speed in ((1, .3, .5), (2, .1, .3)):
            damaged = build_damaged_edges(self.context, {int(road.edge_id): severity})
            actual = damaged.loc[damaged.edge_id == road.edge_id].iloc[0]
            self.assertAlmostEqual(actual.capacity, road.capacity * cap)
            self.assertAlmostEqual(actual.free_flow_time, road.free_flow_time / speed)
        severed = build_damaged_edges(self.context, {int(road.edge_id): 3})
        self.assertNotIn(road.edge_id, severed.edge_id.to_list())

    def test_04_duration_parameters_match_logic_map_exactly(self):
        text = (ROOT / "doc/notes/repository_logic_map.md").read_text(encoding="utf-8")
        for chinese, name in (("支路", "local"), ("主干路", "major"), ("高速道路", "highway")):
            line = next(line for line in text.splitlines() if line.startswith(f"| {chinese} |"))
            cells = re.findall(r"(\d+(?:\.\d+)?)／(\d+(?:\.\d+)?)", line)
            self.assertEqual(len(cells), 3)
            for severity, (mean, sd) in enumerate(cells, 1):
                self.assertEqual(P.DUR_MEAN[name, severity], float(mean))
                self.assertEqual(P.DUR_SD[name, severity], float(sd))
        self.assertEqual(P.DUR_TRUNC_MULT, 2.5)
        example = re.search(r"实际整数工期期望约为([\d.]+)天", text)
        self.assertAlmostEqual(float(example[1]), duration_expectation("major", 1), places=6)

    def test_04_draw_matches_truncated_integer_distribution(self):
        for cell, mean in P.DUR_MEAN.items():
            variance = np.log1p((P.DUR_SD[cell] / mean) ** 2)
            mu, sigma, upper = np.log(mean) - variance / 2, np.sqrt(variance), mean * 2.5
            scale = norm.cdf((np.log(upper) - mu) / sigma)
            days = np.arange(1, int(np.floor(upper + .5)) + 1)
            cdf = norm.cdf((np.log(np.minimum(days + .5, upper)) - mu) / sigma) / scale
            pmf = np.diff(np.r_[0., cdf])
            self.assertAlmostEqual(pmf.sum(), 1.)
            self.assertAlmostEqual(float(days @ pmf), duration_expectation(*cell), places=12)
            for day, lo, hi in zip(days, np.r_[0., cdf[:-1]], cdf):
                if hi - lo > 1e-9:
                    self.assertEqual(duration_draw(*cell, float((lo + hi) / 2)), day)
            self.assertGreaterEqual(duration_draw(*cell, 0), 1)
            self.assertLessEqual(duration_draw(*cell, 1), int(np.floor(upper + .5)))

    def test_04_changed_spread_does_not_change_other_random_labels(self):
        with patch.dict(P.DUR_SD, OLD_SD):
            previous = make_splits(self.problem)
        changes = 0
        for name, sample in self.samples[11].items():
            for old, new in zip(previous[name], sample):
                self.assertEqual(old.scenario_id, new.scenario_id)
                for a, b in zip(old.roads, new.roads):
                    self.assertEqual((a.edge_id, a.estimated_severity, a.true_severity, a.discovery_day),
                                     (b.edge_id, b.estimated_severity, b.true_severity, b.discovery_day))
                    changes += a.true_duration != b.true_duration
        self.assertGreater(changes, 0)

    def test_05_discovery_and_06_information_boundary(self):
        for sample in self.samples[11].values():
            for scenario in sample:
                for road in scenario.roads:
                    if road.edge_id not in self.problem["public_roads"]:
                        self.assertTrue(1 <= road.discovery_day <= 14)
                        self.assertNotIn(road.edge_id, {r.edge_id for r in scenario.visible(road.discovery_day - 1)})
                    visible = next(r for r in scenario.visible(road.discovery_day) if r.edge_id == road.edge_id)
                    self.assertEqual(set(asdict(visible)), {"edge_id", "estimated_severity", "estimated_duration"})
                    self.assertEqual(visible.estimated_severity, road.estimated_severity)
                    self.assertEqual(visible.estimated_duration, road.estimated_duration)

    def test_06_complete_directional_observation_has_no_hidden_labels(self):
        sim = ConstantTraffic()
        sim.ctx = dict(sim.ctx, edges=self.problem["edges"])
        scenario = Scenario("visible", (RoadDamage(1, 1, 2, 3, 4, 0), RoadDamage(2, 2, 7, 1, 3, 14)))
        episode = DailyEpisode(sim, scenario, "response7d", disconnected_flow="unserved")
        edge = self.problem["edges"].iloc[0]
        episode.traffic.links = pd.DataFrame([[int(edge.u), int(edge.v), 12., 4.]],
                                            columns=["from", "to", "volume", "cost"])
        observation = episode.observation()
        self.assertEqual(len(observation.link_traffic), 76)
        actual = next(r for r in observation.link_traffic if (r.origin, r.destination) == (edge.u, edge.v))
        self.assertEqual((actual.flow, actual.travel_time), (12., 4.))
        self.assertEqual(set(asdict(actual)), {"origin", "destination", "flow", "travel_time"})
        self.assertEqual([r.edge_id for r in observation.visible], [1])
        observation.demand[0] = -100
        self.assertGreaterEqual(episode.traffic.served_demand[0], 0)

    def test_07_no_wait_and_no_fixed_horizon(self):
        scenario = Scenario("long", (RoadDamage(1, 1, 2, 3, 45, 0),))
        episode = DailyEpisode(ConstantTraffic(), scenario, "natural", disconnected_flow="unserved")
        with self.assertRaises(ValueError):
            episode.step(None)
        _, reward, done = episode.step(1)
        self.assertTrue(done)
        result = episode.result()
        self.assertEqual(result["repair_completion_day"], 46)
        self.assertEqual(result["recovery_day"], 47)
        self.assertAlmostEqual(result["objective"], 45 * .5 * .5)
        self.assertAlmostEqual(reward, -result["objective"])

    def test_08_flow_loss_is_per_od_and_time_uses_actual_trips(self):
        state = TrafficState("loss", 1, np.array([0., 20., 0.]), np.ones(3),
                             np.array([np.inf, 6., np.inf]), np.array([False, True, False]),
                             np.zeros(38), np.zeros(38), None, {})
        context = dict(H0=np.array([10., 10., 0.]), baseline_u=np.array([2., 4., 1.]))
        result = daily_losses(state, context, disconnected_flow="unserved")
        self.assertEqual(result["flow_loss"], .5)  # one OD's surplus cannot cancel another's loss
        self.assertEqual(result["time_loss"], .5)
        self.assertEqual(result["min_flow_fraction"], 0.)
        state.reachable[:] = False
        result = daily_losses(state, context, disconnected_flow="unserved")
        self.assertEqual(result["flow_loss"], 1.)
        self.assertEqual(result["time_loss"], 0.)

    def test_08_retired_penalty_option_rejected_before_simulation(self):
        sim = ConstantTraffic()
        with self.assertRaisesRegex(ValueError, "unserved"):
            DailyEpisode(sim, self.samples[11]["train"][0], "natural", disconnected_flow="penalized_demand")
        self.assertFalse(sim.observed_damage)
        with self.assertRaisesRegex(ValueError, "unserved"):
            daily_losses(None, None, disconnected_flow="penalized_demand")

    def test_09_all_repairs_individual_od_and_consecutive_days(self):
        scenario = Scenario("stop", (RoadDamage(1, 1, 1, 1, 1, 0),))
        sim = ConstantTraffic()
        episode = DailyEpisode(sim, scenario, "natural", disconnected_flow="unserved")
        sim.ctx = dict(H0=np.array([10., 10.]), baseline_u=np.ones(2))
        def day(flows, cost=1.):
            episode.traffic = TrafficState("test", episode.day, np.array(flows), np.array(flows),
                                           np.array([cost, cost]), np.ones(2, dtype=bool),
                                           np.zeros(38), np.zeros(38), None, {})
            episode._charge_today()
        day([10., 10.])
        self.assertEqual(episode.restored_streak, 0)  # repairs not done
        episode.completed[1] = episode.day
        day([9.8, 10.2])
        self.assertEqual(episode.restored_streak, 0)  # total flow is not sufficient
        day([10., 10.], 1.02)
        self.assertEqual(episode.restored_streak, 0)
        day([9.9, 9.9])
        self.assertFalse(episode.finished)
        day([9.8, 10.2])
        self.assertEqual(episode.restored_streak, 0)  # consecutive, not cumulative
        day([10., 10.])
        day([10., 10.])
        self.assertTrue(episode.finished)

    def test_experiment_samples_repeat_and_splits_do_not_overlap(self):
        for n in QUOTAS:
            fresh = make_splits(self.problems[n])
            self.assertEqual(tuple(map(len, fresh.values())), (64, 11, 50))
            self.assertEqual(fresh, self.samples[n])
            all_signatures = [s.signature for sample in fresh.values() for s in sample]
            self.assertEqual(len(set(all_signatures)), 125)

    def test_experiment_identity_changes_with_rule_or_distribution(self):
        simulator = DailyTraffic(self.problem)
        folder, identity = daily_experiment_identity(self.problem, simulator, self.samples[11])
        with patch.dict(P.DUR_SD, OLD_SD):
            old_folder, _ = daily_experiment_identity(self.problem, simulator, make_splits(self.problem))
        old_rules = dict(self.problem, rules=replace(self.problem["rules"], version="daily_discovery_v2"))
        other_folder, _ = daily_experiment_identity(old_rules, simulator, self.samples[11])
        self.assertEqual(len({folder, old_folder, other_folder}), 3)
        self.assertEqual(identity["evaluation_behavior"], "response7d")
        self.assertEqual(identity["disconnected_flow"], "unserved")
        self.assertEqual(identity["problem"]["rules"]["time_weight"], .5)
        self.assertEqual(identity["problem"]["rules"]["flow_weight"], .5)
        simulator.close()

    def test_saved_problem_is_immutable_and_refuses_conflicting_samples(self):
        simulator = DailyTraffic(self.problem)
        encoder = DailyEncoder(self.problem, simulator.ctx, road_class_input=True)
        _, identity = daily_experiment_identity(self.problem, simulator, self.samples[11])
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            save_problem(folder, self.problem, self.samples[11], identity, encoder)
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()}
            save_problem(folder, self.problem, self.samples[11], identity, encoder)
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
            changed = dict(self.samples[11], test=list(reversed(self.samples[11]["test"])))
            with self.assertRaisesRegex(ValueError, "frozen test"):
                save_problem(folder, self.problem, changed, identity, encoder)
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
        simulator.close()

    def test_current_prepare_defaults_to_public_road_class(self):
        problem, splits, simulator, encoder, folder, identity = prepare()
        self.assertEqual(encoder.node_dimension, 17)
        self.assertEqual(public_spec(problem)["rules"]["version"], "daily_discovery_v3")
        simulator.close()


class EntryPointChecks(unittest.TestCase):
    def test_retired_behavior_is_not_an_optimization_option(self):
        from src.environment.behavior_models import select_behavior_model
        with self.assertRaisesRegex(ValueError, "finalized"):
            select_behavior_model("damage_shortfall", 3., for_methods=True)
        self.assertEqual(select_behavior_model("damage_shortfall", 3., for_methods=False).slot_hours, 3.)
        self.assertEqual(select_behavior_model("response7d_daily", 24., for_methods=True).slot_hours, 24.)

    def test_old_cli_rejected_before_any_method_starts(self):
        from main import daily_cli
        with patch("src.experiment.daily.run") as rl, patch("src.experiment.daily_ga.run") as ga:
            for argv in (["--setting", "legacy"], ["--setting=legacy"],
                         ["--solve", "env-behavior"], ["--solve", "ga,oracle"],
                         ["--solve", "ga,"], ["--n", "10"], ["--workers", "0"]):
                with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    daily_cli(argv)
            rl.assert_not_called()
            ga.assert_not_called()

    def test_default_and_natural_ablation_use_current_runner(self):
        from main import daily_cli
        with patch("src.experiment.daily.run") as run:
            daily_cli([])
            run.assert_called_once_with(11, "response7d", 42, 4, road_class_input=True)
            run.reset_mock()
            daily_cli(["--training-behavior", "natural", "--n", "23"])
            run.assert_called_once_with(23, "natural", 42, 4, road_class_input=True)

    def test_retired_generation_and_solver_apis_fail_explicitly(self):
        for module, name in (("problem.instance", "select_oracle_instance"),
                             ("problem.scenarios", "sample_scenarios"),
                             ("methods.greedy", "run_greedy"), ("methods.metaheuristic", "run_metaheuristic"),
                             ("methods.oracle", "run_oracle"), ("methods.pretrain_milp", "run_pretrain_milp"),
                             ("methods.rl_s2v", "run_s2v"), ("methods.rl_s2v_saa", "run_s2v_saa")):
            with self.subTest(module=module), self.assertRaisesRegex(RuntimeError, "retired problem"):
                getattr(importlib.import_module("src." + module), name)()


if __name__ == "__main__":
    unittest.main()
