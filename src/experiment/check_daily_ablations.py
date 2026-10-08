"""Three-RL experiment checks; fixtures and any figures stay in temporary folders."""
from dataclasses import replace
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.environment.daily import DailyEpisode, Observation, context_from_problem
from src.experiment.check_daily import ConstantTraffic
from src.experiment.daily import ablation_plan, _initialize_worker, _run_world, run_rl_ablations
from src.methods.daily_s2v import (DailyEncoder, DailyQ, train,
                                   TRAFFIC_NODE_COLUMNS, TRAFFIC_GLOBAL_COLUMNS)
from src.problem.daily import build_problem, make_splits


class EncoderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = build_problem()
        cls.context = context_from_problem(cls.problem)
        cls.scenario = make_splits(cls.problem)["train"][0]
        cls.full = DailyEncoder(cls.problem, cls.context, road_class_input=True)
        cls.ablated = DailyEncoder(cls.problem, cls.context, road_class_input=True, traffic_input=False)

    def observation(self):
        c = self.context
        visible = self.scenario.visible(1)
        return Observation(1, visible, (), tuple(r.edge_id for r in visible),
                           c["H0"]*.5, c["H0"]*.75, c["baseline_u"].copy(),
                           np.arange(len(c["H0"])) % 2 == 0, np.ones(38)*1234, np.ones(38)*.4)

    def test_exactly_six_feature_channels_change(self):
        observation = self.observation()
        full, ablated = self.full.encode(observation), self.ablated.encode(observation)
        expected_x, expected_g = full["x"].copy(), full["g"].copy()
        expected_x[:, TRAFFIC_NODE_COLUMNS] = 0
        expected_g[list(TRAFFIC_GLOBAL_COLUMNS)] = 0
        np.testing.assert_array_equal(ablated["x"], expected_x)
        np.testing.assert_array_equal(ablated["g"], expected_g)
        self.assertEqual(ablated["x"].shape, (38, 17))
        self.assertEqual(ablated["g"].shape, (6,))
        self.assertEqual(ablated["cand"], full["cand"])
        self.assertEqual(ablated["roads"], full["roads"])
        self.assertEqual(ablated["day"], full["day"])
        self.assertGreater(np.abs(full["x"][:, TRAFFIC_NODE_COLUMNS]).sum(), 0)

    def test_all_live_traffic_values_can_be_absent_without_affecting_input(self):
        observation = self.observation()
        poisoned = replace(observation, demand=None, costs=None, reachable=None,
                           road_flow=None, road_congestion=None, link_traffic=None)
        first, second = self.ablated.encode(observation), self.ablated.encode(poisoned)
        np.testing.assert_array_equal(first["x"], second["x"])
        np.testing.assert_array_equal(first["g"], second["g"])

    def test_network_initialization_and_learning_parameters_unchanged(self):
        torch.manual_seed(42)
        full = DailyQ(self.full)
        torch.manual_seed(42)
        ablated = DailyQ(self.ablated)
        self.assertEqual(full.hp, ablated.hp)
        for name, value in full.net.state_dict().items():
            self.assertTrue(torch.equal(value, ablated.net.state_dict()[name]))
        schema = self.ablated.specification()
        self.assertEqual(schema["zeroed_node_columns"], [10, 11, 12, 13])
        self.assertEqual(schema["zeroed_global_columns"], [4, 5])
        self.assertEqual(schema["node_dimension"], 17)

    def test_reward_still_uses_environment_and_not_encoded_traffic(self):
        results = []
        for encoder in (self.full, self.ablated):
            sim = ConstantTraffic()
            # Replace only the traffic solver with a fixture, keeping actual road
            # truth/discoveries and scheduling. The policy uses one fixed order.
            episode = DailyEpisode(sim, self.scenario, "response7d", disconnected_flow="unserved")
            rewards = []
            while not episode.finished:
                observation = episode.observation()
                encoder.encode(replace(observation, demand=self.context["H0"]*.5,
                                       external=self.context["H0"]*.75,
                                       costs=self.context["baseline_u"],
                                       reachable=np.ones(len(self.context["H0"]), dtype=bool)))
                _, reward, _ = episode.step(min(observation.candidates))
                rewards.append(reward)
            results.append((episode.result(), rewards))
        self.assertEqual(results[0], results[1])

    def test_ablated_training_resume_and_contract_guard(self):
        state = self.ablated.encode(self.observation())
        def batch(split, indices, weights, epsilon, seeds):
            self.assertIn(split, ("train", "validation"))
            return [dict(states=[state], picks=[0], rewards=[-1.],
                         result=dict(objective=1., order=[state["roads"][0]])) for _ in indices]
        with tempfile.TemporaryDirectory() as folder:
            arguments = dict(train_count=2, validation_count=1, seed=42, run_dir=folder,
                             identity={"fixture": True}, hp=dict(pool_n=2, batch_worlds=1, updates_per_ep=1, batch=1),
                             stop_params=dict(n_val=1, ep_min=0, patience_P=1, probe_every=1), ep_cap=5)
            first, summary = train(self.ablated, batch, **arguments)
            resumed, again = train(self.ablated, batch, **arguments)
            self.assertEqual(summary["episodes"], again["episodes"])
            self.assertTrue(all(torch.equal(w, resumed.net.state_dict()[k]) for k, w in first.net.state_dict().items()))
            with self.assertRaisesRegex(ValueError, "resume settings differ"):
                train(self.full, batch, **arguments)


class PlanChecks(unittest.TestCase):
    def test_plan_has_two_contrasts_with_one_shared_full_model(self):
        plan = ablation_plan()
        self.assertEqual(len(plan["methods"]), 3)
        self.assertEqual([(m["training_behavior"], m["traffic_input"]) for m in plan["methods"]],
                         [("natural", True), ("response7d", True), ("response7d", False)])
        self.assertEqual(len(plan["comparisons"]), 2)
        self.assertEqual(plan["comparisons"][0]["methods"][1], plan["comparisons"][1]["methods"][1])
        self.assertEqual(plan["test_scenarios"], 50)
        self.assertEqual(plan["evaluation_environment"], "response7d")

    def test_suite_calls_each_model_once_with_prespecified_flags(self):
        plan = ablation_plan()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "data/log").mkdir(parents=True)
            for pair in plan["comparisons"]:
                (folder / pair["filename"]).touch()  # existence-only fixture; not a delivered image
            calls = []
            def fake_run(n, behavior, seed, workers, **kwargs):
                i = len(calls)
                calls.append((n, behavior, seed, workers, kwargs))
                path = folder / "data/methods" / plan["methods"][i]["name"] / "log/status.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"status": "complete"}))
                return folder
            with patch("src.experiment.daily.prepare_ablation_experiment", return_value=(folder, plan)), \
                 patch("src.experiment.daily.run", side_effect=fake_run), \
                 patch("src.experiment.daily.compare"):
                self.assertEqual(run_rl_ablations(), folder)
            self.assertEqual([(r[1], r[4]) for r in calls],
                             [("natural", {"road_class_input": True, "traffic_input": True}),
                              ("response7d", {"road_class_input": True, "traffic_input": True}),
                              ("response7d", {"road_class_input": True, "traffic_input": False})])
            self.assertEqual(json.loads((folder / "data/log/rl_ablation_status.json").read_text())["status"], "complete")

    def test_requested_pair_data_and_shared_scale(self):
        from src.analysis.daily import refresh_controlled_comparisons
        plan = ablation_plan()
        rows = [dict(method=method["name"], scenario=f"test_{i:03d}", F=5.+m+i/100)
                for m, method in enumerate(plan["methods"]) for i in range(50)]
        table = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "results").mkdir()
            with patch("src.analysis.viz.daily_viz.make_controlled_pair") as draw:
                summary = refresh_controlled_comparisons(table, folder, plan)
                self.assertEqual(draw.call_count, 2)
                self.assertEqual(draw.call_args_list[0].kwargs["y_max"], draw.call_args_list[1].kwargs["y_max"])
                self.assertEqual(summary[0]["mean_difference"], -1.)
                self.assertEqual(summary[1]["mean_difference"], 1.)
                self.assertEqual(summary[0]["second_wins"], 0)
                self.assertEqual(summary[1]["second_wins"], 50)
            incomplete = table[~((table.method == plan["methods"][2]["name"]) & (table.scenario == "test_000"))]
            with patch("src.analysis.viz.daily_viz.make_controlled_pair"), self.assertRaisesRegex(ValueError, "complete test sample"):
                refresh_controlled_comparisons(incomplete, folder, plan)

    def test_single_method_does_not_create_a_comparison(self):
        from src.analysis.daily import refresh_controlled_comparisons
        plan = ablation_plan()
        first = plan["methods"][0]["name"]
        table = pd.DataFrame([dict(method=first, scenario=f"test_{i:03d}", F=1.) for i in range(50)])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "results").mkdir()
            with patch("src.analysis.viz.daily_viz.make_controlled_pair") as draw:
                result = refresh_controlled_comparisons(table, folder, plan)
                draw.assert_not_called()
                self.assertTrue(all(p["status"] == "waiting_for_methods" for p in result))

    def test_cli_selects_the_suite(self):
        from main import daily_cli
        with patch("src.experiment.daily.run_rl_ablations") as run, redirect_stdout(io.StringIO()):
            daily_cli(["--solve", "rl_ablations"])
            run.assert_called_once_with(11, 42, 4)


def real_traffic_check():
    """One TRAIN world in the full environment, with and without traffic input."""
    problem = build_problem()
    scenario = make_splits(problem)["train"][0]
    context = context_from_problem(problem)
    summaries = []
    for traffic_input in (True, False):
        encoder = DailyEncoder(problem, context, road_class_input=True, traffic_input=traffic_input)
        torch.manual_seed(42)
        weights = DailyQ(encoder).net.state_dict()
        _initialize_worker(problem, None, None, True, traffic_input)
        trajectory = _run_world((scenario, "response7d", weights, 0., 0))
        for state in trajectory["states"]:
            assert state["x"].shape == (38, 17) and state["g"].shape == (6,)
            if not traffic_input:
                assert not state["x"][:, TRAFFIC_NODE_COLUMNS].any()
                assert not state["g"][list(TRAFFIC_GLOBAL_COLUMNS)].any()
        result = trajectory["result"]
        assert abs(sum(trajectory["rewards"]) + result["objective"]) < 1e-8
        assert len(result["order"]) == 11
        summaries.append(dict(traffic_input=traffic_input, objective=result["objective"],
                              recovery_day=result["recovery_day"], **trajectory["runtime"]))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    import sys
    if "--traffic" in sys.argv:
        real_traffic_check()
    else:
        unittest.main()
