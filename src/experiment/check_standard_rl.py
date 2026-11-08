"""Small regression checks for the standard RL protocol, never full training."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.experiment.check_policy_selection import TinyEncoder, STATE, equal_tree
from src.methods.daily_s2v import DailyQ
from src.methods.daily_improvement import (
    ImprovementSettings, fit_round, improve, pair_loss, weights_digest, atomic_torch, freeze_json,
)
from src.experiment import daily_rl_workers as workers


class ImprovementChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.hp = dict(p=8, t_emb=1, readout_hidden=8)
        self.initial = copy.deepcopy(DailyQ(TinyEncoder(), self.hp).net.state_dict())
        self.rows = [dict(state=copy.deepcopy(STATE), costs=np.array([1., 2.])),
                     dict(state=copy.deepcopy(STATE), costs=np.array([2., 4.]))]
        self.settings = ImprovementSettings(validation_updates=(1, 2, 3), validation_interval=1, inner_patience=2)

    def options(self):
        return dict(identity={"synthetic": True}, settings=self.settings, initial_validation=3.)

    def test_signed_all_pairs_objective(self):
        scores = torch.tensor([5., 3., -1.], requires_grad=True)
        self.assertEqual(float(pair_loss(scores, [2., 4., 8.]).detach()), 0.)
        self.assertGreater(float(pair_loss(-scores, [2., 4., 8.]).detach()), 0.)

    def test_exact_resume_scheduler_inheritance_and_idempotent_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = [Path(tmp) / name for name in ("full", "resume", "next")]
            values = iter([4., 5., 6., 7., 8.])
            ma, ra = fit_round(TinyEncoder(), self.hp, self.initial, self.rows, lambda w: next(values), a, **self.options())
            early = iter([4., 5.])
            _, paused = fit_round(TinyEncoder(), self.hp, self.initial, self.rows, lambda w: next(early), b,
                                  pause_after=2, **self.options())
            self.assertFalse(paused["done"])
            rest = iter([6., 7., 8.])
            mb, rb = fit_round(TinyEncoder(), self.hp, self.initial, self.rows, lambda w: next(rest), b, **self.options())
            equal_tree(ra, rb)
            equal_tree(ma.net.state_dict(), mb.net.state_dict())
            ca = torch.load(a / "latest.pt", weights_only=False)
            cb = torch.load(b / "latest.pt", weights_only=False)
            equal_tree(ca, cb)
            self.assertEqual(rb["candidate_update"], 1)
            self.assertEqual(rb["schedule"]["lr"], 0.00025)
            self.assertNotEqual(weights_digest(cb["weights"]), weights_digest(mb.net.state_dict()))
            before = (b / "latest.pt").read_bytes()
            with patch.object(self, "fail", side_effect=AssertionError("validation repeated")):
                fit_round(TinyEncoder(), self.hp, self.initial, self.rows, self.fail, b, **self.options())
            self.assertEqual(before, (b / "latest.pt").read_bytes())
            fit_round(TinyEncoder(), self.hp, mb.net.state_dict(), self.rows, lambda w: 9., c,
                      identity={"synthetic": True}, settings=self.settings,
                      schedule=rb["schedule"], pause_after=1)
            cc = torch.load(c / "latest.pt", weights_only=False)
            self.assertEqual(cc["losses"][0]["lr"], rb["schedule"]["lr"])
            self.assertTrue(all(float(s["step"]) == 1 for s in cc["optimizer"]["state"].values()))
            self.assertEqual(cc["scheduler"]["last_epoch"], cb["scheduler"]["last_epoch"] + 1)

    def test_global_best_patience_and_previous_candidate_survive_restart(self):
        initial = {"value": torch.tensor([0.])}
        losses = [9., 11., 10., 11., 10., 11.]
        starts = []

        def collect(weights, directory, positions):
            number = int(directory.name.split("_")[-1])
            self.assertEqual(float(weights["value"]), number - 1)
            self.assertIsNone(positions)
            starts.append(number)
            return []

        def fit(encoder, hp, current, rows, evaluate, directory, **kwargs):
            number = int(directory.name.split("_")[-1])
            model = SimpleNamespace(net=SimpleNamespace(state_dict=lambda: {"value": torch.tensor([float(number)])}))
            return model, dict(done=True, validation=losses[number-1], candidate_update=300,
                               schedule=dict(lr=.001, scheduler={}))

        with tempfile.TemporaryDirectory() as tmp, patch("src.methods.daily_improvement.fit_round", side_effect=fit):
            kwargs = dict(identity={"test": True}, pause_after_round=3)
            _, pause = improve(None, {}, initial, 10., collect, None, tmp, **kwargs)
            self.assertFalse(pause["complete"])
            best, final = improve(None, {}, initial, 10., collect, None, tmp, identity={"test": True})
            self.assertEqual(starts, [1, 2, 3, 4, 5, 6])
            self.assertEqual(float(best["value"]), 1.)
            self.assertEqual(final["selected_round"], 1)
            self.assertEqual(final["consecutive_no_global_improvement"], 5)
            self.assertEqual([r["consecutive_no_global_improvement"] for r in final["rounds"]], [0, 1, 2, 3, 4, 5])

    def test_changed_label_or_nonfinite_validation_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_round(TinyEncoder(), self.hp, self.initial, self.rows, lambda w: 4., tmp,
                      pause_after=1, **self.options())
            changed = copy.deepcopy(self.rows)
            changed[0]["costs"][0] += 1
            with self.assertRaisesRegex(ValueError, "contract changed"):
                fit_round(TinyEncoder(), self.hp, self.initial, changed, lambda w: 4., tmp, **self.options())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FloatingPointError):
                fit_round(TinyEncoder(), self.hp, self.initial, self.rows, lambda w: float("nan"), tmp, **self.options())


class RewardChecks(unittest.TestCase):
    def test_all_nontrivial_decisions_share_parent_and_round_continuation(self):
        class Episode:
            def __init__(self, simulator, scenario, behavior, **kwargs):
                self.simulator = simulator
                self.remaining = list(range(1, 7))
                self.total_loss = 0.
                self.order = []
                self.finished = False

            def observation(self):
                return tuple(self.remaining)

            def step(self, road):
                # Different orders have different costs. This catches any
                # failure to use the same continuation after the first action.
                cost = float(road * len(self.remaining))
                self.total_loss += cost
                self.order.append(road)
                self.remaining.remove(road)
                self.finished = not self.remaining
                return None, -cost, self.finished

            def result(self):
                return dict(objective=self.total_loss, order=self.order.copy(), starts=[])

        class Encoder:
            def encode(self, observation):
                return dict(roads=list(observation), cand=tuple(range(len(observation))))

        class Policy:
            def __init__(self):
                self.net = SimpleNamespace(load_state_dict=lambda *a, **k: None, eval=lambda: None)

            def __call__(self, state):
                return torch.arange(len(state['roads']), 0, -1)

        scenario = SimpleNamespace(scenario_id='train_000', signature='frozen', roads=tuple(range(6)))
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(workers, 'W', (SimpleNamespace(fingerprint='test'), Encoder(), Policy())), \
                patch.object(workers, 'DemandMemoryObservedEpisode', Episode):
            request = (0, scenario, {}, 'weights', Path(tmp), None)
            result = workers.collect(request)
            self.assertEqual([r['decision'] for r in result['rows']], list(range(5)))
            self.assertEqual([len(r['costs']) for r in result['rows']], [6, 5, 4, 3, 2])
            self.assertEqual(result['branch_replay_max_error'], 0.)
            for row in result['rows']:
                for road, branch in zip(row['state']['roads'], row['branches']):
                    position = row['decision']
                    self.assertEqual(branch['order'][position], road)
                    self.assertEqual(branch['order'][position+1:], sorted(set(row['state']['roads']) - {road}))
            with patch.object(workers, 'DemandMemoryObservedEpisode', side_effect=AssertionError('repeated simulation')):
                reused = workers.collect(request)
            equal_tree(result, reused)

    def test_reward_includes_opening_day_and_preserves_parent_history(self):
        class Simulator:
            fingerprint = "synthetic"

            def solve(self, day, damage, behavior, previous):
                previous.history.append(day)
                return SimpleNamespace(key=str(day), cost=10. if damage else 6., history=previous.history.copy())

        episode = SimpleNamespace(day=1, simulator=Simulator(), behavior="response7d", completed=set(),
            restoration=SimpleNamespace(truth={1: SimpleNamespace(true_duration=2, true_severity=3)}),
            _candidates=lambda: (1,),
            traffic=SimpleNamespace(key="start", cost=10., history=[0]))
        before = pickle.dumps(episode.traffic)
        with patch.object(workers, "day_cost", side_effect=lambda sim, traffic: traffic.cost):
            result = workers.benefit(episode, 1, horizon=5, cache=False)
        self.assertEqual(result["reward"], 12.)
        self.assertEqual([row["day"] for row in result["daily"]], [3, 4, 5])
        self.assertEqual(pickle.dumps(episode.traffic), before)
        with patch.object(episode.simulator, "solve") as solve:
            self.assertEqual(workers.benefit(episode, 1, horizon=2, cache=False)["reward"], 0.)
            solve.assert_not_called()

    def test_full_entry_uses_new_pipeline_and_legacy_contrasts_are_explicit(self):
        from src.experiment.daily import run
        with patch("src.experiment.daily_rl.run", return_value="ok") as standard:
            self.assertEqual(run(11), "ok")
            settings = standard.call_args.kwargs["improvement_settings"]
            self.assertEqual(settings.batch_states, 8)
            self.assertEqual(settings.outer_patience, 5)
            self.assertEqual(settings.learning_rate, .001)
            with self.assertRaises(ValueError):
                run(11, traffic_input=False)


class PipelineChecks(unittest.TestCase):
    def test_complete_pipeline_test_handoff_and_completed_run_reuse(self):
        from src.experiment import daily_rl as runner

        class World:
            def __init__(self, name):
                self.scenario_id = name

            def record(self):
                return dict(scenario_id=self.scenario_id)

        splits = {name: [World(name + '_000')] for name in ('train', 'validation', 'test')}
        seen = []
        test_root = None

        class Pool:
            def __init__(self, **kwargs):
                seen.append(('pool', kwargs['initargs'][2]))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def map(self, function, requests):
                for request in requests:
                    if function is workers.collect:
                        self_scenario = request[1]
                        assert self_scenario.scenario_id.startswith('train_')
                        yield dict(rows=[dict(state=copy.deepcopy(STATE), costs=np.array([1., 2.]))])
                    else:
                        scenario, weights, epsilon, seed, reward_training = request
                        seen.append((scenario.scenario_id, reward_training))
                        if scenario.scenario_id.startswith('test_'):
                            assert (test_root / 'results/policy_selection.json').is_file()
                            assert not reward_training
                        yield dict(states=[copy.deepcopy(STATE)], picks=[0], rewards=[-1.],
                            result=dict(scenario=scenario.scenario_id, objective=1., time_loss=.5, flow_loss=.5,
                                recovery_day=3, repair_completion_day=2, order=[1], daily=[], starts=[], completions=[]),
                            runtime=dict(seconds=0., ue_solves=0, cached_days=0))

        def base_train(encoder, batch, **kwargs):
            self.assertEqual(kwargs['reward_description'], runner.BASE_REWARD)
            self.assertEqual(kwargs['hp']['lr'], .0005)
            model = DailyQ(encoder, kwargs['hp'])
            batch('train', [0], model.net.state_dict(), 1., [42])
            batch('validation', [0], model.net.state_dict(), 0., [0])
            with self.assertRaises(ValueError):
                batch('test', [0], model.net.state_dict(), 0., [0])
            directory = kwargs['run_dir']
            for name in ('model_best_validation.pt', 'model_final.pt', 'training_latest.pt'):
                atomic_torch(directory / 'results' / name, model.net.state_dict())
            freeze_json(directory / 'results/policy_selection.json', dict(validation_objective=1.))
            freeze_json(directory / 'results/training_summary.json', dict(episodes=1))
            freeze_json(directory / 'config/training.json', kwargs['identity'])
            return model, {}

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            test_root = folder / 'data/methods/rl_s2v_dual_od16_memory_horizon_paired_train_natural_seed42'
            sim = SimpleNamespace(ctx={}, gamma=.01, close=lambda: None)
            with patch.object(runner, 'prepare', return_value=({}, splits, sim, None, folder, {'test': True})), \
                    patch.object(runner, 'JunctionEncoder', return_value=TinyEncoder()), \
                    patch.object(runner, 'save_problem'), patch.object(runner, 'improvement_splits', return_value=splits), \
                    patch.object(runner, '_source_digest', return_value=('synthetic', {})), \
                    patch.object(runner, '_source_files', return_value=[]), \
                    patch.object(runner, 'ProcessPoolExecutor', Pool), patch.object(runner, 'train', side_effect=base_train) as train_call:
                settings = ImprovementSettings(outer_patience=1, inner_patience=1,
                    validation_updates=(1,), validation_interval=1)
                runner.run(training_behavior='natural', improvement_settings=settings)
                self.assertIn(('train_000', True), seen)
                self.assertIn(('test_000', False), seen)
                self.assertEqual([item for item in seen if item[0] == 'pool'], [('pool', 'natural'), ('pool', 'response7d')])
                selection = json.loads((test_root / 'results/policy_selection.json').read_text())
                self.assertEqual(selection['selected_round'], 0)
                self.assertFalse(selection['test_used_for_selection'])
                self.assertTrue((test_root / 'results/training_summary.json').is_file())
                seen.clear()
                runner.run(training_behavior='natural', improvement_settings=settings)
                self.assertEqual(train_call.call_count, 1)
                self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
