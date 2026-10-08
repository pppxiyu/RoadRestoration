"""GA adapter contract checks; does not modify official experiment results."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.methods.daily_ga import PriorityPolicy, ReplayFitness, possible_roads
from src.methods.metaheuristic import _ga
from src.experiment.daily_ga import physical_schedule, schedule_key, EvaluationStore, single_runner
from src.experiment.check_daily import ConstantTraffic
from src.environment.daily import DailyEpisode
from src.problem.daily import Scenario, RoadDamage


class Checks(unittest.TestCase):
    def test_policy_only_sees_candidates(self):
        policy = PriorityPolicy((3, 2, 1))
        self.assertEqual(policy.choose((1, 2)), 2)
        with self.assertRaises(ValueError):
            policy.choose((4,))

    def test_possible_universe(self):
        self.assertEqual(possible_roads(dict(public_roads=(2, 5), bins=((1, 2), (3, 4, 5), (6,)),
                                            hidden_quotas=(0, 1))), (2, 3, 4, 5))

    def test_schedule_matches_environment(self):
        rng = random.Random(52)
        for i in range(30):
            roads = tuple(RoadDamage(e, 2, 4, rng.randint(1, 3), rng.randint(1, 5),
                                     0 if e == 1 else rng.randint(1, 14)) for e in range(1, 6))
            world = Scenario(f"schedule_{i}", roads)
            order = rng.sample(range(1, 7), 6)  # includes one unaffected road
            policy = PriorityPolicy(order)
            episode = DailyEpisode(ConstantTraffic(), world, "response7d", disconnected_flow="unserved")
            while not episode.finished:
                episode.step(policy.choose(episode.observation().candidates))
            result = episode.result()
            self.assertEqual(physical_schedule(world, order),
                             tuple((e, result["starts"][e], result["completions"][e]) for e in result["order"]))

    def test_replay_reconstructs_identical_search(self):
        persistent = {}
        def batch(orders):
            for p in orders:
                persistent.setdefault(p, (1.0 + sum((i+1)*e for i, e in enumerate(p)), 0, 0))
            return {p: persistent[p] for p in orders}
        hp = dict(pop_size=8, elite=2, tour_k=3, p_cross=.9, p_mut=.5, max_swaps=2,
                  gen_min=4, stall_K=5, patience_P=7)
        first = ReplayFitness(batch, budget=100)
        stop1, trace1 = _ga(first, list(range(6)), [], random.Random(42), **hp)
        # Preload ALL future scores. They must not enter active fitness early.
        second = ReplayFitness(batch, budget=100)
        stop2, trace2 = _ga(second, list(range(6)), [], random.Random(42), **hp)
        self.assertEqual(trace1, trace2)
        self.assertEqual(stop1.__dict__, stop2.__dict__)
        self.assertEqual(first.best(), second.best())

    def test_resume_after_partial_search(self):
        persistent, calls = {}, [0]
        def score(p):
            return (float(1+sum((i+1)*e for i,e in enumerate(p))), 0., 0.)
        def interrupted(orders):
            for p in orders:
                persistent[p] = score(p)
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("simulated interruption")
            return {p: persistent[p] for p in orders}
        hp = dict(pop_size=8, elite=2, tour_k=3, p_cross=.9, p_mut=.5, max_swaps=2,
                  gen_min=4, stall_K=5, patience_P=7)
        with self.assertRaises(RuntimeError):
            _ga(ReplayFitness(interrupted, 100), list(range(6)), [], random.Random(42), **hp)
        def resume(orders):
            for p in orders:
                persistent.setdefault(p, score(p))
            return {p: persistent[p] for p in orders}
        _, actual = _ga(ReplayFitness(resume, 100), list(range(6)), [], random.Random(42), **hp)
        _, expected = _ga(ReplayFitness(lambda ps: {p: score(p) for p in ps}, 100),
                          list(range(6)), [], random.Random(42), **hp)
        self.assertEqual(actual, expected)

    def test_excludes_test_pool(self):
        with self.assertRaises(ValueError):
            EvaluationStore(Path("not_created"), None, [Scenario(f"test_{i}", ()) for i in range(64)])

    def test_all64_mean_cache_and_restart(self):
        worlds = [Scenario(f"train_{i:03d}", (RoadDamage(1, 1, 1, 1, i+1, 0),)) for i in range(64)]
        calls = []
        def evaluate(request):
            world, order = request
            calls.append(world.scenario_id)
            value = world.roads[0].true_duration
            return dict(result=dict(objective=value, time_loss=value*2, flow_loss=0))
        with tempfile.TemporaryDirectory() as folder, ThreadPoolExecutor(2) as executor:
            run = Path(folder)
            (run / "log").mkdir()
            (run / "results").mkdir()
            with patch("src.experiment.daily_ga.evaluate_priority", evaluate):
                store = EvaluationStore(run, executor, worlds)
                scores = store.score_batch([(1, 2), (2, 1)])
                self.assertEqual(scores[(1, 2)], (32.5, 65., 0.))
                self.assertEqual(scores[(2, 1)], scores[(1, 2)])
                self.assertEqual(len(calls), 64)  # same actual repair sequence
                store.db.close()
                resumed = EvaluationStore(run, executor, worlds)
                self.assertEqual(resumed.score_batch([(1, 2)]), {(1, 2): (32.5, 65., 0.)})
                self.assertEqual(len(calls), 64)
                resumed.db.close()

    def test_single_runner(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock"
            with single_runner(path):
                with self.assertRaises(OSError):
                    with single_runner(path):
                        pass
            with single_runner(path):
                pass


if __name__ == "__main__":
    unittest.main()
