"""Five-branch finalization checks, without policy training or test rollouts."""
from collections import Counter
from contextlib import redirect_stderr
import importlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from src.problem.daily import (
    BRANCHES, CURRENT_PROBLEM, PROBLEM_VARIANTS, build_problem,
    improvement_splits, make_splits, public_spec, sample_split,
)
from src.problem.mixed_daily import build_mixed_problem


class FinalizedProblemChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mixed = build_mixed_problem()
        cls.problems = {**cls.mixed['components'], 'mixed': cls.mixed}

    def test_five_branches_share_full_behavior_and_public60(self):
        self.assertEqual(BRANCHES, ('6', '11', '17', '23', 'mixed'))
        self.assertEqual(PROBLEM_VARIANTS, (CURRENT_PROBLEM,))
        for n, problem in self.problems.items():
            with self.subTest(n=n):
                spec = public_spec(problem)
                self.assertEqual(spec['variant'], CURRENT_PROBLEM)
                self.assertEqual(spec['rules']['version'], CURRENT_PROBLEM)
                self.assertEqual(spec['human_response'], self.mixed['human_response'])
                self.assertEqual(problem['rules'], self.mixed['rules'])
                self.assertNotIn('duration_sd', problem)
                if n != 'mixed':
                    self.assertEqual(len(problem['public_roads']), int(.6 * n + .5))

    def test_four_blocks_preserve_base_and_evaluation_worlds(self):
        for n, problem in self.problems.items():
            with self.subTest(n=n):
                base = make_splits(problem)
                expanded = improvement_splits(problem, base)
                self.assertEqual(tuple(map(len, base.values())), (64, 11, 50))
                self.assertEqual(tuple(map(len, expanded.values())), (256, 11, 50))
                self.assertEqual(expanded['train'][:64], base['train'])
                self.assertIs(expanded['validation'], base['validation'])
                self.assertIs(expanded['test'], base['test'])
                self.assertEqual(len({s.signature for group in expanded.values() for s in group}), 317)
                self.assertEqual([s.scenario_id for s in expanded['train']],
                                 [f'train_{i:03d}' for i in range(256)])
                for block in (1, 2, 3):
                    expected = sample_split(problem, 64, 42021 + 100 * block, 'extension')
                    self.assertEqual([s.roads for s in expanded['train'][64*block:64*(block+1)]],
                                     [s.roads for s in expected])
                if n == 'mixed':
                    self.assertEqual(Counter(len(s.roads) for s in base['train']), {6: 20, 11: 13, 17: 15, 23: 16})
                    self.assertEqual(Counter(len(s.roads) for s in expanded['train']), {6: 73, 11: 61, 17: 60, 23: 62})
                bad = dict(base, train=base['train'][:-1])
                with self.assertRaisesRegex(ValueError, '64-world'):
                    improvement_splits(problem, bad)

    def test_retired_variants_rejected_before_traffic_or_io(self):
        from src.experiment.daily import prepare
        with patch('src.problem.daily.load_toy_network') as load:
            for old in ('current', 'poi_unmet', 'poi_hidden6', 'public75_duration25', 'poi_mixed_n'):
                for entry in (build_problem, prepare):
                    kwargs = {'variant': old} if entry is build_problem else {'problem_variant': old}
                    with self.subTest(old=old, entry=entry.__name__), self.assertRaises(ValueError):
                        entry(**kwargs)
            load.assert_not_called()

    def test_new_results_namespace_and_prepare_for_every_branch(self):
        from src.experiment.daily import prepare
        from src.experiment.layout import ROOT, study_dir
        for branch in BRANCHES:
            with patch('src.experiment.daily.build_problem', side_effect=lambda n, *a, **k: self.problems[n]), \
                 patch('src.problem.mixed_daily.build_mixed_problem', return_value=self.mixed):
                problem, splits, simulator, encoder, folder, identity = prepare(branch)
                try:
                    self.assertTrue(folder.is_relative_to(ROOT / 'outputs/current/experiments'))
                    self.assertEqual(identity['problem']['variant'], CURRENT_PROBLEM)
                    self.assertEqual(len(splits['train']), 64)
                finally:
                    simulator.close()
        self.assertEqual(study_dir('human_behavior'), ROOT / 'outputs/current/studies/human_behavior')


class FinalizedEntryChecks(unittest.TestCase):
    def test_branch_required_and_legacy_cli_options_rejected(self):
        from main import daily_cli
        with patch('src.experiment.daily.run') as run:
            for argv in ([], ['--n', '10'], ['--n', '11', '--problem-variant', 'poi_hidden6'],
                         ['--n', '11', '--solve', 'ga_nominal_public'], ['--n', '11', '--workers', '0']):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    daily_cli(argv)
            run.assert_not_called()

    def test_five_cli_branches_dispatch_current_without_running(self):
        from main import daily_cli
        with patch('src.experiment.daily.run') as run:
            for branch in BRANCHES:
                run.reset_mock()
                daily_cli(['--n', branch])
                run.assert_called_once_with('mixed' if branch == 'mixed' else int(branch),
                                            'response7d', 42, 4, road_class_input=True,
                                            problem_variant=CURRENT_PROBLEM)

    def test_retired_entrypoints_fail_immediately(self):
        entries = (
            ('daily_dual_junction', 'main'), ('daily_junction', 'main'),
            ('daily_mixed', 'main'), ('daily_od_history', 'main'),
            ('daily_ga_nominal', 'run'), ('daily_ga_warmstart', 'run'),
            ('daily_ga_warmstart_no_traffic', 'run'),
            ('check_poi_response', 'audit_and_freeze'),
            ('check_duration_variant', 'audit_and_freeze'), ('check_junction', 'main'),
            ('check_od_history', 'main'),
        )
        for module, name in entries:
            with self.subTest(module=module), self.assertRaisesRegex(RuntimeError, 'retired problem'):
                getattr(importlib.import_module('src.experiment.' + module), name)()

    def test_frozen_calibration_loads_and_cannot_be_refitted_in_place(self):
        from src.environment import gravity
        model = gravity.load_gravity_model()
        self.assertEqual(Path(model['path']), gravity.DEFAULT_MODEL)
        with patch.object(gravity, 'fit_gravity') as fit:
            for operation in (lambda: gravity.fit_and_save_gravity_model(gravity.DEFAULT_MODEL),
                              lambda: gravity.ensure_gravity_model(gravity.DEFAULT_MODEL, force=True)):
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    operation()
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
