"""Results relocation checks without training, HPC access, or full simulation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import copy,unittest
from src import config
from src.analysis import results

class ResultsChecks(unittest.TestCase):
    def test_paths_portable_and_bounded(self):
        with TemporaryDirectory() as t:
            self.assertEqual(results.child(t,'inputs\\weights.pt'),Path(t)/'inputs/weights.pt')
            for rel in ('../escape','..\\escape','C:\\elsewhere','/tmp/escape'):
                with self.assertRaises(ValueError):results.child(t,rel)

    def test_profile_does_not_change_defaults(self):
        before=copy.deepcopy(config.DEFAULT_EXPERIMENT)
        p=config.results_profile('behavior_ranking');p['protocol']['worlds']=1
        self.assertEqual(config.results_profile('behavior_ranking')['protocol']['worlds'],100)
        self.assertEqual(config.DEFAULT_EXPERIMENT,before)
        with self.assertRaises(ValueError):config.results_profile('latest')

    def test_changed_input_is_rejected(self):
        with TemporaryDirectory() as t:
            p=Path(t)/'weights.pt';p.write_bytes(b'original');h=results.sha(p)
            self.assertEqual(results.verify_files(t,{'weights.pt':h}),1)
            p.write_bytes(b'other model')
            with self.assertRaises(ValueError):results.verify_files(t,{'weights.pt':h})

    def test_existing_directory_cannot_be_rerun_target(self):
        with TemporaryDirectory() as t,patch.object(results,'verify',return_value={}):
            with self.assertRaises(FileExistsError):results.prepare('behavior_transfer',t)
            self.assertEqual(list(Path(t).iterdir()),[])

    def test_current_results_scientific_records(self):
        for name in config.results_profiles()['profiles']:
            with self.subTest(profile=name):self.assertTrue(results.verify(name)['verified'])

if __name__=='__main__':unittest.main()
