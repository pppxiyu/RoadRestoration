"""Mixed-scale sampling, input, environment and Windows-worker preflight."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
import unittest
import numpy as np
import torch

from src.problem.daily import make_splits, public_spec, build_problem
from src.problem.mixed_daily import build_mixed_problem, SCALES, PROBABILITIES
from src.methods.daily_ga import possible_roads
from src.methods.daily_junction import JunctionEncoder
from src.methods.daily_s2v import DailyQ
from src.environment.daily import DailyTraffic, DailyEpisode
from src.experiment.daily import _source_digest, _write_json
from src.experiment.daily_ga_saa10 import training_subset
from src.experiment.daily_junction import initialize_worker, run_world
from src.experiment.daily_mixed import setup, count_scales


class MixedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem=build_mixed_problem()
        cls.splits=make_splits(cls.problem)

    def test_original_per_scale_contracts_unchanged(self):
        for n,p in self.problem['components'].items():
            self.assertEqual(public_spec(p),public_spec(build_problem(n,42,variant='poi_unmet')))
            self.assertEqual(p['rules'],self.problem['rules'])
            self.assertEqual(p['human_response'],self.problem['human_response'])
            self.assertEqual(p['baseline_flow'],self.problem['baseline_flow'])

    def test_sampling_reproducible_all_splits_and_disjoint(self):
        repeated=make_splits(self.problem)
        signatures=[]
        for name,worlds in self.splits.items():
            self.assertEqual([s.record() for s in worlds],[s.record() for s in repeated[name]])
            self.assertEqual(set(len(s.roads) for s in worlds),set(SCALES))
            signatures.extend(s.signature for s in worlds)
        self.assertEqual(len(signatures),125)
        self.assertEqual(len(set(signatures)),125)
        self.assertEqual(PROBABILITIES,(.25,.25,.25,.25))

    def test_all_public_hidden_counts_and_discovery_bounds(self):
        expected={6:(4,2),11:(7,4),17:(10,7),23:(14,9)}
        for worlds in self.splits.values():
            for world in worlds:
                n=len(world.roads)
                public={r.edge_id for r in world.roads if r.discovery_day==0}
                self.assertEqual(public,set(self.problem['components'][n]['public_roads']))
                self.assertEqual((len(public),n-len(public)),expected[n])
                self.assertTrue(all(1<=r.discovery_day<=14 for r in world.roads if r.edge_id not in public))

    def test_ga_ten_worlds_cover_all_scales_and_fixed_indices(self):
        indices,worlds=training_subset(self.splits['train'],42)
        self.assertEqual(indices,[4,5,6,12,25,37,43,51,58,61])
        self.assertEqual(set(len(s.roads) for s in worlds),set(SCALES))
        union={r.edge_id for s in worlds for r in s.roads}
        self.assertTrue(union<=set(possible_roads(self.problem)))
        self.assertTrue(set(self.problem['public_roads'])<=union)

    def test_public_spec_has_per_scale_settings_not_union_public(self):
        spec=public_spec(self.problem)
        self.assertIsNone(spec['public_roads'])
        self.assertEqual(set(spec['scale_settings']),set(map(str,SCALES)))
        self.assertEqual(tuple(spec['count_distribution']['probabilities']),PROBABILITIES)

    def test_encoder_and_defaults_unchanged_except_approved_defaults(self):
        sim=DailyTraffic(self.problem)
        try:
            enc=JunctionEncoder(self.problem,sim.ctx,sim.gamma,demand_memory=True,bidirectional=True)
            model=DailyQ(enc)
            self.assertEqual(model.hp['gamma'],1.)
            self.assertEqual(model.hp['lam'],0.)
            self.assertEqual(sum(p.numel() for p in model.net.parameters()),80992)
            self.assertEqual(enc.node_dimension,48)
            self.assertEqual(enc.global_dimension,7)
        finally:
            sim.close()


def audit():
    torch.set_num_threads(1)
    problem,splits,encoder,folder,identity=setup()
    worlds=[next(s for s in splits['train'] if len(s.roads)==n) for n in SCALES]
    torch.manual_seed(42)
    model=DailyQ(encoder)
    results=[]
    with ProcessPoolExecutor(max_workers=4,initializer=initialize_worker,
            initargs=(problem,None,model.hp,True,True)) as pool:
        trajectories=list(pool.map(run_world,[(s,model.net.state_dict(),0.,0) for s in worlds]))
    for world,trajectory in zip(worlds,trajectories):
        n=len(world.roads)
        sim=DailyTraffic(problem['components'][n],cache_dir=None)
        try:
            episode=DailyEpisode(sim,world,'response7d',disconnected_flow='unserved')
            rewards=[]
            for edge in trajectory['result']['order']:
                self_observation=episode.observation()
                assert edge in self_observation.candidates
                _,reward,_=episode.step(edge)
                rewards.append(reward)
            reference=episode.result()
            assert reference['order']==trajectory['result']['order']
            np.testing.assert_allclose(rewards,trajectory['rewards'],rtol=0,atol=1e-8)
            np.testing.assert_allclose(reference['objective'],trajectory['result']['objective'],rtol=0,atol=1e-8)
            for state in trajectory['states']:
                assert state['x'].shape==(38,16) and state['g'].shape==(7,)
                assert state['od'].shape==(24,24,9)
                assert np.isfinite(state['od']).all()
            results.append(dict(n=n,scenario=world.scenario_id,decisions=len(trajectory['picks']),
                objective=reference['objective'],objective_difference=reference['objective']-trajectory['result']['objective'],
                seconds=trajectory['runtime']['seconds']))
        finally:
            sim.close()
    report=dict(passed=True,source_sha256=_source_digest()[0],training_only_smoke=results,
        counts={key:count_scales(value) for key,value in splits.items()},
        no_test_outcomes_used=True,four_windows_workers_passed=True,
        same_environment_and_reward_per_scale=True,raw_input_dimensions_unchanged=True,
        true_n_not_explicit_input=True,probabilities=PROBABILITIES)
    _write_json(folder/'data/analysis/config/mixed_audit.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    import sys
    if '--audit' in sys.argv:
        audit()
    else:
        unittest.main()
