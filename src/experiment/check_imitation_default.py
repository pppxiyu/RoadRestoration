"""Small checks for the prospective zero-imitation default; no experiment runs."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.methods.rl_rank import RANK_PARAMS_DQN, _nstep_rows
from src.methods.rl_s2v import S2V_PARAMS
from src.methods.rl_s2v_saa import S2V_SAA_PARAMS
from src.methods.daily_s2v import DailyQ, train
from src.experiment.check_policy_selection import TinyEncoder, make_batch


class ImitationDefaultChecks(unittest.TestCase):
    def test_undiscounted_nstep_rewards_and_bootstrap(self):
        rows=_nstep_rows(['s0','s1','s2'],[0,0,0],[-2.,-3.,-5.],2,S2V_SAA_PARAMS['gamma'])
        self.assertEqual([row[2] for row in rows],[-5.,-8.,-5.])
        self.assertEqual(rows[0][3: ],('s2',1.0))
        self.assertEqual(rows[1][3: ],(None,0.0))

    def test_default_propagates_and_ramp_cannot_reenable_it(self):
        for hp in (RANK_PARAMS_DQN, S2V_PARAMS, S2V_SAA_PARAMS, DailyQ(TinyEncoder()).hp):
            self.assertEqual(hp['lam'], 0.0)
            self.assertEqual(hp['gamma'], 1.0)
            for episode in (0, 150, 200, 4999):
                weight=min(hp['lam'], hp['lam0']*hp['lam_growth']**max(0,episode-1))
                self.assertEqual(weight, 0.0)

    def test_explicit_historical_configuration_still_supported(self):
        model=DailyQ(TinyEncoder(), hp={'lam':1.0,'lam0':0.35,'lam_growth':1.00707,'gamma':0.9995})
        self.assertEqual(model.hp['lam'],1.0)
        self.assertEqual(model.hp['gamma'],0.9995)
        self.assertEqual(S2V_SAA_PARAMS['gamma'],1.0)
        self.assertEqual(S2V_SAA_PARAMS['lam'],0.0)

    def test_default_training_records_zero_weight_and_td_only_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            batch,_=make_batch([3.,1.,2.])
            train(TinyEncoder(),batch,train_count=1,validation_count=1,seed=42,
                run_dir=root,identity={'synthetic_imitation_default_check':True},
                hp=dict(pool_n=1,batch_worlds=1,updates_per_ep=1,batch=1,p=8,t_emb=1,readout_hidden=8),
                stop_params=dict(n_val=1,ep_min=99,probe_every=1),ep_cap=3)
            records=pd.read_csv(root/'log/training.csv')
            self.assertTrue((records.hinge_weight==0).all())
            self.assertTrue((records.hinge_loss==0).all())
            self.assertTrue((records.loss==records.td_loss).all())
            config=json.loads((root/'config/training.json').read_text())
            self.assertEqual(config['hyperparameters']['lam'],0.0)
            self.assertEqual(config['hyperparameters']['gamma'],1.0)
            selection=json.loads((root/'results/policy_selection.json').read_text())
            self.assertEqual(selection['selected_episode'],1)
            self.assertFalse(selection['test_used_for_selection'])


if __name__=='__main__':
    unittest.main()
