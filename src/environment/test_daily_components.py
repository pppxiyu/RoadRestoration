"""Component contracts for the daily environment. Run with unittest."""
import ast
from dataclasses import replace
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from src.environment.daily import DailyEpisode, DailyTraffic, Observation, TrafficState
from src.environment.daily_cache import TrafficCache
from src.environment.daily_demand import DemandResponse
from src.environment.daily_observation import build_observation
from src.environment.daily_performance import RecoveryMonitor
from src.environment.daily_restoration import RestorationState


def rules():
    return SimpleNamespace(recovery_days=35, response_95_days=7, time_weight=.5,
                           flow_weight=.5, restored_flow_fraction=.99,
                           restored_time_ratio=1.01, consecutive_restored_days=2)


def traffic(day=1, amount=5., reachable=True):
    return TrafficState('state', day, np.array([amount]), np.array([10.]), np.array([2.]),
                        np.array([reachable]), np.array([amount]), np.array([.2]),
                        pd.DataFrame({'from':[1], 'to':[2], 'volume':[amount], 'cost':[2.]}),
                        {'fixed_point_residual':0.},
                        dict(path_arcs=[(0,)],path_od=np.array([0]),path_flow=np.array([amount]),damage=[[4,1]]),
                        np.array([.5]),np.array([.02]))


class Components(unittest.TestCase):
    def test_repairs_finish_on_boundary_and_reports_do_not_interrupt(self):
        roads=(SimpleNamespace(edge_id=1,true_severity=3,true_duration=2,discovery_day=0),
               SimpleNamespace(edge_id=2,true_severity=1,true_duration=1,discovery_day=2))
        scenario=SimpleNamespace(roads=roads,visible=lambda day:tuple(r for r in roads if r.discovery_day<=day))
        state=RestorationState(scenario)
        self.assertEqual(state.candidates(),(1,))
        state.begin(1)
        state.advance_day()
        self.assertEqual(state.damage(),{1:3,2:1})
        self.assertEqual(state.candidates(),(2,))
        with self.assertRaises(ValueError): state.begin(2)
        state.advance_day()
        self.assertEqual(state.completed,{1:3})
        self.assertEqual(state.damage(),{2:1})
        state.begin(2)
        state.advance_day()
        self.assertEqual(state.completed,{1:3,2:4})

    def test_public_projection_copies_arrays_and_hides_behavior_memory(self):
        state=traffic(reachable=False)
        visible=(SimpleNamespace(edge_id=4,estimated_severity=1,estimated_duration=2),)
        edges=pd.DataFrame({'u':[1], 'v':[2]})
        observed=build_observation(1,visible,{},(4,),state,edges)
        self.assertIsInstance(observed,Observation)
        self.assertEqual(observed.demand[0],0.)
        self.assertTrue(np.isinf(observed.costs[0]))
        self.assertEqual(observed.link_traffic[1].flow,0.)
        self.assertTrue(np.isinf(observed.link_traffic[1].travel_time))
        self.assertFalse(hasattr(observed,'cumulative_unmet'))
        self.assertFalse(hasattr(observed,'gamma_od'))
        observed.external[0]=99
        observed.road_flow[0]=99
        self.assertEqual(state.external[0],10.)
        self.assertEqual(state.road_flow[0],5.)

    def test_demand_history_is_supplied_per_scenario_and_updated_once(self):
        spec=dict(counts=[[1]],zone_ids=[2],rates=[.1],floor=.5)
        model=DemandResponse(np.array([10.]),[(1,2)],rules(),.02,spec)
        previous=traffic()
        more_unmet=replace(previous,cumulative_unmet=np.array([5.]))
        a=model.prepare(2,'response7d',previous)
        b=model.prepare(2,'response7d',more_unmet)
        self.assertLess(b.gamma[0],a.gamma[0])
        self.assertEqual(previous.cumulative_unmet[0],.5)
        self.assertAlmostEqual(model.prepare(1,'response7d',None).gamma[0],.02)
        eq=model.equation(np.array([8.]),np.array([2.]),previous,a.gamma)
        expected=8+(1-model.weight)*(5-10)+model.weight*8*np.expm1(-a.gamma*(3-2))
        np.testing.assert_array_equal(eq.target(np.array([3.])),expected)
        np.testing.assert_array_equal(model.advance_unmet(np.array([0.]),a.unmet_before),[1.5])
        self.assertEqual(previous.cumulative_unmet[0],.5)
        with self.assertRaises(ValueError): model.prepare(3,'response7d',previous)
        self.assertIsNone(model.prepare(3,'natural',previous).unmet_before)

    def test_recovery_requires_consecutive_mobility_and_physical_recovery(self):
        monitor=RecoveryMonitor(rules(),dict(H0=np.array([10.]),baseline_u=np.array([2.])), 'unserved')
        for day,(served,repaired) in enumerate(((10,0),(10,1),(5,1),(10,1),(10,1)),1):
            monitor.record(day,traffic(day,served),repaired=repaired,total_damaged=1,discovered=1)
            self.assertEqual(monitor.finished,day==5)
        self.assertEqual(monitor.restored_streak,2)
        self.assertEqual(monitor.total_loss,.25)
        self.assertEqual([r['day'] for r in monitor.records],[1,2,3,4,5])

    def test_cache_keeps_memory_and_paths_and_can_read_when_write_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=TrafficCache('contract',directory)
            state=traffic()
            cache.put(state.key,state)
            cache.memo.clear()
            restored=cache.get(state.key,9)
            self.assertEqual(restored.day,9)
            for field in ('demand','external','costs','reachable','road_flow','road_congestion','cumulative_unmet','gamma_od'):
                np.testing.assert_array_equal(getattr(restored,field),getattr(state,field))
            self.assertEqual(restored.solver_state['path_arcs'],[(0,)])
            np.testing.assert_array_equal(restored.solver_state['path_flow'],[5.])
            cache.limit_bytes=0
            cache.put('not_stored',replace(state,key='not_stored'))
            cache.memo.clear()
            self.assertIsNone(cache.get('not_stored',1))
            self.assertIsNotNone(cache.get('state',1))
            cache.close()
            cache.close()

    def test_runtime_import_does_not_load_other_business_modules(self):
        command = (
            "import sys; import src.environment.daily; "
            "assert not any(n.startswith(('src.experiment','src.methods','src.analysis','src.problem')) "
            "for n in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", command], check=True, timeout=30,
                       cwd=Path(__file__).resolve().parents[2])

    def test_components_have_no_back_dependency_on_coordinator(self):
        folder=Path(__file__).parent
        leaves=('daily_contracts','daily_demand','daily_restoration','daily_observation',
                'daily_performance','daily_cache','daily_context','traffic_network','gravity_model')
        forbidden=('src.experiment','src.methods','src.analysis','src.environment.daily_episode','src.environment.daily')
        for name in leaves:
            parsed=ast.parse((folder/(name+'.py')).read_text(encoding='utf-8'))
            imports=[n.module for n in ast.walk(parsed) if isinstance(n,ast.ImportFrom) and n.module]
            for imported in imports:
                self.assertFalse(any(imported==p or imported.startswith(p+'.') for p in forbidden),
                                 f'{name} depends on {imported}')
        from src.environment.daily_episode import DailyEpisode as Coordinator
        from src.environment.daily_traffic import DailyTraffic as Solver
        self.assertIs(DailyEpisode,Coordinator)
        self.assertIs(DailyTraffic,Solver)


if __name__=='__main__':
    unittest.main()
