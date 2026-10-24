"""One dual-OD Full RL and one GA10 for an equal-probability n mixture.

Run python -X utf8 -u -m src.experiment.daily_mixed. No existing experiment is
reused for fitting or overwritten. Validation chooses RL weights, not test.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from src.experiment.daily import prepare, save_problem, _write_json, _source_digest, _source_files, _save_evaluation
from src.experiment.daily_ga import single_runner
from src.experiment.daily_ga_saa10 import run as run_ga, training_subset
from src.experiment.daily_junction import initialize_worker, run_world, verify_saved_inputs
from src.experiment.daily_dual_junction import freeze
from src.experiment.layout import ROOT
from src.methods.daily_junction import JunctionEncoder
from src.methods.daily_s2v import DailyQ, train, POLICY_DELIVERY
from src.methods.rl_rank import EP_CAP, STOP_PARAMS
from src.methods.daily_ga import possible_roads
from src.problem.mixed_daily import SCALES
from src.experiment.retired import retired_problem

METHOD = 'rl_s2v_dual_od16_memory_g1_noimit_seed42'
GA = 'ga_saa10_flow_insert_seed42'


def count_scales(scenarios):
    counts = Counter(len(s.roads) for s in scenarios)
    return {str(n): counts[n] for n in SCALES}


def protected_results():
    result = {}
    for experiment in (ROOT/'outputs/experiments').glob('*'):
        if experiment.name.startswith('nmixed_'):
            continue
        paths = list(experiment.glob('data/methods/*/results/test_results.csv'))
        paths += list(experiment.glob('data/scenarios/*.json'))
        paths += list(experiment.glob('data/methods/*/config/training.json'))
        for path in paths:
            result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def setup():
    problem, splits, simulator, plain, folder, identity = prepare('mixed')
    encoder = JunctionEncoder(problem, simulator.ctx, simulator.gamma, demand_memory=True, bidirectional=True)
    simulator.close()
    return problem, splits, encoder, folder, identity


def freeze_plan(problem, splits, encoder, folder, identity):
    save_problem(folder, problem, splits, identity, encoder)
    source, hashes = _source_digest()
    indices, selected = training_subset(splits['train'], 42)
    road_union = sorted({r.edge_id for s in selected for r in s.roads})
    hp = DailyQ(encoder).hp
    assert hp['gamma'] == 1 and hp['lam'] == 0
    plan = dict(source_sha256=source, experiment=identity, seed=42, workers=4,
        method=METHOD, ga_method=GA, hyperparameters=hp, stop=STOP_PARAMS, ep_cap=EP_CAP,
        feature_schema=encoder.specification(), delivery=POLICY_DELIVERY,
        counts_by_scale={key: count_scales(values) for key, values in splits.items()},
        ga_training_indices=indices, ga_counts_by_scale=count_scales(selected),
        ga_training_road_union=road_union, possible_roads=possible_roads(problem),
        ga_unseen_possible_roads=sorted(set(possible_roads(problem))-set(road_union)),
        true_n_input=False, scenario_mean='equal weight per sampled scenario; n drawn independently with probability 1/4',
        execution='RL training, validation-selected test, then GA10 search and same-test comparison',
        protected_previous_results=protected_results(),
        figure=dict(backend='Python/matplotlib', format='PNG', title=None,
            question='compare two jointly trained mixed-scale policies without presuming improvement',
            evidence='same 50 held-out scenarios, paired differences and per-scale summaries',
            appearance='existing teal GA/purple RL boxplot, all 50 points, mean and median'))
    assert all(all(v>0 for v in c.values()) for c in plan['counts_by_scale'].values())
    assert all(v>0 for v in plan['ga_counts_by_scale'].values())
    freeze(folder/'data/analysis/config/mixed_plan.json', plan)
    root = folder/'data/methods'/METHOD
    for part in ('log','config','results'):
        (root/part).mkdir(parents=True,exist_ok=True)
    freeze(root/'config/source_hashes.json', hashes)
    archive=root/'config/source_snapshot.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for path in _source_files():
                z.write(path,path.relative_to(ROOT))
    return plan, root


def train_rl(problem, splits, encoder, folder, identity, plan, root):
    status=root/'log/status.json'
    if status.exists() and json.loads(status.read_text()).get('status')=='complete':
        verify_saved_inputs(folder,root,encoder)
        return
    with single_runner(root/'log/runner.lock'):
        # Memory-only traffic cache avoids both disk growth and concurrent WAL
        # initialization. Scientific simulator tolerances remain unchanged.
        with ProcessPoolExecutor(max_workers=4,initializer=initialize_worker,
                initargs=(problem,None,plan['hyperparameters'],True,True)) as pool:
            def batch(split,indices,weights,epsilon,seeds):
                if split not in ('train','validation'):
                    raise ValueError('test data cannot enter training')
                return list(pool.map(run_world,[(splits[split][i],weights,epsilon,seed)
                                                for i,seed in zip(indices,seeds)]))
            summary_path=root/'results/training_summary.json'
            if summary_path.exists():
                policy=DailyQ(encoder,plan['hyperparameters'])
                saved=torch.load(root/'results/model_best_validation.pt',map_location='cpu',weights_only=False)
                policy.net.load_state_dict(saved)
                training=json.loads(summary_path.read_text())
            else:
                _write_json(status,dict(status='training',pid=os.getpid()))
                policy,training=train(encoder,batch,train_count=64,validation_count=11,seed=42,
                    run_dir=root,hp=plan['hyperparameters'],stop_params=plan['stop'],ep_cap=plan['ep_cap'],
                    identity=dict(experiment=identity,training_behavior='response7d',
                                  input_variant='dual_junction_od16_demand_memory',code_sha256=plan['source_sha256']))
            selection=json.loads((root/'results/policy_selection.json').read_text())
            assert selection['model_file']=='model_best_validation.pt' and not selection['test_used_for_selection']
            trajectory_dir=root/'results/test_trajectories'
            trajectory_dir.mkdir(exist_ok=True)
            trajectories=[]
            timing=root/'results/test_wall_time.json'
            before=json.loads(timing.read_text())['seconds'] if timing.exists() else 0.
            started=time.perf_counter()
            pending=[s for s in splits['test'] if not (trajectory_dir/(s.scenario_id+'.pkl')).exists()]
            _write_json(status,dict(status='testing',completed=50-len(pending),total=50,pid=os.getpid(),**training))
            for trajectory in pool.map(run_world,[(s,policy.net.state_dict(),0.,0) for s in pending]):
                path=trajectory_dir/(trajectory['result']['scenario']+'.pkl')
                temporary=path.with_suffix('.tmp')
                temporary.write_bytes(pickle.dumps(trajectory,protocol=pickle.HIGHEST_PROTOCOL))
                temporary.replace(path)
                _write_json(timing,dict(seconds=before+time.perf_counter()-started))
                _write_json(status,dict(status='testing',completed=len(list(trajectory_dir.glob('*.pkl'))),total=50,pid=os.getpid()))
            trajectories=[pickle.loads((trajectory_dir/(s.scenario_id+'.pkl')).read_bytes()) for s in splits['test']]
            seconds=json.loads(timing.read_text())['seconds']
            evaluation=_save_evaluation(root,trajectories,seconds)
            np.savez_compressed(root/'results/test_junction_inputs.npz',
                **{f'scenario_{i}_od':np.stack([s['od'] for s in t['states']]) for i,t in enumerate(trajectories)})
            verify_saved_inputs(folder,root,encoder)
            _write_json(status,{**training,**evaluation,'status':'complete'})


def finish(folder,splits,root,plan):
    source,_=_source_digest()
    assert source==plan['source_sha256']
    assert protected_results()==plan['protected_previous_results']
    ga=folder/'data/methods'/GA
    pair=pd.read_csv(folder/'data/analysis/results/ga_saa10_vs_full_rl_paired.csv')
    size={s.scenario_id:len(s.roads) for s in splits['test']}
    assert set(pair.scenario)==set(size) and len(pair)==50
    pair['damaged_road_count']=pair.scenario.map(size)
    rows=[]
    for n in SCALES:
        group=pair[pair.damaged_road_count==n]
        delta=group.ga_minus_rl.to_numpy()
        draws=np.random.default_rng(42+n).choice(delta,(5000,len(delta)),replace=True).mean(axis=1)
        rows.append(dict(n=n,count=len(group),ga_mean=float(group.ga_saa10.mean()),
            rl_mean=float(group.full_rl.mean()),ga_median=float(group.ga_saa10.median()),
            rl_median=float(group.full_rl.median()),ga_minus_rl=float(delta.mean()),
            rl_reduction_percent=float(100*delta.mean()/group.ga_saa10.mean()),
            rl_wins=int((delta>1e-9).sum()),paired_95_interval=np.quantile(draws,[.025,.975]).tolist()))
    pair.to_csv(folder/'data/analysis/results/mixed_paired_by_scale.csv',index=False)
    comparison=json.loads((folder/'data/analysis/results/ga_saa10_vs_full_rl.json').read_text())
    comparison.update(per_scale=rows,saved_inputs_verified=True,baseline_files_unchanged=True,
        scale_sampling=plan['counts_by_scale'],ga_training_scale_counts=plan['ga_counts_by_scale'],
        equal_scale_descriptive_means=dict(ga=float(np.mean([r['ga_mean'] for r in rows])),
                                          rl=float(np.mean([r['rl_mean'] for r in rows]))),
        rl_training=json.loads((root/'results/training_summary.json').read_text()),
        rl_evaluation=json.loads((root/'results/evaluation_summary.json').read_text()),
        ga_training=json.loads((ga/'results/training_summary.json').read_text()),
        ga_evaluation=json.loads((ga/'results/evaluation_summary.json').read_text()),
        limitation='one training/search seed; empirical scale counts differ under equal-probability sampling; raw unnormalized transport loss')
    _write_json(folder/'data/analysis/results/mixed_comparison.json',comparison)
    return comparison


@retired_problem
def main(prepare_only=False):
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    torch.set_num_threads(1)
    problem,splits,encoder,folder,identity=setup()
    plan,root=freeze_plan(problem,splits,encoder,folder,identity)
    print(json.dumps(dict(experiment=str(folder),counts=plan['counts_by_scale'],ga_counts=plan['ga_counts_by_scale'],
                          ga_unseen=plan['ga_unseen_possible_roads'],source_sha256=plan['source_sha256'])),flush=True)
    if prepare_only:
        return
    audit_path=folder/'data/analysis/config/mixed_audit.json'
    if not audit_path.exists():
        raise RuntimeError('run the mixed-experiment preflight audit before training')
    audit=json.loads(audit_path.read_text())
    if not audit['passed'] or audit['source_sha256']!=plan['source_sha256']:
        raise RuntimeError('preflight audit does not match frozen sources')
    overall=folder/'data/log/mixed_sequence_status.json'
    overall.parent.mkdir(parents=True,exist_ok=True)
    with single_runner(overall.with_suffix('.lock')):
        if overall.exists() and json.loads(overall.read_text()).get('status')=='complete':
            return
        try:
            _write_json(overall,dict(status='running',stage='full_rl',pid=os.getpid()))
            train_rl(problem,splits,encoder,folder,identity,plan,root)
            assert _source_digest()[0]==plan['source_sha256']
            _write_json(overall,dict(status='running',stage='ga10',pid=os.getpid()))
            run_ga(problem_variant='poi_mixed_n',full_rl_method=METHOD)
            result=finish(folder,splits,root,plan)
            _write_json(overall,dict(status='complete',result=result))
        except BaseException as error:
            _write_json(overall,dict(status='failed',error_type=type(error).__name__,message=str(error),pid=os.getpid()))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    main(parser.parse_args().prepare_only)
