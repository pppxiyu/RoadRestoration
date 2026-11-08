"""Counterfactual repair benefits; actual environment and observations unchanged."""
from pathlib import Path
import copy
import hashlib
import json
import os
import pickle
import time
import numpy as np
import torch
from src.environment.daily import DailyTraffic, daily_losses
from src.environment.daily_history import DemandMemoryObservedEpisode
from src.methods.daily_junction import JunctionEncoder
from src.methods.daily_s2v import DailyQ, rollout

from src.methods.daily_improvement import atomic_pickle, read_pickle

REWARD_CACHE = None
BEHAVIOR = "response7d"
HORIZON = 60
W = None


def initialize(problem, hp, behavior="response7d", reward_cache=None):
    global W, BEHAVIOR, REWARD_CACHE
    BEHAVIOR = behavior
    REWARD_CACHE = Path(reward_cache) if reward_cache is not None else None
    torch.set_num_threads(1)
    simulator = DailyTraffic(problem, cache_dir=None)
    encoder = JunctionEncoder(problem, simulator.ctx, simulator.gamma,
                              demand_memory=True, bidirectional=True)
    W = simulator, encoder, DailyQ(encoder, hp)


def day_cost(simulator, traffic):
    losses = daily_losses(traffic, simulator.ctx, disconnected_flow='unserved')
    return float(simulator.rules.time_weight*losses['time_loss']
                 + simulator.rules.flow_weight*losses['flow_loss'])


def benefit(episode, road, *, horizon=HORIZON, cache=True):
    """Simulate only traffic, not subsequent repair decisions, in each branch.

    Completion in the production environment occurs at the start of a day.
    Therefore that first open day is included in the benefit, through day 60.
    Hidden truth is simulator-only, never passed into a policy observation.
    """
    assert road in episode._candidates()
    simulator = episode.simulator
    truth = episode.restoration.truth
    completion = episode.day + truth[road].true_duration
    damaged = {e:r.true_severity for e,r in truth.items() if e not in episode.completed}
    identity = dict(version='horizon_counterfactual_composite_v2', simulator=simulator.fingerprint,
        behavior=episode.behavior, day=episode.day,
        previous_traffic_key=episode.traffic.key, damage=sorted(damaged.items()),
        road=road, completion=completion, horizon=horizon)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = REWARD_CACHE/key[:2]/(key+'.pkl') if cache and REWARD_CACHE is not None else None
    if path is not None and path.exists():
        result = read_pickle(path)
        if result['identity'] != identity:
            raise ValueError('reward cache identity changed')
        return result
    started = time.perf_counter()
    without = copy.deepcopy(episode.traffic)
    with_repair = copy.deepcopy(episode.traffic)
    after_damage = dict(damaged)
    del after_damage[road]
    records = []
    # Identical traffic before completion is solved once, then cloned. Histories
    # evolve independently after completion, including the OD unmet mechanism.
    if completion <= horizon:
        for day in range(episode.day+1, horizon+1):
            without = simulator.solve(day, damaged, episode.behavior, without)
            if day < completion:
                with_repair = copy.deepcopy(without)
            else:
                with_repair = simulator.solve(day, after_damage, episode.behavior, with_repair)
                a, b = day_cost(simulator, without), day_cost(simulator, with_repair)
                records.append(dict(day=day, without_repair=a, with_repair=b, improvement=a-b))
    reward = float(sum(row['improvement'] for row in records))
    assert np.isfinite(reward)
    result = dict(identity=identity, reward=reward, daily=records,
                  seconds=time.perf_counter()-started, no_future_repairs=True)
    if path is not None:
        atomic_pickle(path, result)
    return result


def run_world(request):
    scenario, weights, epsilon, seed, reward_training = request
    simulator, encoder, model = W
    model.net.load_state_dict(weights, strict=True)
    model.net.eval()
    calls, hits, started = simulator.solve_calls, simulator.cache_hits, time.perf_counter()
    episode = DemandMemoryObservedEpisode(simulator, scenario, BEHAVIOR, disconnected_flow='unserved')
    if not reward_training:
        trajectory = rollout(episode, encoder, model, epsilon=epsilon, rng=np.random.RandomState(seed))
    else:
        assert scenario.scenario_id.startswith('train_')
        states, picks, rewards, physical_rewards, audit = [], [], [], [], []
        rng = np.random.RandomState(seed)
        while not episode.finished:
            state = encoder.encode(episode.observation())
            if rng.rand() < epsilon:
                pick = int(rng.randint(len(state['cand'])))
            else:
                with torch.no_grad():
                    pick = int(model(state).argmax())
            road = state['roads'][pick]
            before = pickle.dumps(episode.observation(), protocol=5)
            reward = benefit(episode, road)
            assert pickle.dumps(episode.observation(), protocol=5) == before
            _, physical, _ = episode.step(road)
            states.append(state)
            picks.append(pick)
            rewards.append(reward['reward'])
            physical_rewards.append(physical)
            audit.append(dict(day=state['day'], road=road, reward=reward['reward'],
                              completion=reward['identity']['completion'], horizon=HORIZON))
        result = episode.result()
        assert abs(sum(physical_rewards)+result['objective']-result['initial_loss']) < 1e-8
        trajectory = dict(states=states, picks=picks, rewards=rewards, result=result,
                          physical_rewards=physical_rewards, counterfactual_audit=audit)
    trajectory['runtime'] = dict(seconds=time.perf_counter()-started,
        ue_solves=simulator.solve_calls-calls, cached_days=simulator.cache_hits-hits)
    return trajectory


def clone(episode):
    return copy.deepcopy(episode,{id(episode.simulator):episode.simulator})


def action(episode):
    state=W[1].encode(episode.observation())
    with torch.no_grad():
        pick=int(W[2](state).argmax())
    return state,pick


def finish(episode):
    while not episode.finished:
        state,pick=action(episode)
        episode.step(state['roads'][pick])
    return episode.result()


def state_equal(a,b):
    assert a.keys()==b.keys()
    for k in a:
        if isinstance(a[k],np.ndarray):
            np.testing.assert_array_equal(a[k],b[k])
        else:
            assert a[k]==b[k]


def collect(request):
    index,scenario,weights,weights_hash,directory,steps=request
    assert scenario.scenario_id.startswith('train_'), 'Only training worlds may generate action labels'
    started=time.perf_counter()
    steps=tuple(range(len(scenario.roads)-1)) if steps is None else tuple(steps)
    label_identity=dict(scenario_signature=scenario.signature, simulator=W[0].fingerprint,
                        behavior=BEHAVIOR, positions=steps)
    directory=Path(directory)
    final_path=directory/'world.pkl'
    if final_path.exists():
        saved=read_pickle(final_path)
        assert saved['weights_hash']==weights_hash and saved['scenario']==scenario.scenario_id
        assert saved['identity']==label_identity
        return saved
    W[2].net.load_state_dict(weights,strict=True)
    W[2].net.eval()
    episode=DemandMemoryObservedEpisode(W[0],scenario,BEHAVIOR,disconnected_flow='unserved')
    rows=[]
    step=0
    while not episode.finished:
        state,pick=action(episode)
        if step in steps and len(state['roads'])>1:
            before=pickle.dumps(episode.observation(),protocol=5)
            prefix=float(episode.total_loss)
            meta=directory/f'decision_{step:02d}'/'state.pkl'
            if meta.exists():
                saved=read_pickle(meta)
                assert saved['weights_hash']==weights_hash and abs(saved['prefix']-prefix)<1e-8
                assert saved['identity']==label_identity
                state_equal(saved['state'],state)
            else:
                atomic_pickle(meta,dict(identity=label_identity,weights_hash=weights_hash,state=state,prefix=prefix))
            costs=[]
            branches=[]
            for road in state['roads']:
                path=meta.parent/f'action_{road:02d}.pkl'
                if path.exists():
                    branch_result=read_pickle(path)
                    assert branch_result['weights_hash']==weights_hash and branch_result['action']==road
                    assert branch_result['identity']==label_identity
                else:
                    begin=time.perf_counter()
                    branch=clone(episode)
                    _,first_reward,_=branch.step(road)
                    result=finish(branch)
                    remaining=float(result['objective']-prefix)
                    assert remaining+1e-8>=-first_reward
                    branch_result=dict(identity=label_identity,weights_hash=weights_hash,action=road,remaining_loss=remaining,
                        objective=result['objective'],order=result['order'],starts=result['starts'],
                        seconds=time.perf_counter()-begin)
                    atomic_pickle(path,branch_result)
                assert pickle.dumps(episode.observation(),protocol=5)==before, 'Branch changed parent history'
                if not np.isfinite(branch_result['remaining_loss']):
                    raise FloatingPointError('nonfinite action rollout loss')
                costs.append(branch_result['remaining_loss'])
                branches.append(branch_result)
            rows.append(dict(state=state,costs=np.asarray(costs,dtype=np.float64),decision=step,
                             prefix_cost=prefix,base_pick=pick,branches=branches))
        episode.step(state['roads'][pick])
        step+=1
    baseline=episode.result()
    errors=[abs(row['branches'][row['base_pick']]['objective']-baseline['objective']) for row in rows]
    assert max(errors,default=0)<1e-4, ('Continuation replay mismatch',errors)
    value=dict(identity=label_identity,index=index,scenario=scenario.scenario_id,weights_hash=weights_hash,rows=rows,
        baseline=baseline,branch_replay_max_error=max(errors,default=0),seconds=time.perf_counter()-started,
        branch_seconds=sum(b['seconds'] for row in rows for b in row['branches']))
    atomic_pickle(final_path,value)
    return value
