"""Replay the four recorded schedules with the original full-environment solver.

The original policies' decisions are fixed by their saved scenario schedules.
This verifies system metrics and delay decomposition, without training/search.
For a changed policy, produce new schedules rather than relabel this replay.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,argparse,multiprocessing
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
# Establish the frozen scientific src before importing the archived diagnostics.
import evaluate_actions as original
import compare_abc as diagnostic
import pandas as pd

def initialize():
    problem=original.build_problem(11,42,variant='poi_hidden6')
    global SCENARIOS
    SCENARIOS={s.scenario_id:s for s in original.make_splits(problem)['test']}
    expected=original.read(original.EXP/'data/scenarios/test.json')
    assert [s.record() for s in SCENARIOS.values()]==expected
    diagnostic.initialize(problem)
    diagnostic.OUT=ROOT/'reevaluation/baselines'

def run(row):
    expected={k:row[k] for k in ('F','recovery_day','repair_completion_day')}
    daily=pd.DataFrame(row['daily'])[['day','cost','time_loss','flow_loss','total_flow_fraction']].to_numpy()
    return diagnostic.replay((row['method'],SCENARIOS[row['scenario']],row['order'],expected,daily))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=32);ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    for rel,h in json.loads((ROOT/'prepared_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    requests=json.loads((ROOT/'baseline_requests.json').read_text())
    if args.pilot:requests=requests[:1]
    out=ROOT/'reevaluation/baselines';out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        rows=list(pool.map(run,requests))
    pd.DataFrame([dict(method=r['method'],scenario=r['scenario'],**d) for r in rows for d in r['rows']]).to_csv(out/'daily_abc.csv',index=False)
    (out/('pilot.json' if args.pilot else 'complete.json')).write_text(json.dumps(dict(complete=True,trajectories=len(rows),same_original_daily_losses=True),indent=2))
if __name__=='__main__':main()
