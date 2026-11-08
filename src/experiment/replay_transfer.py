"""Re-evaluate four frozen policies in both original transfer environments.

Copied into a prepared isolated experiment directory by the Results entry point.
It imports that directory's frozen scientific code, never current training code.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,hashlib,argparse,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import evaluation_workers as ev
import torch
from common import _write_json

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def task(request):
    scenario,method,behavior=request
    output=ROOT/f'reevaluation/{method}/{behavior}/{scenario}.json'
    selected=json.loads((ROOT/'replay_inputs.json').read_text(encoding='utf8'))
    expected=selected['expected_objectives'][f'{method}_{behavior}'][scenario]
    if output.exists():
        cached=json.loads(output.read_text(encoding='utf8'))
        assert cached['result']['scenario']==scenario
        assert abs(cached['result']['objective']-expected)<=1e-7
        return cached
    weights=None;order=None
    if method=='rl':
        path=ROOT/selected['weights'];assert sha(path)==selected['weights_sha256']
        weights=torch.load(path,map_location='cpu',weights_only=True)
    if method=='ga':order=json.loads((ROOT/'result/ga/results/solution.json').read_text())['order']
    world=next(s for s in ev.W[1]['test'] if s.scenario_id==scenario)
    value=ev.episode(world,method,behavior,order=order,weights=weights)
    if abs(value['result']['objective']-expected)>1e-7:
        _write_json(ROOT/f'reevaluation/mismatches/{method}_{behavior}_{scenario}.json',
                    dict(request=request,expected_objective=expected,actual=value,
                         difference=value['result']['objective']-expected,
                         accepted=False,platform=sys.platform))
        raise RuntimeError(f'Historical evaluation mismatch: {request}, {value["result"]["objective"]} != {expected}')
    _write_json(output,value);return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--pilot',action='store_true',help='One scenario per method and environment')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'prepared_manifest.json').read_text(encoding='utf8'))
    for rel,h in manifest.items():assert sha(ROOT/rel)==h,rel
    tasks=[(f'test_{i:03d}',m,b) for m in ('static','dynamic','ga','rl') for b in ('matched','response7d') for i in range(1 if args.pilot else 50)]
    errors=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=ev.initialize) as pool:
        futures={pool.submit(task,t):t for t in tasks}
        for i,f in enumerate(as_completed(futures),1):
            try:f.result()
            except Exception as exc:errors.append(dict(request=futures[f],error=repr(exc)))
            print(i,len(tasks),flush=True)
    if errors:
        _write_json(ROOT/'reevaluation/failure.json',dict(complete=False,errors=errors))
        raise RuntimeError(f'{len(errors)} branches did not reproduce; see reevaluation/failure.json')
    _write_json(ROOT/('reevaluation/pilot.json' if args.pilot else 'reevaluation/complete.json'),dict(complete=True,branches=len(tasks),matched_original_objectives=True))

if __name__=='__main__':main()
