"""Inspect, verify, redraw, or prepare isolated reruns of recorded Results.

No command implicitly trains a model, submits a job, or rewrites a historical
scientific result. Configuration is selected through src.config.results_profile.
"""
from pathlib import Path, PureWindowsPath
import argparse,csv,hashlib,json,os,shlex,shutil,subprocess,sys,tarfile,tempfile
from src import config

ROOT=Path(__file__).resolve().parents[2]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def child(root,rel):
    # Old manifests were written on Windows; resolve them identically on HPC.
    rel=str(rel).replace('\\','/')
    if PureWindowsPath(rel).drive or rel.startswith('/'):raise ValueError('Expected relative artifact path: '+rel)
    root=Path(root).resolve();p=(root/rel).resolve()
    if p==root or not p.is_relative_to(root):raise ValueError(f'Path outside {root}: {rel}')
    return p
def verify_files(folder, manifest):
    for rel,h in manifest.items():
        p=child(folder,rel)
        if not p.is_file() or sha(p)!=h:raise ValueError(f'Missing or changed recorded artifact: {p}')
    return len(manifest)

def verify_ranking(folder):
    import numpy as np
    report=folder/'returned/result/report'
    rows=read(report/'state_results.json');summary=read(report/'summary.json')
    branches=read(report/'branch_index.json')
    ix={(r['state_id'],r['setting'],r['road']):r for r in branches}
    assert len(ix)==len(branches)==7534 and len(rows)==467
    excluded=read(report/'excluded.json')
    assert excluded['excluded_states']==['w062_rl_d03']
    assert len(read(folder/'returned/result/states.json'))==468
    tol=read(folder/'protocol.json')['tie_absolute_tolerance']
    matrix=np.zeros((5,6),int);counts=dict(changed=0,unchanged=0,uncertain=0)
    positive=0
    for r in rows:
        sid=r['state_id'];roads=r['candidates'];assert len(roads)>=6
        a=np.asarray(r['fixed_rewards']);b=np.asarray(r['response_rewards'])
        for setting,values in [('fixed',a),('response',b)]:
            baseline=ix[sid,setting,None]['total_loss']
            np.testing.assert_allclose(values,[baseline-ix[sid,setting,e]['total_loss'] for e in roads],rtol=0,atol=1e-9)
        best_a=set(np.flatnonzero(a>=a.max()-tol));best_b=set(np.flatnonzero(b>=b.max()-tol))
        category='changed' if not best_a&best_b else 'unchanged' if len(best_a)==len(best_b)==1 else 'uncertain'
        assert category==r['category'];counts[category]+=1
        penalty=float(b.max()-max(b[list(best_a)]))
        best_loss=min(ix[sid,'response',e]['total_loss'] for e in roads)
        expected=None if best_loss<=0 else 100*penalty/best_loss
        if expected is None:assert r['relative_cost_percent'] is None
        else:
            assert abs(expected-r['relative_cost_percent'])<1e-9
            positive+=expected>0
        order=np.argsort(-a,kind='stable');rank=np.empty(len(b),int);rank[np.argsort(-b,kind='stable')]=np.arange(1,len(b)+1)
        tied=any(abs(a[i]-a[j])<=tol for i in order[:5] for j in range(len(a)) if j!=i)
        tied|=any(abs(b[i]-b[j])<=tol and min(rank[i],rank[j])<=5 for i in order[:5] for j in range(len(b)) if j!=i)
        assert r['sankey_eligible']==(not tied)
        if not tied:
            for i,j in enumerate(order[:5]):matrix[i,min(6,rank[j])-1]+=1
    assert counts==summary['categories']
    saved=np.zeros((5,6),int)
    with (report/'sankey.csv').open() as f:
        for r in csv.DictReader(f):saved[int(r['simplified_rank'])-1,int(r['response_rank_bucket'])-1]=int(r['actions'])
    np.testing.assert_array_equal(matrix,saved)
    assert matrix.sum()==5*summary['sankey_states']
    return dict(states=len(rows),branches=len(branches),categories=counts,positive_states=int(positive),sankey_states=summary['sankey_states'])

def verify_transfer(profile):
    import numpy as np
    folder=ROOT/profile['source_archive'];registry=read(folder/'figure_registry.json')
    assert sha(child(folder,registry['rl_weights']))==profile['protocol']['weights_sha256']
    expected=read(ROOT/profile['source_comparison'])
    for group,entry in registry['policy_observations'].items():
        p=child(folder,entry['path'])
        if entry['format']=='csv':
            with p.open() as f:rows=list(csv.DictReader(f))
            vals=np.array([[float(r['F']),float(r['time_loss']),float(r['flow_loss'])] for r in rows]);ids=[r['scenario'] for r in rows]
        else:
            rows=[read(f)['result'] for f in sorted(p.glob('test_*.json'))]
            vals=np.array([[r['objective'],r['time_loss'],r['flow_loss']] for r in rows]);ids=[r['scenario'] for r in rows]
        assert sorted(ids)==[f'test_{i:03d}' for i in range(50)] and np.isfinite(vals).all()
        np.testing.assert_allclose(vals[:,0],.5*(vals[:,1]+vals[:,2]),atol=1e-9,rtol=0)
        assert abs(vals[:,0].mean()-expected['summaries'][group]['mean_system_loss'])<1e-9
    assert expected['same_rl_weights_in_both_environments']
    return dict(groups=8,scenarios_per_group=50,weights_sha256=registry['rl_sha256'])

def verify_full(folder):
    import numpy as np
    contract=read(folder/'action_contract.json')
    verify_files(folder/'inputs',contract['model_hashes'])
    branch_count=pair_count=0;accuracy=[]
    for i in range(50):
        row=read(folder/f'action_results/test_{i:03d}.json')
        costs=np.array([b['remaining_loss'] for b in row['branches']]);qs=np.array(row['q_values'])
        assert qs.shape==(4,len(costs)) and np.isfinite(qs).all() and np.isfinite(costs).all()
        assert row['encoded_state_matches_saved_test'] and row['native_final_policy_reproduced']
        a,b=np.triu_indices(len(costs),1);actual=costs[b]-costs[a];eligible=np.abs(actual)>1e-8
        branch_count+=len(costs);pair_count+=len(a)
        accuracy.append([np.mean((q[a][eligible]-q[b][eligible])*actual[eligible]>0) for q in qs])
    assert branch_count==347 and pair_count==1087
    expected=read(folder/'plot_audit.json')['statistics']['actions']['accuracy_percent']
    np.testing.assert_allclose(np.mean(accuracy,axis=0)*100,expected,rtol=0,atol=1e-10)
    return dict(states=50,branches=branch_count,pairs=pair_count,accuracy_percent=list(np.mean(accuracy,axis=0)*100))

def verify(name):
    p=config.results_profile(name);folder=ROOT/p['experiment']
    files=verify_files(folder,read(folder/'artifact_manifest.json')['files'])
    result=dict(profile=name,verified_files=files)
    if p['renderer']=='ranking':result['science']=verify_ranking(folder)
    elif p['renderer']=='transfer':result['science']=verify_transfer(p)
    elif p['renderer']=='full':result['science']=verify_full(folder)
    elif p['renderer']=='baselines':
        from src.analysis.verify_baselines import verify_baselines
        result['science']=verify_baselines(folder)
    elif p['renderer']=='animation':
        import gzip,html,re
        text=(ROOT/p['template']).read_text(encoding='utf8')
        assert 'sandbox="allow-scripts"' in text and 'Content-Security-Policy' in text
        inner=html.unescape(re.search(r'data-srcdoc="([\s\S]*?)"',text)[1]);start=inner.index(',D=')+3
        payload,_=json.JSONDecoder().raw_decode(inner[start:])
        measured=json.loads(gzip.decompress((folder/f'data/visual_{p["kind"]}.json.gz').read_bytes()))
        assert all(payload.get(k)==v for k,v in measured.items())
        result['science']=dict(case=payload['state_id'],days=payload['end']-payload['start'],embedded_data_matches_replay=True)
    result['verified']=True
    return result

def render(name,output=None):
    p=config.results_profile(name);verify(name)
    output=Path(output).resolve() if output else ROOT/'outputs/sections'/p['section']
    output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONUTF8='1',RESULTS_OUTPUT=str(output))
    kind=p['renderer']
    if kind=='animation':
        from src.analysis.render_mechanism import main
        main(p['kind'],output);return
    module={'full':'render_full_results','ranking':'render_behavior_ranking','transfer':'render_behavior_transfer','baselines':'render_full_baselines'}[kind]
    command=[sys.executable,'-m','src.analysis.'+module]
    if kind=='transfer':command += [str(ROOT/p['source_comparison']),str(output/'system_loss_dual_axis_round4_both.png')]
    subprocess.run(command,cwd=ROOT,env=env,check=True)

def prepare(name,destination,workers=32):
    """Prepare a NEW isolated directory. Never mutate or submit the original run."""
    p=config.results_profile(name);verify(name)
    dest=Path(destination).resolve()
    if dest.exists():raise FileExistsError('Rerun directory must not exist: '+str(dest))
    folder=ROOT/p['experiment'];kind=p.get('compute');copies={}
    if kind in ('anchor','transfer','animation'):
        case_folder=folder
        if kind=='animation':folder=ROOT/config.results_profile(p['parent'])['experiment']
        copies={f.relative_to(folder/'runtime').as_posix():f for f in (folder/'runtime').rglob('*') if f.is_file() and '__pycache__' not in f.parts}
        if kind in ('anchor','animation'):
            remote=read(folder/'remote_inputs.json');missing=[]
            for rel,h in remote['files'].items():
                local=folder/'runtime'/rel;source=local if local.exists() else child(Path(remote['root']),rel)
                if not source.is_file():missing.append(rel)
                elif sha(source)!=h:raise ValueError('Changed inherited input: '+str(source))
                else:copies[rel]=source
            if missing:raise FileNotFoundError(f'{len(missing)} inherited inputs are on HPC. Run prepare on torch, or fetch-inputs behavior_ranking first. No rerun directory created.')
            command=f'run_anchor.py {workers}'
            if kind=='animation':
                raw_path=folder/'returned/result/report/raw_artifact_manifest.json'
                raw=read(raw_path);sid=p['case'][0]['evidence']['state_id']
                selected=read(case_folder/'data/selected.json')
                roads={int(c['evidence'][label]) for c in selected for label in ('a','b')}
                names=[f'result/states/{sid}/{n}' for n in ('snapshot.pkl','reference_curve.npz','reference_traffic.pkl')]
                names += [f'result/states/{sid}/branches/{setting}_{road}.npz' for setting in ('fixed','response') for road in sorted(roads)]
                for rel in names:
                    local=folder/'runtime'/rel;source=local if local.exists() else child(Path(remote['root']),rel)
                    if not source.is_file():raise FileNotFoundError('Case replay needs HPC artifact: '+str(source))
                    if sha(source)!=raw[rel]:raise ValueError('Changed case array: '+rel)
                    copies[rel]=source
                copies['result/report/raw_artifact_manifest.json']=raw_path
                audit_name='animation_case_v3' if p['kind']=='congestion' else 'animation_latent_v3'
                copies[f'audits/{audit_name}/selected.json']=case_folder/'data/selected.json'
                copies['historical_replay_cases.py']=case_folder/'replay_cases.py'
                copies['replay_animation.py']=ROOT/'src/experiment/replay_animation.py'
                command='replay_animation.py'
        else:
            copies['replay_transfer.py']=ROOT/'src/experiment/replay_transfer.py'
            command=f'replay_transfer.py --workers {workers}'
    elif kind in ('actions','baseline_replay'):
        if kind=='baseline_replay':folder=ROOT/config.results_profile('full_rl')['experiment']
        copies={f.relative_to(folder/'inputs').as_posix():f for f in (folder/'inputs').rglob('*') if f.is_file() and '__pycache__' not in f.parts}
        copies['analysis/rl_full_verification_20261106/evaluate_actions.py']=folder/'evaluate_actions.py'
        command=f'analysis/rl_full_verification_20261106/evaluate_actions.py --workers {workers}'
        if kind=='baseline_replay':
            baseline=ROOT/p['experiment']
            copies['analysis/rl_full_verification_20261106/compare_abc.py']=baseline/'historical_code/compare_abc.py'
            copies['analysis/rl_full_verification_20261106/replay_baselines.py']=ROOT/'src/experiment/replay_baselines.py'
            copies['baseline_requests.json']=baseline/'baseline_requests.json'
            command=f'analysis/rl_full_verification_20261106/replay_baselines.py --workers {workers}'
    else:raise ValueError('This profile redraws archived comparison data; original scientific diagnostic entries are under historical_code. It does not start a new training run.')
    # Resolve and validate all destinations before writing anything.
    for rel in copies:child(dest,rel)
    dest.mkdir(parents=True)
    for rel,source in copies.items():
        target=child(dest,rel);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    manifest={rel:sha(dest/rel) for rel in copies}
    write(dest/'prepared_manifest.json',manifest)
    write(dest/'prepared_profile.json',dict(profile=name,definition=p,source_configuration_sha256=sha(ROOT/'src/results_profiles.json'),command=command,submitted=False))
    # The original entry is kept intact. A distinct launcher changes only cwd.
    sbatch=f'''#!/bin/bash
#SBATCH --job-name=results_replay
#SBATCH --account=torch_pr_64_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={workers}
#SBATCH --hint=nomultithread
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=replay_%j.out
#SBATCH --error=replay_%j.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export ROAD_TRAFFIC_DISK_CACHE=0
srun --cpu-bind=cores /scratch/xp2239/road_mechanism_env_20261028/bin/python -B -u {command}
'''
    (dest/'replay.sbatch').write_text(sbatch,encoding='utf8',newline='\n')
    return dict(directory=str(dest),files=len(copies),command=command,submitted=False)

def fetch_inputs(name):
    """Fetch only hash-listed inherited inputs, never all of HPC scratch."""
    p=config.results_profile(name);folder=ROOT/p['experiment'];remote=read(folder/'remote_inputs.json')
    expected=remote['files'];dest=folder/'runtime'
    pending={rel:h for rel,h in expected.items() if not child(dest,rel).is_file()}
    if not pending:return dict(downloaded=0)
    code="""import sys,json,hashlib,tarfile,pathlib
request=json.load(sys.stdin);root=pathlib.Path(request['root']).resolve()
for rel,h in request['files'].items():
 p=(root/rel).resolve()
 assert p.is_relative_to(root) and p.is_file(),rel
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,rel
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for rel in request['files']:tar.add(root/rel,arcname=rel,recursive=False)
"""
    # No credentials or auth codes are stored. Authentication remains the user's SSH login.
    with tempfile.TemporaryDirectory(prefix='results_fetch_') as temp:
        archive=Path(temp)/'inputs.tar.gz'
        with archive.open('wb') as out:
            proc=subprocess.run(['ssh','-o','ForwardAgent=no','-o','BatchMode=yes','-o','ConnectTimeout=10',remote['host'],
                                 'python3 -c '+shlex.quote(code)],input=json.dumps(dict(root=remote['root'],files=pending)).encode(),stdout=out,stderr=subprocess.PIPE)
        if proc.returncode:raise RuntimeError('SSH transfer unavailable. Authenticate the HPC connection, then retry. '+proc.stderr.decode(errors='replace')[-1500:])
        extracted=Path(temp)/'files';extracted.mkdir()
        with tarfile.open(archive) as tar:
            members=tar.getmembers()
            assert {m.name for m in members}==set(pending)
            for m in members:child(extracted,m.name);assert m.isfile()
            tar.extractall(extracted,filter='data')
        verify_files(extracted,pending)
        for rel in pending:
            target=child(dest,rel)
            if target.exists():raise FileExistsError(target)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(extracted/rel,target)
    verify_files(dest,expected)
    return dict(downloaded=len(pending),hashes_verified=True)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['list','show','verify','render','prepare','fetch-inputs'])
    ap.add_argument('profile',nargs='?',default='all')
    ap.add_argument('--output',type=Path);ap.add_argument('--workers',type=int,default=32)
    args=ap.parse_args(argv);catalog=config.results_profiles()
    if args.action=='list':print(json.dumps(catalog,ensure_ascii=False,indent=2));return
    names=list(catalog['profiles']) if args.profile=='all' else [args.profile]
    if args.action in ('prepare','fetch-inputs') and len(names)!=1:ap.error('Specify one exact profile')
    if args.workers<1:ap.error('workers must be positive')
    for name in names:
        if args.action=='show':result=config.results_profile(name)
        elif args.action=='verify':result=verify(name)
        elif args.action=='render':render(name,args.output);result=dict(profile=name,rendered=True)
        elif args.action=='fetch-inputs':result=fetch_inputs(name)
        else:
            if args.output is None:ap.error('prepare requires --output NEW_DIRECTORY')
            result=prepare(name,args.output,args.workers)
        print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
