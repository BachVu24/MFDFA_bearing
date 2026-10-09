"""Recording-level execution with verified checkpoints and Drive commits."""
import argparse, csv, hashlib, json, os, shutil, sys, time
from pathlib import Path
from execution.provenance import ROOT, resolve, sha, atomic_json, verify, aliases, normalized, migrate

def pipeline(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'result/test/colab_runs'): raise ValueError('Output must be inside result/test/colab_runs; validated results are read-only')
    os.environ['PHM_OUTPUT_DIR']=str(output)
    sys.path.insert(0,str(ROOT/'result/test/step5/scripts'))
    import pipeline as P
    P.BASE=output;output.mkdir(parents=True,exist_ok=True)
    return P

def sync_tree(source,destination):
    """Copy payloads first, completion markers last; never delete Drive inputs."""
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    files=sorted((p for p in source.rglob('*') if p.is_file() and not p.name.endswith('.tmp')),key=lambda p: ('checkpoints' in p.parts,p.as_posix()))
    for p in files:
        target=destination/p.relative_to(source)
        if target.exists() and sha(p)==sha(target): continue
        target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.tmp')
        shutil.copyfile(p,temp)
        if sha(temp)!=sha(p): raise IOError('Drive copy hash mismatch: '+str(target))
        os.replace(temp,target)

def checkpoint_valid(path,source,config):
    if not path.exists(): return False
    try:
        d=json.loads(path.read_text(encoding='utf-8'))
        if d['schema']!=1 or d['source_sha256']!=source['sha256'] or d['config_sha256']!=config['config_sha256']: return False
        if d['execution_sha256']!=sha(Path(__file__)): return False
        if d['recording']!=source['recording'] or d['samples']!=source['samples']: return False
        if d['channels']!=['output_voltage','input_voltage'] or len(d['files'])<2: return False
        if len({f['path'] for f in d['files']})!=len(d['files']): return False
        verify(d['files']);return bool(d['complete'])
    except (KeyError,ValueError,RuntimeError,OSError): return False

def historical_channel(P,source,channel,config):
    """Reuse only artifacts attested by the existing validation and hash manifests."""
    base=P.HISTORICAL_BASE
    validation=json.loads((base/'diagnostics/validation.json').read_text(encoding='utf-8'))
    if not validation.get('all_passed'): return None
    path=base/'diagnostics/recordings'/(source['recording']+'_'+channel+'.json')
    if not path.exists(): return None
    manifest={r['path'].replace('\\','/'):r['sha256'] for r in P.read_table(base/'diagnostics/output_manifest.csv')}
    if manifest.get(path.relative_to(base).as_posix())!=sha(path): raise RuntimeError('Validated recording metadata changed: '+str(path))
    obj=json.loads(path.read_text(encoding='utf-8'))
    signatures={hashlib.sha256((c+source['sha256']).encode()).hexdigest() for c in aliases(config)}
    if obj['signature'] not in signatures: return None
    array_hashes=[]
    baseline={r['path'].replace('\\','/'):r['sha256'] for r in P.read_table(base/'config/reused_step4_array_manifest.csv')}
    for row in obj['features']:
        if row['source_sha256']!=source['sha256'] or row['config_sha256'] not in aliases(config): return None
        relative=row.get('array_path')
        if not relative: continue
        p=resolve(relative)
        expected=manifest.get(p.relative_to(base).as_posix()) if p.is_relative_to(base) else baseline.get(p.relative_to(ROOT).as_posix())
        if expected is None or sha(p)!=expected: raise RuntimeError('Validated array MISSING/CHANGED: '+str(p))
        array_hashes.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=expected))
    # Adapt metadata in memory, retaining numerical features and original arrays.
    obj=normalized(obj);obj['signature']=hashlib.sha256((config['config_sha256']+source['sha256']).encode()).hexdigest()
    obj['array_hashes']=array_hashes
    for row in obj['features']: row['config_sha256']=config['config_sha256']
    obj['historical_config_aliases']=sorted(aliases(config))
    return obj

def run(step='step5',run_id='default',drive_output=None,max_recordings=None,reuse_validated=True,analyze=False):
    if step not in ('step4','step5'): raise NotImplementedError('Step 6 science is not specified yet; register its experiment without changing locked Step 4–5')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id): raise ValueError('Invalid run_id')
    migrate();output=ROOT/'result/test/colab_runs'/step/run_id
    if drive_output and Path(drive_output).exists():
        # Restore only this dedicated generated run. Sources/results remain separate.
        shutil.copytree(drive_output,output,dirs_exist_ok=True)
    P=pipeline(output);config,sources=P.inventory()
    if step=='step4': sources=[s for s in sources if s['speed']==50 and s['load']=='High']
    if max_recordings is not None: sources=sources[:max_recordings]
    runtime=P.accelerator.initialize(output/'_runtime')
    import numpy, scipy
    atomic_json(output/'config/execution_environment.json',dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,platform=sys.platform,accelerator=runtime,execution_sha256=sha(Path(__file__)),thread_environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}))
    for method in ('raw','vmd','emd'): (output/'arrays'/method).mkdir(parents=True,exist_ok=True)
    features=[];decompositions=[];completed=[]
    for source in sources:
        checkpoint=output/'checkpoints'/(source['recording']+'.json')
        valid=checkpoint_valid(checkpoint,source,config)
        objs=[]
        for channel_info in P.CHANNELS:
            channel=channel_info[1];cache=output/'diagnostics/recordings'/(source['recording']+'_'+channel+'.json')
            if valid:
                obj=json.loads(cache.read_text(encoding='utf-8'))
            else:
                obj=historical_channel(P,source,channel,config) if reuse_validated else None
                if obj is not None: P.dump(cache,obj)
                else: obj,_=P.signal_job(source,channel_info,config)
            objs.append(obj)
        files=[]
        for channel_info,obj in zip(P.CHANNELS,objs):
            cache=output/'diagnostics/recordings'/(source['recording']+'_'+channel_info[1]+'.json')
            files.append(dict(path=cache.relative_to(ROOT).as_posix(),sha256=sha(cache)))
            files+=obj['array_hashes'];features+=obj['features'];decompositions+=obj['decompositions']
        verify(files)
        atomic_json(checkpoint,dict(schema=1,complete=True,recording=source['recording'],source_path=source['relative_source'],source_sha256=source['sha256'],samples=source['samples'],channels=[c[1] for c in P.CHANNELS],config_sha256=config['config_sha256'],execution_sha256=sha(Path(__file__)),files=files))
        completed.append(source['recording'])
        P.table(output/'features/mfdfa_features.csv',features)
        P.table(output/'features/valid_features.csv',[r for r in features if r['analysis_valid']])
        P.table(output/'diagnostics/mfdfa_quality_control.csv',features)
        P.dump(output/'diagnostics/decomposition_diagnostics.json',decompositions)
        atomic_json(output/'diagnostics/progress.json',dict(step=step,complete_recordings=completed,total_selected=len(sources),complete=len(completed)==len(sources)))
        if drive_output: sync_tree(output,drive_output)
        print(source['recording'], 'validated checkpoint' if valid else 'committed',flush=True)
    if analyze:
        if step!='step5' or max_recordings is not None: raise ValueError('Full Step 5 analysis requires complete selected inventory')
        import subprocess
        test_env=dict(os.environ,PHM_TEST_OUTPUT_DIR=str(output/'diagnostics'))
        subprocess.run([sys.executable,'-B',str(ROOT/'result/test/step5/scripts/tests.py')],env=test_env,cwd=ROOT,check=True)
        import analysis
        analysis.run()
    if drive_output: sync_tree(output,drive_output)
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',choices=['verify','step4','step5','benchmark'],default='verify')
    parser.add_argument('--run-id',default='default');parser.add_argument('--drive-output');parser.add_argument('--max-recordings',type=int)
    parser.add_argument('--recompute',action='store_true');parser.add_argument('--analyze',action='store_true');args=parser.parse_args()
    if args.step=='verify': migrate();print('All 385 dependency/data references verified; path migration checked')
    elif args.step=='benchmark':
        from execution.benchmark import run_benchmark
        run_benchmark(ROOT/'result/test/colab_runs/benchmark'/args.run_id)
    else: run(args.step,args.run_id,args.drive_output,args.max_recordings,not args.recompute,args.analyze)
if __name__=='__main__': main()
