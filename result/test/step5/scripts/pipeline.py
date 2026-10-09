"""Step 5: locked full-recording validation. No fitting-range/parameter search."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import sys
sys.dontwrite_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HISTORICAL_BASE=ROOT/'result/test/step5'
BASE=Path(os.environ.get('PHM_OUTPUT_DIR',str(HISTORICAL_BASE))).resolve()
sys.path.insert(0,str(ROOT))
from execution.provenance import resolve, normalized, verify, aliases, atomic_json
STEP4=ROOT/'result/test/step4_corrected'
sys.path[:0]=[str(ROOT/'code/core'),str(STEP4/'scripts'),str(ROOT/'result/test/step4/compare')]
import ast,csv,json,hashlib,re,itertools,time,warnings
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from mfdfa import mfdfa
from scaling import fit_scaling,RULE
from execution import accelerator
sys.modules['vmd_accel']=accelerator
from execution.accelerator import vmd
from emd_long_record import emd_long_record
from vmd_sensitivity import mode_metrics

STATES=['H1','H2','H5','H6']; SPEEDS=[30,35,40,45,50]; LOADS=['High','Low']
LABELS={'H1':'Healthy','H2':'24T chipped gear','H5':'24T broken gear + bearing inner-race fault','H6':'bent input shaft'}
CHANNELS=[(1,'output_voltage','primary'),(0,'input_voltage','secondary')]
PATTERN=re.compile(r'^helical[ _]([1256])_(30|35|40|45|50)hz_(High|Low)_([12])\.txt$',re.I)

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,json.loads(json.dumps(value,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())))
def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def read_table(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def typed(row):
    out={}
    for key,val in row.items():
        if val in ['True','False']: out[key]=val=='True'
        elif re.fullmatch(r'-?\d+',val): out[key]=int(val)
        else:
            try: out[key]=float(val)
            except ValueError: out[key]=val
    return out

def lock_config():
    saved=json.loads((HISTORICAL_BASE/'config/locked_step4_configuration.json').read_text(encoding='utf-8'))
    verify(saved['dependencies'])
    m=json.loads((STEP4/'config/mfdfa_configuration.json').read_text(encoding='utf-8'))
    qc=json.loads((STEP4/'config/mfdfa_quality_rules.json').read_text(encoding='utf-8'))
    interval=json.loads((STEP4/'config/selected_scaling_range.json').read_text(encoding='utf-8'))['selected_interval']
    choice=json.loads((STEP4/'config/selected_vmd_configuration.json').read_text(encoding='utf-8'))
    assert interval==[64,512] and RULE==qc['rule']
    assert choice['selected_config']==dict(K=7,alpha=500,tau=0.)
    assert m['detrending_order']==2
    source=ROOT/'result/test/step4/compare/run_step4.py'
    node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
    emd_node=next(k.value for k in node.value.keywords if k.arg=='emd')
    emd={k.arg:ast.literal_eval(k.value) for k in emd_node.keywords}
    assert emd==dict(max_imfs=6,max_siftings=200,sd_threshold=.2,envelope_tol=.05,extrema_relative_tolerance=.005)
    references=[STEP4/'config/mfdfa_configuration.json',STEP4/'config/mfdfa_quality_rules.json',STEP4/'config/selected_scaling_range.json',STEP4/'config/selected_vmd_configuration.json',STEP4/'scripts/scaling.py',STEP4/'scripts/vmd_accel.py',STEP4/'scripts/vmd_kernel.cpp',ROOT/'result/test/step4/compare/emd_long_record.py',source,ROOT/'code/core/mfdfa.py',ROOT/'code/core/vmd.py',ROOT/'code/core/emd.py',ROOT/'DATA/labeled/VidData.xls']
    config=dict(q=m['q'],scales=m['scales'],m=2,fit_interval=interval,qc_rule=qc['rule'],bands_hz=qc['bands_hz'],
        fs=m['fs_hz'],vmd=dict(**choice['selected_config'],DC=False,init=1,tol=1e-7,max_iter=2000),emd=emd,
        channels=CHANNELS,states=STATES,speeds=SPEEDS,loads=LOADS,repeats=[1,2],
        preprocessing='Full original lengths. Raw: only internal MF-DFA mean removal; EMD/VMD: global mean removed. No windows/resampling.',
        dependencies=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in references])
    config['config_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    target=BASE/'config/locked_step4_configuration.json'
    if target.exists(): assert json.dumps(json.loads(target.read_text(encoding='utf-8')),sort_keys=True)==json.dumps(config,sort_keys=True),'Locked Step 4 inputs changed'
    assert json.dumps(saved,sort_keys=True)==json.dumps(config,sort_keys=True),'Locked Step 4 inputs changed'
    if not target.exists(): dump(target,config)
    return config

def parse_name(name):
    match=PATTERN.fullmatch(name)
    if not match: return None
    h,speed,load,repeat=match.groups()
    return ('H'+h,int(speed),load.capitalize(),int(repeat))

def snapshot_reuse_and_runtime():
    import scipy
    files=[STEP4/'_runtime/vmd_kernel.dll',STEP4/'_runtime/libwinpthread-1.dll',STEP4/'features/mfdfa_features.csv',STEP4/'config/source_inventory.json']
    provenance=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        optional_admm_backend='Step 4 parity-validated fused C++ kernel; NumPy FFT',
        dependencies=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in files if p.exists()])
    path=BASE/'config/runtime_provenance.json'
    historical=json.loads((HISTORICAL_BASE/'config/runtime_provenance.json').read_text(encoding='utf-8'))
    verify(historical['dependencies'])
    assert normalized(historical['dependencies'])==provenance['dependencies']
    if path.exists():
        existing=json.loads(path.read_text(encoding='utf-8'));verify(existing['dependencies'])
        assert normalized(existing['dependencies'])==provenance['dependencies']
    else: dump(path,provenance)
    manifest=[]
    for r in read_table(STEP4/'features/mfdfa_features.csv'):
        path=STEP4/'arrays'/r['pipeline']/(r['recording']+'_'+r['channel']+'_'+r['component']+'.npz')
        manifest.append(dict(recording=r['recording'],channel=r['channel'],pipeline=r['pipeline'],component=r['component'],path=path.relative_to(ROOT).as_posix(),sha256=digest(path)))
    target=BASE/'config/reused_step4_array_manifest.csv'
    original=read_table(HISTORICAL_BASE/'config/reused_step4_array_manifest.csv');verify(original)
    assert normalized(original)==manifest
    if target.exists(): assert normalized(read_table(target))==manifest
    else: table(target,manifest)

def inventory():
    config=lock_config();snapshot_reuse_and_runtime(); expected=list(itertools.product(STATES,SPEEDS,LOADS,[1,2]))
    found={};unexpected=[]
    for state in STATES:
        folder=ROOT/'DATA/labeled'/('helical '+state[1:])
        for path in sorted(folder.rglob('*')):
            if not path.is_file(): continue
            key=parse_name(path.name)
            if key is None or key[0]!=state: unexpected.append(dict(path=path.relative_to(ROOT).as_posix(),reason='unexpected filename/state'));continue
            found.setdefault(key,[]).append(path)
    duplicates=[dict(state=k[0],speed=k[1],load=k[2],repeat=k[3],paths=[p.relative_to(ROOT).as_posix() for p in ps]) for k,ps in found.items() if len(ps)>1]
    missing=[dict(state=k[0],speed=k[1],load=k[2],repeat=k[3]) for k in expected if k not in found]
    def inspect(item):
        key,path=item;data=np.loadtxt(path)
        assert data.ndim==2 and data.shape[1]==3 and np.isfinite(data).all(),str(path)
        state,speed,load,repeat=key
        return dict(recording=f'helical_{state[1:]}_{speed}hz_{load}_{repeat}',state=state,label=LABELS[state],
            speed=speed,load=load,repeat=repeat,relative_source=path.relative_to(ROOT).as_posix(),sha256=digest(path),samples=len(data),
            duration_seconds=len(data)/config['fs'],channels=3,analyzed_channels=['output_voltage','input_voltage'],finite=True)
    items=[(k,ps[0]) for k,ps in found.items() if len(ps)==1]
    with ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(inspect,items))
    rows.sort(key=lambda r:(r['state'],r['speed'],r['load'],r['repeat']))
    content_groups={}
    for r in rows:content_groups.setdefault(r['sha256'],[]).append(r['recording'])
    repeated_content=[v for v in content_groups.values() if len(v)>1]
    table(BASE/'diagnostics/recording_inventory.csv',rows);dump(BASE/'config/source_inventory.json',rows)
    summary=dict(expected_recordings=len(expected),found_unique_recordings=len(rows),missing_recordings=missing,
        duplicated_recordings=duplicates,duplicated_content_groups=repeated_content,unexpected_files=unexpected,min_samples=min(r['samples'] for r in rows),max_samples=max(r['samples'] for r in rows))
    dump(BASE/'diagnostics/inventory_summary.json',summary)
    (BASE/'REPORT.md').write_text('# Step 5 — đang chạy\n\nInventory: '+json.dumps(summary,ensure_ascii=False)+'\n\nConfiguration đã khóa theo Step 4. Chưa có kết luận robustness.\n',encoding='utf-8')
    print('INVENTORY',json.dumps(summary,ensure_ascii=False),flush=True)
    if duplicates: raise ValueError('Duplicate acquisition keys; ambiguous files excluded, resolve before analysis')
    if repeated_content: raise ValueError('Duplicate contents cannot be treated as independent recordings')
    return config,rows

def reuse_baseline(source,channel,config):
    if source['speed']!=50 or source['load']!='High': return None
    prior=[typed(r) for r in read_table(STEP4/'features/mfdfa_features.csv') if r['recording']==source['recording'] and r['channel']==channel]
    original=next(s for s in json.loads((STEP4/'config/source_inventory.json').read_text(encoding='utf-8')) if s['recording']==source['recording'])
    assert original['sha256']==source['sha256']
    rows=[]
    for row in prior:
        row.update(label=source['label'],speed=source['speed'],load=source['load'],source_sha256=source['sha256'],config_sha256=config['config_sha256'],
            status='ok',analysis_valid=bool(row['scaling_valid'] and row['decomposition_valid']),reused_step4=True)
        path=STEP4/'arrays'/row['pipeline']/(row['recording']+'_'+channel+'_'+row['component']+'.npz')
        row['array_path']=path.relative_to(ROOT).as_posix()
        # Re-check saved QC instead of trusting a CSV boolean.
        with np.load(path) as a: _,metrics=fit_scaling(a['scales'],a['Fq'],a['q'],config['fit_interval'])
        assert metrics['scaling_valid']==row['scaling_valid']
        rows.append(row)
    return rows

def signal_job(source,channel_info,config):
    column,channel,role=channel_info;stem=source['recording']+'_'+channel
    cached=BASE/'diagnostics/recordings'/(stem+'.json')
    signature=hashlib.sha256((config['config_sha256']+source['sha256']).encode()).hexdigest()
    if cached.exists():
        obj=json.loads(cached.read_text(encoding='utf-8'))
        if obj['signature']==signature and obj.get('array_hashes'):
            try: verify(obj['array_hashes']);return obj,True
            except RuntimeError: pass
    baseline=reuse_baseline(source,channel,config)
    if baseline is not None:
        obj=dict(signature=signature,features=baseline,decompositions=[],reused_step4=True,array_hashes=[dict(path=r['array_path'],sha256=digest(resolve(r['array_path']))) for r in baseline if r.get('array_path')]);dump(cached,obj);return obj,True
    data=np.loadtxt(resolve(source['relative_source']));raw=data[:,column];x=raw-raw.mean()
    assert len(raw)==source['samples']
    identity={k:source[k] for k in ['recording','state','label','speed','load','repeat','samples']}
    identity.update(channel=channel,role=role,source_sha256=source['sha256'],config_sha256=config['config_sha256'],reused_step4=False)
    features,decompositions=[],[]
    def extract(y,method,component,info,extra=None):
        row=dict(**identity,pipeline=method,component=component,**(extra or {}))
        row.update(decomposition_valid=info['decomposition_valid'],decomposition_kind=info['decomposition_kind'],reconstruction_error=info['reconstruction_error'])
        if 'strict_emd_imf_valid' in info:row['strict_emd_imf_valid']=info['strict_emd_imf_valid']
        path=BASE/'arrays'/method/(stem+'_'+component+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        try:
            scales,fq,_,_=mfdfa(y,np.array(config['q']),m=config['m'],scales=np.array(config['scales']))
            arrays,metrics=fit_scaling(scales,fq,np.array(config['q']),config['fit_interval'])
            np.savez_compressed(path,q=config['q'],scales=scales,Fq=fq,**arrays)
            row.update(**metrics,status='ok',analysis_valid=bool(metrics['scaling_valid'] and info['decomposition_valid']),array_path=path.relative_to(ROOT).as_posix())
        except ValueError as error:
            row.update(status='undefined',scaling_valid=False,analysis_valid=False,invalid_reason=str(error),array_path='')
        features.append(row)
    extract(raw,'raw','raw',dict(decomposition_valid=True,decomposition_kind='raw',reconstruction_error=0.))
    for method in ['vmd','emd']:
        start=time.perf_counter()
        if method=='vmd':
            modes,_,history,info=vmd(x,**config['vmd'],fs=config['fs'],return_info=True)
            order=np.argsort(history[-1]);modes=modes[order];centers=history[-1,order];residue=x-modes.sum(axis=0)
            converged=info['converged'];strict=None;kind='VMD'
        else:
            modes,residue,info=emd_long_record(x,**config['emd'])
            freqs=np.fft.rfftfreq(len(x),1/config['fs'])
            power=np.abs(np.fft.rfft(modes,axis=1))**2
            centers=np.sum(power*freqs,axis=1)/np.maximum(power.sum(axis=1),1e-300)
            order=np.argsort(-centers);modes=modes[order];centers=centers[order]
            converged=info.get('all_practical_modes_converged',False);strict=info.get('all_exact_one_count_conditions',False);kind='approximate_EMD'
        reconstruction=float(np.linalg.norm(x-modes.sum(axis=0))/np.linalg.norm(x))
        checks=float(np.linalg.norm(x-modes.sum(axis=0)-residue)/np.linalg.norm(x))
        assert checks<1e-12
        mode_rows,quality=mode_metrics(x,modes,centers)
        diag=dict(**identity,pipeline=method,decomposition_valid=bool(converged),decomposition_kind=kind,
            reconstruction_error=reconstruction,mode_plus_residue_error=checks,algorithm_info=info,
            mode_diagnostics=mode_rows,elapsed_seconds=time.perf_counter()-start,**quality)
        if strict is not None: diag['strict_emd_imf_valid']=strict
        decompositions.append(diag)
        for rank,(mode,center) in enumerate(zip(modes,centers),1):
            extract(mode,method,f'mode{rank}',diag,dict(rank=rank,center_hz=float(center),**{k:mode_rows[rank-1][k] for k in ['mode_energy','mode_energy_fraction','bandwidth_hz','correlation_mode_raw']}))
        extract(modes.sum(axis=0),method,'oscillatory_sum',diag)
        extract(residue,method,'residue',diag)
        # No large mode arrays survive this job: scalar decomposition audit and
        # complete MF-DFA arrays are retained, modes regenerate from locked config.
    obj=dict(signature=signature,features=features,decompositions=decompositions,reused_step4=False)
    obj['array_hashes']=[dict(path=r['array_path'],sha256=digest(resolve(r['array_path']))) for r in features if r.get('array_path')]
    dump(cached,obj);return obj,False

def run():
    if BASE==HISTORICAL_BASE: raise RuntimeError('Use execution.runner or PHM_OUTPUT_DIR: validated Step 5 outputs are read-only')
    accelerator.initialize(BASE/'_runtime')
    config,sources=inventory();features=[];decompositions=[]
    for method in ['raw','vmd','emd']:(BASE/'arrays'/method).mkdir(exist_ok=True)
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    jobs=list(itertools.product(sources,CHANNELS));start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(signal_job,source,channel,config) for source,channel in jobs]
        for done,future in enumerate(as_completed(futures),1):
            obj,cached=future.result();features.extend(obj['features']);decompositions.extend(obj['decompositions'])
            row=obj['features'][0]
            print(f'{done}/{len(jobs)} {row["recording"]} {row["channel"]}'+(' reused/cache' if cached else '')+f' elapsed={time.perf_counter()-start:.0f}s',flush=True)
    features.sort(key=lambda r:(r['recording'],r['channel'],r['pipeline'],r['component']))
    table(BASE/'features/mfdfa_features.csv',features);table(BASE/'diagnostics/mfdfa_quality_control.csv',features)
    table(BASE/'features/valid_features.csv',[r for r in features if r['analysis_valid']])
    dump(BASE/'diagnostics/decomposition_diagnostics.json',decompositions)
    print('EXTRACTION COMPLETE',len(features),'analyses',flush=True)

if __name__=='__main__':run()
