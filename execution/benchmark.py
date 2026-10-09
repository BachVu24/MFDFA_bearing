"""Single full-recording benchmark; never writes scientific Step 4–5 results."""
import argparse, json, os, platform, subprocess, sys, threading, time, warnings
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import numpy as np
import psutil
from execution.provenance import ROOT, resolve, sha, atomic_json
STAGES=['raw_mfdfa','vmd_decomposition','vmd_mfdfa','emd_decomposition','emd_mfdfa']
def worker(stage,output):
    sys.path.insert(0,str(ROOT/'result/test/step5/scripts'))
    import pipeline as P
    config=P.lock_config();output=Path(output);runtime=P.accelerator.initialize(output/'_runtime')
    source=next(s for s in json.loads((P.HISTORICAL_BASE/'config/source_inventory.json').read_text(encoding='utf-8')) if s['recording']=='helical_1_50hz_High_1')
    p=resolve(source['relative_source']);verify_hash=sha(p)
    if verify_hash!=source['sha256']: raise RuntimeError('Raw data changed')
    start=time.perf_counter();cpu=time.process_time();data=np.loadtxt(p)
    read_wall=time.perf_counter()-start;read_cpu=time.process_time()-cpu
    raw=data[:,1];x=raw-raw.mean();peaks=[];stop=threading.Event();process=psutil.Process()
    def sample():
        while not stop.is_set(): peaks.append(process.memory_info().rss);stop.wait(.01)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    def feature(y):
        try:
            scales,fq,_,_=P.mfdfa(y,np.array(config['q']),m=config['m'],scales=np.array(config['scales']))
            arrays,metrics=P.fit_scaling(scales,fq,np.array(config['q']),config['fit_interval'])
            return dict(Fq=fq,**arrays),metrics
        except ValueError as error: return {},dict(scaling_valid=False,reason=str(error))
    start=time.perf_counter();cpu=time.process_time();results=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if stage=='raw_mfdfa': results=[feature(raw)]
        else:
            if stage.startswith('vmd'):
                modes,_,_,info=P.vmd(x,**config['vmd'],fs=config['fs'],return_info=True);residue=x-modes.sum(axis=0)
            else: modes,residue,info=P.emd_long_record(x,**config['emd'])
            if stage.endswith('mfdfa'): results=[feature(y) for y in [*modes,modes.sum(axis=0),residue]]
    compute_wall=time.perf_counter()-start;compute_cpu=time.process_time()-cpu
    stop.set();thread.join();payload={'raw':raw} if stage=='raw_mfdfa' else {'modes':modes,'residue':residue}
    start=time.perf_counter();cpu=time.process_time();temporary=output/(stage+'_io.npz');np.savez_compressed(temporary,**payload)
    write_wall=time.perf_counter()-start;write_cpu=time.process_time()-cpu;write_bytes=temporary.stat().st_size;temporary.unlink()
    result=dict(stage=stage,recording=source['recording'],channel='output_voltage',samples=len(raw),source_sha256=verify_hash,config_sha256=config['config_sha256'],
                backend=runtime['backend'],compute_wall_seconds=compute_wall,compute_cpu_seconds=compute_cpu,peak_rss_mib=max(peaks)/1024**2,
                peak_memory_method='10 ms RSS sampling in isolated stage subprocess; excludes parity warm-up',read_io_wall_seconds=read_wall,read_io_cpu_seconds=read_cpu,
                write_io_wall_seconds=write_wall,write_io_cpu_seconds=write_cpu,write_bytes=write_bytes,spectra_evaluated=len(results),valid_spectra=sum(bool(m.get('scaling_valid')) for _,m in results))
    atomic_json(output/(stage+'.json'),result)
def run_benchmark(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'result/test/colab_runs') and not output.is_relative_to(ROOT/'result/test/colab_migration'): raise ValueError('Dedicated benchmark output required')
    output.mkdir(parents=True,exist_ok=True)
    for stage in STAGES:
        print('BENCHMARK',stage,flush=True)
        subprocess.run([sys.executable,'-B','-m','execution.benchmark','--worker',stage,'--output',str(output)],cwd=ROOT,check=True)
    rows=[json.loads((output/(stage+'.json')).read_text(encoding='utf-8')) for stage in STAGES]
    atomic_json(output/'benchmark.json',dict(platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count(),python=sys.version,results=rows))
    lines=['# Benchmark: one full recording','',f'Platform: {platform.platform()}. CPU only; single numerical thread.','',
           'Recording: helical_1_50hz_High_1, column 2. Locked Step 4 configuration. Separate subprocess per workload; one measurement, not a performance distribution.',
           '', '| Workload | CPU s | Compute wall s | Peak RSS MiB | Read I/O s | Write I/O s |', '|---|---:|---:|---:|---:|---:|']
    for r in rows: lines.append('| '+r['stage']+' | '+' | '.join(f'{r[k]:.3f}' for k in ['compute_cpu_seconds','compute_wall_seconds','peak_rss_mib','read_io_wall_seconds','write_io_wall_seconds'])+' |')
    lines+=['','VMD/EMD + MF-DFA timings include decomposition, all modes, oscillatory_sum and residue. Peak RSS is sampled, not an exact allocation maximum. I/O measures local text loading and compressed decomposition serialization; Drive transfer is timed separately in the notebook.',
            '', 'No Colab speed comparison is available until this same benchmark runs there. Compilation/parity initialization is outside compute timing. No Step 4–5 outputs were replaced.']
    (output/'BENCHMARK.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=STAGES);p.add_argument('--output',required=True);a=p.parse_args()
    if a.worker: worker(a.worker,a.output)
    else: run_benchmark(a.output)
