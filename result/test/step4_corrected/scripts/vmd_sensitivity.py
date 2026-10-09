"""Bounded, resumable VMD sensitivity on user-authorized fixed windows.

All grid configurations use the same first 65,536 samples of every signal.
Only scalar diagnostics are retained; full decompositions are saved later
for the selected configuration on the full original recordings.
No MF-DFA features or class separation enter parameter selection.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys
sys.dontwrite_bytecode=True
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'code/core'))
import csv
import json
import hashlib
import time
import warnings
import itertools
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from scipy.optimize import linear_sum_assignment
from vmd_accel import vmd

GRID=dict(K=[4,5,6,7,8],alpha=[500,1000,2000,4000],tau=[0.,.5],
          init=1,tol=1e-7,max_iter=1000,window_samples=65536,window_start=0,fs=66666.7)
SELECTION_RULE=dict(min_convergence_fraction=.90,max_p90_reconstruction_error=.15,
    max_median_repeat_center_relative_error=.15,max_median_duplicate_fraction=.30,
    duplicate_spectral_overlap=.80,minimum_mean_mode_energy_fraction=1e-4,
    ranking='eligible first; repeat center drift ascending, duplicate fraction ascending, reconstruction p90 ascending, K ascending, alpha ascending, tau ascending')


def dump(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def mode_metrics(x,modes,centers):
    frequencies=np.fft.rfftfreq(len(x),1/GRID['fs'])
    spectra=np.abs(np.fft.rfft(modes,axis=1))**2
    weights=np.full(len(frequencies),2.); weights[0]=1
    if len(x)%2==0: weights[-1]=1
    spectra*=weights
    energy=np.mean(modes**2,axis=1)
    raw_energy=np.mean(x**2)
    powers=spectra.sum(axis=1)
    actual_centers=np.sum(spectra*frequencies,axis=1)/np.maximum(powers,1e-300)
    bandwidth=np.sqrt(np.sum(spectra*(frequencies[None,:]-actual_centers[:,None])**2,axis=1)/np.maximum(powers,1e-300))
    correlations=[]
    for mode in modes:
        centered=mode-mode.mean()
        norm=np.linalg.norm(centered)*np.linalg.norm(x)
        correlations.append(float(np.dot(centered,x)/norm) if norm else 0.)
    normalized=np.sqrt(spectra)/np.sqrt(np.maximum(powers[:,None],1e-300))
    overlap=normalized@normalized.T
    pairs=[(i,j) for i in range(len(modes)) for j in range(i+1,len(modes))]
    duplicated=set(i for i,j in pairs if overlap[i,j]>.80)|set(j for i,j in pairs if overlap[i,j]>.80)
    rows=[dict(mode_rank=i+1,center_hz=float(centers[i]),observed_centroid_hz=float(actual_centers[i]),
        mode_energy=float(energy[i]),mode_energy_fraction=float(energy[i]/raw_energy),
        bandwidth_hz=float(bandwidth[i]),correlation_mode_raw=correlations[i],
        max_other_spectral_overlap=float(max(np.delete(overlap[i],i)))) for i in range(len(modes))]
    return rows,dict(duplicate_fraction=len(duplicated)/len(modes),
        max_spectral_overlap=float(max(overlap[i,j] for i,j in pairs)),
        minimum_mode_energy_fraction=float(np.min(energy/raw_energy)))


def job(params):
    K,alpha,tau=params
    key=f'K{K}_a{alpha}_t{tau:g}'
    path=BASE/'_work'/(key+'.json')
    inventory=json.loads((BASE/'config/source_inventory.json').read_text(encoding='utf-8'))
    signature=hashlib.sha256(json.dumps(GRID,sort_keys=True).encode()+(ROOT/'code/core/vmd.py').read_bytes()
        +json.dumps(inventory,sort_keys=True).encode()+(BASE/'scripts/vmd_kernel.cpp').read_bytes()).hexdigest()
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if saved['signature']==signature: return key,saved,True
    summaries,centers=[],[]
    with np.load(BASE/'_work/pilot_windows.npz') as windows:
        for source in inventory:
            rid=source['recording']
            for channel in ['output_voltage','input_voltage']:
                x=windows[rid+'_'+channel]
                start=time.perf_counter()
                modes,_,history,info=vmd(x,K=K,alpha=alpha,tau=tau,init=GRID['init'],
                    tol=GRID['tol'],max_iter=GRID['max_iter'],fs=GRID['fs'],return_info=True)
                order=np.argsort(history[-1]); modes=modes[order]; mode_centers=history[-1,order]
                identity=dict(configuration=key,K=K,alpha=alpha,tau=tau,recording=rid,channel=channel,
                    acquisition_pair=rid.rsplit('_',1)[0],repeat=source['repeat'],samples=len(x),
                    window_start_sample=0,scope='sensitivity_window',max_iter=GRID['max_iter'])
                mode_rows,quality=mode_metrics(x,modes,mode_centers)
                centers.extend(dict(**identity,**r) for r in mode_rows)
                summaries.append(dict(**identity,converged=bool(info['converged']),iterations=info['iterations'],
                    reconstruction_error=info['reconstruction_error'],constraint_error=info['constraint_error'],
                    relative_change=info['relative_change'],elapsed_seconds=time.perf_counter()-start,
                    warning='' if info['converged'] else 'VMD reached max_iter before convergence',
                    numeric_backend=info.get('numeric_backend','numpy_reference'),**quality))
    saved=dict(signature=signature,summaries=summaries,centers=centers)
    dump(path,saved)
    return key,saved,False


def aggregate(results):
    summaries=sorted([r for result in results for r in result['summaries']],
        key=lambda r:(r['configuration'],r['recording'],r['channel']))
    for row in summaries: row.setdefault('numeric_backend','numpy_reference')
    centers=sorted([r for result in results for r in result['centers']],
        key=lambda r:(r['configuration'],r['recording'],r['channel'],r['mode_rank']))
    stability=[]
    for key in sorted({r['configuration'] for r in summaries}):
        for pair in sorted({r['acquisition_pair'] for r in summaries}):
            for channel in ['output_voltage','input_voltage']:
                first=sorted([r for r in centers if r['configuration']==key and r['acquisition_pair']==pair and r['channel']==channel and r['repeat']==1],key=lambda r:r['mode_rank'])
                second=sorted([r for r in centers if r['configuration']==key and r['acquisition_pair']==pair and r['channel']==channel and r['repeat']==2],key=lambda r:r['mode_rank'])
                c1=np.array([r['center_hz'] for r in first]); c2=np.array([r['center_hz'] for r in second])
                cost=np.abs(np.log(np.maximum(c1[:,None],1)/np.maximum(c2[None,:],1)))
                ii,jj=linear_sum_assignment(cost)
                for i,j in zip(ii,jj):
                    drift=abs(c1[i]-c2[j])/max((c1[i]+c2[j])/2,1.)
                    stability.append(dict(configuration=key,acquisition_pair=pair,channel=channel,
                        repeat1_mode_rank=first[i]['mode_rank'],repeat2_mode_rank=second[j]['mode_rank'],
                        repeat1_center_hz=float(c1[i]),repeat2_center_hz=float(c2[j]),
                        relative_center_difference=float(drift),log_frequency_assignment_cost=float(cost[i,j]),
                        frequency_match_valid=bool(drift<=.15)))
    ranking=[]
    for key in sorted({r['configuration'] for r in summaries}):
        rows=[r for r in summaries if r['configuration']==key]
        matched=[r for r in stability if r['configuration']==key]
        cm=[r for r in centers if r['configuration']==key]
        metrics=dict(configuration=key,K=rows[0]['K'],alpha=rows[0]['alpha'],tau=rows[0]['tau'],
            signals=len(rows),convergence_fraction=float(np.mean([r['converged'] for r in rows])),
            median_iterations=float(np.median([r['iterations'] for r in rows])),
            median_reconstruction_error=float(np.median([r['reconstruction_error'] for r in rows])),
            p90_reconstruction_error=float(np.quantile([r['reconstruction_error'] for r in rows],.9)),
            max_reconstruction_error=float(max(r['reconstruction_error'] for r in rows)),
            median_constraint_error=float(np.median([r['constraint_error'] for r in rows])),
            median_repeat_center_relative_error=float(np.median([r['relative_center_difference'] for r in matched])),
            p90_repeat_center_relative_error=float(np.quantile([r['relative_center_difference'] for r in matched],.9)),
            matched_mode_fraction=float(np.mean([r['frequency_match_valid'] for r in matched])),
            median_duplicate_fraction=float(np.median([r['duplicate_fraction'] for r in rows])),
            minimum_mean_mode_energy_fraction=float(min(np.mean([r['mode_energy_fraction'] for r in cm if r['mode_rank']==rank]) for rank in range(1,rows[0]['K']+1))))
        reasons=[]
        for field,limit,direction in [('convergence_fraction',.9,'min'),('p90_reconstruction_error',.15,'max'),
            ('median_repeat_center_relative_error',.15,'max'),('median_duplicate_fraction',.30,'max'),
            ('minimum_mean_mode_energy_fraction',1e-4,'min')]:
            if (metrics[field]<limit if direction=='min' else metrics[field]>limit): reasons.append(field)
        metrics.update(eligible=not reasons,ineligible_reason=';'.join(reasons))
        ranking.append(metrics)
    ordered=sorted(ranking,key=lambda r:(not r['eligible'],r['median_repeat_center_relative_error'],
        r['median_duplicate_fraction'],r['p90_reconstruction_error'],r['K'],r['alpha'],r['tau']))
    for rank,row in enumerate(ordered,1): row['selection_rank']=rank
    chosen=ordered[0]
    selection=dict(selected_config={k:chosen[k] for k in ('K','alpha','tau')},
        init=1,tol=1e-7,sensitivity_max_iter=1000,full_record_max_iter=2000,
        eligible_configuration_found=bool(chosen['eligible']),selection_rule=SELECTION_RULE,
        selected_summary=chosen,grid=GRID,
        note='Selection uses only pooled convergence/reconstruction/frequency/overlap/energy diagnostics; no MF-DFA features or class separation. Repeat membership is acquisition pairing only.',
        full_record_confirmation_pending=True)
    table(BASE/'diagnostics/vmd_parameter_sensitivity.csv',summaries)
    table(BASE/'diagnostics/vmd_mode_centers.csv',centers)
    table(BASE/'diagnostics/vmd_repeat_frequency_stability.csv',stability)
    table(BASE/'summaries/vmd_reconstruction_summary.csv',ordered)
    dump(BASE/'config/selected_vmd_configuration.json',selection)
    return selection


def run():
    (BASE/'_work').mkdir(exist_ok=True)
    inventory=json.loads((BASE/'config/source_inventory.json').read_text(encoding='utf-8'))
    windows={}
    for source in inventory:
        assert hashlib.sha256((ROOT/source['relative_source']).read_bytes()).hexdigest()==source['sha256']
        data=np.loadtxt(ROOT/source['relative_source'],max_rows=GRID['window_samples'])
        for column,channel in [(1,'output_voltage'),(0,'input_voltage')]:
            x=data[:,column].copy(); x-=x.mean(); windows[source['recording']+'_'+channel]=x
    np.savez_compressed(BASE/'_work/pilot_windows.npz',**windows)
    dump(BASE/'config/vmd_grid.json',dict(grid=GRID,selection_rule=SELECTION_RULE,user_authorized_window=True))
    params=list(itertools.product(GRID['K'],GRID['alpha'],GRID['tau']))
    results=[]
    # NumPy releases the GIL in its numeric kernels. Threads avoid restricted
    # Windows multiprocessing IPC; each worker owns independent mode arrays.
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    print(f'Sensitivity: {len(params)} configs x 16 signals, 4 numeric worker threads',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(job,param):param for param in params}
        for future in as_completed(futures):
            key,result,cached=future.result(); results.append(result)
            print(f'{len(results)}/{len(params)} {key}: {sum(r["converged"] for r in result["summaries"])}/16 converged'+(' (cached)' if cached else ''),flush=True)
    selected=aggregate(results)
    print('Selected:',json.dumps(selected),flush=True)


if __name__=='__main__': run()
