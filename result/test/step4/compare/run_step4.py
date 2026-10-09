"""PHM2009 step 4. All writes stay in the four requested output folders.

Run with python -B .../run_step4.py raw|emd|vmd|compare|all.
Raw signals use their full original length. No resampling, clipping, smoothing,
amplitude normalization or record concatenation is applied to the baseline.
Mode pipelines remove the global mean only; MF-DFA does so internally too.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
sys.dontwrite_bytecode = True
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[2]
os.environ['MPLCONFIGDIR'] = str(BASE/'compare/mplconfig')
sys.path.insert(0, str(ROOT/'code/core'))
import csv
import hashlib
import json
import platform
import time
import warnings
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emd import emd
from vmd import vmd
from mfdfa import mfdfa, multifractal_spectrum
from read_viddata import extract
from emd_long_record import emd_long_record

FS = 66666.7
Q = np.arange(-5., 5.01, 0.5)
SCALES = np.unique(np.rint(np.geomspace(64, 8192, 40)).astype(int))
STATES = {1:'Healthy', 2:'Gear', 5:'Gear+Bearing', 6:'Shaft'}
CHANNELS = [(1, 'output_voltage', 'primary'), (0, 'input_voltage', 'secondary')]
COLORS = {1:'#2274a5', 2:'#e18d22', 5:'#bf3d56', 6:'#39935c'}
CONFIG = dict(fs_hz=FS, q=Q.tolist(), scales_samples=SCALES.tolist(), m=2,
    fit_range_samples=[64,8192], sensitivity_fit_range_samples=[256,4096],
    emd=dict(max_imfs=6,max_siftings=200,sd_threshold=0.2,envelope_tol=0.05,
             extrema_relative_tolerance=0.005),
    vmd=dict(K=6,alpha=2000.,tau=0.,DC=False,init=1,tol=1e-7,max_iter=2000),
    preprocessing='Raw: none outside MF-DFA. EMD/VMD: subtract global mean only.',
    mode_order='Descending spectral centroid; ranks do not establish physical equivalence.',
    alpha0_definition='alpha(q=0), derivative of tau at zero; f(alpha0)=1 by construction.',
    delta_alpha_definition='max(alpha)-min(alpha) on q=-5..5',
    delta_h_definition='max(h)-min(h) on q=-5..5',
    note='Nominal operating speed 50 Hz is not sample rate. fs comes from VidData B41.')
CONFIG_ID = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert), encoding='utf-8')


def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def recordings():
    for h in STATES:
        for repeat in (1,2):
            rid = f'helical_{h}_50hz_High_{repeat}'
            path = ROOT/'DATA/labeled'/f'helical {h}'/f'helical {h}_50hz_High_{repeat}.txt'
            yield h, repeat, rid, path


def analyze(x):
    scales, fq, h, tau, info = mfdfa(x, Q, m=2, scales=SCALES, return_info=True)
    alpha, spectrum = multifractal_spectrum(Q, tau)
    _, _, h_sens, tau_sens, sens = mfdfa(x, Q, m=2, scales=SCALES,
                                      fit_range=(256,4096), return_info=True)
    alpha_sens, spectrum_sens = multifractal_spectrum(Q, tau_sens)
    i0 = int(np.flatnonzero(Q == 0)[0])
    slopes = np.diff(np.log(fq), axis=1)/np.diff(np.log(scales))
    feature = dict(delta_alpha=float(np.ptp(alpha)), alpha0=float(alpha[i0]),
        delta_h=float(np.ptp(h)), h_minus5=float(h[0]), h_plus5=float(h[-1]),
        h2=float(h[np.flatnonzero(Q == 2)[0]]), f_alpha0=float(spectrum[i0]),
        min_r_squared=float(np.min(info['r_squared'])), median_r_squared=float(np.median(info['r_squared'])),
        tau_concavity_violations=int(np.sum(np.diff(np.diff(tau)/np.diff(Q)) > 1e-8)),
        alpha0_sensitivity=float(alpha_sens[i0]), delta_alpha_sensitivity=float(np.ptp(alpha_sens)),
        delta_h_sensitivity=float(np.ptp(h_sens)), sensitivity_min_r_squared=float(np.min(sens['r_squared'])))
    arrays = dict(q=Q, scales=scales, Fq=fq, hq=h, tau=tau, alpha=alpha, f_alpha=spectrum,
        r_squared=info['r_squared'], segment_counts=info['segment_counts'], local_slopes=slopes,
        sensitivity_hq=h_sens, sensitivity_tau=tau_sens, sensitivity_alpha=alpha_sens,
        sensitivity_f_alpha=spectrum_sens)
    return arrays, feature


def save_analysis(folder, stem, x, identity):
    try:
        arrays, feature = analyze(x)
        np.savez_compressed(folder/(stem+'.npz'), **arrays)
        rows = [dict(q=float(Q[i]), hq=float(arrays['hq'][i]), tau=float(arrays['tau'][i]),
                     alpha=float(arrays['alpha'][i]), f_alpha=float(arrays['f_alpha'][i]),
                     r_squared=float(arrays['r_squared'][i])) for i in range(len(Q))]
        write_csv(folder/(stem+'_spectrum.csv'), rows)
        return dict(**identity, status='ok', **feature)
    except ValueError as error:
        return dict(**identity, status='undefined', reason=str(error))


def run_raw():
    folder = BASE/'raw_mfdfa'
    inventory, rows = [], []
    viddata = extract(ROOT/'DATA/labeled/VidData.xls')
    write_json(folder/'viddata_extracted.json', viddata)
    for h, repeat, rid, path in recordings():
        x = np.loadtxt(path)
        assert x.ndim == 2 and x.shape[1] == 3 and np.isfinite(x).all()
        assert SCALES[-1] <= x.shape[0]//4
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory.append(dict(recording=rid, state=f'H{h}', label=STATES[h], repeat=repeat,
            relative_source=str(path.relative_to(ROOT)), sha256=source_hash, samples=len(x), columns=3,
            duration_seconds=len(x)/FS, finite=True, minimum=x.min(axis=0), maximum=x.max(axis=0),
            mean=x.mean(axis=0), std=x.std(axis=0),
            tachometer_rising_edges=int(np.sum((x[:-1,2] < 2.5)&(x[1:,2] >= 2.5)))))
        for column, channel, role in CHANNELS:
            identity = dict(recording=rid, state=f'H{h}', label=STATES[h], repeat=repeat,
                channel=channel, role=role, pipeline='raw', component='raw', samples=len(x))
            row = save_analysis(folder, rid+'_'+channel, x[:,column], identity)
            assert row['status'] == 'ok', row
            rows.append(row)
            print(f"RAW {rid} {channel}: width={row['delta_alpha']:.5f}, alpha0={row['alpha0']:.5f}, R2min={row['min_r_squared']:.4f}", flush=True)
    write_json(folder/'source_inventory.json', inventory)
    write_json(folder/'config.json', dict(config_id=CONFIG_ID, **CONFIG))
    write_csv(folder/'features.csv', rows)
    x = np.loadtxt(next(recordings())[3], max_rows=2000)
    fig, axes = plt.subplots(3,1,figsize=(11,7),sharex=True)
    for i, name in enumerate(['Input voltage (V)','Output voltage (V)','Tachometer (V)']):
        axes[i].plot(np.arange(len(x))/FS, x[:,i], lw=0.8)
        axes[i].set_ylabel(name)
    axes[0].set_title('H1 repeat 1: first 2,000 raw samples, three columns')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout(); fig.savefig(folder/'raw_structure.png',dpi=160); plt.close(fig)
    plot_pipeline('raw')


def centroid(mode):
    power = np.abs(np.fft.rfft(mode))**2
    return float(np.dot(np.fft.rfftfreq(len(mode),1/FS),power)/power.sum()) if power.sum() else 0.


def run_modes(method):
    folder = BASE/(method+'_mfdfa')
    features, diagnostics = [], []
    for h, repeat, rid, path in recordings():
        data = np.loadtxt(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for column, channel, role in CHANNELS:
            start = time.perf_counter()
            stem = rid+'_'+channel
            cache = folder/(stem+'_decomposition.npz')
            diag_path = folder/(stem+'_diagnostics.json')
            x = data[:,column]-data[:,column].mean()
            diag = json.loads(diag_path.read_text(encoding='utf-8')) if diag_path.exists() else None
            if (cache.exists() and diag and diag.get('config_id') == CONFIG_ID
                    and diag.get('source_sha256') == digest):
                with np.load(cache) as saved:
                    modes, residue, frequencies = saved['modes'], saved['residue'], saved['centers_hz']
                print(f'{method.upper()} CACHE {stem}',flush=True)
            else:
                print(f'{method.upper()} START {stem} N={len(x)}',flush=True)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if method == 'emd':
                        modes, residue, info = emd_long_record(x, **CONFIG['emd'])
                        if not info['all_practical_modes_converged']:
                            warnings.warn('EMD includes capped candidates; inspect each mode convergence',RuntimeWarning)
                        frequencies = np.array([centroid(mode) for mode in modes])
                        order = np.argsort(-frequencies)
                        modes, frequencies = modes[order], frequencies[order]
                        info['sorted_original_imf_indices'] = (order+1).tolist()
                    else:
                        modes, _, history, info = vmd(x, fs=FS, return_info=True, **CONFIG['vmd'])
                        order = np.argsort(-history[-1])
                        modes, frequencies = modes[order], history[-1,order]
                        residue = x-modes.sum(axis=0)
                diag = dict(recording=rid, state=f'H{h}', channel=channel, pipeline=method,
                    config_id=CONFIG_ID, source_sha256=digest, mode_count=len(modes), centers_hz=frequencies,
                    algorithm_info=info, warnings=[str(w.message) for w in caught],
                    reconstruction_error=float(np.linalg.norm(x-modes.sum(axis=0))/np.linalg.norm(x)),
                    identity_error=float(np.linalg.norm(x-modes.sum(axis=0)-residue)/np.linalg.norm(x)),
                    elapsed_decomposition_seconds=time.perf_counter()-start)
                np.savez_compressed(cache, modes=modes, residue=residue, centers_hz=frequencies,
                                    removed_mean=data[:,column].mean())
                write_json(diag_path,diag)
            diagnostics.append(diag)
            base = dict(recording=rid,state=f'H{h}',label=STATES[h],repeat=repeat,
                        channel=channel,role=role,pipeline=method,samples=len(x))
            for rank, mode in enumerate(modes,1):
                identity = dict(**base,component=f'mode{rank}',rank=rank,
                                center_hz=float(frequencies[rank-1]))
                features.append(save_analysis(folder,stem+f'_mode{rank}',mode,identity))
            features.append(save_analysis(folder,stem+'_oscillatory_sum',modes.sum(axis=0),
                                          dict(**base,component='oscillatory_sum')))
            features.append(save_analysis(folder,stem+'_residue',residue,dict(**base,component='residue')))
            write_csv(folder/'features.csv',features)
            write_json(folder/'decomposition_diagnostics.json',diagnostics)
            print(f'{method.upper()} DONE {stem}: {len(modes)} modes, {time.perf_counter()-start:.1f}s',flush=True)
    write_json(folder/'config.json',dict(config_id=CONFIG_ID,**CONFIG))
    plot_pipeline(method)


def load_features(method):
    with (BASE/(method+'_mfdfa')/'features.csv').open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def spectrum_path(method, row):
    name = row['recording']+'_'+row['channel']
    if method != 'raw': name += '_'+row['component']
    return BASE/(method+'_mfdfa')/(name+'.npz')


def plot_pipeline(method):
    folder = BASE/(method+'_mfdfa')
    rows = load_features(method)
    for _, channel, role in CHANNELS:
        components = ['raw'] if method == 'raw' else [f'mode{i}' for i in range(1,7)]+['oscillatory_sum']
        for component in components:
            subset = [r for r in rows if r['channel']==channel and r['component']==component and r['status']=='ok']
            if not subset: continue
            fig, axes = plt.subplots(1,3,figsize=(15,4.5))
            for row in subset:
                h, repeat = int(row['state'][1:]),int(row['repeat'])
                with np.load(spectrum_path(method,row)) as result:
                    label = f"{row['state']} {row['label']} r{repeat}"
                    style = '-' if repeat==1 else '--'
                    axes[0].plot(result['alpha'],result['f_alpha'],style,color=COLORS[h],label=label,lw=1.6)
                    axes[1].plot(Q,result['hq'],style,color=COLORS[h],lw=1.6)
                    axes[2].plot(Q,result['tau'],style,color=COLORS[h],lw=1.6)
            axes[0].set(xlabel='alpha',ylabel='f(alpha)',title='Multifractal spectrum')
            axes[1].set(xlabel='q',ylabel='h(q)',title='Generalized scaling exponents')
            axes[2].set(xlabel='q',ylabel='tau(q)',title='Mass exponents')
            for ax in axes: ax.grid(alpha=.25)
            axes[0].legend(fontsize=7)
            fig.suptitle(f'{method.upper()} / {component} / {channel} ({role}) — full recordings, common scales 64–8192')
            fig.tight_layout(); fig.savefig(folder/(component+'_'+channel+'_spectra.png'),dpi=160); plt.close(fig)
        if method=='raw':
            fig, axes=plt.subplots(2,4,figsize=(15,7))
            for ax,row in zip(axes.flat,[r for r in rows if r['channel']==channel]):
                with np.load(spectrum_path(method,row)) as result:
                    for qval in (-5,0,2,5):
                        qi=int(np.flatnonzero(Q==qval)[0])
                        ax.loglog(SCALES,result['Fq'][qi],'o-',ms=2,label=f'q={qval}')
                ax.set(title=row['state']+' r'+row['repeat'],xlabel='Scale s (samples)',ylabel='Fq(s)')
                ax.grid(alpha=.25); ax.legend(fontsize=7)
            fig.suptitle('Raw MF-DFA scaling curves / '+channel)
            fig.tight_layout(); fig.savefig(folder/('scaling_'+channel+'.png'),dpi=160); plt.close(fig)


def compare():
    folder=BASE/'compare'
    allrows=[r for method in ('raw','emd','vmd') for r in load_features(method)]
    diagnostic_map={}
    for method in ('emd','vmd'):
        diagnostic_rows=json.loads((BASE/(method+'_mfdfa')/'decomposition_diagnostics.json').read_text(encoding='utf-8'))
        for d in diagnostic_rows:
            diagnostic_map[(method,d['recording'],d['channel'])]=d
    for row in allrows:
        if row['status']=='ok':
            with np.load(spectrum_path(row['pipeline'],row)) as saved:
                row['minimum_hq']=float(np.min(saved['hq']))
                row['nonpositive_hq_count']=int(np.sum(saved['hq']<=0))
                row['power_law_fit_warning']=float(row['min_r_squared'])<0.95
        if row['pipeline']=='raw':
            row['solver_converged']=True
            row['mode_definition']='raw'
            continue
        d=diagnostic_map[(row['pipeline'],row['recording'],row['channel'])]
        info=d['algorithm_info']
        row['decomposition_error']=d['reconstruction_error']
        if row['pipeline']=='emd':
            row['mode_definition']='practical approximate EMD'
            row['solver_converged']=info['all_practical_modes_converged']
            row['exact_one_count_condition']=info['all_exact_one_count_conditions']
            if row['component'].startswith('mode'):
                original=info['sorted_original_imf_indices'][int(row['rank'])-1]-1
                quality=info['modes'][original]
                row['solver_converged']=quality['converged_practical_rule']
                row['exact_one_count_condition']=quality['exact_one_count_condition']
                row['count_mismatch_fraction']=quality['relative_count_mismatch']
                row['envelope_mean_ratio']=quality['envelope_mean_ratio']
        else:
            row['solver_converged']=info['converged']
            row['mode_definition']='VMD bandwidth-constrained mode'
    expected={(rid,channel) for _,_,rid,_ in recordings() for _,channel,_ in CHANNELS}
    for method in ('raw','emd','vmd'):
        actual={(r['recording'],r['channel']) for r in allrows if r['pipeline']==method}
        assert actual==expected,(method,'not all recording/channel combinations finished')
        if method!='raw':
            for rid,channel in expected:
                count=sum(r['pipeline']==method and r['recording']==rid and r['channel']==channel for r in allrows)
                assert count==diagnostic_map[(method,rid,channel)]['mode_count']+2
    write_csv(folder/'all_features.csv',allrows)
    metrics=['delta_alpha','alpha0','delta_h']
    pairs=[('H1','H2'),('H2','H5'),('H2','H6')]
    separation=[]
    for method in ('raw','emd','vmd'):
        components=['raw'] if method=='raw' else [f'mode{i}' for i in range(1,7)]+['oscillatory_sum']
        for _,channel,_ in CHANNELS:
            for component in components:
                subset=[r for r in allrows if r['pipeline']==method and r['channel']==channel
                        and r['component']==component and r['status']=='ok' and r['solver_converged']]
                for left,right in pairs:
                    a=[r for r in subset if r['state']==left]
                    b=[r for r in subset if r['state']==right]
                    if len(a)!=2 or len(b)!=2: continue
                    for metric in metrics:
                        av,bv=np.array([float(r[metric]) for r in a]),np.array([float(r[metric]) for r in b])
                        gap=float(max(av.min(),bv.min())-min(av.max(),bv.max()))
                        separation.append(dict(pipeline=method,channel=channel,component=component,
                            pair=left+' vs '+right,metric=metric,left_mean=float(av.mean()),right_mean=float(bv.mean()),
                            mean_difference=float(bv.mean()-av.mean()),left_repeat_range=float(np.ptp(av)),
                            right_repeat_range=float(np.ptp(bv)),nonoverlap_gap=gap,
                            observed_repeat_ranges_disjoint=bool(gap>0)))
    write_csv(folder/'state_pair_comparisons.csv',separation)
    changes=[]
    for _,channel,_ in CHANNELS:
        for h,repeat,rid,_ in recordings():
            rows=[r for r in allrows if r['recording']==rid and r['channel']==channel and r['status']=='ok']
            for first,second in [('raw','emd'),('emd','vmd'),('raw','vmd')]:
                components=[f'mode{i}' for i in range(1,7)]+['oscillatory_sum']
                for component in components:
                    aa=[r for r in rows if r['pipeline']==first and r['component']==('raw' if first=='raw' else component)]
                    bb=[r for r in rows if r['pipeline']==second and r['component']==component]
                    if len(aa)!=1 or len(bb)!=1: continue
                    row=dict(recording=rid,state=f'H{h}',repeat=repeat,channel=channel,
                             comparison=first+' vs '+second,component=component,
                             first_solver_converged=aa[0]['solver_converged'],
                             second_solver_converged=bb[0]['solver_converged'])
                    for metric in metrics:
                        row[metric+'_first']=float(aa[0][metric]); row[metric+'_second']=float(bb[0][metric])
                        row[metric+'_difference']=float(bb[0][metric])-float(aa[0][metric])
                    changes.append(row)
    write_csv(folder/'paired_pipeline_comparisons.csv',changes)
    for _,channel,_ in CHANNELS:
        fig,axes=plt.subplots(1,3,figsize=(14,4.5))
        for ax,metric in zip(axes,metrics):
            for offset,method in enumerate(('raw','emd','vmd')):
                component='raw' if method=='raw' else 'oscillatory_sum'
                values=[float(r[metric]) for r in allrows if r['pipeline']==method and
                        r['component']==component and r['channel']==channel and r['status']=='ok']
                ax.plot(np.arange(8),values,'o-',label=method.upper())
            ax.set_xticks(np.arange(8),['H1r1','H1r2','H2r1','H2r2','H5r1','H5r2','H6r1','H6r2'],rotation=40)
            ax.set_title(metric); ax.grid(alpha=.25); ax.legend()
        fig.suptitle('Recording features: raw vs sums of extracted modes / '+channel)
        fig.tight_layout(); fig.savefig(folder/('recording_features_'+channel+'.png'),dpi=160); plt.close(fig)
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for ax,(h,repeat,rid,_) in zip(axes.flat,recordings()):
            for method in ('raw','emd','vmd'):
                component='raw' if method=='raw' else 'oscillatory_sum'
                row=next(r for r in allrows if r['recording']==rid and r['channel']==channel
                         and r['pipeline']==method and r['component']==component)
                with np.load(spectrum_path(method,row)) as data:
                    ax.plot(data['alpha'],data['f_alpha'],label=method.upper())
            ax.set(title=f'H{h} r{repeat}',xlabel='alpha',ylabel='f(alpha)'); ax.legend(fontsize=7); ax.grid(alpha=.25)
        fig.suptitle('Raw vs EMD/VMD oscillatory sums / '+channel)
        fig.tight_layout(); fig.savefig(folder/('paired_spectra_'+channel+'.png'),dpi=160); plt.close(fig)
        for rank in range(1,7):
            fig,axes=plt.subplots(2,4,figsize=(15,7))
            for ax,(h,repeat,rid,_) in zip(axes.flat,recordings()):
                for method in ('raw','emd','vmd'):
                    component='raw' if method=='raw' else f'mode{rank}'
                    matched=[r for r in allrows if r['recording']==rid and r['channel']==channel
                             and r['pipeline']==method and r['component']==component and r['status']=='ok']
                    if not matched: continue
                    row=matched[0]
                    with np.load(spectrum_path(method,row)) as data:
                        label=method.upper() if method=='raw' else f"{method.upper()} {float(row['center_hz']):.0f} Hz"
                        ax.plot(data['alpha'],data['f_alpha'],label=label)
                ax.set(title=f'H{h} r{repeat}',xlabel='alpha',ylabel='f(alpha)')
                ax.legend(fontsize=6); ax.grid(alpha=.25)
            fig.suptitle(f'Raw and individual modes at frequency rank {rank} / {channel}; ranks are not physical matches')
            fig.tight_layout(); fig.savefig(folder/(f'paired_rank{rank}_'+channel+'.png'),dpi=160); plt.close(fig)
    validation=[]
    for row in allrows:
        if row['status']!='ok': continue
        with np.load(spectrum_path(row['pipeline'],row)) as data:
            assert np.array_equal(data['q'],Q) and np.array_equal(data['scales'],SCALES)
            assert all(np.isfinite(data[name]).all() for name in ('Fq','hq','tau','alpha','f_alpha'))
            assert np.all(data['Fq']>0)
            assert np.allclose(data['tau'],Q*data['hq']-1)
            assert abs(data['f_alpha'][10]-1)<1e-12
            assert np.array_equal(data['segment_counts'],2*(int(row['samples'])//SCALES))
        validation.append(dict(recording=row['recording'],channel=row['channel'],pipeline=row['pipeline'],component=row['component'],passed=True))
    inventory=json.loads((BASE/'raw_mfdfa/source_inventory.json').read_text(encoding='utf-8'))
    for row in inventory:
        assert hashlib.sha256((ROOT/row['relative_source']).read_bytes()).hexdigest()==row['sha256']
    write_json(folder/'validation.json',dict(analyses_checked=len(validation),all_passed=True,
        source_files_unchanged=True,checks=['common q/scales','finite arrays','positive Fq',
            'tau=q*h-1','f(alpha0)=1','two-end segment counts','source SHA256'],results=validation))
    solver_summary=[]
    for d in diagnostic_map.values():
        with np.load(BASE/(d['pipeline']+'_mfdfa')/(d['recording']+'_'+d['channel']+'_decomposition.npz')) as saved:
            assert saved['modes'].shape==(d['mode_count'],next(r['samples'] for r in inventory if r['recording']==d['recording']))
            path=next(p for _,_,rid,p in recordings() if rid==d['recording'])
            column=1 if d['channel']=='output_voltage' else 0
            x=np.loadtxt(path)[:,column]; x-=x.mean()
            identity_error=float(np.linalg.norm(x-saved['modes'].sum(axis=0)-saved['residue'])/np.linalg.norm(x))
            assert identity_error<1e-12
            assert np.isfinite(saved['modes']).all() and np.all(np.diff(saved['centers_hz'])<=0)
        info=d['algorithm_info']
        solver_summary.append(dict(recording=d['recording'],channel=d['channel'],pipeline=d['pipeline'],
            solver_converged=info.get('converged',info.get('all_practical_modes_converged')),
            exact_emd_count_conditions=info.get('all_exact_one_count_conditions'),
            mode_count=d['mode_count'],identity_error=identity_error,reconstruction_error=d['reconstruction_error']))
    write_json(folder/'decomposition_validation.json',dict(all_identities_passed=True,results=solver_summary))
    write_json(folder/'analysis_manifest.json',dict(config=CONFIG,
        baseline_config_id=json.loads((BASE/'raw_mfdfa/config.json').read_text(encoding='utf-8'))['config_id'],
        decomposition_config_ids={m:sorted(set(d['config_id'] for d in diagnostic_map.values() if d['pipeline']==m)) for m in ('emd','vmd')},
        core_sha256={name:hashlib.sha256((ROOT/'code/core'/name).read_bytes()).hexdigest() for name in ('emd.py','vmd.py','mfdfa.py')},
        viddata_sha256=hashlib.sha256((ROOT/'DATA/labeled/VidData.xls').read_bytes()).hexdigest(),
        scripts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.py')}))
    write_report(allrows,separation)


def write_report(rows,separation):
    primary=[r for r in rows if r['pipeline']=='raw' and r['channel']=='output_voltage']
    text=['# Bước 4: PHM2009 raw, EMD, VMD và MF-DFA','',
          'Dữ liệu: 8 recording Helical, 50 Hz, High load; cột 2 chính, cột 1 đối chiếu.',
          'Giữ nguyên toàn bộ mỗi recording. Mỗi recording là một đơn vị so sánh độc lập, không nối dữ liệu.', '',
          '**Lưu ý từ dữ liệu thực:** các đường Fq(s) đổi độ dốc và bão hòa. Fit 64–8.192 mẫu không phải một vùng power-law rõ ràng trên mọi recording. Đọc các chỉ số như đặc trưng thực nghiệm của cấu hình này, chưa coi chúng là bằng chứng multifractality ổn định.', '',
          '## Raw MF-DFA: cột 2', '',
          '| Recording | Δα | α0 | Δh | min R² |', '|---|---:|---:|---:|---:|']
    for r in primary:
        text.append(f"| {r['recording']} | {float(r['delta_alpha']):.6f} | {float(r['alpha0']):.6f} | {float(r['delta_h']):.6f} | {float(r['min_r_squared']):.4f} |")
    maximum_alpha_shift=max(abs(float(r['alpha0_sensitivity'])-float(r['alpha0'])) for r in primary)
    text+=['',f"Đổi vùng fit từ 64–8.192 sang 256–4.096 mẫu làm α0 cột 2 thay đổi tối đa {maximum_alpha_shift:.4f}. Do đó vị trí đỉnh phổ nhạy đáng kể với vùng scaling.",
           '','![Raw output spectra](../raw_mfdfa/raw_output_voltage_spectra.png)','',
           '## Khác biệt giữa trạng thái ở baseline', '',
           'Dưới đây chỉ mô tả hai lần đo mỗi trạng thái. Hai khoảng giá trị không giao nhau chưa phải bằng chứng thống kê về khả năng chẩn đoán tổng quát.', '']
    for pair in ['H1 vs H2','H2 vs H5','H2 vs H6']:
        selected=[r for r in separation if r['pipeline']=='raw' and r['channel']=='output_voltage' and r['pair']==pair]
        disjoint=[r['metric'] for r in selected if r['observed_repeat_ranges_disjoint']]
        text.append(f"- {pair}: khoảng của hai lần đo tách nhau ở {', '.join(disjoint) if disjoint else 'không chỉ số nào trong Δα, α0, Δh'}.")
    text+=['','## So sánh EMD với raw và VMD', '',
           'Một số mode hẹp băng có h(q)≤0 trên dải fit, và Fq(s) không biểu hiện một power-law dương ổn định. MF-DFA chuẩn có giới hạn trong trường hợp h gần zero/âm (Kantelhardt, mục 2, trước công thức 7). Các f(alpha) này được lưu đúng phép tính nhưng không diễn giải thành phổ kỳ dị vật lý đã được xác nhận. all_features.csv ghi minimum_hq, nonpositive_hq_count và cảnh báo fit.', '',
           'Các bảng so sánh dùng đúng cùng recording/kênh/q/scales. Mỗi mode, tổng mode và residual đều lưu riêng. Bảng dưới đếm số chỉ số có khoảng hai lần đo không giao nhau (0–3), chỉ mang tính mô tả trên tám recordings.', '',
           '| Component | Pipeline | H1–H2 | H2–H5 | H2–H6 |', '|---|---|---:|---:|---:|']
    for component in ['raw']+[f'mode{i}' for i in range(1,7)]+['oscillatory_sum']:
        for method in (['raw'] if component=='raw' else ['emd','vmd']):
            counts=[]
            for pair in ['H1 vs H2','H2 vs H5','H2 vs H6']:
                selected=[r for r in separation if r['pipeline']==method and r['channel']=='output_voltage'
                          and r['component']==component and r['pair']==pair]
                counts.append(str(sum(r['observed_repeat_ranges_disjoint'] for r in selected)) if len(selected)==3 else 'n.a.')
            text.append('| '+component+' | '+method.upper()+' | '+' | '.join(counts)+' |')
    text+=['','![Các chỉ số theo recording](recording_features_output_voltage.png)', '',
           '![So sánh phổ theo recording](paired_spectra_output_voltage.png)', '',
           'paired_pipeline_comparisons.csv ghi chênh Δα, α0, Δh từng recording cho raw–EMD, EMD–VMD và raw–VMD, cả từng mode lẫn tổng mode. Các hình paired_rank* đặt phổ raw/EMD/VMD cùng một panel theo rank tần số.', '',
           'Bảng đếm trên không đủ để xếp hạng giải thuật: mode ở cùng rank có thể khác dải tần, có mode h(q)≤0/fit kém, VMD tau=0 chủ động loại một phần tín hiệu, và chỉ có hai recording mỗi trạng thái.', '']
    diagnostics=[]
    for method in ('emd','vmd'):
        diagnostics += json.loads((BASE/(method+'_mfdfa')/'decomposition_diagnostics.json').read_text(encoding='utf-8'))
    text+=['## Trạng thái decomposition', '', '| Pipeline | Recording/kênh | Mode | Hội tụ/dừng | Sai số tổng mode so với raw đã trừ mean |',
           '|---|---|---:|---|---:|']
    for d in diagnostics:
        info=d['algorithm_info']
        status=info.get('stop_reason', 'converged' if info.get('converged') else 'NOT CONVERGED')
        text.append(f"| {d['pipeline'].upper()} | {d['recording']} / {d['channel']} | {d['mode_count']} | {status} | {d['reconstruction_error']:.6f} |")
    warning_cases=[d for d in diagnostics if d['warnings']]
    if warning_cases:
        text+=['', 'Các trường hợp có cảnh báo được giữ nguyên trong diagnostics, không đánh dấu là hội tụ:']
        for d in warning_cases:
            text.append(f"- {d['pipeline']} {d['recording']} {d['channel']}: {'; '.join(d['warnings'])}")
    text+=['','## Thiết lập chung và diễn giải', '',
        '- q = -5, -4.5, ..., 5; DFA2; 40 scales log từ 64 đến 8.192 mẫu cho tất cả recordings và components.',
        '- MF-DFA sử dụng toàn bộ tín hiệu, trừ mean khi tạo profile và detrending từng đoạn từ cả hai đầu. Không resample, lọc, chuẩn hóa biên độ hoặc cắt các recording về cùng chiều dài.',
        '- α0 = α(q=0), tương ứng f(α0)=1 theo công thức tau(0)=-1. Δα=max(α)-min(α), Δh=max(h)-min(h) trên dải q đã cho; không ngoại suy ra q vô hạn.',
        '- Đã tính thêm cùng tập scales với vùng fit 256–4.096 mẫu, lưu các chỉ số *_sensitivity. Xem Fq(s), R² và tính nhạy theo vùng fit trước khi diễn giải phổ như một luật scaling duy nhất.',
        '- EMD/VMD chỉ trừ global mean trước decomposition. EMD dùng biến thể dừng thực hành trong compare/emd_long_record.py: tối đa 6 mode, relative energy SD≤0,2, mean envelope RMS ratio≤0,05, chênh cực trị/đổi dấu≤0,5%, cap 200 sifting. Báo riêng điều kiện chính xác chênh≤1 và cờ hội tụ cho mỗi mode.',
        '- Chạy thử EMD gốc (core, pointwise SD tổng≤0,2 và count mismatch≤1) trên H1 output không hội tụ sau 1.000 sifting và trả 0 IMF. Chi tiết được giữ tại strict_emd_pilot.json. Cấu hình thực hành giữ các mode xấp xỉ; chúng không được trình bày như IMF Huang chính xác. Candidate chạm cap vẫn được lưu/phân tích với cờ chưa hội tụ, không che bỏ.',
        '- VMD cố định K=6, alpha=2.000, tau=0, init=1, tol=1e-7, cap 2.000. tau=0 là denoising: tổng mode có thể khác raw, nên giữ và báo residual, reconstruction error và convergence.',
        '- Mỗi mode chạy MF-DFA riêng. Đánh số mode theo tần số centroid giảm dần; cùng rank không bảo đảm cùng cơ chế vật lý giữa EMD và VMD hoặc giữa trạng thái.',
        '- oscillatory_sum = tổng các mode đã trích, rồi chạy lại MF-DFA trên tổng này. Không cộng/trung bình Fq, h hoặc f(alpha) của các mode. So sánh recording raw–EMD–VMD ở biểu đồ tổng này.',
        '- EMD oscillatory_sum chỉ gồm sáu mode đầu, nên không phải tái tạo đầy đủ raw; phần thấp tần còn lại ở residue. VMD sum cũng khác raw vì tau=0. Sai số bảng decomposition là sai số tổng mode, còn mode + residue tái tạo tín hiệu đã trừ mean trong giới hạn làm tròn.',
        '- residue được phân tích riêng khi scaling xác định; undefined được ghi rõ trong features.csv. Không tự gán chỉ số cho chuỗi zero/đã detrend hoàn toàn.',
        '- Không chọn K, scales hoặc mode bằng cách tối đa hóa khác biệt giữa nhãn. Đây là baseline với cấu hình chung, chưa phải đánh giá classifier.', '',
        '## Cấu trúc và nhãn dataset', '',
        'VidData.xls / Sheet1: A17:M22 mô tả trạng thái Helical; B41 ghi sample rate 66.6667 KHz; B34 ghi mỗi acquisition 4 Seconds; A35 ghi hai acquisitions mỗi operating speed/load.',
        'H1: tất cả Good. H2: gear 24T Chipped. H5: gear 24T Broken và bearing ID:OS Inner. H6: Input shaft Bent Shaft. Vì mức hỏng gear H2 và H5 khác nhau, H2↔H5 không cô lập riêng tác động thêm bearing.',
        'Các file TXT không có header, có ba cột số. Tên cột input/output voltage/tachometer theo mô tả người dùng; VidData còn mô tả accelerometers Endevco 6259M31, 10mv/g. Giữ đơn vị voltage, không đổi sang g khi chưa có đầy đủ calibration.',
        'Bảy file dài 266.656 mẫu (xấp xỉ 4 s), H5 repeat 2 dài 245.648 mẫu (xấp xỉ 3,685 s). Số mẫu thực tế được giữ nguyên. 50 Hz là operating speed, không phải sample rate.', '',
        '## File kết quả và chạy lại', '',
        '- raw_mfdfa: features.csv (16 hàng), NPZ/CSV phổ từng recording/kênh, spectrum và scaling plots, source_inventory.json, viddata_extracted.json, config.json.',
        '- emd_mfdfa, vmd_mfdfa: decomposition NPZ, diagnostics JSON, MF-DFA NPZ/CSV mỗi mode/sum/residue, features.csv, plots theo mode và kênh.',
        '- compare: all_features.csv, paired_pipeline_comparisons.csv (EMD–raw, EMD–VMD và raw–VMD theo từng recording), state_pair_comparisons.csv, validation.json và báo cáo này.',
        '- Dữ liệu nguồn và code/core chỉ được đọc. SHA256 tám file raw được kiểm tra lại sau phân tích.',
        '- raw_formula_validation.json: 27 phép đối chiếu Fq(s) trên dữ liệu thật bằng vòng lặp polyfit độc lập; practical_emd_tests.json: 4 kiểm thử biến thể EMD; validation.json và decomposition_validation.json: kiểm tra từng kết quả và đẳng thức mode + residue.', '',
        '```powershell', 'python -B result\\test\\step4\\compare\\run_step4.py all', '```',
        'Có thể chạy riêng raw, emd, vmd hoặc compare. Decomposition cache chỉ được dùng khi hash dữ liệu và toàn bộ cấu hình khớp.', '',
        f'Môi trường: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}.']
    (BASE/'compare/REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    for folder in ('raw_mfdfa','emd_mfdfa','vmd_mfdfa','compare'):
        (BASE/folder).mkdir(parents=True,exist_ok=True)
    stage=sys.argv[1] if len(sys.argv)>1 else 'all'
    if stage not in ('raw','emd','vmd','compare','all'): raise ValueError('Unknown stage')
    if stage in ('raw','all'): run_raw()
    if stage in ('emd','all'): run_modes('emd')
    if stage in ('vmd','all'): run_modes('vmd')
    if stage in ('compare','all'): compare()
