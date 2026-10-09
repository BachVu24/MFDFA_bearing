"""Corrected PHM2009 experiment; generated outputs live under step4_corrected.

Stages: scaling, finalize. VMD grid is run by vmd_sensitivity.py between them.
MF-DFA core formulas are unchanged. Existing raw/EMD Fq arrays are reused.
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
OLD=ROOT/'result/test/step4'
os.environ['MPLCONFIGDIR']=str(BASE/'_work/mplconfig')
sys.path.insert(0,str(ROOT/'code/core'))
import json
import csv
import hashlib
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from mfdfa import mfdfa
from vmd_accel import vmd
from scaling import CANDIDATES,RULE,fit_scaling,select_common
from vmd_sensitivity import GRID,mode_metrics,table,dump

CHANNELS=[(1,'output_voltage','primary'),(0,'input_voltage','secondary')]
COLORS={'H1':'#2274a5','H2':'#e18d22','H5':'#bf3d56','H6':'#39935c'}
FIXED_BANDS=[(0,1000),(1000,4000),(4000,10000),(10000,20000),(20000,33334)]


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))


def prepare():
    for directory in ['config','diagnostics','features','figures','summaries','arrays','decompositions','scripts','_work']:
        (BASE/directory).mkdir(exist_ok=True)
    inventory=json.loads((OLD/'raw_mfdfa/source_inventory.json').read_text(encoding='utf-8'))
    for source in inventory:
        assert hashlib.sha256((ROOT/source['relative_source']).read_bytes()).hexdigest()==source['sha256']
    dump(BASE/'config/source_inventory.json',inventory)
    template=next(legacy_entries())[1]
    dump(BASE/'config/mfdfa_configuration.json',dict(q=template['q'].tolist(),scales=template['scales'].tolist(),
        detrending_order=2,profile='cumulative sum of mean-removed component',
        segmentation='non-overlapping forward and backward segments',
        q_zero='geometric mean of segment RMS fluctuations',fit='OLS log Fq against log scale inside one common interval',
        fs_hz=GRID['fs'],channels=CHANNELS,
        emd_source='result/test/step4/compare/emd_long_record.py',emd_kind='approximate practical EMD, six modes; strict IMF check reported separately'))
    dump(BASE/'config/mfdfa_quality_rules.json',dict(rule=RULE,
        rationale='Positive power-law scaling, sufficient regression points, high R2 coverage, stable adjacent slopes; fixed before selection and independent of labels.',
        local_slope_cv_definition='std(adjacent log-log slopes)/abs(mean); median across q is reported, at least 80% q must have CV<=0.5',
        alpha0_definition='alpha(q=0), numerical derivative of tau at zero',
        width_definition='delta_alpha=max(alpha)-min(alpha); delta_h=max(hq)-min(hq); audit-only when scaling_valid=False',
        bands_hz=FIXED_BANDS,
        band_selection='At most one representative per fixed frequency band: largest mode energy, label-independent; all four repeat centers in a comparison must span a ratio <=1.5. Mode indices are never cross-record matching criteria.'))


def legacy_entries():
    for method in ['raw','emd','vmd']:
        for row in read_csv(OLD/(method+'_mfdfa')/'features.csv'):
            if row['status']!='ok': continue
            identity={k:row[k] for k in ['recording','state','label','repeat','channel','role','pipeline','component','samples']}
            identity['repeat']=int(identity['repeat']); identity['samples']=int(identity['samples'])
            for key in ['rank','center_hz']:
                if row.get(key): identity[key]=float(row[key]) if key=='center_hz' else int(row[key])
            stem=row['recording']+'_'+row['channel']
            if method!='raw': stem+='_'+row['component']
            path=OLD/(method+'_mfdfa')/(stem+'.npz')
            with np.load(path) as saved:
                numerical={key:saved[key].copy() for key in ['q','scales','Fq']}
            yield identity,numerical,path


def corrected_path(row):
    name=row['recording']+'_'+row['channel']+'_'+row['component']+'.npz'
    return BASE/'arrays'/row['pipeline']/name


def apply_qc(entries,bounds):
    rows=[]
    for identity,numerical,source_path in entries:
        arrays,metrics=fit_scaling(numerical['scales'],numerical['Fq'],numerical['q'],bounds)
        target=corrected_path(identity); target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target,**numerical,**arrays)
        row=dict(**identity,**metrics)
        method=identity['pipeline']
        if method=='raw': row.update(decomposition_valid=True,decomposition_kind='raw',reconstruction_error=0.)
        elif method=='emd':
            diag=json.loads((OLD/'emd_mfdfa'/(identity['recording']+'_'+identity['channel']+'_diagnostics.json')).read_text(encoding='utf-8'))
            info=diag['algorithm_info']
            row.update(decomposition_valid=info['all_practical_modes_converged'],
                strict_emd_imf_valid=info['all_exact_one_count_conditions'],decomposition_kind='approximate_EMD',
                reconstruction_error=diag['reconstruction_error'])
        else:
            if source_path is not None and 'step4_corrected' not in str(source_path):
                diag=json.loads((OLD/'vmd_mfdfa'/(identity['recording']+'_'+identity['channel']+'_diagnostics.json')).read_text(encoding='utf-8'))
            else:
                diag=json.loads((BASE/'decompositions/vmd'/(identity['recording']+'_'+identity['channel']+'_diagnostics.json')).read_text(encoding='utf-8'))
            row.update(decomposition_valid=diag['algorithm_info']['converged'],decomposition_kind='VMD',
                reconstruction_error=diag['reconstruction_error'])
        rows.append(row)
    table(BASE/'features/mfdfa_features.csv',rows)
    table(BASE/'diagnostics/mfdfa_quality_control.csv',rows)
    summary=[]
    for method in ['raw','emd','vmd']:
        for _,channel,_ in CHANNELS:
            method_rows=[r for r in rows if r['pipeline']==method and r['channel']==channel]
            for component_type in ['all','individual_modes','raw_or_sum','residue']:
                subset=[r for r in method_rows if component_type=='all' or
                    (component_type=='individual_modes' and r['component'].startswith('mode')) or
                    (component_type=='raw_or_sum' and r['component'] in ['raw','oscillatory_sum']) or
                    (component_type=='residue' and r['component']=='residue')]
                if subset:
                    summary.append(dict(pipeline=method,channel=channel,component_type=component_type,
                        total=len(subset),scaling_valid=sum(r['scaling_valid'] for r in subset),
                        decomposition_valid=sum(r['decomposition_valid'] for r in subset),
                        both_valid=sum(r['scaling_valid'] and r['decomposition_valid'] for r in subset)))
    table(BASE/'summaries/valid_component_summary.csv',summary)
    return rows


def scaling():
    prepare()
    entries=list(legacy_entries())
    diagnostic_rows,raw_quality=[],[]
    for identity,numerical,_ in entries:
        for low,high in CANDIDATES:
            arrays,metrics=fit_scaling(numerical['scales'],numerical['Fq'],numerical['q'],(low,high))
            common=dict(**identity,candidate_min=low,candidate_max=high)
            if identity['pipeline']=='raw': raw_quality.append(dict(**common,**metrics))
            for qi,q in enumerate(numerical['q']):
                diagnostic_rows.append(dict(**common,q=float(q),slope=float(arrays['hq'][qi]),
                    r2=float(arrays['r_squared'][qi]),scale_points=metrics['scale_points'],
                    local_slope_mean=float(arrays['local_slope_mean'][qi]),
                    local_slope_std=float(arrays['local_slope_std'][qi]),
                    local_slope_cv=float(arrays['local_slope_cv'][qi]),
                    r2_pass=bool(arrays['r_squared'][qi]>=.95),
                    local_stability_pass=bool(arrays['local_slope_cv'][qi]<=.5)))
    selection,summary=select_common(raw_quality)
    dump(BASE/'config/selected_scaling_range.json',selection)
    table(BASE/'diagnostics/scaling_range_diagnostics.csv',diagnostic_rows)
    table(BASE/'summaries/scaling_range_summary.csv',summary)
    rows=apply_qc(entries,selection['selected_interval'])
    diagnostic_plots(rows,selection['selected_interval'])
    plot_spectra(rows)
    interim_report(rows,selection)
    print('Selected scaling:',json.dumps(selection),flush=True)
    print('QC counts:',{m:sum(r['scaling_valid'] for r in rows if r['pipeline']==m) for m in ['raw','emd','vmd']},flush=True)


def diagnostic_plots(rows,bounds):
    for _,channel,_ in CHANNELS:
        raw=[r for r in rows if r['pipeline']=='raw' and r['channel']==channel]
        for kind in ['scaling','local_slopes']:
            fig,axes=plt.subplots(2,4,figsize=(15,7))
            for ax,row in zip(axes.flat,raw):
                with np.load(corrected_path(row)) as a:
                    for q in [-5.,0.,2.,5.]:
                        qi=int(np.flatnonzero(a['q']==q)[0])
                        if kind=='scaling': ax.loglog(a['scales'],a['Fq'][qi],'o-',ms=2,label=f'q={q:g}')
                        else:
                            local=np.diff(np.log(a['Fq'][qi]))/np.diff(np.log(a['scales']))
                            centers=np.sqrt(a['scales'][:-1]*a['scales'][1:])
                            ax.semilogx(centers,local,'o-',ms=2,label=f'q={q:g}')
                    ax.axvspan(bounds[0],bounds[1],color='#dfe9a4',alpha=.45,label='selected fit')
                ax.set(title=row['state']+' r'+str(row['repeat']),xlabel='s (samples)',
                       ylabel='Fq(s)' if kind=='scaling' else 'Adjacent log-log slope')
                ax.grid(alpha=.2); ax.legend(fontsize=6)
            fig.suptitle(f'Raw {channel}: common interval {bounds[0]}–{bounds[1]} samples')
            fig.tight_layout(); fig.savefig(BASE/'figures'/(kind+'_'+channel+'.png'),dpi=150); plt.close(fig)
        heat=[]
        for row in raw:
            with np.load(corrected_path(row)) as a: heat.append(a['r_squared'])
        fig,ax=plt.subplots(figsize=(10,4))
        image=ax.imshow(heat,aspect='auto',vmin=.8,vmax=1,cmap='viridis')
        ax.set_xticks(np.arange(21)[::2],np.arange(-5,5.1,.5)[::2])
        ax.set_yticks(range(8),[r['state']+' r'+str(r['repeat']) for r in raw])
        ax.set(xlabel='q',title='Selected range R² / '+channel)
        fig.colorbar(image,ax=ax,label='R²'); fig.tight_layout()
        fig.savefig(BASE/'figures'/('r2_heatmap_'+channel+'.png'),dpi=150); plt.close(fig)


def plot_spectra(rows):
    for method in ['raw','emd','vmd']:
        for _,channel,_ in CHANNELS:
            subset=[r for r in rows if r['pipeline']==method and r['channel']==channel]
            if method=='raw':
                fig,ax=plt.subplots(figsize=(9,6))
                for row in subset:
                    with np.load(corrected_path(row)) as a:
                        valid=row['scaling_valid']; style='-' if row['repeat']==1 else '--'
                        ax.plot(a['alpha'],a['f_alpha'],style if valid else '--',
                            color=COLORS[row['state']] if valid else '#999999',
                            label=row['state']+' r'+str(row['repeat'])+(' / UNRELIABLE SCALING' if not valid else ''))
                ax.set(xlabel='alpha',ylabel='f(alpha)',title=method.upper()+' / '+channel)
                ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
                fig.savefig(BASE/'figures'/(method+'_'+channel+'_spectra.png'),dpi=150); plt.close(fig)
            else:
                recordings=list(dict.fromkeys(r['recording'] for r in subset))
                fig,axes=plt.subplots(2,4,figsize=(16,8))
                for ax,rid in zip(axes.flat,recordings):
                    for row in subset:
                        if row['recording']!=rid: continue
                        with np.load(corrected_path(row)) as a:
                            valid=row['scaling_valid']; component=row['component']
                            label=component+(f" ({row['center_hz']:.0f} Hz)" if 'center_hz' in row else '')
                            if not valid: label+=' / UNRELIABLE SCALING'
                            ax.plot(a['alpha'],a['f_alpha'],'-' if valid else '--',
                                    color=None if valid else '#b0b0b0',lw=1.3,alpha=1 if valid else .6,label=label)
                    sample=next(r for r in subset if r['recording']==rid)
                    ax.set(title=sample['state']+' r'+str(sample['repeat']),xlabel='alpha',ylabel='f(alpha)')
                    ax.legend(fontsize=5); ax.grid(alpha=.2)
                fig.suptitle(f'{method.upper()} / {channel}: components labelled by frequency; invalid scaling is grey/dashed')
                fig.tight_layout(); fig.savefig(BASE/'figures'/(method+'_'+channel+'_spectra.png'),dpi=150); plt.close(fig)


def interim_report(rows,selection):
    lines=['# PHM2009 corrected experiment','',
           'TASK 1 và QC đã hoàn tất; sensitivity VMD và cấu hình cuối đang được tính. Các hàng VMD hiện tại dùng cấu hình pilot cũ và sẽ được thay thế.', '',
           f"Common range: {selection['selected_interval']} samples; common_valid={selection['common_valid']}.",
           'Chọn trên toàn bộ 16 tín hiệu raw, chỉ dùng R²/số điểm/CV local slopes/h dương. Không sử dụng nhãn hoặc separation.', '',
           '| Pipeline | QC pass / total |','|---|---:|']
    for method in ['raw','emd','vmd']:
        subset=[r for r in rows if r['pipeline']==method]
        lines.append(f"| {method.upper()} | {sum(r['scaling_valid'] for r in subset)} / {len(subset)} |")
    lines+=['','Các chỉ số của phổ invalid vẫn được lưu để audit, không dùng tuyên bố separation.',
            'Công thức MF-DFA gốc không đổi. EMD kế thừa mode xấp xỉ đã được báo ở pilot, không được coi là IMF nghiêm ngặt.', '',
            'H2 là 24T chipped gear; H5 là 24T broken gear + bearing inner race. H2–H5 không cô lập riêng bearing contribution.']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def final_vmd():
    selected=json.loads((BASE/'config/selected_vmd_configuration.json').read_text(encoding='utf-8'))
    parameters=selected['selected_config']
    inventory=json.loads((BASE/'config/source_inventory.json').read_text(encoding='utf-8'))
    folder=BASE/'decompositions/vmd'; folder.mkdir(exist_ok=True)
    entries,summary,centers=[],[],[]
    template=next(legacy_entries())[1]
    for source in inventory:
        rid=source['recording']; data=np.loadtxt(ROOT/source['relative_source'])
        for column,channel,role in CHANNELS:
            stem=rid+'_'+channel
            cache=folder/(stem+'.npz'); diagnostic_path=folder/(stem+'_diagnostics.json')
            signature=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()+
                (ROOT/'code/core/vmd.py').read_bytes()+source['sha256'].encode()+
                json.dumps(dict(init=1,tol=1e-7,max_iter=2000,fs=GRID['fs']),sort_keys=True).encode()+
                (BASE/'scripts/vmd_kernel.cpp').read_bytes()).hexdigest()
            x=data[:,column].copy(); x-=x.mean()
            cached=json.loads(diagnostic_path.read_text(encoding='utf-8')) if diagnostic_path.exists() else None
            if cached and cache.exists() and cached.get('signature')==signature:
                with np.load(cache) as saved: modes=saved['modes']; residue=saved['residue']; frequencies=saved['centers_hz']
                diag=cached
            else:
                start=time.perf_counter()
                print('FULL VMD',stem,parameters,flush=True)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    modes,_,history,info=vmd(x,**parameters,init=1,tol=1e-7,max_iter=2000,fs=GRID['fs'],return_info=True)
                order=np.argsort(history[-1]); modes=modes[order]; frequencies=history[-1,order]
                residue=x-modes.sum(axis=0)
                np.savez_compressed(cache,modes=modes,residue=residue,centers_hz=frequencies,removed_mean=data[:,column].mean())
                diag=dict(signature=signature,recording=rid,channel=channel,config=parameters,
                    algorithm_info=info,reconstruction_error=info['reconstruction_error'],
                    warnings=[str(w.message) for w in caught],elapsed_seconds=time.perf_counter()-start)
                dump(diagnostic_path,diag)
            mode_rows,quality=mode_metrics(x,modes,frequencies)
            identity=dict(recording=rid,state=source['state'],label=source['label'],repeat=source['repeat'],
                          channel=channel,role=role,pipeline='vmd',samples=len(x))
            for rank,mode in enumerate(modes,1):
                mode_identity=dict(**identity,component=f'mode{rank}',rank=rank,center_hz=float(frequencies[rank-1]),
                    mode_energy=mode_rows[rank-1]['mode_energy'],mode_energy_fraction=mode_rows[rank-1]['mode_energy_fraction'])
                s,fq,_,_=mfdfa(mode,template['q'],m=2,scales=template['scales'])
                entries.append((mode_identity,dict(q=template['q'],scales=s,Fq=fq),cache))
            for component,y in [('oscillatory_sum',modes.sum(axis=0)),('residue',residue)]:
                s,fq,_,_=mfdfa(y,template['q'],m=2,scales=template['scales'])
                entries.append((dict(**identity,component=component),dict(q=template['q'],scales=s,Fq=fq),cache))
            centers.extend(dict(recording=rid,channel=channel,scope='full_final_record',**parameters,**r) for r in mode_rows)
            summary.append(dict(recording=rid,channel=channel,samples=len(x),scope='full_final_record',**parameters,
                converged=diag['algorithm_info']['converged'],iterations=diag['algorithm_info']['iterations'],
                reconstruction_error=diag['reconstruction_error'],constraint_error=diag['algorithm_info']['constraint_error'],**quality))
            print('FULL VMD DONE',stem,'converged',diag['algorithm_info']['converged'],flush=True)
    table(BASE/'diagnostics/vmd_full_record_validation.csv',summary)
    table(BASE/'diagnostics/vmd_final_mode_centers.csv',centers)
    selected['full_record_confirmation_pending']=False
    selected['full_record_confirmation']=dict(signals=len(summary),converged=sum(r['converged'] for r in summary),
        max_reconstruction_error=max(r['reconstruction_error'] for r in summary),
        median_reconstruction_error=float(np.median([r['reconstruction_error'] for r in summary])))
    matches=[]
    for state in ['H1','H2','H5','H6']:
        rids={s['repeat']:s['recording'] for s in inventory if s['state']==state}
        for _,channel,_ in CHANNELS:
            first=[r for r in centers if r['recording']==rids[1] and r['channel']==channel]
            second=[r for r in centers if r['recording']==rids[2] and r['channel']==channel]
            c1=np.array([r['center_hz'] for r in first]); c2=np.array([r['center_hz'] for r in second])
            cost=np.abs(np.log(np.maximum(c1[:,None],1)/np.maximum(c2[None,:],1)))
            ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj):
                drift=abs(c1[i]-c2[j])/max((c1[i]+c2[j])/2,1.)
                matches.append(dict(state=state,channel=channel,scope='full_final_record',
                    repeat1_mode_rank=int(i+1),repeat2_mode_rank=int(j+1),
                    repeat1_center_hz=float(c1[i]),repeat2_center_hz=float(c2[j]),
                    relative_center_difference=float(drift),frequency_match_valid=bool(drift<=.15)))
    table(BASE/'diagnostics/vmd_full_repeat_frequency_stability.csv',matches)
    full=selected['full_record_confirmation']
    full.update(p90_reconstruction_error=float(np.quantile([r['reconstruction_error'] for r in summary],.9)),
        median_repeat_center_relative_error=float(np.median([r['relative_center_difference'] for r in matches])),
        matched_mode_fraction=float(np.mean([r['frequency_match_valid'] for r in matches])),
        median_duplicate_fraction=float(np.median([r['duplicate_fraction'] for r in summary])),
        minimum_mean_mode_energy_fraction=float(min(np.mean([r['mode_energy_fraction'] for r in centers if r['mode_rank']==rank]) for rank in range(1,parameters['K']+1))))
    full['pooled_robustness_pass']=bool(full['converged']/len(summary)>=.9 and full['p90_reconstruction_error']<=.15
        and full['median_repeat_center_relative_error']<=.15 and full['median_duplicate_fraction']<=.30
        and full['minimum_mean_mode_energy_fraction']>=1e-4)
    dump(BASE/'config/selected_vmd_configuration.json',selected)
    return entries


def separation_tables(rows):
    # Cross-record mode comparisons use fixed physical frequency bands, not rank.
    representatives=[]
    for pipeline in ['raw','emd','vmd']:
        for _,channel,_ in CHANNELS:
            subset=[r for r in rows if r['pipeline']==pipeline and r['channel']==channel]
            for rid in dict.fromkeys(r['recording'] for r in subset):
                components=[r for r in subset if r['recording']==rid]
                for row in components:
                    if row['component'] in ['raw','oscillatory_sum']:
                        representatives.append(dict(**row,comparison_component=row['component']))
                for low,high in FIXED_BANDS:
                    candidates=[r for r in components if r['component'].startswith('mode') and low<=r['center_hz']<high]
                    # Mode energy is filled from full decompositions for EMD below.
                    if candidates:
                        chosen=max(candidates,key=lambda r:r.get('mode_energy',0))
                        representatives.append(dict(**chosen,comparison_component=f'frequency_band_{low}_{high}_Hz'))
    table(BASE/'features/frequency_band_representatives.csv',representatives)
    results=[]
    for pipeline in ['raw','emd','vmd']:
        for _,channel,_ in CHANNELS:
            group=[r for r in representatives if r['pipeline']==pipeline and r['channel']==channel]
            for component in dict.fromkeys(r['comparison_component'] for r in group):
                for left,right in [('H1','H2'),('H2','H5'),('H2','H6')]:
                    a=[r for r in group if r['comparison_component']==component and r['state']==left and r['scaling_valid'] and r['decomposition_valid']]
                    b=[r for r in group if r['comparison_component']==component and r['state']==right and r['scaling_valid'] and r['decomposition_valid']]
                    reason='' if len(a)==2 and len(b)==2 else 'fewer_than_two_valid_recordings_per_condition'
                    center_ratio=None
                    if not reason and component.startswith('frequency_band'):
                        frequencies=[r['center_hz'] for r in a+b]
                        center_ratio=max(frequencies)/max(min(frequencies),1.)
                        if center_ratio>1.5: reason='centers_not_comparable_within_band'
                    for feature in ['delta_alpha','alpha0','delta_h']:
                        row=dict(pipeline=pipeline,channel=channel,comparison_component=component,pair=left+'-'+right,
                            feature=feature,left_valid_n=len(a),right_valid_n=len(b),evaluable=not reason,
                            invalid_reason=reason,center_frequency_ratio=center_ratio,
                            interpretation='pilot_descriptive_only')
                        if not reason:
                            av=np.array([r[feature] for r in a]); bv=np.array([r[feature] for r in b])
                            gap=max(av.min(),bv.min())-min(av.max(),bv.max())
                            row.update(left_mean=float(av.mean()),right_mean=float(bv.mean()),mean_difference=float(bv.mean()-av.mean()),
                                nonoverlap_gap=float(gap),observed_repeat_ranges_disjoint=bool(gap>0))
                        results.append(row)
    table(BASE/'summaries/valid_state_pair_comparisons.csv',results)
    return results


def fill_emd_energy(rows):
    for row in rows:
        if row['pipeline']!='emd' or not row['component'].startswith('mode'): continue
        path=OLD/'emd_mfdfa'/(row['recording']+'_'+row['channel']+'_decomposition.npz')
        with np.load(path) as saved:
            mode=saved['modes'][row['rank']-1]
            row['mode_energy']=float(np.mean(mode**2))


def validate(rows):
    inventory=json.loads((BASE/'config/source_inventory.json').read_text(encoding='utf-8'))
    for source in inventory:
        assert hashlib.sha256((ROOT/source['relative_source']).read_bytes()).hexdigest()==source['sha256']
    for row in rows:
        with np.load(corrected_path(row)) as saved:
            computed,metrics=fit_scaling(saved['scales'],saved['Fq'],saved['q'],
                json.loads((BASE/'config/selected_scaling_range.json').read_text(encoding='utf-8'))['selected_interval'])
            assert metrics['scaling_valid']==row['scaling_valid']
            for key in ['hq','tau','alpha','f_alpha']:
                np.testing.assert_allclose(computed[key],saved[key],rtol=1e-12,atol=1e-12)
            assert np.allclose(saved['tau'],saved['q']*saved['hq']-1)
            assert np.isclose(saved['f_alpha'][10],1)
    original_fq_checked=0
    for identity,numerical,_ in legacy_entries():
        if identity['pipeline']=='vmd': continue
        with np.load(corrected_path(identity)) as saved:
            np.testing.assert_array_equal(saved['Fq'],numerical['Fq'])
        original_fq_checked+=1
    reconstruction_checks=[]
    for source in inventory:
        data=np.loadtxt(ROOT/source['relative_source'])
        for column,channel,_ in CHANNELS:
            with np.load(BASE/'decompositions/vmd'/(source['recording']+'_'+channel+'.npz')) as saved:
                x=data[:,column]-data[:,column].mean()
                reconstructed=saved['modes'].sum(axis=0)+saved['residue']
                error=float(np.linalg.norm(reconstructed-x)/np.linalg.norm(x))
                assert error<1e-12
                reconstruction_checks.append(dict(recording=source['recording'],channel=channel,mode_plus_residue_relative_error=error))
    table(BASE/'diagnostics/reconstruction_identity_checks.csv',reconstruction_checks)
    sensitivity=read_csv(BASE/'diagnostics/vmd_parameter_sensitivity.csv')
    assert len(sensitivity)==640 and len({r['configuration'] for r in sensitivity})==40
    dump(BASE/'diagnostics/validation.json',dict(analyses_checked=len(rows),all_passed=True,source_sha256_unchanged=True,
        original_fq_arrays_unchanged=original_fq_checked,sensitivity_signals_checked=len(sensitivity),
        checks=['refit/QC recomputation','tau=q*h-1','Legendre identity f(alpha(q=0))=1','raw data SHA256','raw/EMD Fq unchanged','modes+residue=mean-removed raw','complete 40 x 16 sensitivity grid'],
        core_mfdfa_sha256=hashlib.sha256((ROOT/'code/core/mfdfa.py').read_bytes()).hexdigest()))


def finalize():
    prepare()
    bounds=json.loads((BASE/'config/selected_scaling_range.json').read_text(encoding='utf-8'))['selected_interval']
    entries=[e for e in legacy_entries() if e[0]['pipeline']!='vmd']+final_vmd()
    # Remove superseded baseline VMD arrays after replacement has been built.
    expected={corrected_path(identity).resolve() for identity,_,_ in entries}
    for path in (BASE/'arrays/vmd').glob('*.npz'):
        if path.resolve() not in expected:
            assert path.resolve().is_relative_to(BASE.resolve())
            path.unlink()
    rows=apply_qc(entries,bounds)
    fill_emd_energy(rows)
    table(BASE/'features/mfdfa_features.csv',rows)
    table(BASE/'diagnostics/mfdfa_quality_control.csv',rows)
    comparisons=separation_tables(rows)
    decomposition_summary=[]
    for method in ['raw','emd','vmd']:
        for _,channel,_ in CHANNELS:
            unique={r['recording']:r for r in rows if r['pipeline']==method and r['channel']==channel}
            subset=list(unique.values())
            decomposition_summary.append(dict(pipeline=method,channel=channel,signals=len(subset),
                decomposition_valid=sum(r['decomposition_valid'] for r in subset),
                median_reconstruction_error=float(np.median([r['reconstruction_error'] for r in subset])),
                maximum_reconstruction_error=max(r['reconstruction_error'] for r in subset),
                strict_emd_valid=sum(bool(r.get('strict_emd_imf_valid',False)) for r in subset) if method=='emd' else ''))
    table(BASE/'summaries/decomposition_validity_summary.csv',decomposition_summary)
    plot_spectra(rows)
    diagnostic_plots(rows,bounds)
    sensitivity_plots()
    comparison_plots(rows)
    validate(rows)
    final_report(rows,comparisons)
    cleanup_and_manifest()
    print('FINAL QC counts:',{m:(sum(r['scaling_valid'] for r in rows if r['pipeline']==m),sum(r['pipeline']==m for r in rows)) for m in ['raw','emd','vmd']},flush=True)


def cleanup_and_manifest():
    # These caches are reconstructible and their scalar information must first
    # be present in permanent CSVs. No input or generated dependency is removed.
    import shutil
    sensitivity=read_csv(BASE/'diagnostics/vmd_parameter_sensitivity.csv')
    centers=read_csv(BASE/'diagnostics/vmd_mode_centers.csv')
    assert len(sensitivity)==640 and len(centers)==3840
    work=(BASE/'_work').resolve()
    assert work.parent==BASE.resolve() and work.name=='_work'
    removed=[dict(relative_path=str(p.relative_to(BASE)),bytes=p.stat().st_size,
        reason='Reconstructible window/job/plot cache; permanent grid CSVs and config retained') for p in work.rglob('*') if p.is_file()]
    if (BASE/'diagnostics/output_cleanup.csv').exists():
        combined={r['relative_path']:r for r in read_csv(BASE/'diagnostics/output_cleanup.csv')}
        combined.update({r['relative_path']:r for r in removed}); removed=list(combined.values())
    # Scripts recreate this cache; reports refer only to permanent outputs.
    shutil.rmtree(work)
    table(BASE/'diagnostics/output_cleanup.csv',removed)
    dump(BASE/'config/runtime_provenance.json',dict(python=sys.version,numpy=np.__version__,
        mfdfa_core_sha256=hashlib.sha256((ROOT/'code/core/mfdfa.py').read_bytes()).hexdigest(),
        vmd_core_sha256=hashlib.sha256((ROOT/'code/core/vmd.py').read_bytes()).hexdigest(),
        fft_backend='numpy.fft',optional_admm_backend='GCC fused C++ kernel (parity validated)',
        sensitivity_source_hashes=json.loads((BASE/'config/source_inventory.json').read_text()),
        generated_input_dependencies=['result/test/step4/raw_mfdfa','result/test/step4/emd_mfdfa','result/test/step4/vmd_mfdfa (candidate-range audit only)']))
    manifest=[]
    for path in sorted(BASE.rglob('*')):
        if path.is_file() and path.name!='output_manifest.csv':
            manifest.append(dict(relative_path=str(path.relative_to(BASE)),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    table(BASE/'diagnostics/output_manifest.csv',manifest)



def comparison_plots(rows):
    for _,channel,_ in CHANNELS:
        fig,axes=plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True)
        for ax,method in zip(axes,['raw','emd','vmd']):
            component='raw' if method=='raw' else 'oscillatory_sum'
            subset=[r for r in rows if r['pipeline']==method and r['channel']==channel and r['component']==component]
            for row in subset:
                with np.load(corrected_path(row)) as a:
                    valid=row['scaling_valid'] and row['decomposition_valid']
                    ax.plot(a['alpha'],a['f_alpha'],'-' if valid and row['repeat']==1 else '--',
                        color=COLORS[row['state']] if valid else '#aaaaaa',
                        label=row['state']+' r'+str(row['repeat'])+(' / UNRELIABLE SCALING' if not valid else ''))
            ax.set(title=method.upper()+'/'+component,xlabel='alpha',ylabel='f(alpha)'); ax.legend(fontsize=6); ax.grid(alpha=.2)
        fig.suptitle('Common-scale comparison / '+channel+' / sum spectra do not summarize individual mode spectra')
        fig.tight_layout(); fig.savefig(BASE/'figures'/('pipeline_comparison_'+channel+'.png'),dpi=150); plt.close(fig)


def sensitivity_plots():
    ranking=read_csv(BASE/'summaries/vmd_reconstruction_summary.csv')
    grid_centers=read_csv(BASE/'diagnostics/vmd_mode_centers.csv')
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for ax,tau in zip(axes,[0.,.5]):
        for j,alpha in enumerate(GRID['alpha']):
            subset=[r for r in grid_centers if float(r['tau'])==tau and int(r['alpha'])==alpha]
            ax.scatter([int(r['K'])+(j-1.5)*.13 for r in subset],
                [float(r['center_hz']) for r in subset],s=5,alpha=.3,label=f'alpha={alpha}')
        ax.set(yscale='log',title=f'tau={tau}',xlabel='K (offset by alpha)',ylabel='Mode center (Hz)')
        ax.set_xticks(GRID['K']); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('All sensitivity centers / sorted ranks are display order, not physical mode matching')
    fig.tight_layout(); fig.savefig(BASE/'figures/vmd_center_frequency_sensitivity.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,tau in zip(axes,[0.,.5]):
        for alpha in GRID['alpha']:
            subset=sorted([r for r in ranking if float(r['tau'])==tau and int(r['alpha'])==alpha],key=lambda r:int(r['K']))
            ax.plot([int(r['K']) for r in subset],[float(r['p90_reconstruction_error']) for r in subset],'o-',label=f'alpha={alpha}')
        ax.axhline(.15,color='k',ls='--',lw=1); ax.set(title=f'tau={tau}',xlabel='K',ylabel='Reconstruction error p90')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('VMD sensitivity: pooled diagnostics, fixed first 65,536 samples')
    fig.tight_layout(); fig.savefig(BASE/'figures/vmd_reconstruction_sensitivity.png',dpi=150); plt.close(fig)
    selected=json.loads((BASE/'config/selected_vmd_configuration.json').read_text(encoding='utf-8'))
    chosen=selected['selected_summary']['configuration']
    matched=[r for r in read_csv(BASE/'diagnostics/vmd_repeat_frequency_stability.csv') if r['configuration']==chosen]
    full_matched=read_csv(BASE/'diagnostics/vmd_full_repeat_frequency_stability.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=True)
    for ax,records,title in zip(axes,[matched,full_matched],['Sensitivity 65,536 samples','Full recordings']):
        for channel,marker in [('output_voltage','o'),('input_voltage','x')]:
            subset=[r for r in records if r['channel']==channel]
            ax.scatter([float(r['repeat1_center_hz']) for r in subset],[float(r['repeat2_center_hz']) for r in subset],label=channel,marker=marker)
        ax.plot([1,33334],[1,33334],'k--'); ax.set(xscale='log',yscale='log',xlabel='Repeat 1 center (Hz)',ylabel='Matched repeat 2 center (Hz)',title=title)
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Hungarian frequency matching / '+chosen)
    fig.tight_layout(); fig.savefig(BASE/'figures/vmd_repeat_center_stability.png',dpi=150); plt.close(fig)
    full=read_csv(BASE/'diagnostics/vmd_final_mode_centers.csv')
    for _,channel,_ in CHANNELS:
        subset=[r for r in full if r['channel']==channel]
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for ax,rid in zip(axes.flat,dict.fromkeys(r['recording'] for r in subset)):
            path=BASE/'decompositions/vmd'/(rid+'_'+channel+'.npz')
            with np.load(path) as saved:
                f=np.fft.rfftfreq(saved['modes'].shape[1],1/GRID['fs'])
                for mode,center in zip(saved['modes'],saved['centers_hz']):
                    power=np.abs(np.fft.rfft(mode))**2
                    ax.semilogy(f,np.maximum(power/power.max(),1e-12),lw=.7,label=f'{center:.0f} Hz')
            ax.set(title=rid.replace('helical_','H').replace('_50hz_High_',' r'),xlabel='Frequency (Hz)',ylabel='Normalized power')
            ax.set_ylim(1e-6,1.1); ax.legend(fontsize=6); ax.grid(alpha=.2)
        fig.suptitle('Full-record final VMD mode spectra / '+channel)
        fig.tight_layout(); fig.savefig(BASE/'figures'/('vmd_mode_power_'+channel+'.png'),dpi=150); plt.close(fig)


def final_report(rows,comparisons):
    selected=json.loads((BASE/'config/selected_scaling_range.json').read_text(encoding='utf-8'))
    vmd_choice=json.loads((BASE/'config/selected_vmd_configuration.json').read_text(encoding='utf-8'))
    bounds=selected['selected_interval']
    window=vmd_choice['selected_summary']; full=vmd_choice['full_record_confirmation']
    grid_rows=read_csv(BASE/'summaries/vmd_reconstruction_summary.csv')
    lines=['# PHM2009: corrected pilot experiment','',
        f"Common scaling interval **{bounds[0]}–{bounds[1]} samples**, actual selected scales {rows[0]['fit_scale_min']}–{rows[0]['fit_scale_max']}, **{rows[0]['scale_points']} points**.",
        f"Actual interval tương ứng {1000*rows[0]['fit_scale_min']/GRID['fs']:.3f}–{1000*rows[0]['fit_scale_max']/GRID['fs']:.3f} ms và {np.log10(rows[0]['fit_scale_max']/rows[0]['fit_scale_min']):.3f} decade. Đây là local scaling window ngắn, chưa chứng minh asymptotic scaling.",
        'Fq(s) của raw/EMD được tái sử dụng nguyên giá trị; chỉ fit lại. MF-DFA core theo Kantelhardt không thay đổi.', '',
        '## Scaling validity và QC', '',
        'QC cố định: ≥8 scales; min R²≥0,90; ≥80% q có R²≥0,95; ≥80% q có local-slope CV≤0,5; min h(q)>0,05. CV dùng std/abs(mean) của các slope giữa scales kề nhau. Ngưỡng là quy tắc pilot thực hành, không phải định lý kiểm định multifractality.',
        'Range selection dùng tất cả 16 tín hiệu raw và không dùng class separation; mỗi tín hiệu phải đạt toàn bộ QC. Chỉ dải 64–512 đạt cho tất cả. Các dải/các q và local slope được giữ trong scaling_range_diagnostics.csv.', '',
        '| Pipeline | QC pass / total | Cột 2 pass / total | Cột 1 pass / total |', '|---|---:|---:|---:|']
    for method in ['raw','emd','vmd']:
        subset=[r for r in rows if r['pipeline']==method]
        counts=[f"{sum(r['scaling_valid'] for r in subset)}/{len(subset)}"]
        for _,channel,_ in CHANNELS:
            cc=[r for r in subset if r['channel']==channel]; counts.append(f"{sum(r['scaling_valid'] for r in cc)}/{len(cc)}")
        lines.append('| '+method.upper()+' | '+' | '.join(counts)+' |')
    lines+=['', 'Các total EMD/VMD gồm từng mode, oscillatory_sum và residue; số pass từng loại có ở summaries/valid_component_summary.csv. Numerical arrays/feature invalid vẫn lưu để audit. Spectrum invalid luôn grey/dashed và ghi UNRELIABLE SCALING, không được dùng separation.', '',
        f"Spectrum-shape audit: {sum(r['scaling_valid'] and r['spectrum_shape_warning'] for r in rows)} phổ scaling-valid có alpha(q) tăng ở ít nhất một bước hoặc f(alpha)>1. Các field shape warning chỉ audit, không thay đổi QC đã cố định trước selection. R² tốt chưa chứng minh tau concave hay multifractality vật lý; chiều rộng phổ hữu hạn có thể chịu bias từ fitting/Legendre derivative.", '',
        '![Raw spectra](figures/raw_output_voltage_spectra.png)', '',
        '## VMD robustness và cấu hình cuối', '',
        f"Cấu hình chọn: **{vmd_choice['selected_config']}**, init=1, tol=1e-7; max_iter=1.000 ở sensitivity và 2.000 ở full recording.",
        f"Có {sum(r['eligible']=='True' for r in grid_rows)}/40 cấu hình eligible. Cấu hình chọn trên windows: convergence {window['convergence_fraction']:.1%}; median repeat drift {window['median_repeat_center_relative_error']:.2%}; drift p90 {window['p90_repeat_center_relative_error']:.2%}; fraction matches trong ngưỡng 15% = {window['matched_mode_fraction']:.1%}; reconstruction p90 {window['p90_reconstruction_error']:.2%}. Median ổn định không có nghĩa mọi mode/repeat đều ổn định.",
        'Theo lựa chọn người dùng: sensitivity dùng first 65.536 samples (~0,98 s) trên mỗi tín hiệu, toàn bộ 40 combinations K={4,5,6,7,8}, alpha={500,1000,2000,4000}, tau={0,0,5}. Hai kênh/tám recordings đều tham gia. Cấu hình chọn sau đó chạy trên toàn bộ dữ liệu gốc, không cắt file H5 r2.',
        'Tiêu chí eligible: ≥90% hội tụ; reconstruction error p90≤0,15; median repeat center drift≤0,15; median duplicate fraction≤0,30; mean energy fraction của mỗi rank≥1e-4. Duplication dùng spectral-overlap cosine>0,8; đây là proxy số học, không chứng minh mode mixing vật lý.',
        'Trong các cấu hình eligible, ưu tiên drift trung tâm nhỏ, ít duplication, reconstruction nhỏ, K thấp. Không dùng QC pass count, Δα, α0, Δh hoặc separation để chọn VMD.',
        f"Full-record confirmation: convergence {full['converged']}/{full['signals']}; reconstruction median {full['median_reconstruction_error']:.2%}, p90 {full['p90_reconstruction_error']:.2%}, max {full['max_reconstruction_error']:.2%}; repeat drift median {full['median_repeat_center_relative_error']:.2%}, matches within 15% {full['matched_mode_fraction']:.1%}; pooled robustness pass = {full['pooled_robustness_pass']}.",
        'Center matching giữa hai acquisitions dùng Hungarian assignment trên khoảng cách log-frequency, không đối chiếu mode chỉ vì cùng index. Dữ liệu pairing sử dụng acquisition membership, không sử dụng mức separation giữa classes.', '',
        '![Reconstruction sensitivity](figures/vmd_reconstruction_sensitivity.png)', '',
        '![Repeat centers](figures/vmd_repeat_center_stability.png)', '',
        '## Descriptive separation, chỉ dùng valid spectra', '',
        'Chỉ đánh giá khi cả hai recordings của mỗi điều kiện đều có scaling_valid và decomposition_valid. Với mode, dùng representative energy cao nhất trong các frequency bands cố định; yêu cầu tỷ số center lớn nhất/nhỏ nhất của bốn repeats≤1,5. Cùng index không được coi là cùng mode vật lý.', '',
        '| Pipeline/component, cột 2 | H1–H2 | H2–H5 | H2–H6 |', '|---|---|---|---|']
    for method in ['raw','emd','vmd']:
        component='raw' if method=='raw' else 'oscillatory_sum'
        values=[]
        for pair in ['H1-H2','H2-H5','H2-H6']:
            subset=[r for r in comparisons if r['pipeline']==method and r['channel']=='output_voltage' and r['comparison_component']==component and r['pair']==pair]
            valid=[r for r in subset if r['evaluable']]
            values.append(', '.join(r['feature'] for r in valid if r['observed_repeat_ranges_disjoint']) or ('không tách khoảng' if len(valid)==3 else 'không đủ phổ valid'))
        lines.append('| '+method.upper()+'/'+component+' | '+' | '.join(values)+' |')
    lines+=['','Toàn bộ comparisons theo frequency band/kênh nằm trong summaries/valid_state_pair_comparisons.csv; rows không evaluable chứa lý do và không có chỉ số separation.',
        '', '![Pipeline spectra comparison](figures/pipeline_comparison_output_voltage.png)', '',
        '| Pipeline/component, cột 2 | Evaluable feature/pair rows | Disjoint repeat ranges |', '|---|---:|---:|']
    for method in ['raw','emd','vmd']:
        component='raw' if method=='raw' else 'oscillatory_sum'
        subset=[r for r in comparisons if r['pipeline']==method and r['channel']=='output_voltage' and r['comparison_component']==component]
        lines.append(f"| {method.upper()}/{component} | {sum(r['evaluable'] for r in subset)}/9 | {sum(bool(r.get('observed_repeat_ranges_disjoint',False)) for r in subset)} |")
    lines+=['', 'Bảng này đối chiếu raw với tổng dao động; không gộp các mode spectra. Mode features có QC riêng và đối chiếu theo frequency bands trong CSV, vì vậy coverage khác nhau không được diễn giải thành thắng/thua giữa algorithms.', '',
        'Trong pilot cột 2, Raw tách khoảng ở 7/9 feature–pair rows, VMD oscillatory_sum ở 6/9; EMD sum chỉ 6/9 rows evaluable và tách ở 4. Vì vậy so sánh raw/sum không hỗ trợ cải thiện separation của VMD so với Raw. Các VMD mode representatives valid ở band 0–1.000 Hz chỉ tách Δα/Δh cho H2–H5 và α0 cho H2–H6, vốn đã tách trong Raw; H1–H2 không có mode band đủ bốn phổ valid/matched. EMD không có mode band đủ điều kiện, nên chưa đủ căn cứ xếp hạng EMD và VMD theo individual modes.', '',
        'Nhiều mode narrowband không đạt QC; mode coverage và dải tần giữa EMD/VMD khác nhau. Kết quả không hỗ trợ một tuyên bố VMD cải thiện tổng quát; đây là descriptive pilot evidence, không phải statistical evidence từ hai recordings/class.', '',
        '## Feature values của các phổ valid', '',
        'Toàn bộ Δα, α0, Δh từng recording/component ở features/mfdfa_features.csv, kèm scaling_valid và invalid_reason. Bảng dưới chỉ lấy raw hoặc oscillatory_sum, cột 2; dấu — nghĩa là invalid và không diễn giải feature.', '',
        '| Condition/repeat | Pipeline | Δα | α0 | Δh |', '|---|---|---:|---:|---:|']
    for row in rows:
        if row['channel']!='output_voltage' or row['component'] not in ['raw','oscillatory_sum']: continue
        values=[f"{row[k]:.5f}" if row['scaling_valid'] and row['decomposition_valid'] else '—' for k in ['delta_alpha','alpha0','delta_h']]
        lines.append('| '+row['state']+' r'+str(row['repeat'])+' | '+row['pipeline'].upper()+' | '+' | '.join(values)+' |')
    lines+=['',
        '## Decomposition validity và reconstruction', '',
        'EMD giữ biến thể sifting thực hành của pilot: mode xấp xỉ, kiểm tra practical convergence và báo riêng strict IMF count condition. Không coi đó là benchmark EMD nghiêm ngặt. Dừng ở sáu mode; residual giữ phần thấp tần còn lại. Tính MF-DFA riêng trên mỗi mode, sum và residual, không cộng/trung bình các spectra.',
        'VMD tau=0 là denoising, không bắt buộc tổng mode bằng raw; tau>0 có constraint và tiêu chí reconstruction riêng. Xem diagnostics/vmd_full_record_validation.csv để tách solver convergence, reconstruction và scaling validity.', '',
        'Reconstruction error là ||x−sum(modes)||/||x|| trên full tín hiệu đã bỏ mean; constraint_error là residual relative trên miền Fourier của mirror extension, vì vậy hai chỉ số có thể khác nhau. Thêm residue thì khớp raw đã bỏ mean theo construction, nhưng identity này không có nghĩa riêng sum(modes) reconstruct tốt.', '',
        '| Pipeline | Channel | Decomposition pass | Reconstruction median | Reconstruction max |', '|---|---|---:|---:|---:|']
    for row in read_csv(BASE/'summaries/decomposition_validity_summary.csv'):
        lines.append(f"| {row['pipeline'].upper()} | {row['channel']} | {row['decomposition_valid']}/{row['signals']} | {float(row['median_reconstruction_error']):.6g} | {float(row['maximum_reconstruction_error']):.6g} |")
    lines+=['',
        '## Dataset và giới hạn nhận định', '',
        'Tám PHM2009 Helical recordings: 50 Hz nominal operating speed, High load; sample rate 66.6667 kHz theo VidData. Cột 2 output voltage chính, cột 1 input voltage secondary; tachometer không phân tích MF-DFA.',
        '**H2 = 24T chipped gear. H5 = 24T broken gear + bearing inner-race fault. H2–H5 KHÔNG cô lập riêng bearing contribution.** H1 healthy, H6 bent input shaft. Giữ nguyên 266.656 mẫu ở bảy file và 245.648 mẫu ở H5 r2.',
        'Không được suy ra classification accuracy, statistical significance, causal bearing contribution, optimal configuration cho dataset khác, hoặc multifractality vật lý chỉ vì spectrum có chiều rộng. Sensitivity window và full-record validation phải được phân biệt.', '',
        '## Reproducibility và output management', '',
        'config/: nguồn/hash, QC, selected scaling, VMD grid và selected config. diagnostics/: per-q candidates, QC audit, grid modes/constraints/center matching, full-record validation. features/: numerical features và representatives. arrays/: Fq gốc và kết quả fit trên selected range, kể cả invalid. decompositions/vmd/: chỉ full decomposition của config cuối. figures/ và summaries/: plots/bảng cuối.',
        'Không giữ full spectra/decompositions của mọi sensitivity combination. Scalar diagnostics đủ audit; windows và job caches có thể tái tạo từ raw + config và được dọn sau validation. step4 cũ cung cấp raw/EMD Fq và decomposition; chưa xóa những input generated mà script còn dùng.', '',
        f"Verification: {json.loads((BASE/'diagnostics/tests.json').read_text())['tests_run']} tests của refit/QC/mode metrics/label-independent selection/frequency matching/accelerator đều pass; validation.json xác nhận nguyên SHA256 raw, Fq raw/EMD bit-for-bit, tái tính QC/Legendre features và modes+residue khớp tín hiệu đã bỏ mean. output_cleanup.csv audit cache đã dọn, output_manifest.csv kiểm kê/hash các file còn lại.", '',
        '```powershell', 'python -B result\\test\\step4_corrected\\scripts\\experiment.py scaling',
        'python -B result\\test\\step4_corrected\\scripts\\vmd_sensitivity.py',
        'python -B result\\test\\step4_corrected\\scripts\\experiment.py finalize', '```',
        'Optional fused ADMM runtime ở _runtime/ đã kiểm tra parity với code/core/vmd.py; NumPy FFT và công thức ADMM giữ nguyên. Source scripts/vmd_kernel.cpp được biên dịch bằng GCC với -O3 -std=c++17 -shared -static-libgcc -static-libstdc++; nếu DLL không có, scripts/vmd_accel.py dùng implementation NumPy gốc. diagnostics/accelerator_parity.json và tests.json lưu kiểm chứng.',
        'Mỗi lần chạy overwrite trực tiếp cùng tên output; không sinh v2/final/corrected variants. Không sửa raw, metadata, papers hoặc code/core.']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    stage=sys.argv[1] if len(sys.argv)>1 else 'scaling'
    if stage=='scaling': scaling()
    elif stage=='finalize': finalize()
    else: raise ValueError('Use scaling or finalize')
