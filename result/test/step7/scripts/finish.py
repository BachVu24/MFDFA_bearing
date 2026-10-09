"""Full Spur validation using locked Step 6 method, independent analysis before topology comparison."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import pipeline as P
os.environ.setdefault('XDG_CACHE_HOME',str(P.BASE/'_work/cache'))
P.Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
import analysis as A
import numpy as np
import itertools, json
from collections import defaultdict
from presentation import figures, save, plt, main_category

F=A.FEATURES
CH=[c[1] for c in P.CHANNELS]

def primary(rows,method,ch):
    return [r for r in rows if r['pipeline']==method and r['channel']==ch and r.get('category')==main_category(method)]

def extras_and_matched(pairs,speed,load,cells):
    matched=[]; extras=[]; recurrence=[]
    lookup={(r['pipeline'],r['channel'],r['category'],r['pair'],r['feature'],r['speed'],r['load']):r for r in pairs}
    for ch,f in itertools.product(CH,F):
        raw={(r['pair'],r['speed'],r['load']):r for r in primary(pairs,'raw',ch) if r['feature']==f and r['evaluable']}
        vmd={(r['pair'],r['speed'],r['load']):r for r in primary(pairs,'vmd',ch) if r['feature']==f and r['evaluable']}
        keys=sorted(raw.keys()&vmd.keys())
        sr={(r['state'],r['load']):r for r in primary(speed,'raw',ch) if r['feature']==f and r['evaluable']}
        sv={(r['state'],r['load']):r for r in primary(speed,'vmd',ch) if r['feature']==f and r['evaluable']}
        lr={(r['state'],r['speed']):r for r in primary(load,'raw',ch) if r['feature']==f and r['evaluable']}
        lv={(r['state'],r['speed']):r for r in primary(load,'vmd',ch) if r['feature']==f and r['evaluable']}
        ss=[sv[k]['normalized_range']/sr[k]['normalized_range'] for k in sr.keys()&sv.keys() if sr[k]['normalized_range']>1e-8]
        ll=[lv[k]['relative_difference']/lr[k]['relative_difference'] for k in lr.keys()&lv.keys() if lr[k]['relative_difference']>1e-8]
        matched.append(dict(channel=ch,feature=f,common_pairs=len(keys),raw_disjoint_count=sum(raw[k]['disjoint_repeat_ranges'] for k in keys),vmd_disjoint_count=sum(vmd[k]['disjoint_repeat_ranges'] for k in keys),
            gained_count=sum(vmd[k]['disjoint_repeat_ranges'] and not raw[k]['disjoint_repeat_ranges'] for k in keys),lost_count=sum(raw[k]['disjoint_repeat_ranges'] and not vmd[k]['disjoint_repeat_ranges'] for k in keys),
            matched_speed_trajectories=len(ss),matched_load_comparisons=len(ll),median_speed_ratio=float(np.median(ss)) if ss else None,median_load_ratio=float(np.median(ll)) if ll else None,
            source_recordings=';'.join(sorted({rid for k in keys for rid in raw[k]['source_recordings'].split(';')}))))
    for r in pairs:
        if not r['category'].startswith('band_'):continue
        raw=lookup[('raw',r['channel'],'raw',r['pair'],r['feature'],r['speed'],r['load'])]
        extra=bool(r['evaluable'] and raw['evaluable'] and r['disjoint_repeat_ranges'] and not raw['disjoint_repeat_ranges'])
        extras.append(dict(r,raw_evaluable=raw['evaluable'],raw_disjoint=raw.get('disjoint_repeat_ranges',''),extra_descriptive_separation=extra))
    groups=defaultdict(list)
    for r in extras:
        if r['pipeline']=='vmd' and r['extra_descriptive_separation']:groups[(r['channel'],r['category'],r['pair'],r['feature'])].append(r)
    for (ch,cat,pair,f),rows in groups.items():
        centers=[v for r in rows for c in cells if (c['pipeline'],c['channel'],c['category'],c['speed'],c['load'])==('vmd',ch,cat,r['speed'],r['load']) and c['state'] in pair.split('-') for v in c['centers_hz']]
        compatible=bool(centers and min(centers)>0 and max(centers)/min(centers)<=1.5)
        recurrence.append(dict(channel=ch,category=cat,pair=pair,feature=f,extra_conditions=len(rows),speeds=len({r['speed'] for r in rows}),loads=len({r['load'] for r in rows}),frequency_compatible_across_extra_conditions=compatible,
            center_min_hz=min(centers),center_max_hz=max(centers),repeated_compatible_evidence=len(rows)>=2 and compatible,conditions=';'.join(f"{r['speed']}/{r['load']}" for r in rows),source_recordings=';'.join(sorted({rid for r in rows for rid in r['source_recordings'].split(';')}))))
    P.table(P.BASE/'summaries/raw_vmd_matched_effects.csv',matched)
    P.table(P.BASE/'summaries/individual_bands_vs_raw.csv',extras)
    P.table(P.BASE/'summaries/vmd_extra_separation_recurrence.csv',recurrence)
    return matched,extras,recurrence

def more_figures(pairs,features,cells):
    pairnames=['-'.join(p) for p in itertools.combinations(P.STATES,2)]
    for ch in CH:
        for m in ['raw','vmd']:
            fig,axes=plt.subplots(3,2,figsize=(13,12),sharex=True)
            for i,f in enumerate(F):
                for j,l in enumerate(P.LOADS):
                    ax=axes[i,j]
                    for s in P.STATES:
                        cc=sorted([c for c in primary(cells,m,ch) if c['state']==s and c['load']==l],key=lambda c:c['speed'])
                        ax.errorbar(P.SPEEDS,[c.get(f+'_mean',np.nan) if c['evaluable'] else np.nan for c in cc],yerr=[c.get(f+'_std',np.nan) if c['evaluable'] else np.nan for c in cc],marker='o',capsize=2,label=s)
                    ax.set(title=l+' / '+f,xlabel='Speed (Hz)',ylabel=f);ax.legend(fontsize=7);ax.grid(alpha=.2)
            fig.suptitle(ch+' — '+m.upper()+', full-recording cell mean ± acquisition SD; gaps=NE')
            save(fig,'feature_vs_speed_'+m+'_'+ch+'.png')
        fig,axes=plt.subplots(3,3,figsize=(17,23))
        for i,m in enumerate(A.METHODS):
            for j,f in enumerate(F):
                ax=axes[i,j];matrix=np.full((28,10),np.nan)
                for r in primary(pairs,m,ch):
                    if r['feature']==f and r['evaluable']:
                        matrix[pairnames.index(r['pair']),list(itertools.product(P.SPEEDS,P.LOADS)).index((r['speed'],r['load']))]=int(r['disjoint_repeat_ranges'])
                ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
                ax.set_yticks(range(28),pairnames);ax.set_xticks(range(10),[str(s)+'/'+l[0] for s,l in itertools.product(P.SPEEDS,P.LOADS)],rotation=60)
                ax.set_title(m.upper()+' / '+f)
                for a,b in zip(*np.where(np.isnan(matrix))):ax.text(b,a,'NE',ha='center',va='center',fontsize=6)
        fig.suptitle(ch+' — disjoint repeat ranges: green=separated, red=overlap, NE=unavailable')
        save(fig,'fault_pair_separation_'+ch+'.png')
        fig,axes=plt.subplots(3,3,figsize=(16,12))
        for i,m in enumerate(A.METHODS):
            for j,f in enumerate(F):
                ax=axes[i,j]
                for si,s in enumerate(P.STATES):
                    for speed in P.SPEEDS:
                        cc=[next(c for c in primary(cells,m,ch) if (c['state'],c['speed'],c['load'])==(s,speed,l)) for l in P.LOADS]
                        yy=[c.get(f+'_mean',np.nan) if c['evaluable'] else np.nan for c in cc]
                        ax.plot([si-.2,si+.2],yy,'o-',alpha=.55,linewidth=.8,label=str(speed) if si==0 else None)
                ax.set_xticks(range(8),P.STATES);ax.set_title(m.upper()+' '+f);ax.grid(alpha=.2);ax.legend(fontsize=6)
        fig.suptitle(ch+' — load paired within fault/speed: left High, right Low; complete-cell means')
        save(fig,'feature_vs_load_'+ch+'.png')
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        modes=[r for r in features if r['pipeline']=='vmd' and r['channel']==ch and r['component'].startswith('mode')]
        for valid,color,label in [(False,'#d95f02','QC invalid'),(True,'#2274a5','QC valid')]:
            rr=[r for r in modes if r['analysis_valid']==valid]
            axes[0].scatter([r['speed'] for r in rr],[r['center_hz'] for r in rr],c=color,s=12,alpha=.4,label=label)
            axes[1].scatter([r['center_hz'] for r in rr],[r.get('delta_alpha',np.nan) for r in rr],c=color,s=12,alpha=.4,label=label)
        for ax in axes:
            ax.legend();ax.grid(alpha=.2)
        axes[0].set(xlabel='Speed (Hz)',ylabel='Mode center (Hz)',yscale='log')
        axes[1].set(xlabel='Mode center (Hz)',ylabel='delta_alpha (invalid shown only for QC audit)',xscale='log')
        for lo,hi in A.BANDS:
            if lo>0:axes[0].axhline(lo,color='k',linestyle=':',linewidth=.6);axes[1].axvline(lo,color='k',linestyle=':',linewidth=.6)
        fig.suptitle(ch+' — all individual VMD modes, fixed frequency bands')
        save(fig,'vmd_frequency_modes_'+ch+'.png')

def validate(features,pairs,sources):
    tests=json.loads((P.BASE/'diagnostics/tests.json').read_text());assert tests['passed']
    P.verify(json.loads((P.BASE/'config/current_source_hashes.json').read_text()))
    assert len(sources)==160 and len({s['recording'] for s in sources})==160
    source={s['recording']:s for s in sources}
    for s in sources:assert P.digest(P.resolve(s['relative_source']))==s['sha256']
    assert len(features)==160*2*18
    assert len({(r['recording'],r['channel'],r['pipeline'],r['component']) for r in features})==len(features)
    for r in features:
        s=source[r['recording']];assert r['samples']==s['samples'] and r['source_sha256']==s['sha256']
        assert r['analysis_valid']==bool(r['scaling_valid'] and r['decomposition_valid'])
        if r.get('array_path'):
            with np.load(P.resolve(r['array_path'])) as a:
                arr,qc=P.fit_scaling(a['scales'],a['Fq'],a['q'],A.CONFIG['fit_interval'])
                assert qc['scaling_valid']==r['scaling_valid']
                np.testing.assert_array_equal(a['q'],A.CONFIG['q']);np.testing.assert_array_equal(a['scales'],A.CONFIG['scales'])
                for k in F:assert np.isclose(qc[k],r[k],rtol=1e-11,atol=1e-12)
    for r in pairs:
        if r['evaluable']:assert len(set(r['source_recordings'].split(';')))==4
    for p in (P.BASE/'summaries').glob('*.csv'):
        for r in P.read_table(p):
            if 'source_recordings' in r:assert set(filter(None,r['source_recordings'].split(';')))<=source.keys()
    import re
    for target in re.findall(r'\]\(([^)]+)\)',(P.BASE/'REPORT.md').read_text()):
        if not target.startswith('http'):assert (P.BASE/target).is_file(),target
    P.lock_config()
    P.verify(json.loads((P.BASE/'config/step6_reference_hashes.json').read_text()))
    assert {s['state'] for s in sources}==set(P.STATES)
    assert len({s['numeric_sha256'] for s in sources})==160
    for s in sources:
        assert P.parse_name(P.resolve(s['relative_source']).name)==(s['state'],s['speed'],s['load'],s['repeat'])
        assert s['label']==P.METADATA[s['state']]['label']
    old_sources=json.loads((P.HISTORICAL_BASE/'config/source_inventory.json').read_text())
    old_roster={s['recording'] for s in old_sources}
    for r in P.read_table(P.BASE/'comparison/helical_vs_spur_summary.csv'):
        allowed=old_roster if r['topology']=='Helical' else source.keys()
        assert set(filter(None,r['source_recordings'].split(';')))<=allowed
    P.dump(P.BASE/'diagnostics/validation.json',dict(all_passed=True,recordings=160,channel_signals=320,analyses=len(features),valid_analyses=sum(r['analysis_valid'] for r in features),full_lengths_preserved=True,no_windows=True,no_classifier=True,no_parameter_tuning=True,locked_processing_configuration_equal_to_step6=True,all_feature_arrays_rechecked=True,source_hashes_unchanged=True,step6_reference_hashes_unchanged=True,compound_metadata_preserved=True,separate_topology_rosters=True,tests=tests))
    P.table(P.BASE/'diagnostics/output_manifest.csv',[dict(path=p.relative_to(P.BASE).as_posix(),bytes=p.stat().st_size,sha256=P.digest(p)) for p in sorted(P.BASE.rglob('*')) if p.is_file() and p.name!='output_manifest.csv'])

def band_figures(pairs,cells,extras):
    cats=[f'band_{lo}_{hi}_Hz' for lo,hi in A.BANDS]
    for ch in CH:
        fig,axes=plt.subplots(1,2,figsize=(15,6))
        coverage=[sum(c['evaluable'] for c in cells if (c['pipeline'],c['channel'],c['category'])==('vmd',ch,cat)) for cat in cats]
        axes[0].bar(range(5),np.array(coverage)/80,color='#2274a5')
        for i,n in enumerate(coverage):axes[0].text(i,n/80+.025,f'{n}/80',ha='center')
        axes[0].set(ylim=(0,1.1),ylabel='Complete matched cells /80',title='Band coverage (selection before QC)')
        for i,f in enumerate(F):
            xpos=np.arange(5)+(i-1)*.24;rates=[];counts=[]
            for cat in cats:
                rr=[r for r in pairs if (r['pipeline'],r['channel'],r['category'],r['feature'])==('vmd',ch,cat,f) and r['evaluable']]
                n=sum(r['disjoint_repeat_ranges'] for r in rr);rates.append(n/len(rr) if rr else 0);counts.append((n,len(rr)))
            axes[1].bar(xpos,rates,width=.23,label=f)
            for x,y,(n,total) in zip(xpos,rates,counts):axes[1].text(x,y+.02,f'{n}/{total}' if total else 'NE',rotation=90,ha='center',fontsize=7)
        for ax in axes:
            ax.set_xticks(range(5),[f'{lo}–{hi}' for lo,hi in A.BANDS],rotation=40);ax.set_xlabel('Fixed frequency band (Hz)');ax.grid(axis='y',alpha=.2)
        axes[1].set(ylim=(0,1.28),ylabel='Disjoint fraction, descriptive',title='Disjoint/evaluable pair conditions (expected 280 each)');axes[1].legend(fontsize=8)
        fig.suptitle(ch+' — individual VMD frequency-band results; no class-specific selection')
        save(fig,'vmd_frequency_band_results_'+ch+'.png')

def coverage_diagnostics(features,reps):
    from factorial_model import design
    reasons=defaultdict(list)
    for r in features:
        if r['component'] not in ['raw','oscillatory_sum'] or r['analysis_valid']:continue
        reason=r.get('invalid_reason','')
        if not r['decomposition_valid']:reason=(reason+';decomposition_invalid').strip(';')
        reasons[(r['channel'],r['pipeline'],r['state'],reason or 'undefined')].append(r)
    P.table(P.BASE/'diagnostics/main_qc_failure_reasons.csv',[dict(channel=ch,pipeline=m,state=s,reason=reason,failed_signals=len(rr),source_recordings=';'.join(r['recording'] for r in rr)) for (ch,m,s,reason),rr in sorted(reasons.items())])
    rows=[]
    for m,ch,cat in sorted({(r['pipeline'],r['channel'],r['category']) for r in reps}):
        if cat=='residue':continue
        rr=[r for r in reps if (r['pipeline'],r['channel'],r['category'])==(m,ch,cat) and r['analysis_valid']]
        counts={s:sum(r['state']==s for r in rr) for s in P.STATES}
        x,_=design(rr) if rr else (np.zeros((0,13)),{})
        rank=int(np.linalg.matrix_rank(x)) if rr else 0
        rows.append(dict(pipeline=m,channel=ch,category=cat,n_valid_recordings=len(rr),represented_states=sum(n>0 for n in counts.values()),missing_states=';'.join(s for s,n in counts.items() if n==0),valid_per_state=json.dumps(counts),observed_factorial_cells=len({(r['state'],r['speed'],r['load']) for r in rr}),expected_factorial_cells=80,additive_columns=13,additive_rank=rank,additive_residual_df=len(rr)-rank,frequency_compatible=A.frequency_compatible(rr),source_recordings=';'.join(r['recording'] for r in rr)))
    P.table(P.BASE/'diagnostics/factorial_design_coverage.csv',rows)

def fixed_condition_qc_figures(features):
    for ch in CH:
        fig,axes=plt.subplots(4,2,figsize=(14,15))
        for ax,state in zip(axes.flat,P.STATES):
            row=next(r for r in features if (r['pipeline'],r['channel'],r['state'],r['speed'],r['load'],r['repeat'])==('raw',ch,state,30,'High',1))
            if not row.get('array_path'):
                ax.text(.5,.5,'Undefined scaling',ha='center',transform=ax.transAxes);continue
            with np.load(P.resolve(row['array_path'])) as a:
                for qval in [-5.,0.,5.]:
                    qi=int(np.where(a['q']==qval)[0][0]);ax.loglog(a['scales'],a['Fq'][qi],marker='.',markersize=3,label=f'q={qval:g}')
            ax.axvspan(64,512,alpha=.13,color='grey')
            ax.set(title=state+f" / {'VALID' if row['analysis_valid'] else 'INVALID'}; min R²={row.get('min_r2',np.nan):.3f}; stable q={row.get('stable_local_slope_fraction',np.nan):.0%}",xlabel='Scale (samples)',ylabel='Fq');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle(ch+' — RAW Fq audit at preset 30 Hz/High/acquisition 1; shaded range fixed 64–512, no range selection')
        save(fig,'raw_scaling_qc_audit_'+ch+'.png')

def run():
    P.dump(P.BASE/'config/analysis_rules.json',dict(A.ANALYSIS_RULE,states=P.STATES,expected_cells=80,expected_pair_conditions=280,
        direction_baseline='50 Hz/High, unchanged from Step 6; NE if baseline invalid',topology_comparison='after independent Spur analysis; no pooled or mapped classes'))
    sources=json.loads((P.BASE/'config/source_inventory.json').read_text())
    features=[P.typed(r) for r in P.read_table(P.BASE/'features/mfdfa_features.csv')]
    A.mode_matching_audit(features);A.decomposition_audit(features)
    reps,coverage=A.representatives(features,sources);P.table(P.BASE/'features/analysis_representatives.csv',reps)
    coverage_diagnostics(features,reps)
    cells=A.build_cells(reps);pairs=A.comparisons(cells);speed,load=A.condition_variation(cells)
    full=A.fault_operating_effect(cells,pairs);grid=A.matched_operating_grid(cells)
    paired,summary=A.paired_raw_vmd(reps);models,diagnostics=A.models(reps)
    matched,extras,recurrence=extras_and_matched(pairs,speed,load,cells)
    (P.BASE/'figures').mkdir(exist_ok=True)
    figures(features,reps,cells,coverage,paired,models);more_figures(pairs,features,cells);band_figures(pairs,cells,extras);fixed_condition_qc_figures(features)
    from reporting import report
    report(features,reps,cells,pairs,speed,load,full,grid,summary,models,matched,extras,recurrence)
    validate(features,pairs,sources)
    print('STEP 7 COMPLETE: Spur independently analyzed; report, figures, comparison and validation saved',flush=True)

if __name__=='__main__':run()

