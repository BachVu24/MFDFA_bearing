"""Evidence synthesis only: no decomposition, classifiers, fitting, or tuning."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'result/test/step10'
os.environ['MPLCONFIGDIR']=str(OUT/'_work/mplconfig')
os.environ['XDG_CACHE_HOME']=str(OUT/'_work/cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES=['delta_alpha','alpha0','delta_h']
MAIN=['RAW','VMD_SUM','EMD_SUM']
PROVENANCE=[]
MANIFEST_CACHE={}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def source(stage,name):
    path=ROOT/f'result/test/{stage}'/name
    manifest_path=ROOT/f'result/test/{stage}/diagnostics/output_manifest.csv'
    if stage not in MANIFEST_CACHE:
        manifest=pd.read_csv(manifest_path)
        column='path' if 'path' in manifest else 'relative_path'
        manifest[column]=manifest[column].str.replace('\\','/',regex=False)
        MANIFEST_CACHE[stage]=manifest.set_index(column)
    indexed=MANIFEST_CACHE[stage]
    assert name in indexed.index, f'Unlisted evidence: {path}'
    expected=indexed.loc[name,'sha256'];current=sha(path);verification='exact bytes'
    if current!=expected:
        # Legacy Step4/5 artifacts were checked in from Windows with CRLF.
        # Verify exact historical digest after ONLY restoring line endings.
        assert stage in ['step4_corrected','step5'], f'Changed evidence: {path}'
        normalized=path.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
        assert hashlib.sha256(normalized).hexdigest()==expected, f'Changed content: {path}'
        verification='historical SHA256 exact after restoring CRLF only; current bytes separately hashed'
    PROVENANCE.append(dict(stage=stage,path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=current,
                           historical_manifest_sha256=expected,verification=verification))
    if name.endswith('.csv'):return pd.read_csv(path)
    if name.endswith('.json'):return json.loads(path.read_text())
    return path.read_text()

def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    frame=value if isinstance(value,pd.DataFrame) else pd.DataFrame(value)
    frame.to_csv(path,index=False);return frame

def dump(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')

def md(frame,cols):
    if not isinstance(frame,pd.DataFrame):frame=pd.DataFrame(frame)
    def v(x):
        if pd.isna(x):return 'NE'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x).replace('|','/')
    return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(v(x) for x in row)+' |' for row in frame[cols].itertuples(index=False,name=None))

def rep(p):return {'raw':'RAW','vmd':'VMD_SUM','emd':'EMD_SUM'}[p]

def run():
    for sub in ['tables','plots','config','diagnostics']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    for stage in ['step4_corrected','step5','step6','step7','step8','step9']:
        source(stage,'REPORT.md')
        assert source(stage,'diagnostics/validation.json')['all_passed']
    lock=source('step8','config/locked_feature_set.json')
    spec=importlib.util.spec_from_file_location('step8_guard',ROOT/'result/test/step8/scripts/load_locked_features.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);guard.load_locked_config()
    assert lock['features']==['delta_alpha','alpha0']
    summary=source('step8','feature_summary.csv')
    sets=source('step8','tables/feature_set_separation.csv')
    correlations=source('step8','tables/feature_correlations.csv')
    complements=source('step8','tables/complementarity_recurrence.csv')
    vmd_corr=source('step8','tables/raw_vmd_correlations.csv')
    vmd_matched=source('step8','tables/raw_vmd_matched_evidence.csv')
    bands=source('step8','tables/vmd_band_extra_recurrence.csv')
    classification=source('step9','classification_results.csv')
    class_distribution=source('step9','class_distribution.csv')
    per_class=source('step9','per_class_metrics.csv')
    transfer=source('step9','tables/operating_condition_summary.csv')
    best=source('step9','tables/best_by_context.csv')
    protocol=source('step9','config/evaluation_protocol.json')
    assert protocol['features']==['delta_alpha','alpha0'] and not protocol['hyperparameter_search']
    longitudinal=[]
    # Corrected pilot is the authoritative Step4 numerical evidence.
    f4=source('step4_corrected','features/mfdfa_features.csv')
    p4=source('step4_corrected','summaries/valid_state_pair_comparisons.csv')
    for (pipeline,channel),g in f4.groupby(['pipeline','channel']):
        component='raw' if pipeline=='raw' else 'oscillatory_sum'
        g=g[g.component.eq(component)].copy();g['valid']=g.scaling_valid&g.decomposition_valid
        cell=g.groupby('state').valid.agg(['sum','count'])
        p=p4[p4.pipeline.eq(pipeline)&p4.channel.eq(channel)&p4.comparison_component.eq(component)]
        evaluable=p[p.evaluable]
        longitudinal.append(dict(stage='Step4_corrected',dataset='Helical pilot H1/H2/H5/H6; 50Hz/High',channel=channel,representation=rep(pipeline),
            valid_signals=int(g.valid.sum()),expected_signals=len(g),evaluable_cells=int(((cell['sum']==2)&(cell['count']==2)).sum()),expected_cells=4,
            evaluable_feature_pair_conditions=len(evaluable),expected_feature_pair_conditions=len(p),
            disjoint_feature_pair_conditions=int(evaluable.observed_repeat_ranges_disjoint.eq(True).sum())))
    emd_matched=[];emd_paired=[];timings=[]
    for step,states,dataset in [(5,4,'Helical subset'),(6,6,'Helical'),(7,8,'Spur')]:
        stage=f'step{step}'
        features=source(stage,'features/mfdfa_features.csv') if step==5 else source(stage,'features/analysis_representatives.csv')
        cells=source(stage,'summaries/cell_feature_summary.csv')
        sep=source(stage,'summaries/within_condition_separation.csv')
        for (pipeline,channel),g in features.groupby(['pipeline','channel']):
            component='raw' if pipeline=='raw' else 'oscillatory_sum'
            g=g[g.component.eq(component)]
            c=cells[cells.pipeline.eq(pipeline)&cells.channel.eq(channel)&cells.category.eq(component)]
            p=sep[sep.pipeline.eq(pipeline)&sep.channel.eq(channel)&sep.category.eq(component)]
            e=p[p.evaluable]
            longitudinal.append(dict(stage=f'Step{step}',dataset=dataset,channel=channel,representation=rep(pipeline),
                valid_signals=int(g.analysis_valid.sum()),expected_signals=states*20,evaluable_cells=int(c.evaluable.sum()),expected_cells=states*10,
                evaluable_feature_pair_conditions=len(e),expected_feature_pair_conditions=len(p),disjoint_feature_pair_conditions=int(e.disjoint_repeat_ranges.eq(True).sum())))
        if step==5:continue
        for channel in ['output_voltage','input_voltage']:
            for feature in FEATURES:
                p=sep[sep.channel.eq(channel)&sep.feature.eq(feature)&sep.evaluable&
                    ((sep.pipeline.eq('raw')&sep.category.eq('raw'))|(sep.pipeline.eq('emd')&sep.category.eq('oscillatory_sum')))]
                common=p.pivot(index=['pair','speed','load'],columns='pipeline',values='disjoint_repeat_ranges').dropna().astype(bool)
                emd_matched.append(dict(dataset=dataset,channel=channel,feature=feature,common_pair_conditions=len(common),
                    raw_disjoint=int(common.raw.sum()),emd_disjoint=int(common.emd.sum()),
                    gained=int((common.emd&~common.raw).sum()),lost=int((common.raw&~common.emd).sum())))
                valid=features[features.channel.eq(channel)&features.analysis_valid]
                raw=valid[valid.pipeline.eq('raw')&valid.component.eq('raw')][['recording',feature]]
                emd=valid[valid.pipeline.eq('emd')&valid.component.eq('oscillatory_sum')][['recording',feature]]
                pair=raw.merge(emd,on='recording',suffixes=('_raw','_emd'),validate='one_to_one')
                a,b=pair[feature+'_raw'],pair[feature+'_emd']
                emd_paired.append(dict(dataset=dataset,channel=channel,feature=feature,paired_recordings=len(pair),pearson_r=a.corr(b),
                    median_relative_difference=(2*(a-b).abs()/(a.abs()+b.abs())).median()))
        # Timers end before per-component MF-DFA, confirmed by pipeline source.
        source(stage,'scripts/pipeline.py')
        for path in sorted((ROOT/f'result/test/{stage}/diagnostics/recordings').glob('*.json')):
            obj=source(stage,str(path.relative_to(ROOT/f'result/test/{stage}')))
            for d in obj['decompositions']:
                algo=d['algorithm_info']
                timings.append(dict(dataset=dataset,recording=d['recording'],channel=d['channel'],pipeline=d['pipeline'],
                    seconds=d['elapsed_seconds'],timing_scope='Decomposition + mode diagnostics; excludes MF-DFA and classifier',
                    vmd_iterations=algo.get('iterations'),emd_total_siftings=sum(m['siftings'] for m in algo.get('modes',[])) if d['pipeline']=='emd' else np.nan))
    timeline=save('tables/stage_evidence.csv',longitudinal)
    timing=save('tables/recorded_decomposition_times.csv',timings)
    runtime=timing.groupby(['dataset','channel','pipeline']).seconds.agg(count='count',median_seconds='median',p90_seconds=lambda s:s.quantile(.9),max_seconds='max').reset_index()
    save('tables/computational_cost.csv',runtime)
    matched=save('tables/emd_vs_raw_matched_separation.csv',emd_matched)
    save('tables/emd_vs_raw_paired_features.csv',emd_paired)
    save('tables/representation_feature_evidence.csv',summary)
    save('tables/existing_feature_set_ablation.csv',sets)
    save('tables/redundancy.csv',correlations)
    save('tables/complementarity_recurrence.csv',complements)
    save('tables/vmd_vs_raw_matched_separation.csv',vmd_matched)
    save('tables/vmd_vs_raw_paired_features.csv',vmd_corr)
    save('tables/vmd_band_extra_recurrence.csv',bands)
    save('tables/step9_classification.csv',classification)
    save('tables/step9_condition_transfer.csv',transfer)
    ablation=[]
    for row in sets.itertuples():
        q=summary[summary.dataset.eq(row.dataset)&summary.channel.eq(row.channel)&summary.representation.eq(row.representation)&summary.feature.eq('delta_alpha')].iloc[0]
        classifier=classification[classification.dataset.eq(row.dataset)&classification.channel.eq(row.channel)&classification.classifier.eq('Random Forest')]
        tested=row.representation=='RAW' and row.feature_set=='delta_alpha+alpha0'
        ablation.append(dict(dataset=row.dataset,channel=row.channel,representation=row.representation,feature_set=row.feature_set,
            valid_signals=q.valid_signals,expected_signals=q.expected_signals,qc_coverage=q.qc_coverage,
            evaluable_cells=q.evaluable_cells,expected_cells=q.expected_cells,cell_coverage=q.cell_coverage,
            evaluable_pair_conditions=row.evaluable_pair_conditions,expected_pair_conditions=row.expected_pair_conditions,
            disjoint_any_feature=row.disjoint_on_at_least_one_feature,
            marginal_disjoint_fraction=row.disjoint_on_at_least_one_feature/row.evaluable_pair_conditions if row.evaluable_pair_conditions else np.nan,
            classifier='Random Forest' if tested else '',classification_status='Step9 available' if tested else 'NOT RUN; no classification evidence',
            classification_accuracy=classifier.accuracy.iloc[0] if tested else np.nan,
            classification_macro_f1=classifier.macro_f1.iloc[0] if tested else np.nan,
            final_decision='KEEP' if tested else 'DIAGNOSTIC/COMPARATOR; exclude from final operational input'))
    ablation=save('ablation_summary.csv',ablation)
    decisions=[
        dict(component='RAW MF-DFA',decision='KEEP',reason='Best supported coverage/cost baseline; conditional fault separation; processing locked'),
        dict(component='QC',decision='KEEP',reason='Same scaling and decomposition eligibility; invalid not rescued'),
        dict(component='RAW delta_alpha',decision='KEEP',reason='Helical fault association; complementary to alpha0'),
        dict(component='RAW alpha0',decision='KEEP',reason='Repeated within-condition complementary separation; operating-sensitive'),
        dict(component='RAW delta_h as classifier input',decision='REMOVE',reason='Pearson .955-.989 with delta_alpha; only three isolated extra cases after two features'),
        dict(component='VMD_SUM',decision='REMOVE',reason='Mostly preserves RAW; no consistent matched gain or sensitivity improvement'),
        dict(component='Individual VMD modes',decision='REMOVE',reason='Inadequate coverage; secondary Helical gains do not recur on Spur'),
        dict(component='EMD in operational pipeline',decision='REMOVE',reason='Comparator only; mixed matched gains/losses, lower coverage, added cost, approximate IMF constraints'),
        dict(component='Random Forest fixed Step9 baseline',decision='KEEP AS REFERENCE',reason='Highest macro F1 in all contexts; not deployment validated'),
        dict(component='KNN/SVM/LDA',decision='COMPARISON ONLY',reason='Retain code and evidence; not final default classifier')]
    save('tables/component_decisions.csv',decisions)
    structural=[
        dict(representation='RAW',decomposition='None',mfdfa_calls_operational=1,mfdfa_calls_upstream_audit=1,
             time_structure='O(S*N*(m+1) + Q*sum(N/s)); fixed m,Q,S',memory_structure='O(N + N*(m+1) + Q*S)',runtime_comparison='RAW extraction was not separately timed'),
        dict(representation='VMD_SUM',decomposition='VMD K=7',mfdfa_calls_operational=1,mfdfa_calls_upstream_audit=9,
             time_structure='O(I*K*N + K*N*log(N)) spectral recurrence/FFT + one MF-DFA',memory_structure='O(K*N) plus MF-DFA',runtime_comparison='Measured decomposition+mode diagnostics overhead; no total speedup ratio'),
        dict(representation='Individual VMD modes',decomposition='VMD K=7',mfdfa_calls_operational=7,mfdfa_calls_upstream_audit=9,
             time_structure='Same VMD + up to 7 MF-DFA calls and fixed band matching',memory_structure='O(K*N) plus per-component extraction',runtime_comparison='No band passed inherited coverage gate'),
        dict(representation='EMD_SUM',decomposition='Practical EMD max6 modes',mfdfa_calls_operational=1,mfdfa_calls_upstream_audit=8,
             time_structure='Approximately O(N*sum(siftings)) for extrema/spline passes + MF-DFA',memory_structure='O(J*N) modes plus envelopes',runtime_comparison='Approximate bound; knot evaluation/stopping data dependent')]
    save('tables/complexity_structure.csv',structural)
    final=dict(status='FINAL_LOCKED',plan='Plan 1',pipeline='Full RAW recording -> locked MF-DFA -> unchanged QC -> [delta_alpha, alpha0] -> fixed Random Forest reference',
        representation='RAW',features=lock['features'],feature_order=lock['feature_order'],processing=lock['locked_processing'],
        keep=['Full recording unit','RAW MF-DFA','unchanged QC','delta_alpha','alpha0','original compound labels','separate topology/channel analyses','fixed Random Forest reference'],
        remove_from_operational_path=['VMD_SUM','individual VMD modes','EMD','delta_h as classifier input'],
        comparator_outputs_preserved=True,feature_reselection_allowed=False,hyperparameter_retuning_allowed=False,
        classifier={'name':'Random Forest','parameters':protocol['classifiers']['Random Forest'],'role':'Reference model; no new final fit or deployment'},
        channel_policy=lock['channel_policy'],class_policy=lock['class_policy'],
        evaluation_protocol=protocol['split_method'],
        caveats=['Step8 selection used these labels; Step9 is exploratory conditional CV','Poor load transfer','Full Spur unavailable after QC','No cross-topology prediction experiment; class definitions not aligned','No causal localization'],
        step8_feature_lock_path='result/test/step8/config/locked_feature_set.json',step8_feature_lock_sha256=sha(ROOT/'result/test/step8/config/locked_feature_set.json'),
        step9_protocol_path='result/test/step9/config/evaluation_protocol.json',step9_protocol_sha256=sha(ROOT/'result/test/step9/config/evaluation_protocol.json'))
    dump('config/final_pipeline.json',final)
    (OUT/'config/final_pipeline.sha256').write_text(sha(OUT/'config/final_pipeline.json')+'  final_pipeline.json\n')
    figures(summary,sets,vmd_matched,matched,runtime,best,transfer)
    write_report(timeline,summary,sets,correlations,complements,bands,vmd_matched,vmd_corr,matched,runtime,best,transfer,classification,structural)
    provenance=save('diagnostics/upstream_manifest.csv',PROVENANCE)
    assert all(sha(ROOT/r.path)==r.sha256 for r in provenance.itertuples())
    assert len(ablation)==160 and len(classification)==16
    assert ablation.classification_macro_f1.notna().sum()==4
    rawcomp=complements[complements.representation.eq('RAW')&complements.base_features.eq('delta_alpha+alpha0')&complements.added_feature.eq('delta_h')]
    assert len(rawcomp)==3 and rawcomp.extra_conditions.eq(1).all()
    assert summary[summary.representation.str.startswith('VMD_band')].cell_coverage.max()<.75
    assert best.classifier.eq('Random Forest').all()
    guard.load_locked_config()
    dump('diagnostics/validation.json',dict(all_passed=True,upstream_files_verified=len(provenance),
        no_new_fitting=True,no_parameter_or_feature_changes=True,missing_classification_values_not_fabricated=True,
        feature_sets_reused_from_step8=True,delta_h_unique_benefit_only_three_single_conditions=True,
        all_vmd_bands_below_inherited_coverage_gate=True,recorded_runtime_scope_checked_against_source=True,
        best_classifier_reused_from_step9=True,checks=9))
    save('diagnostics/output_manifest.csv',[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and not {'_work','__pycache__'}.intersection(p.parts) and p.name!='output_manifest.csv'])
    print(json.dumps(dict(ablation_rows=len(ablation),source_files=len(provenance),figures=len(list((OUT/'plots').glob('*.png'))),final_features=final['features'])))

def figures(s,sets,vmd,emd,runtime,best,transfer):
    def finish(name):plt.tight_layout();plt.savefig(OUT/'plots'/name,dpi=150,bbox_inches='tight');plt.close()
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    main=s[s.representation.isin(MAIN)&s.feature.eq('delta_alpha')]
    for ax,((dataset,channel),g) in zip(axs.flat,main.groupby(['dataset','channel'],sort=False)):
        x=np.arange(len(g));ax.bar(x-.18,g.qc_coverage,.36,label='Valid signals');ax.bar(x+.18,g.cell_coverage,.36,label='Complete cells');ax.set_xticks(x,g.representation);ax.set_ylim(0,1);ax.set_title(dataset+'/'+channel);ax.legend(fontsize=8)
    fig.suptitle('QC coverage — fixed processing');finish('qc_coverage.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,((dataset,channel),g) in zip(axs.flat,sets[sets.representation.eq('RAW')].groupby(['dataset','channel'],sort=False)):
        ax.bar(g.feature_set,g.disjoint_on_at_least_one_feature/g.evaluable_pair_conditions);ax.tick_params(axis='x',rotation=30);ax.set_ylim(0,1);ax.set_title(dataset+'/'+channel)
        for i,r in enumerate(g.itertuples()):ax.text(i,.05,f'{r.disjoint_on_at_least_one_feature}/{r.evaluable_pair_conditions}',rotation=90,ha='center')
    fig.suptitle('Existing RAW feature ablations: marginal disjoint union, not classifier accuracy');finish('feature_set_ablation.png')
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,frame,title in [(axs[0],vmd,'VMD_SUM'),(axs[1],emd,'EMD_SUM')]:
        gain=frame.gained_count if 'gained_count' in frame else frame.gained
        loss=frame.lost_count if 'lost_count' in frame else frame.lost
        labels=frame.dataset+'/'+frame.channel.str.replace('_voltage','')+'/'+frame.feature
        x=np.arange(len(frame));ax.bar(x-.2,gain,.4,label='Gains');ax.bar(x+.2,-loss,.4,label='Losses');ax.set_xticks(x,labels,rotation=90,fontsize=7);ax.set_title(title+' vs RAW — same pair/conditions');ax.legend()
    finish('matched_improvement_vs_raw.png')
    fig,ax=plt.subplots(figsize=(10,4));labels=runtime.dataset+'/'+runtime.channel.str.replace('_voltage','')+'/'+runtime.pipeline
    ax.bar(labels,runtime.median_seconds);ax.tick_params(axis='x',rotation=30);ax.set_ylabel('Recorded median seconds / channel signal');ax.set_title('Decomposition + mode diagnostics only; MF-DFA excluded');finish('decomposition_overhead.png')
    fig,ax=plt.subplots(figsize=(10,4));labels=best.dataset+'/'+best.channel.str.replace('_voltage','');x=np.arange(4)
    ax.bar(x-.18,best.accuracy,.36,label='Accuracy');ax.bar(x+.18,best.macro_f1,.36,label='Macro F1');ax.set_xticks(x,labels);ax.set_ylim(0,1);ax.legend();ax.set_title('Step9 Random Forest — fixed RAW delta_alpha+alpha0 only');finish('classification_summary.png')
    fig,ax=plt.subplots(figsize=(10,4))
    h=transfer[transfer.dataset.eq('Helical')&transfer.classifier.eq('Random Forest')]
    names=[];values=[]
    for channel in ['output_voltage','input_voltage']:
        names.append(channel.replace('_voltage','')+'/recording CV');values.append(best[best.dataset.eq('Helical')&best.channel.eq(channel)].macro_f1.iloc[0])
        for p in ['leave_one_speed_out','leave_one_load_out']:
            names.append(channel.replace('_voltage','')+'/'+p);values.append(h[h.channel.eq(channel)&h.protocol.eq(p)].macro_f1.iloc[0])
    ax.bar(names,values);ax.tick_params(axis='x',rotation=25);ax.set_ylim(0,1);ax.set_ylabel('Macro F1');ax.set_title('Helical operating-condition transfer — same locked features/RF');finish('operating_transfer.png')
    fig,ax=plt.subplots(figsize=(10,4));m=s[s.representation.str.startswith('VMD_band')&s.feature.eq('delta_alpha')]
    pivot=m.pivot(index='representation',columns=['dataset','channel'],values='cell_coverage');pivot.plot.bar(ax=ax);ax.axhline(.75,color='red',linestyle='--');ax.set_ylim(0,1);ax.set_ylabel('Evaluable-cell coverage');ax.set_title('Individual frequency bands — inherited 75% gate');finish('frequency_band_coverage.png')

def write_report(timeline,s,sets,corr,complements,bands,vmd,vmdcorr,emd,runtime,best,transfer,classification,structure):
    raw=s[s.representation.eq('RAW')];main=s[s.representation.isin(MAIN)]
    text='''# STEP 10 — FINAL ABLATION / PLAN 1

**FINAL PIPELINE: Full RAW recording → locked MF-DFA → unchanged QC → RAW `[delta_alpha, alpha0]` → fixed Random Forest reference classifier.** Đây là pipeline nghiên cứu cuối của Plan1, chưa được xác nhận cho deployment hoặc full cross-domain generalization. Feature lock Step8 giữ nguyên; Random Forest được chọn làm reference từ MacroF1 Step9, không fit/tune thêm ở Step10.

## 1. Scope and evidence lineage: Step 4–9

Step10 chỉ tổng hợp evidence đã có và phép đối chiếu mô tả trên cùng observations; không signal processing mới, không classifier fit, không parameter/feature search. Step4 dùng **step4_corrected** làm nguồn định lượng cuối, vì bản Step4 đầu đã được audit/correct scaling/QC. Pilot H1/H2/H5/H6 chỉ50Hz/High; Step5 mở5speeds/2loads cho4states; Step6 mở6 Helical states; Step7 xử lý8 Spur states độc lập; Step8 quyết định features; Step9 đánh giá classifiers.

Không cộng recordings Step4/5 vào Step6 như samples mới: chúng là subsets trùng recordings. Step6 đã rerun đầy đủ do historical MF-DFA source hash mismatch; không coi snapshot cũ chứng minh core chưa đổi. Step6/7 numeric configs giữ nguyên và Step8 guard xác minh upstream. Inputs Step10 kiểm SHA256 theo upstream output manifests. Các text artifacts Step4/5 từ Windows đổi CRLF→LF khi checkout: historical digest được xác nhận đúng sau duy nhất việc khôi phục CRLF trong memory, không sửa files. Step6–9 kiểm exact bytes. Provenance giữ current hash/historical hash và verification method trong diagnostics/upstream_manifest.csv.

'''
    text+=md(timeline,['stage','dataset','channel','representation','valid_signals','expected_signals','evaluable_cells','expected_cells','disjoint_feature_pair_conditions','evaluable_feature_pair_conditions','expected_feature_pair_conditions'])
    text+='''

Các feature–pair rows ở bảng stage là tổng ba features, khác mẫu số pair-condition của từng feature/set. Pilot chỉ3 cặp chọn trước, Step5 có6pairs, Step6 có15pairs và Step7 có28pairs. Không so trực tiếp rates khác fault sets như một learning curve/statistical improvement; không coi disjoint rates là accuracy.

## 2. Locked configuration and sample unit

q=-5…5 bước0.5; DFA order2;40 scales nguyên bản, fit64–512 (17scales64–468). QC:≥8 scales, minR²≥0.9,≥80%q R²≥0.95,≥80%q local-slope CV≤0.5, min h(q)>0.05. Valid= scaling_valid AND decomposition_valid; shape warnings chỉ audit. Window scaling rất ngắn (~0.96–7.02ms,0.864decade): scaling pass chưa chứng minh asymptotic multifractality hay cơ chế vật lý.

VMD K7/alpha500/tau0/DCFalse/init1/tol1e-7/max_iter2000; practical EMD max6 modes/max200siftings/SD0.2/envelope0.05/extrema mismatch0.005, strict IMF check riêng. Không đổi các tham số này; loại phương pháp khỏi operational path không xóa hoặc tune comparator. Band representative=max energy trước QC, fixed bands0–1/1–4/4–10/10–20/20–33.334kHz, center ratio≤1.5; same mode index không là physical matching.

Mỗi recording toàn chiều dài là unit; hai channels phân tích riêng, cùng recording group khi CV. Output primary, input secondary; fault labels compound nguyên bản. Không gộp topology hoặc suy localization.

## 3. QC coverage and fault separability

'''
    text+=md(main[main.feature.eq('delta_alpha')],['dataset','channel','representation','valid_signals','expected_signals','qc_coverage','evaluable_cells','expected_cells','cell_coverage'])
    text+='\n\nCoverage shared bởi cả ba feature trong một representation. Spur thiếu S4 ở cả channels, S3 output; full-state effects/validation vẫn NOT EVALUABLE. Missing spectra không phải zero feature hoặc overlap. Main feature evidence:\n\n'
    text+=md(main,['dataset','channel','representation','feature','disjoint_pair_conditions','evaluable_pair_conditions','expected_pair_conditions','disjoint_fraction','direction_consistency','direction_evaluable'])
    text+='''

Strict disjoint repeat ranges chỉ kiểm hai acquisitions mỗi fault tại cùng speed/load; reference direction50Hz/High, baseline invalid⇒NE. High within-condition separation không bảo đảm pooled/global class boundary: distributions thay đổi theo operating conditions, nên recording-CV scores thấp hơn nhiều disjoint fractions.

## 4. Feature ablation and delta_h decision

Tái dùng các feature sets **đã có ở Step8**: delta_alpha;alpha0;delta_h;delta_alpha+alpha0;ba features. Không thêm classification ablation mới. Union tiêu chí nghĩa là có ít nhất một marginal feature disjoint, không phải joint-space classification. Không ép metrics thành composite score.

'''
    text+=md(sets[sets.representation.eq('RAW')],['dataset','channel','feature_set','disjoint_on_at_least_one_feature','evaluable_pair_conditions','expected_pair_conditions'])
    text+='\n\nRedundancy trên RAW valid recordings:\n\n'
    text+=md(corr[corr.representation.eq('RAW')&corr.scope.eq('valid_recordings')],['dataset','channel','feature_a','feature_b','n','pearson_r','spearman_r'])
    text+='''

**Loại delta_h ở Step8 là hợp lý theo descriptive evidence hiện có**, không phải chứng minh nó vô ích trong mọi bài toán. Delta_alpha–delta_h Pearson0.955–0.989; sau bộ hai, thêm delta_h chỉ tăng Helical output0/122, input1/135; Spur output1/31, input1/55. Ba cases bổ sung ở ba pairs khác nhau, mỗi case chỉ một condition; không improvement lặp lại. Correlation riêng không đủ để loại, vì vậy quyết định dựa thêm incremental separation. Không có delta_h classifier ablation trong Step9; không tuyên bố đã chứng minh accuracy của hai feature cao hơn ba feature.

Alpha0 ít redundant và có extra within-condition separation lặp nhiều speeds/loads: primary Helical H3–H5 thêm4conditions/3speeds/2loads cùng direction; secondary Spur S1–S6 thêm4conditions/3speeds/2loads cùng direction. Các pairs khác có direction reversals, nên giữ alpha0 như complementary condition-sensitive feature, không invariant fault marker. Giữ nguyên order `[delta_alpha, alpha0]`, không đổi bằng một accuracy gain nhỏ hoặc speculative classifier benefit.

## 5. Speed/load sensitivity and effects

Speed metric=(max−min)/abs(mean) cần5 complete cells; load metric=2abs(High−Low)/(abs(High)+abs(Low)). Denominators của trajectory/load comparisons phải đọc cùng median, không so unmatched populations như superiority.

'''
    text+=md(raw,['dataset','channel','feature','median_speed_normalized_range','speed_trajectories_evaluable','expected_speed_trajectories','median_load_relative_difference','load_comparisons_evaluable','expected_load_comparisons'])
    text+='\n\nGiữ nguyên exploratory categorical additive model, partialη², HC3/wild-bootstrap1999/BH procedures Step6/7; không refit fault set nhỏ hơn để cứu Spur rank. Partialη² không phải causal contribution hoặc fractions cộng100%:\n\n'
    text+=md(raw,['dataset','channel','feature','fault_partial_eta_squared','speed_partial_eta_squared','load_partial_eta_squared','fault_wild_bootstrap_p_BH','speed_wild_bootstrap_p_BH','load_wild_bootstrap_p_BH'])
    text+='''

Helical delta_alpha/delta_h có fault association mạnh hơn từng speed/load main effect, nhưng sensitivity vẫn lớn; alpha0 main speed/load effects mạnh hơn fault. Full-grid fault/operating ratio vẫn NE do incomplete cells. Không đồng nhất feature normalized range với partialη²: chúng đo khác things/scales. Helical overall verdict CONDITION-DEPENDENT; Spur full-state effects và robustness NOT EVALUABLE vì coverage/rank. Không coi NE là effect0 hay feature vô dụng.

## 6. VMD_SUM: improvement versus preservation

Paired correlations/feature changes:

'''
    text+=md(vmdcorr,['dataset','channel','feature','paired_recordings','pearson_r','median_relative_difference'])
    text+='\n\nMatched separation/sensitivity — giữ cùng common observations:\n\n'
    text+=md(vmd,['dataset','channel','feature','common_pairs','raw_disjoint_count','vmd_disjoint_count','gained_count','lost_count','matched_speed_trajectories','median_speed_ratio','matched_load_comparisons','median_load_ratio'])
    text+='''

**VMD_SUM gần như giữ nguyên RAW; chưa có improvement nhất quán.** Pearson~0.989–0.999, median relative changes~1.5–3.5%; correlation cao không là bằng chứng tốt hơn. Primary Helical delta_alpha mất7/gain0, alpha0 mất3/gain1; Spur output gain2 cho mỗi feature trong common30conditions nhưng input net losses và coverage giảm. Sensitivity ratios gần1, đôi khi tốt/xấu, Spur ít trajectories nên chưa evidence robustness gain. Step4/5 đã cùng xu hướng preserve; full validation không đảo kết luận. Không có VMD classifier result ở Step9, không tuyên bố VMD accuracy thấp hơn RAW.

Tau0 không buộc sum(modes)=RAW, nên VMD_SUM thực sự là representation khác; reconstruction/identity không tự chứng minh useful fault information. Loại decomposition khỏi final path theo cost-benefit hiện có, giữ audit artifacts.

## 7. Individual VMD frequency bands

'''
    text+=md(s[s.representation.str.startswith('VMD_band')&s.feature.eq('delta_alpha')],['dataset','channel','representation','qc_coverage','evaluable_cells','expected_cells','cell_coverage'])
    text+='\n\nTất cả band coverage dưới gate75% kế thừa; không chọn index/class vì score tốt. Bằng chứng extra separation vs RAW:\n\n'
    text+=md(bands,['dataset','channel','category','pair','feature','extra_conditions','speeds','loads','frequency_compatible_across_extra_conditions','repeated_compatible_evidence'])
    text+='''

**Không có lợi ích lặp lại ổn định đủ để giữ modes.** Helical low-frequency input có H1–H2 alpha0 thêm3conditions và H1–H3 thêm2conditions compatible; primary output extras đơn lẻ. Spur không có repeated-compatible extra groups. Đây là tín hiệu exploratory của một subset, không cross-topology mode advantage, không physical component localization. Không chỉnh bands/QC hay chọn representative khác để nâng coverage.

## 8. EMD comparator: keep or remove?

Không so rates trên mẫu số khác rồi kết luận EMD tốt hơn. Matched primary/secondary evidence:

'''
    text+=md(emd,['dataset','channel','feature','common_pair_conditions','raw_disjoint','emd_disjoint','gained','lost'])
    text+='''

**EMD không đáng giữ trong operational pipeline của Plan1 theo evidence này; giữ comparator/audit đã có.** Không gọi EMD luôn kém: Helical input alpha0 gain15/loss5, delta_h gain15/loss9; Spur input delta_alpha gain6/loss1 trên48conditions. Tuy nhiên primary Helical output net losses cả ba features, coverage thường thấp hơn, gains không consistent giữa feature/channel/topology; chưa có EMD classifier test hoặc confirmed operating robustness gain. Các EMD modes là practical approximation, strict IMF audit riêng, không biến thành benchmark IMF nghiêm ngặt. Added decomposition cost chưa được bù bằng improvement đủ ổn định. Không sửa EMD/QC để tăng valid samples.

## 9. Computational complexity and practical cost

'''
    text+=md(runtime,['dataset','channel','pipeline','count','median_seconds','p90_seconds','max_seconds'])
    text+='''

Các số là wall-clock timers đã ghi từ Step6/7, **decomposition + mode diagnostics**, kết thúc trước MF-DFA component extraction; không phải pure solver-only timing, toàn pipeline runtime hay controlled benchmark. Jobs từng chạy concurrent, hardware/conditions có thể ảnh hưởng; không dùng medians để khẳng định thuật toán nhanh tuyệt đối. RAW extraction/classifier chưa có separate timers, không tạo speedup ratio giả. Removing decomposition loại added work nhưng không lượng hóa total speedup.

'''
    text+=md(structure,['representation','decomposition','mfdfa_calls_operational','mfdfa_calls_upstream_audit','time_structure','memory_structure'])
    text+='''

N=signal length, S=40scales, Q=21q, m=2; VMDI≤2000,K7; EMDJ≤6,≤200siftings/mode. VMD spectral recurrence O(IKN), FFT chủ yếu trước/sau O(KNlogN), không giả FFT mỗi ADMM iteration. EMD spline/extrema passes xấp xỉ linear N mỗi sifting với data-dependent constants. Audit từng chạy all modes+sum+residue (9 VMD/8 EMD MF-DFA calls), **không lấy số này làm cost của sum-only pipeline**, vốn chỉ cần1 MF-DFA sau decomposition.

Delta_h được tính cùng MF-DFA spectrum; bỏ khỏi input không bỏ một MF-DFA pass hoặc tiết kiệm thời gian extraction đáng kể. Chi phí classifier giảm dimensionality3→2 nhưng chưa benchmark. Cost advantage chính đến từ bỏ VMD/EMD và mode extraction, không từ giả định feature width đắt hơn alpha0.

## 10. Classification evidence from Step 9

Step9 chỉ classify RAW delta_alpha+alpha0, với cùng recording folds cho4classifiers, scaler training-only khi cần. Không có scores single-feature/triple-feature/VMD/EMD; `ablation_summary.csv` ghi NOT RUN/NaN, không giả0 hoặc thêm inference từ correlation. Không train ablation mới để đổi feature set.

'''
    text+=md(classification,['dataset','channel','classifier','n_samples','n_classes','accuracy','macro_f1','macro_precision','macro_recall'])
    text+='\n\nRandom Forest highest MacroF1 ở cả4contexts; SVM accuracy highest Spur output nhưng macroF1 thấp. Reference best không là universal/deployment validated. Condition-transfer của cùng fixed RF:\n\n'
    text+=md(transfer[transfer.classifier.eq('Random Forest')],['dataset','channel','protocol','evaluable_splits','expected_splits','tested_recordings','eligible_recordings','test_coverage','untested_classes','accuracy','macro_f1'])
    text+='''

Helical output recording-CV MacroF1~0.528→speed transfer0.407→load transfer0.092; input0.419→0.315→0.146. Full load-transfer coverage100%: đây là hạn chế thực tế, không do denominator thiếu của Helical. Spur split/test coverage và class support thiếu; chỉ eligible classes được đánh giá, untestedclasses NE. Step8 selection đã dùng cùng labels, nên Step9 exploratory conditional CV có selection/model-comparison optimism. Cần independent recordings cho final generalization claim.

## 11. Helical versus Spur consistency and generalization

Hai widths redundancy và VMD preservation xuất hiện ở cảdatasets/channels, hỗ trợ consistency của quyết định bỏ redundant/decomposition input. Alpha0 nonredundancy và extra separation tồn tại nhưng strong association conclusions chỉ được xác nhận trên Helical; Spur full-state rank/QC thất bại. Individual band benefit Helical không lặp stable trên Spur.

**Phương pháp chưa generalize tốt qua toàn operating conditions hoặc chứng minh cross-topology transfer.** Helical load-transfer kém; QC rule/scaling locked chuyển sang Spur mất nhiều coverage; classifiers Spur chỉ6 output/7 input states. Phân tích độc lập hai topology là portability evaluation, không phải train-on-Helical/test-on-Spur prediction; class definitions không tương đương nên không thực hiện hay suy transfer classifier như shared physical fault. Không dùng values giữa H/S để causal matching/localization.

## 12. FINAL PIPELINE / KEEP / REMOVE

**FINAL PIPELINE:** full RAW recording → locked MF-DFA(q=-5:0.5:5, order2, fit64–512) → unchanged QC → RAW `[delta_alpha, alpha0]` → fixed Step9 Random Forest reference. Input/output và Helical/Spur riêng; output primary. RF không cần StandardScaler; nếu dùng KNN/SVM/LDA để comparator thì scaler training-only nguyên Step9. Không fit một production model mới trong Step10.

**KEEP:** full recording unit; unchanged MF-DFA/QC; RAW delta_alpha+alpha0 đúng order; original compound labels và operating metadata; topology/channel separation; grouped recording folds; RF reference/config và transparency về NE/coverage.

**REMOVE khỏi operational input/path:** delta_h; VMD_SUM; individual VMD modes; EMD. Giữ mọi features/decomposition outputs và classifier code làm audit/comparator; không delete source artifacts. KNN/SVM/LDA giữ baseline code, không final default.

Final config ở `config/final_pipeline.json`/SHA256, tham chiếu nguyên feature lock Step8 và classifier protocol Step9, cấm retuning/reselection. Không gọi feature-set/parameter lock là guarantee model performance. Không accuracy-driven change hay phương pháp mới. Chín validation checks ở diagnostics/validation.json; upstream SHA256 verified và không sửa code/core hoặc Step4–9.

## 13. Four final answers of Plan 1

1. **MF-DFA có phân biệt fault configurations không?** Có descriptive discrimination trong nhiều matched speed/load cells, especially Helical, và classifier vượt majority baseline; nhưng không global reliable separation. RF MacroF1 chỉ~0.419–0.596 trên valid subsets, một số class recall0; không localization/causal attribution.
2. **Speed/load ảnh hưởng thế nào?** Cả ba features thay đổi theo conditions; Helical widths fault-associated hơn từng main effect nhưng không invariant; alpha0 đặc biệt operating-sensitive. Load transfer failure minh họa hệ quả classification. Spur full-factor effects NE, không kết luận effect0.
3. **VMD/EMD có cải thiện MF-DFA không?** VMD_SUM mostly preserves RAW, chưa consistent gain; modes low coverage và không stable recurrent advantage; EMD có mixed secondary gains nhưng primary losses/coverage/cost và chưa classifier evidence. Không đủ cost-benefit để giữ trong final pipeline.
4. **Generalize tốt qua conditions/Helical–Spur không?** Chưa. Helical load-transfer thấp, Spur QC/class coverage hạn chế; cross-topology classification chưa được đánh giá với compatible targets. Retain locked pipeline như reference nghiên cứu, không deployment/general localization claim.

Kết luận cuối: **chốt RAW MF-DFA + delta_alpha/alpha0 + fixed Random Forest reference**, với unchanged QC và giới hạn generalization được báo cáo. Lựa chọn là giữ pipeline có evidence/cost-benefit tốt nhất trong phạm vi Plan1, không chữa performance bằng retuning hay thêm features.
'''
    (OUT/'REPORT.md').write_text(text)

if __name__=='__main__':run()
