"""Step 8: descriptive feature selection from immutable Step 6/7 evidence.

No signal processing, classifier, fitted feature selector, or parameter search.
Run from repository root with the analysis Python environment.
"""
from pathlib import Path
import hashlib
import itertools
import json
import os

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'result/test/step8'
os.environ['MPLCONFIGDIR'] = str(OUT / '_work/mplconfig')
os.environ['XDG_CACHE_HOME'] = str(OUT / '_work/cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES = ['delta_alpha', 'alpha0', 'delta_h']
KEYS = ['pipeline', 'channel', 'category']
INPUTS = ['features/analysis_representatives.csv',
          'summaries/cell_feature_summary.csv', 'summaries/within_condition_separation.csv',
          'summaries/speed_robustness.csv', 'summaries/load_robustness.csv',
          'summaries/factorial_effects.csv', 'summaries/paired_raw_vmd_summary.csv',
          'summaries/raw_vmd_matched_effects.csv',
          'summaries/vmd_extra_separation_recurrence.csv',
          'config/locked_step4_configuration.json', 'config/analysis_rules.json',
          'config/current_source_hashes.json', 'diagnostics/validation.json']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False)
    return frame

def candidates(frame):
    return frame[((frame.pipeline == 'raw') & (frame.category == 'raw')) |
                 ((frame.pipeline.isin(['vmd', 'emd'])) & (frame.category == 'oscillatory_sum')) |
                 ((frame.pipeline == 'vmd') & frame.category.str.startswith('band_'))].copy()

def rep(pipeline, category):
    if pipeline == 'raw': return 'RAW'
    if category == 'oscillatory_sum': return pipeline.upper() + '_SUM'
    return 'VMD_' + category

def med(frame, column):
    return frame[column].median() if column in frame and len(frame) else np.nan

def markdown(frame, columns=None):
    frame = frame[columns] if columns else frame
    def val(v):
        if pd.isna(v): return 'NE'
        if isinstance(v, (float, np.floating)): return f'{v:.4f}'
        return str(v).replace('|', '/')
    return '| ' + ' | '.join(frame.columns) + ' |\n|' + '|'.join(['---']*len(frame.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(val(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None))

def run():
    for directory in ['tables', 'plots', 'config', 'diagnostics']:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    manifest = []
    data = {}
    configs = []
    for step, dataset in [(6, 'Helical'), (7, 'Spur')]:
        base = ROOT / f'result/test/step{step}'
        old_manifest = pd.read_csv(base / 'diagnostics/output_manifest.csv').set_index('path')
        for name in INPUTS:
            path = base / name
            digest = sha(path)
            assert name in old_manifest.index and digest == old_manifest.loc[name, 'sha256'], f'Upstream changed: {path}'
            manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=digest, bytes=path.stat().st_size))
        assert json.loads((base / 'diagnostics/validation.json').read_text())['all_passed']
        configs.append(json.loads((base / 'config/locked_step4_configuration.json').read_text()))
        data[dataset] = {Path(name).stem: pd.read_csv(base / name) for name in INPUTS if name.endswith('.csv')}
    numeric_keys = ['q', 'scales', 'm', 'fit_interval', 'qc_rule', 'bands_hz', 'fs', 'vmd', 'emd', 'channels', 'preprocessing']
    assert all(configs[0][k] == configs[1][k] for k in numeric_keys)
    summary, pair_rows, corr_rows, add_rows, recurrence, union_rows, paired, matched, qc_state = [], [], [], [], [], [], [], [], []
    for dataset, tables in data.items():
        states = 6 if dataset == 'Helical' else 8
        expected_cells, expected_signals, expected_pairs = states*10, states*20, states*(states-1)//2
        observations = candidates(tables['analysis_representatives'])
        cells = candidates(tables['cell_feature_summary'])
        comparisons = candidates(tables['within_condition_separation'])
        for (pipeline, channel, category), group in observations.groupby(KEYS):
            representation = rep(pipeline, category)
            valid = group[group.analysis_valid]
            mask = (cells.pipeline == pipeline) & (cells.channel == channel) & (cells.category == category)
            cell = cells[mask]
            assert len(cell) == expected_cells and len(group) <= expected_signals
            if not category.startswith('band_'):
                assert len(group) == expected_signals
            assert not group.duplicated(['recording', 'channel']).any()
            for state in sorted(cell.state.unique()):
                g = group[group.state == state]
                state_cells = cell[cell.state == state]
                qc_state.append(dict(dataset=dataset, channel=channel, representation=representation, state=state,
                                     valid_signals=int(g.analysis_valid.sum()), expected_signals=20,
                                     evaluable_cells=int(state_cells.evaluable.sum()), expected_cells=10))
            correlations = {}
            for a, b in itertools.combinations(FEATURES, 2):
                for scope in ['valid_recordings', 'within_speed_load_centered', 'complete_cell_means']:
                    if scope == 'complete_cell_means':
                        values = cell[cell.evaluable][[a+'_mean', b+'_mean']].rename(columns={a+'_mean': a, b+'_mean': b})
                    else:
                        values = valid[[a,b]].copy()
                        if scope == 'within_speed_load_centered':
                            values -= valid.groupby(['speed','load'])[[a,b]].transform('mean')
                    r = values[a].corr(values[b]) if len(values) >= 3 else np.nan
                    rho = values[a].corr(values[b], method='spearman') if len(values) >= 3 else np.nan
                    corr_rows.append(dict(dataset=dataset, channel=channel, representation=representation,
                                          feature_a=a, feature_b=b, scope=scope, n=len(values), pearson_r=r,
                                          spearman_r=rho, descriptive_only=True))
                    if scope == 'valid_recordings': correlations[(a,b)] = r
            pc = comparisons[(comparisons.pipeline == pipeline) & (comparisons.channel == channel) & (comparisons.category == category)]
            for feature in FEATURES:
                p = pc[pc.feature == feature]
                ev = p[p.evaluable]
                disjoint = ev[ev.disjoint_repeat_ranges.eq(True)]
                direction = ev[ev.same_direction_as_50hz_high.notna()]
                speed = tables['speed_robustness']; load = tables['load_robustness']
                def subset(df):
                    return df[(df.pipeline == pipeline)&(df.channel == channel)&(df.category == category)&(df.feature == feature)]
                sp = subset(speed); sp = sp[sp.evaluable]
                ld = subset(load); ld = ld[ld.evaluable]
                effects = subset(tables['factorial_effects'])
                effects = effects[(effects.model == 'fault+speed+load') & (effects.status == 'exploratory')]
                effect = {}
                for term in ['fault','speed','load']:
                    rows = effects[effects.term == term]
                    effect[term+'_partial_eta_squared'] = rows.partial_eta_squared.iloc[0] if len(rows) and 'partial_eta_squared' in rows else np.nan
                    effect[term+'_wild_bootstrap_p_BH'] = rows.wild_bootstrap_p_BH.iloc[0] if len(rows) and 'wild_bootstrap_p_BH' in rows else np.nan
                hard, incomplete = 0, 0
                for pair, pg in p.groupby('pair'):
                    pe = pg[pg.evaluable]; pdj = pe[pe.disjoint_repeat_ranges.eq(True)]
                    hard += bool(len(pe) and len(pdj)/len(pe) < .5)
                    incomplete += len(pe) < 10
                    directions = pe[pe.same_direction_as_50hz_high.notna()]
                    pair_rows.append(dict(dataset=dataset, channel=channel, representation=representation,
                        feature=feature, pair=pair, evaluable_conditions=len(pe), expected_conditions=10,
                        disjoint_conditions=len(pdj), disjoint_fraction=len(pdj)/len(pe) if len(pe) else np.nan,
                        direction_evaluable=len(directions), direction_consistent=int(directions.same_direction_as_50hz_high.eq(True).sum()),
                        direction_consistency=directions.same_direction_as_50hz_high.eq(True).mean() if len(directions) else np.nan,
                        hard_below_50_percent=bool(len(pe) and len(pdj)/len(pe)<.5),
                        coverage_status='COMPLETE' if len(pe)==10 else 'PARTIAL' if len(pe) else 'NOT EVALUABLE'))
                redundancy = max([abs(r) for pair,r in correlations.items() if feature in pair and np.isfinite(r)], default=np.nan)
                summary.append(dict(dataset=dataset, channel=channel, representation=representation, feature=feature,
                    valid_signals=len(valid), available_representatives=len(group), missing_representatives=expected_signals-len(group),
                    expected_signals=expected_signals, qc_coverage=len(valid)/expected_signals,
                    evaluable_cells=int(cell.evaluable.sum()), expected_cells=expected_cells,
                    cell_coverage=float(cell.evaluable.mean()), evaluable_pair_conditions=len(ev), expected_pair_conditions=expected_pairs*10,
                    pair_condition_coverage=len(ev)/(expected_pairs*10), disjoint_pair_conditions=len(disjoint),
                    disjoint_fraction=len(disjoint)/len(ev) if len(ev) else np.nan,
                    disjoint_fraction_of_expected=len(disjoint)/(expected_pairs*10),
                    direction_evaluable=len(direction), direction_consistent=int(direction.same_direction_as_50hz_high.eq(True).sum()),
                    direction_consistency=direction.same_direction_as_50hz_high.eq(True).mean() if len(direction) else np.nan,
                    hard_pairs_below_50_percent=hard, pairs_with_incomplete_coverage=incomplete,
                    completely_unevaluable_pairs=int(p.groupby('pair').evaluable.sum().eq(0).sum()),
                    speed_trajectories_evaluable=len(sp), expected_speed_trajectories=states*2,
                    median_speed_normalized_range=med(sp,'normalized_range'),
                    load_comparisons_evaluable=len(ld), expected_load_comparisons=states*5,
                    median_load_relative_difference=med(ld,'relative_difference'),
                    max_abs_feature_correlation=redundancy,
                    consistency='Helical association evidence; condition-dependent' if dataset=='Helical' and representation=='RAW' else
                                'Partial valid subset only; full-fault inference NOT EVALUABLE' if dataset=='Spur' else 'See coverage and matched evidence',
                    **effect))
            # Multi-feature union is a descriptive range criterion, never classifier performance.
            pvalid = pc[pc.evaluable].pivot(index=['pair','speed','load'], columns='feature', values='disjoint_repeat_ranges')
            if len(pvalid):
                pvalid = pvalid[FEATURES].astype(bool)
            for chosen in [['delta_alpha'], ['alpha0'], ['delta_h'], ['delta_alpha','alpha0'], FEATURES]:
                union_rows.append(dict(dataset=dataset, channel=channel, representation=representation,
                    feature_set='+'.join(chosen), evaluable_pair_conditions=len(pvalid),
                    disjoint_on_at_least_one_feature=int(pvalid[chosen].any(axis=1).sum()) if len(pvalid) else 0,
                    expected_pair_conditions=expected_pairs*10,
                    criterion='At least one marginal disjoint range; not joint geometry or classification accuracy'))
            for added, base_features in [('alpha0',['delta_alpha']), ('delta_h',['delta_alpha']), ('delta_h',['delta_alpha','alpha0'])]:
                extra = pvalid[pvalid[added] & ~pvalid[base_features].any(axis=1)].reset_index() if len(pvalid) else pd.DataFrame()
                for row in extra.itertuples():
                    original = pc[(pc.feature==added)&(pc.pair==row.pair)&(pc.speed==row.speed)&(pc.load==row.load)].iloc[0]
                    add_rows.append(dict(dataset=dataset, channel=channel, representation=representation,
                        base_features='+'.join(base_features), added_feature=added, pair=row.pair, speed=row.speed, load=row.load,
                        signed_difference=original.mean_difference, reference_direction=original.same_direction_as_50hz_high,
                        source_recordings=original.source_recordings))
                if len(extra):
                    for pair, g in extra.groupby('pair'):
                        signs = pc[(pc.feature==added)&(pc.pair==pair)&pc.evaluable].set_index(['speed','load']).mean_difference
                        selected = [np.sign(signs.loc[(r.speed,r.load)]) for r in g.itertuples()]
                        recurrence.append(dict(dataset=dataset, channel=channel, representation=representation,
                            base_features='+'.join(base_features), added_feature=added, pair=pair, extra_conditions=len(g),
                            speeds=g.speed.nunique(), loads=g.load.nunique(),
                            same_sign_across_extra_conditions=len(set(selected))==1,
                            repeated_across_speeds=len(g)>=2 and g.speed.nunique()>=2,
                            repeated_across_speeds_and_loads=len(g)>=2 and g.speed.nunique()>=2 and g.load.nunique()==2,
                            conditions=';'.join(f'{r.speed}/{r.load}' for r in g.itertuples())))
        for key, target in [('paired_raw_vmd_summary',paired), ('raw_vmd_matched_effects',matched)]:
            frame=tables[key].copy();frame.insert(0,'dataset',dataset);target.append(frame)
    summary = save('feature_summary.csv', summary)
    pairs = save('tables/fault_pair_summary.csv', pair_rows)
    corr = save('tables/feature_correlations.csv', corr_rows)
    addition = save('tables/complementary_separation.csv', add_rows)
    recurrence = save('tables/complementarity_recurrence.csv', recurrence)
    union = save('tables/feature_set_separation.csv', union_rows)
    save('tables/qc_by_state.csv',qc_state)
    save('tables/raw_vmd_correlations.csv',pd.concat(paired,ignore_index=True))
    save('tables/raw_vmd_matched_evidence.csv',pd.concat(matched,ignore_index=True))
    mode_recurrence=[]
    for dataset,tables in data.items():
        frame=tables['vmd_extra_separation_recurrence'].copy();frame.insert(0,'dataset',dataset);mode_recurrence.append(frame)
    save('tables/vmd_band_extra_recurrence.csv',pd.concat(mode_recurrence,ignore_index=True))
    save('diagnostics/upstream_manifest.csv',manifest)
    selected = ['delta_alpha','alpha0']
    def decision(r):
        if r.representation == 'RAW': return 'KEEP' if r.feature in selected else 'REJECT_REDUNDANT'
        if r.representation == 'EMD_SUM': return 'COMPARATOR_ONLY'
        if r.representation == 'VMD_SUM': return 'REJECT_NO_CONSISTENT_ADDED_VALUE'
        return 'REJECT_COVERAGE_AND_RECURRENCE'
    summary['decision'] = summary.apply(decision,axis=1)
    summary['selection_tier'] = summary.apply(lambda r: 1 if r.representation=='RAW' and r.feature=='delta_alpha' else
        2 if r.representation=='RAW' and r.feature=='alpha0' else 3 if r.representation=='RAW' else
        4 if r.representation=='VMD_SUM' else 5 if r.representation.startswith('VMD_band') else 6,axis=1)
    def consistency(r, dataset):
        if r.representation == 'RAW':
            if dataset == 'Spur':
                return 'Partial subset only; redundancy persists; full-fault association unconfirmed; alpha0 adds repeated input separation' if r.feature=='alpha0' else 'Partial subset only; full-fault association unconfirmed'
            return 'Repeated complementary separation; operating-condition-sensitive' if r.feature=='alpha0' else 'Fault association stronger than individual operating main effects; redundant with other width'
        if r.representation == 'VMD_SUM': return 'Closely preserves RAW; no consistent additional benefit'
        if r.representation == 'EMD_SUM': return 'Comparator only; assess with coverage'
        return 'Insufficient coverage; repeated benefit limited to secondary input' if dataset=='Helical' else 'Insufficient coverage; no repeated extra separation'
    summary['helical_consistency']=summary.apply(lambda r: consistency(r,'Helical'),axis=1)
    summary['spur_consistency']=summary.apply(lambda r: consistency(r,'Spur'),axis=1)
    # Priority tiers are a transparent qualitative decision, not an optimized score.
    ranking = save('ranking_table.csv',summary.sort_values(['selection_tier','dataset','channel','representation','feature']))
    save('feature_summary.csv',summary)
    core = {key:configs[0][key] for key in numeric_keys}
    locked = dict(schema_version=1, step=8, status='LOCKED', feature_set_type='B. Two-feature set',
        representation='RAW', pipeline='raw', component='raw', features=selected, feature_order=selected,
        primary_channel='output_voltage', secondary_channel='input_voltage',
        channel_policy='Separate analyses with the identical locked feature list; channels share recording groups.',
        decision_basis='Fault association of delta_alpha plus repeated, nonredundant within-condition separation from alpha0; delta_h adds only isolated separations beyond both.',
        feature_reselection_allowed=False, classification_accuracy_may_change_feature_set=False,
        excluded_features=['delta_h'], excluded_representations=['VMD_SUM','individual_VMD_modes','EMD'],
        locked_processing=core, upstream_inputs=manifest,
        sample_unit='Full original recording; no independent windows; split channels of the same recording together.',
        eligibility='analysis_valid == True; finite values of both selected features; no QC override or imputation of QC-failed features.',
        class_policy='Retain original compound configurations. Analyze Helical and Spur separately. Report unavailable classes; no forced full S1-S8 claim.',
        operating_conditions='Preserve speed/load metadata for condition-aware evaluation; alpha0 is condition-sensitive, not a condition-invariant fault marker.',
        evaluation='Feature selection used labels from Step6/7. Unbiased final evaluation requires untouched recordings; reuse grouped CV only as exploratory conditional evaluation. Fit scaling only on training folds.')
    dump('config/locked_feature_set.json',locked)
    (OUT/'config/locked_feature_set.sha256').write_text(sha(OUT/'config/locked_feature_set.json')+'  locked_feature_set.json\n')
    from load_locked_features import locked_inputs
    feature_export, excluded_export=[],[]
    for dataset,tables in data.items():
        for channel in ['output_voltage','input_voltage']:
            meta,X,excluded=locked_inputs(tables['analysis_representatives'],channel)
            export=pd.concat([meta,X],axis=1);export.insert(0,'dataset',dataset);feature_export.append(export)
            excluded.insert(0,'dataset',dataset);excluded_export.append(excluded)
    save('tables/locked_feature_values.csv',pd.concat(feature_export,ignore_index=True))
    save('tables/excluded_raw_recordings.csv',pd.concat(excluded_export,ignore_index=True))
    make_plots(summary,corr,union,pairs,pd.concat(matched),pd.concat(paired))
    write_report(summary,corr,union,recurrence,pairs,pd.concat(matched),pd.concat(paired),data)
    # Final evidence files are checked again to catch accidental concurrent mutation.
    assert all(sha(ROOT / row['path'])==row['sha256'] for row in manifest)
    dump('diagnostics/validation.json',dict(all_passed=True,
        upstream_manifest_verified=len(manifest), processing_configuration_identical=True,
        expected_candidate_rows=len(summary), main_signal_grids_complete=True,
        missing_mode_representatives_included_in_coverage_denominator=True,
        no_duplicate_recording_channel_per_representation=True,
        finite_selected_valid_features=all(np.isfinite(t['analysis_representatives'].query("pipeline == 'raw' and analysis_valid")[selected]).all().all() for t in data.values()),
        selection_basis='descriptive evidence; no classifier or processing changes',
        selected_features=selected))
    assert json.loads((OUT/'diagnostics/validation.json').read_text())['finite_selected_valid_features']
    import subprocess
    import sys
    check=subprocess.run([sys.executable,str(OUT/'scripts/test_feature_lock.py')],capture_output=True,text=True)
    (OUT/'diagnostics/tests.log').write_text(check.stdout+check.stderr)
    dump('diagnostics/tests.json',dict(passed=check.returncode==0,command='scripts/test_feature_lock.py',test_cases=7))
    assert check.returncode==0, check.stdout+check.stderr
    save('diagnostics/output_manifest.csv',[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and '_work' not in p.parts and '__pycache__' not in p.parts and p.name!='output_manifest.csv'])
    print(json.dumps({'candidate_rows':len(summary),'plots':len(list((OUT/'plots').glob('*.png'))),'selected':selected}))

def make_plots(summary,corr,union,pairs,matched,paired):
    plt.rcParams.update({'figure.dpi':130,'font.size':9})
    def finish(name):
        plt.tight_layout();plt.savefig(OUT/'plots'/name,bbox_inches='tight');plt.close()
    main=summary[summary.representation.isin(['RAW','VMD_SUM','EMD_SUM'])]
    for dataset in ['Helical','Spur']:
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for ax,(channel,g) in zip(axs,main[(main.dataset==dataset)&(main.feature=='delta_alpha')].groupby('channel')):
            x=np.arange(len(g));ax.bar(x-.18,g.qc_coverage,.36,label='Valid signals');ax.bar(x+.18,g.cell_coverage,.36,label='Evaluable cells')
            ax.set_xticks(x,g.representation);ax.set_ylim(0,1);ax.set_title(channel);ax.legend();ax.set_ylabel('Coverage')
        fig.suptitle(dataset+' — QC coverage (shared by all three features)');finish(dataset.lower()+'_coverage.png')
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,channel in zip(axs,['output_voltage','input_voltage']):
            g=corr[(corr.dataset==dataset)&(corr.channel==channel)&(corr.representation=='RAW')&(corr.scope=='valid_recordings')]
            matrix=np.eye(3)
            for row in g.itertuples():
                i,j=FEATURES.index(row.feature_a),FEATURES.index(row.feature_b);matrix[i,j]=matrix[j,i]=row.pearson_r
            im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(3),FEATURES,rotation=20);ax.set_yticks(range(3),FEATURES);ax.set_title(channel)
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center')
        fig.colorbar(im,ax=axs.tolist(),fraction=.03);fig.suptitle(dataset+' — RAW valid-recording redundancy');
        fig.savefig(OUT/'plots'/(dataset.lower()+'_correlations.png'),bbox_inches='tight');plt.close()
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        for ax,channel in zip(axs,['output_voltage','input_voltage']):
            g=union[(union.dataset==dataset)&(union.channel==channel)&(union.representation=='RAW')]
            ax.bar(g.feature_set,g.disjoint_on_at_least_one_feature/g.evaluable_pair_conditions)
            ax.tick_params(axis='x',rotation=30);ax.set_ylim(0,1);ax.set_title(channel);ax.set_ylabel('Fraction among evaluable pair/conditions')
            for i,row in enumerate(g.itertuples()):ax.text(i,.05,f'{row.disjoint_on_at_least_one_feature}/{row.evaluable_pair_conditions}',ha='center',rotation=90)
        fig.suptitle(dataset+' — marginal disjoint-range union; not classification accuracy');finish(dataset.lower()+'_feature_set_separation.png')
        fig,axs=plt.subplots(1,2,figsize=(13,6))
        for ax,channel in zip(axs,['output_voltage','input_voltage']):
            g=pairs[(pairs.dataset==dataset)&(pairs.channel==channel)&(pairs.representation=='RAW')]
            matrix=g.pivot(index='pair',columns='feature',values='disjoint_fraction')[FEATURES]
            im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(3),FEATURES);ax.set_yticks(range(len(matrix)),matrix.index);ax.set_title(channel)
            for i,pair in enumerate(matrix.index):
                for j,f in enumerate(FEATURES):
                    r=g[(g.pair==pair)&(g.feature==f)].iloc[0]
                    ax.text(j,i,f'{r.disjoint_conditions}/{r.evaluable_conditions}' if r.evaluable_conditions else 'NE',ha='center',va='center',fontsize=6)
        fig.suptitle(dataset+' — RAW pair separation; counts show coverage');finish(dataset.lower()+'_pair_heatmap.png')
    g=summary[(summary.dataset=='Helical')&(summary.representation=='RAW')]
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,channel in zip(axs,['output_voltage','input_voltage']):
        x=np.arange(3);sub=g[g.channel==channel].set_index('feature').loc[FEATURES]
        for i,term in enumerate(['fault','speed','load']):ax.bar(x+(i-1)*.25,sub[term+'_partial_eta_squared'],.25,label=term)
        ax.set_xticks(x,FEATURES);ax.set_ylim(0,1);ax.set_title(channel);ax.legend();ax.set_ylabel('Additive partial eta squared')
    fig.suptitle('Helical association only — full Spur model NOT EVALUABLE');finish('fault_speed_load_effects.png')
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    for ax,metric in zip(axs,['median_speed_normalized_range','median_load_relative_difference']):
        g=summary[summary.representation=='RAW'];labels=g.dataset+'/'+g.channel.str.replace('_voltage','')+'/'+g.feature
        ax.bar(np.arange(len(g)),g[metric]);ax.set_xticks(np.arange(len(g)),labels,rotation=90,fontsize=7);ax.set_title(metric+' (available complete subsets)')
    finish('operating_sensitivity.png')
    modes=summary[summary.representation.str.startswith('VMD_band')&(summary.feature=='delta_alpha')]
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    for ax,dataset in zip(axs,['Helical','Spur']):
        g=modes[modes.dataset==dataset].pivot(index='representation',columns='channel',values='cell_coverage');g.plot.bar(ax=ax);ax.set_ylim(0,1);ax.axhline(.75,color='red',linestyle='--',label='Inherited 75% coverage gate');ax.set_title(dataset+' — mode-band evaluable cells');ax.legend(fontsize=7)
    finish('vmd_band_coverage.png')
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    labels=matched.dataset+'/'+matched.channel.str.replace('_voltage','')+'/'+matched.feature
    x=np.arange(len(matched))
    axs[0].bar(x-.2,matched.gained_count,.4,label='VMD gains');axs[0].bar(x+.2,-matched.lost_count,.4,label='VMD losses');axs[0].legend();axs[0].set_title('Common pair/conditions: VMD gains and losses')
    axs[1].bar(x,paired.pearson_r);axs[1].set_ylim(.95,1);axs[1].set_title('Paired RAW/VMD Pearson correlation')
    for ax in axs:ax.set_xticks(x,labels,rotation=90,fontsize=7)
    finish('raw_vmd_added_value.png')

def write_report(s,corr,union,recurrence,pairs,matched,paired,data):
    raw=s[s.representation=='RAW'];main=s[s.representation.isin(['RAW','VMD_SUM','EMD_SUM'])]
    rawcorr=corr[(corr.representation=='RAW')&(corr.scope=='valid_recordings')]
    repeated=recurrence[(recurrence.representation=='RAW')&(recurrence.added_feature=='alpha0')&recurrence.repeated_across_speeds]
    rdh=recurrence[(recurrence.representation=='RAW')&(recurrence.base_features=='delta_alpha+alpha0')]
    modes=s[s.representation.str.startswith('VMD_band')&(s.feature=='delta_alpha')]
    report='''# STEP 8 — FEATURE EVALUATION AND FINAL FEATURE SELECTION

**Khóa bộ B: RAW `delta_alpha` + RAW `alpha0`, theo đúng thứ tự này.** Giữ `delta_alpha` làm feature fault-associated chính; giữ `alpha0` vì thông tin bổ sung thực tế trong cùng speed/load, đồng thời thừa nhận nó nhạy operating condition. Loại `delta_h` khỏi input cuối do redundancy và không có bổ sung lặp lại sau hai feature đã giữ. Không train classifier, không chọn feature theo accuracy, không chạy lại hay tune pipeline.

## 1. Candidate features

Ba features: `delta_alpha`, `alpha0`, `delta_h`; RAW, VMD_SUM, 5 VMD frequency bands cố định; EMD_SUM chỉ comparator. Input/output phân tích riêng; output primary, input secondary. 120 Helical recordings và 160 Spur recordings, mỗi recording có hai channels nhưng không tạo hai acquisitions độc lập. Sử dụng toàn bộ recording, không dùng windows làm samples.

Đọc trực tiếp các bảng Step 6/7 đã xác minh SHA-256 với output manifests và kiểm tra upstream validation. Hai cấu hình xử lý giống nhau ở mọi trường số: q=-5:0.5:5, DFA order=2, fit interval64–512 (17 scales thực tế64–468), QC nguyên bản; VMD K7/alpha500/tau0; EMD nguyên bản. Không sửa metadata compound faults. `diagnostics/upstream_manifest.csv` ghi các đầu vào thực sự dùng. Config khóa giữ đầy đủ cấu hình số và hashes đầu vào. Hash core MF-DFA lịch sử trong snapshot không được dùng để khẳng định core hiện tại chưa từng thay đổi; Step6 đã xử lý việc đó bằng rerun, Step7 dùng cùng code thực tế.

## 2. Coverage

QC là điều kiện chung cho cả ba features của một representation, nên coverage không khác nhau giữa ba feature. Cell evaluable cần đúng hai acquisitions valid và frequency compatibility nguyên bản đối với modes. Pair-condition cần cả hai fault cells evaluable. Không thay missing/invalid bằng zero hoặc overlapping.

'''
    report+=markdown(main[main.feature=='delta_alpha'],['dataset','channel','representation','valid_signals','expected_signals','qc_coverage','evaluable_cells','expected_cells','cell_coverage'])
    report+='''

Spur RAW output không có signals valid ở S3/S4; input không có ở S4. Do đó mô hình full S1–S8 bị thiếu rank và không evaluable. Đây là thiếu bằng chứng/coverage, không phải bằng chứng features hoàn toàn vô dụng. Bảng `tables/qc_by_state.csv` cho từng configuration. Mọi tỷ lệ disjoint bên dưới phải đọc cùng mẫu số evaluable và mẫu số expected.

## 3. Fault separability

Giữ nguyên strict disjoint repeat ranges (gap>0) trong cùng speed/load, reference direction50Hz/High. Không gọi tỷ lệ này là classification accuracy. `hard_pairs_below_50_percent` là nhãn mô tả mới của Step8: <50% disjoint trong các conditions evaluable; không phải threshold tune pipeline và không coi pair chưa evaluable là dễ. Pair thiếu coverage ghi riêng. Direction NE nếu baseline invalid; không chọn direction bằng majority kết quả.

'''
    report+=markdown(raw,['dataset','channel','feature','disjoint_pair_conditions','evaluable_pair_conditions','expected_pair_conditions','disjoint_fraction','direction_consistency','direction_evaluable','hard_pairs_below_50_percent','completely_unevaluable_pairs'])
    report+='\n\nCác complementary cases dùng cùng observations và cùng pair-condition đã QC, không suy luận geometry phân loại đa biến. Union chỉ nghĩa là ít nhất một marginal feature có repeat ranges disjoint.\n\n'
    report+=markdown(union[union.representation=='RAW'],['dataset','channel','feature_set','disjoint_on_at_least_one_feature','evaluable_pair_conditions','expected_pair_conditions'])
    report+='\n\n`alpha0` thêm so với `delta_alpha`: Helical output10 conditions, input18; Spur output3, input13. Những bổ sung lặp lại qua nhiều speeds:\n\n'
    report+=markdown(repeated,['dataset','channel','pair','extra_conditions','speeds','loads','same_sign_across_extra_conditions','conditions'])
    report+='''

Trong primary Helical H3–H5, alpha0 bổ sung4 conditions qua3 speeds và2 loads; H4–H6 bổ sung3 conditions qua2 speeds và2 loads. Các cases bổ sung này xuất hiện trong so sánh cùng operating condition, nên không thể bác bỏ toàn bộ alpha0 chỉ vì main speed/load effects lớn. Tuy nhiên direction across extra conditions phải đọc từ bảng; việc tách lặp lại không đồng nghĩa một threshold global ổn định. Spur primary bổ sung alpha0 chỉ đơn lẻ; bằng chứng lặp lại của Spur nằm ở secondary input và subset valid. Không tuyên bố bổ sung generalize toàn S1–S8.

Sau RAW delta_alpha+alpha0, delta_h chỉ thêm Helical input H1–H4 tại50/High, Spur input S3–S7 tại40/High, Spur output S6–S8 tại45/High; mỗi case một condition, Helical primary không thêm case nào. Không đủ bằng chứng giữ feature thứ ba.

## 4. Operating-condition sensitivity

Tái sử dụng mô hình và procedure Step6/7, không refit một fault set nhỏ hơn: additive categorical fault+speed+load, exploratory partial eta squared; HC3 và null-imposed wild bootstrap1999 với BH như upstream. Full interactions/full-grid robustness giữ verdict upstream. Không so partial eta squared như tỷ trọng cộng100%, không giải thích causal. Spur full-factorial effects đều NE do coverage/rank; không điền bằng0 hoặc fit lại trên các states còn lại.

'''
    report+=markdown(raw,['dataset','channel','feature','fault_partial_eta_squared','speed_partial_eta_squared','load_partial_eta_squared','fault_wild_bootstrap_p_BH','speed_wild_bootstrap_p_BH','load_wild_bootstrap_p_BH'])
    report+='\n\nHelical widths có fault association lớn hơn từng speed/load main effect; alpha0 bị speed/load chi phối mạnh hơn fault ở cả hai channels. Association không đồng nghĩa invariance. Complete-grid fault/operating ratios vẫn NE; robustness main representations của Helical là CONDITION-DEPENDENT, Spur là NOT EVALUABLE. Sensitivity trên complete subsets:\n\n'
    report+=markdown(raw,['dataset','channel','feature','median_speed_normalized_range','speed_trajectories_evaluable','expected_speed_trajectories','median_load_relative_difference','load_comparisons_evaluable','expected_load_comparisons'])
    report+='''

Speed metric=(max−min)/abs(mean) cần5 speeds; load metric=2abs(High−Low)/(abs(High)+abs(Low)) cần hai complete cells. Các median dùng các subset khác nhau, không dùng trực tiếp để khẳng định dataset này robust hơn dataset kia. Alpha0 được giữ như feature thông tin bổ sung có điều kiện, không phải fault-only marker. Step9 cần báo cáo theo speed/load và phép thử chuyển operating condition đã xác định trước.

## 5. Redundancy

Pearson và Spearman chỉ trên observations valid, riêng dataset/channel/representation. Có thêm complete-cell means và within-speed/load centered correlations để kiểm tra pooled association. Centering chỉ là chẩn đoán mô tả trên subset valid, không phải biến đổi input khóa hay fitted Spur full-fault model. Không pool Helical/Spur, không coi hai channels độc lập.

'''
    report+=markdown(rawcorr,['dataset','channel','feature_a','feature_b','n','pearson_r','spearman_r'])
    report+='''

Delta_alpha–delta_h Pearson0.955–0.989 trên RAW ở cả hai datasets/channels: hai width features largely redundant. Alpha0–delta_alpha Pearson0.051–0.348, thấp hơn rõ rệt. Không loại chỉ bằng threshold correlation; kết hợp correlation với kiểm tra incremental separability: delta_h không có improvement lặp lại sau delta_alpha+alpha0, alpha0 có. Bảng correlation tất cả candidates/scope trong `tables/feature_correlations.csv`.

## 6. Helical vs Spur consistency

Helical chứng minh width–fault association dưới mô hình exploratory, alpha0 nhạy condition nhưng thêm separation. Spur chỉ cung cấp evidence trong valid subset: width correlation và ít redundancy của alpha0 vẫn thấy; alpha0 có thêm separations lặp lại ở input S1–S6 (4 conditions/3speeds/2loads), S1–S7 (2conditions/2speeds/1load). Full-fault association, overall robustness và full generalization vẫn NOT EVALUABLE. Giữ lựa chọn để Step9 kiểm tra, không coi nó là feature set đã được xác nhận tổng quát trên full Spur. Không so raw magnitudes giữa topology để nhận diện cùng physical fault, không gộp configurations không tương đương.

## 7. RAW vs VMD

So sánh trên paired valid recordings/common pair conditions, không so tỷ lệ có mẫu số khác để gán improvement. Correlation và median relative changes:

'''
    report+=markdown(paired,['dataset','channel','feature','paired_recordings','pearson_r','median_relative_difference'])
    report+='\n\nMatched separation và sensitivity ratios(VMD/RAW; <1 ít nhạy hơn, số trajectories/comparisons phải đọc cùng):\n\n'
    report+=markdown(matched,['dataset','channel','feature','common_pairs','raw_disjoint_count','vmd_disjoint_count','gained_count','lost_count','matched_speed_trajectories','median_speed_ratio','matched_load_comparisons','median_load_ratio'])
    report+='''

VMD_SUM gần như giữ nguyên RAW: correlation cao, changes nhỏ; Helical có net separation losses, Spur output tăng vài cases nhưng input mất và QC giảm. Speed/load changes không nhất quán và Spur denominator rất nhỏ. Không có evidence separation/robustness hoặc thông tin nonredundant đủ rõ để giữ thêm VMD_SUM.

Individual VMD modes vẫn xét frequency bands, không chọn theo index/class. Gate coverage75% kế thừa robustness rule, representative max-energy before QC, center ratio≤1.5; không đổi band hay rescue mode invalid. Coverage:

'''
    report+=markdown(modes,['dataset','channel','representation','qc_coverage','evaluable_cells','expected_cells','cell_coverage'])
    report+='''

Không band nào đạt gate ở cả datasets/channels. Low band từng có extra Helical input alpha0 H1–H2(3conditions), H1–H3(2conditions); Helical primary extra chỉ đơn lẻ. Spur chỉ extra đơn lẻ input S1–S6 widths tại35/High, output S6–S8 alpha0 tại45/High; không có repeated-compatible extra band groups. Loại tất cả mode candidates vì coverage và recurrence, không kết luận modes vật lý vô ích. EMD_SUM coverage và separability trong feature_summary, ranking và charts chỉ comparator; không sửa EMD/QC để cứu coverage, không xét EMD làm feature input cuối theo scope người dùng.

## 8. Final selected feature set

**B — hai features: RAW `[delta_alpha, alpha0]`.** Cùng danh sách cho primary output và secondary input phân tích riêng. Không chọn lại channel bằng test accuracy. Giữ delta_alpha vì width–fault association và separation, giữ alpha0 vì ít redundant và improvement thực tế lặp lại trong cùng speed/load. Không giữ vì giả định classifier sẽ dùng được. Bộ hai là quyết định trước classification dựa trên descriptive evidence, không phải chứng minh đạt một accuracy nào.

`config/locked_feature_set.json` giữ feature order, representation, QC/processing snapshot, upstream hashes, policy và cấm accuracy-driven reselection. `config/locked_feature_set.sha256` là digest; `scripts/load_locked_features.py` kiểm tra digest/expected feature list/representation/upstream hashes và chỉ trả hai RAW columns trên observations eligible. Đây là integrity guard, không phải cơ chế bất biến chống người cố tình sửa cả config và digest. Step9 phải load config/guard; không silently rewrite.

`ranking_table.csv` gồm mọi metric theo dataset/channel, decision và qualitative selection tier:1 RAW delta_alpha,2 RAW alpha0,3 RAW delta_h(rejected),4 VMD_SUM,5 VMD bands,6 comparator. Không cộng các metric không cùng meaning thành score. Helical/Spur consistency được ghi riêng theo dataset; coverage hạn chế giữ NE. Full evidence nằm trong feature_summary.csv, pair/direction tables và redundancy/complementarity tables.

## 9. Features rejected and reasons

- RAW delta_h: rất correlated với delta_alpha; sau bộ hai chỉ thêm3 single-condition cases trên toàn hai datasets/channels, không repeated benefit. Giữ như diagnostic/comparator, không input classifier.
- VMD_SUM mọi features: gần lặp RAW, matched gains/losses và sensitivity không nhất quán, Spur coverage thấp hơn; không thêm thông tin đủ rõ.
- Individual VMD modes: coverage không đạt gate; lợi ích low-frequency không lặp cross-topology và primary evidence đơn lẻ.
- EMD features: comparator phụ theo scope; QC giữ nguyên, không đưa vào selection final.

Không loại alpha0 vì sensitivity đơn thuần; quyết định giữ có bằng chứng complementary within-condition và kèm giới hạn transfer. Không gọi QC-failed Spur features vô dụng.

## 10. Recommendation for Step 9 classification

Load locked config và exact feature order; không feature selection theo classification accuracy, không thử thêm delta_h/VMD rồi giữ theo test. Không tune MF-DFA/scales/QC/decomposition. Phân tích topology riêng với original compound labels; output primary, input secondary, ghi đủ class/condition coverage, không báo full S1–S8 classifier nếu missing states. Full recording là unit; recording_id grouping giữ input/output chung split. Hai acquisitions mỗi cell rất ít; nếu muốn held-out operating-condition test, giữ toàn conditions cùng fold theo protocol định trước. Không đưa fault label/speed/load vào feature vector khóa; speed/load vẫn metadata để stratified diagnostics và condition-transfer evaluation.

Selection Step8 dùng labels và toàn evidence Step6/7. CV trên đúng recordings này chỉ là exploratory evaluation có điều kiện trên feature set đã chọn; không phải untouched estimate của toàn quy trình selection. Cần recordings mới/held-out chưa dùng selection để có final performance estimate độc lập. Mọi scaler/tuning classifier chỉ fit training fold; giữ test chưa dùng. Không impute QC-failed features để cứu class coverage và không suy luận gear/bearing/shaft localization hoặc causal single-fault effect. Classifier work để Step9; Step8 chưa train gì.

**Trả lời cuối: khóa RAW delta_alpha + RAW alpha0 làm input Step9.** Width signal ổn ở Helical; alpha0 thêm information có điều kiện. Khả năng generalize toàn Spur chưa được xác nhận vì QC coverage, nên Step9 phải giữ giới hạn đó trong báo cáo.
'''
    (OUT/'REPORT.md').write_text(report)

if __name__ == '__main__':
    run()
