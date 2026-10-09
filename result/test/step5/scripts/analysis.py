"""Operating-condition robustness on locked Step 4 representations."""
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(os.environ.get('PHM_OUTPUT_DIR',str(__import__('pathlib').Path(__file__).resolve().parents[4]/'result/test/colab_migration')))/'_work/mplconfig'))
import pipeline as P
import numpy as np
import itertools,json,hashlib
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from factorial_model import factorial,bh_adjust

FEATURES=['delta_alpha','alpha0','delta_h'];METHODS=['raw','vmd','emd']
CONFIG=json.loads((P.BASE/'config/locked_step4_configuration.json').read_text(encoding='utf-8'))
BANDS=CONFIG['bands_hz'];EPS=1e-12
ANALYSIS_RULE=dict(frequency_center_ratio_limit=1.5,representative='largest energy in fixed Step 4 band BEFORE QC',
    complete_cell='two independent acquisitions; no substitute for invalid/missing',
    relative_difference='2*abs(a-b)/(abs(a)+abs(b))',CV='sample std / abs(mean), only if abs(mean)>1e-8',
    bootstrap_draws=1999,seed=6509,primary_channel='output_voltage',
    robustness_verdict='main raw/sum: evaluable complete-cell coverage>=75%; ROBUST if >=2 features have disjoint pair fraction>=75% and fault/operating SD ratio>=1; NOT ROBUST if >=2 features have pair fraction<=25% and fault/operating ratio<1; otherwise CONDITION-DEPENDENT; inadequate coverage = NOT EVALUABLE')

def series_name(row):
    return 'raw' if row['pipeline']=='raw' else row['component']
def frequency_compatible(rows):
    centers=[r['center_hz'] for r in rows if isinstance(r.get('center_hz'),(int,float)) and np.isfinite(r['center_hz'])]
    return not centers or (min(centers)>0 and max(centers)/min(centers)<=1.5)
def roster(rows):return ';'.join(r['recording'] for r in rows)

def mode_matching_audit(features):
    groups=defaultdict(list)
    for r in features:
        if r['component'].startswith('mode'):
            groups[(r['recording'].rsplit('_',1)[0],r['pipeline'],r['channel'])].append(r)
    matches=[]
    for (acquisition,method,ch),rows in sorted(groups.items()):
        first=sorted([r for r in rows if r['repeat']==1],key=lambda r:r['center_hz'])
        second=sorted([r for r in rows if r['repeat']==2],key=lambda r:r['center_hz'])
        a=np.array([r['center_hz'] for r in first]);b=np.array([r['center_hz'] for r in second])
        if not len(a) or not len(b):
            matches.append(dict(acquisition_pair=acquisition,pipeline=method,channel=ch,status='NOT EVALUABLE',reason='missing modes/repeat',source_recordings=roster(rows)));continue
        cost=np.abs(np.log(np.maximum(a[:,None],1)/np.maximum(b[None,:],1)))
        ii,jj=linear_sum_assignment(cost)
        for i,j in zip(ii,jj):
            drift=abs(a[i]-b[j])/max((a[i]+b[j])/2,1.)
            matches.append(dict(acquisition_pair=acquisition,pipeline=method,channel=ch,
                repeat1_component=first[i]['component'],repeat2_component=second[j]['component'],
                repeat1_center_hz=float(a[i]),repeat2_center_hz=float(b[j]),relative_frequency_drift=float(drift),
                frequency_match_valid=bool(drift<=.15),both_scaling_valid=bool(first[i]['analysis_valid'] and second[j]['analysis_valid']),
                source_recordings=first[i]['recording']+';'+second[j]['recording'],status='frequency audit'))
    P.table(P.BASE/'diagnostics/repeat_frequency_matching.csv',matches)

def decomposition_audit(features):
    unique={ (r['recording'],r['pipeline'],r['channel']):r for r in features if r['pipeline']!='raw'}
    output=[]
    for (rid,method,ch),row in sorted(unique.items()):
        if row['reused_step4']:
            folder=P.STEP4/'decompositions/vmd' if method=='vmd' else P.ROOT/'result/test/step4/emd_mfdfa'
            suffix='_diagnostics.json'
            reference=folder/(rid+'_'+ch+suffix)
            info=json.loads(reference.read_text(encoding='utf-8'))['algorithm_info']
            path=reference.relative_to(P.ROOT).as_posix()
        else:
            reference=P.BASE/'diagnostics/recordings'/(rid+'_'+ch+'.json')
            obj=json.loads(reference.read_text(encoding='utf-8'));diag=next(r for r in obj['decompositions'] if r['pipeline']==method)
            info=diag['algorithm_info'];path=reference.relative_to(P.ROOT).as_posix()
        output.append(dict(recording=rid,pipeline=method,channel=ch,state=row['state'],speed=row['speed'],load=row['load'],
            decomposition_valid=row['decomposition_valid'],reconstruction_error=row['reconstruction_error'],
            strict_emd_imf_valid=row.get('strict_emd_imf_valid','') if method=='emd' else '',
            constraint_error=info.get('constraint_error',''),
            iterations=info.get('iterations',''),siftings=sum(d['siftings'] for d in info.get('modes',[])),
            diagnostics_path=path,source_recordings=rid))
    P.table(P.BASE/'diagnostics/decomposition_summary.csv',output)

def representatives(features,sources):
    grouped=defaultdict(list)
    for row in features:grouped[(row['recording'],row['channel'],row['pipeline'])].append(row)
    result=[];coverage=[]
    for source in sources:
        for _,channel,role in P.CHANNELS:
            for method in METHODS:
                rows=grouped[(source['recording'],channel,method)]
                types=['raw'] if method=='raw' else ['oscillatory_sum','residue']+[f'band_{lo}_{hi}_Hz' for lo,hi in BANDS]
                for category in types:
                    if category.startswith('band_'):
                        _,lo,hi,_=category.split('_');lo,hi=float(lo),float(hi)
                        candidates=[r for r in rows if r['component'].startswith('mode') and lo<=r.get('center_hz',-1)<hi]
                        picked=max(candidates,key=lambda r:r['mode_energy']) if candidates else None
                    else:picked=next((r for r in rows if r['component']==category),None)
                    valid=bool(picked and picked['analysis_valid'])
                    audit={k:source[k] for k in ['recording','state','speed','load','repeat']}
                    audit.update(pipeline=method,channel=channel,category=category,analysis_valid=valid,
                        status='EVALUABLE' if valid else 'NOT EVALUABLE',
                        reason='' if valid else ('no component in band' if picked is None else str(picked.get('invalid_reason',''))+'; decomposition_valid='+str(picked['decomposition_valid'])),
                        selected_component=picked['component'] if picked else '',center_hz=picked.get('center_hz','') if picked else '')
                    coverage.append(audit)
                    if picked:result.append(dict(picked,category=category))
    # Explicit expected grid coverage includes missing recordings, if any.
    groups=defaultdict(list)
    for r in coverage:groups[(r['pipeline'],r['channel'],r['category'],r['state'],r['speed'],r['load'])].append(r)
    summaries=[]
    for method in METHODS:
        categories=['raw'] if method=='raw' else ['oscillatory_sum','residue']+[f'band_{lo}_{hi}_Hz' for lo,hi in BANDS]
        for _,ch,_ in P.CHANNELS:
            for category,state,speed,load in itertools.product(categories,P.STATES,P.SPEEDS,P.LOADS):
                rows=groups[(method,ch,category,state,speed,load)]
                valid=[r for r in rows if r['analysis_valid']]
                matched=len(valid)==2 and frequency_compatible([dict(r,center_hz=r['center_hz']) for r in valid if r['center_hz']!=''])
                summaries.append(dict(pipeline=method,channel=ch,category=category,state=state,speed=speed,load=load,
                    expected_recordings=2,found_recordings=len(rows),valid_count=len(valid),valid_fraction=len(valid)/2,
                    evaluable=matched,status='EVALUABLE' if matched else 'NOT EVALUABLE',
                    source_recordings=roster(rows),reason='' if matched else 'missing/invalid or incompatible repeat frequencies'))
    P.table(P.BASE/'diagnostics/mode_band_representatives.csv',coverage)
    P.table(P.BASE/'summaries/qc_coverage.csv',summaries)
    return result,summaries

def build_cells(reps):
    groups=defaultdict(list)
    for r in reps:groups[(r['pipeline'],r['channel'],r['category'],r['state'],r['speed'],r['load'])].append(r)
    cells=[]
    categories_by_method={m:(['raw'] if m=='raw' else ['oscillatory_sum','residue']+[f'band_{lo}_{hi}_Hz' for lo,hi in BANDS]) for m in METHODS}
    for method in METHODS:
        for ch,category,state,speed,load in itertools.product([x[1] for x in P.CHANNELS],categories_by_method[method],P.STATES,P.SPEEDS,P.LOADS):
            rows=groups[(method,ch,category,state,speed,load)]
            valid=[r for r in rows if r['analysis_valid']]
            reason=''
            if len(valid)!=2 or {r['repeat'] for r in valid}!={1,2}:reason='requires two valid independent acquisitions'
            elif not frequency_compatible(valid):reason='repeat frequency ratio exceeds 1.5'
            cell=dict(pipeline=method,channel=ch,category=category,state=state,speed=speed,load=load,
                evaluable=not reason,status='NOT EVALUABLE' if reason else 'EVALUABLE',reason=reason,valid_count=len(valid),
                source_recordings=roster(rows),centers_hz=[r['center_hz'] for r in valid if isinstance(r.get('center_hz'),(int,float))])
            if not reason:
                for f in FEATURES:
                    a=np.array([r[f] for r in valid]);mean=float(a.mean())
                    cell.update({f+'_mean':mean,f+'_std':float(a.std(ddof=1)),f+'_min':float(a.min()),f+'_max':float(a.max()),f+'_range':float(np.ptp(a)),
                        f+'_cv':float(a.std(ddof=1)/abs(mean)) if abs(mean)>1e-8 else None})
            cells.append(cell)
    P.table(P.BASE/'summaries/cell_feature_summary.csv',cells)
    return cells

def comparisons(cells):
    lookup={(c['pipeline'],c['channel'],c['category'],c['state'],c['speed'],c['load']):c for c in cells};out=[]
    identities=sorted({(c['pipeline'],c['channel'],c['category']) for c in cells})
    for method,ch,category in identities:
        for speed,load,(a,b),feature in itertools.product(P.SPEEDS,P.LOADS,itertools.combinations(P.STATES,2),FEATURES):
            ca=lookup[(method,ch,category,a,speed,load)];cb=lookup[(method,ch,category,b,speed,load)]
            reason=''
            if not ca['evaluable'] or not cb['evaluable']:reason='missing/invalid repeat cell'
            centers=ca['centers_hz']+cb['centers_hz']
            if not reason and centers and (min(centers)<=0 or max(centers)/min(centers)>1.5):reason='cross-fault center ratio exceeds 1.5'
            row=dict(pipeline=method,channel=ch,category=category,speed=speed,load=load,pair=a+'-'+b,feature=feature,
                evaluable=not reason,status='NOT EVALUABLE' if reason else 'descriptive',reason=reason,
                source_recordings=ca['source_recordings']+';'+cb['source_recordings'])
            if not reason:
                ma,mb=ca[feature+'_mean'],cb[feature+'_mean'];sa,sb=ca[feature+'_std'],cb[feature+'_std']
                gap=max(ca[feature+'_min'],cb[feature+'_min'])-min(ca[feature+'_max'],cb[feature+'_max'])
                row.update(mean_difference=mb-ma,absolute_fault_difference=abs(mb-ma),
                    symmetric_relative_difference=2*abs(mb-ma)/(abs(ma)+abs(mb)+EPS),
                    J_fault=abs(mb-ma)/(sa+sb+EPS),disjoint_repeat_ranges=bool(gap>0),nonoverlap_gap=float(gap))
            out.append(row)
    # Step 4 generalization: direction and disjoint repeat ranges, same cell logic.
    baseline={(r['pipeline'],r['channel'],r['category'],r['pair'],r['feature']):r for r in out if r['speed']==50 and r['load']=='High'}
    for r in out:
        ref=baseline[(r['pipeline'],r['channel'],r['category'],r['pair'],r['feature'])]
        if r['evaluable'] and ref['evaluable']:
            r['same_direction_as_step4']=bool(np.sign(r['mean_difference'])==np.sign(ref['mean_difference']))
            r['step4_disjoint']=ref['disjoint_repeat_ranges']
            r['step4_disjoint_retained']=bool(ref['disjoint_repeat_ranges'] and r['disjoint_repeat_ranges'] and r['same_direction_as_step4'])
    P.table(P.BASE/'summaries/within_condition_separation.csv',out);return out

def condition_variation(cells):
    speed_rows=[];load_rows=[];groups=defaultdict(list)
    for c in cells:groups[(c['pipeline'],c['channel'],c['category'],c['state'],c['load'])].append(c)
    for key,rows in groups.items():
        valid=sorted([c for c in rows if c['evaluable']],key=lambda c:c['speed']);centers=[x for c in valid for x in c['centers_hz']]
        matched=not centers or (min(centers)>0 and max(centers)/min(centers)<=1.5)
        for f in FEATURES:
            row=dict(zip(['pipeline','channel','category','state','load'],key));row.update(feature=f,valid_speeds=len(valid),evaluable=len(valid)==5 and matched,
                status='EVALUABLE' if len(valid)==5 and matched else 'NOT EVALUABLE',source_recordings=';'.join(c['source_recordings'] for c in valid))
            if row['evaluable']:
                a=np.array([c[f+'_mean'] for c in valid]);mean=float(a.mean())
                row.update(mean_across_speed=mean,range_across_speed=float(np.ptp(a)),std_across_speed=float(a.std(ddof=1)),
                    cv_across_speed=float(a.std(ddof=1)/abs(mean)) if abs(mean)>1e-8 else None,
                    normalized_range=float(np.ptp(a)/(abs(mean)+EPS)),normalized_30_to_50=float((a[-1]-a[0])/(abs(a[0])+EPS)),
                    speed_slope=float(np.polyfit(P.SPEEDS,a,1)[0]))
            speed_rows.append(row)
    lookup={(c['pipeline'],c['channel'],c['category'],c['state'],c['speed'],c['load']):c for c in cells}
    for method,ch,category,state,speed in sorted({k[:-1] for k in lookup}):
        a=lookup[(method,ch,category,state,speed,'High')];b=lookup[(method,ch,category,state,speed,'Low')]
        centers=a['centers_hz']+b['centers_hz'];matched=not centers or (min(centers)>0 and max(centers)/min(centers)<=1.5)
        valid=a['evaluable'] and b['evaluable'] and matched
        for f in FEATURES:
            row=dict(pipeline=method,channel=ch,category=category,state=state,speed=speed,feature=f,evaluable=valid,
                status='EVALUABLE' if valid else 'NOT EVALUABLE',source_recordings=a['source_recordings']+';'+b['source_recordings'])
            if valid:
                ma,mb=a[f+'_mean'],b[f+'_mean'];row.update(high_mean=ma,low_mean=mb,signed_low_minus_high=mb-ma,
                    absolute_difference=abs(mb-ma),relative_difference=2*abs(mb-ma)/(abs(ma)+abs(mb)+EPS))
            load_rows.append(row)
    P.table(P.BASE/'summaries/speed_robustness.csv',speed_rows);P.table(P.BASE/'summaries/load_robustness.csv',load_rows)
    return speed_rows,load_rows

def fault_operating_effect(cells,pairs):
    out=[]
    for method,ch,category in sorted({(c['pipeline'],c['channel'],c['category']) for c in cells}):
        for f in FEATURES:
            within=[];within_cells=[]
            for state in P.STATES:
                valid=[c for c in cells if (c['pipeline'],c['channel'],c['category'],c['state'])==(method,ch,category,state) and c['evaluable']]
                centers=[x for c in valid for x in c['centers_hz']]
                matched=not centers or (min(centers)>0 and max(centers)/min(centers)<=1.5)
                if len(valid)==10 and matched:
                    within.append(float(np.std([c[f+'_mean'] for c in valid],ddof=1)));within_cells.extend(valid)
            selected=[r for r in pairs if (r['pipeline'],r['channel'],r['category'],r['feature'])==(method,ch,category,f) and r['evaluable']]
            row=dict(pipeline=method,channel=ch,category=category,feature=f,complete_states=len(within),evaluable_fault_pairs=len(selected),
                evaluable=len(within)==4 and len(selected)>=30,status='EVALUABLE' if len(within)==4 and len(selected)>=30 else 'NOT EVALUABLE',
                source_recordings=';'.join(sorted({rid for c in within_cells for rid in c['source_recordings'].split(';')})))
            if within and selected:
                w=float(np.median(within));b=float(np.median([r['absolute_fault_difference'] for r in selected]))
                row.update(median_within_fault_operating_sd=w,median_matched_between_fault_difference=b,
                    fault_to_operating_ratio=b/(w+EPS),median_J_fault=float(np.median([r['J_fault'] for r in selected])),
                    disjoint_pair_fraction=float(np.mean([r['disjoint_repeat_ranges'] for r in selected])))
            out.append(row)
    P.table(P.BASE/'summaries/fault_vs_operating_effect.csv',out);return out

def paired_raw_vmd(reps):
    lookup={(r['recording'],r['channel'],r['pipeline'],r['category']):r for r in reps};rows=[]
    for r in reps:
        if r['pipeline']!='raw' or r['category']!='raw':continue
        v=lookup.get((r['recording'],r['channel'],'vmd','oscillatory_sum'))
        valid=r['analysis_valid'] and bool(v and v['analysis_valid'])
        for f in FEATURES:
            row={k:r[k] for k in ['recording','state','speed','load','repeat','channel']};row.update(feature=f,evaluable=valid,source_recordings=r['recording'])
            if valid:
                a,b=r[f],v[f];row.update(raw_feature=a,vmd_feature=b,difference=b-a,
                    symmetric_relative_difference=2*abs(a-b)/(abs(a)+abs(b)+EPS))
            rows.append(row)
    P.table(P.BASE/'summaries/paired_raw_vmd.csv',rows)
    summaries=[]
    for ch,f in itertools.product([c[1] for c in P.CHANNELS],FEATURES):
        valid=[r for r in rows if r['channel']==ch and r['feature']==f and r['evaluable']]
        if len(valid)<3:continue
        a=np.array([r['raw_feature'] for r in valid]);b=np.array([r['vmd_feature'] for r in valid])
        summaries.append(dict(channel=ch,feature=f,paired_recordings=len(valid),pearson_r=float(np.corrcoef(a,b)[0,1]),
            mean_signed_difference=float(np.mean(b-a)),RMSE=float(np.sqrt(np.mean((b-a)**2))),
            median_relative_difference=float(np.median([r['symmetric_relative_difference'] for r in valid])),
            source_recordings=roster(valid)))
    P.table(P.BASE/'summaries/paired_raw_vmd_summary.csv',summaries);return rows,summaries

def matched_operating_grid(cells):
    """Supplemental descriptive effects on identical observed operating cells.

    Does not replace the predeclared full-grid eligibility or verdict rules.
    Inclusion depends only on complete-cell validity, never separation.
    """
    rows=[]
    lookup={(c['pipeline'],c['channel'],c['category'],c['state'],c['speed'],c['load']):c for c in cells}
    for method,ch,category in sorted({k[:3] for k in lookup}):
        common=[]
        for speed,load in itertools.product(P.SPEEDS,P.LOADS):
            cc=[lookup[(method,ch,category,s,speed,load)] for s in P.STATES]
            centers=[v for c in cc for v in c['centers_hz']]
            if all(c['evaluable'] for c in cc) and (not centers or (min(centers)>0 and max(centers)/min(centers)<=1.5)):
                common.append((speed,load))
        cc=[lookup[(method,ch,category,s,speed,load)] for s in P.STATES for speed,load in common]
        all_centers=[v for c in cc for v in c['centers_hz']]
        compatible=not all_centers or (min(all_centers)>0 and max(all_centers)/min(all_centers)<=1.5)
        valid=len(common)>=3 and compatible
        for f in FEATURES:
            row=dict(pipeline=method,channel=ch,category=category,feature=f,common_operating_cells=len(common),
                common_conditions=';'.join(str(s)+'/'+l for s,l in common),evaluable=valid,
                status='partial-grid descriptive' if valid else 'NOT EVALUABLE',
                reason='' if valid else 'fewer than 3 common conditions or incompatible frequency coverage',
                scope='Only observed complete four-state cells; not full 10-condition robustness evidence; verdict rules unchanged',
                source_recordings=';'.join(c['source_recordings'] for c in cc))
            if valid:
                within=[np.std([lookup[(method,ch,category,s,speed,load)][f+'_mean'] for speed,load in common],ddof=1) for s in P.STATES]
                between=[abs(lookup[(method,ch,category,a,speed,load)][f+'_mean']-lookup[(method,ch,category,b,speed,load)][f+'_mean']) for speed,load in common for a,b in itertools.combinations(P.STATES,2)]
                row.update(median_within_fault_operating_sd=float(np.median(within)),
                    median_matched_between_fault_difference=float(np.median(between)),
                    fault_to_operating_ratio=float(np.median(between)/(np.median(within)+EPS)))
            rows.append(row)
    P.table(P.BASE/'summaries/matched_operating_grid_effects.csv',rows);return rows

def models(reps):
    out=[];diags=[]
    for method,ch,category in sorted({(r['pipeline'],r['channel'],r['category']) for r in reps}):
        if category=='residue':continue
        rows=[r for r in reps if (r['pipeline'],r['channel'],r['category'])==(method,ch,category) and r['analysis_valid']]
        # A global band regression needs compatible centers over its entire sample.
        if category.startswith('band_') and not frequency_compatible(rows):
            for f in FEATURES:out.append(dict(pipeline=method,channel=ch,category=category,feature=f,model='all',status='NOT EVALUABLE',reason='frequency coverage incompatible across operating conditions',n=len(rows),source_recordings=roster(rows)))
            continue
        if not rows:continue
        for f in FEATURES:
            results,diagnostics=factorial(rows,f)
            out.extend(dict(pipeline=method,channel=ch,category=category,feature=f,**r) for r in results)
            diags.extend(dict(pipeline=method,channel=ch,category=category,feature=f,**r) for r in diagnostics)
    # One predeclared exploratory family per channel, across representations,
    # features, models and tested terms. No selection based on these results.
    for _,ch,_ in P.CHANNELS:bh_adjust([r for r in out if r['channel']==ch])
    P.table(P.BASE/'summaries/factorial_effects.csv',out);P.table(P.BASE/'diagnostics/factorial_assumptions.csv',diags)
    return out,diags

def run():
    if P.BASE==P.HISTORICAL_BASE: raise RuntimeError('Validated results are read-only; set PHM_OUTPUT_DIR or use execution.runner')
    P.dump(P.BASE/'config/analysis_rules.json',ANALYSIS_RULE)
    sources=json.loads((P.BASE/'config/source_inventory.json').read_text(encoding='utf-8'))
    features=[P.typed(r) for r in P.read_table(P.BASE/'features/mfdfa_features.csv')]
    # Normalize descriptive metadata for reused baseline rows without changing
    # numerical arrays or validity. Original Step 4 labels remain untouched.
    labels={s['recording']:s['label'] for s in sources}
    for row in features:row['label']=labels[row['recording']]
    P.table(P.BASE/'features/mfdfa_features.csv',features)
    P.table(P.BASE/'diagnostics/mfdfa_quality_control.csv',features)
    P.table(P.BASE/'features/valid_features.csv',[r for r in features if r['analysis_valid']])
    mode_matching_audit(features);decomposition_audit(features)
    reps,coverage=representatives(features,sources);P.table(P.BASE/'features/analysis_representatives.csv',reps)
    cells=build_cells(reps);pairs=comparisons(cells);speed,load=condition_variation(cells)
    effects=fault_operating_effect(cells,pairs);paired,paired_summary=paired_raw_vmd(reps)
    matched_operating_grid(cells)
    factorial_rows,assumptions=models(reps)
    from presentation import figures,report,validate_and_cleanup
    figures(features,reps,cells,coverage,paired,factorial_rows)
    report(features,reps,cells,pairs,speed,load,effects,paired_summary,factorial_rows,coverage)
    validate_and_cleanup(features,reps,cells,pairs,sources)
    print('STEP 5 ANALYSIS/REPORT COMPLETE',flush=True)

if __name__=='__main__':run()
