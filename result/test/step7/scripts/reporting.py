"""Recording-level Spur report; compare qualitative trends without class pooling."""
import pipeline as P
import analysis as A
import numpy as np
import itertools,json
from collections import defaultdict

F=A.FEATURES
CH=[c[1] for c in P.CHANNELS]
def category(m):return 'raw' if m=='raw' else 'oscillatory_sum'
def main(rows,m,ch):return [r for r in rows if r['pipeline']==m and r['channel']==ch and r.get('category')==category(m)]
def med(rows,key,percent=False):
    vals=[r[key] for r in rows if isinstance(r.get(key),(float,int)) and np.isfinite(r[key])]
    return (f'{np.median(vals):.1%}' if percent else f'{np.median(vals):.3f}') if vals else 'NE'
def write_table(lines,headers,rows):
    lines+=['','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    lines+=['| '+' | '.join(str(v) for v in row)+' |' for row in rows]
def roster(rows):return ';'.join(sorted({rid for r in rows for rid in r.get('source_recordings',r.get('recording','')).split(';') if rid}))

def separation_summary(pairs):
    out=[]
    for ch,m,cat in sorted({(r['channel'],r['pipeline'],r['category']) for r in pairs}):
        for pair,f in itertools.product(itertools.combinations(P.STATES,2),F):
            name='-'.join(pair)
            allrows=[r for r in pairs if (r['channel'],r['pipeline'],r['category'],r['pair'],r['feature'])==(ch,m,cat,name,f)]
            valid=[r for r in allrows if r['evaluable']]
            ref=next(r for r in allrows if r['speed']==50 and r['load']=='High')
            other=[r for r in valid if (r['speed'],r['load'])!=(50,'High') and ref['evaluable']]
            signs=[int(np.sign(r['mean_difference'])) for r in valid]
            out.append(dict(channel=ch,pipeline=m,category=cat,pair=name,feature=f,expected_conditions=10,evaluable_conditions=len(valid),
                disjoint_conditions=sum(r['disjoint_repeat_ranges'] for r in valid),overlap_conditions=sum(not r['disjoint_repeat_ranges'] for r in valid),unavailable_conditions=10-len(valid),
                baseline_50_high_evaluable=ref['evaluable'],direction_evaluable_conditions=len(other),same_direction_as_baseline=sum(r['same_direction_as_50hz_high'] for r in other),
                baseline_disjoint=ref.get('disjoint_repeat_ranges',None),retained_disjoint_same_direction=sum(r.get('baseline_disjoint_retained',False) for r in other),
                positive_directions=signs.count(1),negative_directions=signs.count(-1),zero_directions=signs.count(0),
                uniform_direction_across_evaluable_conditions=bool(len(valid)>=2 and len(set(signs))==1 and signs[0]!=0),
                source_recordings=roster(allrows)))
    P.table(P.BASE/'summaries/fault_pair_summary.csv',out)
    return out

def decisions(cells,full):
    out=[]
    for ch,m in itertools.product(CH,A.METHODS):
        cc=main(cells,m,ch);n=sum(r['evaluable'] for r in cc);ee=main(full,m,ch)
        good=sum(r['evaluable'] and r.get('disjoint_pair_fraction',0)>=.75 and r.get('fault_to_operating_ratio',0)>=1 for r in ee)
        bad=sum(r['evaluable'] and r.get('disjoint_pair_fraction',1)<=.25 and r.get('fault_to_operating_ratio',1)<1 for r in ee)
        verdict='NOT EVALUABLE' if n/80<.75 else 'ROBUST' if good>=2 else 'NOT ROBUST' if bad>=2 else 'CONDITION-DEPENDENT'
        out.append(dict(channel=ch,pipeline=m,decision=verdict,complete_cells=n,expected_cells=80,full_grid_evaluable_features=sum(r['evaluable'] for r in ee),source_recordings=roster(cc)))
    P.table(P.BASE/'summaries/robustness_decisions.csv',out)
    return out

def qualitative_comparison(cells,pairs,speed,load,models,paired,matched):
    hbase=P.HISTORICAL_BASE
    hread=lambda path:[P.typed(r) for r in P.read_table(hbase/path)]
    hd=dict(cells=hread('summaries/cell_feature_summary.csv'),pairs=hread('summaries/within_condition_separation.csv'),speed=hread('summaries/speed_robustness.csv'),load=hread('summaries/load_robustness.csv'),models=hread('summaries/factorial_effects.csv'),paired=hread('summaries/paired_raw_vmd_summary.csv'),matched=hread('summaries/raw_vmd_matched_effects.csv'))
    sd=dict(cells=cells,pairs=pairs,speed=speed,load=load,models=models,paired=paired,matched=matched)
    rows=[]
    for topology,data,expected_cells,expected_pairs in [('Helical',hd,60,150),('Spur',sd,80,280)]:
        for ch,m,f in itertools.product(CH,['raw','vmd','emd'],F):
            mm=[r for r in main(data['models'],m,ch) if r.get('model')=='fault+speed+load' and r['feature']==f and r.get('term') in ['fault','speed','load']]
            eta={r['term']:r['partial_eta_squared'] for r in mm}
            ee=[r for r in main(data['pairs'],m,ch) if r['feature']==f and r['evaluable']]
            ss=[r for r in main(data['speed'],m,ch) if r['feature']==f and r['evaluable']];ll=[r for r in main(data['load'],m,ch) if r['feature']==f and r['evaluable']]
            pp=next((r for r in data['paired'] if r['channel']==ch and r['feature']==f),{}) if m=='vmd' else {}
            match=next((r for r in data['matched'] if r.get('channel',ch)==ch and r['feature']==f),{}) if m=='vmd' else {}
            direction='FAULT > each operating main effect' if len(eta)==3 and eta['fault']>max(eta['speed'],eta['load']) else 'OPERATING main effect >= fault' if len(eta)==3 else 'NOT EVALUABLE'
            rows.append(dict(topology=topology,channel=ch,pipeline=m,feature=f,expected_cells=expected_cells,complete_cells=sum(r['evaluable'] for r in main(data['cells'],m,ch)),expected_pair_conditions=expected_pairs,evaluable_pair_conditions=len(ee),
                disjoint_pair_conditions=sum(r['disjoint_repeat_ranges'] for r in ee),disjoint_fraction=float(np.mean([r['disjoint_repeat_ranges'] for r in ee])) if ee else None,
                fault_partial_eta_squared=eta.get('fault'),speed_partial_eta_squared=eta.get('speed'),load_partial_eta_squared=eta.get('load'),association_pattern=direction,
                speed_trajectory_n=len(ss),median_speed_range=float(np.median([r['normalized_range'] for r in ss])) if ss else None,
                load_comparison_n=len(ll),median_load_difference=float(np.median([r['relative_difference'] for r in ll])) if ll else None,
                raw_vmd_correlation=pp.get('pearson_r'),raw_vmd_relative_difference=pp.get('median_relative_difference'),
                matched_vmd_gained=match.get('gained_count'),matched_vmd_lost=match.get('lost_count'),
                scope='Independent gearbox-specific fault sets. Qualitative trend comparison; no pooled classes, no equivalent physical-fault claim.',source_recordings=roster(ee)))
    # This is the only cross-topology table. It has explicit topology-scoped sources.
    P.table(P.BASE/'comparison/helical_vs_spur_summary.csv',rows)
    return rows

def report(features,reps,cells,pairs,speed,load,full,grid,paired,models,matched,extras,recurrence):
    inv=json.loads((P.BASE/'diagnostics/inventory_summary.json').read_text())
    L=['# STEP 7 — Full Spur validation','','## 1. Dataset','',
       f"**{inv['found_unique_recordings']}/160 recordings**; missing={len(inv['missing_recordings'])}, duplicate acquisition keys={len(inv['duplicated_recordings'])}, duplicate numeric signal contents={len(inv['duplicated_content_groups'])}, unexpected files={len(inv['unexpected_files'])}. S1–S8 × 30/35/40/45/50 Hz × High/Low × acquisitions 1/2.",
       f"Full lengths {inv['min_samples']}–{inv['max_samples']} samples ({inv['min_samples']/A.CONFIG['fs']:.6f}–{inv['max_samples']/A.CONFIG['fs']:.6f} s), fs=66,666.7 Hz. 160 recording units, 320 analyzed channel signals. Cột 1=input_voltage, cột 2=output_voltage (primary), cột 3 giữ trong source audit nhưng không phân tích. Hai channels cùng recording không phải hai acquisitions độc lập. Không windows, resampling hoặc truncation.",
       'Labels lấy trực tiếp từ VidData.xls, Sheet1 hàng 5–13. Mỗi component, vị trí và nguyên văn fault type được giữ; cột Good lưu đầy đủ trong config/fault_labels_from_workbook.json. Vị trí metadata không phải localization suy ra từ MF-DFA.']
    write_table(L,['State','Metadata nguyên văn — non-Good components'],[(s,P.LABELS[s]) for s in P.STATES])
    L+=['','Workbook có Total Runs=140 nhưng roster Spur được yêu cầu và audit thực tế là 160; không dùng con số tổng chung đó để loại recordings.','',
        '## 2. Locked configuration','',
        'Bằng đúng Step 6: q=-5…5, step 0.5; DFA order=2; cùng 40 scales; scaling fit 64–512 samples, actual 17 scales 64–468. Δα=max(alpha)−min(alpha), α0=alpha(q=0), Δh=max(hq)−min(hq). Không tune tham số hoặc range trên Spur.',
        'QC giữ nguyên: ≥8 scale points; min R²≥0.90; ≥80% q có R²≥0.95; ≥80% q có local adjacent-slope CV≤0.5; min h(q)>0.05. analysis_valid=scaling_valid AND decomposition_valid. Shape warning giữ riêng; không biến thành gate mới.',
        'VMD: K=7, alpha=500, tau=0, DC=False, init=1, tol=1e-7, max_iter=2000; cùng fused recurrence, no fast-math. EMD practical approximation: max_imfs=6, max_siftings=200, sd_threshold=0.2, envelope_tol=0.05, extrema_relative_tolerance=0.005; strict IMF validity audit riêng. Raw mean removal bên trong MF-DFA; decompositions dùng full signal bỏ global mean.',
        'Chỉ đổi roster/fault contrasts từ 6 Helical states sang 8 Spur states và expected grid 60→80 cells, 15→28 pairs. Config equality, source hashes, method source hashes và Step 6 reference hashes được lưu. Không dùng Helical features làm Spur samples.',
        'RAW=raw/raw; VMD_SUM=vmd/oscillatory_sum; EMD comparator=emd/oscillatory_sum. VMD sum-only reconstruction không bị ép bằng raw khi tau=0; residue giữ audit. Sum+residue identity theo construction không chứng minh sum-only reconstruction tốt.','',
        '## 3. QC','']
    qrows=[];qcstate=[]
    for ch,m in itertools.product(CH,A.METHODS):
        allf=[r for r in features if r['channel']==ch and r['pipeline']==m];rr=[r for r in allf if r['component']==category(m)];cc=main(cells,m,ch)
        qrows.append((ch,m.upper(),f"{sum(r['analysis_valid'] for r in rr)}/{len(rr)} ({sum(r['analysis_valid'] for r in rr)/len(rr):.1%})",f"{sum(r['analysis_valid'] for r in allf)}/{len(allf)}",f"{sum(r['evaluable'] for r in cc)}/80"))
        for s in P.STATES:
            ff=[r for r in allf if r['state']==s];ss=[r for r in rr if r['state']==s];sc=[r for r in cc if r['state']==s]
            qcstate.append(dict(channel=ch,pipeline=m,state=s,main_valid=sum(r['analysis_valid'] for r in ss),main_expected=20,component_valid=sum(r['analysis_valid'] for r in ff),component_total=len(ff),complete_cells=sum(r['evaluable'] for r in sc),expected_cells=10,source_recordings=roster(ss)))
    write_table(L,['Channel','Pipeline','Main signals valid','All components valid','Complete main cells'],qrows)
    P.table(P.BASE/'summaries/qc_by_state.csv',qcstate)
    for ch in CH:
        write_table(L,[ch+' state','RAW valid/20; cells/10','VMD_SUM valid/20; cells/10','EMD_SUM valid/20; cells/10'],[(s,*[f"{r['main_valid']}/20; {r['complete_cells']}/10" for m in A.METHODS for r in qcstate if (r['channel'],r['state'],r['pipeline'])==(ch,s,m)]) for s in P.STATES])
    L+=['','Không xếp hạng methods bằng all-components valid rate: số components khác nhau. Invalid numerical features/arrays vẫn giữ audit; mọi effect/separation summary loại invalid. Full QC theo recording/state/speed/load/band ở diagnostics và summaries/qc_coverage.csv.','',
        '## 4. S1–S8 feature results','',
        'Bảng pooled mean±sample SD chỉ mô tả valid recordings qua observed conditions; không dùng nó để suy ra separation giữa faults ở conditions khác nhau. Cell means±SD và repeat ranges theo exact speed/load nằm trong cell_feature_summary.csv và figures.']
    pooled=[]
    for ch,m,s in itertools.product(CH,A.METHODS,P.STATES):
        rr=[r for r in features if (r['channel'],r['pipeline'],r['component'],r['state'])==(ch,m,category(m),s) and r['analysis_valid']]
        pooled.append((ch,m.upper(),s,len(rr),*[f"{np.mean([r[f] for r in rr]):.4f}±{np.std([r[f] for r in rr],ddof=1):.4f}" if len(rr)>1 else 'NE' for f in F]))
    write_table(L,['Channel','Pipeline','State','Valid N/20',*F],pooled)
    L+=['','## 5. Fault-pair separation','',
        '**28 pairs × 10 conditions × 3 features**, riêng mỗi channel/representation. Cùng speed/load, cả 4 independent acquisitions phải valid; modes còn cần compatible frequency centers. Disjoint iff max(A)<min(B) hoặc max(B)<min(A), strict inequality. Overlap là không tách; invalid/missing là NOT EVALUABLE, không gộp thành overlap. Đây là descriptive separation, không phải classification accuracy hay statistical significance.',
        'Direction consistency giữ baseline 50 Hz/High từ Step 6: sign(mean(B)−mean(A)) giống baseline trên conditions khác khi cả baseline và condition valid. Baseline không valid thì baseline consistency NE; không chọn baseline khác theo kết quả. Ghi thêm positive/negative counts và uniform direction trên các conditions evaluable, không thay baseline rule.']
    ps=separation_summary(pairs)
    globalrows=[]
    for ch,m,f in itertools.product(CH,A.METHODS,F):
        rr=[r for r in main(pairs,m,ch) if r['feature']==f and r['evaluable']];dd=[r for r in main(ps,m,ch) if r['feature']==f]
        dn=sum(r['direction_evaluable_conditions'] for r in dd);ds=sum(r['same_direction_as_baseline'] for r in dd)
        globalrows.append((ch,m.upper(),f,f'{len(rr)}/280',f"{sum(r['disjoint_repeat_ranges'] for r in rr)}/{len(rr)} ({np.mean([r['disjoint_repeat_ranges'] for r in rr]):.1%})" if rr else 'NE',f'{ds}/{dn}' if dn else 'NE'))
    write_table(L,['Channel','Pipeline','Feature','Evaluable/280','Disjoint/evaluable','Same baseline direction/evaluable others'],globalrows)
    for ch,m in itertools.product(CH,A.METHODS):
        L+=['',f'### {ch} — {m.upper()}','', 'Mỗi ô = disjoint/evaluable/10 expected conditions. Direction chi tiết lưu ở fault_pair_summary.csv.']
        rows=[]
        for a,b in itertools.combinations(P.STATES,2):
            name=a+'-'+b
            rr=[next(r for r in main(ps,m,ch) if r['pair']==name and r['feature']==f) for f in F]
            rows.append((name,*[f"{r['disjoint_conditions']}/{r['evaluable_conditions']}/10" for r in rr]))
        write_table(L,['Pair',*F],rows)
    L+=['','Descriptive feature ranking:']
    for ch,m in itertools.product(CH,['raw','vmd']):
        score={f:sum(r['disjoint_conditions'] for r in main(ps,m,ch) if r['feature']==f) for f in F}
        best=[f for f in F if score[f]==max(score.values())]
        L+=['',f"{ch}/{m.upper()}: nhiều disjoint conditions nhất = {', '.join(best)}; "+', '.join(f'{f}={score[f]}' for f in F)+'. Ranking within-condition không chứng minh operating invariance.']
    L+=['','## 6. Speed/load effects','',
        'Giữ Step 6 metrics: normalized speed range=(max−min)/abs(mean) trên đủ 5 complete cells; load difference=2|High−Low|/(|High|+|Low|) cùng fault/speed; thiếu valid/matched repeat/cell thì trajectory/comparison NE. Full-grid fault/operating ratio=median matched between-fault mean gap / median within-fault SD của 10 cell means; phải đủ 8 states. Supplemental partial-grid dùng common conditions đầy đủ tất cả 8 states, ≥3 conditions, matching trước kết quả, không interpolation hoặc thay full-grid verdict bằng partial grid.']
    varied=[]
    for ch,m,f in itertools.product(CH,A.METHODS,F):
        ss=[r for r in main(speed,m,ch) if r['feature']==f and r['evaluable']];ll=[r for r in main(load,m,ch) if r['feature']==f and r['evaluable']]
        gg=next(r for r in main(grid,m,ch) if r['feature']==f);ff=next(r for r in main(full,m,ch) if r['feature']==f)
        varied.append((ch,m.upper()+'/'+f,f"{med(ss,'normalized_range',True)} ({len(ss)}/16)",f"{med(ll,'relative_difference',True)} ({len(ll)}/40)",f"{ff['fault_to_operating_ratio']:.3f}" if ff['evaluable'] else 'NE',f"{gg['common_operating_cells']}/10; "+(f"{gg['fault_to_operating_ratio']:.3f}" if gg['evaluable'] else 'NE')))
    write_table(L,['Channel','Pipeline/feature','Speed range median (N/16)','Load difference median (N/40)','Full-grid fault/operating ratio','Common grid; partial ratio'],varied)
    L+=['','Categorical fault+speed+load dùng sum-to-zero Helmert contrasts; additive partial η²=incremental SS/(incremental SS+residual SS), không cộng thành total variance. Full fault*speed*load chỉ fit khi đủ 80 cells, full rank và ≥20 residual df. Với partial coverage, additive main effect không loại trừ interactions và không phải causal effect.',
        'Giữ OLS, HC3 Wald, null-imposed Rademacher wild bootstrap 1,999 draws, seed 6509, HC2-scaled reduced residuals, studentized HC3 Wald; BH riêng mỗi channel trên family representations/features/models/terms. Wild/HC3 exploratory, classical p chỉ reference. Shapiro, heteroskedasticity LM và residual plots giữ diagnostics; 2 repeats/cell hạn chế inference.']
    effectrows=[];patterns={}
    for ch,m,f in itertools.product(CH,A.METHODS,F):
        rr=[r for r in main(models,m,ch) if r['feature']==f and r.get('model')=='fault+speed+load' and r.get('term') in ['fault','speed','load']]
        rr=sorted(rr,key=lambda r:['fault','speed','load'].index(r['term']))
        eta='/'.join(f"{r['partial_eta_squared']:.3f}" for r in rr) if len(rr)==3 else 'NE';p='/'.join(f"{r['wild_bootstrap_p_BH']:.4f}" for r in rr) if len(rr)==3 else 'NE'
        fullok=any(r.get('model')=='fault*speed*load' and r['feature']==f and r['status']=='exploratory' for r in main(models,m,ch))
        effectrows.append((ch,m.upper()+'/'+f,eta,p,'evaluable' if fullok else 'NOT EVALUABLE'))
        if len(rr)==3:patterns[ch,m,f]=[r['partial_eta_squared'] for r in rr]
    write_table(L,['Channel','Pipeline/feature','η² fault/speed/load','Wild p BH fault/speed/load','Full interactions'],effectrows)
    design_path=P.BASE/'diagnostics/factorial_design_coverage.csv'
    if design_path.exists():
        design_rows=[P.typed(r) for r in P.read_table(design_path)]
        write_table(L,['Channel','Main pipeline','Valid recording N','Represented states/8','Missing states','Additive rank/13','Observed cells/80'],[
            (r['channel'],r['pipeline'].upper(),r['n_valid_recordings'],r['represented_states'],r['missing_states'] or '—',r['additive_rank'],r['observed_factorial_cells'])
            for r in design_rows if r['category']==category(r['pipeline'])])
    L+=['','Không bỏ states QC-invalid để fit một fault model với ít classes hơn. Khi thiếu states/rank, toàn bộ S1–S8 factorial result là NOT EVALUABLE; không diễn giải thành fault yếu hơn speed/load. Main QC failure reasons có recording-level trace tại diagnostics/main_qc_failure_reasons.csv.']
    L+=['','**Kiểm tra kết luận Helical trên Spur độc lập:**','']
    for ch in CH:
        for f in F:
            eta=patterns.get((ch,'raw',f))
            verdict='NE' if eta is None else 'fault mạnh hơn từng main speed/load effect' if eta[0]>max(eta[1:]) else 'ít nhất một operating main effect mạnh bằng hoặc hơn fault'
            L.append(f'- {ch}, RAW {f}: {verdict}.')
    L+=['','## 7. Raw vs VMD','', 'Paired valid recordings only; relative feature difference=2|Raw−VMD|/(|Raw|+|VMD|). Correlation cao không tự chứng minh improvement.']
    write_table(L,['Channel','Feature','Paired N','Pearson r','Median relative difference','Mean signed change','RMSE'],[(r['channel'],r['feature'],r['paired_recordings'],f"{r['pearson_r']:.4f}",f"{r['median_relative_difference']:.1%}",f"{r['mean_signed_difference']:.5f}",f"{r['RMSE']:.5f}") for r in paired])
    mr=[];vmdverdicts={}
    for r in matched:
        fmt=lambda key:f"{r[key]:.3f}" if r[key] is not None else 'NE'
        mr.append((r['channel'],r['feature'],r['common_pairs'],f"{r['raw_disjoint_count']}/{r['vmd_disjoint_count']}",f"{r['gained_count']}/{r['lost_count']}",f"{fmt('median_speed_ratio')} ({r['matched_speed_trajectories']})",f"{fmt('median_load_ratio')} ({r['matched_load_comparisons']})"))
    write_table(L,['Channel','Feature','Common pair-conditions','Raw/VMD disjoint','Gained/lost','Speed ratio (N)','Load ratio (N)'],mr)
    L+=['','Sensitivity ratios VMD/Raw <1 là giảm sensitivity, tính trên đúng intersection trajectories/load comparisons. Separation chỉ trên common valid pair-conditions, không trộn coverage gain thành separation gain. Giữ rule diễn giải Step 6: consistent improvement đòi tăng separation và giảm speed/load sensitivity trên cả 3 features; preservation dựa trên r≥0.95, đọc kèm feature differences và gains/losses.']
    for ch in CH:
        rr=[r for r in matched if r['channel']==ch];pp=[r for r in paired if r['channel']==ch]
        improve=len(rr)==3 and all(r['gained_count']>r['lost_count'] and r['median_speed_ratio'] is not None and r['median_speed_ratio']<1 and r['median_load_ratio'] is not None and r['median_load_ratio']<1 for r in rr)
        preserve=len(pp)==3 and all(r['pearson_r']>=.95 for r in pp)
        verdict='cải thiện nhất quán' if improve else 'gần như giữ nguyên RAW; không có cải thiện nhất quán' if preserve else 'representation thay đổi; không có cải thiện nhất quán'
        gain=sum(r['gained_count'] for r in rr);loss=sum(r['lost_count'] for r in rr)
        vmdverdicts[ch]=verdict
        L+=['',f"{ch}: **{verdict}**. Matched feature-pair cases gained={gain}, lost={loss}; "+('separation kém đi tổng thể' if loss>gain else 'separation tăng tổng thể' if gain>loss else 'tổng số disjoint matched cases giữ nguyên')+'.']
    L+=['','## 8. VMD modes','',
        'Giữ fixed bands 0–1000, 1000–4000, 4000–10000, 10000–20000, 20000–33334 Hz. Đại diện = mode energy lớn nhất trong band **trước QC**, không chọn class-specific hoặc rescue bằng mode khác. Repeat/cross-fault centers phải ratio≤1.5; across-condition recurrence cũng ratio≤1.5. Mode rank không chứng minh same physical mode.',
        'Extra separation: RAW cùng condition evaluable nhưng overlap; VMD band evaluable và disjoint. Repeated extra: cùng channel/band/pair/feature qua ≥2 conditions và frequency compatible. Báo riêng distinct speeds và loads; không gộp extras khác feature thành một kết quả lặp.']
    bands=[]
    for ch,(lo,hi) in itertools.product(CH,A.BANDS):
        cat=f'band_{lo}_{hi}_Hz';cc=[r for r in cells if (r['channel'],r['pipeline'],r['category'])==(ch,'vmd',cat)]
        pp=[r for r in pairs if (r['channel'],r['pipeline'],r['category'])==(ch,'vmd',cat) and r['evaluable']]
        xx=[r for r in extras if (r['channel'],r['pipeline'],r['category'])==(ch,'vmd',cat) and r['extra_descriptive_separation']]
        bands.append((ch,f'{lo}–{hi}',f"{sum(r['evaluable'] for r in cc)}/80",f'{len(pp)}/840',sum(r['disjoint_repeat_ranges'] for r in pp),len(xx)))
    write_table(L,['Channel','Band Hz','Complete cells/80','Evaluable feature-pair conditions/840','Disjoint','Extra vs RAW'],bands)
    repeated=[r for r in recurrence if r['repeated_compatible_evidence']]
    write_table(L,['Channel','Band','Pair/feature','Extra conditions','Speeds/loads','Conditions'],[(r['channel'],r['category'],r['pair']+'/'+r['feature'],r['extra_conditions'],f"{r['speeds']}/{r['loads']}",r['conditions']) for r in repeated] or [('—','—','No compatible repeated extra',0,'—','—')])
    L+=['',f'{len(repeated)} repeated compatible groups. Toàn bộ single/repeated/incompatible extras ghi trong vmd_extra_separation_recurrence.csv. Coverage thấp và same-dataset discovery không chứng minh generalized improvement hoặc localization.','',
        '## 9. EMD','',
        'Comparator phụ, practical approximation và QC giữ nguyên; không thay EMD để tăng coverage. Main EMD features/separation/effects được báo với cùng denominator/matching rules; individual-band coverage ở qc_coverage.csv. Strict IMF audit khác practical convergence và scaling validity.']
    dd=[P.typed(r) for r in P.read_table(P.BASE/'diagnostics/decomposition_summary.csv')]
    write_table(L,['Channel','Method','Decomposition valid/160','Sum-only reconstruction error median/max','Strict IMF valid'],[(ch,m.upper(),f"{sum(r['decomposition_valid'] for r in dd if r['channel']==ch and r['pipeline']==m)}/160",f"{np.median([r['reconstruction_error'] for r in dd if r['channel']==ch and r['pipeline']==m]):.2%}/"+f"{max(r['reconstruction_error'] for r in dd if r['channel']==ch and r['pipeline']==m):.2%}",f"{sum(r['strict_emd_imf_valid'] for r in dd if r['channel']==ch and r['pipeline']==m)}/160" if m=='emd' else 'N/A') for ch,m in itertools.product(CH,['vmd','emd'])])
    L+=['','## 10. Helical vs Spur comparison','',
        'So sánh sau khi phân tích Spur độc lập. Không gộp Helical/Spur thành cùng class, không coi Hn tương ứng Sn, không so sánh raw feature magnitudes để tuyên bố cùng physical fault. Cùng speed/load/procedure nhưng topology, gear tooth counts và fault definitions thay đổi cùng nhau: không cô lập causal topology effect.',
        'Bảng dưới so sánh pattern và descriptive metrics với denominator riêng. Partial η² khác số fault levels/coverage; các giá trị không phải kiểm định sự bằng nhau giữa gearbox types. BH families cũng riêng mỗi dataset/channel.']
    compare=qualitative_comparison(cells,pairs,speed,load,models,paired,matched)
    rows=[]
    for ch,f in itertools.product(CH,F):
        rr=[r for r in compare if r['channel']==ch and r['pipeline']=='raw' and r['feature']==f]
        vals=[]
        for topology in ['Helical','Spur']:
            r=next(r for r in rr if r['topology']==topology)
            vals.append(r['association_pattern'])
            vals.append('/'.join(f"{r[k]:.3f}" if r[k] is not None else 'NE' for k in ['fault_partial_eta_squared','speed_partial_eta_squared','load_partial_eta_squared']))
        rows.append((ch,f,*vals,'same qualitative pattern' if rr[0]['association_pattern']==rr[1]['association_pattern'] and rr[0]['association_pattern']!='NOT EVALUABLE' else 'pattern differs / incomplete'))
    write_table(L,['Channel','RAW feature','Helical pattern','Helical η² F/S/L','Spur pattern','Spur η² F/S/L','Transfer'],rows)
    compsep=[]
    for r in compare:
        if r['pipeline'] in ['raw','vmd']:
            compsep.append((r['topology'],r['channel'],r['pipeline'].upper(),r['feature'],f"{r['complete_cells']}/{r['expected_cells']}",f"{r['disjoint_pair_conditions']}/{r['evaluable_pair_conditions']}/{r['expected_pair_conditions']}",f"{r['disjoint_fraction']:.1%}" if r['disjoint_fraction'] is not None else 'NE'))
    write_table(L,['Topology','Channel','Pipeline','Feature','Complete cells','Disjoint/evaluable/expected conditions','Disjoint fraction'],compsep)
    L+=['','Feature operating sensitivity và VMD preservation/gains/losses cho từng topology ở comparison/helical_vs_spur_summary.csv. Không đánh giá topology bằng fault pair counts thô (15 so với 28); đọc fractions và missingness.']
    write_table(L,['Topology','Channel','Pipeline/feature','Median speed normalized range (valid trajectories)','Median load relative difference (valid comparisons)'],[
        (r['topology'],r['channel'],r['pipeline'].upper()+'/'+r['feature'],
         (f"{r['median_speed_range']:.1%} ({r['speed_trajectory_n']})" if r['median_speed_range'] is not None else 'NE'),
         (f"{r['median_load_difference']:.1%} ({r['load_comparison_n']})" if r['median_load_difference'] is not None else 'NE'))
        for r in compare if r['pipeline'] in ['raw','vmd']])
    L+=['','Association pattern giữ được không có nghĩa feature operating-condition invariant. Raw feature distributions và QC khác nhau có thể phụ thuộc topology/fault set/transfer path; không cô lập causal topology effect. Các normalized ranges/load differences và coverage phía trên đánh giá độ ổn định theo điều kiện, không dùng raw magnitudes giữa gearbox types làm same-fault evidence.']
    transfer={}
    evaluable_patterns={}
    for ch in CH:
        for f in F:
            hh=next(r for r in compare if (r['topology'],r['channel'],r['pipeline'],r['feature'])==('Helical',ch,'raw',f))
            ss=next(r for r in compare if (r['topology'],r['channel'],r['pipeline'],r['feature'])==('Spur',ch,'raw',f))
            desired='OPERATING main effect >= fault' if f=='alpha0' else 'FAULT > each operating main effect'
            kept=hh['association_pattern']==desired and ss['association_pattern']==desired
            transfer[ch,f]=kept
            evaluable_patterns[ch,f]=hh['association_pattern']!='NOT EVALUABLE' and ss['association_pattern']!='NOT EVALUABLE'
            L+=['',f"{ch}/{f}: "+('giữ được qualitative association pattern Helical trên Spur.' if kept else 'không giữ nguyên qualitative association pattern Helical trên Spur, hoặc chưa đủ dữ liệu valid để đánh giá.')]
    for ch in CH:
        hpaired=[r for r in compare if r['topology']=='Helical' and r['channel']==ch and r['pipeline']=='vmd']
        hpreserve=all(r['raw_vmd_correlation'] is not None and r['raw_vmd_correlation']>=.95 for r in hpaired)
        L+=['',f"{ch}: Helical VMD_SUM "+('preserves RAW' if hpreserve else 'does not meet preservation rule')+f"; Spur VMD_SUM {vmdverdicts[ch]}. RAW là reference trực tiếp; VMD có consistency trong preservation chỉ khi cả hai topology đáp ứng rule và difference/separation tables phù hợp."]
    L+=['','**RAW hay VMD có kết luận nhất quán hơn?** Không ưu tiên VMD chỉ vì có decomposition. Đọc paired differences và common-condition gains/losses ở cả hai datasets: khi VMD chỉ preserves RAW thì preservation có thể consistent giữa topology, nhưng không tạo bằng chứng fault separation generalize tốt hơn RAW. Nếu Spur main factorial result NE, cả RAW và VMD đều chưa hỗ trợ kết luận toàn bộ fault set; không xếp hạng generalization bằng sparse valid-subset separation.']
    L+=['','## 11. Limitations','',
        '- Hai acquisitions/cell: disjoint ranges và direction consistency chỉ descriptive, không classification accuracy, không generalization estimate của classifier.',
        '- QC missingness có thể phụ thuộc faults/conditions; partial-grid inference không cứu invalid cells. Additive effects không loại trừ interactions; robust p không cứu confounding hoặc sparse coverage.',
        '- Scaling range <1 decade, same sample scales ứng với số vòng quay khác nhau khi speed đổi; QC pass không chứng minh physical multifractality.',
        '- S2/S4/S5/S6/S7/S8 giữ multi-component definitions. Không causal gear/bearing/shaft localization từ pairwise feature differences.',
        '- Fault sets, topology, tooth counts và transfer paths khác nhau; chưa cô lập topology effect. Không map Helical/Spur bằng class index hoặc feature magnitude.',
        '- Channels cùng recording tương quan, chỉ phân tích riêng mỗi channel; independence giữa acquisitions là thiết kế giả định, chưa chứng minh empirically.',
        '- Individual-mode evidence dựa fixed frequency bands và centers, không xác định cùng nguồn cơ học. Repeated same-dataset extras vẫn cần validation độc lập.',
        '- EMD là practical approximation; strict IMF validity và main-feature QC phải đọc riêng.','',
        '## 12. Final conclusion','',
        '**Liệu kết luận MF-DFA trên Helical còn giữ được trên Spur khi áp dụng nguyên phương pháp?**']
    kept=sum(transfer.values())
    evaluated=sum(evaluable_patterns.values())
    adequate_raw={ch:sum(r['evaluable'] for r in main(cells,'raw',ch))/80>=.75 for ch in CH}
    coverage_note=' Cả hai channels có ≥75% complete RAW cells.' if all(adequate_raw.values()) else ' Ít nhất một channel không đạt 75% complete RAW cells theo rule Step 6; coverage không hỗ trợ full-dataset transfer.'
    if evaluated<6:
        answer=f'Chưa thể xác nhận toàn bộ kết luận Helical trên Spur: chỉ {evaluated}/6 feature/channel association patterns evaluable, trong đó {kept} giữ pattern Helical; {6-evaluated} chưa đánh giá được. NOT EVALUABLE không có nghĩa fault effect bằng zero.'+coverage_note
    elif kept==6:
        answer='Qualitative association patterns giữ được trên subset valid ở cả hai channels; mức hoạt động toàn dataset vẫn phải đọc QC và robustness verdict.'+coverage_note
    else:
        answer=f'Chỉ một phần: {kept}/6 feature/channel association patterns giữ được; {6-kept} patterns khác Helical. Không chuyển nguyên toàn bộ kết luận sang Spur.'+coverage_note
    L+=['',answer,'','Coverage failure ở pipeline/range/QC đã khóa không chứng minh MF-DFA nói chung không thể dùng cho Spur. Step 7 không tìm range hoặc tham số khác để cứu coverage, nên kết luận giới hạn đúng phương pháp được chuyển nguyên từ Helical.']
    for ch in CH:L+=['',f'{ch}: VMD_SUM **{vmdverdicts[ch]}**.']
    dd=decisions(cells,full)
    L+=['','Giữ robustness rule Step 6: ≥75% complete-cell coverage; ROBUST nếu ≥2 features có disjoint fraction≥75% và full-grid fault/operating SD ratio≥1; NOT ROBUST nếu ≥2 features có fraction≤25% và ratio<1; còn lại CONDITION-DEPENDENT. Thiếu full grid không dùng partial ratio thay verdict.']
    write_table(L,['Channel','Pipeline','Complete cells/80','Full-grid evaluable features/3','Verdict'],[(r['channel'],r['pipeline'].upper(),f"{r['complete_cells']}/80",r['full_grid_evaluable_features'],r['decision']) for r in dd])
    weakest=[]
    for ch in CH:
        pp=[r for r in main(ps,'raw',ch) if r['evaluable_conditions']]
        pp=sorted(pp,key=lambda r:(r['disjoint_conditions']/r['evaluable_conditions'],r['pair'],r['feature']))[:5]
        weakest.extend((ch,r['pair'],r['feature'],f"{r['disjoint_conditions']}/{r['evaluable_conditions']}") for r in pp)
    L+=['','Các fault-pair/feature cases khó tách nhất ở RAW (chỉ observed evaluable conditions):']
    write_table(L,['Channel','Pair','Feature','Disjoint/evaluable'],weakest)
    L+=['','Kết luận chỉ cho fault configurations trong dataset và conditions valid. Không classifier, không tuning, không localization hoặc equivalent-fault claim giữa topology.','',
        'Reproduce từ repo root với Python env có NumPy/SciPy/Matplotlib: `python -B result/test/step7/scripts/pipeline.py`, `python -B result/test/step7/scripts/tests.py`, `python -B result/test/step7/scripts/finish.py`. Default output là result/test/step7; PHM_OUTPUT_DIR có thể chỉ định một thư mục mới. Source algorithms/configurations Step 6 được giữ nguyên.']
    for ch in CH:
        for stem in ['feature_heatmaps','paired_raw_vmd','fault_pair_separation','qc_coverage','feature_vs_load','vmd_frequency_modes','vmd_frequency_band_results','raw_scaling_qc_audit']:
            L+=['',f'![{stem} {ch}](figures/{stem}_{ch}.png)']
        for m in ['raw','vmd']:L+=['',f'![Speed {m} {ch}](figures/feature_vs_speed_{m}_{ch}.png)']
    L+=['','![Feature distributions](figures/feature_distributions.png)','','![Residual diagnostics](figures/factorial_residual_diagnostics.png)']
    (P.BASE/'REPORT.md').write_text('\n'.join(L)+'\n')
    P.dump(P.BASE/'comparison/generalization_verdict.json',dict(question='Do Helical MF-DFA conclusions transfer unchanged to Spur?',answer=answer,raw_pattern_retained_count=kept,raw_evaluable_patterns=evaluated,expected_patterns=6,raw_complete_cell_coverage_adequate=adequate_raw,pattern_by_channel_feature={ch+'/'+f:v for (ch,f),v in transfer.items()},pattern_evaluable_by_channel_feature={ch+'/'+f:v for (ch,f),v in evaluable_patterns.items()},vmd_verdicts=vmdverdicts,no_topology_class_pooling=True,no_equivalent_physical_fault_claim=True))
