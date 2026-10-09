"""Scientific figures, auditable report and final cleanup for Step 5."""
import pipeline as P
import numpy as np
import json,itertools,hashlib,re,shutil
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from analysis import FEATURES,METHODS,ANALYSIS_RULE,CONFIG

COLORS={'H1':'#2274a5','H2':'#e18d22','H5':'#bf3d56','H6':'#39935c'}
def save(fig,name):
    fig.tight_layout();fig.savefig(P.BASE/'figures'/name,dpi=150);plt.close(fig)
def main_category(method):return 'raw' if method=='raw' else 'oscillatory_sum'

def figures(features,reps,cells,coverage,paired,models):
    primary='output_voltage'
    for method in METHODS:
        category=main_category(method)
        subset=[c for c in cells if c['pipeline']==method and c['channel']==primary and c['category']==category]
        complete=sum(c['evaluable'] for c in subset)
        if method=='emd' and complete<30:continue
        fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True)
        for i,f in enumerate(FEATURES):
            for j,load in enumerate(P.LOADS):
                ax=axes[i,j]
                for state in P.STATES:
                    rows=sorted([c for c in subset if c['state']==state and c['load']==load],key=lambda c:c['speed'])
                    means=[c[f+'_mean'] if c['evaluable'] else np.nan for c in rows]
                    deviations=[c[f+'_std'] if c['evaluable'] else np.nan for c in rows]
                    ax.errorbar(P.SPEEDS,means,yerr=deviations,color=COLORS[state],marker='o',capsize=3,label=state)
                ax.set(title=load+' / '+f,xlabel='Rotational speed (Hz)',ylabel=f);ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.suptitle(method.upper()+'/'+category+' / column 2; mean ± acquisition SD, gaps = NOT EVALUABLE')
        save(fig,'feature_vs_speed_'+method+'.png')
    # Per-feature heatmaps for main representations, both channels explicitly.
    for _,ch,_ in P.CHANNELS:
        fig,axes=plt.subplots(3,3,figsize=(16,10))
        color_limits={}
        for f in FEATURES:
            values=[c[f+'_mean'] for c in cells if c['channel']==ch and c['category']==main_category(c['pipeline']) and c['evaluable']]
            color_limits[f]=(min(values),max(values)) if values else (0,1)
        for i,method in enumerate(METHODS):
            for j,f in enumerate(FEATURES):
                ax=axes[i,j];matrix=np.full((4,10),np.nan)
                for si,state in enumerate(P.STATES):
                    for ci,(speed,load) in enumerate(itertools.product(P.SPEEDS,P.LOADS)):
                        c=next(c for c in cells if (c['pipeline'],c['channel'],c['category'],c['state'],c['speed'],c['load'])==(method,ch,main_category(method),state,speed,load))
                        if c['evaluable']:matrix[si,ci]=c[f+'_mean']
                image=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap='viridis',vmin=color_limits[f][0],vmax=color_limits[f][1])
                ax.set_xticks(range(10),[f'{s}/{l[0]}' for s,l in itertools.product(P.SPEEDS,P.LOADS)],rotation=45)
                ax.set_yticks(range(4),P.STATES);ax.set_title(method.upper()+' '+f);fig.colorbar(image,ax=ax,shrink=.75)
                for si,ci in zip(*np.where(np.isnan(matrix))):ax.text(ci,si,'NE',ha='center',va='center',fontsize=7,color='red')
        fig.suptitle(ch+' / cell means, NE = NOT EVALUABLE')
        save(fig,'feature_heatmaps_'+ch+'.png')
    # Recording-level distributions across speeds (not windows).
    fig,axes=plt.subplots(3,3,figsize=(16,11),sharey='col')
    for i,method in enumerate(METHODS):
        for j,f in enumerate(FEATURES):
            ax=axes[i,j];labels=[];data=[]
            for state,load in itertools.product(P.STATES,P.LOADS):
                values=[r[f] for r in reps if (r['pipeline'],r['channel'],r['category'],r['state'],r['load'])==(method,primary,main_category(method),state,load) and r['analysis_valid']]
                labels.append(state+'/'+load[0]);data.append(values or [np.nan])
            ax.boxplot(data,tick_labels=labels,showfliers=False)
            for k,values in enumerate(data,1):ax.scatter(np.full(len(values),k),values,s=9,alpha=.6)
            ax.tick_params(axis='x',rotation=45);ax.set(title=method.upper()+' '+f,ylabel=f);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Column 2: recording distributions across all speeds; QC-valid only')
    save(fig,'feature_distributions.png')
    for _,ch,_ in P.CHANNELS:
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,f in zip(axes,FEATURES):
            for state in P.STATES:
                for load,marker in [('High','o'),('Low','^')]:
                    subset=[r for r in paired if r['channel']==ch and r['feature']==f and r['state']==state and r['load']==load and r['evaluable']]
                    ax.scatter([r['raw_feature'] for r in subset],[r['vmd_feature'] for r in subset],s=25,color=COLORS[state],marker=marker,label=state+'/'+load[0])
            limits=ax.get_xlim()+ax.get_ylim();lo=min(limits);hi=max(limits)
            ax.plot([lo,hi],[lo,hi],'k--',lw=1);ax.set(title=f,xlabel='Raw feature',ylabel='VMD oscillatory_sum feature');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.suptitle('Paired full-recording features / '+ch)
        save(fig,'paired_raw_vmd_'+ch+'.png')
    # Complete-cell coverage includes every band and all 40 operating/fault cells.
    for _,ch,_ in P.CHANNELS:
        identities=sorted({(r['pipeline'],r['category']) for r in coverage if r['channel']==ch})
        matrix=np.zeros((len(identities),40))
        for ri,(method,category) in enumerate(identities):
            for ci,(state,speed,load) in enumerate(itertools.product(P.STATES,P.SPEEDS,P.LOADS)):
                r=next(r for r in coverage if (r['pipeline'],r['channel'],r['category'],r['state'],r['speed'],r['load'])==(method,ch,category,state,speed,load))
                matrix[ri,ci]=r['valid_fraction']
        fig,ax=plt.subplots(figsize=(18,8));im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu')
        ax.set_yticks(range(len(identities)),[m+'/'+c for m,c in identities]);ax.set_xticks(range(40),[st+':'+str(s)+'/'+l[0] for st,s,l in itertools.product(P.STATES,P.SPEEDS,P.LOADS)],rotation=90,fontsize=6)
        ax.set_title('QC-valid acquisition fraction / '+ch+'; matching eligibility in qc_coverage.csv');fig.colorbar(im,ax=ax,label='valid / 2 expected')
        save(fig,'qc_coverage_'+ch+'.png')
    # Assumption diagnostics for main additive representations, primary channel.
    from factorial_model import design,ols
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for j,method in enumerate(METHODS):
        rows=[r for r in reps if (r['pipeline'],r['channel'],r['category'])==(method,primary,main_category(method)) and r['analysis_valid']]
        if len(rows)<30:continue
        X,_=design(rows)
        y=np.array([r['alpha0'] for r in rows]);inv,b,e,h=ols(X,y)
        axes[0,j].scatter(X@b,e,s=15);axes[0,j].axhline(0,color='k',lw=.7);axes[0,j].set(title=method.upper()+' alpha0 residuals',xlabel='Fitted',ylabel='Residual')
        stats.probplot(e,plot=axes[1,j]);axes[1,j].set_title(method.upper()+' alpha0 Q-Q')
    save(fig,'factorial_residual_diagnostics.png')

def report(features,reps,cells,pairs,speed,load,effects,paired_summary,models,coverage):
    primary='output_voltage';inventory=json.loads((P.BASE/'diagnostics/inventory_summary.json').read_text(encoding='utf-8'))
    def main(rows,method):return [r for r in rows if r['pipeline']==method and r['channel']==primary and r.get('category')==main_category(method)]
    lines=['# STEP 5 — Operating-condition robustness','',
        f"**{inventory['found_unique_recordings']}/{inventory['expected_recordings']} Helical recordings**; missing={len(inventory['missing_recordings'])}, duplicates={len(inventory['duplicated_recordings'])}, unexpected={len(inventory['unexpected_files'])}. Hai acquisitions/cell, full length {inventory['min_samples']}–{inventory['max_samples']} samples (~3,685–4,000 s), 80 independent recording units / 160 channel signals. Không có windows dùng làm samples.",
        'Cột 2 output voltage chính; cột 1 input voltage đối chiếu. H1 Healthy; H2 24T chipped gear; H5 24T broken gear + bearing inner-race fault; H6 bent input shaft. **H2–H5 không cô lập bearing contribution.**', '',
        '## Configuration và separation validity', '',
        'Snapshot config/locked_step4_configuration.json tham chiếu SHA256 các config/source Step 4 và VidData, không nhân bản implementation. q=-5…5 bước 0,5; DFA order=2; scales gốc 40 điểm; fit 64–512 samples (actual 64–468, 17 points). QC nguyên Step 4: min R²≥0,90; ≥80% q R²≥0,95; ≥80% local-slope CV≤0,5; min h(q)>0,05; ≥8 points. Không grid search, đổi range hay đổi QC.',
        'VMD K=7, alpha=500, tau=0, DC=False, init=1, tol=1e-7, max_iter=2000. EMD practical approximation: sáu modes, max_siftings=200, energy SD≤0,2, envelope RMS ratio≤0,05, relative extrema mismatch≤0,005; giữ strict IMF audit riêng. Không coi EMD là benchmark IMF nghiêm ngặt.',
        'Invalid arrays và numerical features lưu audit; valid_features.csv và mọi phân tích chỉ dùng analysis_valid = scaling_valid AND decomposition_valid. Shape-warning của Step 4 được giữ, không đổi thành QC mới; scaling pass không chứng minh multifractality vật lý.', '',
        '| Pipeline | QC pass/all components | Cột 2 pass | Main raw/sum complete cells, cột 2 |','|---|---:|---:|---:|']
    for method in METHODS:
        rows=[r for r in features if r['pipeline']==method];ch=[r for r in rows if r['channel']==primary];cc=main(cells,method)
        lines.append(f"| {method.upper()} | {sum(r['scaling_valid'] for r in rows)}/{len(rows)} | {sum(r['scaling_valid'] for r in ch)}/{len(ch)} | {sum(r['evaluable'] for r in cc)}/40 |")
    lines+=['', 'Solver convergence/practical EMD convergence tách khỏi MF-DFA scaling QC; tau=0 VMD không buộc sum(modes)=raw. Bảng reconstruction dưới dùng ||x−sum(modes)||/||x|| trên tín hiệu đã bỏ mean. Mode+residue identity được kiểm tra riêng; identity theo construction không chứng minh reconstruction riêng sum tốt.', '',
        '| Pipeline, cột 2 | Decomposition pass / recordings | Reconstruction median | max | Strict IMF pass |', '|---|---:|---:|---:|---:|']
    decomposition=[P.typed(r) for r in P.read_table(P.BASE/'diagnostics/decomposition_summary.csv')]
    for method in ['vmd','emd']:
        dd=[r for r in decomposition if r['pipeline']==method and r['channel']==primary]
        strict=f"{sum(bool(r['strict_emd_imf_valid']) for r in dd)}/{len(dd)}" if method=='emd' else 'N/A'
        lines.append(f"| {method.upper()} | {sum(r['decomposition_valid'] for r in dd)}/{len(dd)} | {np.median([r['reconstruction_error'] for r in dd]):.2%} | {max(r['reconstruction_error'] for r in dd):.2%} | {strict} |")
    lines+=['', '## 1. Raw ổn định theo speed/load hay không?', '',
        'Trajectory là mean ± sample SD của hai acquisitions ở cùng cell; normalized speed range=(max−min)/abs(mean qua 5 speeds); load effect=2|High−Low|/(|High|+|Low|). Thiếu một cell/không match thì toàn trajectory/load comparison ghi NOT EVALUABLE.', '',
        '| Pipeline/feature, cột 2 raw/sum | Median speed normalized range | Median load relative difference | Fault / operating variation ratio |','|---|---:|---:|---:|']
    for method,f in itertools.product(METHODS,FEATURES):
        ss=[r for r in main(speed,method) if r['feature']==f and r['evaluable']];ll=[r for r in main(load,method) if r['feature']==f and r['evaluable']]
        ee=next(r for r in main(effects,method) if r['feature']==f)
        sval=f"{np.median([r['normalized_range'] for r in ss]):.1%} (N={len(ss)}/8)" if ss else 'NE (0/8)'
        lval=f"{np.median([r['relative_difference'] for r in ll]):.1%} (N={len(ll)}/20)" if ll else 'NE (0/20)'
        ratio=f"{ee['fault_to_operating_ratio']:.3f}" if ee['evaluable'] else 'NE'
        lines.append(f'| {method.upper()}/{f} | {sval} | {lval} | {ratio} |')
    lines+=['', 'Fault/operating ratio = median absolute between-fault mean difference tại matched speed/load / median within-fault SD của 10 operating-cell means. Đây là descriptive scale ratio, không phải causal variance decomposition; >1 nghĩa fault gap lớn hơn typical operating SD theo định nghĩa này.', '',
        'Full-grid ratio NE nghĩa chưa có đủ cả 10 conditions valid cho cả bốn states, không có nghĩa fault effect bằng zero. Bổ sung bảng dưới trên common observed operating grid có đủ bốn trạng thái; inclusion chỉ theo validity/matching, không theo separation. Bảng partial-grid này không thay rule/verdict full-grid đã khai báo.', '',
        '| Pipeline/feature, cột 2 | Common operating cells / 10 | Partial-grid fault / operating SD ratio |', '|---|---:|---:|']
    grid_effects=[P.typed(r) for r in P.read_table(P.BASE/'summaries/matched_operating_grid_effects.csv')]
    for method,f in itertools.product(METHODS,FEATURES):
        r=next(r for r in grid_effects if (r['pipeline'],r['channel'],r['category'],r['feature'])==(method,primary,main_category(method),f))
        ratio=f"{r['fault_to_operating_ratio']:.3f}" if r['evaluable'] else 'NE'
        lines.append(f"| {method.upper()}/{f} | {r['common_operating_cells']}/10 | {ratio} |")
    lines+=['', 'matched_operating_grid_effects.csv ghi rõ conditions, source recordings và scope; không suy rộng những cells bị QC loại và không interpolate chúng.', '',
        '![Raw speed](figures/feature_vs_speed_raw.png)', '',
        '![VMD speed](figures/feature_vs_speed_vmd.png)', '',
        '## 2. Within-condition separation và generalization Step 4', '',
        'Sáu fault pairs × 10 operating conditions × ba features; disjoint acquisition ranges là mô tả hai repeats, không phải accuracy hoặc significance. Direction retention so với 50 Hz/High và mọi NOT EVALUABLE được giữ ở within_condition_separation.csv.', '',
        '| Pipeline/feature, cột 2 raw/sum | Evaluable pairs / 60 | Disjoint fraction | Step 4 disjoint retained ở conditions khác |','|---|---:|---:|---:|']
    for method,f in itertools.product(METHODS,FEATURES):
        valid=[r for r in main(pairs,method) if r['feature']==f and r['evaluable']]
        general=[r for r in valid if not (r['speed']==50 and r['load']=='High') and r.get('step4_disjoint',False)]
        rate=f"{np.mean([r['disjoint_repeat_ranges'] for r in valid]):.1%}" if valid else 'NE'
        retained=f"{np.mean([r.get('step4_disjoint_retained',False) for r in general]):.1%} ({len(general)} comparisons)" if general else 'NE'
        lines.append(f'| {method.upper()}/{f} | {len(valid)}/60 | {rate} | {retained} |')
    lines+=['', '## 3. Raw vs VMD: preserve, improve hay mất thông tin?', '',
        '| Feature, cột 2 | Paired N | Correlation | Median relative Raw–VMD difference |','|---|---:|---:|---:|']
    for r in paired_summary:
        if r['channel']==primary:lines.append(f"| {r['feature']} | {r['paired_recordings']} | {r['pearson_r']:.4f} | {r['median_relative_difference']:.1%} |")
    lines+=['', '![Paired](figures/paired_raw_vmd_output_voltage.png)', '',
        'Correlation cao riêng nó không chứng minh improvement. So sánh giảm sensitivity/tăng separation phải dùng cùng cells valid ở cả Raw và VMD; bảng dưới giữ intersection trước khi tính descriptive rates.', '',
        '| Feature | Common evaluable pair comparisons | Raw disjoint | VMD disjoint | Median speed-range ratio VMD/Raw | Median load-effect ratio VMD/Raw |','|---|---:|---:|---:|---:|---:|']
    paired_effects=[]
    for f in FEATURES:
        pr={ (r['speed'],r['load'],r['pair']):r for r in main(pairs,'raw') if r['feature']==f and r['evaluable']}
        pv={ (r['speed'],r['load'],r['pair']):r for r in main(pairs,'vmd') if r['feature']==f and r['evaluable']}
        keys=sorted(pr.keys()&pv.keys())
        sr={(r['state'],r['load']):r for r in main(speed,'raw') if r['feature']==f and r['evaluable']};sv={(r['state'],r['load']):r for r in main(speed,'vmd') if r['feature']==f and r['evaluable']}
        lr={(r['state'],r['speed']):r for r in main(load,'raw') if r['feature']==f and r['evaluable']};lv={(r['state'],r['speed']):r for r in main(load,'vmd') if r['feature']==f and r['evaluable']}
        speedratio=[sv[k]['normalized_range']/sr[k]['normalized_range'] for k in sr.keys()&sv.keys() if sr[k]['normalized_range']>1e-8]
        loadratio=[lv[k]['relative_difference']/lr[k]['relative_difference'] for k in lr.keys()&lv.keys() if lr[k]['relative_difference']>1e-8]
        rr=float(np.mean([pr[k]['disjoint_repeat_ranges'] for k in keys])) if keys else np.nan
        vr=float(np.mean([pv[k]['disjoint_repeat_ranges'] for k in keys])) if keys else np.nan
        ss=float(np.median(speedratio)) if speedratio else np.nan;ll=float(np.median(loadratio)) if loadratio else np.nan
        lines.append(f'| {f} | {len(keys)} | {rr:.1%} | {vr:.1%} | {ss:.3f} | {ll:.3f} |')
        paired_effects.append(dict(feature=f,common_pairs=len(keys),raw_disjoint_fraction=rr,vmd_disjoint_fraction=vr,median_speed_ratio=ss,median_load_ratio=ll))
    P.table(P.BASE/'summaries/raw_vmd_matched_effects.csv',paired_effects)
    preservation=all(r['pearson_r']>=.95 for r in paired_summary if r['channel']==primary)
    improvement=all(r['vmd_disjoint_fraction']>r['raw_disjoint_fraction'] and r['median_speed_ratio']<1 and r['median_load_ratio']<1 for r in paired_effects)
    conclusion='A: consistent descriptive improvement on all three features' if improvement else ('B: mostly preserves Raw representation, with feature-specific changes; no consistent A' if preservation else 'C/mixed: representation changes; no consistent improvement over Raw')
    lines+=['', '**Raw–VMD interpretation: '+conclusion+'.** Đây là mô tả dữ liệu hiện có; không tối ưu pipeline theo kết quả.', '',
        '## 4. Individual frequency bands và EMD coverage', '',
        'Frequency bands cố định Step 4; chọn mode energy lớn nhất trong band trước QC; không dùng nhãn để match. Matching requires all two/four repeat centers ratio≤1,5; across-speed/load summaries cần cùng compatibility. Same rank không được coi là same physical mode. Center matching và coverage đều có audit; thiếu valid/matched modes ghi NOT EVALUABLE.', '',
        '| Pipeline/band, cột 2 | Evaluable cells / 40 | Evaluable feature–pair rows / 180 | Disjoint rows |','|---|---:|---:|---:|']
    mode_comparisons=[]
    for method in ['vmd','emd']:
        for lo,hi in CONFIG['bands_hz']:
            cat=f'band_{lo}_{hi}_Hz';cc=[r for r in cells if (r['pipeline'],r['channel'],r['category'])==(method,primary,cat)]
            pp=[r for r in pairs if (r['pipeline'],r['channel'],r['category'])==(method,primary,cat) and r['evaluable']]
            lines.append(f'| {method.upper()}/{lo}–{hi} Hz | {sum(c["evaluable"] for c in cc)}/40 | {len(pp)}/180 | {sum(r["disjoint_repeat_ranges"] for r in pp)} |')
            for r in pp:
                raw=next(x for x in pairs if (x['pipeline'],x['channel'],x['category'],x['speed'],x['load'],x['pair'],x['feature'])==('raw',primary,'raw',r['speed'],r['load'],r['pair'],r['feature']))
                mode_comparisons.append(dict(pipeline=method,category=cat,speed=r['speed'],load=r['load'],pair=r['pair'],feature=r['feature'],
                    mode_disjoint=r['disjoint_repeat_ranges'],raw_evaluable=raw['evaluable'],raw_disjoint=raw.get('disjoint_repeat_ranges',''),
                    extra_descriptive_separation=bool(raw['evaluable'] and r['disjoint_repeat_ranges'] and not raw['disjoint_repeat_ranges']),source_recordings=r['source_recordings']))
    P.table(P.BASE/'summaries/individual_bands_vs_raw.csv',mode_comparisons)
    extras=[r for r in mode_comparisons if r['pipeline']=='vmd' and r['extra_descriptive_separation']]
    lines +=['',f"Individual VMD bands có {len(extras)} matched descriptive feature–pair cases tách khoảng khi Raw chưa tách; xem individual_bands_vs_raw.csv. Đây là sparse band-specific evidence, không chứng minh thông tin mới thống kê hoặc improvement tổng quát. EMD individual-mode coverage phải đánh giá theo bảng; approximate EMD và VMD không được xếp hạng bằng tổng số mode pass khác nhau.", '',
        '## 5. Factorial exploration và inference', '',
        'Categorical fault + speed + load dùng sum-to-zero orthogonal contrasts. Full fault*speed*load chỉ fit nếu đủ 40 cells, full rank và ≥20 residual degrees of freedom. Additive partial η² = incremental SS/(incremental SS+residual SS); các partial η² không cộng thành tổng variance. Interactions kiểm tra conditional terms của full model. Không diễn giải additive main effect như causal effect khi có interactions.',
        'OLS/HC3 covariance và null-imposed Rademacher wild bootstrap 1.999 draws, seed 6509, HC2-scaled reduced-model residuals, studentized HC3 Wald. HC3/wild p là exploratory; classical p chỉ reference. BH adjustment tính riêng mỗi channel trên toàn family models/features/representations/terms. Không có package cài thêm; NumPy/SciPy implementation và behavioral tests được lưu.',
        'Shapiro residuals, heteroskedasticity LM và Q-Q/fitted-residual plot giữ trong diagnostics. Assumptions tests với 2 repeats/cell có power hạn chế; independence là thiết kế người dùng quy định, chưa kiểm chứng empirically. Robust inference giảm lệ thuộc normal/homoskedastic assumptions nhưng không cứu confounding, sparse coverage hay pseudo-replication.', '',
        '| Pipeline/feature, cột 2 | Additive partial η² fault | speed | load | Full interaction availability |','|---|---:|---:|---:|---|']
    for method,f in itertools.product(METHODS,FEATURES):
        subset=[r for r in models if (r['pipeline'],r['channel'],r['category'],r['feature'])==(method,primary,main_category(method),f)]
        values=[]
        for term in ['fault','speed','load']:
            r=next((r for r in subset if r.get('model')=='fault+speed+load' and r.get('term')==term),None)
            values.append(f"{r['partial_eta_squared']:.3f}" if r else 'NE')
        full=any(r.get('model')=='fault*speed*load' and r['status']=='exploratory' for r in subset)
        lines.append('| '+method.upper()+'/'+f+' | '+' | '.join(values)+' | '+('evaluable' if full else 'NOT EVALUABLE')+' |')
    lines+=['','P-values, robust statistics, effect sizes, source recordings và reasons trong factorial_effects.csv. Descriptive repeat-range separation tách rõ khỏi inferential p-values.', '',
        '| Pipeline/feature, cột 2 | Wild p BH fault | speed | load |', '|---|---:|---:|---:|']
    for method,f in itertools.product(METHODS,FEATURES):
        values=[]
        for term in ['fault','speed','load']:
            r=next((r for r in models if (r['pipeline'],r['channel'],r['category'],r['feature'],r.get('model'),r.get('term'))==(method,primary,main_category(method),f,'fault+speed+load',term)),None)
            values.append(f"{r['wild_bootstrap_p_BH']:.4f}" if r else 'NE')
        lines.append('| '+method.upper()+'/'+f+' | '+' | '.join(values)+' |')
    lines+=['', 'Các p trên là exploratory BH-adjusted, không chứng minh classification performance hay causal effects. Interaction effect sizes và wild/HC3/classical p của full model có ở cùng CSV; nếu full model NOT EVALUABLE thì không thay bằng một model đã chọn theo separation.', '',
        '![Diagnostics](figures/factorial_residual_diagnostics.png)', '',
        '## 6. Những kết luận chưa được phép đưa ra', '',
        'Không classification accuracy, không causal bearing-specific effect H2–H5, không khẳng định healthy/fault separability ở mọi gearbox/dataset, không chứng minh physical multifractality từ một local scaling range <1 decade, không coi hai acquisitions/class/cell là statistical classification evidence. Fixed-sample scaling cũng thay đổi số vòng quay cơ học tương ứng khi speed đổi; đây là một phần robustness challenge và không được sửa range để cứu kết quả.', '',
        '## Reproducibility và audit', '',
        'recording_inventory.csv chứa original SHA256, parsing state/speed/load/repeat, sample count/duration/channels. locked_step4_configuration.json chứa snapshot và dependency hashes. features/mfdfa_features.csv và diagnostics/mfdfa_quality_control.csv gồm invalid; features/valid_features.csv chỉ valid. arrays/: complete Fq/h/tau/alpha/f/QC, hoặc reference Step 4 cho baseline. Recording JSON chứa solver/sifting diagnostics, reconstruction identity và scalar mode metrics; không giữ full decomposition arrays có thể regenerate.',
        'Mọi descriptive/factorial summary có source_recordings để trace. 50 Hz/High dùng lại arrays đã validated của Step 4; phụ thuộc này giữ nguyên. output_cleanup.csv audit caches xóa sau validation; output_manifest.csv liệt kê/hash outputs.', '',
        '```powershell','python -B result\\test\\step5\\scripts\\pipeline.py','python -B result\\test\\step5\\scripts\\tests.py','python -B result\\test\\step5\\scripts\\analysis.py','```','',
        'Method references: [SciPy Shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html), [HC3/ANOVA guidance](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html), [Davidson–Flachaire wild bootstrap](https://www.econ.queensu.ca/research/working-papers/1000). Không thay đổi configuration từ những analysis results.', '',
        '## Quyết định cuối Step 5', '',
        'Decision rules mô tả được ghi trong config/analysis_rules.json trước khi tổng hợp dữ liệu; không phải universal scientific cutoff. Đánh giá primary raw/sum representation; individual bands có coverage riêng.', '',
        '| Pipeline | Decision | Evidence summary |','|---|---|---|']
    decision_start=lines.index('## Quyết định cuối Step 5')
    decisions=[]
    for method in METHODS:
        cc=main(cells,method);validcount=sum(c['evaluable'] for c in cc);ee=main(effects,method)
        robust=sum(r.get('evaluable',False) and r.get('disjoint_pair_fraction',0)>=.75 and r.get('fault_to_operating_ratio',0)>=1 for r in ee)
        poor=sum(r.get('evaluable',False) and r.get('disjoint_pair_fraction',1)<=.25 and r.get('fault_to_operating_ratio',1)<1 for r in ee)
        decision='NOT EVALUABLE' if validcount<30 else ('ROBUST' if robust>=2 else ('NOT ROBUST' if poor>=2 else 'CONDITION-DEPENDENT'))
        note=f'{validcount}/40 complete cells; full-grid effects evaluable for {sum(r["evaluable"] for r in ee)}/3 features; {robust}/3 meet ROBUST rule, {poor}/3 meet NOT ROBUST rule; strict verdict limited by missing/invalid operating cells'
        decisions.append(dict(pipeline=method,channel=primary,representation=main_category(method),decision=decision,evidence=note))
        lines.append('| '+method.upper()+' | **'+decision+'** | '+note+' |')
    P.table(P.BASE/'summaries/robustness_decisions.csv',decisions)
    decision_block=lines[decision_start:];lines=lines[:decision_start]
    lines+=['', '## Trả lời trực tiếp về generalization', '',
        'Trong dữ liệu này, Raw Δα/Δh giữ fault association mạnh hơn main speed/load effects ở additive model; α0 chịu ảnh hưởng speed/load lớn hơn fault (xem partial η²). Tuy vậy tuyệt đối các feature thay đổi theo conditions, và direction/separation của baseline chỉ được retained một phần. Vì thế fault information vẫn hiện diện, nhưng không thể coi feature là operating-condition invariant.',
        'Kết quả 50 Hz/High không tự động áp dụng cho các operating conditions khác. Retention table trên đo chính các separation/direction của baseline trên conditions còn lại; unavailable comparisons được báo explicit, không được coi là retained.',
        'Fault-state variation phải đọc đồng thời với normalized speed/load changes và factorial partial effects. Sự tồn tại fault signal trong một số matched cells không đồng nghĩa invariant feature qua speed/load. Chỉ decision ROBUST mới đáp ứng descriptive robustness rule đã khai báo; CONDITION-DEPENDENT nghĩa kết quả fault mang tính điều kiện; NOT ROBUST nghĩa fault/operating ratio và separation thấp theo rule.',
        'Raw–VMD paired summary phía trên trả lời riêng preservation, sensitivity và separation trên cùng recording/cells. Individual-band extra cases chỉ là candidates cho phase độc lập sau này; không dùng chúng để tối ưu lại Step 5 hoặc diễn giải riêng bearing contribution.']
    lines+=['', '## Figures bổ sung', '',
        '[Feature distributions](figures/feature_distributions.png) · [Primary heatmaps](figures/feature_heatmaps_output_voltage.png) · [Primary QC coverage](figures/qc_coverage_output_voltage.png) · [Secondary heatmaps](figures/feature_heatmaps_input_voltage.png) · [Secondary QC coverage](figures/qc_coverage_input_voltage.png)']
    if (P.BASE/'figures/feature_vs_speed_emd.png').exists():lines+=['', '[EMD feature trajectories](figures/feature_vs_speed_emd.png)']
    lines+=['']+decision_block
    (P.BASE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def validate_and_cleanup(features,reps,cells,pairs,sources):
    assert len({s['recording'] for s in sources})==len(sources)
    assert len({s['sha256'] for s in sources})==len(sources),'Duplicate source contents'
    assert {s['state'] for s in sources}==set(P.STATES)
    for source in sources:
        assert P.parse_name(P.resolve(source['relative_source']).name)==(source['state'],source['speed'],source['load'],source['repeat'])
        assert P.digest(P.resolve(source['relative_source']))==source['sha256']
    P.lock_config() # checks immutable Step 4 dependencies and exact QC snapshot
    runtime=json.loads((P.BASE/'config/runtime_provenance.json').read_text(encoding='utf-8'))
    for reference in runtime['dependencies']:assert P.digest(P.resolve(reference['path']))==reference['sha256']
    reused=P.read_table(P.BASE/'config/reused_step4_array_manifest.csv')
    for reference in reused:assert P.digest(P.resolve(reference['path']))==reference['sha256']
    keys=[(r['recording'],r['channel'],r['pipeline'],r['component']) for r in features];assert len(keys)==len(set(keys))
    by_recording={s['recording']:s for s in sources}
    for r in features:
        source=by_recording[r['recording']]
        assert r['samples']==source['samples'] and r['source_sha256']==source['sha256']
        assert all(r[k]==source[k] for k in ['state','speed','load','repeat'])
        assert r['analysis_valid']==bool(r['scaling_valid'] and r['decomposition_valid'])
        assert r['config_sha256'] in P.aliases(CONFIG)
        if r['status']=='ok':
            with np.load(P.resolve(r['array_path'])) as a:
                computed,metrics=P.fit_scaling(a['scales'],a['Fq'],a['q'],CONFIG['fit_interval'])
                assert metrics['scaling_valid']==r['scaling_valid']
                np.testing.assert_array_equal(a['q'],CONFIG['q']);np.testing.assert_array_equal(a['scales'],CONFIG['scales'])
                for key in ['hq','tau','alpha','f_alpha']:np.testing.assert_allclose(a[key],computed[key],rtol=1e-12,atol=1e-12)
                assert np.isclose(metrics['delta_alpha'],r['delta_alpha'])
    expected={s['recording'] for s in sources}
    for path in (P.BASE/'summaries').glob('*.csv'):
        for row in P.read_table(path):
            if 'source_recordings' in row:
                assert set(filter(None,row['source_recordings'].split(';')))<=expected,str(path)
    valid=[P.typed(r) for r in P.read_table(P.BASE/'features/valid_features.csv')]
    assert all(r['analysis_valid'] for r in valid)
    # Every evaluable pair comes from exactly four valid acquisition units.
    for row in pairs:
        if row['evaluable']:assert len(set(row['source_recordings'].split(';')))==4
    tests=json.loads((P.BASE/'diagnostics/tests.json').read_text(encoding='utf-8'));assert tests['passed']
    decomposition=P.read_table(P.BASE/'diagnostics/decomposition_summary.csv')
    assert len(decomposition)==4*len(sources)
    assert all((P.resolve(r['diagnostics_path'])).is_file() for r in decomposition)
    summary=dict(all_passed=True,recordings=len(sources),channel_signals=2*len(sources),analyses=len(features),
        valid_analyses=sum(r['analysis_valid'] for r in features),raw_hashes_unchanged=True,
        step4_dependency_hashes_unchanged=True,QC_thresholds_unchanged=True,configuration_unchanged=True,
        runtime_dependencies_unchanged=True,reused_step4_array_hashes_checked=len(reused),
        exact_full_length_preserved=True,states=P.STATES,speeds=P.SPEEDS,loads=P.LOADS,acquisitions=[1,2],
        no_duplicate_recordings_or_components=True,invalid_excluded_from_valid_analysis=True,
        mode_matching='band/energy/frequency only; no class label used for physical assignment',
        no_parameter_selection_or_windows=True,all_summary_recordings_traceable=True,tests=tests,
        config_sha256=CONFIG['config_sha256'])
    P.dump(P.BASE/'diagnostics/validation.json',summary)
    # Verify report local links before removing rebuildable plot caches.
    links=re.findall(r'\]\(([^)]+)\)',(P.BASE/'REPORT.md').read_text(encoding='utf-8'))
    for link in links:
        if not link.startswith('http'):assert (P.BASE/link).is_file(),link
    work=(P.BASE/'_work').resolve();assert work.parent==P.BASE.resolve() and work.name=='_work'
    removed=[dict(path=p.relative_to(P.BASE).as_posix(),bytes=p.stat().st_size,reason='regenerable plotting/test cache') for p in work.rglob('*') if p.is_file()]
    # Aggregated solver JSON duplicates permanent per-record diagnostics; the
    # decomposition summary references those originals and Step 4 baselines.
    redundant=P.BASE/'diagnostics/decomposition_diagnostics.json'
    if redundant.exists():
        assert redundant.resolve().parent==(P.BASE/'diagnostics').resolve()
        removed.append(dict(path=redundant.relative_to(P.BASE).as_posix(),bytes=redundant.stat().st_size,reason='duplicate aggregate; per-record diagnostics and traced summary retained'))
        redundant.unlink()
    if work.exists():shutil.rmtree(work)
    old_cleanup=P.BASE/'diagnostics/output_cleanup.csv'
    if old_cleanup.exists():
        combined={r['path']:r for r in P.read_table(old_cleanup)}
        combined.update({r['path']:r for r in removed});removed=list(combined.values())
    P.table(P.BASE/'diagnostics/output_cleanup.csv',removed)
    manifest=[dict(path=p.relative_to(P.BASE).as_posix(),bytes=p.stat().st_size,sha256=P.digest(p)) for p in sorted(P.BASE.rglob('*')) if p.is_file() and p.name!='output_manifest.csv']
    P.table(P.BASE/'diagnostics/output_manifest.csv',manifest)
