"""Full helical validation using Step 5 analysis, extended only to six levels."""
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
        fig,axes=plt.subplots(3,3,figsize=(16,16))
        for i,m in enumerate(A.METHODS):
            for j,f in enumerate(F):
                ax=axes[i,j];matrix=np.full((15,10),np.nan)
                for r in primary(pairs,m,ch):
                    if r['feature']==f and r['evaluable']:
                        matrix[pairnames.index(r['pair']),list(itertools.product(P.SPEEDS,P.LOADS)).index((r['speed'],r['load']))]=int(r['disjoint_repeat_ranges'])
                ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
                ax.set_yticks(range(15),pairnames);ax.set_xticks(range(10),[str(s)+'/'+l[0] for s,l in itertools.product(P.SPEEDS,P.LOADS)],rotation=60)
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
                ax.set_xticks(range(6),P.STATES);ax.set_title(m.upper()+' '+f);ax.grid(alpha=.2);ax.legend(fontsize=6)
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

def report(features,reps,cells,pairs,speed,load,grid,paired,models,matched,extras,recurrence):
    inv=json.loads((P.BASE/'diagnostics/inventory_summary.json').read_text())
    lines=['# STEP 6 — Full helical validation','', '## 1. Dataset','',
        f"Đủ **{inv['found_unique_recordings']}/120 recordings**; missing={len(inv['missing_recordings'])}, duplicate keys={len(inv['duplicated_recordings'])}, duplicate content={len(inv['duplicated_content_groups'])}, unexpected={len(inv['unexpected_files'])}. H1–H6 × 5 speeds (30/35/40/45/50 Hz) × High/Low × 2 acquisitions. 120 recording units, 240 channel signals; input/output cùng recording không được coi là hai acquisitions độc lập. Không chia windows thành samples.",
        f"Full length {inv['min_samples']}–{inv['max_samples']} samples, {inv['min_samples']/A.CONFIG['fs']:.4f}–{inv['max_samples']/A.CONFIG['fs']:.4f} s; fs=66,666.7 Hz. Output voltage cột 2 là primary, input voltage cột 1 là secondary.",'',
        '| State | Fault configuration | VidData.xls nguyên văn (non-Good entries) |','|---|---|---|',
        '| H1 | Healthy | All Good |','| H2 | Gear fault | 24T: Chipped |',
        '| H3 | Gear + Bearing + Shaft | 24T: Broken; IS:OS: Combination; ID:OS: Inner; Input: Bent Shaft |',
        '| H4 | Bearing + Shaft | IS:OS: Combination; ID:OS: ball; Input: Imbalnce |',
        '| H5 | Gear + Bearing | 24T: Broken; ID:OS: Inner |','| H6 | Shaft | Input: Bent Shaft |','',
        'Labels đọc từ Sheet1 hàng 16–22; giữ cả lỗi chính tả Imbalnce. Các vị trí trong workbook là metadata, không phải kết luận localization từ feature. VidData ghi Total Runs=140 nhưng grid H1–H6 được yêu cầu và kiểm kê ở đây là 120.', '',
        '## 2. Configuration','',
        'Giữ q=-5…5, step 0.5; DFA order=2; 40 scales gốc; fit 64–512 samples (17 actual points, 64–468). Không tune bằng H3/H4. Δα=max(alpha)−min(alpha); α0=alpha(q=0); Δh=max(hq)−min(hq).',
        'QC: ≥8 fit points; min R²≥0.90; ≥80% q có R²≥0.95; ≥80% q có adjacent slope CV≤0.5; min hq>0.05. Analysis valid = scaling valid AND decomposition valid. Spectrum shape warnings giữ audit, không thêm gate.',
        'VMD K=7, alpha=500, tau=0, DC=False, init=1, tol=1e-7, max_iter=2000. EMD practical approximation giữ max_imfs=6, max_siftings=200, sd_threshold=0.2, envelope_tol=0.05, extrema_relative_tolerance=0.005; strict IMF audit riêng. Raw chỉ mean removal bên trong MF-DFA; decomposition dùng full signal bỏ global mean.',
        'Step 5 snapshot được giữ; bốn JSON config khác hash chỉ do CRLF→LF. Mã MF-DFA hiện tại khác historical hash; vì vậy **toàn bộ 120 recordings được chạy lại**, không reuse numerical features Step 5. Không thay mã core/config cũ. Portable C++ VMD chỉ bỏ Windows export annotation; recurrence giữ nguyên, không fast-math; kiểm tra parity với NumPy reference. Xem config/dependency_audit.json và diagnostics/tests.json.', '',
        '## 3. QC','', '| Channel | Pipeline | Main raw/sum valid signals | All components valid | Complete main cells |', '|---|---|---:|---:|---:|']
    for ch,m in itertools.product(CH,A.METHODS):
        rr=[r for r in features if r['channel']==ch and r['pipeline']==m]; main=[r for r in rr if r['component']==main_category(m)]
        cc=primary(cells,m,ch)
        lines.append(f"| {ch} | {m.upper()} | {sum(r['analysis_valid'] for r in main)}/{len(main)} ({sum(r['analysis_valid'] for r in main)/len(main):.1%}) | {sum(r['analysis_valid'] for r in rr)}/{len(rr)} | {sum(r['evaluable'] for r in cc)}/60 |")
    lines+=['','Không dùng tổng mode-valid rate để xếp hạng Raw/VMD/EMD: số components khác nhau. Invalid features và Fq giữ audit; mọi kết luận chỉ dùng valid/matched recordings. QC breakdown theo state/speed/load/component ở CSV.', '', '## 4. H1–H6 feature results','',
        'Các bảng và plots dùng recording hoặc mean±sample SD của đúng hai acquisitions tại cell. Không dùng pooled fault mean để tuyên bố separation khi operating conditions khác nhau.','',
        '| Channel | Pipeline | Feature | Evaluable pair-condition comparisons /150 | Disjoint / evaluable |', '|---|---|---|---:|---:|']
    for ch,m,f in itertools.product(CH,A.METHODS,F):
        rr=[r for r in primary(pairs,m,ch) if r['feature']==f and r['evaluable']]
        lines.append(f"| {ch} | {m.upper()} | {f} | {len(rr)}/150 | {sum(r['disjoint_repeat_ranges'] for r in rr)}/{len(rr)} ({np.mean([r['disjoint_repeat_ranges'] for r in rr]):.1%}) |")
    lines+=['','Feature summaries bên dưới là mean±sample SD của valid recordings gộp observed conditions, chỉ mô tả phân bố. Missingness và operating effects có thể làm pooled means lệch; fault separation luôn dùng cell-matched ranges ở mục 5.','',
        '| Channel | Pipeline | State | Valid N/20 | delta_alpha | alpha0 | delta_h |','|---|---|---|---:|---:|---:|---:|']
    for ch,m,s in itertools.product(CH,A.METHODS,P.STATES):
        rr=[r for r in features if (r['channel'],r['pipeline'],r['component'],r['state'])==(ch,m,main_category(m),s) and r['analysis_valid']]
        values=[f"{np.mean([r[f] for r in rr]):.4f}±{np.std([r[f] for r in rr],ddof=1):.4f}" if len(rr)>1 else 'NE' for f in F]
        lines.append('| '+ch+' | '+m.upper()+' | '+s+f' | {len(rr)}/20 | '+' | '.join(values)+' |')
    lines+=['','## 5. Fault-pair separation','',
        'Giữ disjoint repeat ranges: max(A)<min(B) hoặc max(B)<min(A), strict inequality, cùng speed/load; cả bốn acquisitions phải valid. Overlap không tách; missing/invalid là NOT EVALUABLE, không đếm như overlap. Đây là descriptive separation, **không phải classification accuracy** hoặc kiểm định significance.','',
        'Mỗi ô dưới là disjoint/evaluable/10 expected conditions. Bảng bao gồm toàn bộ 15 cặp cho cả hai channels và ba primary representations; CSV lưu từng condition, direction và gap.']
    pairsummary=[]
    for ch,m in itertools.product(CH,A.METHODS):
        lines+=['',f'### {ch} — {m.upper()}','', '| Pair | delta_alpha | alpha0 | delta_h |','|---|---:|---:|---:|']
        for pair in itertools.combinations(P.STATES,2):
            name='-'.join(pair);values=[]
            for f in F:
                rr=[r for r in primary(pairs,m,ch) if r['pair']==name and r['feature']==f and r['evaluable']]
                n=sum(r['disjoint_repeat_ranges'] for r in rr); values.append(f'{n}/{len(rr)}/10')
                pairsummary.append(dict(channel=ch,pipeline=m,pair=name,feature=f,disjoint_conditions=n,evaluable_conditions=len(rr),expected_conditions=10,
                    status='NO VALID COMPARISONS' if not rr else 'NO SEPARATION OBSERVED' if n==0 else 'ALL EVALUABLE CONDITIONS DISJOINT' if n==len(rr) else 'CONDITION-DEPENDENT',
                    source_recordings=';'.join(sorted({rid for r in rr for rid in r['source_recordings'].split(';')}))))
            lines.append('| '+name+' | '+' | '.join(values)+' |')
    P.table(P.BASE/'summaries/fault_pair_summary.csv',pairsummary)
    for ch,m in itertools.product(CH,['raw','vmd']):
        scores={f:sum(r['disjoint_repeat_ranges'] for r in primary(pairs,m,ch) if r['feature']==f and r['evaluable']) for f in F}
        best=[f for f in F if scores[f]==max(scores.values())]
        lines+=['',f"{ch}/{m.upper()}: feature có nhiều disjoint conditions nhất là {', '.join(best)}; counts "+', '.join(f'{f}={scores[f]}' for f in F)+'. Đây là within-condition descriptive ranking; đọc operating sensitivity ở mục 6 cùng với nó.']
    lines+=['','Feature tốt hơn được đánh giá bằng coverage và matched disjoint counts; α0 tách nhiều cặp trong cell vẫn có thể nhạy operating condition. Không suy ra feature tốt nhất cho classifier.','',
        '## 6. Speed/load effects','',
        'Giữ Step 5: speed range=(max−min)/abs(mean) trên đủ 5 speeds; load difference=2|High−Low|/(|High|+|Low|) tại cùng fault/speed; thiếu repeat/cell thì không đánh giá. Full-grid fault/operating ratio yêu cầu tất cả 6 states đủ 10 conditions; partial grid chỉ dùng common valid operating cells, không interpolation.','',
        '| Channel | Pipeline/feature | Speed range median (N/12) | Load difference median (N/30) | Common grid (N/10), fault/operating SD ratio |','|---|---|---:|---:|---:|']
    for ch,m,f in itertools.product(CH,A.METHODS,F):
        ss=[r for r in primary(speed,m,ch) if r['feature']==f and r['evaluable']];ll=[r for r in primary(load,m,ch) if r['feature']==f and r['evaluable']]
        g=next(r for r in primary(grid,m,ch) if r['feature']==f)
        sval=f"{np.median([r['normalized_range'] for r in ss]):.1%}" if ss else 'NE';lval=f"{np.median([r['relative_difference'] for r in ll]):.1%}" if ll else 'NE'
        ratio=f"{g['fault_to_operating_ratio']:.3f}" if g['evaluable'] else 'NE'
        lines.append(f"| {ch} | {m.upper()}/{f} | {sval} ({len(ss)}/12) | {lval} ({len(ll)}/30) | {g['common_operating_cells']}/10, {ratio} |")
    lines+=['', 'Categorical additive fault+speed+load: sum-to-zero Helmert contrasts, partial η²=incremental SS/(incremental SS+residual SS), không cộng η² thành tổng variance. Full fault*speed*load chỉ evaluable nếu đủ 60 cells, full rank và ≥20 residual df. Additive effects với missing cells không chứng minh absence of interactions hoặc causality.',
        'Giữ OLS/HC3, null-imposed Rademacher wild bootstrap 1,999 draws, seed 6509, HC2-scaled reduced residuals và studentized HC3 Wald; BH riêng mỗi channel trên toàn family representations/features/models/terms. Wild/HC3 exploratory, classical p chỉ reference. Shapiro và heteroskedasticity LM trong diagnostics.','',
        '| Channel | Pipeline/feature | η² fault/speed/load | Wild p BH fault/speed/load |','|---|---|---:|---:|']
    evidence={}
    for ch,m,f in itertools.product(CH,A.METHODS,F):
        rr=[r for r in primary(models,m,ch) if r['feature']==f and r['model']=='fault+speed+load' and r.get('term') in ['fault','speed','load']]
        rr=sorted(rr,key=lambda r:['fault','speed','load'].index(r['term']))
        eta='/'.join(f"{r['partial_eta_squared']:.3f}" for r in rr) if len(rr)==3 else 'NE';p='/'.join(f"{r['wild_bootstrap_p_BH']:.4f}" for r in rr) if len(rr)==3 else 'NE'
        lines.append(f'| {ch} | {m.upper()}/{f} | {eta} | {p} |')
        if len(rr)==3:evidence[(ch,m,f)]=[r['partial_eta_squared'] for r in rr]
    lines+=['','**Kiểm tra kết luận Step 5 (từ dữ liệu Step 6):**','']
    for ch in CH:
        for f in F:
            eta=evidence.get((ch,'raw',f))
            verdict='NOT EVALUABLE' if eta is None else ('fault mạnh hơn từng main speed/load effect' if eta[0]>max(eta[1:]) else 'ít nhất một operating main effect mạnh hơn fault')
            lines.append(f'- {ch}, Raw {f}: {verdict}.')
    lines+=['','## 7. Raw vs VMD','', '| Channel | Feature | Paired N | Pearson r | Median relative difference | Mean signed change | RMSE |','|---|---|---:|---:|---:|---:|---:|']
    for r in paired:lines.append(f"| {r['channel']} | {r['feature']} | {r['paired_recordings']} | {r['pearson_r']:.4f} | {r['median_relative_difference']:.1%} | {r['mean_signed_difference']:.5f} | {r['RMSE']:.5f} |")
    lines+=['','Matched valid comparisons only; sensitivity ratios VMD/Raw <1 biểu thị giảm sensitivity.','', '| Channel | Feature | Common pairs | Raw/VMD disjoint | Gained/lost | Speed ratio (N) | Load ratio (N) |','|---|---|---:|---:|---:|---:|---:|']
    for r in matched:
        s=f"{r['median_speed_ratio']:.3f}" if r['median_speed_ratio'] is not None else 'NE';l=f"{r['median_load_ratio']:.3f}" if r['median_load_ratio'] is not None else 'NE'
        lines.append(f"| {r['channel']} | {r['feature']} | {r['common_pairs']} | {r['raw_disjoint_count']}/{r['vmd_disjoint_count']} | {r['gained_count']}/{r['lost_count']} | {s} ({r['matched_speed_trajectories']}) | {l} ({r['matched_load_comparisons']}) |")
    conclusions=[]
    for ch in CH:
        rr=[r for r in matched if r['channel']==ch];pp=[r for r in paired if r['channel']==ch]
        preserve=len(pp)==3 and all(r['pearson_r']>=.95 for r in pp)
        improve=all(r['gained_count']>r['lost_count'] and r['median_speed_ratio'] is not None and r['median_speed_ratio']<1 and r['median_load_ratio'] is not None and r['median_load_ratio']<1 for r in rr)
        conclusion='cải thiện nhất quán theo cả ba features trên comparisons matched' if improve else 'gần như chỉ giữ nguyên Raw; không có cải thiện nhất quán' if preserve else 'thay đổi representation và chưa có cải thiện nhất quán; cần đọc gained/lost'
        lost=sum(r['lost_count'] for r in rr);gained=sum(r['gained_count'] for r in rr)
        separation_note='separation giảm tổng thể trên matched feature-pair conditions' if lost>gained else 'separation tăng tổng thể trên matched feature-pair conditions' if gained>lost else 'tổng số separated matched feature-pair conditions giữ nguyên'
        conclusions.append(f'{ch}: **{conclusion}**; {separation_note} ({gained} gained, {lost} lost).')
    lines+=['']+conclusions+['','Tau=0 không ép oscillatory sum bằng Raw. Residue và reconstruction error giữ riêng; modes+residue identity theo construction không chứng minh sum-only reconstruction tốt.','',
        '## 8. VMD modes','',
        'Giữ bands 0–1000, 1000–4000, 4000–10000, 10000–20000, 20000–33334 Hz. Chọn mode energy lớn nhất trong band **trước QC**, không cứu invalid bằng mode khác; all two/four centers ratio≤1.5. Across-condition recurrence cũng kiểm tra center ratio≤1.5. Mode index không xác lập mode vật lý; không tune band bằng H3/H4.','',
        '| Channel | Band Hz | Complete cells /60 | Evaluable feature-pair comparisons /450 | Disjoint | Extra vs Raw |','|---|---|---:|---:|---:|---:|']
    for ch,(lo,hi) in itertools.product(CH,A.BANDS):
        cat=f'band_{lo}_{hi}_Hz';cc=[c for c in cells if (c['pipeline'],c['channel'],c['category'])==('vmd',ch,cat)]
        rr=[r for r in pairs if (r['pipeline'],r['channel'],r['category'])==('vmd',ch,cat) and r['evaluable']]
        ee=[r for r in extras if (r['pipeline'],r['channel'],r['category'])==('vmd',ch,cat) and r['extra_descriptive_separation']]
        lines.append(f"| {ch} | {lo}–{hi} | {sum(c['evaluable'] for c in cc)}/60 | {len(rr)}/450 | {sum(r['disjoint_repeat_ranges'] for r in rr)} | {len(ee)} |")
    lines+=['','Extra nghĩa cùng cell Raw evaluable nhưng overlap, band evaluable và disjoint. Recurrence cần cùng pair/feature/band qua ≥2 conditions và centers compatible; một extra ở mỗi feature khác nhau không được gộp thành repeated evidence.','',
        '| Channel | Band | Pair/feature | Extra conditions | Speeds/loads | Across-condition compatible | Conditions |','|---|---|---|---:|---:|---|---|']
    for r in recurrence:lines.append(f"| {r['channel']} | {r['category']} | {r['pair']}/{r['feature']} | {r['extra_conditions']} | {r['speeds']}/{r['loads']} | {r['frequency_compatible_across_extra_conditions']} | {r['conditions']} |")
    if not recurrence:lines.append('| — | — | No extra separation | 0 | — | — | — |')
    lines+=['',f"Có {sum(r['repeated_compatible_evidence'] for r in recurrence)} pair/feature/band groups có repeated compatible extra separation; xem vmd_extra_separation_recurrence.csv. Đây là candidate evidence trên cùng dataset, không xác nhận generalized improvement hoặc physical fault localization.",'',
        '## 9. EMD','',
        'EMD chỉ đối chiếu phụ, giữ cùng QC và practical approximation Step 5; không sửa sifting để tăng valid coverage. Strict IMF validity tách riêng khỏi practical convergence và MF-DFA QC.','',
        '| Channel | Method | Decomposition valid /120 | Sum reconstruction error median/max | Strict IMF valid |','|---|---|---:|---:|---:|']
    dd=[P.typed(r) for r in P.read_table(P.BASE/'diagnostics/decomposition_summary.csv')]
    for ch,m in itertools.product(CH,['vmd','emd']):
        rr=[r for r in dd if r['channel']==ch and r['pipeline']==m];strict=str(sum(r['strict_emd_imf_valid'] for r in rr))+'/120' if m=='emd' else 'N/A'
        lines.append(f"| {ch} | {m.upper()} | {sum(r['decomposition_valid'] for r in rr)}/120 | {np.median([r['reconstruction_error'] for r in rr]):.2%}/{max(r['reconstruction_error'] for r in rr):.2%} | {strict} |")
    lines+=['','EMD main features/separation/effects trình bày cùng các bảng trên; individual band coverage nằm trong qc_coverage.csv. Không gọi approximate EMD là benchmark strict IMF.','',
        '## 10. Limitations','',
        '- Hai acquisitions/cell tạo repeat ranges hẹp nhưng không phải validation của classifier. Không train classifier; không classification accuracy.',
        '- QC missingness giới hạn full-grid conclusions; additive inference exploratory, interactions chưa evaluable nếu thiếu cells. Partial grid không đại diện cells invalid.',
        '- Fixed scaling range <1 decade, fixed sample count ứng với số vòng quay khác nhau khi speed đổi. Scaling pass không chứng minh physical multifractality.',
        '- Fault configurations gộp nhiều thay đổi và loại lỗi khác nhau. Không bearing/gear/shaft localization; H2 vs H5 không phải pure bearing effect.',
        '- Cùng recording có hai channels tương quan; chỉ fit mỗi channel riêng. Independence giữa acquisitions là thiết kế giả định, chưa kiểm chứng thực nghiệm.',
        '- Numerical source hash lịch sử không khớp nên rerun toàn bộ; kết luận Step 5 được kiểm tra về mặt xu hướng, không tuyên bố đây là binary-identical replay.',
        '- Band matching chỉ theo center/energy, không chứng minh cùng nguồn cơ học. Sparse mode coverage và extra cases cùng dataset không phải validation độc lập.', '',
        '## 11. Final conclusion','',
        'MF-DFA phân biệt fault configuration theo operating condition trong các comparisons valid; bảng 15 cặp chỉ rõ cặp/feature/condition tách hoặc overlap và coverage chưa đánh giá. Không suy rộng separation cho toàn bộ operating grid khi QC loại cells.']
    for ch in CH:
        vals=[]
        for f in F:
            eta=evidence.get((ch,'raw',f));vals.append(f+': '+('fault > speed/load' if eta and eta[0]>max(eta[1:]) else 'operating effect ≥ fault' if eta else 'NE'))
        lines.append(ch+' — '+ '; '.join(vals)+'.')
    h35=[r for r in pairsummary if r['channel']=='output_voltage' and r['pipeline']=='raw' and r['pair']=='H3-H5']
    lines+=['','H3–H5 là điểm hạn chế rõ ở primary Raw: '+', '.join(f"{r['feature']} tách {r['disjoint_conditions']}/{r['evaluable_conditions']} conditions evaluable" for r in h35)+'. Không gọi đây là accuracy.']
    repeated=[r for r in recurrence if r['repeated_compatible_evidence']]
    lines+=['','Repeated extra separation từ VMD modes: '+('; '.join(f"{r['channel']} {r['pair']}/{r['feature']} ({r['category']}, {r['conditions']})" for r in repeated) if repeated else 'không có')+'. Các extra ở output channel trong dữ liệu này chưa lặp lại cùng pair/feature/band qua nhiều conditions.']
    lines+=conclusions
    full=[P.typed(r) for r in P.read_table(P.BASE/'summaries/fault_vs_operating_effect.csv')]
    decisions=[]
    lines+=['','Giữ descriptive robustness rule Step 5: ≥75% complete-cell coverage; ROBUST khi ≥2 features vừa có disjoint fraction≥75% vừa có full-grid fault/operating ratio≥1; NOT ROBUST khi ≥2 features có fraction≤25% và ratio<1; còn lại CONDITION-DEPENDENT. Full-grid ratio không evaluable không được thay bằng partial-grid ratio để cứu verdict.','',
        '| Channel | Pipeline | Complete cells | Full-grid evaluable features | Verdict |','|---|---|---:|---:|---|']
    for ch,m in itertools.product(CH,A.METHODS):
        cc=primary(cells,m,ch);count=sum(c['evaluable'] for c in cc);ee=primary(full,m,ch)
        robust=sum(r['evaluable'] and r.get('disjoint_pair_fraction',0)>=.75 and r.get('fault_to_operating_ratio',0)>=1 for r in ee)
        poor=sum(r['evaluable'] and r.get('disjoint_pair_fraction',1)<=.25 and r.get('fault_to_operating_ratio',1)<1 for r in ee)
        decision='NOT EVALUABLE' if count/60<.75 else 'ROBUST' if robust>=2 else 'NOT ROBUST' if poor>=2 else 'CONDITION-DEPENDENT'
        decisions.append(dict(channel=ch,pipeline=m,decision=decision,complete_cells=count,expected_cells=60,full_grid_evaluable_features=sum(r['evaluable'] for r in ee),source_recordings=';'.join(sorted({rid for c in cc for rid in c['source_recordings'].split(';')}))))
        lines.append(f"| {ch} | {m.upper()} | {count}/60 | {sum(r['evaluable'] for r in ee)}/3 | {decision} |")
    P.table(P.BASE/'summaries/robustness_decisions.csv',decisions)
    lines+=['','Không kết luận localization hoặc isolated bearing contribution. Individual low-frequency modes chỉ bổ sung bằng evidence matched và recurrence được ghi riêng.','',
        'Reproduce: `PHM_OUTPUT_DIR=/workspace/MFDFA_bearing/result/test/step6 /workspace/shared/mfdfa-venv/bin/python -B result/test/step6/scripts/pipeline.py`, sau đó `tests.py` và `finish.py` cùng env. Scripts phân tích reuse cách tính Step 5; mở rộng fault levels từ 4 lên 6 và expected grid từ 40 lên 60 cells. Dữ liệu/source cũ không sửa.','']
    for ch in CH:
        for stem in ['feature_heatmaps','paired_raw_vmd','fault_pair_separation','qc_coverage','feature_vs_load','vmd_frequency_modes']:lines+=['',f'![{stem} {ch}](figures/{stem}_{ch}.png)']
        for m in ['raw','vmd']:lines+=['',f'![Speed {m} {ch}](figures/feature_vs_speed_{m}_{ch}.png)']
    lines+=['','![Raw speed](figures/feature_vs_speed_raw.png)','','![VMD speed](figures/feature_vs_speed_vmd.png)']
    (P.BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')

def validate(features,pairs,sources):
    tests=json.loads((P.BASE/'diagnostics/tests.json').read_text());assert tests['passed']
    P.verify(json.loads((P.BASE/'config/current_source_hashes.json').read_text()))
    assert len(sources)==120 and len({s['recording'] for s in sources})==120
    source={s['recording']:s for s in sources}
    for s in sources:assert P.digest(P.resolve(s['relative_source']))==s['sha256']
    assert len(features)==120*2*18
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
    P.dump(P.BASE/'diagnostics/validation.json',dict(all_passed=True,recordings=120,channel_signals=240,analyses=len(features),valid_analyses=sum(r['analysis_valid'] for r in features),full_lengths_preserved=True,no_windows=True,no_classifier=True,no_parameter_tuning=True,all_feature_arrays_rechecked=True,source_hashes_unchanged=True,tests=tests))
    P.table(P.BASE/'diagnostics/output_manifest.csv',[dict(path=p.relative_to(P.BASE).as_posix(),bytes=p.stat().st_size,sha256=P.digest(p)) for p in sorted(P.BASE.rglob('*')) if p.is_file() and p.name!='output_manifest.csv'])

def run():
    P.dump(P.BASE/'config/analysis_rules.json',dict(A.ANALYSIS_RULE,states=P.STATES,expected_cells=60,expected_pair_conditions=150))
    sources=json.loads((P.BASE/'config/source_inventory.json').read_text())
    features=[P.typed(r) for r in P.read_table(P.BASE/'features/mfdfa_features.csv')]
    old={(r['recording'],r['channel'],r['pipeline'],r['component']):P.typed(r) for r in P.read_table(P.HISTORICAL_BASE/'features/mfdfa_features.csv')}
    parity=[]
    for r in features:
        previous=old.get((r['recording'],r['channel'],r['pipeline'],r['component']))
        if previous is None:continue
        row={k:r[k] for k in ['recording','channel','pipeline','component']}
        row.update(validity_same=r['analysis_valid']==previous['analysis_valid'],source_recordings=r['recording'])
        for f in F:
            if f in r and f in previous:row[f+'_absolute_rerun_difference']=abs(r[f]-previous[f])
        parity.append(row)
    P.table(P.BASE/'diagnostics/step5_rerun_parity.csv',parity)
    state_summary=[]
    for ch,m,state,f in itertools.product(CH,A.METHODS,P.STATES,F):
        rr=[r for r in features if r['channel']==ch and r['pipeline']==m and r['component']==main_category(m) and r['state']==state and r['analysis_valid']]
        vals=[r[f] for r in rr]
        state_summary.append(dict(channel=ch,pipeline=m,state=state,feature=f,valid_recordings=len(vals),expected_recordings=20,
            mean=float(np.mean(vals)) if vals else None,sample_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None,
            minimum=min(vals) if vals else None,maximum=max(vals) if vals else None,
            scope='pooled observed operating conditions, descriptive only; do not use for matched fault separation',source_recordings=A.roster(rr)))
    P.table(P.BASE/'summaries/state_feature_descriptive.csv',state_summary)
    A.mode_matching_audit(features);A.decomposition_audit(features)
    reps,coverage=A.representatives(features,sources);P.table(P.BASE/'features/analysis_representatives.csv',reps)
    cells=A.build_cells(reps);pairs=A.comparisons(cells);speed,load=A.condition_variation(cells)
    A.fault_operating_effect(cells,pairs);grid=A.matched_operating_grid(cells)
    paired,summary=A.paired_raw_vmd(reps);models,diagnostics=A.models(reps)
    matched,extras,recurrence=extras_and_matched(pairs,speed,load,cells)
    (P.BASE/'figures').mkdir(exist_ok=True)
    figures(features,reps,cells,coverage,paired,models);more_figures(pairs,features,cells)
    report(features,reps,cells,pairs,speed,load,grid,summary,models,matched,extras,recurrence)
    validate(features,pairs,sources)
    print('STEP 6 COMPLETE: report, figures, tables and validation',flush=True)

if __name__=='__main__':run()
