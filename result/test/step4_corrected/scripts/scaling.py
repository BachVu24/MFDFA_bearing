"""Label-blind refitting and QC of existing MF-DFA fluctuation functions.

Does not change the profile, detrending, segment averaging, or q=0 formula.
Selection receives only numerical Fq/scales; class labels are not arguments.
"""
import numpy as np

CANDIDATES=[(64,512),(128,1024),(256,2048),(512,4096),(1024,8192)]
RULE=dict(min_scale_points=8,min_r2_floor=.90,r2_threshold=.95,
          min_good_r2_fraction=.80,max_local_slope_cv=.50,
          min_stable_q_fraction=.80,min_hq_exclusive=.05)


def fit_scaling(scales,fq,q,bounds,rule=RULE):
    scales,fq,q=np.asarray(scales),np.asarray(fq),np.asarray(q)
    if fq.shape!=(len(q),len(scales)) or np.any(fq<=0) or not np.isfinite(fq).all():
        raise ValueError('Fq must be finite, positive and have shape (q,scales)')
    mask=(scales>=bounds[0])&(scales<=bounds[1])
    if mask.sum()<2: raise ValueError('Range contains fewer than two scales')
    lx=np.log(scales[mask]); ly=np.log(fq[:,mask])
    coefficients=np.polyfit(lx,ly.T,1)
    h=coefficients[0]
    residual=ly-(h[:,None]*lx+coefficients[1,:,None])
    total=np.sum((ly-ly.mean(axis=1,keepdims=True))**2,axis=1)
    r2=np.full(len(q),np.nan)
    np.divide(np.sum(residual**2,axis=1),total,out=r2,where=total>0)
    r2=1-r2
    local=np.diff(ly,axis=1)/np.diff(lx)
    mean=local.mean(axis=1); std=local.std(axis=1)
    cv=std/np.maximum(np.abs(mean),1e-12)
    tau=q*h-1
    alpha=np.gradient(tau,q,edge_order=2 if len(q)>=3 else 1)
    spectrum=q*alpha-tau
    reasons=[]
    if mask.sum()<rule['min_scale_points']: reasons.append('insufficient_scale_points')
    if not np.isfinite(r2).all(): reasons.append('undefined_r2')
    if np.nanmin(r2)<rule['min_r2_floor']: reasons.append('min_r2_below_floor')
    fraction=float(np.mean(r2>=rule['r2_threshold']))
    if fraction<rule['min_good_r2_fraction']: reasons.append('insufficient_good_r2_fraction')
    stable=float(np.mean(cv<=rule['max_local_slope_cv']))
    if stable<rule['min_stable_q_fraction']: reasons.append('unstable_local_slopes')
    if np.min(h)<=rule['min_hq_exclusive']: reasons.append('hq_near_zero_or_nonpositive')
    if not all(np.isfinite(v).all() for v in (h,tau,alpha,spectrum,cv)): reasons.append('nonfinite_result')
    metrics=dict(scaling_valid=not reasons,invalid_reason=';'.join(reasons),
        min_r2=float(np.nanmin(r2)),median_r2=float(np.nanmedian(r2)),good_r2_fraction=fraction,
        minimum_hq=float(h.min()),nonpositive_hq_count=int(np.sum(h<=0)),
        nonpositive_hq_fraction=float(np.mean(h<=0)),local_slope_cv=float(np.median(cv)),
        local_slope_cv_max=float(cv.max()),stable_local_slope_fraction=stable,
        scale_points=int(mask.sum()),fit_scale_min=int(scales[mask][0]),fit_scale_max=int(scales[mask][-1]),
        delta_alpha=float(np.ptp(alpha)),alpha0=float(alpha[np.flatnonzero(q==0)[0]]),
        delta_h=float(np.ptp(h)),h2=float(h[np.flatnonzero(q==2)[0]]))
    # Shape diagnostics are audit warnings, not thresholds tuned after range
    # selection. Good scaling alone does not establish a concave spectrum.
    metrics.update(hq_increasing_steps=int(np.sum(np.diff(h)>1e-10)),
        alpha_increasing_steps=int(np.sum(np.diff(alpha)>1e-10)),
        f_alpha_above_dimension_count=int(np.sum(spectrum>1+1e-8)),
        maximum_f_alpha=float(spectrum.max()),
        spectrum_shape_warning=bool(np.any(np.diff(alpha)>1e-10) or np.any(spectrum>1+1e-8)))
    arrays=dict(hq=h,tau=tau,alpha=alpha,f_alpha=spectrum,r_squared=r2,
        local_slopes=local,local_slope_mean=mean,local_slope_std=std,local_slope_cv=cv,
        fit_mask=mask,local_slope_scales=np.sqrt(scales[mask][:-1]*scales[mask][1:]))
    return arrays,metrics


def select_common(raw_diagnostics):
    """Require every raw recording/channel to pass. Never use class separation.

    If no candidate passes all inputs, return no valid common range and
    retain the highest-coverage working candidate only for flagged audit.
    Ties: highest worst R2, then lowest median CV, then smallest lower bound.
    """
    summaries=[]
    for bounds in CANDIDATES:
        rows=[r for r in raw_diagnostics if (r['candidate_min'],r['candidate_max'])==bounds]
        summaries.append(dict(candidate_min=bounds[0],candidate_max=bounds[1],
            raw_inputs=len(rows),raw_pass_count=sum(r['scaling_valid'] for r in rows),
            all_raw_pass=all(r['scaling_valid'] for r in rows),worst_r2=min(r['min_r2'] for r in rows),
            median_local_cv=float(np.median([r['local_slope_cv'] for r in rows])),
            min_good_r2_fraction=min(r['good_r2_fraction'] for r in rows),
            min_stable_q_fraction=min(r['stable_local_slope_fraction'] for r in rows),
            scale_points=rows[0]['scale_points']))
    ordered=sorted(summaries,key=lambda r:(-r['raw_pass_count'],-r['worst_r2'],r['median_local_cv'],r['candidate_min']))
    chosen=ordered[0]
    selection=dict(selected_interval=[chosen['candidate_min'],chosen['candidate_max']],
        common_valid=chosen['all_raw_pass'],rule=RULE,
        selection_population='all 16 raw recording/channel signals; no class labels or separation metrics',
        selection_rule='all raw inputs must pass; ties: worst R2 descending, median local CV ascending, lower bound ascending',
        chosen_summary=chosen)
    return selection,summaries
