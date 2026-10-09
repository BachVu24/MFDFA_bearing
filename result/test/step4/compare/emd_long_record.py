"""Explicit practical stopping variant for long, noisy PHM recordings.

Reuses core EMD extrema detection and cubic mirror envelopes, but replaces
the near-zero-sensitive pointwise SD sum with relative energy change.
Requires relative RMS envelope mean <=0.05 and extrema/zero-crossing mismatch
<=0.5% (or one, whichever is larger). Reports the exact one-count IMF check
separately. These are approximate modes, not claimed exact Huang IMFs.
At the sifting cap retains a candidate WITH a nonconvergence flag so that
the requested exploratory comparison is available without hiding failures.
The original code/core/emd.py is read-only and unchanged.
"""
import numpy as np
from emd import _extrema, _envelope, _zero_crossings


def emd_long_record(x, max_imfs=6, max_siftings=200, sd_threshold=0.2,
                    envelope_tol=0.05, extrema_relative_tolerance=0.005):
    x=np.asarray(x,dtype=float)
    amplitude=float(np.max(np.abs(x)))
    if not amplitude: return np.empty((0,len(x))),x.copy(),dict(stop_reason='zero_signal',modes=[])
    residue=x/amplitude
    t=np.arange(len(x),dtype=float)
    modes, diagnostics=[],[]
    stop='max_imfs'
    for _ in range(max_imfs):
        maxima,minima=_extrema(residue)
        if not len(maxima) or not len(minima):
            stop='nonoscillatory_residue'; break
        if np.max(np.abs(residue))<1e-10:
            stop='small_residue'; break
        h=residue.copy()
        converged=False
        energy_sd,mean_ratio=np.inf,np.inf
        n_extrema=len(maxima)+len(minima)
        n_zero=_zero_crossings(h)
        mismatch=abs(n_extrema-n_zero)
        for iteration in range(1,max_siftings+1):
            maxima,minima=_extrema(h)
            if not len(maxima) or not len(minima): break
            mean=(_envelope(h,t,maxima)+_envelope(h,t,minima))/2
            previous=h
            h=previous-mean
            energy_sd=float(np.sum(mean**2)/max(np.sum(previous**2),np.finfo(float).tiny))
            maxima,minima=_extrema(h)
            if not len(maxima) or not len(minima): break
            mean=(_envelope(h,t,maxima)+_envelope(h,t,minima))/2
            mean_ratio=float(np.linalg.norm(mean)/max(np.linalg.norm(h),np.finfo(float).tiny))
            n_extrema=len(maxima)+len(minima)
            n_zero=_zero_crossings(h)
            mismatch=abs(n_extrema-n_zero)
            if (energy_sd<=sd_threshold and mean_ratio<=envelope_tol
                    and mismatch<=max(1,extrema_relative_tolerance*n_extrema)):
                converged=True; break
        if not np.any(h) or not np.isfinite(h).all():
            stop='invalid_candidate'; break
        diagnostics.append(dict(siftings=iteration,energy_sd=energy_sd,
            envelope_mean_ratio=mean_ratio,extrema=n_extrema,zero_crossings=n_zero,
            exact_one_count_condition=bool(mismatch<=1),relative_count_mismatch=mismatch/max(n_extrema,1),
            converged_practical_rule=converged,status='approximate_mode' if converged else 'sifting_cap_candidate'))
        modes.append(h)
        residue=residue-h
    imfs=np.asarray(modes).reshape((-1,len(x)))*amplitude
    info=dict(stop_reason=stop,modes=diagnostics,n_imfs=len(modes),
        stopping_rule='relative energy SD and RMS envelope mean; count mismatch <=0.5%',
        all_practical_modes_converged=all(d['converged_practical_rule'] for d in diagnostics),
        all_exact_one_count_conditions=all(d['exact_one_count_condition'] for d in diagnostics))
    return imfs,x-imfs.sum(axis=0),info
