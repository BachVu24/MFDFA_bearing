"""MF-DFA: Kantelhardt et al. (2002), Physica A 316, 87-114.

Source: paper/Kantelhardt2002.pdf. Eqs. (1)-(6), (14)-(16).
Requires NumPy only; no file I/O. Original public return order is kept.
"""
import numpy as np

__all__ = ["mfdfa", "multifractal_spectrum"]


def mfdfa(x, q_list, m=2, min_scale=32, *, scales=None, max_scale=None,
          n_scales=30, fit_range=None, return_info=False):
    """Return (scales, Fq, hq, tau), Fq shape (len(q_list), len(scales)).

    Integrate the mean-centered series and detrend 2*floor(N/s) segments
    starting at both ends. Degree m fits the profile and removes degree
    m-1 trends from x. q=0 uses logarithmic averaging; other moments
    use log-domain arithmetic. Default scales: min_scale to N//10.
    Explicit scales must be integers with m+2 < s <= N//4 (paper's
    reliability bound), with at least two distinct scales. fit_range
    optionally restricts the log-log regression. Constant/exactly
    detrended series have undefined scaling and raise ValueError.
    Zero segment variance also makes q<=0 moments undefined.
    h(2) equals H for stationary series; tau=q*hq-1 assumes compact
    one-dimensional support. Optional info: segment counts, fit mask, R2.
    """
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise ValueError("x must be real")
    x = np.asarray(raw, dtype=float)
    q = np.asarray(q_list, dtype=float)
    if x.ndim != 1 or x.size < 4 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite 1-D series")
    if q.ndim != 1 or not q.size or not np.all(np.isfinite(q)):
        raise ValueError("q_list must be a nonempty finite 1-D array")
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer")
    if scales is None:
        upper = x.size//10 if max_scale is None else max_scale
        for name, value in [("min_scale", min_scale), ("max_scale", upper), ("n_scales", n_scales)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if min_scale <= m+2 or upper <= min_scale or upper > x.size//4 or n_scales < 2:
            raise ValueError("require m+2 < min_scale < max_scale <= N//4 and n_scales >= 2")
        scales = np.unique(np.floor(np.geomspace(min_scale, upper, n_scales)).astype(int))
        scales[0], scales[-1] = min_scale, upper
    else:
        values = np.asarray(scales, dtype=float)
        if (values.ndim != 1 or not np.all(np.isfinite(values))
                or np.any(values != np.floor(values))
                or np.any(values <= m+2) or np.any(values > x.size//4)):
            raise ValueError("scales must be 1-D integers with m+2 < s <= N//4")
        scales = np.unique(values.astype(int))
    if len(scales) < 2:
        raise ValueError("at least two distinct scales are required")
    mask = np.ones(len(scales), dtype=bool)
    if fit_range is not None:
        bounds = np.asarray(fit_range, dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError("fit_range must be an increasing finite pair")
        mask = (scales >= bounds[0]) & (scales <= bounds[1])
        if mask.sum() < 2:
            raise ValueError("fit_range must contain at least two scales")
    amplitude = np.max(np.abs(x))
    if amplitude == 0:
        raise ValueError("a constant signal has no defined MF-DFA scaling")
    normalized = x/amplitude
    centered = normalized-normalized.mean()
    if not np.any(centered):
        raise ValueError("a constant signal has no defined MF-DFA scaling")
    profile = np.cumsum(centered)
    log_f = np.empty((len(q), len(scales)))
    counts = []
    for si, scale in enumerate(scales):
        ns = x.size//scale
        starts = np.concatenate((np.arange(ns)*scale, x.size-(np.arange(ns)+1)*scale))
        segments = profile[starts[:, None]+np.arange(scale)]
        # Removing segment offsets improves numerical conditioning.
        segments = segments-segments.mean(axis=1, keepdims=True)
        basis = np.polynomial.polynomial.polyvander(np.linspace(-1, 1, scale), m)
        orthogonal, _ = np.linalg.qr(basis, mode="reduced")
        detrended = segments-(segments @ orthogonal) @ orthogonal.T
        variance = np.mean(detrended**2, axis=1)
        numerical_zero = variance <= (100*np.finfo(float).eps)**2*np.mean(segments**2, axis=1)
        if np.all(numerical_zero) or (np.any(numerical_zero) and np.any(q <= 0)):
            raise ValueError("zero detrended variance: scaling or nonpositive q moments undefined")
        variance[numerical_zero] = 0
        with np.errstate(divide="ignore"):
            logs = np.log(variance)
        for qi, moment in enumerate(q):
            if moment == 0:
                log_f[qi, si] = 0.5*logs.mean()
            elif abs(moment) < 1e-6 and np.all(np.isfinite(logs)):
                mean_log = logs.mean()
                log_f[qi, si] = 0.5*mean_log+np.log1p(
                    np.mean(np.expm1(0.5*moment*(logs-mean_log))))/moment
            else:
                terms = 0.5*moment*logs
                peak = terms.max()
                log_f[qi, si] = (peak+np.log(np.exp(terms-peak).mean()))/moment
        counts.append(2*ns)
    log_s = np.log(scales[mask])
    coefficients = np.polyfit(log_s, log_f[:, mask].T, 1)
    hq = coefficients[0]
    fitted = coefficients[0, :, None]*log_s+coefficients[1, :, None]
    errors = np.sum((log_f[:, mask]-fitted)**2, axis=1)
    total = np.sum((log_f[:, mask]-log_f[:, mask].mean(axis=1, keepdims=True))**2, axis=1)
    r_squared = np.full(len(q), np.nan)
    np.divide(errors, total, out=r_squared, where=total > 0)
    r_squared = 1-r_squared
    Fq = np.exp(log_f+np.log(amplitude))
    tau = q*hq-1
    result = (scales, Fq, hq, tau)
    info = dict(segment_counts=np.array(counts), fit_mask=mask, r_squared=r_squared,
                polynomial_order=int(m), profile="cumsum(x - mean(x))")
    return (*result, info) if return_info else result


def multifractal_spectrum(q, tau):
    """Eq. (15): alpha=d(tau)/dq; f(alpha)=q*alpha-tau.

    Require finite, strictly increasing q with at least two entries.
    Use centered interior differences and second-order endpoint
    differences where at least three q values are available.
    """
    q, tau = np.asarray(q, dtype=float), np.asarray(tau, dtype=float)
    if (q.ndim != 1 or q.size < 2 or tau.shape != q.shape
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(tau))
            or np.any(np.diff(q) <= 0)):
        raise ValueError("q and tau must match; q must be finite and strictly increasing")
    alpha = np.gradient(tau, q, edge_order=2 if len(q) >= 3 else 1)
    return alpha, q*alpha-tau
