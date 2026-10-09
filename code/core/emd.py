"""EMD: Huang et al. (1998), sections 3 and 5, Eqs. (5.3)-(5.8).

Source: paper/Huang1996.pdf (received 1996, published 1998).
Cubic splines with mirrored extrema are a numerical endpoint convention;
the paper does not prescribe a unique endpoint implementation.
Requires NumPy and SciPy. No file I/O.
"""
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

__all__ = ["emd"]


def _extrema(x):
    # A plateau counts once, at its midpoint.
    return find_peaks(x)[0], find_peaks(-x)[0]


def _zero_crossings(x):
    signs = np.sign(x[x != 0])
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _envelope(x, t, indices):
    count = min(2, len(indices))
    knots = np.concatenate((2*t[0]-t[indices[:count]][::-1], t[indices],
                            2*t[-1]-t[indices[-count:]][::-1]))
    values = np.concatenate((x[indices[:count]][::-1], x[indices],
                             x[indices[-count:]][::-1]))
    return CubicSpline(knots, values, bc_type="natural")(t)


def emd(x, max_imfs=None, *, t=None, max_siftings=1000,
        sd_threshold=0.2, envelope_tol=0.05, residual_tol=1e-10,
        return_info=False):
    """Return (imfs, residue), imfs shape (n_imfs, n_samples).

    Accept an IMF when extrema/zero-crossing counts differ by <=1,
    relative RMS envelope mean <= envelope_tol, and the pointwise SD
    sum in Eq. (5.5) <= sd_threshold. Regularize near-zero denominators
    by machine epsilon times mode power. Tolerances approximate the
    IMF definition in finite precision. Stop at nonoscillatory/tiny
    residue or max_imfs. Failed sifting warns and retains the remaining
    signal as residue, without labeling an unconverged component an IMF.
    Optional info reports all accepted mode criteria and stopping reason.
    t may contain strictly increasing nonuniform sample times.
    """
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise ValueError("x must be real")
    signal = np.asarray(raw, dtype=float)
    if signal.ndim != 1 or signal.size < 3 or not np.all(np.isfinite(signal)):
        raise ValueError("x must be finite, 1-D, with at least 3 samples")
    times = np.arange(signal.size, dtype=float) if t is None else np.asarray(t, dtype=float)
    if (times.shape != signal.shape or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("t must match x and be finite and strictly increasing")
    if max_imfs is not None and (isinstance(max_imfs, bool)
            or not isinstance(max_imfs, (int, np.integer)) or max_imfs < 0):
        raise ValueError("max_imfs must be a nonnegative integer or None")
    if (isinstance(max_siftings, bool)
            or not isinstance(max_siftings, (int, np.integer)) or max_siftings < 1):
        raise ValueError("max_siftings must be a positive integer")
    for name, value in [("sd_threshold", sd_threshold), ("envelope_tol", envelope_tol),
                        ("residual_tol", residual_tol)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    amplitude = np.max(np.abs(signal))
    residue = signal.copy() / (amplitude if amplitude else 1.0)
    modes, diagnostics = [], []
    reason = "nonoscillatory_residue"
    while max_imfs is None or len(modes) < max_imfs:
        maxima, minima = _extrema(residue)
        if not len(maxima) or not len(minima):
            break
        if np.max(np.abs(residue)) <= residual_tol:
            reason = "small_residue"
            break
        h, accepted = residue.copy(), False
        for iteration in range(1, max_siftings + 1):
            maxima, minima = _extrema(h)
            if not len(maxima) or not len(minima):
                break
            mean = (_envelope(h, times, maxima)+_envelope(h, times, minima))/2
            previous = h
            h = previous-mean
            floor = np.finfo(float).eps * max(np.mean(previous**2), np.finfo(float).tiny)
            sd = float(np.sum((previous-h)**2/np.maximum(previous**2, floor)))
            maxima, minima = _extrema(h)
            if not len(maxima) or not len(minima) or not np.any(h):
                break
            envelope_mean = (_envelope(h, times, maxima)+_envelope(h, times, minima))/2
            mean_ratio = float(np.linalg.norm(envelope_mean)/np.linalg.norm(h))
            count_difference = abs(len(maxima)+len(minima)-_zero_crossings(h))
            if sd <= sd_threshold and mean_ratio <= envelope_tol and count_difference <= 1:
                accepted = True
                diagnostics.append(dict(siftings=iteration, sd=sd,
                                        envelope_mean_ratio=mean_ratio,
                                        extrema=len(maxima)+len(minima),
                                        zero_crossings=_zero_crossings(h)))
                break
        if not accepted:
            reason = "sifting_not_converged"
            warnings.warn("EMD sifting did not converge; remaining signal kept in residue",
                          RuntimeWarning, stacklevel=2)
            break
        modes.append(h.copy())
        residue -= h
    else:
        reason = "max_imfs"
    imfs = np.asarray(modes).reshape((-1, signal.size))*amplitude
    residue = signal-imfs.sum(axis=0)
    info = dict(stop_reason=reason, modes=diagnostics, n_imfs=len(modes))
    return (imfs, residue, info) if return_info else (imfs, residue)
