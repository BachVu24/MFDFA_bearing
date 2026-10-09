"""VMD: Dragomiretskiy & Zosso (2014), Algorithm 2, Eqs. (27)-(29).

Source: paper/dragomiretskiy2014.pdf, IEEE TSP 62(3), 531-544.
One-sided real FFT with half-length mirror extension (section III.D).
Requires NumPy only. No file I/O.
"""
import warnings
import numpy as np

__all__ = ["vmd", "VMD"]


def vmd(x, alpha=2000.0, tau=0.5, K=3, DC=False, init=1, tol=1e-7,
        *, max_iter=2000, fs=1.0, seed=0, return_info=False):
    """Return (modes, spectra, omega_history).

    modes: (K,N); spectra: fftshift(FFT(modes)) transposed to (N,K).
    omega_history: (iterations+1,K), centers in cycles per time unit,
    i.e. Hz when fs is samples/second. alpha weights squared bandwidth
    in cycles/sample: denominator 1+2*alpha*(f-omega)**2; scalar or K values.
    tau=0 disables dual ascent for denoising; modes need not sum to x.
    tau>0 enforces reconstruction iteratively. No artificial correction
    is added to the modes. DC locks the first center at zero.
    init=0: zero centers; 1: uniform; 2: seeded log-uniform random centers.
    Modes retain initialization order, and odd input length is preserved.

    Convergence uses the paper's sum of per-mode relative squared
    spectral changes. For tau>0 also require relative constraint error
    <=sqrt(tol) on the mirrored signal. A cap warns and is exposed in info.
    """
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise ValueError("x must be real")
    signal = np.asarray(raw, dtype=float)
    if signal.ndim != 1 or signal.size < 3 or not np.all(np.isfinite(signal)):
        raise ValueError("x must be finite, 1-D, with at least 3 samples")
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("K must be a positive integer")
    penalties = np.asarray(alpha, dtype=float)
    if penalties.ndim == 0:
        penalties = np.full(K, float(penalties))
    if penalties.shape != (K,) or np.any(~np.isfinite(penalties)) or np.any(penalties <= 0):
        raise ValueError("alpha must be positive and finite, scalar or length K")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    if not np.isfinite(tol) or tol <= 0 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("tol and fs must be positive and finite")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if init not in (0, 1, 2):
        raise ValueError("init must be 0, 1 or 2")
    amplitude = np.max(np.abs(signal))
    normalized = signal/(amplitude if amplitude else 1.0)
    half = signal.size//2
    extended = np.concatenate((normalized[:half][::-1], normalized, normalized[-half:][::-1]))
    length = extended.size
    frequencies = np.fft.rfftfreq(length)
    target = np.fft.rfft(extended)
    modes_hat = np.zeros((K, target.size), dtype=complex)
    dual, centers = np.zeros_like(target), np.zeros(K)
    if init == 1:
        centers = 0.5*np.arange(K)/K
    elif init == 2:
        centers = np.sort(np.exp(np.random.default_rng(seed).uniform(
            np.log(1.0/length), np.log(0.5), K)))
    if DC:
        centers[0] = 0
    history = [centers.copy()]
    target_power = float(np.vdot(target, target).real)
    converged = target_power == 0
    change, constraint, iterations = 0.0, 0.0, 0
    if not converged:
        for iterations in range(1, max_iter+1):
            old = modes_hat.copy()
            total = modes_hat.sum(axis=0)
            for k in range(K):
                # Gauss-Seidel: new modes for i<k, old modes for i>k.
                total -= modes_hat[k]
                modes_hat[k] = (target-total+dual/2)/(
                    1+2*penalties[k]*(frequencies-centers[k])**2)
                total += modes_hat[k]
                power = np.abs(modes_hat[k])**2
                if not (DC and k == 0) and power.sum() > 0:
                    centers[k] = np.dot(frequencies, power)/power.sum()
            error = target-total
            dual += tau*error
            numerator = np.sum(np.abs(modes_hat-old)**2, axis=1)
            denominator = np.sum(np.abs(old)**2, axis=1)
            change = float(np.sum(numerator/np.maximum(
                denominator, np.finfo(float).eps*target_power)))
            constraint = float(np.linalg.norm(error)/np.sqrt(target_power))
            history.append(centers.copy())
            if change <= tol and (tau == 0 or constraint <= np.sqrt(tol)):
                converged = True
                break
    if not converged:
        warnings.warn("VMD reached max_iter before convergence", RuntimeWarning, stacklevel=2)
    modes = np.fft.irfft(modes_hat, n=length, axis=1)[:, half:half+signal.size]*amplitude
    spectra = np.fft.fftshift(np.fft.fft(modes, axis=1), axes=1).T
    history = np.asarray(history)*fs
    norm = np.linalg.norm(normalized)
    reconstruction_error = float(np.linalg.norm(normalized-modes.sum(axis=0)/
                                (amplitude if amplitude else 1.0))/norm) if norm else 0.0
    info = dict(converged=converged, iterations=iterations, relative_change=change,
                constraint_error=constraint, reconstruction_error=reconstruction_error,
                frequency_unit="cycles per time unit", tau=float(tau))
    result = (modes, spectra, history)
    return (*result, info) if return_info else result


def VMD(f, alpha, tau, K, DC, init, tol, **kwargs):
    """Compatibility entry point with the conventional VMD argument order."""
    return vmd(f, alpha, tau, K, DC, init, tol, **kwargs)
