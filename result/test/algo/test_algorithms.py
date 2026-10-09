"""Reproducible numerical checks. Run: python -B result/test/algo/test_algorithms.py.

Writes only beside this script; imports code/core without bytecode caches.
Analytic signals, an independent least-squares MF-DFA implementation,
one-step ADMM formulas and theoretical binomial exponents are the oracles.
"""
import sys
sys.dontwrite_bytecode = True
from pathlib import Path
import os
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
os.environ["MPLCONFIGDIR"] = str(OUT / "mplconfig")
sys.path.insert(0, str(ROOT / "code" / "core"))
import io
import json
import platform
import time
import unittest
import warnings
import numpy as np
import scipy
from scipy.signal import hilbert, chirp
from emd import emd
from vmd import vmd, VMD
from mfdfa import mfdfa, multifractal_spectrum

METRICS = {}
DATA = {}


def relative_error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def dominant(x, fs):
    return float(np.fft.rfftfreq(len(x), 1/fs)[np.argmax(np.abs(np.fft.rfft(x)))])


def reference_mfdfa(x, q, scales, m):
    # Deliberately independent polyfit loops, direct powers, and explicit indices.
    profile = np.cumsum(x-np.mean(x))
    result = np.empty((len(q), len(scales)))
    for j, s in enumerate(scales):
        variances = []
        ns = len(x)//s
        for side in (0, 1):
            for v in range(ns):
                start = v*s if side == 0 else len(x)-(v+1)*s
                segment = profile[start:start+s]
                t = np.arange(s)
                fit = np.polyval(np.polyfit(t, segment, m), t)
                variances.append(np.mean((segment-fit)**2))
        f2 = np.array(variances)
        for i, moment in enumerate(q):
            result[i, j] = (np.exp(np.mean(np.log(f2))/2) if moment == 0
                            else np.mean(f2**(moment/2))**(1/moment))
    return result


class EMDTests(unittest.TestCase):
    def test_three_tones_and_imf_definition(self):
        fs = 1024
        t = np.arange(2048)/fs
        components = np.array([0.4*np.cos(2*np.pi*160*t),
                               0.7*np.cos(2*np.pi*64*t), np.cos(2*np.pi*8*t)])
        x = components.sum(axis=0)
        modes, residue, info = emd(x, max_imfs=3, return_info=True)
        self.assertEqual(modes.shape, (3, len(x)))
        self.assertLess(relative_error(modes.sum(axis=0)+residue, x), 1e-14)
        peaks = [dominant(mode, fs) for mode in modes]
        np.testing.assert_allclose(peaks, [160, 64, 8], atol=0.5)
        correlations = [float(np.corrcoef(a[100:-100], b[100:-100])[0, 1])
                        for a, b in zip(modes, components)]
        self.assertGreater(min(correlations), 0.95)
        for item in info["modes"]:
            self.assertLessEqual(abs(item["extrema"]-item["zero_crossings"]), 1)
            self.assertLessEqual(item["sd"], 0.2)
            self.assertLessEqual(item["envelope_mean_ratio"], 0.05)
        METRICS["emd_three_tones"] = dict(peaks_hz=peaks, correlations=correlations,
            reconstruction_error=relative_error(modes.sum(axis=0)+residue, x), diagnostics=info)
        DATA.update(emd_signal=x, emd_modes=modes, emd_residue=residue, emd_time=t)

    def test_single_tone(self):
        x = np.cos(2*np.pi*32*np.arange(1024)/1024)
        modes, residue = emd(x)
        self.assertEqual(len(modes), 1)
        self.assertLess(relative_error(modes[0], x), 1e-12)

    def test_amplitude_modulated_chirp(self):
        t = np.arange(2048)/1024
        x = (1+0.2*np.cos(2*np.pi*2*t))*chirp(t, f0=20, f1=80, t1=2)
        modes, residue = emd(x, max_imfs=1, t=t)
        self.assertEqual(len(modes), 1)
        self.assertGreater(np.corrcoef(modes[0][100:-100], x[100:-100])[0, 1], 0.99)
        phase = np.unwrap(np.angle(hilbert(modes[0])))
        estimated = np.diff(phase)*1024/(2*np.pi)
        error = float(np.median(np.abs(estimated[100:-100]-(20+30*t[:-1])[100:-100])))
        self.assertLess(error, 1.0)
        METRICS["emd_chirp"] = dict(median_frequency_error_hz=error)

    def test_monotone_and_constant(self):
        for x in [np.zeros(64), np.full(64, 2.0), np.arange(64.), -np.arange(64.)]:
            modes, residue = emd(x)
            self.assertEqual(modes.shape, (0, len(x)))
            np.testing.assert_array_equal(residue, x)

    def test_plateau_and_nonuniform_time(self):
        x = np.tile([0., 1., 1., 0., -1., -1., 0.], 40)
        t = np.cumsum(np.linspace(0.8, 1.2, len(x)))
        modes, residue = emd(x, max_imfs=1, t=t)
        self.assertEqual(len(modes), 1)
        np.testing.assert_allclose(modes.sum(axis=0)+residue, x, atol=1e-14)

    def test_scaling_and_cap(self):
        t = np.arange(1024)/1024
        x = np.cos(2*np.pi*8*t)+0.5*np.cos(2*np.pi*64*t)
        a, r = emd(x, max_imfs=2)
        b, s = emd(x*1e-12, max_imfs=2)
        np.testing.assert_allclose(a, b/1e-12, atol=1e-8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            modes, residue, info = emd(x, max_siftings=1, return_info=True)
        self.assertTrue(caught)
        self.assertEqual(info["stop_reason"], "sifting_not_converged")
        np.testing.assert_allclose(modes.sum(axis=0)+residue, x)

    def test_invalid_inputs(self):
        for x in [[], [1, 2], [[1, 2, 3]], [1, np.nan, 2], [1j, 2, 3]]:
            with self.assertRaises(ValueError):
                emd(x)
        for kwargs in [dict(t=[0, 0, 1]), dict(max_imfs=-1), dict(max_siftings=0),
                       dict(sd_threshold=0), dict(envelope_tol=np.nan)]:
            with self.assertRaises(ValueError):
                emd([0, 1, 0], **kwargs)


class VMDTests(unittest.TestCase):
    def test_three_tones(self):
        fs = 1024
        t = np.arange(2048)/fs
        frequencies = np.array([8., 64., 160.])
        components = np.array([np.cos(2*np.pi*8*t), 0.7*np.cos(2*np.pi*64*t),
                               0.4*np.cos(2*np.pi*160*t)])
        x = components.sum(axis=0)
        modes, spectra, history, info = vmd(x, fs=fs, return_info=True)
        self.assertTrue(info["converged"])
        order = np.argsort(history[-1])
        np.testing.assert_allclose(history[-1, order], frequencies, atol=0.2)
        self.assertLess(info["reconstruction_error"], 4e-4)
        correlations = [float(np.corrcoef(a[100:-100], b[100:-100])[0, 1])
                        for a, b in zip(modes[order], components)]
        self.assertGreater(min(correlations), 0.98)
        self.assertEqual(spectra.shape, (len(x), 3))
        np.testing.assert_allclose(np.fft.ifft(np.fft.ifftshift(spectra, axes=0), axis=0).real.T,
                                   modes, atol=1e-12)
        METRICS["vmd_three_tones"] = dict(centers_hz=history[-1, order].tolist(),
                                        correlations=correlations, diagnostics=info)
        DATA.update(vmd_signal=x, vmd_modes=modes[order], vmd_time=t, vmd_centers=history)

    def test_single_tone_odd_length_and_nyquist(self):
        for n in (511, 512):
            x = np.cos(2*np.pi*23*np.arange(n)/n)
            modes, _, history, info = vmd(x, K=1, alpha=300, return_info=True)
            self.assertEqual(modes.shape, (1, n))
            self.assertTrue(info["converged"])
            self.assertLess(relative_error(modes[0], x), 4e-4)
            self.assertLess(abs(history[-1, 0]-23/n), 0.002)
        x = (-1.)**np.arange(256)
        modes, _, _, info = vmd(x, K=1, alpha=100, return_info=True)
        self.assertTrue(info["converged"])
        self.assertLess(relative_error(modes[0], x), 4e-4)

    def test_zero_constant_and_dc(self):
        modes, _, _, info = vmd(np.zeros(101), return_info=True)
        self.assertTrue(info["converged"])
        np.testing.assert_array_equal(modes, np.zeros((3, 101)))
        x = np.full(128, 3.)
        modes, _, history = vmd(x, K=1, DC=True)
        np.testing.assert_allclose(modes[0], x, atol=1e-10)
        np.testing.assert_array_equal(history, np.zeros_like(history))
        t = np.arange(512)/512
        x = 2+np.cos(2*np.pi*48*t)
        modes, _, history, info = vmd(x, K=2, DC=True, fs=512, return_info=True)
        self.assertTrue(info["converged"])
        np.testing.assert_array_equal(history[:, 0], np.zeros(len(history)))
        self.assertLess(abs(modes[0].mean()-2), 0.001)

    def test_one_admm_step_independent_formula(self):
        x = np.array([0., 1., -0.5, 0.25, -1., 0.7, 0.2, 0.9, -0.1])
        alpha = np.array([50., 80.])
        half = len(x)//2
        extended = np.r_[x[:half][::-1], x, x[-half:][::-1]]
        f = np.fft.rfftfreq(len(extended))
        fhat = np.fft.rfft(extended)
        u0 = fhat/(1+2*alpha[0]*f**2)
        u1 = (fhat-u0)/(1+2*alpha[1]*(f-0.25)**2)
        expected = np.fft.irfft(np.array([u0, u1]), n=len(extended), axis=1)[:, half:half+len(x)]
        expected_centers = [np.sum(f*np.abs(u)**2)/np.sum(np.abs(u)**2) for u in (u0, u1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            modes, _, history, info = vmd(x, alpha=alpha, K=2, max_iter=1, return_info=True)
        np.testing.assert_allclose(modes, expected, atol=1e-14)
        np.testing.assert_allclose(history[-1], expected_centers, atol=1e-14)
        self.assertFalse(info["converged"])

    def test_denoising_seed_scaling_and_alias(self):
        x = np.cos(2*np.pi*32*np.arange(512)/512)
        a, _, o, info = vmd(x, K=1, tau=0, init=2, return_info=True)
        self.assertTrue(info["converged"])
        b, _, p = vmd(x*1e-12, K=1, tau=0, init=2)
        np.testing.assert_allclose(a, b/1e-12, atol=1e-10)
        np.testing.assert_allclose(o, p, atol=1e-12)
        c, _, _ = VMD(x, 2000, 0, 1, False, 2, 1e-7)
        np.testing.assert_array_equal(a, c)
        METRICS["vmd_tau_zero"] = info

    def test_dual_ascent_second_step(self):
        x = np.random.default_rng(14).normal(size=31)
        half = len(x)//2
        extended = np.r_[x[:half][::-1], x, x[-half:][::-1]]
        f, target = np.fft.rfftfreq(len(extended)), np.fft.rfft(extended)
        u = np.zeros((2, len(f)), dtype=complex)
        centers, dual = np.array([0., 0.25]), np.zeros(len(f), dtype=complex)
        for _ in range(2):
            previous = u.copy()
            for k in range(2):
                other = u[0] if k == 1 else previous[1]
                u[k] = (target-other+dual/2)/(1+200*(f-centers[k])**2)
                power = np.abs(u[k])**2
                centers[k] = np.sum(f*power)/np.sum(power)
            dual = dual+0.8*(target-u.sum(axis=0))
        expected = np.fft.irfft(u, n=len(extended), axis=1)[:, half:half+len(x)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            modes, _, history = vmd(x, alpha=100, tau=0.8, K=2, max_iter=2)
        np.testing.assert_allclose(modes, expected, atol=1e-13)
        np.testing.assert_allclose(history[-1], centers, atol=1e-13)

    def test_denoising_improves_noisy_tone(self):
        n = 2048
        clean = np.cos(2*np.pi*64*np.arange(n)/1024)
        noisy = clean+0.3*np.random.default_rng(121).normal(size=n)
        modes, _, _, info = vmd(noisy, K=1, tau=0, return_info=True)
        self.assertTrue(info["converged"])
        noisy_error = relative_error(noisy, clean)
        denoised_error = relative_error(modes[0], clean)
        self.assertLess(denoised_error, noisy_error)
        METRICS["vmd_noisy_tone"] = dict(noisy_error=noisy_error, denoised_error=denoised_error)

    def test_invalid_inputs_and_cap_warning(self):
        x = np.arange(32.)
        for kwargs in [dict(K=0), dict(K=1.5), dict(alpha=-1), dict(alpha=[1, 2]),
                       dict(tau=-1), dict(tol=0), dict(fs=0), dict(init=3), dict(max_iter=0)]:
            with self.assertRaises(ValueError):
                vmd(x, **kwargs)
        for bad in [[], [[1, 2, 3]], [0, np.inf, 1], [0, 1j, 0]]:
            with self.assertRaises(ValueError):
                vmd(bad)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, _, info = vmd(x, max_iter=1, return_info=True)
        self.assertTrue(caught)
        self.assertFalse(info["converged"])


class MFDFATests(unittest.TestCase):
    def test_integrated_white_noise_scaling(self):
        x = np.cumsum(np.random.default_rng(707).normal(size=65536))
        h = mfdfa(x, [2.], m=2, max_scale=2048)[2][0]
        self.assertLess(abs(h-1.5), 0.08)
        METRICS["mfdfa_random_walk"] = dict(h2=float(h), expected_scaling_exponent=1.5)

    def test_independent_formula_both_ends_and_q_zero(self):
        x = np.random.default_rng(73).normal(size=1003)
        q = np.array([-5., -2., 0., 2., 5.])
        scales = np.array([17, 31, 53, 101, 173])
        s, fq, h, tau, info = mfdfa(x, q, m=2, scales=scales, return_info=True)
        expected = reference_mfdfa(x, q, scales, 2)
        error = relative_error(fq, expected)
        self.assertLess(error, 1e-11)
        np.testing.assert_array_equal(info["segment_counts"], 2*(len(x)//scales))
        np.testing.assert_allclose(tau, q*h-1, atol=1e-14)
        self.assertEqual(tau[2], -1.)
        METRICS["mfdfa_reference"] = dict(relative_Fq_error=error)

    def test_white_noise_hurst(self):
        q = np.arange(-5., 6.)
        x = np.random.default_rng(42).normal(size=65536)
        scales, fq, h, tau, info = mfdfa(x, q, return_info=True)
        self.assertLess(np.max(np.abs(h-0.5)), 0.07)
        self.assertGreater(np.min(info["r_squared"]), 0.99)
        METRICS["mfdfa_white_noise"] = dict(q=q.tolist(), hq=h.tolist(),
            h2=float(h[q == 2][0]), max_error_to_half=float(np.max(np.abs(h-0.5))),
            r_squared=info["r_squared"].tolist())
        DATA.update(mfdfa_white_scales=scales, mfdfa_white_fq=fq, mfdfa_white_hq=h)

    def test_binomial_cascade_against_theory(self):
        # Eq. (18), length 65536 and a=0.75, as in Fig. 2.
        a, depth = 0.75, 16
        x = np.ones(1)
        for _ in range(depth):
            x = np.ravel(np.column_stack((x*(1-a), x*a)))
        q = np.arange(-5., 6.)
        scales, fq, h, tau = mfdfa(x, q, m=1, scales=2**np.arange(5, 12))
        theoretical_tau = -np.logaddexp(q*np.log(a), q*np.log(1-a))/np.log(2)
        theoretical_h = np.empty_like(q)
        nonzero = q != 0
        theoretical_h[nonzero] = (theoretical_tau[nonzero]+1)/q[nonzero]
        theoretical_h[~nonzero] = -(np.log(a)+np.log(1-a))/(2*np.log(2))
        self.assertLess(np.max(np.abs(h-theoretical_h)), 0.06)
        self.assertGreater(h[0]-h[-1], 0.6)
        alpha, spectrum = multifractal_spectrum(q, tau)
        METRICS["mfdfa_binomial"] = dict(q=q.tolist(), hq=h.tolist(),
            theoretical_hq=theoretical_h.tolist(),
            max_hq_error=float(np.max(np.abs(h-theoretical_h))),
            parameters=dict(a=a, depth=depth, polynomial_order=1, scales=scales.tolist()))
        DATA.update(mfdfa_q=q, mfdfa_binomial_scales=scales, mfdfa_binomial_fq=fq,
                    mfdfa_binomial_hq=h, mfdfa_binomial_theory_hq=theoretical_h,
                    mfdfa_binomial_alpha=alpha, mfdfa_binomial_spectrum=spectrum)

    def test_polynomial_detrending(self):
        x = np.random.default_rng(101).normal(size=8192)
        t = np.linspace(-1, 1, len(x))
        q, scales = [-2., 0., 2.], [32, 64, 128, 256, 512]
        for degree, order in [(1, 2), (2, 3)]:
            a = mfdfa(x, q, m=order, scales=scales)
            b = mfdfa(x+50*t**degree, q, m=order, scales=scales)
            np.testing.assert_allclose(a[1], b[1], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(a[2], b[2], atol=1e-8)

    def test_scale_offset_and_negative_amplitude_invariance(self):
        x = np.random.default_rng(9).normal(size=2048)
        args = dict(scales=[16, 32, 64, 128, 256])
        a = mfdfa(x, [-3, 0, 2, 5], **args)
        for factor in (1e-12, -3., 1e12):
            b = mfdfa(factor*x, [-3, 0, 2, 5], **args)
            np.testing.assert_allclose(a[1], b[1]/abs(factor), rtol=1e-10)
            np.testing.assert_allclose(a[2], b[2], atol=1e-10)
        b = mfdfa(x+1000, [-3, 0, 2, 5], **args)
        np.testing.assert_allclose(a[1], b[1], rtol=1e-9)

    def test_q_near_zero_extreme_moments_and_fit_range(self):
        x = np.random.default_rng(31).normal(size=4096)
        q = [-100., -1e-10, 0., 1e-10, 100.]
        s, fq, h, _, info = mfdfa(x, q, scales=[16, 32, 64, 128, 256],
                                 fit_range=(32, 128), return_info=True)
        self.assertTrue(np.all(np.isfinite(fq)))
        np.testing.assert_allclose(fq[1], fq[2], rtol=1e-9)
        np.testing.assert_allclose(fq[3], fq[2], rtol=1e-9)
        np.testing.assert_array_equal(info["fit_mask"], [False, True, True, True, False])
        expected = np.polyfit(np.log(s[1:4]), np.log(fq[:, 1:4]).T, 1)[0]
        np.testing.assert_allclose(h, expected, atol=1e-12)

    def test_legendre_transform_linear_and_quadratic(self):
        q = np.array([-4., -1., 0., 2., 5.])
        alpha, f = multifractal_spectrum(q, 0.7*q-1)
        np.testing.assert_allclose(alpha, 0.7, atol=1e-14)
        np.testing.assert_allclose(f, 1., atol=1e-14)
        alpha, _ = multifractal_spectrum(q, 0.3*q*q+0.7*q-1)
        np.testing.assert_allclose(alpha, 0.6*q+0.7, atol=1e-14)

    def test_undefined_series_and_invalid_arguments(self):
        for x in [np.zeros(1024), np.ones(1024), np.arange(1024.)]:
            with self.assertRaises(ValueError):
                mfdfa(x, [-2, 0, 2])
        x = np.random.default_rng(1).normal(size=1024)
        for kwargs in [dict(m=0), dict(m=1.5), dict(scales=[4, 16]),
                       dict(scales=[16.5, 32]), dict(scales=[16, 300]),
                       dict(scales=[32]), dict(min_scale=200), dict(n_scales=1),
                       dict(fit_range=[1000, 2000])]:
            with self.assertRaises(ValueError):
                mfdfa(x, [-2, 0, 2], **kwargs)
        for q in [[], [np.nan], [[1, 2]]]:
            with self.assertRaises(ValueError):
                mfdfa(x, q)
        for q in [[1, 1], [2, 1], [1]]:
            with self.assertRaises(ValueError):
                multifractal_spectrum(q, np.ones(len(q)))


def save_artifacts(summary):
    np.savez_compressed(OUT/"numerical_results.npz", **DATA)
    if "mfdfa_q" in DATA:
        np.savetxt(OUT/"mfdfa_binomial.csv", np.column_stack((DATA["mfdfa_q"],
            DATA["mfdfa_binomial_hq"], DATA["mfdfa_binomial_theory_hq"],
            DATA["mfdfa_binomial_alpha"], DATA["mfdfa_binomial_spectrum"])),
            delimiter=",", header="q,hq_estimated,hq_theory,alpha,f_alpha", comments="")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "emd_modes" in DATA and "vmd_modes" in DATA:
        fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
        for col, label in enumerate(("emd", "vmd")):
            t = DATA[label+"_time"]
            axes[0, col].plot(t, DATA[label+"_signal"], lw=0.7)
            axes[0, col].set_title(label.upper()+": three-tone signal (8, 64, 160 Hz)")
            for row, mode in enumerate(DATA[label+"_modes"][:3], 1):
                axes[row, col].plot(t, mode, lw=0.8)
                axes[row, col].set_ylabel("Mode "+str(row))
            axes[3, col].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(OUT/"emd_vmd_decomposition.png", dpi=150)
        plt.close(fig)
    if "mfdfa_q" in DATA and "mfdfa_white_hq" in DATA:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        q = DATA["mfdfa_q"]
        axes[0].plot(q, DATA["mfdfa_white_hq"], "o-", label="White noise")
        axes[0].axhline(0.5, color="k", ls="--", label="Theory H=0.5")
        axes[0].set(xlabel="q", ylabel="h(q)", title="Monofractal check")
        axes[1].plot(q, DATA["mfdfa_binomial_hq"], "o", label="MF-DFA1")
        axes[1].plot(q, DATA["mfdfa_binomial_theory_hq"], "-", label="Eq. (20)")
        axes[1].set(xlabel="q", ylabel="h(q)", title="Binomial cascade a=0.75")
        axes[2].plot(DATA["mfdfa_binomial_alpha"], DATA["mfdfa_binomial_spectrum"], "o-")
        axes[2].set(xlabel="alpha", ylabel="f(alpha)", title="Legendre spectrum")
        for ax in axes:
            ax.grid(alpha=0.25)
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(OUT/"mfdfa_validation.png", dpi=150)
        plt.close(fig)
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    started = time.perf_counter()
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = dict(tests_run=result.testsRun, failures=len(result.failures),
                   errors=len(result.errors), passed=result.wasSuccessful(),
                   elapsed_seconds=time.perf_counter()-started,
                   environment=dict(python=platform.python_version(), numpy=np.__version__,
                                    scipy=scipy.__version__), metrics=METRICS)
    (OUT/"test_log.txt").write_text(log.getvalue(), encoding="utf-8")
    save_artifacts(summary)
    print(log.getvalue())
    print(json.dumps(summary, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
