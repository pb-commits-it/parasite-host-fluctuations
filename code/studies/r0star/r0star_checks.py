#!/usr/bin/env python3
"""Independent, offline numerical checks for the loop reproduction number.

Run: python3 r0star_checks.py > r0star_checks_output.txt

Dependencies: NumPy and SciPy. If a local C compiler is available, a small
population-update kernel is compiled in a temporary directory inside cwd.
The NumPy fallback computes the same recurrence without a compiler.
Randomness comes exclusively from NumPy's seeded PCG64 generator.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.signal import lfilter
from scipy.special import roots_hermitenorm
from scipy.stats import t as student_t


def perron_root(s, g):
    """Positive solution of product(lambda-s) = product(g)."""
    s, g = np.asarray(s), np.asarray(g)
    G = float(np.prod(g))
    return brentq(lambda v: float(np.prod(v - s)) - G,
                  float(max(s)), float(max(s) + 1 + sum(g)), xtol=1e-14)


def log_product_penalty(tau, sigma, phi):
    """K = T * cost, including all sigma^2 factors."""
    tau, sigma, phi = map(np.asarray, (tau, sigma, phi))
    T = float(sum(tau))
    theta = phi / (1 - phi)
    ratio = np.zeros_like(theta)
    for k, th in enumerate(theta):
        if th:
            log_rho = -np.log1p(tau / th).sum()
            ratio[k] = np.exp(log_rho) / (-np.expm1(log_rho))
    return float(np.sum(sigma**2 * ((1 + 2 * theta) / (2 * T) - ratio)))


def scalar_log_transfer_at_threshold(s, sigma, order):
    nodes, weights = roots_hermitenorm(order)
    weights = weights / math.sqrt(2 * math.pi)
    log_s = math.log(s) if s else -np.inf

    def growth(log_q):
        return float(weights @ np.logaddexp(
            log_s, log_q + sigma * nodes - sigma**2 / 2))

    root = brentq(growth, -45, 15, xtol=5e-15, rtol=1e-14)
    return root, growth


def scalar_checks():
    print('\nSCALAR QUADRATURE (birth is the single transfer)')
    print('Exact r = E log(s + q h), q = g/R. Autocorrelation does not enter.')
    print('Gauss-Hermite orders 128 and 256; root solved in log(q).')
    print('err = log(R_exact) - log(R_second_order).')
    print('s       g       R0     sigma    R_exact         R_second        err'
          '              err/sigma^4     r_at_second')
    max_quad_difference = 0.0
    for s, R0 in [(0.0, 4.0), (0.2, 0.25), (0.65, 8.0), (0.85, 0.3)]:
        g = R0 * (1 - s)
        errors = {}
        for sigma in [0.05, 0.10, 0.20, 0.25, 0.30]:
            log_q, growth = scalar_log_transfer_at_threshold(s, sigma, 256)
            log_q_check, _ = scalar_log_transfer_at_threshold(s, sigma, 128)
            max_quad_difference = max(max_quad_difference, abs(log_q - log_q_check))
            exact = math.log(g) - log_q
            approx = math.log(R0) - (1 - s) * sigma**2 / 2
            err = exact - approx
            errors[sigma] = err
            print(f'{s:5.2f} {g:8.4f} {R0:7.2f} {sigma:7.2f} '
                  f'{math.exp(exact):15.10f} {math.exp(approx):15.10f} '
                  f'{err:+16.8e} {err / sigma**4:+15.8e} '
                  f'{growth(math.log(g) - approx):+14.7e}')
        if s:
            p1 = math.log(abs(errors[0.10] / errors[0.05]), 2)
            p2 = math.log(abs(errors[0.20] / errors[0.10]), 2)
            c4 = s * (1 - s) * (1 - 3 * s) / 8
            print(f'  Error orders (0.05->0.10, 0.10->0.20): {p1:.5f}, {p2:.5f}; '
                  f'analytic limiting err/sigma^4 = {c4:+.10f}')
        else:
            print('  s=0: log(R*) = log(g) - sigma^2/2 exactly.')
    print(f'Max |log(q_256)-log(q_128)| = {max_quad_difference:.3e}')
    assert max_quad_difference < 1e-11


KERNEL_SOURCE = r'''
#include <math.h>
#include <stddef.h>
void advance(int nt, int n, int nr, int ns, const double *s,
             const double *g, const double *h, double *x, double *logs) {
    double y[16];
    for (int t=0; t<nt; ++t) {
        for (int c=0; c<ns; ++c) {
            for (int r=0; r<nr; ++r) {
                double total=0.;
                for (int k=0; k<n; ++k) {
                    int prev=(k+n-1)%n;
                    y[k]=s[k]*x[(c*n+k)*nr+r]
                        +g[c*n+prev]*h[((size_t)t*n+prev)*nr+r]
                         *x[(c*n+prev)*nr+r];
                    total+=y[k];
                }
                logs[c*nr+r]+=log(total);
                for (int k=0; k<n; ++k) x[(c*n+k)*nr+r]=y[k]/total;
            }
        }
    }
}
'''


class UpdateKernel:
    def __init__(self, engine):
        self.temp = None
        self.native = None
        compiler = shutil.which('cc') or shutil.which('clang')
        if engine != 'numpy' and compiler:
            self.temp = tempfile.TemporaryDirectory(prefix='.r0star_build_', dir=Path.cwd())
            directory = Path(self.temp.name)
            source = directory / 'update.c'
            library = directory / ('update.dylib' if platform.system() == 'Darwin' else 'update.so')
            source.write_text(KERNEL_SOURCE)
            env = dict(os.environ, TMPDIR=str(directory))
            try:
                subprocess.run([compiler, '-O3', '-std=c99', '-shared', '-fPIC',
                                str(source), '-o', str(library), '-lm'],
                               check=True, capture_output=True, env=env)
                self.library = ctypes.CDLL(str(library))
                self.native = self.library.advance
                ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
                self.native.argtypes = [ctypes.c_int] * 4 + [ptr] * 5
                self.native.restype = None
            except (subprocess.CalledProcessError, OSError) as exc:
                print(f'Native compilation unavailable, using NumPy: {exc}')
                self.temp.cleanup()
                self.temp = None
        self.name = 'local compiled C kernel; NumPy RNG' if self.native else 'NumPy recurrence'

    def advance(self, s, g, h, x, logs):
        nt, n, nr = h.shape
        ns = g.shape[0]
        if self.native:
            assert n <= 16
            self.native(nt, n, nr, ns, s, g, h, x, logs)
        else:
            for ht in h:
                incoming = np.roll(g[:, :, None] * ht[None, :, :] * x, 1, axis=1)
                y = s[None, :, None] * x + incoming
                total = y.sum(axis=1)
                logs += np.log(total)
                x[:] = y / total[:, None, :]

    def close(self):
        if self.temp:
            self.temp.cleanup()


def check_kernel(kernel):
    """Check compiled update against independently expressed NumPy recurrence."""
    rng = np.random.default_rng(9342)
    s = np.array([0.1, 0.45, 0.8])
    g = np.array([[0.6, 0.3, 0.55], [0.61, 0.3, 0.55]])
    h = np.exp(rng.normal(0, 0.25, (73, 3, 7)) - 0.25**2 / 2)
    x = np.full((2, 3, 7), 1 / 3)
    logs = np.zeros((2, 7))
    x_ref, logs_ref = x.copy(), logs.copy()
    kernel.advance(s, g, h, x, logs)
    for ht in h:
        y = s[None, :, None] * x_ref
        for k in range(3):
            y[:, (k + 1) % 3, :] += g[:, k, None] * ht[k, :] * x_ref[:, k, :]
        total = y.sum(axis=1)
        logs_ref += np.log(total)
        x_ref = y / total[:, None, :]
    discrepancy = max(np.max(abs(x - x_ref)), np.max(abs(logs - logs_ref)))
    print(f'Kernel/reference maximum discrepancy: {discrepancy:.3e}')
    assert discrepancy < 2e-12


def similarity_check(s, g, R, rng):
    """One birth at edge zero versus all n births, with identical product."""
    n = len(s)
    first = g.copy()
    first[0] /= R
    all_births = g / R**(1 / n)
    d = np.ones(n)
    for k in range(n - 1):
        d[k + 1] = d[k] * all_births[k] / first[k]
    assert np.isclose(all_births[-1] / first[-1], d[0] / d[-1])
    x = rng.uniform(0.5, 1.5, n)
    xp = d * x
    max_relative_error = 0.
    for _ in range(80):
        h = np.exp(rng.normal(0, 0.3, n) - 0.3**2 / 2)
        x = s * x + np.roll(first * h * x, 1)
        xp = s * xp + np.roll(all_births * h * xp, 1)
        max_relative_error = max(max_relative_error, float(np.max(abs(xp / (d * x) - 1))))
    return max_relative_error


def simulate_case(kernel, s, g, sigma, phi, args, seed):
    n, nr = len(s), args.replicates
    lam = perron_root(s, g)
    tau0 = 1 / (1 - s)
    tau_original = 1 / (1 - s / lam)
    R0 = float(np.prod(g / (1 - s)))
    K = log_product_penalty(tau0, sigma, phi)
    K_wrong = log_product_penalty(tau_original, sigma, phi)
    R_correct, R_wrong = R0 * np.exp(-np.array([K, K_wrong]))
    gs = np.stack([g.copy(), g.copy()])
    gs[:, 0] /= [R_correct, R_wrong]
    print(f'\nRING n={n}; one birth at edge 1 (Python index 0)')
    print(f's={s.tolist()}, g={g.tolist()}')
    print(f'sigma={sigma.tolist()}, phi={phi.tolist()}')
    print(f'R0={R0:.12g}; unscaled mean lambda={lam:.12g}')
    print(f'threshold durations={tau0.tolist()}, T0={tau0.sum():.12g}')
    print(f'unscaled durations={tau_original.tolist()}, T_unscaled={tau_original.sum():.12g}')
    print(f'K_correct={K:.12g}, K_wrong={K_wrong:.12g}')
    print(f'R_second_order={R_correct:.12g}, R_wrong={R_wrong:.12g}')
    print(f'For all {n} births, R_second_order={R_correct**(1/n):.12g}; same effective product.')
    print(f'Leading wrong-duration growth prediction={(K_wrong-K)/tau0.sum():+.8e}')
    sim_error = similarity_check(s, g, R_correct, np.random.default_rng(582 + n))
    print(f'Pathwise diagonal-similarity check, 80 steps: max relative error={sim_error:.3e}')
    assert sim_error < 1e-12

    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, nr)) * sigma[:, None]
    x = np.full((2, n, nr), 1 / n)
    logs = np.zeros((2, nr))
    innovation_sd = sigma * np.sqrt(1 - phi**2)
    halves = []

    def run_segment(length):
        nonlocal z
        for start in range(0, length, args.block):
            nt = min(args.block, length - start)
            noise = rng.normal(size=(nt, n, nr))
            h = np.empty_like(noise)
            for k in range(n):
                zk, _ = lfilter([innovation_sd[k]], [1, -phi[k]], noise[:, k, :],
                                axis=0, zi=(phi[k] * z[k])[None, :])
                z[k] = zk[-1]
                h[:, k, :] = np.exp(zk - sigma[k]**2 / 2)
            kernel.advance(s, gs, h, x, logs)

    run_segment(args.burn_in)
    logs[:] = 0
    n_first = args.steps // 2
    run_segment(n_first)
    halves.append(logs.copy() / n_first)
    first_logs = logs.copy()
    run_segment(args.steps - n_first)
    halves.append((logs - first_logs) / (args.steps - n_first))
    rates = logs / args.steps
    critical = student_t.ppf(0.975, nr - 1)
    for label, rate in zip(['correct durations', 'wrong durations'], rates):
        mean = rate.mean()
        se = rate.std(ddof=1) / math.sqrt(nr)
        lo, hi = mean - critical * se, mean + critical * se
        print(f'{label:18s}: r={mean:+.9e}, SE={se:.9e}, '
              f'95% t CI=[{lo:+.9e}, {hi:+.9e}], r/SE={mean/se:+.4f}')
    paired = rates[1] - rates[0]
    print(f'paired (wrong-correct): mean={paired.mean():+.9e}, '
          f'SE={paired.std(ddof=1)/math.sqrt(nr):.9e}')
    for i, half in enumerate(halves):
        print(f'correct-duration half {i+1}: r={half[0].mean():+.9e}, '
              f'SE={half[0].std(ddof=1)/math.sqrt(nr):.9e}')
    print('SEs use independent replicate averages and retain all temporal correlation.')
    print('Correct and wrong scenarios share environmental paths; their paired SE is also reported.')
    print('A second-order root can have systematic O(sigma^4) growth; a confidence interval'
          ' measures Monte Carlo uncertainty, not truncation error.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=96)
    parser.add_argument('--steps', type=int, default=500_000)
    parser.add_argument('--burn-in', type=int, default=10_000)
    parser.add_argument('--block', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--engine', choices=['auto', 'numpy'], default='auto')
    args = parser.parse_args()
    if args.replicates < 2 or args.steps < 2 or args.block < 1 or args.burn_in < 0:
        parser.error('Need >=2 replicates, >=2 steps, positive block and nonnegative burn-in.')
    start = time.perf_counter()
    print('OFFLINE REPRODUCTION-NUMBER CHECKS')
    print(f'Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}')
    print(f'Seed={args.seed}; replicates={args.replicates}; '
          f'measured steps/replicate={args.steps}; burn-in={args.burn_in}; block={args.block}')
    print('Simulation sample sizes and seeds fixed before examining results.')
    scalar_checks()
    kernel = UpdateKernel(args.engine)
    try:
        print(f'\nSimulation engine: {kernel.name}')
        check_kernel(kernel)
        cases = [
            ([0.2, 0.65], [1.6, 0.7], [0.22, 0.28], [0.35, 0.7]),
            ([0.1, 0.45, 0.8], [0.18, 0.3, 0.55], [0.2, 0.25, 0.3], [0.2, 0.55, 0.8]),
        ]
        seeds = np.random.SeedSequence(args.seed).spawn(len(cases))
        for case, seed in zip(cases, seeds):
            simulate_case(kernel, *(np.array(a, dtype=float) for a in case), args, seed)
    finally:
        kernel.close()
    print(f'\nElapsed seconds: {time.perf_counter()-start:.2f}')


if __name__ == '__main__':
    main()
