#!/usr/bin/env python3
"""Reproduce the three exact-model figures, offline.

Run: python3 -I ../paper/figures/make_exact.py
Dependencies for data regeneration: numpy and mpmath.

Artwork correction, 2026-11-15 (jmb-fig1): mpmath is retained as an explicit
exception to SPEC.md's numpy-only generator rule, not as a claimed permission.
The existing algorithm uses 45-digit confluent hypergeometric U, derivatives
with respect to its parameter, digamma/polygamma and safeguarded roots. Stays
as small as 10**-48 and cancellation in the long-stay derivative need the
existing precision checks; NumPy has no corresponding special functions.
Replacing this with new quadrature would need independent scientific validation.
The artwork rebuild reads the archived CSV files without regenerating them.

Rendering: python3 -I ../paper/figures/make_exact.py --render-pdf
uses tectonic and pdftops on the three TeX sources. It needs neither numpy nor
mpmath and never falls back to an unembedded-font hand-written PDF.
Source: results/exact-gamma-parking_20261114.txt and Propositions R2, R4,
R5, R6 in studies/referee-notes-18/referee_report_18.md.
No Monte Carlo and no imports of previous project numerical routines.
CSV values are numerical evaluations, not interval-certified enclosures.
"""
from pathlib import Path
import csv
import re


FIG = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../code').resolve()
DATA = FIG / "data"


def moments(p, k):
    """R2 and normalized-integral differentiation: m and k Cov(W,T)."""
    m = mp.diff(lambda b: mp.log(mp.hyperu(p, b, k)), 1)
    v = k * mp.diff(
        lambda b: p * mp.hyperu(p + 1, b + 1, k) / mp.hyperu(p, b, k), 1
    )
    assert mp.isfinite(m) and mp.isfinite(v) and m > 0 and v > 0
    return m, v


def threshold(p, s):
    """Safeguarded Newton iteration in log k, using the proved 0<D<ell bracket."""
    p, s = mp.mpf(p), mp.mpf(s)
    ell = mp.log(p) - mp.digamma(p)
    L = -mp.log(s)
    log_k_mean = 2 * mp.log(s) + mp.log(p) - mp.log1p(-s)
    hi = log_k_mean
    lo = hi - ell
    z = (lo + hi) / 2
    for _ in range(100):
        m, v = moments(p, mp.exp(z))
        residual = m - L
        if abs(residual) < mp.mpf("1e-36") * min(1, L):
            D = log_k_mean - z
            assert 0 < D < ell
            return D, z, m, v
        if residual > 0:
            lo = z
        else:
            hi = z
        trial = z + residual / v
        z = trial if lo < trial < hi else (lo + hi) / 2
    raise ArithmeticError(f"threshold failed for p={p}, s={s}")


def write(name, header, rows):
    with (DATA / name).open("w", newline="") as out:
        w = csv.writer(out)
        w.writerow(header)
        for row in rows:
            w.writerow([mp.nstr(x, 17) if isinstance(x, mp.mpf) else x for x in row])


def render_pdfs():
    """Compile the styled vector sources; preserve all CSV inputs."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-I", str(FIG / "build_figs.py"),
                    "--figures", "exact-1", "exact-2", "exact-3"], check=True)


def main():
    DATA.mkdir(exist_ok=True)
    text = (ROOT / "results/exact-gamma-parking_20261114.txt").read_text()
    shapes = [mp.mpf(2), mp.mpf(1), mp.mpf("0.25")]
    stays = [2, 5, 10, 25, 100, 1000]
    saved = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit() and int(parts[0]) in stays:
            saved[int(parts[0])] = [mp.mpf(parts[i].split("/")[0]) for i in [2, 3, 5]]
    assert len(saved) == 6
    rows = []
    for tau in stays:
        ds = [threshold(p, 1 - mp.mpf(1) / tau)[0] for p in shapes]
        for d, old in zip(ds, saved[tau]):
            assert abs(d - old) <= mp.mpf("0.00005000001"), (tau, d, old)
        rows.append([tau, *ds, *saved[tau]])
    write("exact-1.csv", ["tau", "D2", "D1", "D025", "saved2", "saved1", "saved025"], rows)
    write("exact-1-curves.csv", ["tau", "Q2", "Q1", "Q025"],
          [[x, *[1 / (2 * p * (1 + x)) for p in shapes]] for x in np.geomspace(2, 1000, 161)])
    write("exact-1-long.csv", ["tau", "R6p2", "R6p1", "R6p025"],
          [[x, *[1 / (2*p*x) - (12*p+7)/(24*p*p*x*x) for p in shapes]]
           for x in np.geomspace(25, 1000, 121)])
    print("Figure 1: 18 roots agree with archived rounding", flush=True)

    shapes2 = [mp.mpf("0.25"), mp.mpf(1), mp.mpf(10)]
    saved2 = {}
    for line in text.splitlines():
        match = re.match(r"\s*variance (0\.1|1|4): 1 - D/l = (.*)", line)
        if match:
            vals = re.findall(r"(\d+\.\d+) at s", match[2])
            saved2[mp.mpf(1) / mp.mpf(match[1])] = list(map(mp.mpf, vals))
    rows = []
    for i, exponent in enumerate([6, 12, 24, 48]):
        s = mp.mpf(10) ** -exponent
        recovered = []
        for p in shapes2:
            d, *_ = threshold(p, s)
            value = 1 - d / (mp.log(p) - mp.digamma(p))
            assert abs(value - saved2[p][i]) <= mp.mpf("0.00005000001")
            recovered.append(value)
        rows.append([-mp.log(s), *recovered])
    write("exact-2.csv", ["L", "p025", "p1", "p10"], rows)
    curves = []
    for L in np.geomspace(float(6*mp.log(10)), float(48*mp.log(10)), 161):
        curves.append([L, *[mp.polygamma(1, p)/(2*(mp.log(p)-mp.digamma(p))*L)
                           * (1+(mp.digamma(p)+mp.euler)/L) for p in shapes2]])
    write("exact-2-curves.csv", ["L", "p025", "p1", "p10"], curves)
    print("Figure 2: 12 roots agree with archived rounding", flush=True)

    for p, suffix in [(mp.mpf("0.25"), "025"), (mp.mpf(1), "1"),
                      (mp.mpf(10), "10"), (mp.mpf(100), "100")]:
        # Solve only the two endpoints, then use k itself to parametrize the curve.
        z0 = threshold(p, mp.mpf("1e-6"))[1]
        z1 = threshold(p, 1-mp.mpf("1e-6"))[1]
        rows = []
        for j in range(121):
            z = z0 + (z1-z0) * j/120
            k = mp.exp(z)
            m, v = moments(p, k)
            s = mp.exp(-m)
            g = 1/v - 2 - 1/mp.expm1(m)
            assert mp.isfinite(g) and g > 0
            rows.append([s, g, k, m, v])
        write(f"exact-3-p{suffix}.csv", ["s", "g", "kappa", "m", "V"], rows)
        # Independent finite differences of exact threshold roots at an interior stay.
        s = mp.mpf("0.4")
        _, _, m, v = threshold(p, s)
        g = 1/v - 2 - 1/mp.expm1(m)
        h = mp.mpf("1e-6")
        fd = (threshold(p, s*mp.exp(-h))[0]-threshold(p, s*mp.exp(h))[0])/(2*h)
        assert abs(fd-g) < mp.mpf("1e-10") * max(1, abs(g)), (p, fd, g)
        print(f"Figure 3: p={p}, 121 positive derivatives; finite difference agrees", flush=True)

    # Independent normalization checks at the two special shapes in R2.
    k = mp.mpf("0.7")
    assert abs(mp.hyperu(1,1,k)-mp.exp(k)*mp.e1(k)) < mp.mpf("1e-40")
    assert abs(mp.hyperu(mp.mpf("0.5"),1,k)
               -mp.exp(k/2)*mp.besselk(0,k/2)/mp.sqrt(mp.pi)) < mp.mpf("1e-40")
    print("All finiteness, rounding, special-shape, and derivative checks passed.")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--render-pdf"]:
        render_pdfs()
    elif not sys.argv[1:]:
        import mpmath as mp
        import numpy as np
        mp.mp.dps = 45
        main()
        render_pdfs()
    else:
        raise SystemExit("usage: make_exact.py [--render-pdf]")
