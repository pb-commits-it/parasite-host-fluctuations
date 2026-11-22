"""Independent verification of `studies/derivation.md` (Sections 2-8).

Written from scratch by secondary (2026-11-13) without importing `studies/check.py` or `engine/ring.py`.
Checks:
  1. Sections 2-3: equivalence of eq. (1), Fourier sum eq. (6), lag trace sum eq. (5),
     and Round-1 Tuljapurkar spectral form; cancellation at n = 1 and s = 0 (n = 1).
  2. Section 5: unimodality via F(x) = [x / (2 sinh(x/2))]^2, near-threshold boundary tests
     around phi_thr = (s + gbar)/(3s + gbar) (especially 1 < n_c < 2 where continuous peak > 1
     can still have C_2 <= C_1!), and phi -> 1 asymptotics of n_c.
  3. Section 6 (eq. 9): independent Tuljapurkar subdominant-eigenmode sum and independent
     antithetic/control-variate Monte Carlo simulator tested on five rings primary did not use:
       - U1: strongly unequal 4-stage ring (s from 0.02 to 0.96, g spanning 60x)
       - U2: strongly unequal 6-stage ring
       - U3: 3-stage ring with one stage at s_k = 0
       - U4: 4-stage minimal primitive ring with THREE stages at s_k = 0 and one s_k > 0
       - U5: 3-stage imprimitive unequal ring with ALL s_k = 0 (verifying exact sigma^2 / 2)
  4. Section 8 (eq. 10): exact scalar integral check at n = 1 (including sign change at q = 1/3),
     comparison with primary paired sigma-sweeps at phi = 0, and consistency check against both
     phi = 0 groups in tertiary's `reports/R1a-large-ring-gap.md`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
from scipy import integrate


def primary_eq1_coeff(n: int, s: float, gbar: float, phi: float) -> float:
    """primary eq. (1) second-order coefficient cost / sigma^2 for s > 0 (or n = 1)."""
    lam0 = s + gbar
    p = s / lam0
    q = gbar / lam0
    if phi == 0.0:
        return (q * q) / (2.0 * n)
    alpha = math.log1p((1.0 - phi) / (phi * q))
    return (q * q * (1.0 + phi)) / (2.0 * n * (1.0 - phi)) - q / math.expm1(n * alpha)


def secondary_round1_coeff(n: int, s: float, gbar: float, phi: float) -> float:
    """secondary round-1 closed form (`r1-dobson.secondary.md` line 130)."""
    lam0 = s + gbar
    e = gbar / lam0
    if n == 1 or phi == 0.0:
        return 0.5 * (e * e) / n
    a = lam0 - phi * s
    b = phi * gbar
    tail = n * lam0 * (phi ** n) * (gbar ** (n - 1)) / (a ** n - b ** n)
    return (e * e / n) * (0.5 + phi / (1.0 - phi) - tail)


def tuljapurkar_eigenmode_general(s: np.ndarray, g: np.ndarray, phi: float) -> float:
    """secondary's general second-order coefficient via full spectral decomposition of A_0
    (Tuljapurkar 1990 Chs. 4 & 7, independent of primary's resolvent Q / T formulation):
      C = (1/2) sum_k e_k^2 - sum_{m != perron} sum_{k} (w_0[k+1] g_k v_m[k]) (w_m[k+1] g_k v_0[k]) / lam0^2
                              * phi / (1 - phi * (lam_m / lam0)).
    """
    s = np.asarray(s, dtype=float)
    g = np.asarray(g, dtype=float)
    n = len(s)
    if np.all(s == 0.0):
        return 0.5
    A = np.diag(s).astype(complex)
    for k in range(n):
        A[(k + 1) % n, k] += g[k]
    evals, VR = np.linalg.eig(A)
    idx0 = int(np.argmax(evals.real))
    lam0 = evals[idx0].real
    # Inverse of VR gives rows = left eigenvectors normalized so W_rows @ VR = I!
    W_rows = np.linalg.inv(VR)

    v0 = VR[:, idx0]
    w0 = W_rows[idx0, :]
    # Elasticities e_k = w0[k+1] * g_k * v0[k] / lam0 (invariant to scalar rescaling of v0, w0 since w0 @ v0 = 1)
    nxt = (np.arange(n) + 1) % n
    e_k = (w0[nxt] * g * v0).real / lam0
    white = 0.5 * float(np.sum(e_k ** 2))
    if phi == 0.0 or n == 1:
        return white

    auto = 0.0 + 0.0j
    for m in range(n):
        if m == idx0:
            continue
        vm = VR[:, m]
        wm = W_rows[m, :]
        mu_m = evals[m] / lam0
        # Stage-k coupling between mode 0 and mode m:
        # (w0^T B_k vm) * (wm^T B_k v0) where B_k = (g_k / lam0) e_{k+1} e_k^T
        coupling_k = (w0[nxt] * g * vm / lam0) * (wm[nxt] * g * v0 / lam0)
        auto += np.sum(coupling_k) * (phi / (1.0 - phi * mu_m))
    assert abs(auto.imag) < 1e-13, f"Unexpected imaginary part {auto.imag}"
    return white - float(auto.real)


def primary_eq9_coeff(s: np.ndarray, g: np.ndarray, phi: float) -> float:
    """Direct evaluation of primary eq. (9) as written in Section 6."""
    s = np.asarray(s, dtype=float)
    g = np.asarray(g, dtype=float)
    n = len(s)
    if np.all(s == 0.0):
        return 0.5
    A0 = np.diag(s)
    for k in range(n):
        A0[(k + 1) % n, k] += g[k]
    ev, V = np.linalg.eig(A0)
    i0 = int(np.argmax(ev.real))
    lam = float(ev[i0].real)
    v = np.abs(V[:, i0].real)
    v /= v.sum()
    evl, W = np.linalg.eig(A0.T)
    j0 = int(np.argmin(np.abs(evl - lam)))
    w = np.abs(W[:, j0].real)
    w /= float(w @ v)

    M = A0 / lam
    Q = np.eye(n) - np.outer(v, w)
    T = M - np.outer(v, w)
    res = np.linalg.inv(np.eye(n) - phi * T)
    c_sq = 0.0
    lag_sum = 0.0
    for k in range(n):
        Bk = np.zeros((n, n))
        Bk[(k + 1) % n, k] = g[k] / lam
        ck = float(w @ Bk @ v)
        c_sq += ck * ck
        lag_sum += float(w @ Bk @ res @ Q @ Bk @ v)
    return 0.5 * c_sq - phi * lag_sum


def independent_mc_cost(
    s: np.ndarray,
    g: np.ndarray,
    sigma: float,
    phi: float,
    steps: int = 120000,
    burn: int = 4000,
    reps: int = 400,
    seed: int = 20261113,
) -> tuple[float, float, float]:
    """Independent vectorized Monte Carlo with antithetic (+z / -z) pairing and first-order
    elasticity control variate. Returns (mean_cost, se_cost, max_share_mean)."""
    rng = np.random.default_rng(seed)
    s = np.asarray(s, dtype=float)
    g = np.asarray(g, dtype=float)
    n = len(s)
    A0 = np.diag(s)
    for k in range(n):
        A0[(k + 1) % n, k] += g[k]
    ev, V = np.linalg.eig(A0)
    i0 = int(np.argmax(ev.real))
    lam0 = float(ev[i0].real)
    v = np.abs(V[:, i0].real)
    v /= v.sum()
    evl, W = np.linalg.eig(A0.T)
    w = np.abs(W[:, int(np.argmin(np.abs(evl - lam0)))].real)
    w /= float(w @ v)
    e_g = np.array([w[(k + 1) % n] * g[k] * v[k] / lam0 for k in range(n)])

    z = rng.normal(0.0, sigma, size=(reps, n))
    innov = sigma * math.sqrt(max(0.0, 1.0 - phi * phi))

    xp = np.broadcast_to(v, (reps, n)).copy()
    xm = np.broadcast_to(v, (reps, n)).copy()
    acc = np.zeros(reps, dtype=float)
    max_share_acc = 0.0

    for t in range(steps + burn):
        hp = np.exp(z - 0.5 * sigma * sigma)
        hm = np.exp(-z - 0.5 * sigma * sigma)
        xp = xp * s[None, :] + np.roll(xp * (g[None, :] * hp), 1, axis=1)
        xm = xm * s[None, :] + np.roll(xm * (g[None, :] * hm), 1, axis=1)
        tot_p = xp.sum(axis=1)
        tot_m = xm.sum(axis=1)
        xp /= tot_p[:, None]
        xm /= tot_m[:, None]
        if t >= burn:
            cv_p = (hp - 1.0) @ e_g
            cv_m = (hm - 1.0) @ e_g
            step_growth = 0.5 * ((np.log(tot_p) - cv_p) + (np.log(tot_m) - cv_m))
            acc += math.log(lam0) - step_growth
            if t == steps + burn - 1:
                max_share_acc = float(0.5 * (xp.max(axis=1).mean() + xm.max(axis=1).mean()))
        z = phi * z + innov * rng.normal(0.0, 1.0, size=(reps, n))

    rep_costs = acc / steps
    return float(rep_costs.mean()), float(rep_costs.std(ddof=1) / math.sqrt(reps)), max_share_acc


def check_sections_2_3_5() -> None:
    print("=== CHECK 1: Sections 2, 3, and 5 (Closed form, n=1 cancellation, near-threshold unimodality) ===")
    worst_c1_c2 = 0.0
    worst_c1_c4 = 0.0
    worst_c1_c3 = 0.0
    for n in (1, 2, 3, 5, 16, 64):
        for s in (1e-4, 0.2, 2.0 / 3.0, 0.98):
            for gbar in (0.05, 0.4, 2.5):
                for phi in (0.0, 0.01, 0.5, 0.9, 0.995):
                    c1 = primary_eq1_coeff(n, s, gbar, phi)
                    c2 = secondary_round1_coeff(n, s, gbar, phi)
                    c3 = tuljapurkar_eigenmode_general(np.full(n, s), np.full(n, gbar), phi)
                    c4 = primary_eq9_coeff(np.full(n, s), np.full(n, gbar), phi)
                    worst_c1_c2 = max(worst_c1_c2, abs(c1 - c2))
                    worst_c1_c4 = max(worst_c1_c4, abs(c1 - c4))
                    worst_c1_c3 = max(worst_c1_c3, abs(c1 - c3))
    assert worst_c1_c2 < 1e-11 and worst_c1_c4 < 1e-11 and worst_c1_c3 < 1e-11, (
        worst_c1_c2, worst_c1_c4, worst_c1_c3
    )
    # Check n=1, s=0 cancellation in eq. (1)
    for phi in (0.0, 0.3, 0.9, 0.99):
        assert abs(primary_eq1_coeff(1, 0.0, 1.3, phi) - 0.5) < 1e-12
    print(f"  [PASS] Eq. (1) vs secondary Round-1 max diff {worst_c1_c2:.2e}; "
          f"vs eq. (9) {worst_c1_c4:.2e}; vs general eig sum {worst_c1_c3:.2e}.")

    # Section 5 subtle test: when 1 < n_c < 2 AND phi <= phi_thr, continuous peak n_c > 1 exists,
    # YET C_2 <= C_1, and unimodality forces C_1 >= C_2 > C_3 > ...!
    subtle_cases = 0
    for q in np.linspace(0.05, 0.95, 37):
        s, gbar = 1.0 - q, q
        phi_thr = 1.0 / (3.0 - 2.0 * q)  # = (s + gbar)/(3s + gbar)
        # Check exact tie at phi = phi_thr:
        c1_tie = primary_eq1_coeff(1, s, gbar, phi_thr)
        c2_tie = primary_eq1_coeff(2, s, gbar, phi_thr)
        c3_tie = primary_eq1_coeff(3, s, gbar, phi_thr)
        assert abs(c1_tie - c2_tie) < 1e-13 and c2_tie > c3_tie
        # Check slightly below threshold: continuous n_c is ALREADY > 1!
        for delta in (-1e-2, -1e-4, -1e-7, +1e-7, +1e-4, +1e-2):
            phi = phi_thr + delta
            if not (0.0 < phi < 1.0):
                continue
            alpha = math.log(1.0 + (1.0 - phi) / (phi * q))
            K = q * q * (1.0 + phi) / (2.0 * (1.0 - phi))
            D = K * alpha / q
            # Solve F(x_*) = D via bisection if D < 1
            if D < 1.0:
                lo, hi = 1e-12, 100.0
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    val = (mid / (2.0 * math.sinh(0.5 * mid))) ** 2
                    if val > D:
                        lo = mid
                    else:
                        hi = mid
                n_c = 0.5 * (lo + hi) / alpha
                if delta < 0 and n_c > 1.0:
                    subtle_cases += 1
            vals = [primary_eq1_coeff(k, s, gbar, phi) for k in range(1, 150)]
            peak_n = 1 + int(np.argmax(vals))
            assert (peak_n > 1) == (delta > 0), f"Threshold failed at q={q}, delta={delta}"
    print(f"  [PASS] Section 5 integer threshold verified across exact ties and +/-1e-7 perturbations "
          f"(including {subtle_cases} cases with continuous peak 1 < n_c < 2 where C_1 > C_2 > C_3).")


def check_section_6_unequal() -> None:
    print("\n=== CHECK 2: Section 6 (Unequal stages eq. 9 on new rings, including s_k = 0 stages) ===")
    rings = [
        ("U1_strong4", np.array([0.02, 0.96, 0.15, 0.88]), np.array([2.50, 0.04, 1.30, 0.09])),
        ("U2_strong6", np.array([0.95, 0.05, 0.80, 0.10, 0.60, 0.25]), np.array([0.05, 1.80, 0.15, 1.40, 0.30, 0.90])),
        ("U3_one_s0", np.array([0.0, 0.75, 0.35]), np.array([1.10, 0.25, 0.65])),
        ("U4_min_prim", np.array([0.60, 0.0, 0.0, 0.0]), np.array([0.50, 1.20, 0.80, 1.10])),
    ]
    sigma = 0.025
    for idx, (name, s, g) in enumerate(rings):
        for phi in (0.0, 0.85 if name == "U1_strong4" else 0.9):
            c_independent = primary_eq9_coeff(s, g, phi)
            c_tulj = tuljapurkar_eigenmode_general(s, g, phi)
            assert abs(c_independent - c_tulj) < 1e-13, f"Algebra mismatch on {name}: {c_independent} vs {c_tulj}"
            pred = c_independent * sigma * sigma
            mc, se, max_sh = independent_mc_cost(
                s, g, sigma, phi, steps=100000, burn=5000, reps=320, seed=5000 + 10 * idx + int(10 * phi)
            )
            ratio = mc / pred
            z = (mc - pred) / se
            print(f"  {name:11s} phi={phi:4.2f}: eq9_coeff={c_independent:.8f} (diff vs Tulj={abs(c_independent-c_tulj):.1e})  "
                  f"MC/eq9={ratio:.5f} +/- {se/pred:.5f} (z={z:+.2f})")
            assert abs(ratio - 1.0) < max(3.5 * se / pred, 0.01), f"MC mismatch on {name} phi={phi}: ratio={ratio}"

    # U5: ALL s_k = 0 with unequal g_k -> exact cost = sigma^2 / 2 for any sigma, phi!
    s_zero = np.array([0.0, 0.0, 0.0])
    g_uneq = np.array([1.5, 0.7, 1.2])
    sigma_u5 = 0.20
    for phi in (0.0, 0.9):
        # For a single initial cohort x(0) = e_0, sample mean converges rapidly to sigma^2/2;
        # for uniform x(0), max of 3 walks converges to sigma^2/2 as O(1/sqrt(T)).
        mc_u5, se_u5, max_sh = independent_mc_cost(
            s_zero, g_uneq, sigma_u5, phi, steps=150000, burn=5000, reps=400, seed=9000 + int(10 * phi)
        )
        exact_u5 = 0.5 * sigma_u5 * sigma_u5
        print(f"  U5_all_s0   phi={phi:4.2f}: MC_cost={mc_u5:.6f} +/- {se_u5:.6f} vs exact sigma^2/2={exact_u5:.6f} "
              f"(max_stage_share={max_sh:.3f})")


def primary_eq10_H(n: int, p: float, q: float) -> float:
    """primary eq. (10) fourth-order coefficient H_n at phi = 0."""
    return (
        (q ** 2) / (4.0 * n)
        + (q ** 3) * (n * n - 1) / (24.0 * p * n * n)
        - (q ** 3) / (n * n)
        + 0.75 * (q ** 4) / (n * n)
    )


def exact_scalar_cost(s: float, gbar: float, sigma: float) -> float:
    lam0 = s + gbar
    f = lambda z: math.log(s + gbar * math.exp(sigma * z - 0.5 * sigma * sigma)) * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    val, _ = integrate.quad(f, -13.0, 13.0, epsabs=1e-15, epsrel=1e-14, limit=500)
    return math.log(lam0) - val


def check_section_8() -> None:
    print("\n=== CHECK 3: Section 8 (Fourth-order eq. 10 at phi = 0 vs exact n = 1 integral & tertiary R1a) ===")
    # 1. Check n = 1 across q below, at, and above q = 1/3 (where H_1 changes sign!)
    for q in (0.15, 1.0 / 3.0, 0.375, 0.75, 0.953125):
        p = 1.0 - q
        h1_pred = primary_eq10_H(1, p, q)
        assert abs(h1_pred - 0.25 * q * q * p * (1.0 - 3.0 * q)) < 1e-16
        # Extract H_1 from exact Gauss-Hermite integral via 2-step Richardson eliminating sigma^6 and sigma^8
        xh, wh = np.polynomial.hermite_e.hermegauss(120)
        def sc_cost(sig: float) -> float:
            vals = np.log1p(q * np.expm1(sig * xh - 0.5 * sig * sig))
            return -float((wh @ vals) / math.sqrt(2.0 * math.pi))
        c0 = 0.5 * q * q
        g1 = (sc_cost(0.02) / (0.02 ** 2) - c0) / (0.02 ** 2)
        g2 = (sc_cost(0.01) / (0.01 ** 2) - c0) / (0.01 ** 2)
        g3 = (sc_cost(0.005) / (0.005 ** 2) - c0) / (0.005 ** 2)
        r12 = (4.0 * g2 - g1) / 3.0
        r23 = (4.0 * g3 - g2) / 3.0
        h1_num = (16.0 * r23 - r12) / 15.0
        print(f"  n=1, q={q:.6f}: H1_eq10={h1_pred:+.12f}  H1_exact_integral={h1_num:+.12f}  diff={abs(h1_pred-h1_num):.2e}")
        assert abs(h1_pred - h1_num) < 1e-9

    # 2. Compare eq. (10) against primary's paired differences in `studies/results.txt` at phi = 0 (n = 2, 4, 8, 16, 32):
    p_std, q_std = 5.0 / 8.0, 3.0 / 8.0
    primary_paired_025_0125 = {
        2: (5.0326989e-06, 7.24e-09),
        4: (4.5464015e-06, 4.32e-09),
        8: (3.4064734e-06, 2.12e-09),
        16: (2.6017569e-06, 2.59e-09),
        32: (2.1478810e-06, 2.95e-09),
    }
    denom = 0.025 ** 2 - 0.0125 ** 2
    print("  Comparison of eq. (10) with primary paired differences [C(0.025)-C(0.0125)]/(0.025^2-0.0125^2):")
    for n, (diff_val, diff_se) in primary_paired_025_0125.items():
        h_pred = primary_eq10_H(n, p_std, q_std)
        h_obs = diff_val / denom
        h_se = diff_se / denom
        c_rel = h_pred / (0.5 * q_std * q_std / n)
        print(f"    n={n:2d}: H_eq10={h_pred:.7e}  H_paired_MC={h_obs:.7e} +/- {h_se:.2e}  "
              f"excess/sig^2 = H/C = {c_rel:.4f} (at sig=0.15: {100*c_rel*0.15**2:+.3f}%)")

    # 3. Consistency check against tertiary's `results/R1a-large-ring-gap_20261113.json` for BOTH phi = 0 groups!
    r1a_path = (__import__('pathlib').Path(__file__).resolve().parent / '../results/R1a-large-ring-gap_20261113.json').resolve()
    rows = json.loads(r1a_path.read_text())
    print("\n  Consistency of tertiary's R1a phi = 0 measurements with primary eq. (10):")
    for group_label, n_val, s_val in [("main (n=32, s=2/3)", 32, 2.0 / 3.0), ("H4 little staying (n=4, s=0.05)", 4, 0.05)]:
        lam0 = 2.0 / 3.0 + 0.4
        p = s_val / lam0
        q = 1.0 - p
        c2 = 0.5 * q * q / n_val
        h4 = primary_eq10_H(n_val, p, q)
        c_pred = h4 / c2  # predicted limit of excess / sigma^2 as sigma -> 0
        sub = [r for r in rows if r["n"] == n_val and r["phi"] == 0.0 and abs(r["s"] - s_val) < 1e-6]
        sub.sort(key=lambda r: -r["sigma"])
        print(f"    {group_label}: eq. (10) predicts H_n/C_n = {c_pred:.4f}")
        for r in sub:
            sig = r["sigma"]
            ex_pred = c_pred * sig * sig
            ex_obs = r["ratio"] - 1.0
            ex_se = r["ratio_se"]
            z = (ex_obs - ex_pred) / ex_se
            print(f"      sigma={sig:5.3f}: obs_excess={100*ex_obs:6.3f}% +/- {100*ex_se:5.3f}%  "
                  f"eq10_excess={100*ex_pred:6.3f}%  (obs ex/sig^2={(ex_obs/sig**2):5.2f} vs {c_pred:5.2f}, z={z:+.2f})")


if __name__ == "__main__":
    check_sections_2_3_5()
    check_section_6_unequal()
    check_section_8()
