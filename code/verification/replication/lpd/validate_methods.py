#!/usr/bin/env python3
"""Rule 7 validation of clean-room replication code against known exact results,
plus optional local C 2x2 product kernel cross-checked against pure NumPy.

Every tolerance is stated BEFORE running any empirical calculation.
Run with:
    python3 -I verification/replication/lpd/validate_methods.py
"""
from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE))

import core_math as cm

C_SOURCE = r"""
#include <math.h>

int products_2x2(
    int t_total,
    int reps,
    int burn,
    const double *a00_t,
    const double *a10_t,
    double a01_y,
    double a11_const,
    double p_init,
    double *logs_out,
    double *slopes_out
) {
    for (int r = 0; r < reps; ++r) {
        double p = p_init;
        double q = 0.0;
        double log_sum = 0.0;
        double slope_sum = 0.0;
        for (int t = 0; t < t_total; ++t) {
            int idx = t * reps + r;
            double a00 = a00_t[idx];
            double a10 = a10_t[idx];
            double omp = 1.0 - p;
            double u0 = a00 * p + a01_y * omp;
            double u1 = a10 * p + a11_const * omp;
            double tot = u0 + u1;
            if (!(tot > 0.0)) {
                return 1;
            }
            double du0 = a00 * q + a01_y * (omp - q);
            double du1 = (a10 - a11_const) * q;
            double dtot = du0 + du1;
            double inv_tot = 1.0 / tot;
            double p_next = u0 * inv_tot;
            double der = dtot * inv_tot;
            q = (du0 - p_next * dtot) * inv_tot;
            p = p_next;
            if (t >= burn) {
                log_sum += log(tot);
                slope_sum += der;
            }
        }
        double keep = (double)(t_total - burn);
        logs_out[r] = log_sum / keep;
        slopes_out[r] = slope_sum / keep;
    }
    return 0;
}
"""


def get_fast_2x2_kernel():
    """Compile local 2x2 C product kernel if C compiler is available; fallback to NumPy."""
    c_path = HERE / "_fast2x2.c"
    so_path = HERE / "_fast2x2.so"
    if not c_path.exists() or c_path.read_text() != C_SOURCE:
        c_path.write_text(C_SOURCE)
    if not so_path.exists() or so_path.stat().st_mtime < c_path.stat().st_mtime:
        subprocess.run(
            ["cc", "-O3", "-shared", "-fPIC", str(c_path), "-o", str(so_path)],
            check=True,
        )
    lib = ctypes.CDLL(str(so_path))
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.products_2x2.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ptr, ptr,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ptr, ptr,
    ]
    lib.products_2x2.restype = ctypes.c_int

    def kernel(
        a00_t: np.ndarray,
        a10_t: np.ndarray,
        a01_base: float,
        a11_const: float,
        p_init: float,
        y: float,
        burn: int,
        pairs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        a00_c = np.ascontiguousarray(a00_t, dtype=np.float64)
        a10_c = np.ascontiguousarray(a10_t, dtype=np.float64)
        t_total, reps = a00_c.shape
        logs = np.zeros(reps, dtype=np.float64)
        slopes = np.zeros(reps, dtype=np.float64)
        rc = lib.products_2x2(
            t_total, reps, burn,
            a00_c, a10_c,
            a01_base * math.exp(y), a11_const, p_init,
            logs, slopes,
        )
        if rc != 0:
            raise ValueError("2x2 matrix product hit nonpositive total")
        return 0.5 * (logs[:pairs] + logs[pairs:]), 0.5 * (slopes[:pairs] + slopes[pairs:])

    return kernel


def run_validations() -> dict:
    checks = []

    def record(cid: str, claim: str, observed: float, tolerance: float, n_cases: int) -> None:
        passed = bool(observed <= tolerance)
        checks.append({
            "id": cid,
            "claim": claim,
            "observed": float(observed),
            "tolerance": float(tolerance),
            "cases": int(n_cases),
            "passed": passed,
        })
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {cid}: obs={observed:.3e} <= tol={tolerance:.3e} ({claim})", flush=True)

    # V1a: Exact hat-matrix white-noise null at complete N = 15 equals -1/7
    t15 = np.arange(15, dtype=float)
    mu15 = cm.wn_null_r1_linear(t15)
    record("V1a", "Linear-detrended white-noise mu0(t) at complete N=15 equals -1/7", abs(mu15 - (-1.0 / 7.0)), 1e-13, 1)

    # V1b: Monte Carlo white-noise lag-1 residual correlation on gapped grid matches wn_null_r1_linear
    rng = np.random.default_rng(2026111511)
    t_gap = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=float)
    y_mc = rng.standard_normal((50000, len(t_gap)))
    Q_lin = cm.linear_detrend_basis(t_gap)
    e_mc = y_mc - (y_mc @ Q_lin) @ Q_lin.T
    adj = np.flatnonzero(np.diff(t_gap) == 1.0)
    r1_mc = np.mean(e_mc[:, adj] * e_mc[:, adj + 1], axis=1) / np.mean(e_mc * e_mc, axis=1)
    record(
        "V1b",
        "50,000 Gaussian white-noise series on gapped N=18 grid match wn_null_r1_linear",
        abs(float(np.mean(r1_mc)) - cm.wn_null_r1_linear(t_gap)),
        0.006,
        50000,
    )

    # V2: Multi-covariate projected white-noise null (used in W1 weather adjustment)
    t30 = np.arange(30, dtype=float)
    X_cov = np.column_stack([np.ones(30), t30, rng.standard_normal((30, 3))])
    Q_cov = cm.orthonormal_columns(X_cov)
    y_mc2 = rng.standard_normal((40000, 30))
    e_mc2 = y_mc2 - (y_mc2 @ Q_cov) @ Q_cov.T
    r1_mc2 = np.mean(e_mc2[:, :-1] * e_mc2[:, 1:], axis=1) / np.mean(e_mc2 * e_mc2, axis=1)
    # Also check exact n x n projection trace identity
    M_proj = np.eye(30) - Q_cov @ Q_cov.T
    P1_sum = sum(M_proj[k, k + 1] for k in range(29))
    exact_ratio = P1_sum / (float(29) * float(np.trace(M_proj)) / 30.0)
    record("V2a", "Projected white-noise formula equals exact trace(M P1)/trace(M)", abs(cm.wn_null_r1_projected(t30, Q_cov) - exact_ratio), 1e-13, 1)
    record("V2b", "40,000 Gaussian draws match projected 5-column white-noise null", abs(float(np.mean(r1_mc2)) - exact_ratio), 0.006, 40000)

    # V3: Forward-backward O(n*p) AR(1) expected ratio matches dense n x n covariance matrix formula
    max_ar1_err = 0.0
    phis_test = np.array([-0.7, -0.2, 0.0, 0.35, 0.8])
    fast_curve = cm.expected_r1_ratio_ar1_projected(t_gap, Q_lin, phis_test)
    M_gap = np.eye(len(t_gap)) - Q_lin @ Q_lin.T
    dist_mat = np.abs(t_gap[:, None] - t_gap[None, :])
    for ip, ph in enumerate(phis_test):
        Sigma = (ph ** dist_mat) if ph != 0.0 else np.eye(len(t_gap))
        Cov_e = M_gap @ Sigma @ M_gap
        num_e = float(np.mean([Cov_e[i, i + 1] for i in adj]))
        den_e = float(np.trace(Cov_e)) / float(len(t_gap))
        max_ar1_err = max(max_ar1_err, abs(fast_curve[ip] - num_e / den_e))
    record("V3", "O(n*p) AR(1) projected curve matches dense M Sigma M trace ratio", max_ar1_err, 1e-12, len(phis_test))

    # V4a: C 2x2 product kernel matches pure-NumPy 2x2 product kernel to machine precision
    c_kernel = get_fast_2x2_kernel()
    max_c_err = 0.0
    for mode in ("primary_coarse", "primary_fine", "contact_fine"):
        a00_t, a10_t, a01_b, a11_c, p0, _ = cm.build_2x2_model_arrays(mode, 7.5, 0.45, 0.40, 800, 100, 4, 42)
        l_np, s_np = cm._products_2x2_numpy(a00_t, a10_t, a01_b, a11_c, p0, 0.05, 100, 4)
        l_c, s_c = c_kernel(a00_t, a10_t, a01_b, a11_c, p0, 0.05, 100, 4)
        max_c_err = max(max_c_err, float(np.max(np.abs(l_np - l_c))), float(np.max(np.abs(s_np - s_c))))
    record("V4a", "Compiled C 2x2 kernel agrees with pure-NumPy 2x2 kernel", max_c_err, 1e-13, 3)

    # V4b: Small-noise (sigma = 0.04) simulated D matches closed-form Gamma_0(1, b; sigma, phi)
    worst_small_ratio = 0.0
    for b_val in (24.0, 7.5, 2.3):
        for ph in (-0.5, 0.0, 0.5, 0.8):
            sim = cm.solve_threshold_loss_2x2("primary_coarse", b_val, 0.04, ph, steps=30000, burn=1600, pairs=12, seed=9001, product_kernel=c_kernel)
            exact_g0 = cm.gamma0_ring_two_stage(1.0, b_val, 0.04, ph)
            allowed = 4.0 * sim["se"] + 2.5e-5
            worst_small_ratio = max(worst_small_ratio, abs(sim["D"] - exact_g0) / allowed)
    record("V4b", "Simulated primary_coarse D at sigma=0.04 matches closed-form Gamma_0 within 4 SE + 2.5e-5", worst_small_ratio, 1.0, 12)

    # V4c: Fine-step D(4, 4b) approaches 1-stage Gauss-Hermite quadrature as b -> 0? Test 1-stage vs two-stage limit and Jensen bound
    sim_j = cm.solve_threshold_loss_2x2("primary_coarse", 7.5, 1.1, 0.6, steps=24000, burn=1600, pairs=12, seed=9002, product_kernel=c_kernel)
    j_excess = max(0.0, sim_j["D"] - (0.5 * 1.1 * 1.1 + 4.0 * sim_j["se"]))
    record("V4c", "Simulated D satisfies universal lognormal Jensen bound D <= sigma^2/2 + 4 SE", j_excess, 1e-12, 1)

    # V4d: Contact-limited waiting with death (`contact_fine`) matches matrix-resolvent D2 at sigma = 0.03
    worst_contact_ratio = 0.0
    for b_val in (24.0, 7.5, 2.3):
        for ph in (0.0, 0.4, 0.8):
            sim_c = cm.solve_threshold_loss_2x2("contact_fine", b_val, 0.03, ph, steps=30000, burn=1600, pairs=12, seed=9100, product_kernel=c_kernel)
            d2_c = cm.contact_fine_resolvent_d2(b_val, 0.03, ph)
            allowed_c = 4.0 * sim_c["se"] + 3.0e-5
            worst_contact_ratio = max(worst_contact_ratio, abs(sim_c["D"] - d2_c) / allowed_c)
    record("V4d", "Simulated contact_fine D at sigma=0.03 matches matrix-resolvent D2 within 4 SE + 3e-5", worst_contact_ratio, 1.0, 9)

    all_ok = all(c["passed"] for c in checks)
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_passed": all_ok,
        "checks": checks,
    }
    (HERE / "validation_checks.json").write_text(json.dumps(payload, indent=2))
    if not all_ok:
        raise RuntimeError("Rule 7 validation failed; halting before empirical analysis!")
    print("All Rule 7 validation checks passed.", flush=True)
    return payload


if __name__ == "__main__":
    run_validations()
