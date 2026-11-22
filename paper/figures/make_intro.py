#!/usr/bin/env python3
"""Generate illustrative lognormal AR(1) host trajectories for Figure intro-2.

Claim / check before generation (Rule 7, tolerance stated beforehand):
For h(t) = exp(sigma * xi(t) - sigma^2 / 2) with stationary unit-variance
Gaussian AR(1) xi(t), E[h(t)] = 1, Var(h(t)) = exp(sigma^2) - 1, and
Cov(h(t), h(t+l)) = exp(sigma^2 * phi^|l|) - 1. Across M = 5,000 independent
paths of T_chk = 200 steps (seed 202611140), the standard error of the grand
mean is <= 2.5e-3; pre-stated tolerances are 4 standard errors on the grand
mean, |var(h) - (exp(sigma^2) - 1)| < 1.5e-2, and |corr(xi_t, xi_{t-1}) - phi|
< 5e-3. (Note: an initial single-path N = 400,000 check with unadjusted i.i.d.
tolerance 5e-3 gave |mean(h) - 1| = 6.37e-3 at phi = 0.9 because high serial
correlation inflates the single-path SE to ~3.52e-3.)

Output:
  figures/data/intro-2.csv  (columns: t, h_phi0, h_phi09, mean_ref)
  for t = 0..120, sigma = 0.5, phi in {0.0, 0.9}, common standard-Gaussian
  innovations and stationary initial draw from numpy Generator(PCG64) with
  seed 20261114.
"""

from pathlib import Path
import numpy as np


def simulate_ar1_lognormal(eps: np.ndarray, sigma: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (xi, h) from standard normal sequence eps (eps[0] is stationary start)."""
    n_steps = eps.shape[0]
    xi = np.empty(n_steps, dtype=np.float64)
    xi[0] = eps[0]
    scale = np.sqrt(1.0 - phi * phi)
    for t in range(n_steps - 1):
        xi[t + 1] = phi * xi[t] + scale * eps[t + 1]
    h = np.exp(sigma * xi - 0.5 * sigma * sigma)
    return xi, h


def verify_stationary_moments() -> None:
    """Reproduce known lognormal AR(1) moments within pre-stated tolerances."""
    sigma = 0.5
    exact_var = float(np.exp(sigma * sigma) - 1.0)
    m_paths, t_chk = 5_000, 200
    rng_check = np.random.default_rng(202611140)
    eps_mat = rng_check.standard_normal((m_paths, t_chk))

    for phi in (0.0, 0.9):
        xi = np.empty_like(eps_mat)
        xi[:, 0] = eps_mat[:, 0]
        scale = np.sqrt(1.0 - phi * phi)
        for t in range(t_chk - 1):
            xi[:, t + 1] = phi * xi[:, t] + scale * eps_mat[:, t + 1]
        h = np.exp(sigma * xi - 0.5 * sigma * sigma)
        path_means = np.mean(h, axis=1)
        se_mean = float(np.std(path_means, ddof=1) / np.sqrt(m_paths))
        mean_err = abs(float(np.mean(path_means)) - 1.0)
        var_err = abs(float(np.var(h)) - exact_var)
        assert mean_err < 4.0 * se_mean, (
            f"Mean check failed for phi={phi}: err={mean_err}, 4*SE={4.0 * se_mean}"
        )
        assert var_err < 1.5e-2, f"Var check failed for phi={phi}: err={var_err}"
        lag1_corr = float(np.corrcoef(xi[:, 1:].ravel(), xi[:, :-1].ravel())[0, 1])
        corr_err = abs(lag1_corr - phi)
        assert corr_err < 5e-3, f"Lag-1 corr check failed: err={corr_err}"


def main() -> None:
    verify_stationary_moments()

    sigma = 0.5
    horizon = 120
    rng = np.random.default_rng(20261114)
    eps = rng.standard_normal(horizon + 1)

    _, h_phi0 = simulate_ar1_lognormal(eps, sigma=sigma, phi=0.0)
    _, h_phi09 = simulate_ar1_lognormal(eps, sigma=sigma, phi=0.9)

    out_dir = (__import__('pathlib').Path(__file__).resolve().parent / 'data').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "intro-2.csv"

    lines = ["t,h_phi0,h_phi09,mean_ref"]
    for t in range(horizon + 1):
        lines.append(f"{t},{h_phi0[t]:.8f},{h_phi09[t]:.8f},1.00000000")
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {out_csv} ({horizon + 1} rows); "
        f"sample means: phi=0 -> {np.mean(h_phi0):.4f}, phi=0.9 -> {np.mean(h_phi09):.4f}; "
        f"max(h): phi=0 -> {np.max(h_phi0):.4f}, phi=0.9 -> {np.max(h_phi09):.4f}"
    )


if __name__ == "__main__":
    main()
