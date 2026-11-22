#!/usr/bin/env python3
"""Rule 7 pre-validation tests for `pilot_lib.py` prior to running the 18-cell pilot.

Stated tolerances fixed before execution (`study rules` Rule 7):
  1. Largest-remainder allocation & tie-breaking: exact integer equality (tol = 0).
  2. Constant-host finite-horizon conditional-mean growth vs exact matrix powers: <= 1e-14.
  3. Analytical second-order ring loss sigma^2 * K across pilot cells vs exact rationals: <= 1e-15.
  4. Wilson 95% interval, 4-cell Wald contrast, and 80% power arithmetic vs closed form: <= 1e-12.
  5. Constant-host demographic extinction at pilot parameters (log R0 = 0.05, horizon 500)
     vs exact multitype branching PGF on 5,000 trajectories: <= 4 * sqrt(p*(1-p)/5000) + 0.002.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from pilot_lib import (
    POWER80_FACTOR,
    Z95,
    compute_finite_conditional_growth,
    compute_four_cell_interaction,
    largest_remainder_allocate,
    wilson_ci95,
)
from sim_world import (
    build_geometric_ring,
    exact_ring_small_noise_K,
    multitype_branching_extinction_curve,
    run_individual_world,
)


class PilotPreValidationTests(unittest.TestCase):
    def test_1_largest_remainder_allocations(self) -> None:
        expected = {1.0: [13, 7], 5.0: [6, 14], 20.0: [2, 18]}
        for stay, exp_vec in expected.items():
            mdl = build_geometric_ring([2.0, stay])
            _, v, _ = mdl.perron_at_mean_host(0.05)
            alloc = largest_remainder_allocate(20, v)
            self.assertEqual(alloc.tolist(), exp_vec)
            self.assertEqual(int(alloc.sum()), 20)
            self.assertTrue(np.all(np.abs(alloc - 20.0 * v) < 1.0))

        # Exact tie-breaking by state index
        self.assertEqual(largest_remainder_allocate(3, [0.5, 0.5]).tolist(), [2, 1])
        self.assertEqual(largest_remainder_allocate(5, [1.0, 1.0, 1.0]).tolist(), [2, 2, 1])

    def test_2_constant_host_finite_conditional_growth(self) -> None:
        for stay, init_lr in ((1.0, [13, 7]), (5.0, [6, 14]), (20.0, [2, 18])):
            mdl = build_geometric_ring([2.0, stay])
            lam, v, _ = mdl.perron_at_mean_host(0.05)
            h_const = np.ones((500, 3, 1), dtype=np.float64)

            # Starting from exact Perron vector => growth per step equals log(lam) at every horizon
            g_perron = compute_finite_conditional_growth(mdl, h_const, 0.05, v, horizons=(100, 500))
            for ep in (100, 500):
                self.assertLessEqual(float(np.max(np.abs(g_perron[ep] - math.log(lam)))), 1e-14)

            # Starting from integer largest-remainder vector => matches exact matrix power M^t
            g_lr = compute_finite_conditional_growth(mdl, h_const, 0.05, init_lr, horizons=(100, 500))
            mat = mdl.expectation_matrix(None, tilt_y=0.05)
            x0 = np.asarray(init_lr, dtype=np.float64) / 20.0
            for ep in (100, 500):
                exact_g = math.log(float(np.sum(np.linalg.matrix_power(mat, ep) @ x0))) / ep
                self.assertLessEqual(float(np.max(np.abs(g_lr[ep] - exact_g))), 1e-14)

    def test_3_pilot_sigma2_K_exact_rationals(self) -> None:
        # Exact rational check:
        # phi = 0: K = 1 / (2 * (2 + stay))
        self.assertLessEqual(abs(0.09 * exact_ring_small_noise_K([2.0, 1.0], 0.0) - 0.09 / 6.0), 1e-15)
        self.assertLessEqual(abs(0.36 * exact_ring_small_noise_K([2.0, 5.0], 0.0) - 0.36 / 14.0), 1e-15)
        self.assertLessEqual(abs(0.36 * exact_ring_small_noise_K([2.0, 20.0], 0.0) - 0.36 / 44.0), 1e-15)
        # phi = 0.6, stay = 1: theta = 1.5, T = 3, R = (0.6/1.4)*(0.6/1.0) = 9/35
        # K = 4/6 - (9/35)/(26/35) = 2/3 - 9/26 = 25/78; 0.36 * K = (9/25)*(25/78) = 3/26
        self.assertLessEqual(abs(0.36 * exact_ring_small_noise_K([2.0, 1.0], 0.6) - (3.0 / 26.0)), 1e-15)

    def test_4_wilson_contrast_and_power_formulas(self) -> None:
        lo, hi = wilson_ci95(73, 200)
        self.assertLessEqual(abs(lo - 0.301405264649268), 1e-12)
        self.assertLessEqual(abs(hi - 0.433682973472594), 1e-12)

        mock_cells = {
            "c1": {"n": 200, "extinct": 194, "extinction": 0.97, "sigma": 0.6},
            "c2": {"n": 200, "extinct": 67, "extinction": 0.335, "sigma": 0.0},
            "c3": {"n": 200, "extinct": 77, "extinction": 0.385, "sigma": 0.6},
            "c4": {"n": 200, "extinct": 36, "extinction": 0.18, "sigma": 0.0},
        }
        res = compute_four_cell_interaction(mock_cells, {}, ["c1", "c2", "c3", "c4"], [1, -1, -1, 1], "extinction")
        self.assertLessEqual(abs(float(res["effect"]) - 0.43), 1e-12)
        self.assertLessEqual(abs(float(res["variance_sum"]) - 0.63625), 1e-12)
        expected_se = math.sqrt(0.63625 / 200.0)
        self.assertLessEqual(abs(float(res["se"]) - expected_se), 1e-12)
        self.assertLessEqual(abs(res["ci95"][0] - (0.43 - Z95 * expected_se)), 1e-12)
        self.assertEqual(int(res["n80_per_cell"]), int(math.ceil(POWER80_FACTOR * 0.63625 / (0.43 * 0.43))))
        self.assertEqual(int(res["n80_per_cell"]), 28)

    def test_5_pilot_constant_host_branching_pgf_law(self) -> None:
        n_traj = 5_000
        mdl = build_geometric_ring([2.0, 1.0])
        init_pop = np.broadcast_to(np.array([13, 7], dtype=np.int64), (n_traj, 2)).copy()
        h_const = np.ones((500, n_traj, 1), dtype=np.float64)
        sim = run_individual_world(mdl, init_pop, 500, seed=350999, hosts=h_const, tilt_y=0.05)
        pgf = multitype_branching_extinction_curve((2.0, 1.0), 0.05, (13, 7), 500)
        p_exact = float(pgf[500])
        p_emp = float(np.mean(sim["extinction_step"] <= 500))
        tol = 4.0 * math.sqrt(p_exact * (1.0 - p_exact) / n_traj) + 0.002
        self.assertLessEqual(abs(p_emp - p_exact), tol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
