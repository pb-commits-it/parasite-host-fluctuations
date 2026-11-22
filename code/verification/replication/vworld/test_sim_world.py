#!/usr/bin/env python3
"""Pre-gate structural and analytical unit tests (study rules Rule 7 & protocol.md line 90).

All tolerances are stated prior to execution:
  - Kernel mean durations: <= 1e-10; rounded-gamma tail mass: < 1e-12.
  - Constant-environment critical tilt and log-Perron growth: <= 1e-12.
  - Delay/saturation/patch/alternative matrix identities: <= 1e-12.
  - Averaged-rate critical tilt for saturating & zero-death delay models: <= 1e-11.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from sim_world import (
    EndogenousRickerHost,
    KernelSpec,
    StageSpec,
    TransferSpec,
    WorldModel,
    averaged_rate_critical_tilt,
    build_geometric_ring,
    generate_lognormal_ar1_hosts,
    normalize_recorded_series,
    propagate_invasion_products,
    run_individual_world,
    solve_threshold_D,
)


class StructuralWorldTests(unittest.TestCase):
    def test_kernel_means_and_tails(self) -> None:
        for mu in (1.0, 2.0, 5.0, 20.0):
            self.assertLessEqual(abs(KernelSpec.geometric(mu).mean_duration - mu), 1e-12)
        self.assertLessEqual(abs(KernelSpec.erlang_chain(24.0, 4).mean_duration - 24.0), 1e-12)
        self.assertLessEqual(abs(KernelSpec.empirical_pmf([0.2, 0.3, 0.5]).mean_duration - 2.3), 1e-12)
        for shp in (0.5, 1.0, 4.0, 16.0):
            kg = KernelSpec.rounded_gamma(5.0, shp, tail_cutoff=1e-12)
            self.assertLessEqual(abs(kg.mean_duration - 5.0), 1e-10)
            self.assertLess(kg.truncated_tail, 1e-12)

    def test_constant_host_criticality_and_growth(self) -> None:
        for stays in ([2.0, 5.0], [1.0, 2.0, 7.0], [2.0, 3.0, 4.0, 5.0]):
            mdl = build_geometric_ring(stays)
            self.assertLessEqual(abs(mdl.critical_tilt_mean_host()), 1e-12)
            h_const = np.ones((250, 2, 1), dtype=np.float64)
            r_vec, _, _ = propagate_invasion_products(mdl, h_const, tilt_y=0.12)
            lam = mdl.perron_at_mean_host(0.12)[0]
            self.assertLessEqual(abs(float(np.mean(r_vec)) - math.log(lam)), 1e-12)

    def test_deterministic_fixed_delay_and_zero_absorption(self) -> None:
        fixed3 = KernelSpec.empirical_pmf([0.0, 0.0, 1.0])
        mdl = WorldModel(
            stages=[
                StageSpec("a", fixed3),
                StageSpec("b", KernelSpec.geometric(1.0)),
            ],
            transfers=[
                TransferSpec(0, 1, is_poisson=False),
                TransferSpec(1, 0, is_birth=True, is_poisson=True),
            ],
        )
        cnt = np.array([[1, 0, 0, 0]], dtype=np.int64)
        rng = np.random.default_rng(42)
        for expected_idx in (1, 2, 3):
            cnt = mdl.sample_step(cnt, np.ones((1, 1)), rng)
            self.assertEqual(int(cnt[0, expected_idx]), 1)
            self.assertEqual(int(cnt.sum()), 1)

        mp_mdl = build_geometric_ring([2.0, 5.0], n_patches=4, dispersal_rate=0.2)
        self.assertTrue(np.allclose(mp_mdl.dispersal_mat.sum(axis=0), 1.0, atol=1e-13))
        z_cnt = np.zeros((3, mp_mdl.dim), dtype=np.int64)
        self.assertEqual(int(mp_mdl.sample_step(z_cnt, np.ones((3, 4)), rng).sum()), 0)

    def test_delay_saturation_and_nonlinear_guard(self) -> None:
        for mode in ("delay", "linear_delay"):
            mdl = WorldModel(
                stages=[
                    StageSpec("search", KernelSpec.geometric(4.0), mortality=0.125, scarcity_mode=mode),
                    StageSpec("adult", KernelSpec.geometric(5.0)),
                ],
                transfers=[
                    TransferSpec(0, 1, host_driver=0),
                    TransferSpec(1, 0, is_birth=True),
                ],
            )
            self.assertTrue(np.allclose(mdl.expectation_matrix(), [[0.75, 0.2], [0.125, 0.8]], atol=1e-13))
            rng = np.random.default_rng(7)
            if mode == "linear_delay":
                with self.assertRaises(ValueError):
                    mdl.sample_step(np.ones((1, 2), dtype=np.int64), np.array([[10.0]]), rng)
            else:
                for h_val in (0.0, 1.0, 1e6):
                    mdl.sample_step(np.full((1, 2), 50, dtype=np.int64), np.array([[h_val]]), rng)

        # Saturating contact requires explicit arithmetic-mean-rate reference
        sat_mdl = WorldModel(
            stages=[StageSpec("e", KernelSpec.geometric(2.0)), StageSpec("a", KernelSpec.geometric(5.0))],
            transfers=[TransferSpec(0, 1, host_driver=0), TransferSpec(1, 0, is_birth=True)],
            saturating=True,
        )
        self.assertTrue(
            np.allclose(
                sat_mdl.host_response(np.array([[0.0], [1.0], [100.0]])).ravel(),
                [0.0, 1.0, 200.0 / 101.0],
                atol=1e-13,
            )
        )
        h_grid = np.array([[0.5], [1.5]])
        w_grid = np.array([0.5, 0.5])
        ref = averaged_rate_critical_tilt(sat_mdl, h_grid, w_grid)
        expected_tilt = -math.log(0.5 * (2.0 * 0.5 / 1.5 + 2.0 * 1.5 / 2.5))
        self.assertLessEqual(abs(ref["avg_rate_tilt"] - expected_tilt), 1e-11)
        with self.assertRaises(ValueError):
            solve_threshold_D(sat_mdl, np.ones((10, 2, 1)))

        # Zero-death delay preserves survival probability 1 at every host state => avg_rate_tilt == 0
        nod_mdl = WorldModel(
            stages=[
                StageSpec("search", KernelSpec.geometric(4.0), mortality=0.0, scarcity_mode="delay"),
                StageSpec("adult", KernelSpec.geometric(5.0)),
            ],
            transfers=[TransferSpec(0, 1, host_driver=0), TransferSpec(1, 0, is_birth=True)],
        )
        ref_nod = averaged_rate_critical_tilt(nod_mdl, np.array([[0.2], [1.8]]), np.array([0.5, 0.5]))
        self.assertLessEqual(abs(ref_nod["avg_rate_tilt"]), 1e-11)
        self.assertTrue(np.allclose(nod_mdl.expectation_matrix(np.array([[0.2], [1.8]])).sum(axis=1), 1.0, atol=1e-13))

    def test_patch_and_alternative_route_collapse(self) -> None:
        rich = build_geometric_ring(
            [2.0, 5.0], n_drivers=2, split_alternatives=True, n_patches=4, dispersal_rate=0.2
        )
        base = build_geometric_ring([2.0, 5.0])
        m_rich = rich.expectation_matrix(np.ones(8))
        m_base = base.expectation_matrix()
        for col_stage in range(2):
            collapsed = m_rich[:, col_stage].reshape(4, 2).sum(axis=0)
            self.assertTrue(np.allclose(collapsed, m_base[:, col_stage], atol=1e-13))

    def test_recorded_and_ricker_hosts(self) -> None:
        series, base = normalize_recorded_series([0.0, 1.0, 2.0], np.log([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(base, 2.0, places=12)
        self.assertAlmostEqual(float(np.mean(series)), 1.0, places=12)
        with self.assertRaises(ValueError):
            normalize_recorded_series([0.0, 1.0, 3.0], [0.0, 0.0, 0.0])

        r_clean = EndogenousRickerHost(np.full((2, 1), 1000.0), env_sigma=0.0, burden_alpha=0.0)
        r_inf = EndogenousRickerHost(np.full((2, 1), 1000.0), env_sigma=0.0, burden_alpha=0.1)
        self.assertTrue(np.allclose(r_clean.advance(np.full((2, 1), 10.0)), 1.0, atol=1e-13))
        self.assertTrue(np.all(r_inf.advance(np.full((2, 1), 10.0)) < 1.0))

        mdl = build_geometric_ring([2.0, 5.0])
        out = run_individual_world(mdl, np.zeros((2, 2), dtype=np.int64), 3, seed=9, ricker_host=r_clean)
        self.assertTrue(np.array_equal(out["extinction_step"], np.array([0, 0])))

    def test_input_validation_guards(self) -> None:
        with self.assertRaises(ValueError):
            KernelSpec.geometric(0.5)
        with self.assertRaises(ValueError):
            KernelSpec.erlang_chain(2.0, 4)
        with self.assertRaises(ValueError):
            KernelSpec.empirical_pmf([0.2, 0.3])
        with self.assertRaises(ValueError):
            generate_lognormal_ar1_hosts(5, 2, 0.3, 1.0, seed=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
