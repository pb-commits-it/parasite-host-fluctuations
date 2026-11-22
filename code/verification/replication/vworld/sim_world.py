#!/usr/bin/env python3
"""Clean-room stochastic complex-life-cycle world model for r35-vworld-A.

Implemented independently from the registered specification in
`virtualworld/protocol.md` without importing or copying code
from `virtualworld/` or `studies/`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy import integrate, linalg, optimize, signal, special, stats


@dataclass(frozen=True)
class KernelSpec:
    """Discrete-time stage residence kernel prior to independent mortality.

    `sub_matrix[i, j]` is the probability an individual at internal age/phase `j`
    transitions to internal age/phase `i` in one step; `exit_prob[j]` is the
    probability of completing development from `j` in one step.
    """

    sub_matrix: np.ndarray
    exit_prob: np.ndarray
    kind: str
    truncated_tail: float = 0.0

    def __post_init__(self) -> None:
        u = np.asarray(self.sub_matrix, dtype=np.float64)
        e = np.asarray(self.exit_prob, dtype=np.float64)
        m = e.shape[0]
        if e.ndim != 1 or m < 1 or u.shape != (m, m):
            raise ValueError("Invalid residence kernel dimensions.")
        if np.any(u < 0.0) or np.any(e < 0.0) or np.any(~np.isfinite(u)) or np.any(~np.isfinite(e)):
            raise ValueError("Residence probabilities must be finite and nonnegative.")
        col_sums = u.sum(axis=0) + e
        if not np.allclose(col_sums, 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("Column survival + exit probabilities must sum to 1.")
        spec_rad = float(np.max(np.abs(linalg.eigvals(u))))
        if spec_rad >= 1.0 - 1e-14:
            raise ValueError("Residence kernel must be transient (spectral radius < 1).")
        object.__setattr__(self, "sub_matrix", u)
        object.__setattr__(self, "exit_prob", e)

    @property
    def num_states(self) -> int:
        return int(self.exit_prob.shape[0])

    @property
    def mean_duration(self) -> float:
        m = self.num_states
        e0 = np.zeros(m, dtype=np.float64)
        e0[0] = 1.0
        fundamental = linalg.solve(np.eye(m) - self.sub_matrix, e0)
        return float(np.sum(fundamental))

    @classmethod
    def geometric(cls, mean_steps: float) -> "KernelSpec":
        if not np.isfinite(mean_steps) or mean_steps < 1.0:
            raise ValueError("Geometric mean residence must be >= 1.")
        q = 1.0 / float(mean_steps)
        return cls(np.array([[1.0 - q]], dtype=np.float64), np.array([q], dtype=np.float64), "geometric")

    @classmethod
    def erlang_chain(cls, mean_steps: float, num_phases: int) -> "KernelSpec":
        if int(num_phases) != num_phases or num_phases < 1 or mean_steps < num_phases:
            raise ValueError("Erlang chain requires integer num_phases >= 1 and mean_steps >= num_phases.")
        k = int(num_phases)
        q = k / float(mean_steps)
        u = np.eye(k, dtype=np.float64) * (1.0 - q)
        for j in range(k - 1):
            u[j + 1, j] = q
        ex = np.zeros(k, dtype=np.float64)
        ex[-1] = q
        return cls(u, ex, f"erlang_chain({k})")

    @classmethod
    def empirical_pmf(cls, pmf: Sequence[float]) -> "KernelSpec":
        p = np.asarray(pmf, dtype=np.float64)
        if p.ndim != 1 or p.size == 0 or np.any(~np.isfinite(p)) or np.any(p < 0.0):
            raise ValueError("Empirical duration PMF must be a 1D nonnegative finite array.")
        if not np.isclose(np.sum(p), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("Empirical duration PMF must sum to 1.")
        nz = np.flatnonzero(p > 0.0)
        p = p[: int(nz[-1]) + 1] / float(np.sum(p))
        m = p.size
        surv = np.cumsum(p[::-1])[::-1]
        haz = p / surv
        u = np.zeros((m, m), dtype=np.float64)
        for j in range(m - 1):
            u[j + 1, j] = 1.0 - haz[j]
        return cls(u, haz, "empirical")

    @classmethod
    def rounded_gamma(cls, mean_steps: float, shape: float, tail_cutoff: float = 1e-12) -> "KernelSpec":
        if mean_steps <= 1.0 or shape <= 0.0 or not (0.0 < tail_cutoff <= 1e-10):
            raise ValueError("Rounded gamma requires mean_steps > 1, shape > 0, and tail_cutoff in (0, 1e-10].")
        scale_hi = float(mean_steps) / float(shape)
        max_age = int(math.ceil(stats.gamma.ppf(1.0 - tail_cutoff, a=shape, scale=scale_hi))) + 1
        if max_age > 2500:
            raise ValueError("Rounded gamma age grid exceeds 2500 states.")
        grid = np.arange(max_age, dtype=np.float64)

        def mean_err(sc: float) -> float:
            return float(np.sum(stats.gamma.sf(grid, a=shape, scale=sc))) - float(mean_steps)

        scale_star = float(optimize.brentq(mean_err, scale_hi * 1e-9, scale_hi, xtol=1e-13))
        sf_vals = stats.gamma.sf(grid, a=shape, scale=scale_star)
        pmf = np.empty(max_age, dtype=np.float64)
        pmf[:-1] = sf_vals[:-1] - sf_vals[1:]
        pmf[-1] = sf_vals[-1]
        tail_mass = float(sf_vals[-1])
        base = cls.empirical_pmf(pmf)
        return cls(base.sub_matrix, base.exit_prob, f"rounded_gamma({shape})", truncated_tail=tail_mass)


@dataclass(frozen=True)
class StageSpec:
    name: str
    kernel: KernelSpec
    mortality: float = 0.0
    host_driver: int | None = None
    scarcity_mode: str = "waste"  # 'waste' | 'delay' | 'linear_delay'


@dataclass(frozen=True)
class TransferSpec:
    src_stage: int
    dst_stage: int
    host_driver: int | None = None
    share: float = 1.0
    offspring_yield: float = 1.0
    is_birth: bool = False
    is_poisson: bool = True


@dataclass(frozen=True)
class _TransitionBranch:
    src_state: int
    dst_state: int
    base_prob: float
    linear_prob_slope: tuple[int, float] | None  # (env_channel, slope_wrt_rate_minus_1)
    hazard_params: tuple[int, float, float, bool] | None  # (env_channel, c_rate, mortality, is_exit)
    base_yield: float
    yield_channel: int | None
    is_birth: bool
    is_poisson: bool


class WorldModel:
    """Discrete-time individual branching model with 2..4 biological stages."""

    def __init__(
        self,
        stages: Sequence[StageSpec],
        transfers: Sequence[TransferSpec],
        n_drivers: int = 1,
        n_patches: int = 1,
        dispersal_rate: float = 0.0,
        mobile_stage: int | None = None,
        saturating: bool = False,
    ) -> None:
        k_bio = len(stages)
        if not (2 <= k_bio <= 4) or len({s.name for s in stages}) != k_bio:
            raise ValueError("Model requires 2 to 4 uniquely named biological stages.")
        if int(n_patches) != n_patches or n_patches < 1 or n_drivers < 1:
            raise ValueError("n_patches and n_drivers must be positive integers.")
        if not (0.0 <= dispersal_rate <= 1.0):
            raise ValueError("dispersal_rate must lie in [0, 1].")
        if mobile_stage is not None and not (0 <= mobile_stage < k_bio):
            raise ValueError("Invalid mobile_stage index.")

        self.stages = list(stages)
        self.transfers = list(transfers)
        self.n_drivers = int(n_drivers)
        self.n_patches = int(n_patches)
        self.n_channels = self.n_patches * self.n_drivers
        self.dispersal_rate = float(dispersal_rate)
        self.mobile_stage = mobile_stage
        self.saturating = bool(saturating)

        self.stage_slices: list[list[int]] = []
        offset = 0
        for st in self.stages:
            if not (0.0 <= st.mortality < 1.0):
                raise ValueError("Stage mortality must lie in [0, 1).")
            if st.scarcity_mode not in ("waste", "delay", "linear_delay"):
                raise ValueError(f"Unsupported scarcity_mode: {st.scarcity_mode}")
            if st.host_driver is not None and not (0 <= st.host_driver < self.n_drivers):
                raise ValueError("Invalid stage host_driver.")
            m = st.kernel.num_states
            self.stage_slices.append(list(range(offset, offset + m)))
            offset += m

        self.local_dim = offset
        self.dim = self.local_dim * self.n_patches

        for tr in self.transfers:
            if not (0 <= tr.src_stage < k_bio) or tr.dst_stage != (tr.src_stage + 1) % k_bio:
                raise ValueError("Transfers must follow cyclic stage order j -> (j+1) mod k.")
            if tr.share < 0.0 or tr.offspring_yield <= 0.0:
                raise ValueError("Transfer share must be >= 0 and offspring_yield > 0.")
            if tr.host_driver is not None and not (0 <= tr.host_driver < self.n_drivers):
                raise ValueError("Invalid transfer host_driver.")
            if not tr.is_poisson and (tr.offspring_yield != 1.0 or tr.host_driver is not None or tr.is_birth):
                raise ValueError("Deterministic continuation requires unit yield, no host scaling, and no birth tilt.")
        if not any(tr.is_birth for tr in self.transfers):
            raise ValueError("At least one transfer must carry the designated birth tilt.")

        # Post-demography island dispersal matrix (columns = pre-dispersal, rows = post-dispersal)
        self.dispersal_mat = np.eye(self.dim, dtype=np.float64)
        if self.mobile_stage is not None and self.n_patches > 1 and self.dispersal_rate > 0.0:
            p_cnt = self.n_patches
            island = (1.0 - self.dispersal_rate) * np.eye(p_cnt) + (
                self.dispersal_rate / p_cnt
            ) * np.ones((p_cnt, p_cnt))
            for loc_idx in self.stage_slices[self.mobile_stage]:
                patch_states = np.arange(p_cnt) * self.local_dim + loc_idx
                self.dispersal_mat[np.ix_(patch_states, patch_states)] = island

        branches: list[_TransitionBranch] = []
        for p in range(self.n_patches):
            p_base = p * self.local_dim
            for j, st in enumerate(self.stages):
                out_routes = [tr for tr in self.transfers if tr.src_stage == j and tr.share > 0.0]
                if not out_routes or not np.isclose(sum(tr.share for tr in out_routes), 1.0, atol=1e-12, rtol=0.0):
                    raise ValueError(f"Outgoing shares from stage {st.name} must sum to 1.")
                loc_states = [p_base + idx for idx in self.stage_slices[j]]

                if st.scarcity_mode in ("delay", "linear_delay"):
                    if st.kernel.kind != "geometric" or len(out_routes) != 1 or out_routes[0].host_driver is None:
                        raise ValueError("Delay modes require a geometric stage with a single host-driven route.")
                    tr = out_routes[0]
                    ch = p * self.n_drivers + tr.host_driver
                    q_tot = 1.0 / st.kernel.mean_duration
                    if not (st.mortality < q_tot < 1.0):
                        raise ValueError("Delay modes require mortality < 1/mean < 1.")
                    dst0 = p_base + self.stage_slices[tr.dst_stage][0]
                    if st.scarcity_mode == "linear_delay":
                        c_lin = q_tot - st.mortality
                        branches.append(
                            _TransitionBranch(
                                src_state=loc_states[0],
                                dst_state=loc_states[0],
                                base_prob=1.0 - q_tot,
                                linear_prob_slope=(ch, -c_lin),
                                hazard_params=None,
                                base_yield=1.0,
                                yield_channel=None,
                                is_birth=False,
                                is_poisson=False,
                            )
                        )
                        branches.append(
                            _TransitionBranch(
                                src_state=loc_states[0],
                                dst_state=dst0,
                                base_prob=c_lin,
                                linear_prob_slope=(ch, +c_lin),
                                hazard_params=None,
                                base_yield=tr.offspring_yield,
                                yield_channel=None,
                                is_birth=tr.is_birth,
                                is_poisson=tr.is_poisson,
                            )
                        )
                    else:
                        c_haz = -math.log((1.0 - q_tot) / (1.0 - st.mortality))
                        branches.append(
                            _TransitionBranch(
                                src_state=loc_states[0],
                                dst_state=loc_states[0],
                                base_prob=1.0 - q_tot,
                                linear_prob_slope=None,
                                hazard_params=(ch, c_haz, st.mortality, False),
                                base_yield=1.0,
                                yield_channel=None,
                                is_birth=False,
                                is_poisson=False,
                            )
                        )
                        branches.append(
                            _TransitionBranch(
                                src_state=loc_states[0],
                                dst_state=dst0,
                                base_prob=q_tot - st.mortality,
                                linear_prob_slope=None,
                                hazard_params=(ch, c_haz, st.mortality, True),
                                base_yield=tr.offspring_yield,
                                yield_channel=None,
                                is_birth=tr.is_birth,
                                is_poisson=tr.is_poisson,
                            )
                        )
                    continue

                surv_step = 1.0 - st.mortality
                for age_idx, src_u in enumerate(loc_states):
                    for nxt_age in np.flatnonzero(st.kernel.sub_matrix[:, age_idx] > 0.0):
                        pr = surv_step * float(st.kernel.sub_matrix[nxt_age, age_idx])
                        branches.append(
                            _TransitionBranch(
                                src_state=src_u,
                                dst_state=loc_states[int(nxt_age)],
                                base_prob=pr,
                                linear_prob_slope=None,
                                hazard_params=None,
                                base_yield=1.0,
                                yield_channel=None,
                                is_birth=False,
                                is_poisson=False,
                            )
                        )
                    ex_p = float(st.kernel.exit_prob[age_idx])
                    if ex_p > 0.0:
                        for tr in out_routes:
                            pr = surv_step * ex_p * tr.share
                            if pr > 0.0:
                                ch = None if tr.host_driver is None else (p * self.n_drivers + tr.host_driver)
                                dst0 = p_base + self.stage_slices[tr.dst_stage][0]
                                branches.append(
                                    _TransitionBranch(
                                        src_state=src_u,
                                        dst_state=dst0,
                                        base_prob=pr,
                                        linear_prob_slope=None,
                                        hazard_params=None,
                                        base_yield=tr.offspring_yield,
                                        yield_channel=ch,
                                        is_birth=tr.is_birth,
                                        is_poisson=tr.is_poisson,
                                    )
                                )

        self.branches = branches
        self.branches_by_src: list[list[_TransitionBranch]] = [
            [br for br in self.branches if br.src_state == u] for u in range(self.dim)
        ]
        self.is_rate_affine = not any(br.hazard_params is not None for br in self.branches)

        # Precompile affine expectation tensors: M(rates, y) = (U0 + e^y F0) + sum_c (r_c - 1)*(U1_c + e^y F1_c)
        self.U0 = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.F0 = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.U1 = np.zeros((self.n_channels, self.dim, self.dim), dtype=np.float64)
        self.F1 = np.zeros((self.n_channels, self.dim, self.dim), dtype=np.float64)

        if self.is_rate_affine:
            for br in self.branches:
                col_disp = self.dispersal_mat[:, br.dst_state]
                base_contrib = col_disp * (br.base_prob * br.base_yield)
                if br.is_birth:
                    self.F0[:, br.src_state] += base_contrib
                else:
                    self.U0[:, br.src_state] += base_contrib

                if br.linear_prob_slope is not None and br.yield_channel is not None:
                    raise ValueError("Simultaneous linear contact and yield scaling on one branch unsupported.")
                if br.linear_prob_slope is not None:
                    ch, slp = br.linear_prob_slope
                    slp_contrib = col_disp * (slp * br.base_yield)
                    if br.is_birth:
                        self.F1[ch, :, br.src_state] += slp_contrib
                    else:
                        self.U1[ch, :, br.src_state] += slp_contrib
                elif br.yield_channel is not None:
                    ch = br.yield_channel
                    if br.is_birth:
                        self.F1[ch, :, br.src_state] += base_contrib
                    else:
                        self.U1[ch, :, br.src_state] += base_contrib

    def host_response(self, hosts: np.ndarray) -> np.ndarray:
        h = np.asarray(hosts, dtype=np.float64)
        if h.shape[-1] != self.n_channels or np.any(~np.isfinite(h)) or np.any(h < 0.0):
            raise ValueError("Host array must be finite, nonnegative, with trailing axis n_channels.")
        if self.saturating:
            return (2.0 * h) / (1.0 + h)
        return h

    def _branch_prob_and_mean(
        self, br: _TransitionBranch, rates: np.ndarray, tilt_y: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if br.hazard_params is not None:
            ch, c_haz, d_mort, is_exit = br.hazard_params
            surv_c = np.exp(-c_haz * rates[..., ch])
            prob = (1.0 - d_mort) * ((1.0 - surv_c) if is_exit else surv_c)
        elif br.linear_prob_slope is not None:
            ch, slp = br.linear_prob_slope
            prob = br.base_prob + slp * (rates[..., ch] - 1.0)
        else:
            prob = np.broadcast_to(br.base_prob, rates.shape[:-1]).copy()

        if np.any(prob < -1e-12) or np.any(prob > 1.0 + 1e-12):
            raise ValueError("Invalid event probability encountered (no clipping permitted).")
        prob = np.clip(prob, 0.0, 1.0)

        mean_off = br.base_yield * (math.exp(tilt_y) if br.is_birth else 1.0)
        if br.yield_channel is not None:
            mu = mean_off * rates[..., br.yield_channel]
        else:
            mu = np.broadcast_to(mean_off, rates.shape[:-1]).copy()
        return prob, mu

    def expectation_matrix(self, hosts: np.ndarray | None = None, tilt_y: float = 0.0) -> np.ndarray:
        h_in = np.ones(self.n_channels, dtype=np.float64) if hosts is None else np.asarray(hosts, dtype=np.float64)
        rates = self.host_response(h_in)
        ey = math.exp(tilt_y)
        if self.is_rate_affine:
            base = self.U0 + ey * self.F0
            pert = self.U1 + ey * self.F1
            mat = base + np.einsum("...c,cij->...ij", rates - 1.0, pert)
            if np.any(mat < -1e-12):
                raise ValueError("Negative entry in conditional expectation matrix.")
            return mat

        out = np.zeros(rates.shape[:-1] + (self.dim, self.dim), dtype=np.float64)
        for br in self.branches:
            pr, mu = self._branch_prob_and_mean(br, rates, tilt_y)
            out[..., :, br.src_state] += (pr * mu)[..., None] * self.dispersal_mat[:, br.dst_state]
        return out

    def perron_at_mean_host(self, tilt_y: float = 0.0) -> tuple[float, np.ndarray, np.ndarray]:
        mat = self.expectation_matrix(None, tilt_y=tilt_y)
        vals_r, vecs_r = linalg.eig(mat)
        idx_r = int(np.argmax(vals_r.real))
        lam = float(vals_r[idx_r].real)
        v = np.abs(vecs_r[:, idx_r].real)
        v /= float(np.sum(v))

        vals_l, vecs_l = linalg.eig(mat.T)
        idx_l = int(np.argmax(vals_l.real))
        w = np.abs(vecs_l[:, idx_l].real)
        w /= float(w @ v)
        return lam, v, w

    def critical_tilt_mean_host(self) -> float:
        return float(
            optimize.brentq(
                lambda y: math.log(self.perron_at_mean_host(y)[0]),
                -20.0,
                20.0,
                xtol=1e-13,
            )
        )

    def sample_step(
        self,
        counts: np.ndarray,
        hosts: np.ndarray,
        rng: np.random.Generator,
        tilt_y: float = 0.0,
    ) -> np.ndarray:
        """Advance replicate integer stage counts by one exact demographic + dispersal step."""
        c_arr = np.asarray(counts)
        if (
            c_arr.ndim != 2
            or c_arr.shape[1] != self.dim
            or not np.issubdtype(c_arr.dtype, np.integer)
            or np.any(c_arr < 0)
        ):
            raise ValueError("counts must be a 2D nonnegative integer array of shape (replicates, dim).")
        n_rep = c_arr.shape[0]
        rates = self.host_response(hosts)
        if rates.shape != (n_rep, self.n_channels):
            raise ValueError("hosts must have shape (replicates, n_channels).")

        nxt_counts = np.zeros((n_rep, self.dim), dtype=np.int64)
        for u, br_list in enumerate(self.branches_by_src):
            rem_parents = c_arr[:, u].astype(np.int64, copy=True)
            rem_prob = np.ones(n_rep, dtype=np.float64)
            for br in br_list:
                pr, mu = self._branch_prob_and_mean(br, rates, tilt_y)
                if np.any(pr > rem_prob + 1e-10):
                    raise ValueError("Branch probabilities from a state exceed unity.")
                cond_p = np.divide(pr, rem_prob, out=np.zeros_like(rem_prob), where=rem_prob > 1e-14)
                selected = rng.binomial(rem_parents, np.clip(cond_p, 0.0, 1.0))
                rem_parents -= selected
                rem_prob -= pr

                lam = selected * mu
                if np.any(lam > 1e15):
                    raise OverflowError("Poisson intensity exceeded safe int64 threshold (1e15).")
                offspring = rng.poisson(lam) if br.is_poisson else selected

                # Independent post-demography patch dispersal
                dest_indices = np.flatnonzero(self.dispersal_mat[:, br.dst_state] > 0.0)
                if dest_indices.size == 1:
                    nxt_counts[:, int(dest_indices[0])] += offspring
                else:
                    rem_off = offspring.copy()
                    rem_mass = 1.0
                    for d_idx in dest_indices[:-1]:
                        w_move = float(self.dispersal_mat[int(d_idx), br.dst_state])
                        moved = rng.binomial(rem_off, min(1.0, w_move / rem_mass))
                        nxt_counts[:, int(d_idx)] += moved
                        rem_off -= moved
                        rem_mass -= w_move
                    nxt_counts[:, int(dest_indices[-1])] += rem_off

        if np.any(nxt_counts < 0) or np.any(nxt_counts > 10**15):
            raise OverflowError("Stage counts exceeded safe integer bounds.")
        return nxt_counts

    def host_burdens(self, counts: np.ndarray) -> np.ndarray:
        out = np.zeros((counts.shape[0], self.n_channels), dtype=np.float64)
        for p in range(self.n_patches):
            for j, st in enumerate(self.stages):
                if st.host_driver is not None:
                    cols = [p * self.local_dim + idx for idx in self.stage_slices[j]]
                    out[:, p * self.n_drivers + st.host_driver] += np.sum(counts[:, cols], axis=1)
        return out


def build_geometric_ring(
    mean_stays: Sequence[float],
    n_drivers: int = 1,
    steady_mortality: float = 0.0,
    exposed_mode: str = "waste",
    n_patches: int = 1,
    dispersal_rate: float = 0.0,
    split_alternatives: bool = False,
) -> WorldModel:
    """Construct a standard k-stage geometric ring with host-driven 0->1 transfer."""
    k = len(mean_stays)
    stages = [
        StageSpec(
            name=f"stage_{i}",
            kernel=KernelSpec.geometric(float(m)),
            mortality=steady_mortality if i == 1 else 0.0,
            host_driver=0 if i == 1 else None,
            scarcity_mode=exposed_mode if i == 0 else "waste",
        )
        for i, m in enumerate(mean_stays)
    ]
    transfers: list[TransferSpec] = []
    for i in range(k):
        nxt = (i + 1) % k
        if i == 0 and split_alternatives:
            for d in range(n_drivers):
                transfers.append(TransferSpec(0, 1, host_driver=d, share=1.0 / n_drivers, is_birth=False))
        else:
            transfers.append(
                TransferSpec(
                    i,
                    nxt,
                    host_driver=0 if i == 0 else None,
                    share=1.0,
                    is_birth=(i == k - 1),
                )
            )
    return WorldModel(
        stages=stages,
        transfers=transfers,
        n_drivers=n_drivers,
        n_patches=n_patches,
        dispersal_rate=dispersal_rate,
        mobile_stage=1 if n_patches > 1 else None,
    )


# ---------------------------------------------------------------------------
# Exogenous and endogenous host generators
# ---------------------------------------------------------------------------


def generate_lognormal_ar1_hosts(
    n_steps: int,
    n_reps: int,
    sigma: float,
    phi: float,
    seed: int,
    n_drivers: int = 1,
    n_patches: int = 1,
    patch_synchrony: float = 0.0,
    antithetic: bool = False,
    mean_host: float = 1.0,
) -> np.ndarray:
    """Stationary Gaussian AR(1) z_t mapped to H_t = mean_host * exp(sigma*z_t - sigma^2/2)."""
    if sigma < 0.0 or not (-1.0 < phi < 1.0) or mean_host <= 0.0 or not (0.0 <= patch_synchrony <= 1.0):
        raise ValueError("Invalid stationary lognormal AR(1) parameters.")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_steps, n_reps, n_patches, n_drivers))
    if patch_synchrony > 0.0:
        z_shared = rng.standard_normal((n_steps, n_reps, 1, n_drivers))
        z = math.sqrt(1.0 - patch_synchrony) * z + math.sqrt(patch_synchrony) * z_shared
    if n_steps > 1:
        z[1:] *= math.sqrt(1.0 - phi * phi)
    if phi != 0.0:
        z = signal.lfilter([1.0], [1.0, -phi], z, axis=0)
    z = z.reshape(n_steps, n_reps, n_patches * n_drivers)
    if antithetic:
        z = np.concatenate([z, -z], axis=1)
    if sigma == 0.0:
        return np.full_like(z, mean_host)
    return mean_host * np.exp(sigma * z - 0.5 * sigma * sigma)


def generate_bounded_refresh_hosts(
    n_steps: int,
    n_reps: int,
    sigma: float,
    phi: float,
    seed: int,
    n_drivers: int = 1,
    n_patches: int = 1,
    low_level: float = 0.05,
) -> np.ndarray:
    """Two-state refresh Markov host with mean 1, variance exp(sigma^2)-1, lag-1 corr phi."""
    if sigma < 0.0 or not (0.0 <= phi < 1.0) or not (0.0 < low_level < 1.0):
        raise ValueError("Invalid bounded refresh host parameters.")
    n_ch = n_drivers * n_patches
    if sigma == 0.0:
        return np.ones((n_steps, n_reps, n_ch), dtype=np.float64)
    rng = np.random.default_rng(seed)
    var_h = math.expm1(sigma * sigma)
    gap = 1.0 - low_level
    high_level = 1.0 + var_h / gap
    p_high = (gap * gap) / (gap * gap + var_h)

    shape = (n_steps, n_reps, n_ch)
    cand_high = rng.random(shape) < p_high
    refresh_mask = rng.random(shape) < (1.0 - phi)
    refresh_mask[0] = True
    time_idx = np.arange(n_steps, dtype=np.int64)[:, None, None]
    anchor_step = np.maximum.accumulate(np.where(refresh_mask, time_idx, 0), axis=0)
    is_high = np.take_along_axis(cand_high, anchor_step, axis=0)
    return np.where(is_high, high_level, low_level)


def generate_iid_gamma_hosts(n_steps: int, n_reps: int, shape_p: float, seed: int) -> np.ndarray:
    """Independent mean-1 Gamma(p, scale=1/p) host schedule of variance 1/p."""
    if shape_p <= 0.0:
        raise ValueError("Gamma shape_p must be positive.")
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=shape_p, scale=1.0 / shape_p, size=(n_steps, n_reps, 1))


def normalize_recorded_series(
    step_times: Sequence[float],
    log_counts: Sequence[float],
    baseline: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convert equally spaced empirical log-abundance record to arithmetic-baseline host series."""
    t_arr = np.asarray(step_times, dtype=np.float64)
    z_arr = np.asarray(log_counts, dtype=np.float64)
    if t_arr.ndim != 1 or t_arr.size < 2 or z_arr.shape != t_arr.shape:
        raise ValueError("step_times and log_counts must be paired 1D arrays of length >= 2.")
    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(z_arr)):
        raise ValueError("Nonfinite entries in recorded series.")
    if not np.allclose(np.diff(t_arr), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Irregular or gapped empirical series are refused.")
    raw_h = np.exp(z_arr)
    ref = float(np.mean(raw_h)) if baseline is None else float(baseline)
    if not np.isfinite(ref) or ref <= 0.0:
        raise ValueError("Arithmetic baseline must be finite and positive.")
    return (raw_h / ref)[:, None, None], ref


class EndogenousRickerHost:
    """Endogenous Ricker host with environmental log-shocks and linear burden mortality."""

    def __init__(
        self,
        initial_abundance: np.ndarray,
        carrying_capacity: float = 1000.0,
        intrinsic_r: float = 0.8,
        env_sigma: float = 0.3,
        burden_alpha: float = 0.01,
        seed: int = 1,
    ) -> None:
        h0 = np.asarray(initial_abundance, dtype=np.float64).copy()
        if np.any(h0 <= 0.0) or carrying_capacity <= 0.0 or intrinsic_r < 0.0 or env_sigma < 0.0 or burden_alpha < 0.0:
            raise ValueError("Invalid Ricker parameters.")
        self.abundance = h0
        self.K = float(carrying_capacity)
        self.r = float(intrinsic_r)
        self.sigma = float(env_sigma)
        self.alpha = float(burden_alpha)
        self.rng = np.random.default_rng(seed)

    def advance(self, parasite_burden: np.ndarray) -> np.ndarray:
        b = np.asarray(parasite_burden, dtype=np.float64)
        if b.shape != self.abundance.shape or np.any(b < 0.0):
            raise ValueError("Parasite burden shape mismatch or negative burden.")
        shock = self.sigma * self.rng.standard_normal(self.abundance.shape) - 0.5 * self.sigma * self.sigma
        growth = self.r * (1.0 - self.abundance / self.K) + shock - self.alpha * (b / self.abundance)
        self.abundance = self.abundance * np.exp(growth)
        if np.any(self.abundance <= 0.0) or np.any(~np.isfinite(self.abundance)):
            raise FloatingPointError("Ricker host abundance underflowed or overflowed.")
        return self.abundance / self.K


# ---------------------------------------------------------------------------
# Forward demographic simulation & conditional-mean invasion analysis
# ---------------------------------------------------------------------------


def run_individual_world(
    model: WorldModel,
    init_counts: np.ndarray,
    n_steps: int,
    seed: int,
    hosts: np.ndarray | None = None,
    ricker_host: EndogenousRickerHost | None = None,
    tilt_y: float = 0.0,
) -> dict[str, np.ndarray]:
    """Simulate trajectories forward for `n_steps` and record exact first extinction times."""
    cur = np.asarray(init_counts, dtype=np.int64).copy()
    if cur.ndim != 2 or cur.shape[1] != model.dim or np.any(cur < 0):
        raise ValueError("Invalid initial count array.")
    if (hosts is None) == (ricker_host is None):
        raise ValueError("Provide exactly one of exogenous `hosts` or `ricker_host`.")
    n_rep = cur.shape[0]
    if hosts is not None and hosts.shape != (n_steps, n_rep, model.n_channels):
        raise ValueError("Exogenous host array has wrong shape.")
    if ricker_host is not None and ricker_host.abundance.shape != (n_rep, model.n_channels):
        raise ValueError("Endogenous host array has wrong shape.")

    rng = np.random.default_rng(seed)
    totals = np.empty((n_steps + 1, n_rep), dtype=np.int64)
    totals[0] = np.sum(cur, axis=1)
    first_ext = np.full(n_rep, n_steps + 1, dtype=np.int64)
    first_ext[totals[0] == 0] = 0

    h_hist = np.empty((n_steps, n_rep, model.n_channels), dtype=np.float64) if hosts is None else hosts
    h_cur = (ricker_host.abundance / ricker_host.K) if ricker_host is not None else None

    for t in range(n_steps):
        if hosts is not None:
            h_step = hosts[t]
        else:
            assert h_cur is not None and ricker_host is not None
            h_step = h_cur
            h_hist[t] = h_step
        cur = model.sample_step(cur, h_step, rng, tilt_y=tilt_y)
        totals[t + 1] = np.sum(cur, axis=1)
        newly_extinct = (totals[t + 1] == 0) & (first_ext > n_steps)
        first_ext[newly_extinct] = t + 1
        if ricker_host is not None:
            h_cur = ricker_host.advance(model.host_burdens(cur))

    return {
        "totals": totals,
        "final_counts": cur,
        "extinction_step": first_ext,
        "survived": first_ext > n_steps,
        "hosts": h_hist,
    }


def sample_se(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=np.float64)
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def _fold_antithetic(vals: np.ndarray, n_pairs: int) -> np.ndarray:
    if n_pairs > 0:
        return 0.5 * (vals[:n_pairs] + vals[n_pairs:])
    return vals


def propagate_invasion_products(
    model: WorldModel,
    hosts: np.ndarray,
    tilt_y: float = 0.0,
    burn_steps: int = 0,
    n_pairs: int = 0,
    cv_weight: float = 0.0,
    init_dist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate normalized conditional-mean composition and exact y-tangent."""
    n_total, n_rep, _ = hosts.shape
    if n_total <= burn_steps:
        raise ValueError("Total steps must exceed burn_steps.")
    rates = model.host_response(hosts)

    if init_dist is None:
        v0 = model.perron_at_mean_host(tilt_y)[1]
    else:
        v0 = np.asarray(init_dist, dtype=np.float64)
        v0 = v0 / float(np.sum(v0))

    comp = np.broadcast_to(v0, (n_rep, model.dim)).copy()
    dcomp = np.zeros_like(comp)

    cum_log = np.zeros(n_rep, dtype=np.float64)
    cum_slope = np.zeros(n_rep, dtype=np.float64)
    cum_half = np.zeros(n_rep, dtype=np.float64)

    rec_steps = n_total - burn_steps
    half_start = burn_steps + rec_steps // 2
    half_steps = n_total - half_start

    ey = math.exp(tilt_y)
    if model.is_rate_affine:
        base_m = model.U0 + ey * model.F0
        base_dm = ey * model.F0
        pert_m = model.U1 + ey * model.F1
        pert_dm = ey * model.F1

        # Process in chunks to balance vectorization speed and <=2 GiB memory budget
        chunk = 4096
        for start_t in range(0, n_total, chunk):
            end_t = min(n_total, start_t + chunk)
            dr_chunk = rates[start_t:end_t] - 1.0  # (C, R, ch)
            mats = base_m + np.einsum("trc,cij->trij", dr_chunk, pert_m)
            dmats = base_dm + np.einsum("trc,cij->trij", dr_chunk, pert_dm)
            for idx_t in range(end_t - start_t):
                t = start_t + idx_t
                m_t = mats[idx_t]
                dm_t = dmats[idx_t]
                nxt = np.einsum("rij,rj->ri", m_t, comp)
                dnxt = np.einsum("rij,rj->ri", m_t, dcomp) + np.einsum("rij,rj->ri", dm_t, comp)
                tot = np.sum(nxt, axis=1)
                if np.any(tot <= 0.0) or np.any(nxt < -1e-12):
                    raise FloatingPointError("Invalid conditional-mean state vector.")
                loc_slope = np.sum(dnxt, axis=1) / tot
                comp = nxt / tot[:, None]
                dcomp = dnxt / tot[:, None] - comp * loc_slope[:, None]
                if t >= burn_steps:
                    inc = np.log(tot) - cv_weight * np.sum(hosts[t] - 1.0, axis=1)
                    cum_log += inc
                    cum_slope += loc_slope
                    if t >= half_start:
                        cum_half += inc
    else:
        ln2 = math.log(2.0)
        for t in range(n_total):
            m_t = model.expectation_matrix(hosts[t], tilt_y=tilt_y)
            # Since M(h, y) = U(h) + e^y F(h), M(h, y + ln 2) - M(h, y) == e^y F(h) == dM/dy
            dm_t = model.expectation_matrix(hosts[t], tilt_y=tilt_y + ln2) - m_t
            nxt = np.einsum("rij,rj->ri", m_t, comp)
            dnxt = np.einsum("rij,rj->ri", m_t, dcomp) + np.einsum("rij,rj->ri", dm_t, comp)
            tot = np.sum(nxt, axis=1)
            if np.any(tot <= 0.0) or np.any(nxt < -1e-12):
                raise FloatingPointError("Invalid conditional-mean state vector.")
            loc_slope = np.sum(dnxt, axis=1) / tot
            comp = nxt / tot[:, None]
            dcomp = dnxt / tot[:, None] - comp * loc_slope[:, None]
            if t >= burn_steps:
                inc = np.log(tot) - cv_weight * np.sum(hosts[t] - 1.0, axis=1)
                cum_log += inc
                cum_slope += loc_slope
                if t >= half_start:
                    cum_half += inc

    return (
        _fold_antithetic(cum_log / rec_steps, n_pairs),
        _fold_antithetic(cum_slope / rec_steps, n_pairs),
        _fold_antithetic(cum_half / half_steps, n_pairs),
    )


def averaged_rate_critical_tilt(model: WorldModel, host_grid: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Compute critical birth tilt of the arithmetic-mean-rate operator E[M(H, y)]."""
    hg = np.asarray(host_grid, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if hg.ndim != 2 or w.shape != (hg.shape[0],) or np.any(w < 0.0) or not np.isclose(np.sum(w), 1.0, atol=1e-12):
        raise ValueError("Invalid quadrature host_grid or weights.")
    m0 = np.einsum("q,qij->ij", w, model.expectation_matrix(hg, tilt_y=0.0))
    f0 = np.einsum("q,qij->ij", w, model.expectation_matrix(hg, tilt_y=math.log(2.0))) - m0
    u0 = m0 - f0

    def log_rho(y: float) -> float:
        vals = linalg.eigvals(u0 + math.exp(y) * f0)
        return math.log(float(np.max(vals.real)))

    y_avg = float(optimize.brentq(log_rho, -20.0, 20.0, xtol=1e-13))
    return {"avg_rate_tilt": y_avg, "mean_host_tilt": model.critical_tilt_mean_host()}


def solve_threshold_D(
    model: WorldModel,
    hosts: np.ndarray,
    burn_steps: int = 0,
    n_pairs: int = 0,
    cv_weight: float = 0.0,
    init_offset: float | None = None,
    avg_rate_ref: dict[str, float] | None = None,
    tol: float = 1e-10,
    max_iter: int = 15,
) -> dict[str, object]:
    """Solve sample zero-invasion-growth birth tilt y* and return D = y* - y_ref."""
    if (model.saturating or not model.is_rate_affine) and avg_rate_ref is None:
        raise ValueError("Curved contact/response requires an explicit arithmetic-mean-rate reference.")
    y_mean_host = model.critical_tilt_mean_host()
    y_ref = y_mean_host if avg_rate_ref is None else float(avg_rate_ref["avg_rate_tilt"])
    y = y_ref if init_offset is None else (y_ref + float(init_offset))

    step_corr = math.inf
    r_vec = slope_vec = half_vec = np.zeros(1)
    b_mean = 1.0
    used_iters = 0
    for it in range(1, max_iter + 1):
        used_iters = it
        r_vec, slope_vec, half_vec = propagate_invasion_products(
            model, hosts, tilt_y=y, burn_steps=burn_steps, n_pairs=n_pairs, cv_weight=cv_weight
        )
        b_mean = float(np.mean(slope_vec))
        if b_mean <= 0.0:
            raise RuntimeError("Nonpositive birth sensitivity during Newton root solve.")
        step_corr = float(np.mean(r_vec)) / b_mean
        if abs(step_corr) <= tol:
            break
        y -= float(np.clip(step_corr, -2.0, 2.0))
    else:
        if abs(step_corr) > 1e-7:
            raise RuntimeError(f"Threshold Newton solver failed to converge (residual={abs(step_corr):.3e}).")

    d_samples = (y - y_ref) - r_vec / b_mean
    half_root_diff = (half_vec - r_vec) / b_mean
    return {
        "D": float(y - y_ref),
        "se": sample_se(d_samples),
        "samples": d_samples,
        "root_residual": abs(step_corr),
        "iterations": used_iters,
        "slope": b_mean,
        "half_diff": float(np.mean(half_root_diff)),
        "half_se": sample_se(half_root_diff),
        "baseline_tilt": y_ref,
        "mean_host_tilt": y_mean_host,
    }


# ---------------------------------------------------------------------------
# Independent analytical reference formulas for Gates (i), (iii), (iv), (v)
# ---------------------------------------------------------------------------


def exact_ring_small_noise_K(mean_stays: Sequence[float], phi: float) -> float:
    """Analytical small-noise threshold coefficient K for a primitive ring with one noisy transfer."""
    taus = np.asarray(mean_stays, dtype=np.float64)
    if np.all(taus == 1.0):
        raise ValueError("All-unit ring is periodic; small-noise mixing formula does not apply.")
    total_T = float(np.sum(taus))
    if phi == 0.0:
        return 1.0 / (2.0 * total_T)
    theta = phi / (1.0 - phi)
    ret = float(np.prod(phi / (taus + (1.0 - taus) * phi)))
    return (1.0 + 2.0 * theta) / (2.0 * total_T) - ret / (1.0 - ret)


def exact_matrix_resolvent_K(model: WorldModel, phi: float) -> float:
    """Matrix perturbation coefficient C / b at constant-host criticality for one unit-variance driver."""
    y0 = model.critical_tilt_mean_host()
    _, v, w = model.perron_at_mean_host(y0)
    A = model.expectation_matrix(None, tilt_y=y0)
    B = model.U1[0] + math.exp(y0) * model.F1[0]
    Q = np.eye(model.dim) - np.outer(v, w)
    c_first = float(w @ B @ v)
    resolvent_term = 0.0
    if phi != 0.0:
        rhs = Q @ (B @ v)
        sol = linalg.solve(np.eye(model.dim) - phi * (A @ Q), rhs)
        resolvent_term = phi * float(w @ B @ sol)
    C_step = 0.5 * c_first * c_first - resolvent_term
    b_birth = float(w @ (math.exp(y0) * model.F0) @ v)
    return C_step / b_birth


def roerdink_exact_gamma_D(stay_prob_s: float, shape_p: float, refined: bool = False) -> float:
    """Exact threshold D for 2-stage [1, 1/(1-s)] ring under i.i.d. Gamma(p, 1/p) hosts.

    Derived from W_t = s + c * h_t / W_{t-1} with c = (1-s)*e^y:
    stationary X = W - s has density proportional to x^{p-1} (s+x)^{-p} exp(-(s*p/c)*x),
    so in log-variable z = log x, weight(z) = exp(p*z - p*log(s + e^z) - (s*p/c)*e^z)
    and long-run growth is E[log(s + e^z)].
    """
    if not (0.0 < stay_prob_s < 1.0) or shape_p <= 0.0:
        raise ValueError("Require 0 < s < 1 and p > 0.")
    log_s = math.log(stay_prob_s)

    def mean_log_growth(y: float) -> float:
        c = (1.0 - stay_prob_s) * math.exp(y)
        # Mode solves x*(s + x) = c
        z_mode = math.log((2.0 * c) / (stay_prob_s + math.sqrt(stay_prob_s * stay_prob_s + 4.0 * c)))

        def log_w(z: float) -> float:
            return shape_p * z - shape_p * np.logaddexp(log_s, z) - (stay_prob_s * shape_p / c) * math.exp(z)

        peak = log_w(z_mode)
        span_scale = 2.0 if refined else 1.0
        left_span = max(65.0, 50.0 / shape_p) * span_scale
        right_span = max(14.0, math.log(120.0 / shape_p)) * span_scale
        quad_tol = 1e-12 if refined else 1e-9

        def w_shifted(z: float) -> float:
            return math.exp(log_w(z) - peak)

        def num_integrand(z: float) -> float:
            return math.exp(log_w(z) - peak) * float(np.logaddexp(log_s, z))

        z_lo = z_mode - left_span
        z_hi = z_mode + right_span
        denom = (
            integrate.quad(w_shifted, z_lo, z_mode, epsabs=quad_tol, epsrel=quad_tol, limit=200)[0]
            + integrate.quad(w_shifted, z_mode, z_hi, epsabs=quad_tol, epsrel=quad_tol, limit=200)[0]
        )
        numer = (
            integrate.quad(num_integrand, z_lo, z_mode, epsabs=quad_tol, epsrel=quad_tol, limit=200)[0]
            + integrate.quad(num_integrand, z_mode, z_hi, epsabs=quad_tol, epsrel=quad_tol, limit=200)[0]
        )
        return numer / denom

    y_upper = math.log(shape_p) - float(special.digamma(shape_p))
    return float(optimize.brentq(mean_log_growth, 0.0, y_upper, xtol=1e-12))


def memA_two_stage_coefficient(tau_steady: float, phi: float) -> float:
    """Analytical memA coefficient K(1, tau; phi) = q*(1 + 2*phi/(1 + q*phi)) / (2*(1 + q)), q = 1/tau."""
    q = 1.0 / float(tau_steady)
    return (q / (2.0 * (1.0 + q))) * (1.0 + (2.0 * phi) / (1.0 + q * phi))


def multitype_branching_extinction_curve(
    mean_stays: tuple[float, float],
    log_r0: float,
    init_counts: tuple[int, int],
    max_horizon: int,
) -> np.ndarray:
    """Exact PGF recursion for cumulative extinction probability P(N_t = 0) of a 2-stage Poisson ring."""
    tau0, tau1 = float(mean_stays[0]), float(mean_stays[1])
    s0, q0 = 1.0 - 1.0 / tau0, 1.0 / tau0
    s1, q1 = 1.0 - 1.0 / tau1, 1.0 / tau1
    r0 = math.exp(log_r0)
    n0, n1 = int(init_counts[0]), int(init_counts[1])

    u0, u1 = 0.0, 0.0
    curve = np.zeros(max_horizon + 1, dtype=np.float64)
    curve[0] = (u0**n0) * (u1**n1)
    for t in range(1, max_horizon + 1):
        nxt_u0 = s0 * u0 + q0 * math.exp(u1 - 1.0)
        nxt_u1 = s1 * u1 + q1 * math.exp(r0 * (u0 - 1.0))
        u0, u1 = nxt_u0, nxt_u1
        curve[t] = (u0**n0) * (u1**n1)
    return curve
