"""Independent discrete-time stochastic life-cycle matrix solver for envcov replication.

Implements from first principles (zero imports from original scripts or studies/engine-R16):
  - Two-stage (and P-patch island) Borrelia cycles with memoryless, serial-chain, or fixed stays,
    optional contact-limited waiting on bounded two-state hosts, alternative host share, dilution,
    and correlated Gaussian AR(1) (host, weather) drivers;
  - Exact deterministic R0 = rho(F (I - U)^{-1}), criticality calibration, and graph periodicity;
  - Closed-form serial-chain D2 and general matrix-resolvent second-order D2;
  - Exact visit-weighted edge Jensen upper bound J;
  - Newton-Raphson common-random-number antithetic solver for finite-noise loss D (compiled C kernel
    validated against pure-NumPy reference);
  - Coupled annual weather-bin residence-time and mortality 2x2 engine for Part 4.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from math import gcd
import math
from pathlib import Path
import subprocess
import numpy as np
from scipy import linalg, signal, special

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
_CLIB = None


def _get_clib():
    global _CLIB
    if _CLIB is not None:
        return _CLIB
    c_src = HERE / "_matprod.c"
    so_out = HERE / "_matprod.so"
    if (not so_out.exists()) or (so_out.stat().st_mtime < c_src.stat().st_mtime):
        subprocess.run(
            ["cc", "-O3", "-shared", "-fPIC", str(c_src), "-o", str(so_out)],
            check=True,
            capture_output=True,
        )
    lib = ctypes.CDLL(str(so_out))
    lib.eval_general_products.argtypes = [ctypes.c_int] * 6 + [ctypes.c_void_p] * 8 + [ctypes.c_double] + [ctypes.c_void_p] * 2
    lib.eval_general_products.restype = ctypes.c_int
    lib.eval_coupled_2x2.argtypes = (
        [ctypes.c_int] * 3
        + [ctypes.c_double] * 2
        + [ctypes.c_void_p] * 5
        + [ctypes.c_double]
        + [ctypes.c_void_p] * 2
    )
    lib.eval_coupled_2x2.restype = ctypes.c_int
    _CLIB = lib
    return lib


def closed_form_serial_d2(
    mean_stays: list[float],
    substage_counts: list[int],
    sigma: float,
    phi: float,
    kappa: float = 1.0,
) -> float:
    """Exact analytical second-order loss D_2 for one lognormal edge in a serial chain."""
    t0 = float(sum(mean_stays))
    amp2 = (kappa * sigma) ** 2
    if abs(phi) < 1e-15:
        return amp2 / (2.0 * t0)
    theta = phi / (1.0 - phi)
    ret_gf = 1.0
    for tau, m in zip(mean_stays, substage_counts):
        ret_gf *= (1.0 + tau / (m * theta)) ** (-m)
    return amp2 * ((1.0 + 2.0 * theta) / (2.0 * t0) - ret_gf / (1.0 - ret_gf))


def match_discrete_chain_substages(mean_steps: float, cv: float) -> tuple[int, int]:
    """Protocol Amendment 1 discrete-step CV-matched substage count m_disc and continuous m_cont."""
    m_cap = max(1, int(math.floor(mean_steps)))
    if cv <= 0.0:
        return m_cap, m_cap
    m_disc = max(1, min(int(round(1.0 / (cv * cv + 1.0 / mean_steps))), m_cap))
    m_cont = max(1, min(int(round(1.0 / (cv * cv))), m_cap))
    return m_disc, m_cont


def _perron_pair(mat: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    vals, left_vecs, right_vecs = linalg.eig(mat, left=True, right=True)
    idx = int(np.argmax(vals.real))
    lam = float(vals[idx].real)
    v = np.abs(right_vecs[:, idx].real)
    w = np.abs(left_vecs[:, idx].real)
    v /= v.sum()
    w /= float(w @ v)
    return lam, v, w


def _graph_period(mat: np.ndarray) -> int:
    n = mat.shape[0]
    adj = mat > 0.0
    dist = [-1] * n
    dist[0] = 0
    queue = [0]
    for u in queue:
        for v in np.flatnonzero(adj[:, u]):
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(int(v))
    per = 0
    rows, cols = np.nonzero(adj)
    for r, c in zip(rows, cols):
        per = gcd(per, abs(dist[c] + 1 - dist[r]))
    return max(1, per)


def sample_gaussian_ar1(
    sigmas: np.ndarray,
    phis: np.ndarray,
    corr: np.ndarray,
    n_steps: int,
    n_pairs: int,
    seed: int,
    antithetic: bool = True,
    standardized: bool = False,
) -> np.ndarray:
    """Stationary causal vector AR(1) Gaussian / mean-1 lognormal sampler."""
    d = len(sigmas)
    rng = np.random.default_rng(seed)
    vals0, vecs0 = linalg.eigh(corr)
    if vals0.min(initial=0.0) < -1e-10:
        raise ValueError("Correlation matrix not PSD")
    root0 = vecs0 * np.sqrt(np.maximum(vals0, 0.0))

    innov_cov = corr * (1.0 - phis[:, None] * phis[None, :])
    vals_e, vecs_e = linalg.eigh(innov_cov)
    if vals_e.min(initial=0.0) < -1e-10:
        raise ValueError("AR(1) innovation covariance not PSD")
    root_e = vecs_e * np.sqrt(np.maximum(vals_e, 0.0))

    z0 = rng.standard_normal((n_pairs, d)) @ root0.T
    eps = rng.standard_normal((n_steps, n_pairs, d)) @ root_e.T
    if antithetic:
        z0 = np.concatenate([z0, -z0], axis=0)
        eps = np.concatenate([eps, -eps], axis=1)

    n_tot = z0.shape[0]
    out = np.empty((n_steps, n_tot, d), dtype=float)
    for j in range(d):
        if abs(phis[j]) < 1e-15:
            z = eps[:, :, j]
        else:
            z, _ = signal.lfilter([1.0], [1.0, -phis[j]], eps[:, :, j], axis=0, zi=(phis[j] * z0[:, j])[None, :])
        if standardized:
            out[:, :, j] = z
        else:
            s = sigmas[j]
            out[:, :, j] = np.exp(s * z - 0.5 * s * s)
    return out


def sample_bounded_two_state_ar1(
    sigma: float,
    phi: float,
    low: float,
    n_steps: int,
    n_pairs: int,
    seed: int,
    antithetic: bool = True,
    n_indep_drivers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-1 two-state Markov chain matching lognormal raw variance exp(sigma^2) - 1."""
    var = math.expm1(sigma * sigma)
    if var == 0.0:
        levels = np.array([1.0, 1.0])
        probs = np.array([0.5, 0.5])
    else:
        a = 1.0 - low
        high = 1.0 + var / a
        p_high = (a * a) / (a * a + var)
        levels = np.array([low, high], dtype=float)
        probs = np.array([1.0 - p_high, p_high], dtype=float)

    rng = np.random.default_rng(seed)
    n_tot = 2 * n_pairs if antithetic else n_pairs
    out = np.empty((n_steps, n_tot, n_indep_drivers), dtype=float)
    for d_idx in range(n_indep_drivers):
        u_draw = rng.random((n_steps, n_pairs))
        u_ref = rng.random((n_steps, n_pairs))
        if antithetic:
            u_draw = np.concatenate([u_draw, 1.0 - u_draw], axis=1)
            u_ref = np.concatenate([u_ref, 1.0 - u_ref], axis=1)
        if phi >= 0.0:
            states_fresh = (u_draw >= probs[0]).astype(np.int64)
            refresh = u_ref < (1.0 - phi)
            refresh[0, :] = True
            last_t = np.maximum.accumulate(np.where(refresh, np.arange(n_steps)[:, None], 0), axis=0)
            st = states_fresh[last_t, np.arange(n_tot)[None, :]]
        else:
            st = np.empty((n_steps, n_tot), dtype=np.int64)
            st[0, :] = (u_draw[0, :] >= probs[0]).astype(np.int64)
            for t in range(1, n_steps):
                p1 = phi * st[t - 1, :] + (1.0 - phi) * probs[1]
                st[t, :] = (u_draw[t, :] < p1).astype(np.int64)
        out[:, :, d_idx] = levels[st]
    return out, levels, probs


@dataclass
class BorreliaSpec:
    tau_mouse: float
    tau_tick: float
    sigma_mouse: float
    phi_mouse: float = 0.0
    kappa1: float = 1.0
    kappa2: float = 0.0
    mouse_shape: str = "memoryless"
    mouse_substages: int = 1
    tick_shape: str = "memoryless"
    tick_substages: int = 1
    tick_death: float = 0.0
    contact_tick: bool = False
    driver_kind: str = "lognormal"
    bounded_low: float = 0.2
    sigma_weather: float = 0.0
    phi_weather: float = 0.0
    rho_hw: float = 0.0
    alt_host_share: float = 0.0  # 0.0 means single host; 0.5 means equal independent alt host
    n_patches: int = 1
    patch_move: float = 0.0


class IndependentCycleModel:
    """Clean-room discrete-time Borrelia matrix model."""

    def __init__(self, spec: BorreliaSpec, birth_scale: float = 1.0):
        self.spec = spec
        self.birth_scale = float(birth_scale)
        if spec.tau_mouse < 1.0 or spec.tau_tick < 1.0:
            raise ValueError("Stage mean stay must be >= 1 step")

        # Micro-state layout within one patch: mouse substages then tick substages
        def stage_micro(mean_stay: float, shape: str, m_sub: int):
            if shape == "memoryless":
                return 1, 1.0 / mean_stay
            if shape == "chain":
                if m_sub < 1 or mean_stay < m_sub:
                    raise ValueError("Invalid chain substages")
                return int(m_sub), float(m_sub) / mean_stay
            if shape == "fixed":
                k_int = int(round(mean_stay))
                if abs(k_int - mean_stay) > 1e-10:
                    raise ValueError("Fixed delay must be integer")
                return k_int, 1.0
            raise ValueError(f"Unknown shape {shape}")

        km, hm = stage_micro(spec.tau_mouse, spec.mouse_shape, spec.mouse_substages)
        kt, ht = stage_micro(spec.tau_tick, spec.tick_shape, spec.tick_substages)
        self.km = km
        self.kt = kt
        self.k_local = km + kt
        self.n_patches = int(spec.n_patches)
        self.dim = self.n_patches * self.k_local

        # Drivers per patch
        self.driver_names = []
        self.driver_sigmas = []
        self.driver_phis = []
        base_d_names = ["mouse"]
        base_sig = [spec.sigma_mouse]
        base_phi = [spec.phi_mouse]
        base_corr = [[1.0]]
        if spec.alt_host_share > 0.0:
            base_d_names.append("alt")
            base_sig.append(spec.sigma_mouse)
            base_phi.append(spec.phi_mouse)
            base_corr = [[1.0, 0.0], [0.0, 1.0]]
        if spec.sigma_weather > 0.0:
            base_d_names.append("weather")
            base_sig.append(spec.sigma_weather)
            base_phi.append(spec.phi_weather)
            if len(base_d_names) == 2:
                base_corr = [[1.0, spec.rho_hw], [spec.rho_hw, 1.0]]
            else:
                base_corr = [[1.0, 0.0, spec.rho_hw], [0.0, 1.0, spec.rho_hw], [spec.rho_hw, spec.rho_hw, 1.0]]

        nd_local = len(base_d_names)
        for p in range(self.n_patches):
            for nm, sg, ph in zip(base_d_names, base_sig, base_phi):
                self.driver_names.append(f"p{p}_{nm}")
                self.driver_sigmas.append(sg)
                self.driver_phis.append(ph)
        self.driver_sigmas = np.array(self.driver_sigmas, dtype=float)
        self.driver_phis = np.array(self.driver_phis, dtype=float)
        self.corr = np.kron(np.eye(self.n_patches), np.array(base_corr, dtype=float))
        self.n_drivers = len(self.driver_names)

        # Bounded two-state marginal levels/probs if driver_kind == 'bounded'
        if spec.driver_kind == "bounded":
            var = math.expm1(spec.sigma_mouse ** 2)
            if var == 0.0:
                self.b_levels = np.array([1.0, 1.0])
                self.b_probs = np.array([0.5, 0.5])
            else:
                a = 1.0 - spec.bounded_low
                high = 1.0 + var / a
                ph = (a * a) / (a * a + var)
                self.b_levels = np.array([spec.bounded_low, high], dtype=float)
                self.b_probs = np.array([1.0 - ph, ph], dtype=float)
        else:
            self.b_levels = None
            self.b_probs = None

        # Modifier keys: (driver_idx, elasticity)
        self.mod_keys: list[tuple[int, float]] = []

        def get_mod_idx(d_idx: int, eta: float) -> int:
            item = (int(d_idx), float(eta))
            if item not in self.mod_keys:
                self.mod_keys.append(item)
            return self.mod_keys.index(item)

        # Build pre-movement demographic events: (src, dst_pre, prob_mean, coeff_dict, is_birth)
        demo_events = []
        for p in range(self.n_patches):
            off = p * self.k_local
            d_mouse = p * nd_local + 0
            d_alt = (p * nd_local + base_d_names.index("alt")) if "alt" in base_d_names else None
            d_weath = (p * nd_local + base_d_names.index("weather")) if "weather" in base_d_names else None

            # Mouse micro-states: off + 0 .. off + km - 1
            for q in range(km):
                src = off + q
                stay_m = 1.0 - hm
                if stay_m > 0.0:
                    demo_events.append((src, src, stay_m, {}, False))
                if q < km - 1:
                    demo_events.append((src, src + 1, hm, {}, False))
                else:
                    # Mouse -> tick birth transfer
                    dst_tick0 = off + km
                    y_mult = hm * self.birth_scale
                    if d_weath is not None and abs(spec.kappa2) < 1e-12:
                        mi = get_mod_idx(d_weath, 1.0)
                        demo_events.append((src, dst_tick0, y_mult, {mi: 1.0}, True))
                    elif abs(spec.kappa2) > 1e-12:
                        mi = get_mod_idx(d_mouse, spec.kappa2)
                        demo_events.append((src, dst_tick0, y_mult, {mi: 1.0}, True))
                    else:
                        demo_events.append((src, dst_tick0, y_mult, {}, True))

            # Tick micro-states: off + km .. off + km + kt - 1
            for q in range(kt):
                src = off + km + q
                d_t = spec.tick_death
                if spec.contact_tick:
                    if kt != 1 or ht <= d_t or spec.driver_kind != "bounded":
                        raise ValueError("Contact tick requires memoryless tick, ht > death, bounded driver")
                    c_rate = ht - d_t
                    stay_t = 1.0 - ht
                    mi_m = get_mod_idx(d_mouse, spec.kappa1)
                    # Check positivity at high state
                    norm_m = float(self.b_probs @ (self.b_levels ** spec.kappa1))
                    max_r = float(np.max(self.b_levels ** spec.kappa1) / norm_m)
                    if stay_t - c_rate * (max_r - 1.0) <= 1e-12:
                        raise ValueError("Contact staying probability nonpositive at high host state")
                    demo_events.append((src, src, stay_t, {mi_m: -c_rate / stay_t}, False))
                    demo_events.append((src, off + 0, c_rate, {mi_m: 1.0}, False))
                else:
                    stay_t = (1.0 - d_t) * (1.0 - ht)
                    if stay_t > 0.0:
                        demo_events.append((src, src, stay_t, {}, False))
                    if q < kt - 1:
                        demo_events.append((src, src + 1, (1.0 - d_t) * ht, {}, False))
                    else:
                        exit_base = (1.0 - d_t) * ht
                        if d_alt is not None:
                            s1 = 1.0 - spec.alt_host_share
                            s2 = spec.alt_host_share
                            mi1 = get_mod_idx(d_mouse, spec.kappa1)
                            mi2 = get_mod_idx(d_alt, spec.kappa1)
                            demo_events.append((src, off + 0, exit_base * s1, {mi1: 1.0}, False))
                            demo_events.append((src, off + 0, exit_base * s2, {mi2: 1.0}, False))
                        else:
                            if abs(spec.kappa1) > 1e-12 and spec.sigma_mouse > 0.0:
                                mi1 = get_mod_idx(d_mouse, spec.kappa1)
                                demo_events.append((src, off + 0, exit_base, {mi1: 1.0}, False))
                            else:
                                demo_events.append((src, off + 0, exit_base, {}, False))

        # Island movement matrix M (tick substages mobile when n_patches > 1)
        move_mat = np.eye(self.dim, dtype=float)
        if self.n_patches > 1 and spec.patch_move > 0.0:
            w_isl = np.ones((self.n_patches, self.n_patches), dtype=float) / self.n_patches
            k_isl = (1.0 - spec.patch_move) * np.eye(self.n_patches) + spec.patch_move * w_isl
            for q in range(kt):
                ids = np.arange(self.n_patches) * self.k_local + (km + q)
                move_mat[np.ix_(ids, ids)] = k_isl

        # Compose post-movement edges
        self.edges = []
        for src, dst_pre, base_val, coeff_map, is_birth in demo_events:
            for dst_post in np.flatnonzero(move_mat[:, dst_pre]):
                mv = float(move_mat[dst_post, dst_pre])
                self.edges.append((int(src), int(dst_post), base_val * mv, dict(coeff_map), bool(is_birth)))

        n_mods = len(self.mod_keys)
        self.U_mean = np.zeros((self.dim, self.dim), dtype=float)
        self.F_mean = np.zeros((self.dim, self.dim), dtype=float)
        self.U_mod = np.zeros((n_mods, self.dim, self.dim), dtype=float)
        self.F_mod = np.zeros((n_mods, self.dim, self.dim), dtype=float)

        for src, dst, val, cmap, is_birth in self.edges:
            if is_birth:
                self.F_mean[dst, src] += val
                for mi, cval in cmap.items():
                    self.F_mod[mi, dst, src] += val * cval
            else:
                self.U_mean[dst, src] += val
                for mi, cval in cmap.items():
                    self.U_mod[mi, dst, src] += val * cval

        self.A_mean = self.U_mean + self.F_mean
        self.period = _graph_period(self.A_mean)
        self.lam, self.v, self.w = _perron_pair(self.A_mean)
        self.b_rate = float((self.w @ self.F_mean @ self.v) / self.lam)
        ngm = linalg.solve((np.eye(self.dim) - self.U_mean).T, self.F_mean.T).T
        self.R0 = float(np.max(np.abs(linalg.eigvals(ngm))))

    def critical(self) -> "IndependentCycleModel":
        if abs(self.R0 - 1.0) < 1e-12:
            return self
        return IndependentCycleModel(self.spec, self.birth_scale / self.R0)

    def second_order_d2(self) -> float | None:
        """Exact matrix-resolvent second-order threshold loss D_2."""
        m = self.critical()
        if m.period != 1:
            return None
        B = np.zeros((m.n_drivers, m.dim, m.dim), dtype=float)
        for mi, (di, eta) in enumerate(m.mod_keys):
            B[di] += (m.U_mod[mi] + m.F_mod[mi]) * eta * m.driver_sigmas[di] / m.lam
        Q = np.eye(m.dim) - np.outer(m.v, m.w)
        T = (m.A_mean / m.lam) @ Q
        c_vec = np.einsum("i,dij,j->d", m.w, B, m.v)
        C = 0.5 * float(c_vec @ m.corr @ c_vec)
        for i in range(m.n_drivers):
            phi_i = m.driver_phis[i]
            if abs(phi_i) > 1e-15:
                rhs = Q @ np.einsum("d,dij,j->i", m.corr[i], B, m.v)
                sol = linalg.solve(np.eye(m.dim) - phi_i * T, rhs)
                C -= phi_i * float(m.w @ B[i] @ sol)
        return float(C / m.b_rate)

    def jensen_bound(self) -> float:
        """Exact visit-weighted edge Jensen upper bound J = sum_e f_e penalty_e / b."""
        m = self.critical()
        j_step = 0.0
        for src, dst, val, cmap, _ in m.edges:
            freq = float(m.w[dst] * val * m.v[src] / m.lam)
            if not cmap:
                continue
            mi, coeff = next(iter(cmap.items()))
            di, eta = m.mod_keys[mi]
            if m.spec.driver_kind == "lognormal" or di >= m.n_patches * (2 if m.spec.alt_host_share > 0 else 1):
                # Lognormal driver with coeff == 1
                sig = m.driver_sigmas[di]
                pen = 0.5 * (eta * sig) ** 2
            else:
                norm_val = float(m.b_probs @ (m.b_levels ** eta))
                mults = 1.0 + coeff * ((m.b_levels ** eta) / norm_val - 1.0)
                pen = -float(m.b_probs @ np.log(mults))
            j_step += freq * pen
        return float(j_step / m.b_rate)

    def _sample_mod_centered(self, n_steps: int, n_pairs: int, seed: int, antithetic: bool = True) -> np.ndarray:
        """Return array (n_steps, 2*n_pairs, n_mods) of (r_{t,m} - 1)."""
        n_mods = len(self.mod_keys)
        n_tot = 2 * n_pairs if antithetic else n_pairs
        if n_mods == 0:
            return np.zeros((n_steps, n_tot, 0), dtype=float)

        if self.spec.driver_kind == "lognormal":
            hosts = sample_gaussian_ar1(
                self.driver_sigmas,
                self.driver_phis,
                self.corr,
                n_steps,
                n_pairs,
                seed,
                antithetic=antithetic,
                standardized=False,
            )
            out = np.empty((n_steps, n_tot, n_mods), dtype=float)
            for mi, (di, eta) in enumerate(self.mod_keys):
                sig = self.driver_sigmas[di]
                r_val = (hosts[:, :, di] ** eta) * math.exp(0.5 * (eta - eta * eta) * sig * sig)
                out[:, :, mi] = r_val - 1.0
            return out

        # Bounded two-state host driver
        hosts, levels, probs = sample_bounded_two_state_ar1(
            self.spec.sigma_mouse,
            self.spec.phi_mouse,
            self.spec.bounded_low,
            n_steps,
            n_pairs,
            seed,
            antithetic=antithetic,
            n_indep_drivers=self.n_drivers,
        )
        out = np.empty((n_steps, n_tot, n_mods), dtype=float)
        for mi, (di, eta) in enumerate(self.mod_keys):
            norm_val = float(probs @ (levels ** eta))
            out[:, :, mi] = (hosts[:, :, di] ** eta) / norm_val - 1.0
        return out

    def eval_products_numpy(self, mod_centered: np.ndarray, y: float, burn: int, pairs: int) -> tuple[np.ndarray, np.ndarray]:
        """Pure-NumPy reference product routine for Rule 7 verification."""
        ey = math.exp(y)
        A0 = self.U_mean + ey * self.F_mean
        Ay0 = ey * self.F_mean
        Pm = self.U_mod + ey * self.F_mod
        Pym = ey * self.F_mod
        n_steps, n_reps, _ = mod_centered.shape

        x = np.tile(self.v, (n_reps, 1))
        if self.period > 1:
            x[:] = 0.0
            x[:, 0] = 1.0
        dx = np.zeros_like(x)
        logs = np.zeros(n_reps, dtype=float)
        slopes = np.zeros(n_reps, dtype=float)

        for t in range(n_steps):
            mc = mod_centered[t]
            At = A0[None, :, :] + np.einsum("rm,mij->rij", mc, Pm)
            Ayt = Ay0[None, :, :] + np.einsum("rm,mij->rij", mc, Pym)
            nx = np.einsum("rij,rj->ri", At, x)
            ndx = np.einsum("rij,rj->ri", At, dx) + np.einsum("rij,rj->ri", Ayt, x)
            s = nx.sum(axis=1)
            ds = ndx.sum(axis=1) / s
            x = nx / s[:, None]
            dx = ndx / s[:, None] - x * ds[:, None]
            if t >= burn:
                logs += np.log(s)
                slopes += ds
        logs /= float(n_steps - burn)
        slopes /= float(n_steps - burn)
        if pairs > 0:
            logs = 0.5 * (logs[:pairs] + logs[pairs:])
            slopes = 0.5 * (slopes[:pairs] + slopes[pairs:])
        return logs, slopes

    def eval_products_c(self, mod_centered: np.ndarray, y: float, burn: int, pairs: int) -> tuple[np.ndarray, np.ndarray]:
        """Fast C sparse product evaluation."""
        lib = _get_clib()
        nz_mask = (self.U_mean != 0) | (self.F_mean != 0)
        if len(self.mod_keys) > 0:
            nz_mask |= np.any(self.U_mod != 0, axis=0) | np.any(self.F_mod != 0, axis=0)
        rows, cols = np.nonzero(nz_mask)
        n_edges = len(rows)
        n_mods = len(self.mod_keys)
        n_steps, n_reps, _ = mod_centered.shape

        x_init = np.zeros(self.dim, dtype=np.float64) if self.period > 1 else self.v.astype(np.float64).copy()
        if self.period > 1:
            x_init[0] = 1.0

        row_c = np.ascontiguousarray(rows, dtype=np.int32)
        col_c = np.ascontiguousarray(cols, dtype=np.int32)
        u_c = np.ascontiguousarray(self.U_mean[nz_mask], dtype=np.float64)
        f_c = np.ascontiguousarray(self.F_mean[nz_mask], dtype=np.float64)
        um_c = np.ascontiguousarray(self.U_mod[:, nz_mask] if n_mods > 0 else np.zeros((1, max(n_edges, 1))), dtype=np.float64)
        fm_c = np.ascontiguousarray(self.F_mod[:, nz_mask] if n_mods > 0 else np.zeros((1, max(n_edges, 1))), dtype=np.float64)
        ms_c = np.ascontiguousarray(mod_centered, dtype=np.float64)
        logs = np.zeros(n_reps, dtype=np.float64)
        slopes = np.zeros(n_reps, dtype=np.float64)

        rc = lib.eval_general_products(
            int(n_steps),
            int(n_reps),
            int(self.dim),
            int(n_edges),
            int(n_mods),
            int(burn),
            row_c.ctypes.data,
            col_c.ctypes.data,
            u_c.ctypes.data,
            f_c.ctypes.data,
            um_c.ctypes.data,
            fm_c.ctypes.data,
            ms_c.ctypes.data,
            x_init.ctypes.data,
            float(y),
            logs.ctypes.data,
            slopes.ctypes.data,
        )
        if rc != 0:
            raise RuntimeError(f"C product kernel returned error code {rc}")
        if pairs > 0:
            logs = 0.5 * (logs[:pairs] + logs[pairs:])
            slopes = 0.5 * (slopes[:pairs] + slopes[pairs:])
        return logs, slopes

    def solve_loss(
        self,
        steps: int = 40000,
        burn: int = 2000,
        pairs: int = 24,
        seed: int = 1001,
        tol: float = 1e-8,
        max_iter: int = 14,
    ) -> dict[str, float | int | None]:
        """Solve stochastic threshold loss D via Newton-Raphson on common antithetic draws."""
        m = self.critical()
        mc = m._sample_mod_centered(steps + burn, pairs, seed, antithetic=True)
        r0, s0 = m.eval_products_c(mc, 0.0, burn, pairs)
        y = float(np.clip(-r0.mean() / s0.mean(), -3.0, 3.0))
        r_cur, s_cur = r0, s0
        for it in range(max_iter):
            r_cur, s_cur = m.eval_products_c(mc, y, burn, pairs)
            step = float(r_cur.mean() / s_cur.mean())
            if abs(step) < tol:
                break
            y -= float(np.clip(step, -2.0, 2.0))
        pair_ests = y - r_cur / float(s_cur.mean())
        return {
            "D": float(y),
            "se": float(pair_ests.std(ddof=1) / math.sqrt(pairs)),
            "D2": m.second_order_d2(),
            "J": m.jensen_bound(),
            "generation_steps": float(1.0 / m.b_rate),
            "period": int(m.period),
            "steps": int(steps),
            "burn": int(burn),
            "pairs": int(pairs),
            "seed": int(seed),
        }


def solve_coupled_weather_cell(
    sigma_mouse: float,
    kappa: float,
    gamma: float,
    site_climate: float,
    phi_mouse: float,
    phi_weather: float,
    rho_hw: float,
    weather_vals: list[float] | np.ndarray,
    duration_days: list[float] | np.ndarray,
    steps: int = 40000,
    burn: int = 2000,
    pairs: int = 24,
    seed: int = 5001,
    dt_days: float = 6.0,
    block_steps: int = 61,
) -> dict:
    """Independent 2x2 coupled weather-duration-mortality loss solver for Part 4."""
    lib = _get_clib()
    w_arr = np.asarray(weather_vals, dtype=float)
    d_arr = np.asarray(duration_days, dtype=float)
    order = np.argsort(w_arr)
    w_sorted = w_arr[order]
    dur_sorted = d_arr[order]
    k_bins = len(w_sorted)

    cuts = special.ndtri(np.arange(k_bins + 1, dtype=float) / k_bins)
    shift = kappa * sigma_mouse * rho_hw
    qh = np.diff(special.ndtr(cuts - shift))

    tau1 = 133.0 / dt_days
    a00 = 1.0 - 1.0 / tau1
    f10 = 1.0 / tau1

    u01_bins = np.empty(k_bins, dtype=np.float64)
    u11_bins = np.empty(k_bins, dtype=np.float64)
    state_r0 = np.empty(k_bins, dtype=np.float64)

    for k in range(k_bins):
        death_k = 1.0 - math.exp(-0.001 * dt_days * math.exp(-gamma * (site_climate + w_sorted[k])))
        tau2_k = dur_sorted[k] / dt_days
        h2_k = 1.0 / tau2_k
        u01_bins[k] = (1.0 - death_k) * h2_k
        u11_bins[k] = (1.0 - death_k) * (1.0 - h2_k)
        # Uncoupled state-k R0 = u01_k / (1 - u11_k)
        state_r0[k] = u01_bins[k] / (1.0 - u11_bins[k])

    u01_mean = float(qh @ u01_bins)
    u11_mean = float(np.mean(u11_bins))
    mean_r0 = u01_mean / (1.0 - u11_mean)
    f10_crit = f10 / mean_r0

    mean_mat = np.array([[a00, u01_mean], [f10_crit, u11_mean]], dtype=float)
    lam, v_init, _ = _perron_pair(mean_mat)
    if abs(lam - 1.0) > 1e-10:
        raise RuntimeError(f"Coupled critical Perron root {lam} != 1")

    tot_steps = steps + burn
    phase = seed % block_steps
    n_years = math.ceil((tot_steps + phase) / block_steps)
    z_yr = sample_gaussian_ar1(
        np.array([1.0, 1.0]),
        np.array([phi_mouse, phi_weather]),
        np.array([[1.0, rho_hw], [rho_hw, 1.0]]),
        n_years,
        pairs,
        seed,
        antithetic=True,
        standardized=True,
    )
    bin_yr = np.minimum((special.ndtr(z_yr[:, :, 1]) * k_bins).astype(np.int32), k_bins - 1)
    ks = kappa * sigma_mouse
    host_yr = np.exp(ks * z_yr[:, :, 0] - 0.5 * ks * ks)

    bin_step = np.ascontiguousarray(np.repeat(bin_yr, block_steps, axis=0)[phase : phase + tot_steps], dtype=np.int32)
    host_step = np.ascontiguousarray(np.repeat(host_yr, block_steps, axis=0)[phase : phase + tot_steps], dtype=np.float64)
    x_init = np.ascontiguousarray(v_init, dtype=np.float64)
    n_reps = 2 * pairs

    def run_y(y_val: float) -> tuple[np.ndarray, np.ndarray]:
        logs = np.zeros(n_reps, dtype=np.float64)
        slopes = np.zeros(n_reps, dtype=np.float64)
        rc = lib.eval_coupled_2x2(
            int(tot_steps),
            int(n_reps),
            int(burn),
            float(a00),
            float(f10_crit),
            u01_bins.ctypes.data,
            u11_bins.ctypes.data,
            bin_step.ctypes.data,
            host_step.ctypes.data,
            x_init.ctypes.data,
            float(y_val),
            logs.ctypes.data,
            slopes.ctypes.data,
        )
        if rc != 0:
            raise RuntimeError(f"eval_coupled_2x2 failed with {rc}")
        return 0.5 * (logs[:pairs] + logs[pairs:]), 0.5 * (slopes[:pairs] + slopes[pairs:])

    r0, s0 = run_y(0.0)
    y = float(np.clip(-r0.mean() / s0.mean(), -3.0, 3.0))
    r_cur, s_cur = r0, s0
    for _ in range(14):
        r_cur, s_cur = run_y(y)
        step = float(r_cur.mean() / s_cur.mean())
        if abs(step) < 1e-8:
            break
        y -= float(np.clip(step, -2.0, 2.0))
    ests = y - r_cur / float(s_cur.mean())
    return {
        "D": float(y),
        "se": float(ests.std(ddof=1) / math.sqrt(pairs)),
        "mean_R0": float(mean_r0),
        "state_R0": state_r0.tolist(),
        "qh": qh.tolist(),
        "cuts": cuts.tolist(),
    }
