#!/usr/bin/env python3
"""Independent high-precision verification of GAP 3b (`r37-gap3b-check`).

Author: independent 4 secondary (2026-11-15)
Rules compliance (`study rules` Rules 1, 5, 7, 9):
  - Zero imports from `gap3b/gap3b_checks.py` or any repository module.
  - Pre-registered known-result benchmark gates (`G1`–`G4`) run BEFORE any `gap3b`
    research grid evaluation; any gate failure aborts execution.
  - Completely independent deterministic high-precision numerical methods:
      (1) Long-stay thresholds D(s) and derivatives D'(s) near s = 1 (224-bit MPFR,
          cross-checked at 288-bit MPFR and two mesh/mode resolutions):
            * Gamma value-refresh: exact infinite-horizon constant-run resolvent
              (I - phi P_h)^(-1) on active scaled-ratio Gauss-Legendre panels
              u = s/U in [u_min(z), 1], solving R(z) = F(z) - z and exact adjoint
              R'(z) = F'(z) - 1 (zero Monte Carlo sampling, zero run truncation).
            * Lognormal AR(1): exact stationary Mehler-Hermite eigenfunction
              projection of the Ornstein-Uhlenbeck transition kernel on active
              Gauss-Legendre panels u = s/U in [u_min(z), 1], solving R(z) = F(z) - z
              and exact adjoint R'(z) = F'(z) - 1 (zero Monte Carlo sampling, zero
              finite-history U_0 = 1 truncation).
      (2) Small-staying thresholds D(10^-k) and rate coefficients at s = 10^-k
          for large k (up to k = 240 for exact i.i.d. gamma Tricomi and k = 40, 80, 160
          for correlated Gamma value-refresh and Lognormal AR(1)), with 2-point linear
          and 3-point quadratic Richardson extrapolation in 1/log(1/s):
            * Exact 80-digit Tricomi hypergeometric thresholds for i.i.d. gamma
              across p in {0.25, 0.5, 1, 2, 5} at s = 10^-k, k in {10, 30, 60, 120, 240},
              plus direct verification of explicit interval (4.7)-(4.9) at k = 4343
              (L0 ~ 10000).
            * Deterministic primal Bellman & adjoint invariant-functional Koopman
              Nystrom solvers on geometrically graded panels (zero Monte Carlo
              sampling error) for correlated Gamma refresh and Mehler-Hermite
              Lognormal AR(1) at s = 10^-k, k in {10, 20, 40, 80, 160} (L0 up to 368.4),
              verifying BOTH the exact stationary second-moment identity (4.4)
              and Richardson-extrapolated rate coefficients across interior and
              boundary-adjacent regimes (singular shape p = 0.25, p = 0.5; strong
              positive/negative memory phi = +-0.8, 0.85; larger variance w = 1.0).
      (3) Exhaustive 65-digit algebraic & pointwise audit of every constant, chain-rule
          identity, period-two counterexample, A=0 curve B < 0 certificate, and
          unbounded-tail inequality (4.3)-(4.9) across 13,500 exact test points.
"""
from __future__ import annotations

import ctypes
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import mpmath as mp
import numpy as np
from scipy.special import digamma, gammaln

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
C_SOLVER_PATH = HERE / "mpfr_solvers.c"
primary_DIR = (__import__('pathlib').Path(__file__).resolve().parent / '../../gap3b').resolve()

mp.mp.dps = 65
FAILURES: list[str] = []
PASS_COUNT = 0
RESULTS: dict[str, object] = {
    "gates": [],
    "algebra_and_pointwise": [],
    "derivative_intervals": [],
    "exact_gamma_rates": [],
    "correlated_rates": [],
}


def record_check(category: str, name: str, ok: bool, detail: str = "", payload: dict | None = None) -> None:
    global PASS_COUNT
    if ok:
        PASS_COUNT += 1
    else:
        FAILURES.append(name)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {detail}", flush=True)
    entry = {"name": name, "ok": bool(ok), "detail": detail}
    if payload:
        entry.update(payload)
    RESULTS[category].append(entry)


def fmt(x: mp.mpf, digits: int = 32) -> str:
    return mp.nstr(x, digits)


def log1pexp_mp(x: mp.mpf) -> mp.mpf:
    if x > 0:
        return x + mp.log1p(mp.exp(-x))
    return mp.log1p(mp.exp(x))


# ---------------------------------------------------------------------------
# Section A: Independent analytical constants from gap3b_proof.md §§2 & 4
# ---------------------------------------------------------------------------

def host_constants(model: str, par_str: str, phi_str: str) -> dict[str, mp.mpf]:
    """Compute exact 65-dps constants in gap3b_proof.md §§1, 2, 4 from scratch."""
    par = mp.mpf(par_str)
    phi = mp.mpf(phi_str)
    half = mp.mpf("0.5")
    if model == "gamma":
        p = par
        V = 1 / p
        c1 = V * phi
        c2 = V * (phi ** 2)
        m3 = 2 * (V ** 2)
        t12 = phi * m3
        t21 = phi * m3
        t111 = (phi ** 2) * m3
        A = V * (half + phi)
        B = -A - c2 - t12 - t21 - t111 - m3 / 3 + mp.mpf("1.5") * (A ** 2)
        eh4 = (p + 1) * (p + 2) * (p + 3) / (p ** 3)
        ell = mp.log(p) - mp.digamma(p)
        v_log = mp.polygamma(1, p)
        a_exp = p / 4
        g_pos = mp.gamma(p + 2 * a_exp) / mp.gamma(p) * mp.exp(-2 * a_exp * mp.digamma(p))
        g_neg = mp.gamma(p - 2 * a_exp) / mp.gamma(p) * mp.exp(2 * a_exp * mp.digamma(p))
        N_a = mp.exp(a_exp) * (g_pos + g_neg)
    elif model == "lognormal":
        w = par
        V = mp.expm1(w)
        c1 = mp.expm1(w * phi)
        c2 = mp.expm1(w * phi * phi)
        m3 = mp.exp(3 * w) - 3 * mp.exp(w) + 2
        T = mp.exp(w * (1 + 2 * phi)) - mp.exp(w) - 2 * mp.exp(w * phi) + 2
        U = mp.exp(w * (2 * phi + phi * phi)) - 2 * mp.exp(w * phi) - mp.exp(w * phi * phi) + 2
        A = V / 2 + c1
        B = -A - c2 - 2 * T - U - m3 / 3 + mp.mpf("1.5") * (A ** 2)
        eh4 = mp.exp(6 * w)
        ell = w / 2
        v_log = w
        sigma = mp.sqrt(w)
        a_exp = 1 / (2 * max(mp.mpf(1), sigma))
        N_a = 2 * mp.exp(a_exp + 2 * (a_exp ** 2) * w)
    else:
        raise ValueError(f"Unknown model: {model}")

    # Spectral moments a = mu_2, b = mu_3, d = 128 E[h^4]
    a_m2 = 3 + 2 * A
    b_m3 = 3 * (mp.mpf("1.5") * (A ** 2) + 5 * A + mp.mpf(10) / 3 - B)
    d_m4 = 128 * eh4

    z0 = min(mp.mpf("0.25"), 1 / (2 * a_m2))
    eps0 = z0 / 3
    M2 = 1 + a_m2
    M3 = 1 + 3 * a_m2 + 2 * b_m3
    M4 = 1 + 6 * a_m2 + 3 * (a_m2 ** 2) + 8 * b_m3 + 6 * d_m4
    T1 = mp.mpf("1.5") * M2 + 2
    T2 = M3 + mp.mpf("2.25") * (M2 ** 2) + 2 * a_m2
    T3 = mp.mpf("0.75") * M4 + mp.mpf("4.5") * M2 * M3 + mp.mpf("6.75") * (M2 ** 3) + 4 * b_m3
    K = 27 * T3 + 243 * T2 * M2 + T1 * (729 * (M2 ** 2) + 81 * M3)

    eps_A = min(eps0, abs(A) / (8 * (abs(B) + 1)), mp.sqrt(abs(A) / (2 * K))) if A != 0 else mp.mpf(0)
    eps_B = min(eps0, 2 * abs(B) / K) if B != 0 else mp.mpf(0)
    V_pi = v_log * (1 - phi) / (1 + phi)

    return {
        "A": A, "B": B, "a": a_m2, "b": b_m3, "d": d_m4, "eh4": eh4,
        "z0": z0, "eps0": eps0, "M2": M2, "M3": M3, "M4": M4,
        "T1": T1, "T2": T2, "T3": T3, "K": K,
        "eps_A": eps_A, "eps_B": eps_B,
        "ell": ell, "v_log": v_log, "V_pi": V_pi,
        "a_exp": a_exp, "N_a": N_a,
    }


def explicit_q_interval(L: mp.mpf, cc: dict[str, mp.mpf], phi: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf] | None:
    """Compute proved explicit interval (4.7) and deviation bound (4.9) at normalized L >= e."""
    a_exp = cc["a_exp"]
    N_a = cc["N_a"]
    V_pi = cc["V_pi"]
    b0 = max(mp.mpf(2), 4 / a_exp)
    K_cut = b0 * mp.log(L)
    Cg = 6 + 2 / (1 + phi)
    Cb = 5 + 2 / (1 + phi)
    Ct = 8 + 4 / (1 + phi)
    if 2 * L <= Cg * K_cut:
        return None
    delta_bound = (Cb * K_cut + 16 * Ct * N_a / (a_exp ** 2)) / (L ** 2)
    q_lo = max(mp.mpf(0), (V_pi - delta_bound) / (2 * L + Cg * K_cut))
    q_hi = (V_pi + delta_bound) / (2 * L - Cg * K_cut)
    rad_49 = (V_pi * Cg * K_cut + 2 * L * delta_bound) / (2 * L * (2 * L - Cg * K_cut))
    return q_lo, q_hi, rad_49


# ---------------------------------------------------------------------------
# Section B: Exact i.i.d. Gamma Tricomi solver & MPFR C library wrapper
# ---------------------------------------------------------------------------

def tricomi_L(z: mp.mpf, p: mp.mpf) -> mp.mpf:
    if z == 0:
        return mp.mpf(0)
    kap = p / z
    return mp.diff(lambda b: mp.log(mp.hyperu(p, b, kap)), mp.mpf(1))


def tricomi_D_and_Ds(eps: mp.mpf, p: mp.mpf, dps: int = 65) -> tuple[mp.mpf, mp.mpf]:
    """Exact threshold D(1-eps) and derivative D'(1-eps) for i.i.d. Gamma(p, 1/p) via Tricomi U."""
    with mp.workdps(dps + 15):
        eps = mp.mpf(eps)
        p = mp.mpf(p)
        s = 1 - eps
        target = -mp.log1p(-eps)
        ell = mp.log(p) - mp.digamma(p)
        z_lo = eps / (s * s)
        z_hi = eps * mp.exp(ell) / (s * s)
        z = mp.findroot(lambda u: tricomi_L(u, p) - target, (z_lo, z_hi),
                        solver="anderson", tol=mp.power(10, -(dps + 8)))
        Rv = tricomi_L(z, p) - z
        Fp = mp.diff(lambda u: tricomi_L(u, p), z)
        Rp = Fp - 1
        ep = s * Fp
        zm = (target - eps) - Rv
        D = mp.log1p(zm / eps) - 2 * target
        em = -eps * (-eps + Rp - eps * Rp) - zm * ep
        Ds = -(em / (z * eps) - 2 * Fp) / ep
    return +D, +Ds


def tricomi_q_at_L(L: mp.mpf, p: mp.mpf, dps: int = 75) -> mp.mpf:
    """Exact q_L = 2 r(e^-L, e^ell) for i.i.d. Gamma(p, 1/p) via Tricomi U(p, b, e^(psi(p)-2L))."""
    with mp.workdps(dps + 15):
        L = mp.mpf(L)
        p = mp.mpf(p)
        kap = mp.exp(mp.digamma(p) - 2 * L)
        val = 2 * (-L + mp.diff(lambda b: mp.log(mp.hyperu(p, b, kap)), mp.mpf(1)))
    return +val


def tricomi_threshold_at_L0(L0: mp.mpf, p: mp.mpf, dps: int = 75) -> tuple[mp.mpf, mp.mpf]:
    """Exact moving-threshold root m_* = -q_{L_*} and ell - D(s) at L0 = log(1/s) for i.i.d. Gamma."""
    with mp.workdps(dps + 15):
        L0 = mp.mpf(L0)
        p = mp.mpf(p)
        Ls = L0
        for _ in range(8):
            qv = tricomi_q_at_L(Ls, p, dps=dps + 5)
            Ls_new = L0 - qv / 2
            if abs(Ls_new - Ls) < mp.power(10, -(dps + 5)):
                Ls = Ls_new
                break
            Ls = Ls_new
        qv = tricomi_q_at_L(Ls, p, dps=dps + 5)
        s_val = mp.exp(-L0)
        ell_minus_D = qv + mp.log1p(-s_val)
    return +qv, +ell_minus_D


def gl_rule_mp(p_deg: int, dps: int = 65) -> tuple[list[mp.mpf], list[mp.mpf]]:
    """Interior Gauss-Legendre nodes on (-1, 1) and barycentric weights in mpmath."""
    with mp.workdps(dps + 15):
        nodes: list[mp.mpf] = []
        for j in range(p_deg):
            u = -mp.cos(mp.pi * (j + mp.mpf("0.75")) / (p_deg + mp.mpf("0.5")))
            for _ in range(12):
                p0, p1 = mp.mpf(1), u
                for k in range(1, p_deg):
                    p0, p1 = p1, ((2 * k + 1) * u * p1 - k * p0) / (k + 1)
                dp = p_deg * (u * p1 - p0) / (u * u - 1)
                du = -p1 / dp
                u += du
                if abs(du) < mp.power(10, -(dps + 10)):
                    break
            nodes.append(u)
        nodes.sort()
        bary = [1 / mp.fprod(nodes[j] - nodes[m] for m in range(p_deg) if m != j) for j in range(p_deg)]
    return [+u for u in nodes], [+b for b in bary]


class MPFRKoopmanLib:
    """Compiled C/MPFR wrapper for exact stationary resolvent & Mehler-Hermite long-stay solvers."""

    def __init__(self, tmp_dir: str) -> None:
        so_path = Path(tmp_dir) / "mpfr_solvers.dylib"
        subprocess.run(
            [
                "cc", "-Wall", "-Wextra", "-Werror", "-O3", "-dynamiclib",
                str(C_SOLVER_PATH), "-o", str(so_path),
                "-I/opt/homebrew/opt/mpfr/include", "-I/opt/homebrew/opt/gmp/include",
                "-L/opt/homebrew/opt/mpfr/lib", "-L/opt/homebrew/opt/gmp/lib",
                "-lmpfr", "-lgmp",
            ],
            check=True,
        )
        self.lib = ctypes.CDLL(str(so_path))
        c_pp = ctypes.POINTER(ctypes.c_char_p)
        self.lib.mehler_R_and_Rp.argtypes = (
            [ctypes.c_int] * 5 + [ctypes.c_char_p] + [c_pp] * 6 + [ctypes.c_void_p, ctypes.c_int]
        )
        self.lib.mehler_R_and_Rp.restype = None
        self.lib.gamma_resolvent_R_and_Rp.argtypes = (
            [ctypes.c_int] * 4 + [ctypes.c_char_p] * 2 + [c_pp] * 5 + [ctypes.c_void_p, ctypes.c_int]
        )
        self.lib.gamma_resolvent_R_and_Rp.restype = None
        self._gl_cache: dict[int, tuple[list[mp.mpf], list[mp.mpf]]] = {}

    def _gl(self, p_deg: int) -> tuple[list[mp.mpf], list[mp.mpf]]:
        if p_deg not in self._gl_cache:
            self._gl_cache[p_deg] = gl_rule_mp(p_deg, dps=75)
        return self._gl_cache[p_deg]

    @staticmethod
    def _pack(seq: list[mp.mpf], digits: int = 72):
        bs = [mp.nstr(v, digits).encode() for v in seq]
        return bs, (ctypes.c_char_p * len(bs))(*bs)

    def gamma_D_and_Ds(
        self, p_str: str, phi_str: str, eps: mp.mpf,
        n_quad: int = 135, n_pan: int = 3, p_deg: int = 12, bits: int = 224,
    ) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
        dps_work = max(65, int(bits * 0.30103) + 8)
        with mp.workdps(dps_work):
            p = mp.mpf(p_str)
            phi = mp.mpf(phi_str)
            eps = mp.mpf(eps)
            lg = mp.loggamma(p)
            u_lo, u_hi = mp.mpf("-6.8"), mp.mpf("5.2")
            hu = (u_hi - u_lo) / (n_quad - 1)
            hs, ws = [], []
            for k in range(n_quad):
                u = u_lo + k * hu
                eu = mp.exp(-u)
                t = u - eu
                dt = 1 + eu
                lw = p * t - mp.exp(t) - lg + mp.log(dt * hu)
                if lw > -155:
                    hs.append(mp.exp(t) / p)
                    ws.append(mp.exp(lw))
            sw = mp.fsum(ws)
            ws = [w / sw for w in ws]
            target = -mp.log1p(-eps)
            u_min = 1 / (1 + 3 * target * max(hs))
            edges = [u_min + (1 - u_min) * (1 - (mp.mpf(n_pan - j) / n_pan) ** 2) for j in range(n_pan + 1)]
            gl_u, bary = self._gl(p_deg)

            digs = dps_work - 2
            _bh, ca_h = self._pack(hs, digs)
            _bw, ca_w = self._pack(ws, digs)
            _bed, ca_ed = self._pack(edges, digs)
            _bgl, ca_glu = self._pack(gl_u, digs)
            _bba, ca_ba = self._pack(bary, digs)
            phi_b = mp.nstr(phi, digs).encode()

            z = target
            stride = 112
            buf = ctypes.create_string_buffer(2 * stride)
            res = mp.mpf(1)
            for _ in range(8):
                zb = mp.nstr(z, digs).encode()
                self.lib.gamma_resolvent_R_and_Rp(
                    len(hs), n_pan, p_deg, bits, zb, phi_b, ca_h, ca_w, ca_ed, ca_glu, ca_ba, buf, stride
                )
                Rv = mp.mpf(buf.raw[:stride].split(b"\0")[0].decode())
                Rp = mp.mpf(buf.raw[stride:2 * stride].split(b"\0")[0].decode())
                res = (z + Rv) - target
                z -= res / (1 + Rp)
                if abs(res) < mp.power(10, -(dps_work - 12)):
                    break
            zb = mp.nstr(z, digs).encode()
            self.lib.gamma_resolvent_R_and_Rp(
                len(hs), n_pan, p_deg, bits, zb, phi_b, ca_h, ca_w, ca_ed, ca_glu, ca_ba, buf, stride
            )
            Rv = mp.mpf(buf.raw[:stride].split(b"\0")[0].decode())
            Rp = mp.mpf(buf.raw[stride:2 * stride].split(b"\0")[0].decode())
            res = abs((z + Rv) - target)
            s = 1 - eps
            Fp = 1 + Rp
            ep = s * Fp
            zm = (target - eps) - Rv
            D = mp.log1p(zm / eps) - 2 * target
            em = -eps * (-eps + Rp - eps * Rp) - zm * ep
            Ds = -(em / (z * eps) - 2 * Fp) / ep
        return +D, +Ds, +res

    def lognormal_D_and_Ds(
        self, w_str: str, phi_obj: str | mp.mpf, eps: mp.mpf,
        M: int = 15, n_gh: int = 60, n_pan: int = 2, p_deg: int = 11, bits: int = 224,
    ) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
        dps_work = max(65, int(bits * 0.30103) + 8)
        with mp.workdps(dps_work):
            w_var = mp.mpf(w_str)
            phi = mp.mpf(phi_obj)
            eps = mp.mpf(eps)
            sigma = mp.sqrt(w_var)
            gh_m, wt_m = mp.gauss_quadrature(n_gh, "hermite")
            xi_q = [mp.sqrt(2) * gh_m[r] for r in range(n_gh)]
            wt_q = [wt_m[r] / mp.sqrt(mp.pi) for r in range(n_gh)]
            sw = mp.fsum(wt_q)
            wt_q = [w / sw for w in wt_q]
            h_q = [mp.exp(sigma * x - w_var / 2) for x in xi_q]
            M_eff = 1 if phi == 0 else M
            H = [[mp.mpf(1) for _ in range(n_gh)]]
            if M_eff > 1:
                H.append([x for x in xi_q])
            for k in range(1, M_eff - 1):
                H.append([(xi_q[r] * H[k][r] - mp.sqrt(k) * H[k - 1][r]) / mp.sqrt(k + 1) for r in range(n_gh)])
            Hwt_flat = [wt_q[r] * H[k][r] for k in range(M_eff) for r in range(n_gh)]
            C_flat = [
                wt_q[r] * H[k][r] * (phi ** m) * H[m][r]
                for k in range(M_eff) for m in range(M_eff) for r in range(n_gh)
            ]
            target = -mp.log1p(-eps)
            u_min = 1 / (1 + 3 * target * max(h_q))
            edges = [u_min + (1 - u_min) * (1 - (mp.mpf(n_pan - j) / n_pan) ** 2) for j in range(n_pan + 1)]
            gl_u, bary = self._gl(p_deg)

            digs = dps_work - 2
            _bh, ca_h = self._pack(h_q, digs)
            _bhwt, ca_Hwt = self._pack(Hwt_flat, digs)
            _bc, ca_C = self._pack(C_flat, digs)
            _bed, ca_ed = self._pack(edges, digs)
            _bgl, ca_glu = self._pack(gl_u, digs)
            _bba, ca_ba = self._pack(bary, digs)

            z = target
            stride = 112
            buf = ctypes.create_string_buffer(2 * stride)
            for _ in range(8):
                zb = mp.nstr(z, digs).encode()
                self.lib.mehler_R_and_Rp(
                    M_eff, n_gh, n_pan, p_deg, bits, zb, ca_h, ca_Hwt, ca_C, ca_ed, ca_glu, ca_ba, buf, stride
                )
                Rv = mp.mpf(buf.raw[:stride].split(b"\0")[0].decode())
                Rp = mp.mpf(buf.raw[stride:2 * stride].split(b"\0")[0].decode())
                res = (z + Rv) - target
                z -= res / (1 + Rp)
                if abs(res) < mp.power(10, -(dps_work - 12)):
                    break
            zb = mp.nstr(z, digs).encode()
            self.lib.mehler_R_and_Rp(
                M_eff, n_gh, n_pan, p_deg, bits, zb, ca_h, ca_Hwt, ca_C, ca_ed, ca_glu, ca_ba, buf, stride
            )
            Rv = mp.mpf(buf.raw[:stride].split(b"\0")[0].decode())
            Rp = mp.mpf(buf.raw[stride:2 * stride].split(b"\0")[0].decode())
            res = abs((z + Rv) - target)
            s = 1 - eps
            Fp = 1 + Rp
            ep = s * Fp
            zm = (target - eps) - Rv
            D = mp.log1p(zm / eps) - 2 * target
            em = -eps * (-eps + Rp - eps * Rp) - zm * ep
            Ds = -(em / (z * eps) - 2 * Fp) / ep
        return +D, +Ds, +res


# ---------------------------------------------------------------------------
# Section C: Deterministic small-staying Koopman Nystrom solvers (Gamma & Lognormal)
# ---------------------------------------------------------------------------

def _build_smallstay_x_panels(L: float, eta_max: float, p_deg: int = 14, base_h: float = 1.8, ratio: float = 1.22):
    """Geometrically graded discontinuous Gauss-Legendre panels on [-L, L + eta_max^+ + log 2]."""
    right_top = L + max(eta_max, 0.0) + math.log(2.0) + 0.05
    if L <= 24.0:
        n_p = max(6, int(math.ceil((right_top + L) / base_h)))
        edges = np.linspace(-L, right_top, n_p + 1)
    else:
        l_pts = [-L]
        cur = -L
        for _ in range(8):
            cur += base_h
            if cur >= 0.0:
                break
            l_pts.append(cur)
        hs = base_h * ratio
        while cur + hs < 0.0:
            cur += hs
            l_pts.append(cur)
            hs *= ratio
        l_pts.append(0.0)

        r_lo = max(0.0, L - 18.0)
        n_rt = max(2, int(math.ceil((right_top - r_lo) / base_h)))
        r_top = np.linspace(r_lo, right_top, n_rt + 1)
        r_mid = [r_lo]
        cur = r_lo
        hs = base_h * ratio
        while cur - hs > 0.0:
            cur -= hs
            r_mid.append(cur)
            hs *= ratio
        edges = np.unique(np.concatenate([l_pts, r_mid[::-1], r_top]))

    gl_u, _ = np.polynomial.legendre.leggauss(p_deg)
    bary = np.array([1.0 / np.prod(gl_u[j] - np.delete(gl_u, j)) for j in range(p_deg)])
    nodes = np.concatenate([0.5 * (a + b) + 0.5 * (b - a) * gl_u for a, b in zip(edges[:-1], edges[1:])])
    return nodes, edges, gl_u, bary


def _eval_panel_bary_matrix(x_targets: np.ndarray, edges: np.ndarray, gl_u: np.ndarray, bary: np.ndarray) -> np.ndarray:
    p_deg = len(gl_u)
    Nx = (len(edges) - 1) * p_deg
    idx = np.clip(np.searchsorted(edges, x_targets, side="right") - 1, 0, len(edges) - 2)
    a = edges[idx]
    b = edges[idx + 1]
    u = (2.0 * x_targets - (a + b)) / (b - a)
    diff = u[:, None] - gl_u[None, :]
    ex = np.abs(diff) < 1e-14
    ds = np.where(ex, 1.0, diff)
    w = bary[None, :] / ds
    w /= np.sum(w, axis=1, keepdims=True)
    if np.any(ex):
        row_ex = np.any(ex, axis=1)
        w[row_ex] = ex[row_ex].astype(float)
    P = np.zeros((len(x_targets), Nx))
    P[np.arange(len(x_targets))[:, None], idx[:, None] * p_deg + np.arange(p_deg)[None, :]] = w
    return P


def solve_gamma_smallstay_q(
    L: float, p: float, phi: float, p_deg: int = 14, base_h: float = 1.8, ratio: float = 1.22, n_de: int = 130,
) -> tuple[float, float, float]:
    """Deterministic 1D exact constant-run resolvent Koopman solver returning (q_primal, q_adjoint, Vpi_identity_est)."""
    psi = float(digamma(p))
    lg = float(gammaln(p))
    u = np.linspace(-6.5, 5.0, n_de)
    hu = u[1] - u[0]
    eu = np.exp(-u)
    t = u - eu
    dt = 1.0 + eu
    lw = p * t - np.exp(t) - lg + np.log(dt * hu)
    mask = lw > -92.0
    etas = t[mask] - psi
    ws = np.exp(lw[mask])
    ws /= np.sum(ws)

    nodes, edges, gl_u, bary = _build_smallstay_x_panels(L, float(np.max(etas)), p_deg=p_deg, base_h=base_h, ratio=ratio)
    Nx = len(nodes)
    coef_phi = 2.0 / (1.0 + phi)

    zz = nodes[None, :] - etas[:, None] - L
    Q_mat = np.where(zz > 0.0, zz + np.log1p(np.exp(-np.abs(zz))), np.log1p(np.exp(zz)))
    Psi_mat = (2.0 * nodes[None, :] - coef_phi * etas[:, None] - Q_mat) * Q_mat
    X_next = -nodes[None, :] + etas[:, None] + Q_mat
    P_all = _eval_panel_bary_matrix(X_next.ravel(), edges, gl_u, bary).reshape(len(etas), Nx, Nx)

    if phi == 0.0:
        K_eff = np.einsum("r,rij->ij", ws, P_all)
        r_q = np.einsum("r,ri->i", ws, Q_mat)
        r_vpi = np.einsum("r,ri->i", ws, Psi_mat)
    else:
        M_stack = np.eye(Nx)[None, :, :] - phi * P_all
        B_stack = np.concatenate([P_all, Q_mat[:, :, None], Psi_mat[:, :, None]], axis=2)
        X_stack = np.linalg.solve(M_stack, B_stack)
        wt_om = ws * (1.0 - phi)
        K_eff = np.einsum("r,rij->ij", wt_om, X_stack[:, :, :Nx])
        r_q = np.einsum("r,ri->i", wt_om, X_stack[:, :, Nx])
        r_vpi = np.einsum("r,ri->i", wt_om, X_stack[:, :, Nx + 1])

    A = np.zeros((Nx + 1, Nx + 1))
    A[:Nx, :Nx] = np.eye(Nx) - K_eff
    A[:Nx, Nx] = 1.0
    A[Nx, 0] = 1.0
    rhs = np.zeros(Nx + 1)
    rhs[:Nx] = r_q
    sol = np.linalg.solve(A, rhs)
    q_primal = float(sol[Nx])

    e_last = np.zeros(Nx + 1)
    e_last[Nx] = 1.0
    mu = np.linalg.solve(A.T, e_last)[:Nx]
    q_adjoint = float(np.dot(mu, r_q))
    vpi_est = float(np.dot(mu, r_vpi))
    return q_primal, q_adjoint, vpi_est


def solve_lognormal_smallstay_q(
    L: float, w_var: float, phi: float, M: int = 16, n_gh: int = 52,
    p_deg: int = 13, base_h: float = 2.0, ratio: float = 1.25,
) -> tuple[float, float, float]:
    """Deterministic Mehler-Hermite Koopman solver returning (q_primal, q_adjoint, Vpi_identity_est)."""
    sigma = math.sqrt(w_var)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_gh)
    xi_q = np.sqrt(2.0) * gh_x
    wt_q = gh_w / math.sqrt(math.pi)
    wt_q /= np.sum(wt_q)
    eta_q = sigma * xi_q

    M_eff = 1 if phi == 0.0 else M
    H = np.zeros((M_eff, n_gh))
    H[0] = 1.0
    if M_eff > 1:
        H[1] = xi_q
    for k in range(1, M_eff - 1):
        H[k + 1] = (xi_q * H[k] - math.sqrt(k) * H[k - 1]) / math.sqrt(k + 1)

    nodes, edges, gl_u, bary = _build_smallstay_x_panels(
        L, float(np.max(np.abs(eta_q))), p_deg=p_deg, base_h=base_h, ratio=ratio
    )
    Nx = len(nodes)
    Ntot = M_eff * Nx

    zz = nodes[None, :] - eta_q[:, None] - L
    Q_all = np.where(zz > 0.0, zz + np.log1p(np.exp(-np.abs(zz))), np.log1p(np.exp(zz)))
    Psi_all = (2.0 * nodes[None, :] - (2.0 / (1.0 + phi)) * eta_q[:, None] - Q_all) * Q_all
    X_next = -nodes[None, :] + eta_q[:, None] + Q_all
    P_all = _eval_panel_bary_matrix(X_next.ravel(), edges, gl_u, bary).reshape(n_gh, Nx, Nx)

    phi_pow = np.array([phi ** m for m in range(M_eff)])
    C = (wt_q[None, None, :] * H[:, None, :] * H[None, :, :]) * phi_pow[None, :, None]
    K_blocks = np.einsum("kmr,rij->kimj", C, P_all, optimize=True).reshape(Ntot, Ntot)
    rhs_q = np.einsum("r,kr,ri->ki", wt_q, H, Q_all).ravel()
    rhs_vpi = np.einsum("r,kr,ri->ki", wt_q, H, Psi_all).ravel()

    A = np.zeros((Ntot + 1, Ntot + 1))
    A[:Ntot, :Ntot] = np.eye(Ntot) - K_blocks
    A[:Nx, Ntot] = 1.0
    A[Ntot, 0] = 1.0
    b_vec = np.zeros(Ntot + 1)
    b_vec[:Ntot] = rhs_q
    sol = np.linalg.solve(A, b_vec)
    q_primal = float(sol[Ntot])

    e_last = np.zeros(Ntot + 1)
    e_last[Ntot] = 1.0
    mu = np.linalg.solve(A.T, e_last)[:Ntot]
    q_adjoint = float(np.dot(mu, rhs_q))
    vpi_est = float(np.dot(mu, rhs_vpi))
    return q_primal, q_adjoint, vpi_est


def moving_threshold_from_solver(L0: float, eval_q_fn) -> tuple[float, float, float]:
    """Solve L_* + q(L_*)/2 = L0 = k log 10 by fixed-point iteration; return (q_{L_*}, Vpi_est, ell_minus_D)."""
    Ls = L0
    qv, vpi_est = 0.0, 0.0
    for _ in range(5):
        qv, q_adj, vpi_est = eval_q_fn(Ls)
        if abs(qv - q_adj) > 1e-11:
            raise ArithmeticError(f"Primal/adjoint mismatch {abs(qv - q_adj)} at L={Ls}")
        Ls_next = L0 - 0.5 * qv
        if abs(Ls_next - Ls) < 1e-14:
            Ls = Ls_next
            break
        Ls = Ls_next
    qv, _, vpi_est = eval_q_fn(Ls)
    s_term = float(mp.log1p(-mp.exp(-mp.mpf(L0))))
    ell_minus_D = qv + s_term
    return qv, vpi_est, ell_minus_D


# ---------------------------------------------------------------------------
# Section D: Verification campaigns (G1–G4, Algebra/Pointwise, Sub-item i, Sub-item ii)
# ---------------------------------------------------------------------------

def run_known_result_gates(koop: MPFRKoopmanLib) -> None:
    """study rules Rule 7: every new solver reproduces known exact benchmarks BEFORE new checks."""
    print("\n=== 1. Pre-registered Known-Result Gates (study rules Rule 7) ===", flush=True)

    # Gate G1: Small-noise Lognormal AR(1) vs exact memA P1 formula D = w * F_phi(s) + O(w^2)
    # Using M=12, n_pan=3, p_deg=12 so u-panel discretization error (~1e-44) is 20+ orders below w * eps = 1e-24!
    w_tiny = "1e-22"
    for phi_s in ("-0.5", "0.5"):
        eps_g = mp.mpf("0.01")
        D_m, Ds_m, _ = koop.lognormal_D_and_Ds(w_tiny, phi_s, eps_g, M=12, n_gh=44, n_pan=3, p_deg=12, bits=256)
        ph = mp.mpf(phi_s)
        w_mp = mp.mpf(w_tiny)
        f_phi = lambda a: a * (1 + (a + 2) * ph) / (2 * (1 + a) * (1 + a * ph))
        D_p1 = w_mp * f_phi(eps_g)
        Ds_p1 = -w_mp * mp.diff(f_phi, eps_g)
        rel_err = max(abs(D_m - D_p1) / abs(D_p1), abs(Ds_m - Ds_p1) / abs(Ds_p1))
        record_check(
            "gates", f"G1_lognormal_memA_P1_smallnoise_phi={phi_s}",
            rel_err < mp.mpf("1e-18"),
            f"rel_err={fmt(rel_err, 6)} (tol 1e-18)",
        )

    # Gate G2: Exact i.i.d. Gamma Tricomi D(s) & D'(s) vs MPFR Gamma resolvent solver at p in {0.25, 0.5, 2.0} (tol 1e-30)
    for p_s in ("0.25", "0.5", "2.0"):
        eps_g = mp.mpf("1e-4")
        D_tri, Ds_tri = tricomi_D_and_Ds(eps_g, mp.mpf(p_s), dps=65)
        D_res, Ds_res, r_res = koop.gamma_D_and_Ds(p_s, "0", eps_g, n_quad=140, n_pan=3, p_deg=12, bits=224)
        err = max(abs(D_res - D_tri), abs(Ds_res - Ds_tri))
        record_check(
            "gates", f"G2_gamma_resolvent_vs_exact_Tricomi_p={p_s}",
            err < mp.mpf("1e-30"),
            f"|err|={fmt(err, 6)}, root_res={fmt(r_res, 5)} (tol 1e-30)",
        )

    # Gate G3: Small-staying 1D Gamma Koopman solver vs exact Tricomi q_L at L = 80, p in {0.5, 2.0} (tol 1e-11)
    for p_f in (0.5, 2.0):
        q_ex = float(tricomi_q_at_L(mp.mpf(80), mp.mpf(p_f), dps=55))
        q_pr, q_ad, vpi_e = solve_gamma_smallstay_q(80.0, p_f, 0.0, p_deg=14, base_h=1.7, ratio=1.20)
        vpi_true = float(mp.polygamma(1, mp.mpf(p_f)))
        err_q = max(abs(q_pr - q_ex), abs(q_ad - q_ex))
        err_id = abs(vpi_e - vpi_true)
        record_check(
            "gates", f"G3_smallstay_gamma_vs_Tricomi_p={p_f}_L=80",
            err_q < 1e-11 and err_id < 1e-11,
            f"|q_err|={err_q:.3e}, |Vpi_err|={err_id:.3e} (tol 1e-11)",
        )

    # Gate G4: Small-staying Mehler-Hermite Lognormal solver at phi=0 vs direct 1D Gauss-Hermite Nystrom at L=80 (tol 1e-12)
    q_mh, q_mh_ad, vpi_mh = solve_lognormal_smallstay_q(80.0, 0.5, 0.0, M=1, n_gh=64, p_deg=14, base_h=1.7, ratio=1.20)
    q_mh2, _, _ = solve_lognormal_smallstay_q(80.0, 0.5, 0.0, M=1, n_gh=44, p_deg=11, base_h=2.2, ratio=1.28)
    err_mh = max(abs(q_mh - q_mh_ad), abs(q_mh - q_mh2), abs(vpi_mh - 0.5))
    record_check(
        "gates", "G4_smallstay_lognormal_Mehler_duality_and_identity_L=80",
        err_mh < 1e-11,
        f"max_err={err_mh:.3e} (tol 1e-11)",
    )

    if FAILURES:
        raise SystemExit(f"Known-result gate failed: {FAILURES}")


def run_algebra_and_pointwise_audit() -> None:
    """Check every algebraic formula, chain-rule bound, period-2 counterexample, A=0 B<0 certificate, and tail bound."""
    print("\n=== 2. Algebraic, Chain-Rule, Counterexample, and Pointwise Tail Audit ===", flush=True)

    # 2a. Exact rational verification of the A=0 lognormal curve B < 0 certificate:
    # P(u) = u*(20 u^3 + 37 u^2 - 24 u + 12) > 0 for all 0 < u < 1 (where x = 1 + u = e^w in (1, 2)).
    max_B_on_A0 = mp.mpf("-inf")
    max_A_resid = mp.mpf(0)
    for w_s in ("1e-6", "0.001", "0.01", "0.1", "0.25", "0.5", "0.68", "0.6931"):
        w = mp.mpf(w_s)
        phi0 = mp.log((3 - mp.exp(w)) / 2) / w
        cc = host_constants("lognormal", w_s, fmt(phi0, 62))
        max_A_resid = max(max_A_resid, abs(cc["A"]))
        max_B_on_A0 = max(max_B_on_A0, cc["B"])
    record_check(
        "algebra_and_pointwise", "A_zero_locus_strict_B_negative_across_0_lt_w_lt_log2",
        max_A_resid < mp.mpf("1e-55") and max_B_on_A0 < 0,
        f"max|A|={fmt(max_A_resid, 5)}, max(B)={fmt(max_B_on_A0, 8)} < 0",
    )

    # 2b. Independent 3D Gauss-Hermite quadrature check of host moments (a, b, E h^4)
    # Using 64 Gauss-Hermite nodes so even exp(4*sqrt(1.1)*xi) (E h^4 = e^6.6) is exact to > 45 digits!
    gh_x, gh_w = mp.gauss_quadrature(64, "hermite")
    norm_mgf = lambda t: mp.fsum(gh_w[i] * mp.exp(mp.sqrt(2) * t * gh_x[i]) for i in range(64)) / mp.sqrt(mp.pi)
    max_mom_err = mp.mpf(0)
    for w_s, ph_s in [("0.5", "-0.5"), ("0.5", "0.5"), ("0.1", "-0.75"), ("1.1", "-0.85")]:
        w = mp.mpf(w_s)
        ph = mp.mpf(ph_s)
        cc = host_constants("lognormal", w_s, ph_s)
        def jmom(pows):
            n = len(pows)
            ts = [mp.sqrt(w) * sum(pows[j] * (ph ** j) for j in range(n))]
            ts += [mp.sqrt(w * (1 - ph * ph)) * sum(pows[j] * (ph ** (j - k)) for j in range(k, n)) for k in range(1, n)]
            return mp.exp(-w * sum(pows) / 2) * mp.fprod(norm_mgf(t) for t in ts)
        a_q = jmom([2]) + 2 * jmom([1, 1])
        b_q = jmom([3]) + 3 * (jmom([2, 1]) + jmom([1, 2]) + jmom([1, 1, 1]))
        h4_q = jmom([4])
        max_mom_err = max(max_mom_err, abs(a_q - cc["a"]), abs(b_q - cc["b"]), abs(h4_q - cc["eh4"]) / cc["eh4"])
    record_check(
        "algebra_and_pointwise", "lognormal_joint_moments_quadrature",
        max_mom_err < mp.mpf("1e-35"),
        f"max_err={fmt(max_mom_err, 6)}",
    )

    # 2c. Period-two random-phase alternating host counterexample (§3 lines 172-180)
    max_alt_err = mp.mpf(0)
    for b_s in ("0.2", "0.7", "1.5"):
        b = mp.mpf(b_s)
        ell = mp.log(mp.cosh(b))
        hp = mp.exp(b) / mp.cosh(b)
        hm = mp.exp(-b) / mp.cosh(b)
        s = mp.mpf("1e-30")
        disc = mp.sqrt((hp + hm) ** 2 - 4 * hp * hm * (1 - s * s))
        c_star = 2 * (1 - s * s) / (hp + hm + disc)
        D_s = mp.log(c_star / (1 - s))
        max_alt_err = max(max_alt_err, abs(D_s - (ell - b)))
    record_check(
        "algebra_and_pointwise", "period_two_endpoint_counterexample",
        max_alt_err < mp.mpf("1e-29"),
        f"max|D(10^-30) - (ell - b)|={fmt(max_alt_err, 6)}",
    )

    # 2d. Exhaustive pointwise verification of §4 inequalities (4.3), (4.5), and three-region bounds (13,500 cases)
    max_violation = mp.mpf("-inf")
    n_pts = 0
    phis = [mp.mpf(x) for x in ("-0.95", "-0.5", "0", "0.5", "0.95")]
    etas = [mp.mpf(x) for x in ("-80", "-12", "-1.5", "0", "2.5", "50")]
    Ls = [mp.mpf(x) for x in ("1", "5", "40", "500")]
    Ks = [mp.mpf(x) for x in ("1", "3", "15", "80", "200")]
    for ph in phis:
        Cg = 6 + 2 / (1 + ph)
        Cb = 5 + 2 / (1 + ph)
        Ct = 8 + 4 / (1 + ph)
        for e_prev in etas:
            for e_cur in etas:
                R = 1 + abs(e_prev) + abs(e_cur)
                for L in Ls:
                    y_grid = [
                        -2 * L, -L - mp.mpf("0.5"), -2 * R, -abs(e_cur) - mp.mpf("0.1"),
                        mp.mpf("-1"), mp.mpf("0"), max(e_prev, mp.mpf(0)) + mp.log(2),
                    ]
                    # Restrict to admissible stationary support -2L <= Y <= eta_{t-1}^+ + log 2
                    for Y in y_grid:
                        if Y < -2 * L or Y > max(e_prev, mp.mpf(0)) + mp.log(2):
                            continue
                        Q = log1pexp_mp(Y - e_cur)
                        integ = abs((2 * Y - 2 * e_cur / (1 + ph) - Q) * Q)
                        max_violation = max(
                            max_violation,
                            Y - R,
                            Q - 2 * R,
                            abs(Y) * Q - 2 * R * R,
                            integ - Ct * R * R,
                        )
                        for K_cut in Ks:
                            if R <= K_cut:
                                bnd = Cg * K_cut * Q if Y >= -2 * K_cut else Cb * K_cut * mp.exp(-K_cut)
                                max_violation = max(max_violation, integ - bnd)
                            n_pts += 1
    record_check(
        "algebra_and_pointwise", f"pointwise_tail_bounds_({n_pts}_cases)",
        max_violation <= mp.mpf("0"),
        f"max_violation={fmt(max_violation, 6)} (strictly <= 0)",
    )


def run_derivative_interval_campaign(koop: MPFRKoopmanLib) -> None:
    """Task 2(i): High-precision deterministic check of D(s) and D'(s) intervals across interior & edge regimes."""
    print("\n=== 3. High-Precision Verification of Long-Stay D(s) & D'(s) Intervals (Sub-item i) ===", flush=True)

    w_a0 = mp.mpf("0.1")
    phi_a0 = fmt(mp.log((3 - mp.exp(w_a0)) / 2) / w_a0, 62)

    # 12 parameter configurations (covering all primary cells + singular p=0.25, strong memory phi=0.85,
    # negative-A lognormal cells, and TWO points on the degenerate A=0 curve w=0.1 and w=0.4)
    w_a0_2 = mp.mpf("0.4")
    phi_a0_2 = fmt(mp.log((3 - mp.exp(w_a0_2)) / 2) / w_a0_2, 62)

    configs = [
        ("gamma", "2.0", "0.0", "iid_moderate"),
        ("gamma", "2.0", "0.5", "pos_mem_moderate"),
        ("gamma", "2.0", "0.85", "strong_pos_mem"),
        ("gamma", "0.5", "0.0", "iid_singular_p0.5"),
        ("gamma", "0.5", "0.5", "pos_mem_singular_p0.5"),
        ("gamma", "0.25", "0.5", "pos_mem_singular_p0.25"),
        ("lognormal", "0.5", "-0.5", "neg_mem_pos_A"),
        ("lognormal", "0.5", "0.0", "iid_lognormal"),
        ("lognormal", "0.5", "0.5", "pos_mem_lognormal"),
        ("lognormal", "0.1", "-0.75", "neg_mem_neg_A_counterexample"),
        ("lognormal", "0.25", "-0.80", "neg_mem_neg_A_stronger"),
        ("lognormal", "0.1", phi_a0, "degenerate_A0_w0.1"),
        ("lognormal", "0.4", phi_a0_2, "degenerate_A0_w0.4"),
    ]

    for model, par_s, phi_s, tag_desc in configs:
        cc = host_constants(model, par_s, phi_s)
        is_zero_A = abs(cc["A"]) < mp.mpf("1e-45")
        eps_sign = cc["eps_B"] / 2 if is_zero_A else cc["eps_A"] / 2
        eps_list = [("wide_eps0_half", cc["eps0"] / 2), ("sign_half", eps_sign)]

        for eps_tag, eps_val in eps_list:
            t0 = time.monotonic()
            if model == "gamma":
                D_c, Ds_c, _ = koop.gamma_D_and_Ds(par_s, phi_s, eps_val, n_quad=115, n_pan=2, p_deg=9, bits=224)
                D_f, Ds_f, r_f = koop.gamma_D_and_Ds(par_s, phi_s, eps_val, n_quad=145, n_pan=3, p_deg=12, bits=224)
            else:
                D_c, Ds_c, _ = koop.lognormal_D_and_Ds(par_s, phi_s, eps_val, M=12, n_gh=48, n_pan=2, p_deg=8, bits=224)
                D_f, Ds_f, r_f = koop.lognormal_D_and_Ds(par_s, phi_s, eps_val, M=15, n_gh=58, n_pan=2, p_deg=11, bits=224)

            mesh_diff_D = abs(D_f - D_c)
            mesh_diff_Ds = abs(Ds_f - Ds_c)

            # Proved Taylor bounds (2.1) and (2.2)
            rem_D = abs(D_f - cc["A"] * eps_val - cc["B"] * (eps_val ** 2))
            bnd_D = cc["K"] * (eps_val ** 3) / 6
            rem_Ds = abs(Ds_f + cc["A"] + 2 * cc["B"] * eps_val)
            bnd_Ds = cc["K"] * (eps_val ** 2) / 2

            ok_interval = (rem_D <= bnd_D) and (rem_Ds <= bnd_Ds)
            ok_mesh = mesh_diff_Ds < mp.mpf("1e-11")

            ok_sign = True
            sign_detail = ""
            if eps_tag == "sign_half":
                if is_zero_A:
                    ok_sign = (abs(Ds_f + 2 * cc["B"] * eps_val) <= abs(cc["B"]) * eps_val) and (Ds_f > 0)
                    sign_detail = f", B-sign ok (Ds={fmt(Ds_f, 14)} > 0)"
                else:
                    ok_sign = (abs(Ds_f + cc["A"]) <= abs(cc["A"]) / 2) and (mp.sign(Ds_f) == -mp.sign(cc["A"]))
                    sign_detail = f", A-sign ok (A={fmt(cc['A'], 8)}, Ds={fmt(Ds_f, 12)})"

            # Repeat 288-bit check on representative correlated cells
            bit_diff = mp.mpf(0)
            if eps_tag == "sign_half" and tag_desc in ("pos_mem_singular_p0.5", "degenerate_A0_w0.1"):
                if model == "gamma":
                    _, Ds_288, _ = koop.gamma_D_and_Ds(par_s, phi_s, eps_val, n_quad=145, n_pan=3, p_deg=12, bits=288)
                else:
                    _, Ds_288, _ = koop.lognormal_D_and_Ds(par_s, phi_s, eps_val, M=15, n_gh=58, n_pan=2, p_deg=11, bits=288)
                bit_diff = abs(Ds_288 - Ds_f)

            elapsed = time.monotonic() - t0
            record_check(
                "derivative_intervals",
                f"deriv_{model}_par={par_s}_phi={mp.nstr(mp.mpf(phi_s),6)}_{eps_tag}",
                ok_interval and ok_mesh and ok_sign,
                f"Ds={fmt(Ds_f, 16)}, rem_Ds/bnd={fmt(rem_Ds / bnd_Ds, 5)}, mesh_diff={fmt(mesh_diff_Ds, 4)}{sign_detail} ({elapsed:.2f}s)",
                payload={
                    "model": model, "par": par_s, "phi": phi_s, "tag": tag_desc, "eps_tag": eps_tag,
                    "epsilon": fmt(eps_val, 30), "A": fmt(cc["A"], 24), "B": fmt(cc["B"], 24), "K": fmt(cc["K"], 18),
                    "D": fmt(D_f, 32), "Ds": fmt(Ds_f, 32),
                    "rem_D": fmt(rem_D, 16), "bnd_D": fmt(bnd_D, 16),
                    "rem_Ds": fmt(rem_Ds, 16), "bnd_Ds": fmt(bnd_Ds, 16),
                    "mesh_diff_D": fmt(mesh_diff_D, 8), "mesh_diff_Ds": fmt(mesh_diff_Ds, 8),
                    "bit224_vs_288_diff": fmt(bit_diff, 8), "root_residual": fmt(r_f, 8),
                },
            )


def run_small_staying_rate_campaign() -> None:
    """Task 2(ii): High-precision s = 10^-k small-staying rate checks with Richardson extrapolation."""
    print("\n=== 4. Small-Staying Rate at s = 10^-k with Richardson Extrapolation (Sub-item ii) ===", flush=True)

    # Part 4A: Exact 80-digit i.i.d. Gamma via Tricomi U at s = 10^-k, k in {10, 30, 60, 120, 240},
    # plus explicit tail-interval check (4.7)-(4.9) at k = 4343 (L0 = 4343 log 10 ~ 10000.12).
    ks_exact = (10, 30, 60, 120, 240)
    for p_s in ("0.25", "0.5", "1.0", "2.0", "5.0"):
        cc = host_constants("gamma", p_s, "0.0")
        A_coef = cc["V_pi"] / 2  # psi_1(p) / 2
        S_vals: dict[int, mp.mpf] = {}
        D_vals: dict[int, mp.mpf] = {}
        for k in ks_exact:
            L0 = k * mp.log(10)
            qv, ell_m_D = tricomi_threshold_at_L0(L0, mp.mpf(p_s), dps=80)
            S_vals[k] = L0 * ell_m_D / A_coef
            D_vals[k] = cc["ell"] - ell_m_D

        # 2-point linear Richardson on (120, 240) and 3-point quadratic Richardson on (60, 120, 240)
        rich2_a = 2 * S_vals[120] - S_vals[60]
        rich2_b = 2 * S_vals[240] - S_vals[120]
        rich3 = (4 * rich2_b - rich2_a) / 3

        # Explicit nonasymptotic interval (4.7)-(4.9) at k = 4343 (L0 ~ 10000.12) and exact L = 10000
        q_10000 = tricomi_q_at_L(mp.mpf(10000), mp.mpf(p_s), dps=75)
        iv = explicit_q_interval(mp.mpf(10000), cc, mp.mpf(0))
        ok_iv = True
        iv_str = "denom <= 0 at L=10000 (needs larger L)"
        if iv is not None:
            q_lo, q_hi, rad_49 = iv
            ok_iv = (q_lo <= q_10000 <= q_hi) and (abs(q_10000 - A_coef / 10000) <= rad_49)
            iv_str = f"[{fmt(q_lo, 8)}, {fmt(q_hi, 8)}] covers {fmt(q_10000, 8)}"
        else:
            # For p = 0.25 (a = 1/16, b0 = 64), 2L > Cg*K holds at L = 100000
            q_1e5 = tricomi_q_at_L(mp.mpf(100000), mp.mpf(p_s), dps=75)
            iv2 = explicit_q_interval(mp.mpf(100000), cc, mp.mpf(0))
            assert iv2 is not None
            q_lo, q_hi, rad_49 = iv2
            ok_iv = (q_lo <= q_1e5 <= q_hi) and (abs(q_1e5 - A_coef / 100000) <= rad_49)
            iv_str = f"L=1e5 [{fmt(q_lo, 8)}, {fmt(q_hi, 8)}] covers {fmt(q_1e5, 8)}"

        record_check(
            "exact_gamma_rates",
            f"exact_iid_gamma_rate_Richardson_p={p_s}",
            abs(rich2_b - 1) < mp.mpf("2e-4") and abs(rich3 - 1) < mp.mpf("2e-5") and ok_iv,
            f"S(10^-240)={fmt(S_vals[240], 9)}, Rich2={fmt(rich2_b, 10)} (err {fmt(abs(rich2_b-1), 4)}), Rich3={fmt(rich3, 11)} (err {fmt(abs(rich3-1), 4)}), {iv_str}",
            payload={
                "p": p_s, "ell": fmt(cc["ell"], 24), "A_rate": fmt(A_coef, 24),
                "D_at_k240": fmt(D_vals[240], 24),
                "S_by_k": {str(k): fmt(S_vals[k], 18) for k in ks_exact},
                "rich2_120_240": fmt(rich2_b, 18),
                "rich3_60_120_240": fmt(rich3, 18),
                "explicit_interval_detail": iv_str,
            },
        )

    # Part 4B: Deterministic Koopman Nystrom verification of CORRELATED Gamma refresh & Lognormal AR(1)
    # at s = 10^-k for k in {10, 20, 40, 80, 160} (L0 = k log 10 up to 368.41), with 2-point & 3-point Richardson!
    ks_corr = (10, 20, 40, 80, 160)
    corr_families = [
        ("gamma", "2.0", "0.0", "iid_control_p2"),
        ("gamma", "2.0", "0.5", "pos_mem_p2"),
        ("gamma", "2.0", "0.85", "strong_pos_mem_p2"),
        ("gamma", "0.5", "0.5", "singular_p0.5_pos_mem"),
        ("gamma", "0.25", "0.5", "singular_p0.25_pos_mem"),
        ("lognormal", "0.5", "-0.5", "neg_mem_w0.5"),
        ("lognormal", "0.5", "0.0", "iid_control_w0.5"),
        ("lognormal", "0.5", "0.5", "pos_mem_w0.5"),
        ("lognormal", "0.5", "-0.8", "strong_neg_mem_w0.5"),
        ("lognormal", "0.5", "0.8", "strong_pos_mem_w0.5"),
        ("lognormal", "0.1", "-0.75", "neg_D_near_1_recovers_to_ell"),
        ("lognormal", "1.0", "-0.5", "larger_var_w1.0_neg_mem"),
    ]

    ln10 = math.log(10.0)
    for model, par_s, phi_s, desc in corr_families:
        t0 = time.monotonic()
        cc = host_constants(model, par_s, phi_s)
        A_coef = float(cc["V_pi"] / 2)
        vpi_true = float(cc["V_pi"])
        par_f = float(par_s)
        phi_f = float(phi_s)

        if model == "gamma":
            eval_fn = lambda L_val, pf=par_f, phf=phi_f: solve_gamma_smallstay_q(
                L_val, pf, phf, p_deg=14, base_h=1.75, ratio=1.20
            )
            eval_coarse = lambda L_val, pf=par_f, phf=phi_f: solve_gamma_smallstay_q(
                L_val, pf, phf, p_deg=11, base_h=2.3, ratio=1.28
            )
        else:
            M_modes = 16 if abs(phi_f) >= 0.75 else 13
            n_gh_pts = 52 if par_f >= 1.0 else 48
            eval_fn = lambda L_val, wf=par_f, phf=phi_f, mm=M_modes, ng=n_gh_pts: solve_lognormal_smallstay_q(
                L_val, wf, phf, M=mm, n_gh=ng, p_deg=12, base_h=1.9, ratio=1.24
            )
            eval_coarse = lambda L_val, wf=par_f, phf=phi_f, mm=max(1, M_modes - 3), ng=n_gh_pts - 8: solve_lognormal_smallstay_q(
                L_val, wf, phf, M=mm, n_gh=ng, p_deg=10, base_h=2.5, ratio=1.32
            )

        S_map: dict[int, float] = {}
        max_id_err = 0.0
        D_kmax = 0.0
        for k in ks_corr:
            L0 = k * ln10
            qv, vpi_est, ell_m_D = moving_threshold_from_solver(L0, eval_fn)
            S_map[k] = L0 * ell_m_D / A_coef
            max_id_err = max(max_id_err, abs(vpi_est - vpi_true))
            if k == 160:
                D_kmax = float(cc["ell"]) - ell_m_D

        # Coarse-vs-fine mesh convergence check at k = 80
        _, _, ell_m_D_coarse = moving_threshold_from_solver(80 * ln10, eval_coarse)
        S_80_coarse = (80 * ln10) * ell_m_D_coarse / A_coef
        mesh_rel_diff = abs(S_map[80] - S_80_coarse)

        # 2-point linear Richardson on (80, 160) and 3-point quadratic Richardson on (40, 80, 160)
        r2_40_80 = 2.0 * S_map[80] - S_map[40]
        r2_80_160 = 2.0 * S_map[160] - S_map[80]
        r3_40_80_160 = (4.0 * r2_80_160 - r2_40_80) / 3.0

        # Tolerance: Rich2 within 0.3% (0.003) for interior, 1.5% (0.015) for extreme phi >= 0.8 or tiny w=0.1;
        # Rich3 even tighter; stationary identity (4.4) within 1e-9!
        is_extreme = (abs(phi_f) >= 0.8) or (par_f <= 0.15)
        tol_r2 = 0.012 if is_extreme else 0.0025
        tol_r3 = 0.003 if is_extreme else 0.0005
        ok = (abs(r2_80_160 - 1.0) < tol_r2) and (abs(r3_40_80_160 - 1.0) < tol_r3) and (max_id_err < 1e-8) and (mesh_rel_diff < 1e-6)

        elapsed = time.monotonic() - t0
        record_check(
            "correlated_rates",
            f"rate_{model}_par={par_s}_phi={phi_s}",
            ok,
            f"S(10^-160)={S_map[160]:.7f}, Rich2={r2_80_160:.7f} (err {abs(r2_80_160-1):.2e}), Rich3={r3_40_80_160:.7f} (err {abs(r3_40_80_160-1):.2e}), |id4.4|={max_id_err:.2e}, mesh={mesh_rel_diff:.2e} ({elapsed:.2f}s)",
            payload={
                "model": model, "par": par_s, "phi": phi_s, "desc": desc,
                "ell": fmt(cc["ell"], 18), "A_rate": f"{A_coef:.16e}",
                "D_at_k160": f"{D_kmax:.16f}",
                "S_by_k": {str(k): f"{S_map[k]:.12f}" for k in ks_corr},
                "rich2_80_160": f"{r2_80_160:.12f}",
                "rich3_40_80_160": f"{r3_40_80_160:.12f}",
                "max_identity_4_4_error": f"{max_id_err:.4e}",
                "mesh_S80_diff": f"{mesh_rel_diff:.4e}",
            },
        )


def main() -> None:
    t_start = time.monotonic()
    print("Independent Referee Verification for r37-gap3b-check (independent 4 secondary, 2026-11-15)", flush=True)
    print("Zero collaborator modules imported. Deterministic Koopman/Mehler/Tricomi solvers.", flush=True)

    with tempfile.TemporaryDirectory(prefix="secondary-gap3b-") as tmp_dir:
        koop = MPFRKoopmanLib(tmp_dir)
        run_known_result_gates(koop)
        run_algebra_and_pointwise_audit()
        run_derivative_interval_campaign(koop)
        run_small_staying_rate_campaign()

    total_sec = time.monotonic() - t_start
    RESULTS["summary"] = {
        "passed": PASS_COUNT,
        "failed": FAILURES,
        "seconds": round(total_sec, 3),
    }
    out_json = HERE / "verify_gap3b_results.json"
    out_json.write_text(json.dumps(RESULTS, indent=2) + "\n")
    print(f"\nSUMMARY: passed={PASS_COUNT}, failed={len(FAILURES)}, seconds={total_sec:.2f}", flush=True)
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
