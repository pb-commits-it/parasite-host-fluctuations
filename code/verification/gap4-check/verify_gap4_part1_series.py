#!/usr/bin/env python3
"""Independent mpmath.iv verification of GAP 4 (Part 1: constants, series, stationary points).

Zero imports from gap4_certify.py or any collaborator script.
Uses 2-component vector moment recursion (memA eq. 26) in mpmath.iv (iv.dps = 85)
with O(N^3) suffix-sum factorization and explicit complex-b Cauchy derivative tails.
"""

from decimal import Decimal, ROUND_CEILING, localcontext
import json
import math
from pathlib import Path
import re
import time
import mpmath
from mpmath import mp, iv

mp.dps = 85
iv.dps = 85

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
GAP4_JSON = (__import__('pathlib').Path(__file__).resolve().parent / '../../gap4/gap4_results.json').resolve()

ZERO = iv.mpf(0)
ONE = iv.mpf(1)
HALF = iv.mpf("0.5")


def iv_ball(center, rad):
    """Return closed interval enclosing center +/- |rad|_upper."""
    c = iv.mpf(center)
    r = abs(iv.mpf(rad)).b
    return iv.mpf([ (c - r).a, (c + r).b ])


def iv_abs_upper(x):
    """Rigorous upper bound on |t| for t in x (returned as point/thin iv.mpf >= sup |x|)."""
    return abs(iv.mpf(x)).b


def iv_abs_lower(x):
    """Rigorous lower bound on |t| for t in x."""
    return abs(iv.mpf(x)).a


def iv_lo_mpf(x):
    return mp.mpf(iv.mpf(x)._mpi_[0])


def iv_hi_mpf(x):
    return mp.mpf(iv.mpf(x)._mpi_[1])


def format_outward_pm(x, sig_mid=25, sig_rad=4):
    """Format iv.mpf x as '[mid +/- rad]' guaranteed to contain [x.a, x.b]."""
    x = iv.mpf(x)
    lo_m = iv_lo_mpf(x)
    hi_m = iv_hi_mpf(x)
    if lo_m == 0 and hi_m == 0:
        return "0"
    mid_str = mpmath.nstr((lo_m + hi_m) / 2, sig_mid)
    m_iv = iv.mpf(mid_str)
    req_rad_m = iv_hi_mpf(abs(x - m_iv))
    if req_rad_m == 0 and bool(m_iv.a <= x.a) and bool(x.b <= m_iv.b):
        return mid_str
    raw_rad_str = mpmath.nstr(req_rad_m, sig_rad + 14)
    with localcontext() as dctx:
        dctx.prec = 65
        d_val = Decimal(raw_rad_str) * (Decimal(1) + Decimal(10) ** (-(sig_rad + 6)))
        if d_val == 0:
            d_val = Decimal("1e-90")
        exp = d_val.adjusted() - (sig_rad - 1)
        quant = Decimal(10) ** exp
        d_ceil = d_val.quantize(quant, rounding=ROUND_CEILING)
        rad_str = format(d_ceil.normalize(), "g")
    lo_chk = iv.mpf(mid_str) - iv.mpf(rad_str)
    hi_chk = iv.mpf(mid_str) + iv.mpf(rad_str)
    assert bool(lo_chk.a <= x.a) and bool(x.b <= hi_chk.b), f"Outward format failed for {x}"
    return f"[{mid_str} +/- {rad_str}]"


def parse_arb_str_to_iv(s):
    """Parse Arb's '[m +/- r]' or '[+/- r]' or bare decimal string into iv.mpf."""
    s = s.strip()
    m1 = re.match(r"^\[\s*([+-]?[0-9.eE+-]+)\s*\+/-\s*([0-9.eE+-]+)\s*\]$", s)
    if m1:
        m_iv = iv.mpf(m1.group(1))
        r_iv = iv.mpf(m1.group(2))
        return iv.mpf([(m_iv - r_iv).a, (m_iv + r_iv).b])
    m2 = re.match(r"^\[\s*\+/-\s*([0-9.eE+-]+)\s*\]$", s)
    if m2:
        r_iv = iv.mpf(m2.group(1))
        return iv.mpf([(-r_iv).a, r_iv.b])
    return iv.mpf(s)


def intervals_overlap(x, y):
    x = iv.mpf(x)
    y = iv.mpf(y)
    return bool((x.a <= y.b) and (y.a <= x.b))


# ---------------------------------------------------------------------------
# Second-order AR(1) formulas
# ---------------------------------------------------------------------------
def F_ar1(s, phi):
    a = 1 - s
    return a * (1 + (a + 2) * phi) / (2 * (1 + a) * (1 + a * phi))


def Fs_ar1(s, phi):
    a = 1 - s
    return -(1 + 2 * phi + 2 * a * phi - a * a * phi * phi) / (
        2 * (1 + a) ** 2 * (1 + a * phi) ** 2
    )


def s_star_ar1(phi):
    return 1 - (1 - iv.sqrt(2 * (1 + phi))) / phi


# ---------------------------------------------------------------------------
# Degree-d (d in {0, 1, 2}) Taylor jets in b for 2-state vector recursion
# ---------------------------------------------------------------------------
def jet_zero(deg):
    return (ZERO,) * (deg + 1)


def jet_const(val, deg):
    return (iv.mpf(val),) + (ZERO,) * deg


def jet_add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def jet_sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def jet_neg(a):
    return tuple(-x for x in a)


def jet_scale(a, s):
    return tuple(x * s for x in a)


def jet_mul(a, b):
    deg = len(a) - 1
    if deg == 0:
        return (a[0] * b[0],)
    if deg == 1:
        return (a[0] * b[0], a[0] * b[1] + a[1] * b[0])
    return (
        a[0] * b[0],
        a[0] * b[1] + a[1] * b[0],
        a[0] * b[2] + a[1] * b[1] + a[2] * b[0],
    )


def jet_div(u, d):
    deg = len(u) - 1
    q0 = u[0] / d[0]
    if deg == 0:
        return (q0,)
    q1 = (u[1] - d[1] * q0) / d[0]
    if deg == 1:
        return (q0, q1)
    q2 = (u[2] - d[1] * q1 - d[2] * q0) / d[0]
    return (q0, q1, q2)


def two_state_vector_series(b, delta, phi, order, degree=1):
    """Evaluate G(b, delta) and its b-derivatives up to `degree` (0, 1, or 2)
    by solving the full 2-component vector moment system (memA eq. 26) in R^2,
    using exact Pascal suffix sums over m so runtime is O(order^3).

    Returns (derivs, c2_val) where derivs[j] = d^j/db^j G_N(b, delta) (without
    truncation tail) and c2_val = [delta^2] G(b, delta).
    """
    assert degree in (0, 1, 2)
    b = iv.mpf(b)
    delta = iv.mpf(delta)
    phi = iv.mpf(phi)
    zjet = jet_zero(degree)
    bjet = (b,) if degree == 0 else ((b, ONE) if degree == 1 else (b, ONE, ZERO))
    bp = [jet_const(ONE, degree)]
    for _ in range(1, order + 1):
        bp.append(jet_mul(bp[-1], bjet))

    p_same = (ONE + phi) * HALF
    p_diff = (ONE - phi) * HALF
    comb = math.comb

    # PM[ell][nr] stores P^T M_{ell, nr} = (pm_minus_jet, pm_plus_jet) in R^2
    PM = [[None] * (order + 1) for _ in range(order + 1)]
    PM[0][0] = (jet_const(HALF, degree), jet_const(HALF, degree))
    # Msum[k][n] stores M_{k,n,-} + M_{k,n,+}
    Msum = [[None] * (order + 1) for _ in range(order + 1)]

    # S_tab[nr][k][q] = sum_{m=0}^{nr-q} (-1)^{q+m} comb(k+m-1, m) PM[q+m][nr]
    # Satisfies Pascal identity S(k, q) = S(k-1, q) + S(k, q+1) for k >= 1!
    S_tab = [None] * (order + 1)

    def build_suffix_table(nr):
        prev = [None] * (nr + 1)
        for q in range(nr + 1):
            um, up = PM[q][nr]
            if q % 2 == 1:
                prev[q] = (jet_neg(um), jet_neg(up))
            else:
                prev[q] = (um, up)
        rows = [prev]
        for _k in range(1, order + 1):
            cur = [None] * (nr + 1)
            accm, accp = zjet, zjet
            for q in range(nr, -1, -1):
                pm_q, pp_q = prev[q]
                accm = jet_add(pm_q, accm)
                accp = jet_add(pp_q, accp)
                cur[q] = (accm, accp)
            rows.append(cur)
            prev = cur
        S_tab[nr] = rows

    build_suffix_table(0)
    total_jet = zjet
    c2_val = ZERO
    one_jet = jet_const(ONE, degree)

    for n in range(1, order + 1):
        PM[0][n] = (zjet, zjet)
        for k in range(n, 0, -1):
            rm, rp = zjet, zjet
            # r = 0, m = 1 .. n - k
            for m in range(1, n - k + 1):
                ell = k + m
                w = iv.mpf((-1) ** ell * comb(k + m - 1, m))
                um, up = PM[ell][n]
                rm = jet_add(rm, jet_scale(um, w))
                rp = jet_add(rp, jet_scale(up, w))
            # r = 1 .. k: use precomputed Pascal suffix sums
            for r in range(1, k + 1):
                nr = n - r
                q = k - r
                if q > nr:
                    continue
                c_kr = iv.mpf(comb(k, r))
                sm, sp = S_tab[nr][k][q]
                if r % 2 == 1:
                    rm = jet_sub(rm, jet_scale(sm, c_kr))
                else:
                    rm = jet_add(rm, jet_scale(sm, c_kr))
                rp = jet_add(rp, jet_scale(sp, c_kr))

            bk = bp[k]
            ym = jet_mul(bk, rm)
            yp = jet_mul(bk, rp)
            # Solve 2x2 linear system [I - (-b)^k P^T] M = (ym, yp)^T directly:
            # P^T = [[p_same, p_diff], [p_diff, p_same]].
            sbk = jet_neg(bk) if (k % 2 == 1) else bk
            a_diag = jet_sub(one_jet, jet_scale(sbk, p_same))
            a_off = jet_neg(jet_scale(sbk, p_diff))
            det2 = jet_sub(jet_mul(a_diag, a_diag), jet_mul(a_off, a_off))
            mm = jet_div(jet_sub(jet_mul(a_diag, ym), jet_mul(a_off, yp)), det2)
            mp_ = jet_div(jet_sub(jet_mul(a_diag, yp), jet_mul(a_off, ym)), det2)

            Msum[k][n] = jet_add(mm, mp_)
            PM[k][n] = (
                jet_add(jet_scale(mm, p_same), jet_scale(mp_, p_diff)),
                jet_add(jet_scale(mm, p_diff), jet_scale(mp_, p_same)),
            )
        if n < order:
            build_suffix_table(n)

        cn = zjet
        for k in range(1, n + 1):
            w = iv.mpf((-1) ** (k + 1)) / iv.mpf(k)
            cn = jet_add(cn, jet_scale(Msum[k][n], w))
        if n == 2:
            c2_val = cn[0]
        total_jet = jet_add(total_jet, jet_scale(cn, delta ** n))

    derivs = [total_jet[j] * iv.mpf(math.factorial(j)) for j in range(degree + 1)]
    return derivs, c2_val


def series_with_cauchy_tail(b_iv, delta_iv, phi_iv, order, degree=1):
    """Rigorous enclosure of [G, G_b, ...] on interval b_iv subset (0, 1)
    by adding complex-b Cauchy derivative bounds k! * T_N / h^k.
    """
    b_iv = iv.mpf(b_iv)
    delta_iv = iv.mpf(delta_iv)
    phi_iv = iv.mpf(phi_iv)
    assert bool(b_iv > 0) and bool(b_iv < 1), f"b_iv out of (0,1): {b_iv}"
    # Choose complex disk radius h <= min(b_lo, 1 - b_hi) / 80
    b_lo = b_iv.a
    one_minus_b_hi = (ONE - b_iv).a
    h = min(b_lo, one_minus_b_hi) / iv.mpf(100)
    assert bool(h > 0)
    B = (b_iv + h).b
    assert bool(B < 1)
    R_delta = (ONE - B) ** 2 / (iv.mpf(8) * B)
    q = iv_abs_upper(delta_iv) / R_delta
    assert bool(q < 1), f"Insufficient P7 radius: q={q}"
    T_N = iv.log(iv.mpf(2)) * (q ** (order + 1)) / (ONE - q)
    raw, _ = two_state_vector_series(b_iv, delta_iv, phi_iv, order, degree=degree)
    tails = [iv.mpf(math.factorial(k)) * T_N / (h ** k) for k in range(degree + 1)]
    enclosed = [iv_ball(raw[k], tails[k]) for k in range(degree + 1)]
    return enclosed, R_delta, tails


def enclose_threshold_series(s_iv, delta_iv, phi_iv, order=24, need_deriv=True):
    """Independently bracket the unique root b in (0, 1) of r(b) = log(s/(1-b)) + G(b) = 0
    using interval Mean Value / sign-change theorem, then enclose D(s) and D'(s).
    """
    s_iv = iv.mpf(s_iv)
    delta_iv = iv.mpf(delta_iv)
    phi_iv = iv.mpf(phi_iv)

    # Untrusted midpoint Newton iterations to pick bracket center b0
    d0 = delta_iv ** 2 * F_ar1(s_iv, phi_iv)
    c_init = (ONE - s_iv) * iv.exp(d0)
    lam_init = (s_iv + iv.sqrt(s_iv ** 2 + iv.mpf(4) * c_init)) * HALF
    b0 = iv.mpf((c_init / (lam_init ** 2)).mid)
    for _ in range(6):
        g_mid, _ = two_state_vector_series(b0, delta_iv, phi_iv, min(order, 28), degree=1)
        r_mid = iv.log(s_iv / (ONE - b0)) + g_mid[0]
        rb_mid = ONE / (ONE - b0) + g_mid[1]
        b0 = iv.mpf((b0 - r_mid / rb_mid).mid)

    # Certified residual at b0 and lower slope on outer neighborhood [b0 - h_out, b0 + h_out]
    g0, R_delta, tails0 = series_with_cauchy_tail(b0, delta_iv, phi_iv, order, degree=0)
    r0 = iv.log(s_iv / (ONE - b0)) + g0[0]
    h_out = b0 * iv.mpf("1e-10")
    b_outer = iv_ball(b0, h_out)
    g_outer, _, _ = series_with_cauchy_tail(b_outer, delta_iv, phi_iv, order, degree=1)
    rb_outer = ONE / (ONE - b_outer) + g_outer[1]
    m_slope = rb_outer.a
    assert bool(m_slope > 0), f"Non-positive slope lower bound: {m_slope}"

    r0_up = iv_abs_upper(r0)
    h_root = (iv.mpf(2) * r0_up / m_slope + b0 * iv.mpf("1e-78")).b
    assert bool(h_root < h_out), f"Root radius {h_root} >= outer {h_out}"
    assert bool(r0_up < m_slope * h_root), "Sign-bracket condition |r(b0)| < m * h_root failed"

    b_root = iv_ball(b0, h_root)
    D_iv = iv.log(b_root) + iv.mpf(2) * iv.log(s_iv / (ONE - b_root)) - iv.log(ONE - s_iv)
    out = {
        "b": b_root,
        "D": D_iv,
        "rate_tail": tails0[0],
        "delta_radius": R_delta,
        "root_halfwidth_b": h_root,
        "rate_residual_center": r0,
        "lower_slope_rb": m_slope,
    }
    if need_deriv:
        g_root, _, tails1 = series_with_cauchy_tail(b_root, delta_iv, phi_iv, order, degree=1)
        denom = ONE + (ONE - b_root) * g_root[1]
        assert bool(denom > 0)
        J_val = b_root - ONE + s_iv + b_root * (iv.mpf(2) - s_iv) * (ONE - b_root) * g_root[1]
        Ds_iv = J_val / (s_iv * (ONE - s_iv) * b_root * denom)
        out["Ds"] = Ds_iv
        out["derivative_tail_Gb"] = tails1[1]
    return out


def stationary_J_and_slope(b_iv, delta_iv, phi_iv, order=24, rigorous=True):
    """Evaluate J(b), J_b(b), and s(b) along s(b) = (1 - b) exp(-G(b))."""
    if rigorous:
        g, _, tails2 = series_with_cauchy_tail(b_iv, delta_iv, phi_iv, order, degree=2)
    else:
        g, _ = two_state_vector_series(b_iv, delta_iv, phi_iv, order, degree=2)
        tails2 = [ZERO, ZERO, ZERO]
    exp_mG = iv.exp(-g[0])
    s_b = (ONE - b_iv) * exp_mG
    sb = -exp_mG * (ONE + (ONE - b_iv) * g[1])
    C = b_iv * (iv.mpf(2) - s_b) * (ONE - b_iv)
    Cb = (ONE - iv.mpf(2) * b_iv) * (iv.mpf(2) - s_b) - b_iv * (ONE - b_iv) * sb
    J = b_iv - ONE + s_b + C * g[1]
    Jb = ONE + sb + Cb * g[1] + C * g[2]
    return J, Jb, s_b, tails2


def main():
    t0 = time.monotonic()
    primary_records = {
        r["claim"]: r["enclosure"]
        for r in json.loads(GAP4_JSON.read_text())
    }
    results = []

    def log_check(claim_id, status, summary, details, overlaps_independent=True):
        row = {
            "claim_id": claim_id,
            "status": status,
            "overlaps_independent": overlaps_independent,
            "summary": summary,
            "details": details,
        }
        results.append(row)
        print(f"[{status}] {claim_id}: {summary}", flush=True)

    # ------------------------------------------------------------------
    # GATE G0_series: pre-registered tolerance 1e-55
    # ------------------------------------------------------------------
    max_g0_width = ZERO
    for st in ["0.35", "0.75", "0.99"]:
        s = iv.mpf(st)
        b = ONE - s
        derivs, c2 = two_state_vector_series(b, ZERO, iv.mpf("0.8"), order=4, degree=0)
        assert (0 in derivs[0]) and bool(derivs[0].delta < iv.mpf("1e-55"))
        disc = c2 + (b / (ONE + b)) * F_ar1(s, iv.mpf("0.8"))
        assert (0 in disc) and bool(disc.delta < iv.mpf("1e-55"))
        if bool(disc.delta > max_g0_width):
            max_g0_width = disc.delta
    log_check(
        "G0_series",
        "PASS",
        f"3 zero-noise and 3 second-order benchmarks passed (max width {mpmath.nstr(iv_hi_mpf(max_g0_width), 4)} < 1e-55)",
        {"max_width": mpmath.nstr(iv_hi_mpf(max_g0_width), 6)},
    )

    # ------------------------------------------------------------------
    # CLAIM 1: P5 proof constants + CE1 analytic margin
    # ------------------------------------------------------------------
    R = ONE / iv.mpf(256)
    E14 = iv.exp(ONE / iv.mpf(4))
    p5_vals = {
        "R": R,
        "invariant_ball_factor": ONE + iv.mpf(2) * E14 / (ONE - iv.mpf(4) * R),
        "contraction_factor": iv.mpf(2) * R * E14 / (ONE - iv.mpf(4) * R) ** 2,
        "f_error_coefficient": (iv.mpf(8) * E14 + iv.mpf(8)) / (ONE - iv.mpf(4) * R),
        "19R": iv.mpf(19) * R,
        "Rouche_lower": iv.mpf(3) / iv.mpf(2) - E14,
        "R2_over_4": R ** 2 / iv.mpf(4),
        "R3_over_2": R ** 3 / iv.mpf(2),
    }
    assert bool(p5_vals["invariant_ball_factor"] < iv.mpf(4))
    assert bool(p5_vals["contraction_factor"] < ONE)
    assert bool(p5_vals["f_error_coefficient"] < iv.mpf(19))
    assert bool(p5_vals["19R"] < iv.mpf("0.075"))
    assert bool(iv.mpf("0.075") < p5_vals["Rouche_lower"])
    assert bool(p5_vals["Rouche_lower"] > iv.mpf("0.21"))

    A_ce1 = -ONE / iv.mpf(16)
    eps_ce1 = iv.mpf("1e-8")
    allowed_eps = abs(A_ce1) / iv.mpf(262144)
    deriv_rem = iv.mpf(2) * eps_ce1 / (R ** 2)
    Ds_analytic_ce1 = iv_ball(-A_ce1, deriv_rem)
    assert bool(eps_ce1 < allowed_eps)
    assert bool(Ds_analytic_ce1 > ONE / iv.mpf(32))

    ov1 = all(
        intervals_overlap(v, parse_arb_str_to_iv(primary_records["P5 numerical constants"][k]))
        for k, v in p5_vals.items()
    ) and intervals_overlap(
        Ds_analytic_ce1, parse_arb_str_to_iv(primary_records["CE1 analytic margin"]["Ds"])
    )
    assert ov1
    log_check(
        "Claim_1_P5_constants_and_CE1_margin",
        "PASS",
        "All 8 P5 proof constants and analytic CE1 slope enclosure Ds > 1/32 verified",
        {
            **{k: format_outward_pm(v) for k, v in p5_vals.items()},
            "CE1_allowed_eps": format_outward_pm(allowed_eps),
            "CE1_analytic_Ds": format_outward_pm(Ds_analytic_ce1),
        },
        overlaps_independent=ov1,
    )

    # ------------------------------------------------------------------
    # CLAIM 2 + DATED CORRECTION 2: Sign-table & ring critical values
    # ------------------------------------------------------------------
    phi_m35 = iv.mpf("-0.35")
    st_m35 = s_star_ar1(phi_m35)
    a_crit = (iv.mpf(3) + iv.sqrt(iv.mpf(17))) / iv.mpf(4)
    s_stay_true = ONE - ONE / a_crit
    s_stay_half = (iv.mpf(5) - iv.sqrt(iv.mpf(17))) / iv.mpf(2)
    s_stay_quarter_typo = (iv.mpf(5) - iv.sqrt(iv.mpf(17))) / iv.mpf(4)
    # Verify exact algebraic identity 1 - 1/a_crit == (5 - sqrt(17))/2 and exclusion of /4
    diff_true = s_stay_true - s_stay_half
    assert (0 in diff_true) and bool(diff_true.delta < iv.mpf("1e-80"))
    assert not intervals_overlap(s_stay_true, s_stay_quarter_typo)

    crit_vals = {
        "phi_crit": iv.mpf(2) - iv.sqrt(iv.mpf(5)),
        "phi_exception": -HALF,
        "s_star(-.35)": st_m35,
        "F_at_s_star": F_ar1(st_m35, phi_m35),
        "ring_a_crit": a_crit,
        "ring_staying_crit": s_stay_true,
        "secondary_printed_staying_expression": s_stay_quarter_typo,
        "ring_c3_crit": iv.mpf(2),
        "third_peak_c3_5": iv.mpf(46) / iv.mpf(13),
    }
    ov2 = all(
        intervals_overlap(
            v, parse_arb_str_to_iv(primary_records["Sign-table and ring critical values"][k])
        )
        for k, v in crit_vals.items()
    )
    assert ov2
    log_check(
        "Claim_2_Critical_values_and_Correction_2",
        "PASS",
        f"Critical values verified; Correction 2 confirmed: 1-1/a_crit = (5-sqrt(17))/2 = {format_outward_pm(s_stay_true, 18, 3)} != (5-sqrt(17))/4 = {format_outward_pm(s_stay_quarter_typo, 18, 3)}",
        {k: format_outward_pm(v) for k, v in crit_vals.items()},
        overlaps_independent=ov2,
    )

    # ------------------------------------------------------------------
    # CLAIM 3: Counterexample CE1 (delta=1/2, phi=-3/4, s=1-1e-8)
    # ------------------------------------------------------------------
    ce1 = enclose_threshold_series(
        ONE - iv.mpf("1e-8"), iv.mpf("0.5"), iv.mpf("-0.75"), order=20, need_deriv=True
    )
    assert bool(ce1["D"] < 0) and bool(ce1["Ds"] > ONE / iv.mpf(32))
    ov_ce1 = intervals_overlap(ce1["D"], parse_arb_str_to_iv(primary_records["CE1"]["D"])) and intervals_overlap(
        ce1["Ds"], parse_arb_str_to_iv(primary_records["CE1"]["Ds"])
    )
    assert ov_ce1
    log_check(
        "Claim_3_CE1_nonlinear_series",
        "PASS",
        f"CE1 certified: D = {format_outward_pm(ce1['D'])}, Ds = {format_outward_pm(ce1['Ds'])} > 1/32",
        {k: format_outward_pm(v) for k, v in ce1.items()},
        overlaps_independent=ov_ce1,
    )

    # ------------------------------------------------------------------
    # CLAIM 4a: Counterexample CE2 via P7 order-88 series (s=0.9, delta=0.4, phi=-0.75)
    # ------------------------------------------------------------------
    ce2 = enclose_threshold_series(
        iv.mpf("0.9"), iv.mpf("0.4"), iv.mpf("-0.75"), order=88, need_deriv=True
    )
    assert bool(ce2["D"] < 0) and bool(ce2["Ds"] > 0)
    ov_ce2 = intervals_overlap(ce2["D"], parse_arb_str_to_iv(primary_records["CE2"]["D"])) and intervals_overlap(
        ce2["Ds"], parse_arb_str_to_iv(primary_records["CE2"]["Ds"])
    )
    assert ov_ce2
    log_check(
        "Claim_4a_CE2_series_order88",
        "PASS",
        f"CE2 series (order 88) certified: D = {format_outward_pm(ce2['D'])}, Ds = {format_outward_pm(ce2['Ds'])}",
        {
            **{k: format_outward_pm(v) for k, v in ce2.items()},
            "D_raw_lo": mpmath.nstr(iv_lo_mpf(ce2["D"]), 35),
            "D_raw_hi": mpmath.nstr(iv_hi_mpf(ce2["D"]), 35),
        },
        overlaps_independent=ov_ce2,
    )

    # ------------------------------------------------------------------
    # CLAIM 5: Counterexample CE3 (phi=-1/2, A=0) + P5 long-stay enclosures
    # ------------------------------------------------------------------
    ce3 = enclose_threshold_series(
        iv.mpf("0.999999"), iv.mpf("0.6"), iv.mpf("-0.5"), order=20, need_deriv=True
    )
    ce3["D_over_eps2"] = ce3["D"] / iv.mpf("1e-12")
    ce3["Ds_over_eps"] = ce3["Ds"] / iv.mpf("1e-6")
    assert bool(ce3["D"] < 0) and bool(ce3["Ds"] > 0)
    ov_ce3 = intervals_overlap(ce3["D"], parse_arb_str_to_iv(primary_records["CE3"]["D"])) and intervals_overlap(
        ce3["Ds"], parse_arb_str_to_iv(primary_records["CE3"]["Ds"])
    )
    assert ov_ce3

    p5_subchecks = {}
    for ph_str in [".8", "-.75", "-.5"]:
        d_p5, p_p5 = iv.mpf("0.6"), iv.mpf(ph_str)
        A_p5 = d_p5 ** 2 * (HALF + p_p5)
        for ep_str in [".02", ".005", ".001"]:
            ep_iv = iv.mpf(ep_str)
            out_ep = enclose_threshold_series(ONE - ep_iv, d_p5, p_p5, order=30, need_deriv=False)
            ext_B = (out_ep["D"] - A_p5 * ep_iv) / (ep_iv ** 2)
            cname = f"P5 epsilon={ep_str} phi={ph_str}"
            assert intervals_overlap(out_ep["D"], parse_arb_str_to_iv(primary_records[cname]["D"]))
            assert intervals_overlap(ext_B, parse_arb_str_to_iv(primary_records[cname]["extracted_B"]))
            p5_subchecks[cname] = {
                "D": format_outward_pm(out_ep["D"]),
                "extracted_B": format_outward_pm(ext_B),
            }
        if not (0 in A_p5):
            ep_d = abs(A_p5) / iv.mpf(524288)
            out_d = enclose_threshold_series(ONE - ep_d, d_p5, p_p5, order=20, need_deriv=True)
            assert bool(out_d["Ds"] * A_p5 < 0) and bool(abs(out_d["Ds"] + A_p5) < abs(A_p5) * HALF)
            cname_d = f"P5 explicit derivative test phi={ph_str}"
            assert intervals_overlap(out_d["Ds"], parse_arb_str_to_iv(primary_records[cname_d]["Ds"]))
            p5_subchecks[cname_d] = {"Ds": format_outward_pm(out_d["Ds"])}

    log_check(
        "Claim_5_CE3_and_P5_long_stay_grid",
        "PASS",
        f"CE3 (D/eps^2 = {format_outward_pm(ce3['D_over_eps2'], 18, 3)}, Ds/eps = {format_outward_pm(ce3['Ds_over_eps'], 18, 3)}) and all 11 P5 long-stay/derivative cases certified",
        {
            "CE3": {k: format_outward_pm(v) for k, v in ce3.items()},
            "P5_grid": p5_subchecks,
        },
        overlaps_independent=True,
    )

    # ------------------------------------------------------------------
    # CLAIM 6: Alternating-host exact family CE4 (16 cases)
    # ------------------------------------------------------------------
    ce4_details = {}
    for d_str in [".2", ".5", ".6", ".9"]:
        d_iv = iv.mpf(d_str)
        ell_iv = -iv.log(ONE - d_iv ** 2) * HALF
        for s_str in ["0", ".2", ".7", ".99"]:
            s_iv = iv.mpf(s_str)
            Q = iv.sqrt(d_iv ** 2 + (ONE - d_iv ** 2) * s_iv ** 2)
            D_iv = iv.log((ONE + s_iv) / (ONE + Q))
            Ds_iv = ONE / (ONE + s_iv) - (ONE - d_iv ** 2) * s_iv / (Q * (ONE + Q))
            c_iv = (ONE - s_iv ** 2) / (ONE + Q)
            tr = s_iv ** 2 + iv.mpf(2) * c_iv
            det = c_iv ** 2 * (ONE - d_iv ** 2)
            perron = (tr + iv.sqrt(tr ** 2 - iv.mpf(4) * det)) * HALF
            assert bool(D_iv < 0) and bool(Ds_iv > 0)
            assert (0 in (perron - ONE)) and bool(iv_abs_upper(perron - ONE) < iv.mpf("1e-80"))
            if s_str == "0":
                assert not intervals_overlap(D_iv, ell_iv)
            cname = f"CE4 delta={d_str} s={s_str}"
            assert intervals_overlap(D_iv, parse_arb_str_to_iv(primary_records[cname]["D"]))
            assert intervals_overlap(Ds_iv, parse_arb_str_to_iv(primary_records[cname]["Ds"]))
            ce4_details[cname] = {
                "D": format_outward_pm(D_iv),
                "Ds": format_outward_pm(Ds_iv),
            }
    log_check(
        "Claim_6_CE4_alternating_16_cases",
        "PASS",
        "All 16 alternating-host (phi=-1) cases certified: D < 0, Ds > 0, Perron = 1, D(0) != ell",
        ce4_details,
        overlaps_independent=True,
    )

    # ------------------------------------------------------------------
    # CLAIMS 7 & 8 + DATED CORRECTION 1: Stationary points (delta=.004, .002)
    # ------------------------------------------------------------------
    stat_roots = []
    secant_h = iv.mpf("0.00002")
    for idx, (dt_str, printed_str) in enumerate(
        [(".004", "0.599487404570512478"), (".002", "0.599495940314231187")], start=7
    ):
        d_iv = iv.mpf(dt_str)
        s_printed = iv.mpf(printed_str)
        b0 = iv.mpf((ONE - s_printed).mid)
        for _ in range(6):
            jm, jbm, _, _ = stationary_J_and_slope(b0, d_iv, phi_m35, order=16, rigorous=False)
            b0 = iv.mpf((b0 - jm / jbm).mid)

        j0, _, _, _ = stationary_J_and_slope(b0, d_iv, phi_m35, order=24, rigorous=True)
        _, jb_outer, _, tails2 = stationary_J_and_slope(
            iv_ball(b0, iv.mpf("1e-12")), d_iv, phi_m35, order=24, rigorous=True
        )
        assert bool(jb_outer > 0) or bool(jb_outer < 0)
        m_jb = iv_abs_lower(jb_outer)
        j0_up = iv_abs_upper(j0)
        h_b = (iv.mpf(2) * j0_up / m_jb + iv.mpf("1e-68")).b
        assert bool(h_b < iv.mpf("1e-12")) and bool(j0_up < m_jb * h_b)

        _, _, s_stat, _ = stationary_J_and_slope(
            iv_ball(b0, h_b), d_iv, phi_m35, order=24, rigorous=True
        )
        shift_iv = s_stat - st_m35
        stat_roots.append(s_stat)

        # Evaluate TRUE derivative D'(s) at memA's printed s -> strictly > 0!
        at_printed = enclose_threshold_series(s_printed, d_iv, phi_m35, order=24, need_deriv=True)
        assert bool(at_printed["Ds"] > 0), f"Expected Ds > 0 at printed s for delta={dt_str}"

        # Also evaluate memA_checks.py's central secant (D(s+h)-D(s-h))/(2h) with h = 2e-5 at printed s
        dp_sec = enclose_threshold_series(s_printed + secant_h, d_iv, phi_m35, order=24, need_deriv=False)["D"]
        dm_sec = enclose_threshold_series(s_printed - secant_h, d_iv, phi_m35, order=24, need_deriv=False)["D"]
        secant_at_printed = (dp_sec - dm_sec) / (iv.mpf(2) * secant_h)

        cname = f"Actual stationary point delta={dt_str}"
        ov_stat = (
            intervals_overlap(s_stat, parse_arb_str_to_iv(primary_records[cname]["s_stationary"]))
            and intervals_overlap(shift_iv, parse_arb_str_to_iv(primary_records[cname]["shift"]))
            and intervals_overlap(
                at_printed["Ds"], parse_arb_str_to_iv(primary_records[cname]["Ds_at_printed_s"])
            )
        )
        assert ov_stat
        log_check(
            f"Claim_{idx}_Stationary_point_delta_{dt_str}",
            "PASS",
            f"s_stat = {format_outward_pm(s_stat)}; at memA printed s, true Ds = {format_outward_pm(at_printed['Ds'], 18, 3)} > 0 while h=2e-5 secant = {format_outward_pm(secant_at_printed, 6, 3)}",
            {
                "s_stationary": format_outward_pm(s_stat),
                "shift_from_s_star": format_outward_pm(shift_iv),
                "printed_s": printed_str,
                "true_Ds_at_printed_s": format_outward_pm(at_printed["Ds"]),
                "secant_2e5_at_printed_s": format_outward_pm(secant_at_printed, 12, 3),
                "J_slope_on_bracket": format_outward_pm(jb_outer, 12, 3),
                "b_root_halfwidth": format_outward_pm(h_b, 12, 3),
                "second_deriv_tail_Gbb": format_outward_pm(tails2[2], 12, 3),
            },
            overlaps_independent=ov_stat,
        )

    disp_ratio = (stat_roots[0] - st_m35) / (stat_roots[1] - st_m35)
    assert bool(disp_ratio > iv.mpf("3.99")) and bool(disp_ratio < iv.mpf("4.01"))
    assert intervals_overlap(
        disp_ratio, parse_arb_str_to_iv(primary_records["Stationary-point displacement ratio"]["ratio"])
    )
    log_check(
        "Claim_8b_Stationary_displacement_ratio",
        "PASS",
        f"Displacement ratio = {format_outward_pm(disp_ratio)}",
        {"ratio": format_outward_pm(disp_ratio)},
        overlaps_independent=True,
    )

    # ------------------------------------------------------------------
    # CLAIM 9: P7 moderate-noise validation point & small-noise derivative grid
    # ------------------------------------------------------------------
    p7_val = enclose_threshold_series(
        iv.mpf("0.92"), iv.mpf("0.5"), iv.mpf("0.7"), order=88, need_deriv=True
    )
    ov_p7 = intervals_overlap(
        p7_val["D"],
        parse_arb_str_to_iv(primary_records["P7 validation point s=.92 delta=.5 phi=.7"]["D"]),
    )
    assert ov_p7

    sn_details = {}
    d_sn = iv.mpf("0.002")
    for j_idx, s_pt in enumerate([st_m35 - iv.mpf("0.04"), st_m35, st_m35 + iv.mpf("0.04")], start=1):
        sn = enclose_threshold_series(s_pt, d_sn, phi_m35, order=32, need_deriv=True)
        cname = f"Small noise phi=-.35 point={j_idx}"
        assert intervals_overlap(
            sn["D"] / (d_sn ** 2), parse_arb_str_to_iv(primary_records[cname]["D_over_delta2"])
        )
        assert intervals_overlap(
            sn["Ds"] / (d_sn ** 2), parse_arb_str_to_iv(primary_records[cname]["true_Ds_over_delta2"])
        )
        sn_details[cname] = {
            "D_over_delta2": format_outward_pm(sn["D"] / (d_sn ** 2)),
            "true_Ds_over_delta2": format_outward_pm(sn["Ds"] / (d_sn ** 2)),
        }

    log_check(
        "Claim_9_P7_validation_and_small_noise_grid",
        "PASS",
        f"P7 validation s=0.92 (order 88) D = {format_outward_pm(p7_val['D'])} and small-noise phi=-0.35 grid certified",
        {
            "P7_s092": {k: format_outward_pm(v) for k, v in p7_val.items()},
            "small_noise_phi_m35": sn_details,
        },
        overlaps_independent=True,
    )

    elapsed = round(time.monotonic() - t0, 2)
    payload = {
        "script": "verify_gap4_part1_series.py",
        "mpmath_version": mpmath.__version__,
        "iv_dps": iv.dps,
        "elapsed_seconds": elapsed,
        "all_passed": all(r["status"] == "PASS" for r in results),
        "results": results,
    }
    (HERE / "verify_gap4_part1_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Part 1 completed: {len(results)} check groups PASSED in {elapsed}s", flush=True)


if __name__ == "__main__":
    main()
