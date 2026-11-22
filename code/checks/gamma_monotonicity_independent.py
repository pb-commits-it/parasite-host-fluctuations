#!/usr/bin/env python3
"""Checks for notes/gamma-monotonicity-attempt-tertiary.md.

Claim under test (the inequality (R4.2) of studies/referee-notes-18/referee_report_18.md):
for p > 0, kappa > 0 let T have density proportional to t^(p-1) (1+t)^(-p) exp(-kappa t),
W = ln(1+T), m = E W, V = kappa Cov(W, T), s = exp(-m).  Then  V < (1 - s)/(2 - s).

The note proves it.  This script does NOT prove anything; it tests every step of the
written proof and every number cited in the note.  Nothing here is interval arithmetic.

Sections (tolerances fixed before the run and printed with each line):
  S   exact symbolic verification (sympy) of every algebraic identity used in the proof
  N1  the quadrature used below against a finer rule, against mpmath's Tricomi function
      and against mpmath's own adaptive quadrature
  N2  the two-copy representation 1 - 2V = E[1/(1+T+T')] (two independent ways)
  N3  the three differential equations for a, V, m, and those for Y, Xi, Theta,
      by central differences in x = ln kappa
  N4  the integration-by-parts (Stein) identities behind the equation for V
  N5  sign scan: e > 0, Theta < 0, Y > 0, ln(sigma/s) > 0, F < 0 on a grid of (p, kappa)
  N6  the integral formula ln(sigma/s) = int_x^inf 2Y/(1+R) dy
  N7  by-products: R = d a/d p, the stationarity identity for m, the trigamma identity
  N8  the large-p boundary layer: Bessel limit, and the minimum slack against 1/(16 p^1.5)
  N9  the elementary bounds used for boundedness at large kappa

Run:  python3 -I checks/gamma_monotonicity_independent.py
Working precision: 50 decimal digits unless a line says otherwise.
"""
import sys
import time

import mpmath as mp
import sympy as sp
from mpmath.calculus.quadrature import GaussLegendre

DPS = 50
mp.mp.dps = DPS
FAILS = []
T0 = time.time()


def report(tag, ok, text):
    line = "%-5s %-4s %s" % (tag, "PASS" if ok else "FAIL", text)
    print(line, flush=True)
    if not ok:
        FAILS.append(line)


# --------------------------------------------------------------------------------------
# Numerical evaluation of the stationary moments: composite Gauss-Legendre in z = ln t
# --------------------------------------------------------------------------------------
_GL = {}


def gl(degree):
    key = (degree, mp.mp.prec)
    if key not in _GL:
        _GL[key] = GaussLegendre(mp.mp).calc_nodes(degree, mp.mp.prec)
    return _GL[key]


def logw(z, p, kappa):
    """log density of z = ln T up to a constant: p z - p ln(1+e^z) - kappa e^z."""
    ez = mp.exp(z)
    if z > 0:
        return -p * mp.log1p(1 / ez) - kappa * ez
    return p * z - p * mp.log1p(ez) - kappa * ez


def breakpoints(p, kappa):
    r = p / kappa
    tstar = 2 * r / (1 + mp.sqrt(1 + 4 * r))  # mode in z: kappa t (1+t) = p
    zc = mp.log(tstar)
    curv = p * tstar / (1 + tstar) ** 2 + kappa * tstar
    sd = 1 / mp.sqrt(curv)
    l0 = logw(zc, p, kappa)
    cut = (mp.mp.dps + 20) * mp.log(10)

    def edge(sign):
        step = min(sd, mp.mpf(1))
        z = zc
        while True:
            z2 = z + sign * step
            if logw(z2, p, kappa) - l0 < -cut:
                break
            z = z2
            step *= mp.mpf(3) / 2
        lo, hi = z, z2
        for _ in range(6):
            mid = (lo + hi) / 2
            if logw(mid, p, kappa) - l0 < -cut:
                hi = mid
            else:
                lo = mid
        return hi

    zl = edge(-1)
    zr = edge(+1)
    pts = {zl, zr}
    for k in (0, 0.5, 1, 2, 3.5, 5.5, 8, 11, 15, 20, 26, 33, 41, 50, 60, 72, 86, 100):
        for sg in (-1, 1):
            z = zc + sg * k * sd
            if zl < z < zr:
                pts.add(z)
    lo = max(zl, mp.mpf(-120))
    g = mp.ceil(lo / 4) * 4
    while g < zr:
        if g > zl:
            pts.add(mp.mpf(g))
        g += 4
    g = mp.mpf(-240)
    while g > zl:
        pts.add(g)
        g *= 2
    return sorted(pts), l0


def moments(p, kappa, degree=6, want_nodes=False):
    """Stationary moments of T (density prop. to t^(p-1)(1+t)^(-p)e^(-kappa t))."""
    p = mp.mpf(p)
    kappa = mp.mpf(kappa)
    pts, l0 = breakpoints(p, kappa)
    nodes = gl(degree)
    S = [mp.mpf(0)] * 8
    keep = []
    for i in range(len(pts) - 1):
        A, B = pts[i], pts[i + 1]
        h = (B - A) / 2
        c = (B + A) / 2
        for x, w in nodes:
            z = c + h * x
            ez = mp.exp(z)
            if z > 0:
                l1 = mp.log1p(1 / ez)
                lw = -p * l1 - kappa * ez
                W = z + l1
            else:
                W = mp.log1p(ez)
                lw = p * z - p * W - kappa * ez
            f = mp.exp(lw - l0) * w * h
            q = 1 / (1 + ez)
            S[0] += f
            S[1] += f * W
            S[2] += f * ez
            S[3] += f * W * ez
            S[4] += f * q
            S[5] += f * W * q
            S[6] += f * ez * ez
            S[7] += f * z
            if want_nodes:
                keep.append((ez, f))
    Z = S[0]
    m = S[1] / Z
    ET = S[2] / Z
    d = dict(p=p, kappa=kappa, m=m, ET=ET, a=kappa * ET, V=kappa * (S[3] / Z - m * ET),
             s=mp.exp(-m), Eq=S[4] / Z, EWq=S[5] / Z, ET2=S[6] / Z, ElnT=S[7] / Z,
             EWT=S[3] / Z, lnZ=mp.log(Z) + l0, nint=len(pts) - 1)
    if want_nodes:
        d["nodes"] = [(t, f / Z) for t, f in keep]
    return d


def derived(d):
    """The quantities of the proof, from (p, kappa, a, V, s)."""
    p, k, a, V, s = d["p"], d["kappa"], d["a"], d["V"], d["s"]
    R = 1 - 2 * V
    tau = V / R
    rho = 1 / R
    Psi = (1 - R * R) / 4
    e = a - k * tau
    Y = Psi - e
    Xi = p * R - a + V * V * (1 + k / R)
    Theta = k * rho * Xi - Psi ** 2
    F = V * (2 - s) - (1 - s)
    sig = 2 * R / (1 + R)
    return dict(R=R, tau=tau, rho=rho, Psi=Psi, e=e, Y=Y, Xi=Xi, Theta=Theta, F=F,
                sig=sig, lnratio=mp.log(sig / s))


def state(p, kappa):
    d = moments(p, kappa)
    d.update(derived(d))
    return d


# --------------------------------------------------------------------------------------
def section_S():
    print("\n== S. Exact symbolic identities (sympy); each line must simplify to 0 ==")
    k, a, R, p, s, V = sp.symbols("kappa a R p s V", positive=True)

    def Dx(f):  # d/dx, x = ln kappa, along a' = a^2+ka-pk, R' = (k+2a)R-k
        return (sp.diff(f, k) * k + sp.diff(f, a) * (a ** 2 + k * a - p * k)
                + sp.diff(f, R) * ((k + 2 * a) * R - k))

    tau = (1 - R) / (2 * R)
    rho = 1 / R
    Psi = (1 - R ** 2) / 4
    e = a - k * tau
    Y = Psi - e
    Vr = (1 - R) / 2
    Xi = p * R - a + Vr ** 2 * (1 + k / R)
    Th = k * rho * Xi - Psi ** 2
    checks = [
        ("Psi = tau(1+tau)/rho^2", Psi - tau * (1 + tau) / rho ** 2),
        ("R' = 2 R e", Dx(R) - 2 * R * e),
        ("2 R Y = R(1-R^2)/2 - R'", 2 * R * Y - (R * (1 - R ** 2) / 2 - Dx(R))),
        ("Xi' = (k rho + 2e) Xi - (Psi^2 - Y^2)",
         Dx(Xi) - ((k * rho + 2 * e) * Xi - (Psi ** 2 - Y ** 2))),
        ("Theta' = (1 + k rho) Theta + Psi^2 + k rho Y^2 + 2 Psi R^2 e",
         Dx(Th) - ((1 + k * rho) * Th + Psi ** 2 + k * rho * Y ** 2 + 2 * Psi * R ** 2 * e)),
        ("Y' = (k rho + 2 Psi + R^2) Y - (Psi R^2 - Theta + Y^2)",
         Dx(Y) - ((k * rho + 2 * Psi + R ** 2) * Y - (Psi * R ** 2 - Th + Y ** 2))),
        ("[k tau(1+tau)]' = k rho^2 Y", Dx(k * tau * (1 + tau)) - k * rho ** 2 * Y),
        ("(1-s)/(2-s) - V = (R(2-s) - s)/(2(2-s)) at V=(1-R)/2",
         ((1 - s) / (2 - s) - Vr) - (R * (2 - s) - s) / (2 * (2 - s))),
        ("d ln(2R/(1+R))/dx - (1-R)/2 = -2Y/(1+R)",
         Dx(sp.log(2 * R / (1 + R))) - (1 - R) / 2 + 2 * Y / (1 + R)),
    ]
    for name, ex in checks:
        z = sp.simplify(ex)
        report("S", z == 0, "%s  ->  %s" % (name, z))

    # the observation quoted in the task: F = V(2-s)-(1-s), at F = 0
    def Dx2(f):
        return (sp.diff(f, k) * k + sp.diff(f, a) * (a ** 2 + k * a - p * k)
                + sp.diff(f, V) * ((k + 2 * a) * V - a) + sp.diff(f, s) * (s * V))

    F = V * (2 - s) - (1 - s)
    z = sp.simplify(Dx2(F).subs(V, (1 - s) / (2 - s))
                    - (k * (1 - s) - a * s + s * (1 - s) / (2 - s) ** 2))
    report("S", z == 0, "dF/dx at F=0 equals k(1-s) - a s + s(1-s)/(2-s)^2  ->  %s" % z)

    # Bessel limit: g' = (g^2-z^2)/z ; G1 = g^2-z^2-z ; G2 = z^2 g + z^2/2 + g^2/2 - g^3
    g, zz = sp.symbols("g z", positive=True)
    gp = (g ** 2 - zz ** 2) / zz
    G2 = zz ** 2 * g + zz ** 2 / 2 + g ** 2 / 2 - g ** 3
    dG2 = sp.diff(G2, zz) + sp.diff(G2, g) * gp
    # on G2 = 0: g^2 - z^2 = (g^2+z^2)/(2g); claim dG2 = -(g^2-z^2)^2/(4 g^2 z) there
    lam = sp.symbols("lam")
    rem = sp.simplify(dG2 + (g ** 2 - zz ** 2) ** 2 / (4 * g ** 2 * zz) - lam * G2)
    sol = sp.solve(sp.Poly(sp.numer(sp.together(rem)), zz).coeffs(), lam, dict=True)
    report("S", len(sol) == 1,
           "Bessel limit: dG2/dz + (g^2-z^2)^2/(4 g^2 z) is a multiple of G2 (multiplier %s)"
           % (sol[0][lam] if sol else "none found"))


# --------------------------------------------------------------------------------------
def section_N1():
    print("\n== N1. The quadrature (96-node Gauss-Legendre panels in ln t) ==")
    tol = mp.mpf(10) ** -35
    cases = [(1, 1), (0.25, 0.3), (2, 5), (1000, 0.3), ("0.001", "1e-5"), (10 ** 6, 100),
             (10 ** 6, "1e-9"), ("0.001", "1e-80"), ("0.05", 10 ** 4), (30, "1e-40")]
    worst = 0
    for p, k in cases:
        d6 = moments(p, k, 6)
        d7 = moments(p, k, 7)
        err = max(abs(d6["a"] / d7["a"] - 1), abs(d6["V"] / d7["V"] - 1),
                  abs(d6["m"] / d7["m"] - 1))
        worst = max(worst, err)
    report("N1a", worst < tol, "96 against 192 nodes per panel, 10 parameter pairs incl. p=1e6 and "
           "p=0.001: largest relative difference in a, V, m = %s (tolerance 1e-35)" % mp.nstr(worst, 3))
    worst = 0
    for p, k in [(1, 1), (0.25, 0.3), (2, 5), (7.5, 0.02)]:
        p = mp.mpf(p)
        k = mp.mpf(k)
        d = moments(p, k)
        a_h = k * p * mp.hyperu(p + 1, 2, k) / mp.hyperu(p, 1, k)
        m_h = mp.diff(lambda b: mp.log(mp.hyperu(p, b, k)), 1)
        worst = max(worst, abs(d["a"] - a_h), abs(d["m"] - m_h))
    report("N1b", worst < mp.mpf(10) ** -25, "a and m against mpmath hyperu (m by numerical "
           "b-derivative), 4 pairs: largest absolute difference %s (tolerance 1e-25)" % mp.nstr(worst, 3))
    # mpmath adaptive quadrature in t, independent of the panel rule
    worst = 0
    for p, k in [(1.5, 0.7), (4, 3)]:
        p = mp.mpf(p)
        k = mp.mpf(k)
        w = lambda t: t ** (p - 1) * (1 + t) ** (-p) * mp.exp(-k * t)
        brk = [0, mp.mpf(1) / 10, 1, 5, 20, 80, mp.inf]
        Z = mp.quad(w, brk)
        EW = mp.quad(lambda t: w(t) * mp.log1p(t), brk) / Z
        ET = mp.quad(lambda t: w(t) * t, brk) / Z
        EWT = mp.quad(lambda t: w(t) * t * mp.log1p(t), brk) / Z
        d = moments(p, k)
        worst = max(worst, abs(d["m"] - EW), abs(d["a"] - k * ET),
                    abs(d["V"] - k * (EWT - EW * ET)))
    report("N1c", worst < mp.mpf(10) ** -25, "m, a, V against mpmath tanh-sinh quadrature in t, "
           "2 pairs: largest absolute difference %s (tolerance 1e-25)" % mp.nstr(worst, 3))


def section_N2():
    print("\n== N2. Two-copy representation 1 - 2V = E[1/(1+T+T')] ==")
    # (i) direct double sum over the product of two quadrature rules (30 digits, 24-node panels)
    mp.mp.dps = 30
    worst = 0
    for p, k in [(1, 1), (3, "0.2")]:
        d = moments(p, k, degree=4, want_nodes=True)
        nodes = d["nodes"]
        n = len(nodes)
        acc = mp.mpf(0)
        for i in range(n):
            ti, fi = nodes[i]
            acc += fi * fi / (1 + 2 * ti)
            row = mp.mpf(0)
            for j in range(i + 1, n):
                tj, fj = nodes[j]
                row += fj / (1 + ti + tj)
            acc += 2 * fi * row
        dref = moments(p, k, degree=6)
        err = abs(acc - (1 - 2 * dref["V"]))
        worst = max(worst, err)
        print("        p=%s kappa=%s: double sum %s ; 1-2V %s" % (p, k, mp.nstr(acc, 22),
                                                                 mp.nstr(1 - 2 * dref["V"], 22)), flush=True)
    report("N2a", worst < mp.mpf(10) ** -18, "double quadrature of 1/(1+t+t') against 1 - 2 kappa Cov(W,T), "
           "2 pairs, 30 digits: largest difference %s (tolerance 1e-18)" % mp.nstr(worst, 3))
    # (ii) int_0^inf e^-u (U(k+u)/U(k))^2 du with mpmath's Tricomi function
    worst = 0
    for p, k in [("0.5", "0.1"), (20, 2)]:
        p = mp.mpf(p)
        k = mp.mpf(k)
        U0 = mp.hyperu(p, 1, k)
        Rt = mp.quad(lambda u: mp.exp(-u) * (mp.hyperu(p, 1, k + u) / U0) ** 2,
                     [0, 1, 5, 20, 80, mp.inf])
        d = moments(p, k)
        worst = max(worst, abs(Rt - (1 - 2 * d["V"])))
    report("N2b", worst < mp.mpf(10) ** -22, "int e^-u (U(kappa+u)/U(kappa))^2 du against 1 - 2V, 2 pairs, "
           "30 digits: largest difference %s (tolerance 1e-22)" % mp.nstr(worst, 3))
    mp.mp.dps = DPS


def section_N3():
    print("\n== N3. Differential equations by central differences in x = ln kappa (h = 1e-12) ==")
    h = mp.mpf(10) ** -12
    tol = mp.mpf(10) ** -18
    cases = [(1, 1), ("0.25", "0.3"), (2, 5), (1000, "0.3"), ("0.01", "1e-6"), (10 ** 5, 10),
             (50, 400), ("0.3", 30)]
    worst = {}
    for p, k in cases:
        p = mp.mpf(p)
        k = mp.mpf(k)
        c = state(p, k)
        up = state(p, k * mp.exp(h))
        dn = state(p, k * mp.exp(-h))

        def D(key):
            return (up[key] - dn[key]) / (2 * h)

        a, V, R, e, Y, Xi, Th, Psi, rho = (c[x] for x in
                                             ("a", "V", "R", "e", "Y", "Xi", "Theta", "Psi", "rho"))
        res = {
            "a' = a^2 + k a - p k": (D("a") - (a * a + k * a - p * k), max(1, abs(a * a), k * a)),
            "V' = (k+2a)V - a": (D("V") - ((k + 2 * a) * V - a), max(1, k, a)),
            "m' = -V": (D("m") + V, 1),
            "Y' formula": (D("Y") - ((k * rho + 2 * Psi + R * R) * Y - (Psi * R * R - Th + Y * Y)),
                           max(1, k * rho)),
            "Xi' formula": (D("Xi") - ((k * rho + 2 * e) * Xi - (Psi ** 2 - Y ** 2)), max(1, k * rho)),
            "Theta' formula": (D("Theta") - ((1 + k * rho) * Th + Psi ** 2 + k * rho * Y * Y
                                             + 2 * Psi * R * R * e), max(1, k * rho) ** 2),
        }
        for name, (val, scale) in res.items():
            worst[name] = max(worst.get(name, 0), abs(val) / scale)
    for name, val in worst.items():
        report("N3", val < tol, "%s: largest scaled residual over 8 pairs %s (tolerance 1e-18)"
               % (name, mp.nstr(val, 3)))


def section_N4():
    print("\n== N4. Integration-by-parts identities ==")
    worst1 = worst2 = worst3 = 0
    for p, k in [(1, 1), ("0.25", "0.3"), (2, 5), (1000, "0.3"), ("0.01", "1e-6"), (10 ** 5, 10)]:
        d = moments(p, k)
        p_, k_ = d["p"], d["kappa"]
        # kappa^2 E T^2 = p kappa + a - kappa a
        r1 = k_ * k_ * d["ET2"] - (p_ * k_ + d["a"] - k_ * d["a"])
        worst1 = max(worst1, abs(r1) / max(1, p_ * k_))
        # a = p E[1/(1+T)]
        r2 = d["a"] - p_ * d["Eq"]
        worst2 = max(worst2, abs(r2) / max(1, d["a"]))
        # V = 1 - E[1/(1+T)] + p Cov(W, 1/(1+T))
        r3 = d["V"] - (1 - d["Eq"] + p_ * (d["EWq"] - d["m"] * d["Eq"]))
        worst3 = max(worst3, abs(r3))
    report("N4a", worst1 < mp.mpf(10) ** -35, "kappa^2 E T^2 = p kappa + a - kappa a: largest scaled "
           "residual %s (tolerance 1e-35)" % mp.nstr(worst1, 3))
    report("N4b", worst2 < mp.mpf(10) ** -35, "a = p E[1/(1+T)]: largest scaled residual %s "
           "(tolerance 1e-35)" % mp.nstr(worst2, 3))
    report("N4c", worst3 < mp.mpf(10) ** -30, "V = 1 - E[1/(1+T)] + p Cov(W, 1/(1+T)): largest "
           "residual %s (tolerance 1e-30)" % mp.nstr(worst3, 3))


def kappa_grid(p, n=41):
    p = mp.mpf(p)
    hmax = 14 + mp.sqrt(196 + 28 / p)          # reaches s of about 1e-6
    lo = mp.log(min(p, 1)) - hmax if p < 1 else mp.log(p) - 30
    hi = mp.log(p) + mp.log(10) * 6            # reaches 1 - s of about 1e-6
    return [mp.exp(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def section_N5():
    print("\n== N5. Sign scan (evidence only; 50 digits) ==")
    ps = ["0.001", "0.01", "0.05", "0.25", "0.5", 1, 2, 5, 20, 100, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
    n_pts = 0
    bad = 0
    worst = dict(e=None, Y=None, Th=None, ln=None, F=None)
    smin, smax = mp.mpf(1), mp.mpf(0)
    print("        p        | s range scanned        | min e      | min Y/Psi  | max Theta/Psi^2 | "
          "min ln(sig/s) | max F")
    for p in ps:
        row = dict(e=[], Y=[], Th=[], ln=[], F=[], s=[])
        for k in kappa_grid(p):
            c = state(p, k)
            n_pts += 1
            row["e"].append(c["e"])
            row["Y"].append(c["Y"] / c["Psi"])
            row["Th"].append(c["Theta"] / c["Psi"] ** 2)
            row["ln"].append(c["lnratio"])
            row["F"].append(c["F"])
            row["s"].append(c["s"])
            if not (c["e"] > 0 and c["Y"] > 0 and c["Theta"] < 0 and c["lnratio"] > 0
                    and c["F"] < 0 and 0 < c["R"] < 1):
                bad += 1
                print("        VIOLATION at p=%s kappa=%s" % (p, mp.nstr(k, 10)))
        smin = min(smin, min(row["s"]))
        smax = max(smax, max(row["s"]))
        print("        %-8s | %-9s .. 1-%-8s | %-10s | %-10s | %-15s | %-13s | %s" % (
            p, mp.nstr(min(row["s"]), 3), mp.nstr(1 - max(row["s"]), 3), mp.nstr(min(row["e"]), 3),
            mp.nstr(min(row["Y"]), 3), mp.nstr(max(row["Th"]), 3), mp.nstr(min(row["ln"]), 3),
            mp.nstr(max(row["F"]), 3)), flush=True)
    report("N5", bad == 0, "%d points (14 shapes x 41 kappa): e > 0, 0 < R < 1, Theta < 0, Y > 0, "
           "ln(sigma/s) > 0 and F < 0 at every point; violations: %d" % (n_pts, bad))


def section_N6():
    print("\n== N6. ln(sigma/s) = int_x^inf 2Y/(1+R) dy (30 digits) ==")
    mp.mp.dps = 30
    worst = 0
    for p, k in [(1, "0.5"), (50, 3)]:
        p = mp.mpf(p)
        k = mp.mpf(k)
        x0 = mp.log(k)

        def integrand(y):
            c = state(p, mp.exp(y))
            return 2 * c["Y"] / (1 + c["R"])

        # tail beyond kappa = 1e8 max(p,1): there Y < Psi < V <= p/kappa (proved bounds), so the
        # neglected part is below 2p/kappa_max = 2e-8; tolerance 1e-7 allows for it
        xmax = mp.log(mp.mpf(10) ** 8 * max(p, 1))
        brk = [x0 + (xmax - x0) * i / 12 for i in range(13)]
        val = mp.quad(integrand, brk, maxdegree=5)
        c = state(p, k)
        worst = max(worst, abs(val - c["lnratio"]))
        print("        p=%s kappa=%s: integral %s ; ln(sigma/s) %s" % (p, k, mp.nstr(val, 12),
                                                                       mp.nstr(c["lnratio"], 12)), flush=True)
    report("N6", worst < mp.mpf(10) ** -7, "2 pairs: largest difference %s (tolerance 1e-7; the integral "
           "is truncated at kappa = 1e8 max(p,1))" % mp.nstr(worst, 3))
    mp.mp.dps = DPS


def section_N7():
    print("\n== N7. By-products (not used in the proof) ==")
    h = mp.mpf(10) ** -12
    worst = 0
    worst2 = 0
    for p, k in [(1, 1), ("0.25", "0.3"), (2, 5), (1000, "0.3"), (10 ** 5, 10)]:
        p = mp.mpf(p)
        k = mp.mpf(k)
        c = state(p, k)
        up = moments(p * (1 + h), k)
        dn = moments(p * (1 - h), k)
        dadp = (up["a"] - dn["a"]) / (2 * h * p)
        worst = max(worst, abs(dadp - c["R"]))
        # stationarity: E ln T + E ln(1+T) = psi(p) - ln kappa
        worst2 = max(worst2, abs(c["ElnT"] + c["m"] - (mp.digamma(p) - mp.log(k))))
    report("N7a", worst < mp.mpf(10) ** -18, "R = d a/d p by central difference in p, 5 pairs: largest "
           "difference %s (tolerance 1e-18)" % mp.nstr(worst, 3))
    report("N7b", worst2 < mp.mpf(10) ** -35, "E ln T + m = psi(p) - ln kappa, i.e. m = -(ln kappa + "
           "d ln U/dp)/2: largest residual %s (tolerance 1e-35)" % mp.nstr(worst2, 3))
    mp.mp.dps = 30
    worst3 = 0
    for p in ["0.5", 1, "2.5"]:
        p = mp.mpf(p)
        val = mp.quad(lambda y: mp.exp(-y) * mp.hyperu(p, 1, y) ** 2,
                      [0, mp.mpf(1) / 100, 1, 5, 20, 80, mp.inf]) * mp.gamma(p) ** 2
        worst3 = max(worst3, abs(val - mp.polygamma(1, p)))
    report("N7c", worst3 < mp.mpf(10) ** -20, "Gamma(p)^2 int_0^inf U(p,1,y)^2 e^-y dy = trigamma(p), "
           "p = 0.5, 1, 2.5, 30 digits: largest difference %s (tolerance 1e-20)" % mp.nstr(worst3, 3))
    mp.mp.dps = DPS


def section_N8():
    print("\n== N8. Large-p boundary layer ==")
    # Bessel inequalities at sample points (they are proved in the note by the same barrier argument)
    bad = 0
    for j in range(-12, 13):
        z = mp.mpf(10) ** (mp.mpf(j) / 4)
        g = z * mp.besselk(1, z) / mp.besselk(0, z)
        if not (g * g > z * z + z and z * z * g + z * z / 2 + g * g / 2 - g ** 3 > 0):
            bad += 1
    report("N8a", bad == 0, "g = z K1(z)/K0(z): g^2 > z^2 + z and z^2 g + z^2/2 + g^2/2 - g^3 > 0 at 25 "
           "points z = 1e-3 .. 1e3; violations: %d" % bad)
    # convergence to the Bessel limit: p = 1e6, kappa = omega^2/p
    worst = 0
    for om in ["0.3", 1, 3]:
        om = mp.mpf(om)
        p = mp.mpf(10) ** 6
        c = state(p, om ** 2 / p)
        z = 2 * om
        g = z * mp.besselk(1, z) / mp.besselk(0, z)
        gp = (g * g - z * z) / z
        yb = z / (2 * gp) + mp.mpf(1) / 4 - g / 2
        worst = max(worst, abs(c["Y"] / yb - 1), abs(2 * c["a"] / g - 1), abs(2 * p * c["R"] / (om * gp) - 1))
        print("        omega=%s: Y %s  Bessel limit %s ; 2a/g %s ; 2pR/(omega g') %s" % (
            mp.nstr(om, 3), mp.nstr(c["Y"], 10), mp.nstr(yb, 10), mp.nstr(2 * c["a"] / g, 10),
            mp.nstr(2 * p * c["R"] / (om * gp), 10)), flush=True)
    report("N8b", worst < mp.mpf(10) ** -4, "p = 1e6, kappa = omega^2/p, omega = 0.3, 1, 3: Y, a, pR within "
           "%s (relative) of their Bessel limits (tolerance 1e-4)" % mp.nstr(worst, 3))
    # minimum slack over kappa at fixed p against 1/(16 p^1.5) at kappa = 1/4
    print("        p        | kappa at min | s at min    | min of -F   | 16 p^1.5 min(-F) | dD/dL there")
    ratios = []
    for p in [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]:
        p = mp.mpf(p)
        f = lambda x: state(p, mp.exp(x))["F"]     # negative; maximise
        lo, hi = mp.log(mp.mpf("0.01")), mp.log(mp.mpf(30))
        gr = (mp.sqrt(5) - 1) / 2
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)
        f1, f2 = f(x1), f(x2)
        for _ in range(34):
            if f1 > f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - gr * (hi - lo)
                f1 = f(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + gr * (hi - lo)
                f2 = f(x2)
        xm = (lo + hi) / 2
        c = state(p, mp.exp(xm))
        ratio = -c["F"] * 16 * p ** mp.mpf("1.5")
        ratios.append(ratio)
        dDdL = 1 / c["V"] - 2 - c["s"] / (1 - c["s"])
        print("        %-8s | %-12s | %-11s | %-11s | %-16s | %s" % (
            mp.nstr(p, 3), mp.nstr(mp.exp(xm), 6), mp.nstr(c["s"], 6), mp.nstr(-c["F"], 6),
            mp.nstr(ratio, 8), mp.nstr(dDdL, 8)), flush=True)
    report("N8c", abs(ratios[-1] - 1) < mp.mpf("0.01") and all(r > 0 for r in ratios),
           "16 p^1.5 min_kappa(-F) at p = 1e6 is %s (prediction 1, tolerance 1%%); slack positive at all "
           "five p" % mp.nstr(ratios[-1], 8))
    c = state(1000, mp.mpf("0.2463"))
    print("        referee's point: p=1000, s=%s: dD/dL = %s (referee: 3.86693391329822e-6 at s~0.0156016668)"
          % (mp.nstr(c["s"], 10), mp.nstr(1 / c["V"] - 2 - c["s"] / (1 - c["s"]), 12)))


def section_N9():
    print("\n== N9. Elementary bounds used for boundedness at large kappa ==")
    bad = 0
    n = 0
    for p in ["0.01", 1, 30, 1000]:
        for mult in [4, 10, 1000, 10 ** 6]:
            p_ = mp.mpf(p)
            k = mult * p_
            c = state(p_, k)
            n += 1
            ok = (0 < c["V"] <= c["a"] / k <= p_ / k and 0 < p_ - c["a"] <= p_ * c["a"] / k
                  and c["R"] >= 1 - 2 * p_ / k
                  and abs(k * c["rho"] * c["Xi"]) <= 2 * (5 * p_ ** 2 + p_ ** 2 / k)
                  and abs(c["Y"]) <= mp.mpf(1) / 4 + p_ and c["m"] <= p_ / k)
            if not ok:
                bad += 1
    report("N9", bad == 0, "V <= a/kappa <= p/kappa, 0 < p-a <= p a/kappa, R >= 1-2p/kappa, "
           "|kappa rho Xi| <= 10p^2+2p^2/kappa, |Y| <= 1/4+p, m <= p/kappa at %d points with "
           "kappa >= 4p; violations: %d" % (n, bad))


def main():
    print("gamma_monotonicity_independent.py  (mpmath %s, sympy %s, %d digits)" % (mp.__version__,
                                                                             sp.__version__, DPS))
    only = sys.argv[1:]
    sections = [("S", section_S), ("N1", section_N1), ("N2", section_N2), ("N3", section_N3),
                ("N4", section_N4), ("N5", section_N5), ("N6", section_N6), ("N7", section_N7),
                ("N8", section_N8), ("N9", section_N9)]
    for name, fn in sections:
        if only and name not in only:
            continue
        t = time.time()
        fn()
        print("        [section %s: %.1f s]" % (name, time.time() - t), flush=True)
    print("\nTotal time %.1f s" % (time.time() - T0))
    if FAILS:
        print("FAILED LINES: %d" % len(FAILS))
        for line in FAILS:
            print("  " + line)
        sys.exit(1)
    print("ALL CHECKS PASSED (numerical checks are evidence, not proof)")


if __name__ == "__main__":
    main()
