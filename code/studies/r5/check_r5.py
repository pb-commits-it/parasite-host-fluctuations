"""Reproducible checks; all output and matplotlib cache stay in this directory.

Declared tolerances (before any validation run):
* Matched eigenvalue absolute error / max(1, matrix infinity norm): 2e-8.
* Characteristic residual / sum of absolute term sizes: 2e-7.
  If double-precision eigenvalue rounding makes this evaluation ill-conditioned,
  refine the scalar root at 70 decimal digits; require its correction <2e-8
  on the matrix scale, then apply the same residual tolerance. This refinement
  was added after the initial run exposed cancellation at a near-diagonal root.
* Central finite-difference Jacobian error on the same scale: 3e-7 (step 1e-6).
* Algebraic identities, threshold/optimization inequalities: 2e-9 scale-relative.
* ODE DOP853 rtol=2e-10, atol=2e-12; nonlinear fitted rates: max(2%, 1e-3).
* Nonlinear fitted angular frequencies: max(3%, 2e-3).
* Stable run final max deviation: <1e-6; unstable initial growth is checked.
* Sustained-cycle evidence: late amplitude >1e-3, amplitude ratio within 5%,
  at least 8 late peaks, peak-spacing CV <2% (ODE), or period-2 residual <1e-8.
Finite simulation does not prove a periodic attractor or a Hopf bifurcation.
"""
from __future__ import annotations

import os
from pathlib import Path
import platform
import numpy as np
import scipy
import mpmath as mp
from scipy.integrate import solve_ivp, quad
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
import r5

ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
NS = [1, 2, 3, 4, 8, 16, 32]
KAPPAS = [0., .01, .1, .5, .9, .99, 1., 1.01, 1.5, 2., 2.2, 3., 5., 9., 10., 20.]
TOL_EIG, TOL_CHAR, TOL_FD = 2e-8, 2e-7, 3e-7
COUNTS, WORST = {}, {}
LINES = []


def report(s=""):
    print(s, flush=True)
    LINES.append(s)


def record(name, err, tol):
    err = float(err)
    COUNTS[name] = COUNTS.get(name, 0)+1
    WORST[name] = max(WORST.get(name, 0.), float(err))
    if not np.isfinite(err) or err > tol:
        raise AssertionError(f"{name}: error {err:.6g} > {tol:.6g}")


def match(J, explicit, group):
    eig = np.linalg.eigvals(J)
    cost = abs(eig[:, None]-explicit[None, :])
    rows, cols = linear_sum_assignment(cost)
    scale = max(1., np.linalg.norm(J, np.inf))
    record(group, np.max(cost[rows, cols])/scale, TOL_EIG)
    return eig


def residual(J, terms, group):
    for eig in np.linalg.eigvals(J):
        pieces = terms(eig)
        err = abs(pieces.sum())/max(1., abs(pieces).sum())
        if err > TOL_CHAR:
            with mp.workdps(70):
                root = mp.findroot(lambda z: sum(terms(z)),
                                   (mp.mpc(eig), mp.mpc(eig)+mp.mpf('1e-10')),
                                   tol=mp.mpf('1e-55'), maxsteps=100)
                correction = abs(complex(root)-eig)/max(1., np.linalg.norm(J, np.inf))
                record("characteristic root high-precision correction", correction, TOL_EIG)
                pieces = terms(root)
                err = abs(sum(pieces))/max(mp.mpf(1), sum(abs(pieces)))
        record(group, err, TOL_CHAR)


def algebra(actual, expected, name="algebra", tol=2e-9):
    record(name, abs(actual-expected)/max(1., abs(expected)), tol)


def fdcheck(fun, J):
    n = len(J)
    eye = np.eye(n)*1e-6
    numerical = np.column_stack([(fun(np.ones(n)+e)-fun(np.ones(n)-e))/2e-6 for e in eye])
    record("nonlinear Jacobian finite difference",
           np.max(abs(numerical-J))/max(1., np.linalg.norm(J, np.inf)), TOL_FD)


def check_equal():
    report("Equal-stage grids: n="+str(NS)+"; kappa="+str(KAPPAS))
    for n in NS:
        for tau in [12., 2.*n]:
            a = np.full(n, n/tau)
            for k in KAPPAS:
                J = r5.jacobian_C(a, k)
                eig = match(J, r5.char_roots_C(n, tau, k), "C(i) spectra")
                algebra(max(eig.real), r5.dominant_C(n, tau, k).real, "C(i) dominant")
            for form in ["bh", "ricker"]:
                for L in [.1, 1., 2., 4., 8., 16., 32., 64., .5*n, n, 1.5*n, 2*n, 3*n]:
                    R0 = np.exp(L)
                    kap = r5.local_elasticity(R0, n, form)
                    J = r5.jacobian_C(a, all_kappas=np.full(n, kap))
                    eig = match(J, r5.every_roots_C(n, tau, R0, form), "C(iii) spectra")
                    algebra(-max(eig.real), r5.return_rate_every_stage(n, tau, R0, form), "C(iii) return rate")
        for s in [0., .2, .5, .9, 1-n/64.]:
            ss = np.full(n, s)
            for k in KAPPAS:
                J = r5.jacobian_D(ss, k)
                eig = match(J, r5.char_roots_D(n, s, k), "D(i) spectra")
                algebra(max(abs(eig)), r5.radius_D(n, s, k), "D(i) radius")
            for form in ["bh", "ricker"]:
                for L in [.1, 1., 2., 4., 8., 16., 32., 64., .5*n, n, 1.5*n, 2*n, 3*n]:
                    R0 = np.exp(L)
                    kap = r5.local_elasticity(R0, n, form)
                    J = r5.jacobian_D(ss, all_kappas=np.full(n, kap))
                    eig = match(J, r5.every_roots_D(n, s, R0, form), "D(iii) spectra")
                    algebra(max(abs(eig)), r5.radius_every_D(n, s, R0, form), "D(iii) radius")
        kc = r5.critical_equal_C(n)
        if np.isfinite(kc):
            for factor in [.999, 1., 1.001]:
                k = 1+(kc-1)*factor
                eig = np.linalg.eigvals(r5.jacobian_C(np.ones(n), k))
                algebra(max(eig.real), r5.dominant_C(n, n, k).real, "C(i) boundary")
                if factor != 1:
                    assert (max(eig.real) < 0) == (factor < 1)
        lc = r5.critical_logR_every_C(n)
        if np.isfinite(lc):
            for factor in [.999, 1., 1.001]:
                eig = np.linalg.eigvals(r5.jacobian_C(np.ones(n), all_kappas=np.full(n, factor*lc/n)))
                algebra(max(eig.real), -r5.return_rate_every_stage(n, n, np.exp(factor*lc), "ricker"), "C(iii) boundary")
                if factor != 1:
                    assert (max(eig.real) < 0) == (factor < 1)
        for s in [0., .2, .5, .9]:
            kc = r5.critical_equal_D(n, s)
            lc = r5.critical_logR_every_D(n, s)
            for factor in [.999, 1., 1.001]:
                eig = np.linalg.eigvals(r5.jacobian_D(np.full(n, s), 1+(kc-1)*factor))
                algebra(max(abs(eig)), r5.radius_D(n, s, 1+(kc-1)*factor), "D(i) boundary")
                eig2 = np.linalg.eigvals(r5.jacobian_D(np.full(n, s), all_kappas=np.full(n, factor*lc/n)))
                algebra(max(abs(eig2)), r5.radius_every_D(n, s, np.exp(factor*lc), "ricker"), "D(iii) boundary")
                if factor != 1:
                    assert (max(abs(eig)) < 1) == (factor < 1)
                    assert (max(abs(eig2)) < 1) == (factor < 1)


def check_general():
    rng = np.random.default_rng(20261114)
    for n in NS:
        for rep in range(12):
            times = rng.uniform(.2, 2., n)
            times *= 12/times.sum()
            a = 1/times
            td = 1+32*rng.dirichlet(np.ones(n))
            s = 1-1/td
            k = [.2, .8, 1., 2., 10., 20.][rep % 6]
            for sensor in sorted(set([0, n//2, n-1])):
                J = r5.jacobian_C(a, k, sensor)
                residual(J, lambda z: r5.characteristic_terms_C(z, a, k, sensor), "C(i,ii) characteristic roots")
                JD = r5.jacobian_D(s, k, sensor)
                residual(JD, lambda z: r5.characteristic_terms_D(z, s, k, sensor), "D(i,ii) characteristic roots")
                if rep == 0:
                    fdcheck(lambda y: r5.nonlinear_C(0, y, a, k, sensor=sensor), J)
                    fdcheck(lambda y: r5.nonlinear_D(y, s, k, sensor=sensor), JD)
            ks = rng.uniform(.05, 2.5, n)
            residual(r5.jacobian_C(a, all_kappas=ks), lambda z: r5.characteristic_terms_C(z, a, all_kappas=ks), "C(iii) unequal characteristic roots")
            residual(r5.jacobian_D(s, all_kappas=ks), lambda z: r5.characteristic_terms_D(z, s, all_kappas=ks), "D(iii) unequal characteristic roots")
            if rep == 0:
                for form, kap in [("bh", .4), ("ricker", 1.7)]:
                    fdcheck(lambda y: r5.nonlinear_C(0, y, a, kap, form, every=True), r5.jacobian_C(a, all_kappas=np.full(n, kap)))
                    fdcheck(lambda y: r5.nonlinear_D(y, s, kap, form, every=True), r5.jacobian_D(s, all_kappas=np.full(n, kap)))
            if n >= 3:
                kc = r5.critical_kappa_C(times)
                assert kc >= r5.critical_equal_C(n)-2e-9
                for f in [.999, 1., 1.001]:
                    alpha = max(np.linalg.eigvals(r5.jacobian_C(a, 1+(kc-1)*f)).real)
                    if f == 1:
                        record("C unequal crossing", abs(alpha)/max(a), 2e-8)
                    else:
                        assert (alpha < 0) == (f < 1)
            kc = r5.critical_kappa_D(td)
            for f in [.999, 1., 1.001]:
                rad = max(abs(np.linalg.eigvals(r5.jacobian_D(s, 1+(kc-1)*f))))
                if f == 1:
                    algebra(rad, 1., "D unequal crossing", 2e-8)
                else:
                    assert (rad < 1) == (f < 1)
            for kap in [.2, .8]:
                rate = r5.undercomp_rate_C(times, kap)
                eig = np.linalg.eigvals(r5.jacobian_C(a, kap))
                algebra(rate, -max(eig.real), "C unequal undercomp rate")
                lower = kap/12
                upper = n/12*(1-(1-kap)**(1/n))
                assert lower-2e-9 <= rate <= upper+2e-9
                record("C unequal rate bounds", max(lower-rate, rate-upper, 0), 2e-9)
                rateD = r5.undercomp_rate_C(td, kap)
                eigD = np.linalg.eigvals(r5.jacobian_D(s, kap))
                algebra(1-rateD, max(abs(eigD)), "D unequal undercomp radius")
                equal = r5.undercomp_rate_C(np.full(n, td.mean()), kap)
                extreme = r5.undercomp_rate_C(np.array([td.sum()-n+1]+[1.]*(n-1)), kap)
                assert extreme-2e-9 <= rateD <= equal+2e-9
                record("D unequal rate bounds", max(extreme-rateD, rateD-equal, 0), 2e-9)
    # Explicit concentration sequences, holding the C cycle mean fixed.
    for n in [3, 4, 8, 16, 32]:
        report(f"C duration concentration n={n}, tau=12: epsilon, return(k=.5), critical_kappa")
        for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
            t = np.array([12-(n-1)*eps]+[eps]*(n-1)) if (n-1)*eps < 12 else None
            if t is not None:
                report(f"  {eps:.1e} {r5.undercomp_rate_C(t, .5):.10g} {r5.critical_kappa_C(t):.10g}")


def check_identities():
    # Equilibrium elasticities, product identities, transforms, and time scalings.
    rng = np.random.default_rng(36)
    for L in [.1, .5, 1., 2., 4., 8.]:
        R0 = np.exp(L)
        for form in ["bh", "ricker"]:
            x = np.expm1(L) if form == "bh" else L
            f = (lambda x: 1/(1+x)) if form == "bh" else (lambda x: np.exp(-x))
            algebra(f(x), 1/R0, "equilibrium and elasticity")
            eps = 1e-5
            kap = -(np.log(f(x*np.exp(eps)))-np.log(f(x*np.exp(-eps))))/(2*eps)
            algebra(kap, r5.local_elasticity(R0, 1, form), "equilibrium and elasticity")
        for n in NS:
            ks = L*rng.dirichlet(np.ones(n))
            algebra(np.prod(np.exp(-ks)), 1/R0, "every-stage product identities")
            algebra(np.sum(ks), L, "every-stage product identities")
            for k in [.1, .8, 1.5]:
                algebra(r5.dominant_C(n, 12, k).real*12/n,
                        r5.dominant_C(n, 2*n, k).real*2, "time rescaling")
            for form in ["bh", "ricker"]:
                algebra(r5.return_rate_every_stage(n, 12, R0, form)*12/n,
                        r5.return_rate_every_stage(n, 2*n, R0, form)*2, "time rescaling")
    for n in NS:
        a = rng.uniform(.3, 2, n)
        lam = .3+.4j
        vals = []
        for rate in a:
            real = quad(lambda t: rate*np.exp(-(rate+lam.real)*t)*np.cos(lam.imag*t), 0, np.inf, epsabs=1e-12)[0]
            imag = quad(lambda t: -rate*np.exp(-(rate+lam.real)*t)*np.sin(lam.imag*t), 0, np.inf, epsabs=1e-12)[0]
            vals.append(real+1j*imag)
        algebra(np.prod(vals), np.prod(a/(a+lam)), "C transform integration", 2e-10)
        s = rng.uniform(0, .9, n)
        z = 1.1+.3j
        counts = np.arange(1, 2001)
        vals = [np.sum((1-st)*st**(counts-1)*z**(-counts)) for st in s]
        algebra(np.prod(vals), np.prod((1-s)/(z-s)), "D transform summation", 2e-10)
        algebra(np.sum(1/a), sum(quad(lambda t: t*rate*np.exp(-rate*t), 0, np.inf)[0] for rate in a), "C cycle mean")
        algebra(np.sum(1/(1-s)), sum(np.sum(counts*(1-st)*st**(counts-1)) for st in s), "D cycle mean")
    # Limits and ordering are quantified rather than tested as exact finite-n equalities.
    report("C fixed-cycle limits, tau=12, kappa=.5: n, gamma, limit=-log(.5)/12")
    for n in [32, 128, 1024, 100000]:
        report(f"  {n} {-r5.dominant_C(n, 12, .5).real:.12g} {-np.log(.5)/12:.12g}")
    report("Large-n instability thresholds C and D(s=.5), both tending to 2:")
    for n in [32, 128, 1024, 100000]:
        report(f"  {n} {r5.critical_equal_C(n):.12g} {r5.critical_equal_D(n, .5):.12g}")
    # Placement example, same n, same kappa, same equilibrium and durations.
    for j in [0, 1, 2]:
        ev = np.linalg.eigvals(r5.jacobian_C(np.ones(3), 10, sensor=j))
        report(f"Sensing example C n=3 kappa=10 j={j+1}: alpha={max(ev.real):.10g}")
        evd = np.linalg.eigvals(r5.jacobian_D(np.full(3, .5), 3.2, sensor=j))
        report(f"Sensing example D n=3 s=.5 kappa=3.2 j={j+1}: radius={max(abs(evd)):.10g}")


def check_discrete_arrangement_conjecture():
    """Evidence only: no proof of the D overcompensation duration extrema."""
    rng = np.random.default_rng(19513)
    total = 0
    for n in [3, 4, 8, 16, 32]:
        for h in [1.05, 2., 10.]:
            tau = n*h
            equal = r5.critical_kappa_D(np.full(n, h))
            concentrated = r5.critical_kappa_D(np.array([tau-n+1]+[1.]*(n-1)))
            values = []
            for concentration in [.1, 1., 10.]:
                for _ in range(40):
                    times = 1+(tau-n)*rng.dirichlet(np.full(n, concentration))
                    kc = r5.critical_kappa_D(times)
                    values.append(kc)
                    violation = max(equal-kc, kc-concentrated, 0)
                    record("D arrangement conjecture sampled bounds", violation, 2e-9)
                    total += 1
            report(f"D arrangement CONJECTURE n={n}, tau={tau:g}, 120 samples: equal={equal:.10g}, sampled=[{min(values):.10g}, {max(values):.10g}], concentrated={concentrated:.10g}")
    for tau in [2., 3., 10., 64.]:
        for frac in [.0, .1, .5, .9, 1.]:
            t = np.array([1+frac*(tau-2), 1+(1-frac)*(tau-2)])
            algebra(r5.critical_kappa_D(t), tau, "D n=2 arrangement identity")
    report(f"D arrangement evidence total: {total} random duration vectors; this is not a proof.")


def check_original_coordinates():
    """Independent raw abundance equations, with nonzero mortality in C."""
    rng = np.random.default_rng(44071)
    for n in NS:
        xstar = rng.uniform(.5, 3., n)
        a = rng.uniform(.4, 2., n)
        nu = a*rng.uniform(.3, .9, n)
        mu = a-nu
        s = rng.uniform(.1, .8, n)
        for every in [False, True]:
            sensors = [None] if every else sorted(set([0, n-1]))
            for sensor in sensors:
                for form in ["bh", "ricker"]:
                    ks = rng.uniform(.1, .8, n) if form == "bh" else rng.uniform(.2, 2.2, n)
                    if every:
                        K = xstar*(1-ks)/ks if form == "bh" else xstar/ks
                        def raw_f(x):
                            return 1/(1+x/K) if form == "bh" else np.exp(-x/K)
                        jacargs = dict(all_kappas=ks)
                    else:
                        kap = .6 if form == "bh" else 1.7
                        K = xstar[sensor]*(1-kap)/kap if form == "bh" else xstar[sensor]/kap
                        def raw_f(x):
                            fs = np.ones(n)
                            fs[-1] = 1/(1+x[sensor]/K) if form == "bh" else np.exp(-x[sensor]/K)
                            return fs
                        jacargs = dict(kappa=kap, sensor=sensor)
                    fstar = raw_f(xstar)
                    for mode in ["C", "D"]:
                        loss = nu+mu if mode == "C" else 1-s
                        transfer = np.roll(loss*xstar, -1)/xstar/fstar
                        b = transfer/nu
                        unregulated = b*nu if mode == "C" else transfer
                        logR = np.sum(np.log(unregulated/loss))
                        algebra(logR, -np.log(fstar).sum(), "raw-coordinate R0 and balance")
                        def fun(x):
                            incoming = np.roll(unregulated*x*raw_f(x), 1)
                            return incoming-loss*x if mode == "C" else s*x+incoming
                        equilibrium_error = max(abs(fun(xstar)-(0 if mode == "C" else xstar)))
                        record("raw-coordinate equilibrium", equilibrium_error, 2e-9)
                        J = r5.jacobian_C(loss, **jacargs) if mode == "C" else r5.jacobian_D(s, **jacargs)
                        eye = np.eye(n)*1e-6
                        Jrel = np.column_stack([(fun(xstar*(1+e))-fun(xstar*(1-e)))/(2e-6*xstar) for e in eye])
                        error = np.max(abs(Jrel-J))/max(1., np.linalg.norm(J, np.inf))
                        record("raw-coordinate relative Jacobian", error, TOL_FD)


def nonlinear_checks():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = [
        dict(label="C i: monotone return", mode="C", n=1, k=.5, form="bh"),
        dict(label="C i: damped oscillation", mode="C", n=3, k=2.),
        dict(label="C i: near-boundary damping", mode="C", n=8, k=2.),
        dict(label="C i: sustained-cycle evidence", mode="C", n=8, k=3.2, sustained=True),
        dict(label="C ii: sensor at receiving stage", mode="C", n=3, k=10., sensor=0),
        dict(label="C iii BH: return", mode="C", n=4, k=.5, form="bh", every=True),
        dict(label="C iii Ricker odd: damping", mode="C", n=3, k=2., every=True),
        dict(label="C iii Ricker even: return", mode="C", n=4, k=1.5, every=True),
        dict(label="D i: damped oscillation", mode="D", n=4, k=1.5, s=.5),
        dict(label="D i: period two", mode="D", n=1, k=2.2, s=0., sustained=True),
        dict(label="D ii: receiving-stage sensor", mode="D", n=3, k=.5, s=.5, sensor=0),
        dict(label="D iii Ricker odd: damping", mode="D", n=3, k=2., s=.5, every=True),
        dict(label="D iii BH: return", mode="D", n=8, k=.3, s=.5, form="bh", every=True),
    ]
    fig, axes = plt.subplots(5, 3, figsize=(15, 18), constrained_layout=True)
    data = {}
    for i, case in enumerate(cases):
        n, kap, mode = case["n"], case["k"], case["mode"]
        form, sensor, every = case.get("form", "ricker"), case.get("sensor"), case.get("every", False)
        sustained = case.get("sustained", False)
        vec = np.full(n, case.get("s", 1.))
        kwargs = dict(sensor=sensor, all_kappas=np.full(n, kap) if every else None)
        J = r5.jacobian_C(vec, kap, **kwargs) if mode == "C" else r5.jacobian_D(vec, kap, **kwargs)
        ev, V = np.linalg.eig(J)
        metric = ev.real if mode == "C" else abs(ev)
        candidates = np.flatnonzero(metric >= metric.max()-1e-10)
        ix = candidates[np.argmax(ev[candidates].imag)]
        eigenvalue = ev[ix]
        projection = np.linalg.inv(V)[ix]
        v = V[:, ix].real
        v /= max(abs(v))
        epsilon = 1e-4 if not sustained else 1e-3
        y0 = np.ones(n)+epsilon*v
        predicted = eigenvalue.real if mode == "C" else np.log(abs(eigenvalue))
        predicted_freq = abs(eigenvalue.imag) if mode == "C" else abs(np.angle(eigenvalue))
        if mode == "C":
            horizon = 1600. if sustained else max(40., 18/abs(predicted))
            ts = np.linspace(0, horizon, 16001 if sustained else 2501)
            sol = solve_ivp(lambda t, y: r5.nonlinear_C(t, y, vec, kap, form, sensor, every),
                            (0, horizon), y0, t_eval=ts, method="DOP853", rtol=2e-10, atol=2e-12)
            assert sol.success
            ys = sol.y.T
        else:
            steps = 2000 if sustained else max(80, int(np.ceil(18/abs(predicted))))
            ts = np.arange(steps+1)
            ys = np.empty((steps+1, n))
            ys[0] = y0
            for t in range(steps):
                ys[t+1] = r5.nonlinear_D(ys[t], vec, kap, form, sensor, every)
        assert np.min(ys) > 0
        amplitude = (ys-1)@projection
        magnitude = abs(amplitude)
        use = (magnitude > (epsilon*.8 if sustained else 1e-8)) & (magnitude < (epsilon*3 if sustained else epsilon*.8))
        if sustained:
            # Only the initial small-amplitude departure, before nonlinear saturation.
            first = np.flatnonzero(magnitude > epsilon*3)
            if len(first):
                use[first[0]:] = False
        assert use.sum() >= 4, (case, use.sum())
        fit = np.polyfit(ts[use], np.log(magnitude[use]), 1)[0]
        record("nonlinear fitted growth/decay", abs(fit-predicted), max(.001, .02*abs(predicted)))
        frequency = np.nan
        if predicted_freq > 1e-8 and predicted_freq < np.pi-1e-8:
            phase = np.unwrap(np.angle(amplitude))
            frequency = abs(np.polyfit(ts[use], phase[use], 1)[0])
            record("nonlinear fitted frequency", abs(frequency-predicted_freq), max(.002, .03*predicted_freq))
        final = max(abs(ys[-1]-1))
        if not sustained:
            record("nonlinear return final error", final, 1e-6)
        report(f"NONLINEAR {i+1}: {case['label']}; n={n}, kappa={kap}, form={form}, every={every}, sensor={sensor}")
        report(f"  dominant={eigenvalue}; growth predicted={predicted:.10g}, fitted={fit:.10g}; frequency predicted={predicted_freq:.10g}, fitted={frequency:.10g}; final max deviation={final:.6g}")
        if sustained:
            q = len(ts)//4
            amp1, amp2 = np.ptp(ys[-2*q:-q, 0]), np.ptp(ys[-q:, 0])
            assert amp2 > 1e-3 and abs(amp2/amp1-1) < .05
            if mode == "C":
                peaks, _ = find_peaks(ys[-q:, 0], prominence=.001)
                periods = np.diff(ts[-q:][peaks])
                cv = np.std(periods)/np.mean(periods)
                assert len(peaks) >= 8 and cv < .02
                report(f"  late peak-to-peak={amp2:.9g}, quarter-window amplitude ratio={amp2/amp1:.9g}, peaks={len(peaks)}, period={np.mean(periods):.9g}, period CV={cv:.6g}")
            else:
                period2 = np.max(abs(ys[-100:]-ys[-102:-2]))
                assert period2 < 1e-8
                report(f"  late peak-to-peak={amp2:.9g}, amplitude ratio={amp2/amp1:.9g}, period-two residual={period2:.6g}")
            COUNTS["nonlinear sustained-cycle evidence"] = COUNTS.get("nonlinear sustained-cycle evidence", 0)+1
        ax = axes.flat[i]
        ax.plot(ts, ys[:, 0]-1, lw=.9)
        ax.axhline(0, color="black", lw=.4)
        ax.set_title(case["label"], fontsize=10)
        ax.set_xlabel("time" if mode == "C" else "updates")
        ax.set_ylabel("stage 1 / equilibrium - 1")
        data[f"case{i+1}_t"] = ts
        data[f"case{i+1}_y"] = ys
    for ax, case_id, window, title in [
        (axes.flat[13], 4, 100., "C sustained case: last 100 time units"),
        (axes.flat[14], 10, 24., "D period two: last 24 updates"),
    ]:
        ts, ys = data[f"case{case_id}_t"], data[f"case{case_id}_y"]
        use = ts >= ts[-1]-window
        ax.plot(ts[use], ys[use, 0]-1, ".-" if case_id == 10 else "-", lw=1, markersize=4)
        ax.axhline(0, color="black", lw=.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("stage 1 / equilibrium - 1")
        ax.set_xlabel("updates" if case_id == 10 else "time")
    fig.suptitle("Nonlinear ring models: 13 direct simulations (equilibrium abundance = 1)", fontsize=14)
    fig.savefig(ROOT / "nonlinear_r5.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(ROOT / "nonlinear_r5.npz", **data)


def main():
    report("R5 INDEPENDENT NUMERICAL VALIDATION")
    report(f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}")
    report(__doc__)
    report("Validation note: an initial double-precision characteristic evaluation failed (0.04425 residual) at n=32 near an isolated diagonal root. High-precision scalar refinement now checks such roots without relaxing the declared eigenvalue or residual tolerances.")
    check_equal()
    check_general()
    check_identities()
    check_discrete_arrangement_conjecture()
    check_original_coordinates()
    nonlinear_checks()
    report("\nCHECK COUNTS AND MAXIMUM ERRORS (each spectrum case compares all n roots):")
    for name in COUNTS:
        report(f"{name}: {COUNTS[name]} cases; max error={WORST.get(name, 0.):.12g}")
    total = sum(COUNTS[k] for k in ["C(i) spectra", "C(iii) spectra", "D(i) spectra", "D(iii) spectra"])
    report(f"TOTAL EQUAL-STAGE JACOBIAN SPECTRUM CASES: {total}")
    report("PASS: all declared checks passed. Sustained oscillations are finite-time numerical evidence, not a global theorem.")
    (ROOT / "results_r5.txt").write_text("\n".join(LINES)+"\n")


if __name__ == "__main__":
    main()
