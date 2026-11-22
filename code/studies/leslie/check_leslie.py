"""Independent matrix and Monte Carlo checks; run: python3 check_leslie.py.

All generated files, including the temporary C simulation library, stay in
this script's directory. No network access or existing solution is used.
"""

import ctypes
from decimal import Decimal, localcontext
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from scipy.linalg import eig, solve

from leslie import (
    _coefficient_from_q,
    cost_common_driver,
    cost_independent_drivers,
    cost_persistent_limit,
    cost_semelparous,
    demographic_distribution,
)

BASE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
MATRIX_TOL = 2e-9  # Absolute + relative: tol * (1 + |reference|).
IDENTITY_TOL = 2e-11
DERIVATIVE_TOL = 2e-7
N_TABLES = 1500
SIGMA = 0.04  # Also run 0.02 with the same environmental path.
REPLICATES = 40
STEPS = 200_000
BURN = 10_000
RAW_ALLOWANCE = 0.002
EXTRAPOLATED_ALLOWANCE = 0.0002
SE_TARGET = 0.003


def leslie_matrix(m, p):
    L = np.zeros((len(m), len(m)))
    L[0] = m
    for i, survival in enumerate(p):
        L[i + 1, i] = survival
    return L


def matrix_coefficient(m, p, phi, independent=False):
    """Evaluate the general perturbation formula, using numerical eigenvectors.

    No Euler--Lotka distribution or Leslie resolvent simplification is used.
    At phi=1 this evaluates the Abel limit, not a stationary AR(1) model.
    """
    L = leslie_matrix(m, p)
    values, left, right = eig(L, left=True, right=True)
    index = np.argmax(values.real)
    lam = float(values[index].real)
    w = right[:, index].real
    w /= w.sum()
    v = left[:, index].real
    v /= v @ w
    A = L / lam
    Q = A - np.outer(w, v)
    M = np.eye(len(m)) - phi * Q
    drivers = []
    if independent:
        for i in range(len(m)):
            B = np.zeros_like(L)
            B[0, i] = m[i] / lam
            drivers.append(B)
    else:
        B = np.zeros_like(L)
        B[0] = np.asarray(m) / lam
        drivers.append(B)
    result = 0.0
    for B in drivers:
        b = v @ B @ w
        forcing = B @ w - b * w
        response = solve(M, forcing)
        result += b * b / 2 - phi * (v @ B @ response)
    return float(result)


def life_table(q, p=None, lam=1.0):
    q = np.asarray(q, dtype=float)
    if p is None:
        p = np.ones(len(q) - 1)
    p = np.asarray(p, dtype=float)
    l = np.r_[1.0, np.cumprod(p)]
    m = q * lam ** np.arange(1, len(q) + 1) / l
    return m, p


def displayed_closed_form(q, phi, independent=False):
    """Evaluate the displayed rational formula in high precision.

    This separately checks the pole-free production implementation, including
    phi=1-1e-8 where ordinary double-precision pole cancellation is unreliable.
    """
    with localcontext() as context:
        context.prec = 65
        weights = [Decimal(str(float(x))) for x in q]
        total = sum(weights)
        weights = [x/total for x in weights]
        s = Decimal(str(float(phi)))
        T = sum(Decimal(i+1)*x for i, x in enumerate(weights))
        G = sum(x*s**(i+1) for i, x in enumerate(weights))
        S2 = sum(x*x for x in weights) if independent else Decimal(1)
        R = sum(x*x*s**(i+1) for i, x in enumerate(weights)) if independent else G
        return float(S2*(1+s)/(2*T*T*(1-s)) - R/(T*(1-G)))


def random_checks(log):
    rng = np.random.default_rng(20261114)
    max_error = np.zeros(2)
    max_scaled_error = np.zeros(2)
    max_limit_error = np.zeros(2)
    max_identity_error = 0.0
    max_derivative_error = 0.0
    max_displayed_formula_error = 0.0
    comparison_count = 0
    sign_count = 0
    positive_cost_changes = negative_cost_changes = 0
    for case in range(N_TABLES):
        d = int(rng.integers(1, 19))
        q = rng.dirichlet(np.full(d, rng.uniform(0.35, 2.5)))
        if d > 2 and case % 3 == 1:
            q[rng.random(d) < 0.45] = 0
            q[0] = max(q[0], 0.03)
            q[-1] = max(q[-1], 0.03)
        elif d > 2 and case % 3 == 2:
            q[0] = 0
            # Last two reproductive ages are consecutive: gcd = 1.
            q[-2:] = np.maximum(q[-2:], 0.03)
        q /= q.sum()
        p = rng.uniform(0.70, 1.0, d - 1)
        lam_target = rng.uniform(0.8, 1.2)
        m, p = life_table(q, p, lam_target)
        lam, recovered = demographic_distribution(m, p)
        assert abs(lam - lam_target) <= IDENTITY_TOL
        assert np.max(abs(q - recovered)) <= IDENTITY_TOL
        ages = np.arange(1, d + 1)
        T = ages @ recovered
        V = (ages - T) ** 2 @ recovered
        S2 = recovered @ recovered
        U = ages @ (recovered * recovered)
        phis = [-0.85, 0.0, 0.5, 0.9, float(rng.uniform(-0.97, 0.97)), 1 - 1e-8]
        for phi in phis:
            for mode, fn in enumerate((cost_common_driver, cost_independent_drivers)):
                actual = fn(m, p, phi)
                displayed = displayed_closed_form(recovered, phi, bool(mode))
                displayed_error = abs(actual - displayed)
                max_displayed_formula_error = max(max_displayed_formula_error, displayed_error)
                assert displayed_error < IDENTITY_TOL * (1 + abs(displayed))
                reference = matrix_coefficient(m, p, phi, independent=bool(mode))
                error = abs(actual - reference)
                max_error[mode] = max(max_error[mode], error)
                max_scaled_error[mode] = max(max_scaled_error[mode], error / (1 + abs(reference)))
                assert error <= MATRIX_TOL * (1 + abs(reference)), (case, phi, mode, actual, reference)
                comparison_count += 1
            if 0 < phi < 0.99:
                G = recovered @ phi**ages
                threshold = phi / (T - (T - 1) * phi)
                delta = cost_common_driver(m, p, phi) - 1 / (2 * T * T)
                # Ignore equality within floating-point resolution.
                if abs(threshold - G) > 1e-12 and abs(delta) > 1e-12:
                    assert np.sign(delta) == np.sign(threshold - G)
                    sign_count += 1
                    positive_cost_changes += int(delta > 0)
                    negative_cost_changes += int(delta < 0)
        for mode, fn in enumerate((cost_common_driver, cost_independent_drivers)):
            factor = S2 if mode else 1.0
            reference0 = factor / (2 * T * T)
            error0 = abs(fn(m, p, 0) - reference0)
            max_identity_error = max(max_identity_error, error0)
            assert error0 < IDENTITY_TOL
            limit = cost_persistent_limit(m, p, independent=bool(mode))
            reference_limit = matrix_coefficient(m, p, 1.0, independent=bool(mode))
            max_limit_error[mode] = max(max_limit_error[mode], abs(limit - reference_limit))
            assert abs(limit - reference_limit) < MATRIX_TOL * (1 + abs(reference_limit))
            # Verify the polynomial's removable singularity separately.
            assert abs(_coefficient_from_q(recovered, 1, bool(mode)) - limit) < IDENTITY_TOL
            expected_limit = ((2*T*U-S2*(T*T+V))/(2*T**3) if mode
                              else (T*T-V)/(2*T**3))
            assert abs(limit - expected_limit) < IDENTITY_TOL
            eps = 1e-5
            derivative = (fn(m, p, eps) - fn(m, p, -eps)) / (2 * eps)
            expected_derivative = factor / T**2 - (recovered[0]**2 if mode else recovered[0]) / T
            err = abs(derivative - expected_derivative)
            max_derivative_error = max(max_derivative_error, err)
            assert err < DERIVATIVE_TOL * (1 + abs(expected_derivative))
    log(f"Random life tables: {N_TABLES}; d=1..18; dense, sparse, and no-age-1 fertility.")
    log(f"General matrix comparisons: {comparison_count} ({comparison_count//2} per driver model).")
    log(f"Maximum absolute errors [common, independent]: {max_error.tolist()}")
    log(f"Maximum errors / (1+|reference|): {max_scaled_error.tolist()}")
    log(f"Displayed rational formula vs pole-free implementation: {comparison_count} checks; maximum error {max_displayed_formula_error:.3g}")
    log(f"phi -> 1 comparisons at the matrix Abel limit: {2*N_TABLES}")
    log(f"Maximum limit errors [common, independent]: {max_limit_error.tolist()}")
    log(f"phi=0 identity checks: {2*N_TABLES}; maximum absolute error {max_identity_error:.3g}")
    log(f"phi=0 derivative checks: {2*N_TABLES}; maximum absolute error {max_derivative_error:.3g}")
    log(f"Nontrivial positive-phi sign checks: {sign_count} (raise {positive_cost_changes}, lower {negative_cost_changes})")
    for age in range(2, 13):
        m = np.zeros(age); m[-1] = 1
        for fn in (cost_common_driver, cost_independent_drivers):
            try:
                fn(m, np.ones(age-1), 0.5)
            except ValueError:
                pass
            else:
                raise AssertionError("A semelparous table must be rejected by the primitive formula")
        assert cost_semelparous(age) == 1 / (2 * age)
    log("Nonprimitive safeguards: 22 rejection checks passed (ages 2..12, two models).")


# Direct population-product simulation, independently of the matrix perturbation
# and demographic closed forms. C only accelerates the time stepping.
C_SOURCE = r'''
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { uint64_t state; int cached; double spare; } RNG;
static uint64_t next_u64(RNG *r) {
    uint64_t z = (r->state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static double uniform(RNG *r) {
    return ((next_u64(r) >> 11) + 0.5) * 0x1.0p-53;
}
static double normal(RNG *r) {
    if (r->cached) { r->cached = 0; return r->spare; }
    double radius = sqrt(-2.0 * log(uniform(r)));
    double theta = 6.283185307179586476925286766559 * uniform(r);
    r->spare = radius * sin(theta); r->cached = 1;
    return radius * cos(theta);
}
int simulate(int d, const double *m, const double *p, double lambda,
             const double *initial, double phi, double sigma, int independent,
             int reps, int steps, int burn, uint64_t seed, double *out) {
    double *x = calloc(4*d, sizeof(double));
    double *next = calloc(d, sizeof(double));
    double *z = calloc(d, sizeof(double));
    double *factors = calloc(4*d, sizeof(double));
    if (!x || !next || !z || !factors) return 1;
    const double innovation = sqrt(1.0 - phi*phi);
    const double mean_high = exp(-0.5*sigma*sigma);
    const double mean_low = exp(-0.125*sigma*sigma);
    for (int r=0; r<reps; ++r) {
        RNG rng = {seed + UINT64_C(0xd1342543de82ef95)*(r+1), 0, 0};
        for (int k=0; k<4; ++k)
            for (int i=0; i<d; ++i) x[k*d+i] = initial[i];
        int drivers = independent ? d : 1;
        for (int i=0; i<drivers; ++i) z[i] = normal(&rng);
        double sumlog[4] = {0,0,0,0};
        for (int t=-burn; t<steps; ++t) {
            for (int i=0; i<drivers; ++i) {
                z[i] = phi*z[i] + innovation*normal(&rng);
                double e = exp(0.5*sigma*z[i]);
                factors[i] = e*e*mean_high;
                factors[d+i] = mean_high/(e*e);
                factors[2*d+i] = e*mean_low;
                factors[3*d+i] = mean_low/e;
            }
            for (int k=0; k<4; ++k) {
                double *current = x+k*d;
                double birth = 0;
                for (int i=0; i<d; ++i)
                    birth += m[i]*current[i]*factors[k*d+(independent ? i : 0)]/lambda;
                next[0] = birth;
                double total = birth;
                for (int i=1; i<d; ++i) {
                    next[i] = p[i-1]*current[i-1]/lambda;
                    total += next[i];
                }
                for (int i=0; i<d; ++i) current[i] = next[i]/total;
                if (t>=0) sumlog[k] += log(total);
            }
        }
        double high = -0.5*(sumlog[0]+sumlog[1])/(steps*sigma*sigma);
        double low = -0.5*(sumlog[2]+sumlog[3])/(steps*0.25*sigma*sigma);
        out[3*r] = high;
        out[3*r+1] = low;
        out[3*r+2] = (4*low-high)/3;
    }
    free(x); free(next); free(z); free(factors);
    return 0;
}
'''


def simulate_case(simulator, m, p, phi, independent, seed, semelparous=False):
    lam, q = demographic_distribution(m, p, require_primitive=not semelparous)
    if semelparous:
        initial = np.zeros(len(m)); initial[0] = 1
    else:
        initial = np.r_[1., np.cumprod(p)] / lam**np.arange(len(m))
        initial /= initial.sum()
    m = np.ascontiguousarray(m, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.float64)
    initial = np.ascontiguousarray(initial, dtype=np.float64)
    output = np.empty((REPLICATES, 3), dtype=np.float64)
    ptr = ctypes.POINTER(ctypes.c_double)
    status = simulator(len(m), m.ctypes.data_as(ptr), p.ctypes.data_as(ptr),
                       lam, initial.ctypes.data_as(ptr), phi, SIGMA, int(independent),
                       REPLICATES, STEPS, BURN, seed, output.ctypes.data_as(ptr))
    if status:
        raise RuntimeError("C simulation allocation failed")
    return output


def simulation_checks(log):
    cases = [
        ("scalar", [1], [], 1.07, 0.70),
        ("raises_cost", [.5,.5], None, 1.0, 0.80),
        ("lowers_cost", [.8,0,0,0,.2], None, 1.0, 0.80),
        ("negative_cost", [.9]+[0]*8+[.1], None, 1.0, 0.80),
        ("negative_phi", [.1,.2,.4,.3], [.8,.9,.85], 1.1, -0.65),
        ("iid", [.2,.3,.5], [.75,.9], .95, 0.0),
        ("delayed", [0,.35,.3,.35], [.8,.9,.7], 1.0, 0.60),
        ("persistent", [.35,.2,.25,.2], [.9,.8,.85], 1.03, 0.95),
    ]
    cc = shutil.which("cc")
    if cc is None:
        raise RuntimeError("A C compiler named cc is required for fast direct simulations")
    passed = 0
    worst_z = 0.0
    worst_se = 0.0
    with tempfile.TemporaryDirectory(prefix="leslie-build-", dir=BASE) as folder:
        folder = Path(folder)
        source = folder / "simulate.c"
        library = folder / "simulate.so"
        source.write_text(C_SOURCE)
        env = dict(os.environ, TMPDIR=str(folder))
        subprocess.run([cc, "-O3", "-std=c99", "-shared", "-fPIC", str(source),
                        "-o", str(library), "-lm"], check=True, env=env,
                       capture_output=True, text=True)
        lib = ctypes.CDLL(str(library))
        sim = lib.simulate
        double_ptr = ctypes.POINTER(ctypes.c_double)
        sim.argtypes = [ctypes.c_int, double_ptr, double_ptr, ctypes.c_double,
                        double_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, double_ptr]
        sim.restype = ctypes.c_int
        log("\nSIMULATION: reported quantities are cost / sigma^2; SE from independent replicate means.")
        log("Columns: case model phi theory estimate(s=.04) SE estimate(s=.02) SE extrapolation SE z_extrap PASS")
        for index, (name, q, p, lam, phi) in enumerate(cases):
            m, p = life_table(q, p, lam)
            log(f"TABLE {name}: m={m.tolist()}, p={p.tolist()}, lambda={lam}, q={q}")
            for mode, fn in enumerate((cost_common_driver, cost_independent_drivers)):
                samples = simulate_case(sim, m, p, phi, bool(mode), 892345 + index*12007 + mode*97)
                means = samples.mean(axis=0)
                ses = samples.std(axis=0, ddof=1) / np.sqrt(REPLICATES)
                theory = fn(m, p, phi)
                allowances = np.array([RAW_ALLOWANCE, RAW_ALLOWANCE, EXTRAPOLATED_ALLOWANCE])
                errors = abs(means - theory)
                ok = np.all(errors <= 4*ses + allowances*(1+abs(theory)))
                ok = bool(ok and np.all(ses <= SE_TARGET*(1+abs(theory))))
                z = (means[2]-theory)/ses[2] if ses[2] > 1e-10 else 0.0
                worst_z = max(worst_z, abs(z))
                worst_se = max(worst_se, float(ses.max()))
                log(f"{name:14} {('independent' if mode else 'common'):11} {phi:5.2f} {theory: .9f} "
                    f"{means[0]: .9f} {ses[0]:.3g} {means[1]: .9f} {ses[1]:.3g} "
                    f"{means[2]: .9f} {ses[2]:.3g} {z: .3f} {'PASS' if ok else 'FAIL'}")
                assert ok, (name, mode, theory, means, ses)
                passed += 1
        log("Semelparous simulations start with one occupied age, avoiding finite-time competition between periodic lineages.")
        for age, phi in [(4, .8), (4, -.65), (7, .6)]:
            m = np.zeros(age); m[-1] = 1
            p = np.ones(age-1)
            # Full cycles ensure an exact fraction 1/age of reproductive steps.
            # For age 7, allow the one-cycle finite-window error instead.
            samples = simulate_case(sim, m, p, phi, False, 67291+age+int(100*phi), semelparous=True)
            means = samples.mean(axis=0)
            ses = samples.std(axis=0, ddof=1)/np.sqrt(REPLICATES)
            theory = cost_semelparous(age)
            tolerance = 1 / STEPS + 1e-9
            assert np.all(abs(means - theory) < tolerance)
            log(f"semelparous age={age}, phi={phi}: exact={theory:.9f}, "
                f"estimates={means.tolist()}, SE={ses.tolist()}, finite-window tolerance={tolerance:.3g} PASS")
    log(f"Simulation comparisons passed: {passed} primitive table/model combinations (8 tables, 2 driver models), plus 3 semelparous cases.")
    log(f"Maximum |extrapolated error / SE| excluding exact scalar/semelparous cases: {worst_z:.4f}")
    log(f"Maximum simulation standard error: {worst_se:.6g}")


def example_checks(log):
    log("\nEXPLICIT AUTOCORRELATION EXAMPLES (p_a=1, lambda=1, m=q):")
    examples = [(.5,.5), (.8,0,0,0,.2), (.9,0,0,0,0,0,0,0,0,.1), (.2,.7,0,0,0,0,0,0,0,.1)]
    for q in examples:
        m, p = life_table(q)
        a = np.arange(1, len(q)+1)
        T = a @ q
        V = (a-T)**2 @ q
        values = {phi: cost_common_driver(m,p,phi) for phi in [0,.1,.5,.8,.95]}
        log(f"m={list(q)}, T={T:.9g}, V={V:.9g}, C(phi)={values}, limit={cost_persistent_limit(m,p):.12g}")


def main():
    start = time.monotonic()
    with (BASE / "results_leslie.txt").open("w", buffering=1) as file:
        def log(message):
            print(message, flush=True)
            file.write(message + "\n")
        log("Leslie fertility-noise verification")
        log("Started " + datetime.now(timezone.utc).isoformat())
        log("TOLERANCES FIXED BEFORE CHECKS:")
        log(f"Matrix: |difference| <= {MATRIX_TOL} * (1 + |matrix reference|).")
        log(f"Algebraic identities: absolute {IDENTITY_TOL}; finite-difference derivatives: {DERIVATIVE_TOL}*(1+|reference|).")
        log(f"Simulation raw estimates: |estimate-C| <= 4 SE + {RAW_ALLOWANCE}*(1+|C|).")
        log(f"Simulation extrapolation: |estimate-C| <= 4 SE + {EXTRAPOLATED_ALLOWANCE}*(1+|C|).")
        log(f"Precision requirement: SE <= {SE_TARGET}*(1+|C|).")
        log(f"Simulation: sigma={SIGMA} and {SIGMA/2}, {REPLICATES} independent replicates, "
            f"{STEPS} measured steps and {BURN} burn-in steps each.")
        log("Antithetic +/- z paths cancel first-order sampling noise; each replicate supplies one paired estimate.")
        log("Extrapolation = (4*C_est(.02)-C_est(.04))/3; its SE uses replicate-level coupled estimates.")
        log("Semelparous finite-window tolerance: 1/STEPS + 1e-9; a single occupied initial age is used.")
        try:
            random_checks(log)
            example_checks(log)
            simulation_checks(log)
        except Exception as error:
            log(f"CHECK FAILED: {type(error).__name__}: {error}")
            raise
        log(f"ALL CHECKS PASSED. Elapsed seconds: {time.monotonic()-start:.2f}")


if __name__ == "__main__":
    main()
