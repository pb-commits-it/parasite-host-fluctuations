"""Reproducible numerical checks; run: python3 check_shared.py.

All reported numerical observations are 'checked numerically here', with
counts printed below.  Numerical Jacobians are differentiated from the full
nonlinear vector field, not assembled from the characteristic polynomial.
"""

import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, linear_sum_assignment
from scipy.signal import find_peaks

from shared import eigenvalues_A, eigenvalues_B, threshold_A, threshold_B


SEED = 20261114


def vector_field(rates, equilibrium, kappa):
    rates, equilibrium, kappa = map(np.asarray, (rates, equilibrium, kappa))
    gains = np.roll(rates * equilibrium, -1) / equilibrium * np.exp(kappa)
    regulation = kappa / equilibrium

    def rhs(t, x):
        flux = gains * x * np.exp(-regulation * x)
        return np.roll(flux, 1) - rates * x

    return rhs


def numerical_jacobian(rhs, x, complex_step=True):
    x = np.asarray(x, dtype=float)
    columns = []
    for k in range(len(x)):
        if complex_step:
            z = x.astype(complex)
            z[k] += 1j * 1e-25
            columns.append(np.imag(rhs(0, z)) / 1e-25)
        else:
            h = np.cbrt(np.finfo(float).eps) * max(1.0, abs(x[k]))
            plus, minus = x.copy(), x.copy()
            plus[k] += h
            minus[k] -= h
            columns.append((rhs(0, plus) - rhs(0, minus)) / (2 * h))
    return np.column_stack(columns)


def matching_error(observed, expected):
    cost = abs(observed[:, None] - expected[None, :])
    i, j = linear_sum_assignment(cost)
    return float(np.max(cost[i, j]) / (1.0 + np.max(abs(expected))))


def spectral_checks(log):
    rng = np.random.default_rng(SEED)
    worst_cs = worst_fd = worst_residual = 0.0
    count = 0
    log('CHECKED NUMERICALLY HERE: random equal-stage spectral checks')
    for version in ('A', 'B'):
        for n in range(1, 10):
            local_cs = local_fd = 0.0
            threshold = (threshold_A if version == 'A' else threshold_B)(n)
            upper = max(12.0, 1.5 * threshold) if math.isfinite(threshold) else 14.0
            for _ in range(20):
                L = float(np.exp(rng.uniform(np.log(.03), np.log(upper))))
                a = float(np.exp(rng.uniform(np.log(.15), np.log(9.0))))
                equilibrium = np.exp(rng.uniform(-.8, .8, n))
                kappa = np.zeros(n) if version == 'A' else np.full(n, L / n)
                if version == 'A':
                    kappa[0] = L
                rhs = vector_field(np.full(n, a), equilibrium, kappa)
                expected = (eigenvalues_A if version == 'A' else eigenvalues_B)(n, L, a)
                cs = np.linalg.eigvals(numerical_jacobian(rhs, equilibrium))
                fd = np.linalg.eigvals(numerical_jacobian(rhs, equilibrium, False))
                error_cs, error_fd = matching_error(cs, expected), matching_error(fd, expected)
                residual = max(abs(rhs(0, equilibrium))) / (1 + a * max(equilibrium))
                local_cs, local_fd = max(local_cs, error_cs), max(local_fd, error_fd)
                worst_residual = max(worst_residual, residual)
                assert error_cs < 2e-10, (version, n, L, a, error_cs)
                assert error_fd < 2e-6, (version, n, L, a, error_fd)
                assert residual < 2e-13
                count += 1
            worst_cs, worst_fd = max(worst_cs, local_cs), max(worst_fd, local_fd)
            log(f'  {version} n={n}: 20 cases; max matched relative errors '
                f'complex-step={local_cs:.3e}, central-difference={local_fd:.3e}')
    log(f'PASS: {count} random cases; max errors CS={worst_cs:.3e}, FD={worst_fd:.3e}; '
        f'max equilibrium residual={worst_residual:.3e}')
    return count


def boundary_checks(log):
    count = 0
    for n in range(1, 10):
        for version in ('A', 'B'):
            threshold = (threshold_A if version == 'A' else threshold_B)(n)
            eig = eigenvalues_A if version == 'A' else eigenvalues_B
            if math.isfinite(threshold):
                for factor in (.999, 1.0, 1.001):
                    alpha = max(eig(n, factor * threshold, n).real)
                    if factor < 1:
                        assert alpha < 0
                    elif factor == 1:
                        assert abs(alpha) < 1e-12
                    else:
                        assert alpha > 0
                    count += 1
            else:
                for L in (.01, 1.0, 4.0, 100.0):
                    assert max(eig(n, L, n).real) < 0
                    count += 1
    # Exact repeated-root cases are compared as matrices: numerical eigensolvers
    # are ill-conditioned at A's defective repeated root when n > 1.
    repeated = 0
    for n in range(1, 10):
        for version in ('A', 'B'):
            L = 1.0 if version == 'A' else float(n)
            kappa = np.zeros(n) if version == 'A' else np.ones(n)
            if version == 'A':
                kappa[0] = L
            rhs = vector_field(np.full(n, 2.0), np.ones(n), kappa)
            j = numerical_jacobian(rhs, np.ones(n))
            nilpotent = j + 2 * np.eye(n)
            assert np.linalg.norm(np.linalg.matrix_power(nilpotent, n)) < 1e-12
            repeated += 1
    log(f'CHECKED NUMERICALLY HERE: {count} threshold-side/boundary cases and '
        f'{repeated} repeated-root cases: PASS')


def general_rate_checks(log):
    rng = np.random.default_rng(SEED + 1)
    count = guaranteed = 0
    max_poly_error = 0.0
    for n in range(1, 10):
        for _ in range(30):
            rates = np.exp(rng.uniform(-2.0, 2.0, n))
            equilibrium = np.exp(rng.uniform(-.5, .5, n))
            kappa = rng.uniform(.05, 4.0, n)
            rhs = vector_field(rates, equilibrium, kappa)
            eig = np.linalg.eigvals(numerical_jacobian(rhs, equilibrium))
            P = float(np.prod(1 - kappa))
            for lam in eig:
                lhs = np.prod(lam + rates)
                rhs_char = np.prod(rates) * P
                err = abs(lhs - rhs_char) / (1 + abs(rhs_char) + np.prod(abs(lam) + rates))
                max_poly_error = max(max_poly_error, float(err))
                assert err < 1e-11
            lower = -math.inf if n <= 2 else -math.cos(math.pi / n) ** (-n)
            if lower < P < 1:
                assert max(eig.real) < 0
                guaranteed += 1
            count += 1
    log(f'CHECKED NUMERICALLY HERE: {count} arbitrary-rate characteristic-polynomial cases; '
        f'{guaranteed} satisfy the sharp rate-independent sufficient condition; '
        f'all {guaranteed} stable; max scaled polynomial residual={max_poly_error:.3e}: PASS')


def alternating_checks(log):
    count = 0
    log('CHECKED NUMERICALLY HERE: explicit asymmetric positive equilibria in B')
    for n in (2, 4, 6, 8):
        for s in (2.05, 2.2):
            d = brentq(lambda d: s * np.tanh(d / 2) - d, 1e-7, s)
            x = np.array([s - d, s + d] * (n // 2))
            rhs = lambda t, x: n * (np.roll(x * np.exp(s - x), 1) - x)
            residual = float(max(abs(rhs(0, x))))
            alpha = float(max(np.linalg.eigvals(numerical_jacobian(rhs, x)).real))
            assert residual < 1e-10 and alpha < 0
            log(f'  n={n}, L={n*s:.6g}: low={s-d:.10g}, high={s+d:.10g}, '
                f'max residual={residual:.3e}, spectral abscissa={alpha:.8g}')
            count += 1
    log(f'PASS: {count} asymmetric-equilibrium cases (each has a distinct one-stage rotation).')


def odd_asymmetric_check(log):
    # A sign bracket isolates one nonsymmetric 3-cycle; this does not claim
    # to enumerate every equilibrium or prove an exact equilibrium count.
    n, s = 3, 3.3

    def total_minus_L(u):
        total = u
        for _ in range(n - 1):
            u *= np.exp(s - u)
            total += u
        return total - n * s

    first = brentq(total_minus_L, .02, .03, xtol=1e-15)
    x = [first]
    for _ in range(n - 1):
        x.append(x[-1] * np.exp(s - x[-1]))
    x = np.array(x)
    rhs = lambda t, u: n * (np.roll(u * np.exp(s - u), 1) - u)
    residual = max(abs(rhs(0, x)))
    assert residual < 1e-10 and max(abs(x - s)) > 1
    log('CHECKED NUMERICALLY HERE: 1 additional odd-stage asymmetric-equilibrium '
        f'case: B n=3, L=9.9, x={np.array2string(x, precision=11)}, '
        f'max residual={residual:.3e}; PASS. Its cyclic rotations are also equilibria.')


def normal_form_checks(log):
    count, worst = 0, 0.0
    for version in ('A', 'B'):
        for n in range(3, 10):
            if version == 'B' and n % 2 == 0:
                continue
            L = (threshold_A if version == 'A' else threshold_B)(n)
            level = L if version == 'A' else L / n
            equilibrium = np.full(n, level)
            kappa = np.zeros(n) if version == 'A' else np.full(n, level)
            b, c = np.zeros(n), np.zeros(n)
            if version == 'A':
                kappa[0], b[0], c[0] = L, L - 2, 3 - L
            else:
                b[:], c[:] = level - 2, 3 - level
            rhs = vector_field(np.ones(n), equilibrium, kappa)
            J = numerical_jacobian(rhs, equilibrium)
            omega = math.tan(math.pi / n)
            values, vectors = np.linalg.eig(J.astype(complex))
            index = int(np.argmin(abs(values - 1j * omega)))
            q = vectors[:, index] / vectors[0, index]
            vectors[:, index] = q
            left = np.linalg.inv(vectors)[index, :]
            B = lambda u, v: np.roll(b * u * v, 1)
            C = lambda u, v, w: np.roll(c * u * v * w, 1)
            G_matrix = left @ (C(q, q, q.conj())
                - 2 * B(q, np.linalg.solve(J, B(q, q.conj())))
                + B(q.conj(), np.linalg.solve(2j * omega * np.eye(n) - J, B(q, q))))
            h = 1 + 1j * omega
            if version == 'A':
                Q = L - 1
                G_formula = -h / (n * Q) * (Q - 4 + 8 / (Q + 1)
                    + (Q - 1) ** 2 / ((1 + 2j * omega) ** n + Q))
            else:
                r = level - 1
                zeta = -h / r
                G_formula = zeta * (r - 4 + 8 / (r + 1)
                    + (r - 1) ** 2 * zeta ** 2 / (1 + 2j * omega + r * zeta ** 2))
            error = abs(G_formula - G_matrix)
            assert error < 1e-10 and G_formula.real < 0
            count += 1
            worst = max(worst, float(error))
    log(f'CHECKED NUMERICALLY HERE: {count} Hopf cubic-coefficient formulas '
        f'against full matrix contractions; all real parts negative; '
        f'max absolute difference={worst:.3e}: PASS')


def integration_cases():
    cases = [
        ('A', 1, 6.0, 0), ('B', 1, 6.0, 0), ('A', 2, 6.0, 0),
        ('A', 3, 8.1, 0), ('A', 3, 9.9, 0),
        ('A', 4, 4.5, 0), ('A', 4, 6.0, 0),
        ('A', 5, 3.5, 0), ('A', 5, 4.3, 0),
        ('A', 8, 2.6, 0), ('A', 8, 3.2, 0),
        ('B', 2, 3.6, 0), ('B', 2, 4.4, 0), ('B', 2, 4.4, 1),
        ('B', 3, 8.1, 0), ('B', 3, 9.9, 0),
        ('B', 4, 6.0, 0), ('B', 4, 7.2, 0),
        ('B', 4, 8.8, 0), ('B', 4, 8.8, 1),
        ('B', 6, 10.8, 0), ('B', 6, 13.2, 0), ('B', 6, 13.2, 1),
        ('B', 8, 14.4, 0), ('B', 8, 17.6, 0), ('B', 8, 17.6, 1),
        ('B', 2, 8.0, 0), ('B', 4, 10.4, 0),
        ('B', 6, 16.8, 0), ('B', 8, 20.8, 0),
    ]
    return cases


def integrate_one(version, n, L, phase, tighter=False):
    # tau = 1 in every simulation.  In A, K=1 and r_1=exp(L), r_k=1
    # otherwise, giving the equilibrium (L,...,L).  In B, all K=1 and
    # all r=exp(L/n), giving the symmetric equilibrium (L/n,...,L/n).
    a = float(n)
    level = L if version == 'A' else L / n
    equilibrium = np.full(n, level)
    kappa = np.zeros(n) if version == 'A' else np.full(n, level)
    if version == 'A':
        kappa[0] = L
    rhs = vector_field(np.full(n, a), equilibrium, kappa)
    rng = np.random.default_rng(SEED + 1000 + 19 * n)
    perturbation = .01 * rng.normal(size=n) + .012 * (-1.0) ** np.arange(n)
    # A rotation of the same nonsymmetric perturbation selects the other
    # alternating equilibrium without changing parameters.
    perturbation = np.roll(perturbation, phase)
    initial = equilibrium * (1 + perturbation)
    T = 240.0
    times = np.linspace(0, T, 4801)
    solution = solve_ivp(rhs, (0, T), initial, method='DOP853', t_eval=times,
                         rtol=2e-11 if tighter else 1e-9,
                         atol=2e-13 if tighter else 1e-11,
                         max_step=.05 if tighter else .1)
    assert solution.success, solution.message
    tail = solution.y[:, solution.t >= T - 60]
    span = float(max(np.ptp(tail, axis=1)))
    residual = float(max(abs(rhs(T, solution.y[:, -1]))))
    distance = float(max(abs(solution.y[:, -1] - equilibrium)))
    if span < 1e-5 and residual < 1e-6:
        classification = 'symmetric equilibrium' if distance < 1e-5 else 'asymmetric equilibrium'
    else:
        classification = 'persistent tail variation'
    signal = tail[0]
    peaks, _ = find_peaks(signal, prominence=max(1e-6, .05 * np.ptp(signal)))
    periods = np.diff(peaks) * .05
    period = float(np.mean(periods)) if len(periods) >= 2 else math.nan
    period_sd = float(np.std(periods)) if len(periods) >= 2 else math.nan
    alpha = float(max((eigenvalues_A if version == 'A' else eigenvalues_B)(n, L, a).real))
    # The expected stable cases must actually return, and unstable cases must
    # leave the symmetric equilibrium for the specified nonsymmetric starts.
    if alpha < -1e-6:
        assert classification == 'symmetric equilibrium', (version, n, L, classification)
    if alpha > 1e-6:
        assert distance > 1e-3, (version, n, L, distance)
    return dict(version=version, n=n, L=L, phase=phase, alpha=alpha,
                classification=classification, span=span, residual=residual,
                distance=distance, min_value=float(np.min(solution.y)),
                final=solution.y[:, -1].tolist(), tail_min=float(np.min(signal)),
                tail_max=float(np.max(signal)), period=period, period_sd=period_sd,
                nfev=solution.nfev)


def integration_checks(log):
    records = []
    log('CHECKED NUMERICALLY HERE: nonlinear integration; tau=1, T=240, '
        'tail=[180,240], dt_output=.05, DOP853 rtol=1e-9 atol=1e-11 max_step=.1.')
    log('Starts: reproducible nonsymmetric ~1% perturbations; phase=1 rotates '
        'the same perturbation one stage. Tail variation is finite-time evidence, '
        'not a proof of a periodic/global attractor.')
    for version, n, L, phase in integration_cases():
        result = integrate_one(version, n, L, phase)
        records.append(result)
        log(f"  {version} n={n} L={L:g} phase={phase}: alpha={result['alpha']:.7g}; "
            f"{result['classification']}; tail_span={result['span']:.6g}; "
            f"final_rhs={result['residual']:.3e}; min_state={result['min_value']:.4g}")
        if result['classification'] == 'persistent tail variation':
            log(f"    stage1 tail range=[{result['tail_min']:.8g},{result['tail_max']:.8g}]; "
                f"peak period={result['period']:.6g} (sd={result['period_sd']:.3g})")
        else:
            log('    final=' + np.array2string(np.array(result['final']), precision=9))
    log(f'PASS: {len(records)} nonlinear trajectories; all linearly stable starts returned; '
        'all linearly unstable starts departed from the symmetric equilibrium.')
    # Recheck key conclusions with tighter tolerances and half the maximum step.
    refinements = [('A', 4, 6.0, 0), ('B', 4, 6.0, 0), ('B', 4, 8.8, 0),
                   ('B', 4, 10.4, 0), ('B', 6, 16.8, 0)]
    for case in refinements:
        fine = integrate_one(*case, tighter=True)
        coarse = next(r for r in records if (r['version'], r['n'], r['L'], r['phase']) == case)
        assert fine['classification'] == coarse['classification']
        # Compare amplitude, rather than phase-sensitive final positions.
        amplitude_delta = abs(fine['span'] - coarse['span'])
        assert amplitude_delta < 1e-3 * (1 + coarse['span'])
        log(f'  refinement {case}: same outcome; tail-span difference={amplitude_delta:.3e}')
    log(f'CHECKED NUMERICALLY HERE: {len(refinements)} tighter-tolerance reruns: PASS; '
        f'{len(records) + len(refinements)} integrations total.')


def main():
    output = []

    def log(message):
        output.append(message)
        print(message, flush=True)

    log('Independent checks for derivation_shared.md / shared.py')
    log(f'Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}; seed={SEED}')
    log('DERIVED HERE: thresholds L=ln(R0); equality is a nonhyperbolic boundary.')
    for n in range(1, 9):
        log(f'  n={n}: A={threshold_A(n):.12g}, B={threshold_B(n):.12g}')
    spectral_checks(log)
    boundary_checks(log)
    general_rate_checks(log)
    alternating_checks(log)
    odd_asymmetric_check(log)
    normal_form_checks(log)
    integration_checks(log)
    log('ALL CHECKS PASSED. Numerical observations support, but do not replace, the derivation.')
    Path('results_shared.txt').write_text('\n'.join(output) + '\n')


if __name__ == '__main__':
    main()
