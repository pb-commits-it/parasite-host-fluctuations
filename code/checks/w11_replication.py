"""W1.1: replication of secondary's parking-lot (claim c) and return-rate (claim d) results.

Claims tested, with tolerances: `protocols/W1.1-replication.md`, written before this ran.
Part C uses `engine/ring.py` (verified by `checks/r0_verification.py`). Part D integrates the
equations with scipy and takes the Jacobian by finite differences; it shares no code with
`verification/attack_stability.py`.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import ring  # noqa: E402

LAM0, S2_MEAN, PHI = 1.05, 0.5, 0.9
REPS, STEPS, BURN = 400, 60000, 1000
secondary_C = {1.1: 0.0344, 2.0: 0.0567, 6.0: 0.0737, 20.0: 0.0725}
secondary_D = {1: -7.0000, 2: -0.4742, 3: -0.4160, 4: -0.3909, 6: -0.3683, 8: -0.3579}


class GoodAndBadTimes:
    """Two stages; a hidden Gaussian AR(1) series marks bad times when negative.

    mode 'survival':   stage-2 survival is 0.95 in good times and 0.05 in bad.
    mode 'production': stage-2 survival is 1/2; production out of stage 2 is 1.9 g or 0.1 g.
    """

    def __init__(self, s1, g, mode, reps, rng):
        self.s1, self.g, self.mode, self.rng, self.reps = s1, g, mode, rng, reps
        self.z = rng.normal(0.0, 1.0, reps)

    def step(self):
        self.z = PHI * self.z + math.sqrt(1 - PHI ** 2) * self.rng.normal(0.0, 1.0, self.reps)
        bad = self.z < 0
        s = np.empty((self.reps, 2))
        g = np.full((self.reps, 2), self.g)
        s[:, 0] = self.s1
        if self.mode == "survival":
            s[:, 1] = np.where(bad, 0.05, 0.95)
        else:
            s[:, 1] = S2_MEAN
            g[:, 1] = self.g * np.where(bad, 0.1, 1.9)
        return s, g, None


def part_c():
    print("Part C. Two stages, mean-rate growth factor fixed at 1.05, autocorrelation 0.9")
    ok = True
    costs = {}
    for mode in ("survival", "production"):
        print(f"  fluctuation in {mode} of stage 2")
        for i, r1 in enumerate((1.1, 2.0, 6.0, 20.0)):
            s1 = 1 - 1 / r1
            g = math.sqrt((LAM0 - s1) * (LAM0 - S2_MEAN))
            lam0, e_g, _ = ring.elasticities([s1, S2_MEAN], [g, g])
            assert abs(lam0 - LAM0) < 1e-12
            rng = np.random.default_rng(1100 + i + (50 if mode == "production" else 0))
            out = ring.run(GoodAndBadTimes(s1, g, mode, REPS, rng), 2, REPS, STEPS, BURN)
            m, se = ring.mean_se(out["rate"])
            var = float(out["total"].var(ddof=1)) / (STEPS - BURN)
            cost = math.log(LAM0) - m
            costs[(mode, r1)] = (cost, se)
            line = (f"    stay {r1:4.1f} steps: growth {m:+.4f} (s.e. {se:.4f}), cost {cost:+.4f}, "
                    f"variance {var:.4f}, safety {2 * m / var:5.2f}")
            if mode == "survival":
                good = abs(m - secondary_C[r1]) <= 0.001
                ok &= good
                line += f"; secondary {secondary_C[r1]:+.4f} [{'PASS' if good else 'FAIL'}]"
            else:
                line += f"; elasticity of that production step {e_g[1]:.3f}"
            print(line)
    c_lo, c_hi = costs[("production", 1.1)], costs[("production", 20.0)]
    ratio = c_hi[0] / c_lo[0]
    print(f"  C1 replication of secondary's four values: {'PASS' if ok else 'FAIL'}")
    print(f"  C2 cost with production noise at a stay of 20 steps over a stay of 1.1 steps: {ratio:.3f}"
          f" ('does nothing' needs 0.8 to 1.2: {'stands' if 0.8 <= ratio <= 1.2 else 'does not stand'})")
    return ok


def rhs(t, p, n, g, big_r, c):
    d = np.empty(n)
    if n == 1:
        d[0] = g * big_r * p[0] - g * p[0] - c * p[0] ** 2
        return d
    d[0] = g * big_r * p[n - 1] - g * p[0] - c * p[0] ** 2
    d[1:] = g * p[:-1] - g * p[1:]
    return d


def part_d():
    print("Part D. Return to equilibrium; whole-cycle factor 15, cycle time 2")
    big_r, tau, c = 15.0, 2.0, 0.005
    ok, rates = True, {}
    for n in (1, 2, 3, 4, 6, 8):
        g = n / tau
        args = (n, g, big_r, c)
        sol = solve_ivp(rhs, (0, 400), np.full(n, 10.0), args=args, method="LSODA", rtol=1e-12, atol=1e-10)
        eq = sol.y[:, -1]
        assert np.max(np.abs(rhs(0, eq, *args))) < 1e-6, "not at equilibrium"
        jac = np.empty((n, n))
        for j in range(n):
            h = 1e-5 * max(1.0, eq[j])
            up, dn = eq.copy(), eq.copy()
            up[j] += h
            dn[j] -= h
            jac[:, j] = (rhs(0, up, *args) - rhs(0, dn, *args)) / (2 * h)
        ev = np.linalg.eigvals(jac)
        lead = ev[np.argmax(ev.real)]
        rates[n] = lead.real
        # decay rate fitted to a trajectory started 1% away from equilibrium
        t_fit = np.linspace(6.0, 14.0, 200) if n > 1 else np.linspace(0.3, 1.2, 200)
        tr = solve_ivp(rhs, (0, t_fit[-1]), eq * 1.01, args=args, t_eval=t_fit, method="LSODA", rtol=1e-12, atol=1e-12)
        dist = np.linalg.norm(tr.y - eq[:, None], axis=0)
        slope = np.polyfit(t_fit, np.log(dist), 1)[0]
        good = abs(lead.real - secondary_D[n]) <= 0.002 and abs(slope / lead.real - 1) <= 0.02
        ok &= good
        print(f"  n = {n}: leading eigenvalue {lead.real:+.4f}{lead.imag:+.4f}i, fitted decay {slope:+.4f}, "
              f"secondary {secondary_D[n]:+.4f} [{'PASS' if good else 'FAIL'}]; "
              f"largest imaginary part among all eigenvalues {np.max(np.abs(ev.imag)):.3f}; "
              f"leading eigenvalue is {'real' if abs(lead.imag) < 1e-6 else 'complex'}")
    print(f"  D1 replication of secondary's six values: {'PASS' if ok else 'FAIL'}")
    print(f"  D2 return rate at 8 stages over 2 stages: {rates[8] / rates[2]:.3f}; at 2 stages over 1 stage: {rates[2] / rates[1]:.4f}")
    return ok


if __name__ == "__main__":
    a = part_c()
    b = part_d()
    sys.exit(0 if (a and b) else 1)
