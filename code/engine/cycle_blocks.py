"""A parasite's life cycle as a ring of blocks, each with its own state.

Block k takes a scalar input u_k (infection pressure from the block before it), has a state
vector, and gives a scalar output v_k = u_{k+1}.  The blocks interact in no other way.

Block kinds (all rates positive unless stated):

  L  no regulation                dw/dt = u - a w                       v = c w
  R  regulated transfer           dw/dt = u - a w                       v = c f(w) w
       f(w) = exp(-w/Kd)  (form "exp")  or  1/(1 + w/Kd)  (form "hyp")
  S  N hosts, S or I              dI/dt = u (N - I) - a I               v = c I
  V  dynamic host, harmed         dN/dt = Phi(N) - alpha I
                                  dI/dt = u (N - I) - a I               v = c I
  M  macroparasite, dynamic host  dH/dt = Phi(H) - alpha P
                                  dP/dt = u H - xi P - alpha eps P^2/H  v = lam P

  Phi(N) = r N (1 - N/K)   (K finite)   or   r N   (K = inf: no self-regulation).

A block is a dict; `make_*` build a block whose equilibrium is prescribed (so that no root
finding is needed), leaving the output scale c (or lam) to be set by `close_cycle`.

Nothing here uses the transfer functions derived in notes/coupled-hosts.md except the
functions whose names start with `tf_`; the nonlinear right-hand side and the finite-difference
Jacobian are independent of them, which is what the checks rely on.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------- blocks with a prescribed equilibrium

def make_L(a, ustar):
    return {"kind": "L", "a": a, "u": ustar, "state": np.array([ustar / a]), "c": None}


def make_R(a, ustar, kappa, form="exp"):
    w = ustar / a
    if form == "exp":
        Kd = w / kappa
    elif form == "hyp":
        if not 0 < kappa < 1:
            raise ValueError("hyperbolic form has 0 < kappa < 1")
        Kd = w * (1 - kappa) / kappa
    else:
        raise ValueError(form)
    return {"kind": "R", "a": a, "u": ustar, "Kd": Kd, "form": form, "kappa": kappa,
            "state": np.array([w]), "c": None}


def make_S(a, ustar, N):
    I = ustar * N / (ustar + a)
    return {"kind": "S", "a": a, "u": ustar, "N": N, "state": np.array([I]), "c": None}


def make_V(a, ustar, alpha, r, K):
    """K = math.inf: no self-regulation; then alpha p = r is forced and the caller must have
    chosen ustar = a p/(1 - p) with p = r/alpha (checked).  N* is then free and set to 1."""
    p = ustar / (ustar + a)
    if math.isinf(K):
        if abs(alpha * p - r) > 1e-12 * r:
            raise ValueError("no self-regulation needs alpha*p = r")
        N = 1.0
    else:
        if alpha * p >= r:
            raise ValueError("no positive host equilibrium: alpha*p >= r")
        N = K * (1 - alpha * p / r)
    return {"kind": "V", "a": a, "u": ustar, "alpha": alpha, "r": r, "K": K,
            "state": np.array([N, p * N]), "c": None}


def make_M(xi, m, alpha, eps, r, K):
    """m = P*/H* is prescribed; the input that holds it is u* = xi m + alpha eps m^2."""
    ustar = xi * m + alpha * eps * m * m
    if math.isinf(K):
        if abs(alpha * m - r) > 1e-12 * r:
            raise ValueError("no self-regulation needs alpha*m = r")
        H = 1.0
    else:
        if alpha * m >= r:
            raise ValueError("no positive host equilibrium: alpha*m >= r")
        H = K * (1 - alpha * m / r)
    return {"kind": "M", "xi": xi, "u": ustar, "alpha": alpha, "eps": eps, "r": r, "K": K,
            "state": np.array([H, m * H]), "c": None}


def _phi(x, r, K):
    return r * x if math.isinf(K) else r * x * (1 - x / K)


def _raw_output(b, w):
    """Output with c = 1."""
    k = b["kind"]
    if k == "L":
        return w[0]
    if k == "R":
        f = math.exp(-w[0] / b["Kd"]) if b["form"] == "exp" else 1.0 / (1.0 + w[0] / b["Kd"])
        return f * w[0]
    return w[-1]          # S: I;  V: I;  M: P


def close_cycle(blocks):
    """Set each block's output scale so that v_k* = u_{k+1}*.  Returns the list."""
    n = len(blocks)
    for k, b in enumerate(blocks):
        b["c"] = blocks[(k + 1) % n]["u"] / _raw_output(b, b["state"])
    return blocks


# ---------------------------------------------------------------- the nonlinear model

def sizes(blocks):
    return [len(b["state"]) for b in blocks]


def equilibrium(blocks):
    return np.concatenate([b["state"] for b in blocks])


def rhs(blocks, x, mult=None):
    """dx/dt.  `mult`, if given, is a vector of factors applied to the inputs (one per block)."""
    n = len(blocks)
    out = np.empty_like(x)
    pos = 0
    states = []
    for b in blocks:
        d = len(b["state"])
        states.append(x[pos:pos + d])
        pos += d
    v = [b["c"] * _raw_output(b, w) for b, w in zip(blocks, states)]
    pos = 0
    for k, (b, w) in enumerate(zip(blocks, states)):
        u = v[(k - 1) % n]
        if mult is not None:
            u = u * mult[k]
        kind = b["kind"]
        if kind in ("L", "R"):
            out[pos] = u - b["a"] * w[0]
        elif kind == "S":
            out[pos] = u * (b["N"] - w[0]) - b["a"] * w[0]
        elif kind == "V":
            N, I = w
            out[pos] = _phi(N, b["r"], b["K"]) - b["alpha"] * I
            out[pos + 1] = u * (N - I) - b["a"] * I
        elif kind == "M":
            H, P = w
            out[pos] = _phi(H, b["r"], b["K"]) - b["alpha"] * P
            out[pos + 1] = u * H - b["xi"] * P - b["alpha"] * b["eps"] * P * P / H
        pos += len(w)
    return out


def jacobian_fd(blocks, x=None, rel=1e-6):
    """Central finite differences of `rhs` at x (default: the prescribed equilibrium)."""
    if x is None:
        x = equilibrium(blocks)
    d = len(x)
    J = np.empty((d, d))
    for j in range(d):
        h = rel * max(abs(x[j]), 1e-3)
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        J[:, j] = (rhs(blocks, xp) - rhs(blocks, xm)) / (2 * h)
    return J


def integrate(blocks, x0, t_end, dt, mult_fn=None, record_every=1):
    """Fixed-step fourth-order Runge-Kutta.  mult_fn(t) -> vector of input factors, or None.
    Returns (times, states); the last state is always recorded, whatever record_every is."""
    steps = int(round(t_end / dt))
    x = np.array(x0, dtype=float)
    ts, xs = [0.0], [x.copy()]
    for i in range(steps):
        t = i * dt
        if mult_fn is None:
            k1 = rhs(blocks, x)
            k2 = rhs(blocks, x + 0.5 * dt * k1)
            k3 = rhs(blocks, x + 0.5 * dt * k2)
            k4 = rhs(blocks, x + dt * k3)
        else:
            m1, m2, m3 = mult_fn(t), mult_fn(t + 0.5 * dt), mult_fn(t + dt)
            k1 = rhs(blocks, x, m1)
            k2 = rhs(blocks, x + 0.5 * dt * k1, m2)
            k3 = rhs(blocks, x + 0.5 * dt * k2, m2)
            k4 = rhs(blocks, x + dt * k3, m3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if (i + 1) % record_every == 0 or i == steps - 1:
            ts.append((i + 1) * dt)
            xs.append(x.copy())
    return np.array(ts), np.array(xs)


# ---------------------------------------------------------------- derived transfer functions (the theory under test)

def _rho(b):
    x = b["state"][0]
    return -b["r"] if math.isinf(b["K"]) else -b["r"] * (1 - 2 * x / b["K"])


def tf_polys(b):
    """(numerator, denominator) of E(z) = (u*/v*) T(z) as numpy poly1d, from notes/coupled-hosts.md."""
    k = b["kind"]
    if k == "L":
        return np.poly1d([b["a"]]), np.poly1d([1.0, b["a"]])
    if k == "R":
        return np.poly1d([(1 - b["kappa"]) * b["a"]]), np.poly1d([1.0, b["a"]])
    if k == "S":
        return np.poly1d([b["a"]]), np.poly1d([1.0, b["a"] + b["u"]])
    if k == "V":
        rho, a, u, al = _rho(b), b["a"], b["u"], b["alpha"]
        return np.poly1d([a, a * rho]), np.poly1d([1.0, rho + u + a, rho * (u + a) + al * u])
    if k == "M":
        rho, xi, al, eps = _rho(b), b["xi"], b["alpha"], b["eps"]
        m = b["state"][1] / b["state"][0]
        u = b["u"]
        g = xi + al * eps * m
        A = xi + 2 * al * eps * m
        return np.poly1d([g, g * rho]), np.poly1d([1.0, rho + A, rho * A + al * (u + al * eps * m * m)])
    raise ValueError(k)


def char_poly(blocks):
    """Prod D_k - Prod N_k."""
    num, den = np.poly1d([1.0]), np.poly1d([1.0])
    for b in blocks:
        nk, dk = tf_polys(b)
        num, den = num * nk, den * dk
    return den - num


def loop_gain(blocks, omega):
    """L(i omega) for an array of omega."""
    z = 1j * np.asarray(omega, dtype=float)
    out = np.ones_like(z)
    for b in blocks:
        nk, dk = tf_polys(b)
        out = out * nk(z) / dk(z)
    return out


def block_gain(b, omega):
    z = 1j * np.asarray(omega, dtype=float)
    nk, dk = tf_polys(b)
    return nk(z) / dk(z)


def rho_of(b):
    return _rho(b)
