"""Fourth-order small-noise growth cost for a ring (NumPy only)."""
import numpy as np


def _parameters(n, s, gbar, phi):
    if isinstance(n, (bool, np.bool_)) or int(n) != n or n < 1:
        raise ValueError('n must be a positive integer')
    if not np.isfinite(s) or not 0 < s < 1:
        raise ValueError('s must satisfy 0 < s < 1')
    if not np.isfinite(gbar) or gbar <= 0:
        raise ValueError('gbar must be positive')
    if not np.isfinite(phi) or not 0 <= phi < 1:
        raise ValueError('phi must satisfy 0 <= phi < 1')
    p = s / (s + gbar)
    return int(n), p, 1.0-p


def second_order_coefficient(n, s, gbar, phi):
    n, p, q = _parameters(n, s, gbar, phi)
    if phi == 0 or n == 1:
        return q*q/(2*n)
    r = phi*q/(1-phi*p)
    rn = np.exp(n*np.log(r))
    return q*q*(1+phi)/(2*n*(1-phi)) - q*rn/(-np.expm1(n*np.log(r)))


def _fourth_order_parts(n, p, q, phi):
    """Return H and (V, W, M, T); see derivation_h4.md for definitions."""
    idx = np.arange(n)
    zeta = np.exp(-2j*np.pi*idx/n)
    a = p+q*zeta
    c = 1-phi
    # Covariance of the linear logarithmic increment and its AR(1) input.
    v = c*(1+phi-2*p*phi)/(p*np.abs(1-phi*a)**2)
    v[0] = 1.0
    u = c/(1-phi*a.conj())
    V = np.mean(v)
    d = zeta-1
    b = p*q*d/2
    gap = 4*p*q*np.sin(np.pi*idx/n)**2
    gap[0] = 1.0  # this mode is excluded from the composition
    w1 = a/gap
    w2 = -c/(gap*(1-phi*a.conj()))
    w1[0] = w2[0] = 0
    Fsingle = w1*v+w2*u
    kappa = p*q*((1-2*q)*V-1)/2
    Tlinear = kappa*np.sum(d.conj()*Fsingle).real/n
    Wsum = 0.0
    Msum = 0j
    Tsum = 0j
    # Chunk rows to keep memory O(n), and accumulate all n^2 Fourier pairs.
    for i in range(n):
        k = (i+idx) % n
        keep = k != 0
        z = a[k].conj().copy()
        z[~keep] = 0.0
        # S = (I-z*(T_i kron T_j))^{-1} (v_i kron v_j),
        # solved explicitly as an upper triangular four-dimensional system.
        S4 = u[i]*u/(1-z*phi*phi)
        S2 = (v[i]*u-z*c*phi*S4)/(1-z*a[i]*phi)
        S3 = (u[i]*v-z*c*phi*S4)/(1-z*a*phi)
        S1 = (v[i]*v-z*a[i]*c*S2-z*a*c*S3+z*c*c*S4)/(1-z*a[i]*a)
        F1 = a[i]*a*S1-a[i]*c*S2-a*c*S3+c*c*S4
        F3 = a*phi*S3-c*phi*S4
        Wsum += np.sum((2*S1.real-v[i]*v)[keep])
        Msum += np.sum((b[k].conj()*F1)[keep])
        if i:
            Tsum += np.sum((d[i].conjugate()*b[k].conj()
                            *(w1[i]*F1+w2[i]*F3))[keep])
    W = p*q*Wsum/(2*n*n)
    M = 2*Msum.real/(n*n)
    T = Tlinear + 2*p*q*Tsum.real/(n*n)
    H = -p*q/2*(0.25+W+2*T) - p*q*(1-2*q)/2*(M-V/2) - p*q*(1-6*p*q)/8*V*V
    return float(H), (float(V), float(W), float(M), float(T))


def fourth_order_coefficient(n, s, gbar, phi):
    """Return H_n, using an exact O(n^2) Fourier sum with no lag truncation."""
    n, p, q = _parameters(n, s, gbar, phi)
    return _fourth_order_parts(n, p, q, phi)[0]


def iid_fourth_order(n, s, gbar):
    n, p, q = _parameters(n, s, gbar, 0)
    return q*q/(4*n)+q**3*(n*n-1)/(24*p*n*n)-q**3/(n*n)+3*q**4/(4*n*n)


def main():
    s, gbar = 2/3, 0.4
    print('FOURIER COEFFICIENTS; s=2/3, gbar=0.4')
    print('phi     n                H_n              H_n/C_n')
    for phi in (0.0, 0.5, 0.9):
        for n in (1, 2, 3, 4, 8, 16, 32, 64, 128, 512):
            H = fourth_order_coefficient(n, s, gbar, phi)
            C = second_order_coefficient(n, s, gbar, phi)
            print(f'{phi:3.1f} {n:5d} {H:#20.10g} {H/C:#20.10g}')


if __name__ == '__main__':
    main()
