"""Independent pathwise Taylor check, using generic polynomial arithmetic.

No Fourier covariances, lag sums, or Wick contractions enter the simulation.
Replicas are independent; the standard error is computed across replica means.
"""
import argparse
import time
import numpy as np
from h4 import fourth_order_coefficient, second_order_coefficient, iid_fourth_order


def multiply(a, b):
    """Truncated ordinary power-series product, coefficient axis first."""
    out = np.zeros_like(a)
    for k in range(len(a)):
        for j in range(k+1):
            out[k] += a[j]*b[k-j]
    return out


def divide(a, b):
    """Formal power-series division, coefficient axis first."""
    out = np.empty_like(a)
    for k in range(len(a)):
        out[k] = a[k]
        for j in range(1, k+1):
            out[k] -= b[j]*out[k-j]
        out[k] /= b[0]
    return out


def logarithm(a):
    """log(a) by integrating the formal series a'/a."""
    derivative = np.zeros_like(a)
    for k in range(1, len(a)):
        derivative[k-1] = k*a[k]
    ratio = divide(derivative, a)
    out = np.empty_like(a)
    out[0] = np.log(a[0])
    for k in range(1, len(a)):
        out[k] = ratio[k-1]/k
    return out


def exponential_environment(xi, degree=4):
    """exp(sigma*xi - sigma^2/2), from the formal derivative identity."""
    out = np.empty((degree+1,)+xi.shape)
    out[0] = 1
    out[1] = xi
    for k in range(2, degree+1):
        out[k] = (xi*out[k-1]-out[k-2])/k
    return out


def simulate(n, s, gbar, phi, steps=400000, replicas=128, seed=20261114):
    p = s/(s+gbar)
    q = 1-p
    rng = np.random.default_rng(seed)
    xi = rng.normal(size=(replicas,n))  # stationary from the first sample
    y = np.zeros((5,replicas,n))
    y[0] = 1
    if n > 1:
        rho = abs(p+q*np.exp(2j*np.pi/n))
        gap = min(-np.log(rho), -np.log(phi) if phi else np.inf)
    else:
        gap = -np.log(phi) if phi else 1
    burn = max(1000, int(np.ceil(35/gap)))
    total = np.zeros((5,replicas))
    first = np.zeros_like(total)
    started = time.monotonic()
    for t in range(burn+steps):
        h = exponential_environment(xi)
        raw = p*y + q*np.roll(multiply(h,y),1,axis=-1)
        growth = raw.mean(axis=-1,keepdims=True)
        y = divide(raw,growth)
        if t >= burn:
            obs = -logarithm(growth)[...,0]
            total += obs
            if t-burn < steps//2:
                first += obs
        xi = phi*xi + np.sqrt(1-phi*phi)*rng.normal(size=xi.shape)
    means = total/steps
    H = means[4].mean()
    se = means[4].std(ddof=1)/np.sqrt(replicas)
    C = means[2].mean()
    cse = means[2].std(ddof=1)/np.sqrt(replicas)
    half1 = first[4]/(steps//2)
    half2 = (total[4]-first[4])/(steps-steps//2)
    drift = half2-half1
    return dict(n=n,phi=phi,H=H,se=se,C=C,cse=cse,
                drift=drift.mean(),drift_se=drift.std(ddof=1)/np.sqrt(replicas),
                burn=burn,seconds=time.monotonic()-started)


def exact_checks():
    max_iid = 0.0
    max_scalar = 0.0
    max_c = 0.0
    from h4 import _fourth_order_parts
    for q in (.07,.25,.375,.5,.8,.93):
        s = .6
        gbar = s*q/(1-q)
        for n in (1,2,3,4,8,17,64):
            max_iid = max(max_iid,abs(fourth_order_coefficient(n,s,gbar,0)-iid_fourth_order(n,s,gbar)))
            for phi in (0,.1,.5,.9,.97):
                _,parts = _fourth_order_parts(n,1-q,q,phi)
                derived_c = q/2-(1-q)*q*parts[0]/2
                max_c = max(max_c,abs(derived_c-second_order_coefficient(n,s,gbar,phi)))
        for phi in (0,.1,.5,.9,.99):
            expect = q*q*(1-q)*(1-3*q)/4
            max_scalar = max(max_scalar,abs(fourth_order_coefficient(1,s,gbar,phi)-expect))
    # a_pi = phi is a repeated temporal eigenvalue. The triangular resolvent
    # must remain well-defined; no division by a_k-phi is used.
    resonant = fourth_order_coefficient(4, .6, .2, .5)
    scale_a = fourth_order_coefficient(8,.4,.24,.9)
    scale_b = fourth_order_coefficient(8,2/3,.4,.9)
    print('ALGEBRAIC CHECKS (absolute errors)')
    print(f'phi=0 formula, grid: {max_iid:.4g}')
    print(f'n=1 formula, grid: {max_scalar:.4g}')
    print(f'second-order formula from linear covariance: {max_c:.4g}')
    print(f'common-rate scaling invariance: {abs(scale_a-scale_b):.4g}')
    print(f'repeated temporal eigenvalue, H_4(s=.6,g=.2,phi=.5): {resonant:.12g}')
    assert max_iid < 2e-11 and max_scalar < 1e-13 and max_c < 2e-12
    assert abs(scale_a-scale_b) < 1e-12 and np.isfinite(resonant)
    # Reference from a separate local evaluation of the same finite sum at
    # 40 decimal digits (mpmath); this audits roundoff, not the derivation.
    high_precision_reference = 1.1918790392896263908151976488624974
    precision_error = fourth_order_coefficient(512,2/3,.4,.9)-high_precision_reference
    print(f'40-decimal reference, n=512, phi=.9, NumPy error: {precision_error:.4g}')
    assert abs(precision_error) < 1e-11
    print('Additional limits (numerical checks, not used by the implementation):')
    p, q = .625, .375
    for n in (2,4,16):
        adiabatic = -p*q/8+p*q*(1-2*q)/(4*n)-p*q*(1-6*p*q)/(8*n*n)
        near = fourth_order_coefficient(n,2/3,.4,1-1e-8)
        print(f'  phi -> 1, n={n:2d}: limit={adiabatic:.12g}, error at 1-1e-8={near-adiabatic:.4g}')
    for phi in (0,.5,.9):
        values = [fourth_order_coefficient(n,2/3,.4,phi) for n in (128,256,512)]
        extrapolate = (values[0]-6*values[1]+8*values[2])/3
        limit = q**3/(24*p)*((1+phi)/(1-phi))**2
        print(f'  n -> infinity, phi={phi}: limit={limit:.12g}, extrapolation error={extrapolate-limit:.4g}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=400000)
    parser.add_argument('--replicas',type=int,default=128)
    parser.add_argument('--seed',type=int,default=20261114)
    parser.add_argument('--checks-only',action='store_true')
    args = parser.parse_args()
    exact_checks()
    if args.checks_only:
        return
    print(f'\nINDEPENDENT TAYLOR SIMULATION: steps={args.steps}, replicas={args.replicas}, seed={args.seed}',flush=True)
    print(' phi   n       exact H       estimate H       SE(H)      z-score       C estimate       SE(C)     burn   seconds',flush=True)
    for job,(phi,n) in enumerate([(0.9,n) for n in (2,4,8,16,32)]+[(0.5,n) for n in (4,16)]):
        result = simulate(n,2/3,.4,phi,args.steps,args.replicas,args.seed+1009*job)
        exact = fourth_order_coefficient(n,2/3,.4,phi)
        z = (result['H']-exact)/result['se']
        print(f"{phi:4.1f} {n:3d} {exact:14.10g} {result['H']:16.10g} {result['se']:11.4g} {z:12.3f} {result['C']:16.10g} {result['cse']:11.4g} {result['burn']:8d} {result['seconds']:9.1f}",flush=True)
        print(f"  second-half minus first-half H: {result['drift']:.5g} +/- {result['drift_se']:.5g} (one SE)",flush=True)


if __name__ == '__main__':
    main()
