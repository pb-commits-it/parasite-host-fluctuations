#!/usr/bin/env python3
"""Check GAP 2 using the nonlinear invariant operator, independently of its moment proof.

Claim: log(1/s)*(ell-D) -> atanh(delta)^2*(1-phi)/(2*(1+phi)).
Protocol: protocol.md, registered before running this program.
No Monte Carlo. mpmath builds a piecewise Chebyshev collocation operator.
A sparse double LU is ONLY a preconditioner: residuals and corrections are
computed in arbitrary precision until the full mp linear system is solved.
This controls arithmetic error, not spatial discretization error.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()


def softplus(x):
    return max(x, mp.mpf(0)) + mp.log1p(mp.exp(-abs(x)))


def invariant(L, delta, phi, degree=12, width=2, dps=45):
    """Return E Q and diagnostics for normalized X'=log(exp(-L)+exp(bz-X)).

    Each panel uses degree+1 Chebyshev-Lobatto nodes. Shared endpoints are
    identified. K interpolates observables; its invariant left vector is a
    quadrature rule, possibly signed, rather than an asserted positive density.
    """
    with mp.workdps(dps):
        L, delta, phi = map(mp.mpf, (L, delta, phi))
        b = mp.atanh(delta)
        lo = -L
        hi = softplus(2*L+b)-L
        panels = max(1, int(mp.ceil((hi-lo)/width)))
        step = (hi-lo)/panels
        cn = [-mp.cos(mp.pi*j/degree) for j in range(degree+1)]
        bw = [mp.mpf((-1)**j)*(mp.mpf('.5') if j in (0, degree) else 1)
              for j in range(degree+1)]
        nodes = [lo+step*(p+(cn[j]+1)/2)
                 for p in range(panels) for j in range(degree)] + [hi]
        n = len(nodes)
        size = 2*n
        # Sparse rows of K and exact conditional observables.
        kr = []
        rewards = []
        moment = []
        raw2 = []
        for i in range(2):
            for x in nodes:
                row = {}
                rq = mp.mpf(0)
                rm = mp.mpf(0)
                r2 = mp.mpf(0)
                for j, z in enumerate((-1, 1)):
                    prob = (1+phi if i == j else 1-phi)/2
                    eta = b*z
                    q = softplus(x-eta-L)
                    target = -x+eta+q
                    p = min(panels-1, max(0, int(mp.floor((target-lo)/step))))
                    xx = 2*((target-lo)/step-p)-1
                    dif = [xx-t for t in cn]
                    hit = next((k for k, v in enumerate(dif)
                                if abs(v) < mp.power(10, -dps+8)), None)
                    if hit is not None:
                        vals = [mp.mpf(int(k == hit)) for k in range(degree+1)]
                    else:
                        vals = [bw[k]/dif[k] for k in range(degree+1)]
                        total = mp.fsum(vals)
                        vals = [v/total for v in vals]
                    for k, v in enumerate(vals):
                        col = j*n+p*degree+k
                        row[col] = row.get(col, mp.mpf(0))+prob*v
                    rq += prob*q
                    rm += prob*(2*x-2*eta/(1+phi)-q)*q
                    r2 += prob*(target**2-x**2)
                kr.append(row)
                rewards.append(rq)
                moment.append(rm)
                raw2.append(r2)
        # A = K^T-I, replacing its last equation by sum(w)=1.
        ar = [{} for _ in range(size)]
        for i, row in enumerate(kr):
            for j, v in row.items():
                if j != size-1:
                    ar[j][i] = v
        for i in range(size-1):
            ar[i][i] = ar[i].get(i, mp.mpf(0))-1
        ar[-1] = {i: mp.mpf(1) for i in range(size)}
        rr, cc, vv = [], [], []
        for i, row in enumerate(ar):
            for j, v in row.items():
                rr.append(i)
                cc.append(j)
                vv.append(float(v))
        lu = splu(coo_matrix((vv, (rr, cc)), shape=(size, size)).tocsc())
        rhs = np.zeros(size)
        rhs[-1] = 1
        w = [mp.mpf(float(v)) for v in lu.solve(rhs)]
        residual = mp.inf
        for iteration in range(9):
            err = [(1 if i == size-1 else 0) -
                   mp.fsum(v*w[j] for j, v in row.items())
                   for i, row in enumerate(ar)]
            residual = max(map(abs, err))
            if residual < mp.mpf('1e-35'):
                break
            scale = max(map(abs, err))
            corr = lu.solve(np.array([float(e/scale) for e in err]))
            w = [v+scale*mp.mpf(float(d)) for v, d in zip(w, corr)]
        if residual >= mp.mpf('1e-30'):
            raise ArithmeticError(f'linear residual {residual}')
        eq = mp.fsum(v*q for v, q in zip(w, rewards))
        ex = mp.fsum(w[i]*nodes[i % n] for i in range(size))
        em = mp.fsum(v*q for v, q in zip(w, moment))
        er2 = mp.fsum(v*q for v, q in zip(w, raw2))
        vpi = b*b*(1-phi)/(1+phi)
        return dict(q=+eq, first=abs(2*ex-eq), second=abs(em-vpi),
                    square=abs(er2), linear=+residual, size=size,
                    min_weight=min(w), total_variation=mp.fsum(map(abs,w)),
                    state_mass_error=abs(mp.fsum(w[:n])-mp.mpf('.5')))


def threshold(s, delta, phi, degree=12, width=2, dps=45):
    """Solve the original growth-root equation in the normalized L coordinate."""
    with mp.workdps(dps):
        s, delta, phi = map(mp.mpf, (s, delta, phi))
        L0 = -mp.log(s)
        ell = -mp.log1p(-delta*delta)/2
        cache = {}
        def fun(L):
            key = mp.nstr(L, dps)
            if key not in cache:
                cache[key] = invariant(L, delta, phi, degree, width, dps)
            return L+cache[key]['q']/2-L0
        # No coefficient from the proposed asymptotic enters the solver.
        root = mp.findroot(fun, (L0, L0-mp.mpf('.01')),
                           solver='secant', tol=mp.mpf('1e-32'), maxsteps=12)
        data = invariant(root, delta, phi, degree, width, dps)
        data.update(D=ell-data['q']-mp.log1p(-s), L=root,
                    root=abs(root+data['q']/2-L0), evaluations=len(cache)+1)
        return data


def fmt(x, digits=18):
    return mp.nstr(x, digits)


def pilot():
    for s, delta, phi in [('.5','.4','.7'), ('1e-12','.5','0'), ('1e-30','.5','.5')]:
        for degree, width in [(10, 2), (14, 2), (14, 1)]:
            start = time.monotonic()
            v = threshold(s, delta, phi, degree, width)
            print('PILOT', s, delta, phi, degree, width,
                  {k: fmt(x) for k, x in v.items()},
                  'seconds', round(time.monotonic()-start, 3), flush=True)


def main():
    mp.mp.dps = 50
    started = time.monotonic()
    passed, failed = [], []
    def check(label, condition, detail=''):
        (passed if condition else failed).append(label)
        print(('PASS ' if condition else 'FAIL ')+label+' '+str(detail), flush=True)
    print('GAP 2 deterministic checks; protocol.md registered before runs', flush=True)
    print('45 decimal digits; coarse degree=12,width=2; fine degree=16,width=1')
    print('Sparse float64 LU only preconditions mp residual refinement.')
    print('Interpolation errors are assessed by refinement, NOT interval-certified.')
    for delta, phi, expected, tol, label in [
        ('0','.3','0','1e-12','constant host known threshold'),
        ('.4','.7','.0582995785286115578','2e-10','memA known nonlinear threshold')]:
        v = threshold('.5', delta, phi, 16, 1, 45)
        discrepancy = abs(v['D']-mp.mpf(expected))
        check(label, discrepancy < mp.mpf(tol), fmt(discrepancy))
    if failed:
        raise SystemExit('Known-result validation failed; primary grid not run.')

    grid = [('.5', phi, k) for phi in ('-.5','0','.5')
            for k in (4,8,12,18,24,30,40)]
    grid += [(delta,'0',k) for delta in ('.25','.8') for k in (12,30,40)]
    records = []
    csv_path = HERE/'gap2_results.csv'
    json_path = HERE/'gap2_results.json'
    for delta, phi, k in grid:
        cell_start = time.monotonic()
        s = mp.power(10, -k)
        coarse = threshold(s, delta, phi, 12, 2, 45)
        fine = threshold(s, delta, phi, 16, 1, 45)
        dd, pp = mp.mpf(delta), mp.mpf(phi)
        b = mp.atanh(dd)
        ell = -mp.log1p(-dd*dd)/2
        A = b*b*(1-pp)/(2*(1+pp))
        gap = ell-fine['D']
        L0 = -mp.log(s)
        refinement = abs(coarse['D']-fine['D'])
        label = f'delta={delta} phi={phi} k={k}'
        check('mesh '+label, refinement <= mp.mpf('2e-8'), fmt(refinement, 6))
        check('arithmetic '+label,
              fine['linear'] <= mp.mpf('1e-30') and fine['root'] <= mp.mpf('1e-25'),
              'linear='+fmt(fine['linear'],4)+' root='+fmt(fine['root'],4))
        check('moments '+label,
              max(fine['first'],fine['second']) <= mp.mpf('2e-8'),
              'first='+fmt(fine['first'],4)+' second='+fmt(fine['second'],4))
        # Explicit proved bounds (8)--(9), using the numerical q and L.
        L = fine['L']
        M = 2*mp.log(L)
        cp = 2*(b+1)+2*b/(1+pp)
        cm = 2*b/(1+pp)+softplus(2*b+1)
        lower = 2*A/(2*L+cp)
        upper = (2*A+2*M*mp.exp(b-M))/(2*L-2*M-cm)
        if M >= 1 and 2*L-2*M-cm > 0:
            check('proved bounds '+label, lower <= fine['q'] <= upper,
                  fmt(lower,8)+' <= '+fmt(fine['q'],8)+' <= '+fmt(upper,8))
        record = dict(delta=delta, phi=phi, k=k, D=fmt(fine['D'],32),
                      ell=fmt(ell,32), gap=fmt(gap,32), A=fmt(A,32),
                      scaled=fmt(L0*gap/A,24), mesh_difference=fmt(refinement,12),
                      linear_residual=fmt(fine['linear'],12),
                      root_residual=fmt(fine['root'],12),
                      first_residual=fmt(fine['first'],12),
                      second_residual=fmt(fine['second'],12),
                      dimension=fine['size'], min_weight=fmt(fine['min_weight'],12),
                      total_variation=fmt(fine['total_variation'],12),
                      seconds=round(time.monotonic()-cell_start,3))
        records.append(record)
        print('RESULT '+json.dumps(record), flush=True)
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        json_path.write_text(json.dumps(records, indent=2)+'\n')

    # Same discretization, independent arithmetic precision.
    for delta, phi, k in [('.5','0',12),('.5','.5',30),('.8','0',40)]:
        high = threshold(mp.power(10,-k), delta, phi, 16, 1, 65)
        row = next(r for r in records if (r['delta'],r['phi'],r['k']) == (delta,phi,k))
        difference = abs(high['D']-mp.mpf(row['D']))
        check(f'45/65 digit comparison delta={delta} phi={phi} k={k}',
              difference <= mp.mpf('1e-25'), fmt(difference,12))

    print('\nFits: train k>=12 and k<40, hold out k=40. Errors on ell-D scale.')
    fits = []
    for delta, phi in [('.5','-.5'),('.5','0'),('.5','.5'),('.25','0'),('.8','0')]:
        rows = [r for r in records if r['delta']==delta and r['phi']==phi and r['k']>=12]
        ks = np.array([r['k'] for r in rows],dtype=float)
        Ls = ks*np.log(10.)
        ys = np.array([float(r['gap']) for r in rows])
        train = ks<40
        A = float(rows[0]['A'])
        design = np.column_stack((1/Ls,1/Ls**2))
        coef = np.linalg.lstsq(design[train],ys[train],rcond=None)[0]
        pred = design@coef
        # Competing inverse-square law, with its optimal free coefficient.
        zz = 1/Ls**2
        altcoef = np.dot(zz[train],ys[train])/np.dot(zz[train],zz[train])
        alt = altcoef*zz
        # Competing s^alpha law, both amplitude and exponent fitted in log space.
        pc = np.linalg.lstsq(np.column_stack((np.ones(sum(train)),-Ls[train])),
                            np.log(ys[train]),rcond=None)[0]
        power = np.exp(pc[0]-pc[1]*Ls)
        result = dict(delta=delta,phi=phi,A=A,a=float(coef[0]),d=float(coef[1]),
                      a_over_A=float(coef[0]/A),
                      fixed_leading_rmse=float(np.sqrt(np.mean((A/Ls-ys)**2))),
                      inverse_log_rmse=float(np.sqrt(np.mean((pred-ys)**2))),
                      inverse_square_rmse=float(np.sqrt(np.mean((alt-ys)**2))),
                      power_rmse=float(np.sqrt(np.mean((power-ys)**2))),
                      inverse_log_holdout=float(abs(pred[-1]-ys[-1])),
                      inverse_square_holdout=float(abs(alt[-1]-ys[-1])),
                      power_holdout=float(abs(power[-1]-ys[-1])),
                      fitted_power_alpha=float(pc[1]))
        fits.append(result)
        print('FIT '+json.dumps(result),flush=True)
        check(f'coefficient delta={delta} phi={phi}', abs(coef[0]/A-1)<=.05,
              'a/A='+str(coef[0]/A))
    (HERE/'gap2_fits.json').write_text(json.dumps(fits,indent=2)+'\n')
    print('\nCAUTION: polynomial moment residuals are strong algebra/index checks,')
    print('but collocation reproduces low-degree polynomials; these residuals do')
    print('not independently bound error in the invariant law. Mesh checks do not certify it.')
    print('SUMMARY',len(records),'threshold cells;',len(passed),'passed;',len(failed),
          'failed; elapsed seconds',round(time.monotonic()-started,3),flush=True)
    if failed:
        print('FAILURES',failed)
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.pilot:
        pilot()
    else:
        main()
