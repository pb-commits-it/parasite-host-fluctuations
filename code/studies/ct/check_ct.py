#!/usr/bin/env python3
"""Reproduce algebra checks and independent-path simulation of the CT ring.

Run: python3 check_ct.py
Requires Python, numpy, scipy, and a C++17 compiler (CXX or clang++/g++).
The generated simulator and all outputs stay in this directory.  No external
data or exact-model solution is used.  Run --help for smaller pilot runs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.linalg import eig

from ct import cost_ct, cycle_parameters


CPP = r'''
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

template<int N> struct Trajectory {
    std::array<double,N> x;
    long double growth = 0, control = 0;
    Trajectory() { x.fill(1.0/N); }
    void step(const std::array<double,N>& z, double sig, double dt, bool keep) {
        constexpr double q = 1.25;
        std::array<double,N> h, term, y;
        double sh = 0;
        for(int i=0; i<N; ++i) {
            h[i] = std::exp(sig*z[i] - 0.5*sig*sig);
            sh += h[i];
        }
        double log_increment;
        if constexpr (N == 1) {
            log_increment = dt*q*h[0];
        } else {
            // exp(dt*q*Hshift)x through order 8. The scalar -d*I is
            // treated exactly outside this calculation; it cancels in cost.
            term = x; y = x;
            for(int k=1; k<=8; ++k) {
                std::array<double,N> next;
                for(int i=0; i<N; ++i) {
                    const int prev = (i+N-1)%N;
                    next[i] = (dt*q/k)*h[prev]*term[prev];
                    y[i] += next[i];
                }
                term = next;
            }
            double sum = 0;
            for(double yi: y) sum += yi;
            log_increment = std::log(sum);
            for(int i=0; i<N; ++i) x[i] = y[i]/sum;
        }
        if(keep) {
            growth += log_increment;
            control += dt*q*(sh/N-1.0);
        }
    }
};

template<int N> void simulate(double alpha, double sig, double fine_dt,
                              double duration, double burn, int reps,
                              uint64_t seed) {
    const double coarse_dt = 2*fine_dt;
    const int64_t nb = std::llround(burn/coarse_dt);
    const int64_t nt = std::llround(duration/coarse_dt);
    const double T = nt*coarse_dt;
    const double rho = std::exp(-alpha*fine_dt/2);
    const double sd = std::sqrt(-std::expm1(-alpha*fine_dt));
    for(int rep=0; rep<reps; ++rep) {
        std::mt19937_64 rng(seed + UINT64_C(1000003)*rep);
        std::normal_distribution<double> normal;
        std::array<double,N> z;
        for(double& zi: z) zi = normal(rng);
        // [sigma or sigma/2][fine or coarse][+ or - environment]
        Trajectory<N> tr[2][2][2];
        for(int64_t it=0; it<nb+nt; ++it) {
            const bool keep = it>=nb;
            for(int quarter=0; quarter<4; ++quarter) {
                for(double& zi: z) zi = rho*zi + sd*normal(rng);
                if(quarter==3) continue;
                const int grid = quarter==1 ? 1 : 0;
                const double dt = grid ? coarse_dt : fine_dt;
                for(int level=0; level<2; ++level)
                    for(int sign=0; sign<2; ++sign)
                        tr[level][grid][sign].step(z,
                            sig*(level ? 0.5 : 1.0)*(sign ? -1.0 : 1.0),
                            dt, keep);
            }
        }
        std::cout << rep;
        for(int level=0; level<2; ++level) {
            for(int grid=0; grid<2; ++grid) {
                long double g = (tr[level][grid][0].growth+
                                 tr[level][grid][1].growth)/(2*T);
                long double cv = (tr[level][grid][0].control+
                                  tr[level][grid][1].control)/(2*T);
                double raw = 1.25-g;
                double adjusted = 1.25-g+cv;
                std::cout << ' ' << raw << ' ' << adjusted;
            }
        }
        std::cout << '\n' << std::flush;
    }
}

int main(int argc, char** argv) {
    if(argc!=9) return 2;
    int n=std::atoi(argv[1]), reps=std::atoi(argv[7]);
    double alpha=std::atof(argv[2]), sig=std::atof(argv[3]);
    double dt=std::atof(argv[4]), T=std::atof(argv[5]), burn=std::atof(argv[6]);
    uint64_t seed=std::strtoull(argv[8],nullptr,10);
    std::cout << std::setprecision(17);
    if(n==1) simulate<1>(alpha,sig,dt,T,burn,reps,seed);
    else if(n==2) simulate<2>(alpha,sig,dt,T,burn,reps,seed);
    else if(n==4) simulate<4>(alpha,sig,dt,T,burn,reps,seed);
    else return 3;
}
'''


def matrix_coefficient(nu, mu, b, alpha):
    """Independent resolvent evaluation, without using the ring product."""
    n = len(nu)
    A = -np.diag(nu + mu)
    Bs = []
    for k in range(n):
        B = np.zeros((n, n))
        B[(k + 1) % n, k] = b[k] * nu[k]
        A += B
        Bs.append(B)
    vals, left, right = eig(A, left=True, right=True)
    j = np.argmax(vals.real)
    r0 = vals[j].real
    u, v = left[:, j].real, right[:, j].real
    u /= u @ v
    Q = np.eye(n) - np.outer(v, u)
    R = np.linalg.inv((alpha + r0) * np.eye(n) - A)
    return np.array([-u @ B @ R @ Q @ B @ v for B in Bs])


def algebra_checks():
    rng = np.random.default_rng(914701)
    max_error = 0
    count = 0
    for n in (1, 2, 3, 4, 7, 12):
        for _ in range(20):
            nu = np.exp(rng.uniform(-1.5, 1.5, n))
            mu = rng.uniform(0, 1, n)
            b = np.exp(rng.uniform(-1, 1, n))
            alpha = np.exp(rng.uniform(-4, 4))
            closed = cost_ct(nu, mu, b, alpha, 1)
            matrix = matrix_coefficient(nu, mu, b, alpha)
            error = np.max(np.abs(closed - matrix))
            max_error = max(max_error, error)
            np.testing.assert_allclose(closed, matrix, rtol=2e-9, atol=2e-11)
            count += 1
    # These exercise cancellation-prone small-alpha calculations too.
    limit_error = 0
    limit_count = 0
    for n in (2, 3, 4, 10, 100):
        nu, mu, b = np.ones(n), np.full(n, .2), np.full(n, 1.25)
        cycle = cycle_parameters(nu, mu, b)
        slow = (1-cycle.variance/cycle.mean**2)/(2*cycle.mean)
        observed = cost_ct(nu, mu, b, 1e-10, 1)[0]
        limit_error = max(limit_error, abs(observed/slow-1))
        np.testing.assert_allclose(observed, slow, rtol=1e-8)
        fast = 1/(1e9*cycle.mean**2)
        np.testing.assert_allclose(cost_ct(nu, mu, b, 1e9, 1)[0],
                                   fast, rtol=1e-8)
        limit_count += 2
    # A small discrete-time step with staying, computed directly as matrices.
    nu=np.array([.7,1.1,1.8]); mu=np.array([.1,.3,.2]); b=np.array([2.,.8,1.4])
    alpha=.6; n=len(nu)
    A=-np.diag(nu+mu); Bs=[]
    for k in range(n):
        B=np.zeros((n,n)); B[(k+1)%n,k]=nu[k]*b[k]; A+=B; Bs.append(B)
    vals,left,right=eig(A,left=True,right=True); j=np.argmax(vals.real)
    r=vals[j].real; u=left[:,j].real; v=right[:,j].real; u/=u@v
    Q=np.eye(n)-np.outer(v,u)
    target=cost_ct(nu,mu,b,alpha,1)
    discrete=[]
    for dt in (1e-2,1e-3,1e-4,1e-5):
        lam=1+r*dt; P=(np.eye(n)+dt*A)/lam; rho=np.exp(-alpha*dt)
        R=rho*np.linalg.inv(np.eye(n)-rho*P)
        values=[]
        for B in Bs:
            D=dt*B/lam
            values.append((.5*(u@D@v)**2-u@D@R@Q@D@v)/dt)
        discrete.append((dt,float(np.max(np.abs(np.array(values)-target)))))
    assert discrete[-1][1] < discrete[0][1]/500
    return count, max_error, limit_count, limit_error, discrete


def summary(a):
    return float(np.mean(a)), float(np.std(a, ddof=1) / np.sqrt(len(a)))


def midpoint_cost_coefficient(n, alpha, dt, q=1.25):
    """Exact O(sigma^2) coefficient for the frozen-midpoint OU approximation.

    Its phase-averaged covariance linearly interpolates exp(-alpha*t) at
    multiples of dt. Integrate that covariance against the cycle's Fourier
    modes. This calculation measures environment discretisation bias;
    it is not used by the simulator or its control variate.
    """
    if n == 1:
        return 0.0
    roots = np.exp(2j * np.pi * np.arange(1, n) / n)
    beta = q * (1 - roots)
    x = beta * dt
    j0 = -np.expm1(-x) / beta
    j1 = (-np.expm1(-x) - x * np.exp(-x)) / beta**2
    numerator = j0 + np.expm1(-alpha * dt) / dt * j1
    laplace = numerator / -np.expm1(-(alpha + beta) * dt)
    return float((-q**2 / n * np.sum(roots * laplace)).real)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replicates', type=int, default=32)
    p.add_argument('--duration', type=float, default=20000)
    p.add_argument('--burn', type=float, default=200)
    p.add_argument('--dt', type=float, default=.005, help='fine step; coarse is 2*dt')
    p.add_argument('--sigma', type=float, default=.05)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261114)
    p.add_argument('--output', default='results_ct.txt')
    args = p.parse_args()
    if args.replicates < 2 or min(args.duration,args.dt,args.sigma) <= 0 or args.burn < 0:
        p.error('need replicates >= 2, duration/dt/sigma > 0, and burn >= 0')
    start=time.monotonic()
    checks=algebra_checks()
    here=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
    source=here/'.ct_sim.cpp'; binary=here/'.ct_sim'
    source.write_text(CPP)
    compiler=os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        raise RuntimeError('C++17 compiler required for reproducible simulation')
    subprocess.run([compiler,'-O3','-std=c++17',str(source),'-o',str(binary)],check=True)
    cases=[(n,a) for n in (1,2,4) for a in (.2,1.,5.)]

    def run_case(index,n,alpha):
        seed=args.seed+index*100000007
        cmd=[str(binary),str(n),str(alpha),str(args.sigma),str(args.dt),
             str(args.duration),str(args.burn),str(args.replicates),str(seed)]
        result=subprocess.run(cmd,text=True,capture_output=True,check=True)
        data=np.loadtxt(io.StringIO(result.stdout),ndmin=2)
        return index,n,alpha,data

    data_by_case={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,i,n,a) for i,(n,a) in enumerate(cases)]
        for fut in as_completed(futures):
            index,n,alpha,data=fut.result()
            data_by_case[index]=data
            mean,se=summary(data[:,2])
            print(f'finished n={n}, alpha={alpha:g}: adjusted cost={mean:.9g}, SE={se:.3g}',flush=True)

    lines=[
        'CONTINUOUS-TIME RING: INDEPENDENT DERIVATION CHECK',
        'All costs below sum over all n independent hosts; units are inverse time.',
        f'nu=1, mu=0.2, b=1.25 => r0=0.05, q=1.25; sigma={args.sigma:g}.',
        f'{args.replicates} independent replicates per case, duration={args.duration:g}, burn={args.burn:g}.',
        f'Fine dt={args.dt:g}; coarse dt={2*args.dt:g}; exact OU mesh dt={args.dt/2:g}.',
        f'Each replicate uses antithetic +z/-z environments; the replicate is the SE unit.',
        'The main estimate additionally subtracts the known-zero-mean host-average term from growth.',
        'This control variate does not use the predicted cost; raw antithetic estimates follow below.',
        'OU starts in its stationary N(0,1) law; stages start equally and are burned in.',
        'Linear system: exponential propagation with frozen midpoint hosts, Taylor order 8;',
        '-d*I treated exactly. Fine/coarse paths and sigma/sigma/2 paths share OU randomness.',
        'SE = sample standard deviation of independent replicate estimates / sqrt(replicates).',
        'No confidence interval here includes unknown residual time-discretisation bias.',
        f'Seed base={args.seed}; case seed=base+index*100000007; replicate increment=1000003.',
        '',
        'MAIN RESULTS (fine step, control variate; ratio undefined for n=1):',
        ' n alpha   simulated_cost     SE              formula            ratio     ratio_SE',
    ]
    for i,(n,alpha) in enumerate(cases):
        data=data_by_case[i]
        formula=cost_ct(np.ones(n),np.full(n,.2),np.full(n,1.25),alpha,args.sigma).sum()
        mean,se=summary(data[:,2])
        ratio=f'{mean/formula:9.6f} {se/formula:10.6f}' if formula else '      N/A        N/A'
        lines.append(f'{n:2d} {alpha:5.1f} {mean: .10e} {se:.6e} {formula:.10e} {ratio}')
    lines += ['', 'RAW ANTITHETIC COSTS (same fine simulations, no host control variate):',
              ' n alpha   raw_cost           SE']
    for i,(n,alpha) in enumerate(cases):
        mean,se=summary(data_by_case[i][:,1])
        lines.append(f'{n:2d} {alpha:5.1f} {mean: .10e} {se:.6e}')
    lines += ['', 'TIME DISCRETISATION: paired coarse-minus-fine adjusted cost:',
              ' n alpha   coarse_cost        fine_cost          difference         paired_SE      difference/formula']
    for i,(n,alpha) in enumerate(cases):
        data=data_by_case[i]
        coarse=np.mean(data[:,4]); fine=np.mean(data[:,2])
        diff,se=summary(data[:,4]-data[:,2])
        formula=cost_ct(np.ones(n),np.full(n,.2),np.full(n,1.25),alpha,args.sigma).sum()
        rel=f'{diff/formula:.6e}' if formula else 'N/A'
        lines.append(f'{n:2d} {alpha:5.1f} {coarse: .10e} {fine: .10e} {diff: .6e} {se:.6e} {rel}')
    lines += ['', 'ANALYTICAL O(sigma^2) MIDPOINT DISCRETISATION BIAS (derived independently):',
              ' n alpha   fine_bias/formula  predicted_coarse_minus_fine']
    for n,alpha in cases:
        if n == 1:
            continue
        formula=cost_ct(np.ones(n),np.full(n,.2),np.full(n,1.25),alpha,args.sigma).sum()
        fine=midpoint_cost_coefficient(n,alpha,args.dt)*args.sigma**2
        coarse=midpoint_cost_coefficient(n,alpha,2*args.dt)*args.sigma**2
        lines.append(f'{n:2d} {alpha:5.1f} {fine/formula-1:.6e} {coarse-fine:.6e}')
    lines.append('This is a bias calculation for the second-order coefficient, not an all-orders bias bound.')
    lines += ['', 'SMALL-NOISE CHECK: sigma/2 fine results, and paired change of cost/sigma^2:',
              ' n alpha   half_sigma_cost    SE              ratio     ratio_SE   C(sigma)/sigma^2 - C(sigma/2)/(sigma/2)^2   paired_SE']
    for i,(n,alpha) in enumerate(cases):
        data=data_by_case[i]
        formula=cost_ct(np.ones(n),np.full(n,.2),np.full(n,1.25),alpha,args.sigma/2).sum()
        mean,se=summary(data[:,6])
        delta,delta_se=summary((data[:,2]-4*data[:,6])/args.sigma**2)
        ratio=f'{mean/formula:9.6f} {se/formula:10.6f}' if formula else '      N/A        N/A'
        lines.append(f'{n:2d} {alpha:5.1f} {mean: .10e} {se:.6e} {ratio} {delta: .6e} {delta_se:.6e}')
    lines += ['', 'ALGEBRA CHECKS (numerical, not simulations):',
              f'{checks[0]} heterogeneous matrix-resolvent comparisons; max absolute discrepancy={checks[1]:.6e}.',
              f'{checks[2]} fast/slow limit comparisons; max slow-limit relative discrepancy={checks[3]:.6e}.',
              'Discrete-to-continuous check (one heterogeneous 3-stage model, four time steps):']
    lines += [f'  dt={dt:.1e}: max per-host coefficient discrepancy={error:.6e}' for dt,error in checks[4]]
    lines += ['',f'Elapsed wall time: {time.monotonic()-start:.1f} seconds.',
              f'Compiler: {compiler}',f'numpy: {np.__version__}',
              '', 'RAW REPLICATE DATA:',
              '# n alpha rep raw_fine cv_fine raw_coarse cv_coarse raw_half_fine cv_half_fine raw_half_coarse cv_half_coarse']
    for i,(n,alpha) in enumerate(cases):
        for row in data_by_case[i]:
            lines.append(f'{n} {alpha:g} {int(row[0])} '+' '.join(f'{x:.17g}' for x in row[1:]))
    output=here/args.output
    output.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:lines.index('RAW REPLICATE DATA:')]),flush=True)
    print(f'Wrote {output}',flush=True)


if __name__=='__main__':
    main()
