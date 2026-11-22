#!/usr/bin/env python3
"""Independent direct simulations and algebra checks; run with python3 check_cts.py.

All temporary build/input/output files are created below this directory.
The C++ kernel advances the original linear random ODE after an exact
diagonal similarity transform, using exact joint OU endpoints/integrals
and Strang splitting.  No perturbation formula is used in the simulation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from scipy.linalg import expm

from cts import cost_death, lineage_parameters


CASES = [
    dict(name="n1_slow", nu=[.4], mu=[.5], b=[2.5], alpha=.1, j=0),
    dict(name="n1_fast", nu=[.4], mu=[.5], b=[2.5], alpha=10., j=0),
    dict(name="n2_slow", nu=[.4]*2, mu=[.5]*2, b=[2.5]*2, alpha=.1, j=0),
    dict(name="n2_fast", nu=[.4]*2, mu=[.5]*2, b=[2.5]*2, alpha=10., j=0),
    dict(name="n3_slow", nu=[.4]*3, mu=[.5]*3, b=[2.5]*3, alpha=.1, j=1),
    dict(name="n3_fast", nu=[.4]*3, mu=[.5]*3, b=[2.5]*3, alpha=10., j=1),
    dict(name="n3_unequal_slow", nu=[.2,.5,1.1], mu=[.25,.75,1.45],
         b=[6.,1.4,2.7/1.1], alpha=.2, j=0),
    dict(name="n3_unequal_fast", nu=[.2,.5,1.1], mu=[.25,.75,1.45],
         b=[6.,1.4,2.7/1.1], alpha=5., j=2),
]


CPP = r'''
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
using V = std::array<double,3>;
using M = std::array<double,9>;

V multiply(const M& m, const V& y, int n) {
    V out{};
    for(int k=0;k<n;k++) for(int l=0;l<n;l++) out[k]+=m[3*k+l]*y[l];
    return out;
}

// A direct, mass-normalized Strang step.  L is a column Markov generator.
double step(V& y, const M& e, int n, int j, double s, double integral) {
    if(n==1) return -s*integral;
    V mid=multiply(e,y,n);
    double total=0.; for(int k=0;k<n;k++) total+=mid[k];
    for(int k=0;k<n;k++) mid[k]/=total;
    double change=std::expm1(-s*integral);
    double delta=mid[j]*change;
    double mass=1.+delta;
    mid[j]*=1.+change;
    for(int k=0;k<n;k++) mid[k]/=mass;
    y=multiply(e,mid,n);
    total=0.; for(int k=0;k<n;k++) total+=y[k];
    for(int k=0;k<n;k++) y[k]/=total;
    return std::log1p(delta);
}

int main(int argc,char** argv) {
    std::ifstream in(argv[1]);
    int n,j,reps; double alpha,sigma,h,T,burn; uint64_t seed;
    in>>n>>j>>reps>>alpha>>sigma>>h>>T>>burn>>seed;
    V p{}; for(int k=0;k<n;k++) in>>p[k];
    M E[3]{};
    for(int q=0;q<3;q++) for(int k=0;k<n;k++) for(int l=0;l<n;l++) in>>E[q][3*k+l];
    const int stride[3]={4,2,1};
    const double x=alpha*h, rho=std::exp(-x), sd=std::sqrt(-std::expm1(-2*x));
    const double factor=std::tanh(x/2)/alpha;
    const double residual=x<.01 ? x*x*x*(1./12-x*x/120+17*x*x*x*x/20160) : x-2*std::tanh(x/2);
    const double bridge_sd=std::sqrt(2*residual)/alpha;
    int64_t nb=4*int64_t(std::ceil(burn/(4*h)));
    int64_t nt=4*int64_t(std::ceil(T/(4*h)));
    T=nt*h;
    std::cout<<std::setprecision(17);
    for(int rep=0;rep<reps;rep++) {
        std::mt19937_64 rng(seed+uint64_t(rep)*0x9e3779b97f4a7c15ULL);
        std::normal_distribution<double> normal(0.,1.);
        double z=normal(rng);
        V y[3][4]; double logs[3][4]{}; double ints[3]{};
        for(int q=0;q<3;q++) for(int amp=0;amp<4;amp++) y[q][amp]=p;
        for(int64_t k=0;k<nb+nt;k++) {
            double next=rho*z+sd*normal(rng);
            double I=factor*(z+next)+bridge_sd*normal(rng);
            z=next;
            for(int q=0;q<3;q++) {
                ints[q]+=I;
                if((k+1)%stride[q]) continue;
                for(int amp=0;amp<4;amp++) {
                    double s=sigma*(amp<2 ? 1. : .5)*(amp%2 ? -1. : 1.);
                    double increment=step(y[q][amp],E[q],n,j,s,ints[q]);
                    if(k>=nb) logs[q][amp]+=increment;
                }
                ints[q]=0.;
            }
        }
        for(int q=0;q<3;q++) for(int amp=0;amp<2;amp++)
            std::cout<<-(logs[q][2*amp]+logs[q][2*amp+1])/(2*T)<<" ";
        std::cout<<"\n";
    }
}
'''


def stats(x):
    return float(np.mean(x)), float(np.std(x, ddof=1)/np.sqrt(len(x)))


def algebra_checks(count=200):
    """Check independently against a matrix resolvent and Perron eigensystem."""
    rng = np.random.default_rng(15071987)
    err = 0.0
    for trial in range(count):
        n = int(rng.integers(2, 13))
        nu, mu, b = np.exp(rng.uniform(-1, 1, (3, n)))
        alpha = float(np.exp(rng.uniform(-2, 3)))
        j = int(rng.integers(n))
        r0, a, p = lineage_parameters(nu, mu, b)
        A = -np.diag(nu+mu)
        for k in range(n):
            A[(k+1)%n,k] += b[k]*nu[k]
        values, vectors = np.linalg.eig(A)
        index = np.argmax(values.real)
        assert abs(values[index].real-r0) < 1e-10
        v = vectors[:,index].real
        wvalues, wvectors = np.linalg.eig(A.T)
        w = wvectors[:,np.argmin(abs(wvalues-r0))].real
        w /= w@v
        assert np.max(abs(w*v-p)) < 1e-10
        B = np.zeros((n,n)); B[j,j] = -1
        Q = np.eye(n)-np.outer(v,w)
        gain = w@B@np.linalg.solve(alpha*np.eye(n)-A+r0*np.eye(n), Q@B@v)
        formula = -cost_death(nu,mu,b,alpha,1.,j)
        err = max(err, abs(formula-gain)/formula)
        assert formula > 0 and np.isclose(formula,gain,rtol=1e-9,atol=1e-12)
    return err


def equal_loss_checks(count=1000):
    """Protect the root bracket when the analytic upper bound is the root."""
    rng = np.random.default_rng(420)
    error = 0.0
    for trial in range(count):
        n = int(rng.integers(2,16))
        nu = np.full(n,.4); mu = np.full(n,.6)
        b = np.exp(rng.uniform(-2,2,n))
        r0,a,p = lineage_parameters(nu,mu,b)
        expected = np.exp(np.mean(np.log(nu)+np.log(b)))
        error = max(error,abs(a[0]/expected-1))
        assert np.allclose(a,expected,rtol=1e-11,atol=0.)
        assert abs(r0-(expected-1)) < 1e-11
        assert np.allclose(p,1./n,rtol=1e-11,atol=0.)
    return error


def run_case(case, index, args, temp, exe):
    r0,a,p = lineage_parameters(case['nu'],case['mu'],case['b'])
    n=len(a)
    L=-np.diag(a)
    for k in range(n):
        L[(k+1)%n,k] += a[k]
    dt=args.dt
    burn=max(100.,30./case['alpha'],30./min(a))
    seed=2718281828+index*1000003
    fields=[n,case['j'],args.replicates,case['alpha'],args.sigma,dt/4,args.time,burn,seed]
    lines=[' '.join(map(str,fields)), ' '.join(map(str,p))]
    for h in [dt,dt/2,dt/4]:
        e=expm(L*h/2)
        e/=np.sum(e,axis=0)[None,:]
        lines.append(' '.join(format(v,'.17g') for v in e.flat))
    input_file=temp/(case['name']+'.in')
    input_file.write_text('\n'.join(lines)+'\n')
    proc=subprocess.run([str(exe),str(input_file)],check=True,text=True,capture_output=True)
    samples=np.array([[float(v) for v in line.split()] for line in proc.stdout.splitlines()])
    assert samples.shape == (args.replicates,6)
    predicted=cost_death(case['nu'],case['mu'],case['b'],case['alpha'],args.sigma,case['j'])
    return dict(case=case,r0=r0,a=a,p=p,burn=burn,seed=seed,samples=samples,predicted=predicted)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates',type=int,default=32)
    parser.add_argument('--time',type=float,default=12000.)
    parser.add_argument('--dt',type=float,default=.02,help='coarsest step; also simulate dt/2 and dt/4')
    parser.add_argument('--sigma',type=float,default=.04)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',default='results_cts.txt')
    args=parser.parse_args()
    if args.replicates<2 or min(args.time,args.dt,args.sigma)<=0:
        parser.error('need at least two replicates and positive time, dt, sigma')
    started=time.monotonic()
    error=algebra_checks()
    root_error=equal_loss_checks()
    directory=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
    with tempfile.TemporaryDirectory(prefix='.cts_build_',dir=directory) as tmp:
        temp=Path(tmp)
        source=temp/'kernel.cpp'; source.write_text(CPP)
        exe=temp/'kernel'
        env=dict(os.environ,TMPDIR=str(temp))
        subprocess.run(['c++','-O3','-std=c++17',str(source),'-o',str(exe)],check=True,env=env)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_case,c,i,args,temp,exe) for i,c in enumerate(CASES)]
            rows=[]
            for future in futures:
                result=future.result(); rows.append(result)
                print('Completed',result['case']['name'],flush=True)

    out=[]
    def write(s=''): out.append(s)
    write('CHECKED NUMERICALLY HERE: 8 direct-simulation cases; 200 randomized algebra cases; 1000 equal-loss-rate root cases.')
    write('The prediction is DERIVED HERE; standard errors are between independent replicate pairs.')
    write(f'Randomized matrix/Perron checks: 200 passed; maximum relative coefficient error = {error:.3e}')
    write(f'Equal-loss-rate root checks: 1000 passed; maximum relative a error = {root_error:.3e}')
    write(f'Command: python3 check_cts.py --replicates {args.replicates} --time {args.time:g} --dt {args.dt:g} --sigma {args.sigma:g} --workers {args.workers}')
    write(f'sigma={args.sigma:g}; sigma/2={args.sigma/2:g}; independent pairs per case={args.replicates}; measurement T={args.time:g}')
    write(f'Three coupled step sizes: {args.dt:g}, {args.dt/2:g}, {args.dt/4:g}; table uses finest step.')
    write('Each pair averages direct growth costs for +z and -z; ratio = simulated/predicted.')
    write('For n=1 both costs are exactly zero and the ratio is undefined (0/0).')
    write()
    write('case                  n alpha j    simulated_cost       SE             predicted_cost       ratio')
    for row in rows:
        c=row['case']; m,se=stats(row['samples'][:,4]); pred=row['predicted']
        ratio=f'{m/pred:.7f}' if pred else 'undefined'
        write(f"{c['name']:22s} {len(c['nu']):1d} {c['alpha']:5g} {c['j']:1d} {m: .10e} {se:.4e} {pred: .10e} {ratio}")
    write()
    write('TIME-STEP CHECK (8 cases): coupled mean cost differences and their SE; compare against main-table SE.')
    write('case                    coarse-minus-middle (SE)        middle-minus-fine (SE)       latter/main_SE')
    for row in rows:
        s=row['samples']; a,ae=stats(s[:,0]-s[:,2]); b,be=stats(s[:,2]-s[:,4]); _,se=stats(s[:,4])
        rel=f'{abs(b)/se:.5f}' if se else 'undefined'
        write(f"{row['case']['name']:22s} {a: .6e} ({ae:.2e}) {b: .6e} ({be:.2e}) {rel}")
    write()
    write('AMPLITUDE CHECK (8 cases): fine-step cost(sigma) - 4*cost(sigma/2), with paired SE.')
    write('A second-order law predicts zero; these differences also measure finite-sigma corrections.')
    write('case                    difference       SE         fraction_of_prediction   half_sigma_ratio')
    for row in rows:
        s=row['samples']; m,se=stats(s[:,4]-4*s[:,5]); pred=row['predicted']
        rel=f'{m/pred:.6g}' if pred else 'undefined'
        hr=f'{4*np.mean(s[:,5])/pred:.7f}' if pred else 'undefined'
        write(f"{row['case']['name']:22s} {m: .6e} {se:.3e} {rel:>20s} {hr:>17s}")
    write()
    write('INPUTS AND LINEAGE PARAMETERS (8 cases; derived parameters evaluated numerically):')
    for row in rows:
        c=row['case']
        write(f"{c['name']}: nu={c['nu']}; mu={c['mu']}; b={c['b']}; alpha={c['alpha']}; j={c['j']}")
        write(f"  r0={row['r0']:.12g}; a={row['a'].tolist()}; p={row['p'].tolist()}; burn={row['burn']:g}; seed={row['seed']}")
    write()
    write('REPLICATE COSTS, for independent reanalysis (columns: dt sigma, dt sigma/2, dt/2 sigma, dt/2 sigma/2, dt/4 sigma, dt/4 sigma/2):')
    for row in rows:
        write(row['case']['name'])
        for values in row['samples']:
            write(' '.join(f'{v:.16e}' for v in values))
    write()
    write(f'Elapsed wall time: {time.monotonic()-started:.1f} seconds.')
    report='\n'.join(out)+'\n'
    path=directory/args.output
    path.write_text(report)
    print(report.split('REPLICATE COSTS')[0],flush=True)
    print('Wrote',path,flush=True)


if __name__=='__main__':
    main()
