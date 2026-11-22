"""Claim: finite algebraic threshold coefficients and the iid gamma continuation cores.
Tolerances and scope were fixed in plan.md before execution. No simulations,
collaborator imports, or claims of interval-certified arithmetic. See the report
for which formal series have a proved interpretation for the physical threshold.
"""
from decimal import Decimal as D, getcontext
from fractions import Fraction
from math import comb, factorial
from pathlib import Path
import argparse, json, time
import mpmath as mp

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ZERO = D(0)
ONE = D(1)

def exp_series(x, N):
    out = [x[0]*0+1] + [x[0]*0]*N
    for n in range(1,N+1):
        out[n] = sum(j*x[j]*out[n-j] for j in range(1,n+1))/n
    return out

def log_series(x, N):
    # x[0]=1; x'=x (log x)'.
    out=[x[0]*0]*(N+1)
    for n in range(1,N+1):
        out[n]=x[n]-sum((j*out[j]*x[n-j] for j in range(1,n)),x[0]*0)/n
    return out

def gamma_stein(sv, K, prec=120, rational=False):
    """Independent iid gamma route: integration by parts of Roerdink's density.
    No stochastic moment-transition equation is used in this implementation.
    C=c/(1-s), M[k,n]=[v^n] E[(U-1)^k], with E log U=0.
    """
    getcontext().prec=prec
    T=Fraction if rational else D
    s=T(sv); a=1-s; z=T(0)
    M=[[z]*(2*K+3) for _ in range(K+1)]; M[0][0]=T(1)
    C=[T(1)]+[z]*K
    for n in range(1,K+1):
        for k in range(2*n-1,0,-1):
            rhs=a*sum(C[l]*M[n-l][k] for l in range(1,n+1))
            rhs+=a/s*sum(C[l]*((k+1)*M[n-1-l][k+1]+(a+k*(1+a))*M[n-1-l][k]+k*a*M[n-1-l][k-1]) for l in range(n))
            M[n][k+1]=(rhs-M[n][k+2])/(1+a)
        M[n][1]=sum((-1)**k*M[n][k]/k for k in range(2,2*n+1))
        C[n]=((1+a)*M[n][1]+M[n][2])/a-sum(C[l]*((a if n-1-l==0 else z)+M[n-1-l][1]) for l in range(n))/s
    return log_series(C,K)[1:]

def iid_moments(sv, host, K, prec=120, rational=False):
    """iid stationary moment equation, using raw moments of h. Formal exact recursion."""
    getcontext().prec=prec
    T=Fraction if rational else D
    s=T(sv); a=1-s; zero=T(0); one=T(1)
    H=[]
    for r in range(2*K+1):
        if host=='gamma':
            h=[one]+[zero]*K
            for j in range(r):
                for n in range(min(K,j+1),0,-1): h[n]+=j*h[n-1]
        elif host=='lognormal':
            b=T(r*(r-1))/2
            h=[one]+[b**n/T(factorial(n)) for n in range(1,K+1)]
        elif host=='binary':
            h=[T(comb(r,2*n)) if 2*n<=r else zero for n in range(K+1)]
        H.append(h)
    M=[[zero]*(2*K+1) for _ in range(K+1)]; M[0][0]=one
    loss=[zero]*(K+1)
    for n in range(1,K+1):
        forcing=[zero]*(2*n+1)
        for r in range(1,2*n+1):
            e=exp_series([r*x for x in loss[:n+1]],n)
            g=[sum(e[j]*H[r][l-j] for j in range(l+1)) for l in range(n+1)]
            g[0]-=1
            for l in range(1,n+1):
                m=n-l
                inv=one if m==0 else sum((-1)**k*comb(r+k-1,k)*M[m][k] for k in range(1,2*m+1))
                forcing[r]+=g[l]*inv
        for k in range(2*n,0,-1):
            rhs=(-a)**k*sum((-1)**(m-k)*comb(m-1,k-1)*M[n][m] for m in range(k+1,2*n+1))
            rhs+=a**k*sum((-1)**(k-r)*comb(k,r)*forcing[r] for r in range(1,k+1))
            M[n][k]=rhs/(1-(-a)**k)
        residual=sum((-1)**(k+1)*M[n][k]/k for k in range(1,2*n+1))
        loss[n]=-(1+a)*residual/a
        M[n][1]-=residual
    return loss[1:]

def gamma_refresh(sv,pv,K,prec=120,rational=False):
    """Mixed moments with lazy cached multiplication by h*C-1.
    m[n,k,j]=[v^n] E[z^k (h-1)^j]. Solve decreasing k+j,
    then increasing k. Refresh is phi identity+(1-phi) marginal.
    """
    from functools import lru_cache
    getcontext().prec=prec
    T=Fraction if rational else D
    s=T(sv); a=1-s; phi=T(pv); zero=T(0); one=T(1)
    mu=[[zero]*(K+1) for _ in range(2*K+1)]; mu[0][0]=one
    # Gamma centered-moment identity mu_j=(j-1)v(mu_{j-1}+mu_{j-2}).
    for j in range(2,2*K+1):
        for n in range(1,K+1): mu[j][n]=(j-1)*(mu[j-1][n-1]+mu[j-2][n-1])
    m={(n,0,j):mu[j][n] for n in range(K+1) for j in range(2*n+1)}
    C=[one]+[zero]*K
    def get(n,k,j):
        if n<0 or k+j>2*n: return zero
        return m.get((n,k,j),zero)
    @lru_cache(None)
    def shifted(n,k,j,r):
        if k+j+r>2*n or n<0:return zero
        if r==0:
            return phi*get(n,k,j)+(1-phi)*sum(mu[j][l]*get(n-l,k,0) for l in range(n+1))
        val=shifted(n,k,j+1,r-1)
        for l in range(1,n+1):
            if C[l]: val+=C[l]*(shifted(n-l,k,j,r-1)+shifted(n-l,k,j+1,r-1))
        return val
    for n in range(1,K+1):
        for total in range(2*n,0,-1):
            for k in range(1,total+1):
                j=total-k
                rhs=zero
                for r in range(k+1):
                    for b in range(2*n-total+1):
                        if r==0 and b==0:
                            if j:
                                rhs+=(-1)**k*(1-phi)*sum(mu[j][l]*get(n-l,k,0) for l in range(1,n+1))
                        else:
                            rhs+=(-1)**(k-r+b)*comb(k,r)*comb(k+b-1,b)*shifted(n,k-r+b,j,r)
                m[n,k,j]=a**k*rhs/(1-(-a)**k*(phi if j else 1))
        residual=sum((-1)**(k+1)*get(n,k,0)/k for k in range(1,2*n+1))
        C[n]=-(1+a)*residual/a
        m[n,1,0]-=residual
        # Cache entries of this order depended on provisional C[n]=0 and m[n,1,0].
        # Lower orders cannot change. Invalidate this order only (simple full clear).
        shifted.cache_clear()
    return log_series(C,K)[1:]

def poly_add_scaled(out,p,c):
    if c:
        for j,x in enumerate(p):
            if x:out[j]+=c*x

def lognormal_conditional(sv,pv,K,prec=120,rational=False):
    """Conditional polynomial recursion in raw Gaussian coordinate, not the
    mixed Hermite recursion of the prior attempt. P acts by exact Gaussian
    moments; (I-(-a)^k P) is solved by polynomial back substitution.
    """
    getcontext().prec=prec
    T=Fraction if rational else D
    s=T(sv); a=1-s; phi=T(pv); z=T(0); one=T(1); N=2*K
    P=[[z]*(N+1) for _ in range(N+1)]
    for j in range(N+1):
        for l in range(j%2,j+1,2):
            d=(j-l)//2
            P[j][l]=T(factorial(j)//(factorial(l)*2**d*factorial(d)))*(phi**l if l else one)*(1-phi**2)**d
    def apply(p):
        out=[z]*len(p)
        for j,x in enumerate(p):
            if x:
                for l in range(j%2,j+1,2): out[l]+=x*P[j][l]
        return out
    def solve(rhs,b):
        out=rhs[:]
        for j in range(len(out)-1,-1,-1):
            out[j]/=1-b*P[j][j]
            if out[j]:
                for l in range(j%2,j,2):out[l]+=b*P[j][l]*out[j]
        return out
    # W[r][m]=P E[(1+z)^(-r)|xi], at sigma order m.
    W=[[[one]] for _ in range(N+1)]
    loss=[z]*(K+1)
    for n in range(1,N+1):
        forcing=[[z]*(n+1) for _ in range(n+1)]
        for r in range(1,n+1):
            arg=[r*x for x in loss[:n//2+1]]
            if n>=2:arg[1]-=T(r)/2
            beta=exp_series(arg,n//2)
            powers=[T(r)**j/T(factorial(j)) for j in range(n+1)]
            for old in range(n):
                pol=W[r][old]
                for b in range((n-old)//2+1):
                    j=n-old-2*b
                    c=beta[b]*powers[j]
                    if c:
                        for l,x in enumerate(pol):
                            if x:forcing[r][l+j]+=c*x
        F=[[z]*(n+1) for _ in range(n+1)]
        PF=[[z]*(n+1) for _ in range(n+1)]
        for k in range(n,0,-1):
            rhs=[z]*(n+1)
            for m in range(k+1,n+1):poly_add_scaled(rhs,PF[m],(-a)**k*(-1)**(m-k)*comb(m-1,k-1))
            for r in range(1,k+1):poly_add_scaled(rhs,forcing[r],a**k*(-1)**(k-r)*comb(k,r))
            F[k]=solve(rhs,(-a)**k)
            PF[k]=apply(F[k])
        meanlog=sum((-1)**(k+1)/T(k)*sum(F[k][j]*P[j][0] for j in range(0,n+1,2)) for k in range(1,n+1)) if phi==0 else sum((-1)**(k+1)/T(k)*sum(F[k][j]*T(factorial(j)//(2**(j//2)*factorial(j//2))) for j in range(0,n+1,2)) for k in range(1,n+1))
        if n%2==0:
            loss[n//2]=-(1+a)*meanlog/a
            F[1][0]-=meanlog;PF[1][0]-=meanlog
        else:
            assert abs(meanlog)<T('1e-35')
        for r in range(1,N+1):
            pol=[z]*(n+1)
            for k in range(1,n+1):poly_add_scaled(pol,PF[k],T((-1)**k*comb(r+k-1,k)))
            W[r].append(pol)
    return loss[1:]

def ffirst(s,phi):
    a=1-s
    return a*(1+(3-s)*phi)/(2*(2-s)*(1+a*phi))

def compare(a,b):
    return max(abs(D(str(x))-D(str(y)))/max(ONE,abs(D(str(x))),abs(D(str(y)))) for x,y in zip(a,b))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['gate','series'],default='gate');parser.add_argument('--order',type=int,default=32);parser.add_argument('--precision',type=int,default=120)
    args=parser.parse_args(); start=time.time()
    if args.mode=='gate':
        for s in ['0.3','0.6','0.9']:
            for phi in ['0','0.5']:
                for host,fn in [('gamma',gamma_refresh),('lognormal',lognormal_conditional)]:
                    got=fn(s,phi,3,args.precision)
                    exact=fn(s,phi,3,args.precision,True)
                    rat=[D(x.numerator)/D(x.denominator) for x in exact]
                    err=compare(got,rat); first=abs(got[0]-ffirst(D(s),D(phi)))
                    print(host,s,phi,'rational error',err,'first error',first,flush=True)
                    assert max(err,first)<D('1e-35')
                    if phi=='0':
                        other=iid_moments(s,host,3,args.precision)
                        print('  iid agreement',compare(got,other),flush=True);assert compare(got,other)<D('1e-35')
            a=gamma_stein(s,12,args.precision);b=iid_moments(s,'gamma',12,args.precision)
            print('Stein vs iid order12',s,compare(a,b),flush=True);assert compare(a,b)<D('1e-35')
    else:
        results={}
        for phi in ['0','0.5']:
            for s in ['0.3','0.6','0.9']:
                for host in ['gamma','lognormal']:
                    K=args.order
                    if phi=='0':fn=lambda precision: (gamma_stein(s,K,precision) if host=='gamma' else iid_moments(s,host,K,precision))
                    else:fn=lambda precision: (gamma_refresh(s,phi,K,precision) if host=='gamma' else lognormal_conditional(s,phi,K,precision))
                    coefs=fn(args.precision)
                    key=f'{host}/s={s}/phi={phi}'
                    results[key]={'K':K,'precision':args.precision,'coefficients':[str(x) for x in coefs]}
                    (HERE/f'coefficients_K{K}_P{args.precision}.json').write_text(json.dumps(results,indent=2)+'\n')
                    print(key,'K',K,'a1',str(coefs[0])[:25],'aK',f'{coefs[-1]:.10E}','seconds',round(time.time()-start,1),flush=True)
                    assert abs(coefs[0]-ffirst(D(s),D(phi)))<D('1e-35')
    print('PASS completed',args.mode,'seconds',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
