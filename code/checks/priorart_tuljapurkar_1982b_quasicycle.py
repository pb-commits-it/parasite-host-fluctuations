# Adopted into checks/ on 14 November 2026 at 09:42 from the working files of the reader that read the paper in full
# (cards in notes/literature/). Formula against formula; nothing here is a simulation. Unchanged below this line.
"""Claim: Tuljapurkar (1982b) eq. (11), summed as in his eq. (15) but for an autocorrelation
Re[B z^l] (second-order autoregressive quasi-cycle, z = r exp(i w0)), equals minus the project's eq. (6)
for one fluctuating transfer of a ring.  Formula against formula, 400 random rings.
Tolerance set beforehand: 1e-9 relative.  Also: direct lag sum of (11) truncated at 4000 lags agrees (1e-8)."""
import numpy as np
rng=np.random.default_rng(7)
worst=0; worst_direct=0
for it in range(400):
    n=int(rng.integers(1,9)); s=rng.uniform(0.05,0.9,n); g=rng.uniform(0.2,2.0,n)
    A=np.diag(s)
    for k in range(n): A[(k+1)%n,k]+=g[k]
    w,U=np.linalg.eig(A); o=np.argsort(-abs(w)); w=w[o]; U=U[:,o]; Vt=np.linalg.inv(U)
    lam=w[0].real; m=s/lam; tau=1/(1-m); Tg=tau.sum()
    r=rng.uniform(0.1,0.95); w0=rng.uniform(0.1,3.0); z=r*np.exp(1j*w0)
    rho1=2*r*np.cos(w0)/(1+r*r); Bc=1+1j*(np.cos(w0)-rho1/r)/np.sin(w0)
    k=int(rng.integers(0,n)); F=np.zeros((n,n)); F[(k+1)%n,k]=g[k]
    if n==1: F[0,0]=g[0]
    U0=U[:,0]; V0=Vt[0]; T0=V0@U0
    tau2=(V0@F@U0)**2/T0**2
    th=0
    for a in range(1,n):
        Ta=Vt[a]@U[:,a]
        th+=(Bc*z/(1-z*w[a]/lam)).real*0  # placeholder, real part must be taken after the product (complex eigenvalues)
    # correct: theta is real overall; sum over alpha of Re over the two conjugate geometric terms
    th=0
    for a in range(1,n):
        Ta=Vt[a]@U[:,a]; coef=(V0@F@U[:,a])*(Vt[a]@F@U0)/(lam**2*T0*Ta)
        th+=0.5*coef*(Bc*z/(1-z*w[a]/lam)+np.conj(Bc)*np.conj(z)/(1-np.conj(z)*w[a]/lam))
    his=(-tau2/(2*lam**2)+th).real
    Rz=np.prod((1-m)*z/(1-m*z)); S0=1+2*(Bc*z/(1-z)).real
    ours=S0/(2*Tg**2)-(1/Tg)*(Bc*Rz/(1-Rz)).real
    worst=max(worst,abs(his+ours)/abs(ours))
    if it<40:
        # direct lag sum of (11): theta = sum_l rho(l) * [V0' F (A/lam)^(l-1) F U0/(lam^2 T0) - (V0'FU0)^2/(lam^2 T0^2)]
        acc=0; M=np.eye(n)
        for l in range(1,4001):
            rho=(Bc*z**l).real
            acc+=rho*((V0@F@M@F@U0)/(lam**2*T0)-(V0@F@U0)**2/(lam**2*T0**2))
            M=M@A/lam
        worst_direct=max(worst_direct,abs((-tau2/(2*lam**2)+acc).real+ours)/abs(ours))
print("worst relative difference, (15)-style sum vs project eq. (6): %.1e (400 rings)"%worst)
print("worst relative difference, direct lag sum of (11) vs project eq. (6): %.1e (40 rings)"%worst_direct)
assert worst<1e-9 and worst_direct<1e-8
