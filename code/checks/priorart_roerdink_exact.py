# Adopted into checks/ on 14 November 2026 at 09:42 from the working files of the reader that read the paper in full
# (cards in notes/literature/). Formula against formula; nothing here is a simulation. Unchanged below this line.
"""Check, by quadrature of Roerdink's (1988) exact formula (4.5) with z from (3.16), that
(ln lam0 - gamma)/CV^2 -> 1/(2 T^2), T = tau1 + tau2 = T0 of his (6.7a).
Tolerance fixed beforehand: relative deviation below 2% at CV^2 = 1/400 and shrinking with CV^2."""
import math
def gamma_exact(s,f,a,k,n=200001):
    z = k*s*(1-f)**2/f
    # integrate over u = ln tau; integrand w(u) = tau^a (1+tau)^(-a) exp(-z tau)
    # find peak of log w
    def logw(u):
        t = math.exp(u); return a*u - a*math.log1p(t) - z*t
    # peak: a/(1+t) = z t  -> t = (-1+sqrt(1+4a/z))/2
    tp = (-1+math.sqrt(1+4*a/z))/2; up = math.log(tp)
    # curvature-based width
    h=1e-4; c = -(logw(up+h)-2*logw(up)+logw(up-h))/h**2; w = 12/math.sqrt(c)
    lo,hi = up-w, up+w; du=(hi-lo)/(n-1); m=logw(up)
    num=den=0.0
    for i in range(n):
        u=lo+i*du; e=math.exp(logw(u)-m); wt = 0.5 if i in (0,n-1) else 1.0
        den+=wt*e; num+=wt*e*math.log1p(math.exp(u))
    return math.log(s*(1-f)) + num/den
for (s,f,pb) in [(0.9,0.5,2.0),(0.9,0.2,2.0),(0.9,0.8,3.0)]:
    stay=s*(1-f); l0=0.5*(stay+math.sqrt(stay**2+4*f*pb*s)); T=1+l0**2/(f*pb*s)
    for a in [25.,100.,400.,1600.]:
        k=a/pb; g=gamma_exact(s,f,a,k)
        print(f"s={s} f={f} phibar={pb} CV^2={1/a:.5f}: (ln lam0-gamma)/CV^2={(math.log(l0)-g)*a:.5f}  1/(2T^2)={1/(2*T*T):.5f}")
