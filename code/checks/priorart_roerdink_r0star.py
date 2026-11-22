# Adopted into checks/ on 14 November 2026 at 09:42 from the working files of the reader that read the paper in full
# (cards in notes/literature/). Formula against formula; nothing here is a simulation. Unchanged below this line.
"""Claim checked: from Roerdink's (1988) exact growth rate (4.5), gamma = ln(stay) + G(a, z), stay = s(1-f),
z = k s (1-f)^2 / f, dividing the births f*phi by R is z -> z R. So the root R0* of gamma = 0 satisfies
G(a, z R0*) = -ln(stay), and D = ln R0 - ln R0* = -ln[ zeta (1-stay) / (a stay^2) ], zeta = z R0*,
depends on (a, stay) only. Prediction (project's N8 at n = 2, no autocorrelation, one fluctuating transfer):
D / CV^2 -> 1/(2 T0), T0 = 1 + 1/(1-stay), as CV^2 = 1/a -> 0.
Tolerance fixed beforehand: within 2% at CV^2 = 1/400, deviation shrinking in proportion to CV^2."""
import math
def G(a,z,n=40001):
    def logw(u):
        t=math.exp(u); return a*u - a*math.log1p(t) - z*t
    tp=(-1+math.sqrt(1+4*a/z))/2; up=math.log(tp)
    h=1e-4; c=-(logw(up+h)-2*logw(up)+logw(up-h))/h**2; w=12/math.sqrt(c)
    lo=up-w; du=2*w/(n-1); m=logw(up); num=den=0.0
    for i in range(n):
        u=lo+i*du; e=math.exp(logw(u)-m); wt=0.5 if i in (0,n-1) else 1.0
        den+=wt*e; num+=wt*e*math.log1p(math.exp(u))
    return num/den
def D(a,stay):
    target=-math.log(stay)          # G decreasing in z
    lo,hi=1e-6*a,1e6*a
    for _ in range(80):
        mid=math.sqrt(lo*hi)
        if G(a,mid)>target: lo=mid
        else: hi=mid
    zeta=math.sqrt(lo*hi)
    return -math.log(zeta*(1-stay)/(a*stay*stay))
for stay in (0.2,0.45,0.8):
    T0=1+1/(1-stay)
    for a in (25.,100.,400.):
        print(f"stay={stay} CV^2={1/a:.4f}: D/CV^2={D(a,stay)*a:.5f}   1/(2 T0)={1/(2*T0):.5f}")
