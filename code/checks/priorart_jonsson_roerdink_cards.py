# Adopted into checks/ on 14 November 2026 at 09:42 from the working files of the reader that read the paper in full
# (cards in notes/literature/). Formula against formula; nothing here is a simulation. Unchanged below this line.
"""Checks behind statements labelled 'my algebra' in the prior-art cards.
(1) Jonsson (2021) Table 1: the elasticities printed there equal the two-stage ring quantities
    e(f)=1/T, e(s1)=tau1/T, e(s2)=(tau2-1)/T, e(g)=1/T-(tau1-1)/T*g/(1-g), with tau_k=1/(1-stay_k/lambda).
(2) Roerdink (1988) eqs (6.1),(6.7a),(6.8): T0 = 1+lam0^2/(f*phibar*s) equals tau1+tau2 of the ring,
    so the printed small-noise cost tau^2/(2 lam0^2) = CV^2/(2 T^2).
(3) Roerdink at f=1: exact cost per step vs small-noise value (factor 2 at leading order).
(4) Simulation of Roerdink's model (iid gamma) at small CV to confirm (2) numerically.
"""
import math, random
lam = 2.78
rows = {  # name: (s1, s2, g, f_printed, e_s1, e_s2, e_g, e_f)
 'SDM': (0.001,0.1,0.1,74480, 0.4909,0.0183,0.4907,0.4908),
 'SDF': (0.9,0.1,0.1,58.7,    0.5763,0.0152,0.3898,0.4084),
 'SPM': (0.001,0.1,0.9,8278., 0.4909,0.0183,0.4907,0.4908),
 'SPF': (0.9,0.1,0.9,8.9,     0.4991,0.0180,0.3375,0.4829),
 'IDM': (0.001,0.9,0.1,52247.,0.4035,0.1931,0.4034,0.4034),
 'IDF': (0.9,0.9,0.1,41.2,    0.4883,0.1657,0.3302,0.3460),
 'IPM': (0.001,0.9,0.9,5807., 0.4034,0.1931,0.4033,0.4034),
 'IPF': (0.9,0.9,0.9,6.2,     0.4114,0.1906,0.2782,0.3981),
}
print("(1) Jonsson Table 1 vs ring quantities (max abs diff per row)")
worst = 0
for k,(s1,s2,g,fp,es1,es2,eg,ef) in rows.items():
    t1 = 1/(1 - s1*(1-g)/lam); t2 = 1/(1 - s2/lam); T = t1+t2
    f9 = (lam**2 + (s1*g - s1 - s2)*lam + s1*s2*(1-g))/(s1*g)   # her eq (9)
    pred = (t1/T, (t2-1)/T, 1/T - (t1-1)/T*g/(1-g), 1/T)
    d = max(abs(a-b) for a,b in zip(pred,(es1,es2,eg,ef)))
    worst = max(worst,d)
    print(f"  {k}: T={T:.4f} pred e(s1,s2,g,f)=({pred[0]:.4f},{pred[1]:.4f},{pred[2]:.4f},{pred[3]:.4f}) maxdiff={d:.5f}  f(eq9)={f9:.4g} printed {fp}")
print("  worst diff", worst)

print("(2) Roerdink T0 vs ring generation time")
for (s,f,pb) in [(0.9,0.2,2.0),(0.9,0.6,2.0),(0.9,0.95,3.0),(0.5,0.3,8.0)]:
    stay = s*(1-f)
    l0 = 0.5*(stay + math.sqrt(stay**2 + 4*f*pb*s))
    T0 = 1 + l0**2/(f*pb*s)
    T = 1 + 1/(1 - stay/l0)
    print(f"  s={s} f={f} phibar={pb}: T0={T0:.6f}  tau1+tau2={T:.6f}")

print("(3) f=1: exact cost per step = 0.5*(ln phibar - E ln phi) ; small-noise = CV^2/8")
def digamma(x):
    r=0
    while x<6: r-=1/x; x+=1
    return r+math.log(x)-1/(2*x)-1/(12*x**2)+1/(120*x**4)-1/(252*x**6)
for a in [400.,100.,25.]:
    cv2=1/a
    exact = 0.5*(math.log(a)-digamma(a))
    print(f"  CV^2={cv2:.4f}: exact/CV^2={exact/cv2:.4f}  small-noise/CV^2={1/8:.4f}")

print("(4) simulation, s=0.9 f=0.5 phibar=2, gamma shape a (CV^2=1/a)")
random.seed(1)
s,f,pb=0.9,0.5,2.0
stay=s*(1-f); l0=0.5*(stay+math.sqrt(stay**2+4*f*pb*s)); T=1+1/(1-stay/l0)
for a in [100.,400.]:
    k=a/pb; n1,n2=1.0,1.0; acc=0.0; N=4_000_000; burn=1000
    for t in range(N+burn):
        phi=random.gammavariate(a,1/k)
        m1=f*phi*n2; m2=s*n1+stay*n2; tot=m1+m2
        if t>=burn: acc+=math.log(tot)
        n1,n2=m1/tot,m2/tot
    g=acc/N
    print(f"  a={a}: (ln lam0 - gamma)/CV^2 = {(math.log(l0)-g)*a:.4f}   1/(2T^2) = {1/(2*T*T):.4f}")
