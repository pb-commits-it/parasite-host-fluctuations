"""Second-order loss per generation, Gamma/sigma^2, from paper 1's ring propositions.
prop:ring-gamma (one fluctuating transfer, geometric stays tau_k), prop:ring-alt (parallel hosts)."""
import itertools
def gamma_one(taus, phi):
    T = sum(taus)
    if phi == 0: return 1/(2*T)
    th = phi/(1-phi); R = 1.0
    for t in taus: R /= (1 + t/th)
    return (1+2*th)/(2*T) - R/(1-R)
def gamma_alt(taus, phi, shares, rel_var=None, rho=0.0):
    # equal-memory hosts; rel_var = variance multipliers; rho = common cross-correlation
    M = len(shares); rel_var = rel_var or [1.0]*M
    base = gamma_one(taus, phi)
    s = [w*v**0.5 for w, v in zip(shares, rel_var)]
    V = sum(x*x for x in s) + rho*sum(s[i]*s[j] for i in range(M) for j in range(M) if i != j)
    return V*base
if __name__ == "__main__":
    for phi in (0.0, 0.5, 0.8, 0.9):
        a = 6
        rows = {"1 host (a)": [a]}
        for b in (1, 3, 6, 12, 24, 60, 120):
            rows[f"2 hosts (a,{b})"] = [a, b]
        for b1, b2 in ((3,3),(6,6),(12,12),(30,30),(60,60),(1,59),(1,119)):
            rows[f"3 hosts (a,{b1},{b2})"] = [a, b1, b2]
        print(f"phi={phi}  one-stage ref 1/(2a)={1/(2*a):.4f}")
        for k, v in rows.items(): print(f"   {k:22s} T={sum(v):4d}  G/s2={gamma_one(v, phi):.4f}")
