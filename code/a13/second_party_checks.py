"""Independent check: outage proof algebra and explicit remainders agree with
80-digit finite periodic matrix products. Read second_party_protocol.md first.
This file imports no author code. Every environment here is deterministic with
uniform random phase when stationarity is invoked; none is called iid.
"""
import itertools
import time
import mpmath as mp

mp.mp.dps = 80
M = mp.mpf
stats = {}
counts = {}
start = time.monotonic()


def record(name, value, tolerance):
    value = abs(value)
    stats[name] = max(stats.get(name, M(0)), value)
    counts[name] = counts.get(name, 0) + 1
    assert value <= tolerance, (name, mp.nstr(value, 30))


def nonnegative(name, value, slack=M('1e-55')):
    record(name, max(M(0), -value), slack)


def entrymax(A):
    return max(abs(x) for x in A)


def product(matrices):
    out = mp.eye(matrices[0].rows)
    for A in matrices:
        out = A * out
    return out


def ring(p, z, h):
    C = mp.diag(p)
    for j in range(1, len(p)):
        C[j, j-1] = 1
    C[0, len(p)-1] += z*h
    return C


def ring2(a, eps, c, h):
    b = 1/a
    return mp.matrix([[1-b, eps*c], [b*h, 1-eps]])


def perron2(A):
    return (A[0,0]+A[1,1] + mp.sqrt((A[0,0]-A[1,1])**2+4*A[0,1]*A[1,0]))/2


def det_I2(P):
    return (1-P[0,0])*(1-P[1,1])-P[0,1]*P[1,0]


def threshold2(a, eps, hosts):
    def residual(c):
        P = product([ring2(a, eps, c, h) for h in hosts])
        return det_I2(P)
    c = mp.findroot(residual, (M('.9'), M('1.1')), tol=M('1e-72'))
    assert c > 0
    P = product([ring2(a, eps, c, h) for h in hosts])
    record('threshold_perron_residual', perron2(P)-1, M('1e-55'))
    return mp.log(c)


def perron_data(p, z, G):
    lo = max(p)
    hi = lo + 1 + z*G
    for _ in range(290):
        mid = (lo+hi)/2
        if mp.fprod(mid-x for x in p) < z*G:
            lo = mid
        else:
            hi = mid
    lam = (lo+hi)/2
    k = len(p)
    v = mp.matrix([M(1)]*k)
    w = mp.matrix([M(1)]*k)
    for j in range(1, k):
        v[j] = v[j-1]/(lam-p[j])
        w[j] = w[j-1]*(lam-p[j-1])
    w /= (w.T*v)[0]
    return lam, w, v


# Gate: known constant-host arithmetic threshold before new tests.
for a, s in itertools.product(map(M, ['1.01', '2', '20']), map(M, ['1', '2', '100'])):
    record('known_constant_host', perron2(ring2(a, 1/s, M(1), M(1)))-1, M('1e-60'))
print('Known-result gate passed: 9 constant-host thresholds.')

# Direct matrix determinant, root and derivative for the pure-outage formula.
for a, s in itertools.product(map(M, ['1.01', '2', '50']), map(M, ['1', '1.5', '10', '1000'])):
    f = lambda t: mp.log((2*a-1)*(2*t-1)/(2*((2*a-1)*t-a+1)))
    D = f(s)
    P = ring2(a, 1/s, mp.exp(D), M(0))*ring2(a, 1/s, mp.exp(D), M(2))
    record('alternation_determinant', det_I2(P), M('1e-55'))
    record('alternation_perron', perron2(P)-1, M('1e-55'))
    p,q=1-1/a,1-1/s
    record('alternation_corrected_diagonal',P[0,0]-(1-q*(p+q)*(1-p*p)/(1+p*q)),M('1e-55'))
    record('alternation_derivative', mp.diff(f,s)-1/((2*s-1)*((2*a-1)*s-a+1)), M('1e-50'))
    assert D < 0 and mp.diff(f,s) > 0

# Independent companion construction and finite cone deletion.
stay_sets = [[M(x) for x in xs] for xs in [
    ['0', '.8'], ['.2', '.8', '.8'], ['0', '.3', '.9', '.1'],
    ['.2', '.2', '0', '.95', '.4']]]
for base in stay_sets:
    k = len(base)
    transforms = []
    for shift in range(3):
        p = base[shift % k:] + base[:shift % k]
        C = ring(p, M('.37'), M(0))
        e = mp.matrix([M(1)] + [M(0)]*(k-1))
        columns = [C**j*e for j in range(k)]
        K = mp.matrix([[columns[j][i] for j in range(k)] for i in range(k)])
        record('K_upper_triangular', max([abs(K[i,j]) for i in range(k) for j in range(i)] or [M(0)]), M('1e-55'))
        record('K_determinant', mp.det(K)-1, M('1e-55'))
        transforms.append([K**-1*ring(p,M('.37'),h)*K for h in map(M,['0','.01','7'])])
    for transformed in transforms[1:]:
        for X, Y in zip(transformed, transforms[0]):
            record('simultaneous_similarity', entrymax(X-Y), M('1e-55'))
    p = [max(base)] + list(base)
    p.pop(1+p[1:].index(max(base)))
    C = ring(p,M('.37'),M(0))
    e = mp.matrix([M(1)] + [M(0)]*(k-1))
    for hs in ([0,0,2,0,1,3], [0,1,0,2,0,3], [1,2,3,0,0,0]):
        hosts = list(map(M, hs))
        x = e.copy()
        for h in hosts:
            x = ring(p,M('.37'),h)*x
            nonnegative('cone_invariance', min((C-max(p)*mp.eye(k))*x))
        Q = product([ring(p,M('.37'),h) for h in hosts if h>0])
        gap = x-max(p)**hosts.count(0)*Q*e
        nonnegative('outage_deletion', min(gap))
    assert min(C**(k-1)*e)>0

# Enumerate labelled reference paths, including scalar stay/host self-edges.
for p in [[M('.7')], [M('.8'),M(0)], [M('.1'),M('.9'),M('.3')], [M('.8'),M(0),M('.2'),M('.8')]]:
    z, G = M('.37'), M('1.3')
    k = len(p)
    lam,w,v = perron_data(p,z,G)
    Aref = ring(p,z,G)
    record('reference_perron_left',entrymax(w.T*Aref-lam*w.T),M('1e-55'))
    record('reference_perron_right',entrymax(Aref*v-lam*v),M('1e-55'))
    nu = w[0]*z*G*v[k-1]/lam
    for hosts in [[G]*4, list(map(M,['.001','1000','.2','4']))]:
        paths = [(j,w[j]*v[j],M(0)) for j in range(k)]
        for h in hosts:
            new = []
            for j,prob,reward in paths:
                if p[j]>0:
                    new.append((j,prob*p[j]/lam,reward))
                i = (j+1)%k
                edgeweight = z*G if j==k-1 else M(1)
                P = w[i]*edgeweight/(lam*w[j])
                new.append((i,prob*P,reward+(mp.log(h/G) if j==k-1 else 0)))
            paths = new
        prob = mp.fsum(x[1] for x in paths)
        meanlog = mp.fsum(x[1]*x[2] for x in paths)
        mgf = mp.fsum(x[1]*mp.exp(x[2]) for x in paths)
        direct = (w.T*product([ring(p,z,h) for h in hosts])*v)[0]/lam**len(hosts)
        record('reference_path_probability',prob-1,M('1e-50'))
        record('reference_path_log_mean',meanlog-nu*mp.fsum(mp.log(h/G) for h in hosts),M('1e-50'))
        record('reference_path_mgf',mgf-direct,M('1e-50'))
        nonnegative('finite_path_jensen',mp.log(direct)-meanlog,M('1e-50'))
        if hosts==[G]*4:
            record('finite_path_constant_equality',mp.log(direct),M('1e-50'))

# Periodic covariance and finite-fourth-moment constants and remainder.
laws = [list(map(M,hs)) for hs in [[2,0],[3,0,0],[0,0,M('.5'),M('3.5')],[0,M('.01'),0,M('.99'),4]]]
for hs,a in itertools.product(laws,map(M,['1.25','2','5'])):
    m = len(hs)
    assert mp.fsum(hs)/m == 1
    b,p = 1/a,1-1/a
    H = (mp.fsum(h**4 for h in hs)/m)**M('.25')
    cov = lambda ell: mp.fsum((hs[t]-1)*(hs[(t-ell)%m]-1) for t in range(m))/m
    A = cov(0)/2+mp.fsum(p**(ell-1)*cov(ell) for ell in range(1,m+1))/(1-p**m)
    L = 2*(4*H+1)/b
    N = L*(6*H+1)/b
    Rg = 2*H*N+H*L*(6*H+2)+M(8)/3*(4*H+1)**3
    B0 = 4*H**2/b+2/b+2*H**2+3
    J = 4*H**2/b+2*H**2+1/b+1
    K0 = B0+Rg
    Cbound = J*K0+Rg+2*K0**2
    e0 = min(b/2,1/(4*(B0+Rg+1)))
    S = mp.fsum(p**(ell-1)*mp.fsum(hs[t]*hs[(t-ell)%m] for t in range(m))/m for ell in range(1,m+1))/(1-p**m)
    record('coefficient_covariance_identity',S-1/b+cov(0)/2-A,M('1e-50'))
    for power in range(3):
        eps = e0/(2*10**power)
        D = threshold2(a,eps,hs)
        ratio = abs(D-A*eps)/(Cbound*eps**2)
        stats['largest_remainder_fraction'] = max(stats.get('largest_remainder_fraction',M(0)),ratio)
        nonnegative('explicit_D_remainder',Cbound*eps**2-abs(D-A*eps),M('1e-45'))
    # A separate ratio iteration at three fixed tilts, not the product root.
    eps = e0/2
    for c in map(M,['.5','1','2']):
        vv = M(1)
        for iteration in range(10000):
            old = vv
            for h in hs:
                vv = (p*vv+b)/(1+eps*(c*h*vv-1))
            if abs(vv-old)<M('1e-72'):
                break
        else:
            raise AssertionError('ratio did not converge')
        vs=[]
        for h in hs:
            vs.append(vv)
            vv = (p*vv+b)/(1+eps*(c*h*vv-1))
        us=[-mp.fsum(p**(ell-1)*(c*hs[(t-ell)%m]-1) for ell in range(1,m+1))/(1-p**m) for t in range(m)]
        ds=[v-1 for v in vs]
        etas=[d-eps*u for d,u in zip(ds,us)]
        Ws=[c*h*v-1 for h,v in zip(hs,vs)]
        for t in range(m):
            j=(t+1)%m
            record('delta_recursion',ds[j]-p*ds[t]+eps*vs[j]*Ws[t],M('1e-45'))
            record('eta_recursion',etas[j]-p*etas[t]+eps*((vs[j]-1)*Ws[t]+c*hs[t]*(vs[t]-1)),M('1e-45'))
        n4=lambda xs:(mp.fsum(abs(x)**4 for x in xs)/m)**M('.25')
        n2=lambda xs:mp.sqrt(mp.fsum(abs(x)**2 for x in xs)/m)
        nonnegative('delta_L4_bound',L*eps-n4(ds),M('1e-45'))
        nonnegative('eta_L2_bound',N*eps**2-n2(etas),M('1e-45'))
        nonnegative('W_L4_bound',4*H+1-n4(Ws),M('1e-45'))
        B=c*c*S-c/b+(c*c*mp.fsum(h*h for h in hs)/m-2*c+1)/2
        rr=mp.fsum(mp.log1p(eps*W) for W in Ws)/m
        err=abs(rr/eps-(c-1)+eps*B)
        nonnegative('growth_remainder',Rg*eps**2-err,M('1e-45'))

print('All independent checks passed.')
for key in sorted(stats):
    print(f'{key}: count={counts.get(key, "diagnostic")}; max={mp.nstr(stats[key],18)}')
print(f'Elapsed seconds: {time.monotonic()-start:.3f}')
