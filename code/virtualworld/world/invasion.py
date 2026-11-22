"""Conditional-mean invasion growth and birth-threshold roots.

Normalized matrix products avoid overflow; they do not condition on survival.
Pathwise derivatives propagate alongside composition. A known-mean host
control variate is optional and changes sampling error, not the target.
Error bars describe independent histories or antithetic pair means. They do
not enclose finite-time/burn-in bias or uncertainty in the biological inputs.
"""
import math
import numpy as np
from scipy import integrate, optimize, special


def units(values, pairs):
    return (values[:pairs]+values[pairs:])/2 if pairs else values


def se(values):
    a = np.asarray(values)
    return float(a.std(ddof=1)/np.sqrt(len(a)))


def products(model, hosts, y=0., burn=0, pairs=0, control=0., initial=None, control_mean=None):
    """Return per-history mean log growth, slope, and last-half growth.

For affine event laws, derivatives are analytic. Nonlinear contact still has
an analytic derivative in the birth tilt, evaluated from its event matrices.
All arrays include the burn-in; host starts should be stationary.
With control_mean supplied, it is the known expectation of the normalized
host driver (not the abundance normalization baseline). With control_mean=None,
center each finite record on its own measured-window mean; its summed control
is exactly zero. Full and last-half windows center separately, so recorded or
externally normalized series keep their unadjusted target and diagnostics.
Finite-record centering offers no variance reduction of a fixed record.
"""
    hosts = np.asarray(hosts, float)
    if hosts.ndim != 3 or not 0 <= burn < len(hosts):
        raise ValueError('nonempty time x replicate x driver histories and valid burn required')
    center = None if control_mean is None else np.asarray(control_mean, float)
    if center is not None and (np.any(~np.isfinite(center)) or np.any(center < 0)):
        raise ValueError('control mean must be finite and nonnegative')
    if center is not None:
        center = np.broadcast_to(center, hosts.shape[1:])
    rates = model.rates(hosts)
    rcount = hosts.shape[1]
    v = model.constant(y)[1] if initial is None else np.asarray(initial,float)
    x = np.broadcast_to(v,(rcount,model.n)).copy()
    x /= x.sum(axis=1)[:,None]
    dx = np.zeros_like(x)
    logs, derivatives, halves = (np.zeros(rcount) for _ in range(3))
    start_half = burn+(len(hosts)-burn)//2
    A = model.parts[0]+np.exp(y)*model.parts[1]
    Ay = np.exp(y)*model.parts[1]
    B = model.noise_parts[0]+np.exp(y)*model.noise_parts[1]
    By = np.exp(y)*model.noise_parts[1]
    for t,rr in enumerate(rates):
        if model.affine:
            mat = A+np.einsum('rd,dij->rij',rr-1,B,optimize=False)
            maty = Ay+np.einsum('rd,dij->rij',rr-1,By,optimize=False)
        else:
            mat = model.matrix(hosts[t],y)
            maty = (model.matrix(hosts[t],y+math.log(2))-mat)
        z = np.einsum('rij,rj->ri',mat,x,optimize=False)
        dz = np.einsum('rij,rj->ri',mat,dx,optimize=False)+np.einsum('rij,rj->ri',maty,x,optimize=False)
        total = z.sum(axis=1)
        if np.any(total<=0) or np.any(z < -1e-12):
            raise FloatingPointError('invalid conditional mean propagation')
        der = dz.sum(axis=1)/total
        x = z/total[:,None]
        dx = dz/total[:,None]-x*der[:,None]
        if t>=burn:
            inc = np.log(total)
            # Empirical centering has exactly zero window sum: leave raw logs
            # untouched instead of accumulating cancellation roundoff.
            if control and center is not None:
                inc = inc-control*(hosts[t]-center).sum(axis=1)
            logs += inc; derivatives += der
            if t>=start_half:
                halves += inc
    n = len(hosts)-burn
    return (units(logs/n,pairs),units(derivatives/n,pairs),
            units(halves/(len(hosts)-start_half),pairs))


def averaged_reference(model, host_states, weights):
    """Arithmetic-mean-rate critical tilt from explicit quadrature/states.

Weights must describe a prespecified marginal host law, or a declared finite
record. No temporal independence is assumed by averaging the single-step
operator; memory still enters the subsequent random product. At nonlinear
contact this supplies the separate mean-rate reference required for D.
"""
    h=np.asarray(host_states,float);w=np.asarray(weights,float)
    if h.ndim!=2 or w.shape!=(len(h),) or np.any(w<0) or not np.isclose(w.sum(),1.,rtol=0,atol=1e-12):
        raise ValueError('supply normalized weights for host states')
    A=np.einsum('r,rij->ij',w,model.matrix(h,0.))
    F=np.einsum('r,rij->ij',w,model.matrix(h,math.log(2)))-A
    U=A-F
    def rate(y):
        return math.log(float(max(np.linalg.eigvals(U+math.exp(y)*F).real)))
    tilt=float(optimize.brentq(rate,-20,20,xtol=1e-13))
    return dict(tilt=tilt,mean_host_tilt=model.critical_tilt(),U=U,F=F)


def threshold(model, hosts, burn=0, pairs=0, control=0., initial_y=None, reference=None, control_mean=None):
    """Root of mean conditional log growth relative to constant-host criticality.

    For mass action or bounded linear contact with mean-one hosts, the mean-host
    operator equals the arithmetic-mean-rate operator. Curved contact requires
    an explicit averaged_reference; omission raises rather than mislabelling
    a mean-host contrast as D. The two baseline shifts are returned separately.
"""
    if (model.saturation or not model.affine) and reference is None:
        raise ValueError('nonlinear contact requires an explicit arithmetic-mean-rate reference')
    host_baseline=model.critical_tilt()
    baseline=host_baseline if reference is None else float(reference['tilt'])
    y = baseline if initial_y is None else baseline+initial_y
    for iteration in range(12):
        r, slope, half = products(model,hosts,y,burn,pairs,control,control_mean=control_mean)
        b = float(slope.mean())
        if b<=0:
            raise RuntimeError('nonpositive birth sensitivity')
        correction = float(r.mean()/b)
        if abs(correction)<=1e-7:
            break
        y -= float(np.clip(correction,-2,2))
    else:
        raise RuntimeError('threshold root did not converge')
    influence = (y-baseline)-r/b
    half_diff = (half-r)/b
    return dict(D=float(y-baseline),se=se(influence),samples=influence.tolist(),
                baseline=baseline,y=float(y),residual=abs(correction),iterations=iteration+1,
                slope=b,half_difference=float(half_diff.mean()),half_se=se(half_diff),
                raw_growth=float(r.mean()),control=control,
                control_mean=None if control_mean is None else np.asarray(control_mean).tolist(),
                mean_rate_shift=float(baseline-host_baseline),
                total_mean_host_shift=float(y-host_baseline))


def ring_coefficient(means,phi):
    """Known one-transfer small-noise threshold coefficient for primitive rings."""
    t = np.asarray(means,float)
    if np.all(t==1):
        raise ValueError('periodic endpoint is outside the primitive coefficient')
    theta = phi/(1-phi)
    R = np.prod(phi/(t+(1-t)*phi))
    return float((1+2*theta)/(2*t.sum())-R/(1-R))


def resolvent_coefficient(model,phi):
    """Independent matrix perturbation formula, unit variance, one noisy driver.

The right vector sums to one; w*v=1 normalizes the left vector. All noise
contributions sharing driver zero are combined before the resolvent is formed.
"""
    y = model.critical_tilt()
    A = model.matrix(y=y)
    val,right = np.linalg.eig(A)
    v = right[:,np.argmin(abs(val-1))].real
    v /= v.sum()
    val,left = np.linalg.eig(A.T)
    w = left[:,np.argmin(abs(val-1))].real
    w /= w@v
    B = model.noise_parts[0,0]+np.exp(y)*model.noise_parts[1,0]
    Q = np.eye(model.n)-np.outer(v,w)
    c = float(w@B@v)
    C = c*c/2-phi*float(w@B@np.linalg.solve(np.eye(model.n)-phi*A@Q,Q@B@v))
    birth = float(w@(np.exp(y)*model.parts[1])@v)
    return C/birth


def gamma_exact(s,p,refined=False):
    """Independent quadrature of Roerdink's invariant density in log X.

Density times dx becomes exp[p*z-p*log(s+exp(z))-s*p*exp(z)/c] dz.
The log-density mode solves x*(s+x)=c. Integration is centered there, with
long left tails for p<1. Range refinement is numerical evidence, not rigorous
interval certification. Only independent gamma hosts and 0<s<1 apply.
"""
    if not 0<s<1 or p<=0:
        raise ValueError('require interior staying and positive gamma shape')
    def growth(y):
        c = (1-s)*np.exp(y)
        mode = math.log(2*c/(s+math.sqrt(s*s+4*c)))
        def ld(z):
            return p*z-p*np.logaddexp(math.log(s),z)-s*p/c*math.exp(z)
        peak = ld(mode)
        width = max(60.,45/p)*(2 if refined else 1)
        hi = max(12.,math.log(100/p))*(2 if refined else 1)
        def f(z):
            return math.exp(ld(z)-peak)
        tol = 1e-11 if refined else 1e-9
        norm = integrate.quad(f,mode-width,mode,epsabs=tol,epsrel=tol)[0]+integrate.quad(f,mode,mode+hi,epsabs=tol,epsrel=tol)[0]
        def g(z):
            return f(z)*np.logaddexp(math.log(s),z)
        num = integrate.quad(g,mode-width,mode,epsabs=tol,epsrel=tol)[0]+integrate.quad(g,mode,mode+hi,epsabs=tol,epsrel=tol)[0]
        return num/norm
    upper = math.log(p)-special.digamma(p)
    return float(optimize.brentq(growth,0,upper,xtol=2e-12))
