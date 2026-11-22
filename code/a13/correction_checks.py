"""Check the corrected intermediate diagonal and complex-contraction constants."""
import mpmath as mp
mp.mp.dps=80
errs=[]; gaps=[]
for aa in ['1.01','1.25','2','5','50']:
    a=mp.mpf(aa); b=1/a; p=1-b
    for ss in ['1','1.1','2','5','20','1000']:
        s=mp.mpf(ss); e=1/s; q=1-e
        c=(2*a-1)*(2*s-1)/(2*((2*a-1)*s-a+1))
        direct=p*p+2*e*c*b
        corrected=1-q*(p+q)*(1-p*p)/(1+p*q)
        errs.append(abs(direct-corrected))
        assert errs[-1]<mp.mpf('1e-55') and direct<=1+mp.mpf('1e-55')
    for M in [1,2,7]:
        R=b/(256*M*M)
        direct=(p+R*(p+2*b*M))/(1-4*M*R)**2
        upper=(1-61*b/64)/(1-b/8)
        assert direct<=upper and upper<1
        gaps.append(upper-direct)
print('PASS corrected diagonal: 30 cells, max error',mp.nstr(max(errs),20))
print('PASS contraction bound: 15 cells, minimum upper-bound slack',mp.nstr(min(gaps),20))
