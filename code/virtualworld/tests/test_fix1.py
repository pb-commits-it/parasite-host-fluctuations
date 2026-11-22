"""Fix1 regressions; numerical tolerances were frozen in protocols/vworld-fix1.md.

Known laws: independent binomial/Poisson additivity, constant-host branching
PGF, expectation matrix, and finite-record centering. No pilot outcomes are
used as acceptance targets. Run by unittest discovery alongside test_world.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '..').resolve()
sys.path.insert(0, str(HERE))
import json
import math
import unittest
import numpy as np
from scipy.stats import norm
from world.counts import binomial, poisson_product, archive
from world.design import allocate, Z95, POWER80_FACTOR, RICKER_SETTINGS, GENERATIONS, generation_steps, matched_run_steps
from world.model import ring, Model, Stage, Route
from world.durations import Duration
from world.simulation import simulate, extinct_by
from world.hosts import recorded
from world.invasion import products, threshold, averaged_reference
from run_pilot import contrast, wilson


def pgf(y, steps):
    # Independently written backward law for (2,5), ten stage-zero ancestors.
    a = b = 0.
    result = {}
    for t in range(1, steps+1):
        a, b = .5*a+.5*math.exp(b-1), .8*b+.2*math.exp(math.exp(y)*(a-1))
        result[t] = a**10
    return result


class Fix1(unittest.TestCase):
    def test_1_largest_remainders(self):
        for stay, expected in ((1,[13,7]),(5,[6,14]),(20,[2,18])):
            np.testing.assert_array_equal(allocate(20, ring([2,stay]).constant(.05)[1]), expected)
        # Three leftover units must reach three distinct largest remainders.
        np.testing.assert_array_equal(allocate(3,[.26,.25,.25,.24]), [1,1,1,0])
        np.testing.assert_array_equal(allocate(2,[1,1,1,1]), [1,1,0,0])
        for p in ([.31,.29,.21,.19], [1,2,3], [0,0,1]):
            actual=allocate(20,p); quota=20*np.array(p)/sum(p)
            self.assertEqual(actual.sum(),20)
            self.assertTrue(np.all((actual==np.floor(quota)) | (actual==np.ceil(quota))))

    def check_moments(self, values, mean, variance, fraction=.05):
        a = np.asarray(values,float)
        self.assertLessEqual(abs(a.mean()-mean),5*math.sqrt(variance/len(a))+2*abs(np.spacing(mean)))
        self.assertLessEqual(abs(a.var(ddof=1)-variance),fraction*variance)

    def test_2_chunk_additivity_known_moments(self):
        rng=np.random.default_rng(961010)
        n=np.full(50000,83)
        self.check_moments(binomial(n,.37,rng,chunk=7),83*.37,83*.37*.63)
        self.check_moments(poisson_product(n,1.7,rng,chunk=7),83*1.7,83*1.7)
        # Also exercise a per-parent rate larger than one chunk.
        self.check_moments(poisson_product(np.ones(50000,dtype=int),100.,rng,chunk=7),100,100)

    def test_2_large_chunk_moments(self):
        rng=np.random.default_rng(961011)
        n=np.full(2000,2**55)
        self.check_moments(binomial(n,.3,rng),2**55*.3,2**55*.3*.7,.15)
        self.check_moments(poisson_product(n,1.,rng),2**55,2**55,.15)

    def test_2_arbitrary_integers_and_history(self):
        huge=2**64+31
        n=np.array([huge],object); rng=np.random.default_rng(961012)
        self.assertEqual(binomial(n,1.,rng)[0],huge)
        self.assertEqual(binomial(n,0.,rng)[0],0)
        self.assertEqual(poisson_product(n,0.,rng)[0],0)
        draw=poisson_product(n,1.,rng)[0]
        self.assertIsInstance(draw,int)
        self.assertLess(abs(draw-huge),8*math.sqrt(huge))
        m=ring([2,5])
        out=simulate(m,[[huge,huge],[0,0]],3,961013,hosts=np.ones((3,2,1)),y=.2)
        self.assertEqual(out['totals'][0,0],2*huge)
        self.assertTrue(all(isinstance(x,int) for x in out['totals'].flat))
        self.assertTrue(np.all(out['totals'][:,0]>np.iinfo(np.int64).max))
        self.assertTrue(np.all(out['totals'][:,1]==0))
        self.assertEqual(out['extinction_time'].tolist(),[4,0])
        np.testing.assert_array_equal(archive(out['totals']).astype(object).astype(str),out['totals'].astype(str))
        self.assertEqual(int(archive(out['totals'])[0,0]),2*huge)
        with self.assertRaises(ValueError):
            simulate(m,[[1.5,0]],1,1,hosts=np.ones((1,1,1)))

    def test_2_known_branching_law(self):
        reference=json.loads((HERE/'results/validation.json').read_text())['gates']['v']['checks']
        for idx,y in enumerate((-.1,0.,.1)):
            predictions=pgf(y,120)
            out=simulate(ring([2,5]),np.tile([10,0],(10000,1)),120,961100+idx,
                         hosts=np.ones((120,10000,1)),y=y)
            for ti,t in enumerate((20,60,120)):
                p=predictions[t]
                self.assertLessEqual(abs(p-reference[3*idx+ti]['expected']),1e-9)
                observed=np.mean(out['extinction_time']<=t)
                self.assertLessEqual(abs(observed-p),4*math.sqrt(p*(1-p)/10000)+.002)

    def test_3_continuing_mobile_residents(self):
        # Nobody reproduces here: stage b residents advance one age and move.
        m=Model([Stage('a',Duration.geometric(1)),Stage('b',Duration.empirical([0,0,1]))],
                [Route(0,1),Route(1,0,birth=True)],patches=4,movement=1.,mobile_stage=1)
        n=np.zeros((50000,m.n),int); n[:,m.spans[1][0]]=1
        result=m.step(n,np.ones((len(n),4)),np.random.default_rng(961014))
        self.assertTrue(np.all(result.sum(axis=1)==1))
        targets=np.arange(4)*m.k+m.spans[1][1]
        self.assertEqual(result[:,targets].sum(),len(n))
        for dest in targets:
            self.assertLessEqual(abs(result[:,dest].mean()-.25),4*math.sqrt(.25*.75/len(n))+.002)
        # Independent numerical mean check covers births, continuations and movement.
        m=ring([2,5],patches=2,movement=.4)
        for source in range(m.n):
            n=np.zeros((50000,m.n),int);n[:,source]=1
            result=m.step(n,np.ones((len(n),2)),np.random.default_rng(961020+source))
            means=result.mean(axis=0);errors=4*result.std(axis=0,ddof=1)/math.sqrt(len(n))+.002
            self.assertTrue(np.all(abs(means-m.matrix()[:,source])<=errors))

    def test_4_gamma_mean_one(self):
        for shape in (1,4,16):
            d=Duration.gamma(1,shape)
            self.assertEqual(d.mean,1)
            self.assertEqual(d.tail,0)
            np.testing.assert_array_equal(d.U,[[0]])
            np.testing.assert_array_equal(d.exit,[1])
        for args in ((.99,1),(1,0),(1,-1),(1,float('nan'))):
            with self.assertRaises(ValueError):Duration.gamma(*args)

    def test_5_external_record_centering(self):
        h,base=recorded(np.arange(30),np.log(np.linspace(3,8,30)),baseline=2.)
        self.assertEqual(base,2.)
        h=np.concatenate([h,h[::-1]],axis=1); m=ring([2,5]); original=h.copy()
        for burn in (0,7):
            raw=products(m,h,.03,burn=burn)
            centered=products(m,h,.03,burn=burn,control=.3)
            for a,b in zip(raw,centered):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
            ref=averaged_reference(m,h[burn:].reshape(-1,1),np.full((30-burn)*2,1/((30-burn)*2)))
            a=threshold(m,h,burn=burn,reference=ref)
            b=threshold(m,h,burn=burn,control=.3,reference=ref)
            self.assertLessEqual(abs(a['D']-b['D']),1e-8)
            self.assertLessEqual(abs(a['half_difference']-b['half_difference']),1e-12)
        np.testing.assert_array_equal(h,original)
        # Known-mean stationary controls retain the declared correction.
        a=products(m,h,.03,burn=7)
        b=products(m,h,.03,burn=7,control=.3,control_mean=1.)
        np.testing.assert_allclose(b[0],a[0]-.3*(h[7:]-1).mean(axis=0).sum(axis=1),atol=1e-12)
        np.testing.assert_allclose(b[2],a[2]-.3*(h[18:]-1).mean(axis=0).sum(axis=1),atol=1e-12)
        constant=np.full((30,2,1),2.)
        np.testing.assert_allclose(products(m,constant,control=.3,control_mean=2.)[0],products(m,constant)[0],atol=1e-12)

    def test_6_quantile_grid_and_power(self):
        self.assertEqual(Z95,float(norm.ppf(.975)))
        self.assertEqual(len(RICKER_SETTINGS),72)
        self.assertEqual(len(set(RICKER_SETTINGS)),72)
        p=np.array([.7,.3,.4,.2]);keys=list(range(4));weights=np.array([1,-1,-1,1])
        rows={i:dict(n=400,extinct=int(400*q),extinction=q) for i,q in enumerate(p)}
        result=contrast(rows,{},keys,weights,'extinction')
        delta=float(p@weights); var=float((p*(1-p)).sum())
        np.testing.assert_allclose(result['ci95'],[delta-Z95*math.sqrt(var/400),delta+Z95*math.sqrt(var/400)],atol=1e-14)
        self.assertEqual(result['n80_per_cell'],math.ceil(POWER80_FACTOR*var/delta**2))
        self.assertAlmostEqual(wilson(0,400)[1],Z95**2/(400+Z95**2),places=14)

    def test_matched_generation_endpoints(self):
        self.assertEqual([(generation_steps(1,g),generation_steps(20,g)) for g in GENERATIONS],
                         [(69,506),(150,1100),(300,2200)])
        self.assertEqual(matched_run_steps(1),500)
        self.assertEqual(matched_run_steps(20),2200)

    def test_censoring_is_not_a_later_extinction(self):
        result=dict(totals=np.array([[1,1,0],[1,0,0],[1,0,0]]),extinction_time=np.array([3,1,0]))
        self.assertEqual(extinct_by(result,2),2)
        self.assertEqual(extinct_by(result,0),1)
        for horizon in (3,500,-1,1.5):
            with self.assertRaises(ValueError):extinct_by(result,horizon)


if __name__=='__main__':unittest.main(verbosity=2)
