"""Structural and analytic checks before scientific validation; run with -I.

These tests verify law equivalences, event boundaries and state bookkeeping.
Scientific stochastic tolerances and gate results live in protocol/validation.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from world import Duration,Stage,Route,Model
from world.model import ring
from world.hosts import recorded, Ricker, lognormal
from world.simulation import simulate
from world.invasion import products,averaged_reference,threshold


class Laws(unittest.TestCase):
    def test_residence_means(self):
        for m in (2.,5.,20.):
            self.assertAlmostEqual(Duration.geometric(m).mean,m,places=10)
        self.assertAlmostEqual(Duration.chain(20,4).mean,20,places=10)
        self.assertAlmostEqual(Duration.empirical([.2,.3,.5]).mean,2.3,places=10)
        for shape in (1,4,16):
            d=Duration.gamma(5,shape)
            self.assertAlmostEqual(d.mean,5,places=8)
            self.assertLess(d.tail,1e-12)

    def test_noiseless_growth(self):
        for means in ([2,5],[1,2,7],[2,3,4,5]):
            m=ring(means)
            self.assertAlmostEqual(m.critical_tilt(),0.,places=10)
            h=np.ones((300,2,1))
            r,_,_=products(m,h,.1)
            self.assertLess(abs(r.mean()-np.log(m.constant(.1)[0])),1e-12)

    def test_exact_fixed_delay(self):
        d=Duration.empirical([0,0,1])
        m=Model([Stage('a',d),Stage('b',Duration.geometric(1))],
                [Route(0,1,poisson=False),Route(1,0,birth=True)])
        counts=np.array([[1,0,0,0]])
        rng=np.random.default_rng(1)
        for target in (1,2,3):
            counts=m.step(counts,np.ones((1,1)),rng)
            self.assertEqual(counts[0,target],1)
            self.assertEqual(counts.sum(),1)

    def test_conservation_and_absorption(self):
        m=ring([2,5],patches=4,movement=.2)
        n=np.zeros((3,m.n),dtype=int)
        self.assertFalse(m.step(n,np.ones((3,4)),np.random.default_rng(2)).any())
        self.assertTrue(np.allclose(m.move.sum(axis=0),1))

    def test_delay_and_saturation(self):
        for mode in ('delay','linear_delay'):
            m=Model([Stage('a',Duration.geometric(4),.125,mode=mode),Stage('b',Duration.geometric(5))],
                    [Route(0,1,host=0),Route(1,0,birth=True)])
            self.assertTrue(np.allclose(m.matrix(),[[.75,.2],[.125,.8]]))
            if mode=='linear_delay':
                with self.assertRaises(ValueError): m.step(np.ones((1,2),int),np.array([[10.]]),np.random.default_rng(1))
            else:
                for h in (0.,1.,1e6):
                    m.step(np.ones((1,2),int)*100,np.array([[h]]),np.random.default_rng(1))
        m=ring([2,5]); m.saturation=True
        self.assertTrue(np.allclose(m.rates(np.array([0.,1.,100.]).reshape(-1,1)).ravel(),[0,1,200/101]))

    def test_branch_and_patch_mean(self):
        # This is a nonstochastic identity: equal alternatives at equal hosts
        # reduce to one route at the level of expectations, even after movement.
        m=ring([2,5],drivers=2,alternatives=True,patches=4,movement=.2)
        a=m.matrix(np.ones(8))
        plain=ring([2,5]).matrix()
        for j in range(2):
            self.assertTrue(np.allclose(a[:,j].reshape(4,2).sum(axis=0),plain[:,j]))

    def test_record_contract(self):
        h,base=recorded([0,1,2],np.log([1,2,3]))
        self.assertEqual(base,2)
        self.assertAlmostEqual(h.mean(),1)
        with self.assertRaises(ValueError): recorded([0,1,3],[0,0,0])

    def test_ricker_feedback(self):
        no=Ricker(np.ones((2,1))*1000,sigma=0,alpha=0)
        yes=Ricker(np.ones((2,1))*1000,sigma=0,alpha=.1)
        self.assertTrue(np.all(no.step(np.ones((2,1))*10)==1))
        self.assertTrue(np.all(yes.step(np.ones((2,1))*10)<1))
        m=ring([2,5])
        out=simulate(m,np.zeros((2,2),int),3,7,endogenous=no)
        self.assertTrue(np.all(out['extinction_time']==0))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): Duration.geometric(.5)
        with self.assertRaises(ValueError): Duration.chain(2,4)
        with self.assertRaises(ValueError): Duration.empirical([.2,.3])
        with self.assertRaises(ValueError): lognormal(3,2,.3,1.,3)
        with self.assertRaises(ValueError): Model([Stage('a',Duration.geometric(2))],[])

    def test_mean_rate_reference(self):
        m=ring([2,5]);m.saturation=True
        h=np.array([[.5],[1.5]])
        ref=averaged_reference(m,h,np.array([.5,.5]))
        self.assertAlmostEqual(ref['tilt'],-np.log((2*.5/1.5+2*1.5/2.5)/2),places=11)
        with self.assertRaises(ValueError): threshold(m,np.ones((10,2,1)))

    def test_delay_no_death_threshold(self):
        m=Model([Stage('search',Duration.geometric(4),mode='delay'),Stage('host',Duration.geometric(5))],
                [Route(0,1,host=0),Route(1,0,birth=True)])
        h=np.array([[.2],[1.8]])
        ref=averaged_reference(m,h,np.array([.5,.5]))
        self.assertAlmostEqual(ref['tilt'],0.,places=11)
        self.assertTrue(np.allclose(m.matrix(h).sum(axis=1),1.))


if __name__=='__main__': unittest.main(verbosity=2)
