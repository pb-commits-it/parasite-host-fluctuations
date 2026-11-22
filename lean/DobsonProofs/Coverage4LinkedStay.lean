import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Topology.Instances.Matrix
import Mathlib.Topology.Algebra.InfiniteSum.Ring
import Mathlib.Topology.Algebra.InfiniteSum.NatInt
import Mathlib.Tactic

/-!
# Batch 4: linked-host stay derivative and rational long-stay residuals
Core of lem:mech-linked-stay. Actual complex derivatives and exact Laurent
identities are proved. Missing analytic/model inputs: covariance-mode reduction,
real-parameter restrictions, denominator bounds on the unbounded stay domain,
Big-O estimates and their differentiated form, and eventual sign equivalence.
-/
namespace Dobson.Coverage4LinkedStay
open scoped BigOperators
open Filter

/-- Actual complex derivative of the geometric stay return transform.
On the real axis this is the logarithmic derivative L_b times the transform. -/
theorem stay_derivative (b z : ℂ) (hden : b-(b-1)*z ≠ 0) :
    HasDerivAt (fun x : ℂ => z/(x-(x-1)*z))
      ((z/(b-(b-1)*z))*(-(1-z)/(b-(b-1)*z))) b := by
  convert (hasDerivAt_const b z).div
    ((hasDerivAt_id b).sub (((hasDerivAt_id b).sub_const 1).mul_const z)) hden using 1 <;> (try funext y) <;>
    dsimp <;> field_simp <;> ring

/-- Actual quotient-rule derivative of the linked-host mode bracket.
F and R carry their actual complex derivatives; no asymptotic premise is used. -/
theorem mode_bracket_derivative (T b z : ℂ) (F R : ℂ → ℂ) (F' R' : ℂ)
    (hF : HasDerivAt F F' b) (hR : HasDerivAt R R' b)
    (hT : T+b ≠ 0) (hz : 1-z ≠ 0) (hret : 1-R b ≠ 0) :
    HasDerivAt (fun x : ℂ => ((1+z)/(2*(1-z)))/(T+x)-F x/(1-R x))
      (-(1+z)/(2*(T+b)^2*(1-z))-
        (F'*(1-R b)+F b*R')/(1-R b)^2) b := by
  convert ((hasDerivAt_const b ((1+z)/(2*(1-z)))).div
    ((hasDerivAt_const b T).add (hasDerivAt_id b)) hT).sub
    (hF.div ((hasDerivAt_const b (1 : ℂ)).sub hR) hret) using 1 <;> (try funext y) <;>
    dsimp <;> field_simp <;> ring

/-- Exact first-order expansion, with its rational remainder, for an arbitrary
complex stay b and mode z. This does not assert a uniform Big-O domain. -/
theorem stay_laurent_identity (b z : ℂ) (hb : b ≠ 0) (hz : 1-z ≠ 0)
    (hd : b+z/(1-z) ≠ 0) :
    z/(b-(b-1)*z) = (z/(1-z))/b-
      (z/(1-z))^2/(b*(b+z/(1-z))) := by
  have he : b-(b-1)*z = (1-z)*(b+z/(1-z)) := by field_simp <;> ring
  have hformula (t : ℂ) (ht : b+t ≠ 0) :
      t/(b+t) = t/b-t^2/(b*(b+t)) := by field_simp <;> ring
  rw [he]
  simpa only [div_mul_eq_div_div] using hformula (z/(1-z)) hd

/-- Exact rotation-family decompositions. Set t=z/(1-z), A=f_a(z),
B=f_5(z), C=(1+z)/(1-z), c=a+5. The 1/b coefficients below give the real and
imaginary parts of the manuscript J_*; both residuals are explicit rational
functions. Covariance-mode identification and Big-O/derivative bounds are absent. -/
theorem rotation_rational_remainders (b t A B C c : ℂ)
    (hb : b ≠ 0) (hbt : b+t ≠ 0) (hbc : b+c ≠ 0)
    (hd : b+t-A*B*t ≠ 0) :
    C/(b+c)-2*(A*B*(t/(b+t)))/(1-A*B*(t/(b+t))) =
      (C-2*A*B*t)/b-C*c/(b*(b+c))+
        2*A*B*t*(t-A*B*t)/(b*(b+t-A*B*t)) ∧
    (A-B*(t/(b+t)))/(1-A*B*(t/(b+t))) =
      A+((A^2-1)*B*t)/b-
        (A^2-1)*B*t*(t-A*B*t)/(b*(b+t-A*B*t)) := by
  have hret : 1-A*B*(t/(b+t)) ≠ 0 := by
    have he : 1-A*B*(t/(b+t)) = (b+t-A*B*t)/(b+t) := by field_simp <;> ring
    rw [he]
    exact div_ne_zero hd hbt
  constructor <;> field_simp <;> ring


end Dobson.Coverage4LinkedStay
