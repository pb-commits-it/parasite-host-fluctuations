import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic

/-!
# Fluctuating staying entries: exact GIG stationarity substitution

Core of `prop:exact-R8`. The transition kernel is K(z,x) proportional to
c/x² · (c/x-z)^(p-1) exp(-(c/x-z)/θ), for 0<z<c/x. The proposed
invariant density is proportional to z^(-p-1) exp(-(z+c/z)/θ).
The substitution z=c/(v+x) is constructed and its derivative, domain,
real-power Jacobian identity, and exponential cancellation are proved.

This is the actual invariant-kernel integrand calculation. No invariant law,
change-of-variables integral theorem, Bessel normalization/moments, strong law,
uniqueness, threshold sign or endpoint divergence is assumed or asserted.
-/
namespace Dobson.Coverage6ExactStaying

/-- The substitution maps the positive half-line into the exact transition support. -/
theorem gig_substitution_support (c x v : ℝ) (hc : 0 < c) (hx : 0 < x)
    (hv : 0 < v) : 0 < c/(v+x) ∧ c/(v+x) < c/x := by
  constructor
  · positivity
  · exact div_lt_div_of_pos_left hc hx (by linarith)

/-- Its Jacobian is the one used in the invariant-kernel calculation. -/
theorem gig_substitution_derivative (c x v : ℝ) (h : v+x ≠ 0) :
    HasDerivAt (fun t => c/(t+x)) (-c/(v+x)^2) v := by
  convert (hasDerivAt_const v c).div ((hasDerivAt_id v).add_const x) h using 1
  · ext t; rfl
  · simp

/-- Exact real-power Jacobian identity for every real shape p. -/
theorem gig_power_jacobian (c x v p : ℝ) (hc : 0 < c) (hx : 0 < x) (hv : 0 < v) :
    (c/(v+x))^(-p-1) * (c/x-c/(v+x))^(p-1) * (c/(v+x)^2) =
      c⁻¹*x^(1-p)*v^(p-1) := by
  have hvx : 0 < v+x := by positivity
  have hz : 0 < c/(v+x) := by positivity
  have hdiff : c/x-c/(v+x)=c*v/(x*(v+x)) := by field_simp; ring
  have hdiffpos : 0 < c/x-c/(v+x) := sub_pos.mpr (gig_substitution_support c x v hc hx hv).2
  rw [show -p-1= -(p+1) by ring, Real.rpow_neg hz.le,
    Real.rpow_add_one hz.ne', Real.rpow_sub_one hdiffpos.ne',
    Real.rpow_sub hx 1 p, Real.rpow_one,
    Real.rpow_sub_one hv.ne', hdiff]
  rw [Real.div_rpow hc.le hvx.le,
    Real.div_rpow (mul_nonneg hc.le hv.le) (mul_nonneg hx.le hvx.le),
    Real.mul_rpow hc.le hv.le, Real.mul_rpow hx.le hvx.le]
  have hcP : c^p ≠ 0 := (Real.rpow_pos_of_pos hc p).ne'
  have hxP : x^p ≠ 0 := (Real.rpow_pos_of_pos hx p).ne'
  have hvxP : (v+x)^p ≠ 0 := (Real.rpow_pos_of_pos hvx p).ne'
  field_simp
  <;> ring

/-- The exponential factors of the proposed invariant density and the Gamma
transition kernel cancel to the target density times the Gamma integral factor. -/
theorem gig_exponential_cancellation (c x v θ : ℝ) (hc : 0 < c)
    (hx : 0 < x) (hv : 0 < v) :
    Real.exp (-((c/(v+x))+c/(c/(v+x)))/θ) *
      Real.exp (-(c/x-c/(v+x))/θ) =
      Real.exp (-(x+c/x)/θ) * Real.exp (-v/θ) := by
  rw [← Real.exp_add, ← Real.exp_add]
  congr 1
  have hvx : v+x ≠ 0 := by positivity
  field_simp
  <;> ring

/-- The transformed unnormalized stationarity integrand factors exactly into
the proposed x-density and a Gamma(v) factor, including the outer c/x² term.
Only the common Gamma normalizing constant is suppressed on both sides. -/
theorem gig_kernel_integrand (c x v p θ : ℝ) (hc : 0 < c)
    (hx : 0 < x) (hv : 0 < v) :
    (c/(v+x))^(-p-1) * Real.exp (-((c/(v+x))+c/(c/(v+x)))/θ) *
      (c/x^2*(c/x-c/(v+x))^(p-1)*Real.exp (-(c/x-c/(v+x))/θ)) *
      (c/(v+x)^2) =
      x^(-p-1)*Real.exp (-(x+c/x)/θ) * (v^(p-1)*Real.exp (-v/θ)) := by
  have hid := gig_power_jacobian c x v p hc hx hv
  have he := gig_exponential_cancellation c x v θ hc hx hv
  have hp : x^(1-p)/x^2=x^(-p-1) := by
    rw [show 1-p= -p+1 by ring, Real.rpow_add_one hx.ne',
      show -p-1= -p-1 by rfl, Real.rpow_sub_one hx.ne']
    field_simp
    <;> ring
  calc
    _ = (c/(v+x))^(-p-1)*(c/x-c/(v+x))^(p-1)*(c/(v+x)^2) *
        (Real.exp (-((c/(v+x))+c/(c/(v+x)))/θ)*Real.exp (-(c/x-c/(v+x))/θ)) * (c/x^2) := by ring
    _ = (c⁻¹*x^(1-p)*v^(p-1)) *
        (Real.exp (-(x+c/x)/θ)*Real.exp (-v/θ)) * (c/x^2) := by rw [hid, he]
    _ = (x^(1-p)/x^2)*Real.exp (-(x+c/x)/θ)*(v^(p-1)*Real.exp (-v/θ)) := by
      field_simp
      <;> ring
    _ = _ := by rw [hp]

end Dobson.Coverage6ExactStaying
#print axioms Dobson.Coverage6ExactStaying.gig_substitution_support
#print axioms Dobson.Coverage6ExactStaying.gig_substitution_derivative
#print axioms Dobson.Coverage6ExactStaying.gig_power_jacobian
#print axioms Dobson.Coverage6ExactStaying.gig_exponential_cancellation
#print axioms Dobson.Coverage6ExactStaying.gig_kernel_integrand
