import DobsonProofs.GammaThresholdJets
import DobsonProofs.GammaThird
import DobsonProofs.GammaHigher
import DobsonProofs.GammaFirstJet
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Tactic

/-!
# Batch 5, item 34 (`prop:exact-R6`): explicit Gamma quotient and threshold jets

The Gamma-integral Taylor remainders and implicit-root regularity are not proved.
The checked core computes the cubic quotient from the lower-level numerator and
normalization coefficients, solves the threshold coefficient equations uniquely,
and evaluates the normalized long-stay polynomial with its exact next error term.
-/
/-! ## Coverage correction — 2026-11-22

The three Gamma quotient jets are proved for the rescaled Kummer expectation.
This module proves the first actual threshold jet; `P1bB_ExactR6` proves
the second threshold residual limit and the cubic matching equation.
`prop:exact-R6` is core in the version-2 inventory. The stronger manuscript
Big-O remainder orders remain outside credit. -/

namespace Dobson.Coverage5GammaLongStay

/-- The first three Gamma moment ratios, derived from Mathlib's gamma function.
Identification with expectations under a normalized Gamma density is separate. -/
theorem gamma_first_three_moments (p : ℝ) (hp : 0 < p) :
    Real.Gamma (p + 1) / Real.Gamma p = p ∧
      Real.Gamma (p + 2) / Real.Gamma p = p * (p + 1) ∧
      Real.Gamma (p + 3) / Real.Gamma p = p * (p + 1) * (p + 2) := by
  have hG : Real.Gamma p ≠ 0 := (Real.Gamma_pos_of_pos hp).ne'
  have h₁ := Real.Gamma_add_one hp.ne'
  have h₂ : Real.Gamma (p + 2) = (p + 1) * Real.Gamma (p + 1) := by
    convert Real.Gamma_add_one (show p + 1 ≠ 0 by positivity) using 1 <;> congr 1 <;> ring
  have h₃ : Real.Gamma (p + 3) = (p + 2) * Real.Gamma (p + 2) := by
    convert Real.Gamma_add_one (show p + 2 ≠ 0 by positivity) using 1 <;> congr 1 <;> ring
  rw [h₃, h₂, h₁]
  constructor
  · field_simp [hG]
  constructor <;> field_simp [hG] <;> ring

/-- Cubic division of the Gamma-moment numerator by its normalization.
The three equations are coefficient matching of `denominator * m = numerator`;
no quotient coefficient is assumed. -/
theorem gamma_cubic_coefficients (p : ℝ) (hp : 0 < p) :
    GammaFirstJet.firstJet p = p ∧ GammaHigher.secondJet p = -p * (3 * p + 1) / 2 ∧
      GammaHigher.thirdJet p = p * (10 * p ^ 2 + 9 * p + 2) / 3 := by
  exact ⟨GammaFirstJet.firstJet_eq p hp,GammaHigher.secondJet_eq p hp,
    GammaHigher.thirdJet_eq p hp⟩

/-! 2026-11-16 leanp1: all three displayed Gamma quotient coefficients are
now actual right asymptotic jets of the rescaled Kummer expectation. Dominated
Taylor quotients through order three prove the cubic residual limit. No supplied
coefficient-matching equation remains in gamma_cubic_coefficients. The separate
threshold-composition identities and higher remainder are not claimed here. -/

/-! Actual quadratic Gamma jet, 2026-11-22: dominated one-sided
Taylor quotients prove the second residual limit of the normalized integral.
The cubic quotient identity is also proved; threshold composition is handled
by this module and `P1bB_ExactR6`. -/

/-! Actual first Gamma jet, 2026-11-22:
`GammaFirstJet.scaledLogMean_actual` identifies the normalized Gamma integral
with the Kummer logarithmic mean for positive scale. Its first jet is `p`;
the second and third quotient jets are also proved. See `P1bB_ExactR6` for
the completed two-term threshold residual limits. -/

/-- Matching the quadratic and cubic threshold coefficients determines both
long-stay coefficients. The displayed equations come from substituting
`q = ε exp D / (p(1-ε)^2)` into the cubic quotient and `log(1-ε)`. -/
theorem threshold_jet_coefficients (p B : ℝ) (hp : 0 < p)
    (h₃ : p * ((B + GammaThresholdJets.firstThresholdJet p ^ 2 / 2 +
      2 * GammaThresholdJets.firstThresholdJet p + 3) / p) -
      p * (3 * p + 1) / 2 * (2 * (1 / p) * ((GammaThresholdJets.firstThresholdJet p + 2) / p)) +
      p * (10 * p ^ 2 + 9 * p + 2) / 3 * (1 / p) ^ 3 - 1 / 3 = 0) :
    GammaThresholdJets.firstThresholdJet p = 1 / (2 * p) ∧
      B = -(12 * p + 7) / (24 * p ^ 2) := by
  have hA := GammaThresholdJets.firstThresholdJet_eq p hp
  refine ⟨hA, ?_⟩
  rw [hA] at h₃
  have hid : p * ((B + (1 / (2*p)) ^ 2 / 2 + 2 * (1 / (2*p)) + 3) / p) -
      p * (3 * p + 1) / 2 * (2 * (1 / p) * (((1 / (2*p)) + 2) / p)) +
      p * (10 * p ^ 2 + 9 * p + 2) / 3 * (1 / p) ^ 3 - 1 / 3 =
      B + (12*p+7)/(24*p^2) := by
    field_simp
    <;> ring
  rw [hid] at h₃
  rw [neg_div]
  linarith only [h₃]

/-! 2026-11-16 leanp1: the first threshold coefficient is now the actual
right residual limit D(1-epsilon)/epsilon of the unique physical threshold.
Its proof uses the actual threshold mean equation, the actual Gamma second jet,
and elementary exponential and logarithmic composition. Only the second
threshold coefficient matching equation remains supplied here. -/

/-- The ratio of the two-term loss polynomial to the second-order approximation
has the manuscript's `-7/(12p)` correction and an explicit quadratic remainder. -/
theorem normalized_long_stay_polynomial (p ε : ℝ) (hp : p ≠ 0)
    (hε : ε ≠ 0) (hε1 : 1 + ε ≠ 0) :
    (ε / (2 * p) - (12 * p + 7) / (24 * p ^ 2) * ε ^ 2) /
      (ε / (2 * p * (1 + ε))) =
      1 - 7 / (12 * p) * ε - (12 * p + 7) / (12 * p) * ε ^ 2 := by
  field_simp [hp, hε, hε1]
  ring

end Dobson.Coverage5GammaLongStay
#print axioms Dobson.Coverage5GammaLongStay.gamma_first_three_moments
#print axioms Dobson.Coverage5GammaLongStay.gamma_cubic_coefficients
#print axioms Dobson.Coverage5GammaLongStay.threshold_jet_coefficients
#print axioms Dobson.Coverage5GammaLongStay.normalized_long_stay_polynomial
