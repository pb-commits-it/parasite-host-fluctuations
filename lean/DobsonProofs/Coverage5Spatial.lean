import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Tactic

/-!
# Batch 5: exact long-stay rational expansion

For a fixed movement factor f, put c = θ U, A=q(1+2θ)/2, B=q c,
E=(1-q) f c, C=sum of other stays, D=θ-c, G=θ-f c. The conservative
island coefficient, after cancelling Ra/(1-Ra) and Rb/(1-Rb), is
A/(L+C)-B/(L+D)-E/(L+G). We prove its exact remainder and derivative.
Identification with stochastic threshold loss and the case where the moving
stage itself lengthens (so f depends on L) are not proved in this file.
-/
namespace Dobson.Coverage5Spatial

noncomputable def longStayCoefficient (A B E C D G L : ℝ) : ℝ :=
  A / (L + C) - B / (L + D) - E / (L + G)

/-- Exact remainder, stronger than the formal first-order rational expansion. -/
theorem long_stay_exact_remainder (A B E C D G L : ℝ)
    (hL : L ≠ 0) (hC : L + C ≠ 0) (hD : L + D ≠ 0) (hG : L + G ≠ 0) :
    longStayCoefficient A B E C D G L - (A - B - E) / L =
      -A * C / (L * (L + C)) + B * D / (L * (L + D)) +
        E * G / (L * (L + G)) := by
  unfold longStayCoefficient
  field_simp
  <;> ring

/-- Quantitative O(L^-2) bound for nonnegative physical coefficients and stays. -/
theorem long_stay_remainder_bound (A B E C D G L : ℝ)
    (hA : 0 ≤ A) (hB : 0 ≤ B) (hE : 0 ≤ E)
    (hC : 0 ≤ C) (hD : 0 ≤ D) (hG : 0 ≤ G) (hL : 0 < L) :
    |longStayCoefficient A B E C D G L - (A - B - E) / L| ≤
      (A * C + B * D + E * G) / L ^ 2 := by
  rw [long_stay_exact_remainder A B E C D G L (ne_of_gt hL)
    (by positivity) (by positivity) (by positivity)]
  have hAC : 0 ≤ A * C / (L * (L + C)) := by positivity
  have hBD : 0 ≤ B * D / (L * (L + D)) := by positivity
  have hEG : 0 ≤ E * G / (L * (L + G)) := by positivity
  have hbA : A * C / (L * (L + C)) ≤ A * C / L ^ 2 := by
    apply div_le_div_of_nonneg_left (by positivity) (by positivity)
    nlinarith
  have hbB : B * D / (L * (L + D)) ≤ B * D / L ^ 2 := by
    apply div_le_div_of_nonneg_left (by positivity) (by positivity)
    nlinarith
  have hbE : E * G / (L * (L + G)) ≤ E * G / L ^ 2 := by
    apply div_le_div_of_nonneg_left (by positivity) (by positivity)
    nlinarith
  rw [neg_mul, neg_div, add_div, add_div]
  apply abs_le.mpr
  constructor <;> linarith

/-- Differentiate the actual partial-fraction coefficient. -/
theorem long_stay_derivative (A B E C D G L : ℝ)
    (hC : L + C ≠ 0) (hD : L + D ≠ 0) (hG : L + G ≠ 0) :
    HasDerivAt (longStayCoefficient A B E C D G)
      (-A / (L + C) ^ 2 + B / (L + D) ^ 2 + E / (L + G) ^ 2) L := by
  have ha := (hasDerivAt_const L A).div ((hasDerivAt_id L).add_const C) hC
  have hb := (hasDerivAt_const L B).div ((hasDerivAt_id L).add_const D) hD
  have he := (hasDerivAt_const L E).div ((hasDerivAt_id L).add_const G) hG
  convert (ha.sub hb).sub he using 1
  · funext x
    rfl
  · dsimp
    ring

/-- Identify the actual island-mode return fractions with the long-stay model,
and recover the manuscript constant K_infinity without an assumed asymptote. -/
theorem island_coefficient_reduction (q θ U f C L : ℝ)
    (h0 : L + θ ≠ 0) (hC : L + C ≠ 0)
    (ha : L + θ - θ * U ≠ 0) (hb : L + θ - f * θ * U ≠ 0) :
    q * ((1 + 2 * θ) / (2 * (L + C)) -
        (θ * U / (L + θ)) / (1 - θ * U / (L + θ))) -
      (1 - q) * ((f * θ * U / (L + θ)) / (1 - f * θ * U / (L + θ))) =
      longStayCoefficient (q * (1 + 2 * θ) / 2) (q * θ * U)
        ((1 - q) * f * θ * U) C (θ - θ * U) (θ - f * θ * U) L ∧
    q * (1 + 2 * θ) / 2 - q * θ * U - (1 - q) * f * θ * U =
      q * (1 + 2 * θ) / 2 - θ * U * (q + (1 - q) * f) := by
  have hea : 1 - θ * U / (L + θ) = (L + θ - θ * U) / (L + θ) := by
    field_simp
    <;> ring
  have heb : 1 - f * θ * U / (L + θ) = (L + θ - f * θ * U) / (L + θ) := by
    field_simp
    <;> ring
  have hda : L + (θ - θ * U) ≠ 0 := by convert ha using 1 <;> ring
  have hdb : L + (θ - f * θ * U) ≠ 0 := by convert hb using 1 <;> ring
  constructor
  · rw [hea, heb]
    unfold longStayCoefficient
    simp only [← add_sub_assoc]
    field_simp [h0, hC, ha, hb]
    <;> ring
  · ring

end Dobson.Coverage5Spatial
