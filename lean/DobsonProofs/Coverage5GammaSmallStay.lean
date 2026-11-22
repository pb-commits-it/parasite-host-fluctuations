import Mathlib.Analysis.SpecialFunctions.Sqrt
import DobsonProofs.Coverage2Exact
import DobsonProofs.KummerSmallStayAsymptotics
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.NumberTheory.Harmonic.EulerMascheroni
import Mathlib.Tactic

/-!
# Batch 5, item 36 (`prop:exact-R5`): selected quadratic root and inverse-log jets

These are exact scalar cores of the small-staying expansion. The Kummer connection
formula, its normally convergent remainder bounds, the threshold relation with its
error, the gamma/digamma identification, and the parity-cohort strong law are not
asserted. In particular this file does not prove the original model asymptotic.
-/
/-! ## Coverage correction — 2026-11-22

The first two inverse-log threshold residual limits are proved in this module;
`P1bB_ExactR5` proves the third coefficient and discharges the remaining
matching equation. `prop:exact-R5` is core in the version-2 inventory.
Stronger Big-O/radical error orders and the exact zero-stay population endpoint
remain outside credit. -/

/-! Actual threshold limits, 2026-11-22: the unique integral root
and its first two inverse-log residual limits are proved using Kummer integral
asymptotics. The third residual limit is proved in `P1bB_ExactR5`; these results
do not supply the stronger manuscript Big-O errors. -/

/-! leanp1 correction — 2026-11-16. Part1:60 is now closed by the
actual second residual limit of the unique physical threshold. The partition
constant and its remainder are proved for the actual Kummer integrals, using
Gamma/digamma identities, dominated convergence and moment derivatives. The
third coefficient premise remains explicit until the next closure. -/

namespace Dobson.Coverage5GammaSmallStay

open Filter Set MeasureTheory
open scoped Topology
open Dobson.Coverage2Exact Dobson.NoiseAnalyticity

/-- The unique physical threshold in its logarithmic restoring coordinate. -/
noncomputable def gammaThreshold (p s : ℝ) : ℝ :=
  if h : 0 < p ∧ 0 < s ∧ s < 1 then
    (exact_root p s h.1 h.2.1 h.2.2).exists.choose else 0

theorem gammaThreshold_spec (p s : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1) :
    exactGrowth p s (gammaThreshold p s) = 0 ∧
      0 < gammaThreshold p s ∧ gammaThreshold p s < gammaLogLoss p := by
  simp only [gammaThreshold, dif_pos (show 0 < p ∧ 0 < s ∧ s < 1 from ⟨hp, hs, hs1⟩)]
  exact (exact_root p s hp hs hs1).exists.choose_spec

noncomputable def thresholdKappa (p s : ℝ) : ℝ := exactKappa p s (gammaThreshold p s)

theorem thresholdKappa_pos (p s : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1) :
    0 < thresholdKappa p s := by
  dsimp [thresholdKappa, exactKappa]
  positivity

theorem threshold_mean (p s : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1) :
    kummerLog1p p (thresholdKappa p s) = Real.log (1 / s) := by
  have h := (gammaThreshold_spec p s hp hs hs1).1
  change Real.log s + kummerLog1p p (thresholdKappa p s) = 0 at h
  rw [one_div, Real.log_inv]
  linarith

theorem thresholdKappa_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (thresholdKappa p) (𝓝[>] 0) (𝓝[>] 0) := by
  have hupper : Tendsto (fun s : ℝ => s ^ 2 * p / (1 - s)) (𝓝[>] 0) (𝓝 0) := by
    have hh : ContinuousAt (fun s : ℝ => s ^ 2 * p / (1 - s)) 0 := by
      apply ContinuousAt.div (by fun_prop) (by fun_prop)
      norm_num
    simpa using hh.tendsto.mono_left nhdsWithin_le_nhds
  have hlim : Tendsto (thresholdKappa p) (𝓝[>] 0) (𝓝 0) := by
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hupper
    · filter_upwards [self_mem_nhdsWithin,
        mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with s hs hs1
      exact (thresholdKappa_pos p s hp hs hs1).le
    · filter_upwards [self_mem_nhdsWithin,
        mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with s hs hs1
      change s < 1 at hs1
      have hD := (gammaThreshold_spec p s hp hs hs1).2.1
      have hexp : 1 ≤ Real.exp (gammaThreshold p s) := Real.one_le_exp_iff.mpr hD.le
      dsimp [thresholdKappa, exactKappa]
      apply div_le_div_of_nonneg_left (by positivity) (by linarith : 0 < 1 - s)
      nlinarith
  apply tendsto_nhdsWithin_iff.mpr
  refine ⟨hlim, ?_⟩
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with s hs hs1
  exact thresholdKappa_pos p s hp hs hs1

private theorem log_mul_log_one_sub_tendsto_zero :
    Tendsto (fun s : ℝ => Real.log (1 / s) * Real.log (1 - s)) (𝓝[>] 0) (𝓝 0) := by
  have hd : HasDerivAt (fun s : ℝ => Real.log (1 - s)) (-1) 0 := by
    convert (((hasDerivAt_id (0 : ℝ)).const_sub 1).log (by norm_num)) using 1 <;> norm_num
  have hr : Tendsto (fun s : ℝ => Real.log (1 - s) / s) (𝓝[>] 0) (𝓝 (-1)) := by
    simpa only [zero_add, sub_zero, Real.log_one, smul_eq_mul, div_eq_mul_inv, mul_comm] using
      hd.tendsto_slope_zero_right
  have hslog : Tendsto (fun s : ℝ => s * Real.log (1 / s)) (𝓝[>] 0) (𝓝 0) := by
    have hh := (tendsto_log_mul_rpow_nhdsGT_zero (r := (1 : ℝ)) (by norm_num)).neg
    simpa only [Real.rpow_one, neg_zero, one_div, Real.log_inv, mul_neg, neg_mul, mul_comm] using hh
  have hh := hslog.mul hr
  simp only [zero_mul] at hh
  apply hh.congr'
  filter_upwards [self_mem_nhdsWithin] with s hs
  change 0 < s at hs
  field_simp

/-- The physical deficit is exactly the normalized score numerator plus the
vanishing discrete-time correction log(1-s). -/
theorem threshold_deficit_identity (p s : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1) :
    gammaLogLoss p - gammaThreshold p s =
      (∫ t in Ioi (0 : ℝ), -kummerShapeScore t * kummerWeight p (thresholdKappa p s) t) /
        kummerZ p (thresholdKappa p s) + Real.log (1 - s) := by
  have hκ := thresholdKappa_pos p s hp hs hs1
  rw [kummerScoreNumerator_identity p _ hp hκ, threshold_mean p s hp hs hs1]
  have hlog : Real.log (thresholdKappa p s) =
      2 * Real.log s + Real.log p - Real.log (1 - s) - gammaThreshold p s := by
    dsimp [thresholdKappa, exactKappa]
    rw [Real.log_div (mul_pos (sq_pos_of_pos hs) hp).ne'
      (mul_pos (sub_pos.mpr hs1) (Real.exp_pos _)).ne',
      Real.log_mul (sq_pos_of_pos hs).ne' hp.ne',
      Real.log_mul (sub_pos.mpr hs1).ne' (Real.exp_pos _).ne', Real.log_pow, Real.log_exp]
    ring
  rw [hlog, one_div, Real.log_inv]
  dsimp [gammaLogLoss]
  ring

/-- First inverse-log coefficient of the actual physical threshold as s tends
to zero, without a connection-formula or coefficient-matching hypothesis. -/
theorem threshold_first_inverse_log_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun s : ℝ => Real.log (1 / s) * (gammaLogLoss p - gammaThreshold p s))
      (𝓝[>] 0) (𝓝 (gammaTrigamma p / 2)) := by
  have hmZ : Tendsto (fun κ => kummerLog1p p κ / kummerZ p κ)
      (𝓝[>] 0) (𝓝 (1 / 2)) := by
    have hh := (kummerLog1p_log_scale_tendsto p hp).div
      (kummerZ_log_ratio_tendsto p hp) (by norm_num : (1 : ℝ) ≠ 0)
    simp only [div_one] at hh
    apply hh.congr'
    filter_upwards [self_mem_nhdsWithin,
      mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
    have hL : 0 < Real.log (1 / κ) :=
      Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
    dsimp only [Pi.div_apply]
    field_simp [hL.ne']
  have hnum := hmZ.mul (kummerScoreNumerator_tendsto p hp)
  have hcomp := hnum.comp (thresholdKappa_tendsto_zero p hp)
  have hsum := hcomp.add log_mul_log_one_sub_tendsto_zero
  have hlim : (1 / 2 : ℝ) * gammaTrigamma p + 0 = gammaTrigamma p / 2 := by ring
  rw [hlim] at hsum
  apply hsum.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with s hs hs1
  dsimp only [Function.comp_def]
  rw [threshold_mean p s hp hs hs1, threshold_deficit_identity p s hp hs hs1]
  ring

/-- The first coefficient is defined by the actual threshold's asymptotic limit. -/
noncomputable def firstInverseLogCoefficient (p : ℝ) : ℝ :=
  limUnder (𝓝[>] (0 : ℝ))
    (fun s => Real.log (1 / s) * (gammaLogLoss p - gammaThreshold p s))

theorem firstInverseLogCoefficient_eq (p : ℝ) (hp : 0 < p) :
    firstInverseLogCoefficient p = gammaTrigamma p / 2 :=
  (threshold_first_inverse_log_limit p hp).limUnder_eq



private theorem log_pow_mul_log_one_sub_tendsto_zero (n : ℕ) :
    Tendsto (fun s : ℝ => Real.log (1 / s)^n * Real.log (1 - s))
      (𝓝[>] 0) (𝓝 0) := by
  have hd : HasDerivAt (fun s : ℝ => Real.log (1 - s)) (-1) 0 := by
    convert (((hasDerivAt_id (0 : ℝ)).const_sub 1).log (by norm_num)) using 1 <;> norm_num
  have hr : Tendsto (fun s : ℝ => Real.log (1 - s) / s) (𝓝[>] 0) (𝓝 (-1)) := by
    simpa only [zero_add, sub_zero, Real.log_one, smul_eq_mul, div_eq_mul_inv, mul_comm] using
      hd.tendsto_slope_zero_right
  have hslog : Tendsto (fun s : ℝ => s * Real.log (1 / s)^n) (𝓝[>] 0) (𝓝 0) := by
    simpa only [one_mul, mul_one, add_zero, Function.comp_def, one_div, div_inv_eq_mul, mul_comm] using
      (Real.tendsto_pow_log_div_mul_add_atTop 1 0 n one_ne_zero).comp tendsto_inv_nhdsGT_zero
  have hh := hslog.mul hr
  simp only [zero_mul] at hh
  apply hh.congr'
  filter_upwards [self_mem_nhdsWithin] with s hs
  change 0 < s at hs
  field_simp

/-- The second inverse-log coefficient comes from the actual physical threshold. -/
theorem threshold_second_inverse_log_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun s : ℝ => Real.log (1 / s)^2 * (gammaLogLoss p - gammaThreshold p s) -
      gammaTrigamma p / 2 * Real.log (1 / s)) (𝓝[>] 0)
      (𝓝 (gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) / 2)) := by
  have ht := ((kummer_second_inverse_log_limit p hp).comp
    (thresholdKappa_tendsto_zero p hp)).add (log_pow_mul_log_one_sub_tendsto_zero 2)
  simp only [add_zero] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with s hs hs1
  dsimp only [Function.comp_def]
  rw [threshold_mean p s hp hs hs1, threshold_deficit_identity p s hp hs hs1]
  ring

/-- The second coefficient is defined by the actual threshold's residual limit. -/
noncomputable def secondInverseLogCoefficient (p : ℝ) : ℝ :=
  limUnder (𝓝[>] (0 : ℝ)) (fun s =>
    Real.log (1 / s)^2 * (gammaLogLoss p - gammaThreshold p s) -
      firstInverseLogCoefficient p * Real.log (1 / s))

theorem secondInverseLogCoefficient_eq (p : ℝ) (hp : 0 < p) :
    secondInverseLogCoefficient p =
      gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) / 2 := by
  unfold secondInverseLogCoefficient
  rw [firstInverseLogCoefficient_eq p hp]
  exact (threshold_second_inverse_log_limit p hp).limUnder_eq


noncomputable def smallRoot (B v : ℝ) : ℝ :=
  (B - Real.sqrt (B ^ 2 - 4 * v)) / 2

/-- The radical in R5 is the unique root on the smaller half of the quadratic.
Its strict positivity and the branch-selection inequality are proved. -/
theorem selected_quadratic_root (B v : ℝ) (hB : 0 < B) (hv : 0 < v)
    (hdisc : 4 * v < B ^ 2) :
    0 < smallRoot B v ∧ smallRoot B v < B / 2 ∧
      smallRoot B v * (B - smallRoot B v) = v ∧
      ∀ x : ℝ, 0 ≤ x → x ≤ B / 2 → x * (B - x) = v → x = smallRoot B v := by
  have hd : 0 < B ^ 2 - 4 * v := by linarith
  have hs0 : 0 < Real.sqrt (B ^ 2 - 4 * v) := Real.sqrt_pos.mpr hd
  have hs2 := Real.sq_sqrt hd.le
  have hsB : Real.sqrt (B ^ 2 - 4 * v) < B := by nlinarith
  have hr0 : 0 < smallRoot B v := by unfold smallRoot; linarith
  have hrB : smallRoot B v < B / 2 := by unfold smallRoot; linarith
  have hroot : smallRoot B v * (B - smallRoot B v) = v := by
    unfold smallRoot
    nlinarith [hs2]
  refine ⟨hr0, hrB, hroot, ?_⟩
  intro x _hx0 hxB hx
  have hf : (x - smallRoot B v) * (B - x - smallRoot B v) = 0 := by
    nlinarith [hx, hroot]
  rcases mul_eq_zero.mp hf with h | h
  · linarith
  · linarith

/-- Explicit normalized error in the expansion `u = q + q² + O(q³)` for
`u(1-u)=q`. With `u=smallRoot(B,v)/B` and `q=v/B²`, this gives the
`v/B + v²/B³` radical expansion; no asymptotic equality is a hypothesis. -/
theorem quadratic_root_error (q u : ℝ) (hq0 : 0 ≤ q) (hq : q ≤ 1 / 8)
    (hu0 : 0 ≤ u) (hu : u ≤ 1 / 2) (hroot : u * (1 - u) = q) :
    0 ≤ u - (q + q ^ 2) ∧ |u - (q + q ^ 2)| ≤ 6 * q ^ 3 := by
  have huq : u ≤ 2 * q := by nlinarith [mul_nonneg hu0 (sub_nonneg.mpr hu)]
  have hu4 : u ≤ 1 / 4 := by linarith
  have hq2 : q ^ 2 ≤ q / 8 := by nlinarith [mul_nonneg hq0 (sub_nonneg.mpr hq)]
  have hw : q + q ^ 2 ≤ 1 / 4 := by linarith
  have hden : 1 / 2 ≤ 1 - u - (q + q ^ 2) := by linarith
  have hfactor : (u - (q + q ^ 2)) * (1 - u - (q + q ^ 2)) =
      2 * q ^ 3 + q ^ 4 := by linear_combination hroot
  have hd0 : 0 ≤ u - (q + q ^ 2) := by
    by_contra h
    have hn := mul_neg_of_neg_of_pos (lt_of_not_ge h) (by linarith : 0 < 1 - u - (q + q ^ 2))
    nlinarith [pow_nonneg hq0 3, pow_nonneg hq0 4]
  have hq4 : q ^ 4 ≤ q ^ 3 := by
    have h := mul_le_mul_of_nonneg_right (show q ≤ 1 by linarith) (pow_nonneg hq0 3)
    nlinarith [h]
  refine ⟨hd0, ?_⟩
  rw [abs_of_nonneg hd0]
  nlinarith [mul_nonneg hd0 (show 0 ≤ (1 - u - (q + q ^ 2)) - 1 / 2 by linarith)]

/-- Inserting `δ=c₁/L+c₂/L²+c₃/L³` into `δ(2(L-a)-δ)=v`
uniquely determines all three inverse-log coefficients of the deficit. -/
/- The first two inverse-log coefficients are proved for the unique
integral threshold. This helper retains the third matching premise, discharged
for that threshold by `P1bB_ExactR5.inverse_log_third_balance`. -/
theorem inverse_log_coefficients (p c₃ : ℝ) (hp : 0 < p)
    (h₃ : 2 * c₃ - 2 * (gammaLogDerivative p + Real.eulerMascheroniConstant) *
      secondInverseLogCoefficient p - firstInverseLogCoefficient p ^ 2 = 0) :
    firstInverseLogCoefficient p = gammaTrigamma p / 2 ∧
      secondInverseLogCoefficient p =
        gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) / 2 ∧
      c₃ = gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) ^ 2 / 2 +
        gammaTrigamma p ^ 2 / 8 := by
  have hc₁ := firstInverseLogCoefficient_eq p hp
  have hc₂ := secondInverseLogCoefficient_eq p hp
  refine ⟨hc₁, hc₂, ?_⟩
  rw [hc₁, hc₂] at h₃
  nlinarith

/-- Explicit remainder for the manuscript's radical expansion, proved for all
`B>0` and `0<v≤B²/8`: the omitted term is bounded by `6v³/B⁵`. -/
theorem scaled_quadratic_root_error (B v : ℝ) (hB : 0 < B) (hv : 0 < v)
    (hvsmall : v ≤ B ^ 2 / 8) :
    |smallRoot B v - (v / B + v ^ 2 / B ^ 3)| ≤ 6 * v ^ 3 / B ^ 5 := by
  have hdisc : 4 * v < B ^ 2 := by nlinarith [sq_pos_of_pos hB]
  have hr := selected_quadratic_root B v hB hv hdisc
  have hq0 : 0 ≤ v / B ^ 2 := by positivity
  have hq : v / B ^ 2 ≤ 1 / 8 := by
    apply (div_le_iff₀ (sq_pos_of_pos hB)).mpr
    linarith
  have hu0 : 0 ≤ smallRoot B v / B := div_nonneg hr.1.le hB.le
  have hu : smallRoot B v / B ≤ 1 / 2 := by
    apply (div_le_iff₀ hB).mpr
    linarith [hr.2.1]
  have hroot : smallRoot B v / B * (1 - smallRoot B v / B) = v / B ^ 2 := by
    field_simp
    nlinarith [hr.2.2.1]
  have he := (quadratic_root_error (v / B ^ 2) (smallRoot B v / B)
    hq0 hq hu0 hu hroot).2
  calc
    |smallRoot B v - (v / B + v ^ 2 / B ^ 3)| =
        |B * (smallRoot B v / B - (v / B ^ 2 + (v / B ^ 2) ^ 2))| := by
      congr 1
      field_simp <;> ring
    _ = B * |smallRoot B v / B - (v / B ^ 2 + (v / B ^ 2) ^ 2)| := by
      rw [abs_mul, abs_of_pos hB]
    _ ≤ B * (6 * (v / B ^ 2) ^ 3) := mul_le_mul_of_nonneg_left he hB.le
    _ = 6 * v ^ 3 / B ^ 5 := by field_simp <;> ring

end Dobson.Coverage5GammaSmallStay
#print axioms Dobson.Coverage5GammaSmallStay.selected_quadratic_root
#print axioms Dobson.Coverage5GammaSmallStay.quadratic_root_error
#print axioms Dobson.Coverage5GammaSmallStay.inverse_log_coefficients
#print axioms Dobson.Coverage5GammaSmallStay.scaled_quadratic_root_error

#print axioms Dobson.Coverage5GammaSmallStay.threshold_first_inverse_log_limit
