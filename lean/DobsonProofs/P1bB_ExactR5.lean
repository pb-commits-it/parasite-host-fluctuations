import DobsonProofs.Coverage5GammaSmallStay

/-!
# Part B (`prop:exact-R5`): third inverse-log threshold coefficient and `p = 1` vanishing

Closes `part1:61` (`Coverage5GammaSmallStay.inverse_log_coefficients.h₃`) — noting that
`part1:60` was already closed in `Coverage5GammaSmallStay.lean` — by proving the actual
third inverse-log residual limit of the unique physical threshold `gammaThreshold p s` as
`s ↓ 0`, discharging `h₃` as a lemma for `thirdInverseLogCoefficient p`, and stating a
corollary under the positive-shape hypothesis applying `Coverage5GammaSmallStay.inverse_log_coefficients`
together with the `p = 1` vanishing of the `L⁻²` coefficient and the `s ↓ 0` endpoint limit.
Coverage correction, 2026-11-22: R5 is core; stronger Big-O/radical errors
and the exact zero-stay population endpoint remain outside credit.
-/

namespace Dobson.P1bB_ExactR5

open Filter Set MeasureTheory
open scoped Topology
open Dobson.Coverage2Exact Dobson.NoiseAnalyticity Dobson.Coverage5GammaSmallStay

private noncomputable def scoreNumerator (p κ : ℝ) : ℝ :=
  ∫ t in Ioi (0 : ℝ), -kummerShapeScore t * kummerWeight p κ t

private noncomputable def partitionError (p κ : ℝ) : ℝ :=
  kummerZ p κ + Real.log κ + gammaLogDerivative p + 2 * Real.eulerMascheroniConstant

private noncomputable def normalizedScore (p κ : ℝ) : ℝ :=
  scoreNumerator p κ / kummerZ p κ

private noncomputable def thresholdOffset (p : ℝ) : ℝ :=
  gammaLogDerivative p + Real.eulerMascheroniConstant

private theorem meanNormalizedScore_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerLog1p p κ * normalizedScore p κ)
      (𝓝[>] 0) (𝓝 (gammaTrigamma p / 2)) := by
  have hmZ : Tendsto (fun κ => kummerLog1p p κ / kummerZ p κ) (𝓝[>] 0) (𝓝 (1 / 2)) := by
    have ht := (kummerLog1p_log_scale_tendsto p hp).div
      (kummerZ_log_ratio_tendsto p hp) (by norm_num : (1 : ℝ) ≠ 0)
    simp only [div_one] at ht
    apply ht.congr'
    filter_upwards [self_mem_nhdsWithin,
      mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
    have hL : 0 < Real.log (1 / κ) :=
      Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
    dsimp only [Pi.div_apply]
    field_simp [hL.ne']
  have ht := hmZ.mul (kummerScoreNumerator_tendsto p hp)
  have he : (1 / 2 : ℝ) * gammaTrigamma p = gammaTrigamma p / 2 := by ring
  rw [he] at ht
  convert ht using 1
  funext κ
  dsimp [normalizedScore, scoreNumerator]
  ring

private theorem meanPartitionError_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerLog1p p κ * partitionError p κ) (𝓝[>] 0) (𝓝 0) := by
  have ht := (kummerZ_constant_log_rate p hp).mul (kummerLog1p_log_scale_tendsto p hp)
  simp only [zero_mul] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
  have hL : 0 < Real.log (1 / κ) :=
    Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
  dsimp only [partitionError]
  field_simp [hL.ne']
  rw [one_div, Real.log_inv]
  ring

private theorem meanSqScoreError_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => (kummerLog1p p κ) ^ 2 * (scoreNumerator p κ - gammaTrigamma p))
      (𝓝[>] 0) (𝓝 0) := by
  have ht := (scoreNumerator_log_sq_rate p hp).mul ((kummerLog1p_log_scale_tendsto p hp).pow 2)
  simp only [zero_mul] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
  have hL : 0 < Real.log (1 / κ) :=
    Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
  dsimp only [scoreNumerator, Pi.pow_apply]
  field_simp [hL.ne']

private theorem normalizedScore_quadratic (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    normalizedScore p κ *
        (2 * kummerLog1p p κ - 2 * thresholdOffset p - normalizedScore p κ + partitionError p κ) =
      scoreNumerator p κ := by
  have hi := kummerScoreNumerator_identity p κ hp hκ
  have he : 2 * kummerLog1p p κ - 2 * thresholdOffset p - normalizedScore p κ +
      partitionError p κ = kummerZ p κ := by
    dsimp [normalizedScore, scoreNumerator, thresholdOffset, partitionError]
    linarith
  rw [he]
  dsimp [normalizedScore]
  field_simp [(L3_kummer_moments p κ hp hκ).1.ne']

/-- The third inverse-log coefficient of the actual Kummer root correction as `κ ↓ 0`. -/
theorem kummer_third_inverse_log_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => (kummerLog1p p κ) ^ 3 *
      ((∫ t in Ioi (0 : ℝ), -kummerShapeScore t * kummerWeight p κ t) / kummerZ p κ) -
      gammaTrigamma p / 2 * (kummerLog1p p κ) ^ 2 -
      gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) / 2 *
        kummerLog1p p κ)
      (𝓝[>] 0) (𝓝 (gammaTrigamma p *
        (gammaLogDerivative p + Real.eulerMascheroniConstant) ^ 2 / 2 +
        (gammaTrigamma p) ^ 2 / 8)) := by
  have hA := (kummer_second_inverse_log_limit p hp).const_mul (thresholdOffset p)
  have hB := ((meanNormalizedScore_tendsto p hp).pow 2).div_const 2
  have hC := ((meanNormalizedScore_tendsto p hp).mul
    (meanPartitionError_tendsto_zero p hp)).div_const 2
  have hD := (meanSqScoreError_tendsto_zero p hp).div_const 2
  have ht := ((hA.add hB).sub hC).add hD
  have he : thresholdOffset p *
      (gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) / 2) +
      (gammaTrigamma p / 2) ^ 2 / 2 - (gammaTrigamma p / 2) * 0 / 2 + 0 / 2 =
      gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) ^ 2 / 2 +
        (gammaTrigamma p) ^ 2 / 8 := by
    dsimp [thresholdOffset]
    ring
  rw [he] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  have hq := normalizedScore_quadratic p κ hp hκ
  change thresholdOffset p *
      ((kummerLog1p p κ) ^ 2 * normalizedScore p κ - gammaTrigamma p / 2 * kummerLog1p p κ) +
    (kummerLog1p p κ * normalizedScore p κ) ^ 2 / 2 -
    (kummerLog1p p κ * normalizedScore p κ) * (kummerLog1p p κ * partitionError p κ) / 2 +
    (kummerLog1p p κ) ^ 2 * (scoreNumerator p κ - gammaTrigamma p) / 2 =
    (kummerLog1p p κ) ^ 3 * normalizedScore p κ -
      gammaTrigamma p / 2 * (kummerLog1p p κ) ^ 2 -
      gammaTrigamma p * thresholdOffset p / 2 * kummerLog1p p κ
  linear_combination -((kummerLog1p p κ) ^ 2 / 2) * hq

private theorem log_pow_mul_log_one_sub_tendsto_zero (n : ℕ) :
    Tendsto (fun s : ℝ => Real.log (1 / s) ^ n * Real.log (1 - s))
      (𝓝[>] 0) (𝓝 0) := by
  have hd : HasDerivAt (fun s : ℝ => Real.log (1 - s)) (-1) 0 := by
    convert (((hasDerivAt_id (0 : ℝ)).const_sub 1).log (by norm_num)) using 1 <;> norm_num
  have hr : Tendsto (fun s : ℝ => Real.log (1 - s) / s) (𝓝[>] 0) (𝓝 (-1)) := by
    simpa only [zero_add, sub_zero, Real.log_one, smul_eq_mul, div_eq_mul_inv, mul_comm] using
      hd.tendsto_slope_zero_right
  have hslog : Tendsto (fun s : ℝ => s * Real.log (1 / s) ^ n) (𝓝[>] 0) (𝓝 0) := by
    simpa only [one_mul, mul_one, add_zero, Function.comp_def, one_div, div_inv_eq_mul, mul_comm] using
      (Real.tendsto_pow_log_div_mul_add_atTop 1 0 n one_ne_zero).comp tendsto_inv_nhdsGT_zero
  have hh := hslog.mul hr
  simp only [zero_mul] at hh
  apply hh.congr'
  filter_upwards [self_mem_nhdsWithin] with s hs
  change 0 < s at hs
  field_simp

/-- The third inverse-log coefficient comes from the actual physical threshold `gammaThreshold p s`. -/
theorem threshold_third_inverse_log_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun s : ℝ => Real.log (1 / s) ^ 3 * (gammaLogLoss p - gammaThreshold p s) -
      gammaTrigamma p / 2 * Real.log (1 / s) ^ 2 -
      gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) / 2 *
        Real.log (1 / s)) (𝓝[>] 0)
      (𝓝 (gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) ^ 2 / 2 +
        gammaTrigamma p ^ 2 / 8)) := by
  have ht := ((kummer_third_inverse_log_limit p hp).comp
    (thresholdKappa_tendsto_zero p hp)).add (log_pow_mul_log_one_sub_tendsto_zero 3)
  simp only [add_zero] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with s hs hs1
  dsimp only [Function.comp_def]
  rw [threshold_mean p s hp hs hs1, threshold_deficit_identity p s hp hs hs1]
  ring

/-- The third coefficient defined by the actual threshold's cubic inverse-log residual limit. -/
noncomputable def thirdInverseLogCoefficient (p : ℝ) : ℝ :=
  limUnder (𝓝[>] (0 : ℝ)) (fun s =>
    Real.log (1 / s) ^ 3 * (gammaLogLoss p - gammaThreshold p s) -
      firstInverseLogCoefficient p * Real.log (1 / s) ^ 2 -
      secondInverseLogCoefficient p * Real.log (1 / s))

theorem thirdInverseLogCoefficient_eq (p : ℝ) (hp : 0 < p) :
    thirdInverseLogCoefficient p =
      gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) ^ 2 / 2 +
        gammaTrigamma p ^ 2 / 8 := by
  unfold thirdInverseLogCoefficient
  rw [firstInverseLogCoefficient_eq p hp, secondInverseLogCoefficient_eq p hp]
  exact (threshold_third_inverse_log_limit p hp).limUnder_eq

/-- Lemma closing `part1:61` (`Coverage5GammaSmallStay.inverse_log_coefficients.h₃`)
for the actual third inverse-log threshold coefficient `thirdInverseLogCoefficient p`. -/
lemma inverse_log_third_balance (p : ℝ) (hp : 0 < p) :
    2 * thirdInverseLogCoefficient p -
      2 * (gammaLogDerivative p + Real.eulerMascheroniConstant) *
        secondInverseLogCoefficient p -
      firstInverseLogCoefficient p ^ 2 = 0 := by
  rw [firstInverseLogCoefficient_eq p hp, secondInverseLogCoefficient_eq p hp,
    thirdInverseLogCoefficient_eq p hp]
  ring

/-- For positive shape, the proved `inverse_log_third_balance` completes
the three inverse-log coefficients, the limit `D(s,p) → ℓ(p)` as `s ↓ 0`,
and vanishing of the second coefficient at `p = 1`. This is the credited
coefficient core, without the stronger manuscript Big-O errors. -/
theorem inverse_log_coefficients_unconditional (p : ℝ) (hp : 0 < p) :
    (firstInverseLogCoefficient p = gammaTrigamma p / 2 ∧
      secondInverseLogCoefficient p =
        gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) / 2 ∧
      thirdInverseLogCoefficient p =
        gammaTrigamma p * (gammaLogDerivative p + Real.eulerMascheroniConstant) ^ 2 / 2 +
          gammaTrigamma p ^ 2 / 8) ∧
    secondInverseLogCoefficient 1 = 0 ∧
    Tendsto (gammaThreshold p) (𝓝[>] 0) (𝓝 (gammaLogLoss p)) := by
  have hmain := inverse_log_coefficients p (thirdInverseLogCoefficient p) hp
    (inverse_log_third_balance p hp)
  have hp1 : secondInverseLogCoefficient 1 = 0 := by
    rw [secondInverseLogCoefficient_eq 1 zero_lt_one]
    have hψ1 : gammaLogDerivative 1 = -Real.eulerMascheroniConstant := by
      dsimp [gammaLogDerivative]
      rw [Real.hasDerivAt_Gamma_one.deriv, Real.Gamma_one, div_one]
    rw [hψ1]
    ring
  have hlim0 : Tendsto (fun s : ℝ => gammaLogLoss p - gammaThreshold p s) (𝓝[>] 0) (𝓝 0) := by
    have hL : Tendsto (fun s : ℝ => 1 / Real.log (1 / s)) (𝓝[>] 0) (𝓝 0) := by
      have htop : Tendsto (fun s : ℝ => Real.log (1 / s)) (𝓝[>] 0) atTop := by
        simpa only [one_div, Function.comp_def] using
          Real.tendsto_log_atTop.comp tendsto_inv_nhdsGT_zero
      exact tendsto_const_nhds.div_atTop htop
    have hh := (threshold_first_inverse_log_limit p hp).mul hL
    simp only [mul_zero] at hh
    apply hh.congr'
    filter_upwards [self_mem_nhdsWithin,
      mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with s hs hs1
    have hLpos : 0 < Real.log (1 / s) :=
      Real.log_pos ((lt_div_iff₀ hs).mpr (by simpa using hs1))
    field_simp [hLpos.ne']
  have hlim : Tendsto (gammaThreshold p) (𝓝[>] 0) (𝓝 (gammaLogLoss p)) := by
    have hh := tendsto_const_nhds (x := gammaLogLoss p).sub hlim0
    simpa only [sub_zero, sub_sub_cancel] using hh
  exact ⟨hmain, hp1, hlim⟩

end Dobson.P1bB_ExactR5

#print axioms Dobson.P1bB_ExactR5.inverse_log_third_balance
#print axioms Dobson.P1bB_ExactR5.inverse_log_coefficients_unconditional
