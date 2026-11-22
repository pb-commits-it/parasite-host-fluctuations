import DobsonProofs.Coverage5GammaLongStay

/-!
# Part B (`prop:exact-R6`): second long-stay threshold jet and normalized limit

Closes `part1:58` (`Coverage5GammaLongStay.threshold_jet_coefficients.h₃`) by proving
the cubic threshold-composition equation and the actual quadratic residual limit of
the unique physical threshold `longThreshold p ε = gammaThreshold p (1 - ε)`, then
instantiating `Coverage5GammaLongStay.threshold_jet_coefficients` without owed hypotheses
and proving the normalized long-stay limit as `τ = 1 / (1 - s) → ∞`.
Coverage correction, 2026-11-22: R6 is core; stronger Big-O remainder orders
are outside credit.
-/

namespace Dobson.P1bB_ExactR6

open Set Filter MeasureTheory
open scoped Topology
open Dobson.GammaFirstJet Dobson.GammaHigher Dobson.GammaThresholdJets
open Dobson.Coverage2Exact Dobson.Coverage5GammaSmallStay Dobson.Coverage5GammaLongStay
open Dobson.NoiseAnalyticity

theorem exp_taylor_three (c x : ℝ) :
    taylorWithinEval (fun t : ℝ => Real.exp (c * t)) 3 univ 0 x =
      1 + c * x + c ^ 2 * x ^ 2 / 2 + c ^ 3 * x ^ 3 / 6 := by
  have hd : deriv (fun t : ℝ => Real.exp (c * t)) 0 = c := by
    simpa using (((hasDerivAt_id (0 : ℝ)).const_mul c).exp).deriv
  rw [taylorWithinEval_succ (n := 2), taylorWithinEval_succ (n := 1),
    taylorWithinEval_succ (n := 0)]
  simp [iteratedDerivWithin_univ, iteratedDeriv_exp_const_mul, taylor_within_zero_eval, hd]
  ring

theorem exp_third_limit (c : ℝ) :
    Tendsto (fun x : ℝ => (Real.exp (c * x) - 1 - c * x - c ^ 2 * x ^ 2 / 2) / x ^ 3)
      (𝓝[≠] 0) (𝓝 (c ^ 3 / 6)) := by
  have hc : ContDiff ℝ 3 (fun x : ℝ => Real.exp (c * x)) := by fun_prop
  have ht := Real.taylor_tendsto convex_univ (mem_univ (0 : ℝ)) hc.contDiffOn
  rw [nhdsWithin_univ] at ht
  have hh := (ht.mono_left (show (𝓝[≠] (0 : ℝ)) ≤ 𝓝 (0 : ℝ) from nhdsWithin_le_nhds)).add_const
    (c ^ 3 / 6)
  simp only [zero_add] at hh
  apply Tendsto.congr' _ hh
  filter_upwards [self_mem_nhdsWithin] with x hx
  rw [exp_taylor_three]
  simp only [sub_zero]
  have hx0 : x ≠ 0 := hx
  field_simp [hx0]
  ring

theorem exp_neg_third_limit :
    Tendsto (fun x : ℝ => (1 - Real.exp (-x) - x + x ^ 2 / 2) / x ^ 3)
      (𝓝[>] 0) (𝓝 ((1 : ℝ) / 6)) := by
  have ht := ((exp_third_limit (-1)).mono_left (nhdsGT_le_nhdsNE 0)).neg
  convert ht using 1 <;> norm_num
  ext x
  ring

theorem log_taylor_two (x : ℝ) :
    taylorWithinEval Real.log 2 (Ioi 0) 1 x = (x - 1) - (x - 1) ^ 2 / 2 := by
  have h1 : ∀ y ∈ Ioi (0 : ℝ), iteratedDerivWithin 1 Real.log (Ioi 0) y = y⁻¹ := by
    intro y hy
    simp only [iteratedDerivWithin_one]
    exact (Real.hasDerivAt_log hy.ne').hasDerivWithinAt.derivWithin
      (isOpen_Ioi.uniqueDiffWithinAt hy)
  have h2 : iteratedDerivWithin 2 Real.log (Ioi 0) 1 = -1 := by
    rw [iteratedDerivWithin_succ (n := 1), derivWithin_congr h1 (h1 1 (by norm_num))]
    have hh : HasDerivWithinAt (fun x : ℝ => x⁻¹) (-1 / (1 : ℝ) ^ 2) (Ioi 0) 1 :=
      ((hasDerivAt_id (1 : ℝ)).inv (by norm_num)).hasDerivWithinAt
    convert hh.derivWithin (isOpen_Ioi.uniqueDiffWithinAt (by norm_num : (1 : ℝ) ∈ Ioi (0 : ℝ)))
      using 1
    norm_num
  rw [taylorWithinEval_succ (n := 1), taylorWithinEval_succ (n := 0), h2, h1 1 (by norm_num)]
  simp [taylor_within_zero_eval]
  ring

theorem log_second_limit :
    Tendsto (fun x : ℝ => (Real.log (1 + x) - x) / x ^ 2) (𝓝[≠] 0) (𝓝 (-(1 : ℝ) / 2)) := by
  have hc : ContDiffOn ℝ 2 Real.log (Ioi 0) :=
    contDiffOn_id.log (fun x hx => hx.ne')
  have ht := Real.taylor_tendsto (convex_Ioi (0 : ℝ)) (by norm_num : (1 : ℝ) ∈ Ioi (0 : ℝ)) hc
  rw [nhdsWithin_eq_nhds.mpr (Ioi_mem_nhds (by norm_num : (0 : ℝ) < 1))] at ht
  have hs : Tendsto (fun x : ℝ => 1 + x) (𝓝[≠] 0) (𝓝 1) := by
    convert (tendsto_const_nhds.add (tendsto_id.mono_left nhdsWithin_le_nhds) :
      Tendsto (fun x : ℝ => 1 + x) (𝓝[≠] 0) (𝓝 (1 + 0))) using 1
    norm_num
  have hh := (ht.comp hs).sub_const ((1 : ℝ) / 2)
  simp only [zero_sub, neg_div] at hh ⊢
  apply Tendsto.congr' _ hh
  filter_upwards [self_mem_nhdsWithin] with x hx
  dsimp only [Function.comp_def]
  rw [log_taylor_two]
  have hx0 : x ≠ 0 := hx
  field_simp [hx0]
  ring

noncomputable def epsilonSecond (p : ℝ) : ℝ := -p * (4 * p + 1) / 2
noncomputable def epsilonThird (p : ℝ) : ℝ := p * (30 * p ^ 2 + 21 * p + 4) / 6

theorem epsilon_cubic_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ =>
        (ε - p * longQ p ε - epsilonSecond p * (longQ p ε) ^ 2) / (longQ p ε) ^ 3)
      (𝓝[>] 0) (𝓝 (epsilonThird p)) := by
  have he := exp_neg_third_limit.comp mean_at_threshold_tendsto
  have hm := (scaledLogMean_ratio p hp).comp (longQ_tendsto p hp)
  have h2 := (second_limit p hp).comp (longQ_tendsto p hp)
  have h3 := (third_limit p hp).comp (longQ_tendsto p hp)
  have ht := ((he.mul (hm.pow 3)).sub ((h2.mul (hm.add_const p)).div_const 2)).add h3
  convert Tendsto.congr' ?_ ht using 1
  · dsimp [epsilonThird]
    ring
  filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with ε hε
  dsimp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.div_apply, Function.comp_def]
  rw [long_mean p ε hp hε.1 hε.2]
  rw [neg_neg, Real.exp_log (show 0 < 1 - ε by linarith [hε.2]), secondJet_eq p hp]
  have hq : longQ p ε ≠ 0 := (longQ_pos p ε hp hε.1 hε.2).ne'
  have hm0 : Real.log (1 - ε) ≠ 0 := (Real.log_neg (by linarith [hε.2]) (by linarith [hε.1])).ne
  dsimp [epsilonSecond]
  field_simp [hq, hm0]
  ring

theorem longQ_second_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ => (longQ p ε - ε / p) / ε ^ 2) (𝓝[>] 0)
      (𝓝 (-epsilonSecond p / p ^ 3)) := by
  have ht := ((epsilon_quadratic_limit p hp).mul ((longQ_ratio p hp).pow 2)).div_const (-p)
  convert Tendsto.congr' ?_ ht using 1
  · dsimp [epsilonSecond]
    field_simp
    ring
  filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with ε hε
  field_simp [hp.ne', hε.1.ne', (longQ_pos p ε hp hε.1 hε.2).ne']
  ring

theorem thresholdRatio_second_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ => (thresholdRatio p ε - 1 - (epsilonSecond p / p ^ 2) * ε) / ε ^ 2)
      (𝓝[>] 0) (𝓝 (epsilonThird p / p ^ 3 - (epsilonSecond p) ^ 2 / p ^ 4)) := by
  have ht := (((epsilon_cubic_limit p hp).mul ((longQ_ratio p hp).pow 2)).div_const p).add
    ((longQ_second_limit p hp).const_mul (epsilonSecond p / p))
  convert Tendsto.congr' ?_ ht using 1
  · ring
  filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with ε hε
  have htr : thresholdRatio p ε = ε / (p * longQ p ε) := by
    unfold thresholdRatio
    simp [hε.1.ne']
  rw [htr]
  field_simp [hp.ne', hε.1.ne', (longQ_pos p ε hp hε.1 hε.2).ne']
  ring

theorem thresholdRatio_slope (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ => (thresholdRatio p ε - 1) / ε) (𝓝[>] 0)
      (𝓝 (epsilonSecond p / p ^ 2)) := by
  have ht := thresholdRatio_deriv p hp
  rw [hasDerivWithinAt_iff_tendsto_slope' (by simp : (0 : ℝ) ∉ Ioi (0 : ℝ))] at ht
  simp only [slope_fun_def_field, thresholdRatio_zero, sub_zero] at ht
  convert ht using 1
  dsimp [epsilonSecond]
  field_simp [hp.ne']

theorem thresholdRatio_sub_one_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ => thresholdRatio p ε - 1) (𝓝[>] 0) (𝓝[≠] 0) := by
  apply tendsto_nhdsWithin_iff.mpr
  constructor
  · have ht := ((thresholdRatio_deriv p hp).continuousWithinAt.tendsto).sub_const 1
    simpa only [thresholdRatio_zero, sub_self] using ht
  · have hc : epsilonSecond p / p ^ 2 < 0 := by
      dsimp [epsilonSecond]
      exact div_neg_of_neg_of_pos (div_neg_of_neg_of_pos (by nlinarith) (by norm_num))
        (sq_pos_of_pos hp)
    have he := (thresholdRatio_slope p hp).eventually (gt_mem_nhds hc)
    filter_upwards [he, self_mem_nhdsWithin] with ε hε hε0
    change 0 < ε at hε0
    change thresholdRatio p ε - 1 ≠ 0
    exact (neg_of_div_neg_left hε hε0.le).ne

theorem thresholdLog_second_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ => (Real.log (thresholdRatio p ε) - (epsilonSecond p / p ^ 2) * ε) / ε ^ 2)
      (𝓝[>] 0)
      (𝓝 (epsilonThird p / p ^ 3 - 3 * (epsilonSecond p) ^ 2 / (2 * p ^ 4))) := by
  have he := log_second_limit.comp (thresholdRatio_sub_one_tendsto p hp)
  have ht := (he.mul ((thresholdRatio_slope p hp).pow 2)).add (thresholdRatio_second_limit p hp)
  convert Tendsto.congr' ?_ ht using 1
  · ring
  filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num),
    (thresholdRatio_sub_one_tendsto p hp).eventually self_mem_nhdsWithin] with ε hε hR
  dsimp only [Pi.add_apply, Pi.mul_apply, Function.comp_def]
  have hR0 : thresholdRatio p ε - 1 ≠ 0 := hR
  rw [show 1 + (thresholdRatio p ε - 1) = thresholdRatio p ε by ring]
  field_simp [hε.1.ne', hR0]
  ring

theorem log_one_sub_second_limit :
    Tendsto (fun ε : ℝ => (Real.log (1 - ε) + ε) / ε ^ 2) (𝓝[>] 0) (𝓝 (-(1 : ℝ) / 2)) := by
  have hn : Tendsto (fun ε : ℝ => -ε) (𝓝[>] 0) (𝓝[≠] 0) := by
    apply tendsto_nhdsWithin_iff.mpr
    constructor
    · simpa using (tendsto_id.mono_left nhdsWithin_le_nhds :
        Tendsto (fun ε : ℝ => ε) (𝓝[>] 0) (𝓝 0)).neg
    · filter_upwards [self_mem_nhdsWithin] with ε hε
      exact neg_ne_zero.mpr (ne_of_gt hε)
  simpa only [Function.comp_def, sub_neg_eq_add, even_two, Even.neg_pow, ← sub_eq_add_neg] using
    log_second_limit.comp hn

/-- The second long-stay threshold coefficient defined directly from the physical root. -/
noncomputable def secondThresholdJet (p : ℝ) : ℝ :=
  limUnder (𝓝[>] (0 : ℝ)) (fun ε => (longThreshold p ε - firstThresholdJet p * ε) / ε ^ 2)

/-- Actual quadratic residual limit of the unique physical threshold as `ε = 1 - s ↓ 0`. -/
theorem secondThreshold_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ => (longThreshold p ε - firstThresholdJet p * ε) / ε ^ 2)
      (𝓝[>] 0) (𝓝 (-(12 * p + 7) / (24 * p ^ 2))) := by
  have ht := (thresholdLog_second_limit p hp).neg.add (log_one_sub_second_limit.const_mul 2)
  convert Tendsto.congr' ?_ ht using 1
  · dsimp [epsilonSecond, epsilonThird]
    field_simp
    ring
  filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with ε hε
  rw [threshold_log_identity p ε hp hε.1 hε.2, firstThresholdJet_eq p hp]
  dsimp only [Pi.add_apply, Pi.neg_apply, epsilonSecond]
  field_simp [hp.ne', hε.1.ne']
  ring

theorem secondThresholdJet_eq (p : ℝ) (hp : 0 < p) :
    secondThresholdJet p = -(12 * p + 7) / (24 * p ^ 2) :=
  (secondThreshold_limit p hp).limUnder_eq

/-- Lemma closing `part1:58` (`Coverage5GammaLongStay.threshold_jet_coefficients.h₃`)
for the actual physical threshold coefficient `secondThresholdJet p`. -/
lemma threshold_jet_cubic_balance (p : ℝ) (hp : 0 < p) :
    p * ((secondThresholdJet p + firstThresholdJet p ^ 2 / 2 +
      2 * firstThresholdJet p + 3) / p) -
      p * (3 * p + 1) / 2 * (2 * (1 / p) * ((firstThresholdJet p + 2) / p)) +
      p * (10 * p ^ 2 + 9 * p + 2) / 3 * (1 / p) ^ 3 - 1 / 3 = 0 := by
  rw [firstThresholdJet_eq p hp, secondThresholdJet_eq p hp]
  field_simp [hp.ne']
  ring

/-- For positive shape, `threshold_jet_cubic_balance` supplies the last
matching equation for the actual two threshold coefficients. -/
theorem threshold_jet_coefficients_unconditional (p : ℝ) (hp : 0 < p) :
    firstThresholdJet p = 1 / (2 * p) ∧
      secondThresholdJet p = -(12 * p + 7) / (24 * p ^ 2) :=
  threshold_jet_coefficients p (secondThresholdJet p) hp (threshold_jet_cubic_balance p hp)

/-- Scaled normalized long-stay ratio limit `D(1 - ε) / (ε / (2p(1 + ε))) → 1` and its
first-order correction `-7 / (12p)` as `ε ↓ 0`. -/
theorem normalized_long_stay_limits (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ =>
        (longThreshold p ε / (ε / (2 * p * (1 + ε))) - 1) / ε)
      (𝓝[>] 0) (𝓝 (-7 / (12 * p))) ∧
    Tendsto (fun ε : ℝ =>
        longThreshold p ε / (ε / (2 * p * (1 + ε))))
      (𝓝[>] 0) (𝓝 1) := by
  have hjet := secondThreshold_limit p hp
  rw [firstThresholdJet_eq p hp] at hjet
  have h1 : Tendsto (fun ε : ℝ => 2 * p * (1 + ε)) (𝓝[>] 0) (𝓝 (2 * p)) := by
    have hc : ContinuousAt (fun ε : ℝ => 2 * p * (1 + ε)) 0 := by fun_prop
    simpa using hc.tendsto.mono_left nhdsWithin_le_nhds
  have hcorr : Tendsto (fun ε : ℝ =>
      (longThreshold p ε / (ε / (2 * p * (1 + ε))) - 1) / ε)
      (𝓝[>] 0) (𝓝 (-7 / (12 * p))) := by
    have ht := (hjet.mul h1).add_const 1
    have hval : -(12 * p + 7) / (24 * p ^ 2) * (2 * p) + 1 = -7 / (12 * p) := by
      field_simp [hp.ne']
      ring
    rw [hval] at ht
    apply ht.congr'
    filter_upwards [self_mem_nhdsWithin] with ε hε
    change 0 < ε at hε
    have hε1 : 1 + ε ≠ 0 := by linarith
    field_simp [hp.ne', hε.ne', hε1]
    ring
  refine ⟨hcorr, ?_⟩
  have hratio := (firstThreshold_limit p hp).mul h1
  have hval1 : 1 / (2 * p) * (2 * p) = (1 : ℝ) := by field_simp [hp.ne']
  rw [hval1] at hratio
  apply hratio.congr'
  filter_upwards [self_mem_nhdsWithin] with ε hε
  change 0 < ε at hε
  have hε1 : 1 + ε ≠ 0 := by linarith
  field_simp [hp.ne', hε.ne', hε1]

/-- Core corollary for `prop:exact-R6`: as `ε = 1 - s ↓ 0`,
`D(1 - ε, p) = ε / (2p) - (12p + 7) ε² / (24p²) + o(ε²)`.
The theorem gives the exact first and second residual limits, the normalized
`1 / [2p(1 + τ)]` ratio limit, and its `-7 / (12p)` scaled correction as
`τ → ∞`. It does not prove an `O(ε³)` remainder. -/
theorem exact_R6_unconditional (p : ℝ) (hp : 0 < p) :
    firstThresholdJet p = 1 / (2 * p) ∧
    secondThresholdJet p = -(12 * p + 7) / (24 * p ^ 2) ∧
    Tendsto (fun ε : ℝ => gammaThreshold p (1 - ε) / ε)
      (𝓝[>] 0) (𝓝 (1 / (2 * p))) ∧
    Tendsto (fun ε : ℝ =>
        (gammaThreshold p (1 - ε) - ε / (2 * p)) / ε ^ 2)
      (𝓝[>] 0) (𝓝 (-(12 * p + 7) / (24 * p ^ 2))) ∧
    Tendsto (fun τ : ℝ =>
        gammaThreshold p (1 - 1 / τ) / (1 / (2 * p * (1 + τ))))
      atTop (𝓝 1) ∧
    Tendsto (fun τ : ℝ =>
        τ * (gammaThreshold p (1 - 1 / τ) / (1 / (2 * p * (1 + τ))) - 1))
      atTop (𝓝 (-7 / (12 * p))) := by
  obtain ⟨hA, hB⟩ := threshold_jet_coefficients_unconditional p hp
  obtain ⟨hslope, hnorm⟩ := normalized_long_stay_limits p hp
  have hinv : Tendsto (fun τ : ℝ => 1 / τ) atTop (𝓝[>] 0) := by
    apply tendsto_nhdsWithin_iff.mpr
    refine ⟨?_, ?_⟩
    · simpa [one_div] using tendsto_inv_atTop_zero
    · filter_upwards [eventually_gt_atTop (0 : ℝ)] with τ hτ
      exact one_div_pos.mpr hτ
  have hjet2 : Tendsto (fun ε : ℝ =>
      (gammaThreshold p (1 - ε) - ε / (2 * p)) / ε ^ 2)
      (𝓝[>] 0) (𝓝 (-(12 * p + 7) / (24 * p ^ 2))) := by
    have hh := secondThreshold_limit p hp
    rw [hA] at hh
    simpa [longThreshold, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hh
  have hnorm_tau : Tendsto (fun τ : ℝ =>
      gammaThreshold p (1 - 1 / τ) / (1 / (2 * p * (1 + τ))))
      atTop (𝓝 1) := by
    apply (hnorm.comp hinv).congr'
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with τ hτ
    dsimp only [Function.comp_def, longThreshold]
    congr 1
    field_simp [hτ.ne']
    ring
  have hslope_tau : Tendsto (fun τ : ℝ =>
      τ * (gammaThreshold p (1 - 1 / τ) / (1 / (2 * p * (1 + τ))) - 1))
      atTop (𝓝 (-7 / (12 * p))) := by
    apply (hslope.comp hinv).congr'
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with τ hτ
    dsimp only [Function.comp_def, longThreshold]
    have h1 : 1 / τ / (2 * p * (1 + 1 / τ)) = 1 / (2 * p * (1 + τ)) := by
      field_simp [hτ.ne']
      ring
    rw [h1]
    field_simp [hτ.ne']
  exact ⟨hA, hB, firstThreshold_limit p hp, hjet2, hnorm_tau, hslope_tau⟩

end Dobson.P1bB_ExactR6

#print axioms Dobson.P1bB_ExactR6.threshold_jet_cubic_balance
#print axioms Dobson.P1bB_ExactR6.threshold_jet_coefficients_unconditional
#print axioms Dobson.P1bB_ExactR6.exact_R6_unconditional
