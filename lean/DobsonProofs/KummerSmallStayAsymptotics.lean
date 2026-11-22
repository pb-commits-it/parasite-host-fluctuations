import DobsonProofs.Coverage2Exact
import DobsonProofs.GammaDigammaSeries
import Mathlib.Analysis.Calculus.LHopital

/-!
# Actual small-kappa partition constant and inverse-log coefficients

The Gamma/digamma constant is obtained by a shape difference and one ordinary
integration by parts at shape one. Moment identities and l'Hopital give the
remainder rates. These are limits of the actual Kummer integrals.
-/
namespace Dobson.Coverage2Exact
open Filter Set MeasureTheory
open scoped Topology
open NoiseAnalyticity

private theorem weight_shape_hasDerivAt (p κ t : ℝ) (ht : 0 < t) :
    HasDerivAt (fun q => kummerWeight q κ t)
      (kummerShapeScore t * kummerWeight p κ t) p := by
  have h1t : 0 < 1+t := by linarith
  have h1 := ((hasDerivAt_id p).sub_const 1).const_rpow ht
  have h2 := (hasDerivAt_id p).neg.const_rpow h1t
  convert (h1.mul h2).mul_const (Real.exp (-κ*t)) using 1
  · rfl
  · dsimp [kummerShapeScore,kummerWeight]
    ring

private theorem shapeScore_nonpos (t : ℝ) (ht : 0 < t) : kummerShapeScore t ≤ 0 := by
  dsimp [kummerShapeScore]
  exact sub_nonpos.mpr (Real.log_le_log ht (by linarith))

private theorem weight_difference_bound (p q κ t : ℝ) (ht : 0 < t) :
    |kummerWeight p κ t - kummerWeight q κ t| ≤
      |p-q| * (-kummerShapeScore t * kummerWeight (min p q) κ t) := by
  let f := fun x => kummerWeight x κ t
  let B := -kummerShapeScore t * kummerWeight (min p q) κ t
  have hb : ∀ x ∈ Icc (min p q) (max p q), ‖deriv f x‖ ≤ B := by
    intro x hx
    rw [(weight_shape_hasDerivAt x κ t ht).deriv]
    dsimp [B]
    rw [abs_mul, abs_of_nonpos (shapeScore_nonpos t ht),
      abs_of_nonneg (weight_nonneg x κ t ht)]
    exact mul_le_mul_of_nonneg_left (weight_shape_antitone _ _ κ t ht hx.1)
      (neg_nonneg.mpr (shapeScore_nonpos t ht))
  have h := Convex.norm_image_sub_le_of_norm_deriv_le
    (fun x _ => (weight_shape_hasDerivAt x κ t ht).differentiableAt) hb
    (convex_Icc (min p q) (max p q))
    (show p ∈ Icc (min p q) (max p q) from ⟨min_le_left _ _,le_max_left _ _⟩)
    (show q ∈ Icc (min p q) (max p q) from ⟨min_le_right _ _,le_max_right _ _⟩)
  have hh : |kummerWeight q κ t-kummerWeight p κ t| ≤
      |q-p| *(-kummerShapeScore t*kummerWeight (min p q) κ t) := by
    simpa only [f,B,Real.norm_eq_abs,mul_comm] using h
  rwa [abs_sub_comm (kummerWeight q κ t) (kummerWeight p κ t),abs_sub_comm q p] at hh

private theorem zeroWeightDiff_integrable (p : ℝ) (hp : 0 < p) :
    IntegrableOn (fun t => kummerWeight p 0 t - kummerWeight 1 0 t) (Ioi 0) := by
  have hi := (scoreZero_integrable (min p 1) (lt_min hp zero_lt_one)).const_mul |p-1|
  refine hi.mono' (((weight_cont p 0).sub (weight_cont 1 0)).aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
  exact weight_difference_bound p 1 0 t ht

private noncomputable def zeroWeightDifference (p : ℝ) : ℝ :=
  ∫ t in Ioi (0 : ℝ), kummerWeight p 0 t - kummerWeight 1 0 t

private theorem zeroWeightDifference_deriv (p : ℝ) (hp : 0 < p) :
    HasDerivAt zeroWeightDifference (-gammaTrigamma p) p := by
  let F := fun q t => kummerWeight q 0 t - kummerWeight 1 0 t
  let F' := fun q t => kummerShapeScore t * kummerWeight q 0 t
  let B := fun t => -kummerShapeScore t * kummerWeight (p/2) 0 t
  have hF : ∀ q, AEStronglyMeasurable (F q) (volume.restrict (Ioi 0)) := by
    intro q
    exact ((weight_cont q 0).sub (weight_cont 1 0)).aestronglyMeasurable measurableSet_Ioi
  have hF' : AEStronglyMeasurable (F' p) (volume.restrict (Ioi 0)) :=
    (shapeScore_cont.mul (weight_cont p 0)).aestronglyMeasurable measurableSet_Ioi
  have hb : ∀ᵐ t ∂volume.restrict (Ioi (0 : ℝ)), ∀ q ∈ Ioi (p/2), ‖F' q t‖ ≤ B t := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    intro q hq
    dsimp [F',B]
    rw [abs_mul, abs_of_nonpos (shapeScore_nonpos t ht), abs_of_nonneg (weight_nonneg q 0 t ht)]
    exact mul_le_mul_of_nonneg_left (weight_shape_antitone _ _ 0 t ht hq.le)
      (neg_nonneg.mpr (shapeScore_nonpos t ht))
  have hd : ∀ᵐ t ∂volume.restrict (Ioi (0 : ℝ)),
      ∀ q ∈ Ioi (p/2), HasDerivAt (F · t) (F' q t) q := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    intro q _
    exact (weight_shape_hasDerivAt q 0 t ht).sub_const (kummerWeight 1 0 t)
  have h := (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := volume.restrict (Ioi 0)) (s := Ioi (p/2))
    (Ioi_mem_nhds (by linarith)) (Filter.Eventually.of_forall hF)
    (zeroWeightDiff_integrable p hp) hF' hb (scoreZero_integrable (p/2) (by linarith)) hd).2
  have he : (∫ t in Ioi (0 : ℝ), F' p t) = -gammaTrigamma p := by
    have hh := scoreZero_integral p hp
    simp_rw [neg_mul] at hh
    rw [integral_neg] at hh
    dsimp only [F']
    linarith
  rwa [he] at h

private theorem zeroWeightDifference_eq (p : ℝ) (hp : 0 < p) :
    zeroWeightDifference p = -gammaLogDerivative p - Real.eulerMascheroniConstant := by
  let f := fun q => zeroWeightDifference q + gammaLogDerivative q
  have hd : ∀ q ∈ Ioi (0 : ℝ), HasDerivAt f 0 q := by
    intro q hq
    have hh := (zeroWeightDifference_deriv q hq).fun_add
      (GammaDigammaSeries.gammaLogDerivative_hasDerivAt q hq)
    simpa only [neg_add_cancel] using hh
  have he := isOpen_Ioi.is_const_of_deriv_eq_zero (convex_Ioi (0 : ℝ)).isPreconnected
    (fun q hq => (hd q hq).differentiableAt.differentiableWithinAt)
    (fun q hq => (hd q hq).deriv) hp (show (0 : ℝ) < 1 by norm_num)
  have hψ1 : gammaLogDerivative 1 = -Real.eulerMascheroniConstant := by
    dsimp [gammaLogDerivative]
    rw [Real.hasDerivAt_Gamma_one.deriv,Real.Gamma_one,div_one]
  have hC1 : zeroWeightDifference 1 = 0 := by simp [zeroWeightDifference]
  dsimp only [f] at he
  rw [hC1,hψ1] at he
  linarith

/-- The finite shape-dependent part of the logarithmically divergent Kummer
partition integral is identified with the actual digamma difference. -/
theorem kummerZ_shape_difference_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerZ p κ - kummerZ 1 κ) (𝓝[>] 0)
      (𝓝 (-gammaLogDerivative p - Real.eulerMascheroniConstant)) := by
  let F := fun κ t => (kummerWeight p 0 t - kummerWeight 1 0 t) * Real.exp (-κ*t)
  have hm : ∀ᶠ κ in 𝓝[>] (0 : ℝ), AEStronglyMeasurable (F κ) (volume.restrict (Ioi 0)) := by
    filter_upwards with κ
    exact (((weight_cont p 0).sub (weight_cont 1 0)).mul (by fun_prop)).aestronglyMeasurable
      measurableSet_Ioi
  have hb : ∀ᶠ κ in 𝓝[>] (0 : ℝ), ∀ᵐ t ∂volume.restrict (Ioi 0),
      ‖F κ t‖ ≤ |kummerWeight p 0 t - kummerWeight 1 0 t| := by
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    dsimp [F]
    rw [abs_mul,abs_of_pos (Real.exp_pos _)]
    have he : Real.exp (-κ*t) ≤ 1 := Real.exp_le_one_iff.mpr (by have hκ0 : 0 < κ := hκ; have ht0 : 0 < t := ht; nlinarith)
    simpa using mul_le_mul_of_nonneg_left he (abs_nonneg (kummerWeight p 0 t-kummerWeight 1 0 t))
  have hl : ∀ᵐ t ∂volume.restrict (Ioi 0),
      Tendsto (fun κ => F κ t) (𝓝[>] 0) (𝓝 (kummerWeight p 0 t-kummerWeight 1 0 t)) := by
    filter_upwards with t
    have hc : ContinuousAt (fun κ => F κ t) 0 := by dsimp [F]; fun_prop
    simpa only [F,neg_zero,zero_mul,Real.exp_zero,mul_one] using hc.tendsto.mono_left nhdsWithin_le_nhds
  have ht := tendsto_integral_filter_of_dominated_convergence
    (fun t => |kummerWeight p 0 t-kummerWeight 1 0 t|) hm hb
    (zeroWeightDiff_integrable p hp).norm hl
  change Tendsto _ _ (𝓝 (zeroWeightDifference p)) at ht
  rw [zeroWeightDifference_eq p hp] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  have he : F κ =ᵐ[volume.restrict (Ioi 0)] (fun t => kummerWeight p κ t-kummerWeight 1 κ t) := by
    filter_upwards with t
    dsimp [F,kummerWeight]
    simp only [neg_zero,zero_mul,Real.exp_zero,mul_one]
    ring
  rw [integral_congr_ae he,integral_sub (L3_kummer_moments p κ hp hκ).2.1
    (L3_kummer_moments 1 κ (by norm_num) hκ).2.1]
  rfl


private theorem shiftLogGamma_integrable (κ : ℝ) (hκ : 0 < κ) :
    IntegrableOn (fun u => Real.log (κ+u)*Real.exp (-u)) (Ioi 0) := by
  have h1 : IntegrableOn (fun u : ℝ => Real.log (1+κ⁻¹*u)*Real.exp (-u)) (Ioi 0) := by
    simpa only [sub_self, Real.rpow_zero, one_mul, mul_one, neg_mul] using
      gamma_log_affine_integrable 1 1 κ⁻¹ (by norm_num) (by norm_num) (inv_pos.mpr hκ)
  have h2 : IntegrableOn (fun u : ℝ => Real.log κ*Real.exp (-u)) (Ioi 0) := by
    change Integrable (fun u : ℝ => Real.log κ*Real.exp (-u)) (volume.restrict (Ioi 0))
    simpa using (integrableOn_exp_mul_Ioi (by norm_num : (-1 : ℝ) < 0) 0).const_mul (Real.log κ)
  apply (h2.add h1).congr_fun _ measurableSet_Ioi
  intro u hu
  have he : κ+u = κ*(1+κ⁻¹*u) := by field_simp <;> ring
  dsimp only [Pi.add_apply]
  rw [he,Real.log_mul hκ.ne' (by have hu0 : 0 < u := hu; positivity)]
  ring

private theorem kummerZ_one_ibp (κ : ℝ) (hκ : 0 < κ) :
    kummerZ 1 κ = κ * (∫ t in Ioi (0 : ℝ), Real.log (1+t)*Real.exp (-κ*t)) := by
  let f := fun t : ℝ => Real.log (1+t)*Real.exp (-κ*t)
  let f' := fun t : ℝ => kummerWeight 1 κ t - κ*(Real.log (1+t)*Real.exp (-κ*t))
  have hfi : IntegrableOn f (Ioi 0) := by
    simpa only [f,sub_self,Real.rpow_zero,one_mul] using
      gamma_log_affine_integrable 1 κ 1 (by norm_num) hκ (by norm_num)
  have hf'i : IntegrableOn f' (Ioi 0) :=
    (L3_kummer_moments 1 κ (by norm_num) hκ).2.1.sub (hfi.const_mul κ)
  have hc : ContinuousOn f (Ici 0) := by
    intro t ht
    have ht1 : 0 < 1+t := by have ht0 : 0 ≤ t := ht; linarith
    apply ContinuousAt.continuousWithinAt
    dsimp [f]
    fun_prop (disch := positivity)
  have hd : ∀ t ∈ Ioi (0 : ℝ), HasDerivAt f (f' t) t := by
    intro t ht
    have ht0 : 0 < t := ht
    have ht1 : 0 < 1+t := by linarith
    have hlog := ((hasDerivAt_id t).const_add 1).log ht1.ne'
    have hexp := ((hasDerivAt_id t).const_mul (-κ)).exp
    convert hlog.mul hexp using 1
    · rfl
    · dsimp [f',kummerWeight]
      rw [sub_self,Real.rpow_zero,one_mul,Real.rpow_neg_one]
      ring
  have hlim : Tendsto f atTop (𝓝 0) := by
    have ht := tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero 1 κ hκ
    rw [show (fun t : ℝ => t ^ (1:ℝ)*Real.exp (-κ*t)) = (fun t => t*Real.exp (-κ*t)) by
      funext t; rw [Real.rpow_one]] at ht
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds ht
    · filter_upwards [Ioi_mem_atTop (0 : ℝ)] with t ht
      change 0 < t at ht
      exact mul_nonneg (Real.log_nonneg (by linarith)) (Real.exp_pos _).le
    · filter_upwards [Ioi_mem_atTop (0 : ℝ)] with t ht
      change 0 < t at ht
      have hl := Real.log_le_sub_one_of_pos (by linarith : 0 < 1+t)
      dsimp [f]
      exact mul_le_mul_of_nonneg_right (by linarith : Real.log (1+t) ≤ t) (Real.exp_pos _).le
  have hh := integral_Ioi_of_hasDerivAt_of_tendsto (hc 0 (by simp)) hd hf'i hlim
  have hz : f 0 = 0 := by simp [f]
  rw [hz,sub_zero] at hh
  dsimp only [f'] at hh
  rw [integral_sub (L3_kummer_moments 1 κ (by norm_num) hκ).2.1 (hfi.const_mul κ),
    integral_const_mul] at hh
  change kummerZ 1 κ - κ*(∫ t in Ioi (0 : ℝ), f t) = 0 at hh
  linarith

private theorem kummerZ_one_shift (κ : ℝ) (hκ : 0 < κ) :
    kummerZ 1 κ + Real.log κ = ∫ u in Ioi (0 : ℝ), Real.log (κ+u)*Real.exp (-u) := by
  rw [kummerZ_one_ibp κ hκ]
  have hc := integral_comp_mul_left_Ioi'
    (fun t : ℝ => Real.log (1+t)*Real.exp (-κ*t)) 0 (inv_pos.mpr hκ)
  simp only [mul_zero,smul_eq_mul] at hc
  have hcomp : (fun u : ℝ => Real.log (1+κ⁻¹*u)*Real.exp (-κ*(κ⁻¹*u))) =ᵐ[volume.restrict (Ioi 0)]
      (fun u => Real.log (κ+u)*Real.exp (-u)-Real.log κ*Real.exp (-u)) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
    have hu0 : 0 < u := hu
    have he : 1+κ⁻¹*u = (κ+u)/κ := by field_simp <;> ring
    have hexp : -κ*(κ⁻¹*u) = -u := by field_simp
    rw [he,hexp,Real.log_div (by positivity : κ+u ≠ 0) hκ.ne']
    ring
  rw [integral_congr_ae hcomp] at hc
  have hi : IntegrableOn (fun u : ℝ => Real.exp (-u)) (Ioi 0) := by
    simpa using integrableOn_exp_mul_Ioi (by norm_num : (-1 : ℝ) < 0) 0
  rw [integral_sub (shiftLogGamma_integrable κ hκ) (hi.const_mul (Real.log κ)),integral_const_mul] at hc
  have he : (∫ u in Ioi (0 : ℝ), Real.exp (-u)) = 1 := by
    simpa using integral_exp_mul_Ioi (by norm_num : (-1 : ℝ) < 0) 0
  rw [he,mul_one] at hc
  field_simp [hκ.ne'] at hc
  simp only [neg_mul] at hc ⊢
  linarith

private theorem shiftLogGamma_tendsto :
    Tendsto (fun κ : ℝ => ∫ u in Ioi (0 : ℝ), Real.log (κ+u)*Real.exp (-u))
      (𝓝[>] 0) (𝓝 (-Real.eulerMascheroniConstant)) := by
  let F := fun κ u : ℝ => Real.log (κ+u)*Real.exp (-u)
  let B := fun u : ℝ => (|Real.log u|+u+1)*Real.exp (-u)
  have hlog : IntegrableOn (fun u : ℝ => |Real.log u| *Real.exp (-u)) (Ioi 0) := by
    change Integrable (fun u : ℝ => |Real.log u| * Real.exp (-u)) (volume.restrict (Ioi 0))
    have hi := (gamma_log_integrable 1 1 (by norm_num) (by norm_num)).norm
    simpa only [sub_self,Real.rpow_zero,one_mul,neg_mul,Real.norm_eq_abs,abs_mul,
      abs_of_pos (Real.exp_pos _)] using hi
  have hlin : IntegrableOn (fun u : ℝ => u*Real.exp (-u)) (Ioi 0) := by
    simpa only [sub_self,Real.rpow_zero,one_mul,neg_mul] using
      gamma_first_integrable 1 1 (by norm_num) (by norm_num)
  have hexp : IntegrableOn (fun u : ℝ => Real.exp (-u)) (Ioi 0) := by
    simpa using integrableOn_exp_mul_Ioi (by norm_num : (-1 : ℝ) < 0) 0
  have hB : IntegrableOn B (Ioi 0) := by
    convert (hlog.add hlin).add hexp using 1
    funext u
    dsimp [B,Pi.add_apply]
    ring
  have hm : ∀ᶠ κ in 𝓝[>] (0 : ℝ), AEStronglyMeasurable (F κ) (volume.restrict (Ioi 0)) := by
    filter_upwards with κ
    exact (by dsimp [F]; fun_prop : Measurable (F κ)).aestronglyMeasurable
  have hb : ∀ᶠ κ in 𝓝[>] (0 : ℝ), ∀ᵐ u ∂volume.restrict (Ioi 0), ‖F κ u‖ ≤ B u := by
    filter_upwards [self_mem_nhdsWithin,
      mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
    have hu0 : 0 < u := hu
    have hκ0 : 0 < κ := hκ
    have hκ1' : κ < 1 := hκ1
    have hlo := Real.log_le_log hu0 (by linarith : u ≤ κ+u)
    have hhi := Real.log_le_sub_one_of_pos (by positivity : 0 < κ+u)
    have ha : |Real.log (κ+u)| ≤ |Real.log u|+u+1 := by
      rw [abs_le]
      constructor
      · have hh := neg_abs_le (Real.log u); linarith
      · have hh := abs_nonneg (Real.log u); linarith
    dsimp [F,B]
    rw [abs_mul,abs_of_pos (Real.exp_pos _)]
    exact mul_le_mul_of_nonneg_right ha (Real.exp_pos _).le
  have hl : ∀ᵐ u ∂volume.restrict (Ioi 0), Tendsto (fun κ => F κ u) (𝓝[>] 0)
      (𝓝 (Real.log u*Real.exp (-u))) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
    have hc : ContinuousAt (fun κ => F κ u) 0 := by
      dsimp [F]
      apply ContinuousAt.mul _ continuousAt_const
      apply ContinuousAt.log (by fun_prop)
      simpa using ne_of_gt hu
    simpa only [F,zero_add] using hc.tendsto.mono_left nhdsWithin_le_nhds
  have ht := tendsto_integral_filter_of_dominated_convergence B hm hb hB hl
  have he := Real.eulerMascheroniConstant_eq_neg_integral_log
  have he' : (∫ u in Ioi (0 : ℝ), Real.log u*Real.exp (-u)) = -Real.eulerMascheroniConstant := by linarith
  rwa [he'] at ht

/-- The actual additive partition constant, with Gamma digamma and Euler's
constant, follows from a difference integral and an integration by parts. -/
theorem kummerZ_constant_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerZ p κ - Real.log (1/κ)) (𝓝[>] 0)
      (𝓝 (-gammaLogDerivative p - 2*Real.eulerMascheroniConstant)) := by
  have h1 : Tendsto (fun κ : ℝ => kummerZ 1 κ+Real.log κ) (𝓝[>] 0)
      (𝓝 (-Real.eulerMascheroniConstant)) := by
    apply shiftLogGamma_tendsto.congr'
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    exact (kummerZ_one_shift κ hκ).symm
  have ht := (kummerZ_shape_difference_tendsto p hp).add h1
  have he : -gammaLogDerivative p-Real.eulerMascheroniConstant + -Real.eulerMascheroniConstant =
      -gammaLogDerivative p-2*Real.eulerMascheroniConstant := by ring
  rw [he] at ht
  apply ht.congr'
  filter_upwards with κ
  rw [one_div,Real.log_inv]
  ring


private noncomputable def scoreNumerator (p κ : ℝ) : ℝ :=
  ∫ t in Ioi (0 : ℝ), -kummerShapeScore t*kummerWeight p κ t

private theorem scoreNumerator_deriv (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    HasDerivAt (scoreNumerator p)
      (∫ t in Ioi (0 : ℝ), t*kummerShapeScore t*kummerWeight p κ t) κ := by
  have hi : IntegrableOn (fun t => -kummerShapeScore t*kummerWeight p κ t) (Ioi 0) := by
    apply (shapeScore_integrable p κ hp hκ).neg.congr_fun _ measurableSet_Ioi
    intro t _
    dsimp only [Pi.neg_apply]
    ring
  have hdom : IntegrableOn (fun t => ‖-kummerShapeScore t‖*t*kummerWeight p (κ/2) t) (Ioi 0) := by
    simpa only [norm_neg,mul_comm] using shapeScore_t_abs_integrable p (κ/2) hp (by linarith)
  have hd := raw_moment_deriv p κ hκ (fun t => -kummerShapeScore t) shapeScore_cont.neg hi hdom
  convert hd using 1
  · rfl
  · apply setIntegral_congr_fun measurableSet_Ioi
    intro t _
    ring

private theorem scoreNumerator_deriv_bounds (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    -kummerZ p κ ≤ deriv (scoreNumerator p) κ ∧ deriv (scoreNumerator p) κ ≤ 0 := by
  rw [(scoreNumerator_deriv p κ hp hκ).deriv]
  have hi : IntegrableOn (fun t => t*kummerShapeScore t*kummerWeight p κ t) (Ioi 0) := by
    have hh := (shapeScore_t_abs_integrable p κ hp hκ)
    apply hh.mono' (((continuousOn_id.mul shapeScore_cont).mul (weight_cont p κ)).aestronglyMeasurable measurableSet_Ioi)
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    change 0 < t at ht
    simp only [Pi.mul_apply,id_eq,Real.norm_eq_abs,abs_mul,abs_of_pos ht,abs_of_nonneg (weight_nonneg p κ t ht)]
    exact le_rfl
  constructor
  · rw [kummerZ,← integral_neg]
    apply integral_mono_ae (L3_kummer_moments p κ hp hκ).2.1.neg hi
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have h := mul_le_mul_of_nonneg_right (t_shapeScore_bounds t ht).1 (weight_nonneg p κ t ht)
    simpa only [neg_one_mul,Pi.neg_apply] using h
  · apply integral_nonpos_of_ae
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    exact mul_nonpos_of_nonpos_of_nonneg (t_shapeScore_bounds t ht).2 (weight_nonneg p κ t ht)

private theorem sqrt_log_tendsto_zero :
    Tendsto (fun κ : ℝ => Real.sqrt κ*Real.log (1/κ)) (𝓝[>] 0) (𝓝 0) := by
  have h := (tendsto_log_mul_rpow_nhdsGT_zero (r := (1/2:ℝ)) (by norm_num)).neg
  simpa only [Real.sqrt_eq_rpow,one_div,Real.log_inv,neg_zero,mul_neg,neg_mul,mul_comm] using h

private theorem sqrt_log_sq_tendsto_zero :
    Tendsto (fun κ : ℝ => Real.sqrt κ*(Real.log (1/κ))^2) (𝓝[>] 0) (𝓝 0) := by
  have h := (tendsto_log_mul_rpow_nhdsGT_zero (r := (1/4:ℝ)) (by norm_num)).pow 2
  simp only [zero_pow (by norm_num : (2:ℕ) ≠ 0)] at h
  apply h.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  change 0 < κ at hκ
  rw [one_div κ,Real.log_inv,neg_sq,mul_pow]
  have he : (κ^(1/4:ℝ))^2 = Real.sqrt κ := by
    rw [← Real.rpow_natCast,← Real.rpow_mul hκ.le,Real.sqrt_eq_rpow]
    norm_num
  rw [he]
  ring

private theorem kummerZ_sqrt_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerZ p κ*Real.sqrt κ) (𝓝[>] 0) (𝓝 0) := by
  have ht := (kummerZ_log_ratio_tendsto p hp).mul sqrt_log_tendsto_zero
  simp only [mul_zero] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0:ℝ)<1 by norm_num))] with κ hκ hκ1
  have hL : 0 < Real.log (1/κ) := Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
  field_simp [hL.ne']

private theorem quotient_sqrt_tendsto_zero (f f' : ℝ → ℝ)
    (hd : ∀ᶠ κ in 𝓝[>] (0:ℝ), HasDerivAt f (f' κ) κ)
    (hf : Tendsto f (𝓝[>] 0) (𝓝 0))
    (hdf : Tendsto (fun κ => f' κ*Real.sqrt κ) (𝓝[>] 0) (𝓝 0)) :
    Tendsto (fun κ => f κ/Real.sqrt κ) (𝓝[>] 0) (𝓝 0) := by
  have hs : ∀ᶠ κ in 𝓝[>] (0:ℝ), HasDerivAt Real.sqrt (1/(2*Real.sqrt κ)) κ := by
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    change 0 < κ at hκ
    exact Real.hasDerivAt_sqrt (ne_of_gt hκ)
  have hn : ∀ᶠ κ in 𝓝[>] (0:ℝ), 1/(2*Real.sqrt κ) ≠ 0 := by
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    change 0 < κ at hκ
    have hκ0 : 0 < κ := hκ
    positivity
  have hsl : Tendsto Real.sqrt (𝓝[>] (0:ℝ)) (𝓝 0) := by
    simpa using (Real.continuous_sqrt.tendsto (0 : ℝ)).mono_left nhdsWithin_le_nhds
  have hr : Tendsto (fun κ => f' κ/(1/(2*Real.sqrt κ))) (𝓝[>] 0) (𝓝 0) := by
    convert hdf.const_mul 2 using 1
    · funext κ
      simp only [div_eq_mul_inv,mul_inv_rev,inv_inv,inv_one,mul_one]
      ring
    · ring
  exact HasDerivAt.lhopital_zero_nhdsGT hd hs hn hf hsl hr

private theorem scoreNumerator_sqrt_rate (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => (scoreNumerator p κ-gammaTrigamma p)/Real.sqrt κ) (𝓝[>] 0) (𝓝 0) := by
  apply quotient_sqrt_tendsto_zero (fun κ => scoreNumerator p κ-gammaTrigamma p)
    (fun κ => deriv (scoreNumerator p) κ)
  · filter_upwards [self_mem_nhdsWithin] with κ hκ
    change 0 < κ at hκ
    exact ((scoreNumerator_deriv p κ hp hκ).differentiableAt.hasDerivAt.sub_const _)
  · simpa only [scoreNumerator,sub_self] using (kummerScoreNumerator_tendsto p hp).sub_const (gammaTrigamma p)
  · have hlo := (kummerZ_sqrt_tendsto_zero p hp).neg
    simp only [neg_zero] at hlo
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' hlo tendsto_const_nhds
    · filter_upwards [self_mem_nhdsWithin] with κ hκ
      change 0 < κ at hκ
      have hh := mul_le_mul_of_nonneg_right (scoreNumerator_deriv_bounds p κ hp hκ).1 (Real.sqrt_nonneg κ)
      simpa only [neg_mul] using hh
    · filter_upwards [self_mem_nhdsWithin] with κ hκ
      change 0 < κ at hκ
      exact mul_nonpos_of_nonpos_of_nonneg (scoreNumerator_deriv_bounds p κ hp hκ).2 (Real.sqrt_nonneg κ)

private theorem kummerN0_deriv (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    HasDerivAt (kummerN0 p) (kummerN0 p κ-p*kummerZ p κ) κ := by
  have h := L3_kummer_moments p κ hp hκ
  have hhalf := L3_kummer_moments p (κ/2) hp (by linarith)
  have hdom : IntegrableOn (fun t => ‖t‖*t*kummerWeight p (κ/2) t) (Ioi 0) := by
    apply hhalf.2.2.2.1.congr_fun _ measurableSet_Ioi
    intro t ht
    dsimp only
    rw [Real.norm_of_nonneg ht.le]
    ring
  have hd := raw_moment_deriv p κ hκ (fun t => t) continuousOn_id h.2.2.1 hdom
  have hdi : (∫ t in Ioi (0 : ℝ), -(t*t*kummerWeight p κ t)) =
      -(∫ t in Ioi (0 : ℝ), t^2*kummerWeight p κ t) := by
    rw [← integral_neg]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro t _
    ring
  rw [hdi] at hd
  have hh := (hasDerivAt_id κ).mul hd
  have hibp := h.2.2.2.2.2.1
  dsimp [kummerExpect] at hibp
  field_simp [h.1.ne'] at hibp
  convert hh using 1
  · rfl
  · dsimp [kummerN0]
    nlinarith

private theorem kummerN0_sqrt_rate (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => (1-kummerN0 p κ)/Real.sqrt κ) (𝓝[>] 0) (𝓝 0) := by
  apply quotient_sqrt_tendsto_zero (fun κ => 1-kummerN0 p κ)
    (fun κ => p*kummerZ p κ-kummerN0 p κ)
  · filter_upwards [self_mem_nhdsWithin] with κ hκ
    change 0 < κ at hκ
    convert (kummerN0_deriv p κ hp hκ).const_sub 1 using 1 <;> ring
  · simpa only [sub_self] using (kummerN0_tendsto_one p hp).const_sub 1
  · have hs : Tendsto Real.sqrt (𝓝[>] (0:ℝ)) (𝓝 0) := by
      simpa using (Real.continuous_sqrt.tendsto (0 : ℝ)).mono_left nhdsWithin_le_nhds
    have ht := ((kummerZ_sqrt_tendsto_zero p hp).const_mul p).sub ((kummerN0_tendsto_one p hp).mul hs)
    simp only [mul_zero,sub_self] at ht
    convert ht using 1
    funext κ
    ring

private noncomputable def partitionError (p κ : ℝ) : ℝ :=
  kummerZ p κ+Real.log κ+gammaLogDerivative p+2*Real.eulerMascheroniConstant

private theorem partitionError_sqrt_rate (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => partitionError p κ/Real.sqrt κ) (𝓝[>] 0) (𝓝 0) := by
  apply quotient_sqrt_tendsto_zero (partitionError p) (fun κ => (1-kummerN0 p κ)/κ)
  · filter_upwards [self_mem_nhdsWithin] with κ hκ
    change 0 < κ at hκ
    have hh := (((kummerZ_deriv_actual p κ hp hκ).fun_add (Real.hasDerivAt_log hκ.ne')).add_const
      (gammaLogDerivative p)).add_const (2*Real.eulerMascheroniConstant)
    convert hh using 1
    · rfl
    · dsimp [kummerN0]
      field_simp [hκ.ne']
      <;> ring
  · have ht := (kummerZ_constant_tendsto p hp).add_const
      (gammaLogDerivative p+2*Real.eulerMascheroniConstant)
    have he : -gammaLogDerivative p-2*Real.eulerMascheroniConstant +
        (gammaLogDerivative p+2*Real.eulerMascheroniConstant) = 0 := by ring
    rw [he] at ht
    convert ht using 1
    funext κ
    dsimp [partitionError]
    rw [one_div,Real.log_inv]
    ring
  · apply (kummerN0_sqrt_rate p hp).congr'
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    change 0 < κ at hκ
    have hs : 0 < Real.sqrt κ := Real.sqrt_pos.mpr hκ
    have he := Real.sq_sqrt hκ.le
    field_simp [hκ.ne',hs.ne']
    rw [he]

/-- The score remainder vanishes after two logarithmic rescalings. -/
theorem scoreNumerator_log_sq_rate (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => ((∫ t in Ioi (0 : ℝ), -kummerShapeScore t*kummerWeight p κ t)-gammaTrigamma p)*
      (Real.log (1/κ))^2) (𝓝[>] 0) (𝓝 0) := by
  have ht := (scoreNumerator_sqrt_rate p hp).mul sqrt_log_sq_tendsto_zero
  simp only [mul_zero] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  change 0 < κ at hκ
  have hs : 0 < Real.sqrt κ := Real.sqrt_pos.mpr hκ
  dsimp only [Pi.mul_apply,scoreNumerator]
  field_simp [hs.ne'] <;> ring

/-- The partition constant has an error smaller than the inverse logarithm. -/
theorem kummerZ_constant_log_rate (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => (kummerZ p κ-Real.log (1/κ)+gammaLogDerivative p+2*Real.eulerMascheroniConstant)*
      Real.log (1/κ)) (𝓝[>] 0) (𝓝 0) := by
  have ht := (partitionError_sqrt_rate p hp).mul sqrt_log_tendsto_zero
  simp only [mul_zero] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  change 0 < κ at hκ
  have hs : 0 < Real.sqrt κ := Real.sqrt_pos.mpr hκ
  dsimp only [Pi.mul_apply,partitionError]
  rw [one_div,Real.log_inv]
  field_simp [hs.ne'] <;> ring


private noncomputable def normalizedScore (p κ : ℝ) := scoreNumerator p κ/kummerZ p κ
private noncomputable def thresholdOffset (p : ℝ) := gammaLogDerivative p+Real.eulerMascheroniConstant

private theorem inverse_log_tendsto_zero :
    Tendsto (fun κ : ℝ => 1/Real.log (1/κ)) (𝓝[>] 0) (𝓝 0) := by
  have hL : Tendsto (fun κ : ℝ => Real.log (1/κ)) (𝓝[>] 0) atTop := by
    simpa only [one_div,Function.comp_def] using Real.tendsto_log_atTop.comp tendsto_inv_nhdsGT_zero
  exact tendsto_const_nhds.div_atTop hL

private theorem normalizedScore_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (normalizedScore p) (𝓝[>] 0) (𝓝 0) := by
  exact (kummerScoreNumerator_tendsto p hp).div_atTop (kummerZ_tendsto_atTop p hp)

private theorem meanNormalizedScore_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerLog1p p κ*normalizedScore p κ) (𝓝[>] 0) (𝓝 (gammaTrigamma p/2)) := by
  have hmZ : Tendsto (fun κ => kummerLog1p p κ/kummerZ p κ) (𝓝[>] 0) (𝓝 (1/2)) := by
    have ht := (kummerLog1p_log_scale_tendsto p hp).div (kummerZ_log_ratio_tendsto p hp) (by norm_num : (1:ℝ) ≠ 0)
    simp only [div_one] at ht
    apply ht.congr'
    filter_upwards [self_mem_nhdsWithin,
      mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0:ℝ)<1 by norm_num))] with κ hκ hκ1
    have hL : 0 < Real.log (1/κ) := Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
    dsimp only [Pi.div_apply]
    field_simp [hL.ne']
  have ht := hmZ.mul (kummerScoreNumerator_tendsto p hp)
  have he : (1/2:ℝ)*gammaTrigamma p = gammaTrigamma p/2 := by ring
  rw [he] at ht
  convert ht using 1
  funext κ
  dsimp [normalizedScore,scoreNumerator]
  ring

private theorem partitionError_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (partitionError p) (𝓝[>] 0) (𝓝 0) := by
  have ht := (kummerZ_constant_tendsto p hp).add_const (gammaLogDerivative p+2*Real.eulerMascheroniConstant)
  have he : -gammaLogDerivative p-2*Real.eulerMascheroniConstant +
      (gammaLogDerivative p+2*Real.eulerMascheroniConstant) = 0 := by ring
  rw [he] at ht
  convert ht using 1
  funext κ
  dsimp [partitionError]
  rw [one_div,Real.log_inv]
  ring

private theorem meanPartitionError_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerLog1p p κ*partitionError p κ) (𝓝[>] 0) (𝓝 0) := by
  have ht := (kummerZ_constant_log_rate p hp).mul (kummerLog1p_log_scale_tendsto p hp)
  simp only [zero_mul] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0:ℝ)<1 by norm_num))] with κ hκ hκ1
  have hL : 0 < Real.log (1/κ) := Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
  dsimp only [partitionError]
  field_simp [hL.ne']
  rw [one_div,Real.log_inv]
  ring

private theorem meanSqScoreError_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => (kummerLog1p p κ)^2*(scoreNumerator p κ-gammaTrigamma p)) (𝓝[>] 0) (𝓝 0) := by
  have ht := (scoreNumerator_log_sq_rate p hp).mul ((kummerLog1p_log_scale_tendsto p hp).pow 2)
  simp only [zero_mul] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0:ℝ)<1 by norm_num))] with κ hκ hκ1
  have hL : 0 < Real.log (1/κ) := Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
  dsimp only [scoreNumerator,Pi.pow_apply]
  field_simp [hL.ne']
  <;> ring

private theorem meanScoreError_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerLog1p p κ*(scoreNumerator p κ-gammaTrigamma p)) (𝓝[>] 0) (𝓝 0) := by
  have ht := ((scoreNumerator_log_sq_rate p hp).mul (kummerLog1p_log_scale_tendsto p hp)).mul inverse_log_tendsto_zero
  simp only [zero_mul] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0:ℝ)<1 by norm_num))] with κ hκ hκ1
  have hL : 0 < Real.log (1/κ) := Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
  dsimp only [scoreNumerator]
  field_simp [hL.ne']
  <;> ring

private theorem normalizedScore_quadratic (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    normalizedScore p κ*(2*kummerLog1p p κ-2*thresholdOffset p-normalizedScore p κ+partitionError p κ) =
      scoreNumerator p κ := by
  have hi := kummerScoreNumerator_identity p κ hp hκ
  have he : 2*kummerLog1p p κ-2*thresholdOffset p-normalizedScore p κ+partitionError p κ = kummerZ p κ := by
    dsimp [normalizedScore,scoreNumerator,thresholdOffset,partitionError]
    linarith
  rw [he]
  dsimp [normalizedScore]
  field_simp [(L3_kummer_moments p κ hp hκ).1.ne']

/-- The second inverse-log coefficient of the actual Kummer root correction. -/
theorem kummer_second_inverse_log_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => (kummerLog1p p κ)^2*
      ((∫ t in Ioi (0 : ℝ), -kummerShapeScore t*kummerWeight p κ t)/kummerZ p κ) -
      gammaTrigamma p/2*kummerLog1p p κ) (𝓝[>] 0)
      (𝓝 (gammaTrigamma p*(gammaLogDerivative p+Real.eulerMascheroniConstant)/2)) := by
  have hA := (meanNormalizedScore_tendsto p hp).const_mul (thresholdOffset p)
  have hB := ((meanNormalizedScore_tendsto p hp).mul (normalizedScore_tendsto_zero p hp)).div_const 2
  have hC := ((meanNormalizedScore_tendsto p hp).mul (partitionError_tendsto_zero p hp)).div_const 2
  have hD := (meanScoreError_tendsto_zero p hp).div_const 2
  have ht := ((hA.add hB).sub hC).add hD
  have he : thresholdOffset p*(gammaTrigamma p/2)+(gammaTrigamma p/2)*0/2-
      (gammaTrigamma p/2)*0/2+0/2 = gammaTrigamma p*(gammaLogDerivative p+Real.eulerMascheroniConstant)/2 := by
    dsimp [thresholdOffset]
    ring
  rw [he] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  have hq := normalizedScore_quadratic p κ hp hκ
  change thresholdOffset p*(kummerLog1p p κ*normalizedScore p κ)+
    (kummerLog1p p κ*normalizedScore p κ)*normalizedScore p κ/2-
    (kummerLog1p p κ*normalizedScore p κ)*partitionError p κ/2+
    kummerLog1p p κ*(scoreNumerator p κ-gammaTrigamma p)/2 =
    (kummerLog1p p κ)^2*normalizedScore p κ-gammaTrigamma p/2*kummerLog1p p κ
  linear_combination -(kummerLog1p p κ/2)*hq



end Dobson.Coverage2Exact
#print axioms Dobson.Coverage2Exact.kummer_second_inverse_log_limit
