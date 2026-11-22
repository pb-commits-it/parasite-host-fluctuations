import Mathlib
import DobsonProofs.Coverage2Exact
import DobsonProofs.JensenGap

open MeasureTheory Set Filter
open scoped Topology

namespace Dobson.KummerGap
open NoiseAnalyticity Coverage2Exact

/-- Unnormalized density of W=log(1+T), obtained by the actual substitution. -/
noncomputable def logWeight (p κ w : ℝ) : ℝ :=
  Real.exp w * kummerWeight p κ (Real.exp w - 1)

private theorem exp_sub_one_image : (fun w : ℝ => Real.exp w - 1) '' Ioi 0 = Ioi 0 := by
  ext t
  constructor
  · rintro ⟨w, hw, rfl⟩
    exact sub_pos.mpr (Real.one_lt_exp_iff.mpr hw)
  · intro ht
    have ht0 : 0 < t := ht
    refine ⟨Real.log (1 + t), Real.log_pos (by linarith : 1 < 1 + t), ?_⟩
    dsimp only
    rw [Real.exp_log (by linarith : 0 < 1 + t)]
    ring

private theorem exp_sub_one_inj : InjOn (fun w : ℝ => Real.exp w - 1) (Ioi 0) := by
  intro x _ y _ h
  exact Real.exp_injective (by linarith)

/-- Change of variables for every test function, including all moments used below. -/
theorem integral_logWeight (p κ : ℝ) (f : ℝ → ℝ) :
    (∫ t in Ioi (0 : ℝ), f (Real.log (1 + t)) * kummerWeight p κ t) =
      ∫ w in Ioi (0 : ℝ), f w * logWeight p κ w := by
  have h := integral_image_eq_integral_abs_deriv_smul measurableSet_Ioi
    (fun w _ => ((Real.hasDerivAt_exp w).sub_const 1).hasDerivWithinAt)
    exp_sub_one_inj (fun t => f (Real.log (1 + t)) * kummerWeight p κ t)
  rw [exp_sub_one_image] at h
  rw [h]
  apply setIntegral_congr_fun measurableSet_Ioi
  intro w _
  dsimp only
  rw [abs_of_pos (Real.exp_pos w), smul_eq_mul]
  have he : 1 + (Real.exp w - 1) = Real.exp w := by ring
  rw [he, Real.log_exp]
  dsimp [logWeight]
  ring

/-- The same substitution preserves integrability. -/
theorem integrable_logWeight_iff (p κ : ℝ) (f : ℝ → ℝ) :
    IntegrableOn (fun w => f w * logWeight p κ w) (Ioi 0) ↔
      IntegrableOn (fun t => f (Real.log (1 + t)) * kummerWeight p κ t) (Ioi 0) := by
  have h := integrableOn_image_iff_integrableOn_abs_deriv_smul measurableSet_Ioi
    (fun w _ => ((Real.hasDerivAt_exp w).sub_const 1).hasDerivWithinAt)
    exp_sub_one_inj (fun t => f (Real.log (1 + t)) * kummerWeight p κ t)
  rw [exp_sub_one_image] at h
  rw [h]
  apply integrableOn_congr_fun _ measurableSet_Ioi
  intro w _
  dsimp only
  rw [abs_of_pos (Real.exp_pos w), smul_eq_mul]
  have he : 1 + (Real.exp w - 1) = Real.exp w := by ring
  rw [he, Real.log_exp]
  dsimp [logWeight]
  ring

/-- The elementary reflected-density formula used by the manuscript. -/
theorem logWeight_formula (p κ w : ℝ) (hw : 0 < w) :
    logWeight p κ w = (1 - Real.exp (-w)) ^ (p - 1) *
      Real.exp (-κ * (Real.exp w - 1)) := by
  have he : 0 < Real.exp w := Real.exp_pos w
  have hx : 0 < Real.exp w - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr hw)
  have ha : 1 + (Real.exp w - 1) = Real.exp w := by ring
  have hrat : 1 - Real.exp (-w) = (Real.exp w - 1) / Real.exp w := by
    rw [Real.exp_neg]
    field_simp
  dsimp [logWeight, kummerWeight]
  rw [ha, hrat, Real.div_rpow hx.le he.le]
  have hpow : Real.exp w * (Real.exp w) ^ (-p) = ((Real.exp w) ^ (p - 1))⁻¹ := by
    calc Real.exp w * (Real.exp w) ^ (-p) = (Real.exp w) ^ (1 : ℝ) * (Real.exp w) ^ (-p) := by rw [Real.rpow_one]
      _ = (Real.exp w) ^ (1 - p) := by rw [← Real.rpow_add he]; congr 1
      _ = ((Real.exp w) ^ (p - 1))⁻¹ := by rw [show 1 - p = -(p - 1) by ring, Real.rpow_neg he.le]
  rw [div_eq_mul_inv]
  calc _ = (Real.exp w - 1) ^ (p - 1) *
      (Real.exp w * (Real.exp w) ^ (-p)) * Real.exp (-κ * (Real.exp w - 1)) := by ring
    _ = _ := by rw [hpow]

/-- The reflected logarithmic ratio after its positive hyperbolic factor. -/
noncomputable def reflectedH (m x : ℝ) : ℝ :=
  (Real.log (1 - Real.exp (-m) * Real.exp (-x)) -
    Real.log (1 - Real.exp (-m) * Real.exp x)) / (2 * Real.sinh x)

private theorem reflectedH_hasSum (m x : ℝ) (hx : 0 < x) (hxm : x < m) :
    HasSum (fun n : ℕ => (Real.exp (-m)) ^ (n + 1) / ((n + 1 : ℕ) : ℝ) *
      (Real.sinh (((n + 1 : ℕ) : ℝ) * x) / Real.sinh x)) (reflectedH m x) := by
  have ha : |Real.exp (-m) * Real.exp x| < 1 := by
    rw [← Real.exp_add, abs_of_pos (Real.exp_pos _), Real.exp_lt_one_iff]
    linarith
  have hb : |Real.exp (-m) * Real.exp (-x)| < 1 := by
    rw [← Real.exp_add, abs_of_pos (Real.exp_pos _), Real.exp_lt_one_iff]
    linarith
  have hA := Real.hasSum_pow_div_log_of_abs_lt_one ha
  have hB := Real.hasSum_pow_div_log_of_abs_lt_one hb
  have hsum := (hA.sub hB).div_const (2 * Real.sinh x)
  convert hsum using 1
  · funext n
    rw [mul_pow, mul_pow, Real.sinh_eq]
    simp_rw [← Real.exp_nat_mul]
    have hn : ((n + 1 : ℕ) : ℝ) = (n : ℝ) + 1 := by norm_cast
    have he : ((n + 1 : ℕ) : ℝ) * (-x) = -(((n + 1 : ℕ) : ℝ) * x) := by ring
    rw [he, hn]
    field_simp
    <;> ring
  · dsimp [reflectedH]
    ring

/-- Strict increase follows from the proved hyperbolic-ratio monotonicity,
with the n=2 term giving strictness and all other coefficients positive. -/
theorem reflectedH_strictMonoOn (m : ℝ) : StrictMonoOn (reflectedH m) (Ioo 0 m) := by
  intro x hx y hy hxy
  have hsx := reflectedH_hasSum m x hx.1 hx.2
  have hsy := reflectedH_hasSum m y hy.1 hy.2
  rw [← hsx.tsum_eq, ← hsy.tsum_eq]
  apply Summable.tsum_lt_tsum (i := 1) _ _ hsx.summable hsy.summable
  · intro n
    exact mul_le_mul_of_nonneg_left
      (JensenGap.sinh_ratio_le (n + 1) (by omega) x y hx.1 hxy) (by positivity)
  · apply mul_lt_mul_of_pos_left
    · exact JensenGap.S1 2 (by omega) x y hx.1 hxy
    · positivity

/-- Normalized W density, with its actual support made explicit. -/
noncomputable def logDensity (p κ w : ℝ) : ℝ :=
  (Ioi 0).indicator (fun w => (kummerZ p κ)⁻¹ * logWeight p κ w) w

private theorem logDensity_test_eq (p κ : ℝ) (f : ℝ → ℝ) :
    (fun w => f w * logDensity p κ w) =
      (Ioi 0).indicator (fun w => (kummerZ p κ)⁻¹ * (f w * logWeight p κ w)) := by
  funext w
  by_cases hw : 0 < w
  · simp [logDensity, hw, mul_left_comm]
  · simp [logDensity, hw]

theorem integral_logDensity (p κ : ℝ) (f : ℝ → ℝ) :
    (∫ w, f w * logDensity p κ w) =
      kummerExpect p κ (fun t => f (Real.log (1 + t))) := by
  rw [logDensity_test_eq, integral_indicator measurableSet_Ioi, integral_const_mul,
    ← integral_logWeight]
  rfl

private theorem integrable_logDensity (p κ : ℝ) (f : ℝ → ℝ)
    (hi : IntegrableOn (fun t => f (Real.log (1 + t)) * kummerWeight p κ t) (Ioi 0)) :
    Integrable (fun w => f w * logDensity p κ w) := by
  rw [logDensity_test_eq, integrable_indicator_iff measurableSet_Ioi]
  exact ((integrable_logWeight_iff p κ f).mpr hi).const_mul _

theorem logDensity_pos (p κ w : ℝ) (hp : 0 < p) (hκ : 0 < κ) (hw : 0 < w) :
    0 < logDensity p κ w := by
  have hz := (L3_kummer_moments p κ hp hκ).1
  have he := Real.exp_pos w
  have ht : 0 < Real.exp w - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr hw)
  simp only [logDensity, mem_Ioi, hw, indicator_of_mem]
  dsimp [logWeight, kummerWeight]
  positivity

theorem logDensity_eq_zero (p κ w : ℝ) (hw : w ≤ 0) : logDensity p κ w = 0 := by
  simp [logDensity, not_lt.mpr hw]

theorem logDensity_nonneg (p κ w : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    0 ≤ logDensity p κ w := by
  by_cases hw : 0 < w
  · exact (logDensity_pos p κ w hp hκ hw).le
  · rw [logDensity_eq_zero p κ w (le_of_not_gt hw)]

private theorem log_kummer_integrability (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) ∧
    IntegrableOn (fun t => t * Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
  have hh := L3_kummer_moments p κ hp hκ
  have hi1 := hh.2.2.1
  have hil0 : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    refine hi1.mono' (by exact (by dsimp [kummerWeight]; fun_prop : Measurable
      (fun t => Real.log (1 + t) * kummerWeight p κ t)).aestronglyMeasurable) ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have ht0 : 0 < t := ht
    have hw : 0 ≤ kummerWeight p κ t := by dsimp [kummerWeight]; positivity
    have hl0 : 0 ≤ Real.log (1 + t) := Real.log_nonneg (by linarith)
    have hlt : Real.log (1 + t) ≤ t := by linarith [Real.log_le_sub_one_of_pos (by linarith : 0 < 1 + t)]
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg hl0 hw)]
    exact mul_le_mul_of_nonneg_right hlt hw
  have hil1 : IntegrableOn (fun t => t * Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    refine hh.2.2.2.1.mono' (by exact (by dsimp [kummerWeight]; fun_prop : Measurable
      (fun t => t * Real.log (1 + t) * kummerWeight p κ t)).aestronglyMeasurable) ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have ht0 : 0 < t := ht
    have hw : 0 ≤ kummerWeight p κ t := by dsimp [kummerWeight]; positivity
    have hl0 : 0 ≤ Real.log (1 + t) := Real.log_nonneg (by linarith)
    have hlt : Real.log (1 + t) ≤ t := by linarith [Real.log_le_sub_one_of_pos (by linarith : 0 < 1 + t)]
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    nlinarith [mul_le_mul_of_nonneg_right hlt (mul_nonneg ht0.le hw)]
  exact ⟨hil0, hil1⟩

theorem logDensity_moments (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    Integrable (logDensity p κ) ∧
    Integrable (fun w => w * logDensity p κ w) ∧
    Integrable (fun w => Real.exp w * logDensity p κ w) ∧
    Integrable (fun w => w * Real.exp w * logDensity p κ w) ∧
    (∫ w, logDensity p κ w) = 1 ∧
    (∫ w, w * logDensity p κ w) = kummerLog1p p κ := by
  have hh := L3_kummer_moments p κ hp hκ
  have hi0 := hh.2.1
  have hi1 := hh.2.2.1
  obtain ⟨hil0, hil1⟩ := log_kummer_integrability p κ hp hκ
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa only [one_mul] using integrable_logDensity p κ (fun _ => 1) (by simpa using hi0)
  · exact integrable_logDensity p κ id hil0
  · apply integrable_logDensity p κ Real.exp
    apply (hi0.add hi1).congr_fun _ measurableSet_Ioi
    intro t ht
    dsimp only [Pi.add_apply]
    rw [Real.exp_log (by linarith [show 0 < t from ht] : 0 < 1 + t)]
    ring
  · apply integrable_logDensity p κ (fun w => w * Real.exp w)
    apply (hil0.add hil1).congr_fun _ measurableSet_Ioi
    intro t ht
    dsimp only [Pi.add_apply]
    rw [Real.exp_log (by linarith [show 0 < t from ht] : 0 < 1 + t)]
    ring
  · have hi := integral_logDensity p κ (fun _ => 1)
    simp only [one_mul] at hi
    rw [hi]
    dsimp [kummerExpect]
    simp only [one_mul]
    exact inv_mul_cancel₀ hh.1.ne'
  · exact integral_logDensity p κ id

private theorem integral_pair (f : ℝ → ℝ) (m : ℝ) (hi : Integrable f) :
    (∫ w, f w) = ∫ x in Ioi (0 : ℝ), (f (m + x) + f (m - x)) := by
  have hp := (hi.comp_add_left m).integrableOn (s := Ioi 0)
  have hn := (hi.comp_sub_left m).integrableOn (s := Ioi 0)
  rw [integral_add hp hn]
  have hnval : (∫ x in Ioi (0 : ℝ), f (m - x)) = ∫ x in Iic (0 : ℝ), f (m + x) := by
    simpa only [neg_zero, sub_eq_add_neg] using integral_comp_neg_Ioi 0 (fun x => f (m + x))
  rw [hnval, add_comm, intervalIntegral.integral_Iic_add_Ioi
    (hi.comp_add_left m).integrableOn (hi.comp_add_left m).integrableOn,
    integral_add_left_eq_self]

theorem logMean_pos (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) : 0 < kummerLog1p p κ := by
  obtain ⟨hi0, hi1, _, _, _, hm⟩ := logDensity_moments p κ hp hκ
  rw [← hm, integral_pos_iff_support_of_nonneg_ae]
  · have hsub : Ioc (1 : ℝ) 2 ⊆ Function.support (fun w => w * logDensity p κ w) := by
      intro w hw
      have hw0 : 0 < w := by linarith [hw.1]
      exact Function.mem_support.mpr (mul_pos hw0 (logDensity_pos p κ w hp hκ hw0)).ne'
    refine lt_of_lt_of_le ?_ (measure_mono hsub)
    rw [Real.volume_Ioc]
    norm_num
  · filter_upwards [] with w
    by_cases hw : 0 < w
    · exact mul_nonneg hw.le (logDensity_nonneg p κ w hp hκ)
    · simp [logDensity_eq_zero p κ w (le_of_not_gt hw)]
  · exact hi1

private theorem centered_integrable (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    Integrable (fun w => (w - kummerLog1p p κ) * logDensity p κ w) := by
  obtain ⟨hi0, hi1, _⟩ := logDensity_moments p κ hp hκ
  convert hi1.sub (hi0.const_mul (kummerLog1p p κ)) using 1
  funext w
  dsimp only [Pi.sub_apply]
  ring

private theorem centered_integral_zero (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    (∫ w, (w - kummerLog1p p κ) * logDensity p κ w) = 0 := by
  obtain ⟨hi0, hi1, _, _, hmass, hmean⟩ := logDensity_moments p κ hp hκ
  simp_rw [sub_mul]
  rw [integral_sub hi1 (hi0.const_mul (kummerLog1p p κ)), integral_const_mul, hmass, hmean]
  ring

private theorem reflected_center_zero (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    (∫ x in Ioi (0 : ℝ), x *
      (logDensity p κ (kummerLog1p p κ + x) - logDensity p κ (kummerLog1p p κ - x))) = 0 := by
  have he := integral_pair (fun w => (w - kummerLog1p p κ) * logDensity p κ w)
    (kummerLog1p p κ) (centered_integrable p κ hp hκ)
  rw [centered_integral_zero p κ hp hκ] at he
  apply Eq.trans _ he.symm
  apply setIntegral_congr_fun measurableSet_Ioi
  intro x _
  dsimp only
  ring

private theorem reflected_center_integrable (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    IntegrableOn (fun x => x *
      (logDensity p κ (kummerLog1p p κ + x) - logDensity p κ (kummerLog1p p κ - x))) (Ioi 0) := by
  have hp' := ((centered_integrable p κ hp hκ).comp_add_left (kummerLog1p p κ)).integrableOn (s := Ioi 0)
  have hn' := ((centered_integrable p κ hp hκ).comp_sub_left (kummerLog1p p κ)).integrableOn (s := Ioi 0)
  apply (hp'.add hn').congr_fun _ measurableSet_Ioi
  intro x _
  dsimp only [Pi.add_apply]
  ring

private theorem logWeight_log (p κ w : ℝ) (hw : 0 < w) :
    Real.log (logWeight p κ w) = (p - 1) * Real.log (1 - Real.exp (-w)) -
      κ * (Real.exp w - 1) := by
  have hb : 0 < 1 - Real.exp (-w) := sub_pos.mpr (Real.exp_lt_one_iff.mpr (by linarith))
  rw [logWeight_formula p κ w hw,
    Real.log_mul (Real.rpow_pos_of_pos hb _).ne' (Real.exp_pos _).ne',
    Real.log_rpow hb, Real.log_exp]
  ring

private theorem logDensity_log (p κ w : ℝ) (hp : 0 < p) (hκ : 0 < κ) (hw : 0 < w) :
    Real.log (logDensity p κ w) = Real.log ((kummerZ p κ)⁻¹) +
      (p - 1) * Real.log (1 - Real.exp (-w)) - κ * (Real.exp w - 1) := by
  have hz : 0 < kummerZ p κ := (L3_kummer_moments p κ hp hκ).1
  have hwp : 0 < logWeight p κ w := by
    have ht : 0 < Real.exp w - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr hw)
    dsimp [logWeight, kummerWeight]
    positivity
  simp only [logDensity, mem_Ioi, hw, indicator_of_mem]
  rw [Real.log_mul (inv_pos.mpr hz).ne' hwp.ne', logWeight_log p κ w hw]
  ring

noncomputable def reflectedScore (p κ m x : ℝ) : ℝ :=
  (p - 1) * reflectedH m x - κ * Real.exp m

private theorem logDensity_log_reflection (p κ m x : ℝ) (hp : 0 < p) (hκ : 0 < κ)
    (hx : 0 < x) (hxm : x < m) :
    Real.log (logDensity p κ (m + x)) - Real.log (logDensity p κ (m - x)) =
      2 * Real.sinh x * reflectedScore p κ m x := by
  rw [logDensity_log p κ (m + x) hp hκ (by linarith),
    logDensity_log p κ (m - x) hp hκ (by linarith)]
  have he1 : Real.exp (-(m + x)) = Real.exp (-m) * Real.exp (-x) := by
    rw [neg_add, Real.exp_add]
  have he2 : Real.exp (-(m - x)) = Real.exp (-m) * Real.exp x := by
    rw [show -(m - x) = -m + x by ring, Real.exp_add]
  have he3 : Real.exp (m + x) - Real.exp (m - x) = 2 * Real.exp m * Real.sinh x := by
    rw [Real.exp_add, Real.exp_sub, Real.sinh_eq, Real.exp_neg]
    field_simp
  rw [he1, he2]
  have hs : Real.sinh x ≠ 0 := (Real.sinh_pos_iff.mpr hx).ne'
  dsimp [reflectedScore, reflectedH]
  field_simp
  nlinarith [he3]

private theorem reflection_nonneg_iff (p κ m x : ℝ) (hp : 0 < p) (hκ : 0 < κ)
    (hx : 0 < x) (hxm : x < m) :
    0 ≤ logDensity p κ (m + x) - logDensity p κ (m - x) ↔
      0 ≤ reflectedScore p κ m x := by
  have hp1 := logDensity_pos p κ (m + x) hp hκ (by linarith)
  have hp2 := logDensity_pos p κ (m - x) hp hκ (by linarith)
  rw [sub_nonneg, ← Real.log_le_log_iff hp2 hp1, ← sub_nonneg,
    logDensity_log_reflection p κ m x hp hκ hx hxm]
  exact mul_nonneg_iff_of_pos_left (mul_pos (by norm_num) (Real.sinh_pos_iff.mpr hx))

private theorem reflection_nonpos_iff (p κ m x : ℝ) (hp : 0 < p) (hκ : 0 < κ)
    (hx : 0 < x) (hxm : x < m) :
    logDensity p κ (m + x) - logDensity p κ (m - x) ≤ 0 ↔
      reflectedScore p κ m x ≤ 0 := by
  have hp1 := logDensity_pos p κ (m + x) hp hκ (by linarith)
  have hp2 := logDensity_pos p κ (m - x) hp hκ (by linarith)
  rw [sub_nonpos, ← Real.log_le_log_iff hp1 hp2, ← sub_nonpos,
    logDensity_log_reflection p κ m x hp hκ hx hxm]
  constructor <;> intro h <;> nlinarith [Real.sinh_pos_iff.mpr hx]

private theorem reflectedH_pos (m x : ℝ) (hx : 0 < x) (hxm : x < m) :
    0 < reflectedH m x := by
  have ha : Real.exp (-m) * Real.exp x < 1 := by
    rw [← Real.exp_add, Real.exp_lt_one_iff]
    linarith
  have he : Real.exp (-x) < Real.exp x := Real.exp_lt_exp.mpr (by linarith)
  have hdiff : 1 - Real.exp (-m) * Real.exp x < 1 - Real.exp (-m) * Real.exp (-x) := by
    nlinarith [mul_lt_mul_of_pos_left he (Real.exp_pos (-m))]
  unfold reflectedH
  exact div_pos (sub_pos.mpr (Real.log_lt_log (sub_pos.mpr ha) hdiff))
    (mul_pos (by norm_num) (Real.sinh_pos_iff.mpr hx))

private theorem reflectedScore_continuous (p κ m : ℝ) :
    ContinuousOn (reflectedScore p κ m) (Ioo 0 m) := by
  have hcont : ContinuousOn (reflectedH m) (Ioo 0 m) := by
    intro x hx
    have ha : 0 < 1 - Real.exp (-m) * Real.exp x := by
      apply sub_pos.mpr
      rw [← Real.exp_add, Real.exp_lt_one_iff]
      linarith [hx.2]
    have hb : 0 < 1 - Real.exp (-m) * Real.exp (-x) := by
      apply sub_pos.mpr
      rw [← Real.exp_add, Real.exp_lt_one_iff]
      linarith [hx.1, hx.2]
    have hs : 2 * Real.sinh x ≠ 0 := (mul_pos (by norm_num) (Real.sinh_pos_iff.mpr hx.1)).ne'
    exact (((continuousAt_const.sub (continuousAt_const.mul (Real.continuous_exp.continuousAt.comp
      continuousAt_id.neg))).log hb.ne').sub
      ((continuousAt_const.sub (continuousAt_const.mul Real.continuous_exp.continuousAt)).log ha.ne')).div
      (continuousAt_const.mul Real.continuous_sinh.continuousAt) hs |>.continuousWithinAt
  exact (continuousOn_const.mul hcont).sub continuousOn_const

private theorem reflection_tail_pos (p κ m x : ℝ) (hp : 0 < p) (hκ : 0 < κ)
    (hm : 0 < m) (hmx : m ≤ x) :
    0 < logDensity p κ (m + x) - logDensity p κ (m - x) := by
  rw [logDensity_eq_zero p κ (m - x) (by linarith), sub_zero]
  exact logDensity_pos p κ (m + x) hp hκ (by linarith)

private theorem reflected_score_negative_somewhere (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    ∃ x ∈ Ioo 0 (kummerLog1p p κ), reflectedScore p κ (kummerLog1p p κ) x < 0 := by
  let m := kummerLog1p p κ
  have hm : 0 < m := logMean_pos p κ hp hκ
  by_contra hn
  push_neg at hn
  have hnonneg : ∀ x > 0, 0 ≤ logDensity p κ (m + x) - logDensity p κ (m - x) := by
    intro x hx
    by_cases hxm : x < m
    · exact (reflection_nonneg_iff p κ m x hp hκ hx hxm).mpr (hn x ⟨hx, hxm⟩)
    · exact (reflection_tail_pos p κ m x hp hκ hm (le_of_not_gt hxm)).le
  have hint : 0 < ∫ x in Ioi (0 : ℝ), x * (logDensity p κ (m + x) - logDensity p κ (m - x)) := by
    rw [setIntegral_pos_iff_support_of_nonneg_ae]
    · have hsub : Ioc (m + 1) (m + 2) ⊆
          Function.support (fun x => x * (logDensity p κ (m + x) - logDensity p κ (m - x))) ∩ Ioi 0 := by
        intro x hx
        have hx0 : 0 < x := by linarith [hx.1]
        refine ⟨Function.mem_support.mpr ?_, hx0⟩
        exact (mul_pos hx0 (reflection_tail_pos p κ m x hp hκ hm (by linarith [hx.1]))).ne'
      refine lt_of_lt_of_le ?_ (measure_mono hsub)
      rw [Real.volume_Ioc]
      norm_num
    · filter_upwards [ae_restrict_mem measurableSet_Ioi] with x hx
      exact mul_nonneg (le_of_lt hx) (hnonneg x hx)
    · exact reflected_center_integrable p κ hp hκ
  have hz := reflected_center_zero p κ hp hκ
  linarith

/-- The exact Kummer log density has the single crossing required by the
strict centered Jensen-gap argument, on both sides of shape p=1. -/
theorem reflected_single_crossing (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    ∃ c : ℝ, 0 < c ∧
      (∀ x : ℝ, 0 < x → x < c →
        logDensity p κ (kummerLog1p p κ + x) - logDensity p κ (kummerLog1p p κ - x) ≤ 0) ∧
      (∀ x : ℝ, c < x →
        0 ≤ logDensity p κ (kummerLog1p p κ + x) - logDensity p κ (kummerLog1p p κ - x)) := by
  let m := kummerLog1p p κ
  have hm : 0 < m := logMean_pos p κ hp hκ
  by_cases hp1 : p ≤ 1
  · refine ⟨m, hm, ?_, ?_⟩
    · intro x hx hxm
      apply (reflection_nonpos_iff p κ m x hp hκ hx hxm).mpr
      dsimp [reflectedScore]
      have hH := reflectedH_pos m x hx hxm
      have hmul := mul_nonpos_of_nonpos_of_nonneg (by linarith : p - 1 ≤ 0) hH.le
      have hpos := mul_pos hκ (Real.exp_pos m)
      linarith
    · intro x hmx
      exact (reflection_tail_pos p κ m x hp hκ hm hmx.le).le
  · have hp1' : 0 < p - 1 := by linarith
    have hmono : StrictMonoOn (reflectedScore p κ m) (Ioo 0 m) := by
      intro x hx y hy hxy
      have h := mul_lt_mul_of_pos_left (reflectedH_strictMonoOn m hx hy hxy) hp1'
      dsimp [reflectedScore]
      linarith
    obtain ⟨a, ha, haneg⟩ := reflected_score_negative_somewhere p κ hp hκ
    by_cases hex : ∃ b ∈ Ioo 0 m, 0 < reflectedScore p κ m b
    · obtain ⟨b, hb, hbpos⟩ := hex
      have hab : a < b := by
        by_contra hh
        have hba := hmono.monotoneOn hb ha (le_of_not_gt hh)
        linarith
      have hcont : ContinuousOn (reflectedScore p κ m) (Icc a b) :=
        (reflectedScore_continuous p κ m).mono (by intro x hx; exact ⟨ha.1.trans_le hx.1, hx.2.trans_lt hb.2⟩)
      obtain ⟨c, hc, hcz⟩ := intermediate_value_Icc hab.le hcont ⟨haneg.le, hbpos.le⟩
      have hc0 : 0 < c := ha.1.trans_le hc.1
      have hcm : c < m := hc.2.trans_lt hb.2
      refine ⟨c, hc0, ?_, ?_⟩
      · intro x hx hxc
        have hxm : x < m := hxc.trans hcm
        apply (reflection_nonpos_iff p κ m x hp hκ hx hxm).mpr
        have hh := hmono ⟨hx, hxm⟩ ⟨hc0, hcm⟩ hxc
        linarith
      · intro x hcx
        by_cases hxm : x < m
        · apply (reflection_nonneg_iff p κ m x hp hκ (hc0.trans hcx) hxm).mpr
          have hh := hmono ⟨hc0, hcm⟩ ⟨hc0.trans hcx, hxm⟩ hcx
          linarith
        · exact (reflection_tail_pos p κ m x hp hκ hm (le_of_not_gt hxm)).le
    · push_neg at hex
      refine ⟨m, hm, ?_, ?_⟩
      · intro x hx hxm
        exact (reflection_nonpos_iff p κ m x hp hκ hx hxm).mpr (hex x ⟨hx, hxm⟩)
      · intro x hmx
        exact (reflection_tail_pos p κ m x hp hκ hm hmx.le).le

private theorem logDensity_exp_values (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    (∫ w, Real.exp w * logDensity p κ w) = 1 + kummerMean p κ ∧
    (∫ w, w * Real.exp w * logDensity p κ w) = kummerLog1p p κ + kummerCrossLog1p p κ := by
  have hh := L3_kummer_moments p κ hp hκ
  have hi0 := hh.2.1
  have hi1 := hh.2.2.1
  obtain ⟨hil0, hil1⟩ := log_kummer_integrability p κ hp hκ
  constructor
  · rw [integral_logDensity]
    unfold kummerExpect
    dsimp only
    have he : (∫ t in Ioi (0 : ℝ), Real.exp (Real.log (1 + t)) * kummerWeight p κ t) =
        kummerZ p κ + ∫ t in Ioi (0 : ℝ), t * kummerWeight p κ t := by
      calc
        _ = ∫ t in Ioi (0 : ℝ), (kummerWeight p κ t + t * kummerWeight p κ t) := by
          apply setIntegral_congr_fun measurableSet_Ioi
          intro t ht
          dsimp only
          rw [Real.exp_log (by linarith [show 0 < t from ht] : 0 < 1 + t)]
          ring
        _ = _ := integral_add hi0 hi1
    rw [he]
    dsimp [kummerMean, kummerExpect]
    field_simp [hh.1.ne']
  · rw [integral_logDensity]
    unfold kummerExpect
    dsimp only
    have he : (∫ t in Ioi (0 : ℝ), Real.log (1 + t) * Real.exp (Real.log (1 + t)) * kummerWeight p κ t) =
        (∫ t in Ioi (0 : ℝ), Real.log (1 + t) * kummerWeight p κ t) +
        (∫ t in Ioi (0 : ℝ), t * Real.log (1 + t) * kummerWeight p κ t) := by
      calc
        _ = ∫ t in Ioi (0 : ℝ), (Real.log (1 + t) * kummerWeight p κ t +
          t * Real.log (1 + t) * kummerWeight p κ t) := by
          apply setIntegral_congr_fun measurableSet_Ioi
          intro t ht
          dsimp only
          rw [Real.exp_log (by linarith [show 0 < t from ht] : 0 < 1 + t)]
          ring
        _ = _ := integral_add hil0 hil1
    rw [he]
    dsimp [kummerLog1p, kummerCrossLog1p, kummerExpect]
    ring

private theorem centered_gap_expansion (m B w : ℝ) (ρ : ℝ → ℝ) :
    (JensenGap.h (w - m) - B * (w - m)) * ρ w =
      (Real.exp (-m) * (2 + m)) * (Real.exp w * ρ w) -
      Real.exp (-m) * (w * Real.exp w * ρ w) -
      (1 + B) * (w * ρ w) + ((m - 2) + B * m) * ρ w := by
  dsimp [JensenGap.h]
  rw [show w - m = w + -m by ring, Real.exp_add]
  ring

private theorem paired_gap_neg (x c dp dm : ℝ) (hx : 0 < x) (hc : 0 < c)
    (hdp : 0 < dp) (hdm : 0 ≤ dm)
    (hbelow : x < c → dp - dm ≤ 0) (habove : c < x → 0 ≤ dp - dm) :
    (JensenGap.h x - JensenGap.B c * x) * dp +
      (JensenGap.h (-x) - JensenGap.B c * (-x)) * dm < 0 := by
  let A := 2 * (Real.cosh x - 1) - x * Real.sinh x
  have hA : A < 0 := JensenGap.A1 x hx
  have he := (JensenGap.H1 x).1
  have ho := (JensenGap.H1 x).2 hx.ne'
  have ho' := (div_eq_iff (by positivity : 2 * x ≠ 0)).mp ho
  have hp : JensenGap.h x = A + x * JensenGap.B x := by dsimp [A]; linarith
  have hn : JensenGap.h (-x) = A - x * JensenGap.B x := by dsimp [A]; linarith
  have hodd : x * (JensenGap.B x - JensenGap.B c) * (dp - dm) ≤ 0 := by
    rcases lt_trichotomy x c with h | h | h
    · have hb := JensenGap.B1 x c hx h
      exact mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hx.le (by linarith)) (hbelow h)
    · subst c
      simp
    · have hb := JensenGap.B1 c x hc h
      exact mul_nonpos_of_nonpos_of_nonneg (mul_nonpos_of_nonneg_of_nonpos hx.le (by linarith)) (habove h)
  have heq : (JensenGap.h x - JensenGap.B c * x) * dp +
      (JensenGap.h (-x) - JensenGap.B c * (-x)) * dm =
      A * (dp + dm) + x * (JensenGap.B x - JensenGap.B c) * (dp - dm) := by
    rw [hp, hn]
    ring
  rw [heq]
  have hstrict := mul_neg_of_neg_of_pos hA (by linarith : 0 < dp + dm)
  linarith

/-- The strict factor-two Jensen gap for the actual normalized Kummer law. -/
theorem kummer_jensen_gap (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    2 * (1 + kummerMean p κ - Real.exp (kummerLog1p p κ)) <
      kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ := by
  let m := kummerLog1p p κ
  have hm : 0 < m := logMean_pos p κ hp hκ
  obtain ⟨c, hc, hbelow, habove⟩ := reflected_single_crossing p κ hp hκ
  let B := JensenGap.B c
  let f : ℝ → ℝ := fun w => (JensenGap.h (w - m) - B * (w - m)) * logDensity p κ w
  obtain ⟨hi0, hiw, hie, hiwe, hmass, hmean⟩ := logDensity_moments p κ hp hκ
  have hfeq : f = fun w =>
      (Real.exp (-m) * (2 + m)) * (Real.exp w * logDensity p κ w) -
      Real.exp (-m) * (w * Real.exp w * logDensity p κ w) -
      (1 + B) * (w * logDensity p κ w) + ((m - 2) + B * m) * logDensity p κ w := by
    funext w
    exact centered_gap_expansion m B w (logDensity p κ)
  have hiAB : Integrable (fun w =>
      (Real.exp (-m) * (2 + m)) * (Real.exp w * logDensity p κ w) -
      Real.exp (-m) * (w * Real.exp w * logDensity p κ w)) :=
    (hie.const_mul _).sub (hiwe.const_mul _)
  have hiABC : Integrable (fun w =>
      (Real.exp (-m) * (2 + m)) * (Real.exp w * logDensity p κ w) -
      Real.exp (-m) * (w * Real.exp w * logDensity p κ w) -
      (1 + B) * (w * logDensity p κ w)) := hiAB.sub (hiw.const_mul _)
  have hif : Integrable f := by
    rw [hfeq]
    exact hiABC.add (hi0.const_mul _)
  let g : ℝ → ℝ := fun x => f (m + x) + f (m - x)
  have hig : IntegrableOn g (Ioi 0) :=
    ((hif.comp_add_left m).integrableOn).add ((hif.comp_sub_left m).integrableOn)
  have hgneg : ∀ x > 0, g x < 0 := by
    intro x hx
    have he1 : m + x - m = x := by ring
    have he2 : m - x - m = -x := by ring
    dsimp [g, f]
    rw [he1, he2]
    exact paired_gap_neg x c _ _ hx hc
      (logDensity_pos p κ (m + x) hp hκ (by linarith))
      (logDensity_nonneg p κ (m - x) hp hκ) (hbelow x hx) (habove x)
  have hgint : (∫ x in Ioi (0 : ℝ), g x) < 0 := by
    have hpos : 0 < ∫ x in Ioi (0 : ℝ), -g x := by
      rw [setIntegral_pos_iff_support_of_nonneg_ae]
      · have hsub : Ioc (1 : ℝ) 2 ⊆ Function.support (fun x => -g x) ∩ Ioi 0 := by
          intro x hx
          have hx0 : 0 < x := by linarith [hx.1]
          exact ⟨Function.mem_support.mpr (neg_pos.mpr (hgneg x hx0)).ne', hx0⟩
        refine lt_of_lt_of_le ?_ (measure_mono hsub)
        rw [Real.volume_Ioc]
        norm_num
      · filter_upwards [ae_restrict_mem measurableSet_Ioi] with x hx
        exact (neg_pos.mpr (hgneg x hx)).le
      · exact hig.neg
    rw [integral_neg] at hpos
    linarith
  have hfint : (∫ w, f w) < 0 := by
    rw [integral_pair f m hif]
    exact hgint
  obtain ⟨hexp, hwexp⟩ := logDensity_exp_values p κ hp hκ
  have heval : (∫ w, f w) = Real.exp (-m) *
      (2 + 2 * kummerMean p κ + m * kummerMean p κ - kummerCrossLog1p p κ) - 2 := by
    rw [hfeq, integral_add hiABC (hi0.const_mul _),
      integral_sub hiAB (hiw.const_mul _), integral_sub (hie.const_mul _) (hiwe.const_mul _)]
    simp_rw [integral_const_mul]
    rw [hexp, hwexp, hmass, hmean]
    dsimp [m]
    ring
  rw [heval] at hfint
  have hmul := mul_lt_mul_of_pos_left hfint (Real.exp_pos m)
  have he : Real.exp m * Real.exp (-m) = 1 := by rw [← Real.exp_add]; simp
  rw [mul_sub, ← mul_assoc, he, one_mul, mul_zero] at hmul
  dsimp [m] at hmul
  linarith

/-- Rescaled strict gap, in exactly the variables used by the barrier proof. -/
theorem kummer_strict_gap (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    κ * kummerMean p κ + κ - κ / Real.exp (-kummerLog1p p κ) <
      κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ) / 2 := by
  have h := mul_lt_mul_of_pos_left (kummer_jensen_gap p κ hp hκ) hκ
  rw [Real.exp_neg, div_inv_eq_mul]
  nlinarith

/-- Strict exponential Jensen makes the Kummer gap itself positive. -/
theorem kummer_positive_gap (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    0 < κ * kummerMean p κ + κ - κ / Real.exp (-kummerLog1p p κ) := by
  let m := kummerLog1p p κ
  have hm : 0 < m := logMean_pos p κ hp hκ
  obtain ⟨hi0, _, hie, _, hmass, _⟩ := logDensity_moments p κ hp hκ
  have hic := centered_integrable p κ hp hκ
  let f : ℝ → ℝ := fun w => (Real.exp (w - m) - 1 - (w - m)) * logDensity p κ w
  have hfeq : f = fun w => Real.exp (-m) * (Real.exp w * logDensity p κ w) -
      logDensity p κ w - (w - m) * logDensity p κ w := by
    funext w
    dsimp [f]
    rw [show w - m = w + -m by ring, Real.exp_add]
    ring
  have hif : Integrable f := by
    rw [hfeq]
    exact ((hie.const_mul _).sub hi0).sub hic
  have hfpos : 0 < ∫ w, f w := by
    rw [integral_pos_iff_support_of_nonneg_ae]
    · have hsub : Ioc (m + 1) (m + 2) ⊆ Function.support f := by
        intro w hw
        have hw0 : 0 < w := by linarith [hw.1]
        have hne : w - m ≠ 0 := by linarith [hw.1]
        have hgap : 0 < Real.exp (w - m) - 1 - (w - m) := by
          linarith [Real.add_one_lt_exp hne]
        exact Function.mem_support.mpr (mul_pos hgap (logDensity_pos p κ w hp hκ hw0)).ne'
      refine lt_of_lt_of_le ?_ (measure_mono hsub)
      rw [Real.volume_Ioc]
      norm_num
    · filter_upwards [] with w
      exact mul_nonneg (by linarith [Real.add_one_le_exp (w - m)])
        (logDensity_nonneg p κ w hp hκ)
    · exact hif
  have hiA : Integrable (fun w => Real.exp (-m) * (Real.exp w * logDensity p κ w) - logDensity p κ w) :=
    (hie.const_mul _).sub hi0
  have heval : (∫ w, f w) = Real.exp (-m) * (1 + kummerMean p κ) - 1 := by
    rw [hfeq]
    dsimp only
    rw [integral_sub hiA hic,
      integral_sub (hie.const_mul _) hi0, integral_const_mul,
      (logDensity_exp_values p κ hp hκ).1, hmass, centered_integral_zero p κ hp hκ]
    ring
  rw [heval] at hfpos
  have hmul := mul_pos (Real.exp_pos m) hfpos
  have he : Real.exp m * Real.exp (-m) = 1 := by rw [← Real.exp_add]; simp
  rw [mul_sub, ← mul_assoc, he, one_mul, mul_one] at hmul
  have hh := mul_pos hκ hmul
  rw [Real.exp_neg, div_inv_eq_mul]
  dsimp [m] at hh
  nlinarith

#print axioms kummer_positive_gap
#print axioms kummer_jensen_gap
#print axioms kummer_strict_gap
#print axioms logDensity_exp_values
#print axioms paired_gap_neg
#print axioms reflected_single_crossing
#print axioms logDensity_log_reflection
#print axioms reflected_center_zero
#print axioms logMean_pos
#print axioms logDensity_moments
#print axioms integral_pair
#print axioms reflectedH_strictMonoOn
#print axioms integral_logWeight
#print axioms integrable_logWeight_iff
#print axioms logWeight_formula
end Dobson.KummerGap
