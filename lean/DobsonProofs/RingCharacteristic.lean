import DobsonProofs.Coverage6Remainders
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open Set Filter
open scoped BigOperators Topology

namespace Dobson.RingCharacteristic

noncomputable def circuitLog {n : ℕ} (s : Fin n → ℝ) (L : ℝ) : ℝ :=
  ∑ i, (Real.log (Real.exp L - s i) - Real.log (1 - s i))

noncomputable def duration {n : ℕ} (s : Fin n → ℝ) : ℝ :=
  ∑ i, 1 / (1 - s i)

lemma circuitLog_zero {n : ℕ} (s : Fin n → ℝ) : circuitLog s 0 = 0 := by
  simp [circuitLog]

lemma circuitLog_hasDerivAt {n : ℕ} (s : Fin n → ℝ) (L : ℝ)
    (hL : ∀ i, s i < Real.exp L) :
    HasDerivAt (circuitLog s) (∑ i, Real.exp L / (Real.exp L - s i)) L := by
  unfold circuitLog
  convert HasDerivAt.sum (u := Finset.univ) (fun i _ =>
    (((Real.hasDerivAt_exp L).sub_const (s i)).log
      (ne_of_gt (sub_pos.mpr (hL i)))).sub_const (Real.log (1 - s i))) using 1 <;>
    ext x <;> simp [Finset.sum_apply]

lemma circuitLog_contDiffAt {n : ℕ} (s : Fin n → ℝ) (L : ℝ)
    (hL : ∀ i, s i < Real.exp L) : ContDiffAt ℝ 2 (circuitLog s) L := by
  apply ContDiffAt.sum
  intro i _
  exact ((Real.contDiff_exp.contDiffAt.sub contDiffAt_const).log
    (ne_of_gt (sub_pos.mpr (hL i)))).sub contDiffAt_const

lemma duration_pos {n : ℕ} (hn : 1 ≤ n) (s : Fin n → ℝ)
    (hs : ∀ i, s i < 1) : 0 < duration s := by
  haveI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  exact Finset.sum_pos (fun i _ => one_div_pos.mpr (sub_pos.mpr (hs i)))
    Finset.univ_nonempty

lemma domain_of_mem {n : ℕ} (s : Fin n → ℝ) (δ L : ℝ)
    (hsep : ∀ i, s i < Real.exp (-δ)) (hL : L ∈ Icc (-δ) δ) :
    ∀ i, s i < Real.exp L :=
  fun i => (hsep i).trans_le (Real.exp_le_exp.mpr hL.1)

lemma circuitLog_abs_lower {n : ℕ} (hn : 1 ≤ n) (s : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (δ : ℝ) (hδ : 0 < δ)
    (hsep : ∀ i, s i < Real.exp (-δ)) (L : ℝ) (hL : L ∈ Icc (-δ) δ) :
    |L| ≤ |circuitLog s L| := by
  have hder (x : ℝ) (hx : x ∈ Icc (-δ) δ) :=
    circuitLog_hasDerivAt s x (domain_of_mem s δ x hsep hx)
  have hlo (x : ℝ) (hx : x ∈ Icc (-δ) δ) : 1 ≤ deriv (circuitLog s) x := by
    rw [(hder x hx).deriv]
    have hsum : (n : ℝ) ≤ ∑ i, Real.exp x / (Real.exp x - s i) := by
      calc
        (n : ℝ) = ∑ _ : Fin n, (1 : ℝ) := by simp
        _ ≤ _ := Finset.sum_le_sum fun i _ => by
          apply (le_div_iff₀ (sub_pos.mpr (domain_of_mem s δ x hsep hx i))).mpr
          nlinarith [hs0 i]
    exact (by exact_mod_cast hn : (1 : ℝ) ≤ n).trans hsum
  have hmono := (convex_Icc (-δ) δ).mul_sub_le_image_sub_of_le_deriv
    (fun x hx => (hder x hx).continuousAt.continuousWithinAt)
    (fun x hx => (hder x (interior_subset hx)).differentiableAt.differentiableWithinAt)
    (fun x hx => hlo x (interior_subset hx))
  have hzero : (0 : ℝ) ∈ Icc (-δ) δ := ⟨by linarith, hδ.le⟩
  by_cases hpos : 0 ≤ L
  · have hb := hmono 0 hzero L hL hpos
    simp only [circuitLog_zero, sub_zero, one_mul] at hb
    rw [abs_of_nonneg hpos, abs_of_nonneg (hpos.trans hb)]
    exact hb
  · have hneg : L ≤ 0 := le_of_not_ge hpos
    have hb := hmono L hL 0 hzero hneg
    simp only [circuitLog_zero, zero_sub, one_mul] at hb
    rw [abs_of_nonpos hneg, abs_of_nonpos (by linarith : circuitLog s L ≤ 0)]
    exact hb

theorem linear_taylor_bound {n : ℕ} (hn : 1 ≤ n) (s : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (δ : ℝ) (hδ : 0 < δ) (hsep : ∀ i, s i < Real.exp (-δ)) :
    ∃ A : ℝ, 0 ≤ A ∧ ∀ L ∈ Icc (-δ) δ,
      |L - circuitLog s L / duration s| ≤ A * (circuitLog s L) ^ 2 := by
  have hcd (x : ℝ) (hx : x ∈ Icc (-δ) δ) :=
    circuitLog_contDiffAt s x (domain_of_mem s δ x hsep hx)
  have hcdon : ContDiffOn ℝ 2 (circuitLog s) (Icc (-δ) δ) :=
    fun x hx => (hcd x hx).contDiffWithinAt
  have huniq : UniqueDiffOn ℝ (Icc (-δ) δ) := uniqueDiffOn_Icc (by linarith)
  have heq (x : ℝ) (hx : x ∈ Icc (-δ) δ) :
      iteratedDerivWithin 2 (circuitLog s) (Icc (-δ) δ) x =
        iteratedDeriv 2 (circuitLog s) x :=
    iteratedDerivWithin_eq_iteratedDeriv huniq (hcd x hx) hx
  have hcont : ContinuousOn (fun x => |iteratedDeriv 2 (circuitLog s) x|)
      (Icc (-δ) δ) := by
    apply (hcdon.continuousOn_iteratedDerivWithin (m := 2) (by norm_num) huniq).abs.congr
    intro x hx
    change |iteratedDeriv 2 (circuitLog s) x| = |iteratedDerivWithin 2 (circuitLog s) (Icc (-δ) δ) x|
    rw [heq x hx]
  have hzero : (0 : ℝ) ∈ Icc (-δ) δ := ⟨by linarith, hδ.le⟩
  obtain ⟨z, hz, hzmax⟩ := isCompact_Icc.exists_isMaxOn ⟨0, hzero⟩ hcont
  let M := |iteratedDeriv 2 (circuitLog s) z|
  have hM : 0 ≤ M := abs_nonneg _
  have hT := duration_pos hn s hs1
  refine ⟨M / (2 * duration s), by positivity, ?_⟩
  intro L hL
  by_cases hL0 : L = 0
  · simp [hL0, circuitLog_zero]
  have hne : (0 : ℝ) ≠ L := Ne.symm hL0
  have hseg : uIcc 0 L ⊆ Icc (-δ) δ := uIcc_subset_Icc hzero hL
  obtain ⟨x, hx, ht⟩ := taylor_mean_remainder_lagrange_iteratedDeriv
    (n := 1) hne (hcdon.mono hseg)
  have hd : iteratedDerivWithin 1 (circuitLog s) (uIcc 0 L) 0 = duration s := by
    rw [iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_uIcc hne)
      ((hcd 0 hzero).of_le (by norm_num)) left_mem_uIcc, iteratedDeriv_one]
    simpa [duration] using (circuitLog_hasDerivAt s 0 (by simpa using hs1)).deriv
  have htay : taylorWithinEval (circuitLog s) 1 (uIcc 0 L) 0 L = duration s * L := by
    norm_num [show 1 = 0 + 1 from rfl, taylorWithinEval_succ, hd, circuitLog_zero,
      smul_eq_mul]
    ring
  rw [htay] at ht
  norm_num at ht
  have hxm : |iteratedDeriv 2 (circuitLog s) x| ≤ M := hzmax (hseg ⟨hx.1.le, hx.2.le⟩)
  have herr : |circuitLog s L - duration s * L| ≤ M * L ^ 2 / 2 := by
    rw [ht, abs_div, abs_mul, abs_of_nonneg (sq_nonneg L)]
    norm_num
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hxm (sq_nonneg L)) (by norm_num)
  have hLL : L ^ 2 ≤ (circuitLog s L) ^ 2 := by
    have h := mul_self_le_mul_self (abs_nonneg L) (circuitLog_abs_lower hn s hs0 δ hδ hsep L hL)
    simpa only [← sq, sq_abs] using h
  have hcalc : |L - circuitLog s L / duration s| =
      |circuitLog s L - duration s * L| / duration s := by
    have he : L - circuitLog s L / duration s =
        -(circuitLog s L - duration s * L) / duration s := by field_simp; ring
    rw [he, abs_div, abs_neg, abs_of_pos hT]
  rw [hcalc]
  calc
    |circuitLog s L - duration s * L| / duration s ≤ (M * L ^ 2 / 2) / duration s :=
      div_le_div_of_nonneg_right herr hT.le
    _ ≤ (M * (circuitLog s L) ^ 2 / 2) / duration s := by gcongr
    _ = M / (2 * duration s) * (circuitLog s L) ^ 2 := by ring

theorem exists_radius {n : ℕ} (hn : 1 ≤ n) (s : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ i, s i < Real.exp (-δ) := by
  haveI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  let S := Finset.univ.sup' Finset.univ_nonempty s
  have hS (i : Fin n) : s i ≤ S := Finset.le_sup' s (Finset.mem_univ i)
  have hS0 : 0 ≤ S := (hs0 ⟨0, by omega⟩).trans (hS _)
  have hS1 : S < 1 := (Finset.sup'_lt_iff Finset.univ_nonempty).mpr (fun i _ => hs1 i)
  have hq0 : 0 < (1 + S) / 2 := by linarith
  have hq1 : (1 + S) / 2 < 1 := by linarith
  refine ⟨-Real.log ((1 + S) / 2), neg_pos.mpr (Real.log_neg hq0 hq1), ?_⟩
  intro i
  rw [neg_neg, Real.exp_log hq0]
  linarith [hS i]

lemma circuitLog_ge_of_nonneg {n : ℕ} (hn : 1 ≤ n) (s : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) (L : ℝ) (hL : 0 ≤ L) :
    L ≤ circuitLog s L := by
  have he : 1 ≤ Real.exp L := Real.one_le_exp_iff.mpr hL
  have hi (i : Fin n) : L ≤ Real.log (Real.exp L - s i) - Real.log (1 - s i) := by
    have hp : 0 < Real.exp L * (1 - s i) := mul_pos (Real.exp_pos L) (sub_pos.mpr (hs1 i))
    have ho : Real.exp L * (1 - s i) ≤ Real.exp L - s i := by
      nlinarith [mul_nonneg (sub_nonneg.mpr he) (hs0 i)]
    have hh := Real.log_le_log hp ho
    rw [Real.log_mul (Real.exp_pos L).ne' (sub_pos.mpr (hs1 i)).ne', Real.log_exp] at hh
    linarith
  have hh := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hi i)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hh
  have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
  exact (by nlinarith : L ≤ (n : ℝ) * L).trans hh

/-- An actual deterministic-ring slope bound on the chosen compact interval. -/
theorem circuitLog_abs_upper {n : ℕ} (s : Fin n → ℝ) (δ : ℝ) (hδ : 0 < δ)
    (hsep : ∀ i, s i < Real.exp (-δ)) :
    ∃ U : ℝ, 0 ≤ U ∧ ∀ L ∈ Icc (-δ) δ, |circuitLog s L| ≤ U * |L| := by
  have hder (x : ℝ) (hx : x ∈ Icc (-δ) δ) :=
    circuitLog_hasDerivAt s x (domain_of_mem s δ x hsep hx)
  have hcont : ContinuousOn (fun x => |∑ i, Real.exp x / (Real.exp x - s i)|)
      (Icc (-δ) δ) := by
    apply ContinuousOn.abs
    apply continuousOn_finsetSum
    intro i _
    apply Real.continuous_exp.continuousOn.div
      (Real.continuous_exp.continuousOn.sub continuousOn_const)
    intro x hx
    exact ne_of_gt (sub_pos.mpr (domain_of_mem s δ x hsep hx i))
  have hzero : (0 : ℝ) ∈ Icc (-δ) δ := ⟨by linarith, hδ.le⟩
  obtain ⟨z, hz, hzmax⟩ := isCompact_Icc.exists_isMaxOn ⟨0, hzero⟩ hcont
  refine ⟨|∑ i, Real.exp z / (Real.exp z - s i)|, abs_nonneg _, ?_⟩
  intro L hL
  have hb := (convex_Icc (-δ) δ).norm_image_sub_le_of_norm_deriv_le
    (fun x hx => (hder x hx).differentiableAt)
    (fun x hx => by
      rw [Real.norm_eq_abs, (hder x hx).deriv]
      exact hzmax hx)
    hzero hL
  simpa only [circuitLog_zero, sub_zero, Real.norm_eq_abs] using hb

end Dobson.RingCharacteristic
