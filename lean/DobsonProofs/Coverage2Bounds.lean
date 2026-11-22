import DobsonProofs.ReferencePaths
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Log.ENNRealLog
import Mathlib.Data.Matrix.Mul
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Tactic

/-! # Coverage batch 2: means, iid loss, threshold displacement, and path bounds

This module contains scalar and conditional helpers. The version-2 inventory
credits full coverage of the means sandwich through the actual-product results.
The threshold estimate here assumes localization and Taylor/Lipschitz estimates;
its conditional algebra does not establish those premises. -/
/-! ## Coverage correction — 2026-11-22

`cor:bounds-means` has full coverage: the actual geometric lower bound and
temporally independent arithmetic upper bound are proved, with a named
expected-growth limit as a textbook input. The former `hupper` and `hlower`
entries belong to the unused private helper `means_sandwich_of_bounds`.

`root_quartic_bound` still takes remainder, linearization, coefficient and
localization estimates; `prop:bounds-r0star` is partial-P1. The path-bound
helpers take their displayed Jensen/path premises. The credited finite-path
and actual-ring bounds give core coverage of `prop:bounds-any`, not a
construction of every general-network stationary process. -/

namespace Dobson.Coverage2Bounds
open scoped BigOperators ENNReal
open Filter

/-- `cor:bounds-means`: the log gap for actual products of mean transfers. -/
theorem means_log_gap {ι : Type*} (S : Finset ι) (g A G s : ι → ℝ)
    (hg : ∀ i ∈ S, 0 < g i) (hA : ∀ i ∈ S, 0 < A i)
    (hG : ∀ i ∈ S, 0 < G i) (hs : ∀ i ∈ S, s i < 1) :
    Real.log (∏ i ∈ S, g i * A i / (1 - s i)) -
      Real.log (∏ i ∈ S, g i * G i / (1 - s i)) =
      ∑ i ∈ S, (Real.log (A i) - Real.log (G i)) := by
  rw [Real.log_prod (fun i hi => ne_of_gt (div_pos (mul_pos (hg i hi) (hA i hi)) (sub_pos.mpr (hs i hi)))),
    Real.log_prod (fun i hi => ne_of_gt (div_pos (mul_pos (hg i hi) (hG i hi)) (sub_pos.mpr (hs i hi)))),
    ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Real.log_div (ne_of_gt (mul_pos (hg i hi) (hA i hi))) (ne_of_gt (sub_pos.mpr (hs i hi))),
    Real.log_div (ne_of_gt (mul_pos (hg i hi) (hG i hi))) (ne_of_gt (sub_pos.mpr (hs i hi))),
    Real.log_mul (ne_of_gt (hg i hi)) (ne_of_gt (hA i hi)),
    Real.log_mul (ne_of_gt (hg i hi)) (ne_of_gt (hG i hi))]
  ring

/-- The two probabilistic loss bounds imply the arithmetic/geometric sandwich. -/
private theorem means_sandwich_of_bounds (RA RG Rstar : ℝ) (m : ℕ)
    (hA : 0 < RA) (hG : 0 < RG) (hstar : 0 < Rstar)
    (hupper : Real.log RA - m * Real.log Rstar ≤ Real.log RA - Real.log RG)
    (hlower : 0 ≤ Real.log RA - m * Real.log Rstar) :
    RG ≤ Rstar ^ m ∧ Rstar ^ m ≤ RA := by
  constructor
  · apply (Real.log_le_log_iff hG (pow_pos hstar m)).mp
    rw [Real.log_pow]
    linarith
  · apply (Real.log_le_log_iff (pow_pos hstar m) hA).mp
    rw [Real.log_pow]
    linarith

/-- `prop:bounds-iid`: expectation evolution from the conditional-expectation bridge. -/
theorem mean_evolution {ι : Type*} [Fintype ι] [DecidableEq ι]
    (A : Matrix ι ι ℝ) (μ : ℕ → ι → ℝ)
    (hstep : ∀ t, μ (t + 1) = A.mulVec (μ t)) (t : ℕ) :
    μ t = (A ^ t).mulVec (μ 0) := by
  induction t with
  | zero => simp
  | succ t ih => rw [hstep, ih, pow_succ', Matrix.mulVec_mulVec]

/-- No negative restoring tilt can be a root under the annealed growth bound.
2026-11-16 auditfix2 (flag 16): extended-real growth includes extinction, and
`ENNReal.log 0 = ⊥` handles zero spectral radius. The annealed bound itself
remains an explicit probability input; this theorem does not construct it. -/
theorem iid_root_nonnegative (r : ℝ → EReal) (ρ : ℝ → ℝ≥0∞) (D : ℝ)
    (hupper : ∀ y, y < 0 → r y ≤ ENNReal.log (ρ y))
    (hsub : ∀ y, y < 0 → ρ y < 1) (hroot : r D = 0) :
    0 ≤ D := by
  by_contra h
  have hD : D < 0 := lt_of_not_ge h
  have hlog : ENNReal.log (ρ D) < 0 := ENNReal.log_lt_zero_iff.mpr (hsub D hD)
  have hb := hupper D hD
  rw [hroot] at hb
  exact (not_lt_of_ge hb) hlog

/-- Zero spectral radius forces negative-infinite growth under the annealed
bound. There is no positivity side condition on the radius. -/
theorem iid_zero_radius_growth (r : EReal) (hupper : r ≤ ENNReal.log 0) :
    r = ⊥ := by
  simpa only [ENNReal.log_zero, le_bot_iff] using hupper

/-- `prop:bounds-r0star`: quantitative fourth-order root displacement.
The four bounds are analytic bridges, not conclusions inferred from convexity. -/
private theorem root_quartic_bound_of_bounds (D σ T C0 LD CD K A B M : ℝ)
    (hT : 0 < T) (hA : 0 ≤ A) (hB : 0 ≤ B) (_hM : 0 ≤ M)
    (hrem : |LD - CD * σ ^ 2| ≤ K * σ ^ 4)
    (hlin : |LD - D / T| ≤ A * D ^ 2)
    (hcoef : |CD - C0| ≤ B * |D|)
    (hloc : |D| ≤ M * σ ^ 2) :
    |D - T * C0 * σ ^ 2| ≤ T * (K + A * M ^ 2 + B * M) * σ ^ 4 := by
  have hsq : D ^ 2 ≤ M ^ 2 * σ ^ 4 := by
    have := mul_self_le_mul_self (abs_nonneg D) hloc
    nlinarith [sq_abs D]
  have h1 := mul_le_mul_of_nonneg_left hsq hA
  have h2 := mul_le_mul_of_nonneg_left hloc hB
  have h3 := mul_le_mul_of_nonneg_right (hcoef.trans h2) (sq_nonneg σ)
  have htri := abs_add_three (D / T - LD) (LD - CD * σ ^ 2) ((CD - C0) * σ ^ 2)
  rw [abs_sub_comm (D / T) LD, abs_mul, abs_of_nonneg (sq_nonneg σ)] at htri
  have hb : |D / T - C0 * σ ^ 2| ≤ (K + A * M ^ 2 + B * M) * σ ^ 4 := by
    have he : D / T - LD + (LD - CD * σ ^ 2) + (CD - C0) * σ ^ 2 = D / T - C0 * σ ^ 2 := by ring
    rw [he] at htri
    nlinarith
  have hmul := mul_le_mul_of_nonneg_left hb hT.le
  have he : D - T * C0 * σ ^ 2 = T * (D / T - C0 * σ ^ 2) := by
    field_simp
  rw [he, abs_mul, abs_of_pos hT]
  simpa [mul_assoc] using hmul

/-- `prop:bounds-path`: exact telescoping for each finite path, before summing. -/
theorem path_telescoping (lam : ℝ) (w p h : ℕ → ℝ)
    (hw : ∀ t, w t ≠ 0) (n : ℕ) :
    (∏ t ∈ Finset.range n, (lam * w t / w (t + 1) * p t * h t)) =
      lam ^ n * (w 0 / w n) * ∏ t ∈ Finset.range n, (p t * h t) := by
  induction n with
  | zero => simp [hw]
  | succ n ih =>
    rw [Finset.prod_range_succ, ih, Finset.prod_range_succ, pow_succ]
    field_simp [hw]

/-- The finite-time path Jensen bound is derived from the actual labeled
Markov path partition. Only the textbook expected growth limit is supplied.
2026-11-16 leanp1: replaces the arbitrary q/hjensen interface. `meanScore`
is the stationary mean log multiplier, i.e. minus the manuscript path cost. -/
theorem path_limit_bound {ι E Ω : Type*} [Fintype ι] [Fintype E]
    [MeasurableSpace Ω] (K : ReferencePaths.Kernel ι E)
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasureTheory.MeasurePreserving T μ μ)
    (H : Ω → E → ℝ) (hH : ∀ e, MeasureTheory.Integrable (fun ω => H ω e) μ)
    (r logLam : ℝ)
    (hFurstenbergKesten : Tendsto (ReferencePaths.normalizedGrowth K μ T H)
      atTop (nhds (r - logLam))) :
    ReferencePaths.meanScore K μ H ≤ r - logLam :=
  ge_of_tendsto hFurstenbergKesten
    (Filter.Eventually.of_forall (ReferencePaths.normalized_growth_lower K μ T hT H hH))

/-- Frequency of counted birth edges in the stationary reference chain. -/
noncomputable def birthFrequency {ι E : Type*} [Fintype ι] [Fintype E]
    (K : ReferencePaths.Kernel ι E) (β : E → ℝ) : ℝ :=
  ∑ i, K.initial i * ∑ e, K.prob i e * β e

private lemma meanScore_tilt {ι E Ω : Type*} [Fintype ι] [Fintype E]
    [MeasurableSpace Ω] (K : ReferencePaths.Kernel ι E)
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (H : Ω → E → ℝ) (hH : ∀ e, MeasureTheory.Integrable (fun ω => H ω e) μ)
    (β : E → ℝ) (D : ℝ) :
    ReferencePaths.meanScore K μ (fun ω e => H ω e + D * β e) =
      ReferencePaths.meanScore K μ H + D * birthFrequency K β := by
  open MeasureTheory in
    have hi (e : E) : (∫ ω, H ω e + D * β e ∂μ) = (∫ ω, H ω e ∂μ) + D * β e := by
      rw [integral_add (hH e) (integrable_const _)]
      simp
    simp only [ReferencePaths.meanScore, hi, mul_add, Finset.sum_add_distrib]
    congr 1
    unfold birthFrequency
    simp only [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro e _
    ring

/-- The actual birth-tilted finite path bound, evaluated at a zero-growth
root. The only limit input is the textbook expected growth limit for the
concrete matrix/path product; no tilted inequality is supplied.
`-meanScore` is the visit-weighted environmental log cost. -/
theorem path_root_bound {ι E Ω : Type*} [Fintype ι] [Fintype E]
    [MeasurableSpace Ω] (K : ReferencePaths.Kernel ι E)
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasureTheory.MeasurePreserving T μ μ)
    (H : Ω → E → ℝ) (hH : ∀ e, MeasureTheory.Integrable (fun ω => H ω e) μ)
    (β : E → ℝ) (D : ℝ) (hb : 0 < birthFrequency K β)
    (hFurstenbergKesten_at_root : Tendsto
      (ReferencePaths.normalizedGrowth K μ T (fun ω e => H ω e + D * β e))
      atTop (nhds 0)) :
    D ≤ -ReferencePaths.meanScore K μ H / birthFrequency K β := by
  have hi (e : E) : MeasureTheory.Integrable (fun ω => H ω e + D * β e) μ :=
    (hH e).add (MeasureTheory.integrable_const _)
  have h := path_limit_bound K μ T hT (fun ω e => H ω e + D * β e) hi 0 0
    (by simpa using hFurstenbergKesten_at_root)
  rw [meanScore_tilt K μ H hH β D] at h
  apply (le_div_iff₀ hb).mpr
  linarith

/-- The ring visit weights reduce to transfer and staying log penalties. -/
theorem ring_visit_weights {ι : Type*} (S : Finset ι) (τ lg ls : ι → ℝ)
    (T : ℝ) (hT : T ≠ 0) :
    T * (∑ i ∈ S, (lg i / T + (τ i - 1) / T * ls i)) =
      (∑ i ∈ S, lg i) + ∑ i ∈ S, (τ i - 1) * ls i := by
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  field_simp

/-- Finite path-space Jensen is proved directly, for arbitrary real path scores. -/
theorem finite_path_jensen {ι : Type*} (S : Finset ι) (w score : ι → ℝ)
    (hw : ∀ i ∈ S, 0 ≤ w i) (hsum : ∑ i ∈ S, w i = 1) :
    (∑ i ∈ S, w i * score i) ≤
      Real.log (∑ i ∈ S, w i * Real.exp (score i)) := by
  have h := strictConcaveOn_log_Ioi.concaveOn.le_map_sum hw hsum
    (fun i (_hi : i ∈ S) => Real.exp_pos (score i))
  simpa only [smul_eq_mul, Real.log_exp] using h

#print axioms path_limit_bound
#print axioms path_root_bound

end Dobson.Coverage2Bounds
