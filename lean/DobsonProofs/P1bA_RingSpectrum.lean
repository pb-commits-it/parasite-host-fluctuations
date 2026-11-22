import DobsonProofs.Coverage1Lineage
import DobsonProofs.Coverage3Renewal
import DobsonProofs.Coverage6Remainders
import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-! # Part A: renewal mixtures and spectral-kernel calculations

This module contains discrete and finite-chain return generating functions,
finite-path cost mixtures, AR(2) recurrence formulas and spectral-kernel
calculations under the displayed hypotheses.
Coverage correction, 2026-11-22: this module is uncredited in the version-2
inventory. `prop:ring-spectrum` retains its recorded partial-P1 scope;
these supplementary declarations do not establish a credited closure of
`part1:41` or an upgrade of the physical growth statement. -/

namespace Dobson.P1bA_RingSpectrum

open scoped BigOperators Topology Real
open Filter MeasureTheory Set Dobson.FiniteLineageVariance

/-! ## Section 1: Single-edge score helpers and discrete renewal PGF -/

section HelpersAndDiscreteRenewal

/-- Score supported on a single designated edge `a : E`. -/
def singleEdgeScore {E Ω : Type*} [DecidableEq E] (a : E) (ξ : Ω → ℝ) : E → Ω → ℝ :=
  fun e ω => if e = a then ξ ω else 0

lemma singleEdgeScore_memLp {E Ω : Type*} [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) (a : E) (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ) (e : E) :
    MemLp (singleEdgeScore a ξ e) 2 μ := by
  unfold singleEdgeScore
  split_ifs
  · exact hξ
  · exact MemLp.zero

lemma singleEdgeScore_lagCovariance {E Ω : Type*} [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) (S : Ω → Ω) (a : E) (ξ : Ω → ℝ) (e e' : E) (m : ℕ) :
    lagCovariance μ S (singleEdgeScore a ξ) e e' m =
      if e = a ∧ e' = a then ∫ ω, ξ (S^[m] ω) * ξ ω ∂μ else 0 := by
  by_cases he : e = a <;> by_cases he' : e' = a <;>
    simp [lagCovariance, singleEdgeScore, he, he']

lemma singleEdgeScore_covarianceCoefficient {E Ω : Type*} [Fintype E] [DecidableEq E]
    [MeasurableSpace Ω] (μ : Measure Ω) (K : Chain E) (S : Ω → Ω) (a : E) (ξ : Ω → ℝ) :
    Dobson.FinitePathCoefficient.covarianceCoefficient μ K S (singleEdgeScore a ξ) =
      (1 / 2 : ℝ) * (K.initial a ^ 2 * ∫ ω, ξ ω * ξ ω ∂μ) -
        ∑' m : ℕ, (∫ ω, ξ (S^[m + 1] ω) * ξ ω ∂μ) * K.initial a *
          ((K.prob ^ (m + 1)) a a - K.initial a) := by
  unfold Dobson.FinitePathCoefficient.covarianceCoefficient
  simp only [singleEdgeScore_lagCovariance, Function.iterate_zero_apply,
    ite_and, mul_ite, ite_mul, mul_zero, zero_mul,
    Finset.sum_ite_irrel, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, Finset.sum_const_zero]
  ring_nf

lemma chain_pow_nonneg {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (n : ℕ) (i j : E) : 0 ≤ (K.prob ^ n) i j := by
  induction n generalizing i j with
  | zero => simp only [pow_zero, Matrix.one_apply]; split_ifs <;> norm_num
  | succ n ih =>
    rw [pow_succ', Matrix.mul_apply]
    exact Finset.sum_nonneg fun k _ => mul_nonneg (K.nonneg i k) (ih k j)

lemma chain_pow_row_sum {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (n : ℕ) (i : E) : ∑ j, (K.prob ^ n) i j = 1 := by
  induction n generalizing i with
  | zero => simp [Matrix.one_apply]
  | succ n ih =>
    simp only [pow_succ', Matrix.mul_apply]
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, ih, mul_one, K.total]

lemma chain_pow_le_one {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (n : ℕ) (i j : E) : (K.prob ^ n) i j ≤ 1 := by
  rw [← chain_pow_row_sum K n i]
  exact Finset.single_le_sum (fun k _ => chain_pow_nonneg K n i k) (Finset.mem_univ j)

lemma chain_initial_le_one {E : Type*} [Fintype E] (K : Chain E) (j : E) : K.initial j ≤ 1 := by
  rw [← K.initial_total]
  exact Finset.single_le_sum (fun k _ => K.initial_nonneg k) (Finset.mem_univ j)

lemma chain_pow_sub_initial_abs_le_one {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (n : ℕ) (i j : E) : |(K.prob ^ n) i j - K.initial j| ≤ 1 := by
  apply abs_le.mpr
  have h1 := chain_pow_nonneg K n i j
  have h2 := chain_pow_le_one K n i j
  have h3 := K.initial_nonneg j
  have h4 := chain_initial_le_one K j
  constructor <;> linarith

/-- Complex discounted return series `U(z) = ∑' m : ℕ, z ^ (m + 1) * u (m + 1)`. -/
noncomputable def complexVisitSeries (u : ℕ → ℝ) (z : ℂ) : ℂ :=
  ∑' m : ℕ, z ^ (m + 1) * (u (m + 1) : ℂ)

/-- Complex positive-lag geometric series `G(z) = ∑' m : ℕ, z ^ (m + 1)`. -/
noncomputable def complexGeomSeries (z : ℂ) : ℂ :=
  ∑' m : ℕ, z ^ (m + 1)

lemma complexGeomSeries_summable (z : ℂ) (hz : ‖z‖ < 1) :
    Summable (fun m : ℕ => z ^ (m + 1)) := by
  have h := (summable_geometric_of_norm_lt_one hz).mul_left z
  refine h.congr fun m => ?_
  rw [pow_succ']

lemma complexGeomSeries_hasSum (z : ℂ) (hz : ‖z‖ < 1) :
    HasSum (fun m : ℕ => z ^ (m + 1)) (z / (1 - z)) := by
  simpa only [pow_succ', div_eq_mul_inv] using
    (hasSum_geometric_of_norm_lt_one hz).mul_left z

lemma complexGeomSeries_eq (z : ℂ) (hz : ‖z‖ < 1) :
    complexGeomSeries z = z / (1 - z) :=
  (complexGeomSeries_hasSum z hz).tsum_eq

lemma complexVisitSeries_summable_of_bounded (u : ℕ → ℝ) (Bu : ℝ)
    (hu : ∀ m : ℕ, |u (m + 1)| ≤ Bu) (z : ℂ) (hz : ‖z‖ < 1) :
    Summable (fun m : ℕ => z ^ (m + 1) * (u (m + 1) : ℂ)) := by
  apply Summable.of_norm_bounded
    (((summable_geometric_of_lt_one (norm_nonneg z) hz).mul_left (Bu * ‖z‖)))
  intro m
  rw [norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs, pow_succ]
  calc ‖z‖ ^ m * ‖z‖ * |u (m + 1)|
      ≤ ‖z‖ ^ m * ‖z‖ * Bu :=
        mul_le_mul_of_nonneg_left (hu m) (mul_nonneg (pow_nonneg (norm_nonneg z) m) (norm_nonneg z))
    _ = (Bu * ‖z‖) * ‖z‖ ^ m := by ring

/-- Convolution power `p^{*r}` of a discrete gap distribution `p : ℕ → ℝ`. -/
noncomputable def gapConvPow (p : ℕ → ℝ) : ℕ → ℕ → ℝ
  | 0, k => if k = 0 then 1 else 0
  | r + 1, k => ∑ j ∈ Finset.range (k + 1), gapConvPow p r j * p (k - j)

lemma gapConvPow_nonneg (p : ℕ → ℝ) (hp : ∀ k, 0 ≤ p k) :
    ∀ r k, 0 ≤ gapConvPow p r k
  | 0, k => by simp only [gapConvPow]; split_ifs <;> norm_num
  | r + 1, k => Finset.sum_nonneg fun j _ => mul_nonneg (gapConvPow_nonneg p hp r j) (hp _)

lemma gapConvPow_zero_at_zero (p : ℕ → ℝ) (hp0 : p 0 = 0) (r : ℕ) :
    gapConvPow p (r + 1) 0 = 0 := by
  simp [gapConvPow, hp0]

/-- Generating function of a positive-integer gap distribution `p` (`p 0 = 0`):
for `‖z‖ < 1`, `‖∑' k, p k * z ^ k‖ < 1` and the renewal visit probability
`u(ℓ) = ∑' r : ℕ, p^{*(r+1)}(ℓ)` satisfies `complexVisitSeries u z = R_e(z) / (1 - R_e(z))`. -/
theorem complex_renewal_pgf (p : ℕ → ℝ) (hp0 : p 0 = 0) (hp_nn : ∀ k, 0 ≤ p k)
    (hp_sum : HasSum p 1) (z : ℂ) (hz : ‖z‖ < 1) :
    let R : ℂ := ∑' k : ℕ, (p k : ℂ) * z ^ k
    let u : ℕ → ℝ := fun ℓ => ∑' r : ℕ, gapConvPow p (r + 1) ℓ
    ‖R‖ < 1 ∧
      (∀ ℓ : ℕ, 0 ≤ u (ℓ + 1) ∧ u (ℓ + 1) ≤ 1) ∧
      HasSum (fun m : ℕ => z ^ (m + 1) * (u (m + 1) : ℂ)) (R / (1 - R)) ∧
      complexVisitSeries u z = R / (1 - R) := by
  intro R u
  have hp_summable : Summable p := hp_sum.summable
  have hp_tsum : ∑' k : ℕ, p k = 1 := hp_sum.tsum_eq
  have hpz_norm_sum : Summable (fun k : ℕ => ‖(p k : ℂ) * z ^ k‖) := by
    refine Summable.of_nonneg_of_le (fun _ => norm_nonneg _) (fun k => ?_) hp_summable
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (hp_nn k), norm_pow]
    exact mul_le_of_le_one_right (hp_nn k) (pow_le_one₀ (norm_nonneg z) hz.le)
  have hR_norm_lt : ‖R‖ < 1 := by
    have hle1 : ‖R‖ ≤ ∑' k : ℕ, ‖(p k : ℂ) * z ^ k‖ :=
      norm_tsum_le_tsum_norm hpz_norm_sum
    have hle2 : (∑' k : ℕ, ‖(p k : ℂ) * z ^ k‖) ≤ ∑' k : ℕ, p k * ‖z‖ := by
      refine hpz_norm_sum.tsum_le_tsum (fun k => ?_) (hp_summable.mul_right ‖z‖)
      cases k with
      | zero => simp [hp0]
      | succ k =>
        rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (hp_nn _),
          norm_pow, pow_succ]
        have hpow : ‖z‖ ^ k ≤ 1 := pow_le_one₀ (norm_nonneg z) hz.le
        calc p (k + 1) * (‖z‖ ^ k * ‖z‖)
            = (p (k + 1) * ‖z‖) * ‖z‖ ^ k := by ring
          _ ≤ (p (k + 1) * ‖z‖) * 1 :=
              mul_le_mul_of_nonneg_left hpow (mul_nonneg (hp_nn _) (norm_nonneg z))
          _ = p (k + 1) * ‖z‖ := mul_one _
    calc ‖R‖ ≤ ∑' k : ℕ, p k * ‖z‖ := hle1.trans hle2
      _ = (∑' k : ℕ, p k) * ‖z‖ := tsum_mul_right
      _ = ‖z‖ := by rw [hp_tsum, one_mul]
      _ < 1 := hz
  have hsupp : ∀ r k : ℕ, k < r → gapConvPow p r k = 0 := by
    intro r
    induction r with
    | zero => intro k hk; omega
    | succ r ih =>
      intro k hk
      simp only [gapConvPow]
      apply Finset.sum_eq_zero
      intro j hj
      have hjk : j ≤ k := Nat.le_of_lt_succ (Finset.mem_range.mp hj)
      by_cases hkj : k = j
      · subst hkj; simp [hp0]
      · have hjr : j < r := by omega
        rw [ih j hjr, zero_mul]
  have hu_summable : ∀ ℓ : ℕ, Summable (fun r : ℕ => gapConvPow p (r + 1) ℓ) := by
    intro ℓ
    refine summable_of_ne_finset_zero (s := Finset.range ℓ) ?_
    intro r hr
    simp only [Finset.mem_range, not_lt] at hr
    exact hsupp (r + 1) ℓ (by omega)
  have hp_fin_le : ∀ s : Finset ℕ, ∑ m ∈ s, p m ≤ 1 := by
    intro s
    rw [← hp_tsum]
    exact hp_summable.sum_le_tsum s (fun m _ => hp_nn m)
  have hu_rec : ∀ k : ℕ, 1 ≤ k →
      u k = p k + ∑ j ∈ Finset.Ico 1 k, u j * p (k - j) := by
    intro k _hk
    dsimp [u]
    rw [← (hu_summable k).sum_add_tsum_nat_add 1]
    simp only [Finset.sum_range_one, zero_add]
    have hfirst : gapConvPow p 1 k = p k := by
      simp only [gapConvPow, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_range,
        Nat.zero_lt_succ, ↓reduceIte, Nat.sub_zero]
    rw [hfirst]
    congr 1
    have hstep : ∀ r : ℕ, gapConvPow p (r + 2) k =
        ∑ j ∈ Finset.Ico 1 k, gapConvPow p (r + 1) j * p (k - j) := by
      intro r
      change (∑ j ∈ Finset.range (k + 1), gapConvPow p (r + 1) j * p (k - j)) = _
      have hsub : Finset.Ico 1 k ⊆ Finset.range (k + 1) := by
        intro j hj; simp only [Finset.mem_Ico, Finset.mem_range] at hj ⊢; omega
      rw [← Finset.sum_subset hsub]
      intro j hj hnot
      simp only [Finset.mem_range, Finset.mem_Ico, not_and, not_lt] at hj hnot
      by_cases hj0 : j = 0
      · subst hj0; rw [gapConvPow_zero_at_zero p hp0 r, zero_mul]
      · have hjk : j = k := by omega
        subst hjk; simp [hp0]
    simp_rw [hstep]
    rw [Summable.tsum_finsetSum (fun j _ => (hu_summable j).mul_right (p (k - j)))]
    refine Finset.sum_congr rfl fun j _ => tsum_mul_right
  have hu_bounds : ∀ ℓ : ℕ, 0 ≤ u (ℓ + 1) ∧ u (ℓ + 1) ≤ 1 := by
    intro ℓ
    refine ⟨tsum_nonneg fun r => gapConvPow_nonneg p hp_nn (r + 1) (ℓ + 1), ?_⟩
    induction ℓ using Nat.strong_induction_on with
    | h n ih =>
      rw [hu_rec (n + 1) (by omega)]
      have hle_sum : ∑ j ∈ Finset.Ico 1 (n + 1), u j * p (n + 1 - j) ≤
          ∑ j ∈ Finset.Ico 1 (n + 1), p (n + 1 - j) := by
        refine Finset.sum_le_sum fun j hj => ?_
        simp only [Finset.mem_Ico] at hj
        obtain ⟨m, rfl⟩ : ∃ m, j = m + 1 := ⟨j - 1, by omega⟩
        exact mul_le_of_le_one_left (hp_nn _) (ih m (by omega))
      have hbij : ∑ j ∈ Finset.Ico 1 (n + 1), p (n + 1 - j) =
          ∑ m ∈ Finset.Icc 1 n, p m := by
        refine Finset.sum_bij (fun j _ => n + 1 - j) ?_ ?_ ?_ ?_
        · intro j hj; simp only [Finset.mem_Ico, Finset.mem_Icc] at hj ⊢; omega
        · intro j₁ hj₁ j₂ hj₂ heq; simp only [Finset.mem_Ico] at hj₁ hj₂; omega
        · intro m hm; simp only [Finset.mem_Icc] at hm
          exact ⟨n + 1 - m, Finset.mem_Ico.mpr ⟨by omega, by omega⟩, by omega⟩
        · intro _ _; rfl
      have hins : p (n + 1) + ∑ m ∈ Finset.Icc 1 n, p m =
          ∑ m ∈ Finset.Icc 1 (n + 1), p m := by
        have hunion : insert (n + 1) (Finset.Icc 1 n) = Finset.Icc 1 (n + 1) := by
          ext m; simp only [Finset.mem_insert, Finset.mem_Icc]; omega
        have hnotmem : n + 1 ∉ Finset.Icc 1 n := by simp [Finset.mem_Icc]
        rw [← hunion, Finset.sum_insert hnotmem]
      linarith [hp_fin_le (Finset.Icc 1 (n + 1))]
  have hU_summable : Summable (fun m : ℕ => z ^ (m + 1) * (u (m + 1) : ℂ)) :=
    complexVisitSeries_summable_of_bounded u 1 (fun m => by
      rw [abs_of_nonneg (hu_bounds m).1]; exact (hu_bounds m).2) z hz
  have hU_norm_sum : Summable (fun k : ℕ => ‖(if k = 0 then (0 : ℂ) else (u k : ℂ) * z ^ k)‖) := by
    have hshift : Summable (fun m : ℕ => ‖z ^ (m + 1) * (u (m + 1) : ℂ)‖) := hU_summable.norm
    rw [← summable_nat_add_iff 1]
    refine hshift.congr fun m => ?_
    simp [mul_comm]
  let f : ℕ → ℂ := fun k => (p k : ℂ) * z ^ k
  let g : ℕ → ℂ := fun k => if k = 0 then 0 else (u k : ℂ) * z ^ k
  have hf_tsum : ∑' k : ℕ, f k = R := rfl
  have hg_tsum : ∑' k : ℕ, g k = complexVisitSeries u z := by
    rw [← (hU_norm_sum.of_norm).sum_add_tsum_nat_add 1, Finset.sum_range_one]
    dsimp [g]
    rw [zero_add]
    unfold complexVisitSeries
    refine tsum_congr fun m => ?_
    ring
  have hcauchy := tsum_mul_tsum_eq_tsum_sum_range_of_summable_norm hpz_norm_sum hU_norm_sum
  change (∑' k, f k) * (∑' k, g k) = ∑' n, ∑ j ∈ Finset.range (n + 1), f j * g (n - j) at hcauchy
  have hterm_eq : ∀ n : ℕ, f n + ∑ j ∈ Finset.range (n + 1), f j * g (n - j) = g n := by
    intro n
    cases n with
    | zero => simp [f, g, hp0]
    | succ m =>
      have hk1 : 1 ≤ m + 1 := by omega
      have hrec_c : (u (m + 1) : ℂ) = (p (m + 1) : ℂ) +
          ∑ l ∈ Finset.Ico 1 (m + 1), (u l : ℂ) * (p (m + 1 - l) : ℂ) := by
        exact_mod_cast hu_rec (m + 1) hk1
      simp only [g, Nat.add_eq_zero_iff, one_ne_zero, and_false, ↓reduceIte]
      rw [hrec_c, add_mul, Finset.sum_mul]
      dsimp [f]
      congr 1
      have hsub : Finset.Ico 1 (m + 1) ⊆ Finset.range (m + 2) := by
        intro j hj; simp only [Finset.mem_Ico, Finset.mem_range] at hj ⊢; omega
      rw [← Finset.sum_subset hsub]
      · refine Finset.sum_bij (fun j _ => m + 1 - j) ?_ ?_ ?_ ?_
        · intro j hj; simp only [Finset.mem_Ico] at hj ⊢; omega
        · intro j₁ hj₁ j₂ hj₂ heq; simp only [Finset.mem_Ico] at hj₁ hj₂; omega
        · intro l hl; simp only [Finset.mem_Ico] at hl
          exact ⟨m + 1 - l, Finset.mem_Ico.mpr ⟨by omega, by omega⟩, by omega⟩
        · intro j hj
          simp only [Finset.mem_Ico] at hj
          have hpos : m + 1 - j ≠ 0 := by omega
          have hsub2 : m + 1 - (m + 1 - j) = j := by omega
          simp only [hpos, ↓reduceIte, hsub2]
          have hpow : z ^ j * z ^ (m + 1 - j) = z ^ (m + 1) := by
            rw [← pow_add]; congr 1; omega
          calc (p j : ℂ) * z ^ j * ((u (m + 1 - j) : ℂ) * z ^ (m + 1 - j))
              = (u (m + 1 - j) : ℂ) * (p j : ℂ) * (z ^ j * z ^ (m + 1 - j)) := by ring
            _ = (u (m + 1 - j) : ℂ) * (p j : ℂ) * z ^ (m + 1) := by rw [hpow]
      · intro j hj hnot
        simp only [Finset.mem_range, Finset.mem_Ico, not_and, not_lt] at hj hnot
        by_cases hj0 : j = 0
        · subst hj0; simp [hp0]
        · have hjm : j = m + 1 := by omega
          subst hjm; simp
  have hprod_sum : Summable (fun n : ℕ => ∑ j ∈ Finset.range (n + 1), f j * g (n - j)) := by
    have heq : (fun n : ℕ => ∑ j ∈ Finset.range (n + 1), f j * g (n - j)) =
        fun n : ℕ => g n - f n := by
      funext n; linear_combination hterm_eq n
    rw [heq]
    exact (hU_norm_sum.of_norm).sub hpz_norm_sum.of_norm
  have hrenewal : complexVisitSeries u z = R + R * complexVisitSeries u z := by
    rw [← hf_tsum, ← hg_tsum, hcauchy, ← Summable.tsum_add hpz_norm_sum.of_norm hprod_sum]
    exact (tsum_congr hterm_eq).symm
  have h1R : 1 - R ≠ 0 := by
    intro h0
    have h1 : R = 1 := by linear_combination -h0
    rw [h1, norm_one] at hR_norm_lt
    exact lt_irrefl 1 hR_norm_lt
  have hsol : complexVisitSeries u z = R / (1 - R) :=
    Dobson.Coverage3Renewal.renewal_solution (complexVisitSeries u z) R h1R hrenewal
  refine ⟨hR_norm_lt, hu_bounds, ?_, hsol⟩
  rw [← hsol]
  exact hU_summable.hasSum

end HelpersAndDiscreteRenewal

/-! ## Section 2: Markov-chain first-return PGF, Kac's mean gap, and `part1:41` -/

section ChainRenewalAndMixture

variable {E : Type*} [Fintype E] [DecidableEq E]

/-- First hitting probability at time `n` (allowing `n = 0` at target `a`). -/
noncomputable def hit (K : Chain E) (a : E) : ℕ → E → ℝ
  | 0, i => if i = a then 1 else 0
  | n + 1, i => if i = a then 0 else ∑ j, K.prob i j * hit K a n j

/-- Positive-integer first-return probability mass function `L_a` on `{1, 2, ...}`. -/
noncomputable def firstReturnProb (K : Chain E) (a : E) : ℕ → ℝ
  | 0 => 0
  | m + 1 => ∑ j, K.prob a j * hit K a m j

lemma hit_bounds (K : Chain E) (a : E) (n : ℕ) (i : E) :
    0 ≤ hit K a n i ∧ hit K a n i ≤ 1 := by
  induction n generalizing i with
  | zero => simp only [hit]; split_ifs <;> norm_num
  | succ n ih =>
    simp only [hit]
    split_ifs
    · norm_num
    · refine ⟨Finset.sum_nonneg fun j _ => mul_nonneg (K.nonneg i j) (ih j).1, ?_⟩
      calc ∑ j, K.prob i j * hit K a n j
          ≤ ∑ j, K.prob i j * 1 :=
            Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (ih j).2 (K.nonneg i j)
        _ = 1 := by simp [K.total]

lemma firstReturnProb_nonneg (K : Chain E) (a : E) (k : ℕ) :
    0 ≤ firstReturnProb K a k := by
  cases k with
  | zero => simp [firstReturnProb]
  | succ m =>
    exact Finset.sum_nonneg fun j _ => mul_nonneg (K.nonneg a j) (hit_bounds K a m j).1

lemma firstReturnProb_le_one (K : Chain E) (a : E) (k : ℕ) :
    firstReturnProb K a k ≤ 1 := by
  cases k with
  | zero => simp [firstReturnProb]
  | succ m =>
    calc ∑ j, K.prob a j * hit K a m j
        ≤ ∑ j, K.prob a j * 1 :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hit_bounds K a m j).2 (K.nonneg a j)
      _ = 1 := by simp [K.total]

/-- Complex Green's function `∑' n, z ^ n * (K.prob ^ n) i a`. -/
noncomputable def green (K : Chain E) (a : E) (z : ℂ) (i : E) : ℂ :=
  ∑' n : ℕ, z ^ n * ((K.prob ^ n) i a : ℂ)

/-- Complex first-hitting PGF `∑' n, z ^ n * hit K a n i`. -/
noncomputable def hitTransform (K : Chain E) (a : E) (z : ℂ) (i : E) : ℂ :=
  ∑' n : ℕ, z ^ n * (hit K a n i : ℂ)

/-- Complex positive first-return PGF `R_a(z) = E_a[z^{L_a}]`. -/
noncomputable def returnTransform (K : Chain E) (a : E) (z : ℂ) : ℂ :=
  z * ∑ j, (K.prob a j : ℂ) * hitTransform K a z j

private lemma unit_weighted_summable (f : ℕ → ℝ) (hf : ∀ n, 0 ≤ f n ∧ f n ≤ 1)
    (z : ℂ) (hz : ‖z‖ < 1) : Summable (fun n : ℕ => z ^ n * (f n : ℂ)) := by
  apply Summable.of_norm_bounded (summable_geometric_of_lt_one (norm_nonneg z) hz)
  intro n
  rw [norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (hf n).1]
  exact mul_le_of_le_one_right (pow_nonneg (norm_nonneg z) n) (hf n).2

lemma green_summable (K : Chain E) (a : E) (z : ℂ) (hz : ‖z‖ < 1) (i : E) :
    Summable (fun n : ℕ => z ^ n * ((K.prob ^ n) i a : ℂ)) :=
  unit_weighted_summable _ (fun n => ⟨chain_pow_nonneg K n i a, chain_pow_le_one K n i a⟩) z hz

lemma hit_summable (K : Chain E) (a : E) (z : ℂ) (hz : ‖z‖ < 1) (i : E) :
    Summable (fun n : ℕ => z ^ n * (hit K a n i : ℂ)) :=
  unit_weighted_summable _ (fun n => hit_bounds K a n i) z hz

lemma returnTransform_eq_tsum (K : Chain E) (a : E) (z : ℂ) (hz : ‖z‖ < 1) :
    returnTransform K a z = ∑' k : ℕ, (firstReturnProb K a k : ℂ) * z ^ k := by
  have hsum : Summable (fun k : ℕ => (firstReturnProb K a k : ℂ) * z ^ k) := by
    refine (unit_weighted_summable _
      (fun k => ⟨firstReturnProb_nonneg K a k, firstReturnProb_le_one K a k⟩) z hz).congr ?_
    intro k; ring
  rw [← hsum.sum_add_tsum_nat_add 1]
  simp only [Finset.sum_range_one, firstReturnProb.eq_1, Complex.ofReal_zero, pow_zero, mul_one,
    zero_add]
  unfold returnTransform hitTransform
  rw [Finset.mul_sum]
  have hj_sum : ∀ j : E, Summable (fun m : ℕ => (K.prob a j : ℂ) * (hit K a m j : ℂ) * z ^ (m + 1)) := by
    intro j
    refine ((hit_summable K a z hz j).mul_left (z * (K.prob a j : ℂ))).congr fun m => ?_
    rw [pow_succ']; ring
  have hswap : (∑' m : ℕ, (firstReturnProb K a (m + 1) : ℂ) * z ^ (m + 1)) =
      ∑ j : E, ∑' m : ℕ, (K.prob a j : ℂ) * (hit K a m j : ℂ) * z ^ (m + 1) := by
    rw [← Summable.tsum_finsetSum (fun j _ => hj_sum j)]
    refine tsum_congr fun m => ?_
    simp only [firstReturnProb, Complex.ofReal_sum, Complex.ofReal_mul, Finset.sum_mul]
  rw [hswap]
  refine Finset.sum_congr rfl fun j _ => ?_
  calc z * ((K.prob a j : ℂ) * ∑' n : ℕ, z ^ n * (hit K a n j : ℂ))
      = (z * (K.prob a j : ℂ)) * ∑' n : ℕ, z ^ n * (hit K a n j : ℂ) := (mul_assoc _ _ _).symm
    _ = ∑' n : ℕ, (z * (K.prob a j : ℂ)) * (z ^ n * (hit K a n j : ℂ)) := tsum_mul_left.symm
    _ = ∑' m : ℕ, (K.prob a j : ℂ) * (hit K a m j : ℂ) * z ^ (m + 1) :=
        tsum_congr fun m => by rw [pow_succ']; ring

/-- Exact Markov-chain renewal transform (`‖z‖ < 1`), proved by maximum-norm Dirichlet uniqueness
without assuming a renewal equation. -/
theorem chain_return_series (K : Chain E) (a : E) (z : ℂ) (hz : ‖z‖ < 1) :
    HasSum (fun m : ℕ => z ^ (m + 1) * ((K.prob ^ (m + 1)) a a : ℂ))
      (returnTransform K a z / (1 - returnTransform K a z)) ∧
    complexVisitSeries (fun ℓ => (K.prob ^ ℓ) a a) z =
      returnTransform K a z / (1 - returnTransform K a z) := by
  have hgreen_eq : ∀ i : E, green K a z i = (if i = a then 1 else 0) +
      z * ∑ j, (K.prob i j : ℂ) * green K a z j := by
    intro i
    unfold green
    rw [(green_summable K a z hz i).tsum_eq_zero_add]
    congr 1
    · simp [Matrix.one_apply, apply_ite]
    simp_rw [pow_succ' K.prob, Matrix.mul_apply, Complex.ofReal_sum, Complex.ofReal_mul,
      Finset.mul_sum]
    rw [Summable.tsum_finsetSum]
    · refine Finset.sum_congr rfl fun j _ => ?_
      calc (∑' n : ℕ, z ^ (n + 1) * ((K.prob i j : ℂ) * ((K.prob ^ n) j a : ℂ)))
          = ∑' n : ℕ, (z * (K.prob i j : ℂ)) * (z ^ n * ((K.prob ^ n) j a : ℂ)) :=
            tsum_congr fun n => by rw [pow_succ' z]; ring
        _ = z * ((K.prob i j : ℂ) * ∑' n : ℕ, z ^ n * ((K.prob ^ n) j a : ℂ)) := by
            rw [tsum_mul_left]; ring
    · intro j _
      exact ((green_summable K a z hz j).mul_left (z * (K.prob i j : ℂ))).congr fun n => by
        rw [pow_succ' z]; ring
  have hhit_self : hitTransform K a z a = 1 := by
    unfold hitTransform
    rw [(hit_summable K a z hz a).tsum_eq_zero_add]
    simp [hit]
  have hhit_eq : ∀ i : E, i ≠ a →
      hitTransform K a z i = z * ∑ j, (K.prob i j : ℂ) * hitTransform K a z j := by
    intro i hi
    unfold hitTransform
    rw [(hit_summable K a z hz i).tsum_eq_zero_add]
    simp only [hit, hi, ↓reduceIte, Complex.ofReal_zero, mul_zero, zero_add,
      Complex.ofReal_sum, Complex.ofReal_mul]
    simp_rw [Finset.mul_sum]
    rw [Summable.tsum_finsetSum]
    · refine Finset.sum_congr rfl fun j _ => ?_
      calc (∑' n : ℕ, z ^ (n + 1) * ((K.prob i j : ℂ) * (hit K a n j : ℂ)))
          = ∑' n : ℕ, (z * (K.prob i j : ℂ)) * (z ^ n * (hit K a n j : ℂ)) :=
            tsum_congr fun n => by rw [pow_succ' z]; ring
        _ = z * ((K.prob i j : ℂ) * ∑' n : ℕ, z ^ n * (hit K a n j : ℂ)) := by
            rw [tsum_mul_left]; ring
    · intro j _
      exact ((hit_summable K a z hz j).mul_left (z * (K.prob i j : ℂ))).congr fun n => by
        rw [pow_succ' z]; ring
  let d : E → ℂ := fun i => green K a z i - hitTransform K a z i * green K a z a
  have hd_zero : ∀ i : E, d i = 0 := by
    have : Nonempty E := ⟨a⟩
    obtain ⟨k, _, hk⟩ := Finset.exists_max_image Finset.univ (fun i => ‖d i‖) Finset.univ_nonempty
    have hb (i : E) : ‖d i‖ ≤ ‖d k‖ := hk i (Finset.mem_univ i)
    have hk0 : ‖d k‖ = 0 := by
      by_cases hka : k = a
      · subst hka; simp [d, hhit_self]
      · have hdk : d k = z * ∑ j, (K.prob k j : ℂ) * d j := by
          dsimp [d]
          rw [hgreen_eq k, hhit_eq k hka]
          simp only [hka, ↓reduceIte, zero_add, mul_sub, Finset.sum_sub_distrib,
            ← mul_assoc, ← Finset.sum_mul]
        have hsum_le : ‖∑ j, (K.prob k j : ℂ) * d j‖ ≤ ‖d k‖ := by
          calc ‖∑ j, (K.prob k j : ℂ) * d j‖
              ≤ ∑ j, ‖(K.prob k j : ℂ) * d j‖ := norm_sum_le _ _
            _ ≤ ∑ j, K.prob k j * ‖d k‖ := Finset.sum_le_sum fun j _ => by
                rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (K.nonneg k j)]
                exact mul_le_mul_of_nonneg_left (hb j) (K.nonneg k j)
            _ = ‖d k‖ := by rw [← Finset.sum_mul, K.total, one_mul]
        have hh : ‖d k‖ ≤ ‖z‖ * ‖d k‖ := by
          calc ‖d k‖ = ‖z‖ * ‖∑ j, (K.prob k j : ℂ) * d j‖ := by rw [hdk, norm_mul]
            _ ≤ ‖z‖ * ‖d k‖ := mul_le_mul_of_nonneg_left hsum_le (norm_nonneg z)
        nlinarith [norm_nonneg (d k)]
    intro i
    exact norm_eq_zero.mp (le_antisymm ((hb i).trans_eq hk0) (norm_nonneg _))
  have hga : green K a z a = 1 + returnTransform K a z * green K a z a := by
    have hge := hgreen_eq a
    simp only [↓reduceIte] at hge
    calc green K a z a
        = 1 + z * ∑ j, (K.prob a j : ℂ) * green K a z j := hge
      _ = 1 + z * ∑ j, (K.prob a j : ℂ) * (hitTransform K a z j * green K a z a) := by
          congr 2; refine Finset.sum_congr rfl fun j _ => ?_
          rw [sub_eq_zero.mp (hd_zero j)]
      _ = 1 + returnTransform K a z * green K a z a := by
          simp only [returnTransform, ← mul_assoc, ← Finset.sum_mul]
  have h1R : 1 - returnTransform K a z ≠ 0 := by
    intro h0
    have hR1 : returnTransform K a z = 1 := by linear_combination -h0
    rw [hR1, one_mul] at hga
    have : (1 : ℂ) = 0 := by linear_combination -hga
    exact one_ne_zero this
  have hsol : green K a z a - 1 = returnTransform K a z / (1 - returnTransform K a z) := by
    apply Dobson.Coverage3Renewal.renewal_solution _ _ h1R
    linear_combination hga
  have hhas : HasSum (fun m : ℕ => z ^ (m + 1) * ((K.prob ^ (m + 1)) a a : ℂ))
      (returnTransform K a z / (1 - returnTransform K a z)) := by
    rw [← hsol]
    have hs := (hasSum_nat_add_iff' 1).mpr (green_summable K a z hz a).hasSum
    simpa [green, Matrix.one_apply] using hs
  exact ⟨hhas, hhas.tsum_eq⟩

/-- **Kac's recurrence-time theorem on `Chain E`.**
Under geometric return mixing `|(K.prob ^ (m + 1)) a a - K.initial a| ≤ Ku * β ^ m` (`0 ≤ β < 1`)
with `0 < K.initial a`, the first-return distribution `firstReturnProb K a` is a proper probability
law (`HasSum (firstReturnProb K a) 1`) with mean recurrence time `1 / K.initial a`. -/
theorem kac_mean_return_time (K : Chain E) (a : E) (hc : 0 < K.initial a)
    (Ku β : ℝ) (_hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ m : ℕ, |(K.prob ^ (m + 1)) a a - K.initial a| ≤ Ku * β ^ m) :
    HasSum (firstReturnProb K a) 1 ∧
      HasSum (fun m : ℕ => ((m + 1 : ℕ) : ℝ) * firstReturnProb K a (m + 1))
        (1 / K.initial a) := by
  let f : ℕ → ℝ := firstReturnProb K a
  let δ : ℕ → ℝ := fun m => (K.prob ^ (m + 1)) a a - K.initial a
  have hδ_abs : Summable (fun m : ℕ => |δ m|) :=
    Summable.of_nonneg_of_le (fun _ => abs_nonneg _) hmix
      ((summable_geometric_of_lt_one hβ0 hβ1).mul_left Ku)
  let rseq : ℕ → ℝ := fun n => 1 - 1 / ((n : ℝ) + 2)
  have hr_pos : ∀ n : ℕ, 0 < rseq n := by
    intro n; dsimp [rseq]
    have hgt : (1 : ℝ) < (n : ℝ) + 2 := by linarith [Nat.cast_nonneg (α := ℝ) n]
    linarith [div_lt_one (by positivity : (0 : ℝ) < (n : ℝ) + 2) |>.mpr hgt]
  have hr_lt1 : ∀ n : ℕ, rseq n < 1 := by
    intro n; dsimp [rseq]
    have hp : 0 < 1 / ((n : ℝ) + 2) := by positivity
    linarith
  have hr_tend : Tendsto rseq atTop (𝓝 1) := by
    have hinv : Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 2)) atTop (𝓝 0) :=
      tendsto_const_nhds.div_atTop
        (tendsto_natCast_atTop_atTop.atTop_add tendsto_const_nhds)
    simpa [rseq] using (tendsto_const_nhds (x := (1 : ℝ))).sub hinv
  let R_real : ℝ → ℝ := fun r => ∑' k : ℕ, f k * r ^ k
  let D_real : ℝ → ℝ := fun r => ∑' m : ℕ, r ^ (m + 1) * δ m
  have hD_tend : Tendsto (fun n : ℕ => D_real (rseq n)) atTop (𝓝 (∑' m : ℕ, δ m)) := by
    refine tendsto_tsum_of_dominated_convergence hδ_abs (fun m => ?_)
      (Eventually.of_forall fun n m => ?_)
    · simpa using (hr_tend.pow (m + 1)).mul_const (δ m)
    · rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_pos (hr_pos n)]
      exact mul_le_of_le_one_left (abs_nonneg _) (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le)
  -- Real renewal identity at `rseq n`
  have hR_eq : ∀ n : ℕ,
      let r := rseq n
      let den := (1 - r) * (1 + D_real r) + K.initial a * r
      0 ≤ R_real r ∧ R_real r < 1 ∧
      R_real r * den = (1 - r) * D_real r + K.initial a * r ∧
      ((1 - R_real r) / (1 - r)) * den = 1 := by
    intro n r den
    have hnorm : ‖(r : ℂ)‖ < 1 := by
      rw [Complex.norm_real, Real.norm_eq_abs, abs_of_pos (hr_pos n)]
      exact hr_lt1 n
    have hRc : returnTransform K a (r : ℂ) = (R_real r : ℂ) := by
      rw [returnTransform_eq_tsum K a (r : ℂ) hnorm]
      dsimp [R_real, f]
      rw [Complex.ofReal_tsum]
      refine tsum_congr fun k => by push_cast; rfl
    have hser_c := (chain_return_series K a (r : ℂ) hnorm).1
    rw [hRc] at hser_c
    have hser_r : HasSum (fun m : ℕ => r ^ (m + 1) * (K.prob ^ (m + 1)) a a)
        (R_real r / (1 - R_real r)) := by
      apply Complex.hasSum_ofReal.mp
      simpa only [Complex.ofReal_mul, Complex.ofReal_pow, Complex.ofReal_div,
        Complex.ofReal_sub, Complex.ofReal_one] using hser_c
    have hR_nn : 0 ≤ R_real r :=
      tsum_nonneg fun k => mul_nonneg (firstReturnProb_nonneg K a k) (pow_nonneg (hr_pos n).le k)
    have hhit_fin_le : ∀ N : ℕ, ∀ i : E, ∑ k ∈ Finset.range (N + 1), hit K a k i ≤ 1 := by
      intro N
      induction N with
      | zero =>
        intro i; rw [Finset.sum_range_one]; dsimp [hit]; split_ifs <;> linarith
      | succ N ih =>
        intro i
        rw [Finset.sum_range_succ']
        by_cases hia : i = a
        · rw [hia]
          have h0 : ∑ k ∈ Finset.range (N + 1), hit K a (k + 1) a = 0 :=
            Finset.sum_eq_zero fun k _ => by simp [hit]
          rw [h0]; simp [hit]
        · have h0 : hit K a 0 i = 0 := by simp [hit, hia]
          rw [h0, add_zero]
          have hstep : ∑ k ∈ Finset.range (N + 1), hit K a (k + 1) i =
              ∑ j : E, K.prob i j * ∑ k ∈ Finset.range (N + 1), hit K a k j := by
            simp only [hit, hia, ↓reduceIte]
            rw [Finset.sum_comm]
            exact Finset.sum_congr rfl fun j _ => (Finset.mul_sum _ _ _).symm
          rw [hstep]
          calc ∑ j : E, K.prob i j * ∑ k ∈ Finset.range (N + 1), hit K a k j
              ≤ ∑ j : E, K.prob i j * 1 :=
                Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (ih j) (K.nonneg i j)
            _ = 1 := by simp [K.total]
    have hf_range_le : ∀ N : ℕ, ∑ k ∈ Finset.range N, f k ≤ 1 := by
      intro N
      cases N with
      | zero => simp
      | succ N =>
        rw [Finset.sum_range_succ']
        have hf0 : f 0 = 0 := rfl
        rw [hf0, add_zero]
        cases N with
        | zero => simp
        | succ M =>
          have hstep : ∑ m ∈ Finset.range (M + 1), f (m + 1) =
              ∑ j : E, K.prob a j * ∑ m ∈ Finset.range (M + 1), hit K a m j := by
            dsimp [f, firstReturnProb]
            rw [Finset.sum_comm]
            exact Finset.sum_congr rfl fun j _ => (Finset.mul_sum _ _ _).symm
          rw [hstep]
          calc ∑ j : E, K.prob a j * ∑ m ∈ Finset.range (M + 1), hit K a m j
              ≤ ∑ j : E, K.prob a j * 1 :=
                Finset.sum_le_sum fun j _ =>
                  mul_le_mul_of_nonneg_left (hhit_fin_le M j) (K.nonneg a j)
            _ = 1 := by simp [K.total]
    have hf_fin_le : ∀ s : Finset ℕ, ∑ k ∈ s, f k ≤ 1 := by
      intro s
      obtain ⟨N, hN⟩ : ∃ N, s ⊆ Finset.range N :=
        ⟨s.sup id + 1, fun x hx => Finset.mem_range.mpr (Nat.lt_succ_of_le (Finset.le_sup (f := id) hx))⟩
      exact (Finset.sum_le_sum_of_subset_of_nonneg hN fun k _ _ => firstReturnProb_nonneg K a k).trans
        (hf_range_le N)
    have hf_summable : Summable f := summable_of_sum_le (firstReturnProb_nonneg K a) hf_fin_le
    have hf_tsum_le : ∑' k : ℕ, f k ≤ 1 := hf_summable.tsum_le_of_sum_le hf_fin_le
    have hR_lt1 : R_real r < 1 := by
      have hsum_fr : Summable (fun k : ℕ => f k * r ^ k) := by
        refine Summable.of_nonneg_of_le
          (fun k => mul_nonneg (firstReturnProb_nonneg K a k) (pow_nonneg (hr_pos n).le k))
          (fun k => mul_le_of_le_one_right (firstReturnProb_nonneg K a k)
            (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le)) hf_summable
      have hle_r : R_real r ≤ (∑' k : ℕ, f k) * r := by
        rw [← tsum_mul_right]
        refine hsum_fr.tsum_le_tsum (fun k => ?_) (hf_summable.mul_right r)
        cases k with
        | zero => simp [f, firstReturnProb]
        | succ m =>
          rw [pow_succ']
          exact mul_le_mul_of_nonneg_left
            (mul_le_of_le_one_right (hr_pos n).le (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le))
            (firstReturnProb_nonneg K a (m + 1))
      calc R_real r ≤ (∑' k : ℕ, f k) * r := hle_r
        _ ≤ 1 * r := mul_le_mul_of_nonneg_right hf_tsum_le (hr_pos n).le
        _ = r := one_mul r
        _ < 1 := hr_lt1 n
    have hD_has : HasSum (fun m : ℕ => r ^ (m + 1) * δ m) (D_real r) := by
      refine (Summable.of_norm_bounded hδ_abs fun m => ?_).hasSum
      rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_pos (hr_pos n)]
      exact mul_le_of_le_one_left (abs_nonneg _) (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le)
    have hgeom_r : HasSum (fun m : ℕ => r ^ (m + 1) * K.initial a)
        (K.initial a * (r / (1 - r))) := by
      have hg := ((hasSum_geometric_of_lt_one (hr_pos n).le (hr_lt1 n)).mul_left r).mul_right
        (K.initial a)
      convert hg using 1
      · funext m; rw [pow_succ']
      · ring
    have hsplit_ser : D_real r + K.initial a * (r / (1 - r)) = R_real r / (1 - R_real r) := by
      have hcomb := hD_has.add hgeom_r
      have hfn : (fun m : ℕ => r ^ (m + 1) * δ m + r ^ (m + 1) * K.initial a) =
          fun m : ℕ => r ^ (m + 1) * (K.prob ^ (m + 1)) a a := by
        funext m; dsimp [δ]; ring
      rw [hfn] at hcomb
      exact hcomb.unique hser_r
    have h1r_ne : 1 - r ≠ 0 := (sub_pos.mpr (hr_lt1 n)).ne'
    have h1R_ne : 1 - R_real r ≠ 0 := (sub_pos.mpr hR_lt1).ne'
    have hden_eq : den = (1 - r) / (1 - R_real r) := by
      have h1 : 1 + (D_real r + K.initial a * (r / (1 - r))) = 1 + R_real r / (1 - R_real r) :=
        congrArg (1 + ·) hsplit_ser
      have h2 : 1 + R_real r / (1 - R_real r) = 1 / (1 - R_real r) := by
        field_simp [h1R_ne]; ring
      rw [h2] at h1
      dsimp [den]
      calc (1 - r) * (1 + D_real r) + K.initial a * r
          = (1 - r) * (1 + (D_real r + K.initial a * (r / (1 - r)))) := by
              field_simp [h1r_ne]; ring
        _ = (1 - r) * (1 / (1 - R_real r)) := by rw [h1]
        _ = (1 - r) / (1 - R_real r) := by ring
    refine ⟨hR_nn, hR_lt1, ?_, ?_⟩
    · rw [hden_eq]
      have hmul := congrArg (· * (1 - r)) hsplit_ser
      calc R_real r * ((1 - r) / (1 - R_real r))
          = (R_real r / (1 - R_real r)) * (1 - r) := by ring
        _ = (D_real r + K.initial a * (r / (1 - r))) * (1 - r) := hmul.symm
        _ = (1 - r) * D_real r + K.initial a * r := by field_simp [h1r_ne]
    · rw [hden_eq]
      field_simp [h1r_ne, h1R_ne]
  have hhit_le_top : ∀ N : ℕ, ∀ i : E, ∑ k ∈ Finset.range (N + 1), hit K a k i ≤ 1 := by
    intro N
    induction N with
    | zero => intro i; rw [Finset.sum_range_one]; dsimp [hit]; split_ifs <;> linarith
    | succ N ih =>
      intro i; rw [Finset.sum_range_succ']
      by_cases hia : i = a
      · rw [hia]
        rw [Finset.sum_eq_zero fun k _ => by simp [hit]]; simp [hit]
      · rw [show hit K a 0 i = 0 by simp [hit, hia], add_zero]
        have hstep : ∑ k ∈ Finset.range (N + 1), hit K a (k + 1) i =
            ∑ j : E, K.prob i j * ∑ k ∈ Finset.range (N + 1), hit K a k j := by
          simp only [hit, hia, ↓reduceIte]; rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun j _ => (Finset.mul_sum _ _ _).symm
        rw [hstep]
        exact (Finset.sum_le_sum fun j _ =>
          mul_le_mul_of_nonneg_left (ih j) (K.nonneg i j)).trans (by simp [K.total])
  have hf_sum_top : Summable f := by
    refine summable_of_sum_le (c := 1) (firstReturnProb_nonneg K a) fun s => ?_
    obtain ⟨N, hN⟩ : ∃ N, s ⊆ Finset.range (N + 2) :=
      ⟨s.sup id, fun x hx => Finset.mem_range.mpr (by
        have := Finset.le_sup (f := id) hx; dsimp at this; omega)⟩
    refine (Finset.sum_le_sum_of_subset_of_nonneg hN fun k _ _ => firstReturnProb_nonneg K a k).trans ?_
    rw [Finset.sum_range_succ']
    rw [show firstReturnProb K a 0 = 0 from rfl, add_zero]
    have hstep : ∑ m ∈ Finset.range (N + 1), f (m + 1) =
        ∑ j : E, K.prob a j * ∑ m ∈ Finset.range (N + 1), hit K a m j := by
      dsimp [f, firstReturnProb]; rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun j _ => (Finset.mul_sum _ _ _).symm
    rw [hstep]
    exact (Finset.sum_le_sum fun j _ =>
      mul_le_mul_of_nonneg_left (hhit_le_top N j) (K.nonneg a j)).trans (by simp [K.total])
  let den_seq : ℕ → ℝ := fun n =>
    (1 - rseq n) * (1 + D_real (rseq n)) + K.initial a * rseq n
  have h1sub_tend : Tendsto (fun n : ℕ => 1 - rseq n) atTop (𝓝 0) := by
    simpa using (tendsto_const_nhds (x := (1 : ℝ))).sub hr_tend
  have hden_tend : Tendsto den_seq atTop (𝓝 (K.initial a)) := by
    have h1 := h1sub_tend.mul ((tendsto_const_nhds (x := (1 : ℝ))).add hD_tend)
    have h2 := hr_tend.const_mul (K.initial a)
    simpa [den_seq] using h1.add h2
  have hnum_tend : Tendsto (fun n : ℕ => (1 - rseq n) * D_real (rseq n) + K.initial a * rseq n)
      atTop (𝓝 (K.initial a)) := by
    simpa using (h1sub_tend.mul hD_tend).add (hr_tend.const_mul (K.initial a))
  have hR_tend_one : Tendsto (fun n : ℕ => R_real (rseq n)) atTop (𝓝 1) := by
    have hdiv := hnum_tend.div hden_tend hc.ne'
    rw [div_self hc.ne'] at hdiv
    refine hdiv.congr' ?_
    filter_upwards [ hden_tend.eventually_ne hc.ne' ] with n hden_ne
    have heq := (hR_eq n).2.2.1
    exact ((eq_div_iff hden_ne).mpr heq).symm
  have hR_tend_tsum : Tendsto (fun n : ℕ => R_real (rseq n)) atTop (𝓝 (∑' k : ℕ, f k)) := by
    have hlim : Tendsto (fun n : ℕ => ∑' k : ℕ, f k * (rseq n) ^ k) atTop
        (𝓝 (∑' k : ℕ, f k * (1 : ℝ) ^ k)) := by
      refine tendsto_tsum_of_dominated_convergence hf_sum_top (fun k => ?_)
        (Eventually.of_forall fun n k => ?_)
      · exact (hr_tend.pow k).const_mul (f k)
      · rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (firstReturnProb_nonneg K a k),
          abs_pow, abs_of_pos (hr_pos n)]
        exact mul_le_of_le_one_right (firstReturnProb_nonneg K a k)
          (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le)
    simpa [R_real] using hlim
  have hf_tsum_eq : ∑' k : ℕ, f k = 1 := tendsto_nhds_unique hR_tend_tsum hR_tend_one
  have hf_hasSum : HasSum f 1 := hf_tsum_eq ▸ hf_sum_top.hasSum
  -- Second conclusion: mean recurrence time `1 / K.initial a`
  have hquot_tend : Tendsto (fun n : ℕ => (1 - R_real (rseq n)) / (1 - rseq n))
      atTop (𝓝 (1 / K.initial a)) := by
    have hinv := tendsto_const_nhds (x := (1 : ℝ)).div hden_tend hc.ne'
    refine hinv.congr' ?_
    filter_upwards [ hden_tend.eventually_ne hc.ne' ] with n hden_ne
    exact ((eq_div_iff hden_ne).mpr (hR_eq n).2.2.2).symm
  let geomSum : ℕ → ℝ → ℝ := fun m r => ∑ j ∈ Finset.range (m + 1), r ^ j
  have hgeom_id : ∀ n : ℕ, ∀ m : ℕ,
      (1 - (rseq n) ^ (m + 1)) / (1 - rseq n) = geomSum m (rseq n) := by
    intro n m
    have hs := geom_sum_mul (rseq n) (m + 1)
    apply (div_eq_iff (sub_pos.mpr (hr_lt1 n)).ne').mpr
    calc 1 - rseq n ^ (m + 1)
        = -((∑ i ∈ Finset.range (m + 1), rseq n ^ i) * (rseq n - 1)) := by linarith [hs]
      _ = geomSum m (rseq n) * (1 - rseq n) := by dsimp [geomSum]; ring
  have hquot_tsum : ∀ n : ℕ,
      (1 - R_real (rseq n)) / (1 - rseq n) =
        ∑' m : ℕ, f (m + 1) * geomSum m (rseq n) := by
    intro n
    have hf_shift : HasSum (fun m : ℕ => f (m + 1)) 1 := by
      have hs := (hasSum_nat_add_iff' 1).mpr hf_hasSum
      simpa [f, firstReturnProb] using hs
    have hR_shift : HasSum (fun m : ℕ => f (m + 1) * (rseq n) ^ (m + 1)) (R_real (rseq n)) := by
      have hsum_fr : Summable (fun k : ℕ => f k * (rseq n) ^ k) := by
        refine Summable.of_nonneg_of_le
          (fun k => mul_nonneg (firstReturnProb_nonneg K a k) (pow_nonneg (hr_pos n).le k))
          (fun k => mul_le_of_le_one_right (firstReturnProb_nonneg K a k)
            (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le)) hf_sum_top
      have hs := (hasSum_nat_add_iff' 1).mpr hsum_fr.hasSum
      simpa [R_real, f, firstReturnProb] using hs
    have hsub := (hf_shift.sub hR_shift).div_const (1 - rseq n)
    have hcongr : (fun m : ℕ => (f (m + 1) - f (m + 1) * (rseq n) ^ (m + 1)) / (1 - rseq n)) =
        fun m : ℕ => f (m + 1) * geomSum m (rseq n) := by
      funext m
      rw [← hgeom_id n m]
      ring
    rw [hcongr] at hsub
    exact hsub.tsum_eq.symm
  have hmom_fin_le : ∀ s : Finset ℕ,
      ∑ m ∈ s, ((m + 1 : ℕ) : ℝ) * f (m + 1) ≤ 1 / K.initial a := by
    intro s
    have hfin_tend : Tendsto (fun n : ℕ => ∑ m ∈ s, f (m + 1) * geomSum m (rseq n))
        atTop (𝓝 (∑ m ∈ s, ((m + 1 : ℕ) : ℝ) * f (m + 1))) := by
      have ht : Tendsto (fun n : ℕ => ∑ m ∈ s, f (m + 1) * geomSum m (rseq n))
          atTop (𝓝 (∑ m ∈ s, f (m + 1) * geomSum m 1)) :=
        tendsto_finsetSum s fun m _ =>
          (tendsto_finsetSum (Finset.range (m + 1)) fun j _ => hr_tend.pow j).const_mul (f (m + 1))
      have heq_s : (∑ m ∈ s, f (m + 1) * geomSum m 1) =
          ∑ m ∈ s, ((m + 1 : ℕ) : ℝ) * f (m + 1) := by
        refine Finset.sum_congr rfl fun m _ => ?_
        simp [geomSum, mul_comm]
      rwa [heq_s] at ht
    have hle_seq : ∀ n : ℕ,
        ∑ m ∈ s, f (m + 1) * geomSum m (rseq n) ≤ (1 - R_real (rseq n)) / (1 - rseq n) := by
      intro n
      rw [hquot_tsum n]
      have hsum_n : Summable (fun m : ℕ => f (m + 1) * geomSum m (rseq n)) := by
        have hf_shift : Summable (fun m : ℕ => f (m + 1)) :=
          (summable_nat_add_iff 1).mpr hf_sum_top
        refine Summable.of_nonneg_of_le
          (fun m => mul_nonneg (firstReturnProb_nonneg K a _)
            (Finset.sum_nonneg fun j _ => pow_nonneg (hr_pos n).le j))
          (fun m => ?_) (hf_shift.mul_right (1 / (1 - rseq n)))
        rw [← hgeom_id n m]
        apply mul_le_mul_of_nonneg_left _ (firstReturnProb_nonneg K a _)
        exact div_le_div_of_nonneg_right
          (sub_le_self 1 (pow_nonneg (hr_pos n).le (m + 1))) (sub_pos.mpr (hr_lt1 n)).le
      exact hsum_n.sum_le_tsum s fun m _ =>
        mul_nonneg (firstReturnProb_nonneg K a _)
          (Finset.sum_nonneg fun j _ => pow_nonneg (hr_pos n).le j)
    exact le_of_tendsto_of_tendsto hfin_tend hquot_tend (Eventually.of_forall hle_seq)
  have hmom_sum : Summable (fun m : ℕ => ((m + 1 : ℕ) : ℝ) * f (m + 1)) :=
    summable_of_sum_le (fun m => mul_nonneg (Nat.cast_nonneg _) (firstReturnProb_nonneg K a _))
      hmom_fin_le
  have hquot_tend_tsum : Tendsto (fun n : ℕ => (1 - R_real (rseq n)) / (1 - rseq n))
      atTop (𝓝 (∑' m : ℕ, ((m + 1 : ℕ) : ℝ) * f (m + 1))) := by
    simp_rw [hquot_tsum]
    have hdom : Tendsto (fun n : ℕ => ∑' m : ℕ, f (m + 1) * geomSum m (rseq n))
        atTop (𝓝 (∑' m : ℕ, f (m + 1) * geomSum m 1)) := by
      refine tendsto_tsum_of_dominated_convergence hmom_sum (fun m => ?_)
        (Eventually.of_forall fun n m => ?_)
      · exact (tendsto_finsetSum (Finset.range (m + 1)) fun j _ => hr_tend.pow j).const_mul _
      · rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (firstReturnProb_nonneg K a _),
          abs_of_nonneg (Finset.sum_nonneg fun j _ => pow_nonneg (hr_pos n).le j)]
        have hgs_le : geomSum m (rseq n) ≤ ((m + 1 : ℕ) : ℝ) := by
          calc ∑ j ∈ Finset.range (m + 1), (rseq n) ^ j
              ≤ ∑ _j ∈ Finset.range (m + 1), (1 : ℝ) :=
                Finset.sum_le_sum fun j _ => pow_le_one₀ (hr_pos n).le (hr_lt1 n).le
            _ = ((m + 1 : ℕ) : ℝ) := by simp
        nlinarith [firstReturnProb_nonneg K a (m + 1)]
    have heq_t : (∑' m : ℕ, f (m + 1) * geomSum m 1) =
        ∑' m : ℕ, ((m + 1 : ℕ) : ℝ) * f (m + 1) :=
      tsum_congr fun m => by simp [geomSum, mul_comm]
    rwa [heq_t] at hdom
  have hmom_eq : ∑' m : ℕ, ((m + 1 : ℕ) : ℝ) * f (m + 1) = 1 / K.initial a :=
    tendsto_nhds_unique hquot_tend_tsum hquot_tend
  exact ⟨hf_hasSum, hmom_eq ▸ hmom_sum.hasSum⟩

/-- Mean first-return gap `𝒯_e = E_a[L_a] = ∑_{m ≥ 0} (m + 1) P_a(L_a = m + 1)`. -/
noncomputable def meanReturnTime (K : Chain E) (a : E) : ℝ :=
  ∑' m : ℕ, ((m + 1 : ℕ) : ℝ) * firstReturnProb K a (m + 1)

lemma meanReturnTime_eq_inv_initial (K : Chain E) (a : E) (hc : 0 < K.initial a)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ m : ℕ, |(K.prob ^ (m + 1)) a a - K.initial a| ≤ Ku * β ^ m) :
    meanReturnTime K a = 1 / K.initial a ∧ 0 < meanReturnTime K a := by
  have hkac : meanReturnTime K a = 1 / K.initial a :=
    (kac_mean_return_time K a hc Ku β hKu hβ0 hβ1 hmix).2.tsum_eq
  exact ⟨hkac, by rw [hkac]; exact one_div_pos.mpr hc⟩

/-- Actual finite-horizon lognormal path-growth second-derivative cost limit on a single edge:
`C = -(1/2) * lim_{N → ∞} (d²/dσ²) finiteGrowth(N, σ)|_{σ = 0}`. -/
noncomputable def edgeLocalCost {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (K : Chain E) (S : Ω → Ω) (a : E) (ξ : Ω → ℝ) : ℝ :=
  -(1 / 2 : ℝ) * limUnder atTop (fun N : ℕ =>
    iteratedDeriv 2
      (Dobson.FinitePathJet.finiteGrowth μ K S (singleEdgeScore a ξ) 0 .lognormal N) 0)

/-- Stationary `L²` Cauchy–Schwarz lag-covariance bound `|∫ ξ(S^[m] ω) ξ(ω) dμ| ≤ ∫ ξ² dμ`. -/
lemma stationary_autocov_abs_le {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (S : Ω ≃ᵐ Ω) (hS : MeasurePreserving S μ μ)
    (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ) (m : ℕ) :
    |∫ ω, ξ ((S : Ω → Ω)^[m] ω) * ξ ω ∂μ| ≤ ∫ ω, ξ ω * ξ ω ∂μ := by
  have hshift_m : MeasurePreserving ((S : Ω → Ω)^[m]) μ μ := hS.iterate m
  have hξm : MemLp (fun ω => ξ ((S : Ω → Ω)^[m] ω)) 2 μ := hξ.comp_measurePreserving hshift_m
  have hsq : Integrable (fun ω => ξ ω * ξ ω) μ := hξ.integrable_mul hξ
  have hsqm : Integrable (fun ω => ξ ((S : Ω → Ω)^[m] ω) * ξ ((S : Ω → Ω)^[m] ω)) μ :=
    hξm.integrable_mul hξm
  have hprod : Integrable (fun ω => ξ ((S : Ω → Ω)^[m] ω) * ξ ω) μ := hξm.integrable_mul hξ
  have heq_sq : (∫ ω, ξ ((S : Ω → Ω)^[m] ω) * ξ ((S : Ω → Ω)^[m] ω) ∂μ) =
      ∫ ω, ξ ω * ξ ω ∂μ := by
    have hshift_int : ∀ k : ℕ,
        (∫ ω, ξ ((S : Ω → Ω)^[k] ω) * ξ ((S : Ω → Ω)^[k] ω) ∂μ) = ∫ ω, ξ ω * ξ ω ∂μ := by
      intro k
      induction k with
      | zero => simp
      | succ k ihk =>
        simpa only [Function.iterate_succ_apply] using
          (hS.integral_comp' (fun ω => ξ ((S : Ω → Ω)^[k] ω) * ξ ((S : Ω → Ω)^[k] ω))).trans ihk
    exact hshift_int m
  have hupper : (∫ ω, |ξ ((S : Ω → Ω)^[m] ω) * ξ ω| ∂μ) ≤ ∫ ω, ξ ω * ξ ω ∂μ := by
    have hhalf : Integrable
        (fun ω => (1 / 2 : ℝ) * (ξ ((S : Ω → Ω)^[m] ω) * ξ ((S : Ω → Ω)^[m] ω) + ξ ω * ξ ω)) μ :=
      (hsqm.add hsq).const_mul (1 / 2)
    calc (∫ ω, |ξ ((S : Ω → Ω)^[m] ω) * ξ ω| ∂μ)
        ≤ ∫ ω, (1 / 2 : ℝ) * (ξ ((S : Ω → Ω)^[m] ω) * ξ ((S : Ω → Ω)^[m] ω) + ξ ω * ξ ω) ∂μ := by
          refine integral_mono hprod.norm hhalf fun ω => ?_
          change |ξ ((S : Ω → Ω)^[m] ω) * ξ ω| ≤ _
          rw [abs_mul]
          have h1 := sq_nonneg (|ξ ((S : Ω → Ω)^[m] ω)| - |ξ ω|)
          have h2 : |ξ ((S : Ω → Ω)^[m] ω)| ^ 2 =
              ξ ((S : Ω → Ω)^[m] ω) * ξ ((S : Ω → Ω)^[m] ω) := by rw [sq_abs, sq]
          have h3 : |ξ ω| ^ 2 = ξ ω * ξ ω := by rw [sq_abs, sq]
          linarith
      _ = (1 / 2 : ℝ) * ((∫ ω, ξ ((S : Ω → Ω)^[m] ω) * ξ ((S : Ω → Ω)^[m] ω) ∂μ) +
            ∫ ω, ξ ω * ξ ω ∂μ) := by
          rw [integral_const_mul, integral_add hsqm hsq]
      _ = ∫ ω, ξ ω * ξ ω ∂μ := by rw [heq_sq]; ring
  exact abs_integral_le_integral_abs.trans hupper

/-- `edgeLocalCost` equals the directed stationary lineage covariance coefficient. -/
theorem edgeLocalCost_eq_covarianceCoefficient {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (K : Chain E)
    (S : Ω ≃ᵐ Ω) (hS : MeasurePreserving S μ μ) (a : E) (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ e e' m, |(K.prob ^ (m + 1)) e' e - K.initial e| ≤ Ku * β ^ m) :
    Tendsto (fun N : ℕ => -(1 / 2 : ℝ) *
      iteratedDeriv 2
        (Dobson.FinitePathJet.finiteGrowth μ K S (singleEdgeScore a ξ) 0 .lognormal N) 0)
      atTop (𝓝 (Dobson.FinitePathCoefficient.covarianceCoefficient μ K S (singleEdgeScore a ξ))) ∧
    edgeLocalCost μ K S a ξ =
      Dobson.FinitePathCoefficient.covarianceCoefficient μ K S (singleEdgeScore a ξ) := by
  let B : ℝ := ∫ ω, ξ ω * ξ ω ∂μ
  have hB_nn : 0 ≤ B := integral_nonneg fun ω => mul_self_nonneg (ξ ω)
  have hcov : ∀ e e' m, |lagCovariance μ S (singleEdgeScore a ξ) e e' m| ≤ B := by
    intro e e' m
    rw [singleEdgeScore_lagCovariance]
    split_ifs
    · exact stationary_autocov_abs_le μ S hS ξ hξ m
    · simpa using hB_nn
  have htend := Dobson.FinitePathCoefficient.finiteCoefficient_tendsto μ K S hS
    (singleEdgeScore a ξ) (singleEdgeScore_memLp μ a ξ hξ) B Ku β hB_nn hKu hβ0 hβ1 hcov hmix
  have hsec : ∀ N : ℕ,
      iteratedDeriv 2
        (Dobson.FinitePathJet.finiteGrowth μ K S (singleEdgeScore a ξ) 0 .lognormal N) 0 =
      -2 * Dobson.FinitePathCoefficient.finiteCoefficient μ K S (singleEdgeScore a ξ) N :=
    fun N => Dobson.FinitePathJet.finiteGrowth_second μ K S hS
      (singleEdgeScore a ξ) (singleEdgeScore_memLp μ a ξ hξ) 0 .lognormal (by simp) N
  have htend_scaled : Tendsto (fun N : ℕ => -(1 / 2 : ℝ) *
      iteratedDeriv 2
        (Dobson.FinitePathJet.finiteGrowth μ K S (singleEdgeScore a ξ) 0 .lognormal N) 0)
      atTop (𝓝 (Dobson.FinitePathCoefficient.covarianceCoefficient μ K S (singleEdgeScore a ξ))) := by
    refine htend.congr fun N => ?_
    rw [hsec N]; ring
  have hlim_deriv : Tendsto (fun N : ℕ =>
      iteratedDeriv 2
        (Dobson.FinitePathJet.finiteGrowth μ K S (singleEdgeScore a ξ) 0 .lognormal N) 0)
      atTop (𝓝 (-2 * Dobson.FinitePathCoefficient.covarianceCoefficient μ K S (singleEdgeScore a ξ))) := by
    have h2 := htend.const_mul (-2 : ℝ)
    refine h2.congr fun N => (hsec N).symm
  refine ⟨htend_scaled, ?_⟩
  unfold edgeLocalCost
  rw [hlim_deriv.limUnder_eq]
  ring

/-- **Finite-path second-derivative cost mixture:**
For any stationary unit-variance driver `ξ` whose positive-lag autocorrelation is a finite
complex mixture `ρ(m + 1) = ∑ j ∈ J, B j * z j ^ (m + 1)` (`‖z j‖ < 1`, **without** assuming
`∑ j ∈ J, B j = 1` or mixture representation at lag `0`), the actual finite-path second-derivative
cost limit `C = edgeLocalCost μ K S a ξ` and the genuine mean renewal gap `T = meanReturnTime K a`
satisfy `(C : ℂ) = 1 / (2 * T ^ 2) - 1 / T * ∑ j ∈ J, B j * (U j - G j / T)`, together with
`U j = R j / (1 - R j)` and `G j = z j / (1 - z j)`. -/
lemma mixture_localCost_eq {ι Ω : Type*} [MeasurableSpace Ω]
    (J : Finset ι) (B z : ι → ℂ) (hz : ∀ j ∈ J, ‖z j‖ < 1)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (K : Chain E)
    (S : Ω ≃ᵐ Ω) (hS : MeasurePreserving S μ μ) (a : E) (hc : 0 < K.initial a)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ e e' m, |(K.prob ^ (m + 1)) e' e - K.initial e| ≤ Ku * β ^ m)
    (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ) (hvar : (∫ ω, ξ ω * ξ ω ∂μ) = 1)
    (hρ : ∀ m : ℕ, ((∫ ω, ξ ((S : Ω → Ω)^[m + 1] ω) * ξ ω ∂μ : ℝ) : ℂ) =
      ∑ j ∈ J, B j * z j ^ (m + 1)) :
    let T : ℝ := meanReturnTime K a
    let C : ℝ := edgeLocalCost μ K S a ξ
    let U : ι → ℂ := fun j => complexVisitSeries (fun ℓ => (K.prob ^ ℓ) a a) (z j)
    let G : ι → ℂ := fun j => complexGeomSeries (z j)
    let R : ι → ℂ := fun j => returnTransform K a (z j)
    (T : ℂ) ≠ 0 ∧
    (C : ℂ) = 1 / (2 * (T : ℂ) ^ 2) - 1 / (T : ℂ) * ∑ j ∈ J, B j * (U j - G j / (T : ℂ)) ∧
    (∀ j ∈ J, U j = R j / (1 - R j)) ∧
    (∀ j ∈ J, G j = z j / (1 - z j)) := by
  intro T C U G R
  obtain ⟨hT_eq, hT_pos⟩ := meanReturnTime_eq_inv_initial K a hc Ku β hKu hβ0 hβ1 (hmix a a)
  have hTc_ne : (T : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hT_pos.ne'
  have hinv_T : (K.initial a : ℂ) = 1 / (T : ℂ) := by
    have h1 : K.initial a = 1 / T := by dsimp [T]; rw [hT_eq, one_div_one_div]
    exact_mod_cast h1
  have hC_eq : C = Dobson.FinitePathCoefficient.covarianceCoefficient μ K S (singleEdgeScore a ξ) :=
    (edgeLocalCost_eq_covarianceCoefficient μ K S hS a ξ hξ Ku β hKu hβ0 hβ1 hmix).2
  have hj_has (j : ι) (hj : j ∈ J) :
      HasSum (fun m : ℕ => B j * z j ^ (m + 1) * (K.initial a : ℂ) *
        (((K.prob ^ (m + 1)) a a : ℂ) - K.initial a))
        ((K.initial a : ℂ) * (B j * (U j - G j * (K.initial a : ℂ)))) := by
    obtain ⟨hu_has, hu_eq⟩ := chain_return_series K a (z j) (hz j hj)
    have hg_has := complexGeomSeries_hasSum (z j) (hz j hj)
    have hg_eq := complexGeomSeries_eq (z j) (hz j hj)
    have hcomb := ((hu_has.sub (hg_has.mul_right (K.initial a : ℂ))).mul_left (B j)).mul_left
      (K.initial a : ℂ)
    dsimp [U, G]
    rw [hu_eq, hg_eq]
    convert hcomb using 1
    funext m; ring
  have hsum_has := hasSum_sum hj_has
  have hsum_eq : (∑' m : ℕ, ((∫ ω, ξ ((S : Ω → Ω)^[m + 1] ω) * ξ ω ∂μ : ℝ) : ℂ) *
      (K.initial a : ℂ) * (((K.prob ^ (m + 1)) a a : ℂ) - K.initial a)) =
      (K.initial a : ℂ) * ∑ j ∈ J, B j * (U j - G j * (K.initial a : ℂ)) := by
    rw [Finset.mul_sum, ← hsum_has.tsum_eq]
    refine tsum_congr fun m => ?_
    rw [hρ m, Finset.sum_mul, Finset.sum_mul]
  refine ⟨hTc_ne, ?_, ?_, ?_⟩
  · rw [hC_eq, singleEdgeScore_covarianceCoefficient, hvar, mul_one]
    simp only [Complex.ofReal_sub, Complex.ofReal_mul, Complex.ofReal_div, Complex.ofReal_one,
      Complex.ofReal_ofNat, Complex.ofReal_pow, Complex.ofReal_tsum]
    rw [hsum_eq, hinv_T]
    field_simp [hTc_ne]
  · intro j hj
    exact (chain_return_series K a (z j) (hz j hj)).2
  · intro j hj
    exact complexGeomSeries_eq (z j) (hz j hj)

/-- **Reference-chain mixture corollary:**
Applies `Dobson.Coverage3Renewal.mixture_coefficient` to `mixture_localCost_eq` (`part1:41`),
giving the complex renewal mixture formula for `edgeLocalCost` in terms of the genuine mean
renewal gap `T = meanReturnTime K a` and first-return PGF `returnTransform K a`. -/
theorem mixture_coefficient_unconditional {ι Ω : Type*} [MeasurableSpace Ω]
    (J : Finset ι) (B z : ι → ℂ) (hz : ∀ j ∈ J, ‖z j‖ < 1)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (K : Chain E)
    (S : Ω ≃ᵐ Ω) (hS : MeasurePreserving S μ μ) (a : E) (hc : 0 < K.initial a)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ e e' m, |(K.prob ^ (m + 1)) e' e - K.initial e| ≤ Ku * β ^ m)
    (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ) (hvar : (∫ ω, ξ ω * ξ ω ∂μ) = 1)
    (hρ : ∀ m : ℕ, ((∫ ω, ξ ((S : Ω → Ω)^[m + 1] ω) * ξ ω ∂μ : ℝ) : ℂ) =
      ∑ j ∈ J, B j * z j ^ (m + 1)) :
    let T : ℝ := meanReturnTime K a
    let C : ℝ := edgeLocalCost μ K S a ξ
    (C : ℂ) = (1 + 2 * ∑ j ∈ J, B j * z j / (1 - z j)) / (2 * (T : ℂ) ^ 2) -
      1 / (T : ℂ) * ∑ j ∈ J, B j * returnTransform K a (z j) / (1 - returnTransform K a (z j)) := by
  obtain ⟨hT_ne, hcost, hret, hgeom⟩ :=
    mixture_localCost_eq J B z hz μ K S hS a hc Ku β hKu hβ0 hβ1 hmix ξ hξ hvar hρ
  exact Dobson.Coverage3Renewal.mixture_coefficient J B z
    (fun j => returnTransform K a (z j))
    (fun j => complexVisitSeries (fun ℓ => (K.prob ^ ℓ) a a) (z j))
    (fun j => complexGeomSeries (z j))
    (meanReturnTime K a : ℂ) (edgeLocalCost μ K S a ξ : ℂ)
    hT_ne hcost hret hgeom

end ChainRenewalAndMixture

/-! ## Section 3: AR(2) Yule–Walker specialization (`prop:ring-spectrum`(a)) -/

section AR2Mixture

/-- Theoretical autocorrelation sequence `ρ(ℓ)` of a causal unit-variance AR(2) process
`X_t = 2 r_c cos ω_0 X_{t-1} - r_c^2 X_{t-2} + η_t`. -/
noncomputable def ar2Autocorr (rc ω0 : ℝ) : ℕ → ℝ
  | 0 => 1
  | 1 => 2 * rc * Real.cos ω0 / (1 + rc ^ 2)
  | n + 2 => 2 * rc * Real.cos ω0 * ar2Autocorr rc ω0 (n + 1) - rc ^ 2 * ar2Autocorr rc ω0 n

/-- Complex pole `z = r_c exp(i ω_0)` of the AR(2) autocorrelation mixture. -/
noncomputable def ar2Pole (rc ω0 : ℝ) : ℂ :=
  (rc : ℂ) * Complex.exp ((ω0 : ℂ) * Complex.I)

/-- Complex mixture weight `B = 1 + i (cos ω_0 - ρ(1) / r_c) / sin ω_0`. -/
noncomputable def ar2Weight (rc ω0 : ℝ) : ℂ :=
  1 + (((Real.cos ω0 - ar2Autocorr rc ω0 1 / rc) / Real.sin ω0 : ℝ) : ℂ) * Complex.I

/-- Any stationary unit-variance process satisfying the Yule–Walker equations of
`X_t = 2 r_c cos ω_0 X_{t-1} - r_c^2 X_{t-2} + η_t` has autocorrelation `ar2Autocorr rc ω0`. -/
theorem ar2_process_autocorrelation_eq {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (S : Ω → Ω) (X : Ω → ℝ) (rc ω0 : ℝ)
    (hvar : (∫ ω, X ω * X ω ∂μ) = 1)
    (hyw1 : (1 + rc ^ 2) * (∫ ω, X (S ω) * X ω ∂μ) = 2 * rc * Real.cos ω0)
    (hyw : ∀ n : ℕ, (∫ ω, X (S^[n + 2] ω) * X ω ∂μ) =
      2 * rc * Real.cos ω0 * (∫ ω, X (S^[n + 1] ω) * X ω ∂μ) -
        rc ^ 2 * (∫ ω, X (S^[n] ω) * X ω ∂μ)) :
    ∀ ℓ : ℕ, (∫ ω, X (S^[ℓ] ω) * X ω ∂μ) = ar2Autocorr rc ω0 ℓ := by
  intro ℓ
  induction ℓ using Nat.strong_induction_on with
  | h k ih =>
    match k with
    | 0 => simpa [ar2Autocorr] using hvar
    | 1 =>
      have hden : 1 + rc ^ 2 ≠ 0 := by positivity
      simpa [ar2Autocorr] using (eq_div_iff hden).mpr (by linarith [hyw1])
    | n + 2 =>
      rw [hyw n, ih (n + 1) (by omega), ih n (by omega)]
      rfl

/-- Complex-conjugate mixture decomposition of the defined sequence
`ar2Autocorr`, with both poles of modulus less than one. Identification with
a supplied stationary AR(2) process is handled separately. -/
theorem ar2_yule_walker_mixture (rc ω0 : ℝ) (hrc0 : 0 < rc) (hrc1 : rc < 1)
    (hω0 : 0 < ω0) (hωpi : ω0 < Real.pi) :
    ‖ar2Pole rc ω0‖ < 1 ∧ ‖star (ar2Pole rc ω0)‖ < 1 ∧
      ∀ ℓ : ℕ, (ar2Autocorr rc ω0 ℓ : ℂ) =
        (ar2Weight rc ω0 / 2) * (ar2Pole rc ω0) ^ ℓ +
          (star (ar2Weight rc ω0) / 2) * (star (ar2Pole rc ω0)) ^ ℓ := by
  have hsin_pos : 0 < Real.sin ω0 := Real.sin_pos_of_pos_of_lt_pi hω0 hωpi
  have hpole_norm : ‖ar2Pole rc ω0‖ = rc := by
    unfold ar2Pole
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hrc0,
      Complex.norm_exp_ofReal_mul_I, mul_one]
  have hpole_alg : ar2Pole rc ω0 = (rc : ℂ) * ((Real.cos ω0 : ℂ) + (Real.sin ω0 : ℂ) * Complex.I) := by
    unfold ar2Pole
    rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
  have htrig_c : (Real.cos ω0 : ℂ) ^ 2 + (Real.sin ω0 : ℂ) ^ 2 = 1 := by
    exact_mod_cast Real.cos_sq_add_sin_sq ω0
  have hpole_quad : (ar2Pole rc ω0) ^ 2 =
      (2 * rc * Real.cos ω0 : ℂ) * ar2Pole rc ω0 - (rc : ℂ) ^ 2 := by
    rw [hpole_alg]
    linear_combination -(rc : ℂ) ^ 2 * htrig_c +
      (rc : ℂ) ^ 2 * ((Real.sin ω0 : ℂ) ^ 2 * Complex.I_sq)
  have hstar_alg : star (ar2Pole rc ω0) =
      (rc : ℂ) * ((Real.cos ω0 : ℂ) - (Real.sin ω0 : ℂ) * Complex.I) := by
    rw [hpole_alg]
    apply Complex.ext <;>
      simp only [Complex.star_def, Complex.conj_re, Complex.conj_im, Complex.mul_re,
        Complex.mul_im, Complex.add_re, Complex.add_im, Complex.sub_re, Complex.sub_im,
        Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im] <;>
      ring
  have hstar_quad : (star (ar2Pole rc ω0)) ^ 2 =
      (2 * rc * Real.cos ω0 : ℂ) * star (ar2Pole rc ω0) - (rc : ℂ) ^ 2 := by
    rw [hstar_alg]
    linear_combination -(rc : ℂ) ^ 2 * htrig_c +
      (rc : ℂ) ^ 2 * ((Real.sin ω0 : ℂ) ^ 2 * Complex.I_sq)
  refine ⟨hpole_norm.trans_lt hrc1, (norm_star _).trans hpole_norm |>.trans_lt hrc1, ?_⟩
  let q : ℝ := (Real.cos ω0 - ar2Autocorr rc ω0 1 / rc) / Real.sin ω0
  have hw_eq : ar2Weight rc ω0 = 1 + (q : ℂ) * Complex.I := rfl
  intro ℓ
  induction ℓ using Nat.strong_induction_on with
  | h k ih =>
    match k with
    | 0 =>
      simp only [ar2Autocorr, pow_zero, mul_one, Complex.ofReal_one, hw_eq]
      apply Complex.ext
      · simp only [Complex.one_re, Complex.add_re, Complex.div_ofNat, Complex.mul_re,
          Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im,
          Complex.star_def, Complex.conj_re, Complex.conj_im, Complex.add_im, Complex.mul_im,
          Complex.one_im]
        ring
      · simp only [Complex.one_im, Complex.add_im, Complex.div_ofNat, Complex.mul_im,
          Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im,
          Complex.star_def, Complex.conj_re, Complex.conj_im, Complex.add_re, Complex.one_re]
        ring
    | 1 =>
      simp only [pow_one, hw_eq]
      rw [hstar_alg, hpole_alg]
      apply Complex.ext
      · simp only [Complex.ofReal_re, Complex.ofReal_im, Complex.add_re, Complex.add_im,
          Complex.sub_re, Complex.sub_im, Complex.div_ofNat, Complex.mul_re, Complex.mul_im,
          Complex.one_re, Complex.one_im, Complex.I_re, Complex.I_im,
          Complex.star_def, Complex.conj_re, Complex.conj_im]
        dsimp [q]
        field_simp [hrc0.ne', hsin_pos.ne']
        ring
      · simp only [Complex.ofReal_re, Complex.ofReal_im, Complex.add_re, Complex.add_im,
          Complex.sub_re, Complex.sub_im, Complex.div_ofNat, Complex.mul_re, Complex.mul_im,
          Complex.one_re, Complex.one_im, Complex.I_re, Complex.I_im,
          Complex.star_def, Complex.conj_re, Complex.conj_im]
        ring
    | n + 2 =>
      simp only [ar2Autocorr, Complex.ofReal_sub, Complex.ofReal_mul, Complex.ofReal_pow,
        Complex.ofReal_ofNat]
      rw [ih (n + 1) (by omega), ih n (by omega)]
      have hz_step : (ar2Pole rc ω0) ^ (n + 2) =
          2 * (rc : ℂ) * (Real.cos ω0 : ℂ) * (ar2Pole rc ω0) ^ (n + 1) -
            (rc : ℂ) ^ 2 * (ar2Pole rc ω0) ^ n := by
        calc (ar2Pole rc ω0) ^ (n + 2) = (ar2Pole rc ω0) ^ n * (ar2Pole rc ω0) ^ 2 := by ring
          _ = (ar2Pole rc ω0) ^ n * ((2 * rc * Real.cos ω0 : ℂ) * ar2Pole rc ω0 - (rc : ℂ) ^ 2) := by
              rw [hpole_quad]
          _ = 2 * (rc : ℂ) * (Real.cos ω0 : ℂ) * (ar2Pole rc ω0) ^ (n + 1) -
                (rc : ℂ) ^ 2 * (ar2Pole rc ω0) ^ n := by ring
      have hsz_step : (star (ar2Pole rc ω0)) ^ (n + 2) =
          2 * (rc : ℂ) * (Real.cos ω0 : ℂ) * (star (ar2Pole rc ω0)) ^ (n + 1) -
            (rc : ℂ) ^ 2 * (star (ar2Pole rc ω0)) ^ n := by
        calc (star (ar2Pole rc ω0)) ^ (n + 2)
            = (star (ar2Pole rc ω0)) ^ n * (star (ar2Pole rc ω0)) ^ 2 := by ring
          _ = (star (ar2Pole rc ω0)) ^ n *
                ((2 * rc * Real.cos ω0 : ℂ) * star (ar2Pole rc ω0) - (rc : ℂ) ^ 2) := by
              rw [hstar_quad]
          _ = 2 * (rc : ℂ) * (Real.cos ω0 : ℂ) * (star (ar2Pole rc ω0)) ^ (n + 1) -
                (rc : ℂ) ^ 2 * (star (ar2Pole rc ω0)) ^ n := by ring
      rw [hz_step, hsz_step]
      ring

end AR2Mixture

/-! ## Section 4: Spectral integral representation and kernel properties (`prop:ring-spectrum`(b,c)) -/

section SpectralKernel

/-- Demographic frequency-response kernel `K(ω) = -(1 / T) Re[g(ω) / (1 - g(ω))]`. -/
noncomputable def spectralKernel (T : ℝ) (g : ℝ → ℂ) (ω : ℝ) : ℝ :=
  -(1 / T) * (g ω / (1 - g ω)).re

/-- First-return characteristic function `g(ω) = E_a[e^{i ω L_a}]` on `Chain E`. -/
noncomputable def returnCharFun {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : E) (ω : ℝ) : ℂ :=
  ∑' k : ℕ, (firstReturnProb K a k : ℂ) * Complex.exp ((((k : ℕ) : ℝ) * ω : ℂ) * Complex.I)

/-- Second moment `E_a[L_a^2]` of the first-return gap on `Chain E`. -/
noncomputable def returnSecondMoment {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : E) : ℝ :=
  ∑' m : ℕ, (((m + 1 : ℕ) : ℝ) ^ 2) * firstReturnProb K a (m + 1)

/-- Squared coefficient of variation `CV² = Var(L_a) / E[L_a]²` of the first-return gap. -/
noncomputable def returnCV2 {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : E) : ℝ :=
  let T := meanReturnTime K a
  (returnSecondMoment K a - T ^ 2) / T ^ 2

lemma unitCircle_div_one_sub_re (ζ : ℂ) (hnorm : Complex.normSq ζ = 1) (hne : ζ ≠ 1) :
    (ζ / (1 - ζ)).re = -1 / 2 := by
  rw [Dobson.Coverage3Renewal.spectral_real_part, hnorm]
  have h1ne : Complex.normSq (1 - ζ) ≠ 0 :=
    (Complex.normSq_pos.mpr (sub_ne_zero.mpr hne.symm)).ne'
  have hden : Complex.normSq (1 - ζ) = 2 * (1 - ζ.re) := by
    have hn : ζ.re * ζ.re + ζ.im * ζ.im = 1 := by simpa [Complex.normSq_apply] using hnorm
    simp only [Complex.normSq_apply, Complex.sub_re, Complex.sub_im, Complex.one_re, Complex.one_im]
    linarith
  have hre_ne : 1 - ζ.re ≠ 0 := by
    intro h0; apply h1ne; rw [hden, h0, mul_zero]
  rw [hden]
  field_simp [hre_ne]
  ring

lemma exp_ofReal_mul_I_ne_one_of_Ioo {ω : ℝ} (h0 : 0 < ω) (h2pi : ω < 2 * Real.pi) :
    Complex.exp ((ω : ℂ) * Complex.I) ≠ 1 := by
  rcases lt_trichotomy ω Real.pi with hlt | rfl | hgt
  · intro heq
    have him := congrArg Complex.im heq
    rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin] at him
    simp only [Complex.add_im, Complex.ofReal_im, Complex.mul_im, Complex.I_re, Complex.I_im,
      Complex.ofReal_re, mul_one, mul_zero, add_zero, zero_add, Complex.one_im] at him
    exact (Real.sin_pos_of_pos_of_lt_pi h0 hlt).ne' him
  · intro heq
    have hre := congrArg Complex.re heq
    rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_pi, Real.sin_pi] at hre
    norm_num at hre
  · intro heq
    have him := congrArg Complex.im heq
    rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin] at him
    simp only [Complex.add_im, Complex.ofReal_im, Complex.mul_im, Complex.I_re, Complex.I_im,
      Complex.ofReal_re, mul_one, mul_zero, add_zero, zero_add, Complex.one_im] at him
    have hsub0 : 0 < ω - Real.pi := by linarith
    have hsubpi : ω - Real.pi < Real.pi := by linarith
    have hsin : Real.sin ω = -Real.sin (ω - Real.pi) := by
      conv_lhs => rw [show ω = (ω - Real.pi) + Real.pi by ring]
      exact Real.sin_add_pi (ω - Real.pi)
    rw [hsin] at him
    linarith [Real.sin_pos_of_pos_of_lt_pi hsub0 hsubpi]

private lemma intervalIntegral_finset_sum {ι : Type*} (s : Finset ι) (f : ι → ℝ → ℝ)
    (a b : ℝ) (hf : ∀ i ∈ s, IntervalIntegrable (f i) volume a b) :
    (∫ x in a..b, ∑ i ∈ s, f i x) = ∑ i ∈ s, ∫ x in a..b, f i x := by
  classical
  have hsum_int : ∀ t : Finset ι, (∀ i ∈ t, IntervalIntegrable (f i) volume a b) →
      IntervalIntegrable (fun x => ∑ i ∈ t, f i x) volume a b := by
    intro t ht
    induction t using Finset.induction_on with
    | empty => simp
    | @insert k u hk ihu =>
      have hfn : (fun x => ∑ i ∈ insert k u, f i x) = fun x => f k x + ∑ i ∈ u, f i x := by
        funext x; exact Finset.sum_insert hk
      rw [hfn]
      exact (ht k (Finset.mem_insert_self k u)).add
        (ihu fun i hi => ht i (Finset.mem_insert_of_mem hi))
  induction s using Finset.induction_on with
  | empty => simp
  | @insert j t hj ih =>
    have hfn : (fun x => ∑ i ∈ insert j t, f i x) = fun x => f j x + ∑ i ∈ t, f i x := by
      funext x; exact Finset.sum_insert hj
    rw [hfn, intervalIntegral.integral_add (hf j (Finset.mem_insert_self j t))
      (hsum_int t fun i hi => hf i (Finset.mem_insert_of_mem hi)),
      Finset.sum_insert hj, ih (fun i hi => hf i (Finset.mem_insert_of_mem hi))]

/-- Radial Abel limit (`r ↑ 1`) proving `returnCharFun K a ω ≠ 1` and the exact cosine-series
identity `∑_{m ≥ 0} (u(m+1) - c) cos((m+1) ω) = Re[g(ω) / (1 - g(ω))] + c / 2` whenever
`e^{i ω} ≠ 1` (in particular for all `ω ∈ (0, 2π)`). -/
theorem spectral_cosine_series {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : E) (hc : 0 < K.initial a)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ m : ℕ, |(K.prob ^ (m + 1)) a a - K.initial a| ≤ Ku * β ^ m)
    (ω : ℝ) (hζ : Complex.exp ((ω : ℂ) * Complex.I) ≠ 1) :
    let g := returnCharFun K a
    let T := meanReturnTime K a
    g ω ≠ 1 ∧
      HasSum (fun m : ℕ =>
        ((K.prob ^ (m + 1)) a a - K.initial a) * Real.cos (((m + 1 : ℕ) : ℝ) * ω))
        ((g ω / (1 - g ω)).re + K.initial a / 2) ∧
      spectralKernel T g ω =
        1 / (2 * T ^ 2) - (1 / T) * ∑' m : ℕ,
          ((K.prob ^ (m + 1)) a a - K.initial a) * Real.cos (((m + 1 : ℕ) : ℝ) * ω) := by
  intro g T
  let ζ : ℂ := Complex.exp ((ω : ℂ) * Complex.I)
  have hζ_norm : ‖ζ‖ = 1 := Complex.norm_exp_ofReal_mul_I ω
  have hζ_nsq : Complex.normSq ζ = 1 := by
    rw [Complex.normSq_eq_norm_sq, hζ_norm, one_pow]
  have hf_sum : Summable (firstReturnProb K a) :=
    (kac_mean_return_time K a hc Ku β hKu hβ0 hβ1 hmix).1.summable
  let δ : ℕ → ℝ := fun m => (K.prob ^ (m + 1)) a a - K.initial a
  have hδ_abs : Summable (fun m : ℕ => |δ m|) :=
    Summable.of_nonneg_of_le (fun _ => abs_nonneg _) hmix
      ((summable_geometric_of_lt_one hβ0 hβ1).mul_left Ku)
  let rseq : ℕ → ℝ := fun n => 1 - 1 / ((n : ℝ) + 2)
  have hr_pos : ∀ n : ℕ, 0 < rseq n := by
    intro n; dsimp [rseq]
    have hgt : (1 : ℝ) < (n : ℝ) + 2 := by linarith [Nat.cast_nonneg (α := ℝ) n]
    linarith [div_lt_one (by positivity : (0 : ℝ) < (n : ℝ) + 2) |>.mpr hgt]
  have hr_lt1 : ∀ n : ℕ, rseq n < 1 := by
    intro n; dsimp [rseq]
    have hp : 0 < 1 / ((n : ℝ) + 2) := by positivity
    linarith
  have hr_tend : Tendsto rseq atTop (𝓝 1) := by
    have hinv : Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 2)) atTop (𝓝 0) :=
      tendsto_const_nhds.div_atTop (tendsto_natCast_atTop_atTop.atTop_add tendsto_const_nhds)
    simpa [rseq] using (tendsto_const_nhds (x := (1 : ℝ))).sub hinv
  have hrc_tend : Tendsto (fun n : ℕ => (rseq n : ℂ)) atTop (𝓝 (1 : ℂ)) :=
    Complex.continuous_ofReal.continuousAt.tendsto.comp hr_tend
  let zseq : ℕ → ℂ := fun n => (rseq n : ℂ) * ζ
  have hz_norm : ∀ n : ℕ, ‖zseq n‖ < 1 := by
    intro n; dsimp [zseq]
    rw [norm_mul, hζ_norm, mul_one, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (hr_pos n)]
    exact hr_lt1 n
  have hz_tend : Tendsto zseq atTop (𝓝 ζ) := by
    simpa [zseq] using hrc_tend.mul_const ζ
  have hpow_ζ : ∀ k : ℕ, ζ ^ k = Complex.exp ((((k : ℕ) : ℝ) * ω : ℂ) * Complex.I) := by
    intro k
    dsimp [ζ]
    rw [← Complex.exp_nat_mul]
    congr 1; push_cast; ring
  have hR_tend : Tendsto (fun n : ℕ => returnTransform K a (zseq n)) atTop (𝓝 (g ω)) := by
    have hfn : (fun n : ℕ => returnTransform K a (zseq n)) =
        fun n : ℕ => ∑' k : ℕ, (firstReturnProb K a k : ℂ) * (zseq n) ^ k := by
      funext n; exact returnTransform_eq_tsum K a (zseq n) (hz_norm n)
    rw [hfn]
    have hlim : Tendsto (fun n : ℕ => ∑' k : ℕ, (firstReturnProb K a k : ℂ) * (zseq n) ^ k)
        atTop (𝓝 (∑' k : ℕ, (firstReturnProb K a k : ℂ) * ζ ^ k)) := by
      refine tendsto_tsum_of_dominated_convergence hf_sum (fun k => ?_)
        (Eventually.of_forall fun n k => ?_)
      · exact (hz_tend.pow k).const_mul _
      · rw [norm_mul, Complex.norm_real, Real.norm_eq_abs,
          abs_of_nonneg (firstReturnProb_nonneg K a k), norm_pow]
        exact mul_le_of_le_one_right (firstReturnProb_nonneg K a k)
          (pow_le_one₀ (norm_nonneg _) (hz_norm n).le)
    refine hlim.trans_eq (congrArg 𝓝 ?_)
    dsimp [g, returnCharFun]
    refine tsum_congr fun k => ?_
    rw [hpow_ζ k]
    push_cast
    rfl
  let D_lim : ℂ := ∑' m : ℕ, (δ m : ℂ) * ζ ^ (m + 1)
  have hD_tend : Tendsto (fun n : ℕ => ∑' m : ℕ, (zseq n) ^ (m + 1) * (δ m : ℂ))
      atTop (𝓝 D_lim) := by
    refine tendsto_tsum_of_dominated_convergence hδ_abs (fun m => ?_)
      (Eventually.of_forall fun n m => ?_)
    · simpa [mul_comm] using (hz_tend.pow (m + 1)).mul_const (δ m : ℂ)
    · rw [norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs]
      exact mul_le_of_le_one_left (abs_nonneg _) (pow_le_one₀ (norm_nonneg _) (hz_norm n).le)
  have hprod_one : ∀ n : ℕ,
      (1 - returnTransform K a (zseq n)) *
        (1 + (∑' m : ℕ, (zseq n) ^ (m + 1) * (δ m : ℂ)) +
          (K.initial a : ℂ) * (zseq n / (1 - zseq n))) = 1 := by
    intro n
    have hser := (chain_return_series K a (zseq n) (hz_norm n)).1
    have hgeom := (complexGeomSeries_hasSum (zseq n) (hz_norm n)).mul_right (K.initial a : ℂ)
    have hδ_has : HasSum (fun m : ℕ => (zseq n) ^ (m + 1) * (δ m : ℂ))
        (∑' m : ℕ, (zseq n) ^ (m + 1) * (δ m : ℂ)) := by
      refine (Summable.of_norm_bounded hδ_abs fun m => ?_).hasSum
      rw [norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs]
      exact mul_le_of_le_one_left (abs_nonneg _) (pow_le_one₀ (norm_nonneg _) (hz_norm n).le)
    have hcomb := hδ_has.add hgeom
    have hfn : (fun m : ℕ => (zseq n) ^ (m + 1) * (δ m : ℂ) +
        (zseq n) ^ (m + 1) * (K.initial a : ℂ)) =
        fun m : ℕ => (zseq n) ^ (m + 1) * (((K.prob ^ (m + 1)) a a : ℂ)) := by
      funext m; dsimp [δ]; push_cast; ring
    rw [hfn] at hcomb
    have heq := hcomb.unique hser
    have h1R_ne : 1 - returnTransform K a (zseq n) ≠ 0 := by
      intro h0
      have hR1 : returnTransform K a (zseq n) = 1 := by linear_combination -h0
      have hnormR : ‖returnTransform K a (zseq n)‖ ≤ ‖zseq n‖ := by
        rw [returnTransform_eq_tsum K a (zseq n) (hz_norm n)]
        have hsn : Summable (fun k : ℕ => ‖(firstReturnProb K a k : ℂ) * (zseq n) ^ k‖) := by
          refine Summable.of_nonneg_of_le (fun _ => norm_nonneg _) (fun k => ?_) hf_sum
          rw [norm_mul, Complex.norm_real, Real.norm_eq_abs,
            abs_of_nonneg (firstReturnProb_nonneg K a k), norm_pow]
          exact mul_le_of_le_one_right (firstReturnProb_nonneg K a k)
            (pow_le_one₀ (norm_nonneg _) (hz_norm n).le)
        calc ‖∑' k : ℕ, (firstReturnProb K a k : ℂ) * (zseq n) ^ k‖
            ≤ ∑' k : ℕ, ‖(firstReturnProb K a k : ℂ) * (zseq n) ^ k‖ := norm_tsum_le_tsum_norm hsn
          _ ≤ ∑' k : ℕ, firstReturnProb K a k * ‖zseq n‖ := by
              refine hsn.tsum_le_tsum (fun k => ?_) (hf_sum.mul_right ‖zseq n‖)
              cases k with
              | zero => simp [firstReturnProb]
              | succ m =>
                rw [norm_mul, Complex.norm_real, Real.norm_eq_abs,
                  abs_of_nonneg (firstReturnProb_nonneg K a _), norm_pow, pow_succ]
                have hp : ‖zseq n‖ ^ m ≤ 1 := pow_le_one₀ (norm_nonneg _) (hz_norm n).le
                calc firstReturnProb K a (m + 1) * (‖zseq n‖ ^ m * ‖zseq n‖)
                    ≤ firstReturnProb K a (m + 1) * (1 * ‖zseq n‖) :=
                      mul_le_mul_of_nonneg_left
                        (mul_le_mul_of_nonneg_right hp (norm_nonneg _))
                        (firstReturnProb_nonneg K a _)
                  _ = firstReturnProb K a (m + 1) * ‖zseq n‖ := by ring
          _ = (∑' k : ℕ, firstReturnProb K a k) * ‖zseq n‖ := tsum_mul_right
          _ = ‖zseq n‖ := by
              rw [(kac_mean_return_time K a hc Ku β hKu hβ0 hβ1 hmix).1.tsum_eq, one_mul]
      rw [hR1, norm_one] at hnormR
      exact not_lt_of_ge hnormR (hz_norm n)
    have hrew : 1 + ((∑' m : ℕ, (zseq n) ^ (m + 1) * (δ m : ℂ)) +
        (zseq n / (1 - zseq n)) * (K.initial a : ℂ)) =
        1 / (1 - returnTransform K a (zseq n)) := by
      rw [heq]; field_simp [h1R_ne]; ring
    calc (1 - returnTransform K a (zseq n)) *
          (1 + (∑' m : ℕ, (zseq n) ^ (m + 1) * (δ m : ℂ)) +
            (K.initial a : ℂ) * (zseq n / (1 - zseq n)))
        = (1 - returnTransform K a (zseq n)) *
            (1 + ((∑' m : ℕ, (zseq n) ^ (m + 1) * (δ m : ℂ)) +
              (zseq n / (1 - zseq n)) * (K.initial a : ℂ))) := by ring
      _ = (1 - returnTransform K a (zseq n)) * (1 / (1 - returnTransform K a (zseq n))) := by
          rw [hrew]
      _ = 1 := by field_simp [h1R_ne]
  have h1ζ_ne : 1 - ζ ≠ 0 := sub_ne_zero.mpr hζ.symm
  have hlim_prod : Tendsto (fun n : ℕ =>
      (1 - returnTransform K a (zseq n)) *
        (1 + (∑' m : ℕ, (zseq n) ^ (m + 1) * (δ m : ℂ)) +
          (K.initial a : ℂ) * (zseq n / (1 - zseq n))))
      atTop (𝓝 ((1 - g ω) * (1 + D_lim + (K.initial a : ℂ) * (ζ / (1 - ζ))))) := by
    exact (tendsto_const_nhds.sub hR_tend).mul
      ((tendsto_const_nhds.add hD_tend).add
        ((hz_tend.div (tendsto_const_nhds.sub hz_tend) h1ζ_ne).const_mul _))
  have hlim_eq : (1 - g ω) * (1 + D_lim + (K.initial a : ℂ) * (ζ / (1 - ζ))) = 1 :=
    tendsto_nhds_unique hlim_prod (by simp [hprod_one])
  have h1g_ne : 1 - g ω ≠ 0 := fun h0 => by rw [h0, zero_mul] at hlim_eq; exact zero_ne_one hlim_eq
  have hg_ne_one : g ω ≠ 1 := fun h1 => h1g_ne (sub_eq_zero.mpr h1.symm)
  have hD_eq : D_lim = g ω / (1 - g ω) - (K.initial a : ℂ) * (ζ / (1 - ζ)) := by
    have hdiv : 1 + D_lim + (K.initial a : ℂ) * (ζ / (1 - ζ)) = 1 / (1 - g ω) :=
      (eq_div_iff h1g_ne).mpr (by linear_combination hlim_eq)
    have h1 : 1 / (1 - g ω) - 1 = g ω / (1 - g ω) := by field_simp [h1g_ne]; ring
    linear_combination hdiv + h1
  have hD_has : HasSum (fun m : ℕ => (δ m : ℂ) * ζ ^ (m + 1)) D_lim := by
    refine (Summable.of_norm_bounded hδ_abs fun m => ?_).hasSum
    rw [norm_mul, norm_pow, hζ_norm, one_pow, mul_one, Complex.norm_real, Real.norm_eq_abs]
  have hre_has : HasSum (fun m : ℕ => δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) D_lim.re := by
    have hre := Complex.reCLM.hasSum hD_has
    refine hre.congr_fun fun m => ?_
    change δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω) = ((δ m : ℂ) * ζ ^ (m + 1)).re
    rw [hpow_ζ (m + 1), ← Complex.ofReal_mul, Complex.exp_mul_I, ← Complex.ofReal_cos,
      ← Complex.ofReal_sin]
    simp only [Complex.mul_re, Complex.add_re, Complex.ofReal_re, Complex.ofReal_im,
      Complex.I_re, Complex.I_im, mul_one, mul_zero, add_zero, sub_zero]
    ring
  have hDre : D_lim.re = (g ω / (1 - g ω)).re + K.initial a / 2 := by
    rw [hD_eq, Complex.sub_re, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
      zero_mul, sub_zero, unitCircle_div_one_sub_re ζ hζ_nsq hζ]
    ring
  rw [hDre] at hre_has
  obtain ⟨hT_eq, hT_pos⟩ := meanReturnTime_eq_inv_initial K a hc Ku β hKu hβ0 hβ1 hmix
  have hT_ne : T ≠ 0 := hT_pos.ne'
  have hc_eq : K.initial a = 1 / T := by dsimp [T]; rw [hT_eq, one_div_one_div]
  refine ⟨hg_ne_one, hre_has, ?_⟩
  unfold spectralKernel
  rw [hre_has.tsum_eq, hc_eq]
  field_simp [hT_ne]
  ring

/-- **Spectral integral representation (`prop:ring-spectrum`(b), `eq:ring-spectrum`):**
For any stationary unit-variance driver `ξ` with continuous spectral density `f` on `[0, π]`
satisfying `(1 / π) ∫_0^π f(ω) dω = 1` and `ρ(m + 1) = (1 / π) ∫_0^π f(ω) cos((m + 1) ω) dω`,
the actual finite-path cost limit `edgeLocalCost μ K S a ξ` equals
`(1 / π) ∫_0^π f(ω) K(ω) dω`. -/
theorem spectral_integral_representation {E Ω : Type*} [Fintype E] [DecidableEq E]
    [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (S : Ω ≃ᵐ Ω) (hS : MeasurePreserving S μ μ)
    (a : E) (hc : 0 < K.initial a)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ e e' m, |(K.prob ^ (m + 1)) e' e - K.initial e| ≤ Ku * β ^ m)
    (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ) (hvar : (∫ ω, ξ ω * ξ ω ∂μ) = 1)
    (f : ℝ → ℝ) (hf_cont : ContinuousOn f (Icc 0 Real.pi))
    (hf_norm : (1 / Real.pi) * (∫ ω in (0 : ℝ)..Real.pi, f ω) = 1)
    (hρ : ∀ m : ℕ, (∫ x, ξ ((S : Ω → Ω)^[m + 1] x) * ξ x ∂μ) =
      (1 / Real.pi) * ∫ ω in (0 : ℝ)..Real.pi, f ω * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) :
    let T := meanReturnTime K a
    let g := returnCharFun K a
    edgeLocalCost μ K S a ξ =
      (1 / Real.pi) * ∫ ω in (0 : ℝ)..Real.pi, f ω * spectralKernel T g ω := by
  intro T g
  obtain ⟨hT_eq, hT_pos⟩ := meanReturnTime_eq_inv_initial K a hc Ku β hKu hβ0 hβ1 (hmix a a)
  have hc_eq : K.initial a = 1 / T := by dsimp [T]; rw [hT_eq, one_div_one_div]
  let δ : ℕ → ℝ := fun m => (K.prob ^ (m + 1)) a a - K.initial a
  let K_ser : ℝ → ℝ := fun ω =>
    1 / (2 * T ^ 2) - (1 / T) * ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)
  -- `K_ser` agrees with `spectralKernel T g` on `Ioo 0 π`, hence a.e. on `uIoc 0 π`!
  have hae_eq : ∀ᵐ ω ∂volume, ω ∈ Set.uIoc (0 : ℝ) Real.pi →
      f ω * spectralKernel T g ω = f ω * K_ser ω := by
    have h1 : ({0, Real.pi} : Set ℝ)ᶜ ∈ ae volume := by
      rw [compl_mem_ae_iff]
      exact (Set.Finite.insert 0 (Set.finite_singleton Real.pi)).measure_zero volume
    filter_upwards [h1] with ω hω hmem
    rw [Set.uIoc_of_le Real.pi_pos.le] at hmem
    have h0 : 0 < ω := lt_of_le_of_ne hmem.1.le (fun h => hω (by left; exact h.symm))
    have hpi : ω < Real.pi := lt_of_le_of_ne hmem.2 (fun h => hω (by right; exact h))
    have h2pi : ω < 2 * Real.pi := by linarith [Real.pi_pos]
    have hspec := (spectral_cosine_series K a hc Ku β hKu hβ0 hβ1 (hmix a a) ω
      (exp_ofReal_mul_I_ne_one_of_Ioo h0 h2pi)).2.2
    dsimp [T, g, K_ser, δ] at hspec ⊢
    rw [hspec]
  rw [intervalIntegral.integral_congr_ae hae_eq]
  -- Swap `∫_0^π f(ω) * (∑' m, δ m * cos((m+1)ω))` with `∑' m` by dominated convergence on `[0, π]`:
  obtain ⟨Mf, hMf⟩ := isCompact_Icc.exists_bound_of_continuousOn hf_cont
  have hMf_nn : 0 ≤ Mf :=
    (norm_nonneg (f 0)).trans (hMf 0 ⟨le_rfl, Real.pi_pos.le⟩)
  have hδ_abs : Summable (fun m : ℕ => |δ m|) :=
    Summable.of_nonneg_of_le (fun _ => abs_nonneg _) (hmix a a)
      ((summable_geometric_of_lt_one hβ0 hβ1).mul_left Ku)
  have hcos_sum : ∀ ω : ℝ, Summable (fun m : ℕ => δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) := by
    intro ω
    refine Summable.of_norm_bounded hδ_abs fun m => ?_
    rw [Real.norm_eq_abs, abs_mul]
    exact mul_le_of_le_one_right (abs_nonneg _) (Real.abs_cos_le_one _)
  have hf_int : IntervalIntegrable f volume 0 Real.pi :=
    hf_cont.intervalIntegrable_of_Icc Real.pi_pos.le
  have hterm_int : ∀ m : ℕ,
      IntervalIntegrable (fun ω => f ω * (δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))) volume 0 Real.pi := by
    intro m
    refine ContinuousOn.intervalIntegrable_of_Icc Real.pi_pos.le ?_
    exact hf_cont.mul (continuousOn_const.mul
      (Real.continuous_cos.comp (continuous_const.mul continuous_id)).continuousOn)
  let bound_fn : ℝ := Mf * ∑' m : ℕ, |δ m|
  have hdom_tend : Tendsto
      (fun N : ℕ => ∫ ω in (0 : ℝ)..Real.pi,
        ∑ m ∈ Finset.range N, f ω * (δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)))
      atTop
      (𝓝 (∫ ω in (0 : ℝ)..Real.pi,
        f ω * ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))) := by
    apply intervalIntegral.tendsto_integral_filter_of_dominated_convergence (fun _ => bound_fn)
    · refine Eventually.of_forall fun N => ?_
      rw [Set.uIoc_of_le Real.pi_pos.le]
      convert Finset.aestronglyMeasurable_sum (Finset.range N)
        (fun m _ => (hterm_int m).1.aestronglyMeasurable) using 1
      funext ω; simp [Finset.sum_apply]
    · refine Eventually.of_forall fun N => Eventually.of_forall fun ω hω => ?_
      rw [Set.uIoc_of_le Real.pi_pos.le] at hω
      have hω_Icc : ω ∈ Icc 0 Real.pi := ⟨hω.1.le, hω.2⟩
      have hf_le : |f ω| ≤ Mf := by simpa [Real.norm_eq_abs] using hMf ω hω_Icc
      rw [Real.norm_eq_abs]
      calc |∑ m ∈ Finset.range N, f ω * (δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))|
          ≤ ∑ m ∈ Finset.range N, |f ω * (δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))| :=
            Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ m ∈ Finset.range N, Mf * |δ m| := by
            refine Finset.sum_le_sum fun m _ => ?_
            rw [abs_mul, abs_mul]
            exact mul_le_mul hf_le
              (mul_le_of_le_one_right (abs_nonneg _) (Real.abs_cos_le_one _))
              (by positivity) hMf_nn
        _ = Mf * ∑ m ∈ Finset.range N, |δ m| := (Finset.mul_sum _ _ _).symm
        _ ≤ Mf * ∑' m : ℕ, |δ m| :=
            mul_le_mul_of_nonneg_left
              (hδ_abs.sum_le_tsum (Finset.range N) fun m _ => abs_nonneg _) hMf_nn
    · exact intervalIntegrable_const
    · refine Eventually.of_forall fun ω _ => ?_
      simpa only [Finset.mul_sum] using
        ((hcos_sum ω).hasSum.tendsto_sum_nat).const_mul (f ω)
  have hpart_eq : ∀ N : ℕ,
      (∫ ω in (0 : ℝ)..Real.pi,
        ∑ m ∈ Finset.range N, f ω * (δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))) =
      Real.pi * ∑ m ∈ Finset.range N,
        (∫ x, ξ ((S : Ω → Ω)^[m + 1] x) * ξ x ∂μ) * δ m := by
    intro N
    rw [intervalIntegral_finset_sum (Finset.range N) _ 0 Real.pi (fun m _ => hterm_int m),
      Finset.mul_sum]
    refine Finset.sum_congr rfl fun m _ => ?_
    have hm_eq : (fun ω => f ω * (δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))) =
        fun ω => δ m * (f ω * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) := by funext ω; ring
    simp_rw [hm_eq]
    rw [intervalIntegral.integral_const_mul, hρ m]
    field_simp [Real.pi_pos.ne']
  have hlag_sum : Summable (fun m : ℕ => (∫ x, ξ ((S : Ω → Ω)^[m + 1] x) * ξ x ∂μ) * δ m) := by
    refine Summable.of_norm_bounded hδ_abs fun m => ?_
    rw [Real.norm_eq_abs, abs_mul]
    have hcov_le : |∫ x, ξ ((S : Ω → Ω)^[m + 1] x) * ξ x ∂μ| ≤ 1 := by
      have hb := stationary_autocov_abs_le μ S hS ξ hξ (m + 1)
      rwa [hvar] at hb
    exact mul_le_of_le_one_left (abs_nonneg _) hcov_le
  have hlim_rhs : Tendsto
      (fun N : ℕ => Real.pi * ∑ m ∈ Finset.range N,
        (∫ x, ξ ((S : Ω → Ω)^[m + 1] x) * ξ x ∂μ) * δ m)
      atTop
      (𝓝 (Real.pi * ∑' m : ℕ, (∫ x, ξ ((S : Ω → Ω)^[m + 1] x) * ξ x ∂μ) * δ m)) :=
    hlag_sum.hasSum.tendsto_sum_nat.const_mul Real.pi
  simp_rw [hpart_eq] at hdom_tend
  have hswap_int :
      (∫ ω in (0 : ℝ)..Real.pi, f ω * ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) =
      Real.pi * ∑' m : ℕ, (∫ x, ξ ((S : Ω → Ω)^[m + 1] x) * ξ x ∂μ) * δ m :=
    tendsto_nhds_unique hdom_tend hlim_rhs
  have hcont_ser : Continuous (fun ω : ℝ =>
      ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) :=
    continuous_tsum
      (fun m => continuous_const.mul
        (Real.continuous_cos.comp (continuous_const.mul continuous_id)))
      hδ_abs
      (fun m ω => by
        rw [Real.norm_eq_abs, abs_mul]
        exact mul_le_of_le_one_right (abs_nonneg _) (Real.abs_cos_le_one _))
  have hser_int : IntervalIntegrable
      (fun ω => f ω * ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) volume 0 Real.pi :=
    (hf_cont.mul hcont_ser.continuousOn).intervalIntegrable_of_Icc Real.pi_pos.le
  have hsplit_int :
      (∫ ω in (0 : ℝ)..Real.pi, f ω * K_ser ω) =
        (1 / (2 * T ^ 2)) * (∫ ω in (0 : ℝ)..Real.pi, f ω) -
          (1 / T) * (∫ ω in (0 : ℝ)..Real.pi,
            f ω * ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) := by
    have hfn : (fun ω => f ω * K_ser ω) =
        fun ω => (1 / (2 * T ^ 2)) * f ω -
          (1 / T) * (f ω * ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) := by
      funext ω; dsimp [K_ser]; ring
    simp_rw [hfn]
    rw [intervalIntegral.integral_sub (hf_int.const_mul _) (hser_int.const_mul _),
      intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul]
  rw [hsplit_int, hswap_int]
  have hf_pi : (∫ ω in (0 : ℝ)..Real.pi, f ω) = Real.pi := by
    have hp := Real.pi_pos.ne'
    field_simp [hp] at hf_norm
    linarith
  rw [hf_pi]
  have hC_eq := (edgeLocalCost_eq_covarianceCoefficient μ K S hS a ξ hξ Ku β hKu hβ0 hβ1 hmix).2
  rw [hC_eq, singleEdgeScore_covarianceCoefficient, hvar, mul_one]
  have htsum_rew :
      (∑' m : ℕ, (∫ x, ξ ((S : Ω → Ω)^[m + 1] x) * ξ x ∂μ) * K.initial a *
        ((K.prob ^ (m + 1)) a a - K.initial a)) =
      K.initial a * ∑' m : ℕ, (∫ x, ξ ((S : Ω → Ω)^[m + 1] x) * ξ x ∂μ) * δ m := by
    rw [← tsum_mul_left]
    refine tsum_congr fun m => by dsimp [δ]; ring
  rw [htsum_rew, hc_eq]
  field_simp [Real.pi_pos.ne', hT_pos.ne']

/-- **Frequency average (`prop:ring-spectrum`(c)):**
`(1 / π) ∫_0^π K(ω) dω = 1 / (2 T²)`. -/
theorem spectral_kernel_mean {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : E) (hc : 0 < K.initial a)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ m : ℕ, |(K.prob ^ (m + 1)) a a - K.initial a| ≤ Ku * β ^ m) :
    let T := meanReturnTime K a
    let g := returnCharFun K a
    (1 / Real.pi) * (∫ ω in (0 : ℝ)..Real.pi, spectralKernel T g ω) = 1 / (2 * T ^ 2) := by
  intro T g
  obtain ⟨_hT_eq, hT_pos⟩ := meanReturnTime_eq_inv_initial K a hc Ku β hKu hβ0 hβ1 hmix
  let δ : ℕ → ℝ := fun m => (K.prob ^ (m + 1)) a a - K.initial a
  let K_ser : ℝ → ℝ := fun ω =>
    1 / (2 * T ^ 2) - (1 / T) * ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)
  have hae_eq : ∀ᵐ ω ∂volume, ω ∈ Set.uIoc (0 : ℝ) Real.pi →
      spectralKernel T g ω = K_ser ω := by
    have h1 : ({0, Real.pi} : Set ℝ)ᶜ ∈ ae volume := by
      rw [compl_mem_ae_iff]
      exact (Set.Finite.insert 0 (Set.finite_singleton Real.pi)).measure_zero volume
    filter_upwards [h1] with ω hω hmem
    rw [Set.uIoc_of_le Real.pi_pos.le] at hmem
    have h0 : 0 < ω := lt_of_le_of_ne hmem.1.le (fun h => hω (by left; exact h.symm))
    have hpi : ω < Real.pi := lt_of_le_of_ne hmem.2 (fun h => hω (by right; exact h))
    exact (spectral_cosine_series K a hc Ku β hKu hβ0 hβ1 hmix ω
      (exp_ofReal_mul_I_ne_one_of_Ioo h0 (by linarith [Real.pi_pos]))).2.2
  rw [intervalIntegral.integral_congr_ae hae_eq]
  have hδ_abs : Summable (fun m : ℕ => |δ m|) :=
    Summable.of_nonneg_of_le (fun _ => abs_nonneg _) hmix
      ((summable_geometric_of_lt_one hβ0 hβ1).mul_left Ku)
  have hcos_sum : ∀ ω : ℝ, Summable (fun m : ℕ => δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) := by
    intro ω
    refine Summable.of_norm_bounded hδ_abs fun m => ?_
    rw [Real.norm_eq_abs, abs_mul]
    exact mul_le_of_le_one_right (abs_nonneg _) (Real.abs_cos_le_one _)
  have hcont_ser : Continuous (fun ω : ℝ =>
      ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) :=
    continuous_tsum
      (fun m => continuous_const.mul
        (Real.continuous_cos.comp (continuous_const.mul continuous_id)))
      hδ_abs
      (fun m ω => by
        rw [Real.norm_eq_abs, abs_mul]
        exact mul_le_of_le_one_right (abs_nonneg _) (Real.abs_cos_le_one _))
  have hcos_int_zero : ∀ m : ℕ,
      (∫ ω in (0 : ℝ)..Real.pi, Real.cos (((m + 1 : ℕ) : ℝ) * ω)) = 0 := by
    intro m
    let k : ℝ := ((m + 1 : ℕ) : ℝ)
    have hk_ne : k ≠ 0 := by dsimp [k]; positivity
    have hderiv : ∀ x ∈ Set.uIcc (0 : ℝ) Real.pi,
        HasDerivAt (fun ω => Real.sin (k * ω) / k) (Real.cos (k * x)) x := by
      intro x _
      have hd := ((hasDerivAt_id x).const_mul k).sin.div_const k
      dsimp only [id_eq] at hd
      convert hd using 1
      field_simp [hk_ne]
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv
      ((Real.continuous_cos.comp (continuous_const.mul continuous_id)).intervalIntegrable _ _)]
    have hsin_pi : Real.sin (k * Real.pi) = 0 := by
      dsimp [k]; exact Real.sin_nat_mul_pi (m + 1)
    simp [hsin_pi]
  have hterm_int : ∀ m : ℕ,
      IntervalIntegrable (fun ω => δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) volume 0 Real.pi := by
    intro m
    exact (continuous_const.mul
      (Real.continuous_cos.comp (continuous_const.mul continuous_id))).intervalIntegrable _ _
  have hdom_zero : Tendsto
      (fun N : ℕ => ∫ ω in (0 : ℝ)..Real.pi,
        ∑ m ∈ Finset.range N, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))
      atTop
      (𝓝 (∫ ω in (0 : ℝ)..Real.pi,
        ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))) := by
    apply intervalIntegral.tendsto_integral_filter_of_dominated_convergence
      (fun _ => ∑' m : ℕ, |δ m|)
    · refine Eventually.of_forall fun N => ?_
      rw [Set.uIoc_of_le Real.pi_pos.le]
      convert Finset.aestronglyMeasurable_sum (Finset.range N)
        (fun m _ => (hterm_int m).1.aestronglyMeasurable) using 1
      funext ω; simp [Finset.sum_apply]
    · refine Eventually.of_forall fun N => Eventually.of_forall fun ω _ => ?_
      rw [Real.norm_eq_abs]
      calc |∑ m ∈ Finset.range N, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)|
          ≤ ∑ m ∈ Finset.range N, |δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)| :=
            Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ m ∈ Finset.range N, |δ m| := Finset.sum_le_sum fun m _ => by
            rw [abs_mul]; exact mul_le_of_le_one_right (abs_nonneg _) (Real.abs_cos_le_one _)
        _ ≤ ∑' m : ℕ, |δ m| := hδ_abs.sum_le_tsum (Finset.range N) fun m _ => abs_nonneg _
    · exact intervalIntegrable_const
    · exact Eventually.of_forall fun ω _ => (hcos_sum ω).hasSum.tendsto_sum_nat
  have hpart_zero : ∀ N : ℕ,
      (∫ ω in (0 : ℝ)..Real.pi,
        ∑ m ∈ Finset.range N, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) = 0 := by
    intro N
    rw [intervalIntegral_finset_sum (Finset.range N) _ 0 Real.pi (fun m _ => hterm_int m)]
    apply Finset.sum_eq_zero
    intro m _
    rw [intervalIntegral.integral_const_mul, hcos_int_zero m, mul_zero]
  simp_rw [hpart_zero] at hdom_zero
  have hint_ser_zero :
      (∫ ω in (0 : ℝ)..Real.pi, ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) = 0 :=
    tendsto_nhds_unique hdom_zero tendsto_const_nhds
  have hT_ne : T ≠ 0 := hT_pos.ne'
  dsimp [K_ser]
  rw [intervalIntegral.integral_sub intervalIntegrable_const
      (hcont_ser.intervalIntegrable _ _ |>.const_mul _),
    intervalIntegral.integral_const, intervalIntegral.integral_const_mul, hint_ser_zero]
  simp only [sub_zero, smul_eq_mul, mul_zero]
  field_simp [Real.pi_pos.ne', hT_ne]

/-- **Low-frequency limit (`prop:ring-spectrum`(c)):**
`lim_{ω → 0, ω ≠ 0} K(ω) = (1 - CV²) / (2 T)`, together with finiteness of the second
recurrence-time moment `E_a[L_a²]` under geometric mixing. -/
theorem spectral_kernel_zero_limit {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : E) (hc : 0 < K.initial a)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ m : ℕ, |(K.prob ^ (m + 1)) a a - K.initial a| ≤ Ku * β ^ m) :
    let T := meanReturnTime K a
    let g := returnCharFun K a
    Summable (fun m : ℕ => (((m + 1 : ℕ) : ℝ) ^ 2) * firstReturnProb K a (m + 1)) ∧
      Tendsto (spectralKernel T g) (𝓝[≠] 0) (𝓝 ((1 - returnCV2 K a) / (2 * T))) := by
  intro T g
  obtain ⟨hf_has, hmom1_has⟩ := kac_mean_return_time K a hc Ku β hKu hβ0 hβ1 hmix
  obtain ⟨hT_eq, hT_pos⟩ := meanReturnTime_eq_inv_initial K a hc Ku β hKu hβ0 hβ1 hmix
  have hc_eq : K.initial a = 1 / T := by dsimp [T]; rw [hT_eq, one_div_one_div]
  let f : ℕ → ℝ := firstReturnProb K a
  let δ : ℕ → ℝ := fun m => (K.prob ^ (m + 1)) a a - K.initial a
  have hδ_abs : Summable (fun m : ℕ => |δ m|) :=
    Summable.of_nonneg_of_le (fun _ => abs_nonneg _) hmix
      ((summable_geometric_of_lt_one hβ0 hβ1).mul_left Ku)
  -- Step 1: Kac's second-moment identity via `rseq n = 1 - 1 / (n + 2) ↑ 1`.
  let rseq : ℕ → ℝ := fun n => 1 - 1 / ((n : ℝ) + 2)
  have hr_pos : ∀ n : ℕ, 0 < rseq n := by
    intro n; dsimp [rseq]
    have hgt : (1 : ℝ) < (n : ℝ) + 2 := by linarith [Nat.cast_nonneg (α := ℝ) n]
    linarith [div_lt_one (by positivity : (0 : ℝ) < (n : ℝ) + 2) |>.mpr hgt]
  have hr_lt1 : ∀ n : ℕ, rseq n < 1 := by
    intro n; dsimp [rseq]
    have hp : 0 < 1 / ((n : ℝ) + 2) := by positivity
    linarith
  have hr_tend : Tendsto rseq atTop (𝓝 1) := by
    have hinv : Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 2)) atTop (𝓝 0) :=
      tendsto_const_nhds.div_atTop (tendsto_natCast_atTop_atTop.atTop_add tendsto_const_nhds)
    simpa [rseq] using (tendsto_const_nhds (x := (1 : ℝ))).sub hinv
  let geomSum : ℕ → ℝ → ℝ := fun m r => ∑ j ∈ Finset.range (m + 1), r ^ j
  let secondGeom : ℕ → ℝ → ℝ := fun m r => ∑ j ∈ Finset.Ico 1 (m + 1), geomSum (j - 1) r
  have hsg_id : ∀ n m : ℕ,
      (((m + 1 : ℕ) : ℝ) - geomSum m (rseq n)) / (1 - rseq n) = secondGeom m (rseq n) := by
    intro n m
    have h1r_ne : (1 - rseq n) ≠ 0 := (sub_pos.mpr (hr_lt1 n)).ne'
    apply (div_eq_iff h1r_ne).mpr
    induction m with
    | zero => simp [geomSum, secondGeom]
    | succ m ihm =>
      have hgs : geomSum (m + 1) (rseq n) = geomSum m (rseq n) + (rseq n) ^ (m + 1) :=
        Finset.sum_range_succ _ (m + 1)
      have hsg_step : secondGeom (m + 1) (rseq n) = secondGeom m (rseq n) + geomSum m (rseq n) := by
        dsimp [secondGeom]
        rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ m + 1)]
        simp
      have hm_geom : geomSum m (rseq n) * (1 - rseq n) = 1 - (rseq n) ^ (m + 1) := by
        have hs := geom_sum_mul (rseq n) (m + 1)
        calc geomSum m (rseq n) * (1 - rseq n)
            = -((∑ i ∈ Finset.range (m + 1), (rseq n) ^ i) * (rseq n - 1)) := by
                dsimp [geomSum]; ring
          _ = 1 - (rseq n) ^ (m + 1) := by linarith [hs]
      calc (((m + 1 + 1 : ℕ) : ℝ) - geomSum (m + 1) (rseq n))
          = (((m + 1 : ℕ) : ℝ) - geomSum m (rseq n)) + (1 - (rseq n) ^ (m + 1)) := by
              rw [hgs]; push_cast; ring
        _ = secondGeom m (rseq n) * (1 - rseq n) + geomSum m (rseq n) * (1 - rseq n) := by
              rw [ihm, hm_geom]
        _ = secondGeom (m + 1) (rseq n) * (1 - rseq n) := by
              rw [hsg_step]; ring
  let A_real : ℝ → ℝ := fun r => ∑' m : ℕ, f (m + 1) * geomSum m r
  let D_real : ℝ → ℝ := fun r => ∑' m : ℕ, r ^ (m + 1) * δ m
  have hD_tend : Tendsto (fun n : ℕ => D_real (rseq n)) atTop (𝓝 (∑' m : ℕ, δ m)) := by
    refine tendsto_tsum_of_dominated_convergence hδ_abs (fun m => ?_)
      (Eventually.of_forall fun n m => ?_)
    · simpa using (hr_tend.pow (m + 1)).mul_const (δ m)
    · rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_pos (hr_pos n)]
      exact mul_le_of_le_one_left (abs_nonneg _) (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le)
  have hA_tend : Tendsto (fun n : ℕ => A_real (rseq n)) atTop (𝓝 T) := by
    have hlim : Tendsto (fun n : ℕ => ∑' m : ℕ, f (m + 1) * geomSum m (rseq n))
        atTop (𝓝 (∑' m : ℕ, f (m + 1) * geomSum m 1)) := by
      refine tendsto_tsum_of_dominated_convergence hmom1_has.summable (fun m => ?_)
        (Eventually.of_forall fun n m => ?_)
      · exact (tendsto_finsetSum (Finset.range (m + 1)) fun j _ => hr_tend.pow j).const_mul _
      · rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (firstReturnProb_nonneg K a _),
          abs_of_nonneg (Finset.sum_nonneg fun j _ => pow_nonneg (hr_pos n).le j)]
        have hle : geomSum m (rseq n) ≤ ((m + 1 : ℕ) : ℝ) := by
          calc ∑ j ∈ Finset.range (m + 1), (rseq n) ^ j
              ≤ ∑ _j ∈ Finset.range (m + 1), (1 : ℝ) :=
                Finset.sum_le_sum fun j _ => pow_le_one₀ (hr_pos n).le (hr_lt1 n).le
            _ = ((m + 1 : ℕ) : ℝ) := by simp
        nlinarith [firstReturnProb_nonneg K a (m + 1)]
    refine hlim.trans_eq (congrArg 𝓝 ?_)
    dsimp [T, meanReturnTime, f]
    refine tsum_congr fun m => ?_
    have hgs1 : geomSum m 1 = ((m + 1 : ℕ) : ℝ) := by simp [geomSum]
    rw [hgs1, mul_comm]
  -- Algebraic identity `(T - A_real (rseq n)) / (1 - rseq n) = T * A_real (rseq n) * (1 + D_real (rseq n) - K.initial a)`
  have hsec_eq : ∀ n : ℕ,
      (T - A_real (rseq n)) / (1 - rseq n) =
        T * A_real (rseq n) * (1 + D_real (rseq n) - K.initial a) := by
    intro n
    let r := rseq n
    have h1r_ne : 1 - r ≠ 0 := (sub_pos.mpr (hr_lt1 n)).ne'
    have hnorm : ‖(r : ℂ)‖ < 1 := by
      rw [Complex.norm_real, Real.norm_eq_abs, abs_of_pos (hr_pos n)]
      exact hr_lt1 n
    have hser_c := (chain_return_series K a (r : ℂ) hnorm).1
    let R_r : ℝ := ∑' k : ℕ, f k * r ^ k
    have hRc : returnTransform K a (r : ℂ) = (R_r : ℂ) := by
      rw [returnTransform_eq_tsum K a (r : ℂ) hnorm]
      dsimp [R_r, f]
      rw [Complex.ofReal_tsum]
      exact tsum_congr fun k => by push_cast; rfl
    rw [hRc] at hser_c
    have hser_r : HasSum (fun m : ℕ => r ^ (m + 1) * (K.prob ^ (m + 1)) a a)
        (R_r / (1 - R_r)) := by
      apply Complex.hasSum_ofReal.mp
      simpa only [Complex.ofReal_mul, Complex.ofReal_pow, Complex.ofReal_div,
        Complex.ofReal_sub, Complex.ofReal_one] using hser_c
    have hD_has : HasSum (fun m : ℕ => r ^ (m + 1) * δ m) (D_real r) := by
      refine (Summable.of_norm_bounded hδ_abs fun m => ?_).hasSum
      rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_pos (hr_pos n)]
      exact mul_le_of_le_one_left (abs_nonneg _) (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le)
    have hgeom_r : HasSum (fun m : ℕ => r ^ (m + 1) * K.initial a)
        (K.initial a * (r / (1 - r))) := by
      have hg := ((hasSum_geometric_of_lt_one (hr_pos n).le (hr_lt1 n)).mul_left r).mul_right
        (K.initial a)
      convert hg using 1
      · funext m; rw [pow_succ']
      · ring
    have hcomb := hD_has.add hgeom_r
    have hfn : (fun m : ℕ => r ^ (m + 1) * δ m + r ^ (m + 1) * K.initial a) =
        fun m : ℕ => r ^ (m + 1) * (K.prob ^ (m + 1)) a a := by
      funext m; dsimp [δ]; ring
    rw [hfn] at hcomb
    have hsplit_ser : D_real r + K.initial a * (r / (1 - r)) = R_r / (1 - R_r) :=
      hcomb.unique hser_r
    have hR_lt1 : R_r < 1 := by
      have hsum_fr : Summable (fun k : ℕ => f k * r ^ k) := by
        refine Summable.of_nonneg_of_le
          (fun k => mul_nonneg (firstReturnProb_nonneg K a k) (pow_nonneg (hr_pos n).le k))
          (fun k => mul_le_of_le_one_right (firstReturnProb_nonneg K a k)
            (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le)) hf_has.summable
      calc R_r ≤ ∑' k : ℕ, f k * r := by
            refine hsum_fr.tsum_le_tsum (fun k => ?_) (hf_has.summable.mul_right r)
            cases k with
            | zero => simp [f, firstReturnProb]
            | succ m =>
              rw [pow_succ']
              exact mul_le_mul_of_nonneg_left
                (mul_le_of_le_one_right (hr_pos n).le (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le))
                (firstReturnProb_nonneg K a (m + 1))
        _ = (∑' k : ℕ, f k) * r := tsum_mul_right
        _ = r := by rw [hf_has.tsum_eq, one_mul]
        _ < 1 := hr_lt1 n
    have h1R_ne : 1 - R_r ≠ 0 := (sub_pos.mpr hR_lt1).ne'
    -- Show `A_real r = (1 - R_r) / (1 - r)`
    have hA_eq : A_real r = (1 - R_r) / (1 - r) := by
      have hf_shift : HasSum (fun m : ℕ => f (m + 1)) 1 := by
        simpa [f, firstReturnProb] using (hasSum_nat_add_iff' 1).mpr hf_has
      have hR_shift : HasSum (fun m : ℕ => f (m + 1) * r ^ (m + 1)) R_r := by
        have hsum_fr : Summable (fun k : ℕ => f k * r ^ k) := by
          refine Summable.of_nonneg_of_le
            (fun k => mul_nonneg (firstReturnProb_nonneg K a k) (pow_nonneg (hr_pos n).le k))
            (fun k => mul_le_of_le_one_right (firstReturnProb_nonneg K a k)
              (pow_le_one₀ (hr_pos n).le (hr_lt1 n).le)) hf_has.summable
        simpa [R_r, f, firstReturnProb] using (hasSum_nat_add_iff' 1).mpr hsum_fr.hasSum
      have hsub := (hf_shift.sub hR_shift).div_const (1 - r)
      have hcongr : (fun m : ℕ => (f (m + 1) - f (m + 1) * r ^ (m + 1)) / (1 - r)) =
          fun m : ℕ => f (m + 1) * geomSum m r := by
        funext m
        have hs := geom_sum_mul r (m + 1)
        apply (div_eq_iff h1r_ne).mpr
        have h1 : 1 - r ^ (m + 1) = geomSum m r * (1 - r) := by
          calc 1 - r ^ (m + 1) = -((∑ i ∈ Finset.range (m + 1), r ^ i) * (r - 1)) := by linarith [hs]
            _ = geomSum m r * (1 - r) := by dsimp [geomSum]; ring
        calc f (m + 1) - f (m + 1) * r ^ (m + 1) = f (m + 1) * (1 - r ^ (m + 1)) := by ring
          _ = f (m + 1) * (geomSum m r * (1 - r)) := by rw [h1]
          _ = f (m + 1) * geomSum m r * (1 - r) := by ring
      rw [hcongr] at hsub
      exact hsub.tsum_eq
    have hunit : A_real r * ((1 - r) * (1 + D_real r) + K.initial a * r) = 1 := by
      rw [hA_eq]
      have h1 : 1 + (D_real r + K.initial a * (r / (1 - r))) = 1 / (1 - R_r) := by
        rw [hsplit_ser]; field_simp [h1R_ne]; ring
      calc ((1 - R_r) / (1 - r)) * ((1 - r) * (1 + D_real r) + K.initial a * r)
          = ((1 - R_r) / (1 - r)) * ((1 - r) * (1 + (D_real r + K.initial a * (r / (1 - r))))) := by
              congr 1; field_simp [h1r_ne]; ring
        _ = ((1 - R_r) / (1 - r)) * ((1 - r) * (1 / (1 - R_r))) := by rw [h1]
        _ = 1 := by field_simp [h1r_ne, h1R_ne]
    apply (div_eq_iff h1r_ne).mpr
    have hcT : K.initial a * T = 1 := by
      have hTne : T ≠ 0 := hT_pos.ne'
      rw [hc_eq]; field_simp [hTne]
    calc T - A_real r
        = T * (A_real r * ((1 - r) * (1 + D_real r) + K.initial a * r)) -
            (K.initial a * T) * A_real r := by rw [hunit, hcT]; ring
      _ = T * A_real r * (1 + D_real r - K.initial a) * (1 - r) := by ring
  have hsec_tsum : ∀ n : ℕ,
      (T - A_real (rseq n)) / (1 - rseq n) =
        ∑' m : ℕ, f (m + 1) * secondGeom m (rseq n) := by
    intro n
    have hA_has : HasSum (fun m : ℕ => f (m + 1) * geomSum m (rseq n)) (A_real (rseq n)) := by
      refine (Summable.of_nonneg_of_le
        (fun m => mul_nonneg (firstReturnProb_nonneg K a _)
          (Finset.sum_nonneg fun j _ => pow_nonneg (hr_pos n).le j))
        (fun m => ?_) hmom1_has.summable).hasSum
      have hle : geomSum m (rseq n) ≤ ((m + 1 : ℕ) : ℝ) := by
        calc ∑ j ∈ Finset.range (m + 1), (rseq n) ^ j
            ≤ ∑ _j ∈ Finset.range (m + 1), (1 : ℝ) :=
              Finset.sum_le_sum fun j _ => pow_le_one₀ (hr_pos n).le (hr_lt1 n).le
          _ = ((m + 1 : ℕ) : ℝ) := by simp
      nlinarith [firstReturnProb_nonneg K a (m + 1)]
    have hT_has : HasSum (fun m : ℕ => ((m + 1 : ℕ) : ℝ) * f (m + 1)) T := by
      dsimp [T, meanReturnTime, f]
      exact hmom1_has.summable.hasSum
    have hsub := (hT_has.sub hA_has).div_const (1 - rseq n)
    have hcongr : (fun m : ℕ =>
        (((m + 1 : ℕ) : ℝ) * f (m + 1) - f (m + 1) * geomSum m (rseq n)) / (1 - rseq n)) =
        fun m : ℕ => f (m + 1) * secondGeom m (rseq n) := by
      funext m
      rw [← hsg_id n m]
      ring
    rw [hcongr] at hsub
    exact hsub.tsum_eq.symm
  have hsg_at_one : ∀ m : ℕ, secondGeom m 1 = ((m : ℝ) * ((m : ℝ) + 1)) / 2 := by
    intro m
    induction m with
    | zero => simp [secondGeom]
    | succ k ihk =>
      have hstep : secondGeom (k + 1) 1 = secondGeom k 1 + geomSum k 1 := by
        dsimp [secondGeom]
        rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ k + 1)]
        simp
      have hgs1 : geomSum k 1 = (k : ℝ) + 1 := by simp [geomSum]
      rw [hstep, ihk, hgs1]
      push_cast; ring
  have hrhs_tend : Tendsto (fun n : ℕ => (T - A_real (rseq n)) / (1 - rseq n))
      atTop (𝓝 (T ^ 2 * (1 + (∑' m : ℕ, δ m) - K.initial a))) := by
    simp_rw [hsec_eq]
    have ht := (hA_tend.const_mul T).mul
      (((tendsto_const_nhds (x := (1 : ℝ))).add hD_tend).sub (tendsto_const_nhds (x := K.initial a)))
    simpa [sq] using ht
  -- Finite partial sums give `Summable` of `f(m+1) * m(m+1)/2` and hence `f(m+1) * (m+1)^2`:
  let L_sec : ℝ := T ^ 2 * (1 + (∑' m : ℕ, δ m) - K.initial a)
  have hsg_nn : ∀ n m : ℕ, 0 ≤ secondGeom m (rseq n) := fun n m =>
    Finset.sum_nonneg fun j _ => Finset.sum_nonneg fun i _ => pow_nonneg (hr_pos n).le i
  have hsg_le : ∀ n m : ℕ, secondGeom m (rseq n) ≤ ((m : ℝ) * ((m : ℝ) + 1)) / 2 := by
    intro n m
    rw [← hsg_at_one m]
    refine Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun i _ => ?_
    simpa using pow_le_one₀ (hr_pos n).le (hr_lt1 n).le
  have hbinom_fin_le : ∀ s : Finset ℕ,
      ∑ m ∈ s, (((m : ℝ) * ((m : ℝ) + 1)) / 2) * f (m + 1) ≤ L_sec := by
    intro s
    have hfin_tend : Tendsto (fun n : ℕ => ∑ m ∈ s, f (m + 1) * secondGeom m (rseq n))
        atTop (𝓝 (∑ m ∈ s, (((m : ℝ) * ((m : ℝ) + 1)) / 2) * f (m + 1))) := by
      have ht : Tendsto (fun n : ℕ => ∑ m ∈ s, f (m + 1) * secondGeom m (rseq n))
          atTop (𝓝 (∑ m ∈ s, f (m + 1) * secondGeom m 1)) :=
        tendsto_finsetSum s fun m _ =>
          (tendsto_finsetSum (Finset.Ico 1 (m + 1)) fun j _ =>
            tendsto_finsetSum (Finset.range (j - 1 + 1)) fun i _ => hr_tend.pow i).const_mul _
      have heq_s : (∑ m ∈ s, f (m + 1) * secondGeom m 1) =
          ∑ m ∈ s, (((m : ℝ) * ((m : ℝ) + 1)) / 2) * f (m + 1) :=
        Finset.sum_congr rfl fun m _ => by rw [hsg_at_one m, mul_comm]
      rwa [heq_s] at ht
    have hle_seq : ∀ n : ℕ,
        ∑ m ∈ s, f (m + 1) * secondGeom m (rseq n) ≤ (T - A_real (rseq n)) / (1 - rseq n) := by
      intro n
      rw [hsec_tsum n]
      have hsum_n : Summable (fun m : ℕ => f (m + 1) * secondGeom m (rseq n)) := by
        refine Summable.of_nonneg_of_le
          (fun m => mul_nonneg (firstReturnProb_nonneg K a _) (hsg_nn n m))
          (fun m => ?_) (hmom1_has.summable.mul_right (1 / (1 - rseq n)))
        rw [← hsg_id n m]
        have hgs_nn : 0 ≤ geomSum m (rseq n) :=
          Finset.sum_nonneg fun j _ => pow_nonneg (hr_pos n).le j
        calc f (m + 1) * ((((m + 1 : ℕ) : ℝ) - geomSum m (rseq n)) / (1 - rseq n))
            ≤ f (m + 1) * (((m + 1 : ℕ) : ℝ) / (1 - rseq n)) :=
              mul_le_mul_of_nonneg_left
                (div_le_div_of_nonneg_right (sub_le_self _ hgs_nn) (sub_pos.mpr (hr_lt1 n)).le)
                (firstReturnProb_nonneg K a _)
          _ = (((m + 1 : ℕ) : ℝ) * f (m + 1)) * (1 / (1 - rseq n)) := by ring
      exact hsum_n.sum_le_tsum s fun m _ =>
        mul_nonneg (firstReturnProb_nonneg K a _) (hsg_nn n m)
    exact le_of_tendsto_of_tendsto hfin_tend hrhs_tend (Eventually.of_forall hle_seq)
  have hbinom_sum : Summable (fun m : ℕ => (((m : ℝ) * ((m : ℝ) + 1)) / 2) * f (m + 1)) :=
    summable_of_sum_le
      (fun m => mul_nonneg (by positivity) (firstReturnProb_nonneg K a (m + 1))) hbinom_fin_le
  have hmom2_sum : Summable (fun m : ℕ => (((m + 1 : ℕ) : ℝ) ^ 2) * f (m + 1)) := by
    have hcomb := (hbinom_sum.mul_left 2).add hmom1_has.summable
    refine hcomb.congr fun m => ?_
    dsimp [f]; push_cast; ring
  have hsec_tend_tsum : Tendsto (fun n : ℕ => (T - A_real (rseq n)) / (1 - rseq n))
      atTop (𝓝 (∑' m : ℕ, (((m : ℝ) * ((m : ℝ) + 1)) / 2) * f (m + 1))) := by
    simp_rw [hsec_tsum]
    have hdom : Tendsto (fun n : ℕ => ∑' m : ℕ, f (m + 1) * secondGeom m (rseq n))
        atTop (𝓝 (∑' m : ℕ, f (m + 1) * secondGeom m 1)) := by
      refine tendsto_tsum_of_dominated_convergence hbinom_sum (fun m => ?_)
        (Eventually.of_forall fun n m => ?_)
      · exact (tendsto_finsetSum (Finset.Ico 1 (m + 1)) fun j _ =>
          tendsto_finsetSum (Finset.range (j - 1 + 1)) fun i _ => hr_tend.pow i).const_mul _
      · rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (firstReturnProb_nonneg K a _),
          abs_of_nonneg (hsg_nn n m)]
        nlinarith [hsg_le n m, firstReturnProb_nonneg K a (m + 1)]
    refine hdom.trans_eq (congrArg 𝓝 ?_)
    exact tsum_congr fun m => by rw [hsg_at_one m, mul_comm]
  have hbinom_eq : (∑' m : ℕ, (((m : ℝ) * ((m : ℝ) + 1)) / 2) * f (m + 1)) = L_sec :=
    tendsto_nhds_unique hsec_tend_tsum hrhs_tend
  have hT_has_top : HasSum (fun m : ℕ => ((m + 1 : ℕ) : ℝ) * f (m + 1)) T := by
    dsimp [T, meanReturnTime, f]
    exact hmom1_has.summable.hasSum
  have hmom2_rel : returnSecondMoment K a = 2 * L_sec + T := by
    have h1 := (hbinom_sum.hasSum.mul_left 2).add hT_has_top
    have hfn : (fun m : ℕ => 2 * ((((m : ℝ) * ((m : ℝ) + 1)) / 2) * f (m + 1)) +
        ((m + 1 : ℕ) : ℝ) * f (m + 1)) =
        fun m : ℕ => (((m + 1 : ℕ) : ℝ) ^ 2) * f (m + 1) := by
      funext m; push_cast; ring
    rw [hfn] at h1
    rw [hbinom_eq] at h1
    exact h1.tsum_eq
  -- Step 2: Evaluate `lim_{ω → 0, ω ≠ 0} spectralKernel T g ω`.
  have hcont_ser : Continuous (fun ω : ℝ =>
      ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) :=
    continuous_tsum
      (fun m => continuous_const.mul
        (Real.continuous_cos.comp (continuous_const.mul continuous_id)))
      hδ_abs
      (fun m ω => by
        rw [Real.norm_eq_abs, abs_mul]
        exact mul_le_of_le_one_right (abs_nonneg _) (Real.abs_cos_le_one _))
  have hser_at_zero : Tendsto
      (fun ω : ℝ => 1 / (2 * T ^ 2) - (1 / T) * ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))
      (𝓝[≠] 0) (𝓝 (1 / (2 * T ^ 2) - (1 / T) * ∑' m : ℕ, δ m)) := by
    have hcont0 : Tendsto (fun ω : ℝ => ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω))
        (𝓝 (0 : ℝ)) (𝓝 (∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * (0 : ℝ)))) :=
      (hcont_ser.continuousAt (x := (0 : ℝ))).tendsto
    have h0 := ((tendsto_const_nhds (x := 1 / (2 * T ^ 2))).sub
      (hcont0.const_mul (1 / T))).mono_left (nhdsWithin_le_nhds (s := ({0} : Set ℝ)ᶜ))
    simpa [mul_zero, Real.cos_zero, mul_one] using h0
  have hevent_eq : (fun ω : ℝ =>
      1 / (2 * T ^ 2) - (1 / T) * ∑' m : ℕ, δ m * Real.cos (((m + 1 : ℕ) : ℝ) * ω)) =ᶠ[𝓝[≠] 0]
      spectralKernel T g := by
    have hball : Ioo (-Real.pi) Real.pi ∈ 𝓝 (0 : ℝ) :=
      Ioo_mem_nhds (neg_neg_of_pos Real.pi_pos) Real.pi_pos
    filter_upwards [mem_nhdsWithin_of_mem_nhds hball, self_mem_nhdsWithin] with ω hω hne
    have hζ_ne : Complex.exp ((ω : ℂ) * Complex.I) ≠ 1 := by
      rcases lt_trichotomy ω 0 with hneg | rfl | hpos
      · intro heq
        have him := congrArg Complex.im heq
        rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin] at him
        simp only [Complex.add_im, Complex.ofReal_im, Complex.mul_im, Complex.I_re, Complex.I_im,
          Complex.ofReal_re, mul_one, mul_zero, add_zero, zero_add, Complex.one_im] at him
        have hsin_pos := Real.sin_pos_of_pos_of_lt_pi (neg_pos.mpr hneg) (by linarith [hω.1])
        rw [Real.sin_neg] at hsin_pos
        linarith
      · exact False.elim (hne rfl)
      · exact exp_ofReal_mul_I_ne_one_of_Ioo hpos (by linarith [hω.2, Real.pi_pos])
    exact (spectral_cosine_series K a hc Ku β hKu hβ0 hβ1 hmix ω hζ_ne).2.2.symm
  have htarget_eq : 1 / (2 * T ^ 2) - (1 / T) * (∑' m : ℕ, δ m) =
      (1 - returnCV2 K a) / (2 * T) := by
    have hT_ne : meanReturnTime K a ≠ 0 := hT_pos.ne'
    dsimp [L_sec] at hmom2_rel
    dsimp [returnCV2, T] at hmom2_rel hc_eq ⊢
    rw [hmom2_rel, hc_eq]
    field_simp [hT_ne]
    ring
  refine ⟨hmom2_sum, ?_⟩
  rw [← htarget_eq]
  exact hser_at_zero.congr' hevent_eq

/-- Characteristic function `(1 / τ) e^{i ω} / (1 - (1 - 1 / τ) e^{i ω})` of a single geometric
stay with mean `τ ≥ 1`. -/
noncomputable def oneStageCharFun (τ ω : ℝ) : ℂ :=
  ((1 / τ : ℝ) : ℂ) * Complex.exp ((ω : ℂ) * Complex.I) /
    (1 - ((1 - 1 / τ : ℝ) : ℂ) * Complex.exp ((ω : ℂ) * Complex.I))

/-- **Constant one-stage transfer kernel (`prop:ring-spectrum`(c)):**
For a single geometric stay of mean `τ ≥ 1`, `K(ω) = 1 / (2 τ²)` for every `ω` with `e^{i ω} ≠ 1`. -/
theorem spectral_kernel_one_stage (τ ω : ℝ) (hτ : 1 ≤ τ)
    (hζ : Complex.exp ((ω : ℂ) * Complex.I) ≠ 1) :
    spectralKernel τ (oneStageCharFun τ) ω = 1 / (2 * τ ^ 2) := by
  have hτ_pos : 0 < τ := lt_of_lt_of_le zero_lt_one hτ
  have hτc_ne : (τ : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hτ_pos.ne'
  let ζ : ℂ := Complex.exp ((ω : ℂ) * Complex.I)
  have hζ_norm : ‖ζ‖ = 1 := Complex.norm_exp_ofReal_mul_I ω
  have hζ_nsq : Complex.normSq ζ = 1 := by rw [Complex.normSq_eq_norm_sq, hζ_norm, one_pow]
  have h1ζ_ne : 1 - ζ ≠ 0 := sub_ne_zero.mpr hζ.symm
  have hm_nn : 0 ≤ 1 - 1 / τ := by
    rw [sub_nonneg, div_le_one hτ_pos]; exact hτ
  have hden_ne : 1 - ((1 - 1 / τ : ℝ) : ℂ) * ζ ≠ 0 := by
    intro h0
    have heq : ((1 - 1 / τ : ℝ) : ℂ) * ζ = 1 := by linear_combination -h0
    have hn := congrArg (‖·‖) heq
    rw [norm_mul, hζ_norm, mul_one, norm_one, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonneg hm_nn] at hn
    have : 1 / τ = 0 := by linarith
    exact (one_div_pos.mpr hτ_pos).ne' this
  have h1sub : 1 - oneStageCharFun τ ω =
      (1 - ζ) / (1 - ((1 - 1 / τ : ℝ) : ℂ) * ζ) := by
    unfold oneStageCharFun
    change 1 - ((1 / τ : ℝ) : ℂ) * ζ / (1 - ((1 - 1 / τ : ℝ) : ℂ) * ζ) = _
    rw [one_sub_div hden_ne]
    congr 1
    push_cast; ring
  have h1g_ne : 1 - oneStageCharFun τ ω ≠ 0 := by
    rw [h1sub]; exact div_ne_zero h1ζ_ne hden_ne
  have hratio : oneStageCharFun τ ω / (1 - oneStageCharFun τ ω) =
      ((1 / τ : ℝ) : ℂ) * (ζ / (1 - ζ)) := by
    rw [h1sub]
    unfold oneStageCharFun
    change (((1 / τ : ℝ) : ℂ) * ζ / (1 - ((1 - 1 / τ : ℝ) : ℂ) * ζ)) /
        ((1 - ζ) / (1 - ((1 - 1 / τ : ℝ) : ℂ) * ζ)) =
      ((1 / τ : ℝ) : ℂ) * (ζ / (1 - ζ))
    rw [div_div_div_cancel_right₀ hden_ne, mul_div_assoc]
  unfold spectralKernel
  rw [hratio, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero,
    unitCircle_div_one_sub_re ζ hζ_nsq hζ]
  field_simp [hτ_pos.ne']

/-- **Multi-stage low-frequency vs. mean inequality (`prop:ring-spectrum`(c)):**
For `n ≥ 2` stages with positive mean stays `τ k > 0`, `1 / (2 T²) < (1 - CV²) / (2 T)`. -/
theorem ring_low_freq_gt_mean (n : ℕ) (hn : 2 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 0 < τ k) :
    let T := ∑ k : Fin n, τ k
    let CV2 := (∑ k : Fin n, τ k * (τ k - 1)) / T ^ 2
    1 / (2 * T ^ 2) < (1 - CV2) / (2 * T) := by
  intro T CV2
  let i0 : Fin n := ⟨0, by omega⟩
  let i1 : Fin n := ⟨1, by omega⟩
  have hne01 : i0 ≠ i1 := by
    intro h
    have hv := congrArg Fin.val h
    dsimp [i0, i1] at hv
    omega
  have hT_pos : 0 < T := Finset.sum_pos (fun k _ => hτ k) ⟨i0, Finset.mem_univ _⟩
  have hsq_lt : (∑ k : Fin n, τ k ^ 2) < T ^ 2 := by
    dsimp [T]
    rw [sq, Finset.sum_mul_sum]
    have hdiag : (∑ k : Fin n, τ k ^ 2) =
        ∑ k : Fin n, ∑ j : Fin n, (if k = j then τ k * τ j else 0) := by
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [Finset.sum_ite_eq]
      simp [sq]
    rw [hdiag]
    refine Finset.sum_lt_sum
      (fun k _ => Finset.sum_le_sum fun j _ => by
        have hk := hτ k; have hj := hτ j
        split_ifs <;> linarith [mul_pos hk hj])
      ⟨i0, Finset.mem_univ i0, ?_⟩
    refine Finset.sum_lt_sum
      (fun j _ => by
        have hk := hτ i0; have hj := hτ j
        split_ifs <;> linarith [mul_pos hk hj])
      ⟨i1, Finset.mem_univ i1, ?_⟩
    simp only [hne01, ↓reduceIte]
    exact mul_pos (hτ i0) (hτ i1)
  have hdiff : (1 - CV2) / (2 * T) - 1 / (2 * T ^ 2) =
      (T ^ 2 - ∑ k : Fin n, τ k ^ 2) / (2 * T ^ 3) := by
    have hvar_expand : (∑ k : Fin n, τ k * (τ k - 1)) = (∑ k : Fin n, τ k ^ 2) - T := by
      have h1 : ∀ k : Fin n, τ k * (τ k - 1) = τ k ^ 2 - τ k := fun k => by ring
      simp_rw [h1, Finset.sum_sub_distrib]
      rfl
    dsimp [CV2]
    rw [hvar_expand]
    have hT_ne : T ≠ 0 := hT_pos.ne'
    field_simp [hT_ne]
    ring
  have hpos_rhs : 0 < (T ^ 2 - ∑ k : Fin n, τ k ^ 2) / (2 * T ^ 3) :=
    div_pos (sub_pos.mpr hsq_lt) (by positivity)
  linarith

/-- Characteristic function of the Figure-2 eight-stage primitive ring (`τ = 3/2`, `T = 12`). -/
noncomputable def eightStageCharFun (ω : ℝ) : ℂ :=
  (oneStageCharFun (3 / 2) ω) ^ 8

/-- **Negativity and non-monotonicity of `K(ω)` (`prop:ring-spectrum`(c), Figure 2):**
For the eight-stage primitive ring with `τ = 3/2` (`T = 12`), `K(ω)` achieves strictly negative
values on `(0, π)` and is neither monotone nor antitone on `(0, π)`. -/
theorem spectral_kernel_nonmonotone :
    let K := spectralKernel 12 eightStageCharFun
    (∃ ω ∈ Ioo 0 Real.pi, K ω < 0) ∧
      ¬ MonotoneOn K (Ioo 0 Real.pi) ∧
      ¬ AntitoneOn K (Ioo 0 Real.pi) := by
  intro K
  let ω₁ : ℝ := Real.arccos (3 / 5)
  let ω₂ : ℝ := Real.arccos (8 / 17)
  let ω₃ : ℝ := Real.arccos 0
  have hω₁_mem : ω₁ ∈ Ioo 0 Real.pi :=
    ⟨Real.arccos_pos.mpr (by norm_num), Real.arccos_lt_pi.mpr (by norm_num)⟩
  have hω₂_mem : ω₂ ∈ Ioo 0 Real.pi :=
    ⟨Real.arccos_pos.mpr (by norm_num), Real.arccos_lt_pi.mpr (by norm_num)⟩
  have hω₃_mem : ω₃ ∈ Ioo 0 Real.pi :=
    ⟨Real.arccos_pos.mpr (by norm_num), Real.arccos_lt_pi.mpr (by norm_num)⟩
  have hω₁₂ : ω₁ < ω₂ :=
    Real.strictAntiOn_arccos ⟨by norm_num, by norm_num⟩ ⟨by norm_num, by norm_num⟩ (by norm_num)
  have hω₂₃ : ω₂ < ω₃ :=
    Real.strictAntiOn_arccos ⟨by norm_num, by norm_num⟩ ⟨by norm_num, by norm_num⟩ (by norm_num)
  have hexp_arccos : ∀ c s : ℝ, -1 ≤ c → c ≤ 1 → 0 ≤ s → c ^ 2 + s ^ 2 = 1 →
      Complex.exp ((Real.arccos c : ℂ) * Complex.I) = (c : ℂ) + (s : ℂ) * Complex.I := by
    intro c s hc1 hc2 hs hsq
    rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_arccos hc1 hc2]
    have hsin : Real.sin (Real.arccos c) = s := by
      rw [Real.sin_arccos, show 1 - c ^ 2 = s ^ 2 by linarith, Real.sqrt_sq hs]
    rw [hsin]
  have hζ₁ : Complex.exp ((ω₁ : ℂ) * Complex.I) = (3 / 5 : ℂ) + (4 / 5 : ℂ) * Complex.I := by
    have h := hexp_arccos (3 / 5) (4 / 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    push_cast at h
    exact h
  have hζ₂ : Complex.exp ((ω₂ : ℂ) * Complex.I) = (8 / 17 : ℂ) + (15 / 17 : ℂ) * Complex.I := by
    have h := hexp_arccos (8 / 17) (15 / 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    push_cast at h
    exact h
  have hζ₃ : Complex.exp ((ω₃ : ℂ) * Complex.I) = Complex.I := by
    have h := hexp_arccos 0 1 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    simp only [Complex.ofReal_zero, Complex.ofReal_one, zero_add, one_mul] at h
    exact h
  have hg₁_step : oneStageCharFun (3 / 2) ω₁ = (1 / 4 : ℂ) + (3 / 4 : ℂ) * Complex.I := by
    unfold oneStageCharFun
    rw [hζ₁]
    have hden : (1 : ℂ) - ((1 - 1 / (3 / 2 : ℝ) : ℝ) : ℂ) * ((3 / 5 : ℂ) + (4 / 5 : ℂ) * Complex.I) ≠ 0 := by
      intro h0
      have hr := congrArg Complex.re h0
      norm_num at hr
    apply (div_eq_iff hden).mpr
    apply Complex.ext <;> norm_num
  have hg₂_step : oneStageCharFun (3 / 2) ω₂ = (7 / 61 : ℂ) + (45 / 61 : ℂ) * Complex.I := by
    unfold oneStageCharFun
    rw [hζ₂]
    have hden : (1 : ℂ) - ((1 - 1 / (3 / 2 : ℝ) : ℝ) : ℂ) * ((8 / 17 : ℂ) + (15 / 17 : ℂ) * Complex.I) ≠ 0 := by
      intro h0
      have hr := congrArg Complex.re h0
      norm_num at hr
    apply (div_eq_iff hden).mpr
    apply Complex.ext <;> norm_num
  have hg₃_step : oneStageCharFun (3 / 2) ω₃ = (-1 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I := by
    unfold oneStageCharFun
    rw [hζ₃]
    have hden : (1 : ℂ) - ((1 - 1 / (3 / 2 : ℝ) : ℝ) : ℂ) * Complex.I ≠ 0 := by
      intro h0
      have hr := congrArg Complex.re h0
      norm_num at hr
    apply (div_eq_iff hden).mpr
    apply Complex.ext <;> norm_num
  have hp₁ : ((1 / 4 : ℂ) + (3 / 4 : ℂ) * Complex.I) ^ 8 =
      (-527 / 4096 : ℂ) - (336 / 4096 : ℂ) * Complex.I := by
    have h2 : ((1 / 4 : ℂ) + (3 / 4 : ℂ) * Complex.I) ^ 2 = (-8 / 16 : ℂ) + (6 / 16 : ℂ) * Complex.I := by
      apply Complex.ext <;> simp [sq] <;> norm_num
    have h4 : (((1 / 4 : ℂ) + (3 / 4 : ℂ) * Complex.I) ^ 2) ^ 2 = (28 / 256 : ℂ) - (96 / 256 : ℂ) * Complex.I := by
      rw [h2]; apply Complex.ext <;> simp [sq] <;> norm_num
    calc ((1 / 4 : ℂ) + (3 / 4 : ℂ) * Complex.I) ^ 8
        = ((((1 / 4 : ℂ) + (3 / 4 : ℂ) * Complex.I) ^ 2) ^ 2) ^ 2 := by ring
      _ = (-527 / 4096 : ℂ) - (336 / 4096 : ℂ) * Complex.I := by
          rw [h4]; apply Complex.ext <;> simp [sq] <;> norm_num
  let a₂ : ℝ := (3507676 / 13845841) ^ 2 - (2489760 / 13845841) ^ 2
  let b₂ : ℝ := -2 * (3507676 / 13845841) * (2489760 / 13845841)
  have hp₂ : ((7 / 61 : ℂ) + (45 / 61 : ℂ) * Complex.I) ^ 8 =
      (a₂ : ℂ) + (b₂ : ℂ) * Complex.I := by
    have h2 : ((7 / 61 : ℂ) + (45 / 61 : ℂ) * Complex.I) ^ 2 = (-1976 / 3721 : ℂ) + (630 / 3721 : ℂ) * Complex.I := by
      apply Complex.ext <;> simp [sq] <;> norm_num
    have h4 : (((7 / 61 : ℂ) + (45 / 61 : ℂ) * Complex.I) ^ 2) ^ 2 =
        (3507676 / 13845841 : ℂ) - (2489760 / 13845841 : ℂ) * Complex.I := by
      rw [h2]; apply Complex.ext <;> simp [sq] <;> norm_num
    calc ((7 / 61 : ℂ) + (45 / 61 : ℂ) * Complex.I) ^ 8
        = ((((7 / 61 : ℂ) + (45 / 61 : ℂ) * Complex.I) ^ 2) ^ 2) ^ 2 := by ring
      _ = (a₂ : ℂ) + (b₂ : ℂ) * Complex.I := by
          rw [h4]; dsimp [a₂, b₂]; apply Complex.ext <;> (simp [sq]; try norm_num)
  have hp₃ : ((-1 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I) ^ 8 =
      (-8432 / 390625 : ℂ) + (5376 / 390625 : ℂ) * Complex.I := by
    have h2 : ((-1 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I) ^ 2 = (-8 / 25 : ℂ) - (6 / 25 : ℂ) * Complex.I := by
      apply Complex.ext <;> simp [sq] <;> norm_num
    have h4 : (((-1 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I) ^ 2) ^ 2 = (28 / 625 : ℂ) + (96 / 625 : ℂ) * Complex.I := by
      rw [h2]; apply Complex.ext <;> simp [sq] <;> norm_num
    calc ((-1 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I) ^ 8
        = ((((-1 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I) ^ 2) ^ 2) ^ 2 := by ring
      _ = (-8432 / 390625 : ℂ) + (5376 / 390625 : ℂ) * Complex.I := by
          rw [h4]; apply Complex.ext <;> simp [sq] <;> norm_num
  have hK₁_pos : 0 < K ω₁ := by
    dsimp [K]
    unfold spectralKernel eightStageCharFun
    rw [Dobson.Coverage3Renewal.spectral_real_part, hg₁_step, hp₁]
    norm_num [Complex.normSq_apply]
  have hK₂_neg : K ω₂ < 0 := by
    dsimp [K]
    unfold spectralKernel eightStageCharFun
    rw [Dobson.Coverage3Renewal.spectral_real_part, hg₂_step, hp₂]
    dsimp [a₂, b₂]
    norm_num [Complex.normSq_apply]
  have hK₃_pos : 0 < K ω₃ := by
    dsimp [K]
    unfold spectralKernel eightStageCharFun
    rw [Dobson.Coverage3Renewal.spectral_real_part, hg₃_step, hp₃]
    norm_num [Complex.normSq_apply]
  refine ⟨⟨ω₂, hω₂_mem, hK₂_neg⟩, ?_, ?_⟩
  · intro hmono
    have hle := hmono hω₁_mem hω₂_mem hω₁₂.le
    linarith
  · intro hanti
    have hge := hanti hω₂_mem hω₃_mem hω₂₃.le
    linarith

end SpectralKernel

end Dobson.P1bA_RingSpectrum

#print axioms Dobson.P1bA_RingSpectrum.complex_renewal_pgf
#print axioms Dobson.P1bA_RingSpectrum.chain_return_series
#print axioms Dobson.P1bA_RingSpectrum.kac_mean_return_time
#print axioms Dobson.P1bA_RingSpectrum.mixture_localCost_eq
#print axioms Dobson.P1bA_RingSpectrum.mixture_coefficient_unconditional
#print axioms Dobson.P1bA_RingSpectrum.ar2_process_autocorrelation_eq
#print axioms Dobson.P1bA_RingSpectrum.ar2_yule_walker_mixture
#print axioms Dobson.P1bA_RingSpectrum.spectral_cosine_series
#print axioms Dobson.P1bA_RingSpectrum.spectral_integral_representation
#print axioms Dobson.P1bA_RingSpectrum.spectral_kernel_mean
#print axioms Dobson.P1bA_RingSpectrum.spectral_kernel_zero_limit
#print axioms Dobson.P1bA_RingSpectrum.spectral_kernel_one_stage
#print axioms Dobson.P1bA_RingSpectrum.ring_low_freq_gt_mean
#print axioms Dobson.P1bA_RingSpectrum.spectral_kernel_nonmonotone

