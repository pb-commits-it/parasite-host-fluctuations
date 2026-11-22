import DobsonProofs.Coverage3Networks
import DobsonProofs.ReferenceRing

/-! # Part B: quadratic parallel balance and a reference-path bypass coefficient

Coverage correction, 2026-11-22: `prop:mech-routes` remains partial-P1.
The independent fidelity audit (2026-11-16) did not credit this module as
closing the physical threshold and per-step-cost bridges `part2:37–38`.

The parallel results differentiate a linear expected-reward surrogate and
solve the supplied quadratic balance; they do not derive a restoring root
from tilted log-partition growth. `bypass_localCost_eq` proves a finite-path
second-derivative cost for the explicit bypass reference chain, but is
uncredited in the version-2 inventory. Full physical growth/threshold
identification remains outside credit. -/

namespace Dobson.P1bB_MechRoutes

open MeasureTheory Filter Dobson.FiniteLineageVariance
open scoped BigOperators Topology

/-! ## Section 1: Parallel-network expected reward and quadratic balance -/

section ParallelBirthTilt

variable {J Ω : Type*} [Fintype J] [DecidableEq J] [MeasurableSpace Ω]

/-- Indicator reward of reproductive return-to-central edges `(some j, none)`. -/
def returnEdgeIndicator : (Option J × Option J) → ℝ
  | (some _, none) => 1
  | _ => 0

/-- Stationary expected one-step birth indicator equals the return-to-central birth frequency. -/
lemma parallel_birth_indicator_expect (τ0 : ℝ) (τ w : J → ℝ)
    (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1) :
    (∑ e : Option J × Option J,
        ((Dobson.ParallelReference.stateChain τ0 τ w hτ0 hτ hw hwsum).edges).initial e *
          returnEdgeIndicator e) =
      ∑ j : J, Dobson.ParallelReference.stationaryMass τ0 τ w (some j) *
        Dobson.ParallelReference.transition τ0 τ w (some j) none := by
  simp only [Chain.edges, Fintype.sum_prod_type, Fintype.sum_option, returnEdgeIndicator,
    mul_zero, mul_one, Finset.sum_const_zero, zero_add, add_zero]
  rfl

/-- Derivative of `y` times the normalized expected return-edge reward.
Its slope is the stationary birth frequency `b = 1 / totalTime`. This linear
function is not the tilted path log partition. -/
theorem parallel_deterministic_birth_slope
    (τ0 : ℝ) (τ w : J → ℝ) (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j)
    (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1) (N : ℕ) (y : ℝ) :
    let K := (Dobson.ParallelReference.stateChain τ0 τ w hτ0 hτ hw hwsum).edges
    let b := ∑ j : J, Dobson.ParallelReference.stationaryMass τ0 τ w (some j) *
      Dobson.ParallelReference.transition τ0 τ w (some j) none
    HasDerivAt (fun z : ℝ => z * (Dobson.FiniteLineageVariance.expect K (N + 1)
        (Dobson.FiniteLineageVariance.reward (N + 1) (fun _ e => returnEdgeIndicator e)) /
        (N + 1 : ℕ)))
      b y ∧
    b = 1 / Dobson.ParallelReference.totalTime τ0 τ w := by
  intro K b
  have hexp := Dobson.FinitePathJet.expect_constant_reward K returnEdgeIndicator (N + 1)
  have hinit : (∑ e, K.initial e * returnEdgeIndicator e) = b :=
    parallel_birth_indicator_expect τ0 τ w hτ0 hτ hw hwsum
  have hpos : (N + 1 : ℝ) ≠ 0 := by positivity
  have hval : Dobson.FiniteLineageVariance.expect K (N + 1)
      (Dobson.FiniteLineageVariance.reward (N + 1) (fun _ e => returnEdgeIndicator e)) /
        (N + 1 : ℕ) = b := by
    rw [hexp, hinit, Nat.cast_add, Nat.cast_one]
    field_simp [hpos]
  refine ⟨?_, Dobson.ParallelReference.birth_frequency τ0 τ w hτ hwsum⟩
  rw [hval]
  convert (hasDerivAt_id y).mul_const b using 1
  · funext z; simp
  · ring

/-- Scalar consequence of the supplied balance `b * Δ - quadraticCost = 0`:
positivity of `b` gives `Δ = quadraticCost / b`. The balance is a hypothesis;
this does not construct a physical restoring threshold. -/
lemma parallel_threshold_from_balance
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (τ0 : ℝ) (τ w κ : J → ℝ) (ξ : J → Ω → ℝ)
    (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1)
    (S : Ω → Ω) (σ Δ : ℝ)
    (hbalance : (∑ j, Dobson.ParallelReference.stationaryMass τ0 τ w (some j) *
        Dobson.ParallelReference.transition τ0 τ w (some j) none) * Δ -
      Dobson.ParallelReference.quadraticCost μ τ0 τ w κ ξ hτ0 hτ hw hwsum S σ = 0) :
    Δ = Dobson.ParallelReference.quadraticCost μ τ0 τ w κ ξ hτ0 hτ hw hwsum S σ /
      (∑ j, Dobson.ParallelReference.stationaryMass τ0 τ w (some j) *
        Dobson.ParallelReference.transition τ0 τ w (some j) none) := by
  have hT : 0 < Dobson.ParallelReference.totalTime τ0 τ w :=
    Dobson.ParallelReference.totalTime_pos τ0 τ w hτ0 hτ hw
  have hb_eq := Dobson.ParallelReference.birth_frequency τ0 τ w hτ hwsum
  have hb_pos : 0 < ∑ j, Dobson.ParallelReference.stationaryMass τ0 τ w (some j) *
      Dobson.ParallelReference.transition τ0 τ w (some j) none := by
    rw [hb_eq]
    positivity
  apply (eq_div_iff hb_pos.ne').mpr
  linarith

/-- The quadratic balance candidate is defined as `quadraticCost / b`,
where `b` is the return-edge frequency. No moving-root expansion is proved
by this definition. -/
noncomputable def parallelQuadraticThreshold
    (μ : Measure Ω) (τ0 : ℝ) (τ w κ : J → ℝ) (ξ : J → Ω → ℝ)
    (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1)
    (S : Ω → Ω) (σ : ℝ) : ℝ :=
  let b := ∑ j, Dobson.ParallelReference.stationaryMass τ0 τ w (some j) *
    Dobson.ParallelReference.transition τ0 τ w (some j) none
  Dobson.ParallelReference.quadraticCost μ τ0 τ w κ ξ hτ0 hτ hw hwsum S σ / b

/-- Unique solution of the displayed scalar quadratic balance, using the
positive return-edge frequency and the reference-chain cost formula. This
is not existence or uniqueness of a physical growth root. -/
theorem parallel_threshold_unconditional
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (S : Ω ≃ᵐ Ω) (hS : MeasurePreserving S μ μ)
    (τ0 : ℝ) (τ w κ : J → ℝ) (ξ : J → Ω → ℝ)
    (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j)
    (hwsum : ∑ j, w j = 1) (hξ : ∀ j, MemLp (ξ j) 2 μ)
    (hlag : ∀ i j m, lagCovariance μ S ξ i j (m + 1) = 0) (σ : ℝ) :
    ∃! Δ : ℝ,
      (∑ j, Dobson.ParallelReference.stationaryMass τ0 τ w (some j) *
          Dobson.ParallelReference.transition τ0 τ w (some j) none) * Δ -
        Dobson.ParallelReference.quadraticCost μ τ0 τ w κ ξ hτ0 hτ hw hwsum S σ = 0 ∧
      Δ = σ ^ 2 * Dobson.ParallelReference.routeVariance μ κ w ξ /
        (2 * Dobson.ParallelReference.totalTime τ0 τ w) := by
  let b := ∑ j, Dobson.ParallelReference.stationaryMass τ0 τ w (some j) *
    Dobson.ParallelReference.transition τ0 τ w (some j) none
  let C := Dobson.ParallelReference.quadraticCost μ τ0 τ w κ ξ hτ0 hτ hw hwsum S σ
  have hT : 0 < Dobson.ParallelReference.totalTime τ0 τ w :=
    Dobson.ParallelReference.totalTime_pos τ0 τ w hτ0 hτ hw
  have hb_eq : b = 1 / Dobson.ParallelReference.totalTime τ0 τ w :=
    Dobson.ParallelReference.birth_frequency τ0 τ w hτ hwsum
  have hb_ne : b ≠ 0 := by rw [hb_eq]; positivity
  refine ⟨C / b, ⟨?_, ?_⟩, ?_⟩
  · change b * (C / b) - C = 0
    rw [mul_div_cancel₀ C hb_ne, sub_self]
  · exact Dobson.Coverage3Networks.parallel_threshold μ S hS τ0 τ w κ ξ
      hτ0 hτ hw hwsum hξ hlag σ (C / b) rfl
  · rintro Δ ⟨hbal, _⟩
    exact parallel_threshold_from_balance μ τ0 τ w κ ξ hτ0 hτ hw hwsum S σ Δ hbal

end ParallelBirthTilt

/-! ## Section 2: Actual two-state bypass chain and AR(1) return resolvent (`part2:38`) -/

section BypassResolvent

lemma chain_pow_nonneg {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (n : ℕ) (i j : E) : 0 ≤ (K.prob ^ n) i j := by
  induction n generalizing i j with
  | zero => simp only [pow_zero, Matrix.one_apply]; split_ifs <;> norm_num
  | succ n ih =>
    rw [pow_succ, Matrix.mul_apply]
    exact Finset.sum_nonneg fun k _ => mul_nonneg (ih i k) (K.nonneg k j)

lemma chain_pow_total {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (n : ℕ) (i : E) : ∑ j, (K.prob ^ n) i j = 1 := by
  induction n generalizing i with
  | zero => simp [Matrix.one_apply]
  | succ n ih =>
    simp only [pow_succ, Matrix.mul_apply]
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, K.total, mul_one, ih]

lemma chain_pow_le_one {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (n : ℕ) (i j : E) : (K.prob ^ n) i j ≤ 1 := by
  rw [← chain_pow_total K n i]
  exact Finset.single_le_sum (fun k _ => chain_pow_nonneg K n i k) (Finset.mem_univ _)

lemma chain_initial_le_one {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (i : E) : K.initial i ≤ 1 := by
  rw [← K.initial_total]
  exact Finset.single_le_sum (fun k _ => K.initial_nonneg k) (Finset.mem_univ _)

lemma chain_pow_sub_initial_abs_le_one {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (n : ℕ) (i j : E) :
    |(K.prob ^ n) i j - K.initial j| ≤ 1 :=
  abs_le.mpr ⟨by linarith [chain_pow_nonneg K n i j, chain_initial_le_one K j],
    by linarith [chain_pow_le_one K n i j, K.initial_nonneg j]⟩

lemma realVisits_summable {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (z : ℝ) (hz : |z| < 1) (i j : E) :
    Summable (fun m : ℕ => z ^ (m + 1) * (K.prob ^ (m + 1)) i j) := by
  apply Summable.of_norm_bounded (summable_geometric_of_lt_one (abs_nonneg z) hz)
  intro m
  rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_nonneg (chain_pow_nonneg K _ _ _), pow_succ]
  calc |z| ^ m * |z| * (K.prob ^ (m + 1)) i j ≤ |z| ^ m * |z| :=
      mul_le_of_le_one_right (by positivity) (chain_pow_le_one K _ _ _)
    _ ≤ |z| ^ m := mul_le_of_le_one_right (by positivity) hz.le

noncomputable def realVisits {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : E) (z : ℝ) (i : E) : ℝ :=
  ∑' m : ℕ, z ^ (m + 1) * (K.prob ^ (m + 1)) i a

lemma realVisits_rec {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : E) (z : ℝ) (hz : |z| < 1) (i : E) :
    realVisits K a z i = z * K.prob i a + z * ∑ j, K.prob i j * realVisits K a z j := by
  have hs := realVisits_summable K z hz i a
  have hsplit := hs.sum_add_tsum_nat_add 1
  simp only [Finset.sum_range_one, zero_add, pow_one] at hsplit
  unfold realVisits
  rw [← hsplit]
  congr 1
  have he (m : ℕ) : z ^ (m + 1 + 1) * (K.prob ^ (m + 1 + 1)) i a =
      z * ∑ j, K.prob i j * (z ^ (m + 1) * (K.prob ^ (m + 1)) j a) := by
    rw [pow_succ' K.prob, Matrix.mul_apply, Finset.mul_sum, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  simp_rw [he]
  rw [tsum_mul_left, Summable.tsum_finsetSum
    (fun j _ => (realVisits_summable K z hz j a).mul_left (K.prob i j))]
  simp_rw [tsum_mul_left]

lemma realVisits_eq_of_row {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : E) (z : ℝ) (i j : E)
    (hrow : ∀ k, K.prob i k = K.prob j k) :
    realVisits K a z i = realVisits K a z j := by
  unfold realVisits
  apply tsum_congr
  intro m
  congr 2
  rw [pow_succ', Matrix.mul_apply, Matrix.mul_apply]
  simp_rw [hrow]

/-- Total generation time of the two-state bypass reference chain: `T = τ₀ + α * τ₁`. -/
noncomputable def bypassTotalTime (τ0 τ1 α : ℝ) : ℝ := τ0 + α * τ1

noncomputable def bypassProb (τ0 τ1 α : ℝ) : Matrix Bool Bool ℝ
  | false, false => 1 - α / τ0
  | false, true => α / τ0
  | true, false => 1 / τ1
  | true, true => 1 - 1 / τ1

noncomputable def bypassInitial (τ0 τ1 α : ℝ) : Bool → ℝ
  | false => τ0 / bypassTotalTime τ0 τ1 α
  | true => α * τ1 / bypassTotalTime τ0 τ1 α

lemma bypassTotalTime_pos (τ0 τ1 α : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1) (hα : 0 ≤ α) :
    0 < bypassTotalTime τ0 τ1 α := by
  unfold bypassTotalTime
  have := mul_nonneg hα (show 0 ≤ τ1 by linarith)
  linarith

noncomputable def bypassStateChain (τ0 τ1 α : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) : Chain Bool where
  prob := bypassProb τ0 τ1 α
  nonneg := by
    intro i j
    have h0 : 0 < τ0 := by linarith
    have h1 : 0 < τ1 := by linarith
    cases i <;> cases j
    · exact sub_nonneg.mpr ((div_le_one h0).mpr (by linarith))
    · exact div_nonneg hα0 h0.le
    · exact one_div_nonneg.mpr h1.le
    · exact sub_nonneg.mpr ((div_le_one h1).mpr hτ1)
  total := by intro i; cases i <;> simp [bypassProb]
  initial := bypassInitial τ0 τ1 α
  initial_nonneg := by
    intro i
    have ht := (bypassTotalTime_pos τ0 τ1 α hτ0 hτ1 hα0).le
    cases i
    · exact div_nonneg (by linarith) ht
    · exact div_nonneg (mul_nonneg hα0 (by linarith)) ht
  initial_total := by
    have ht := (bypassTotalTime_pos τ0 τ1 α hτ0 hτ1 hα0).ne'
    simp only [Fintype.sum_bool, bypassInitial, ← add_div]
    unfold bypassTotalTime at ht ⊢
    field_simp [ht]
    ring
  stationary := by
    funext j
    have h0 : τ0 ≠ 0 := by linarith
    have h1 : τ1 ≠ 0 := by linarith
    have ht := (bypassTotalTime_pos τ0 τ1 α hτ0 hτ1 hα0).ne'
    unfold bypassTotalTime at ht
    cases j <;> simp [Matrix.vecMul, dotProduct, bypassInitial, bypassProb, bypassTotalTime] <;>
      field_simp [h0, h1, ht] <;> ring

noncomputable def bypassEdgeChain (τ0 τ1 α : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) : Chain (Bool × Bool) :=
  (bypassStateChain τ0 τ1 α hτ0 hτ1 hα0 hα1).edges

lemma bypass_entrance_frequency (τ0 τ1 α : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) :
    (bypassEdgeChain τ0 τ1 α hτ0 hτ1 hα0 hα1).initial (false, true) =
      α / bypassTotalTime τ0 τ1 α := by
  have h0 : τ0 ≠ 0 := by linarith
  dsimp [bypassEdgeChain, Chain.edges, bypassStateChain, bypassInitial, bypassProb]
  field_simp

lemma bypass_visit_source (τ0 τ1 α : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (z : ℝ) (i j : Bool) :
    realVisits (bypassEdgeChain τ0 τ1 α hτ0 hτ1 hα0 hα1) (false, true) z (i, j) =
      realVisits (bypassEdgeChain τ0 τ1 α hτ0 hτ1 hα0 hα1) (false, true) z (j, j) := by
  apply realVisits_eq_of_row
  intro _
  rfl

lemma bypass_visit_equations (τ0 τ1 α : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (z : ℝ) (hz : |z| < 1) :
    let K := bypassEdgeChain τ0 τ1 α hτ0 hτ1 hα0 hα1
    let U0 := realVisits K (false, true) z (false, false)
    let U1 := realVisits K (false, true) z (true, true)
    U0 = z * α / τ0 + z * ((1 - α / τ0) * U0 + (α / τ0) * U1) ∧
      U1 = z * ((1 / τ1) * U0 + (1 - 1 / τ1) * U1) := by
  intro K U0 U1
  have h0 := realVisits_rec K (false, true) z hz (false, false)
  have h1 := realVisits_rec K (false, true) z hz (true, true)
  simp only [Fintype.sum_prod_type, Fintype.sum_bool] at h0 h1
  have hp00 : K.prob (false, false) (false, false) = 1 - α / τ0 := rfl
  have hp01 : K.prob (false, false) (false, true) = α / τ0 := rfl
  have hp02 : K.prob (false, false) (true, false) = 0 := rfl
  have hp03 : K.prob (false, false) (true, true) = 0 := rfl
  have hp10 : K.prob (true, true) (false, false) = 0 := rfl
  have hp11 : K.prob (true, true) (false, true) = 0 := rfl
  have hp12 : K.prob (true, true) (true, false) = 1 / τ1 := rfl
  have hp13 : K.prob (true, true) (true, true) = 1 - 1 / τ1 := rfl
  rw [hp00, hp01, hp02, hp03] at h0
  rw [hp10, hp11, hp12, hp13] at h1
  have h01 : realVisits K (false, true) z (false, true) = U1 :=
    bypass_visit_source τ0 τ1 α hτ0 hτ1 hα0 hα1 z false true
  have h10 : realVisits K (false, true) z (true, false) = U0 :=
    bypass_visit_source τ0 τ1 α hτ0 hτ1 hα0 hα1 z true false
  have h00 : realVisits K (false, true) z (false, false) = U0 := rfl
  have h11 : realVisits K (false, true) z (true, true) = U1 := rfl
  rw [h01, h10, h00, h11] at h0 h1
  constructor
  · linear_combination h0
  · linear_combination h1

noncomputable def waitTransform (τ z : ℝ) : ℝ := z / (τ - (τ - 1) * z)
noncomputable def bypassCircuitTransform (τ0 τ1 α z : ℝ) : ℝ :=
  waitTransform τ0 z * (1 - α + α * waitTransform τ1 z)

lemma wait_denominator_pos (τ z : ℝ) (hτ : 1 ≤ τ) (hz : |z| < 1) :
    0 < τ - (τ - 1) * z := by
  have hh := mul_nonneg (sub_nonneg.mpr hτ) (show 0 ≤ 1 - z by have := (abs_lt.mp hz).2; linarith)
  nlinarith

lemma waitTransform_abs_lt (τ z : ℝ) (hτ : 1 ≤ τ) (hz : |z| < 1) :
    |waitTransform τ z| < 1 := by
  have hd := wait_denominator_pos τ z hτ hz
  have hge : 1 ≤ τ - (τ - 1) * z := by
    have hh := mul_nonneg (sub_nonneg.mpr hτ)
      (show 0 ≤ 1 - z by have := (abs_lt.mp hz).2; linarith)
    nlinarith
  rw [waitTransform, abs_div, abs_of_pos hd]
  exact (div_lt_one hd).mpr (hz.trans_le hge)

lemma bypassCircuitTransform_abs_lt (τ0 τ1 α z : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (hz : |z| < 1) :
    |bypassCircuitTransform τ0 τ1 α z| < 1 := by
  have hq0 := waitTransform_abs_lt τ0 z hτ0 hz
  have hq1 := waitTransform_abs_lt τ1 z hτ1 hz
  have hb : |1 - α + α * waitTransform τ1 z| ≤ 1 := by
    calc |1 - α + α * waitTransform τ1 z| ≤ |1 - α| + |α * waitTransform τ1 z| := abs_add_le _ _
      _ = 1 - α + α * |waitTransform τ1 z| := by
        rw [abs_of_nonneg (sub_nonneg.mpr hα1), abs_mul, abs_of_nonneg hα0]
      _ ≤ 1 := by nlinarith [mul_le_mul_of_nonneg_left hq1.le hα0]
  rw [bypassCircuitTransform, abs_mul]
  exact (mul_le_of_le_one_right (abs_nonneg _) hb).trans_lt hq0

/-- Exact discounted return series of the noisy bypass entrance `(false, true)`. -/
theorem bypass_return_formula (τ0 τ1 α : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (z : ℝ) (hz : |z| < 1) :
    realVisits (bypassEdgeChain τ0 τ1 α hτ0 hτ1 hα0 hα1) (false, true) z (false, true) =
      α * waitTransform τ0 z * waitTransform τ1 z /
        (1 - bypassCircuitTransform τ0 τ1 α z) := by
  rw [bypass_visit_source]
  obtain ⟨h0, h1⟩ := bypass_visit_equations τ0 τ1 α hτ0 hτ1 hα0 hα1 z hz
  let K := bypassEdgeChain τ0 τ1 α hτ0 hτ1 hα0 hα1
  let U0 := realVisits K (false, true) z (false, false)
  let U1 := realVisits K (false, true) z (true, true)
  change U0 = z * α / τ0 + z * ((1 - α / τ0) * U0 + (α / τ0) * U1) at h0
  change U1 = z * ((1 / τ1) * U0 + (1 - 1 / τ1) * U1) at h1
  change U1 = α * waitTransform τ0 z * waitTransform τ1 z /
    (1 - bypassCircuitTransform τ0 τ1 α z)
  have hτ0n : τ0 ≠ 0 := by linarith
  have hτ1n : τ1 ≠ 0 := by linarith
  have hd0 := (wait_denominator_pos τ0 z hτ0 hz).ne'
  have hd1 := (wait_denominator_pos τ1 z hτ1 hz).ne'
  have hr : 1 - bypassCircuitTransform τ0 τ1 α z ≠ 0 := by
    have hh := (abs_lt.mp (bypassCircuitTransform_abs_lt τ0 τ1 α z hτ0 hτ1 hα0 hα1 hz)).2
    linarith
  apply (eq_div_iff hr).mpr
  have h0' : (τ0 - z * (τ0 - α)) * U0 = z * α + z * α * U1 := by
    have hh : U0 * τ0 = (z * α / τ0 + z * ((1 - α / τ0) * U0 + α / τ0 * U1)) * τ0 :=
      congrArg (fun x => x * τ0) h0
    field_simp [hτ0n] at hh
    linear_combination hh
  have h1' : (τ1 - (τ1 - 1) * z) * U1 = z * U0 := by
    have hh : U1 * τ1 = z * (1 / τ1 * U0 + (1 - 1 / τ1) * U1) * τ1 :=
      congrArg (fun x => x * τ1) h1
    field_simp [hτ1n] at hh
    linear_combination hh
  have htarget : ∀ D0 D1 : ℝ, D0 ≠ 0 → D1 ≠ 0 →
      D0 = τ0 - (τ0 - 1) * z → D1 = τ1 - (τ1 - 1) * z →
      U1 * (1 - z / D0 * (1 - α + α * (z / D1))) = α * (z / D0) * (z / D1) := by
    intro D0 D1 hD0 hD1 hD0_eq hD1_eq
    have hpoly : U1 * (D0 * D1 - z * ((1 - α) * D1 + α * z)) = α * z ^ 2 := by
      rw [hD0_eq, hD1_eq]
      linear_combination z * h0' + (τ0 - z * (τ0 - α)) * h1'
    field_simp [hD0, hD1]
    linear_combination hpoly
  simpa only [bypassCircuitTransform, waitTransform] using
    htarget (τ0 - (τ0 - 1) * z) (τ1 - (τ1 - 1) * z) hd0 hd1 rfl rfl

noncomputable def singleEdgeScore {E Ω : Type*} [DecidableEq E]
    (a : E) (ξ : Ω → ℝ) (e : E) (ω : Ω) : ℝ :=
  if e = a then ξ ω else 0

lemma singleEdgeScore_memLp {E Ω : Type*} [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) (a : E) (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ) (e : E) :
    MemLp (singleEdgeScore a ξ e) 2 μ := by
  change MemLp (fun ω => if e = a then ξ ω else 0) 2 μ
  by_cases h : e = a
  · simp [h, hξ]
  · simp [h]

lemma singleEdgeScore_lagCovariance {E Ω : Type*} [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) (S : Ω → Ω) (a : E) (ξ : Ω → ℝ) (e e' : E) (m : ℕ) :
    lagCovariance μ S (singleEdgeScore a ξ) e e' m =
      if e = a ∧ e' = a then ∫ ω, ξ (S^[m] ω) * ξ ω ∂μ else 0 := by
  by_cases h : e = a <;> by_cases h' : e' = a <;>
    simp [singleEdgeScore, lagCovariance, h, h']

lemma singleEdgeScore_varianceLag {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) (K : Chain E) (S : Ω → Ω) (a : E) (ξ : Ω → ℝ) (m : ℕ) :
    varianceLag μ K S (singleEdgeScore a ξ) (m + 1) =
      (∫ ω, ξ (S^[m + 1] ω) * ξ ω ∂μ) * K.initial a * ((K.prob ^ (m + 1)) a a - K.initial a) := by
  simp only [varianceLag_pos, singleEdgeScore_lagCovariance]
  simp [ite_and, ite_mul]

/-- Actual bypass per-step local quadratic cost defined from the limit of finite expected
log-growth second derivatives on the bypass edge chain, matching `ParallelReference.quadraticCost`. -/
noncomputable def bypassLocalCost {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (τ0 τ1 α : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (S : Ω → Ω) (ξ : Ω → ℝ) : ℝ :=
  -(1 / 2 : ℝ) * limUnder atTop (fun N : ℕ =>
    iteratedDeriv 2 (Dobson.FinitePathJet.finiteGrowth μ
      (bypassEdgeChain τ0 τ1 α hτ0 hτ1 hα0 hα1) S
      (singleEdgeScore (false, true) ξ) 0 .lognormal N) 0)

/-- Finite-path bypass cost identity for the explicit reference chain:
for the supplied stationary unit-variance driver with covariance `φ^m`,
`bypassLocalCost = α² * ((1 + 2 * θ) / (2 * T²) - q₀*q₁ / (T*(1-R)))`.
This theorem is uncredited in the version-2 inventory and does not identify
a physical restoring threshold. -/
lemma bypass_localCost_eq {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (S : Ω ≃ᵐ Ω) (hS : MeasurePreserving S μ μ)
    (τ0 τ1 α φ : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (hφ : |φ| < 1)
    (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ)
    (hcov : ∀ m : ℕ, (∫ ω, ξ (S^[m] ω) * ξ ω ∂μ) = φ ^ m) :
    let T := bypassTotalTime τ0 τ1 α
    let θ := φ / (1 - φ)
    let q0 := waitTransform τ0 φ
    let q1 := waitTransform τ1 φ
    let R := bypassCircuitTransform τ0 τ1 α φ
    bypassLocalCost μ τ0 τ1 α hτ0 hτ1 hα0 hα1 S ξ =
      α ^ 2 * ((1 + 2 * θ) / (2 * T ^ 2) - q0 * q1 / (T * (1 - R))) := by
  intro T θ q0 q1 R
  let K := bypassEdgeChain τ0 τ1 α hτ0 hτ1 hα0 hα1
  let a : Bool × Bool := (false, true)
  have hlag_bound : ∀ m : ℕ,
      |varianceLag μ K S (singleEdgeScore a ξ) (m + 1)| ≤ 1 * |φ| ^ m := by
    intro m
    rw [singleEdgeScore_varianceLag, hcov (m + 1), abs_mul, abs_mul, abs_pow,
      abs_of_nonneg (K.initial_nonneg a), pow_succ]
    have hnn1 : 0 ≤ |φ| ^ m * |φ| * K.initial a :=
      mul_nonneg (mul_nonneg (pow_nonneg (abs_nonneg φ) m) (abs_nonneg φ)) (K.initial_nonneg a)
    have hnn2 : 0 ≤ |φ| ^ m * |φ| :=
      mul_nonneg (pow_nonneg (abs_nonneg φ) m) (abs_nonneg φ)
    calc |φ| ^ m * |φ| * K.initial a * |(K.prob ^ (m + 1)) a a - K.initial a|
        ≤ |φ| ^ m * |φ| * K.initial a :=
          mul_le_of_le_one_right hnn1 (chain_pow_sub_initial_abs_le_one K _ _ _)
      _ ≤ |φ| ^ m * |φ| :=
          mul_le_of_le_one_right hnn2 (chain_initial_le_one K a)
      _ ≤ 1 * |φ| ^ m := by
          rw [one_mul]
          exact mul_le_of_le_one_right (pow_nonneg (abs_nonneg φ) m) hφ.le
  have hcesaro := (Dobson.Coverage1Lineage.tendsto_cesaro_lag_sum
    (fun m => varianceLag μ K S (singleEdgeScore a ξ) (m + 1))
    1 |φ| zero_le_one (abs_nonneg φ) hφ hlag_bound).2.comp (tendsto_add_atTop_nat 1)
  have hfin : ∀ N : ℕ,
      Dobson.FinitePathCoefficient.finiteCoefficient μ K S (singleEdgeScore a ξ) N =
        (∑ e, ∑ e', K.initial e * K.initial e' *
          lagCovariance μ S (singleEdgeScore a ξ) e e' 0) / 2 -
        ∑ k ∈ Finset.range N, (1 - ((k + 1 : ℕ) : ℝ) / (N + 1 : ℕ)) *
          varianceLag μ K S (singleEdgeScore a ξ) (k + 1) := by
    intro N
    have hh := withinVariance_div μ K S hS (singleEdgeScore a ξ)
      (singleEdgeScore_memLp μ a ξ hξ) (N + 1) (by omega)
    simp only [Nat.add_sub_cancel, ← varianceLag_pos] at hh
    unfold Dobson.FinitePathCoefficient.finiteCoefficient
    rw [show withinVariance μ K S (singleEdgeScore a ξ) (N + 1) / (2 * (N + 1 : ℕ)) =
      (withinVariance μ K S (singleEdgeScore a ξ) (N + 1) / (N + 1 : ℕ)) / 2 by ring, hh]
    ring
  have hcoef_tendsto : Tendsto
      (Dobson.FinitePathCoefficient.finiteCoefficient μ K S (singleEdgeScore a ξ))
      atTop (𝓝 (Dobson.FinitePathCoefficient.covarianceCoefficient μ K S
        (singleEdgeScore a ξ))) := by
    have ht := (tendsto_const_nhds (x :=
      (∑ e, ∑ e', K.initial e * K.initial e' *
        lagCovariance μ S (singleEdgeScore a ξ) e e' 0) / 2)).sub hcesaro
    convert ht using 1
    · funext N
      simp only [Function.comp_def, Nat.add_sub_cancel]
      exact hfin N
    · simp only [Dobson.FinitePathCoefficient.covarianceCoefficient, varianceLag_pos]
      ring
  have hderiv_lim : Tendsto (fun N : ℕ =>
      iteratedDeriv 2 (Dobson.FinitePathJet.finiteGrowth μ K S
        (singleEdgeScore a ξ) 0 .lognormal N) 0)
      atTop (𝓝 (-2 * Dobson.FinitePathCoefficient.covarianceCoefficient μ K S
        (singleEdgeScore a ξ))) := by
    simpa only [Dobson.FinitePathJet.finiteGrowth_second μ K S hS (singleEdgeScore a ξ)
      (singleEdgeScore_memLp μ a ξ hξ) 0 .lognormal (by intro h; cases h)] using
      hcoef_tendsto.const_mul (-2)
  have hlocal : bypassLocalCost μ τ0 τ1 α hτ0 hτ1 hα0 hα1 S ξ =
      Dobson.FinitePathCoefficient.covarianceCoefficient μ K S (singleEdgeScore a ξ) := by
    unfold bypassLocalCost
    change -(1 / 2 : ℝ) * limUnder atTop (fun N : ℕ =>
      iteratedDeriv 2 (Dobson.FinitePathJet.finiteGrowth μ K S
        (singleEdgeScore a ξ) 0 .lognormal N) 0) = _
    rw [hderiv_lim.limUnder_eq]
    ring
  rw [hlocal]
  have hinst : (∑ e, ∑ e', K.initial e * K.initial e' *
      lagCovariance μ S (singleEdgeScore a ξ) e e' 0) = K.initial a ^ 2 := by
    simp only [singleEdgeScore_lagCovariance, hcov 0, pow_zero]
    simp [ite_and, mul_ite, pow_two]
  have htsum_eq : (∑' m : ℕ, ∑ e, ∑ e',
      lagCovariance μ S (singleEdgeScore a ξ) e e' (m + 1) * K.initial e' *
        ((K.prob ^ (m + 1)) e' e - K.initial e)) =
      K.initial a * (realVisits K a φ a - K.initial a * θ) := by
    simp_rw [← varianceLag_pos, singleEdgeScore_varianceLag, hcov]
    have hs1 : Summable (fun m : ℕ => φ ^ (m + 1) * (K.prob ^ (m + 1)) a a) :=
      realVisits_summable K φ hφ a a
    have hs2 : Summable (fun m : ℕ => φ ^ (m + 1)) := by
      simpa only [pow_succ] using (summable_geometric_of_abs_lt_one hφ).mul_right φ
    have hgeom : (∑' m : ℕ, φ ^ (m + 1)) = θ := by
      simp_rw [pow_succ]
      rw [tsum_mul_right, tsum_geometric_of_abs_lt_one hφ]
      dsimp [θ]
      ring
    have hterm : (fun m : ℕ => φ ^ (m + 1) * K.initial a *
        ((K.prob ^ (m + 1)) a a - K.initial a)) =
        fun m : ℕ => K.initial a *
          (φ ^ (m + 1) * (K.prob ^ (m + 1)) a a - (φ ^ (m + 1)) * K.initial a) := by
      funext m; ring
    rw [hterm, tsum_mul_left, Summable.tsum_sub hs1 (hs2.mul_right (K.initial a)),
      tsum_mul_right, hgeom]
    dsimp [realVisits]
    ring
  unfold Dobson.FinitePathCoefficient.covarianceCoefficient
  rw [hinst, htsum_eq]
  have hinit_a : K.initial a = α / T :=
    bypass_entrance_frequency τ0 τ1 α hτ0 hτ1 hα0 hα1
  have hret : realVisits K a φ a = α * q0 * q1 / (1 - R) :=
    bypass_return_formula τ0 τ1 α hτ0 hτ1 hα0 hα1 φ hφ
  have hT_ne : T ≠ 0 := (bypassTotalTime_pos τ0 τ1 α hτ0 hτ1 hα0).ne'
  have hR_ne : 1 - R ≠ 0 := by
    have hh := (abs_lt.mp (bypassCircuitTransform_abs_lt τ0 τ1 α φ hτ0 hτ1 hα0 hα1 hφ)).2
    dsimp [R]
    linarith
  rw [hinit_a, hret]
  have halg : ∀ X Y Z W U : ℝ, X ≠ 0 → 1 - W ≠ 0 →
      (1 / 2 : ℝ) * (Y / X) ^ 2 - Y / X * (Y * Z * U / (1 - W) - Y / X * θ) =
        Y ^ 2 * ((1 + 2 * θ) / (2 * X ^ 2) - Z * U / (X * (1 - W))) := by
    intro X Y Z W U hX hW
    field_simp [hX, hW]
    ring
  exact halg T α q0 R q1 hT_ne hR_ne

/-- Generation normalization of the explicit reference-path bypass cost:
`T * C = α² * ((1 + 2 * θ) / (2 * T) - q₀*q₁ / (1-R))`.
The physical growth/threshold identification is not a conclusion. -/
theorem bypass_coefficient_unconditional {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (S : Ω ≃ᵐ Ω) (hS : MeasurePreserving S μ μ)
    (τ0 τ1 α φ : ℝ) (hτ0 : 1 ≤ τ0) (hτ1 : 1 ≤ τ1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (hφ : |φ| < 1)
    (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ)
    (hcov : ∀ m : ℕ, (∫ ω, ξ (S^[m] ω) * ξ ω ∂μ) = φ ^ m) :
    let T := bypassTotalTime τ0 τ1 α
    let θ := φ / (1 - φ)
    let q0 := waitTransform τ0 φ
    let q1 := waitTransform τ1 φ
    let R := bypassCircuitTransform τ0 τ1 α φ
    let C := bypassLocalCost μ τ0 τ1 α hτ0 hτ1 hα0 hα1 S ξ
    T * C = α ^ 2 * ((1 + 2 * θ) / (2 * T) - q0 * q1 / (1 - R)) := by
  intro T θ q0 q1 R C
  exact Dobson.Coverage3Networks.bypass_coefficient T α θ q0 q1 R C
    (bypassTotalTime_pos τ0 τ1 α hτ0 hτ1 hα0).ne'
    (bypass_localCost_eq μ S hS τ0 τ1 α φ hτ0 hτ1 hα0 hα1 hφ ξ hξ hcov)

end BypassResolvent

end Dobson.P1bB_MechRoutes

#print axioms Dobson.P1bB_MechRoutes.parallel_threshold_from_balance
#print axioms Dobson.P1bB_MechRoutes.parallel_threshold_unconditional
#print axioms Dobson.P1bB_MechRoutes.bypass_localCost_eq
#print axioms Dobson.P1bB_MechRoutes.bypass_coefficient_unconditional
