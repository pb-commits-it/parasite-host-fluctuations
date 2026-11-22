import DobsonProofs.Coverage6Remainders
import Mathlib.Analysis.Calculus.ContDiff.Deriv

/-!
# Part B (`lem:mech-bounded`): projective fourth-derivative bounds and cubic/quartic remainders

Closes the three derivative-bound steps in the additive reference-path core:
- `part1:81` (`Coverage6Remainders.growth_limit_cubic.hM`),
- `part1:82` (`Coverage6Remainders.growth_limit_cubic.hthird`), and
- `part1:69` (`Coverage6Remainders.growth_limit_quartic.hM`, bounded-additive case),
for the actual stationary path-partition growth `FinitePathJet.finiteGrowth μ K T ξ L .additive N`:
1. Constructs the adapted `ℓ¹` contraction gauge `gauge K m` on the zero-sum subspace from the
   primitive geometric-mixing hypothesis `|(K.prob^(m+1)) e' e - K.initial e| ≤ Ku * β^m`
   (`0 ≤ β < 1`).
2. Identifies the pathwise bounded-additive partition `expect K (N + 1) (∏_{t ≤ N} (1 + x ξ_t))`
   with the product of normalized projective step gains `∏_{t ≤ N} projGain_t(x)`, so its
   logarithm telescopes into the sum of smooth one-step log gains `∑_{t ≤ N} log projGain_t(x)`.
3. Differentiates the normalized projective recursion through fourth order on a uniform open noise
   interval `(-δ₀, δ₀)` and proves horizon- and path-independent bounds on all four derivatives
   via projective `gauge` contraction (`projective_logGain_bounds`).
4. Applies four dominated-differentiation integral exchanges and `continuousOn_of_dominated` to
   prove `ContDiffOn ℝ 4`, vanishing first derivative at `0` from centeredness, uniform third-
   and fourth-derivative bounds across all `N : ℕ`, and whole-process sign-symmetry evenness.
5. Instantiates both `Coverage6Remainders.growth_limit_cubic` (`mech_bounded_cubic_unconditional`)
   and `Coverage6Remainders.growth_limit_quartic` (`mech_bounded_quartic_unconditional`) without
   any owed derivative-bound hypotheses.

Coverage correction, 2026-11-22: `lem:mech-bounded` is core. The final
corollaries assume convergence of the finite-path growth to the supplied `r`;
general-matrix reduction and physical exponent identification remain outside credit.
-/

namespace Dobson.P1bB_MechBounded

open Set Filter MeasureTheory ProbabilityTheory
open scoped BigOperators Topology
open Dobson.FiniteLineageVariance Dobson.FinitePathJet Dobson.Coverage6Remainders

/-! ## Section 1: Adapted `ℓ¹` contraction norm on zero-sum vectors -/

section AdaptedGauge

noncomputable def l1 {E : Type*} [Fintype E] (z : E → ℝ) : ℝ := ∑ i, |z i|

noncomputable def gauge {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (z : E → ℝ) : ℝ :=
  ∑ k ∈ Finset.range m, l1 ((K.prob ^ k).vecMul z)

lemma l1_nonneg {E : Type*} [Fintype E] (z : E → ℝ) : 0 ≤ l1 z :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

lemma l1_zero {E : Type*} [Fintype E] : l1 (fun _ : E => (0 : ℝ)) = 0 := by
  simp [l1]

lemma l1_add_le {E : Type*} [Fintype E] (z w : E → ℝ) :
    l1 (fun i => z i + w i) ≤ l1 z + l1 w := by
  unfold l1
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun i _ => abs_add_le (z i) (w i)

lemma l1_sub_le {E : Type*} [Fintype E] (z w : E → ℝ) :
    l1 (fun i => z i - w i) ≤ l1 z + l1 w := by
  unfold l1
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun i _ => by
    simpa only [sub_zero, zero_sub, abs_neg] using abs_sub_le (z i) 0 (w i)

lemma l1_smul {E : Type*} [Fintype E] (c : ℝ) (z : E → ℝ) :
    l1 (fun i => c * z i) = |c| * l1 z := by
  unfold l1
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => abs_mul c (z i)

lemma l1_div_le {E : Type*} [Fintype E] (z : E → ℝ) {G : ℝ} (hG : 1 / 2 ≤ G) :
    l1 (fun i => z i / G) ≤ 2 * l1 z := by
  have hG0 : 0 < G := by linarith
  unfold l1
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  rw [abs_div, abs_of_pos hG0, div_le_iff₀ hG0]
  nlinarith [abs_nonneg (z i)]

lemma abs_dot_le_mul_l1 {E : Type*} [Fintype E] (b z : E → ℝ) (A : ℝ)
    (hb : ∀ i, |b i| ≤ A) :
    |∑ i, b i * z i| ≤ A * l1 z := by
  calc
    |∑ i, b i * z i| ≤ ∑ i, |b i * z i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, A * |z i| := Finset.sum_le_sum fun i _ => by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right (hb i) (abs_nonneg _)
    _ = A * l1 z := by rw [← Finset.mul_sum]; rfl

lemma l1_step {E : Type*} [Fintype E] (K : Chain E) (z : E → ℝ) :
    l1 (K.prob.vecMul z) ≤ l1 z := by
  unfold l1
  calc
    ∑ j, |∑ i, z i * K.prob i j| ≤ ∑ j, ∑ i, |z i| * K.prob i j := by
      apply Finset.sum_le_sum; intro j _
      calc
        |∑ i, z i * K.prob i j| ≤ ∑ i, |z i * K.prob i j| := Finset.abs_sum_le_sum_abs _ _
        _ = _ := by simp only [abs_mul, abs_of_nonneg (K.nonneg _ _)]
    _ = ∑ i, |z i| := by rw [Finset.sum_comm]; simp [← Finset.mul_sum, K.total]

lemma l1_power {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (z : E → ℝ) (n : ℕ) : l1 ((K.prob ^ n).vecMul z) ≤ l1 z := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [pow_succ, ← Matrix.vecMul_vecMul]
    exact (l1_step K _).trans ih

lemma gauge_zero {E : Type*} [Fintype E] [DecidableEq E] (K : Chain E) (m : ℕ) :
    gauge K m (fun _ => 0) = 0 := by
  unfold gauge
  apply Finset.sum_eq_zero
  intro k _
  have h0 : ((K.prob ^ k).vecMul fun _ : E => (0 : ℝ)) = fun _ => 0 := by
    funext j; simp [Matrix.vecMul, dotProduct]
  rw [h0, l1_zero]

lemma gauge_add_le {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (z w : E → ℝ) :
    gauge K m (fun i => z i + w i) ≤ gauge K m z + gauge K m w := by
  unfold gauge
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro k _
  have hadd : ((K.prob ^ k).vecMul fun i => z i + w i) =
      fun j => (K.prob ^ k).vecMul z j + (K.prob ^ k).vecMul w j := by
    funext j
    simp [Matrix.vecMul, dotProduct, add_mul, Finset.sum_add_distrib]
  rw [hadd]
  exact l1_add_le _ _

lemma zero_sum_decay {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (z : E → ℝ) (hz : ∑ i, z i = 0)
    (Ku β : ℝ) (n : ℕ)
    (hmix : ∀ e e', |(K.prob ^ (n + 1)) e' e - K.initial e| ≤ Ku * β ^ n) :
    l1 ((K.prob ^ (n + 1)).vecMul z) ≤ (Fintype.card E : ℝ) * Ku * β ^ n * l1 z := by
  have he (j : E) : (K.prob ^ (n + 1)).vecMul z j =
      ∑ i, z i * ((K.prob ^ (n + 1)) i j - K.initial j) := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hz, zero_mul, sub_zero,
      Matrix.vecMul, dotProduct]
  unfold l1
  simp_rw [he]
  calc
    ∑ j, |∑ i, z i * ((K.prob ^ (n + 1)) i j - K.initial j)| ≤
        ∑ _j : E, ∑ i, |z i| * (Ku * β ^ n) := by
      apply Finset.sum_le_sum; intro j _
      apply (Finset.abs_sum_le_sum_abs _ _).trans
      apply Finset.sum_le_sum; intro i _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (hmix j i) (abs_nonneg _)
    _ = _ := by simp [← Finset.sum_mul]; ring

lemma exists_contraction_block {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (Ku β : ℝ) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ e e' n, |(K.prob ^ (n + 1)) e' e - K.initial e| ≤ Ku * β ^ n) :
    ∃ m : ℕ, 0 < m ∧ ∀ z : E → ℝ, (∑ i, z i) = 0 →
      l1 ((K.prob ^ m).vecMul z) ≤ l1 z / 2 := by
  have ht := (tendsto_pow_atTop_nhds_zero_of_lt_one hβ0 hβ1).const_mul
    ((Fintype.card E : ℝ) * Ku)
  simp only [mul_zero] at ht
  obtain ⟨n, hn⟩ := (ht.eventually (gt_mem_nhds (show (0 : ℝ) < 1 / 2 by norm_num))).exists
  refine ⟨n + 1, by omega, ?_⟩
  intro z hz
  apply (zero_sum_decay K z hz Ku β n (fun e e' => hmix e e' n)).trans
  nlinarith [l1_nonneg z]

lemma gauge_lower {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (hm : 0 < m) (z : E → ℝ) :
    l1 z ≤ gauge K m z := by
  have h := Finset.single_le_sum
    (fun k (_ : k ∈ Finset.range m) => l1_nonneg ((K.prob ^ k).vecMul z))
    (show 0 ∈ Finset.range m by simpa using hm)
  simpa [gauge] using h

lemma gauge_upper {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (z : E → ℝ) :
    gauge K m z ≤ (m : ℝ) * l1 z := by
  calc
    gauge K m z ≤ ∑ _k ∈ Finset.range m, l1 z := Finset.sum_le_sum fun k _ => l1_power K z k
    _ = _ := by simp

lemma gauge_step_le {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (z : E → ℝ) :
    gauge K m (K.prob.vecMul z) ≤ (m : ℝ) * l1 z :=
  (gauge_upper K m (K.prob.vecMul z)).trans
    (mul_le_mul_of_nonneg_left (l1_step K z) (Nat.cast_nonneg m))

lemma gauge_shift {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (z : E → ℝ) :
    gauge K m (K.prob.vecMul z) + l1 z = gauge K m z + l1 ((K.prob ^ m).vecMul z) := by
  have h := Finset.sum_range_succ' (fun k => l1 ((K.prob ^ k).vecMul z)) m
  rw [Finset.sum_range_succ] at h
  simp only [pow_zero, Matrix.vecMul_one] at h
  unfold gauge
  rw [h]
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  rw [pow_succ', ← Matrix.vecMul_vecMul]

lemma gauge_contraction {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (hm : 0 < m)
    (hblock : ∀ z : E → ℝ, (∑ i, z i) = 0 → l1 ((K.prob ^ m).vecMul z) ≤ l1 z / 2)
    (z : E → ℝ) (hz : ∑ i, z i = 0) :
    gauge K m (K.prob.vecMul z) ≤ (1 - 1 / (2 * (m : ℝ))) * gauge K m z := by
  have hshift := gauge_shift K m z
  have hb := hblock z hz
  have hu := gauge_upper K m z
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hmul : (m : ℝ) * gauge K m (K.prob.vecMul z) ≤
      (m : ℝ) * ((1 - 1 / (2 * (m : ℝ))) * gauge K m z) := by
    have hrhs : (m : ℝ) * ((1 - 1 / (2 * (m : ℝ))) * gauge K m z) =
        ((m : ℝ) - 1 / 2) * gauge K m z := by
      field_simp [hmR.ne']
    rw [hrhs]
    nlinarith
  exact le_of_mul_le_mul_left hmul hmR

end AdaptedGauge

/-! ## Section 2: Normalized projective step recursion and path-partition telescoping -/

section ProjectiveTelescoping

noncomputable def rawState {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (u0 : E → ℝ) : ℕ → ℝ → E → ℝ
  | 0, _ => u0
  | t + 1, x => K.prob.vecMul (fun e => rawState K a u0 t x e * (1 + x * a t e))

lemma rawState_shift {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (u0 : E → ℝ) (n : ℕ) (x : ℝ) :
    rawState K (fun t => a (t + 1)) (K.prob.vecMul (fun i => u0 i * (1 + x * a 0 i))) n x =
      rawState K a u0 (n + 1) x := by
  induction n generalizing a u0 with
  | zero => rfl
  | succ n ih =>
    have hstep := ih a u0
    simp only [rawState, hstep]

lemma genExpect_prod_eq_sum_rawState {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : ℕ → E → ℝ) (u0 : E → ℝ) (n : ℕ) (x : ℝ) :
    (∑ i, u0 i * pathExpect K n i
        (fun p => ∏ t ∈ Finset.range n, (1 + x * a t (p t)))) =
      ∑ e, rawState K a u0 n x e := by
  induction n generalizing a u0 with
  | zero =>
    simp [pathExpect, rawState]
  | succ n ih =>
    have hpeel (i : E) (j : E) :
        pathExpect K n j (fun p => ∏ t ∈ Finset.range (n + 1), (1 + x * a t (prepend i p t))) =
          (1 + x * a 0 i) * pathExpect K n j
            (fun p => ∏ t ∈ Finset.range n, (1 + x * a (t + 1) (p t))) := by
      have hfn : (fun p : ℕ → E => ∏ t ∈ Finset.range (n + 1), (1 + x * a t (prepend i p t))) =
          fun p => (1 + x * a 0 i) * ∏ t ∈ Finset.range n, (1 + x * a (t + 1) (p t)) := by
        funext p
        rw [Finset.prod_range_succ']
        simp [prepend, mul_comm]
      rw [hfn, pathExpect_const_mul]
    simp only [pathExpect, hpeel]
    have hswap : (∑ i, u0 i * ∑ j, K.prob i j * ((1 + x * a 0 i) *
        pathExpect K n j (fun p => ∏ t ∈ Finset.range n, (1 + x * a (t + 1) (p t))))) =
        ∑ j, (K.prob.vecMul (fun i => u0 i * (1 + x * a 0 i)) j) *
          pathExpect K n j (fun p => ∏ t ∈ Finset.range n, (1 + x * a (t + 1) (p t))) := by
      simp only [Matrix.vecMul, dotProduct, Finset.sum_mul, Finset.mul_sum]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl; intro j _
      apply Finset.sum_congr rfl; intro i _; ring
    rw [hswap, ih (fun t => a (t + 1)) (K.prob.vecMul (fun i => u0 i * (1 + x * a 0 i))),
      rawState_shift]

noncomputable def projState {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) : ℕ → ℝ → E → ℝ
  | 0, _ => K.initial
  | t + 1, x =>
      let G := ∑ e, projState K a t x e * (1 + x * a t e)
      K.prob.vecMul (fun e => projState K a t x e * (1 + x * a t e) / G)

noncomputable def projGain {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (t : ℕ) (x : ℝ) : ℝ :=
  ∑ e, projState K a t x e * (1 + x * a t e)

lemma projState_prob_and_gain_bounds {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (A : ℝ)
    (hA : 0 < A) (ha : ∀ t e, |a t e| ≤ A) (x : ℝ)
    (hx : x ∈ Ioo (-(1 / (2 * A))) (1 / (2 * A))) (t : ℕ) :
    (∀ e, 0 ≤ projState K a t x e) ∧
    (∑ e, projState K a t x e = 1) ∧
    (1 / 2 < projGain K a t x ∧ projGain K a t x < 2) := by
  have hprob : ∀ s : ℕ, (∀ e, 0 ≤ projState K a s x e) ∧ (∑ e, projState K a s x e = 1) := by
    intro s
    induction s with
    | zero => exact ⟨K.initial_nonneg, K.initial_total⟩
    | succ s ih =>
      obtain ⟨hnn, hsum⟩ := ih
      have hfac (e : E) := additive_factor_bounds (a s e) x A hA (ha s e) hx
      have hG_lo : 1 / 2 ≤ ∑ e, projState K a s x e * (1 + x * a s e) := by
        calc
          (1 / 2 : ℝ) = ∑ e, projState K a s x e * (1 / 2) := by
            rw [← Finset.sum_mul, hsum, one_mul]
          _ ≤ ∑ e, projState K a s x e * (1 + x * a s e) :=
            Finset.sum_le_sum fun e _ => mul_le_mul_of_nonneg_left (hfac e).1.le (hnn e)
      have hG_pos : 0 < ∑ e, projState K a s x e * (1 + x * a s e) := by linarith
      refine ⟨?_, ?_⟩
      · intro j
        simp only [projState, Matrix.vecMul, dotProduct]
        exact Finset.sum_nonneg fun i _ => mul_nonneg
          (div_nonneg (mul_nonneg (hnn i) (by linarith [(hfac i).1])) hG_pos.le)
          (K.nonneg i j)
      · simp only [projState, Matrix.vecMul, dotProduct]
        rw [Finset.sum_comm]
        simp only [← Finset.mul_sum, K.total, mul_one, ← Finset.sum_div]
        exact div_self hG_pos.ne'
  obtain ⟨hnn, hsum⟩ := hprob t
  have hfac (e : E) := additive_factor_bounds (a t e) x A hA (ha t e) hx
  refine ⟨hnn, hsum, ?_, ?_⟩
  · have hex : ∃ e, 0 < projState K a t x e := by
      by_contra hc; push Not at hc
      have := Finset.sum_nonpos (s := Finset.univ) fun e _ => hc e
      rw [hsum] at this; norm_num at this
    obtain ⟨e0, he0⟩ := hex
    unfold projGain
    have hstrict : (∑ e, projState K a t x e * (1 / 2)) <
        ∑ e, projState K a t x e * (1 + x * a t e) :=
      Finset.sum_lt_sum (fun e _ => mul_le_mul_of_nonneg_left (hfac e).1.le (hnn e))
        ⟨e0, Finset.mem_univ e0, mul_lt_mul_of_pos_left (hfac e0).1 he0⟩
    have hlhs : (∑ e, projState K a t x e * (1 / 2)) = 1 / 2 := by
      rw [← Finset.sum_mul, hsum, one_mul]
    linarith
  · have hex : ∃ e, 0 < projState K a t x e := by
      by_contra hc; push Not at hc
      have := Finset.sum_nonpos (s := Finset.univ) fun e _ => hc e
      rw [hsum] at this; norm_num at this
    obtain ⟨e0, he0⟩ := hex
    unfold projGain
    have hstrict : (∑ e, projState K a t x e * (1 + x * a t e)) <
        ∑ e, projState K a t x e * 2 :=
      Finset.sum_lt_sum (fun e _ => mul_le_mul_of_nonneg_left (hfac e).2.le (hnn e))
        ⟨e0, Finset.mem_univ e0, mul_lt_mul_of_pos_left (hfac e0).2 he0⟩
    have hrhs : (∑ e, projState K a t x e * 2) = 2 := by
      rw [← Finset.sum_mul, hsum, one_mul]
    linarith

lemma rawState_eq_prod_projGain {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (A : ℝ)
    (hA : 0 < A) (ha : ∀ t e, |a t e| ≤ A) (x : ℝ)
    (hx : x ∈ Ioo (-(1 / (2 * A))) (1 / (2 * A))) (n : ℕ) :
    rawState K a K.initial n x =
      fun e => (∏ t ∈ Finset.range n, projGain K a t x) * projState K a n x e := by
  induction n with
  | zero =>
    funext e; simp [rawState, projState]
  | succ n ih =>
    have hG_pos : 0 < projGain K a n x := by
      linarith [(projState_prob_and_gain_bounds K a A hA ha x hx n).2.2.1]
    funext j
    simp only [rawState, ih, projState, Matrix.vecMul, dotProduct, Finset.prod_range_succ,
      Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    change (∏ t ∈ Finset.range n, projGain K a t x) * projState K a n x i * (1 + x * a n i) *
        K.prob i j =
      (∏ t ∈ Finset.range n, projGain K a t x) * projGain K a n x *
        (projState K a n x i * (1 + x * a n i) / projGain K a n x * K.prob i j)
    field_simp [hG_pos.ne']

/-- Exact path-partition telescoping into normalized step gains on `(-1/(2A), 1/(2A))`. -/
theorem log_expect_prod_eq_sum_log_projGain {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (a : ℕ → E → ℝ) (A : ℝ)
    (hA : 0 < A) (ha : ∀ t e, |a t e| ≤ A) (x : ℝ)
    (hx : x ∈ Ioo (-(1 / (2 * A))) (1 / (2 * A))) (n : ℕ) :
    Real.log (Dobson.FiniteLineageVariance.expect K n
        (fun p => ∏ t ∈ Finset.range n, (1 + x * a t (p t)))) =
      ∑ t ∈ Finset.range n, Real.log (projGain K a t x) := by
  have hexp : Dobson.FiniteLineageVariance.expect K n
      (fun p => ∏ t ∈ Finset.range n, (1 + x * a t (p t))) =
      ∏ t ∈ Finset.range n, projGain K a t x := by
    unfold Dobson.FiniteLineageVariance.expect
    rw [genExpect_prod_eq_sum_rawState, rawState_eq_prod_projGain K a A hA ha x hx n,
      ← Finset.mul_sum, (projState_prob_and_gain_bounds K a A hA ha x hx n).2.1, mul_one]
  rw [hexp, Real.log_prod]
  intro t _
  have hG := (projState_prob_and_gain_bounds K a A hA ha x hx t).2.2.1
  linarith

end ProjectiveTelescoping

/-! ## Section 3: Fourth-order quotient calculus and projective contraction bounds -/

section ProjectiveDerivatives

noncomputable def q0 (W G : ℝ → ℝ) (x : ℝ) : ℝ := W x / G x
noncomputable def q1 (W W1 G G1 : ℝ → ℝ) (x : ℝ) : ℝ :=
  (W1 x - G1 x * q0 W G x) / G x
noncomputable def q2 (W W1 W2 G G1 G2 : ℝ → ℝ) (x : ℝ) : ℝ :=
  (W2 x - 2 * G1 x * q1 W W1 G G1 x - G2 x * q0 W G x) / G x
noncomputable def q3 (W W1 W2 W3 G G1 G2 G3 : ℝ → ℝ) (x : ℝ) : ℝ :=
  (W3 x - 3 * G1 x * q2 W W1 W2 G G1 G2 x - 3 * G2 x * q1 W W1 G G1 x -
    G3 x * q0 W G x) / G x
noncomputable def q4 (W W1 W2 W3 W4 G G1 G2 G3 G4 : ℝ → ℝ) (x : ℝ) : ℝ :=
  (W4 x - 4 * G1 x * q3 W W1 W2 W3 G G1 G2 G3 x -
    6 * G2 x * q2 W W1 W2 G G1 G2 x - 4 * G3 x * q1 W W1 G G1 x -
    G4 x * q0 W G x) / G x

lemma hasDerivAt_q0 {W W1 G G1 : ℝ → ℝ} {x : ℝ} (hG : G x ≠ 0)
    (hW0 : HasDerivAt W (W1 x) x) (hG0 : HasDerivAt G (G1 x) x) :
    HasDerivAt (q0 W G) (q1 W W1 G G1 x) x := by
  have h0eq : W x = q0 W G x * G x := by unfold q0; field_simp [hG]
  convert hW0.div hG0 hG using 1
  · rfl
  · unfold q1; rw [h0eq]; field_simp [hG]

lemma hasDerivAt_q1 {W W1 W2 G G1 G2 : ℝ → ℝ} {x : ℝ} (hG : G x ≠ 0)
    (hW0 : HasDerivAt W (W1 x) x) (hW1 : HasDerivAt W1 (W2 x) x)
    (hG0 : HasDerivAt G (G1 x) x) (hG1 : HasDerivAt G1 (G2 x) x) :
    HasDerivAt (q1 W W1 G G1) (q2 W W1 W2 G G1 G2 x) x := by
  have hd0 := hasDerivAt_q0 hG hW0 hG0
  have hnum := hW1.sub (hG1.mul hd0)
  have h1eq : W1 x - G1 x * q0 W G x = q1 W W1 G G1 x * G x := by
    unfold q1; field_simp [hG]
  convert hnum.div hG0 hG using 1
  · rfl
  · simp only [Pi.sub_apply, Pi.mul_apply]
    unfold q2; rw [h1eq]; field_simp [hG]; ring

lemma hasDerivAt_q2 {W W1 W2 W3 G G1 G2 G3 : ℝ → ℝ} {x : ℝ} (hG : G x ≠ 0)
    (hW0 : HasDerivAt W (W1 x) x) (hW1 : HasDerivAt W1 (W2 x) x)
    (hW2 : HasDerivAt W2 (W3 x) x)
    (hG0 : HasDerivAt G (G1 x) x) (hG1 : HasDerivAt G1 (G2 x) x)
    (hG2 : HasDerivAt G2 (G3 x) x) :
    HasDerivAt (q2 W W1 W2 G G1 G2) (q3 W W1 W2 W3 G G1 G2 G3 x) x := by
  have hd0 := hasDerivAt_q0 hG hW0 hG0
  have hd1 := hasDerivAt_q1 hG hW0 hW1 hG0 hG1
  have hnum := (hW2.sub ((hG1.const_mul 2).mul hd1)).sub (hG2.mul hd0)
  have h2eq : W2 x - 2 * G1 x * q1 W W1 G G1 x - G2 x * q0 W G x =
      q2 W W1 W2 G G1 G2 x * G x := by
    unfold q2; field_simp [hG]
  convert hnum.div hG0 hG using 1
  · rfl
  · simp only [Pi.sub_apply, Pi.mul_apply]
    unfold q3; rw [h2eq]; field_simp [hG]; ring

lemma hasDerivAt_q3 {W W1 W2 W3 W4 G G1 G2 G3 G4 : ℝ → ℝ} {x : ℝ} (hG : G x ≠ 0)
    (hW0 : HasDerivAt W (W1 x) x) (hW1 : HasDerivAt W1 (W2 x) x)
    (hW2 : HasDerivAt W2 (W3 x) x) (hW3 : HasDerivAt W3 (W4 x) x)
    (hG0 : HasDerivAt G (G1 x) x) (hG1 : HasDerivAt G1 (G2 x) x)
    (hG2 : HasDerivAt G2 (G3 x) x) (hG3 : HasDerivAt G3 (G4 x) x) :
    HasDerivAt (q3 W W1 W2 W3 G G1 G2 G3) (q4 W W1 W2 W3 W4 G G1 G2 G3 G4 x) x := by
  have hd0 := hasDerivAt_q0 hG hW0 hG0
  have hd1 := hasDerivAt_q1 hG hW0 hW1 hG0 hG1
  have hd2 := hasDerivAt_q2 hG hW0 hW1 hW2 hG0 hG1 hG2
  have hnum := ((hW3.sub ((hG1.const_mul 3).mul hd2)).sub
    ((hG2.const_mul 3).mul hd1)).sub (hG3.mul hd0)
  have h3eq : W3 x - 3 * G1 x * q2 W W1 W2 G G1 G2 x - 3 * G2 x * q1 W W1 G G1 x -
      G3 x * q0 W G x = q3 W W1 W2 W3 G G1 G2 G3 x * G x := by
    unfold q3; field_simp [hG]
  convert hnum.div hG0 hG using 1
  · rfl
  · simp only [Pi.sub_apply, Pi.mul_apply]
    unfold q4; rw [h3eq]; field_simp [hG]; ring

def w0 (b : ℝ) (u0 : ℝ → ℝ) (x : ℝ) : ℝ := u0 x * (1 + x * b)
def w1 (b : ℝ) (u0 u1 : ℝ → ℝ) (x : ℝ) : ℝ := u1 x * (1 + x * b) + b * u0 x
def w2 (b : ℝ) (u1 u2 : ℝ → ℝ) (x : ℝ) : ℝ := u2 x * (1 + x * b) + 2 * b * u1 x
def w3 (b : ℝ) (u2 u3 : ℝ → ℝ) (x : ℝ) : ℝ := u3 x * (1 + x * b) + 3 * b * u2 x
def w4 (b : ℝ) (u3 u4 : ℝ → ℝ) (x : ℝ) : ℝ := u4 x * (1 + x * b) + 4 * b * u3 x

lemma hasDerivAt_w0 (b : ℝ) {u0 u1 : ℝ → ℝ} {x : ℝ} (hu0 : HasDerivAt u0 (u1 x) x) :
    HasDerivAt (w0 b u0) (w1 b u0 u1 x) x := by
  have hlin : HasDerivAt (fun y : ℝ => 1 + y * b) b x := by
    simpa using ((hasDerivAt_id x).mul_const b).const_add 1
  convert hu0.mul hlin using 1
  · rfl
  · unfold w1; ring

lemma hasDerivAt_w1 (b : ℝ) {u0 u1 u2 : ℝ → ℝ} {x : ℝ}
    (hu0 : HasDerivAt u0 (u1 x) x) (hu1 : HasDerivAt u1 (u2 x) x) :
    HasDerivAt (w1 b u0 u1) (w2 b u1 u2 x) x := by
  have hlin : HasDerivAt (fun y : ℝ => 1 + y * b) b x := by
    simpa using ((hasDerivAt_id x).mul_const b).const_add 1
  convert (hu1.mul hlin).add (hu0.const_mul b) using 1
  · rfl
  · unfold w2; ring

lemma hasDerivAt_w2 (b : ℝ) {u1 u2 u3 : ℝ → ℝ} {x : ℝ}
    (hu1 : HasDerivAt u1 (u2 x) x) (hu2 : HasDerivAt u2 (u3 x) x) :
    HasDerivAt (w2 b u1 u2) (w3 b u2 u3 x) x := by
  have hlin : HasDerivAt (fun y : ℝ => 1 + y * b) b x := by
    simpa using ((hasDerivAt_id x).mul_const b).const_add 1
  convert (hu2.mul hlin).add (hu1.const_mul (2 * b)) using 1
  · rfl
  · unfold w3; ring

lemma hasDerivAt_w3 (b : ℝ) {u2 u3 u4 : ℝ → ℝ} {x : ℝ}
    (hu2 : HasDerivAt u2 (u3 x) x) (hu3 : HasDerivAt u3 (u4 x) x) :
    HasDerivAt (w3 b u2 u3) (w4 b u3 u4 x) x := by
  have hlin : HasDerivAt (fun y : ℝ => 1 + y * b) b x := by
    simpa using ((hasDerivAt_id x).mul_const b).const_add 1
  convert (hu3.mul hlin).add (hu2.const_mul (3 * b)) using 1
  · rfl
  · unfold w4; ring

structure StateJet (E : Type*) where
  u0 : ℝ → E → ℝ
  u1 : ℝ → E → ℝ
  u2 : ℝ → E → ℝ
  u3 : ℝ → E → ℝ
  u4 : ℝ → E → ℝ

noncomputable def jetW0 {E : Type*} (b : E → ℝ) (J : StateJet E) (e : E) : ℝ → ℝ :=
  w0 (b e) (fun x => J.u0 x e)
noncomputable def jetW1 {E : Type*} (b : E → ℝ) (J : StateJet E) (e : E) : ℝ → ℝ :=
  w1 (b e) (fun x => J.u0 x e) (fun x => J.u1 x e)
noncomputable def jetW2 {E : Type*} (b : E → ℝ) (J : StateJet E) (e : E) : ℝ → ℝ :=
  w2 (b e) (fun x => J.u1 x e) (fun x => J.u2 x e)
noncomputable def jetW3 {E : Type*} (b : E → ℝ) (J : StateJet E) (e : E) : ℝ → ℝ :=
  w3 (b e) (fun x => J.u2 x e) (fun x => J.u3 x e)
noncomputable def jetW4 {E : Type*} (b : E → ℝ) (J : StateJet E) (e : E) : ℝ → ℝ :=
  w4 (b e) (fun x => J.u3 x e) (fun x => J.u4 x e)

noncomputable def jetG0 {E : Type*} [Fintype E] (b : E → ℝ) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW0 b J e x
noncomputable def jetG1 {E : Type*} [Fintype E] (b : E → ℝ) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW1 b J e x
noncomputable def jetG2 {E : Type*} [Fintype E] (b : E → ℝ) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW2 b J e x
noncomputable def jetG3 {E : Type*} [Fintype E] (b : E → ℝ) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW3 b J e x
noncomputable def jetG4 {E : Type*} [Fintype E] (b : E → ℝ) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW4 b J e x

noncomputable def nextStateJet {E : Type*} [Fintype E]
    (K : Chain E) (b : E → ℝ) (J : StateJet E) : StateJet E where
  u0 := fun x => K.prob.vecMul (fun e =>
    q0 (jetW0 b J e) (jetG0 b J) x)
  u1 := fun x => K.prob.vecMul (fun e =>
    q1 (jetW0 b J e) (jetW1 b J e) (jetG0 b J) (jetG1 b J) x)
  u2 := fun x => K.prob.vecMul (fun e =>
    q2 (jetW0 b J e) (jetW1 b J e) (jetW2 b J e) (jetG0 b J) (jetG1 b J) (jetG2 b J) x)
  u3 := fun x => K.prob.vecMul (fun e =>
    q3 (jetW0 b J e) (jetW1 b J e) (jetW2 b J e) (jetW3 b J e)
      (jetG0 b J) (jetG1 b J) (jetG2 b J) (jetG3 b J) x)
  u4 := fun x => K.prob.vecMul (fun e =>
    q4 (jetW0 b J e) (jetW1 b J e) (jetW2 b J e) (jetW3 b J e) (jetW4 b J e)
      (jetG0 b J) (jetG1 b J) (jetG2 b J) (jetG3 b J) (jetG4 b J) x)

noncomputable def stateJetAt {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) : ℕ → StateJet E
  | 0 => {
      u0 := fun _ => K.initial
      u1 := fun _ _ => 0
      u2 := fun _ _ => 0
      u3 := fun _ _ => 0
      u4 := fun _ _ => 0 }
  | t + 1 => nextStateJet K (a t) (stateJetAt K a t)

lemma stateJetAt_u0_eq_projState {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (t : ℕ) (x : ℝ) :
    (stateJetAt K a t).u0 x = projState K a t x := by
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [stateJetAt, nextStateJet, projState, q0, jetW0, jetG0, w0, ih]

lemma jetG0_eq_projGain {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (t : ℕ) (x : ℝ) :
    jetG0 (a t) (stateJetAt K a t) x = projGain K a t x := by
  simp only [jetG0, jetW0, w0, stateJetAt_u0_eq_projState, projGain]

lemma stateJetAt_hasDerivAt {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (A : ℝ)
    (hA : 0 < A) (ha : ∀ t e, |a t e| ≤ A) (t : ℕ) (x : ℝ)
    (hx : x ∈ Ioo (-(1 / (2 * A))) (1 / (2 * A))) :
    let J := stateJetAt K a t
    (∀ e, HasDerivAt (fun y => J.u0 y e) (J.u1 x e) x) ∧
    (∀ e, HasDerivAt (fun y => J.u1 y e) (J.u2 x e) x) ∧
    (∀ e, HasDerivAt (fun y => J.u2 y e) (J.u3 x e) x) ∧
    (∀ e, HasDerivAt (fun y => J.u3 y e) (J.u4 x e) x) := by
  induction t with
  | zero =>
    refine ⟨fun e => hasDerivAt_const x (K.initial e),
      fun _ => hasDerivAt_const x 0,
      fun _ => hasDerivAt_const x 0,
      fun _ => hasDerivAt_const x 0⟩
  | succ t ih =>
    obtain ⟨hd0, hd1, hd2, hd3⟩ := ih
    let J := stateJetAt K a t
    have hG_pos : 0 < jetG0 (a t) J x := by
      rw [jetG0_eq_projGain]
      linarith [(projState_prob_and_gain_bounds K a A hA ha x hx t).2.2.1]
    have hG_ne : jetG0 (a t) J x ≠ 0 := hG_pos.ne'
    have hW0 (e : E) : HasDerivAt (jetW0 (a t) J e) (jetW1 (a t) J e x) x :=
      hasDerivAt_w0 (a t e) (hd0 e)
    have hW1 (e : E) : HasDerivAt (jetW1 (a t) J e) (jetW2 (a t) J e x) x :=
      hasDerivAt_w1 (a t e) (hd0 e) (hd1 e)
    have hW2 (e : E) : HasDerivAt (jetW2 (a t) J e) (jetW3 (a t) J e x) x :=
      hasDerivAt_w2 (a t e) (hd1 e) (hd2 e)
    have hW3 (e : E) : HasDerivAt (jetW3 (a t) J e) (jetW4 (a t) J e x) x :=
      hasDerivAt_w3 (a t e) (hd2 e) (hd3 e)
    have hG0 : HasDerivAt (jetG0 (a t) J) (jetG1 (a t) J x) x :=
      HasDerivAt.fun_sum fun e _ => hW0 e
    have hG1 : HasDerivAt (jetG1 (a t) J) (jetG2 (a t) J x) x :=
      HasDerivAt.fun_sum fun e _ => hW1 e
    have hG2 : HasDerivAt (jetG2 (a t) J) (jetG3 (a t) J x) x :=
      HasDerivAt.fun_sum fun e _ => hW2 e
    have hG3 : HasDerivAt (jetG3 (a t) J) (jetG4 (a t) J x) x :=
      HasDerivAt.fun_sum fun e _ => hW3 e
    have vecMul_deriv : ∀ (f f' : E → ℝ → ℝ),
        (∀ e, HasDerivAt (f e) (f' e x) x) →
        ∀ j, HasDerivAt (fun y => K.prob.vecMul (fun e => f e y) j)
          (K.prob.vecMul (fun e => f' e x) j) x := by
      intro f f' hf j
      simp only [Matrix.vecMul, dotProduct]
      exact HasDerivAt.fun_sum fun i _ => (hf i).mul_const (K.prob i j)
    refine ⟨?_, ?_, ?_, ?_⟩
    · exact vecMul_deriv _ _ (fun e => hasDerivAt_q0 hG_ne (hW0 e) hG0)
    · exact vecMul_deriv _ _ (fun e => hasDerivAt_q1 hG_ne (hW0 e) (hW1 e) hG0 hG1)
    · exact vecMul_deriv _ _ (fun e => hasDerivAt_q2 hG_ne (hW0 e) (hW1 e) (hW2 e) hG0 hG1 hG2)
    · exact vecMul_deriv _ _
        (fun e => hasDerivAt_q3 hG_ne (hW0 e) (hW1 e) (hW2 e) (hW3 e) hG0 hG1 hG2 hG3)

/-- Derivatives 1 through 4 of `log(projGain K a t x)` expressed in `jetG0..jetG4`. -/
noncomputable def logGain1 {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (t : ℕ) (x : ℝ) : ℝ :=
  q0 (jetG1 (a t) (stateJetAt K a t)) (jetG0 (a t) (stateJetAt K a t)) x

noncomputable def logGain2 {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (t : ℕ) (x : ℝ) : ℝ :=
  q1 (jetG1 (a t) (stateJetAt K a t)) (jetG2 (a t) (stateJetAt K a t))
    (jetG0 (a t) (stateJetAt K a t)) (jetG1 (a t) (stateJetAt K a t)) x

noncomputable def logGain3 {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (t : ℕ) (x : ℝ) : ℝ :=
  q2 (jetG1 (a t) (stateJetAt K a t)) (jetG2 (a t) (stateJetAt K a t))
    (jetG3 (a t) (stateJetAt K a t))
    (jetG0 (a t) (stateJetAt K a t)) (jetG1 (a t) (stateJetAt K a t))
    (jetG2 (a t) (stateJetAt K a t)) x

noncomputable def logGain4 {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (t : ℕ) (x : ℝ) : ℝ :=
  q3 (jetG1 (a t) (stateJetAt K a t)) (jetG2 (a t) (stateJetAt K a t))
    (jetG3 (a t) (stateJetAt K a t)) (jetG4 (a t) (stateJetAt K a t))
    (jetG0 (a t) (stateJetAt K a t)) (jetG1 (a t) (stateJetAt K a t))
    (jetG2 (a t) (stateJetAt K a t)) (jetG3 (a t) (stateJetAt K a t)) x

lemma logGain_hasDerivAt {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (A : ℝ)
    (hA : 0 < A) (ha : ∀ t e, |a t e| ≤ A) (t : ℕ) (x : ℝ)
    (hx : x ∈ Ioo (-(1 / (2 * A))) (1 / (2 * A))) :
    HasDerivAt (fun y => Real.log (projGain K a t y)) (logGain1 K a t x) x ∧
    HasDerivAt (logGain1 K a t) (logGain2 K a t x) x ∧
    HasDerivAt (logGain2 K a t) (logGain3 K a t x) x ∧
    HasDerivAt (logGain3 K a t) (logGain4 K a t x) x := by
  obtain ⟨hd0, hd1, hd2, hd3⟩ := stateJetAt_hasDerivAt K a A hA ha t x hx
  let J := stateJetAt K a t
  have hG_pos : 0 < jetG0 (a t) J x := by
    rw [jetG0_eq_projGain]
    linarith [(projState_prob_and_gain_bounds K a A hA ha x hx t).2.2.1]
  have hG_ne : jetG0 (a t) J x ≠ 0 := hG_pos.ne'
  have hG0 : HasDerivAt (jetG0 (a t) J) (jetG1 (a t) J x) x :=
    HasDerivAt.fun_sum fun e _ => hasDerivAt_w0 (a t e) (hd0 e)
  have hG1 : HasDerivAt (jetG1 (a t) J) (jetG2 (a t) J x) x :=
    HasDerivAt.fun_sum fun e _ => hasDerivAt_w1 (a t e) (hd0 e) (hd1 e)
  have hG2 : HasDerivAt (jetG2 (a t) J) (jetG3 (a t) J x) x :=
    HasDerivAt.fun_sum fun e _ => hasDerivAt_w2 (a t e) (hd1 e) (hd2 e)
  have hG3 : HasDerivAt (jetG3 (a t) J) (jetG4 (a t) J x) x :=
    HasDerivAt.fun_sum fun e _ => hasDerivAt_w3 (a t e) (hd2 e) (hd3 e)
  refine ⟨?_, ?_, ?_, ?_⟩
  · have hlog := hG0.log hG_ne
    have hfun : (fun y => Real.log (jetG0 (a t) J y)) = fun y => Real.log (projGain K a t y) := by
      funext y; rw [jetG0_eq_projGain]
    rw [hfun] at hlog
    simpa only [logGain1, q0] using hlog
  · exact hasDerivAt_q0 hG_ne hG1 hG0
  · exact hasDerivAt_q1 hG_ne hG1 hG2 hG0 hG1
  · exact hasDerivAt_q2 hG_ne hG1 hG2 hG3 hG0 hG1 hG2

end ProjectiveDerivatives

/-! ## Section 4: Uniform projective contraction bounds through fourth order -/

section ProjectiveBounds

lemma stateJetAt_zero_sum {E : Type*} [Fintype E]
    (K : Chain E) (a : ℕ → E → ℝ) (A : ℝ)
    (hA : 0 < A) (ha : ∀ t e, |a t e| ≤ A) (t : ℕ) (x : ℝ)
    (hx : x ∈ Ioo (-(1 / (2 * A))) (1 / (2 * A))) :
    let J := stateJetAt K a t
    (∑ e, J.u0 x e = 1) ∧
    (∑ e, J.u1 x e = 0) ∧
    (∑ e, J.u2 x e = 0) ∧
    (∑ e, J.u3 x e = 0) ∧
    (∑ e, J.u4 x e = 0) := by
  induction t with
  | zero =>
    simp [stateJetAt, K.initial_total]
  | succ t ih =>
    obtain ⟨hs0, hs1, hs2, hs3, _hs4⟩ := ih
    let J := stateJetAt K a t
    have hG_pos : 0 < jetG0 (a t) J x := by
      rw [jetG0_eq_projGain]
      linarith [(projState_prob_and_gain_bounds K a A hA ha x hx t).2.2.1]
    have hG_ne : jetG0 (a t) J x ≠ 0 := hG_pos.ne'
    have sum_vecMul : ∀ v : E → ℝ, ∑ j, K.prob.vecMul v j = ∑ i, v i := by
      intro v
      simp only [Matrix.vecMul, dotProduct]
      rw [Finset.sum_comm]
      simp [← Finset.mul_sum, K.total]
    have hq0_sum : ∑ e, q0 (jetW0 (a t) J e) (jetG0 (a t) J) x = 1 := by
      simp only [q0, ← Finset.sum_div]
      exact div_self hG_ne
    have hq1_sum : ∑ e, q1 (jetW0 (a t) J e) (jetW1 (a t) J e)
        (jetG0 (a t) J) (jetG1 (a t) J) x = 0 := by
      simp only [q1, ← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum, hq0_sum, mul_one]
      change (jetG1 (a t) J x - jetG1 (a t) J x) / jetG0 (a t) J x = 0
      simp
    have hq2_sum : ∑ e, q2 (jetW0 (a t) J e) (jetW1 (a t) J e) (jetW2 (a t) J e)
        (jetG0 (a t) J) (jetG1 (a t) J) (jetG2 (a t) J) x = 0 := by
      simp only [q2, ← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum,
        hq0_sum, hq1_sum, mul_one, mul_zero, sub_zero]
      change (jetG2 (a t) J x - jetG2 (a t) J x) / jetG0 (a t) J x = 0
      simp
    have hq3_sum : ∑ e, q3 (jetW0 (a t) J e) (jetW1 (a t) J e) (jetW2 (a t) J e)
        (jetW3 (a t) J e) (jetG0 (a t) J) (jetG1 (a t) J) (jetG2 (a t) J)
        (jetG3 (a t) J) x = 0 := by
      simp only [q3, ← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum,
        hq0_sum, hq1_sum, hq2_sum, mul_one, mul_zero, sub_zero]
      change (jetG3 (a t) J x - jetG3 (a t) J x) / jetG0 (a t) J x = 0
      simp
    have hq4_sum : ∑ e, q4 (jetW0 (a t) J e) (jetW1 (a t) J e) (jetW2 (a t) J e)
        (jetW3 (a t) J e) (jetW4 (a t) J e) (jetG0 (a t) J) (jetG1 (a t) J)
        (jetG2 (a t) J) (jetG3 (a t) J) (jetG4 (a t) J) x = 0 := by
      simp only [q4, ← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum,
        hq0_sum, hq1_sum, hq2_sum, hq3_sum, mul_one, mul_zero, sub_zero]
      change (jetG4 (a t) J x - jetG4 (a t) J x) / jetG0 (a t) J x = 0
      simp
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq0_sum
    · simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq1_sum
    · simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq2_sum
    · simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq3_sum
    · simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq4_sum

/-- Contraction step bound for any higher-order projective state derivative `u_k`:
handles `(u * (1 + x * b) - (x * ∑ b * u) * q0 + S) / G`, where `S` involves ONLY strictly
lower-order state derivatives (`0 .. k - 1`) and `q0` satisfies `l1 q0 = 1`. -/
lemma gauge_projective_linear_step {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (hm : 0 < m)
    (hblock : ∀ z : E → ℝ, (∑ i, z i) = 0 → l1 ((K.prob ^ m).vecMul z) ≤ l1 z / 2)
    (b u q0f S : E → ℝ) (G x A : ℝ) (hA : 0 < A)
    (hb : ∀ e, |b e| ≤ A) (hu_sum : ∑ e, u e = 0) (hq0_l1 : l1 q0f = 1)
    (hx : |x| ≤ 1 / (24 * (m : ℝ) ^ 2 * A))
    (hG_lo : 1 / 2 ≤ G) (hG_diff : |1 - G| ≤ |x| * A) :
    gauge K m (K.prob.vecMul
      (fun e => (u e * (1 + x * b e) - (x * ∑ i, b i * u i) * q0f e + S e) / G)) ≤
      (1 - 1 / (4 * (m : ℝ))) * gauge K m u + 2 * (m : ℝ) * l1 S := by
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hG0 : 0 < G := by linarith
  let d : ℝ := ∑ i, b i * u i
  let rem : E → ℝ := fun e => ((1 - G) * u e + x * (b e * u e) - (x * d) * q0f e + S e) / G
  have hsplit : K.prob.vecMul
      (fun e => (u e * (1 + x * b e) - (x * d) * q0f e + S e) / G) =
      fun j => K.prob.vecMul u j + K.prob.vecMul rem j := by
    funext j
    simp only [Matrix.vecMul, dotProduct, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i _
    dsimp [rem]
    field_simp [hG0.ne']
    ring
  rw [hsplit]
  have htri := gauge_add_le K m (K.prob.vecMul u) (K.prob.vecMul rem)
  have hmain := gauge_contraction K m hm hblock u hu_sum
  have hbu : l1 (fun e => b e * u e) ≤ A * l1 u := by
    unfold l1
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro e _
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_right (hb e) (abs_nonneg _)
  have hd_le : |d| ≤ A * l1 u := abs_dot_le_mul_l1 b u A hb
  have hrem_l1 : l1 rem ≤ 6 * (|x| * A) * l1 u + 2 * l1 S := by
    have hdiv := l1_div_le
      (fun e => (1 - G) * u e + x * (b e * u e) - (x * d) * q0f e + S e) hG_lo
    have ha1 := l1_add_le
      (fun e => (1 - G) * u e + x * (b e * u e) - (x * d) * q0f e) S
    have ha2 := l1_sub_le
      (fun e => (1 - G) * u e + x * (b e * u e)) (fun e => (x * d) * q0f e)
    have ha3 := l1_add_le (fun e => (1 - G) * u e) (fun e => x * (b e * u e))
    rw [l1_smul, hq0_l1, mul_one, abs_mul] at ha2
    rw [l1_smul, l1_smul] at ha3
    have h1G : |1 - G| * l1 u ≤ (|x| * A) * l1 u :=
      mul_le_mul_of_nonneg_right hG_diff (l1_nonneg u)
    have hxbu : |x| * l1 (fun e => b e * u e) ≤ (|x| * A) * l1 u := by
      calc |x| * l1 (fun e => b e * u e) ≤ |x| * (A * l1 u) :=
            mul_le_mul_of_nonneg_left hbu (abs_nonneg x)
        _ = (|x| * A) * l1 u := by ring
    have hxd : |x| * |d| ≤ (|x| * A) * l1 u := by
      calc |x| * |d| ≤ |x| * (A * l1 u) :=
            mul_le_mul_of_nonneg_left hd_le (abs_nonneg x)
        _ = (|x| * A) * l1 u := by ring
    dsimp [rem]
    linarith
  have hrem_gauge : gauge K m (K.prob.vecMul rem) ≤
      (1 / (4 * (m : ℝ))) * gauge K m u + 2 * (m : ℝ) * l1 S := by
    have hst := gauge_step_le K m rem
    have hgu := gauge_lower K m hm u
    have hxA : 6 * (m : ℝ) * (|x| * A) ≤ 1 / (4 * (m : ℝ)) := by
      have hxA0 : |x| * A ≤ 1 / (24 * (m : ℝ) ^ 2) := by
        have hh := mul_le_mul_of_nonneg_right hx hA.le
        have hrhs : 1 / (24 * (m : ℝ) ^ 2 * A) * A = 1 / (24 * (m : ℝ) ^ 2) := by
          field_simp [hA.ne', hmR.ne']
        rwa [hrhs] at hh
      calc
        6 * (m : ℝ) * (|x| * A) ≤ 6 * (m : ℝ) * (1 / (24 * (m : ℝ) ^ 2)) :=
          mul_le_mul_of_nonneg_left hxA0 (by positivity)
        _ = 1 / (4 * (m : ℝ)) := by field_simp [hmR.ne']; ring
    have h1 : (m : ℝ) * l1 rem ≤ 6 * (m : ℝ) * (|x| * A) * l1 u + 2 * (m : ℝ) * l1 S := by
      have hh := mul_le_mul_of_nonneg_left hrem_l1 hmR.le
      linarith
    have h2 : 6 * (m : ℝ) * (|x| * A) * l1 u ≤ (1 / (4 * (m : ℝ))) * gauge K m u := by
      calc
        6 * (m : ℝ) * (|x| * A) * l1 u ≤ (1 / (4 * (m : ℝ))) * l1 u :=
          mul_le_mul_of_nonneg_right hxA (l1_nonneg u)
        _ ≤ (1 / (4 * (m : ℝ))) * gauge K m u :=
          mul_le_mul_of_nonneg_left hgu (by positivity)
    linarith
  have halg : (1 - 1 / (2 * (m : ℝ))) * gauge K m u +
      (1 / (4 * (m : ℝ))) * gauge K m u =
      (1 - 1 / (4 * (m : ℝ))) * gauge K m u := by
    field_simp [hmR.ne']; ring
  linarith

lemma jetG_bounds {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (hm : 0 < m)
    (b : E → ℝ) (J : StateJet E) (x A : ℝ) (hA : 0 < A)
    (hb : ∀ e, |b e| ≤ A) (hx : |x| ≤ 1 / (2 * A))
    (hu0_nn : ∀ e, 0 ≤ J.u0 x e) (hu0_sum : ∑ e, J.u0 x e = 1)
    (hu1_sum : ∑ e, J.u1 x e = 0) (hu2_sum : ∑ e, J.u2 x e = 0)
    (hu3_sum : ∑ e, J.u3 x e = 0) (hu4_sum : ∑ e, J.u4 x e = 0) :
    l1 (J.u0 x) = 1 ∧
    |1 - jetG0 b J x| ≤ |x| * A ∧
    |jetG1 b J x| ≤ A + gauge K m (J.u1 x) ∧
    |jetG2 b J x| ≤ 2 * A * gauge K m (J.u1 x) + gauge K m (J.u2 x) ∧
    |jetG3 b J x| ≤ 3 * A * gauge K m (J.u2 x) + gauge K m (J.u3 x) ∧
    |jetG4 b J x| ≤ 4 * A * gauge K m (J.u3 x) + gauge K m (J.u4 x) := by
  have hl1_u0 : l1 (J.u0 x) = 1 := by
    unfold l1
    rw [Finset.sum_congr rfl fun e _ => abs_of_nonneg (hu0_nn e), hu0_sum]
  have hxA_le : |x| * A ≤ 1 := by
    have hh := mul_le_mul_of_nonneg_right hx hA.le
    have hrhs : 1 / (2 * A) * A = 1 / 2 := by field_simp [hA.ne']
    rw [hrhs] at hh
    linarith
  have h1G : |1 - jetG0 b J x| ≤ |x| * A := by
    have hid : 1 - jetG0 b J x = -x * ∑ e, b e * J.u0 x e := by
      simp only [jetG0, jetW0, w0, mul_add, mul_one, Finset.sum_add_distrib]
      have hxb : (∑ e, J.u0 x e * (x * b e)) = x * ∑ e, b e * J.u0 x e := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun e _ => by ring
      rw [hu0_sum, hxb]
      ring
    rw [hid, abs_mul, abs_neg]
    have hdot := abs_dot_le_mul_l1 b (J.u0 x) A hb
    rw [hl1_u0, mul_one] at hdot
    exact mul_le_mul_of_nonneg_left hdot (abs_nonneg x)
  have hG1_eq : jetG1 b J x = x * (∑ e, b e * J.u1 x e) + ∑ e, b e * J.u0 x e := by
    simp only [jetG1, jetW1, w1, Finset.sum_add_distrib, mul_add, mul_one]
    have hxb : (∑ e, J.u1 x e * (x * b e)) = x * ∑ e, b e * J.u1 x e := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun e _ => by ring
    rw [hu1_sum, zero_add, hxb]
  have hG2_eq : jetG2 b J x = x * (∑ e, b e * J.u2 x e) + 2 * ∑ e, b e * J.u1 x e := by
    simp only [jetG2, jetW2, w2, Finset.sum_add_distrib, mul_add, mul_one]
    have hxb : (∑ e, J.u2 x e * (x * b e)) = x * ∑ e, b e * J.u2 x e := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun e _ => by ring
    have h2b : (∑ e, 2 * b e * J.u1 x e) = 2 * ∑ e, b e * J.u1 x e := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun e _ => by ring
    rw [hu2_sum, zero_add, hxb, h2b]
  have hG3_eq : jetG3 b J x = x * (∑ e, b e * J.u3 x e) + 3 * ∑ e, b e * J.u2 x e := by
    simp only [jetG3, jetW3, w3, Finset.sum_add_distrib, mul_add, mul_one]
    have hxb : (∑ e, J.u3 x e * (x * b e)) = x * ∑ e, b e * J.u3 x e := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun e _ => by ring
    have h3b : (∑ e, 3 * b e * J.u2 x e) = 3 * ∑ e, b e * J.u2 x e := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun e _ => by ring
    rw [hu3_sum, zero_add, hxb, h3b]
  have hG4_eq : jetG4 b J x = x * (∑ e, b e * J.u4 x e) + 4 * ∑ e, b e * J.u3 x e := by
    simp only [jetG4, jetW4, w4, Finset.sum_add_distrib, mul_add, mul_one]
    have hxb : (∑ e, J.u4 x e * (x * b e)) = x * ∑ e, b e * J.u4 x e := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun e _ => by ring
    have h4b : (∑ e, 4 * b e * J.u3 x e) = 4 * ∑ e, b e * J.u3 x e := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun e _ => by ring
    rw [hu4_sum, zero_add, hxb, h4b]
  have hd0 : |∑ e, b e * J.u0 x e| ≤ A := by
    simpa only [hl1_u0, mul_one] using abs_dot_le_mul_l1 b (J.u0 x) A hb
  have hd1 : |∑ e, b e * J.u1 x e| ≤ A * gauge K m (J.u1 x) :=
    (abs_dot_le_mul_l1 b (J.u1 x) A hb).trans
      (mul_le_mul_of_nonneg_left (gauge_lower K m hm (J.u1 x)) hA.le)
  have hd2 : |∑ e, b e * J.u2 x e| ≤ A * gauge K m (J.u2 x) :=
    (abs_dot_le_mul_l1 b (J.u2 x) A hb).trans
      (mul_le_mul_of_nonneg_left (gauge_lower K m hm (J.u2 x)) hA.le)
  have hd3 : |∑ e, b e * J.u3 x e| ≤ A * gauge K m (J.u3 x) :=
    (abs_dot_le_mul_l1 b (J.u3 x) A hb).trans
      (mul_le_mul_of_nonneg_left (gauge_lower K m hm (J.u3 x)) hA.le)
  have hd4 : |∑ e, b e * J.u4 x e| ≤ A * gauge K m (J.u4 x) :=
    (abs_dot_le_mul_l1 b (J.u4 x) A hb).trans
      (mul_le_mul_of_nonneg_left (gauge_lower K m hm (J.u4 x)) hA.le)
  have hg1_nn : 0 ≤ gauge K m (J.u1 x) := (l1_nonneg _).trans (gauge_lower K m hm _)
  have hg2_nn : 0 ≤ gauge K m (J.u2 x) := (l1_nonneg _).trans (gauge_lower K m hm _)
  have hg3_nn : 0 ≤ gauge K m (J.u3 x) := (l1_nonneg _).trans (gauge_lower K m hm _)
  have hg4_nn : 0 ≤ gauge K m (J.u4 x) := (l1_nonneg _).trans (gauge_lower K m hm _)
  refine ⟨hl1_u0, h1G, ?_, ?_, ?_, ?_⟩
  · rw [hG1_eq]
    have htri := abs_add_le (x * ∑ e, b e * J.u1 x e) (∑ e, b e * J.u0 x e)
    rw [abs_mul] at htri
    have hx1 : |x| * |∑ e, b e * J.u1 x e| ≤ gauge K m (J.u1 x) := by
      calc |x| * |∑ e, b e * J.u1 x e| ≤ |x| * (A * gauge K m (J.u1 x)) :=
            mul_le_mul_of_nonneg_left hd1 (abs_nonneg x)
        _ = (|x| * A) * gauge K m (J.u1 x) := by ring
        _ ≤ 1 * gauge K m (J.u1 x) := mul_le_mul_of_nonneg_right hxA_le hg1_nn
        _ = gauge K m (J.u1 x) := one_mul _
    linarith
  · rw [hG2_eq]
    have htri := abs_add_le (x * ∑ e, b e * J.u2 x e) (2 * ∑ e, b e * J.u1 x e)
    rw [abs_mul, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] at htri
    have hx2 : |x| * |∑ e, b e * J.u2 x e| ≤ gauge K m (J.u2 x) := by
      calc |x| * |∑ e, b e * J.u2 x e| ≤ |x| * (A * gauge K m (J.u2 x)) :=
            mul_le_mul_of_nonneg_left hd2 (abs_nonneg x)
        _ = (|x| * A) * gauge K m (J.u2 x) := by ring
        _ ≤ 1 * gauge K m (J.u2 x) := mul_le_mul_of_nonneg_right hxA_le hg2_nn
        _ = gauge K m (J.u2 x) := one_mul _
    nlinarith
  · rw [hG3_eq]
    have htri := abs_add_le (x * ∑ e, b e * J.u3 x e) (3 * ∑ e, b e * J.u2 x e)
    rw [abs_mul, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 3)] at htri
    have hx3 : |x| * |∑ e, b e * J.u3 x e| ≤ gauge K m (J.u3 x) := by
      calc |x| * |∑ e, b e * J.u3 x e| ≤ |x| * (A * gauge K m (J.u3 x)) :=
            mul_le_mul_of_nonneg_left hd3 (abs_nonneg x)
        _ = (|x| * A) * gauge K m (J.u3 x) := by ring
        _ ≤ 1 * gauge K m (J.u3 x) := mul_le_mul_of_nonneg_right hxA_le hg3_nn
        _ = gauge K m (J.u3 x) := one_mul _
    nlinarith
  · rw [hG4_eq]
    have htri := abs_add_le (x * ∑ e, b e * J.u4 x e) (4 * ∑ e, b e * J.u3 x e)
    rw [abs_mul, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 4)] at htri
    have hx4 : |x| * |∑ e, b e * J.u4 x e| ≤ gauge K m (J.u4 x) := by
      calc |x| * |∑ e, b e * J.u4 x e| ≤ |x| * (A * gauge K m (J.u4 x)) :=
            mul_le_mul_of_nonneg_left hd4 (abs_nonneg x)
        _ = (|x| * A) * gauge K m (J.u4 x) := by ring
        _ ≤ 1 * gauge K m (J.u4 x) := mul_le_mul_of_nonneg_right hxA_le hg4_nn
        _ = gauge K m (J.u4 x) := one_mul _
    nlinarith

/-- Explicit uniform upper bounds on `gauge K m (u_k)` (`k = 1..4`) across all time steps `t : ℕ`
and all `|x| ≤ 1 / (24 * m^2 * A)`. -/
noncomputable def stateBound1 (m : ℕ) (A : ℝ) : ℝ :=
  8 * (m : ℝ) ^ 2 * (2 * A)

noncomputable def stateBound2 (m : ℕ) (A : ℝ) : ℝ :=
  let U1 := stateBound1 m A
  let g1 := A + U1
  let Q1 := 4 * U1 + 2 * A + 4 * g1
  8 * (m : ℝ) ^ 2 * (4 * A * U1 + 2 * g1 * Q1)

noncomputable def stateBound3 (m : ℕ) (A : ℝ) : ℝ :=
  let U1 := stateBound1 m A
  let U2 := stateBound2 m A
  let g1 := A + U1
  let g2 := 2 * A * U1 + U2
  let Q1 := 4 * U1 + 2 * A + 4 * g1
  let Q2 := 4 * U2 + 4 * A * U1 + 4 * g1 * Q1 + 4 * g2
  8 * (m : ℝ) ^ 2 * (6 * A * U2 + 3 * g1 * Q2 + 3 * g2 * Q1)

noncomputable def stateBound4 (m : ℕ) (A : ℝ) : ℝ :=
  let U1 := stateBound1 m A
  let U2 := stateBound2 m A
  let U3 := stateBound3 m A
  let g1 := A + U1
  let g2 := 2 * A * U1 + U2
  let g3 := 3 * A * U2 + U3
  let Q1 := 4 * U1 + 2 * A + 4 * g1
  let Q2 := 4 * U2 + 4 * A * U1 + 4 * g1 * Q1 + 4 * g2
  let Q3 := 4 * U3 + 6 * A * U2 + 6 * g1 * Q2 + 6 * g2 * Q1 + 4 * g3
  8 * (m : ℝ) ^ 2 * (8 * A * U3 + 4 * g1 * Q3 + 6 * g2 * Q2 + 4 * g3 * Q1)

/-- Uniform bounds on `|logGain1..4|` over all steps `t` and `|x| ≤ 1 / (24 * m^2 * A)`. -/
noncomputable def logGain1Bound (m : ℕ) (A : ℝ) : ℝ :=
  2 * (A + stateBound1 m A)

noncomputable def logGain2Bound (m : ℕ) (A : ℝ) : ℝ :=
  let U1 := stateBound1 m A
  let U2 := stateBound2 m A
  let g1 := A + U1
  let g2 := 2 * A * U1 + U2
  2 * (g2 + g1 * logGain1Bound m A)

noncomputable def logGain3Bound (m : ℕ) (A : ℝ) : ℝ :=
  let U1 := stateBound1 m A
  let U2 := stateBound2 m A
  let U3 := stateBound3 m A
  let g1 := A + U1
  let g2 := 2 * A * U1 + U2
  let g3 := 3 * A * U2 + U3
  2 * (g3 + 2 * g1 * logGain2Bound m A + g2 * logGain1Bound m A)

noncomputable def logGain4Bound (m : ℕ) (A : ℝ) : ℝ :=
  let U1 := stateBound1 m A
  let U2 := stateBound2 m A
  let U3 := stateBound3 m A
  let U4 := stateBound4 m A
  let g1 := A + U1
  let g2 := 2 * A * U1 + U2
  let g3 := 3 * A * U2 + U3
  let g4 := 4 * A * U3 + U4
  2 * (g4 + 3 * g1 * logGain3Bound m A + 3 * g2 * logGain2Bound m A + g3 * logGain1Bound m A)

lemma l1_bu_le {E : Type*} [Fintype E] (b u : E → ℝ) (A : ℝ) (hb : ∀ e, |b e| ≤ A) :
    l1 (fun e => b e * u e) ≤ A * l1 u := by
  unfold l1
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro e _
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_right (hb e) (abs_nonneg _)

lemma l1_mul_factor_le {E : Type*} [Fintype E] (b u : E → ℝ) (x : ℝ)
    (hfac : ∀ e, |1 + x * b e| ≤ 2) :
    l1 (fun e => u e * (1 + x * b e)) ≤ 2 * l1 u := by
  unfold l1
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro e _
  rw [abs_mul]
  calc |u e| * |1 + x * b e| ≤ |u e| * 2 :=
        mul_le_mul_of_nonneg_left (hfac e) (abs_nonneg _)
    _ = 2 * |u e| := by ring

lemma contraction_fixed_point_step {m : ℕ} {g g' U H : ℝ} (hm : 0 < m)
    (hg : g ≤ U)
    (hstep : g' ≤ (1 - 1 / (4 * (m : ℝ))) * g + 2 * (m : ℝ) * H)
    (hU : U = 8 * (m : ℝ) ^ 2 * H) : g' ≤ U := by
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have h1m : 0 ≤ 1 - 1 / (4 * (m : ℝ)) := by
    have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
    have hh : 1 / (4 * (m : ℝ)) ≤ 1 := (div_le_iff₀ (by positivity)).mpr (by linarith)
    linarith
  have h1 : (1 - 1 / (4 * (m : ℝ))) * g ≤ (1 - 1 / (4 * (m : ℝ))) * U :=
    mul_le_mul_of_nonneg_left hg h1m
  have hrhs : (1 - 1 / (4 * (m : ℝ))) * U + 2 * (m : ℝ) * H = U := by
    rw [hU]; field_simp [hmR.ne']; ring
  linarith

/-- Uniform horizon- and path-independent bounds on all four normalized state derivatives
and all four one-step logarithmic gain derivatives (`part1:81`, `part1:82`, `part1:69`). -/
theorem projective_state_and_logGain_bounds {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (hm : 0 < m)
    (hblock : ∀ z : E → ℝ, (∑ i, z i) = 0 → l1 ((K.prob ^ m).vecMul z) ≤ l1 z / 2)
    (a : ℕ → E → ℝ) (A : ℝ) (hA : 0 < A) (ha : ∀ t e, |a t e| ≤ A)
    (x : ℝ) (hx : |x| ≤ 1 / (24 * (m : ℝ) ^ 2 * A)) (t : ℕ) :
    (gauge K m ((stateJetAt K a t).u1 x) ≤ stateBound1 m A ∧
      gauge K m ((stateJetAt K a t).u2 x) ≤ stateBound2 m A ∧
      gauge K m ((stateJetAt K a t).u3 x) ≤ stateBound3 m A ∧
      gauge K m ((stateJetAt K a t).u4 x) ≤ stateBound4 m A) ∧
    (|logGain1 K a t x| ≤ logGain1Bound m A ∧
      |logGain2 K a t x| ≤ logGain2Bound m A ∧
      |logGain3 K a t x| ≤ logGain3Bound m A ∧
      |logGain4 K a t x| ≤ logGain4Bound m A) := by
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hx_lt_half : x ∈ Ioo (-(1 / (2 * A))) (1 / (2 * A)) := by
    have hden : 2 * A < 24 * (m : ℝ) ^ 2 * A := by nlinarith [sq_nonneg ((m : ℝ) - 1)]
    have hinv : 1 / (24 * (m : ℝ) ^ 2 * A) < 1 / (2 * A) :=
      one_div_lt_one_div_of_lt (by positivity) hden
    exact abs_lt.mp (hx.trans_lt hinv)
  have hx_le_half : |x| ≤ 1 / (2 * A) := (abs_lt.mpr hx_lt_half).le
  have hstate : ∀ s : ℕ,
      gauge K m ((stateJetAt K a s).u1 x) ≤ stateBound1 m A ∧
      gauge K m ((stateJetAt K a s).u2 x) ≤ stateBound2 m A ∧
      gauge K m ((stateJetAt K a s).u3 x) ≤ stateBound3 m A ∧
      gauge K m ((stateJetAt K a s).u4 x) ≤ stateBound4 m A := by
    intro s
    induction s with
    | zero =>
      have hU1_nn : 0 ≤ stateBound1 m A := by unfold stateBound1; positivity
      have hU2_nn : 0 ≤ stateBound2 m A := by unfold stateBound2 stateBound1; positivity
      have hU3_nn : 0 ≤ stateBound3 m A := by
        unfold stateBound3 stateBound2 stateBound1; positivity
      have hU4_nn : 0 ≤ stateBound4 m A := by
        unfold stateBound4 stateBound3 stateBound2 stateBound1; positivity
      simp only [stateJetAt, gauge_zero]
      exact ⟨hU1_nn, hU2_nn, hU3_nn, hU4_nn⟩
    | succ s ih =>
      obtain ⟨hu1, hu2, hu3, hu4⟩ := ih
      let J := stateJetAt K a s
      let b := a s
      have hprob := projState_prob_and_gain_bounds K a A hA ha x hx_lt_half s
      have hzs := stateJetAt_zero_sum K a A hA ha s x hx_lt_half
      have hG_lo : 1 / 2 ≤ jetG0 b J x := by
        rw [jetG0_eq_projGain]; exact hprob.2.2.1.le
      have hu0_nn : ∀ e, 0 ≤ J.u0 x e := by
        intro e; rw [stateJetAt_u0_eq_projState]; exact hprob.1 e
      obtain ⟨hl1_u0, h1G, hG1, hG2, hG3, _hG4⟩ := jetG_bounds K m hm b J x A hA
        (ha s) hx_le_half hu0_nn hzs.1 hzs.2.1 hzs.2.2.1 hzs.2.2.2.1 hzs.2.2.2.2
      have hfac2 (e : E) : |1 + x * b e| ≤ 2 := by
        have hf := additive_factor_bounds (b e) x A hA (ha s e) hx_lt_half
        exact abs_le.mpr ⟨by linarith [hf.1], hf.2.le⟩
      let U1 := stateBound1 m A
      let U2 := stateBound2 m A
      let U3 := stateBound3 m A
      let g1 := A + U1
      let g2 := 2 * A * U1 + U2
      let g3 := 3 * A * U2 + U3
      let Q1 := 4 * U1 + 2 * A + 4 * g1
      let Q2 := 4 * U2 + 4 * A * U1 + 4 * g1 * Q1 + 4 * g2
      let Q3 := 4 * U3 + 6 * A * U2 + 6 * g1 * Q2 + 6 * g2 * Q1 + 4 * g3
      let q0f : E → ℝ := fun e => q0 (jetW0 b J e) (jetG0 b J) x
      let q1f : E → ℝ := fun e => q1 (jetW0 b J e) (jetW1 b J e) (jetG0 b J) (jetG1 b J) x
      let q2f : E → ℝ := fun e => q2 (jetW0 b J e) (jetW1 b J e) (jetW2 b J e)
        (jetG0 b J) (jetG1 b J) (jetG2 b J) x
      let q3f : E → ℝ := fun e => q3 (jetW0 b J e) (jetW1 b J e) (jetW2 b J e) (jetW3 b J e)
        (jetG0 b J) (jetG1 b J) (jetG2 b J) (jetG3 b J) x
      have hU1_nn : 0 ≤ U1 := by dsimp [U1, stateBound1]; positivity
      have hU2_nn : 0 ≤ U2 := by dsimp [U2, stateBound2, stateBound1]; positivity
      have hU3_nn : 0 ≤ U3 := by dsimp [U3, stateBound3, stateBound2, stateBound1]; positivity
      have hg1_nn : 0 ≤ g1 := by dsimp [g1]; positivity
      have hg2_nn : 0 ≤ g2 := by dsimp [g2]; positivity
      have hg3_nn : 0 ≤ g3 := by dsimp [g3]; positivity
      have hQ1_nn : 0 ≤ Q1 := by dsimp [Q1]; positivity
      have hQ2_nn : 0 ≤ Q2 := by dsimp [Q2]; positivity
      have hQ3_nn : 0 ≤ Q3 := by dsimp [Q3]; positivity
      have hg1_le : |jetG1 b J x| ≤ g1 := by change |jetG1 b J x| ≤ A + U1; linarith
      have hg2_le : |jetG2 b J x| ≤ g2 := by
        change |jetG2 b J x| ≤ 2 * A * U1 + U2
        have h1 : 2 * A * gauge K m (J.u1 x) ≤ 2 * A * U1 :=
          mul_le_mul_of_nonneg_left hu1 (by positivity)
        linarith
      have hg3_le : |jetG3 b J x| ≤ g3 := by
        change |jetG3 b J x| ≤ 3 * A * U2 + U3
        have h1 : 3 * A * gauge K m (J.u2 x) ≤ 3 * A * U2 :=
          mul_le_mul_of_nonneg_left hu2 (by positivity)
        linarith
      have hl1_u1 : l1 (J.u1 x) ≤ U1 := (gauge_lower K m hm _).trans hu1
      have hl1_u2 : l1 (J.u2 x) ≤ U2 := (gauge_lower K m hm _).trans hu2
      have hl1_u3 : l1 (J.u3 x) ≤ U3 := (gauge_lower K m hm _).trans hu3
      have hq0_l1 : l1 q0f = 1 := by
        unfold l1
        have hq0_nn (e : E) : 0 ≤ q0f e := by
          dsimp [q0f, q0, jetW0, w0]
          exact div_nonneg (mul_nonneg (hu0_nn e)
            (by linarith [(additive_factor_bounds (b e) x A hA (ha s e) hx_lt_half).1]))
            (by linarith)
        rw [Finset.sum_congr rfl fun e _ => abs_of_nonneg (hq0_nn e)]
        simp only [q0f, q0, ← Finset.sum_div]
        exact div_self (by linarith : jetG0 b J x ≠ 0)
      have hq1_l1 : l1 q1f ≤ Q1 := by
        have hd := l1_div_le (fun e => jetW1 b J e x - jetG1 b J x * q0f e) hG_lo
        have hs1 := l1_sub_le (fun e => jetW1 b J e x) (fun e => jetG1 b J x * q0f e)
        rw [l1_smul, hq0_l1, mul_one] at hs1
        have hw1_l1 : l1 (fun e => jetW1 b J e x) ≤ 2 * U1 + A := by
          have ha1 := l1_add_le (fun e => J.u1 x e * (1 + x * b e)) (fun e => b e * J.u0 x e)
          have hm_f := (l1_mul_factor_le b (J.u1 x) x hfac2).trans
            (mul_le_mul_of_nonneg_left hl1_u1 (by norm_num))
          have hm_b := l1_bu_le b (J.u0 x) A (ha s)
          rw [hl1_u0, mul_one] at hm_b
          change l1 (fun e => J.u1 x e * (1 + x * b e) + b e * J.u0 x e) ≤ _
          linarith
        change l1 (fun e => (jetW1 b J e x - jetG1 b J x * q0f e) / jetG0 b J x) ≤
          4 * U1 + 2 * A + 4 * g1
        linarith
      have hq2_l1 : l1 q2f ≤ Q2 := by
        have hd := l1_div_le (fun e => jetW2 b J e x -
          2 * jetG1 b J x * q1f e - jetG2 b J x * q0f e) hG_lo
        have hs1 := l1_sub_le (fun e => jetW2 b J e x - 2 * jetG1 b J x * q1f e)
          (fun e => jetG2 b J x * q0f e)
        have hs2 := l1_sub_le (fun e => jetW2 b J e x) (fun e => (2 * jetG1 b J x) * q1f e)
        rw [l1_smul, hq0_l1, mul_one] at hs1
        rw [l1_smul, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] at hs2
        have hw2_l1 : l1 (fun e => jetW2 b J e x) ≤ 2 * U2 + 2 * A * U1 := by
          have ha1 := l1_add_le (fun e => J.u2 x e * (1 + x * b e))
            (fun e => 2 * (b e * J.u1 x e))
          rw [l1_smul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] at ha1
          have hm_f := (l1_mul_factor_le b (J.u2 x) x hfac2).trans
            (mul_le_mul_of_nonneg_left hl1_u2 (by norm_num))
          have hm_b := (l1_bu_le b (J.u1 x) A (ha s)).trans
            (mul_le_mul_of_nonneg_left hl1_u1 hA.le)
          have heq_w2 : (fun e => jetW2 b J e x) =
              fun e => J.u2 x e * (1 + x * b e) + 2 * (b e * J.u1 x e) := by
            funext e; dsimp [jetW2, w2]; ring
          rw [heq_w2]
          linarith
        have hprod1 : 2 * |jetG1 b J x| * l1 q1f ≤ 2 * g1 * Q1 :=
          mul_le_mul (by linarith) hq1_l1 (l1_nonneg _) (by positivity)
        change l1 (fun e =>
          (jetW2 b J e x - 2 * jetG1 b J x * q1f e - jetG2 b J x * q0f e) / jetG0 b J x) ≤
          4 * U2 + 4 * A * U1 + 4 * g1 * Q1 + 4 * g2
        linarith
      have hq3_l1 : l1 q3f ≤ Q3 := by
        have hd := l1_div_le (fun e => jetW3 b J e x - 3 * jetG1 b J x * q2f e -
          3 * jetG2 b J x * q1f e - jetG3 b J x * q0f e) hG_lo
        have hs1 := l1_sub_le (fun e => jetW3 b J e x - 3 * jetG1 b J x * q2f e -
          3 * jetG2 b J x * q1f e) (fun e => jetG3 b J x * q0f e)
        have hs2 := l1_sub_le (fun e => jetW3 b J e x - 3 * jetG1 b J x * q2f e)
          (fun e => (3 * jetG2 b J x) * q1f e)
        have hs3 := l1_sub_le (fun e => jetW3 b J e x) (fun e => (3 * jetG1 b J x) * q2f e)
        rw [l1_smul, hq0_l1, mul_one] at hs1
        rw [l1_smul, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 3)] at hs2 hs3
        have hw3_l1 : l1 (fun e => jetW3 b J e x) ≤ 2 * U3 + 3 * A * U2 := by
          have ha1 := l1_add_le (fun e => J.u3 x e * (1 + x * b e))
            (fun e => 3 * (b e * J.u2 x e))
          rw [l1_smul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 3)] at ha1
          have hm_f := (l1_mul_factor_le b (J.u3 x) x hfac2).trans
            (mul_le_mul_of_nonneg_left hl1_u3 (by norm_num))
          have hm_b := (l1_bu_le b (J.u2 x) A (ha s)).trans
            (mul_le_mul_of_nonneg_left hl1_u2 hA.le)
          have heq_w3 : (fun e => jetW3 b J e x) =
              fun e => J.u3 x e * (1 + x * b e) + 3 * (b e * J.u2 x e) := by
            funext e; dsimp [jetW3, w3]; ring
          rw [heq_w3]
          linarith
        have hp1 : 3 * |jetG1 b J x| * l1 q2f ≤ 3 * g1 * Q2 :=
          mul_le_mul (by linarith) hq2_l1 (l1_nonneg _) (by positivity)
        have hp2 : 3 * |jetG2 b J x| * l1 q1f ≤ 3 * g2 * Q1 :=
          mul_le_mul (by linarith) hq1_l1 (l1_nonneg _) (by positivity)
        change l1 (fun e =>
          (jetW3 b J e x - 3 * jetG1 b J x * q2f e - 3 * jetG2 b J x * q1f e -
            jetG3 b J x * q0f e) / jetG0 b J x) ≤
          4 * U3 + 6 * A * U2 + 6 * g1 * Q2 + 6 * g2 * Q1 + 4 * g3
        linarith
      -- Explicit jetG1..jetG4 decompositions isolating `x * ∑ b * u_k`
      have hG1_split : jetG1 b J x = x * (∑ i, b i * J.u1 x i) + ∑ i, b i * J.u0 x i := by
        simp only [jetG1, jetW1, w1, Finset.sum_add_distrib, mul_add, mul_one]
        have hxb : (∑ i, J.u1 x i * (x * b i)) = x * ∑ i, b i * J.u1 x i := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring
        rw [hzs.2.1, zero_add, hxb]
      have hG2_split : jetG2 b J x = x * (∑ i, b i * J.u2 x i) + 2 * ∑ i, b i * J.u1 x i := by
        simp only [jetG2, jetW2, w2, Finset.sum_add_distrib, mul_add, mul_one]
        have hxb : (∑ i, J.u2 x i * (x * b i)) = x * ∑ i, b i * J.u2 x i := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring
        have h2b : (∑ i, 2 * b i * J.u1 x i) = 2 * ∑ i, b i * J.u1 x i := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring
        rw [hzs.2.2.1, zero_add, hxb, h2b]
      have hG3_split : jetG3 b J x = x * (∑ i, b i * J.u3 x i) + 3 * ∑ i, b i * J.u2 x i := by
        simp only [jetG3, jetW3, w3, Finset.sum_add_distrib, mul_add, mul_one]
        have hxb : (∑ i, J.u3 x i * (x * b i)) = x * ∑ i, b i * J.u3 x i := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring
        have h3b : (∑ i, 3 * b i * J.u2 x i) = 3 * ∑ i, b i * J.u2 x i := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring
        rw [hzs.2.2.2.1, zero_add, hxb, h3b]
      have hG4_split : jetG4 b J x = x * (∑ i, b i * J.u4 x i) + 4 * ∑ i, b i * J.u3 x i := by
        simp only [jetG4, jetW4, w4, Finset.sum_add_distrib, mul_add, mul_one]
        have hxb : (∑ i, J.u4 x i * (x * b i)) = x * ∑ i, b i * J.u4 x i := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring
        have h4b : (∑ i, 4 * b i * J.u3 x i) = 4 * ∑ i, b i * J.u3 x i := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring
        rw [hzs.2.2.2.2, zero_add, hxb, h4b]
      refine ⟨?_, ?_, ?_, ?_⟩
      · -- u1 step
        let S1 : E → ℝ := fun e => b e * J.u0 x e - (∑ i, b i * J.u0 x i) * q0f e
        have heq1 : (stateJetAt K a (s + 1)).u1 x =
            K.prob.vecMul (fun e =>
              (J.u1 x e * (1 + x * b e) - (x * ∑ i, b i * J.u1 x i) * q0f e + S1 e) /
                jetG0 b J x) := by
          funext j
          simp only [stateJetAt, nextStateJet, q1, jetW1, w1]
          congr 1; funext e
          change _ / jetG0 b J x = _ / jetG0 b J x
          rw [hG1_split]
          dsimp [S1, q0f]
          ring
        have hS1_l1 : l1 S1 ≤ 2 * A := by
          have hs := l1_sub_le (fun e => b e * J.u0 x e)
            (fun e => (∑ i, b i * J.u0 x i) * q0f e)
          rw [l1_smul, hq0_l1, mul_one] at hs
          have hb0 := l1_bu_le b (J.u0 x) A (ha s)
          have hd0 := abs_dot_le_mul_l1 b (J.u0 x) A (ha s)
          rw [hl1_u0, mul_one] at hb0 hd0
          change l1 (fun e => b e * J.u0 x e - (∑ i, b i * J.u0 x i) * q0f e) ≤ 2 * A
          linarith
        have hstep1 : gauge K m ((stateJetAt K a (s + 1)).u1 x) ≤
            (1 - 1 / (4 * (m : ℝ))) * gauge K m (J.u1 x) + 2 * (m : ℝ) * (2 * A) := by
          rw [heq1]
          have hlin := gauge_projective_linear_step K m hm hblock b (J.u1 x) q0f S1
            (jetG0 b J x) x A hA (ha s) hzs.2.1 hq0_l1 hx hG_lo h1G
          have hmS := mul_le_mul_of_nonneg_left hS1_l1 (by positivity : (0 : ℝ) ≤ 2 * (m : ℝ))
          linarith
        exact contraction_fixed_point_step hm hu1 hstep1 rfl
      · -- u2 step
        let S2 : E → ℝ := fun e => 2 * (b e * J.u1 x e) -
          (2 * jetG1 b J x) * q1f e - (2 * ∑ i, b i * J.u1 x i) * q0f e
        have heq2 : (stateJetAt K a (s + 1)).u2 x =
            K.prob.vecMul (fun e =>
              (J.u2 x e * (1 + x * b e) - (x * ∑ i, b i * J.u2 x i) * q0f e + S2 e) /
                jetG0 b J x) := by
          funext j
          simp only [stateJetAt, nextStateJet, q2, jetW2, w2]
          congr 1; funext e
          change _ / jetG0 b J x = _ / jetG0 b J x
          rw [hG2_split]
          dsimp [S2, q0f, q1f]
          ring
        have hS2_l1 : l1 S2 ≤ 4 * A * U1 + 2 * g1 * Q1 := by
          have hs1 := l1_sub_le (fun e => 2 * (b e * J.u1 x e) - (2 * jetG1 b J x) * q1f e)
            (fun e => (2 * ∑ i, b i * J.u1 x i) * q0f e)
          have hs2 := l1_sub_le (fun e => 2 * (b e * J.u1 x e))
            (fun e => (2 * jetG1 b J x) * q1f e)
          rw [l1_smul, hq0_l1, mul_one, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] at hs1
          rw [l1_smul, l1_smul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2),
            abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] at hs2
          have hb1 := (l1_bu_le b (J.u1 x) A (ha s)).trans
            (mul_le_mul_of_nonneg_left hl1_u1 hA.le)
          have hd1 := (abs_dot_le_mul_l1 b (J.u1 x) A (ha s)).trans
            (mul_le_mul_of_nonneg_left hl1_u1 hA.le)
          have hp1 : 2 * |jetG1 b J x| * l1 q1f ≤ 2 * g1 * Q1 :=
            mul_le_mul (by linarith) hq1_l1 (l1_nonneg _) (by positivity)
          change l1 (fun e => 2 * (b e * J.u1 x e) - (2 * jetG1 b J x) * q1f e -
            (2 * ∑ i, b i * J.u1 x i) * q0f e) ≤ _
          linarith
        have hstep2 : gauge K m ((stateJetAt K a (s + 1)).u2 x) ≤
            (1 - 1 / (4 * (m : ℝ))) * gauge K m (J.u2 x) +
              2 * (m : ℝ) * (4 * A * U1 + 2 * g1 * Q1) := by
          rw [heq2]
          have hlin := gauge_projective_linear_step K m hm hblock b (J.u2 x) q0f S2
            (jetG0 b J x) x A hA (ha s) hzs.2.2.1 hq0_l1 hx hG_lo h1G
          have hmS := mul_le_mul_of_nonneg_left hS2_l1 (by positivity : (0 : ℝ) ≤ 2 * (m : ℝ))
          linarith
        exact contraction_fixed_point_step hm hu2 hstep2 rfl
      · -- u3 step
        let S3 : E → ℝ := fun e => 3 * (b e * J.u2 x e) -
          (3 * jetG1 b J x) * q2f e - (3 * jetG2 b J x) * q1f e -
          (3 * ∑ i, b i * J.u2 x i) * q0f e
        have heq3 : (stateJetAt K a (s + 1)).u3 x =
            K.prob.vecMul (fun e =>
              (J.u3 x e * (1 + x * b e) - (x * ∑ i, b i * J.u3 x i) * q0f e + S3 e) /
                jetG0 b J x) := by
          funext j
          simp only [stateJetAt, nextStateJet, q3, jetW3, w3]
          congr 1; funext e
          change _ / jetG0 b J x = _ / jetG0 b J x
          rw [hG3_split]
          dsimp [S3, q0f, q1f, q2f]
          ring
        have hS3_l1 : l1 S3 ≤ 6 * A * U2 + 3 * g1 * Q2 + 3 * g2 * Q1 := by
          have hs1 := l1_sub_le
            (fun e => 3 * (b e * J.u2 x e) - (3 * jetG1 b J x) * q2f e -
              (3 * jetG2 b J x) * q1f e)
            (fun e => (3 * ∑ i, b i * J.u2 x i) * q0f e)
          have hs2 := l1_sub_le
            (fun e => 3 * (b e * J.u2 x e) - (3 * jetG1 b J x) * q2f e)
            (fun e => (3 * jetG2 b J x) * q1f e)
          have hs3 := l1_sub_le (fun e => 3 * (b e * J.u2 x e))
            (fun e => (3 * jetG1 b J x) * q2f e)
          rw [l1_smul, hq0_l1, mul_one, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 3)] at hs1
          rw [l1_smul, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 3)] at hs2
          rw [l1_smul, l1_smul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 3),
            abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 3)] at hs3
          have hb2 := (l1_bu_le b (J.u2 x) A (ha s)).trans
            (mul_le_mul_of_nonneg_left hl1_u2 hA.le)
          have hd2 := (abs_dot_le_mul_l1 b (J.u2 x) A (ha s)).trans
            (mul_le_mul_of_nonneg_left hl1_u2 hA.le)
          have hp1 : 3 * |jetG1 b J x| * l1 q2f ≤ 3 * g1 * Q2 :=
            mul_le_mul (by linarith) hq2_l1 (l1_nonneg _) (by positivity)
          have hp2 : 3 * |jetG2 b J x| * l1 q1f ≤ 3 * g2 * Q1 :=
            mul_le_mul (by linarith) hq1_l1 (l1_nonneg _) (by positivity)
          change l1 (fun e => 3 * (b e * J.u2 x e) - (3 * jetG1 b J x) * q2f e -
            (3 * jetG2 b J x) * q1f e - (3 * ∑ i, b i * J.u2 x i) * q0f e) ≤ _
          linarith
        have hstep3 : gauge K m ((stateJetAt K a (s + 1)).u3 x) ≤
            (1 - 1 / (4 * (m : ℝ))) * gauge K m (J.u3 x) +
              2 * (m : ℝ) * (6 * A * U2 + 3 * g1 * Q2 + 3 * g2 * Q1) := by
          rw [heq3]
          have hlin := gauge_projective_linear_step K m hm hblock b (J.u3 x) q0f S3
            (jetG0 b J x) x A hA (ha s) hzs.2.2.2.1 hq0_l1 hx hG_lo h1G
          have hmS := mul_le_mul_of_nonneg_left hS3_l1 (by positivity : (0 : ℝ) ≤ 2 * (m : ℝ))
          linarith
        exact contraction_fixed_point_step hm hu3 hstep3 rfl
      · -- u4 step
        let S4 : E → ℝ := fun e => 4 * (b e * J.u3 x e) -
          (4 * jetG1 b J x) * q3f e - (6 * jetG2 b J x) * q2f e -
          (4 * jetG3 b J x) * q1f e - (4 * ∑ i, b i * J.u3 x i) * q0f e
        have heq4 : (stateJetAt K a (s + 1)).u4 x =
            K.prob.vecMul (fun e =>
              (J.u4 x e * (1 + x * b e) - (x * ∑ i, b i * J.u4 x i) * q0f e + S4 e) /
                jetG0 b J x) := by
          funext j
          simp only [stateJetAt, nextStateJet, q4, jetW4, w4]
          congr 1; funext e
          change _ / jetG0 b J x = _ / jetG0 b J x
          rw [hG4_split]
          dsimp [S4, q0f, q1f, q2f, q3f]
          ring
        have hS4_l1 : l1 S4 ≤ 8 * A * U3 + 4 * g1 * Q3 + 6 * g2 * Q2 + 4 * g3 * Q1 := by
          have hs1 := l1_sub_le
            (fun e => 4 * (b e * J.u3 x e) - (4 * jetG1 b J x) * q3f e -
              (6 * jetG2 b J x) * q2f e - (4 * jetG3 b J x) * q1f e)
            (fun e => (4 * ∑ i, b i * J.u3 x i) * q0f e)
          have hs2 := l1_sub_le
            (fun e => 4 * (b e * J.u3 x e) - (4 * jetG1 b J x) * q3f e -
              (6 * jetG2 b J x) * q2f e)
            (fun e => (4 * jetG3 b J x) * q1f e)
          have hs3 := l1_sub_le
            (fun e => 4 * (b e * J.u3 x e) - (4 * jetG1 b J x) * q3f e)
            (fun e => (6 * jetG2 b J x) * q2f e)
          have hs4 := l1_sub_le (fun e => 4 * (b e * J.u3 x e))
            (fun e => (4 * jetG1 b J x) * q3f e)
          rw [l1_smul, hq0_l1, mul_one, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 4)] at hs1
          rw [l1_smul, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 4)] at hs2
          rw [l1_smul, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 6)] at hs3
          rw [l1_smul, l1_smul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 4),
            abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 4)] at hs4
          have hb3 := (l1_bu_le b (J.u3 x) A (ha s)).trans
            (mul_le_mul_of_nonneg_left hl1_u3 hA.le)
          have hd3 := (abs_dot_le_mul_l1 b (J.u3 x) A (ha s)).trans
            (mul_le_mul_of_nonneg_left hl1_u3 hA.le)
          have hp1 : 4 * |jetG1 b J x| * l1 q3f ≤ 4 * g1 * Q3 :=
            mul_le_mul (by linarith) hq3_l1 (l1_nonneg _) (by positivity)
          have hp2 : 6 * |jetG2 b J x| * l1 q2f ≤ 6 * g2 * Q2 :=
            mul_le_mul (by linarith) hq2_l1 (l1_nonneg _) (by positivity)
          have hp3 : 4 * |jetG3 b J x| * l1 q1f ≤ 4 * g3 * Q1 :=
            mul_le_mul (by linarith) hq1_l1 (l1_nonneg _) (by positivity)
          change l1 (fun e => 4 * (b e * J.u3 x e) - (4 * jetG1 b J x) * q3f e -
            (6 * jetG2 b J x) * q2f e - (4 * jetG3 b J x) * q1f e -
            (4 * ∑ i, b i * J.u3 x i) * q0f e) ≤ _
          linarith
        have hstep4 : gauge K m ((stateJetAt K a (s + 1)).u4 x) ≤
            (1 - 1 / (4 * (m : ℝ))) * gauge K m (J.u4 x) +
              2 * (m : ℝ) * (8 * A * U3 + 4 * g1 * Q3 + 6 * g2 * Q2 + 4 * g3 * Q1) := by
          rw [heq4]
          have hlin := gauge_projective_linear_step K m hm hblock b (J.u4 x) q0f S4
            (jetG0 b J x) x A hA (ha s) hzs.2.2.2.2 hq0_l1 hx hG_lo h1G
          have hmS := mul_le_mul_of_nonneg_left hS4_l1 (by positivity : (0 : ℝ) ≤ 2 * (m : ℝ))
          linarith
        exact contraction_fixed_point_step hm hu4 hstep4 rfl
  obtain ⟨hu1, hu2, hu3, hu4⟩ := hstate t
  refine ⟨⟨hu1, hu2, hu3, hu4⟩, ?_⟩
  let J := stateJetAt K a t
  let b := a t
  have hprob := projState_prob_and_gain_bounds K a A hA ha x hx_lt_half t
  have hzs := stateJetAt_zero_sum K a A hA ha t x hx_lt_half
  have hG_lo : 1 / 2 ≤ jetG0 b J x := by
    rw [jetG0_eq_projGain]; exact hprob.2.2.1.le
  have hG_pos : 0 < jetG0 b J x := by linarith
  have hu0_nn : ∀ e, 0 ≤ J.u0 x e := by
    intro e; rw [stateJetAt_u0_eq_projState]; exact hprob.1 e
  obtain ⟨_hl1_u0, _h1G, hG1, hG2, hG3, hG4⟩ := jetG_bounds K m hm b J x A hA
    (ha t) hx_le_half hu0_nn hzs.1 hzs.2.1 hzs.2.2.1 hzs.2.2.2.1 hzs.2.2.2.2
  let U1 := stateBound1 m A
  let U2 := stateBound2 m A
  let U3 := stateBound3 m A
  let U4 := stateBound4 m A
  let g1 := A + U1
  let g2 := 2 * A * U1 + U2
  let g3 := 3 * A * U2 + U3
  let g4 := 4 * A * U3 + U4
  have hU1_nn : 0 ≤ U1 := by dsimp [U1, stateBound1]; positivity
  have hU2_nn : 0 ≤ U2 := by dsimp [U2, stateBound2, stateBound1]; positivity
  have hU3_nn : 0 ≤ U3 := by dsimp [U3, stateBound3, stateBound2, stateBound1]; positivity
  have hU4_nn : 0 ≤ U4 := by
    dsimp [U4, stateBound4, stateBound3, stateBound2, stateBound1]; positivity
  have hg1_nn : 0 ≤ g1 := by dsimp [g1]; positivity
  have hg2_nn : 0 ≤ g2 := by dsimp [g2]; positivity
  have hg3_nn : 0 ≤ g3 := by dsimp [g3]; positivity
  have hg1 : |jetG1 b J x| ≤ g1 := by change |jetG1 b J x| ≤ A + U1; linarith
  have hg2 : |jetG2 b J x| ≤ g2 := by
    change |jetG2 b J x| ≤ 2 * A * U1 + U2
    have h1 : 2 * A * gauge K m (J.u1 x) ≤ 2 * A * U1 :=
      mul_le_mul_of_nonneg_left hu1 (by positivity)
    linarith
  have hg3 : |jetG3 b J x| ≤ g3 := by
    change |jetG3 b J x| ≤ 3 * A * U2 + U3
    have h1 : 3 * A * gauge K m (J.u2 x) ≤ 3 * A * U2 :=
      mul_le_mul_of_nonneg_left hu2 (by positivity)
    linarith
  have hg4 : |jetG4 b J x| ≤ g4 := by
    change |jetG4 b J x| ≤ 4 * A * U3 + U4
    have h1 : 4 * A * gauge K m (J.u3 x) ≤ 4 * A * U3 :=
      mul_le_mul_of_nonneg_left hu3 (by positivity)
    linarith
  have div_half_le : ∀ Y B : ℝ, |Y| ≤ B → |Y / jetG0 b J x| ≤ 2 * B := by
    intro Y B hY
    rw [abs_div, abs_of_pos hG_pos, div_le_iff₀ hG_pos]
    nlinarith [abs_nonneg Y]
  have hL1 : |logGain1 K a t x| ≤ logGain1Bound m A := by
    change |jetG1 b J x / jetG0 b J x| ≤ 2 * (A + U1)
    exact div_half_le _ (A + U1) hg1
  have hL2 : |logGain2 K a t x| ≤ logGain2Bound m A := by
    have hnum : |jetG2 b J x - jetG1 b J x * logGain1 K a t x| ≤
        g2 + g1 * logGain1Bound m A := by
      have ht := abs_sub_le (jetG2 b J x) 0 (jetG1 b J x * logGain1 K a t x)
      simp only [sub_zero, zero_sub, abs_neg, abs_mul] at ht
      have hp := mul_le_mul hg1 hL1 (abs_nonneg _) hg1_nn
      linarith
    change |(jetG2 b J x - jetG1 b J x * logGain1 K a t x) / jetG0 b J x| ≤
      2 * (g2 + g1 * logGain1Bound m A)
    exact div_half_le _ _ hnum
  have hL3 : |logGain3 K a t x| ≤ logGain3Bound m A := by
    have hnum : |jetG3 b J x - 2 * jetG1 b J x * logGain2 K a t x -
        jetG2 b J x * logGain1 K a t x| ≤
        g3 + 2 * g1 * logGain2Bound m A + g2 * logGain1Bound m A := by
      have ht1 := abs_sub_le (jetG3 b J x - 2 * jetG1 b J x * logGain2 K a t x) 0
        (jetG2 b J x * logGain1 K a t x)
      have ht2 := abs_sub_le (jetG3 b J x) 0 (2 * jetG1 b J x * logGain2 K a t x)
      simp only [sub_zero, zero_sub, abs_neg, abs_mul,
        abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] at ht1 ht2
      have hp1 := mul_le_mul (by linarith : 2 * |jetG1 b J x| ≤ 2 * g1) hL2 (abs_nonneg _)
        (by positivity)
      have hp2 := mul_le_mul hg2 hL1 (abs_nonneg _) hg2_nn
      linarith
    change |(jetG3 b J x - 2 * jetG1 b J x * logGain2 K a t x -
      jetG2 b J x * logGain1 K a t x) / jetG0 b J x| ≤
      2 * (g3 + 2 * g1 * logGain2Bound m A + g2 * logGain1Bound m A)
    exact div_half_le _ _ hnum
  have hL4 : |logGain4 K a t x| ≤ logGain4Bound m A := by
    have hnum : |jetG4 b J x - 3 * jetG1 b J x * logGain3 K a t x -
        3 * jetG2 b J x * logGain2 K a t x - jetG3 b J x * logGain1 K a t x| ≤
        g4 + 3 * g1 * logGain3Bound m A + 3 * g2 * logGain2Bound m A +
          g3 * logGain1Bound m A := by
      have ht1 := abs_sub_le
        (jetG4 b J x - 3 * jetG1 b J x * logGain3 K a t x - 3 * jetG2 b J x * logGain2 K a t x)
        0 (jetG3 b J x * logGain1 K a t x)
      have ht2 := abs_sub_le (jetG4 b J x - 3 * jetG1 b J x * logGain3 K a t x) 0
        (3 * jetG2 b J x * logGain2 K a t x)
      have ht3 := abs_sub_le (jetG4 b J x) 0 (3 * jetG1 b J x * logGain3 K a t x)
      simp only [sub_zero, zero_sub, abs_neg, abs_mul,
        abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 3)] at ht1 ht2 ht3
      have hp1 := mul_le_mul (by linarith : 3 * |jetG1 b J x| ≤ 3 * g1) hL3 (abs_nonneg _)
        (by positivity)
      have hp2 := mul_le_mul (by linarith : 3 * |jetG2 b J x| ≤ 3 * g2) hL2 (abs_nonneg _)
        (by positivity)
      have hp3 := mul_le_mul hg3 hL1 (abs_nonneg _) hg3_nn
      linarith
    change |(jetG4 b J x - 3 * jetG1 b J x * logGain3 K a t x -
      3 * jetG2 b J x * logGain2 K a t x - jetG3 b J x * logGain1 K a t x) / jetG0 b J x| ≤
      2 * (g4 + 3 * g1 * logGain3Bound m A + 3 * g2 * logGain2Bound m A +
        g3 * logGain1Bound m A)
    exact div_half_le _ _ hnum
  exact ⟨hL1, hL2, hL3, hL4⟩

end ProjectiveBounds

/-! ## Section 5: Expectation differentiation, `C^4` smoothness, symmetry, and corollaries -/

section ExpectationAndCorollaries

lemma contDiffOn_four_of_open_derivs {U : Set ℝ} {δ : ℝ} (hU : IsOpen U) (hsub : Icc (-δ) δ ⊆ U)
    {g g1 g2 g3 g4 : ℝ → ℝ}
    (hd0 : ∀ x ∈ U, HasDerivAt g (g1 x) x)
    (hd1 : ∀ x ∈ U, HasDerivAt g1 (g2 x) x)
    (hd2 : ∀ x ∈ U, HasDerivAt g2 (g3 x) x)
    (hd3 : ∀ x ∈ U, HasDerivAt g3 (g4 x) x)
    (hc4 : ContinuousOn g4 U) :
    ContDiffOn ℝ 4 g (Icc (-δ) δ) ∧
    (∀ y ∈ Icc (-δ) δ, deriv g y = g1 y) ∧
    (∀ y ∈ Icc (-δ) δ, iteratedDeriv 3 g y = g3 y) ∧
    (∀ y ∈ Icc (-δ) δ, iteratedDeriv 4 g y = g4 y) := by
  open scoped ContDiff in
  have heq1 : Set.EqOn (deriv g) g1 U := fun x hx => (hd0 x hx).deriv
  have heq2 : Set.EqOn (deriv g1) g2 U := fun x hx => (hd1 x hx).deriv
  have heq3 : Set.EqOn (deriv g2) g3 U := fun x hx => (hd2 x hx).deriv
  have heq4 : Set.EqOn (deriv g3) g4 U := fun x hx => (hd3 x hx).deriv
  have hc_g3 : ContDiffOn ℝ 1 g3 U := by
    open scoped ContDiff in
    rw [show (1 : WithTop ℕ∞) = 0 + 1 from rfl, contDiffOn_succ_iff_deriv_of_isOpen hU]
    exact ⟨fun x hx => (hd3 x hx).differentiableAt.differentiableWithinAt,
      by simp, contDiffOn_zero.mpr (hc4.congr fun x hx => heq4 hx)⟩
  have hc_g2 : ContDiffOn ℝ 2 g2 U := by
    open scoped ContDiff in
    rw [show (2 : WithTop ℕ∞) = 1 + 1 from rfl, contDiffOn_succ_iff_deriv_of_isOpen hU]
    exact ⟨fun x hx => (hd2 x hx).differentiableAt.differentiableWithinAt,
      by simp, hc_g3.congr fun x hx => heq3 hx⟩
  have hc_g1 : ContDiffOn ℝ 3 g1 U := by
    open scoped ContDiff in
    rw [show (3 : WithTop ℕ∞) = 2 + 1 from rfl, contDiffOn_succ_iff_deriv_of_isOpen hU]
    exact ⟨fun x hx => (hd1 x hx).differentiableAt.differentiableWithinAt,
      by simp, hc_g2.congr fun x hx => heq2 hx⟩
  have hc_g0 : ContDiffOn ℝ 4 g U := by
    open scoped ContDiff in
    rw [show (4 : WithTop ℕ∞) = 3 + 1 from rfl, contDiffOn_succ_iff_deriv_of_isOpen hU]
    exact ⟨fun x hx => (hd0 x hx).differentiableAt.differentiableWithinAt,
      by simp, hc_g1.congr fun x hx => heq1 hx⟩
  have hit2 : ∀ x ∈ U, iteratedDeriv 2 g x = g2 x := by
    intro x hx
    have h1 : deriv g =ᶠ[𝓝 x] g1 :=
      Filter.eventuallyEq_of_mem (hU.mem_nhds hx) fun z hz => heq1 hz
    rw [iteratedDeriv_succ, iteratedDeriv_one, h1.deriv_eq, heq2 hx]
  have hit3 : ∀ x ∈ U, iteratedDeriv 3 g x = g3 x := by
    intro x hx
    have h2 : iteratedDeriv 2 g =ᶠ[𝓝 x] g2 :=
      Filter.eventuallyEq_of_mem (hU.mem_nhds hx) fun z hz => hit2 z hz
    rw [iteratedDeriv_succ, h2.deriv_eq, heq3 hx]
  have hit4 : ∀ x ∈ U, iteratedDeriv 4 g x = g4 x := by
    intro x hx
    have h3 : iteratedDeriv 3 g =ᶠ[𝓝 x] g3 :=
      Filter.eventuallyEq_of_mem (hU.mem_nhds hx) fun z hz => hit3 z hz
    rw [iteratedDeriv_succ, h3.deriv_eq, heq4 hx]
  exact ⟨hc_g0.mono hsub,
    fun y hy => heq1 (hsub hy),
    fun y hy => hit3 y (hsub hy),
    fun y hy => hit4 y (hsub hy)⟩

lemma continuousOn_sum_finset {α : Type*} (s : Finset α) {f : α → ℝ → ℝ} {U : Set ℝ}
    (hf : ∀ a ∈ s, ContinuousOn (f a) U) :
    ContinuousOn (fun x => ∑ a ∈ s, f a x) U := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [continuousOn_const]
  | @insert a s ha ih =>
    simp_rw [Finset.sum_insert ha]
    exact (hf a (Finset.mem_insert_self a s)).add
      (ih fun b hb => hf b (Finset.mem_insert_of_mem hb))

lemma aestronglyMeasurable_sum_finset {α Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (s : Finset α) {f : α → Ω → ℝ}
    (hf : ∀ a ∈ s, AEStronglyMeasurable (f a) μ) :
    AEStronglyMeasurable (fun ω => ∑ a ∈ s, f a ω) μ := by
  convert Finset.aestronglyMeasurable_sum s hf using 1
  funext ω
  simp only [Finset.sum_apply]

lemma stateJetAt_meas {E Ω : Type*} [Fintype E] [MeasurableSpace Ω]
    (μ : Measure Ω) (K : Chain E) (T : Ω → Ω) (ξ : E → Ω → ℝ)
    (hmeas : ∀ t e, AEStronglyMeasurable (fun ω => timeReward T ξ ω t e) μ)
    (t : ℕ) (x : ℝ) :
    let J := fun ω => stateJetAt K (timeReward T ξ ω) t
    (∀ e, AEStronglyMeasurable (fun ω => (J ω).u0 x e) μ) ∧
    (∀ e, AEStronglyMeasurable (fun ω => (J ω).u1 x e) μ) ∧
    (∀ e, AEStronglyMeasurable (fun ω => (J ω).u2 x e) μ) ∧
    (∀ e, AEStronglyMeasurable (fun ω => (J ω).u3 x e) μ) ∧
    (∀ e, AEStronglyMeasurable (fun ω => (J ω).u4 x e) μ) := by
  induction t with
  | zero =>
    exact ⟨fun _ => aestronglyMeasurable_const,
      fun _ => aestronglyMeasurable_const,
      fun _ => aestronglyMeasurable_const,
      fun _ => aestronglyMeasurable_const,
      fun _ => aestronglyMeasurable_const⟩
  | succ t ih =>
    obtain ⟨hm0, hm1, hm2, hm3, hm4⟩ := ih
    let J := fun ω => stateJetAt K (timeReward T ξ ω) t
    let b := fun ω e => timeReward T ξ ω t e
    have hfac (e : E) : AEStronglyMeasurable (fun ω => 1 + x * b ω e) μ :=
      aestronglyMeasurable_const.add ((hmeas t e).const_mul x)
    have hw0m (e : E) : AEStronglyMeasurable (fun ω => jetW0 (b ω) (J ω) e x) μ :=
      (hm0 e).mul (hfac e)
    have hw1m (e : E) : AEStronglyMeasurable (fun ω => jetW1 (b ω) (J ω) e x) μ :=
      ((hm1 e).mul (hfac e)).add ((hmeas t e).mul (hm0 e))
    have hw2m (e : E) : AEStronglyMeasurable (fun ω => jetW2 (b ω) (J ω) e x) μ :=
      ((hm2 e).mul (hfac e)).add (((hmeas t e).const_mul 2).mul (hm1 e))
    have hw3m (e : E) : AEStronglyMeasurable (fun ω => jetW3 (b ω) (J ω) e x) μ :=
      ((hm3 e).mul (hfac e)).add (((hmeas t e).const_mul 3).mul (hm2 e))
    have hw4m (e : E) : AEStronglyMeasurable (fun ω => jetW4 (b ω) (J ω) e x) μ :=
      ((hm4 e).mul (hfac e)).add (((hmeas t e).const_mul 4).mul (hm3 e))
    have hg0m : AEStronglyMeasurable (fun ω => jetG0 (b ω) (J ω) x) μ :=
      aestronglyMeasurable_sum_finset μ Finset.univ fun e _ => hw0m e
    have hg1m : AEStronglyMeasurable (fun ω => jetG1 (b ω) (J ω) x) μ :=
      aestronglyMeasurable_sum_finset μ Finset.univ fun e _ => hw1m e
    have hg2m : AEStronglyMeasurable (fun ω => jetG2 (b ω) (J ω) x) μ :=
      aestronglyMeasurable_sum_finset μ Finset.univ fun e _ => hw2m e
    have hg3m : AEStronglyMeasurable (fun ω => jetG3 (b ω) (J ω) x) μ :=
      aestronglyMeasurable_sum_finset μ Finset.univ fun e _ => hw3m e
    have hg4m : AEStronglyMeasurable (fun ω => jetG4 (b ω) (J ω) x) μ :=
      aestronglyMeasurable_sum_finset μ Finset.univ fun e _ => hw4m e
    have hq0m (e : E) : AEStronglyMeasurable
        (fun ω => q0 (jetW0 (b ω) (J ω) e) (jetG0 (b ω) (J ω)) x) μ :=
      ((hw0m e).aemeasurable.div hg0m.aemeasurable).aestronglyMeasurable
    have hq1m (e : E) : AEStronglyMeasurable
        (fun ω => q1 (jetW0 (b ω) (J ω) e) (jetW1 (b ω) (J ω) e)
          (jetG0 (b ω) (J ω)) (jetG1 (b ω) (J ω)) x) μ :=
      (((hw1m e).sub (hg1m.mul (hq0m e))).aemeasurable.div hg0m.aemeasurable).aestronglyMeasurable
    have hq2m (e : E) : AEStronglyMeasurable
        (fun ω => q2 (jetW0 (b ω) (J ω) e) (jetW1 (b ω) (J ω) e) (jetW2 (b ω) (J ω) e)
          (jetG0 (b ω) (J ω)) (jetG1 (b ω) (J ω)) (jetG2 (b ω) (J ω)) x) μ :=
      ((((hw2m e).sub ((hg1m.const_mul 2).mul (hq1m e))).sub (hg2m.mul (hq0m e))).aemeasurable.div
        hg0m.aemeasurable).aestronglyMeasurable
    have hq3m (e : E) : AEStronglyMeasurable
        (fun ω => q3 (jetW0 (b ω) (J ω) e) (jetW1 (b ω) (J ω) e) (jetW2 (b ω) (J ω) e)
          (jetW3 (b ω) (J ω) e) (jetG0 (b ω) (J ω)) (jetG1 (b ω) (J ω)) (jetG2 (b ω) (J ω))
          (jetG3 (b ω) (J ω)) x) μ :=
      (((((hw3m e).sub ((hg1m.const_mul 3).mul (hq2m e))).sub
        ((hg2m.const_mul 3).mul (hq1m e))).sub (hg3m.mul (hq0m e))).aemeasurable.div
          hg0m.aemeasurable).aestronglyMeasurable
    have hq4m (e : E) : AEStronglyMeasurable
        (fun ω => q4 (jetW0 (b ω) (J ω) e) (jetW1 (b ω) (J ω) e) (jetW2 (b ω) (J ω) e)
          (jetW3 (b ω) (J ω) e) (jetW4 (b ω) (J ω) e) (jetG0 (b ω) (J ω)) (jetG1 (b ω) (J ω))
          (jetG2 (b ω) (J ω)) (jetG3 (b ω) (J ω)) (jetG4 (b ω) (J ω)) x) μ :=
      ((((((hw4m e).sub ((hg1m.const_mul 4).mul (hq3m e))).sub
        ((hg2m.const_mul 6).mul (hq2m e))).sub ((hg3m.const_mul 4).mul (hq1m e))).sub
          (hg4m.mul (hq0m e))).aemeasurable.div hg0m.aemeasurable).aestronglyMeasurable
    have vecMul_meas : ∀ f : E → Ω → ℝ, (∀ e, AEStronglyMeasurable (f e) μ) →
        ∀ j, AEStronglyMeasurable (fun ω => K.prob.vecMul (fun e => f e ω) j) μ := by
      intro f hf j
      exact aestronglyMeasurable_sum_finset μ Finset.univ fun i _ => (hf i).mul_const (K.prob i j)
    exact ⟨vecMul_meas _ hq0m, vecMul_meas _ hq1m, vecMul_meas _ hq2m,
      vecMul_meas _ hq3m, vecMul_meas _ hq4m⟩

lemma logGain_meas_and_cont {E Ω : Type*} [Fintype E] [MeasurableSpace Ω]
    (μ : Measure Ω) (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (ξ : E → Ω → ℝ) (hξ : ∀ e, MemLp (ξ e) 2 μ)
    (A : ℝ) (hA : 0 < A) (hbound : ∀ e ω, |ξ e ω| ≤ A) (t : ℕ) :
    let U := Ioo (-(1 / (2 * A))) (1 / (2 * A))
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => Real.log (projGain K (timeReward T ξ ω) t x)) μ) ∧
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => logGain1 K (timeReward T ξ ω) t x) μ) ∧
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => logGain2 K (timeReward T ξ ω) t x) μ) ∧
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => logGain3 K (timeReward T ξ ω) t x) μ) ∧
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => logGain4 K (timeReward T ξ ω) t x) μ) ∧
    (∀ ω, ContinuousOn (logGain4 K (timeReward T ξ ω) t) U) := by
  intro U
  have hmeas_tr : ∀ s e, AEStronglyMeasurable (fun ω => timeReward T ξ ω s e) μ :=
    fun s e => ((hξ e).comp_measurePreserving (hT.iterate s)).aestronglyMeasurable
  have hjet_m (x : ℝ) := stateJetAt_meas μ K T ξ hmeas_tr t x
  let J := fun ω => stateJetAt K (timeReward T ξ ω) t
  let b := fun ω e => timeReward T ξ ω t e
  have hg_m (x : ℝ) :
      AEStronglyMeasurable (fun ω => jetG0 (b ω) (J ω) x) μ ∧
      AEStronglyMeasurable (fun ω => jetG1 (b ω) (J ω) x) μ ∧
      AEStronglyMeasurable (fun ω => jetG2 (b ω) (J ω) x) μ ∧
      AEStronglyMeasurable (fun ω => jetG3 (b ω) (J ω) x) μ ∧
      AEStronglyMeasurable (fun ω => jetG4 (b ω) (J ω) x) μ := by
    obtain ⟨hm0, hm1, hm2, hm3, hm4⟩ := hjet_m x
    have hfac (e : E) : AEStronglyMeasurable (fun ω => 1 + x * b ω e) μ :=
      aestronglyMeasurable_const.add ((hmeas_tr t e).const_mul x)
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · exact aestronglyMeasurable_sum_finset μ Finset.univ fun e _ => (hm0 e).mul (hfac e)
    · exact aestronglyMeasurable_sum_finset μ Finset.univ fun e _ =>
        ((hm1 e).mul (hfac e)).add ((hmeas_tr t e).mul (hm0 e))
    · exact aestronglyMeasurable_sum_finset μ Finset.univ fun e _ =>
        ((hm2 e).mul (hfac e)).add (((hmeas_tr t e).const_mul 2).mul (hm1 e))
    · exact aestronglyMeasurable_sum_finset μ Finset.univ fun e _ =>
        ((hm3 e).mul (hfac e)).add (((hmeas_tr t e).const_mul 3).mul (hm2 e))
    · exact aestronglyMeasurable_sum_finset μ Finset.univ fun e _ =>
        ((hm4 e).mul (hfac e)).add (((hmeas_tr t e).const_mul 4).mul (hm3 e))
  have hl1_m (x : ℝ) : AEStronglyMeasurable (fun ω => logGain1 K (timeReward T ξ ω) t x) μ :=
    (((hg_m x).2.1.aemeasurable).div (hg_m x).1.aemeasurable).aestronglyMeasurable
  have hl2_m (x : ℝ) : AEStronglyMeasurable (fun ω => logGain2 K (timeReward T ξ ω) t x) μ :=
    ((((hg_m x).2.2.1.sub ((hg_m x).2.1.mul (hl1_m x))).aemeasurable).div
      (hg_m x).1.aemeasurable).aestronglyMeasurable
  have hl3_m (x : ℝ) : AEStronglyMeasurable (fun ω => logGain3 K (timeReward T ξ ω) t x) μ :=
    (((((hg_m x).2.2.2.1.sub (((hg_m x).2.1.const_mul 2).mul (hl2_m x))).sub
      ((hg_m x).2.2.1.mul (hl1_m x))).aemeasurable).div (hg_m x).1.aemeasurable).aestronglyMeasurable
  have hl4_m (x : ℝ) : AEStronglyMeasurable (fun ω => logGain4 K (timeReward T ξ ω) t x) μ :=
    ((((((hg_m x).2.2.2.2.sub (((hg_m x).2.1.const_mul 3).mul (hl3_m x))).sub
      (((hg_m x).2.2.1.const_mul 3).mul (hl2_m x))).sub
        ((hg_m x).2.2.2.1.mul (hl1_m x))).aemeasurable).div
          (hg_m x).1.aemeasurable).aestronglyMeasurable
  have hlog_m (x : ℝ) :
      AEStronglyMeasurable (fun ω => Real.log (projGain K (timeReward T ξ ω) t x)) μ := by
    have hfun : (fun ω => Real.log (projGain K (timeReward T ξ ω) t x)) =
        fun ω => Real.log (jetG0 (b ω) (J ω) x) := by
      funext ω
      exact congrArg Real.log (jetG0_eq_projGain K (timeReward T ξ ω) t x).symm
    rw [hfun]
    exact ((hg_m x).1.aemeasurable.log).aestronglyMeasurable
  -- Pathwise continuity of `logGain4` on `U` via induction on `s` for `stateJetAt.u0..u4`
  have hcont_all (ω : Ω) : ContinuousOn (logGain4 K (timeReward T ξ ω) t) U := by
    have haω : ∀ s e, |timeReward T ξ ω s e| ≤ A := fun s e => hbound e _
    have hstate_cont : ∀ s : ℕ,
        (∀ e, ContinuousOn (fun x => (stateJetAt K (timeReward T ξ ω) s).u0 x e) U) ∧
        (∀ e, ContinuousOn (fun x => (stateJetAt K (timeReward T ξ ω) s).u1 x e) U) ∧
        (∀ e, ContinuousOn (fun x => (stateJetAt K (timeReward T ξ ω) s).u2 x e) U) ∧
        (∀ e, ContinuousOn (fun x => (stateJetAt K (timeReward T ξ ω) s).u3 x e) U) ∧
        (∀ e, ContinuousOn (fun x => (stateJetAt K (timeReward T ξ ω) s).u4 x e) U) := by
      intro s
      induction s with
      | zero =>
        exact ⟨fun _ => continuousOn_const, fun _ => continuousOn_const,
          fun _ => continuousOn_const, fun _ => continuousOn_const,
          fun _ => continuousOn_const⟩
      | succ s ih =>
        obtain ⟨hc0, hc1, hc2, hc3, hc4⟩ := ih
        let Js := stateJetAt K (timeReward T ξ ω) s
        let bs := timeReward T ξ ω s
        have hG_ne : ∀ x ∈ U, jetG0 bs Js x ≠ 0 := by
          intro x hx
          rw [jetG0_eq_projGain]
          linarith [(projState_prob_and_gain_bounds K (timeReward T ξ ω) A hA haω x hx s).2.2.1]
        have hfac_c (e : E) : ContinuousOn (fun x => 1 + x * bs e) U :=
          continuousOn_const.add (continuousOn_id.mul continuousOn_const)
        have hw0_c (e : E) : ContinuousOn (jetW0 bs Js e) U := (hc0 e).mul (hfac_c e)
        have hw1_c (e : E) : ContinuousOn (jetW1 bs Js e) U :=
          ((hc1 e).mul (hfac_c e)).add (continuousOn_const.mul (hc0 e))
        have hw2_c (e : E) : ContinuousOn (jetW2 bs Js e) U :=
          ((hc2 e).mul (hfac_c e)).add (continuousOn_const.mul (hc1 e))
        have hw3_c (e : E) : ContinuousOn (jetW3 bs Js e) U :=
          ((hc3 e).mul (hfac_c e)).add (continuousOn_const.mul (hc2 e))
        have hw4_c (e : E) : ContinuousOn (jetW4 bs Js e) U :=
          ((hc4 e).mul (hfac_c e)).add (continuousOn_const.mul (hc3 e))
        have hg0_c : ContinuousOn (jetG0 bs Js) U :=
          continuousOn_sum_finset _ fun e _ => hw0_c e
        have hg1_c : ContinuousOn (jetG1 bs Js) U :=
          continuousOn_sum_finset _ fun e _ => hw1_c e
        have hg2_c : ContinuousOn (jetG2 bs Js) U :=
          continuousOn_sum_finset _ fun e _ => hw2_c e
        have hg3_c : ContinuousOn (jetG3 bs Js) U :=
          continuousOn_sum_finset _ fun e _ => hw3_c e
        have hg4_c : ContinuousOn (jetG4 bs Js) U :=
          continuousOn_sum_finset _ fun e _ => hw4_c e
        have hq0_c (e : E) : ContinuousOn (q0 (jetW0 bs Js e) (jetG0 bs Js)) U :=
          (hw0_c e).div hg0_c hG_ne
        have hq1_c (e : E) : ContinuousOn
            (q1 (jetW0 bs Js e) (jetW1 bs Js e) (jetG0 bs Js) (jetG1 bs Js)) U :=
          ((hw1_c e).sub (hg1_c.mul (hq0_c e))).div hg0_c hG_ne
        have hq2_c (e : E) : ContinuousOn
            (q2 (jetW0 bs Js e) (jetW1 bs Js e) (jetW2 bs Js e)
              (jetG0 bs Js) (jetG1 bs Js) (jetG2 bs Js)) U :=
          (((hw2_c e).sub ((continuousOn_const.mul hg1_c).mul (hq1_c e))).sub
            (hg2_c.mul (hq0_c e))).div hg0_c hG_ne
        have hq3_c (e : E) : ContinuousOn
            (q3 (jetW0 bs Js e) (jetW1 bs Js e) (jetW2 bs Js e) (jetW3 bs Js e)
              (jetG0 bs Js) (jetG1 bs Js) (jetG2 bs Js) (jetG3 bs Js)) U :=
          ((((hw3_c e).sub ((continuousOn_const.mul hg1_c).mul (hq2_c e))).sub
            ((continuousOn_const.mul hg2_c).mul (hq1_c e))).sub
              (hg3_c.mul (hq0_c e))).div hg0_c hG_ne
        have hq4_c (e : E) : ContinuousOn
            (q4 (jetW0 bs Js e) (jetW1 bs Js e) (jetW2 bs Js e) (jetW3 bs Js e) (jetW4 bs Js e)
              (jetG0 bs Js) (jetG1 bs Js) (jetG2 bs Js) (jetG3 bs Js) (jetG4 bs Js)) U :=
          (((((hw4_c e).sub ((continuousOn_const.mul hg1_c).mul (hq3_c e))).sub
            ((continuousOn_const.mul hg2_c).mul (hq2_c e))).sub
              ((continuousOn_const.mul hg3_c).mul (hq1_c e))).sub
                (hg4_c.mul (hq0_c e))).div hg0_c hG_ne
        have vecMul_c : ∀ f : E → ℝ → ℝ, (∀ e, ContinuousOn (f e) U) →
            ∀ j, ContinuousOn (fun x => K.prob.vecMul (fun e => f e x) j) U := by
          intro f hf j
          exact continuousOn_sum_finset _ fun i _ => (hf i).mul continuousOn_const
        exact ⟨vecMul_c _ hq0_c, vecMul_c _ hq1_c, vecMul_c _ hq2_c,
          vecMul_c _ hq3_c, vecMul_c _ hq4_c⟩
    obtain ⟨hc0, hc1, hc2, hc3, hc4⟩ := hstate_cont t
    let Jt := stateJetAt K (timeReward T ξ ω) t
    let bt := timeReward T ξ ω t
    have hG_ne : ∀ x ∈ U, jetG0 bt Jt x ≠ 0 := by
      intro x hx
      rw [jetG0_eq_projGain]
      linarith [(projState_prob_and_gain_bounds K (timeReward T ξ ω) A hA haω x hx t).2.2.1]
    have hfac_c (e : E) : ContinuousOn (fun x => 1 + x * bt e) U :=
      continuousOn_const.add (continuousOn_id.mul continuousOn_const)
    have hg0_c : ContinuousOn (jetG0 bt Jt) U :=
      continuousOn_sum_finset _ fun e _ => (hc0 e).mul (hfac_c e)
    have hg1_c : ContinuousOn (jetG1 bt Jt) U :=
      continuousOn_sum_finset _ fun e _ =>
        ((hc1 e).mul (hfac_c e)).add (continuousOn_const.mul (hc0 e))
    have hg2_c : ContinuousOn (jetG2 bt Jt) U :=
      continuousOn_sum_finset _ fun e _ =>
        ((hc2 e).mul (hfac_c e)).add (continuousOn_const.mul (hc1 e))
    have hg3_c : ContinuousOn (jetG3 bt Jt) U :=
      continuousOn_sum_finset _ fun e _ =>
        ((hc3 e).mul (hfac_c e)).add (continuousOn_const.mul (hc2 e))
    have hg4_c : ContinuousOn (jetG4 bt Jt) U :=
      continuousOn_sum_finset _ fun e _ =>
        ((hc4 e).mul (hfac_c e)).add (continuousOn_const.mul (hc3 e))
    have hl1_c : ContinuousOn (logGain1 K (timeReward T ξ ω) t) U := hg1_c.div hg0_c hG_ne
    have hl2_c : ContinuousOn (logGain2 K (timeReward T ξ ω) t) U :=
      (hg2_c.sub (hg1_c.mul hl1_c)).div hg0_c hG_ne
    have hl3_c : ContinuousOn (logGain3 K (timeReward T ξ ω) t) U :=
      ((hg3_c.sub ((continuousOn_const.mul hg1_c).mul hl2_c)).sub (hg2_c.mul hl1_c)).div hg0_c hG_ne
    exact (((hg4_c.sub ((continuousOn_const.mul hg1_c).mul hl3_c)).sub
      ((continuousOn_const.mul hg2_c).mul hl2_c)).sub (hg3_c.mul hl1_c)).div hg0_c hG_ne
  exact ⟨fun x _ => hlog_m x, fun x _ => hl1_m x, fun x _ => hl2_m x,
    fun x _ => hl3_m x, fun x _ => hl4_m x, hcont_all⟩

/-- Complete `C^4`, centered first-derivative, third-derivative (`part1:82`), and uniform
fourth-derivative (`part1:81`, `part1:69`) bounds for the actual `finiteGrowth ... .additive N`. -/
theorem additive_finiteGrowth_projective_controls {E Ω : Type*}
    [Fintype E] [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (m : ℕ) (hm : 0 < m)
    (hblock : ∀ z : E → ℝ, (∑ i, z i) = 0 → l1 ((K.prob ^ m).vecMul z) ≤ l1 z / 2)
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (ξ : E → Ω → ℝ) (hξ : ∀ e, MemLp (ξ e) 2 μ)
    (hcenter : ∀ e, ∫ ω, ξ e ω ∂μ = 0)
    (A : ℝ) (hA : 0 < A) (hbound : ∀ e ω, |ξ e ω| ≤ A) (L : ℝ) :
    let δ := 1 / (48 * (m : ℝ) ^ 2 * A)
    let B3 := logGain3Bound m A
    let M4 := logGain4Bound m A
    0 < δ ∧
    (∀ N, ContDiffOn ℝ 4 (finiteGrowth μ K T ξ L .additive N) (Icc (-δ) δ)) ∧
    (∀ N, deriv (finiteGrowth μ K T ξ L .additive N) 0 = 0) ∧
    (∀ N, |iteratedDeriv 3 (finiteGrowth μ K T ξ L .additive N) 0| ≤ B3) ∧
    (∀ N y, y ∈ Icc (-δ) δ →
      |iteratedDeriv 4 (finiteGrowth μ K T ξ L .additive N) y| ≤ M4) := by
  intro δ B3 M4
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hδ : 0 < δ := by dsimp [δ]; positivity
  let Ropen : ℝ := 1 / (24 * (m : ℝ) ^ 2 * A)
  have hδ_lt_R : δ < Ropen := by
    have h24_lt_48 : 24 * (m : ℝ) ^ 2 * A < 48 * (m : ℝ) ^ 2 * A := by
      have hp : 0 < (m : ℝ) ^ 2 * A := by positivity
      linarith
    dsimp [δ, Ropen]
    exact one_div_lt_one_div_of_lt (by positivity) h24_lt_48
  have hR_lt_half : Ropen < 1 / (2 * A) := by
    have h2_lt_24 : 2 * A < 24 * (m : ℝ) ^ 2 * A := by
      nlinarith [sq_nonneg ((m : ℝ) - 1)]
    dsimp [Ropen]
    exact one_div_lt_one_div_of_lt (by positivity) h2_lt_24
  let U : Set ℝ := Ioo (-Ropen) Ropen
  have hU_open : IsOpen U := isOpen_Ioo
  have hsub_U : Icc (-δ) δ ⊆ U := fun x hx => ⟨by linarith [hx.1], by linarith [hx.2]⟩
  have hU_sub_half : U ⊆ Ioo (-(1 / (2 * A))) (1 / (2 * A)) :=
    fun x hx => ⟨by linarith [hx.1], by linarith [hx.2]⟩
  have h0_mem_Icc : (0 : ℝ) ∈ Icc (-δ) δ := ⟨by linarith, hδ.le⟩
  have haω (ω : Ω) : ∀ t e, |timeReward T ξ ω t e| ≤ A := fun t e => hbound e _
  -- Step-level expected log gain smoothness and bounds for each `t : ℕ`
  have hstep_smooth (t : ℕ) :
      let E0 := fun x => ∫ ω, Real.log (projGain K (timeReward T ξ ω) t x) ∂μ
      let E1 := fun x => ∫ ω, logGain1 K (timeReward T ξ ω) t x ∂μ
      let E2 := fun x => ∫ ω, logGain2 K (timeReward T ξ ω) t x ∂μ
      let E3 := fun x => ∫ ω, logGain3 K (timeReward T ξ ω) t x ∂μ
      let E4 := fun x => ∫ ω, logGain4 K (timeReward T ξ ω) t x ∂μ
      (∀ x ∈ U, HasDerivAt E0 (E1 x) x) ∧
      (∀ x ∈ U, HasDerivAt E1 (E2 x) x) ∧
      (∀ x ∈ U, HasDerivAt E2 (E3 x) x) ∧
      (∀ x ∈ U, HasDerivAt E3 (E4 x) x) ∧
      ContinuousOn E4 U ∧
      E1 0 = 0 ∧
      |E3 0| ≤ B3 ∧
      (∀ y ∈ Icc (-δ) δ, |E4 y| ≤ M4) := by
    intro E0 E1 E2 E3 E4
    obtain ⟨hm0, hm1_m, hm2_m, hm3_m, hm4_m, hcont4⟩ :=
      logGain_meas_and_cont μ K T hT ξ hξ A hA hbound t
    have hpt_bounds : ∀ ω, ∀ x ∈ U,
        |Real.log (projGain K (timeReward T ξ ω) t x)| ≤ |Real.log (1 / 2)| + |Real.log 2| ∧
        |logGain1 K (timeReward T ξ ω) t x| ≤ logGain1Bound m A ∧
        |logGain2 K (timeReward T ξ ω) t x| ≤ logGain2Bound m A ∧
        |logGain3 K (timeReward T ξ ω) t x| ≤ B3 ∧
        |logGain4 K (timeReward T ξ ω) t x| ≤ M4 := by
      intro ω x hx
      have hx_le : |x| ≤ Ropen := abs_le.mpr ⟨hx.1.le, hx.2.le⟩
      have hp := (projective_state_and_logGain_bounds K m hm hblock (timeReward T ξ ω)
        A hA (haω ω) x hx_le t).2
      have hgb := (projState_prob_and_gain_bounds K (timeReward T ξ ω) A hA (haω ω)
        x (hU_sub_half hx) t).2.2
      have hgp : 0 < projGain K (timeReward T ξ ω) t x := by linarith [hgb.1]
      have hlo := Real.log_le_log (by norm_num : (0 : ℝ) < 1 / 2) hgb.1.le
      have hhi := Real.log_le_log hgp hgb.2.le
      have hlog_b : |Real.log (projGain K (timeReward T ξ ω) t x)| ≤
          |Real.log (1 / 2)| + |Real.log 2| := by
        apply abs_le.mpr
        constructor <;> linarith [neg_abs_le (Real.log (1 / 2)), le_abs_self (Real.log 2),
          abs_nonneg (Real.log (1 / 2)), abs_nonneg (Real.log 2)]
      exact ⟨hlog_b, hp.1, hp.2.1, hp.2.2.1, hp.2.2.2⟩
    have hint0 : ∀ x ∈ U, Integrable (fun ω => Real.log (projGain K (timeReward T ξ ω) t x)) μ := by
      intro x hx
      refine Integrable.mono' (integrable_const (|Real.log (1 / 2)| + |Real.log 2|))
        (hm0 x (hU_sub_half hx)) (Eventually.of_forall fun ω => ?_)
      rw [Real.norm_eq_abs]; exact (hpt_bounds ω x hx).1
    have hd0 : ∀ x ∈ U, HasDerivAt E0 (E1 x) x := by
      intro x hx
      exact (hasDerivAt_integral_of_dominated_loc_of_deriv_le
        (μ := μ) (s := U) (bound := fun _ => logGain1Bound m A)
        (hU_open.mem_nhds hx)
        (Filter.eventually_of_mem (hU_open.mem_nhds hx) fun y hy => hm0 y (hU_sub_half hy))
        (hint0 x hx) (hm1_m x (hU_sub_half hx))
        (Eventually.of_forall fun ω y hy => by
          rw [Real.norm_eq_abs]; exact (hpt_bounds ω y hy).2.1)
        (integrable_const _)
        (Eventually.of_forall fun ω y hy =>
          (logGain_hasDerivAt K (timeReward T ξ ω) A hA (haω ω) t y (hU_sub_half hy)).1)).2
    have hint1 : ∀ x ∈ U, Integrable (fun ω => logGain1 K (timeReward T ξ ω) t x) μ := by
      intro x hx
      refine Integrable.mono' (integrable_const (logGain1Bound m A))
        (hm1_m x (hU_sub_half hx)) (Eventually.of_forall fun ω => ?_)
      rw [Real.norm_eq_abs]; exact (hpt_bounds ω x hx).2.1
    have hd1 : ∀ x ∈ U, HasDerivAt E1 (E2 x) x := by
      intro x hx
      exact (hasDerivAt_integral_of_dominated_loc_of_deriv_le
        (μ := μ) (s := U) (bound := fun _ => logGain2Bound m A)
        (hU_open.mem_nhds hx)
        (Filter.eventually_of_mem (hU_open.mem_nhds hx) fun y hy => hm1_m y (hU_sub_half hy))
        (hint1 x hx) (hm2_m x (hU_sub_half hx))
        (Eventually.of_forall fun ω y hy => by
          rw [Real.norm_eq_abs]; exact (hpt_bounds ω y hy).2.2.1)
        (integrable_const _)
        (Eventually.of_forall fun ω y hy =>
          (logGain_hasDerivAt K (timeReward T ξ ω) A hA (haω ω) t y (hU_sub_half hy)).2.1)).2
    have hint2 : ∀ x ∈ U, Integrable (fun ω => logGain2 K (timeReward T ξ ω) t x) μ := by
      intro x hx
      refine Integrable.mono' (integrable_const (logGain2Bound m A))
        (hm2_m x (hU_sub_half hx)) (Eventually.of_forall fun ω => ?_)
      rw [Real.norm_eq_abs]; exact (hpt_bounds ω x hx).2.2.1
    have hd2 : ∀ x ∈ U, HasDerivAt E2 (E3 x) x := by
      intro x hx
      exact (hasDerivAt_integral_of_dominated_loc_of_deriv_le
        (μ := μ) (s := U) (bound := fun _ => B3)
        (hU_open.mem_nhds hx)
        (Filter.eventually_of_mem (hU_open.mem_nhds hx) fun y hy => hm2_m y (hU_sub_half hy))
        (hint2 x hx) (hm3_m x (hU_sub_half hx))
        (Eventually.of_forall fun ω y hy => by
          rw [Real.norm_eq_abs]; exact (hpt_bounds ω y hy).2.2.2.1)
        (integrable_const _)
        (Eventually.of_forall fun ω y hy =>
          (logGain_hasDerivAt K (timeReward T ξ ω) A hA (haω ω) t y (hU_sub_half hy)).2.2.1)).2
    have hint3 : ∀ x ∈ U, Integrable (fun ω => logGain3 K (timeReward T ξ ω) t x) μ := by
      intro x hx
      refine Integrable.mono' (integrable_const B3)
        (hm3_m x (hU_sub_half hx)) (Eventually.of_forall fun ω => ?_)
      rw [Real.norm_eq_abs]; exact (hpt_bounds ω x hx).2.2.2.1
    have hd3 : ∀ x ∈ U, HasDerivAt E3 (E4 x) x := by
      intro x hx
      exact (hasDerivAt_integral_of_dominated_loc_of_deriv_le
        (μ := μ) (s := U) (bound := fun _ => M4)
        (hU_open.mem_nhds hx)
        (Filter.eventually_of_mem (hU_open.mem_nhds hx) fun y hy => hm3_m y (hU_sub_half hy))
        (hint3 x hx) (hm4_m x (hU_sub_half hx))
        (Eventually.of_forall fun ω y hy => by
          rw [Real.norm_eq_abs]; exact (hpt_bounds ω y hy).2.2.2.2)
        (integrable_const _)
        (Eventually.of_forall fun ω y hy =>
          (logGain_hasDerivAt K (timeReward T ξ ω) A hA (haω ω) t y (hU_sub_half hy)).2.2.2)).2
    have hc4_E4 : ContinuousOn E4 U := by
      apply MeasureTheory.continuousOn_of_dominated (bound := fun _ => M4)
      · intro x hx; exact hm4_m x (hU_sub_half hx)
      · intro x hx; exact Eventually.of_forall fun ω => by
          rw [Real.norm_eq_abs]; exact (hpt_bounds ω x hx).2.2.2.2
      · exact integrable_const M4
      · exact Eventually.of_forall fun ω => (hcont4 ω).mono hU_sub_half
    -- Evaluate `E1 0 = 0`: at `x = 0`, `projState K a t 0 = K.initial` and `projGain K a t 0 = 1`!
    have hp0_eq : ∀ ω, ∀ s, projState K (timeReward T ξ ω) s 0 = K.initial := by
      intro ω s
      induction s with
      | zero => rfl
      | succ s ihs =>
        dsimp [projState]
        simp only [ihs, zero_mul, add_zero, mul_one, K.initial_total, div_one, K.stationary]
    have hE1_0 : E1 0 = 0 := by
      have hval : ∀ ω, logGain1 K (timeReward T ξ ω) t 0 =
          ∑ e, K.initial e * timeReward T ξ ω t e := by
        intro ω
        have hu0_0 : (stateJetAt K (timeReward T ξ ω) t).u0 0 = K.initial := by
          rw [stateJetAt_u0_eq_projState, hp0_eq]
        have hzs0 := stateJetAt_zero_sum K (timeReward T ξ ω) A hA (haω ω) t 0
          (hU_sub_half (hsub_U h0_mem_Icc))
        have hu1_sum0 : ∑ e, (stateJetAt K (timeReward T ξ ω) t).u1 0 e = 0 := hzs0.2.1
        have hg0_0 : jetG0 (timeReward T ξ ω t) (stateJetAt K (timeReward T ξ ω) t) 0 = 1 := by
          simp [jetG0, jetW0, w0, hu0_0, K.initial_total]
        have hg1_0 : jetG1 (timeReward T ξ ω t) (stateJetAt K (timeReward T ξ ω) t) 0 =
            ∑ e, K.initial e * timeReward T ξ ω t e := by
          simp only [jetG1, jetW1, w1, zero_mul, add_zero, mul_one, Finset.sum_add_distrib,
            hu1_sum0, zero_add, hu0_0]
          exact Finset.sum_congr rfl fun e _ => mul_comm _ _
        unfold logGain1 q0
        rw [hg0_0, hg1_0, div_one]
      dsimp [E1]
      simp_rw [hval]
      have hint_term (e : E) : Integrable (fun ω => K.initial e * timeReward T ξ ω t e) μ :=
        (((hξ e).comp_measurePreserving (hT.iterate t)).integrable (by norm_num)).const_mul _
      rw [integral_finsetSum _ fun e _ => hint_term e]
      apply Finset.sum_eq_zero
      intro e _
      rw [integral_const_mul]
      have hshift_zero : (∫ ω, timeReward T ξ ω t e ∂μ) = 0 := by
        have hshift_fn : ∀ n : ℕ, (∫ ω, ξ e ((T : Ω → Ω)^[n] ω) ∂μ) = ∫ ω, ξ e ω ∂μ := by
          intro n
          induction n with
          | zero => simp
          | succ n ihn =>
            simpa only [Function.iterate_succ_apply] using
              (hT.integral_comp' (fun ω => ξ e ((T : Ω → Ω)^[n] ω))).trans ihn
        simpa [timeReward, hcenter e] using hshift_fn t
      rw [hshift_zero, mul_zero]
    have hE3_0 : |E3 0| ≤ B3 := by
      have h0U : (0 : ℝ) ∈ U := hsub_U h0_mem_Icc
      calc
        |∫ ω, logGain3 K (timeReward T ξ ω) t 0 ∂μ|
            ≤ ∫ ω, |logGain3 K (timeReward T ξ ω) t 0| ∂μ := abs_integral_le_integral_abs
        _ ≤ ∫ _ω, B3 ∂μ := integral_mono
            ((hint3 0 h0U).abs) (integrable_const B3)
            (fun ω => (hpt_bounds ω 0 h0U).2.2.2.1)
        _ = B3 := by simp
    have hE4_y : ∀ y ∈ Icc (-δ) δ, |E4 y| ≤ M4 := by
      intro y hy
      have hyU : y ∈ U := hsub_U hy
      have hint4 : Integrable (fun ω => logGain4 K (timeReward T ξ ω) t y) μ := by
        refine Integrable.mono' (integrable_const M4)
          (hm4_m y (hU_sub_half hyU)) (Eventually.of_forall fun ω => ?_)
        rw [Real.norm_eq_abs]; exact (hpt_bounds ω y hyU).2.2.2.2
      calc
        |∫ ω, logGain4 K (timeReward T ξ ω) t y ∂μ|
            ≤ ∫ ω, |logGain4 K (timeReward T ξ ω) t y| ∂μ := abs_integral_le_integral_abs
        _ ≤ ∫ _ω, M4 ∂μ := integral_mono hint4.abs (integrable_const M4)
            (fun ω => (hpt_bounds ω y hyU).2.2.2.2)
        _ = M4 := by simp
    exact ⟨hd0, hd1, hd2, hd3, hc4_E4, hE1_0, hE3_0, hE4_y⟩
  -- Assemble across horizon `N + 1` via telescoping on `U`!
  have hN_all (N : ℕ) :
      ContDiffOn ℝ 4 (finiteGrowth μ K T ξ L .additive N) (Icc (-δ) δ) ∧
      deriv (finiteGrowth μ K T ξ L .additive N) 0 = 0 ∧
      |iteratedDeriv 3 (finiteGrowth μ K T ξ L .additive N) 0| ≤ B3 ∧
      (∀ y ∈ Icc (-δ) δ,
        |iteratedDeriv 4 (finiteGrowth μ K T ξ L .additive N) y| ≤ M4) := by
    let E0 (t : ℕ) (x : ℝ) : ℝ := ∫ ω, Real.log (projGain K (timeReward T ξ ω) t x) ∂μ
    let E1 (t : ℕ) (x : ℝ) : ℝ := ∫ ω, logGain1 K (timeReward T ξ ω) t x ∂μ
    let E2 (t : ℕ) (x : ℝ) : ℝ := ∫ ω, logGain2 K (timeReward T ξ ω) t x ∂μ
    let E3 (t : ℕ) (x : ℝ) : ℝ := ∫ ω, logGain3 K (timeReward T ξ ω) t x ∂μ
    let E4 (t : ℕ) (x : ℝ) : ℝ := ∫ ω, logGain4 K (timeReward T ξ ω) t x ∂μ
    let g0 (x : ℝ) : ℝ := L + (∑ t ∈ Finset.range (N + 1), E0 t x) / (N + 1 : ℕ)
    let g1 (x : ℝ) : ℝ := (∑ t ∈ Finset.range (N + 1), E1 t x) / (N + 1 : ℕ)
    let g2 (x : ℝ) : ℝ := (∑ t ∈ Finset.range (N + 1), E2 t x) / (N + 1 : ℕ)
    let g3 (x : ℝ) : ℝ := (∑ t ∈ Finset.range (N + 1), E3 t x) / (N + 1 : ℕ)
    let g4 (x : ℝ) : ℝ := (∑ t ∈ Finset.range (N + 1), E4 t x) / (N + 1 : ℕ)
    have hfg_eq : Set.EqOn (finiteGrowth μ K T ξ L .additive N) g0 U := by
      intro x hx
      have hx_half := hU_sub_half hx
      dsimp [finiteGrowth, additiveGrowth, g0, E0]
      congr 2
      have htel : (fun ω => Real.log (Dobson.FiniteLineageVariance.expect K (N + 1)
          (fun p => ∏ t ∈ Finset.range (N + 1), (1 + x * timeReward T ξ ω t (p t))))) =
          fun ω => ∑ t ∈ Finset.range (N + 1), Real.log (projGain K (timeReward T ξ ω) t x) := by
        funext ω
        exact log_expect_prod_eq_sum_log_projGain K (timeReward T ξ ω) A hA (haω ω) x hx_half (N + 1)
      rw [htel, integral_finsetSum]
      intro t _
      have hm0 := (logGain_meas_and_cont μ K T hT ξ hξ A hA hbound t).1 x hx_half
      refine Integrable.mono' (integrable_const (|Real.log (1 / 2)| + |Real.log 2|)) hm0
        (Eventually.of_forall fun ω => ?_)
      rw [Real.norm_eq_abs]
      have hgb := (projState_prob_and_gain_bounds K (timeReward T ξ ω) A hA (haω ω)
        x hx_half t).2.2
      have hgp : 0 < projGain K (timeReward T ξ ω) t x := by linarith [hgb.1]
      have hlo := Real.log_le_log (by norm_num : (0 : ℝ) < 1 / 2) hgb.1.le
      have hhi := Real.log_le_log hgp hgb.2.le
      apply abs_le.mpr
      constructor <;> linarith [neg_abs_le (Real.log (1 / 2)), le_abs_self (Real.log 2),
        abs_nonneg (Real.log (1 / 2)), abs_nonneg (Real.log 2)]
    have hd_g0 : ∀ x ∈ U, HasDerivAt g0 (g1 x) x := by
      intro x hx
      simpa [g0, g1] using ((HasDerivAt.fun_sum fun t _ =>
        (hstep_smooth t).1 x hx).div_const ((N + 1 : ℕ) : ℝ)).const_add L
    have hd_g1 : ∀ x ∈ U, HasDerivAt g1 (g2 x) x := by
      intro x hx
      simpa [g1, g2] using (HasDerivAt.fun_sum fun t _ =>
        (hstep_smooth t).2.1 x hx).div_const ((N + 1 : ℕ) : ℝ)
    have hd_g2 : ∀ x ∈ U, HasDerivAt g2 (g3 x) x := by
      intro x hx
      simpa [g2, g3] using (HasDerivAt.fun_sum fun t _ =>
        (hstep_smooth t).2.2.1 x hx).div_const ((N + 1 : ℕ) : ℝ)
    have hd_g3 : ∀ x ∈ U, HasDerivAt g3 (g4 x) x := by
      intro x hx
      simpa [g3, g4] using (HasDerivAt.fun_sum fun t _ =>
        (hstep_smooth t).2.2.2.1 x hx).div_const ((N + 1 : ℕ) : ℝ)
    have hc_g4 : ContinuousOn g4 U :=
      (continuousOn_sum_finset _ fun t _ => (hstep_smooth t).2.2.2.2.1).div_const _
    have hd_fg0 : ∀ x ∈ U, HasDerivAt (finiteGrowth μ K T ξ L .additive N) (g1 x) x := by
      intro x hx
      exact (hd_g0 x hx).congr_of_eventuallyEq
        (Filter.eventuallyEq_of_mem (hU_open.mem_nhds hx) fun z hz => hfg_eq hz)
    obtain ⟨hcd, hder1, hder3, hder4⟩ :=
      contDiffOn_four_of_open_derivs hU_open hsub_U hd_fg0 hd_g1 hd_g2 hd_g3 hc_g4
    have hpos_N : (0 : ℝ) < (N + 1 : ℕ) := by positivity
    refine ⟨hcd, ?_, ?_, ?_⟩
    · rw [hder1 0 h0_mem_Icc]
      dsimp [g1]
      rw [Finset.sum_eq_zero fun t _ => (hstep_smooth t).2.2.2.2.2.1, zero_div]
    · rw [hder3 0 h0_mem_Icc]
      dsimp [g3]
      rw [abs_div, abs_of_pos hpos_N, div_le_iff₀ hpos_N]
      calc
        |∑ t ∈ Finset.range (N + 1), E3 t 0|
            ≤ ∑ t ∈ Finset.range (N + 1), |E3 t 0| := Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _t ∈ Finset.range (N + 1), B3 :=
            Finset.sum_le_sum fun t _ => (hstep_smooth t).2.2.2.2.2.2.1
        _ = B3 * (N + 1 : ℕ) := by simp [mul_comm]
    · intro y hy
      rw [hder4 y hy]
      dsimp [g4]
      rw [abs_div, abs_of_pos hpos_N, div_le_iff₀ hpos_N]
      calc
        |∑ t ∈ Finset.range (N + 1), E4 t y|
            ≤ ∑ t ∈ Finset.range (N + 1), |E4 t y| := Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _t ∈ Finset.range (N + 1), M4 :=
            Finset.sum_le_sum fun t _ => (hstep_smooth t).2.2.2.2.2.2.2 y hy
        _ = M4 * (N + 1 : ℕ) := by simp [mul_comm]
  exact ⟨hδ, fun N => (hN_all N).1, fun N => (hN_all N).2.1,
    fun N => (hN_all N).2.2.1, fun N => (hN_all N).2.2.2⟩

/-- Whole-process sign symmetry (`IdentDistrib` under `ξ ↦ -ξ`) implies exact evenness of
`finiteGrowth ... .additive N` (`C-mechanisms.tex` line 44). -/
theorem additive_finiteGrowth_even_of_jointSymmetry {E Ω : Type*}
    [Fintype E] [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) (K : Chain E) (T : Ω → Ω) (ξ : E → Ω → ℝ)
    (L : ℝ) (N : ℕ)
    (hsign : IdentDistrib (timeReward T ξ)
      (fun ω t e => -timeReward T ξ ω t e) μ μ)
    (y : ℝ) :
    finiteGrowth μ K T ξ L .additive N (-y) = finiteGrowth μ K T ξ L .additive N y := by
  let F : (ℕ → E → ℝ) → ℝ := fun z =>
    Real.log (Dobson.FiniteLineageVariance.expect K (N + 1)
      (fun p => ∏ t ∈ Finset.range (N + 1), (1 + y * z t (p t))))
  have hFm : AEMeasurable F (Measure.map (timeReward T ξ) μ) := by
    have hpath_m (p : ℕ → E) :
        Measurable (fun z : ℕ → E → ℝ =>
          ∏ t ∈ Finset.range (N + 1), (1 + y * z t (p t))) :=
      Finset.measurable_prod _ fun t _ =>
        measurable_const.add (measurable_const.mul
          ((measurable_pi_apply (p t)).comp (measurable_pi_apply t)))
    exact ((expect_meas K (N + 1) _ _ fun p =>
      (hpath_m p).aestronglyMeasurable).aemeasurable.log)
  have he := (hsign.comp_of_aemeasurable hFm).integral_eq
  have href (ω : Ω) : F (fun t e => -timeReward T ξ ω t e) =
      Real.log (Dobson.FiniteLineageVariance.expect K (N + 1)
        (fun p => ∏ t ∈ Finset.range (N + 1), (1 + (-y) * timeReward T ξ ω t (p t)))) := by
    dsimp [F]
    congr 2
    funext p
    apply Finset.prod_congr rfl
    intro t _; ring
  simp only [Function.comp_def, href] at he
  dsimp [finiteGrowth, additiveGrowth]
  rw [← he]

/-- Cubic bound for bounded additive reference-path growth. Uniform
derivative controls are proved; `hr` supplies convergence to `r` on the
stated interval. General-matrix reduction and physical exponent identification
are not conclusions of this core theorem. -/
theorem mech_bounded_cubic_unconditional {E Ω : Type*}
    [Fintype E] [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (ξ : E → Ω → ℝ) (hξ : ∀ e, MemLp (ξ e) 2 μ)
    (hcenter : ∀ e, ∫ ω, ξ e ω ∂μ = 0)
    (Ku β A : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ e e' n, |(K.prob ^ (n + 1)) e' e - K.initial e| ≤ Ku * β ^ n)
    (hA : 0 < A) (hbound : ∀ e ω, |ξ e ω| ≤ A)
    (r : ℝ → ℝ) (L : ℝ)
    (hr : ∀ x, |x| ≤ 1 / (2 * A) →
      Tendsto (fun N => finiteGrowth μ K T ξ L .additive N x) atTop (𝓝 (r x))) :
    ∃ δ K₃ : ℝ, 0 < δ ∧ 0 ≤ K₃ ∧
      ∀ x, |x| ≤ δ →
        |L - r x - Dobson.FinitePathCoefficient.covarianceCoefficient μ K T ξ * x ^ 2| ≤
          K₃ * |x| ^ 3 := by
  obtain ⟨m, hm, hblock⟩ := exists_contraction_block K Ku β hβ0 hβ1 hmix
  obtain ⟨hδ, hf, hfirst, hthird, hM⟩ :=
    additive_finiteGrowth_projective_controls μ K m hm hblock T hT ξ hξ hcenter A hA hbound L
  let δ := 1 / (48 * (m : ℝ) ^ 2 * A)
  let B3 := logGain3Bound m A
  let M4 := logGain4Bound m A
  have hcov_bdd : ∀ e e' lag, |lagCovariance μ T ξ e e' lag| ≤ A ^ 2 := by
    intro e e' lag
    have hint : Integrable (fun ω => ξ e ((T : Ω → Ω)^[lag] ω) * ξ e' ω) μ := by
      simpa only [timeReward, Function.iterate_zero_apply] using
        timeReward_product_integrable μ T hT ξ hξ e e' lag 0
    calc
      |∫ ω, ξ e ((T : Ω → Ω)^[lag] ω) * ξ e' ω ∂μ|
          ≤ ∫ ω, |ξ e ((T : Ω → Ω)^[lag] ω) * ξ e' ω| ∂μ := abs_integral_le_integral_abs
      _ ≤ ∫ _ω, A ^ 2 ∂μ :=
          integral_mono hint.norm (integrable_const (A ^ 2)) fun ω => by
            change |ξ e ((T : Ω → Ω)^[lag] ω) * ξ e' ω| ≤ A ^ 2
            rw [abs_mul, sq]
            exact mul_le_mul (hbound e _) (hbound e' _) (abs_nonneg _) hA.le
      _ = A ^ 2 := by simp
  have hδ_le_half : δ ≤ 1 / (2 * A) := by
    have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
    dsimp [δ]
    exact one_div_le_one_div_of_le (by positivity) (by nlinarith [sq_nonneg ((m : ℝ) - 1)])
  have hB3_nn : 0 ≤ B3 := (abs_nonneg _).trans (hthird 0)
  have hM4_nn : 0 ≤ M4 := (abs_nonneg _).trans (hM 0 0 ⟨by linarith, hδ.le⟩)
  refine ⟨δ, B3 / 6 + M4 * δ / 24, hδ, by positivity, ?_⟩
  exact growth_limit_cubic μ K T hT ξ hξ (A ^ 2) Ku β (sq_nonneg A) hKu hβ0 hβ1
    hcov_bdd hmix .additive (fun _ => ⟨A, hA, hbound⟩)
    r L δ M4 B3 hδ hf hM hfirst hthird
    (fun x hx => hr x (hx.trans hδ_le_half))

/-- Quartic bound for bounded additive reference paths with whole-process
sign symmetry. Uniform fourth-derivative control and evenness are proved;
`hr` supplies the growth limit. Physical exponent identification remains
outside this core. -/
theorem mech_bounded_quartic_unconditional {E Ω : Type*}
    [Fintype E] [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (ξ : E → Ω → ℝ) (hξ : ∀ e, MemLp (ξ e) 2 μ)
    (hcenter : ∀ e, ∫ ω, ξ e ω ∂μ = 0)
    (hsign : IdentDistrib (timeReward T ξ)
      (fun ω t e => -timeReward T ξ ω t e) μ μ)
    (Ku β A : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ e e' n, |(K.prob ^ (n + 1)) e' e - K.initial e| ≤ Ku * β ^ n)
    (hA : 0 < A) (hbound : ∀ e ω, |ξ e ω| ≤ A)
    (r : ℝ → ℝ) (L : ℝ)
    (hr : ∀ x, |x| ≤ 1 / (2 * A) →
      Tendsto (fun N => finiteGrowth μ K T ξ L .additive N x) atTop (𝓝 (r x))) :
    ∃ δ K₄ : ℝ, 0 < δ ∧ 0 ≤ K₄ ∧
      ∀ x, |x| ≤ δ →
        |L - r x - Dobson.FinitePathCoefficient.covarianceCoefficient μ K T ξ * x ^ 2| ≤
          K₄ * x ^ 4 := by
  obtain ⟨m, hm, hblock⟩ := exists_contraction_block K Ku β hβ0 hβ1 hmix
  obtain ⟨hδ, hf, _hfirst, _hthird, hM⟩ :=
    additive_finiteGrowth_projective_controls μ K m hm hblock T hT ξ hξ hcenter A hA hbound L
  let δ := 1 / (48 * (m : ℝ) ^ 2 * A)
  let M4 := logGain4Bound m A
  have hcov_bdd : ∀ e e' lag, |lagCovariance μ T ξ e e' lag| ≤ A ^ 2 := by
    intro e e' lag
    have hint : Integrable (fun ω => ξ e ((T : Ω → Ω)^[lag] ω) * ξ e' ω) μ := by
      simpa only [timeReward, Function.iterate_zero_apply] using
        timeReward_product_integrable μ T hT ξ hξ e e' lag 0
    calc
      |∫ ω, ξ e ((T : Ω → Ω)^[lag] ω) * ξ e' ω ∂μ|
          ≤ ∫ ω, |ξ e ((T : Ω → Ω)^[lag] ω) * ξ e' ω| ∂μ := abs_integral_le_integral_abs
      _ ≤ ∫ _ω, A ^ 2 ∂μ :=
          integral_mono hint.norm (integrable_const (A ^ 2)) fun ω => by
            change |ξ e ((T : Ω → Ω)^[lag] ω) * ξ e' ω| ≤ A ^ 2
            rw [abs_mul, sq]
            exact mul_le_mul (hbound e _) (hbound e' _) (abs_nonneg _) hA.le
      _ = A ^ 2 := by simp
  have hδ_le_half : δ ≤ 1 / (2 * A) := by
    have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
    dsimp [δ]
    exact one_div_le_one_div_of_le (by positivity) (by nlinarith [sq_nonneg ((m : ℝ) - 1)])
  have hM4_nn : 0 ≤ M4 := (abs_nonneg _).trans (hM 0 0 ⟨by linarith, hδ.le⟩)
  refine ⟨δ, M4 / 24, hδ, by positivity, ?_⟩
  exact growth_limit_quartic μ K T hT ξ hξ (A ^ 2) Ku β (sq_nonneg A) hKu hβ0 hβ1
    hcov_bdd hmix .additive (fun _ => ⟨A, hA, hbound⟩)
    r L δ M4 hδ hf hM
    (fun N y _ => additive_finiteGrowth_even_of_jointSymmetry μ K T ξ L N hsign y)
    (fun x hx => hr x (hx.trans hδ_le_half))

end ExpectationAndCorollaries

end Dobson.P1bB_MechBounded

#print axioms Dobson.P1bB_MechBounded.projective_state_and_logGain_bounds
#print axioms Dobson.P1bB_MechBounded.additive_finiteGrowth_projective_controls
#print axioms Dobson.P1bB_MechBounded.mech_bounded_cubic_unconditional
#print axioms Dobson.P1bB_MechBounded.mech_bounded_quartic_unconditional
