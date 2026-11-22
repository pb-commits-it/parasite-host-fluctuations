import DobsonProofs.Coverage1Gamma
import DobsonProofs.Coverage1Lineage
import DobsonProofs.Coverage2Bounds
import DobsonProofs.Coverage6Remainders
import DobsonProofs.RingQuartic
import Mathlib.Analysis.Calculus.ContDiff.Deriv
import Mathlib.Probability.Independence.Integration
import Mathlib.Probability.Moments.SubGaussian

/-! # Part A: supporting projective-derivative calculations

This module contains contraction-gauge, normalized-recursion, derivative and
forcing estimates. It does not contain the six manuscript-row remainder and
threshold assemblies formerly promised by this header.
Coverage correction, 2026-11-22: no owed-step closure or inventory upgrade
is credited to this module. The corresponding Gaussian remainder, lineage,
ring-coefficient and threshold rows retain their recorded partial-P1 scope. -/

namespace Dobson.P1bA_LognormalRemainders

open Set Filter MeasureTheory ProbabilityTheory
open scoped BigOperators Topology ENNReal NNReal
open Dobson.FiniteLineageVariance Dobson.FinitePathJet Dobson.Coverage6Remainders

/-! ## Section 1: Adapted `ℓ¹` contraction gauge and exponential absorption -/

section AdaptedGauge

noncomputable def l1 {E : Type*} [Fintype E] (z : E → ℝ) : ℝ := ∑ i, |z i|

noncomputable def gauge {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (z : E → ℝ) : ℝ :=
  ∑ k ∈ Finset.range m, l1 ((K.prob ^ k).vecMul z)

lemma l1_nonneg {E : Type*} [Fintype E] (z : E → ℝ) : 0 ≤ l1 z :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

lemma l1_zero {E : Type*} [Fintype E] : l1 (fun _ : E => (0 : ℝ)) = 0 := by simp [l1]

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

lemma l1_step {E : Type*} [Fintype E] (K : Chain E) (z : E → ℝ) :
    l1 (K.prob.vecMul z) ≤ l1 z := by
  unfold l1
  calc ∑ j, |∑ i, z i * K.prob i j|
      ≤ ∑ j, ∑ i, |z i| * K.prob i j := by
        refine Finset.sum_le_sum fun j _ => ?_
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
        rw [abs_mul, abs_of_nonneg (K.nonneg i j)]
    _ = ∑ i, |z i| := by rw [Finset.sum_comm]; simp [← Finset.mul_sum, K.total]

lemma l1_power {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (z : E → ℝ) (n : ℕ) : l1 ((K.prob ^ n).vecMul z) ≤ l1 z := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [pow_succ, ← Matrix.vecMul_vecMul]
    exact (l1_step K _).trans ih

lemma gauge_nonneg {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (z : E → ℝ) : 0 ≤ gauge K m z :=
  Finset.sum_nonneg fun _ _ => l1_nonneg _

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
  refine Finset.sum_le_sum fun k _ => ?_
  have hadd : ((K.prob ^ k).vecMul fun i => z i + w i) =
      fun j => (K.prob ^ k).vecMul z j + (K.prob ^ k).vecMul w j := by
    funext j; simp [Matrix.vecMul, dotProduct, add_mul, Finset.sum_add_distrib]
  rw [hadd]; exact l1_add_le _ _

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
  calc gauge K m z ≤ ∑ _k ∈ Finset.range m, l1 z := Finset.sum_le_sum fun k _ => l1_power K z k
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
  refine Finset.sum_congr rfl fun k _ => ?_
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
        ((m : ℝ) - 1 / 2) * gauge K m z := by field_simp [hmR.ne']
    rw [hrhs]; nlinarith
  exact le_of_mul_le_mul_left hmul hmR

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
  calc ∑ j, |∑ i, z i * ((K.prob ^ (n + 1)) i j - K.initial j)|
      ≤ ∑ _j : E, ∑ i, |z i| * (Ku * β ^ n) := by
        refine Finset.sum_le_sum fun j _ => ?_
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_left (hmix j i) (abs_nonneg _)
    _ = _ := by simp [← Finset.sum_mul]; ring

/-- Uniform contraction-block witness depending only on `(Fintype.card E, Ku, β)` so a single
`m` works simultaneously for any family sharing `(Ku, β)`. -/
lemma exists_uniform_contraction_block (cardE : ℕ) (Ku β : ℝ)
    (_hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1) :
    ∃ m : ℕ, 0 < m ∧ (cardE : ℝ) * Ku * β ^ (m - 1) ≤ 1 / 2 := by
  have ht := (tendsto_pow_atTop_nhds_zero_of_lt_one hβ0 hβ1).const_mul ((cardE : ℝ) * Ku)
  simp only [mul_zero] at ht
  obtain ⟨n, hn⟩ := (ht.eventually (gt_mem_nhds (show (0 : ℝ) < 1 / 2 by norm_num))).exists
  exact ⟨n + 1, by omega, hn.le⟩

lemma exp_dilation (c η : ℝ) (hc : 1 ≤ c) :
    1 + c * (Real.exp η - 1) ≤ Real.exp (c * η) := by
  have hc0 : 0 < c := by linarith
  have h := convexOn_exp.2 (mem_univ (0 : ℝ)) (mem_univ (c * η))
    (show 0 ≤ 1 - 1 / c by rw [sub_nonneg]; exact (div_le_one hc0).mpr hc)
    (by positivity : 0 ≤ 1 / c) (by ring : 1 - 1 / c + 1 / c = 1)
  have hi : (1 - 1 / c) * (0 : ℝ) + (1 / c) * (c * η) = η := by field_simp [hc0.ne']; ring
  simp only [smul_eq_mul, hi, Real.exp_zero, mul_one] at h
  have hh := mul_le_mul_of_nonneg_left h hc0.le
  field_simp [hc0.ne'] at hh
  nlinarith

lemma exponential_absorption (β c η : ℝ) (hβ : 0 < β) (hη : 0 ≤ η) :
    Real.exp η * (β + c * (Real.exp η - 1)) ≤
      β * Real.exp ((1 + max 1 (c / β)) * η) := by
  let d := max 1 (c / β)
  have hd : 1 ≤ d := le_max_left _ _
  have hcb : c ≤ β * d := by
    have := le_max_right (1 : ℝ) (c / β)
    exact (div_le_iff₀ hβ).mp this |>.trans_eq (mul_comm _ _)
  have hex : 0 ≤ Real.exp η - 1 := sub_nonneg.mpr (Real.one_le_exp_iff.mpr hη)
  have hh := mul_le_mul_of_nonneg_left (exp_dilation d η hd) hβ.le
  calc Real.exp η * (β + c * (Real.exp η - 1))
      ≤ Real.exp η * (β * (1 + d * (Real.exp η - 1))) := by
        apply mul_le_mul_of_nonneg_left _ (Real.exp_pos _).le
        nlinarith [mul_le_mul_of_nonneg_right hcb hex]
    _ ≤ Real.exp η * (β * Real.exp (d * η)) := mul_le_mul_of_nonneg_left hh (Real.exp_pos _).le
    _ = β * Real.exp ((1 + max 1 (c / β)) * η) := by
        rw [mul_left_comm, ← Real.exp_add]; congr 2; dsimp [d]; ring

end AdaptedGauge

/-! ## Section 2: Unified fourth-order projective recursion for positive edge multipliers -/

section UnifiedProjectiveJet

/-- 4-jet of step-`t` edge multipliers `h0` (`h1 = h0'`, `h2 = h0''`, `h3 = h0'''`, `h4 = h0''''`). -/
structure EdgeStepJet (E : Type*) where
  h0 : ℝ → E → ℝ
  h1 : ℝ → E → ℝ
  h2 : ℝ → E → ℝ
  h3 : ℝ → E → ℝ
  h4 : ℝ → E → ℝ

noncomputable def rawStateGen {E : Type*} [Fintype E]
    (K : Chain E) (h : ℕ → ℝ → E → ℝ) (u0 : E → ℝ) : ℕ → ℝ → E → ℝ
  | 0, _ => u0
  | t + 1, x => K.prob.vecMul (fun e => rawStateGen K h u0 t x e * h t x e)

lemma rawStateGen_shift {E : Type*} [Fintype E]
    (K : Chain E) (h : ℕ → ℝ → E → ℝ) (u0 : E → ℝ) (n : ℕ) (x : ℝ) :
    rawStateGen K (fun t => h (t + 1)) (K.prob.vecMul (fun i => u0 i * h 0 x i)) n x =
      rawStateGen K h u0 (n + 1) x := by
  induction n generalizing h u0 with
  | zero => rfl
  | succ n ih => simp only [rawStateGen, ih h u0]

lemma genExpect_prod_eq_sum_rawStateGen {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (h : ℕ → ℝ → E → ℝ) (u0 : E → ℝ) (n : ℕ) (x : ℝ) :
    (∑ i, u0 i * pathExpect K n i (fun p => ∏ t ∈ Finset.range n, h t x (p t))) =
      ∑ e, rawStateGen K h u0 n x e := by
  induction n generalizing h u0 with
  | zero => simp [pathExpect, rawStateGen]
  | succ n ih =>
    have hpeel (i j : E) :
        pathExpect K n j (fun p => ∏ t ∈ Finset.range (n + 1), h t x (prepend i p t)) =
          h 0 x i * pathExpect K n j (fun p => ∏ t ∈ Finset.range n, h (t + 1) x (p t)) := by
      have hfn : (fun p : ℕ → E => ∏ t ∈ Finset.range (n + 1), h t x (prepend i p t)) =
          fun p => h 0 x i * ∏ t ∈ Finset.range n, h (t + 1) x (p t) := by
        funext p; rw [Finset.prod_range_succ']; simp [prepend, mul_comm]
      rw [hfn, pathExpect_const_mul]
    simp only [pathExpect, hpeel]
    have hswap : (∑ i, u0 i * ∑ j, K.prob i j * (h 0 x i *
        pathExpect K n j (fun p => ∏ t ∈ Finset.range n, h (t + 1) x (p t)))) =
        ∑ j, (K.prob.vecMul (fun i => u0 i * h 0 x i) j) *
          pathExpect K n j (fun p => ∏ t ∈ Finset.range n, h (t + 1) x (p t)) := by
      simp only [Matrix.vecMul, dotProduct, Finset.sum_mul, Finset.mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => by ring
    rw [hswap, ih (fun t => h (t + 1)) (K.prob.vecMul (fun i => u0 i * h 0 x i)),
      rawStateGen_shift]

noncomputable def projStateGen {E : Type*} [Fintype E]
    (K : Chain E) (h : ℕ → ℝ → E → ℝ) : ℕ → ℝ → E → ℝ
  | 0, _ => K.initial
  | t + 1, x =>
      let G := ∑ e, projStateGen K h t x e * h t x e
      K.prob.vecMul (fun e => projStateGen K h t x e * h t x e / G)

noncomputable def projGainGen {E : Type*} [Fintype E]
    (K : Chain E) (h : ℕ → ℝ → E → ℝ) (t : ℕ) (x : ℝ) : ℝ :=
  ∑ e, projStateGen K h t x e * h t x e

lemma projStateGen_prob_and_gain_pos {E : Type*} [Fintype E]
    (K : Chain E) (h : ℕ → ℝ → E → ℝ) (x : ℝ)
    (hh_pos : ∀ s e, 0 < h s x e) (t : ℕ) :
    (∀ e, 0 ≤ projStateGen K h t x e) ∧
    (∑ e, projStateGen K h t x e = 1) ∧
    0 < projGainGen K h t x := by
  have hprob : ∀ s : ℕ, (∀ e, 0 ≤ projStateGen K h s x e) ∧ (∑ e, projStateGen K h s x e = 1) := by
    intro s
    induction s with
    | zero => exact ⟨K.initial_nonneg, K.initial_total⟩
    | succ s ih =>
      obtain ⟨hnn, hsum⟩ := ih
      have hex : ∃ e, 0 < projStateGen K h s x e := by
        by_contra hc; push Not at hc
        have := Finset.sum_nonpos (s := Finset.univ) fun e _ => hc e
        rw [hsum] at this; norm_num at this
      obtain ⟨e0, he0⟩ := hex
      have hG_pos : 0 < ∑ e, projStateGen K h s x e * h s x e :=
        Finset.sum_pos' (fun e _ => mul_nonneg (hnn e) (hh_pos s e).le)
          ⟨e0, Finset.mem_univ e0, mul_pos he0 (hh_pos s e0)⟩
      refine ⟨?_, ?_⟩
      · intro j
        simp only [projStateGen, Matrix.vecMul, dotProduct]
        exact Finset.sum_nonneg fun i _ =>
          mul_nonneg (div_nonneg (mul_nonneg (hnn i) (hh_pos s i).le) hG_pos.le) (K.nonneg i j)
      · simp only [projStateGen, Matrix.vecMul, dotProduct]
        rw [Finset.sum_comm]
        simp only [← Finset.mul_sum, K.total, mul_one, ← Finset.sum_div]
        exact div_self hG_pos.ne'
  obtain ⟨hnn, hsum⟩ := hprob t
  have hex : ∃ e, 0 < projStateGen K h t x e := by
    by_contra hc; push Not at hc
    have := Finset.sum_nonpos (s := Finset.univ) fun e _ => hc e
    rw [hsum] at this; norm_num at this
  obtain ⟨e0, he0⟩ := hex
  exact ⟨hnn, hsum, Finset.sum_pos' (fun e _ => mul_nonneg (hnn e) (hh_pos t e).le)
    ⟨e0, Finset.mem_univ e0, mul_pos he0 (hh_pos t e0)⟩⟩

lemma log_expect_prod_eq_sum_log_projGainGen {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (h : ℕ → ℝ → E → ℝ) (x : ℝ)
    (hh_pos : ∀ s e, 0 < h s x e) (n : ℕ) :
    Real.log (expect K n (fun p => ∏ t ∈ Finset.range n, h t x (p t))) =
      ∑ t ∈ Finset.range n, Real.log (projGainGen K h t x) := by
  have hraw : ∀ m : ℕ, rawStateGen K h K.initial m x =
      fun e => (∏ t ∈ Finset.range m, projGainGen K h t x) * projStateGen K h m x e := by
    intro m
    induction m with
    | zero => funext e; simp [rawStateGen, projStateGen]
    | succ m ihm =>
      have hG_pos := (projStateGen_prob_and_gain_pos K h x hh_pos m).2.2
      funext j
      simp only [rawStateGen, ihm, projStateGen, Matrix.vecMul, dotProduct,
        Finset.prod_range_succ, Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      change (∏ t ∈ Finset.range m, projGainGen K h t x) * projStateGen K h m x i * h m x i *
          K.prob i j =
        (∏ t ∈ Finset.range m, projGainGen K h t x) * projGainGen K h m x *
          (projStateGen K h m x i * h m x i / projGainGen K h m x * K.prob i j)
      field_simp [hG_pos.ne']
  have hexp : expect K n (fun p => ∏ t ∈ Finset.range n, h t x (p t)) =
      ∏ t ∈ Finset.range n, projGainGen K h t x := by
    unfold expect
    rw [genExpect_prod_eq_sum_rawStateGen, hraw n, ← Finset.mul_sum,
      (projStateGen_prob_and_gain_pos K h x hh_pos n).2.1, mul_one]
  rw [hexp, Real.log_prod]
  intro t _
  exact (projStateGen_prob_and_gain_pos K h x hh_pos t).2.2.ne'

noncomputable def q0 (W G : ℝ → ℝ) (x : ℝ) : ℝ := W x / G x
noncomputable def q1 (W W1 G G1 : ℝ → ℝ) (x : ℝ) : ℝ := (W1 x - G1 x * q0 W G x) / G x
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
  have h1eq : W1 x - G1 x * q0 W G x = q1 W W1 G G1 x * G x := by unfold q1; field_simp [hG]
  convert hnum.div hG0 hG using 1
  · rfl
  · simp only [Pi.sub_apply, Pi.mul_apply]; unfold q2; rw [h1eq]; field_simp [hG]; ring

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
      q2 W W1 W2 G G1 G2 x * G x := by unfold q2; field_simp [hG]
  convert hnum.div hG0 hG using 1
  · rfl
  · simp only [Pi.sub_apply, Pi.mul_apply]; unfold q3; rw [h2eq]; field_simp [hG]; ring

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
      G3 x * q0 W G x = q3 W W1 W2 W3 G G1 G2 G3 x * G x := by unfold q3; field_simp [hG]
  convert hnum.div hG0 hG using 1
  · rfl
  · simp only [Pi.sub_apply, Pi.mul_apply]; unfold q4; rw [h3eq]; field_simp [hG]; ring

structure StateJet (E : Type*) where
  u0 : ℝ → E → ℝ
  u1 : ℝ → E → ℝ
  u2 : ℝ → E → ℝ
  u3 : ℝ → E → ℝ
  u4 : ℝ → E → ℝ

noncomputable def jetW0 {E : Type*} (H : EdgeStepJet E) (J : StateJet E) (e : E) (x : ℝ) : ℝ :=
  J.u0 x e * H.h0 x e
noncomputable def jetW1 {E : Type*} (H : EdgeStepJet E) (J : StateJet E) (e : E) (x : ℝ) : ℝ :=
  J.u1 x e * H.h0 x e + J.u0 x e * H.h1 x e
noncomputable def jetW2 {E : Type*} (H : EdgeStepJet E) (J : StateJet E) (e : E) (x : ℝ) : ℝ :=
  J.u2 x e * H.h0 x e + 2 * J.u1 x e * H.h1 x e + J.u0 x e * H.h2 x e
noncomputable def jetW3 {E : Type*} (H : EdgeStepJet E) (J : StateJet E) (e : E) (x : ℝ) : ℝ :=
  J.u3 x e * H.h0 x e + 3 * J.u2 x e * H.h1 x e + 3 * J.u1 x e * H.h2 x e + J.u0 x e * H.h3 x e
noncomputable def jetW4 {E : Type*} (H : EdgeStepJet E) (J : StateJet E) (e : E) (x : ℝ) : ℝ :=
  J.u4 x e * H.h0 x e + 4 * J.u3 x e * H.h1 x e + 6 * J.u2 x e * H.h2 x e +
    4 * J.u1 x e * H.h3 x e + J.u0 x e * H.h4 x e

lemma hasDerivAt_jetW {E : Type*} (H : EdgeStepJet E) (J : StateJet E) (e : E) (x : ℝ)
    (hh0 : HasDerivAt (fun y => H.h0 y e) (H.h1 x e) x)
    (hh1 : HasDerivAt (fun y => H.h1 y e) (H.h2 x e) x)
    (hh2 : HasDerivAt (fun y => H.h2 y e) (H.h3 x e) x)
    (hh3 : HasDerivAt (fun y => H.h3 y e) (H.h4 x e) x)
    (hu0 : HasDerivAt (fun y => J.u0 y e) (J.u1 x e) x)
    (hu1 : HasDerivAt (fun y => J.u1 y e) (J.u2 x e) x)
    (hu2 : HasDerivAt (fun y => J.u2 y e) (J.u3 x e) x)
    (hu3 : HasDerivAt (fun y => J.u3 y e) (J.u4 x e) x) :
    HasDerivAt (jetW0 H J e) (jetW1 H J e x) x ∧
    HasDerivAt (jetW1 H J e) (jetW2 H J e x) x ∧
    HasDerivAt (jetW2 H J e) (jetW3 H J e x) x ∧
    HasDerivAt (jetW3 H J e) (jetW4 H J e x) x := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · convert hu0.mul hh0 using 1 <;> [rfl; (unfold jetW1; ring)]
  · convert (hu1.mul hh0).add (hu0.mul hh1) using 1 <;> [rfl; (unfold jetW2; ring)]
  · convert ((hu2.mul hh0).add ((hu1.const_mul 2).mul hh1)).add (hu0.mul hh2) using 1 <;>
      [rfl; (unfold jetW3; ring)]
  · convert (((hu3.mul hh0).add ((hu2.const_mul 3).mul hh1)).add
      ((hu1.const_mul 3).mul hh2)).add (hu0.mul hh3) using 1 <;>
      [rfl; (unfold jetW4; ring)]

noncomputable def jetG0 {E : Type*} [Fintype E] (H : EdgeStepJet E) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW0 H J e x
noncomputable def jetG1 {E : Type*} [Fintype E] (H : EdgeStepJet E) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW1 H J e x
noncomputable def jetG2 {E : Type*} [Fintype E] (H : EdgeStepJet E) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW2 H J e x
noncomputable def jetG3 {E : Type*} [Fintype E] (H : EdgeStepJet E) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW3 H J e x
noncomputable def jetG4 {E : Type*} [Fintype E] (H : EdgeStepJet E) (J : StateJet E) (x : ℝ) : ℝ :=
  ∑ e, jetW4 H J e x

noncomputable def nextStateJet {E : Type*} [Fintype E]
    (K : Chain E) (H : EdgeStepJet E) (J : StateJet E) : StateJet E where
  u0 := fun x => K.prob.vecMul (fun e => q0 (jetW0 H J e) (jetG0 H J) x)
  u1 := fun x => K.prob.vecMul (fun e => q1 (jetW0 H J e) (jetW1 H J e) (jetG0 H J) (jetG1 H J) x)
  u2 := fun x => K.prob.vecMul (fun e =>
    q2 (jetW0 H J e) (jetW1 H J e) (jetW2 H J e) (jetG0 H J) (jetG1 H J) (jetG2 H J) x)
  u3 := fun x => K.prob.vecMul (fun e =>
    q3 (jetW0 H J e) (jetW1 H J e) (jetW2 H J e) (jetW3 H J e)
      (jetG0 H J) (jetG1 H J) (jetG2 H J) (jetG3 H J) x)
  u4 := fun x => K.prob.vecMul (fun e =>
    q4 (jetW0 H J e) (jetW1 H J e) (jetW2 H J e) (jetW3 H J e) (jetW4 H J e)
      (jetG0 H J) (jetG1 H J) (jetG2 H J) (jetG3 H J) (jetG4 H J) x)

noncomputable def stateJetAt {E : Type*} [Fintype E]
    (K : Chain E) (H : ℕ → EdgeStepJet E) : ℕ → StateJet E
  | 0 => {
      u0 := fun _ => K.initial
      u1 := fun _ _ => 0
      u2 := fun _ _ => 0
      u3 := fun _ _ => 0
      u4 := fun _ _ => 0 }
  | t + 1 => nextStateJet K (H t) (stateJetAt K H t)

lemma stateJetAt_u0_eq_projStateGen {E : Type*} [Fintype E]
    (K : Chain E) (H : ℕ → EdgeStepJet E) (t : ℕ) (x : ℝ) :
    (stateJetAt K H t).u0 x = projStateGen K (fun s => (H s).h0) t x := by
  induction t with
  | zero => rfl
  | succ t ih => simp only [stateJetAt, nextStateJet, projStateGen, q0, jetW0, jetG0, ih]

lemma jetG0_eq_projGainGen {E : Type*} [Fintype E]
    (K : Chain E) (H : ℕ → EdgeStepJet E) (t : ℕ) (x : ℝ) :
    jetG0 (H t) (stateJetAt K H t) x = projGainGen K (fun s => (H s).h0) t x := by
  simp only [jetG0, jetW0, stateJetAt_u0_eq_projStateGen, projGainGen]

noncomputable def logGain1 {E : Type*} [Fintype E]
    (K : Chain E) (H : ℕ → EdgeStepJet E) (t : ℕ) (x : ℝ) : ℝ :=
  q0 (jetG1 (H t) (stateJetAt K H t)) (jetG0 (H t) (stateJetAt K H t)) x

noncomputable def logGain2 {E : Type*} [Fintype E]
    (K : Chain E) (H : ℕ → EdgeStepJet E) (t : ℕ) (x : ℝ) : ℝ :=
  q1 (jetG1 (H t) (stateJetAt K H t)) (jetG2 (H t) (stateJetAt K H t))
    (jetG0 (H t) (stateJetAt K H t)) (jetG1 (H t) (stateJetAt K H t)) x

noncomputable def logGain3 {E : Type*} [Fintype E]
    (K : Chain E) (H : ℕ → EdgeStepJet E) (t : ℕ) (x : ℝ) : ℝ :=
  q2 (jetG1 (H t) (stateJetAt K H t)) (jetG2 (H t) (stateJetAt K H t))
    (jetG3 (H t) (stateJetAt K H t))
    (jetG0 (H t) (stateJetAt K H t)) (jetG1 (H t) (stateJetAt K H t))
    (jetG2 (H t) (stateJetAt K H t)) x

noncomputable def logGain4 {E : Type*} [Fintype E]
    (K : Chain E) (H : ℕ → EdgeStepJet E) (t : ℕ) (x : ℝ) : ℝ :=
  q3 (jetG1 (H t) (stateJetAt K H t)) (jetG2 (H t) (stateJetAt K H t))
    (jetG3 (H t) (stateJetAt K H t)) (jetG4 (H t) (stateJetAt K H t))
    (jetG0 (H t) (stateJetAt K H t)) (jetG1 (H t) (stateJetAt K H t))
    (jetG2 (H t) (stateJetAt K H t)) (jetG3 (H t) (stateJetAt K H t)) x

lemma stateJetAt_and_logGain_hasDerivAt {E : Type*} [Fintype E]
    (K : Chain E) (H : ℕ → EdgeStepJet E) (x : ℝ)
    (hh_pos : ∀ s e, 0 < (H s).h0 x e)
    (hh_der : ∀ s e,
      HasDerivAt (fun y => (H s).h0 y e) ((H s).h1 x e) x ∧
      HasDerivAt (fun y => (H s).h1 y e) ((H s).h2 x e) x ∧
      HasDerivAt (fun y => (H s).h2 y e) ((H s).h3 x e) x ∧
      HasDerivAt (fun y => (H s).h3 y e) ((H s).h4 x e) x) (t : ℕ) :
    HasDerivAt (fun y => Real.log (projGainGen K (fun s => (H s).h0) t y)) (logGain1 K H t x) x ∧
    HasDerivAt (logGain1 K H t) (logGain2 K H t x) x ∧
    HasDerivAt (logGain2 K H t) (logGain3 K H t x) x ∧
    HasDerivAt (logGain3 K H t) (logGain4 K H t x) x := by
  have hstate : ∀ s : ℕ,
      let J := stateJetAt K H s
      (∀ e, HasDerivAt (fun y => J.u0 y e) (J.u1 x e) x) ∧
      (∀ e, HasDerivAt (fun y => J.u1 y e) (J.u2 x e) x) ∧
      (∀ e, HasDerivAt (fun y => J.u2 y e) (J.u3 x e) x) ∧
      (∀ e, HasDerivAt (fun y => J.u3 y e) (J.u4 x e) x) := by
    intro s
    induction s with
    | zero =>
      exact ⟨fun e => hasDerivAt_const x (K.initial e),
        fun _ => hasDerivAt_const x 0,
        fun _ => hasDerivAt_const x 0,
        fun _ => hasDerivAt_const x 0⟩
    | succ s ih =>
      obtain ⟨hd0, hd1, hd2, hd3⟩ := ih
      let J := stateJetAt K H s
      have hG_ne : jetG0 (H s) J x ≠ 0 := by
        rw [jetG0_eq_projGainGen]
        exact (projStateGen_prob_and_gain_pos K (fun r => (H r).h0) x hh_pos s).2.2.ne'
      have hW (e : E) := hasDerivAt_jetW (H s) J e x
        (hh_der s e).1 (hh_der s e).2.1 (hh_der s e).2.2.1 (hh_der s e).2.2.2
        (hd0 e) (hd1 e) (hd2 e) (hd3 e)
      have hG0 : HasDerivAt (jetG0 (H s) J) (jetG1 (H s) J x) x :=
        HasDerivAt.fun_sum fun e _ => (hW e).1
      have hG1 : HasDerivAt (jetG1 (H s) J) (jetG2 (H s) J x) x :=
        HasDerivAt.fun_sum fun e _ => (hW e).2.1
      have hG2 : HasDerivAt (jetG2 (H s) J) (jetG3 (H s) J x) x :=
        HasDerivAt.fun_sum fun e _ => (hW e).2.2.1
      have hG3 : HasDerivAt (jetG3 (H s) J) (jetG4 (H s) J x) x :=
        HasDerivAt.fun_sum fun e _ => (hW e).2.2.2
      have vecMul_deriv : ∀ (f f' : E → ℝ → ℝ),
          (∀ e, HasDerivAt (f e) (f' e x) x) →
          ∀ j, HasDerivAt (fun y => K.prob.vecMul (fun e => f e y) j)
            (K.prob.vecMul (fun e => f' e x) j) x := by
        intro f f' hf j
        simp only [Matrix.vecMul, dotProduct]
        exact HasDerivAt.fun_sum fun i _ => (hf i).mul_const (K.prob i j)
      exact ⟨vecMul_deriv _ _ (fun e => hasDerivAt_q0 hG_ne (hW e).1 hG0),
        vecMul_deriv _ _ (fun e => hasDerivAt_q1 hG_ne (hW e).1 (hW e).2.1 hG0 hG1),
        vecMul_deriv _ _ (fun e => hasDerivAt_q2 hG_ne (hW e).1 (hW e).2.1 (hW e).2.2.1 hG0 hG1 hG2),
        vecMul_deriv _ _ (fun e => hasDerivAt_q3 hG_ne (hW e).1 (hW e).2.1 (hW e).2.2.1 (hW e).2.2.2
          hG0 hG1 hG2 hG3)⟩
  obtain ⟨hd0, hd1, hd2, hd3⟩ := hstate t
  let J := stateJetAt K H t
  have hG_ne : jetG0 (H t) J x ≠ 0 := by
    rw [jetG0_eq_projGainGen]
    exact (projStateGen_prob_and_gain_pos K (fun r => (H r).h0) x hh_pos t).2.2.ne'
  have hW (e : E) := hasDerivAt_jetW (H t) J e x
    (hh_der t e).1 (hh_der t e).2.1 (hh_der t e).2.2.1 (hh_der t e).2.2.2
    (hd0 e) (hd1 e) (hd2 e) (hd3 e)
  have hG0 : HasDerivAt (jetG0 (H t) J) (jetG1 (H t) J x) x :=
    HasDerivAt.fun_sum fun e _ => (hW e).1
  have hG1 : HasDerivAt (jetG1 (H t) J) (jetG2 (H t) J x) x :=
    HasDerivAt.fun_sum fun e _ => (hW e).2.1
  have hG2 : HasDerivAt (jetG2 (H t) J) (jetG3 (H t) J x) x :=
    HasDerivAt.fun_sum fun e _ => (hW e).2.2.1
  have hG3 : HasDerivAt (jetG3 (H t) J) (jetG4 (H t) J x) x :=
    HasDerivAt.fun_sum fun e _ => (hW e).2.2.2
  refine ⟨?_, hasDerivAt_q0 hG_ne hG1 hG0, hasDerivAt_q1 hG_ne hG1 hG2 hG0 hG1,
    hasDerivAt_q2 hG_ne hG1 hG2 hG3 hG0 hG1 hG2⟩
  have hlog := hG0.log hG_ne
  have hfun : (fun y => Real.log (jetG0 (H t) J y)) =
      fun y => Real.log (projGainGen K (fun s => (H s).h0) t y) := by
    funext y; rw [jetG0_eq_projGainGen]
  rw [hfun] at hlog
  simpa only [logGain1, q0] using hlog

lemma stateJetAt_zero_sum {E : Type*} [Fintype E]
    (K : Chain E) (H : ℕ → EdgeStepJet E) (x : ℝ)
    (hh_pos : ∀ s e, 0 < (H s).h0 x e) (t : ℕ) :
    let J := stateJetAt K H t
    (∑ e, J.u0 x e = 1) ∧
    (∑ e, J.u1 x e = 0) ∧
    (∑ e, J.u2 x e = 0) ∧
    (∑ e, J.u3 x e = 0) ∧
    (∑ e, J.u4 x e = 0) := by
  induction t with
  | zero => simp [stateJetAt, K.initial_total]
  | succ t ih =>
    let J := stateJetAt K H t
    have hG_ne : jetG0 (H t) J x ≠ 0 := by
      rw [jetG0_eq_projGainGen]
      exact (projStateGen_prob_and_gain_pos K (fun r => (H r).h0) x hh_pos t).2.2.ne'
    have sum_vecMul : ∀ v : E → ℝ, ∑ j, K.prob.vecMul v j = ∑ i, v i := by
      intro v; simp only [Matrix.vecMul, dotProduct]; rw [Finset.sum_comm]
      simp [← Finset.mul_sum, K.total]
    have hq0_sum : ∑ e, q0 (jetW0 (H t) J e) (jetG0 (H t) J) x = 1 := by
      simp only [q0, ← Finset.sum_div]; exact div_self hG_ne
    have hq1_sum : ∑ e, q1 (jetW0 (H t) J e) (jetW1 (H t) J e) (jetG0 (H t) J) (jetG1 (H t) J) x = 0 := by
      simp only [q1, ← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum, hq0_sum, mul_one]
      change (jetG1 (H t) J x - jetG1 (H t) J x) / jetG0 (H t) J x = 0; simp
    have hq2_sum : ∑ e, q2 (jetW0 (H t) J e) (jetW1 (H t) J e) (jetW2 (H t) J e)
        (jetG0 (H t) J) (jetG1 (H t) J) (jetG2 (H t) J) x = 0 := by
      simp only [q2, ← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum,
        hq0_sum, hq1_sum, mul_one, mul_zero, sub_zero]
      change (jetG2 (H t) J x - jetG2 (H t) J x) / jetG0 (H t) J x = 0; simp
    have hq3_sum : ∑ e, q3 (jetW0 (H t) J e) (jetW1 (H t) J e) (jetW2 (H t) J e) (jetW3 (H t) J e)
        (jetG0 (H t) J) (jetG1 (H t) J) (jetG2 (H t) J) (jetG3 (H t) J) x = 0 := by
      simp only [q3, ← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum,
        hq0_sum, hq1_sum, hq2_sum, mul_one, mul_zero, sub_zero]
      change (jetG3 (H t) J x - jetG3 (H t) J x) / jetG0 (H t) J x = 0; simp
    have hq4_sum : ∑ e, q4 (jetW0 (H t) J e) (jetW1 (H t) J e) (jetW2 (H t) J e) (jetW3 (H t) J e)
        (jetW4 (H t) J e) (jetG0 (H t) J) (jetG1 (H t) J) (jetG2 (H t) J) (jetG3 (H t) J)
        (jetG4 (H t) J) x = 0 := by
      simp only [q4, ← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum,
        hq0_sum, hq1_sum, hq2_sum, hq3_sum, mul_one, mul_zero, sub_zero]
      change (jetG4 (H t) J x - jetG4 (H t) J x) / jetG0 (H t) J x = 0; simp
    exact ⟨by simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq0_sum,
      by simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq1_sum,
      by simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq2_sum,
      by simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq3_sum,
      by simp only [stateJetAt, nextStateJet]; rw [sum_vecMul]; exact hq4_sum⟩

private lemma aestronglyMeasurable_sum_univ {E Ω : Type*} [Fintype E] [MeasurableSpace Ω]
    (μ : Measure Ω) (f : E → Ω → ℝ) (hf : ∀ e, AEStronglyMeasurable (f e) μ) :
    AEStronglyMeasurable (fun ω => ∑ e : E, f e ω) μ := by
  convert Finset.aestronglyMeasurable_sum (Finset.univ : Finset E) (fun e _ => hf e) using 1
  funext ω; simp [Finset.sum_apply]

private lemma continuousOn_sum_univ {E : Type*} [Fintype E]
    (U : Set ℝ) (f : E → ℝ → ℝ) (hf : ∀ e, ContinuousOn (f e) U) :
    ContinuousOn (fun x => ∑ e : E, f e x) U := by
  classical
  have h : ∀ s : Finset E, (∀ e ∈ s, ContinuousOn (f e) U) →
      ContinuousOn (fun x => ∑ e ∈ s, f e x) U := by
    intro s hs
    induction s using Finset.induction_on with
    | empty => simp [continuousOn_const]
    | @insert a s ha ih =>
      have hfn : (fun x => ∑ e ∈ insert a s, f e x) = fun x => f a x + ∑ e ∈ s, f e x := by
        funext x; exact Finset.sum_insert ha
      rw [hfn]
      exact (hs a (Finset.mem_insert_self a s)).add (ih fun e he => hs e (Finset.mem_insert_of_mem he))
  exact h Finset.univ fun e _ => hf e

/-- Measurability and pathwise continuity of `log projGainGen` and `logGain1..4` for any random
family of `EdgeStepJet`s positive on `U`. -/
lemma logGain_meas_and_cont_gen {E Ω : Type*} [Fintype E] [MeasurableSpace Ω]
    (μ : Measure Ω) (K : Chain E) (H : Ω → ℕ → EdgeStepJet E) (U : Set ℝ)
    (hh_pos : ∀ ω, ∀ x ∈ U, ∀ s e, 0 < (H ω s).h0 x e)
    (hh_meas : ∀ s x e,
      AEStronglyMeasurable (fun ω => (H ω s).h0 x e) μ ∧
      AEStronglyMeasurable (fun ω => (H ω s).h1 x e) μ ∧
      AEStronglyMeasurable (fun ω => (H ω s).h2 x e) μ ∧
      AEStronglyMeasurable (fun ω => (H ω s).h3 x e) μ ∧
      AEStronglyMeasurable (fun ω => (H ω s).h4 x e) μ)
    (hh_cont : ∀ ω s e,
      ContinuousOn (fun x => (H ω s).h0 x e) U ∧
      ContinuousOn (fun x => (H ω s).h1 x e) U ∧
      ContinuousOn (fun x => (H ω s).h2 x e) U ∧
      ContinuousOn (fun x => (H ω s).h3 x e) U ∧
      ContinuousOn (fun x => (H ω s).h4 x e) U) (t : ℕ) :
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => Real.log (projGainGen K (fun s => (H ω s).h0) t x)) μ) ∧
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => logGain1 K (H ω) t x) μ) ∧
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => logGain2 K (H ω) t x) μ) ∧
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => logGain3 K (H ω) t x) μ) ∧
    (∀ x ∈ U, AEStronglyMeasurable (fun ω => logGain4 K (H ω) t x) μ) ∧
    (∀ ω, ContinuousOn (logGain4 K (H ω) t) U) := by
  have hstate_m : ∀ s : ℕ, ∀ x : ℝ,
      let J := fun ω => stateJetAt K (H ω) s
      (∀ e, AEStronglyMeasurable (fun ω => (J ω).u0 x e) μ) ∧
      (∀ e, AEStronglyMeasurable (fun ω => (J ω).u1 x e) μ) ∧
      (∀ e, AEStronglyMeasurable (fun ω => (J ω).u2 x e) μ) ∧
      (∀ e, AEStronglyMeasurable (fun ω => (J ω).u3 x e) μ) ∧
      (∀ e, AEStronglyMeasurable (fun ω => (J ω).u4 x e) μ) := by
    intro s
    induction s with
    | zero =>
      intro x
      exact ⟨fun _ => aestronglyMeasurable_const, fun _ => aestronglyMeasurable_const,
        fun _ => aestronglyMeasurable_const, fun _ => aestronglyMeasurable_const,
        fun _ => aestronglyMeasurable_const⟩
    | succ s ih =>
      intro x
      obtain ⟨hm0, hm1, hm2, hm3, hm4⟩ := ih x
      let J := fun ω => stateJetAt K (H ω) s
      have hw0 (e : E) : AEStronglyMeasurable (fun ω => jetW0 (H ω s) (J ω) e x) μ :=
        (hm0 e).mul (hh_meas s x e).1
      have hw1 (e : E) : AEStronglyMeasurable (fun ω => jetW1 (H ω s) (J ω) e x) μ :=
        ((hm1 e).mul (hh_meas s x e).1).add ((hm0 e).mul (hh_meas s x e).2.1)
      have hw2 (e : E) : AEStronglyMeasurable (fun ω => jetW2 (H ω s) (J ω) e x) μ :=
        (((hm2 e).mul (hh_meas s x e).1).add (((hm1 e).const_mul 2).mul (hh_meas s x e).2.1)).add
          ((hm0 e).mul (hh_meas s x e).2.2.1)
      have hw3 (e : E) : AEStronglyMeasurable (fun ω => jetW3 (H ω s) (J ω) e x) μ :=
        ((((hm3 e).mul (hh_meas s x e).1).add (((hm2 e).const_mul 3).mul (hh_meas s x e).2.1)).add
          (((hm1 e).const_mul 3).mul (hh_meas s x e).2.2.1)).add ((hm0 e).mul (hh_meas s x e).2.2.2.1)
      have hw4 (e : E) : AEStronglyMeasurable (fun ω => jetW4 (H ω s) (J ω) e x) μ :=
        (((((hm4 e).mul (hh_meas s x e).1).add (((hm3 e).const_mul 4).mul (hh_meas s x e).2.1)).add
          (((hm2 e).const_mul 6).mul (hh_meas s x e).2.2.1)).add
            (((hm1 e).const_mul 4).mul (hh_meas s x e).2.2.2.1)).add
              ((hm0 e).mul (hh_meas s x e).2.2.2.2)
      have hg0 : AEStronglyMeasurable (fun ω => jetG0 (H ω s) (J ω) x) μ :=
        aestronglyMeasurable_sum_univ μ (fun e ω => jetW0 (H ω s) (J ω) e x) hw0
      have hg1 : AEStronglyMeasurable (fun ω => jetG1 (H ω s) (J ω) x) μ :=
        aestronglyMeasurable_sum_univ μ (fun e ω => jetW1 (H ω s) (J ω) e x) hw1
      have hg2 : AEStronglyMeasurable (fun ω => jetG2 (H ω s) (J ω) x) μ :=
        aestronglyMeasurable_sum_univ μ (fun e ω => jetW2 (H ω s) (J ω) e x) hw2
      have hg3 : AEStronglyMeasurable (fun ω => jetG3 (H ω s) (J ω) x) μ :=
        aestronglyMeasurable_sum_univ μ (fun e ω => jetW3 (H ω s) (J ω) e x) hw3
      have hg4 : AEStronglyMeasurable (fun ω => jetG4 (H ω s) (J ω) x) μ :=
        aestronglyMeasurable_sum_univ μ (fun e ω => jetW4 (H ω s) (J ω) e x) hw4
      have hq0 (e : E) := ((hw0 e).aemeasurable.div hg0.aemeasurable).aestronglyMeasurable
      have hq1 (e : E) := (((hw1 e).sub (hg1.mul (hq0 e))).aemeasurable.div hg0.aemeasurable).aestronglyMeasurable
      have hq2 (e : E) := ((((hw2 e).sub ((hg1.const_mul 2).mul (hq1 e))).sub
        (hg2.mul (hq0 e))).aemeasurable.div hg0.aemeasurable).aestronglyMeasurable
      have hq3 (e : E) := (((((hw3 e).sub ((hg1.const_mul 3).mul (hq2 e))).sub
        ((hg2.const_mul 3).mul (hq1 e))).sub (hg3.mul (hq0 e))).aemeasurable.div
          hg0.aemeasurable).aestronglyMeasurable
      have hq4 (e : E) := ((((((hw4 e).sub ((hg1.const_mul 4).mul (hq3 e))).sub
        ((hg2.const_mul 6).mul (hq2 e))).sub ((hg3.const_mul 4).mul (hq1 e))).sub
          (hg4.mul (hq0 e))).aemeasurable.div hg0.aemeasurable).aestronglyMeasurable
      have vecMul_m : ∀ f : E → Ω → ℝ, (∀ e, AEStronglyMeasurable (f e) μ) →
          ∀ j, AEStronglyMeasurable (fun ω => K.prob.vecMul (fun e => f e ω) j) μ := fun f hf j =>
        aestronglyMeasurable_sum_univ μ (fun i ω => f i ω * K.prob i j) fun i => (hf i).mul_const _
      exact ⟨vecMul_m _ hq0, vecMul_m _ hq1, vecMul_m _ hq2, vecMul_m _ hq3, vecMul_m _ hq4⟩
  have hg_m (x : ℝ) :
      let J := fun ω => stateJetAt K (H ω) t
      AEStronglyMeasurable (fun ω => jetG0 (H ω t) (J ω) x) μ ∧
      AEStronglyMeasurable (fun ω => jetG1 (H ω t) (J ω) x) μ ∧
      AEStronglyMeasurable (fun ω => jetG2 (H ω t) (J ω) x) μ ∧
      AEStronglyMeasurable (fun ω => jetG3 (H ω t) (J ω) x) μ ∧
      AEStronglyMeasurable (fun ω => jetG4 (H ω t) (J ω) x) μ := by
    obtain ⟨hm0, hm1, hm2, hm3, hm4⟩ := hstate_m t x
    let J := fun ω => stateJetAt K (H ω) t
    have hw0 (e : E) : AEStronglyMeasurable (fun ω => jetW0 (H ω t) (J ω) e x) μ :=
      (hm0 e).mul (hh_meas t x e).1
    have hw1 (e : E) : AEStronglyMeasurable (fun ω => jetW1 (H ω t) (J ω) e x) μ :=
      ((hm1 e).mul (hh_meas t x e).1).add ((hm0 e).mul (hh_meas t x e).2.1)
    have hw2 (e : E) : AEStronglyMeasurable (fun ω => jetW2 (H ω t) (J ω) e x) μ :=
      (((hm2 e).mul (hh_meas t x e).1).add (((hm1 e).const_mul 2).mul (hh_meas t x e).2.1)).add
        ((hm0 e).mul (hh_meas t x e).2.2.1)
    have hw3 (e : E) : AEStronglyMeasurable (fun ω => jetW3 (H ω t) (J ω) e x) μ :=
      ((((hm3 e).mul (hh_meas t x e).1).add (((hm2 e).const_mul 3).mul (hh_meas t x e).2.1)).add
        (((hm1 e).const_mul 3).mul (hh_meas t x e).2.2.1)).add ((hm0 e).mul (hh_meas t x e).2.2.2.1)
    have hw4 (e : E) : AEStronglyMeasurable (fun ω => jetW4 (H ω t) (J ω) e x) μ :=
      (((((hm4 e).mul (hh_meas t x e).1).add (((hm3 e).const_mul 4).mul (hh_meas t x e).2.1)).add
        (((hm2 e).const_mul 6).mul (hh_meas t x e).2.2.1)).add
          (((hm1 e).const_mul 4).mul (hh_meas t x e).2.2.2.1)).add
            ((hm0 e).mul (hh_meas t x e).2.2.2.2)
    exact ⟨aestronglyMeasurable_sum_univ μ (fun e ω => jetW0 (H ω t) (J ω) e x) hw0,
      aestronglyMeasurable_sum_univ μ (fun e ω => jetW1 (H ω t) (J ω) e x) hw1,
      aestronglyMeasurable_sum_univ μ (fun e ω => jetW2 (H ω t) (J ω) e x) hw2,
      aestronglyMeasurable_sum_univ μ (fun e ω => jetW3 (H ω t) (J ω) e x) hw3,
      aestronglyMeasurable_sum_univ μ (fun e ω => jetW4 (H ω t) (J ω) e x) hw4⟩
  have hl1_m (x : ℝ) := (((hg_m x).2.1.aemeasurable).div (hg_m x).1.aemeasurable).aestronglyMeasurable
  have hl2_m (x : ℝ) := ((((hg_m x).2.2.1.sub ((hg_m x).2.1.mul (hl1_m x))).aemeasurable).div
    (hg_m x).1.aemeasurable).aestronglyMeasurable
  have hl3_m (x : ℝ) := (((((hg_m x).2.2.2.1.sub (((hg_m x).2.1.const_mul 2).mul (hl2_m x))).sub
    ((hg_m x).2.2.1.mul (hl1_m x))).aemeasurable).div (hg_m x).1.aemeasurable).aestronglyMeasurable
  have hl4_m (x : ℝ) := ((((((hg_m x).2.2.2.2.sub (((hg_m x).2.1.const_mul 3).mul (hl3_m x))).sub
    (((hg_m x).2.2.1.const_mul 3).mul (hl2_m x))).sub ((hg_m x).2.2.2.1.mul (hl1_m x))).aemeasurable).div
      (hg_m x).1.aemeasurable).aestronglyMeasurable
  have hlog_m (x : ℝ) :
      AEStronglyMeasurable (fun ω => Real.log (projGainGen K (fun s => (H ω s).h0) t x)) μ := by
    have hfun : (fun ω => Real.log (projGainGen K (fun s => (H ω s).h0) t x)) =
        fun ω => Real.log (jetG0 (H ω t) (stateJetAt K (H ω) t) x) := by
      funext ω; exact congrArg Real.log (jetG0_eq_projGainGen K (H ω) t x).symm
    rw [hfun]
    exact ((hg_m x).1.aemeasurable.log).aestronglyMeasurable
  have hcont_all (ω : Ω) : ContinuousOn (logGain4 K (H ω) t) U := by
    have hstate_c : ∀ s : ℕ,
        let J := stateJetAt K (H ω) s
        (∀ e, ContinuousOn (fun x => J.u0 x e) U) ∧
        (∀ e, ContinuousOn (fun x => J.u1 x e) U) ∧
        (∀ e, ContinuousOn (fun x => J.u2 x e) U) ∧
        (∀ e, ContinuousOn (fun x => J.u3 x e) U) ∧
        (∀ e, ContinuousOn (fun x => J.u4 x e) U) := by
      intro s
      induction s with
      | zero =>
        exact ⟨fun _ => continuousOn_const, fun _ => continuousOn_const,
          fun _ => continuousOn_const, fun _ => continuousOn_const, fun _ => continuousOn_const⟩
      | succ s ih =>
        obtain ⟨hc0, hc1, hc2, hc3, hc4⟩ := ih
        let Js := stateJetAt K (H ω) s
        have hG_ne : ∀ x ∈ U, jetG0 (H ω s) Js x ≠ 0 := by
          intro x hx; rw [jetG0_eq_projGainGen]
          exact (projStateGen_prob_and_gain_pos K (fun r => (H ω r).h0) x (hh_pos ω x hx) s).2.2.ne'
        have hw0 (e : E) : ContinuousOn (jetW0 (H ω s) Js e) U := (hc0 e).mul (hh_cont ω s e).1
        have hw1 (e : E) : ContinuousOn (jetW1 (H ω s) Js e) U :=
          ((hc1 e).mul (hh_cont ω s e).1).add ((hc0 e).mul (hh_cont ω s e).2.1)
        have hw2 (e : E) : ContinuousOn (jetW2 (H ω s) Js e) U :=
          (((hc2 e).mul (hh_cont ω s e).1).add ((continuousOn_const.mul (hc1 e)).mul (hh_cont ω s e).2.1)).add
            ((hc0 e).mul (hh_cont ω s e).2.2.1)
        have hw3 (e : E) : ContinuousOn (jetW3 (H ω s) Js e) U :=
          ((((hc3 e).mul (hh_cont ω s e).1).add ((continuousOn_const.mul (hc2 e)).mul (hh_cont ω s e).2.1)).add
            ((continuousOn_const.mul (hc1 e)).mul (hh_cont ω s e).2.2.1)).add ((hc0 e).mul (hh_cont ω s e).2.2.2.1)
        have hw4 (e : E) : ContinuousOn (jetW4 (H ω s) Js e) U :=
          (((((hc4 e).mul (hh_cont ω s e).1).add ((continuousOn_const.mul (hc3 e)).mul (hh_cont ω s e).2.1)).add
            ((continuousOn_const.mul (hc2 e)).mul (hh_cont ω s e).2.2.1)).add
              ((continuousOn_const.mul (hc1 e)).mul (hh_cont ω s e).2.2.2.1)).add
                ((hc0 e).mul (hh_cont ω s e).2.2.2.2)
        have hg0 : ContinuousOn (jetG0 (H ω s) Js) U :=
          continuousOn_sum_univ U (fun e => jetW0 (H ω s) Js e) hw0
        have hg1 : ContinuousOn (jetG1 (H ω s) Js) U :=
          continuousOn_sum_univ U (fun e => jetW1 (H ω s) Js e) hw1
        have hg2 : ContinuousOn (jetG2 (H ω s) Js) U :=
          continuousOn_sum_univ U (fun e => jetW2 (H ω s) Js e) hw2
        have hg3 : ContinuousOn (jetG3 (H ω s) Js) U :=
          continuousOn_sum_univ U (fun e => jetW3 (H ω s) Js e) hw3
        have hg4 : ContinuousOn (jetG4 (H ω s) Js) U :=
          continuousOn_sum_univ U (fun e => jetW4 (H ω s) Js e) hw4
        have hq0 (e : E) := (hw0 e).div hg0 hG_ne
        have hq1 (e : E) := ((hw1 e).sub (hg1.mul (hq0 e))).div hg0 hG_ne
        have hq2 (e : E) := (((hw2 e).sub ((continuousOn_const.mul hg1).mul (hq1 e))).sub
          (hg2.mul (hq0 e))).div hg0 hG_ne
        have hq3 (e : E) := ((((hw3 e).sub ((continuousOn_const.mul hg1).mul (hq2 e))).sub
          ((continuousOn_const.mul hg2).mul (hq1 e))).sub (hg3.mul (hq0 e))).div hg0 hG_ne
        have hq4 (e : E) := (((((hw4 e).sub ((continuousOn_const.mul hg1).mul (hq3 e))).sub
          ((continuousOn_const.mul hg2).mul (hq2 e))).sub ((continuousOn_const.mul hg3).mul (hq1 e))).sub
            (hg4.mul (hq0 e))).div hg0 hG_ne
        have vecMul_c : ∀ f : E → ℝ → ℝ, (∀ e, ContinuousOn (f e) U) →
            ∀ j, ContinuousOn (fun x => K.prob.vecMul (fun e => f e x) j) U := fun f hf j =>
          continuousOn_sum_univ U (fun i x => f i x * K.prob i j) fun i => (hf i).mul continuousOn_const
        exact ⟨vecMul_c _ hq0, vecMul_c _ hq1, vecMul_c _ hq2, vecMul_c _ hq3, vecMul_c _ hq4⟩
    obtain ⟨hc0, hc1, hc2, hc3, hc4⟩ := hstate_c t
    let Jt := stateJetAt K (H ω) t
    have hG_ne : ∀ x ∈ U, jetG0 (H ω t) Jt x ≠ 0 := by
      intro x hx; rw [jetG0_eq_projGainGen]
      exact (projStateGen_prob_and_gain_pos K (fun r => (H ω r).h0) x (hh_pos ω x hx) t).2.2.ne'
    have hw0 (e : E) : ContinuousOn (jetW0 (H ω t) Jt e) U := (hc0 e).mul (hh_cont ω t e).1
    have hw1 (e : E) : ContinuousOn (jetW1 (H ω t) Jt e) U :=
      ((hc1 e).mul (hh_cont ω t e).1).add ((hc0 e).mul (hh_cont ω t e).2.1)
    have hw2 (e : E) : ContinuousOn (jetW2 (H ω t) Jt e) U :=
      (((hc2 e).mul (hh_cont ω t e).1).add ((continuousOn_const.mul (hc1 e)).mul (hh_cont ω t e).2.1)).add
        ((hc0 e).mul (hh_cont ω t e).2.2.1)
    have hw3 (e : E) : ContinuousOn (jetW3 (H ω t) Jt e) U :=
      ((((hc3 e).mul (hh_cont ω t e).1).add ((continuousOn_const.mul (hc2 e)).mul (hh_cont ω t e).2.1)).add
        ((continuousOn_const.mul (hc1 e)).mul (hh_cont ω t e).2.2.1)).add ((hc0 e).mul (hh_cont ω t e).2.2.2.1)
    have hw4 (e : E) : ContinuousOn (jetW4 (H ω t) Jt e) U :=
      (((((hc4 e).mul (hh_cont ω t e).1).add ((continuousOn_const.mul (hc3 e)).mul (hh_cont ω t e).2.1)).add
        ((continuousOn_const.mul (hc2 e)).mul (hh_cont ω t e).2.2.1)).add
          ((continuousOn_const.mul (hc1 e)).mul (hh_cont ω t e).2.2.2.1)).add
            ((hc0 e).mul (hh_cont ω t e).2.2.2.2)
    have hg0 : ContinuousOn (jetG0 (H ω t) Jt) U :=
      continuousOn_sum_univ U (fun e => jetW0 (H ω t) Jt e) hw0
    have hg1 : ContinuousOn (jetG1 (H ω t) Jt) U :=
      continuousOn_sum_univ U (fun e => jetW1 (H ω t) Jt e) hw1
    have hg2 : ContinuousOn (jetG2 (H ω t) Jt) U :=
      continuousOn_sum_univ U (fun e => jetW2 (H ω t) Jt e) hw2
    have hg3 : ContinuousOn (jetG3 (H ω t) Jt) U :=
      continuousOn_sum_univ U (fun e => jetW3 (H ω t) Jt e) hw3
    have hg4 : ContinuousOn (jetG4 (H ω t) Jt) U :=
      continuousOn_sum_univ U (fun e => jetW4 (H ω t) Jt e) hw4
    have hl1 := hg1.div hg0 hG_ne
    have hl2 := (hg2.sub (hg1.mul hl1)).div hg0 hG_ne
    have hl3 := ((hg3.sub ((continuousOn_const.mul hg1).mul hl2)).sub (hg2.mul hl1)).div hg0 hG_ne
    exact (((hg4.sub ((continuousOn_const.mul hg1).mul hl3)).sub
      ((continuousOn_const.mul hg2).mul hl2)).sub (hg3.mul hl1)).div hg0 hG_ne
  exact ⟨fun x _ => hlog_m x, fun x _ => hl1_m x, fun x _ => hl2_m x,
    fun x _ => hl3_m x, fun x _ => hl4_m x, hcont_all⟩

/-- **Unified projective Jacobian-plus-forcing step bound.**
Whenever `h0 e ∈ [exp(-η), exp(η)]` and `G0 ∈ [exp(-η), exp(η)]` (`η ≥ 0`), `∑ z = 0`, and
`l1 q0f = 1`, the next-step zero-sum derivative satisfies
`gauge K m v_next ≤ (θ₀ * exp(c₄ * η)) * gauge K m z + m * l1 F`
with `θ₀ = 1 - 1 / (2 * m)` and `c₄ = 1 + max 1 (3 * m / θ₀)`. -/
lemma gauge_jacobian_forcing_step {E : Type*} [Fintype E] [DecidableEq E]
    (K : Chain E) (m : ℕ) (hm : 0 < m)
    (hblock : ∀ z : E → ℝ, (∑ i, z i) = 0 → l1 ((K.prob ^ m).vecMul z) ≤ l1 z / 2)
    (h0 z q0f F : E → ℝ) (G0 η : ℝ) (hη : 0 ≤ η)
    (hh0 : ∀ e, Real.exp (-η) ≤ h0 e ∧ h0 e ≤ Real.exp η)
    (hG0 : Real.exp (-η) ≤ G0 ∧ G0 ≤ Real.exp η)
    (hz_sum : ∑ e, z e = 0) (hq0_l1 : l1 q0f = 1) :
    let θ₀ : ℝ := 1 - 1 / (2 * (m : ℝ))
    let c₄ : ℝ := 1 + max 1 (3 * (m : ℝ) / θ₀)
    gauge K m (K.prob.vecMul (fun e =>
      (z e * h0 e - (∑ i, z i * h0 i) * q0f e) / G0 + F e)) ≤
      (θ₀ * Real.exp (c₄ * η)) * gauge K m z + (m : ℝ) * l1 F := by
  intro θ₀ c₄
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hθ₀_pos : 0 < θ₀ := by
    dsimp [θ₀]
    have : 1 / (2 * (m : ℝ)) ≤ 1 / 2 :=
      one_div_le_one_div_of_le (by norm_num) (by linarith)
    linarith
  have hG0_pos : 0 < G0 := lt_of_lt_of_le (Real.exp_pos _) hG0.1
  have hexp_ge_one : 1 ≤ Real.exp η := Real.one_le_exp hη
  have hexp_neg_inv : (Real.exp (-η))⁻¹ = Real.exp η := by rw [Real.exp_neg, inv_inv]
  have hinv_G0 : G0⁻¹ ≤ Real.exp η := by
    rw [← hexp_neg_inv]
    exact inv_anti₀ (Real.exp_pos _) hG0.1
  have hexp_sub : 1 - Real.exp (-η) ≤ Real.exp η - 1 := by
    have hprod : Real.exp (-η) * Real.exp η = 1 := by rw [← Real.exp_add]; simp
    have hle : Real.exp (-η) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    nlinarith
  let d : ℝ := ∑ i, z i * (h0 i - 1)
  have hd_eq : ∑ i, z i * h0 i = d := by
    dsimp [d]; simp only [mul_sub, mul_one, Finset.sum_sub_distrib, hz_sum, sub_zero]
  let rem : E → ℝ := fun e => (z e * (h0 e - G0) - d * q0f e) / G0 + F e
  have hsplit : K.prob.vecMul (fun e => (z e * h0 e - (∑ i, z i * h0 i) * q0f e) / G0 + F e) =
      fun j => K.prob.vecMul z j + K.prob.vecMul rem j := by
    funext j
    rw [hd_eq]
    simp only [Matrix.vecMul, dotProduct, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    dsimp [rem]; field_simp [hG0_pos.ne']; ring
  rw [hsplit]
  have htri := gauge_add_le K m (K.prob.vecMul z) (K.prob.vecMul rem)
  have hmain := gauge_contraction K m hm hblock z hz_sum
  have hh0_G0_abs : ∀ e, |h0 e - G0| ≤ 2 * (Real.exp η - 1) := by
    intro e
    apply abs_le.mpr
    constructor <;> linarith [(hh0 e).1, (hh0 e).2, hG0.1, hG0.2, hexp_sub]
  have hh0_one_abs : ∀ e, |h0 e - 1| ≤ Real.exp η - 1 := by
    intro e
    apply abs_le.mpr
    constructor <;> linarith [(hh0 e).1, (hh0 e).2, hexp_sub]
  have hd_le : |d| ≤ (Real.exp η - 1) * l1 z := by
    dsimp [d]; unfold l1
    rw [Finset.mul_sum]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
    rw [abs_mul, mul_comm |z i|]
    exact mul_le_mul_of_nonneg_right (hh0_one_abs i) (abs_nonneg _)
  have hzdiff_l1 : l1 (fun e => z e * (h0 e - G0)) ≤ 2 * (Real.exp η - 1) * l1 z := by
    unfold l1
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun e _ => ?_
    rw [abs_mul, mul_comm |z e|]
    exact mul_le_mul_of_nonneg_right (hh0_G0_abs e) (abs_nonneg _)
  have hnum_l1 : l1 (fun e => z e * (h0 e - G0) - d * q0f e) ≤ 3 * (Real.exp η - 1) * l1 z := by
    have h1 := l1_sub_le (fun e => z e * (h0 e - G0)) (fun e => d * q0f e)
    rw [l1_smul, hq0_l1, mul_one] at h1
    linarith
  have hdiv_l1 : l1 (fun e => (z e * (h0 e - G0) - d * q0f e) / G0) ≤
      Real.exp η * (3 * (Real.exp η - 1) * l1 z) := by
    calc l1 (fun e => (z e * (h0 e - G0) - d * q0f e) / G0)
        = G0⁻¹ * l1 (fun e => z e * (h0 e - G0) - d * q0f e) := by
          unfold l1; rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun e _ => ?_
          dsimp only
          rw [div_eq_inv_mul, abs_mul, abs_of_pos (inv_pos.mpr hG0_pos)]
      _ ≤ Real.exp η * (3 * (Real.exp η - 1) * l1 z) :=
          mul_le_mul hinv_G0 hnum_l1 (l1_nonneg _) (Real.exp_pos _).le
  have hrem_l1 : l1 rem ≤ 3 * Real.exp η * (Real.exp η - 1) * gauge K m z + l1 F := by
    have h1 := l1_add_le (fun e => (z e * (h0 e - G0) - d * q0f e) / G0) F
    have hlow := gauge_lower K m hm z
    have hex_nn : 0 ≤ 3 * Real.exp η * (Real.exp η - 1) :=
      mul_nonneg (by positivity) (by linarith)
    nlinarith [mul_le_mul_of_nonneg_left hlow hex_nn]
  have hrem_gauge : gauge K m (K.prob.vecMul rem) ≤
      3 * (m : ℝ) * Real.exp η * (Real.exp η - 1) * gauge K m z + (m : ℝ) * l1 F := by
    calc gauge K m (K.prob.vecMul rem)
        ≤ (m : ℝ) * l1 rem := gauge_step_le K m rem
      _ ≤ (m : ℝ) * (3 * Real.exp η * (Real.exp η - 1) * gauge K m z + l1 F) :=
          mul_le_mul_of_nonneg_left hrem_l1 hmR.le
      _ = 3 * (m : ℝ) * Real.exp η * (Real.exp η - 1) * gauge K m z + (m : ℝ) * l1 F := by ring
  have habs := exponential_absorption θ₀ (3 * (m : ℝ)) η hθ₀_pos hη
  have hcoef_le : θ₀ + 3 * (m : ℝ) * Real.exp η * (Real.exp η - 1) ≤
      θ₀ * Real.exp (c₄ * η) := by
    have hle1 : θ₀ + 3 * (m : ℝ) * Real.exp η * (Real.exp η - 1) ≤
        Real.exp η * (θ₀ + 3 * (m : ℝ) * (Real.exp η - 1)) := by
      nlinarith [hθ₀_pos, hexp_ge_one]
    exact hle1.trans habs
  have hgz_nn := gauge_nonneg K m z
  calc gauge K m (fun j => K.prob.vecMul z j + K.prob.vecMul rem j)
      ≤ gauge K m (K.prob.vecMul z) + gauge K m (K.prob.vecMul rem) := htri
    _ ≤ θ₀ * gauge K m z +
          (3 * (m : ℝ) * Real.exp η * (Real.exp η - 1) * gauge K m z + (m : ℝ) * l1 F) :=
        add_le_add hmain hrem_gauge
    _ = (θ₀ + 3 * (m : ℝ) * Real.exp η * (Real.exp η - 1)) * gauge K m z + (m : ℝ) * l1 F := by
        ring
    _ ≤ (θ₀ * Real.exp (c₄ * η)) * gauge K m z + (m : ℝ) * l1 F :=
        add_le_add_left (mul_le_mul_of_nonneg_right hcoef_le hgz_nn) _

end UnifiedProjectiveJet

end Dobson.P1bA_LognormalRemainders
