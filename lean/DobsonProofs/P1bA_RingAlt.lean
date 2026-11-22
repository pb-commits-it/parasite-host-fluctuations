import DobsonProofs.Coverage1Gamma
import DobsonProofs.Coverage2Alternatives
import DobsonProofs.Coverage6Remainders
import DobsonProofs.ReferenceRing

/-! # Part A: split-transfer and staying reference-chain coefficients

This module develops geometric-ring mixing, discounted resolvents and
finite-path second-derivative cost formulas for split transfers and staying
entries. These calculations concern the explicit reference chains.
Coverage correction, 2026-11-22: this module is uncredited in the version-2
inventory. `prop:ring-alt` remains partial-P1; the credited physical expansion
and renewal bridge remain supplied. No row upgrade follows from these helpers. -/

namespace Dobson.P1bA_RingAlt

open scoped BigOperators Topology
open Filter MeasureTheory Set
open Dobson.Coverage1Gamma Dobson.FiniteLineageVariance

/-! ## Section 1: Geometric-ring state matrix, cyclic distance, and Doeblin mixing -/

section RingStateMixing

/-- State-to-state transition matrix of the `n`-stage geometric ring with residence times `τ`. -/
noncomputable def ringStateProb (n : ℕ) (τ : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j =>
    (1 - 1 / τ i) * (if i = j then 1 else 0) +
      (1 / τ i) * (if finRotate n i = j then 1 else 0)

/-- Stationary state distribution `π_i = τ_i / 𝒯` of the `n`-stage geometric ring. -/
noncomputable def ringStateInitial (n : ℕ) (τ : Fin n → ℝ) (i : Fin n) : ℝ :=
  τ i / genTime n τ

lemma genTime_pos_of_ge_one (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k) :
    0 < genTime n τ :=
  Finset.sum_pos (fun k _ => lt_of_lt_of_le zero_lt_one (hτ k)) ⟨⟨0, by omega⟩, Finset.mem_univ _⟩

lemma ringStateProb_nonneg (n : ℕ) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k) (i j : Fin n) :
    0 ≤ ringStateProb n τ i j := by
  have hpos : 0 < τ i := lt_of_lt_of_le zero_lt_one (hτ i)
  have hstay : 0 ≤ 1 - 1 / τ i := by rw [sub_nonneg, div_le_one hpos]; exact hτ i
  have htr : 0 ≤ 1 / τ i := by positivity
  unfold ringStateProb
  apply add_nonneg
  · split_ifs <;> nlinarith
  · split_ifs <;> nlinarith

lemma ringStateProb_row_sum (n : ℕ) (τ : Fin n → ℝ) (i : Fin n) :
    ∑ j : Fin n, ringStateProb n τ i j = 1 := by
  unfold ringStateProb
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte, mul_one]
  ring

lemma ringStateProb_mulVec (n : ℕ) (τ : Fin n → ℝ) (f : Fin n → ℝ) (i : Fin n) :
    (ringStateProb n τ).mulVec f i =
      (1 - 1 / τ i) * f i + (1 / τ i) * f (finRotate n i) := by
  simp only [Matrix.mulVec, dotProduct, ringStateProb, add_mul, Finset.sum_add_distrib]
  congr 1
  · simp [ite_mul, zero_mul]
  · simp [ite_mul, zero_mul]

lemma ringStateProb_stationary (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k) :
    (ringStateProb n τ).vecMul (ringStateInitial n τ) = ringStateInitial n τ := by
  have hT : 0 < genTime n τ := genTime_pos_of_ge_one n hn τ hτ
  funext j
  simp only [Matrix.vecMul, dotProduct, ringStateProb, ringStateInitial, mul_add,
    Finset.sum_add_distrib]
  have h1 : (∑ i : Fin n, (τ i / genTime n τ) * ((1 - 1 / τ i) * if i = j then 1 else 0)) =
      (τ j - 1) / genTime n τ := by
    have hfn : ∀ i : Fin n,
        (τ i / genTime n τ) * ((1 - 1 / τ i) * if i = j then 1 else 0) =
          if i = j then (τ i - 1) / genTime n τ else 0 := by
      intro i
      have hne : τ i ≠ 0 := (lt_of_lt_of_le zero_lt_one (hτ i)).ne'
      split_ifs
      · field_simp [hne, hT.ne']
      · ring
    simp_rw [hfn]
    simp
  have h2 : (∑ i : Fin n, (τ i / genTime n τ) * ((1 / τ i) * if finRotate n i = j then 1 else 0)) =
      1 / genTime n τ := by
    have hfn : ∀ i : Fin n,
        (τ i / genTime n τ) * ((1 / τ i) * if finRotate n i = j then 1 else 0) =
          if i = (finRotate n).symm j then 1 / genTime n τ else 0 := by
      intro i
      have hne : τ i ≠ 0 := (lt_of_lt_of_le zero_lt_one (hτ i)).ne'
      have heq : (finRotate n i = j) ↔ (i = (finRotate n).symm j) :=
        ((finRotate n).eq_symm_apply).symm
      by_cases h : finRotate n i = j
      · have hi_eq : i = (finRotate n).symm j := heq.mp h
        rw [h]
        simp only [hi_eq, ↓reduceIte]
        have hne' : τ ((finRotate n).symm j) ≠ 0 := hi_eq ▸ hne
        field_simp [hne', hT.ne']
      · have hi_ne : i ≠ (finRotate n).symm j := mt heq.mpr h
        simp only [h, hi_ne, ↓reduceIte]
        ring
    simp_rw [hfn]
    simp
  rw [h1, h2]
  field_simp [hT.ne']
  ring

lemma ringStateProb_pow_nonneg (n : ℕ) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (m : ℕ) (i j : Fin n) : 0 ≤ (ringStateProb n τ ^ m) i j := by
  induction m generalizing i j with
  | zero => simp only [pow_zero, Matrix.one_apply]; split_ifs <;> norm_num
  | succ m ih =>
    rw [pow_succ', Matrix.mul_apply]
    exact Finset.sum_nonneg fun k _ => mul_nonneg (ringStateProb_nonneg n τ hτ i k) (ih k j)

lemma ringStateProb_pow_row_sum (n : ℕ) (τ : Fin n → ℝ) (m : ℕ) (i : Fin n) :
    ∑ j : Fin n, (ringStateProb n τ ^ m) i j = 1 := by
  induction m generalizing i with
  | zero => simp [Matrix.one_apply]
  | succ m ih =>
    simp only [pow_succ', Matrix.mul_apply]
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, ih, mul_one, ringStateProb_row_sum]

lemma finRotate_iterate_val {n : ℕ} (hn : 1 ≤ n) (r : ℕ) (i : Fin n) :
    (((finRotate n)^[r] i : Fin n) : ℕ) = (i.val + r) % n := by
  have : NeZero n := ⟨by omega⟩
  induction r generalizing i with
  | zero =>
    simp [Nat.mod_eq_of_lt i.isLt]
  | succ r ih =>
    rw [Function.iterate_succ_apply, ih]
    have hrot : ((finRotate n i : Fin n) : ℕ) = (i.val + 1) % n := by
      rw [finRotate_apply, Fin.val_add, Fin.val_one']
      simp [Nat.add_mod]
    rw [hrot, Nat.mod_add_mod, Nat.add_assoc, Nat.add_comm 1 r]

/-- Forward cyclic distance `(j - i) mod n` in `{0, ..., n - 1}`. -/
def cycDist {n : ℕ} (i j : Fin n) : ℕ :=
  (j.val + n - i.val) % n

lemma cycDist_lt {n : ℕ} (hn : 1 ≤ n) (i j : Fin n) : cycDist i j < n :=
  Nat.mod_lt _ (by omega)

lemma finRotate_iterate_cycDist {n : ℕ} (hn : 1 ≤ n) (i j : Fin n) :
    (finRotate n)^[cycDist i j] i = j := by
  apply Fin.ext
  rw [finRotate_iterate_val hn]
  dsimp [cycDist]
  rw [Nat.add_mod_mod]
  have hi := i.isLt
  have hj := j.isLt
  have h1 : i.val + (j.val + n - i.val) = j.val + n := by omega
  rw [h1, Nat.add_mod_right, Nat.mod_eq_of_lt hj]

lemma ringStateProb_pow_rotate_pos (n : ℕ) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (r : ℕ) (i : Fin n) :
    0 < (ringStateProb n τ ^ r) i ((finRotate n)^[r] i) := by
  induction r generalizing i with
  | zero => simp
  | succ r ih =>
    rw [pow_succ', Matrix.mul_apply, Function.iterate_succ_apply]
    have hstep : 0 < ringStateProb n τ i (finRotate n i) := by
      have hpos : 0 < τ i := lt_of_lt_of_le zero_lt_one (hτ i)
      have hstay : 0 ≤ (1 - 1 / τ i) * (if i = finRotate n i then 1 else 0) := by
        have hs : 0 ≤ 1 - 1 / τ i := by rw [sub_nonneg, div_le_one hpos]; exact hτ i
        split_ifs <;> nlinarith
      unfold ringStateProb
      simp only [↓reduceIte, mul_one]
      linarith [one_div_pos.mpr hpos]
    exact Finset.sum_pos'
      (fun k _ => mul_nonneg (ringStateProb_nonneg n τ hτ i k)
        (ringStateProb_pow_nonneg n τ hτ r k _))
      ⟨finRotate n i, Finset.mem_univ _, mul_pos hstep (ih (finRotate n i))⟩

lemma ringStateProb_pow_stay_pos (n : ℕ) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (k_s : Fin n) (hk_s : 1 < τ k_s) (d : ℕ) :
    0 < (ringStateProb n τ ^ d) k_s k_s := by
  induction d with
  | zero => simp
  | succ d ih =>
    rw [pow_succ', Matrix.mul_apply]
    have hstay_pos : 0 < ringStateProb n τ k_s k_s := by
      have hpos : 0 < τ k_s := by linarith
      have hs : 0 < 1 - 1 / τ k_s := by rw [sub_pos, div_lt_one hpos]; exact hk_s
      have htr : 0 ≤ (1 / τ k_s) * (if finRotate n k_s = k_s then 1 else 0) := by
        split_ifs <;> positivity
      unfold ringStateProb
      simp only [↓reduceIte, mul_one]
      linarith
    exact Finset.sum_pos'
      (fun k _ => mul_nonneg (ringStateProb_nonneg n τ hτ k_s k)
        (ringStateProb_pow_nonneg n τ hτ d k k_s))
      ⟨k_s, Finset.mem_univ _, mul_pos hstay_pos ih⟩

lemma ringStateProb_pow_mul_entry_ge (n : ℕ) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (a b : ℕ) (i k j : Fin n) :
    (ringStateProb n τ ^ a) i k * (ringStateProb n τ ^ b) k j ≤
      (ringStateProb n τ ^ (a + b)) i j := by
  rw [pow_add, Matrix.mul_apply]
  exact Finset.single_le_sum
    (fun x _ => mul_nonneg (ringStateProb_pow_nonneg n τ hτ a i x)
      (ringStateProb_pow_nonneg n τ hτ b x j))
    (Finset.mem_univ k)

private noncomputable def l1Fin {n : ℕ} (z : Fin n → ℝ) : ℝ := ∑ i : Fin n, |z i|

private lemma l1Fin_step (n : ℕ) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k) (z : Fin n → ℝ) :
    l1Fin ((ringStateProb n τ).vecMul z) ≤ l1Fin z := by
  unfold l1Fin
  calc ∑ j, |∑ i, z i * ringStateProb n τ i j|
      ≤ ∑ j, ∑ i, |z i| * ringStateProb n τ i j := by
        refine Finset.sum_le_sum fun j _ => ?_
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
        rw [abs_mul, abs_of_nonneg (ringStateProb_nonneg n τ hτ i j)]
    _ = ∑ i, |z i| := by
        rw [Finset.sum_comm]
        simp [← Finset.mul_sum, ringStateProb_row_sum]

private lemma l1Fin_pow (n : ℕ) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k) (m : ℕ) (z : Fin n → ℝ) :
    l1Fin ((ringStateProb n τ ^ m).vecMul z) ≤ l1Fin z := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [pow_succ, ← Matrix.vecMul_vecMul]
    exact (l1Fin_step n τ hτ _).trans ih

/-- **Doeblin geometric mixing of the primitive `n`-stage geometric ring.**
Under `n = 1 ∨ ∃ k_s, 1 < τ k_s`, there exist `Ku ≥ 0` and `0 ≤ β < 1` such that
`|(ringStateProb n τ ^ m) i j - ringStateInitial n τ j| ≤ Ku * β ^ m` for all `m, i, j`. -/
theorem ringState_geometric_mixing (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (hprim : n = 1 ∨ ∃ k_s : Fin n, 1 < τ k_s) :
    ∃ Ku β : ℝ, 0 ≤ Ku ∧ 0 ≤ β ∧ β < 1 ∧
      ∀ m : ℕ, ∀ i j : Fin n,
        |(ringStateProb n τ ^ m) i j - ringStateInitial n τ j| ≤ Ku * β ^ m := by
  by_cases hn1 : n = 1
  · subst hn1
    refine ⟨0, 0, le_rfl, le_rfl, zero_lt_one, fun m i j => ?_⟩
    have hi : i = ⟨0, by omega⟩ := Fin.ext (by omega)
    have hj : j = ⟨0, by omega⟩ := Fin.ext (by omega)
    subst hi; subst hj
    have hrow := ringStateProb_pow_row_sum 1 τ m ⟨0, by omega⟩
    have hsum1 : (∑ x : Fin 1, (ringStateProb 1 τ ^ m) ⟨0, by omega⟩ x) =
        (ringStateProb 1 τ ^ m) ⟨0, by omega⟩ ⟨0, by omega⟩ := by
      rw [Fin.sum_univ_one]; rfl
    have hinit1 : ringStateInitial 1 τ ⟨0, by omega⟩ = 1 := by
      unfold ringStateInitial genTime
      rw [Fin.sum_univ_one]
      have hpos : 0 < τ ⟨0, by omega⟩ := lt_of_lt_of_le zero_lt_one (hτ _)
      exact div_self hpos.ne'
    rw [hsum1] at hrow
    rw [hrow, hinit1, sub_self, abs_zero, zero_mul]
  · have hn2 : 2 ≤ n := by omega
    rcases hprim with rfl | ⟨k_s, hk_s⟩
    · omega
    let M : ℕ := 2 * n - 1
    have hM_pos : 1 ≤ M := by dsimp [M]; omega
    let Q : Matrix (Fin n) (Fin n) ℝ := ringStateProb n τ ^ M
    have hQ_pos : ∀ i j : Fin n, 0 < Q i j := by
      intro i j
      let r := cycDist i k_s
      let s := cycDist k_s j
      have hr_lt : r < n := cycDist_lt hn i k_s
      have hs_lt : s < n := cycDist_lt hn k_s j
      let d : ℕ := M - (r + s)
      have hrds : r + d + s = M := by dsimp [d, M]; omega
      have h1 : 0 < (ringStateProb n τ ^ r) i k_s := by
        have hrot := finRotate_iterate_cycDist hn i k_s
        simpa [r, hrot] using ringStateProb_pow_rotate_pos n τ hτ r i
      have h2 : 0 < (ringStateProb n τ ^ d) k_s k_s :=
        ringStateProb_pow_stay_pos n τ hτ k_s hk_s d
      have h3 : 0 < (ringStateProb n τ ^ s) k_s j := by
        have hrot := finRotate_iterate_cycDist hn k_s j
        simpa [s, hrot] using ringStateProb_pow_rotate_pos n τ hτ s k_s
      have hle1 := ringStateProb_pow_mul_entry_ge n τ hτ r d i k_s k_s
      have hle2 := ringStateProb_pow_mul_entry_ge n τ hτ (r + d) s i k_s j
      dsimp [Q]
      rw [← hrds]
      exact lt_of_lt_of_le (mul_pos (mul_pos h1 h2) h3)
        ((mul_le_mul_of_nonneg_right hle1 h3.le).trans hle2)
    have hne_prod : (Finset.univ : Finset (Fin n × Fin n)).Nonempty :=
      ⟨(⟨0, by omega⟩, ⟨0, by omega⟩), Finset.mem_univ _⟩
    let ε : ℝ := Finset.univ.inf' hne_prod (fun p => Q p.1 p.2)
    have hε_pos : 0 < ε := (Finset.lt_inf'_iff hne_prod).mpr fun p _ => hQ_pos p.1 p.2
    have hε_le : ∀ i j : Fin n, ε ≤ Q i j := fun i j =>
      Finset.inf'_le (fun p => Q p.1 p.2) (Finset.mem_univ (i, j))
    have hnε_le_one : (n : ℝ) * ε ≤ 1 := by
      let i0 : Fin n := ⟨0, by omega⟩
      calc (n : ℝ) * ε = ∑ _j : Fin n, ε := by simp
        _ ≤ ∑ j : Fin n, Q i0 j := Finset.sum_le_sum fun j _ => hε_le i0 j
        _ = 1 := ringStateProb_pow_row_sum n τ M i0
    let ρ₀ : ℝ := max (1 / 2) (1 - (n : ℝ) * ε)
    have hρ₀_pos : 0 < ρ₀ := lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1 / 2) (le_max_left _ _)
    have hρ₀_lt1 : ρ₀ < 1 := by
      dsimp [ρ₀]
      apply max_lt (by norm_num)
      have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
      nlinarith [mul_pos hn_pos hε_pos]
    have hQ_contract : ∀ z : Fin n → ℝ, (∑ i, z i) = 0 → l1Fin (Q.vecMul z) ≤ ρ₀ * l1Fin z := by
      intro z hz
      have hrewrite : ∀ j : Fin n, Q.vecMul z j = ∑ i : Fin n, z i * (Q i j - ε) := by
        intro j
        simp only [Matrix.vecMul, dotProduct, mul_sub, Finset.sum_sub_distrib,
          ← Finset.sum_mul, hz, zero_mul, sub_zero]
      have hnonneg : ∀ i j : Fin n, 0 ≤ Q i j - ε := fun i j => sub_nonneg.mpr (hε_le i j)
      have hrow_rem : ∀ i : Fin n, (∑ j : Fin n, (Q i j - ε)) = 1 - (n : ℝ) * ε := by
        intro i
        rw [Finset.sum_sub_distrib, ringStateProb_pow_row_sum n τ M i]
        simp
      have hl1 : l1Fin (Q.vecMul z) ≤ (1 - (n : ℝ) * ε) * l1Fin z := by
        unfold l1Fin
        simp_rw [hrewrite]
        calc ∑ j, |∑ i, z i * (Q i j - ε)|
            ≤ ∑ j, ∑ i, |z i| * (Q i j - ε) := by
              refine Finset.sum_le_sum fun j _ => ?_
              refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
              rw [abs_mul, abs_of_nonneg (hnonneg i j)]
          _ = ∑ i, |z i| * ∑ j, (Q i j - ε) := by
              rw [Finset.sum_comm]
              exact Finset.sum_congr rfl fun i _ => (Finset.mul_sum _ _ _).symm
          _ = (1 - (n : ℝ) * ε) * ∑ i, |z i| := by
              simp_rw [hrow_rem, ← Finset.sum_mul, mul_comm]
      exact hl1.trans (mul_le_mul_of_nonneg_right (le_max_right _ _)
        (Finset.sum_nonneg fun i _ => abs_nonneg _))
    have hQ_pow : ∀ q : ℕ, ∀ z : Fin n → ℝ, (∑ i, z i) = 0 →
        l1Fin ((Q ^ q).vecMul z) ≤ ρ₀ ^ q * l1Fin z := by
      intro q
      induction q with
      | zero => intro z _; simp
      | succ q ih =>
        intro z hz
        have hsum_step : ∑ j : Fin n, (Q ^ q).vecMul z j = 0 := by
          have hrowQ : ∀ i : Fin n, ∑ j : Fin n, (Q ^ q) i j = 1 := by
            intro i
            dsimp [Q]; rw [← pow_mul]
            exact ringStateProb_pow_row_sum n τ (M * q) i
          simp only [Matrix.vecMul, dotProduct]
          rw [Finset.sum_comm]
          simp [← Finset.mul_sum, hrowQ, hz]
        rw [pow_succ, ← Matrix.vecMul_vecMul, pow_succ']
        calc l1Fin (Q.vecMul ((Q ^ q).vecMul z))
            ≤ ρ₀ * l1Fin ((Q ^ q).vecMul z) := hQ_contract _ hsum_step
          _ ≤ ρ₀ * (ρ₀ ^ q * l1Fin z) := mul_le_mul_of_nonneg_left (ih z hz) hρ₀_pos.le
          _ = ρ₀ * ρ₀ ^ q * l1Fin z := by ring
    let β : ℝ := ρ₀ ^ (1 / (M : ℝ))
    have hMR_pos : (0 : ℝ) < M := by exact_mod_cast hM_pos
    have hβ_pos : 0 < β := Real.rpow_pos_of_pos hρ₀_pos _
    have hβ_lt1 : β < 1 := Real.rpow_lt_one hρ₀_pos.le hρ₀_lt1 (one_div_pos.mpr hMR_pos)
    have hβM : β ^ M = ρ₀ := by
      dsimp [β]
      rw [← Real.rpow_natCast, ← Real.rpow_mul hρ₀_pos.le, one_div_mul_cancel hMR_pos.ne',
        Real.rpow_one]
    let Ku : ℝ := 2 / ρ₀
    refine ⟨Ku, β, by positivity, hβ_pos.le, hβ_lt1, fun m i j => ?_⟩
    let q : ℕ := m / M
    let rem : ℕ := m % M
    have hm_decomp : m = M * q + rem := (Nat.div_add_mod m M).symm
    have hrem_lt : rem < M := Nat.mod_lt m (by omega)
    let δi : Fin n → ℝ := fun k => (if k = i then 1 else 0) - ringStateInitial n τ k
    have hδi_sum : ∑ k : Fin n, δi k = 0 := by
      dsimp [δi, ringStateInitial]
      rw [Finset.sum_sub_distrib]
      simp only [Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, ← Finset.sum_div]
      have hT := genTime_pos_of_ge_one n hn τ hτ
      change 1 - genTime n τ / genTime n τ = 0
      rw [div_self hT.ne', sub_self]
    have hδi_l1 : l1Fin δi ≤ 2 := by
      unfold l1Fin
      have hT := genTime_pos_of_ge_one n hn τ hτ
      have hinit_nn : ∀ k : Fin n, 0 ≤ ringStateInitial n τ k := fun k =>
        div_nonneg (lt_of_lt_of_le zero_lt_one (hτ k)).le hT.le
      calc ∑ k : Fin n, |δi k|
          ≤ ∑ k : Fin n, (|(if k = i then (1 : ℝ) else 0)| + |ringStateInitial n τ k|) :=
            Finset.sum_le_sum fun k _ => by
              simpa [δi, sub_eq_add_neg] using abs_sub (if k = i then (1 : ℝ) else 0) (ringStateInitial n τ k)
        _ = (∑ k : Fin n, if k = i then (1 : ℝ) else 0) + ∑ k : Fin n, ringStateInitial n τ k := by
            rw [Finset.sum_add_distrib]
            congr 1
            · refine Finset.sum_congr rfl fun k _ => by split_ifs <;> simp
            · refine Finset.sum_congr rfl fun k _ => abs_of_nonneg (hinit_nn k)
        _ = 2 := by
            simp only [Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, ringStateInitial,
              ← Finset.sum_div]
            change 1 + genTime n τ / genTime n τ = 2
            rw [div_self hT.ne']; norm_num
    have hstat_pow : ∀ s : ℕ, (ringStateProb n τ ^ s).vecMul (ringStateInitial n τ) =
        ringStateInitial n τ := by
      intro s
      induction s with
      | zero => simp
      | succ s ihs =>
        rw [pow_succ, ← Matrix.vecMul_vecMul, ihs, ringStateProb_stationary n hn τ hτ]
    have hentry_eq : (ringStateProb n τ ^ m) i j - ringStateInitial n τ j =
        (ringStateProb n τ ^ m).vecMul δi j := by
      have hvec : (ringStateProb n τ ^ m).vecMul (fun k => if k = i then (1 : ℝ) else 0) j =
          (ringStateProb n τ ^ m) i j := by
        simp [Matrix.vecMul, dotProduct]
      have hlin : (ringStateProb n τ ^ m).vecMul δi j =
          (ringStateProb n τ ^ m).vecMul (fun k => if k = i then (1 : ℝ) else 0) j -
            (ringStateProb n τ ^ m).vecMul (ringStateInitial n τ) j := by
        simp [δi, Matrix.vecMul, dotProduct, sub_mul, Finset.sum_sub_distrib]
      rw [hlin, hvec, hstat_pow m]
    have hstep_le : |(ringStateProb n τ ^ m) i j - ringStateInitial n τ j| ≤
        l1Fin ((ringStateProb n τ ^ m).vecMul δi) := by
      rw [hentry_eq]
      exact Finset.single_le_sum
        (f := fun k => |((ringStateProb n τ ^ m).vecMul δi) k|)
        (fun k _ => abs_nonneg _) (Finset.mem_univ j)
    have hl1_m : l1Fin ((ringStateProb n τ ^ m).vecMul δi) ≤ 2 * ρ₀ ^ q := by
      rw [hm_decomp, pow_add, pow_mul, ← Matrix.vecMul_vecMul]
      calc l1Fin ((ringStateProb n τ ^ rem).vecMul ((Q ^ q).vecMul δi))
          ≤ l1Fin ((Q ^ q).vecMul δi) := l1Fin_pow n τ hτ rem _
        _ ≤ ρ₀ ^ q * l1Fin δi := hQ_pow q δi hδi_sum
        _ ≤ ρ₀ ^ q * 2 := mul_le_mul_of_nonneg_left hδi_l1 (pow_nonneg hρ₀_pos.le q)
        _ = 2 * ρ₀ ^ q := by ring
    have hpow_le : ρ₀ ^ q ≤ (1 / ρ₀) * β ^ m := by
      have hrem_le : rem ≤ M := hrem_lt.le
      have hβ_rem : β ^ M ≤ β ^ rem := pow_le_pow_of_le_one hβ_pos.le hβ_lt1.le hrem_le
      have h1 : ρ₀ * ρ₀ ^ q ≤ ρ₀ ^ q * β ^ rem := by
        calc ρ₀ * ρ₀ ^ q = ρ₀ ^ q * β ^ M := by rw [hβM]; ring
          _ ≤ ρ₀ ^ q * β ^ rem := mul_le_mul_of_nonneg_left hβ_rem (pow_nonneg hρ₀_pos.le q)
      calc ρ₀ ^ q = (1 / ρ₀) * (ρ₀ * ρ₀ ^ q) := by field_simp [hρ₀_pos.ne']
        _ ≤ (1 / ρ₀) * (ρ₀ ^ q * β ^ rem) :=
            mul_le_mul_of_nonneg_left h1 (one_div_pos.mpr hρ₀_pos).le
        _ = (1 / ρ₀) * β ^ m := by rw [hm_decomp, pow_add, pow_mul, hβM]
    calc |(ringStateProb n τ ^ m) i j - ringStateInitial n τ j|
        ≤ l1Fin ((ringStateProb n τ ^ m).vecMul δi) := hstep_le
      _ ≤ 2 * ρ₀ ^ q := hl1_m
      _ ≤ 2 * ((1 / ρ₀) * β ^ m) := mul_le_mul_of_nonneg_left hpow_le (by norm_num)
      _ = Ku * β ^ m := by dsimp [Ku]; ring

/-- For any reference kernel `K` on `Fin n` projecting to `ringStateProb n τ` with stationary
law `ringStateInitial n τ`, `(m + 1)`-step edge transition probabilities factor through
`ringStateProb n τ ^ m`. -/
lemma ofKernel_pow_succ {E : Type*} [Fintype E] [DecidableEq E]
    (n : ℕ) (τ : Fin n → ℝ) (K : Dobson.ReferencePaths.Kernel (Fin n) E)
    (hproj : ∀ i j : Fin n,
      (∑ e : E, K.prob i e * (if K.dest e = j then 1 else 0)) = ringStateProb n τ i j)
    (m : ℕ) (e' e : E) :
    ((Chain.ofKernel K).prob ^ (m + 1)) e' e =
      ∑ j : Fin n, (ringStateProb n τ ^ m) (K.dest e') j * K.prob j e := by
  induction m generalizing e' e with
  | zero =>
    rw [pow_one, pow_zero]
    simp [Chain.ofKernel, Matrix.one_apply, ite_mul]
  | succ m ih =>
    rw [pow_succ', Matrix.mul_apply]
    simp_rw [ih]
    calc ∑ e₁ : E, (Chain.ofKernel K).prob e' e₁ *
            ∑ j : Fin n, (ringStateProb n τ ^ m) (K.dest e₁) j * K.prob j e
        = ∑ e₁ : E, ∑ x : Fin n, ∑ j : Fin n,
            K.prob (K.dest e') e₁ * (if K.dest e₁ = x then 1 else 0) *
              ((ringStateProb n τ ^ m) x j * K.prob j e) := by
          refine Finset.sum_congr rfl fun e₁ _ => ?_
          have hp_def : (Chain.ofKernel K).prob e' e₁ = K.prob (K.dest e') e₁ := rfl
          rw [hp_def, Finset.mul_sum, Finset.sum_comm]
          refine Finset.sum_congr rfl fun j _ => ?_
          have hfn : ∀ x : Fin n,
              K.prob (K.dest e') e₁ * (if K.dest e₁ = x then 1 else 0) *
                ((ringStateProb n τ ^ m) x j * K.prob j e) =
              if x = K.dest e₁ then
                K.prob (K.dest e') e₁ * ((ringStateProb n τ ^ m) (K.dest e₁) j * K.prob j e)
              else 0 := by
            intro x
            by_cases hx : x = K.dest e₁
            · subst hx; simp
            · have hne : K.dest e₁ ≠ x := Ne.symm hx
              simp [hx, hne]
          simp_rw [hfn]
          simp only [Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
      _ = ∑ x : Fin n, ∑ j : Fin n, ∑ e₁ : E,
            K.prob (K.dest e') e₁ * (if K.dest e₁ = x then 1 else 0) *
              ((ringStateProb n τ ^ m) x j * K.prob j e) := by
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun x _ => Finset.sum_comm
      _ = ∑ j : Fin n, (∑ x : Fin n,
            (∑ e₁ : E, K.prob (K.dest e') e₁ * (if K.dest e₁ = x then 1 else 0)) *
              (ringStateProb n τ ^ m) x j) * K.prob j e := by
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [Finset.sum_mul, Finset.sum_mul]
          refine Finset.sum_congr rfl fun e₁ _ => by ring
      _ = ∑ j : Fin n, (ringStateProb n τ ^ (m + 1)) (K.dest e') j * K.prob j e := by
          simp_rw [hproj, pow_succ', Matrix.mul_apply]

/-- Transfers state-level Doeblin mixing to `Chain.ofKernel K`. -/
lemma ofKernel_geometric_mixing {E : Type*} [Fintype E] [DecidableEq E]
    (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (hprim : n = 1 ∨ ∃ k_s : Fin n, 1 < τ k_s)
    (K : Dobson.ReferencePaths.Kernel (Fin n) E)
    (hinit : K.initial = ringStateInitial n τ)
    (hproj : ∀ i j : Fin n,
      (∑ e : E, K.prob i e * (if K.dest e = j then 1 else 0)) = ringStateProb n τ i j) :
    ∃ Ku β : ℝ, 0 ≤ Ku ∧ 0 ≤ β ∧ β < 1 ∧
      ∀ e e' : E, ∀ m : ℕ,
        |((Chain.ofKernel K).prob ^ (m + 1)) e' e - (Chain.ofKernel K).initial e| ≤ Ku * β ^ m := by
  obtain ⟨Ks, β, hKs, hβ0, hβ1, hmix_s⟩ := ringState_geometric_mixing n hn τ hτ hprim
  refine ⟨(n : ℝ) * Ks, β, by positivity, hβ0, hβ1, fun e e' m => ?_⟩
  have hprob_le : ∀ j : Fin n, K.prob j e ≤ 1 := by
    intro j
    rw [← K.total j]
    exact Finset.single_le_sum (fun x _ => K.nonneg j x) (Finset.mem_univ e)
  have hdiff : ((Chain.ofKernel K).prob ^ (m + 1)) e' e - (Chain.ofKernel K).initial e =
      ∑ j : Fin n, ((ringStateProb n τ ^ m) (K.dest e') j - ringStateInitial n τ j) * K.prob j e := by
    rw [ofKernel_pow_succ n τ K hproj m e' e]
    change (∑ j : Fin n, (ringStateProb n τ ^ m) (K.dest e') j * K.prob j e) -
        (∑ j : Fin n, K.initial j * K.prob j e) = _
    rw [hinit, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [hdiff]
  calc |∑ j : Fin n, ((ringStateProb n τ ^ m) (K.dest e') j - ringStateInitial n τ j) * K.prob j e|
      ≤ ∑ j : Fin n, |((ringStateProb n τ ^ m) (K.dest e') j - ringStateInitial n τ j) * K.prob j e| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _j : Fin n, Ks * β ^ m := by
        refine Finset.sum_le_sum fun j _ => ?_
        rw [abs_mul, abs_of_nonneg (K.nonneg j e)]
        exact (mul_le_of_le_one_right (abs_nonneg _) (hprob_le j)).trans (hmix_s m (K.dest e') j)
    _ = ((n : ℝ) * Ks) * β ^ m := by simp [mul_assoc]

end RingStateMixing

/-! ## Section 2: Exact discounted return resolvent on the geometric ring -/

section RingResolvent

lemma stayPGF_abs_lt_one (φ τ : ℝ) (hφ : |φ| < 1) (hτ : 1 ≤ τ) :
    0 < τ - (τ - 1) * φ ∧
      |stayPGF φ τ| < 1 ∧
      stayPGF φ τ * (1 - φ * (1 - 1 / τ)) = φ / τ := by
  obtain ⟨hφ_lo, hφ_hi⟩ := abs_lt.mp hφ
  have hτ_pos : 0 < τ := lt_of_lt_of_le zero_lt_one hτ
  have hden_pos : 0 < τ - (τ - 1) * φ := by
    have : τ - (τ - 1) * φ = 1 + (τ - 1) * (1 - φ) := by ring
    rw [this]
    nlinarith
  refine ⟨hden_pos, ?_, ?_⟩
  · unfold stayPGF
    rw [abs_div, abs_of_pos hden_pos, div_lt_one hden_pos]
    apply abs_lt.mpr
    constructor
    · have : -(τ - (τ - 1) * φ) - φ = -(1 + φ) - (τ - 1) * (1 - φ) := by ring
      nlinarith
    · have : (τ - (τ - 1) * φ) - φ = τ * (1 - φ) := by ring
      nlinarith
  · unfold stayPGF
    have hden_eq : 1 - φ * (1 - 1 / τ) = (τ - (τ - 1) * φ) / τ := by
      field_simp [hτ_pos.ne']
    rw [hden_eq, div_mul_div_cancel₀ hden_pos.ne']

lemma circuitR_abs_lt_one (n : ℕ) (hn : 1 ≤ n) (φ : ℝ) (hφ : |φ| < 1)
    (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k) :
    |circuitR n φ τ| < 1 ∧ 1 - circuitR n φ τ ≠ 0 := by
  have hprod_lt : ∀ s : Finset (Fin n), s.Nonempty →
      |∏ k ∈ s, stayPGF φ (τ k)| < 1 := by
    intro s hs
    classical
    induction s using Finset.induction_on with
    | empty => exact False.elim (Finset.not_nonempty_empty hs)
    | @insert a t ha iht =>
      rw [Finset.prod_insert ha, abs_mul]
      have ha_lt := (stayPGF_abs_lt_one φ (τ a) hφ (hτ a)).2.1
      by_cases ht : t.Nonempty
      · have ht_lt := iht ht
        calc |stayPGF φ (τ a)| * |∏ k ∈ t, stayPGF φ (τ k)|
            ≤ |stayPGF φ (τ a)| * 1 := mul_le_mul_of_nonneg_left ht_lt.le (abs_nonneg _)
          _ < 1 := by simpa using ha_lt
      · rw [Finset.not_nonempty_iff_eq_empty.mp ht]
        simpa using ha_lt
  have hlt := hprod_lt Finset.univ ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  change |circuitR n φ τ| < 1 at hlt
  refine ⟨hlt, ?_⟩
  intro h0
  have h1 : circuitR n φ τ = 1 := by linarith
  rw [h1, abs_one] at hlt
  exact lt_irrefl 1 hlt

/-- Discounted visit resolvent `G(x) = ∑_{m ≥ 0} φ^{m+1} (P^m)(x, k₀) (1 / τ_{k₀})`. -/
noncomputable def ringResolvent (n : ℕ) (τ : Fin n → ℝ) (k₀ : Fin n) (φ : ℝ) (x : Fin n) : ℝ :=
  ∑' m : ℕ, φ ^ (m + 1) * ((ringStateProb n τ ^ m) x k₀ * (1 / τ k₀))

lemma ringResolvent_summable (n : ℕ) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (k₀ : Fin n) (φ : ℝ) (hφ : |φ| < 1) (x : Fin n) :
    Summable (fun m : ℕ => φ ^ (m + 1) * ((ringStateProb n τ ^ m) x k₀ * (1 / τ k₀))) := by
  apply Summable.of_norm_bounded ((summable_geometric_of_lt_one (abs_nonneg φ) hφ).mul_left |φ|)
  intro m
  have hpos : 0 < τ k₀ := lt_of_lt_of_le zero_lt_one (hτ k₀)
  have htr_le : 1 / τ k₀ ≤ 1 := (div_le_one hpos).mpr (hτ k₀)
  have hp_le : (ringStateProb n τ ^ m) x k₀ ≤ 1 := by
    rw [← ringStateProb_pow_row_sum n τ m x]
    exact Finset.single_le_sum (fun j _ => ringStateProb_pow_nonneg n τ hτ m x j)
      (Finset.mem_univ k₀)
  have hprod_le : (ringStateProb n τ ^ m) x k₀ * (1 / τ k₀) ≤ 1 := by
    nlinarith [ringStateProb_pow_nonneg n τ hτ m x k₀, one_div_pos.mpr hpos]
  have hprod_nn : 0 ≤ (ringStateProb n τ ^ m) x k₀ * (1 / τ k₀) :=
    mul_nonneg (ringStateProb_pow_nonneg n τ hτ m x k₀) (one_div_pos.mpr hpos).le
  rw [Real.norm_eq_abs, abs_mul, abs_pow, pow_succ', abs_of_nonneg hprod_nn]
  exact mul_le_of_le_one_right (mul_nonneg (abs_nonneg φ) (pow_nonneg (abs_nonneg φ) m)) hprod_le

/-- **Exact geometric-ring resolvent theorem (`|φ| < 1`).**
Proved by maximum-norm Dirichlet uniqueness on `Fin n` without assuming a renewal equation. -/
theorem ring_resolvent_values (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (k₀ : Fin n) (φ : ℝ) (hφ : |φ| < 1) :
    ringResolvent n τ k₀ φ (finRotate n k₀) =
        circuitR n φ τ / (1 - circuitR n φ τ) ∧
      ringResolvent n τ k₀ φ k₀ =
        stayPGF φ (τ k₀) / (1 - circuitR n φ τ) := by
  have h1R_ne : 1 - circuitR n φ τ ≠ 0 := (circuitR_abs_lt_one n hn φ hφ τ hτ).2
  -- Step 1: One-step harmonic equation for `ringResolvent`.
  have hG_step : ∀ x : Fin n,
      ringResolvent n τ k₀ φ x =
        (if x = k₀ then φ * (1 / τ k₀) else 0) +
          φ * ∑ y : Fin n, ringStateProb n τ x y * ringResolvent n τ k₀ φ y := by
    intro x
    unfold ringResolvent
    rw [(ringResolvent_summable n τ hτ k₀ φ hφ x).tsum_eq_zero_add]
    congr 1
    · simp [Matrix.one_apply, mul_ite, ite_mul]
    simp_rw [pow_succ' (ringStateProb n τ), Matrix.mul_apply, Finset.sum_mul, Finset.mul_sum]
    rw [Summable.tsum_finsetSum]
    · refine Finset.sum_congr rfl fun y _ => ?_
      calc (∑' m : ℕ, φ ^ (m + 1 + 1) * (ringStateProb n τ x y * (ringStateProb n τ ^ m) y k₀ * (1 / τ k₀)))
          = ∑' m : ℕ, (φ * ringStateProb n τ x y) *
              (φ ^ (m + 1) * ((ringStateProb n τ ^ m) y k₀ * (1 / τ k₀))) :=
            tsum_congr fun m => by rw [pow_succ' φ]; ring
        _ = φ * (ringStateProb n τ x y *
              ∑' m : ℕ, φ ^ (m + 1) * ((ringStateProb n τ ^ m) y k₀ * (1 / τ k₀))) := by
            rw [tsum_mul_left]; ring
    · intro y _
      refine ((ringResolvent_summable n τ hτ k₀ φ hφ y).mul_left
        (φ * ringStateProb n τ x y)).congr fun m => ?_
      rw [pow_succ' φ]; ring
  -- Step 2: Explicit candidate vector `g_cand : Fin n → ℝ`.
  let g_cand : Fin n → ℝ := fun x =>
    (∏ j ∈ Finset.range (cycDist x k₀), stayPGF φ (τ ((finRotate n)^[j] x))) *
      (stayPGF φ (τ k₀) / (1 - circuitR n φ τ))
  have hcyc_self : cycDist k₀ k₀ = 0 := by
    dsimp [cycDist]
    have hk := k₀.isLt
    have : k₀.val + n - k₀.val = n := by omega
    rw [this, Nat.mod_self]
  have hcyc_rot_self : cycDist (finRotate n k₀) k₀ = n - 1 := by
    have hval := finRotate_iterate_val hn 1 k₀
    simp only [Function.iterate_one] at hval
    dsimp [cycDist]
    rw [hval]
    have hk := k₀.isLt
    by_cases hlast : k₀.val + 1 = n
    · rw [hlast, Nat.mod_self, Nat.sub_zero]
      have h1 : k₀.val + n = (n - 1) + n := by omega
      rw [h1, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    · have hlt : k₀.val + 1 < n := by omega
      rw [Nat.mod_eq_of_lt hlt]
      have h1 : k₀.val + n - (k₀.val + 1) = n - 1 := by omega
      rw [h1, Nat.mod_eq_of_lt (by omega)]
  have hcyc_step : ∀ x : Fin n, x ≠ k₀ →
      cycDist x k₀ = cycDist (finRotate n x) k₀ + 1 := by
    intro x hx
    have hne_val : x.val ≠ k₀.val := fun h => hx (Fin.ext h)
    have hval := finRotate_iterate_val hn 1 x
    simp only [Function.iterate_one] at hval
    dsimp [cycDist]
    rw [hval]
    have hxlt := x.isLt
    have hklt := k₀.isLt
    by_cases hlast : x.val + 1 = n
    · rw [hlast, Nat.mod_self, Nat.sub_zero, Nat.add_mod_right, Nat.mod_eq_of_lt hklt]
      have h1 : k₀.val + n - x.val = k₀.val + 1 := by omega
      rw [h1, Nat.mod_eq_of_lt (by omega)]
    · have hlt1 : x.val + 1 < n := by omega
      rw [Nat.mod_eq_of_lt hlt1]
      by_cases hxk : x.val < k₀.val
      · have h1 : k₀.val + n - x.val = (k₀.val - x.val) + n := by omega
        have h2 : k₀.val + n - (x.val + 1) = (k₀.val - x.val - 1) + n := by omega
        rw [h1, h2, Nat.add_mod_right, Nat.add_mod_right,
          Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
        omega
      · have hkx : k₀.val < x.val := by omega
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
        omega
  -- Orbit bijection `j : Fin n ↦ (finRotate n)^[j.val] k₀` gives `∏ j < n, stayPGF ... = circuitR n φ τ`.
  have horbit_prod : (∏ j ∈ Finset.range n, stayPGF φ (τ ((finRotate n)^[j] k₀))) =
      circuitR n φ τ := by
    let orbEquiv : Fin n ≃ Fin n :=
      { toFun := fun j => (finRotate n)^[j.val] k₀
        invFun := fun y => ⟨cycDist k₀ y, cycDist_lt hn k₀ y⟩
        left_inv := fun j => by
          apply Fin.ext
          dsimp [cycDist]
          rw [finRotate_iterate_val hn j.val k₀]
          have hj := j.isLt
          have hk := k₀.isLt
          by_cases hsum : k₀.val + j.val < n
          · rw [Nat.mod_eq_of_lt hsum,
              show k₀.val + j.val + n - k₀.val = j.val + n by omega,
              Nat.add_mod_right, Nat.mod_eq_of_lt hj]
          · have hlt2 : k₀.val + j.val - n < n := by omega
            rw [show k₀.val + j.val = (k₀.val + j.val - n) + n by omega,
              Nat.add_mod_right, Nat.mod_eq_of_lt hlt2,
              show k₀.val + j.val - n + n - k₀.val = j.val by omega,
              Nat.mod_eq_of_lt hj]
        right_inv := fun y => finRotate_iterate_cycDist hn k₀ y }
    rw [← Fin.prod_univ_eq_prod_range (fun j => stayPGF φ (τ ((finRotate n)^[j] k₀))) n]
    unfold circuitR
    exact Equiv.prod_comp orbEquiv (fun y => stayPGF φ (τ y))
  have hg_cand_self : g_cand k₀ = stayPGF φ (τ k₀) / (1 - circuitR n φ τ) := by
    dsimp [g_cand]; rw [hcyc_self]; simp
  have hg_cand_rot : g_cand (finRotate n k₀) = circuitR n φ τ / (1 - circuitR n φ τ) := by
    dsimp [g_cand]
    rw [hcyc_rot_self]
    have hn_succ : n = (n - 1) + 1 := by omega
    have hprod_split : (∏ j ∈ Finset.range n, stayPGF φ (τ ((finRotate n)^[j] k₀))) =
        stayPGF φ (τ k₀) *
          ∏ j ∈ Finset.range (n - 1), stayPGF φ (τ ((finRotate n)^[j] (finRotate n k₀))) := by
      have hp := Finset.prod_range_succ' (fun j => stayPGF φ (τ ((finRotate n)^[j] k₀))) (n - 1)
      rw [← hn_succ] at hp
      simpa only [Function.iterate_zero_apply, Function.iterate_succ_apply, mul_comm] using hp
    have h1 : (∏ j ∈ Finset.range (n - 1), stayPGF φ (τ ((finRotate n)^[j] (finRotate n k₀)))) *
        stayPGF φ (τ k₀) = circuitR n φ τ := by
      linarith [horbit_prod, hprod_split]
    calc (∏ j ∈ Finset.range (n - 1), stayPGF φ (τ ((finRotate n)^[j] (finRotate n k₀)))) *
          (stayPGF φ (τ k₀) / (1 - circuitR n φ τ))
        = ((∏ j ∈ Finset.range (n - 1), stayPGF φ (τ ((finRotate n)^[j] (finRotate n k₀)))) *
            stayPGF φ (τ k₀)) / (1 - circuitR n φ τ) := by ring
      _ = circuitR n φ τ / (1 - circuitR n φ τ) := by rw [h1]
  have hg_cand_step : ∀ x : Fin n,
      g_cand x =
        (if x = k₀ then φ * (1 / τ k₀) else 0) +
          φ * ∑ y : Fin n, ringStateProb n τ x y * g_cand y := by
    intro x
    have hmulVec := ringStateProb_mulVec n τ g_cand x
    rw [show (∑ y : Fin n, ringStateProb n τ x y * g_cand y) =
      (1 - 1 / τ x) * g_cand x + (1 / τ x) * g_cand (finRotate n x) from hmulVec]
    have hpgf_id : stayPGF φ (τ x) =
        φ * (1 / τ x) + φ * (1 - 1 / τ x) * stayPGF φ (τ x) := by
      have hid := (stayPGF_abs_lt_one φ (τ x) hφ (hτ x)).2.2
      have hdiv : φ / τ x = φ * (1 / τ x) := div_eq_mul_one_div φ (τ x)
      linarith [hid, hdiv]
    by_cases hx : x = k₀
    · rw [hx]
      simp only [↓reduceIte]
      rw [hg_cand_self, hg_cand_rot]
      have hpgf_k₀ : stayPGF φ (τ k₀) =
          φ * (1 / τ k₀) + φ * (1 - 1 / τ k₀) * stayPGF φ (τ k₀) := by
        have hid := (stayPGF_abs_lt_one φ (τ k₀) hφ (hτ k₀)).2.2
        have hdiv : φ / τ k₀ = φ * (1 / τ k₀) := div_eq_mul_one_div φ (τ k₀)
        linarith [hid, hdiv]
      have h1R_inv : 1 + circuitR n φ τ / (1 - circuitR n φ τ) =
          1 / (1 - circuitR n φ τ) := by
        field_simp [h1R_ne]; ring
      calc stayPGF φ (τ k₀) / (1 - circuitR n φ τ)
          = (φ * (1 / τ k₀) + φ * (1 - 1 / τ k₀) * stayPGF φ (τ k₀)) /
              (1 - circuitR n φ τ) := by
              conv_lhs => rw [hpgf_k₀]
        _ = φ * (1 / τ k₀) * (1 / (1 - circuitR n φ τ)) +
              φ * (1 - 1 / τ k₀) * (stayPGF φ (τ k₀) / (1 - circuitR n φ τ)) := by ring
        _ = φ * (1 / τ k₀) * (1 + circuitR n φ τ / (1 - circuitR n φ τ)) +
              φ * (1 - 1 / τ k₀) * (stayPGF φ (τ k₀) / (1 - circuitR n φ τ)) := by
              rw [← h1R_inv]
        _ = φ * (1 / τ k₀) +
              φ * ((1 - 1 / τ k₀) * (stayPGF φ (τ k₀) / (1 - circuitR n φ τ)) +
                (1 / τ k₀) * (circuitR n φ τ / (1 - circuitR n φ τ))) := by ring
    · simp only [hx, ↓reduceIte, zero_add]
      have hg_rec : g_cand x = stayPGF φ (τ x) * g_cand (finRotate n x) := by
        dsimp [g_cand]
        rw [hcyc_step x hx, Finset.prod_range_succ']
        simp only [Function.iterate_zero_apply, Function.iterate_succ_apply]
        ring
      calc g_cand x
          = (φ * (1 / τ x) + φ * (1 - 1 / τ x) * stayPGF φ (τ x)) * g_cand (finRotate n x) := by
              rw [hg_rec]; conv_lhs => rw [hpgf_id]
        _ = φ * ((1 - 1 / τ x) * (stayPGF φ (τ x) * g_cand (finRotate n x)) +
              (1 / τ x) * g_cand (finRotate n x)) := by ring
        _ = φ * ((1 - 1 / τ x) * g_cand x + (1 / τ x) * g_cand (finRotate n x)) := by
              rw [← hg_rec]
  -- Step 3: Maximum-norm Dirichlet uniqueness `ringResolvent = g_cand`.
  let d : Fin n → ℝ := fun x => ringResolvent n τ k₀ φ x - g_cand x
  have hd_eq : ∀ x : Fin n, d x = φ * ∑ y : Fin n, ringStateProb n τ x y * d y := by
    intro x
    dsimp [d]
    rw [hG_step x, hg_cand_step x]
    simp only [mul_sub, Finset.sum_sub_distrib]
    ring
  have hd_zero : ∀ x : Fin n, d x = 0 := by
    have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨k₀, Finset.mem_univ _⟩
    obtain ⟨m, _, hm⟩ := Finset.exists_max_image Finset.univ (fun x => |d x|) hne
    have hb (x : Fin n) : |d x| ≤ |d m| := hm x (Finset.mem_univ x)
    have hm0 : |d m| = 0 := by
      have hsum_le : |∑ y : Fin n, ringStateProb n τ m y * d y| ≤ |d m| := by
        calc |∑ y : Fin n, ringStateProb n τ m y * d y|
            ≤ ∑ y : Fin n, |ringStateProb n τ m y * d y| := Finset.abs_sum_le_sum_abs _ _
          _ ≤ ∑ y : Fin n, ringStateProb n τ m y * |d m| := Finset.sum_le_sum fun y _ => by
              rw [abs_mul, abs_of_nonneg (ringStateProb_nonneg n τ hτ m y)]
              exact mul_le_mul_of_nonneg_left (hb y) (ringStateProb_nonneg n τ hτ m y)
          _ = |d m| := by rw [← Finset.sum_mul, ringStateProb_row_sum, one_mul]
      have hh : |d m| ≤ |φ| * |d m| := by
        calc |d m| = |φ * ∑ y : Fin n, ringStateProb n τ m y * d y| := by rw [hd_eq m]
          _ = |φ| * |∑ y : Fin n, ringStateProb n τ m y * d y| := abs_mul _ _
          _ ≤ |φ| * |d m| := mul_le_mul_of_nonneg_left hsum_le (abs_nonneg φ)
      nlinarith [abs_nonneg (d m)]
    intro x
    exact abs_eq_zero.mp (le_antisymm ((hb x).trans_eq hm0) (abs_nonneg _))
  exact ⟨sub_eq_zero.mp (hd_zero (finRotate n k₀)) |>.trans hg_cand_rot,
    sub_eq_zero.mp (hd_zero k₀) |>.trans hg_cand_self⟩

end RingResolvent

/-! ## Section 3: Split-transfer reference-chain coefficient -/

section SplitTransferAlternatives

/-- Labeled edges when stage `k₀`'s transfer is split across alternative hosts `S : Finset ι`. -/
inductive AltEdge (n : ℕ) {ι : Type*} (S : Finset ι)
  | stay (i : Fin n)
  | unsplitTransfer (i : Fin n)
  | altTransfer (j : {x // x ∈ S})
  deriving DecidableEq

/-- Canonical equivalence `Fin n ⊕ Fin n ⊕ {x // x ∈ S} ≃ AltEdge n S`. -/
def altEdgeEquiv (n : ℕ) {ι : Type*} (S : Finset ι) :
    (Fin n ⊕ Fin n ⊕ {x // x ∈ S}) ≃ AltEdge n S where
  toFun
    | .inl i => .stay i
    | .inr (.inl i) => .unsplitTransfer i
    | .inr (.inr j) => .altTransfer j
  invFun
    | .stay i => .inl i
    | .unsplitTransfer i => .inr (.inl i)
    | .altTransfer j => .inr (.inr j)
  left_inv x := by rcases x with i | (i | j) <;> rfl
  right_inv e := by cases e <;> rfl

noncomputable instance instFintypeAltEdge (n : ℕ) {ι : Type*} [DecidableEq ι] (S : Finset ι) :
    Fintype (AltEdge n S) :=
  Fintype.ofEquiv _ (altEdgeEquiv n S)

lemma sum_altEdge (n : ℕ) {ι : Type*} [DecidableEq ι] (S : Finset ι) (f : AltEdge n S → ℝ) :
    (∑ e : AltEdge n S, f e) =
      (∑ i : Fin n, f (.stay i)) +
        (∑ i : Fin n, f (.unsplitTransfer i)) +
        (∑ j : {x // x ∈ S}, f (.altTransfer j)) := by
  rw [← Equiv.sum_comp (altEdgeEquiv n S) f]
  simp only [Fintype.sum_sum_type, altEdgeEquiv, Equiv.coe_fn_mk, add_assoc]

/-- Outgoing edge probabilities in the split-transfer ring. -/
noncomputable def splitProb (n : ℕ) (τ : Fin n → ℝ) (k₀ : Fin n)
    {ι : Type*} (S : Finset ι) (π : ι → ℝ) (i : Fin n) : AltEdge n S → ℝ
  | .stay x => if i = x then 1 - 1 / τ i else 0
  | .unsplitTransfer x => if i = x ∧ i ≠ k₀ then 1 / τ i else 0
  | .altTransfer j => if i = k₀ then π j.1 * (1 / τ k₀) else 0

/-- Edge destination state in the split-transfer ring. -/
def splitDest (n : ℕ) (k₀ : Fin n) {ι : Type*} (S : Finset ι) : AltEdge n S → Fin n
  | .stay x => x
  | .unsplitTransfer x => finRotate n x
  | .altTransfer _ => finRotate n k₀

lemma splitProb_nonneg (n : ℕ) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k) (k₀ : Fin n)
    {ι : Type*} (S : Finset ι) (π : ι → ℝ) (hπ0 : ∀ j ∈ S, 0 ≤ π j)
    (i : Fin n) (e : AltEdge n S) :
    0 ≤ splitProb n τ k₀ S π i e := by
  have hpos_i : 0 < τ i := lt_of_lt_of_le zero_lt_one (hτ i)
  have hpos_k : 0 < τ k₀ := lt_of_lt_of_le zero_lt_one (hτ k₀)
  cases e with
  | stay x =>
    dsimp [splitProb]
    split_ifs
    · rw [sub_nonneg, div_le_one hpos_i]; exact hτ i
    · exact le_rfl
  | unsplitTransfer x =>
    dsimp [splitProb]
    split_ifs <;> positivity
  | altTransfer j =>
    dsimp [splitProb]
    split_ifs
    · exact mul_nonneg (hπ0 j.1 j.2) (by positivity)
    · exact le_rfl

lemma splitProb_proj (n : ℕ) (τ : Fin n → ℝ) (k₀ : Fin n)
    {ι : Type*} [DecidableEq ι] (S : Finset ι) (π : ι → ℝ)
    (hπsum : ∑ j ∈ S, π j = 1) (i y : Fin n) :
    (∑ e : AltEdge n S, splitProb n τ k₀ S π i e * (if splitDest n k₀ S e = y then 1 else 0)) =
      ringStateProb n τ i y := by
  rw [sum_altEdge]
  unfold ringStateProb
  have hs1 : (∑ x : Fin n, splitProb n τ k₀ S π i (.stay x) *
      (if splitDest n k₀ S (.stay x) = y then 1 else 0)) =
      (1 - 1 / τ i) * (if i = y then 1 else 0) := by
    have hfn : ∀ x : Fin n,
        splitProb n τ k₀ S π i (.stay x) * (if splitDest n k₀ S (.stay x) = y then 1 else 0) =
          if x = i then (1 - 1 / τ i) * (if i = y then 1 else 0) else 0 := by
      intro x
      change (if i = x then 1 - 1 / τ i else 0) * (if x = y then 1 else 0) = _
      by_cases hx : x = i
      · subst hx; simp
      · have hne : i ≠ x := Ne.symm hx
        simp [hx, hne]
    simp_rw [hfn]; simp
  rw [hs1]
  by_cases hik : i = k₀
  · subst hik
    have hs2 : (∑ x : Fin n, splitProb n τ i S π i (.unsplitTransfer x) *
        (if splitDest n i S (.unsplitTransfer x) = y then 1 else 0)) = 0 := by
      refine Finset.sum_eq_zero fun x _ => ?_
      change (if i = x ∧ i ≠ i then 1 / τ i else 0) * (if finRotate n x = y then 1 else 0) = 0
      simp
    have hs3 : (∑ j : {x // x ∈ S}, splitProb n τ i S π i (.altTransfer j) *
        (if splitDest n i S (.altTransfer j) = y then 1 else 0)) =
        (1 / τ i) * (if finRotate n i = y then 1 else 0) := by
      have hfn : ∀ j : {x // x ∈ S},
          splitProb n τ i S π i (.altTransfer j) *
            (if splitDest n i S (.altTransfer j) = y then 1 else 0) =
          π j.1 * ((1 / τ i) * (if finRotate n i = y then 1 else 0)) := by
        intro j
        change (if i = i then π j.1 * (1 / τ i) else 0) * (if finRotate n i = y then 1 else 0) = _
        simp only [↓reduceIte, mul_assoc]
      simp_rw [hfn, ← Finset.sum_mul, Finset.univ_eq_attach, Finset.sum_attach S π, hπsum, one_mul]
    rw [hs2, hs3, add_zero]
  · have hs3 : (∑ j : {x // x ∈ S}, splitProb n τ k₀ S π i (.altTransfer j) *
        (if splitDest n k₀ S (.altTransfer j) = y then 1 else 0)) = 0 := by
      refine Finset.sum_eq_zero fun j _ => ?_
      change (if i = k₀ then π j.1 * (1 / τ k₀) else 0) * (if finRotate n k₀ = y then 1 else 0) = 0
      simp [hik]
    have hs2 : (∑ x : Fin n, splitProb n τ k₀ S π i (.unsplitTransfer x) *
        (if splitDest n k₀ S (.unsplitTransfer x) = y then 1 else 0)) =
        (1 / τ i) * (if finRotate n i = y then 1 else 0) := by
      have hfn : ∀ x : Fin n,
          splitProb n τ k₀ S π i (.unsplitTransfer x) *
            (if splitDest n k₀ S (.unsplitTransfer x) = y then 1 else 0) =
          if x = i then (1 / τ i) * (if finRotate n i = y then 1 else 0) else 0 := by
        intro x
        change (if i = x ∧ i ≠ k₀ then 1 / τ i else 0) * (if finRotate n x = y then 1 else 0) = _
        by_cases hx : x = i
        · subst hx; simp [hik]
        · have hne : i ≠ x := Ne.symm hx
          simp [hx, hne]
      simp_rw [hfn]; simp
    rw [hs2, hs3, add_zero]

lemma splitProb_total (n : ℕ) (τ : Fin n → ℝ) (k₀ : Fin n)
    {ι : Type*} [DecidableEq ι] (S : Finset ι) (π : ι → ℝ)
    (hπsum : ∑ j ∈ S, π j = 1) (i : Fin n) :
    ∑ e : AltEdge n S, splitProb n τ k₀ S π i e = 1 := by
  have h1 : (∑ e : AltEdge n S, splitProb n τ k₀ S π i e) =
      ∑ y : Fin n, ∑ e : AltEdge n S,
        splitProb n τ k₀ S π i e * (if splitDest n k₀ S e = y then 1 else 0) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun e _ => ?_
    rw [← Finset.mul_sum]
    simp
  rw [h1]
  simp_rw [splitProb_proj n τ k₀ S π hπsum i]
  exact ringStateProb_row_sum n τ i

/-- Reference Markov kernel of the split-transfer geometric ring. -/
noncomputable def splitRingKernel (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (k₀ : Fin n) {ι : Type*} [DecidableEq ι] (S : Finset ι) (π : ι → ℝ)
    (hπ0 : ∀ j ∈ S, 0 ≤ π j) (hπsum : ∑ j ∈ S, π j = 1) :
    Dobson.ReferencePaths.Kernel (Fin n) (AltEdge n S) where
  prob := splitProb n τ k₀ S π
  dest := splitDest n k₀ S
  nonneg := splitProb_nonneg n τ hτ k₀ S π hπ0
  total := splitProb_total n τ k₀ S π hπsum
  initial := ringStateInitial n τ
  initial_pos := fun i =>
    div_pos (lt_of_lt_of_le zero_lt_one (hτ i)) (genTime_pos_of_ge_one n hn τ hτ)
  initial_nonneg := fun i =>
    (div_pos (lt_of_lt_of_le zero_lt_one (hτ i)) (genTime_pos_of_ge_one n hn τ hτ)).le
  initial_total := by
    unfold ringStateInitial
    rw [← Finset.sum_div]
    exact div_self (genTime_pos_of_ge_one n hn τ hτ).ne'
  stationary := fun f => by
    have hstep (i : Fin n) : (∑ e : AltEdge n S, splitProb n τ k₀ S π i e * f (splitDest n k₀ S e)) =
        ∑ y : Fin n, ringStateProb n τ i y * f y := by
      simp_rw [← splitProb_proj n τ k₀ S π hπsum i, Finset.sum_mul]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun e _ => ?_
      have hfn : ∀ y : Fin n,
          splitProb n τ k₀ S π i e * (if splitDest n k₀ S e = y then 1 else 0) * f y =
            if y = splitDest n k₀ S e then splitProb n τ k₀ S π i e * f y else 0 := by
        intro y
        by_cases hy : y = splitDest n k₀ S e
        · subst hy; simp
        · have hne : splitDest n k₀ S e ≠ y := Ne.symm hy
          simp [hy, hne]
      simp_rw [hfn]; simp
    simp_rw [hstep, Finset.mul_sum]
    rw [Finset.sum_comm]
    have hstat := ringStateProb_stationary n hn τ hτ
    refine Finset.sum_congr rfl fun y _ => ?_
    have hy := congrFun hstat y
    simp only [Matrix.vecMul, dotProduct] at hy
    calc (∑ x : Fin n, ringStateInitial n τ x * (ringStateProb n τ x y * f y))
        = (∑ x : Fin n, ringStateInitial n τ x * ringStateProb n τ x y) * f y := by
            rw [Finset.sum_mul]; refine Finset.sum_congr rfl fun x _ => by ring
      _ = ringStateInitial n τ y * f y := by rw [hy]

/-- Stationary edge Markov chain of the split-transfer geometric ring. -/
noncomputable def splitRingChain (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (k₀ : Fin n) {ι : Type*} [DecidableEq ι] (S : Finset ι) (π : ι → ℝ)
    (hπ0 : ∀ j ∈ S, 0 ≤ π j) (hπsum : ∑ j ∈ S, π j = 1) :
    Chain (AltEdge n S) :=
  Chain.ofKernel (splitRingKernel n hn τ hτ k₀ S π hπ0 hπsum)

lemma splitRingChain_initial_altTransfer (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ)
    (hτ : ∀ k, 1 ≤ τ k) (k₀ : Fin n) {ι : Type*} [DecidableEq ι] (S : Finset ι) (π : ι → ℝ)
    (hπ0 : ∀ j ∈ S, 0 ≤ π j) (hπsum : ∑ j ∈ S, π j = 1) (j : {x // x ∈ S}) :
    (splitRingChain n hn τ hτ k₀ S π hπ0 hπsum).initial (.altTransfer j) =
      π j.1 / genTime n τ := by
  change (∑ i : Fin n, ringStateInitial n τ i * splitProb n τ k₀ S π i (.altTransfer j)) = _
  dsimp [splitProb, ringStateInitial]
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
  have hne : τ k₀ ≠ 0 := (lt_of_lt_of_le zero_lt_one (hτ k₀)).ne'
  have hT_ne : genTime n τ ≠ 0 := (genTime_pos_of_ge_one n hn τ hτ).ne'
  field_simp [hne, hT_ne]

lemma splitRingChain_pow_altTransfer (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ)
    (hτ : ∀ k, 1 ≤ τ k) (k₀ : Fin n) {ι : Type*} [DecidableEq ι] (S : Finset ι) (π : ι → ℝ)
    (hπ0 : ∀ j ∈ S, 0 ≤ π j) (hπsum : ∑ j ∈ S, π j = 1)
    (m : ℕ) (j' j : {x // x ∈ S}) :
    ((splitRingChain n hn τ hτ k₀ S π hπ0 hπsum).prob ^ (m + 1)) (.altTransfer j') (.altTransfer j) =
      π j.1 * ((ringStateProb n τ ^ m) (finRotate n k₀) k₀ * (1 / τ k₀)) := by
  unfold splitRingChain
  rw [ofKernel_pow_succ n τ (splitRingKernel n hn τ hτ k₀ S π hπ0 hπsum)
    (splitProb_proj n τ k₀ S π hπsum) m (.altTransfer j') (.altTransfer j)]
  change (∑ x : Fin n, (ringStateProb n τ ^ m) (finRotate n k₀) x *
    (if x = k₀ then π j.1 * (1 / τ k₀) else 0)) = _
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
  ring

/-- Edge score assigning host fluctuation `ξ j` to alternative branch `.altTransfer ⟨j, hj⟩`. -/
def altScore {n : ℕ} {ι Ω : Type*} (S : Finset ι) (ξ : ι → Ω → ℝ) :
    AltEdge n S → Ω → ℝ
  | .altTransfer j => ξ j.1
  | _ => fun _ => 0

lemma altScore_memLp {n : ℕ} {ι Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (S : Finset ι) (ξ : ι → Ω → ℝ) (hξ : ∀ j ∈ S, MemLp (ξ j) 2 μ) (e : AltEdge n S) :
    MemLp (altScore S ξ e) 2 μ := by
  cases e with
  | altTransfer j => exact hξ j.1 j.2
  | _ => exact MemLp.zero

/-- Actual finite-horizon lognormal path-growth second-derivative cost limit of the split-transfer
geometric ring: `-(1/2) * lim_{N → ∞} (d²/dσ²) finiteGrowth(N, σ)|_{σ = 0}`. -/
noncomputable def altLocalCost {ι Ω : Type*} [DecidableEq ι] [MeasurableSpace Ω]
    (μ : Measure Ω) (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (k₀ : Fin n) (S : Finset ι) (π : ι → ℝ) (hπ0 : ∀ j ∈ S, 0 ≤ π j)
    (hπsum : ∑ j ∈ S, π j = 1) (T_env : Ω → Ω) (ξ : ι → Ω → ℝ) : ℝ :=
  -(1 / 2 : ℝ) * limUnder atTop (fun N : ℕ =>
    iteratedDeriv 2
      (Dobson.FinitePathJet.finiteGrowth μ
        (splitRingChain n hn τ hτ k₀ S π hπ0 hπsum) T_env (altScore S ξ) 0 .lognormal N) 0)

private lemma stationary_autocov_abs_le_sum {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T_env : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T_env μ μ) (f g : Ω → ℝ)
    (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) (m : ℕ) :
    |∫ ω, f ((T_env : Ω → Ω)^[m] ω) * g ω ∂μ| ≤
      (1 / 2 : ℝ) * ((∫ ω, f ω * f ω ∂μ) + ∫ ω, g ω * g ω ∂μ) := by
  have hfm : MemLp (fun ω => f ((T_env : Ω → Ω)^[m] ω)) 2 μ :=
    hf.comp_measurePreserving (hT.iterate m)
  have hsq_f : Integrable (fun ω => f ω * f ω) μ := hf.integrable_mul hf
  have hsq_fm : Integrable (fun ω => f ((T_env : Ω → Ω)^[m] ω) * f ((T_env : Ω → Ω)^[m] ω)) μ :=
    hfm.integrable_mul hfm
  have hsq_g : Integrable (fun ω => g ω * g ω) μ := hg.integrable_mul hg
  have hprod : Integrable (fun ω => f ((T_env : Ω → Ω)^[m] ω) * g ω) μ := hfm.integrable_mul hg
  have hshift_f : (∫ ω, f ((T_env : Ω → Ω)^[m] ω) * f ((T_env : Ω → Ω)^[m] ω) ∂μ) =
      ∫ ω, f ω * f ω ∂μ := by
    have hrec : ∀ k : ℕ, (∫ ω, f ((T_env : Ω → Ω)^[k] ω) * f ((T_env : Ω → Ω)^[k] ω) ∂μ) =
        ∫ ω, f ω * f ω ∂μ := by
      intro k
      induction k with
      | zero => simp
      | succ k ihk =>
        simpa only [Function.iterate_succ_apply] using
          (hT.integral_comp' (fun ω => f ((T_env : Ω → Ω)^[k] ω) * f ((T_env : Ω → Ω)^[k] ω))).trans ihk
    exact hrec m
  have hmaj : Integrable (fun ω => (1 / 2 : ℝ) *
      (f ((T_env : Ω → Ω)^[m] ω) * f ((T_env : Ω → Ω)^[m] ω) + g ω * g ω)) μ :=
    (hsq_fm.add hsq_g).const_mul (1 / 2)
  refine abs_integral_le_integral_abs.trans ?_
  calc (∫ ω, |f ((T_env : Ω → Ω)^[m] ω) * g ω| ∂μ)
      ≤ ∫ ω, (1 / 2 : ℝ) *
          (f ((T_env : Ω → Ω)^[m] ω) * f ((T_env : Ω → Ω)^[m] ω) + g ω * g ω) ∂μ := by
        refine integral_mono hprod.norm hmaj fun ω => ?_
        change |f ((T_env : Ω → Ω)^[m] ω) * g ω| ≤ _
        rw [abs_mul]
        have h1 := sq_nonneg (|f ((T_env : Ω → Ω)^[m] ω)| - |g ω|)
        have h2 : |f ((T_env : Ω → Ω)^[m] ω)| ^ 2 =
            f ((T_env : Ω → Ω)^[m] ω) * f ((T_env : Ω → Ω)^[m] ω) := by rw [sq_abs, sq]
        have h3 : |g ω| ^ 2 = g ω * g ω := by rw [sq_abs, sq]
        linarith
    _ = (1 / 2 : ℝ) * ((∫ ω, f ((T_env : Ω → Ω)^[m] ω) * f ((T_env : Ω → Ω)^[m] ω) ∂μ) +
          ∫ ω, g ω * g ω ∂μ) := by
        rw [integral_const_mul, integral_add hsq_fm hsq_g]
    _ = (1 / 2 : ℝ) * ((∫ ω, f ω * f ω ∂μ) + ∫ ω, g ω * g ω ∂μ) := by rw [hshift_f]

/-- **Split-transfer finite-path coefficient:**
In a primitive `n`-stage geometric ring (`n = 1 ∨ ∃ k_s, 1 < τ k_s`) with stage `k₀`'s transfer
split across `S : Finset ι` with shares `π` (`0 ≤ π j`, `∑ j ∈ S, π j = 1`) and uncorrelated
stationary AR(1) log-host fluctuations `ξ j` (`|φ j| < 1`), the actual finite-path second-derivative
cost limit `altLocalCost` equals
`∑ j ∈ S, π j ^ 2 * v j * ((1 + 2 * θ j) / (2 * T ^ 2) - 1 / T * (R j / (1 - R j)))`. -/
lemma altLocalCost_eq {ι Ω : Type*} [DecidableEq ι] [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T_env : Ω ≃ᵐ Ω) (hT : MeasurePreserving T_env μ μ)
    (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (hprim : n = 1 ∨ ∃ k_s : Fin n, 1 < τ k_s)
    (k₀ : Fin n) (S : Finset ι) (π φ : ι → ℝ)
    (hπ0 : ∀ j ∈ S, 0 ≤ π j) (hπsum : ∑ j ∈ S, π j = 1)
    (hφ : ∀ j ∈ S, |φ j| < 1)
    (ξ : ι → Ω → ℝ) (hξ : ∀ j ∈ S, MemLp (ξ j) 2 μ)
    (hcov_self : ∀ j ∈ S, ∀ m : ℕ,
      (∫ ω, ξ j ((T_env : Ω → Ω)^[m] ω) * ξ j ω ∂μ) =
        (∫ ω, ξ j ω * ξ j ω ∂μ) * (φ j) ^ m)
    (hcov_cross : ∀ i ∈ S, ∀ j ∈ S, i ≠ j → ∀ m : ℕ,
      (∫ ω, ξ i ((T_env : Ω → Ω)^[m] ω) * ξ j ω ∂μ) = 0) :
    let T := genTime n τ
    let v : ι → ℝ := fun j => ∫ ω, ξ j ω * ξ j ω ∂μ
    let θ : ι → ℝ := fun j => thetaOfPhi (φ j)
    let R : ι → ℝ := fun j => circuitR n (φ j) τ
    altLocalCost μ n hn τ hτ k₀ S π hπ0 hπsum T_env ξ =
      ∑ j ∈ S, π j ^ 2 * v j * ((1 + 2 * θ j) / (2 * T ^ 2) - 1 / T * (R j / (1 - R j))) := by
  intro T v θ R
  let K := splitRingChain n hn τ hτ k₀ S π hπ0 hπsum
  have hT_pos : 0 < T := genTime_pos_of_ge_one n hn τ hτ
  -- Step 1: Identify `altLocalCost` with `FinitePathCoefficient.covarianceCoefficient`.
  obtain ⟨Ku, β, hKu, hβ0, hβ1, hmix⟩ :=
    ofKernel_geometric_mixing n hn τ hτ hprim
      (splitRingKernel n hn τ hτ k₀ S π hπ0 hπsum) rfl (splitProb_proj n τ k₀ S π hπsum)
  let B : ℝ := ∑ j ∈ S, v j
  have hv_nn : ∀ j, 0 ≤ v j := fun j => integral_nonneg fun ω => mul_self_nonneg (ξ j ω)
  have hB_nn : 0 ≤ B := Finset.sum_nonneg fun j _ => hv_nn j
  have hcov_bdd : ∀ e e' : AltEdge n S, ∀ m : ℕ,
      |lagCovariance μ T_env (altScore S ξ) e e' m| ≤ B := by
    intro e e' m
    cases e with
    | stay x =>
      have h0 : lagCovariance μ T_env (altScore S ξ) (.stay x) e' m = 0 := by
        simp [lagCovariance, altScore]
      rw [h0, abs_zero]; exact hB_nn
    | unsplitTransfer x =>
      have h0 : lagCovariance μ T_env (altScore S ξ) (.unsplitTransfer x) e' m = 0 := by
        simp [lagCovariance, altScore]
      rw [h0, abs_zero]; exact hB_nn
    | altTransfer j =>
      cases e' with
      | stay x =>
        have h0 : lagCovariance μ T_env (altScore S ξ) (.altTransfer j) (.stay x) m = 0 := by
          simp [lagCovariance, altScore]
        rw [h0, abs_zero]; exact hB_nn
      | unsplitTransfer x =>
        have h0 : lagCovariance μ T_env (altScore S ξ) (.altTransfer j) (.unsplitTransfer x) m = 0 := by
          simp [lagCovariance, altScore]
        rw [h0, abs_zero]; exact hB_nn
      | altTransfer j' =>
        change |∫ ω, ξ j.1 ((T_env : Ω → Ω)^[m] ω) * ξ j'.1 ω ∂μ| ≤ B
        have hcs := stationary_autocov_abs_le_sum μ T_env hT (ξ j.1) (ξ j'.1)
          (hξ j.1 j.2) (hξ j'.1 j'.2) m
        have hj_le : v j.1 ≤ B := Finset.single_le_sum (fun x _ => hv_nn x) j.2
        have hj'_le : v j'.1 ≤ B := Finset.single_le_sum (fun x _ => hv_nn x) j'.2
        dsimp [v] at hj_le hj'_le
        linarith
  have htend := Dobson.FinitePathCoefficient.finiteCoefficient_tendsto μ K T_env hT
    (altScore S ξ) (altScore_memLp μ S ξ hξ) B Ku β hB_nn hKu hβ0 hβ1 hcov_bdd hmix
  have hsec : ∀ N : ℕ,
      iteratedDeriv 2 (Dobson.FinitePathJet.finiteGrowth μ K T_env (altScore S ξ) 0 .lognormal N) 0 =
        -2 * Dobson.FinitePathCoefficient.finiteCoefficient μ K T_env (altScore S ξ) N :=
    fun N => Dobson.FinitePathJet.finiteGrowth_second μ K T_env hT
      (altScore S ξ) (altScore_memLp μ S ξ hξ) 0 .lognormal (by simp) N
  have hlim_deriv : Tendsto
      (fun N : ℕ => iteratedDeriv 2
        (Dobson.FinitePathJet.finiteGrowth μ K T_env (altScore S ξ) 0 .lognormal N) 0)
      atTop (𝓝 (-2 * Dobson.FinitePathCoefficient.covarianceCoefficient μ K T_env (altScore S ξ))) :=
    (htend.const_mul (-2 : ℝ)).congr fun N => (hsec N).symm
  have hC_eq : altLocalCost μ n hn τ hτ k₀ S π hπ0 hπsum T_env ξ =
      Dobson.FinitePathCoefficient.covarianceCoefficient μ K T_env (altScore S ξ) := by
    unfold altLocalCost
    change -(1 / 2 : ℝ) * limUnder atTop (fun N : ℕ =>
      iteratedDeriv 2 (Dobson.FinitePathJet.finiteGrowth μ K T_env (altScore S ξ) 0 .lognormal N) 0) = _
    rw [hlim_deriv.limUnder_eq]
    ring
  rw [hC_eq]
  -- Step 2: Collapse `covarianceCoefficient` over `AltEdge n S` to diagonal sum over `j ∈ S`.
  have hlag_alt : ∀ (j j' : {x // x ∈ S}) (m : ℕ),
      lagCovariance μ T_env (altScore (n := n) S ξ) (.altTransfer j) (.altTransfer j') m =
        if j = j' then v j.1 * (φ j.1) ^ m else 0 := by
    intro j j' m
    by_cases hjj : j = j'
    · subst hjj
      simp only [↓reduceIte, lagCovariance, altScore]
      exact hcov_self j.1 j.2 m
    · have hne_val : j.1 ≠ j'.1 := fun h => hjj (Subtype.ext h)
      simp only [hjj, ↓reduceIte, lagCovariance, altScore]
      exact hcov_cross j.1 j.2 j'.1 j'.2 hne_val m
  have hpair_zero_left : ∀ e' : AltEdge n S, ∀ m : ℕ,
      (∀ i, lagCovariance μ T_env (altScore S ξ) (.stay i) e' m = 0) ∧
      (∀ i, lagCovariance μ T_env (altScore S ξ) (.unsplitTransfer i) e' m = 0) := by
    intro e' m
    exact ⟨fun _ => by simp [lagCovariance, altScore],
      fun _ => by simp [lagCovariance, altScore]⟩
  have hpair_zero_right : ∀ e : AltEdge n S, ∀ m : ℕ,
      (∀ i, lagCovariance μ T_env (altScore S ξ) e (.stay i) m = 0) ∧
      (∀ i, lagCovariance μ T_env (altScore S ξ) e (.unsplitTransfer i) m = 0) := by
    intro e m
    exact ⟨fun _ => by simp [lagCovariance, altScore],
      fun _ => by simp [lagCovariance, altScore]⟩
  have hdiag0 : (∑ e : AltEdge n S, ∑ e' : AltEdge n S,
      K.initial e * K.initial e' * lagCovariance μ T_env (altScore S ξ) e e' 0) =
      ∑ j ∈ S, (π j / T) ^ 2 * v j := by
    rw [sum_altEdge]
    have hs1 : (∑ i : Fin n, ∑ e' : AltEdge n S,
        K.initial (.stay i) * K.initial e' * lagCovariance μ T_env (altScore S ξ) (.stay i) e' 0) = 0 :=
      Finset.sum_eq_zero fun i _ => Finset.sum_eq_zero fun e' _ => by
        rw [(hpair_zero_left e' 0).1 i, mul_zero]
    have hs2 : (∑ i : Fin n, ∑ e' : AltEdge n S,
        K.initial (.unsplitTransfer i) * K.initial e' *
          lagCovariance μ T_env (altScore S ξ) (.unsplitTransfer i) e' 0) = 0 :=
      Finset.sum_eq_zero fun i _ => Finset.sum_eq_zero fun e' _ => by
        rw [(hpair_zero_left e' 0).2 i, mul_zero]
    rw [hs1, hs2, zero_add, zero_add]
    have hinner : ∀ j : {x // x ∈ S},
        (∑ e' : AltEdge n S, K.initial (.altTransfer j) * K.initial e' *
          lagCovariance μ T_env (altScore S ξ) (.altTransfer j) e' 0) =
        (π j.1 / T) ^ 2 * v j.1 := by
      intro j
      rw [sum_altEdge]
      have hr1 : (∑ i : Fin n, K.initial (.altTransfer j) * K.initial (.stay i) *
          lagCovariance μ T_env (altScore S ξ) (.altTransfer j) (.stay i) 0) = 0 :=
        Finset.sum_eq_zero fun i _ => by rw [(hpair_zero_right _ 0).1 i, mul_zero]
      have hr2 : (∑ i : Fin n, K.initial (.altTransfer j) * K.initial (.unsplitTransfer i) *
          lagCovariance μ T_env (altScore S ξ) (.altTransfer j) (.unsplitTransfer i) 0) = 0 :=
        Finset.sum_eq_zero fun i _ => by rw [(hpair_zero_right _ 0).2 i, mul_zero]
      rw [hr1, hr2, zero_add, zero_add]
      simp_rw [fun j' => hlag_alt j j' 0]
      simp only [mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]
      dsimp [K, T]
      rw [splitRingChain_initial_altTransfer, pow_zero, mul_one, sq]
    simp_rw [hinner]
    rw [Finset.univ_eq_attach]
    exact Finset.sum_attach S (fun x => (π x / T) ^ 2 * v x)
  have hlag_step : ∀ m : ℕ,
      (∑ e : AltEdge n S, ∑ e' : AltEdge n S,
        lagCovariance μ T_env (altScore S ξ) e e' (m + 1) * K.initial e' *
          ((K.prob ^ (m + 1)) e' e - K.initial e)) =
      ∑ j ∈ S, (π j ^ 2 * v j / T) * ((φ j) ^ (m + 1) *
        ((ringStateProb n τ ^ m) (finRotate n k₀) k₀ * (1 / τ k₀) - 1 / T)) := by
    intro m
    rw [sum_altEdge]
    have hs1 : (∑ i : Fin n, ∑ e' : AltEdge n S,
        lagCovariance μ T_env (altScore S ξ) (.stay i) e' (m + 1) * K.initial e' *
          ((K.prob ^ (m + 1)) e' (.stay i) - K.initial (.stay i))) = 0 :=
      Finset.sum_eq_zero fun i _ => Finset.sum_eq_zero fun e' _ => by
        rw [(hpair_zero_left e' (m + 1)).1 i, zero_mul, zero_mul]
    have hs2 : (∑ i : Fin n, ∑ e' : AltEdge n S,
        lagCovariance μ T_env (altScore S ξ) (.unsplitTransfer i) e' (m + 1) * K.initial e' *
          ((K.prob ^ (m + 1)) e' (.unsplitTransfer i) - K.initial (.unsplitTransfer i))) = 0 :=
      Finset.sum_eq_zero fun i _ => Finset.sum_eq_zero fun e' _ => by
        rw [(hpair_zero_left e' (m + 1)).2 i, zero_mul, zero_mul]
    rw [hs1, hs2, zero_add, zero_add]
    have hinner : ∀ j : {x // x ∈ S},
        (∑ e' : AltEdge n S,
          lagCovariance μ T_env (altScore S ξ) (.altTransfer j) e' (m + 1) * K.initial e' *
            ((K.prob ^ (m + 1)) e' (.altTransfer j) - K.initial (.altTransfer j))) =
        (π j.1 ^ 2 * v j.1 / T) * ((φ j.1) ^ (m + 1) *
          ((ringStateProb n τ ^ m) (finRotate n k₀) k₀ * (1 / τ k₀) - 1 / T)) := by
      intro j
      rw [sum_altEdge]
      have hr1 : (∑ i : Fin n,
          lagCovariance μ T_env (altScore S ξ) (.altTransfer j) (.stay i) (m + 1) *
            K.initial (.stay i) *
            ((K.prob ^ (m + 1)) (.stay i) (.altTransfer j) - K.initial (.altTransfer j))) = 0 :=
        Finset.sum_eq_zero fun i _ => by rw [(hpair_zero_right _ (m + 1)).1 i, zero_mul, zero_mul]
      have hr2 : (∑ i : Fin n,
          lagCovariance μ T_env (altScore S ξ) (.altTransfer j) (.unsplitTransfer i) (m + 1) *
            K.initial (.unsplitTransfer i) *
            ((K.prob ^ (m + 1)) (.unsplitTransfer i) (.altTransfer j) -
              K.initial (.altTransfer j))) = 0 :=
        Finset.sum_eq_zero fun i _ => by rw [(hpair_zero_right _ (m + 1)).2 i, zero_mul, zero_mul]
      rw [hr1, hr2, zero_add, zero_add]
      simp_rw [fun j' => hlag_alt j j' (m + 1)]
      simp only [ite_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]
      dsimp [K, T]
      rw [splitRingChain_initial_altTransfer, splitRingChain_pow_altTransfer]
      ring
    simp_rw [hinner]
    rw [Finset.univ_eq_attach]
    exact Finset.sum_attach S (fun x =>
      (π x ^ 2 * v x / T) * ((φ x) ^ (m + 1) *
        ((ringStateProb n τ ^ m) (finRotate n k₀) k₀ * (1 / τ k₀) - 1 / T)))
  -- Step 3: Sum over `m : ℕ` for each `j ∈ S` using `ring_resolvent_values`.
  have hj_hasSum : ∀ j ∈ S,
      HasSum (fun m : ℕ => (π j ^ 2 * v j / T) * ((φ j) ^ (m + 1) *
        ((ringStateProb n τ ^ m) (finRotate n k₀) k₀ * (1 / τ k₀) - 1 / T)))
        ((π j ^ 2 * v j / T) * (R j / (1 - R j) - θ j / T)) := by
    intro j hj
    have hres_eq := (ring_resolvent_values n hn τ hτ k₀ (φ j) (hφ j hj)).1
    have hres_has : HasSum (fun m : ℕ => (φ j) ^ (m + 1) *
        ((ringStateProb n τ ^ m) (finRotate n k₀) k₀ * (1 / τ k₀))) (R j / (1 - R j)) := by
      rw [← hres_eq]
      exact (ringResolvent_summable n τ hτ k₀ (φ j) (hφ j hj) (finRotate n k₀)).hasSum
    have hgeom_has : HasSum (fun m : ℕ => (φ j) ^ (m + 1) * (1 / T)) (θ j / T) := by
      have hnorm : ‖φ j‖ < 1 := by rw [Real.norm_eq_abs]; exact hφ j hj
      have hg := ((hasSum_geometric_of_norm_lt_one hnorm).mul_left (φ j)).mul_right (1 / T)
      dsimp [θ, thetaOfPhi]
      convert hg using 1
      · funext m; rw [pow_succ']
      · ring
    have hdiff := (hres_has.sub hgeom_has).mul_left (π j ^ 2 * v j / T)
    convert hdiff using 1
    funext m; ring
  have htsum_eq : (∑' m : ℕ, ∑ e : AltEdge n S, ∑ e' : AltEdge n S,
      lagCovariance μ T_env (altScore S ξ) e e' (m + 1) * K.initial e' *
        ((K.prob ^ (m + 1)) e' e - K.initial e)) =
      ∑ j ∈ S, (π j ^ 2 * v j / T) * (R j / (1 - R j) - θ j / T) := by
    simp_rw [hlag_step]
    exact (hasSum_sum hj_hasSum).tsum_eq
  unfold Dobson.FinitePathCoefficient.covarianceCoefficient
  rw [hdiag0, htsum_eq, Finset.mul_sum, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j hj => ?_
  have hT_ne : T ≠ 0 := hT_pos.ne'
  have h1R_ne : 1 - R j ≠ 0 := (circuitR_abs_lt_one n hn (φ j) (hφ j hj) τ hτ).2
  field_simp [hT_ne, h1R_ne]
  ring

/-- Generation-normalized alternative-host formula for the explicitly
defined finite-path cost `altLocalCost`. The formula uses `altLocalCost_eq`;
it does not identify a physical infinite-time growth expansion. -/
theorem alternative_loss_unconditional {ι Ω : Type*} [DecidableEq ι] [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T_env : Ω ≃ᵐ Ω) (hT : MeasurePreserving T_env μ μ)
    (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (hprim : n = 1 ∨ ∃ k_s : Fin n, 1 < τ k_s)
    (k₀ : Fin n) (S : Finset ι) (π φ : ι → ℝ)
    (hπ0 : ∀ j ∈ S, 0 ≤ π j) (hπsum : ∑ j ∈ S, π j = 1)
    (hφ : ∀ j ∈ S, |φ j| < 1)
    (ξ : ι → Ω → ℝ) (hξ : ∀ j ∈ S, MemLp (ξ j) 2 μ)
    (hcov_self : ∀ j ∈ S, ∀ m : ℕ,
      (∫ ω, ξ j ((T_env : Ω → Ω)^[m] ω) * ξ j ω ∂μ) =
        (∫ ω, ξ j ω * ξ j ω ∂μ) * (φ j) ^ m)
    (hcov_cross : ∀ i ∈ S, ∀ j ∈ S, i ≠ j → ∀ m : ℕ,
      (∫ ω, ξ i ((T_env : Ω → Ω)^[m] ω) * ξ j ω ∂μ) = 0) :
    let T := genTime n τ
    let C := altLocalCost μ n hn τ hτ k₀ S π hπ0 hπsum T_env ξ
    let v : ι → ℝ := fun j => ∫ ω, ξ j ω * ξ j ω ∂μ
    let θ : ι → ℝ := fun j => thetaOfPhi (φ j)
    let R : ι → ℝ := fun j => circuitR n (φ j) τ
    T * C = ∑ j ∈ S, π j ^ 2 * v j * ((1 + 2 * θ j) / (2 * T) - R j / (1 - R j)) := by
  have hT_ne : genTime n τ ≠ 0 := (genTime_pos_of_ge_one n hn τ hτ).ne'
  have hexp := altLocalCost_eq μ T_env hT n hn τ hτ hprim k₀ S π φ hπ0 hπsum hφ ξ hξ
    hcov_self hcov_cross
  exact Dobson.Coverage2Alternatives.alternative_loss S π
    (fun j => ∫ ω, ξ j ω * ξ j ω ∂μ)
    (fun j => thetaOfPhi (φ j))
    (fun j => circuitR n (φ j) τ)
    (genTime n τ) (altLocalCost μ n hn τ hτ k₀ S π hπ0 hπsum T_env ξ)
    hT_ne hexp

end SplitTransferAlternatives

/-! ## Section 4: Fluctuating staying entry (`prop:ring-alt`, second half) -/

section FluctuatingStaying

/-- Score supported on the staying edge `(k₀, false)` of `ReferenceRing`. -/
def stayScore {n : ℕ} {Ω : Type*} (k₀ : Fin n) (ξ : Ω → ℝ) : Fin n × Bool → Ω → ℝ :=
  fun e ω => if e = (k₀, false) then ξ ω else 0

/-- Actual finite-horizon lognormal path-growth second-derivative cost limit for a fluctuating
staying entry `(k₀, false)` in `ReferenceRing`. -/
noncomputable def stayLocalCost {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (k₀ : Fin n) (T_env : Ω → Ω) (ξ : Ω → ℝ) : ℝ :=
  (-1 / 2 : ℝ) * limUnder atTop (fun N : ℕ =>
    iteratedDeriv 2 (Dobson.FinitePathJet.finiteGrowth μ
      (Chain.ofKernel (@Dobson.ReferenceRing.kernel (Fin n) inferInstance ⟨⟨0, by omega⟩⟩ inferInstance τ hτ (finRotate n)))
      T_env (stayScore k₀ ξ) 0 .lognormal N) 0)

/-- **Fluctuating staying cost formula (`prop:ring-alt`, second part):**
On `ReferenceRing`, an AR(1)-fluctuating staying entry (`1 < τ k₀`, `|ψ| < 1`) has actual
finite-path second-derivative cost limit
`v * (e_k ^ 2 * (1 + 2 * θ) / 2 - e_k * ψ * m_k / ((1 - ψ * m_k) * (1 - R)))`. -/
theorem staying_localCost_eq {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T_env : Ω ≃ᵐ Ω) (hT : MeasurePreserving T_env μ μ)
    (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (k₀ : Fin n) (hk₀ : 1 < τ k₀) (ψ : ℝ) (hψ : |ψ| < 1)
    (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ)
    (hcov : ∀ m : ℕ, (∫ ω, ξ ((T_env : Ω → Ω)^[m] ω) * ξ ω ∂μ) =
      (∫ ω, ξ ω * ξ ω ∂μ) * ψ ^ m) :
    let T := genTime n τ
    let v := ∫ ω, ξ ω * ξ ω ∂μ
    let m_k := 1 - 1 / τ k₀
    let e_k := (τ k₀ - 1) / T
    let θ := thetaOfPhi ψ
    let R := circuitR n ψ τ
    stayLocalCost μ n hn τ hτ k₀ T_env ξ =
      v * (e_k ^ 2 * (1 + 2 * θ) / 2 - e_k * ψ * m_k / ((1 - ψ * m_k) * (1 - R))) := by
  intro T v m_k e_k θ R
  let K_ker :=
    @Dobson.ReferenceRing.kernel (Fin n) inferInstance ⟨⟨0, by omega⟩⟩ inferInstance τ hτ (finRotate n)
  let K := Chain.ofKernel K_ker
  have hT_pos : 0 < T := genTime_pos_of_ge_one n hn τ hτ
  have hτk_pos : 0 < τ k₀ := by linarith
  have hproj : ∀ i j : Fin n,
      (∑ e : Fin n × Bool, K_ker.prob i e * (if K_ker.dest e = j then 1 else 0)) =
        ringStateProb n τ i j := by
    intro i j
    change (∑ e : Fin n × Bool, Dobson.ReferenceRing.probability τ i e *
      (if Dobson.ReferenceRing.destination (finRotate n) e = j then 1 else 0)) = _
    rw [Dobson.ReferenceRing.outgoing_sum]
    simp [Dobson.ReferenceRing.destination, ringStateProb]
  have hinit_eq : K_ker.initial = ringStateInitial n τ := rfl
  obtain ⟨Ku, β, hKu, hβ0, hβ1, hmix⟩ :=
    ofKernel_geometric_mixing n hn τ hτ (Or.inr ⟨k₀, hk₀⟩) K_ker hinit_eq hproj
  have hf_lp : ∀ e : Fin n × Bool, MemLp (stayScore k₀ ξ e) 2 μ := by
    intro e; unfold stayScore; split_ifs; exact hξ; exact MemLp.zero
  have hv_nn : 0 ≤ v := integral_nonneg fun ω => mul_self_nonneg (ξ ω)
  have hcov_bdd : ∀ e e' : Fin n × Bool, ∀ m : ℕ,
      |lagCovariance μ T_env (stayScore k₀ ξ) e e' m| ≤ v := by
    intro e e' m
    by_cases he : e = (k₀, false) <;> by_cases he' : e' = (k₀, false)
    · subst he; subst he'
      have h0 : lagCovariance μ T_env (stayScore k₀ ξ) (k₀, false) (k₀, false) m =
          ∫ ω, ξ ((T_env : Ω → Ω)^[m] ω) * ξ ω ∂μ := by simp [lagCovariance, stayScore]
      rw [h0]
      have hcs := stationary_autocov_abs_le_sum μ T_env hT ξ ξ hξ hξ m
      dsimp [v] at hcs ⊢; linarith
    · simp [lagCovariance, stayScore, he, he', hv_nn]
    · simp [lagCovariance, stayScore, he, he', hv_nn]
    · simp [lagCovariance, stayScore, he, he', hv_nn]
  have htend := Dobson.FinitePathCoefficient.finiteCoefficient_tendsto μ K T_env hT
    (stayScore k₀ ξ) hf_lp v Ku β hv_nn hKu hβ0 hβ1 hcov_bdd hmix
  have hsec : ∀ N : ℕ,
      iteratedDeriv 2 (Dobson.FinitePathJet.finiteGrowth μ K T_env (stayScore k₀ ξ) 0 .lognormal N) 0 =
        -2 * Dobson.FinitePathCoefficient.finiteCoefficient μ K T_env (stayScore k₀ ξ) N :=
    fun N => Dobson.FinitePathJet.finiteGrowth_second μ K T_env hT
      (stayScore k₀ ξ) hf_lp 0 .lognormal (by simp) N
  have hlim_deriv : Tendsto
      (fun N : ℕ => iteratedDeriv 2
        (Dobson.FinitePathJet.finiteGrowth μ K T_env (stayScore k₀ ξ) 0 .lognormal N) 0)
      atTop (𝓝 (-2 * Dobson.FinitePathCoefficient.covarianceCoefficient μ K T_env (stayScore k₀ ξ))) :=
    (htend.const_mul (-2 : ℝ)).congr fun N => (hsec N).symm
  have hC_eq : stayLocalCost μ n hn τ hτ k₀ T_env ξ =
      Dobson.FinitePathCoefficient.covarianceCoefficient μ K T_env (stayScore k₀ ξ) := by
    unfold stayLocalCost
    rw [hlim_deriv.limUnder_eq]; ring
  rw [hC_eq]
  have a_init : K.initial (k₀, false) = e_k := by
    change (∑ i : Fin n, ringStateInitial n τ i * Dobson.ReferenceRing.probability τ i (k₀, false)) = _
    simp only [Dobson.ReferenceRing.probability, Bool.false_eq_true, ↓reduceIte,
      mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ]
    dsimp [ringStateInitial, e_k, T]
    field_simp [hτk_pos.ne', hT_pos.ne']
  have a_pow : ∀ m : ℕ,
      (K.prob ^ (m + 1)) (k₀, false) (k₀, false) =
        (τ k₀ - 1) * ((ringStateProb n τ ^ m) k₀ k₀ * (1 / τ k₀)) := by
    intro m
    dsimp [K]
    rw [ofKernel_pow_succ n τ K_ker hproj m (k₀, false) (k₀, false)]
    change (∑ j : Fin n, (ringStateProb n τ ^ m) k₀ j *
      Dobson.ReferenceRing.probability τ j (k₀, false)) = _
    simp only [Dobson.ReferenceRing.probability, Bool.false_eq_true, ↓reduceIte,
      mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ]
    field_simp [hτk_pos.ne']
  have hlag_stay : ∀ e e' : Fin n × Bool, ∀ m : ℕ,
      lagCovariance μ T_env (stayScore k₀ ξ) e e' m =
        if e = (k₀, false) ∧ e' = (k₀, false) then v * ψ ^ m else 0 := by
    intro e e' m
    by_cases he : e = (k₀, false) <;> by_cases he' : e' = (k₀, false)
    · subst he; subst he'
      simp only [and_self, ↓reduceIte, lagCovariance, stayScore]
      exact hcov m
    · simp [lagCovariance, stayScore, he, he']
    · simp [lagCovariance, stayScore, he, he']
    · simp [lagCovariance, stayScore, he, he']
  have hcov_coeff : Dobson.FinitePathCoefficient.covarianceCoefficient μ K T_env (stayScore k₀ ξ) =
      (1 / 2 : ℝ) * (e_k ^ 2 * v) -
        ∑' m : ℕ, (v * ψ ^ (m + 1)) * e_k *
          ((K.prob ^ (m + 1)) (k₀, false) (k₀, false) - e_k) := by
    unfold Dobson.FinitePathCoefficient.covarianceCoefficient
    simp only [hlag_stay, pow_zero, mul_one, ite_and, mul_ite, ite_mul, mul_zero, zero_mul,
      Finset.sum_ite_irrel, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, Finset.sum_const_zero]
    rw [a_init]
    congr 1; ring
  rw [hcov_coeff]
  have hres_k₀ := (ring_resolvent_values n hn τ hτ k₀ ψ hψ).2
  have hres_has : HasSum (fun m : ℕ => ψ ^ (m + 1) *
      ((ringStateProb n τ ^ m) k₀ k₀ * (1 / τ k₀))) (stayPGF ψ (τ k₀) / (1 - R)) := by
    dsimp [R]
    rw [← hres_k₀]
    exact (ringResolvent_summable n τ hτ k₀ ψ hψ k₀).hasSum
  have hgeom_has : HasSum (fun m : ℕ => ψ ^ (m + 1) * e_k) (θ * e_k) := by
    have hnorm : ‖ψ‖ < 1 := by rw [Real.norm_eq_abs]; exact hψ
    have hg := ((hasSum_geometric_of_norm_lt_one hnorm).mul_left ψ).mul_right e_k
    dsimp [θ, thetaOfPhi]
    convert hg using 1
    · funext m; rw [pow_succ']
    · ring
  have hcomb := ((hres_has.mul_left (τ k₀ - 1)).sub hgeom_has).mul_left (v * e_k)
  have hfn : (fun m : ℕ => (v * e_k) * ((τ k₀ - 1) *
      (ψ ^ (m + 1) * ((ringStateProb n τ ^ m) k₀ k₀ * (1 / τ k₀))) - ψ ^ (m + 1) * e_k)) =
      fun m : ℕ => (v * ψ ^ (m + 1)) * e_k *
        ((K.prob ^ (m + 1)) (k₀, false) (k₀, false) - e_k) := by
    funext m; rw [a_pow m]; ring
  rw [hfn] at hcomb
  rw [hcomb.tsum_eq]
  have hden_pgf := (stayPGF_abs_lt_one ψ (τ k₀) hψ (hτ k₀)).1
  have h1mψ_pos : 0 < 1 - ψ * m_k := by
    have heq : τ k₀ * (1 - ψ * m_k) = τ k₀ - (τ k₀ - 1) * ψ := by
      dsimp [m_k]; field_simp [hτk_pos.ne']
    nlinarith
  have h1R_ne : 1 - R ≠ 0 := (circuitR_abs_lt_one n hn ψ hψ τ hτ).2
  have hstay_ratio : (τ k₀ - 1) * (stayPGF ψ (τ k₀) / (1 - R)) =
      ψ * m_k / ((1 - ψ * m_k) * (1 - R)) := by
    unfold stayPGF
    dsimp [m_k]
    field_simp [hτk_pos.ne', hden_pgf.ne', h1mψ_pos.ne', h1R_ne]
  rw [hstay_ratio]
  field_simp [h1mψ_pos.ne', h1R_ne]
  ring

/-- Sign criterion for the defined finite-path staying cost: for `1 < τ k₀`
and positive variance, the displayed inequality is equivalent to
`stayLocalCost < 0`. Identification with a physical growth gain is separate. -/
theorem staying_gain_iff_unconditional {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T_env : Ω ≃ᵐ Ω) (hT : MeasurePreserving T_env μ μ)
    (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (k₀ : Fin n) (hk₀ : 1 < τ k₀) (ψ : ℝ) (hψ : |ψ| < 1)
    (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ) (hv : 0 < ∫ ω, ξ ω * ξ ω ∂μ)
    (hcov : ∀ m : ℕ, (∫ ω, ξ ((T_env : Ω → Ω)^[m] ω) * ξ ω ∂μ) =
      (∫ ω, ξ ω * ξ ω ∂μ) * ψ ^ m) :
    let T := genTime n τ
    let m_k := 1 - 1 / τ k₀
    let e_k := (τ k₀ - 1) / T
    let θ := thetaOfPhi ψ
    let R := circuitR n ψ τ
    stayLocalCost μ n hn τ hτ k₀ T_env ξ < 0 ↔
      e_k * (1 + 2 * θ) / 2 < ψ * m_k / ((1 - ψ * m_k) * (1 - R)) := by
  intro T m_k e_k θ R
  have he_pos : 0 < e_k := div_pos (by linarith) (genTime_pos_of_ge_one n hn τ hτ)
  rw [staying_localCost_eq μ T_env hT n hn τ hτ k₀ hk₀ ψ hψ ξ hξ hcov]
  have hrew : (∫ ω, ξ ω * ξ ω ∂μ) *
      (e_k ^ 2 * (1 + 2 * θ) / 2 - e_k * ψ * m_k / ((1 - ψ * m_k) * (1 - R))) =
      (∫ ω, ξ ω * ξ ω ∂μ) *
        (e_k ^ 2 * (1 + 2 * θ) / 2 - e_k * (ψ * m_k / ((1 - ψ * m_k) * (1 - R)))) := by ring
  rw [hrew]
  exact Dobson.Coverage2Alternatives.staying_gain_iff _ e_k θ
    (ψ * m_k / ((1 - ψ * m_k) * (1 - R))) hv he_pos

end FluctuatingStaying

end Dobson.P1bA_RingAlt

#print axioms Dobson.P1bA_RingAlt.ringState_geometric_mixing
#print axioms Dobson.P1bA_RingAlt.ring_resolvent_values
#print axioms Dobson.P1bA_RingAlt.altLocalCost_eq
#print axioms Dobson.P1bA_RingAlt.alternative_loss_unconditional
#print axioms Dobson.P1bA_RingAlt.staying_localCost_eq
#print axioms Dobson.P1bA_RingAlt.staying_gain_iff_unconditional
