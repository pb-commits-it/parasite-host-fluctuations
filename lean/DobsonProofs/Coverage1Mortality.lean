import Mathlib

/-!
# Coverage batch 1 — Item 5: Mortality during a steady stay (`lem:mech-mortality`)

Formalises the scalar optimum core of `lem:mech-mortality` (`appendices/C-mechanisms.tex`, lines 270–281, and Table 2 of
`sections/05-mechanisms.tex`):

With one unit-sensitivity fluctuating transfer of noise amplitude `σ ≥ 0`, independent
environments across time, remaining primitive cycle mean reference duration `a > 0`, baseline
constant `C`, and per-step survival `exp(−μ)` (`μ > 0`) in an added fixed delay `b ≥ 0`, the
continuous relaxation of the second-order persistence margin is
`N₂(b) = C − μ b − σ² / (2 (a + b))`. This is the displayed scalar model;
its identification with a physical kernel and persistence margin is outside credit.

1. On `b ≥ 0`, `N₂` has unique maximizer `b* = max(0, σ / √(2 μ) − a)`:
   `N₂(b) ≤ N₂(b*)` for every `b ≥ 0`, with equality iff `b = b*`.
2. For any `b ≥ 0` (and in particular every integer delay `b : ℕ`), one additional step strictly
   improves `N₂` (`N₂(b + 1) > N₂(b)`) if and only if `μ < σ² / (2 (a + b) (a + b + 1))`.
   Consequently, when `μ ≥ σ² / (2 a (a + 1))`, no positive integer step helps.
-/

namespace Dobson.Coverage1Mortality

/-- Second-order persistence margin `N₂(b) = C − μ b − σ² / (2 (a + b))` under per-step survival
`exp(−μ)` during a fixed delay `b`. -/
noncomputable def N2 (C μ σ a b : ℝ) : ℝ :=
  C - μ * b - σ ^ 2 / (2 * (a + b))

/-- Continuous maximizer `b* = max(0, σ / √(2 μ) − a)` on `[0, ∞)`. -/
noncomputable def bStar (μ σ a : ℝ) : ℝ :=
  max 0 (σ / Real.sqrt (2 * μ) - a)

/-- First derivative `N₂'(b) = −μ + σ² / (2 (a + b)²)` for `a + b ≠ 0`. -/
theorem hasDerivAt_N2 (C μ σ a b : ℝ) (hab : a + b ≠ 0) :
    HasDerivAt (N2 C μ σ a) (-μ + σ ^ 2 / (2 * (a + b) ^ 2)) b := by
  unfold N2
  have hlin : HasDerivAt (fun x : ℝ => C - μ * x) (0 - μ) b :=
    (hasDerivAt_const b C).fun_sub (hasDerivAt_const_mul μ)
  have hden : HasDerivAt (fun x : ℝ => 2 * (a + x)) (2 * 1) b :=
    ((hasDerivAt_id' b).const_add a).const_mul 2
  have hden_ne : 2 * (a + b) ≠ 0 := mul_ne_zero two_ne_zero hab
  have hc : HasDerivAt (fun _ : ℝ => σ ^ 2) 0 b := hasDerivAt_const b (σ ^ 2)
  have hfrac := hc.fun_div hden hden_ne
  have hcomb := hlin.fun_sub hfrac
  have e : (0 - μ) - (0 * (2 * (a + b)) - σ ^ 2 * (2 * 1)) / (2 * (a + b)) ^ 2 =
      -μ + σ ^ 2 / (2 * (a + b) ^ 2) := by
    field_simp [hab]
    ring
  rw [e] at hcomb
  exact hcomb

/-- **Continuous optimum (`lem:mech-mortality`, part a).**
For `a > 0`, `μ > 0`, `σ ≥ 0`, and `b ≥ 0`, `b* = max(0, σ / √(2 μ) − a) ≥ 0` satisfies
`N₂(b) ≤ N₂(b*)`, with equality if and only if `b = b*`. -/
theorem N2_le_bStar_and_eq_iff (C μ σ a : ℝ) (ha : 0 < a) (hμ : 0 < μ) (hσ : 0 ≤ σ) :
    0 ≤ bStar μ σ a ∧
      ∀ b : ℝ, 0 ≤ b → (N2 C μ σ a b ≤ N2 C μ σ a (bStar μ σ a) ∧
        (N2 C μ σ a b = N2 C μ σ a (bStar μ σ a) ↔ b = bStar μ σ a)) := by
  have hbStar_nn : 0 ≤ bStar μ σ a := le_max_left _ _
  refine ⟨hbStar_nn, fun b hb => ?_⟩
  have hab : 0 < a + b := by linarith
  have habS : 0 < a + bStar μ σ a := by linarith
  have h2μ : 0 < 2 * μ := by positivity
  have hsqrt_pos : 0 < Real.sqrt (2 * μ) := Real.sqrt_pos.mpr h2μ
  have hsq : Real.sqrt (2 * μ) ^ 2 = 2 * μ := Real.sq_sqrt h2μ.le
  -- Exact algebraic difference:
  -- N₂(b*) - N₂(b) = (b - b*) * [2 μ (a + b) (a + b*) - σ²] / (2 (a + b) (a + b*))
  have hdiff : N2 C μ σ a (bStar μ σ a) - N2 C μ σ a b =
      (b - bStar μ σ a) * (2 * μ * (a + b) * (a + bStar μ σ a) - σ ^ 2) /
        (2 * (a + b) * (a + bStar μ σ a)) := by
    unfold N2
    field_simp [hab.ne', habS.ne']
    ring
  have hden_pos : 0 < 2 * (a + b) * (a + bStar μ σ a) := by positivity
  by_cases hcase : σ / Real.sqrt (2 * μ) - a ≤ 0
  · -- Endpoint regime: `b* = 0` and `σ² ≤ 2 μ a²`.
    have hbS0 : bStar μ σ a = 0 := max_eq_left hcase
    have hσ_le : σ ≤ a * Real.sqrt (2 * μ) := by
      have h1 : σ / Real.sqrt (2 * μ) ≤ a := by linarith
      exact (div_le_iff₀ hsqrt_pos).mp h1
    have hσ2_le : σ ^ 2 ≤ 2 * μ * a ^ 2 := by
      have hmul : σ ^ 2 ≤ (a * Real.sqrt (2 * μ)) ^ 2 :=
        pow_le_pow_left₀ hσ hσ_le 2
      calc σ ^ 2 ≤ a ^ 2 * Real.sqrt (2 * μ) ^ 2 := by linarith [hmul]
        _ = 2 * μ * a ^ 2 := by rw [hsq]; ring
    rw [hbS0] at hdiff habS hden_pos ⊢
    simp only [sub_zero, add_zero] at hdiff hden_pos
    have hnum_ge : 2 * μ * b * a ≤ 2 * μ * (a + b) * a - σ ^ 2 := by nlinarith
    constructor
    · have hnn : 0 ≤ b * (2 * μ * (a + b) * a - σ ^ 2) / (2 * (a + b) * a) := by
        apply div_nonneg _ hden_pos.le
        exact mul_nonneg hb (by nlinarith)
      linarith [hdiff]
    · constructor
      · intro heq
        have hzero : b * (2 * μ * (a + b) * a - σ ^ 2) = 0 := by
          have hd0 : N2 C μ σ a 0 - N2 C μ σ a b = 0 := by linarith
          rw [hdiff, div_eq_zero_iff] at hd0
          exact hd0.resolve_right hden_pos.ne'
        rcases mul_eq_zero.mp hzero with hb0 | hbr
        · exact hb0
        · rcases hb.eq_or_lt with hb0 | hb_pos
          · exact hb0.symm
          · exfalso
            have : 0 < 2 * μ * b * a := by positivity
            linarith
      · intro hb0
        rw [hb0]
  · -- Interior regime: `b* = σ / √(2 μ) − a > 0` and `a + b* = σ / √(2 μ)`.
    have hcase' : 0 < σ / Real.sqrt (2 * μ) - a := not_le.mp hcase
    have hbS_eq : bStar μ σ a = σ / Real.sqrt (2 * μ) - a := max_eq_right hcase'.le
    have habS_eq : a + bStar μ σ a = σ / Real.sqrt (2 * μ) := by linarith
    have hprod : 2 * μ * (a + bStar μ σ a) ^ 2 = σ ^ 2 := by
      have h1 : (a + bStar μ σ a) * Real.sqrt (2 * μ) = σ := by
        rw [habS_eq]
        exact div_mul_cancel₀ σ hsqrt_pos.ne'
      calc 2 * μ * (a + bStar μ σ a) ^ 2
          = (a + bStar μ σ a) ^ 2 * Real.sqrt (2 * μ) ^ 2 := by rw [hsq]; ring
        _ = ((a + bStar μ σ a) * Real.sqrt (2 * μ)) ^ 2 := by ring
        _ = σ ^ 2 := by rw [h1]
    have hident : 2 * μ * (a + b) * (a + bStar μ σ a) - σ ^ 2 =
        2 * μ * (a + bStar μ σ a) * (b - bStar μ σ a) := by
      rw [← hprod]
      ring
    have hdiff2 : N2 C μ σ a (bStar μ σ a) - N2 C μ σ a b =
        μ * (b - bStar μ σ a) ^ 2 / (a + b) := by
      rw [hdiff, hident]
      field_simp [hab.ne', habS.ne']
    constructor
    · have hnn : 0 ≤ μ * (b - bStar μ σ a) ^ 2 / (a + b) := by positivity
      linarith [hdiff2]
    · constructor
      · intro heq
        have hd0 : μ * (b - bStar μ σ a) ^ 2 / (a + b) = 0 := by linarith [hdiff2]
        rw [div_eq_zero_iff] at hd0
        have hmul0 := hd0.resolve_right hab.ne'
        have hsq0 : (b - bStar μ σ a) ^ 2 = 0 := (mul_eq_zero.mp hmul0).resolve_left hμ.ne'
        linarith [sq_eq_zero_iff.mp hsq0]
      · intro hbeq
        rw [hbeq]

/-- **Discrete one-step comparison (`lem:mech-mortality`, part b).**
For `a > 0` and `b ≥ 0`, `N₂(b + 1) > N₂(b) ↔ μ < σ² / (2 (a + b) (a + b + 1))`. -/
theorem N2_add_one_gt_iff (C μ σ a b : ℝ) (ha : 0 < a) (hb : 0 ≤ b) :
    N2 C μ σ a (b + 1) > N2 C μ σ a b ↔
      μ < σ ^ 2 / (2 * (a + b) * (a + b + 1)) := by
  have hab : 0 < a + b := by linarith
  have hab1 : 0 < a + b + 1 := by linarith
  have hdiff : N2 C μ σ a (b + 1) - N2 C μ σ a b =
      σ ^ 2 / (2 * (a + b) * (a + b + 1)) - μ := by
    unfold N2
    field_simp [hab.ne', hab1.ne', (by linarith : a + (b + 1) ≠ 0)]
    ring
  constructor <;> intro h <;> linarith

/-- Specialization of `N2_add_one_gt_iff` to integer delays `b : ℕ`. -/
theorem N2_nat_succ_gt_iff (C μ σ a : ℝ) (ha : 0 < a) (b : ℕ) :
    N2 C μ σ a ((b : ℝ) + 1) > N2 C μ σ a (b : ℝ) ↔
      μ < σ ^ 2 / (2 * (a + (b : ℝ)) * (a + (b : ℝ) + 1)) :=
  N2_add_one_gt_iff C μ σ a (b : ℝ) ha (Nat.cast_nonneg b)

/-- **Threshold for any positive integer delay (`lem:mech-mortality`, corollary in proof).**
When `μ ≥ σ² / (2 a (a + 1))` and `a > 0`, `N₂(b) ≤ N₂(0)` for every natural `n`
(and `N₂(b + 1) ≤ N₂(b)` for every `b ≥ 0`). -/
theorem N2_nonincreasing_of_hazard_ge (C μ σ a : ℝ) (ha : 0 < a) (_hσ : 0 ≤ σ)
    (hμ : σ ^ 2 / (2 * a * (a + 1)) ≤ μ) :
    (∀ b : ℝ, 0 ≤ b → N2 C μ σ a (b + 1) ≤ N2 C μ σ a b) ∧
      (∀ n : ℕ, N2 C μ σ a (n : ℝ) ≤ N2 C μ σ a 0) := by
  have hstep : ∀ b : ℝ, 0 ≤ b → N2 C μ σ a (b + 1) ≤ N2 C μ σ a b := by
    intro b hb
    by_contra hnot
    have hlt : N2 C μ σ a (b + 1) > N2 C μ σ a b := not_le.mp hnot
    have hμlt := (N2_add_one_gt_iff C μ σ a b ha hb).mp hlt
    have hden0 : 0 < 2 * a * (a + 1) := by positivity
    have hdenb : 2 * a * (a + 1) ≤ 2 * (a + b) * (a + b + 1) := by nlinarith
    have hfrac : σ ^ 2 / (2 * (a + b) * (a + b + 1)) ≤ σ ^ 2 / (2 * a * (a + 1)) :=
      div_le_div_of_nonneg_left (sq_nonneg σ) hden0 hdenb
    linarith
  refine ⟨hstep, fun n => ?_⟩
  induction n with
  | zero => simp
  | succ k ih =>
    push_cast
    exact (hstep (k : ℝ) (Nat.cast_nonneg k)).trans ih

end Dobson.Coverage1Mortality

#print axioms Dobson.Coverage1Mortality.hasDerivAt_N2
#print axioms Dobson.Coverage1Mortality.N2_le_bStar_and_eq_iff
#print axioms Dobson.Coverage1Mortality.N2_add_one_gt_iff
#print axioms Dobson.Coverage1Mortality.N2_nat_succ_gt_iff
#print axioms Dobson.Coverage1Mortality.N2_nonincreasing_of_hazard_ge
