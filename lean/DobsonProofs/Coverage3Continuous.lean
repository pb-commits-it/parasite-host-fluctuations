import Mathlib.Tactic

/-!
# Coverage batch 3: continuous-time ring coefficients
Deterministic core of `prop:ring-ct`. General finite-ring transfer positivity,
scalar cancellation, two-stage loss sign, and the unequal-stay comparison are
proved for the displayed coefficients. Their identification with OU perturbation
coefficients, monotonicity in stage count and asymptotic claims remain outside scope.
-/
namespace Dobson.Coverage3Continuous
open scoped BigOperators

/-- Product bound, reproduced here to avoid importing unrelated batch files. -/
theorem product_ge_linear {ι : Type*} (S : Finset ι) (x : ι → ℝ)
    (hx : ∀ i ∈ S, 0 ≤ x i) :
    1 + ∑ i ∈ S, x i ≤ ∏ i ∈ S, (1 + x i) := by
  classical
  induction S using Finset.induction_on with
  | empty => simp
  | @insert a S ha ih =>
    have hxa := hx a (Finset.mem_insert_self a S)
    have hxS : ∀ i ∈ S, 0 ≤ x i := fun i hi => hx i (Finset.mem_insert_of_mem hi)
    have hb := ih hxS
    have hs := Finset.sum_nonneg hxS
    rw [Finset.sum_insert ha, Finset.prod_insert ha]
    nlinarith [mul_nonneg hxa hs, mul_le_mul_of_nonneg_left hb (by linarith : 0 ≤ 1 + x a)]

/-- Two strictly positive terms make the product bound strict. -/
theorem product_strict {ι : Type*} [DecidableEq ι] (S : Finset ι) (x : ι → ℝ)
    (hx : ∀ i ∈ S, 0 < x i) (a b : ι) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
    1 + ∑ i ∈ S, x i < ∏ i ∈ S, (1 + x i) := by
  have hxS : ∀ i ∈ S.erase a, 0 < x i := fun i hi => hx i (Finset.mem_of_mem_erase hi)
  have hrest := product_ge_linear (S.erase a) x (fun i hi => (hxS i hi).le)
  have hsum : 0 < ∑ i ∈ S.erase a, x i :=
    Finset.sum_pos hxS ⟨b, Finset.mem_erase.mpr ⟨Ne.symm hab, hb⟩⟩
  have hxa := hx a ha
  have hnot : a ∉ S.erase a := by simp
  rw [← Finset.insert_erase ha, Finset.sum_insert hnot, Finset.prod_insert hnot]
  nlinarith [mul_pos hxa hsum,
    mul_le_mul_of_nonneg_left hrest (by linarith : 0 ≤ 1 + x a)]

/-- Reciprocal-rate product equals the circuit Laplace transform. -/
theorem circuit_product {ι : Type*} (S : Finset ι) (a : ι → ℝ)
    (α : ℝ) (hα : 0 < α) (ha : ∀ i ∈ S, 0 < a i) :
    (∏ i ∈ S, a i / (a i + α)) = 1 / ∏ i ∈ S, (1 + α / a i) := by
  rw [one_div, ← Finset.prod_inv_distrib]
  apply Finset.prod_congr rfl
  intro i hi
  have hsum : a i + α ≠ 0 := ne_of_gt (add_pos (ha i hi) hα)
  have hane : a i ≠ 0 := ne_of_gt (ha i hi)
  rw [← one_div]
  field_simp

/-- Strict product slack gives both transfer-coefficient bounds. -/
theorem transfer_band (α T P : ℝ) (hα : 0 < α) (hT : 0 < T)
    (hP : 1 + α * T < P) :
    0 < 1 / (α * T ^ 2) - 1 / T * ((1 / P) / (1 - 1 / P)) ∧
    1 / (α * T ^ 2) - 1 / T * ((1 / P) / (1 - 1 / P)) < 1 / (α * T ^ 2) := by
  have hp : 1 < P := by nlinarith [mul_pos hα hT]
  have hr : (1 / P) / (1 - 1 / P) = 1 / (P - 1) := by field_simp
  rw [hr]
  have he : 1 / T * (1 / (P - 1)) = 1 / (T * (P - 1)) := by field_simp
  rw [he]
  have hc : α * T ^ 2 < T * (P - 1) := by
    nlinarith [mul_pos hT (sub_pos.mpr hP)]
  have hlt := one_div_lt_one_div_of_lt (by positivity : 0 < α * T ^ 2) hc
  have hpos : 0 < 1 / (T * (P - 1)) := by positivity
  exact ⟨by linarith, by linarith⟩

/-- Every finite ring with at least two positive stays has positive transfer cost. -/
theorem transfer_general {ι : Type*} [DecidableEq ι] (S : Finset ι) (τ : ι → ℝ)
    (hτ : ∀ i ∈ S, 0 < τ i) (a b : ι) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b)
    (α : ℝ) (hα : 0 < α) :
    0 < 1 / (α * (∑ i ∈ S, τ i) ^ 2) - 1 / (∑ i ∈ S, τ i) *
      ((1 / ∏ i ∈ S, (1 + α * τ i)) / (1 - 1 / ∏ i ∈ S, (1 + α * τ i))) ∧
    1 / (α * (∑ i ∈ S, τ i) ^ 2) - 1 / (∑ i ∈ S, τ i) *
      ((1 / ∏ i ∈ S, (1 + α * τ i)) / (1 - 1 / ∏ i ∈ S, (1 + α * τ i))) <
      1 / (α * (∑ i ∈ S, τ i) ^ 2) := by
  apply transfer_band α _ _ hα (Finset.sum_pos hτ ⟨a, ha⟩)
  simpa only [Finset.mul_sum] using
    product_strict S (fun i => α * τ i) (fun i hi => mul_pos hα (hτ i hi)) a b ha hb hab

/-- Both displayed coefficients cancel for one stage. This is a coefficient claim. -/
theorem scalar_coefficients (a α : ℝ) (ha : 0 < a) (hα : 0 < α) :
    1 / (α * (1 / a) ^ 2) - a * ((a / (a + α)) / (1 - a / (a + α))) = 0 ∧
    1 / α - 1 / ((a + α) * (1 - a / (a + α))) = 0 := by
  have hsum : a + α ≠ 0 := ne_of_gt (add_pos ha hα)
  constructor <;> field_simp [ne_of_gt hα, ne_of_gt ha] <;> ring

/-- Two-stage net-loss cost is strictly negative, with an exact rational formula. -/
theorem two_stage_loss (a b α : ℝ) (ha : 0 < a) (hb : 0 < b) (hα : 0 < α) :
    b / (a + b) * (b / (a + b) / α -
      1 / ((a + α) * (1 - a * b / ((a + α) * (b + α))))) =
        -(a * b) / ((a + b) ^ 2 * (a + b + α)) ∧
    -(a * b) / ((a + b) ^ 2 * (a + b + α)) < 0 := by
  have hab : a + b ≠ 0 := ne_of_gt (add_pos ha hb)
  have haa : a + α ≠ 0 := ne_of_gt (add_pos ha hα)
  have hba : b + α ≠ 0 := ne_of_gt (add_pos hb hα)
  have habα : a + b + α ≠ 0 := ne_of_gt (by positivity)
  have hreturn : 1 - a * b / ((a + α) * (b + α)) =
      α * (a + b + α) / ((a + α) * (b + α)) := by
    field_simp
    ring
  constructor
  · rw [hreturn]
    field_simp [ne_of_gt hα]
    <;> ring
  · exact div_neg_of_neg_of_pos (neg_neg_of_pos (mul_pos ha hb)) (by positivity)

/-- Exact values behind the manuscript's rounded 0.022 comparison. -/
theorem unequal_stay_comparison :
    2 * ((1 : ℝ) / 2 - 1 / 3) = 1 / 3 ∧
    3 * ((1 : ℝ) / 102 - 1 / 403) = 301 / 13702 ∧
    (301 / 13702 : ℝ) < 1 / 3 := by norm_num

end Dobson.Coverage3Continuous
