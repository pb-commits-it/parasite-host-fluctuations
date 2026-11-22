import Mathlib

/-!
# More hosts at a fixed reproduction number (items N1 to N3)

All variables are real. `Real.exp` is the exponential and `Real.log` the natural logarithm.

Encoding of the finite family in N3: an arbitrary index type `ι`, a finite set `s : Finset ι`
of indices, and functions `r p : ι → ℝ`; the hypotheses are assumed for `k ∈ s` and the
products run over `k ∈ s`.
-/

namespace Dobson.ReturnRate

/-- **N1.** For real `L > 0` the function `x ↦ x·(exp(L/x) − 1)` is strictly decreasing on
the open half-line `(0, ∞)`: if `0 < x < y` then `y·(exp(L/y) − 1) < x·(exp(L/x) − 1)`. -/
theorem N1 (L : ℝ) (hL : 0 < L) :
    StrictAntiOn (fun x : ℝ => x * (Real.exp (L / x) - 1)) (Set.Ioi 0) := by
  intro x hx y hy hxy
  simp only [Set.mem_Ioi] at hx hy
  have hLx : 0 < L / x := div_pos hL hx
  have hLy : 0 < L / y := div_pos hL hy
  have hlt : L / y < L / x := div_lt_div_of_pos_left hL hx hxy
  -- the secant slope of the strictly convex function `exp`, from `0`, is strictly increasing
  have h := strictConvexOn_exp.secant_strict_mono (a := 0) (x := L / y) (y := L / x)
    (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _) hLy.ne' hLx.ne' hlt
  simp only [Real.exp_zero, sub_zero] at h
  have e1 : (Real.exp (L / y) - 1) / (L / y) = y * (Real.exp (L / y) - 1) / L := by
    rw [div_div_eq_mul_div]
    ring
  have e2 : (Real.exp (L / x) - 1) / (L / x) = x * (Real.exp (L / x) - 1) / L := by
    rw [div_div_eq_mul_div]
    ring
  rw [e1, e2, div_lt_div_iff_of_pos_right hL] at h
  exact h

/-- **N2.** For real `L > 0` and `x > 0`: `x·(exp(L/x) − 1) > L`. -/
theorem N2 (L x : ℝ) (hL : 0 < L) (hx : 0 < x) : x * (Real.exp (L / x) - 1) > L := by
  have ht : 0 < L / x := div_pos hL hx
  have h := Real.add_one_lt_exp ht.ne'
  have h2 : x * (L / x) = L := by field_simp
  calc L = x * (L / x) := h2.symm
    _ < x * (Real.exp (L / x) - 1) := mul_lt_mul_of_pos_left (by linarith) hx

/-- **N3.** (Sum rule.) For a finite family of reals `r k > 0` and `p k` with `0 ≤ p k < 1`:
if `∏ (r k · (1 − p k)) = 1` then `(∏ r k)·(∏ (1 − p k)) = 1`.

The three sign hypotheses are not needed for this identity (it is the distributivity of a
finite product over multiplication); they are kept because they are in the statement. -/
theorem N3 {ι : Type*} (s : Finset ι) (r p : ι → ℝ)
    (_hr : ∀ k ∈ s, 0 < r k) (_hp0 : ∀ k ∈ s, 0 ≤ p k) (_hp1 : ∀ k ∈ s, p k < 1)
    (h : ∏ k ∈ s, (r k * (1 - p k)) = 1) :
    (∏ k ∈ s, r k) * (∏ k ∈ s, (1 - p k)) = 1 := by
  rw [← Finset.prod_mul_distrib]
  exact h

/-- Companion to N3, not one of the listed items: the same rule written as a sum of
logarithms. Under the hypotheses of N3, `∑ log (r k) = ∑ −log (1 − p k)`. Here `r k > 0` and
`p k < 1` are used (so that the logarithms are of positive numbers); `0 ≤ p k` is not. -/
theorem N3_log {ι : Type*} (s : Finset ι) (r p : ι → ℝ)
    (hr : ∀ k ∈ s, 0 < r k) (_hp0 : ∀ k ∈ s, 0 ≤ p k) (hp1 : ∀ k ∈ s, p k < 1)
    (h : ∏ k ∈ s, (r k * (1 - p k)) = 1) :
    ∑ k ∈ s, Real.log (r k) = ∑ k ∈ s, -Real.log (1 - p k) := by
  have hr' : ∀ k ∈ s, r k ≠ 0 := fun k hk => (hr k hk).ne'
  have hp' : ∀ k ∈ s, 1 - p k ≠ 0 := fun k hk => (sub_pos.mpr (hp1 k hk)).ne'
  have h1 : ∑ k ∈ s, Real.log (r k * (1 - p k)) = 0 := by
    rw [← Real.log_prod (fun k hk => mul_ne_zero (hr' k hk) (hp' k hk)), h, Real.log_one]
  have h2 : ∑ k ∈ s, Real.log (r k * (1 - p k))
      = ∑ k ∈ s, Real.log (r k) + ∑ k ∈ s, Real.log (1 - p k) := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl (fun k hk => Real.log_mul (hr' k hk) (hp' k hk))
  rw [Finset.sum_neg_distrib]
  linarith

end Dobson.ReturnRate

#print axioms Dobson.ReturnRate.N1
#print axioms Dobson.ReturnRate.N2
#print axioms Dobson.ReturnRate.N3
#print axioms Dobson.ReturnRate.N3_log
