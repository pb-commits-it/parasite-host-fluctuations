import Mathlib

/-!
# The two auxiliary functions of the Jensen-gap lemma (items A1, A2, B1, B2, H1, H2, S1)

The elementary skeleton of Proposition 2 (and one step of Proposition 3) of the proof that the
loss decreases as a stay lengthens. All variables are real unless said.

* The even part `2·(cosh x − 1) − x·sinh x` is written out in every statement (A1, A2, H1); it
  has no definition of its own.
* `B x = 2·sinh x / x − cosh x − 1` is the definition `B`. It is a total function: at `x = 0`
  Lean reads `2·sinh 0 / 0` as `0`, so `B 0 = −2`. Every statement about `B` below is for
  `x > 0` or `x ≠ 0`, where the division is the usual one.
* `h y = (2 − y)·e^y − 2 − y` is the definition `h`.
* `g x = 2·sinh x − x·cosh x − x` is a helper (`g x = x·B x`, lemma `B_eq_g_div`), not an item.
* In S1 the natural number `n` is coerced to a real: `Real.sinh ((n : ℝ) * x)`.

Routes. A1: the function vanishes at `0` and has derivative `sinh x − x·cosh x`, which is
negative for `x > 0` because it vanishes at `0` and has derivative `−x·sinh x`. B2: `g` vanishes
at `0` and has derivative `cosh x − x·sinh x − 1`, negative for `x > 0` by A1 and `cosh x > 1`.
B1: `g` is strictly concave on `[0, ∞)` (second derivative `−x·cosh x`) and `g 0 = 0`, so the
slope `g x / x` of its secant from `0` is strictly decreasing. S1: induction on `n` through
`sinh((n+1)x)/sinh x = cosh(n x) + cosh x · sinh(n x)/sinh x`.
-/

namespace Dobson.JensenGap

/-- `B(x) = 2·sinh x / x − cosh x − 1` (the odd part of `h` divided by `x`). -/
noncomputable def B (x : ℝ) : ℝ := 2 * Real.sinh x / x - Real.cosh x - 1

/-- `h(y) = (2 − y)·e^y − 2 − y`. -/
noncomputable def h (y : ℝ) : ℝ := (2 - y) * Real.exp y - 2 - y

/-- Helper: `g(x) = 2·sinh x − x·cosh x − x`, which is `x·B(x)`. -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sinh x - x * Real.cosh x - x

/-! ## A1 and A2: the even part -/

/-- The derivative of `sinh t − t·cosh t` is `−t·sinh t`. -/
theorem hasDerivAt_sinh_sub_mul_cosh (x : ℝ) :
    HasDerivAt (fun t : ℝ => Real.sinh t - t * Real.cosh t) (-(x * Real.sinh x)) x := by
  have h1 : HasDerivAt (fun t : ℝ => t * Real.cosh t) (1 * Real.cosh x + x * Real.sinh x) x :=
    (hasDerivAt_id' x).fun_mul (Real.hasDerivAt_cosh x)
  have h2 := (Real.hasDerivAt_sinh x).fun_sub h1
  have e : Real.cosh x - (1 * Real.cosh x + x * Real.sinh x) = -(x * Real.sinh x) := by ring
  rw [e] at h2
  exact h2

/-- For `x > 0`: `sinh x − x·cosh x < 0` (that is, `tanh x < x`). -/
theorem sinh_sub_mul_cosh_neg {x : ℝ} (hx : 0 < x) : Real.sinh x - x * Real.cosh x < 0 := by
  have hanti : StrictAntiOn (fun t : ℝ => Real.sinh t - t * Real.cosh t) (Set.Ici 0) := by
    refine strictAntiOn_of_deriv_neg (convex_Ici 0) ?_ ?_
    · exact fun t _ => (hasDerivAt_sinh_sub_mul_cosh t).continuousAt.continuousWithinAt
    · intro t ht
      rw [interior_Ici] at ht
      have ht0 : 0 < t := ht
      rw [(hasDerivAt_sinh_sub_mul_cosh t).deriv]
      have hpos : 0 < t * Real.sinh t := mul_pos ht0 (Real.sinh_pos_iff.mpr ht0)
      linarith
  have key := hanti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx.le) hx
  simpa using key

/-- The derivative of `2·(cosh t − 1) − t·sinh t` is `sinh t − t·cosh t`. -/
theorem hasDerivAt_even_part (x : ℝ) :
    HasDerivAt (fun t : ℝ => 2 * (Real.cosh t - 1) - t * Real.sinh t)
      (Real.sinh x - x * Real.cosh x) x := by
  have h1 : HasDerivAt (fun t : ℝ => 2 * (Real.cosh t - 1)) (2 * Real.sinh x) x :=
    ((Real.hasDerivAt_cosh x).sub_const 1).const_mul 2
  have h2 : HasDerivAt (fun t : ℝ => t * Real.sinh t) (1 * Real.sinh x + x * Real.cosh x) x :=
    (hasDerivAt_id' x).fun_mul (Real.hasDerivAt_sinh x)
  have h3 := h1.fun_sub h2
  have e : 2 * Real.sinh x - (1 * Real.sinh x + x * Real.cosh x)
      = Real.sinh x - x * Real.cosh x := by ring
  rw [e] at h3
  exact h3

/-- **A1.** For `x > 0`: `2·(cosh x − 1) − x·sinh x < 0`. -/
theorem A1 (x : ℝ) (hx : 0 < x) : 2 * (Real.cosh x - 1) - x * Real.sinh x < 0 := by
  have hanti :
      StrictAntiOn (fun t : ℝ => 2 * (Real.cosh t - 1) - t * Real.sinh t) (Set.Ici 0) := by
    refine strictAntiOn_of_deriv_neg (convex_Ici 0) ?_ ?_
    · exact fun t _ => (hasDerivAt_even_part t).continuousAt.continuousWithinAt
    · intro t ht
      rw [interior_Ici] at ht
      rw [(hasDerivAt_even_part t).deriv]
      exact sinh_sub_mul_cosh_neg ht
  have key := hanti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx.le) hx
  simpa using key

/-- **A2.** For every real `x`: `2·(cosh x − 1) − x·sinh x ≤ 0`, with equality iff `x = 0`. -/
theorem A2 (x : ℝ) :
    2 * (Real.cosh x - 1) - x * Real.sinh x ≤ 0
      ∧ (2 * (Real.cosh x - 1) - x * Real.sinh x = 0 ↔ x = 0) := by
  rcases lt_trichotomy x 0 with hx | hx | hx
  · -- the function is even
    have h1 := A1 (-x) (neg_pos.mpr hx)
    rw [Real.cosh_neg, Real.sinh_neg] at h1
    have e : -x * -Real.sinh x = x * Real.sinh x := by ring
    rw [e] at h1
    exact ⟨h1.le, fun h0 => absurd h0 h1.ne, fun h0 => absurd h0 hx.ne⟩
  · subst hx
    simp
  · have h1 := A1 x hx
    exact ⟨h1.le, fun h0 => absurd h0 h1.ne, fun h0 => absurd h0 hx.ne'⟩

/-! ## B1 and B2: the odd part divided by `x` -/

/-- `g(0) = 0`. -/
theorem g_zero : g 0 = 0 := by
  simp [g]

/-- For `x ≠ 0`: `B(x) = g(x)/x`. -/
theorem B_eq_g_div (x : ℝ) (hx : x ≠ 0) : B x = g x / x := by
  unfold B g
  field_simp

/-- The derivative of `g` is `cosh x − x·sinh x − 1`. -/
theorem hasDerivAt_g (x : ℝ) : HasDerivAt g (Real.cosh x - x * Real.sinh x - 1) x := by
  have h1 : HasDerivAt (fun t : ℝ => 2 * Real.sinh t) (2 * Real.cosh x) x :=
    (Real.hasDerivAt_sinh x).const_mul 2
  have h2 : HasDerivAt (fun t : ℝ => t * Real.cosh t) (1 * Real.cosh x + x * Real.sinh x) x :=
    (hasDerivAt_id' x).fun_mul (Real.hasDerivAt_cosh x)
  have h3 := (h1.fun_sub h2).fun_sub (hasDerivAt_id' x)
  have e : 2 * Real.cosh x - (1 * Real.cosh x + x * Real.sinh x) - 1
      = Real.cosh x - x * Real.sinh x - 1 := by ring
  rw [e] at h3
  exact h3

/-- `deriv` form of `hasDerivAt_g`, as an equality of functions. -/
theorem deriv_g : deriv g = fun x : ℝ => Real.cosh x - x * Real.sinh x - 1 :=
  funext fun x => (hasDerivAt_g x).deriv

/-- The second derivative of `g` is `−x·cosh x`. -/
theorem hasDerivAt_deriv_g (x : ℝ) : HasDerivAt (deriv g) (-(x * Real.cosh x)) x := by
  rw [deriv_g]
  have h2 : HasDerivAt (fun t : ℝ => t * Real.sinh t) (1 * Real.sinh x + x * Real.cosh x) x :=
    (hasDerivAt_id' x).fun_mul (Real.hasDerivAt_sinh x)
  have h3 := ((Real.hasDerivAt_cosh x).fun_sub h2).sub_const 1
  have e : Real.sinh x - (1 * Real.sinh x + x * Real.cosh x) = -(x * Real.cosh x) := by ring
  rw [e] at h3
  exact h3

/-- For `x > 0`: `g(x) < 0`, that is, `2·sinh x < x·(cosh x + 1)`. -/
theorem g_neg {x : ℝ} (hx : 0 < x) : g x < 0 := by
  have hanti : StrictAntiOn g (Set.Ici 0) := by
    refine strictAntiOn_of_deriv_neg (convex_Ici 0) ?_ ?_
    · exact fun t _ => (hasDerivAt_g t).continuousAt.continuousWithinAt
    · intro t ht
      rw [interior_Ici] at ht
      have ht0 : 0 < t := ht
      rw [(hasDerivAt_g t).deriv]
      have h1 := A1 t ht0
      have h2 : 1 < Real.cosh t := Real.one_lt_cosh.mpr ht0.ne'
      linarith
  have key := hanti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx.le) hx
  rwa [g_zero] at key

/-- `g` is strictly concave on `[0, ∞)`. -/
theorem strictConcaveOn_g : StrictConcaveOn ℝ (Set.Ici 0) g := by
  refine strictConcaveOn_of_deriv2_neg (convex_Ici 0) ?_ ?_
  · exact fun t _ => (hasDerivAt_g t).continuousAt.continuousWithinAt
  · intro t ht
    rw [interior_Ici] at ht
    have ht0 : 0 < t := ht
    have e : deriv^[2] g t = -(t * Real.cosh t) := (hasDerivAt_deriv_g t).deriv
    rw [e]
    have hpos : 0 < t * Real.cosh t := mul_pos ht0 (Real.cosh_pos t)
    linarith

/-- **B1.** For `0 < x < y`: `B y < B x` (`B` is strictly decreasing on the positive
half-line). -/
theorem B1 (x y : ℝ) (hx : 0 < x) (hxy : x < y) : B y < B x := by
  have hy : 0 < y := hx.trans hxy
  have key := strictConcaveOn_g.secant_strict_mono (a := 0) (x := x) (y := y)
    (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx.le) (Set.mem_Ici.mpr hy.le) hx.ne' hy.ne' hxy
  rw [g_zero, sub_zero, sub_zero, sub_zero, sub_zero] at key
  rw [B_eq_g_div x hx.ne', B_eq_g_div y hy.ne']
  exact key

/-- **B1, as a statement about the function.** `B` is strictly decreasing on `(0, ∞)`. -/
theorem B1_strictAntiOn : StrictAntiOn B (Set.Ioi 0) :=
  fun x hx y _ hxy => B1 x y hx hxy

/-- **B2.** For `x > 0`: `B x < 0`. -/
theorem B2 (x : ℝ) (hx : 0 < x) : B x < 0 := by
  rw [B_eq_g_div x hx.ne']
  exact div_neg_of_neg_of_pos (g_neg hx) hx

/-! ## H1 and H2: the decomposition of `h` -/

/-- **H1 (the decomposition used in the proof).** For every `x`,
`(h x + h (−x))/2 = 2·(cosh x − 1) − x·sinh x`, and for `x ≠ 0`,
`(h x − h (−x))/(2·x) = B x`. -/
theorem H1 (x : ℝ) :
    (h x + h (-x)) / 2 = 2 * (Real.cosh x - 1) - x * Real.sinh x
      ∧ (x ≠ 0 → (h x - h (-x)) / (2 * x) = B x) := by
  refine ⟨?_, fun hx => ?_⟩
  · unfold h
    rw [Real.cosh_eq, Real.sinh_eq]
    ring
  · unfold h B
    rw [Real.cosh_eq, Real.sinh_eq]
    field_simp
    ring

/-- **H2.** For every `y`: `h(y) = 2·e^y − y·e^y − 2 − y`. -/
theorem H2 (y : ℝ) : h y = 2 * Real.exp y - y * Real.exp y - 2 - y := by
  unfold h
  ring

/-! ## S1: the ratio `sinh(n x)/sinh x` -/

/-- The recursion `sinh((n+1)x)/sinh x = cosh(n x) + cosh x · (sinh(n x)/sinh x)`, for
`sinh x ≠ 0`. -/
theorem sinh_ratio_succ (n : ℕ) (x : ℝ) (hx : Real.sinh x ≠ 0) :
    Real.sinh (((n + 1 : ℕ) : ℝ) * x) / Real.sinh x
      = Real.cosh ((n : ℝ) * x) + Real.cosh x * (Real.sinh ((n : ℝ) * x) / Real.sinh x) := by
  have e : ((n + 1 : ℕ) : ℝ) * x = (n : ℝ) * x + x := by
    push_cast
    ring
  rw [e, Real.sinh_add]
  field_simp
  ring

/-- One step of the induction: if `sinh(n x)/sinh x` is non-decreasing on `(0, ∞)` and `n ≥ 1`,
then `sinh((n+1)x)/sinh x` is strictly increasing on `(0, ∞)`. -/
theorem sinh_ratio_step (n : ℕ) (hn : 1 ≤ n)
    (ih : ∀ x y : ℝ, 0 < x → x < y →
      Real.sinh ((n : ℝ) * x) / Real.sinh x ≤ Real.sinh ((n : ℝ) * y) / Real.sinh y)
    (x y : ℝ) (hx : 0 < x) (hxy : x < y) :
    Real.sinh (((n + 1 : ℕ) : ℝ) * x) / Real.sinh x
      < Real.sinh (((n + 1 : ℕ) : ℝ) * y) / Real.sinh y := by
  have hy : 0 < y := hx.trans hxy
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hsx : 0 < Real.sinh x := Real.sinh_pos_iff.mpr hx
  have hsy : 0 < Real.sinh y := Real.sinh_pos_iff.mpr hy
  have hnx : 0 < (n : ℝ) * x := mul_pos hn' hx
  have hnxy : (n : ℝ) * x < (n : ℝ) * y := mul_lt_mul_of_pos_left hxy hn'
  have hny : 0 < (n : ℝ) * y := hnx.trans hnxy
  -- `cosh(n x) < cosh(n y)` and `cosh x < cosh y`
  have hc1 : Real.cosh ((n : ℝ) * x) < Real.cosh ((n : ℝ) * y) := by
    rw [Real.cosh_lt_cosh, abs_of_pos hnx, abs_of_pos hny]
    exact hnxy
  have hc2 : Real.cosh x < Real.cosh y := by
    rw [Real.cosh_lt_cosh, abs_of_pos hx, abs_of_pos hy]
    exact hxy
  -- the ratio at `x` is non-negative
  have hr0 : 0 ≤ Real.sinh ((n : ℝ) * x) / Real.sinh x :=
    (div_pos (Real.sinh_pos_iff.mpr hnx) hsx).le
  have hmul : Real.cosh x * (Real.sinh ((n : ℝ) * x) / Real.sinh x)
      ≤ Real.cosh y * (Real.sinh ((n : ℝ) * y) / Real.sinh y) :=
    mul_le_mul hc2.le (ih x y hx hxy) hr0 (Real.cosh_pos y).le
  rw [sinh_ratio_succ n x hsx.ne', sinh_ratio_succ n y hsy.ne']
  linarith

/-- For every natural `n ≥ 1` and `0 < x < y`: `sinh(n x)/sinh x ≤ sinh(n y)/sinh y`
(for `n = 1` both sides are `1`). -/
theorem sinh_ratio_le (n : ℕ) (hn : 1 ≤ n) (x y : ℝ) (hx : 0 < x) (hxy : x < y) :
    Real.sinh ((n : ℝ) * x) / Real.sinh x ≤ Real.sinh ((n : ℝ) * y) / Real.sinh y := by
  induction n, hn using Nat.le_induction generalizing x y with
  | base =>
    have hy : 0 < y := hx.trans hxy
    have hsx : Real.sinh x ≠ 0 := (Real.sinh_pos_iff.mpr hx).ne'
    have hsy : Real.sinh y ≠ 0 := (Real.sinh_pos_iff.mpr hy).ne'
    simp [hsx, hsy]
  | succ n hn ih =>
    exact (sinh_ratio_step n hn (fun x y hx hxy => ih x y hx hxy) x y hx hxy).le

/-- **S1 (used for the single crossing).** For every natural `n ≥ 2` and `0 < x < y`:
`sinh(n x)/sinh x < sinh(n y)/sinh y`. -/
theorem S1 (n : ℕ) (hn : 2 ≤ n) (x y : ℝ) (hx : 0 < x) (hxy : x < y) :
    Real.sinh ((n : ℝ) * x) / Real.sinh x < Real.sinh ((n : ℝ) * y) / Real.sinh y := by
  obtain ⟨m, rfl⟩ : ∃ m : ℕ, n = m + 1 := ⟨n - 1, by omega⟩
  have hm : 1 ≤ m := by omega
  exact sinh_ratio_step m hm (fun x y hx hxy => sinh_ratio_le m hm x y hx hxy) x y hx hxy

/-- **S1, the case `n = 2` in closed form.** For `x ≠ 0`: `sinh(2x)/sinh x = 2·cosh x`. -/
theorem S1_two (x : ℝ) (hx : x ≠ 0) : Real.sinh (2 * x) / Real.sinh x = 2 * Real.cosh x := by
  have hs : Real.sinh x ≠ 0 := fun h0 => hx (Real.sinh_eq_zero.mp h0)
  rw [Real.sinh_two_mul]
  field_simp

/-- **S1, the case `n = 3` in closed form.** For `x ≠ 0`:
`sinh(3x)/sinh x = 4·cosh² x − 1` and `sinh(3x)/sinh x = 2·cosh(2x) + 1`. -/
theorem S1_three (x : ℝ) (hx : x ≠ 0) :
    Real.sinh (3 * x) / Real.sinh x = 4 * Real.cosh x ^ 2 - 1
      ∧ Real.sinh (3 * x) / Real.sinh x = 2 * Real.cosh (2 * x) + 1 := by
  have hs : Real.sinh x ≠ 0 := fun h0 => hx (Real.sinh_eq_zero.mp h0)
  have hsq : Real.cosh x ^ 2 - Real.sinh x ^ 2 = 1 := Real.cosh_sq_sub_sinh_sq x
  have e1 : Real.sinh (3 * x) / Real.sinh x = 4 * Real.cosh x ^ 2 - 1 := by
    rw [Real.sinh_three_mul]
    field_simp
    linear_combination (-4 : ℝ) * hsq
  refine ⟨e1, ?_⟩
  rw [e1, Real.cosh_two_mul]
  linear_combination (2 : ℝ) * hsq

end Dobson.JensenGap

#print axioms Dobson.JensenGap.A1
#print axioms Dobson.JensenGap.A2
#print axioms Dobson.JensenGap.B1
#print axioms Dobson.JensenGap.B1_strictAntiOn
#print axioms Dobson.JensenGap.B2
#print axioms Dobson.JensenGap.H1
#print axioms Dobson.JensenGap.H2
#print axioms Dobson.JensenGap.S1
#print axioms Dobson.JensenGap.S1_two
#print axioms Dobson.JensenGap.S1_three
