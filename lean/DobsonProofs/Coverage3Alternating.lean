import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic

/-!
# Coverage batch 3: alternating host
Core of `prop:exact-mem-alternating`. The closed threshold candidate solves the
characteristic equation; its log loss is negative with positive derivative,
including s = 0. Perron identification and invariant-law attraction are not proved.
-/
namespace Dobson.Coverage3Alternating
noncomputable def Q (δ s : ℝ) : ℝ := Real.sqrt (δ ^ 2 + (1 - δ ^ 2) * s ^ 2)
noncomputable def loss (δ s : ℝ) : ℝ := Real.log ((1 + s) / (1 + Q δ s))

/-- The square root lies strictly between s and one on the whole allowed interval. -/
theorem root_bounds (δ s : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hs0 : 0 ≤ s) (hs1 : s < 1) : s < Q δ s ∧ Q δ s < 1 := by
  have hd : 0 < δ ^ 2 := sq_pos_of_pos hδ0
  have hd1 : 0 < 1 - δ ^ 2 := by nlinarith
  have hs : 0 < 1 - s ^ 2 := by nlinarith
  have hr : 0 ≤ δ ^ 2 + (1 - δ ^ 2) * s ^ 2 := by positivity
  have hq : Q δ s ^ 2 = δ ^ 2 + (1 - δ ^ 2) * s ^ 2 := Real.sq_sqrt hr
  have hq0 : 0 ≤ Q δ s := Real.sqrt_nonneg _
  constructor
  · nlinarith [mul_pos hd hs]
  · nlinarith [mul_pos hd1 hs]

/-- Exact characteristic equation and restoring-ratio simplification. -/
theorem threshold_candidate (δ s : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hs0 : 0 ≤ s) (hs1 : s < 1) :
    1 - s ^ 2 - 2 * ((1 - s ^ 2) / (1 + Q δ s)) +
      ((1 - s ^ 2) / (1 + Q δ s)) ^ 2 * (1 - δ ^ 2) = 0 ∧
    ((1 - s ^ 2) / (1 + Q δ s)) / (1 - s) = (1 + s) / (1 + Q δ s) := by
  have hq0 := (root_bounds δ s hδ0 hδ1 hs0 hs1).1
  have hden : 1 + Q δ s ≠ 0 := by linarith
  have hs : 1 - s ≠ 0 := by linarith
  have hd : 0 ≤ 1 - δ ^ 2 := by nlinarith
  have hq : Q δ s ^ 2 = δ ^ 2 + (1 - δ ^ 2) * s ^ 2 := Real.sq_sqrt (by positivity)
  constructor
  · field_simp
    linear_combination (1 - s ^ 2) * hq
  · field_simp
    ring

/-- Negative threshold loss at every s in [0,1). -/
theorem loss_negative (δ s : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hs0 : 0 ≤ s) (hs1 : s < 1) : loss δ s < 0 := by
  have hq := (root_bounds δ s hδ0 hδ1 hs0 hs1).1
  have hd : 0 < 1 + Q δ s := by linarith
  apply Real.log_neg (div_pos (by linarith) hd)
  exact (div_lt_one hd).mpr (by linarith)

/-- A genuine derivative of the logarithmic formula, with no derivative hypothesis. -/
theorem loss_derivative (δ s : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hs0 : 0 ≤ s) (hs1 : s < 1) :
    HasDerivAt (loss δ)
      (1 / (1 + s) - (1 - δ ^ 2) * s / (Q δ s * (1 + Q δ s))) s := by
  have hd : 0 < 1 - δ ^ 2 := by nlinarith
  have hr : 0 < δ ^ 2 + (1 - δ ^ 2) * s ^ 2 := by positivity
  have hq : 0 < Q δ s := lt_of_le_of_lt hs0 (root_bounds δ s hδ0 hδ1 hs0 hs1).1
  have h1 : 1 + Q δ s ≠ 0 := ne_of_gt (by linarith)
  have h2 : (1 + s) / (1 + Q δ s) ≠ 0 := ne_of_gt (div_pos (by linarith) (by linarith))
  have hrad := (hasDerivAt_const s (δ ^ 2)).add
    (((hasDerivAt_id s).pow 2).const_mul (1 - δ ^ 2))
  have hroot := hrad.sqrt (ne_of_gt hr)
  have hlog := (((hasDerivAt_const s 1).add (hasDerivAt_id s)).div
    ((hasDerivAt_const s 1).add hroot) h1).log h2
  convert hlog using 1
  · rfl
  · dsimp [Q] at *
    field_simp
    ring

/-- The derivative is strictly positive even at the s = 0 endpoint. -/
theorem loss_derivative_positive (δ s : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hs0 : 0 ≤ s) (hs1 : s < 1) : 0 < deriv (loss δ) s := by
  rw [(loss_derivative δ s hδ0 hδ1 hs0 hs1).deriv]
  have hqs := (root_bounds δ s hδ0 hδ1 hs0 hs1).1
  have hq0 : 0 < Q δ s := lt_of_le_of_lt hs0 hqs
  have hd : 0 ≤ 1 - δ ^ 2 := by nlinarith
  have hq : Q δ s ^ 2 = δ ^ 2 + (1 - δ ^ 2) * s ^ 2 := Real.sq_sqrt (by positivity)
  have h1 : 1 + s ≠ 0 := ne_of_gt (by linarith)
  have h2 : 1 + Q δ s ≠ 0 := ne_of_gt (by linarith)
  have he : 1 / (1 + s) - (1 - δ ^ 2) * s / (Q δ s * (1 + Q δ s)) =
      (Q δ s + δ ^ 2 - (1 - δ ^ 2) * s) / ((1 + s) * Q δ s * (1 + Q δ s)) := by
    field_simp
    nlinarith [hq]
  rw [he]
  apply div_pos _ (by positivity)
  nlinarith [mul_nonneg (sq_nonneg δ) hs0, sq_pos_of_pos hδ0]

/-- The s = 0 threshold differs in sign from the usual log-host penalty. -/
theorem zero_staying (δ : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1) :
    loss δ 0 = -Real.log (1 + δ) ∧
    loss δ 0 ≠ -(1 / 2 : ℝ) * Real.log (1 - δ ^ 2) := by
  have hQ : Q δ 0 = δ := by simp [Q, Real.sqrt_sq_eq_abs, abs_of_pos hδ0]
  have hval : loss δ 0 = -Real.log (1 + δ) := by
    simp [loss, hQ]
  refine ⟨hval, ?_⟩
  have hn := loss_negative δ 0 hδ0 hδ1 (by norm_num) (by norm_num)
  have harg : 0 < 1 - δ ^ 2 := by nlinarith
  have hl := Real.log_neg harg (by nlinarith [sq_pos_of_pos hδ0])
  intro he
  rw [he] at hn
  nlinarith

end Dobson.Coverage3Alternating
