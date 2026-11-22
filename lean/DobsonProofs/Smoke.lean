import Mathlib

-- 1. arithmetic
example : (2 : ℝ) + 2 = 4 := by norm_num

-- 2. a rational identity of the kind the project needs
example (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    1 / a + 1 / b = (a + b) / (a * b) := by
  field_simp
  ring

-- 3. an inequality of the kind the project needs: (a+b+c)(1/a+1/b+1/c) ≥ 9
example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    9 ≤ (a + b + c) * (1 / a + 1 / b + 1 / c) := by
  have h : (a + b + c) * (1 / a + 1 / b + 1 / c)
      = 3 + (a / b + b / a) + (a / c + c / a) + (b / c + c / b) := by
    field_simp
    ring
  have h1 : 2 ≤ a / b + b / a := by
    have : a / b + b / a - 2 = (a - b) ^ 2 / (a * b) := by field_simp; ring
    have h0 : 0 ≤ (a - b) ^ 2 / (a * b) := by positivity
    linarith
  have h2 : 2 ≤ a / c + c / a := by
    have : a / c + c / a - 2 = (a - c) ^ 2 / (a * c) := by field_simp; ring
    have h0 : 0 ≤ (a - c) ^ 2 / (a * c) := by positivity
    linarith
  have h3 : 2 ≤ b / c + c / b := by
    have : b / c + c / b - 2 = (b - c) ^ 2 / (b * c) := by field_simp; ring
    have h0 : 0 ≤ (b - c) ^ 2 / (b * c) := by positivity
    linarith
  linarith
