import Mathlib

/-!
# Three stages (items T1 and T2)
-/

namespace Dobson.ThreeStages

/-- **T1.** For real `a, b, c > 0`: `(a + b + c)(1/a + 1/b + 1/c) ≥ 9`,
and equality holds if and only if `a = b = c` (written `a = b` and `b = c`). -/
theorem sum_mul_sum_inv_ge_nine (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 ∧
      ((a + b + c) * (1 / a + 1 / b + 1 / c) = 9 ↔ a = b ∧ b = c) := by
  have habc : 0 < a * b * c := by positivity
  have key : (a + b + c) * (1 / a + 1 / b + 1 / c) - 9
      = ((a - b) ^ 2 * c + (b - c) ^ 2 * a + (a - c) ^ 2 * b) / (a * b * c) := by
    field_simp
    ring
  have e1 : 0 ≤ (a - b) ^ 2 * c := by positivity
  have e2 : 0 ≤ (b - c) ^ 2 * a := by positivity
  have e3 : 0 ≤ (a - c) ^ 2 * b := by positivity
  constructor
  · have h : 0 ≤ ((a - b) ^ 2 * c + (b - c) ^ 2 * a + (a - c) ^ 2 * b) / (a * b * c) :=
      div_nonneg (by linarith) habc.le
    linarith
  · constructor
    · intro h
      have h0 : ((a - b) ^ 2 * c + (b - c) ^ 2 * a + (a - c) ^ 2 * b) / (a * b * c) = 0 := by
        rw [← key]
        linarith
      have h1 : (a - b) ^ 2 * c + (b - c) ^ 2 * a + (a - c) ^ 2 * b = 0 := by
        rcases div_eq_zero_iff.mp h0 with h | h
        · exact h
        · exact absurd h habc.ne'
      have z1 : (a - b) ^ 2 * c = 0 := by linarith
      have z2 : (b - c) ^ 2 * a = 0 := by linarith
      have hab : a - b = 0 := by
        rcases mul_eq_zero.mp z1 with h | h
        · exact pow_eq_zero_iff (two_ne_zero) |>.mp h
        · exact absurd h hc.ne'
      have hbc : b - c = 0 := by
        rcases mul_eq_zero.mp z2 with h | h
        · exact pow_eq_zero_iff (two_ne_zero) |>.mp h
        · exact absurd h ha.ne'
      exact ⟨by linarith, by linarith⟩
    · rintro ⟨hab, hbc⟩
      have ha' : a ≠ 0 := ha.ne'
      rw [← hbc, ← hab]
      field_simp
      ring

/-- **T2.** For real `a, b, c > 0`, real `g`, and real `ω ≠ 0`, with all of them regarded as
complex numbers and `i` the imaginary unit:
`(iω + a)(iω + b)(iω + c) + g·abc = 0` holds in `ℂ`
if and only if `ω² = ab + bc + ca` and `1 + g = (a + b + c)(1/a + 1/b + 1/c)`
(these last two are equations between real numbers). -/
theorem char_eq_zero_iff (a b c g ω : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hω : ω ≠ 0) :
    (Complex.I * (ω : ℂ) + (a : ℂ)) * (Complex.I * (ω : ℂ) + (b : ℂ))
          * (Complex.I * (ω : ℂ) + (c : ℂ))
        + (g : ℂ) * (a : ℂ) * (b : ℂ) * (c : ℂ) = 0
      ↔ ω ^ 2 = a * b + b * c + c * a ∧
        1 + g = (a + b + c) * (1 / a + 1 / b + 1 / c) := by
  have habc : 0 < a * b * c := by positivity
  have hexp : (Complex.I * (ω : ℂ) + (a : ℂ)) * (Complex.I * (ω : ℂ) + (b : ℂ))
          * (Complex.I * (ω : ℂ) + (c : ℂ))
        + (g : ℂ) * (a : ℂ) * (b : ℂ) * (c : ℂ)
      = (⟨(1 + g) * (a * b * c) - (a + b + c) * ω ^ 2,
          ω * ((a * b + b * c + c * a) - ω ^ 2)⟩ : ℂ) := by
    rw [Complex.mk_eq_add_mul_I]
    push_cast
    linear_combination
      ((ω : ℂ) ^ 3 * Complex.I + ((a : ℂ) + b + c) * (ω : ℂ) ^ 2) * Complex.I_sq
  have hS : (a + b + c) * (1 / a + 1 / b + 1 / c)
      = (a + b + c) * (a * b + b * c + c * a) / (a * b * c) := by
    field_simp
    ring
  rw [hexp, Complex.ext_iff]
  simp only [Complex.zero_re, Complex.zero_im]
  constructor
  · rintro ⟨hre, him⟩
    have h2 : (a * b + b * c + c * a) - ω ^ 2 = 0 := by
      rcases mul_eq_zero.mp him with h | h
      · exact absurd h hω
      · exact h
    have hω2 : ω ^ 2 = a * b + b * c + c * a := by linarith
    refine ⟨hω2, ?_⟩
    rw [hS, eq_div_iff habc.ne']
    rw [hω2] at hre
    linarith
  · rintro ⟨hω2, hg⟩
    rw [hS, eq_div_iff habc.ne'] at hg
    constructor
    · rw [hω2]
      linarith
    · rw [hω2]
      ring

end Dobson.ThreeStages

#print axioms Dobson.ThreeStages.sum_mul_sum_inv_ge_nine
#print axioms Dobson.ThreeStages.char_eq_zero_iff
