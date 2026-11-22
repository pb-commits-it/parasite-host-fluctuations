import Mathlib

/-!
# The exact test for a host with its own dynamics (items X1 to X3)

`d`, `q`, `h`, `ρ`, `ω` are real numbers, regarded as complex numbers inside `Complex.normSq`
(the squared modulus `|w|² = (Re w)² + (Im w)²`), and `Complex.I` is the imaginary unit `i`.

The derived quantities `x = ω²`, `U = ρ² + d² − 2dh − q²` and `V = d²(ρ + h)² − q²ρ²` are
variables carrying those equations as hypotheses.
-/

namespace Dobson.ExactTest

/-- **X1.** For real `d`, `q`, `h`, `ρ`, `ω`, with `x = ω²`, `U = ρ² + d² − 2dh − q²` and
`V = d²(ρ + h)² − q²ρ²`:
`|(iω + ρ)(iω + d) + d·h|² − q²·|iω + ρ|² = x² + U·x + V`. -/
theorem X1 (d q h ρ ω x U V : ℝ) (hx : x = ω ^ 2)
    (hU : U = ρ ^ 2 + d ^ 2 - 2 * d * h - q ^ 2)
    (hV : V = d ^ 2 * (ρ + h) ^ 2 - q ^ 2 * ρ ^ 2) :
    Complex.normSq ((Complex.I * (ω : ℂ) + (ρ : ℂ)) * (Complex.I * (ω : ℂ) + (d : ℂ))
        + (d : ℂ) * (h : ℂ))
      - q ^ 2 * Complex.normSq (Complex.I * (ω : ℂ) + (ρ : ℂ))
      = x ^ 2 + U * x + V := by
  rw [hx, hU, hV]
  simp only [Complex.normSq_apply, Complex.add_re, Complex.add_im, Complex.mul_re,
    Complex.mul_im, Complex.I_re, Complex.I_im, Complex.ofReal_re, Complex.ofReal_im]
  ring

/-- **X2.** For real `U`, `V`: the quadratic `x² + U·x + V` is positive for every `x ≥ 0`
if and only if `0 < V` and `−2·√V < U`. -/
theorem X2 (U V : ℝ) :
    (∀ x : ℝ, 0 ≤ x → 0 < x ^ 2 + U * x + V) ↔ (0 < V ∧ -2 * Real.sqrt V < U) := by
  constructor
  · intro h
    have hV : 0 < V := by simpa using h 0 le_rfl
    refine ⟨hV, ?_⟩
    have hr : 0 < Real.sqrt V := Real.sqrt_pos.mpr hV
    have hsq : Real.sqrt V ^ 2 = V := Real.sq_sqrt hV.le
    have h1 := h (Real.sqrt V) hr.le
    have h2 : 0 < Real.sqrt V * (2 * Real.sqrt V + U) := by nlinarith
    by_contra hc
    rw [not_lt] at hc
    nlinarith [mul_nonneg hr.le (by linarith : (0 : ℝ) ≤ -(2 * Real.sqrt V + U))]
  · rintro ⟨hV, hU⟩ x hx
    have hr : 0 < Real.sqrt V := Real.sqrt_pos.mpr hV
    have hsq : Real.sqrt V ^ 2 = V := Real.sq_sqrt hV.le
    rcases hx.eq_or_lt with hx0 | hxpos
    · rw [← hx0]
      simpa using hV
    · nlinarith [sq_nonneg (x - Real.sqrt V),
        mul_pos hxpos (by linarith : (0 : ℝ) < U + 2 * Real.sqrt V)]

/-- **X3.** For real `d > 0`, `0 < q < d`, `ρ ≥ 0`, `h ≥ 0` with `ρ + h > 0`, and
`U = ρ² + d² − 2dh − q²`, `V = d²(ρ + h)² − q²ρ²`: `0 < V` and `−2·√V < U`. -/
theorem X3 (d q h ρ U V : ℝ) (hd : 0 < d) (hq0 : 0 < q) (hqd : q < d) (hρ : 0 ≤ ρ)
    (hh : 0 ≤ h) (hρh : 0 < ρ + h)
    (hU : U = ρ ^ 2 + d ^ 2 - 2 * d * h - q ^ 2)
    (hV : V = d ^ 2 * (ρ + h) ^ 2 - q ^ 2 * ρ ^ 2) :
    0 < V ∧ -2 * Real.sqrt V < U := by
  refine (X2 U V).mp ?_
  intro x hx
  have hdq : 0 < d ^ 2 - q ^ 2 := by nlinarith
  rcases (add_nonneg hx (sq_nonneg ρ)).eq_or_lt with hs | hs
  · -- `x + ρ² = 0`, so `x = 0` and `ρ = 0`; then the value is `d²h²` with `h > 0`.
    have hx0 : x = 0 := by linarith [sq_nonneg ρ]
    have hρsq : ρ ^ 2 = 0 := by linarith [sq_nonneg ρ]
    have hρ0 : ρ = 0 := by simpa using hρsq
    have hhpos : 0 < h := by rw [hρ0] at hρh; linarith
    have e : x ^ 2 + U * x + V = d ^ 2 * h ^ 2 := by rw [hV, hx0, hρ0]; ring
    rw [e]
    positivity
  · -- `x + ρ² > 0`: multiply by it and use an explicit identity.
    have key : (x ^ 2 + U * x + V) * (x + ρ ^ 2)
        = (d * (x + ρ ^ 2) + d * h * ρ) ^ 2 + x * (x + ρ ^ 2 - d * h) ^ 2
          - q ^ 2 * (x + ρ ^ 2) ^ 2 := by
      rw [hU, hV]; ring
    have hpos : 0 < (x ^ 2 + U * x + V) * (x + ρ ^ 2) := by
      rw [key]
      have h1 : 0 ≤ d * h * ρ := mul_nonneg (mul_nonneg hd.le hh) hρ
      have h2 : 0 ≤ x * (x + ρ ^ 2 - d * h) ^ 2 := mul_nonneg hx (sq_nonneg _)
      have h3 : 0 < (d ^ 2 - q ^ 2) * (x + ρ ^ 2) ^ 2 := mul_pos hdq (pow_pos hs 2)
      nlinarith [mul_nonneg (mul_nonneg hd.le hs.le) h1, sq_nonneg (d * h * ρ)]
    by_contra hc
    rw [not_lt] at hc
    nlinarith [mul_nonneg (neg_nonneg.mpr hc) hs.le]

end Dobson.ExactTest

#print axioms Dobson.ExactTest.X1
#print axioms Dobson.ExactTest.X2
#print axioms Dobson.ExactTest.X3
