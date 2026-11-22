import Mathlib

/-!
# Contraction bounds (items C1 to C3)

Throughout, `a`, `u`, `α`, `ρ`, `ω`, `ξ`, `ε`, `m` are real numbers, regarded as complex numbers
inside the norm, `Complex.I` is the imaginary unit `i`, and `‖z‖` is the modulus of the complex
number `z`.

Note on division by zero. In Lean `x / 0 = 0` by convention. C2 and C3 carry the hypothesis
"not (`ρ = 0` and `ω = 0`)", which is exactly what makes `iω + ρ ≠ 0`, so under the stated
hypotheses every division in the statements is a genuine division. The proofs below do not
need that hypothesis (with Lean's convention the bounds also hold in the excluded case), but
it is kept in the statements so that they say what the mathematics says.
-/

namespace Dobson.Contraction

/-- Helper. For real `K ≥ 0`, `ρ ≥ 0` and real `ω`: `Re[ K / (iω + ρ) ] ≥ 0`. -/
theorem re_div_I_mul_add_nonneg (K ρ ω : ℝ) (hK : 0 ≤ K) (hρ : 0 ≤ ρ) :
    0 ≤ ((K : ℂ) / (Complex.I * (ω : ℂ) + (ρ : ℂ))).re := by
  have hw : Complex.I * (ω : ℂ) + (ρ : ℂ) = (⟨ρ, ω⟩ : ℂ) := by
    apply Complex.ext <;> simp
  rw [hw, Complex.div_re, Complex.normSq_mk]
  simp only [Complex.ofReal_re, Complex.ofReal_im, zero_mul, zero_div, add_zero]
  exact div_nonneg (mul_nonneg hK hρ) (add_nonneg (mul_self_nonneg ρ) (mul_self_nonneg ω))

/-- Helper. If `N ≥ 0` and `A > 0` are real and `D` is a complex number with `Re D ≥ A`,
then `‖N / D‖ ≤ N / A`. -/
theorem norm_ofReal_div_le (N A : ℝ) (D : ℂ) (hN : 0 ≤ N) (hA : 0 < A) (hD : A ≤ D.re) :
    ‖(N : ℂ) / D‖ ≤ N / A := by
  have hnorm : A ≤ ‖D‖ := le_trans hD (Complex.re_le_norm D)
  rw [norm_div, Complex.norm_real, Real.norm_of_nonneg hN]
  exact div_le_div_of_nonneg_left hN hA hnorm

/-- Helper. `Re(iω) = 0` for real `ω`. -/
theorem re_I_mul_ofReal (ω : ℝ) : (Complex.I * (ω : ℂ)).re = 0 := by simp

/-- **C1.** For real `a > 0`, `u > 0` and real `ω`:
`‖ a / (iω + a + u) ‖ ≤ a / (a + u)`. -/
theorem C1 (a u ω : ℝ) (ha : 0 < a) (hu : 0 < u) :
    ‖(a : ℂ) / (Complex.I * (ω : ℂ) + (a : ℂ) + (u : ℂ))‖ ≤ a / (a + u) := by
  refine norm_ofReal_div_le a (a + u) _ ha.le (by positivity) ?_
  rw [Complex.add_re, Complex.add_re, re_I_mul_ofReal, Complex.ofReal_re, Complex.ofReal_re]
  linarith

/-- **C2.** For real `a > 0`, `u > 0`, `α ≥ 0`, `ρ ≥ 0` and real `ω`, with not
(`ρ = 0` and `ω = 0`):
`‖ a / (iω + a + u + αu / (iω + ρ)) ‖ ≤ a / (a + u)`. -/
theorem C2 (a u α ρ ω : ℝ) (ha : 0 < a) (hu : 0 < u) (hα : 0 ≤ α) (hρ : 0 ≤ ρ)
    (_hne : ¬(ρ = 0 ∧ ω = 0)) :
    ‖(a : ℂ) / (Complex.I * (ω : ℂ) + (a : ℂ) + (u : ℂ)
        + (α : ℂ) * (u : ℂ) / (Complex.I * (ω : ℂ) + (ρ : ℂ)))‖ ≤ a / (a + u) := by
  have hK := re_div_I_mul_add_nonneg (α * u) ρ ω (mul_nonneg hα hu.le) hρ
  rw [Complex.ofReal_mul] at hK
  refine norm_ofReal_div_le a (a + u) _ ha.le (by positivity) ?_
  rw [Complex.add_re, Complex.add_re, Complex.add_re, re_I_mul_ofReal, Complex.ofReal_re,
    Complex.ofReal_re]
  linarith

/-- **C3.** For real `ξ > 0`, `α > 0`, `ε ≥ 1`, `m > 0`, `ρ ≥ 0` and real `ω`, with not
(`ρ = 0` and `ω = 0`), and with `u = ξm + αεm²`:
`‖ (ξ + αεm) / (iω + ξ + 2αεm + α(u + αεm²) / (iω + ρ)) ‖ ≤ (ξ + αεm) / (ξ + 2αεm)`. -/
theorem C3 (ξ α ε m ρ ω u : ℝ) (hξ : 0 < ξ) (hα : 0 < α) (hε : 1 ≤ ε) (hm : 0 < m)
    (hρ : 0 ≤ ρ) (_hne : ¬(ρ = 0 ∧ ω = 0)) (hu : u = ξ * m + α * ε * m ^ 2) :
    ‖((ξ : ℂ) + (α : ℂ) * (ε : ℂ) * (m : ℂ))
        / (Complex.I * (ω : ℂ) + (ξ : ℂ) + 2 * (α : ℂ) * (ε : ℂ) * (m : ℂ)
          + (α : ℂ) * ((u : ℂ) + (α : ℂ) * (ε : ℂ) * (m : ℂ) ^ 2)
            / (Complex.I * (ω : ℂ) + (ρ : ℂ)))‖
      ≤ (ξ + α * ε * m) / (ξ + 2 * α * ε * m) := by
  have hε0 : 0 < ε := by linarith
  have hu0 : 0 < u := by rw [hu]; positivity
  have hK := re_div_I_mul_add_nonneg (α * (u + α * ε * m ^ 2)) ρ ω (by positivity) hρ
  have hN : ((ξ : ℂ) + (α : ℂ) * (ε : ℂ) * (m : ℂ)) = ((ξ + α * ε * m : ℝ) : ℂ) := by
    push_cast
    ring
  have hD : (Complex.I * (ω : ℂ) + (ξ : ℂ) + 2 * (α : ℂ) * (ε : ℂ) * (m : ℂ)
          + (α : ℂ) * ((u : ℂ) + (α : ℂ) * (ε : ℂ) * (m : ℂ) ^ 2)
            / (Complex.I * (ω : ℂ) + (ρ : ℂ)))
      = Complex.I * (ω : ℂ) + ((ξ + 2 * α * ε * m : ℝ) : ℂ)
          + ((α * (u + α * ε * m ^ 2) : ℝ) : ℂ) / (Complex.I * (ω : ℂ) + (ρ : ℂ)) := by
    push_cast
    ring
  rw [hN, hD]
  refine norm_ofReal_div_le (ξ + α * ε * m) (ξ + 2 * α * ε * m) _ (by positivity)
    (by positivity) ?_
  rw [Complex.add_re, Complex.add_re, re_I_mul_ofReal, Complex.ofReal_re]
  linarith

end Dobson.Contraction

#print axioms Dobson.Contraction.C1
#print axioms Dobson.Contraction.C2
#print axioms Dobson.Contraction.C3
