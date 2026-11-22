import Mathlib

/-!
# One host, one delay (items D1 to D3)

Abbreviations used in D1 and D2: `q = u + a − s` and `Δ = αu − s(u + a)`.
They are introduced as variables `q`, `Δ` together with the two defining equations as
hypotheses. No sign conditions are assumed in D1 and D2.
-/

namespace Dobson.OneHostOneDelay

/-- **D1 (general form).** In any commutative ring (in particular `ℝ` and `ℂ`), with
`q = u + a − s` and `Δ = αu − s(u + a)`:
`(1 + zτ)·((z − s)(z + u + a) + αu) − a(z − s)
   = τz³ + (1 + τq)z² + ((u − s) + τΔ)z + u(α − s)`. -/
theorem D1 {K : Type*} [CommRing K] (z τ s u a α q Δ : K)
    (hq : q = u + a - s) (hΔ : Δ = α * u - s * (u + a)) :
    (1 + z * τ) * ((z - s) * (z + u + a) + α * u) - a * (z - s)
      = τ * z ^ 3 + (1 + τ * q) * z ^ 2 + ((u - s) + τ * Δ) * z + u * (α - s) := by
  subst hq hΔ
  ring

/-- **D1 for a real variable `z`** and real `τ, s, u, a, α`, with `q = u + a − s` and
`Δ = αu − s(u + a)`:
`(1 + zτ)·((z − s)(z + u + a) + αu) − a(z − s)
   = τz³ + (1 + τq)z² + ((u − s) + τΔ)z + u(α − s)`. -/
theorem D1_real (z τ s u a α q Δ : ℝ)
    (hq : q = u + a - s) (hΔ : Δ = α * u - s * (u + a)) :
    (1 + z * τ) * ((z - s) * (z + u + a) + α * u) - a * (z - s)
      = τ * z ^ 3 + (1 + τ * q) * z ^ 2 + ((u - s) + τ * Δ) * z + u * (α - s) :=
  D1 z τ s u a α q Δ hq hΔ

/-- **D1 for a complex variable `z`** and real `τ, s, u, a, α` (regarded as complex numbers),
with real `q = u + a − s` and `Δ = αu − s(u + a)`:
`(1 + zτ)·((z − s)(z + u + a) + αu) − a(z − s)
   = τz³ + (1 + τq)z² + ((u − s) + τΔ)z + u(α − s)`. -/
theorem D1_complex (z : ℂ) (τ s u a α q Δ : ℝ)
    (hq : q = u + a - s) (hΔ : Δ = α * u - s * (u + a)) :
    (1 + z * (τ : ℂ)) * ((z - (s : ℂ)) * (z + (u : ℂ) + (a : ℂ)) + (α : ℂ) * (u : ℂ))
        - (a : ℂ) * (z - (s : ℂ))
      = (τ : ℂ) * z ^ 3 + (1 + (τ : ℂ) * (q : ℂ)) * z ^ 2
        + (((u : ℂ) - (s : ℂ)) + (τ : ℂ) * (Δ : ℂ)) * z + (u : ℂ) * ((α : ℂ) - (s : ℂ)) := by
  refine D1 z (τ : ℂ) (s : ℂ) (u : ℂ) (a : ℂ) (α : ℂ) (q : ℂ) (Δ : ℂ) ?_ ?_
  · rw [hq]; push_cast; ring
  · rw [hΔ]; push_cast; ring

/-- **D2.** For real `τ, s, u, a, α`, with `q = u + a − s` and `Δ = αu − s(u + a)`:
`(1 + τq)·((u − s) + τΔ) − τ·u(α − s) = qΔτ² + (q(u − s) − a·s)τ + (u − s)`. -/
theorem D2 (τ s u a α q Δ : ℝ)
    (hq : q = u + a - s) (hΔ : Δ = α * u - s * (u + a)) :
    (1 + τ * q) * ((u - s) + τ * Δ) - τ * (u * (α - s))
      = q * Δ * τ ^ 2 + (q * (u - s) - a * s) * τ + (u - s) := by
  subst hq hΔ
  ring

/-- **D3 (Routh–Hurwitz for a cubic, one direction).** Let `c3, c2, c1, c0` be real and
positive, with `c2·c1 > c3·c0`. Then every complex root `z` of
`c3 z³ + c2 z² + c1 z + c0 = 0` has negative real part. -/
theorem D3 (c3 c2 c1 c0 : ℝ) (h3 : 0 < c3) (h2 : 0 < c2) (h1 : 0 < c1) (h0 : 0 < c0)
    (hRH : c2 * c1 > c3 * c0) (z : ℂ)
    (hz : (c3 : ℂ) * z ^ 3 + (c2 : ℂ) * z ^ 2 + (c1 : ℂ) * z + (c0 : ℂ) = 0) :
    z.re < 0 := by
  obtain ⟨x, y⟩ := z
  change x < 0
  by_contra hcon
  have hx : 0 ≤ x := not_lt.mp hcon
  -- real and imaginary parts of the cubic at `z = x + i y`
  have hexp : (c3 : ℂ) * (⟨x, y⟩ : ℂ) ^ 3 + (c2 : ℂ) * (⟨x, y⟩ : ℂ) ^ 2
        + (c1 : ℂ) * (⟨x, y⟩ : ℂ) + (c0 : ℂ)
      = (⟨c3 * (x ^ 3 - 3 * x * y ^ 2) + c2 * (x ^ 2 - y ^ 2) + c1 * x + c0,
          y * (c3 * (3 * x ^ 2 - y ^ 2) + 2 * c2 * x + c1)⟩ : ℂ) := by
    rw [Complex.mk_eq_add_mul_I, Complex.mk_eq_add_mul_I]
    push_cast
    linear_combination
      (3 * (c3 : ℂ) * x * y ^ 2 + (c3 : ℂ) * y ^ 3 * Complex.I + (c2 : ℂ) * y ^ 2)
        * Complex.I_sq
  rw [hexp, Complex.ext_iff] at hz
  simp only [Complex.zero_re, Complex.zero_im] at hz
  obtain ⟨hre, him⟩ := hz
  rcases eq_or_ne y 0 with hy | hy
  · -- real root `x ≥ 0`: the cubic is positive there
    rw [hy] at hre
    have hpos : 0 < c3 * x ^ 3 + c2 * x ^ 2 + c1 * x + c0 := by positivity
    have hzero : c3 * x ^ 3 + c2 * x ^ 2 + c1 * x + c0 = 0 := by linear_combination hre
    linarith
  · -- non-real root: eliminate `y²` between the two equations
    have hQ : c3 * (3 * x ^ 2 - y ^ 2) + 2 * c2 * x + c1 = 0 := by
      rcases mul_eq_zero.mp him with h | h
      · exact absurd h hy
      · exact h
    have hid : -(8 * c3 ^ 2 * x ^ 3 + 8 * c2 * c3 * x ^ 2 + (2 * c1 * c3 + 2 * c2 ^ 2) * x)
        + (c0 * c3 - c1 * c2) = 0 := by
      linear_combination c3 * hre - (3 * c3 * x + c2) * hQ
    have hpos : 0 ≤ 8 * c3 ^ 2 * x ^ 3 + 8 * c2 * c3 * x ^ 2
        + (2 * c1 * c3 + 2 * c2 ^ 2) * x := by positivity
    nlinarith

end Dobson.OneHostOneDelay

#print axioms Dobson.OneHostOneDelay.D1
#print axioms Dobson.OneHostOneDelay.D1_real
#print axioms Dobson.OneHostOneDelay.D1_complex
#print axioms Dobson.OneHostOneDelay.D2
#print axioms Dobson.OneHostOneDelay.D3
