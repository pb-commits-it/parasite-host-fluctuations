import Mathlib

/-!
# Hosts that cannot amplify, on the whole closed right half-plane (items H1 to H6)

The first batch (`Contraction.lean`, `OneHostOneDelay.lean`) proved these bounds on the
imaginary axis `z = iω`. Here `z` is any complex number with `Re z ≥ 0`.

Throughout, `a`, `u`, `α`, `ρ`, `ξ`, `ε`, `m`, `κ` are real numbers, regarded as complex numbers
inside the norm, and `‖w‖` is the modulus of the complex number `w`.

Note on division by zero. In Lean `x / 0 = 0` by convention. H3 and H4 carry the hypothesis
`z + ρ ≠ 0`, which makes the inner division a genuine division. The proofs below do not need
that hypothesis (with Lean's convention the bounds also hold when `z + ρ = 0`), but it is kept
in the statements so that they say what the mathematics says. The outer denominators have real
part at least `a > 0` (respectively `a + u`, `ξ + 2αεm`), so they are never zero.
-/

namespace Dobson.HalfPlane

/-- Helper. If `N ≥ 0` and `A > 0` are real and `D` is a complex number with `Re D ≥ A`,
then `‖N / D‖ ≤ N / A`. -/
theorem norm_ofReal_div_le (N A : ℝ) (D : ℂ) (hN : 0 ≤ N) (hA : 0 < A) (hD : A ≤ D.re) :
    ‖(N : ℂ) / D‖ ≤ N / A := by
  have hnorm : A ≤ ‖D‖ := le_trans hD (Complex.re_le_norm D)
  rw [norm_div, Complex.norm_real, Real.norm_of_nonneg hN]
  exact div_le_div_of_nonneg_left hN hA hnorm

/-- Helper. For real `K ≥ 0` and a complex number `w` with `Re w ≥ 0`: `Re (K / w) ≥ 0`. -/
theorem re_ofReal_div_nonneg (K : ℝ) (w : ℂ) (hK : 0 ≤ K) (hw : 0 ≤ w.re) :
    0 ≤ ((K : ℂ) / w).re := by
  rw [Complex.div_re]
  simp only [Complex.ofReal_re, Complex.ofReal_im, zero_mul, zero_div, add_zero]
  exact div_nonneg (mul_nonneg hK hw) (Complex.normSq_nonneg w)

/-- **H1.** For real `a > 0` and complex `z` with `Re z ≥ 0`: `‖ a / (z + a) ‖ ≤ 1`. -/
theorem H1 (a : ℝ) (z : ℂ) (ha : 0 < a) (hz : 0 ≤ z.re) :
    ‖(a : ℂ) / (z + (a : ℂ))‖ ≤ 1 := by
  have h := norm_ofReal_div_le a a (z + (a : ℂ)) ha.le ha
    (by rw [Complex.add_re, Complex.ofReal_re]; linarith)
  rwa [div_self ha.ne'] at h

/-- **H2.** For real `a > 0`, `u > 0` and complex `z` with `Re z ≥ 0`:
`‖ a / (z + a + u) ‖ ≤ a / (a + u)`. -/
theorem H2 (a u : ℝ) (z : ℂ) (ha : 0 < a) (hu : 0 < u) (hz : 0 ≤ z.re) :
    ‖(a : ℂ) / (z + (a : ℂ) + (u : ℂ))‖ ≤ a / (a + u) := by
  refine norm_ofReal_div_le a (a + u) _ ha.le (by positivity) ?_
  rw [Complex.add_re, Complex.add_re, Complex.ofReal_re, Complex.ofReal_re]
  linarith

/-- **H3.** For real `a > 0`, `u > 0`, `α ≥ 0`, `ρ ≥ 0` and complex `z` with `Re z ≥ 0` and
`z + ρ ≠ 0`: `‖ a / (z + a + u + αu / (z + ρ)) ‖ ≤ a / (a + u)`. -/
theorem H3 (a u α ρ : ℝ) (z : ℂ) (ha : 0 < a) (hu : 0 < u) (hα : 0 ≤ α) (hρ : 0 ≤ ρ)
    (hz : 0 ≤ z.re) (_hne : z + (ρ : ℂ) ≠ 0) :
    ‖(a : ℂ) / (z + (a : ℂ) + (u : ℂ) + (α : ℂ) * (u : ℂ) / (z + (ρ : ℂ)))‖
      ≤ a / (a + u) := by
  have hK := re_ofReal_div_nonneg (α * u) (z + (ρ : ℂ)) (mul_nonneg hα hu.le)
    (by rw [Complex.add_re, Complex.ofReal_re]; linarith)
  rw [Complex.ofReal_mul] at hK
  refine norm_ofReal_div_le a (a + u) _ ha.le (by positivity) ?_
  rw [Complex.add_re, Complex.add_re, Complex.add_re, Complex.ofReal_re, Complex.ofReal_re]
  linarith

/-- **H4.** For real `ξ > 0`, `α > 0`, `ε ≥ 1`, `m > 0`, `ρ ≥ 0`, with `u = ξm + αεm²`, and
complex `z` with `Re z ≥ 0` and `z + ρ ≠ 0`:
`‖ (ξ + αεm) / (z + ξ + 2αεm + α(u + αεm²) / (z + ρ)) ‖ ≤ (ξ + αεm) / (ξ + 2αεm)`. -/
theorem H4 (ξ α ε m ρ u : ℝ) (z : ℂ) (hξ : 0 < ξ) (hα : 0 < α) (hε : 1 ≤ ε) (hm : 0 < m)
    (hρ : 0 ≤ ρ) (hu : u = ξ * m + α * ε * m ^ 2) (hz : 0 ≤ z.re)
    (_hne : z + (ρ : ℂ) ≠ 0) :
    ‖((ξ : ℂ) + (α : ℂ) * (ε : ℂ) * (m : ℂ))
        / (z + (ξ : ℂ) + 2 * (α : ℂ) * (ε : ℂ) * (m : ℂ)
          + (α : ℂ) * ((u : ℂ) + (α : ℂ) * (ε : ℂ) * (m : ℂ) ^ 2) / (z + (ρ : ℂ)))‖
      ≤ (ξ + α * ε * m) / (ξ + 2 * α * ε * m) := by
  have hε0 : 0 < ε := by linarith
  have hu0 : 0 < u := by rw [hu]; positivity
  have hK := re_ofReal_div_nonneg (α * (u + α * ε * m ^ 2)) (z + (ρ : ℂ)) (by positivity)
    (by rw [Complex.add_re, Complex.ofReal_re]; linarith)
  have hN : ((ξ : ℂ) + (α : ℂ) * (ε : ℂ) * (m : ℂ)) = ((ξ + α * ε * m : ℝ) : ℂ) := by
    push_cast
    ring
  have hD : (z + (ξ : ℂ) + 2 * (α : ℂ) * (ε : ℂ) * (m : ℂ)
          + (α : ℂ) * ((u : ℂ) + (α : ℂ) * (ε : ℂ) * (m : ℂ) ^ 2) / (z + (ρ : ℂ)))
      = z + ((ξ + 2 * α * ε * m : ℝ) : ℂ)
          + ((α * (u + α * ε * m ^ 2) : ℝ) : ℂ) / (z + (ρ : ℂ)) := by
    push_cast
    ring
  rw [hN, hD]
  refine norm_ofReal_div_le (ξ + α * ε * m) (ξ + 2 * α * ε * m) _ (by positivity)
    (by positivity) ?_
  rw [Complex.add_re, Complex.add_re, Complex.ofReal_re]
  linarith

/-- **H5.** For real `κ` with `0 ≤ κ ≤ 2`, real `a > 0` and complex `z` with `Re z ≥ 0`:
`‖ (1 − κ)·a / (z + a) ‖ ≤ |1 − κ|` and `|1 − κ| ≤ 1`. -/
theorem H5 (κ a : ℝ) (z : ℂ) (hκ0 : 0 ≤ κ) (hκ2 : κ ≤ 2) (ha : 0 < a) (hz : 0 ≤ z.re) :
    ‖((1 : ℂ) - (κ : ℂ)) * (a : ℂ) / (z + (a : ℂ))‖ ≤ |1 - κ| ∧ |1 - κ| ≤ 1 := by
  constructor
  · have h1 := H1 a z ha hz
    have hc : ((1 : ℂ) - (κ : ℂ)) = ((1 - κ : ℝ) : ℂ) := by push_cast; rfl
    rw [mul_div_assoc, norm_mul, hc, Complex.norm_real, Real.norm_eq_abs]
    calc |1 - κ| * ‖(a : ℂ) / (z + (a : ℂ))‖ ≤ |1 - κ| * 1 :=
          mul_le_mul_of_nonneg_left h1 (abs_nonneg _)
      _ = |1 - κ| := mul_one _
  · rw [abs_le]
    constructor <;> linarith

/-- The inequality behind H6. For a finite family of complex numbers `e k` and reals `γ k`
with `‖e k‖ ≤ γ k` and `0 ≤ γ k ≤ 1` for every `k`, and `γ j < 1` for at least one `j`:
`‖∏ e k‖ ≤ ∏ γ k` and `∏ γ k < 1`. -/
theorem norm_prod_le_prod_and_prod_lt_one {ι : Type*} (s : Finset ι) (e : ι → ℂ) (γ : ι → ℝ)
    (he : ∀ k ∈ s, ‖e k‖ ≤ γ k) (hγ0 : ∀ k ∈ s, 0 ≤ γ k) (hγ1 : ∀ k ∈ s, γ k ≤ 1)
    (hj : ∃ j ∈ s, γ j < 1) :
    ‖∏ k ∈ s, e k‖ ≤ ∏ k ∈ s, γ k ∧ ∏ k ∈ s, γ k < 1 := by
  classical
  constructor
  · rw [norm_prod]
    exact Finset.prod_le_prod₀ (fun k _ => norm_nonneg _) he
  · obtain ⟨j, hjs, hj1⟩ := hj
    rw [← Finset.mul_prod_erase s γ hjs]
    have hrest : ∏ k ∈ s.erase j, γ k ≤ 1 :=
      Finset.prod_le_one₀ (fun k hk => hγ0 k (Finset.mem_of_mem_erase hk))
        (fun k hk => hγ1 k (Finset.mem_of_mem_erase hk))
    nlinarith [mul_nonneg (hγ0 j hjs) (sub_nonneg.mpr hrest)]

/-- **H6.** (The product cannot equal one.) For a finite family of complex numbers `e k` and
reals `γ k` (`k` in a finite set `s`) with `‖e k‖ ≤ γ k` and `0 ≤ γ k ≤ 1` for every `k`, and
`γ j < 1` for at least one `j`: `∏ e k ≠ 1`. -/
theorem H6 {ι : Type*} (s : Finset ι) (e : ι → ℂ) (γ : ι → ℝ)
    (he : ∀ k ∈ s, ‖e k‖ ≤ γ k) (hγ0 : ∀ k ∈ s, 0 ≤ γ k) (hγ1 : ∀ k ∈ s, γ k ≤ 1)
    (hj : ∃ j ∈ s, γ j < 1) :
    ∏ k ∈ s, e k ≠ 1 := by
  obtain ⟨h1, h2⟩ := norm_prod_le_prod_and_prod_lt_one s e γ he hγ0 hγ1 hj
  intro h
  rw [h, norm_one] at h1
  linarith

end Dobson.HalfPlane

#print axioms Dobson.HalfPlane.H1
#print axioms Dobson.HalfPlane.H2
#print axioms Dobson.HalfPlane.H3
#print axioms Dobson.HalfPlane.H4
#print axioms Dobson.HalfPlane.H5
#print axioms Dobson.HalfPlane.norm_prod_le_prod_and_prod_lt_one
#print axioms Dobson.HalfPlane.H6
