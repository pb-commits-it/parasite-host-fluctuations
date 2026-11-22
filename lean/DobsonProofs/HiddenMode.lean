import Mathlib

/-!
# A hidden mode (items M1 to M3)

M1 and M2 are each given twice: as an identity for every real `z` (`M1`, `M2`), and for
polynomials in `X` over `ℚ` (`M1_poly`, `M2_poly`), where the constants `1/2` and `3/2` are
`Polynomial.C (1/2)` and `Polynomial.C (3/2)`.

`fullPoly` is the left side of M1 (the full characteristic polynomial of the example) and
`loopPoly` is `(X+1)(X+2) + 1/2 − 1` (the cancelled loop equation).

M3 is stated in an arbitrary commutative ring.
-/

namespace Dobson.HiddenMode

open Polynomial

/-- **M1 (the concrete example), for every real `z`.**
`((z+1)(z+2) + 1/2)(z+1) − (z+1) = (z+1)(z² + 3z + 3/2)`. -/
theorem M1 (z : ℝ) :
    ((z + 1) * (z + 2) + 1 / 2) * (z + 1) - (z + 1) = (z + 1) * (z ^ 2 + 3 * z + 3 / 2) := by
  ring

/-- The left side of M1 as a polynomial over `ℚ`: `((X+1)(X+2) + 1/2)(X+1) − (X+1)`. -/
noncomputable def fullPoly : ℚ[X] := ((X + 1) * (X + 2) + C (1 / 2)) * (X + 1) - (X + 1)

/-- The cancelled loop equation as a polynomial over `ℚ`: `(X+1)(X+2) + 1/2 − 1`. -/
noncomputable def loopPoly : ℚ[X] := (X + 1) * (X + 2) + C (1 / 2) - 1

/-- **M1 (the concrete example), as polynomials over `ℚ`.**
`((X+1)(X+2) + 1/2)(X+1) − (X+1) = (X+1)(X² + 3X + 3/2)`. -/
theorem M1_poly :
    (((X + 1) * (X + 2) + C (1 / 2)) * (X + 1) - (X + 1) : ℚ[X])
      = (X + 1) * (X ^ 2 + 3 * X + C (3 / 2)) := by
  have h : (C (3 / 2) : ℚ[X]) = 1 + C (1 / 2) := by
    rw [← C_1, ← C_add]
    norm_num
  rw [h]
  ring

/-- **M2, for real numbers.** At `z = −1` the left side of M1 is `0`, and
`(z+1)(z+2) + 1/2 − 1` at `z = −1` equals `−1/2`, which is not `0`. -/
theorem M2 :
    (((-1 : ℝ) + 1) * ((-1 : ℝ) + 2) + 1 / 2) * ((-1 : ℝ) + 1) - ((-1 : ℝ) + 1) = 0
      ∧ ((-1 : ℝ) + 1) * ((-1 : ℝ) + 2) + 1 / 2 - 1 = -1 / 2
      ∧ ((-1 : ℝ) + 1) * ((-1 : ℝ) + 2) + 1 / 2 - 1 ≠ 0 := by
  norm_num

/-- **M2, as polynomials over `ℚ`.** `−1` is a root of the full characteristic polynomial
`fullPoly`; the cancelled loop equation `loopPoly` takes the value `−1/2` at `−1`; so `−1` is
not a root of `loopPoly`. -/
theorem M2_poly :
    fullPoly.IsRoot (-1) ∧ loopPoly.eval (-1) = -1 / 2 ∧ ¬ loopPoly.IsRoot (-1) := by
  have h1 : fullPoly.eval (-1) = 0 := by
    simp [fullPoly]
  have h2 : loopPoly.eval (-1) = -1 / 2 := by
    simp [loopPoly]
    norm_num
  refine ⟨h1, h2, ?_⟩
  intro h3
  have h4 : loopPoly.eval (-1) = 0 := h3
  rw [h2] at h4
  norm_num at h4

/-- **M3 (general statement).** In a commutative ring, if `f` divides `D1` and `f` divides
`N2`, then `f` divides `D1·D2 − N1·N2`. -/
theorem M3 {R : Type*} [CommRing R] (f D1 D2 N1 N2 : R) (h1 : f ∣ D1) (h2 : f ∣ N2) :
    f ∣ D1 * D2 - N1 * N2 :=
  dvd_sub (Dvd.dvd.mul_right h1 D2) (Dvd.dvd.mul_left h2 N1)

/-- **M3, the remark in terms of roots.** For polynomials over a commutative ring: if `f`
divides `D1` and `f` divides `N2`, then every root of `f` is a root of `D1·D2 − N1·N2`. -/
theorem M3_root {R : Type*} [CommRing R] (f D1 D2 N1 N2 : R[X]) (h1 : f ∣ D1) (h2 : f ∣ N2)
    (r : R) (hr : f.IsRoot r) :
    (D1 * D2 - N1 * N2).IsRoot r := by
  obtain ⟨g, hg⟩ := M3 f D1 D2 N1 N2 h1 h2
  have h0 : f.eval r = 0 := hr
  show (D1 * D2 - N1 * N2).eval r = 0
  rw [hg, eval_mul, h0, zero_mul]

end Dobson.HiddenMode

#print axioms Dobson.HiddenMode.M1
#print axioms Dobson.HiddenMode.M1_poly
#print axioms Dobson.HiddenMode.M2
#print axioms Dobson.HiddenMode.M2_poly
#print axioms Dobson.HiddenMode.M3
#print axioms Dobson.HiddenMode.M3_root
