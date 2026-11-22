import Mathlib

/-!
# When one fluctuating transfer is a gain (items G1 to G4)

All variables are real. In G3 the derived quantity `θ = φ/(1 − φ)` is a variable carrying
that equation as a hypothesis. In G4 the powers are real powers (`Real.rpow`).
-/

namespace Dobson.GainCondition

/-- **G1 (when one fluctuating transfer is a gain).** For `θ ≥ 0`, `T > 0`, `R < 1`:
`(1 + 2θ)/(2T) − R/(1 − R) < 0 ↔ R > (1 + 2θ)/(1 + 2θ + 2T)`.

Audit correction, 2026-11-16 (auditfix1): removed the unused `0 < R` premise, which
excluded the model's zero-memory boundary `R = 0`. The same comparison proof applies
for every `R < 1`. -/
theorem G1 (θ T R : ℝ) (hθ : θ ≥ 0) (hT : T > 0) (hR1 : R < 1) :
    (1 + 2 * θ) / (2 * T) - R / (1 - R) < 0 ↔ R > (1 + 2 * θ) / (1 + 2 * θ + 2 * T) := by
  have h1 : 0 < 1 - R := by linarith
  have h2 : 0 < 2 * T := by linarith
  have h3 : 0 < 1 + 2 * θ + 2 * T := by linarith
  rw [sub_neg, div_lt_div_iff₀ h2 h1, gt_iff_lt, div_lt_iff₀ h3]
  constructor <;> intro h <;> linarith

/-- **G2 (the band excludes it for a ring).** If `0 ≤ θ`, `T > 0`, `0 ≤ R < 1` and
`R/(1 − R) ≤ θ/T`, then `(1 + 2θ)/(2T) − R/(1 − R) ≥ 1/(2T)`.
(The hypotheses `0 ≤ θ`, `0 ≤ R` and `R < 1` are part of the statement but are not needed by
the proof; `T > 0` is used only through `T ≠ 0`.) -/
theorem G2 (θ T R : ℝ) (_hθ : 0 ≤ θ) (hT : T > 0) (_hR0 : 0 ≤ R) (_hR1 : R < 1)
    (h : R / (1 - R) ≤ θ / T) :
    (1 + 2 * θ) / (2 * T) - R / (1 - R) ≥ 1 / (2 * T) := by
  -- `0 ≤ θ`, `0 ≤ R` and `R < 1` are not needed. They are cleared here so that no later
  -- step can use them.
  clear _hθ _hR0 _hR1
  have hT' : T ≠ 0 := hT.ne'
  have e : (1 + 2 * θ) / (2 * T) - θ / T = 1 / (2 * T) := by
    field_simp
    ring
  linarith

/-- **G2, the remark "so it is > 0".** Under the hypotheses of G2,
`(1 + 2θ)/(2T) − R/(1 − R) > 0`. -/
theorem G2_pos (θ T R : ℝ) (hθ : 0 ≤ θ) (hT : T > 0) (hR0 : 0 ≤ R) (hR1 : R < 1)
    (h : R / (1 - R) ≤ θ / T) :
    (1 + 2 * θ) / (2 * T) - R / (1 - R) > 0 := by
  have h1 := G2 θ T R hθ hT hR0 hR1 h
  have h2 : 0 < 1 / (2 * T) := by positivity
  linarith

/-- **G3 (the most over-dispersed life cycle).** For `0 < φ < 1`, `θ = φ/(1 − φ)`, `T > 0`:
`φ > (1 + 2θ)/(1 + 2θ + 2T) ↔ T > (1 + φ)/(2φ)`. -/
theorem G3 (φ θ T : ℝ) (hφ0 : 0 < φ) (hφ1 : φ < 1) (hθ : θ = φ / (1 - φ)) (hT : T > 0) :
    φ > (1 + 2 * θ) / (1 + 2 * θ + 2 * T) ↔ T > (1 + φ) / (2 * φ) := by
  have h1 : 0 < 1 - φ := by linarith
  have h1' : 1 - φ ≠ 0 := h1.ne'
  have hθ0 : 0 < θ := by
    rw [hθ]
    exact div_pos hφ0 h1
  have h3 : 0 < 1 + 2 * θ + 2 * T := by linarith
  have key : (1 + 2 * θ) * (1 - φ) = 1 + φ := by
    rw [hθ]
    field_simp
    ring
  rw [gt_iff_lt, div_lt_iff₀ h3, gt_iff_lt, div_lt_iff₀ (by linarith : (0 : ℝ) < 2 * φ)]
  constructor <;> intro h <;> linarith

/-- **G4 (Jensen side).** For `0 < φ < 1`, `0 ≤ w ≤ 1` and real `a`, `b`:
`φ^(w·a + (1−w)·b) ≤ w·φ^a + (1−w)·φ^b` (real powers; convexity of `x ↦ φ^x`).
(The hypothesis `φ < 1` is part of the statement but is not needed by the proof: the
inequality holds for every `φ > 0`.) -/
theorem G4 (φ w a b : ℝ) (hφ0 : 0 < φ) (_hφ1 : φ < 1) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) :
    φ ^ (w * a + (1 - w) * b) ≤ w * φ ^ a + (1 - w) * φ ^ b := by
  -- `φ < 1` is not needed. It is cleared here so that no later step can use it.
  clear _hφ1
  have h := Real.geom_mean_le_arith_mean2_weighted hw0 (by linarith : (0 : ℝ) ≤ 1 - w)
    (Real.rpow_nonneg hφ0.le a) (Real.rpow_nonneg hφ0.le b) (by ring)
  rw [← Real.rpow_mul hφ0.le, ← Real.rpow_mul hφ0.le, ← Real.rpow_add hφ0] at h
  have e : w * a + (1 - w) * b = a * w + b * (1 - w) := by ring
  rw [e]
  exact h

end Dobson.GainCondition

#print axioms Dobson.GainCondition.G1
#print axioms Dobson.GainCondition.G2
#print axioms Dobson.GainCondition.G2_pos
#print axioms Dobson.GainCondition.G3
#print axioms Dobson.GainCondition.G4
