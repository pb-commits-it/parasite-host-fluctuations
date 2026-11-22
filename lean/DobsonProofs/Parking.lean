import Mathlib

/-!
# Parking (items P1 to P4)

Data: reals `a > 0`, `b > 0`, `θ ≥ 0` and a real `L`.

2026-11-16 auditfix2: P1–P3 include zero memory. P4 retains `θ > 0`: at
`θ = L = 0` the difference is zero, so its strict conclusion would be false.
-/

namespace Dobson.Parking

/-- `E1 = (2aL − 1) / (1 + 2θ)`. -/
noncomputable def E1 (a θ L : ℝ) : ℝ :=
  (2 * a * L - 1) / (1 + 2 * θ)

/-- `G2 = (1 + 2θ) / (2(a + b)) − θ² / (θ(a + b) + ab)`. -/
noncomputable def G2 (a b θ : ℝ) : ℝ :=
  (1 + 2 * θ) / (2 * (a + b)) - θ ^ 2 / (θ * (a + b) + a * b)

/-- `E2 = (2(a + b)L − 2(a + b)·G2) / (1 + 2θ)`. -/
noncomputable def E2 (a b θ L : ℝ) : ℝ :=
  (2 * (a + b) * L - 2 * (a + b) * G2 a b θ) / (1 + 2 * θ)

/-- **P1.** For `a > 0`, `b > 0`, `θ ≥ 0` and real `L`:
`E2 − E1 = (2b / (1 + 2θ)) · (L − aθ / (aθ + b(a + θ)))`. -/
theorem E2_sub_E1 (a b θ L : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 ≤ θ) :
    E2 a b θ L - E1 a θ L
      = (2 * b / (1 + 2 * θ)) * (L - a * θ / (a * θ + b * (a + θ))) := by
  unfold E2 E1 G2
  have h1 : (1 + 2 * θ) ≠ 0 := by positivity
  have h2 : a + b ≠ 0 := by positivity
  have h3 : θ * (a + b) + a * b ≠ 0 := by positivity
  have h4 : a * θ + b * (a + θ) ≠ 0 := by positivity
  field_simp
  ring

/-- **P2.** For `a > 0`, `b > 0`, `θ ≥ 0` and `0 < L < 1`:
`E2 > E1` if and only if `b > (aθ / (a + θ)) · (1 − L) / L`.
(The hypothesis `L < 1` is part of the statement but is not needed by the proof.) -/
theorem E2_gt_E1_iff (a b θ L : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 ≤ θ)
    (hL0 : 0 < L) (_hL1 : L < 1) :
    E2 a b θ L > E1 a θ L ↔ b > (a * θ / (a + θ)) * (1 - L) / L := by
  have hD : 0 < a * θ + b * (a + θ) := by positivity
  have haθ : 0 < a + θ := by positivity
  have hkey := E2_sub_E1 a b θ L ha hb hθ
  have hpos : 0 < 2 * b / (1 + 2 * θ) := by positivity
  have h1 : E2 a b θ L > E1 a θ L ↔ 0 < L - a * θ / (a * θ + b * (a + θ)) := by
    rw [gt_iff_lt, ← sub_pos, hkey]
    exact mul_pos_iff_of_pos_left hpos
  have h2 : (a * θ / (a + θ)) * (1 - L) / L = a * θ * (1 - L) / ((a + θ) * L) := by
    field_simp
  rw [h1, sub_pos, div_lt_iff₀ hD, gt_iff_lt, h2, div_lt_iff₀ (by positivity)]
  constructor <;> intro h <;> nlinarith

/-- **P3.** For `a > 0`, `b > 0`, `θ ≥ 0` and `L ≥ 1`: `E2 > E1`. -/
theorem E2_gt_E1_of_one_le (a b θ L : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 ≤ θ)
    (hL : L ≥ 1) :
    E2 a b θ L > E1 a θ L := by
  have hD : 0 < a * θ + b * (a + θ) := by positivity
  have hfrac : a * θ / (a * θ + b * (a + θ)) < 1 := by
    rw [div_lt_one hD]
    nlinarith [mul_pos hb (add_pos_of_pos_of_nonneg ha hθ)]
  have hpos : 0 < 2 * b / (1 + 2 * θ) := by positivity
  have hkey := E2_sub_E1 a b θ L ha hb hθ
  have h : 0 < E2 a b θ L - E1 a θ L := by
    rw [hkey]
    exact mul_pos hpos (by linarith)
  linarith

/-- **P4.** For `a > 0`, `b > 0`, `θ > 0` and `L ≤ 0`: `E2 < E1`. -/
theorem E2_lt_E1_of_nonpos (a b θ L : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 < θ)
    (hL : L ≤ 0) :
    E2 a b θ L < E1 a θ L := by
  have hfrac : 0 < a * θ / (a * θ + b * (a + θ)) := by positivity
  have hpos : 0 < 2 * b / (1 + 2 * θ) := by positivity
  have hkey := E2_sub_E1 a b θ L ha hb hθ.le
  have h : E2 a b θ L - E1 a θ L < 0 := by
    rw [hkey]
    exact mul_neg_of_pos_of_neg hpos (by linarith)
  linarith

end Dobson.Parking

#print axioms Dobson.Parking.E2_sub_E1
#print axioms Dobson.Parking.E2_gt_E1_iff
#print axioms Dobson.Parking.E2_gt_E1_of_one_le
#print axioms Dobson.Parking.E2_lt_E1_of_nonpos
