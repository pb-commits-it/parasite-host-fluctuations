import Mathlib

/-!
# The band (items B1 to B4)

Data: a real `θ > 0`, a finite nonempty family of reals `τ i > 0`, and a real `v ≥ 0`
(standing for `σ²`).

Encoding of the finite family: an arbitrary index type `ι`, a finite set `s : Finset ι` of
indices with `s.Nonempty`, and a function `τ : ι → ℝ` that is assumed positive on `s`.
All sums and products run over `i ∈ s`.
-/

namespace Dobson.Band

open Finset

/-- **B1.** For a finite index set `s` and real numbers `x i ≥ 0` (for every `i` in `s`):
`1 + ∑_{i ∈ s} x i ≤ ∏_{i ∈ s} (1 + x i)`. -/
theorem one_add_sum_le_prod_one_add {ι : Type*} (s : Finset ι) (x : ι → ℝ)
    (hx : ∀ i ∈ s, 0 ≤ x i) :
    1 + ∑ i ∈ s, x i ≤ ∏ i ∈ s, (1 + x i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    have hxa : 0 ≤ x a := hx a (mem_insert_self a s)
    have hxs : ∀ i ∈ s, 0 ≤ x i := fun i hi => hx i (mem_insert_of_mem hi)
    have ih' := ih hxs
    have hsum : 0 ≤ ∑ i ∈ s, x i := sum_nonneg hxs
    rw [sum_insert ha, prod_insert ha]
    nlinarith [mul_le_mul_of_nonneg_left ih' (by linarith : (0 : ℝ) ≤ 1 + x a),
      mul_nonneg hxa hsum]

/-- `𝒯 = ∑_{i ∈ s} τ i`. -/
noncomputable def total {ι : Type*} (s : Finset ι) (τ : ι → ℝ) : ℝ :=
  ∑ i ∈ s, τ i

/-- `P = ∏_{i ∈ s} (1 + τ i / θ)`. -/
noncomputable def P {ι : Type*} (θ : ℝ) (s : Finset ι) (τ : ι → ℝ) : ℝ :=
  ∏ i ∈ s, (1 + τ i / θ)

/-- `R = 1 / P`. -/
noncomputable def R {ι : Type*} (θ : ℝ) (s : Finset ι) (τ : ι → ℝ) : ℝ :=
  1 / P θ s τ

/-- `Γ = v · [ (1 + 2θ)/(2𝒯) − R/(1 − R) ]`. -/
noncomputable def Gamma {ι : Type*} (v θ : ℝ) (s : Finset ι) (τ : ι → ℝ) : ℝ :=
  v * ((1 + 2 * θ) / (2 * total s τ) - R θ s τ / (1 - R θ s τ))

/-- **B2.** Let `θ > 0`, and let `τ i > 0` for `i` in a finite nonempty set `s`.
With `𝒯 = ∑ τ i`, `P = ∏ (1 + τ i / θ)` and `R = 1 / P`:
`0 < R`, `R < 1`, and `R / (1 − R) ≤ θ / 𝒯`. -/
theorem R_pos_and_R_lt_one_and_ratio_le {ι : Type*} (θ : ℝ) (hθ : 0 < θ)
    (s : Finset ι) (hs : s.Nonempty) (τ : ι → ℝ) (hτ : ∀ i ∈ s, 0 < τ i) :
    0 < R θ s τ ∧ R θ s τ < 1 ∧ R θ s τ / (1 - R θ s τ) ≤ θ / total s τ := by
  have hT : 0 < total s τ := sum_pos hτ hs
  have hB1 : 1 + ∑ i ∈ s, τ i / θ ≤ P θ s τ :=
    one_add_sum_le_prod_one_add s (fun i => τ i / θ) (fun i hi => (div_pos (hτ i hi) hθ).le)
  have hsum : ∑ i ∈ s, τ i / θ = total s τ / θ := by rw [total, Finset.sum_div]
  rw [hsum] at hB1
  have hTθ : 0 < total s τ / θ := div_pos hT hθ
  have hP1 : 1 < P θ s τ := by linarith
  have hP0 : 0 < P θ s τ := by linarith
  have hR : R θ s τ = 1 / P θ s τ := rfl
  have hRpos : 0 < R θ s τ := by rw [hR]; exact one_div_pos.mpr hP0
  have hRlt : R θ s τ < 1 := by rw [hR, div_lt_one hP0]; exact hP1
  refine ⟨hRpos, hRlt, ?_⟩
  have hPne : P θ s τ ≠ 0 := hP0.ne'
  have hP1ne : P θ s τ - 1 ≠ 0 := by linarith [hP1.ne']
  have hratio : R θ s τ / (1 - R θ s τ) = 1 / (P θ s τ - 1) := by
    rw [hR]
    have h1 : 1 - 1 / P θ s τ = (P θ s τ - 1) / P θ s τ := by field_simp
    rw [h1, div_div_div_cancel_right₀ hPne]
  rw [hratio, div_le_div_iff₀ (by linarith) hT]
  have h2 : θ * (total s τ / θ) = total s τ := by field_simp
  have h3 : θ * (total s τ / θ) ≤ θ * (P θ s τ - 1) :=
    mul_le_mul_of_nonneg_left (by linarith) hθ.le
  linarith

/-- **B3 (the band).** Let `θ > 0`, let `τ i > 0` for `i` in a finite nonempty set `s`,
and let `v ≥ 0`. With `𝒯 = ∑ τ i`, `R = 1 / ∏ (1 + τ i / θ)` and
`Γ = v · [ (1 + 2θ)/(2𝒯) − R/(1 − R) ]`:
`v / (2𝒯) ≤ Γ` and `Γ ≤ v (1 + 2θ) / (2𝒯)`. -/
theorem band {ι : Type*} (θ : ℝ) (hθ : 0 < θ)
    (s : Finset ι) (hs : s.Nonempty) (τ : ι → ℝ) (hτ : ∀ i ∈ s, 0 < τ i)
    (v : ℝ) (hv : 0 ≤ v) :
    v / (2 * total s τ) ≤ Gamma v θ s τ ∧
      Gamma v θ s τ ≤ v * (1 + 2 * θ) / (2 * total s τ) := by
  obtain ⟨hR0, hR1, hle⟩ := R_pos_and_R_lt_one_and_ratio_le θ hθ s hs τ hτ
  have hT : 0 < total s τ := sum_pos hτ hs
  have hnn : 0 ≤ R θ s τ / (1 - R θ s τ) := div_nonneg hR0.le (by linarith)
  unfold Gamma
  constructor
  · have h1 : (1 + 2 * θ) / (2 * total s τ) - θ / total s τ = 1 / (2 * total s τ) := by
      field_simp
      ring
    have h2 : 1 / (2 * total s τ)
        ≤ (1 + 2 * θ) / (2 * total s τ) - R θ s τ / (1 - R θ s τ) := by linarith
    calc v / (2 * total s τ) = v * (1 / (2 * total s τ)) := by ring
      _ ≤ v * ((1 + 2 * θ) / (2 * total s τ) - R θ s τ / (1 - R θ s τ)) :=
        mul_le_mul_of_nonneg_left h2 hv
  · calc v * ((1 + 2 * θ) / (2 * total s τ) - R θ s τ / (1 - R θ s τ))
        ≤ v * ((1 + 2 * θ) / (2 * total s τ)) :=
          mul_le_mul_of_nonneg_left (by linarith) hv
      _ = v * (1 + 2 * θ) / (2 * total s τ) := by ring

/-- **B4, first part (equality case, one direction).** If the family has exactly one member
(the index set is the singleton `{i}`), with `θ > 0` and `τ i > 0`, then
`R / (1 − R) = θ / 𝒯`. -/
theorem singleton_ratio_eq {ι : Type*} (θ : ℝ) (hθ : 0 < θ) (i : ι) (τ : ι → ℝ)
    (hτ : 0 < τ i) :
    R θ {i} τ / (1 - R θ {i} τ) = θ / total {i} τ := by
  have hθ' : θ ≠ 0 := hθ.ne'
  have hτ' : τ i ≠ 0 := hτ.ne'
  have hsum : θ + τ i ≠ 0 := (add_pos hθ hτ).ne'
  have hR : R θ {i} τ = θ / (θ + τ i) := by
    simp only [R, P, prod_singleton]
    field_simp
  have h1R : 1 - R θ {i} τ = τ i / (θ + τ i) := by
    rw [hR]
    field_simp
    ring
  have hT : total {i} τ = τ i := by simp only [total, sum_singleton]
  rw [h1R, hR, hT, div_div_div_cancel_right₀ hsum]

/-- **B4, second part.** If the family has exactly one member (index set `{i}`), with
`θ > 0`, `τ i > 0` and `v ≥ 0`, then `Γ = v / (2𝒯)`: the lower end of the band is attained.
(The hypothesis `v ≥ 0` is part of the data but is not needed for this equality.) -/
theorem singleton_Gamma_eq {ι : Type*} (θ : ℝ) (hθ : 0 < θ) (i : ι) (τ : ι → ℝ)
    (hτ : 0 < τ i) (v : ℝ) (_hv : 0 ≤ v) :
    Gamma v θ {i} τ = v / (2 * total {i} τ) := by
  have hT : total {i} τ = τ i := by simp only [total, sum_singleton]
  have hτ' : τ i ≠ 0 := hτ.ne'
  unfold Gamma
  rw [singleton_ratio_eq θ hθ i τ hτ, hT]
  field_simp
  ring

end Dobson.Band

#print axioms Dobson.Band.one_add_sum_le_prod_one_add
#print axioms Dobson.Band.R_pos_and_R_lt_one_and_ratio_le
#print axioms Dobson.Band.band
#print axioms Dobson.Band.singleton_ratio_eq
#print axioms Dobson.Band.singleton_Gamma_eq
