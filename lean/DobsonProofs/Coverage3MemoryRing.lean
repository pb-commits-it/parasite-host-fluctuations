import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-!
# Coverage batch 3: distinct ring comparisons
Core of `prop:exact-mem-ring`: exact two-stage comparison and both existence
quantifiers, the critical quadratic and the three-stage limiting peak algebra.
Identification of the finite-memory derivative/peak and its limit is not asserted.
-/
namespace Dobson.Coverage3MemoryRing
noncomputable def gamma2 (a b θ : ℝ) : ℝ :=
  (1 + 2 * θ) / (2 * (a + b)) - θ ^ 2 / (θ * (a + b) + a * b)

/-- Exact difference of the two-stage and one-stage coefficients. -/
theorem gamma_difference (a b θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 < θ) :
    gamma2 a b θ - 1 / (2 * a) =
      b * (a * (2 * a - 1) * θ - b * (a + θ)) /
        (2 * a * (a + b) * (θ * (a + b) + a * b)) := by
  have hab : a + b ≠ 0 := ne_of_gt (add_pos ha hb)
  have hQ : θ * (a + b) + a * b ≠ 0 := ne_of_gt (by positivity)
  unfold gamma2
  field_simp
  ring

/-- The manuscript's exact stage-addition criterion. -/
theorem gamma_add_iff (a b θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 < θ) :
    1 / (2 * a) < gamma2 a b θ ↔ b < a * (2 * a - 1) * θ / (a + θ) := by
  rw [← sub_pos, gamma_difference a b θ ha hb hθ,
    div_pos_iff_of_pos_right (by positivity : 0 < 2 * a * (a + b) * (θ * (a + b) + a * b)),
    mul_pos_iff_of_pos_left hb, sub_pos, lt_div_iff₀ (add_pos ha hθ)]

/-- At a fixed memory, existence of a physically allowed added stay is exact. -/
theorem admissible_at_fixed_memory (a θ : ℝ) (ha : 1 ≤ a) (hθ : 0 < θ) :
    (∃ b : ℝ, 1 ≤ b ∧ 1 / (2 * a) < gamma2 a b θ) ↔
      a < (2 * a + 1) * (a - 1) * θ := by
  have ha0 : 0 < a := by linarith
  have hcrit : (1 : ℝ) < a * (2 * a - 1) * θ / (a + θ) ↔
      a < (2 * a + 1) * (a - 1) * θ := by
    rw [lt_div_iff₀ (add_pos ha0 hθ)]
    constructor <;> intro h <;> nlinarith
  constructor
  · rintro ⟨b, hb, hgain⟩
    apply hcrit.mp
    exact lt_of_le_of_lt hb ((gamma_add_iff a b θ ha0 (by linarith) hθ).mp hgain)
  · intro h
    exact ⟨1, le_rfl, (gamma_add_iff a 1 θ ha0 (by norm_num) hθ).mpr (hcrit.mpr h)⟩

/-- Some positive memory and physical added stay work exactly when a > 1. -/
theorem admissible_some_memory (a : ℝ) (ha : 1 ≤ a) :
    (∃ θ : ℝ, 0 < θ ∧ ∃ b : ℝ, 1 ≤ b ∧ 1 / (2 * a) < gamma2 a b θ) ↔ 1 < a := by
  constructor
  · rintro ⟨θ, hθ, hb⟩
    have h := (admissible_at_fixed_memory a θ ha hθ).mp hb
    by_contra hn
    have he : a = 1 := le_antisymm (le_of_not_gt hn) ha
    rw [he] at h
    norm_num at h
  · intro ha1
    have hp : 0 < (2 * a + 1) * (a - 1) := by positivity
    refine ⟨(a + 1) / ((2 * a + 1) * (a - 1)), by positivity, ?_⟩
    apply (admissible_at_fixed_memory a _ ha (by positivity)).mpr
    have he : (2 * a + 1) * (a - 1) * ((a + 1) / ((2 * a + 1) * (a - 1))) = a + 1 := by
      field_simp
    rw [he]
    linarith

/-- The critical polynomial has exactly the stated upper root on a >= 1. -/
theorem critical_quadratic (a : ℝ) (ha : 1 ≤ a) :
    0 < 2 * a ^ 2 - 3 * a - 1 ↔ (3 + Real.sqrt 17) / 4 < a := by
  have hs : Real.sqrt 17 ^ 2 = 17 := Real.sq_sqrt (by norm_num)
  have hs0 := Real.sqrt_nonneg (17 : ℝ)
  constructor
  · intro h
    by_contra hn
    have he : 4 * a - 3 ≤ Real.sqrt 17 := by linarith
    have hx : 0 ≤ 4 * a - 3 := by linarith
    have hh := mul_self_le_mul_self hx he
    nlinarith
  · intro h
    have hp : 0 < (4 * a - 3 - Real.sqrt 17) * (4 * a - 3 + Real.sqrt 17) := by
      apply mul_pos <;> linarith
    nlinarith

/-- Algebra of the stated limiting derivative: peak location, physical boundary,
and the exact example. No convergence of finite-memory peaks is assumed or claimed. -/
theorem limiting_peak (c : ℝ) (hc : 1 ≤ c) :
    2 * c ^ 2 - c + 1 - (2 * c + 3) * ((2 * c ^ 2 - c + 1) / (2 * c + 3)) = 0 ∧
    ((1 : ℝ) < (2 * c ^ 2 - c + 1) / (2 * c + 3) ↔ 2 < c) ∧
    (2 * (5 : ℝ) ^ 2 - 5 + 1) / (2 * 5 + 3) = 46 / 13 := by
  have hd : 0 < 2 * c + 3 := by linarith
  refine ⟨?_, ?_, by norm_num⟩
  · field_simp
    ring
  · rw [lt_div_iff₀ hd]
    constructor
    · intro h
      by_contra hn
      have := mul_nonpos_of_nonpos_of_nonneg (sub_nonpos.mpr (le_of_not_gt hn)) (by linarith : 0 ≤ 2 * c + 1)
      nlinarith
    · intro h
      nlinarith [mul_pos (sub_pos.mpr h) (by linarith : 0 < 2 * c + 1)]

end Dobson.Coverage3MemoryRing
