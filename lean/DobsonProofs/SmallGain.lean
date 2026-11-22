import Mathlib

/-!
# The general small-gain statement (item G1)

`f : ℂ → ℂ`; `Complex.I` is the imaginary unit `i`; `‖w‖` is the modulus of `w`.

Encoding of the hypotheses:
* "continuous on `{z : Re z ≥ 0}`" is `ContinuousOn f {z | 0 ≤ z.re}`;
* "differentiable (complex) on `{z : Re z > 0}`" is `DifferentiableOn ℂ f {z | 0 < z.re}`;
* "bounded on `{z : Re z ≥ 0}`" is `∃ M, ∀ z, 0 ≤ z.re → ‖f z‖ ≤ M`;
* the bound on the imaginary axis is `∀ y : ℝ, ‖f (i·y)‖ ≤ c`.

The proof applies Mathlib's Phragmén–Lindelöf theorem for the right half-plane,
`PhragmenLindelof.right_half_plane_of_bounded_on_real`. Its hypotheses are weaker than
"bounded": a growth bound `f = O(exp(B·‖z‖^c'))` with `c' < 2` in the open half-plane, and
boundedness on the positive real axis. Both are derived here from the single bound `M`
(with `c' = 0`, `B = 0`), so nothing beyond the stated hypotheses is assumed.
-/

namespace Dobson.SmallGain

/-- The bound behind G1. Under the hypotheses of G1 except `c < 1`:
`‖f z‖ ≤ c` for every `z` with `Re z ≥ 0`. -/
theorem norm_le_of_bounded (f : ℂ → ℂ) (c : ℝ)
    (hcont : ContinuousOn f {z : ℂ | 0 ≤ z.re})
    (hdiff : DifferentiableOn ℂ f {z : ℂ | 0 < z.re})
    (hbdd : ∃ M : ℝ, ∀ z : ℂ, 0 ≤ z.re → ‖f z‖ ≤ M)
    (him : ∀ y : ℝ, ‖f (Complex.I * (y : ℂ))‖ ≤ c) :
    ∀ z : ℂ, 0 ≤ z.re → ‖f z‖ ≤ c := by
  obtain ⟨M, hM⟩ := hbdd
  intro z hz
  have hd : DiffContOnCl ℂ f {z : ℂ | 0 < z.re} :=
    ⟨hdiff, by rwa [Complex.closure_setOfPred_lt_re]⟩
  have hexp : ∃ c' < (2 : ℝ), ∃ B : ℝ,
      f =O[Bornology.cobounded ℂ ⊓ Filter.principal {z : ℂ | 0 < z.re}]
        fun z => Real.exp (B * ‖z‖ ^ c') := by
    refine ⟨0, by norm_num, 0, Asymptotics.IsBigO.of_bound M ?_⟩
    refine Filter.eventually_inf_principal.mpr (Filter.Eventually.of_forall fun w hw => ?_)
    simp only [zero_mul, Real.exp_zero, norm_one, mul_one]
    exact hM w (le_of_lt hw)
  have hre : Filter.IsBoundedUnder (· ≤ ·) Filter.atTop fun x : ℝ => ‖f (x : ℂ)‖ := by
    refine ⟨M, Filter.eventually_map.mpr ?_⟩
    filter_upwards [Filter.eventually_ge_atTop (0 : ℝ)] with x hx
    exact hM x (by simpa using hx)
  exact PhragmenLindelof.right_half_plane_of_bounded_on_real hd hexp hre
    (fun x => by rw [mul_comm]; exact him x) hz

/-- **G1.** Let `f : ℂ → ℂ` be continuous on `{z : Re z ≥ 0}`, complex differentiable on
`{z : Re z > 0}`, bounded on `{z : Re z ≥ 0}`, with `‖f(iy)‖ ≤ c` for every real `y`, where
`c < 1`. Then `f(z) ≠ 1` for every `z` with `Re z ≥ 0`. -/
theorem G1 (f : ℂ → ℂ) (c : ℝ)
    (hcont : ContinuousOn f {z : ℂ | 0 ≤ z.re})
    (hdiff : DifferentiableOn ℂ f {z : ℂ | 0 < z.re})
    (hbdd : ∃ M : ℝ, ∀ z : ℂ, 0 ≤ z.re → ‖f z‖ ≤ M)
    (him : ∀ y : ℝ, ‖f (Complex.I * (y : ℂ))‖ ≤ c)
    (hc : c < 1) :
    ∀ z : ℂ, 0 ≤ z.re → f z ≠ 1 := by
  intro z hz h1
  have hbound := norm_le_of_bounded f c hcont hdiff hbdd him z hz
  rw [h1, norm_one] at hbound
  linarith

end Dobson.SmallGain

#print axioms Dobson.SmallGain.norm_le_of_bounded
#print axioms Dobson.SmallGain.G1
