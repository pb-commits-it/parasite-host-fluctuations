import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic

/-!
# Unbounded-host long-stay expansion (`prop:exact-mem-unbounded-long`)

A global scalar logarithm remainder is proved by differentiation, then integrated
against an arbitrary spectral measure. The spectral representation of stochastic
growth, identification of its moments with host moments, and the implicit-root
analytic estimates remain outside this module. No Taylor remainder is assumed.
-/
/-! ## Coverage correction — 2026-11-22

`prop:exact-mem-unbounded-long` remains partial-P1. The spectral/growth
identification, spectral-to-host moment identifications and physical threshold
jets remain owed. The coefficient equations in `threshold_coefficients` and
`centered_threshold_coefficients` are displayed premises. The independent
fidelity audit (2026-11-16) did not credit the supplementary conditional
P1B module as discharging these physical-model steps. -/

namespace Dobson.Coverage6ExactLongStay
open Set MeasureTheory Filter
open scoped Topology

noncomputable def cubicGap (x : ℝ) : ℝ :=
  x-x^2/2+x^3/3-Real.log (1+x)

private theorem cubicGap_derivative (x : ℝ) (hx : 0 ≤ x) :
    HasDerivAt cubicGap (x^3/(1+x)) x := by
  have hd := (((hasDerivAt_id x).sub (((hasDerivAt_id x).pow 2).div_const 2)).add
    (((hasDerivAt_id x).pow 3).div_const 3)).sub
    (((hasDerivAt_id x).const_add 1).log (by linarith : 1+x ≠ 0))
  convert hd using 1
  · ext t; rfl
  · dsimp
    field_simp
    <;> ring

private theorem quarticGap_derivative (x : ℝ) (hx : 0 ≤ x) :
    HasDerivAt (fun y => y^4/4-cubicGap y) (x^4/(1+x)) x := by
  convert (((hasDerivAt_id x).pow 4).div_const 4).sub (cubicGap_derivative x hx) using 1
  · ext y; rfl
  · dsimp
    field_simp
    <;> ring

/-- Global logarithm remainder for all y≥0, including arbitrarily large y. -/
theorem log_cubic_remainder (y : ℝ) (hy : 0 ≤ y) :
    |Real.log (1+y)-y+y^2/2-y^3/3| ≤ y^4/4 := by
  have hmono : MonotoneOn cubicGap (Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici (0:ℝ))
    · intro x hx
      exact (cubicGap_derivative x hx).continuousAt.continuousWithinAt
    · intro x hx
      exact (cubicGap_derivative x (interior_subset hx)).differentiableAt.differentiableWithinAt
    · intro x hx
      rw [(cubicGap_derivative x (interior_subset hx)).deriv]
      have hx0 : 0 ≤ x := interior_subset hx
      exact div_nonneg (pow_nonneg hx0 _) (by linarith)
  have hmono' : MonotoneOn (fun x => x^4/4-cubicGap x) (Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici (0:ℝ))
    · intro x hx
      exact (quarticGap_derivative x hx).continuousAt.continuousWithinAt
    · intro x hx
      exact (quarticGap_derivative x (interior_subset hx)).differentiableAt.differentiableWithinAt
    · intro x hx
      rw [(quarticGap_derivative x (interior_subset hx)).deriv]
      have hx0 : 0 ≤ x := interior_subset hx
      exact div_nonneg (pow_nonneg hx0 _) (by linarith)
  have hg := hmono (by simp) hy hy
  have hb := hmono' (by simp) hy hy
  simp only [cubicGap, zero_pow, zero_div, sub_zero, zero_add, add_zero, Real.log_one] at hg hb
  apply abs_le.mpr
  constructor <;> linarith

/-- Integration of the proved scalar remainder. Only positivity and integrability
of the first four spectral powers and logarithm are required. -/
theorem spectral_integral_quartic {Ω : Type*} [MeasurableSpace Ω]
    (ν : Measure Ω) (X : Ω → ℝ) (z : ℝ) (hz : 0 ≤ z)
    (hX : ∀ᵐ ω ∂ν, 0 ≤ X ω)
    (h1 : Integrable X ν) (h2 : Integrable (fun ω => X ω^2) ν)
    (h3 : Integrable (fun ω => X ω^3) ν) (h4 : Integrable (fun ω => X ω^4) ν)
    (hlog : Integrable (fun ω => Real.log (1+z*X ω)) ν) :
    |(∫ ω, Real.log (1+z*X ω) ∂ν) - z*(∫ ω, X ω ∂ν) +
      z^2/2*(∫ ω, X ω^2 ∂ν) - z^3/3*(∫ ω, X ω^3 ∂ν)| ≤
      z^4/4*(∫ ω, X ω^4 ∂ν) := by
  let R : Ω → ℝ := fun ω => Real.log (1+z*X ω)-z*X ω+z^2/2*X ω^2-z^3/3*X ω^3
  have hi : Integrable R ν := ((hlog.sub (h1.const_mul z)).add
    (h2.const_mul (z^2/2))).sub (h3.const_mul (z^3/3))
  have hr : ∀ᵐ ω ∂ν, |R ω| ≤ z^4/4*X ω^4 := by
    filter_upwards [hX] with ω hω
    convert log_cubic_remainder (z*X ω) (mul_nonneg hz hω) using 1 <;>
      try dsimp only [R]
    · congr 1; ring
    · ring
  have hb := (abs_integral_le_integral_abs (f := R) (μ := ν)).trans
    (integral_mono_ae hi.abs (h4.const_mul (z^4/4)) hr)
  dsimp only [R] at hb
  have hs₃ := integral_sub ((hlog.sub (h1.const_mul z)).add (h2.const_mul (z^2/2)))
    (h3.const_mul (z^3/3))
  have hs₂ := integral_add (hlog.sub (h1.const_mul z)) (h2.const_mul (z^2/2))
  have hs₁ := integral_sub hlog (h1.const_mul z)
  simp only [Pi.add_apply, Pi.sub_apply] at hs₁ hs₂ hs₃
  rw [hs₃, hs₂, hs₁] at hb
  simpa only [integral_const_mul] using hb

/-- Raw first-through-third host moments expanded about mean one. This is exactly
what enumerating closed walks of lengths four and six requires. -/
theorem spectral_moments_centered (V C₁ C₂ m₃ T₁₂ T₂₁ T₁₁₁ : ℝ) :
    (1+V)+2*(1+C₁) = 3+2*(V/2+C₁) ∧
    ((1+3*V+m₃)+3*((1+V+2*C₁+T₂₁)+(1+V+2*C₁+T₁₂)+
      (1+2*C₁+C₂+T₁₁₁)))/3 =
      10/3+6*(V/2+C₁)+C₂+T₁₂+T₂₁+T₁₁₁+m₃/3 := by
  constructor <;> ring

/-- The two threshold-jet equations uniquely determine the displayed A and B.
Inputs are coefficient equations from the lower-level spectral logarithm. -/
theorem threshold_coefficients (μ₂ μ₃ A B : ℝ)
    (h₂ : A+2-μ₂/2-1/2=0)
    (h₃ : B+A^2/2+2*A+3-μ₂*(A+2)+μ₃/3-1/3=0) :
    A=(μ₂-3)/2 ∧ B=3*A^2/2+5*A+10/3-μ₃/3 := by
  have hA : μ₂=2*A+3 := by linarith
  constructor
  · linarith
  · rw [hA] at h₃
    nlinarith

/-- The general moment formula for B follows algebraically from the spectral
moment identifications, not from an assumed value of B. -/
theorem centered_threshold_coefficients (V C₁ C₂ m₃ T₁₂ T₂₁ T₁₁₁ μ₂ μ₃ A B : ℝ)
    (hm₂ : μ₂=3+V+2*C₁)
    (hm₃ : μ₃=10+9*V+18*C₁+3*C₂+3*T₁₂+3*T₂₁+3*T₁₁₁+m₃)
    (h₂ : A+2-μ₂/2-1/2=0)
    (h₃ : B+A^2/2+2*A+3-μ₂*(A+2)+μ₃/3-1/3=0) :
    A=V/2+C₁ ∧
      B= -A-C₂-T₁₂-T₂₁-T₁₁₁-m₃/3+3*A^2/2 := by
  obtain ⟨hA,hB⟩ := threshold_coefficients μ₂ μ₃ A B h₂ h₃
  have hAV : A=V/2+C₁ := by rw [hm₂] at hA; linarith
  refine ⟨hAV, ?_⟩
  rw [hm₃] at hB
  nlinarith [hAV]

/-- Gamma-refresh specialization of both long-stay coefficients. -/
theorem gamma_refresh_coefficients (V φ : ℝ) :
    V/2+V*φ=V*(1/2+φ) ∧
    -(V/2+V*φ)-V*φ^2-φ*(2*V^2)-φ*(2*V^2)-φ^2*(2*V^2)-(2*V^2)/3+
      3*(V/2+V*φ)^2/2 =
      -V*(1/2+φ+φ^2)-2*V^2*(1/3+2*φ+φ^2)+3*V^2*(1/2+φ)^2/2 := by
  constructor <;> ring

end Dobson.Coverage6ExactLongStay
#print axioms Dobson.Coverage6ExactLongStay.log_cubic_remainder
#print axioms Dobson.Coverage6ExactLongStay.spectral_integral_quartic
#print axioms Dobson.Coverage6ExactLongStay.spectral_moments_centered
#print axioms Dobson.Coverage6ExactLongStay.threshold_coefficients
#print axioms Dobson.Coverage6ExactLongStay.centered_threshold_coefficients
#print axioms Dobson.Coverage6ExactLongStay.gamma_refresh_coefficients
