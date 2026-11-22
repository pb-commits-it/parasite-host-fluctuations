import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Probability.IdentDistrib
import Mathlib.Tactic

/-! # Strict upper bound for unbounded hosts (`prop:exact-mem-unbounded-bounds`)

Coverage correction, 2026-11-22: this row is core. Strict Jensen is proved
for supplied integrable stationary threshold laws. The stationary expected-log
identity follows by integrating the pathwise ratio recursion. Construction,
integrability and nondegeneracy of the physical law, and the separate
kernel-specific lower bound `D > 0`, remain outside credit. -/
namespace Dobson.Coverage6ExactBounds
open Set MeasureTheory Filter
open scoped Topology

noncomputable def logRemovedStay (s v : ℝ) : ℝ := Real.log (Real.exp v-s)

private theorem logRemovedStay_derivative (s v : ℝ) (h : s < Real.exp v) :
    HasDerivAt (logRemovedStay s) (Real.exp v/(Real.exp v-s)) v := by
  exact ((Real.hasDerivAt_exp v).sub_const s).log (sub_pos.mpr h).ne'

/-- Strict concavity of the exact stationary logarithmic observable. -/
theorem logRemovedStay_strictConcave (s : ℝ) (hs : 0 < s) :
    StrictConcaveOn ℝ (Ioi (Real.log s)) (logRemovedStay s) := by
  apply StrictAntiOn.strictConcaveOn_of_deriv (convex_Ioi _) 
  · intro v hv
    exact (logRemovedStay_derivative s v ((Real.log_lt_iff_lt_exp hs).mp hv)).continuousAt.continuousWithinAt
  · intro v hv w hw hvw
    have hv' : s < Real.exp v := (Real.log_lt_iff_lt_exp hs).mp (interior_subset hv)
    have hw' : s < Real.exp w := (Real.log_lt_iff_lt_exp hs).mp (interior_subset hw)
    rw [(logRemovedStay_derivative s v hv').deriv, (logRemovedStay_derivative s w hw').deriv]
    rw [div_lt_div_iff₀ (sub_pos.mpr hw') (sub_pos.mpr hv')]
    nlinarith [Real.exp_lt_exp.mpr hvw]

/-- Strict tangent bound at the zero-growth mean log ratio. -/
theorem stationary_log_strict_tangent (s v : ℝ) (hs : 0 < s) (hs1 : s < 1)
    (hv : Real.log s < v) (hv0 : v ≠ 0) :
    logRemovedStay s v < Real.log (1-s)+v/(1-s) := by
  have hc := logRemovedStay_strictConcave s hs
  have hzero : (0:ℝ) ∈ Ioi (Real.log s) := Real.log_neg hs hs1
  have hd : HasDerivAt (logRemovedStay s) (1/(1-s)) 0 := by
    simpa using logRemovedStay_derivative s 0 (by simpa using hs1)
  rcases lt_or_gt_of_ne hv0 with hvn | hvp
  · have h := hc.lt_slope_of_hasDerivAt hv hzero hvn hd
    rw [slope_def_field] at h
    have h' := (lt_div_iff₀ (by linarith : 0 < 0-v)).mp h
    simp only [logRemovedStay, Real.exp_zero] at h'
    rw [div_eq_mul_inv] at h' ⊢
    dsimp only [logRemovedStay]
    nlinarith
  · have h := hc.slope_lt_of_hasDerivAt hzero hv hvp hd
    rw [slope_def_field, sub_zero] at h
    have h' := (div_lt_iff₀ hvp).mp h
    simp only [logRemovedStay, Real.exp_zero] at h'
    rw [div_eq_mul_inv] at h' ⊢
    dsimp only [logRemovedStay]
    nlinarith

/-- Strict Jensen inequality for the stationary observable, with an actual
probability measure and positive mass off the zero log ratio. -/
theorem stationary_log_strict_jensen {Ω : Type*} [MeasurableSpace Ω]
    (ν : Measure Ω) [IsProbabilityMeasure ν] (v : Ω → ℝ) (s : ℝ)
    (hs : 0 < s) (hs1 : s < 1) (hv : ∀ᵐ ω ∂ν, Real.log s < v ω)
    (hi : Integrable v ν) (hf : Integrable (fun ω => logRemovedStay s (v ω)) ν)
    (hmean : ∫ ω, v ω ∂ν=0) (hnontrivial : 0 < ν {ω | v ω ≠ 0}) :
    (∫ ω, logRemovedStay s (v ω) ∂ν) < Real.log (1-s) := by
  let G : Ω → ℝ := fun ω => Real.log (1-s)+v ω/(1-s)-logRemovedStay s (v ω)
  have hGint : Integrable G ν := ((integrable_const _).add (hi.div_const (1-s))).sub hf
  have hG : ∀ᵐ ω ∂ν, 0 ≤ G ω := by
    filter_upwards [hv] with ω hω
    by_cases hz : v ω=0
    · simp [G,hz,logRemovedStay]
    · have := stationary_log_strict_tangent s (v ω) hs hs1 hω hz
      dsimp only [G]
      linarith
  have hsupp : 0 < ν (Function.support G) := by
    apply hnontrivial.trans_le
    apply measure_mono_ae
    filter_upwards [hv] with ω hω
    intro hz
    have hh := stationary_log_strict_tangent s (v ω) hs hs1 hω hz
    change G ω ≠ 0
    dsimp only [G]
    linarith
  have hpos := (integral_pos_iff_support_of_nonneg_ae hG hGint).mpr hsupp
  have hsplit := integral_sub ((integrable_const (Real.log (1-s))).add (hi.div_const (1-s))) hf
  have hsplit' := integral_add (integrable_const (Real.log (1-s))) (hi.div_const (1-s))
  simp only [Pi.add_apply] at hsplit
  dsimp only [G] at hpos
  rw [hsplit, hsplit', integral_div, hmean] at hpos
  simpa using hpos

/-- Exact pathwise identity from the ratio recursion. It is valid for unbounded
positive host values, and is the identity used before taking stationary means. -/
theorem log_ratio_recursion (s c h U Uprev : ℝ) (hc : 0 < c)
    (hh : 0 < h) (hprev : 0 < Uprev) (hrec : U=s+c*h/Uprev) :
    Real.log (U-s)=Real.log c+Real.log h-Real.log Uprev := by
  have he : U-s=c*h/Uprev := by linarith
  rw [he, Real.log_div (mul_pos hc hh).ne' hprev.ne', Real.log_mul hc.ne' hh.ne']

/-- Integrating the actual ratio recursion proves the expected-log identity.
The current and previous log ratios are `v` and `vprev`; `h` is the host. -/
theorem stationary_log_identity {Ω : Type*} [MeasurableSpace Ω]
    (ν : Measure Ω) [IsProbabilityMeasure ν]
    (v vprev h : Ω → ℝ) (s c : ℝ) (hc : 0 < c)
    (hh : ∀ᵐ ω ∂ν, 0 < h ω)
    (hrec : ∀ᵐ ω ∂ν, Real.exp (v ω) = s+c*h ω/Real.exp (vprev ω))
    (hhost : Integrable (fun ω => Real.log (h ω)) ν)
    (hprev : Integrable vprev ν) :
    Integrable (fun ω => logRemovedStay s (v ω)) ν ∧
      (∫ ω, logRemovedStay s (v ω) ∂ν) =
        Real.log c + (∫ ω, Real.log (h ω) ∂ν) - ∫ ω, vprev ω ∂ν := by
  have he : (fun ω => logRemovedStay s (v ω)) =ᵐ[ν]
      (fun ω => Real.log c + Real.log (h ω) - vprev ω) := by
    filter_upwards [hh, hrec] with ω hhω hrecω
    simpa only [logRemovedStay, Real.log_exp] using
      log_ratio_recursion s c (h ω) (Real.exp (v ω)) (Real.exp (vprev ω))
        hc hhω (Real.exp_pos _) hrecω
  have hsum : Integrable (fun ω => Real.log c + Real.log (h ω)) ν :=
    (integrable_const _).add hhost
  refine ⟨(hsum.sub hprev).congr he.symm, ?_⟩
  rw [integral_congr_ae he, integral_sub hsum hprev,
    integral_add (integrable_const _) hhost]
  simp

/-- Applying strict Jensen to a supplied stationary threshold model yields D<ell.
`hstationary` states equality of successive log-ratio laws, `hmean` defines
zero growth, while `hD` and `hell` define the loss and host log gap. The integrated
identity is proved above instead of being a separate theorem hypothesis. -/
theorem loss_lt_log_gap {Ω : Type*} [MeasurableSpace Ω]
    (ν : Measure Ω) [IsProbabilityMeasure ν] (v vprev h : Ω → ℝ) (s c D ell : ℝ)
    (hs : 0 < s) (hs1 : s < 1) (hv : ∀ᵐ ω ∂ν, Real.log s < v ω)
    (hi : Integrable v ν)
    (hc : 0 < c) (hh : ∀ᵐ ω ∂ν, 0 < h ω)
    (hrec : ∀ᵐ ω ∂ν, Real.exp (v ω) = s+c*h ω/Real.exp (vprev ω))
    (hhost : Integrable (fun ω => Real.log (h ω)) ν)
    (hmean : ∫ ω, v ω ∂ν=0) (hnontrivial : 0 < ν {ω | v ω ≠ 0})
    (hstationary : ProbabilityTheory.IdentDistrib vprev v ν ν)
    (hD : D = Real.log c - Real.log (1-s))
    (hell : ell = -(∫ ω, Real.log (h ω) ∂ν)) :
    D < ell := by
  have hprev : Integrable vprev ν := hstationary.symm.integrable_snd hi
  obtain ⟨hf, he⟩ := stationary_log_identity ν v vprev h s c hc hh hrec hhost hprev
  have hj := stationary_log_strict_jensen ν v s hs hs1 hv hi hf hmean hnontrivial
  rw [he, hstationary.integral_eq, hmean] at hj
  linarith

end Dobson.Coverage6ExactBounds
#print axioms Dobson.Coverage6ExactBounds.logRemovedStay_strictConcave
#print axioms Dobson.Coverage6ExactBounds.stationary_log_strict_tangent
#print axioms Dobson.Coverage6ExactBounds.stationary_log_strict_jensen
#print axioms Dobson.Coverage6ExactBounds.loss_lt_log_gap
#print axioms Dobson.Coverage6ExactBounds.log_ratio_recursion
#print axioms Dobson.Coverage6ExactBounds.stationary_log_identity
