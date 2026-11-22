import Mathlib.Analysis.SpecialFunctions.Arcosh
import Mathlib.Analysis.SpecialFunctions.Complex.LogDeriv
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals
import Mathlib.Analysis.SpecialFunctions.Integrability.Basic
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.MeasureTheory.Integral.IntegralEqImproper
import Mathlib.MeasureTheory.Integral.ExpDecay
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Tactic

/-!
# GAP 3 Step B: Noise-amplitude analyticity vs. small-noise divergence (`NoiseAnalyticity.lean`)

Machine-checked Lean 4 / Mathlib formalisation of the six load-bearing targets `L1`–`L6` from
the noise-analyticity target specification (`lean_targets.md`) (companion to `gap3_holo.md` and
`gap3_holo_checks.py`):

* **Target L1 (`L1_mobius_contraction`):** Exact complex one-step trapping, Poincaré derivative
  identity, and two-step hyperbolic Lipschitz contraction on `{w ∈ ℂ : Re w > 0}` for
  `g_b(w) = σ + b / w` (`Re σ = σ₁ > 0`, `b₀, b₁ > 0`).
* **Target L2 (`L2_log_poincare_pointwise`, `L2_log_poincare_path`, `L2_log_poincare_dist`,
  `L2_initial_step`):** Hyperbolic metric domination of the rotated principal logarithm
  `w ↦ log(e^{iψ/2} w)` (pointwise derivative bound `≤ 2`, integral bound along any `C¹` path in
  `ℍ`, global closed-form Poincaré distance bound `‖rotatedLog ψ w₁ - rotatedLog ψ w₂‖ ≤ 2 d_ℍ`,
  and initial-step displacement bound from `w₀ = e^{-iψ/2}`).
* **Target L3 (`L3_kummer_moments`, `L3_threshold_limit`):** Exact integration-by-parts quadratic
  moment identity on `(0, ∞)` for the Kummer weight `t^{q-1}(1 + t)^{-q} e^{-x t}`, strict upper
  bounds on `E[T]` and `E[log T]`, and threshold contradiction limit `→ log(1 - s) < 0`.
* **Target L4 (`L4_stokes_elimination`):** Exact `B₁`-free linear combination and simultaneous
  zero reduction for the Stokes complex growth-rate ratio.
* **Target L5 (`L5_instanton_action`, `L5_borel_disk`):** Exact square derivative and linear lower
  bound of the Gamma instanton action `A_gamma(s)`, and certified positivity lower bound of
  `Re Φ''(u)` on the complex Borel disk `‖u - t_+‖ ≤ t_+ / 5`.
* **Target L6 (`L6_three_step_elasticity`, `L6_perturbation_diag`):** Three-step `2 × 2` positive
  matrix product and entrywise log-`c` elasticity lower bound, plus perturbation-recursion
  diagonal invertibility for all `(s, φ) ∈ (0, 1) × (-1, 1)`.

2026-11-16, auditfix1, ID 110: `hratio` in `L3_threshold_limit` is
an ordinary input to the abstract sequence lemma, but its physical Tricomi
application requires a model-specific zero-noise threshold limit and negative
shape continuation. D-exact.tex, prop:exact-R7, supplies the positive-noise
limit argument; the continuation/scaling handoff is not proved in this Lean
module. The referenced gap3holo2 proof source is absent from this checkout,
so the complete bridge is classified as an unproved step, not a model
assumption. That application is partial; the independent explicit Kummer
integrals, deterministic maps and matrix computations retain their own scope.
ID 109 (unused but correct s < 1) is intentionally unchanged.
-/

namespace Dobson.NoiseAnalyticity

open MeasureTheory Filter Set
open scoped Topology Real Matrix

/-! ## Target L1: Exact Complex Two-Step Poincaré Contraction on a Rotated Half-Plane -/

/-- One-step complex Möbius map `g_b(w) = σ + b / w`. -/
noncomputable def mobiusStep (σ : ℂ) (b : ℝ) (w : ℂ) : ℂ :=
  σ + (b : ℂ) / w

/-- Two-step composed complex Möbius map `G(w) = g_{b₁}(g_{b₀}(w))`. -/
noncomputable def mobiusTwoStep (σ : ℂ) (b₀ b₁ : ℝ) (w : ℂ) : ℂ :=
  mobiusStep σ b₁ (mobiusStep σ b₀ w)

private theorem mobiusStep_re (σ : ℂ) (b : ℝ) (w : ℂ) :
    (mobiusStep σ b w).re = σ.re + b * w.re / ‖w‖ ^ 2 := by
  unfold mobiusStep
  rw [Complex.add_re, Complex.div_re, Complex.ofReal_re, Complex.ofReal_im,
    zero_mul, zero_div, add_zero, Complex.normSq_eq_norm_sq]

private theorem hasDerivAt_mobiusStep (σ : ℂ) (b : ℝ) {w : ℂ} (hw : w ≠ 0) :
    HasDerivAt (mobiusStep σ b) (-(b : ℂ) / w ^ 2) w := by
  unfold mobiusStep
  have hdiv : HasDerivAt (fun z : ℂ => (b : ℂ) / z) ((0 * w - (b : ℂ) * 1) / w ^ 2) w :=
    (hasDerivAt_const w (b : ℂ)).fun_div (hasDerivAt_id w) hw
  have hsum := (hasDerivAt_const w σ).add hdiv
  convert hsum using 1
  ring

/-- **L1 (`lean_targets.md` Target L1).**
For `σ₁ > 0`, `σ₂ ∈ ℝ`, `σ = σ₁ + i σ₂`, `b₀ > 0`, `b₁ > 0`, and `w ∈ ℂ` with `Re w > 0`:
1. `Re(g₀(w)) = σ₁ + b₀ Re(w) / ‖w‖² > σ₁ > 0`;
2. `g₀` is complex differentiable at `w` and
   `‖deriv g₀ w‖ * Re w / Re(g₀ w) = t / (σ₁ + t) < 1` with `t = b₀ Re w / ‖w‖² > 0`;
3. `G = g₁ ∘ g₀` is complex differentiable at `w` and
   `‖deriv G w‖ * Re w / Re(G w) ≤ b₁ / (√(σ₁² + b₁) + σ₁)²`
   `= (√(σ₁² + b₁) - σ₁) / (√(σ₁² + b₁) + σ₁) < 1`. -/
theorem L1_mobius_contraction (σ₁ σ₂ b₀ b₁ : ℝ) (w : ℂ)
    (hσ₁ : 0 < σ₁) (hb₀ : 0 < b₀) (hb₁ : 0 < b₁) (hw : 0 < w.re) :
    let σ : ℂ := (σ₁ : ℂ) + (σ₂ : ℂ) * Complex.I
    let t : ℝ := b₀ * w.re / ‖w‖ ^ 2
    0 < t ∧
    (mobiusStep σ b₀ w).re = σ₁ + t ∧
    σ₁ < (mobiusStep σ b₀ w).re ∧
    0 < (mobiusStep σ b₀ w).re ∧
    DifferentiableAt ℂ (mobiusStep σ b₀) w ∧
    ‖deriv (mobiusStep σ b₀) w‖ * w.re / (mobiusStep σ b₀ w).re = t / (σ₁ + t) ∧
    t / (σ₁ + t) < 1 ∧
    DifferentiableAt ℂ (mobiusTwoStep σ b₀ b₁) w ∧
    ‖deriv (mobiusTwoStep σ b₀ b₁) w‖ * w.re / (mobiusTwoStep σ b₀ b₁ w).re ≤
      b₁ / (Real.sqrt (σ₁ ^ 2 + b₁) + σ₁) ^ 2 ∧
    b₁ / (Real.sqrt (σ₁ ^ 2 + b₁) + σ₁) ^ 2 =
      (Real.sqrt (σ₁ ^ 2 + b₁) - σ₁) / (Real.sqrt (σ₁ ^ 2 + b₁) + σ₁) ∧
    (Real.sqrt (σ₁ ^ 2 + b₁) - σ₁) / (Real.sqrt (σ₁ ^ 2 + b₁) + σ₁) < 1 := by
  dsimp only
  set σ : ℂ := (σ₁ : ℂ) + (σ₂ : ℂ) * Complex.I
  have hσ_re : σ.re = σ₁ := by simp [σ]
  have hw_ne : w ≠ 0 := by intro h0; simp [h0] at hw
  have hwnorm_pos : 0 < ‖w‖ := norm_pos_iff.mpr hw_ne
  set t : ℝ := b₀ * w.re / ‖w‖ ^ 2
  have ht_pos : 0 < t := div_pos (mul_pos hb₀ hw) (sq_pos_of_pos hwnorm_pos)
  set w₁ : ℂ := mobiusStep σ b₀ w
  have hw₁_re : w₁.re = σ₁ + t := by
    dsimp [w₁, t]
    rw [mobiusStep_re, hσ_re]
  have hw₁_gt : σ₁ < w₁.re := by linarith
  have hw₁_pos : 0 < w₁.re := by linarith
  have hw₁_ne : w₁ ≠ 0 := by intro h0; simp [h0] at hw₁_pos
  have hw₁norm_pos : 0 < ‖w₁‖ := norm_pos_iff.mpr hw₁_ne
  have hd0 : HasDerivAt (mobiusStep σ b₀) (-(b₀ : ℂ) / w ^ 2) w :=
    hasDerivAt_mobiusStep σ b₀ hw_ne
  have hd1 : HasDerivAt (mobiusStep σ b₁) (-(b₁ : ℂ) / w₁ ^ 2) w₁ :=
    hasDerivAt_mobiusStep σ b₁ hw₁_ne
  have hdG : HasDerivAt (mobiusTwoStep σ b₀ b₁)
      ((-(b₁ : ℂ) / w₁ ^ 2) * (-(b₀ : ℂ) / w ^ 2)) w :=
    hd1.comp w hd0
  have hnorm_d0 : ‖deriv (mobiusStep σ b₀) w‖ = b₀ / ‖w‖ ^ 2 := by
    rw [hd0.deriv, norm_div, norm_neg, Complex.norm_real, Real.norm_of_nonneg hb₀.le, norm_pow]
  have hpoinc0 : ‖deriv (mobiusStep σ b₀) w‖ * w.re / w₁.re = t / (σ₁ + t) := by
    rw [hnorm_d0, hw₁_re]
    dsimp [t]
    ring
  have ht_lt1 : t / (σ₁ + t) < 1 := (div_lt_one (by linarith)).mpr (by linarith)
  set t₁ : ℝ := b₁ * w₁.re / ‖w₁‖ ^ 2
  have ht₁_pos : 0 < t₁ := div_pos (mul_pos hb₁ hw₁_pos) (sq_pos_of_pos hw₁norm_pos)
  have hG_re : (mobiusTwoStep σ b₀ b₁ w).re = σ₁ + t₁ := by
    change (mobiusStep σ b₁ w₁).re = σ₁ + t₁
    rw [mobiusStep_re, hσ_re]
  have hnorm_dG : ‖deriv (mobiusTwoStep σ b₀ b₁) w‖ = (b₁ / ‖w₁‖ ^ 2) * (b₀ / ‖w‖ ^ 2) := by
    rw [hdG.deriv, norm_mul, norm_div, norm_div, norm_neg, norm_neg,
      Complex.norm_real, Complex.norm_real, Real.norm_of_nonneg hb₁.le,
      Real.norm_of_nonneg hb₀.le, norm_pow, norm_pow]
  have hpoincG :
      ‖deriv (mobiusTwoStep σ b₀ b₁) w‖ * w.re / (mobiusTwoStep σ b₀ b₁ w).re =
        (t₁ / (σ₁ + t₁)) * (t / (σ₁ + t)) := by
    rw [hnorm_dG, hG_re]
    have hw₁_re_ne : w₁.re ≠ 0 := hw₁_pos.ne'
    have hG_re_ne : σ₁ + t₁ ≠ 0 := by linarith
    calc (b₁ / ‖w₁‖ ^ 2) * (b₀ / ‖w‖ ^ 2) * w.re / (σ₁ + t₁)
        = ((b₁ * w₁.re / ‖w₁‖ ^ 2) / (σ₁ + t₁)) *
            ((b₀ * w.re / ‖w‖ ^ 2) / w₁.re) := by
          field_simp [hw₁_re_ne, hG_re_ne, hwnorm_pos.ne', hw₁norm_pos.ne']
      _ = (t₁ / (σ₁ + t₁)) * (t / w₁.re) := rfl
      _ = (t₁ / (σ₁ + t₁)) * (t / (σ₁ + t)) := by rw [hw₁_re]
  have hw₁_re_le : w₁.re ≤ ‖w₁‖ := Complex.re_le_norm w₁
  have hw₁_sq_le : w₁.re ^ 2 ≤ ‖w₁‖ ^ 2 :=
    pow_le_pow_left₀ hw₁_pos.le hw₁_re_le 2
  have ht₁_le : t₁ ≤ b₁ / (σ₁ + t) := by
    dsimp [t₁]
    have hdiv_le : b₁ * w₁.re / ‖w₁‖ ^ 2 ≤ b₁ * w₁.re / w₁.re ^ 2 :=
      div_le_div_of_nonneg_left (mul_pos hb₁ hw₁_pos).le (sq_pos_of_pos hw₁_pos) hw₁_sq_le
    have hsimp : b₁ * w₁.re / w₁.re ^ 2 = b₁ / (σ₁ + t) := by
      rw [sq, mul_div_mul_right _ _ hw₁_pos.ne', hw₁_re]
    linarith
  have hstep1_le : t₁ / (σ₁ + t₁) ≤ b₁ / (σ₁ ^ 2 + b₁ + σ₁ * t) := by
    have hden_pos : 0 < σ₁ ^ 2 + b₁ + σ₁ * t := by positivity
    rw [div_le_div_iff₀ (by linarith) hden_pos]
    have ht₁_mul : t₁ * (σ₁ + t) ≤ b₁ :=
      (le_div_iff₀ (by linarith : 0 < σ₁ + t)).mp ht₁_le
    nlinarith
  set S : ℝ := Real.sqrt (σ₁ ^ 2 + b₁)
  have hS_sq : S ^ 2 = σ₁ ^ 2 + b₁ := Real.sq_sqrt (by positivity)
  have hS_pos : 0 < S := Real.sqrt_pos.mpr (by positivity)
  have hS_gt : σ₁ < S := by
    have : σ₁ ^ 2 < S ^ 2 := by linarith
    nlinarith
  have hamgm : t * (S + σ₁) ^ 2 ≤ (σ₁ + t) * (σ₁ ^ 2 + b₁ + σ₁ * t) := by
    have hdiff : (σ₁ + t) * (σ₁ ^ 2 + b₁ + σ₁ * t) - t * (S + σ₁) ^ 2 = σ₁ * (S - t) ^ 2 := by
      linear_combination (σ₁ + t) * hS_sq.symm
    have hnn : 0 ≤ σ₁ * (S - t) ^ 2 := by positivity
    linarith
  have hprod_le : (t₁ / (σ₁ + t₁)) * (t / (σ₁ + t)) ≤ b₁ / (S + σ₁) ^ 2 := by
    have h1 : (t₁ / (σ₁ + t₁)) * (t / (σ₁ + t)) ≤
        (b₁ / (σ₁ ^ 2 + b₁ + σ₁ * t)) * (t / (σ₁ + t)) :=
      mul_le_mul_of_nonneg_right hstep1_le (by positivity)
    have h2 : (b₁ / (σ₁ ^ 2 + b₁ + σ₁ * t)) * (t / (σ₁ + t)) =
        (b₁ * t) / ((σ₁ + t) * (σ₁ ^ 2 + b₁ + σ₁ * t)) := by
      rw [div_mul_div_comm]
      congr 1
      ring
    have h3 : (b₁ * t) / ((σ₁ + t) * (σ₁ ^ 2 + b₁ + σ₁ * t)) ≤
        (b₁ * t) / (t * (S + σ₁) ^ 2) :=
      div_le_div_of_nonneg_left (by positivity) (by positivity) hamgm
    have h4 : (b₁ * t) / (t * (S + σ₁) ^ 2) = b₁ / (S + σ₁) ^ 2 := by
      rw [mul_comm b₁ t, mul_div_mul_left _ _ ht_pos.ne']
    linarith
  have hrat_eq : b₁ / (S + σ₁) ^ 2 = (S - σ₁) / (S + σ₁) := by
    have hsum_ne : S + σ₁ ≠ 0 := by positivity
    have hb₁_eq : b₁ = (S - σ₁) * (S + σ₁) := by linear_combination -hS_sq
    rw [hb₁_eq, sq, mul_div_mul_right _ _ hsum_ne]
  have hrat_lt1 : (S - σ₁) / (S + σ₁) < 1 :=
    (div_lt_one (by positivity)).mpr (by linarith)
  exact ⟨ht_pos, hw₁_re, hw₁_gt, hw₁_pos, hd0.differentiableAt, hpoinc0, ht_lt1,
    hdG.differentiableAt, hpoincG ▸ hprod_le, hrat_eq, hrat_lt1⟩

/-! ## Target L2: Logarithmic Lipschitz Constant and Initial-Step Distance on `ℍ_{ψ/2}` -/

/-- Rotated principal logarithm `w ↦ log(e^{iψ/2} w)` on `ℍ = {Re w > 0}`. -/
noncomputable def rotatedLog (ψ : ℝ) (w : ℂ) : ℂ :=
  Complex.log (Complex.exp (((ψ / 2 : ℝ) : ℂ) * Complex.I) * w)

/-- Closed-form Poincaré hyperbolic distance on the right half-plane `ℍ = {Re w > 0}`
with metric element `ds = |dw| / (2 Re w)`:
`d_ℍ(w₁, w₂) = (1 / 2) arcosh(1 + ‖w₁ - w₂‖² / (2 Re w₁ Re w₂))`. -/
noncomputable def poincareDistRHP (w₁ w₂ : ℂ) : ℝ :=
  (1 / 2 : ℝ) * Real.arcosh (1 + ‖w₁ - w₂‖ ^ 2 / (2 * w₁.re * w₂.re))

private theorem rot_mul_mem_slitPlane {ψ : ℝ} (hψ_lo : -Real.pi < ψ) (hψ_hi : ψ < Real.pi)
    {w : ℂ} (hw : 0 < w.re) :
    Complex.exp (((ψ / 2 : ℝ) : ℂ) * Complex.I) * w ∈ Complex.slitPlane := by
  rw [Complex.mem_slitPlane_iff]
  set c : ℝ := Real.cos (ψ / 2)
  set sn : ℝ := Real.sin (ψ / 2)
  have hc_pos : 0 < c := Real.cos_pos_of_mem_Ioo ⟨by linarith, by linarith⟩
  have hprod_re : (Complex.exp (((ψ / 2 : ℝ) : ℂ) * Complex.I) * w).re =
      c * w.re - sn * w.im := by
    rw [Complex.mul_re, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im]
  have hprod_im : (Complex.exp (((ψ / 2 : ℝ) : ℂ) * Complex.I) * w).im =
      c * w.im + sn * w.re := by
    rw [Complex.mul_im, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im, mul_comm sn w.re]
  by_cases him : (Complex.exp (((ψ / 2 : ℝ) : ℂ) * Complex.I) * w).im = 0
  · left
    have him' : c * w.im + sn * w.re = 0 := hprod_im ▸ him
    have hc_re : c * (Complex.exp (((ψ / 2 : ℝ) : ℂ) * Complex.I) * w).re = w.re := by
      rw [hprod_re]
      have hcs : c ^ 2 + sn ^ 2 = 1 := Real.cos_sq_add_sin_sq (ψ / 2)
      linear_combination w.re * hcs - sn * him'
    nlinarith
  · exact Or.inr him

private theorem hasDerivAt_rotatedLog {ψ : ℝ} (hψ_lo : -Real.pi < ψ) (hψ_hi : ψ < Real.pi)
    {w : ℂ} (hw : 0 < w.re) :
    HasDerivAt (rotatedLog ψ) w⁻¹ w := by
  unfold rotatedLog
  set α : ℂ := Complex.exp (((ψ / 2 : ℝ) : ℂ) * Complex.I)
  have hα_ne : α ≠ 0 := Complex.exp_ne_zero _
  have hw_ne : w ≠ 0 := by intro h0; simp [h0] at hw
  have hlin : HasDerivAt (fun z : ℂ => α * z) α w := by
    simpa using (hasDerivAt_id w).const_mul α
  have hlog := hlin.clog (rot_mul_mem_slitPlane hψ_lo hψ_hi hw)
  have hsimp : α / (α * w) = w⁻¹ := by
    field_simp [hα_ne, hw_ne]
  rwa [hsimp] at hlog

/-- Helper: `sinh u * cos u ≤ sin u` for `0 ≤ u ≤ π / 2`. -/
private theorem sinh_mul_cos_le_sin {u : ℝ} (hu0 : 0 ≤ u) (hu1 : u ≤ Real.pi / 2) :
    Real.sinh u * Real.cos u ≤ Real.sin u := by
  let g : ℝ → ℝ := fun t => Real.sin t - Real.sinh t * Real.cos t
  let g' : ℝ → ℝ := fun t => Real.cos t - Real.cosh t * Real.cos t + Real.sinh t * Real.sin t
  have hg_deriv : ∀ t : ℝ, HasDerivAt g (g' t) t := by
    intro t
    have hmul := (Real.hasDerivAt_sinh t).mul (Real.hasDerivAt_cos t)
    have hsub := (Real.hasDerivAt_sin t).sub hmul
    convert hsub using 1
    dsimp [g']
    ring
  have hg'_deriv : ∀ t : ℝ, HasDerivAt g' (-Real.sin t + 2 * Real.cosh t * Real.sin t) t := by
    intro t
    have h1 := Real.hasDerivAt_cos t
    have h2 := (Real.hasDerivAt_cosh t).mul (Real.hasDerivAt_cos t)
    have h3 := (Real.hasDerivAt_sinh t).mul (Real.hasDerivAt_sin t)
    have htot := (h1.sub h2).add h3
    convert htot using 1
    ring
  have hg'_mono : MonotoneOn g' (Icc 0 (Real.pi / 2)) := by
    refine monotoneOn_of_deriv_nonneg (convex_Icc 0 (Real.pi / 2))
      (fun t _ => (hg'_deriv t).continuousAt.continuousWithinAt)
      (fun t ht => (hg'_deriv t).differentiableAt.differentiableWithinAt) ?_
    intro t ht
    rw [interior_Icc] at ht
    rw [(hg'_deriv t).deriv]
    have hsin_nn : 0 ≤ Real.sin t := Real.sin_nonneg_of_nonneg_of_le_pi ht.1.le (by linarith [ht.2])
    have hcosh_ge1 : 1 ≤ Real.cosh t := Real.one_le_cosh t
    nlinarith
  have hg_mono : MonotoneOn g (Icc 0 (Real.pi / 2)) := by
    refine monotoneOn_of_deriv_nonneg (convex_Icc 0 (Real.pi / 2))
      (fun t _ => (hg_deriv t).continuousAt.continuousWithinAt)
      (fun t ht => (hg_deriv t).differentiableAt.differentiableWithinAt) ?_
    intro t ht
    rw [interior_Icc] at ht
    rw [(hg_deriv t).deriv]
    have hg'0 : g' 0 = 0 := by simp [g']
    rw [← hg'0]
    exact hg'_mono ⟨le_rfl, by positivity⟩ ⟨ht.1.le, ht.2.le⟩ ht.1.le
  have hg0 : g 0 = 0 := by simp [g]
  have hgu : 0 ≤ g u := by
    rw [← hg0]
    exact hg_mono ⟨le_rfl, by positivity⟩ ⟨hu0, hu1⟩ hu0
  dsimp [g] at hgu
  linarith

/-- Helper: `cosh (√(a² + b²)) ≤ cosh a * cosh b` for all `a, b : ℝ`. -/
private theorem cosh_sqrt_add_sq_le (a b : ℝ) :
    Real.cosh (Real.sqrt (a ^ 2 + b ^ 2)) ≤ Real.cosh a * Real.cosh b := by
  let f : ℝ → ℝ := fun s =>
    Real.log (Real.cosh a) + Real.log (Real.cosh (Real.sqrt s)) -
      Real.log (Real.cosh (Real.sqrt (a ^ 2 + s)))
  have h2t_le_sinh : ∀ y : ℝ, 0 ≤ y → y ≤ Real.sinh y := by
    intro y hy
    let k : ℝ → ℝ := fun z => Real.sinh z - z
    have hk_deriv : ∀ z : ℝ, HasDerivAt k (Real.cosh z - 1) z := fun z =>
      (Real.hasDerivAt_sinh z).sub (hasDerivAt_id z)
    have hk_mono : MonotoneOn k (Ici 0) := by
      refine monotoneOn_of_deriv_nonneg (convex_Ici 0)
        (fun z _ => (hk_deriv z).continuousAt.continuousWithinAt)
        (fun z _ => (hk_deriv z).differentiableAt.differentiableWithinAt) ?_
      intro z _
      rw [(hk_deriv z).deriv]
      linarith [Real.one_le_cosh z]
    have h0 : k 0 ≤ k y := hk_mono (mem_Ici.mpr le_rfl) (mem_Ici.mpr hy) hy
    dsimp [k] at h0
    rw [Real.sinh_zero, sub_zero] at h0
    linarith
  have hq_anti : AntitoneOn (fun t : ℝ => Real.tanh t / t) (Ioi 0) := by
    have hq_deriv : ∀ t : ℝ, 0 < t →
        HasDerivAt (fun z : ℝ => Real.tanh z / z)
          ((t / Real.cosh t ^ 2 - Real.tanh t) / t ^ 2) t := by
      intro t ht
      have hdtanh : HasDerivAt Real.tanh (1 / Real.cosh t ^ 2) t := by
        have heq : Real.tanh = fun z => Real.sinh z / Real.cosh z :=
          funext Real.tanh_eq_sinh_div_cosh
        rw [heq]
        have hd := (Real.hasDerivAt_sinh t).div (Real.hasDerivAt_cosh t)
          (Real.cosh_pos t).ne'
        exact hd.congr_deriv (by
          have hsq := Real.cosh_sq_sub_sinh_sq t
          field_simp [(Real.cosh_pos t).ne']
          linarith)
      have hdiv := hdtanh.div (hasDerivAt_id' t) ht.ne'
      exact hdiv.congr_deriv (by field_simp [ht.ne', (Real.cosh_pos t).ne'])
    refine antitoneOn_of_deriv_nonpos (convex_Ioi 0)
      (fun t ht => (hq_deriv t ht).continuousAt.continuousWithinAt)
      (fun t ht =>
        (hq_deriv t (by rwa [interior_Ioi] at ht)).differentiableAt.differentiableWithinAt) ?_
    intro t ht
    rw [interior_Ioi] at ht
    have ht0 : 0 < t := ht
    rw [(hq_deriv t ht0).deriv]
    apply div_nonpos_of_nonpos_of_nonneg _ (sq_nonneg t)
    rw [Real.tanh_eq_sinh_div_cosh]
    have hc_pos : 0 < Real.cosh t := Real.cosh_pos t
    have hnum_eq : t / Real.cosh t ^ 2 - Real.sinh t / Real.cosh t =
        (2 * t - Real.sinh (2 * t)) / (2 * Real.cosh t ^ 2) := by
      rw [Real.sinh_two_mul]
      field_simp [hc_pos.ne']
    rw [hnum_eq]
    apply div_nonpos_of_nonpos_of_nonneg _ (by positivity)
    linarith [h2t_le_sinh (2 * t) (by linarith)]
  by_cases ha0 : a = 0
  · subst ha0
    simp [Real.sqrt_sq_eq_abs, Real.cosh_abs]
  · have ha2_pos : 0 < a ^ 2 := sq_pos_of_ne_zero ha0
    have hf_cont : Continuous f := by
      dsimp [f]
      have hc1 : Continuous fun s : ℝ => Real.log (Real.cosh (Real.sqrt s)) :=
        (Real.continuous_cosh.comp Real.continuous_sqrt).log fun s => (Real.cosh_pos _).ne'
      have hc2 : Continuous fun s : ℝ => Real.log (Real.cosh (Real.sqrt (a ^ 2 + s))) :=
        (Real.continuous_cosh.comp
          (Real.continuous_sqrt.comp (continuous_const.add continuous_id))).log
          fun s => (Real.cosh_pos _).ne'
      exact (continuous_const.add hc1).sub hc2
    have hlog_cosh_sqrt_deriv : ∀ u : ℝ, 0 < u →
        HasDerivAt (fun z : ℝ => Real.log (Real.cosh (Real.sqrt z)))
          (Real.tanh (Real.sqrt u) / (2 * Real.sqrt u)) u := by
      intro u hu
      have hds := Real.hasDerivAt_sqrt hu.ne'
      have hdc : HasDerivAt (fun z : ℝ => Real.cosh (Real.sqrt z))
          (Real.sinh (Real.sqrt u) * (1 / (2 * Real.sqrt u))) u :=
        (Real.hasDerivAt_cosh (Real.sqrt u)).comp u hds
      have hdl : HasDerivAt (fun z : ℝ => Real.log (Real.cosh (Real.sqrt z)))
          (Real.sinh (Real.sqrt u) * (1 / (2 * Real.sqrt u)) / Real.cosh (Real.sqrt u)) u :=
        hdc.log (Real.cosh_pos _).ne'
      exact hdl.congr_deriv (by
        rw [Real.tanh_eq_sinh_div_cosh]
        field_simp [(Real.cosh_pos _).ne', (Real.sqrt_pos.mpr hu).ne'])
    have hf_deriv : ∀ s : ℝ, 0 < s →
        HasDerivAt f
          (Real.tanh (Real.sqrt s) / (2 * Real.sqrt s) -
            Real.tanh (Real.sqrt (a ^ 2 + s)) / (2 * Real.sqrt (a ^ 2 + s))) s := by
      intro s hs
      have hd1 := hlog_cosh_sqrt_deriv s hs
      have hd2_inner : HasDerivAt (fun z : ℝ => a ^ 2 + z) (0 + 1) s :=
        (hasDerivAt_const s (a ^ 2)).fun_add (hasDerivAt_id' s)
      have hd2 : HasDerivAt (fun z : ℝ => Real.log (Real.cosh (Real.sqrt (a ^ 2 + z))))
          (Real.tanh (Real.sqrt (a ^ 2 + s)) / (2 * Real.sqrt (a ^ 2 + s)) * (0 + 1)) s :=
        (hlog_cosh_sqrt_deriv (a ^ 2 + s) (by positivity)).comp s hd2_inner
      have hsum := ((hasDerivAt_const s (Real.log (Real.cosh a))).fun_add hd1).fun_sub hd2
      exact hsum.congr_deriv (by ring)
    have hf_mono : MonotoneOn f (Ici 0) := by
      refine monotoneOn_of_deriv_nonneg (convex_Ici 0) hf_cont.continuousOn
        (fun s hs =>
          (hf_deriv s (by rwa [interior_Ici] at hs)).differentiableAt.differentiableWithinAt) ?_
      intro s hs
      rw [interior_Ici] at hs
      have hs0 : 0 < s := hs
      rw [(hf_deriv s hs0).deriv]
      have h1 : 0 < Real.sqrt s := Real.sqrt_pos.mpr hs0
      have h2 : 0 < Real.sqrt (a ^ 2 + s) := Real.sqrt_pos.mpr (by positivity)
      have hle : Real.sqrt s ≤ Real.sqrt (a ^ 2 + s) := Real.sqrt_le_sqrt (by linarith)
      have hq := hq_anti h1 h2 hle
      have heq1 : Real.tanh (Real.sqrt s) / (2 * Real.sqrt s) =
          (1 / 2 : ℝ) * (Real.tanh (Real.sqrt s) / Real.sqrt s) := by ring
      have heq2 : Real.tanh (Real.sqrt (a ^ 2 + s)) / (2 * Real.sqrt (a ^ 2 + s)) =
          (1 / 2 : ℝ) * (Real.tanh (Real.sqrt (a ^ 2 + s)) / Real.sqrt (a ^ 2 + s)) := by ring
      rw [heq1, heq2]
      linarith
    have hf0 : f 0 = 0 := by
      dsimp [f]
      simp [Real.sqrt_sq_eq_abs, Real.cosh_abs]
    have hfb : 0 ≤ f (b ^ 2) := by
      rw [← hf0]
      exact hf_mono (mem_Ici.mpr le_rfl) (mem_Ici.mpr (sq_nonneg b)) (sq_nonneg b)
    dsimp [f] at hfb
    rw [Real.sqrt_sq_eq_abs, Real.cosh_abs,
      ← Real.log_mul (Real.cosh_pos a).ne' (Real.cosh_pos b).ne'] at hfb
    exact (Real.log_le_log_iff (Real.cosh_pos _)
      (mul_pos (Real.cosh_pos a) (Real.cosh_pos b))).mp (by linarith)

/-- **L2a (`lean_targets.md` Target L2, part 1 — pointwise metric ratio).**
For `ψ ∈ (-π, π)` and `w ∈ ℂ` with `Re w > 0`, `rotatedLog ψ` has derivative `w⁻¹` at `w` and
`‖deriv (rotatedLog ψ) w‖ / (1 / (2 * Re w)) = 2 * Re w / ‖w‖ ≤ 2`. -/
theorem L2_log_poincare_pointwise (ψ : ℝ) (w : ℂ)
    (hψ_lo : -Real.pi < ψ) (hψ_hi : ψ < Real.pi) (hw : 0 < w.re) :
    HasDerivAt (rotatedLog ψ) w⁻¹ w ∧
    ‖deriv (rotatedLog ψ) w‖ / (1 / (2 * w.re)) = 2 * w.re / ‖w‖ ∧
    2 * w.re / ‖w‖ ≤ 2 := by
  have hd := hasDerivAt_rotatedLog hψ_lo hψ_hi hw
  have hw_ne : w ≠ 0 := by intro h0; simp [h0] at hw
  have hwnorm_pos : 0 < ‖w‖ := norm_pos_iff.mpr hw_ne
  refine ⟨hd, ?_, ?_⟩
  · rw [hd.deriv, norm_inv]
    field_simp [hw.ne', hwnorm_pos.ne']
  · rw [div_le_iff₀ hwnorm_pos]
    linarith [Complex.re_le_norm w]

/-- **L2b (`lean_targets.md` Target L2, part 1 — `C¹` path integral domination).**
Along any `C¹` path `γ : [0, 1] → ℍ`, the displacement of `rotatedLog ψ` is at most twice the
hyperbolic path length `∫ t in 0..1, ‖γ'(t)‖ / (2 * Re(γ(t)))`. -/
theorem L2_log_poincare_path (ψ : ℝ) (γ γ' : ℝ → ℂ)
    (hψ_lo : -Real.pi < ψ) (hψ_hi : ψ < Real.pi)
    (hγ_re : ∀ t ∈ Icc (0 : ℝ) 1, 0 < (γ t).re)
    (hγ_deriv : ∀ t ∈ Icc (0 : ℝ) 1, HasDerivAt γ (γ' t) t)
    (hγ'_cont : ContinuousOn γ' (Icc (0 : ℝ) 1)) :
    ‖rotatedLog ψ (γ 1) - rotatedLog ψ (γ 0)‖ ≤
      2 * ∫ t in (0 : ℝ)..1, ‖γ' t‖ / (2 * (γ t).re) := by
  have hγ_cont : ContinuousOn γ (Icc (0 : ℝ) 1) :=
    fun t ht => (hγ_deriv t ht).continuousAt.continuousWithinAt
  have hγ_ne : ∀ t ∈ Icc (0 : ℝ) 1, γ t ≠ 0 := by
    intro t ht h0
    have hre := hγ_re t ht
    simp [h0] at hre
  have hcomp_deriv : ∀ t ∈ uIcc (0 : ℝ) 1,
      HasDerivAt (fun s => rotatedLog ψ (γ s)) ((γ t)⁻¹ * γ' t) t := by
    rw [uIcc_of_le zero_le_one]
    intro t ht
    exact (hasDerivAt_rotatedLog hψ_lo hψ_hi (hγ_re t ht)).comp t (hγ_deriv t ht)
  have hcomp_int : IntervalIntegrable (fun t => (γ t)⁻¹ * γ' t) volume 0 1 := by
    apply ContinuousOn.intervalIntegrable_of_Icc zero_le_one
    exact (hγ_cont.inv₀ hγ_ne).mul hγ'_cont
  have hftc : ∫ t in (0 : ℝ)..1, (γ t)⁻¹ * γ' t = rotatedLog ψ (γ 1) - rotatedLog ψ (γ 0) :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt hcomp_deriv hcomp_int
  have hrhs_cont : ContinuousOn (fun t : ℝ => 2 * (‖γ' t‖ / (2 * (γ t).re))) (Icc (0 : ℝ) 1) := by
    apply continuousOn_const.mul
    apply hγ'_cont.norm.div
    · exact continuousOn_const.mul (Complex.continuous_re.comp_continuousOn hγ_cont)
    · intro t ht
      exact (mul_pos two_pos (hγ_re t ht)).ne'
  have hrhs_int : IntervalIntegrable (fun t : ℝ => 2 * (‖γ' t‖ / (2 * (γ t).re))) volume 0 1 :=
    hrhs_cont.intervalIntegrable_of_Icc zero_le_one
  rw [← hftc, ← intervalIntegral.integral_const_mul]
  refine intervalIntegral.norm_integral_le_of_norm_le zero_le_one ?_ hrhs_int
  filter_upwards with t ht
  have ht' : t ∈ Icc (0 : ℝ) 1 := ⟨ht.1.le, ht.2⟩
  have hre_pos : 0 < (γ t).re := hγ_re t ht'
  have hre_le : (γ t).re ≤ ‖γ t‖ := Complex.re_le_norm (γ t)
  rw [norm_mul, norm_inv]
  have hsimp : 2 * (‖γ' t‖ / (2 * (γ t).re)) = ‖γ' t‖ / (γ t).re := by
    field_simp [hre_pos.ne']
  rw [hsimp, inv_mul_eq_div]
  exact div_le_div_of_nonneg_left (norm_nonneg _) hre_pos hre_le

/-- **L2c (`lean_targets.md` Target L2, part 1 — closed-form Poincaré distance domination).**
For `ψ ∈ (-π, π)` and any `w₁, w₂ ∈ ℍ = {Re w > 0}`,
`‖rotatedLog ψ w₁ - rotatedLog ψ w₂‖ ≤ 2 * poincareDistRHP w₁ w₂`. -/
theorem L2_log_poincare_dist (ψ : ℝ) (w₁ w₂ : ℂ)
    (hψ_lo : -Real.pi < ψ) (hψ_hi : ψ < Real.pi)
    (hw₁ : 0 < w₁.re) (hw₂ : 0 < w₂.re) :
    ‖rotatedLog ψ w₁ - rotatedLog ψ w₂‖ ≤ 2 * poincareDistRHP w₁ w₂ := by
  -- Step 1: `rotatedLog ψ w₁ - rotatedLog ψ w₂ = Complex.log w₁ - Complex.log w₂`
  -- by integrating the identical derivative along the segment in `ℍ`.
  let seg : ℝ → ℂ := fun t => (1 - (t : ℂ)) * w₂ + (t : ℂ) * w₁
  have hseg_re : ∀ t ∈ Icc (0 : ℝ) 1, 0 < (seg t).re := by
    intro t ht
    have hre : (seg t).re = (1 - t) * w₂.re + t * w₁.re := by simp [seg]
    rw [hre]
    rcases eq_or_lt_of_le ht.1 with rfl | ht0
    · simpa using hw₂
    · have h1 : 0 ≤ (1 - t) * w₂.re := mul_nonneg (by linarith [ht.2]) hw₂.le
      have h2 : 0 < t * w₁.re := mul_pos ht0 hw₁
      linarith
  have hseg_ne : ∀ t ∈ Icc (0 : ℝ) 1, seg t ≠ 0 := by
    intro t ht h0
    have hre := hseg_re t ht
    simp [h0] at hre
  have hseg_deriv : ∀ t : ℝ, HasDerivAt seg (w₁ - w₂) t := by
    intro t
    dsimp [seg]
    have hofr : HasDerivAt (fun s : ℝ => (s : ℂ)) 1 t :=
      (hasDerivAt_id' t).ofReal_comp
    have hd1 : HasDerivAt (fun s : ℝ => (1 - (s : ℂ)) * w₂) ((0 - 1) * w₂) t :=
      ((hasDerivAt_const t (1 : ℂ)).fun_sub hofr).mul_const w₂
    have hd2 : HasDerivAt (fun s : ℝ => (s : ℂ) * w₁) (1 * w₁) t :=
      hofr.mul_const w₁
    exact (hd1.fun_add hd2).congr_deriv (by ring)
  have hseg_cont : ContinuousOn seg (Icc (0 : ℝ) 1) :=
    fun t _ => (hseg_deriv t).continuousAt.continuousWithinAt
  have hint : IntervalIntegrable (fun t => (seg t)⁻¹ * (w₁ - w₂)) volume 0 1 :=
    ((hseg_cont.inv₀ hseg_ne).mul continuousOn_const).intervalIntegrable_of_Icc zero_le_one
  have hftc_ψ : ∫ t in (0 : ℝ)..1, (seg t)⁻¹ * (w₁ - w₂) = rotatedLog ψ w₁ - rotatedLog ψ w₂ := by
    have h := intervalIntegral.integral_eq_sub_of_hasDerivAt
      (f := fun s => rotatedLog ψ (seg s))
      (fun t ht => (hasDerivAt_rotatedLog hψ_lo hψ_hi
        (hseg_re t (by rwa [uIcc_of_le zero_le_one] at ht))).comp t (hseg_deriv t)) hint
    simpa [seg] using h
  have hftc_0 : ∫ t in (0 : ℝ)..1, (seg t)⁻¹ * (w₁ - w₂) =
      rotatedLog 0 w₁ - rotatedLog 0 w₂ := by
    have h := intervalIntegral.integral_eq_sub_of_hasDerivAt
      (f := fun s => rotatedLog 0 (seg s))
      (fun t ht => (hasDerivAt_rotatedLog (neg_neg_of_pos Real.pi_pos) Real.pi_pos
        (hseg_re t (by rwa [uIcc_of_le zero_le_one] at ht))).comp t (hseg_deriv t)) hint
    simpa [seg] using h
  have hrot0 : ∀ z : ℂ, rotatedLog 0 z = Complex.log z := by
    intro z
    simp [rotatedLog]
  have hdiff_eq : rotatedLog ψ w₁ - rotatedLog ψ w₂ = Complex.log w₁ - Complex.log w₂ := by
    rw [← hftc_ψ, hftc_0, hrot0, hrot0]
  rw [hdiff_eq]
  -- Step 2: Bound `‖Complex.log w₁ - Complex.log w₂‖` by `2 * poincareDistRHP w₁ w₂`.
  have hw₁_ne : w₁ ≠ 0 := by intro h0; simp [h0] at hw₁
  have hw₂_ne : w₂ ≠ 0 := by intro h0; simp [h0] at hw₂
  set r₁ : ℝ := ‖w₁‖
  set r₂ : ℝ := ‖w₂‖
  have hr₁ : 0 < r₁ := norm_pos_iff.mpr hw₁_ne
  have hr₂ : 0 < r₂ := norm_pos_iff.mpr hw₂_ne
  set θ₁ : ℝ := w₁.arg
  set θ₂ : ℝ := w₂.arg
  have hθ₁ : |θ₁| < Real.pi / 2 := Complex.abs_arg_lt_pi_div_two_iff.mpr (Or.inl hw₁)
  have hθ₂ : |θ₂| < Real.pi / 2 := Complex.abs_arg_lt_pi_div_two_iff.mpr (Or.inl hw₂)
  set a : ℝ := (Real.log r₁ - Real.log r₂) / 2
  set b : ℝ := |θ₁ - θ₂| / 2
  have hb0 : 0 ≤ b := by positivity
  have hb_lt : b < Real.pi / 2 := by
    have htri : |θ₁ - θ₂| ≤ |θ₁| + |θ₂| := by
      simpa [sub_eq_add_neg] using abs_add_le θ₁ (-θ₂)
    dsimp [b]
    linarith
  have hlog_norm : ‖Complex.log w₁ - Complex.log w₂‖ = 2 * Real.sqrt (a ^ 2 + b ^ 2) := by
    rw [Complex.norm_def, Complex.normSq_apply]
    simp only [Complex.sub_re, Complex.sub_im, Complex.log_re, Complex.log_im]
    have hsq_eq : (Real.log r₁ - Real.log r₂) * (Real.log r₁ - Real.log r₂) +
        (θ₁ - θ₂) * (θ₁ - θ₂) = (2 : ℝ) ^ 2 * (a ^ 2 + b ^ 2) := by
      have habs_sq : |θ₁ - θ₂| ^ 2 = (θ₁ - θ₂) ^ 2 := sq_abs (θ₁ - θ₂)
      dsimp [a, b]
      linear_combination -habs_sq
    change Real.sqrt ((Real.log r₁ - Real.log r₂) * (Real.log r₁ - Real.log r₂) +
      (θ₁ - θ₂) * (θ₁ - θ₂)) = 2 * Real.sqrt (a ^ 2 + b ^ 2)
    rw [hsq_eq, Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
  have hcos_b_pos : 0 < Real.cos b := Real.cos_pos_of_mem_Ioo ⟨by linarith, hb_lt⟩
  have hcosh_step1 : Real.cosh (2 * Real.sqrt (a ^ 2 + b ^ 2)) ≤
      2 * Real.cosh a ^ 2 * Real.cosh b ^ 2 - 1 := by
    have h1 := cosh_sqrt_add_sq_le a b
    have hnn : 0 ≤ Real.cosh (Real.sqrt (a ^ 2 + b ^ 2)) := (Real.cosh_pos _).le
    have hsq : Real.cosh (Real.sqrt (a ^ 2 + b ^ 2)) ^ 2 ≤ (Real.cosh a * Real.cosh b) ^ 2 :=
      pow_le_pow_left₀ hnn h1 2
    have hch_sq := Real.cosh_sq_sub_sinh_sq (Real.sqrt (a ^ 2 + b ^ 2))
    rw [Real.cosh_two_mul]
    nlinarith
  have hcosh_cos_le1 : Real.cosh b ^ 2 * Real.cos b ^ 2 ≤ 1 := by
    have hsc := sinh_mul_cos_le_sin hb0 hb_lt.le
    have hsc_sq : (Real.sinh b * Real.cos b) ^ 2 ≤ Real.sin b ^ 2 :=
      pow_le_pow_left₀ (mul_nonneg (Real.sinh_nonneg_iff.mpr hb0) hcos_b_pos.le) hsc 2
    have hch : Real.cosh b ^ 2 = 1 + Real.sinh b ^ 2 := by
      linarith [Real.cosh_sq_sub_sinh_sq b]
    have hcs : Real.cos b ^ 2 + Real.sin b ^ 2 = 1 := Real.cos_sq_add_sin_sq b
    nlinarith
  have hcosh_step2 : 2 * Real.cosh a ^ 2 * Real.cosh b ^ 2 - 1 ≤
      1 + (Real.cosh (2 * a) - Real.cos (2 * b)) / Real.cos b ^ 2 := by
    rw [← sub_nonneg]
    have hrew : 1 + (Real.cosh (2 * a) - Real.cos (2 * b)) / Real.cos b ^ 2 -
        (2 * Real.cosh a ^ 2 * Real.cosh b ^ 2 - 1) =
        2 * Real.cosh a ^ 2 * (1 - Real.cosh b ^ 2 * Real.cos b ^ 2) / Real.cos b ^ 2 := by
      have hch_a := Real.cosh_sq_sub_sinh_sq a
      have hcs_b := Real.cos_sq_add_sin_sq b
      rw [Real.cosh_two_mul, Real.cos_two_mul]
      field_simp [hcos_b_pos.ne']
      linear_combination -hch_a
    rw [hrew]
    exact div_nonneg (mul_nonneg (by positivity) (by linarith)) (by positivity)
  have hcos_2b : Real.cos (2 * b) = Real.cos (θ₁ - θ₂) := by
    have h2b : 2 * b = |θ₁ - θ₂| := by dsimp [b]; ring
    rw [h2b, Real.cos_abs]
  have hcos_prod_pos : 0 < Real.cos θ₁ * Real.cos θ₂ := by
    rw [Complex.cos_arg hw₁_ne, Complex.cos_arg hw₂_ne]
    positivity
  have hcos_prod_le : Real.cos θ₁ * Real.cos θ₂ ≤ Real.cos b ^ 2 := by
    have hprod : 2 * (Real.cos θ₁ * Real.cos θ₂) =
        Real.cos (θ₁ - θ₂) + Real.cos (θ₁ + θ₂) := by
      rw [Real.cos_sub, Real.cos_add]
      ring
    have h2b_sq : 2 * Real.cos b ^ 2 = Real.cos (2 * b) + 1 := by
      linarith [Real.cos_two_mul b]
    linarith [Real.cos_le_one (θ₁ + θ₂)]
  have hnum_nn : 0 ≤ Real.cosh (2 * a) - Real.cos (θ₁ - θ₂) := by
    linarith [Real.one_le_cosh (2 * a), Real.cos_le_one (θ₁ - θ₂)]
  have hcosh_step3 : 1 + (Real.cosh (2 * a) - Real.cos (2 * b)) / Real.cos b ^ 2 ≤
      1 + (Real.cosh (2 * a) - Real.cos (θ₁ - θ₂)) / (Real.cos θ₁ * Real.cos θ₂) := by
    rw [hcos_2b]
    gcongr
  have hpolar_ident :
      1 + (Real.cosh (2 * a) - Real.cos (θ₁ - θ₂)) / (Real.cos θ₁ * Real.cos θ₂) =
        1 + ‖w₁ - w₂‖ ^ 2 / (2 * w₁.re * w₂.re) := by
    have h2a : 2 * a = Real.log r₁ - Real.log r₂ := by dsimp [a]; ring
    have hexp_pos : Real.exp (2 * a) = r₁ / r₂ := by
      rw [h2a, Real.exp_sub, Real.exp_log hr₁, Real.exp_log hr₂]
    have hexp_neg : Real.exp (-(2 * a)) = r₂ / r₁ := by
      rw [Real.exp_neg, hexp_pos, inv_div]
    have hcosh_r : 2 * r₁ * r₂ * Real.cosh (2 * a) = r₁ ^ 2 + r₂ ^ 2 := by
      rw [Real.cosh_eq, hexp_pos, hexp_neg]
      field_simp [hr₁.ne', hr₂.ne']
    have hre1 : r₁ * Real.cos θ₁ = w₁.re := by
      rw [Complex.cos_arg hw₁_ne, mul_div_cancel₀ _ hr₁.ne']
    have hre2 : r₂ * Real.cos θ₂ = w₂.re := by
      rw [Complex.cos_arg hw₂_ne, mul_div_cancel₀ _ hr₂.ne']
    have him1 : r₁ * Real.sin θ₁ = w₁.im := by
      rw [Complex.sin_arg, mul_div_cancel₀ _ hr₁.ne']
    have him2 : r₂ * Real.sin θ₂ = w₂.im := by
      rw [Complex.sin_arg, mul_div_cancel₀ _ hr₂.ne']
    have hcos_r : 2 * r₁ * r₂ * Real.cos (θ₁ - θ₂) =
        2 * (w₁.re * w₂.re + w₁.im * w₂.im) := by
      rw [Real.cos_sub]
      calc 2 * r₁ * r₂ * (Real.cos θ₁ * Real.cos θ₂ + Real.sin θ₁ * Real.sin θ₂)
          = 2 * ((r₁ * Real.cos θ₁) * (r₂ * Real.cos θ₂) +
              (r₁ * Real.sin θ₁) * (r₂ * Real.sin θ₂)) := by ring
        _ = 2 * (w₁.re * w₂.re + w₁.im * w₂.im) := by rw [hre1, hre2, him1, him2]
    have hr1_sq : r₁ ^ 2 = w₁.re ^ 2 + w₁.im ^ 2 := by
      dsimp [r₁]
      rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
      ring
    have hr2_sq : r₂ ^ 2 = w₂.re ^ 2 + w₂.im ^ 2 := by
      dsimp [r₂]
      rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
      ring
    have hdiff_sq : ‖w₁ - w₂‖ ^ 2 = (w₁.re - w₂.re) ^ 2 + (w₁.im - w₂.im) ^ 2 := by
      rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
      simp only [Complex.sub_re, Complex.sub_im]
      ring
    have hden_r : 2 * r₁ * r₂ * (Real.cos θ₁ * Real.cos θ₂) = 2 * w₁.re * w₂.re := by
      calc 2 * r₁ * r₂ * (Real.cos θ₁ * Real.cos θ₂)
          = 2 * (r₁ * Real.cos θ₁) * (r₂ * Real.cos θ₂) := by ring
        _ = 2 * w₁.re * w₂.re := by rw [hre1, hre2]
    congr 1
    calc (Real.cosh (2 * a) - Real.cos (θ₁ - θ₂)) / (Real.cos θ₁ * Real.cos θ₂)
        = (2 * r₁ * r₂ * (Real.cosh (2 * a) - Real.cos (θ₁ - θ₂))) /
            (2 * r₁ * r₂ * (Real.cos θ₁ * Real.cos θ₂)) := by
          rw [mul_div_mul_left _ _ (by positivity : (2 * r₁ * r₂) ≠ 0)]
      _ = (2 * r₁ * r₂ * Real.cosh (2 * a) - 2 * r₁ * r₂ * Real.cos (θ₁ - θ₂)) /
            (2 * w₁.re * w₂.re) := by
          rw [hden_r]
          ring
      _ = ‖w₁ - w₂‖ ^ 2 / (2 * w₁.re * w₂.re) := by
          rw [hcosh_r, hcos_r, hr1_sq, hr2_sq, hdiff_sq]
          ring
  have hcosh_le : Real.cosh (2 * Real.sqrt (a ^ 2 + b ^ 2)) ≤
      1 + ‖w₁ - w₂‖ ^ 2 / (2 * w₁.re * w₂.re) := by
    linarith [hcosh_step1, hcosh_step2, hcosh_step3, hpolar_ident]
  have harcosh := (Real.arcosh_le_arcosh (Real.cosh_pos _) (by positivity)).mpr hcosh_le
  rw [Real.arcosh_cosh (by positivity)] at harcosh
  rw [hlog_norm]
  unfold poincareDistRHP
  linarith

/-- **L2d (`lean_targets.md` Target L2, part 2 — initial-step displacement bound from `u = 1`).**
For `s ∈ (0, 1)`, `|ψ| ≤ ψ₀ < π`, `w₀ = e^{-iψ/2}`, and `w₁ = s e^{-iψ/2} + b / w₀` with `b > 0`,
`1 + ‖w₁ - w₀‖² / (2 Re w₀ Re w₁) ≤ 1 + (1 + s + b)² / (2 s cos²(ψ₀ / 2))`. -/
theorem L2_initial_step (s b ψ ψ₀ : ℝ)
    (hs0 : 0 < s) (_hs1 : s < 1) (hb : 0 < b)
    (hψ : |ψ| ≤ ψ₀) (hψ₀ : ψ₀ < Real.pi) :
    let w₀ : ℂ := Complex.exp (-(((ψ / 2 : ℝ) : ℂ) * Complex.I))
    let w₁ : ℂ := (s : ℂ) * w₀ + (b : ℂ) / w₀
    1 + ‖w₁ - w₀‖ ^ 2 / (2 * w₀.re * w₁.re) ≤
      1 + (1 + s + b) ^ 2 / (2 * s * Real.cos (ψ₀ / 2) ^ 2) := by
  dsimp only
  set w₀ : ℂ := Complex.exp (-(((ψ / 2 : ℝ) : ℂ) * Complex.I))
  set w₁ : ℂ := (s : ℂ) * w₀ + (b : ℂ) / w₀
  have hψ₀_nn : 0 ≤ ψ₀ := (abs_nonneg ψ).trans hψ
  have hc₀_pos : 0 < Real.cos (ψ₀ / 2) :=
    Real.cos_pos_of_mem_Ioo ⟨by linarith, by linarith⟩
  have hc_ge : Real.cos (ψ₀ / 2) ≤ Real.cos (ψ / 2) := by
    have habs_div : |ψ / 2| = |ψ| / 2 := by rw [abs_div, abs_of_pos (two_pos : (0 : ℝ) < 2)]
    rw [← Real.cos_abs (ψ / 2), habs_div]
    exact Real.strictAntiOn_cos.antitoneOn
      ⟨by positivity, by linarith⟩ ⟨by positivity, by linarith⟩ (by linarith)
  have hw₀_norm : ‖w₀‖ = 1 := by
    dsimp [w₀]
    have hrew : -(((ψ / 2 : ℝ) : ℂ) * Complex.I) = ((-ψ / 2 : ℝ) : ℂ) * Complex.I := by
      push_cast
      ring
    rw [hrew, Complex.norm_exp_ofReal_mul_I]
  have hw₀_re : w₀.re = Real.cos (ψ / 2) := by
    dsimp [w₀]
    have hrew : -(((ψ / 2 : ℝ) : ℂ) * Complex.I) = ((-ψ / 2 : ℝ) : ℂ) * Complex.I := by
      push_cast
      ring
    rw [hrew, Complex.exp_ofReal_mul_I_re]
    have hneg : (-ψ / 2 : ℝ) = -(ψ / 2) := by ring
    rw [hneg, Real.cos_neg]
  have hw₁_re : w₁.re = (s + b) * Real.cos (ψ / 2) := by
    have hw₁_mob : w₁ = mobiusStep ((s : ℂ) * w₀) b w₀ := rfl
    rw [hw₁_mob, mobiusStep_re, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
      zero_mul, sub_zero, hw₀_norm, hw₀_re]
    ring
  have hc_pos : 0 < Real.cos (ψ / 2) := lt_of_lt_of_le hc₀_pos hc_ge
  have hden_ge : 2 * s * Real.cos (ψ₀ / 2) ^ 2 ≤ 2 * w₀.re * w₁.re := by
    rw [hw₀_re, hw₁_re]
    have hcs_sq : Real.cos (ψ₀ / 2) ^ 2 ≤ Real.cos (ψ / 2) ^ 2 :=
      pow_le_pow_left₀ hc₀_pos.le hc_ge 2
    nlinarith [mul_le_mul_of_nonneg_left hcs_sq (by positivity : 0 ≤ 2 * s),
      mul_pos hb (sq_pos_of_pos hc_pos)]
  have hnorm_le : ‖w₁ - w₀‖ ≤ 1 + s + b := by
    have h1 : ‖w₁ - w₀‖ ≤ ‖w₁‖ + ‖w₀‖ := norm_sub_le w₁ w₀
    have h2 : ‖w₁‖ ≤ ‖(s : ℂ) * w₀‖ + ‖(b : ℂ) / w₀‖ := norm_add_le _ _
    have hs_norm : ‖(s : ℂ) * w₀‖ = s := by
      rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg hs0.le, hw₀_norm, mul_one]
    have hb_norm : ‖(b : ℂ) / w₀‖ = b := by
      rw [norm_div, Complex.norm_real, Real.norm_of_nonneg hb.le, hw₀_norm, div_one]
    linarith [hw₀_norm]
  have hsq_le : ‖w₁ - w₀‖ ^ 2 ≤ (1 + s + b) ^ 2 :=
    pow_le_pow_left₀ (norm_nonneg _) hnorm_le 2
  have hdiv_le : ‖w₁ - w₀‖ ^ 2 / (2 * w₀.re * w₁.re) ≤
      (1 + s + b) ^ 2 / (2 * s * Real.cos (ψ₀ / 2) ^ 2) := by
    have hden_pos : 0 < 2 * w₀.re * w₁.re :=
      lt_of_lt_of_le (by positivity) hden_ge
    calc ‖w₁ - w₀‖ ^ 2 / (2 * w₀.re * w₁.re)
        ≤ (1 + s + b) ^ 2 / (2 * w₀.re * w₁.re) :=
          div_le_div_of_nonneg_right hsq_le hden_pos.le
      _ ≤ (1 + s + b) ^ 2 / (2 * s * Real.cos (ψ₀ / 2) ^ 2) :=
          div_le_div_of_nonneg_left (by positivity) (by positivity) hden_ge
  linarith

/-! ## Target L3: Kummer Moment Quadratic Identity and `E[ln T]` Upper Bound -/

/-- Unnormalized Kummer density `w_{q, x}(t) = t^{q - 1} * (1 + t)^{-q} * exp(-x t)` on `(0, ∞)`. -/
noncomputable def kummerWeight (q x t : ℝ) : ℝ :=
  t ^ (q - 1) * (1 + t) ^ (-q) * Real.exp (-x * t)

/-- Normalizing partition integral `Z_{q, x} = ∫ t in Ioi 0, w_{q, x}(t)`. -/
noncomputable def kummerZ (q x : ℝ) : ℝ :=
  ∫ t in Ioi (0 : ℝ), kummerWeight q x t

/-- Kummer expectation `E_{q, x}[g(T)] = (kummerZ q x)⁻¹ * ∫_{0}^∞ g(t) w_{q, x}(t) dt`. -/
noncomputable def kummerExpect (q x : ℝ) (g : ℝ → ℝ) : ℝ :=
  (kummerZ q x)⁻¹ * ∫ t in Ioi (0 : ℝ), g t * kummerWeight q x t

/-- Explicit quadratic upper-root bound `μ_+(q, x) = (1 - x + √((x - 1)² + 4 x q)) / (2 x)`. -/
noncomputable def kummerMuPlus (q x : ℝ) : ℝ :=
  (1 - x + Real.sqrt ((x - 1) ^ 2 + 4 * x * q)) / (2 * x)

private theorem integrableOn_rpow_kummerWeight {q x r : ℝ} (hq : 0 < q) (hx : 0 < x)
    (hqr : 0 < q + r) :
    IntegrableOn (fun t : ℝ => t ^ r * kummerWeight q x t) (Ioi 0) := by
  have hdom : IntegrableOn (fun t : ℝ => t ^ (q + r - 1) * Real.exp (-x * t)) (Ioi 0) := by
    simpa [Real.rpow_one] using
      (integrableOn_rpow_mul_exp_neg_mul_rpow (p := 1) (s := q + r - 1) (b := x)
        (by linarith) one_pos hx)
  have hcont : ContinuousOn (fun t : ℝ => t ^ r * kummerWeight q x t) (Ioi 0) := by
    unfold kummerWeight
    have hc_r : ContinuousOn (fun t : ℝ => t ^ r) (Ioi 0) :=
      continuousOn_id.rpow_const fun t ht => Or.inl (mem_Ioi.mp ht).ne'
    have hc_q1 : ContinuousOn (fun t : ℝ => t ^ (q - 1)) (Ioi 0) :=
      continuousOn_id.rpow_const fun t ht => Or.inl (mem_Ioi.mp ht).ne'
    have hc_1t : ContinuousOn (fun t : ℝ => (1 + t) ^ (-q)) (Ioi 0) :=
      (continuousOn_const.add continuousOn_id).rpow_const fun t ht =>
        Or.inl (by change 1 + t ≠ 0; linarith [mem_Ioi.mp ht])
    have hc_exp : ContinuousOn (fun t : ℝ => Real.exp (-x * t)) (Ioi 0) :=
      (continuous_const.mul continuous_id).rexp.continuousOn
    exact hc_r.mul ((hc_q1.mul hc_1t).mul hc_exp)
  refine hdom.mono' (hcont.aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
  have ht0 : 0 < t := ht
  have h1t : 1 ≤ 1 + t := by linarith
  have hpow_le : (1 + t) ^ (-q) ≤ 1 := by
    have hle := Real.rpow_le_rpow_of_exponent_le h1t (by linarith : -q ≤ 0)
    rwa [Real.rpow_zero] at hle
  unfold kummerWeight
  have hnn : 0 ≤ t ^ r * (t ^ (q - 1) * (1 + t) ^ (-q) * Real.exp (-x * t)) := by positivity
  rw [Real.norm_of_nonneg hnn]
  have hcomb : t ^ r * t ^ (q - 1) = t ^ (q + r - 1) := by
    rw [← Real.rpow_add ht0]
    congr 1
    ring
  calc t ^ r * (t ^ (q - 1) * (1 + t) ^ (-q) * Real.exp (-x * t))
      = (t ^ r * t ^ (q - 1)) * (1 + t) ^ (-q) * Real.exp (-x * t) := by ring
    _ = t ^ (q + r - 1) * (1 + t) ^ (-q) * Real.exp (-x * t) := by rw [hcomb]
    _ ≤ t ^ (q + r - 1) * 1 * Real.exp (-x * t) := by gcongr
    _ = t ^ (q + r - 1) * Real.exp (-x * t) := by ring

private theorem integrableOn_log_kummerWeight {q x : ℝ} (hq : 0 < q) (hx : 0 < x) :
    IntegrableOn (fun t : ℝ => Real.log t * kummerWeight q x t) (Ioi 0) := by
  -- Bound `|log t|` on `(0, 1]` by `2 / q * t ^ (-q / 2)` and on `(1, ∞)` by `t`.
  have hdom1 : IntegrableOn (fun t : ℝ => (2 / q) * (t ^ (-q / 2) * kummerWeight q x t)) (Ioi 0) :=
    (integrableOn_rpow_kummerWeight hq hx (by linarith)).const_mul (2 / q)
  have hdom2 : IntegrableOn (fun t : ℝ => t ^ (1 : ℝ) * kummerWeight q x t) (Ioi 0) :=
    integrableOn_rpow_kummerWeight hq hx (by linarith)
  have hdom := hdom1.add hdom2
  have hcont : ContinuousOn (fun t : ℝ => Real.log t * kummerWeight q x t) (Ioi 0) := by
    unfold kummerWeight
    have hc_log : ContinuousOn Real.log (Ioi 0) :=
      Real.continuousOn_log.mono fun t ht => (mem_Ioi.mp ht).ne'
    have hc_q1 : ContinuousOn (fun t : ℝ => t ^ (q - 1)) (Ioi 0) :=
      continuousOn_id.rpow_const fun t ht => Or.inl (mem_Ioi.mp ht).ne'
    have hc_1t : ContinuousOn (fun t : ℝ => (1 + t) ^ (-q)) (Ioi 0) :=
      (continuousOn_const.add continuousOn_id).rpow_const fun t ht =>
        Or.inl (by change 1 + t ≠ 0; linarith [mem_Ioi.mp ht])
    have hc_exp : ContinuousOn (fun t : ℝ => Real.exp (-x * t)) (Ioi 0) :=
      (continuous_const.mul continuous_id).rexp.continuousOn
    exact hc_log.mul ((hc_q1.mul hc_1t).mul hc_exp)
  refine hdom.mono' (hcont.aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
  have ht0 : 0 < t := ht
  have hw_nn : 0 ≤ kummerWeight q x t := by unfold kummerWeight; positivity
  rw [norm_mul, Real.norm_eq_abs, Real.norm_of_nonneg hw_nn]
  rcases le_or_gt t 1 with ht1 | ht1
  · -- On `(0, 1]`, `|log t| = -log t = (2/q) * log(t^{-q/2}) ≤ (2/q) * t^{-q/2}`.
    have hlog_np : Real.log t ≤ 0 := Real.log_nonpos ht0.le ht1
    have habs : |Real.log t| = (2 / q) * Real.log (t ^ (-q / 2)) := by
      rw [abs_of_nonpos hlog_np, Real.log_rpow ht0]
      field_simp [hq.ne']
    have hrpow_pos : 0 < t ^ (-q / 2) := Real.rpow_pos_of_pos ht0 _
    have hlog_le : Real.log (t ^ (-q / 2)) ≤ t ^ (-q / 2) := by
      linarith [Real.log_le_sub_one_of_pos hrpow_pos]
    have habs_le : |Real.log t| ≤ (2 / q) * t ^ (-q / 2) := by
      rw [habs]
      exact mul_le_mul_of_nonneg_left hlog_le (by positivity)
    have hterm2_nn : 0 ≤ t ^ (1 : ℝ) * kummerWeight q x t := by positivity
    calc |Real.log t| * kummerWeight q x t
        ≤ ((2 / q) * t ^ (-q / 2)) * kummerWeight q x t :=
          mul_le_mul_of_nonneg_right habs_le hw_nn
      _ = (2 / q) * (t ^ (-q / 2) * kummerWeight q x t) := by ring
      _ ≤ (2 / q) * (t ^ (-q / 2) * kummerWeight q x t) + t ^ (1 : ℝ) * kummerWeight q x t :=
          le_add_of_nonneg_right hterm2_nn
  · -- On `(1, ∞)`, `|log t| = log t ≤ t - 1 ≤ t = t ^ (1 : ℝ)`.
    have hlog_pos : 0 < Real.log t := Real.log_pos ht1
    have habs_le : |Real.log t| ≤ t ^ (1 : ℝ) := by
      rw [abs_of_pos hlog_pos, Real.rpow_one]
      linarith [Real.log_le_sub_one_of_pos ht0]
    have hterm1_nn : 0 ≤ (2 / q) * (t ^ (-q / 2) * kummerWeight q x t) := by positivity
    calc |Real.log t| * kummerWeight q x t
        ≤ t ^ (1 : ℝ) * kummerWeight q x t :=
          mul_le_mul_of_nonneg_right habs_le hw_nn
      _ ≤ (2 / q) * (t ^ (-q / 2) * kummerWeight q x t) + t ^ (1 : ℝ) * kummerWeight q x t :=
          le_add_of_nonneg_left hterm1_nn

private theorem kummerZ_pos {q x : ℝ} (hq : 0 < q) (hx : 0 < x) :
    0 < kummerZ q x := by
  unfold kummerZ
  have hint : IntegrableOn (kummerWeight q x) (Ioi 0) := by
    have h0 := integrableOn_rpow_kummerWeight (r := 0) hq hx (by linarith)
    refine h0.congr_fun (fun t _ => ?_) measurableSet_Ioi
    simp
  rw [setIntegral_pos_iff_support_of_nonneg_ae]
  · have hsub : Ioc (1 : ℝ) 2 ⊆ Function.support (kummerWeight q x) ∩ Ioi 0 := by
      intro t ht
      have ht0 : 0 < t := lt_trans zero_lt_one ht.1
      refine ⟨Function.mem_support.mpr ?_, ht0⟩
      unfold kummerWeight
      positivity
    refine lt_of_lt_of_le ?_ (measure_mono hsub)
    rw [Real.volume_Ioc]
    norm_num
  · filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have ht0 : 0 < t := ht
    unfold kummerWeight
    positivity
  · exact hint

/-- **L3a (`lean_targets.md` Target L3, parts 1 & 2 — exact IBP moment identity & strict bounds).**
For `q > 0` and `x > 0`:
1. `Z_{q, x} > 0`, and `1`, `T`, `T²`, `log T` are integrable against `w_{q, x}` on `(0, ∞)`;
2. `x * E_{q, x}[T²] + (x - 1) * E_{q, x}[T] = q`;
3. `E_{q, x}[T] ^ 2 < E_{q, x}[T²]`, `E_{q, x}[T] < μ_+(q, x)`, and
   `E_{q, x}[log T] < log μ_+(q, x)`. -/
theorem L3_kummer_moments (q x : ℝ) (hq : 0 < q) (hx : 0 < x) :
    0 < kummerZ q x ∧
    IntegrableOn (kummerWeight q x) (Ioi 0) ∧
    IntegrableOn (fun t => t * kummerWeight q x t) (Ioi 0) ∧
    IntegrableOn (fun t => t ^ 2 * kummerWeight q x t) (Ioi 0) ∧
    IntegrableOn (fun t => Real.log t * kummerWeight q x t) (Ioi 0) ∧
    x * kummerExpect q x (fun t => t ^ 2) + (x - 1) * kummerExpect q x (fun t => t) = q ∧
    (kummerExpect q x (fun t => t)) ^ 2 < kummerExpect q x (fun t => t ^ 2) ∧
    kummerExpect q x (fun t => t) < kummerMuPlus q x ∧
    kummerExpect q x Real.log < Real.log (kummerMuPlus q x) := by
  have hZ_pos : 0 < kummerZ q x := kummerZ_pos hq hx
  have hint0 : IntegrableOn (kummerWeight q x) (Ioi 0) := by
    have h := integrableOn_rpow_kummerWeight (r := 0) hq hx (by linarith)
    exact h.congr_fun (fun t _ => by simp) measurableSet_Ioi
  have hint1 : IntegrableOn (fun t => t * kummerWeight q x t) (Ioi 0) := by
    have h := integrableOn_rpow_kummerWeight (r := 1) hq hx (by linarith)
    exact h.congr_fun (fun t ht => by simp) measurableSet_Ioi
  have hint2 : IntegrableOn (fun t => t ^ 2 * kummerWeight q x t) (Ioi 0) := by
    have h := integrableOn_rpow_kummerWeight (r := (2 : ℝ)) hq hx (by linarith)
    exact h.congr_fun (fun t _ => by dsimp only; rw [Real.rpow_two]) measurableSet_Ioi
  have hint_log : IntegrableOn (fun t => Real.log t * kummerWeight q x t) (Ioi 0) :=
    integrableOn_log_kummerWeight hq hx
  -- Integration by parts on `f(t) = t ^ q * (1 + t) ^ (1 - q) * exp(-x t)` on `[0, ∞)`.
  let f : ℝ → ℝ := fun t => t ^ q * (1 + t) ^ (1 - q) * Real.exp (-x * t)
  let f' : ℝ → ℝ := fun t =>
    q * kummerWeight q x t - (x - 1) * (t * kummerWeight q x t) -
      x * (t ^ 2 * kummerWeight q x t)
  have hf0 : f 0 = 0 := by
    dsimp [f]
    rw [Real.zero_rpow hq.ne', zero_mul, zero_mul]
  have hf_cont : ContinuousWithinAt f (Ici 0) 0 := by
    dsimp [f]
    have hc1 : ContinuousWithinAt (fun t : ℝ => t ^ q) (Ici 0) 0 :=
      continuousWithinAt_id.rpow_const (Or.inr hq.le)
    have hc2 : ContinuousWithinAt (fun t : ℝ => (1 + t) ^ (1 - q)) (Ici 0) 0 :=
      ((continuousAt_const.add continuousAt_id).rpow_const
        (Or.inl (by norm_num))).continuousWithinAt
    have hc3 : ContinuousWithinAt (fun t : ℝ => Real.exp (-x * t)) (Ici 0) 0 :=
      (continuous_const.mul continuous_id).rexp.continuousAt.continuousWithinAt
    exact (hc1.mul hc2).mul hc3
  have hf_deriv : ∀ t ∈ Ioi (0 : ℝ), HasDerivAt f (f' t) t := by
    intro t ht
    have ht0 : 0 < t := ht
    have h1t : 0 < 1 + t := by linarith
    have hd1 : HasDerivAt (fun s : ℝ => s ^ q) (q * t ^ (q - 1)) t :=
      Real.hasDerivAt_rpow_const (Or.inl ht0.ne')
    have hd2_in : HasDerivAt (fun s : ℝ => 1 + s) (0 + 1) t :=
      (hasDerivAt_const t (1 : ℝ)).fun_add (hasDerivAt_id' t)
    have hd2 : HasDerivAt (fun s : ℝ => (1 + s) ^ (1 - q))
        ((1 - q) * (1 + t) ^ (1 - q - 1) * (0 + 1)) t :=
      (Real.hasDerivAt_rpow_const (Or.inl h1t.ne')).comp t hd2_in
    have hd3 : HasDerivAt (fun s : ℝ => Real.exp (-x * s)) (Real.exp (-x * t) * (-x * 1)) t :=
      ((hasDerivAt_id' t).const_mul (-x)).exp
    have hprod := (hd1.fun_mul hd2).fun_mul hd3
    refine hprod.congr_deriv ?_
    dsimp [f', kummerWeight]
    have hp_t1 : t ^ q = t * t ^ (q - 1) := by
      have := Real.rpow_add ht0 1 (q - 1)
      rw [Real.rpow_one] at this
      have heq : 1 + (q - 1) = q := by ring
      rw [heq] at this
      exact this
    have hp_u1 : (1 + t) ^ (1 - q) = (1 + t) * (1 + t) ^ (-q) := by
      have := Real.rpow_add h1t 1 (-q)
      rw [Real.rpow_one] at this
      have heq : 1 + -q = 1 - q := by ring
      rw [heq] at this
      exact this
    have hp_u2 : (1 + t) ^ (1 - q - 1) = (1 + t) ^ (-q) := by
      congr 1
      ring
    rw [hp_t1, hp_u1, hp_u2]
    ring
  have hf'_int : IntegrableOn f' (Ioi 0) := by
    dsimp [f']
    exact ((hint0.const_mul q).sub' (hint1.const_mul (x - 1))).sub' (hint2.const_mul x)
  have hf_tendsto : Tendsto f atTop (𝓝 0) := by
    have hdom_tendsto : Tendsto (fun t : ℝ => (2 : ℝ) * t ^ (q + 1) * Real.exp (-x * t))
        atTop (𝓝 0) := by
      have h0 := tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero (q + 1) x hx
      simpa [mul_assoc] using h0.const_mul (2 : ℝ)
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hdom_tendsto ?_ ?_
    · filter_upwards [Ioi_mem_atTop (1 : ℝ)] with t ht
      have ht0 : 0 < t := lt_trans zero_lt_one ht
      dsimp [f]
      positivity
    · filter_upwards [Ioi_mem_atTop (1 : ℝ)] with t ht
      have ht1 : 1 < t := ht
      have ht0 : 0 < t := lt_trans zero_lt_one ht1
      dsimp [f]
      have h1t_le : (1 + t) ^ (1 - q) ≤ 2 * t ^ (1 - q) := by
        rcases le_or_gt (1 - q) 0 with hq1 | hq1
        · have hle : t ≤ 1 + t := by linarith
          have hrp : (1 + t) ^ (-(1 - q)) ≥ t ^ (-(1 - q)) :=
            Real.rpow_le_rpow ht0.le hle (by linarith)
          have h1 : (1 + t) ^ (1 - q) = ((1 + t) ^ (-(1 - q)))⁻¹ := by
            rw [← Real.rpow_neg (by positivity), neg_neg]
          have h2 : t ^ (1 - q) = (t ^ (-(1 - q)))⁻¹ := by
            rw [← Real.rpow_neg ht0.le, neg_neg]
          have hinv : (1 + t) ^ (1 - q) ≤ t ^ (1 - q) := by
            rw [h1, h2]
            exact inv_anti₀ (Real.rpow_pos_of_pos ht0 _) hrp
          have hpos : 0 < t ^ (1 - q) := Real.rpow_pos_of_pos ht0 _
          linarith
        · have hle : 1 + t ≤ 2 * t := by linarith
          have hrp := Real.rpow_le_rpow (by positivity) hle hq1.le
          rw [Real.mul_rpow two_pos.le ht0.le] at hrp
          have h2_le : (2 : ℝ) ^ (1 - q) ≤ 2 := by
            have := Real.rpow_le_rpow_of_exponent_le (by norm_num : (1 : ℝ) ≤ 2)
              (by linarith : 1 - q ≤ 1)
            rwa [Real.rpow_one] at this
          exact hrp.trans (mul_le_mul_of_nonneg_right h2_le (Real.rpow_nonneg ht0.le _))
      have hpow_le : t ^ (1 : ℝ) ≤ t ^ (q + 1) :=
        Real.rpow_le_rpow_of_exponent_le ht1.le (by linarith)
      calc t ^ q * (1 + t) ^ (1 - q) * Real.exp (-x * t)
          ≤ t ^ q * (2 * t ^ (1 - q)) * Real.exp (-x * t) := by gcongr
        _ = 2 * (t ^ q * t ^ (1 - q)) * Real.exp (-x * t) := by ring
        _ = 2 * t ^ (1 : ℝ) * Real.exp (-x * t) := by
            have heq : q + (1 - q) = (1 : ℝ) := by ring
            rw [← Real.rpow_add ht0, heq]
        _ ≤ 2 * t ^ (q + 1) * Real.exp (-x * t) := by gcongr
  have hibp_raw : ∫ t in Ioi (0 : ℝ), f' t = 0 - f 0 :=
    integral_Ioi_of_hasDerivAt_of_tendsto hf_cont hf_deriv hf'_int hf_tendsto
  rw [hf0, sub_zero] at hibp_raw
  have hibp_split : ∫ t in Ioi (0 : ℝ), f' t =
      q * kummerZ q x - (x - 1) * (∫ t in Ioi (0 : ℝ), t * kummerWeight q x t) -
        x * (∫ t in Ioi (0 : ℝ), t ^ 2 * kummerWeight q x t) := by
    dsimp [f']
    rw [integral_sub ((hint0.const_mul q).sub' (hint1.const_mul (x - 1))) (hint2.const_mul x),
      integral_sub (hint0.const_mul q) (hint1.const_mul (x - 1)),
      integral_const_mul, integral_const_mul, integral_const_mul]
    rfl
  have hibp :
      x * kummerExpect q x (fun t => t ^ 2) + (x - 1) * kummerExpect q x (fun t => t) = q := by
    unfold kummerExpect
    have hZ_ne : kummerZ q x ≠ 0 := hZ_pos.ne'
    have hlin : q * kummerZ q x - (x - 1) * (∫ t in Ioi (0 : ℝ), t * kummerWeight q x t) -
        x * (∫ t in Ioi (0 : ℝ), t ^ 2 * kummerWeight q x t) = 0 := hibp_split ▸ hibp_raw
    field_simp [hZ_ne]
    linarith
  -- Normalization `E[1] = 1` and strict variance `E[(T - m₁)²] > 0`
  set m₁ : ℝ := kummerExpect q x (fun t => t)
  set m₂ : ℝ := kummerExpect q x (fun t => t ^ 2)
  have hm₁_pos : 0 < m₁ := by
    unfold m₁ kummerExpect
    apply mul_pos (inv_pos.mpr hZ_pos)
    rw [setIntegral_pos_iff_support_of_nonneg_ae]
    · have hsub : Ioc (1 : ℝ) 2 ⊆
          Function.support (fun t => t * kummerWeight q x t) ∩ Ioi 0 := by
        intro t ht
        have ht0 : 0 < t := lt_trans zero_lt_one ht.1
        refine ⟨Function.mem_support.mpr ?_, ht0⟩
        unfold kummerWeight
        positivity
      refine lt_of_lt_of_le ?_ (measure_mono hsub)
      rw [Real.volume_Ioc]
      norm_num
    · filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
      have ht0 : 0 < t := ht
      unfold kummerWeight
      positivity
    · exact hint1
  have hvar_pos : m₁ ^ 2 < m₂ := by
    let vfun : ℝ → ℝ := fun t => (t - m₁) ^ 2 * kummerWeight q x t
    have hv_int : IntegrableOn vfun (Ioi 0) := by
      have heq : vfun = fun t =>
          t ^ 2 * kummerWeight q x t - (2 * m₁) * (t * kummerWeight q x t) +
            m₁ ^ 2 * kummerWeight q x t := by
        ext t; dsimp [vfun]; ring
      rw [heq]
      exact (hint2.sub (hint1.const_mul (2 * m₁))).add (hint0.const_mul (m₁ ^ 2))
    have hv_pos : 0 < ∫ t in Ioi (0 : ℝ), vfun t := by
      rw [setIntegral_pos_iff_support_of_nonneg_ae]
      · have hsub : Ioc (m₁ + 1) (m₁ + 2) ⊆ Function.support vfun ∩ Ioi 0 := by
          intro t ht
          have ht0 : 0 < t := by linarith [ht.1]
          have hne : t - m₁ ≠ 0 := by linarith [ht.1]
          refine ⟨Function.mem_support.mpr ?_, ht0⟩
          dsimp [vfun]
          unfold kummerWeight
          exact mul_ne_zero (pow_ne_zero 2 hne) (by positivity)
        refine lt_of_lt_of_le ?_ (measure_mono hsub)
        rw [Real.volume_Ioc]
        norm_num
      · filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
        have ht0 : 0 < t := ht
        dsimp [vfun]
        unfold kummerWeight
        positivity
      · exact hv_int
    have hv_eval : (kummerZ q x)⁻¹ * ∫ t in Ioi (0 : ℝ), vfun t = m₂ - m₁ ^ 2 := by
      have hint_v : ∫ t in Ioi (0 : ℝ), vfun t =
          (∫ t in Ioi (0 : ℝ), t ^ 2 * kummerWeight q x t) -
            (2 * m₁) * (∫ t in Ioi (0 : ℝ), t * kummerWeight q x t) +
            m₁ ^ 2 * (∫ t in Ioi (0 : ℝ), kummerWeight q x t) := by
        have hv_eq : ∀ t : ℝ, vfun t =
            (t ^ 2 * kummerWeight q x t - (2 * m₁) * (t * kummerWeight q x t)) +
              m₁ ^ 2 * kummerWeight q x t := fun t => by dsimp [vfun]; ring
        simp_rw [hv_eq]
        rw [integral_add ((hint2.sub' (hint1.const_mul (2 * m₁))))
            (hint0.const_mul (m₁ ^ 2)),
          integral_sub hint2 (hint1.const_mul (2 * m₁)),
          integral_const_mul, integral_const_mul]
      rw [hint_v]
      dsimp [m₁, m₂, kummerExpect, kummerZ]
      have hZ_ne : (∫ t in Ioi (0 : ℝ), kummerWeight q x t) ≠ 0 := hZ_pos.ne'
      field_simp [hZ_ne]
      ring
    have hm₂_sub : 0 < m₂ - m₁ ^ 2 := by
      rw [← hv_eval]
      exact mul_pos (inv_pos.mpr hZ_pos) hv_pos
    linarith
  -- Deduce `m₁ < kummerMuPlus q x`
  have hm₁_lt_mu : m₁ < kummerMuPlus q x := by
    have hdisc : 0 ≤ (x - 1) ^ 2 + 4 * x * q := by positivity
    have hsqrt_sq := Real.sq_sqrt hdisc
    have hsqrt_nn := Real.sqrt_nonneg ((x - 1) ^ 2 + 4 * x * q)
    have hpoly : x * m₁ ^ 2 + (x - 1) * m₁ < q := by
      nlinarith [mul_lt_mul_of_pos_left hvar_pos hx]
    unfold kummerMuPlus
    rw [lt_div_iff₀ (by positivity : 0 < 2 * x)]
    by_contra hnot
    have hside : Real.sqrt ((x - 1) ^ 2 + 4 * x * q) ≤ 2 * x * m₁ + x - 1 := by linarith
    have hsq := mul_self_le_mul_self hsqrt_nn hside
    nlinarith [mul_pos hx (sub_pos.mpr hpoly)]
  -- Deduce `E[log T] ≤ log m₁ < log(kummerMuPlus q x)` via `log(t / m₁) ≤ t / m₁ - 1`
  have hlog_le : kummerExpect q x Real.log ≤ Real.log m₁ := by
    let jfun : ℝ → ℝ := fun t =>
      ((Real.log m₁ - 1) + m₁⁻¹ * t - Real.log t) * kummerWeight q x t
    have hj_int : IntegrableOn jfun (Ioi 0) := by
      have heq : jfun = fun t =>
          (Real.log m₁ - 1) * kummerWeight q x t + m₁⁻¹ * (t * kummerWeight q x t) -
            Real.log t * kummerWeight q x t := by
        ext t; dsimp [jfun]; ring
      rw [heq]
      exact ((hint0.const_mul _).add (hint1.const_mul _)).sub hint_log
    have hj_nn : 0 ≤ ∫ t in Ioi (0 : ℝ), jfun t := by
      refine setIntegral_nonneg measurableSet_Ioi fun t ht => ?_
      have ht0 : 0 < t := ht
      have hdiv_pos : 0 < t / m₁ := div_pos ht0 hm₁_pos
      have h1 := Real.log_le_sub_one_of_pos hdiv_pos
      rw [Real.log_div ht0.ne' hm₁_pos.ne'] at h1
      have h2 : 0 ≤ (Real.log m₁ - 1) + m₁⁻¹ * t - Real.log t := by
        have htm : t / m₁ = m₁⁻¹ * t := by ring
        linarith
      have hw_nn : 0 ≤ kummerWeight q x t := by unfold kummerWeight; positivity
      exact mul_nonneg h2 hw_nn
    have hj_eval : (kummerZ q x)⁻¹ * ∫ t in Ioi (0 : ℝ), jfun t =
        Real.log m₁ - kummerExpect q x Real.log := by
      have hint_j : ∫ t in Ioi (0 : ℝ), jfun t =
          (Real.log m₁ - 1) * kummerZ q x +
            m₁⁻¹ * (∫ t in Ioi (0 : ℝ), t * kummerWeight q x t) -
            (∫ t in Ioi (0 : ℝ), Real.log t * kummerWeight q x t) := by
        have hj_eq : ∀ t : ℝ, jfun t =
            ((Real.log m₁ - 1) * kummerWeight q x t - Real.log t * kummerWeight q x t) +
              m₁⁻¹ * (t * kummerWeight q x t) := fun t => by dsimp [jfun]; ring
        simp_rw [hj_eq]
        rw [integral_add ((hint0.const_mul _).sub' hint_log) (hint1.const_mul _),
          integral_sub (hint0.const_mul _) hint_log,
          integral_const_mul, integral_const_mul]
        unfold kummerZ
        ring
      rw [hint_j]
      have hm₁_def : m₁ = (kummerZ q x)⁻¹ * ∫ t in Ioi (0 : ℝ), t * kummerWeight q x t := rfl
      have hm₁_ne : m₁ ≠ 0 := hm₁_pos.ne'
      have hZ_ne : kummerZ q x ≠ 0 := hZ_pos.ne'
      have hint1_eq : ∫ t in Ioi (0 : ℝ), t * kummerWeight q x t = m₁ * kummerZ q x := by
        rw [hm₁_def]
        field_simp [hZ_ne]
      rw [hint1_eq]
      unfold kummerExpect
      field_simp [hZ_ne, hm₁_ne]
      ring
    have hdiff_nn : 0 ≤ Real.log m₁ - kummerExpect q x Real.log := by
      rw [← hj_eval]
      exact mul_nonneg (inv_nonneg.mpr hZ_pos.le) hj_nn
    linarith
  have hlog_strict : kummerExpect q x Real.log < Real.log (kummerMuPlus q x) :=
    lt_of_le_of_lt hlog_le (Real.log_lt_log hm₁_pos hm₁_lt_mu)
  exact ⟨hZ_pos, hint0, hint1, hint2, hint_log, hibp, hvar_pos, hm₁_lt_mu, hlog_strict⟩

/-- **L3b (`lean_targets.md` Target L3, part 3 — threshold contradiction limit).**
For every `s ∈ (0, 1)`, if `q_m → ∞` and `x_m / q_m → s² / (1 - s)`, then
`log s + log μ_+(q_m, x_m) → log(1 - s) < 0`. -/
theorem L3_threshold_limit (s : ℝ) (q_seq x_seq : ℕ → ℝ)
    (hs0 : 0 < s) (hs1 : s < 1)
    (hq_top : Tendsto q_seq atTop atTop)
    (hratio : Tendsto (fun m => x_seq m / q_seq m) atTop (𝓝 (s ^ 2 / (1 - s)))) :
    Tendsto (fun m => Real.log s + Real.log (kummerMuPlus (q_seq m) (x_seq m)))
      atTop (𝓝 (Real.log (1 - s))) ∧
    Real.log (1 - s) < 0 := by
  have h1s : 0 < 1 - s := by linarith
  set κ₀ : ℝ := s ^ 2 / (1 - s)
  have hκ₀_pos : 0 < κ₀ := div_pos (sq_pos_of_pos hs0) h1s
  set κ : ℕ → ℝ := fun m => x_seq m / q_seq m
  set ε : ℕ → ℝ := fun m => (q_seq m)⁻¹
  have hε_zero : Tendsto ε atTop (𝓝 0) := tendsto_inv_atTop_zero.comp hq_top
  -- Express `kummerMuPlus (q_seq m) (x_seq m)` for large `m` as
  -- `(ε m - κ m + √((κ m - ε m)² + 4 * κ m * ε m)) / (2 * κ m)`
  have hq_pos : ∀ᶠ m in atTop, 0 < q_seq m := hq_top.eventually_gt_atTop 0
  have hκ_pos : ∀ᶠ m in atTop, 0 < κ m := hratio.eventually (eventually_gt_nhds hκ₀_pos)
  let F : ℕ → ℝ := fun m =>
    (ε m - κ m + Real.sqrt ((κ m - ε m) ^ 2 + 4 * κ m)) / (2 * κ m)
  have hmu_eq : (fun m => kummerMuPlus (q_seq m) (x_seq m)) =ᶠ[atTop] F := by
    filter_upwards [hq_pos, hκ_pos] with m hqm hκm
    dsimp [kummerMuPlus, F, κ, ε]
    have hqm_ne : q_seq m ≠ 0 := hqm.ne'
    have hx_pos : 0 < x_seq m := by
      have : 0 < x_seq m / q_seq m := hκm
      exact (div_pos_iff_of_pos_right hqm).mp this
    have hrad : (x_seq m - 1) ^ 2 + 4 * x_seq m * q_seq m =
        (q_seq m) ^ 2 * ((x_seq m / q_seq m - (q_seq m)⁻¹) ^ 2 +
          4 * (x_seq m / q_seq m)) := by
      field_simp [hqm_ne]
    rw [hrad, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hqm.le]
    field_simp [hqm_ne, hx_pos.ne']
  have hrad_lim : Tendsto (fun m => (κ m - ε m) ^ 2 + 4 * κ m) atTop (𝓝 (κ₀ ^ 2 + 4 * κ₀)) := by
    have hsub := hratio.sub hε_zero
    simpa using (hsub.pow 2).add (hratio.const_mul 4)
  have hsqrt_val : Real.sqrt (κ₀ ^ 2 + 4 * κ₀) = s * (2 - s) / (1 - s) := by
    have h2s : 0 < 2 - s := by linarith
    have hnn : 0 ≤ s * (2 - s) / (1 - s) := div_nonneg (mul_nonneg hs0.le h2s.le) h1s.le
    have hsq : κ₀ ^ 2 + 4 * κ₀ = (s * (2 - s) / (1 - s)) ^ 2 := by
      dsimp [κ₀]
      field_simp [h1s.ne']
      ring
    rw [hsq, Real.sqrt_sq hnn]
  have hF_val : (0 - κ₀ + Real.sqrt (κ₀ ^ 2 + 4 * κ₀)) / (2 * κ₀) = (1 - s) / s := by
    rw [hsqrt_val]
    dsimp [κ₀]
    field_simp [hs0.ne', h1s.ne']
    ring
  have hF_lim : Tendsto F atTop (𝓝 ((1 - s) / s)) := by
    rw [← hF_val]
    refine Tendsto.div ?_ (hratio.const_mul 2) (by positivity)
    exact (hε_zero.sub hratio).add (Real.continuous_sqrt.continuousAt.tendsto.comp hrad_lim)
  have hmu_lim : Tendsto (fun m => kummerMuPlus (q_seq m) (x_seq m)) atTop (𝓝 ((1 - s) / s)) :=
    hF_lim.congr' hmu_eq.symm
  have hlim_pos : 0 < (1 - s) / s := div_pos h1s hs0
  have hlog_lim : Tendsto (fun m => Real.log s + Real.log (kummerMuPlus (q_seq m) (x_seq m)))
      atTop (𝓝 (Real.log s + Real.log ((1 - s) / s))) :=
    tendsto_const_nhds.add ((Real.continuousAt_log hlim_pos.ne').tendsto.comp hmu_lim)
  have hlog_eq : Real.log s + Real.log ((1 - s) / s) = Real.log (1 - s) := by
    rw [Real.log_div h1s.ne' hs0.ne']
    ring
  rw [hlog_eq] at hlog_lim
  refine ⟨hlog_lim, ?_⟩
  exact Real.log_neg h1s (by linarith)

/-! ## Target L4: Stokes Complex-Ratio Elimination Identity -/

/-- **L4 (`lean_targets.md` Target L4).**
For `A₀, A₁, B₀, B₁, r₀ ∈ ℝ` with `A₀ ≠ 0` and
`R = r₀ + (A₁ + i (π A₀ + B₁)) / (A₀ + i B₀) ∈ ℂ`:
1. `Re R - (B₀ / A₀) * Im R = r₀ + A₁ / A₀`;
2. If `R = 0`, then `r₀ + A₁ / A₀ = 0` independently of `B₀` and `B₁`. -/
theorem L4_stokes_elimination (A₀ A₁ B₀ B₁ r₀ : ℝ) (hA₀ : A₀ ≠ 0) :
    let R : ℂ := (r₀ : ℂ) +
      ((A₁ : ℂ) + ((Real.pi * A₀ + B₁ : ℝ) : ℂ) * Complex.I) /
        ((A₀ : ℂ) + (B₀ : ℂ) * Complex.I)
    R.re - (B₀ / A₀) * R.im = r₀ + A₁ / A₀ ∧
    (R = 0 → r₀ + A₁ / A₀ = 0) := by
  dsimp only
  set num : ℂ := (A₁ : ℂ) + ((Real.pi * A₀ + B₁ : ℝ) : ℂ) * Complex.I
  set den : ℂ := (A₀ : ℂ) + (B₀ : ℂ) * Complex.I
  set R : ℂ := (r₀ : ℂ) + num / den
  have hden_normSq : Complex.normSq den = A₀ ^ 2 + B₀ ^ 2 := by
    dsimp [den]
    rw [Complex.normSq_apply]
    simp only [Complex.add_re, Complex.add_im, Complex.ofReal_re, Complex.ofReal_im,
      Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im]
    ring
  have hden_pos : 0 < A₀ ^ 2 + B₀ ^ 2 :=
    add_pos_of_pos_of_nonneg (sq_pos_of_ne_zero hA₀) (sq_nonneg B₀)
  have hR_re : R.re = r₀ + (A₁ * A₀ + (Real.pi * A₀ + B₁) * B₀) / (A₀ ^ 2 + B₀ ^ 2) := by
    have hdiv_re : (num / den).re = (A₁ * A₀ + (Real.pi * A₀ + B₁) * B₀) / (A₀ ^ 2 + B₀ ^ 2) := by
      rw [Complex.div_re, hden_normSq]
      dsimp [num, den]
      simp only [Complex.ofReal_re, Complex.ofReal_im,
        Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im]
      ring
    have hadd_re : R.re = r₀ + (num / den).re := by
      dsimp [R]
    rw [hadd_re, hdiv_re]
  have hR_im : R.im = ((Real.pi * A₀ + B₁) * A₀ - A₁ * B₀) / (A₀ ^ 2 + B₀ ^ 2) := by
    have hdiv_im : (num / den).im = ((Real.pi * A₀ + B₁) * A₀ - A₁ * B₀) / (A₀ ^ 2 + B₀ ^ 2) := by
      rw [Complex.div_im, hden_normSq]
      dsimp [num, den]
      simp only [Complex.ofReal_re, Complex.ofReal_im,
        Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im]
      ring
    have hadd_im : R.im = (num / den).im := by
      dsimp [R]
      simp
    rw [hadd_im, hdiv_im]
  have hcomb : R.re - (B₀ / A₀) * R.im = r₀ + A₁ / A₀ := by
    rw [hR_re, hR_im]
    field_simp [hA₀, hden_pos.ne']
    ring
  refine ⟨hcomb, fun hR0 => ?_⟩
  rw [← hcomb, hR0]
  simp

/-! ## Target L5: Instanton Action Derivative Identity and Certified Borel Disk Bound -/

/-- Borel instanton action `A_gamma(s) = s (2 - s) / (1 - s) - 2 log(1 - s)` for i.i.d. Gamma. -/
noncomputable def gammaInstantonAction (s : ℝ) : ℝ :=
  s * (2 - s) / (1 - s) - 2 * Real.log (1 - s)

/-- Complex Laplace second derivative `Φ''(u) = (2 u + 1) / (u² (1 + u)²)` on the Borel plane. -/
noncomputable def laplacePhiSecondDeriv (u : ℂ) : ℂ :=
  (2 * u + 1) / (u ^ 2 * (1 + u) ^ 2)

/-- **L5a (`lean_targets.md` Target L5, part 1 — instanton action derivative and lower bound).**
`A_gamma(0) = 0`, and for every `s ∈ (0, 1)`:
1. `HasDerivAt gammaInstantonAction (((2 - s) / (1 - s)) ^ 2) s`;
2. `((2 - s) / (1 - s)) ^ 2 > 4`;
3. `gammaInstantonAction s > 4 * s > 0`. -/
theorem L5_instanton_action :
    gammaInstantonAction 0 = 0 ∧
    ∀ s : ℝ, 0 < s → s < 1 →
      HasDerivAt gammaInstantonAction (((2 - s) / (1 - s)) ^ 2) s ∧
      4 < ((2 - s) / (1 - s)) ^ 2 ∧
      4 * s < gammaInstantonAction s ∧
      0 < gammaInstantonAction s := by
  have h0 : gammaInstantonAction 0 = 0 := by
    unfold gammaInstantonAction
    simp
  have hderiv_on : ∀ t : ℝ, t < 1 →
      HasDerivAt gammaInstantonAction (((2 - t) / (1 - t)) ^ 2) t := by
    intro t ht1
    have h1t : 0 < 1 - t := by linarith
    unfold gammaInstantonAction
    have hd_num : HasDerivAt (fun x : ℝ => x * (2 - x)) (1 * (2 - t) + t * (0 - 1)) t :=
      (hasDerivAt_id' t).fun_mul ((hasDerivAt_const t (2 : ℝ)).fun_sub (hasDerivAt_id' t))
    have hd_den : HasDerivAt (fun x : ℝ => 1 - x) (0 - 1) t :=
      (hasDerivAt_const t (1 : ℝ)).fun_sub (hasDerivAt_id' t)
    have hd_div := hd_num.div hd_den h1t.ne'
    have hd_log := (hd_den.log h1t.ne').const_mul 2
    exact (hd_div.fun_sub hd_log).congr_deriv (by
      field_simp [h1t.ne']
      ring)
  have hsq_gt4 : ∀ t : ℝ, 0 < t → t < 1 → 4 < ((2 - t) / (1 - t)) ^ 2 := by
    intro t ht0 ht1
    have h1t : 0 < 1 - t := by linarith
    have hgt2 : 2 < (2 - t) / (1 - t) := by
      rw [lt_div_iff₀ h1t]
      linarith
    nlinarith
  refine ⟨h0, fun s hs0 hs1 => ?_⟩
  have hd_s := hderiv_on s hs1
  have hsq_s := hsq_gt4 s hs0 hs1
  -- Strict monotonicity of `A_gamma(t) - 4 * t` on `[0, s]`
  let g : ℝ → ℝ := fun t => gammaInstantonAction t - 4 * t
  have hg_deriv : ∀ t ∈ Icc (0 : ℝ) s, HasDerivAt g (((2 - t) / (1 - t)) ^ 2 - 4) t := by
    intro t ht
    have ht1 : t < 1 := lt_of_le_of_lt ht.2 hs1
    exact (hderiv_on t ht1).fun_sub
      (((hasDerivAt_id' t).const_mul 4).congr_deriv (by ring))
  have hg_strict : StrictMonoOn g (Icc 0 s) := by
    refine strictMonoOn_of_deriv_pos (convex_Icc 0 s)
      (fun t ht => (hg_deriv t ht).continuousAt.continuousWithinAt) ?_
    intro t ht
    rw [interior_Icc] at ht
    rw [(hg_deriv t ⟨ht.1.le, ht.2.le⟩).deriv]
    linarith [hsq_gt4 t ht.1 (lt_trans ht.2 hs1)]
  have hg0 : g 0 = 0 := by simp [g, h0]
  have hgs : 0 < g s := by
    rw [← hg0]
    exact hg_strict ⟨le_rfl, hs0.le⟩ ⟨hs0.le, le_rfl⟩ hs0
  dsimp [g] at hgs
  refine ⟨hd_s, hsq_s, by linarith, by linarith⟩

/-- Helper: bound `|arg z| ≤ arcsin(r / c)` when `‖z - c‖ ≤ r < c` for real `c > 0`, `r ≥ 0`. -/
private theorem abs_arg_le_arcsin_of_disk {z : ℂ} {c r : ℝ}
    (hc : 0 < c) (hr0 : 0 ≤ r) (hrc : r < c) (hz : ‖z - (c : ℂ)‖ ≤ r) :
    |z.arg| ≤ Real.arcsin (r / c) := by
  have hre_sub : |(z - (c : ℂ)).re| ≤ ‖z - (c : ℂ)‖ := Complex.abs_re_le_norm _
  have hre_pos : 0 < z.re := by
    have h1 : c - z.re ≤ |z.re - c| := by
      rw [abs_sub_comm]
      exact le_abs_self (c - z.re)
    have h2 : |z.re - c| ≤ r := by
      have : (z - (c : ℂ)).re = z.re - c := by simp
      linarith [this ▸ hre_sub]
    linarith
  have hz_ne : z ≠ 0 := by intro h0; simp [h0] at hre_pos
  have hznorm_pos : 0 < ‖z‖ := norm_pos_iff.mpr hz_ne
  rw [Complex.arg_of_re_nonneg hre_pos.le]
  have hasin_abs : |Real.arcsin (z.im / ‖z‖)| = Real.arcsin (|z.im| / ‖z‖) := by
    rcases le_or_gt 0 z.im with him | him
    · have hnn : 0 ≤ z.im / ‖z‖ := div_nonneg him hznorm_pos.le
      rw [abs_of_nonneg (Real.arcsin_nonneg.mpr hnn), abs_of_nonneg him]
    · have hanp : Real.arcsin (z.im / ‖z‖) ≤ 0 :=
        (Real.arcsin_lt_zero.mpr (div_neg_of_neg_of_pos him hznorm_pos)).le
      rw [abs_of_nonpos hanp, abs_of_neg him, neg_div, Real.arcsin_neg]
  rw [hasin_abs]
  apply Real.arcsin_le_arcsin
  rw [div_le_div_iff₀ hznorm_pos hc]
  -- Use Lagrange's identity `(x² + y²)((x - c)² + y²) = c² y² + (x² - c x + y²)²`
  have hdisk_sq : (z.re - c) ^ 2 + z.im ^ 2 ≤ r ^ 2 := by
    have hnsq : ‖z - (c : ℂ)‖ ^ 2 ≤ r ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hz 2
    rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply] at hnsq
    simp only [Complex.sub_re, Complex.sub_im, Complex.ofReal_re, Complex.ofReal_im,
      sub_zero] at hnsq
    linarith
  have hz_sq : ‖z‖ ^ 2 = z.re ^ 2 + z.im ^ 2 := by
    rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
    ring
  have hlagrange : (|z.im| * c) ^ 2 ≤ (r * ‖z‖) ^ 2 := by
    have him_sq : |z.im| ^ 2 = z.im ^ 2 := sq_abs z.im
    calc (|z.im| * c) ^ 2
        = c ^ 2 * z.im ^ 2 := by linear_combination c ^ 2 * him_sq
      _ ≤ c ^ 2 * z.im ^ 2 + (z.re ^ 2 - c * z.re + z.im ^ 2) ^ 2 :=
          le_add_of_nonneg_right (sq_nonneg _)
      _ = ((z.re - c) ^ 2 + z.im ^ 2) * (z.re ^ 2 + z.im ^ 2) := by ring
      _ ≤ r ^ 2 * (z.re ^ 2 + z.im ^ 2) :=
          mul_le_mul_of_nonneg_right hdisk_sq (by positivity)
      _ = (r * ‖z‖) ^ 2 := by rw [← hz_sq]; ring
  nlinarith [mul_nonneg (abs_nonneg z.im) hc.le, mul_nonneg hr0 hznorm_pos.le]

/-- **L5b (`lean_targets.md` Target L5, part 2 — certified Borel disk lower bound).**
For `t_+ > 0` and every `u ∈ ℂ` satisfying `‖u - t_+‖ ≤ t_+ / 5`,
`Re(Φ''(u)) ≥ cos(5 * arcsin(1 / 5)) * ((8 / 5) * t_+ + 1) /`
`(((6 / 5) * t_+) ^ 2 * (1 + (6 / 5) * t_+) ^ 2) > 0`. -/
theorem L5_borel_disk (t_pos : ℝ) (u : ℂ)
    (ht : 0 < t_pos) (hu : ‖u - (t_pos : ℂ)‖ ≤ t_pos / 5) :
    let bound : ℝ := Real.cos (5 * Real.arcsin (1 / 5)) *
      ((8 / 5 : ℝ) * t_pos + 1) /
      (((6 / 5 : ℝ) * t_pos) ^ 2 * (1 + (6 / 5 : ℝ) * t_pos) ^ 2)
    0 < bound ∧ bound ≤ (laplacePhiSecondDeriv u).re := by
  dsimp only
  -- First establish `5 * arcsin(1 / 5) < π / 2` so `cos(5 * arcsin(1 / 5)) > 0`.
  have hasin_pos : 0 < Real.arcsin (1 / 5) :=
    Real.arcsin_pos.mpr (by norm_num)
  have hasin_lt : Real.arcsin (1 / 5) < (21 / 100 : ℝ) := by
    have h21_lt : (21 / 100 : ℝ) < Real.pi / 2 := by linarith [Real.pi_gt_three]
    rw [Real.arcsin_lt_iff_lt_sin ⟨by norm_num, by norm_num⟩ ⟨by linarith, h21_lt.le⟩]
    have hcube := Real.sin_gt_sub_cube (by norm_num : (0 : ℝ) < 21 / 100)
    linarith
  have h5asin_lt : 5 * Real.arcsin (1 / 5) < Real.pi / 2 := by
    linarith [Real.pi_gt_three]
  have hcos5_pos : 0 < Real.cos (5 * Real.arcsin (1 / 5)) :=
    Real.cos_pos_of_mem_Ioo ⟨by linarith, h5asin_lt⟩
  have hbound_pos : 0 < Real.cos (5 * Real.arcsin (1 / 5)) *
      ((8 / 5 : ℝ) * t_pos + 1) /
      (((6 / 5 : ℝ) * t_pos) ^ 2 * (1 + (6 / 5 : ℝ) * t_pos) ^ 2) := by
    positivity
  refine ⟨hbound_pos, ?_⟩
  -- Norm bounds on `u`, `1 + u`, and `2 u + 1`
  have hu_lo : (4 / 5 : ℝ) * t_pos ≤ ‖u‖ := by
    have htri : ‖(t_pos : ℂ)‖ - ‖u - (t_pos : ℂ)‖ ≤ ‖u‖ := by
      have h := norm_sub_norm_le (t_pos : ℂ) (t_pos - u : ℂ)
      have heq : (t_pos : ℂ) - ((t_pos : ℂ) - u) = u := by ring
      rw [heq, norm_sub_rev] at h
      exact h
    rw [Complex.norm_real, Real.norm_of_nonneg ht.le] at htri
    linarith
  have hu_hi : ‖u‖ ≤ (6 / 5 : ℝ) * t_pos := by
    have htri : ‖u‖ - ‖(t_pos : ℂ)‖ ≤ ‖u - (t_pos : ℂ)‖ := norm_sub_norm_le u (t_pos : ℂ)
    rw [Complex.norm_real, Real.norm_of_nonneg ht.le] at htri
    linarith
  have h1u_sub : ‖(1 + u) - ((1 + t_pos : ℝ) : ℂ)‖ ≤ t_pos / 5 := by
    have heq : (1 + u) - ((1 + t_pos : ℝ) : ℂ) = u - (t_pos : ℂ) := by push_cast; ring
    rwa [heq]
  have h1u_hi : ‖1 + u‖ ≤ 1 + (6 / 5 : ℝ) * t_pos := by
    have htri : ‖1 + u‖ - ‖((1 + t_pos : ℝ) : ℂ)‖ ≤ ‖(1 + u) - ((1 + t_pos : ℝ) : ℂ)‖ :=
      norm_sub_norm_le (1 + u) ((1 + t_pos : ℝ) : ℂ)
    rw [Complex.norm_real, Real.norm_of_nonneg (by positivity)] at htri
    linarith
  have h2u1_sub : ‖(2 * u + 1) - ((2 * t_pos + 1 : ℝ) : ℂ)‖ ≤ (2 / 5 : ℝ) * t_pos := by
    have heq : (2 * u + 1) - ((2 * t_pos + 1 : ℝ) : ℂ) = (2 : ℂ) * (u - (t_pos : ℂ)) := by
      push_cast; ring
    rw [heq, norm_mul, Complex.norm_ofNat]
    linarith
  have h2u1_lo : (8 / 5 : ℝ) * t_pos + 1 ≤ ‖2 * u + 1‖ := by
    have htri : ‖((2 * t_pos + 1 : ℝ) : ℂ)‖ - ‖(2 * u + 1) - ((2 * t_pos + 1 : ℝ) : ℂ)‖ ≤
        ‖2 * u + 1‖ := by
      have h := norm_sub_norm_le ((2 * t_pos + 1 : ℝ) : ℂ) (((2 * t_pos + 1 : ℝ) : ℂ) - (2 * u + 1))
      have heq : ((2 * t_pos + 1 : ℝ) : ℂ) - (((2 * t_pos + 1 : ℝ) : ℂ) - (2 * u + 1)) =
          2 * u + 1 := by ring
      rw [heq, norm_sub_rev] at h
      exact h
    rw [Complex.norm_real, Real.norm_of_nonneg (by positivity)] at htri
    linarith
  have hu_ne : u ≠ 0 := norm_pos_iff.mp (lt_of_lt_of_le (by positivity) hu_lo)
  have h1u_pos : 0 < ‖1 + u‖ := by
    have htri : ‖((1 + t_pos : ℝ) : ℂ)‖ - ‖(1 + u) - ((1 + t_pos : ℝ) : ℂ)‖ ≤ ‖1 + u‖ := by
      have h := norm_sub_norm_le ((1 + t_pos : ℝ) : ℂ) (((1 + t_pos : ℝ) : ℂ) - (1 + u))
      have heq : ((1 + t_pos : ℝ) : ℂ) - (((1 + t_pos : ℝ) : ℂ) - (1 + u)) = 1 + u := by ring
      rw [heq, norm_sub_rev] at h
      exact h
    rw [Complex.norm_real, Real.norm_of_nonneg (by positivity)] at htri
    linarith
  have h1u_ne : 1 + u ≠ 0 := norm_pos_iff.mp h1u_pos
  have h2u1_ne : 2 * u + 1 ≠ 0 := norm_pos_iff.mp (lt_of_lt_of_le (by positivity) h2u1_lo)
  -- Argument bounds on each factor
  have harg_u : |u.arg| ≤ Real.arcsin (1 / 5) := by
    have h := abs_arg_le_arcsin_of_disk ht (by positivity) (by linarith) hu
    have hrat : (t_pos / 5) / t_pos = (1 / 5 : ℝ) := by field_simp [ht.ne']
    rwa [hrat] at h
  have harg_1u : |(1 + u).arg| ≤ Real.arcsin (1 / 5) := by
    have h := abs_arg_le_arcsin_of_disk (c := 1 + t_pos) (r := t_pos / 5)
      (by positivity) (by positivity) (by linarith) h1u_sub
    have hle : (t_pos / 5) / (1 + t_pos) ≤ 1 / 5 := by
      rw [div_le_div_iff₀ (by positivity) (by positivity)]
      linarith
    exact h.trans (Real.arcsin_le_arcsin hle)
  have harg_2u1 : |(2 * u + 1).arg| ≤ Real.arcsin (1 / 5) := by
    have h := abs_arg_le_arcsin_of_disk (c := 2 * t_pos + 1) (r := (2 / 5 : ℝ) * t_pos)
      (by positivity) (by positivity) (by linarith) h2u1_sub
    have hle : ((2 / 5 : ℝ) * t_pos) / (2 * t_pos + 1) ≤ 1 / 5 := by
      rw [div_le_div_iff₀ (by positivity) (by positivity)]
      linarith
    exact h.trans (Real.arcsin_le_arcsin hle)
  -- Polar representation of `Φ''(u)`:
  -- Write `θ := (2 * u + 1).arg - 2 * u.arg - 2 * (1 + u).arg`.
  -- Since `|θ| ≤ 5 * arcsin(1 / 5) < π / 2`, `Φ''(u)` has modulus `‖2u+1‖ / (‖u‖² ‖1+u‖²)`
  -- and real part `‖Φ''(u)‖ * cos θ`!
  set θ : ℝ := (2 * u + 1).arg - 2 * u.arg - 2 * (1 + u).arg
  have hθ_abs : |θ| ≤ 5 * Real.arcsin (1 / 5) := by
    have h1 : |θ| ≤ |(2 * u + 1).arg - 2 * u.arg| + |2 * (1 + u).arg| := by
      dsimp [θ]
      simpa [sub_eq_add_neg] using abs_add_le ((2 * u + 1).arg - 2 * u.arg) (-(2 * (1 + u).arg))
    have h2 : |(2 * u + 1).arg - 2 * u.arg| ≤ |(2 * u + 1).arg| + |2 * u.arg| := by
      simpa [sub_eq_add_neg] using abs_add_le (2 * u + 1).arg (-(2 * u.arg))
    have h3 : |2 * u.arg| = 2 * |u.arg| := by rw [abs_mul, abs_of_pos (two_pos : (0 : ℝ) < 2)]
    have h4 : |2 * (1 + u).arg| = 2 * |(1 + u).arg| := by
      rw [abs_mul, abs_of_pos (two_pos : (0 : ℝ) < 2)]
    linarith
  have hpolar_u : u = (‖u‖ : ℂ) * Complex.exp ((u.arg : ℂ) * Complex.I) :=
    (Complex.norm_mul_exp_arg_mul_I u).symm
  have hpolar_1u : 1 + u = (‖1 + u‖ : ℂ) * Complex.exp (((1 + u).arg : ℂ) * Complex.I) :=
    (Complex.norm_mul_exp_arg_mul_I (1 + u)).symm
  have hpolar_2u1 : 2 * u + 1 =
      (‖2 * u + 1‖ : ℂ) * Complex.exp (((2 * u + 1).arg : ℂ) * Complex.I) :=
    (Complex.norm_mul_exp_arg_mul_I (2 * u + 1)).symm
  set M_phi : ℝ := ‖2 * u + 1‖ / (‖u‖ ^ 2 * ‖1 + u‖ ^ 2)
  have hphi_polar : laplacePhiSecondDeriv u =
      (M_phi : ℂ) * Complex.exp ((θ : ℂ) * Complex.I) := by
    unfold laplacePhiSecondDeriv
    have hexp_u2 : u ^ 2 = (‖u‖ ^ 2 : ℂ) * Complex.exp (((2 * u.arg : ℝ) : ℂ) * Complex.I) := by
      conv_lhs => rw [hpolar_u]
      rw [mul_pow, ← Complex.exp_nat_mul]
      push_cast
      ring_nf
    have hexp_1u2 : (1 + u) ^ 2 =
        (‖1 + u‖ ^ 2 : ℂ) * Complex.exp (((2 * (1 + u).arg : ℝ) : ℂ) * Complex.I) := by
      conv_lhs => rw [hpolar_1u]
      rw [mul_pow, ← Complex.exp_nat_mul]
      push_cast
      ring_nf
    conv_lhs => rw [hpolar_2u1, hexp_u2, hexp_1u2]
    have hden_exp : ((‖u‖ ^ 2 : ℂ) * Complex.exp (((2 * u.arg : ℝ) : ℂ) * Complex.I)) *
        ((‖1 + u‖ ^ 2 : ℂ) * Complex.exp (((2 * (1 + u).arg : ℝ) : ℂ) * Complex.I)) =
        ((‖u‖ ^ 2 * ‖1 + u‖ ^ 2 : ℝ) : ℂ) *
          Complex.exp (((2 * u.arg + 2 * (1 + u).arg : ℝ) : ℂ) * Complex.I) := by
      push_cast
      rw [mul_mul_mul_comm, ← Complex.exp_add]
      congr 1
      ring_nf
    rw [hden_exp]
    have hden_scalar_ne : ((‖u‖ ^ 2 * ‖1 + u‖ ^ 2 : ℝ) : ℂ) ≠ 0 := by
      exact_mod_cast (mul_pos (sq_pos_of_pos (norm_pos_iff.mpr hu_ne)) (sq_pos_of_pos h1u_pos)).ne'
    have hexp_ne : Complex.exp (((2 * u.arg + 2 * (1 + u).arg : ℝ) : ℂ) * Complex.I) ≠ 0 :=
      Complex.exp_ne_zero _
    have hM_cast : (M_phi : ℂ) = (‖2 * u + 1‖ : ℂ) / ((‖u‖ ^ 2 * ‖1 + u‖ ^ 2 : ℝ) : ℂ) := by
      dsimp [M_phi]
      push_cast
      rfl
    have hexp_quot : Complex.exp (((2 * u + 1).arg : ℂ) * Complex.I) /
        Complex.exp (((2 * u.arg + 2 * (1 + u).arg : ℝ) : ℂ) * Complex.I) =
        Complex.exp ((θ : ℂ) * Complex.I) := by
      rw [← Complex.exp_sub]
      congr 1
      dsimp [θ]
      push_cast
      ring
    rw [hM_cast, ← hexp_quot]
    field_simp [hden_scalar_ne, hexp_ne]
  have hphi_re : (laplacePhiSecondDeriv u).re = M_phi * Real.cos θ := by
    rw [hphi_polar, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
      Complex.exp_ofReal_mul_I_re, zero_mul, sub_zero]
  have hcos_ge : Real.cos (5 * Real.arcsin (1 / 5)) ≤ Real.cos θ := by
    rw [← Real.cos_abs θ]
    exact Real.strictAntiOn_cos.antitoneOn
      ⟨abs_nonneg θ, le_trans hθ_abs (by linarith [Real.pi_pos])⟩
      ⟨by positivity, by linarith [Real.pi_pos]⟩ hθ_abs
  have hM_ge : ((8 / 5 : ℝ) * t_pos + 1) /
      (((6 / 5 : ℝ) * t_pos) ^ 2 * (1 + (6 / 5 : ℝ) * t_pos) ^ 2) ≤ M_phi := by
    dsimp [M_phi]
    have hden_le : ‖u‖ ^ 2 * ‖1 + u‖ ^ 2 ≤
        ((6 / 5 : ℝ) * t_pos) ^ 2 * (1 + (6 / 5 : ℝ) * t_pos) ^ 2 :=
      mul_le_mul (pow_le_pow_left₀ (norm_nonneg _) hu_hi 2)
        (pow_le_pow_left₀ (norm_nonneg _) h1u_hi 2) (by positivity) (by positivity)
    have hden_pos : 0 < ‖u‖ ^ 2 * ‖1 + u‖ ^ 2 :=
      mul_pos (sq_pos_of_pos (norm_pos_iff.mpr hu_ne)) (sq_pos_of_pos h1u_pos)
    calc ((8 / 5 : ℝ) * t_pos + 1) /
            (((6 / 5 : ℝ) * t_pos) ^ 2 * (1 + (6 / 5 : ℝ) * t_pos) ^ 2)
        ≤ ((8 / 5 : ℝ) * t_pos + 1) / (‖u‖ ^ 2 * ‖1 + u‖ ^ 2) :=
          div_le_div_of_nonneg_left (by positivity) hden_pos hden_le
      _ ≤ ‖2 * u + 1‖ / (‖u‖ ^ 2 * ‖1 + u‖ ^ 2) :=
          div_le_div_of_nonneg_right h2u1_lo hden_pos.le
  rw [hphi_re]
  calc Real.cos (5 * Real.arcsin (1 / 5)) *
          ((8 / 5 : ℝ) * t_pos + 1) /
          (((6 / 5 : ℝ) * t_pos) ^ 2 * (1 + (6 / 5 : ℝ) * t_pos) ^ 2)
      = Real.cos (5 * Real.arcsin (1 / 5)) *
          (((8 / 5 : ℝ) * t_pos + 1) /
            (((6 / 5 : ℝ) * t_pos) ^ 2 * (1 + (6 / 5 : ℝ) * t_pos) ^ 2)) := by ring
    _ ≤ Real.cos θ * M_phi :=
        mul_le_mul hcos_ge hM_ge (by positivity) (by linarith [hcos5_pos])
    _ = M_phi * Real.cos θ := mul_comm _ _

/-! ## Target L6: Three-Step Positive Matrix Log-`c` Slope and Perturbation Diagonals -/

/-- One-step `2 × 2` Roerdink companion matrix `[[s, c * h], [1, 0]]`. -/
def roerdinkMat (s c h : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![s, c * h; 1, 0]

/-- Three-step companion matrix product `M(c) = A(h₂) * A(h₁) * A(h₀)`. -/
def roerdinkThreeStep (s c h₀ h₁ h₂ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  roerdinkMat s c h₂ * roerdinkMat s c h₁ * roerdinkMat s c h₀

/-- **L6a (`lean_targets.md` Target L6, part 1 — three-step matrix product & log-`c` elasticity).**
For `s > 0`, `c > 0`, and `h₀, h₁, h₂ > 0`:
1. `roerdinkThreeStep s c h₀ h₁ h₂ = !![s³ + s c (h₂ + h₁), c h₀ (s² + c h₂); s² + c h₁, s c h₀]`;
2. Every entry `(i, j) ∈ Fin 2 × Fin 2` is strictly positive and differentiable in `c`, with
   `c * deriv (fun u => roerdinkThreeStep s u h₀ h₁ h₂ i j) c / roerdinkThreeStep s c h₀ h₁ h₂ i j`
   `≥ c * min h₁ h₂ / (s ^ 2 + c * min h₁ h₂) > 0`. -/
theorem L6_three_step_elasticity (s c h₀ h₁ h₂ : ℝ)
    (hs : 0 < s) (hc : 0 < c) (hh₀ : 0 < h₀) (hh₁ : 0 < h₁) (hh₂ : 0 < h₂) :
    let hmin : ℝ := min h₁ h₂
    let β : ℝ := c * hmin / (s ^ 2 + c * hmin)
    roerdinkThreeStep s c h₀ h₁ h₂ =
      !![s ^ 3 + s * c * (h₂ + h₁), c * h₀ * (s ^ 2 + c * h₂);
         s ^ 2 + c * h₁, s * c * h₀] ∧
    0 < β ∧
    ∀ i j : Fin 2,
      0 < roerdinkThreeStep s c h₀ h₁ h₂ i j ∧
      DifferentiableAt ℝ (fun u => roerdinkThreeStep s u h₀ h₁ h₂ i j) c ∧
      β ≤ c * deriv (fun u => roerdinkThreeStep s u h₀ h₁ h₂ i j) c /
        roerdinkThreeStep s c h₀ h₁ h₂ i j := by
  dsimp only
  have hm_eq : ∀ u : ℝ, roerdinkThreeStep s u h₀ h₁ h₂ =
      !![s ^ 3 + s * u * (h₂ + h₁), u * h₀ * (s ^ 2 + u * h₂);
         s ^ 2 + u * h₁, s * u * h₀] := by
    intro u
    unfold roerdinkThreeStep roerdinkMat
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [Matrix.mul_apply, Fin.sum_univ_two] <;> ring
  set hmin : ℝ := min h₁ h₂
  have hhmin_pos : 0 < hmin := lt_min hh₁ hh₂
  set β : ℝ := c * hmin / (s ^ 2 + c * hmin)
  have hβ_pos : 0 < β := by dsimp [β]; positivity
  have hhmin_le1 : hmin ≤ h₁ := min_le_left h₁ h₂
  have hhmin_le2 : hmin ≤ h₂ := min_le_right h₁ h₂
  refine ⟨hm_eq c, hβ_pos, ?_⟩
  have h00 : 0 < roerdinkThreeStep s c h₀ h₁ h₂ 0 0 ∧
      DifferentiableAt ℝ (fun u => roerdinkThreeStep s u h₀ h₁ h₂ 0 0) c ∧
      β ≤ c * deriv (fun u => roerdinkThreeStep s u h₀ h₁ h₂ 0 0) c /
        roerdinkThreeStep s c h₀ h₁ h₂ 0 0 := by
    have hentry : ∀ u : ℝ, roerdinkThreeStep s u h₀ h₁ h₂ 0 0 = s ^ 3 + s * u * (h₂ + h₁) := by
      intro u; rw [hm_eq u]; rfl
    simp_rw [hentry]
    have hpos : 0 < s ^ 3 + s * c * (h₂ + h₁) := by positivity
    have hd : HasDerivAt (fun u : ℝ => s ^ 3 + s * u * (h₂ + h₁)) (s * (h₂ + h₁)) c := by
      have h1 : HasDerivAt (fun u : ℝ => s * u * (h₂ + h₁)) (s * 1 * (h₂ + h₁)) c :=
        ((hasDerivAt_id' c).const_mul s).mul_const (h₂ + h₁)
      exact ((hasDerivAt_const c (s ^ 3)).fun_add h1).congr_deriv (by ring)
    refine ⟨hpos, hd.differentiableAt, ?_⟩
    rw [hd.deriv]
    dsimp [β]
    rw [div_le_div_iff₀ (by positivity) hpos]
    have hle : hmin ≤ h₂ + h₁ := by linarith
    nlinarith [mul_le_mul_of_nonneg_left hle (by positivity : 0 ≤ s ^ 3 * c)]
  have h01 : 0 < roerdinkThreeStep s c h₀ h₁ h₂ 0 1 ∧
      DifferentiableAt ℝ (fun u => roerdinkThreeStep s u h₀ h₁ h₂ 0 1) c ∧
      β ≤ c * deriv (fun u => roerdinkThreeStep s u h₀ h₁ h₂ 0 1) c /
        roerdinkThreeStep s c h₀ h₁ h₂ 0 1 := by
    have hentry : ∀ u : ℝ, roerdinkThreeStep s u h₀ h₁ h₂ 0 1 = u * h₀ * (s ^ 2 + u * h₂) := by
      intro u; rw [hm_eq u]; rfl
    simp_rw [hentry]
    have hpos : 0 < c * h₀ * (s ^ 2 + c * h₂) := by positivity
    have hd : HasDerivAt (fun u : ℝ => u * h₀ * (s ^ 2 + u * h₂))
        (h₀ * (s ^ 2 + 2 * c * h₂)) c := by
      have h1 : HasDerivAt (fun u : ℝ => u * h₀) (1 * h₀) c :=
        (hasDerivAt_id' c).mul_const h₀
      have h2 : HasDerivAt (fun u : ℝ => s ^ 2 + u * h₂) (0 + 1 * h₂) c :=
        (hasDerivAt_const c (s ^ 2)).fun_add ((hasDerivAt_id' c).mul_const h₂)
      exact (h1.fun_mul h2).congr_deriv (by ring)
    refine ⟨hpos, hd.differentiableAt, ?_⟩
    rw [hd.deriv]
    dsimp [β]
    have hβ_lt1 : c * hmin / (s ^ 2 + c * hmin) ≤ 1 :=
      div_le_one_of_le₀ (by linarith [sq_nonneg s]) (by positivity)
    have hel_ge1 : 1 ≤ c * (h₀ * (s ^ 2 + 2 * c * h₂)) / (c * h₀ * (s ^ 2 + c * h₂)) := by
      rw [le_div_iff₀ hpos]
      nlinarith [mul_pos (mul_pos hc hh₀) (mul_pos hc hh₂)]
    linarith
  have h10 : 0 < roerdinkThreeStep s c h₀ h₁ h₂ 1 0 ∧
      DifferentiableAt ℝ (fun u => roerdinkThreeStep s u h₀ h₁ h₂ 1 0) c ∧
      β ≤ c * deriv (fun u => roerdinkThreeStep s u h₀ h₁ h₂ 1 0) c /
        roerdinkThreeStep s c h₀ h₁ h₂ 1 0 := by
    have hentry : ∀ u : ℝ, roerdinkThreeStep s u h₀ h₁ h₂ 1 0 = s ^ 2 + u * h₁ := by
      intro u; rw [hm_eq u]; rfl
    simp_rw [hentry]
    have hpos : 0 < s ^ 2 + c * h₁ := by positivity
    have hd : HasDerivAt (fun u : ℝ => s ^ 2 + u * h₁) h₁ c := by
      have h1 := (hasDerivAt_const c (s ^ 2)).fun_add ((hasDerivAt_id' c).mul_const h₁)
      exact h1.congr_deriv (by ring)
    refine ⟨hpos, hd.differentiableAt, ?_⟩
    rw [hd.deriv]
    dsimp [β]
    rw [div_le_div_iff₀ (by positivity) hpos]
    nlinarith [mul_le_mul_of_nonneg_left hhmin_le1 (by positivity : 0 ≤ s ^ 2 * c)]
  have h11 : 0 < roerdinkThreeStep s c h₀ h₁ h₂ 1 1 ∧
      DifferentiableAt ℝ (fun u => roerdinkThreeStep s u h₀ h₁ h₂ 1 1) c ∧
      β ≤ c * deriv (fun u => roerdinkThreeStep s u h₀ h₁ h₂ 1 1) c /
        roerdinkThreeStep s c h₀ h₁ h₂ 1 1 := by
    have hentry : ∀ u : ℝ, roerdinkThreeStep s u h₀ h₁ h₂ 1 1 = s * u * h₀ := by
      intro u; rw [hm_eq u]; rfl
    simp_rw [hentry]
    have hpos : 0 < s * c * h₀ := by positivity
    have hd : HasDerivAt (fun u : ℝ => s * u * h₀) (s * h₀) c := by
      have h1 := ((hasDerivAt_id' c).const_mul s).mul_const h₀
      exact h1.congr_deriv (by ring)
    refine ⟨hpos, hd.differentiableAt, ?_⟩
    rw [hd.deriv]
    dsimp [β]
    have hβ_lt1 : c * hmin / (s ^ 2 + c * hmin) ≤ 1 :=
      div_le_one_of_le₀ (by linarith [sq_nonneg s]) (by positivity)
    have hel_eq1 : c * (s * h₀) / (s * c * h₀) = 1 := by
      field_simp [hs.ne', hc.ne', hh₀.ne']
    linarith
  intro i j
  match i, j with
  | ⟨0, _⟩, ⟨0, _⟩ => exact h00
  | ⟨0, _⟩, ⟨1, _⟩ => exact h01
  | ⟨1, _⟩, ⟨0, _⟩ => exact h10
  | ⟨1, _⟩, ⟨1, _⟩ => exact h11

/-- **L6b (`lean_targets.md` Target L6, part 2 — perturbation recursion diagonal invertibility).**
For every `s ∈ (0, 1)` (`a = 1 - s ∈ (0, 1)`), `φ ∈ (-1, 1)`, `k ≥ 1`, and `j ≥ 0`:
1. `0 < a` (`(k, j) = (1, 0)` threshold normalization denominator);
2. `1 - (-a) ^ k ≥ 1 - a = s > 0`, hence `1 - (-a) ^ k ≠ 0`;
3. `1 - (-a) ^ k * φ ^ j ≥ 1 - a * |φ| ^ j > 0`, hence `1 - (-a) ^ k * φ ^ j ≠ 0`. -/
theorem L6_perturbation_diag (s φ : ℝ) (k j : ℕ)
    (hs0 : 0 < s) (hs1 : s < 1) (hφ_lo : -1 < φ) (hφ_hi : φ < 1) (hk : 1 ≤ k) :
    let a : ℝ := 1 - s
    0 < a ∧
    1 - a = s ∧
    1 - a ≤ 1 - (-a) ^ k ∧
    0 < 1 - (-a) ^ k ∧
    1 - (-a) ^ k ≠ 0 ∧
    1 - a * |φ| ^ j ≤ 1 - (-a) ^ k * φ ^ j ∧
    0 < 1 - a * |φ| ^ j ∧
    1 - (-a) ^ k * φ ^ j ≠ 0 := by
  dsimp only
  set a : ℝ := 1 - s
  have ha0 : 0 < a := by dsimp [a]; linarith
  have ha1 : a < 1 := by dsimp [a]; linarith
  have h1a : 1 - a = s := by dsimp [a]; ring
  have hak_le : |(-a) ^ k| ≤ a := by
    rw [abs_pow, abs_neg, abs_of_pos ha0]
    calc a ^ k ≤ a ^ 1 := pow_le_pow_of_le_one ha0.le ha1.le hk
      _ = a := pow_one a
  have hneg_ak_le : (-a) ^ k ≤ a := (le_abs_self _).trans hak_le
  have h1_le : 1 - a ≤ 1 - (-a) ^ k := by linarith
  have h1_pos : 0 < 1 - (-a) ^ k := by linarith
  have hφ_abs_lt1 : |φ| < 1 := abs_lt.mpr ⟨hφ_lo, hφ_hi⟩
  have hφj_le1 : |φ| ^ j ≤ 1 := pow_le_one₀ (abs_nonneg φ) hφ_abs_lt1.le
  have hprod_le : (-a) ^ k * φ ^ j ≤ a * |φ| ^ j := by
    calc (-a) ^ k * φ ^ j
        ≤ |(-a) ^ k * φ ^ j| := le_abs_self _
      _ = |(-a) ^ k| * |φ| ^ j := by rw [abs_mul, abs_pow φ j]
      _ ≤ a * |φ| ^ j := mul_le_mul_of_nonneg_right hak_le (by positivity)
  have h2_le : 1 - a * |φ| ^ j ≤ 1 - (-a) ^ k * φ ^ j := by linarith
  have haφ_lt1 : a * |φ| ^ j < 1 := by
    calc a * |φ| ^ j ≤ a * 1 := mul_le_mul_of_nonneg_left hφj_le1 ha0.le
      _ = a := mul_one a
      _ < 1 := ha1
  have h2_pos : 0 < 1 - a * |φ| ^ j := by linarith
  have h2_ne : 1 - (-a) ^ k * φ ^ j ≠ 0 := by linarith
  exact ⟨ha0, h1a, h1_le, h1_pos, h1_pos.ne', h2_le, h2_pos, h2_ne⟩

end Dobson.NoiseAnalyticity

#print axioms Dobson.NoiseAnalyticity.L1_mobius_contraction
#print axioms Dobson.NoiseAnalyticity.L2_log_poincare_pointwise
#print axioms Dobson.NoiseAnalyticity.L2_log_poincare_path
#print axioms Dobson.NoiseAnalyticity.L2_log_poincare_dist
#print axioms Dobson.NoiseAnalyticity.L2_initial_step
#print axioms Dobson.NoiseAnalyticity.L3_kummer_moments
#print axioms Dobson.NoiseAnalyticity.L3_threshold_limit
#print axioms Dobson.NoiseAnalyticity.L4_stokes_elimination
#print axioms Dobson.NoiseAnalyticity.L5_instanton_action
#print axioms Dobson.NoiseAnalyticity.L5_borel_disk
#print axioms Dobson.NoiseAnalyticity.L6_three_step_elasticity
#print axioms Dobson.NoiseAnalyticity.L6_perturbation_diag
