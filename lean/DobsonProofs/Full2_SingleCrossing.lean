import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.Convex.Deriv
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.Tactic

/-!
`prop:exact-P2`: the strict covariance inequality, including integration,
centering, nonconcentration and exponential rescaling.

`TextbookReflection` is the ordinary Lebesgue change-of-variables formula
splitting the real line at an arbitrary point and reflecting its left half.
It is quantified over ALL integrable real functions, and contains no kernel,
crossing condition, probability law, or conclusion specific to this paper.
It is a named textbook hypothesis, not an axiom. No stochastic or
project-specific assertion is hidden in it.
-/

namespace Dobson.Full2SingleCrossing
open MeasureTheory Filter Set

/- The established kernel calculus is reproduced in this namespace to avoid
loading the existing JensenGap module's whole-Mathlib import on a shared machine. -/
namespace Kernel
noncomputable def B (x : ℝ) : ℝ := 2 * Real.sinh x / x - Real.cosh x - 1

/-- `h(y) = (2 − y)·e^y − 2 − y`. -/
noncomputable def h (y : ℝ) : ℝ := (2 - y) * Real.exp y - 2 - y

/-- Helper: `g(x) = 2·sinh x − x·cosh x − x`, which is `x·B(x)`. -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sinh x - x * Real.cosh x - x

/-! ## A1 and A2: the even part -/

/-- The derivative of `sinh t − t·cosh t` is `−t·sinh t`. -/
theorem hasDerivAt_sinh_sub_mul_cosh (x : ℝ) :
    HasDerivAt (fun t : ℝ => Real.sinh t - t * Real.cosh t) (-(x * Real.sinh x)) x := by
  have h1 : HasDerivAt (fun t : ℝ => t * Real.cosh t) (1 * Real.cosh x + x * Real.sinh x) x :=
    (hasDerivAt_id' x).fun_mul (Real.hasDerivAt_cosh x)
  have h2 := (Real.hasDerivAt_sinh x).fun_sub h1
  have e : Real.cosh x - (1 * Real.cosh x + x * Real.sinh x) = -(x * Real.sinh x) := by ring
  rw [e] at h2
  exact h2

/-- For `x > 0`: `sinh x − x·cosh x < 0` (that is, `tanh x < x`). -/
theorem sinh_sub_mul_cosh_neg {x : ℝ} (hx : 0 < x) : Real.sinh x - x * Real.cosh x < 0 := by
  have hanti : StrictAntiOn (fun t : ℝ => Real.sinh t - t * Real.cosh t) (Set.Ici 0) := by
    refine strictAntiOn_of_deriv_neg (convex_Ici 0) ?_ ?_
    · exact fun t _ => (hasDerivAt_sinh_sub_mul_cosh t).continuousAt.continuousWithinAt
    · intro t ht
      rw [interior_Ici] at ht
      have ht0 : 0 < t := ht
      rw [(hasDerivAt_sinh_sub_mul_cosh t).deriv]
      have hpos : 0 < t * Real.sinh t := mul_pos ht0 (Real.sinh_pos_iff.mpr ht0)
      linarith
  have key := hanti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx.le) hx
  simpa using key

/-- The derivative of `2·(cosh t − 1) − t·sinh t` is `sinh t − t·cosh t`. -/
theorem hasDerivAt_even_part (x : ℝ) :
    HasDerivAt (fun t : ℝ => 2 * (Real.cosh t - 1) - t * Real.sinh t)
      (Real.sinh x - x * Real.cosh x) x := by
  have h1 : HasDerivAt (fun t : ℝ => 2 * (Real.cosh t - 1)) (2 * Real.sinh x) x :=
    ((Real.hasDerivAt_cosh x).sub_const 1).const_mul 2
  have h2 : HasDerivAt (fun t : ℝ => t * Real.sinh t) (1 * Real.sinh x + x * Real.cosh x) x :=
    (hasDerivAt_id' x).fun_mul (Real.hasDerivAt_sinh x)
  have h3 := h1.fun_sub h2
  have e : 2 * Real.sinh x - (1 * Real.sinh x + x * Real.cosh x)
      = Real.sinh x - x * Real.cosh x := by ring
  rw [e] at h3
  exact h3

/-- **A1.** For `x > 0`: `2·(cosh x − 1) − x·sinh x < 0`. -/
theorem A1 (x : ℝ) (hx : 0 < x) : 2 * (Real.cosh x - 1) - x * Real.sinh x < 0 := by
  have hanti :
      StrictAntiOn (fun t : ℝ => 2 * (Real.cosh t - 1) - t * Real.sinh t) (Set.Ici 0) := by
    refine strictAntiOn_of_deriv_neg (convex_Ici 0) ?_ ?_
    · exact fun t _ => (hasDerivAt_even_part t).continuousAt.continuousWithinAt
    · intro t ht
      rw [interior_Ici] at ht
      rw [(hasDerivAt_even_part t).deriv]
      exact sinh_sub_mul_cosh_neg ht
  have key := hanti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx.le) hx
  simpa using key

/-- **A2.** For every real `x`: `2·(cosh x − 1) − x·sinh x ≤ 0`, with equality iff `x = 0`. -/
theorem A2 (x : ℝ) :
    2 * (Real.cosh x - 1) - x * Real.sinh x ≤ 0
      ∧ (2 * (Real.cosh x - 1) - x * Real.sinh x = 0 ↔ x = 0) := by
  rcases lt_trichotomy x 0 with hx | hx | hx
  · -- the function is even
    have h1 := A1 (-x) (neg_pos.mpr hx)
    rw [Real.cosh_neg, Real.sinh_neg] at h1
    have e : -x * -Real.sinh x = x * Real.sinh x := by ring
    rw [e] at h1
    exact ⟨h1.le, fun h0 => absurd h0 h1.ne, fun h0 => absurd h0 hx.ne⟩
  · subst hx
    simp
  · have h1 := A1 x hx
    exact ⟨h1.le, fun h0 => absurd h0 h1.ne, fun h0 => absurd h0 hx.ne'⟩

/-! ## B1 and B2: the odd part divided by `x` -/

/-- `g(0) = 0`. -/
theorem g_zero : g 0 = 0 := by
  simp [g]

/-- For `x ≠ 0`: `B(x) = g(x)/x`. -/
theorem B_eq_g_div (x : ℝ) (hx : x ≠ 0) : B x = g x / x := by
  unfold B g
  field_simp

/-- The derivative of `g` is `cosh x − x·sinh x − 1`. -/
theorem hasDerivAt_g (x : ℝ) : HasDerivAt g (Real.cosh x - x * Real.sinh x - 1) x := by
  have h1 : HasDerivAt (fun t : ℝ => 2 * Real.sinh t) (2 * Real.cosh x) x :=
    (Real.hasDerivAt_sinh x).const_mul 2
  have h2 : HasDerivAt (fun t : ℝ => t * Real.cosh t) (1 * Real.cosh x + x * Real.sinh x) x :=
    (hasDerivAt_id' x).fun_mul (Real.hasDerivAt_cosh x)
  have h3 := (h1.fun_sub h2).fun_sub (hasDerivAt_id' x)
  have e : 2 * Real.cosh x - (1 * Real.cosh x + x * Real.sinh x) - 1
      = Real.cosh x - x * Real.sinh x - 1 := by ring
  rw [e] at h3
  exact h3

/-- `deriv` form of `hasDerivAt_g`, as an equality of functions. -/
theorem deriv_g : deriv g = fun x : ℝ => Real.cosh x - x * Real.sinh x - 1 :=
  funext fun x => (hasDerivAt_g x).deriv

/-- The second derivative of `g` is `−x·cosh x`. -/
theorem hasDerivAt_deriv_g (x : ℝ) : HasDerivAt (deriv g) (-(x * Real.cosh x)) x := by
  rw [deriv_g]
  have h2 : HasDerivAt (fun t : ℝ => t * Real.sinh t) (1 * Real.sinh x + x * Real.cosh x) x :=
    (hasDerivAt_id' x).fun_mul (Real.hasDerivAt_sinh x)
  have h3 := ((Real.hasDerivAt_cosh x).fun_sub h2).sub_const 1
  have e : Real.sinh x - (1 * Real.sinh x + x * Real.cosh x) = -(x * Real.cosh x) := by ring
  rw [e] at h3
  exact h3

/-- For `x > 0`: `g(x) < 0`, that is, `2·sinh x < x·(cosh x + 1)`. -/
theorem g_neg {x : ℝ} (hx : 0 < x) : g x < 0 := by
  have hanti : StrictAntiOn g (Set.Ici 0) := by
    refine strictAntiOn_of_deriv_neg (convex_Ici 0) ?_ ?_
    · exact fun t _ => (hasDerivAt_g t).continuousAt.continuousWithinAt
    · intro t ht
      rw [interior_Ici] at ht
      have ht0 : 0 < t := ht
      rw [(hasDerivAt_g t).deriv]
      have h1 := A1 t ht0
      have h2 : 1 < Real.cosh t := Real.one_lt_cosh.mpr ht0.ne'
      linarith
  have key := hanti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx.le) hx
  rwa [g_zero] at key

/-- `g` is strictly concave on `[0, ∞)`. -/
theorem strictConcaveOn_g : StrictConcaveOn ℝ (Set.Ici 0) g := by
  refine strictConcaveOn_of_deriv2_neg (convex_Ici 0) ?_ ?_
  · exact fun t _ => (hasDerivAt_g t).continuousAt.continuousWithinAt
  · intro t ht
    rw [interior_Ici] at ht
    have ht0 : 0 < t := ht
    have e : deriv^[2] g t = -(t * Real.cosh t) := (hasDerivAt_deriv_g t).deriv
    rw [e]
    have hpos : 0 < t * Real.cosh t := mul_pos ht0 (Real.cosh_pos t)
    linarith

/-- **B1.** For `0 < x < y`: `B y < B x` (`B` is strictly decreasing on the positive
half-line). -/
theorem B1 (x y : ℝ) (hx : 0 < x) (hxy : x < y) : B y < B x := by
  have hy : 0 < y := hx.trans hxy
  have key := strictConcaveOn_g.secant_strict_mono (a := 0) (x := x) (y := y)
    (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx.le) (Set.mem_Ici.mpr hy.le) hx.ne' hy.ne' hxy
  rw [g_zero, sub_zero, sub_zero, sub_zero, sub_zero] at key
  rw [B_eq_g_div x hx.ne', B_eq_g_div y hy.ne']
  exact key

theorem H1 (x : ℝ) :
    (h x + h (-x)) / 2 = 2 * (Real.cosh x - 1) - x * Real.sinh x
      ∧ (x ≠ 0 → (h x - h (-x)) / (2 * x) = B x) := by
  refine ⟨?_, fun hx => ?_⟩
  · unfold h
    rw [Real.cosh_eq, Real.sinh_eq]
    ring
  · unfold h B
    rw [Real.cosh_eq, Real.sinh_eq]
    field_simp
    ring


end Kernel

noncomputable abbrev positiveLine : Measure ℝ := volume.restrict (Ioi (0 : ℝ))

def TextbookReflection : Prop :=
  ∀ (u : ℝ → ℝ) (m : ℝ), Integrable u volume →
    Integrable (fun x => u (m+x)+u (m-x)) positiveLine ∧
    (∫ x, u x) = ∫ x, u (m+x)+u (m-x) ∂positiveLine

noncomputable def evenKernel (x : ℝ) : ℝ :=
  2*(Real.cosh x-1)-x*Real.sinh x

/-- This is the signed-density argument, with no integral premise. -/
theorem paired_sign (x c p m : ℝ) (hx : 0 < x) (hc : 0 < c)
    (hp : 0 ≤ p) (hm : 0 ≤ m)
    (hleft : x < c → p-m ≤ 0) (hright : c < x → 0 ≤ p-m) :
    Kernel.h x*p+Kernel.h (-x)*m-Kernel.B c*(x*p+(-x)*m) ≤ 0 ∧
    (0 < p+m →
      Kernel.h x*p+Kernel.h (-x)*m-Kernel.B c*(x*p+(-x)*m) < 0) := by
  have heven := (Kernel.H1 x).1
  have hodd := (Kernel.H1 x).2 hx.ne'
  have hid : Kernel.h x*p+Kernel.h (-x)*m-Kernel.B c*(x*p+(-x)*m) =
      evenKernel x*(p+m)+x*(Kernel.B x-Kernel.B c)*(p-m) := by
    dsimp [evenKernel]
    have ho : Kernel.h x-Kernel.h (-x) = Kernel.B x*(2*x) :=
      (div_eq_iff (by positivity : 2*x ≠ 0)).mp hodd
    nlinarith
  have hsecond : x*(Kernel.B x-Kernel.B c)*(p-m) ≤ 0 := by
    rcases lt_trichotomy x c with hxc | hxc | hxc
    · exact mul_nonpos_of_nonneg_of_nonpos
        (mul_nonneg hx.le (sub_nonneg.mpr (Kernel.B1 x c hx hxc).le)) (hleft hxc)
    · simp [hxc]
    · exact mul_nonpos_of_nonpos_of_nonneg
        (mul_nonpos_of_nonneg_of_nonpos hx.le (sub_nonpos.mpr (Kernel.B1 c x hc hxc).le))
        (hright hxc)
  have hA : evenKernel x < 0 := Kernel.A1 x hx
  rw [hid]
  exact ⟨add_nonpos (mul_nonpos_of_nonpos_of_nonneg hA.le (add_nonneg hp hm)) hsecond,
    fun hpm => add_neg_of_neg_of_nonpos (mul_neg_of_neg_of_pos hA hpm) hsecond⟩

/-- Complete density formulation of the manuscript theorem. The covariance
on the right is `E[X exp X] - E[X] E[exp X]`. The half-line support hypothesis
is retained, although the proof works without it under the stated moment bound. -/
theorem strict_jensen_gap
    (textbook_reflection : TextbookReflection)
    (f : ℝ → ℝ) (hf : Integrable f volume) (hfpos : ∀ x, 0 ≤ f x)
    (hmass : (∫ x, f x) = 1)
    (hmean : Integrable (fun x => x*f x) volume)
    (b : ℝ) (_hsupport : ∀ᵐ x ∂volume, x < b → f x = 0)
    (hmoment : Integrable (fun x =>
      (1+|x-(∫ t, t*f t)|)*Real.exp (x-(∫ t, t*f t))*f x) volume)
    (c : ℝ) (hc : 0 < c)
    (hcross : ∀ᵐ x ∂positiveLine,
      (x < c → f ((∫ t, t*f t)+x)-f ((∫ t, t*f t)-x) ≤ 0) ∧
      (c < x → 0 ≤ f ((∫ t, t*f t)+x)-f ((∫ t, t*f t)-x))) :
    2*((∫ x, Real.exp x*f x)-Real.exp (∫ x, x*f x)) <
      (∫ x, x*Real.exp x*f x)-(∫ x, x*f x)*(∫ x, Real.exp x*f x) := by
  let μ := ∫ x, x*f x
  let e := fun x => Real.exp (x-μ)*f x
  let y := fun x => (x-μ)*f x
  let ye := fun x => (x-μ)*Real.exp (x-μ)*f x
  have hy : Integrable y volume := by
    convert hmean.sub (hf.const_mul μ) using 1
    funext x
    dsimp [y]
    ring
  have he : Integrable e volume := by
    apply hmoment.mono' ((Real.continuous_exp.comp (continuous_id.sub continuous_const)).aestronglyMeasurable.mul hf.aestronglyMeasurable)
    filter_upwards [] with x
    dsimp [e, μ]
    rw [abs_of_nonneg (mul_nonneg (Real.exp_pos _).le (hfpos x))]
    nlinarith [mul_nonneg (abs_nonneg (x-∫ t, t*f t))
      (mul_nonneg (Real.exp_pos (x-∫ t, t*f t)).le (hfpos x))]
  have hye : Integrable ye volume := by
    apply hmoment.mono' (((continuous_id.sub continuous_const).aestronglyMeasurable.mul
      (Real.continuous_exp.comp (continuous_id.sub continuous_const)).aestronglyMeasurable).mul hf.aestronglyMeasurable)
    filter_upwards [] with x
    dsimp [ye, μ]
    rw [abs_mul, abs_mul, abs_of_pos (Real.exp_pos _), abs_of_nonneg (hfpos x)]
    nlinarith [mul_nonneg (Real.exp_pos (x-∫ t, t*f t)).le (hfpos x)]
  have hh : Integrable (fun x => Kernel.h (x-μ)*f x) volume := by
    convert (((he.const_mul 2).sub hye).sub (hf.const_mul 2)).sub hy using 1
    funext x
    dsimp [e, ye, y, Kernel.h]
    ring
  have hiy : (∫ x, y x) = 0 := by
    have hyfun : y = fun x => x*f x-μ*f x := by funext x; dsimp [y]; ring
    rw [hyfun, integral_sub hmean (hf.const_mul μ), integral_const_mul, hmass]
    dsimp [μ]
    ring
  obtain ⟨hfoldh, heqh⟩ := textbook_reflection _ μ hh
  obtain ⟨hfoldy, heqy⟩ := textbook_reflection y μ hy
  obtain ⟨hfoldf, heqf⟩ := textbook_reflection f μ hf
  let H := fun x => Kernel.h x*f (μ+x)+Kernel.h (-x)*f (μ-x)-
    Kernel.B c*(x*f (μ+x)+(-x)*f (μ-x))
  have hiH : Integrable H positiveLine := by
    convert hfoldh.sub (hfoldy.const_mul (Kernel.B c)) using 1
    funext x
    simp [H, y]
  have hiH_eq : (∫ x, H x ∂positiveLine) = ∫ x, Kernel.h (x-μ)*f x := by
    have he : H = fun x =>
        (Kernel.h (μ+x-μ)*f (μ+x)+Kernel.h (μ-x-μ)*f (μ-x))-
        Kernel.B c*(y (μ+x)+y (μ-x)) := by funext x; simp [H, y]
    rw [he, integral_sub hfoldh (hfoldy.const_mul _), integral_const_mul, ← heqh, ← heqy, hiy]
    ring
  have hs : ∀ᵐ x ∂positiveLine, H x ≤ 0 ∧
      (0 < f (μ+x)+f (μ-x) → H x < 0) := by
    filter_upwards [hcross, ae_restrict_mem measurableSet_Ioi] with x hxc hx
    exact paired_sign x c (f (μ+x)) (f (μ-x)) hx hc (hfpos _) (hfpos _) hxc.1 hxc.2
  have hstrict : (∫ x, H x ∂positiveLine) < 0 := by
    have hn := integral_nonpos_of_ae (hs.mono fun _ h => h.1)
    apply lt_of_le_of_ne hn
    intro heq
    have hz : ∀ᵐ x ∂positiveLine, -H x = 0 :=
      (integral_eq_zero_iff_of_nonneg_ae (hs.mono fun _ h => neg_nonneg.mpr h.1) hiH.neg).mp
        (by rw [integral_neg, heq]; ring)
    have hfzero : ∀ᵐ x ∂positiveLine, f (μ+x)+f (μ-x) = 0 := by
      filter_upwards [hz, hs] with x hx hx'
      have hnn := add_nonneg (hfpos (μ+x)) (hfpos (μ-x))
      by_contra hne
      have := hx'.2 (lt_of_le_of_ne hnn (Ne.symm hne))
      linarith
    have : (∫ x, f (μ+x)+f (μ-x) ∂positiveLine) = 0 :=
      integral_eq_zero_of_ae hfzero
    rw [← heqf, hmass] at this
    norm_num at this
  have heh : (∫ x, Kernel.h (x-μ)*f x) =
      2*(∫ x, e x)-(∫ x, ye x)-2 := by
    have hefun : (fun x => Kernel.h (x-μ)*f x) =
        fun x => (2*e x-ye x-2*f x)-y x := by
      funext x; dsimp [e, ye, y, Kernel.h]; ring
    have h21 : Integrable (fun x => 2*e x-ye x) volume := (he.const_mul 2).sub hye
    have h22 : Integrable (fun x => 2*e x-ye x-2*f x) volume := h21.sub (hf.const_mul 2)
    rw [hefun, integral_sub h22 hy,
      integral_sub h21 (hf.const_mul 2),
      integral_sub (he.const_mul 2) hye, integral_const_mul, integral_const_mul, hmass, hiy]
    ring
  have hex : (fun x => Real.exp x*f x) = fun x => Real.exp μ*e x := by
    funext x
    dsimp [e]
    rw [← mul_assoc, ← Real.exp_add]
    congr 2
    ring
  have hxex : (fun x => x*Real.exp x*f x) =
      fun x => Real.exp μ*(ye x+μ*e x) := by
    funext x
    calc
      x*Real.exp x*f x = x*(Real.exp x*f x) := by ring
      _ = x*(Real.exp μ*e x) := by rw [congrFun hex x]
      _ = Real.exp μ*(ye x+μ*e x) := by dsimp [ye, e]; ring
  rw [hex, hxex, integral_const_mul, integral_const_mul,
    integral_add hye (he.const_mul μ), integral_const_mul]
  change 2*(Real.exp μ*(∫ x, e x)-Real.exp μ) <
    Real.exp μ*((∫ x, ye x)+μ*(∫ x, e x))-μ*(Real.exp μ*(∫ x, e x))
  rw [hiH_eq, heh] at hstrict
  nlinarith [mul_pos (Real.exp_pos μ) (show 0 < (∫ x, ye x)+2-2*(∫ x, e x) by linarith)]

end Dobson.Full2SingleCrossing

#print axioms Dobson.Full2SingleCrossing.strict_jensen_gap
