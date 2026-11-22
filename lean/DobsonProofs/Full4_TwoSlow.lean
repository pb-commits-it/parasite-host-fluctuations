import DobsonProofs.Full4_LeftHalfPlane

/-! Two equal sufficiently slow L stages cause a strictly unstable root.
A necessary weak cubic Routh inequality for the derivative of the reciprocal
quartic suffices; no continuity-of-roots theorem is assumed. -/
namespace Dobson.Full4.TwoSlow
open Polynomial Complex Filter
open scoped Topology
open Dobson.Full4.LeftHalfPlane Dobson.Full1.DelayStability

/-- A necessary inequality, deliberately not called a full quartic criterion. -/
theorem quartic_necessary (b₃ b₂ b₁ b₀ : ℝ) (h0 : 0 < b₀)
    (h : ClosedLeft (fun z : ℂ => z^4+b₃*z^3+b₂*z^2+b₁*z+b₀)) :
    4*b₀*b₃ ≤ 6*b₁*b₂ := by
  let p : ℂ[X] := C (b₀ : ℂ)*X^4+C (b₁ : ℂ)*X^3+C (b₂ : ℂ)*X^2+C (b₃ : ℂ)*X+1
  have hpEval (z : ℂ) : p.eval z = (b₀ : ℂ)*z^4+b₁*z^3+b₂*z^2+b₃*z+1 := by
    simp [p]
  have hp : ClosedLeft p.eval := by
    intro z hz
    have hz0 : z ≠ 0 := by intro he; subst z; simpa [p] using hz
    have hid : z^4*((z⁻¹)^4+b₃*(z⁻¹)^3+b₂*(z⁻¹)^2+b₁*z⁻¹+b₀) = p.eval z := by
      rw [hpEval]
      field_simp
      ring
    have hr : (z⁻¹)^4+b₃*(z⁻¹)^3+b₂*(z⁻¹)^2+b₁*z⁻¹+b₀ = 0 :=
      (mul_eq_zero.mp (hid.trans hz)).resolve_left (pow_ne_zero _ hz0)
    have hh := h z⁻¹ hr
    rw [Complex.inv_re] at hh
    simpa using (div_le_iff₀ (Complex.normSq_pos.mpr hz0)).mp hh
  have hb0 : (b₀ : ℂ) ≠ 0 := by exact_mod_cast h0.ne'
  have hcoeff : p.coeff 4 ≠ 0 := by simpa [p, Polynomial.coeff_one] using hb0
  have hdeg : 0 < p.degree := lt_of_lt_of_le (by norm_num : (0 : WithBot ℕ) < 4)
    (Polynomial.le_degree_of_ne_zero hcoeff)
  have hd := derivative_closedLeft p hdeg hp
  have h4 : (4*b₀ : ℝ) ≠ 0 := by positivity
  have hc : ClosedLeft (cubic (3*b₁/(4*b₀)) (2*b₂/(4*b₀)) (b₃/(4*b₀))) := by
    intro z hz
    apply hd z
    have he : p.derivative.eval z = ((4*b₀ : ℝ) : ℂ)*
        cubic (3*b₁/(4*b₀)) (2*b₂/(4*b₀)) (b₃/(4*b₀)) z := by
      simp [p, Polynomial.derivative_mul, cubic]
      push_cast
      field_simp [hb0]
      <;> ring
    change p.derivative.eval z = 0
    rw [he,hz,mul_zero]
  have hi := cubic_weak_routh _ _ _ hc
  have hden : 0 < 4*b₀ := by positivity
  have hh := (div_le_iff₀ hden).mp hi
  field_simp at hh
  nlinarith

/-- Full two-slow-stage assertion from dyn-k5: a right-half-plane root for
all sufficiently small positive common rates, at any A>0 and q,h>0. -/
theorem two_slow_unstable (A q h : ℝ) (hA : 0 < A) (hq : 0 < q) (hh : 0 < h) :
    ∃ ε : ℝ, 0 < ε ∧ ∀ a : ℝ, 0 < a → a < ε →
      ∃ z : ℂ, z*(z+A)*(z+a)^2-(q : ℂ)*a^2*(z-h) = 0 ∧ 0 < z.re := by
  let f : ℝ → ℝ := fun a => 4*q*h*(A+2*a)-6*(A-q)*(2*A*a+a^2)
  have hf : Continuous f := by unfold f; fun_prop
  have hf0 : 0 < f 0 := by dsimp [f]; nlinarith [mul_pos hq hh, mul_pos (mul_pos hq hh) hA]
  have he : ∀ᶠ a in 𝓝 (0 : ℝ), 0 < f a := hf.continuousAt.eventually (lt_mem_nhds hf0)
  obtain ⟨ε,hε,hball⟩ := Metric.eventually_nhds_iff.mp he
  refine ⟨ε,hε,fun a ha hae => ?_⟩
  have hfpos : 0 < f a := hball (by simpa [Real.dist_eq, abs_of_pos ha] using hae)
  by_contra! hn
  have hclosed : ClosedLeft (fun z : ℂ => z^4+(A+2*a : ℝ)*z^3+
      (2*A*a+a^2 : ℝ)*z^2+((A-q)*a^2 : ℝ)*z+(q*h*a^2 : ℝ)) := by
    intro z hz
    apply hn z
    have hid : z*(z+A)*(z+a)^2-(q : ℂ)*a^2*(z-h) =
        z^4+(A+2*a : ℝ)*z^3+(2*A*a+a^2 : ℝ)*z^2+
        ((A-q)*a^2 : ℝ)*z+(q*h*a^2 : ℝ) := by push_cast; ring
    exact hid.trans hz
  have hi := quartic_necessary (A+2*a) (2*A*a+a^2) ((A-q)*a^2) (q*h*a^2)
    (by positivity) hclosed
  have hprod := mul_pos (sq_pos_of_pos ha) hfpos
  dsimp [f] at hprod
  nlinarith [hi]

end Dobson.Full4.TwoSlow
