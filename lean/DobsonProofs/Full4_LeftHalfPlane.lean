import DobsonProofs.Full1_DelayStability
import Mathlib.Analysis.Complex.Polynomial.GaussLucas

/-! Root lemmas used to prove the strict cascade-instability clauses of dyn-k5.
Closed-left stability is used so the conclusions give a positive-real-part root,
not merely failure of strict Hurwitz stability. -/
namespace Dobson.Full4.LeftHalfPlane
open Polynomial Complex Filter
open scoped Topology
open Dobson.Full1.DelayStability Dobson.Full1.BlockQualification

def ClosedLeft (f : ℂ → ℂ) : Prop := ∀ z, f z = 0 → z.re ≤ 0

theorem derivative_closedLeft (p : ℂ[X]) (hp : 0 < p.degree)
    (h : ClosedLeft p.eval) : ClosedLeft p.derivative.eval := by
  intro z hz
  rw [Polynomial.eq_centerMass_of_eval_derivative_eq_zero hp hz]
  simp only [Finset.centerMass, Complex.smul_re, Complex.re_sum, id_eq, smul_eq_mul]
  apply mul_nonpos_of_nonneg_of_nonpos
  · exact inv_nonneg.mpr (Polynomial.sum_derivRootWeight_pos hp z).le
  · apply Finset.sum_nonpos
    intro w hw
    exact mul_nonpos_of_nonneg_of_nonpos (Polynomial.derivRootWeight_nonneg p z w)
      (h w ((Polynomial.mem_roots (by rintro rfl; simp at hp)).mp
        (Multiset.mem_toFinset.mp hw)))

/-- Necessary weak Routh inequality for a cubic with no right-half-plane root. -/
theorem cubic_weak_routh (b₂ b₁ b₀ : ℝ)
    (h : ClosedLeft (cubic b₂ b₁ b₀)) : b₀ ≤ b₂*b₁ := by
  have hs (e : ℝ) (he : 0 < e) :
      Hurwitz (cubic (b₂+3*e) (b₁+2*b₂*e+3*e^2)
        (b₀+b₁*e+b₂*e^2+e^3)) := by
    intro z hz
    have hid : cubic b₂ b₁ b₀ (z+e) =
        cubic (b₂+3*e) (b₁+2*b₂*e+3*e^2) (b₀+b₁*e+b₂*e^2+e^3) z := by
      simp only [cubic]; push_cast; ring
    have hh := h (z+e) (hid.trans hz)
    simp only [Complex.add_re, Complex.ofReal_re] at hh
    linarith
  have ht : Tendsto (fun e : ℝ =>
      (b₂+3*e)*(b₁+2*b₂*e+3*e^2)-(b₀+b₁*e+b₂*e^2+e^3))
      (𝓝[>] 0) (𝓝 (b₂*b₁-b₀)) := by
    have hc : Continuous (fun e : ℝ =>
      (b₂+3*e)*(b₁+2*b₂*e+3*e^2)-(b₀+b₁*e+b₂*e^2+e^3)) := by fun_prop
    simpa using hc.continuousAt.tendsto.mono_left (show 𝓝[>] (0 : ℝ) ≤ 𝓝 (0 : ℝ) from nhdsWithin_le_nhds)
  have hn : 0 ≤ b₂*b₁-b₀ := ge_of_tendsto ht (by
    filter_upwards [self_mem_nhdsWithin (s := Set.Ioi (0 : ℝ))] with e he
    exact sub_nonneg.mpr ((cubic_hurwitz_iff _ _ _).mp (hs e he)).2.2.2.le)
  linarith

/-- For a polynomial with nonzero constant term, a negative sum of root real
parts and closed-left roots force a strictly positive logarithmic derivative
at zero. This includes the J=0 cascade boundary. -/
theorem logDerivative_pos (p : ℂ[X]) (hp0 : p.eval 0 ≠ 0)
    (h : ClosedLeft p.eval) (hsum : (p.roots.map Complex.re).sum < 0) :
    0 < (p.derivative.eval 0 / p.eval 0).re := by
  have hp : p ≠ 0 := by rintro rfl; simp at hp0
  have hroot (z : ℂ) (hz : z ∈ p.roots) : z.re ≤ 0 :=
    h z ((Polynomial.mem_roots hp).mp hz)
  have hex : ∃ z ∈ p.roots, z.re < 0 := by
    by_contra hn
    push_neg at hn
    have hh : 0 ≤ (p.roots.map Complex.re).sum :=
      Multiset.sum_nonneg (by
        intro v hv
        obtain ⟨w,hw,rfl⟩ := Multiset.mem_map.mp hv
        exact hn w hw)
    linarith
  obtain ⟨z,hz,hzneg⟩ := hex
  rw [(IsAlgClosed.splits p).eval_derivative_div_eval_of_ne_zero hp0]
  change 0 < Complex.reAddGroupHom _
  rw [map_multiset_sum]
  simp only [Multiset.map_map, Function.comp_def]
  have hall : ∀ v ∈ p.roots.map (fun w => (1/(0-w)).re), 0 ≤ v := by
    intro v hv
    obtain ⟨w,hw,rfl⟩ := Multiset.mem_map.mp hv
    simp only [one_div, zero_sub, inv_neg, Complex.neg_re, Complex.inv_re]
    exact neg_nonneg.mpr (div_nonpos_of_nonpos_of_nonneg (hroot w hw) (Complex.normSq_nonneg w))
  have hzpos : 0 < (1/(0-z)).re := by
    simp only [one_div, zero_sub, inv_neg, Complex.neg_re, Complex.inv_re]
    have hz0 : z ≠ 0 := by intro he; subst z; simp at hzneg
    exact neg_pos.mpr (div_neg_of_neg_of_pos hzneg (Complex.normSq_pos.mpr hz0))
  exact hzpos.trans_le (Multiset.single_le_sum hall _ (Multiset.mem_map.mpr ⟨z,hz,rfl⟩))

end Dobson.Full4.LeftHalfPlane
