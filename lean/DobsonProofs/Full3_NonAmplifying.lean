import DobsonProofs.Full3_BlockRing
import Mathlib.Analysis.Complex.PhragmenLindelof
import Mathlib.Analysis.Complex.AbsMax
import Mathlib.Analysis.Matrix.Normed
import Mathlib.Analysis.Normed.Algebra.Spectrum
import Mathlib.Analysis.Calculus.Deriv.Polynomial
import Mathlib.Analysis.Convex.Topology
import Mathlib.Topology.Order.Compact

/-! Complete non-amplifying-ring theorem. Pointwise boundary damping suffices:
properness is derived from the state-space resolvent, boundedness from the
decay at infinity, and strictness from the maximum modulus principle.
No uniform boundary bound is a hypothesis. -/

namespace Dobson.Full3.NonAmplifying
open Finset Set Filter Bornology
open scoped Topology Matrix.Norms.Operator
open Dobson.Full3.BlockRing

variable {n : ℕ} (m : Fin n → ℕ)

theorem response_at_infinity
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) (i : Fin n) :
    Tendsto (fun z => response m A b c z i) (cobounded ℂ) (nhds 0) := by
  have h := spectrum.resolvent_tendsto_cobounded (𝕜 := ℂ) (A i)
  have hc : Continuous (fun M : Matrix (Fin (m i)) (Fin (m i)) ℂ =>
      ∑ p, ∑ q, c i p*M p q*b i q) := by fun_prop
  have ht := (hc.tendsto 0).comp h
  have hs (z : ℂ) : algebraMap ℂ (Matrix (Fin (m i)) (Fin (m i)) ℂ) z =
      Matrix.diagonal (fun _ => z) := by
    ext p q
    simp [Matrix.algebraMap_eq_diagonal, Matrix.diagonal_apply]
  simpa [response, contraction, resolvent,
    Matrix.nonsing_inv_eq_ringInverse, Function.comp_def, hs] using ht

theorem bounded_halfplane (f : ℂ → ℂ)
    (hc : ContinuousOn f {z | 0 ≤ z.re})
    (ht : Tendsto f (cobounded ℂ) (nhds 0)) :
    ∃ M : ℝ, ∀ z : ℂ, 0 ≤ z.re → ‖f z‖ ≤ M := by
  have htail : ∀ᶠ z in cobounded ℂ, ‖f z‖ < 1 :=
    ht.norm.eventually (eventually_lt_nhds (show ‖(0:ℂ)‖ < (1:ℝ) by simp))
  obtain ⟨R,_,hR⟩ := atTop_basis_Ioi.cobounded_of_norm.eventually_iff.mp htail
  have hK : IsCompact (Metric.closedBall (0:ℂ) R ∩ {z | 0 ≤ z.re}) :=
    (isCompact_closedBall _ _).inter_right (isClosed_le continuous_const Complex.continuous_re)
  obtain ⟨M,hM⟩ := hK.bddAbove_image (hc.norm.mono inter_subset_right)
  refine ⟨max M 1, fun z hz => ?_⟩
  by_cases hzr : ‖z‖ ≤ R
  · exact (hM ⟨z, ⟨by simpa using hzr,hz⟩,rfl⟩).trans (le_max_left _ _)
  · exact (hR (lt_of_not_ge hzr)).le.trans (le_max_right _ _)

/-- Narrow-import version of the existing `SmallGain.norm_le_of_bounded`.
The hypotheses here are subsequently derived for the physical responses. -/
theorem norm_le_halfplane (f : ℂ → ℂ)
    (hc : ContinuousOn f {z | 0 ≤ z.re})
    (hd : DifferentiableOn ℂ f {z | 0 < z.re})
    (hb : ∃ M : ℝ, ∀ z : ℂ, 0 ≤ z.re → ‖f z‖ ≤ M)
    (hi : ∀ y : ℝ, ‖f (Complex.I*(y:ℂ))‖ ≤ 1) :
    ∀ z : ℂ, 0 ≤ z.re → ‖f z‖ ≤ 1 := by
  obtain ⟨M,hM⟩ := hb
  have hdc : DiffContOnCl ℂ f {z : ℂ | 0 < z.re} :=
    ⟨hd, by rwa [Complex.closure_setOfPred_lt_re]⟩
  have he : ∃ p < (2:ℝ), ∃ B : ℝ,
      f =O[cobounded ℂ ⊓ principal {z : ℂ | 0 < z.re}]
        fun z => Real.exp (B*‖z‖^p) := by
    refine ⟨0,by norm_num,0,Asymptotics.IsBigO.of_bound M ?_⟩
    refine eventually_inf_principal.mpr (Eventually.of_forall fun z hz => ?_)
    simp only [zero_mul, Real.exp_zero, norm_one, mul_one]
    exact hM z hz.le
  have hr : IsBoundedUnder (· ≤ ·) atTop (fun x : ℝ => ‖f (x:ℂ)‖) := by
    refine ⟨M,eventually_map.mpr ?_⟩
    filter_upwards [eventually_ge_atTop (0:ℝ)] with x hx
    exact hM x (by simpa using hx)
  intro z hz
  exact PhragmenLindelof.right_half_plane_of_bounded_on_real hdc he hr
    (fun y => by rw [mul_comm]; exact hi y) hz

theorem strict_halfplane (f : ℂ → ℂ)
    (hc : ContinuousOn f {z | 0 ≤ z.re})
    (hd : DifferentiableOn ℂ f {z | 0 < z.re})
    (hb : ∃ M : ℝ, ∀ z : ℂ, 0 ≤ z.re → ‖f z‖ ≤ M)
    (hi : ∀ y : ℝ, ‖f (Complex.I*(y:ℂ))‖ < 1) :
    ∀ z : ℂ, 0 ≤ z.re → ‖f z‖ < 1 := by
  have hle := norm_le_halfplane f hc hd hb (fun y => (hi y).le)
  intro z hz
  by_cases hz0 : z.re=0
  · have he : z=Complex.I*(z.im:ℂ) := by apply Complex.ext <;> simp [hz0]
    rw [he]
    exact hi _
  · have hzp : 0 < z.re := lt_of_le_of_ne hz (Ne.symm hz0)
    by_contra hn
    have hnorm : ‖f z‖=1 := le_antisymm (hle z hz) (le_of_not_gt hn)
    have hconv : Convex ℝ {w : ℂ | 0 < w.re} :=
      (convex_Ioi (0:ℝ)).linear_preimage Complex.reCLM.toLinearMap
    have hopen : IsOpen {w : ℂ | 0 < w.re} :=
      isOpen_lt continuous_const Complex.continuous_re
    have hdc : DiffContOnCl ℂ f {w : ℂ | 0 < w.re} :=
      ⟨hd, by rwa [Complex.closure_setOfPred_lt_re]⟩
    have heq := Complex.norm_eqOn_closure_of_isPreconnected_of_isMaxOn
      hconv.isPreconnected hopen hdc hzp
      (show IsMaxOn (norm ∘ f) {w : ℂ | 0 < w.re} z from
        fun w hw => by simpa [hnorm] using hle w hw.le)
    have he := heq (show (0:ℂ) ∈ closure {w : ℂ | 0 < w.re} by
      rw [Complex.closure_setOfPred_lt_re]; simp)
    have hi0 := hi 0
    simp only [Function.comp_apply, Function.const_apply, hnorm] at he
    simpa [he] using hi0

/-- The exact manuscript hypotheses: internally Hurwitz blocks, every
imaginary-axis response at most one, and one pointwise strict damper. -/
theorem thm_dyn_k2 (hn : 1 ≤ n)
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ)
    (hH : ∀ i z, z ∈ spectrum ℂ (A i) → z.re < 0)
    (hN : ∀ i (y : ℝ), ‖response m A b c (Complex.I*(y:ℂ)) i‖ ≤ 1)
    (hD : ∃ j, ∀ y : ℝ, ‖response m A b c (Complex.I*(y:ℂ)) j‖ < 1) :
    ∀ z ∈ spectrum ℂ (jacobian m A b c), z.re < 0 := by
  have hden (i : Fin n) (z : ℂ) (hz : 0 ≤ z.re) : (A i).charpoly.eval z ≠ 0 := by
    intro he
    have hs : z ∈ spectrum ℂ (A i) :=
      Matrix.mem_spectrum_iff_isRoot_charpoly.mpr he
    exact (not_lt_of_ge hz) (hH i z hs)
  have hc (i : Fin n) : ContinuousOn (fun z => response m A b c z i) {z | 0 ≤ z.re} := by
    simp_rw [response_eq_unreduced]
    exact (numerator m A b c i).continuous.continuousOn.div
      (A i).charpoly.continuous.continuousOn (fun z hz => hden i z hz)
  have hd (i : Fin n) : DifferentiableOn ℂ (fun z => response m A b c z i) {z | 0 < z.re} := by
    simp_rw [response_eq_unreduced]
    exact (numerator m A b c i).differentiableOn.div
      (A i).charpoly.differentiableOn (fun z hz => hden i z hz.le)
  have hb (i : Fin n) := bounded_halfplane _ (hc i) (response_at_infinity m A b c i)
  have hbound (i : Fin n) := norm_le_halfplane _ (hc i) (hd i) (hb i) (hN i)
  obtain ⟨j,hj⟩ := hD
  have hstrict := strict_halfplane _ (hc j) (hd j) (hb j) hj
  intro z hz
  by_contra hnz
  have hz0 : 0 ≤ z.re := le_of_not_gt hnz
  have he := (loop_equation_iff m hn A b c z (fun i => hden i z hz0)).mp hz
  have hp : ‖∏ i, response m A b c z i‖ < 1 := by
    rw [norm_prod, ← Finset.mul_prod_erase univ _ (mem_univ j)]
    have hrest : (∏ i ∈ univ.erase j, ‖response m A b c z i‖) ≤ 1 :=
      prod_le_one₀ (fun i _ => norm_nonneg _) (fun i _ => hbound i z hz0)
    exact (mul_le_of_le_one_right (norm_nonneg _) hrest).trans_lt (hstrict z hz0)
  simpa [he] using hp

end Dobson.Full3.NonAmplifying

#print axioms Dobson.Full3.NonAmplifying.thm_dyn_k2
