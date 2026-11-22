import DobsonProofs.NoiseExceptional
import Mathlib.Analysis.SpecialFunctions.Log.ENNRealLogExp
import Mathlib.MeasureTheory.Integral.Lebesgue.Markov
import Mathlib.Topology.MetricSpace.HausdorffDistance
import Mathlib.Topology.Semicontinuity.Basic
import Mathlib.Analysis.Complex.Basic

/-!
Gap3exc, §4 Lemma 1: restriction of a finite-logarithmic-energy probability
to a bounded-potential probability. The logarithmic kernel is +infinity at
coincident points. In particular, finite energy is not confused with an
everywhere bounded potential of the original measure.
-/
namespace Dobson.Owed2_FiniteEnergy
open MeasureTheory Filter Set
open scoped ENNReal Topology

variable {X : Type*} [MetricSpace X] [MeasurableSpace X] [BorelSpace X]

/-- The actual log-plus reciprocal-distance kernel, including its singularity. -/
noncomputable def logKernel (x y : X) : ℝ≥0∞ :=
  (ENNReal.log ((edist x y)⁻¹)).toENNReal

noncomputable def potential (μ : Measure X) (x : X) : ℝ≥0∞ :=
  ∫⁻ y, logKernel x y ∂μ

noncomputable def energy (μ : Measure X) : ℝ≥0∞ := ∫⁻ x, potential μ x ∂μ

def BoundedPotential (μ : Measure X) : Prop :=
  ∃ C : ℝ≥0∞, C ≠ ∞ ∧ ∀ x, potential μ x ≤ C

@[simp] theorem logKernel_self (x : X) : logKernel x x = ∞ := by
  simp [logKernel]

theorem logKernel_of_ne (x y : X) (h : x ≠ y) :
    logKernel x y = ENNReal.ofReal (Real.log (1 / dist x y)) := by
  have hd : 0 < dist x y := dist_pos.mpr h
  rw [logKernel, edist_dist, ← ENNReal.ofReal_inv_of_pos hd,
    ENNReal.log_ofReal_of_pos (inv_pos.mpr hd)]
  simp [one_div]

theorem continuous_logKernel : Continuous (fun p : X × X => logKernel p.1 p.2) := by
  exact EReal.continuous_toENNReal.comp
    (ENNReal.continuous_log.comp (continuous_inv.comp continuous_edist))

theorem potential_lowerSemicontinuous (μ : Measure X) : LowerSemicontinuous (potential μ) := by
  rw [lowerSemicontinuous_iff_le_liminf]
  intro x
  have hmeas : ∀ z : X, Measurable (logKernel z) := fun z =>
    (continuous_logKernel.comp (continuous_const.prodMk continuous_id)).measurable
  have heq : ∀ y, liminf (fun z : X => logKernel z y) (𝓝 x) = logKernel x y := fun y =>
    ((continuous_logKernel.comp (continuous_id.prodMk continuous_const)).tendsto x).liminf_eq
  calc
    potential μ x = ∫⁻ y, liminf (fun z : X => logKernel z y) (𝓝 x) ∂μ := by
      simp only [heq, potential]
    _ ≤ _ := lintegral_liminf_le hmeas

/-- Nearest-point inequality for the singular log-plus kernel. -/
theorem logKernel_le_of_dist_le (x y z : X) (h : dist x y ≤ 2 * dist z y) :
    logKernel z y ≤ ENNReal.ofReal (Real.log 2) + logKernel x y := by
  have hd : edist x y ≤ 2 * edist z y := by
    simpa [edist_dist, ENNReal.ofReal_mul (by norm_num : (0:ℝ) ≤ 2)] using
      ENNReal.ofReal_le_ofReal h
  have hi : (edist z y)⁻¹ ≤ 2 * (edist x y)⁻¹ := by
    have hdiv : edist x y / 2 ≤ edist z y :=
      (ENNReal.div_le_iff (by norm_num) (by norm_num)).mpr (by simpa [mul_comm] using hd)
    simpa [ENNReal.div_eq_inv_mul, ENNReal.mul_inv, mul_comm] using
      ENNReal.inv_le_inv.mpr hdiv
  have hl := EReal.toENNReal_le_toENNReal (ENNReal.log_le_log hi)
  rw [ENNReal.log_mul_add] at hl
  apply hl.trans
  have ha := EReal.toENNReal_add_le (x := ENNReal.log 2) (y := ENNReal.log (edist x y)⁻¹)
  convert ha using 1
  rw [show (2:ℝ≥0∞) = ENNReal.ofReal (2:ℝ) by norm_num,
    ENNReal.log_ofReal_of_pos (by norm_num)]
  rfl

variable [ProperSpace X]

/-- Gap3exc §4 Lemma 1, for the actual logarithmic kernel. The new
probability is absolutely continuous with respect to the original one,
and its potential is bounded at every point, not only almost everywhere. -/
theorem finite_energy_restriction (μ : Measure X) [IsProbabilityMeasure μ]
    (henergy : energy μ ≠ ∞) :
    ∃ ν : Measure X, IsProbabilityMeasure ν ∧ ν ≪ μ ∧ BoundedPotential ν := by
  have hlsc := potential_lowerSemicontinuous μ
  have hfinite : ∀ᵐ x ∂μ, potential μ x < ∞ := ae_lt_top hlsc.measurable henergy
  have hex : ∃ n : ℕ, μ {x | potential μ x ≤ (n:ℝ≥0∞)} ≠ 0 := by
    by_contra! hn
    have hz : μ (⋃ n : ℕ, {x | potential μ x ≤ (n:ℝ≥0∞)}) = 0 :=
      measure_iUnion_null hn
    have hone : μ (⋃ n : ℕ, {x | potential μ x ≤ (n:ℝ≥0∞)}) = 1 := by
      apply (mem_ae_iff_prob_eq_one ?_).mp
      · filter_upwards [hfinite] with x hx
        obtain ⟨n,hn⟩ := ENNReal.exists_nat_gt hx.ne
        exact mem_iUnion.mpr ⟨n,hn.le⟩
      · exact MeasurableSet.iUnion fun (n : ℕ) =>
          (hlsc.isClosed_preimage (n:ℝ≥0∞)).measurableSet
    exact zero_ne_one (hz.symm.trans hone)
  obtain ⟨n,hn⟩ := hex
  let S : Set X := {x | potential μ x ≤ (n:ℝ≥0∞)}
  have hS : IsClosed S := hlsc.isClosed_preimage n
  have hSne : S.Nonempty := nonempty_of_measure_ne_zero hn
  have hp : μ S ≠ ∞ := measure_ne_top μ S
  let ν : Measure X := (μ S)⁻¹ • μ.restrict S
  have hprob : IsProbabilityMeasure ν := by
    constructor
    change (μ S)⁻¹ * (μ.restrict S) univ = 1
    rw [Measure.restrict_apply_univ]
    exact ENNReal.inv_mul_cancel hn hp
  letI : IsProbabilityMeasure ν := hprob
  have hac : ν ≪ μ := Measure.smul_absolutelyContinuous.trans Measure.absolutelyContinuous_restrict
  have hcarry : ∀ᵐ y ∂ν, y ∈ S :=
    Measure.smul_absolutelyContinuous.ae_le (ae_restrict_mem hS.measurableSet)
  have hbound : ∀ x ∈ S, potential ν x ≤ (μ S)⁻¹ * n := by
    intro x hx
    calc
      potential ν x = (μ S)⁻¹ * ∫⁻ y in S, logKernel x y ∂μ := by
        simp [potential, ν, lintegral_smul_measure]
      _ ≤ (μ S)⁻¹ * potential μ x := by
        gcongr
        exact lintegral_mono' Measure.restrict_le_self le_rfl
      _ ≤ _ := by gcongr; exact hx
  refine ⟨ν,hprob,hac,ENNReal.ofReal (Real.log 2)+(μ S)⁻¹*n,?_,?_⟩
  · exact ENNReal.add_ne_top.mpr ⟨ENNReal.ofReal_ne_top,
      ENNReal.mul_ne_top (ENNReal.inv_ne_top.mpr hn) (by simp)⟩
  · intro z
    obtain ⟨x,hx,hnear⟩ := hS.exists_infDist_eq_dist hSne z
    have hpoint : ∀ᵐ y ∂ν, logKernel z y ≤ ENNReal.ofReal (Real.log 2)+logKernel x y := by
      filter_upwards [hcarry] with y hy
      apply logKernel_le_of_dist_le
      have hd : dist z x ≤ dist z y := hnear ▸ Metric.infDist_le_dist_of_mem hy
      calc
        dist x y ≤ dist x z+dist z y := dist_triangle _ _ _
        _ ≤ 2*dist z y := by rw [dist_comm x z]; linarith
    calc
      potential ν z ≤ ∫⁻ y, ENNReal.ofReal (Real.log 2)+logKernel x y ∂ν := lintegral_mono_ae hpoint
      _ = ENNReal.ofReal (Real.log 2)+potential ν x := by
        rw [lintegral_add_left measurable_const]; simp [potential]
      _ ≤ _ := add_le_add_right (hbound x hx) _

/-- The corrected exceptional-set reduction with its restriction handoff
removed. Domination is requested only for the new bounded-potential
probability. Logarithmic domination itself is still a separate obligation. -/
theorem exceptional_reduction (g : X → ℕ → ℝ)
    (tests : Measure X → ℕ → X → ℝ≥0∞)
    (hdom : ∀ ν : Measure X, IsProbabilityMeasure ν → BoundedPotential ν →
      ∃ C : ℝ≥0∞, C ≠ ∞ ∧ (∀ n, AEMeasurable (tests ν n) ν) ∧
        ∀ n, ∫⁻ x, tests ν n x ∂ν ≤ C)
    (hblow : ∀ ν : Measure X, IsProbabilityMeasure ν → BoundedPotential ν →
      (∀ᵐ x ∂ν, NoiseExceptional.ZeroFactorialType (g x)) →
        ∀ᵐ x ∂ν, liminf (fun n => tests ν n x) atTop = ∞) :
    ¬∃ μ : Measure X, IsProbabilityMeasure μ ∧ energy μ ≠ ∞ ∧
      ∀ᵐ x ∂μ, NoiseExceptional.ZeroFactorialType (g x) := by
  apply NoiseExceptional.finite_energy_restriction_reduction g
    (fun μ => energy μ ≠ ∞) BoundedPotential tests _ hdom hblow
  intro μ hμ he hz
  letI : IsProbabilityMeasure μ := hμ
  obtain ⟨ν,hν,hac,hb⟩ := finite_energy_restriction μ he
  exact ⟨ν,hν,hb,hac.ae_le hz⟩

theorem complex_kernel_le_real (z : ℂ) (y : ℝ) :
    logKernel z (y:ℂ) ≤ logKernel z.re y := by
  have hd : edist z.re y ≤ edist z (y:ℂ) := by
    rw [edist_dist, edist_dist]
    apply ENNReal.ofReal_le_ofReal
    simpa [Real.dist_eq, Complex.dist_eq, Complex.sub_re] using
      Complex.abs_re_le_norm (z-(y:ℂ))
  exact EReal.toENNReal_le_toENNReal
    (ENNReal.log_le_log (ENNReal.inv_le_inv.mpr hd))

/-- Exact real-measure/complex-plane formulation of gap3exc §4 Lemma 1.
No compact-support assumption is needed for this stronger conclusion. -/
theorem real_finite_energy_restriction (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (he : energy μ ≠ ∞) :
    ∃ ν : Measure ℝ, IsProbabilityMeasure ν ∧ ν ≪ μ ∧
      ∃ C : ℝ≥0∞, C ≠ ∞ ∧ ∀ z : ℂ, ∫⁻ y, logKernel z (y:ℂ) ∂ν ≤ C := by
  obtain ⟨ν,hν,hac,C,hC,hb⟩ := finite_energy_restriction μ he
  refine ⟨ν,hν,hac,C,hC,fun z => ?_⟩
  exact (lintegral_mono fun y => complex_kernel_le_real z y).trans (hb z.re)

end Dobson.Owed2_FiniteEnergy

#print axioms Dobson.Owed2_FiniteEnergy.finite_energy_restriction
#print axioms Dobson.Owed2_FiniteEnergy.exceptional_reduction
#print axioms Dobson.Owed2_FiniteEnergy.real_finite_energy_restriction
