import DobsonProofs.IidRing
import DobsonProofs.Coverage1Loss

namespace Dobson.RingMeans
open MeasureTheory Filter
open scoped BigOperators Topology

variable {Ω : Type*} [MeasurableSpace Ω]

lemma arithmetic_mean_pos (μ : Measure Ω) [IsProbabilityMeasure μ]
    (N : Ω → ℝ) (hN : Integrable N μ) (hpos : ∀ᵐ ω ∂μ, 0 < N ω) :
    0 < ∫ ω, N ω ∂μ := by
  apply (integral_pos_iff_support_of_nonneg_ae (hpos.mono (fun _ h => h.le)) hN).mpr
  have he : Function.support N =ᵐ[μ] Set.univ := by
    filter_upwards [hpos] with ω hω
    simp [Function.mem_support, hω.ne']
  rw [measure_congr he]
  simp

noncomputable def arithmeticR0 (n : ℕ) (s g : Fin n → ℝ)
    (N : Ω → Fin n → ℝ) (μ : Measure Ω) : ℝ :=
  Coverage1Loss.circuitR0 n s (fun i => g i * ∫ ω, N ω i ∂μ)

noncomputable def geometricR0 (n : ℕ) (s g : Fin n → ℝ)
    (N : Ω → Fin n → ℝ) (μ : Measure Ω) : ℝ :=
  Coverage1Loss.circuitR0 n s (fun i => g i * Real.exp (∫ ω, Real.log (N ω i) ∂μ))

lemma circuit_pos (n : ℕ) (s g : Fin n → ℝ) (hs : ∀ i, s i < 1) (hg : ∀ i, 0 < g i) :
    0 < Coverage1Loss.circuitR0 n s g :=
  Finset.prod_pos (fun i _ => div_pos (hg i) (sub_pos.mpr (hs i)))

lemma log_circuit (n : ℕ) (s g : Fin n → ℝ) (hs : ∀ i, s i < 1) (hg : ∀ i, 0 < g i) :
    Real.log (Coverage1Loss.circuitR0 n s g) =
      ∑ i, (Real.log (g i) + Real.log (RingGrowth.times s i)) := by
  unfold Coverage1Loss.circuitR0
  rw [Real.log_prod (fun i _ => (div_pos (hg i) (sub_pos.mpr (hs i))).ne')]
  apply Finset.sum_congr rfl
  intro i _
  rw [Real.log_div (hg i).ne' (sub_pos.mpr (hs i)).ne']
  simp [RingGrowth.times, sub_eq_add_neg]

/-- The geometric mean product is below the actual physical root divisor. -/
theorem geometric_below_root (n : ℕ) (hn : 1 ≤ n) (s g : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) (hg : ∀ i, 0 < g i)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → Fin n → ℝ)
    (hNpos : ∀ᵐ ω ∂μ, ∀ i, 0 < N ω i)
    (hNlog : ∀ i, Integrable (fun ω => Real.log (N ω i)) μ)
    (B : Finset (Fin n)) (Rstar : ℝ) (hR : 0 < Rstar)
    (hFurstenbergKesten_at_root : Tendsto
      (Coverage1Loss.ringNormalizedGrowth n hn s N μ T (Coverage1Loss.tiltBirths B Rstar g))
      atTop (nhds 0)) :
    geometricR0 n s g N μ ≤ Rstar ^ B.card := by
  letI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  obtain ⟨htilt, hsum⟩ := Coverage1Loss.sum_log_tiltBirths B Rstar hR g hg
  have hb := RingGrowth.physical_growth_lower s (Coverage1Loss.tiltBirths B Rstar g) N
    μ T hT hs0 hs1 htilt hNpos hNlog (finRotate n) ⟨0, by omega⟩ 0
    hFurstenbergKesten_at_root
  have htime : 0 < ReferenceRing.totalTime (RingGrowth.times s) :=
    Finset.sum_pos (fun i _ => lt_of_lt_of_le zero_lt_one (RingGrowth.times_ge_one s hs0 hs1 i))
      Finset.univ_nonempty
  have hnum := (div_le_iff₀ htime).mp hb
  simp only [zero_mul] at hnum
  have hlog : Real.log (geometricR0 n s g N μ) =
      (∑ i, Real.log (g i)) + ∑ i, (∫ ω, Real.log (N ω i) ∂μ) +
        ∑ i, Real.log (RingGrowth.times s i) := by
    rw [geometricR0, log_circuit n s _ hs1 (fun i => mul_pos (hg i) (Real.exp_pos _))]
    simp_rw [Real.log_mul (hg _).ne' (Real.exp_pos _).ne', Real.log_exp]
    simp only [Finset.sum_add_distrib]
  apply (Real.log_le_log_iff
    (circuit_pos n s _ hs1 (fun i => mul_pos (hg i) (Real.exp_pos _))) (pow_pos hR _)).mp
  unfold geometricR0 at hlog
  rw [Real.log_pow, hlog]
  simp only [Finset.sum_add_distrib] at hnum
  linarith

/-- Independence across time bounds the actual root by the arithmetic host product. -/
theorem root_below_arithmetic (n : ℕ) (hn : 1 ≤ n) (s g : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) (hg : ∀ i, 0 < g i)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → Fin n → ℝ) (hNmeas : Measurable N)
    (hNpos : ∀ᵐ ω ∂μ, ∀ i, 0 < N ω i)
    (hNmean : ∀ i, Integrable (fun ω => N ω i) μ)
    (hNlog : ∀ i, Integrable (fun ω => Real.log (N ω i)) μ)
    (hIndependent_times : ProbabilityTheory.iIndepFun (fun t : ℕ => fun ω => N ((T^[t]) ω)) μ)
    (B : Finset (Fin n)) (Rstar : ℝ) (hR : 0 < Rstar)
    (hFurstenbergKesten_at_root : Tendsto
      (Coverage1Loss.ringNormalizedGrowth n hn s N μ T (Coverage1Loss.tiltBirths B Rstar g))
      atTop (nhds 0)) :
    Rstar ^ B.card ≤ arithmeticR0 n s g N μ := by
  letI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  have hA (i : Fin n) : 0 < ∫ ω, N ω i ∂μ :=
    arithmetic_mean_pos μ _ (hNmean i) (hNpos.mono (fun _ h => h i))
  obtain ⟨htilt, hsum⟩ := Coverage1Loss.sum_log_tiltBirths B Rstar hR g hg
  let a := fun i => Coverage1Loss.tiltBirths B Rstar g i * ∫ ω, N ω i ∂μ
  have ha (i : Fin n) : 0 < a i := mul_pos (htilt i) (hA i)
  have hlog : Real.log (Coverage1Loss.circuitR0 n s a) =
      Real.log (arithmeticR0 n s g N μ) - B.card*Real.log Rstar := by
    rw [arithmeticR0, log_circuit n s a hs1 ha,
      log_circuit n s _ hs1 (fun i => mul_pos (hg i) (hA i))]
    dsimp [a]
    simp_rw [Real.log_mul (htilt _).ne' (hA _).ne',
      Real.log_mul (hg _).ne' (hA _).ne']
    simp only [Finset.sum_add_distrib]
    linarith
  by_contra h
  have hlt : arithmeticR0 n s g N μ < Rstar ^ B.card := lt_of_not_ge h
  have hloglt := Real.log_lt_log (circuit_pos n s _ hs1 (fun i => mul_pos (hg i) (hA i))) hlt
  rw [Real.log_pow] at hloglt
  have hP : Coverage1Loss.circuitR0 n s a < 1 := by
    apply (Real.log_lt_log_iff (circuit_pos n s a hs1 ha) zero_lt_one).mp
    rw [hlog, Real.log_one]
    exact sub_neg.mpr hloglt
  obtain ⟨v, ρ, hv, hρ0, hρ1, hrow⟩ :=
    IidRing.subcritical_mean_vector n hn s a hs0 hs1 ha hP
  have hb := IidRing.annealed_growth_le μ T hT N hNmeas hNmean hNlog hNpos
    hIndependent_times (finRotate n) ⟨0, by omega⟩ s (Coverage1Loss.tiltBirths B Rstar g) v
    hs0 hs1 htilt hv ρ hρ0 hrow 0 hFurstenbergKesten_at_root
  exact (not_lt_of_ge hb) (Real.log_neg hρ0 hρ1)


end Dobson.RingMeans

namespace Dobson.Coverage2Bounds
open MeasureTheory Filter
open scoped BigOperators Topology

/-- Actual arithmetic/geometric host means and the actual physical divisor root.
The geometric side uses stationarity; the arithmetic side uses independence across time. -/
theorem means_sandwich {Ω : Type*} [MeasurableSpace Ω]
    (n : ℕ) (hn : 1 ≤ n) (s g : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) (hg : ∀ i, 0 < g i)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → Fin n → ℝ) (hNmeas : Measurable N)
    (hNpos : ∀ᵐ ω ∂μ, ∀ i, 0 < N ω i)
    (hNmean : ∀ i, Integrable (fun ω => N ω i) μ)
    (hNlog : ∀ i, Integrable (fun ω => Real.log (N ω i)) μ)
    (hIndependent_times : ProbabilityTheory.iIndepFun (fun t : ℕ => fun ω => N ((T^[t]) ω)) μ)
    (B : Finset (Fin n)) (_hB : B.Nonempty) (Rstar : ℝ) (hR : 0 < Rstar)
    (hFurstenbergKesten_at_root : Tendsto
      (Coverage1Loss.ringNormalizedGrowth n hn s N μ T (Coverage1Loss.tiltBirths B Rstar g))
      atTop (nhds 0)) :
    RingMeans.geometricR0 n s g N μ ≤ Rstar ^ B.card ∧
      Rstar ^ B.card ≤ RingMeans.arithmeticR0 n s g N μ := by
  exact ⟨RingMeans.geometric_below_root n hn s g hs0 hs1 hg μ T hT N hNpos hNlog
    B Rstar hR hFurstenbergKesten_at_root,
    RingMeans.root_below_arithmetic n hn s g hs0 hs1 hg μ T hT N hNmeas hNpos hNmean hNlog
      hIndependent_times B Rstar hR hFurstenbergKesten_at_root⟩

end Dobson.Coverage2Bounds
#print axioms Dobson.RingMeans.geometric_below_root
#print axioms Dobson.Coverage2Bounds.means_sandwich
