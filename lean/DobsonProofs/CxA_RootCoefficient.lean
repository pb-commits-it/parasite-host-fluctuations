import DobsonProofs.RingQuartic
import DobsonProofs.Coverage1Gamma

/-!
The coefficient-variation step in prop:bounds-r0star (audit part2:20).
All coefficients below are the manuscript's explicit ring formula, at the
actual Perron coordinate L = log lambda. No coefficient Lipschitz premise
is used. Identification of this formula with the stochastic growth expansion
is a separate obligation, which is not asserted by the smoothness lemmas.
-/

open Set Filter MeasureTheory
open scoped BigOperators Topology

namespace Dobson.CxA_RootCoefficient

noncomputable def tiltedTime {n : ℕ} (s : Fin n → ℝ) (L : ℝ) (i : Fin n) : ℝ :=
  Real.exp L / (Real.exp L - s i)

noncomputable def cost {n : ℕ} (s η φ : Fin n → ℝ) (L : ℝ) : ℝ :=
  ∑ j, η j ^ 2 * Coverage1Gamma.stepCostCoeff n (φ j) (tiltedTime s L)

lemma tiltedTime_eq {n : ℕ} (s : Fin n → ℝ) (L : ℝ) (i : Fin n) :
    tiltedTime s L i = 1 / (1 - s i / Real.exp L) := by
  unfold tiltedTime
  rw [one_sub_div (Real.exp_ne_zero L), one_div_div]

lemma tiltedTime_zero {n : ℕ} (s : Fin n → ℝ) (i : Fin n) :
    tiltedTime s 0 i = 1 / (1 - s i) := by simp [tiltedTime]

lemma tiltedTime_pos {n : ℕ} (s : Fin n → ℝ) (L : ℝ)
    (hL : ∀ i, s i < Real.exp L) (i : Fin n) : 0 < tiltedTime s L i :=
  div_pos (Real.exp_pos L) (sub_pos.mpr (hL i))

lemma cost_contDiffAt {n : ℕ} (hn : 1 ≤ n) (s η φ : Fin n → ℝ)
    (hφ0 : ∀ j, 0 ≤ φ j) (hφ1 : ∀ j, φ j < 1) (L : ℝ)
    (hL : ∀ i, s i < Real.exp L) : ContDiffAt ℝ 1 (cost s η φ) L := by
  have ht (i : Fin n) : ContDiffAt ℝ 1 (fun x => tiltedTime s x i) L :=
    Real.contDiff_exp.contDiffAt.div
      (Real.contDiff_exp.contDiffAt.sub contDiffAt_const)
      (ne_of_gt (sub_pos.mpr (hL i)))
  have htp := tiltedTime_pos s L hL
  haveI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  have hgen : 0 < Coverage1Gamma.genTime n (tiltedTime s L) :=
    Finset.sum_pos (fun i _ => htp i) Finset.univ_nonempty
  have hgenCD : ContDiffAt ℝ 1
      (fun x => Coverage1Gamma.genTime n (tiltedTime s x)) L :=
    ContDiffAt.sum (fun i _ => ht i)
  apply ContDiffAt.sum
  intro j _
  apply ContDiffAt.mul contDiffAt_const
  have hden (i : Fin n) :
      tiltedTime s L i - (tiltedTime s L i - 1) * φ j ≠ 0 := by
    have hp := mul_pos (htp i) (sub_pos.mpr (hφ1 j))
    have hz := hφ0 j
    nlinarith
  have hrCD : ContDiffAt ℝ 1
      (fun x => Coverage1Gamma.circuitR n (φ j) (tiltedTime s x)) L := by
    apply contDiffAt_prod
    intro i _
    exact contDiffAt_const.div
      ((ht i).sub (((ht i).sub contDiffAt_const).mul contDiffAt_const)) (hden i)
  have hr := (Coverage1Gamma.circuitR_properties n hn (φ j) (hφ0 j) (hφ1 j)
    (tiltedTime s L) htp).2.2.2
  exact (contDiffAt_const.div (contDiffAt_const.mul (hgenCD.pow 2)) (by positivity)).sub
    (hrCD.div (hgenCD.mul (contDiffAt_const.sub hrCD))
      (mul_ne_zero hgen.ne' (sub_ne_zero.mpr (ne_of_gt hr))))

/-- The moving ring coefficient is Lipschitz in actual log circuit reproduction.
The bound is derived on every compact interval separated from the staying poles. -/
theorem coefficient_variation {n : ℕ} (hn : 1 ≤ n) (s η φ : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hφ0 : ∀ j, 0 ≤ φ j) (hφ1 : ∀ j, φ j < 1)
    (δ : ℝ) (hδ : 0 < δ) (hsep : ∀ i, s i < Real.exp (-δ)) :
    ∃ B : ℝ, 0 ≤ B ∧ ∀ L ∈ Icc (-δ) δ,
      |cost s η φ L - cost s η φ 0| ≤ B * |RingCharacteristic.circuitLog s L| := by
  have hc (L : ℝ) (hL : L ∈ Icc (-δ) δ) :=
    cost_contDiffAt hn s η φ hφ0 hφ1 L
      (RingCharacteristic.domain_of_mem s δ L hsep hL)
  have hd : ContinuousOn (fun L => |deriv (cost s η φ) L|) (Icc (-δ) δ) := by
    intro L hL
    exact (((hc L hL).derivWithin (m := 0) (by norm_num)).continuousAt.abs).continuousWithinAt
  have hz : (0 : ℝ) ∈ Icc (-δ) δ := ⟨by linarith, hδ.le⟩
  obtain ⟨z, _, hmax⟩ := isCompact_Icc.exists_isMaxOn ⟨0, hz⟩ hd
  refine ⟨|deriv (cost s η φ) z|, abs_nonneg _, ?_⟩
  intro L hL
  have hb := (convex_Icc (-δ) δ).norm_image_sub_le_of_norm_deriv_le
    (fun x hx => (hc x hx).differentiableAt (by norm_num))
    (fun x hx => by rw [Real.norm_eq_abs]; exact hmax hx) hz hL
  simp only [Real.norm_eq_abs, sub_zero] at hb
  exact hb.trans (mul_le_mul_of_nonneg_left
    (RingCharacteristic.circuitLog_abs_lower hn s hs0 δ hδ hsep L hL) (abs_nonneg _))

/-- The root theorem with the coefficient-variation premise discharged.
The only project-specific analytic premise left is the quartic stochastic
growth residual, evaluated with the explicit moving ring coefficient. -/
theorem root_quartic_bound {Ω : Type*} [MeasurableSpace Ω]
    (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (η φ : Fin n → ℝ)
    (hφ0 : ∀ i, 0 ≤ φ i) (hφ1 : ∀ i, φ i < 1)
    (ξ : Ω → Fin n → ℝ) (hξ : ∀ i, Integrable (fun ω => ξ ω i) μ)
    (r : ℝ → (Fin n → ℝ) → ℝ)
    (hFurstenbergKesten : ∀ σ g, (∀ i, 0 < g i) →
      Tendsto (Coverage1Loss.ringNormalizedGrowth n hn s
        (RingQuartic.lognormalHost η ξ σ) μ T g) atTop (nhds (r σ g)))
    (δ : ℝ) (hδ : 0 < δ) (hsep : ∀ i, s i < Real.exp (-δ))
    (K : ℝ) (hK : 0 ≤ K) :
    ∃ σ0 H : ℝ, 0 < σ0 ∧ 0 ≤ H ∧
      ∀ (L σ : ℝ) (g : Fin n → ℝ), (∀ i, 0 < g i) →
      (∀ i, s i < Real.exp L) → (∏ i, g i) = ∏ i, (Real.exp L - s i) →
      |σ| ≤ σ0 → r σ g = 0 →
      |L - cost s η φ L * σ ^ 2| ≤ K * σ ^ 4 →
      |Real.log (Coverage1Loss.circuitR0 n s g) -
        RingCharacteristic.duration s * cost s η φ 0 * σ ^ 2| ≤ H * σ ^ 4 := by
  obtain ⟨B, hB, hcoef⟩ := coefficient_variation hn s η φ hs0 hφ0 hφ1 δ hδ hsep
  obtain ⟨A, U, σ1, hA, hU, hσ1, hroot⟩ := Coverage2Bounds.root_quartic_bound n hn s
    hs0 hs1 μ T hT η ξ hξ r hFurstenbergKesten δ hδ hsep (cost s η φ 0) K B hK hB
  let a := ∑ i, |η i| * ∫ ω, |ξ ω i| ∂μ
  let b := ∑ i, η i ^ 2 / 2
  have ha : 0 ≤ a := Finset.sum_nonneg (fun i _ =>
    mul_nonneg (abs_nonneg _) (integral_nonneg (fun ω => abs_nonneg _)))
  have hb : 0 ≤ b := Finset.sum_nonneg (fun i _ => by positivity)
  obtain ⟨σ2, hσ2, hsmall⟩ := RingQuartic.exists_small_radius a b 0 δ ha hb (by norm_num) hδ
  refine ⟨min σ1 σ2,
    RingCharacteristic.duration s *
      (K + A * (RingQuartic.localizationConstant U (cost s η φ 0) K σ1) ^ 2 +
        B * RingQuartic.localizationConstant U (cost s η φ 0) K σ1),
    lt_min hσ1 hσ2, ?_, ?_⟩
  · have hduration := (RingCharacteristic.duration_pos hn s hs1).le
    unfold RingQuartic.localizationConstant
    positivity
  intro L σ g hg hL hprod hσ hr hrem
  have hσa := hσ.trans (min_le_left _ _)
  have hσb := hσ.trans (min_le_right _ _)
  apply hroot L σ (cost s η φ L) g hg hL hprod hσa hr hrem
  have hcan : ∀ i, 0 < Real.exp L - s i := fun i => sub_pos.mpr (hL i)
  have hNpos : ∀ᵐ ω ∂μ, ∀ i, 0 < RingQuartic.lognormalHost η ξ σ ω i :=
    Eventually.of_forall (fun ω i => Real.exp_pos _)
  have he := RingQuartic.physical_limits_eq_of_prod n hn s g (fun i => Real.exp L - s i)
    hs0 hs1 hg hcan hprod μ T hT (RingQuartic.lognormalHost η ξ σ) hNpos
    (RingQuartic.lognormal_log_integrable μ η ξ hξ σ)
    (r σ g) (r σ (fun i => Real.exp L - s i))
    (hFurstenbergKesten σ g hg) (hFurstenbergKesten σ _ hcan)
  have hFK : Tendsto (Coverage1Loss.ringNormalizedGrowth n hn s
      (RingQuartic.lognormalHost η ξ σ) μ T (fun i => Real.exp L - s i))
      atTop (nhds 0) := by
    rw [← hr, he]
    exact hFurstenbergKesten σ _ hcan
  have hwindow := RingQuartic.root_lognormal_window n hn s hs0 hs1 L hL μ T hT
    η ξ hξ σ δ (hsmall σ hσb).1 hFK
  have heD : RingCharacteristic.circuitLog s L = Real.log (Coverage1Loss.circuitR0 n s g) := by
    rw [RingQuartic.circuitR0_eq_of_prod s g (fun i => Real.exp L - s i) hprod]
    exact RingQuartic.circuitLog_eq_logR0 s L hs1 hL
  rw [← heD]
  exact hcoef L hwindow

end Dobson.CxA_RootCoefficient

#print axioms Dobson.CxA_RootCoefficient.coefficient_variation
#print axioms Dobson.CxA_RootCoefficient.root_quartic_bound
