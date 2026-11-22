import DobsonProofs.Coverage1Loss
import DobsonProofs.RingCharacteristic

open Set Filter MeasureTheory
open scoped BigOperators Topology

namespace Dobson.RingQuartic

/-- The characteristic coordinate is exactly log circuit reproduction for
this positive mean-transfer representative. -/
lemma circuitLog_eq_logR0 {n : ℕ} (s : Fin n → ℝ) (L : ℝ)
    (hs : ∀ i, s i < 1) (hL : ∀ i, s i < Real.exp L) :
    RingCharacteristic.circuitLog s L = Real.log
      (Coverage1Loss.circuitR0 n s (fun i => Real.exp L - s i)) := by
  unfold RingCharacteristic.circuitLog Coverage1Loss.circuitR0
  rw [Real.log_prod (fun i _ =>
    (div_pos (sub_pos.mpr (hL i)) (sub_pos.mpr (hs i))).ne')]
  apply Finset.sum_congr rfl
  intro i _
  exact (Real.log_div (sub_pos.mpr (hL i)).ne' (sub_pos.mpr (hs i)).ne').symm

private theorem quartic_algebra (D σ T C0 LD CD K A B M : ℝ)
    (hT : 0 < T) (hA : 0 ≤ A) (hB : 0 ≤ B) (_hM : 0 ≤ M)
    (hrem : |LD - CD * σ ^ 2| ≤ K * σ ^ 4)
    (hlin : |LD - D / T| ≤ A * D ^ 2)
    (hcoef : |CD - C0| ≤ B * |D|)
    (hloc : |D| ≤ M * σ ^ 2) :
    |D - T * C0 * σ ^ 2| ≤ T * (K + A * M ^ 2 + B * M) * σ ^ 4 := by
  have hsq : D ^ 2 ≤ M ^ 2 * σ ^ 4 := by
    have := mul_self_le_mul_self (abs_nonneg D) hloc
    nlinarith [sq_abs D]
  have h1 := mul_le_mul_of_nonneg_left hsq hA
  have h2 := mul_le_mul_of_nonneg_left hloc hB
  have h3 := mul_le_mul_of_nonneg_right (hcoef.trans h2) (sq_nonneg σ)
  have htri := abs_add_three (D / T - LD) (LD - CD * σ ^ 2) ((CD - C0) * σ ^ 2)
  rw [abs_sub_comm (D / T) LD, abs_mul, abs_of_nonneg (sq_nonneg σ)] at htri
  have hb : |D / T - C0 * σ ^ 2| ≤ (K + A * M ^ 2 + B * M) * σ ^ 4 := by
    have he : D / T - LD + (LD - CD * σ ^ 2) + (CD - C0) * σ ^ 2 = D / T - C0 * σ ^ 2 := by ring
    rw [he] at htri
    nlinarith
  have hmul := mul_le_mul_of_nonneg_left hb hT.le
  have he : D - T * C0 * σ ^ 2 = T * (D / T - C0 * σ ^ 2) := by
    field_simp
  rw [he, abs_mul, abs_of_pos hT]
  simpa [mul_assoc] using hmul

end Dobson.RingQuartic


namespace Dobson.RingQuartic

noncomputable def localizationConstant (U C0 K σ0 : ℝ) : ℝ :=
  2 * U * (|C0| + K * σ0 ^ 2)

lemma localize_from_residual (D L σ σ0 C0 CD K B U : ℝ)
    (hK : 0 ≤ K) (hB : 0 ≤ B) (hU : 0 ≤ U)
    (hnoise : |σ| ≤ σ0) (hsmall : 2 * U * B * σ ^ 2 ≤ 1)
    (hslope : |D| ≤ U * |L|)
    (hrem : |L - CD * σ ^ 2| ≤ K * σ ^ 4)
    (hcoef : |CD - C0| ≤ B * |D|) :
    |D| ≤ localizationConstant U C0 K σ0 * σ ^ 2 := by
  have hcd : |CD| ≤ B * |D| + |C0| := by
    have ht := abs_add_le (CD - C0) C0
    have he : CD - C0 + C0 = CD := by ring
    rw [he] at ht
    linarith
  have hL : |L| ≤ K * σ ^ 4 + (B * |D| + |C0|) * σ ^ 2 := by
    have ht := abs_add_le (L - CD * σ ^ 2) (CD * σ ^ 2)
    have he : L - CD * σ ^ 2 + CD * σ ^ 2 = L := by ring
    rw [he, abs_mul, abs_of_nonneg (sq_nonneg σ)] at ht
    have hc := mul_le_mul_of_nonneg_right hcd (sq_nonneg σ)
    linarith
  have hsquare : σ ^ 2 ≤ σ0 ^ 2 := by
    have hh := mul_self_le_mul_self (abs_nonneg σ) hnoise
    simpa only [← sq, sq_abs] using hh
  have hfour : K * σ ^ 4 ≤ K * σ0 ^ 2 * σ ^ 2 := by
    have hh := mul_le_mul_of_nonneg_left hsquare (mul_nonneg hK (sq_nonneg σ))
    nlinarith only [hh]
  have hupper := mul_le_mul_of_nonneg_left hL hU
  have hfourU := mul_le_mul_of_nonneg_left hfour hU
  have hsmallD := mul_le_mul_of_nonneg_right hsmall (abs_nonneg D)
  unfold localizationConstant
  nlinarith only [hslope, hupper, hfourU, hsmallD]

end Dobson.RingQuartic


open Set Filter MeasureTheory
open scoped BigOperators Topology

namespace Dobson.RingQuartic

noncomputable def lognormalHost {n : ℕ} {Ω : Type*}
    (η : Fin n → ℝ) (ξ : Ω → Fin n → ℝ) (σ : ℝ) (ω : Ω) (i : Fin n) : ℝ :=
  Real.exp (σ * η i * ξ ω i - σ ^ 2 * η i ^ 2 / 2)

noncomputable def noiseBudget {n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (η : Fin n → ℝ) (ξ : Ω → Fin n → ℝ) (σ : ℝ) : ℝ :=
  |σ| * (∑ i, |η i| * ∫ ω, |ξ ω i| ∂μ) + σ ^ 2 * (∑ i, η i ^ 2 / 2)

lemma lognormal_log_integrable {n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (η : Fin n → ℝ) (ξ : Ω → Fin n → ℝ)
    (hξ : ∀ i, Integrable (fun ω => ξ ω i) μ) (σ : ℝ) (i : Fin n) :
    Integrable (fun ω => Real.log (lognormalHost η ξ σ ω i)) μ := by
  simpa only [lognormalHost, Real.log_exp, Pi.sub_def] using
    ((hξ i).const_mul (σ * η i)).sub (integrable_const (σ ^ 2 * η i ^ 2 / 2))

lemma lognormal_budget {n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (η : Fin n → ℝ) (ξ : Ω → Fin n → ℝ)
    (hξ : ∀ i, Integrable (fun ω => ξ ω i) μ) (σ : ℝ) :
    (∑ i, ∫ ω, |Real.log (lognormalHost η ξ σ ω i)| ∂μ) ≤ noiseBudget μ η ξ σ := by
  have hi (i : Fin n) : (∫ ω, |Real.log (lognormalHost η ξ σ ω i)| ∂μ) ≤
      |σ| * |η i| * (∫ ω, |ξ ω i| ∂μ) + σ ^ 2 * η i ^ 2 / 2 := by
    have hbudget : Integrable (fun ω => |σ| * |η i| * |ξ ω i| + σ ^ 2 * η i ^ 2 / 2) μ := by
      simpa only [Pi.add_def] using (((hξ i).abs.const_mul (|σ| * |η i|)).add
        (integrable_const (σ ^ 2 * η i ^ 2 / 2)))
    have hh := integral_mono (lognormal_log_integrable μ η ξ hξ σ i).abs
      hbudget
      (fun ω => by
        dsimp [lognormalHost]
        rw [Real.log_exp]
        have ht := abs_sub (σ * η i * ξ ω i) (σ ^ 2 * η i ^ 2 / 2)
        simpa only [abs_mul, abs_of_nonneg (by positivity : 0 ≤ σ ^ 2 * η i ^ 2 / 2)] using ht)
    simpa only [integral_add ((hξ i).abs.const_mul (|σ| * |η i|))
      (integrable_const (σ ^ 2 * η i ^ 2 / 2)), integral_const_mul,
      integral_const, probReal_univ, one_smul] using hh
  have hh := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hi i)
  simpa only [noiseBudget, Finset.sum_add_distrib, ← Finset.mul_sum, mul_assoc, mul_div_assoc] using hh

lemma characteristic_growth_upper {Ω : Type*} [MeasurableSpace Ω]
    (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℝ) (hs0 : ∀ i, 0 ≤ s i)
    (hs1 : ∀ i, s i < 1) (L : ℝ) (hL : ∀ i, s i < Real.exp L)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → Fin n → ℝ)
    (hNpos : ∀ᵐ ω ∂μ, ∀ i, 0 < N ω i)
    (hNlog : ∀ i, Integrable (fun ω => Real.log (N ω i)) μ)
    (r : ℝ) (hFK : Tendsto (Coverage1Loss.ringNormalizedGrowth n hn s N μ T
      (fun i => Real.exp L - s i)) atTop (nhds r)) :
    r ≤ L + ∑ i, ∫ ω, |Real.log (N ω i)| ∂μ := by
  letI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  let g := fun i => Real.exp L - s i
  have hg : ∀ i, 0 < g i := fun i => sub_pos.mpr (hL i)
  let K := ReferenceRing.kernel (RingGrowth.times s) (RingGrowth.times_ge_one s hs0 hs1) (finRotate n)
  let B := fun ω => L + ∑ i, |Real.log (N ω i)|
  have hB : Integrable B μ := (integrable_const L).add
    (integrable_finsetSum _ (fun i _ => (hNlog i).abs))
  have hrow : ∀ᵐ ω ∂μ, ∀ i, (∑ e, K.prob i e * Real.exp (RingGrowth.score s g N ω e)) ≤ Real.exp (B ω) := by
    filter_upwards [hNpos] with ω hω
    intro i
    rw [RingGrowth.score_row_sum s g N hs0 hs1 hg (finRotate n) ⟨0, by omega⟩ ω hω i]
    have habs : |Real.log (N ω i)| ≤ ∑ j, |Real.log (N ω j)| :=
      Finset.single_le_sum (f := fun j => |Real.log (N ω j)|) (fun j _ => abs_nonneg _) (Finset.mem_univ i)
    have hN : N ω i ≤ Real.exp (∑ j, |Real.log (N ω j)|) := by
      rw [← Real.exp_log (hω i)]
      exact Real.exp_le_exp.mpr ((le_abs_self _).trans habs)
    have h1 : 1 ≤ Real.exp (∑ j, |Real.log (N ω j)|) :=
      Real.one_le_exp_iff.mpr (Finset.sum_nonneg (fun j _ => abs_nonneg _))
    have ha := mul_le_mul_of_nonneg_left hN (hg i).le
    have hb := mul_le_mul_of_nonneg_left h1 (hs0 i)
    dsimp [B]
    rw [Real.exp_add]
    dsimp [g] at ha
    nlinarith
  have hbound := fun k => RingGrowth.normalized_growth_upper K μ T hT
    (RingGrowth.score s g N) (RingGrowth.score_integrable s g N μ hs0 hg hNpos hNlog)
    B hB hrow k
  have he := RingGrowth.normalized_eq_physical s g N hs0 hs1 hg
    (finRotate n) ⟨0, by omega⟩ μ T hT hNpos
  have hlim : Tendsto (ReferencePaths.normalizedGrowth K μ T (RingGrowth.score s g N))
      atTop (nhds r) := by rw [he]; exact hFK
  have hb := le_of_tendsto hlim (Eventually.of_forall hbound)
  simpa only [B, integral_add (integrable_const L)
      (integrable_finsetSum _ (fun i _ => (hNlog i).abs)), integral_const,
    probReal_univ, one_smul, integral_finsetSum _ (fun i _ => (hNlog i).abs)] using hb

end Dobson.RingQuartic

namespace Dobson.RingQuartic

lemma root_abs_log_le {Ω : Type*} [MeasurableSpace Ω]
    (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℝ) (hs0 : ∀ i, 0 ≤ s i)
    (hs1 : ∀ i, s i < 1) (L : ℝ) (hL : ∀ i, s i < Real.exp L)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → Fin n → ℝ)
    (hNpos : ∀ᵐ ω ∂μ, ∀ i, 0 < N ω i)
    (hNlog : ∀ i, Integrable (fun ω => Real.log (N ω i)) μ)
    (hFK : Tendsto (Coverage1Loss.ringNormalizedGrowth n hn s N μ T
      (fun i => Real.exp L - s i)) atTop (nhds 0)) :
    |L| ≤ ∑ i, ∫ ω, |Real.log (N ω i)| ∂μ := by
  letI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  have hupper := characteristic_growth_upper n hn s hs0 hs1 L hL μ T hT N hNpos hNlog 0 hFK
  have hlower := RingGrowth.physical_growth_lower s (fun i => Real.exp L - s i) N
    μ T hT hs0 hs1 (fun i => sub_pos.mpr (hL i)) hNpos hNlog
    (finRotate n) ⟨0, by omega⟩ 0 hFK
  have htime : 0 < ReferenceRing.totalTime (RingGrowth.times s) :=
    Finset.sum_pos (fun i _ => lt_of_lt_of_le zero_lt_one
      (RingGrowth.times_ge_one s hs0 hs1 i)) Finset.univ_nonempty
  have hnum := (div_le_iff₀ htime).mp hlower
  have hid : (∑ i, (Real.log (Real.exp L - s i) + (∫ ω, Real.log (N ω i) ∂μ) +
      Real.log (RingGrowth.times s i))) = RingCharacteristic.circuitLog s L +
      ∑ i, ∫ ω, Real.log (N ω i) ∂μ := by
    simp only [RingCharacteristic.circuitLog, RingGrowth.times, one_div, Real.log_inv,
      Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_neg_distrib]
    ring
  rw [hid, zero_mul] at hnum
  have habs : -(∑ i, ∫ ω, Real.log (N ω i) ∂μ) ≤ ∑ i, ∫ ω, |Real.log (N ω i)| ∂μ := by
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_le_sum
    intro i _
    rw [← integral_neg]
    exact integral_mono (hNlog i).neg (hNlog i).abs (fun ω => neg_le_abs _)
  have hbudget : 0 ≤ ∑ i, ∫ ω, |Real.log (N ω i)| ∂μ :=
    Finset.sum_nonneg (fun i _ => integral_nonneg (fun ω => abs_nonneg _))
  apply abs_le.mpr
  constructor
  · linarith
  · by_cases hp : 0 ≤ L
    · have hh := RingCharacteristic.circuitLog_ge_of_nonneg hn s hs0 hs1 L hp
      linarith
    · exact (le_of_not_ge hp).trans hbudget

lemma root_lognormal_window {Ω : Type*} [MeasurableSpace Ω]
    (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℝ) (hs0 : ∀ i, 0 ≤ s i)
    (hs1 : ∀ i, s i < 1) (L : ℝ) (hL : ∀ i, s i < Real.exp L)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (η : Fin n → ℝ) (ξ : Ω → Fin n → ℝ)
    (hξ : ∀ i, Integrable (fun ω => ξ ω i) μ) (σ δ : ℝ)
    (hnoise : noiseBudget μ η ξ σ ≤ δ)
    (hFK : Tendsto (Coverage1Loss.ringNormalizedGrowth n hn s
      (lognormalHost η ξ σ) μ T (fun i => Real.exp L - s i)) atTop (nhds 0)) :
    L ∈ Icc (-δ) δ := by
  apply abs_le.mp
  exact (root_abs_log_le n hn s hs0 hs1 L hL μ T hT (lognormalHost η ξ σ)
    (Eventually.of_forall (fun ω i => Real.exp_pos _))
    (lognormal_log_integrable μ η ξ hξ σ) hFK).trans
    ((lognormal_budget μ η ξ hξ σ).trans hnoise)

end Dobson.RingQuartic

namespace Dobson.RingQuartic

/-- The actual physical growth limit is unchanged by redistributing mean
transfers while preserving their product. -/
lemma physical_limits_eq_of_prod {Ω : Type*} [MeasurableSpace Ω]
    (n : ℕ) (hn : 1 ≤ n) (s g g' : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (hg : ∀ i, 0 < g i) (hg' : ∀ i, 0 < g' i) (hprod : ∏ i, g i = ∏ i, g' i)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → Fin n → ℝ)
    (hNpos : ∀ᵐ ω ∂μ, ∀ i, 0 < N ω i)
    (hNlog : ∀ i, Integrable (fun ω => Real.log (N ω i)) μ)
    (r r' : ℝ)
    (hFK : Tendsto (Coverage1Loss.ringNormalizedGrowth n hn s N μ T g) atTop (nhds r))
    (hFK' : Tendsto (Coverage1Loss.ringNormalizedGrowth n hn s N μ T g') atTop (nhds r')) :
    r = r' := by
  letI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  obtain ⟨d, hd, hD⟩ := Coverage1Loss.exists_pos_diag_conj_of_prod_eq n hn g g' hg hg' hprod
  apply GrowthShape.physical_growth_eq_of_conj s g g' N μ T hT hs0 hs1 hg hg'
    hNpos hNlog (finRotate n) ⟨0, by omega⟩ d hd
  · apply Eventually.of_forall
    intro ω
    have ht := congrArg Matrix.transpose (hD s (N ω))
    simp only [Matrix.transpose_mul, Matrix.diagonal_transpose] at ht
    have hA := (Coverage1Loss.physical_step_transpose n hn s g (N ω)).symm
    have hB := (Coverage1Loss.physical_step_transpose n hn s g' (N ω)).symm
    rw [hA, hB, Matrix.transpose_transpose, Matrix.transpose_transpose] at ht
    exact ht
  · exact hFK
  · exact hFK'

lemma circuitR0_eq_of_prod {n : ℕ} (s g g' : Fin n → ℝ)
    (hprod : ∏ i, g i = ∏ i, g' i) :
    Coverage1Loss.circuitR0 n s g = Coverage1Loss.circuitR0 n s g' := by
  simp only [Coverage1Loss.circuitR0, Finset.prod_div_distrib, hprod]

end Dobson.RingQuartic

namespace Dobson.RingQuartic

lemma exists_small_radius (a b p δ : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hp : 0 ≤ p) (hδ : 0 < δ) :
    ∃ σ0 : ℝ, 0 < σ0 ∧ ∀ σ : ℝ, |σ| ≤ σ0 →
      |σ| * a + σ ^ 2 * b ≤ δ ∧ 2 * p * σ ^ 2 ≤ 1 := by
  let σ0 := min 1 (min (δ / (a + b + 1)) (1 / (2 * (p + 1))))
  have hab : 0 < a + b + 1 := by linarith
  have hpp : 0 < 2 * (p + 1) := by positivity
  have hσ0 : 0 < σ0 := lt_min (by norm_num)
    (lt_min (div_pos hδ hab) (one_div_pos.mpr hpp))
  have h1 : σ0 ≤ 1 := min_le_left ..
  have ha0 : σ0 ≤ δ / (a + b + 1) := (min_le_right ..).trans (min_le_left ..)
  have hp0 : σ0 ≤ 1 / (2 * (p + 1)) := (min_le_right ..).trans (min_le_right ..)
  have ha1 := (le_div_iff₀ hab).mp ha0
  have hp1 := (le_div_iff₀ hpp).mp hp0
  refine ⟨σ0, hσ0, ?_⟩
  intro σ hσ
  have hs1 : |σ| ≤ 1 := hσ.trans h1
  have hs2 : σ ^ 2 ≤ |σ| := by nlinarith [sq_abs σ, mul_nonneg (abs_nonneg σ) (sub_nonneg.mpr hs1)]
  have hbb := mul_le_mul_of_nonneg_right hs2 hb
  have haa := mul_le_mul_of_nonneg_right hσ (add_nonneg ha hb)
  have hpp2 := mul_le_mul_of_nonneg_left (hs2.trans hσ) (show 0 ≤ 2 * p from mul_nonneg (by norm_num) hp)
  constructor <;> nlinarith

end Dobson.RingQuartic

namespace Dobson.Coverage2Bounds

/-- The actual ring characteristic Taylor bound and root localization.
The noise radius is constructed from the driver moments. The fourth-order
growth residual and variation of its second coefficient remain explicit. -/
theorem root_quartic_bound {Ω : Type*} [MeasurableSpace Ω]
    (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (η : Fin n → ℝ) (ξ : Ω → Fin n → ℝ)
    (hξ : ∀ i, Integrable (fun ω => ξ ω i) μ)
    (r : ℝ → (Fin n → ℝ) → ℝ)
    (hFurstenbergKesten : ∀ σ g, (∀ i, 0 < g i) →
      Tendsto (Coverage1Loss.ringNormalizedGrowth n hn s
        (RingQuartic.lognormalHost η ξ σ) μ T g) atTop (nhds (r σ g)))
    (δ : ℝ) (hδ : 0 < δ) (hsep : ∀ i, s i < Real.exp (-δ))
    (C0 K B : ℝ) (hK : 0 ≤ K) (hB : 0 ≤ B) :
    ∃ A U σ0 : ℝ, 0 ≤ A ∧ 0 ≤ U ∧ 0 < σ0 ∧
      ∀ (L σ CD : ℝ) (g : Fin n → ℝ), (∀ i, 0 < g i) →
      (∀ i, s i < Real.exp L) → (∏ i, g i) = ∏ i, (Real.exp L - s i) →
      |σ| ≤ σ0 → r σ g = 0 →
      |L - CD * σ ^ 2| ≤ K * σ ^ 4 →
      |CD - C0| ≤ B * |Real.log (Coverage1Loss.circuitR0 n s g)| →
      |Real.log (Coverage1Loss.circuitR0 n s g) -
        RingCharacteristic.duration s * C0 * σ ^ 2| ≤
        RingCharacteristic.duration s *
          (K + A * (RingQuartic.localizationConstant U C0 K σ0) ^ 2 +
            B * RingQuartic.localizationConstant U C0 K σ0) * σ ^ 4 := by
  obtain ⟨A, hA, hlin⟩ := RingCharacteristic.linear_taylor_bound hn s hs0 hs1 δ hδ hsep
  obtain ⟨U, hU, hslope⟩ := RingCharacteristic.circuitLog_abs_upper s δ hδ hsep
  let a := ∑ i, |η i| * ∫ ω, |ξ ω i| ∂μ
  let b := ∑ i, η i ^ 2 / 2
  have ha : 0 ≤ a := Finset.sum_nonneg (fun i _ =>
    mul_nonneg (abs_nonneg _) (integral_nonneg (fun ω => abs_nonneg _)))
  have hb : 0 ≤ b := Finset.sum_nonneg (fun i _ => by positivity)
  obtain ⟨σ0, hσ0, hsmall⟩ := RingQuartic.exists_small_radius a b (U * B) δ
    ha hb (mul_nonneg hU hB) hδ
  refine ⟨A, U, σ0, hA, hU, hσ0, ?_⟩
  intro L σ CD g hg hL hprod hnoise hroot hrem hcoef
  have hcan : ∀ i, 0 < Real.exp L - s i := fun i => sub_pos.mpr (hL i)
  have hNpos : ∀ᵐ ω ∂μ, ∀ i, 0 < RingQuartic.lognormalHost η ξ σ ω i :=
    Eventually.of_forall (fun ω i => Real.exp_pos _)
  have hNlog := RingQuartic.lognormal_log_integrable μ η ξ hξ σ
  have he := RingQuartic.physical_limits_eq_of_prod n hn s g (fun i => Real.exp L - s i)
    hs0 hs1 hg hcan hprod μ T hT (RingQuartic.lognormalHost η ξ σ) hNpos hNlog
    (r σ g) (r σ (fun i => Real.exp L - s i))
    (hFurstenbergKesten σ g hg) (hFurstenbergKesten σ _ hcan)
  rw [hroot] at he
  have hFKcan : Tendsto (Coverage1Loss.ringNormalizedGrowth n hn s
      (RingQuartic.lognormalHost η ξ σ) μ T (fun i => Real.exp L - s i))
      atTop (nhds 0) := by
    rw [he]
    exact hFurstenbergKesten σ _ hcan
  have hbudget : RingQuartic.noiseBudget μ η ξ σ ≤ δ := (hsmall σ hnoise).1
  have hsmall' : 2 * U * B * σ ^ 2 ≤ 1 := by
    simpa only [mul_assoc] using (hsmall σ hnoise).2
  have hwindow := RingQuartic.root_lognormal_window n hn s hs0 hs1 L hL μ T hT
    η ξ hξ σ δ hbudget hFKcan
  have heD : RingCharacteristic.circuitLog s L = Real.log (Coverage1Loss.circuitR0 n s g) := by
    rw [RingQuartic.circuitR0_eq_of_prod s g (fun i => Real.exp L - s i) hprod]
    exact RingQuartic.circuitLog_eq_logR0 s L hs1 hL
  rw [← heD] at hcoef ⊢
  have hloc := RingQuartic.localize_from_residual (RingCharacteristic.circuitLog s L)
    L σ σ0 C0 CD K B U hK hB hU hnoise hsmall' (hslope L hwindow) hrem hcoef
  exact RingQuartic.quartic_algebra (RingCharacteristic.circuitLog s L) σ
    (RingCharacteristic.duration s) C0 L CD K A B
    (RingQuartic.localizationConstant U C0 K σ0)
    (RingCharacteristic.duration_pos hn s hs1) hA hB
    (by unfold RingQuartic.localizationConstant; positivity)
    hrem (hlin L hwindow) hcoef hloc

end Dobson.Coverage2Bounds
