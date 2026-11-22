import Mathlib
import DobsonProofs.NoiseAnalyticity
import Mathlib.Analysis.Calculus.FDeriv.Extend
import Mathlib.Analysis.Calculus.Taylor
import Mathlib.MeasureTheory.Integral.DominatedConvergence

namespace Dobson.GammaAtZero
open Set Filter MeasureTheory
open scoped Topology

/-- Differentiation under an integral at a one-sided endpoint. The bound is
on the right interval only, so no false negative-q Gamma integrability is used. -/
theorem integral_right_deriv {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (F F' : ℝ → Ω → ℝ) (B : Ω → ℝ)
    (hI : ∀ q ∈ Icc (0 : ℝ) 1, Integrable (F q) μ)
    (hB : Integrable B μ)
    (hd : ∀ᵐ ω ∂μ, ∀ q ∈ Icc (0 : ℝ) 1, HasDerivAt (fun t => F t ω) (F' q ω) q)
    (hb : ∀ᵐ ω ∂μ, ∀ q ∈ Icc (0 : ℝ) 1, ‖F' q ω‖ ≤ B ω) :
    HasDerivWithinAt (fun q => ∫ ω, F q ω ∂μ) (∫ ω, F' 0 ω ∂μ) (Ici 0) 0 := by
  have h0 : (0 : ℝ) ∈ Icc (0 : ℝ) 1 := by constructor <;> norm_num
  have he : ∀ᶠ q : ℝ in 𝓝[>] (0 : ℝ), q ∈ Ioc (0 : ℝ) 1 :=
    Ioc_mem_nhdsGT zero_lt_one
  have hm : ∀ᶠ q : ℝ in 𝓝[>] (0 : ℝ),
      AEStronglyMeasurable (fun ω => (F q ω-F 0 ω)/q) μ := by
    filter_upwards [he] with q hq
    exact ((hI q ⟨hq.1.le,hq.2⟩).sub (hI 0 h0)).div_const q |>.aestronglyMeasurable
  have hbd : ∀ᶠ q : ℝ in 𝓝[>] (0 : ℝ),
      ∀ᵐ ω ∂μ, ‖(F q ω-F 0 ω)/q‖ ≤ B ω := by
    filter_upwards [he] with q hq
    filter_upwards [hd,hb] with ω hd hb
    have hmvt := Convex.norm_image_sub_le_of_norm_deriv_le
      (s := Icc (0 : ℝ) q) (f := fun t => F t ω)
      (fun t ht => (hd t ⟨ht.1, ht.2.trans hq.2⟩).differentiableAt)
      (fun t ht => by rw [(hd t ⟨ht.1, ht.2.trans hq.2⟩).deriv]; exact hb t ⟨ht.1,ht.2.trans hq.2⟩)
      (convex_Icc (0 : ℝ) q) (left_mem_Icc.mpr hq.1.le) (right_mem_Icc.mpr hq.1.le)
    simp only [norm_div, Real.norm_eq_abs, abs_of_pos hq.1]
    rw [div_le_iff₀ hq.1]
    simpa only [sub_zero, Real.norm_eq_abs, abs_of_pos hq.1] using hmvt
  have hl : ∀ᵐ ω ∂μ, Tendsto (fun q : ℝ => (F q ω-F 0 ω)/q)
      (𝓝[>] (0 : ℝ)) (𝓝 (F' 0 ω)) := by
    filter_upwards [hd] with ω hd
    simpa [slope_fun_def_field] using (hd 0 h0).tendsto_slope.mono_left (nhdsGT_le_nhdsNE 0)
  have ht := tendsto_integral_filter_of_dominated_convergence B hm hbd hB hl
  apply HasDerivWithinAt.Ici_of_Ioi
  rw [hasDerivWithinAt_iff_tendsto_slope' (by simp : (0 : ℝ) ∉ Ioi (0 : ℝ))]
  apply Tendsto.congr' _ ht
  filter_upwards [he] with q hq
  rw [integral_div, integral_sub (hI q ⟨hq.1.le,hq.2⟩) (hI 0 h0), slope_def_field]
  simp

noncomputable def gammaWeight (p z : ℝ) : ℝ := z^(p-1)*Real.exp (-z)

theorem gammaWeight_integrable (p : ℝ) (hp : 0 < p) :
    IntegrableOn (gammaWeight p) (Ioi 0) := by
  change IntegrableOn (fun z : ℝ => z^(p-1)*Real.exp (-z)) (Ioi 0)
  simpa [Real.rpow_one, neg_mul] using
    (integrableOn_rpow_mul_exp_neg_mul_rpow (p := 1) (s := p-1) (b := 1)
      (by linarith) one_pos one_pos)

noncomputable def gammaIntegral (p : ℝ) (f : ℝ → ℝ) : ℝ :=
  (∫ z in Ioi (0 : ℝ), f z * gammaWeight p z) / Real.Gamma p

theorem gammaWeight_moment (p : ℝ) (k : ℕ) (hp : 0 < p) :
    gammaIntegral p (fun z => z^k) = Real.Gamma (p+k) / Real.Gamma p := by
  unfold gammaIntegral
  congr 1
  have he : (fun z => z^k * gammaWeight p z) =ᵐ[volume.restrict (Ioi 0)]
      (fun z => z^(p+k-1)*Real.exp (-z)) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    dsimp [gammaWeight]
    rw [← mul_assoc, ← Real.rpow_natCast, ← Real.rpow_add hz]
    congr 2
    ring
  rw [integral_congr_ae he]
  have hpk : 0 < p+(k:ℝ) := by positivity
  simpa using Real.integral_rpow_mul_exp_neg_mul_Ioi hpk (r := 1) one_pos


end Dobson.GammaAtZero
namespace Dobson.GammaFirstJet
open Set Filter MeasureTheory
open scoped Topology
open GammaAtZero NoiseAnalyticity

noncomputable def tiltedWeight (p q z : ℝ) : ℝ :=
  z^(p-1) * (1+q*z)^(-p) * Real.exp (-z)
noncomputable def tiltedMoment (p q : ℝ) (n : ℕ) : ℝ :=
  ∫ z in Ioi (0 : ℝ), z^n * tiltedWeight p q z

theorem gamma_polynomial_integrable (p : ℝ) (n : ℕ) (hp : 0 < p) :
    IntegrableOn (fun z : ℝ => z^n*z^(p-1)*Real.exp (-z)) (Ioi 0) := by
  have hd : IntegrableOn (fun z : ℝ => z^(p+n-1)*Real.exp (-z)) (Ioi 0) := by
    simpa [Real.rpow_one] using
      (integrableOn_rpow_mul_exp_neg_mul_rpow (p := 1) (s := p+n-1) (b := 1)
        (by have := Nat.cast_nonneg (α := ℝ) n; linarith) one_pos one_pos)
  apply hd.congr
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
  rw [← Real.rpow_natCast, ← Real.rpow_add hz]
  congr 2
  ring

theorem tilted_integrable (p q : ℝ) (n : ℕ) (hp : 0 < p) (hq : 0 ≤ q) :
    IntegrableOn (fun z : ℝ => z^n * tiltedWeight p q z) (Ioi 0) := by
  have hc : ContinuousOn (fun z : ℝ => z^n * tiltedWeight p q z) (Ioi 0) := by
    unfold tiltedWeight
    apply ContinuousOn.mul (continuousOn_id.pow n)
    apply ContinuousOn.mul
    · apply ContinuousOn.mul
      · exact continuousOn_id.rpow_const (fun z hz => Or.inl (ne_of_gt hz))
      · apply (continuousOn_const.add (continuousOn_const.mul continuousOn_id)).rpow_const
        intro z hz
        left
        have : 0 ≤ q*z := mul_nonneg hq hz.le
        change 1+q*z ≠ 0
        linarith
    · fun_prop
  refine (gamma_polynomial_integrable p n hp).mono'
    (hc.aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
  have hz0 : 0 < z := hz
  have hpow : (1+q*z)^(-p) ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos
    (by nlinarith) (by linarith)
  have hnn : 0 ≤ z^n * tiltedWeight p q z := by unfold tiltedWeight; positivity
  rw [Real.norm_of_nonneg hnn]
  dsimp only [tiltedWeight]
  calc
    z^n * (z^(p-1)*(1+q*z)^(-p)*Real.exp (-z)) =
      (z^n*z^(p-1)*Real.exp (-z)) * (1+q*z)^(-p) := by ring
    _ ≤ (z^n*z^(p-1)*Real.exp (-z)) * 1 :=
      mul_le_mul_of_nonneg_left hpow (by positivity)
    _ = _ := mul_one _


noncomputable def logNumerator (p q : ℝ) : ℝ :=
  ∫ z in Ioi (0 : ℝ), tiltedWeight p q z * Real.log (1+q*z)
noncomputable def normalizer (p q : ℝ) : ℝ := tiltedMoment p q 0
noncomputable def scaledLogMean (p q : ℝ) : ℝ := logNumerator p q / normalizer p q
noncomputable def firstJet (p : ℝ) : ℝ := derivWithin (scaledLogMean p) (Ici 0) 0

theorem gamma_moment_integrable (p : ℝ) (n : ℕ) (hp : 0 < p) :
    IntegrableOn (fun z => z^n*gammaWeight p z) (Ioi 0) := by
  have hi := gamma_polynomial_integrable p n hp
  simpa only [gammaWeight, mul_assoc] using hi

theorem normalizer_zero (p : ℝ) (hp : 0 < p) : normalizer p 0 = Real.Gamma p := by
  unfold normalizer tiltedMoment tiltedWeight
  simp only [pow_zero, one_mul, zero_mul, add_zero, Real.one_rpow, mul_one]
  simpa using Real.integral_rpow_mul_exp_neg_mul_Ioi hp (r := 1) zero_lt_one

theorem gamma_first (p : ℝ) (hp : 0 < p) :
    (∫ z in Ioi (0 : ℝ), z*gammaWeight p z) = Real.Gamma (p+1) := by
  have hm := gammaWeight_moment p 1 hp
  unfold gammaIntegral at hm
  simp only [pow_one, Nat.cast_one] at hm
  exact (div_left_inj' (Real.Gamma_pos_of_pos hp).ne').mp hm

theorem normalizer_deriv (p : ℝ) (hp : 0 < p) :
    HasDerivWithinAt (normalizer p) (-p*Real.Gamma (p+1)) (Ici 0) 0 := by
  let F := fun q z => tiltedWeight p q z
  let F' := fun q z => z*(-p)*(1+q*z)^(-p-1)*gammaWeight p z
  let B := fun z => p*z*gammaWeight p z
  have hI : ∀ q ∈ Icc (0 : ℝ) 1, IntegrableOn (F q) (Ioi 0) := by
    intro q hq
    simpa [F] using tilted_integrable p q 0 hp hq.1
  have hB : IntegrableOn B (Ioi 0) := by
    change Integrable (fun z => p*z*gammaWeight p z) (volume.restrict (Ioi 0))
    simpa only [mul_assoc, pow_one] using (gamma_moment_integrable p 1 hp).const_mul p
  have hd : ∀ᵐ z ∂volume.restrict (Ioi 0), ∀ q ∈ Icc (0 : ℝ) 1,
      HasDerivAt (fun t => F t z) (F' q z) q := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    intro q hq
    have hu : 0 < 1+q*z := by have hz0 : 0 < z := hz; nlinarith [hq.1]
    have h := (((hasDerivAt_id q).mul_const z).const_add 1).rpow_const (p := -p) (Or.inl hu.ne')
    convert h.mul_const (gammaWeight p z) using 1
    · funext t; dsimp [F,tiltedWeight,gammaWeight]; ring
    · dsimp [F']; ring
  have hb : ∀ᵐ z ∂volume.restrict (Ioi 0), ∀ q ∈ Icc (0 : ℝ) 1, ‖F' q z‖ ≤ B z := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    intro q hq
    have hz0 : 0 < z := hz
    have hu : 1 ≤ 1+q*z := by nlinarith [hq.1]
    have hw : 0 ≤ gammaWeight p z := by dsimp [gammaWeight]; positivity
    have ht : 0 ≤ (1+q*z)^(-p-1) := Real.rpow_nonneg (by linarith) _
    have hle : (1+q*z)^(-p-1) ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos hu (by linarith)
    change |z*(-p)*(1+q*z)^(-p-1)*gammaWeight p z| ≤ p*z*gammaWeight p z
    have hb0 : 0 ≤ p*z*gammaWeight p z := mul_nonneg (mul_nonneg hp.le hz0.le) hw
    rw [show z*(-p)*(1+q*z)^(-p-1)*gammaWeight p z =
      -(p*z*gammaWeight p z)*(1+q*z)^(-p-1) by ring, abs_mul, abs_neg,
      abs_of_nonneg hb0, abs_of_nonneg ht]
    simpa using mul_le_mul_of_nonneg_left hle hb0
  have h := integral_right_deriv (volume.restrict (Ioi 0)) F F' B hI hB hd hb
  have he : (∫ z in Ioi (0 : ℝ), F' 0 z) = -p*Real.Gamma (p+1) := by
    dsimp [F']
    simp only [zero_mul,add_zero,Real.one_rpow,mul_one]
    simp_rw [show ∀ z : ℝ, z*(-p)*gammaWeight p z = -p*(z*gammaWeight p z) by intro z; ring]
    rw [integral_const_mul, gamma_first p hp]
  rw [he] at h
  convert h using 1
  funext q
  simp [normalizer,tiltedMoment,F]


theorem logNumerator_zero (p : ℝ) : logNumerator p 0 = 0 := by
  simp [logNumerator]

theorem logNumerator_deriv (p : ℝ) (hp : 0 < p) :
    HasDerivWithinAt (logNumerator p) (Real.Gamma (p+1)) (Ici 0) 0 := by
  let F := fun q z => tiltedWeight p q z * Real.log (1+q*z) / q
  let B := fun z => z*gammaWeight p z
  have he : ∀ᶠ q : ℝ in 𝓝[>] (0 : ℝ), 0 < q := self_mem_nhdsWithin
  have hm : ∀ᶠ q : ℝ in 𝓝[>] (0 : ℝ),
      AEStronglyMeasurable (F q) (volume.restrict (Ioi 0)) := by
    filter_upwards [he] with q hq
    have hw : AEStronglyMeasurable (tiltedWeight p q) (volume.restrict (Ioi 0)) := by
      simpa using (tilted_integrable p q 0 hp hq.le).aestronglyMeasurable
    have hm : AEStronglyMeasurable (fun z : ℝ => Real.log (1+q*z)) (volume.restrict (Ioi 0)) :=
      (Real.measurable_log.comp (measurable_const.add (measurable_const.mul measurable_id))).aestronglyMeasurable
    simpa only [F, div_eq_mul_inv, Pi.mul_apply] using (hw.mul hm).mul_const q⁻¹
  have hb : ∀ᶠ q : ℝ in 𝓝[>] (0 : ℝ),
      ∀ᵐ z ∂volume.restrict (Ioi 0), ‖F q z‖ ≤ B z := by
    filter_upwards [he] with q hq
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    have hz0 : 0 < z := hz
    have hu : 1 ≤ 1+q*z := by nlinarith
    have hw : 0 ≤ gammaWeight p z := by dsimp [gammaWeight]; positivity
    have ht0 : 0 ≤ (1+q*z)^(-p) := Real.rpow_nonneg (by linarith) _
    have ht : (1+q*z)^(-p) ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos hu (by linarith)
    have hl0 : 0 ≤ Real.log (1+q*z)/q := div_nonneg (Real.log_nonneg hu) hq.le
    have hl : Real.log (1+q*z)/q ≤ z := by
      rw [div_le_iff₀ hq]
      have := Real.log_le_sub_one_of_pos (show 0 < 1+q*z by linarith)
      nlinarith
    have heq : F q z = gammaWeight p z * ((1+q*z)^(-p) * (Real.log (1+q*z)/q)) := by
      dsimp [F,tiltedWeight,gammaWeight]; ring
    rw [heq, Real.norm_of_nonneg (mul_nonneg hw (mul_nonneg ht0 hl0))]
    change _ ≤ z*gammaWeight p z
    nlinarith [mul_le_mul_of_nonneg_left (mul_le_mul ht hl hl0 (by norm_num : (0:ℝ)≤1)) hw]
  have hB : Integrable B (volume.restrict (Ioi 0)) := by
    change IntegrableOn (fun z => z*gammaWeight p z) (Ioi 0)
    simpa only [pow_one] using gamma_moment_integrable p 1 hp
  have hl : ∀ᵐ z ∂volume.restrict (Ioi 0),
      Tendsto (fun q => F q z) (𝓝[>] (0 : ℝ)) (𝓝 (z*gammaWeight p z)) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    have hd : HasDerivAt (fun q : ℝ => Real.log (1+q*z)) z 0 := by
      convert (((hasDerivAt_id (0:ℝ)).mul_const z).const_add 1).log (by norm_num) using 1 <;> simp
    have hs : Tendsto (fun q : ℝ => Real.log (1+q*z)/q)
        (𝓝[>] (0 : ℝ)) (𝓝 z) := by
      simpa [slope_fun_def_field] using hd.tendsto_slope.mono_left (nhdsGT_le_nhdsNE 0)
    have hc : ContinuousAt (fun q : ℝ => tiltedWeight p q z) 0 := by
      unfold tiltedWeight
      apply ContinuousAt.mul
      · apply ContinuousAt.const_mul
        exact (continuousAt_const.add (continuousAt_id.mul_const z)).rpow_const (Or.inl (by norm_num))
      · fun_prop
    have ht : Tendsto (fun q : ℝ => tiltedWeight p q z)
        (𝓝[>] (0 : ℝ)) (𝓝 (gammaWeight p z)) := by
      simpa [tiltedWeight,gammaWeight] using hc.tendsto.mono_left nhdsWithin_le_nhds
    convert ht.mul hs using 1
    · funext q; dsimp [F]; ring
    · ring
  have ht := tendsto_integral_filter_of_dominated_convergence B hm hb hB hl
  rw [gamma_first p hp] at ht
  apply HasDerivWithinAt.Ici_of_Ioi
  rw [hasDerivWithinAt_iff_tendsto_slope' (by simp : (0 : ℝ) ∉ Ioi (0 : ℝ))]
  convert ht using 1
  funext q
  rw [slope_def_field,logNumerator_zero]
  simp only [sub_zero]
  exact (integral_div q (fun z => tiltedWeight p q z * Real.log (1+q*z))).symm

theorem scaledLogMean_deriv (p : ℝ) (hp : 0 < p) :
    HasDerivWithinAt (scaledLogMean p) p (Ici 0) 0 := by
  have hg : Real.Gamma p ≠ 0 := (Real.Gamma_pos_of_pos hp).ne'
  have hd := (logNumerator_deriv p hp).div (normalizer_deriv p hp)
    (by rw [normalizer_zero p hp]; exact hg)
  have he : (Real.Gamma (p+1)*normalizer p 0-logNumerator p 0*(-p*Real.Gamma (p+1)))/
      normalizer p 0^2 = p := by
    rw [normalizer_zero p hp,logNumerator_zero,Real.Gamma_add_one hp.ne']
    simp only [zero_mul, sub_zero]
    field_simp
  rwa [he] at hd

theorem firstJet_eq (p : ℝ) (hp : 0 < p) : firstJet p = p := by
  exact (scaledLogMean_deriv p hp).derivWithin (uniqueDiffWithinAt_Ici 0)


/-- The rescaled quotient is the actual Kummer logarithmic expectation, by
change of variables t=qz. -/
theorem scaledLogMean_actual (p q : ℝ) (hq : 0 < q) :
    scaledLogMean p q = kummerExpect p q⁻¹ (fun t => Real.log (1+t)) := by
  have he (f : ℝ → ℝ) : (∫ t in Ioi (0 : ℝ), f t*kummerWeight p q⁻¹ t) =
      (q*q^(p-1)) * (∫ z in Ioi (0 : ℝ), f (q*z)*tiltedWeight p q z) := by
    have hc := integral_comp_mul_left_Ioi' (fun t => f t*kummerWeight p q⁻¹ t) 0 hq
    simp only [mul_zero, smul_eq_mul] at hc
    rw [← hc]
    have hf : (fun z : ℝ => f (q*z)*kummerWeight p q⁻¹ (q*z)) =ᵐ[volume.restrict (Ioi 0)]
        (fun z => q^(p-1) * (f (q*z)*tiltedWeight p q z)) := by
      filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
      dsimp [kummerWeight,tiltedWeight]
      rw [Real.mul_rpow hq.le (le_of_lt hz)]
      have harg : -q⁻¹*(q*z) = -z := by field_simp
      rw [harg]
      ring
    rw [integral_congr_ae hf, integral_const_mul]
    ring
  have hden : kummerZ p q⁻¹ = (q*q^(p-1))*normalizer p q := by
    simpa [kummerZ, normalizer, tiltedMoment] using he (fun _ => 1)
  have hnum : (∫ t in Ioi (0 : ℝ), Real.log (1+t)*kummerWeight p q⁻¹ t) =
      (q*q^(p-1))*logNumerator p q := by
    simpa [logNumerator,mul_comm] using he (fun t => Real.log (1+t))
  unfold scaledLogMean kummerExpect
  rw [hden,hnum]
  have hn : q*q^(p-1) ≠ 0 := ne_of_gt (by positivity)
  field_simp

end Dobson.GammaFirstJet
#print axioms Dobson.GammaFirstJet.firstJet_eq
