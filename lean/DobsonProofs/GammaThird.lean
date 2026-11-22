import DobsonProofs.GammaHigher
namespace Dobson.GammaHigher
open Set Filter MeasureTheory
open scoped Topology
open GammaAtZero GammaFirstJet

theorem normalizer_second_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun q : ℝ => (normalizer p q-Real.Gamma p+p*Real.Gamma (p+1)*q)/q^2)
      (𝓝[>] 0) (𝓝 (p*(p+1)*Real.Gamma (p+2)/2)) := by
  have hc : ∀ᵐ z ∂volume.restrict (Ioi 0), ContDiffOn ℝ (1+1) (d0 p z) (Icc 0 1) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    exact (d_contDiff p z hz).of_succ
  have hm : ∀ᶠ q : ℝ in 𝓝[>] (0:ℝ), AEStronglyMeasurable
      (fun z => (d0 p z q-taylorWithinEval (d0 p z) 1 (Icc 0 1) 0 q)/q^(1+1))
      (volume.restrict (Ioi 0)) := by
    apply Filter.Eventually.of_forall
    intro q
    have hmeas : Measurable (fun z => (d0 p z q-(gammaWeight p z-p*z*gammaWeight p z*q))/q^2) := by
      unfold d0 gammaWeight
      fun_prop
    apply hmeas.aestronglyMeasurable.congr
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    rw [d_taylor_one p z q hz]
  have hb : ∀ᵐ z ∂volume.restrict (Ioi 0), ∀ q ∈ Icc (0:ℝ) 1,
      ‖iteratedDerivWithin (1+1) (d0 p z) (Icc 0 1) q‖ ≤ envelope p 2 0 (p*(p+1)) z := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    intro q hq
    rw [(d_jets p z hz).2.1 q hq]
    exact d2_bound p z q hp hz hq
  have ht := GammaTaylor.integral_taylor_next (volume.restrict (Ioi 0)) 1
    (fun q z => d0 p z q) (envelope p 2 0 (p*(p+1))) hc hm (envelope_integrable p hp 2 _ _) hb
  have he : (∫ z in Ioi (0:ℝ), iteratedDerivWithin 2 (d0 p z) (Icc 0 1) 0 / (2:ℕ).factorial) =
      p*(p+1)*Real.Gamma (p+2)/2 := by
    have hh : (fun z => iteratedDerivWithin 2 (d0 p z) (Icc 0 1) 0 / (2:ℕ).factorial)
        =ᵐ[volume.restrict (Ioi 0)] (fun z => (p*(p+1)/2)*(z^2*gammaWeight p z)) := by
      filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
      rw [(d_jets p z hz).2.1 0 (by norm_num)]
      norm_num [d2]
      <;> ring
    rw [integral_congr_ae hh,integral_const_mul,moment_integral p 2 hp]
    norm_num
    <;> ring
  change Tendsto _ _ (𝓝 (∫ z in Ioi (0:ℝ), iteratedDerivWithin 2 (d0 p z) (Icc 0 1) 0 / (2:ℕ).factorial)) at ht
  rw [he] at ht
  apply ht.congr'
  filter_upwards [Ioc_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with q hq
  have hh : (fun z => (d0 p z q-taylorWithinEval (d0 p z) 1 (Icc 0 1) 0 q)/q^(1+1))
      =ᵐ[volume.restrict (Ioi 0)] (fun z => (d0 p z q-gammaWeight p z+p*(z*gammaWeight p z)*q)/q^2) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    rw [d_taylor_one p z q hz]
    ring
  have hg : IntegrableOn (gammaWeight p) (Ioi 0) := gammaWeight_integrable p hp
  have hi : IntegrableOn (fun z => p*(z*gammaWeight p z)*q) (Ioi 0) := by
    change Integrable (fun z => p*(z*gammaWeight p z)*q) (volume.restrict (Ioi 0))
    simpa only [pow_one] using ((gamma_moment_integrable p 1 hp).const_mul p).mul_const q
  rw [integral_congr_ae hh,integral_div,integral_add (f := fun z => d0 p z q-gammaWeight p z) ((d0_integrable p q hp hq.1.le).sub hg) hi,
    integral_sub (d0_integrable p q hp hq.1.le) hg,d0_integral,
    integral_mul_const,integral_const_mul,gamma_first p hp]
  have hgamma : (∫ z in Ioi (0:ℝ), gammaWeight p z) = Real.Gamma p := by
    simpa using moment_integral p 0 hp
  rw [hgamma]

theorem numerator_third_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun q : ℝ =>
      (logNumerator p q-Real.Gamma (p+1)*q+(2*p+1)*Real.Gamma (p+2)*q^2/2)/q^3)
      (𝓝[>] 0) (𝓝 ((3*p^2+6*p+2)*Real.Gamma (p+3)/6)) := by
  have hc : ∀ᵐ z ∂volume.restrict (Ioi 0), ContDiffOn ℝ (2+1) (n0 p z) (Icc 0 1) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    exact n_contDiff p z hz
  have hm : ∀ᶠ q : ℝ in 𝓝[>] (0:ℝ), AEStronglyMeasurable
      (fun z => (n0 p z q-taylorWithinEval (n0 p z) 2 (Icc 0 1) 0 q)/q^(2+1))
      (volume.restrict (Ioi 0)) := by
    apply Filter.Eventually.of_forall
    intro q
    have hmeas : Measurable (fun z =>
        (n0 p z q-(z*gammaWeight p z*q-(2*p+1)*z^2*gammaWeight p z*q^2/2))/q^3) := by
      unfold n0 gammaWeight
      fun_prop
    apply hmeas.aestronglyMeasurable.congr
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    rw [n_taylor_two p z q hz]
  have hb : ∀ᵐ z ∂volume.restrict (Ioi 0), ∀ q ∈ Icc (0:ℝ) 1,
      ‖iteratedDerivWithin (2+1) (n0 p z) (Icc 0 1) q‖ ≤
        envelope p 3 (-p*(p+1)*(p+2)) (3*p^2+6*p+2) z := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    intro q hq
    rw [(n_jets p z hz).2.2 q hq]
    exact n3_bound p z q hp hz hq
  have ht := GammaTaylor.integral_taylor_next (volume.restrict (Ioi 0)) 2
    (fun q z => n0 p z q) (envelope p 3 (-p*(p+1)*(p+2)) (3*p^2+6*p+2)) hc hm
    (envelope_integrable p hp 3 _ _) hb
  have he : (∫ z in Ioi (0:ℝ), iteratedDerivWithin 3 (n0 p z) (Icc 0 1) 0 / (3:ℕ).factorial) =
      (3*p^2+6*p+2)*Real.Gamma (p+3)/6 := by
    have hh : (fun z => iteratedDerivWithin 3 (n0 p z) (Icc 0 1) 0 / (3:ℕ).factorial)
        =ᵐ[volume.restrict (Ioi 0)] (fun z => ((3*p^2+6*p+2)/6)*(z^3*gammaWeight p z)) := by
      filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
      rw [(n_jets p z hz).2.2 0 (by norm_num)]
      norm_num [n3]
      <;> ring
    rw [integral_congr_ae hh,integral_const_mul,moment_integral p 3 hp]
    norm_num
    <;> ring
  change Tendsto _ _ (𝓝 (∫ z in Ioi (0:ℝ), iteratedDerivWithin 3 (n0 p z) (Icc 0 1) 0 / (3:ℕ).factorial)) at ht
  rw [he] at ht
  apply ht.congr'
  filter_upwards [Ioc_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with q hq
  have hh : (fun z => (n0 p z q-taylorWithinEval (n0 p z) 2 (Icc 0 1) 0 q)/q^(2+1))
      =ᵐ[volume.restrict (Ioi 0)] (fun z =>
        (n0 p z q-z*gammaWeight p z*q+(2*p+1)*(z^2*gammaWeight p z)*q^2/2)/q^3) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    rw [n_taylor_two p z q hz]
    ring
  have hi1 : IntegrableOn (fun z => z*gammaWeight p z*q) (Ioi 0) := by
    change Integrable (fun z => z*gammaWeight p z*q) (volume.restrict (Ioi 0))
    simpa only [pow_one] using (gamma_moment_integrable p 1 hp).mul_const q
  have hi2 : IntegrableOn (fun z => (2*p+1)*(z^2*gammaWeight p z)*q^2/2) (Ioi 0) :=
    (((gamma_moment_integrable p 2 hp).const_mul (2*p+1)).mul_const (q^2)).div_const 2
  rw [integral_congr_ae hh,integral_div,
    integral_add (f := fun z => n0 p z q-z*gammaWeight p z*q) ((n0_integrable p q hp ⟨hq.1.le,hq.2⟩).sub hi1) hi2,
    integral_sub (n0_integrable p q hp ⟨hq.1.le,hq.2⟩) hi1,n0_integral,
    integral_mul_const,gamma_first p hp,integral_div,integral_mul_const,integral_const_mul,
    moment_integral p 2 hp]
  norm_num


noncomputable def thirdJet (p : ℝ) : ℝ :=
  limUnder (𝓝[>] (0:ℝ)) (fun q => (scaledLogMean p q-p*q-secondJet p*q^2)/q^3)

theorem third_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun q : ℝ => (scaledLogMean p q-p*q-secondJet p*q^2)/q^3)
      (𝓝[>] 0) (𝓝 (p*(10*p^2+9*p+2)/3)) := by
  have hg : Real.Gamma p ≠ 0 := (Real.Gamma_pos_of_pos hp).ne'
  have hd : Tendsto (fun q : ℝ => (normalizer p q-Real.Gamma p)/q)
      (𝓝[>] 0) (𝓝 (-p*Real.Gamma (p+1))) := by
    have hh := (normalizer_deriv p hp).mono Ioi_subset_Ici_self
    rw [hasDerivWithinAt_iff_tendsto_slope' (by simp : (0:ℝ)∉Ioi (0:ℝ))] at hh
    simpa [slope_fun_def_field,normalizer_zero p hp] using hh
  have hz : Tendsto (normalizer p) (𝓝[>] (0:ℝ)) (𝓝 (Real.Gamma p)) := by
    have hh := (normalizer_deriv p hp).continuousWithinAt.tendsto.mono_left
      (nhdsWithin_mono (0:ℝ) Ioi_subset_Ici_self)
    simpa [normalizer_zero p hp] using hh
  have ht := (((numerator_third_limit p hp).sub ((normalizer_second_limit p hp).const_mul p)).sub
    (hd.const_mul (secondJet p))).div hz hg
  have hG2 : Real.Gamma (p+2) = (p+1)*Real.Gamma (p+1) := by
    convert Real.Gamma_add_one (show p+1 ≠ 0 by positivity) using 1 <;> congr 1 <;> ring
  have hG3 : Real.Gamma (p+3) = (p+2)*Real.Gamma (p+2) := by
    convert Real.Gamma_add_one (show p+2 ≠ 0 by positivity) using 1 <;> congr 1 <;> ring
  have he : ((3*p^2+6*p+2)*Real.Gamma (p+3)/6-p*(p*(p+1)*Real.Gamma (p+2)/2)-
      secondJet p*(-p*Real.Gamma (p+1)))/Real.Gamma p = p*(10*p^2+9*p+2)/3 := by
    rw [hG3,hG2,Real.Gamma_add_one hp.ne',secondJet_eq p hp]
    field_simp
    <;> ring
  rw [he] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,hz.eventually (eventually_ne_nhds hg)] with q hq hZ
  change 0 < q at hq
  dsimp only [scaledLogMean,Pi.div_apply,Pi.sub_apply,Pi.mul_apply]
  rw [hG2,Real.Gamma_add_one hp.ne',secondJet_eq p hp]
  field_simp [hq.ne',hZ]
  <;> ring

theorem thirdJet_eq (p : ℝ) (hp : 0 < p) : thirdJet p = p*(10*p^2+9*p+2)/3 :=
  (third_limit p hp).limUnder_eq

end Dobson.GammaHigher
#print axioms Dobson.GammaHigher.normalizer_second_limit
#print axioms Dobson.GammaHigher.numerator_third_limit

#print axioms Dobson.GammaHigher.thirdJet_eq
