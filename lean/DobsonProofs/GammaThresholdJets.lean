import DobsonProofs.GammaHigher
import DobsonProofs.Coverage5GammaSmallStay

namespace Dobson.GammaThresholdJets
open Set Filter MeasureTheory
open scoped Topology
open GammaFirstJet GammaHigher Coverage2Exact Coverage5GammaSmallStay NoiseAnalyticity

noncomputable def longThreshold (p ε : ℝ) : ℝ := gammaThreshold p (1-ε)
noncomputable def longQ (p ε : ℝ) : ℝ := (thresholdKappa p (1-ε))⁻¹

theorem longQ_formula (p ε : ℝ) :
    longQ p ε = ε*Real.exp (longThreshold p ε)/(p*(1-ε)^2) := by
  unfold longQ thresholdKappa exactKappa longThreshold
  rw [inv_div]
  congr 1
  · ring
  · ring

theorem longQ_pos (p ε : ℝ) (hp : 0 < p) (hε : 0 < ε) (hε1 : ε < 1) : 0 < longQ p ε := by
  exact inv_pos.mpr (thresholdKappa_pos p (1-ε) hp (by linarith) (by linarith))

theorem longQ_zero (p : ℝ) : longQ p 0 = 0 := by simp [longQ_formula]

theorem longThreshold_zero (p : ℝ) : longThreshold p 0 = 0 := by
  simp [longThreshold,gammaThreshold]

theorem long_mean (p ε : ℝ) (hp : 0 < p) (hε : 0 < ε) (hε1 : ε < 1) :
    scaledLogMean p (longQ p ε) = -Real.log (1-ε) := by
  rw [scaledLogMean_actual p _ (longQ_pos p ε hp hε hε1)]
  dsimp only [longQ]
  rw [inv_inv]
  change kummerLog1p p (thresholdKappa p (1-ε)) = _
  rw [threshold_mean p (1-ε) hp (by linarith) (by linarith)]
  simp

theorem longQ_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (longQ p) (𝓝[>] 0) (𝓝[>] 0) := by
  have hu : Tendsto (fun ε : ℝ => ε*Real.exp (gammaLogLoss p)/(p*(1-ε)^2))
      (𝓝[>] 0) (𝓝 0) := by
    have hc : ContinuousAt (fun ε : ℝ => ε*Real.exp (gammaLogLoss p)/(p*(1-ε)^2)) 0 := by
      apply ContinuousAt.div (by fun_prop) (by fun_prop)
      simpa using hp.ne'
    simpa using hc.tendsto.mono_left nhdsWithin_le_nhds
  have hl : Tendsto (longQ p) (𝓝[>] 0) (𝓝 0) := by
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hu
    · filter_upwards [Ioo_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with ε hε
      exact (longQ_pos p ε hp hε.1 hε.2).le
    · filter_upwards [Ioo_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with ε hε
      rw [longQ_formula]
      apply div_le_div_of_nonneg_right _ (by positivity)
      apply mul_le_mul_of_nonneg_left _ hε.1.le
      exact Real.exp_le_exp.mpr (gammaThreshold_spec p (1-ε) hp (by linarith [hε.2])
        (by linarith [hε.1])).2.2.le
  apply tendsto_nhdsWithin_iff.mpr
  refine ⟨hl,?_⟩
  filter_upwards [Ioo_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with ε hε
  exact longQ_pos p ε hp hε.1 hε.2

theorem scaledLogMean_ratio (p : ℝ) (hp : 0 < p) :
    Tendsto (fun q : ℝ => scaledLogMean p q/q) (𝓝[>] 0) (𝓝 p) := by
  have hh := (scaledLogMean_deriv p hp).mono Ioi_subset_Ici_self
  rw [hasDerivWithinAt_iff_tendsto_slope' (by simp : (0:ℝ)∉Ioi (0:ℝ))] at hh
  simpa [slope_fun_def_field,scaledLogMean,logNumerator_zero] using hh

theorem negative_log_ratio :
    Tendsto (fun ε : ℝ => -Real.log (1-ε)/ε) (𝓝[>] 0) (𝓝 1) := by
  have hd : HasDerivAt (fun ε : ℝ => -Real.log (1-ε)) 1 0 := by
    convert ((((hasDerivAt_id (0:ℝ)).const_sub 1).log (by norm_num)).neg) using 1 <;> norm_num <;> rfl
  simpa [slope_fun_def_field] using hd.tendsto_slope.mono_left (nhdsGT_le_nhdsNE 0)

theorem longQ_ratio (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ => longQ p ε/ε) (𝓝[>] 0) (𝓝 (1/p)) := by
  have hd := (scaledLogMean_ratio p hp).comp (longQ_tendsto p hp)
  have ht := negative_log_ratio.div hd hp.ne'
  apply ht.congr'
  filter_upwards [Ioo_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with ε hε
  dsimp only [Pi.div_apply,Function.comp_def]
  rw [long_mean p ε hp hε.1 hε.2]
  have hm : -Real.log (1-ε) ≠ 0 := ne_of_gt (by
    have hh := Real.log_neg (show 0 < 1-ε by linarith [hε.2]) (show 1-ε<1 by linarith [hε.1])
    linarith)
  have hlog : Real.log (1-ε) ≠ 0 := by simpa using hm
  field_simp [hlog,hε.1.ne',(longQ_pos p ε hp hε.1 hε.2).ne']

end Dobson.GammaThresholdJets
#print axioms Dobson.GammaThresholdJets.longQ_ratio

namespace Dobson.GammaThresholdJets
open Set Filter
open scoped Topology

theorem exp_taylor_two (c x : ℝ) :
    taylorWithinEval (fun t : ℝ => Real.exp (c*t)) 2 univ 0 x =
      1+c*x+c^2*x^2/2 := by
  have hd : deriv (fun t : ℝ => Real.exp (c*t)) 0 = c := by
    simpa using (((hasDerivAt_id (0:ℝ)).const_mul c).exp).deriv
  rw [taylorWithinEval_succ (n := 1),taylorWithinEval_succ (n := 0)]
  simp [iteratedDerivWithin_univ,iteratedDeriv_exp_const_mul,taylor_within_zero_eval,hd]
  <;> ring

theorem exp_second_limit (c : ℝ) :
    Tendsto (fun x : ℝ => (Real.exp (c*x)-1-c*x)/x^2) (𝓝[≠] 0) (𝓝 (c^2/2)) := by
  have hc : ContDiff ℝ 2 (fun x : ℝ => Real.exp (c*x)) := by fun_prop
  have ht := Real.taylor_tendsto convex_univ (mem_univ (0:ℝ)) hc.contDiffOn
  rw [nhdsWithin_univ] at ht
  have hh := (ht.mono_left (show (𝓝[≠] (0:ℝ)) ≤ 𝓝 (0:ℝ) from nhdsWithin_le_nhds)).add_const (c^2/2)
  simp only [zero_add] at hh
  apply Tendsto.congr' _ hh
  filter_upwards [self_mem_nhdsWithin] with x hx
  rw [exp_taylor_two]
  simp only [sub_zero]
  have hx0 : x ≠ 0 := hx
  field_simp [hx0]
  <;> ring

theorem exp_neg_second_limit :
    Tendsto (fun x : ℝ => (1-Real.exp (-x)-x)/x^2) (𝓝[>] 0) (𝓝 (-(1:ℝ)/2)) := by
  have ht := ((exp_second_limit (-1)).mono_left (nhdsGT_le_nhdsNE 0)).neg
  convert ht using 1 <;> simp only [neg_mul,one_mul,even_two,Even.neg_pow,one_pow]
  · ext x
    ring
  · norm_num
end Dobson.GammaThresholdJets
#print axioms Dobson.GammaThresholdJets.exp_neg_second_limit


namespace Dobson.GammaThresholdJets
open Set Filter MeasureTheory
open scoped Topology
open GammaFirstJet GammaHigher Coverage2Exact Coverage5GammaSmallStay NoiseAnalyticity

theorem mean_at_threshold_tendsto :
    Tendsto (fun ε : ℝ => -Real.log (1-ε)) (𝓝[>] 0) (𝓝[>] 0) := by
  apply tendsto_nhdsWithin_iff.mpr
  constructor
  · have hc : ContinuousAt (fun ε : ℝ => -Real.log (1-ε)) 0 := by
      exact (((continuousAt_id.const_sub 1).log (by norm_num)).neg)
    simpa using hc.tendsto.mono_left nhdsWithin_le_nhds
  · filter_upwards [Ioo_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with ε hε
    exact neg_pos.mpr (Real.log_neg (by linarith [hε.2]) (by linarith [hε.1]))

theorem epsilon_quadratic_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun ε : ℝ => (ε-p*longQ p ε)/(longQ p ε)^2) (𝓝[>] 0)
      (𝓝 (-p*(3*p+1)/2-p^2/2)) := by
  have he := exp_neg_second_limit.comp mean_at_threshold_tendsto
  have hm := (scaledLogMean_ratio p hp).comp (longQ_tendsto p hp)
  have h2 := (second_limit p hp).comp (longQ_tendsto p hp)
  have ht := (he.mul (hm.pow 2)).add h2
  convert Tendsto.congr' ?_ ht using 1
  · ring
  filter_upwards [Ioo_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with ε hε
  dsimp only [Pi.add_apply,Pi.mul_apply,Function.comp_def]
  rw [long_mean p ε hp hε.1 hε.2]
  rw [neg_neg,Real.exp_log (show 0<1-ε by linarith [hε.2])]
  have hq : longQ p ε ≠ 0 := (longQ_pos p ε hp hε.1 hε.2).ne'
  have hm0 : Real.log (1-ε) ≠ 0 := (Real.log_neg (by linarith [hε.2]) (by linarith [hε.1])).ne
  field_simp [hq,hm0]
  <;> ring

noncomputable def thresholdRatio (p ε : ℝ) : ℝ :=
  if ε=0 then 1 else ε/(p*longQ p ε)

theorem thresholdRatio_zero (p : ℝ) : thresholdRatio p 0 = 1 := by simp [thresholdRatio]

theorem thresholdRatio_deriv (p : ℝ) (hp : 0 < p) :
    HasDerivWithinAt (thresholdRatio p) (-(4*p+1)/(2*p)) (Ioi 0) 0 := by
  rw [hasDerivWithinAt_iff_tendsto_slope' (by simp : (0:ℝ)∉Ioi (0:ℝ))]
  have ht := ((epsilon_quadratic_limit p hp).mul (longQ_ratio p hp)).div_const p
  convert Tendsto.congr' ?_ ht using 1
  · field_simp
    <;> ring
  filter_upwards [Ioo_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with ε hε
  simp only [slope_fun_def_field,thresholdRatio_zero,sub_zero,Pi.div_apply,Pi.mul_apply]
  rw [thresholdRatio,if_neg hε.1.ne']
  field_simp [hp.ne',hε.1.ne',(longQ_pos p ε hp hε.1 hε.2).ne']
  <;> ring

theorem threshold_log_identity (p ε : ℝ) (hp : 0 < p) (hε : 0 < ε) (hε1 : ε<1) :
    longThreshold p ε = -Real.log (thresholdRatio p ε)+2*Real.log (1-ε) := by
  have hp0 := hp.ne'
  have hε0 := hε.ne'
  have hs : 0<1-ε := by linarith
  have hq := (longQ_pos p ε hp hε hε1).ne'
  have hR : thresholdRatio p ε = (1-ε)^2/Real.exp (longThreshold p ε) := by
    rw [thresholdRatio,if_neg hε.ne',longQ_formula]
    field_simp [hp0,hε0,hs.ne',Real.exp_ne_zero]
  rw [hR,Real.log_div (pow_ne_zero 2 hs.ne') (Real.exp_ne_zero _),Real.log_exp,
    Real.log_pow]
  ring

noncomputable def firstThresholdJet (p : ℝ) : ℝ :=
  limUnder (𝓝[>] (0:ℝ)) (fun ε => longThreshold p ε/ε)

theorem firstThreshold_limit (p : ℝ) (hp : 0<p) :
    Tendsto (fun ε : ℝ => longThreshold p ε/ε) (𝓝[>] 0) (𝓝 (1/(2*p))) := by
  have hd := (thresholdRatio_deriv p hp).log (by simp [thresholdRatio_zero])
  rw [hasDerivWithinAt_iff_tendsto_slope' (by simp : (0:ℝ)∉Ioi (0:ℝ))] at hd
  simp only [thresholdRatio_zero,div_one,Real.log_one,sub_zero,slope_fun_def_field] at hd
  have ht := hd.neg.sub (negative_log_ratio.const_mul 2)
  convert Tendsto.congr' ?_ ht using 1
  · field_simp
    <;> ring
  filter_upwards [Ioo_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with ε hε
  rw [threshold_log_identity p ε hp hε.1 hε.2]
  ring

theorem firstThresholdJet_eq (p : ℝ) (hp : 0<p) : firstThresholdJet p=1/(2*p) :=
  (firstThreshold_limit p hp).limUnder_eq
end Dobson.GammaThresholdJets
#print axioms Dobson.GammaThresholdJets.firstThresholdJet_eq
