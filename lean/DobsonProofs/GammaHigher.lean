import DobsonProofs.GammaFirstJet

namespace Dobson.GammaTaylor
open Set Filter MeasureTheory
open scoped Topology

/-- Dominated Taylor quotients on the right half-line. This lemma does not
assume an integral extension to negative parameters. -/
theorem integral_taylor_next {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (n : ℕ) (F : ℝ → Ω → ℝ) (B : Ω → ℝ)
    (hc : ∀ᵐ ω ∂μ, ContDiffOn ℝ (n+1) (fun q => F q ω) (Icc 0 1))
    (hm : ∀ᶠ q : ℝ in 𝓝[>] (0:ℝ), AEStronglyMeasurable
      (fun ω => (F q ω-taylorWithinEval (fun t => F t ω) n (Icc 0 1) 0 q)/q^(n+1)) μ)
    (hB : Integrable B μ)
    (hb : ∀ᵐ ω ∂μ, ∀ q ∈ Icc (0:ℝ) 1,
      ‖iteratedDerivWithin (n+1) (fun t => F t ω) (Icc 0 1) q‖ ≤ B ω) :
    Tendsto (fun q : ℝ => ∫ ω,
      (F q ω-taylorWithinEval (fun t => F t ω) n (Icc 0 1) 0 q)/q^(n+1) ∂μ)
      (𝓝[>] (0:ℝ))
      (𝓝 (∫ ω, iteratedDerivWithin (n+1) (fun t => F t ω) (Icc 0 1) 0 /
        ((n+1).factorial:ℝ) ∂μ)) := by
  have he : ∀ᶠ q : ℝ in 𝓝[>] (0 : ℝ), q ∈ Ioc (0 : ℝ) 1 :=
    Ioc_mem_nhdsGT zero_lt_one
  have hbd : ∀ᶠ q : ℝ in 𝓝[>] (0:ℝ), ∀ᵐ ω ∂μ,
      ‖(F q ω-taylorWithinEval (fun t => F t ω) n (Icc 0 1) 0 q)/q^(n+1)‖ ≤
        B ω/(n.factorial:ℝ) := by
    filter_upwards [he] with q hq
    filter_upwards [hc,hb] with ω hc hb
    have h := taylor_mean_remainder_bound (n := n) (by norm_num : (0:ℝ)≤1)
      hc ⟨hq.1.le,hq.2⟩ hb
    rw [norm_div,Real.norm_of_nonneg (pow_pos hq.1 _).le,div_le_iff₀ (pow_pos hq.1 _)]
    simpa only [sub_zero,div_mul_eq_mul_div] using h
  have hl : ∀ᵐ ω ∂μ, Tendsto
      (fun q : ℝ => (F q ω-taylorWithinEval (fun t => F t ω) n (Icc 0 1) 0 q)/q^(n+1))
      (𝓝[>] (0:ℝ))
      (𝓝 (iteratedDerivWithin (n+1) (fun t => F t ω) (Icc 0 1) 0 /
        ((n+1).factorial:ℝ))) := by
    filter_upwards [hc] with ω hc
    have ht := Real.taylor_tendsto (convex_Icc (0:ℝ) 1)
      (by constructor <;> norm_num : (0:ℝ)∈Icc (0:ℝ) 1) hc
    have hfilter : 𝓝[>] (0:ℝ) ≤ 𝓝[Icc (0:ℝ) 1] 0 := by
      rw [nhdsWithin_Icc_eq_nhdsGE zero_lt_one]
      exact nhdsWithin_mono _ Ioi_subset_Ici_self
    have ht' := (ht.mono_left hfilter).add_const
      (iteratedDerivWithin (n+1) (fun t => F t ω) (Icc 0 1) 0 / ((n+1).factorial:ℝ))
    simp only [zero_add] at ht'
    apply Tendsto.congr' _ ht'
    filter_upwards [he] with q hq
    rw [taylorWithinEval_succ]
    simp only [sub_zero,smul_eq_mul,Nat.factorial_succ,Nat.cast_mul,Nat.cast_add,Nat.cast_one]
    have hn : (n.factorial:ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero n
    have hn1 : (n:ℝ)+1 ≠ 0 := by positivity
    field_simp [hq.1.ne',hn,hn1]
    <;> ring
  exact tendsto_integral_filter_of_dominated_convergence (fun ω => B ω/(n.factorial:ℝ))
    hm hbd (hB.div_const _) hl

end Dobson.GammaTaylor
#print axioms Dobson.GammaTaylor.integral_taylor_next


namespace Dobson.GammaHigher
open Set Filter MeasureTheory
open scoped Topology
open GammaAtZero GammaFirstJet

noncomputable def d0 (p z q : ℝ) := (1+q*z)^(-p)*gammaWeight p z
noncomputable def d1 (p z q : ℝ) := -p*z*(1+q*z)^(-p-1)*gammaWeight p z
noncomputable def d2 (p z q : ℝ) := p*(p+1)*z^2*(1+q*z)^(-p-2)*gammaWeight p z
noncomputable def d3 (p z q : ℝ) := -p*(p+1)*(p+2)*z^3*(1+q*z)^(-p-3)*gammaWeight p z
noncomputable def n0 (p z q : ℝ) := (1+q*z)^(-p)*Real.log (1+q*z)*gammaWeight p z
noncomputable def n1 (p z q : ℝ) := z*(1+q*z)^(-p-1)*(1-p*Real.log (1+q*z))*gammaWeight p z
noncomputable def n2 (p z q : ℝ) := z^2*(1+q*z)^(-p-2)*(p*(p+1)*Real.log (1+q*z)-(2*p+1))*gammaWeight p z
noncomputable def n3 (p z q : ℝ) := z^3*(1+q*z)^(-p-3)*
  (-p*(p+1)*(p+2)*Real.log (1+q*z)+(3*p^2+6*p+2))*gammaWeight p z

private theorem upow (z q a : ℝ) (hu : 0 < 1+q*z) :
    HasDerivAt (fun t : ℝ => (1+t*z)^a) (z*a*(1+q*z)^(a-1)) q := by
  convert (((hasDerivAt_id q).mul_const z).const_add 1).rpow_const (p := a) (Or.inl hu.ne') using 1 <;> simp

private theorem ulog (z q : ℝ) (hu : 0 < 1+q*z) :
    HasDerivAt (fun t : ℝ => Real.log (1+t*z)) (z/(1+q*z)) q := by
  convert (((hasDerivAt_id q).mul_const z).const_add 1).log hu.ne' using 1 <;> simp

theorem d0_deriv (p z q : ℝ) (hu : 0 < 1+q*z) :
    HasDerivAt (d0 p z) (d1 p z q) q := by
  convert (upow z q (-p) hu).mul_const (gammaWeight p z) using 1
  · rfl
  · dsimp [d1]; ring

theorem d1_deriv (p z q : ℝ) (hu : 0 < 1+q*z) :
    HasDerivAt (d1 p z) (d2 p z q) q := by
  convert ((upow z q (-p-1) hu).const_mul (-p*z)).mul_const (gammaWeight p z) using 1
  · rfl
  · dsimp [d2]
    rw [show -p-1-1 = -p-2 by ring]
    ring

theorem d2_deriv (p z q : ℝ) (hu : 0 < 1+q*z) :
    HasDerivAt (d2 p z) (d3 p z q) q := by
  convert ((upow z q (-p-2) hu).const_mul (p*(p+1)*z^2)).mul_const (gammaWeight p z) using 1
  · rfl
  · dsimp [d3]
    rw [show -p-2-1 = -p-3 by ring]
    ring

theorem n0_deriv (p z q : ℝ) (hu : 0 < 1+q*z) :
    HasDerivAt (n0 p z) (n1 p z q) q := by
  convert ((upow z q (-p) hu).mul (ulog z q hu)).mul_const (gammaWeight p z) using 1
  · rfl
  · dsimp [n1]
    rw [Real.rpow_sub_one hu.ne']
    ring

theorem n1_deriv (p z q : ℝ) (hu : 0 < 1+q*z) :
    HasDerivAt (n1 p z) (n2 p z q) q := by
  convert (((upow z q (-p-1) hu).const_mul z).mul
    ((ulog z q hu).const_mul p |>.const_sub 1)).mul_const (gammaWeight p z) using 1
  · rfl
  · dsimp [n2]
    rw [show -p-1-1 = -p-2 by ring, show -p-2 = (-p-1)-1 by ring,
      Real.rpow_sub_one hu.ne']
    ring

theorem n2_deriv (p z q : ℝ) (hu : 0 < 1+q*z) :
    HasDerivAt (n2 p z) (n3 p z q) q := by
  convert (((upow z q (-p-2) hu).const_mul (z^2)).mul
    ((ulog z q hu).const_mul (p*(p+1)) |>.sub_const (2*p+1))).mul_const (gammaWeight p z) using 1
  · rfl
  · dsimp [n3]
    rw [show -p-2-1 = -p-3 by ring, show -p-3 = (-p-2)-1 by ring,
      Real.rpow_sub_one hu.ne']
    ring


private theorem iterated_of_derivatives (f f1 f2 f3 : ℝ → ℝ)
    (h0 : ∀ q ∈ Icc (0:ℝ) 1, HasDerivAt f (f1 q) q)
    (h1 : ∀ q ∈ Icc (0:ℝ) 1, HasDerivAt f1 (f2 q) q)
    (h2 : ∀ q ∈ Icc (0:ℝ) 1, HasDerivAt f2 (f3 q) q) :
    (∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 1 f (Icc 0 1) q = f1 q) ∧
    (∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 2 f (Icc 0 1) q = f2 q) ∧
    (∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 3 f (Icc 0 1) q = f3 q) := by
  have hdiff := uniqueDiffOn_Icc (by norm_num : (0:ℝ)<1)
  have hfirst : ∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 1 f (Icc 0 1) q = f1 q := by
    intro q hq
    rw [iteratedDerivWithin_succ,iteratedDerivWithin_zero]
    exact (h0 q hq).hasDerivWithinAt.derivWithin (hdiff q hq)
  have hsecond : ∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 2 f (Icc 0 1) q = f2 q := by
    intro q hq
    rw [show 2=1+1 from rfl,iteratedDerivWithin_succ,
      derivWithin_congr hfirst (hfirst q hq)]
    exact (h1 q hq).hasDerivWithinAt.derivWithin (hdiff q hq)
  refine ⟨hfirst,hsecond,?_⟩
  intro q hq
  rw [show 3=2+1 from rfl,iteratedDerivWithin_succ,
    derivWithin_congr hsecond (hsecond q hq)]
  exact (h2 q hq).hasDerivWithinAt.derivWithin (hdiff q hq)

theorem d_jets (p z : ℝ) (hz : 0 < z) :
    (∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 1 (d0 p z) (Icc 0 1) q = d1 p z q) ∧
    (∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 2 (d0 p z) (Icc 0 1) q = d2 p z q) ∧
    (∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 3 (d0 p z) (Icc 0 1) q = d3 p z q) := by
  apply iterated_of_derivatives
  · intro q hq; exact d0_deriv p z q (by nlinarith [hq.1])
  · intro q hq; exact d1_deriv p z q (by nlinarith [hq.1])
  · intro q hq; exact d2_deriv p z q (by nlinarith [hq.1])

theorem n_jets (p z : ℝ) (hz : 0 < z) :
    (∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 1 (n0 p z) (Icc 0 1) q = n1 p z q) ∧
    (∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 2 (n0 p z) (Icc 0 1) q = n2 p z q) ∧
    (∀ q ∈ Icc (0:ℝ) 1, iteratedDerivWithin 3 (n0 p z) (Icc 0 1) q = n3 p z q) := by
  apply iterated_of_derivatives
  · intro q hq; exact n0_deriv p z q (by nlinarith [hq.1])
  · intro q hq; exact n1_deriv p z q (by nlinarith [hq.1])
  · intro q hq; exact n2_deriv p z q (by nlinarith [hq.1])

theorem d_contDiff (p z : ℝ) (hz : 0 < z) :
    ContDiffOn ℝ 3 (d0 p z) (Icc 0 1) := by
  unfold d0
  apply ContDiffOn.mul _ contDiffOn_const
  exact (contDiffOn_const.add (contDiffOn_id.mul contDiffOn_const)).rpow_const_of_ne
    (fun q hq => ne_of_gt (by nlinarith [hq.1] : 0 < 1+q*z))

theorem n_contDiff (p z : ℝ) (hz : 0 < z) :
    ContDiffOn ℝ 3 (n0 p z) (Icc 0 1) := by
  unfold n0
  apply ContDiffOn.mul _ contDiffOn_const
  apply ContDiffOn.mul
  · exact (contDiffOn_const.add (contDiffOn_id.mul contDiffOn_const)).rpow_const_of_ne
      (fun q hq => ne_of_gt (by nlinarith [hq.1] : 0 < 1+q*z))
  · exact (contDiffOn_const.add (contDiffOn_id.mul contDiffOn_const)).log
      (fun q hq => ne_of_gt (by nlinarith [hq.1] : 0 < 1+q*z))


noncomputable def envelope (p : ℝ) (n : ℕ) (A B z : ℝ) :=
  (|A| *z^(n+1)+|B| *z^n)*gammaWeight p z

theorem envelope_integrable (p : ℝ) (hp : 0 < p) (n : ℕ) (A B : ℝ) :
    IntegrableOn (envelope p n A B) (Ioi 0) := by
  have hi := ((gamma_moment_integrable p (n+1) hp).const_mul |A|).add
    ((gamma_moment_integrable p n hp).const_mul |B|)
  change Integrable (envelope p n A B) (volume.restrict (Ioi 0))
  convert hi using 1
  funext z
  dsimp [envelope,Pi.add_apply]
  ring

theorem log_power_bound (p z q a A B : ℝ) (n : ℕ) (hz : 0 < z)
    (hq : q ∈ Icc (0:ℝ) 1) (ha : a ≤ 0) :
    ‖z^n*(1+q*z)^a*(A*Real.log (1+q*z)+B)*gammaWeight p z‖ ≤
      envelope p n A B z := by
  have hu : 1 ≤ 1+q*z := by nlinarith [hq.1]
  have ht0 : 0 ≤ (1+q*z)^a := Real.rpow_nonneg (by linarith) _
  have ht1 : (1+q*z)^a ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos hu ha
  have hl0 : 0 ≤ Real.log (1+q*z) := Real.log_nonneg hu
  have hl1 : Real.log (1+q*z) ≤ z := by
    have := Real.log_le_sub_one_of_pos (show 0 < 1+q*z by linarith)
    nlinarith [hq.2]
  have hw : 0 ≤ gammaWeight p z := by dsimp [gammaWeight]; positivity
  have hA : |A*Real.log (1+q*z)+B| ≤ |A| *z+|B| := by
    calc
      _ ≤ |A*Real.log (1+q*z)| +|B| := abs_add_le _ _
      _ = |A| *Real.log (1+q*z)+|B| := by rw [abs_mul,abs_of_nonneg hl0]
      _ ≤ _ := by nlinarith [mul_le_mul_of_nonneg_left hl1 (abs_nonneg A)]
  rw [Real.norm_eq_abs,abs_mul,abs_mul,abs_mul,
    abs_of_nonneg (pow_nonneg hz.le n),abs_of_nonneg ht0,abs_of_nonneg hw]
  calc
    _ ≤ z^n*1*(|A| *z+|B|)*gammaWeight p z := by
      apply mul_le_mul_of_nonneg_right _ hw
      exact mul_le_mul (mul_le_mul_of_nonneg_left ht1 (pow_nonneg hz.le n)) hA
        (abs_nonneg _) (by positivity)
    _ = envelope p n A B z := by dsimp [envelope]; rw [pow_succ]; ring



theorem d_taylor_one (p z q : ℝ) (hz : 0 < z) :
    taylorWithinEval (d0 p z) 1 (Icc 0 1) 0 q =
      gammaWeight p z-p*z*gammaWeight p z*q := by
  rw [show 1=0+1 from rfl,taylorWithinEval_succ,
    (d_jets p z hz).1 0 (by norm_num)]
  simp [d0,d1,taylor_within_zero_eval]
  <;> ring

theorem n_taylor_one (p z q : ℝ) (hz : 0 < z) :
    taylorWithinEval (n0 p z) 1 (Icc 0 1) 0 q = z*gammaWeight p z*q := by
  rw [show 1=0+1 from rfl,taylorWithinEval_succ,
    (n_jets p z hz).1 0 (by norm_num)]
  simp [n0,n1,taylor_within_zero_eval]
  <;> ring

theorem n_taylor_two (p z q : ℝ) (hz : 0 < z) :
    taylorWithinEval (n0 p z) 2 (Icc 0 1) 0 q =
      z*gammaWeight p z*q-(2*p+1)*z^2*gammaWeight p z*q^2/2 := by
  rw [show 2=1+1 from rfl,taylorWithinEval_succ,
    (n_jets p z hz).2.1 0 (by norm_num),n_taylor_one p z q hz]
  norm_num [n2]
  <;> ring

theorem d2_bound (p z q : ℝ) (hp : 0 < p) (hz : 0 < z) (hq : q ∈ Icc (0:ℝ) 1) :
    ‖d2 p z q‖ ≤ envelope p 2 0 (p*(p+1)) z := by
  convert log_power_bound p z q (-p-2) 0 (p*(p+1)) 2 hz hq (by linarith) using 1
  congr 1
  dsimp [d2]
  ring

theorem n2_bound (p z q : ℝ) (hp : 0 < p) (hz : 0 < z) (hq : q ∈ Icc (0:ℝ) 1) :
    ‖n2 p z q‖ ≤ envelope p 2 (p*(p+1)) (-(2*p+1)) z := by
  simpa only [n2,sub_eq_add_neg] using
    log_power_bound p z q (-p-2) (p*(p+1)) (-(2*p+1)) 2 hz hq (by linarith)

theorem n3_bound (p z q : ℝ) (hp : 0 < p) (hz : 0 < z) (hq : q ∈ Icc (0:ℝ) 1) :
    ‖n3 p z q‖ ≤ envelope p 3 (-p*(p+1)*(p+2)) (3*p^2+6*p+2) z := by
  exact log_power_bound p z q (-p-3) (-p*(p+1)*(p+2)) (3*p^2+6*p+2) 3 hz hq (by linarith)

theorem moment_integral (p : ℝ) (n : ℕ) (hp : 0 < p) :
    (∫ z in Ioi (0:ℝ), z^n*gammaWeight p z) = Real.Gamma (p+n) := by
  have h := gammaWeight_moment p n hp
  unfold gammaIntegral at h
  exact (div_left_inj' (Real.Gamma_pos_of_pos hp).ne').mp h

theorem d0_integrable (p q : ℝ) (hp : 0 < p) (hq : 0 ≤ q) :
    IntegrableOn (fun z => d0 p z q) (Ioi 0) := by
  have hi := tilted_integrable p q 0 hp hq
  apply hi.congr
  filter_upwards [] with z
  simp only [d0,tiltedWeight,gammaWeight,pow_zero,one_mul]
  ring

theorem n0_integrable (p q : ℝ) (hp : 0 < p) (hq : q ∈ Icc (0:ℝ) 1) :
    IntegrableOn (fun z => n0 p z q) (Ioi 0) := by
  refine (envelope_integrable p hp 0 1 0).mono' ?_ ?_
  · exact (by unfold n0 gammaWeight; fun_prop : Measurable (fun z => n0 p z q)).aestronglyMeasurable
  · filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    simpa [n0] using log_power_bound p z q (-p) 1 0 0 hz hq (by linarith)

theorem d0_integral (p q : ℝ) : (∫ z in Ioi (0:ℝ), d0 p z q) = normalizer p q := by
  unfold normalizer tiltedMoment
  apply integral_congr_ae
  filter_upwards [] with z
  simp only [d0,tiltedWeight,gammaWeight,pow_zero,one_mul]
  ring

theorem n0_integral (p q : ℝ) : (∫ z in Ioi (0:ℝ), n0 p z q) = logNumerator p q := by
  unfold logNumerator
  apply integral_congr_ae
  filter_upwards [] with z
  dsimp [n0,tiltedWeight,gammaWeight]
  ring


theorem numerator_second_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun q : ℝ => (logNumerator p q-Real.Gamma (p+1)*q)/q^2)
      (𝓝[>] 0) (𝓝 (-(2*p+1)*Real.Gamma (p+2)/2)) := by
  have hc : ∀ᵐ z ∂volume.restrict (Ioi 0),
      ContDiffOn ℝ (1+1) (n0 p z) (Icc 0 1) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    exact (n_contDiff p z hz).of_succ
  have hm : ∀ᶠ q : ℝ in 𝓝[>] (0:ℝ), AEStronglyMeasurable
      (fun z => (n0 p z q-taylorWithinEval (n0 p z) 1 (Icc 0 1) 0 q)/q^(1+1))
      (volume.restrict (Ioi 0)) := by
    apply Filter.Eventually.of_forall
    intro q
    have hmeas : Measurable (fun z => (n0 p z q-z*gammaWeight p z*q)/q^2) := by
      unfold n0 gammaWeight
      fun_prop
    apply hmeas.aestronglyMeasurable.congr
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    rw [n_taylor_one p z q hz]
  have hb : ∀ᵐ z ∂volume.restrict (Ioi 0), ∀ q ∈ Icc (0:ℝ) 1,
      ‖iteratedDerivWithin (1+1) (n0 p z) (Icc 0 1) q‖ ≤
        envelope p 2 (p*(p+1)) (-(2*p+1)) z := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    intro q hq
    rw [(n_jets p z hz).2.1 q hq]
    exact n2_bound p z q hp hz hq
  have ht := GammaTaylor.integral_taylor_next (volume.restrict (Ioi 0)) 1
    (fun q z => n0 p z q) (envelope p 2 (p*(p+1)) (-(2*p+1))) hc hm
    (envelope_integrable p hp 2 _ _) hb
  have he : (∫ z in Ioi (0:ℝ), iteratedDerivWithin 2 (n0 p z) (Icc 0 1) 0 / (2:ℕ).factorial) =
      -(2*p+1)*Real.Gamma (p+2)/2 := by
    have hh : (fun z => iteratedDerivWithin 2 (n0 p z) (Icc 0 1) 0 / (2:ℕ).factorial)
        =ᵐ[volume.restrict (Ioi 0)] (fun z => (-(2*p+1)/2)*(z^2*gammaWeight p z)) := by
      filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
      rw [(n_jets p z hz).2.1 0 (by norm_num)]
      norm_num [n2]
      <;> ring
    rw [integral_congr_ae hh,integral_const_mul,moment_integral p 2 hp]
    norm_num
    <;> ring
  change Tendsto _ _ (𝓝 (∫ z in Ioi (0:ℝ), iteratedDerivWithin 2 (n0 p z) (Icc 0 1) 0 / (2:ℕ).factorial)) at ht
  rw [he] at ht
  apply ht.congr'
  filter_upwards [Ioc_mem_nhdsGT (show (0:ℝ)<1 by norm_num)] with q hq
  have hh : (fun z => (n0 p z q-taylorWithinEval (n0 p z) 1 (Icc 0 1) 0 q)/q^(1+1))
      =ᵐ[volume.restrict (Ioi 0)] (fun z => (n0 p z q-z*gammaWeight p z*q)/q^2) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
    rw [n_taylor_one p z q hz]
  rw [integral_congr_ae hh,integral_div,integral_sub (n0_integrable p q hp ⟨hq.1.le,hq.2⟩)
    (by simpa only [pow_one] using (gamma_moment_integrable p 1 hp).mul_const q),
    n0_integral,integral_mul_const,gamma_first p hp]


noncomputable def secondJet (p : ℝ) : ℝ :=
  limUnder (𝓝[>] (0:ℝ)) (fun q => (scaledLogMean p q-p*q)/q^2)

theorem second_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (fun q : ℝ => (scaledLogMean p q-p*q)/q^2)
      (𝓝[>] 0) (𝓝 (-p*(3*p+1)/2)) := by
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
  have ht := ((numerator_second_limit p hp).sub (hd.const_mul p)).div hz hg
  have he : (-(2*p+1)*Real.Gamma (p+2)/2-p*(-p*Real.Gamma (p+1)))/Real.Gamma p =
      -p*(3*p+1)/2 := by
    have hG2 : Real.Gamma (p+2) = (p+1)*Real.Gamma (p+1) := by
      convert Real.Gamma_add_one (show p+1 ≠ 0 by positivity) using 1 <;> congr 1 <;> ring
    rw [hG2,Real.Gamma_add_one hp.ne']
    field_simp
    <;> ring
  rw [he] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin,hz.eventually (eventually_ne_nhds hg)] with q hq hZ
  change 0 < q at hq
  dsimp only [scaledLogMean,Pi.div_apply,Pi.sub_apply,Pi.mul_apply]
  rw [Real.Gamma_add_one hp.ne']
  field_simp [hq.ne',hZ]
  <;> ring

theorem secondJet_eq (p : ℝ) (hp : 0 < p) : secondJet p = -p*(3*p+1)/2 :=
  (second_limit p hp).limUnder_eq

end Dobson.GammaHigher
#print axioms Dobson.GammaHigher.secondJet_eq
