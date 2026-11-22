import DobsonProofs.Coverage1Loss

namespace Dobson.IidRing
open MeasureTheory ProbabilityTheory Filter
open scoped BigOperators Topology

variable {ι Ω : Type*} [Fintype ι] [DecidableEq ι] [MeasurableSpace Ω]

/-- The actual two outgoing ring edges, evaluated on a whole host trajectory. -/
noncomputable def pathSum (next : Equiv.Perm ι) (s g : ι → ℝ) :
    ℕ → (ℕ → ι → ℝ) → ι → ℝ
  | 0, _, _ => 1
  | n+1, X, i => s i * pathSum next s g n (fun t => X (t+1)) i +
      g i * X 0 i * pathSum next s g n (fun t => X (t+1)) (next i)

noncomputable def observedPath (next : Equiv.Perm ι) (s g : ι → ℝ)
    (N : Ω → ι → ℝ) (T : Ω → Ω) (n : ℕ) (ω : Ω) (i : ι) : ℝ :=
  pathSum next s g n (fun t => N ((T^[t]) ω)) i

lemma observed_zero (next : Equiv.Perm ι) (s g : ι → ℝ)
    (N : Ω → ι → ℝ) (T : Ω → Ω) (ω : Ω) (i : ι) :
    observedPath next s g N T 0 ω i = 1 := rfl

lemma observed_succ (next : Equiv.Perm ι) (s g : ι → ℝ)
    (N : Ω → ι → ℝ) (T : Ω → Ω) (n : ℕ) (ω : Ω) (i : ι) :
    observedPath next s g N T (n+1) ω i =
      s i * observedPath next s g N T n (T ω) i +
      g i * N ω i * observedPath next s g N T n (T ω) (next i) := by
  simp only [observedPath, pathSum, Function.iterate_zero_apply, Function.iterate_succ_apply]

lemma measurable_pathSum (next : Equiv.Perm ι) (s g : ι → ℝ) (n : ℕ) (i : ι) :
    Measurable (fun X : ℕ → ι → ℝ => pathSum next s g n X i) := by
  induction n generalizing i with
  | zero => exact measurable_const
  | succ n ih =>
    have ht : Measurable (fun X : ℕ → ι → ℝ => fun t => X (t+1)) := by fun_prop
    have h0 : Measurable (fun X : ℕ → ι → ℝ => X 0 i) := by fun_prop
    simpa only [pathSum, Pi.mul_def, Pi.add_def, Function.comp_def] using
      ((ih i).comp ht |>.const_mul (s i)).add
        ((h0.const_mul (g i)).mul ((ih (next i)).comp ht))

lemma current_future_indep (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (N : Ω → ι → ℝ)
    (hN : Measurable N)
    (hI : iIndepFun (fun t : ℕ => fun ω => N ((T^[t]) ω)) μ) :
    IndepFun N (fun ω t => N ((T^[t+1]) ω)) μ := by
  have hi := hI.indepFun_set ({0} : Set ℕ) (Set.Ioi 0)
    (by simp [Set.disjoint_left]) (fun t => hN.comp (T.measurable.iterate t))
  have hl : Measurable (fun X : ({0} : Set ℕ) → (ι → ℝ) => X ⟨0, by simp⟩) := by fun_prop
  have hr : Measurable (fun X : Set.Ioi (0 : ℕ) → (ι → ℝ) =>
      fun t : ℕ => X ⟨t+1, by simp⟩) := by fun_prop
  simpa only [Function.comp_def, Function.iterate_zero_apply] using hi.comp hl hr

lemma host_future_path_indep (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (N : Ω → ι → ℝ)
    (hN : Measurable N)
    (hI : iIndepFun (fun t : ℕ => fun ω => N ((T^[t]) ω)) μ)
    (next : Equiv.Perm ι) (s g : ι → ℝ) (n : ℕ) (i j : ι) :
    IndepFun (fun ω => N ω i) (fun ω => observedPath next s g N T n (T ω) j) μ := by
  have hi := (current_future_indep μ T N hN hI).comp
    (measurable_pi_apply i) (measurable_pathSum next s g n j)
  simpa only [Function.comp_def, observedPath, Function.iterate_succ_apply] using hi

lemma observed_integrable (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → ι → ℝ)
    (hN : Measurable N) (hNm : ∀ i, Integrable (fun ω => N ω i) μ)
    (hI : iIndepFun (fun t : ℕ => fun ω => N ((T^[t]) ω)) μ)
    (next : Equiv.Perm ι) (s g : ι → ℝ) (n : ℕ) (i : ι) :
    Integrable (fun ω => observedPath next s g N T n ω i) μ := by
  induction n generalizing i with
  | zero => exact integrable_const _
  | succ n ih =>
    have hf (j : ι) : Integrable (fun ω => observedPath next s g N T n (T ω) j) μ :=
      hT.integrable_comp_of_integrable (ih j)
    have hm := (host_future_path_indep μ T N hN hI next s g n i (next i)).integrable_mul
      (hNm i) (hf (next i))
    simpa only [observed_succ, Pi.mul_def, Pi.add_def, mul_assoc] using
      ((hf i).const_mul (s i)).add (hm.const_mul (g i))

/-- Independence gives the expectation recurrence; it is not supplied as a premise. -/
lemma observed_integral_succ (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → ι → ℝ)
    (hN : Measurable N) (hNm : ∀ i, Integrable (fun ω => N ω i) μ)
    (hI : iIndepFun (fun t : ℕ => fun ω => N ((T^[t]) ω)) μ)
    (next : Equiv.Perm ι) (s g : ι → ℝ) (n : ℕ) (i : ι) :
    (∫ ω, observedPath next s g N T (n+1) ω i ∂μ) =
      s i * (∫ ω, observedPath next s g N T n ω i ∂μ) +
      (g i * ∫ ω, N ω i ∂μ) * ∫ ω, observedPath next s g N T n ω (next i) ∂μ := by
  have hf (j : ι) : Integrable (fun ω => observedPath next s g N T n (T ω) j) μ :=
    hT.integrable_comp_of_integrable (observed_integrable μ T hT N hN hNm hI next s g n j)
  have hind := host_future_path_indep μ T N hN hI next s g n i (next i)
  have hm := hind.integrable_mul (hNm i) (hf (next i))
  have hm' : Integrable (fun ω => N ω i * observedPath next s g N T n (T ω) (next i)) μ := by
    simpa only [Pi.mul_def] using hm
  simp only [observed_succ, mul_assoc]
  rw [integral_add ((hf i).const_mul (s i)) (hm'.const_mul (g i)), integral_const_mul, integral_const_mul]
  have he := hind.integral_mul_eq_mul_integral (hNm i).aestronglyMeasurable
    (hf (next i)).aestronglyMeasurable
  change (∫ ω, N ω i * observedPath next s g N T n (T ω) (next i) ∂μ) = _ at he
  rw [he]
  have hs (j : ι) := hT.integral_comp' (fun ω => observedPath next s g N T n ω j)
  rw [hs i, hs (next i)]

lemma physical_step_action (next : Equiv.Perm ι) (s g : ι → ℝ)
    (N : Ω → ι → ℝ) (birth : ι) (ω : Ω) (v : ι → ℝ) (i : ι) :
    (Coverage4ThermalFixed.physicalRingStep next (fun ω i => g i*N ω i)
      (fun _ => s) 1 birth ω).mulVec v i = s i*v i+g i*N ω i*v (next i) := by
  classical
  simp [Coverage4ThermalFixed.physicalRingStep, Matrix.mulVec, dotProduct,
    Finset.sum_add_distrib, add_mul, ite_mul]

lemma observed_eq_physical (next : Equiv.Perm ι) (s g : ι → ℝ)
    (N : Ω → ι → ℝ) (T : Ω → Ω) (birth : ι) (n : ℕ) (ω : Ω) :
    observedPath next s g N T n ω =
      (Coverage4ThermalFixed.physicalRingProduct next (fun ω i => g i*N ω i)
        (fun _ => s) 1 birth T n ω).mulVec (fun _ => 1) := by
  classical
  induction n generalizing ω with
  | zero => ext i; simp [observed_zero, Coverage4ThermalFixed.physicalRingProduct]
  | succ n ih =>
    ext i
    change _ = (Coverage4ThermalFixed.physicalRingStep next (fun ω i => g i*N ω i)
      (fun _ => s) 1 birth ω * Coverage4ThermalFixed.physicalRingProduct next
      (fun ω i => g i*N ω i) (fun _ => s) 1 birth T n (T ω)).mulVec (fun _ => 1) i
    rw [← Matrix.mulVec_mulVec, physical_step_action, observed_succ, ih]

section Nonempty
variable [Nonempty ι]

lemma reference_eq_observed (s g : ι → ℝ) (next : Equiv.Perm ι)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) (hg : ∀ i, 0 < g i)
    (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (N : Ω → ι → ℝ) (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < N ω i) (n : ℕ) :
    ∀ᵐ ω ∂μ, ReferencePaths.partition
      (ReferenceRing.kernel (RingGrowth.times s) (RingGrowth.times_ge_one s hs0 hs1) next)
      T (RingGrowth.score s g N) n ω = observedPath next s g N T n ω := by
  let K := ReferenceRing.kernel (RingGrowth.times s) (RingGrowth.times_ge_one s hs0 hs1) next
  let H := RingGrowth.score s g N
  let birth : ι := Classical.arbitrary ι
  have hstep : ∀ᵐ ω ∂μ, ReferencePaths.stepMatrix K H ω =
      Coverage4ThermalFixed.physicalRingStep next (fun ω i => g i*N ω i) (fun _ => s) 1 birth ω := by
    filter_upwards [hh] with ω hω
    have he := Coverage4ThermalFixed.physicalRing_step_identification
      (RingGrowth.times s) (fun _ => 1) (RingGrowth.times_ge_one s hs0 hs1) next
      (fun ω i => g i*N ω i) (fun _ => s) 1 birth (by intro; norm_num)
      (by norm_num) ω (fun i => mul_pos (hg i) (hω i)) (RingGrowth.stay_domain s hs0)
    simpa only [K, H, RingGrowth.score, Real.log_one, zero_mul, sub_zero] using he
  induction n with
  | zero => exact Eventually.of_forall (fun ω => by ext i; simp [ReferencePaths.partition, observed_zero])
  | succ n ih =>
    filter_upwards [hstep, hT.quasiMeasurePreserving.ae ih] with ω hω hn
    ext i
    change ReferencePaths.partition K T H (n+1) ω i = _
    change (∑ e, K.prob i e*Real.exp (H ω e)*ReferencePaths.partition K T H n (T ω) (K.dest e)) = _
    rw [← ReferencePaths.stepMatrix_mulVec, hω, physical_step_action, hn, observed_succ]

lemma weighted_expectation_bound (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → ι → ℝ)
    (hN : Measurable N) (hNm : ∀ i, Integrable (fun ω => N ω i) μ)
    (hI : iIndepFun (fun t : ℕ => fun ω => N ((T^[t]) ω)) μ)
    (next : Equiv.Perm ι) (s g v : ι → ℝ) (hs : ∀ i, 0 ≤ s i)
    (hgm : ∀ i, 0 ≤ g i * ∫ ω, N ω i ∂μ) (hv : ∀ i, 0 < v i)
    (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hrow : ∀ i, s i*v i+(g i*∫ ω, N ω i ∂μ)*v (next i) ≤ ρ*v i)
    (n : ℕ) (i : ι) :
    (∫ ω, observedPath next s g N T n ω i ∂μ) ≤
      (∑ j, (v j)⁻¹)*ρ^n*v i := by
  have hC : 0 ≤ ∑ j, (v j)⁻¹ := Finset.sum_nonneg (fun j _ => (inv_pos.mpr (hv j)).le)
  induction n generalizing i with
  | zero =>
    simp only [observed_zero, integral_const, pow_zero, mul_one]
    simp only [probReal_univ, smul_eq_mul, one_mul]
    have hi := mul_le_mul_of_nonneg_right
      (Finset.single_le_sum (fun j _ => (inv_pos.mpr (hv j)).le) (Finset.mem_univ i)) (hv i).le
    simpa only [inv_mul_cancel₀ (hv i).ne'] using hi
  | succ n ih =>
    rw [observed_integral_succ μ T hT N hN hNm hI]
    calc
      _ ≤ s i*((∑ j, (v j)⁻¹)*ρ^n*v i) +
          (g i*∫ ω, N ω i ∂μ)*((∑ j, (v j)⁻¹)*ρ^n*v (next i)) :=
        add_le_add (mul_le_mul_of_nonneg_left (ih i) (hs i))
          (mul_le_mul_of_nonneg_left (ih (next i)) (hgm i))
      _ = ((∑ j, (v j)⁻¹)*ρ^n)*(s i*v i+(g i*∫ ω, N ω i ∂μ)*v (next i)) := by ring
      _ ≤ ((∑ j, (v j)⁻¹)*ρ^n)*(ρ*v i) :=
        mul_le_mul_of_nonneg_left (hrow i) (mul_nonneg hC (pow_nonneg hρ n))
      _ = _ := by rw [pow_succ]; ring

lemma integral_log_le_bound (μ : Measure Ω) [IsProbabilityMeasure μ] (X : Ω → ℝ)
    (hX : Integrable X μ) (hlog : Integrable (fun ω => Real.log (X ω)) μ)
    (hpos : ∀ᵐ ω ∂μ, 0 < X ω) (B : ℝ) (hB : 0 < B) (hb : (∫ ω, X ω ∂μ) ≤ B) :
    (∫ ω, Real.log (X ω) ∂μ) ≤ Real.log B := by
  have hdom : ∀ᵐ ω ∂μ, Real.log (X ω) ≤ Real.log B + X ω / B - 1 := by
    filter_upwards [hpos] with ω hω
    have hh := Real.log_le_sub_one_of_pos (div_pos hω hB)
    rw [Real.log_div hω.ne' hB.ne'] at hh
    linarith
  have hi : Integrable (fun ω => Real.log B + X ω / B - 1) μ :=
    ((integrable_const _).add (hX.div_const B)).sub (integrable_const _)
  have hh := integral_mono_ae hlog hi hdom
  have hii : Integrable (fun ω => Real.log B + X ω / B) μ :=
    (integrable_const _).add (hX.div_const B)
  rw [integral_sub hii (integrable_const (1 : ℝ)),
    integral_add (integrable_const (Real.log B)) (hX.div_const B), integral_div] at hh
  simp at hh
  have hd := (div_le_one hB).mpr hb
  linarith

/-- An explicitly proved subeigenvector of the mean ring bounds actual IID growth. -/
theorem annealed_growth_le (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (N : Ω → ι → ℝ)
    (hN : Measurable N) (hNm : ∀ i, Integrable (fun ω => N ω i) μ)
    (hNlog : ∀ i, Integrable (fun ω => Real.log (N ω i)) μ)
    (hNpos : ∀ᵐ ω ∂μ, ∀ i, 0 < N ω i)
    (hI : iIndepFun (fun t : ℕ => fun ω => N ((T^[t]) ω)) μ)
    (next : Equiv.Perm ι) (birth : ι) (s g v : ι → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) (hg : ∀ i, 0 < g i)
    (hv : ∀ i, 0 < v i) (ρ : ℝ) (hρ : 0 < ρ)
    (hrow : ∀ i, s i*v i+(g i*∫ ω, N ω i ∂μ)*v (next i) ≤ ρ*v i)
    (r : ℝ) (hFK : Tendsto (Coverage4ThermalFixed.physicalRingGrowth (RingGrowth.times s) next
      (fun ω i => g i*N ω i) (fun _ => s) 1 birth μ T) atTop (nhds r)) :
    r ≤ Real.log ρ := by
  let K := ReferenceRing.kernel (RingGrowth.times s) (RingGrowth.times_ge_one s hs0 hs1) next
  let X := fun n ω => ∑ i, K.initial i * observedPath next s g N T n ω i
  let C := (∑ i, (v i)⁻¹) * ∑ i, K.initial i*v i
  have hC : 0 < C := mul_pos
    (Finset.sum_pos (fun i _ => inv_pos.mpr (hv i)) Finset.univ_nonempty)
    (Finset.sum_pos (fun i _ => mul_pos (K.initial_pos i) (hv i)) Finset.univ_nonempty)
  have hgm (i : ι) : 0 ≤ g i * ∫ ω, N ω i ∂μ :=
    mul_nonneg (hg i).le (integral_nonneg_of_ae (hNpos.mono (fun _ h => (h i).le)))
  have hi (n : ℕ) : Integrable (X n) μ := integrable_finsetSum _
    (fun i _ => (observed_integrable μ T hT N hN hNm hI next s g n i).const_mul (K.initial i))
  have he (n : ℕ) : (fun ω => ReferencePaths.totalPartition K T (RingGrowth.score s g N) n ω) =ᵐ[μ] X n := by
    filter_upwards [reference_eq_observed s g next hs0 hs1 hg μ T hT N hNpos n] with ω hω
    simp only [ReferencePaths.totalPartition, hω, X, K]
  have hilog (n : ℕ) : Integrable (fun ω => Real.log (X n ω)) μ :=
    (ReferencePaths.total_log_integrable K μ T hT (RingGrowth.score s g N)
      (RingGrowth.score_integrable s g N μ hs0 hg hNpos hNlog) n).congr
      ((he n).mono (fun _ hh => congrArg Real.log hh))
  have hpos (n : ℕ) : ∀ᵐ ω ∂μ, 0 < X n ω := by
    filter_upwards [he n] with ω hω
    rw [← hω]
    exact Finset.sum_pos (fun i _ => mul_pos (K.initial_pos i)
      (ReferencePaths.partition_pos K T _ n ω i)) Finset.univ_nonempty
  have hib (n : ℕ) : (∫ ω, X n ω ∂μ) ≤ C*ρ^n := by
    dsimp [X]
    rw [integral_finsetSum _ (fun i _ =>
      (observed_integrable μ T hT N hN hNm hI next s g n i).const_mul (K.initial i))]
    simp only [integral_const_mul]
    calc
      _ ≤ ∑ i, K.initial i*((∑ j, (v j)⁻¹)*ρ^n*v i) :=
        Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left
          (weighted_expectation_bound μ T hT N hN hNm hI next s g v hs0 hgm hv ρ hρ.le hrow n i)
          (K.initial_nonneg i))
      _ = _ := by
        dsimp [C]
        simp only [Finset.mul_sum, Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro i _
        apply Finset.sum_congr rfl
        intro j _
        ring
  have hf (n : ℕ) : ReferencePaths.normalizedGrowth K μ T (RingGrowth.score s g N) n ≤
      Real.log C / (n+1 : ℕ) + Real.log ρ := by
    have hb := integral_log_le_bound μ (X (n+1)) (hi _) (hilog _) (hpos _)
      (C*ρ^(n+1)) (mul_pos hC (pow_pos hρ _)) (hib _)
    rw [Real.log_mul hC.ne' (pow_pos hρ _).ne', Real.log_pow] at hb
    have heint : (∫ ω, Real.log (ReferencePaths.totalPartition K T (RingGrowth.score s g N) (n+1) ω) ∂μ) =
        ∫ ω, Real.log (X (n+1) ω) ∂μ := integral_congr_ae ((he _).mono (fun _ hh => congrArg Real.log hh))
    unfold ReferencePaths.normalizedGrowth
    rw [heint]
    have hn : (0 : ℝ) < (n+1 : ℕ) := by positivity
    apply (div_le_iff₀ hn).mpr
    convert hb using 1 <;> field_simp <;> ring
  have hr : Tendsto (ReferencePaths.normalizedGrowth K μ T (RingGrowth.score s g N)) atTop (nhds r) := by
    rw [RingGrowth.normalized_eq_physical s g N hs0 hs1 hg next birth μ T hT hNpos]
    exact hFK
  have hc : Tendsto (fun n : ℕ => Real.log C/(n+1 : ℕ)+Real.log ρ) atTop (nhds (Real.log ρ)) := by
    have hinv : Tendsto (fun n : ℕ => ((n+1 : ℕ) : ℝ)⁻¹) atTop (nhds 0) := by
      simpa [one_div, Function.comp_def] using
        (tendsto_one_div_atTop_nhds_zero_nat (𝕜 := ℝ)).comp (tendsto_add_atTop_nat 1)
    simpa [div_eq_mul_inv] using (hinv.const_mul (Real.log C)).add_const (Real.log ρ)
  exact le_of_tendsto_of_tendsto' hr hc hf

end Nonempty
lemma subcritical_mean_vector (n : ℕ) (hn : 1 ≤ n) (s a : Fin n → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) (ha : ∀ i, 0 < a i)
    (hP : Coverage1Loss.circuitR0 n s a < 1) :
    ∃ (v : Fin n → ℝ) (ρ : ℝ), (∀ i, 0 < v i) ∧ 0 < ρ ∧ ρ < 1 ∧
      ∀ i, s i*v i+a i*v (finRotate n i) ≤ ρ*v i := by
  letI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  let P := Coverage1Loss.circuitR0 n s a
  have hP0 : 0 < P := Finset.prod_pos (fun i _ => div_pos (ha i) (sub_pos.mpr (hs1 i)))
  let u := Real.exp (Real.log P / n)
  have hu : 0 < u := Real.exp_pos _
  have hu1 : u < 1 := Real.exp_lt_one_iff.mpr (div_neg_of_neg_of_pos
    (Real.log_neg hP0 hP) hnR)
  let b := fun i => (1-s i)*u
  have hb (i : Fin n) : 0 < b i := mul_pos (sub_pos.mpr (hs1 i)) hu
  have hprodb : (∏ i, b i) = ∏ i, a i := by
    obtain ⟨_, huProd, _⟩ := Coverage1Loss.prod_uniformTransfer_eq n hn (Real.log P)
    have huProd' : (∏ _i : Fin n, u) = P := by
      simpa only [Coverage1Loss.uniformTransfer, u, Real.exp_log hP0] using huProd
    dsimp [b]
    rw [Finset.prod_mul_distrib, huProd']
    dsimp [P, Coverage1Loss.circuitR0]
    rw [Finset.prod_div_distrib]
    have hz : (∏ i, (1-s i)) ≠ 0 :=
      (Finset.prod_pos (fun i _ => sub_pos.mpr (hs1 i))).ne'
    field_simp
  obtain ⟨v, hv, hD⟩ := Coverage1Loss.exists_pos_diag_conj_of_prod_eq n hn a b ha hb hprodb.symm
  let q := fun i => s i+(1-s i)*u
  have hq (i : Fin n) : 0 < q i :=
    add_pos_of_nonneg_of_pos (hs0 i) (mul_pos (sub_pos.mpr (hs1 i)) hu)
  have hq1 (i : Fin n) : q i < 1 := by dsimp [q]; nlinarith [hs1 i]
  let ρ := Finset.univ.sup' Finset.univ_nonempty q
  have hρ0 : 0 < ρ := lt_of_lt_of_le (hq ⟨0, by omega⟩)
    (Finset.le_sup' q (Finset.mem_univ _))
  have hρ1 : ρ < 1 := (Finset.sup'_lt_iff Finset.univ_nonempty).mpr (fun i _ => hq1 i)
  refine ⟨v, ρ, hv, hρ0, hρ1, ?_⟩
  intro i
  have ht := congrArg Matrix.transpose (hD s (fun _ => 1))
  simp only [Matrix.transpose_mul, Matrix.diagonal_transpose] at ht
  have hA := Coverage1Loss.physical_step_transpose n hn s a (fun _ => 1)
  have hB := Coverage1Loss.physical_step_transpose n hn s b (fun _ => 1)
  rw [← hA, ← hB, Matrix.transpose_transpose, Matrix.transpose_transpose] at ht
  have hh := congrArg (fun A : Matrix (Fin n) (Fin n) ℝ => A.mulVec (fun _ => 1) i) ht
  simp only [← Matrix.mulVec_mulVec, Matrix.mulVec_diagonal, mul_one] at hh
  have hv1 : (Matrix.diagonal v).mulVec (fun _ => 1) = v := by ext j; simp [Matrix.mulVec_diagonal]
  rw [hv1] at hh
  have hactA := physical_step_action (finRotate n) s a (fun (_ : Unit) _ => (1 : ℝ))
    ⟨0, by omega⟩ () v i
  have hactB := physical_step_action (finRotate n) s b (fun (_ : Unit) _ => (1 : ℝ))
    ⟨0, by omega⟩ () (fun _ => 1) i
  simp only [mul_one] at hh hactA hactB
  rw [hactA, hactB] at hh
  calc
    _ = v i*q i := by simpa only [q, b] using hh
    _ ≤ v i*ρ := mul_le_mul_of_nonneg_left (Finset.le_sup' q (Finset.mem_univ i)) (hv i).le
    _ = _ := mul_comm _ _

end Dobson.IidRing
#print axioms Dobson.IidRing.observed_integral_succ
#print axioms Dobson.IidRing.annealed_growth_le
#print axioms Dobson.IidRing.subcritical_mean_vector
