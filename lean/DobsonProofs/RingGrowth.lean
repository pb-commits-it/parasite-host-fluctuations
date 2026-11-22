import DobsonProofs.ReferencePaths
import DobsonProofs.ReferenceRing
import DobsonProofs.Coverage2Bounds
import DobsonProofs.Coverage4ThermalFixed

namespace Dobson.RingGrowth
open MeasureTheory Filter
open scoped BigOperators Topology

variable {ι E Ω : Type*} [Fintype ι] [Fintype E] [MeasurableSpace Ω]

noncomputable def budget (T : Ω → Ω) (B : Ω → ℝ) : ℕ → Ω → ℝ
  | 0, _ => 0
  | n+1, ω => B ω + budget T B n (T ω)

lemma budget_integrable (μ : Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (B : Ω → ℝ) (hB : Integrable B μ) (n : ℕ) :
    Integrable (budget T B n) μ := by
  induction n with
  | zero => exact integrable_zero _ _ _
  | succ n ih => exact hB.add (hT.integrable_comp_of_integrable ih)

lemma integral_budget (μ : Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (B : Ω → ℝ) (hB : Integrable B μ) (n : ℕ) :
    (∫ ω, budget T B n ω ∂μ) = n * ∫ ω, B ω ∂μ := by
  induction n with
  | zero => simp [budget]
  | succ n ih =>
    simp only [budget]
    change (∫ ω, B ω + (budget T B n ∘ T) ω ∂μ) = _
    rw [integral_add hB (hT.integrable_comp_of_integrable (budget_integrable μ T hT B hB n))]
    simp only [Function.comp_apply]
    rw [hT.integral_comp' (budget T B n), ih]
    push_cast
    ring

/-- A proved one-step upper bound controls the actual finite partition. -/
lemma partition_le_budget (K : ReferencePaths.Kernel ι E) (μ : Measure Ω)
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (H : Ω → E → ℝ) (B : Ω → ℝ)
    (hrow : ∀ᵐ ω ∂μ, ∀ i, (∑ e, K.prob i e * Real.exp (H ω e)) ≤ Real.exp (B ω))
    (n : ℕ) : ∀ᵐ ω ∂μ, ∀ i, ReferencePaths.partition K T H n ω i ≤ Real.exp (budget T B n ω) := by
  induction n with
  | zero => exact Eventually.of_forall (fun _ _ => by simp [ReferencePaths.partition, budget])
  | succ n ih =>
    filter_upwards [hrow, hT.quasiMeasurePreserving.ae ih] with ω hω hn
    intro i
    calc
      ReferencePaths.partition K T H (n+1) ω i ≤
          ∑ e, K.prob i e * Real.exp (H ω e) * Real.exp (budget T B n (T ω)) := by
        exact Finset.sum_le_sum (fun e _ => mul_le_mul_of_nonneg_left (hn (K.dest e))
          (mul_nonneg (K.nonneg i e) (Real.exp_pos _).le))
      _ = (∑ e, K.prob i e * Real.exp (H ω e)) * Real.exp (budget T B n (T ω)) :=
        (Finset.sum_mul ..).symm
      _ ≤ Real.exp (B ω) * Real.exp (budget T B n (T ω)) :=
        mul_le_mul_of_nonneg_right (hω i) (Real.exp_pos _).le
      _ = Real.exp (budget T B (n+1) ω) := by rw [← Real.exp_add]; rfl

lemma normalized_growth_upper (K : ReferencePaths.Kernel ι E) (μ : Measure Ω)
    [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (H : Ω → E → ℝ) (hH : ∀ e, Integrable (fun ω => H ω e) μ)
    (B : Ω → ℝ) (hB : Integrable B μ)
    (hrow : ∀ᵐ ω ∂μ, ∀ i, (∑ e, K.prob i e * Real.exp (H ω e)) ≤ Real.exp (B ω))
    (n : ℕ) : ReferencePaths.normalizedGrowth K μ T H n ≤ ∫ ω, B ω ∂μ := by
  have htotal : ∀ᵐ ω ∂μ, Real.log (ReferencePaths.totalPartition K T H (n+1) ω) ≤
      budget T B (n+1) ω := by
    filter_upwards [partition_le_budget K μ T hT H B hrow (n+1)] with ω hω
    have hpos : 0 < ReferencePaths.totalPartition K T H (n+1) ω := by
      have hex : ∃ i, 0 < K.initial i := by
        by_contra h
        push Not at h
        have hh := Finset.sum_nonpos (fun i (_ : i ∈ Finset.univ) => h i)
        rw [K.initial_total] at hh
        norm_num at hh
      obtain ⟨i, hi⟩ := hex
      exact Finset.sum_pos' (fun j _ => mul_nonneg (K.initial_nonneg j)
        (ReferencePaths.partition_pos K T H _ _ _).le)
        ⟨i, Finset.mem_univ _, mul_pos hi (ReferencePaths.partition_pos K T H _ _ _)⟩
    have hbound : ReferencePaths.totalPartition K T H (n+1) ω ≤ Real.exp (budget T B (n+1) ω) := by
      calc
        _ ≤ ∑ i, K.initial i * Real.exp (budget T B (n+1) ω) :=
          Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hω i) (K.initial_nonneg i))
        _ = _ := by rw [← Finset.sum_mul, K.initial_total, one_mul]
    simpa using Real.log_le_log hpos hbound
  have hi := integral_mono_ae (ReferencePaths.total_log_integrable K μ T hT H hH (n+1))
    (budget_integrable μ T hT B hB (n+1)) htotal
  rw [integral_budget μ T hT B hB (n+1)] at hi
  exact (div_le_iff₀ (by positivity : (0 : ℝ) < (n+1 : ℕ))).mpr (by simpa only [mul_comm] using hi)

section PhysicalRing
variable [Nonempty ι] [DecidableEq ι]

noncomputable def times (s : ι → ℝ) (i : ι) : ℝ := 1 / (1 - s i)

lemma times_ge_one (s : ι → ℝ) (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) :
    ∀ i, 1 ≤ times s i := by
  intro i
  exact (le_div_iff₀ (sub_pos.mpr (hs1 i))).mpr (by have := hs0 i; linarith)

lemma stay_reconstruct (s : ι → ℝ) (i : ι) : 1 - 1 / times s i = s i := by
  simp [times]

lemma stay_domain (s : ι → ℝ) (hs0 : ∀ i, 0 ≤ s i) (i : ι) :
    if times s i = 1 then s i = 0 else 0 < s i := by
  split_ifs with hi
  · have := stay_reconstruct s i
    rw [hi] at this
    norm_num at this
    exact this.symm
  · have hn : s i ≠ 0 := by
      intro he
      apply hi
      simp [times, he]
    exact lt_of_le_of_ne (hs0 i) (Ne.symm hn)

noncomputable def score (s g : ι → ℝ) (host : Ω → ι → ℝ) : Ω → (ι × Bool) → ℝ :=
  Coverage4ThermalFixed.physicalRingScore (times s) (fun _ => 1)
    (fun ω i => g i * host ω i) (fun _ => s)

lemma score_stay (s g : ι → ℝ) (host : Ω → ι → ℝ) (hs0 : ∀ i, 0 ≤ s i)
    (ω : Ω) (i : ι) : score s g host ω (i, false) = 0 := by
  unfold score Coverage4ThermalFixed.physicalRingScore
  simp only [Bool.false_eq_true, ↓reduceIte, stay_reconstruct]
  split_ifs with hi
  · rfl
  · have hh : s i ≠ 0 := by
      have h := stay_domain s hs0 i
      simp only [hi, ↓reduceIte] at h
      exact h.ne'
    rw [div_self hh, Real.log_one]

lemma score_transfer (s g : ι → ℝ) (host : Ω → ι → ℝ)
    (hg : ∀ i, 0 < g i) (ω : Ω) (hh : ∀ i, 0 < host ω i) (i : ι) :
    score s g host ω (i, true) = Real.log (g i) + Real.log (host ω i) + Real.log (times s i) := by
  simp [score, Coverage4ThermalFixed.physicalRingScore,
    Real.log_mul (hg i).ne' (hh i).ne']

lemma score_integrable (s g : ι → ℝ) (host : Ω → ι → ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (hs0 : ∀ i, 0 ≤ s i)
    (hg : ∀ i, 0 < g i) (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i)
    (hlog : ∀ i, Integrable (fun ω => Real.log (host ω i)) μ) :
    ∀ e, Integrable (fun ω => score s g host ω e) μ := by
  rintro ⟨i, b⟩
  cases b
  · simp only [score_stay s g host hs0]
    exact integrable_const _
  · apply ((integrable_const (Real.log (g i))).add (hlog i) |>.add
      (integrable_const (Real.log (times s i)))).congr
    filter_upwards [hh] with ω hω
    exact (score_transfer s g host hg ω hω i).symm

lemma score_row_sum (s g : ι → ℝ) (host : Ω → ι → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) (hg : ∀ i, 0 < g i)
    (next : Equiv.Perm ι) (birth : ι) (ω : Ω) (hh : ∀ i, 0 < host ω i) (i : ι) :
    (∑ e, (ReferenceRing.kernel (times s) (times_ge_one s hs0 hs1) next).prob i e *
      Real.exp (score s g host ω e)) = s i + g i * host ω i := by
  let K := ReferenceRing.kernel (times s) (times_ge_one s hs0 hs1) next
  have he := Coverage4ThermalFixed.physicalRing_step_identification
    (times s) (fun _ => 1) (times_ge_one s hs0 hs1) next
    (fun ω i => g i * host ω i) (fun _ => s) 1 birth (by intro; norm_num)
    (by norm_num) ω (fun i => mul_pos (hg i) (hh i)) (stay_domain s hs0)
  simp only [Real.log_one, zero_mul, sub_zero] at he
  have hr := congrArg (fun A : Matrix ι ι ℝ => A.mulVec (fun _ => 1) i) he
  rw [ReferencePaths.stepMatrix_mulVec] at hr
  simp only [mul_one] at hr
  simpa [Coverage4ThermalFixed.physicalRingStep, Matrix.mulVec, dotProduct,
    Finset.sum_add_distrib, score] using hr

lemma normalized_eq_physical (s g : ι → ℝ) (host : Ω → ι → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1) (hg : ∀ i, 0 < g i)
    (next : Equiv.Perm ι) (birth : ι) (μ : Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i) :
    ReferencePaths.normalizedGrowth (ReferenceRing.kernel (times s) (times_ge_one s hs0 hs1) next)
      μ T (score s g host) =
    Coverage4ThermalFixed.physicalRingGrowth (times s) next (fun ω i => g i * host ω i)
      (fun _ => s) 1 birth μ T := by
  have hg' : ∀ᵐ ω ∂μ, ∀ i, 0 < g i * host ω i :=
    hh.mono (fun _ h i => mul_pos (hg i) (h i))
  have hs' : ∀ᵐ _ω ∂μ, ∀ i, if times s i = 1 then s i = 0 else 0 < s i :=
    Eventually.of_forall (fun _ => stay_domain s hs0)
  simpa only [Real.log_one, zero_mul, sub_zero, score] using
    Coverage4ThermalFixed.physicalRing_growth_identification
      (times s) (fun _ => 1) (times_ge_one s hs0 hs1) next
      (fun ω i => g i * host ω i) (fun _ => s) 1 birth (by intro; norm_num)
      (by norm_num) μ T hT hg' hs'

lemma mean_score_formula (s g : ι → ℝ) (host : Ω → ι → ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (hg : ∀ i, 0 < g i) (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i)
    (hlog : ∀ i, Integrable (fun ω => Real.log (host ω i)) μ)
    (next : Equiv.Perm ι) :
    ReferencePaths.meanScore (ReferenceRing.kernel (times s) (times_ge_one s hs0 hs1) next)
      μ (score s g host) =
      (∑ i, (Real.log (g i) + (∫ ω, Real.log (host ω i) ∂μ) + Real.log (times s i))) /
        ReferenceRing.totalTime (times s) := by
  rw [ReferenceRing.meanScore_formula]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  simp only [score_stay s g host hs0, integral_zero, mul_zero, add_zero]
  have he : (fun ω => score s g host ω (i, true)) =ᵐ[μ]
      (fun ω => Real.log (g i) + Real.log (host ω i) + Real.log (times s i)) :=
    hh.mono (fun ω h => score_transfer s g host hg ω h i)
  rw [integral_congr_ae he]
  have hi : Integrable (fun ω => Real.log (g i) + Real.log (host ω i)) μ :=
    (integrable_const _).add (hlog i)
  rw [integral_add hi (integrable_const (Real.log (times s i))),
    integral_add (integrable_const (Real.log (g i))) (hlog i)]
  simp

end PhysicalRing

section PhysicalLimits
variable [Nonempty ι] [DecidableEq ι]

lemma physical_growth_lower (s g : ι → ℝ) (host : Ω → ι → ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (hg : ∀ i, 0 < g i) (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i)
    (hlog : ∀ i, Integrable (fun ω => Real.log (host ω i)) μ)
    (next : Equiv.Perm ι) (birth : ι) (r : ℝ)
    (hFK : Tendsto (Coverage4ThermalFixed.physicalRingGrowth (times s) next
      (fun ω i => g i * host ω i) (fun _ => s) 1 birth μ T) atTop (nhds r)) :
    (∑ i, (Real.log (g i) + (∫ ω, Real.log (host ω i) ∂μ) + Real.log (times s i))) /
      ReferenceRing.totalTime (times s) ≤ r := by
  have hf : Tendsto
      (ReferencePaths.normalizedGrowth (ReferenceRing.kernel (times s) (times_ge_one s hs0 hs1) next)
        μ T (score s g host)) atTop (nhds (r - 0)) := by
    rw [normalized_eq_physical s g host hs0 hs1 hg next birth μ T hT hh]
    simpa using hFK
  have he := Coverage2Bounds.path_limit_bound
    (ReferenceRing.kernel (times s) (times_ge_one s hs0 hs1) next) μ T hT
    (score s g host) (score_integrable s g host μ hs0 hg hh hlog) r 0 hf
  rw [mean_score_formula s g host μ hs0 hs1 hg hh hlog next] at he
  simpa using he

/-- The actual growth diverges to +infinity as all transfers increase uniformly. -/
theorem growth_tendsto_atTop (s : ι → ℝ) (host : Ω → ι → ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i)
    (hlog : ∀ i, Integrable (fun ω => Real.log (host ω i)) μ)
    (next : Equiv.Perm ι) (birth : ι) (r : ℝ → ℝ)
    (hFK : ∀ y, Tendsto (Coverage4ThermalFixed.physicalRingGrowth (times s) next
      (fun ω i => Real.exp (y / Fintype.card ι) * host ω i) (fun _ => s) 1 birth μ T)
      atTop (nhds (r y))) : Tendsto r atTop atTop := by
  let C := ∑ i, ((∫ ω, Real.log (host ω i) ∂μ) + Real.log (times s i))
  let W := ReferenceRing.totalTime (times s)
  have hW : 0 < W := ReferenceRing.totalTime_pos (times s) (times_ge_one s hs0 hs1)
  have hN : (Fintype.card ι : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have hlower (y : ℝ) : (y + C) / W ≤ r y := by
    have he := physical_growth_lower s (fun _ => Real.exp (y / Fintype.card ι)) host μ T hT
      hs0 hs1 (fun _ => Real.exp_pos _) hh hlog next birth (r y) (hFK y)
    have hsum : (∑ i : ι, (Real.log (Real.exp (y / Fintype.card ι)) +
        (∫ ω, Real.log (host ω i) ∂μ) + Real.log (times s i))) = y + C := by
      simp_rw [Real.log_exp, add_assoc]
      rw [Finset.sum_add_distrib]
      simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      dsimp [C]
      field_simp
    rw [hsum] at he
    exact he
  apply tendsto_atTop.2
  intro b
  apply eventually_atTop.2
  refine ⟨W * b - C, ?_⟩
  intro y hy
  apply le_trans _ (hlower y)
  exact (le_div_iff₀ hW).mpr (by linarith)

/-- The tangent-line upper estimate on finite products needs no independence
between times and no exponential moments of log multipliers. -/
lemma physical_growth_upper_tangent (s : ι → ℝ) (host : Ω → ι → ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i)
    (hhost : ∀ i, Integrable (fun ω => host ω i) μ)
    (hlog : ∀ i, Integrable (fun ω => Real.log (host ω i)) μ)
    (next : Equiv.Perm ι) (birth : ι) (a S ρ r : ℝ)
    (ha : 0 < a) (hS : ∀ i, s i ≤ S) (hρ : 0 < ρ)
    (hFK : Tendsto (Coverage4ThermalFixed.physicalRingGrowth (times s) next
      (fun ω i => a * host ω i) (fun _ => s) 1 birth μ T) atTop (nhds r)) :
    r ≤ Real.log ρ + (S + a * ∑ i, ∫ ω, host ω i ∂μ) / ρ - 1 := by
  let K := ReferenceRing.kernel (times s) (times_ge_one s hs0 hs1) next
  let H := score s (fun _ => a) host
  let C : Ω → ℝ := fun ω => S + a * ∑ i, host ω i
  let B : Ω → ℝ := fun ω => Real.log ρ + C ω / ρ - 1
  have hC : Integrable C μ := (integrable_const S).add
    ((integrable_finsetSum _ (fun i _ => hhost i)).const_mul a)
  have hB : Integrable B μ := ((integrable_const _).add (hC.div_const ρ)).sub (integrable_const _)
  have hrow : ∀ᵐ ω ∂μ, ∀ i, (∑ e, K.prob i e * Real.exp (H ω e)) ≤ Real.exp (B ω) := by
    filter_upwards [hh] with ω hω
    intro i
    have hsum : host ω i ≤ ∑ j, host ω j :=
      Finset.single_le_sum (fun j _ => (hω j).le) (Finset.mem_univ i)
    have hrowC : s i + a * host ω i ≤ C ω := by
      dsimp [C]
      exact add_le_add (hS i) (mul_le_mul_of_nonneg_left hsum ha.le)
    have hCpos : 0 < C ω := lt_of_lt_of_le
      (add_pos_of_nonneg_of_pos (hs0 i) (mul_pos ha (hω i))) hrowC
    have hlogC : Real.log (C ω) ≤ B ω := by
      have ht := Real.log_le_sub_one_of_pos (div_pos hCpos hρ)
      rw [Real.log_div hCpos.ne' hρ.ne'] at ht
      dsimp [B]
      linarith
    calc
      _ = s i + a * host ω i := score_row_sum s (fun _ => a) host hs0 hs1
        (fun _ => ha) next birth ω hω i
      _ ≤ C ω := hrowC
      _ ≤ Real.exp (B ω) := by simpa only [Real.exp_log hCpos] using Real.exp_le_exp.mpr hlogC
  have hu (n : ℕ) := normalized_growth_upper K μ T hT H
    (score_integrable s (fun _ => a) host μ hs0 (fun _ => ha) hh hlog) B hB hrow n
  have hf : Tendsto (ReferencePaths.normalizedGrowth K μ T H) atTop (nhds r) := by
    rw [show ReferencePaths.normalizedGrowth K μ T H =
      Coverage4ThermalFixed.physicalRingGrowth (times s) next
      (fun ω i => a * host ω i) (fun _ => s) 1 birth μ T from
        normalized_eq_physical s (fun _ => a) host hs0 hs1 (fun _ => ha) next birth μ T hT hh]
    exact hFK
  have hr := le_of_tendsto hf (Eventually.of_forall hu)
  have hEC : (∫ ω, C ω ∂μ) = S + a * ∑ i, ∫ ω, host ω i ∂μ := by
    dsimp [C]
    rw [integral_add (integrable_const S)
      ((integrable_finsetSum _ (fun i _ => hhost i)).const_mul a), integral_const_mul,
      integral_finsetSum _ (fun i _ => hhost i)]
    simp
  have hEB : (∫ ω, B ω ∂μ) = Real.log ρ +
      (S + a * ∑ i, ∫ ω, host ω i ∂μ) / ρ - 1 := by
    dsimp [B]
    have hi : Integrable (fun ω => Real.log ρ + C ω / ρ) μ :=
      (integrable_const _).add (hC.div_const ρ)
    rw [integral_sub hi (integrable_const (1 : ℝ)),
      integral_add (integrable_const (Real.log ρ)) (hC.div_const ρ), integral_div, hEC]
    simp
  rwa [hEB] at hr

/-- Actual finite-mean ring growth is eventually negative at the negative tail,
including rings whose staying contributions are all zero. -/
theorem growth_eventually_neg (s : ι → ℝ) (host : Ω → ι → ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i)
    (hhost : ∀ i, Integrable (fun ω => host ω i) μ)
    (hlog : ∀ i, Integrable (fun ω => Real.log (host ω i)) μ)
    (next : Equiv.Perm ι) (birth : ι) (r : ℝ → ℝ)
    (hFK : ∀ y, Tendsto (Coverage4ThermalFixed.physicalRingGrowth (times s) next
      (fun ω i => Real.exp (y / Fintype.card ι) * host ω i) (fun _ => s) 1 birth μ T)
      atTop (nhds (r y))) : ∀ᶠ y in atBot, r y < 0 := by
  let S := Finset.univ.sup' Finset.univ_nonempty s
  have hS (i : ι) : s i ≤ S := Finset.le_sup' s (Finset.mem_univ i)
  have hS0 : 0 ≤ S := (hs0 (Classical.arbitrary ι)).trans (hS _)
  have hS1 : S < 1 := (Finset.sup'_lt_iff Finset.univ_nonempty).mpr (fun i _ => hs1 i)
  let ρ := (S + 1) / 2
  have hρ0 : 0 < ρ := by dsimp [ρ]; linarith
  have hSρ : S < ρ := by dsimp [ρ]; linarith
  have hρ1 : ρ < 1 := by dsimp [ρ]; linarith
  have hN : (0 : ℝ) < Fintype.card ι := Nat.cast_pos.mpr Fintype.card_pos
  have he : Tendsto (fun y : ℝ => Real.exp (y / Fintype.card ι)) atBot (nhds 0) :=
    Real.tendsto_exp_atBot.comp ((tendsto_div_const_atBot_of_pos hN).mpr tendsto_id)
  have hlim : Tendsto (fun y : ℝ => S + Real.exp (y / Fintype.card ι) *
      ∑ i, ∫ ω, host ω i ∂μ) atBot (nhds S) := by
    simpa using tendsto_const_nhds.add (he.mul_const (∑ i, ∫ ω, host ω i ∂μ))
  filter_upwards [hlim.eventually (gt_mem_nhds hSρ)] with y hy
  have hr := physical_growth_upper_tangent s host μ T hT hs0 hs1 hh hhost hlog next birth
    (Real.exp (y / Fintype.card ι)) S ρ (r y) (Real.exp_pos _) hS hρ0 (hFK y)
  have hfrac : (S + Real.exp (y / Fintype.card ι) * ∑ i, ∫ ω, host ω i ∂μ) / ρ < 1 :=
    (div_lt_one hρ0).mpr hy
  have hlogρ : Real.log ρ < 0 := Real.log_neg hρ0 hρ1
  linarith

end PhysicalLimits

end Dobson.RingGrowth
#print axioms Dobson.RingGrowth.normalized_growth_upper

#print axioms Dobson.RingGrowth.growth_tendsto_atTop
#print axioms Dobson.RingGrowth.growth_eventually_neg
