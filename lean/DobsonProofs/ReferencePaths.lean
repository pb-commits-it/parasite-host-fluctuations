import Mathlib

/-! Finite reference-path growth for a stationary finite Markov kernel.
The environment is an invertible measure-preserving shift. No independence
between different times is used. -/
namespace Dobson.ReferencePaths
open MeasureTheory Filter
open scoped BigOperators Topology

variable {ι E Ω : Type*} [Fintype ι] [Fintype E] [MeasurableSpace Ω]

/-- A finite transition kernel; parallel labeled edges are allowed. -/
structure Kernel (ι E : Type*) [Fintype ι] [Fintype E] where
  prob : ι → E → ℝ
  dest : E → ι
  nonneg : ∀ i e, 0 ≤ prob i e
  total : ∀ i, ∑ e, prob i e = 1
  initial : ι → ℝ
  initial_nonneg : ∀ i, 0 ≤ initial i
  initial_pos : ∀ i, 0 < initial i
  initial_total : ∑ i, initial i = 1
  stationary : ∀ f : ι → ℝ,
    (∑ i, initial i * ∑ e, prob i e * f (dest e)) = ∑ i, initial i * f i

/-- Sum of the reference probabilities times exponential path scores. -/
noncomputable def partition (K : Kernel ι E) (T : Ω → Ω) (H : Ω → E → ℝ) :
    ℕ → Ω → ι → ℝ
  | 0, _, _ => 1
  | n + 1, ω, i => ∑ e, K.prob i e * Real.exp (H ω e) * partition K T H n (T ω) (K.dest e)

lemma partition_pos (K : Kernel ι E) (T : Ω → Ω) (H : Ω → E → ℝ)
    (n : ℕ) (ω : Ω) (i : ι) : 0 < partition K T H n ω i := by
  induction n generalizing ω i with
  | zero => simp [partition]
  | succ n ih =>
    have hex : ∃ e, 0 < K.prob i e := by
      by_contra h
      push Not at h
      have : (∑ e, K.prob i e) ≤ 0 := Finset.sum_nonpos fun e _ => h e
      rw [K.total i] at this
      norm_num at this
    obtain ⟨e, he⟩ := hex
    exact Finset.sum_pos' (fun a _ => mul_nonneg
      (mul_nonneg (K.nonneg i a) (Real.exp_pos _).le) (ih _ _).le)
      ⟨e, Finset.mem_univ _, mul_pos (mul_pos he (Real.exp_pos _)) (ih _ _)⟩

lemma log_sum_exp_bound (w x : E → ℝ) (hw : ∀ e, 0 ≤ w e)
    (hws : ∑ e, w e = 1) :
    |Real.log (∑ e, w e * Real.exp (x e))| ≤ ∑ e, |x e| := by
  let B := ∑ e, |x e|
  have hx (e : E) : |x e| ≤ B := Finset.single_le_sum
    (fun a _ => abs_nonneg (x a)) (Finset.mem_univ e)
  have hlo : Real.exp (-B) ≤ ∑ e, w e * Real.exp (x e) := by
    calc
      _ = ∑ e, w e * Real.exp (-B) := by rw [← Finset.sum_mul, hws, one_mul]
      _ ≤ _ := Finset.sum_le_sum fun e _ => mul_le_mul_of_nonneg_left
        (Real.exp_le_exp.mpr (abs_le.mp (hx e)).1) (hw e)
  have hhi : (∑ e, w e * Real.exp (x e)) ≤ Real.exp B := by
    calc
      _ ≤ ∑ e, w e * Real.exp B := Finset.sum_le_sum fun e _ =>
        mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (abs_le.mp (hx e)).2) (hw e)
      _ = _ := by rw [← Finset.sum_mul, hws, one_mul]
  have hp := lt_of_lt_of_le (Real.exp_pos (-B)) hlo
  apply abs_le.mpr
  constructor
  · simpa using Real.log_le_log (Real.exp_pos (-B)) hlo
  · simpa using Real.log_le_log hp hhi

lemma integrable_log_sum_exp (μ : Measure Ω) (w : E → ℝ) (x : E → Ω → ℝ)
    (hw : ∀ e, 0 ≤ w e) (hws : ∑ e, w e = 1)
    (hx : ∀ e, Integrable (x e) μ) :
    Integrable (fun ω => Real.log (∑ e, w e * Real.exp (x e ω))) μ := by
  have hmeas : AEStronglyMeasurable
      (fun ω => Real.log (∑ e, w e * Real.exp (x e ω))) μ := by
    have he (e : E) : AEMeasurable (fun ω => w e * Real.exp (x e ω)) μ :=
      aemeasurable_const.mul (Real.measurable_exp.comp_aemeasurable (hx e).aemeasurable)
    simpa only [Finset.sum_apply] using
      ((Finset.aemeasurable_sum Finset.univ fun e _ => he e).log).aestronglyMeasurable
  apply Integrable.mono' (integrable_finsetSum _ fun e _ => (hx e).abs) hmeas
  exact Eventually.of_forall fun ω => by
    simpa only [Real.norm_eq_abs] using log_sum_exp_bound w (fun e => x e ω) hw hws

lemma log_partition_integrable (K : Kernel ι E) (μ : Measure Ω)
    [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (H : Ω → E → ℝ) (hH : ∀ e, Integrable (fun ω => H ω e) μ)
    (n : ℕ) (i : ι) : Integrable (fun ω => Real.log (partition K T H n ω i)) μ := by
  induction n generalizing i with
  | zero => simp [partition]
  | succ n ih =>
    have h := integrable_log_sum_exp μ (K.prob i)
      (fun e ω => H ω e + Real.log (partition K T H n (T ω) (K.dest e)))
      (K.nonneg i) (K.total i)
      (fun e => (hH e).add (hT.integrable_comp_of_integrable (ih (K.dest e))))
    simpa only [Real.exp_add, Real.exp_log (partition_pos K T H n _ _),
      mul_assoc, partition] using h

/-- Mean log-score of a single stationary reference edge. -/
noncomputable def meanScore (K : Kernel ι E) (μ : Measure Ω) (H : Ω → E → ℝ) : ℝ :=
  ∑ i, K.initial i * ∑ e, K.prob i e * ∫ ω, H ω e ∂μ

/-- Expected log partition, initially averaged in the stationary law. -/
noncomputable def growth (K : Kernel ι E) (μ : Measure Ω) (T : Ω → Ω)
    (H : Ω → E → ℝ) (n : ℕ) : ℝ :=
  ∑ i, K.initial i * ∫ ω, Real.log (partition K T H n ω i) ∂μ

lemma growth_lower (K : Kernel ι E) (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (H : Ω → E → ℝ) (hH : ∀ e, Integrable (fun ω => H ω e) μ) (n : ℕ) :
    n * meanScore K μ H ≤ growth K μ T H n := by
  induction n with
  | zero => simp [growth, partition]
  | succ n ih =>
    have hi (i : ι) :
        (∑ e, K.prob i e * (∫ ω, H ω e ∂μ +
          ∫ ω, Real.log (partition K T H n ω (K.dest e)) ∂μ)) ≤
        ∫ ω, Real.log (partition K T H (n+1) ω i) ∂μ := by
      have hj (ω : Ω) := strictConcaveOn_log_Ioi.concaveOn.le_map_sum
        (t := Finset.univ)
        (p := fun e => Real.exp (H ω e) * partition K T H n (T ω) (K.dest e))
        (fun e _ => K.nonneg i e) (K.total i)
        (fun e (_ : e ∈ Finset.univ) => mul_pos (Real.exp_pos (H ω e))
          (partition_pos K T H n (T ω) (K.dest e)))
      have hint := integral_mono
        (integrable_finsetSum _ fun e _ =>
          ((hH e).add (hT.integrable_comp_of_integrable
            (log_partition_integrable K μ T hT H hH n (K.dest e)))).const_mul (K.prob i e))
        (log_partition_integrable K μ T hT H hH (n+1) i)
        (fun ω => by
          simpa only [smul_eq_mul, Real.log_mul (Real.exp_pos _).ne'
            (partition_pos K T H n _ _).ne', Real.log_exp, partition, mul_assoc, Pi.add_apply, Function.comp_apply] using hj ω)
      rw [integral_finsetSum _ (fun e _ =>
          ((hH e).add (hT.integrable_comp_of_integrable
            (log_partition_integrable K μ T hT H hH n (K.dest e)))).const_mul (K.prob i e))] at hint
      simp only [Pi.add_apply, Function.comp_apply] at hint
      simp_rw [integral_const_mul] at hint
      have he (e : E) :
          (∫ ω, H ω e + Real.log (partition K T H n (T ω) (K.dest e)) ∂μ) =
          (∫ ω, H ω e ∂μ) + ∫ ω, Real.log (partition K T H n ω (K.dest e)) ∂μ := by
        calc
          _ = (∫ ω, H ω e ∂μ) +
              ∫ ω, Real.log (partition K T H n (T ω) (K.dest e)) ∂μ :=
            integral_add (hH e) (hT.integrable_comp_of_integrable
              (log_partition_integrable K μ T hT H hH n (K.dest e)))
          _ = _ := congrArg ((∫ ω, H ω e ∂μ) + ·)
            (hT.integral_comp' (fun ω => Real.log (partition K T H n ω (K.dest e))))
      simp_rw [he] at hint
      exact hint
    have hb := Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) =>
      mul_le_mul_of_nonneg_left (hi i) (K.initial_nonneg i)
    simp_rw [mul_add, Finset.sum_add_distrib, mul_add, Finset.sum_add_distrib] at hb
    rw [K.stationary (fun i => ∫ ω, Real.log (partition K T H n ω i) ∂μ)] at hb
    change meanScore K μ H + growth K μ T H n ≤ growth K μ T H (n+1) at hb
    push_cast
    nlinarith

/-- The actual path partition with stationary initial distribution. -/
noncomputable def totalPartition (K : Kernel ι E) (T : Ω → Ω)
    (H : Ω → E → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ∑ i, K.initial i * partition K T H n ω i

lemma total_log_integrable (K : Kernel ι E) (μ : Measure Ω)
    [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (H : Ω → E → ℝ) (hH : ∀ e, Integrable (fun ω => H ω e) μ) (n : ℕ) :
    Integrable (fun ω => Real.log (totalPartition K T H n ω)) μ := by
  simpa only [Real.exp_log (partition_pos K T H n _ _), totalPartition] using
    integrable_log_sum_exp μ K.initial
      (fun i ω => Real.log (partition K T H n ω i))
      K.initial_nonneg K.initial_total (log_partition_integrable K μ T hT H hH n)

lemma total_growth_lower (K : Kernel ι E) (μ : Measure Ω)
    [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (H : Ω → E → ℝ) (hH : ∀ e, Integrable (fun ω => H ω e) μ) (n : ℕ) :
    n * meanScore K μ H ≤ ∫ ω, Real.log (totalPartition K T H n ω) ∂μ := by
  apply (growth_lower K μ T hT H hH n).trans
  have h := integral_mono
    (integrable_finsetSum Finset.univ fun i _ =>
      (log_partition_integrable K μ T hT H hH n i).const_mul (K.initial i))
    (total_log_integrable K μ T hT H hH n)
    (fun ω => by
      simpa only [smul_eq_mul, totalPartition] using
        strictConcaveOn_log_Ioi.concaveOn.le_map_sum
          (t := Finset.univ) (p := fun i => partition K T H n ω i)
          (fun i _ => K.initial_nonneg i) K.initial_total
          (fun i _ => partition_pos K T H n ω i))
  simpa only [integral_finsetSum _ (fun i _ =>
    (log_partition_integrable K μ T hT H hH n i).const_mul (K.initial i)),
    integral_const_mul, growth] using h

/-- Positive horizons avoid an artificial zero-time inequality. -/
noncomputable def normalizedGrowth (K : Kernel ι E) (μ : Measure Ω)
    (T : Ω → Ω) (H : Ω → E → ℝ) (n : ℕ) : ℝ :=
  (∫ ω, Real.log (totalPartition K T H (n+1) ω) ∂μ) / (n+1 : ℕ)

lemma normalized_growth_lower (K : Kernel ι E) (μ : Measure Ω)
    [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (H : Ω → E → ℝ) (hH : ∀ e, Integrable (fun ω => H ω e) μ) (n : ℕ) :
    meanScore K μ H ≤ normalizedGrowth K μ T H n := by
  apply (le_div_iff₀ (by positivity : (0 : ℝ) < (n+1 : ℕ))).mpr
  simpa only [mul_comm] using total_growth_lower K μ T hT H hH (n+1)

/-- The concrete random row matrix of the labeled reference network. -/
noncomputable def stepMatrix (K : Kernel ι E) (H : Ω → E → ℝ) (ω : Ω) :
    Matrix ι ι ℝ := by
  classical
  exact fun i j => ∑ e, if K.dest e = j then K.prob i e * Real.exp (H ω e) else 0

lemma stepMatrix_mulVec (K : Kernel ι E) (H : Ω → E → ℝ) (ω : Ω) (v : ι → ℝ)
    (i : ι) : (stepMatrix K H ω).mulVec v i =
      ∑ e, K.prob i e * Real.exp (H ω e) * v (K.dest e) := by
  classical
  simp only [stepMatrix, Matrix.mulVec, dotProduct, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro e _
  simp only [ite_mul, zero_mul]
  simp

/-- Products have the same chronological order as the path recursion. -/
noncomputable def matrixProduct (K : Kernel ι E) (T : Ω → Ω) (H : Ω → E → ℝ) :
    ℕ → Ω → Matrix ι ι ℝ := by
  classical
  exact fun n => Nat.rec (fun _ => 1)
    (fun _ P ω => stepMatrix K H ω * P (T ω)) n

lemma partition_eq_matrixProduct (K : Kernel ι E) (T : Ω → Ω)
    (H : Ω → E → ℝ) (n : ℕ) (ω : Ω) :
    partition K T H n ω = (matrixProduct K T H n ω).mulVec (fun _ => 1) := by
  classical
  induction n generalizing ω with
  | zero => simp [partition, matrixProduct]
  | succ n ih =>
    ext i
    change _ = (stepMatrix K H ω * matrixProduct K T H n (T ω)).mulVec (fun _ => 1) i
    rw [← Matrix.mulVec_mulVec, stepMatrix_mulVec]
    simp only [partition, ih]

#print axioms total_growth_lower
#print axioms partition_eq_matrixProduct

end Dobson.ReferencePaths
