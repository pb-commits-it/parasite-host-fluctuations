import DobsonProofs.RingGrowth

namespace Dobson.GrowthShape
open MeasureTheory Filter
open scoped BigOperators Topology

variable {ι E Ω : Type*} [Fintype ι] [Fintype E] [MeasurableSpace Ω]

lemma weighted_exp_pos (w x : E → ℝ) (hw : ∀ e, 0 ≤ w e)
    (hws : ∑ e, w e = 1) : 0 < ∑ e, w e * Real.exp (x e) := by
  have hex : ∃ e, 0 < w e := by
    by_contra h
    push Not at h
    have hh : (∑ e, w e) ≤ 0 := Finset.sum_nonpos (fun e _ => h e)
    rw [hws] at hh
    norm_num at hh
  obtain ⟨e, he⟩ := hex
  exact Finset.sum_pos' (fun i _ => mul_nonneg (hw i) (Real.exp_pos _).le)
    ⟨e, Finset.mem_univ _, mul_pos he (Real.exp_pos _)⟩

/-- The elementary exponential-convexity proof of finite log-sum-exp convexity. -/
lemma log_sum_exp_mix (w x y : E → ℝ) (hw : ∀ e, 0 ≤ w e)
    (hws : ∑ e, w e = 1) (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hab : a + b = 1) :
    Real.log (∑ e, w e * Real.exp (a*x e+b*y e)) ≤
      a*Real.log (∑ e, w e*Real.exp (x e)) +
      b*Real.log (∑ e, w e*Real.exp (y e)) := by
  let A := Real.log (∑ e, w e*Real.exp (x e))
  let B := Real.log (∑ e, w e*Real.exp (y e))
  have hA := weighted_exp_pos w x hw hws
  have hB := weighted_exp_pos w y hw hws
  have hx : (∑ e, w e*Real.exp (x e-A)) = 1 := by
    simp only [Real.exp_sub, ← mul_div_assoc, ← Finset.sum_div]
    dsimp [A]
    rw [Real.exp_log hA, div_self hA.ne']
  have hy : (∑ e, w e*Real.exp (y e-B)) = 1 := by
    simp only [Real.exp_sub, ← mul_div_assoc, ← Finset.sum_div]
    dsimp [B]
    rw [Real.exp_log hB, div_self hB.ne']
  have hbound : (∑ e, w e*Real.exp (a*(x e-A)+b*(y e-B))) ≤ 1 := by
    calc
      _ ≤ ∑ e, w e*(a*Real.exp (x e-A)+b*Real.exp (y e-B)) := by
        apply Finset.sum_le_sum
        intro e _
        apply mul_le_mul_of_nonneg_left _ (hw e)
        simpa only [smul_eq_mul] using convexOn_exp.2
          (Set.mem_univ (x e-A)) (Set.mem_univ (y e-B)) ha hb hab
      _ = 1 := by
        simp only [mul_add, Finset.sum_add_distrib]
        simp_rw [← mul_assoc, mul_comm (w _) a, mul_comm (w _) b, mul_assoc]
        rw [← Finset.mul_sum, ← Finset.mul_sum, hx, hy, mul_one, mul_one, hab]
  have he : (∑ e, w e * Real.exp (a*x e+b*y e)) ≤ Real.exp (a*A+b*B) := by
    calc
      _ = Real.exp (a*A+b*B) * ∑ e, w e*Real.exp (a*(x e-A)+b*(y e-B)) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro e _
        rw [← mul_assoc, mul_comm (Real.exp _) (w e), mul_assoc, ← Real.exp_add]
        congr 2
        ring
      _ ≤ Real.exp (a*A+b*B)*1 := mul_le_mul_of_nonneg_left hbound (Real.exp_pos _).le
      _ = _ := mul_one _
  simpa only [Real.log_exp] using
    Real.log_le_log (weighted_exp_pos w (fun e => a*x e+b*y e) hw hws) he

lemma convex_log_sum_exp (w : E → ℝ) (hw : ∀ e, 0 ≤ w e)
    (hws : ∑ e, w e = 1) (f : E → ℝ → ℝ)
    (hf : ∀ e, ConvexOn ℝ Set.univ (f e)) :
    ConvexOn ℝ Set.univ (fun y => Real.log (∑ e, w e*Real.exp (f e y))) := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  simp only [smul_eq_mul]
  apply (Real.log_le_log (weighted_exp_pos w (fun e => f e (a*x+b*y)) hw hws) ?_).trans
    (log_sum_exp_mix w (fun e => f e x) (fun e => f e y) hw hws a b ha hb hab)
  exact Finset.sum_le_sum (fun e _ => mul_le_mul_of_nonneg_left
    (Real.exp_le_exp.mpr (hf e |>.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab)) (hw e))

lemma log_partition_convex (K : ReferencePaths.Kernel ι E) (T : Ω → Ω)
    (H : ℝ → Ω → E → ℝ)
    (hH : ∀ ω e, ConvexOn ℝ Set.univ (fun y => H y ω e))
    (n : ℕ) (ω : Ω) (i : ι) :
    ConvexOn ℝ Set.univ (fun y => Real.log (ReferencePaths.partition K T (H y) n ω i)) := by
  induction n generalizing ω i with
  | zero => simpa [ReferencePaths.partition] using (convexOn_const (0 : ℝ) convex_univ)
  | succ n ih =>
    have hh := convex_log_sum_exp (K.prob i) (K.nonneg i) (K.total i)
      (fun e y => H y ω e + Real.log (ReferencePaths.partition K T (H y) n (T ω) (K.dest e)))
      (fun e => (hH ω e).add (ih (T ω) (K.dest e)))
    simpa only [Real.exp_add, Real.exp_log (ReferencePaths.partition_pos K T _ n _ _),
      mul_assoc, ReferencePaths.partition] using hh

lemma total_log_convex (K : ReferencePaths.Kernel ι E) (T : Ω → Ω)
    (H : ℝ → Ω → E → ℝ)
    (hH : ∀ ω e, ConvexOn ℝ Set.univ (fun y => H y ω e)) (n : ℕ) (ω : Ω) :
    ConvexOn ℝ Set.univ (fun y => Real.log (ReferencePaths.totalPartition K T (H y) n ω)) := by
  have hh := convex_log_sum_exp K.initial K.initial_nonneg K.initial_total
    (fun i y => Real.log (ReferencePaths.partition K T (H y) n ω i))
    (fun i => log_partition_convex K T H hH n ω i)
  simpa only [Real.exp_log (ReferencePaths.partition_pos K T _ n _ _),
    ReferencePaths.totalPartition] using hh

lemma normalized_growth_convex (K : ReferencePaths.Kernel ι E)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (H : ℝ → Ω → E → ℝ)
    (hH : ∀ ω e, ConvexOn ℝ Set.univ (fun y => H y ω e))
    (hI : ∀ y e, Integrable (fun ω => H y ω e) μ) (n : ℕ) :
    ConvexOn ℝ Set.univ (fun y => ReferencePaths.normalizedGrowth K μ T (H y) n) := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  have hi (z : ℝ) := ReferencePaths.total_log_integrable K μ T hT (H z) (hI z) (n+1)
  have hh := integral_mono (hi (a*x+b*y)) ((hi x).const_mul a |>.add ((hi y).const_mul b))
    (fun ω => total_log_convex K T H hH (n+1) ω |>.2
      (Set.mem_univ x) (Set.mem_univ y) ha hb hab)
  simp only [Pi.add_apply] at hh
  rw [integral_add ((hi x).const_mul a) ((hi y).const_mul b),
    integral_const_mul, integral_const_mul] at hh
  simpa only [ReferencePaths.normalizedGrowth, smul_eq_mul, add_div, mul_div_assoc] using
    div_le_div_of_nonneg_right hh (by positivity : (0 : ℝ) ≤ (n+1 : ℕ))

lemma limit_convex (f : ℕ → ℝ → ℝ) (r : ℝ → ℝ)
    (hf : ∀ n, ConvexOn ℝ Set.univ (f n))
    (hr : ∀ y, Tendsto (fun n => f n y) atTop (nhds (r y))) :
    ConvexOn ℝ Set.univ r := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  exact le_of_tendsto_of_tendsto' (hr (a • x+b • y))
    ((hr x).const_mul a |>.add ((hr y).const_mul b))
    (fun n => hf n |>.2 (Set.mem_univ x) (Set.mem_univ y) ha hb hab)

section Physical
variable [Nonempty ι] [DecidableEq ι]

lemma score_uniform_convex (s : ι → ℝ) (host : Ω → ι → ℝ)
    (hs0 : ∀ i, 0 ≤ s i) (ω : Ω) (e : ι × Bool) :
    ConvexOn ℝ Set.univ (fun y => RingGrowth.score s
      (fun _ => Real.exp (y / Fintype.card ι)) host ω e) := by
  rcases e with ⟨i, b⟩
  cases b
  · simp only [RingGrowth.score_stay s _ host hs0]
    exact convexOn_const 0 convex_univ
  · by_cases hh : host ω i = 0
    · simp only [RingGrowth.score, Coverage4ThermalFixed.physicalRingScore,
        ↓reduceIte, hh, mul_zero, zero_div, Real.log_zero, zero_add, one_mul]
      exact convexOn_const _ convex_univ
    · have he (y : ℝ) : RingGrowth.score s
          (fun _ => Real.exp (y / Fintype.card ι)) host ω (i, true) =
          y / Fintype.card ι + Real.log (host ω i) + Real.log (RingGrowth.times s i) := by
        simp [RingGrowth.score, Coverage4ThermalFixed.physicalRingScore,
          Real.log_mul (Real.exp_ne_zero _) hh]
      simp_rw [he]
      refine ⟨convex_univ, ?_⟩
      intro x _ y _ a b _ _ hab
      simp only [smul_eq_mul]
      apply le_of_eq
      calc
        _ = (a*x+b*y) / Fintype.card ι + (a+b)*
            (Real.log (host ω i)+Real.log (RingGrowth.times s i)) := by rw [hab]; ring
        _ = _ := by ring

/-- Actual expected physical ring growth is a finite convex function of log-transfer. -/
theorem physical_growth_convex (s : ι → ℝ) (host : Ω → ι → ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i)
    (hlog : ∀ i, Integrable (fun ω => Real.log (host ω i)) μ)
    (next : Equiv.Perm ι) (birth : ι) (r : ℝ → ℝ)
    (hFK : ∀ y, Tendsto (Coverage4ThermalFixed.physicalRingGrowth (RingGrowth.times s) next
      (fun ω i => Real.exp (y / Fintype.card ι) * host ω i) (fun _ => s) 1 birth μ T)
      atTop (nhds (r y))) : ConvexOn ℝ Set.univ r := by
  let K := ReferenceRing.kernel (RingGrowth.times s) (RingGrowth.times_ge_one s hs0 hs1) next
  let H := fun y => RingGrowth.score s (fun _ => Real.exp (y / Fintype.card ι)) host
  apply limit_convex (fun n y => ReferencePaths.normalizedGrowth K μ T (H y) n) r
  · intro n
    exact normalized_growth_convex K μ T hT H
      (score_uniform_convex s host hs0)
      (fun y => RingGrowth.score_integrable s _ host μ hs0 (fun _ => Real.exp_pos _) hh hlog) n
  · intro y
    dsimp [H, K]
    rw [RingGrowth.normalized_eq_physical s _ host hs0 hs1
      (fun _ => Real.exp_pos _) next birth μ T hT hh]
    exact hFK y

theorem continuous_of_convex (r : ℝ → ℝ) (hr : ConvexOn ℝ Set.univ r) :
    Continuous r := by
  exact continuousOn_univ.mp (hr.continuousOn isOpen_univ)

end Physical

section Conjugacy
variable [Nonempty ι] [DecidableEq ι]

lemma matrix_product_nonneg (K : ReferencePaths.Kernel ι E) (T : Ω → Ω)
    (H : Ω → E → ℝ) (n : ℕ) (ω : Ω) (i j : ι) :
    0 ≤ ReferencePaths.matrixProduct K T H n ω i j := by
  induction n generalizing ω i j with
  | zero =>
    simp [ReferencePaths.matrixProduct, Matrix.one_apply]
    split_ifs <;> norm_num
  | succ n ih =>
    change 0 ≤ ∑ k, ReferencePaths.stepMatrix K H ω i k *
      ReferencePaths.matrixProduct K T H n (T ω) k j
    apply Finset.sum_nonneg
    intro k _
    apply mul_nonneg _ (ih _ _ _)
    apply Finset.sum_nonneg
    intro e _
    split_ifs
    · exact mul_nonneg (K.nonneg _ _) (Real.exp_pos _).le
    · rfl

lemma matrix_product_conj (K : ReferencePaths.Kernel ι E)
    (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (H J : Ω → E → ℝ) (d : ι → ℝ)
    (hconj : ∀ᵐ ω ∂μ, ReferencePaths.stepMatrix K H ω * Matrix.diagonal d =
      Matrix.diagonal d * ReferencePaths.stepMatrix K J ω) (n : ℕ) :
    ∀ᵐ ω ∂μ, ReferencePaths.matrixProduct K T H n ω * Matrix.diagonal d =
      Matrix.diagonal d * ReferencePaths.matrixProduct K T J n ω := by
  induction n with
  | zero => exact Eventually.of_forall (fun _ => by simp [ReferencePaths.matrixProduct])
  | succ n ih =>
    filter_upwards [hconj, hT.quasiMeasurePreserving.ae ih] with ω hω hn
    change (ReferencePaths.stepMatrix K H ω * ReferencePaths.matrixProduct K T H n (T ω)) *
        Matrix.diagonal d = Matrix.diagonal d *
      (ReferencePaths.stepMatrix K J ω * ReferencePaths.matrixProduct K T J n (T ω))
    rw [Matrix.mul_assoc, hn, ← Matrix.mul_assoc, hω, Matrix.mul_assoc]

noncomputable def ratioBound (d : ι → ℝ) : ℝ := (∑ i, d i) * ∑ i, (d i)⁻¹

lemma ratio_bound_pos (d : ι → ℝ) (hd : ∀ i, 0 < d i) : 0 < ratioBound d := by
  exact mul_pos (Finset.sum_pos (fun i _ => hd i) Finset.univ_nonempty)
    (Finset.sum_pos (fun i _ => inv_pos.mpr (hd i)) Finset.univ_nonempty)

lemma ratio_le_bound (d : ι → ℝ) (hd : ∀ i, 0 < d i) (i j : ι) :
    d i / d j ≤ ratioBound d := by
  apply mul_le_mul (Finset.single_le_sum (fun k _ => (hd k).le) (Finset.mem_univ i))
    (Finset.single_le_sum (fun k _ => (inv_pos.mpr (hd k)).le) (Finset.mem_univ j))
    (inv_pos.mpr (hd j)).le
    (Finset.sum_nonneg (fun k _ => (hd k).le))

lemma total_partition_comparison (K : ReferencePaths.Kernel ι E) (T : Ω → Ω)
    (H J : Ω → E → ℝ) (d : ι → ℝ) (hd : ∀ i, 0 < d i)
    (n : ℕ) (ω : Ω)
    (hconj : ReferencePaths.matrixProduct K T H n ω * Matrix.diagonal d =
      Matrix.diagonal d * ReferencePaths.matrixProduct K T J n ω) :
    ReferencePaths.totalPartition K T H n ω ≤
      ratioBound d * ReferencePaths.totalPartition K T J n ω ∧
    ReferencePaths.totalPartition K T J n ω ≤
      ratioBound d * ReferencePaths.totalPartition K T H n ω := by
  have hentry (i j : ι) : ReferencePaths.matrixProduct K T H n ω i j * d j =
      d i * ReferencePaths.matrixProduct K T J n ω i j := by
    simpa only [Matrix.mul_diagonal, Matrix.diagonal_mul] using congrFun (congrFun hconj i) j
  have he (i j : ι) : ReferencePaths.matrixProduct K T H n ω i j ≤
        ratioBound d * ReferencePaths.matrixProduct K T J n ω i j ∧
      ReferencePaths.matrixProduct K T J n ω i j ≤
        ratioBound d * ReferencePaths.matrixProduct K T H n ω i j := by
    constructor
    · calc
        _ = (d i / d j) * ReferencePaths.matrixProduct K T J n ω i j := by
          apply (mul_right_cancel₀ (hd j).ne')
          field_simp [(hd j).ne']
          exact hentry i j
        _ ≤ _ := mul_le_mul_of_nonneg_right (ratio_le_bound d hd i j)
          (matrix_product_nonneg K T J n ω i j)
    · calc
        _ = (d j / d i) * ReferencePaths.matrixProduct K T H n ω i j := by
          apply (mul_right_cancel₀ (hd i).ne')
          field_simp [(hd i).ne']
          nlinarith [hentry i j]
        _ ≤ _ := mul_le_mul_of_nonneg_right (ratio_le_bound d hd j i)
          (matrix_product_nonneg K T H n ω i j)
  simp only [ReferencePaths.totalPartition, ReferencePaths.partition_eq_matrixProduct,
    Matrix.mulVec, dotProduct, mul_one]
  constructor
  · rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    rw [mul_left_comm]
    apply mul_le_mul_of_nonneg_left _ (K.initial_nonneg i)
    simpa only [Finset.mul_sum] using
      (Finset.sum_le_sum (s := Finset.univ) (fun j _ => (he i j).1))
  · rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    rw [mul_left_comm]
    apply mul_le_mul_of_nonneg_left _ (K.initial_nonneg i)
    simpa only [Finset.mul_sum] using
      (Finset.sum_le_sum (s := Finset.univ) (fun j _ => (he i j).2))

lemma total_partition_pos (K : ReferencePaths.Kernel ι E) (T : Ω → Ω)
    (H : Ω → E → ℝ) (n : ℕ) (ω : Ω) :
    0 < ReferencePaths.totalPartition K T H n ω := by
  exact Finset.sum_pos (fun i _ => mul_pos (K.initial_pos i)
    (ReferencePaths.partition_pos K T H n ω i)) Finset.univ_nonempty

lemma limit_le_of_total_bound (K : ReferencePaths.Kernel ι E)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (H J : Ω → E → ℝ)
    (hH : ∀ e, Integrable (fun ω => H ω e) μ)
    (hJ : ∀ e, Integrable (fun ω => J ω e) μ)
    (C : ℝ) (hC : 0 < C)
    (hb : ∀ n, ∀ᵐ ω ∂μ, ReferencePaths.totalPartition K T H n ω ≤
      C * ReferencePaths.totalPartition K T J n ω)
    (rH rJ : ℝ)
    (hrH : Tendsto (ReferencePaths.normalizedGrowth K μ T H) atTop (nhds rH))
    (hrJ : Tendsto (ReferencePaths.normalizedGrowth K μ T J) atTop (nhds rJ)) : rH ≤ rJ := by
  have hfinite (n : ℕ) : ReferencePaths.normalizedGrowth K μ T H n ≤
      Real.log C / (n+1 : ℕ) + ReferencePaths.normalizedGrowth K μ T J n := by
    have hI := ReferencePaths.total_log_integrable K μ T hT J hJ (n+1)
    have hI' : Integrable (fun ω => Real.log C +
        Real.log (ReferencePaths.totalPartition K T J (n+1) ω)) μ :=
      (integrable_const _).add hI
    have hi := integral_mono_ae (ReferencePaths.total_log_integrable K μ T hT H hH (n+1)) hI'
      ((hb (n+1)).mono (fun ω hω => by
        simpa only [Real.log_mul hC.ne' (total_partition_pos K T J (n+1) ω).ne'] using
          Real.log_le_log (total_partition_pos K T H (n+1) ω) hω))
    rw [integral_add (integrable_const _) hI] at hi
    simp only [integral_const, smul_eq_mul] at hi
    simp at hi
    simpa only [ReferencePaths.normalizedGrowth, add_div] using
      div_le_div_of_nonneg_right hi (by positivity : (0 : ℝ) ≤ (n+1 : ℕ))
  have hc : Tendsto (fun n : ℕ => Real.log C / (n+1 : ℕ)) atTop (nhds 0) := by
    have hinv : Tendsto (fun n : ℕ => ((n+1 : ℕ) : ℝ)⁻¹) atTop (nhds 0) := by
      simpa [one_div, Function.comp_def] using
        (tendsto_one_div_atTop_nhds_zero_nat (𝕜 := ℝ)).comp (tendsto_add_atTop_nat 1)
    simpa [div_eq_mul_inv] using hinv.const_mul (Real.log C)
  exact le_of_tendsto_of_tendsto' hrH (by simpa only [zero_add] using hc.add hrJ) hfinite

/-- Positive diagonal conjugacy changes finite log growth by a bounded term only. -/
theorem limits_eq_of_diagonal_conjugacy (K : ReferencePaths.Kernel ι E)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (H J : Ω → E → ℝ)
    (hH : ∀ e, Integrable (fun ω => H ω e) μ)
    (hJ : ∀ e, Integrable (fun ω => J ω e) μ)
    (d : ι → ℝ) (hd : ∀ i, 0 < d i)
    (hconj : ∀ᵐ ω ∂μ, ReferencePaths.stepMatrix K H ω * Matrix.diagonal d =
      Matrix.diagonal d * ReferencePaths.stepMatrix K J ω)
    (rH rJ : ℝ)
    (hrH : Tendsto (ReferencePaths.normalizedGrowth K μ T H) atTop (nhds rH))
    (hrJ : Tendsto (ReferencePaths.normalizedGrowth K μ T J) atTop (nhds rJ)) : rH = rJ := by
  have hb (n : ℕ) := (matrix_product_conj K μ T hT H J d hconj n).mono
    (fun ω hω => total_partition_comparison K T H J d hd n ω hω)
  exact le_antisymm
    (limit_le_of_total_bound K μ T hT H J hH hJ (ratioBound d) (ratio_bound_pos d hd)
      (fun n => (hb n).mono (fun _ h => h.1)) rH rJ hrH hrJ)
    (limit_le_of_total_bound K μ T hT J H hJ hH (ratioBound d) (ratio_bound_pos d hd)
      (fun n => (hb n).mono (fun _ h => h.2)) rJ rH hrJ hrH)

theorem physical_growth_eq_of_conj (s g g' : ι → ℝ) (host : Ω → ι → ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (hs0 : ∀ i, 0 ≤ s i) (hs1 : ∀ i, s i < 1)
    (hg : ∀ i, 0 < g i) (hg' : ∀ i, 0 < g' i)
    (hh : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i)
    (hlog : ∀ i, Integrable (fun ω => Real.log (host ω i)) μ)
    (next : Equiv.Perm ι) (birth : ι) (d : ι → ℝ) (hd : ∀ i, 0 < d i)
    (hconj : ∀ᵐ ω ∂μ,
      Coverage4ThermalFixed.physicalRingStep next (fun ω i => g i*host ω i) (fun _ => s) 1 birth ω *
        Matrix.diagonal d = Matrix.diagonal d *
      Coverage4ThermalFixed.physicalRingStep next (fun ω i => g' i*host ω i) (fun _ => s) 1 birth ω)
    (r r' : ℝ)
    (hFK : Tendsto (Coverage4ThermalFixed.physicalRingGrowth (RingGrowth.times s) next
      (fun ω i => g i*host ω i) (fun _ => s) 1 birth μ T) atTop (nhds r))
    (hFK' : Tendsto (Coverage4ThermalFixed.physicalRingGrowth (RingGrowth.times s) next
      (fun ω i => g' i*host ω i) (fun _ => s) 1 birth μ T) atTop (nhds r')) : r = r' := by
  let K := ReferenceRing.kernel (RingGrowth.times s) (RingGrowth.times_ge_one s hs0 hs1) next
  have hstep (f : ι → ℝ) (hf : ∀ i, 0 < f i) : ∀ᵐ ω ∂μ,
      ReferencePaths.stepMatrix K (RingGrowth.score s f host) ω =
      Coverage4ThermalFixed.physicalRingStep next (fun ω i => f i*host ω i) (fun _ => s) 1 birth ω := by
    filter_upwards [hh] with ω hω
    have he := Coverage4ThermalFixed.physicalRing_step_identification
      (RingGrowth.times s) (fun _ => 1) (RingGrowth.times_ge_one s hs0 hs1) next
      (fun ω i => f i*host ω i) (fun _ => s) 1 birth (by intro; norm_num)
      (by norm_num) ω (fun i => mul_pos (hf i) (hω i)) (RingGrowth.stay_domain s hs0)
    simpa only [RingGrowth.score, K, Real.log_one, zero_mul, sub_zero] using he
  apply limits_eq_of_diagonal_conjugacy K μ T hT (RingGrowth.score s g host)
    (RingGrowth.score s g' host)
    (RingGrowth.score_integrable s g host μ hs0 hg hh hlog)
    (RingGrowth.score_integrable s g' host μ hs0 hg' hh hlog) d hd
  · filter_upwards [hconj, hstep g hg, hstep g' hg'] with ω hω he he'
    rw [he, he']
    exact hω
  · rw [RingGrowth.normalized_eq_physical s g host hs0 hs1 hg next birth μ T hT hh]
    exact hFK
  · rw [RingGrowth.normalized_eq_physical s g' host hs0 hs1 hg' next birth μ T hT hh]
    exact hFK'

end Conjugacy
end Dobson.GrowthShape

#print axioms Dobson.GrowthShape.total_log_convex
#print axioms Dobson.GrowthShape.physical_growth_convex

#print axioms Dobson.GrowthShape.physical_growth_eq_of_conj
