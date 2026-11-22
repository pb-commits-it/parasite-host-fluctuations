import DobsonProofs.GrowthShape
import DobsonProofs.GapOneNearZero

/-! Convexity of the actual second-order population recurrence in log staying.
The finite population starts from (1,1). No convexity hypothesis is used. -/
namespace Dobson.LaterGrowthConvexity
open MeasureTheory Filter Set
open scoped Topology

noncomputable def logAdd (x y : ℝ) : ℝ := Real.log (Real.exp x + Real.exp y)

lemma logAdd_mono {x y z w : ℝ} (hx : x ≤ z) (hy : y ≤ w) :
    logAdd x y ≤ logAdd z w :=
  Real.log_le_log (by positivity) (add_le_add (Real.exp_le_exp.mpr hx) (Real.exp_le_exp.mpr hy))

lemma logAdd_shift (x y c : ℝ) : logAdd (x+c) (y+c) = logAdd x y+c := by
  unfold logAdd
  rw [Real.exp_add, Real.exp_add, ← add_mul, Real.log_mul (by positivity) (Real.exp_ne_zero _), Real.log_exp]

lemma logAdd_distance (x y z w K : ℝ) (hx : |x-z| ≤ K) (hy : |y-w| ≤ K) :
    |logAdd x y-logAdd z w| ≤ K := by
  have h1 := logAdd_mono (show x ≤ z+K by linarith [(abs_le.mp hx).2])
    (show y ≤ w+K by linarith [(abs_le.mp hy).2])
  have h2 := logAdd_mono (show z ≤ x+K by linarith [(abs_le.mp hx).1])
    (show w ≤ y+K by linarith [(abs_le.mp hy).1])
  rw [logAdd_shift] at h1 h2
  exact abs_le.mpr ⟨by linarith, by linarith⟩

lemma logAdd_convex (f g : ℝ → ℝ) (hf : ConvexOn ℝ univ f) (hg : ConvexOn ℝ univ g) :
    ConvexOn ℝ univ (fun L => logAdd (f L) (g L)) := by
  have h := GrowthShape.convex_log_sum_exp (fun _ : Fin 2 => (1/2:ℝ))
    (by intro; norm_num) (by norm_num [Fin.sum_univ_two])
    (fun i L => if i=0 then f L+Real.log 2 else g L+Real.log 2)
    (fun i => by
      split_ifs
      · exact hf.add (convexOn_const _ convex_univ)
      · exact hg.add (convexOn_const _ convex_univ))
  have hhalf (z : ℝ) : (2:ℝ)⁻¹*(z*2)=z := by ring
  simpa [Fin.sum_univ_two, Real.exp_add, Real.exp_log (by norm_num : (0:ℝ)<2),
    logAdd, hhalf] using h

variable {Ω : Type*}

/-- Log of the original finite population: exp(P[n+2]) =
exp(-L) exp(P[n+1]) + exp(eta) exp(P[n]), read along the past shift. -/
noncomputable def population (T : Ω → Ω) (η : Ω → ℝ) (L : ℝ) : ℕ → Ω → ℝ
  | 0, _ => 0
  | 1, _ => 0
  | n+2, ω => logAdd (-L + population T η L (n+1) (T ω))
      (η ω + population T η L n (T (T ω)))

theorem population_convex (T : Ω → Ω) (η : Ω → ℝ) (n : ℕ) (ω : Ω) :
    ConvexOn ℝ univ (fun L => population T η L n ω) := by
  induction n using Nat.twoStepInduction generalizing ω with
  | zero => exact convexOn_const _ convex_univ
  | one => exact convexOn_const _ convex_univ
  | more n ih0 ih1 =>
    exact logAdd_convex _ _ ((show ConvexOn ℝ univ (fun L : ℝ => -L) from
        ⟨convex_univ, fun _ _ _ _ _ _ _ _ _ => by simp only [smul_eq_mul]; ring_nf; rfl⟩).add (ih1 (T ω)))
      ((convexOn_const _ convex_univ).add (ih0 (T (T ω))))

theorem population_antitone (T : Ω → Ω) (η : Ω → ℝ) (n : ℕ) (ω : Ω) :
    Antitone (fun L => population T η L n ω) := by
  induction n using Nat.twoStepInduction generalizing ω with
  | zero => intro x y _; rfl
  | one => intro x y _; rfl
  | more n ih0 ih1 =>
    intro x y hxy
    exact logAdd_mono (add_le_add (neg_le_neg hxy) (ih1 _ hxy)) (add_le_add_right (ih0 _ hxy) _)

variable [MeasurableSpace Ω]

lemma logAdd_integrable (μ : Measure Ω) [IsFiniteMeasure μ] (f g : Ω → ℝ)
    (hf : Integrable f μ) (hg : Integrable g μ) : Integrable (fun ω => logAdd (f ω) (g ω)) μ := by
  have h := ReferencePaths.integrable_log_sum_exp μ (fun _ : Fin 2 => (1/2:ℝ))
    (fun i ω => if i=0 then f ω+Real.log 2 else g ω+Real.log 2)
    (by intro; norm_num) (by norm_num [Fin.sum_univ_two])
    (fun i => by
      split_ifs
      · exact hf.add (integrable_const _)
      · exact hg.add (integrable_const _))
  have hhalf (z : ℝ) : (2:ℝ)⁻¹*(z*2)=z := by ring
  simpa [Fin.sum_univ_two, Real.exp_add, Real.exp_log (by norm_num : (0:ℝ)<2),
    logAdd, hhalf] using h

theorem population_integrable (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (η : Ω → ℝ)
    (hη : Integrable η μ) (L : ℝ) (n : ℕ) : Integrable (population T η L n) μ := by
  induction n using Nat.twoStepInduction with
  | zero => exact integrable_zero _ _ _
  | one => exact integrable_zero _ _ _
  | more n ih0 ih1 =>
    exact logAdd_integrable μ _ _ ((integrable_const (-L)).add (hT.integrable_comp_of_integrable ih1))
      (hη.add (hT.integrable_comp_of_integrable (hT.integrable_comp_of_integrable ih0)))

noncomputable def finiteGrowth (μ : Measure Ω) (T : Ω → Ω) (η : Ω → ℝ)
    (n : ℕ) (L : ℝ) : ℝ := (∫ ω, population T η L (n+1) ω ∂μ) / (n+1:ℕ)

theorem finiteGrowth_convex (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (η : Ω → ℝ)
    (hη : Integrable η μ) (n : ℕ) : ConvexOn ℝ univ (finiteGrowth μ T η n) := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  have hi (L : ℝ) := population_integrable μ T hT η hη L (n+1)
  have hh := integral_mono (hi (a*x+b*y)) ((hi x).const_mul a |>.add ((hi y).const_mul b))
    (fun ω => (population_convex T η (n+1) ω).2 (mem_univ x) (mem_univ y) ha hb hab)
  simp only [Pi.add_apply] at hh
  rw [integral_add ((hi x).const_mul a) ((hi y).const_mul b), integral_const_mul, integral_const_mul] at hh
  simpa [finiteGrowth, smul_eq_mul, add_div, mul_div_assoc] using
    div_le_div_of_nonneg_right hh (by positivity : (0:ℝ) ≤ (n+1:ℕ))

/-- Textbook-only limit corollary for the actual finite recurrence. The sole
limit input is the named expected-growth conclusion of the ergodic theorem;
no convexity, coefficient, remainder or derivative assertion is assumed. -/
theorem growth_convex (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (η : Ω → ℝ)
    (hη : Integrable η μ) (r : ℝ → ℝ)
    (textbook_expected_growth_limit : ∀ L,
      Tendsto (fun n => finiteGrowth μ T η n L) atTop (𝓝 (r L))) :
    ConvexOn ℝ univ r :=
  GrowthShape.limit_convex _ _ (finiteGrowth_convex μ T hT η hη) textbook_expected_growth_limit


/-- A bounded stationary ratio solution determines the finite expected growth
limit. This is proved by comparison, without an ergodic-limit assumption. -/
theorem stationary_growth_limit (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (η X : Ω → ℝ)
    (hη : Integrable η μ) (hX : Integrable X μ) (L K : ℝ) (hK : 0 ≤ K)
    (hbound : ∀ ω, |X ω| ≤ K)
    (hrec : ∀ ω, X ω = logAdd (-L) (η ω-X (T ω))) :
    Tendsto (fun n => finiteGrowth μ T η n L) atTop (𝓝 (∫ ω, X ω ∂μ)) := by
  have hbudget (n : ℕ) (ω : Ω) : RingGrowth.budget T X (n+2) ω =
      logAdd (-L+RingGrowth.budget T X (n+1) (T ω))
        (η ω+RingGrowth.budget T X n (T (T ω))) := by
    simp only [RingGrowth.budget]
    have he := logAdd_shift (-L) (η ω-X (T ω))
      (X (T ω)+RingGrowth.budget T X n (T (T ω)))
    rw [← hrec ω] at he
    convert he.symm using 1 <;> congr 1 <;> ring
  have herr : ∀ n ω, |population T η L n ω-RingGrowth.budget T X n ω| ≤ K := by
    intro n
    induction n using Nat.twoStepInduction with
    | zero => intro ω; simpa [population, RingGrowth.budget] using hK
    | one => intro ω; simpa [population, RingGrowth.budget] using hbound ω
    | more n ih0 ih1 =>
      intro ω
      rw [population, hbudget]
      apply logAdd_distance
      · simpa only [add_sub_add_left_eq_sub] using ih1 (T ω)
      · simpa only [add_sub_add_left_eq_sub] using ih0 (T (T ω))
  have hfinite (n : ℕ) : |finiteGrowth μ T η n L - ∫ ω, X ω ∂μ| ≤ K/(n+1:ℕ) := by
    have hp := population_integrable μ T hT η hη L (n+1)
    have hb := RingGrowth.budget_integrable μ T hT X hX (n+1)
    have hh := norm_integral_le_of_norm_le_const (μ := μ)
      (f := fun ω => population T η L (n+1) ω-RingGrowth.budget T X (n+1) ω)
      (Filter.Eventually.of_forall (fun ω => by simpa only [Real.norm_eq_abs] using herr (n+1) ω))
    rw [integral_sub hp hb, RingGrowth.integral_budget μ T hT X hX (n+1)] at hh
    have hn : (0:ℝ) < (n+1:ℕ) := by positivity
    have hid : finiteGrowth μ T η n L - ∫ ω, X ω ∂μ =
        ((∫ ω, population T η L (n+1) ω ∂μ)-(n+1:ℕ)*(∫ ω, X ω ∂μ))/(n+1:ℕ) := by
      unfold finiteGrowth; field_simp
    rw [hid, abs_div, abs_of_pos hn]
    apply div_le_div_of_nonneg_right _ hn.le
    simpa [Real.norm_eq_abs] using hh
  have hz : Tendsto (fun n : ℕ => K/(n+1:ℕ)) atTop (𝓝 0) := by
    simpa only [Function.comp_def] using
      (tendsto_const_div_atTop_nhds_zero_nat K).comp (tendsto_add_atTop_nat 1)
  have hh : Tendsto (fun n => finiteGrowth μ T η n L - ∫ ω, X ω ∂μ)
      atTop (𝓝 0) := squeeze_zero_norm (fun n => by simpa only [Real.norm_eq_abs] using hfinite n) hz
  have ht := hh.add_const (∫ ω, X ω ∂μ)
  simpa using ht

/-- Convexity of the actual stationary mean. The recurrence, bounded support,
integrability and measure-preserving shift are the model assumptions. In
particular no growth limit or convexity is supplied by the caller. -/
theorem stationary_mean_convex (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (η : Ω → ℝ)
    (hη : Integrable η μ) (X : ℝ → Ω → ℝ) (hX : ∀ L, Integrable (X L) μ)
    (hbound : ∀ L, ∃ K : ℝ, 0 ≤ K ∧ ∀ ω, |X L ω| ≤ K)
    (hrec : ∀ L ω, X L ω = logAdd (-L) (η ω-X L (T ω))) :
    ConvexOn ℝ univ (fun L => ∫ ω, X L ω ∂μ) := by
  apply growth_convex μ T hT η hη
  intro L
  obtain ⟨K,hK,hb⟩ := hbound L
  exact stationary_growth_limit μ T hT η (X L) hη (hX L) L K hK hb (hrec L)


/-- The reflection expectation q(L) of gap1zero, on a common stationary path
space. The argument of Q is the previous log ratio, X at the past shift. -/
noncomputable def reflection (μ : Measure Ω) (T : Ω → Ω) (η : Ω → ℝ)
    (X : ℝ → Ω → ℝ) (L : ℝ) : ℝ :=
  ∫ ω, GapTwoCorrection.Qstep L (X L (T ω)) (η ω) ∂μ

theorem reflection_eq_twice_mean (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (η : Ω → ℝ)
    (hη : Integrable η μ) (hmean : ∫ ω, η ω ∂μ = 0)
    (X : ℝ → Ω → ℝ) (L : ℝ) (hX : Integrable (X L) μ)
    (hrec : ∀ ω, X L ω = logAdd (-L) (η ω-X L (T ω))) :
    reflection μ T η X L = 2 * ∫ ω, X L ω ∂μ := by
  have he (ω : Ω) : GapTwoCorrection.Qstep L (X L (T ω)) (η ω) =
      X L ω + X L (T ω) - η ω := by
    have hh := GapTwoCorrection.Fstep_eq L (X L (T ω)) (η ω)
    change logAdd (-L) (η ω-X L (T ω)) = _ at hh
    rw [← hrec ω] at hh
    linarith
  unfold reflection
  simp_rw [he]
  have hs := integral_sub (hX.add (hT.integrable_comp_of_integrable hX)) hη
  have ha := integral_add hX (hT.integrable_comp_of_integrable hX)
  simp only [Pi.add_apply, Function.comp_apply] at hs ha
  rw [hs, ha, hmean, hT.integral_comp' (X L)]
  ring

/-- The convexity owed in the near-zero proof, for the actual stationary
reflection observable. All hypotheses describe the bounded stationary model;
neither convexity nor a limit or derivative estimate is assumed. -/
theorem reflection_convex (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (η : Ω → ℝ)
    (hη : Integrable η μ) (hmean : ∫ ω, η ω ∂μ = 0)
    (X : ℝ → Ω → ℝ) (hX : ∀ L, Integrable (X L) μ)
    (hbound : ∀ L, ∃ K : ℝ, 0 ≤ K ∧ ∀ ω, |X L ω| ≤ K)
    (hrec : ∀ L ω, X L ω = logAdd (-L) (η ω-X L (T ω))) :
    ConvexOn ℝ univ (reflection μ T η X) := by
  have he : reflection μ T η X = fun L => 2*(∫ ω, X L ω ∂μ) := by
    funext L
    exact reflection_eq_twice_mean μ T hT η hη hmean X L (hX L) (hrec L)
  rw [he]
  exact (stationary_mean_convex μ T hT η hη X hX hbound hrec).smul (by norm_num : (0:ℝ) ≤ 2)

/-- The paper's exact two-sided secant bounds with convexity discharged. The
derivative is still an explicitly scoped premise; smoothness is not relabelled
as textbook probability. -/
theorem reflection_secant_bounds (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (η : Ω → ℝ)
    (hη : Integrable η μ) (hmean : ∫ ω, η ω ∂μ = 0)
    (X : ℝ → Ω → ℝ) (hX : ∀ L, Integrable (X L) μ)
    (hbound : ∀ L, ∃ K : ℝ, 0 ≤ K ∧ ∀ ω, |X L ω| ≤ K)
    (hrec : ∀ L ω, X L ω = logAdd (-L) (η ω-X L (T ω)))
    (L p h : ℝ) (hh : 0 < h) (hd : HasDerivAt (reflection μ T η X) (-p) L) :
    (reflection μ T η X L-reflection μ T η X (L+h))/h ≤ p ∧
      p ≤ (reflection μ T η X (L-h)-reflection μ T η X L)/h :=
  GapOneNearZero.convex_secant_bounds _ L p h hh
    (reflection_convex μ T hT η hη hmean X hX hbound hrec) hd


/-- Bounded hosts force bounded stationary log ratios directly from the two
successive recurrence equations; a support estimate is not assumed. -/
theorem stationary_support (T : Ω → Ω) (η X : Ω → ℝ) (L b : ℝ)
    (hη : ∀ ω, η ω ≤ b) (hrec : ∀ ω, X ω = logAdd (-L) (η ω-X (T ω))) :
    ∀ ω, -L ≤ X ω ∧ X ω ≤ logAdd (-L) (b+L) := by
  have hlo (ω : Ω) : -L ≤ X ω := by
    rw [hrec]
    have hh := Real.log_le_log (Real.exp_pos (-L))
      (le_add_of_nonneg_right (Real.exp_pos (η ω-X (T ω))).le)
    simpa only [Real.log_exp, logAdd] using hh
  intro ω
  refine ⟨hlo ω, ?_⟩
  rw [hrec]
  exact logAdd_mono le_rfl (by linarith [hlo (T ω), hη ω])

/-- The GAP 1 stationary reflection convexity from primitive bounded-host
model data. The stationary solution's measurability and recurrence are given;
support, integrability, growth identification and convexity are all proved. -/
theorem bounded_host_reflection_convex (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (η : Ω → ℝ)
    (hη : Integrable η μ) (hmean : ∫ ω, η ω ∂μ = 0) (b : ℝ)
    (hhost : ∀ ω, η ω ≤ b) (X : ℝ → Ω → ℝ)
    (hmeas : ∀ L, AEStronglyMeasurable (X L) μ)
    (hrec : ∀ L ω, X L ω = logAdd (-L) (η ω-X L (T ω))) :
    ConvexOn ℝ univ (reflection μ T η X) := by
  let K := fun L => |L|+|logAdd (-L) (b+L)|
  have hb : ∀ L ω, |X L ω| ≤ K L := by
    intro L ω
    obtain ⟨hlo,hhi⟩ := stationary_support T η (X L) L b hhost (hrec L) ω
    dsimp [K]
    apply abs_le.mpr
    constructor <;> linarith [le_abs_self L, le_abs_self (logAdd (-L) (b+L)),
      abs_nonneg L, abs_nonneg (logAdd (-L) (b+L))]
  have hi (L : ℝ) : Integrable (X L) μ :=
    Integrable.mono' (integrable_const (K L)) (hmeas L)
      (Eventually.of_forall (fun ω => by simpa only [Real.norm_eq_abs] using hb L ω))
  exact reflection_convex μ T hT η hη hmean X hi
    (fun L => ⟨K L, by dsimp [K]; positivity, hb L⟩) hrec

end Dobson.LaterGrowthConvexity

#print axioms Dobson.LaterGrowthConvexity.population_convex
#print axioms Dobson.LaterGrowthConvexity.growth_convex

#print axioms Dobson.LaterGrowthConvexity.stationary_growth_limit
#print axioms Dobson.LaterGrowthConvexity.reflection_convex
#print axioms Dobson.LaterGrowthConvexity.reflection_secant_bounds

#print axioms Dobson.LaterGrowthConvexity.bounded_host_reflection_convex
