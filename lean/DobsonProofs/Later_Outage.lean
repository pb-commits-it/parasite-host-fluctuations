import DobsonProofs.RingGrowth

/-! Negative tilt for the actual ring recurrence with zero host multipliers
allowed. Only the expected-growth limit from the standard random-matrix
ergodic theorem is left as a named textbook input. -/
namespace Dobson.LaterOutage
open MeasureTheory Filter Set
open scoped Topology BigOperators

variable {ι Ω : Type*} [Fintype ι] [Nonempty ι] [MeasurableSpace Ω]

/-- Row action of diag(s) plus a weighted cyclic transfer, applied to ones.
This is the transpose convention of Coverage4ThermalFixed.physicalRingStep. -/
noncomputable def population (T : Ω → Ω) (next : ι → ι) (s : ι → ℝ)
    (h : Ω → ι → ℝ) (a : ℝ) : ℕ → Ω → ι → ℝ
  | 0, _, _ => 1
  | n+1, ω, i => s i * population T next s h a n (T ω) i +
      a*h ω i * population T next s h a n (T ω) (next i)

noncomputable def total (T : Ω → Ω) (next : ι → ι) (s : ι → ℝ)
    (h : Ω → ι → ℝ) (a : ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ∑ i, population T next s h a n ω i

lemma population_nonneg (T : Ω → Ω) (next : ι → ι) (s : ι → ℝ)
    (h : Ω → ι → ℝ) (a : ℝ) (hs : ∀ i, 0 ≤ s i) (hh : ∀ ω i, 0 ≤ h ω i)
    (ha : 0 ≤ a) (n : ℕ) (ω : Ω) (i : ι) : 0 ≤ population T next s h a n ω i := by
  induction n generalizing ω i with
  | zero => exact zero_le_one
  | succ n ih =>
    exact add_nonneg (mul_nonneg (hs i) (ih _ _))
      (mul_nonneg (mul_nonneg ha (hh ω i)) (ih _ _))

lemma staying_path_lower (T : Ω → Ω) (next : ι → ι) (s : ι → ℝ)
    (h : Ω → ι → ℝ) (a : ℝ) (hs : ∀ i, 0 ≤ s i) (hh : ∀ ω i, 0 ≤ h ω i)
    (ha : 0 ≤ a) (n : ℕ) (ω : Ω) (i : ι) : (s i)^n ≤ population T next s h a n ω i := by
  induction n generalizing ω with
  | zero => rfl
  | succ n ih =>
    rw [population, pow_succ']
    exact (mul_le_mul_of_nonneg_left (ih (T ω)) (hs i)).trans
      (le_add_of_nonneg_right (mul_nonneg (mul_nonneg ha (hh ω i))
        (population_nonneg T next s h a hs hh ha n (T ω) (next i))))

lemma total_positive (T : Ω → Ω) (next : ι → ι) (s : ι → ℝ)
    (h : Ω → ι → ℝ) (a : ℝ) (hs : ∀ i, 0 ≤ s i) (hh : ∀ ω i, 0 ≤ h ω i)
    (ha : 0 ≤ a) (i : ι) (hi : 0 < s i) (n : ℕ) (ω : Ω) :
    0 < total T next s h a n ω := by
  apply (pow_pos hi n).trans_le
  exact (staying_path_lower T next s h a hs hh ha n ω i).trans
    (Finset.single_le_sum (fun j _ => population_nonneg T next s h a hs hh ha n ω j)
      (Finset.mem_univ i))

lemma population_measurable (T : Ω → Ω) (hT : Measurable T) (next : ι → ι)
    (s : ι → ℝ) (h : Ω → ι → ℝ) (hh : ∀ i, Measurable (fun ω => h ω i))
    (a : ℝ) (n : ℕ) (i : ι) : Measurable (fun ω => population T next s h a n ω i) := by
  induction n generalizing i with
  | zero => exact measurable_const
  | succ n ih =>
    exact (measurable_const.mul ((ih i).comp hT)).add
      ((measurable_const.mul (hh i)).mul ((ih (next i)).comp hT))

noncomputable def budgetRate (S a ρ : ℝ) (h : Ω → ι → ℝ) (ω : Ω) : ℝ :=
  Real.log ρ + (S+a*∑ i, h ω i)/ρ-1

lemma row_bound (s : ι → ℝ) (h : Ω → ι → ℝ) (a S ρ : ℝ)
    (hS : 0 < S) (hsS : ∀ i, s i ≤ S) (hh : ∀ ω i, 0 ≤ h ω i)
    (ha : 0 ≤ a) (hρ : 0 < ρ) (ω : Ω) (i : ι) :
    s i+a*h ω i ≤ Real.exp (budgetRate S a ρ h ω) := by
  let C := S+a*∑ j, h ω j
  have hC : 0 < C := add_pos_of_pos_of_nonneg hS
    (mul_nonneg ha (Finset.sum_nonneg (fun j _ => hh ω j)))
  have hc : s i+a*h ω i ≤ C := add_le_add (hsS i)
    (mul_le_mul_of_nonneg_left (Finset.single_le_sum (fun j _ => hh ω j) (Finset.mem_univ i)) ha)
  have hl := Real.log_le_sub_one_of_pos (div_pos hC hρ)
  rw [Real.log_div hC.ne' hρ.ne'] at hl
  apply hc.trans
  have hx : Real.log C ≤ budgetRate S a ρ h ω := by dsimp [budgetRate, C] at *; linarith
  simpa only [Real.exp_log hC] using Real.exp_le_exp.mpr hx

lemma population_upper (T : Ω → Ω) (next : ι → ι) (s : ι → ℝ)
    (h : Ω → ι → ℝ) (a S ρ : ℝ) (hs : ∀ i, 0 ≤ s i)
    (hS : 0 < S) (hsS : ∀ i, s i ≤ S) (hh : ∀ ω i, 0 ≤ h ω i)
    (ha : 0 ≤ a) (hρ : 0 < ρ) (n : ℕ) (ω : Ω) (i : ι) :
    population T next s h a n ω i ≤ Real.exp (RingGrowth.budget T (budgetRate S a ρ h) n ω) := by
  induction n generalizing ω i with
  | zero => simp [population, RingGrowth.budget]
  | succ n ih =>
    calc
      _ ≤ s i*Real.exp (RingGrowth.budget T (budgetRate S a ρ h) n (T ω)) +
          a*h ω i*Real.exp (RingGrowth.budget T (budgetRate S a ρ h) n (T ω)) :=
        add_le_add (mul_le_mul_of_nonneg_left (ih _ _) (hs i))
          (mul_le_mul_of_nonneg_left (ih _ _) (mul_nonneg ha (hh ω i)))
      _ = (s i+a*h ω i)*Real.exp (RingGrowth.budget T (budgetRate S a ρ h) n (T ω)) := by ring
      _ ≤ Real.exp (budgetRate S a ρ h ω)*Real.exp (RingGrowth.budget T (budgetRate S a ρ h) n (T ω)) :=
        mul_le_mul_of_nonneg_right (row_bound s h a S ρ hS hsS hh ha hρ ω i) (Real.exp_pos _).le
      _ = _ := by rw [← Real.exp_add]; rfl

lemma log_total_bounds (T : Ω → Ω) (next : ι → ι) (s : ι → ℝ)
    (h : Ω → ι → ℝ) (a S ρ : ℝ) (hs : ∀ i, 0 ≤ s i)
    (hS : 0 < S) (hsS : ∀ i, s i ≤ S) (hh : ∀ ω i, 0 ≤ h ω i)
    (ha : 0 ≤ a) (hρ : 0 < ρ) (i : ι) (hi : 0 < s i) (n : ℕ) (ω : Ω) :
    n*Real.log (s i) ≤ Real.log (total T next s h a n ω) ∧
      Real.log (total T next s h a n ω) ≤ Real.log (Fintype.card ι)+
        RingGrowth.budget T (budgetRate S a ρ h) n ω := by
  have hpos := total_positive T next s h a hs hh ha i hi n ω
  constructor
  · have hl := Real.log_le_log (pow_pos hi n)
      ((staying_path_lower T next s h a hs hh ha n ω i).trans
        (Finset.single_le_sum (fun j _ => population_nonneg T next s h a hs hh ha n ω j)
          (Finset.mem_univ i)))
    simpa only [Real.log_pow, total] using hl
  · have hu : total T next s h a n ω ≤ (Fintype.card ι:ℝ)*
        Real.exp (RingGrowth.budget T (budgetRate S a ρ h) n ω) := by
      simpa [total] using Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) =>
        population_upper T next s h a S ρ hs hS hsS hh ha hρ n ω j)
    have hc : (0:ℝ) < Fintype.card ι := Nat.cast_pos.mpr Fintype.card_pos
    simpa [Real.log_mul hc.ne' (Real.exp_ne_zero _), Real.log_exp] using Real.log_le_log hpos hu

noncomputable def growth (μ : Measure Ω) (T : Ω → Ω) (next : ι → ι)
    (s : ι → ℝ) (h : Ω → ι → ℝ) (y : ℝ) (n : ℕ) : ℝ :=
  (∫ ω, Real.log (total T next s h (Real.exp (y/Fintype.card ι)) (n+1) ω) ∂μ)/(n+1:ℕ)

lemma budgetRate_integrable (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h : Ω → ι → ℝ) (hh : ∀ i, Integrable (fun ω => h ω i) μ) (a S ρ : ℝ) :
    Integrable (budgetRate S a ρ h) μ :=
  ((integrable_const _).add (((integrable_const _).add
    ((integrable_finsetSum _ (fun i _ => hh i)).const_mul a)).div_const ρ)).sub (integrable_const _)

lemma budgetRate_integral (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h : Ω → ι → ℝ) (hh : ∀ i, Integrable (fun ω => h ω i) μ) (a S ρ : ℝ) :
    (∫ ω, budgetRate S a ρ h ω ∂μ) = Real.log ρ+(S+a*∑ i, ∫ ω, h ω i ∂μ)/ρ-1 := by
  unfold budgetRate
  have hs := integrable_finsetSum Finset.univ (fun i _ => hh i)
  have hc : Integrable (fun ω => S+a*∑ i, h ω i) μ := (integrable_const S).add (hs.const_mul a)
  have hb : Integrable (fun ω => Real.log ρ+(S+a*∑ i, h ω i)/ρ) μ :=
    (integrable_const _).add (hc.div_const ρ)
  rw [integral_sub hb (integrable_const (1:ℝ)),
    integral_add (integrable_const (Real.log ρ)) (hc.div_const ρ), integral_div,
    integral_add (integrable_const S) (hs.const_mul a), integral_const_mul,
    integral_finsetSum _ (fun i _ => hh i)]
  simp

/-- Finite log growth is integrable even when host values vanish and their
unconditional logarithms are unusable. Only the first host moment is needed. -/
lemma log_total_integrable (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (next : ι → ι)
    (s : ι → ℝ) (h : Ω → ι → ℝ) (a S ρ : ℝ) (hs : ∀ i, 0 ≤ s i)
    (hS : 0 < S) (hsS : ∀ i, s i ≤ S) (hh : ∀ ω i, 0 ≤ h ω i)
    (hm : ∀ i, Measurable (fun ω => h ω i)) (hI : ∀ i, Integrable (fun ω => h ω i) μ)
    (ha : 0 ≤ a) (hρ : 0 < ρ) (i : ι) (hi : 0 < s i) (n : ℕ) :
    Integrable (fun ω => Real.log (total T next s h a n ω)) μ := by
  have hb := RingGrowth.budget_integrable μ T hT _ (budgetRate_integrable μ h hI a S ρ) n
  have hmeas : Measurable (fun ω => Real.log (total T next s h a n ω)) :=
    (Finset.measurable_sum Finset.univ (fun j _ => population_measurable T hT.measurable next s h hm a n j)).log
  apply Integrable.mono' (((integrable_const |Real.log (Fintype.card ι)|).add hb.abs).add
    (integrable_const |(n:ℝ)*Real.log (s i)|)) hmeas.aestronglyMeasurable
  apply Eventually.of_forall
  intro ω
  have hbounds := log_total_bounds T next s h a S ρ hs hS hsS hh ha hρ i hi n ω
  simp only [Real.norm_eq_abs, Pi.add_apply]
  apply abs_le.mpr
  constructor <;> linarith [abs_nonneg (Real.log (Fintype.card ι)),
    abs_nonneg (RingGrowth.budget T (budgetRate S a ρ h) n ω),
    le_abs_self (Real.log (Fintype.card ι)), le_abs_self (RingGrowth.budget T (budgetRate S a ρ h) n ω),
    neg_le_abs ((n:ℝ)*Real.log (s i)), abs_nonneg ((n:ℝ)*Real.log (s i))]

theorem growth_upper (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (next : ι → ι)
    (s : ι → ℝ) (h : Ω → ι → ℝ) (S ρ y r : ℝ) (hs : ∀ i, 0 ≤ s i)
    (hS : 0 < S) (hsS : ∀ i, s i ≤ S) (hh : ∀ ω i, 0 ≤ h ω i)
    (hm : ∀ i, Measurable (fun ω => h ω i)) (hI : ∀ i, Integrable (fun ω => h ω i) μ)
    (hρ : 0 < ρ) (i : ι) (hi : 0 < s i)
    (textbook_expected_growth_limit : Tendsto (growth μ T next s h y) atTop (𝓝 r)) :
    r ≤ Real.log ρ+(S+Real.exp (y/Fintype.card ι)*∑ j, ∫ ω, h ω j ∂μ)/ρ-1 := by
  let a := Real.exp (y/Fintype.card ι)
  let B := budgetRate S a ρ h
  have hB := budgetRate_integrable μ h hI a S ρ
  have hfinite (n : ℕ) : growth μ T next s h y n ≤
      Real.log (Fintype.card ι)/(n+1:ℕ) + ∫ ω, B ω ∂μ := by
    have hb := RingGrowth.budget_integrable μ T hT B hB (n+1)
    have hu := integral_mono (log_total_integrable μ T hT next s h a S ρ hs hS hsS hh hm hI
      (Real.exp_pos _).le hρ i hi (n+1)) ((integrable_const _).add hb)
      (fun ω => (log_total_bounds T next s h a S ρ hs hS hsS hh (Real.exp_pos _).le hρ i hi (n+1) ω).2)
    simp only [Pi.add_apply] at hu
    rw [integral_add (integrable_const _) hb, RingGrowth.integral_budget μ T hT B hB (n+1)] at hu
    simp only [integral_const, probReal_univ, one_smul] at hu
    have hn : (0:ℝ) < (n+1:ℕ) := by positivity
    apply (div_le_div_of_nonneg_right hu hn.le).trans_eq
    field_simp
  have hz : Tendsto (fun n : ℕ => Real.log (Fintype.card ι)/(n+1:ℕ)) atTop (𝓝 0) := by
    simpa only [Function.comp_def] using
      (tendsto_const_div_atTop_nhds_zero_nat (Real.log (Fintype.card ι))).comp (tendsto_add_atTop_nat 1)
  have hlim := le_of_tendsto_of_tendsto' textbook_expected_growth_limit
    (by simpa only [zero_add] using hz.add_const (∫ ω, B ω ∂μ)) hfinite
  rwa [budgetRate_integral μ h hI a S ρ] at hlim

/-- H2: a negative tilt exists for the outage model, including unbounded
nonnegative hosts. No negative-point or growth-upper-bound premise remains. -/
theorem negative_tilt_point (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (next : ι → ι)
    (s : ι → ℝ) (h : Ω → ι → ℝ) (S : ℝ) (hs : ∀ i, 0 ≤ s i)
    (hS : 0 < S) (hS1 : S < 1) (hsS : ∀ i, s i ≤ S) (hh : ∀ ω i, 0 ≤ h ω i)
    (hm : ∀ i, Measurable (fun ω => h ω i)) (hI : ∀ i, Integrable (fun ω => h ω i) μ)
    (i : ι) (hi : 0 < s i) (r : ℝ → ℝ)
    (textbook_expected_growth_limit : ∀ y, Tendsto (growth μ T next s h y) atTop (𝓝 (r y))) :
    (∀ᶠ y in atBot, r y < 0) ∧ ∃ y, r y < 0 := by
  have hN : (0:ℝ) < Fintype.card ι := Nat.cast_pos.mpr Fintype.card_pos
  have he : Tendsto (fun y : ℝ => Real.exp (y/Fintype.card ι)) atBot (𝓝 0) :=
    Real.tendsto_exp_atBot.comp ((tendsto_div_const_atBot_of_pos hN).mpr tendsto_id)
  have hl : Tendsto (fun y : ℝ => Real.log S+
      (S+Real.exp (y/Fintype.card ι)*∑ j, ∫ ω, h ω j ∂μ)/S-1) atBot (𝓝 (Real.log S)) := by
    convert (tendsto_const_nhds.add ((tendsto_const_nhds.add
      (he.mul_const (∑ j, ∫ ω, h ω j ∂μ))).div_const S)).sub_const 1 using 1 <;>
      simp [hS.ne']
  have hneg : ∀ᶠ y in atBot, r y < 0 := by
    filter_upwards [hl.eventually_lt_const (Real.log_neg hS hS1)] with y hy
    exact (growth_upper μ T hT next s h S S y (r y) hs hS hsS hh hm hI hS i hi
      (textbook_expected_growth_limit y)).trans_lt hy
  exact ⟨hneg,hneg.exists⟩


/-- Identification with the existing actual ring matrix product. Zeros in h
are permitted and the one-stage overlap is included by the sum of both terms. -/
theorem population_eq_matrix_action [DecidableEq ι]
    (T : Ω → Ω) (next : Equiv.Perm ι) (s : ι → ℝ) (h : Ω → ι → ℝ)
    (a : ℝ) (birth : ι) (n : ℕ) (ω : Ω) (i : ι) :
    population T next s h a n ω i =
      (Coverage4ThermalFixed.physicalRingProduct next (fun ω i => a*h ω i)
        (fun _ => s) 1 birth T n ω).mulVec (fun _ => 1) i := by
  have hstep (ω : Ω) (v : ι → ℝ) (i : ι) :
      (Coverage4ThermalFixed.physicalRingStep next (fun ω i => a*h ω i)
        (fun _ => s) 1 birth ω).mulVec v i = s i*v i+a*h ω i*v (next i) := by
    simp [Coverage4ThermalFixed.physicalRingStep, Matrix.mulVec, dotProduct,
      Finset.sum_add_distrib, add_mul, ite_mul]
  induction n generalizing ω i with
  | zero => simp [population, Coverage4ThermalFixed.physicalRingProduct]
  | succ n ih =>
    rw [population]
    change _ = (Coverage4ThermalFixed.physicalRingStep next (fun ω i => a*h ω i)
      (fun _ => s) 1 birth ω * Coverage4ThermalFixed.physicalRingProduct next
      (fun ω i => a*h ω i) (fun _ => s) 1 birth T n (T ω)).mulVec (fun _ => 1) i
    rw [← Matrix.mulVec_mulVec, hstep, ih (T ω) i, ih (T ω) (next i)]

end Dobson.LaterOutage

#print axioms Dobson.LaterOutage.negative_tilt_point

#print axioms Dobson.LaterOutage.population_eq_matrix_action
