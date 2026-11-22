import DobsonProofs.FixedCycleLimits
import DobsonProofs.ReferenceRing
import DobsonProofs.Coverage2Bounds
import DobsonProofs.FixedCycle
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Topology.Instances.Matrix
import Mathlib.Topology.Algebra.InfiniteSum.Ring
import Mathlib.Topology.Algebra.InfiniteSum.NatInt
import Mathlib.Tactic

/-! # Batch 4: thermal clocks, fixed-duration cohorts and edge frequencies

The actual ring matrix conserves supplied cyclic reproductive-value weights.
Daily cyclic cohort products, the fixed-duration loss identity at an existing
zero-growth divisor, and reference-kernel edge bounds are proved. See the dated
coverage note below for the remaining model-construction and growth limits. -/
/-! ## Coverage correction — 2026-11-22

`prop:mech-fixed`, `prop:mech-thermal` and `prop:mech-edge-bound` have core
coverage in the version-2 inventory. Daily cyclic cohort products and the loss
identity `D = -E log h` at a supplied zero-growth divisor are proved. Existence
of that divisor, including all-daily-time growth and root construction, is
outside the fixed-duration credit. The cycle-shift averages remain explicit.

Thermal conservation uses supplied cyclic weights; their positive construction,
return-product factorization, growth/root identification and expectation
expansion remain outside credit. The edge bound is connected to the actual
reference kernel; general stationary-network construction remains outside credit. -/

namespace Dobson.Coverage4ThermalFixed
open scoped BigOperators
open Filter

/-- A ring step, allowing the successor to be any permutation (including n=1). -/
def thermalStep {ι : Type*} [DecidableEq ι] (next : ι ≃ ι) (β q : ι → ℝ) : Matrix ι ι ℝ :=
  fun i j => (if i = j then 1-q j else 0) + (if i = next j then β j*q j else 0)

/-- Exact reproductive-value conservation for the time-dependent ring.
The positive cyclic weights are explicit inputs; their construction is not hidden. -/
theorem thermal_conservation {ι : Type*} [Fintype ι] [DecidableEq ι]
    (next : ι ≃ ι) (β q w : ι → ℝ) (hw : ∀ j, w (next j)*β j = w j) :
    Matrix.vecMul w (thermalStep next β q) = w := by
  funext j
  simp only [Matrix.vecMul, dotProduct, thermalStep, mul_add, Finset.sum_add_distrib]
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, if_true]
  rw [← mul_assoc, hw]
  ring

/-- The same conserved row vector survives every finite ordered product. -/
theorem thermal_product_conservation {ι : Type*} [Fintype ι] [DecidableEq ι]
    (w : ι → ℝ) (L : List (Matrix ι ι ℝ))
    (hL : ∀ A ∈ L, Matrix.vecMul w A = w) :
    Matrix.vecMul w L.prod = w := by
  induction L with
  | nil => simp
  | cons A L ih =>
    rw [List.prod_cons, ← Matrix.vecMul_vecMul, hL A (by simp)]
    exact ih (fun B hB => hL B (by simp [hB]))

/-- Arithmetic mean transfer/exit ratios cancel the same thermal clock. -/
theorem thermal_mean_ratio {ι : Type*} (S : Finset ι) (β q : ι → ℝ)
    (hq : ∀ i ∈ S, q i ≠ 0) :
    (∏ i ∈ S, (β i * q i) / q i) = ∏ i ∈ S, β i := by
  apply Finset.prod_congr rfl
  intro i hi
  exact mul_div_cancel_right₀ (β i) (hq i hi)

/-- Positive return excess when the other stages' return product r is below one. -/
theorem thermal_return_excess (q r : ℝ) (hq : 0 < q) (hq1 : q < 1)
    (hr : 0 ≤ r) (hr1 : r < 1) :
    0 < (q-q*r)/(1-q*r) := by
  have hqr : q*r < q := mul_lt_of_lt_one_right hq hr1
  exact div_pos (by linarith) (by linarith)

/-- Exact local gain equivalence, with positivity assumptions made explicit. -/
theorem thermal_gain (π χ K X : ℝ) (hπ : π ≠ 0) (hK : 0 ≤ K) (hX : 0 < X) :
    π*(π*K+χ*X) < 0 ↔ π*χ < 0 ∧ |χ/π| > K/X := by
  have hp : 0 < π^2 := sq_pos_of_ne_zero hπ
  have he : π*(π*K+χ*X) = π^2*(K+(χ/π)*X) := by field_simp <;> ring
  have he2 : π*χ = π^2*(χ/π) := by field_simp <;> ring
  have hsign (v : ℝ) : π^2*v < 0 ↔ v < 0 := by
    constructor
    · intro h
      by_contra hn
      have hh := mul_nonneg hp.le (le_of_not_gt hn)
      linarith
    · exact mul_neg_of_pos_of_neg hp
  rw [he, hsign, he2, hsign]
  constructor
  · intro h
    have hn : χ/π < 0 := by
      by_contra hnot
      have := mul_nonneg (le_of_not_gt hnot) hX.le
      linarith
    refine ⟨hn, ?_⟩
    rw [abs_of_neg hn, gt_iff_lt, div_lt_iff₀ hX]
    linarith
  · rintro ⟨hn, hb⟩
    rw [abs_of_neg hn, gt_iff_lt, div_lt_iff₀ hX] at hb
    linarith

/-- The product-rule curvature difference underlying the thermal baseline.
Values of p,p',p'',q,q',q'' are explicit; differentiability of the responses and
exchange of expectation and Taylor expansion remain separate analytic inputs. -/
theorem thermal_curvature (p p1 p2 q q1 q2 : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
    (p2*q+2*p1*q1+p*q2)/(p*q)-q2/q = p2/p+2*(p1/p)*(q1/q) := by
  field_simp
  ring

/-- Actual daily matrix of the two-position fixed cycle, with one noisy transfer. -/
def fixedTwoStep (g0 g1 : ℝ) (h : ℕ → ℝ) (t : ℕ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0,g1;g0*h t,0]

/-- The actual period product separates the two residue cohorts. -/
theorem fixed_two_period (g0 g1 : ℝ) (h : ℕ → ℝ) (q : ℕ) :
    fixedTwoStep g0 g1 h (2*q+1) * fixedTwoStep g0 g1 h (2*q) =
      Matrix.diagonal ![g0*g1*h (2*q), g0*g1*h (2*q+1)] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [fixedTwoStep, Matrix.mul_apply, Fin.sum_univ_two, Matrix.diagonal] <;> ring

/-- Chronologically ordered products of complete two-position circuits. -/
def fixedTwoProduct (g0 g1 : ℝ) (h : ℕ → ℝ) : ℕ → Matrix (Fin 2) (Fin 2) ℝ
  | 0 => 1
  | q+1 => (fixedTwoStep g0 g1 h (2*q+1) * fixedTwoStep g0 g1 h (2*q)) *
      fixedTwoProduct g0 g1 h q

/-- Exact q-circuit matrix product for the actual two-position fixed-duration model. -/
theorem fixed_two_all_circuits (g0 g1 : ℝ) (h : ℕ → ℝ) (q : ℕ) :
    fixedTwoProduct g0 g1 h q = Matrix.diagonal
      ![(g0*g1)^q*(∏ j ∈ Finset.range q, h (2*j)),
        (g0*g1)^q*(∏ j ∈ Finset.range q, h (2*j+1))] := by
  induction q with
  | zero =>
    ext i j
    fin_cases i <;> fin_cases j <;> simp [fixedTwoProduct, Matrix.diagonal]
  | succ q ih =>
    rw [fixedTwoProduct, fixed_two_period, ih]
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [Matrix.mul_apply, Fin.sum_univ_two, Matrix.diagonal, pow_succ,
        Finset.prod_range_succ] <;> ring

/-- The arbitrary finite daily weighted cycle gives the manuscript cohort product.
The residue r locates the unique noisy edge along the chosen cohort's orbit;
this is a coordinate choice, not an assumed once-per-circuit recurrence. -/
theorem fixed_cohort_product {ι : Type*} [Fintype ι] [DecidableEq ι]
    (next : Equiv.Perm ι) (hc : next.IsCycleOn (Finset.univ : Finset ι))
    (g : ι → ℝ) (birth i : ι) (h : ℕ → ℝ) (x₀ : ι → ℝ) (q r : ℕ)
    (hr : r < Fintype.card ι) (hri : (next^r) i = birth) :
    Dobson.FixedCycle.population next g birth h x₀ (q*Fintype.card ι) i =
      (∏ j, g j)^q * (∏ j ∈ Finset.range q, h (j*Fintype.card ι+r)) * x₀ i :=
  Dobson.FixedCycle.cohort_product next hc g birth i h x₀ q r hr hri

/-- Exact logarithm of the cohort product, for every finite number of circuits. -/
theorem fixed_cohort_log (n r q : ℕ) (R0 : ℝ) (h : ℕ → ℝ)
    (hR : 0 < R0) (hh : ∀ j, 0 < h j) :
    Real.log (R0^q * ∏ j ∈ Finset.range q, h (j*n+r)) =
      q*Real.log R0 + ∑ j ∈ Finset.range q, Real.log (h (j*n+r)) := by
  rw [Real.log_mul (ne_of_gt (pow_pos hR q))
    (ne_of_gt (Finset.prod_pos (fun j _ => hh (j*n+r)))), Real.log_pow,
    Real.log_prod (fun j _ => ne_of_gt (hh (j*n+r)))]

/-- The actual zero-growth daily fixed-cycle population has the stated log-loss.
Birkhoff is required for the complete-cycle shift on every residue. The root
premise is the actual daily aggregate-growth limit; its formula is proved. -/
/- 2026-11-16 domain extension: arbitrary nonzero nonnegative initial population,
   and independently designated reproductive/noisy edges. -/
theorem fixed_threshold_loss {ι : Type*} [Fintype ι] [DecidableEq ι] [Nonempty ι]
    (next : Equiv.Perm ι) (hc : next.IsCycleOn (Finset.univ : Finset ι))
    (g : ι → ℝ) (birth noise : ι) (h : ℕ → ℝ) (x₀ : ι → ℝ)
    (hg : ∀ j, 0 < g j) (hh : ∀ t, 0 < h t) (hx : ∀ j, 0 ≤ x₀ j) (hxne : ∃ j, 0 < x₀ j)
    (Rstar meanLog σ : ℝ) (hR : 0 < Rstar)
    (hBirkhoff_cycle : ∀ r < Fintype.card ι, Tendsto (fun q : ℕ =>
      (∑ j ∈ Finset.range (q+1), Real.log (h (j*Fintype.card ι+r))) / (q+1 : ℕ))
      atTop (nhds meanLog))
    (hzero_growth : Tendsto (fun t : ℕ => Real.log
      (∑ i, Dobson.FixedCycle.population next (Dobson.FixedCycle.dividedFactors g birth Rstar)
        noise h x₀ t i) / t) atTop (nhds 0))
    (hlognormal : meanLog = -σ^2/2) :
    Real.log (∏ i, g i) - Real.log Rstar = -meanLog ∧
      Real.log (∏ i, g i) - Real.log Rstar = σ^2/2 := by
  have he := Dobson.FixedCycle.threshold_log_loss next hc g birth noise h x₀ hg hh hx hxne
    Rstar meanLog hR hBirkhoff_cycle hzero_growth
  exact ⟨he, by linarith⟩

/-- Exact staying/transfer frequencies in a geometric threshold reference ring. -/
theorem edge_frequencies (τ T : ℝ) (hτ : τ ≠ 0) (hT : T ≠ 0) :
    (τ/T)*(1-1/τ) = (τ-1)/T ∧ (τ/T)*(1/τ) = 1/T := by
  constructor <;> field_simp <;> ring

/-- The weighted edge bound for the constructed geometric reference ring.
The finite Jensen inequality and the visit frequencies are proved, and the
only limiting premise is the named textbook expected growth limit at a root.
2026-11-16 leanp1: removes the supplied weighted inequality hgrowth. -/
private theorem edge_bound_normalized {ι Ω : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    [MeasurableSpace Ω] (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i) (next : Equiv.Perm ι)
    (birth : ι) (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasureTheory.MeasurePreserving T μ μ)
    (H : Ω → (ι × Bool) → ℝ)
    (hH : ∀ e, MeasureTheory.Integrable (fun ω => H ω e) μ) (D : ℝ)
    (hFurstenbergKesten_at_root : Tendsto
      (ReferencePaths.normalizedGrowth (ReferenceRing.kernel τ hτ next) μ T
        (fun ω e => H ω e + D * (if e = (birth, true) then 1 else 0)))
      atTop (nhds 0)) :
    D ≤ (∑ i, -(∫ ω, H ω (i, true) ∂μ)) +
      ∑ i, (τ i - 1) * (-(∫ ω, H ω (i, false) ∂μ)) := by
  let K := ReferenceRing.kernel τ hτ next
  let β : ι × Bool → ℝ := fun e => if e = (birth, true) then 1 else 0
  have htot := ReferenceRing.totalTime_pos τ hτ
  have hb : Coverage2Bounds.birthFrequency K β = 1 / ReferenceRing.totalTime τ := by
    unfold Coverage2Bounds.birthFrequency β
    simp only [mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
    exact ReferenceRing.transfer_frequency τ hτ next birth
  have h := Coverage2Bounds.path_root_bound K μ T hT H hH β D
    (by rw [hb]; positivity) hFurstenbergKesten_at_root
  rw [hb, ReferenceRing.meanScore_formula τ hτ next μ H] at h
  calc
    D ≤ -((∑ i, ((∫ ω, H ω (i, true) ∂μ) +
        (τ i - 1) * ∫ ω, H ω (i, false) ∂μ)) / ReferenceRing.totalTime τ) /
        (1 / ReferenceRing.totalTime τ) := h
    _ = -(∑ i, ((∫ ω, H ω (i, true) ∂μ) +
        (τ i - 1) * ∫ ω, H ω (i, false) ∂μ)) := by field_simp
    _ = _ := by
      simp only [Finset.sum_add_distrib, Finset.sum_neg_distrib, mul_neg]
      ring

/-- Actual geometric-ring row matrix: column-population evolution is its transpose.
The existing divisor Rstar acts on the designated birth transfer. -/
noncomputable def physicalRingStep {ι Ω : Type*} [DecidableEq ι]
    (next : Equiv.Perm ι) (g stay : Ω → ι → ℝ) (Rstar : ℝ) (birth : ι)
    (ω : Ω) : Matrix ι ι ℝ := fun i j =>
  (if i = j then stay ω i else 0) +
    (if next i = j then g ω i / (if i = birth then Rstar else 1) else 0)

/-- Chronological row product, the transpose of the ecological column product. -/
noncomputable def physicalRingProduct {ι Ω : Type*} [Fintype ι] [DecidableEq ι]
    (next : Equiv.Perm ι) (g stay : Ω → ι → ℝ) (Rstar : ℝ) (birth : ι)
    (T : Ω → Ω) : ℕ → Ω → Matrix ι ι ℝ :=
  Nat.rec (fun _ => 1)
    (fun _ P ω => physicalRingStep next g stay Rstar birth ω * P (T ω))

noncomputable def physicalRingGrowth {ι Ω : Type*} [Fintype ι] [DecidableEq ι]
    [MeasurableSpace Ω] (τ : ι → ℝ) (next : Equiv.Perm ι)
    (g stay : Ω → ι → ℝ) (Rstar : ℝ) (birth : ι) (μ : MeasureTheory.Measure Ω)
    (T : Ω → Ω) (n : ℕ) : ℝ :=
  (∫ ω, Real.log (∑ i, (τ i / ReferenceRing.totalTime τ) *
    (physicalRingProduct next g stay Rstar birth T (n+1) ω).mulVec (fun _ => 1) i) ∂μ) /
      (n+1 : ℕ)

/-- No stay label is used when its mean stay is zero. -/
noncomputable def physicalRingScore {ι Ω : Type*} (τ gbar : ι → ℝ)
    (g stay : Ω → ι → ℝ) (ω : Ω) (e : ι × Bool) : ℝ :=
  if e.2 then Real.log (g ω e.1 / gbar e.1) + Real.log (gbar e.1 * τ e.1)
  else if τ e.1 = 1 then 0 else Real.log (stay ω e.1 / (1 - 1 / τ e.1))

lemma physicalRing_step_identification {ι Ω : Type*} [Fintype ι] [Nonempty ι]
    [DecidableEq ι] (τ gbar : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i)
    (next : Equiv.Perm ι) (g stay : Ω → ι → ℝ) (Rstar : ℝ) (birth : ι)
    (hgbar : ∀ i, 0 < gbar i) (hR : 0 < Rstar) (ω : Ω)
    (hg : ∀ i, 0 < g ω i)
    (hs : ∀ i, if τ i = 1 then stay ω i = 0 else 0 < stay ω i) :
    ReferencePaths.stepMatrix (ReferenceRing.kernel τ hτ next)
      (fun ω e => physicalRingScore τ gbar g stay ω e -
        Real.log Rstar * (if e = (birth, true) then 1 else 0)) ω =
      physicalRingStep next g stay Rstar birth ω := by
  classical
  have hp (i : ι) : 0 < τ i := lt_of_lt_of_le zero_lt_one (hτ i)
  have htransfer (i : ι) :
      (1 / τ i) * Real.exp (physicalRingScore τ gbar g stay ω (i, true) -
        Real.log Rstar * (if (i, true) = (birth, true) then 1 else 0)) =
      g ω i / (if i = birth then Rstar else 1) := by
    simp only [physicalRingScore, ↓reduceIte, Prod.mk.injEq, and_true]
    rw [Real.exp_sub, Real.exp_add, Real.exp_log (div_pos (hg i) (hgbar i)),
      Real.exp_log (mul_pos (hgbar i) (hp i))]
    split_ifs with hi
    · simp only [mul_one, Real.exp_log hR]
      field_simp [(hp i).ne', (hgbar i).ne', hR.ne']
    · simp only [mul_zero, Real.exp_zero, div_one]
      field_simp [(hp i).ne', (hgbar i).ne']
  have hstay (i : ι) :
      (1 - 1 / τ i) * Real.exp (physicalRingScore τ gbar g stay ω (i, false) -
        Real.log Rstar * (if (i, false) = (birth, true) then 1 else 0)) = stay ω i := by
    simp only [physicalRingScore, Bool.false_eq_true, ↓reduceIte, Prod.mk.injEq,
      and_false, mul_zero, sub_zero]
    by_cases hi : τ i = 1
    · have hh : stay ω i = 0 := by simpa [hi] using hs i
      simp [hi, hh]
    · have hh : 0 < 1 - 1 / τ i := by
        have ht : 1 < τ i := lt_of_le_of_ne (hτ i) (Ne.symm hi)
        have := (div_lt_one (hp i)).mpr ht
        linarith
      rw [ite_eq_right hi, Real.exp_log (div_pos (by simpa [hi] using hs i) hh)]
      field_simp
  unfold physicalRingStep
  ext i j
  simp only [ReferencePaths.stepMatrix, ReferenceRing.kernel,
    ReferenceRing.probability, ReferenceRing.destination, Fintype.sum_prod_type]
  simp only [Fintype.sum_bool, ↓reduceIte, ite_mul, zero_mul]
  simp only [Bool.false_eq_true, ↓reduceIte, Finset.sum_add_distrib]
  have heq (f : ι → ℝ) (P : ι → Prop) [DecidablePred P] :
      (∑ x, if P x then if i = x then f x else 0 else 0) =
        if P i then f i else 0 := by
    calc
      _ = ∑ x, if i = x then if P x then f x else 0 else 0 := by
        apply Finset.sum_congr rfl
        intro x _
        split_ifs <;> rfl
      _ = _ := by simp
  rw [heq, heq, htransfer, hstay]
  exact add_comm _ _

/-- The physical product and the reference product agree almost everywhere,
using only the primitive positive-edge domain and the mean-zero-stay exception. -/
lemma physicalRing_growth_identification {ι Ω : Type*} [Fintype ι] [Nonempty ι]
    [DecidableEq ι] [MeasurableSpace Ω] (τ gbar : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i)
    (next : Equiv.Perm ι) (g stay : Ω → ι → ℝ) (Rstar : ℝ) (birth : ι)
    (hgbar : ∀ i, 0 < gbar i) (hR : 0 < Rstar)
    (μ : MeasureTheory.Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasureTheory.MeasurePreserving T μ μ)
    (hg : ∀ᵐ ω ∂μ, ∀ i, 0 < g ω i)
    (hs : ∀ᵐ ω ∂μ, ∀ i, if τ i = 1 then stay ω i = 0 else 0 < stay ω i) :
    ReferencePaths.normalizedGrowth (ReferenceRing.kernel τ hτ next) μ T
      (fun ω e => physicalRingScore τ gbar g stay ω e -
        Real.log Rstar * (if e = (birth, true) then 1 else 0)) =
      physicalRingGrowth τ next g stay Rstar birth μ T := by
  classical
  let K := ReferenceRing.kernel τ hτ next
  let H := fun ω e => physicalRingScore τ gbar g stay ω e -
    Real.log Rstar * (if e = (birth, true) then 1 else 0)
  have hstep : ∀ᵐ ω ∂μ, ReferencePaths.stepMatrix K H ω =
      physicalRingStep next g stay Rstar birth ω := by
    filter_upwards [hg, hs] with ω hgω hsω
    exact physicalRing_step_identification τ gbar hτ next g stay Rstar birth hgbar hR ω hgω hsω
  have hprod (n : ℕ) : ∀ᵐ ω ∂μ, ReferencePaths.matrixProduct K T H n ω =
      physicalRingProduct next g stay Rstar birth T n ω := by
    induction n with
    | zero => exact Filter.Eventually.of_forall (fun _ => by
        simp only [ReferencePaths.matrixProduct, physicalRingProduct]
        ext i j
        simp [Matrix.one_apply])
    | succ n ih =>
      filter_upwards [hstep, hT.quasiMeasurePreserving.ae ih] with ω hω hn
      simpa only [ReferencePaths.matrixProduct, physicalRingProduct] using congrArg₂ (· * ·) hω hn
  funext n
  unfold ReferencePaths.normalizedGrowth physicalRingGrowth
  congr 1
  apply MeasureTheory.integral_congr_ae
  filter_upwards [hprod (n+1)] with ω hω
  congr 1
  unfold ReferencePaths.totalPartition
  change (∑ i, K.initial i * ReferencePaths.partition K T H (n+1) ω i) = _
  rw [ReferencePaths.partition_eq_matrixProduct, hω]
  rfl

/-- The manuscript edge bound for the actual physical ring at an existing
zero-growth divisor. Transfer and staying scores use their actual normalized
rates. The displayed D is log(R0)-log(Rstar), since R0=prod(gbar*tau).
When tau=1 the physical stay is zero and its log penalty is defined to be zero.
Only the named textbook expected growth limit remains an asymptotic premise. -/
theorem edge_bound_from_growth {ι Ω : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    [MeasurableSpace Ω] (τ gbar : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i)
    (next : Equiv.Perm ι) (birth : ι)
    (g stay : Ω → ι → ℝ) (Rstar : ℝ)
    (hgbar : ∀ i, 0 < gbar i) (hR : 0 < Rstar)
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasureTheory.MeasurePreserving T μ μ)
    (hg : ∀ᵐ ω ∂μ, ∀ i, 0 < g ω i)
    (hs : ∀ᵐ ω ∂μ, ∀ i, if τ i = 1 then stay ω i = 0 else 0 < stay ω i)
    (hglog : ∀ i, MeasureTheory.Integrable (fun ω => Real.log (g ω i / gbar i)) μ)
    (hslog : ∀ i, MeasureTheory.Integrable
      (fun ω => if τ i = 1 then 0 else Real.log (stay ω i / (1 - 1 / τ i))) μ)
    (hFurstenbergKesten_at_root : Tendsto
      (physicalRingGrowth τ next g stay Rstar birth μ T) atTop (nhds 0)) :
    (∑ i, Real.log (gbar i * τ i)) - Real.log Rstar ≤
      (∑ i, -(∫ ω, Real.log (g ω i / gbar i) ∂μ)) +
      ∑ i, (τ i - 1) * (-(∫ ω,
        if τ i = 1 then 0 else Real.log (stay ω i / (1 - 1 / τ i)) ∂μ)) := by
  have hscore : ∀ e, MeasureTheory.Integrable (fun ω => physicalRingScore τ gbar g stay ω e) μ := by
    rintro ⟨i, b⟩
    cases b
    · exact hslog i
    · exact (hglog i).add (MeasureTheory.integrable_const _)
  have hroot : Tendsto
      (ReferencePaths.normalizedGrowth (ReferenceRing.kernel τ hτ next) μ T
        (fun ω e => physicalRingScore τ gbar g stay ω e +
          (-Real.log Rstar) * (if e = (birth, true) then 1 else 0))) atTop (nhds 0) := by
    simp only [neg_mul, ← sub_eq_add_neg]
    rw [physicalRing_growth_identification τ gbar hτ next g stay Rstar birth hgbar hR μ T hT hg hs]
    exact hFurstenbergKesten_at_root
  have h := edge_bound_normalized τ hτ next birth μ T hT
    (physicalRingScore τ gbar g stay) hscore (-Real.log Rstar) hroot
  have hint (i : ι) : (∫ ω, physicalRingScore τ gbar g stay ω (i, true) ∂μ) =
      (∫ ω, Real.log (g ω i / gbar i) ∂μ) + Real.log (gbar i * τ i) := by
    simp only [physicalRingScore, ↓reduceIte]
    rw [MeasureTheory.integral_add (hglog i) (MeasureTheory.integrable_const _)]
    simp
  simp_rw [hint] at h
  simp only [physicalRingScore, Bool.false_eq_true, ↓reduceIte] at h
  simp only [neg_add_rev, Finset.sum_add_distrib, Finset.sum_neg_distrib] at h
  simp only [Finset.sum_neg_distrib]
  linarith

#print axioms physicalRing_step_identification
#print axioms physicalRing_growth_identification
#print axioms edge_bound_from_growth


end Dobson.Coverage4ThermalFixed

/-! 2026-11-16 leanp1: the former circuit-recursion and scalar root-formula
premises are proved from the daily finite cyclic population. The statistical
input explicitly averages under the complete-cycle shift; no one-step
ergodicity substitution is made. Positive full initial support is stated. -/
