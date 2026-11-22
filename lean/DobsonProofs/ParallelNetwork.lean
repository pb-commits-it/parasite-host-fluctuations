import DobsonProofs.Coverage6Remainders
namespace Dobson.FiniteLineageVariance
open scoped BigOperators
variable {S : Type*} [Fintype S] [DecidableEq S]

/-- Consecutive-state edges of a stationary finite Markov chain. Impossible edges
remain in the type with zero stationary mass. -/
noncomputable def Chain.edges (K : Chain S) : Chain (S × S) where
  prob := fun e e' => if e'.1 = e.2 then K.prob e.2 e'.2 else 0
  nonneg := by intro e e'; split_ifs <;> simp_all [K.nonneg]
  total := by
    intro e
    simp [Fintype.sum_prod_type, K.total]
  initial := fun e => K.initial e.1 * K.prob e.1 e.2
  initial_nonneg := fun e => mul_nonneg (K.initial_nonneg e.1) (K.nonneg e.1 e.2)
  initial_total := by
    simp only [Fintype.sum_prod_type, ← Finset.mul_sum, K.total, mul_one, K.initial_total]
  stationary := by
    funext e
    change (∑ e' : S × S, (K.initial e'.1*K.prob e'.1 e'.2) *
      (if e.1=e'.2 then K.prob e'.2 e.2 else 0)) = K.initial e.1*K.prob e.1 e.2
    simp only [Fintype.sum_prod_type]
    have hsum : ∀ i, (∑ j, (K.initial i*K.prob i j) *
        (if e.1=j then K.prob j e.2 else 0)) =
          (K.initial i*K.prob i e.1)*K.prob e.1 e.2 := by
      intro i
      simp [mul_ite]
    simp_rw [hsum]
    rw [← Finset.sum_mul]
    have hstat := congrFun K.stationary e.1
    change (∑ i, K.initial i*K.prob i e.1) = K.initial e.1 at hstat
    rw [hstat]

end Dobson.FiniteLineageVariance
#print axioms Dobson.FiniteLineageVariance.Chain.edges

namespace Dobson.ParallelReference
open scoped BigOperators

variable {J : Type*} [Fintype J] [DecidableEq J]

noncomputable def totalTime (τ0 : ℝ) (τ w : J → ℝ) : ℝ := τ0 + ∑ j, w j * τ j

noncomputable def transition (τ0 : ℝ) (τ w : J → ℝ) : Option J → Option J → ℝ
  | none, none => 1 - 1 / τ0
  | none, some j => w j / τ0
  | some i, none => 1 / τ i
  | some i, some j => if i = j then 1 - 1 / τ i else 0

noncomputable def stationaryMass (τ0 : ℝ) (τ w : J → ℝ) : Option J → ℝ
  | none => τ0 / totalTime τ0 τ w
  | some j => w j * τ j / totalTime τ0 τ w

lemma totalTime_pos (τ0 : ℝ) (τ w : J → ℝ) (hτ0 : 1 ≤ τ0)
    (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j) : 0 < totalTime τ0 τ w := by
  have hh : 0 ≤ ∑ j, w j * τ j := Finset.sum_nonneg (fun j _ => mul_nonneg (hw j) (by linarith [hτ j]))
  unfold totalTime
  linarith

lemma transition_nonneg (τ0 : ℝ) (τ w : J → ℝ) (hτ0 : 1 ≤ τ0)
    (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j) (i j : Option J) :
    0 ≤ transition τ0 τ w i j := by
  have h0 : 0 < τ0 := by linarith
  cases i with
  | none =>
    cases j with
    | none => exact sub_nonneg.mpr ((div_le_one h0).mpr hτ0)
    | some j => exact div_nonneg (hw j) h0.le
  | some i =>
    have hi : 0 < τ i := by linarith [hτ i]
    cases j with
    | none => exact one_div_nonneg.mpr hi.le
    | some j =>
      dsimp [transition]
      split_ifs
      · exact sub_nonneg.mpr ((div_le_one hi).mpr (hτ i))
      · exact le_rfl

lemma transition_total (τ0 : ℝ) (τ w : J → ℝ) (hw : ∑ j, w j = 1) (i : Option J) :
    ∑ j, transition τ0 τ w i j = 1 := by
  cases i with
  | none => simp [Fintype.sum_option, transition, ← Finset.sum_div, hw]
  | some i => simp [Fintype.sum_option, transition]

lemma stationaryMass_nonneg (τ0 : ℝ) (τ w : J → ℝ) (hτ0 : 1 ≤ τ0)
    (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j) (i : Option J) :
    0 ≤ stationaryMass τ0 τ w i := by
  have hT := (totalTime_pos τ0 τ w hτ0 hτ hw).le
  cases i with
  | none => exact div_nonneg (by linarith) hT
  | some i => exact div_nonneg (mul_nonneg (hw i) (by linarith [hτ i])) hT

lemma stationaryMass_total (τ0 : ℝ) (τ w : J → ℝ) (hτ0 : 1 ≤ τ0)
    (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j) :
    ∑ i, stationaryMass τ0 τ w i = 1 := by
  have hT := (totalTime_pos τ0 τ w hτ0 hτ hw).ne'
  simp only [Fintype.sum_option, stationaryMass, ← Finset.sum_div, ← add_div]
  exact div_self hT

lemma entrance_frequency (τ0 : ℝ) (τ w : J → ℝ) (hτ0 : 1 ≤ τ0) (j : J) :
    stationaryMass τ0 τ w none * transition τ0 τ w none (some j) =
      w j / totalTime τ0 τ w := by
  have h0 : τ0 ≠ 0 := by linarith
  dsimp [stationaryMass, transition]
  field_simp

lemma return_frequency (τ0 : ℝ) (τ w : J → ℝ) (hτ : ∀ j, 1 ≤ τ j) (j : J) :
    stationaryMass τ0 τ w (some j) * transition τ0 τ w (some j) none =
      w j / totalTime τ0 τ w := by
  have hj : τ j ≠ 0 := by linarith [hτ j]
  dsimp [stationaryMass, transition]
  field_simp

lemma stationaryMass_stationary (τ0 : ℝ) (τ w : J → ℝ) (hτ0 : 1 ≤ τ0)
    (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1) :
    Matrix.vecMul (stationaryMass τ0 τ w) (transition τ0 τ w) = stationaryMass τ0 τ w := by
  have hT := (totalTime_pos τ0 τ w hτ0 hτ hw).ne'
  have h0 : τ0 ≠ 0 := by linarith
  funext j
  change (∑ i, stationaryMass τ0 τ w i * transition τ0 τ w i j) = _
  rw [Fintype.sum_option]
  cases j with
  | none =>
    simp_rw [return_frequency τ0 τ w hτ]
    rw [← Finset.sum_div, hwsum]
    dsimp [stationaryMass, transition]
    field_simp
    ring
  | some j =>
    rw [entrance_frequency τ0 τ w hτ0 j]
    have he : (∑ i, stationaryMass τ0 τ w (some i) * transition τ0 τ w (some i) (some j)) =
        w j * τ j / totalTime τ0 τ w * (1 - 1 / τ j) := by
      simp [stationaryMass, transition]
    rw [he]
    have hj : τ j ≠ 0 := by linarith [hτ j]
    dsimp [stationaryMass]
    field_simp
    ring

/-- The actual threshold parallel-network reference chain. Zero-share hosts
are allowed; their stationary mass is zero. -/
noncomputable def stateChain (τ0 : ℝ) (τ w : J → ℝ) (hτ0 : 1 ≤ τ0)
    (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1) :
    FiniteLineageVariance.Chain (Option J) where
  prob := transition τ0 τ w
  nonneg := transition_nonneg τ0 τ w hτ0 hτ hw
  total := transition_total τ0 τ w hwsum
  initial := stationaryMass τ0 τ w
  initial_nonneg := stationaryMass_nonneg τ0 τ w hτ0 hτ hw
  initial_total := stationaryMass_total τ0 τ w hτ0 hτ hw
  stationary := stationaryMass_stationary τ0 τ w hτ0 hτ hw hwsum

end Dobson.ParallelReference
namespace Dobson.FinitePathCoefficient
open MeasureTheory Dobson.FiniteLineageVariance
open scoped BigOperators
variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

/-- With no temporal covariance, the quadratic coefficient is already constant
at every finite horizon. No mixing assertion or interchange of derivatives and
an infinite-time limit is needed. -/
lemma finiteCoefficient_of_zero_lags (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (hlag : ∀ e e' m, lagCovariance μ T f e e' (m+1) = 0) (N : ℕ) :
    finiteCoefficient μ K T f N =
      (∑ e, ∑ e', K.initial e*K.initial e'*lagCovariance μ T f e e' 0)/2 := by
  have hn : (N+1 : ℝ) ≠ 0 := by positivity
  have hz : ∀ m, varianceLag μ K T f (m+1) = 0 := by
    intro m
    simp [varianceLag_pos, hlag]
  rw [finiteCoefficient, withinVariance_finite μ K T hT f hf]
  simp only [hz, mul_zero, Finset.sum_const_zero, add_zero, varianceLag_zero,
    Nat.cast_add, Nat.cast_one]
  field_simp
  ring

end Dobson.FinitePathCoefficient
#print axioms Dobson.FinitePathCoefficient.finiteCoefficient_of_zero_lags
namespace Dobson.ParallelReference
open MeasureTheory Filter Dobson.FiniteLineageVariance
open scoped BigOperators Topology
variable {J Ω : Type*} [Fintype J] [DecidableEq J] [MeasurableSpace Ω]

noncomputable def entranceScore (κ : J → ℝ) (ξ : J → Ω → ℝ) :
    (Option J × Option J) → Ω → ℝ
  | (none, some j), ω => κ j * ξ j ω
  | _, _ => 0

lemma entranceScore_memLp (μ : Measure Ω) (κ : J → ℝ) (ξ : J → Ω → ℝ)
    (hξ : ∀ j, MemLp (ξ j) 2 μ) (e : Option J × Option J) :
    MemLp (entranceScore κ ξ e) 2 μ := by
  rcases e with ⟨a,b⟩
  cases a <;> cases b
  · exact MemLp.zero
  · exact (hξ _).const_mul _
  · exact MemLp.zero
  · exact MemLp.zero

lemma entranceScore_covariance (μ : Measure Ω) (T : Ω → Ω) (κ : J → ℝ)
    (ξ : J → Ω → ℝ) (i j : J) (m : ℕ) :
    lagCovariance μ T (entranceScore κ ξ) (none,some i) (none,some j) m =
      κ i * κ j * lagCovariance μ T ξ i j m := by
  unfold lagCovariance entranceScore
  simp_rw [show ∀ ω, κ i * ξ i (T^[m] ω) * (κ j * ξ j ω) =
      (κ i * κ j) * (ξ i (T^[m] ω) * ξ j ω) from fun ω => by ring]
  exact integral_const_mul _ _

lemma entranceScore_zero_lags (μ : Measure Ω) (T : Ω → Ω) (κ : J → ℝ)
    (ξ : J → Ω → ℝ) (hlag : ∀ i j m, lagCovariance μ T ξ i j (m+1) = 0)
    (e e' : Option J × Option J) (m : ℕ) :
    lagCovariance μ T (entranceScore κ ξ) e e' (m+1) = 0 := by
  rcases e with ⟨a,b⟩
  rcases e' with ⟨c,d⟩
  cases a <;> cases b <;> cases c <;> cases d <;>
    try solve | simp [lagCovariance, entranceScore]
  rw [entranceScore_covariance, hlag, mul_zero]

/-- The arbitrary contemporaneous covariance quadratic form of the loaded route
shares. No independence between routes is assumed. -/
noncomputable def routeVariance (μ : Measure Ω) (κ w : J → ℝ) (ξ : J → Ω → ℝ) : ℝ :=
  ∑ i, ∑ j, (w i*κ i)*(w j*κ j)*(∫ ω, ξ i ω*ξ j ω ∂μ)

lemma weighted_covariance (μ : Measure Ω) (T : Ω → Ω) (τ0 : ℝ) (τ w κ : J → ℝ)
    (ξ : J → Ω → ℝ) (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j)
    (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1) :
    (∑ e, ∑ e', ((stateChain τ0 τ w hτ0 hτ hw hwsum).edges).initial e *
      ((stateChain τ0 τ w hτ0 hτ hw hwsum).edges).initial e' *
        lagCovariance μ T (entranceScore κ ξ) e e' 0) =
      routeVariance μ κ w ξ / totalTime τ0 τ w ^ 2 := by
  simp only [Chain.edges, Fintype.sum_prod_type, Fintype.sum_option,
    lagCovariance, entranceScore, Function.iterate_zero, id_eq, zero_mul, mul_zero,
    integral_zero, Finset.sum_const_zero, zero_add, add_zero]
  change (∑ i, ∑ j, (stationaryMass τ0 τ w none * transition τ0 τ w none (some i)) *
    (stationaryMass τ0 τ w none * transition τ0 τ w none (some j)) *
      (∫ ω, (κ i*ξ i ω)*(κ j*ξ j ω) ∂μ)) = _
  simp_rw [entrance_frequency τ0 τ w hτ0]
  unfold routeVariance
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro j hj
  have hc : (∫ ω, (κ i*ξ i ω)*(κ j*ξ j ω) ∂μ) =
      κ i*κ j*(∫ ω, ξ i ω*ξ j ω ∂μ) := by
    simp_rw [show ∀ ω, (κ i*ξ i ω)*(κ j*ξ j ω) = (κ i*κ j)*(ξ i ω*ξ j ω)
      from fun ω => by ring]
    exact integral_const_mul _ _
  rw [hc]
  ring

lemma birth_frequency (τ0 : ℝ) (τ w : J → ℝ) (hτ : ∀ j, 1 ≤ τ j)
    (hwsum : ∑ j, w j = 1) :
    (∑ j, stationaryMass τ0 τ w (some j)*transition τ0 τ w (some j) none) =
      1/totalTime τ0 τ w := by
  simp_rw [return_frequency τ0 τ w hτ]
  rw [← Finset.sum_div, hwsum]

end Dobson.ParallelReference
namespace Dobson.ParallelReference
open MeasureTheory Filter Dobson.FiniteLineageVariance
open scoped BigOperators Topology
variable {J Ω : Type*} [Fintype J] [DecidableEq J] [MeasurableSpace Ω]

lemma finiteCoefficient_eq (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (τ0 : ℝ) (τ w κ : J → ℝ)
    (ξ : J → Ω → ℝ) (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j)
    (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1)
    (hξ : ∀ j, MemLp (ξ j) 2 μ)
    (hlag : ∀ i j m, lagCovariance μ T ξ i j (m+1) = 0) (N : ℕ) :
    FinitePathCoefficient.finiteCoefficient μ
      ((stateChain τ0 τ w hτ0 hτ hw hwsum).edges) T (entranceScore κ ξ) N =
        routeVariance μ κ w ξ / (2*totalTime τ0 τ w^2) := by
  rw [FinitePathCoefficient.finiteCoefficient_of_zero_lags μ _ T hT _
    (entranceScore_memLp μ κ ξ hξ) (entranceScore_zero_lags μ T κ ξ hlag)]
  rw [weighted_covariance]
  ring

lemma finiteGrowth_second_eq (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (τ0 : ℝ) (τ w κ : J → ℝ)
    (ξ : J → Ω → ℝ) (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j)
    (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1)
    (hξ : ∀ j, MemLp (ξ j) 2 μ)
    (hlag : ∀ i j m, lagCovariance μ T ξ i j (m+1) = 0) (N : ℕ) :
    iteratedDeriv 2 (FinitePathJet.finiteGrowth μ
      ((stateChain τ0 τ w hτ0 hτ hw hwsum).edges) T (entranceScore κ ξ) 0 .lognormal N) 0 =
        -(routeVariance μ κ w ξ / totalTime τ0 τ w^2) := by
  rw [FinitePathJet.finiteGrowth_second μ _ T hT _
    (entranceScore_memLp μ κ ξ hξ) 0 .lognormal (by intro h; cases h)]
  rw [finiteCoefficient_eq μ T hT τ0 τ w κ ξ hτ0 hτ hw hwsum hξ hlag]
  ring

/-- The local quadratic cost is defined through the limit of actual finite
expected growth second derivatives. Identifying this with a Lyapunov expansion
is deliberately not asserted here (owed62); the threshold map remains owed37. -/
noncomputable def quadraticCost (μ : Measure Ω) (τ0 : ℝ) (τ w κ : J → ℝ)
    (ξ : J → Ω → ℝ) (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j)
    (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1) (T : Ω → Ω) (σ : ℝ) : ℝ :=
  -(σ^2/2) * Filter.limUnder atTop (fun N : ℕ =>
    iteratedDeriv 2 (FinitePathJet.finiteGrowth μ
      ((stateChain τ0 τ w hτ0 hτ hw hwsum).edges) T (entranceScore κ ξ) 0 .lognormal N) 0)

lemma quadraticCost_eq (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (τ0 : ℝ) (τ w κ : J → ℝ)
    (ξ : J → Ω → ℝ) (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j)
    (hw : ∀ j, 0 ≤ w j) (hwsum : ∑ j, w j = 1)
    (hξ : ∀ j, MemLp (ξ j) 2 μ)
    (hlag : ∀ i j m, lagCovariance μ T ξ i j (m+1) = 0) (σ : ℝ) :
    quadraticCost μ τ0 τ w κ ξ hτ0 hτ hw hwsum T σ =
      σ^2*routeVariance μ κ w ξ / (2*totalTime τ0 τ w^2) := by
  unfold quadraticCost
  simp_rw [finiteGrowth_second_eq μ T hT τ0 τ w κ ξ hτ0 hτ hw hwsum hξ hlag]
  have hlim : Tendsto (fun _ : ℕ => -(routeVariance μ κ w ξ / totalTime τ0 τ w^2))
      atTop (𝓝 (-(routeVariance μ κ w ξ / totalTime τ0 τ w^2))) := tendsto_const_nhds
  rw [hlim.limUnder_eq]
  ring

end Dobson.ParallelReference

#print axioms Dobson.ParallelReference.finiteGrowth_second_eq
#print axioms Dobson.ParallelReference.quadraticCost_eq
