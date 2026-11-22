import DobsonProofs.ReferencePaths

/-! Explicit reference kernel for a finite ring with staying residence times. -/
namespace Dobson.ReferenceRing
open MeasureTheory
open scoped BigOperators

variable {ι Ω : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]

noncomputable def totalTime (τ : ι → ℝ) : ℝ := ∑ i, τ i

omit [DecidableEq ι] in
lemma totalTime_pos (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i) : 0 < totalTime τ := by
  classical
  exact Finset.sum_pos (fun i _ => lt_of_lt_of_le zero_lt_one (hτ i)) Finset.univ_nonempty

/-- A label consists of the source state and whether the edge transfers (true) or stays (false). -/
noncomputable def probability (τ : ι → ℝ) (i : ι) (e : ι × Bool) : ℝ :=
  if i = e.1 then if e.2 then 1 / τ i else 1 - 1 / τ i else 0

def destination (next : Equiv.Perm ι) (e : ι × Bool) : ι :=
  if e.2 then next e.1 else e.1

omit [Nonempty ι] in
lemma outgoing_sum (τ : ι → ℝ) (i : ι) (f : ι × Bool → ℝ) :
    (∑ e, probability τ i e * f e) =
      (1 - 1 / τ i) * f (i, false) + (1 / τ i) * f (i, true) := by
  simp [probability, Fintype.sum_prod_type]
  ring

omit [Fintype ι] [Nonempty ι] in
lemma probability_nonneg (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i) (i : ι) (e : ι × Bool) :
    0 ≤ probability τ i e := by
  have hp : 0 < τ i := lt_of_lt_of_le zero_lt_one (hτ i)
  unfold probability
  split_ifs
  · exact div_nonneg zero_le_one hp.le
  · have hi : 1 / τ i ≤ 1 := (div_le_iff₀ hp).mpr (by simpa using hτ i)
    linarith
  · rfl

omit [Nonempty ι] in
lemma probability_total (τ : ι → ℝ) (i : ι) : ∑ e, probability τ i e = 1 := by
  have h := outgoing_sum τ i (fun _ => 1)
  simpa using h

omit [Nonempty ι] in
lemma stationary_identity (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i)
    (next : Equiv.Perm ι) (f : ι → ℝ) :
    (∑ i, (τ i / totalTime τ) *
      ∑ e, probability τ i e * f (destination next e)) =
        ∑ i, (τ i / totalTime τ) * f i := by
  simp_rw [outgoing_sum]
  simp only [destination, Bool.false_eq_true, ↓reduceIte]
  have hi (i : ι) : τ i / totalTime τ *
      ((1 - 1 / τ i) * f i + (1 / τ i) * f (next i)) =
      (τ i * f i - f i + f (next i)) / totalTime τ := by
    have hne : τ i ≠ 0 := (lt_of_lt_of_le zero_lt_one (hτ i)).ne'
    field_simp
  simp_rw [hi]
  rw [← Finset.sum_div]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib]
  rw [Equiv.sum_comp next f]
  simp_rw [div_mul_eq_mul_div]
  rw [← Finset.sum_div]
  ring

/-- The actual reference chain with residence times `τᵢ ≥ 1`. -/
noncomputable def kernel (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i)
    (next : Equiv.Perm ι) : ReferencePaths.Kernel ι (ι × Bool) where
  prob := probability τ
  dest := destination next
  nonneg := probability_nonneg τ hτ
  total := probability_total τ
  initial := fun i => τ i / totalTime τ
  initial_pos := fun i => div_pos (lt_of_lt_of_le zero_lt_one (hτ i)) (totalTime_pos τ hτ)
  initial_nonneg := fun i => (div_pos (lt_of_lt_of_le zero_lt_one (hτ i))
    (totalTime_pos τ hτ)).le
  initial_total := by
    rw [← Finset.sum_div]
    exact div_self (totalTime_pos τ hτ).ne'
  stationary := stationary_identity τ hτ next

/-- The exact weighted path cost: transfer visit weights are `1/T`, while stay
visit weights are `(τᵢ-1)/T`, with `T = ∑ᵢ τᵢ`. -/
theorem meanScore_formula [MeasurableSpace Ω] (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i)
    (next : Equiv.Perm ι) (μ : Measure Ω) (H : Ω → (ι × Bool) → ℝ) :
    ReferencePaths.meanScore (kernel τ hτ next) μ H =
      (∑ i, ((∫ ω, H ω (i, true) ∂μ) + (τ i - 1) * ∫ ω, H ω (i, false) ∂μ)) /
        totalTime τ := by
  unfold ReferencePaths.meanScore
  change (∑ i, τ i / totalTime τ *
    ∑ e, probability τ i e * ∫ ω, H ω e ∂μ) = _
  simp_rw [outgoing_sum]
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  have hne : τ i ≠ 0 := (lt_of_lt_of_le zero_lt_one (hτ i)).ne'
  field_simp
  ring

/-- The stationary traversal frequency of any designated transfer edge is `1/T`. -/
theorem transfer_frequency (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i)
    (next : Equiv.Perm ι) (birth : ι) :
    (∑ i, (kernel τ hτ next).initial i * (kernel τ hτ next).prob i (birth, true)) =
      1 / totalTime τ := by
  simp only [kernel, probability, ↓reduceIte]
  simp_rw [mul_ite, mul_zero]
  rw [Finset.sum_ite_eq']
  simp only [Finset.mem_univ, ↓reduceIte]
  have hne : τ birth ≠ 0 := (lt_of_lt_of_le zero_lt_one (hτ birth)).ne'
  field_simp

#print axioms kernel
#print axioms meanScore_formula
#print axioms transfer_frequency
end Dobson.ReferenceRing
