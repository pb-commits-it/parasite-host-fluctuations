import Mathlib.GroupTheory.Perm.Cycle.Basic
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.Tactic

/-! Actual daily weighted cyclic population updates. -/
namespace Dobson.FixedCycle
open scoped BigOperators

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

def weight (g : ι → ℝ) (birth : ι) (h : ℕ → ℝ) (t : ℕ) (i : ι) : ℝ :=
  g i * if i = birth then h t else 1

def step (next : Equiv.Perm ι) (g : ι → ℝ) (birth : ι) (h : ℕ → ℝ)
    (t : ℕ) : Matrix ι ι ℝ :=
  fun i j => if i = next j then weight g birth h t j else 0

def population (next : Equiv.Perm ι) (g : ι → ℝ) (birth : ι) (h : ℕ → ℝ)
    (x₀ : ι → ℝ) : ℕ → ι → ℝ
  | 0 => x₀
  | t+1 => fun i => weight g birth h t (next.symm i) *
      population next g birth h x₀ t (next.symm i)

theorem population_matrix (next : Equiv.Perm ι) (g : ι → ℝ) (birth : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (t : ℕ) :
    population next g birth h x₀ (t+1) =
      (step next g birth h t).mulVec (population next g birth h x₀ t) := by
  funext i
  simp only [population, Matrix.mulVec, dotProduct, step]
  symm
  rw [Finset.sum_eq_single (next.symm i)]
  · simp
  · intro j _ hj
    have hij : i ≠ next j := by
      intro he
      apply hj
      simpa using (congrArg next.symm he).symm
    simp [hij]
  · simp

theorem population_segment (next : Equiv.Perm ι) (g : ι → ℝ) (birth : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (i : ι) (t k : ℕ) :
    population next g birth h x₀ (t+k) ((next^(t+k)) i) =
      (∏ j ∈ Finset.range k, weight g birth h (t+j) ((next^(t+j)) i)) *
        population next g birth h x₀ t ((next^t) i) := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Nat.add_succ, population, pow_succ', Equiv.Perm.mul_apply]
    simp only [Equiv.symm_apply_apply]
    rw [ih, Finset.prod_range_succ]
    ring

noncomputable def orbitEquiv (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (i : ι) :
    Fin (Fintype.card ι) ≃ ι :=
  Equiv.ofBijective (fun j => (next^j.val) i) (by
    constructor
    · intro a b hab
      apply Fin.ext
      exact hc.injOn_pow_apply (Finset.mem_univ i)
        (by simpa using a.isLt) (by simpa using b.isLt) hab
    · intro j
      obtain ⟨k, hk, he⟩ := hc.exists_pow_eq (Finset.mem_univ i) (Finset.mem_univ j)
      exact ⟨⟨k, by simpa using hk⟩, he⟩)

theorem orbit_product (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (i : ι) (g : ι → ℝ) :
    (∏ j ∈ Finset.range (Fintype.card ι), g ((next^j) i)) = ∏ j, g j := by
  rw [← Fin.prod_univ_eq_prod_range]
  exact (orbitEquiv next hc i).prod_comp g

theorem orbit_cycle (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (i : ι) (q j : ℕ) :
    (next^(q*Fintype.card ι+j)) i = (next^j) i := by
  have hq : ∀ k : ι, (next^(q*Fintype.card ι)) k = k := by
    intro k
    apply (hc.pow_apply_eq (Finset.mem_univ k)).mpr
    simp
  rw [pow_add, Equiv.Perm.mul_apply, hq]

/-- A circuit visits every deterministic factor once and the noisy edge once. -/
theorem circuit_product (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (g : ι → ℝ) (birth i : ι)
    (h : ℕ → ℝ) (q r : ℕ) (hr : r < Fintype.card ι) (hri : (next^r) i = birth) :
    (∏ j ∈ Finset.range (Fintype.card ι),
      weight g birth h (q*Fintype.card ι+j) ((next^(q*Fintype.card ι+j)) i)) =
        (∏ j, g j) * h (q*Fintype.card ι+r) := by
  simp_rw [orbit_cycle next hc, weight]
  rw [Finset.prod_mul_distrib, orbit_product next hc i]
  congr 1
  rw [Finset.prod_eq_single r]
  · simp [hri]
  · intro j hj hjr
    have hji : (next^j) i ≠ birth := by
      intro he
      apply hjr
      exact hc.injOn_pow_apply (Finset.mem_univ i)
        (by simpa using hj) (by simpa using hr) (he.trans hri.symm)
    simp [hji]
  · simp [Finset.mem_range, hr]

/-- Derived circuit recurrence for the actual daily matrix model, including zero cohorts. -/
theorem cohort_recurrence (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (g : ι → ℝ) (birth i : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (q r : ℕ) (hr : r < Fintype.card ι)
    (hri : (next^r) i = birth) :
    population next g birth h x₀ ((q+1)*Fintype.card ι) i =
      (∏ j, g j) * h (q*Fintype.card ι+r) *
        population next g birth h x₀ (q*Fintype.card ι) i := by
  have he := population_segment next g birth h x₀ i
    (q*Fintype.card ι) (Fintype.card ι)
  rw [circuit_product next hc g birth i h q r hr hri] at he
  have hstart := orbit_cycle next hc i q 0
  have hend := orbit_cycle next hc i (q+1) 0
  simp only [Nat.add_zero, pow_zero, Equiv.Perm.one_apply] at hstart hend
  rw [← Nat.succ_mul, hend, hstart] at he
  exact he

theorem cohort_product (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (g : ι → ℝ) (birth i : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (q r : ℕ) (hr : r < Fintype.card ι)
    (hri : (next^r) i = birth) :
    population next g birth h x₀ (q*Fintype.card ι) i =
      (∏ j, g j)^q * (∏ j ∈ Finset.range q, h (j*Fintype.card ι+r)) * x₀ i := by
  induction q with
  | zero => simp [population]
  | succ q ih =>
    rw [cohort_recurrence next hc g birth i h x₀ q r hr hri, ih,
      pow_succ, Finset.prod_range_succ]
    ring

end Dobson.FixedCycle
#print axioms Dobson.FixedCycle.cohort_product
