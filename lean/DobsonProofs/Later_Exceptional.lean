import DobsonProofs.NoiseExceptional
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Topology.Instances.ENNReal.Lemmas

/-! The logarithmic blow-up step for the faithful finite-energy restriction
argument. Zeros have infinite test value; Real.log 0 is never used as a
substitute for that singularity. -/
namespace Dobson.LaterExceptional
open Filter MeasureTheory Set
open scoped Topology ENNReal

noncomputable def logarithmicTest (g : ℕ → ℝ) (M A : ℝ) (n : ℕ) : ℝ≥0∞ :=
  if g n = 0 then ∞ else
    ENNReal.ofReal (Real.log (M*A^n*(n.factorial:ℝ)/|g n|)/(n:ℝ))

/-- Canonical zero factorial type implies divergence of the actual normalized
logarithmic tests, including sequences with infinitely many exact zeros. -/
theorem zeroType_logarithmic_blowup (g : ℕ → ℝ) (M A : ℝ)
    (hM : 1 ≤ M) (hA : 0 < A) (hg : NoiseExceptional.ZeroFactorialType g) :
    Tendsto (logarithmicTest g M A) atTop (𝓝 ∞) := by
  apply ENNReal.tendsto_nhds_top
  intro k
  let b : ℝ := (k:ℝ)+1
  obtain ⟨N,hN⟩ := hg (A*Real.exp (-b)) (mul_pos hA (Real.exp_pos _))
  filter_upwards [eventually_ge_atTop (max N 1)] with n hn
  have hnN : N ≤ n := (le_max_left _ _).trans hn
  have hn1 : 1 ≤ n := (le_max_right _ _).trans hn
  have hn0 : (0:ℝ) < n := by exact_mod_cast (by omega : 0<n)
  by_cases hz : g n=0
  · simp [logarithmicTest, hz]
  have hg0 : 0 < |g n| := abs_pos.mpr hz
  have hf0 : (0:ℝ) < n.factorial := by positivity
  have hbound := hN n hnN
  have hcancel : Real.exp (b*(n:ℝ))*(A*Real.exp (-b))^n = A^n := by
    rw [mul_pow, ← Real.exp_nat_mul]
    rw [mul_left_comm, ← Real.exp_add]
    have he : b*(n:ℝ)+(n:ℝ)*(-b)=0 := by ring
    rw [he, Real.exp_zero, mul_one]
  have hex : Real.exp (b*(n:ℝ)) ≤ M*A^n*(n.factorial:ℝ)/|g n| := by
    apply (le_div_iff₀ hg0).mpr
    calc
      _ ≤ Real.exp (b*(n:ℝ))*((A*Real.exp (-b))^n*(n.factorial:ℝ)) :=
        mul_le_mul_of_nonneg_left hbound (Real.exp_pos _).le
      _ = A^n*(n.factorial:ℝ) := by rw [← mul_assoc, hcancel]
      _ ≤ M*A^n*(n.factorial:ℝ) := by nlinarith [mul_pos (pow_pos hA n) hf0]
  have hlog := Real.log_le_log (Real.exp_pos (b*(n:ℝ))) hex
  rw [Real.log_exp] at hlog
  have hb : b ≤ Real.log (M*A^n*(n.factorial:ℝ)/|g n|)/(n:ℝ) :=
    (le_div_iff₀ hn0).mpr hlog
  rw [logarithmicTest, ite_eq_right hz]
  have hk : (k:ℝ) < Real.log (M*A^n*(n.factorial:ℝ)/|g n|)/(n:ℝ) := by
    dsimp [b] at hb
    linarith
  exact_mod_cast ENNReal.ofReal_lt_ofReal_iff (by linarith [Nat.cast_nonneg (α := ℝ) k]) |>.mpr hk

/-- The same assertion along any increasing coefficient subsequence. -/
theorem zeroType_subsequence_liminf (g : ℕ → ℝ) (M A : ℝ) (ns : ℕ → ℕ)
    (hM : 1 ≤ M) (hA : 0 < A) (hg : NoiseExceptional.ZeroFactorialType g)
    (hns : StrictMono ns) :
    liminf (fun j => logarithmicTest g M A (ns j)) atTop = ∞ :=
  ((zeroType_logarithmic_blowup g M A hM hA hg).comp hns.tendsto_atTop).liminf_eq

/-- The actual zero-type exceptional set is Borel when each coefficient is
continuous. This supplies the measurable-set prerequisite of capacity arguments. -/
theorem measurable_zeroType_set {α : Type*} [TopologicalSpace α]
    [MeasurableSpace α] [BorelSpace α] (g : α → ℕ → ℝ)
    (hg : ∀ n, Continuous (fun x => g x n)) :
    MeasurableSet {x | NoiseExceptional.ZeroFactorialType (g x)} := by
  have he : {x | NoiseExceptional.ZeroFactorialType (g x)} =
      ⋂ k : ℕ, ⋃ N : ℕ, ⋂ n : ℕ, {x | N ≤ n → |g x n| ≤ (1/((k:ℝ)+1))^n*(n.factorial:ℝ)} := by
    ext x
    simp only [mem_ofPred_eq, mem_iInter, mem_iUnion]
    constructor
    · intro h k
      exact h (1/((k:ℝ)+1)) (by positivity)
    · intro h ε hε
      obtain ⟨k,hk⟩ := exists_nat_gt (1/ε)
      obtain ⟨N,hN⟩ := h k
      refine ⟨N,fun n hn => (hN n hn).trans ?_⟩
      have he : 1/((k:ℝ)+1) ≤ ε := by
        apply (div_le_iff₀ (by positivity : (0:ℝ)<(k:ℝ)+1)).mpr
        have hh := (div_lt_iff₀ hε).mp hk
        nlinarith
      exact mul_le_mul_of_nonneg_right (pow_le_pow_left₀ (by positivity) he n) (by positivity)
  rw [he]
  apply MeasurableSet.iInter
  intro k
  apply MeasurableSet.iUnion
  intro N
  apply MeasurableSet.iInter
  intro n
  by_cases hn : N ≤ n
  · simpa [hn] using (isClosed_le (hg n).abs continuous_const).measurableSet
  · simp [hn]

/-- The faithful reduction with its blow-up premise discharged. Restriction
and domination remain plainly named project-specific obligations; no claim
of capacity zero is made without them. -/
theorem finite_energy_reduction {α : Type*} [MeasurableSpace α]
    (g : α → ℕ → ℝ) (finiteEnergy boundedPotential : Measure α → Prop)
    (M A : ℝ) (ns : ℕ → ℕ) (hM : 1 ≤ M) (hA : 0 < A) (hns : StrictMono ns)
    (hrestrict : ∀ μ : Measure α, IsProbabilityMeasure μ → finiteEnergy μ →
      (∀ᵐ x ∂μ, NoiseExceptional.ZeroFactorialType (g x)) →
      ∃ ν : Measure α, IsProbabilityMeasure ν ∧ boundedPotential ν ∧
        ∀ᵐ x ∂ν, NoiseExceptional.ZeroFactorialType (g x))
    (hdom : ∀ ν : Measure α, IsProbabilityMeasure ν → boundedPotential ν →
      ∃ C : ℝ≥0∞, C ≠ ∞ ∧
        (∀ j, AEMeasurable (fun x => logarithmicTest (g x) M A (ns j)) ν) ∧
        ∀ j, ∫⁻ x, logarithmicTest (g x) M A (ns j) ∂ν ≤ C) :
    ¬∃ μ : Measure α, IsProbabilityMeasure μ ∧ finiteEnergy μ ∧
      ∀ᵐ x ∂μ, NoiseExceptional.ZeroFactorialType (g x) := by
  rintro ⟨μ,hprob,henergy,hzero⟩
  obtain ⟨ν,hν,hpot,hz⟩ := hrestrict μ hprob henergy hzero
  let : IsProbabilityMeasure ν := hν
  obtain ⟨C,hC,hm,hb⟩ := hdom ν hν hpot
  apply NoiseExceptional.fatou_exceptional_contradiction ν
    (fun j x => logarithmicTest (g x) M A (ns j)) C hC hm hb
  filter_upwards [hz] with x hx
  exact zeroType_subsequence_liminf (g x) M A ns hM hA hx hns

end Dobson.LaterExceptional

#print axioms Dobson.LaterExceptional.zeroType_logarithmic_blowup
#print axioms Dobson.LaterExceptional.measurable_zeroType_set
#print axioms Dobson.LaterExceptional.finite_energy_reduction
