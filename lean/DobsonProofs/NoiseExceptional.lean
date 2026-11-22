import Mathlib.Topology.Algebra.Order.Floor
import Mathlib.MeasureTheory.Integral.Lebesgue.Add
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.Combinatorics.Enumerative.Composition
import Mathlib.Data.Nat.Factorial.BigOperators
import Mathlib.Analysis.Normed.Group.Bounded
import Mathlib.Tactic

/-!
# Exceptional-set reduction and subsequential factorial bounds

Zero factorial type is expressed by its equivalent eventual epsilon bounds;
no extended-real root/limsup API is hidden in that definition. The Fatou step
is proved for actual measures. Holomorphic logarithmic domination and the
finite-energy restriction supplying its hypotheses are explicit handoffs.
-/
namespace Dobson.NoiseExceptional
open Filter MeasureTheory Set
open scoped Topology BigOperators ENNReal

/-- Actual ordered positive compositions, using Mathlib's list representation. -/
theorem ordered_composition_count (N : ℕ) : Fintype.card (Composition N) = 2^(N-1) :=
  composition_card N

theorem composition_factorial_bound (N : ℕ) (j : Composition N) :
    (∏ i : Fin j.length, (j.blocksFun i).factorial) ≤ N.factorial := by
  have h := Nat.le_of_dvd (Nat.factorial_pos _)
    (Nat.prod_factorial_dvd_factorial_sum Finset.univ j.blocksFun)
  simpa only [j.sum_blocksFun] using h

/-- Finite exponential majorant after the ordered-composition expansion.
The expansion itself is an explicit formal-series interface. -/
theorem exponential_coefficient_bound (coeff ε : ℝ) (n m : ℕ) (hε : 0 ≤ ε)
    (hexpansion : |coeff| ≤ (2*ε)^m*(m.factorial:ℝ)*
      ∑ k ∈ Finset.range (m+1), (n:ℝ)^k/(k.factorial:ℝ)) :
    |coeff| ≤ Real.exp n*(2*ε)^m*(m.factorial:ℝ) := by
  apply hexpansion.trans
  have hs := Real.sum_le_exp_of_nonneg (show (0:ℝ) ≤ n by positivity) (m+1)
  have hm := mul_le_mul_of_nonneg_left hs
    (show 0 ≤ (2*ε)^m*(m.factorial:ℝ) by positivity)
  nlinarith

noncomputable def endpointTerm (c : ℝ) (N l : ℕ) : ℝ :=
  c^l * ((N-l).factorial:ℝ) / (N.factorial:ℝ)

theorem endpointTerm_pos (c : ℝ) (N l : ℕ) (hc : 0 < c) :
    0 < endpointTerm c N l := by unfold endpointTerm; positivity

/-- Successive ratio, with no restriction c≤1. -/
theorem endpointTerm_step (c : ℝ) (N l : ℕ) (hl : l < N) :
    endpointTerm c N (l+1) * (N-l:ℕ) = c * endpointTerm c N l := by
  have he : N-l=(N-(l+1))+1 := by omega
  unfold endpointTerm
  conv_rhs => rw [he, Nat.factorial_succ]
  rw [pow_succ]
  push_cast
  rw [he, Nat.cast_add, Nat.cast_one]
  ring

/-- A finite maximum cannot be at an interior index: the two neighbor
inequalities would require N-l+1≤c≤N-l. -/
theorem endpointTerm_bound (c : ℝ) (N l : ℕ) (hc : 0 < c) (hl : l ≤ N) :
    endpointTerm c N l ≤ max 1 (c^N/(N.factorial:ℝ)) := by
  obtain ⟨m,hm,hmax⟩ := (Finset.range (N+1)).exists_max_image (endpointTerm c N)
    (by simp)
  have hmN : m ≤ N := by simpa using hm
  have hends : m=0 ∨ m=N := by
    by_contra hh
    push Not at hh
    have hm0 : 0 < m := Nat.pos_of_ne_zero hh.1
    have hmN' : m < N := lt_of_le_of_ne hmN hh.2
    have hprev := hmax (m-1) (Finset.mem_range.mpr (by omega))
    have hnext := hmax (m+1) (Finset.mem_range.mpr (by omega))
    have hstep := endpointTerm_step c N m hmN'
    have hstep' := endpointTerm_step c N (m-1) (by omega)
    rw [show m-1+1=m by omega] at hstep'
    have hpos := endpointTerm_pos c N m hc
    have hd : (N-(m-1):ℕ) = (N-m)+1 := by omega
    rw [hd, Nat.cast_add, Nat.cast_one] at hstep'
    have hmul := mul_le_mul_of_nonneg_right hnext (show (0:ℝ) ≤ (N-m:ℕ) by positivity)
    have hmul' := mul_le_mul_of_nonneg_left hprev hc.le
    nlinarith
  apply (hmax l (Finset.mem_range.mpr (by omega))).trans
  rcases hends with rfl | rfl
  · simpa [endpointTerm, Nat.factorial_ne_zero] using (le_max_left (1:ℝ) (c^N/(N.factorial:ℝ)))
  · simpa [endpointTerm] using (le_max_right (1:ℝ) (c^N/(N.factorial:ℝ)))

theorem endpoint_sum_bound (c : ℝ) (N : ℕ) (hc : 0 < c) :
    (∑ l ∈ Finset.range (N+1), endpointTerm c N l) ≤
      (N+1:ℕ)*max 1 (c^N/(N.factorial:ℝ)) := by
  calc
    _ ≤ ∑ _l ∈ Finset.range (N+1), max 1 (c^N/(N.factorial:ℝ)) :=
      Finset.sum_le_sum fun l hl => endpointTerm_bound c N l hc (by simpa using hl)
    _ = _ := by simp

theorem endpoint_sum_eventual (c : ℝ) (hc : 0 < c) :
    ∀ᶠ N : ℕ in atTop, (∑ l ∈ Finset.range (N+1), endpointTerm c N l) ≤ (N+1:ℕ) := by
  have h := (FloorSemiring.tendsto_pow_div_factorial_atTop c).eventually_lt_const
    (by norm_num : (0:ℝ)<1)
  filter_upwards [h] with N hN
  simpa [max_eq_left hN.le] using endpoint_sum_bound c N hc

/-- Quantitative consequence of the Lagrange type inequality, including tau=0.
Formal inversion and the type inequality itself remain an analytic input. -/
theorem diagonal_type_lower (τ Ftype : ℝ) (h : Ftype ≤ 2*Real.exp 1*τ) :
    Ftype/(2*Real.exp 1) ≤ τ := (div_le_iff₀ (by positivity)).mpr (by nlinarith)

theorem diagonal_subsequence_constants :
    0 < 1/(4*Real.exp 1) ∧ 1/(4*Real.exp 1) < 1/(2*Real.exp 1) ∧
    0 < 1/Real.exp 1 ∧ 1/Real.exp 1 < 2/Real.exp 1 := by
  have he := Real.exp_pos (1:ℝ)
  constructor
  · positivity
  constructor
  · apply div_lt_div_of_pos_left (by norm_num) (by positivity); nlinarith
  constructor
  · positivity
  · apply (div_lt_div_iff_of_pos_right he).mpr; norm_num

/-- Equivalent epsilon formulation of zero factorial type (orders n→∞). -/
def ZeroFactorialType (f : ℕ → ℝ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ N : ℕ, ∀ n ≥ N, |f n| ≤ ε^n*(n.factorial:ℝ)

def FactorialLower (f : ℕ → ℝ) (A : ℝ) : Prop :=
  ∀ N : ℕ, ∃ n ≥ N, A^n*(n.factorial:ℝ) ≤ |f n|

/-- The bound is subsequential, never asserted eventually. -/
theorem subsequential_factorial_bound (f : ℕ → ℝ) (h : ¬ZeroFactorialType f) :
    ∃ A : ℝ, 0 < A ∧ FactorialLower f A := by
  unfold ZeroFactorialType at h
  push Not at h
  obtain ⟨A,hA,h⟩ := h
  refine ⟨A,hA,?_⟩
  intro N
  obtain ⟨n,hn,hf⟩ := h N
  exact ⟨n,hn,hf.le⟩

theorem increasing_subsequence (f : ℕ → ℝ) (A : ℝ) (h : FactorialLower f A) :
    ∃ ns : ℕ → ℕ, StrictMono ns ∧ ∀ j, A^(ns j)*((ns j).factorial:ℝ) ≤ |f (ns j)| := by
  exact extraction_of_frequently_atTop (frequently_atTop.mpr h)

/-- Multiplication by a^n preserves the canonical zero-type exceptional set. -/
theorem zeroType_scale_iff (f : ℕ → ℝ) (a : ℝ) (ha : 0 < a) :
    ZeroFactorialType (fun n => a^n*f n) ↔ ZeroFactorialType f := by
  constructor
  · intro h ε hε
    obtain ⟨N,hN⟩ := h (a*ε) (mul_pos ha hε)
    refine ⟨N,fun n hn => ?_⟩
    have hh := hN n hn
    rw [abs_mul, abs_of_pos (pow_pos ha n), mul_pow] at hh
    exact (mul_le_mul_iff_right₀ (pow_pos ha n)).mp (by nlinarith [hh])
  · intro h ε hε
    obtain ⟨N,hN⟩ := h (ε/a) (div_pos hε ha)
    refine ⟨N,fun n hn => ?_⟩
    rw [abs_mul, abs_of_pos (pow_pos ha n)]
    calc
      a^n*|f n| ≤ a^n*((ε/a)^n*(n.factorial:ℝ)) :=
        mul_le_mul_of_nonneg_left (hN n hn) (pow_pos ha n).le
      _ = ε^n*(n.factorial:ℝ) := by rw [div_pow]; field_simp

/-- A factorial lower subsequence violates every geometric coefficient bound. -/
theorem factorial_obstruction (f : ℕ → ℝ) (A C R : ℝ) (hA : 0 < A)
    (_hC : 0 < C) (_hR : 0 < R) (hf : FactorialLower f A) :
    ∃ n, C*R^n < |f n| := by
  have ht := (FloorSemiring.tendsto_pow_div_factorial_atTop (R/A)).const_mul C
  have he : ∀ᶠ n : ℕ in atTop, C*((R/A)^n/(n.factorial:ℝ)) < 1 :=
    ht.eventually_lt_const (by simp)
  obtain ⟨N,hN⟩ := eventually_atTop.mp he
  obtain ⟨n,hn,hf⟩ := hf N
  have hh := hN n hn
  have ha : 0 < A^n := pow_pos hA n
  have hfact : (0:ℝ) < n.factorial := by positivity
  have hsmall : C*R^n < A^n*(n.factorial:ℝ) := by
    rw [div_pow] at hh
    have hid : C*(R^n/A^n/(n.factorial:ℝ)) = (C*R^n)/(A^n*(n.factorial:ℝ)) := by ring
    rw [hid, div_lt_one (mul_pos ha hfact)] at hh
    exact hh
  exact ⟨n,hsmall.trans_le hf⟩

/-- The ordinary power series diverges at every nonzero real argument. -/
theorem factorial_series_not_summable (f : ℕ → ℝ) (A z : ℝ) (hA : 0 < A)
    (hz : z ≠ 0) (hf : FactorialLower f A) : ¬Summable (fun n => f n*z^n) := by
  intro hs
  obtain ⟨C,hC⟩ := hs.tendsto_atTop_zero.cauchySeq.isBounded_range.exists_norm_le
  have hC0 : 0 ≤ C := (norm_nonneg (f 0*z^0)).trans (hC _ (Set.mem_range_self 0))
  have hz0 : 0 < |z| := abs_pos.mpr hz
  obtain ⟨n,hn⟩ := factorial_obstruction f A (C+1) (1/|z|) hA (by linarith)
    (by positivity) hf
  have hb := hC _ (Set.mem_range_self n)
  rw [Real.norm_eq_abs, abs_mul, abs_pow] at hb
  have hh : |f n| ≤ (C+1)*(1/|z|)^n := by
    rw [one_div, inv_pow, ← div_eq_mul_inv]
    exact (le_div_iff₀ (pow_pos hz0 n)).mpr (by linarith)
  exact (not_lt_of_ge hh) hn

/-- Fatou supplies the contradiction on a probability measure. Log singularities
are allowed to be +infinity; C must be finite. -/
theorem fatou_exceptional_contradiction {α : Type*} [MeasurableSpace α]
    (μ : Measure α) [IsProbabilityMeasure μ] (f : ℕ → α → ℝ≥0∞) (C : ℝ≥0∞)
    (hC : C ≠ ∞) (hmeas : ∀ n, AEMeasurable (f n) μ)
    (hbound : ∀ n, ∫⁻ x, f n x ∂μ ≤ C)
    (hblow : ∀ᵐ x ∂μ, liminf (fun n => f n x) atTop = ∞) : False := by
  have hfatou := lintegral_liminf_le' (u := (atTop : Filter ℕ)) hmeas
  have hleft : (∫⁻ x, liminf (fun n => f n x) atTop ∂μ) = ∞ := by
    rw [lintegral_congr_ae hblow]
    simp
  have hright : liminf (fun n => ∫⁻ x, f n x ∂μ) atTop ≤ C :=
    liminf_le_of_frequently_le' (Eventually.frequently (Eventually.of_forall hbound))
  rw [hleft] at hfatou
  exact hC (top_le_iff.mp (hfatou.trans hright))

/-- Exceptional-set reduction, phrased as absence of a finite-energy probability.
`hdom` is logarithmic integral domination after measure restriction and disk
normalization. `hblow` is the analytic normalization implication of zero type.
They are separate, explicit hypotheses; capacity is not inferred from samples. -/
theorem exceptional_set_reduction {α : Type*} [MeasurableSpace α]
    (g : α → ℕ → ℝ) (finiteEnergy : Measure α → Prop)
    (tests : Measure α → ℕ → α → ℝ≥0∞)
    (hdom : ∀ μ : Measure α, IsProbabilityMeasure μ → finiteEnergy μ →
      ∃ C : ℝ≥0∞, C ≠ ∞ ∧ (∀ n, AEMeasurable (tests μ n) μ) ∧
        ∀ n, ∫⁻ x, tests μ n x ∂μ ≤ C)
    (hblow : ∀ μ : Measure α, IsProbabilityMeasure μ → finiteEnergy μ →
      (∀ᵐ x ∂μ, ZeroFactorialType (g x)) →
        ∀ᵐ x ∂μ, liminf (fun n => tests μ n x) atTop = ∞) :
    ¬∃ μ : Measure α, IsProbabilityMeasure μ ∧ finiteEnergy μ ∧
      ∀ᵐ x ∂μ, ZeroFactorialType (g x) := by
  rintro ⟨μ,hprob,henergy,hzero⟩
  let : IsProbabilityMeasure μ := hprob
  obtain ⟨C,hC,hmeas,hbound⟩ := hdom μ hprob henergy
  exact fatou_exceptional_contradiction μ (tests μ) C hC hmeas hbound
    (hblow μ hprob henergy hzero)

/-- The finite-energy restriction is a separate handoff: the measure with a
bounded potential may differ from the original finite-energy measure. Only
carrying the zero set a.e. is preserved; no support-containment assumption occurs. -/
theorem finite_energy_restriction_reduction {α : Type*} [MeasurableSpace α]
    (g : α → ℕ → ℝ) (finiteEnergy boundedPotential : Measure α → Prop)
    (tests : Measure α → ℕ → α → ℝ≥0∞)
    (hrestrict : ∀ μ : Measure α, IsProbabilityMeasure μ → finiteEnergy μ →
      (∀ᵐ x ∂μ, ZeroFactorialType (g x)) →
      ∃ ν : Measure α, IsProbabilityMeasure ν ∧ boundedPotential ν ∧
        ∀ᵐ x ∂ν, ZeroFactorialType (g x))
    (hdom : ∀ ν : Measure α, IsProbabilityMeasure ν → boundedPotential ν →
      ∃ C : ℝ≥0∞, C ≠ ∞ ∧ (∀ n, AEMeasurable (tests ν n) ν) ∧
        ∀ n, ∫⁻ x, tests ν n x ∂ν ≤ C)
    (hblow : ∀ ν : Measure α, IsProbabilityMeasure ν → boundedPotential ν →
      (∀ᵐ x ∂ν, ZeroFactorialType (g x)) →
        ∀ᵐ x ∂ν, liminf (fun n => tests ν n x) atTop = ∞) :
    ¬∃ μ : Measure α, IsProbabilityMeasure μ ∧ finiteEnergy μ ∧
      ∀ᵐ x ∂μ, ZeroFactorialType (g x) := by
  rintro ⟨μ,hprob,henergy,hzero⟩
  exact exceptional_set_reduction g boundedPotential tests hdom hblow
    (hrestrict μ hprob henergy hzero)

/-- Outside the canonical zero-type set, scaling back by a^n gives the actual
subsequential factorial inequality. The rate may depend on the chosen point. -/
theorem outside_exceptional_subsequence (g : ℕ → ℝ) (a : ℝ) (ha : 0 < a)
    (h : ¬ZeroFactorialType g) :
    ∃ A : ℝ, 0 < A ∧ ∃ ns : ℕ → ℕ, StrictMono ns ∧
      ∀ j, A^(ns j)*((ns j).factorial:ℝ) ≤ |a^(ns j)*g (ns j)| := by
  have hz : ¬ZeroFactorialType (fun n => a^n*g n) :=
    fun hh => h ((zeroType_scale_iff g a ha).mp hh)
  obtain ⟨A,hA,hf⟩ := subsequential_factorial_bound _ hz
  obtain ⟨ns,hns,hbound⟩ := increasing_subsequence _ A hf
  exact ⟨A,hA,ns,hns,hbound⟩

/-- Nearest-point potential estimate's geometric core. -/
theorem nearest_point_distance (x y z : ℝ) (hnear : |x-z| ≤ |y-z|) :
    |x-y| ≤ 2*|z-y| := by
  have htri : |x-y| ≤ |x-z|+|z-y| := by
    simpa [sub_add_sub_cancel] using abs_add_le (x-z) (z-y)
  rw [abs_sub_comm y z] at hnear
  linarith

/-- Abstract limitations: a prescribed interior point can vanish without
changing the normalized diagonal. This is not a gamma-model counterexample. -/
theorem prescribed_zero_and_diagonal (g : ℕ → ℝ → ℝ) (a : ℝ) (ha : a ≠ 0) (n : ℕ) :
    (1-(0:ℝ)/a)*g n 0 = g n 0 ∧ (1-a/a)*g n a = 0 := by simp [ha]

/-- A second abstract family may have infinitely many exact zero coefficients. -/
theorem odd_family_values (b : ℕ → ℝ) (a : ℝ) (ha : a ≠ 0) (n : ℕ) :
    (b n*(1-(0:ℝ)/a)+(0:ℝ)/a*(n.factorial:ℝ)*(if Odd n then 1 else 0)) = b n ∧
    a^n*(b n*(1-a/a)+a/a*(n.factorial:ℝ)*(if Odd n then 1 else 0)) =
      a^n*(n.factorial:ℝ)*(if Odd n then 1 else 0) := by simp [ha]

end Dobson.NoiseExceptional

#print axioms Dobson.NoiseExceptional.ordered_composition_count
#print axioms Dobson.NoiseExceptional.composition_factorial_bound
#print axioms Dobson.NoiseExceptional.exponential_coefficient_bound
#print axioms Dobson.NoiseExceptional.endpointTerm_pos
#print axioms Dobson.NoiseExceptional.endpointTerm_step
#print axioms Dobson.NoiseExceptional.endpointTerm_bound
#print axioms Dobson.NoiseExceptional.endpoint_sum_bound
#print axioms Dobson.NoiseExceptional.endpoint_sum_eventual
#print axioms Dobson.NoiseExceptional.diagonal_type_lower
#print axioms Dobson.NoiseExceptional.diagonal_subsequence_constants
#print axioms Dobson.NoiseExceptional.subsequential_factorial_bound
#print axioms Dobson.NoiseExceptional.increasing_subsequence
#print axioms Dobson.NoiseExceptional.zeroType_scale_iff
#print axioms Dobson.NoiseExceptional.factorial_obstruction
#print axioms Dobson.NoiseExceptional.factorial_series_not_summable
#print axioms Dobson.NoiseExceptional.fatou_exceptional_contradiction
#print axioms Dobson.NoiseExceptional.exceptional_set_reduction
#print axioms Dobson.NoiseExceptional.finite_energy_restriction_reduction
#print axioms Dobson.NoiseExceptional.outside_exceptional_subsequence
#print axioms Dobson.NoiseExceptional.nearest_point_distance
#print axioms Dobson.NoiseExceptional.prescribed_zero_and_diagonal
#print axioms Dobson.NoiseExceptional.odd_family_values
