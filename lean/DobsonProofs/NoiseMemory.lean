import Mathlib.Algebra.Polynomial.Degree.SmallDegree
import Mathlib.Data.Nat.Factorial.BigOperators
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Analytic.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.Tactic

/-!
# Gamma refresh with memory: finite coefficient cores

The centered gamma recurrence is represented by its natural-number coefficient
array. Raw moments are actual polynomials. Walks enter through a finite span
table; identifying that table with nearest-neighbor walks is separate.
Formal inversion, Gevrey closure, analytic propagation and identification with
physical asymptotic coefficients are NOT asserted by the finite calculations.
-/
namespace Dobson.NoiseMemory
open Polynomial
open scoped BigOperators

/-- Coefficients of μ₀=1, μ₁=0, μₘ₊₂=(m+1)X(μₘ₊₁+μₘ). -/
def gammaCoeff : ℕ → ℕ → ℕ
  | 0, n => if n = 0 then 1 else 0
  | 1, _ => 0
  | _m+2, 0 => 0
  | m+2, n+1 => (m+1)*(gammaCoeff (m+1) n + gammaCoeff m n)

theorem gamma_constant (m : ℕ) : gammaCoeff m 0 = if m=0 then 1 else 0 := by
  cases m with
  | zero => rfl
  | succ m => cases m <;> simp [gammaCoeff]

/-- Support, including the exceptional order-zero constant. -/
theorem gamma_support (m n : ℕ) (h : gammaCoeff m n ≠ 0) :
    (m=0 ∧ n=0) ∨ (n+1 ≤ m ∧ m ≤ 2*n) := by
  induction m using Nat.twoStepInduction generalizing n with
  | zero => simp only [gammaCoeff] at h; left; constructor; rfl; split_ifs at h <;> simp_all
  | one => simp [gammaCoeff] at h
  | more m ih₀ ih₁ =>
    cases n with
    | zero => simp [gammaCoeff] at h
    | succ n =>
      simp only [gammaCoeff] at h
      have hn : gammaCoeff (m+1) n ≠ 0 ∨ gammaCoeff m n ≠ 0 := by
        by_contra hh
        push Not at hh
        simp [hh.1, hh.2] at h
      rcases hn with hn | hn
      · have := ih₁ n hn
        right; omega
      · have := ih₀ n hn
        right; omega

/-- The all-order factorial majorant, not a finite-order numerical check. -/
theorem gamma_factorial_bound (n m : ℕ) : gammaCoeff m n ≤ 8^n*n.factorial := by
  induction n generalizing m with
  | zero => rw [gamma_constant]; split_ifs <;> norm_num
  | succ n ih =>
    by_cases hz : gammaCoeff m (n+1)=0
    · rw [hz]; omega
    have hm := gamma_support m (n+1) hz
    cases m with
    | zero => simp [gammaCoeff]
    | succ m =>
      cases m with
      | zero => simp [gammaCoeff]
      | succ m =>
        rw [gammaCoeff, pow_succ, Nat.factorial_succ]
        have h₁ := ih (m+1)
        have h₀ := ih m
        have hmn : m+1 ≤ 2*(n+1) := by omega
        calc
          (m+1)*(gammaCoeff (m+1) n+gammaCoeff m n) ≤
              (2*(n+1))*(2*(8^n*n.factorial)) := Nat.mul_le_mul hmn (by omega)
          _ ≤ 8^n*8*((n+1)*n.factorial) := by nlinarith

theorem gamma_low_coefficients :
    gammaCoeff 2 1=1 ∧ gammaCoeff 3 2=2 ∧
    gammaCoeff 4 2=3 ∧ gammaCoeff 4 3=6 := by norm_num [gammaCoeff]

/-- H_(n+1), the raw moment with n nonconstant factors. -/
noncomputable def rawMoment : ℕ → ℝ[X]
  | 0 => 1
  | n+1 => (C ((n:ℝ)+1)*X+C 1)*rawMoment n

theorem rawMoment_leading (n : ℕ) : (rawMoment n).leadingCoeff = (n.factorial:ℝ) := by
  induction n with
  | zero => simp [rawMoment]
  | succ n ih =>
    rw [rawMoment, leadingCoeff_mul, leadingCoeff_linear (by positivity), ih, Nat.factorial_succ]
    push_cast
    ring

theorem rawMoment_degree (n : ℕ) : (rawMoment n).natDegree = n := by
  induction n with
  | zero => simp [rawMoment]
  | succ n ih =>
    have hn : rawMoment n ≠ 0 := by
      intro h; have := rawMoment_leading n; rw [h, leadingCoeff_zero] at this
      exact (Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero n)) this.symm
    have hl : (C ((n:ℝ)+1)*X+C 1 : ℝ[X]) ≠ 0 := by
      intro h
      have hh := leadingCoeff_linear (b := (1:ℝ)) (show (n:ℝ)+1 ≠ 0 by positivity)
      rw [h, leadingCoeff_zero] at hh
      have : (0:ℝ)<n+1 := by positivity
      linarith
    rw [rawMoment, natDegree_mul hl hn, natDegree_linear (by positivity), ih]
    omega

theorem rawMoment_top_coefficient (n : ℕ) : (rawMoment n).coeff n = (n.factorial:ℝ) := by
  calc
    (rawMoment n).coeff n = (rawMoment n).coeff (rawMoment n).natDegree := by rw [rawMoment_degree]
    _ = (n.factorial:ℝ) := rawMoment_leading n

/-- All cut terms with smaller degree vanish at the top coefficient. -/
theorem refreshed_top_coefficient {ι : Type*} (S : Finset ι) (P : ι → ℝ[X])
    (w : ι → ℝ) (noCut : ι) (k : ℕ) (φ : ℝ) (sites : ℕ)
    (hmem : noCut ∈ S) (hcut : ∀ i ∈ S, i ≠ noCut → (P i).natDegree < k)
    (hmain : (P noCut).coeff k = (k.factorial:ℝ))
    (hweight : w noCut = φ^sites) :
    (∑ i ∈ S, C (w i)*P i).coeff k = (k.factorial:ℝ)*φ^sites := by
  simp only [finsetSum_coeff, coeff_C_mul]
  rw [Finset.sum_eq_single_of_mem noCut hmem]
  · rw [hmain, hweight]; ring
  · intro i hi hne
    rw [coeff_eq_zero_of_natDegree_lt (hcut i hi hne), mul_zero]

/-- A finite span table, with span minus one as its natural exponent. -/
noncomputable def walkWeight {ι : Type*} (S : Finset ι) (spanMinusOne : ι → ℕ)
    (φ : ℝ) : ℝ := (∑ i ∈ S, φ^(spanMinusOne i))/2

theorem walkWeight_upper {ι : Type*} (S : Finset ι) (span : ι → ℕ)
    (φ : ℝ) (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    walkWeight S span φ ≤ (S.card:ℝ)/2 := by
  unfold walkWeight
  gcongr
  calc
    (∑ i ∈ S, φ^(span i)) ≤ ∑ _i ∈ S, (1:ℝ) :=
      Finset.sum_le_sum fun i _ => pow_le_one₀ hφ0 hφ1
    _ = (S.card:ℝ) := by simp

/-- Keeping just the two single-edge walks proves the lower bound. -/
theorem walkWeight_lower {ι : Type*} [DecidableEq ι] (S : Finset ι) (span : ι → ℕ)
    (φ : ℝ) (hφ : 0 ≤ φ) (i j : ι) (hi : i ∈ S) (hj : j ∈ S)
    (hne : i ≠ j) (hsi : span i=0) (hsj : span j=0) :
    1 ≤ walkWeight S span φ := by
  have hsub : ({i,j}:Finset ι) ⊆ S := by simp_all [Finset.insert_subset_iff]
  have hsum := Finset.sum_le_sum_of_subset_of_nonneg hsub
    (fun a _ _ => pow_nonneg hφ (span a))
  have hlo : (2:ℝ) ≤ ∑ a ∈ S, φ^(span a) := by simpa only [Finset.sum_pair hne, hsi, hsj, pow_zero, one_add_one_eq_two] using hsum
  unfold walkWeight
  linarith

theorem closed_walk_bounds {ι : Type*} [DecidableEq ι] (S : Finset ι) (span : ι → ℕ)
    (φ : ℝ) (k : ℕ) (hφ : 0 ≤ φ) (hφ1 : φ ≤ 1)
    (hcard : S.card ≤ 4^k) (i j : ι) (hi : i ∈ S) (hj : j ∈ S)
    (hne : i ≠ j) (hsi : span i=0) (hsj : span j=0) :
    1 ≤ walkWeight S span φ ∧ walkWeight S span φ ≤ (4:ℝ)^k/2 := by
  refine ⟨walkWeight_lower S span φ hφ i j hi hj hne hsi hsj, ?_⟩
  apply (walkWeight_upper S span φ hφ hφ1).trans
  gcongr
  exact_mod_cast hcard

/-- The normalized diagonal inverse coefficient; n indexes coefficient n+1. -/
noncomputable def diagonalInverseCoeff (W : ℕ → ℝ) (n : ℕ) : ℝ :=
  (-1:ℝ)^(n+2)*(n.factorial:ℝ)*W (n+1)/(n+1:ℕ)

theorem diagonal_factorial_ratio (W : ℕ → ℝ) (n : ℕ) (hW : 0 ≤ W (n+1)) :
    |diagonalInverseCoeff W n|/((n+1).factorial:ℝ) = W (n+1)/((n+1:ℕ):ℝ)^2 := by
  unfold diagonalInverseCoeff
  rw [abs_div, abs_mul, abs_mul, abs_pow]
  simp only [abs_neg, abs_one, one_pow, one_mul,
    abs_of_nonneg (show (0:ℝ) ≤ n.factorial by positivity), abs_of_nonneg hW,
    abs_of_nonneg (show (0:ℝ) ≤ (n+1:ℕ) by positivity), Nat.factorial_succ, Nat.cast_mul]
  have hf : (n.factorial:ℝ) ≠ 0 := by positivity
  have hn : ((n+1:ℕ):ℝ) ≠ 0 := by positivity
  field_simp

theorem diagonal_factorial_sandwich (W : ℕ → ℝ) (n : ℕ)
    (hlo : 1 ≤ W (n+1)) (hhi : W (n+1) ≤ (4:ℝ)^(n+1)/2) :
    1/((n+1:ℕ):ℝ)^2 ≤ |diagonalInverseCoeff W n|/((n+1).factorial:ℝ) ∧
      |diagonalInverseCoeff W n|/((n+1).factorial:ℝ) ≤
        (4:ℝ)^(n+1)/(2*((n+1:ℕ):ℝ)^2) := by
  rw [diagonal_factorial_ratio W n (by linarith)]
  constructor
  · gcongr
  · calc
      _ ≤ ((4:ℝ)^(n+1)/2)/((n+1:ℕ):ℝ)^2 := by gcongr
      _ = _ := by ring

/-- The low walk polynomials give the two displayed diagonal coefficients. -/
theorem diagonal_low_coefficients (φ : ℝ) :
    (1+2*φ)/2 = 1/2+φ ∧
    -2*(1+6*φ+3*φ^2)/3 + (3/2)*((1+2*φ)/2)^2 =
      -2*(1/3+2*φ+φ^2)+(3/2)*(1/2+φ)^2 := by constructor <;> ring

/-- Exhaustive nearest-neighbor step words; true is +1 and false is -1. -/
def stepWords : ℕ → List (List Bool)
  | 0 => [[]]
  | n+1 => (stepWords n).map (true :: ·) ++ (stepWords n).map (false :: ·)

/-- Displacement, maximum visited vertex, minimum visited vertex, including 0. -/
def wordData : List Bool → ℤ × ℤ × ℤ
  | [] => (0,0,0)
  | b :: w =>
    let d := wordData w
    let e : ℤ := if b then 1 else -1
    (e+d.1, max 0 (e+d.2.1), min 0 (e+d.2.2))

noncomputable def enumeratedWalkWeight (k : ℕ) (φ : ℝ) : ℝ :=
  (((stepWords (2*k)).filter (fun w => (wordData w).1=0)).map
    (fun w => φ^(((wordData w).2.1-(wordData w).2.2).toNat-1))).sum/2

/-- The printed W₁,W₂,W₃ are verified against actual exhaustive words. -/
theorem enumerated_low_walks (φ : ℝ) :
    enumeratedWalkWeight 1 φ=1 ∧ enumeratedWalkWeight 2 φ=1+2*φ ∧
      enumeratedWalkWeight 3 φ=1+6*φ+3*φ^2 := by
  norm_num [enumeratedWalkWeight, stepWords, wordData, List.map_append, List.filter_append]
  constructor <;> ring

/-- Placement of span-l walks in a finite path, with natural truncated lengths. -/
noncomputable def placements (c : ℕ → ℝ) (L : ℕ) : ℝ :=
  ∑ l ∈ Finset.range (L+1), c l * ((L+1-l:ℕ):ℝ)

private theorem placements_step (c : ℕ → ℝ) (L : ℕ) :
    placements c (L+1) = placements c L + ∑ l ∈ Finset.range (L+2), c l := by
  unfold placements
  rw [Finset.sum_range_succ (fun l => c l * ((L+1+1-l:ℕ):ℝ)) (L+1)]
  rw [show L+2=(L+1)+1 by omega, Finset.sum_range_succ (fun l => c l) (L+1)]
  simp only [Nat.add_sub_cancel_left, Nat.cast_one, mul_one]
  rw [← add_assoc, ← Finset.sum_add_distrib]
  congr 1
  apply Finset.sum_congr rfl
  intro l hl
  have he : L+1+1-l = (L+1-l)+1 := by have := Finset.mem_range.mp hl; omega
  rw [he, Nat.cast_add, Nat.cast_one]
  ring

/-- Exact second difference of the finite placement sum. -/
theorem placement_second_difference (c : ℕ → ℝ) (L : ℕ) :
    placements c (L+2)-2*placements c (L+1)+placements c L = c (L+2) := by
  have h₀ := placements_step c L
  have h₁ := placements_step c (L+1)
  have hs := Finset.sum_range_succ (fun l => c l) (L+2)
  simp only [show L+1+1=L+2 by omega, show L+1+2=L+2+1 by omega] at h₁
  linarith

/-- Products of component factorials are bounded by the total factorial. -/
theorem factorial_product_bound {ι : Type*} (S : Finset ι) (j : ι → ℕ) :
    (∏ i ∈ S, (j i).factorial) ≤ (∑ i ∈ S, j i).factorial :=
  Nat.le_of_dvd (Nat.factorial_pos _) (Nat.prod_factorial_dvd_factorial_sum S j)

/-- Abstract nonanalyticity reduction. The first premise is the missing
physical-asymptotic identification plus analytic coefficient bounds; the second
is the missing Gevrey/holomorphic propagation result at the chosen slice.
There is no claim that a finite walk computation proves either premise. -/
theorem nonanalyticity_of_coefficient_obstruction (analytic : Prop) (d : ℕ → ℝ)
    (hanalytic : analytic → ∃ C A : ℝ, 0 < C ∧ 0 < A ∧ ∀ n, |d n| ≤ C*A^n)
    (hgrowth : ∀ C A : ℝ, 0 < C → 0 < A → ∃ n, C*A^n < |d n|) : ¬analytic := by
  intro h
  obtain ⟨C,A,hC,hA,hbound⟩ := hanalytic h
  obtain ⟨n,hn⟩ := hgrowth C A hC hA
  exact (not_lt_of_ge (hbound n)) hn

/-- Conditional almost-everywhere physical conclusion for one fixed host law.
The analytic-propagation hypothesis is explicit, not proved by the finite cores.
There is no simultaneous exceptional-set assertion over all memory parameters. -/
theorem ae_nonanalyticity (μ : MeasureTheory.Measure ℝ) (D : ℝ → ℝ → ℝ)
    (d : ℝ → ℕ → ℝ)
    (hphysical : ∀ a, AnalyticAt ℝ (D a) 0 →
      ∃ C A : ℝ, 0 < C ∧ 0 < A ∧ ∀ n, |d a n| ≤ C*A^n)
    (hpropagation : ∀ᵐ a ∂μ, ∀ C A : ℝ, 0 < C → 0 < A →
      ∃ n, C*A^n < |d a n|) : ∀ᵐ a ∂μ, ¬AnalyticAt ℝ (D a) 0 := by
  filter_upwards [hpropagation] with a ha
  exact nonanalyticity_of_coefficient_obstruction _ (d a) (hphysical a) ha

end Dobson.NoiseMemory

#print axioms Dobson.NoiseMemory.gamma_constant
#print axioms Dobson.NoiseMemory.gamma_support
#print axioms Dobson.NoiseMemory.gamma_factorial_bound
#print axioms Dobson.NoiseMemory.gamma_low_coefficients
#print axioms Dobson.NoiseMemory.rawMoment_leading
#print axioms Dobson.NoiseMemory.rawMoment_degree
#print axioms Dobson.NoiseMemory.rawMoment_top_coefficient
#print axioms Dobson.NoiseMemory.refreshed_top_coefficient
#print axioms Dobson.NoiseMemory.walkWeight_upper
#print axioms Dobson.NoiseMemory.walkWeight_lower
#print axioms Dobson.NoiseMemory.closed_walk_bounds
#print axioms Dobson.NoiseMemory.diagonal_factorial_ratio
#print axioms Dobson.NoiseMemory.diagonal_factorial_sandwich
#print axioms Dobson.NoiseMemory.diagonal_low_coefficients
#print axioms Dobson.NoiseMemory.enumerated_low_walks
#print axioms Dobson.NoiseMemory.placement_second_difference
#print axioms Dobson.NoiseMemory.factorial_product_bound
#print axioms Dobson.NoiseMemory.nonanalyticity_of_coefficient_obstruction
#print axioms Dobson.NoiseMemory.ae_nonanalyticity
