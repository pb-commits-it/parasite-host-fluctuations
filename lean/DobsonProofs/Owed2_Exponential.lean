import Mathlib.RingTheory.PowerSeries.Exp
import Mathlib.RingTheory.PowerSeries.Substitution
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Complex.Basic
import Mathlib.Tactic

/-!
Gap3exc §3, the exponential coefficient estimate (8). The coefficients
are those of Mathlib's actual formal exponential after substitution.
The convolution argument bounds all powers at once, so no ordered-
composition expansion or coefficient inequality is a supplied premise.
-/
namespace Dobson.Owed2_Exponential
open PowerSeries Finset
open scoped BigOperators

lemma factorial_pair (j k : ℕ) :
    (j+1).factorial * (k+1).factorial ≤ (j+k+1).factorial := by
  induction j with
  | zero => simp
  | succ j ih =>
    calc
      (j+1+1).factorial * (k+1).factorial =
          (j+2)*((j+1).factorial*(k+1).factorial) := by rw [Nat.factorial_succ]; ring
      _ ≤ (j+2)*(j+k+1).factorial := Nat.mul_le_mul_left _ ih
      _ ≤ (j+k+2)*(j+k+1).factorial := Nat.mul_le_mul_right _ (by omega)
      _ = _ := by
        rw [show j+1+k+1=j+k+1+1 by omega]
        exact (Nat.factorial_succ (j+k+1)).symm

/-- Products of any number of zero-constant factorial-majorized series
retain this same majorant. The degree-zero coefficient is included. -/
theorem power_coefficient_bound (R : PowerSeries ℂ) (ε : ℝ) (hε : 0 ≤ ε)
    (hR0 : constantCoeff R = 0)
    (hR : ∀ n, ‖coeff n R‖ ≤ ε^n*(n.factorial:ℝ)) :
    ∀ k n : ℕ, ‖coeff n (R^k)‖ ≤ ε^n*(n.factorial:ℝ) := by
  intro k
  induction k with
  | zero =>
    intro n
    by_cases hn : n=0
    · subst n; simp
    · simp only [pow_zero, PowerSeries.coeff_one, ite_eq_right hn, norm_zero]
      positivity
  | succ k ih =>
    intro n
    by_cases hk : k=0
    · subst k; simpa using hR n
    cases n with
    | zero => simp [coeff_zero_eq_constantCoeff_apply, map_pow, hR0]
    | succ m =>
      rw [pow_succ, coeff_mul, Nat.sum_antidiagonal_eq_sum_range_succ_mk,
        Finset.sum_range_succ]
      simp only [Nat.sub_self, coeff_zero_eq_constantCoeff_apply, hR0, mul_zero, add_zero]
      calc
        _ ≤ ∑ j ∈ range (m+1), ‖coeff j (R^k) * coeff (m+1-j) R‖ := norm_sum_le _ _
        _ ≤ ∑ _j ∈ range (m+1), ε^(m+1)*(m.factorial:ℝ) := by
          apply sum_le_sum
          intro j hj
          by_cases hj0 : j=0
          · subst j
            simp only [coeff_zero_eq_constantCoeff_apply, map_pow, hR0, zero_pow hk,
              zero_mul, norm_zero]
            positivity
          have hjm : j ≤ m := by simpa using hj
          have hfact : (j.factorial:ℝ)*((m+1-j).factorial:ℝ) ≤ m.factorial := by
            have hh := factorial_pair (j-1) (m-j)
            have hjs : j-1+1=j := by omega
            have hks : m-j+1=m+1-j := by omega
            have ht : j-1+(m-j)+1=m := by omega
            rw [hjs,hks,ht] at hh
            exact_mod_cast hh
          calc
            _ = ‖coeff j (R^k)‖ * ‖coeff (m+1-j) R‖ := norm_mul _ _
            _ ≤ (ε^j*(j.factorial:ℝ))*(ε^(m+1-j)*((m+1-j).factorial:ℝ)) := by
              gcongr
              · exact ih j
              · exact hR _
            _ = ε^(m+1)*((j.factorial:ℝ)*((m+1-j).factorial:ℝ)) := by
              rw [mul_mul_mul_comm, ← pow_add, Nat.add_sub_of_le (by omega : j≤m+1)]
            _ ≤ _ := mul_le_mul_of_nonneg_left hfact (pow_nonneg hε _)
        _ = ε^(m+1)*((m+1).factorial:ℝ) := by
          simp [Nat.factorial_succ]; ring

/-- Actual substitution truncates at the requested degree. This proves
the formal exponential expansion, rather than assuming its estimate. -/
theorem exponential_coefficient (R : PowerSeries ℂ) (hR0 : constantCoeff R=0) (m : ℕ) :
    coeff m ((PowerSeries.exp ℂ).subst R) =
      ∑ k ∈ range (m+1), (k.factorial:ℂ)⁻¹ * coeff m (R^k) := by
  rw [coeff_subst' (HasSubst.of_constantCoeff_zero' hR0)]
  simp only [coeff_exp, one_div, map_inv₀, map_natCast, smul_eq_mul]
  apply finsum_eq_sum_of_support_subset
  intro k hk
  by_contra hkm
  have hm : m < k := by simpa using hkm
  have hz : coeff m (R^k)=0 := coeff_of_lt_order m
    ((by exact_mod_cast hm : (m:ℕ∞)<k).trans_le (le_order_pow_of_constantCoeff_eq_zero k hR0))
  exact hk (by simp [hz])

/-- Finite coefficient majorant for exp(a R), allowing complex signs and
phases. In particular a=-n gives the manuscript's exp(-n R). -/
theorem exponential_coefficient_majorant (R : PowerSeries ℂ) (a : ℂ) (ε : ℝ)
    (hε : 0 ≤ ε) (hR0 : constantCoeff R=0)
    (hR : ∀ n, ‖coeff n R‖ ≤ ε^n*(n.factorial:ℝ)) (m : ℕ) :
    ‖coeff m ((PowerSeries.exp ℂ).subst (C a * R))‖ ≤
      ε^m*(m.factorial:ℝ) * ∑ k ∈ range (m+1), ‖a‖^k/(k.factorial:ℝ) := by
  rw [exponential_coefficient _ (by simp [hR0])]
  calc
    _ ≤ ∑ k ∈ range (m+1), ‖(k.factorial:ℂ)⁻¹ * coeff m ((C a * R)^k)‖ := norm_sum_le _ _
    _ ≤ ∑ k ∈ range (m+1), (‖a‖^k/(k.factorial:ℝ))*(ε^m*(m.factorial:ℝ)) := by
      apply sum_le_sum
      intro k hk
      rw [mul_pow, ← map_pow, coeff_C_mul, norm_mul, norm_mul, norm_inv,
        Complex.norm_natCast, norm_pow]
      have hh := power_coefficient_bound R ε hε hR0 hR k m
      calc
        _ ≤ (k.factorial:ℝ)⁻¹ * (‖a‖^k * (ε^m*(m.factorial:ℝ))) := by gcongr
        _ = _ := by ring
    _ = _ := by rw [← sum_mul]; ring

/-- The actual formal exponential bound; slightly sharper than (8). -/
theorem exponential_coefficient_bound (R : PowerSeries ℂ) (a : ℂ) (ε : ℝ)
    (hε : 0 ≤ ε) (hR0 : constantCoeff R=0)
    (hR : ∀ n, ‖coeff n R‖ ≤ ε^n*(n.factorial:ℝ)) (m : ℕ) :
    ‖coeff m ((PowerSeries.exp ℂ).subst (C a * R))‖ ≤
      Real.exp ‖a‖ * ε^m*(m.factorial:ℝ) := by
  have hh := Real.sum_le_exp_of_nonneg (norm_nonneg a) (m+1)
  calc
    _ ≤ ε^m*(m.factorial:ℝ) * ∑ k ∈ range (m+1), ‖a‖^k/(k.factorial:ℝ) :=
      exponential_coefficient_majorant R a ε hε hR0 hR m
    _ ≤ ε^m*(m.factorial:ℝ) * Real.exp ‖a‖ := mul_le_mul_of_nonneg_left hh (by positivity)
    _ = _ := by ring

/-- Gap3exc §3 (8), with coefficients of exp(-nR), including m=0.
The former `hexpansion` handoff is entirely proved. -/
theorem manuscript_exponential_bound (R : PowerSeries ℂ) (ε : ℝ)
    (hε : 0 ≤ ε) (hR0 : constantCoeff R=0)
    (hR : ∀ n, ‖coeff n R‖ ≤ ε^n*(n.factorial:ℝ)) (n m : ℕ) :
    ‖coeff m ((PowerSeries.exp ℂ).subst (C (-(n:ℂ)) * R))‖ ≤
      Real.exp n * (2*ε)^m*(m.factorial:ℝ) := by
  have hh := exponential_coefficient_bound R (-(n:ℂ)) ε hε hR0 hR m
  simp only [norm_neg, Complex.norm_natCast] at hh
  apply hh.trans
  gcongr
  linarith

end Dobson.Owed2_Exponential

#print axioms Dobson.Owed2_Exponential.power_coefficient_bound
#print axioms Dobson.Owed2_Exponential.exponential_coefficient
#print axioms Dobson.Owed2_Exponential.manuscript_exponential_bound
