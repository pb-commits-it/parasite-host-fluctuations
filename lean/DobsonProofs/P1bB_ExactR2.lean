import DobsonProofs.Coverage5GammaFunctions
import DobsonProofs.GammaDigammaSeries
import Mathlib.Analysis.Calculus.Taylor

/-! # Part B: convergent connection-series expressions and scalar ratios

Coverage correction, 2026-11-22: `prop:exact-R2` remains partial-P1. The
independent fidelity audit (2026-11-16) did not confirm closure of
`part2:44` or `part2:45` by this uncredited module.

The Pochhammer/digamma/trigamma series are summable for positive parameters;
`kummerSeriesMass` and `kummerSeriesMoment` are defined as those sums.
Their equality to the actual mass and log-moment integrals is not proved.
`nthPolePair` is an exponential-quadratic surrogate, not the actual Gamma
connection pair. The combined logarithmic-derivative theorem takes the
value and derivative identifications for a supplied function `U` as premises. -/

namespace Dobson.P1bB_ExactR2

open Filter Set
open scoped BigOperators Topology
open Dobson.Coverage2Exact Dobson.GammaDigammaSeries Dobson.Coverage5GammaFunctions

/-- Rising Pochhammer symbol `(p)_n = ∏_{k < n} (p + k)`. -/
def pochhammerReal (p : ℝ) (n : ℕ) : ℝ :=
  ∏ k ∈ Finset.range n, (p + (k : ℝ))

lemma pochhammerReal_zero (p : ℝ) : pochhammerReal p 0 = 1 := by
  simp [pochhammerReal]

lemma pochhammerReal_succ (p : ℝ) (n : ℕ) :
    pochhammerReal p (n + 1) = pochhammerReal p n * (p + (n : ℝ)) := by
  simp [pochhammerReal, Finset.prod_range_succ]

lemma pochhammerReal_pos (p : ℝ) (hp : 0 < p) (n : ℕ) :
    0 < pochhammerReal p n := by
  unfold pochhammerReal
  exact Finset.prod_pos fun k _ => by positivity

/-- Ratio identity `(p)_n = Γ(p + n) / Γ(p)` for `p > 0`. -/
lemma pochhammerReal_eq_gamma_ratio (p : ℝ) (hp : 0 < p) (n : ℕ) :
    pochhammerReal p n = Real.Gamma (p + (n : ℝ)) / Real.Gamma p := by
  have hGp : Real.Gamma p ≠ 0 := (Real.Gamma_pos_of_pos hp).ne'
  induction n with
  | zero =>
    simp [pochhammerReal_zero, hGp]
  | succ n ih =>
    have hpn_pos : 0 < p + (n : ℝ) := by positivity
    rw [pochhammerReal_succ, ih, Nat.cast_add, Nat.cast_one, ← add_assoc,
      Real.Gamma_add_one hpn_pos.ne']
    field_simp [hGp]

lemma pochhammer_div_factorial_bound (p : ℝ) (hp : 0 < p) (n : ℕ) :
    pochhammerReal p n / (n.factorial : ℝ) ≤ (Nat.ceil p + 1 : ℝ) ^ n := by
  induction n with
  | zero => simp [pochhammerReal_zero]
  | succ n ih =>
    have hfac : ((n + 1).factorial : ℝ) = (n.factorial : ℝ) * ((n : ℝ) + 1) := by
      rw [Nat.factorial_succ]
      push_cast
      ring
    rw [pochhammerReal_succ, hfac, mul_div_mul_comm, pow_succ]
    have hp_le : p ≤ (Nat.ceil p : ℝ) := Nat.le_ceil p
    have hstep : (p + (n : ℝ)) / ((n : ℝ) + 1) ≤ (Nat.ceil p + 1 : ℝ) := by
      have hden : 0 < (n : ℝ) + 1 := by positivity
      rw [div_le_iff₀ hden]
      have hceil_nn : (0 : ℝ) ≤ (Nat.ceil p : ℝ) := Nat.cast_nonneg _
      have hn_nn : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg _
      nlinarith
    have hnn1 : 0 ≤ pochhammerReal p n / (n.factorial : ℝ) :=
      div_nonneg (pochhammerReal_pos p hp n).le (Nat.cast_nonneg _)
    have hnn2 : 0 ≤ (p + (n : ℝ)) / ((n : ℝ) + 1) :=
      div_nonneg (by positivity) (by positivity)
    exact mul_le_mul ih hstep hnn2 (pow_nonneg (by positivity) n)

/-- Actual Kummer series weight `C_n = (p)_n κ^n / (n!)²` (`D-exact.tex` line 78). -/
noncomputable def kummerSeriesCoeff (p κ : ℝ) (n : ℕ) : ℝ :=
  pochhammerReal p n * κ ^ n / ((n.factorial : ℝ) ^ 2)

lemma kummerSeriesCoeff_nonneg (p κ : ℝ) (hp : 0 < p) (hκ : 0 ≤ κ) (n : ℕ) :
    0 ≤ kummerSeriesCoeff p κ n := by
  unfold kummerSeriesCoeff
  exact div_nonneg (mul_nonneg (pochhammerReal_pos p hp n).le (pow_nonneg hκ n)) (sq_nonneg _)

lemma kummerSeriesCoeff_le_exp_term (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) (n : ℕ) :
    kummerSeriesCoeff p κ n ≤ (((Nat.ceil p + 1 : ℝ) * κ) ^ n) / (n.factorial : ℝ) := by
  have hpoch := pochhammer_div_factorial_bound p hp n
  have hfac_pos : (0 : ℝ) < (n.factorial : ℝ) := Nat.cast_pos.mpr (Nat.factorial_pos n)
  unfold kummerSeriesCoeff
  calc
    pochhammerReal p n * κ ^ n / (n.factorial : ℝ) ^ 2
        = (pochhammerReal p n / (n.factorial : ℝ)) * (κ ^ n / (n.factorial : ℝ)) := by
          rw [sq]
          field_simp [hfac_pos.ne']
    _ ≤ (Nat.ceil p + 1 : ℝ) ^ n * (κ ^ n / (n.factorial : ℝ)) :=
          mul_le_mul_of_nonneg_right hpoch (div_nonneg (pow_nonneg hκ.le n) hfac_pos.le)
    _ = (((Nat.ceil p + 1 : ℝ) * κ) ^ n) / (n.factorial : ℝ) := by
          rw [mul_pow, mul_div_assoc]

/-- Harmonic number `H_n = ∑_{k < n} 1 / (k + 1) = ψ(n + 1) + γ` (`D-exact.tex` line 86). -/
noncomputable def harmonicNumber (n : ℕ) : ℝ :=
  ∑ k ∈ Finset.range n, 1 / ((k : ℝ) + 1)

/-- Identification `H_n = ψ(n + 1) + γ` proved from `digamma_nat_recurrence` at `x = 1`. -/
lemma harmonicNumber_eq_digamma_add_gamma (n : ℕ) :
    harmonicNumber n = gammaLogDerivative ((n : ℝ) + 1) + Real.eulerMascheroniConstant := by
  have hrec := digamma_nat_recurrence 1 zero_lt_one n
  have hψ1 : gammaLogDerivative 1 = -Real.eulerMascheroniConstant := by
    dsimp [gammaLogDerivative]
    rw [Real.hasDerivAt_Gamma_one.deriv, Real.Gamma_one, div_one]
  rw [hψ1, add_comm (1 : ℝ) (n : ℝ)] at hrec
  unfold harmonicNumber
  have hsum : (∑ k ∈ Finset.range n, 1 / ((k : ℝ) + 1)) =
      ∑ k ∈ Finset.range n, 1 / (1 + (k : ℝ)) := by
    apply Finset.sum_congr rfl
    intro k _
    ring
  rw [hsum]
  linarith

lemma harmonicNumber_nonneg (n : ℕ) : 0 ≤ harmonicNumber n :=
  Finset.sum_nonneg fun _ _ => by positivity

lemma harmonicNumber_le_nat (n : ℕ) : harmonicNumber n ≤ (n : ℝ) := by
  unfold harmonicNumber
  calc
    (∑ k ∈ Finset.range n, 1 / ((k : ℝ) + 1)) ≤ ∑ _k ∈ Finset.range n, (1 : ℝ) := by
      apply Finset.sum_le_sum
      intro k _
      have hk : (1 : ℝ) ≤ (k : ℝ) + 1 := by linarith [Nat.cast_nonneg (α := ℝ) k]
      exact div_le_one_of_le₀ hk (by positivity)
    _ = (n : ℝ) := by simp

lemma digamma_shift_abs_le (p : ℝ) (hp : 0 < p) (n : ℕ) :
    |gammaLogDerivative (p + (n : ℝ))| ≤ |gammaLogDerivative p| + (n : ℝ) / p := by
  rw [digamma_nat_recurrence p hp n]
  apply (abs_add_le _ _).trans
  apply add_le_add le_rfl
  rw [abs_of_nonneg (Finset.sum_nonneg fun k _ => by positivity)]
  calc
    (∑ k ∈ Finset.range n, 1 / (p + (k : ℝ))) ≤ ∑ _k ∈ Finset.range n, (1 / p) := by
      apply Finset.sum_le_sum
      intro k _
      exact one_div_le_one_div_of_le hp (by linarith [Nat.cast_nonneg (α := ℝ) k])
    _ = (n : ℝ) / p := by simp [div_eq_mul_inv]

lemma trigamma_shift_bounds (p : ℝ) (hp : 0 < p) (n : ℕ) :
    0 ≤ gammaTrigamma (p + (n : ℝ)) ∧ gammaTrigamma (p + (n : ℝ)) ≤ gammaTrigamma p := by
  have hs_pn : Summable (fun k : ℕ => 1 / (p + (n : ℝ) + (k : ℝ)) ^ 2) := by
    have hh := (Real.summable_one_div_nat_add_rpow (p + (n : ℝ)) (s := 2)).mpr (by norm_num)
    simpa only [Real.rpow_two, sq_abs, add_comm] using hh
  have hs_p : Summable (fun k : ℕ => 1 / (p + (k : ℝ)) ^ 2) := by
    have hh := (Real.summable_one_div_nat_add_rpow p (s := 2)).mpr (by norm_num)
    simpa only [Real.rpow_two, sq_abs, add_comm] using hh
  refine ⟨tsum_nonneg fun _ => by positivity, ?_⟩
  exact hs_pn.tsum_le_tsum (fun k => by
    have h1 : 0 < p + (k : ℝ) := by positivity
    have h2 : p + (k : ℝ) ≤ p + (n : ℝ) + (k : ℝ) := by linarith [Nat.cast_nonneg (α := ℝ) n]
    exact one_div_le_one_div_of_le (sq_pos_of_pos h1) (pow_le_pow_left₀ h1.le h2 2)) hs_p

lemma nat_le_two_pow (n : ℕ) : (n : ℝ) ≤ (2 : ℝ) ^ n := by
  induction n with
  | zero => norm_num
  | succ n ih =>
    rw [pow_succ]
    push_cast
    cases n with
    | zero => norm_num
    | succ m =>
      have hm : (1 : ℝ) ≤ (2 : ℝ) ^ (m + 1) := one_le_pow₀ (by norm_num)
      linarith

lemma summable_pow_mul_exp_series (R : ℝ) (hR : 0 ≤ R) (d : ℕ) :
    Summable (fun n : ℕ => ((n : ℝ) + 1) ^ d * (R ^ n / (n.factorial : ℝ))) := by
  let S : ℝ := (2 : ℝ) ^ d * R
  have hexp : Summable (fun n : ℕ => (2 : ℝ) ^ d * (S ^ n / (n.factorial : ℝ))) :=
    (Real.summable_pow_div_factorial S).mul_left ((2 : ℝ) ^ d)
  apply Summable.of_norm_bounded hexp
  intro n
  have hn2 : (n : ℝ) + 1 ≤ (2 : ℝ) ^ (n + 1) := by
    simpa [Nat.cast_add, Nat.cast_one] using nat_le_two_pow (n + 1)
  have hpow : ((n : ℝ) + 1) ^ d ≤ ((2 : ℝ) ^ d) * ((2 : ℝ) ^ d) ^ n := by
    calc
      ((n : ℝ) + 1) ^ d ≤ ((2 : ℝ) ^ (n + 1)) ^ d :=
        pow_le_pow_left₀ (by positivity) hn2 d
      _ = (2 : ℝ) ^ ((n + 1) * d) := by rw [← pow_mul]
      _ = (2 : ℝ) ^ (d + d * n) := by congr 1; ring
      _ = (2 : ℝ) ^ d * ((2 : ℝ) ^ d) ^ n := by rw [pow_add, pow_mul]
  rw [Real.norm_of_nonneg (by positivity)]
  calc
    ((n : ℝ) + 1) ^ d * (R ^ n / (n.factorial : ℝ))
        ≤ ((2 : ℝ) ^ d * ((2 : ℝ) ^ d) ^ n) * (R ^ n / (n.factorial : ℝ)) :=
          mul_le_mul_of_nonneg_right hpow (div_nonneg (pow_nonneg hR n) (Nat.cast_nonneg _))
    _ = (2 : ℝ) ^ d * (S ^ n / (n.factorial : ℝ)) := by
          dsimp [S]
          rw [mul_pow]
          ring

/-- Absolute summability of the pole-cancelled mass series (`part2:44`) for all `p > 0, κ > 0`. -/
theorem pole_cancelled_mass_summable (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    let H := -Real.log κ
    let γ := Real.eulerMascheroniConstant
    let P := gammaLogDerivative p
    let ψn : ℕ → ℝ := fun n => gammaLogDerivative (p + (n : ℝ))
    let Hn : ℕ → ℝ := harmonicNumber
    Summable (fun n : ℕ => kummerSeriesCoeff p κ n *
      ((H - γ - ψn n + P + Hn n) - (P + γ - Hn n))) := by
  intro H γ P ψn Hn
  let R : ℝ := (Nat.ceil p + 1 : ℝ) * κ
  have hR : 0 ≤ R := by dsimp [R]; positivity
  let K : ℝ := |H| + 2 * |γ| + |gammaLogDerivative p| + 2 + 1 / p
  have hK : 0 ≤ K := by dsimp [K]; positivity
  have hmaj : Summable (fun n : ℕ => K * (((n : ℝ) + 1) ^ 1 * (R ^ n / (n.factorial : ℝ)))) :=
    (summable_pow_mul_exp_series R hR 1).mul_left K
  apply Summable.of_norm_bounded hmaj
  intro n
  have hcoeff := kummerSeriesCoeff_le_exp_term p κ hp hκ n
  have hcoeff_nn := kummerSeriesCoeff_nonneg p κ hp hκ.le n
  have halg : (H - γ - ψn n + P + Hn n) - (P + γ - Hn n) = H + 2 * (Hn n - γ) - ψn n := by ring
  rw [halg, Real.norm_eq_abs, abs_mul, abs_of_nonneg hcoeff_nn]
  have hHn_abs : |Hn n| ≤ (n : ℝ) := by
    rw [abs_of_nonneg (harmonicNumber_nonneg n)]
    exact harmonicNumber_le_nat n
  have hψn_abs : |ψn n| ≤ |gammaLogDerivative p| + (n : ℝ) / p :=
    digamma_shift_abs_le p hp n
  have hbrack : |H + 2 * (Hn n - γ) - ψn n| ≤ K * ((n : ℝ) + 1) ^ 1 := by
    have htri1 := abs_sub_le (H + 2 * (Hn n - γ)) 0 (ψn n)
    have htri2 := abs_add_le H (2 * (Hn n - γ))
    have htri3 := abs_sub_le (Hn n) 0 γ
    simp only [sub_zero, zero_sub, abs_neg, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
      at htri1 htri2 htri3
    have hn_nn : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
    have hp_inv : 0 ≤ 1 / p := by positivity
    have hdiv_eq : (n : ℝ) / p = (1 / p) * (n : ℝ) := by ring
    have hc_mul : (|H| + 2 * |γ| + |gammaLogDerivative p|) * (1 : ℝ) ≤
        (|H| + 2 * |γ| + |gammaLogDerivative p|) * ((n : ℝ) + 1) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have hn_mul : (2 + 1 / p) * (n : ℝ) ≤ (2 + 1 / p) * ((n : ℝ) + 1) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    dsimp [K]
    rw [pow_one]
    linarith
  calc
    kummerSeriesCoeff p κ n * |H + 2 * (Hn n - γ) - ψn n|
        ≤ (R ^ n / (n.factorial : ℝ)) * (K * ((n : ℝ) + 1) ^ 1) :=
          mul_le_mul hcoeff hbrack (abs_nonneg _) (div_nonneg (pow_nonneg hR n) (Nat.cast_nonneg _))
    _ = K * (((n : ℝ) + 1) ^ 1 * (R ^ n / (n.factorial : ℝ))) := by ring

/-- Absolute summability of the pole-cancelled moment series (`part2:45`) for all `p > 0, κ > 0`. -/
theorem pole_cancelled_moment_summable (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    let H := -Real.log κ
    let γ := Real.eulerMascheroniConstant
    let P := gammaLogDerivative p
    let ψn : ℕ → ℝ := fun n => gammaLogDerivative (p + (n : ℝ))
    let Hn : ℕ → ℝ := harmonicNumber
    let v : ℕ → ℝ := fun n => gammaTrigamma (p + (n : ℝ))
    Summable (fun n : ℕ => kummerSeriesCoeff p κ n *
      (((H - γ - ψn n + P + Hn n) ^ 2 - (P + γ - Hn n) ^ 2 + v n) / 2)) := by
  intro H γ P ψn Hn v
  let R : ℝ := (Nat.ceil p + 1 : ℝ) * κ
  have hR : 0 ≤ R := by dsimp [R]; positivity
  let K1 : ℝ := |H| + 2 * |γ| + |gammaLogDerivative p| + 2 + 1 / p
  let K2 : ℝ := |H| + 3 * |gammaLogDerivative p| + 1 / p
  let K : ℝ := K1 * K2 + gammaTrigamma p
  have hK1 : 0 ≤ K1 := by dsimp [K1]; positivity
  have hK2 : 0 ≤ K2 := by dsimp [K2]; positivity
  have hmaj : Summable (fun n : ℕ => K * (((n : ℝ) + 1) ^ 2 * (R ^ n / (n.factorial : ℝ)))) :=
    (summable_pow_mul_exp_series R hR 2).mul_left K
  apply Summable.of_norm_bounded hmaj
  intro n
  have hcoeff := kummerSeriesCoeff_le_exp_term p κ hp hκ n
  have hcoeff_nn := kummerSeriesCoeff_nonneg p κ hp hκ.le n
  have halg := (pole_cancellation_coefficients P (ψn n) (Hn n) γ H (v n)).2
  rw [halg, Real.norm_eq_abs, abs_mul, abs_of_nonneg hcoeff_nn]
  have hHn_abs : |Hn n| ≤ (n : ℝ) := by
    rw [abs_of_nonneg (harmonicNumber_nonneg n)]
    exact harmonicNumber_le_nat n
  have hψn_abs : |ψn n| ≤ |gammaLogDerivative p| + (n : ℝ) / p :=
    digamma_shift_abs_le p hp n
  have hv_bounds := trigamma_shift_bounds p hp n
  have hb1 : |H + 2 * (Hn n - γ) - ψn n| ≤ K1 * ((n : ℝ) + 1) := by
    have htri1 := abs_sub_le (H + 2 * (Hn n - γ)) 0 (ψn n)
    have htri2 := abs_add_le H (2 * (Hn n - γ))
    have htri3 := abs_sub_le (Hn n) 0 γ
    simp only [sub_zero, zero_sub, abs_neg, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
      at htri1 htri2 htri3
    have hn_nn : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
    have hp_inv : 0 ≤ 1 / p := by positivity
    have hdiv_eq : (n : ℝ) / p = (1 / p) * (n : ℝ) := by ring
    have hc_mul : (|H| + 2 * |γ| + |gammaLogDerivative p|) * (1 : ℝ) ≤
        (|H| + 2 * |γ| + |gammaLogDerivative p|) * ((n : ℝ) + 1) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have hn_mul : (2 + 1 / p) * (n : ℝ) ≤ (2 + 1 / p) * ((n : ℝ) + 1) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    dsimp [K1]
    linarith
  have hb2 : |H - ψn n + 2 * P| ≤ K2 * ((n : ℝ) + 1) := by
    have htri1 := abs_add_le (H - ψn n) (2 * P)
    have htri2 := abs_sub_le H 0 (ψn n)
    simp only [sub_zero, zero_sub, abs_neg, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
      at htri1 htri2
    have hn_nn : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
    have hp_inv : 0 ≤ 1 / p := by positivity
    have hdiv_eq : (n : ℝ) / p = (1 / p) * (n : ℝ) := by ring
    have hc_mul : (|H| + 3 * |gammaLogDerivative p|) * (1 : ℝ) ≤
        (|H| + 3 * |gammaLogDerivative p|) * ((n : ℝ) + 1) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have hn_mul : (1 / p) * (n : ℝ) ≤ (1 / p) * ((n : ℝ) + 1) :=
      mul_le_mul_of_nonneg_left (by linarith) hp_inv
    dsimp [K2, P]
    linarith
  have hbrack : |((H + 2 * (Hn n - γ) - ψn n) * (H - ψn n + 2 * P) + v n) / 2| ≤
      K * ((n : ℝ) + 1) ^ 2 := by
    rw [abs_div, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
    have hprod : |(H + 2 * (Hn n - γ) - ψn n) * (H - ψn n + 2 * P)| ≤
        (K1 * K2) * ((n : ℝ) + 1) ^ 2 := by
      rw [abs_mul]
      calc
        |H + 2 * (Hn n - γ) - ψn n| * |H - ψn n + 2 * P|
            ≤ (K1 * ((n : ℝ) + 1)) * (K2 * ((n : ℝ) + 1)) :=
              mul_le_mul hb1 hb2 (abs_nonneg _) (mul_nonneg hK1 (by positivity))
        _ = (K1 * K2) * ((n : ℝ) + 1) ^ 2 := by ring
    have htri := abs_add_le ((H + 2 * (Hn n - γ) - ψn n) * (H - ψn n + 2 * P)) (v n)
    have hv_abs : |v n| ≤ gammaTrigamma p * ((n : ℝ) + 1) ^ 2 := by
      rw [abs_of_nonneg hv_bounds.1]
      have h1 : (1 : ℝ) ≤ ((n : ℝ) + 1) ^ 2 := by
        have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
        nlinarith
      calc
        v n ≤ gammaTrigamma p * 1 := by linarith [hv_bounds.2]
        _ ≤ gammaTrigamma p * ((n : ℝ) + 1) ^ 2 :=
          mul_le_mul_of_nonneg_left h1 (hv_bounds.1.trans hv_bounds.2)
    have hv0 : 0 ≤ gammaTrigamma p := hv_bounds.1.trans hv_bounds.2
    have hn1_sq : 0 ≤ ((n : ℝ) + 1) ^ 2 := sq_nonneg _
    dsimp [K]
    nlinarith [mul_nonneg (mul_nonneg hK1 hK2) hn1_sq, mul_nonneg hv0 hn1_sq]
  calc
    kummerSeriesCoeff p κ n * |((H + 2 * (Hn n - γ) - ψn n) * (H - ψn n + 2 * P) + v n) / 2|
        ≤ (R ^ n / (n.factorial : ℝ)) * (K * ((n : ℝ) + 1) ^ 2) :=
          mul_le_mul hcoeff hbrack (abs_nonneg _) (div_nonneg (pow_nonneg hR n) (Nat.cast_nonneg _))
    _ = K * (((n : ℝ) + 1) ^ 2 * (R ^ n / (n.factorial : ℝ))) := by ring

/-- Exponential-quadratic pole-cancellation surrogate:
`F(η) = C * (exp(B η + β_B η² / 2) - exp(A η + β_A η² / 2)) / η`,
extended at zero by its limit. This is not the actual Gamma connection pair. -/
noncomputable def nthPolePair (C A_n B_n β_A β_B η : ℝ) : ℝ :=
  if η = 0 then C * (B_n - A_n)
  else C * (Real.exp (B_n * η + β_B * η ^ 2 / 2) - Real.exp (A_n * η + β_A * η ^ 2 / 2)) / η

/-- Value and derivative at zero of the exponential-quadratic surrogate
`nthPolePair`. This calculation does not identify it with a Gamma/Pochhammer
connection pair. -/
theorem nthPolePair_value_and_deriv (C A_n B_n β_A β_B : ℝ) :
    nthPolePair C A_n B_n β_A β_B 0 = C * (B_n - A_n) ∧
    HasDerivAt (nthPolePair C A_n B_n β_A β_B)
      (C * ((B_n ^ 2 - A_n ^ 2 + (β_B - β_A)) / 2)) 0 := by
  refine ⟨by simp [nthPolePair], ?_⟩
  rw [hasDerivAt_iff_tendsto_slope_zero]
  let fB : ℝ → ℝ := fun η => Real.exp (B_n * η + β_B * η ^ 2 / 2)
  let fA : ℝ → ℝ := fun η => Real.exp (A_n * η + β_A * η ^ 2 / 2)
  have quad_jet : ∀ a β : ℝ,
      Tendsto (fun η : ℝ => (Real.exp (a * η + β * η ^ 2 / 2) - 1 - a * η) / η ^ 2)
        (𝓝[≠] 0) (𝓝 ((a ^ 2 + β) / 2)) := by
    intro a β
    let g : ℝ → ℝ := fun η => Real.exp (a * η + β * η ^ 2 / 2)
    have hg_cont : ContDiff ℝ 2 g := by dsimp [g]; fun_prop
    have ht := Real.taylor_tendsto convex_univ (mem_univ (0 : ℝ)) hg_cont.contDiffOn
    rw [nhdsWithin_univ] at ht
    have hh := (ht.mono_left (nhdsWithin_le_nhds (s := ({0}ᶜ : Set ℝ)))).add_const
      ((a ^ 2 + β) / 2)
    simp only [zero_add] at hh
    have hd1_at : ∀ y : ℝ, HasDerivAt g ((a + β * y) * g y) y := by
      intro y
      have hinner : HasDerivAt (fun t : ℝ => a * t + β * t ^ 2 / 2) (a + β * y) y := by
        convert ((hasDerivAt_id y).const_mul a).add
          ((((hasDerivAt_id y).pow 2).const_mul β).div_const 2) using 1
        · funext t; rfl
        · simp; ring
      simpa [g, mul_comm] using hinner.exp
    have hd2_at0 : HasDerivAt (deriv g) (a ^ 2 + β) 0 := by
      have hderiv_eq : deriv g = fun y => (a + β * y) * g y := funext fun y => (hd1_at y).deriv
      rw [hderiv_eq]
      have hlin : HasDerivAt (fun y : ℝ => a + β * y) β 0 := by
        convert ((hasDerivAt_id (0 : ℝ)).const_mul β).const_add a using 1
        · funext y; simp
        · simp
      convert hlin.mul (hd1_at 0) using 1
      simp [g]
      ring
    apply hh.congr'
    filter_upwards [self_mem_nhdsWithin] with η hη
    have hη' : η ≠ 0 := hη
    have hteval : taylorWithinEval g 2 univ 0 η = 1 + a * η + (a ^ 2 + β) / 2 * η ^ 2 := by
      rw [taylorWithinEval_succ (n := 1), taylorWithinEval_succ (n := 0)]
      have hg0 : g 0 = 1 := by simp [g]
      have hg1 : iteratedDeriv 1 g 0 = a := by
        rw [iteratedDeriv_one, (hd1_at 0).deriv]
        simp [g]
      have hg2 : iteratedDeriv 2 g 0 = a ^ 2 + β := by
        rw [iteratedDeriv_succ, iteratedDeriv_one, hd2_at0.deriv]
      simp [taylor_within_zero_eval, iteratedDerivWithin_univ, hg0, hg1, hg2]
      ring
    rw [hteval]
    simp only [g, sub_zero]
    field_simp [hη']
    ring
  have hdiff := ((quad_jet B_n β_B).sub (quad_jet A_n β_A)).const_mul C
  have hlim_eq : C * ((B_n ^ 2 + β_B) / 2 - (A_n ^ 2 + β_A) / 2) =
      C * ((B_n ^ 2 - A_n ^ 2 + (β_B - β_A)) / 2) := by ring
  rw [hlim_eq] at hdiff
  apply hdiff.congr'
  filter_upwards [self_mem_nhdsWithin] with η hη
  have hη' : η ≠ 0 := hη
  simp only [zero_add, smul_eq_mul, inv_mul_eq_div, nthPolePair, ite_true,
    ite_eq_right_iff.mpr fun h => (hη' h).elim]
  field_simp [hη']
  ring

/-- Mass-series expression defined by a convergent sum for positive `p, κ`.
Equality to the Kummer mass integral is not proved here. -/
noncomputable def kummerSeriesMass (p κ : ℝ) : ℝ :=
  let H := -Real.log κ
  let γ := Real.eulerMascheroniConstant
  let P := gammaLogDerivative p
  let ψn : ℕ → ℝ := fun n => gammaLogDerivative (p + (n : ℝ))
  let Hn : ℕ → ℝ := harmonicNumber
  ∑' n : ℕ, kummerSeriesCoeff p κ n * ((H - γ - ψn n + P + Hn n) - (P + γ - Hn n))

/-- Log-moment-series expression defined by a convergent sum for positive
`p, κ`. Equality to the Kummer log-moment integral is not proved here. -/
noncomputable def kummerSeriesMoment (p κ : ℝ) : ℝ :=
  let H := -Real.log κ
  let γ := Real.eulerMascheroniConstant
  let P := gammaLogDerivative p
  let ψn : ℕ → ℝ := fun n => gammaLogDerivative (p + (n : ℝ))
  let Hn : ℕ → ℝ := harmonicNumber
  let v : ℕ → ℝ := fun n => gammaTrigamma (p + (n : ℝ))
  ∑' n : ℕ, kummerSeriesCoeff p κ n *
    (((H - γ - ψn n + P + Hn n) ^ 2 - (P + γ - Hn n) ^ 2 + v n) / 2)

/-- The mass series sums to `kummerSeriesMass`, which is defined as its
sum. This does not discharge the mass-integral identification `part2:44`. -/
lemma pole_cancelled_mass_hasSum (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    let H := -Real.log κ
    let γ := Real.eulerMascheroniConstant
    let P := gammaLogDerivative p
    let ψn : ℕ → ℝ := fun n => gammaLogDerivative (p + (n : ℝ))
    let Hn : ℕ → ℝ := harmonicNumber
    HasSum (fun n : ℕ => kummerSeriesCoeff p κ n *
      ((H - γ - ψn n + P + Hn n) - (P + γ - Hn n))) (kummerSeriesMass p κ) :=
  (pole_cancelled_mass_summable p κ hp hκ).hasSum

/-- The log-moment series sums to `kummerSeriesMoment`, which is defined
as its sum. This does not discharge the integral identification `part2:45`. -/
lemma pole_cancelled_moment_hasSum (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    let H := -Real.log κ
    let γ := Real.eulerMascheroniConstant
    let P := gammaLogDerivative p
    let ψn : ℕ → ℝ := fun n => gammaLogDerivative (p + (n : ℝ))
    let Hn : ℕ → ℝ := harmonicNumber
    let v : ℕ → ℝ := fun n => gammaTrigamma (p + (n : ℝ))
    HasSum (fun n : ℕ => kummerSeriesCoeff p κ n *
      (((H - γ - ψn n + P + Hn n) ^ 2 - (P + γ - Hn n) ^ 2 + v n) / 2))
      (kummerSeriesMoment p κ) :=
  (pole_cancelled_moment_summable p κ hp hκ).hasSum

/-- Ratio identity for the two defined convergent series. It follows from
their HasSum lemmas and pole-cancellation algebra; equality with actual Kummer
integrals or a Tricomi logarithmic derivative is not asserted here. -/
theorem pole_cancelled_series_ratio_unconditional (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    let H := -Real.log κ
    let γ := Real.eulerMascheroniConstant
    let P := gammaLogDerivative p
    let ψn : ℕ → ℝ := fun n => gammaLogDerivative (p + (n : ℝ))
    let Hn : ℕ → ℝ := harmonicNumber
    let v : ℕ → ℝ := fun n => gammaTrigamma (p + (n : ℝ))
    kummerSeriesMoment p κ / kummerSeriesMass p κ =
      (∑' n : ℕ, kummerSeriesCoeff p κ n *
        ((H + 2 * (Hn n - γ) - ψn n) * (H - ψn n + 2 * P) + v n)) /
      (2 * ∑' n : ℕ, kummerSeriesCoeff p κ n * (H + 2 * (Hn n - γ) - ψn n)) := by
  intro H γ P ψn Hn v
  exact pole_cancelled_series_ratio (kummerSeriesCoeff p κ) ψn Hn v P γ H
    (kummerSeriesMass p κ) (kummerSeriesMoment p κ)
    (pole_cancelled_mass_hasSum p κ hp hκ)
    (pole_cancelled_moment_hasSum p κ hp hκ)

/-- Conditional logarithmic-derivative identity for a supplied function
`U`. Its value and derivative at `1` are assumed to equal the normalized
series sums, and the mass sum is assumed nonzero. These paper-specific
identifications do not construct Tricomi's function or physical growth. -/
theorem exact_R2_series_and_log_deriv (p κ s : ℝ) (U : ℝ → ℝ)
    (hp : 0 < p) (hκ : 0 < κ)
    (hZ : kummerSeriesMass p κ ≠ 0)
    (hU_val : U 1 = kummerSeriesMass p κ / Real.Gamma p)
    (hU_deriv : HasDerivAt U (kummerSeriesMoment p κ / Real.Gamma p) 1) :
    let H := -Real.log κ
    let γ := Real.eulerMascheroniConstant
    let P := gammaLogDerivative p
    let ψn : ℕ → ℝ := fun n => gammaLogDerivative (p + (n : ℝ))
    let Hn : ℕ → ℝ := harmonicNumber
    let v : ℕ → ℝ := fun n => gammaTrigamma (p + (n : ℝ))
    Real.log s + deriv (fun b => Real.log (U b)) 1 =
      Real.log s +
        (∑' n : ℕ, kummerSeriesCoeff p κ n *
          ((H + 2 * (Hn n - γ) - ψn n) * (H - ψn n + 2 * P) + v n)) /
        (2 * ∑' n : ℕ, kummerSeriesCoeff p κ n * (H + 2 * (Hn n - γ) - ψn n)) := by
  intro H γ P ψn Hn v
  have hGp : Real.Gamma p ≠ 0 := (Real.Gamma_pos_of_pos hp).ne'
  have hlog := logarithmic_parameter_formula U (Real.Gamma p)
    (kummerSeriesMass p κ) (kummerSeriesMoment p κ) s hGp hZ hU_val hU_deriv
  rw [← hlog, pole_cancelled_series_ratio_unconditional p κ hp hκ]

end Dobson.P1bB_ExactR2

#print axioms Dobson.P1bB_ExactR2.pole_cancelled_mass_hasSum
#print axioms Dobson.P1bB_ExactR2.pole_cancelled_moment_hasSum
#print axioms Dobson.P1bB_ExactR2.nthPolePair_value_and_deriv
#print axioms Dobson.P1bB_ExactR2.pole_cancelled_series_ratio_unconditional
#print axioms Dobson.P1bB_ExactR2.exact_R2_series_and_log_deriv
