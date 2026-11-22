import DobsonProofs.Coverage1Lineage
import DobsonProofs.Coverage3Renewal
import DobsonProofs.Coverage6Remainders
import DobsonProofs.Coverage2Alternatives

/-! Actual first-return generating functions for a finite Markov chain.
The target is the renewal bridge used by prop:ring-spectrum. -/

open scoped BigOperators

namespace Dobson.CxA_Renewal
open Dobson.FiniteLineageVariance
variable {E : Type*} [Fintype E] [DecidableEq E]

lemma power_nonneg (K : Chain E) (n : ℕ) (i j : E) : 0 ≤ (K.prob ^ n) i j := by
  induction n generalizing i j with
  | zero => simp only [pow_zero, Matrix.one_apply]; split_ifs <;> norm_num
  | succ n ih =>
    rw [pow_succ', Matrix.mul_apply]
    exact Finset.sum_nonneg (fun k _ => mul_nonneg (K.nonneg i k) (ih k j))

lemma power_total (K : Chain E) (n : ℕ) (i : E) : ∑ j, (K.prob ^ n) i j = 1 := by
  induction n generalizing i with
  | zero => simp [Matrix.one_apply]
  | succ n ih =>
    simp only [pow_succ', Matrix.mul_apply]
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, ih, mul_one, K.total]

lemma power_le_one (K : Chain E) (n : ℕ) (i j : E) : (K.prob ^ n) i j ≤ 1 := by
  rw [← power_total K n i]
  exact Finset.single_le_sum (fun k _ => power_nonneg K n i k) (Finset.mem_univ j)

/-- First hitting at exactly time n, with hitting time zero allowed at the target. -/
noncomputable def hit (K : Chain E) (e : E) : ℕ → E → ℝ
  | 0, i => if i = e then 1 else 0
  | n+1, i => if i = e then 0 else ∑ j, K.prob i j * hit K e n j

lemma hit_bounds (K : Chain E) (e : E) (n : ℕ) (i : E) :
    0 ≤ hit K e n i ∧ hit K e n i ≤ 1 := by
  induction n generalizing i with
  | zero => simp only [hit]; split_ifs <;> norm_num
  | succ n ih =>
    simp only [hit]
    split_ifs
    · norm_num
    · constructor
      · exact Finset.sum_nonneg (fun j _ => mul_nonneg (K.nonneg i j) (ih j).1)
      · calc
          _ ≤ ∑ j, K.prob i j * 1 := Finset.sum_le_sum (fun j _ =>
            mul_le_mul_of_nonneg_left (ih j).2 (K.nonneg i j))
          _ = 1 := by simp [K.total]

lemma weighted_summable (a : ℕ → ℝ) (ha : ∀ n, 0 ≤ a n ∧ a n ≤ 1)
    (z : ℂ) (hz : ‖z‖ < 1) : Summable (fun n => z^n * (a n : ℂ)) := by
  apply Summable.of_norm_bounded (summable_geometric_of_lt_one (norm_nonneg z) hz)
  intro n
  rw [norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (ha n).1]
  exact mul_le_of_le_one_right (by positivity) (ha n).2

noncomputable def green (K : Chain E) (e : E) (z : ℂ) (i : E) : ℂ :=
  ∑' n : ℕ, z^n * ((K.prob^n) i e : ℂ)

noncomputable def hitTransform (K : Chain E) (e : E) (z : ℂ) (i : E) : ℂ :=
  ∑' n : ℕ, z^n * (hit K e n i : ℂ)

/-- The PGF of the actual positive first return, by its first transition. -/
noncomputable def returnTransform (K : Chain E) (e : E) (z : ℂ) : ℂ :=
  z * ∑ j, (K.prob e j : ℂ) * hitTransform K e z j

lemma green_summable (K : Chain E) (e : E) (z : ℂ) (hz : ‖z‖ < 1) (i : E) :
    Summable (fun n : ℕ => z^n * ((K.prob^n) i e : ℂ)) :=
  weighted_summable _ (fun n => ⟨power_nonneg K n i e, power_le_one K n i e⟩) z hz

lemma hit_summable (K : Chain E) (e : E) (z : ℂ) (hz : ‖z‖ < 1) (i : E) :
    Summable (fun n : ℕ => z^n * (hit K e n i : ℂ)) :=
  weighted_summable _ (fun n => hit_bounds K e n i) z hz

lemma hitTransform_self (K : Chain E) (e : E) (z : ℂ) (hz : ‖z‖ < 1) :
    hitTransform K e z e = 1 := by
  unfold hitTransform
  rw [(hit_summable K e z hz e).tsum_eq_zero_add]
  simp [hit]

lemma green_equation (K : Chain E) (e : E) (z : ℂ) (hz : ‖z‖ < 1) (i : E) :
    green K e z i = (if i = e then 1 else 0) +
      z * ∑ j, (K.prob i j : ℂ) * green K e z j := by
  unfold green
  rw [(green_summable K e z hz i).tsum_eq_zero_add]
  congr 1
  · simp [Matrix.one_apply, apply_ite]
  simp only [pow_succ', Matrix.mul_apply, Complex.ofReal_sum, Complex.ofReal_mul]
  simp_rw [Finset.mul_sum]
  rw [Summable.tsum_finsetSum]
  · apply Finset.sum_congr rfl
    intro j _
    calc
      _ = ∑' n : ℕ, (z * (K.prob i j : ℂ)) * (z^n * ((K.prob^n) j e : ℂ)) := by
        apply tsum_congr; intro n; ring
      _ = _ := by rw [tsum_mul_left]; ring
  · intro j _
    convert (green_summable K e z hz j).mul_left (z * (K.prob i j : ℂ)) using 1
    funext n
    ring

lemma hitTransform_equation (K : Chain E) (e : E) (z : ℂ) (hz : ‖z‖ < 1)
    (i : E) (hi : i ≠ e) :
    hitTransform K e z i = z * ∑ j, (K.prob i j : ℂ) * hitTransform K e z j := by
  unfold hitTransform
  rw [(hit_summable K e z hz i).tsum_eq_zero_add]
  simp only [hit, hi, ↓reduceIte, Complex.ofReal_zero, mul_zero, zero_add,
    Complex.ofReal_sum, Complex.ofReal_mul, pow_succ']
  simp_rw [Finset.mul_sum]
  rw [Summable.tsum_finsetSum]
  · apply Finset.sum_congr rfl
    intro j _
    calc
      _ = ∑' n : ℕ, (z * (K.prob i j : ℂ)) * (z^n * (hit K e n j : ℂ)) := by
        apply tsum_congr; intro n; ring
      _ = _ := by rw [tsum_mul_left]; ring
  · intro j _
    convert (hit_summable K e z hz j).mul_left (z * (K.prob i j : ℂ)) using 1
    funext n
    ring

/-- Uniqueness of the discounted Dirichlet equation, by the maximum norm. -/
lemma killed_unique (K : Chain E) (e : E) (z : ℂ) (hz : ‖z‖ < 1)
    (d : E → ℂ) (he : d e = 0)
    (hd : ∀ i, i ≠ e → d i = z * ∑ j, (K.prob i j : ℂ) * d j) :
    ∀ i, d i = 0 := by
  haveI : Nonempty E := ⟨e⟩
  obtain ⟨k, _, hk⟩ := Finset.exists_max_image Finset.univ (fun i => ‖d i‖) Finset.univ_nonempty
  have hb (i : E) : ‖d i‖ ≤ ‖d k‖ := hk i (Finset.mem_univ i)
  have hsum : ‖∑ j, (K.prob k j : ℂ) * d j‖ ≤ ‖d k‖ := by
    calc
      _ ≤ ∑ j, ‖(K.prob k j : ℂ) * d j‖ := norm_sum_le _ _
      _ ≤ ∑ j, K.prob k j * ‖d k‖ := Finset.sum_le_sum (fun j _ => by
        rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (K.nonneg k j)]
        exact mul_le_mul_of_nonneg_left (hb j) (K.nonneg k j))
      _ = ‖d k‖ := by rw [← Finset.sum_mul, K.total, one_mul]
  have hk0 : ‖d k‖ = 0 := by
    by_cases hke : k = e
    · simp [hke, he]
    · have hh : ‖d k‖ ≤ ‖z‖ * ‖d k‖ := by
        calc
          ‖d k‖ = ‖z‖ * ‖∑ j, (K.prob k j : ℂ) * d j‖ := by rw [hd k hke, norm_mul]
          _ ≤ _ := mul_le_mul_of_nonneg_left hsum (norm_nonneg _)
      nlinarith [norm_nonneg (d k)]
  intro i
  apply norm_eq_zero.mp
  exact le_antisymm (by simpa [hk0] using hb i) (norm_nonneg _)

lemma green_eq_hit_mul (K : Chain E) (e : E) (z : ℂ) (hz : ‖z‖ < 1) (i : E) :
    green K e z i = hitTransform K e z i * green K e z e := by
  have hh := killed_unique K e z hz
    (fun j => green K e z j - hitTransform K e z j * green K e z e)
    (by rw [hitTransform_self K e z hz]; ring)
    (by
      intro j hj
      rw [green_equation K e z hz j, hitTransform_equation K e z hz j hj]
      simp only [hj, ↓reduceIte, zero_add, mul_sub, Finset.sum_sub_distrib,
        ← mul_assoc, ← Finset.sum_mul]
      )
  exact sub_eq_zero.mp (hh i)

/-- Renewal solution for the actual Markov-chain return probabilities,
including complex discount factors. No supplied renewal identity. -/
theorem return_series (K : Chain E) (e : E) (z : ℂ) (hz : ‖z‖ < 1) :
    HasSum (fun m : ℕ => z^(m+1) * ((K.prob^(m+1)) e e : ℂ))
      (returnTransform K e z / (1 - returnTransform K e z)) := by
  have hge := green_equation K e z hz e
  simp only [↓reduceIte] at hge
  have hr : green K e z e = 1 + returnTransform K e z * green K e z e := by
    calc
      _ = 1 + z * ∑ j, (K.prob e j : ℂ) * green K e z j := hge
      _ = 1 + z * ∑ j, (K.prob e j : ℂ) *
          (hitTransform K e z j * green K e z e) := by
        congr 2
        apply Finset.sum_congr rfl
        intro j _
        rw [green_eq_hit_mul K e z hz j]
      _ = _ := by simp only [returnTransform, ← mul_assoc, ← Finset.sum_mul]
  have hd : 1 - returnTransform K e z ≠ 0 := by
    intro hd
    have : returnTransform K e z = 1 := by linear_combination -hd
    rw [this, one_mul] at hr
    have : (1 : ℂ) = 0 := by linear_combination -hr
    exact one_ne_zero this
  have heq : green K e z e - 1 = returnTransform K e z / (1 - returnTransform K e z) := by
    apply Coverage3Renewal.renewal_solution _ _ hd
    linear_combination hr
  rw [← heq]
  have hs := (hasSum_nat_add_iff' 1).mpr (green_summable K e z hz e).hasSum
  simpa [green, Matrix.one_apply] using hs

end Dobson.CxA_Renewal

#print axioms Dobson.CxA_Renewal.return_series

namespace Dobson.CxA_Renewal
open Dobson.FiniteLineageVariance MeasureTheory
variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

def edgeDriver (e : E) (ξ : Ω → ℝ) (a : E) : Ω → ℝ :=
  if a = e then ξ else fun _ => 0

lemma lag_edgeDriver (μ : Measure Ω) (T : Ω → Ω) (e : E) (ξ : Ω → ℝ)
    (a b : E) (m : ℕ) :
    lagCovariance μ T (edgeDriver e ξ) a b m =
      if a = e then if b = e then (∫ ω, ξ (T^[m] ω) * ξ ω ∂μ) else 0 else 0 := by
  by_cases ha : a = e <;> by_cases hb : b = e <;>
    simp [lagCovariance, edgeDriver, ha, hb]

/-- Reduction of the actual stationary covariance coefficient to the designated edge. -/
lemma single_edge_coefficient (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (e : E) (ξ : Ω → ℝ) :
    FinitePathCoefficient.covarianceCoefficient μ K T (edgeDriver e ξ) =
      K.initial e ^ 2 / 2 * (∫ ω, ξ ω * ξ ω ∂μ) -
        ∑' m : ℕ, (∫ ω, ξ (T^[m+1] ω) * ξ ω ∂μ) * K.initial e *
          ((K.prob^(m+1)) e e - K.initial e) := by
  unfold FinitePathCoefficient.covarianceCoefficient
  simp only [lag_edgeDriver, Function.iterate_zero, id_eq, mul_ite, ite_mul,
    mul_zero, zero_mul, Finset.sum_ite_irrel, Finset.sum_ite_eq',
    Finset.mem_univ, ↓reduceIte, Finset.sum_const_zero]
  ring

lemma geometric_positive_powers (z : ℂ) (hz : ‖z‖ < 1) :
    HasSum (fun m : ℕ => z^(m+1)) (z / (1-z)) := by
  simpa only [pow_succ', div_eq_mul_inv] using
    (hasSum_geometric_of_norm_lt_one hz).mul_left z

/-- Finite-mixture interchange for an actual edge-return process.
No first-return equation or correlation-sum identity is supplied. -/
lemma mixture_lag_hasSum {J : Type*} (S : Finset J) (B z : J → ℂ)
    (hz : ∀ j ∈ S, ‖z j‖ < 1) (K : Chain E) (e : E) :
    HasSum (fun m : ℕ => (∑ j ∈ S, B j * z j^(m+1)) * (K.initial e : ℂ) *
      (((K.prob^(m+1)) e e : ℂ) - K.initial e))
      ((K.initial e : ℂ) * ∑ j ∈ S, B j *
        (returnTransform K e (z j) / (1-returnTransform K e (z j)) -
          z j / (1-z j) * K.initial e)) := by
  have hj (j : J) (hjs : j ∈ S) :=
    (((return_series K e (z j) (hz j hjs)).sub
      ((geometric_positive_powers (z j) (hz j hjs)).mul_right (K.initial e : ℂ))).mul_left (B j))
  have hs := (hasSum_sum hj).mul_left (K.initial e : ℂ)
  convert hs using 1
  funext m
  simp only [Finset.sum_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- The formerly supplied mixture-cost identity, now for the actual covariance
coefficient of a single fluctuating edge and its genuine first-return PGF. -/
theorem mixture_cost_identity {J : Type*} (S : Finset J) (B z : J → ℂ)
    (hz : ∀ j ∈ S, ‖z j‖ < 1) (hnorm : ∑ j ∈ S, B j = 1)
    (μ : Measure Ω) (K : Chain E) (T : Ω → Ω) (e : E) (ξ : Ω → ℝ)
    (hρ : ∀ m : ℕ, ((∫ ω, ξ (T^[m] ω) * ξ ω ∂μ : ℝ) : ℂ) = ∑ j ∈ S, B j * z j^m) :
    (FinitePathCoefficient.covarianceCoefficient μ K T (edgeDriver e ξ) : ℂ) =
      (K.initial e : ℂ)^2 / 2 - (K.initial e : ℂ) * ∑ j ∈ S, B j *
        (returnTransform K e (z j) / (1-returnTransform K e (z j)) -
          z j / (1-z j) * K.initial e) := by
  have hzero := hρ 0
  simp only [Function.iterate_zero, id_eq, pow_zero, mul_one, hnorm] at hzero
  rw [single_edge_coefficient]
  simp only [Complex.ofReal_sub, Complex.ofReal_mul, Complex.ofReal_div,
    Complex.ofReal_pow, Complex.ofReal_ofNat, Complex.ofReal_tsum, hzero, mul_one]
  have hs := mixture_lag_hasSum S B z hz K e
  have heq : (fun m : ℕ =>
      ((∫ ω, ξ (T^[m+1] ω) * ξ ω ∂μ : ℝ) : ℂ) * K.initial e *
        (((K.prob^(m+1)) e e : ℂ) - K.initial e)) =
      (fun m : ℕ => (∑ j ∈ S, B j * z j^(m+1)) * (K.initial e : ℂ) *
        (((K.prob^(m+1)) e e : ℂ) - K.initial e)) := by
    funext m
    rw [hρ]
  rw [heq, hs.tsum_eq]

/-- The complex-mixture formula of prop:ring-spectrum for the actual directed
covariance coefficient. T_e is the reciprocal stationary edge frequency.
The growth remainder and spectral-integral conclusions are not part of this lemma. -/
theorem mixture_coefficient {J : Type*} (S : Finset J) (B z : J → ℂ)
    (hz : ∀ j ∈ S, ‖z j‖ < 1) (hnorm : ∑ j ∈ S, B j = 1)
    (μ : Measure Ω) (K : Chain E) (T : Ω → Ω) (e : E) (ξ : Ω → ℝ)
    (hc : 0 < K.initial e)
    (hρ : ∀ m : ℕ, ((∫ ω, ξ (T^[m] ω) * ξ ω ∂μ : ℝ) : ℂ) = ∑ j ∈ S, B j * z j^m) :
    (FinitePathCoefficient.covarianceCoefficient μ K T (edgeDriver e ξ) : ℂ) =
      (1 + 2 * ∑ j ∈ S, B j * z j / (1-z j)) /
        (2 * (1 / (K.initial e : ℂ)) ^ 2) -
      (K.initial e : ℂ) * ∑ j ∈ S,
        B j * returnTransform K e (z j) / (1-returnTransform K e (z j)) := by
  have hc' : (K.initial e : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hc.ne'
  have hcost := mixture_cost_identity S B z hz hnorm μ K T e ξ hρ
  have ht : (1 / (K.initial e : ℂ)) ≠ 0 := one_div_ne_zero hc'
  have htbase : (1 : ℂ) / (2 * (1 / (K.initial e : ℂ)) ^ 2) =
      (K.initial e : ℂ)^2 / 2 := by field_simp
  have hdiv (a : ℂ) : a / (1 / (K.initial e : ℂ)) = a * K.initial e := by
    simp [div_eq_mul_inv]
  have h := Coverage3Renewal.mixture_coefficient S B z
    (fun j => returnTransform K e (z j))
    (fun j => returnTransform K e (z j) / (1-returnTransform K e (z j)))
    (fun j => z j / (1-z j)) (1 / (K.initial e : ℂ))
    (FinitePathCoefficient.covarianceCoefficient μ K T (edgeDriver e ξ) : ℂ) ht
    (by simpa only [htbase, one_div_one_div, hdiv, one_mul] using hcost)
    (fun _ _ => rfl) (fun _ _ => rfl)
  simpa only [one_div_one_div] using h

end Dobson.CxA_Renewal

#print axioms Dobson.CxA_Renewal.mixture_cost_identity
#print axioms Dobson.CxA_Renewal.mixture_coefficient

namespace Dobson.CxA_Renewal
open Dobson.FiniteLineageVariance MeasureTheory
variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

noncomputable def realReturn (K : Chain E) (e : E) (φ : ℝ) : ℝ :=
  φ * ∑ j, K.prob e j * ∑' n : ℕ, φ^n * hit K e n j

lemma returnTransform_ofReal (K : Chain E) (e : E) (φ : ℝ) :
    returnTransform K e (φ : ℂ) = (realReturn K e φ : ℂ) := by
  simp only [returnTransform, hitTransform, realReturn, Complex.ofReal_mul,
    Complex.ofReal_sum, Complex.ofReal_tsum, Complex.ofReal_pow]

lemma real_return_series (K : Chain E) (e : E) (φ : ℝ) (hφ : |φ| < 1) :
    HasSum (fun m : ℕ => φ^(m+1) * (K.prob^(m+1)) e e)
      (realReturn K e φ / (1-realReturn K e φ)) := by
  apply Complex.hasSum_ofReal.mp
  simpa only [returnTransform_ofReal, Complex.ofReal_mul, Complex.ofReal_pow,
    Complex.ofReal_div, Complex.ofReal_sub, Complex.ofReal_one] using
      return_series K e (φ : ℂ) (by simpa using hφ)

/-- The actual covariance sum for finitely many independent first-order hosts.
The displayed pair moments are the defining independent AR covariance model,
including their variances; no perturbation-coefficient equation is assumed. -/
lemma weighted_host_covariance {J : Type*} [DecidableEq J] (S : Finset J) (π v φ : J → ℝ)
    (μ : Measure Ω) (T : Ω → Ω) (ξ : J → Ω → ℝ)
    (hint : ∀ i ∈ S, ∀ j ∈ S, ∀ m : ℕ,
      Integrable (fun ω => ξ i (T^[m] ω) * ξ j ω) μ)
    (hpair : ∀ i ∈ S, ∀ j ∈ S, ∀ m : ℕ,
      (∫ ω, ξ i (T^[m] ω) * ξ j ω ∂μ) = if i = j then v i * φ i^m else 0)
    (m : ℕ) :
    (∫ ω, (∑ i ∈ S, π i * ξ i (T^[m] ω)) * (∑ j ∈ S, π j * ξ j ω) ∂μ) =
      ∑ j ∈ S, π j^2 * v j * φ j^m := by
  classical
  have hi (i : J) (his : i ∈ S) (j : J) (hjs : j ∈ S) :
      Integrable (fun ω => (π i * ξ i (T^[m] ω)) * (π j * ξ j ω)) μ := by
    convert (hint i his j hjs m).const_mul (π i * π j) using 1
    funext ω; ring
  simp_rw [Finset.sum_mul, Finset.mul_sum]
  rw [integral_finsetSum _ (fun i his => integrable_finsetSum S (fun j hjs => hi i his j hjs))]
  apply Finset.sum_congr rfl
  intro i his
  rw [integral_finsetSum _ (fun j hjs => hi i his j hjs)]
  have hterm (j : J) :
      (∫ ω, (π i * ξ i (T^[m] ω)) * (π j * ξ j ω) ∂μ) =
        (π i * π j) * ∫ ω, ξ i (T^[m] ω) * ξ j ω ∂μ := by
    rw [← integral_const_mul]
    congr 1; funext ω; ring
  simp_rw [hterm]
  rw [Finset.sum_congr rfl (fun j hjs => by rw [hpair i his j hjs m])]
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq, his, ↓reduceIte]
  ring

/-- The independent-host per-step coefficient from the actual edge-return law.
The shares can vanish; the memory may be negative, zero, or positive. -/
theorem alternative_coefficient {J : Type*} [DecidableEq J] (S : Finset J) (π v φ : J → ℝ)
    (hφ : ∀ j ∈ S, |φ j| < 1)
    (μ : Measure Ω) (K : Chain E) (T : Ω → Ω) (e : E) (ξ : J → Ω → ℝ)
    (hc : 0 < K.initial e)
    (hint : ∀ i ∈ S, ∀ j ∈ S, ∀ m : ℕ,
      Integrable (fun ω => ξ i (T^[m] ω) * ξ j ω) μ)
    (hpair : ∀ i ∈ S, ∀ j ∈ S, ∀ m : ℕ,
      (∫ ω, ξ i (T^[m] ω) * ξ j ω ∂μ) = if i = j then v i * φ i^m else 0) :
    FinitePathCoefficient.covarianceCoefficient μ K T
      (edgeDriver e (fun ω => ∑ j ∈ S, π j * ξ j ω)) =
      ∑ j ∈ S, π j^2 * v j *
        ((1 + 2 * (φ j / (1-φ j))) / (2 * (1 / K.initial e)^2) -
          1 / (1 / K.initial e) * (realReturn K e (φ j) / (1-realReturn K e (φ j)))) := by
  have hρ := weighted_host_covariance S π v φ μ T ξ hint hpair
  have hj (j : J) (hjs : j ∈ S) :
      HasSum (fun m : ℕ => π j^2 * v j * φ j^(m+1) * K.initial e *
        ((K.prob^(m+1)) e e - K.initial e))
        (π j^2 * v j * K.initial e *
          (realReturn K e (φ j) / (1-realReturn K e (φ j)) -
            φ j / (1-φ j) * K.initial e)) := by
    have hg : HasSum (fun m : ℕ => φ j^(m+1)) (φ j / (1-φ j)) := by
      simpa only [pow_succ', div_eq_mul_inv] using
        (hasSum_geometric_of_abs_lt_one (hφ j hjs)).mul_left (φ j)
    have hh := (((real_return_series K e (φ j) (hφ j hjs)).sub
      (hg.mul_right (K.initial e))).mul_left (π j^2 * v j * K.initial e))
    convert hh using 1
    funext m; ring
  have hs := hasSum_sum hj
  have hlag : (∑' m : ℕ, (∫ ω,
      (∑ j ∈ S, π j * ξ j (T^[m+1] ω)) * (∑ j ∈ S, π j * ξ j ω) ∂μ) *
        K.initial e * ((K.prob^(m+1)) e e - K.initial e)) =
      ∑ j ∈ S, π j^2 * v j * K.initial e *
        (realReturn K e (φ j) / (1-realReturn K e (φ j)) - φ j / (1-φ j) * K.initial e) := by
    simp_rw [hρ, Finset.sum_mul]
    exact hs.tsum_eq
  rw [single_edge_coefficient]
  have hz : (∫ ω, (∑ j ∈ S, π j * ξ j ω) * (∑ j ∈ S, π j * ξ j ω) ∂μ) =
      ∑ j ∈ S, π j^2 * v j := by simpa using hρ 0
  rw [hz, hlag, Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  have hd : 1 / (2 * (1 / K.initial e)^2) = K.initial e^2 / 2 := by field_simp
  rw [div_eq_mul_one_div (1+_), hd, one_div_one_div]
  ring

/-- Apply the existing generation-time conversion with its expansion premise
proved from the independent-host covariance model and actual first returns. -/
theorem alternative_loss {J : Type*} [DecidableEq J] (S : Finset J) (π v φ : J → ℝ)
    (hφ : ∀ j ∈ S, |φ j| < 1)
    (μ : Measure Ω) (K : Chain E) (T : Ω → Ω) (e : E) (ξ : J → Ω → ℝ)
    (hc : 0 < K.initial e)
    (hint : ∀ i ∈ S, ∀ j ∈ S, ∀ m : ℕ,
      Integrable (fun ω => ξ i (T^[m] ω) * ξ j ω) μ)
    (hpair : ∀ i ∈ S, ∀ j ∈ S, ∀ m : ℕ,
      (∫ ω, ξ i (T^[m] ω) * ξ j ω ∂μ) = if i = j then v i * φ i^m else 0) :
    (1 / K.initial e) * FinitePathCoefficient.covarianceCoefficient μ K T
      (edgeDriver e (fun ω => ∑ j ∈ S, π j * ξ j ω)) =
      ∑ j ∈ S, π j^2 * v j *
        ((1 + 2 * (φ j / (1-φ j))) / (2 * (1 / K.initial e)) -
          realReturn K e (φ j) / (1-realReturn K e (φ j))) := by
  exact Coverage2Alternatives.alternative_loss S π v (fun j => φ j / (1-φ j))
    (fun j => realReturn K e (φ j)) (1 / K.initial e) _
    (one_div_ne_zero hc.ne') (alternative_coefficient S π v φ hφ μ K T e ξ hc hint hpair)

end Dobson.CxA_Renewal

#print axioms Dobson.CxA_Renewal.alternative_coefficient
#print axioms Dobson.CxA_Renewal.alternative_loss

namespace Dobson.CxA_Renewal
open Dobson.FiniteLineageVariance MeasureTheory Filter
open scoped Topology

/-- Connect the coefficient used above to actual finite expected path growth.
This is a limit of derivatives, not a differentiation of the growth-rate limit.
The displayed stationary L2 and geometric-mixing controls are the same as in
the existing finite-path theorem; no growth remainder is inferred from them. -/
theorem coefficient_is_actual_second_variation_limit
    {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (K : Chain E)
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (f : E → Ω → ℝ)
    (hf : ∀ e, MemLp (f e) 2 μ)
    (B Ku β : ℝ) (hB : 0 ≤ B) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hcov : ∀ e e' m, |lagCovariance μ T f e e' m| ≤ B)
    (hmix : ∀ e e' m, |(K.prob^(m+1)) e' e-K.initial e| ≤ Ku*β^m)
    (L : ℝ) :
    Tendsto (fun N => -iteratedDeriv 2
      (FinitePathJet.finiteGrowth μ K T f L .lognormal N) 0 / 2)
      atTop (𝓝 (FinitePathCoefficient.covarianceCoefficient μ K T f)) := by
  have h := FinitePathCoefficient.finiteCoefficient_tendsto μ K T hT f hf
    B Ku β hB hKu hβ0 hβ1 hcov hmix
  convert h using 1
  funext N
  rw [FinitePathJet.finiteGrowth_second μ K T hT f hf L .lognormal (by simp)]
  ring

end Dobson.CxA_Renewal

#print axioms Dobson.CxA_Renewal.coefficient_is_actual_second_variation_limit
