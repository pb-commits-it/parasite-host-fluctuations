import DobsonProofs.ParallelNetwork
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic
import Mathlib.Analysis.Normed.Operator.NormedSpace

/-!
# Coverage batch 3: sheltered kernels, feedback, and routes
Cores of `lem:mech-kernel`, `prop:dyn-k9`, and `prop:mech-routes`.
Minimal-realisation similarity and the bounded log-norm comparison are not
constructed. Feedback is derived from the reduced closure equation. Pointwise
filter limits do not assert limits of spectral integrals. Route coefficients
retain their local perturbation/threshold inputs explicitly.
-/
/-! ## Coverage correction — 2026-11-22

`lem:mech-kernel` and `prop:dyn-k9` are core. Similarity of finite products,
bounded log-norm comparison and finite loop closure are proved. General
kernel realization and spectral-integral limits remain outside credit.
`prop:mech-routes` is partial-P1: the parallel coefficient is computed from
its reference chain, but the restoring-threshold bridge and credited bypass
per-step cost remain supplied. An uncredited bypass theorem does not change
that inventory grade. -/

/-! 2026-11-16, leanp1: the norm comparison is now proved for products
conjugated by an actual continuous linear equivalence. Zero products are
handled on both sides using Lean's real-logarithm convention; no claim about
extended-real negative-infinite exponents follows. The construction of that
equivalence from equal minimal sheltered realizations is still outside this
component. The feedback theorem now derives its loop closure by induction
from the local equations of a nonempty finite cycle and its forced input.
Neither public theorem takes its former comparison/closure as a premise. -/

namespace Dobson.Coverage3Networks
open scoped BigOperators
open Filter

/-- Sheltered transfer kernels are invariant under all stage permutations. -/
theorem kernel_permutation {ι : Type*} [Fintype ι] (e : ι ≃ ι) (g s : ι → ℂ) (z : ℂ) :
    (∏ i, g (e i) * z / (1 - s (e i) * z)) = ∏ i, g i * z / (1 - s i * z) :=
  e.prod_comp (fun i => g i * z / (1 - s i * z))

/-- A fixed similarity conjugates every finite ordered product, including matrices. -/
theorem conjugate_product {R : Type*} [Monoid R] (S : Rˣ) (L : List R) :
    (L.map (fun A => (S : R) * A * (↑S⁻¹ : R))).prod =
      (S : R) * L.prod * (↑S⁻¹ : R) := by
  induction L with
  | nil => simp
  | cons A L ih =>
    simp only [List.map_cons, List.prod_cons, ih]
    simp [mul_assoc]

/-- Private scalar helper: a supplied uniformly bounded log-norm
difference preserves an existing growth limit. The public similarity result
below derives the comparison for actual conjugated products. -/
private theorem bounded_log_growth_of_bound (a b : ℕ → ℝ) (C r : ℝ)
    (hcomparison : ∀ n, |b n - a n| ≤ C)
    (hlimit : Tendsto (fun n : ℕ => a n / n) atTop (nhds r)) :
    Tendsto (fun n : ℕ => b n / n) atTop (nhds r) := by
  have hz : Tendsto (fun n : ℕ => (b n - a n) / n) atTop (nhds 0) := by
    apply squeeze_zero_norm _ (tendsto_const_div_atTop_nhds_zero_nat C)
    intro n
    rw [Real.norm_eq_abs, abs_div, abs_of_nonneg (Nat.cast_nonneg n : (0 : ℝ) ≤ (n : ℝ))]
    exact div_le_div_of_nonneg_right (hcomparison n) (Nat.cast_nonneg n)
  have h := hlimit.add hz
  have hfun : (fun n : ℕ => a n / n + (b n - a n) / n) = (fun n : ℕ => b n / n) := by
    funext n
    ring
  rw [hfun, add_zero] at h
  exact h

/-- Fixed similarity gives a uniform log-norm bound on nonzero operators. -/
private theorem conjugate_log_norm_bound_ne_zero {R : Type*} [NormedRing R] [NormOneClass R]
    [Nontrivial R] (S : Rˣ) (x : R) (hx : x ≠ 0) :
    |Real.log ‖(S : R) * x * (↑S⁻¹ : R)‖ - Real.log ‖x‖| ≤
      Real.log (‖(S : R)‖ * ‖(↑S⁻¹ : R)‖) := by
  let C := ‖(S : R)‖ * ‖(↑S⁻¹ : R)‖
  have hC : 0 < C := mul_pos (norm_pos_iff.mpr S.ne_zero)
    (norm_pos_iff.mpr S⁻¹.ne_zero)
  have hnorm (U : Rˣ) (y : R) :
      ‖(U : R) * y * (↑U⁻¹ : R)‖ ≤ ‖(U : R)‖ * ‖(↑U⁻¹ : R)‖ * ‖y‖ := by
    calc
      _ ≤ ‖(U : R) * y‖ * ‖(↑U⁻¹ : R)‖ := norm_mul_le _ _
      _ ≤ (‖(U : R)‖ * ‖y‖) * ‖(↑U⁻¹ : R)‖ :=
        mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _)
      _ = _ := by ring
  have hinverse : (↑S⁻¹ : R) * ((S : R) * x * (↑S⁻¹ : R)) * (S : R) = x := by
    simp [mul_assoc]
  have hy : (S : R) * x * (↑S⁻¹ : R) ≠ 0 := by
    intro h
    rw [h] at hinverse
    exact hx (by simpa using hinverse.symm)
  have hupper := Real.log_le_log (norm_pos_iff.mpr hy) (hnorm S x)
  rw [Real.log_mul hC.ne' (norm_pos_iff.mpr hx).ne'] at hupper
  have hlower := hnorm S⁻¹ ((S : R) * x * (↑S⁻¹ : R))
  simp only [inv_inv] at hlower
  rw [hinverse] at hlower
  rw [mul_comm ‖(↑S⁻¹ : R)‖ ‖(S : R)‖] at hlower
  have hloglower := Real.log_le_log (norm_pos_iff.mpr hx) hlower
  rw [Real.log_mul hC.ne' (norm_pos_iff.mpr hy).ne'] at hloglower
  exact abs_le.mpr ⟨by linarith, by linarith⟩

/-- The bound includes zero operators: conjugation preserves zero, so both
real logarithms are zero under Lean's convention. -/
theorem conjugate_log_norm_bound {R : Type*} [NormedRing R] [NormOneClass R]
    [Nontrivial R] (S : Rˣ) (x : R) :
    |Real.log ‖(S : R) * x * (↑S⁻¹ : R)‖ - Real.log ‖x‖| ≤
      Real.log (‖(S : R)‖ * ‖(↑S⁻¹ : R)‖) := by
  by_cases hx : x = 0
  · have hC : 1 ≤ ‖(S : R)‖ * ‖(↑S⁻¹ : R)‖ := by
      calc
        _ = ‖(S : R) * (↑S⁻¹ : R)‖ := by simp
        _ ≤ _ := norm_mul_le _ _
    simpa [hx] using Real.log_nonneg hC
  · exact conjugate_log_norm_bound_ne_zero S x hx

/-- The actual conjugate finite products have the same real-valued growth limit.
Zero products use `Real.log 0 = 0` on both sides; this is not a claim about
extended-real negative-infinite exponents. The coordinate change is an actual continuous linear equivalence. Constructing
it from equal sheltered kernels is a separate minimal-realization step. -/
theorem bounded_log_growth {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    [Nontrivial V] (S : V ≃L[ℝ] V) (L : ℕ → List (V →L[ℝ] V)) (r : ℝ)
    (hlimit : Tendsto (fun n : ℕ => Real.log ‖(L n).prod‖ / n) atTop (nhds r)) :
    Tendsto (fun n : ℕ => Real.log ‖((L n).map (fun A =>
      (S.toUnit : V →L[ℝ] V) * A * (↑S.toUnit⁻¹ : V →L[ℝ] V))).prod‖ / n)
      atTop (nhds r) := by
  apply bounded_log_growth_of_bound _ _
    (Real.log (‖(S.toUnit : V →L[ℝ] V)‖ * ‖(↑S.toUnit⁻¹ : V →L[ℝ] V)‖)) r _ hlimit
  intro n
  rw [conjugate_product]
  exact conjugate_log_norm_bound S.toUnit (L n).prod

/-- Product of block transfers before the selected block of a finite cycle. -/
noncomputable def prefixTransfer {n : ℕ} (E : Fin (n + 1) → ℂ)
    (k : Fin (n + 1)) : ℂ :=
  ∏ i : Fin k.val, E ⟨i.val, lt_trans i.isLt k.isLt⟩

lemma prefixTransfer_zero {n : ℕ} (E : Fin (n + 1) → ℂ) :
    prefixTransfer E 0 = 1 := by
  unfold prefixTransfer
  exact Fin.prod_univ_zero _

lemma prefixTransfer_succ {n : ℕ} (E : Fin (n + 1) → ℂ) (k : Fin n) :
    prefixTransfer E k.succ = prefixTransfer E k.castSucc * E k.castSucc := by
  simp only [prefixTransfer, Fin.val_succ, Fin.val_castSucc, Fin.prod_univ_castSucc]
  rfl

/-- Local block equations propagate the input through each prefix of the cycle. -/
theorem cycle_propagation {n : ℕ} (E x : Fin (n + 1) → ℂ)
    (hstep : ∀ i : Fin n, x i.succ = E i.castSucc * x i.castSucc)
    (k : Fin (n + 1)) : x k = prefixTransfer E k * x 0 := by
  induction k using Fin.induction with
  | zero => simp [prefixTransfer_zero]
  | succ k ih => rw [hstep k, ih, prefixTransfer_succ]; ring

/-- Derive the loop closure from the individual block equations and the one
forced input, then obtain the actual response at every block. The indexing
starts at the forced block and includes the one-block cycle. -/
theorem feedback_response {n : ℕ} (E x : Fin (n + 1) → ℂ) (η F : ℂ)
    (hstep : ∀ i : Fin n, x i.succ = E i.castSucc * x i.castSucc)
    (hforced : x 0 = η + E (Fin.last n) * x (Fin.last n))
    (hden : 1 - ∏ i, E i ≠ 0) (k : Fin (n + 1)) :
    F * x k = F * prefixTransfer E k * η / (1 - ∏ i, E i) := by
  have hprod : (∏ i, E i) = prefixTransfer E (Fin.last n) * E (Fin.last n) := by
    rw [Fin.prod_univ_castSucc]
    rfl
  have hclosure : x 0 = η + (∏ i, E i) * x 0 := by
    rw [cycle_propagation E x hstep (Fin.last n)] at hforced
    rw [hprod]
    linear_combination hforced
  have hx : x 0 = η / (1 - ∏ i, E i) := by
    apply (eq_div_iff hden).mpr
    linear_combination hclosure
  rw [cycle_propagation E x hstep k, hx]
  ring

/-- At each nonzero frequency, a slow included stage suppresses the response;
a bypass retains the open-path response. These are pointwise filter limits. -/
theorem slow_and_bypass_limits (z P G : ℂ) (hz : z ≠ 0) :
    Tendsto (fun a : ℂ => a * P / (z + a * (1 - G))) (nhds 0) (nhds 0) ∧
    Tendsto (fun a : ℂ => (z + a) * P / (z + a * (1 - G))) (nhds 0) (nhds P) := by
  have hc1 : ContinuousAt (fun a : ℂ => a * P / (z + a * (1 - G))) 0 := by
    fun_prop (disch := simpa using hz)
  have hc2 : ContinuousAt (fun a : ℂ => (z + a) * P / (z + a * (1 - G))) 0 := by
    fun_prop (disch := simpa using hz)
  constructor
  · simpa using hc1.tendsto
  · simpa [hz] using hc2.tendsto

/-- The S/L static feedback gain is exactly R0/(R0-1), and positive. -/
theorem static_feedback (R0 P : ℝ) (hR : 1 < R0) :
    P / (1 - 1 / R0) = R0 * P / (R0 - 1) ∧ 0 < 1 - 1 / R0 := by
  have h0 : 0 < R0 := by linarith
  constructor
  · field_simp
  · have := (div_lt_one h0).mpr hR
    linarith

/-- The actual parallel-network coefficient: entrance scores are carried only by
central-to-host edges. Arbitrary simultaneous host covariance and zero route
shares are allowed. The coefficient-to-threshold identification remains the
explicit owed37 input; this theorem does not assert the growth expansion62. -/
theorem parallel_threshold {J Ω : Type*} [Fintype J] [DecidableEq J]
    [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (S : Ω ≃ᵐ Ω) (hS : MeasureTheory.MeasurePreserving S μ μ)
    (τ0 : ℝ) (τ w κ : J → ℝ) (ξ : J → Ω → ℝ)
    (hτ0 : 1 ≤ τ0) (hτ : ∀ j, 1 ≤ τ j) (hw : ∀ j, 0 ≤ w j)
    (hwsum : ∑ j, w j = 1) (hξ : ∀ j, MeasureTheory.MemLp (ξ j) 2 μ)
    (hlag : ∀ i j m, Dobson.FiniteLineageVariance.lagCovariance μ S ξ i j (m+1) = 0)
    (σ Δ : ℝ)
    (hthreshold : Δ = Dobson.ParallelReference.quadraticCost μ τ0 τ w κ ξ hτ0 hτ hw hwsum S σ /
      (∑ j, Dobson.ParallelReference.stationaryMass τ0 τ w (some j) *
        Dobson.ParallelReference.transition τ0 τ w (some j) none)) :
    Δ = σ^2*Dobson.ParallelReference.routeVariance μ κ w ξ /
      (2*Dobson.ParallelReference.totalTime τ0 τ w) := by
  rw [hthreshold, Dobson.ParallelReference.birth_frequency τ0 τ w hτ hwsum,
    Dobson.ParallelReference.quadraticCost_eq μ S hS τ0 τ w κ ξ hτ0 hτ hw hwsum hξ hlag]
  have hT := (Dobson.ParallelReference.totalTime_pos τ0 τ w hτ0 hτ hw).ne'
  field_simp

/-- Exact newcomer criterion, including the changed generation time. -/
theorem newcomer_iff (f T V C U dτ : ℝ) (hf : 0 < f) (hT : 0 < T)
    (hnew : 0 < T + f * dτ) :
    ((1 - f) ^ 2 * V + 2 * f * (1 - f) * C + f ^ 2 * U) / (T + f * dτ) < V / T ↔
      f * T * (V - 2 * C + U) < 2 * T * (V - C) + V * dτ := by
  rw [div_lt_div_iff₀ hnew hT, ← sub_neg]
  have hid : ((1 - f) ^ 2 * V + 2 * f * (1 - f) * C + f ^ 2 * U) * T - V * (T + f * dτ) =
      f * (f * T * (V - 2 * C + U) - (2 * T * (V - C) + V * dτ)) := by ring
  rw [hid]
  constructor
  · intro h
    by_contra hn
    have := mul_nonneg hf.le (sub_nonneg.mpr (le_of_not_gt hn))
    linarith
  · intro h
    exact mul_neg_of_pos_of_neg hf (sub_neg.mpr h)

/-- Optional-route normalization gives the mean stay and return transform. -/
theorem optional_mixture (f q τ0 τ1 τp q0 q1 qp : ℝ)
    (hY : 1 - f + f * q ≠ 0) :
    ((1 - f) * (τ0 + τ1) + f * q * (τ0 + τ1 + τp)) / (1 - f + f * q) =
      τ0 + τ1 + (f * q / (1 - f + f * q)) * τp ∧
    q0 * q1 * ((1 - f) + f * q * qp) / (1 - f + f * q) =
      q0 * q1 * (1 - f * q / (1 - f + f * q) + f * q / (1 - f + f * q) * qp) := by
  constructor <;> field_simp <;> ring

/-- Bypass-route normalization gives the shorter mean stay and its return transform. -/
theorem bypass_mixture (f q τ0 τ1 q0 q1 : ℝ)
    (hY : 1 - f + f * q ≠ 0) :
    ((1 - f) * (τ0 + τ1) + f * q * τ0) / (1 - f + f * q) =
      τ0 + ((1 - f) / (1 - f + f * q)) * τ1 ∧
    ((1 - f) * q0 * q1 + f * q * q0) / (1 - f + f * q) =
      q0 * (1 - (1 - f) / (1 - f + f * q) + (1 - f) / (1 - f + f * q) * q1) := by
  constructor <;> field_simp <;> ring

/-- The bypass per-step coefficient has the stated generation normalization. -/
theorem bypass_coefficient (T α θ q0 q1 R C : ℝ) (hT : T ≠ 0)
    (hcost : C = α ^ 2 * ((1 + 2 * θ) / (2 * T ^ 2) - q0 * q1 / (T * (1 - R)))) :
    T * C = α ^ 2 * ((1 + 2 * θ) / (2 * T) - q0 * q1 / (1 - R)) := by
  rw [hcost]
  by_cases hR : 1 - R = 0
  · simp only [hR, mul_zero, div_zero, sub_zero]
    field_simp
  · field_simp

/-- Exact fixed-rate advantage when the route multiplies mean reproduction by Y. -/
theorem fixed_rate_advantage (R Y Δbase Δroute : ℝ) (hR : 0 < R) (hY : 0 < Y) :
    (Real.log (R * Y) - Δroute) - (Real.log R - Δbase) =
      Real.log Y + Δbase - Δroute := by
  rw [Real.log_mul (ne_of_gt hR) (ne_of_gt hY)]
  ring

end Dobson.Coverage3Networks

#print axioms Dobson.Coverage3Networks.conjugate_log_norm_bound
#print axioms Dobson.Coverage3Networks.bounded_log_growth
#print axioms Dobson.Coverage3Networks.cycle_propagation
#print axioms Dobson.Coverage3Networks.feedback_response

/-! 2026-11-16, leanp1: part2:36 hcost is proved from the actual stationary
parallel edge chain and finite expected log-growth second derivative. Zero
shares and arbitrary same-time covariance are allowed. Owed37 and shared62
(the threshold map and full growth expansion) remain explicit. -/
