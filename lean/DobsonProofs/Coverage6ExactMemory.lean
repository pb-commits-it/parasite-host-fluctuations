import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Tactic
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-! # Exact-memory coefficient (`prop:exact-mem-coefficient`)

Coverage correction, 2026-11-22: this row is core. The stationary variance
and cross-moment equations are derived from an actual square-integrable
stationary linearized recurrence, so the former `hV` and `hK` premises are
discharged. Covariance bounds prove absolute convergence of the coefficient
series, and the geometric specialization and rational signs are checked.

Construction from the nonlinear Gaussian dynamics, identification of the
physical growth expansion and its remainder, and the full turning-point
conclusions remain outside the credited scope. -/
namespace Dobson.Coverage6ExactMemory
open Filter
open scoped Topology

/-- Absolute convergence needs only the covariance bound, not summability. -/
theorem covariance_series_summable (q : ℝ) (hq : |q| < 1)
    (ρ : ℕ → ℝ) (hρ : ∀ j, |ρ j| ≤ 1) :
    Summable (fun j : ℕ => (-q)^j * ρ (j+1)) := by
  apply Summable.of_norm_bounded (summable_geometric_of_abs_lt_one (show |(|q|)| < 1 by simpa using hq))
  intro j
  rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_neg]
  exact mul_le_of_le_one_right (pow_nonneg (abs_nonneg q) _) (hρ _)

/-- Exact autoregressive lag sum, including negative memory. -/
theorem ar_lag_sum (q φ : ℝ) (h : |q * φ| < 1) :
    HasSum (fun j : ℕ => (-q)^j * φ^(j+1)) (φ / (1+q*φ)) := by
  have hg := (hasSum_geometric_of_abs_lt_one (show |(-q)*φ| < 1 by
    simpa only [neg_mul, abs_neg] using h)).mul_left φ
  convert hg using 1
  · ext j
    rw [pow_succ, mul_pow]
    ring
  · field_simp
    <;> ring

noncomputable def arCoeff (s φ : ℝ) : ℝ :=
  (1-s) * (1+(3-s)*φ) / (2*(2-s)*(1+(1-s)*φ))

/-- Summing the covariance expression gives the manuscript's rational function. -/
theorem ar_coefficient (s φ : ℝ) (hs : 0 < s) (hs1 : s < 1)
    (hφ : |φ| < 1) :
    (1-s)/(2*(2-s)) * (1+2*∑' j : ℕ, (-(1-s))^j * φ^(j+1)) =
      arCoeff s φ := by
  have hprod : |(1-s)*φ| < 1 := by
    rw [abs_mul, abs_of_pos (by linarith : 0 < 1-s)]
    nlinarith [abs_nonneg φ]
  rw [(ar_lag_sum (1-s) φ hprod).tsum_eq]
  have hD : 1+(1-s)*φ ≠ 0 := by
    have := (abs_lt.mp hprod).1
    linarith
  unfold arCoeff
  field_simp
  <;> ring

/-- The derivative is calculated, not supplied as an assumption. -/
theorem ar_coefficient_derivative (s φ : ℝ) (hs2 : s ≠ 2)
    (hD : 1+(1-s)*φ ≠ 0) :
    HasDerivAt (fun t => arCoeff t φ)
      (-(1+2*φ+2*(1-s)*φ-(1-s)^2*φ^2) /
        (2*(2-s)^2*(1+(1-s)*φ)^2)) s := by
  have hq := (hasDerivAt_id s).const_sub 1
  have hthree := (hasDerivAt_id s).const_sub 3
  have htwo := (hasDerivAt_id s).const_sub 2
  have hn := hq.mul ((hthree.mul_const φ).const_add 1)
  have hd := (htwo.const_mul 2).mul ((hq.mul_const φ).const_add 1)
  have hd0 : 2*(2-s)*(1+(1-s)*φ) ≠ 0 := by
    exact mul_ne_zero (mul_ne_zero (by norm_num) (sub_ne_zero.mpr hs2.symm)) hD
  convert hn.div hd hd0 using 1 <;> try rfl
  dsimp only [id_eq, Pi.mul_apply]
  field_simp [hD, sub_ne_zero.mpr hs2.symm]
  <;> ring

/-- Sign of the coefficient, with exactly the interior model parameters. -/
theorem ar_coefficient_negative_iff (s φ : ℝ) (hs : 0 < s) (hs1 : s < 1)
    (hφ : -1 < φ) : arCoeff s φ < 0 ↔ φ < -1/(3-s) := by
  have hq : 0 < 1-s := by linarith
  have hD : 0 < 1+(1-s)*φ := by nlinarith
  have hs2 : 0 < 2-s := by linarith
  unfold arCoeff
  rw [div_lt_iff₀ (by positivity : 0 < 2*(2-s)*(1+(1-s)*φ)),
    zero_mul, lt_div_iff₀ (by linarith : 0 < 3-s)]
  have he : (1-s)*(1+(3-s)*φ) < 0 ↔ 1+(3-s)*φ < 0 := by
    simpa using (mul_lt_mul_iff_right₀ hq (b := 1+(3-s)*φ) (c := 0))
  rw [he]
  constructor <;> intro h <;> nlinarith

/-- Derivative signs for all nonnegative memories and all φ≤-1/2.
The intermediate sign changes are equivalently the exact polynomial criterion
in `ar_coefficient_derivative`. -/
theorem ar_derivative_extreme_signs (q φ : ℝ) (hq : 0 < q) (hq1 : q < 1) :
    (0 ≤ φ → φ < 1 → 0 < 1+2*φ+2*q*φ-q^2*φ^2) ∧
    (-1 < φ → φ ≤ -1/2 → 1+2*φ+2*q*φ-q^2*φ^2 < 0) := by
  constructor
  · intro hp hp1
    have hqp : 0 ≤ q*φ := mul_nonneg hq.le hp
    have hqp1 : q*φ < 1 := by nlinarith
    nlinarith [sq_nonneg (q*φ), mul_nonneg hqp (by linarith : 0 ≤ 1-q*φ)]
  · intro hp hp2
    have hprod : q*φ < 0 := mul_neg_of_pos_of_neg hq (by linarith)
    nlinarith [sq_nonneg (q*φ)]

/-- Exact coefficient reduction from the second-moment stationarity equations.
`K` and `V` represent the lag covariance and variance; only the lower-level
moment recursions are assumed, rather than the target coefficient formula. -/
private theorem moment_reduction_from_equations (q S K V : ℝ) (hq : q ≠ 0)
    (hq1 : q ≠ 1) (hqm : q ≠ -1)
    (hK : K=q*S) (hV : (1-q^2)*V=q^2*(1-2*q*S)) :
    (1-q)*V/(2*q)+K = q/(2*(1+q))*(1+2*S) := by
  have hden : 1-q^2 ≠ 0 := by
    intro h
    have hfac : (q-1)*(q+1)=0 := by nlinarith
    rcases mul_eq_zero.mp hfac with h | h <;> [apply hq1; apply hqm] <;> linarith
  have hv : V=q^2*(1-2*q*S)/(1-q^2) := (eq_div_iff hden).mpr (by nlinarith)
  rw [hK, hv]
  have hsum : 1+q ≠ 0 := by intro h; apply hqm; linarith
  field_simp [hq, hden, hsum]
  <;> ring

open MeasureTheory

/-- The actual stationary linearized recurrence implies its second-moment
identity. The driver has unit second moment; no variance equation is assumed. -/
theorem stationary_variance_equation {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (ξ X : Ω → ℝ) (q : ℝ)
    (hξ : MemLp ξ 2 μ) (hX : MemLp X 2 μ)
    (hunit : ∫ ω, (ξ ω) ^ 2 ∂μ = 1)
    (hrec : ∀ᵐ ω ∂μ, X (T ω) = q * (ξ ω - X ω)) :
    (1 - q ^ 2) * (∫ ω, (X ω) ^ 2 ∂μ) =
      q ^ 2 * (1 - 2 * ∫ ω, ξ ω * X ω ∂μ) := by
  have hiξ := hξ.integrable_sq
  have hiX := hX.integrable_sq
  have hiξX : Integrable (fun ω => ξ ω * X ω) μ := hξ.integrable_mul hX
  have hiD : Integrable (fun ω => (ξ ω) ^ 2 - 2 * (ξ ω * X ω)) μ :=
    hiξ.sub (hiξX.const_mul 2)
  have hshift : (∫ ω, (X (T ω)) ^ 2 ∂μ) = ∫ ω, (X ω) ^ 2 ∂μ :=
    hT.integral_comp' (fun ω => (X ω) ^ 2)
  have he : (∫ ω, (X (T ω)) ^ 2 ∂μ) =
      q ^ 2 * ((∫ ω, (ξ ω) ^ 2 ∂μ) -
        2 * (∫ ω, ξ ω * X ω ∂μ) + (∫ ω, (X ω) ^ 2 ∂μ)) := by
    calc
      _ = ∫ ω, q ^ 2 * ((ξ ω) ^ 2 - 2 * (ξ ω * X ω) + (X ω) ^ 2) ∂μ := by
        apply integral_congr_ae
        filter_upwards [hrec] with ω hω
        rw [hω]
        ring
      _ = _ := by
        rw [integral_const_mul,
          integral_add hiD hiX,
          integral_sub hiξ (hiξX.const_mul 2), integral_const_mul]
  rw [hshift, hunit] at he
  nlinarith

/-- Young's inequality gives a uniform bound for an L² cross moment. -/
private theorem integral_mul_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (f g : Ω → ℝ) (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) :
    |∫ ω, f ω * g ω ∂μ| ≤ ((∫ ω, (f ω)^2 ∂μ) + (∫ ω, (g ω)^2 ∂μ)) / 2 := by
  have hm : Integrable (fun ω => f ω * g ω) μ := hf.integrable_mul hg
  calc
    _ ≤ ∫ ω, |f ω * g ω| ∂μ := abs_integral_le_integral_abs
    _ ≤ ∫ ω, ((f ω)^2 + (g ω)^2) / 2 ∂μ := by
      apply integral_mono hm.abs ((hf.integrable_sq.add hg.integrable_sq).div_const 2)
      intro ω
      change |f ω * g ω| ≤ ((f ω)^2 + (g ω)^2) / 2
      rw [abs_mul]
      nlinarith [sq_nonneg (|f ω| - |g ω|), sq_abs (f ω), sq_abs (g ω)]
    _ = _ := by rw [integral_div, integral_add hf.integrable_sq hg.integrable_sq]

/-- Forward covariance of a stationary driver. -/
noncomputable def lagCovariance {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (ξ : Ω → ℝ) (n : ℕ) : ℝ :=
  ∫ ω, ξ ((T : Ω → Ω)^[n] ω) * ξ ω ∂μ

private theorem shifted_integral {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : Ω → ℝ) (n : ℕ) :
    (∫ ω, f ((T : Ω → Ω)^[n] ω) ∂μ) = ∫ ω, f ω ∂μ := by
  induction n with
  | zero => simp
  | succ n ih =>
    simpa only [Function.iterate_succ_apply] using
      (hT.integral_comp' (fun ω => f ((T : Ω → Ω)^[n] ω))).trans ih

/-- Unit second moment bounds every covariance, independently of covariance
summability. -/
theorem lagCovariance_abs_le_one {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (ξ : Ω → ℝ) (hξ : MemLp ξ 2 μ) (hunit : ∫ ω, (ξ ω)^2 ∂μ = 1) (n : ℕ) :
    |lagCovariance μ T ξ n| ≤ 1 := by
  have hs : MemLp (fun ω => ξ ((T : Ω → Ω)^[n] ω)) 2 μ :=
    hξ.comp_measurePreserving (hT.iterate n)
  have h := integral_mul_bound μ _ ξ hs hξ
  rw [shifted_integral μ T hT (fun ω => (ξ ω)^2), hunit] at h
  simpa [lagCovariance] using h

/-- A stable alternating linear recursion determines its bounded solution by
an absolutely convergent covariance series. -/
private theorem bounded_recursion_sum (q : ℝ) (hq : |q| < 1)
    (ρ b : ℕ → ℝ) (hρ : ∀ n, |ρ n| ≤ 1)
    (B : ℝ) (hb : ∀ n, |b n| ≤ B)
    (hrec : ∀ n, b n = q * ρ (n+1) - q * b (n+1)) :
    b 0 = q * ∑' j : ℕ, (-q)^j * ρ (j+1) := by
  have hfinite : ∀ n : ℕ,
      b 0 = q * (∑ j ∈ Finset.range n, (-q)^j * ρ (j+1)) + (-q)^n * b n := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      rw [Finset.sum_range_succ, pow_succ]
      rw [hrec n] at ih
      nlinarith [ih]
  have hrem : Tendsto (fun n : ℕ => (-q)^n * b n) atTop (𝓝 0) := by
    apply squeeze_zero_norm (a := fun n : ℕ => |q|^n * B)
    · intro n
      rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_neg]
      exact mul_le_mul_of_nonneg_left (hb n) (pow_nonneg (abs_nonneg q) n)
    · simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one (abs_nonneg q) hq).mul_const B
  have hsum := ((covariance_series_summable q hq ρ hρ).hasSum.tendsto_sum_nat.const_mul q).add hrem
  have hc : Tendsto (fun _ : ℕ => b 0) atTop (𝓝 (q * ∑' j : ℕ, (-q)^j * ρ (j+1))) := by
    simpa only [add_zero, ← hfinite] using hsum
  exact tendsto_nhds_unique tendsto_const_nhds hc

/-- The actual stationary local recurrence identifies its cross moment with
its driver's covariance series; the cross-moment identity is not assumed. -/
theorem stationary_cross_moment_series {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (ξ X : Ω → ℝ) (q : ℝ) (hq : |q| < 1)
    (hξ : MemLp ξ 2 μ) (hX : MemLp X 2 μ)
    (hunit : ∫ ω, (ξ ω)^2 ∂μ = 1)
    (hrec : ∀ᵐ ω ∂μ, X (T ω) = q * (ξ ω - X ω)) :
    (∫ ω, ξ ω * X ω ∂μ) = q * ∑' j : ℕ, (-q)^j * lagCovariance μ T ξ (j+1) := by
  let b : ℕ → ℝ := fun n => ∫ ω, ξ ((T : Ω → Ω)^[n] ω) * X ω ∂μ
  have hs (n : ℕ) : MemLp (fun ω => ξ ((T : Ω → Ω)^[n] ω)) 2 μ :=
    hξ.comp_measurePreserving (hT.iterate n)
  have hb (n : ℕ) : |b n| ≤ (1 + ∫ ω, (X ω)^2 ∂μ) / 2 := by
    have h := integral_mul_bound μ _ X (hs n) hX
    rw [shifted_integral μ T hT (fun ω => (ξ ω)^2), hunit] at h
    exact h
  have hstep (n : ℕ) : b n = q * lagCovariance μ T ξ (n+1) - q * b (n+1) := by
    have hiξ : Integrable (fun ω => ξ ((T : Ω → Ω)^[n+1] ω) * ξ ω) μ :=
      (hs (n+1)).integrable_mul hξ
    have hiX : Integrable (fun ω => ξ ((T : Ω → Ω)^[n+1] ω) * X ω) μ :=
      (hs (n+1)).integrable_mul hX
    calc
      b n = ∫ ω, ξ ((T : Ω → Ω)^[n] (T ω)) * X (T ω) ∂μ :=
        (hT.integral_comp' (fun ω => ξ ((T : Ω → Ω)^[n] ω) * X ω)).symm
      _ = ∫ ω, q * (ξ ((T : Ω → Ω)^[n+1] ω) * ξ ω) -
          q * (ξ ((T : Ω → Ω)^[n+1] ω) * X ω) ∂μ := by
        apply integral_congr_ae
        filter_upwards [hrec] with ω hω
        rw [hω, Function.iterate_succ_apply]
        ring
      _ = _ := by
        rw [integral_sub (hiξ.const_mul q) (hiX.const_mul q), integral_const_mul,
          integral_const_mul]
        rfl
  have h := bounded_recursion_sum q hq (lagCovariance μ T ξ) b
    (lagCovariance_abs_le_one μ T hT ξ hξ hunit) _ hb hstep
  simpa [b] using h

/-- Actual stationary moments give the manuscript's exact covariance-series
coefficient reduction. Both moment identities follow from the local recurrence. -/
theorem stationary_moment_reduction {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (ξ X : Ω → ℝ) (q : ℝ)
    (hq : q ≠ 0) (hstable : |q| < 1)
    (hξ : MemLp ξ 2 μ) (hX : MemLp X 2 μ)
    (hunit : ∫ ω, (ξ ω) ^ 2 ∂μ = 1)
    (hrec : ∀ᵐ ω ∂μ, X (T ω) = q * (ξ ω - X ω)) :
    (1 - q) * (∫ ω, (X ω) ^ 2 ∂μ) / (2 * q) + (∫ ω, ξ ω * X ω ∂μ) =
      q / (2 * (1 + q)) *
        (1 + 2 * ∑' j : ℕ, (-q)^j * lagCovariance μ T ξ (j+1)) := by
  have hK := stationary_cross_moment_series μ T hT ξ X q hstable hξ hX hunit hrec
  have hv := stationary_variance_equation μ T hT ξ X q hξ hX hunit hrec
  have hq1 : q ≠ 1 := by have := (abs_lt.mp hstable).2; linarith
  have hqm : q ≠ -1 := by have := (abs_lt.mp hstable).1; linarith
  rw [hK] at hv
  exact moment_reduction_from_equations q _ _ _ hq hq1 hqm hK (by nlinarith [hv])

end Dobson.Coverage6ExactMemory
#print axioms Dobson.Coverage6ExactMemory.covariance_series_summable
#print axioms Dobson.Coverage6ExactMemory.ar_lag_sum
#print axioms Dobson.Coverage6ExactMemory.ar_coefficient
#print axioms Dobson.Coverage6ExactMemory.ar_coefficient_derivative
#print axioms Dobson.Coverage6ExactMemory.ar_coefficient_negative_iff
#print axioms Dobson.Coverage6ExactMemory.ar_derivative_extreme_signs
#print axioms Dobson.Coverage6ExactMemory.stationary_moment_reduction

#print axioms Dobson.Coverage6ExactMemory.stationary_variance_equation

#print axioms Dobson.Coverage6ExactMemory.lagCovariance_abs_le_one
#print axioms Dobson.Coverage6ExactMemory.stationary_cross_moment_series
