import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.Topology.Instances.Matrix
import Mathlib.Tactic

/-!
# Batch 6: ordered covariance for general perturbation matrices

Core of Appendix C, `lem:mech-general`. The matrices may have arbitrary signed
entries, and the lag covariance keeps the ordered pair of driver labels.

The finite response is constructed from the transverse matrix T and projection Q.
Actual Bochner integrals give its expected quadratic logarithmic term. The infinite
coefficient follows from summability of the ordered covariance sequences and a
separately listed identification of the quadratic growth coefficient as the limit
of these finite-response expectations. No matrix-product stochastic growth theorem,
stationary pullback, spectral decay, or birth sensitivity theorem is asserted here.

2026-11-16, auditfix1, IDs 62-63: `hgrowth` (identification with the
Lyapunov quadratic coefficient) and `hbirth` (implicit-threshold balance) are
results argued in ../paper/appendices/C-mechanisms.tex, not model
definitions or textbook consequences of the hypotheses on arbitrary matrices.
`expected_quadratic_limit` proves the series limit but does not identify it with
actual growth derivatives. These two bridges remain unproved in Lean; coverage
of lem:mech-general is partial. The explicit finite matrix/integral core is
still checked. No desired coefficient identity has been renamed as a model.
-/
namespace Dobson.Coverage6Covariance
open scoped BigOperators
open MeasureTheory Filter

variable {ι κ Ω : Type*} [Fintype ι] [Fintype κ] [DecidableEq κ]

/-- Signed first-order perturbation matrix for a realization of all drivers. -/
noncomputable def perturbation (E : ι → Matrix κ κ ℝ) (x : ι → ℝ) : Matrix κ κ ℝ :=
  ∑ i, x i • E i

/-- Truncation of the stationary first variation at N past time steps. -/
noncomputable def response (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ)
    (v : κ → ℝ) (x : ι → ℕ → ℝ) (N : ℕ) : κ → ℝ :=
  ∑ m ∈ Finset.range N, ∑ j, x j (m+1) • ((T^m * Q * E j).mulVec v)

/-- The constructed finite response obeys the causal first-variation recursion.
The shifted history is the previous time's past, and the newest forcing has lag 1. -/
theorem response_recurrence (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ)
    (v : κ → ℝ) (x : ι → ℕ → ℝ) (N : ℕ) :
    response T Q E v x (N+1) =
      T.mulVec (response T Q E v (fun i lag => x i (lag+1)) N) +
        ∑ j, x j 1 • ((Q * E j).mulVec v) := by
  classical
  simp only [response, Finset.sum_range_succ', pow_zero, one_mul,
    zero_add, Matrix.mulVec_sum, Matrix.mulVec_smul]
  congr 1
  apply Finset.sum_congr rfl
  intro m hm
  apply Finset.sum_congr rfl
  intro j hj
  simp only [pow_succ', Matrix.mulVec_mulVec, Matrix.mul_assoc]

/-- The ordered contraction: i acts at the present, j at the past lag m+1. -/
noncomputable def kernel (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ)
    (v w : κ → ℝ) (i j : ι) (m : ℕ) : ℝ :=
  w ⬝ᵥ ((E i * T^m * Q * E j).mulVec v)

/-- Exact finite-response expansion, with no commutation of E_i and E_j. -/
theorem response_contraction (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ)
    (v w : κ → ℝ) (x : ι → ℕ → ℝ) (N : ℕ) :
    w ⬝ᵥ ((perturbation E (fun i => x i 0)).mulVec (response T Q E v x N)) =
      ∑ i, ∑ j, ∑ m ∈ Finset.range N,
        kernel T Q E v w i j m * (x i 0 * x j (m+1)) := by
  classical
  simp only [perturbation, response, Matrix.sum_mulVec, Matrix.mulVec_sum,
    dotProduct_sum, Matrix.smul_mulVec, Matrix.mulVec_smul, dotProduct_smul,
    smul_eq_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  conv_lhs => arg 2; ext j; rw [Finset.sum_comm]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  apply Finset.sum_congr rfl
  intro m hm
  simp only [kernel, Matrix.mulVec_mulVec, Matrix.mul_assoc]
  ring

/-- Exact expansion of the instantaneous square into ordered pairs. -/
theorem instantaneous_square (c : ι → ℝ) (x : ι → ℝ) :
    (∑ i, c i * x i)^2 = ∑ i, ∑ j, c i * c j * (x i * x j) := by
  classical
  rw [pow_two, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  ring

variable [MeasurableSpace Ω]

/-- Ordered lag covariance; lag zero is the contemporaneous second moment. -/
noncomputable def covariance (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ)
    (i j : ι) (lag : ℕ) : ℝ :=
  ∫ ω, ξ i 0 ω * ξ j lag ω ∂μ

/-- Integrability of the actual finite matrix response follows from the listed
pair-product integrability hypotheses. -/
theorem integrable_response (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ)
    (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ) (N : ℕ)
    (hξ : ∀ i j lag, Integrable (fun ω => ξ i 0 ω * ξ j lag ω) μ) :
    Integrable (fun ω => w ⬝ᵥ ((perturbation E (fun i => ξ i 0 ω)).mulVec
      (response T Q E v (fun i lag => ξ i lag ω) N))) μ := by
  classical
  have hp (ω : Ω) := response_contraction T Q E v w (fun i lag => ξ i lag ω) N
  simp only [hp]
  apply integrable_finsetSum
  intro i hi
  apply integrable_finsetSum
  intro j hj
  apply integrable_finsetSum
  intro m hm
  exact (hξ i j (m+1)).const_mul _

/-- Taking an actual expectation produces the ordered covariance contraction. -/
theorem integral_response (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ)
    (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ) (N : ℕ)
    (hξ : ∀ i j lag, Integrable (fun ω => ξ i 0 ω * ξ j lag ω) μ) :
    (∫ ω, w ⬝ᵥ ((perturbation E (fun i => ξ i 0 ω)).mulVec
      (response T Q E v (fun i lag => ξ i lag ω) N)) ∂μ) =
      ∑ i, ∑ j, ∑ m ∈ Finset.range N,
        kernel T Q E v w i j m * covariance μ ξ i j (m+1) := by
  classical
  have hp (ω : Ω) := response_contraction T Q E v w (fun i lag => ξ i lag ω) N
  simp only [hp]
  rw [integral_finsetSum]
  · apply Finset.sum_congr rfl
    intro i hi
    rw [integral_finsetSum]
    · apply Finset.sum_congr rfl
      intro j hj
      rw [integral_finsetSum]
      · simp only [integral_const_mul, covariance]
      · intro m hm
        exact (hξ i j (m+1)).const_mul _
    · intro j hj
      exact integrable_finsetSum _ fun m hm => (hξ i j (m+1)).const_mul _
  · intro i hi
    exact integrable_finsetSum _ fun j hj =>
      integrable_finsetSum _ fun m hm => (hξ i j (m+1)).const_mul _

/-- The square of the instantaneous elasticity-weighted driver is integrable. -/
theorem integrable_square (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ) (c : ι → ℝ)
    (hξ : ∀ i j, Integrable (fun ω => ξ i 0 ω * ξ j 0 ω) μ) :
    Integrable (fun ω => (∑ i, c i * ξ i 0 ω)^2) μ := by
  classical
  simp_rw [instantaneous_square]
  exact integrable_finsetSum _ fun i hi =>
    integrable_finsetSum _ fun j hj => (hξ i j).const_mul _

/-- The instantaneous log term gives every ordered covariance, including cross terms. -/
theorem integral_square (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ) (c : ι → ℝ)
    (hξ : ∀ i j, Integrable (fun ω => ξ i 0 ω * ξ j 0 ω) μ) :
    (∫ ω, (∑ i, c i * ξ i 0 ω)^2 ∂μ) =
      ∑ i, ∑ j, c i * c j * covariance μ ξ i j 0 := by
  classical
  simp_rw [instantaneous_square]
  rw [integral_finsetSum]
  · apply Finset.sum_congr rfl
    intro i hi
    rw [integral_finsetSum]
    · simp only [integral_const_mul, covariance]
    · intro j hj
      exact (hξ i j).const_mul _
  · intro i hi
    exact integrable_finsetSum _ fun j hj => (hξ i j).const_mul _

/-- Centered drivers make the first-order expected logarithmic term zero. -/
theorem linear_term_centered (μ : Measure Ω) (ξ : ι → Ω → ℝ) (c : ι → ℝ)
    (hξ : ∀ i, Integrable (ξ i) μ) (hcenter : ∀ i, ∫ ω, ξ i ω ∂μ = 0) :
    (∫ ω, ∑ i, c i * ξ i ω ∂μ) = 0 := by
  rw [integral_finsetSum _ fun i hi => (hξ i).const_mul _]
  simp only [integral_const_mul, hcenter, mul_zero, Finset.sum_const_zero]

/-- Deterministic elasticity of labeled perturbation i. -/
noncomputable def elasticity (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ) (i : ι) : ℝ :=
  w ⬝ᵥ ((E i).mulVec v)

/-- The actual finite-response second-order term in the normalized one-step log.
`direct` is the direct second-order matrix contribution before expectation. -/
noncomputable def logarithmicQuadratic (ξ : ι → ℕ → Ω → ℝ)
    (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ)
    (direct : Ω → ℝ) (N : ℕ) (ω : Ω) : ℝ :=
  direct ω + w ⬝ᵥ ((perturbation E (fun i => ξ i 0 ω)).mulVec
    (response T Q E v (fun i lag => ξ i lag ω) N)) -
      (∑ i, elasticity E v w i * ξ i 0 ω)^2 / 2

/-- Finite-horizon expected log coefficient, derived from matrix algebra and
Bochner integration; the direct quadratic contribution vanishes by its own input. -/
theorem finite_expected_quadratic (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ)
    (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ)
    (direct : Ω → ℝ) (N : ℕ)
    (hξ : ∀ i j lag, Integrable (fun ω => ξ i 0 ω * ξ j lag ω) μ)
    (hd : Integrable direct μ) (hd0 : ∫ ω, direct ω ∂μ = 0) :
    (∫ ω, logarithmicQuadratic ξ T Q E v w direct N ω ∂μ) =
      ∑ i, ∑ j, ((∑ m ∈ Finset.range N,
        kernel T Q E v w i j m * covariance μ ξ i j (m+1)) -
          elasticity E v w i * elasticity E v w j * covariance μ ξ i j 0 / 2) := by
  classical
  have hr := integrable_response μ ξ T Q E v w N hξ
  have hs := integrable_square μ ξ (elasticity E v w) (fun i j => hξ i j 0)
  have hds : Integrable (fun ω => direct ω + w ⬝ᵥ
      ((perturbation E (fun i => ξ i 0 ω)).mulVec
        (response T Q E v (fun i lag => ξ i lag ω) N))) μ := hd.add hr
  unfold logarithmicQuadratic
  rw [integral_sub hds (hs.div_const 2), integral_add hd hr, hd0,
    zero_add, integral_div, integral_response μ ξ T Q E v w N hξ,
    integral_square μ ξ (elasticity E v w) (fun i j => hξ i j 0)]
  simp only [Finset.sum_div, Finset.sum_sub_distrib]

/-- The limit of the expected finite-response log terms is the ordered lag series.
Pairwise covariance summability is explicit, rather than inferred from primitivity. -/
theorem expected_quadratic_limit (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ)
    (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ)
    (direct : Ω → ℝ)
    (hξ : ∀ i j lag, Integrable (fun ω => ξ i 0 ω * ξ j lag ω) μ)
    (hd : Integrable direct μ) (hd0 : ∫ ω, direct ω ∂μ = 0)
    (hsum : ∀ i j, Summable (fun m : ℕ =>
      kernel T Q E v w i j m * covariance μ ξ i j (m+1))) :
    Tendsto (fun N => ∫ ω, logarithmicQuadratic ξ T Q E v w direct N ω ∂μ)
      atTop (nhds (∑ i, ∑ j, ((∑' m : ℕ,
        kernel T Q E v w i j m * covariance μ ξ i j (m+1)) -
          elasticity E v w i * elasticity E v w j * covariance μ ξ i j 0 / 2))) := by
  classical
  simp_rw [finite_expected_quadratic μ ξ T Q E v w direct _ hξ hd hd0]
  apply tendsto_finsetSum
  intro i hi
  apply tendsto_finsetSum
  intro j hj
  exact (hsum i j).hasSum.tendsto_sum_nat.sub_const _

/-- Ordered covariance coefficient of the restoring log birth multiplier.

Extra analytic inputs are listed explicitly: `hξ` supplies the integrable lag
products; `hsum` supplies covariance-series convergence; `hgrowth` identifies the
actual quadratic growth coefficient g2 with the finite-response limit; and
`hbirth` is the second-order zero-growth balance for birth sensitivity b. The
latter two are not derived from stationary random matrix products here. -/
theorem ordered_covariance_coefficient (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ)
    (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ)
    (direct : Ω → ℝ) (b K g2 : ℝ) (hb : 0 < b)
    (hξ : ∀ i j lag, Integrable (fun ω => ξ i 0 ω * ξ j lag ω) μ)
    (hd : Integrable direct μ) (hd0 : ∫ ω, direct ω ∂μ = 0)
    (hsum : ∀ i j, Summable (fun m : ℕ =>
      kernel T Q E v w i j m * covariance μ ξ i j (m+1)))
    (hgrowth : Tendsto (fun N => ∫ ω,
      logarithmicQuadratic ξ T Q E v w direct N ω ∂μ) atTop (nhds g2))
    (hbirth : b * K + g2 = 0) :
    K = (1 / b) * ∑ i, ∑ j,
      (elasticity E v w i * elasticity E v w j * covariance μ ξ i j 0 / 2 -
        ∑' m : ℕ, covariance μ ξ i j (m+1) * kernel T Q E v w i j m) := by
  classical
  have hg := tendsto_nhds_unique hgrowth
    (expected_quadratic_limit μ ξ T Q E v w direct hξ hd hd0 hsum)
  have hneg : (∑ i, ∑ j,
      (elasticity E v w i * elasticity E v w j * covariance μ ξ i j 0 / 2 -
        ∑' m : ℕ, covariance μ ξ i j (m+1) * kernel T Q E v w i j m)) = -g2 := by
    rw [hg, ← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro j hj
    simp only [neg_sub]
    congr 1
    apply tsum_congr
    intro m
    ring
  rw [hneg]
  field_simp
  nlinarith

end Dobson.Coverage6Covariance

#print axioms Dobson.Coverage6Covariance.response_recurrence
#print axioms Dobson.Coverage6Covariance.response_contraction
#print axioms Dobson.Coverage6Covariance.instantaneous_square
#print axioms Dobson.Coverage6Covariance.integrable_response
#print axioms Dobson.Coverage6Covariance.integral_response
#print axioms Dobson.Coverage6Covariance.integrable_square
#print axioms Dobson.Coverage6Covariance.integral_square
#print axioms Dobson.Coverage6Covariance.linear_term_centered
#print axioms Dobson.Coverage6Covariance.finite_expected_quadratic
#print axioms Dobson.Coverage6Covariance.expected_quadratic_limit
#print axioms Dobson.Coverage6Covariance.ordered_covariance_coefficient
