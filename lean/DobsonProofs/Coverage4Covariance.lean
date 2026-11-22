import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Topology.Instances.Matrix
import Mathlib.Topology.Algebra.InfiniteSum.Ring
import Mathlib.Topology.Algebra.InfiniteSum.NatInt
import Mathlib.Tactic
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Probability.Independence.Integration

/-!
# Batch 4: covariance and Lyapunov algebra
Core of lem:mech-var. Finite covariance telescoping, the infinite Lyapunov equation
under explicit summability, and covariance-family algebra are checked. Missing:
spectral-radius-to-summability bounds, PSD/uniqueness, stationary Gaussian filter,
probabilistic covariance identification, lag/transposition laws and mixing.
-/
/-! ## Coverage correction — 2026-11-22

`lem:mech-var` has core coverage. `diagonal_covariance` derives its covariance
equation from square-integrable stationary diagonal recurrences and centered
innovations; the old `hG` entry is discharged. Stationary-filter construction,
uniqueness and mixing remain outside the credited finite/infinite covariance
algebra and Lyapunov equation under summability. -/

/-! 2026-11-16, leanp1, part2:39: `diagonal_covariance` now proves the
covariance equation from actual square-integrable stationary diagonal AR
recurrences, with centered innovations independent of the opposite past state.
The innovation cross moment and state cross moment are actual integrals, and
no Lyapunov entry is supplied. For standardized centered innovations and states
these are the manuscript's correlation gamma and off-diagonal covariance G12.
Construction of the stationary Gaussian process and its full lag law remain
outside this component. -/

namespace Dobson.Coverage4Covariance
open scoped BigOperators
open Filter

/-- Exact telescoping residual of the finite propagated-innovation covariance sum. -/
theorem finite_lyapunov_residual {ι : Type*} [Fintype ι] [DecidableEq ι]
    (M S : Matrix ι ι ℝ) (N : ℕ) :
    (∑ j ∈ Finset.range N, M^j*S*(M.transpose)^j) -
      M*(∑ j ∈ Finset.range N, M^j*S*(M.transpose)^j)*M.transpose =
      S-M^N*S*(M.transpose)^N := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [Finset.sum_range_succ, mul_add, add_mul]
    have hp : M*(M^N*S*(M.transpose)^N)*M.transpose =
        M^(N+1)*S*(M.transpose)^(N+1) := by
      rw [pow_succ' M, pow_succ M.transpose]
      simp only [Matrix.mul_assoc]
    rw [hp]
    rw [add_sub_add_comm, ih]
    abel

/-- The summed covariance solves the discrete Lyapunov equation.
Summability is an explicit analytic input: no spectral-radius implication or
stationary Gaussian construction is asserted here. -/
theorem infinite_lyapunov {ι : Type*} [Fintype ι] [DecidableEq ι]
    (M S : Matrix ι ι ℝ)
    (hs : Summable (fun j : ℕ => M^j*S*(M.transpose)^j)) :
    (∑' j : ℕ, M^j*S*(M.transpose)^j) =
      M*(∑' j : ℕ, M^j*S*(M.transpose)^j)*M.transpose+S := by
  have htail : (∑' j : ℕ, M^(j+1)*S*(M.transpose)^(j+1)) =
      M*(∑' j : ℕ, M^j*S*(M.transpose)^j)*M.transpose := by
    calc
      _ = ∑' j : ℕ, M*(M^j*S*(M.transpose)^j)*M.transpose := by
        apply tsum_congr
        intro j
        rw [pow_succ' M, pow_succ M.transpose]
        simp only [Matrix.mul_assoc]
      _ = _ := by rw [(hs.mul_left M).tsum_mul_right, hs.tsum_mul_left]
  have h := hs.sum_add_tsum_nat_add 1
  simp only [Finset.sum_range_one, pow_zero, one_mul, mul_one] at h
  rw [htail] at h
  simpa only [add_comm] using h.symm

/-- Rotation covariance G=I solves the actual 2-by-2 Lyapunov equation.
The unit-circle identity may be instantiated with cosine and sine. -/
theorem rotation_lyapunov (r c s : ℝ) (hunit : c^2+s^2 = 1) :
    let M : Matrix (Fin 2) (Fin 2) ℝ := !![r*c, -r*s; r*s, r*c]
    M*M.transpose + (1-r^2) • (1 : Matrix (Fin 2) (Fin 2) ℝ) = 1 := by
  have hu := congrArg (fun x : ℝ => r^2*x) hunit
  change r^2*(c^2+s^2) = r^2*1 at hu
  dsimp
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Matrix.vecMul, dotProduct, Fin.sum_univ_two, Matrix.transpose_apply, Matrix.one_apply] <;>
    nlinarith [hu]

open MeasureTheory ProbabilityTheory

/-- The covariance of two actual stationary diagonal AR recurrences. Centered
innovations independent of the opposite previous state remove the mixed terms.
The innovation cross moment is the actual integral, not a supplied covariance
balance or an arbitrary scalar named a covariance. -/
theorem diagonal_covariance {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (X Y ε η : Ω → ℝ) (φ1 φ2 : ℝ)
    (hφ1 : |φ1| < 1) (hφ2 : |φ2| < 1)
    (hX : MemLp X 2 μ) (hY : MemLp Y 2 μ) (hε : MemLp ε 2 μ) (hη : MemLp η 2 μ)
    (hcenterε : ∫ ω, ε ω ∂μ = 0) (hcenterη : ∫ ω, η ω ∂μ = 0)
    (hindXη : IndepFun X η μ) (hindεY : IndepFun ε Y μ)
    (hrecX : ∀ᵐ ω ∂μ, X (T ω) = φ1 * X ω + Real.sqrt (1-φ1^2) * ε ω)
    (hrecY : ∀ᵐ ω ∂μ, Y (T ω) = φ2 * Y ω + Real.sqrt (1-φ2^2) * η ω) :
    (∫ ω, X ω * Y ω ∂μ) =
      (∫ ω, ε ω * η ω ∂μ) * Real.sqrt ((1-φ1^2)*(1-φ2^2)) / (1-φ1*φ2) := by
  have hiXY : Integrable (fun ω => X ω * Y ω) μ := hX.integrable_mul hY
  have hiXη : Integrable (fun ω => X ω * η ω) μ := hX.integrable_mul hη
  have hiεY : Integrable (fun ω => ε ω * Y ω) μ := hε.integrable_mul hY
  have hiεη : Integrable (fun ω => ε ω * η ω) μ := hε.integrable_mul hη
  have hi01 : Integrable (fun ω => (φ1 * φ2) * (X ω * Y ω) +
      (φ1 * Real.sqrt (1-φ2^2)) * (X ω * η ω)) μ :=
    (hiXY.const_mul _).add (hiXη.const_mul _)
  have hi012 : Integrable (fun ω => (φ1 * φ2) * (X ω * Y ω) +
      (φ1 * Real.sqrt (1-φ2^2)) * (X ω * η ω) +
      (φ2 * Real.sqrt (1-φ1^2)) * (ε ω * Y ω)) μ :=
    hi01.add (hiεY.const_mul _)
  have hzXη : (∫ ω, X ω * η ω ∂μ) = 0 := by
    rw [hindXη.integral_fun_mul_eq_mul_integral hX.aestronglyMeasurable hη.aestronglyMeasurable,
      hcenterη, mul_zero]
  have hzεY : (∫ ω, ε ω * Y ω ∂μ) = 0 := by
    rw [hindεY.integral_fun_mul_eq_mul_integral hε.aestronglyMeasurable hY.aestronglyMeasurable,
      hcenterε, zero_mul]
  have hbal : (∫ ω, X ω * Y ω ∂μ) =
      φ1 * φ2 * (∫ ω, X ω * Y ω ∂μ) +
        Real.sqrt (1-φ1^2) * Real.sqrt (1-φ2^2) * (∫ ω, ε ω * η ω ∂μ) := by
    calc
      _ = ∫ ω, X (T ω) * Y (T ω) ∂μ := (hT.integral_comp' (fun ω => X ω * Y ω)).symm
      _ = ∫ ω, (φ1 * φ2) * (X ω * Y ω) +
          (φ1 * Real.sqrt (1-φ2^2)) * (X ω * η ω) +
          (φ2 * Real.sqrt (1-φ1^2)) * (ε ω * Y ω) +
          (Real.sqrt (1-φ1^2) * Real.sqrt (1-φ2^2)) * (ε ω * η ω) ∂μ := by
        apply integral_congr_ae
        filter_upwards [hrecX, hrecY] with ω hx hy
        rw [hx, hy]
        ring
      _ = _ := by
        rw [integral_add hi012 (hiεη.const_mul (Real.sqrt (1-φ1^2) * Real.sqrt (1-φ2^2))),
          integral_add hi01 (hiεY.const_mul (φ2 * Real.sqrt (1-φ1^2))),
          integral_add (hiXY.const_mul (φ1*φ2)) (hiXη.const_mul (φ1 * Real.sqrt (1-φ2^2)))]
        simp only [integral_const_mul, hzXη, hzεY, mul_zero, add_zero]
  have hp : |φ1 * φ2| < 1 := by
    rw [abs_mul]
    nlinarith [abs_nonneg φ1, abs_nonneg φ2]
  have hden : 1-φ1*φ2 ≠ 0 := by have := (abs_lt.mp hp).2; linarith
  have hpos : 0 ≤ 1-φ1^2 := by
    have hsq : φ1^2 < 1 := by nlinarith [sq_abs φ1, abs_nonneg φ1]
    linarith
  rw [Real.sqrt_mul hpos] 
  apply (eq_div_iff hden).mpr
  nlinarith [hbal]

/-- Exact sum/difference modes of the symmetric competition transition matrix. -/
theorem competition_modes (a c : ℝ) :
    (!![a, -c; -c, a] : Matrix (Fin 2) (Fin 2) ℝ).mulVec ![1,1] = ![a-c,a-c] ∧
    (!![a, -c; -c, a] : Matrix (Fin 2) (Fin 2) ℝ).mulVec ![1,-1] = ![a+c,-(a+c)] := by
  constructor <;> ext i <;> fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> ring

/-- Independent-copy interpolation preserves each marginal covariance and scales
both cross covariances. The vanishing mixed-copy moments are represented by the
zeros in the bilinear expressions; the probability construction is not formalized. -/
theorem interpolated_covariances (q self cross : ℝ) (hq : |q| ≤ 1) :
    q^2*self+(Real.sqrt (1-q^2))^2*self = self ∧
    q*cross+Real.sqrt (1-q^2)*0 = q*cross := by
  have hsq : q^2 ≤ 1 := by
    have := (sq_le_sq₀ (abs_nonneg q) (by norm_num : (0:ℝ) ≤ 1)).mpr hq
    simpa only [sq_abs, one_pow] using this
  rw [Real.sq_sqrt (by linarith : 0 ≤ 1-q^2)]
  constructor <;> ring


end Dobson.Coverage4Covariance

#print axioms Dobson.Coverage4Covariance.diagonal_covariance
