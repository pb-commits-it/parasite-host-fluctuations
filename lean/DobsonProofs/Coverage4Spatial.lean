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

/-!
# Batch 4: spatial uniform and transverse modes
Core of prop:mech-dispersal. Exact island/equicorrelation eigenspaces and the
conservative movement-location derivative are proved. Missing probability inputs:
stationary Gaussian perturbation expansion, covariance contraction, Perron birth
elasticity, and identification of the physical coefficient with q*S-(1-q)*H.
The theorem does not assert full diagonalization or a global finite-noise order.
-/
namespace Dobson.Coverage4Spatial
open scoped BigOperators
open Filter

/-- Scalar plus uniform rank-one matrix, the common spatial eigenspace model. -/
def rankOne {ι : Type*} [DecidableEq ι] (a b : ℝ) : Matrix ι ι ℝ :=
  fun i j => (if i = j then a else 0) + b

/-- Exact multiplication, before identifying uniform and transverse modes. -/
theorem rank_one_action {ι : Type*} [Fintype ι] [DecidableEq ι]
    (a b : ℝ) (v : ι → ℝ) (i : ι) :
    (rankOne a b).mulVec v i = a * v i + b * ∑ j, v j := by
  simp [rankOne, Matrix.mulVec, dotProduct, add_mul, Finset.sum_add_distrib,
    ite_mul, ← Finset.mul_sum]

/-- Uniform island eigenvalue is 1-m*loss for any nonempty patch set. -/
theorem island_uniform {ι : Type*} [Fintype ι] [DecidableEq ι] [Nonempty ι]
    (m loss : ℝ) (i : ι) :
    (rankOne (1-m) (m*(1-loss)/(Fintype.card ι : ℝ))).mulVec (fun _ => 1) i =
      1 - m * loss := by
  rw [rank_one_action]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
  have hcard : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  field_simp
  ring

/-- Every zero-sum vector is a transverse eigenvector, with eigenvalue 1-m. -/
theorem island_transverse {ι : Type*} [Fintype ι] [DecidableEq ι]
    (m loss : ℝ) (v : ι → ℝ) (hv : ∑ j, v j = 0) :
    (rankOne (1-m) (m*(1-loss)/(Fintype.card ι : ℝ))).mulVec v =
      fun i => (1-m) * v i := by
  funext i
  rw [rank_one_action, hv]
  ring

/-- Equicorrelation has the same eigenspaces, with eigenvalues P*q and 1-rho. -/
theorem patch_covariance_modes {ι : Type*} [Fintype ι] [DecidableEq ι] [Nonempty ι]
    (ρ : ℝ) (v : ι → ℝ) (hv : ∑ j, v j = 0) (i : ι) :
    (rankOne (1-ρ) ρ).mulVec (fun _ => 1) i =
      (Fintype.card ι : ℝ) * (ρ + (1-ρ)/(Fintype.card ι : ℝ)) ∧
    (rankOne (1-ρ) ρ).mulVec v i = (1-ρ)*v i := by
  constructor
  · rw [rank_one_action]
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
    have hcard : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
    field_simp
    ring
  · rw [rank_one_action, hv]
    ring

/-- Literal spatial coefficient: no memory and complete mixing identities.
The Gaussian cost formula and the physical return products remain model inputs. -/
theorem spatial_coefficient_limits (q T φ Ra a sd : ℝ) :
    q * ((1 + (0:ℝ))/(2*(1-0)*T) - 0/(1-0)) - (1-q)*(0/(1-0)) = q/(2*T) ∧
    q * ((1+φ)/(2*(1-φ)*T) - Ra/(1-Ra)) -
      (1-q) * ((Ra*(0/a)*(1-φ*a*sd)/(1-φ*0*sd)) /
        (1-Ra*(0/a)*(1-φ*a*sd)/(1-φ*0*sd))) =
      q * ((1+φ)/(2*(1-φ)*T) - Ra/(1-Ra)) := by
  constructor <;> simp <;> ring

/-- Actual stay-location derivative of the conservative transverse return factor. -/
theorem movement_location_derivative (m φ s : ℝ) (hden : 1-φ*(1-m)*s ≠ 0) :
    HasDerivAt (fun x : ℝ => (1-m)*(1-φ*x)/(1-φ*(1-m)*x))
      (-(1-m)*φ*m/(1-φ*(1-m)*s)^2) s := by
  convert (((hasDerivAt_const s (1 : ℝ)).sub ((hasDerivAt_id s).const_mul φ)).const_mul (1-m)).div
    ((hasDerivAt_const s (1 : ℝ)).sub ((hasDerivAt_id s).const_mul (φ*(1-m)))) hden using 1 <;> (try funext y) <;> dsimp <;>
    ring


end Dobson.Coverage4Spatial
