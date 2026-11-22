import DobsonProofs.Full3_CyclicDet
import Mathlib.LinearAlgebra.Matrix.SchurComplement
import Mathlib.LinearAlgebra.Matrix.Charpoly.Eigs
import Mathlib.Logic.Equiv.Fin.Rotate
import Mathlib.FieldTheory.RatFunc.Basic

/-! Arbitrary finite-dimensional scalar-coupled blocks. Internal state spaces
may have different dimensions. The input and output matrices construct the
actual closed-ring Jacobian; denominators are never reduced. -/

namespace Dobson.Full3.BlockRing
open Finset Matrix Polynomial
open scoped BigOperators
attribute [-simp] finRotate_apply finRotate_symm_apply

variable {n : ℕ} (m : Fin n → ℕ)
abbrev State := (i : Fin n) × Fin (m i)

def fiberEquiv (i : Fin n) : {s : State m // s.1=i} ≃ Fin (m i) where
  toFun s := by rcases s with ⟨⟨j,k⟩,h⟩; dsimp at h; subst j; exact k
  invFun k := ⟨⟨i,k⟩,rfl⟩
  left_inv s := by rcases s with ⟨⟨j,k⟩,h⟩; dsimp at h; subst j; rfl
  right_inv k := rfl

section Algebra
variable {F : Type*} [CommRing F]

theorem det_blocks (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) F) :
    (Matrix.blockDiagonal' A).det = ∏ i, (A i).det := by
  have ht : (Matrix.blockDiagonal' A).BlockTriangular Sigma.fst := by
    intro s t h
    exact Matrix.blockDiagonal'_apply_ne _ _ _ (ne_of_gt h)
  rw [ht.det_fintype]
  apply prod_congr rfl
  intro i _
  have he : ((Matrix.blockDiagonal' A).toSquareBlock Sigma.fst i).submatrix
      (fiberEquiv m i).symm (fiberEquiv m i).symm = A i := by
    ext j k
    simp [Matrix.toSquareBlock, Matrix.toSquareBlockProp, Matrix.toBlock,
      fiberEquiv, Matrix.blockDiagonal'_apply_eq]
  rw [← he, Matrix.det_submatrix_equiv_self]

def injectInput (b : ∀ i, Fin (m i) → F) : Matrix (State m) (Fin n) F :=
  fun s j => if s.1=j then b s.1 s.2 else 0

def routeOutput (c : ∀ i, Fin (m i) → F) : Matrix (Fin n) (State m) F :=
  fun i s => if i=finRotate n s.1 then c s.1 s.2 else 0

noncomputable def contraction (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) F)
    (b c : ∀ i, Fin (m i) → F) (i : Fin n) : F :=
  ∑ p, ∑ q, c i p * A i p q * b i q

theorem routed_contraction (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) F)
    (b c : ∀ i, Fin (m i) → F) :
    routeOutput m c * Matrix.blockDiagonal' A * injectInput m b =
      Matrix.of (fun i j => if i=finRotate n j then contraction m A b c j else 0) := by
  ext i j
  simp only [Matrix.mul_apply, Fintype.sum_sigma, routeOutput, injectInput, Matrix.of_apply,
    Matrix.blockDiagonal'_apply]
  simp only [ite_mul, zero_mul, mul_ite, mul_zero, sum_ite_irrel,
    Finset.sum_ite_eq', mem_univ, ite_true]
  by_cases h : i=finRotate n j
  · simp [h, contraction, mul_sum, sum_mul, mul_assoc]
    rw [Finset.sum_comm]
  · simp [h, contraction, mul_sum, sum_mul, mul_assoc]
    apply sum_eq_zero
    intro q _
    apply sum_eq_zero
    intro k _
    by_cases hkj : k=j
    · subst k; simp [h]
    · simp [hkj]

end Algebra

section Field
variable {F : Type*} [Field F]

theorem inverse_blocks (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) F)
    (hA : ∀ i, (A i).det ≠ 0) :
    (Matrix.blockDiagonal' A)⁻¹ = Matrix.blockDiagonal' (fun i => (A i)⁻¹) := by
  apply Matrix.inv_eq_left_inv
  rw [← Matrix.blockDiagonal'_mul]
  simp only [Matrix.nonsing_inv_mul _ (isUnit_iff_ne_zero.mpr (hA _))]
  exact Matrix.blockDiagonal'_one

theorem rotate_val (j : Fin n) : (finRotate n j).val = (j.val+1)%n := by
  cases n with
  | zero => exact j.elim0
  | succ k =>
    rw [coe_finRotate]
    split_ifs with h
    · simp [h]
    · exact (Nat.mod_eq_of_lt (Nat.succ_lt_succ (Fin.val_lt_last h))).symm

theorem det_one_sub_route (hn : 1 ≤ n) (t : Fin n → F) :
    (1 - (Matrix.of fun i j => if i=finRotate n j then t j else 0) :
      Matrix (Fin n) (Fin n) F).det = 1-∏ i, t i := by
  have he : (1 - (Matrix.of fun i j => if i=finRotate n j then t j else 0) :
      Matrix (Fin n) (Fin n) F) =
      Dobson.Full3.CyclicDet.diagSubCyclic n (fun _ => 1) t := by
    ext i j
    have hr : i.val=(j.val+1)%n ↔ i=finRotate n j := by
      rw [← rotate_val, Fin.ext_iff]
    simp [Dobson.Full3.CyclicDet.diagSubCyclic, Matrix.one_apply, hr]
  rw [he, Dobson.Full3.CyclicDet.det_diagSubCyclic n hn, prod_const_one]

/-- Schur reduction is derived for the actual interconnected matrix. -/
theorem schur_reduction (hn : 1 ≤ n)
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) F)
    (b c : ∀ i, Fin (m i) → F) (hA : ∀ i, (A i).det ≠ 0) :
    (Matrix.blockDiagonal' A - injectInput m b * routeOutput m c).det =
      (∏ i, (A i).det) *
        (1 - ∏ i, contraction m (fun k => (A k)⁻¹) b c i) := by
  have hu : IsUnit (Matrix.blockDiagonal' A).det := by
    rw [det_blocks]
    exact isUnit_iff_ne_zero.mpr (prod_ne_zero_iff.mpr fun i _ => hA i)
  have he := Matrix.det_add_mul (-injectInput m b) (routeOutput m c) hu
  simp only [Matrix.neg_mul, ← sub_eq_add_neg, Matrix.mul_neg] at he
  rw [inverse_blocks m A hA, routed_contraction, det_one_sub_route hn, det_blocks] at he
  exact he

theorem contraction_inverse (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) F)
    (b c : ∀ i, Fin (m i) → F) (i : Fin n) :
    contraction m (fun k => (A k)⁻¹) b c i =
      contraction m (fun k => (A k).adjugate) b c i / (A i).det := by
  unfold contraction
  simp only [Matrix.inv_def, Ring.inverse_eq_inv, Matrix.smul_apply, smul_eq_mul,
    div_eq_mul_inv, sum_mul]
  apply sum_congr rfl
  intro p _
  apply sum_congr rfl
  intro q _
  ring

theorem unreduced_determinant (hn : 1 ≤ n)
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) F)
    (b c : ∀ i, Fin (m i) → F) (hA : ∀ i, (A i).det ≠ 0) :
    (Matrix.blockDiagonal' A - injectInput m b * routeOutput m c).det =
      (∏ i, (A i).det) - ∏ i, contraction m (fun k => (A k).adjugate) b c i := by
  rw [schur_reduction m hn A b c hA]
  simp_rw [contraction_inverse]
  rw [prod_div_distrib]
  have hp : (∏ i, (A i).det) ≠ 0 := prod_ne_zero_iff.mpr fun i _ => hA i
  field_simp

end Field

section Maps
variable {R T : Type*} [CommRing R] [CommRing T]

theorem map_blocks (f : R →+* T) (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) R) :
    (Matrix.blockDiagonal' A).map f = Matrix.blockDiagonal' (fun i => (A i).map f) :=
  Matrix.blockDiagonal'_map A f (map_zero f)

theorem map_det' {k : ℕ} (f : R →+* T) (A : Matrix (Fin k) (Fin k) R) :
    f A.det = (A.map f).det := f.map_det A

theorem map_adjugate' {k : ℕ} (f : R →+* T) (A : Matrix (Fin k) (Fin k) R) :
    A.adjugate.map f = (A.map f).adjugate := f.map_adjugate A

theorem map_input (f : R →+* T) (b : ∀ i, Fin (m i) → R) :
    (injectInput m b).map f = injectInput m (fun i j => f (b i j)) := by
  ext i j
  simp [injectInput, apply_ite f]

theorem map_output (f : R →+* T) (c : ∀ i, Fin (m i) → R) :
    (routeOutput m c).map f = routeOutput m (fun i j => f (c i j)) := by
  ext i j
  simp [routeOutput, apply_ite f]

theorem map_contraction (f : R →+* T)
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) R)
    (b c : ∀ i, Fin (m i) → R) (i : Fin n) :
    f (contraction m A b c i) = contraction m (fun k => (A k).map f)
      (fun k j => f (b k j)) (fun k j => f (c k j)) i := by
  simp [contraction, map_sum, map_mul]

end Maps

/-- Prove the identity in the fraction field of complex polynomials, then
use injectivity to return to polynomials. This includes every singular value
and preserves all algebraic multiplicities. -/
theorem polynomial_determinant (hn : 1 ≤ n)
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ[X])
    (b c : ∀ i, Fin (m i) → ℂ[X]) (hA : ∀ i, (A i).det ≠ 0) :
    (Matrix.blockDiagonal' A - injectInput m b * routeOutput m c).det =
      (∏ i, (A i).det) - ∏ i, contraction m (fun k => (A k).adjugate) b c i := by
  let f : ℂ[X] →+* RatFunc ℂ := algebraMap _ _
  apply RatFunc.algebraMap_injective ℂ
  change f _ = f _
  have hdet : ∀ i, ((A i).map f).det ≠ 0 := by
    intro i
    rw [← map_det']
    exact (map_ne_zero_iff f (RatFunc.algebraMap_injective ℂ)).mpr (hA i)
  have hh := unreduced_determinant m hn (fun i => (A i).map f)
    (fun i j => f (b i j)) (fun i j => f (c i j)) hdet
  simp only [map_sub, map_prod, map_contraction, map_adjugate', map_det']
  rw [RingHom.map_det, RingHom.mapMatrix_apply, Matrix.map_sub f (map_sub f),
    Matrix.map_mul, map_blocks, map_input, map_output]
  exact hh

noncomputable def jacobian (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) : Matrix (State m) (State m) ℂ :=
  Matrix.blockDiagonal' A + injectInput m b * routeOutput m c

noncomputable def numerator (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) (i : Fin n) : ℂ[X] :=
  contraction m (fun k => (A k).charmatrix.adjugate)
    (fun k j => Polynomial.C (b k j)) (fun k j => Polynomial.C (c k j)) i

/-- Entrywise verification of the actual scalar closure `u_i=y_(i-1)`,
including self-feedback when the ring has one block. -/
theorem jacobian_entry (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) (s t : State m) :
    jacobian m A b c s t = Matrix.blockDiagonal' A s t +
      (if s.1=finRotate n t.1 then b s.1 s.2*c t.1 t.2 else 0) := by
  simp [jacobian, Matrix.mul_apply, injectInput, routeOutput, ite_mul,
    mul_ite, eq_comm]

theorem charmatrix_ring (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) :
    (jacobian m A b c).charmatrix =
      Matrix.blockDiagonal' (fun i => (A i).charmatrix) -
      injectInput m (fun i j => Polynomial.C (b i j)) *
        routeOutput m (fun i j => Polynomial.C (c i j)) := by
  have hs : Matrix.scalar (State m) (X : ℂ[X]) =
      Matrix.blockDiagonal' (fun i => Matrix.scalar (Fin (m i)) (X : ℂ[X])) := by
    simp [Matrix.scalar, Matrix.blockDiagonal'_diagonal]
  simp only [Matrix.charmatrix, jacobian, RingHom.mapMatrix_apply,
    Matrix.map_add _ (map_add Polynomial.C), Matrix.map_mul, map_blocks,
    map_input, map_output, hs, sub_add_eq_sub_sub, ← Matrix.blockDiagonal'_sub]
  rfl

/-- Equality of the complete characteristic polynomial, without exclusions
or canceled factors. Blocks of any finite internal dimensions are allowed. -/
theorem characteristic_polynomial (hn : 1 ≤ n)
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) :
    (jacobian m A b c).charpoly = (∏ i, (A i).charpoly) - ∏ i, numerator m A b c i := by
  rw [Matrix.charpoly, charmatrix_ring, polynomial_determinant m hn]
  · rfl
  · intro i
    exact (A i).charpoly_monic.ne_zero

noncomputable def response (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) (z : ℂ) (i : Fin n) : ℂ :=
  contraction m (fun k => (Matrix.scalar (Fin (m k)) z - A k)⁻¹) b c i

theorem eval_charmatrix {k : ℕ} (A : Matrix (Fin k) (Fin k) ℂ) (z : ℂ) :
    A.charmatrix.map (Polynomial.evalRingHom z) = Matrix.scalar (Fin k) z-A := by
  ext i j
  simp [Matrix.charmatrix, Matrix.scalar, Matrix.diagonal_apply, apply_ite]

theorem eval_numerator (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) (z : ℂ) (i : Fin n) :
    (numerator m A b c i).eval z =
      contraction m (fun k => (Matrix.scalar (Fin (m k)) z-A k).adjugate) b c i := by
  change Polynomial.evalRingHom z (contraction m _ _ _ i) = _
  rw [map_contraction]
  simp only [map_adjugate', eval_charmatrix]
  simp only [Polynomial.coe_evalRingHom, Polynomial.eval_C]

theorem response_eq_unreduced (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) (z : ℂ) (i : Fin n) :
    response m A b c z i = (numerator m A b c i).eval z / (A i).charpoly.eval z := by
  rw [response, contraction_inverse, eval_numerator, Matrix.eval_charpoly]

theorem eigenvalue_iff (hn : 1 ≤ n)
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) (z : ℂ) :
    z ∈ spectrum ℂ (jacobian m A b c) ↔
      (∏ i, (A i).charpoly.eval z) - ∏ i, (numerator m A b c i).eval z = 0 := by
  rw [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot,
    characteristic_polynomial m hn]
  simp only [Polynomial.eval_sub, Polynomial.eval_prod]

theorem loop_equation_iff (hn : 1 ≤ n)
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) (z : ℂ)
    (hD : ∀ i, (A i).charpoly.eval z ≠ 0) :
    z ∈ spectrum ℂ (jacobian m A b c) ↔ ∏ i, response m A b c z i = 1 := by
  rw [eigenvalue_iff m hn, sub_eq_zero]
  simp_rw [response_eq_unreduced]
  rw [prod_div_distrib, div_eq_one_iff_eq (prod_ne_zero_iff.mpr fun i _ => hD i), eq_comm]

/-- The exact manuscript theorem: the full polynomial of the actual closed
matrix, multiplicities, and the response equation only off every internal
pole. There is no assumption about minimal realizations or nonzero outputs. -/
theorem prop_dyn_k1 (hn : 1 ≤ n)
    (A : ∀ i, Matrix (Fin (m i)) (Fin (m i)) ℂ)
    (b c : ∀ i, Fin (m i) → ℂ) :
    (jacobian m A b c).charpoly = (∏ i, (A i).charpoly) - ∏ i, numerator m A b c i ∧
    (∀ z : ℂ, Polynomial.rootMultiplicity z (jacobian m A b c).charpoly =
      Polynomial.rootMultiplicity z ((∏ i, (A i).charpoly) - ∏ i, numerator m A b c i)) ∧
    (∀ z : ℂ, z ∈ spectrum ℂ (jacobian m A b c) ↔
      (∏ i, (A i).charpoly.eval z) - ∏ i, (numerator m A b c i).eval z = 0) ∧
    (∀ z : ℂ, (∀ i, (A i).charpoly.eval z ≠ 0) →
      (∀ i, response m A b c z i = (numerator m A b c i).eval z / (A i).charpoly.eval z) ∧
      (z ∈ spectrum ℂ (jacobian m A b c) ↔ ∏ i, response m A b c z i = 1)) := by
  refine ⟨characteristic_polynomial m hn A b c, ?_, eigenvalue_iff m hn A b c, ?_⟩
  · intro z; rw [characteristic_polynomial m hn]
  · intro z hD
    exact ⟨response_eq_unreduced m A b c z, loop_equation_iff m hn A b c z hD⟩

end Dobson.Full3.BlockRing

#print axioms Dobson.Full3.BlockRing.prop_dyn_k1
