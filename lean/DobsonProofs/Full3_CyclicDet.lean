import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.Tactic

/-! The existing `Coverage1` cyclic determinant proof, with narrow imports.
This avoids loading the umbrella Mathlib module while proving the nonlinear
S/L model bridge. No new manuscript credit is claimed for this helper.
-/

namespace Dobson.Full3.CyclicDet

open scoped BigOperators

section DiagSubCyclic

variable {R : Type*} [CommRing R]

/-- General `n × n` matrix with diagonal `d` minus cyclic subdiagonal `c`:
entry `(i, j)` is `(if i = j then d i else 0) − (if i ≡ j + 1 (mod n) then c j else 0)`. -/
def diagSubCyclic (n : ℕ) (d c : Fin n → R) : Matrix (Fin n) (Fin n) R :=
  Matrix.of fun i j =>
    (if i = j then d i else 0) - (if (i : ℕ) = ((j : ℕ) + 1) % n then c j else 0)

private theorem succ_mod_add_two (n : ℕ) (j : Fin (n + 2)) :
    ((j : ℕ) + 1) % (n + 2) = if (j : ℕ) = n + 1 then 0 else (j : ℕ) + 1 := by
  have hj : (j : ℕ) < n + 2 := j.isLt
  split_ifs with h
  · rw [h]; exact Nat.mod_self _
  · exact Nat.mod_eq_of_lt (by omega)

private theorem diagSubCyclic_entry_add_two (n : ℕ) (d c : Fin (n + 2) → R)
    (i j : Fin (n + 2)) :
    diagSubCyclic (n + 2) d c i j =
      (if (i : ℕ) = (j : ℕ) then d i else 0) -
        (if (i : ℕ) = (if (j : ℕ) = n + 1 then 0 else (j : ℕ) + 1) then c j else 0) := by
  simp only [diagSubCyclic, Matrix.of_apply, succ_mod_add_two, Fin.ext_iff]

/-- Determinant of `diagSubCyclic (n + 2) d c` over any commutative ring `R`. -/
theorem det_diagSubCyclic_add_two (n : ℕ) (d c : Fin (n + 2) → R) :
    (diagSubCyclic (n + 2) d c).det = (∏ i : Fin (n + 2), d i) - ∏ i : Fin (n + 2), c i := by
  have hlow : ∀ i j : Fin (n + 1), i < j → diagSubCyclic (n + 2) d c i.succ j.succ = 0 := by
    intro i j hij
    have hij' : (i : ℕ) < (j : ℕ) := hij
    rw [diagSubCyclic_entry_add_two]
    have h1 : ((i.succ : ℕ) = (j.succ : ℕ)) → False := by simp only [Fin.val_succ]; omega
    have h2 : (i.succ : ℕ) ≠ (if (j.succ : ℕ) = n + 1 then 0 else (j.succ : ℕ) + 1) := by
      change (i : ℕ) + 1 ≠ (if (j : ℕ) + 1 = n + 1 then 0 else (j : ℕ) + 1 + 1)
      split_ifs <;> omega
    rw [ite_eq_right h1, ite_eq_right h2, sub_zero]
  have hlowdiag : ∀ i : Fin (n + 1), diagSubCyclic (n + 2) d c i.succ i.succ = d i.succ := by
    intro i
    rw [diagSubCyclic_entry_add_two, ite_eq_left rfl]
    have h2 : (i.succ : ℕ) ≠ (if (i.succ : ℕ) = n + 1 then 0 else (i.succ : ℕ) + 1) := by
      simp only [Fin.val_succ]; split_ifs <;> omega
    rw [ite_eq_right h2, sub_zero]
  have hup : ∀ i j : Fin (n + 1), j < i → diagSubCyclic (n + 2) d c i.succ j.castSucc = 0 := by
    intro i j hij
    have hij' : (j : ℕ) < (i : ℕ) := hij
    rw [diagSubCyclic_entry_add_two]
    have h1 : (i.succ : ℕ) ≠ (j.castSucc : ℕ) := by
      simp only [Fin.val_succ, Fin.val_castSucc]; omega
    have h2 : (i.succ : ℕ) ≠ (if (j.castSucc : ℕ) = n + 1 then 0 else (j.castSucc : ℕ) + 1) := by
      simp only [Fin.val_succ, Fin.val_castSucc]; split_ifs <;> omega
    rw [ite_eq_right h1, ite_eq_right h2, sub_zero]
  have hupdiag : ∀ i : Fin (n + 1),
      diagSubCyclic (n + 2) d c i.succ i.castSucc = -c i.castSucc := by
    intro i
    have hi : (i : ℕ) < n + 1 := i.isLt
    rw [diagSubCyclic_entry_add_two]
    have h1 : (i.succ : ℕ) ≠ (i.castSucc : ℕ) := by
      simp only [Fin.val_succ, Fin.val_castSucc]; omega
    have h2 : (i.succ : ℕ) = (if (i.castSucc : ℕ) = n + 1 then 0 else (i.castSucc : ℕ) + 1) := by
      simp only [Fin.val_succ, Fin.val_castSucc]; split_ifs <;> omega
    rw [ite_eq_right h1, ite_eq_left h2, zero_sub]
  have hrow0 : diagSubCyclic (n + 2) d c 0 0 = d 0 := by
    rw [diagSubCyclic_entry_add_two]
    simp
  have hrowlast : diagSubCyclic (n + 2) d c 0 (Fin.last (n + 1)) = -c (Fin.last (n + 1)) := by
    rw [diagSubCyclic_entry_add_two]
    simp
  have hrow : ∀ j : Fin (n + 2), j ≠ 0 → j ≠ Fin.last (n + 1) →
      diagSubCyclic (n + 2) d c 0 j = 0 := by
    intro j hj0 hjl
    have h2 : (j : ℕ) ≠ n + 1 := fun h => hjl (Fin.ext h)
    simp [diagSubCyclic, Ne.symm hj0, succ_mod_add_two, h2]
  have hdetL : ((diagSubCyclic (n + 2) d c).submatrix Fin.succ Fin.succ).det =
      ∏ i : Fin (n + 1), d i.succ := by
    rw [Matrix.det_of_isLowerTriangular]
    · simp only [Matrix.submatrix_apply, hlowdiag]
    · intro i j hij
      exact hlow i j (OrderDual.toDual_lt_toDual.mp hij)
  have hdetU : ((diagSubCyclic (n + 2) d c).submatrix Fin.succ Fin.castSucc).det =
      (-1) ^ (n + 1) * ∏ i : Fin (n + 1), c i.castSucc := by
    rw [Matrix.det_of_isUpperTriangular]
    · simp only [Matrix.submatrix_apply, hupdiag]
      rw [Finset.prod_neg, Finset.card_univ, Fintype.card_fin]
    · intro i j hij
      exact hup i j hij
  have hsq : ((-1 : R) ^ (n + 1)) * ((-1 : R) ^ (n + 1)) = 1 := by
    rw [← mul_pow]
    simp
  rw [Matrix.det_succ_row_zero,
    Finset.sum_eq_add (0 : Fin (n + 2)) (Fin.last (n + 1)) (by simp [Fin.ext_iff]) ?_
      (by simp) (by simp)]
  · rw [Fin.succAbove_zero, Fin.succAbove_last, hdetL, hdetU, hrow0, hrowlast,
      Fin.prod_univ_succ d, Fin.prod_univ_castSucc c]
    simp only [Fin.val_zero, Fin.val_last, pow_zero]
    linear_combination
      (-c (Fin.last (n + 1)) * ∏ i : Fin (n + 1), c i.castSucc) * hsq
  · intro j _ hj
    rw [hrow j hj.1 hj.2]
    ring

/-- The determinant formula for every nonempty ring, including overlapping
staying and transfer entries at one stage. -/
theorem det_diagSubCyclic (n : ℕ) (hn : 1 ≤ n) (d c : Fin n → R) :
    (diagSubCyclic n d c).det = (∏ i, d i) - ∏ i, c i := by
  obtain _ | _ | n := n
  · omega
  · rw [Matrix.det_fin_one, Fin.prod_univ_one, Fin.prod_univ_one]
    simp [diagSubCyclic]
  · exact det_diagSubCyclic_add_two n d c

/-- Singularity of the characteristic matrix is exactly the product equation.
Identification of its distinguished root with the Perron root is not asserted. -/
theorem characteristic_eq_iff (n : ℕ) (hn : 1 ≤ n) (s g : Fin n → R) (z : R) :
    (diagSubCyclic n (fun i => z - s i) g).det = 0 ↔
      (∏ i, (z - s i)) = ∏ i, g i := by
  rw [det_diagSubCyclic n hn, sub_eq_zero]

end DiagSubCyclic
end Dobson.Full3.CyclicDet

#print axioms Dobson.Full3.CyclicDet.det_diagSubCyclic_add_two
#print axioms Dobson.Full3.CyclicDet.det_diagSubCyclic
#print axioms Dobson.Full3.CyclicDet.characteristic_eq_iff
