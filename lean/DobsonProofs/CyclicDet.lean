import Mathlib

/-!
# Determinant of the cyclic matrix (items Y1 and Y2)

The mathematical matrix is `n × n`, with ones on the diagonal and `−t_k` in position
`(k+1, k)` for `k = 1, …, n`, indices cyclic (so `−t_n` sits in position `(1, n)`).

Encoding.
* Y1 is stated for the matrices written out entry by entry (`!![row; row; …]`, rows separated
  by `;`), for `n = 2, 3, 4`, over an arbitrary commutative ring `R`.
* For general `n` the matrix is `cyclicMat n t`, indexed by `Fin n = {0, …, n−1}` (so the
  mathematical index `k` is the Lean index `k − 1`): its `(i, j)` entry is
  `[i = j] − [i ≡ j + 1 (mod n)]·t j`.
  For `n ≥ 2` this is literally "ones on the diagonal, `−t j` in position `(j+1, j)` cyclically,
  zero elsewhere". For `n = 1` the two positions coincide and the single entry is `1 − t 0`.
  The lemmas `cyclicMat_two`, `cyclicMat_three`, `cyclicMat_four` check that for `n = 2, 3, 4`
  the general definition is exactly the written-out matrix of Y1.
-/

namespace Dobson.CyclicDet

variable {R : Type*} [CommRing R]

/-- **Y1, `n = 2`.** `det [[1, −t₂], [−t₁, 1]] = 1 − t₁ t₂`. -/
theorem det_two (t₁ t₂ : R) :
    Matrix.det !![1, -t₂;
                  -t₁, 1] = 1 - t₁ * t₂ := by
  rw [Matrix.det_fin_two_of]
  ring

/-- **Y1, `n = 3`.** `det [[1, 0, −t₃], [−t₁, 1, 0], [0, −t₂, 1]] = 1 − t₁ t₂ t₃`. -/
theorem det_three (t₁ t₂ t₃ : R) :
    Matrix.det !![1, 0, -t₃;
                  -t₁, 1, 0;
                  0, -t₂, 1] = 1 - t₁ * t₂ * t₃ := by
  rw [Matrix.det_fin_three]
  simp
  ring

/-- **Y1, `n = 4`.**
`det [[1, 0, 0, −t₄], [−t₁, 1, 0, 0], [0, −t₂, 1, 0], [0, 0, −t₃, 1]] = 1 − t₁ t₂ t₃ t₄`. -/
theorem det_four (t₁ t₂ t₃ t₄ : R) :
    Matrix.det !![1, 0, 0, -t₄;
                  -t₁, 1, 0, 0;
                  0, -t₂, 1, 0;
                  0, 0, -t₃, 1] = 1 - t₁ * t₂ * t₃ * t₄ := by
  simp [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  ring

/-- The general cyclic matrix, indexed by `Fin n = {0, …, n−1}`:
entry `(i, j)` is `[i = j] − [i ≡ j + 1 (mod n)]·t j`. -/
def cyclicMat (n : ℕ) (t : Fin n → R) : Matrix (Fin n) (Fin n) R :=
  Matrix.of fun i j =>
    (if i = j then 1 else 0) - (if (i : ℕ) = ((j : ℕ) + 1) % n then t j else 0)

theorem cyclicMat_apply (n : ℕ) (t : Fin n → R) (i j : Fin n) :
    cyclicMat n t i j
      = (if i = j then 1 else 0) - (if (i : ℕ) = ((j : ℕ) + 1) % n then t j else 0) :=
  rfl

/-- For `n = 2` the general matrix is the written-out matrix of `det_two`. -/
theorem cyclicMat_two (t₁ t₂ : R) :
    cyclicMat 2 ![t₁, t₂] = !![1, -t₂;
                               -t₁, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [cyclicMat_apply]

/-- For `n = 3` the general matrix is the written-out matrix of `det_three`. -/
theorem cyclicMat_three (t₁ t₂ t₃ : R) :
    cyclicMat 3 ![t₁, t₂, t₃] = !![1, 0, -t₃;
                                   -t₁, 1, 0;
                                   0, -t₂, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [cyclicMat_apply]

/-- For `n = 4` the general matrix is the written-out matrix of `det_four`. -/
theorem cyclicMat_four (t₁ t₂ t₃ t₄ : R) :
    cyclicMat 4 ![t₁, t₂, t₃, t₄] = !![1, 0, 0, -t₄;
                                       -t₁, 1, 0, 0;
                                       0, -t₂, 1, 0;
                                       0, 0, -t₃, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [cyclicMat_apply]

private theorem succ_mod (n : ℕ) (j : Fin (n + 2)) :
    ((j : ℕ) + 1) % (n + 2) = if (j : ℕ) = n + 1 then 0 else (j : ℕ) + 1 := by
  have hj : (j : ℕ) < n + 2 := j.isLt
  split_ifs with h
  · rw [h]
    exact Nat.mod_self _
  · exact Nat.mod_eq_of_lt (by omega)

private theorem entry (n : ℕ) (t : Fin (n + 2) → R) (i j : Fin (n + 2)) :
    cyclicMat (n + 2) t i j
      = (if (i : ℕ) = (j : ℕ) then 1 else 0)
        - (if (i : ℕ) = (if (j : ℕ) = n + 1 then 0 else (j : ℕ) + 1) then t j else 0) := by
  rw [cyclicMat_apply, succ_mod]
  simp only [Fin.ext_iff]

private theorem entry_eq_zero (n : ℕ) (t : Fin (n + 2) → R) (i j : Fin (n + 2))
    (h1 : (i : ℕ) ≠ (j : ℕ))
    (h2 : (i : ℕ) ≠ (if (j : ℕ) = n + 1 then 0 else (j : ℕ) + 1)) :
    cyclicMat (n + 2) t i j = 0 := by
  rw [entry, ite_eq_right h1, ite_eq_right h2, sub_zero]

private theorem entry_eq_one (n : ℕ) (t : Fin (n + 2) → R) (i j : Fin (n + 2))
    (h1 : (i : ℕ) = (j : ℕ))
    (h2 : (i : ℕ) ≠ (if (j : ℕ) = n + 1 then 0 else (j : ℕ) + 1)) :
    cyclicMat (n + 2) t i j = 1 := by
  rw [entry, ite_eq_left h1, ite_eq_right h2, sub_zero]

private theorem entry_eq_neg (n : ℕ) (t : Fin (n + 2) → R) (i j : Fin (n + 2))
    (h1 : (i : ℕ) ≠ (j : ℕ))
    (h2 : (i : ℕ) = (if (j : ℕ) = n + 1 then 0 else (j : ℕ) + 1)) :
    cyclicMat (n + 2) t i j = -t j := by
  rw [entry, ite_eq_right h1, ite_eq_left h2, zero_sub]

/-- Y2 for size `n + 2` (that is, size at least 2). -/
theorem det_cyclicMat_add_two (n : ℕ) (t : Fin (n + 2) → R) :
    (cyclicMat (n + 2) t).det = 1 - ∏ i, t i := by
  -- entries of the two minors and of row 0
  have hlow : ∀ i j : Fin (n + 1), i < j → cyclicMat (n + 2) t i.succ j.succ = 0 := by
    intro i j hij
    have hij' : (i : ℕ) < (j : ℕ) := hij
    refine entry_eq_zero n t i.succ j.succ ?_ ?_
    · simp only [Fin.val_succ]
      omega
    · simp only [Fin.val_succ]
      split_ifs <;> omega
  have hlowdiag : ∀ i : Fin (n + 1), cyclicMat (n + 2) t i.succ i.succ = 1 := by
    intro i
    refine entry_eq_one n t i.succ i.succ rfl ?_
    simp only [Fin.val_succ]
    split_ifs <;> omega
  have hup : ∀ i j : Fin (n + 1), j < i → cyclicMat (n + 2) t i.succ j.castSucc = 0 := by
    intro i j hij
    have hij' : (j : ℕ) < (i : ℕ) := hij
    refine entry_eq_zero n t i.succ j.castSucc ?_ ?_
    · simp only [Fin.val_succ, Fin.val_castSucc]
      omega
    · simp only [Fin.val_succ, Fin.val_castSucc]
      split_ifs <;> omega
  have hupdiag : ∀ i : Fin (n + 1),
      cyclicMat (n + 2) t i.succ i.castSucc = -t i.castSucc := by
    intro i
    have hi : (i : ℕ) < n + 1 := i.isLt
    refine entry_eq_neg n t i.succ i.castSucc ?_ ?_
    · simp only [Fin.val_succ, Fin.val_castSucc]
      omega
    · simp only [Fin.val_succ, Fin.val_castSucc]
      split_ifs <;> omega
  have hrow0 : cyclicMat (n + 2) t 0 0 = 1 :=
    entry_eq_one n t 0 0 rfl (by simp)
  have hrowlast : cyclicMat (n + 2) t 0 (Fin.last (n + 1)) = -t (Fin.last (n + 1)) :=
    entry_eq_neg n t 0 (Fin.last (n + 1)) (by simp) (by simp)
  have hrow : ∀ c : Fin (n + 2), c ≠ 0 → c ≠ Fin.last (n + 1) →
      cyclicMat (n + 2) t 0 c = 0 := by
    intro c hc0 hcl
    have h1 : (c : ℕ) ≠ 0 := fun h => hc0 (Fin.ext h)
    have h2 : (c : ℕ) ≠ n + 1 := fun h => hcl (Fin.ext h)
    refine entry_eq_zero n t 0 c ?_ ?_
    · simp only [Fin.val_zero]
      omega
    · simp only [Fin.val_zero]
      split_ifs <;> omega
  -- the two minors are triangular
  have hdetL : ((cyclicMat (n + 2) t).submatrix Fin.succ Fin.succ).det = 1 := by
    rw [Matrix.det_of_isLowerTriangular]
    · simp only [Matrix.submatrix_apply, hlowdiag, Finset.prod_const_one]
    · intro i j hij
      exact hlow i j (OrderDual.toDual_lt_toDual.mp hij)
  have hdetU : ((cyclicMat (n + 2) t).submatrix Fin.succ Fin.castSucc).det
      = (-1) ^ (n + 1) * ∏ i : Fin (n + 1), t i.castSucc := by
    rw [Matrix.det_of_isUpperTriangular]
    · simp only [Matrix.submatrix_apply, hupdiag]
      rw [Finset.prod_neg, Finset.card_univ, Fintype.card_fin]
    · intro i j hij
      exact hup i j hij
  have hsq : ((-1 : R) ^ (n + 1)) * ((-1 : R) ^ (n + 1)) = 1 := by
    rw [← mul_pow]
    simp
  -- Laplace expansion along row 0: only columns 0 and `last` contribute
  rw [Matrix.det_succ_row_zero,
    Finset.sum_eq_add (0 : Fin (n + 2)) (Fin.last (n + 1)) (by simp [Fin.ext_iff]) ?_
      (by simp) (by simp)]
  · rw [Fin.succAbove_zero, Fin.succAbove_last, hdetL, hdetU, hrow0, hrowlast,
      Fin.prod_univ_castSucc t]
    simp only [Fin.val_zero, Fin.val_last, pow_zero]
    linear_combination
      (-t (Fin.last (n + 1)) * ∏ i : Fin (n + 1), t i.castSucc) * hsq
  · intro c _ hc
    rw [hrow c hc.1 hc.2]
    ring

/-- **Y2.** For every `n ≥ 1` and every `t : {0, …, n−1} → R` over a commutative ring `R`,
the `n × n` matrix with ones on the diagonal and `−t j` in position `(j+1, j)` (indices
cyclic) has determinant `1 − ∏ t j`. -/
theorem det_cyclicMat (n : ℕ) (hn : 1 ≤ n) (t : Fin n → R) :
    (cyclicMat n t).det = 1 - ∏ i, t i := by
  obtain _ | _ | n := n
  · omega
  · rw [Matrix.det_fin_one, Fin.prod_univ_one, cyclicMat_apply]
    simp
  · exact det_cyclicMat_add_two n t

end Dobson.CyclicDet

#print axioms Dobson.CyclicDet.det_two
#print axioms Dobson.CyclicDet.det_three
#print axioms Dobson.CyclicDet.det_four
#print axioms Dobson.CyclicDet.cyclicMat_two
#print axioms Dobson.CyclicDet.cyclicMat_three
#print axioms Dobson.CyclicDet.cyclicMat_four
#print axioms Dobson.CyclicDet.det_cyclicMat
