import DobsonProofs.Full4_LeftHalfPlane
import DobsonProofs.Full3_BlockRing

/-! A defined unself-regulated host block followed by a finite L cascade.
The characteristic polynomial is derived from that block matrix. The host
block itself is entered explicitly, not obtained by nonlinear differentiation;
this calculation supports the core coverage of `prop:dyn-k5`. -/
namespace Dobson.Full4.Cascade
open Polynomial Matrix Finset Complex
open scoped BigOperators
open Dobson.Full4.LeftHalfPlane
namespace BR
export Dobson.Full3.BlockRing (jacobian characteristic_polynomial numerator contraction State jacobian_entry)
end BR

/-- Host state dimension two; each subsequent L block has one state. -/
def dims (n : ℕ) : Fin (n+1) → ℕ := Fin.cases 2 (fun _ => 1)

def blocks {n : ℕ} (d h : ℝ) (a : Fin n → ℝ) :
    (i : Fin (n+1)) → Matrix (Fin (dims n i)) (Fin (dims n i)) ℂ :=
  Fin.cases (!![(h : ℂ), -(h : ℂ); (d : ℂ), -(d : ℂ)])
    (fun j => !![-(a j : ℂ)])

def inputs {n : ℕ} (q : ℝ) (a : Fin n → ℝ) :
    (i : Fin (n+1)) → Fin (dims n i) → ℂ :=
  Fin.cases ![0, (q : ℂ)] (fun j => ![(a j : ℂ)])

def outputs (n : ℕ) : (i : Fin (n+1)) → Fin (dims n i) → ℂ :=
  Fin.cases ![0, 1] (fun _ => ![1])

noncomputable def jacobian {n : ℕ} (d q h : ℝ) (a : Fin n → ℝ) :=
  BR.jacobian (dims n) (blocks d h a) (inputs q a) (outputs n)

noncomputable def polynomial {n : ℕ} (d q h : ℝ) (a : Fin n → ℝ) : ℂ[X] :=
  X * (X + C ((d-h : ℝ) : ℂ)) * (∏ j, (X + C (a j : ℂ))) -
    C (q : ℂ) * (X - C (h : ℂ)) * C (∏ j, (a j : ℂ))

theorem characteristic_polynomial {n : ℕ} (d q h : ℝ) (a : Fin n → ℝ) :
    (jacobian d q h a).charpoly = polynomial d q h a := by
  rw [jacobian, BR.characteristic_polynomial _ (by omega)]
  rw [Fin.prod_univ_succ, Fin.prod_univ_succ]
  have hhost : (blocks d h a 0).charpoly = X*(X+C ((d-h : ℝ) : ℂ)) := by
    change (!![(h : ℂ), -(h : ℂ); (d : ℂ), -(d : ℂ)]).charpoly = _
    simp [Matrix.charpoly, Matrix.charmatrix, Matrix.det_fin_two,
      Matrix.scalar, Matrix.diagonal_apply]
    push_cast
    ring
  have hdelay (j : Fin n) : (blocks d h a j.succ).charpoly = X+C (a j : ℂ) := by
    change (!![-(a j : ℂ)]).charpoly = _
    simp [Matrix.charpoly, Matrix.charmatrix, Matrix.det_fin_one,
      Matrix.scalar, Matrix.diagonal_apply]
  have hnHost : BR.numerator (dims n) (blocks d h a) (inputs q a) (outputs n) 0 =
      C (q : ℂ)*(X-C (h : ℂ)) := by
    change (∑ p : Fin 2, ∑ r : Fin 2, C ((![(0 : ℂ),1]) p) *
      (!![(h : ℂ), -(h : ℂ); (d : ℂ), -(d : ℂ)]).charmatrix.adjugate p r *
      C ((![(0 : ℂ), (q : ℂ)]) r)) = _
    simp [Matrix.charmatrix, Matrix.adjugate_fin_two, Matrix.scalar, Matrix.diagonal_apply,
      Fin.sum_univ_two]
    ring
  have hnDelay (j : Fin n) :
      BR.numerator (dims n) (blocks d h a) (inputs q a) (outputs n) j.succ =
      C (a j : ℂ) := by
    change (∑ p : Fin 1, ∑ r : Fin 1, C ((![(1 : ℂ)]) p) *
      (!![-(a j : ℂ)]).charmatrix.adjugate p r * C ((![(a j : ℂ)]) r)) = _
    simp [Matrix.adjugate_fin_one]
  simp only [hhost, hdelay, hnHost, hnDelay, polynomial, map_prod]

theorem eval_polynomial {n : ℕ} (d q h : ℝ) (a : Fin n → ℝ) (z : ℂ) :
    (polynomial d q h a).eval z =
      z*(z+((d-h : ℝ) : ℂ))*(∏ j, (z+(a j : ℂ))) -
        (q : ℂ)*(z-(h : ℂ))*(∏ j, (a j : ℂ)) := by
  simp only [polynomial, eval_sub, eval_mul, eval_add, eval_X, eval_C, eval_prod]

theorem derivative_at_zero {n : ℕ} (d q h : ℝ) (a : Fin n → ℝ) :
    (polynomial d q h a).derivative.eval 0 = ((d-h-q : ℝ) : ℂ)*(∏ j, (a j : ℂ)) := by
  simp only [polynomial, derivative_sub, derivative_mul, derivative_add, derivative_X,
    derivative_C, eval_sub, eval_add, eval_mul, eval_X, eval_C, eval_one, eval_zero,
    eval_prod, zero_mul, mul_zero, one_mul, mul_one, add_zero, zero_add, sub_zero]
  push_cast
  ring

theorem trace_jacobian {n : ℕ} (hn : 1 ≤ n) (d q h : ℝ) (a : Fin n → ℝ) :
    (jacobian d q h a).trace = -((d-h : ℝ) : ℂ) - ∑ i, (a i : ℂ) := by
  have hneq (i : Fin (n+1)) : i ≠ finRotate (n+1) i := by
    intro he
    have hv := congrArg Fin.val he
    rw [coe_finRotate] at hv
    split_ifs at hv <;> simp_all <;> omega
  unfold jacobian Matrix.trace Matrix.diag
  simp only [Fintype.sum_sigma, BR.jacobian_entry, hneq, if_false, add_zero,
    Matrix.blockDiagonal'_apply_eq]
  rw [Fin.sum_univ_succ]
  change (∑ x : Fin 2, (!![(h : ℂ), -(h : ℂ); (d : ℂ), -(d : ℂ)]) x x) +
    (∑ i : Fin n, ∑ x : Fin 1, (!![-(a i : ℂ)]) x x) = _
  simp [Fin.sum_univ_two]
  push_cast
  ring

private theorem logDerivative_nonneg (p : ℂ[X]) (hp0 : p.eval 0 ≠ 0)
    (h : ClosedLeft p.eval) : 0 ≤ (p.derivative.eval 0 / p.eval 0).re := by
  have hp : p ≠ 0 := by rintro rfl; simp at hp0
  rw [(IsAlgClosed.splits p).eval_derivative_div_eval_of_ne_zero hp0]
  change 0 ≤ Complex.reAddGroupHom _
  rw [map_multiset_sum]
  simp only [Multiset.map_map, Function.comp_def]
  apply Multiset.sum_nonneg
  intro v hv
  obtain ⟨z,hz,rfl⟩ := Multiset.mem_map.mp hv
  change 0 ≤ (1/(0-z)).re
  simp only [one_div, zero_sub, inv_neg, Complex.neg_re, Complex.inv_re]
  exact neg_nonneg.mpr (div_nonpos_of_nonpos_of_nonneg
    (h z ((Polynomial.mem_roots hp).mp hz)) (Complex.normSq_nonneg z))

/-- Every nonempty finite L cascade is strictly unstable when J<=0, including
J=0: there is a positive-real-part eigenvalue of the actual matrix. -/
theorem cascade_unstable {n : ℕ} (hn : 1 ≤ n) (d q h : ℝ) (a : Fin n → ℝ)
    (hq : 0 < q) (hh : 0 < h) (ha : ∀ i, 0 < a i) (hJ : d-h-q ≤ 0) :
    ∃ z : ℂ, z ∈ spectrum ℂ (jacobian d q h a) ∧ 0 < z.re := by
  have hap : 0 < ∏ i, a i := Finset.prod_pos (fun i _ => ha i)
  have hqc : (q : ℂ) ≠ 0 := by exact_mod_cast hq.ne'
  have hhc : (h : ℂ) ≠ 0 := by exact_mod_cast hh.ne'
  have hac : (∏ i, (a i : ℂ)) ≠ 0 := by
    exact Finset.prod_ne_zero_iff.mpr (fun i _ => by exact_mod_cast (ha i).ne')
  have hp0 : (polynomial d q h a).eval 0 ≠ 0 := by
    rw [eval_polynomial]
    simpa using mul_ne_zero (mul_ne_zero hqc hhc) hac
  have heval : ((polynomial d q h a).derivative.eval 0 /
      (polynomial d q h a).eval 0).re = (d-h-q)/(q*h) := by
    rw [derivative_at_zero, eval_polynomial]
    have he : (((d-h-q : ℝ) : ℂ)*(∏ i, (a i : ℂ))) /
        (0*(0+((d-h : ℝ) : ℂ))*(∏ i, (0+(a i : ℂ))) -
          (q : ℂ)*(0-(h : ℂ))*(∏ i, (a i : ℂ))) = (((d-h-q)/(q*h) : ℝ) : ℂ) := by
      push_cast
      field_simp
      ring
    exact congrArg Complex.re he
  by_contra! hnroot
  have hleft : ClosedLeft (polynomial d q h a).eval := by
    intro z hz
    apply hnroot z
    rw [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot, characteristic_polynomial]
    exact hz
  rcases lt_or_eq_of_le hJ with hj | hj
  · have hl := logDerivative_nonneg (polynomial d q h a) hp0 hleft
    rw [heval] at hl
    exact (not_lt_of_ge hl) (div_neg_of_neg_of_pos hj (mul_pos hq hh))
  · have hsum : ((polynomial d q h a).roots.map Complex.re).sum < 0 := by
      rw [← characteristic_polynomial]
      change ((jacobian d q h a).charpoly.roots.map Complex.reAddGroupHom).sum < 0
      rw [← map_multiset_sum, ← Matrix.trace_eq_sum_roots_charpoly, trace_jacobian hn]
      simp only [map_sub, map_neg, Complex.ofReal_re, map_sum]
      change -(d-h) - ∑ i, a i < 0
      have has : 0 ≤ ∑ i, a i := Finset.sum_nonneg (fun i _ => (ha i).le)
      linarith
    have hl := logDerivative_pos (polynomial d q h a) hp0 hleft hsum
    rw [heval, hj, zero_div] at hl
    exact lt_irrefl 0 hl

end Dobson.Full4.Cascade
