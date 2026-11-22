import Mathlib.Data.Matrix.Basis
import Mathlib.Analysis.Complex.Norm
import Mathlib.Tactic

/-!
# Coverage batch 3: lineage reduction and spectral algebra
Cores of `prop:ring-lineage` and `prop:ring-spectrum`. First-return renewal and
correlation-sum inputs are explicit. Spectral integration, interchange of infinite
sums, low-frequency limits, and the AR(2) stochastic construction are not asserted.
-/
/-! Audit correction, 2026-11-16 (auditfix1).
`mixture_coefficient.hcost` is class (b), the lineage perturbation-cost
identity used in A-ring.tex's proof of prop:ring-spectrum. The rank-one
and finite-mixture algebra here do not identify C with the stochastic
cost coefficient. That step remains unproved in Lean and the dependent
prop:ring-spectrum row is partial. The return and geometric identities
alone cannot determine an otherwise unconstrained C.
-/

namespace Dobson.Coverage3Renewal
open scoped BigOperators

/-- The actual rank-one matrix contraction, with H = M^(ell-1) in the application. -/
theorem rank_one_reduction {ι : Type*} [Fintype ι] [DecidableEq ι]
    (H : Matrix ι ι ℝ) (v w : ι → ℝ) (i j : ι) (b : ℝ) :
    w ⬝ᵥ ((Matrix.single i j b * (H - Matrix.vecMulVec v w) *
      Matrix.single i j b).mulVec v) =
      (b * w i * v j) * (b * H j i - b * w i * v j) := by
  rw [Matrix.single_mul_mul_single, Matrix.single_mulVec_eq, dotProduct_smul,
    dotProduct_single]
  simp only [Matrix.sub_apply, Matrix.vecMulVec_apply, smul_eq_mul]
  ring

/-- Ordered pairs of distinct labeled edges retain their direction and weights. -/
theorem ordered_pair_reduction {ι : Type*} [Fintype ι] [DecidableEq ι]
    (H : Matrix ι ι ℝ) (v w : ι → ℝ) (ia ja ib jb : ι) (ba bb : ℝ)
    (hw : w ib ≠ 0) :
    w ⬝ᵥ ((Matrix.single ia ja ba * (H - Matrix.vecMulVec v w) *
      Matrix.single ib jb bb).mulVec v) =
      (bb * w ib * v jb) *
        (ba * w ia / w ib * H ja ib - ba * w ia * v ja) := by
  rw [Matrix.single_mul_mul_single, Matrix.single_mulVec_eq, dotProduct_smul,
    dotProduct_single]
  simp only [Matrix.sub_apply, Matrix.vecMulVec_apply, smul_eq_mul]
  field_simp

/-- The generating-function renewal equation is solved, not assumed solved. -/
theorem renewal_solution {K : Type*} [Field K] (U R : K) (hR : 1 - R ≠ 0)
    (hrenewal : U = R + R * U) : U = R / (1 - R) := by
  apply (eq_div_iff hR).mpr
  linear_combination hrenewal

/-- Geometric-memory reduction from the first-return renewal equation. -/
theorem renewal_coefficient (T θ R U : ℝ) (hT : T ≠ 0) (hR : 1 - R ≠ 0)
    (hrenewal : U = R + R * U) :
    (1 / T) ^ 2 / 2 - (1 / T) * (U - θ / T) =
      (1 + 2 * θ) / (2 * T ^ 2) - (1 / T) * R / (1 - R) := by
  rw [renewal_solution U R hR hrenewal]
  field_simp
  ring

/-- Finite complex mixtures reduce to the manuscript expression, conditional on
correlation/return sums. The sum identities must be proved for the stochastic model. -/
theorem mixture_coefficient {ι : Type*} (S : Finset ι) (B z R U G : ι → ℂ)
    (T C : ℂ) (hT : T ≠ 0)
    (hcost : C = 1 / (2 * T ^ 2) - 1 / T *
      ∑ j ∈ S, B j * (U j - G j / T))
    (hreturn : ∀ j ∈ S, U j = R j / (1 - R j))
    (hgeom : ∀ j ∈ S, G j = z j / (1 - z j)) :
    C = (1 + 2 * ∑ j ∈ S, B j * z j / (1 - z j)) / (2 * T ^ 2) -
      1 / T * ∑ j ∈ S, B j * R j / (1 - R j) := by
  have hs : (∑ j ∈ S, B j * (U j - G j / T)) =
      (∑ j ∈ S, B j * R j / (1 - R j)) -
        (∑ j ∈ S, B j * z j / (1 - z j)) / T := by
    rw [Finset.sum_div, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro j hj
    rw [hreturn j hj, hgeom j hj]
    ring
  rw [hcost, hs]
  field_simp
  ring

/-- The spectral real part is an explicit signed numerator over a positive norm. -/
theorem spectral_real_part (g : ℂ) :
    (g / (1 - g)).re = (g.re - Complex.normSq g) / Complex.normSq (1 - g) := by
  rw [Complex.div_re]
  simp only [Complex.sub_re, Complex.sub_im, Complex.one_re, Complex.one_im,
    Complex.normSq_apply]
  ring

/-- Exact negative-kernel criterion, excluding the pole g = 1. -/
theorem spectral_negative_iff (T : ℝ) (hT : 0 < T) (g : ℂ) (hg : g ≠ 1) :
    -(1 / T) * (g / (1 - g)).re < 0 ↔ Complex.normSq g < g.re := by
  rw [spectral_real_part]
  have hd : 0 < Complex.normSq (1 - g) :=
    Complex.normSq_pos.mpr (sub_ne_zero.mpr (Ne.symm hg))
  have ht : 0 < 1 / T := one_div_pos.mpr hT
  constructor
  · intro h
    have hnum : 0 < (g.re - Complex.normSq g) / Complex.normSq (1 - g) := by
      nlinarith
    have := (div_pos_iff_of_pos_right hd).mp hnum
    linarith
  · intro h
    have hnum := div_pos (sub_pos.mpr h) hd
    nlinarith [mul_pos ht hnum]

end Dobson.Coverage3Renewal
