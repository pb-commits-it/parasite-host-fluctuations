import Mathlib.Tactic

/-!
# Coverage batch 2: alternative hosts and fluctuating staying
Core of `prop:ring-alt`. The renewal and perturbation expansions are supplied
explicitly. No stationary random process or fourth-order remainder is asserted.
-/
/-! Audit correction, 2026-11-16 (auditfix1).
`alternative_loss.hexpansion` is class (b), the model-specific per-step
cost derived in A-ring.tex's proof of prop:ring-alt from prop:ring-lineage
and the independent-host perturbation expansion. It is not a definition
of the stochastic loss. It remains unproved for the model in Lean, so
prop:ring-alt is partial. Multiplication by generation time and the
share-factor algebra do not establish that missing coefficient identity.
-/

namespace Dobson.Coverage2Alternatives
open scoped BigOperators

/-- Convert a per-step multi-contribution expansion to the generation coefficient. -/
theorem alternative_loss {ι : Type*} (S : Finset ι) (π v θ R : ι → ℝ)
    (T C : ℝ) (hT : T ≠ 0)
    (hexpansion : C = ∑ j ∈ S, π j ^ 2 * v j *
      ((1 + 2 * θ j) / (2 * T ^ 2) - 1 / T * (R j / (1 - R j)))) :
    T * C = ∑ j ∈ S, π j ^ 2 * v j *
      ((1 + 2 * θ j) / (2 * T) - R j / (1 - R j)) := by
  rw [hexpansion, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  field_simp

/-- Equal independent shares give exactly the factor `1/M`. -/
theorem equal_shares (M : ℕ) (hM : 0 < M) (v K : ℝ) :
    (∑ _j : Fin M, (1 / (M : ℝ)) ^ 2 * v * K) = v * K / M := by
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt hM
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- The unequal-variance example is a factor `2.5`, not a reduction. -/
theorem unequal_variances (v K : ℝ) :
    (1 / 2 : ℝ) ^ 2 * v * K + (1 / 2 : ℝ) ^ 2 * (9 * v) * K =
      (5 / 2 : ℝ) * v * K := by ring

/-- The diagonal and ordered off-diagonal covariance terms give the stated factor. -/
theorem correlated_shares (M ρ : ℝ) (hM : M ≠ 0) :
    M * (1 / M) ^ 2 + M * (M - 1) * ρ * (1 / M) ^ 2 =
      (1 + ρ * (M - 1)) / M := by field_simp

/-- The staying-return identity follows from its renewal equation and circuit PGF. -/
theorem staying_return (m ψ A W : ℝ)
    (hm : 1 - m * ψ ≠ 0) (hW : 1 - W ≠ 0)
    (hR : 1 - (1 - m) * ψ * A / (1 - m * ψ) ≠ 0)
    (hrenewal : W = m * ψ + (1 - m) * ψ * A * W) :
    W / (1 - W) = m * ψ /
      ((1 - m * ψ) * (1 - (1 - m) * ψ * A / (1 - m * ψ))) := by
  apply (div_eq_div_iff hW (mul_ne_zero hm hR)).mpr
  field_simp
  nlinarith [hrenewal]

/-- For positive staying frequency and variance, the gain criterion is an iff. -/
theorem staying_gain_iff (v e θ G : ℝ) (hv : 0 < v) (he : 0 < e) :
    v * (e ^ 2 * (1 + 2 * θ) / 2 - e * G) < 0 ↔
      e * (1 + 2 * θ) / 2 < G := by
  have hid : v * (e ^ 2 * (1 + 2 * θ) / 2 - e * G) =
      (v * e) * (e * (1 + 2 * θ) / 2 - G) := by ring
  rw [hid]
  constructor
  · intro h
    by_contra hn
    have := mul_nonneg (mul_pos hv he).le (sub_nonneg.mpr (le_of_not_gt hn))
    linarith
  · intro h
    exact mul_neg_of_pos_of_neg (mul_pos hv he) (sub_neg.mpr h)

end Dobson.Coverage2Alternatives
