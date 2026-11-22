import Mathlib

/-!
# Equal stages are the least stable (item E1)

Encoding: the `n` angles `θ_1, …, θ_n` are a function `θ : Fin n → ℝ` (Lean index `k` in
`{0, …, n−1}` stands for the mathematical index `k + 1`); sums and products run over all of
`Fin n`. "In the open interval `(0, π/2)`" is `θ k ∈ Set.Ioo 0 (π / 2)`.

Proof: `cos` is concave on `[−π/2, π/2]` and `log` is concave and increasing on `(0, ∞)`, so
by Jensen's inequality (twice) the mean of `log cos θ_k` is at most `log cos (π / n)`.
This is the statement that `−log cos` is convex, applied through Jensen.
-/

namespace Dobson.Secant

/-- The product form of E1 before taking reciprocals: under the hypotheses of E1,
`0 < cos (π / n)`, `0 < ∏ cos θ_k` and `∏ cos θ_k ≤ (cos (π / n))^n`. -/
theorem prod_cos_le (n : ℕ) (hn : 3 ≤ n) (θ : Fin n → ℝ)
    (hθ : ∀ k, θ k ∈ Set.Ioo 0 (Real.pi / 2)) (hsum : ∑ k, θ k = Real.pi) :
    0 < Real.cos (Real.pi / n) ∧ 0 < ∏ k, Real.cos (θ k)
      ∧ ∏ k, Real.cos (θ k) ≤ Real.cos (Real.pi / n) ^ n := by
  have hnpos : 0 < n := by omega
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hnpos
  have hcos : ∀ k, 0 < Real.cos (θ k) := fun k =>
    Real.cos_pos_of_mem_Ioo ⟨by linarith [(hθ k).1, Real.pi_pos], (hθ k).2⟩
  have hw0 : ∀ k ∈ (Finset.univ : Finset (Fin n)), (0 : ℝ) ≤ 1 / (n : ℝ) :=
    fun _ _ => by positivity
  have hw1 : ∑ _k : Fin n, (1 / (n : ℝ)) = 1 := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  -- Jensen for `cos` on `[−π/2, π/2]`
  have hJcos := strictConcaveOn_cos_Icc.concaveOn.le_map_sum (t := Finset.univ)
    (w := fun _ : Fin n => 1 / (n : ℝ)) (p := θ) hw0 hw1
    (fun k _ => ⟨by linarith [(hθ k).1, Real.pi_pos], (hθ k).2.le⟩)
  simp only [smul_eq_mul] at hJcos
  have hmean : ∑ k, 1 / (n : ℝ) * θ k = Real.pi / n := by
    rw [← Finset.mul_sum, hsum]
    ring
  rw [hmean] at hJcos
  -- Jensen for `log` on `(0, ∞)`
  have hJlog := strictConcaveOn_log_Ioi.concaveOn.le_map_sum (t := Finset.univ)
    (w := fun _ : Fin n => 1 / (n : ℝ)) (p := fun k => Real.cos (θ k)) hw0 hw1
    (fun k _ => Set.mem_Ioi.mpr (hcos k))
  simp only [smul_eq_mul] at hJlog
  have hpos : 0 < ∑ k : Fin n, 1 / (n : ℝ) * Real.cos (θ k) := by
    have : Nonempty (Fin n) := ⟨⟨0, hnpos⟩⟩
    exact Finset.sum_pos (fun k _ => mul_pos (by positivity) (hcos k)) Finset.univ_nonempty
  have hcpos : 0 < Real.cos (Real.pi / n) := lt_of_lt_of_le hpos hJcos
  have hlog : ∑ k, 1 / (n : ℝ) * Real.log (Real.cos (θ k))
      ≤ Real.log (Real.cos (Real.pi / n)) :=
    hJlog.trans (Real.log_le_log hpos hJcos)
  have hprodpos : 0 < ∏ k, Real.cos (θ k) := Finset.prod_pos (fun k _ => hcos k)
  refine ⟨hcpos, hprodpos, ?_⟩
  rw [← Real.log_le_log_iff hprodpos (pow_pos hcpos n),
    Real.log_prod (fun k _ => (hcos k).ne'), Real.log_pow]
  rw [← Finset.mul_sum] at hlog
  have h := mul_le_mul_of_nonneg_left hlog hn0.le
  rw [← mul_assoc, mul_one_div_cancel hn0.ne', one_mul] at h
  exact h

/-- **E1.** For a natural number `n ≥ 3` and angles `θ_1, …, θ_n` in the open interval
`(0, π/2)` with `∑ θ_k = π`: `∏ (1 / cos θ_k) ≥ (1 / cos (π / n))^n`. -/
theorem E1 (n : ℕ) (hn : 3 ≤ n) (θ : Fin n → ℝ)
    (hθ : ∀ k, θ k ∈ Set.Ioo 0 (Real.pi / 2)) (hsum : ∑ k, θ k = Real.pi) :
    ∏ k, (1 / Real.cos (θ k)) ≥ (1 / Real.cos (Real.pi / n)) ^ n := by
  obtain ⟨_, hprodpos, hle⟩ := prod_cos_le n hn θ hθ hsum
  show (1 / Real.cos (Real.pi / n)) ^ n ≤ ∏ k, (1 / Real.cos (θ k))
  rw [one_div_pow, Finset.prod_div_distrib, Finset.prod_const_one]
  exact one_div_le_one_div_of_le hprodpos hle

end Dobson.Secant

#print axioms Dobson.Secant.prod_cos_le
#print axioms Dobson.Secant.E1
