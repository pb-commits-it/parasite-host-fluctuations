import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Tactic

/-!
# The long-stay logarithm series has zero convergence radius

Core of `prop:exact-mem-zero-radius`, for the correct long-stay variable z.
The input is a spectral moment sequence whose moments dominate M^k times a
positive tail mass for every M>0. This lower bound is the explicit probability
handoff: the manuscript derives it from unbounded host support and spectral
moment domination. The conclusion proves divergence at every nonzero argument;
it is neither noise-parameter nonanalyticity nor an assumed remainder.

The identification of these coefficients with growth derivatives, the spectral
measure construction, all-order asymptotic expansions, and analytic inversion
transferring divergence to D are not formalized here.

2026-11-16, auditfix1, IDs 67-68: the two occurrences of `htail`
are the same model-specific spectral-moment domination step. D-exact.tex,
eq:exact-mem-zero-radius-moments, supplies a written trace/moment argument;
it is not a definition of a host with unbounded support. No spectral measure
or link from host moments to the supplied sequence is constructed in Lean.
Both divergence conclusions and prop:exact-mem-zero-radius remain partial
at the physical-model level. The conditional sequence argument is checked.
-/
namespace Dobson.Coverage6ExactZeroRadius
open Filter
open scoped Topology

/-- A convenient elementary bound on the denominator of logarithm coefficients. -/
theorem linear_le_two_pow (n : ℕ) : (n:ℝ)+1 ≤ (2:ℝ)^(n+1) := by
  induction n with
  | zero => norm_num
  | succ n ih =>
      rw [show n+1+1=(n+1)+1 by omega, pow_succ]
      push_cast
      nlinarith [show (0:ℝ) ≤ n by positivity]

/-- The alternating logarithm series cannot converge at any nonzero real z.
`htail` is a moment lower bound, not an assumption about power-series convergence. -/
theorem logarithm_series_not_summable (μ : ℕ → ℝ)
    (htail : ∀ M : ℝ, 0 < M → ∃ p : ℝ, 0 < p ∧
      ∀ n : ℕ, p*M^(n+1) ≤ μ (n+1))
    (z : ℝ) (hz : z ≠ 0) :
    ¬ Summable (fun n : ℕ => (-1:ℝ)^n * μ (n+1) / ((n:ℝ)+1) * z^(n+1)) := by
  have hz0 : 0 < |z| := abs_pos.mpr hz
  obtain ⟨p,hp,hμ⟩ := htail (2/|z|) (by positivity)
  have hlower : ∀ n : ℕ,
      p ≤ |(-1:ℝ)^n * μ (n+1) / ((n:ℝ)+1) * z^(n+1)| := by
    intro n
    have hn : (0:ℝ) < (n:ℝ)+1 := by positivity
    have hm : 0 ≤ μ (n+1) := (by positivity : 0 ≤ p*(2/|z|)^(n+1)).trans (hμ n)
    rw [abs_mul, abs_div, abs_mul, abs_pow, abs_pow, abs_neg, abs_one,
      one_pow, one_mul, abs_of_nonneg hm, abs_of_pos hn, div_mul_eq_mul_div]
    apply (le_div_iff₀ hn).mpr
    have hmul := mul_le_mul_of_nonneg_right (hμ n) (pow_nonneg (abs_nonneg z) (n+1))
    have he : p*(2/|z|)^(n+1)*|z|^(n+1)=p*2^(n+1) := by
      rw [mul_assoc, ← mul_pow, div_mul_cancel₀ _ hz0.ne']
    rw [he] at hmul
    exact (mul_le_mul_of_nonneg_left (linear_le_two_pow n) hp.le).trans hmul
  intro hs
  have ht : Tendsto (fun n : ℕ =>
      |(-1:ℝ)^n * μ (n+1) / ((n:ℝ)+1) * z^(n+1)|) atTop (𝓝 0) := by
    simpa using hs.tendsto_atTop_zero.abs
  have hcontr : p ≤ 0 := ge_of_tendsto ht (Filter.Eventually.of_forall hlower)
  exact (not_le_of_gt hp) hcontr

/-- This explicit zero-radius statement quantifies over every nonzero argument. -/
theorem no_positive_convergence_radius (μ : ℕ → ℝ)
    (htail : ∀ M : ℝ, 0 < M → ∃ p : ℝ, 0 < p ∧
      ∀ n : ℕ, p*M^(n+1) ≤ μ (n+1)) :
    ¬ ∃ R : ℝ, 0 < R ∧ ∀ z : ℝ, |z| < R →
      Summable (fun n : ℕ => (-1:ℝ)^n * μ (n+1) / ((n:ℝ)+1) * z^(n+1)) := by
  rintro ⟨R,hR,hconv⟩
  apply logarithm_series_not_summable μ htail (R/2) (by positivity)
  apply hconv
  rw [abs_of_pos (by positivity : 0 < R/2)]
  linarith

end Dobson.Coverage6ExactZeroRadius
#print axioms Dobson.Coverage6ExactZeroRadius.linear_le_two_pow
#print axioms Dobson.Coverage6ExactZeroRadius.logarithm_series_not_summable
#print axioms Dobson.Coverage6ExactZeroRadius.no_positive_convergence_radius
