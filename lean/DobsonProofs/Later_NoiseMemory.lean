import DobsonProofs.NoiseExceptional
import DobsonProofs.NoiseMemory
import Mathlib.Analysis.Calculus.IteratedDeriv.Defs

/-! The genuine analytic-to-geometric-coefficient implication. Coefficients
are actual Taylor coefficients, not an arbitrary sequence attached to an
uninterpreted proposition called analytic. Physical asymptotic identification
and almost-everywhere propagation remain separate obligations. -/
namespace Dobson.LaterNoiseMemory
open Filter Set
open scoped Topology

noncomputable def taylorCoefficient (f : ℝ → ℝ) (n : ℕ) : ℝ :=
  iteratedDeriv n f 0 / (n.factorial:ℝ)

/-- Real analyticity forces a geometric bound on the actual Taylor
coefficients. This proves the analytic coefficient estimate from definitions. -/
theorem analytic_taylor_geometric_bound (f : ℝ → ℝ) (hf : AnalyticAt ℝ f 0) :
    ∃ C A : ℝ, 0 < C ∧ 0 < A ∧ ∀ n, |taylorCoefficient f n| ≤ C*A^n := by
  have hs := hasFPowerSeriesAt_iff.mp hf.hasFPowerSeriesAt
  obtain ⟨r,hr,hball⟩ := Metric.eventually_nhds_iff.mp hs
  let z := r/2
  have hz : 0 < z := by dsimp [z]; positivity
  have hzball : dist z (0:ℝ) < r := by
    rw [Real.dist_eq, sub_zero, abs_of_pos hz]
    dsimp [z]; linarith
  have hsum := (hball hzball).summable
  simp only [FormalMultilinearSeries.coeff_ofScalars, smul_eq_mul, zero_add] at hsum
  obtain ⟨C,hC⟩ := hsum.tendsto_atTop_zero.cauchySeq.isBounded_range.exists_norm_le
  refine ⟨|C|+1,1/z,by positivity,by positivity,?_⟩
  intro n
  have hh := hC _ (Set.mem_range_self n)
  simp only [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_pos hz] at hh
  change z^n*|taylorCoefficient f n| ≤ C at hh
  rw [one_div, inv_pow, ← div_eq_mul_inv]
  apply (le_div_iff₀ (pow_pos hz n)).mpr
  have hc := le_abs_self C
  nlinarith

/-- A factorial lower subsequence of the actual Taylor coefficients excludes
real analyticity, with no analytic-coefficient premise. -/
theorem factorial_taylor_not_analytic (f : ℝ → ℝ) (A : ℝ) (hA : 0 < A)
    (hfactorial : NoiseExceptional.FactorialLower (taylorCoefficient f) A) :
    ¬AnalyticAt ℝ f 0 := by
  intro hf
  obtain ⟨C,R,hC,hR,hb⟩ := analytic_taylor_geometric_bound f hf
  obtain ⟨n,hn⟩ := NoiseExceptional.factorial_obstruction (taylorCoefficient f) A C R hA hC hR hfactorial
  exact (not_lt_of_ge (hb n)) hn

/-- An almost-everywhere factorial obstruction gives almost-everywhere
nonanalyticity for the genuine Taylor coefficients. This does not assert
the missing propagation theorem for the gamma-refresh model. -/
theorem ae_factorial_taylor_not_analytic (μ : MeasureTheory.Measure ℝ)
    (D : ℝ → ℝ → ℝ)
    (hfactorial : ∀ᵐ s ∂μ, ∃ A : ℝ, 0 < A ∧
      NoiseExceptional.FactorialLower (taylorCoefficient (D s)) A) :
    ∀ᵐ s ∂μ, ¬AnalyticAt ℝ (D s) 0 := by
  filter_upwards [hfactorial] with s hs
  obtain ⟨A,hA,hf⟩ := hs
  exact factorial_taylor_not_analytic (D s) A hA hf

end Dobson.LaterNoiseMemory

#print axioms Dobson.LaterNoiseMemory.analytic_taylor_geometric_bound
#print axioms Dobson.LaterNoiseMemory.factorial_taylor_not_analytic
#print axioms Dobson.LaterNoiseMemory.ae_factorial_taylor_not_analytic
