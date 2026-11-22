import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Tactic

/-! # Batch 5, item 33 (`prop:exact-R2`): derivative and series components

`U` is supplied as a real function, with value and derivative identifications
for the intended Tricomi section stated explicitly. The conclusions derive
its logarithmic derivative and a pole-cancelled ratio from summation data.
Integrated holomorphy, differentiation under the integral, connection-series
identification and the special-function formulas remain outside credit. -/
/-! ## Coverage correction — 2026-11-22

`prop:exact-R2` remains partial-P1. The two convergent connection-series
integral evaluations `hmass` and `hmoment` remain supplied, as recorded in
the independent hypothesis audit (2026-11-16). Pointwise integrand calculus
and ratio algebra do not establish integrated holomorphy, differentiation
under the integral or the physical growth identification. -/

namespace Dobson.Coverage5GammaFunctions

/-- A normalized integral derivative produces the actual logarithmic derivative.
`G = Γ(p)`, `Z` is the unnormalized mass, and `N` its log-weighted moment. -/
theorem logarithmic_parameter_formula (U : ℝ → ℝ) (G Z N s : ℝ)
    (hG : G ≠ 0) (hZ : Z ≠ 0)
    (hvalue : U 1 = Z / G) (hderiv : HasDerivAt U (N / G) 1) :
    Real.log s + N / Z = Real.log s + deriv (fun b => Real.log (U b)) 1 := by
  have hU : U 1 ≠ 0 := by rw [hvalue]; exact div_ne_zero hZ hG
  rw [(hderiv.log hU).deriv, hvalue]
  congr 1
  field_simp

/-- The constant and linear coefficients after the two simple poles cancel.
`P = ψ(p)`, `Pn = ψ(p+n)`, `Hn = ψ(n+1)+γ`, and `v = ψ₁(p+n)`.
The opposite residues have already been isolated in the connection formula. -/
theorem pole_cancellation_coefficients (P Pn Hn γ H v : ℝ) :
    (H - γ - Pn + P + Hn) - (P + γ - Hn) =
      H + 2 * (Hn - γ) - Pn ∧
    ((H - γ - Pn + P + Hn) ^ 2 - (P + γ - Hn) ^ 2 + v) / 2 =
      ((H + 2 * (Hn - γ) - Pn) * (H - Pn + 2 * P) + v) / 2 := by
  constructor <;> ring

/-- Once convergence and the two connection-series evaluations are supplied,
Lean checks the exact pole-cancelled ratio. Its summands use the manuscript's
`dₙ = H + 2ψ(n+1) - ψ(p+n)`; convergence is not inferred from formal cancellation. -/
theorem pole_cancelled_series_ratio (C ψn Hn v : ℕ → ℝ) (P γ H Z N : ℝ)
    (hmass : HasSum (fun n => C n *
      ((H - γ - ψn n + P + Hn n) - (P + γ - Hn n))) Z)
    (hmoment : HasSum (fun n => C n *
      (((H - γ - ψn n + P + Hn n) ^ 2 - (P + γ - Hn n) ^ 2 + v n) / 2)) N) :
    N / Z = (∑' n, C n *
        ((H + 2 * (Hn n - γ) - ψn n) * (H - ψn n + 2 * P) + v n)) /
      (2 * ∑' n, C n * (H + 2 * (Hn n - γ) - ψn n)) := by
  have hm : HasSum (fun n => C n * (H + 2 * (Hn n - γ) - ψn n)) Z := by
    convert hmass using 1
    funext n
    rw [(pole_cancellation_coefficients P (ψn n) (Hn n) γ H (v n)).1]
  have hn : HasSum (fun n => C n *
      ((H + 2 * (Hn n - γ) - ψn n) * (H - ψn n + 2 * P) + v n)) (2 * N) := by
    convert hmoment.mul_left 2 using 1
    funext n
    rw [(pole_cancellation_coefficients P (ψn n) (Hn n) γ H (v n)).2]
    ring
  rw [hm.tsum_eq, hn.tsum_eq]
  ring

/-- The actual real-p, real-positive-integration-coordinate Kummer integrand,
complexified in b using real logarithms of positive t and 1+t. The derivative
below is algebraically valid even outside those integration-domain restrictions. -/
noncomputable def kummerKernel (p κ t : ℝ) (b : ℂ) : ℂ :=
  (Real.exp (-κ * t) * Real.rpow t (p - 1) : ℝ) *
    Complex.exp ((b - (p : ℂ) - 1) * (Real.log (1 + t) : ℂ))

/-- Exact complex b-derivative of the identified Tricomi integrand. The factor
`log(1+t)` is obtained by the complex exponential chain rule, not assumed as an
under-integral differentiation identity. -/
theorem kummer_kernel_derivative (p κ t : ℝ) (b : ℂ) :
    HasDerivAt (kummerKernel p κ t)
      ((Real.log (1 + t) : ℂ) * kummerKernel p κ t b) b := by
  have h := (((hasDerivAt_id b).sub_const (p : ℂ)).sub_const 1).mul_const
    (Real.log (1 + t) : ℂ)
  convert h.cexp.const_mul ((Real.exp (-κ * t) * Real.rpow t (p - 1) : ℝ) : ℂ) using 1
  · rfl
  · dsimp [kummerKernel]
    ring

/-- The integrand is entire in b. This does not exchange a complex derivative
with its improper integral; locally uniform integrable domination is still needed. -/
theorem kummer_kernel_entire (p κ t : ℝ) (b : ℂ) :
    AnalyticAt ℂ (kummerKernel p κ t) b := by
  unfold kummerKernel
  fun_prop

end Dobson.Coverage5GammaFunctions
#print axioms Dobson.Coverage5GammaFunctions.logarithmic_parameter_formula

#print axioms Dobson.Coverage5GammaFunctions.pole_cancellation_coefficients

#print axioms Dobson.Coverage5GammaFunctions.pole_cancelled_series_ratio

#print axioms Dobson.Coverage5GammaFunctions.kummer_kernel_derivative

#print axioms Dobson.Coverage5GammaFunctions.kummer_kernel_entire
