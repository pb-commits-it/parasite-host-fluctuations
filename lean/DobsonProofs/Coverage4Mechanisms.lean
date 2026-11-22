import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Topology.Instances.Matrix
import Mathlib.Topology.Algebra.InfiniteSum.Ring
import Mathlib.Topology.Algebra.InfiniteSum.NatInt
import Mathlib.Tactic

/-!
# Batch 4: curved encounters and linked transfers
Cores of prop:mech-curvature and prop:mech-linked. Encounter derivatives are actual
HasDerivAt statements. Missing model inputs: Gaussian mean-rate/growth expansions,
identification with the threshold coefficient, reference-chain reduction, and
summability derived from the stationary covariance law. The infinite-lag identity
lists Summable hypotheses explicitly. Real-power differentiation is not proved.
-/
namespace Dobson.Coverage4Mechanisms
open scoped BigOperators
open Filter

/-- Actual derivative of the dimensionless type-II encounter x/(1+x). -/
theorem typeII_derivative (x : ℝ) (hx : 1 + x ≠ 0) :
    HasDerivAt (fun y : ℝ => y / (1 + y)) (1 / (1 + x) ^ 2) x := by
  convert (hasDerivAt_id x).div ((hasDerivAt_const x (1 : ℝ)).add (hasDerivAt_id x)) hx using 1 <;> (try funext y) <;> dsimp <;> ring

/-- The derivative of the type-II first derivative. -/
theorem typeII_second_derivative (x : ℝ) (hx : 1 + x ≠ 0) :
    HasDerivAt (fun y : ℝ => 1 / (1 + y) ^ 2) (-2 / (1 + x) ^ 3) x := by
  convert (hasDerivAt_const x (1 : ℝ)).div
    (((hasDerivAt_const x (1 : ℝ)).add (hasDerivAt_id x)).pow 2) (pow_ne_zero 2 hx) using 1 <;> (try funext y) <;> dsimp <;>
    field_simp <;> ring

/-- Actual derivative of the dimensionless type-III encounter x²/(1+x²). -/
theorem typeIII_derivative (x : ℝ) :
    HasDerivAt (fun y : ℝ => y ^ 2 / (1 + y ^ 2))
      (2 * x / (1 + x ^ 2) ^ 2) x := by
  have hx : 1 + x ^ 2 ≠ 0 := by positivity
  convert ((hasDerivAt_id x).pow 2).div
    ((hasDerivAt_const x (1 : ℝ)).add ((hasDerivAt_id x).pow 2)) hx using 1 <;> (try funext y) <;> dsimp <;> ring

/-- The derivative of the type-III first derivative. -/
theorem typeIII_second_derivative (x : ℝ) :
    HasDerivAt (fun y : ℝ => 2 * y / (1 + y ^ 2) ^ 2)
      (2 * (1 - 3 * x ^ 2) / (1 + x ^ 2) ^ 3) x := by
  have hx : 1 + x ^ 2 ≠ 0 := by positivity
  convert ((hasDerivAt_id x).const_mul 2).div
    (((hasDerivAt_const x (1 : ℝ)).add ((hasDerivAt_id x).pow 2)).pow 2)
    (pow_ne_zero 2 hx) using 1 <;> (try funext y) <;> dsimp <;> field_simp <;> ring

/-- The dimensionless elasticities and curvatures, using the derivatives above. -/
theorem encounter_ratios (x : ℝ) (hx : 0 < x) :
    x * (1 / (1 + x) ^ 2) / (x / (1 + x)) = 1 / (1 + x) ∧
    x ^ 2 * (-2 / (1 + x) ^ 3) / (x / (1 + x)) = -2 * x / (1 + x) ^ 2 ∧
    x * (2 * x / (1 + x ^ 2) ^ 2) / (x ^ 2 / (1 + x ^ 2)) = 2 / (1 + x ^ 2) ∧
    x ^ 2 * (2 * (1 - 3 * x ^ 2) / (1 + x ^ 2) ^ 3) /
      (x ^ 2 / (1 + x ^ 2)) = 2 * (1 - 3 * x ^ 2) / (1 + x ^ 2) ^ 2 := by
  have hx0 : x ≠ 0 := ne_of_gt hx
  have h1 : 1 + x ≠ 0 := by positivity
  have h2 : 1 + x ^ 2 ≠ 0 := by positivity
  constructor
  · field_simp
  constructor
  · field_simp
  constructor <;> field_simp <;> ring

/-- Exact quadratic sign, assuming the model's coefficient expansion separately.
Nonzero amplitude is necessary: at sigma=0 strict positivity fails. -/
theorem curvature_sign (χ η K σ : ℝ) (hσ : σ ≠ 0) :
    0 < (χ / 2 - η ^ 2 * K) * σ ^ 2 ↔ 2 * η ^ 2 * K < χ := by
  rw [mul_pos_iff_of_pos_right (sq_pos_of_ne_zero hσ)]
  constructor <;> intro h <;> linarith

/-- Saturating type II cannot benefit when the gain-only coefficient is nonnegative. -/
theorem typeII_no_benefit (x K : ℝ) (hx : 0 < x) (hK : 0 ≤ K) :
    -2 * x / (1 + x) ^ 2 / 2 - (1 / (1 + x)) ^ 2 * K < 0 := by
  have hcurv : -2 * x / (1 + x) ^ 2 < 0 :=
    div_neg_of_neg_of_pos (by linarith) (by positivity)
  have hcost : 0 ≤ (1 / (1 + x)) ^ 2 * K := mul_nonneg (sq_nonneg _) hK
  linarith

/-- The type-III, power-response, and inverse-response benefit boundaries. -/
theorem typeIII_and_power_signs (x a K : ℝ) :
    (2 * (2 / (1 + x ^ 2)) ^ 2 * K < 2 * (1 - 3 * x ^ 2) / (1 + x ^ 2) ^ 2 ↔
      x ^ 2 < (1 - 4 * K) / 3) ∧
    (2 * a ^ 2 * K < a * (a - 1) ↔ 0 < a * ((1 - 2 * K) * a - 1)) ∧
    (2 * (-1 : ℝ) ^ 2 * K < (-1 : ℝ) * (-1 - 1) ↔ K < 1) := by
  have hx : 0 < (1 + x ^ 2) ^ 2 := by positivity
  have hd : 1 + x^2 ≠ 0 := by positivity
  constructor
  · have he : 2 * (2 / (1 + x ^ 2)) ^ 2 * K = 8 * K / (1 + x ^ 2) ^ 2 := by field_simp; ring
    rw [he, div_lt_div_iff_of_pos_right hx]
    constructor <;> intro h <;> linarith
  constructor
  · constructor <;> intro h <;> nlinarith
  · constructor <;> intro h <;> norm_num at * <;> linarith

/-- Expansion of the actual two-transfer quadratic contraction and both comparisons. -/
theorem two_host_decomposition (κ : Fin 2 → ℝ) (C : Fin 2 → Fin 2 → ℝ) :
    (∑ i, ∑ j, κ i * κ j * C i j) =
      κ 0 ^ 2 * C 0 0 + κ 1 ^ 2 * C 1 1 + κ 0 * κ 1 * (C 0 1 + C 1 0) ∧
    ((∑ i, ∑ j, κ i * κ j * C i j) < 0 ↔
      κ 0 * κ 1 * (C 0 1 + C 1 0) < -(κ 0 ^ 2 * C 0 0 + κ 1 ^ 2 * C 1 1)) ∧
    ((∑ i, ∑ j, κ i * κ j * C i j) < κ 0 ^ 2 * C 0 0 ↔
      κ 0 * κ 1 * (C 0 1 + C 1 0) < -(κ 1 ^ 2 * C 1 1)) := by
  simp only [Fin.sum_univ_two]
  constructor
  · ring
  constructor <;> constructor <;> intro h <;> nlinarith

/-- Infinite-lag host-swap identity. Absolute/summable contractions are explicit
analytic hypotheses; their covariance origin and renewal representation are not assumed proved.
For the manuscript identification, Nat index n denotes physical positive lag n+1. -/
theorem directed_swap (ρ21 ρ12 u12 u21 : ℕ → ℝ) (c : ℝ)
    (ha : Summable (fun l => ρ21 l * (c - u12 l) + ρ12 l * (c - u21 l)))
    (hb : Summable (fun l => ρ12 l * (c - u12 l) + ρ21 l * (c - u21 l))) :
    (∑' l, (ρ21 l * (c - u12 l) + ρ12 l * (c - u21 l))) -
      (∑' l, (ρ12 l * (c - u12 l) + ρ21 l * (c - u21 l))) =
      -(∑' l, (ρ21 l - ρ12 l) * (u12 l - u21 l)) := by
  rw [← ha.tsum_sub hb, ← tsum_neg]
  apply tsum_congr
  intro l
  ring


end Dobson.Coverage4Mechanisms
