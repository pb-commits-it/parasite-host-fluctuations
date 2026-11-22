import DobsonProofs.Coverage2Exact
import DobsonProofs.GammaDigammaSeries
import Mathlib.Analysis.Asymptotics.AsymptoticEquivalent
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv
import Mathlib.Tactic

/-! # Batch 5: exact-gamma endpoint coefficients (P6, P7, P8)

Coverage correction, 2026-11-22: P6 is full in the version-2 inventory. The
actual Kummer identities and both endpoint limits are proved, including
Gamma/digamma/trigamma identification and asymptotic equivalence.
P7 and P8 retain core coverage: leading coefficient calculus is checked;
uniform differentiated expansions and transfer to the actual root/minimum
remain outside credit. -/
/-! ## Audit reconciliation — 2026-11-22

The ten P6 premises listed in the 2026-11-16 audit have been discharged for
the actual Kummer integrals. `kummer_mean_identity` and
`kummer_shape_derivative` use proved joint-law, normalization, ODE and mixed-
derivative identities. `kummer_endpoint_limits` uses proved shape, stay and
logarithmic-moment limits and assumes only `p > 0`. The coefficient calculations
for P7/P8 are separate cores; they do not prove their uniform remainders. -/

/-! Historical step, 2026-11-16: `Coverage2Exact.kummerLog1p_deriv`
discharged the former `hode` premise. Coverage correction, 2026-11-22: the
joint-law, normalization, mixed-derivative and endpoint premises have also
been discharged; none remains in the public P6 theorems. -/

/-! Historical shape-normalization step, 2026-11-16:
`Coverage2Exact.kummerLogZ_shape_deriv` proves the normalization identity
by dominated differentiation. As of 2026-11-22, the joint Gamma/Kummer
identity is also proved and is not a premise of `kummer_mean_identity`. -/

/-! Historical mixed-derivative step, 2026-11-16: both derivative
orders are identified by dominated Kummer integrals in `kummer_shape_mixed`.
As of 2026-11-22, the differentiated joint-law identity is proved as well;
`kummer_shape_derivative` takes only positive Kummer parameters. -/

/-!
## leanp1 joint gamma-law correction — 2026-11-16

The symmetric bivariate gamma density and its logarithmic moments are now
integrable in Lean. Fubini, the Gamma integral and its actual derivative give
`kummer_log_joint_identity`, removing part1:42. `kummer_mean_identity` uses
real digamma `Gamma'(p)/Gamma(p)` and actual Kummer moments throughout.
-/

/-! Historical differentiated-mean step, 2026-11-16:
`kummer_differentiated_mean_identity` follows by differentiating the proved
joint-law identity. Coverage correction, 2026-11-22: the endpoint limits
are also proved, as assembled in `kummer_endpoint_limits`. -/

open Filter Set
open scoped Topology

/-! Historical small-κ stay step, 2026-11-16: the joint Gamma
identity bounds the stay probability by a constant times `√κ`. As of
2026-11-22 the rescaled stay and shape/moment limits are also proved.
The endpoint theorem derives its eventual denominator conditions near zero. -/

/-!
## leanp1 actual trigamma endpoint correction — 2026-11-16

All five small-κ endpoint limits are proved for the actual Kummer law.
The score numerator tends to the actual positive series Σn1/(p+n)² by
change of variables, a geometric series and dominated convergence; the
scaled first numerator tends to1 and Z/log(1/κ) tends to1. Their quotient
gives the precise shape coefficient, removing part1:49. The public endpoint
theorem uses the actual V and stay probability and has only the domain p>0.
The separate special-function identification of that series with ψ′(p)
is not asserted here.
-/

namespace Dobson.Coverage5Endpoints

open Coverage2Exact

/-- P6: both the joint gamma-law identity and the shape derivative of the
actual partition integral are proved. `gammaLogDerivative` is real digamma. -/
theorem kummer_mean_identity (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    kummerLog1p p κ = (gammaLogDerivative p - Real.log κ -
      deriv (fun q => Real.log (NoiseAnalyticity.kummerZ q κ)) p) / 2 := by
  have hjoint := kummer_log_joint_identity p κ hp hκ
  have hnorm := (kummerLogZ_shape_deriv p κ hp hκ).deriv
  linarith

/-- P6: the actual shape derivative identity follows from the proved joint
gamma-law identity and both actual mixed derivatives of the Kummer integral. -/
theorem kummer_shape_derivative (κ p : ℝ) (hκ : 0 < κ) (hp : 0 < p)
    :
    deriv (fun q => κ * kummerMean q κ) p =
      1 - 2 * (κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ)) := by
  have hmean := kummer_differentiated_mean_identity p κ hp hκ
  have hmixed := kummer_shape_mixed p κ hp hκ
  have hode : κ * deriv (kummerLog1p p) κ =
      -(κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ)) := by
    rw [(kummerLog1p_deriv p κ hp hκ).deriv]
    ring
  have h := congrArg (fun x : ℝ => κ * x) hmean
  field_simp [hκ.ne'] at h
  nlinarith

/-- Algebraic assembly uses only eventual denominator conditions, as required
for endpoint limits; no assertion is made at the singular coordinate κ=1. -/
private theorem endpoint_limit_assembly {α : Type*} (l : Filter α)
    (a s m t : α → ℝ) (v : ℝ)
    (ha : Filter.Tendsto a l (nhds 0))
    (hs : Filter.Tendsto s l (nhds 0))
    (ha2 : Filter.Tendsto (fun x => a x / (t x)^2) l (nhds v))
    (hs2 : Filter.Tendsto (fun x => s x / (t x)^2) l (nhds 0))
    (hm : Filter.Tendsto (fun x => m x * t x) l (nhds (1 / 2)))
    (ht : ∀ᶠ x in l, t x ≠ 0) :
    let g := fun x => 1 / ((1 - a x) / 2) - 2 - s x / (1 - s x)
    Filter.Tendsto (fun x => g x / (t x)^2) l (nhds (2 * v)) ∧
    Filter.Tendsto (fun x => g x * (m x)^2) l (nhds (v / 2)) := by
  dsimp only
  have ha1 : Filter.Tendsto (fun x => 1 - a x) l (nhds (1 - 0 : ℝ)) :=
    tendsto_const_nhds.sub ha
  have hs1 : Filter.Tendsto (fun x => 1 - s x) l (nhds (1 - 0 : ℝ)) :=
    tendsto_const_nhds.sub hs
  have hfirst : Filter.Tendsto
      (fun x => (1 / ((1 - a x) / 2) - 2 - s x / (1 - s x)) / (t x)^2)
      l (nhds (2 * v)) := by
    have h := ((ha2.const_mul 2).div ha1 (by norm_num)).sub
      (hs2.div hs1 (by norm_num))
    have hh : Filter.Tendsto (fun x => 2 * (a x / t x ^ 2) / (1 - a x) -
        (s x / t x ^ 2) / (1 - s x)) l (nhds (2 * v)) := by simpa using h
    apply hh.congr'
    filter_upwards [ha1.eventually_ne (by norm_num : (1 - 0 : ℝ) ≠ 0),
      hs1.eventually_ne (by norm_num : (1 - 0 : ℝ) ≠ 0), ht] with x ha' hs' ht'
    field_simp
    <;> ring
  refine ⟨hfirst, ?_⟩
  have h := hfirst.mul (hm.pow 2)
  have hh : Filter.Tendsto
      (fun x => ((1 / ((1 - a x) / 2) - 2 - s x / (1 - s x)) / t x ^ 2) *
        (m x * t x) ^ 2) l (nhds (v / 2)) := by convert h using 1 <;> ring
  apply hh.congr'
  filter_upwards [ht] with x ht'
  field_simp

/-- The manuscript's inverse logarithmic endpoint coordinate. -/
noncomputable def kummerEndpointTime (κ : ℝ) : ℝ := 1 / Real.log (1 / κ)

/-- The manuscript's actual shape derivative `∂ₚ(κ E T)`. -/
noncomputable def kummerShapeA (p κ : ℝ) : ℝ :=
  deriv (fun q => κ * kummerMean q κ) p

private theorem endpointTime_eventually_ne :
    ∀ᶠ κ in nhdsWithin (0 : ℝ) (Set.Ioi 0), kummerEndpointTime κ ≠ 0 := by
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
  have hlog : 0 < Real.log (1 / κ) := Real.log_pos (by
    exact (lt_div_iff₀ hκ).mpr (by simpa using hκ1))
  exact div_ne_zero one_ne_zero hlog.ne'

/-- P6: both endpoint limits for the actual Kummer growth derivative.
Shape, logarithmic-moment and stay asymptotics are proved for the integrals;
the only hypothesis is positive shape `p`. -/
/- 2026-11-16 leanp1: the proved square-root bound absorbs two logarithms,
removing the rescaled stay premise part1:50. -/
/- 2026-11-16 leanp1: positivity of the actual shape covariance bounds m
between two constant offsets of −log κ/2, removing part1:51 by squeezing. -/
/- 2026-11-16 leanp1: the actual shape derivative is between zero
and a constant times 1/Z; the logarithmic lower bound Z→∞ removes part1:47. -/
/- P6: both limits for the actual Kummer growth derivative, with the manuscript's
positive-series coefficient. All five former endpoint premises are discharged. -/
theorem kummer_endpoint_limits (p : ℝ) (hp : 0 < p) :
    let g := fun κ => 1 / (κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ)) - 2 -
      Real.exp (-kummerLog1p p κ) / (1 - Real.exp (-kummerLog1p p κ))
    Filter.Tendsto (fun κ => g κ / (kummerEndpointTime κ)^2)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds (2 * gammaTrigamma p)) ∧
    Filter.Tendsto (fun κ => g κ * (kummerLog1p p κ)^2)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds (gammaTrigamma p / 2)) := by
  have hs2 : Filter.Tendsto (fun κ => Real.exp (-kummerLog1p p κ) / (kummerEndpointTime κ)^2)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
    simpa only [kummerEndpointTime] using kummerStay_log_sq_tendsto_zero p hp
  have hm : Filter.Tendsto (fun κ => kummerLog1p p κ * kummerEndpointTime κ)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds (1 / 2)) := by
    simpa only [kummerEndpointTime] using kummerLog1p_log_scale_tendsto p hp
  have ha : Filter.Tendsto (kummerShapeA p) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
    change Filter.Tendsto (fun κ => deriv (fun q => κ * kummerMean q κ) p) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0)
    exact kummerShapeA_tendsto_zero p hp
  have ha2 : Filter.Tendsto (fun κ => kummerShapeA p κ / (kummerEndpointTime κ)^2)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds (gammaTrigamma p)) := by
    simpa only [kummerShapeA, kummerEndpointTime] using kummerShapeA_log_sq_tendsto p hp
  have h := endpoint_limit_assembly (nhdsWithin 0 (Set.Ioi 0)) (kummerShapeA p)
    (fun κ => Real.exp (-kummerLog1p p κ)) (kummerLog1p p) kummerEndpointTime (gammaTrigamma p)
    ha (kummerStay_tendsto_zero p hp) ha2 hs2 hm endpointTime_eventually_ne
  have heq : ∀ κ > 0, (1 - kummerShapeA p κ) / 2 =
      κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ) := by
    intro κ hκ
    dsimp [kummerShapeA]
    rw [kummer_shape_derivative κ p hκ hp]
    ring
  dsimp only at h ⊢
  constructor
  · apply h.1.congr'
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    rw [heq κ hκ]
  · apply h.2.congr'
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    rw [heq κ hκ]

/-- P7: actual first-order Laplace coefficient and division by the deterministic
birth elasticity, evaluated at the threshold c=1-s. -/
theorem small_noise_coefficients (s : ℝ) (hs : s ≠ 0) (hs1 : s ≠ 1)
    (hs2 : s ≠ 2) :
    s * (1 - s) / (2 * (s * (2 - s))) -
        (1 - s) * (s ^ 2 * (3 - 2 * s)) / (2 * (s * (2 - s)) ^ 2) =
      -(1 - s) ^ 2 / (2 * (2 - s) ^ 2) ∧
    ((1 - s) ^ 2 / (2 * (2 - s) ^ 2)) / ((1 - s) / (2 - s)) =
      (1 - s) / (2 * (2 - s)) := by
  have h1 : 1 - s ≠ 0 := by intro h; apply hs1; linarith
  have h2 : 2 - s ≠ 0 := by intro h; apply hs2; linarith
  constructor <;> field_simp <;> ring

/-- P7: differentiate the leading D coefficient, obtaining the stated g
coefficient. Its extension to s=0 vanishes exactly there on [0,1). -/
theorem small_noise_derivative (s : ℝ) (hs : 0 ≤ s) (hs1 : s < 1) :
    HasDerivAt (fun x : ℝ => (1 - x) / (2 * (2 - x)))
      (-1 / (2 * (2 - s) ^ 2)) s ∧
    (0 < s / (2 * (2 - s) ^ 2) ↔ 0 < s) ∧
    (s / (2 * (2 - s) ^ 2) = 0 ↔ s = 0) := by
  have h2 : 0 < 2 - s := by linarith
  have hd : 2 * (2 - s) ≠ 0 := by positivity
  have hsq : 0 < 2 * (2 - s) ^ 2 := by positivity
  refine ⟨?_, ?_, ?_⟩
  · convert ((hasDerivAt_const s 1).sub (hasDerivAt_id s)).div
      (((hasDerivAt_const s 2).sub (hasDerivAt_id s)).const_mul 2) hd using 1 <;>
      (try funext x) <;> dsimp <;> field_simp <;> ring
  · exact div_pos_iff_of_pos_right hsq
  · exact div_eq_zero_iff.trans (or_iff_left (ne_of_gt hsq))

/-- P8: a square identity proves the unique global minimum of the actual
leading coefficient, not merely a stationary point of an abstract function. -/
theorem fixed_kappa_unique_minimum (κ : ℝ) (hκ : 0 < κ) :
    (4 * κ + 1) / (32 * Real.sqrt κ) - 1 / 8 =
      (2 * Real.sqrt κ - 1) ^ 2 / (32 * Real.sqrt κ) ∧
    1 / 8 ≤ (4 * κ + 1) / (32 * Real.sqrt κ) ∧
    ((4 * κ + 1) / (32 * Real.sqrt κ) = 1 / 8 ↔ κ = 1 / 4) := by
  have hr : 0 < Real.sqrt κ := Real.sqrt_pos.2 hκ
  have hr2 := Real.sq_sqrt hκ.le
  have hid : (4 * κ + 1) / (32 * Real.sqrt κ) - 1 / 8 =
      (2 * Real.sqrt κ - 1) ^ 2 / (32 * Real.sqrt κ) := by
    field_simp
    nlinarith
  refine ⟨hid, ?_, ?_⟩
  · have h : 0 ≤ (2 * Real.sqrt κ - 1) ^ 2 / (32 * Real.sqrt κ) :=
      div_nonneg (sq_nonneg _) (by positivity)
    linarith
  · constructor
    · intro h
      have hz : (2 * Real.sqrt κ - 1) ^ 2 = 0 := by
        have : (2 * Real.sqrt κ - 1) ^ 2 / (32 * Real.sqrt κ) = 0 := by linarith
        exact (div_eq_zero_iff.mp this).resolve_right (by positivity)
      have hrval : 2 * Real.sqrt κ - 1 = 0 := sq_eq_zero_iff.mp hz
      nlinarith
    · intro h
      have hrval : Real.sqrt κ = 1 / 2 := by nlinarith
      rw [hrval, h]
      norm_num

/-- P8: actual derivative of the fixed-kappa leading coefficient. -/
theorem fixed_kappa_coefficient_derivative (κ : ℝ) (hκ : 0 < κ) :
    HasDerivAt (fun x : ℝ => (4 * x + 1) / (32 * Real.sqrt x))
      ((4 * κ - 1) / (64 * (Real.sqrt κ) ^ 3)) κ := by
  have hr : 0 < Real.sqrt κ := Real.sqrt_pos.2 hκ
  have hr2 := Real.sq_sqrt hκ.le
  have h := (((hasDerivAt_id κ).const_mul 4).add_const 1).div
    ((Real.hasDerivAt_sqrt (ne_of_gt hκ)).const_mul 32) (by positivity)
  convert h using 1
  · funext x
    rfl
  · dsimp
    field_simp
    nlinarith

/-- P8: second derivative and strict convexity of the leading coefficient on
[1/8,1/2] (indeed throughout 0<kappa<3/4). -/
theorem fixed_kappa_coefficient_second_derivative (κ : ℝ) (hκ : 0 < κ) :
    HasDerivAt (fun x : ℝ => (4 * x - 1) / (64 * (Real.sqrt x) ^ 3))
      ((3 - 4 * κ) / (128 * (Real.sqrt κ) ^ 5)) κ ∧
    (κ < 3 / 4 → 0 < (3 - 4 * κ) / (128 * (Real.sqrt κ) ^ 5)) := by
  have hr : 0 < Real.sqrt κ := Real.sqrt_pos.2 hκ
  have hr2 := Real.sq_sqrt hκ.le
  refine ⟨?_, ?_⟩
  · have h := (((hasDerivAt_id κ).const_mul 4).sub_const 1).div
      (((Real.hasDerivAt_sqrt (ne_of_gt hκ)).pow 3).const_mul 64) (by dsimp; positivity)
    convert h using 1
    · funext x
      rfl
    · dsimp
      field_simp
      nlinarith
  · intro h
    apply div_pos
    · linarith
    · positivity

#print axioms kummer_endpoint_limits
#print axioms kummer_shape_derivative

/-- Both actual small-kappa endpoint limits with the manuscript coefficient written as ψ′(p). -/
theorem kummer_endpoint_limits_digamma (p : ℝ) (hp : 0 < p) :
    let g := fun κ => 1 / (κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ)) - 2 -
      Real.exp (-kummerLog1p p κ) / (1 - Real.exp (-kummerLog1p p κ))
    Filter.Tendsto (fun κ => g κ / (kummerEndpointTime κ)^2)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds (2 * deriv gammaLogDerivative p)) ∧
    Filter.Tendsto (fun κ => g κ * (kummerLog1p p κ)^2)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds (deriv gammaLogDerivative p / 2)) := by
  simpa only [GammaDigammaSeries.gammaTrigamma_eq_deriv p hp] using kummer_endpoint_limits p hp

/-- The first endpoint assertion in its literal asymptotic-equivalence form. -/
theorem kummer_endpoint_asymptotic_digamma (p : ℝ) (hp : 0 < p) :
    let g := fun κ => 1 / (κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ)) - 2 -
      Real.exp (-kummerLog1p p κ) / (1 - Real.exp (-kummerLog1p p κ))
    Asymptotics.IsEquivalent (nhdsWithin 0 (Set.Ioi 0)) g
      (fun κ => 2 * deriv gammaLogDerivative p / (Real.log (1 / κ))^2) := by
  dsimp only
  apply Asymptotics.isEquivalent_of_tendsto_one
  have hv : 0 < deriv gammaLogDerivative p := by
    rw [← GammaDigammaSeries.gammaTrigamma_eq_deriv p hp]
    exact GammaDigammaSeries.gammaTrigamma_pos p hp
  have h := (kummer_endpoint_limits_digamma p hp).1.div_const (2 * deriv gammaLogDerivative p)
  rw [div_self (ne_of_gt (by positivity : 0 < 2 * deriv gammaLogDerivative p))] at h
  apply h.congr'
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
  have hL : 0 < Real.log (1 / κ) :=
    Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
  dsimp only [Pi.div_apply, kummerEndpointTime]
  field_simp [hL.ne', hv.ne']


end Dobson.Coverage5Endpoints

#print axioms Dobson.Coverage5Endpoints.kummer_endpoint_limits_digamma
