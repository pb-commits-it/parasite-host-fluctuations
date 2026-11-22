import DobsonProofs.NoiseAnalyticity
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.PSeries
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.SpecialFunctions.Gamma.Deriv
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic

/-! # Coverage batch 2: Kummer differential identities and the exact root

Coverage correction, 2026-11-22: `prop:exact-P1` is full. Normalized moment
derivatives, ordinary and log-weighted integration by parts, and strict
covariance positivity are proved for the actual Kummer integrals.
`prop:exact-R3` is core: the actual integral-root existence and bounds are
proved; identification with the population Lyapunov threshold is outside credit. -/
/-! ## Audit reconciliation — 2026-11-22

The former `kummer_odes` premises `hm`, `hμ`, `hν`, `hwibp` and `hcov`
are proved lemmas about the actual normalized integrals. The former
`exact_root` continuity, monotonicity and endpoint premises are likewise
proved for `exactGrowth`. These historical entries are not current debts;
the integral-root theorem still does not identify a population exponent. -/

/-! Historical derivative step, 2026-11-16: the logarithmic
moments are the actual `kummerLog1p`, `kummerCrossLog1p` and
`kummerSecondLog1p`; dominated differentiation proves their derivatives.
Coverage correction, 2026-11-22: weighted integration by parts and strict
covariance positivity are also proved, so `kummer_odes` has only the
positive parameter-domain hypotheses. -/

/-! Actual-integral identities, 2026-11-16:
`kummerLog1p_cov_pos` proves strict covariance positivity and
`kummerLog1p_weighted_ibp` proves log-weighted integration by parts.
They discharge the former premises of `kummer_odes`; its four conclusions
use only the positive Kummer parameter domain. -/

/-! Actual-growth correction, 2026-11-22: `exact_root` uses
`exactGrowth`, the Kummer integral formula at `c = (1-s) exp D` on
`p > 0`, `0 < s < 1`. Continuity, strict monotonicity and both endpoint
signs are proved; population-growth identification is outside this core. -/

/-! Gamma log-loss endpoint, 2026-11-22: strict softplus Jensen
and the symmetric Gamma-law identity prove positive growth at
`gammaLogLoss p = log p - Gamma'(p)/Gamma(p)`. The lower endpoint sign is
also proved, so neither sign is a premise of `exact_root`. -/

/-!
## leanp1 negative lower-root endpoint correction — 2026-11-16

Under the symmetric bivariate gamma law, conditional strict logarithmic Jensen
and symmetry prove negative actual growth at `D=0`, removing part2:32.
The gamma conditional first moment and every integrability bound are proved
from the defining density. Thus `exact_root` now uses only the manuscript's
positive-shape and interior-stay domain assumptions, with both strict bounds.
-/

namespace Dobson.Coverage2Exact

open Set MeasureTheory Filter
open scoped Topology
open NoiseAnalyticity

/-- First moment of the normalized Kummer density, not an independent scalar. -/
noncomputable def kummerMean (p κ : ℝ) : ℝ :=
  NoiseAnalyticity.kummerExpect p κ (fun t => t)

/-- Second moment of the same normalized density. -/
noncomputable def kummerSecondMoment (p κ : ℝ) : ℝ :=
  NoiseAnalyticity.kummerExpect p κ (fun t => t ^ 2)

theorem weight_cont (q x : ℝ) :
    ContinuousOn (kummerWeight q x) (Ioi 0) := by
  unfold kummerWeight
  have h1 : ContinuousOn (fun t : ℝ => t ^ (q - 1)) (Ioi 0) :=
    continuousOn_id.rpow_const fun t ht => Or.inl (mem_Ioi.mp ht).ne'
  have h2 : ContinuousOn (fun t : ℝ => (1 + t) ^ (-q)) (Ioi 0) :=
    (continuousOn_const.add continuousOn_id).rpow_const fun t ht =>
      Or.inl (by change 1 + t ≠ 0; linarith [mem_Ioi.mp ht])
  exact (h1.mul h2).mul (continuous_const.mul continuous_id).rexp.continuousOn

theorem weight_nonneg (q x t : ℝ) (ht : 0 < t) :
    0 ≤ kummerWeight q x t := by
  unfold kummerWeight
  positivity

private theorem weight_antitone (q x y t : ℝ) (ht : 0 < t) (hxy : x ≤ y) :
    kummerWeight q y t ≤ kummerWeight q x t := by
  unfold kummerWeight
  apply mul_le_mul_of_nonneg_left
  · exact Real.exp_le_exp.mpr (by nlinarith)
  · positivity


private theorem rpow_weight_integrable {q x r : ℝ} (hq : 0 < q) (hx : 0 < x)
    (hqr : 0 < q + r) :
    IntegrableOn (fun t : ℝ => t ^ r * kummerWeight q x t) (Ioi 0) := by
  have hdom : IntegrableOn (fun t : ℝ => t ^ (q + r - 1) * Real.exp (-x * t)) (Ioi 0) := by
    simpa [Real.rpow_one] using
      (integrableOn_rpow_mul_exp_neg_mul_rpow (p := 1) (s := q + r - 1) (b := x)
        (by linarith) one_pos hx)
  have hcont : ContinuousOn (fun t : ℝ => t ^ r * kummerWeight q x t) (Ioi 0) :=
    (continuousOn_id.rpow_const fun t ht => Or.inl (mem_Ioi.mp ht).ne').mul
      (weight_cont q x)
  refine hdom.mono' (hcont.aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
  have ht0 : 0 < t := ht
  have h1t : 1 ≤ 1 + t := by linarith
  have hpow_le : (1 + t) ^ (-q) ≤ 1 := by
    have hle := Real.rpow_le_rpow_of_exponent_le h1t (by linarith : -q ≤ 0)
    rwa [Real.rpow_zero] at hle
  unfold kummerWeight
  have hnn : 0 ≤ t ^ r * (t ^ (q - 1) * (1 + t) ^ (-q) * Real.exp (-x * t)) := by positivity
  rw [Real.norm_of_nonneg hnn]
  have hcomb : t ^ r * t ^ (q - 1) = t ^ (q + r - 1) := by
    rw [← Real.rpow_add ht0]
    congr 1
    ring
  calc t ^ r * (t ^ (q - 1) * (1 + t) ^ (-q) * Real.exp (-x * t))
      = (t ^ r * t ^ (q - 1)) * (1 + t) ^ (-q) * Real.exp (-x * t) := by ring
    _ = t ^ (q + r - 1) * (1 + t) ^ (-q) * Real.exp (-x * t) := by rw [hcomb]
    _ ≤ t ^ (q + r - 1) * 1 * Real.exp (-x * t) := by gcongr
    _ = t ^ (q + r - 1) * Real.exp (-x * t) := by ring

private theorem log1p_cont : ContinuousOn (fun t : ℝ => Real.log (1 + t)) (Ioi 0) := by
  exact (continuousOn_const.add continuousOn_id).log fun t ht => by
    change 1 + t ≠ 0
    linarith [mem_Ioi.mp ht]

private theorem log1p_nonneg {t : ℝ} (ht : 0 < t) : 0 ≤ Real.log (1 + t) := by
  apply Real.log_nonneg
  linarith

private theorem log1p_moment_integrable (q x : ℝ) (hq : 0 < q) (hx : 0 < x) (n : ℕ) :
    IntegrableOn (fun t : ℝ => t ^ n * Real.log (1 + t) * kummerWeight q x t) (Ioi 0) := by
  have hdom : IntegrableOn (fun t : ℝ => t ^ (n + 1) * kummerWeight q x t) (Ioi 0) := by
    have hd := rpow_weight_integrable (r := ((n + 1 : ℕ) : ℝ)) hq hx (by positivity)
    simpa only [Real.rpow_natCast] using hd
  have hcont : ContinuousOn (fun t : ℝ => t ^ n * Real.log (1 + t) * kummerWeight q x t)
      (Ioi 0) := ((continuousOn_id.pow n).mul log1p_cont).mul (weight_cont q x)
  refine hdom.mono' (hcont.aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
  have ht0 : 0 < t := ht
  have hl := log1p_nonneg ht0
  have hw := weight_nonneg q x t ht0
  have hlog : Real.log (1 + t) ≤ t := by
    have := Real.log_le_sub_one_of_pos (by linarith : 0 < 1 + t)
    linarith
  rw [Real.norm_of_nonneg (by positivity)]
  calc t ^ n * Real.log (1 + t) * kummerWeight q x t
      ≤ t ^ n * t * kummerWeight q x t := by gcongr
    _ = t ^ (n + 1) * kummerWeight q x t := by rw [pow_succ]

theorem raw_moment_deriv (q x : ℝ) (hx : 0 < x) (g : ℝ → ℝ)
    (hg : ContinuousOn g (Ioi 0))
    (hint : IntegrableOn (fun t => g t * kummerWeight q x t) (Ioi 0))
    (hdom : IntegrableOn (fun t => ‖g t‖ * t * kummerWeight q (x / 2) t) (Ioi 0)) :
    HasDerivAt (fun y => ∫ t in Ioi (0 : ℝ), g t * kummerWeight q y t)
      (∫ t in Ioi (0 : ℝ), -(g t * t * kummerWeight q x t)) x := by
  let F : ℝ → ℝ → ℝ := fun y t => g t * kummerWeight q y t
  let F' : ℝ → ℝ → ℝ := fun y t => -(g t * t * kummerWeight q y t)
  have hF : ∀ y, AEStronglyMeasurable (F y) (volume.restrict (Ioi 0)) := by
    intro y
    exact (hg.mul (weight_cont q y)).aestronglyMeasurable measurableSet_Ioi
  have hF' : AEStronglyMeasurable (F' x) (volume.restrict (Ioi 0)) :=
    (((hg.mul continuousOn_id).mul (weight_cont q x)).neg).aestronglyMeasurable
      measurableSet_Ioi
  have hbound : ∀ᵐ t ∂volume.restrict (Ioi (0 : ℝ)),
      ∀ y ∈ Ioi (x / 2), ‖F' y t‖ ≤ ‖g t‖ * t * kummerWeight q (x / 2) t := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    intro y hy
    have ht0 : 0 < t := ht
    dsimp [F']
    rw [abs_neg, abs_mul, abs_mul, abs_of_nonneg ht0.le,
      abs_of_nonneg (weight_nonneg q y t ht0)]
    exact mul_le_mul_of_nonneg_left (weight_antitone q (x / 2) y t ht0 hy.le)
      (by positivity)
  have hdiff : ∀ᵐ t ∂volume.restrict (Ioi (0 : ℝ)),
      ∀ y ∈ Ioi (x / 2), HasDerivAt (F · t) (F' y t) y := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    intro y _
    dsimp [F, F', kummerWeight]
    have hd := (((hasDerivAt_id y).neg.mul_const t).exp).const_mul
      (g t * (t ^ (q - 1) * (1 + t) ^ (-q)))
    simp only [Pi.neg_apply, id_eq] at hd
    convert hd using 1
    · funext z
      ring
    · ring
  exact (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := volume.restrict (Ioi 0)) (s := Ioi (x / 2))
    (Ioi_mem_nhds (by linarith)) (Filter.Eventually.of_forall hF) hint hF'
    hbound hdom hdiff).2

/-- The Kummer mean derivative follows from dominated differentiation of the actual integral. -/
theorem kummerMean_deriv (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    HasDerivAt (kummerMean p) (-(kummerSecondMoment p κ - kummerMean p κ ^ 2)) κ := by
  have h := L3_kummer_moments p κ hp hκ
  have hhalf := L3_kummer_moments p (κ / 2) hp (by linarith)
  have hz : HasDerivAt (kummerZ p)
      (-(∫ t in Ioi (0 : ℝ), t * kummerWeight p κ t)) κ := by
    have hd := raw_moment_deriv p κ hκ (fun _ => 1) continuousOn_const
      (by simpa using h.2.1) (by simpa using hhalf.2.2.1)
    change HasDerivAt (fun x => ∫ t in Ioi (0 : ℝ), kummerWeight p x t) _ κ
    simpa only [one_mul, integral_neg] using hd
  have hm : HasDerivAt (fun x => ∫ t in Ioi (0 : ℝ), t * kummerWeight p x t)
      (-(∫ t in Ioi (0 : ℝ), t ^ 2 * kummerWeight p κ t)) κ := by
    have hdom : IntegrableOn (fun t : ℝ => ‖t‖ * t * kummerWeight p (κ / 2) t)
        (Ioi 0) := by
      refine hhalf.2.2.2.1.congr_fun (fun t ht => ?_) measurableSet_Ioi
      rw [Real.norm_of_nonneg (le_of_lt ht)]
      ring
    have hd := raw_moment_deriv p κ hκ (fun t => t) continuousOn_id h.2.2.1 hdom
    simpa only [← pow_two, integral_neg] using hd
  have hd := hm.div hz h.1.ne'
  change HasDerivAt (fun x => (kummerZ p x)⁻¹ *
    ∫ t in Ioi (0 : ℝ), t * kummerWeight p x t) _ κ
  convert hd using 1
  · funext x
    dsimp
    ring
  · dsimp [kummerSecondMoment, kummerMean, kummerExpect]
    field_simp
    ring


/-- Normalized-integral derivative for an integrable observable, used with concrete
Kummer observables below. Domination is a proved input, not a derivative assumption. -/
private theorem expect_deriv (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) (g : ℝ → ℝ)
    (hg : ContinuousOn g (Ioi 0))
    (hint : IntegrableOn (fun t => g t * kummerWeight p κ t) (Ioi 0))
    (hdom : IntegrableOn (fun t => ‖g t‖ * t * kummerWeight p (κ / 2) t) (Ioi 0)) :
    HasDerivAt (fun x => kummerExpect p x g)
      (-(kummerExpect p κ (fun t => g t * t) - kummerExpect p κ g * kummerMean p κ)) κ := by
  have h := L3_kummer_moments p κ hp hκ
  have hhalf := L3_kummer_moments p (κ / 2) hp (by linarith)
  have hz : HasDerivAt (kummerZ p)
      (-(∫ t in Ioi (0 : ℝ), t * kummerWeight p κ t)) κ := by
    have hd := raw_moment_deriv p κ hκ (fun _ => 1) continuousOn_const
      (by simpa using h.2.1) (by simpa using hhalf.2.2.1)
    change HasDerivAt (fun x => ∫ t in Ioi (0 : ℝ), kummerWeight p x t) _ κ
    simpa only [one_mul, integral_neg] using hd
  have hm := raw_moment_deriv p κ hκ g hg hint hdom
  simp only [integral_neg] at hm
  have hd := hm.div hz h.1.ne'
  change HasDerivAt (fun x => (kummerZ p x)⁻¹ *
    ∫ t in Ioi (0 : ℝ), g t * kummerWeight p x t) _ κ
  convert hd using 1
  · funext x
    dsimp
    ring
  · dsimp [kummerMean, kummerExpect]
    field_simp
    ring

noncomputable def kummerLog1p (p κ : ℝ) : ℝ :=
  kummerExpect p κ (fun t => Real.log (1 + t))
noncomputable def kummerCrossLog1p (p κ : ℝ) : ℝ :=
  kummerExpect p κ (fun t => t * Real.log (1 + t))
noncomputable def kummerSecondLog1p (p κ : ℝ) : ℝ :=
  kummerExpect p κ (fun t => t ^ 2 * Real.log (1 + t))

/-- The actual logarithmic Kummer moment satisfies the first covariance ODE. -/
theorem kummerLog1p_deriv (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    HasDerivAt (kummerLog1p p)
      (-(kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ)) κ := by
  have hi : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  have hd : IntegrableOn (fun t => ‖Real.log (1 + t)‖ * t * kummerWeight p (κ / 2) t)
      (Ioi 0) := by
    refine (log1p_moment_integrable p (κ / 2) hp (by linarith) 1).congr_fun
      (fun t ht => ?_) measurableSet_Ioi
    rw [Real.norm_of_nonneg (log1p_nonneg ht)]
    ring
  have h := expect_deriv p κ hp hκ (fun t => Real.log (1 + t)) log1p_cont hi hd
  change HasDerivAt (fun x => kummerExpect p x (fun t => Real.log (1 + t))) _ κ
  simpa only [kummerLog1p, kummerCrossLog1p, mul_comm (Real.log _) _] using h

/-- The actual first cross moment satisfies the next covariance ODE. -/
theorem kummerCrossLog1p_deriv (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    HasDerivAt (kummerCrossLog1p p)
      (-(kummerSecondLog1p p κ - kummerCrossLog1p p κ * kummerMean p κ)) κ := by
  have hi : IntegrableOn (fun t => t * Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_one] using log1p_moment_integrable p κ hp hκ 1
  have hd : IntegrableOn (fun t => ‖t * Real.log (1 + t)‖ * t * kummerWeight p (κ / 2) t)
      (Ioi 0) := by
    refine (log1p_moment_integrable p (κ / 2) hp (by linarith) 2).congr_fun
      (fun t ht => ?_) measurableSet_Ioi
    have ht0 : 0 < t := ht
    have hl := log1p_nonneg ht0
    rw [Real.norm_of_nonneg (by positivity)]
    ring
  have h := expect_deriv p κ hp hκ (fun t => t * Real.log (1 + t))
    (continuousOn_id.mul log1p_cont) hi hd
  have heq : (fun t : ℝ => t * Real.log (1 + t) * t) =
      (fun t : ℝ => t ^ 2 * Real.log (1 + t)) := by funext t; ring
  rw [heq] at h
  exact h


/-- The covariance of `T` and `log(1+T)` is strictly positive under the
actual Kummer law. Centering both functions at the mean gives a pointwise
nonnegative integrand, strictly positive on a nonempty interval. -/
theorem kummerLog1p_cov_pos (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    0 < kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ := by
  have h := L3_kummer_moments p κ hp hκ
  have hi0 := h.2.1
  have hi1 := h.2.2.1
  have hil : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  have hitl : IntegrableOn (fun t => t * Real.log (1 + t) * kummerWeight p κ t)
      (Ioi 0) := by
    simpa only [pow_one] using log1p_moment_integrable p κ hp hκ 1
  let m := kummerMean p κ
  have hm : 0 ≤ m := by
    dsimp [m, kummerMean, kummerExpect]
    apply mul_nonneg (inv_nonneg.mpr h.1.le)
    apply integral_nonneg_of_ae
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    exact mul_nonneg (le_of_lt ht) (weight_nonneg p κ t ht)
  let L := Real.log (1 + m)
  let v : ℝ → ℝ := fun t => (t - m) * (Real.log (1 + t) - L) * kummerWeight p κ t
  have hv_eq : v = fun t =>
      (t * Real.log (1 + t) * kummerWeight p κ t -
        m * (Real.log (1 + t) * kummerWeight p κ t)) -
        L * (t * kummerWeight p κ t) + (m * L) * kummerWeight p κ t := by
    funext t
    dsimp [v]
    ring
  have hiv : IntegrableOn v (Ioi 0) := by
    rw [hv_eq]
    exact ((hitl.sub (hil.const_mul m)).sub (hi1.const_mul L)).add
      (hi0.const_mul (m * L))
  have hvpos : 0 < ∫ t in Ioi (0 : ℝ), v t := by
    rw [setIntegral_pos_iff_support_of_nonneg_ae]
    · have hsub : Ioc (m + 1) (m + 2) ⊆ Function.support v ∩ Ioi 0 := by
        intro t ht
        have htm : m < t := by linarith [ht.1]
        have ht0 : 0 < t := lt_of_le_of_lt hm htm
        have hlog : L < Real.log (1 + t) := by
          dsimp [L]
          exact Real.log_lt_log (by linarith) (by linarith)
        refine ⟨Function.mem_support.mpr ?_, ht0⟩
        dsimp [v]
        exact ne_of_gt (mul_pos (mul_pos (sub_pos.mpr htm) (sub_pos.mpr hlog))
          (by unfold kummerWeight; positivity))
      refine lt_of_lt_of_le ?_ (measure_mono hsub)
      rw [Real.volume_Ioc]
      norm_num
    · filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
      have ht0 : 0 < t := ht
      dsimp [v]
      apply mul_nonneg _ (weight_nonneg p κ t ht0)
      rcases le_total m t with hmt | htm
      · apply mul_nonneg (sub_nonneg.mpr hmt)
        exact sub_nonneg.mpr (Real.log_le_log (by linarith) (by linarith))
      · apply mul_nonneg_of_nonpos_of_nonpos (sub_nonpos.mpr htm)
        exact sub_nonpos.mpr (Real.log_le_log (by linarith) (by linarith))
    · exact hiv
  have heval : (kummerZ p κ)⁻¹ * (∫ t in Ioi (0 : ℝ), v t) =
      kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ := by
    rw [hv_eq]
    rw [integral_add ((hitl.sub' (hil.const_mul m)).sub' (hi1.const_mul L))
        (hi0.const_mul (m * L)),
      integral_sub (hitl.sub' (hil.const_mul m)) (hi1.const_mul L),
      integral_sub hitl (hil.const_mul m), integral_const_mul, integral_const_mul,
      integral_const_mul]
    dsimp [m, kummerCrossLog1p, kummerLog1p, kummerMean, kummerExpect]
    change (kummerZ p κ)⁻¹ *
      ((_ - (kummerZ p κ)⁻¹ * _ * _) - L * _ +
        ((kummerZ p κ)⁻¹ * _ * L) * kummerZ p κ) = _
    field_simp [h.1.ne']
    ring
  rw [← heval]
  exact mul_pos (inv_pos.mpr h.1) hvpos

/-- Log-weighted integration by parts for the actual Kummer integrals.
The primitive has zero boundary limits and an integrable derivative. -/
theorem kummerLog1p_weighted_ibp (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    κ * kummerSecondLog1p p κ = kummerMean p κ + p * kummerLog1p p κ +
      (1 - κ) * kummerCrossLog1p p κ := by
  have h := L3_kummer_moments p κ hp hκ
  have hi1 := h.2.2.1
  have hil : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  have hitl : IntegrableOn (fun t => t * Real.log (1 + t) * kummerWeight p κ t)
      (Ioi 0) := by
    simpa only [pow_one] using log1p_moment_integrable p κ hp hκ 1
  have hit2l := log1p_moment_integrable p κ hp hκ 2
  let f : ℝ → ℝ := fun t =>
    t ^ p * (1 + t) ^ (1 - p) * Real.exp (-κ * t) * Real.log (1 + t)
  let f' : ℝ → ℝ := fun t =>
    p * (Real.log (1 + t) * kummerWeight p κ t) -
      (κ - 1) * (t * Real.log (1 + t) * kummerWeight p κ t) -
      κ * (t ^ 2 * Real.log (1 + t) * kummerWeight p κ t) + t * kummerWeight p κ t
  have hf0 : f 0 = 0 := by simp [f]
  have hf_cont : ContinuousWithinAt f (Ici 0) 0 := by
    dsimp [f]
    have hc1 : ContinuousWithinAt (fun t : ℝ => t ^ p) (Ici 0) 0 :=
      continuousWithinAt_id.rpow_const (Or.inr hp.le)
    have hc2 : ContinuousWithinAt (fun t : ℝ => (1 + t) ^ (1 - p)) (Ici 0) 0 :=
      ((continuousAt_const.add continuousAt_id).rpow_const
        (Or.inl (by norm_num))).continuousWithinAt
    have hc3 : ContinuousWithinAt (fun t : ℝ => Real.exp (-κ * t)) (Ici 0) 0 :=
      (continuous_const.mul continuous_id).rexp.continuousAt.continuousWithinAt
    have hc4 : ContinuousWithinAt (fun t : ℝ => Real.log (1 + t)) (Ici 0) 0 :=
      ((continuousAt_const.add continuousAt_id).log (by norm_num)).continuousWithinAt
    exact ((hc1.mul hc2).mul hc3).mul hc4
  have hf_deriv : ∀ t ∈ Ioi (0 : ℝ), HasDerivAt f (f' t) t := by
    intro t ht
    have ht0 : 0 < t := ht
    have h1t : 0 < 1 + t := by linarith
    have hd1 : HasDerivAt (fun s : ℝ => s ^ p) (p * t ^ (p - 1)) t :=
      Real.hasDerivAt_rpow_const (Or.inl ht0.ne')
    have hd2_in : HasDerivAt (fun s : ℝ => 1 + s) (0 + 1) t :=
      (hasDerivAt_const t (1 : ℝ)).fun_add (hasDerivAt_id' t)
    have hd2 : HasDerivAt (fun s : ℝ => (1 + s) ^ (1 - p))
        ((1 - p) * (1 + t) ^ (1 - p - 1) * (0 + 1)) t :=
      (Real.hasDerivAt_rpow_const (Or.inl h1t.ne')).comp t hd2_in
    have hd3 : HasDerivAt (fun s : ℝ => Real.exp (-κ * s))
        (Real.exp (-κ * t) * (-κ * 1)) t :=
      ((hasDerivAt_id' t).const_mul (-κ)).exp
    have hd4 := hd2_in.log h1t.ne'
    have hprod := ((hd1.fun_mul hd2).fun_mul hd3).fun_mul hd4
    refine hprod.congr_deriv ?_
    dsimp [f', kummerWeight]
    have hp_t1 : t ^ p = t * t ^ (p - 1) := by
      have := Real.rpow_add ht0 1 (p - 1)
      rw [Real.rpow_one] at this
      have heq : 1 + (p - 1) = p := by ring
      rwa [heq] at this
    have hp_u1 : (1 + t) ^ (1 - p) = (1 + t) * (1 + t) ^ (-p) := by
      have := Real.rpow_add h1t 1 (-p)
      rw [Real.rpow_one] at this
      have heq : 1 + -p = 1 - p := by ring
      rwa [heq] at this
    have hp_u2 : (1 + t) ^ (1 - p - 1) = (1 + t) ^ (-p) := by
      congr 1
      ring
    rw [hp_t1, hp_u1, hp_u2]
    field_simp
    ring
  have hf'_int : IntegrableOn f' (Ioi 0) := by
    dsimp [f']
    exact (((hil.const_mul p).sub' (hitl.const_mul (κ - 1))).sub'
      (hit2l.const_mul κ)).add hi1
  have hf_tendsto : Tendsto f atTop (𝓝 0) := by
    have hdom : Tendsto (fun t : ℝ => 2 * t ^ (p + 2) * Real.exp (-κ * t))
        atTop (𝓝 0) := by
      have h0 := tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero (p + 2) κ hκ
      simpa [mul_assoc] using h0.const_mul (2 : ℝ)
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hdom ?_ ?_
    · filter_upwards [Ioi_mem_atTop (1 : ℝ)] with t ht
      have ht0 : 0 < t := lt_trans zero_lt_one ht
      dsimp [f]
      exact mul_nonneg (by positivity) (log1p_nonneg ht0)
    · filter_upwards [Ioi_mem_atTop (1 : ℝ)] with t ht
      have ht0 : 0 < t := lt_trans zero_lt_one ht
      have ht1 : 1 < t := ht
      have hp_le : (1 + t) ^ (1 - p) ≤ 2 * t := by
        have hr := Real.rpow_le_rpow_of_exponent_le (by linarith : 1 ≤ 1 + t)
          (by linarith : 1 - p ≤ 1)
        rw [Real.rpow_one] at hr
        linarith
      have hl_le : Real.log (1 + t) ≤ t := by
        have := Real.log_le_sub_one_of_pos (by linarith : 0 < 1 + t)
        linarith
      have hl_nonneg := log1p_nonneg ht0
      dsimp [f]
      calc t ^ p * (1 + t) ^ (1 - p) * Real.exp (-κ * t) * Real.log (1 + t)
          ≤ t ^ p * (2 * t) * Real.exp (-κ * t) * t := by gcongr
        _ = 2 * (t ^ p * t ^ (2 : ℝ)) * Real.exp (-κ * t) := by
          rw [Real.rpow_two]
          ring
        _ = 2 * t ^ (p + 2) * Real.exp (-κ * t) := by rw [← Real.rpow_add ht0]
  have hibp_raw : ∫ t in Ioi (0 : ℝ), f' t = 0 := by
    simpa only [hf0, sub_zero] using
      integral_Ioi_of_hasDerivAt_of_tendsto hf_cont hf_deriv hf'_int hf_tendsto
  have hibp_split : ∫ t in Ioi (0 : ℝ), f' t =
      p * (∫ t in Ioi (0 : ℝ), Real.log (1 + t) * kummerWeight p κ t) -
      (κ - 1) * (∫ t in Ioi (0 : ℝ), t * Real.log (1 + t) * kummerWeight p κ t) -
      κ * (∫ t in Ioi (0 : ℝ), t ^ 2 * Real.log (1 + t) * kummerWeight p κ t) +
      (∫ t in Ioi (0 : ℝ), t * kummerWeight p κ t) := by
    dsimp [f']
    rw [integral_add (((hil.const_mul p).sub' (hitl.const_mul (κ - 1))).sub'
        (hit2l.const_mul κ)) hi1,
      integral_sub ((hil.const_mul p).sub' (hitl.const_mul (κ - 1))) (hit2l.const_mul κ),
      integral_sub (hil.const_mul p) (hitl.const_mul (κ - 1)),
      integral_const_mul, integral_const_mul, integral_const_mul]
  rw [hibp_split] at hibp_raw
  dsimp [kummerSecondLog1p, kummerMean, kummerLog1p, kummerCrossLog1p, kummerExpect]
  field_simp [h.1.ne']
  linarith

/-- The logarithmic score obtained by differentiating Kummer shape. -/
noncomputable def kummerShapeScore (t : ℝ) : ℝ := Real.log t - Real.log (1 + t)

theorem shapeScore_cont : ContinuousOn kummerShapeScore (Ioi 0) := by
  exact (Real.continuousOn_log.mono fun t ht => (mem_Ioi.mp ht).ne').sub log1p_cont

theorem shapeScore_integrable (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    IntegrableOn (fun t => kummerShapeScore t * kummerWeight p κ t) (Ioi 0) := by
  have hi := (L3_kummer_moments p κ hp hκ).2.2.2.2.1
  have hl : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  convert hi.sub hl using 1
  funext t
  dsimp [kummerShapeScore]
  ring

private theorem shapeScore_abs_integrable (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    IntegrableOn (fun t => ‖kummerShapeScore t‖ * kummerWeight p κ t) (Ioi 0) := by
  have hi : IntegrableOn (fun t => ‖kummerShapeScore t * kummerWeight p κ t‖) (Ioi 0) :=
    (shapeScore_integrable p κ hp hκ).norm
  refine hi.congr_fun (fun t ht => ?_) measurableSet_Ioi
  change |kummerShapeScore t * kummerWeight p κ t| = |kummerShapeScore t| * kummerWeight p κ t
  rw [abs_mul, abs_of_nonneg (weight_nonneg p κ t ht)]

theorem weight_shape_antitone (q r κ t : ℝ) (ht : 0 < t) (hqr : q ≤ r) :
    kummerWeight r κ t ≤ kummerWeight q κ t := by
  have h1 : 0 < 1 + t := by linarith
  have heq : ∀ a : ℝ, kummerWeight a κ t =
      (t / (1 + t)) ^ a / t * Real.exp (-κ * t) := by
    intro a
    unfold kummerWeight
    rw [Real.div_rpow ht.le h1.le, Real.rpow_sub ht, Real.rpow_one,
      Real.rpow_neg h1.le]
    ring
  rw [heq r, heq q]
  apply mul_le_mul_of_nonneg_right _ (Real.exp_pos _).le
  apply div_le_div_of_nonneg_right _ ht.le
  exact Real.rpow_le_rpow_of_exponent_ge (div_pos ht h1)
    ((div_le_one h1).mpr (by linarith)) hqr

private theorem raw_shape_deriv (p κ : ℝ) (hp : 0 < p) (g : ℝ → ℝ)
    (hg : ContinuousOn g (Ioi 0))
    (hint : IntegrableOn (fun t => g t * kummerWeight p κ t) (Ioi 0))
    (hdom : IntegrableOn (fun t => ‖g t‖ * ‖kummerShapeScore t‖ *
      kummerWeight (p / 2) κ t) (Ioi 0)) :
    HasDerivAt (fun q => ∫ t in Ioi (0 : ℝ), g t * kummerWeight q κ t)
      (∫ t in Ioi (0 : ℝ), g t * kummerShapeScore t * kummerWeight p κ t) p := by
  let F : ℝ → ℝ → ℝ := fun q t => g t * kummerWeight q κ t
  let F' : ℝ → ℝ → ℝ := fun q t => g t * kummerShapeScore t * kummerWeight q κ t
  have hF : ∀ q, AEStronglyMeasurable (F q) (volume.restrict (Ioi 0)) := by
    intro q
    exact (hg.mul (weight_cont q κ)).aestronglyMeasurable measurableSet_Ioi
  have hF' : AEStronglyMeasurable (F' p) (volume.restrict (Ioi 0)) :=
    ((hg.mul shapeScore_cont).mul (weight_cont p κ)).aestronglyMeasurable measurableSet_Ioi
  have hbound : ∀ᵐ t ∂volume.restrict (Ioi (0 : ℝ)),
      ∀ q ∈ Ioi (p / 2), ‖F' q t‖ ≤
        ‖g t‖ * ‖kummerShapeScore t‖ * kummerWeight (p / 2) κ t := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    intro q hq
    dsimp [F']
    rw [abs_mul, abs_mul, abs_of_nonneg (weight_nonneg q κ t ht)]
    exact mul_le_mul_of_nonneg_left (weight_shape_antitone (p / 2) q κ t ht hq.le)
      (by positivity)
  have hdiff : ∀ᵐ t ∂volume.restrict (Ioi (0 : ℝ)),
      ∀ q ∈ Ioi (p / 2), HasDerivAt (F · t) (F' q t) q := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    intro q _
    have ht0 : 0 < t := ht
    have h1t : 0 < 1 + t := by linarith
    have hd1 := ((hasDerivAt_id q).sub_const 1).const_rpow ht0
    have hd2 := (hasDerivAt_id q).neg.const_rpow h1t
    have hd := (((hd1.mul hd2).mul_const (Real.exp (-κ * t))).const_mul (g t))
    convert hd using 1
    · rfl
    · dsimp [F', kummerShapeScore, kummerWeight]
      ring
  exact (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := volume.restrict (Ioi 0)) (s := Ioi (p / 2))
    (Ioi_mem_nhds (by linarith)) (Filter.Eventually.of_forall hF) hint hF'
    hbound hdom hdiff).2

/-- Shape differentiation of the actual Kummer partition function. -/
theorem kummerLogZ_shape_deriv (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    HasDerivAt (fun q => Real.log (kummerZ q κ))
      (kummerExpect p κ Real.log - kummerLog1p p κ) p := by
  have h := L3_kummer_moments p κ hp hκ
  have hd := raw_shape_deriv p κ hp (fun _ => 1) continuousOn_const
    (by simpa only [one_mul] using h.2.1)
    (by simpa only [norm_one, one_mul] using
      shapeScore_abs_integrable (p / 2) κ (by linarith) hκ)
  simp only [one_mul] at hd
  have hl := hd.log h.1.ne'
  have hil : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  convert hl using 1
  · dsimp [kummerZ]
  · dsimp [kummerShapeScore, kummerExpect, kummerLog1p]
    simp_rw [sub_mul]
    rw [integral_sub h.2.2.2.2.1 hil]
    dsimp [kummerZ]
    ring

private theorem tlog_weight_integrable (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    IntegrableOn (fun t => t * Real.log t * kummerWeight p κ t) (Ioi 0) := by
  have h := L3_kummer_moments p κ hp hκ
  have hc : ContinuousOn (fun t => t * Real.log t * kummerWeight p κ t) (Ioi 0) :=
    (continuousOn_id.mul (Real.continuousOn_log.mono
      fun t ht => (mem_Ioi.mp ht).ne')).mul (weight_cont p κ)
  refine (h.2.1.add h.2.2.2.1).mono' (hc.aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
  have ht0 : 0 < t := ht
  have hw := weight_nonneg p κ t ht0
  have hbound : t * |Real.log t| ≤ 1 + t ^ 2 := by
    rcases le_or_gt t 1 with ht1 | ht1
    · rw [abs_of_nonpos (Real.log_nonpos ht0.le ht1)]
      have hl := Real.log_le_sub_one_of_pos (inv_pos.mpr ht0)
      rw [Real.log_inv] at hl
      have hm := mul_le_mul_of_nonneg_left hl ht0.le
      rw [mul_sub, mul_inv_cancel₀ ht0.ne'] at hm
      nlinarith [sq_nonneg t]
    · rw [abs_of_pos (Real.log_pos ht1)]
      have hl : Real.log t ≤ t := by linarith [Real.log_le_sub_one_of_pos ht0]
      nlinarith [mul_le_mul_of_nonneg_left hl ht0.le]
  change |t * Real.log t * kummerWeight p κ t| ≤
    kummerWeight p κ t + t ^ 2 * kummerWeight p κ t
  rw [abs_mul, abs_mul, abs_of_pos ht0, abs_of_nonneg hw]
  nlinarith [mul_le_mul_of_nonneg_right hbound hw]

theorem shapeScore_t_abs_integrable (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    IntegrableOn (fun t => t * ‖kummerShapeScore t‖ * kummerWeight p κ t) (Ioi 0) := by
  have hil := tlog_weight_integrable p κ hp hκ
  have hi1l : IntegrableOn (fun t => t * Real.log (1 + t) * kummerWeight p κ t)
      (Ioi 0) := by
    simpa only [pow_one] using log1p_moment_integrable p κ hp hκ 1
  have hi : IntegrableOn (fun t => t * kummerShapeScore t * kummerWeight p κ t)
      (Ioi 0) := by
    convert hil.sub hi1l using 1
    funext t
    dsimp [kummerShapeScore]
    ring
  have hn : IntegrableOn (fun t => ‖t * kummerShapeScore t * kummerWeight p κ t‖)
      (Ioi 0) := hi.norm
  refine hn.congr_fun (fun t ht => ?_) measurableSet_Ioi
  change |t * kummerShapeScore t * kummerWeight p κ t| =
    t * |kummerShapeScore t| * kummerWeight p κ t
  rw [abs_mul, abs_mul, abs_of_pos ht, abs_of_nonneg (weight_nonneg p κ t ht)]

private theorem expect_shape_deriv (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ)
    (g : ℝ → ℝ) (hg : ContinuousOn g (Ioi 0))
    (hint : IntegrableOn (fun t => g t * kummerWeight p κ t) (Ioi 0))
    (hdom : IntegrableOn (fun t => ‖g t‖ * ‖kummerShapeScore t‖ *
      kummerWeight (p / 2) κ t) (Ioi 0)) :
    HasDerivAt (fun q => kummerExpect q κ g)
      (kummerExpect p κ (fun t => g t * kummerShapeScore t) -
        kummerExpect p κ g * kummerExpect p κ kummerShapeScore) p := by
  have h := L3_kummer_moments p κ hp hκ
  have hz := raw_shape_deriv p κ hp (fun _ => 1) continuousOn_const
    (by simpa only [one_mul] using h.2.1)
    (by simpa only [norm_one, one_mul] using
      shapeScore_abs_integrable (p / 2) κ (by linarith) hκ)
  simp only [one_mul] at hz
  have hd := (raw_shape_deriv p κ hp g hg hint hdom).div hz h.1.ne'
  change HasDerivAt (fun q => (kummerZ q κ)⁻¹ *
    ∫ t in Ioi (0 : ℝ), g t * kummerWeight q κ t) _ p
  convert hd using 1
  · funext q
    dsimp [kummerZ]
    ring
  · dsimp [kummerExpect, kummerZ]
    field_simp <;> ring

/-- Shape derivative of the actual Kummer first moment. -/
theorem kummerMean_shape_deriv (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    HasDerivAt (fun q => kummerMean q κ)
      (kummerExpect p κ (fun t => t * kummerShapeScore t) -
        kummerMean p κ * kummerExpect p κ kummerShapeScore) p := by
  apply expect_shape_deriv p κ hp hκ (fun t => t) continuousOn_id
    (L3_kummer_moments p κ hp hκ).2.2.1
  refine (shapeScore_t_abs_integrable (p / 2) κ (by linarith) hκ).congr_fun
    (fun t ht => ?_) measurableSet_Ioi
  rw [Real.norm_of_nonneg (le_of_lt ht)]

private theorem expect_shapeScore (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    kummerExpect p κ kummerShapeScore = kummerExpect p κ Real.log - kummerLog1p p κ := by
  have hil := (L3_kummer_moments p κ hp hκ).2.2.2.2.1
  have hil1 : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  dsimp [kummerExpect, kummerShapeScore, kummerLog1p]
  simp_rw [sub_mul]
  rw [integral_sub hil hil1]
  ring

/-- Actual mixed differentiation identity in the order used by the mean
identity. Both derivative orders are identified by dominated integrals. -/
theorem kummer_shape_mixed (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    deriv (fun q => κ * kummerMean q κ) p =
      -κ * deriv (fun x => deriv (fun q => Real.log (kummerZ q x)) p) κ := by
  have heq : (fun x => deriv (fun q => Real.log (kummerZ q x)) p) =ᶠ[𝓝 κ]
      (fun x => kummerExpect p x kummerShapeScore) := by
    filter_upwards [Ioi_mem_nhds hκ] with x hx
    rw [(kummerLogZ_shape_deriv p x hp hx).deriv, expect_shapeScore p x hp hx]
  have hdom : IntegrableOn (fun t => ‖kummerShapeScore t‖ * t * kummerWeight p (κ / 2) t)
      (Ioi 0) := by
    refine (shapeScore_t_abs_integrable p (κ / 2) hp (by linarith)).congr_fun
      (fun t _ => ?_) measurableSet_Ioi
    ring
  have hd := expect_deriv p κ hp hκ kummerShapeScore shapeScore_cont
    (shapeScore_integrable p κ hp hκ) hdom
  rw [heq.deriv_eq, hd.deriv,
    ((kummerMean_shape_deriv p κ hp hκ).const_mul κ).deriv]
  have hcomm : (fun t : ℝ => t * kummerShapeScore t) =
      (fun t : ℝ => kummerShapeScore t * t) := by funext t; ring
  rw [hcomm]
  ring

private theorem gamma_weight_integrable (p b : ℝ) (hp : 0 < p) (hb : 0 < b) :
    IntegrableOn (fun t : ℝ => t ^ (p - 1) * Real.exp (-b * t)) (Ioi 0) := by
  simpa [Real.rpow_one] using
    (integrableOn_rpow_mul_exp_neg_mul_rpow (p := 1) (s := p - 1) (b := b)
      (by linarith) one_pos hb)

theorem gamma_log_integrable (p b : ℝ) (hp : 0 < p) (hb : 0 < b) :
    IntegrableOn (fun t : ℝ => Real.log t * (t ^ (p - 1) * Real.exp (-b * t)))
      (Ioi 0) := by
  have hd1 := (gamma_weight_integrable (p / 2) b (by linarith) hb).const_mul (2 / p)
  have hd2 := gamma_weight_integrable (p + 1) b (by linarith) hb
  have hc : ContinuousOn (fun t : ℝ => Real.log t * (t ^ (p - 1) * Real.exp (-b * t)))
      (Ioi 0) :=
    (Real.continuousOn_log.mono fun t ht => (mem_Ioi.mp ht).ne').mul
      ((continuousOn_id.rpow_const fun t ht => Or.inl (mem_Ioi.mp ht).ne').mul
        (continuous_const.mul continuous_id).rexp.continuousOn)
  refine (hd1.add hd2).mono' (hc.aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
  have ht0 : 0 < t := ht
  have hg : 0 ≤ t ^ (p - 1) * Real.exp (-b * t) := by positivity
  have hl : |Real.log t| ≤ (2 / p) * t ^ (-p / 2) + t := by
    rcases le_or_gt t 1 with ht1 | ht1
    · have habs : |Real.log t| = (2 / p) * Real.log (t ^ (-p / 2)) := by
        rw [abs_of_nonpos (Real.log_nonpos ht0.le ht1), Real.log_rpow ht0]
        field_simp [hp.ne']
      have hlog : Real.log (t ^ (-p / 2)) ≤ t ^ (-p / 2) := by
        linarith [Real.log_le_sub_one_of_pos (Real.rpow_pos_of_pos ht0 (-p / 2))]
      rw [habs]
      have hm := mul_le_mul_of_nonneg_left hlog (by positivity : 0 ≤ 2 / p)
      linarith
    · rw [abs_of_pos (Real.log_pos ht1)]
      have hnonneg : 0 ≤ (2 / p) * t ^ (-p / 2) := by positivity
      linarith [Real.log_le_sub_one_of_pos ht0]
  change |Real.log t * (t ^ (p - 1) * Real.exp (-b * t))| ≤
    2 / p * (t ^ (p / 2 - 1) * Real.exp (-b * t)) +
      t ^ (p + 1 - 1) * Real.exp (-b * t)
  rw [abs_mul, abs_of_nonneg hg]
  calc |Real.log t| * (t ^ (p - 1) * Real.exp (-b * t))
      ≤ ((2 / p) * t ^ (-p / 2) + t) * (t ^ (p - 1) * Real.exp (-b * t)) :=
        mul_le_mul_of_nonneg_right hl hg
    _ = 2 / p * (t ^ (p / 2 - 1) * Real.exp (-b * t)) +
        t ^ (p + 1 - 1) * Real.exp (-b * t) := by
      have he1 : t ^ (-p / 2) * t ^ (p - 1) = t ^ (p / 2 - 1) := by
        rw [← Real.rpow_add ht0]
        congr 1
        ring
      have he2 : t * t ^ (p - 1) = t ^ (p + 1 - 1) := by
        calc t * t ^ (p - 1) = t ^ (1 : ℝ) * t ^ (p - 1) := by rw [Real.rpow_one]
          _ = t ^ (p + 1 - 1) := by
            rw [← Real.rpow_add ht0]
            congr 1
            ring
      calc _ = (2 / p) * (t ^ (-p / 2) * t ^ (p - 1)) * Real.exp (-b * t) +
          (t * t ^ (p - 1)) * Real.exp (-b * t) := by ring
        _ = _ := by rw [he1, he2]; ring

private theorem gamma_log_integral (p b : ℝ) (hp : 0 < p) (hb : 0 < b) :
    (∫ t in Ioi (0 : ℝ), Real.log t * (t ^ (p - 1) * Real.exp (-b * t))) =
      (1 / b) ^ p * (deriv Real.Gamma p - Real.log b * Real.Gamma p) := by
  have hi : IntegrableOn (fun t : ℝ => Real.log t * (t ^ (p - 1) * Real.exp (-t)))
      (Ioi 0) := by simpa using gamma_log_integrable p 1 hp one_pos
  have hi0 : IntegrableOn (fun t : ℝ => t ^ (p - 1) * Real.exp (-t))
      (Ioi 0) := by simpa using gamma_weight_integrable p 1 hp one_pos
  have hG : ∫ t in Ioi (0 : ℝ), t ^ (p - 1) * Real.exp (-t) = Real.Gamma p := by
    simpa using Real.integral_rpow_mul_exp_neg_mul_Ioi hp (r := 1) one_pos
  have hG' : ∫ t in Ioi (0 : ℝ), Real.log t * (t ^ (p - 1) * Real.exp (-t)) =
      deriv Real.Gamma p := by
    rw [(Real.hasDerivAt_Gamma_of_pos hp).deriv]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro t _
    ring
  have hc := integral_comp_mul_left_Ioi
    (fun t : ℝ => (Real.log t - Real.log b) * (t ^ (p - 1) * Real.exp (-t))) 0 hb
  simp only [mul_zero, smul_eq_mul] at hc
  have he : ∫ t in Ioi (0 : ℝ),
      (Real.log (b * t) - Real.log b) * ((b * t) ^ (p - 1) * Real.exp (-(b * t))) =
      b ^ (p - 1) * ∫ t in Ioi (0 : ℝ), Real.log t * (t ^ (p - 1) * Real.exp (-b * t)) := by
    rw [← integral_const_mul]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro t ht
    dsimp only
    rw [Real.log_mul hb.ne' (ne_of_gt ht), Real.mul_rpow hb.le (le_of_lt ht)]
    ring
  rw [he] at hc
  have hr : (∫ t in Ioi (0 : ℝ),
      (Real.log t - Real.log b) * (t ^ (p - 1) * Real.exp (-t))) =
      deriv Real.Gamma p - Real.log b * Real.Gamma p := by
    simp_rw [sub_mul]
    rw [integral_sub hi (hi0.const_mul (Real.log b)),
      integral_const_mul, hG', hG]
  rw [hr] at hc
  have hpow : b ^ (p - 1) * (1 / b) ^ p = b⁻¹ := by
    rw [one_div, Real.inv_rpow hb.le, Real.rpow_sub hb, Real.rpow_one]
    field_simp
  have hbn : b ^ (p - 1) ≠ 0 := (Real.rpow_pos_of_pos hb _).ne'
  apply (mul_left_cancel₀ hbn)
  rw [← mul_assoc, hpow]
  exact hc

/-- Real digamma, written as the logarithmic derivative of the Gamma function. -/
noncomputable def gammaLogDerivative (p : ℝ) : ℝ := deriv Real.Gamma p / Real.Gamma p

/-- The symmetric bivariate gamma density proves the actual joint logarithmic
identity. Fubini is justified by an integrable product of gamma log moments. -/
theorem kummer_log_joint_identity (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    kummerExpect p κ Real.log + kummerLog1p p κ = gammaLogDerivative p - Real.log κ := by
  let μ : Measure ℝ := volume.restrict (Ioi 0)
  let a : ℝ → ℝ := fun t => t ^ (p - 1) * Real.exp (-κ * t)
  let J : ℝ → ℝ → ℝ := fun t u => a t * a u * Real.exp (-κ * t * u)
  let K := (1 / κ) ^ p
  have hK : 0 < K := Real.rpow_pos_of_pos (by positivity) _
  have hG : 0 < Real.Gamma p := Real.Gamma_pos_of_pos hp
  have hZ : 0 < kummerZ p κ := (L3_kummer_moments p κ hp hκ).1
  have ha : Integrable a μ := gamma_weight_integrable p κ hp hκ
  have hal : Integrable (fun t => Real.log t * a t) μ := gamma_log_integrable p κ hp hκ
  have hJ : Integrable (Function.uncurry (fun t u => Real.log t * J t u)) (μ.prod μ) := by
    have hdom := hal.norm.mul_prod ha
    refine hdom.mono' ?_ ?_
    · exact (by fun_prop : Measurable (Function.uncurry (fun t u => Real.log t * J t u))).aestronglyMeasurable
    · have hpos : ∀ᵐ z : ℝ × ℝ ∂μ.prod μ, 0 < z.1 ∧ 0 < z.2 := by
        dsimp only [μ]
        rw [Measure.prod_restrict]
        exact ae_restrict_mem (measurableSet_Ioi.prod measurableSet_Ioi)
      filter_upwards [hpos] with z hz
      have ha2 : 0 ≤ a z.2 := mul_nonneg (Real.rpow_nonneg hz.2.le _) (Real.exp_pos _).le
      have hexp : Real.exp (-κ * z.1 * z.2) ≤ 1 := by
        apply Real.exp_le_one_iff.mpr
        nlinarith [mul_pos (mul_pos hκ hz.1) hz.2]
      change |Real.log z.1 * (a z.1 * a z.2 * Real.exp (-κ * z.1 * z.2))| ≤
        |Real.log z.1 * a z.1| * a z.2
      have heq : Real.log z.1 * (a z.1 * a z.2 * Real.exp (-κ * z.1 * z.2)) =
          (Real.log z.1 * a z.1) * a z.2 * Real.exp (-κ * z.1 * z.2) := by ring
      rw [heq, abs_mul, abs_mul, abs_of_nonneg ha2, abs_of_pos (Real.exp_pos _)]
      nlinarith [mul_le_mul_of_nonneg_left hexp (mul_nonneg (abs_nonneg (Real.log z.1 * a z.1)) ha2)]
  have hsymm : ∀ t u, J t u = J u t := by
    intro t u
    dsimp [J]
    have he : -κ * t * u = -κ * u * t := by ring
    rw [he]
    ring
  have hfactor : ∀ t u, J t u = a t * (u ^ (p - 1) * Real.exp (-(κ * (1 + t)) * u)) := by
    intro t u
    calc J t u = a t * (u ^ (p - 1) *
          (Real.exp (-κ * u) * Real.exp (-κ * t * u))) := by dsimp [J, a]; ring
      _ = a t * (u ^ (p - 1) * Real.exp (-(κ * (1 + t)) * u)) := by
        rw [← Real.exp_add]
        have he : -κ * u + -κ * t * u = -(κ * (1 + t)) * u := by ring
        rw [he]
  have hrate : ∀ t, 0 < t → (1 / (κ * (1 + t))) ^ p = K * (1 + t) ^ (-p) := by
    intro t ht
    have h1 : 0 < 1 + t := by linarith
    dsimp [K]
    rw [one_div, mul_inv, Real.mul_rpow (inv_nonneg.mpr hκ.le) (inv_nonneg.mpr h1.le),
      Real.inv_rpow h1.le, Real.rpow_neg h1.le]
    simp only [one_div]
  have hmarg : ∀ t, 0 < t → (∫ u in Ioi (0 : ℝ), J t u) =
      K * Real.Gamma p * kummerWeight p κ t := by
    intro t ht
    simp_rw [hfactor t]
    rw [integral_const_mul]
    have hg : (∫ u in Ioi (0 : ℝ), u ^ (p - 1) * Real.exp (-(κ * (1 + t)) * u)) =
        (1 / (κ * (1 + t))) ^ p * Real.Gamma p := by
      simpa only [neg_mul] using Real.integral_rpow_mul_exp_neg_mul_Ioi hp
        (mul_pos hκ (by linarith : 0 < 1 + t))
    rw [hg, hrate t ht]
    dsimp [a, kummerWeight]
    ring
  have hlogmarg : ∀ t, 0 < t → (∫ u in Ioi (0 : ℝ), Real.log u * J t u) =
      K * kummerWeight p κ t *
        (deriv Real.Gamma p - (Real.log κ + Real.log (1 + t)) * Real.Gamma p) := by
    intro t ht
    have heq : (fun u => Real.log u * J t u) =
        (fun u => a t * (Real.log u * (u ^ (p - 1) * Real.exp (-(κ * (1 + t)) * u)))) := by
      funext u
      rw [hfactor]
      ring
    rw [heq, integral_const_mul,
      gamma_log_integral p (κ * (1 + t)) hp (mul_pos hκ (by linarith)), hrate t ht]
    rw [Real.log_mul hκ.ne' (by linarith : 1 + t ≠ 0)]
    dsimp [a, kummerWeight]
    ring
  have hs := integral_integral_swap hJ
  have hleft : (∫ t, ∫ u, Real.log t * J t u ∂μ ∂μ) =
      K * Real.Gamma p * (∫ t in Ioi (0 : ℝ), Real.log t * kummerWeight p κ t) := by
    simp_rw [integral_const_mul]
    change (∫ t in Ioi (0 : ℝ), Real.log t * (∫ u in Ioi (0 : ℝ), J t u)) = _
    calc
      _ = (∫ t in Ioi (0 : ℝ), (K * Real.Gamma p) *
          (Real.log t * kummerWeight p κ t)) := by
        apply setIntegral_congr_fun measurableSet_Ioi
        intro t ht
        dsimp only
        rw [hmarg t ht]
        ring
      _ = _ := integral_const_mul _ _
  have hir := (L3_kummer_moments p κ hp hκ).2.1
  have hil : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  have hright : (∫ u, ∫ t, Real.log t * J t u ∂μ ∂μ) =
      K * ((deriv Real.Gamma p - Real.log κ * Real.Gamma p) * kummerZ p κ -
        Real.Gamma p * (∫ u in Ioi (0 : ℝ), Real.log (1 + u) * kummerWeight p κ u)) := by
    change (∫ u in Ioi (0 : ℝ), ∫ t in Ioi (0 : ℝ), Real.log t * J t u) = _
    calc
      _ = (∫ u in Ioi (0 : ℝ), K *
          ((deriv Real.Gamma p - Real.log κ * Real.Gamma p) * kummerWeight p κ u -
            Real.Gamma p * (Real.log (1 + u) * kummerWeight p κ u))) := by
        apply setIntegral_congr_fun measurableSet_Ioi
        intro u hu
        dsimp only
        have heq : (fun t => Real.log t * J t u) = (fun t => Real.log t * J u t) := by
          funext t
          rw [hsymm t u]
        rw [heq, hlogmarg u hu]
        ring
      _ = _ := by
        rw [integral_const_mul,
          integral_sub (hir.const_mul (deriv Real.Gamma p - Real.log κ * Real.Gamma p))
            (hil.const_mul (Real.Gamma p)), integral_const_mul, integral_const_mul]
        rfl
  rw [hleft, hright] at hs
  have hraw : Real.Gamma p * (∫ t in Ioi (0 : ℝ), Real.log t * kummerWeight p κ t) =
      (deriv Real.Gamma p - Real.log κ * Real.Gamma p) * kummerZ p κ -
        Real.Gamma p * (∫ t in Ioi (0 : ℝ), Real.log (1 + t) * kummerWeight p κ t) := by
    apply mul_left_cancel₀ hK.ne'
    simpa only [mul_assoc] using hs
  dsimp [kummerExpect, kummerLog1p, gammaLogDerivative]
  field_simp [hG.ne', hZ.ne']
  nlinarith [hraw]

/-- Differentiating the proved actual joint-law identity gives the mean identity
in the original mixed-derivative order `∂κ∂p log Z`. -/
theorem kummer_differentiated_mean_identity (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    2 * deriv (kummerLog1p p) κ = -1 / κ -
      deriv (fun x => deriv (fun q => Real.log (kummerZ q x)) p) κ := by
  have heq : (fun x => deriv (fun q => Real.log (kummerZ q x)) p) =ᶠ[𝓝 κ]
      (fun x => gammaLogDerivative p - Real.log x - 2 * kummerLog1p p x) := by
    filter_upwards [Ioi_mem_nhds hκ] with x hx
    rw [(kummerLogZ_shape_deriv p x hp hx).deriv]
    have hJ := kummer_log_joint_identity p x hp hx
    linarith
  have hm := kummerLog1p_deriv p κ hp hκ
  have hd := ((hasDerivAt_const κ (gammaLogDerivative p)).fun_sub
    (Real.hasDerivAt_log hκ.ne')).fun_sub (hm.const_mul 2)
  rw [heq.deriv_eq, hd.deriv, hm.deriv]
  ring

private theorem gamma_first_integral (p b : ℝ) (hp : 0 < p) (hb : 0 < b) :
    (∫ t in Ioi (0 : ℝ), t * (t ^ (p - 1) * Real.exp (-b * t))) =
      (p / b) * ((1 / b) ^ p * Real.Gamma p) := by
  have heq : ∀ t > 0, t * (t ^ (p - 1) * Real.exp (-b * t)) =
      t ^ (p + 1 - 1) * Real.exp (-b * t) := by
    intro t ht
    have htprod : t * t ^ (p - 1) = t ^ (p + 1 - 1) := by
      calc t * t ^ (p - 1) = t ^ (1 : ℝ) * t ^ (p - 1) := by rw [Real.rpow_one]
        _ = t ^ (p + 1 - 1) := by rw [← Real.rpow_add ht]; congr 1; ring
    rw [← mul_assoc, htprod]
  have he := setIntegral_congr_fun (μ := volume) measurableSet_Ioi heq
  rw [he]
  have hg : (∫ t in Ioi (0 : ℝ), t ^ (p + 1 - 1) * Real.exp (-b * t)) =
      (1 / b) ^ (p + 1) * Real.Gamma (p + 1) := by
    simpa only [neg_mul] using Real.integral_rpow_mul_exp_neg_mul_Ioi
      (show 0 < p + 1 by linarith) hb
  rw [hg, Real.Gamma_add_one hp.ne', Real.rpow_add (by positivity : 0 < 1 / b),
    Real.rpow_one]
  ring

theorem gamma_first_integrable (p b : ℝ) (hp : 0 < p) (hb : 0 < b) :
    IntegrableOn (fun t => t * (t ^ (p - 1) * Real.exp (-b * t))) (Ioi 0) := by
  apply (gamma_weight_integrable (p + 1) b (by linarith) hb).congr_fun _ measurableSet_Ioi
  intro t ht
  have he : t * t ^ (p - 1) = t ^ (p + 1 - 1) := by
    calc t * t ^ (p - 1) = t ^ (1 : ℝ) * t ^ (p - 1) := by rw [Real.rpow_one]
      _ = t ^ (p + 1 - 1) := by rw [← Real.rpow_add ht]; congr 1; ring
  dsimp only
  rw [← he]
  ring

theorem gamma_log_affine_integrable (p b s : ℝ) (hp : 0 < p) (hb : 0 < b)
    (hs : 0 < s) : IntegrableOn
      (fun t => Real.log (1 + s * t) * (t ^ (p - 1) * Real.exp (-b * t))) (Ioi 0) := by
  apply ((gamma_first_integrable p b hp hb).const_mul s).mono' (by fun_prop)
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
  have ht0 : 0 < t := ht
  have hx : 0 < s * t := mul_pos hs ht0
  have hw : 0 ≤ t ^ (p - 1) * Real.exp (-b * t) := by positivity
  have hl : 0 ≤ Real.log (1 + s * t) := Real.log_nonneg (by linarith)
  rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg hl, abs_of_nonneg hw]
  have h := Real.log_le_sub_one_of_pos (by linarith : 0 < 1 + s * t)
  nlinarith [mul_le_mul_of_nonneg_right h hw]

private theorem log_affine_tangent (s m t : ℝ) (hs : 0 < s) (hm : 0 < m) (ht : 0 < t) :
    0 ≤ Real.log (1 + s * m) + (s / (1 + s * m)) * (t - m) - Real.log (1 + s * t) := by
  have ha : 0 < 1 + s * m := by positivity
  have hc : 0 < 1 + s * t := by positivity
  have h := Real.log_le_sub_one_of_pos (div_pos hc ha)
  rw [Real.log_div hc.ne' ha.ne'] at h
  have he : (1 + s * t) / (1 + s * m) - 1 = (s / (1 + s * m)) * (t - m) := by
    field_simp
    ring
  rw [he] at h
  linarith

private theorem log_affine_tangent_pos (s m t : ℝ) (hs : 0 < s) (hm : 0 < m)
    (ht : 0 < t) (hne : t ≠ m) :
    0 < Real.log (1 + s * m) + (s / (1 + s * m)) * (t - m) - Real.log (1 + s * t) := by
  have ha : 0 < 1 + s * m := by positivity
  have hc : 0 < 1 + s * t := by positivity
  have hne' : (1 + s * t) / (1 + s * m) ≠ 1 := by
    intro hh
    have h := (div_eq_one_iff_eq ha.ne').mp hh
    apply hne
    nlinarith
  have h := Real.log_lt_sub_one_of_pos (div_pos hc ha) hne'
  rw [Real.log_div hc.ne' ha.ne'] at h
  have he : (1 + s * t) / (1 + s * m) - 1 = (s / (1 + s * m)) * (t - m) := by
    field_simp
    ring
  rw [he] at h
  linarith

private theorem gamma_log_affine_lt (p b s : ℝ) (hp : 0 < p) (hb : 0 < b) (hs : 0 < s) :
    (∫ t in Ioi (0 : ℝ), Real.log (1 + s * t) * (t ^ (p - 1) * Real.exp (-b * t))) <
      Real.log (1 + s * (p / b)) * ((1 / b) ^ p * Real.Gamma p) := by
  let a : ℝ → ℝ := fun t => t ^ (p - 1) * Real.exp (-b * t)
  let m := p / b
  let A := Real.log (1 + s * m)
  let B := s / (1 + s * m)
  have hm : 0 < m := div_pos hp hb
  have hi := gamma_weight_integrable p b hp hb
  have hi1 := gamma_first_integrable p b hp hb
  have hil := gamma_log_affine_integrable p b s hp hb hs
  let f : ℝ → ℝ := fun t => (A + B * (t - m) - Real.log (1 + s * t)) * a t
  have hfeq : f = fun t => A * a t + B * (t * a t) - (B * m) * a t -
      Real.log (1 + s * t) * a t := by funext t; dsimp [f]; ring
  have hif : IntegrableOn f (Ioi 0) := by
    rw [hfeq]
    exact (((hi.const_mul A).add (hi1.const_mul B)).sub (hi.const_mul (B * m))).sub hil
  have hpos : 0 < ∫ t in Ioi (0 : ℝ), f t := by
    rw [setIntegral_pos_iff_support_of_nonneg_ae]
    · have hsub : Ioc (m + 1) (m + 2) ⊆ Function.support f ∩ Ioi 0 := by
        intro t ht
        have ht0 : 0 < t := by linarith [ht.1]
        have hg := log_affine_tangent_pos s m t hs hm ht0 (by linarith [ht.1])
        refine ⟨Function.mem_support.mpr ?_, ht0⟩
        exact (mul_pos hg (by dsimp [a]; positivity)).ne'
      refine lt_of_lt_of_le ?_ (measure_mono hsub)
      rw [Real.volume_Ioc]
      norm_num
    · filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
      have ht0 : 0 < t := ht
      exact mul_nonneg (log_affine_tangent s m t hs hm ht) (by dsimp [a]; positivity)
    · exact hif
  have hiA : IntegrableOn (fun t => A * a t + B * (t * a t)) (Ioi 0) :=
    (hi.const_mul A).add (hi1.const_mul B)
  have hiAB : IntegrableOn (fun t => A * a t + B * (t * a t) - (B * m) * a t) (Ioi 0) :=
    hiA.sub (hi.const_mul (B * m))
  rw [hfeq] at hpos
  dsimp only at hpos
  rw [integral_sub hiAB hil, integral_sub hiA (hi.const_mul (B * m)),
    integral_add (hi.const_mul A) (hi1.const_mul B), integral_const_mul, integral_const_mul,
    integral_const_mul] at hpos
  have ha : (∫ t in Ioi (0 : ℝ), a t) = (1 / b) ^ p * Real.Gamma p := by
    simpa only [a, neg_mul] using Real.integral_rpow_mul_exp_neg_mul_Ioi hp hb
  have ha1 : (∫ t in Ioi (0 : ℝ), t * a t) = m * ((1 / b) ^ p * Real.Gamma p) :=
    gamma_first_integral p b hp hb
  rw [ha, ha1] at hpos
  change _ < A * _
  nlinarith

/-- The three Kummer ODEs and strict covariance positivity for the actual
normalized moments, assuming only `κ > 0` and `p > 0`. The moment derivatives,
`kummerLog1p_weighted_ibp` and `kummerLog1p_cov_pos` are proved lemmas;
ordinary integration by parts follows from `NoiseAnalyticity.L3_kummer_moments`.
Coverage correction, 2026-11-22: no project-specific input remains here. -/
theorem kummer_odes (κ p : ℝ) (hκ : 0 < κ) (hp : 0 < p) :
    κ * deriv (kummerLog1p p) κ = -(κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ)) ∧
    κ * deriv (fun x => x * kummerMean p x) κ = (κ * kummerMean p κ) ^ 2 + κ * (κ * kummerMean p κ) - p * κ ∧
    κ * deriv (fun x => x * (kummerCrossLog1p p x - kummerLog1p p x * kummerMean p x)) κ =
      (κ + 2 * (κ * kummerMean p κ)) * (κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ)) - κ * kummerMean p κ ∧
    0 < κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ) := by
  have hwibp := kummerLog1p_weighted_ibp p κ hp hκ
  have hcov := kummerLog1p_cov_pos p κ hp hκ
  have hm := kummerLog1p_deriv p κ hp hκ
  have hμ := kummerMean_deriv p κ hp hκ
  have hν := kummerCrossLog1p_deriv p κ hp hκ
  have hibp : κ * kummerSecondMoment p κ = p + (1 - κ) * kummerMean p κ := by
    have h := (NoiseAnalyticity.L3_kummer_moments p κ hp hκ).2.2.2.2.2.1
    change κ * kummerSecondMoment p κ + (κ - 1) * kummerMean p κ = p at h
    nlinarith
  have ha := (hasDerivAt_id κ).mul hμ
  have hV := (hasDerivAt_id κ).mul (hν.sub (hm.mul hμ))
  change HasDerivAt (fun x => x * kummerMean p x) _ κ at ha
  change HasDerivAt (fun x => x * (kummerCrossLog1p p x - kummerLog1p p x * kummerMean p x)) _ κ at hV
  refine ⟨?_, ?_, ?_, mul_pos hκ hcov⟩
  · rw [hm.deriv]
    ring
  · rw [ha.deriv]
    dsimp
    nlinarith [hibp]
  · rw [hV.deriv]
    dsimp
    linear_combination -κ * hwibp + (κ * kummerLog1p p κ) * hibp

/-- The positive Kummer argument at log threshold `D`, where the physical
transfer parameter is `c = (1-s) exp D`. -/
noncomputable def exactKappa (p s D : ℝ) : ℝ :=
  s ^ 2 * p / ((1 - s) * Real.exp D)

/-- Actual gamma-transfer growth rate in the manuscript's log threshold
coordinate, expressed through its normalized Kummer integral. -/
noncomputable def exactGrowth (p s D : ℝ) : ℝ :=
  Real.log s + kummerLog1p p (exactKappa p s D)

private theorem exactKappa_pos (p s D : ℝ) (hp : 0 < p) (hs : 0 < s)
    (hs1 : s < 1) : 0 < exactKappa p s D := by
  unfold exactKappa
  exact div_pos (mul_pos (sq_pos_of_pos hs) hp)
    (mul_pos (sub_pos.mpr hs1) (Real.exp_pos D))

private theorem exactKappa_deriv (p s D : ℝ) (hs1 : s < 1) :
    HasDerivAt (exactKappa p s) (-exactKappa p s D) D := by
  have hden : (1 - s) * Real.exp D ≠ 0 :=
    (mul_pos (sub_pos.mpr hs1) (Real.exp_pos D)).ne'
  have hd := (hasDerivAt_const D (s ^ 2 * p)).div
    ((Real.hasDerivAt_exp D).const_mul (1 - s)) hden
  convert hd using 1
  · rfl
  · dsimp [exactKappa]
    field_simp
    ring

/-- Differentiating the actual growth integral gives a strictly positive
Kummer covariance times its positive argument. -/
theorem exactGrowth_deriv (p s D : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1) :
    HasDerivAt (exactGrowth p s)
      (exactKappa p s D * (kummerCrossLog1p p (exactKappa p s D) -
        kummerLog1p p (exactKappa p s D) * kummerMean p (exactKappa p s D))) D := by
  have hd := ((kummerLog1p_deriv p (exactKappa p s D) hp
    (exactKappa_pos p s D hp hs hs1)).comp D (exactKappa_deriv p s D hs1)).const_add
      (Real.log s)
  convert hd using 1
  · rfl
  · ring

/-- Continuity of the actual growth function follows from the proved
normalized-integral derivative; it is not supplied as a model premise. -/
theorem exactGrowth_continuous (p s : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1) :
    Continuous (exactGrowth p s) := by
  exact continuous_iff_continuousAt.mpr fun D => (exactGrowth_deriv p s D hp hs hs1).continuousAt

/-- Strict increase of actual growth is the positive-covariance argument in
D-exact.tex, expressed in the strictly increasing log-transfer coordinate. -/
theorem exactGrowth_strictMono (p s : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1) :
    StrictMono (exactGrowth p s) := by
  apply strictMono_of_deriv_pos
  intro D
  rw [(exactGrowth_deriv p s D hp hs hs1).deriv]
  exact mul_pos (exactKappa_pos p s D hp hs hs1)
    (kummerLog1p_cov_pos p (exactKappa p s D) hp (exactKappa_pos p s D hp hs hs1))

private noncomputable def softplus (x : ℝ) : ℝ := Real.log (1 + Real.exp x)
private noncomputable def softplusSlope (x : ℝ) : ℝ := Real.exp x / (1 + Real.exp x)

private theorem softplus_deriv (x : ℝ) : HasDerivAt softplus (softplusSlope x) x := by
  exact ((Real.hasDerivAt_exp x).const_add 1).log (by positivity)

private theorem softplus_strictConvex : StrictConvexOn ℝ univ softplus := by
  have hc : Continuous softplus :=
    continuous_iff_continuousAt.mpr fun x => (softplus_deriv x).continuousAt
  have hm : StrictMono (deriv softplus) := by
    intro x y hxy
    rw [(softplus_deriv x).deriv, (softplus_deriv y).deriv]
    dsimp [softplusSlope]
    apply (div_lt_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith [Real.exp_lt_exp.mpr hxy]
  exact hm.strictConvexOn_univ_of_deriv hc

private theorem softplus_tangent_pos (l x : ℝ) (hne : x ≠ l) :
    0 < softplus x - softplus l - softplusSlope l * (x - l) := by
  rcases lt_or_gt_of_ne hne with hxl | hlx
  · have h := softplus_strictConvex.slope_lt_of_hasDerivAt (mem_univ x) (mem_univ l)
      hxl (softplus_deriv l)
    rw [slope_def_field] at h
    have hh := (div_lt_iff₀ (sub_pos.mpr hxl)).mp h
    nlinarith
  · have h := softplus_strictConvex.lt_slope_of_hasDerivAt (mem_univ l) (mem_univ x)
      hlx (softplus_deriv l)
    rw [slope_def_field] at h
    have hh := (lt_div_iff₀ (sub_pos.mpr hlx)).mp h
    nlinarith

private theorem softplus_tangent_nonneg (l x : ℝ) :
    0 ≤ softplus x - softplus l - softplusSlope l * (x - l) := by
  by_cases he : x = l
  · subst x
    simp
  · exact (softplus_tangent_pos l x he).le

/-- Strict Jensen inequality for the actual Kummer log coordinate, proved by
integrating a strictly positive softplus tangent gap against the density. -/
theorem kummer_log_strict_jensen (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    Real.log (1 + Real.exp (kummerExpect p κ Real.log)) < kummerLog1p p κ := by
  have h := L3_kummer_moments p κ hp hκ
  have hi0 := h.2.1
  have hil := h.2.2.2.2.1
  have hi1l : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  let l := kummerExpect p κ Real.log
  let b := softplusSlope l
  let B := softplus l
  let v : ℝ → ℝ := fun t =>
    (Real.log (1 + t) - B - b * (Real.log t - l)) * kummerWeight p κ t
  have hveq : v = fun t =>
      (Real.log (1 + t) * kummerWeight p κ t - B * kummerWeight p κ t) -
      b * (Real.log t * kummerWeight p κ t) + (b * l) * kummerWeight p κ t := by
    funext t
    dsimp [v]
    ring
  have hiv : IntegrableOn v (Ioi 0) := by
    rw [hveq]
    exact ((hi1l.sub (hi0.const_mul B)).sub (hil.const_mul b)).add (hi0.const_mul (b * l))
  have hvpos : 0 < ∫ t in Ioi (0 : ℝ), v t := by
    rw [setIntegral_pos_iff_support_of_nonneg_ae]
    · have hsub : Ioc (Real.exp l + 1) (Real.exp l + 2) ⊆ Function.support v ∩ Ioi 0 := by
        intro t ht
        have ht0 : 0 < t := by linarith [ht.1, Real.exp_pos l]
        have hlog : l < Real.log t := by
          have hh := Real.log_lt_log (Real.exp_pos l) (by linarith [ht.1] : Real.exp l < t)
          simpa only [Real.log_exp] using hh
        have hg := softplus_tangent_pos l (Real.log t) hlog.ne'
        have hg' : 0 < Real.log (1 + t) - B - b * (Real.log t - l) := by
          simpa only [B, b, softplus, Real.exp_log ht0] using hg
        refine ⟨Function.mem_support.mpr ?_, ht0⟩
        exact (mul_pos hg' (by unfold kummerWeight; positivity)).ne'
      refine lt_of_lt_of_le ?_ (measure_mono hsub)
      rw [Real.volume_Ioc]
      norm_num
    · filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
      have hg := softplus_tangent_nonneg l (Real.log t)
      have hg' : 0 ≤ Real.log (1 + t) - B - b * (Real.log t - l) := by
        simpa only [B, b, softplus, Real.exp_log ht] using hg
      exact mul_nonneg hg' (weight_nonneg p κ t ht)
    · exact hiv
  have hlraw : (∫ t in Ioi (0 : ℝ), Real.log t * kummerWeight p κ t) = l * kummerZ p κ := by
    dsimp [l, kummerExpect]
    field_simp [h.1.ne']
  have heval : (kummerZ p κ)⁻¹ * (∫ t in Ioi (0 : ℝ), v t) = kummerLog1p p κ - B := by
    rw [hveq, integral_add ((hi1l.sub' (hi0.const_mul B)).sub' (hil.const_mul b))
      (hi0.const_mul (b * l)),
      integral_sub (hi1l.sub' (hi0.const_mul B)) (hil.const_mul b),
      integral_sub hi1l (hi0.const_mul B), integral_const_mul, integral_const_mul,
      integral_const_mul, hlraw]
    change (kummerZ p κ)⁻¹ *
      ((_ - B * kummerZ p κ) - b * (l * kummerZ p κ) + (b * l) * kummerZ p κ) = _
    dsimp [kummerLog1p, kummerExpect]
    field_simp [h.1.ne']
    ring
  have hgap : 0 < kummerLog1p p κ - B := by
    rw [← heval]
    exact mul_pos (inv_pos.mpr h.1) hvpos
  exact sub_pos.mp hgap

/-- The gamma log loss appearing in the manuscript's upper root bound. -/
noncomputable def gammaLogLoss (p : ℝ) : ℝ := Real.log p - gammaLogDerivative p

/-- Actual growth is positive at the manuscript's gamma log-loss bound. -/
theorem exactGrowth_gammaLogLoss_pos (p s : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1) :
    0 < exactGrowth p s (gammaLogLoss p) := by
  let κ := exactKappa p s (gammaLogLoss p)
  have hκ : 0 < κ := exactKappa_pos p s (gammaLogLoss p) hp hs hs1
  have hlogκ : Real.log κ = 2 * Real.log s + gammaLogDerivative p - Real.log (1 - s) := by
    dsimp [κ, exactKappa, gammaLogLoss]
    rw [Real.log_div (mul_pos (sq_pos_of_pos hs) hp).ne'
      (mul_pos (sub_pos.mpr hs1) (Real.exp_pos _)).ne',
      Real.log_mul (sq_pos_of_pos hs).ne' hp.ne',
      Real.log_mul (sub_pos.mpr hs1).ne' (Real.exp_pos _).ne', Real.log_pow, Real.log_exp]
    ring
  have hJ := kummer_log_joint_identity p κ hp hκ
  rw [hlogκ] at hJ
  have hj := kummer_log_strict_jensen p κ hp hκ
  by_contra hn
  have hr : Real.log s + kummerLog1p p κ ≤ 0 := le_of_not_gt hn
  have hbase : 0 < (1 - s) / s := div_pos (sub_pos.mpr hs1) hs
  have hlogbase : Real.log ((1 - s) / s) ≤ kummerExpect p κ Real.log := by
    rw [Real.log_div (sub_pos.mpr hs1).ne' hs.ne']
    linarith
  have hexp := Real.exp_le_exp.mpr hlogbase
  rw [Real.exp_log hbase] at hexp
  have hlog := Real.log_le_log (by linarith : 0 < 1 + (1 - s) / s)
    (add_le_add_right hexp 1)
  have heq : Real.log (1 + (1 - s) / s) = -Real.log s := by
    have ha : 1 + (1 - s) / s = 1 / s := by field_simp; ring
    rw [ha, Real.log_div one_ne_zero hs.ne', Real.log_one, zero_sub]
  rw [heq] at hlog
  linarith

/-- Conditional strict logarithmic Jensen under the symmetric gamma density
proves negative growth at the deterministic threshold `c=1-s`. -/
theorem exactGrowth_zero_neg (p s : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1) :
    exactGrowth p s 0 < 0 := by
  let κ := exactKappa p s 0
  have hκ : 0 < κ := exactKappa_pos p s 0 hp hs hs1
  let μ : Measure ℝ := volume.restrict (Ioi 0)
  let a : ℝ → ℝ := fun t => t ^ (p - 1) * Real.exp (-κ * t)
  let J : ℝ → ℝ → ℝ := fun t u => a t * a u * Real.exp (-κ * t * u)
  let F : ℝ → ℝ := fun t => Real.log (1 + s * t)
  let K := (1 / κ) ^ p
  let C := K * Real.Gamma p
  have hK : 0 < K := Real.rpow_pos_of_pos (by positivity) _
  have hC : 0 < C := mul_pos hK (Real.Gamma_pos_of_pos hp)
  have hZ := (L3_kummer_moments p κ hp hκ).1
  have ha := gamma_weight_integrable p κ hp hκ
  have ha1 := gamma_first_integrable p κ hp hκ
  have hF : IntegrableOn (fun t => F t * kummerWeight p κ t) (Ioi 0) := by
    have hm := (L3_kummer_moments p κ hp hκ).2.2.1
    apply (hm.const_mul s).mono' (by
      exact (by dsimp [F, kummerWeight]; fun_prop : Measurable (fun t => F t * kummerWeight p κ t)).aestronglyMeasurable)
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have ht0 : 0 < t := ht
    have hw := weight_nonneg p κ t ht
    have hf : 0 ≤ F t := Real.log_nonneg (by nlinarith [mul_pos hs ht0])
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg hf, abs_of_nonneg hw]
    have hl : F t ≤ s * t := by
      dsimp [F]
      linarith [Real.log_le_sub_one_of_pos (by positivity : 0 < 1 + s * t)]
    nlinarith [mul_le_mul_of_nonneg_right hl hw]
  have hJ : Integrable (Function.uncurry (fun t u => F t * J t u)) (μ.prod μ) := by
    have hdom := (ha1.const_mul s).mul_prod ha
    refine hdom.mono' (by fun_prop) ?_
    have hpos : ∀ᵐ z : ℝ × ℝ ∂μ.prod μ, 0 < z.1 ∧ 0 < z.2 := by
      dsimp only [μ]
      rw [Measure.prod_restrict]
      exact ae_restrict_mem (measurableSet_Ioi.prod measurableSet_Ioi)
    filter_upwards [hpos] with z hz
    have ha1p : 0 ≤ a z.1 := mul_nonneg (Real.rpow_nonneg hz.1.le _) (Real.exp_pos _).le
    have ha2p : 0 ≤ a z.2 := mul_nonneg (Real.rpow_nonneg hz.2.le _) (Real.exp_pos _).le
    have he : Real.exp (-κ * z.1 * z.2) ≤ 1 := by
      apply Real.exp_le_one_iff.mpr
      nlinarith [mul_pos (mul_pos hκ hz.1) hz.2]
    have hf : 0 ≤ F z.1 := Real.log_nonneg (by nlinarith [mul_pos hs hz.1])
    have hfb : F z.1 ≤ s * z.1 := by
      dsimp [F]
      linarith [Real.log_le_sub_one_of_pos (by nlinarith [mul_pos hs hz.1] : 0 < 1 + s * z.1)]
    have hsp : 0 ≤ s * z.1 := (mul_pos hs hz.1).le
    change |F z.1 * (a z.1 * a z.2 * Real.exp (-κ * z.1 * z.2))| ≤
      (s * (z.1 * a z.1)) * a z.2
    rw [abs_of_nonneg (by positivity)]
    calc F z.1 * (a z.1 * a z.2 * Real.exp (-κ * z.1 * z.2))
        ≤ (s * z.1) * (a z.1 * a z.2 * Real.exp (-κ * z.1 * z.2)) := by gcongr
      _ ≤ (s * z.1) * (a z.1 * a z.2 * 1) := by gcongr
      _ = _ := by ring
  have hsymm : ∀ t u, J t u = J u t := by
    intro t u
    dsimp [J]
    have he : -κ * t * u = -κ * u * t := by ring
    rw [he]
    ring
  have hfactor : ∀ t u, J t u = a t * (u ^ (p - 1) * Real.exp (-(κ * (1 + t)) * u)) := by
    intro t u
    calc J t u = a t * (u ^ (p - 1) *
          (Real.exp (-κ * u) * Real.exp (-κ * t * u))) := by dsimp [J, a]; ring
      _ = a t * (u ^ (p - 1) * Real.exp (-(κ * (1 + t)) * u)) := by
        rw [← Real.exp_add]
        have he : -κ * u + -κ * t * u = -(κ * (1 + t)) * u := by ring
        rw [he]
  have hrate : ∀ t, 0 < t → (1 / (κ * (1 + t))) ^ p = K * (1 + t) ^ (-p) := by
    intro t ht
    have h1 : 0 < 1 + t := by linarith
    dsimp [K]
    rw [one_div, mul_inv, Real.mul_rpow (inv_nonneg.mpr hκ.le) (inv_nonneg.mpr h1.le),
      Real.inv_rpow h1.le, Real.rpow_neg h1.le]
    simp only [one_div]
  have hmarg : ∀ t, 0 < t → (∫ u in Ioi (0 : ℝ), J t u) =
      K * Real.Gamma p * kummerWeight p κ t := by
    intro t ht
    simp_rw [hfactor t]
    rw [integral_const_mul]
    have hg : (∫ u in Ioi (0 : ℝ), u ^ (p - 1) * Real.exp (-(κ * (1 + t)) * u)) =
        (1 / (κ * (1 + t))) ^ p * Real.Gamma p := by
      simpa only [neg_mul] using Real.integral_rpow_mul_exp_neg_mul_Ioi hp
        (mul_pos hκ (by linarith : 0 < 1 + t))
    rw [hg, hrate t ht]
    dsimp [a, kummerWeight]
    ring
  have hbound : ∀ u > 0,
      Real.log (1 + s * (p / (κ * (1 + u)))) = F u - Real.log s - Real.log (1 + u) := by
    intro u hu
    have he : 1 + s * (p / (κ * (1 + u))) = (1 + s * u) / (s * (1 + u)) := by
      dsimp [κ, exactKappa]
      rw [Real.exp_zero, mul_one]
      field_simp
      <;> ring
    rw [he, Real.log_div (by positivity : 1 + s * u ≠ 0)
      (by positivity : s * (1 + u) ≠ 0), Real.log_mul hs.ne' (by positivity : 1 + u ≠ 0)]
    dsimp [F]
    ring
  have hcond : ∀ u > 0, (∫ t in Ioi (0 : ℝ), F t * J u t) <
      C * ((F u - Real.log s - Real.log (1 + u)) * kummerWeight p κ u) := by
    intro u hu
    have hu1 : 0 < 1 + u := by linarith
    have hb : 0 < κ * (1 + u) := mul_pos hκ hu1
    have hau : 0 < a u := by dsimp [a]; positivity
    have hh := mul_lt_mul_of_pos_left (gamma_log_affine_lt p (κ * (1 + u)) s hp hb hs) hau
    have heq : (fun t => F t * J u t) =
        (fun t => a u * (Real.log (1 + s * t) * (t ^ (p - 1) * Real.exp (-(κ * (1 + u)) * t)))) := by
      funext t
      rw [hfactor]
      dsimp [F]
      ring
    rw [heq, integral_const_mul]
    apply lt_of_lt_of_eq hh
    rw [hbound u hu, hrate u hu]
    dsimp [C, a, kummerWeight]
    ring
  have hiz := (L3_kummer_moments p κ hp hκ).2.1
  have hil : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  let upper : ℝ → ℝ := fun u => C * ((F u - Real.log s - Real.log (1 + u)) * kummerWeight p κ u)
  let lower : ℝ → ℝ := fun u => ∫ t in Ioi (0 : ℝ), F t * J u t
  have hilower : IntegrableOn lower (Ioi 0) := by
    have hh := hJ.integral_prod_right
    apply hh.congr
    filter_upwards [] with u
    dsimp [lower]
    apply integral_congr_ae
    filter_upwards [] with t
    rw [hsymm]
  have hupper_eq : upper = fun u => C * (F u * kummerWeight p κ u -
      Real.log s * kummerWeight p κ u - Real.log (1 + u) * kummerWeight p κ u) := by
    funext u
    dsimp [upper]
    ring
  have hiupper : IntegrableOn upper (Ioi 0) := by
    rw [hupper_eq]
    exact ((hF.sub (hiz.const_mul (Real.log s))).sub hil).const_mul C
  have hgap : 0 < ∫ u in Ioi (0 : ℝ), upper u - lower u := by
    rw [setIntegral_pos_iff_support_of_nonneg_ae]
    · have hsub : Ioc (1 : ℝ) 2 ⊆ Function.support (fun u => upper u - lower u) ∩ Ioi 0 := by
        intro u hu
        have hu0 : 0 < u := by linarith [hu.1]
        exact ⟨Function.mem_support.mpr (sub_pos.mpr (hcond u hu0)).ne', hu0⟩
      refine lt_of_lt_of_le ?_ (measure_mono hsub)
      rw [Real.volume_Ioc]
      norm_num
    · filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
      exact (sub_pos.mpr (hcond u hu)).le
    · exact hiupper.sub hilower
  have hlower_val : (∫ u in Ioi (0 : ℝ), lower u) =
      C * (∫ t in Ioi (0 : ℝ), F t * kummerWeight p κ t) := by
    have hh := integral_integral_swap hJ
    have hl : (∫ t, ∫ u, F t * J t u ∂μ ∂μ) =
        C * (∫ t in Ioi (0 : ℝ), F t * kummerWeight p κ t) := by
      simp_rw [integral_const_mul]
      change (∫ t in Ioi (0 : ℝ), F t * (∫ u in Ioi (0 : ℝ), J t u)) = _
      rw [← integral_const_mul]
      apply setIntegral_congr_fun measurableSet_Ioi
      intro t ht
      dsimp only
      rw [hmarg t ht]
      dsimp [C]
      ring
    rw [hl] at hh
    rw [hh]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro u _
    dsimp [lower]
    apply integral_congr_ae
    filter_upwards [] with t
    rw [hsymm]
  have hupper_val : (∫ u in Ioi (0 : ℝ), upper u) =
      C * ((∫ t in Ioi (0 : ℝ), F t * kummerWeight p κ t) - Real.log s * kummerZ p κ -
        (∫ t in Ioi (0 : ℝ), Real.log (1 + t) * kummerWeight p κ t)) := by
    rw [hupper_eq, integral_const_mul,
      integral_sub (hF.sub' (hiz.const_mul (Real.log s))) hil,
      integral_sub hF (hiz.const_mul (Real.log s)), integral_const_mul]
    rfl
  rw [integral_sub hiupper hilower, hupper_val, hlower_val] at hgap
  have hraw : Real.log s * kummerZ p κ +
      (∫ t in Ioi (0 : ℝ), Real.log (1 + t) * kummerWeight p κ t) < 0 := by
    nlinarith
  change Real.log s + kummerLog1p p κ < 0
  dsimp [kummerLog1p, kummerExpect]
  have hm := mul_lt_mul_of_pos_right hraw (inv_pos.mpr hZ)
  field_simp [hZ.ne'] at hm ⊢
  linarith

/-- A continuous strictly increasing growth function with a strict bracket has a
unique global root, and its location satisfies both strict manuscript bounds. -/
/- Coverage correction, 2026-11-22: the strictly positive growth
derivative and both endpoint signs are proved for the actual Kummer integral. -/
theorem exact_root (p s : ℝ) (hp : 0 < p) (hs : 0 < s) (hs1 : s < 1)
    :
    ∃! D : ℝ, exactGrowth p s D = 0 ∧ 0 < D ∧ D < gammaLogLoss p := by
  have hzero := exactGrowth_zero_neg p s hp hs hs1
  let ℓ := gammaLogLoss p
  have hmono := exactGrowth_strictMono p s hp hs hs1
  have hell : 0 < exactGrowth p s ℓ := exactGrowth_gammaLogLoss_pos p s hp hs hs1
  have hℓ : 0 < ℓ := by
    apply hmono.lt_iff_lt.mp
    linarith
  have hcont := (exactGrowth_continuous p s hp hs hs1).continuousOn (s := Set.Icc 0 ℓ)
  obtain ⟨D, hD, hroot⟩ := intermediate_value_Icc hℓ.le hcont ⟨hzero.le, hell.le⟩
  have hD0 : 0 < D := by
    by_contra h
    have he : D = 0 := le_antisymm (le_of_not_gt h) hD.1
    subst D
    linarith
  have hDl : D < gammaLogLoss p := by
    by_contra h
    have he : D = ℓ := le_antisymm hD.2 (le_of_not_gt h)
    subst D
    linarith
  refine ⟨D, ⟨hroot, hD0, hDl⟩, ?_⟩
  intro y hy
  exact hmono.injective (hy.1.trans hroot.symm)

/-- The log restoring-coordinate map covers positive c and is strictly increasing. -/
theorem restoring_coordinate (s : ℝ) (hs : s < 1) :
    (∀ y : ℝ, 0 < (1 - s) * Real.exp y) ∧
    StrictMono (fun y : ℝ => (1 - s) * Real.exp y) := by
  refine ⟨fun y => mul_pos (sub_pos.mpr hs) (Real.exp_pos y), ?_⟩
  intro x y hxy
  exact mul_lt_mul_of_pos_left (Real.exp_lt_exp.mpr hxy) (sub_pos.mpr hs)

#print axioms kummerMean_deriv
#print axioms kummerLog1p_deriv
#print axioms kummerCrossLog1p_deriv
#print axioms kummerLog1p_cov_pos
#print axioms kummerLog1p_weighted_ibp
#print axioms exactGrowth_continuous
#print axioms exactGrowth_strictMono
#print axioms exact_root
#print axioms kummerLogZ_shape_deriv
#print axioms kummerMean_shape_deriv
#print axioms kummer_shape_mixed
#print axioms kummer_log_joint_identity
#print axioms kummer_differentiated_mean_identity
#print axioms kummer_log_strict_jensen
#print axioms exactGrowth_gammaLogLoss_pos
#print axioms exactGrowth_zero_neg
#print axioms kummer_odes

/-- The actual joint-law identity bounds the logarithmic moment below by half
of its gamma conditional mean; no small-parameter expansion is used. -/
theorem kummerLog1p_lower (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    (gammaLogDerivative p - Real.log κ) / 2 ≤ kummerLog1p p κ := by
  have hZ := (L3_kummer_moments p κ hp hκ).1
  have hil := (L3_kummer_moments p κ hp hκ).2.2.2.2.1
  have hiL : IntegrableOn (fun t => Real.log (1 + t) * kummerWeight p κ t) (Ioi 0) := by
    simpa only [pow_zero, one_mul] using log1p_moment_integrable p κ hp hκ 0
  have hcmp : kummerExpect p κ Real.log ≤ kummerLog1p p κ := by
    unfold kummerExpect kummerLog1p
    apply mul_le_mul_of_nonneg_left _ (inv_nonneg.mpr hZ.le)
    apply setIntegral_mono_on hil hiL measurableSet_Ioi
    intro t ht
    exact mul_le_mul_of_nonneg_right
      (Real.log_le_log ht (by linarith : t ≤ 1 + t)) (weight_nonneg p κ t ht)
  have hj := kummer_log_joint_identity p κ hp hκ
  linarith

/-- A concrete square-root upper bound for the actual stay probability. -/
theorem kummerStay_sqrt_bound (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    Real.exp (-kummerLog1p p κ) ≤
      Real.exp (-gammaLogDerivative p / 2) * κ ^ (1 / 2 : ℝ) := by
  calc Real.exp (-kummerLog1p p κ)
      ≤ Real.exp (-(gammaLogDerivative p - Real.log κ) / 2) := by
        apply Real.exp_le_exp.mpr
        linarith [kummerLog1p_lower p κ hp hκ]
    _ = _ := by
      rw [Real.rpow_def_of_pos hκ, ← Real.exp_add]
      congr 1
      ring

/-- The actual stay probability tends to zero at the singular endpoint. -/
theorem kummerStay_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => Real.exp (-kummerLog1p p κ)) (𝓝[>] 0) (𝓝 0) := by
  have hb : Tendsto (fun κ : ℝ => Real.exp (-gammaLogDerivative p / 2) *
      κ ^ (1 / 2 : ℝ)) (𝓝[>] 0) (𝓝 0) := by
    have h := (Real.continuousAt_rpow_const 0 (1 / 2 : ℝ) (Or.inr (by norm_num))).tendsto
    simpa using (h.mono_left nhdsWithin_le_nhds).const_mul (Real.exp (-gammaLogDerivative p / 2))
  refine squeeze_zero' (Eventually.of_forall fun _ => (Real.exp_pos _).le) ?_ hb
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  exact kummerStay_sqrt_bound p κ hp hκ


#print axioms kummerStay_tendsto_zero
/-- The square-root bound also absorbs the two logarithms in the endpoint
rescaling, proving the actual stay contribution is negligible. -/
theorem kummerStay_log_sq_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => Real.exp (-kummerLog1p p κ) /
      (1 / Real.log (1 / κ)) ^ 2) (𝓝[>] 0) (𝓝 0) := by
  have hl := tendsto_log_mul_rpow_nhdsGT_zero (r := (1 / 4 : ℝ)) (by norm_num)
  have hb : Tendsto (fun κ : ℝ => Real.exp (-gammaLogDerivative p / 2) *
      (Real.log κ * κ ^ (1 / 4 : ℝ)) ^ 2) (𝓝[>] 0) (𝓝 0) := by
    simpa using (hl.pow 2).const_mul (Real.exp (-gammaLogDerivative p / 2))
  refine squeeze_zero' (Eventually.of_forall fun κ => div_nonneg (Real.exp_pos _).le (sq_nonneg _)) ?_ hb
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  have hκ0 : 0 < κ := hκ
  have hpow : (κ ^ (1 / 4 : ℝ)) ^ 2 = κ ^ (1 / 2 : ℝ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hκ0.le]
    norm_num
  have hbound := mul_le_mul_of_nonneg_right (kummerStay_sqrt_bound p κ hp hκ0)
    (sq_nonneg (Real.log κ))
  rw [mul_pow, hpow]
  simpa only [one_div, Real.log_inv, inv_pow, div_inv_eq_mul, neg_sq,
    mul_assoc, mul_comm, mul_left_comm] using hbound


#print axioms kummerStay_log_sq_tendsto_zero
private theorem shapeScore_mono : MonotoneOn kummerShapeScore (Ioi 0) := by
  intro x hx y hy hxy
  change 0 < x at hx
  change 0 < y at hy
  have hx1 : 0 < 1 + x := by linarith
  have hy1 : 0 < 1 + y := by linarith
  dsimp [kummerShapeScore]
  rw [← Real.log_div hx.ne' hx1.ne', ← Real.log_div hy.ne' hy1.ne']
  apply Real.log_le_log (div_pos hx hx1)
  exact (div_le_div_iff₀ hx1 hy1).mpr (by nlinarith)

/-- Increasing T and increasing shape score have nonnegative covariance,
so the actual first moment increases with shape. -/
theorem kummerShapeA_nonneg (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    0 ≤ deriv (fun q => κ * kummerMean q κ) p := by
  have h := L3_kummer_moments p κ hp hκ
  have hi0 := h.2.1
  have hi1 := h.2.2.1
  have hil := shapeScore_integrable p κ hp hκ
  have hitl : IntegrableOn (fun t => t * kummerShapeScore t * kummerWeight p κ t) (Ioi 0) := by
    convert (tlog_weight_integrable p κ hp hκ).sub
      (log1p_moment_integrable p κ hp hκ 1) using 1
    funext t
    dsimp only [Pi.sub_apply]
    simp only [pow_one, kummerShapeScore]
    ring
  let m := kummerMean p κ
  have hm : 0 < m := by
    dsimp [m, kummerMean, kummerExpect]
    apply mul_pos (inv_pos.mpr h.1)
    rw [setIntegral_pos_iff_support_of_nonneg_ae]
    · have hsub : Ioc (1 : ℝ) 2 ⊆ Function.support (fun t => t * kummerWeight p κ t) ∩ Ioi 0 := by
        intro t ht
        have ht0 : 0 < t := by linarith [ht.1]
        refine ⟨Function.mem_support.mpr ?_, ht0⟩
        exact ne_of_gt (mul_pos ht0 (by unfold kummerWeight; positivity))
      refine lt_of_lt_of_le ?_ (measure_mono hsub)
      rw [Real.volume_Ioc]
      norm_num
    · filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
      exact mul_nonneg (le_of_lt ht) (weight_nonneg p κ t ht)
    · exact hi1
  let L := kummerShapeScore m
  let f : ℝ → ℝ := fun t => (t - m) * (kummerShapeScore t - L) * kummerWeight p κ t
  have hf_eq : f = fun t => (t * kummerShapeScore t * kummerWeight p κ t -
      m * (kummerShapeScore t * kummerWeight p κ t)) -
      L * (t * kummerWeight p κ t) + (m * L) * kummerWeight p κ t := by
    funext t
    dsimp [f]
    ring
  have hf : 0 ≤ ∫ t in Ioi (0 : ℝ), f t := by
    apply integral_nonneg_of_ae
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    dsimp [f]
    apply mul_nonneg _ (weight_nonneg p κ t ht)
    rcases le_total m t with hmt | htm
    · exact mul_nonneg (sub_nonneg.mpr hmt)
        (sub_nonneg.mpr (shapeScore_mono hm ht hmt))
    · exact mul_nonneg_of_nonpos_of_nonpos (sub_nonpos.mpr htm)
        (sub_nonpos.mpr (shapeScore_mono ht hm htm))
  have heval : (kummerZ p κ)⁻¹ * (∫ t in Ioi (0 : ℝ), f t) =
      kummerExpect p κ (fun t => t * kummerShapeScore t) -
        kummerMean p κ * kummerExpect p κ kummerShapeScore := by
    rw [hf_eq, integral_add ((hitl.sub' (hil.const_mul m)).sub' (hi1.const_mul L))
        (hi0.const_mul (m * L)),
      integral_sub (hitl.sub' (hil.const_mul m)) (hi1.const_mul L),
      integral_sub hitl (hil.const_mul m), integral_const_mul, integral_const_mul,
      integral_const_mul]
    dsimp [m, kummerMean, kummerExpect]
    change (kummerZ p κ)⁻¹ *
      ((_ - (kummerZ p κ)⁻¹ * _ * _) - L * _ +
        ((kummerZ p κ)⁻¹ * _ * L) * kummerZ p κ) = _
    field_simp [h.1.ne']
    <;> ring
  rw [((kummerMean_shape_deriv p κ hp hκ).const_mul κ).deriv, ← heval]
  exact mul_nonneg hκ.le (mul_nonneg (inv_nonneg.mpr h.1.le) hf)

/-- The covariance sign bounds the mean above without an asymptotic expansion. -/
theorem kummerLog1p_upper (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) (hκ1 : κ ≤ 1) :
    kummerLog1p p κ ≤ kummerLog1p p 1 - Real.log κ / 2 := by
  let f := fun x => kummerLog1p p x + Real.log x / 2
  have hf : ∀ x > 0, HasDerivAt f (deriv (kummerLog1p p) x + 1 / x / 2) x := by
    intro x hx
    have hd := (kummerLog1p_deriv p x hp hx).fun_add ((Real.hasDerivAt_log hx.ne').div_const 2)
    simpa only [f, Pi.add_apply, one_div, (kummerLog1p_deriv p x hp hx).deriv] using hd
  have hm : MonotoneOn f (Ioi 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ioi 0)
    · exact fun x hx => (hf x hx).continuousAt.continuousWithinAt
    · intro x hx
      exact (hf x (interior_subset hx)).differentiableAt.differentiableWithinAt
    · intro x hx
      have hx0 : 0 < x := interior_subset hx
      rw [(hf x hx0).deriv]
      have hpos := kummerShapeA_nonneg p x hp hx0
      have hmix := kummer_shape_mixed p x hp hx0
      have hmean := kummer_differentiated_mean_identity p x hp hx0
      have heq : deriv (kummerLog1p p) x + 1 / x / 2 =
          deriv (fun q => x * kummerMean q x) p / (2 * x) := by
        rw [hmix]
        field_simp [hx0.ne'] at hmean ⊢
        nlinarith
      rw [heq]
      exact div_nonneg hpos (by positivity)
  have h := hm hκ (by norm_num : (1 : ℝ) ∈ Ioi 0) hκ1
  dsimp [f] at h
  rw [Real.log_one] at h
  linarith

/-- The actual logarithmic moment has the leading one-half logarithmic limit,
obtained by sandwiching between two constant offsets of −log κ/2. -/
theorem kummerLog1p_log_scale_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerLog1p p κ * (1 / Real.log (1 / κ)))
      (𝓝[>] 0) (𝓝 (1 / 2)) := by
  have hL : Tendsto (fun κ : ℝ => Real.log (1 / κ)) (𝓝[>] 0) atTop := by
    simpa only [one_div, Function.comp_def] using Real.tendsto_log_atTop.comp tendsto_inv_nhdsGT_zero
  have hi : Tendsto (fun κ : ℝ => 1 / Real.log (1 / κ)) (𝓝[>] 0) (𝓝 0) :=
    tendsto_const_nhds.div_atTop hL
  have hlo : Tendsto (fun κ : ℝ => (1 / 2 : ℝ) +
      (gammaLogDerivative p / 2) * (1 / Real.log (1 / κ))) (𝓝[>] 0) (𝓝 (1 / 2)) := by
    simpa using tendsto_const_nhds.add (hi.const_mul (gammaLogDerivative p / 2))
  have hhi : Tendsto (fun κ : ℝ => (1 / 2 : ℝ) +
      kummerLog1p p 1 * (1 / Real.log (1 / κ))) (𝓝[>] 0) (𝓝 (1 / 2)) := by
    simpa using tendsto_const_nhds.add (hi.const_mul (kummerLog1p p 1))
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' hlo hhi
  · filter_upwards [self_mem_nhdsWithin,
      mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
    have hlog : 0 < Real.log (1 / κ) := Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
    have heq : (gammaLogDerivative p - Real.log κ) / 2 * (1 / Real.log (1 / κ)) =
        1 / 2 + gammaLogDerivative p / 2 * (1 / Real.log (1 / κ)) := by
      rw [one_div κ, Real.log_inv] at *
      field_simp [show Real.log κ ≠ 0 by linarith]
      <;> ring
    rw [← heq]
    exact mul_le_mul_of_nonneg_right (kummerLog1p_lower p κ hp hκ) (by positivity)
  · filter_upwards [self_mem_nhdsWithin,
      mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
    have hlog : 0 < Real.log (1 / κ) := Real.log_pos ((lt_div_iff₀ hκ).mpr (by simpa using hκ1))
    have heq : (kummerLog1p p 1 - Real.log κ / 2) * (1 / Real.log (1 / κ)) =
        1 / 2 + kummerLog1p p 1 * (1 / Real.log (1 / κ)) := by
      rw [one_div κ, Real.log_inv] at *
      field_simp [show Real.log κ ≠ 0 by linarith]
      <;> ring
    rw [← heq]
    exact mul_le_mul_of_nonneg_right (kummerLog1p_upper p κ hp hκ hκ1.le) (by positivity)


#print axioms kummerLog1p_log_scale_tendsto
private theorem t_mul_weight (p κ t : ℝ) (ht : 0 < t) :
    t * kummerWeight p κ t = (t / (1 + t)) ^ p * Real.exp (-κ * t) := by
  have ht1 : 0 < 1 + t := by linarith
  dsimp [kummerWeight]
  rw [Real.div_rpow ht.le ht1.le, Real.rpow_neg ht1.le, Real.rpow_sub ht, Real.rpow_one]
  field_simp

/-- The partition integral grows at least logarithmically at small κ. -/
theorem kummerZ_log_lower (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) (hκ1 : κ ≤ 1) :
    ((1 / 2 : ℝ) ^ p * Real.exp (-1)) * Real.log (1 / κ) ≤ kummerZ p κ := by
  let C := (1 / 2 : ℝ) ^ p * Real.exp (-1)
  have hC : 0 < C := by dsimp [C]; positivity
  have hr : 1 ≤ 1 / κ := (le_div_iff₀ hκ).mpr (by simpa using hκ1)
  have hint : IntervalIntegrable (fun t : ℝ => 1 / t) volume 1 (1 / κ) := by
    apply ContinuousOn.intervalIntegrable
    apply continuousOn_const.div continuousOn_id
    intro x hx
    rw [uIcc_of_le hr] at hx
    change x ≠ 0
    linarith [hx.1]
  have hbound : ∀ t ∈ Ioc 1 (1 / κ), C * (1 / t) ≤ kummerWeight p κ t := by
    intro t ht
    have ht0 : 0 < t := by linarith [ht.1]
    have ht1 : 0 < 1 + t := by linarith
    have hratio : (1 / 2 : ℝ) ≤ t / (1 + t) := by
      apply (le_div_iff₀ ht1).mpr
      linarith [ht.1]
    have hexp : Real.exp (-1) ≤ Real.exp (-κ * t) := by
      apply Real.exp_le_exp.mpr
      have htκ := (le_div_iff₀ hκ).mp ht.2
      nlinarith
    have hmul : C ≤ t * kummerWeight p κ t := by
      rw [t_mul_weight p κ t ht0]
      exact mul_le_mul (Real.rpow_le_rpow (by norm_num) hratio hp.le) hexp
        (Real.exp_pos _).le (Real.rpow_nonneg (by positivity) _)
    calc C * (1 / t) = C / t := by ring
      _ ≤ kummerWeight p κ t := (div_le_iff₀ ht0).mpr (by nlinarith)
  have hi := (L3_kummer_moments p κ hp hκ).2.1
  have hsub : Ioc (1 : ℝ) (1 / κ) ⊆ Ioi 0 := by intro t ht; change 0 < t; linarith [ht.1]
  have hlo := setIntegral_mono_on (hint.1.const_mul C) (hi.mono_set hsub)
    measurableSet_Ioc hbound
  have hupper : (∫ t in Ioc 1 (1 / κ), kummerWeight p κ t) ≤ kummerZ p κ := by
    apply setIntegral_mono_set hi
    · filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
      exact weight_nonneg p κ t ht
    · exact hsub.eventuallySubset
  have heval : (∫ t in Ioc 1 (1 / κ), C * (1 / t)) = C * Real.log (1 / κ) := by
    rw [integral_const_mul, ← intervalIntegral.integral_of_le hr,
      integral_one_div_of_pos (by norm_num) (by positivity), div_one]
  rw [heval] at hlo
  exact hlo.trans hupper

/-- The actual partition integral diverges at small κ. -/
theorem kummerZ_tendsto_atTop (p : ℝ) (hp : 0 < p) :
    Tendsto (kummerZ p) (𝓝[>] 0) atTop := by
  have hL : Tendsto (fun κ : ℝ => Real.log (1 / κ)) (𝓝[>] 0) atTop := by
    simpa only [one_div, Function.comp_def] using Real.tendsto_log_atTop.comp tendsto_inv_nhdsGT_zero
  have hC : 0 < (1 / 2 : ℝ) ^ p * Real.exp (-1) := by positivity
  apply tendsto_atTop_mono' _ _ (hL.const_mul_atTop hC)
  filter_upwards [self_mem_nhdsWithin,
    mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
  exact kummerZ_log_lower p κ hp hκ hκ1.le

/-- The scaled first moment is bounded by the reciprocal partition integral. -/
theorem kummerA_le_inv_Z (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    κ * kummerMean p κ ≤ (kummerZ p κ)⁻¹ := by
  have h := L3_kummer_moments p κ hp hκ
  have hcmp : (∫ t in Ioi (0 : ℝ), t * kummerWeight p κ t) ≤ 1 / κ := by
    have he : IntegrableOn (fun t : ℝ => Real.exp (-κ * t)) (Ioi 0) :=
      integrableOn_exp_mul_Ioi (by linarith : -κ < 0) 0
    have hb := setIntegral_mono_on h.2.2.1 he measurableSet_Ioi (fun t ht => ?_)
    · have hev : (∫ t in Ioi (0 : ℝ), Real.exp (-κ * t)) = 1 / κ := by
        rw [integral_exp_mul_Ioi (by linarith : -κ < 0) 0]
        simp
      rwa [hev] at hb
    · change 0 < t at ht
      rw [t_mul_weight p κ t ht]
      have hr : t / (1 + t) ≤ 1 := (div_le_one (by linarith : 0 < 1 + t)).mpr (by linarith)
      have hpow : (t / (1 + t)) ^ p ≤ 1 :=
        Real.rpow_le_one (by positivity) hr hp.le
      simpa using mul_le_mul_of_nonneg_right hpow (Real.exp_pos (-κ * t)).le
  dsimp [kummerMean, kummerExpect]
  have hc := mul_le_mul_of_nonneg_left hcmp (mul_nonneg hκ.le (inv_nonneg.mpr h.1.le))
  calc κ * ((kummerZ p κ)⁻¹ * _) = (κ * (kummerZ p κ)⁻¹) * _ := by ring
    _ ≤ (κ * (kummerZ p κ)⁻¹) * (1 / κ) := hc
    _ = _ := by field_simp

/-- A direct bound for the actual shape derivative at the singular endpoint. -/
theorem kummerShapeA_le_inv_Z (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) (hκ1 : κ ≤ 1) :
    deriv (fun q => κ * kummerMean q κ) p ≤
      (2 * kummerLog1p p 1 - gammaLogDerivative p) * (kummerZ p κ)⁻¹ := by
  have h := L3_kummer_moments p κ hp hκ
  have hscore : ∀ t > 0, kummerShapeScore t ≤ 0 := by
    intro t ht
    dsimp [kummerShapeScore]
    exact sub_nonpos.mpr (Real.log_le_log ht (by linarith))
  have hts : kummerExpect p κ (fun t => t * kummerShapeScore t) ≤ 0 := by
    dsimp [kummerExpect]
    apply mul_nonpos_of_nonneg_of_nonpos (inv_nonneg.mpr h.1.le)
    apply integral_nonpos_of_ae
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    exact mul_nonpos_of_nonpos_of_nonneg
      (mul_nonpos_of_nonneg_of_nonpos (le_of_lt ht) (hscore t ht)) (weight_nonneg p κ t ht)
  have hmean : 0 ≤ kummerMean p κ := by
    dsimp [kummerMean, kummerExpect]
    apply mul_nonneg (inv_nonneg.mpr h.1.le)
    apply integral_nonneg_of_ae
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    exact mul_nonneg (le_of_lt ht) (weight_nonneg p κ t ht)
  have hs : -kummerExpect p κ kummerShapeScore ≤
      2 * kummerLog1p p 1 - gammaLogDerivative p := by
    rw [expect_shapeScore p κ hp hκ]
    have hj := kummer_log_joint_identity p κ hp hκ
    have hm := kummerLog1p_upper p κ hp hκ hκ1
    linarith
  have hC : 0 ≤ 2 * kummerLog1p p 1 - gammaLogDerivative p := by
    have hl := kummerLog1p_lower p 1 hp (by norm_num)
    rw [Real.log_one] at hl
    linarith
  rw [((kummerMean_shape_deriv p κ hp hκ).const_mul κ).deriv]
  calc κ * (kummerExpect p κ (fun t => t * kummerShapeScore t) -
      kummerMean p κ * kummerExpect p κ kummerShapeScore)
      ≤ (κ * kummerMean p κ) * (-kummerExpect p κ kummerShapeScore) := by nlinarith
    _ ≤ (κ * kummerMean p κ) * (2 * kummerLog1p p 1 - gammaLogDerivative p) :=
      mul_le_mul_of_nonneg_left hs (mul_nonneg hκ.le hmean)
    _ ≤ (kummerZ p κ)⁻¹ * (2 * kummerLog1p p 1 - gammaLogDerivative p) :=
      mul_le_mul_of_nonneg_right (kummerA_le_inv_Z p κ hp hκ) hC
    _ = _ := by ring

/-- The actual shape derivative tends to zero, independently of its sharper
trigamma asymptotic. -/
theorem kummerShapeA_tendsto_zero (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => deriv (fun q => κ * kummerMean q κ) p) (𝓝[>] 0) (𝓝 0) := by
  have hi : Tendsto (fun κ => (kummerZ p κ)⁻¹) (𝓝[>] 0) (𝓝 0) :=
    tendsto_inv_atTop_zero.comp (kummerZ_tendsto_atTop p hp)
  have hb := hi.const_mul (2 * kummerLog1p p 1 - gammaLogDerivative p)
  simp only [mul_zero] at hb
  apply squeeze_zero' _ _ hb
  · filter_upwards [self_mem_nhdsWithin] with κ hκ
    exact kummerShapeA_nonneg p κ hp hκ
  · filter_upwards [self_mem_nhdsWithin,
      mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (show (0 : ℝ) < 1 by norm_num))] with κ hκ hκ1
    exact kummerShapeA_le_inv_Z p κ hp hκ hκ1.le


#print axioms kummerShapeA_tendsto_zero
/-- The unnormalized scaled Kummer first moment. -/
noncomputable def kummerN0 (p κ : ℝ) : ℝ :=
  κ * ∫ t in Ioi (0 : ℝ), t * kummerWeight p κ t

private theorem kummerN0_rescale (p κ : ℝ) (hκ : 0 < κ) :
    kummerN0 p κ = ∫ u in Ioi (0 : ℝ), (u / (u + κ)) ^ p * Real.exp (-u) := by
  let f := fun u : ℝ => (u / (u + κ)) ^ p * Real.exp (-u)
  have hc := integral_comp_mul_left_Ioi f 0 hκ
  simp only [mul_zero, smul_eq_mul] at hc
  have heq : (∫ t in Ioi (0 : ℝ), f (κ * t)) =
      ∫ t in Ioi (0 : ℝ), t * kummerWeight p κ t := by
    apply setIntegral_congr_fun measurableSet_Ioi
    intro t ht
    dsimp [f]
    rw [t_mul_weight p κ t ht]
    have hr : κ * t / (κ * t + κ) = t / (1 + t) := by field_simp; ring
    rw [hr]
    ring
  rw [heq] at hc
  dsimp [kummerN0]
  rw [hc]
  field_simp [hκ.ne']
  rfl

/-- Rescaled Gamma domination gives the first-moment numerator limit. -/
theorem kummerN0_tendsto_one (p : ℝ) (hp : 0 < p) :
    Tendsto (kummerN0 p) (𝓝[>] 0) (𝓝 1) := by
  let F := fun κ u : ℝ => (u / (u + κ)) ^ p * Real.exp (-u)
  have hi := integrableOn_exp_neg_Ioi (0 : ℝ)
  have hm : ∀ᶠ κ in 𝓝[>] (0 : ℝ), AEStronglyMeasurable (F κ) (volume.restrict (Ioi 0)) := by
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    apply ContinuousOn.aestronglyMeasurable _ measurableSet_Ioi
    intro u hu
    change 0 < κ at hκ
    change 0 < u at hu
    apply ContinuousAt.continuousWithinAt
    dsimp [F]
    fun_prop (disch := positivity)
  have hb : ∀ᶠ κ in 𝓝[>] (0 : ℝ), ∀ᵐ u ∂volume.restrict (Ioi 0), ‖F κ u‖ ≤ Real.exp (-u) := by
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
    change 0 < κ at hκ
    change 0 < u at hu
    dsimp [F]
    rw [abs_of_nonneg (by positivity)]
    have hr : u / (u + κ) ≤ 1 := (div_le_one (by positivity)).mpr (by linarith)
    have hpow := Real.rpow_le_one (by positivity : 0 ≤ u / (u + κ)) hr hp.le
    simpa using mul_le_mul_of_nonneg_right hpow (Real.exp_pos (-u)).le
  have hl : ∀ᵐ u ∂volume.restrict (Ioi 0),
      Tendsto (fun κ => F κ u) (𝓝[>] 0) (𝓝 (Real.exp (-u))) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
    have hu0 : 0 < u := hu
    have hc : ContinuousAt (fun κ => F κ u) 0 := by
      dsimp [F]
      fun_prop (disch := positivity)
    simpa [F, hu0.ne'] using hc.tendsto.mono_left nhdsWithin_le_nhds
  have h := tendsto_integral_filter_of_dominated_convergence (fun u => Real.exp (-u)) hm hb hi hl
  rw [integral_exp_neg_Ioi_zero] at h
  apply h.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  exact (kummerN0_rescale p κ hκ).symm

/-- The actual trigamma coefficient in the manuscript, as its positive series. -/
noncomputable def gammaTrigamma (p : ℝ) : ℝ := ∑' n : ℕ, 1 / (p + n) ^ 2

private noncomputable def trigammaKernel (p y : ℝ) : ℝ :=
  y * Real.exp (-p * y) / (1 - Real.exp (-y))

private theorem trigammaKernel_integrable (p : ℝ) (hp : 0 < p) :
    IntegrableOn (trigammaKernel p) (Ioi 0) := by
  have h0 : IntegrableOn (fun y : ℝ => Real.exp (-p * y)) (Ioi 0) :=
    integrableOn_exp_mul_Ioi (by linarith : -p < 0) 0
  have h1 : IntegrableOn (fun y : ℝ => y * Real.exp (-p * y)) (Ioi 0) := by
    simpa only [sub_self, Real.rpow_zero, one_mul] using gamma_first_integrable 1 p (by norm_num) hp
  have hc : ContinuousOn (trigammaKernel p) (Ioi 0) := by
    intro y hy
    have hy0 : 0 < y := hy
    have he : 0 < 1 - Real.exp (-y) := sub_pos.mpr (Real.exp_lt_one_iff.mpr (by linarith))
    apply ContinuousAt.continuousWithinAt
    change ContinuousAt (fun z : ℝ => z * Real.exp (-p * z) / (1 - Real.exp (-z))) y
    fun_prop (disch := positivity)
  refine (h0.add h1).mono' (hc.aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with y hy
  have hy0 : 0 < y := hy
  have he : 0 < 1 - Real.exp (-y) := sub_pos.mpr (Real.exp_lt_one_iff.mpr (by linarith))
  have hxy : (1 + y) * Real.exp (-y) ≤ 1 := by
    have hh := mul_le_mul_of_nonneg_right (Real.add_one_le_exp y) (Real.exp_pos (-y)).le
    rw [← Real.exp_add] at hh
    simp only [add_neg_cancel, Real.exp_zero] at hh
    nlinarith
  have hratio : y / (1 - Real.exp (-y)) ≤ 1 + y := (div_le_iff₀ he).mpr (by nlinarith)
  dsimp [trigammaKernel, Pi.add_apply]
  rw [abs_of_nonneg (by positivity)]
  have hh := mul_le_mul_of_nonneg_right hratio (Real.exp_pos (-p * y)).le
  convert hh using 1 <;> ring

private theorem trigammaKernel_series (p y : ℝ) (hy : 0 < y) :
    HasSum (fun n : ℕ => y * Real.exp (-(p + n) * y)) (trigammaKernel p y) := by
  have he : |Real.exp (-y)| < 1 := by rw [abs_of_pos (Real.exp_pos _)]; exact Real.exp_lt_one_iff.mpr (by linarith)
  have h := (hasSum_geometric_of_abs_lt_one he).mul_left (y * Real.exp (-p * y))
  convert h using 1
  · funext n
    rw [← Real.exp_nat_mul, mul_assoc, ← Real.exp_add]
    congr 2
    push_cast
    ring
  · dsimp [trigammaKernel]
    ring

private theorem gamma_first_simple (b : ℝ) (hb : 0 < b) :
    (∫ y in Ioi (0 : ℝ), y * Real.exp (-b * y)) = 1 / b ^ 2 := by
  have h := gamma_first_integral 1 b (by norm_num) hb
  simpa [pow_two, mul_inv_rev] using h

/-- The positive kernel integral is exactly the manuscript's trigamma series. -/
theorem integral_trigammaKernel (p : ℝ) (hp : 0 < p) :
    (∫ y in Ioi (0 : ℝ), trigammaKernel p y) = gammaTrigamma p := by
  let f := fun (n : ℕ) (y : ℝ) => y * Real.exp (-(p + n) * y)
  have hfi : ∀ n, IntegrableOn (f n) (Ioi 0) := by
    intro n
    simpa only [f, sub_self, Real.rpow_zero, one_mul] using
      gamma_first_integrable 1 (p + n) (by norm_num) (by positivity)
  have hsum : Summable (fun n : ℕ => 1 / (p + n) ^ 2) := by
    have h := (Real.summable_one_div_nat_add_rpow p 2).mpr (by norm_num)
    apply h.congr
    intro n
    rw [abs_of_pos (by positivity : 0 < (n : ℝ) + p), Real.rpow_two]
    congr 2
    ring
  have hin : ∀ n : ℕ, (∫ y in Ioi (0 : ℝ), ‖f n y‖) = 1 / (p + n)^2 := by
    intro n
    calc _ = ∫ y in Ioi (0 : ℝ), f n y := by
           apply setIntegral_congr_fun measurableSet_Ioi
           intro y hy
           exact Real.norm_of_nonneg (mul_nonneg (le_of_lt hy) (Real.exp_pos _).le)
      _ = _ := gamma_first_simple (p + n) (by positivity)
  have his : Summable (fun n : ℕ => ∫ y in Ioi (0 : ℝ), ‖f n y‖) := by simp_rw [hin]; exact hsum
  have h := integral_tsum_of_summable_integral_norm hfi his
  have hts : (∫ y in Ioi (0 : ℝ), ∑' n, f n y) = ∫ y in Ioi (0 : ℝ), trigammaKernel p y := by
    apply setIntegral_congr_fun measurableSet_Ioi
    intro y hy
    exact (trigammaKernel_series p y hy).tsum_eq
  rw [hts] at h
  rw [← h]
  apply tsum_congr
  intro n
  exact gamma_first_simple (p + n) (by positivity)

private noncomputable def scoreInverse (y : ℝ) : ℝ := (Real.exp y - 1)⁻¹

private theorem scoreInverse_pos (y : ℝ) (hy : 0 < y) : 0 < scoreInverse y := by
  dsimp [scoreInverse]
  exact inv_pos.mpr (sub_pos.mpr (Real.one_lt_exp_iff.mpr hy))

private theorem scoreInverse_image : scoreInverse '' Ioi 0 = Ioi 0 := by
  ext t
  constructor
  · rintro ⟨y, hy, rfl⟩
    exact scoreInverse_pos y hy
  · intro ht
    have ht0 : 0 < t := ht
    refine ⟨Real.log (1 + t⁻¹), Real.log_pos (by linarith [inv_pos.mpr ht0] : 1 < 1 + t⁻¹), ?_⟩
    dsimp [scoreInverse]
    rw [Real.exp_log (by positivity : 0 < 1 + t⁻¹)]
    simp

private theorem scoreInverse_inj : InjOn scoreInverse (Ioi 0) := by
  intro x _ y _ h
  apply Real.exp_injective
  dsimp [scoreInverse] at h
  have hh := inv_injective h
  linarith

private theorem scoreInverse_deriv (y : ℝ) (hy : 0 < y) :
    HasDerivAt scoreInverse (-Real.exp y / (Real.exp y - 1)^2) y := by
  exact ((Real.hasDerivAt_exp y).sub_const 1).inv
    (ne_of_gt (sub_pos.mpr (Real.one_lt_exp_iff.mpr hy)))

private theorem scoreInverse_ratio (y : ℝ) (hy : 0 < y) :
    scoreInverse y / (1 + scoreInverse y) = Real.exp (-y) := by
  have he : Real.exp y - 1 ≠ 0 := ne_of_gt (sub_pos.mpr (Real.one_lt_exp_iff.mpr hy))
  dsimp [scoreInverse]
  rw [Real.exp_neg]
  field_simp
  <;> ring

private theorem scoreInverse_score (y : ℝ) (hy : 0 < y) :
    kummerShapeScore (scoreInverse y) = -y := by
  have ht := scoreInverse_pos y hy
  dsimp [kummerShapeScore]
  rw [← Real.log_div ht.ne' (by linarith : 1 + scoreInverse y ≠ 0),
    scoreInverse_ratio y hy, Real.log_exp]

private theorem scoreInverse_jacobian (p y : ℝ) (hy : 0 < y) :
    |(-Real.exp y / (Real.exp y - 1)^2)| *
      (-kummerShapeScore (scoreInverse y) * kummerWeight p 0 (scoreInverse y)) =
      trigammaKernel p y := by
  have he : 0 < Real.exp y - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr hy)
  have ht := scoreInverse_pos y hy
  have hw := t_mul_weight p 0 (scoreInverse y) ht
  rw [scoreInverse_ratio y hy, Real.rpow_def_of_pos (Real.exp_pos _), Real.log_exp] at hw
  simp only [neg_zero, zero_mul, Real.exp_zero, mul_one] at hw
  have heq : -(y * p) = -p * y := by ring
  have hex : -y * p = -p * y := by ring
  rw [hex] at hw
  rw [abs_of_neg (div_neg_of_neg_of_pos (neg_lt_zero.mpr (Real.exp_pos y)) (sq_pos_of_pos he)),
    scoreInverse_score y hy]
  dsimp [trigammaKernel]
  rw [Real.exp_neg]
  dsimp [scoreInverse] at hw
  have hw' : kummerWeight p 0 (scoreInverse y) = (Real.exp y - 1) * Real.exp (-p * y) := by
    field_simp [he.ne'] at hw
    simpa only [scoreInverse, one_div, neg_mul] using hw
  rw [hw']
  field_simp
  <;> ring

theorem scoreZero_integrable (p : ℝ) (hp : 0 < p) :
    IntegrableOn (fun t => -kummerShapeScore t * kummerWeight p 0 t) (Ioi 0) := by
  have h := integrableOn_image_iff_integrableOn_abs_deriv_smul measurableSet_Ioi
    (fun y hy => (scoreInverse_deriv y hy).hasDerivWithinAt) scoreInverse_inj
    (fun t => -kummerShapeScore t * kummerWeight p 0 t)
  rw [scoreInverse_image] at h
  apply h.mpr
  refine (trigammaKernel_integrable p hp).congr_fun (fun y hy => ?_) measurableSet_Ioi
  simpa only [smul_eq_mul] using (scoreInverse_jacobian p y hy).symm

/-- The zero-κ shape numerator is evaluated by a genuine change of variables
and the positive trigamma series, not supplied as a special-function identity. -/
theorem scoreZero_integral (p : ℝ) (hp : 0 < p) :
    (∫ t in Ioi (0 : ℝ), -kummerShapeScore t * kummerWeight p 0 t) = gammaTrigamma p := by
  have h := integral_image_eq_integral_abs_deriv_smul measurableSet_Ioi
    (fun y hy => (scoreInverse_deriv y hy).hasDerivWithinAt) scoreInverse_inj
    (fun t => -kummerShapeScore t * kummerWeight p 0 t)
  rw [scoreInverse_image] at h
  rw [h, ← integral_trigammaKernel p hp]
  apply setIntegral_congr_fun measurableSet_Ioi
  intro y hy
  simpa only [smul_eq_mul] using scoreInverse_jacobian p y hy

/-- Dominated convergence of the actual unnormalized shape numerator. -/
theorem kummerScoreNumerator_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => ∫ t in Ioi (0 : ℝ),
      -kummerShapeScore t * kummerWeight p κ t) (𝓝[>] 0) (𝓝 (gammaTrigamma p)) := by
  let F := fun κ t : ℝ => -kummerShapeScore t * kummerWeight p κ t
  have hm : ∀ᶠ κ in 𝓝[>] (0 : ℝ), AEStronglyMeasurable (F κ) (volume.restrict (Ioi 0)) := by
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    exact ((shapeScore_cont.neg).mul (weight_cont p κ)).aestronglyMeasurable measurableSet_Ioi
  have hb : ∀ᶠ κ in 𝓝[>] (0 : ℝ), ∀ᵐ t ∂volume.restrict (Ioi 0),
      ‖F κ t‖ ≤ -kummerShapeScore t * kummerWeight p 0 t := by
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have ht0 : 0 < t := ht
    have hs : 0 ≤ -kummerShapeScore t := by
      dsimp [kummerShapeScore]
      have hh := Real.log_le_log ht0 (by linarith : t ≤ 1 + t)
      linarith
    dsimp [F]
    rw [abs_of_nonneg (mul_nonneg hs (weight_nonneg p κ t ht))]
    exact mul_le_mul_of_nonneg_left (weight_antitone p 0 κ t ht hκ.le) hs
  have hl : ∀ᵐ t ∂volume.restrict (Ioi 0),
      Tendsto (fun κ => F κ t) (𝓝[>] 0) (𝓝 (F 0 t)) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have hc : Continuous (fun κ => F κ t) := by
      dsimp [F, kummerWeight]
      fun_prop
    exact hc.continuousAt.tendsto.mono_left nhdsWithin_le_nhds
  have h := tendsto_integral_filter_of_dominated_convergence
    (fun t => -kummerShapeScore t * kummerWeight p 0 t) hm hb (scoreZero_integrable p hp) hl
  change Tendsto _ _ (𝓝 (∫ t in Ioi (0 : ℝ), -kummerShapeScore t * kummerWeight p 0 t)) at h
  rwa [scoreZero_integral p hp] at h

/- The elementary derivative-ratio lemma was independently supplied by the captain. -/
private theorem endpoint_derivative_limit_ratio (f : ℝ → ℝ) (r : ℝ)
    (hf : Differentiable ℝ f) (hd : Tendsto (deriv f) atTop (𝓝 r)) :
    Tendsto (fun x => f x/x) atTop (𝓝 r) := by
  rw [Metric.tendsto_atTop] at hd ⊢
  intro ε hε
  obtain ⟨A,hA⟩ := hd (ε/2) (by positivity)
  let K := max (1 : ℝ) A
  let g := fun x => f x-r*x
  have hK : 0 < K := lt_of_lt_of_le zero_lt_one (le_max_left _ _)
  have hgd (x : ℝ) : HasDerivAt g (deriv f x-r) x := by
    convert (hf x).hasDerivAt.sub ((hasDerivAt_id x).const_mul r) using 1
    · funext y; rfl
    · simp
  refine ⟨max K (4*|g K|/ε),?_⟩
  intro x hx
  have hKx : K ≤ x := (le_max_left _ _).trans hx
  have hx0 : 0 < x := hK.trans_le hKx
  have hscale : 4*|g K| ≤ x*ε := (div_le_iff₀ hε).mp ((le_max_right _ _).trans hx)
  have hb : ∀ t ∈ Icc K x, ‖deriv g t‖ ≤ ε/2 := by
    intro t ht
    rw [(hgd t).deriv, Real.norm_eq_abs]
    have hh := hA t ((le_max_right _ _).trans ht.1)
    simpa [Real.dist_eq] using hh.le
  have hm := Convex.norm_image_sub_le_of_norm_deriv_le
    (s := Icc K x) (f := g) (fun t _ => (hgd t).differentiableAt) hb
    (convex_Icc K x) (left_mem_Icc.mpr hKx) (right_mem_Icc.mpr hKx)
  simp only [Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr hKx)] at hm
  have htri : |g x| ≤ |g x-g K|+|g K| := by
    simpa using abs_add_le (g x-g K) (g K)
  have hfinal : |g x|/x ≤ 3*ε/4 := by
    rw [div_le_iff₀ hx0]
    nlinarith
  have he : f x/x-r = g x/x := by dsimp [g]; field_simp
  rw [Real.dist_eq, he, abs_div, abs_of_pos hx0]
  exact hfinal.trans_lt (by linarith)

theorem kummerZ_deriv_actual (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    HasDerivAt (kummerZ p) (-(∫ t in Ioi (0 : ℝ), t * kummerWeight p κ t)) κ := by
  have h := L3_kummer_moments p κ hp hκ
  have hhalf := L3_kummer_moments p (κ / 2) hp (by linarith)
  have hd := raw_moment_deriv p κ hκ (fun _ => 1) continuousOn_const
    (by simpa using h.2.1) (by simpa using hhalf.2.2.1)
  change HasDerivAt (fun x => ∫ t in Ioi (0 : ℝ), kummerWeight p x t) _ κ
  simpa only [one_mul, integral_neg] using hd

/-- The partition function has exactly unit leading logarithmic coefficient. -/
theorem kummerZ_log_ratio_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => kummerZ p κ / Real.log (1 / κ)) (𝓝[>] 0) (𝓝 1) := by
  let f := fun H : ℝ => kummerZ p (Real.exp (-H))
  have hf : ∀ H, HasDerivAt f (kummerN0 p (Real.exp (-H))) H := by
    intro H
    have hz := kummerZ_deriv_actual p (Real.exp (-H)) hp (Real.exp_pos _)
    have he := ((hasDerivAt_id H).neg).exp
    convert hz.comp H he using 1
    · rfl
    · dsimp [kummerN0]
      ring
  have hk : Tendsto (fun H : ℝ => Real.exp (-H)) atTop (𝓝[>] 0) :=
    Real.tendsto_exp_atBot_nhdsGT.comp tendsto_neg_atTop_atBot
  have hd : Tendsto (deriv f) atTop (𝓝 1) := by
    have h := (kummerN0_tendsto_one p hp).comp hk
    apply h.congr
    intro H
    exact (hf H).deriv.symm
  have hratio := endpoint_derivative_limit_ratio f 1 (fun H => (hf H).differentiableAt) hd
  have hL : Tendsto (fun κ : ℝ => Real.log (1 / κ)) (𝓝[>] 0) atTop := by
    simpa only [one_div, Function.comp_def] using Real.tendsto_log_atTop.comp tendsto_inv_nhdsGT_zero
  have hh := hratio.comp hL
  apply hh.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  dsimp [f, Function.comp_def]
  rw [one_div κ, Real.log_inv, neg_neg, Real.exp_log hκ]

theorem t_shapeScore_bounds (t : ℝ) (ht : 0 < t) :
    -1 ≤ t * kummerShapeScore t ∧ t * kummerShapeScore t ≤ 0 := by
  have ht1 : 0 < 1 + t := by linarith
  have hl := Real.log_le_sub_one_of_pos (div_pos ht1 ht)
  rw [Real.log_div ht1.ne' ht.ne'] at hl
  have hl' := mul_le_mul_of_nonneg_left hl ht.le
  have heq : t * ((1 + t) / t - 1) = 1 := by field_simp; ring
  rw [heq] at hl'
  have hs := Real.log_le_log ht (by linarith : t ≤ 1 + t)
  dsimp [kummerShapeScore]
  constructor
  · nlinarith
  · exact mul_nonpos_of_nonneg_of_nonpos ht.le (sub_nonpos.mpr hs)

private theorem expect_t_shapeScore_bounds (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    -1 ≤ kummerExpect p κ (fun t => t * kummerShapeScore t) ∧
      kummerExpect p κ (fun t => t * kummerShapeScore t) ≤ 0 := by
  have h := L3_kummer_moments p κ hp hκ
  have hitl : IntegrableOn (fun t => t * kummerShapeScore t * kummerWeight p κ t) (Ioi 0) := by
    convert (tlog_weight_integrable p κ hp hκ).sub
      (log1p_moment_integrable p κ hp hκ 1) using 1
    funext t
    dsimp only [Pi.sub_apply]
    simp only [pow_one, kummerShapeScore]
    ring
  have hlo : -(kummerZ p κ) ≤ ∫ t in Ioi (0 : ℝ), t * kummerShapeScore t * kummerWeight p κ t := by
    calc -(kummerZ p κ) = ∫ t in Ioi (0 : ℝ), -kummerWeight p κ t := by rw [integral_neg]; rfl
      _ ≤ _ := by
        apply setIntegral_mono_on h.2.1.neg hitl measurableSet_Ioi
        intro t ht
        have hh := mul_le_mul_of_nonneg_right (t_shapeScore_bounds t ht).1 (weight_nonneg p κ t ht)
        simpa using hh
  constructor
  · dsimp [kummerExpect]
    calc -1 = (kummerZ p κ)⁻¹ * (-kummerZ p κ) := by field_simp [h.1.ne']
      _ ≤ _ := mul_le_mul_of_nonneg_left hlo (inv_nonneg.mpr h.1.le)
  · dsimp [kummerExpect]
    apply mul_nonpos_of_nonneg_of_nonpos (inv_nonneg.mpr h.1.le)
    apply integral_nonpos_of_ae
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    exact mul_nonpos_of_nonpos_of_nonneg (t_shapeScore_bounds t ht).2 (weight_nonneg p κ t ht)

private theorem kappa_log_sq_tendsto_zero :
    Tendsto (fun κ : ℝ => κ * (Real.log (1 / κ))^2) (𝓝[>] 0) (𝓝 0) := by
  have h := (tendsto_log_mul_rpow_nhdsGT_zero (r := (1 / 2 : ℝ)) (by norm_num)).pow 2
  simp only [zero_pow (by norm_num : (2 : ℕ) ≠ 0)] at h
  apply h.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  have hpow : (κ ^ (1 / 2 : ℝ)) ^ 2 = κ := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hκ.le]
    norm_num
  rw [mul_pow, hpow, one_div κ, Real.log_inv, neg_sq]
  ring

/-- The full actual shape asymptotic with the manuscript's trigamma coefficient. -/
theorem kummerShapeA_log_sq_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun κ => deriv (fun q => κ * kummerMean q κ) p /
      (1 / Real.log (1 / κ))^2) (𝓝[>] 0) (𝓝 (gammaTrigamma p)) := by
  have hsmall : Tendsto (fun κ => κ * kummerExpect p κ (fun t => t * kummerShapeScore t) *
      (Real.log (1 / κ))^2) (𝓝[>] 0) (𝓝 0) := by
    have hlo := kappa_log_sq_tendsto_zero.neg
    simp only [neg_zero] at hlo
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' hlo tendsto_const_nhds
    · filter_upwards [self_mem_nhdsWithin] with κ hκ
      have hh := mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left (expect_t_shapeScore_bounds p κ hp hκ).1 hκ.le)
        (sq_nonneg (Real.log (1 / κ)))
      simpa only [mul_neg, mul_one, neg_mul] using hh
    · filter_upwards [self_mem_nhdsWithin] with κ hκ
      exact mul_nonpos_of_nonpos_of_nonneg
        (mul_nonpos_of_nonneg_of_nonpos hκ.le (expect_t_shapeScore_bounds p κ hp hκ).2)
        (sq_nonneg _)
  have hratio : Tendsto (fun κ => Real.log (1 / κ) / kummerZ p κ) (𝓝[>] 0) (𝓝 1) := by
    simpa only [inv_div, inv_one] using (kummerZ_log_ratio_tendsto p hp).inv₀ (by norm_num)
  have hbig := ((kummerN0_tendsto_one p hp).mul (kummerScoreNumerator_tendsto p hp)).mul (hratio.pow 2)
  simp only [one_mul, one_pow, mul_one] at hbig
  have h := hsmall.add hbig
  simp only [zero_add] at h
  apply h.congr'
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  have hZ := (L3_kummer_moments p κ hp hκ).1
  rw [((kummerMean_shape_deriv p κ hp hκ).const_mul κ).deriv]
  dsimp only [kummerMean, kummerExpect, kummerN0]
  simp_rw [neg_mul]
  rw [integral_neg]
  field_simp [hZ.ne']
  <;> ring


#print axioms kummerShapeA_log_sq_tendsto

/-- The normalized negative shape score is the exact correction to the leading
half-log mean, obtained from the joint Gamma identity. -/
theorem kummerScoreNumerator_identity (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    (∫ t in Ioi (0 : ℝ), -kummerShapeScore t * kummerWeight p κ t) /
      kummerZ p κ = 2 * kummerLog1p p κ - gammaLogDerivative p + Real.log κ := by
  have hscore := expect_shapeScore p κ hp hκ
  have hjoint := kummer_log_joint_identity p κ hp hκ
  have he : (∫ t in Ioi (0 : ℝ), -kummerShapeScore t * kummerWeight p κ t) /
      kummerZ p κ = -kummerExpect p κ kummerShapeScore := by
    dsimp only [kummerExpect]
    simp_rw [neg_mul]
    rw [integral_neg]
    ring
  rw [he, hscore]
  linarith

#print axioms kummerScoreNumerator_identity
end Dobson.Coverage2Exact
