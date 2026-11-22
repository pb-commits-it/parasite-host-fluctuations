import Mathlib
import DobsonProofs.Coverage2Exact
import DobsonProofs.KummerGap

/-!
# The barrier argument: its algebra and its topological step (items T1 to T6)

The elementary skeleton of Theorem 4 (and the derivative formula of Proposition 5) of the proof
that the loss decreases as a stay lengthens. All variables are real.

* T1, T2, T3, T5, T6 are stated on variables. `kappa` is the proof's `κ`; in T3 the variable
  `d` stands for `κδ`; in T5 `q` stands for `1/κ`.
* T4. "`F` is differentiable on `(0, ∞)`" is: a function `F' : ℝ → ℝ` is given and
  `HasDerivAt F (F' x) x` holds for every `x > 0`. Nothing is assumed about `F` or `F'` at
  `x ≤ 0`. "There is `K > 0` with `F x < 0` for all `x ≥ K`" is
  `∃ K : ℝ, 0 < K ∧ ∀ x : ℝ, x ≥ K → F x < 0`.

Route for T4. Suppose `F x₀ ≥ 0` for some `x₀ > 0`; then `x₀ < K`. The set of points of
`[x₀, K]` where `F ≥ 0` is closed, bounded and contains `x₀`, so it has a largest element `z`,
and `z < K`. Every point of `(z, K]` has `F < 0`. If `F z > 0`, continuity at `z` gives points
just to the right of `z` with `F > 0`. If `F z = 0`, then `F' z > 0` and the difference
quotient gives points just to the right of `z` with `F > F z = 0`. Either way a contradiction.
(The written proof takes the first zero after a point where `F > 0`; this is the same argument
run from the right, and it covers the cases `F x₀ > 0` and `F x₀ = 0` together.)
-/

/-! ## Coverage correction — 2026-11-22

The actual Kummer strict gap, derivative identities and large-κ endpoint
sign are proved and used by `assembly` and `assembly_derivative`.
`thm:exact-mono` and `prop:exact-P5` have core coverage in the version-2
inventory: the actual barrier inequality and positive scalar derivative
expression are checked. Full parametrization, chain-rule assembly and the
additional endpoint/infimum conclusions remain outside that credit. -/

/-!
## leanp1 actual strict-gap correction — 2026-11-16

The new KummerGap module proves the actual normalized log-density change of
variables, single crossing for every positive shape, and the strict centered
Jensen-gap comparison, with all required integrability proved. Public T3 now
uses actual Kummer quantities and derives its gap, removing part1:18.
The former scalar implication remains a private algebra lemma; its conditional
algebra is not counted as proof of the actual distributional inequality.
-/

namespace Dobson.Barrier

/-- **T1 (the identity at a zero).** For real `κ`, `a`, `s`, `V` with `s ≠ 0` and `s ≠ 2`, if
`(2 − s)·V = 1 − s` then
`(2 − s)·((κ + 2a)·V − a) + s·V·(1 − V) = s·(V·(1 − V) − (a + κ − κ/s))`.
(The hypothesis `s ≠ 2` is part of the statement but is not needed by the proof: with `s = 2`
the hypothesis `(2 − s)·V = 1 − s` reads `0 = −1`.) -/
theorem T1 (kappa a s V : ℝ) (hs0 : s ≠ 0) (_hs2 : s ≠ 2) (hV : (2 - s) * V = 1 - s) :
    (2 - s) * ((kappa + 2 * a) * V - a) + s * V * (1 - V)
      = s * (V * (1 - V) - (a + kappa - kappa / s)) := by
  -- `s ≠ 2` is not needed. It is cleared here so that no later step can use it.
  clear _hs2
  have e : s * (kappa / s) = kappa := mul_div_cancel₀ kappa hs0
  linear_combination (kappa + 2 * a) * hV - e

/-- **T2 (the general derivative formula is consistent with T1).** For all real `κ`, `a`, `s`,
`V`: `(2 − s)·((κ + 2a)·V − a) − (s·V)·V + s·V = (2 − s)·((κ + 2a)·V − a) + s·V·(1 − V)`. -/
theorem T2 (kappa a s V : ℝ) :
    (2 - s) * ((kappa + 2 * a) * V - a) - (s * V) * V + s * V
      = (2 - s) * ((kappa + 2 * a) * V - a) + s * V * (1 - V) := by
  ring

/-- **T3 (the sign at a zero).** If `0 < s`, `s < 1`, `(2 − s)·V = 1 − s` and `d < V/2`, then
`0 < V`, `V < 1/2`, `s·(V·(1 − V) − d) > s·V·(1/2 − V)` and `s·V·(1/2 − V) > 0`. -/
private theorem barrier_zero_sign (s V d : ℝ) (hs0 : 0 < s) (hs1 : s < 1) (hV : (2 - s) * V = 1 - s)
    (hd : d < V / 2) :
    0 < V ∧ V < 1 / 2 ∧ s * (V * (1 - V) - d) > s * V * (1 / 2 - V)
      ∧ s * V * (1 / 2 - V) > 0 := by
  have h2 : 0 < 2 - s := by linarith
  have hV0 : 0 < V := by
    have hprod : 0 < (2 - s) * V := by
      rw [hV]
      linarith
    exact (mul_pos_iff_of_pos_left h2).mp hprod
  have hV12 : V < 1 / 2 := by
    have e : (2 - s) * (1 / 2 - V) = s / 2 := by linear_combination -hV
    have hprod : 0 < (2 - s) * (1 / 2 - V) := by
      rw [e]
      linarith
    have := (mul_pos_iff_of_pos_left h2).mp hprod
    linarith
  refine ⟨hV0, hV12, ?_, ?_⟩
  · have hgap : 0 < s * (V / 2 - d) := mul_pos hs0 (by linarith)
    have e : s * (V * (1 - V) - d) - s * V * (1 / 2 - V) = s * (V / 2 - d) := by ring
    rw [gt_iff_lt, ← sub_pos, e]
    exact hgap
  · exact mul_pos (mul_pos hs0 hV0) (by linarith)

/-- **T4 (the topological step).** Let `F : ℝ → ℝ` have derivative `F' x` at every `x > 0`.
Suppose there is `K > 0` with `F x < 0` for all `x ≥ K`, and that `F' z > 0` at every `z > 0`
with `F z = 0`. Then `F x < 0` for every `x > 0`.
(The condition `K > 0` is part of the statement but is not needed by the proof.) -/
theorem T4 (F F' : ℝ → ℝ) (hF : ∀ x : ℝ, 0 < x → HasDerivAt F (F' x) x)
    (hK : ∃ K : ℝ, 0 < K ∧ ∀ x : ℝ, x ≥ K → F x < 0)
    (hzero : ∀ z : ℝ, 0 < z → F z = 0 → F' z > 0) :
    ∀ x : ℝ, 0 < x → F x < 0 := by
  -- `K > 0` is not needed: it is discarded here and never named.
  obtain ⟨K, -, hKneg⟩ := hK
  intro x₀ hx₀
  by_contra hnot
  have hx₀F : 0 ≤ F x₀ := not_lt.mp hnot
  have hx₀K : x₀ < K := by
    by_contra hge
    exact hnot (hKneg x₀ (not_lt.mp hge))
  -- `A`: the points of `[x₀, K]` where `F ≥ 0`. It is compact and contains `x₀`.
  have hcont : ContinuousOn F (Set.Icc x₀ K) := fun t ht =>
    (hF t (hx₀.trans_le ht.1)).continuousAt.continuousWithinAt
  have hAclosed : IsClosed (Set.Icc x₀ K ∩ F ⁻¹' Set.Ici 0) :=
    hcont.preimage_isClosed_of_isClosed isClosed_Icc isClosed_Ici
  have hAcompact : IsCompact (Set.Icc x₀ K ∩ F ⁻¹' Set.Ici 0) :=
    isCompact_Icc.of_isClosed_subset hAclosed Set.inter_subset_left
  have hAne : (Set.Icc x₀ K ∩ F ⁻¹' Set.Ici 0).Nonempty := ⟨x₀, ⟨le_rfl, hx₀K.le⟩, hx₀F⟩
  -- its largest element `z`
  obtain ⟨z, hzA, hzmax⟩ := hAcompact.exists_isGreatest hAne
  have hxz : x₀ ≤ z := hzA.1.1
  have hz0 : 0 < z := hx₀.trans_le hxz
  have hzF : 0 ≤ F z := hzA.2
  have hzK : z < K := by
    rcases lt_or_eq_of_le hzA.1.2 with hlt | heq
    · exact hlt
    · exfalso
      have hneg := hKneg z heq.ge
      linarith
  -- every point of `(z, K]` has `F < 0`
  have hright : ∀ t : ℝ, z < t → t ≤ K → F t < 0 := by
    intro t hzt htK
    by_contra hnn
    have htA : t ∈ Set.Icc x₀ K ∩ F ⁻¹' Set.Ici 0 := ⟨⟨hxz.trans hzt.le, htK⟩, not_lt.mp hnn⟩
    exact not_le.mpr hzt (hzmax htA)
  -- but just to the right of `z` there are points with `F > 0`
  have hpos : ∀ᶠ t in nhdsWithin z (Set.Ioi z), 0 < F t := by
    rcases hzF.lt_or_eq with hgt | heq
    · -- `F z > 0`: continuity at `z`
      have hc : ContinuousAt F z := (hF z hz0).continuousAt
      exact (hc.eventually (lt_mem_nhds hgt)).filter_mono nhdsWithin_le_nhds
    · -- `F z = 0`: the derivative at `z` is positive
      have hd : 0 < F' z := hzero z hz0 heq.symm
      have hs : Filter.Tendsto (slope F z) (nhdsWithin z {z}ᶜ) (nhds (F' z)) :=
        hasDerivAt_iff_tendsto_slope.mp (hF z hz0)
      have hsub : Set.Ioi z ⊆ ({z}ᶜ : Set ℝ) := fun t (ht : z < t) => ne_of_gt ht
      have hs' : ∀ᶠ t in nhdsWithin z (Set.Ioi z), 0 < slope F z t :=
        (hs.eventually (lt_mem_nhds hd)).filter_mono (nhdsWithin_mono z hsub)
      filter_upwards [hs', self_mem_nhdsWithin] with t ht htz
      have htz' : 0 < t - z := sub_pos.mpr htz
      rw [slope_def_field, ← heq, sub_zero] at ht
      exact (div_pos_iff_of_pos_right htz').mp ht
  have hIoc : ∀ᶠ t in nhdsWithin z (Set.Ioi z), t ∈ Set.Ioc z K := Ioc_mem_nhdsGT hzK
  obtain ⟨t, htpos, htIoc⟩ := (hpos.and hIoc).exists
  exact absurd (hright t htIoc.1 htIoc.2) (not_lt.mpr htpos.le)

/-- **T5 (the endpoint expansion's algebra).** For real `p`, `q`:
`(p·q − p·(3p + 1)·q²) − ((p·q − p·(3p + 1)/2·q²) − (3/2)·(p·q)²) = −(p/2)·q²`. -/
theorem T5 (p q : ℝ) :
    (p * q - p * (3 * p + 1) * q ^ 2)
        - ((p * q - p * (3 * p + 1) / 2 * q ^ 2) - (3 / 2) * (p * q) ^ 2)
      = -(p / 2) * q ^ 2 := by
  ring

/-- **T6 (the model's derivative).** For real `V`, `s` with `V > 0`, `0 < s`, `s < 1`:
`1/V − 2 − s/(1 − s) > 0 ↔ V < (1 − s)/(2 − s)`.
(The hypothesis `0 < s` is part of the statement but is not needed by the proof.) -/
theorem T6 (V s : ℝ) (hV : V > 0) (_hs0 : 0 < s) (hs1 : s < 1) :
    1 / V - 2 - s / (1 - s) > 0 ↔ V < (1 - s) / (2 - s) := by
  -- `0 < s` is not needed. It is cleared here so that no later step can use it.
  clear _hs0
  have h1 : 0 < 1 - s := by linarith
  have h2 : 0 < 2 - s := by linarith
  have hV' : V ≠ 0 := hV.ne'
  have h1' : 1 - s ≠ 0 := h1.ne'
  have e : 1 / V - 2 - s / (1 - s) = ((1 - s) - (2 - s) * V) / (V * (1 - s)) := by
    field_simp
    ring
  rw [e, gt_iff_lt, div_pos_iff_of_pos_right (mul_pos hV h1), lt_div_iff₀ h2]
  constructor <;> intro h <;> linarith

/-! ## How the items fit together

The private scalar helpers assemble the topological argument from supplied
functions. The public `assembly` and `assembly_derivative` use actual Kummer
integrals and prove the required gap, derivative and endpoint conditions.
Their barrier and positive-expression conclusions are the credited cores,
not the complete manuscript parametrization and endpoint claims. -/

/-- **Assembly of T1, T3 and T4 (an addition, not a listed item).** Let `s`, `V`, `a`, `d` be
real functions of `κ` and put `F κ = (2 − s κ)·V κ − (1 − s κ)`. Suppose that for every `κ > 0`:
`0 < s κ < 1`; `F` has derivative `F' κ` at `κ` with
`κ·F' κ = (2 − s)·((κ + 2a)·V − a) + s·V·(1 − V)`; `d κ = a κ + κ − κ/s κ`; and `d κ < V κ/2`.
Suppose also that there is `K > 0` with `F κ < 0` for all `κ ≥ K`. Then
`V κ < (1 − s κ)/(2 − s κ)` for every `κ > 0`. -/
private theorem barrier_assembly (s V a d F' : ℝ → ℝ)
    (hs : ∀ κ : ℝ, 0 < κ → 0 < s κ ∧ s κ < 1)
    (hF : ∀ κ : ℝ, 0 < κ →
      HasDerivAt (fun k : ℝ => (2 - s k) * V k - (1 - s k)) (F' κ) κ)
    (hF' : ∀ κ : ℝ, 0 < κ →
      κ * F' κ = (2 - s κ) * ((κ + 2 * a κ) * V κ - a κ) + s κ * V κ * (1 - V κ))
    (hd : ∀ κ : ℝ, 0 < κ → d κ = a κ + κ - κ / s κ)
    (hgap : ∀ κ : ℝ, 0 < κ → d κ < V κ / 2)
    (hend : ∃ K : ℝ, 0 < K ∧ ∀ κ : ℝ, κ ≥ K → (2 - s κ) * V κ - (1 - s κ) < 0) :
    ∀ κ : ℝ, 0 < κ → V κ < (1 - s κ) / (2 - s κ) := by
  -- at a zero of `F` the derivative is positive: T1, then T3
  have hzero : ∀ z : ℝ, 0 < z →
      (fun k : ℝ => (2 - s k) * V k - (1 - s k)) z = 0 → F' z > 0 := by
    intro z hz hz0
    have hz0' : (2 - s z) * V z - (1 - s z) = 0 := hz0
    have hzV : (2 - s z) * V z = 1 - s z := by linarith
    obtain ⟨hs0, hs1⟩ := hs z hz
    have h1 := T1 z (a z) (s z) (V z) hs0.ne' (by linarith) hzV
    have h3 := barrier_zero_sign (s z) (V z) (d z) hs0 hs1 hzV (hgap z hz)
    rw [← hd z hz] at h1
    have hpos : 0 < z * F' z := by
      rw [hF' z hz, h1]
      exact lt_trans h3.2.2.2 h3.2.2.1
    exact (mul_pos_iff_of_pos_left hz).mp hpos
  -- T4
  intro κ hκ
  have hneg : (2 - s κ) * V κ - (1 - s κ) < 0 :=
    T4 (fun k : ℝ => (2 - s k) * V k - (1 - s k)) F' hF hend hzero κ hκ
  have h2 : 0 < 2 - s κ := by linarith [(hs κ hκ).2]
  rw [lt_div_iff₀ h2]
  linarith

/-! Coverage correction, 2026-11-22: the public assemblies use
actual normalized Kummer integrals. Their differential identities, strict
single-crossing gap and large-κ endpoint sign are proved in the imported
and local integral results. -/

open Coverage2Exact

/-- The stay coordinate in the manuscript, computed from the logarithmic moment. -/
noncomputable def kummerStay (p κ : ℝ) : ℝ := Real.exp (-kummerLog1p p κ)

/-- The rescaled first moment in the Kummer differential identities. -/
noncomputable def kummerA (p κ : ℝ) : ℝ := κ * kummerMean p κ

/-- The rescaled log/linear covariance in the manuscript. -/
noncomputable def kummerV (p κ : ℝ) : ℝ :=
  κ * (kummerCrossLog1p p κ - kummerLog1p p κ * kummerMean p κ)

/-- The actual barrier whose sign controls the stay derivative. -/
noncomputable def kummerBarrier (p κ : ℝ) : ℝ :=
  (2 - kummerStay p κ) * kummerV p κ - (1 - kummerStay p κ)

/-- The sign at a barrier zero for the actual Kummer model. The strict gap
is proved by the normalized reflected-density argument, not supplied. -/
theorem T3 (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ)
    (hV : (2 - kummerStay p κ) * kummerV p κ = 1 - kummerStay p κ) :
    0 < kummerV p κ ∧ kummerV p κ < 1 / 2 ∧
      kummerStay p κ * (kummerV p κ * (1 - kummerV p κ) -
        (kummerA p κ + κ - κ / kummerStay p κ)) >
        kummerStay p κ * kummerV p κ * (1 / 2 - kummerV p κ) ∧
      kummerStay p κ * kummerV p κ * (1 / 2 - kummerV p κ) > 0 := by
  have hs0 : 0 < kummerStay p κ := Real.exp_pos _
  have hs1 : kummerStay p κ < 1 :=
    Real.exp_lt_one_iff.mpr (neg_neg_of_pos (KummerGap.logMean_pos p κ hp hκ))
  exact barrier_zero_sign (kummerStay p κ) (kummerV p κ)
    (kummerA p κ + κ - κ / kummerStay p κ) hs0 hs1 hV
    (KummerGap.kummer_strict_gap p κ hp hκ)

/-- Differentiate the actual barrier, using the normalized Kummer moments. -/
theorem kummer_barrier_derivative (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    HasDerivAt (kummerBarrier p) (deriv (kummerBarrier p) κ) κ ∧
      κ * deriv (kummerBarrier p) κ =
        (2 - kummerStay p κ) *
          ((κ + 2 * kummerA p κ) * kummerV p κ - kummerA p κ) +
        kummerStay p κ * kummerV p κ * (1 - kummerV p κ) := by
  have hm := kummerLog1p_deriv p κ hp hκ
  have hμ := kummerMean_deriv p κ hp hκ
  have hν := kummerCrossLog1p_deriv p κ hp hκ
  have hs := hm.neg.exp
  have hv := (hasDerivAt_id κ).mul (hν.sub (hm.mul hμ))
  change HasDerivAt (kummerStay p) _ κ at hs
  change HasDerivAt (kummerV p) _ κ at hv
  have hs_eq : κ * deriv (kummerStay p) κ = kummerStay p κ * kummerV p κ := by
    rw [hs.deriv]
    dsimp [kummerStay, kummerV]
    ring
  have hv_eq : κ * deriv (kummerV p) κ =
      (κ + 2 * kummerA p κ) * kummerV p κ - kummerA p κ := by
    exact (kummer_odes κ p hκ hp).2.2.1
  have hf := (((hasDerivAt_const κ 2).sub hs.differentiableAt.hasDerivAt).mul
    hv.differentiableAt.hasDerivAt).sub
    ((hasDerivAt_const κ 1).sub hs.differentiableAt.hasDerivAt)
  change HasDerivAt (kummerBarrier p) _ κ at hf
  refine ⟨hf.differentiableAt.hasDerivAt, ?_⟩
  rw [hf.deriv]
  simp only [Pi.sub_apply]
  linear_combination (2 - kummerStay p κ) * hv_eq +
    (1 - kummerV p κ) * hs_eq

end Dobson.Barrier

namespace Dobson.KummerLargeKappa
open Set Filter MeasureTheory
open scoped Topology
open NoiseAnalyticity Coverage2Exact Barrier
noncomputable def tiltedWeight (p q z : ℝ) : ℝ :=
  z^(p-1) * (1+q*z)^(-p) * Real.exp (-z)
noncomputable def tiltedMoment (p q : ℝ) (n : ℕ) : ℝ :=
  ∫ z in Ioi (0 : ℝ), z^n * tiltedWeight p q z

theorem gamma_polynomial_integrable (p : ℝ) (n : ℕ) (hp : 0 < p) :
    IntegrableOn (fun z : ℝ => z^n*z^(p-1)*Real.exp (-z)) (Ioi 0) := by
  have hd : IntegrableOn (fun z : ℝ => z^(p+n-1)*Real.exp (-z)) (Ioi 0) := by
    simpa [Real.rpow_one] using
      (integrableOn_rpow_mul_exp_neg_mul_rpow (p := 1) (s := p+n-1) (b := 1)
        (by have := Nat.cast_nonneg (α := ℝ) n; linarith) one_pos one_pos)
  apply hd.congr
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
  rw [← Real.rpow_natCast, ← Real.rpow_add hz]
  congr 2
  ring

theorem tilted_integrable (p q : ℝ) (n : ℕ) (hp : 0 < p) (hq : 0 ≤ q) :
    IntegrableOn (fun z : ℝ => z^n * tiltedWeight p q z) (Ioi 0) := by
  have hc : ContinuousOn (fun z : ℝ => z^n * tiltedWeight p q z) (Ioi 0) := by
    unfold tiltedWeight
    apply ContinuousOn.mul (continuousOn_id.pow n)
    apply ContinuousOn.mul
    · apply ContinuousOn.mul
      · exact continuousOn_id.rpow_const (fun z hz => Or.inl (ne_of_gt hz))
      · apply (continuousOn_const.add (continuousOn_const.mul continuousOn_id)).rpow_const
        intro z hz
        left
        have : 0 ≤ q*z := mul_nonneg hq hz.le
        change 1+q*z ≠ 0
        linarith
    · fun_prop
  refine (gamma_polynomial_integrable p n hp).mono'
    (hc.aestronglyMeasurable measurableSet_Ioi) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
  have hz0 : 0 < z := hz
  have hpow : (1+q*z)^(-p) ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos
    (by nlinarith) (by linarith)
  have hnn : 0 ≤ z^n * tiltedWeight p q z := by unfold tiltedWeight; positivity
  rw [Real.norm_of_nonneg hnn]
  dsimp only [tiltedWeight]
  calc
    z^n * (z^(p-1)*(1+q*z)^(-p)*Real.exp (-z)) =
      (z^n*z^(p-1)*Real.exp (-z)) * (1+q*z)^(-p) := by ring
    _ ≤ (z^n*z^(p-1)*Real.exp (-z)) * 1 :=
      mul_le_mul_of_nonneg_left hpow (by positivity)
    _ = _ := mul_one _

theorem tiltedMoment_upper (p q : ℝ) (n : ℕ) (hp : 0 < p) (hq : 0 ≤ q) :
    tiltedMoment p q n ≤ Real.Gamma (p+n) := by
  have hi := tilted_integrable p q n hp hq
  have hd := gamma_polynomial_integrable p n hp
  have he : (∫ z in Ioi (0 : ℝ), z^n*z^(p-1)*Real.exp (-z)) = Real.Gamma (p+n) := by
    have hfun : (fun z : ℝ => z^n*z^(p-1)*Real.exp (-z)) =ᵐ[volume.restrict (Ioi 0)]
        (fun z => z^(p+n-1)*Real.exp (-z)) := by
      filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
      rw [← Real.rpow_natCast, ← Real.rpow_add hz]
      congr 2
      ring
    rw [integral_congr_ae hfun]
    simpa using Real.integral_rpow_mul_exp_neg_mul_Ioi (by positivity : 0 < p+n) (r := 1) one_pos
  rw [← he]
  apply integral_mono_ae hi hd
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
  have hz0 : 0 < z := hz
  have hpow : (1+q*z)^(-p) ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos
    (by nlinarith) (by linarith)
  dsimp [tiltedWeight]
  nlinarith [mul_le_mul_of_nonneg_left hpow (by positivity : 0 ≤ z^n*z^(p-1)*Real.exp (-z))]

theorem tiltedMoment_zero_lower (p q : ℝ) (hp : 0 < p) (hq : 0 ≤ q) (hq1 : q ≤ 1) :
    kummerZ p 1 ≤ tiltedMoment p q 0 := by
  have hl := tilted_integrable p 1 0 hp zero_le_one
  have hr := tilted_integrable p q 0 hp hq
  have he : tiltedMoment p 1 0 = kummerZ p 1 := by
    simp [tiltedMoment, tiltedWeight, kummerZ, kummerWeight]
  rw [← he]
  apply integral_mono_ae hl hr
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
  have hz0 : 0 < z := hz
  simp only [pow_zero, one_mul, tiltedWeight]
  apply mul_le_mul_of_nonneg_right _ (Real.exp_pos _).le
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply Real.rpow_le_rpow_of_nonpos
  · positivity
  · nlinarith
  · linarith

theorem kummer_scaled_moment (p q : ℝ) (n : ℕ) (hq : 0 < q) :
    kummerExpect p q⁻¹ (fun t => t^n) = q^n * tiltedMoment p q n / tiltedMoment p q 0 := by
  have he (n : ℕ) : (∫ t in Ioi (0 : ℝ), t^n*kummerWeight p q⁻¹ t) =
      (q*q^(p-1))*q^n*tiltedMoment p q n := by
    have hc := integral_comp_mul_left_Ioi' (fun t => t^n*kummerWeight p q⁻¹ t) 0 hq
    simp only [mul_zero, smul_eq_mul] at hc
    rw [← hc]
    have hf : (fun z : ℝ => (q*z)^n*kummerWeight p q⁻¹ (q*z)) =ᵐ[volume.restrict (Ioi 0)]
        (fun z => (q^(p-1)*q^n) * (z^n*tiltedWeight p q z)) := by
      filter_upwards [ae_restrict_mem measurableSet_Ioi] with z hz
      dsimp [kummerWeight, tiltedWeight]
      rw [mul_pow, Real.mul_rpow hq.le (le_of_lt hz)]
      have harg : -q⁻¹*(q*z) = -z := by field_simp
      rw [harg]
      ring
    rw [integral_congr_ae hf, integral_const_mul]
    dsimp [tiltedMoment]
    ring
  have hden : kummerZ p q⁻¹ = (q*q^(p-1))*tiltedMoment p q 0 := by
    simpa [kummerZ] using he 0
  dsimp [kummerExpect]
  rw [hden, he n]
  have hn : q*q^(p-1) ≠ 0 := ne_of_gt (by positivity)
  field_simp
  <;> ring

theorem kummer_moment_bound (p κ : ℝ) (n : ℕ) (hp : 0 < p) (hκ : 1 ≤ κ) :
    kummerExpect p κ (fun t => t^n) ≤ κ⁻¹^n * (Real.Gamma (p+n)/kummerZ p 1) := by
  have hk : 0 < κ := lt_of_lt_of_le zero_lt_one hκ
  have hq : 0 < κ⁻¹ := inv_pos.mpr hk
  have hq1 : κ⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hκ
  have hz := (L3_kummer_moments p 1 hp zero_lt_one).1
  have hlo := tiltedMoment_zero_lower p κ⁻¹ hp hq.le hq1
  have hden : 0 < tiltedMoment p κ⁻¹ 0 := hz.trans_le hlo
  have hu := tiltedMoment_upper p κ⁻¹ n hp hq.le
  have hg : 0 ≤ Real.Gamma (p+n) := (Real.Gamma_pos_of_pos (by positivity)).le
  rw [show κ = (κ⁻¹)⁻¹ from (inv_inv κ).symm, kummer_scaled_moment p κ⁻¹ n hq, inv_inv,
    mul_div_assoc]
  apply mul_le_mul_of_nonneg_left _ (pow_nonneg hq.le _)
  exact (div_le_div_of_nonneg_right hu hden.le).trans
    (div_le_div_of_nonneg_left hg hz hlo)

theorem log_and_cross_bound (p κ : ℝ) (hp : 0 < p) (hκ : 0 < κ) :
    0 ≤ kummerMean p κ ∧ kummerLog1p p κ ≤ kummerMean p κ ∧
    kummerCrossLog1p p κ ≤ kummerSecondMoment p κ := by
  obtain ⟨hz, hi0, hi1, hi2, _⟩ := L3_kummer_moments p κ hp hκ
  have hcont : ContinuousOn (kummerWeight p κ) (Ioi 0) := by
    unfold kummerWeight
    apply ContinuousOn.mul
    · apply ContinuousOn.mul
      · exact continuousOn_id.rpow_const (fun t ht => Or.inl (ne_of_gt ht))
      · apply (continuousOn_const.add continuousOn_id).rpow_const
        intro t ht
        left
        change 1+t ≠ 0
        have : 0 < t := ht
        linarith
    · fun_prop
  have hlogcont : ContinuousOn (fun t : ℝ => Real.log (1+t)) (Ioi 0) := by
    apply (continuousOn_const.add continuousOn_id).log
    intro t ht
    change 1+t ≠ 0
    have : 0 < t := ht
    linarith
  have hw (t : ℝ) (ht : 0 < t) : 0 ≤ kummerWeight p κ t := by
    unfold kummerWeight
    positivity
  have hlog (t : ℝ) (ht : 0 < t) : 0 ≤ Real.log (1+t) ∧ Real.log (1+t) ≤ t := by
    constructor
    · exact Real.log_nonneg (by linarith)
    · linarith [Real.log_le_sub_one_of_pos (by linarith : 0 < 1+t)]
  have hilog : IntegrableOn (fun t => Real.log (1+t)*kummerWeight p κ t) (Ioi 0) := by
    refine hi1.mono' ((hlogcont.mul hcont).aestronglyMeasurable measurableSet_Ioi) ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    rw [Real.norm_of_nonneg (mul_nonneg (hlog t ht).1 (hw t ht))]
    exact mul_le_mul_of_nonneg_right (hlog t ht).2 (hw t ht)
  have hicross : IntegrableOn (fun t => (t*Real.log (1+t))*kummerWeight p κ t) (Ioi 0) := by
    refine hi2.mono' (((continuousOn_id.mul hlogcont).mul hcont).aestronglyMeasurable measurableSet_Ioi) ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    rw [Real.norm_of_nonneg (mul_nonneg (mul_nonneg (le_of_lt ht) (hlog t ht).1) (hw t ht))]
    have hh := mul_le_mul_of_nonneg_left (hlog t ht).2 (le_of_lt ht)
    exact mul_le_mul_of_nonneg_right (by simpa [pow_two] using hh) (hw t ht)
  have hm : 0 ≤ kummerMean p κ := by
    unfold kummerMean kummerExpect
    apply mul_nonneg (inv_nonneg.mpr hz.le)
    apply integral_nonneg_of_ae
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    exact mul_nonneg (le_of_lt ht) (hw t ht)
  refine ⟨hm, ?_, ?_⟩
  · unfold kummerLog1p kummerMean kummerExpect
    apply mul_le_mul_of_nonneg_left _ (inv_nonneg.mpr hz.le)
    apply integral_mono_ae hilog hi1
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    exact mul_le_mul_of_nonneg_right (hlog t ht).2 (hw t ht)
  · unfold kummerCrossLog1p kummerSecondMoment kummerExpect
    apply mul_le_mul_of_nonneg_left _ (inv_nonneg.mpr hz.le)
    apply integral_mono_ae hicross hi2
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have hh := mul_le_mul_of_nonneg_left (hlog t ht).2 (le_of_lt ht)
    exact mul_le_mul_of_nonneg_right (by simpa [pow_two] using hh) (hw t ht)


theorem kummer_limits (p : ℝ) (hp : 0 < p) :
    Tendsto (kummerLog1p p) atTop (𝓝 0) ∧ Tendsto (kummerV p) atTop (𝓝 0) := by
  have hq : Tendsto (fun κ : ℝ => κ⁻¹) atTop (𝓝 0) := tendsto_inv_atTop_zero
  have hmupper : Tendsto (fun κ : ℝ => κ⁻¹*(Real.Gamma (p+1)/kummerZ p 1)) atTop (𝓝 0) := by
    simpa using hq.mul_const (Real.Gamma (p+1)/kummerZ p 1)
  have hvupper : Tendsto (fun κ : ℝ => κ⁻¹*(Real.Gamma (p+2)/kummerZ p 1)) atTop (𝓝 0) := by
    simpa using hq.mul_const (Real.Gamma (p+2)/kummerZ p 1)
  constructor
  · apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hmupper
    · filter_upwards [eventually_ge_atTop (1 : ℝ)] with κ hκ
      exact (KummerGap.logMean_pos p κ hp (by linarith)).le
    · filter_upwards [eventually_ge_atTop (1 : ℝ)] with κ hκ
      have hb := kummer_moment_bound p κ 1 hp hκ
      have hm := (log_and_cross_bound p κ hp (by linarith)).2.1
      exact hm.trans (by simpa [kummerMean] using hb)
  · apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hvupper
    · filter_upwards [eventually_ge_atTop (1 : ℝ)] with κ hκ
      exact ((kummer_odes κ p (by linarith) hp).2.2.2).le
    · filter_upwards [eventually_ge_atTop (1 : ℝ)] with κ hκ
      have hk : 0 < κ := by linarith
      obtain ⟨hm, _, hc⟩ := log_and_cross_bound p κ hp hk
      have hl : 0 ≤ kummerLog1p p κ := (KummerGap.logMean_pos p κ hp hk).le
      have hb := kummer_moment_bound p κ 2 hp hκ
      change kummerSecondMoment p κ ≤ _ at hb
      have he : κ*(κ⁻¹^2*(Real.Gamma (p+2)/kummerZ p 1)) =
          κ⁻¹*(Real.Gamma (p+2)/kummerZ p 1) := by field_simp
      rw [← he]
      unfold kummerV
      apply mul_le_mul_of_nonneg_left _ hk.le
      exact (sub_le_self _ (mul_nonneg hl hm)).trans (hc.trans hb)

theorem barrier_limit (p : ℝ) (hp : 0 < p) :
    Tendsto (kummerBarrier p) atTop (𝓝 0) := by
  obtain ⟨hm,hv⟩ := kummer_limits p hp
  have hs : Tendsto (kummerStay p) atTop (𝓝 1) := by
    change Tendsto (fun κ => Real.exp (-kummerLog1p p κ)) atTop (𝓝 1)
    have he : Tendsto (fun κ => Real.exp (-kummerLog1p p κ)) atTop (𝓝 (Real.exp (-0))) :=
      Real.continuous_exp.continuousAt.tendsto.comp hm.neg
    simpa only [neg_zero, Real.exp_zero] using he
  have h2 : Tendsto (fun _ : ℝ => (2 : ℝ)) atTop (𝓝 2) := tendsto_const_nhds
  have h1 : Tendsto (fun _ : ℝ => (1 : ℝ)) atTop (𝓝 1) := tendsto_const_nhds
  change Tendsto (fun κ => (2-kummerStay p κ)*kummerV p κ-(1-kummerStay p κ)) atTop (𝓝 0)
  simpa using ((h2.sub hs).mul hv).sub (h1.sub hs)

/-- A function tending to zero cannot have positive derivative at every
nonnegative point. A maximum on a finite interval gives the contradiction. -/
theorem eventually_negative_of_positive_deriv (F : ℝ → ℝ) (K : ℝ)
    (hderiv : ∀ x, K ≤ x → DifferentiableAt ℝ F x)
    (hpos : ∀ x, K ≤ x → 0 ≤ F x → 0 < deriv F x)
    (hlim : Tendsto F atTop (𝓝 0)) : ∀ x, K ≤ x → F x < 0 := by
  have hnpos : ∀ x, K ≤ x → ¬0 < F x := by
    intro x hx hFx
    have hev := hlim.eventually (eventually_lt_nhds hFx)
    obtain ⟨b, hb⟩ := eventually_atTop.1 hev
    let y := max b (x+1)
    have hxy : x < y := lt_of_lt_of_le (by linarith : x < x+1) (le_max_right _ _)
    have hy : F y < F x := hb y (le_max_left _ _)
    have hc : ContinuousOn F (Icc x y) := fun t ht =>
      (hderiv t (hx.trans ht.1)).continuousAt.continuousWithinAt
    obtain ⟨z, hz, hzmax⟩ := isCompact_Icc.exists_isMaxOn (nonempty_Icc.mpr hxy.le) hc
    have hFz : F x ≤ F z := hzmax (left_mem_Icc.mpr hxy.le)
    have hzy : z < y := by
      rcases hz.2.lt_or_eq with h | h
      · exact h
      · subst z; linarith
    have hd : 0 < deriv F z := hpos z (hx.trans hz.1) (by linarith)
    have hs := (hderiv z (hx.trans hz.1)).hasDerivAt.tendsto_slope
    have hes : ∀ᶠ t in 𝓝[>] z, 0 < slope F z t :=
      (hs.mono_left (nhdsGT_le_nhdsNE z)).eventually (eventually_gt_nhds hd)
    obtain ⟨t, ht, htz⟩ := (hes.and (Ioc_mem_nhdsGT hzy)).exists
    rw [slope_def_field, div_pos_iff_of_pos_right (sub_pos.mpr htz.1)] at ht
    have hh : F t ≤ F z := hzmax ⟨hz.1.trans htz.1.le,htz.2⟩
    linarith
  intro x hx
  have hle : F x ≤ 0 := le_of_not_gt (hnpos x hx)
  rcases hle.lt_or_eq with h | h
  · exact h
  · have hd : 0 < deriv F x := hpos x hx h.ge
    have hs := (hderiv x hx).hasDerivAt.tendsto_slope
    have hes : ∀ᶠ t in 𝓝[>] x, 0 < slope F x t :=
      (hs.mono_left (nhdsGT_le_nhdsNE x)).eventually (eventually_gt_nhds hd)
    obtain ⟨t, ht, htx⟩ := (hes.and self_mem_nhdsWithin).exists
    rw [slope_def_field, div_pos_iff_of_pos_right (sub_pos.mpr htx), h, sub_zero] at ht
    exact (hnpos t (hx.trans htx.le) ht).elim


/-- The endpoint sign follows from the strict gap and zeroth-order moment
bounds. No differentiated asymptotic expansion is assumed. -/
theorem barrier_eventually_negative (p : ℝ) (hp : 0 < p) :
    ∃ K : ℝ, 0 < K ∧ ∀ κ : ℝ, K ≤ κ → kummerBarrier p κ < 0 := by
  have hvsmall := (kummer_limits p hp).2.eventually
    (eventually_lt_nhds (by norm_num : (0 : ℝ) < 1/2))
  obtain ⟨K₀,hK₀⟩ := eventually_atTop.1 hvsmall
  let K := max (1 : ℝ) K₀
  have hK : 0 < K := lt_of_lt_of_le zero_lt_one (le_max_left _ _)
  refine ⟨K,hK,?_⟩
  apply eventually_negative_of_positive_deriv (kummerBarrier p) K
  · intro κ hκ
    exact (kummer_barrier_derivative p κ hp (hK.trans_le hκ)).1.differentiableAt
  · intro κ hκ hF
    have hk : 0 < κ := hK.trans_le hκ
    have hv : 0 < kummerV p κ := (kummer_odes κ p hk hp).2.2.2
    have hvlt : kummerV p κ < 1/2 := hK₀ κ ((le_max_right _ _).trans hκ)
    have hs : 0 < kummerStay p κ := Real.exp_pos _
    have ha : 0 ≤ kummerA p κ := mul_nonneg hk.le (log_and_cross_bound p κ hp hk).1
    have hgap := KummerGap.kummer_strict_gap p κ hp hk
    change kummerA p κ+κ-κ/kummerStay p κ < kummerV p κ/2 at hgap
    have hsa : kummerStay p κ*(κ/kummerStay p κ) = κ := mul_div_cancel₀ κ hs.ne'
    have he : κ*deriv (kummerBarrier p) κ =
        (κ+2*kummerA p κ)*kummerBarrier p κ +
        kummerStay p κ*(kummerV p κ*(1-kummerV p κ)-
          (kummerA p κ+κ-κ/kummerStay p κ)) := by
      rw [(kummer_barrier_derivative p κ hp hk).2]
      unfold kummerBarrier
      linear_combination -hsa
    have hterm : 0 < kummerV p κ*(1-kummerV p κ)-
        (kummerA p κ+κ-κ/kummerStay p κ) := by
      nlinarith [mul_pos hv (sub_pos.mpr hvlt)]
    have hprod : 0 ≤ (κ+2*kummerA p κ)*kummerBarrier p κ :=
      mul_nonneg (by linarith) hF
    have hh : 0 < κ*deriv (kummerBarrier p) κ := by
      rw [he]
      exact add_pos_of_nonneg_of_pos hprod (mul_pos hs hterm)
    exact (mul_pos_iff_of_pos_left hk).mp hh
  · exact barrier_limit p hp
end Dobson.KummerLargeKappa
#print axioms Dobson.KummerLargeKappa.barrier_eventually_negative

namespace Dobson.Barrier
open Coverage2Exact

/-- The barrier inequality for the actual Kummer integrals, assuming only
`p > 0`; the derivative, strict gap and endpoint sign are proved. -/
/- 2026-11-16 leanp1: part1:22 is discharged by the proved actual strict gap. -/
theorem assembly (p : ℝ) (hp : 0 < p)
 :
    ∀ κ : ℝ, 0 < κ → kummerV p κ < (1 - kummerStay p κ) / (2 - kummerStay p κ) := by
  have hs : ∀ κ : ℝ, 0 < κ → 0 < kummerStay p κ ∧ kummerStay p κ < 1 := fun κ hκ =>
    ⟨Real.exp_pos _, Real.exp_lt_one_iff.mpr (neg_neg_of_pos (KummerGap.logMean_pos p κ hp hκ))⟩
  exact barrier_assembly (kummerStay p) (kummerV p) (kummerA p)
    (fun κ => kummerA p κ + κ - κ / kummerStay p κ)
    (deriv (kummerBarrier p)) hs
    (fun κ hκ => (kummer_barrier_derivative p κ hp hκ).1)
    (fun κ hκ => (kummer_barrier_derivative p κ hp hκ).2)
    (fun _ _ => rfl) (fun κ hκ => KummerGap.kummer_strict_gap p κ hp hκ)
    (KummerLargeKappa.barrier_eventually_negative p hp)

/-- Assembly with T6 for the actual Kummer model. Positive covariance and the
barrier derivative are proved from the integrals, not supplied as premises. -/
/- 2026-11-16 leanp1: part1:25 is discharged by the proved actual strict gap. -/
theorem assembly_derivative (p : ℝ) (hp : 0 < p)
 :
    ∀ κ : ℝ, 0 < κ → 1 / kummerV p κ - 2 - kummerStay p κ / (1 - kummerStay p κ) > 0 := by
  have hs : ∀ κ : ℝ, 0 < κ → 0 < kummerStay p κ ∧ kummerStay p κ < 1 := fun κ hκ =>
    ⟨Real.exp_pos _, Real.exp_lt_one_iff.mpr (neg_neg_of_pos (KummerGap.logMean_pos p κ hp hκ))⟩
  intro κ hκ
  have hVpos : 0 < kummerV p κ := (kummer_odes κ p hκ hp).2.2.2
  exact (T6 (kummerV p κ) (kummerStay p κ) hVpos (hs κ hκ).1 (hs κ hκ).2).mpr
    (assembly p hp κ hκ)

end Dobson.Barrier

#print axioms Dobson.Barrier.T1
#print axioms Dobson.Barrier.T2
#print axioms Dobson.Barrier.T3
#print axioms Dobson.Barrier.T4
#print axioms Dobson.Barrier.T5
#print axioms Dobson.Barrier.T6
#print axioms Dobson.Barrier.assembly
#print axioms Dobson.Barrier.assembly_derivative
