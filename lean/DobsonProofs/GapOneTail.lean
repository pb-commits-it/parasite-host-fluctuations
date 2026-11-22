import Mathlib.Analysis.Calculus.InverseFunctionTheorem.Deriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

/-!
# GAP 1: the uniform tail interval

Finite recurrence (4)-(5), degree-eight coefficient bounds, the remainder-to-sign
inequality, and the exact threshold `4 / sqrt 17` from `gap1/gap1_proof.md`.
The stationary-process inputs are explicit hypotheses; see `tail_derivative_neg`
and the dated STATUS entry. No stationary law is constructed in this file.
The derivative lemma and finite algebra implement the displayed recurrence.
-/

/-!
## Hypothesis audit correction — 2026-11-16 (auditfix2)

The following are class (c), unproved steps for the intended ecological
objects, not modelling definitions or textbook results. Their dependent
model results have **partial** coverage in `../COVERAGE_part2.md`.
The conditional Lean implications remain proved. Audit row numbers refer
to the independent hypothesis audit (2026-11-16). These entries describe
the historical interfaces; current theorem signatures and the version-2
coverage inventory (2026-11-22) determine the credited scope.

* `derivative_identity`: hc_pos (#67), hc_cont (#68), hR_sderiv (#69), hV_pos (#70), h_thresh
  (#71).
* `sign_of_remainder`: htail (#72).
* `q_le_of_chord`: hchord (#73).
* `tail_derivative_neg`: hc_pos (#74), hc_cont (#75), hR_sderiv (#76), hV_pos (#77), h_thresh
  (#78), hchord (#79), htail (#80).

Full-model instantiation is not established by naming an arbitrary scalar
a covariance, a growth rate, or a threshold. Missing model identification
is included in the coverage limitation.
-/

namespace Dobson.GapOneTail

open Filter Set
open scoped Topology

noncomputable def Gfun (Rval Vval : ℝ) : ℝ :=
  1 - Real.exp (-Rval) - (2 - Real.exp (-Rval)) * Vval

noncomputable def zSign (j : Bool) : ℝ := if j then 1 else -1

noncomputable def M_rec (δ φ : ℝ) : ℕ → ℕ → Bool → ℝ
  | 0, 0, _ => 1
  | 0, _ + 1, _ => 0
  | k + 1, n, j =>
      if k + 1 ≤ n then
        (1 + zSign j * δ) ^ (k + 1) *
          ∑ m ∈ Finset.range (n - k),
            (-1 : ℝ) ^ m * (Nat.choose (k + m) m : ℝ) *
              (((1 + φ) / 2) * M_rec δ φ m (n - (k + 1)) j +
                ((1 - φ) / 2) * M_rec δ φ m (n - (k + 1)) (!j))
      else 0
termination_by _ n _ => n

noncomputable def r_rec (δ φ : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else (1 / 2 : ℝ) * ∑ j ∈ Finset.range n,
    ((-1 : ℝ) ^ (j + 2) / ((j + 1 : ℕ) : ℝ)) *
      (M_rec δ φ (j + 1) n false + M_rec δ φ (j + 1) n true)

noncomputable def e_rec (δ φ : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 =>
      -(1 / ((n + 1 : ℕ) : ℝ)) * ∑ j ∈ Finset.range (n + 1),
        ((j + 1 : ℕ) : ℝ) * r_rec δ φ (j + 1) * e_rec δ φ (n - j)
termination_by n => n

noncomputable def g_rec (δ φ : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else -e_rec δ φ n - 2 * (n : ℝ) * r_rec δ φ n +
    ∑ j ∈ Finset.range n, e_rec δ φ (n - 1 - j) * (((j + 1 : ℕ) : ℝ) * r_rec δ φ (j + 1))

/-! ## G1 -/

theorem derivative_identity (s R' : ℝ) (c_star R_real : ℝ → ℝ)
    (hs0 : 0 < s) (hs1 : s < 1)
    (hc_pos : 0 < c_star s) (hc_cont : ContinuousAt c_star s)
    (hR_sderiv : HasStrictDerivAt R_real R' (c_star s / s ^ 2))
    (hV_pos : 0 < (c_star s / s ^ 2) * R')
    (h_thresh : ∀ᶠ u in 𝓝 s, Real.log u + R_real (c_star u / u ^ 2) = 0) :
    let q := c_star s / s ^ 2
    let V := q * R'
    (∀ c h w_prev : ℝ, 1 + w_prev ≠ 0 →
      s + c * h / (s * (1 + w_prev)) = s * (1 + (c / s ^ 2) * h / (1 + w_prev)))
      ∧ 0 < q
      ∧ s = Real.exp (-R_real q)
      ∧ HasDerivAt (fun u => Real.log (c_star u / (1 - u)))
          (-Gfun (R_real q) V / (s * (1 - s) * V)) s
      ∧ (-Gfun (R_real q) V / (s * (1 - s) * V) < 0 ↔ 0 < Gfun (R_real q) V) := by
  dsimp only
  set q := c_star s / s ^ 2
  set V := q * R'
  have hs_ne : s ≠ 0 := hs0.ne'
  have hq_pos : 0 < q := div_pos hc_pos (pow_pos hs0 2)
  have h1s_pos : 0 < 1 - s := sub_pos.mpr hs1
  have h1s_ne : 1 - s ≠ 0 := h1s_pos.ne'
  have hc_ne : c_star s ≠ 0 := hc_pos.ne'
  have hV_ne : V ≠ 0 := hV_pos.ne'
  have hR'_pos : 0 < R' := (mul_pos_iff_of_pos_left hq_pos).mp hV_pos
  have hR'_ne : R' ≠ 0 := hR'_pos.ne'
  -- Scaling identity
  have hscale : ∀ c h w_prev : ℝ, 1 + w_prev ≠ 0 →
      s + c * h / (s * (1 + w_prev)) = s * (1 + (c / s ^ 2) * h / (1 + w_prev)) := by
    intro c h w_prev hw
    field_simp
  -- At `u = s`: `log s + R_real q = 0`
  have h0_at_s : Real.log s + R_real q = 0 := h_thresh.self_of_nhds
  have hRq_eq : R_real q = -Real.log s := by linarith
  have hs_exp : s = Real.exp (-R_real q) := by
    rw [hRq_eq, neg_neg, Real.exp_log hs0]
  -- Local inverse of `R_real` near `q` proves `HasDerivAt c_star` at `s`
  let invR := hR_sderiv.localInverse R_real R' q hR'_ne
  have hinv_deriv : HasDerivAt invR R'⁻¹ (-Real.log s) := by
    rw [← hRq_eq]
    exact (hR_sderiv.to_localInverse hR'_ne).hasDerivAt
  have hq_cont : ContinuousAt (fun u : ℝ => c_star u / u ^ 2) s :=
    hc_cont.div (continuousAt_id.pow 2) (pow_ne_zero 2 hs_ne)
  have hev_left : ∀ᶠ u in 𝓝 s, invR (R_real (c_star u / u ^ 2)) = c_star u / u ^ 2 :=
    hq_cont.tendsto.eventually (hR_sderiv.eventually_left_inverse hR'_ne)
  have hev_pos : ∀ᶠ u in 𝓝 s, (0 : ℝ) < u := eventually_gt_nhds hs0
  have hev_cstar : (fun u : ℝ => u ^ 2 * invR (-Real.log u)) =ᶠ[𝓝 s] c_star := by
    filter_upwards [h_thresh, hev_left, hev_pos] with u hth hleft hu0
    have hRu : R_real (c_star u / u ^ 2) = -Real.log u := by linarith
    rw [← hRu, hleft]
    have hu_ne : u ≠ 0 := hu0.ne'
    field_simp
  have hneg_log_deriv : HasDerivAt (fun u : ℝ => -Real.log u) (-s⁻¹) s :=
    (Real.hasDerivAt_log hs_ne).neg
  have hcomp_inv : HasDerivAt (fun u : ℝ => invR (-Real.log u)) (R'⁻¹ * (-s⁻¹)) s :=
    hinv_deriv.comp s hneg_log_deriv
  have hpow2 : HasDerivAt (fun u : ℝ => u ^ 2) (2 * s) s := by
    simpa using hasDerivAt_pow 2 s
  have hc_star' :
      HasDerivAt c_star
        ((2 * s) * invR (-Real.log s) + s ^ 2 * (R'⁻¹ * (-s⁻¹))) s :=
    (hpow2.mul hcomp_inv).congr_of_eventuallyEq hev_cstar.symm
  have hinv_at_s : invR (-Real.log s) = q := by
    rw [← hRq_eq]
    exact (hR_sderiv.eventually_left_inverse hR'_ne).self_of_nhds
  rw [hinv_at_s] at hc_star'
  -- Differentiate `u ↦ log(c_star u / (1 - u))`
  set c_star' := (2 * s) * q + s ^ 2 * (R'⁻¹ * (-s⁻¹))
  have h1u : HasDerivAt (fun u : ℝ => 1 - u) (-1) s := by
    simpa using (hasDerivAt_const s (1 : ℝ)).fun_sub (hasDerivAt_id s)
  have hquot : HasDerivAt (fun u => c_star u / (1 - u))
      ((c_star' * (1 - s) - c_star s * (-1)) / (1 - s) ^ 2) s :=
    hc_star'.div h1u h1s_ne
  have hD_raw : HasDerivAt (fun u => Real.log (c_star u / (1 - u)))
      ((c_star s / (1 - s))⁻¹ * ((c_star' * (1 - s) - c_star s * (-1)) / (1 - s) ^ 2)) s := by
    simpa [Function.comp_def] using
      (Real.hasDerivAt_log (div_pos hc_pos h1s_pos).ne').comp s hquot
  have hD_eq :
      (c_star s / (1 - s))⁻¹ * ((c_star' * (1 - s) - c_star s * (-1)) / (1 - s) ^ 2) =
        -Gfun (R_real q) V / (s * (1 - s) * V) := by
    unfold Gfun
    rw [← hs_exp]
    dsimp [c_star', V, q]
    field_simp
    ring
  rw [hD_eq] at hD_raw
  have hsign : -Gfun (R_real q) V / (s * (1 - s) * V) < 0 ↔ 0 < Gfun (R_real q) V := by
    have hden_pos : 0 < s * (1 - s) * V := mul_pos (mul_pos hs0 h1s_pos) hV_pos
    rw [div_lt_iff₀ hden_pos, zero_mul, neg_lt_zero]
  exact ⟨hscale, hq_pos, hs_exp, hD_raw, hsign⟩


private lemma M_0_0_false (δ φ : ℝ) : M_rec δ φ 0 0 false = 1 := by rw [M_rec]

private lemma M_0_0_true (δ φ : ℝ) : M_rec δ φ 0 0 true = 1 := by rw [M_rec]

private lemma M_1_0_false (δ φ : ℝ) : M_rec δ φ 0 1 false = 0 := by rw [M_rec]

private lemma M_1_0_true (δ φ : ℝ) : M_rec δ φ 0 1 true = 0 := by rw [M_rec]

private lemma M_2_0_false (δ φ : ℝ) : M_rec δ φ 0 2 false = 0 := by rw [M_rec]

private lemma M_2_0_true (δ φ : ℝ) : M_rec δ φ 0 2 true = 0 := by rw [M_rec]

private lemma M_3_0_false (δ φ : ℝ) : M_rec δ φ 0 3 false = 0 := by rw [M_rec]

private lemma M_3_0_true (δ φ : ℝ) : M_rec δ φ 0 3 true = 0 := by rw [M_rec]

private lemma M_4_0_false (δ φ : ℝ) : M_rec δ φ 0 4 false = 0 := by rw [M_rec]

private lemma M_4_0_true (δ φ : ℝ) : M_rec δ φ 0 4 true = 0 := by rw [M_rec]

private lemma M_5_0_false (δ φ : ℝ) : M_rec δ φ 0 5 false = 0 := by rw [M_rec]

private lemma M_5_0_true (δ φ : ℝ) : M_rec δ φ 0 5 true = 0 := by rw [M_rec]

private lemma M_6_0_false (δ φ : ℝ) : M_rec δ φ 0 6 false = 0 := by rw [M_rec]

private lemma M_6_0_true (δ φ : ℝ) : M_rec δ φ 0 6 true = 0 := by rw [M_rec]

private lemma M_7_0_false (δ φ : ℝ) : M_rec δ φ 0 7 false = 0 := by rw [M_rec]

private lemma M_7_0_true (δ φ : ℝ) : M_rec δ φ 0 7 true = 0 := by rw [M_rec]

private lemma M_1_1_false (δ φ : ℝ) :
    M_rec δ φ 1 1 false =
      (-1 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_1_1_true (δ φ : ℝ) :
    M_rec δ φ 1 1 true =
      (1 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_2_1_false (δ φ : ℝ) :
    M_rec δ φ 1 2 false =
      (-1 : ℝ) * δ ^ 2 * φ + (1 : ℝ) * δ * φ + (1 : ℝ) * δ + (-1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_2_1_true (δ φ : ℝ) :
    M_rec δ φ 1 2 true =
      (-1 : ℝ) * δ ^ 2 * φ + (-1 : ℝ) * δ * φ + (-1 : ℝ) * δ + (-1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_2_2_false (δ φ : ℝ) :
    M_rec δ φ 2 2 false =
      (1 : ℝ) * δ ^ 2 + (-2 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_2_2_true (δ φ : ℝ) :
    M_rec δ φ 2 2 true =
      (1 : ℝ) * δ ^ 2 + (2 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_3_1_false (δ φ : ℝ) :
    M_rec δ φ 1 3 false =
      (-1 : ℝ) * δ ^ 3 * φ + (-1 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 2 + (4 : ℝ) * δ ^ 2 * φ
      + (1 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 2 + (-3 : ℝ) * δ * φ + (-2 : ℝ) * δ + (2 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_3_1_true (δ φ : ℝ) :
    M_rec δ φ 1 3 true =
      (1 : ℝ) * δ ^ 3 * φ + (1 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 2 + (4 : ℝ) * δ ^ 2 * φ
      + (1 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 2 + (3 : ℝ) * δ * φ + (2 : ℝ) * δ + (2 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_3_2_false (δ φ : ℝ) :
    M_rec δ φ 2 3 false =
      (2 : ℝ) * δ ^ 3 * φ + (-4 : ℝ) * δ ^ 2 * φ + (-2 : ℝ) * δ ^ 2 + (2 : ℝ) * δ * φ
      + (4 : ℝ) * δ + (-2 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_3_2_true (δ φ : ℝ) :
    M_rec δ φ 2 3 true =
      (-2 : ℝ) * δ ^ 3 * φ + (-4 : ℝ) * δ ^ 2 * φ + (-2 : ℝ) * δ ^ 2 + (-2 : ℝ) * δ * φ
      + (-4 : ℝ) * δ + (-2 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_3_3_false (δ φ : ℝ) :
    M_rec δ φ 3 3 false =
      (-1 : ℝ) * δ ^ 3 + (3 : ℝ) * δ ^ 2 + (-3 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_3_3_true (δ φ : ℝ) :
    M_rec δ φ 3 3 true =
      (1 : ℝ) * δ ^ 3 + (3 : ℝ) * δ ^ 2 + (3 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_4_1_false (δ φ : ℝ) :
    M_rec δ φ 1 4 false =
      (-3 : ℝ) * δ ^ 4 * φ ^ 2 + (-2 : ℝ) * δ ^ 4 * φ + (4 : ℝ) * δ ^ 3 * φ ^ 2
      + (10 : ℝ) * δ ^ 3 * φ + (6 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 3
      + (-6 : ℝ) * δ ^ 2 * φ ^ 2 + (-17 : ℝ) * δ ^ 2 * φ + (-6 : ℝ) * δ ^ 2
      + (1 : ℝ) * δ * φ ^ 3 + (5 : ℝ) * δ * φ ^ 2 + (9 : ℝ) * δ * φ + (5 : ℝ) * δ + (-5 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_4_1_true (δ φ : ℝ) :
    M_rec δ φ 1 4 true =
      (-3 : ℝ) * δ ^ 4 * φ ^ 2 + (-2 : ℝ) * δ ^ 4 * φ + (-4 : ℝ) * δ ^ 3 * φ ^ 2
      + (-10 : ℝ) * δ ^ 3 * φ + (-6 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 3
      + (-6 : ℝ) * δ ^ 2 * φ ^ 2 + (-17 : ℝ) * δ ^ 2 * φ + (-6 : ℝ) * δ ^ 2
      + (-1 : ℝ) * δ * φ ^ 3 + (-5 : ℝ) * δ * φ ^ 2 + (-9 : ℝ) * δ * φ + (-5 : ℝ) * δ + (-5 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_4_2_false (δ φ : ℝ) :
    M_rec δ φ 2 4 false =
      (2 : ℝ) * δ ^ 4 * φ + (3 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 2 + (-12 : ℝ) * δ ^ 3 * φ
      + (-6 : ℝ) * δ ^ 3 + (4 : ℝ) * δ ^ 2 * φ ^ 2 + (18 : ℝ) * δ ^ 2 * φ + (8 : ℝ) * δ ^ 2
      + (-2 : ℝ) * δ * φ ^ 2 + (-8 : ℝ) * δ * φ + (-10 : ℝ) * δ + (5 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_4_2_true (δ φ : ℝ) :
    M_rec δ φ 2 4 true =
      (2 : ℝ) * δ ^ 4 * φ + (3 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 2 + (12 : ℝ) * δ ^ 3 * φ
      + (6 : ℝ) * δ ^ 3 + (4 : ℝ) * δ ^ 2 * φ ^ 2 + (18 : ℝ) * δ ^ 2 * φ + (8 : ℝ) * δ ^ 2
      + (2 : ℝ) * δ * φ ^ 2 + (8 : ℝ) * δ * φ + (10 : ℝ) * δ + (5 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_4_3_false (δ φ : ℝ) :
    M_rec δ φ 3 4 false =
      (-3 : ℝ) * δ ^ 4 * φ + (9 : ℝ) * δ ^ 3 * φ + (3 : ℝ) * δ ^ 3 + (-9 : ℝ) * δ ^ 2 * φ
      + (-9 : ℝ) * δ ^ 2 + (3 : ℝ) * δ * φ + (9 : ℝ) * δ + (-3 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_4_3_true (δ φ : ℝ) :
    M_rec δ φ 3 4 true =
      (-3 : ℝ) * δ ^ 4 * φ + (-9 : ℝ) * δ ^ 3 * φ + (-3 : ℝ) * δ ^ 3 + (-9 : ℝ) * δ ^ 2 * φ
      + (-9 : ℝ) * δ ^ 2 + (-3 : ℝ) * δ * φ + (-9 : ℝ) * δ + (-3 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_4_4_false (δ φ : ℝ) :
    M_rec δ φ 4 4 false =
      (1 : ℝ) * δ ^ 4 + (-4 : ℝ) * δ ^ 3 + (6 : ℝ) * δ ^ 2 + (-4 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_4_4_true (δ φ : ℝ) :
    M_rec δ φ 4 4 true =
      (1 : ℝ) * δ ^ 4 + (4 : ℝ) * δ ^ 3 + (6 : ℝ) * δ ^ 2 + (4 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_5_1_false (δ φ : ℝ) :
    M_rec δ φ 1 5 false =
      (-3 : ℝ) * δ ^ 5 * φ ^ 2 + (-7 : ℝ) * δ ^ 5 * φ + (-4 : ℝ) * δ ^ 5
      + (6 : ℝ) * δ ^ 4 * φ ^ 3 + (34 : ℝ) * δ ^ 4 * φ ^ 2 + (26 : ℝ) * δ ^ 4 * φ
      + (4 : ℝ) * δ ^ 4 + (-7 : ℝ) * δ ^ 3 * φ ^ 3 + (-41 : ℝ) * δ ^ 3 * φ ^ 2
      + (-63 : ℝ) * δ ^ 3 * φ + (-29 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 4
      + (8 : ℝ) * δ ^ 2 * φ ^ 3 + (30 : ℝ) * δ ^ 2 * φ ^ 2 + (72 : ℝ) * δ ^ 2 * φ
      + (29 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 4 + (-7 : ℝ) * δ * φ ^ 3 + (-20 : ℝ) * δ * φ ^ 2
      + (-28 : ℝ) * δ * φ + (-14 : ℝ) * δ + (14 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_5_1_true (δ φ : ℝ) :
    M_rec δ φ 1 5 true =
      (3 : ℝ) * δ ^ 5 * φ ^ 2 + (7 : ℝ) * δ ^ 5 * φ + (4 : ℝ) * δ ^ 5 + (6 : ℝ) * δ ^ 4 * φ ^ 3
      + (34 : ℝ) * δ ^ 4 * φ ^ 2 + (26 : ℝ) * δ ^ 4 * φ + (4 : ℝ) * δ ^ 4
      + (7 : ℝ) * δ ^ 3 * φ ^ 3 + (41 : ℝ) * δ ^ 3 * φ ^ 2 + (63 : ℝ) * δ ^ 3 * φ
      + (29 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 4 + (8 : ℝ) * δ ^ 2 * φ ^ 3
      + (30 : ℝ) * δ ^ 2 * φ ^ 2 + (72 : ℝ) * δ ^ 2 * φ + (29 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 4
      + (7 : ℝ) * δ * φ ^ 3 + (20 : ℝ) * δ * φ ^ 2 + (28 : ℝ) * δ * φ + (14 : ℝ) * δ + (14 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_5_2_false (δ φ : ℝ) :
    M_rec δ φ 2 5 false =
      (8 : ℝ) * δ ^ 5 * φ ^ 2 + (6 : ℝ) * δ ^ 5 * φ + (-18 : ℝ) * δ ^ 4 * φ ^ 2
      + (-32 : ℝ) * δ ^ 4 * φ + (-20 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 3
      + (24 : ℝ) * δ ^ 3 * φ ^ 2 + (74 : ℝ) * δ ^ 3 * φ + (40 : ℝ) * δ ^ 3
      + (-4 : ℝ) * δ ^ 2 * φ ^ 3 + (-26 : ℝ) * δ ^ 2 * φ ^ 2 + (-76 : ℝ) * δ ^ 2 * φ
      + (-34 : ℝ) * δ ^ 2 + (2 : ℝ) * δ * φ ^ 3 + (12 : ℝ) * δ * φ ^ 2 + (28 : ℝ) * δ * φ
      + (28 : ℝ) * δ + (-14 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_5_2_true (δ φ : ℝ) :
    M_rec δ φ 2 5 true =
      (-8 : ℝ) * δ ^ 5 * φ ^ 2 + (-6 : ℝ) * δ ^ 5 * φ + (-18 : ℝ) * δ ^ 4 * φ ^ 2
      + (-32 : ℝ) * δ ^ 4 * φ + (-20 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 3
      + (-24 : ℝ) * δ ^ 3 * φ ^ 2 + (-74 : ℝ) * δ ^ 3 * φ + (-40 : ℝ) * δ ^ 3
      + (-4 : ℝ) * δ ^ 2 * φ ^ 3 + (-26 : ℝ) * δ ^ 2 * φ ^ 2 + (-76 : ℝ) * δ ^ 2 * φ
      + (-34 : ℝ) * δ ^ 2 + (-2 : ℝ) * δ * φ ^ 3 + (-12 : ℝ) * δ * φ ^ 2 + (-28 : ℝ) * δ * φ
      + (-28 : ℝ) * δ + (-14 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_5_3_false (δ φ : ℝ) :
    M_rec δ φ 3 5 false =
      (-3 : ℝ) * δ ^ 5 * φ + (-6 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 2 + (24 : ℝ) * δ ^ 4 * φ
      + (18 : ℝ) * δ ^ 4 + (-9 : ℝ) * δ ^ 3 * φ ^ 2 + (-54 : ℝ) * δ ^ 3 * φ + (-27 : ℝ) * δ ^ 3
      + (9 : ℝ) * δ ^ 2 * φ ^ 2 + (48 : ℝ) * δ ^ 2 * φ + (33 : ℝ) * δ ^ 2 + (-3 : ℝ) * δ * φ ^ 2
      + (-15 : ℝ) * δ * φ + (-27 : ℝ) * δ + (9 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_5_3_true (δ φ : ℝ) :
    M_rec δ φ 3 5 true =
      (3 : ℝ) * δ ^ 5 * φ + (6 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 2 + (24 : ℝ) * δ ^ 4 * φ
      + (18 : ℝ) * δ ^ 4 + (9 : ℝ) * δ ^ 3 * φ ^ 2 + (54 : ℝ) * δ ^ 3 * φ + (27 : ℝ) * δ ^ 3
      + (9 : ℝ) * δ ^ 2 * φ ^ 2 + (48 : ℝ) * δ ^ 2 * φ + (33 : ℝ) * δ ^ 2 + (3 : ℝ) * δ * φ ^ 2
      + (15 : ℝ) * δ * φ + (27 : ℝ) * δ + (9 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_5_4_false (δ φ : ℝ) :
    M_rec δ φ 4 5 false =
      (4 : ℝ) * δ ^ 5 * φ + (-16 : ℝ) * δ ^ 4 * φ + (-4 : ℝ) * δ ^ 4 + (24 : ℝ) * δ ^ 3 * φ
      + (16 : ℝ) * δ ^ 3 + (-16 : ℝ) * δ ^ 2 * φ + (-24 : ℝ) * δ ^ 2 + (4 : ℝ) * δ * φ
      + (16 : ℝ) * δ + (-4 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_5_4_true (δ φ : ℝ) :
    M_rec δ φ 4 5 true =
      (-4 : ℝ) * δ ^ 5 * φ + (-16 : ℝ) * δ ^ 4 * φ + (-4 : ℝ) * δ ^ 4 + (-24 : ℝ) * δ ^ 3 * φ
      + (-16 : ℝ) * δ ^ 3 + (-16 : ℝ) * δ ^ 2 * φ + (-24 : ℝ) * δ ^ 2 + (-4 : ℝ) * δ * φ
      + (-16 : ℝ) * δ + (-4 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_5_5_false (δ φ : ℝ) :
    M_rec δ φ 5 5 false =
      (-1 : ℝ) * δ ^ 5 + (5 : ℝ) * δ ^ 4 + (-10 : ℝ) * δ ^ 3 + (10 : ℝ) * δ ^ 2 + (-5 : ℝ) * δ
      + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_5_5_true (δ φ : ℝ) :
    M_rec δ φ 5 5 true =
      (1 : ℝ) * δ ^ 5 + (5 : ℝ) * δ ^ 4 + (10 : ℝ) * δ ^ 3 + (10 : ℝ) * δ ^ 2 + (5 : ℝ) * δ
      + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_6_1_false (δ φ : ℝ) :
    M_rec δ φ 1 6 false =
      (-11 : ℝ) * δ ^ 6 * φ ^ 3 + (-20 : ℝ) * δ ^ 6 * φ ^ 2 + (-11 : ℝ) * δ ^ 6 * φ
      + (17 : ℝ) * δ ^ 5 * φ ^ 3 + (75 : ℝ) * δ ^ 5 * φ ^ 2 + (109 : ℝ) * δ ^ 5 * φ
      + (51 : ℝ) * δ ^ 5 + (-9 : ℝ) * δ ^ 4 * φ ^ 4 + (-80 : ℝ) * δ ^ 4 * φ ^ 3
      + (-270 : ℝ) * δ ^ 4 * φ ^ 2 + (-220 : ℝ) * δ ^ 4 * φ + (-51 : ℝ) * δ ^ 4
      + (10 : ℝ) * δ ^ 3 * φ ^ 4 + (86 : ℝ) * δ ^ 3 * φ ^ 3 + (280 : ℝ) * δ ^ 3 * φ ^ 2
      + (334 : ℝ) * δ ^ 3 * φ + (130 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 5
      + (-10 : ℝ) * δ ^ 2 * φ ^ 4 + (-47 : ℝ) * δ ^ 2 * φ ^ 3 + (-140 : ℝ) * δ ^ 2 * φ ^ 2
      + (-302 : ℝ) * δ ^ 2 * φ + (-130 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 5 + (9 : ℝ) * δ * φ ^ 4
      + (35 : ℝ) * δ * φ ^ 3 + (75 : ℝ) * δ * φ ^ 2 + (90 : ℝ) * δ * φ + (42 : ℝ) * δ
      + (-42 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_6_1_true (δ φ : ℝ) :
    M_rec δ φ 1 6 true =
      (-11 : ℝ) * δ ^ 6 * φ ^ 3 + (-20 : ℝ) * δ ^ 6 * φ ^ 2 + (-11 : ℝ) * δ ^ 6 * φ
      + (-17 : ℝ) * δ ^ 5 * φ ^ 3 + (-75 : ℝ) * δ ^ 5 * φ ^ 2 + (-109 : ℝ) * δ ^ 5 * φ
      + (-51 : ℝ) * δ ^ 5 + (-9 : ℝ) * δ ^ 4 * φ ^ 4 + (-80 : ℝ) * δ ^ 4 * φ ^ 3
      + (-270 : ℝ) * δ ^ 4 * φ ^ 2 + (-220 : ℝ) * δ ^ 4 * φ + (-51 : ℝ) * δ ^ 4
      + (-10 : ℝ) * δ ^ 3 * φ ^ 4 + (-86 : ℝ) * δ ^ 3 * φ ^ 3 + (-280 : ℝ) * δ ^ 3 * φ ^ 2
      + (-334 : ℝ) * δ ^ 3 * φ + (-130 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 5
      + (-10 : ℝ) * δ ^ 2 * φ ^ 4 + (-47 : ℝ) * δ ^ 2 * φ ^ 3 + (-140 : ℝ) * δ ^ 2 * φ ^ 2
      + (-302 : ℝ) * δ ^ 2 * φ + (-130 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 5
      + (-9 : ℝ) * δ * φ ^ 4 + (-35 : ℝ) * δ * φ ^ 3 + (-75 : ℝ) * δ * φ ^ 2 + (-90 : ℝ) * δ * φ
      + (-42 : ℝ) * δ + (-42 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_6_2_false (δ φ : ℝ) :
    M_rec δ φ 2 6 false =
      (6 : ℝ) * δ ^ 6 * φ ^ 2 + (22 : ℝ) * δ ^ 6 * φ + (14 : ℝ) * δ ^ 6
      + (-14 : ℝ) * δ ^ 5 * φ ^ 3 + (-104 : ℝ) * δ ^ 5 * φ ^ 2 + (-106 : ℝ) * δ ^ 5 * φ
      + (-28 : ℝ) * δ ^ 5 + (30 : ℝ) * δ ^ 4 * φ ^ 3 + (214 : ℝ) * δ ^ 4 * φ ^ 2
      + (270 : ℝ) * δ ^ 4 * φ + (116 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 4
      + (-34 : ℝ) * δ ^ 3 * φ ^ 3 + (-194 : ℝ) * δ ^ 3 * φ ^ 2 + (-406 : ℝ) * δ ^ 3 * φ
      + (-204 : ℝ) * δ ^ 3 + (4 : ℝ) * δ ^ 2 * φ ^ 4 + (34 : ℝ) * δ ^ 2 * φ ^ 3
      + (132 : ℝ) * δ ^ 2 * φ ^ 2 + (316 : ℝ) * δ ^ 2 * φ + (144 : ℝ) * δ ^ 2
      + (-2 : ℝ) * δ * φ ^ 4 + (-16 : ℝ) * δ * φ ^ 3 + (-54 : ℝ) * δ * φ ^ 2 + (-96 : ℝ) * δ * φ
      + (-84 : ℝ) * δ + (42 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_6_2_true (δ φ : ℝ) :
    M_rec δ φ 2 6 true =
      (6 : ℝ) * δ ^ 6 * φ ^ 2 + (22 : ℝ) * δ ^ 6 * φ + (14 : ℝ) * δ ^ 6
      + (14 : ℝ) * δ ^ 5 * φ ^ 3 + (104 : ℝ) * δ ^ 5 * φ ^ 2 + (106 : ℝ) * δ ^ 5 * φ
      + (28 : ℝ) * δ ^ 5 + (30 : ℝ) * δ ^ 4 * φ ^ 3 + (214 : ℝ) * δ ^ 4 * φ ^ 2
      + (270 : ℝ) * δ ^ 4 * φ + (116 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 4
      + (34 : ℝ) * δ ^ 3 * φ ^ 3 + (194 : ℝ) * δ ^ 3 * φ ^ 2 + (406 : ℝ) * δ ^ 3 * φ
      + (204 : ℝ) * δ ^ 3 + (4 : ℝ) * δ ^ 2 * φ ^ 4 + (34 : ℝ) * δ ^ 2 * φ ^ 3
      + (132 : ℝ) * δ ^ 2 * φ ^ 2 + (316 : ℝ) * δ ^ 2 * φ + (144 : ℝ) * δ ^ 2
      + (2 : ℝ) * δ * φ ^ 4 + (16 : ℝ) * δ * φ ^ 3 + (54 : ℝ) * δ * φ ^ 2 + (96 : ℝ) * δ * φ
      + (84 : ℝ) * δ + (42 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_6_3_false (δ φ : ℝ) :
    M_rec δ φ 3 6 false =
      (-15 : ℝ) * δ ^ 6 * φ ^ 2 + (-13 : ℝ) * δ ^ 6 * φ + (48 : ℝ) * δ ^ 5 * φ ^ 2
      + (75 : ℝ) * δ ^ 5 * φ + (45 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 3
      + (-75 : ℝ) * δ ^ 4 * φ ^ 2 + (-207 : ℝ) * δ ^ 4 * φ + (-135 : ℝ) * δ ^ 4
      + (9 : ℝ) * δ ^ 3 * φ ^ 3 + (87 : ℝ) * δ ^ 3 * φ ^ 2 + (301 : ℝ) * δ ^ 3 * φ
      + (163 : ℝ) * δ ^ 3 + (-9 : ℝ) * δ ^ 2 * φ ^ 3 + (-66 : ℝ) * δ ^ 2 * φ ^ 2
      + (-216 : ℝ) * δ ^ 2 * φ + (-129 : ℝ) * δ ^ 2 + (3 : ℝ) * δ * φ ^ 3 + (21 : ℝ) * δ * φ ^ 2
      + (60 : ℝ) * δ * φ + (84 : ℝ) * δ + (-28 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_6_3_true (δ φ : ℝ) :
    M_rec δ φ 3 6 true =
      (-15 : ℝ) * δ ^ 6 * φ ^ 2 + (-13 : ℝ) * δ ^ 6 * φ + (-48 : ℝ) * δ ^ 5 * φ ^ 2
      + (-75 : ℝ) * δ ^ 5 * φ + (-45 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 3
      + (-75 : ℝ) * δ ^ 4 * φ ^ 2 + (-207 : ℝ) * δ ^ 4 * φ + (-135 : ℝ) * δ ^ 4
      + (-9 : ℝ) * δ ^ 3 * φ ^ 3 + (-87 : ℝ) * δ ^ 3 * φ ^ 2 + (-301 : ℝ) * δ ^ 3 * φ
      + (-163 : ℝ) * δ ^ 3 + (-9 : ℝ) * δ ^ 2 * φ ^ 3 + (-66 : ℝ) * δ ^ 2 * φ ^ 2
      + (-216 : ℝ) * δ ^ 2 * φ + (-129 : ℝ) * δ ^ 2 + (-3 : ℝ) * δ * φ ^ 3
      + (-21 : ℝ) * δ * φ ^ 2 + (-60 : ℝ) * δ * φ + (-84 : ℝ) * δ + (-28 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_6_4_false (δ φ : ℝ) :
    M_rec δ φ 4 6 false =
      (4 : ℝ) * δ ^ 6 * φ + (10 : ℝ) * δ ^ 6 + (-4 : ℝ) * δ ^ 5 * φ ^ 2 + (-40 : ℝ) * δ ^ 5 * φ
      + (-40 : ℝ) * δ ^ 5 + (16 : ℝ) * δ ^ 4 * φ ^ 2 + (120 : ℝ) * δ ^ 4 * φ + (74 : ℝ) * δ ^ 4
      + (-24 : ℝ) * δ ^ 3 * φ ^ 2 + (-160 : ℝ) * δ ^ 3 * φ + (-96 : ℝ) * δ ^ 3
      + (16 : ℝ) * δ ^ 2 * φ ^ 2 + (100 : ℝ) * δ ^ 2 * φ + (94 : ℝ) * δ ^ 2
      + (-4 : ℝ) * δ * φ ^ 2 + (-24 : ℝ) * δ * φ + (-56 : ℝ) * δ + (14 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_6_4_true (δ φ : ℝ) :
    M_rec δ φ 4 6 true =
      (4 : ℝ) * δ ^ 6 * φ + (10 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 2 + (40 : ℝ) * δ ^ 5 * φ
      + (40 : ℝ) * δ ^ 5 + (16 : ℝ) * δ ^ 4 * φ ^ 2 + (120 : ℝ) * δ ^ 4 * φ + (74 : ℝ) * δ ^ 4
      + (24 : ℝ) * δ ^ 3 * φ ^ 2 + (160 : ℝ) * δ ^ 3 * φ + (96 : ℝ) * δ ^ 3
      + (16 : ℝ) * δ ^ 2 * φ ^ 2 + (100 : ℝ) * δ ^ 2 * φ + (94 : ℝ) * δ ^ 2
      + (4 : ℝ) * δ * φ ^ 2 + (24 : ℝ) * δ * φ + (56 : ℝ) * δ + (14 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_6_5_false (δ φ : ℝ) :
    M_rec δ φ 5 6 false =
      (-5 : ℝ) * δ ^ 6 * φ + (25 : ℝ) * δ ^ 5 * φ + (5 : ℝ) * δ ^ 5 + (-50 : ℝ) * δ ^ 4 * φ
      + (-25 : ℝ) * δ ^ 4 + (50 : ℝ) * δ ^ 3 * φ + (50 : ℝ) * δ ^ 3 + (-25 : ℝ) * δ ^ 2 * φ
      + (-50 : ℝ) * δ ^ 2 + (5 : ℝ) * δ * φ + (25 : ℝ) * δ + (-5 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_6_5_true (δ φ : ℝ) :
    M_rec δ φ 5 6 true =
      (-5 : ℝ) * δ ^ 6 * φ + (-25 : ℝ) * δ ^ 5 * φ + (-5 : ℝ) * δ ^ 5 + (-50 : ℝ) * δ ^ 4 * φ
      + (-25 : ℝ) * δ ^ 4 + (-50 : ℝ) * δ ^ 3 * φ + (-50 : ℝ) * δ ^ 3 + (-25 : ℝ) * δ ^ 2 * φ
      + (-50 : ℝ) * δ ^ 2 + (-5 : ℝ) * δ * φ + (-25 : ℝ) * δ + (-5 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_6_6_false (δ φ : ℝ) :
    M_rec δ φ 6 6 false =
      (1 : ℝ) * δ ^ 6 + (-6 : ℝ) * δ ^ 5 + (15 : ℝ) * δ ^ 4 + (-20 : ℝ) * δ ^ 3
      + (15 : ℝ) * δ ^ 2 + (-6 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_6_6_true (δ φ : ℝ) :
    M_rec δ φ 6 6 true =
      (1 : ℝ) * δ ^ 6 + (6 : ℝ) * δ ^ 5 + (15 : ℝ) * δ ^ 4 + (20 : ℝ) * δ ^ 3 + (15 : ℝ) * δ ^ 2
      + (6 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_7_1_false (δ φ : ℝ) :
    M_rec δ φ 1 7 false =
      (-11 : ℝ) * δ ^ 7 * φ ^ 3 + (-41 : ℝ) * δ ^ 7 * φ ^ 2 + (-55 : ℝ) * δ ^ 7 * φ
      + (-25 : ℝ) * δ ^ 7 + (31 : ℝ) * δ ^ 6 * φ ^ 4 + (242 : ℝ) * δ ^ 6 * φ ^ 3
      + (396 : ℝ) * δ ^ 6 * φ ^ 2 + (230 : ℝ) * δ ^ 6 * φ + (25 : ℝ) * δ ^ 6
      + (-40 : ℝ) * δ ^ 5 * φ ^ 4 + (-344 : ℝ) * δ ^ 5 * φ ^ 3 + (-930 : ℝ) * δ ^ 5 * φ ^ 2
      + (-1042 : ℝ) * δ ^ 5 * φ + (-416 : ℝ) * δ ^ 5 + (12 : ℝ) * δ ^ 4 * φ ^ 5
      + (138 : ℝ) * δ ^ 4 * φ ^ 4 + (698 : ℝ) * δ ^ 4 * φ ^ 3 + (1826 : ℝ) * δ ^ 4 * φ ^ 2
      + (1530 : ℝ) * δ ^ 4 * φ + (416 : ℝ) * δ ^ 4 + (-13 : ℝ) * δ ^ 3 * φ ^ 5
      + (-143 : ℝ) * δ ^ 3 * φ ^ 4 + (-675 : ℝ) * δ ^ 3 * φ ^ 3 + (-1605 : ℝ) * δ ^ 3 * φ ^ 2
      + (-1622 : ℝ) * δ ^ 3 * φ + (-562 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 6
      + (12 : ℝ) * δ ^ 2 * φ ^ 5 + (68 : ℝ) * δ ^ 2 * φ ^ 4 + (244 : ℝ) * δ ^ 2 * φ ^ 3
      + (629 : ℝ) * δ ^ 2 * φ ^ 2 + (1256 : ℝ) * δ ^ 2 * φ + (562 : ℝ) * δ ^ 2
      + (-1 : ℝ) * δ * φ ^ 6 + (-11 : ℝ) * δ * φ ^ 5 + (-54 : ℝ) * δ * φ ^ 4
      + (-154 : ℝ) * δ * φ ^ 3 + (-275 : ℝ) * δ * φ ^ 2 + (-297 : ℝ) * δ * φ + (-132 : ℝ) * δ
      + (132 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_7_1_true (δ φ : ℝ) :
    M_rec δ φ 1 7 true =
      (11 : ℝ) * δ ^ 7 * φ ^ 3 + (41 : ℝ) * δ ^ 7 * φ ^ 2 + (55 : ℝ) * δ ^ 7 * φ
      + (25 : ℝ) * δ ^ 7 + (31 : ℝ) * δ ^ 6 * φ ^ 4 + (242 : ℝ) * δ ^ 6 * φ ^ 3
      + (396 : ℝ) * δ ^ 6 * φ ^ 2 + (230 : ℝ) * δ ^ 6 * φ + (25 : ℝ) * δ ^ 6
      + (40 : ℝ) * δ ^ 5 * φ ^ 4 + (344 : ℝ) * δ ^ 5 * φ ^ 3 + (930 : ℝ) * δ ^ 5 * φ ^ 2
      + (1042 : ℝ) * δ ^ 5 * φ + (416 : ℝ) * δ ^ 5 + (12 : ℝ) * δ ^ 4 * φ ^ 5
      + (138 : ℝ) * δ ^ 4 * φ ^ 4 + (698 : ℝ) * δ ^ 4 * φ ^ 3 + (1826 : ℝ) * δ ^ 4 * φ ^ 2
      + (1530 : ℝ) * δ ^ 4 * φ + (416 : ℝ) * δ ^ 4 + (13 : ℝ) * δ ^ 3 * φ ^ 5
      + (143 : ℝ) * δ ^ 3 * φ ^ 4 + (675 : ℝ) * δ ^ 3 * φ ^ 3 + (1605 : ℝ) * δ ^ 3 * φ ^ 2
      + (1622 : ℝ) * δ ^ 3 * φ + (562 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 6
      + (12 : ℝ) * δ ^ 2 * φ ^ 5 + (68 : ℝ) * δ ^ 2 * φ ^ 4 + (244 : ℝ) * δ ^ 2 * φ ^ 3
      + (629 : ℝ) * δ ^ 2 * φ ^ 2 + (1256 : ℝ) * δ ^ 2 * φ + (562 : ℝ) * δ ^ 2
      + (1 : ℝ) * δ * φ ^ 6 + (11 : ℝ) * δ * φ ^ 5 + (54 : ℝ) * δ * φ ^ 4
      + (154 : ℝ) * δ * φ ^ 3 + (275 : ℝ) * δ * φ ^ 2 + (297 : ℝ) * δ * φ + (132 : ℝ) * δ
      + (132 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_7_2_false (δ φ : ℝ) :
    M_rec δ φ 2 7 false =
      (30 : ℝ) * δ ^ 7 * φ ^ 3 + (64 : ℝ) * δ ^ 7 * φ ^ 2 + (38 : ℝ) * δ ^ 7 * φ
      + (-72 : ℝ) * δ ^ 6 * φ ^ 3 + (-262 : ℝ) * δ ^ 6 * φ ^ 2 + (-400 : ℝ) * δ ^ 6 * φ
      + (-190 : ℝ) * δ ^ 6 + (20 : ℝ) * δ ^ 5 * φ ^ 4 + (244 : ℝ) * δ ^ 5 * φ ^ 3
      + (1016 : ℝ) * δ ^ 5 * φ ^ 2 + (1112 : ℝ) * δ ^ 5 * φ + (380 : ℝ) * δ ^ 5
      + (-42 : ℝ) * δ ^ 4 * φ ^ 4 + (-420 : ℝ) * δ ^ 4 * φ ^ 3 + (-1676 : ℝ) * δ ^ 4 * φ ^ 2
      + (-1820 : ℝ) * δ ^ 4 * φ + (-662 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 5
      + (44 : ℝ) * δ ^ 3 * φ ^ 4 + (334 : ℝ) * δ ^ 3 * φ ^ 3 + (1252 : ℝ) * δ ^ 3 * φ ^ 2
      + (2044 : ℝ) * δ ^ 3 * φ + (944 : ℝ) * δ ^ 3 + (-4 : ℝ) * δ ^ 2 * φ ^ 5
      + (-42 : ℝ) * δ ^ 2 * φ ^ 4 + (-204 : ℝ) * δ ^ 2 * φ ^ 3 + (-614 : ℝ) * δ ^ 2 * φ ^ 2
      + (-1304 : ℝ) * δ ^ 2 * φ + (-604 : ℝ) * δ ^ 2 + (2 : ℝ) * δ * φ ^ 5
      + (20 : ℝ) * δ * φ ^ 4 + (88 : ℝ) * δ * φ ^ 3 + (220 : ℝ) * δ * φ ^ 2 + (330 : ℝ) * δ * φ
      + (264 : ℝ) * δ + (-132 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_7_2_true (δ φ : ℝ) :
    M_rec δ φ 2 7 true =
      (-30 : ℝ) * δ ^ 7 * φ ^ 3 + (-64 : ℝ) * δ ^ 7 * φ ^ 2 + (-38 : ℝ) * δ ^ 7 * φ
      + (-72 : ℝ) * δ ^ 6 * φ ^ 3 + (-262 : ℝ) * δ ^ 6 * φ ^ 2 + (-400 : ℝ) * δ ^ 6 * φ
      + (-190 : ℝ) * δ ^ 6 + (-20 : ℝ) * δ ^ 5 * φ ^ 4 + (-244 : ℝ) * δ ^ 5 * φ ^ 3
      + (-1016 : ℝ) * δ ^ 5 * φ ^ 2 + (-1112 : ℝ) * δ ^ 5 * φ + (-380 : ℝ) * δ ^ 5
      + (-42 : ℝ) * δ ^ 4 * φ ^ 4 + (-420 : ℝ) * δ ^ 4 * φ ^ 3 + (-1676 : ℝ) * δ ^ 4 * φ ^ 2
      + (-1820 : ℝ) * δ ^ 4 * φ + (-662 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 5
      + (-44 : ℝ) * δ ^ 3 * φ ^ 4 + (-334 : ℝ) * δ ^ 3 * φ ^ 3 + (-1252 : ℝ) * δ ^ 3 * φ ^ 2
      + (-2044 : ℝ) * δ ^ 3 * φ + (-944 : ℝ) * δ ^ 3 + (-4 : ℝ) * δ ^ 2 * φ ^ 5
      + (-42 : ℝ) * δ ^ 2 * φ ^ 4 + (-204 : ℝ) * δ ^ 2 * φ ^ 3 + (-614 : ℝ) * δ ^ 2 * φ ^ 2
      + (-1304 : ℝ) * δ ^ 2 * φ + (-604 : ℝ) * δ ^ 2 + (-2 : ℝ) * δ * φ ^ 5
      + (-20 : ℝ) * δ * φ ^ 4 + (-88 : ℝ) * δ * φ ^ 3 + (-220 : ℝ) * δ * φ ^ 2
      + (-330 : ℝ) * δ * φ + (-264 : ℝ) * δ + (-132 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_7_3_false (δ φ : ℝ) :
    M_rec δ φ 3 7 false =
      (-9 : ℝ) * δ ^ 7 * φ ^ 2 + (-48 : ℝ) * δ ^ 7 * φ + (-33 : ℝ) * δ ^ 7
      + (24 : ℝ) * δ ^ 6 * φ ^ 3 + (219 : ℝ) * δ ^ 6 * φ ^ 2 + (288 : ℝ) * δ ^ 6 * φ
      + (99 : ℝ) * δ ^ 6 + (-75 : ℝ) * δ ^ 5 * φ ^ 3 + (-645 : ℝ) * δ ^ 5 * φ ^ 2
      + (-825 : ℝ) * δ ^ 5 * φ + (-345 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 4
      + (108 : ℝ) * δ ^ 4 * φ ^ 3 + (816 : ℝ) * δ ^ 4 * φ ^ 2 + (1452 : ℝ) * δ ^ 4 * φ
      + (771 : ℝ) * δ ^ 4 + (-9 : ℝ) * δ ^ 3 * φ ^ 4 + (-114 : ℝ) * δ ^ 3 * φ ^ 3
      + (-633 : ℝ) * δ ^ 3 * φ ^ 2 + (-1566 : ℝ) * δ ^ 3 * φ + (-828 : ℝ) * δ ^ 3
      + (9 : ℝ) * δ ^ 2 * φ ^ 4 + (84 : ℝ) * δ ^ 2 * φ ^ 3 + (357 : ℝ) * δ ^ 2 * φ ^ 2
      + (924 : ℝ) * δ ^ 2 * φ + (516 : ℝ) * δ ^ 2 + (-3 : ℝ) * δ * φ ^ 4 + (-27 : ℝ) * δ * φ ^ 3
      + (-105 : ℝ) * δ * φ ^ 2 + (-225 : ℝ) * δ * φ + (-270 : ℝ) * δ + (90 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_7_3_true (δ φ : ℝ) :
    M_rec δ φ 3 7 true =
      (9 : ℝ) * δ ^ 7 * φ ^ 2 + (48 : ℝ) * δ ^ 7 * φ + (33 : ℝ) * δ ^ 7
      + (24 : ℝ) * δ ^ 6 * φ ^ 3 + (219 : ℝ) * δ ^ 6 * φ ^ 2 + (288 : ℝ) * δ ^ 6 * φ
      + (99 : ℝ) * δ ^ 6 + (75 : ℝ) * δ ^ 5 * φ ^ 3 + (645 : ℝ) * δ ^ 5 * φ ^ 2
      + (825 : ℝ) * δ ^ 5 * φ + (345 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 4
      + (108 : ℝ) * δ ^ 4 * φ ^ 3 + (816 : ℝ) * δ ^ 4 * φ ^ 2 + (1452 : ℝ) * δ ^ 4 * φ
      + (771 : ℝ) * δ ^ 4 + (9 : ℝ) * δ ^ 3 * φ ^ 4 + (114 : ℝ) * δ ^ 3 * φ ^ 3
      + (633 : ℝ) * δ ^ 3 * φ ^ 2 + (1566 : ℝ) * δ ^ 3 * φ + (828 : ℝ) * δ ^ 3
      + (9 : ℝ) * δ ^ 2 * φ ^ 4 + (84 : ℝ) * δ ^ 2 * φ ^ 3 + (357 : ℝ) * δ ^ 2 * φ ^ 2
      + (924 : ℝ) * δ ^ 2 * φ + (516 : ℝ) * δ ^ 2 + (3 : ℝ) * δ * φ ^ 4 + (27 : ℝ) * δ * φ ^ 3
      + (105 : ℝ) * δ * φ ^ 2 + (225 : ℝ) * δ * φ + (270 : ℝ) * δ + (90 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_7_4_false (δ φ : ℝ) :
    M_rec δ φ 4 7 false =
      (24 : ℝ) * δ ^ 7 * φ ^ 2 + (24 : ℝ) * δ ^ 7 * φ + (-100 : ℝ) * δ ^ 6 * φ ^ 2
      + (-152 : ℝ) * δ ^ 6 * φ + (-84 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 3
      + (192 : ℝ) * δ ^ 5 * φ ^ 2 + (476 : ℝ) * δ ^ 5 * φ + (336 : ℝ) * δ ^ 5
      + (-16 : ℝ) * δ ^ 4 * φ ^ 3 + (-248 : ℝ) * δ ^ 4 * φ ^ 2 + (-864 : ℝ) * δ ^ 4 * φ
      + (-552 : ℝ) * δ ^ 4 + (24 : ℝ) * δ ^ 3 * φ ^ 3 + (232 : ℝ) * δ ^ 3 * φ ^ 2
      + (896 : ℝ) * δ ^ 3 * φ + (528 : ℝ) * δ ^ 3 + (-16 : ℝ) * δ ^ 2 * φ ^ 3
      + (-132 : ℝ) * δ ^ 2 * φ ^ 2 + (-488 : ℝ) * δ ^ 2 * φ + (-372 : ℝ) * δ ^ 2
      + (4 : ℝ) * δ * φ ^ 3 + (32 : ℝ) * δ * φ ^ 2 + (108 : ℝ) * δ * φ + (192 : ℝ) * δ
      + (-48 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_7_4_true (δ φ : ℝ) :
    M_rec δ φ 4 7 true =
      (-24 : ℝ) * δ ^ 7 * φ ^ 2 + (-24 : ℝ) * δ ^ 7 * φ + (-100 : ℝ) * δ ^ 6 * φ ^ 2
      + (-152 : ℝ) * δ ^ 6 * φ + (-84 : ℝ) * δ ^ 6 + (-4 : ℝ) * δ ^ 5 * φ ^ 3
      + (-192 : ℝ) * δ ^ 5 * φ ^ 2 + (-476 : ℝ) * δ ^ 5 * φ + (-336 : ℝ) * δ ^ 5
      + (-16 : ℝ) * δ ^ 4 * φ ^ 3 + (-248 : ℝ) * δ ^ 4 * φ ^ 2 + (-864 : ℝ) * δ ^ 4 * φ
      + (-552 : ℝ) * δ ^ 4 + (-24 : ℝ) * δ ^ 3 * φ ^ 3 + (-232 : ℝ) * δ ^ 3 * φ ^ 2
      + (-896 : ℝ) * δ ^ 3 * φ + (-528 : ℝ) * δ ^ 3 + (-16 : ℝ) * δ ^ 2 * φ ^ 3
      + (-132 : ℝ) * δ ^ 2 * φ ^ 2 + (-488 : ℝ) * δ ^ 2 * φ + (-372 : ℝ) * δ ^ 2
      + (-4 : ℝ) * δ * φ ^ 3 + (-32 : ℝ) * δ * φ ^ 2 + (-108 : ℝ) * δ * φ + (-192 : ℝ) * δ
      + (-48 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_7_5_false (δ φ : ℝ) :
    M_rec δ φ 5 7 false =
      (-5 : ℝ) * δ ^ 7 * φ + (-15 : ℝ) * δ ^ 7 + (5 : ℝ) * δ ^ 6 * φ ^ 2 + (60 : ℝ) * δ ^ 6 * φ
      + (75 : ℝ) * δ ^ 6 + (-25 : ℝ) * δ ^ 5 * φ ^ 2 + (-225 : ℝ) * δ ^ 5 * φ
      + (-170 : ℝ) * δ ^ 5 + (50 : ℝ) * δ ^ 4 * φ ^ 2 + (400 : ℝ) * δ ^ 4 * φ
      + (250 : ℝ) * δ ^ 4 + (-50 : ℝ) * δ ^ 3 * φ ^ 2 + (-375 : ℝ) * δ ^ 3 * φ
      + (-275 : ℝ) * δ ^ 3 + (25 : ℝ) * δ ^ 2 * φ ^ 2 + (180 : ℝ) * δ ^ 2 * φ
      + (215 : ℝ) * δ ^ 2 + (-5 : ℝ) * δ * φ ^ 2 + (-35 : ℝ) * δ * φ + (-100 : ℝ) * δ + (20 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_7_5_true (δ φ : ℝ) :
    M_rec δ φ 5 7 true =
      (5 : ℝ) * δ ^ 7 * φ + (15 : ℝ) * δ ^ 7 + (5 : ℝ) * δ ^ 6 * φ ^ 2 + (60 : ℝ) * δ ^ 6 * φ
      + (75 : ℝ) * δ ^ 6 + (25 : ℝ) * δ ^ 5 * φ ^ 2 + (225 : ℝ) * δ ^ 5 * φ + (170 : ℝ) * δ ^ 5
      + (50 : ℝ) * δ ^ 4 * φ ^ 2 + (400 : ℝ) * δ ^ 4 * φ + (250 : ℝ) * δ ^ 4
      + (50 : ℝ) * δ ^ 3 * φ ^ 2 + (375 : ℝ) * δ ^ 3 * φ + (275 : ℝ) * δ ^ 3
      + (25 : ℝ) * δ ^ 2 * φ ^ 2 + (180 : ℝ) * δ ^ 2 * φ + (215 : ℝ) * δ ^ 2
      + (5 : ℝ) * δ * φ ^ 2 + (35 : ℝ) * δ * φ + (100 : ℝ) * δ + (20 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_7_6_false (δ φ : ℝ) :
    M_rec δ φ 6 7 false =
      (6 : ℝ) * δ ^ 7 * φ + (-36 : ℝ) * δ ^ 6 * φ + (-6 : ℝ) * δ ^ 6 + (90 : ℝ) * δ ^ 5 * φ
      + (36 : ℝ) * δ ^ 5 + (-120 : ℝ) * δ ^ 4 * φ + (-90 : ℝ) * δ ^ 4 + (90 : ℝ) * δ ^ 3 * φ
      + (120 : ℝ) * δ ^ 3 + (-36 : ℝ) * δ ^ 2 * φ + (-90 : ℝ) * δ ^ 2 + (6 : ℝ) * δ * φ
      + (36 : ℝ) * δ + (-6 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_7_6_true (δ φ : ℝ) :
    M_rec δ φ 6 7 true =
      (-6 : ℝ) * δ ^ 7 * φ + (-36 : ℝ) * δ ^ 6 * φ + (-6 : ℝ) * δ ^ 6 + (-90 : ℝ) * δ ^ 5 * φ
      + (-36 : ℝ) * δ ^ 5 + (-120 : ℝ) * δ ^ 4 * φ + (-90 : ℝ) * δ ^ 4 + (-90 : ℝ) * δ ^ 3 * φ
      + (-120 : ℝ) * δ ^ 3 + (-36 : ℝ) * δ ^ 2 * φ + (-90 : ℝ) * δ ^ 2 + (-6 : ℝ) * δ * φ
      + (-36 : ℝ) * δ + (-6 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_7_7_false (δ φ : ℝ) :
    M_rec δ φ 7 7 false =
      (-1 : ℝ) * δ ^ 7 + (7 : ℝ) * δ ^ 6 + (-21 : ℝ) * δ ^ 5 + (35 : ℝ) * δ ^ 4
      + (-35 : ℝ) * δ ^ 3 + (21 : ℝ) * δ ^ 2 + (-7 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_7_7_true (δ φ : ℝ) :
    M_rec δ φ 7 7 true =
      (1 : ℝ) * δ ^ 7 + (7 : ℝ) * δ ^ 6 + (21 : ℝ) * δ ^ 5 + (35 : ℝ) * δ ^ 4 + (35 : ℝ) * δ ^ 3
      + (21 : ℝ) * δ ^ 2 + (7 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_8_1_false (δ φ : ℝ) :
    M_rec δ φ 1 8 false =
      (-41 : ℝ) * δ ^ 8 * φ ^ 4 + (-138 : ℝ) * δ ^ 8 * φ ^ 3 + (-176 : ℝ) * δ ^ 8 * φ ^ 2
      + (-74 : ℝ) * δ ^ 8 * φ + (72 : ℝ) * δ ^ 7 * φ ^ 4 + (476 : ℝ) * δ ^ 7 * φ ^ 3
      + (1158 : ℝ) * δ ^ 7 * φ ^ 2 + (1240 : ℝ) * δ ^ 7 * φ + (486 : ℝ) * δ ^ 7
      + (-60 : ℝ) * δ ^ 6 * φ ^ 5 + (-698 : ℝ) * δ ^ 6 * φ ^ 4 + (-3146 : ℝ) * δ ^ 6 * φ ^ 3
      + (-4752 : ℝ) * δ ^ 6 * φ ^ 2 + (-2870 : ℝ) * δ ^ 6 * φ + (-486 : ℝ) * δ ^ 6
      + (72 : ℝ) * δ ^ 5 * φ ^ 5 + (850 : ℝ) * δ ^ 5 * φ ^ 4 + (4050 : ℝ) * δ ^ 5 * φ ^ 3
      + (8386 : ℝ) * δ ^ 5 * φ ^ 2 + (7890 : ℝ) * δ ^ 5 * φ + (2776 : ℝ) * δ ^ 5
      + (-15 : ℝ) * δ ^ 4 * φ ^ 6 + (-208 : ℝ) * δ ^ 4 * φ ^ 5 + (-1330 : ℝ) * δ ^ 4 * φ ^ 4
      + (-5014 : ℝ) * δ ^ 4 * φ ^ 3 + (-11209 : ℝ) * δ ^ 4 * φ ^ 2 + (-9478 : ℝ) * δ ^ 4 * φ
      + (-2776 : ℝ) * δ ^ 4 + (16 : ℝ) * δ ^ 3 * φ ^ 6 + (212 : ℝ) * δ ^ 3 * φ ^ 5
      + (1266 : ℝ) * δ ^ 3 * φ ^ 4 + (4320 : ℝ) * δ ^ 3 * φ ^ 3 + (8350 : ℝ) * δ ^ 3 * φ ^ 2
      + (7480 : ℝ) * δ ^ 3 * φ + (2380 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 7
      + (-14 : ℝ) * δ ^ 2 * φ ^ 6 + (-93 : ℝ) * δ ^ 2 * φ ^ 5 + (-392 : ℝ) * δ ^ 2 * φ ^ 4
      + (-1185 : ℝ) * δ ^ 2 * φ ^ 3 + (-2758 : ℝ) * δ ^ 2 * φ ^ 2 + (-5189 : ℝ) * δ ^ 2 * φ
      + (-2380 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 7 + (13 : ℝ) * δ * φ ^ 6 + (77 : ℝ) * δ * φ ^ 5
      + (273 : ℝ) * δ * φ ^ 4 + (637 : ℝ) * δ * φ ^ 3 + (1001 : ℝ) * δ * φ ^ 2
      + (1001 : ℝ) * δ * φ + (429 : ℝ) * δ + (-429 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_8_1_true (δ φ : ℝ) :
    M_rec δ φ 1 8 true =
      (-41 : ℝ) * δ ^ 8 * φ ^ 4 + (-138 : ℝ) * δ ^ 8 * φ ^ 3 + (-176 : ℝ) * δ ^ 8 * φ ^ 2
      + (-74 : ℝ) * δ ^ 8 * φ + (-72 : ℝ) * δ ^ 7 * φ ^ 4 + (-476 : ℝ) * δ ^ 7 * φ ^ 3
      + (-1158 : ℝ) * δ ^ 7 * φ ^ 2 + (-1240 : ℝ) * δ ^ 7 * φ + (-486 : ℝ) * δ ^ 7
      + (-60 : ℝ) * δ ^ 6 * φ ^ 5 + (-698 : ℝ) * δ ^ 6 * φ ^ 4 + (-3146 : ℝ) * δ ^ 6 * φ ^ 3
      + (-4752 : ℝ) * δ ^ 6 * φ ^ 2 + (-2870 : ℝ) * δ ^ 6 * φ + (-486 : ℝ) * δ ^ 6
      + (-72 : ℝ) * δ ^ 5 * φ ^ 5 + (-850 : ℝ) * δ ^ 5 * φ ^ 4 + (-4050 : ℝ) * δ ^ 5 * φ ^ 3
      + (-8386 : ℝ) * δ ^ 5 * φ ^ 2 + (-7890 : ℝ) * δ ^ 5 * φ + (-2776 : ℝ) * δ ^ 5
      + (-15 : ℝ) * δ ^ 4 * φ ^ 6 + (-208 : ℝ) * δ ^ 4 * φ ^ 5 + (-1330 : ℝ) * δ ^ 4 * φ ^ 4
      + (-5014 : ℝ) * δ ^ 4 * φ ^ 3 + (-11209 : ℝ) * δ ^ 4 * φ ^ 2 + (-9478 : ℝ) * δ ^ 4 * φ
      + (-2776 : ℝ) * δ ^ 4 + (-16 : ℝ) * δ ^ 3 * φ ^ 6 + (-212 : ℝ) * δ ^ 3 * φ ^ 5
      + (-1266 : ℝ) * δ ^ 3 * φ ^ 4 + (-4320 : ℝ) * δ ^ 3 * φ ^ 3 + (-8350 : ℝ) * δ ^ 3 * φ ^ 2
      + (-7480 : ℝ) * δ ^ 3 * φ + (-2380 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 7
      + (-14 : ℝ) * δ ^ 2 * φ ^ 6 + (-93 : ℝ) * δ ^ 2 * φ ^ 5 + (-392 : ℝ) * δ ^ 2 * φ ^ 4
      + (-1185 : ℝ) * δ ^ 2 * φ ^ 3 + (-2758 : ℝ) * δ ^ 2 * φ ^ 2 + (-5189 : ℝ) * δ ^ 2 * φ
      + (-2380 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 7 + (-13 : ℝ) * δ * φ ^ 6
      + (-77 : ℝ) * δ * φ ^ 5 + (-273 : ℝ) * δ * φ ^ 4 + (-637 : ℝ) * δ * φ ^ 3
      + (-1001 : ℝ) * δ * φ ^ 2 + (-1001 : ℝ) * δ * φ + (-429 : ℝ) * δ + (-429 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_8_2_false (δ φ : ℝ) :
    M_rec δ φ 2 8 false =
      (22 : ℝ) * δ ^ 8 * φ ^ 3 + (118 : ℝ) * δ ^ 8 * φ ^ 2 + (190 : ℝ) * δ ^ 8 * φ
      + (99 : ℝ) * δ ^ 8 + (-76 : ℝ) * δ ^ 7 * φ ^ 4 + (-718 : ℝ) * δ ^ 7 * φ ^ 3
      + (-1422 : ℝ) * δ ^ 7 * φ ^ 2 + (-1018 : ℝ) * δ ^ 7 * φ + (-198 : ℝ) * δ ^ 7
      + (170 : ℝ) * δ ^ 6 * φ ^ 4 + (1632 : ℝ) * δ ^ 6 * φ ^ 3 + (4052 : ℝ) * δ ^ 6 * φ ^ 2
      + (4444 : ℝ) * δ ^ 6 * φ + (1714 : ℝ) * δ ^ 6 + (-26 : ℝ) * δ ^ 5 * φ ^ 5
      + (-422 : ℝ) * δ ^ 5 * φ ^ 4 + (-2808 : ℝ) * δ ^ 5 * φ ^ 3 + (-8500 : ℝ) * δ ^ 5 * φ ^ 2
      + (-9038 : ℝ) * δ ^ 5 * φ + (-3230 : ℝ) * δ ^ 5 + (54 : ℝ) * δ ^ 4 * φ ^ 5
      + (670 : ℝ) * δ ^ 4 * φ ^ 4 + (3714 : ℝ) * δ ^ 4 * φ ^ 3 + (10962 : ℝ) * δ ^ 4 * φ ^ 2
      + (10932 : ℝ) * δ ^ 4 * φ + (3698 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 6
      + (-54 : ℝ) * δ ^ 3 * φ ^ 5 + (-504 : ℝ) * δ ^ 3 * φ ^ 4 + (-2490 : ℝ) * δ ^ 3 * φ ^ 3
      + (-7088 : ℝ) * δ ^ 3 * φ ^ 2 + (-9720 : ℝ) * δ ^ 3 * φ + (-4166 : ℝ) * δ ^ 3
      + (4 : ℝ) * δ ^ 2 * φ ^ 6 + (50 : ℝ) * δ ^ 2 * φ ^ 5 + (292 : ℝ) * δ ^ 2 * φ ^ 4
      + (1064 : ℝ) * δ ^ 2 * φ ^ 3 + (2736 : ℝ) * δ ^ 2 * φ ^ 2 + (5354 : ℝ) * δ ^ 2 * φ
      + (2512 : ℝ) * δ ^ 2 + (-2 : ℝ) * δ * φ ^ 6 + (-24 : ℝ) * δ * φ ^ 5
      + (-130 : ℝ) * δ * φ ^ 4 + (-416 : ℝ) * δ * φ ^ 3 + (-858 : ℝ) * δ * φ ^ 2
      + (-1144 : ℝ) * δ * φ + (-858 : ℝ) * δ + (429 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_8_2_true (δ φ : ℝ) :
    M_rec δ φ 2 8 true =
      (22 : ℝ) * δ ^ 8 * φ ^ 3 + (118 : ℝ) * δ ^ 8 * φ ^ 2 + (190 : ℝ) * δ ^ 8 * φ
      + (99 : ℝ) * δ ^ 8 + (76 : ℝ) * δ ^ 7 * φ ^ 4 + (718 : ℝ) * δ ^ 7 * φ ^ 3
      + (1422 : ℝ) * δ ^ 7 * φ ^ 2 + (1018 : ℝ) * δ ^ 7 * φ + (198 : ℝ) * δ ^ 7
      + (170 : ℝ) * δ ^ 6 * φ ^ 4 + (1632 : ℝ) * δ ^ 6 * φ ^ 3 + (4052 : ℝ) * δ ^ 6 * φ ^ 2
      + (4444 : ℝ) * δ ^ 6 * φ + (1714 : ℝ) * δ ^ 6 + (26 : ℝ) * δ ^ 5 * φ ^ 5
      + (422 : ℝ) * δ ^ 5 * φ ^ 4 + (2808 : ℝ) * δ ^ 5 * φ ^ 3 + (8500 : ℝ) * δ ^ 5 * φ ^ 2
      + (9038 : ℝ) * δ ^ 5 * φ + (3230 : ℝ) * δ ^ 5 + (54 : ℝ) * δ ^ 4 * φ ^ 5
      + (670 : ℝ) * δ ^ 4 * φ ^ 4 + (3714 : ℝ) * δ ^ 4 * φ ^ 3 + (10962 : ℝ) * δ ^ 4 * φ ^ 2
      + (10932 : ℝ) * δ ^ 4 * φ + (3698 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 6
      + (54 : ℝ) * δ ^ 3 * φ ^ 5 + (504 : ℝ) * δ ^ 3 * φ ^ 4 + (2490 : ℝ) * δ ^ 3 * φ ^ 3
      + (7088 : ℝ) * δ ^ 3 * φ ^ 2 + (9720 : ℝ) * δ ^ 3 * φ + (4166 : ℝ) * δ ^ 3
      + (4 : ℝ) * δ ^ 2 * φ ^ 6 + (50 : ℝ) * δ ^ 2 * φ ^ 5 + (292 : ℝ) * δ ^ 2 * φ ^ 4
      + (1064 : ℝ) * δ ^ 2 * φ ^ 3 + (2736 : ℝ) * δ ^ 2 * φ ^ 2 + (5354 : ℝ) * δ ^ 2 * φ
      + (2512 : ℝ) * δ ^ 2 + (2 : ℝ) * δ * φ ^ 6 + (24 : ℝ) * δ * φ ^ 5 + (130 : ℝ) * δ * φ ^ 4
      + (416 : ℝ) * δ * φ ^ 3 + (858 : ℝ) * δ * φ ^ 2 + (1144 : ℝ) * δ * φ + (858 : ℝ) * δ
      + (429 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_8_3_false (δ φ : ℝ) :
    M_rec δ φ 3 8 false =
      (-57 : ℝ) * δ ^ 8 * φ ^ 3 + (-147 : ℝ) * δ ^ 8 * φ ^ 2 + (-93 : ℝ) * δ ^ 8 * φ
      + (189 : ℝ) * δ ^ 7 * φ ^ 3 + (681 : ℝ) * δ ^ 7 * φ ^ 2 + (1029 : ℝ) * δ ^ 7 * φ
      + (477 : ℝ) * δ ^ 7 + (-33 : ℝ) * δ ^ 6 * φ ^ 4 + (-582 : ℝ) * δ ^ 6 * φ ^ 3
      + (-2694 : ℝ) * δ ^ 6 * φ ^ 2 + (-3576 : ℝ) * δ ^ 6 * φ + (-1431 : ℝ) * δ ^ 6
      + (102 : ℝ) * δ ^ 5 * φ ^ 4 + (1230 : ℝ) * δ ^ 5 * φ ^ 3 + (5802 : ℝ) * δ ^ 5 * φ ^ 2
      + (6876 : ℝ) * δ ^ 5 * φ + (2622 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 5
      + (-141 : ℝ) * δ ^ 4 * φ ^ 4 + (-1395 : ℝ) * δ ^ 4 * φ ^ 3 + (-6309 : ℝ) * δ ^ 4 * φ ^ 2
      + (-8892 : ℝ) * δ ^ 4 * φ + (-4050 : ℝ) * δ ^ 4 + (9 : ℝ) * δ ^ 3 * φ ^ 5
      + (141 : ℝ) * δ ^ 3 * φ ^ 4 + (987 : ℝ) * δ ^ 3 * φ ^ 3 + (3927 : ℝ) * δ ^ 3 * φ ^ 2
      + (7698 : ℝ) * δ ^ 3 * φ + (3870 : ℝ) * δ ^ 3 + (-9 : ℝ) * δ ^ 2 * φ ^ 5
      + (-102 : ℝ) * δ ^ 2 * φ ^ 4 + (-534 : ℝ) * δ ^ 2 * φ ^ 3 + (-1722 : ℝ) * δ ^ 2 * φ ^ 2
      + (-3867 : ℝ) * δ ^ 2 * φ + (-2082 : ℝ) * δ ^ 2 + (3 : ℝ) * δ * φ ^ 5
      + (33 : ℝ) * δ * φ ^ 4 + (162 : ℝ) * δ * φ ^ 3 + (462 : ℝ) * δ * φ ^ 2 + (825 : ℝ) * δ * φ
      + (891 : ℝ) * δ + (-297 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_8_3_true (δ φ : ℝ) :
    M_rec δ φ 3 8 true =
      (-57 : ℝ) * δ ^ 8 * φ ^ 3 + (-147 : ℝ) * δ ^ 8 * φ ^ 2 + (-93 : ℝ) * δ ^ 8 * φ
      + (-189 : ℝ) * δ ^ 7 * φ ^ 3 + (-681 : ℝ) * δ ^ 7 * φ ^ 2 + (-1029 : ℝ) * δ ^ 7 * φ
      + (-477 : ℝ) * δ ^ 7 + (-33 : ℝ) * δ ^ 6 * φ ^ 4 + (-582 : ℝ) * δ ^ 6 * φ ^ 3
      + (-2694 : ℝ) * δ ^ 6 * φ ^ 2 + (-3576 : ℝ) * δ ^ 6 * φ + (-1431 : ℝ) * δ ^ 6
      + (-102 : ℝ) * δ ^ 5 * φ ^ 4 + (-1230 : ℝ) * δ ^ 5 * φ ^ 3 + (-5802 : ℝ) * δ ^ 5 * φ ^ 2
      + (-6876 : ℝ) * δ ^ 5 * φ + (-2622 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 5
      + (-141 : ℝ) * δ ^ 4 * φ ^ 4 + (-1395 : ℝ) * δ ^ 4 * φ ^ 3 + (-6309 : ℝ) * δ ^ 4 * φ ^ 2
      + (-8892 : ℝ) * δ ^ 4 * φ + (-4050 : ℝ) * δ ^ 4 + (-9 : ℝ) * δ ^ 3 * φ ^ 5
      + (-141 : ℝ) * δ ^ 3 * φ ^ 4 + (-987 : ℝ) * δ ^ 3 * φ ^ 3 + (-3927 : ℝ) * δ ^ 3 * φ ^ 2
      + (-7698 : ℝ) * δ ^ 3 * φ + (-3870 : ℝ) * δ ^ 3 + (-9 : ℝ) * δ ^ 2 * φ ^ 5
      + (-102 : ℝ) * δ ^ 2 * φ ^ 4 + (-534 : ℝ) * δ ^ 2 * φ ^ 3 + (-1722 : ℝ) * δ ^ 2 * φ ^ 2
      + (-3867 : ℝ) * δ ^ 2 * φ + (-2082 : ℝ) * δ ^ 2 + (-3 : ℝ) * δ * φ ^ 5
      + (-33 : ℝ) * δ * φ ^ 4 + (-162 : ℝ) * δ * φ ^ 3 + (-462 : ℝ) * δ * φ ^ 2
      + (-825 : ℝ) * δ * φ + (-891 : ℝ) * δ + (-297 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_8_4_false (δ φ : ℝ) :
    M_rec δ φ 4 8 false =
      (12 : ℝ) * δ ^ 8 * φ ^ 2 + (88 : ℝ) * δ ^ 8 * φ + (65 : ℝ) * δ ^ 8
      + (-36 : ℝ) * δ ^ 7 * φ ^ 3 + (-388 : ℝ) * δ ^ 7 * φ ^ 2 + (-636 : ℝ) * δ ^ 7 * φ
      + (-260 : ℝ) * δ ^ 7 + (148 : ℝ) * δ ^ 6 * φ ^ 3 + (1496 : ℝ) * δ ^ 6 * φ ^ 2
      + (2092 : ℝ) * δ ^ 6 * φ + (884 : ℝ) * δ ^ 6 + (-4 : ℝ) * δ ^ 5 * φ ^ 4
      + (-272 : ℝ) * δ ^ 5 * φ ^ 3 + (-2520 : ℝ) * δ ^ 5 * φ ^ 2 + (-4208 : ℝ) * δ ^ 5 * φ
      + (-2236 : ℝ) * δ ^ 5 + (16 : ℝ) * δ ^ 4 * φ ^ 4 + (328 : ℝ) * δ ^ 4 * φ ^ 3
      + (2460 : ℝ) * δ ^ 4 * φ ^ 2 + (5552 : ℝ) * δ ^ 4 * φ + (3194 : ℝ) * δ ^ 4
      + (-24 : ℝ) * δ ^ 3 * φ ^ 4 + (-292 : ℝ) * δ ^ 3 * φ ^ 3 + (-1652 : ℝ) * δ ^ 3 * φ ^ 2
      + (-4636 : ℝ) * δ ^ 3 * φ + (-2636 : ℝ) * δ ^ 3 + (16 : ℝ) * δ ^ 2 * φ ^ 4
      + (164 : ℝ) * δ ^ 2 * φ ^ 3 + (768 : ℝ) * δ ^ 2 * φ ^ 2 + (2188 : ℝ) * δ ^ 2 * φ
      + (1484 : ℝ) * δ ^ 2 + (-4 : ℝ) * δ * φ ^ 4 + (-40 : ℝ) * δ * φ ^ 3
      + (-176 : ℝ) * δ * φ ^ 2 + (-440 : ℝ) * δ * φ + (-660 : ℝ) * δ + (165 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_8_4_true (δ φ : ℝ) :
    M_rec δ φ 4 8 true =
      (12 : ℝ) * δ ^ 8 * φ ^ 2 + (88 : ℝ) * δ ^ 8 * φ + (65 : ℝ) * δ ^ 8
      + (36 : ℝ) * δ ^ 7 * φ ^ 3 + (388 : ℝ) * δ ^ 7 * φ ^ 2 + (636 : ℝ) * δ ^ 7 * φ
      + (260 : ℝ) * δ ^ 7 + (148 : ℝ) * δ ^ 6 * φ ^ 3 + (1496 : ℝ) * δ ^ 6 * φ ^ 2
      + (2092 : ℝ) * δ ^ 6 * φ + (884 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 4
      + (272 : ℝ) * δ ^ 5 * φ ^ 3 + (2520 : ℝ) * δ ^ 5 * φ ^ 2 + (4208 : ℝ) * δ ^ 5 * φ
      + (2236 : ℝ) * δ ^ 5 + (16 : ℝ) * δ ^ 4 * φ ^ 4 + (328 : ℝ) * δ ^ 4 * φ ^ 3
      + (2460 : ℝ) * δ ^ 4 * φ ^ 2 + (5552 : ℝ) * δ ^ 4 * φ + (3194 : ℝ) * δ ^ 4
      + (24 : ℝ) * δ ^ 3 * φ ^ 4 + (292 : ℝ) * δ ^ 3 * φ ^ 3 + (1652 : ℝ) * δ ^ 3 * φ ^ 2
      + (4636 : ℝ) * δ ^ 3 * φ + (2636 : ℝ) * δ ^ 3 + (16 : ℝ) * δ ^ 2 * φ ^ 4
      + (164 : ℝ) * δ ^ 2 * φ ^ 3 + (768 : ℝ) * δ ^ 2 * φ ^ 2 + (2188 : ℝ) * δ ^ 2 * φ
      + (1484 : ℝ) * δ ^ 2 + (4 : ℝ) * δ * φ ^ 4 + (40 : ℝ) * δ * φ ^ 3 + (176 : ℝ) * δ * φ ^ 2
      + (440 : ℝ) * δ * φ + (660 : ℝ) * δ + (165 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_8_5_false (δ φ : ℝ) :
    M_rec δ φ 5 8 false =
      (-35 : ℝ) * δ ^ 8 * φ ^ 2 + (-40 : ℝ) * δ ^ 8 * φ + (180 : ℝ) * δ ^ 7 * φ ^ 2
      + (280 : ℝ) * δ ^ 7 * φ + (140 : ℝ) * δ ^ 7 + (-5 : ℝ) * δ ^ 6 * φ ^ 3
      + (-420 : ℝ) * δ ^ 6 * φ ^ 2 + (-975 : ℝ) * δ ^ 6 * φ + (-700 : ℝ) * δ ^ 6
      + (25 : ℝ) * δ ^ 5 * φ ^ 3 + (625 : ℝ) * δ ^ 5 * φ ^ 2 + (2075 : ℝ) * δ ^ 5 * φ
      + (1475 : ℝ) * δ ^ 5 + (-50 : ℝ) * δ ^ 4 * φ ^ 3 + (-675 : ℝ) * δ ^ 4 * φ ^ 2
      + (-2750 : ℝ) * δ ^ 4 * φ + (-1775 : ℝ) * δ ^ 4 + (50 : ℝ) * δ ^ 3 * φ ^ 3
      + (510 : ℝ) * δ ^ 3 * φ ^ 2 + (2190 : ℝ) * δ ^ 3 * φ + (1450 : ℝ) * δ ^ 3
      + (-25 : ℝ) * δ ^ 2 * φ ^ 3 + (-230 : ℝ) * δ ^ 2 * φ ^ 2 + (-955 : ℝ) * δ ^ 2 * φ
      + (-890 : ℝ) * δ ^ 2 + (5 : ℝ) * δ * φ ^ 3 + (45 : ℝ) * δ * φ ^ 2 + (175 : ℝ) * δ * φ
      + (375 : ℝ) * δ + (-75 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_8_5_true (δ φ : ℝ) :
    M_rec δ φ 5 8 true =
      (-35 : ℝ) * δ ^ 8 * φ ^ 2 + (-40 : ℝ) * δ ^ 8 * φ + (-180 : ℝ) * δ ^ 7 * φ ^ 2
      + (-280 : ℝ) * δ ^ 7 * φ + (-140 : ℝ) * δ ^ 7 + (-5 : ℝ) * δ ^ 6 * φ ^ 3
      + (-420 : ℝ) * δ ^ 6 * φ ^ 2 + (-975 : ℝ) * δ ^ 6 * φ + (-700 : ℝ) * δ ^ 6
      + (-25 : ℝ) * δ ^ 5 * φ ^ 3 + (-625 : ℝ) * δ ^ 5 * φ ^ 2 + (-2075 : ℝ) * δ ^ 5 * φ
      + (-1475 : ℝ) * δ ^ 5 + (-50 : ℝ) * δ ^ 4 * φ ^ 3 + (-675 : ℝ) * δ ^ 4 * φ ^ 2
      + (-2750 : ℝ) * δ ^ 4 * φ + (-1775 : ℝ) * δ ^ 4 + (-50 : ℝ) * δ ^ 3 * φ ^ 3
      + (-510 : ℝ) * δ ^ 3 * φ ^ 2 + (-2190 : ℝ) * δ ^ 3 * φ + (-1450 : ℝ) * δ ^ 3
      + (-25 : ℝ) * δ ^ 2 * φ ^ 3 + (-230 : ℝ) * δ ^ 2 * φ ^ 2 + (-955 : ℝ) * δ ^ 2 * φ
      + (-890 : ℝ) * δ ^ 2 + (-5 : ℝ) * δ * φ ^ 3 + (-45 : ℝ) * δ * φ ^ 2 + (-175 : ℝ) * δ * φ
      + (-375 : ℝ) * δ + (-75 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_8_6_false (δ φ : ℝ) :
    M_rec δ φ 6 8 false =
      (6 : ℝ) * δ ^ 8 * φ + (21 : ℝ) * δ ^ 8 + (-6 : ℝ) * δ ^ 7 * φ ^ 2 + (-84 : ℝ) * δ ^ 7 * φ
      + (-126 : ℝ) * δ ^ 7 + (36 : ℝ) * δ ^ 6 * φ ^ 2 + (378 : ℝ) * δ ^ 6 * φ
      + (342 : ℝ) * δ ^ 6 + (-90 : ℝ) * δ ^ 5 * φ ^ 2 + (-840 : ℝ) * δ ^ 5 * φ
      + (-582 : ℝ) * δ ^ 5 + (120 : ℝ) * δ ^ 4 * φ ^ 2 + (1050 : ℝ) * δ ^ 4 * φ
      + (720 : ℝ) * δ ^ 4 + (-90 : ℝ) * δ ^ 3 * φ ^ 2 + (-756 : ℝ) * δ ^ 3 * φ
      + (-666 : ℝ) * δ ^ 3 + (36 : ℝ) * δ ^ 2 * φ ^ 2 + (294 : ℝ) * δ ^ 2 * φ
      + (426 : ℝ) * δ ^ 2 + (-6 : ℝ) * δ * φ ^ 2 + (-48 : ℝ) * δ * φ + (-162 : ℝ) * δ + (27 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_8_6_true (δ φ : ℝ) :
    M_rec δ φ 6 8 true =
      (6 : ℝ) * δ ^ 8 * φ + (21 : ℝ) * δ ^ 8 + (6 : ℝ) * δ ^ 7 * φ ^ 2 + (84 : ℝ) * δ ^ 7 * φ
      + (126 : ℝ) * δ ^ 7 + (36 : ℝ) * δ ^ 6 * φ ^ 2 + (378 : ℝ) * δ ^ 6 * φ + (342 : ℝ) * δ ^ 6
      + (90 : ℝ) * δ ^ 5 * φ ^ 2 + (840 : ℝ) * δ ^ 5 * φ + (582 : ℝ) * δ ^ 5
      + (120 : ℝ) * δ ^ 4 * φ ^ 2 + (1050 : ℝ) * δ ^ 4 * φ + (720 : ℝ) * δ ^ 4
      + (90 : ℝ) * δ ^ 3 * φ ^ 2 + (756 : ℝ) * δ ^ 3 * φ + (666 : ℝ) * δ ^ 3
      + (36 : ℝ) * δ ^ 2 * φ ^ 2 + (294 : ℝ) * δ ^ 2 * φ + (426 : ℝ) * δ ^ 2
      + (6 : ℝ) * δ * φ ^ 2 + (48 : ℝ) * δ * φ + (162 : ℝ) * δ + (27 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_8_7_false (δ φ : ℝ) :
    M_rec δ φ 7 8 false =
      (-7 : ℝ) * δ ^ 8 * φ + (49 : ℝ) * δ ^ 7 * φ + (7 : ℝ) * δ ^ 7 + (-147 : ℝ) * δ ^ 6 * φ
      + (-49 : ℝ) * δ ^ 6 + (245 : ℝ) * δ ^ 5 * φ + (147 : ℝ) * δ ^ 5 + (-245 : ℝ) * δ ^ 4 * φ
      + (-245 : ℝ) * δ ^ 4 + (147 : ℝ) * δ ^ 3 * φ + (245 : ℝ) * δ ^ 3 + (-49 : ℝ) * δ ^ 2 * φ
      + (-147 : ℝ) * δ ^ 2 + (7 : ℝ) * δ * φ + (49 : ℝ) * δ + (-7 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_8_7_true (δ φ : ℝ) :
    M_rec δ φ 7 8 true =
      (-7 : ℝ) * δ ^ 8 * φ + (-49 : ℝ) * δ ^ 7 * φ + (-7 : ℝ) * δ ^ 7 + (-147 : ℝ) * δ ^ 6 * φ
      + (-49 : ℝ) * δ ^ 6 + (-245 : ℝ) * δ ^ 5 * φ + (-147 : ℝ) * δ ^ 5 + (-245 : ℝ) * δ ^ 4 * φ
      + (-245 : ℝ) * δ ^ 4 + (-147 : ℝ) * δ ^ 3 * φ + (-245 : ℝ) * δ ^ 3 + (-49 : ℝ) * δ ^ 2 * φ
      + (-147 : ℝ) * δ ^ 2 + (-7 : ℝ) * δ * φ + (-49 : ℝ) * δ + (-7 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_8_8_false (δ φ : ℝ) :
    M_rec δ φ 8 8 false =
      (1 : ℝ) * δ ^ 8 + (-8 : ℝ) * δ ^ 7 + (28 : ℝ) * δ ^ 6 + (-56 : ℝ) * δ ^ 5
      + (70 : ℝ) * δ ^ 4 + (-56 : ℝ) * δ ^ 3 + (28 : ℝ) * δ ^ 2 + (-8 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_8_8_true (δ φ : ℝ) :
    M_rec δ φ 8 8 true =
      (1 : ℝ) * δ ^ 8 + (8 : ℝ) * δ ^ 7 + (28 : ℝ) * δ ^ 6 + (56 : ℝ) * δ ^ 5 + (70 : ℝ) * δ ^ 4
      + (56 : ℝ) * δ ^ 3 + (28 : ℝ) * δ ^ 2 + (8 : ℝ) * δ + (1 : ℝ) := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma r_val_1 (δ φ : ℝ) :
    r_rec δ φ 1 =
      (1 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_1_1_false, M_1_1_true]
  ring

private lemma r_val_2 (δ φ : ℝ) :
    r_rec δ φ 2 =
      (-1 : ℝ) * δ ^ 2 * φ + (-1 / 2 : ℝ) * δ ^ 2 + (-3 / 2 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma r_val_3 (δ φ : ℝ) :
    r_rec δ φ 3 =
      (1 : ℝ) * δ ^ 2 * φ ^ 2 + (6 : ℝ) * δ ^ 2 * φ + (3 : ℝ) * δ ^ 2 + (10 / 3 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false, M_3_3_true]
  ring

private lemma r_val_4 (δ φ : ℝ) :
    r_rec δ φ 4 =
      (-3 : ℝ) * δ ^ 4 * φ ^ 2 + (-4 : ℝ) * δ ^ 4 * φ + (-7 / 4 : ℝ) * δ ^ 4
      + (-1 : ℝ) * δ ^ 2 * φ ^ 3 + (-8 : ℝ) * δ ^ 2 * φ ^ 2 + (-29 : ℝ) * δ ^ 2 * φ
      + (-29 / 2 : ℝ) * δ ^ 2 + (-35 / 4 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false, M_4_3_true, M_4_4_false,
    M_4_4_true]
  ring

private lemma r_val_5 (δ φ : ℝ) :
    r_rec δ φ 5 =
      (6 : ℝ) * δ ^ 4 * φ ^ 3 + (44 : ℝ) * δ ^ 4 * φ ^ 2 + (54 : ℝ) * δ ^ 4 * φ
      + (22 : ℝ) * δ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 4 + (10 : ℝ) * δ ^ 2 * φ ^ 3
      + (46 : ℝ) * δ ^ 2 * φ ^ 2 + (130 : ℝ) * δ ^ 2 * φ + (65 : ℝ) * δ ^ 2 + (126 / 5 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false, M_5_3_true, M_5_4_false,
    M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma r_val_6 (δ φ : ℝ) :
    r_rec δ φ 6 =
      (-11 : ℝ) * δ ^ 6 * φ ^ 3 + (-28 : ℝ) * δ ^ 6 * φ ^ 2 + (-85 / 3 : ℝ) * δ ^ 6 * φ
      + (-29 / 3 : ℝ) * δ ^ 6 + (-9 : ℝ) * δ ^ 4 * φ ^ 4 + (-96 : ℝ) * δ ^ 4 * φ ^ 3
      + (-406 : ℝ) * δ ^ 4 * φ ^ 2 + (-464 : ℝ) * δ ^ 4 * φ + (-180 : ℝ) * δ ^ 4
      + (-1 : ℝ) * δ ^ 2 * φ ^ 5 + (-12 : ℝ) * δ ^ 2 * φ ^ 4 + (-67 : ℝ) * δ ^ 2 * φ ^ 3
      + (-232 : ℝ) * δ ^ 2 * φ ^ 2 + (-562 : ℝ) * δ ^ 2 * φ + (-281 : ℝ) * δ ^ 2 + (-77 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false, M_6_3_true, M_6_4_false,
    M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma r_val_7 (δ φ : ℝ) :
    r_rec δ φ 7 =
      (31 : ℝ) * δ ^ 6 * φ ^ 4 + (286 : ℝ) * δ ^ 6 * φ ^ 3 + (626 : ℝ) * δ ^ 6 * φ ^ 2
      + (582 : ℝ) * δ ^ 6 * φ + (191 : ℝ) * δ ^ 6 + (12 : ℝ) * δ ^ 4 * φ ^ 5
      + (160 : ℝ) * δ ^ 4 * φ ^ 4 + (948 : ℝ) * δ ^ 4 * φ ^ 3 + (3008 : ℝ) * δ ^ 4 * φ ^ 2
      + (3240 : ℝ) * δ ^ 4 * φ + (1212 : ℝ) * δ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 6
      + (14 : ℝ) * δ ^ 2 * φ ^ 5 + (92 : ℝ) * δ ^ 2 * φ ^ 4 + (378 : ℝ) * δ ^ 2 * φ ^ 3
      + (1093 : ℝ) * δ ^ 2 * φ ^ 2 + (2380 : ℝ) * δ ^ 2 * φ + (1190 : ℝ) * δ ^ 2
      + (1716 / 7 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false, M_7_3_true, M_7_4_false,
    M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true, M_7_7_false, M_7_7_true]
  ring

private lemma r_val_8 (δ φ : ℝ) :
    r_rec δ φ 8 =
      (-41 : ℝ) * δ ^ 8 * φ ^ 4 + (-168 : ℝ) * δ ^ 8 * φ ^ 3 + (-294 : ℝ) * δ ^ 8 * φ ^ 2
      + (-232 : ℝ) * δ ^ 8 * φ + (-555 / 8 : ℝ) * δ ^ 8 + (-60 : ℝ) * δ ^ 6 * φ ^ 5
      + (-794 : ℝ) * δ ^ 6 * φ ^ 4 + (-4194 : ℝ) * δ ^ 6 * φ ^ 3 + (-8140 : ℝ) * δ ^ 6 * φ ^ 2
      + (-7086 : ℝ) * δ ^ 6 * φ + (-4497 / 2 : ℝ) * δ ^ 6 + (-15 : ℝ) * δ ^ 4 * φ ^ 6
      + (-236 : ℝ) * δ ^ 4 * φ ^ 5 + (-1716 : ℝ) * δ ^ 4 * φ ^ 4 + (-7428 : ℝ) * δ ^ 4 * φ ^ 3
      + (-19563 : ℝ) * δ ^ 4 * φ ^ 2 + (-20056 : ℝ) * δ ^ 4 * φ + (-29169 / 4 : ℝ) * δ ^ 4
      + (-1 : ℝ) * δ ^ 2 * φ ^ 7 + (-16 : ℝ) * δ ^ 2 * φ ^ 6 + (-121 : ℝ) * δ ^ 2 * φ ^ 5
      + (-576 : ℝ) * δ ^ 2 * φ ^ 4 + (-1941 : ℝ) * δ ^ 2 * φ ^ 3 + (-4944 : ℝ) * δ ^ 2 * φ ^ 2
      + (-9949 : ℝ) * δ ^ 2 * φ + (-9949 / 2 : ℝ) * δ ^ 2 + (-6435 / 8 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false, M_8_3_true, M_8_4_false,
    M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true, M_8_7_false, M_8_7_true,
    M_8_8_false, M_8_8_true]
  ring

private lemma e_val_0 (δ φ : ℝ) : e_rec δ φ 0 = 1 := by rw [e_rec]

private lemma e_val_1 (δ φ : ℝ) :
    e_rec δ φ 1 =
      (-1 : ℝ) := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, e_val_0]
  ring

private lemma e_val_2 (δ φ : ℝ) :
    e_rec δ φ 2 =
      (1 : ℝ) * δ ^ 2 * φ + (1 / 2 : ℝ) * δ ^ 2 + (2 : ℝ) := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, e_val_0, e_val_1]
  ring

private lemma e_val_3 (δ φ : ℝ) :
    e_rec δ φ 3 =
      (-1 : ℝ) * δ ^ 2 * φ ^ 2 + (-7 : ℝ) * δ ^ 2 * φ + (-7 / 2 : ℝ) * δ ^ 2 + (-5 : ℝ) := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, e_val_0, e_val_1, e_val_2]
  ring

private lemma e_val_4 (δ φ : ℝ) :
    e_rec δ φ 4 =
      (7 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (9 / 2 : ℝ) * δ ^ 4 * φ + (15 / 8 : ℝ) * δ ^ 4
      + (1 : ℝ) * δ ^ 2 * φ ^ 3 + (9 : ℝ) * δ ^ 2 * φ ^ 2 + (37 : ℝ) * δ ^ 2 * φ
      + (37 / 2 : ℝ) * δ ^ 2 + (14 : ℝ) := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, e_val_0, e_val_1, e_val_2, e_val_3]
  ring

private lemma e_val_5 (δ φ : ℝ) :
    e_rec δ φ 5 =
      (-7 : ℝ) * δ ^ 4 * φ ^ 3 + (-54 : ℝ) * δ ^ 4 * φ ^ 2 + (-129 / 2 : ℝ) * δ ^ 4 * φ
      + (-203 / 8 : ℝ) * δ ^ 4 + (-1 : ℝ) * δ ^ 2 * φ ^ 4 + (-11 : ℝ) * δ ^ 2 * φ ^ 3
      + (-56 : ℝ) * δ ^ 2 * φ ^ 2 + (-176 : ℝ) * δ ^ 2 * φ + (-88 : ℝ) * δ ^ 2 + (-42 : ℝ) := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4]
  ring

private lemma e_val_6 (δ φ : ℝ) :
    e_rec δ φ 6 =
      (85 / 6 : ℝ) * δ ^ 6 * φ ^ 3 + (135 / 4 : ℝ) * δ ^ 6 * φ ^ 2 + (773 / 24 : ℝ) * δ ^ 6 * φ
      + (169 / 16 : ℝ) * δ ^ 6 + (21 / 2 : ℝ) * δ ^ 4 * φ ^ 4 + (235 / 2 : ℝ) * δ ^ 4 * φ ^ 3
      + (1035 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (580 : ℝ) * δ ^ 4 * φ + (219 : ℝ) * δ ^ 4
      + (1 : ℝ) * δ ^ 2 * φ ^ 5 + (13 : ℝ) * δ ^ 2 * φ ^ 4 + (79 : ℝ) * δ ^ 2 * φ ^ 3
      + (299 : ℝ) * δ ^ 2 * φ ^ 2 + (794 : ℝ) * δ ^ 2 * φ + (397 : ℝ) * δ ^ 2 + (132 : ℝ) := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, e_val_0, e_val_1, e_val_2, e_val_3,
    e_val_4, e_val_5]
  ring

private lemma e_val_7 (δ φ : ℝ) :
    e_rec δ φ 7 =
      (-81 / 2 : ℝ) * δ ^ 6 * φ ^ 4 + (-1118 / 3 : ℝ) * δ ^ 6 * φ ^ 3
      + (-6201 / 8 : ℝ) * δ ^ 6 * φ ^ 2 + (-16511 / 24 : ℝ) * δ ^ 6 * φ
      + (-3491 / 16 : ℝ) * δ ^ 6 + (-14 : ℝ) * δ ^ 4 * φ ^ 5 + (-195 : ℝ) * δ ^ 4 * φ ^ 4
      + (-2407 / 2 : ℝ) * δ ^ 4 * φ ^ 3 + (-3952 : ℝ) * δ ^ 4 * φ ^ 2
      + (-8395 / 2 : ℝ) * δ ^ 4 * φ + (-12289 / 8 : ℝ) * δ ^ 4 + (-1 : ℝ) * δ ^ 2 * φ ^ 6
      + (-15 : ℝ) * δ ^ 2 * φ ^ 5 + (-106 : ℝ) * δ ^ 2 * φ ^ 4 + (-470 : ℝ) * δ ^ 2 * φ ^ 3
      + (-1471 : ℝ) * δ ^ 2 * φ ^ 2 + (-3473 : ℝ) * δ ^ 2 * φ + (-3473 / 2 : ℝ) * δ ^ 2
      + (-429 : ℝ) := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, e_val_0, e_val_1, e_val_2,
    e_val_3, e_val_4, e_val_5, e_val_6]
  ring

private lemma e_val_8 (δ φ : ℝ) :
    e_rec δ φ 8 =
      (1393 / 24 : ℝ) * δ ^ 8 * φ ^ 4 + (2605 / 12 : ℝ) * δ ^ 8 * φ ^ 3
      + (16939 / 48 : ℝ) * δ ^ 8 * φ ^ 2 + (12683 / 48 : ℝ) * δ ^ 8 * φ
      + (9723 / 128 : ℝ) * δ ^ 8 + (79 : ℝ) * δ ^ 6 * φ ^ 5 + (4215 / 4 : ℝ) * δ ^ 6 * φ ^ 4
      + (132821 / 24 : ℝ) * δ ^ 6 * φ ^ 3 + (82549 / 8 : ℝ) * δ ^ 6 * φ ^ 2
      + (207487 / 24 : ℝ) * δ ^ 6 * φ + (42603 / 16 : ℝ) * δ ^ 6 + (35 / 2 : ℝ) * δ ^ 4 * φ ^ 6
      + (573 / 2 : ℝ) * δ ^ 4 * φ ^ 5 + (4329 / 2 : ℝ) * δ ^ 4 * φ ^ 4
      + (9711 : ℝ) * δ ^ 4 * φ ^ 3 + (52711 / 2 : ℝ) * δ ^ 4 * φ ^ 2
      + (53495 / 2 : ℝ) * δ ^ 4 * φ + (76385 / 8 : ℝ) * δ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 7
      + (17 : ℝ) * δ ^ 2 * φ ^ 6 + (137 : ℝ) * δ ^ 2 * φ ^ 5 + (697 : ℝ) * δ ^ 2 * φ ^ 4
      + (2517 : ℝ) * δ ^ 2 * φ ^ 3 + (6885 : ℝ) * δ ^ 2 * φ ^ 2 + (14893 : ℝ) * δ ^ 2 * φ
      + (14893 / 2 : ℝ) * δ ^ 2 + (1430 : ℝ) := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, e_val_0, e_val_1,
    e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7]
  ring

private lemma g_val_0 (δ φ : ℝ) : g_rec δ φ 0 = 0 := by unfold g_rec; simp

private lemma g_val_1 (δ φ : ℝ) :
    g_rec δ φ 1 =
      0 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, e_val_0, e_val_1]
  ring

private lemma g_val_2 (δ φ : ℝ) :
    g_rec δ φ 2 =
      (1 : ℝ) * δ ^ 2 * φ + (1 / 2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, e_val_0, e_val_1, e_val_2]
  ring

private lemma g_val_3 (δ φ : ℝ) :
    g_rec δ φ 3 =
      (-2 : ℝ) * δ ^ 2 * φ ^ 2 + (-8 : ℝ) * δ ^ 2 * φ + (-4 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, e_val_0, e_val_1, e_val_2, e_val_3]
  ring

private lemma g_val_4 (δ φ : ℝ) :
    g_rec δ φ 4 =
      (13 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (19 / 2 : ℝ) * δ ^ 4 * φ + (37 / 8 : ℝ) * δ ^ 4
      + (3 : ℝ) * δ ^ 2 * φ ^ 3 + (19 : ℝ) * δ ^ 2 * φ ^ 2 + (47 : ℝ) * δ ^ 2 * φ
      + (47 / 2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4]
  ring

private lemma g_val_5 (δ φ : ℝ) :
    g_rec δ φ 5 =
      (-18 : ℝ) * δ ^ 4 * φ ^ 3 + (-116 : ℝ) * δ ^ 4 * φ ^ 2 + (-153 : ℝ) * δ ^ 4 * φ
      + (-271 / 4 : ℝ) * δ ^ 4 + (-4 : ℝ) * δ ^ 2 * φ ^ 4 + (-34 : ℝ) * δ ^ 2 * φ ^ 3
      + (-124 : ℝ) * δ ^ 2 * φ ^ 2 + (-244 : ℝ) * δ ^ 2 * φ + (-122 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4,
    e_val_5]
  ring

private lemma g_val_6 (δ φ : ℝ) :
    g_rec δ φ 6 =
      (197 / 6 : ℝ) * δ ^ 6 * φ ^ 3 + (399 / 4 : ℝ) * δ ^ 6 * φ ^ 2
      + (2749 / 24 : ℝ) * δ ^ 6 * φ + (673 / 16 : ℝ) * δ ^ 6 + (69 / 2 : ℝ) * δ ^ 4 * φ ^ 4
      + (659 / 2 : ℝ) * δ ^ 4 * φ ^ 3 + (2499 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (1508 : ℝ) * δ ^ 4 * φ
      + (627 : ℝ) * δ ^ 4 + (5 : ℝ) * δ ^ 2 * φ ^ 5 + (53 : ℝ) * δ ^ 2 * φ ^ 4
      + (251 : ℝ) * δ ^ 2 * φ ^ 3 + (691 : ℝ) * δ ^ 2 * φ ^ 2 + (1186 : ℝ) * δ ^ 2 * φ
      + (593 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, e_val_0, e_val_1, e_val_2, e_val_3,
    e_val_4, e_val_5, e_val_6]
  ring

private lemma g_val_7 (δ φ : ℝ) :
    g_rec δ φ 7 =
      (-110 : ℝ) * δ ^ 6 * φ ^ 4 + (-3068 / 3 : ℝ) * δ ^ 6 * φ ^ 3 + (-2563 : ℝ) * δ ^ 6 * φ ^ 2
      + (-7933 / 3 : ℝ) * δ ^ 6 * φ + (-1857 / 2 : ℝ) * δ ^ 6 + (-56 : ℝ) * δ ^ 4 * φ ^ 5
      + (-680 : ℝ) * δ ^ 4 * φ ^ 4 + (-3644 : ℝ) * δ ^ 4 * φ ^ 3 + (-10496 : ℝ) * δ ^ 4 * φ ^ 2
      + (-11780 : ℝ) * δ ^ 4 * φ + (-4679 : ℝ) * δ ^ 4 + (-6 : ℝ) * δ ^ 2 * φ ^ 6
      + (-76 : ℝ) * δ ^ 2 * φ ^ 5 + (-440 : ℝ) * δ ^ 2 * φ ^ 4 + (-1532 : ℝ) * δ ^ 2 * φ ^ 3
      + (-3534 : ℝ) * δ ^ 2 * φ ^ 2 + (-5536 : ℝ) * δ ^ 2 * φ + (-2768 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, e_val_0, e_val_1, e_val_2,
    e_val_3, e_val_4, e_val_5, e_val_6, e_val_7]
  ring

private lemma g_val_8 (δ φ : ℝ) :
    g_rec δ φ 8 =
      (1069 / 8 : ℝ) * δ ^ 8 * φ ^ 4 + (2937 / 4 : ℝ) * δ ^ 8 * φ ^ 3
      + (24447 / 16 : ℝ) * δ ^ 8 * φ ^ 2 + (21343 / 16 : ℝ) * δ ^ 8 * φ
      + (54573 / 128 : ℝ) * δ ^ 8 + (249 : ℝ) * δ ^ 6 * φ ^ 5 + (12881 / 4 : ℝ) * δ ^ 6 * φ ^ 4
      + (138369 / 8 : ℝ) * δ ^ 6 * φ ^ 3 + (298979 / 8 : ℝ) * δ ^ 6 * φ ^ 2
      + (284547 / 8 : ℝ) * δ ^ 6 * φ + (192189 / 16 : ℝ) * δ ^ 6 + (165 / 2 : ℝ) * δ ^ 4 * φ ^ 6
      + (2395 / 2 : ℝ) * δ ^ 4 * φ ^ 5 + (15951 / 2 : ℝ) * δ ^ 4 * φ ^ 4
      + (31449 : ℝ) * δ ^ 4 * φ ^ 3 + (151617 / 2 : ℝ) * δ ^ 4 * φ ^ 2
      + (160337 / 2 : ℝ) * δ ^ 4 * φ + (245943 / 8 : ℝ) * δ ^ 4 + (7 : ℝ) * δ ^ 2 * φ ^ 7
      + (103 : ℝ) * δ ^ 2 * φ ^ 6 + (703 : ℝ) * δ ^ 2 * φ ^ 5 + (2943 : ℝ) * δ ^ 2 * φ ^ 4
      + (8403 : ℝ) * δ ^ 2 * φ ^ 3 + (17139 : ℝ) * δ ^ 2 * φ ^ 2 + (25147 : ℝ) * δ ^ 2 * φ
      + (25147 / 2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, e_val_0, e_val_1,
    e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7, e_val_8]
  ring

private lemma monom_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ)
    (c : ℝ) (hc : 0 ≤ c) (a b : ℕ) :
    0 ≤ c * (δ ^ a * φ ^ b) :=
  mul_nonneg hc (mul_nonneg (pow_nonneg hδ0 a) (pow_nonneg hφ0 b))

private lemma monom_le_one {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (a b : ℕ) :
    0 ≤ 1 - δ ^ a * φ ^ b := by
  have h1 : δ ^ a ≤ 1 := pow_le_one₀ hδ0 hδ1
  have h2 : φ ^ b ≤ 1 := pow_le_one₀ hφ0 hφ1
  have h3 : δ ^ a * φ ^ b ≤ 1 * 1 := mul_le_mul h1 h2 (pow_nonneg hφ0 b) zero_le_one
  linarith

private lemma g4_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ g_rec δ φ 4 := by
  have hid : g_rec δ φ 4 =
      (13 / 2 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (19 / 2 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (37 / 8 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (3 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (19 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (47 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (47 / 2 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_4]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (13 / 2 : ℝ) (by norm_num) 4 2)
    (monom_nonneg hδ0 hφ0 (19 / 2 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (37 / 8 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (3 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (19 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (47 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (47 / 2 : ℝ) (by norm_num) 2 0)

private lemma g6_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ g_rec δ φ 6 := by
  have hid : g_rec δ φ 6 =
      (197 / 6 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (399 / 4 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (2749 / 24 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (673 / 16 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (69 / 2 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (659 / 2 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (2499 / 2 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (1508 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (627 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (5 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (53 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (251 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (691 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (1186 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (593 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_6]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (197 / 6 : ℝ) (by norm_num) 6 3)
    (monom_nonneg hδ0 hφ0 (399 / 4 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (2749 / 24 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (673 / 16 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (69 / 2 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (659 / 2 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (2499 / 2 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (1508 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (627 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (5 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (53 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (251 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (691 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (1186 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (593 : ℝ) (by norm_num) 2 0)

private lemma g8_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ g_rec δ φ 8 := by
  have hid : g_rec δ φ 8 =
      (1069 / 8 : ℝ) * (δ ^ 8 * φ ^ 4)
      + (2937 / 4 : ℝ) * (δ ^ 8 * φ ^ 3)
      + (24447 / 16 : ℝ) * (δ ^ 8 * φ ^ 2)
      + (21343 / 16 : ℝ) * (δ ^ 8 * φ ^ 1)
      + (54573 / 128 : ℝ) * (δ ^ 8 * φ ^ 0)
      + (249 : ℝ) * (δ ^ 6 * φ ^ 5)
      + (12881 / 4 : ℝ) * (δ ^ 6 * φ ^ 4)
      + (138369 / 8 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (298979 / 8 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (284547 / 8 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (192189 / 16 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (165 / 2 : ℝ) * (δ ^ 4 * φ ^ 6)
      + (2395 / 2 : ℝ) * (δ ^ 4 * φ ^ 5)
      + (15951 / 2 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (31449 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (151617 / 2 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (160337 / 2 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (245943 / 8 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (7 : ℝ) * (δ ^ 2 * φ ^ 7)
      + (103 : ℝ) * (δ ^ 2 * φ ^ 6)
      + (703 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (2943 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (8403 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (17139 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (25147 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (25147 / 2 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_8]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (1069 / 8 : ℝ) (by norm_num) 8 4)
    (monom_nonneg hδ0 hφ0 (2937 / 4 : ℝ) (by norm_num) 8 3))
    (monom_nonneg hδ0 hφ0 (24447 / 16 : ℝ) (by norm_num) 8 2))
    (monom_nonneg hδ0 hφ0 (21343 / 16 : ℝ) (by norm_num) 8 1))
    (monom_nonneg hδ0 hφ0 (54573 / 128 : ℝ) (by norm_num) 8 0))
    (monom_nonneg hδ0 hφ0 (249 : ℝ) (by norm_num) 6 5))
    (monom_nonneg hδ0 hφ0 (12881 / 4 : ℝ) (by norm_num) 6 4))
    (monom_nonneg hδ0 hφ0 (138369 / 8 : ℝ) (by norm_num) 6 3))
    (monom_nonneg hδ0 hφ0 (298979 / 8 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (284547 / 8 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (192189 / 16 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (165 / 2 : ℝ) (by norm_num) 4 6))
    (monom_nonneg hδ0 hφ0 (2395 / 2 : ℝ) (by norm_num) 4 5))
    (monom_nonneg hδ0 hφ0 (15951 / 2 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (31449 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (151617 / 2 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (160337 / 2 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (245943 / 8 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (7 : ℝ) (by norm_num) 2 7))
    (monom_nonneg hδ0 hφ0 (103 : ℝ) (by norm_num) 2 6))
    (monom_nonneg hδ0 hφ0 (703 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (2943 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (8403 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (17139 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (25147 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (25147 / 2 : ℝ) (by norm_num) 2 0)

private lemma neg_g5_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ -g_rec δ φ 5 := by
  have hid : -g_rec δ φ 5 =
      (18 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (116 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (153 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (271 / 4 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (4 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (34 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (124 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (244 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (122 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_5]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (18 : ℝ) (by norm_num) 4 3)
    (monom_nonneg hδ0 hφ0 (116 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (153 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (271 / 4 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (4 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (34 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (124 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (244 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (122 : ℝ) (by norm_num) 2 0)

private lemma neg_g7_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ -g_rec δ φ 7 := by
  have hid : -g_rec δ φ 7 =
      (110 : ℝ) * (δ ^ 6 * φ ^ 4)
      + (3068 / 3 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (2563 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (7933 / 3 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (1857 / 2 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (56 : ℝ) * (δ ^ 4 * φ ^ 5)
      + (680 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (3644 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (10496 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (11780 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (4679 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (6 : ℝ) * (δ ^ 2 * φ ^ 6)
      + (76 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (440 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (1532 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (3534 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (5536 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (2768 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_7]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (110 : ℝ) (by norm_num) 6 4)
    (monom_nonneg hδ0 hφ0 (3068 / 3 : ℝ) (by norm_num) 6 3))
    (monom_nonneg hδ0 hφ0 (2563 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (7933 / 3 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (1857 / 2 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (56 : ℝ) (by norm_num) 4 5))
    (monom_nonneg hδ0 hφ0 (680 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (3644 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (10496 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (11780 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (4679 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (6 : ℝ) (by norm_num) 2 6))
    (monom_nonneg hδ0 hφ0 (76 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (440 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (1532 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (3534 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (5536 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (2768 : ℝ) (by norm_num) 2 0)

private lemma g5_l1_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    0 ≤ (3531 / 4 : ℝ) * δ ^ 2 + g_rec δ φ 5 := by
  have hid : (3531 / 4 : ℝ) * δ ^ 2 + g_rec δ φ 5 =
      (18 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 3)
      + (116 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 2)
      + (153 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 1)
      + (271 / 4 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 0)
      + (4 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 4)
      + (34 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 3)
      + (124 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 2)
      + (244 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 1) := by
    rw [g_val_5]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 3))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 1))

private lemma g7_l1_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    0 ≤ (104991 / 2 : ℝ) * δ ^ 2 + g_rec δ φ 7 := by
  have hid : (104991 / 2 : ℝ) * δ ^ 2 + g_rec δ φ 7 =
      (110 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 4)
      + (3068 / 3 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 3)
      + (2563 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 2)
      + (7933 / 3 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 1)
      + (1857 / 2 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 0)
      + (56 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 5)
      + (680 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 4)
      + (3644 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 3)
      + (10496 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 2)
      + (11780 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 1)
      + (4679 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 0)
      + (6 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 6)
      + (76 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 5)
      + (440 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 4)
      + (1532 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 3)
      + (3534 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 2)
      + (5536 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 1) := by
    rw [g_val_7]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 4))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 6)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 1))

/-- Exact finite algebra from recurrence (4)-(5). Helper certificates are generated
from rational polynomials; every identity and inequality is checked by Lean. -/
theorem coefficient_bounds (δ φ : ℝ) (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    g_rec δ φ 0 = 0 ∧ g_rec δ φ 1 = 0 ∧
    g_rec δ φ 2 = δ ^ 2 * (1 / 2 + φ) ∧
    g_rec δ φ 3 = -δ ^ 2 * (4 + 8 * φ + 2 * φ ^ 2) ∧
    0 ≤ g_rec δ φ 4 ∧ 0 ≤ g_rec δ φ 6 ∧ 0 ≤ g_rec δ φ 8 ∧
    -(3531 / 4) * δ ^ 2 ≤ g_rec δ φ 5 ∧
    -(104991 / 2) * δ ^ 2 ≤ g_rec δ φ 7 := by
  refine ⟨g_val_0 δ φ, g_val_1 δ φ, ?_, ?_, g4_nonneg hδ0 hφ0,
    g6_nonneg hδ0 hφ0, g8_nonneg hδ0 hφ0, ?_, ?_⟩
  · rw [g_val_2]; ring
  · rw [g_val_3]; ring
  · linarith [g5_l1_nonneg hδ0 hδ1 hφ0 hφ1]
  · linarith [g7_l1_nonneg hδ0 hδ1 hφ0 hφ1]

/-- Degree-eight polynomial, including the zero constant and linear terms. -/
noncomputable def partialSum (δ φ q : ℝ) : ℝ :=
  ∑ n ∈ Finset.range 9, g_rec δ φ n * q ^ n

/-- Rational lower bound after division by `δ² q²`. -/
noncomputable def lowerMargin (q : ℝ) : ℝ :=
  1 / 2 - 4 * q - (3531 / 4) * q ^ 3 - (104991 / 2) * q ^ 5 -
    (25 / 9) * 10 ^ 9 * q ^ 7 / (1 - 10 * q)

/-- Discarding the nonnegative even terms and bounding both negative odd terms. -/
theorem partialSum_lower (δ φ q : ℝ) (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 30) :
    δ ^ 2 * q ^ 2 * (1 / 2 - 4 * q - (3531 / 4) * q ^ 3 -
      (104991 / 2) * q ^ 5) ≤ partialSum δ φ q := by
  obtain ⟨h0, h1, h2, h3, h4, h6, h8, h5, h7⟩ :=
    coefficient_bounds δ φ hδ0 hδ1 hφ0 hφ1
  have hlead : δ ^ 2 * (1 / 2 - 4 * q) ≤ g_rec δ φ 2 + q * g_rec δ φ 3 := by
    rw [h2, h3]
    have ha : 0 ≤ φ * (1 - 10 * q) := mul_nonneg hφ0 (by linarith)
    have hb : 0 ≤ 2 * q * φ * (1 - φ) := by positivity
    nlinarith only [mul_nonneg (sq_nonneg δ) ha, mul_nonneg (sq_nonneg δ) hb]
  unfold partialSum
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, h0, h1, zero_mul, zero_add]
  nlinarith only [mul_le_mul_of_nonneg_right hlead (sq_nonneg q),
    mul_le_mul_of_nonneg_right h5 (pow_nonneg hq0 5),
    mul_le_mul_of_nonneg_right h7 (pow_nonneg hq0 7),
    mul_nonneg h4 (pow_nonneg hq0 4), mul_nonneg h6 (pow_nonneg hq0 6),
    mul_nonneg h8 (pow_nonneg hq0 8)]
/-- The endpoint margin is exact and is strictly greater than `0.14`. -/
theorem lowerMargin_endpoint :
    lowerMargin (1 / 30) = (185403593 / 1312200000 : ℝ) ∧
    (14 / 100 : ℝ) < 185403593 / 1312200000 := by
  norm_num [lowerMargin]

/-- Uniform positive margin, including the endpoint `q = 1/30`. -/
theorem lowerMargin_lower (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 30) :
    (185403593 / 1312200000 : ℝ) ≤ lowerMargin q := by
  have h3 : q ^ 3 ≤ (1 / 30 : ℝ) ^ 3 := by gcongr
  have h5 : q ^ 5 ≤ (1 / 30 : ℝ) ^ 5 := by gcongr
  have ht : (25 / 9 : ℝ) * 10 ^ 9 * q ^ 7 / (1 - 10 * q) ≤
      (25 / 9 : ℝ) * 10 ^ 9 * (1 / 30) ^ 7 / (1 - 10 * (1 / 30)) := by
    gcongr
  unfold lowerMargin
  norm_num at h3 h5 ht
  linarith

/-- The entire remainder is subtracted before concluding positivity. -/
theorem sign_of_remainder (δ φ q G : ℝ) (hδ0 : 0 < δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (hq0 : 0 < q) (hq1 : q ≤ 1 / 30)
    (htail : |G - partialSum δ φ q| ≤ (25 / 9) * δ ^ 2 * (10 * q) ^ 9 /
      (1 - 10 * q)) :
    (185403593 / 1312200000 : ℝ) * (δ ^ 2 * q ^ 2) ≤ G ∧ 0 < G := by
  have hden : 1 - 10 * q ≠ 0 := by linarith
  have hp := partialSum_lower δ φ q hδ0.le hδ1 hφ0 hφ1 hq0.le hq1
  have ht := (abs_le.mp htail).1
  have hid : δ ^ 2 * q ^ 2 * lowerMargin q =
      δ ^ 2 * q ^ 2 * (1 / 2 - 4 * q - (3531 / 4) * q ^ 3 -
      (104991 / 2) * q ^ 5) - (25 / 9) * δ ^ 2 * (10 * q) ^ 9 / (1 - 10 * q) := by
    unfold lowerMargin
    field_simp
  have hg : δ ^ 2 * q ^ 2 * lowerMargin q ≤ G := by linarith
  have hm := mul_le_mul_of_nonneg_left (lowerMargin_lower q hq0.le hq1)
    (by positivity : 0 ≤ δ ^ 2 * q ^ 2)
  constructor
  · nlinarith only [hg, hm]
  · have hpos : 0 < δ ^ 2 * q ^ 2 * (185403593 / 1312200000 : ℝ) := by positivity
    linarith
/-- The endpoint has square exactly `16/17`. -/
theorem threshold_square : (4 / Real.sqrt 17 : ℝ) ^ 2 = 16 / 17 := by
  rw [div_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 17)]
  norm_num

/-- The chord lower bound on `R`, together with the threshold identity, implies
`q ≤ 1/30` throughout the exact interval. -/
theorem q_le_of_chord (q R s : ℝ) (hq : 0 ≤ q)
    (hs : 4 / Real.sqrt 17 ≤ s) (hthreshold : s = Real.exp (-R))
    (hchord : (1 / 2 : ℝ) * Real.log ((1 + 4 * q) / (1 + 2 * q)) ≤ R) :
    q ≤ 1 / 30 := by
  have hs0 : 0 < s := by rw [hthreshold]; exact Real.exp_pos _
  have hsq : (16 / 17 : ℝ) ≤ s ^ 2 := by
    rw [← threshold_square]
    exact pow_le_pow_left₀ (by positivity) hs 2
  have hden : 0 < 1 + 2 * q := by positivity
  have hnum : 0 < 1 + 4 * q := by positivity
  have he : (1 + 4 * q) / (1 + 2 * q) ≤ Real.exp (2 * R) := by
    rw [← Real.exp_log (div_pos hnum hden)]
    apply Real.exp_le_exp.mpr
    linarith
  have hid : Real.exp (2 * R) * s ^ 2 = 1 := by
    rw [hthreshold, sq, ← Real.exp_add, ← Real.exp_add]
    rw [show 2 * R + (-R + -R) = 0 by ring, Real.exp_zero]
  have hm := mul_le_mul_of_nonneg_right he (sq_nonneg s)
  rw [hid] at hm
  have hprod : (1 + 4 * q) * s ^ 2 ≤ 1 + 2 * q := by
    apply (div_le_one hden).mp
    simpa [div_mul_eq_mul_div] using hm
  have hh := mul_le_mul_of_nonneg_left hsq hnum.le
  nlinarith only [hh, hprod]

/-- Conditional form of the paper's proved interval. The seven stationary-process
inputs are named `hc_pos`, `hc_cont`, `hR_sderiv`, `hV_pos`, `h_thresh`, `hchord`,
and `htail`. In particular the remainder estimate is an input, not a consequence
of a finite polynomial computation. -/
theorem tail_derivative_neg (δ φ s R' : ℝ) (c_star R_real : ℝ → ℝ)
    (hδ0 : 0 < δ) (hδ1 : δ < 1) (hφ0 : 0 ≤ φ) (hφ1 : φ < 1)
    (hs : 4 / Real.sqrt 17 ≤ s) (hs1 : s < 1)
    (hc_pos : 0 < c_star s) (hc_cont : ContinuousAt c_star s)
    (hR_sderiv : HasStrictDerivAt R_real R' (c_star s / s ^ 2))
    (hV_pos : 0 < (c_star s / s ^ 2) * R')
    (h_thresh : ∀ᶠ u in 𝓝 s, Real.log u + R_real (c_star u / u ^ 2) = 0)
    (hchord : (1 / 2 : ℝ) * Real.log
      ((1 + 4 * (c_star s / s ^ 2)) / (1 + 2 * (c_star s / s ^ 2))) ≤
        R_real (c_star s / s ^ 2))
    (htail : |Gfun (R_real (c_star s / s ^ 2)) ((c_star s / s ^ 2) * R') -
      partialSum δ φ (c_star s / s ^ 2)| ≤
      (25 / 9) * δ ^ 2 * (10 * (c_star s / s ^ 2)) ^ 9 /
        (1 - 10 * (c_star s / s ^ 2))) :
    deriv (fun u => Real.log (c_star u / (1 - u))) s < 0 := by
  have hs0 : 0 < s := lt_of_lt_of_le (by positivity) hs
  obtain ⟨_, hq, he, hD, hsign⟩ :=
    derivative_identity s R' c_star R_real hs0 hs1 hc_pos hc_cont hR_sderiv hV_pos h_thresh
  have hq1 := q_le_of_chord (c_star s / s ^ 2) (R_real (c_star s / s ^ 2)) s
    hq.le hs he hchord
  have hg := (sign_of_remainder δ φ (c_star s / s ^ 2)
    (Gfun (R_real (c_star s / s ^ 2)) ((c_star s / s ^ 2) * R'))
    hδ0 hδ1.le hφ0 hφ1.le hq hq1 htail).2
  rw [hD.deriv]
  exact hsign.mpr hg

end Dobson.GapOneTail

#print axioms Dobson.GapOneTail.derivative_identity

#print axioms Dobson.GapOneTail.coefficient_bounds

#print axioms Dobson.GapOneTail.partialSum_lower

#print axioms Dobson.GapOneTail.lowerMargin_endpoint

#print axioms Dobson.GapOneTail.lowerMargin_lower

#print axioms Dobson.GapOneTail.sign_of_remainder

#print axioms Dobson.GapOneTail.threshold_square

#print axioms Dobson.GapOneTail.q_le_of_chord

#print axioms Dobson.GapOneTail.tail_derivative_neg
