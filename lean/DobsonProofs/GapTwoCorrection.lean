import Mathlib.Analysis.Asymptotics.Lemmas
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Normed.Group.Bounded
import Mathlib.Analysis.SpecialFunctions.Artanh
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Probability.IdentDistrib
import Mathlib.Tactic

/-!
# Small-staying correction for the two-stage Markov host (items Q1 to Q12)

Machine-checked algebraic, elementary-analytic, and conditional measure-theoretic core of GAP 2
(`gap2/gap2_proof.md`; independently checked in the small-staying review): for fixed `0 < δ < 1` and `-1 < φ < 1`,
with `b = artanh δ`, `ℓ = -(1/2)·log(1 - δ²)`, `L₀ = log(1/s)` and
`V_π = artanh(δ)²·(1 - φ)/(1 + φ)`, the zero-growth threshold loss `D(s) = log(c_*(s)/(1 - s))`
of `x_{t+1} = s·x_t + c·(1 + δ·Z_t)·x_{t-1}` satisfies
`D(s) - ℓ = -V_π / (2·L₀) + O_{δ,φ}(log L₀ / L₀²)` as `s ↓ 0`.

## Definitions

* `ell δ = -(1/2)·log(1 - δ²)` (the logarithmic Jensen endpoint `ℓ = -E[log h]`).
* `Vpi δ φ = (artanh δ)²·(1 - φ)/(1 + φ)` (using Mathlib's `Real.artanh`).
* `Fstep L η x = log(exp(-L) + exp(η - x))` and `Qstep L x η = log(1 + exp(x - η - L))`
  (the normalized log-ratio map and soft-boundary reflection term of `gap2_proof.md` eq. `(2)`).
* `Qstar b = log(1 + exp(2b + 1))`, `Cplus b φ = 2(b + 1) + 2b/(1 + φ)`, and
  `Cminus b φ = 2b/(1 + φ) + Qstar b` (the explicit constants of `gap2_proof.md` §3).
* `altVar b φ n` (the exact variance of the alternating two-step cohort sum
  `∑_{t=0}^{n-1} (-1)^t η_t` under lag-`j` covariance `b²·φ^j`).
* `Lmin b φ`, `Kq b φ`, `L0min b φ`, `KD δ φ` (explicit closed-form thresholds and remainder
  constants depending only on the fixed host parameters).

## Explicit probability, invariant-law, and Lyapunov hypotheses (`H1` to `H6`)

Where a step requires random-matrix-product or Markov-chain probability theory not in Mathlib
(`memA` Proposition P3), it is stated as an explicit hypothesis of the Lean theorem (`6` distinct
bridge inputs in total across the file):

1. **`H1` (`h_stat`, joint stationarity of the pullback law, `memA` P3 / `gap2_proof.md` §1):**
   `IdentDistrib (fun ω => (Fstep L (b * Z_curr ω) (X_prev ω), Z_curr ω))`
   `(fun ω => (X_prev ω, Z_prev ω)) μ μ` on a probability space `(Ω, μ)`.
2. **`H2` (`h_Z_pm`, binary host marginal support, `gap2_proof.md` §1):**
   `∀ᵐ ω ∂μ, Z_curr ω = 1 ∨ Z_curr ω = -1`.
   (From `H1` + `H2`, a.e. compact support `X_prev ∈ [-L, log(e^{-L} + e^{L+b})]` and
   Bochner integrability of all continuous observables of `(X_prev, Z_prev, Z_curr)` are *proved*
   in Lean, not assumed.)
3. **`H3` (`h_Z_zero`, symmetric zero-mean host marginal, `gap2_proof.md` §1):**
   `∫ ω, Z_curr ω ∂μ = 0`.
4. **`H4` (`h_markov`, past measurability and Markov conditional regression, `gap2_proof.md` §2):**
   `∫ ω, X_prev ω * Z_curr ω ∂μ = φ * ∫ ω, X_prev ω * Z_prev ω ∂μ`.
5. **`H5` (`h_autocorr`, one-step Markov autocorrelation, `gap2_proof.md` Dated Algebra Audit):**
   `∫ ω, Z_prev ω * Z_curr ω ∂μ = φ` (used only in `Q6`, the `W_t, ξ_t` cross-check; not used in
   `Q5` or the main theorems `Q11` and `Q12`).
6. **`H6` (`h_zero_growth`, Furstenberg–Kifer / `memA` P3 Lyapunov growth rate at threshold,
   `gap2_proof.md` eqs. `(4)` and `(11)`):**
   `(Real.log c_star - ell δ) + 2 * (∫ ω, X_prev ω ∂μ) = 0`
   (i.e. `2·r(s, c_*(s)) = m_* + 2·E[X] = 0`).
Dated closure, 2026-11-16 (`leanp1`): former input `H7`, the constant-host threshold bracket,
is now proved by `stationary_threshold_bracket` from the stationary ratio law and zero growth,
using the logarithmic potential `log(U + 1-s)`. `Q10`, `Q11`, and `Q12` explicitly use the
physical parameter domain `c_star > 0`; no comparison bound is supplied as a hypothesis.
-/

namespace Dobson.GapTwoCorrection

open MeasureTheory ProbabilityTheory Filter Set
open scoped Topology Asymptotics

/-! ## Definitions of model quantities and explicit error-term constants -/

/-- `ℓ(δ) = -(1/2)·log(1 - δ²)`. -/
noncomputable def ell (δ : ℝ) : ℝ := -(1 / 2 : ℝ) * Real.log (1 - δ ^ 2)

/-- `V_π(δ, φ) = artanh(δ)²·(1 - φ)/(1 + φ)`. -/
noncomputable def Vpi (δ φ : ℝ) : ℝ := (Real.artanh δ) ^ 2 * (1 - φ) / (1 + φ)

/-- One-step normalized log-ratio map `F_L(η, x) = log(exp(-L) + exp(η - x))`. -/
noncomputable def Fstep (L η x : ℝ) : ℝ := Real.log (Real.exp (-L) + Real.exp (η - x))

/-- Soft-boundary reflection term `Q_L(x, η) = log(1 + exp(x - η - L))`. -/
noncomputable def Qstep (L x η : ℝ) : ℝ := Real.log (1 + Real.exp (x - η - L))

/-- Uniform soft-boundary cap `Q_*(b) = log(1 + exp(2b + 1))`. -/
noncomputable def Qstar (b : ℝ) : ℝ := Real.log (1 + Real.exp (2 * b + 1))

/-- Upper soft-boundary constant `C_+(b, φ) = 2(b + 1) + 2b/(1 + φ)`. -/
noncomputable def Cplus (b φ : ℝ) : ℝ := 2 * (b + 1) + 2 * b / (1 + φ)

/-- Lower soft-boundary constant `C_-(b, φ) = 2b/(1 + φ) + Q_*(b)`. -/
noncomputable def Cminus (b φ : ℝ) : ℝ := 2 * b / (1 + φ) + Qstar b

/-- Variance of the alternating sum `∑_{t=0}^{n-1} (-1)^t η_t` under
`Cov(η_i, η_j) = b²·φ^{|i-j|}`. -/
noncomputable def altVar (b φ : ℝ) (n : ℕ) : ℝ :=
  ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n,
    (-1 : ℝ) ^ (i + j) * (b ^ 2 * φ ^ Int.natAbs ((i : ℤ) - (j : ℤ)))

/-- Explicit threshold `L_min(b, φ) = (8 + C_-(b, φ))²` above which `M = 2·log L ≥ 1` and
`2L - 4·log L - C_- ≥ L`. -/
noncomputable def Lmin (b φ : ℝ) : ℝ := (8 + Cminus b φ) ^ 2

/-- Explicit remainder constant `K_q(b, φ)` in `|q_L - V_π/(2L)| ≤ K_q·(log L / L²)`. -/
noncomputable def Kq (b φ : ℝ) : ℝ :=
  let V := b ^ 2 * (1 - φ) / (1 + φ)
  max (V * Cplus b φ / 4) ((V * (4 + Cminus b φ) + 8 * Real.exp b) / 2)

/-- Explicit small-staying threshold `L0_min(b, φ)` on `L₀ = log(1/s)`. -/
noncomputable def L0min (b φ : ℝ) : ℝ := max (2 * Lmin b φ) (2 * (b + 1))

/-- Explicit remainder constant `K_D(δ, φ)` in
`|(D(s) - ℓ) - (-V_π/(2·L₀))| ≤ K_D·(log L₀ / L₀²)`. -/
noncomputable def KD (δ φ : ℝ) : ℝ :=
  let b := Real.artanh δ
  let K := Kq b φ
  let V := Vpi δ φ
  4 * K + V * (V + 4 * K) / 2 + 8

/-! ## Q1: `artanh(δ)²` as the variance of `log h`, mean `-ℓ`, and strict parameter signs -/

/-- Exact log-host values at `z = +1` and `z = -1`: `log(1 ± δ) = -ℓ(δ) ± artanh(δ)`. -/
theorem log_one_add_sub_eq (δ : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1) :
    Real.log (1 + δ) = -ell δ + Real.artanh δ
      ∧ Real.log (1 - δ) = -ell δ - Real.artanh δ := by
  have h1p : 0 < 1 + δ := by linarith
  have h1m : 0 < 1 - δ := by linarith
  have hart : Real.artanh δ = 1 / 2 * Real.log ((1 + δ) / (1 - δ)) :=
    Real.artanh_eq_half_log (mem_Icc.mpr ⟨by linarith, hδ1.le⟩)
  have hdiv : Real.log ((1 + δ) / (1 - δ)) = Real.log (1 + δ) - Real.log (1 - δ) :=
    Real.log_div h1p.ne' h1m.ne'
  have hprod : 1 - δ ^ 2 = (1 + δ) * (1 - δ) := by ring
  have hmul : Real.log (1 - δ ^ 2) = Real.log (1 + δ) + Real.log (1 - δ) := by
    rw [hprod, Real.log_mul h1p.ne' h1m.ne']
  unfold ell
  rw [hart, hdiv, hmul]
  constructor <;> ring

/-- Helper: `tanh x < x` for `x > 0` (from `sinh x < x·cosh x`). -/
theorem tanh_lt_self_of_pos {x : ℝ} (hx : 0 < x) : Real.tanh x < x := by
  have hderiv : ∀ t : ℝ,
      HasDerivAt (fun u : ℝ => Real.sinh u - u * Real.cosh u) (-(t * Real.sinh t)) t := by
    intro t
    have h1 : HasDerivAt (fun u : ℝ => u * Real.cosh u) (1 * Real.cosh t + t * Real.sinh t) t :=
      (hasDerivAt_id' t).fun_mul (Real.hasDerivAt_cosh t)
    have h2 := (Real.hasDerivAt_sinh t).fun_sub h1
    have e : Real.cosh t - (1 * Real.cosh t + t * Real.sinh t) = -(t * Real.sinh t) := by ring
    rw [e] at h2
    exact h2
  have hanti : StrictAntiOn (fun u : ℝ => Real.sinh u - u * Real.cosh u) (Ici 0) := by
    refine strictAntiOn_of_deriv_neg (convex_Ici 0) ?_ ?_
    · exact fun t _ => (hderiv t).continuousAt.continuousWithinAt
    · intro t ht
      rw [interior_Ici] at ht
      rw [(hderiv t).deriv]
      have hp : 0 < t * Real.sinh t := mul_pos ht (Real.sinh_pos_iff.mpr ht)
      linarith
  have hlt := hanti (mem_Ici.mpr le_rfl) (mem_Ici.mpr hx.le) hx
  simp only [Real.sinh_zero, zero_mul, sub_zero] at hlt
  rw [Real.tanh_eq_sinh_div_cosh, div_lt_iff₀ (Real.cosh_pos x)]
  linarith

/-- **Q1 (log-host representation, symmetric two-point mean and variance, and strict signs).**
For `0 < δ < 1` and `-1 < φ < 1`:
1. For `z ∈ {1, -1}`, `1 + δ·z > 0`, `log(1 + δ·z) = -ℓ(δ) + artanh(δ)·z`, and
   `1 + δ·z = exp(-ℓ(δ) + artanh(δ)·z)`.
2. Under the symmetric two-point law `(1/2, 1/2)`, `log(1 + δ·Z)` has mean `-ℓ(δ)` and variance
   `artanh(δ)²`.
3. `0 < ℓ(δ)`, `0 < artanh(δ)`, `δ² < artanh(δ)²` (strict separation of `Var(h) = δ²` from
   `Var(log h) = artanh(δ)²`), and `0 < V_π(δ, φ)`. -/
theorem Q1 (δ φ : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1) (hφ_lo : -1 < φ) (hφ_hi : φ < 1) :
    (∀ z : ℝ, z = 1 ∨ z = -1 →
      0 < 1 + δ * z ∧ Real.log (1 + δ * z) = -ell δ + Real.artanh δ * z
        ∧ 1 + δ * z = Real.exp (-ell δ + Real.artanh δ * z))
      ∧ (1 / 2 : ℝ) * Real.log (1 + δ) + (1 / 2 : ℝ) * Real.log (1 - δ) = -ell δ
      ∧ (1 / 2 : ℝ) * (Real.log (1 + δ) - ((1 / 2 : ℝ) * Real.log (1 + δ)
          + (1 / 2 : ℝ) * Real.log (1 - δ))) ^ 2
        + (1 / 2 : ℝ) * (Real.log (1 - δ) - ((1 / 2 : ℝ) * Real.log (1 + δ)
          + (1 / 2 : ℝ) * Real.log (1 - δ))) ^ 2 = (Real.artanh δ) ^ 2
      ∧ 0 < ell δ ∧ 0 < Real.artanh δ ∧ δ ^ 2 < (Real.artanh δ) ^ 2 ∧ 0 < Vpi δ φ := by
  obtain ⟨hlp, hlm⟩ := log_one_add_sub_eq δ hδ0 hδ1
  have hbpos : 0 < Real.artanh δ := Real.artanh_pos (mem_Ioo.mpr ⟨hδ0, hδ1⟩)
  have h1sq : 0 < 1 - δ ^ 2 := by nlinarith
  have h1sq_lt : 1 - δ ^ 2 < 1 := by nlinarith
  have hell_pos : 0 < ell δ := by
    unfold ell
    have hlogneg : Real.log (1 - δ ^ 2) < 0 := Real.log_neg h1sq h1sq_lt
    linarith
  have hδ_lt_b : δ < Real.artanh δ := by
    have ht := tanh_lt_self_of_pos hbpos
    rwa [Real.tanh_artanh (mem_Ioo.mpr ⟨by linarith, hδ1⟩)] at ht
  have hsq_lt : δ ^ 2 < (Real.artanh δ) ^ 2 := by nlinarith
  have hVpos : 0 < Vpi δ φ := by
    unfold Vpi
    exact div_pos (mul_pos (pow_pos hbpos 2) (by linarith)) (by linarith)
  refine ⟨?_, by rw [hlp, hlm]; ring, by rw [hlp, hlm]; ring,
    hell_pos, hbpos, hsq_lt, hVpos⟩
  intro z hz
  rcases hz with rfl | rfl
  · have hp : 0 < 1 + δ * 1 := by linarith
    have hl : Real.log (1 + δ * 1) = -ell δ + Real.artanh δ * 1 := by simpa using hlp
    exact ⟨hp, hl, by rw [← hl, Real.exp_log hp]⟩
  · have hm : 0 < 1 + δ * (-1) := by linarith
    have hl : Real.log (1 + δ * (-1)) = -ell δ + Real.artanh δ * (-1) := by
      simpa [sub_eq_add_neg] using hlm
    exact ⟨hm, hl, by rw [← hl, Real.exp_log hm]⟩

/-! ## Q2 and Q3: Normalized recursion eq. `(2)` and deterministic support bounds eq. `(3)` -/

/-- Exact decomposition `F_L(η, x) = -x + η + Q_L(x, η)` for all real `L, x, η`. -/
theorem Fstep_eq (L x η : ℝ) : Fstep L η x = -x + η + Qstep L x η := by
  unfold Fstep Qstep
  have h1 : 0 < Real.exp (η - x) := Real.exp_pos _
  have h2 : 0 < 1 + Real.exp (x - η - L) := by positivity
  have hfac : Real.exp (-L) + Real.exp (η - x)
      = Real.exp (η - x) * (1 + Real.exp (x - η - L)) := by
    rw [mul_add, mul_one, ← Real.exp_add]
    have e : (η - x) + (x - η - L) = -L := by ring
    rw [e]
    ring
  rw [hfac, Real.log_mul h1.ne' h2.ne', Real.log_exp]
  ring

/-- **Q2 (exact normalization of the ratio step, `gap2_proof.md` eq. `(2)`).**
For `0 < δ < 1`, `0 < s`, `0 < c`, `0 < u` and `z ∈ {1, -1}`, set `b = artanh δ`,
`m = log c - ℓ(δ)`, `L = log(1/s) + m/2`, `x = log u - m/2`, `η = b·z`,
`u' = s + c·(1 + δ·z)/u`, and `x' = log u' - m/2`. Then `u' > 0`, `x' = F_L(η, x)`, and
`x' = -x + η + Q_L(x, η)`. -/
theorem Q2 (δ s c u z : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1) (hs : 0 < s) (hc : 0 < c)
    (hu : 0 < u) (hz : z = 1 ∨ z = -1) :
    let b := Real.artanh δ
    let m := Real.log c - ell δ
    let L := Real.log (1 / s) + m / 2
    let x := Real.log u - m / 2
    let η := b * z
    let u' := s + c * (1 + δ * z) / u
    let x' := Real.log u' - m / 2
    0 < u' ∧ x' = Fstep L η x ∧ x' = -x + η + Qstep L x η := by
  dsimp only
  obtain ⟨h1dz_all, _, _, _, _, _, _⟩ := Q1 δ 0 hδ0 hδ1 (by norm_num) (by norm_num)
  obtain ⟨h1dz_pos, _, h1dz_exp⟩ := h1dz_all z hz
  set b := Real.artanh δ
  set m := Real.log c - ell δ
  set L := Real.log (1 / s) + m / 2
  set x := Real.log u - m / 2
  set η := b * z
  have hu'_pos : 0 < s + c * (1 + δ * z) / u := by positivity
  have hs_eq : s = Real.exp (m / 2) * Real.exp (-L) := by
    have h1s : 0 < 1 / s := one_div_pos.mpr hs
    rw [← Real.exp_add]
    have hsum : m / 2 + -L = -Real.log (1 / s) := by unfold L; ring
    rw [hsum, Real.exp_neg, Real.exp_log h1s]
    field_simp
  have hc_eq : c = Real.exp (m + ell δ) := by
    have e : m + ell δ = Real.log c := by unfold m; ring
    rw [e, Real.exp_log hc]
  have hu_eq : u = Real.exp (x + m / 2) := by
    have e : x + m / 2 = Real.log u := by unfold x; ring
    rw [e, Real.exp_log hu]
  have hterm : c * (1 + δ * z) / u = Real.exp (m / 2) * Real.exp (η - x) := by
    rw [hc_eq, h1dz_exp, hu_eq, ← Real.exp_add, ← Real.exp_sub, ← Real.exp_add]
    congr 1
    unfold η
    ring
  have hu'_fac : s + c * (1 + δ * z) / u
      = Real.exp (m / 2) * (Real.exp (-L) + Real.exp (η - x)) := by
    rw [hs_eq, hterm, mul_add]
  have hx' : Real.log (s + c * (1 + δ * z) / u) - m / 2 = Fstep L η x := by
    have hsum_pos : 0 < Real.exp (-L) + Real.exp (η - x) := by positivity
    rw [hu'_fac, Real.log_mul (Real.exp_pos _).ne' hsum_pos.ne', Real.log_exp]
    unfold Fstep
    ring
  exact ⟨hu'_pos, hx', hx'.trans (Fstep_eq L x η)⟩

/-- One-step deterministic lower bound `-L < F_L(η, x)`. -/
theorem neg_lt_Fstep (L η x : ℝ) : -L < Fstep L η x := by
  unfold Fstep
  have hpos : 0 < Real.exp (η - x) := Real.exp_pos _
  have hlt : Real.exp (-L) < Real.exp (-L) + Real.exp (η - x) := by linarith
  have hlog := Real.log_lt_log (Real.exp_pos (-L)) hlt
  rwa [Real.log_exp] at hlog

/-- One-step deterministic upper bound when `-L ≤ x` and `η ≤ b`. -/
theorem Fstep_le_of_ge_neg (L b η x : ℝ) (hx : -L ≤ x) (hη : η ≤ b) :
    Fstep L η x ≤ Real.log (Real.exp (-L) + Real.exp (L + b)) := by
  unfold Fstep
  apply Real.log_le_log (by positivity)
  have hexp : Real.exp (η - x) ≤ Real.exp (L + b) := Real.exp_le_exp.mpr (by linarith)
  linarith

/-- Cap `log(exp(-L) + exp(L + b)) < L + b + 1` for `b ≥ 0` and `L ≥ 0`. -/
theorem log_exp_neg_add_exp_lt (L b : ℝ) (hb : 0 ≤ b) (hL : 0 ≤ L) :
    Real.log (Real.exp (-L) + Real.exp (L + b)) < L + b + 1 := by
  have hfac : Real.exp (-L) + Real.exp (L + b)
      = Real.exp (L + b) * (1 + Real.exp (-2 * L - b)) := by
    rw [mul_add, mul_one, ← Real.exp_add]
    have e : (L + b) + (-2 * L - b) = -L := by ring
    rw [e]
    ring
  have hexp_le : Real.exp (-2 * L - b) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have h2_lt_e : (2 : ℝ) < Real.exp 1 := by
    have h := Real.add_one_lt_exp (one_ne_zero)
    linarith
  have h1p_lt : 1 + Real.exp (-2 * L - b) < Real.exp 1 := by linarith
  have hlog_lt : Real.log (1 + Real.exp (-2 * L - b)) < 1 := by
    have h := Real.log_lt_log (by positivity) h1p_lt
    rwa [Real.log_exp] at h
  rw [hfac, Real.log_mul (Real.exp_pos _).ne' (by positivity), Real.log_exp]
  linarith

/-- **Q3 (two-step deterministic trapping and `L + b + 1` cap, `gap2_proof.md` eq. `(3)`).**
For any `b ≥ 0`, `L ≥ 1`, and `η₁, η₂, x₀` with `η₂ ≤ b`, two steps of `F_L` satisfy
`-L ≤ F_L(η₂, F_L(η₁, x₀)) ≤ log(exp(-L) + exp(L + b)) < L + b + 1`. -/
theorem Q3 (L b η₁ η₂ x₀ : ℝ) (hb : 0 ≤ b) (hL : 1 ≤ L) (hη₂ : η₂ ≤ b) :
    -L ≤ Fstep L η₂ (Fstep L η₁ x₀)
      ∧ Fstep L η₂ (Fstep L η₁ x₀) ≤ Real.log (Real.exp (-L) + Real.exp (L + b))
      ∧ Real.log (Real.exp (-L) + Real.exp (L + b)) < L + b + 1 := by
  have hstep1 : -L ≤ Fstep L η₁ x₀ := (neg_lt_Fstep L η₁ x₀).le
  exact ⟨(neg_lt_Fstep L η₂ (Fstep L η₁ x₀)).le,
    Fstep_le_of_ge_neg L b η₂ (Fstep L η₁ x₀) hstep1 hη₂,
    log_exp_neg_add_exp_lt L b hb (by linarith)⟩

/-! ## Q4: Memory factor `(1 - φ)/(1 + φ)` from the alternating two-step cohort sum -/

/-- Helper: one-sided geometric sum `∑_{i < m} (-φ)^{m - i}` multiplied by `1 + φ`. -/
theorem one_add_mul_sum_neg_pow_sub (φ : ℝ) (m : ℕ) :
    (1 + φ) * (∑ i ∈ Finset.range m, (-φ) ^ (m - i)) = -φ * (1 - (-φ) ^ m) := by
  induction m with
  | zero => simp
  | succ m ih =>
    have hsum : ∑ i ∈ Finset.range (m + 1), (-φ) ^ (m + 1 - i)
        = (-φ) * (∑ i ∈ Finset.range m, (-φ) ^ (m - i)) + (-φ) := by
      rw [Finset.sum_range_succ, show m + 1 - m = 1 by omega, pow_one]
      congr 1
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i hi => ?_
      have him : i < m := Finset.mem_range.mp hi
      rw [show m + 1 - i = (m - i) + 1 by omega, pow_succ']
    rw [hsum, mul_add, pow_succ]
    linear_combination -φ * ih

/-- Exact polynomial identity for the finite-`n` alternating-sum variance `altVar b φ n`. -/
theorem altVar_polynomial_identity (b φ : ℝ) (n : ℕ) :
    (1 + φ) ^ 2 * altVar b φ n
      = (n : ℝ) * b ^ 2 * (1 - φ ^ 2) + 2 * b ^ 2 * φ * (1 - (-φ) ^ n) := by
  unfold altVar
  induction n with
  | zero => simp
  | succ m ih =>
    -- Split the outer and inner `Finset.range (m + 1)` sums into `< m` and `= m`
    have hsplit :
        ∑ i ∈ Finset.range (m + 1), ∑ j ∈ Finset.range (m + 1),
          (-1 : ℝ) ^ (i + j) * (b ^ 2 * φ ^ Int.natAbs ((i : ℤ) - (j : ℤ)))
        = (∑ i ∈ Finset.range m, ∑ j ∈ Finset.range m,
            (-1 : ℝ) ^ (i + j) * (b ^ 2 * φ ^ Int.natAbs ((i : ℤ) - (j : ℤ))))
          + 2 * b ^ 2 * (∑ i ∈ Finset.range m, (-φ) ^ (m - i)) + b ^ 2 := by
      rw [Finset.sum_range_succ]
      simp_rw [Finset.sum_range_succ]
      rw [Finset.sum_add_distrib]
      have hdiag : (-1 : ℝ) ^ (m + m) * (b ^ 2 * φ ^ Int.natAbs ((m : ℤ) - (m : ℤ))) = b ^ 2 := by
        have h1 : (-1 : ℝ) ^ (m + m) = 1 := by
          rw [show m + m = 2 * m by omega, pow_mul]
          norm_num
        simp [h1]
      have hrow : ∀ i ∈ Finset.range m,
          (-1 : ℝ) ^ (i + m) * (b ^ 2 * φ ^ Int.natAbs ((i : ℤ) - (m : ℤ)))
            = b ^ 2 * (-φ) ^ (m - i) := by
        intro i hi
        have him : i < m := Finset.mem_range.mp hi
        have habs : Int.natAbs ((i : ℤ) - (m : ℤ)) = m - i := by omega
        have hsign : (-1 : ℝ) ^ (i + m) = (-1 : ℝ) ^ (m - i) := by
          have hsum_idx : i + m = 2 * i + (m - i) := by omega
          rw [hsum_idx, pow_add, pow_mul]
          norm_num
        rw [habs, hsign, neg_eq_neg_one_mul φ, mul_pow]
        ring
      have hcol : ∀ j ∈ Finset.range m,
          (-1 : ℝ) ^ (m + j) * (b ^ 2 * φ ^ Int.natAbs ((m : ℤ) - (j : ℤ)))
            = b ^ 2 * (-φ) ^ (m - j) := by
        intro j hj
        have hjm : j < m := Finset.mem_range.mp hj
        have habs : Int.natAbs ((m : ℤ) - (j : ℤ)) = m - j := by omega
        have hsign : (-1 : ℝ) ^ (m + j) = (-1 : ℝ) ^ (m - j) := by
          have hsum_idx : m + j = 2 * j + (m - j) := by omega
          rw [hsum_idx, pow_add, pow_mul]
          norm_num
        rw [habs, hsign, neg_eq_neg_one_mul φ, mul_pow]
        ring
      rw [Finset.sum_congr rfl hrow, Finset.sum_congr rfl hcol, hdiag, ← Finset.mul_sum]
      ring
    rw [hsplit]
    have hgeom := one_add_mul_sum_neg_pow_sub φ m
    push_cast
    rw [pow_succ']
    linear_combination ih + 2 * b ^ 2 * (1 + φ) * hgeom

/-- **Q4 (memory factor `(1 - φ)/(1 + φ)` from the alternating two-step cohort sum,
`gap2_proof.md` §2 lines 112–121).** For `-1 < φ < 1` and real `b`:
1. Exact finite-`n` formula (`n ≥ 1`):
   `altVar b φ n / n = b²·(1 - φ)/(1 + φ) + 2·b²·φ·(1 - (-φ)^n) / (n·(1 + φ)²)`.
2. Infinite lag-series identity: `HasSum (fun j : ℕ => (-φ)^(j + 1)) (-φ / (1 + φ))`, hence
   `b²·(1 + 2·∑'_{j=0}^∞ (-φ)^{j+1}) = b²·(1 - φ)/(1 + φ)`.
3. Variance-rate limit: `altVar b φ n / n → b²·(1 - φ)/(1 + φ)` as `n → ∞`. -/
theorem Q4 (b φ : ℝ) (hφ_lo : -1 < φ) (hφ_hi : φ < 1) :
    (∀ n : ℕ, 1 ≤ n →
      altVar b φ n / (n : ℝ)
        = b ^ 2 * (1 - φ) / (1 + φ)
          + 2 * b ^ 2 * φ * (1 - (-φ) ^ n) / ((n : ℝ) * (1 + φ) ^ 2))
      ∧ HasSum (fun j : ℕ => (-φ) ^ (j + 1)) (-φ / (1 + φ))
      ∧ b ^ 2 * (1 + 2 * ∑' j : ℕ, (-φ) ^ (j + 1)) = b ^ 2 * (1 - φ) / (1 + φ)
      ∧ Tendsto (fun n : ℕ => altVar b φ n / (n : ℝ)) atTop
          (nhds (b ^ 2 * (1 - φ) / (1 + φ))) := by
  have h1p : 0 < 1 + φ := by linarith
  have h1p_ne : 1 + φ ≠ 0 := h1p.ne'
  have habs : |-φ| < 1 := by
    rw [abs_neg, abs_lt]
    exact ⟨hφ_lo, hφ_hi⟩
  -- Part 1: exact finite-`n` formula
  have hfin : ∀ n : ℕ, 1 ≤ n →
      altVar b φ n / (n : ℝ)
        = b ^ 2 * (1 - φ) / (1 + φ)
          + 2 * b ^ 2 * φ * (1 - (-φ) ^ n) / ((n : ℝ) * (1 + φ) ^ 2) := by
    intro n hn
    have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
    have hn_ne : (n : ℝ) ≠ 0 := hn_pos.ne'
    have hid := altVar_polynomial_identity b φ n
    apply mul_left_cancel₀ (mul_ne_zero hn_ne (pow_ne_zero 2 h1p_ne))
    calc (n : ℝ) * (1 + φ) ^ 2 * (altVar b φ n / (n : ℝ))
        = (1 + φ) ^ 2 * altVar b φ n := by field_simp
      _ = (n : ℝ) * b ^ 2 * (1 - φ ^ 2) + 2 * b ^ 2 * φ * (1 - (-φ) ^ n) := hid
      _ = (n : ℝ) * (1 + φ) ^ 2 * (b ^ 2 * (1 - φ) / (1 + φ)
            + 2 * b ^ 2 * φ * (1 - (-φ) ^ n) / ((n : ℝ) * (1 + φ) ^ 2)) := by
          field_simp
          ring
  -- Part 2: geometric lag series
  have hgeom0 : HasSum (fun j : ℕ => (-φ) ^ j) ((1 - (-φ))⁻¹) :=
    hasSum_geometric_of_abs_lt_one habs
  have hgeom : HasSum (fun j : ℕ => (-φ) ^ (j + 1)) (-φ / (1 + φ)) := by
    have hmul := hgeom0.mul_left (-φ)
    have heq_fn : (fun j : ℕ => (-φ) * (-φ) ^ j) = (fun j : ℕ => (-φ) ^ (j + 1)) := by
      ext j; rw [pow_succ']
    have heq_val : (-φ) * (1 - -φ)⁻¹ = -φ / (1 + φ) := by
      rw [sub_neg_eq_add, div_eq_mul_inv]
    rwa [heq_fn, heq_val] at hmul
  have htsum : b ^ 2 * (1 + 2 * ∑' j : ℕ, (-φ) ^ (j + 1)) = b ^ 2 * (1 - φ) / (1 + φ) := by
    rw [hgeom.tsum_eq]
    field_simp
    ring
  -- Part 3: limit as `n → ∞`
  have hlim : Tendsto (fun n : ℕ => altVar b φ n / (n : ℝ)) atTop
      (nhds (b ^ 2 * (1 - φ) / (1 + φ))) := by
    have hpow0 : Tendsto (fun n : ℕ => (-φ) ^ n) atTop (nhds 0) :=
      tendsto_pow_atTop_nhds_zero_of_abs_lt_one habs
    have hnum : Tendsto (fun n : ℕ => 2 * b ^ 2 * φ * (1 - (-φ) ^ n)) atTop
        (nhds (2 * b ^ 2 * φ * (1 - 0))) :=
      tendsto_const_nhds.mul (tendsto_const_nhds.sub hpow0)
    have hinv : Tendsto (fun n : ℕ => ((1 + φ) ^ 2)⁻¹ * (1 / (n : ℝ))) atTop
        (nhds (((1 + φ) ^ 2)⁻¹ * 0)) :=
      tendsto_const_nhds.mul tendsto_one_div_atTop_nhds_zero_nat
    have hrem0 : Tendsto
        (fun n : ℕ => 2 * b ^ 2 * φ * (1 - (-φ) ^ n) / ((n : ℝ) * (1 + φ) ^ 2))
        atTop (nhds 0) := by
      have hprod := hnum.mul hinv
      simp only [sub_zero, mul_zero] at hprod
      refine hprod.congr fun n => ?_
      rw [one_div, div_eq_mul_inv, mul_inv, mul_comm ((n : ℝ))⁻¹]
    have hrhs : Tendsto
        (fun n : ℕ => b ^ 2 * (1 - φ) / (1 + φ)
          + 2 * b ^ 2 * φ * (1 - (-φ) ^ n) / ((n : ℝ) * (1 + φ) ^ 2))
        atTop (nhds (b ^ 2 * (1 - φ) / (1 + φ))) := by
      simpa using tendsto_const_nhds.add hrem0
    refine hrhs.congr' ?_
    filter_upwards [eventually_ge_atTop 1] with n hn
    exact (hfin n hn).symm
  exact ⟨hfin, hgeom, htsum, hlim⟩

/-! ## Measure-theoretic stationary support & integrability from `H1` + `H2`, and items Q5 & Q6 -/

/-- From `H1` (`IdentDistrib` of the joint pair) and `H2` (`Z_curr ∈ {1, -1}` a.e.), we deduce:
- `Z_prev ∈ {1, -1}` a.e.,
- `-L ≤ X_prev ≤ log(exp(-L) + exp(L + b))` a.e. (via two applications of `H1`),
- `AEStronglyMeasurable` for `X_prev`, `Z_prev`, `Z_curr`. -/
theorem stationary_support_ae {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (L b : ℝ) (hb : 0 ≤ b)
    (X_prev Z_prev Z_curr : Ω → ℝ)
    (h_stat : IdentDistrib
      (fun ω => (Fstep L (b * Z_curr ω) (X_prev ω), Z_curr ω))
      (fun ω => (X_prev ω, Z_prev ω)) μ μ)
    (h_Z_pm : ∀ᵐ ω ∂μ, Z_curr ω = 1 ∨ Z_curr ω = -1) :
    AEStronglyMeasurable X_prev μ
      ∧ AEStronglyMeasurable Z_prev μ
      ∧ AEStronglyMeasurable Z_curr μ
      ∧ (∀ᵐ ω ∂μ, Z_prev ω = 1 ∨ Z_prev ω = -1)
      ∧ (∀ᵐ ω ∂μ, -L ≤ X_prev ω ∧ X_prev ω ≤ Real.log (Real.exp (-L) + Real.exp (L + b))) := by
  have hX_meas : AEStronglyMeasurable X_prev μ :=
    (h_stat.symm.comp measurable_fst).aestronglyMeasurable_fst
  have hZp_meas : AEStronglyMeasurable Z_prev μ :=
    (h_stat.symm.comp measurable_snd).aestronglyMeasurable_fst
  have hZc_meas : AEStronglyMeasurable Z_curr μ :=
    (h_stat.comp measurable_snd).aestronglyMeasurable_fst
  have hZ_id : IdentDistrib Z_curr Z_prev μ μ := h_stat.comp measurable_snd
  have hX_id : IdentDistrib (fun ω => Fstep L (b * Z_curr ω) (X_prev ω)) X_prev μ μ :=
    h_stat.comp measurable_fst
  have hZp_pm : ∀ᵐ ω ∂μ, Z_prev ω = 1 ∨ Z_prev ω = -1 := by
    have hset : MeasurableSet {z : ℝ | z = 1 ∨ z = -1} :=
      (isClosed_singleton.union isClosed_singleton).measurableSet
    exact hZ_id.ae_snd hset h_Z_pm
  -- First pass of `hX_id`: `-L ≤ X_prev` a.e.
  have h_lo_step : ∀ᵐ ω ∂μ, -L ≤ Fstep L (b * Z_curr ω) (X_prev ω) :=
    ae_of_all μ fun ω => (neg_lt_Fstep L (b * Z_curr ω) (X_prev ω)).le
  have h_lo : ∀ᵐ ω ∂μ, -L ≤ X_prev ω :=
    hX_id.ae_snd measurableSet_Ici h_lo_step
  -- Second pass of `hX_id`: `X_prev ≤ log(exp(-L) + exp(L + b))` a.e.
  have h_both_step : ∀ᵐ ω ∂μ, Fstep L (b * Z_curr ω) (X_prev ω)
      ∈ Icc (-L) (Real.log (Real.exp (-L) + Real.exp (L + b))) := by
    filter_upwards [h_lo, h_Z_pm] with ω hlo hpm
    refine ⟨(neg_lt_Fstep L (b * Z_curr ω) (X_prev ω)).le, ?_⟩
    have hη_le : b * Z_curr ω ≤ b := by
      rcases hpm with h | h <;> rw [h] <;> linarith
    exact Fstep_le_of_ge_neg L b (b * Z_curr ω) (X_prev ω) hlo hη_le
  have h_both : ∀ᵐ ω ∂μ,
      X_prev ω ∈ Icc (-L) (Real.log (Real.exp (-L) + Real.exp (L + b))) :=
    hX_id.ae_snd measurableSet_Icc h_both_step
  exact ⟨hX_meas, hZp_meas, hZc_meas, hZp_pm, h_both⟩

/-- Every continuous function of `(X_prev, Z_prev, Z_curr)` is Bochner-integrable on `(Ω, μ)`
once `(X_prev, Z_prev, Z_curr)` lies a.e. in a compact box. -/
theorem integrable_of_continuous_triple {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (L b : ℝ) (hb : 0 ≤ b)
    (X_prev Z_prev Z_curr : Ω → ℝ)
    (h_stat : IdentDistrib
      (fun ω => (Fstep L (b * Z_curr ω) (X_prev ω), Z_curr ω))
      (fun ω => (X_prev ω, Z_prev ω)) μ μ)
    (h_Z_pm : ∀ᵐ ω ∂μ, Z_curr ω = 1 ∨ Z_curr ω = -1)
    (G : ℝ × ℝ × ℝ → ℝ) (hG : Continuous G) :
    Integrable (fun ω => G (X_prev ω, Z_prev ω, Z_curr ω)) μ := by
  obtain ⟨hX_m, hZp_m, hZc_m, hZp_pm, hX_bd⟩ :=
    stationary_support_ae μ L b hb X_prev Z_prev Z_curr h_stat h_Z_pm
  set hi := Real.log (Real.exp (-L) + Real.exp (L + b))
  set K : Set (ℝ × ℝ × ℝ) := Icc (-L) hi ×ˢ (Icc (-1 : ℝ) 1 ×ˢ Icc (-1 : ℝ) 1)
  have hK_comp : IsCompact K := isCompact_Icc.prod (isCompact_Icc.prod isCompact_Icc)
  obtain ⟨M, hM⟩ := hK_comp.exists_bound_of_continuousOn hG.continuousOn
  have h_meas : AEStronglyMeasurable (fun ω => G (X_prev ω, Z_prev ω, Z_curr ω)) μ :=
    hG.comp_aestronglyMeasurable (hX_m.prodMk (hZp_m.prodMk hZc_m))
  refine Integrable.of_bound h_meas M ?_
  filter_upwards [hX_bd, hZp_pm, h_Z_pm] with ω hx hzp hzc
  apply hM (X_prev ω, Z_prev ω, Z_curr ω)
  refine ⟨hx, ?_, ?_⟩
  · rcases hzp with h | h <;> rw [h] <;> constructor <;> norm_num
  · rcases hzc with h | h <;> rw [h] <;> constructor <;> norm_num

/-- Continuity of `(x, η) ↦ Q_L(x, η)`. -/
theorem continuous_Qstep (L : ℝ) : Continuous (fun p : ℝ × ℝ => Qstep L p.1 p.2) := by
  unfold Qstep
  refine Continuous.log ?_ (fun _ => by positivity)
  exact continuous_const.add
    (Real.continuous_exp.comp ((continuous_fst.sub continuous_snd).sub continuous_const))

/-- **Q5 (stationary first-moment identity eq. `(4)`, correlation elimination eq. `(5)`, and
exact second-moment identity eq. `(6)` of `gap2_proof.md`).**
Assume `H1` (`h_stat`), `H2`/`H3` (`h_Z_pm`, `h_Z_zero`), and `H4` (`h_markov`), with `b ≥ 0` and
`-1 < φ < 1`. Put `X_curr = F_L(b·Z_curr, X_prev)`, `η_prev = b·Z_prev`, `η_curr = b·Z_curr`,
and `Q = Q_L(X_prev, η_curr)`. Then:
1. `-L ≤ X_prev ≤ log(exp(-L) + exp(L + b))` a.e. (`eq. (3)`).
2. `2·E[X_prev] = E[Q]` (`eq. (4)`).
3. `(1 + φ)·E[X_curr·η_curr] = b² + E[η_curr·Q]` (`eq. (5)`).
4. `b²·(1 - φ)/(1 + φ) = E[(2·X_prev - 2·η_curr/(1 + φ) - Q)·Q]` (`eq. (6)`). -/
theorem Q5 {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (L b φ : ℝ) (hb : 0 ≤ b) (hφ_lo : -1 < φ) (_hφ_hi : φ < 1)
    (X_prev Z_prev Z_curr : Ω → ℝ)
    (h_stat : IdentDistrib
      (fun ω => (Fstep L (b * Z_curr ω) (X_prev ω), Z_curr ω))
      (fun ω => (X_prev ω, Z_prev ω)) μ μ)
    (h_Z_pm : ∀ᵐ ω ∂μ, Z_curr ω = 1 ∨ Z_curr ω = -1)
    (h_Z_zero : ∫ ω, Z_curr ω ∂μ = 0)
    (h_markov : ∫ ω, X_prev ω * Z_curr ω ∂μ = φ * ∫ ω, X_prev ω * Z_prev ω ∂μ) :
    let X_curr := fun ω => Fstep L (b * Z_curr ω) (X_prev ω)
    let η_curr := fun ω => b * Z_curr ω
    let Q := fun ω => Qstep L (X_prev ω) (b * Z_curr ω)
    (∀ᵐ ω ∂μ, -L ≤ X_prev ω ∧ X_prev ω ≤ Real.log (Real.exp (-L) + Real.exp (L + b)))
      ∧ 2 * (∫ ω, X_prev ω ∂μ) = ∫ ω, Q ω ∂μ
      ∧ (1 + φ) * (∫ ω, X_curr ω * η_curr ω ∂μ) = b ^ 2 + ∫ ω, η_curr ω * Q ω ∂μ
      ∧ b ^ 2 * (1 - φ) / (1 + φ)
          = ∫ ω, (2 * X_prev ω - 2 * η_curr ω / (1 + φ) - Q ω) * Q ω ∂μ := by
  clear _hφ_hi
  dsimp only
  have h1p : 0 < 1 + φ := by linarith
  have h1p_ne : 1 + φ ≠ 0 := h1p.ne'
  obtain ⟨_, _, _, _, hX_bd⟩ := stationary_support_ae μ L b hb X_prev Z_prev Z_curr h_stat h_Z_pm
  set X_curr : Ω → ℝ := fun ω => Fstep L (b * Z_curr ω) (X_prev ω)
  set η_curr : Ω → ℝ := fun ω => b * Z_curr ω
  set Q : Ω → ℝ := fun ω => Qstep L (X_prev ω) (b * Z_curr ω)
  have hQ_cont : Continuous (fun p : ℝ × ℝ × ℝ => Qstep L p.1 (b * p.2.2)) :=
    (continuous_Qstep L).comp
      (continuous_fst.prodMk (continuous_const.mul (continuous_snd.comp continuous_snd)))
  have int_triple : ∀ G : ℝ × ℝ × ℝ → ℝ, Continuous G →
      Integrable (fun ω => G (X_prev ω, Z_prev ω, Z_curr ω)) μ :=
    integrable_of_continuous_triple μ L b hb X_prev Z_prev Z_curr h_stat h_Z_pm
  -- Integrability of all observables used below
  have iX : Integrable X_prev μ := int_triple (fun p => p.1) continuous_fst
  have iη : Integrable η_curr μ :=
    int_triple (fun p => b * p.2.2) (continuous_const.mul (continuous_snd.comp continuous_snd))
  have iQ : Integrable Q μ := int_triple _ hQ_cont
  have iXη : Integrable (fun ω => X_prev ω * η_curr ω) μ :=
    int_triple (fun p => p.1 * (b * p.2.2))
      (continuous_fst.mul (continuous_const.mul (continuous_snd.comp continuous_snd)))
  have iη2 : Integrable (fun ω => η_curr ω ^ 2) μ :=
    int_triple (fun p => (b * p.2.2) ^ 2)
      ((continuous_const.mul (continuous_snd.comp continuous_snd)).pow 2)
  have iηQ : Integrable (fun ω => η_curr ω * Q ω) μ :=
    int_triple (fun p => (b * p.2.2) * Qstep L p.1 (b * p.2.2))
      ((continuous_const.mul (continuous_snd.comp continuous_snd)).mul hQ_cont)
  have iXQ : Integrable (fun ω => X_prev ω * Q ω) μ :=
    int_triple (fun p => p.1 * Qstep L p.1 (b * p.2.2)) (continuous_fst.mul hQ_cont)
  have iX2 : Integrable (fun ω => X_prev ω ^ 2) μ :=
    int_triple (fun p => p.1 ^ 2) (continuous_fst.pow 2)
  have iQ2 : Integrable (fun ω => Q ω ^ 2) μ := int_triple _ (hQ_cont.pow 2)
  -- Stationary equalities from `h_stat`
  have hX_id : IdentDistrib X_curr X_prev μ μ := h_stat.comp measurable_fst
  have hEX_eq : ∫ ω, X_curr ω ∂μ = ∫ ω, X_prev ω ∂μ := hX_id.integral_eq
  have hEX2_eq : ∫ ω, X_curr ω ^ 2 ∂μ = ∫ ω, X_prev ω ^ 2 ∂μ := hX_id.sq.integral_eq
  have hC_id : IdentDistrib (fun ω => X_curr ω * η_curr ω)
      (fun ω => X_prev ω * (b * Z_prev ω)) μ μ :=
    h_stat.comp (measurable_fst.mul (measurable_const.mul measurable_snd))
  have hC_eq : ∫ ω, X_curr ω * η_curr ω ∂μ = ∫ ω, X_prev ω * (b * Z_prev ω) ∂μ :=
    hC_id.integral_eq
  -- Basic host moments: `E[η_curr] = 0` and `E[η_curr²] = b²`
  have hEη : ∫ ω, η_curr ω ∂μ = 0 := by
    change ∫ ω, b * Z_curr ω ∂μ = 0
    rw [integral_const_mul, h_Z_zero, mul_zero]
  have hEη2 : ∫ ω, η_curr ω ^ 2 ∂μ = b ^ 2 := by
    have hae : (fun ω => η_curr ω ^ 2) =ᵐ[μ] fun _ => b ^ 2 := by
      filter_upwards [h_Z_pm] with ω hpm
      dsimp [η_curr]
      rcases hpm with h | h <;> rw [h] <;> ring
    rw [integral_congr_ae hae]
    simp
  -- Part 2 (`eq. (4)`): `2·E[X_prev] = E[Q]`
  have h_eq4 : 2 * (∫ ω, X_prev ω ∂μ) = ∫ ω, Q ω ∂μ := by
    have hint : ∫ ω, X_curr ω ∂μ
        = -(∫ ω, X_prev ω ∂μ) + (∫ ω, η_curr ω ∂μ) + ∫ ω, Q ω ∂μ := by
      have hpt : X_curr = fun ω => (-X_prev ω + η_curr ω) + Q ω := by
        ext ω; dsimp [X_curr, η_curr, Q]; rw [Fstep_eq]
      rw [hpt, integral_add (by exact iX.neg.add iη) iQ,
        integral_add (by exact iX.neg) iη, integral_neg]
    linarith [hEX_eq, hint, hEη]
  -- Cross-moment Markov regression: `E[X_prev·η_curr] = φ·C`
  set C := ∫ ω, X_curr ω * η_curr ω ∂μ
  have hEXprev_η : ∫ ω, X_prev ω * η_curr ω ∂μ = φ * C := by
    have h1 : ∫ ω, X_prev ω * η_curr ω ∂μ = b * ∫ ω, X_prev ω * Z_curr ω ∂μ := by
      have e : (fun ω => X_prev ω * η_curr ω) = fun ω => b * (X_prev ω * Z_curr ω) := by
        ext ω; dsimp [η_curr]; ring
      rw [e, integral_const_mul]
    have h2 : C = b * ∫ ω, X_prev ω * Z_prev ω ∂μ := by
      rw [show C = ∫ ω, X_prev ω * (b * Z_prev ω) ∂μ from hC_eq]
      have e : (fun ω => X_prev ω * (b * Z_prev ω)) = fun ω => b * (X_prev ω * Z_prev ω) := by
        ext ω; ring
      rw [e, integral_const_mul]
    rw [h1, h_markov, h2]
    ring
  -- Part 3 (`eq. (5)`): `(1 + φ)·C = b² + E[η_curr·Q]`
  have h_eq5 : (1 + φ) * C = b ^ 2 + ∫ ω, η_curr ω * Q ω ∂μ := by
    have hint : C = -(∫ ω, X_prev ω * η_curr ω ∂μ) + (∫ ω, η_curr ω ^ 2 ∂μ)
        + ∫ ω, η_curr ω * Q ω ∂μ := by
      change ∫ ω, X_curr ω * η_curr ω ∂μ = _
      have hpt : (fun ω => X_curr ω * η_curr ω)
          = fun ω => (-(X_prev ω * η_curr ω) + η_curr ω ^ 2) + η_curr ω * Q ω := by
        ext ω; dsimp [X_curr, η_curr, Q]; rw [Fstep_eq]; ring
      rw [hpt, integral_add (by exact iXη.neg.add iη2) iηQ,
        integral_add (by exact iXη.neg) iη2, integral_neg]
    linarith [hint, hEXprev_η, hEη2]
  -- Part 4 (`eq. (6)`): `b²·(1 - φ)/(1 + φ) = E[(2·X_prev - 2·η_curr/(1 + φ) - Q)·Q]`
  have h_eq6 : b ^ 2 * (1 - φ) / (1 + φ)
      = ∫ ω, (2 * X_prev ω - 2 * η_curr ω / (1 + φ) - Q ω) * Q ω ∂μ := by
    have hsq_int : ∫ ω, X_curr ω ^ 2 ∂μ
        = (((∫ ω, X_prev ω ^ 2 ∂μ) + (∫ ω, η_curr ω ^ 2 ∂μ))
            - 2 * (∫ ω, X_prev ω * η_curr ω ∂μ))
          + (2 * (∫ ω, η_curr ω * Q ω ∂μ) - 2 * (∫ ω, X_prev ω * Q ω ∂μ))
          + ∫ ω, Q ω ^ 2 ∂μ := by
      have hsq_pt : (fun ω => X_curr ω ^ 2) = fun ω =>
          (((X_prev ω ^ 2 + η_curr ω ^ 2) - 2 * (X_prev ω * η_curr ω))
            + (2 * (η_curr ω * Q ω) - 2 * (X_prev ω * Q ω))) + Q ω ^ 2 := by
        ext ω; dsimp [X_curr, η_curr, Q]; rw [Fstep_eq]; ring
      have i12 : Integrable
          (fun ω => ((X_prev ω ^ 2 + η_curr ω ^ 2) - 2 * (X_prev ω * η_curr ω))
            + (2 * (η_curr ω * Q ω) - 2 * (X_prev ω * Q ω))) μ :=
        ((iX2.add iη2).sub (iXη.const_mul 2)).add ((iηQ.const_mul 2).sub (iXQ.const_mul 2))
      have i1 : Integrable
          (fun ω => (X_prev ω ^ 2 + η_curr ω ^ 2) - 2 * (X_prev ω * η_curr ω)) μ :=
        (iX2.add iη2).sub (iXη.const_mul 2)
      have i2 : Integrable
          (fun ω => 2 * (η_curr ω * Q ω) - 2 * (X_prev ω * Q ω)) μ :=
        (iηQ.const_mul 2).sub (iXQ.const_mul 2)
      rw [hsq_pt, integral_add i12 iQ2, integral_add i1 i2,
        integral_sub (by exact (iX2.add iη2)) (by exact (iXη.const_mul 2)),
        integral_add iX2 iη2,
        integral_sub (by exact (iηQ.const_mul 2)) (by exact (iXQ.const_mul 2)),
        integral_const_mul, integral_const_mul, integral_const_mul]
    have hzero : 0 = b ^ 2 - 2 * φ * C + 2 * (∫ ω, η_curr ω * Q ω ∂μ)
        - 2 * (∫ ω, X_prev ω * Q ω ∂μ) + ∫ ω, Q ω ^ 2 ∂μ := by
      linarith [hEX2_eq, hsq_int, hEη2, hEXprev_η]
    have hrhs_int : ∫ ω, (2 * X_prev ω - 2 * η_curr ω / (1 + φ) - Q ω) * Q ω ∂μ
        = 2 * (∫ ω, X_prev ω * Q ω ∂μ) - (2 / (1 + φ)) * (∫ ω, η_curr ω * Q ω ∂μ)
          - ∫ ω, Q ω ^ 2 ∂μ := by
      have hrhs_pt : (fun ω => (2 * X_prev ω - 2 * η_curr ω / (1 + φ) - Q ω) * Q ω)
          = fun ω => (2 * (X_prev ω * Q ω) - (2 / (1 + φ)) * (η_curr ω * Q ω)) - Q ω ^ 2 := by
        ext ω; ring
      rw [hrhs_pt,
        integral_sub (by exact (iXQ.const_mul 2).sub (iηQ.const_mul (2 / (1 + φ)))) iQ2,
        integral_sub (by exact iXQ.const_mul 2) (by exact iηQ.const_mul (2 / (1 + φ))),
        integral_const_mul, integral_const_mul]
    rw [hrhs_int]
    apply mul_left_cancel₀ h1p_ne
    calc (1 + φ) * (b ^ 2 * (1 - φ) / (1 + φ))
        = b ^ 2 * (1 - φ) := by field_simp
      _ = (1 + φ) * (2 * (∫ ω, X_prev ω * Q ω ∂μ)
            - (2 / (1 + φ)) * (∫ ω, η_curr ω * Q ω ∂μ) - ∫ ω, Q ω ^ 2 ∂μ) := by
          have hcancel : (1 + φ) * ((2 / (1 + φ)) * (∫ ω, η_curr ω * Q ω ∂μ))
              = 2 * (∫ ω, η_curr ω * Q ω ∂μ) := by field_simp
          rw [mul_sub, mul_sub, mul_sub, hcancel]
          linear_combination -(1 + φ) * hzero + 2 * φ * h_eq5
  exact ⟨hX_bd, h_eq4, h_eq5, h_eq6⟩

/-- **Q6 (martingale-difference cross-check via `W_t` and `ξ_t`, `gap2_proof.md` Dated Algebra
Audit lines 215–244).**
Assume `H1` (`h_stat`), `H2` (`h_Z_pm`), `H4` (`h_markov`), and `H5` (`h_autocorr`), with `b ≥ 0`
and `-1 < φ < 1`. Define `W_prev = X_prev - (φ/(1 + φ))·b·Z_prev`,
`W_curr = X_curr - (φ/(1 + φ))·b·Z_curr`, `ξ = (b·Z_curr - φ·b·Z_prev)/(1 + φ)`, and
`Q = Q_L(X_prev, b·Z_curr)`. Then:
1. Pointwise: `W_curr = -W_prev + ξ + Q` and
   `2·W_prev - 2·ξ - Q = 2·X_prev - 2·(b·Z_curr)/(1 + φ) - Q`.
2. Innovation orthogonality: `E[W_prev·ξ] = 0`.
3. Innovation variance: `E[ξ²] = b²·(1 - φ)/(1 + φ)`.
4. Stationary square difference of `W`:
   `b²·(1 - φ)/(1 + φ) = E[(2·W_prev - 2·ξ - Q)·Q]`. -/
theorem Q6 {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (L b φ : ℝ) (hb : 0 ≤ b) (hφ_lo : -1 < φ) (_hφ_hi : φ < 1)
    (X_prev Z_prev Z_curr : Ω → ℝ)
    (h_stat : IdentDistrib
      (fun ω => (Fstep L (b * Z_curr ω) (X_prev ω), Z_curr ω))
      (fun ω => (X_prev ω, Z_prev ω)) μ μ)
    (h_Z_pm : ∀ᵐ ω ∂μ, Z_curr ω = 1 ∨ Z_curr ω = -1)
    (h_markov : ∫ ω, X_prev ω * Z_curr ω ∂μ = φ * ∫ ω, X_prev ω * Z_prev ω ∂μ)
    (h_autocorr : ∫ ω, Z_prev ω * Z_curr ω ∂μ = φ) :
    let X_curr := fun ω => Fstep L (b * Z_curr ω) (X_prev ω)
    let W_prev := fun ω => X_prev ω - φ / (1 + φ) * (b * Z_prev ω)
    let W_curr := fun ω => X_curr ω - φ / (1 + φ) * (b * Z_curr ω)
    let ξ := fun ω => (b * Z_curr ω - φ * (b * Z_prev ω)) / (1 + φ)
    let Q := fun ω => Qstep L (X_prev ω) (b * Z_curr ω)
    (∀ ω, W_curr ω = -W_prev ω + ξ ω + Q ω
        ∧ 2 * W_prev ω - 2 * ξ ω - Q ω = 2 * X_prev ω - 2 * (b * Z_curr ω) / (1 + φ) - Q ω)
      ∧ (∫ ω, W_prev ω * ξ ω ∂μ = 0)
      ∧ (∫ ω, ξ ω ^ 2 ∂μ = b ^ 2 * (1 - φ) / (1 + φ))
      ∧ (b ^ 2 * (1 - φ) / (1 + φ) = ∫ ω, (2 * W_prev ω - 2 * ξ ω - Q ω) * Q ω ∂μ) := by
  clear _hφ_hi
  dsimp only
  have h1p : 0 < 1 + φ := by linarith
  have h1p_ne : 1 + φ ≠ 0 := h1p.ne'
  obtain ⟨_, _, _, hZp_pm, _⟩ := stationary_support_ae μ L b hb X_prev Z_prev Z_curr h_stat h_Z_pm
  set X_curr : Ω → ℝ := fun ω => Fstep L (b * Z_curr ω) (X_prev ω)
  set W_prev : Ω → ℝ := fun ω => X_prev ω - φ / (1 + φ) * (b * Z_prev ω)
  set W_curr : Ω → ℝ := fun ω => X_curr ω - φ / (1 + φ) * (b * Z_curr ω)
  set ξ : Ω → ℝ := fun ω => (b * Z_curr ω - φ * (b * Z_prev ω)) / (1 + φ)
  set Q : Ω → ℝ := fun ω => Qstep L (X_prev ω) (b * Z_curr ω)
  -- Part 1: exact pointwise identities
  have hpt : ∀ ω, W_curr ω = -W_prev ω + ξ ω + Q ω
      ∧ 2 * W_prev ω - 2 * ξ ω - Q ω = 2 * X_prev ω - 2 * (b * Z_curr ω) / (1 + φ) - Q ω := by
    intro ω
    dsimp [W_curr, W_prev, X_curr, ξ, Q]
    rw [Fstep_eq]
    constructor <;> field_simp <;> ring
  -- Integrability helpers
  have hQ_cont : Continuous (fun p : ℝ × ℝ × ℝ => Qstep L p.1 (b * p.2.2)) :=
    (continuous_Qstep L).comp
      (continuous_fst.prodMk (continuous_const.mul (continuous_snd.comp continuous_snd)))
  have hWp_cont : Continuous (fun p : ℝ × ℝ × ℝ => p.1 - φ / (1 + φ) * (b * p.2.1)) :=
    continuous_fst.sub
      (continuous_const.mul (continuous_const.mul (continuous_fst.comp continuous_snd)))
  have hξ_cont : Continuous (fun p : ℝ × ℝ × ℝ => (b * p.2.2 - φ * (b * p.2.1)) / (1 + φ)) :=
    ((continuous_const.mul (continuous_snd.comp continuous_snd)).sub
      (continuous_const.mul
        (continuous_const.mul (continuous_fst.comp continuous_snd)))).div_const _
  have int_triple : ∀ G : ℝ × ℝ × ℝ → ℝ, Continuous G →
      Integrable (fun ω => G (X_prev ω, Z_prev ω, Z_curr ω)) μ :=
    integrable_of_continuous_triple μ L b hb X_prev Z_prev Z_curr h_stat h_Z_pm
  have iXZc : Integrable (fun ω => X_prev ω * Z_curr ω) μ :=
    int_triple (fun p => p.1 * p.2.2) (continuous_fst.mul (continuous_snd.comp continuous_snd))
  have iXZp : Integrable (fun ω => X_prev ω * Z_prev ω) μ :=
    int_triple (fun p => p.1 * p.2.1) (continuous_fst.mul (continuous_fst.comp continuous_snd))
  have iZZ : Integrable (fun ω => Z_prev ω * Z_curr ω) μ :=
    int_triple (fun p => p.2.1 * p.2.2)
      ((continuous_fst.comp continuous_snd).mul (continuous_snd.comp continuous_snd))
  have iZp2 : Integrable (fun ω => Z_prev ω ^ 2) μ :=
    int_triple (fun p => p.2.1 ^ 2) ((continuous_fst.comp continuous_snd).pow 2)
  have iZc2 : Integrable (fun ω => Z_curr ω ^ 2) μ :=
    int_triple (fun p => p.2.2 ^ 2) ((continuous_snd.comp continuous_snd).pow 2)
  have iW2 : Integrable (fun ω => W_prev ω ^ 2) μ := int_triple _ (hWp_cont.pow 2)
  have iWξ : Integrable (fun ω => W_prev ω * ξ ω) μ := int_triple _ (hWp_cont.mul hξ_cont)
  have iξ2 : Integrable (fun ω => ξ ω ^ 2) μ := int_triple _ (hξ_cont.pow 2)
  have iRHS : Integrable (fun ω => (2 * W_prev ω - 2 * ξ ω - Q ω) * Q ω) μ :=
    int_triple _
      ((((continuous_const.mul hWp_cont).sub
        (continuous_const.mul hξ_cont)).sub hQ_cont).mul hQ_cont)
  -- Unit second moments of `Z_prev` and `Z_curr`
  have hEZp2 : ∫ ω, Z_prev ω ^ 2 ∂μ = 1 := by
    have hae : (fun ω => Z_prev ω ^ 2) =ᵐ[μ] fun _ => (1 : ℝ) := by
      filter_upwards [hZp_pm] with ω hpm; rcases hpm with h | h <;> rw [h] <;> norm_num
    rw [integral_congr_ae hae]; simp
  have hEZc2 : ∫ ω, Z_curr ω ^ 2 ∂μ = 1 := by
    have hae : (fun ω => Z_curr ω ^ 2) =ᵐ[μ] fun _ => (1 : ℝ) := by
      filter_upwards [h_Z_pm] with ω hpm; rcases hpm with h | h <;> rw [h] <;> norm_num
    rw [integral_congr_ae hae]; simp
  -- Part 2: `E[W_prev·ξ] = 0`
  have hEWξ : ∫ ω, W_prev ω * ξ ω ∂μ = 0 := by
    have h_exp : (fun ω => W_prev ω * ξ ω) = fun ω =>
        (b / (1 + φ)) * (X_prev ω * Z_curr ω - φ * (X_prev ω * Z_prev ω))
          - (φ * b ^ 2 / (1 + φ) ^ 2) * (Z_prev ω * Z_curr ω - φ * Z_prev ω ^ 2) := by
      ext ω; dsimp [W_prev, ξ]; field_simp
    rw [h_exp,
      integral_sub
        (by exact (iXZc.sub (iXZp.const_mul φ)).const_mul (b / (1 + φ)))
        (by exact (iZZ.sub (iZp2.const_mul φ)).const_mul (φ * b ^ 2 / (1 + φ) ^ 2)),
      integral_const_mul, integral_const_mul,
      integral_sub iXZc (by exact iXZp.const_mul φ),
      integral_sub iZZ (by exact iZp2.const_mul φ),
      integral_const_mul, integral_const_mul,
      h_markov, h_autocorr, hEZp2]
    ring
  -- Part 3: `E[ξ²] = b²·(1 - φ)/(1 + φ)`
  have hEξ2 : ∫ ω, ξ ω ^ 2 ∂μ = b ^ 2 * (1 - φ) / (1 + φ) := by
    have h_exp : (fun ω => ξ ω ^ 2) = fun ω =>
        (b ^ 2 / (1 + φ) ^ 2)
          * ((Z_curr ω ^ 2 - 2 * φ * (Z_prev ω * Z_curr ω)) + φ ^ 2 * Z_prev ω ^ 2) := by
      ext ω; dsimp [ξ]; field_simp; ring
    rw [h_exp, integral_const_mul,
      integral_add (by exact iZc2.sub (iZZ.const_mul (2 * φ))) (by exact iZp2.const_mul (φ ^ 2)),
      integral_sub iZc2 (by exact iZZ.const_mul (2 * φ)),
      integral_const_mul, integral_const_mul, hEZc2, h_autocorr, hEZp2]
    field_simp
    ring
  -- Part 4: stationarity of `W²` gives `b²·(1 - φ)/(1 + φ) = E[(2·W_prev - 2·ξ - Q)·Q]`
  have hW_id : IdentDistrib (fun ω => W_curr ω ^ 2) (fun ω => W_prev ω ^ 2) μ μ :=
    h_stat.comp
      ((measurable_fst.sub
        (measurable_const.mul (measurable_const.mul measurable_snd))).pow_const 2)
  have hEW2_eq : ∫ ω, W_curr ω ^ 2 ∂μ = ∫ ω, W_prev ω ^ 2 ∂μ := hW_id.integral_eq
  have hW_sq_int : ∫ ω, W_curr ω ^ 2 ∂μ
      = (((∫ ω, W_prev ω ^ 2 ∂μ) + (∫ ω, ξ ω ^ 2 ∂μ)) - 2 * (∫ ω, W_prev ω * ξ ω ∂μ))
        - ∫ ω, (2 * W_prev ω - 2 * ξ ω - Q ω) * Q ω ∂μ := by
    have hsq_pt : (fun ω => W_curr ω ^ 2) = fun ω =>
        ((W_prev ω ^ 2 + ξ ω ^ 2) - 2 * (W_prev ω * ξ ω))
          - (2 * W_prev ω - 2 * ξ ω - Q ω) * Q ω := by
      ext ω; rw [(hpt ω).1]; ring
    rw [hsq_pt,
      integral_sub (by exact (iW2.add iξ2).sub (iWξ.const_mul 2)) iRHS,
      integral_sub (by exact iW2.add iξ2) (by exact iWξ.const_mul 2),
      integral_add iW2 iξ2, integral_const_mul]
  have h_eq6_W : b ^ 2 * (1 - φ) / (1 + φ) = ∫ ω, (2 * W_prev ω - 2 * ξ ω - Q ω) * Q ω ∂μ := by
    linarith [hEW2_eq, hW_sq_int, hEWξ, hEξ2]
  exact ⟨hpt, hEWξ, hEξ2, h_eq6_W⟩

/-! ## Q7, Q8, Q9: Direct soft-boundary inequalities eqs. `(7)`–`(10)` with explicit constants -/

/-- **Q7 (pointwise soft-boundary cap and integrand bounds, `gap2_proof.md` §3 lines 124–158).**
Let `0 ≤ b`, `-1 < φ`, `x - L ≤ b + 1`, and `|η| ≤ b`, and put `y = x - L` and
`Q = Q_L(x, η) = log(1 + exp(y - η))`. Then:
1. `0 < Q ≤ Q_*(b)`.
2. `(2·y - 2·η/(1 + φ) - Q)·Q ≤ C_+(b, φ)·Q`.
3. For every `M ≥ 1`,
   `-(2·M + C_-(b, φ))·Q - 2·M·exp(b - M) ≤ (2·y - 2·η/(1 + φ) - Q)·Q`. -/
theorem Q7 (L b φ x η : ℝ) (_hb : 0 ≤ b) (hφ_lo : -1 < φ)
    (hy_hi : x - L ≤ b + 1) (hη : |η| ≤ b) :
    let y := x - L
    let Q := Qstep L x η
    (0 < Q ∧ Q ≤ Qstar b)
      ∧ (2 * y - 2 * η / (1 + φ) - Q) * Q ≤ Cplus b φ * Q
      ∧ ∀ M : ℝ, 1 ≤ M →
          -(2 * M + Cminus b φ) * Q - 2 * M * Real.exp (b - M)
            ≤ (2 * y - 2 * η / (1 + φ) - Q) * Q := by
  clear _hb
  dsimp only
  have h1p : 0 < 1 + φ := by linarith
  obtain ⟨hη_lo, hη_hi⟩ := abs_le.mp hη
  set y := x - L
  set Q := Qstep L x η
  have hQ_pos : 0 < Q := by
    unfold Q Qstep
    exact Real.log_pos (by linarith [Real.exp_pos (x - η - L)])
  have hQ_le : Q ≤ Qstar b := by
    unfold Q Qstep Qstar
    apply Real.log_le_log (by positivity)
    have hexp : Real.exp (x - η - L) ≤ Real.exp (2 * b + 1) :=
      Real.exp_le_exp.mpr (by unfold y at hy_hi; linarith)
    linarith
  -- Upper bound on `(2y - 2η/(1+φ) - Q)·Q`
  have h_upper : (2 * y - 2 * η / (1 + φ) - Q) * Q ≤ Cplus b φ * Q := by
    have hdiv : -(2 * b / (1 + φ)) ≤ 2 * η / (1 + φ) := by
      rw [← neg_div]
      exact div_le_div_of_nonneg_right (by linarith) h1p.le
    have hcoef : 2 * y - 2 * η / (1 + φ) - Q ≤ Cplus b φ := by
      unfold Cplus
      linarith
    exact mul_le_mul_of_nonneg_right hcoef hQ_pos.le
  -- Lower bound with split at `y = -M`
  refine ⟨⟨hQ_pos, hQ_le⟩, h_upper, ?_⟩
  intro M hM
  have hdiv_hi : 2 * η / (1 + φ) ≤ 2 * b / (1 + φ) :=
    div_le_div_of_nonneg_right (by linarith) h1p.le
  have htail_coef : -Cminus b φ ≤ -(2 * η / (1 + φ)) - Q := by
    unfold Cminus
    linarith
  have htail_mul : -Cminus b φ * Q ≤ (-(2 * η / (1 + φ)) - Q) * Q :=
    mul_le_mul_of_nonneg_right htail_coef hQ_pos.le
  have hsplit_Q : (2 * y - 2 * η / (1 + φ) - Q) * Q
      = 2 * y * Q + (-(2 * η / (1 + φ)) - Q) * Q := by ring
  have hsplit_lhs : -(2 * M + Cminus b φ) * Q
      = -(2 * M * Q) + (-Cminus b φ * Q) := by ring
  rcases le_or_gt (-M) y with hyM | hyM
  · have h2y : -(2 * M * Q) ≤ 2 * y * Q := by
      have h := mul_le_mul_of_nonneg_right (show -2 * M ≤ 2 * y by linarith) hQ_pos.le
      linarith
    have hexp_nn : 0 ≤ 2 * M * Real.exp (b - M) := by positivity
    linarith
  · set d := -y
    have hdM : M < d := by unfold d; linarith
    have hd_pos : 0 < d := by linarith
    have hQ_exp : Q ≤ Real.exp (b - d) := by
      set u := Real.exp (x - η - L)
      have hu_pos : 0 < u := Real.exp_pos _
      have h1u : 1 + u ≤ Real.exp u := by linarith [Real.add_one_le_exp u]
      have hQ_u : Q ≤ u := by
        unfold Q Qstep
        have h := Real.log_le_log (by positivity) h1u
        rwa [Real.log_exp] at h
      have hu_bd : u ≤ Real.exp (b - d) := by
        unfold u
        apply Real.exp_le_exp.mpr
        have e : x - η - L = -d - η := by unfold d y; ring
        linarith
      exact hQ_u.trans hu_bd
    have hd_le_Mexp : d * Real.exp (b - d) ≤ M * Real.exp (b - M) := by
      have hd_lin : d ≤ M * ((d - M) + 1) := by nlinarith
      have hexp_dm : (d - M) + 1 ≤ Real.exp (d - M) := Real.add_one_le_exp (d - M)
      have hd_M : d ≤ M * Real.exp (d - M) :=
        hd_lin.trans (mul_le_mul_of_nonneg_left hexp_dm (by linarith))
      calc d * Real.exp (b - d)
          ≤ (M * Real.exp (d - M)) * Real.exp (b - d) :=
            mul_le_mul_of_nonneg_right hd_M (Real.exp_pos _).le
        _ = M * (Real.exp (d - M) * Real.exp (b - d)) := by ring
        _ = M * Real.exp (b - M) := by
            rw [← Real.exp_add]
            congr 2
            ring
    have hdQ : d * Q ≤ M * Real.exp (b - M) :=
      (mul_le_mul_of_nonneg_left hQ_exp hd_pos.le).trans hd_le_Mexp
    have h2MQ_nn : 0 ≤ 2 * M * Q := mul_nonneg (by linarith) hQ_pos.le
    have h2yd : 2 * y * Q = -2 * (d * Q) := by unfold d; ring
    linarith

/-- **Q8 (stationary soft-boundary decomposition eq. `(7)` and explicit lower/upper bounds
eqs. `(8)` and `(9)` of `gap2_proof.md`).**
Assume `H1` (`h_stat`), `H2`/`H3` (`h_Z_pm`, `h_Z_zero`), and `H4` (`h_markov`), with `b ≥ 0`,
`-1 < φ < 1`, and `L ≥ 1`. Put `q_L = E[Q_L(X_prev, b·Z_curr)]` and
`H_L = E[(2·(X_prev - L) - 2·(b·Z_curr)/(1 + φ) - Q_L(X_prev, b·Z_curr))·Q_L(X_prev, b·Z_curr)]`.
Then, for every stationary law (requiring no smoothness, density, or nonlattice hypothesis):
1. `0 ≤ q_L` and `b²·(1 - φ)/(1 + φ) = 2·L·q_L + H_L` (`eq. (7)`).
2. `H_L ≤ C_+(b, φ)·q_L` and `b²·(1 - φ)/(1 + φ) / (2·L + C_+(b, φ)) ≤ q_L` (`eq. (8)`).
3. For every `M ≥ 1`, `-(2·M + C_-(b, φ))·q_L - 2·M·exp(b - M) ≤ H_L`, and whenever
   `0 < 2·L - 2·M - C_-(b, φ)`,
   `q_L ≤ (b²·(1 - φ)/(1 + φ) + 2·M·exp(b - M)) / (2·L - 2·M - C_-(b, φ))` (`eq. (9)`). -/
theorem Q8 {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (L b φ : ℝ) (hb : 0 ≤ b) (hφ_lo : -1 < φ) (hφ_hi : φ < 1) (hL : 1 ≤ L)
    (X_prev Z_prev Z_curr : Ω → ℝ)
    (h_stat : IdentDistrib
      (fun ω => (Fstep L (b * Z_curr ω) (X_prev ω), Z_curr ω))
      (fun ω => (X_prev ω, Z_prev ω)) μ μ)
    (h_Z_pm : ∀ᵐ ω ∂μ, Z_curr ω = 1 ∨ Z_curr ω = -1)
    (h_Z_zero : ∫ ω, Z_curr ω ∂μ = 0)
    (h_markov : ∫ ω, X_prev ω * Z_curr ω ∂μ = φ * ∫ ω, X_prev ω * Z_prev ω ∂μ) :
    let Q := fun ω => Qstep L (X_prev ω) (b * Z_curr ω)
    let q_L := ∫ ω, Q ω ∂μ
    let H_L := ∫ ω, (2 * (X_prev ω - L) - 2 * (b * Z_curr ω) / (1 + φ) - Q ω) * Q ω ∂μ
    let V := b ^ 2 * (1 - φ) / (1 + φ)
    (0 ≤ q_L ∧ V = 2 * L * q_L + H_L)
      ∧ (H_L ≤ Cplus b φ * q_L ∧ V / (2 * L + Cplus b φ) ≤ q_L)
      ∧ ∀ M : ℝ, 1 ≤ M →
          -(2 * M + Cminus b φ) * q_L - 2 * M * Real.exp (b - M) ≤ H_L
            ∧ (0 < 2 * L - 2 * M - Cminus b φ →
                q_L ≤ (V + 2 * M * Real.exp (b - M)) / (2 * L - 2 * M - Cminus b φ)) := by
  dsimp only
  obtain ⟨hX_bd, _, _, h_eq6⟩ :=
    Q5 μ L b φ hb hφ_lo hφ_hi X_prev Z_prev Z_curr h_stat h_Z_pm h_Z_zero h_markov
  set Q : Ω → ℝ := fun ω => Qstep L (X_prev ω) (b * Z_curr ω)
  set g_H : Ω → ℝ := fun ω => (2 * (X_prev ω - L) - 2 * (b * Z_curr ω) / (1 + φ) - Q ω) * Q ω
  set q_L := ∫ ω, Q ω ∂μ
  set H_L := ∫ ω, g_H ω ∂μ
  set V := b ^ 2 * (1 - φ) / (1 + φ)
  have h1p : 0 < 1 + φ := by linarith
  have hcap : Real.log (Real.exp (-L) + Real.exp (L + b)) < L + b + 1 :=
    log_exp_neg_add_exp_lt L b hb (by linarith)
  -- Integrability
  have hQ_cont : Continuous (fun p : ℝ × ℝ × ℝ => Qstep L p.1 (b * p.2.2)) :=
    (continuous_Qstep L).comp
      (continuous_fst.prodMk (continuous_const.mul (continuous_snd.comp continuous_snd)))
  have int_triple : ∀ G : ℝ × ℝ × ℝ → ℝ, Continuous G →
      Integrable (fun ω => G (X_prev ω, Z_prev ω, Z_curr ω)) μ :=
    integrable_of_continuous_triple μ L b hb X_prev Z_prev Z_curr h_stat h_Z_pm
  have iQ : Integrable Q μ := int_triple _ hQ_cont
  have igH : Integrable g_H μ := int_triple _
    (((((continuous_const.mul (continuous_fst.sub continuous_const)).sub
      ((continuous_const.mul
        (continuous_const.mul (continuous_snd.comp continuous_snd))).div_const _)).sub
      hQ_cont)).mul hQ_cont)
  -- Almost-everywhere pointwise bounds from `Q7`
  have hae_Q7 : ∀ᵐ ω ∂μ,
      0 ≤ Q ω
        ∧ g_H ω ≤ Cplus b φ * Q ω
        ∧ ∀ M : ℝ, 1 ≤ M → -(2 * M + Cminus b φ) * Q ω - 2 * M * Real.exp (b - M) ≤ g_H ω := by
    filter_upwards [hX_bd, h_Z_pm] with ω hx hpm
    have hy_hi : X_prev ω - L ≤ b + 1 := by linarith [hx.2, hcap]
    have hη_abs : |b * Z_curr ω| ≤ b := by
      rcases hpm with h | h <;> rw [h] <;> simp [abs_of_nonneg hb]
    obtain ⟨⟨hQp, _⟩, hup, hlo⟩ := Q7 L b φ (X_prev ω) (b * Z_curr ω) hb hφ_lo hy_hi hη_abs
    exact ⟨hQp.le, hup, hlo⟩
  -- Part 1: `0 ≤ q_L` and `V = 2·L·q_L + H_L`
  have hq_nn : 0 ≤ q_L := by
    have h0 : (∫ _ : Ω, (0 : ℝ) ∂μ) ≤ ∫ ω, Q ω ∂μ :=
      integral_mono_ae (integrable_const 0) iQ (hae_Q7.mono fun _ h => h.1)
    simpa using h0
  have h_eq7 : V = 2 * L * q_L + H_L := by
    rw [show V = ∫ ω, (2 * X_prev ω - 2 * (b * Z_curr ω) / (1 + φ) - Q ω) * Q ω ∂μ from h_eq6]
    have hpt : (fun ω => (2 * X_prev ω - 2 * (b * Z_curr ω) / (1 + φ) - Q ω) * Q ω)
        = fun ω => (2 * L) * Q ω + g_H ω := by
      ext ω; dsimp [g_H]; ring
    rw [hpt, integral_add (by exact iQ.const_mul (2 * L)) igH, integral_const_mul]
  -- Part 2: `H_L ≤ C_+·q_L` and `eq. (8)`
  have hH_upper : H_L ≤ Cplus b φ * q_L := by
    have h := integral_mono_ae igH (iQ.const_mul (Cplus b φ)) (hae_Q7.mono fun _ h => h.2.1)
    rwa [integral_const_mul] at h
  have hCplus_pos : 0 < Cplus b φ := by
    unfold Cplus
    have : 0 ≤ 2 * b / (1 + φ) := div_nonneg (by linarith) h1p.le
    linarith
  have hden_plus : 0 < 2 * L + Cplus b φ := by linarith
  have h_eq8 : V / (2 * L + Cplus b φ) ≤ q_L := by
    rw [div_le_iff₀ hden_plus]
    nlinarith [h_eq7, hH_upper]
  -- Part 3: lower bound on `H_L` and `eq. (9)`
  refine ⟨⟨hq_nn, h_eq7⟩, ⟨hH_upper, h_eq8⟩, ?_⟩
  intro M hM
  have hH_lower : -(2 * M + Cminus b φ) * q_L - 2 * M * Real.exp (b - M) ≤ H_L := by
    have iL : Integrable
        (fun ω => -(2 * M + Cminus b φ) * Q ω - 2 * M * Real.exp (b - M)) μ :=
      (iQ.const_mul (-(2 * M + Cminus b φ))).sub (integrable_const (2 * M * Real.exp (b - M)))
    have h := integral_mono_ae iL igH (hae_Q7.mono fun _ h => h.2.2 M hM)
    rw [integral_sub (by exact iQ.const_mul _) (integrable_const _), integral_const_mul] at h
    simpa using h
  refine ⟨hH_lower, fun hden_minus => ?_⟩
  rw [le_div_iff₀ hden_minus]
  nlinarith [h_eq7, hH_lower]

/-- Elementary bound `exp 1 ≤ 4` from `1 + (-1/2) ≤ exp(-1/2)`. -/
theorem exp_one_le_four : Real.exp 1 ≤ 4 := by
  have h1 : (1 / 2 : ℝ) ≤ Real.exp (-1 / 2 : ℝ) := by
    linarith [Real.add_one_le_exp (-1 / 2 : ℝ)]
  have hprod : Real.exp (1 / 2 : ℝ) * Real.exp (-1 / 2 : ℝ) = 1 := by
    rw [← Real.exp_add]
    norm_num
  have hhalf : Real.exp (1 / 2 : ℝ) ≤ 2 := by
    nlinarith [Real.exp_pos (1 / 2 : ℝ)]
  have hsq : Real.exp 1 = Real.exp (1 / 2 : ℝ) * Real.exp (1 / 2 : ℝ) := by
    rw [← Real.exp_add]
    norm_num
  nlinarith [Real.exp_pos (1 / 2 : ℝ)]

/-- For `x ≥ 4`, `1 ≤ log x` (since `exp 1 ≤ 4 ≤ x`). -/
theorem one_le_log_of_four_le {x : ℝ} (hx : 4 ≤ x) : 1 ≤ Real.log x := by
  have h := Real.log_le_log (Real.exp_pos 1) (exp_one_le_four.trans hx)
  rwa [Real.log_exp] at h

/-- **Q9 (explicit non-asymptotic soft-boundary error bound eq. `(10)` of `gap2_proof.md`).**
Assume `H1` (`h_stat`), `H2`/`H3` (`h_Z_pm`, `h_Z_zero`), and `H4` (`h_markov`), with `b ≥ 0`,
`-1 < φ < 1`, and `L ≥ L_min(b, φ) = (8 + C_-(b, φ))²`. Put `q_L = E[Q_L(X_prev, b·Z_curr)]` and
`V = b²·(1 - φ)/(1 + φ)`. Then `0 ≤ K_q(b, φ)` and
`|q_L - V / (2·L)| ≤ K_q(b, φ)·(log L / L²)`. -/
theorem Q9 {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (L b φ : ℝ) (hb : 0 ≤ b) (hφ_lo : -1 < φ) (hφ_hi : φ < 1) (hL : Lmin b φ ≤ L)
    (X_prev Z_prev Z_curr : Ω → ℝ)
    (h_stat : IdentDistrib
      (fun ω => (Fstep L (b * Z_curr ω) (X_prev ω), Z_curr ω))
      (fun ω => (X_prev ω, Z_prev ω)) μ μ)
    (h_Z_pm : ∀ᵐ ω ∂μ, Z_curr ω = 1 ∨ Z_curr ω = -1)
    (h_Z_zero : ∫ ω, Z_curr ω ∂μ = 0)
    (h_markov : ∫ ω, X_prev ω * Z_curr ω ∂μ = φ * ∫ ω, X_prev ω * Z_prev ω ∂μ) :
    let q_L := ∫ ω, Qstep L (X_prev ω) (b * Z_curr ω) ∂μ
    let V := b ^ 2 * (1 - φ) / (1 + φ)
    0 ≤ Kq b φ ∧ |q_L - V / (2 * L)| ≤ Kq b φ * (Real.log L / L ^ 2) := by
  dsimp only
  have h1p : 0 < 1 + φ := by linarith
  have hQstar_pos : 0 < Qstar b := by
    unfold Qstar
    exact Real.log_pos (by linarith [Real.exp_pos (2 * b + 1)])
  have hCminus_pos : 0 < Cminus b φ := by
    unfold Cminus
    have : 0 ≤ 2 * b / (1 + φ) := div_nonneg (by linarith) h1p.le
    linarith
  have hCplus_pos : 0 < Cplus b φ := by
    unfold Cplus
    have : 0 ≤ 2 * b / (1 + φ) := div_nonneg (by linarith) h1p.le
    linarith
  have hLmin_gt : 64 < Lmin b φ := by
    unfold Lmin
    nlinarith
  have hL64 : 64 < L := lt_of_lt_of_le hLmin_gt hL
  have hL1 : 1 ≤ L := by linarith
  have hlogL1 : 1 ≤ Real.log L := one_le_log_of_four_le (by linarith)
  set V := b ^ 2 * (1 - φ) / (1 + φ)
  have hV_nn : 0 ≤ V := div_nonneg (mul_nonneg (sq_nonneg b) (by linarith)) h1p.le
  have hKq_nn : 0 ≤ Kq b φ := by
    unfold Kq
    exact le_trans (by positivity) (le_max_left _ _)
  obtain ⟨_, ⟨_, h_eq8⟩, h_eq9_all⟩ :=
    Q8 μ L b φ hb hφ_lo hφ_hi hL1 X_prev Z_prev Z_curr h_stat h_Z_pm h_Z_zero h_markov
  set q_L := ∫ ω, Qstep L (X_prev ω) (b * Z_curr ω) ∂μ
  set M := 2 * Real.log L
  have hM1 : 1 ≤ M := by unfold M; linarith
  -- Show `4·log L + C_- ≤ L`, hence `L ≤ 2·L - 2·M - C_-`
  set D_den := 2 * L - 2 * M - Cminus b φ
  have hD_ge_L : L ≤ D_den := by
    set u := Real.sqrt L
    have hu_pos : 0 < u := Real.sqrt_pos.mpr (by linarith)
    have hu_sq : u ^ 2 = L := Real.sq_sqrt (by linarith)
    have hu_ge : 8 + Cminus b φ ≤ u := by
      have hsq_le : (8 + Cminus b φ) ^ 2 ≤ u ^ 2 := by
        rw [hu_sq]; exact hL
      by_contra hlt
      nlinarith
    have hu_exp : u ≤ Real.exp (u - 1) := by linarith [Real.add_one_le_exp (u - 1)]
    have hlogu : Real.log u ≤ u - 1 := by
      have h := Real.log_le_log hu_pos hu_exp
      rwa [Real.log_exp] at h
    have hlogL_eq : Real.log L = 2 * Real.log u := by
      have hL_uu : L = u * u := by linarith [hu_sq]
      rw [hL_uu, Real.log_mul hu_pos.ne' hu_pos.ne']
      ring
    have h4log : 4 * Real.log L + Cminus b φ ≤ L := by
      rw [hlogL_eq]
      have hCu : Cminus b φ ≤ Cminus b φ * u := by nlinarith
      nlinarith
    unfold D_den M
    linarith
  have hD_pos : 0 < D_den := by linarith
  have h_eq9 := (h_eq9_all M hM1).2 hD_pos
  have hratio_nn : 0 ≤ Real.log L / L ^ 2 := div_nonneg (by linarith) (sq_nonneg L)
  refine ⟨hKq_nn, abs_sub_le_iff.mpr ⟨?_, ?_⟩⟩
  · -- Upper error: `q_L - V/(2L) ≤ K_q·(log L / L²)`
    have hexp_M : Real.exp (b - M) = Real.exp b / L ^ 2 := by
      unfold M
      rw [Real.exp_sub, show (2 : ℝ) * Real.log L = Real.log L + Real.log L by ring,
        Real.exp_add, Real.exp_log (by linarith : 0 < L)]
      ring
    have hterm1 : V / D_den - V / (2 * L)
        ≤ (V * (4 + Cminus b φ) / 2) * (Real.log L / L ^ 2) := by
      have hid : V / D_den - V / (2 * L) = V * (4 * Real.log L + Cminus b φ) / (2 * L * D_den) := by
        have hsub : 2 * L - D_den = 4 * Real.log L + Cminus b φ := by unfold D_den M; ring
        have hL_ne : 2 * L ≠ 0 := by linarith
        have hD_ne : D_den ≠ 0 := hD_pos.ne'
        calc V / D_den - V / (2 * L)
            = V * (2 * L - D_den) / (2 * L * D_den) := by field_simp
          _ = V * (4 * Real.log L + Cminus b φ) / (2 * L * D_den) := by rw [hsub]
      rw [hid]
      have hnum : V * (4 * Real.log L + Cminus b φ)
          ≤ V * ((4 + Cminus b φ) * Real.log L) :=
        mul_le_mul_of_nonneg_left (by nlinarith) hV_nn
      have hden : 2 * L ^ 2 ≤ 2 * L * D_den := by nlinarith
      calc V * (4 * Real.log L + Cminus b φ) / (2 * L * D_den)
          ≤ (V * ((4 + Cminus b φ) * Real.log L)) / (2 * L * D_den) :=
            div_le_div_of_nonneg_right hnum (by positivity)
        _ ≤ (V * ((4 + Cminus b φ) * Real.log L)) / (2 * L ^ 2) :=
            div_le_div_of_nonneg_left (by positivity) (by positivity) hden
        _ = (V * (4 + Cminus b φ) / 2) * (Real.log L / L ^ 2) := by ring
    have hterm2 : (2 * M * Real.exp (b - M)) / D_den ≤ 4 * Real.exp b * (Real.log L / L ^ 2) := by
      rw [hexp_M]
      have hid : (2 * M * (Real.exp b / L ^ 2)) / D_den
          = (4 * Real.exp b * Real.log L) / (L ^ 2 * D_den) := by
        unfold M
        field_simp
        ring
      rw [hid]
      have hden : L ^ 2 ≤ L ^ 2 * D_den := by nlinarith [sq_nonneg L]
      calc (4 * Real.exp b * Real.log L) / (L ^ 2 * D_den)
          ≤ (4 * Real.exp b * Real.log L) / L ^ 2 :=
            div_le_div_of_nonneg_left (by positivity) (by positivity) hden
        _ = 4 * Real.exp b * (Real.log L / L ^ 2) := by ring
    have hsplit : q_L - V / (2 * L)
        ≤ (V / D_den - V / (2 * L)) + (2 * M * Real.exp (b - M)) / D_den := by
      have hdiv_add : (V + 2 * M * Real.exp (b - M)) / D_den
          = V / D_den + (2 * M * Real.exp (b - M)) / D_den := add_div _ _ _
      linarith [h_eq9, hdiv_add]
    have hsum : (V * (4 + Cminus b φ) / 2) * (Real.log L / L ^ 2)
        + 4 * Real.exp b * (Real.log L / L ^ 2)
        ≤ Kq b φ * (Real.log L / L ^ 2) := by
      have hcoef : V * (4 + Cminus b φ) / 2 + 4 * Real.exp b ≤ Kq b φ := by
        have heq : V * (4 + Cminus b φ) / 2 + 4 * Real.exp b
            = (V * (4 + Cminus b φ) + 8 * Real.exp b) / 2 := by ring
        rw [heq]
        exact le_max_right _ _
      calc (V * (4 + Cminus b φ) / 2) * (Real.log L / L ^ 2)
            + 4 * Real.exp b * (Real.log L / L ^ 2)
          = (V * (4 + Cminus b φ) / 2 + 4 * Real.exp b) * (Real.log L / L ^ 2) := by ring
        _ ≤ Kq b φ * (Real.log L / L ^ 2) := mul_le_mul_of_nonneg_right hcoef hratio_nn
    linarith
  · -- Lower error: `V/(2L) - q_L ≤ K_q·(log L / L²)`
    have hden_plus : 0 < 2 * L + Cplus b φ := by linarith
    have hid : V / (2 * L) - V / (2 * L + Cplus b φ)
        = (V * Cplus b φ) / (2 * L * (2 * L + Cplus b φ)) := by
      have hL_ne : 2 * L ≠ 0 := by linarith
      have hDp_ne : 2 * L + Cplus b φ ≠ 0 := hden_plus.ne'
      field_simp
      ring
    have hstep1 : V / (2 * L) - q_L ≤ (V * Cplus b φ) / (2 * L * (2 * L + Cplus b φ)) := by
      linarith [h_eq8, hid]
    have hstep2 : (V * Cplus b φ) / (2 * L * (2 * L + Cplus b φ))
        ≤ (V * Cplus b φ / 4) * (Real.log L / L ^ 2) := by
      have hden : 4 * L ^ 2 ≤ 2 * L * (2 * L + Cplus b φ) := by nlinarith
      calc (V * Cplus b φ) / (2 * L * (2 * L + Cplus b φ))
          ≤ (V * Cplus b φ) / (4 * L ^ 2) :=
            div_le_div_of_nonneg_left (by positivity) (by positivity) hden
        _ = (V * Cplus b φ / 4) * (1 / L ^ 2) := by ring
        _ ≤ (V * Cplus b φ / 4) * (Real.log L / L ^ 2) :=
            mul_le_mul_of_nonneg_left
              (div_le_div_of_nonneg_right hlogL1 (sq_nonneg L)) (by positivity)
    have hstep3 : (V * Cplus b φ / 4) * (Real.log L / L ^ 2)
        ≤ Kq b φ * (Real.log L / L ^ 2) :=
      mul_le_mul_of_nonneg_right (le_max_left _ _) hratio_nn
    linarith

/-! ## Q10, Q11, Q12: Threshold bracket, explicit small-staying remainder, and `s ↓ 0` asymptotic -/

/-- The constant-host threshold bracket follows from the stationary normalized ratio law.
The logarithmic potential `log(U + 1-s)` has zero stationary drift, while zero growth gives
`E log U = 0`; consequently `E log(U+c h) = E log(U+1-s)`. Strict ordering of the
integrands rules out thresholds outside the two constant-host thresholds. -/
theorem stationary_threshold_bracket {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (δ s c : ℝ)
    (hδ0 : 0 < δ) (hδ1 : δ < 1) (hs0 : 0 < s) (hs1 : s < 1) (hc : 0 < c)
    (X Zprev Z : Ω → ℝ)
    (hstat : IdentDistrib
      (fun ω =>
        (Fstep (Real.log (1 / s) + (Real.log c - ell δ) / 2)
          (Real.artanh δ * Z ω) (X ω), Z ω))
      (fun ω => (X ω, Zprev ω)) μ μ)
    (hZ : ∀ᵐ ω ∂μ, Z ω = 1 ∨ Z ω = -1)
    (hzero : (Real.log c - ell δ) + 2 * (∫ ω, X ω ∂μ) = 0) :
    (1 - s) / (1 + δ) ≤ c ∧ c ≤ (1 - s) / (1 - δ) := by
  let b := Real.artanh δ
  let m := Real.log c - ell δ
  let L := Real.log (1 / s) + m / 2
  let U := fun x : ℝ => Real.exp (x + m / 2)
  let P := fun x : ℝ => Real.log (U x + (1 - s))
  let H := fun x z : ℝ => Real.log (U x + Real.exp (m + b * z))
  have hb : 0 ≤ b := (Real.artanh_pos (mem_Ioo.mpr ⟨hδ0, hδ1⟩)).le
  have ha : 0 < 1 - s := by linarith
  have hU : ∀ x, 0 < U x := fun x => Real.exp_pos _
  have hPc : Continuous P := by
    apply Continuous.log
    · exact (Real.continuous_exp.comp (continuous_id.add continuous_const)).add continuous_const
    · intro x
      exact (add_pos (hU x) ha).ne'
  have hHc : Continuous (fun p : ℝ × ℝ × ℝ => H p.1 p.2.2) := by
    apply Continuous.log
    · exact (Real.continuous_exp.comp (continuous_fst.add continuous_const)).add
        (Real.continuous_exp.comp (continuous_const.add
          (continuous_const.mul (continuous_snd.comp continuous_snd))))
    · intro p
      exact (add_pos (hU p.1) (Real.exp_pos _)).ne'
  have hXi : Integrable X μ := by
    simpa using integrable_of_continuous_triple μ L b hb X Zprev Z hstat hZ
      (fun p => p.1) continuous_fst
  have hPi : Integrable (fun ω => P (X ω)) μ :=
    integrable_of_continuous_triple μ L b hb X Zprev Z hstat hZ
      (fun p => P p.1) (hPc.comp continuous_fst)
  have hHi : Integrable (fun ω => H (X ω) (Z ω)) μ :=
    integrable_of_continuous_triple μ L b hb X Zprev Z hstat hZ _ hHc
  have hXid : IdentDistrib (fun ω => Fstep L (b * Z ω) (X ω)) X μ μ :=
    hstat.comp measurable_fst
  have hPid := hXid.comp hPc.measurable
  have hPFi : Integrable (fun ω => P (Fstep L (b * Z ω) (X ω))) μ :=
    hPid.integrable_iff.mpr hPi
  have hhost : ∀ z : ℝ, z = 1 ∨ z = -1 → Real.exp (m + b * z) = c * (1 + δ * z) := by
    intro z hz
    obtain ⟨hhpos, hhlog, _⟩ := (Q1 δ 0 hδ0 hδ1 (by norm_num) (by norm_num)).1 z hz
    rw [← Real.exp_log hc, ← Real.exp_log hhpos, ← Real.exp_add, hhlog]
    congr 1
    dsimp [m, b]
    ring
  have hpoint : ∀ᵐ ω ∂μ,
      H (X ω) (Z ω) = (X ω + m / 2) + P (Fstep L (b * Z ω) (X ω)) := by
    filter_upwards [hZ] with ω hz
    obtain ⟨hu', hnorm, _⟩ := Q2 δ s c (U (X ω)) (Z ω) hδ0 hδ1 hs0 hc (hU _) hz
    have hlogU : Real.log (U (X ω)) = X ω + m / 2 := Real.log_exp _
    rw [hlogU] at hnorm
    have harg : X ω + m / 2 - (Real.log c - ell δ) / 2 = X ω := by dsimp [m]; ring
    rw [harg] at hnorm
    have hratio : U (Fstep L (b * Z ω) (X ω)) = s + c * (1 + δ * Z ω) / U (X ω) := by
      dsimp [U]
      have hnorm' : Fstep L (b * Z ω) (X ω) + m / 2 =
          Real.log (s + c * (1 + δ * Z ω) / U (X ω)) := by
        change Real.log (s + c * (1 + δ * Z ω) / U (X ω)) - m / 2 =
          Fstep L (b * Z ω) (X ω) at hnorm
        linarith
      rw [hnorm', Real.exp_log hu']
    have hprod : U (X ω) * (U (Fstep L (b * Z ω) (X ω)) + (1 - s)) =
        U (X ω) + Real.exp (m + b * Z ω) := by
      rw [hratio, hhost _ hz]
      field_simp [(hU (X ω)).ne']
      ring
    dsimp only [H, P]
    rw [← hprod, Real.log_mul (hU _).ne' (add_pos (hU _) ha).ne', hlogU]
  have heq : (∫ ω, H (X ω) (Z ω) ∂μ) = ∫ ω, P (X ω) ∂μ := by
    have hXmi : Integrable (fun ω => X ω + m / 2) μ := hXi.add (integrable_const (m / 2))
    rw [integral_congr_ae hpoint, integral_add hXmi hPFi,
      integral_add hXi (integrable_const (m / 2)), integral_const]
    have hP : (∫ ω, P (Fstep L (b * Z ω) (X ω)) ∂μ) =
        ∫ ω, P (X ω) ∂μ := hPid.integral_eq
    simp only [probReal_univ, one_smul] at *
    change m + 2 * (∫ ω, X ω ∂μ) = 0 at hzero
    linarith
  have hnotstrict : ¬ ((∀ᵐ ω ∂μ, H (X ω) (Z ω) < P (X ω)) ∨
      (∀ᵐ ω ∂μ, P (X ω) < H (X ω) (Z ω))) := by
    rintro (hlt | hlt)
    · have he := (integral_eq_iff_of_ae_le hHi hPi (hlt.mono fun _ h => h.le)).mp heq
      obtain ⟨ω, hltω, heω⟩ := (hlt.and he).exists
      exact (ne_of_lt hltω) heω
    · have he := (integral_eq_iff_of_ae_le hPi hHi (hlt.mono fun _ h => h.le)).mp heq.symm
      obtain ⟨ω, hltω, heω⟩ := (hlt.and he).exists
      exact (ne_of_lt hltω) heω
  constructor
  · by_contra h
    have hsmall : c * (1 + δ) < 1 - s :=
      (lt_div_iff₀ (by linarith : 0 < 1 + δ)).mp (lt_of_not_ge h)
    apply hnotstrict
    left
    filter_upwards [hZ] with ω hz
    apply Real.log_lt_log (add_pos (hU _) (Real.exp_pos _))
    rw [hhost _ hz]
    have hh : c * (1 + δ * Z ω) ≤ c * (1 + δ) := by
      rcases hz with hz | hz <;> rw [hz] <;> nlinarith
    linarith
  · by_contra h
    have hlarge : 1 - s < c * (1 - δ) :=
      (div_lt_iff₀ (by linarith : 0 < 1 - δ)).mp (lt_of_not_ge h)
    apply hnotstrict
    right
    filter_upwards [hZ] with ω hz
    apply Real.log_lt_log (add_pos (hU _) ha)
    rw [hhost _ hz]
    have hh : c * (1 - δ) ≤ c * (1 + δ * Z ω) := by
      rcases hz with hz | hz <;> rw [hz] <;> nlinarith
    linarith


/-- **Q10 (constant-host threshold bracket and normalization tail bound, `gap2_proof.md` §4
eq. `(11)` and Dated Dependency Reduction lines 246–269).**
Let `0 < δ < 1`, `0 < s < 1`, and `c_star > 0` be a threshold with a stationary
normalized ratio law (`H1`, `H2`) and zero growth (`H6`). The constant-host bracket is
proved by `stationary_threshold_bracket` from those model hypotheses.
Set `b = artanh δ`, `L₀ = log(1/s)`, `m_* = log c_star - ℓ(δ)`, and `D = log(c_star/(1 - s))`. Then:
1. `0 < c_star` and `D - ℓ(δ) = m_* - log(1 - s)`.
2. `log(1 - s) - b ≤ m_* ≤ log(1 - s) + b`.
3. Whenever `4 ≤ L₀`, we have `s ≤ 1/2`, `0 < -log(1 - s) ≤ 2·s`, and
   `|log(1 - s)| ≤ 8·(log L₀ / L₀²)`. -/
theorem Q10 {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (δ s c_star : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1) (hs0 : 0 < s) (hs1 : s < 1)
    (hc : 0 < c_star) (X_prev Z_prev Z_curr : Ω → ℝ)
    (h_stat : IdentDistrib
      (fun ω =>
        (Fstep (Real.log (1 / s) + (Real.log c_star - ell δ) / 2)
          (Real.artanh δ * Z_curr ω) (X_prev ω), Z_curr ω))
      (fun ω => (X_prev ω, Z_prev ω)) μ μ)
    (h_Z_pm : ∀ᵐ ω ∂μ, Z_curr ω = 1 ∨ Z_curr ω = -1)
    (h_zero_growth : (Real.log c_star - ell δ) + 2 * (∫ ω, X_prev ω ∂μ) = 0) :
    let b := Real.artanh δ
    let L₀ := Real.log (1 / s)
    let m_star := Real.log c_star - ell δ
    let D_val := Real.log (c_star / (1 - s))
    (0 < c_star ∧ D_val - ell δ = m_star - Real.log (1 - s))
      ∧ (Real.log (1 - s) - b ≤ m_star ∧ m_star ≤ Real.log (1 - s) + b)
      ∧ (4 ≤ L₀ →
          s ≤ 1 / 2 ∧ 0 < -Real.log (1 - s) ∧ -Real.log (1 - s) ≤ 2 * s
            ∧ |Real.log (1 - s)| ≤ 8 * (Real.log L₀ / L₀ ^ 2)) := by
  dsimp only
  have h_bracket := stationary_threshold_bracket μ δ s c_star hδ0 hδ1 hs0 hs1 hc
    X_prev Z_prev Z_curr h_stat h_Z_pm h_zero_growth
  have h1s : 0 < 1 - s := by linarith
  have h1pδ : 0 < 1 + δ := by linarith
  have h1mδ : 0 < 1 - δ := by linarith
  have hlo_pos : 0 < (1 - s) / (1 + δ) := div_pos h1s h1pδ
  have hc_pos : 0 < c_star := lt_of_lt_of_le hlo_pos h_bracket.1
  obtain ⟨hlog1p, hlog1m⟩ := log_one_add_sub_eq δ hδ0 hδ1
  set b := Real.artanh δ
  set L₀ := Real.log (1 / s)
  set m_star := Real.log c_star - ell δ
  -- Part 1
  have hD_eq : Real.log (c_star / (1 - s)) - ell δ = m_star - Real.log (1 - s) := by
    rw [Real.log_div hc_pos.ne' h1s.ne']
    unfold m_star
    ring
  -- Part 2
  have hm_lo : Real.log (1 - s) - b ≤ m_star := by
    have hlog := Real.log_le_log hlo_pos h_bracket.1
    rw [Real.log_div h1s.ne' h1pδ.ne', hlog1p] at hlog
    unfold m_star
    linarith
  have hm_hi : m_star ≤ Real.log (1 - s) + b := by
    have hlog := Real.log_le_log hc_pos h_bracket.2
    rw [Real.log_div h1s.ne' h1mδ.ne', hlog1m] at hlog
    unfold m_star
    linarith
  refine ⟨⟨hc_pos, hD_eq⟩, ⟨hm_lo, hm_hi⟩, ?_⟩
  -- Part 3: small-`s` bound on `|log(1 - s)|` when `4 ≤ L₀`
  intro hL0_4
  have h1_div_s_pos : 0 < 1 / s := one_div_pos.mpr hs0
  have hexp_L0 : Real.exp L₀ = 1 / s := Real.exp_log h1_div_s_pos
  have hsq_le_exp : L₀ ^ 2 / 4 ≤ 1 / s := by
    have ht : L₀ / 2 ≤ Real.exp (L₀ / 2) := by linarith [Real.add_one_le_exp (L₀ / 2)]
    have ht_nn : 0 ≤ L₀ / 2 := by linarith
    calc L₀ ^ 2 / 4
        = (L₀ / 2) * (L₀ / 2) := by ring
      _ ≤ Real.exp (L₀ / 2) * Real.exp (L₀ / 2) :=
          mul_le_mul ht ht ht_nn (Real.exp_pos _).le
      _ = Real.exp L₀ := by
          rw [← Real.exp_add]
          congr 1; ring
      _ = 1 / s := hexp_L0
  have hs_prod : s * (1 / s) = 1 := mul_one_div_cancel hs0.ne'
  have hs_le_inv : s ≤ 4 / L₀ ^ 2 := by
    have hL0_sq_pos : 0 < L₀ ^ 2 := by positivity
    rw [le_div_iff₀ hL0_sq_pos]
    have hm := mul_le_mul_of_nonneg_left hsq_le_exp hs0.le
    nlinarith
  have hs_half : s ≤ 1 / 2 := by
    have h16 : (16 : ℝ) ≤ L₀ ^ 2 := by nlinarith
    have h4_le : (4 : ℝ) / L₀ ^ 2 ≤ 1 / 2 := by
      rw [div_le_div_iff₀ (by positivity) (by norm_num)]
      linarith
    exact hs_le_inv.trans h4_le
  have hlog1s_neg : Real.log (1 - s) < 0 := Real.log_neg h1s (by linarith)
  have hneg_log_le : -Real.log (1 - s) ≤ 2 * s := by
    have hquad : 1 ≤ (1 - s) * (1 + 2 * s) := by nlinarith
    have hexp2s : 1 + 2 * s ≤ Real.exp (2 * s) := by linarith [Real.add_one_le_exp (2 * s)]
    have hprod : 1 ≤ (1 - s) * Real.exp (2 * s) :=
      hquad.trans (mul_le_mul_of_nonneg_left hexp2s h1s.le)
    have hlog := Real.log_le_log zero_lt_one hprod
    rw [Real.log_one, Real.log_mul h1s.ne' (Real.exp_pos _).ne', Real.log_exp] at hlog
    linarith
  have hlogL0_1 : 1 ≤ Real.log L₀ := one_le_log_of_four_le hL0_4
  have habs_le : |Real.log (1 - s)| ≤ 8 * (Real.log L₀ / L₀ ^ 2) := by
    rw [abs_of_neg hlog1s_neg]
    calc -Real.log (1 - s)
        ≤ 2 * s := hneg_log_le
      _ ≤ 2 * (4 / L₀ ^ 2) := mul_le_mul_of_nonneg_left hs_le_inv (by norm_num)
      _ = 8 * (1 / L₀ ^ 2) := by ring
      _ ≤ 8 * (Real.log L₀ / L₀ ^ 2) :=
          mul_le_mul_of_nonneg_left
            (div_le_div_of_nonneg_right hlogL0_1 (sq_nonneg L₀)) (by norm_num)
  exact ⟨hs_half, neg_pos.mpr hlog1s_neg, hneg_log_le, habs_le⟩

/-- **Q11 (explicit non-asymptotic threshold correction for `L₀ ≥ L0_min(b, φ)`,
`gap2_proof.md` §4 eq. `(11)` and eq. `(1)`).**
Fix `0 < δ < 1`, `-1 < φ < 1`, and `0 < s < 1` with `L0_min(artanh δ, φ) ≤ L₀ := log(1/s)`.
Let `c_star > 0` satisfy the threshold zero-growth identity `H6` (`h_zero_growth`),
where `(X_prev, Z_prev, Z_curr)` on `(Ω, μ)` satisfies
`H1` (`h_stat` at `L_* := L₀ + (log c_star - ℓ(δ))/2`), `H2`/`H3` (`h_Z_pm`, `h_Z_zero`), and `H4`
(`h_markov`). Put `D = log(c_star/(1 - s))`. Then:
1. `L_min(artanh δ, φ) ≤ L_*`, `L₀ / 2 ≤ L_*`, and `L_* ≤ L₀`.
2. `0 < K_D(δ, φ)` and
   `|(D - ℓ(δ)) - (-V_π(δ, φ) / (2·L₀))| ≤ K_D(δ, φ)·(log L₀ / L₀²)`. -/
theorem Q11 {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (δ φ s c_star : ℝ) (hδ0 : 0 < δ) (hδ1 : δ < 1) (hφ_lo : -1 < φ) (hφ_hi : φ < 1)
    (hs0 : 0 < s) (hs1 : s < 1)
    (hL0 : L0min (Real.artanh δ) φ ≤ Real.log (1 / s))
    (X_prev Z_prev Z_curr : Ω → ℝ)
    (h_stat : IdentDistrib
      (fun ω =>
        (Fstep (Real.log (1 / s) + (Real.log c_star - ell δ) / 2)
          (Real.artanh δ * Z_curr ω) (X_prev ω), Z_curr ω))
      (fun ω => (X_prev ω, Z_prev ω)) μ μ)
    (h_Z_pm : ∀ᵐ ω ∂μ, Z_curr ω = 1 ∨ Z_curr ω = -1)
    (h_Z_zero : ∫ ω, Z_curr ω ∂μ = 0)
    (h_markov : ∫ ω, X_prev ω * Z_curr ω ∂μ = φ * ∫ ω, X_prev ω * Z_prev ω ∂μ)
    (h_zero_growth : (Real.log c_star - ell δ) + 2 * (∫ ω, X_prev ω ∂μ) = 0)
    (hc_pos : 0 < c_star) :
    let b := Real.artanh δ
    let L₀ := Real.log (1 / s)
    let m_star := Real.log c_star - ell δ
    let L_star := L₀ + m_star / 2
    let D_val := Real.log (c_star / (1 - s))
    (Lmin b φ ≤ L_star ∧ L₀ / 2 ≤ L_star ∧ L_star ≤ L₀)
      ∧ 0 < KD δ φ
      ∧ |(D_val - ell δ) - (-Vpi δ φ / (2 * L₀))| ≤ KD δ φ * (Real.log L₀ / L₀ ^ 2) := by
  dsimp only
  obtain ⟨_, _, _, _, hb_pos, _, hV_pos⟩ := Q1 δ φ hδ0 hδ1 hφ_lo hφ_hi
  set b := Real.artanh δ
  set L₀ := Real.log (1 / s)
  set m_star := Real.log c_star - ell δ
  set L_star := L₀ + m_star / 2
  set D_val := Real.log (c_star / (1 - s))
  set V := Vpi δ φ
  have h1p : 0 < 1 + φ := by linarith
  have hQstar_pos : 0 < Qstar b := by
    unfold Qstar
    exact Real.log_pos (by linarith [Real.exp_pos (2 * b + 1)])
  have hCminus_pos : 0 < Cminus b φ := by
    unfold Cminus
    have : 0 ≤ 2 * b / (1 + φ) := div_nonneg (by linarith) h1p.le
    linarith
  have hLmin_gt : 64 < Lmin b φ := by unfold Lmin; nlinarith
  have hL0_ge_2Lmin : 2 * Lmin b φ ≤ L₀ := le_trans (le_max_left _ _) hL0
  have hL0_ge_2b1 : 2 * (b + 1) ≤ L₀ := le_trans (le_max_right _ _) hL0
  have hL0_gt128 : 128 < L₀ := by linarith
  have hL0_ge4 : 4 ≤ L₀ := by linarith
  have hlogL0_1 : 1 ≤ Real.log L₀ := one_le_log_of_four_le hL0_ge4
  -- Apply `Q10`
  obtain ⟨⟨_, hD_eq⟩, ⟨hm_lo, _⟩, hQ10_tail⟩ := Q10 μ δ s c_star hδ0 hδ1 hs0 hs1 hc_pos
    X_prev Z_prev Z_curr h_stat h_Z_pm h_zero_growth
  obtain ⟨hs_half, _, hneg_log_le, hlog1s_abs⟩ := hQ10_tail hL0_ge4
  -- Deduce `L_min ≤ L₀/2 ≤ L_*`
  have hm_ge : -2 * (b + 1) ≤ m_star := by linarith
  have hLs_ge_half : L₀ / 2 ≤ L_star := by unfold L_star; linarith
  have hLs_ge_Lmin : Lmin b φ ≤ L_star := by linarith
  have hLs_gt64 : 64 < L_star := lt_of_lt_of_le hLmin_gt hLs_ge_Lmin
  -- Apply `Q5`, `Q8`, and `Q9` at `L = L_*`
  obtain ⟨_, h_eq4, _, _⟩ :=
    Q5 μ L_star b φ hb_pos.le hφ_lo hφ_hi X_prev Z_prev Z_curr h_stat h_Z_pm h_Z_zero h_markov
  obtain ⟨⟨hq_nn, _⟩, _, _⟩ :=
    Q8 μ L_star b φ hb_pos.le hφ_lo hφ_hi (by linarith)
      X_prev Z_prev Z_curr h_stat h_Z_pm h_Z_zero h_markov
  obtain ⟨hK_nn, hq_err_Ls⟩ :=
    Q9 μ L_star b φ hb_pos.le hφ_lo hφ_hi hLs_ge_Lmin
      X_prev Z_prev Z_curr h_stat h_Z_pm h_Z_zero h_markov
  set q_star := ∫ ω, Qstep L_star (X_prev ω) (b * Z_curr ω) ∂μ
  set K := Kq b φ
  have hm_eq_neg_q : m_star = -q_star := by linarith [h_zero_growth, h_eq4]
  have hLs_le_L0 : L_star ≤ L₀ := by unfold L_star; linarith
  have hKD_pos : 0 < KD δ φ := by
    unfold KD
    have : 0 ≤ V * (V + 4 * K) / 2 := by positivity
    linarith
  -- Convert `K·(log L_* / L_*²)` to `4·K·(log L₀ / L₀²)`
  have hV_def : b ^ 2 * (1 - φ) / (1 + φ) = V := rfl
  rw [hV_def] at hq_err_Ls
  have hlogLs_le : Real.log L_star ≤ Real.log L₀ :=
    Real.log_le_log (by linarith) hLs_le_L0
  have hlogLs_nn : 0 ≤ Real.log L_star :=
    le_trans zero_le_one (one_le_log_of_four_le (by linarith))
  have hLs_sq_ge : L₀ ^ 2 / 4 ≤ L_star ^ 2 := by nlinarith only [hLs_ge_half, hL0_ge4]
  have hratio_Ls_le : Real.log L_star / L_star ^ 2 ≤ 4 * (Real.log L₀ / L₀ ^ 2) := by
    calc Real.log L_star / L_star ^ 2
        ≤ Real.log L₀ / L_star ^ 2 :=
          div_le_div_of_nonneg_right hlogLs_le (sq_nonneg L_star)
      _ ≤ Real.log L₀ / (L₀ ^ 2 / 4) :=
          div_le_div_of_nonneg_left (by linarith only [hlogL0_1]) (by positivity) hLs_sq_ge
      _ = 4 * (Real.log L₀ / L₀ ^ 2) := by ring
  have h_err1 : |q_star - V / (2 * L_star)| ≤ (4 * K) * (Real.log L₀ / L₀ ^ 2) := by
    calc |q_star - V / (2 * L_star)|
        ≤ K * (Real.log L_star / L_star ^ 2) := hq_err_Ls
      _ ≤ K * (4 * (Real.log L₀ / L₀ ^ 2)) := mul_le_mul_of_nonneg_left hratio_Ls_le hK_nn
      _ = (4 * K) * (Real.log L₀ / L₀ ^ 2) := by ring
  -- Bound `|V/(2·L_*) - V/(2·L₀)|`
  have hq_upper : q_star ≤ (V + 4 * K) * (Real.log L₀ / L₀) := by
    have hq1 : q_star ≤ V / (2 * L_star) + (4 * K) * (Real.log L₀ / L₀ ^ 2) := by
      linarith only [(abs_le.mp h_err1).2]
    have hV_div : V / (2 * L_star) ≤ V * (Real.log L₀ / L₀) := by
      calc V / (2 * L_star)
          ≤ V / L₀ := div_le_div_of_nonneg_left hV_pos.le (by linarith only [hL0_ge4])
              (by linarith only [hLs_ge_half])
        _ = V * (1 / L₀) := by ring
        _ ≤ V * (Real.log L₀ / L₀) :=
            mul_le_mul_of_nonneg_left
              (div_le_div_of_nonneg_right hlogL0_1 (by linarith only [hL0_ge4])) hV_pos.le
    have hK_div : Real.log L₀ / L₀ ^ 2 ≤ Real.log L₀ / L₀ :=
      div_le_div_of_nonneg_left (by linarith only [hlogL0_1]) (by linarith only [hL0_ge4])
        (by nlinarith only [hL0_ge4])
    have hK_mul : (4 * K) * (Real.log L₀ / L₀ ^ 2) ≤ (4 * K) * (Real.log L₀ / L₀) :=
      mul_le_mul_of_nonneg_left hK_div (by positivity)
    have hsum_VK : V * (Real.log L₀ / L₀) + (4 * K) * (Real.log L₀ / L₀)
        = (V + 4 * K) * (Real.log L₀ / L₀) := by ring
    linarith only [hq1, hV_div, hK_mul, hsum_VK]
  have h_shift_id : V / (2 * L_star) - V / (2 * L₀) = (V * q_star) / (4 * L_star * L₀) := by
    have hq_eq : q_star = 2 * (L₀ - L_star) := by unfold L_star; linarith only [hm_eq_neg_q]
    have h2Ls : 2 * L_star ≠ 0 := by linarith only [hLs_gt64]
    have h2L0 : 2 * L₀ ≠ 0 := by linarith only [hL0_ge4]
    have h4 : 4 * L_star * L₀ ≠ 0 := by positivity
    rw [div_sub_div _ _ h2Ls h2L0, div_eq_div_iff (mul_ne_zero h2Ls h2L0) h4, hq_eq]
    ring
  have h_shift_nn : 0 ≤ V / (2 * L_star) - V / (2 * L₀) := by
    rw [h_shift_id]; positivity
  have h_err2 : |V / (2 * L_star) - V / (2 * L₀)|
      ≤ (V * (V + 4 * K) / 2) * (Real.log L₀ / L₀ ^ 2) := by
    rw [abs_of_nonneg h_shift_nn, h_shift_id]
    have hden : 2 * L₀ ^ 2 ≤ 4 * L_star * L₀ := by nlinarith only [hLs_ge_half, hL0_ge4]
    calc (V * q_star) / (4 * L_star * L₀)
        ≤ (V * q_star) / (2 * L₀ ^ 2) :=
          div_le_div_of_nonneg_left (by positivity) (by positivity) hden
      _ ≤ (V * ((V + 4 * K) * (Real.log L₀ / L₀))) / (2 * L₀ ^ 2) :=
          div_le_div_of_nonneg_right
            (mul_le_mul_of_nonneg_left hq_upper hV_pos.le) (by positivity)
      _ = (V * (V + 4 * K) / (2 * L₀)) * (Real.log L₀ / L₀ ^ 2) := by ring
      _ ≤ (V * (V + 4 * K) / 2) * (Real.log L₀ / L₀ ^ 2) := by
          refine mul_le_mul_of_nonneg_right ?_
            (div_nonneg (by linarith only [hlogL0_1]) (sq_nonneg L₀))
          exact div_le_div_of_nonneg_left (by positivity) (by norm_num)
            (by linarith only [hL0_ge4])
  -- Assemble the three error pieces via `abs_le`
  refine ⟨⟨hLs_ge_Lmin, hLs_ge_half, hLs_le_L0⟩, hKD_pos, ?_⟩
  have h_decomp : (D_val - ell δ) - (-V / (2 * L₀))
      = -(q_star - V / (2 * L_star)) - (V / (2 * L_star) - V / (2 * L₀)) - Real.log (1 - s) := by
    rw [hD_eq, show Real.log c_star - ell δ = -q_star from hm_eq_neg_q]
    ring
  have hKD_sum : KD δ φ * (Real.log L₀ / L₀ ^ 2)
      = (4 * K) * (Real.log L₀ / L₀ ^ 2)
        + (V * (V + 4 * K) / 2) * (Real.log L₀ / L₀ ^ 2)
        + 8 * (Real.log L₀ / L₀ ^ 2) := by
    unfold KD
    dsimp only
    ring
  obtain ⟨he1_lo, he1_hi⟩ := abs_le.mp h_err1
  obtain ⟨he2_lo, he2_hi⟩ := abs_le.mp h_err2
  obtain ⟨he3_lo, he3_hi⟩ := abs_le.mp hlog1s_abs
  rw [h_decomp, hKD_sum]
  exact abs_le.mpr
    ⟨by linarith only [he1_lo, he1_hi, he2_lo, he2_hi, he3_lo, he3_hi],
     by linarith only [he1_lo, he1_hi, he2_lo, he2_hi, he3_lo, he3_hi]⟩

/-- Elementary bound `0 ≤ log x / x ≤ 2 / √x` for `x ≥ 1`. -/
theorem log_div_self_le_two_div_sqrt {x : ℝ} (hx : 1 ≤ x) :
    0 ≤ Real.log x / x ∧ Real.log x / x ≤ 2 / Real.sqrt x := by
  have hx_pos : 0 < x := by linarith
  refine ⟨div_nonneg (Real.log_nonneg hx) hx_pos.le, ?_⟩
  set u := Real.sqrt x
  have hu_pos : 0 < u := Real.sqrt_pos.mpr hx_pos
  have hu_sq : u * u = x := Real.mul_self_sqrt hx_pos.le
  have hu_exp : u ≤ Real.exp (u - 1) := by linarith [Real.add_one_le_exp (u - 1)]
  have hlogu : Real.log u ≤ u - 1 := by
    have h := Real.log_le_log hu_pos hu_exp
    rwa [Real.log_exp] at h
  have hlogx : Real.log x = 2 * Real.log u := by
    rw [← hu_sq, Real.log_mul hu_pos.ne' hu_pos.ne']
    ring
  rw [hlogx, ← hu_sq]
  calc (2 * Real.log u) / (u * u)
      ≤ (2 * u) / (u * u) := div_le_div_of_nonneg_right (by linarith) (by positivity)
    _ = 2 / u := mul_div_mul_right 2 u hu_pos.ne'

/-- For every real `B`, eventually as `s ↓ 0` we have `0 < s < 1` and `B ≤ log(1/s)`. -/
theorem eventually_Ioo_and_le_log_inv (B : ℝ) :
    ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), s ∈ Ioo 0 1 ∧ B ≤ Real.log (1 / s) := by
  set s_hi := min (1 / 2 : ℝ) (Real.exp (-B))
  have hs_hi_pos : 0 < s_hi := lt_min (by norm_num) (Real.exp_pos _)
  filter_upwards [Ioo_mem_nhdsGT hs_hi_pos] with s hs
  have hs0 : 0 < s := hs.1
  have hs_half : s < 1 / 2 := lt_of_lt_of_le hs.2 (min_le_left _ _)
  have hs_exp : s < Real.exp (-B) := lt_of_lt_of_le hs.2 (min_le_right _ _)
  refine ⟨⟨hs0, by linarith⟩, ?_⟩
  have hexpB : Real.exp B ≤ 1 / s := by
    rw [le_div_iff₀ hs0]
    calc Real.exp B * s
        ≤ Real.exp B * Real.exp (-B) :=
          mul_le_mul_of_nonneg_left hs_exp.le (Real.exp_pos B).le
      _ = 1 := by rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hlog := Real.log_le_log (Real.exp_pos B) hexpB
  rwa [Real.log_exp] at hlog

/-- **Q12 (main `s ↓ 0` asymptotic theorem, `gap2_proof.md` Proposition eq. `(1)`).**
Fix `0 < δ < 1` and `-1 < φ < 1`. Suppose that for each `s ∈ (0, 1)`, `c_star s` and a stationary
realization `(X_prev s, Z_prev s, Z_curr s)` on `(Ω, μ s)` satisfy `H1` (`h_stat`), `H2`/`H3`
(`h_Z_pm`, `h_Z_zero`), `H4` (`h_markov`), and `H6` (`h_zero_growth`), with `c_star s > 0`.
Write `D(s) = log(c_star s / (1 - s))`. Then as `s ↓ 0` (along `nhdsWithin 0 (Ioi 0)`):
1. `D(s) - ℓ(δ) - (-V_π(δ, φ) / (2·log(1/s))) = O(log(log(1/s)) / (log(1/s))²)` (`eq. (1)`).
2. `log(1/s)·(D(s) - ℓ(δ)) → -V_π(δ, φ) / 2` (exact nonzero negative leading coefficient).
3. Eventually `D(s) < ℓ(δ)` strictly.
4. `D(s) → ℓ(δ)` (reproving the zero-staying endpoint `memA` P6). -/
theorem Q12 {Ω : Type*} [MeasurableSpace Ω] (μ : ℝ → Measure Ω)
    (δ φ : ℝ) (c_star : ℝ → ℝ) (X_prev Z_prev Z_curr : ℝ → Ω → ℝ)
    (hδ0 : 0 < δ) (hδ1 : δ < 1) (hφ_lo : -1 < φ) (hφ_hi : φ < 1)
    (h_prob : ∀ s ∈ Ioo (0 : ℝ) 1, IsProbabilityMeasure (μ s))
    (h_stat : ∀ s ∈ Ioo (0 : ℝ) 1,
      IdentDistrib
        (fun ω =>
          (Fstep (Real.log (1 / s) + (Real.log (c_star s) - ell δ) / 2)
            (Real.artanh δ * Z_curr s ω) (X_prev s ω), Z_curr s ω))
        (fun ω => (X_prev s ω, Z_prev s ω)) (μ s) (μ s))
    (h_Z_pm : ∀ s ∈ Ioo (0 : ℝ) 1, ∀ᵐ ω ∂(μ s), Z_curr s ω = 1 ∨ Z_curr s ω = -1)
    (h_Z_zero : ∀ s ∈ Ioo (0 : ℝ) 1, ∫ ω, Z_curr s ω ∂(μ s) = 0)
    (h_markov : ∀ s ∈ Ioo (0 : ℝ) 1,
      ∫ ω, X_prev s ω * Z_curr s ω ∂(μ s) = φ * ∫ ω, X_prev s ω * Z_prev s ω ∂(μ s))
    (h_zero_growth : ∀ s ∈ Ioo (0 : ℝ) 1,
      (Real.log (c_star s) - ell δ) + 2 * (∫ ω, X_prev s ω ∂(μ s)) = 0)
    (hc_pos : ∀ s ∈ Ioo (0 : ℝ) 1, 0 < c_star s) :
    let D := fun s => Real.log (c_star s / (1 - s))
    ((fun s => (D s - ell δ) - (-Vpi δ φ / (2 * Real.log (1 / s))))
        =O[nhdsWithin (0 : ℝ) (Ioi 0)]
        (fun s => Real.log (Real.log (1 / s)) / (Real.log (1 / s)) ^ 2))
      ∧ Tendsto (fun s => Real.log (1 / s) * (D s - ell δ))
          (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds (-Vpi δ φ / 2))
      ∧ (∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), D s < ell δ)
      ∧ Tendsto D (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds (ell δ)) := by
  dsimp only
  obtain ⟨_, _, _, _, _, _, hV_pos⟩ := Q1 δ φ hδ0 hδ1 hφ_lo hφ_hi
  set D := fun s : ℝ => Real.log (c_star s / (1 - s))
  -- Pointwise wrapper around `Q11` for any `s ∈ (0, 1)` with `L0min ≤ log(1/s)`
  have hQ11_at : ∀ s ∈ Ioo (0 : ℝ) 1, L0min (Real.artanh δ) φ ≤ Real.log (1 / s) →
      0 < KD δ φ ∧
        |(D s - ell δ) - (-Vpi δ φ / (2 * Real.log (1 / s)))|
          ≤ KD δ φ * (Real.log (Real.log (1 / s)) / (Real.log (1 / s)) ^ 2) := by
    intro s hs hL0
    have : IsProbabilityMeasure (μ s) := h_prob s hs
    obtain ⟨_, hKD, hbd⟩ := Q11 (μ s) δ φ s (c_star s) hδ0 hδ1 hφ_lo hφ_hi hs.1 hs.2 hL0
      (X_prev s) (Z_prev s) (Z_curr s)
      (h_stat s hs) (h_Z_pm s hs) (h_Z_zero s hs) (h_markov s hs)
      (h_zero_growth s hs) (hc_pos s hs)
    exact ⟨hKD, hbd⟩
  -- Part 1: `=O[nhdsWithin 0 (Ioi 0)]`
  have h_bigO :
      (fun s => (D s - ell δ) - (-Vpi δ φ / (2 * Real.log (1 / s))))
        =O[nhdsWithin (0 : ℝ) (Ioi 0)]
        (fun s => Real.log (Real.log (1 / s)) / (Real.log (1 / s)) ^ 2) := by
    refine Asymptotics.IsBigO.of_bound (KD δ φ) ?_
    filter_upwards [eventually_Ioo_and_le_log_inv (max 4 (L0min (Real.artanh δ) φ))]
      with s ⟨hs, hB⟩
    have h4 : 4 ≤ Real.log (1 / s) := le_trans (le_max_left _ _) hB
    have hL0 : L0min (Real.artanh δ) φ ≤ Real.log (1 / s) := le_trans (le_max_right _ _) hB
    have hbd := (hQ11_at s hs hL0).2
    have hlog_nn : 0 ≤ Real.log (Real.log (1 / s)) / (Real.log (1 / s)) ^ 2 :=
      div_nonneg (le_trans zero_le_one (one_le_log_of_four_le h4)) (sq_nonneg _)
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg hlog_nn]
    exact hbd
  -- Part 2: scaled limit `log(1/s)·(D(s) - ℓ) → -V_π / 2`
  have h_scaled : Tendsto (fun s => Real.log (1 / s) * (D s - ell δ))
      (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds (-Vpi δ φ / 2)) := by
    rw [Metric.tendsto_nhds]
    intro ε hε
    set B := max (L0min (Real.artanh δ) φ) (max 4 ((2 * KD δ φ / ε + 1) ^ 2))
    filter_upwards [eventually_Ioo_and_le_log_inv B] with s ⟨hs, hB⟩
    set L₀ := Real.log (1 / s)
    have hL0_min : L0min (Real.artanh δ) φ ≤ L₀ := le_trans (le_max_left _ _) hB
    have hB2 : max 4 ((2 * KD δ φ / ε + 1) ^ 2) ≤ L₀ := le_trans (le_max_right _ _) hB
    have hL0_4 : 4 ≤ L₀ := le_trans (le_max_left _ _) hB2
    have hL0_sq : (2 * KD δ φ / ε + 1) ^ 2 ≤ L₀ := le_trans (le_max_right _ _) hB2
    obtain ⟨hKD_pos, hbd⟩ := hQ11_at s hs hL0_min
    have hL0_pos : 0 < L₀ := by linarith only [hL0_4]
    have hdist_eq : dist (L₀ * (D s - ell δ)) (-Vpi δ φ / 2)
        = L₀ * |(D s - ell δ) - (-Vpi δ φ / (2 * L₀))| := by
      have h_in : L₀ * (D s - ell δ) - (-Vpi δ φ / 2)
          = L₀ * ((D s - ell δ) - (-Vpi δ φ / (2 * L₀))) := by
        have hL0_ne : L₀ ≠ 0 := hL0_pos.ne'
        field_simp
      rw [Real.dist_eq, h_in, abs_mul, abs_of_pos hL0_pos]
    have hmul_bd : L₀ * |(D s - ell δ) - (-Vpi δ φ / (2 * L₀))|
        ≤ KD δ φ * (Real.log L₀ / L₀) := by
      calc L₀ * |(D s - ell δ) - (-Vpi δ φ / (2 * L₀))|
          ≤ L₀ * (KD δ φ * (Real.log L₀ / L₀ ^ 2)) :=
            mul_le_mul_of_nonneg_left hbd hL0_pos.le
        _ = KD δ φ * (Real.log L₀ / L₀) := by
            have hL0_ne : L₀ ≠ 0 := hL0_pos.ne'
            field_simp
    have hsqrt_bd : Real.log L₀ / L₀ ≤ 2 / Real.sqrt L₀ :=
      (log_div_self_le_two_div_sqrt (by linarith only [hL0_4])).2
    have hsqrt_pos : 0 < Real.sqrt L₀ := Real.sqrt_pos.mpr hL0_pos
    have hsqrt_gt : 2 * KD δ φ / ε < Real.sqrt L₀ := by
      have hsq_eq : (Real.sqrt L₀) ^ 2 = L₀ := Real.sq_sqrt hL0_pos.le
      have hpos_lhs : 0 < 2 * KD δ φ / ε + 1 := by positivity
      have hle : 2 * KD δ φ / ε + 1 ≤ Real.sqrt L₀ := by
        by_contra hlt
        nlinarith only [hL0_sq, hsq_eq, hpos_lhs, hsqrt_pos, hlt]
      linarith only [hle]
    have hlt_ε : KD δ φ * (2 / Real.sqrt L₀) < ε := by
      have h1 : 2 * KD δ φ < Real.sqrt L₀ * ε := (div_lt_iff₀ hε).mp hsqrt_gt
      have h2 : KD δ φ * (2 / Real.sqrt L₀) = (2 * KD δ φ) / Real.sqrt L₀ := by ring
      rw [h2, div_lt_iff₀ hsqrt_pos]
      linarith only [h1]
    calc dist (L₀ * (D s - ell δ)) (-Vpi δ φ / 2)
        = L₀ * |(D s - ell δ) - (-Vpi δ φ / (2 * L₀))| := hdist_eq
      _ ≤ KD δ φ * (Real.log L₀ / L₀) := hmul_bd
      _ ≤ KD δ φ * (2 / Real.sqrt L₀) :=
          mul_le_mul_of_nonneg_left hsqrt_bd hKD_pos.le
      _ < ε := hlt_ε
  -- Part 3: eventual strict negativity `D(s) < ℓ(δ)`
  have h_strict_neg : ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), D s < ell δ := by
    have hneg_lim : -Vpi δ φ / 2 < 0 := by linarith only [hV_pos]
    filter_upwards [h_scaled.eventually (gt_mem_nhds hneg_lim),
      eventually_Ioo_and_le_log_inv 1] with s hprod_neg ⟨_, hL1⟩
    by_contra hge
    have hnn : 0 ≤ Real.log (1 / s) * (D s - ell δ) :=
      mul_nonneg (by linarith only [hL1]) (by linarith only [hge])
    linarith only [hprod_neg, hnn]
  -- Part 4: endpoint limit `D(s) → ℓ(δ)`
  have h_inv_zero : Tendsto (fun s => (Real.log (1 / s))⁻¹)
      (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds 0) := by
    rw [Metric.tendsto_nhds]
    intro ε hε
    filter_upwards [eventually_Ioo_and_le_log_inv (max 1 (1 / ε + 1))] with s ⟨_, hB⟩
    have h1 : 1 ≤ Real.log (1 / s) := le_trans (le_max_left _ _) hB
    have hgt : 1 / ε < Real.log (1 / s) := by
      have : 1 / ε + 1 ≤ Real.log (1 / s) := le_trans (le_max_right _ _) hB
      linarith only [this]
    have hpos : 0 < Real.log (1 / s) := by linarith only [h1]
    rw [Real.dist_eq, sub_zero, abs_of_pos (inv_pos.mpr hpos), ← one_div]
    rw [div_lt_iff₀ hpos]
    have hm : 1 < Real.log (1 / s) * ε := (div_lt_iff₀ hε).mp hgt
    linarith only [hm]
  have h_endpt : Tendsto D (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds (ell δ)) := by
    have hdiff0 : Tendsto (fun s => (Real.log (1 / s))⁻¹ * (Real.log (1 / s) * (D s - ell δ)))
        (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds (0 * (-Vpi δ φ / 2))) :=
      h_inv_zero.mul h_scaled
    simp only [zero_mul] at hdiff0
    have hadd := hdiff0.add (tendsto_const_nhds (x := ell δ))
    simp only [zero_add] at hadd
    refine hadd.congr' ?_
    filter_upwards [eventually_Ioo_and_le_log_inv 1] with s ⟨_, hL1⟩
    have hne : Real.log (1 / s) ≠ 0 := by linarith only [hL1]
    calc (Real.log (1 / s))⁻¹ * (Real.log (1 / s) * (D s - ell δ)) + ell δ
        = (D s - ell δ) + ell δ := by rw [inv_mul_cancel_left₀ hne]
      _ = D s := sub_add_cancel _ _
  exact ⟨h_bigO, h_scaled, h_strict_neg, h_endpt⟩

end Dobson.GapTwoCorrection

#print axioms Dobson.GapTwoCorrection.Q1
#print axioms Dobson.GapTwoCorrection.Q2
#print axioms Dobson.GapTwoCorrection.Q3
#print axioms Dobson.GapTwoCorrection.Q4
#print axioms Dobson.GapTwoCorrection.Q5
#print axioms Dobson.GapTwoCorrection.Q6
#print axioms Dobson.GapTwoCorrection.Q7
#print axioms Dobson.GapTwoCorrection.Q8
#print axioms Dobson.GapTwoCorrection.Q9
#print axioms Dobson.GapTwoCorrection.stationary_threshold_bracket
#print axioms Dobson.GapTwoCorrection.Q10
#print axioms Dobson.GapTwoCorrection.Q11
#print axioms Dobson.GapTwoCorrection.Q12
