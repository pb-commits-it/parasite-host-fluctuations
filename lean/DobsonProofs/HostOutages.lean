import Mathlib

/-!
# Hosts with zeros (extension A13): machine-checked core (`DobsonProofs/HostOutages.lean`)

Formalises in Lean 4 / Mathlib the core of the hosts-with-zeros extension (`a13/a13_proof.md`),
incorporating the corrections from the independent host-outage referee audit:

1. **Exact period-2 outage formula (`O1`–`O3`, `a13_proof.md` §1 Proposition 1 & dated fix):**
   - Two-step matrix product `A(0) A(2)`, determinant identity for both phases, corrected threshold
     `(1, 1)` entry `1 − q (p + q) (1 − p²) / (1 + p q)`, and the trace-minus-one bound needed for Perron identification (`O1`);
   - Exact root `c = cAlt(a, s)` and `D = DAlt(a, s)`, numerator-minus-denominator `= −1`, and
     global negativity `DAlt(a, s) < 0` for all `a > 1, s ≥ 1` (`O2`);
   - Exact derivative `dD/ds = 1 / [(2s − 1)((2a − 1)s − a + 1)] > 0`, strict monotonicity in `s` on
     `[1, ∞)`, and `s → ∞` limits `DAlt(a, s) → 0` and `s · DAlt(a, s) → −1 / [2(2a − 1)]` (`O3`).
2. **Algebraic and inequality core of the proved Part ii bound (`O4`–`O8`, `a13_proof.md` §6 Proposition 7 & dated fixes):**
   - Simultaneous Krylov companion similarity to permute stage stays, with `krylovMat` proved
     **upper triangular** with unit diagonal (`O4`);
   - Invariance of `Ccone = {x ≥ 0 : (C − pmax I) x ≥ 0}` under `A(h)` for all `h ≥ 0`, entrywise
     zero-deletion inequality `A(h_{T−1}) ⋯ A(h_0) e₁ ≥ pmax^{n₀} Q e₁` `(6.3)`, and `(k − 1)`-step
     positivity `(C^{k−1} e₁)_i ≥ pmax^{k−1−i} > 0` (`O5`);
   - Deterministic finite-path reference-chain identity and Jensen inequality `(6.4)` for all `k ≥ 1`,
     keeping two labelled self-edges at `k = 1` (`O6`);
   - General ring determinant and characteristic identity at `(λ_*, Dsup)` and conditional Part ii threshold theorem `D ≤ Dsup`
     under three explicit probability hypotheses (`O7`);
   - Corrected `s → ∞` limit `Dsup → −log[(1 − q₀) G₊] ≥ 0` `(6.5)` and strict finite-law Jensen
     characterization vanishing iff pure binary outages `h_i = 1 / (1 − q₀)` (`O8`).
3. **Sign criterion of Part i, as an algebraic statement, where it is proved (`O9`–`O12`, `a13_proof.md` §§4–5 Propositions 4–6):**
   - Raw-covariance bracket `A = V / 2 + ∑_{ℓ ≥ 1} p^{ℓ−1} C_ℓ` `(4.1)`, absolute summability,
     normalized autocorrelation sign equivalence for `V > 0`, and evaluation on `h = (2, 0)` (`O9`);
   - Stage-ratio `[0, Vmax]` invariance `(4.4)`, exact `(δ, η)` recursions `(4.5)`, `B(1) = A`, and
     conditional fourth-moment root localization `|D − A ε| ≤ C ε²` `(4.3)` under two explicit
     hypotheses (`O10`);
   - Unconditional algebraic sign and fixed-factor stay-reversal criterion `(4.8)` (`Proposition 5`):
     `|D(s) − A/s| ≤ C/s²` implies `sign(D(s)) = sign(A)` and `sign(D(κ s) − D(s)) = −sign(A)` (`O11`);
   - Bounded-host contraction constants, Rouché margin `25/256 < 3/2 − exp(1/4)`, Cauchy series
     derivative bounds `(5.1)`, and eventual pointwise derivative sign `sign(dD/ds) = −sign(A)` on
     `(5.2)` (`Proposition 6`) under one explicit Cauchy-series hypothesis (`O12`–`O13`).

There are six named probability/analysis hypothesis bundles, `H1`–`H6`, and
four further coefficient/remainder handoffs listed in `STATUS.md` (ten in total).
They are explicit premises of conditional theorems, not axioms. Covariance bounds
and positive normalized eigenvector identities are also structural premises of
the finite algebra. No stationary process is constructed here; see `STATUS.md`
for the exact fidelity boundaries and counting convention.
-/

/-! ## Coverage correction — 2026-11-22

Both outage rows retain partial-P1 coverage. The actual covariance polynomial
now supplies `B(1)`, its interval bound and its Lipschitz bound. The pullback
expansion `H4`, the supplied root/monotonicity input `H5`, the `hrem` expansion
and analytic-series input `H6` remain explicit in their respective conclusions.
Finite path, cone, conjugacy and root algebra do not establish the general
physical growth/root construction. See the independent hypothesis audit
(2026-11-16) and the version-2 inventory for the credited scope. -/

namespace Dobson.HostOutages

open scoped BigOperators Matrix Topology

/-! ## Section 1: Exact Period-2 Outage Formula (`O1`–`O3`) -/

/-- Two-stage step matrix `[[p, g], [b * h, q]]` (`g = ε * c`, `b = 1 / a`, `p = 1 − b`, `q = 1 − ε`). -/
def stepMat2 (p q b g h : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![p, g;
     b * h, q]

/-- Period-2 alternating outage threshold multiplier
`cAlt(a, s) = (2a − 1)(2s − 1) / [2 ((2a − 1)s − a + 1)]`. -/
noncomputable def cAlt (a s : ℝ) : ℝ :=
  (2 * a - 1) * (2 * s - 1) / (2 * ((2 * a - 1) * s - a + 1))

/-- Period-2 alternating outage restoring log-multiplier `DAlt(a, s) = log(cAlt(a, s))`. -/
noncomputable def DAlt (a s : ℝ) : ℝ :=
  Real.log (cAlt a s)

/-- **O1 (`a13_proof.md` §1 & dated fix lines 406–414; `a13-check.md` lines 20, 140).**
Two-step matrix product `A(0) A(2)`, determinant identity for both phases `A(0) A(2)` and `A(2) A(0)`,
corrected threshold `(1, 1)` entry `1 − q (p + q) (1 − p²) / (1 + p q)`, and Perron dominance check
(`|trace(A(0) A(2)) − 1| < 1` for `0 < p < 1` and `0 ≤ q < 1`, including `q = 0` at `s = 1`). -/
theorem O1_period2_prod_det_and_perron (p q b g : ℝ) :
    (stepMat2 p q b g 0 * stepMat2 p q b g 2 =
      !![p ^ 2 + 2 * g * b, g * (p + q);
         2 * q * b, q ^ 2]) ∧
    (((1 : Matrix (Fin 2) (Fin 2) ℝ) - stepMat2 p q b g 0 * stepMat2 p q b g 2).det =
      (1 - p ^ 2) * (1 - q ^ 2) - 2 * g * b * (1 + p * q)) ∧
    (((1 : Matrix (Fin 2) (Fin 2) ℝ) - stepMat2 p q b g 2 * stepMat2 p q b g 0).det =
      (1 - p ^ 2) * (1 - q ^ 2) - 2 * g * b * (1 + p * q)) ∧
    (0 < p → p < 1 → 0 ≤ q → q < 1 →
      2 * g * b = (1 - p ^ 2) * (1 - q ^ 2) / (1 + p * q) →
        p ^ 2 + 2 * g * b = 1 - q * (p + q) * (1 - p ^ 2) / (1 + p * q) ∧
        0 < p ^ 2 + 2 * g * b ∧
        p ^ 2 + 2 * g * b ≤ 1 ∧
        (0 < q → p ^ 2 + 2 * g * b < 1) ∧
        |(stepMat2 p q b g 0 * stepMat2 p q b g 2).trace - 1| < 1) := by
  have hprod02 : stepMat2 p q b g 0 * stepMat2 p q b g 2 =
      !![p ^ 2 + 2 * g * b, g * (p + q);
         2 * q * b, q ^ 2] := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [stepMat2, Matrix.mul_apply, Fin.sum_univ_two] <;> ring
  have hprod20 : stepMat2 p q b g 2 * stepMat2 p q b g 0 =
      !![p ^ 2, p * g + g * q;
         2 * b * p, 2 * b * g + q ^ 2] := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [stepMat2, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;> simp
  refine ⟨hprod02, ?_, ?_, ?_⟩
  · rw [hprod02]
    have hsub : (1 : Matrix (Fin 2) (Fin 2) ℝ) - !![p ^ 2 + 2 * g * b, g * (p + q); 2 * q * b, q ^ 2] =
        !![1 - (p ^ 2 + 2 * g * b), -(g * (p + q)); -(2 * q * b), 1 - q ^ 2] := by
      ext i j; fin_cases i <;> fin_cases j <;> simp
    rw [hsub, Matrix.det_fin_two_of]
    ring
  · rw [hprod20]
    have hsub : (1 : Matrix (Fin 2) (Fin 2) ℝ) - !![p ^ 2, p * g + g * q; 2 * b * p, 2 * b * g + q ^ 2] =
        !![1 - p ^ 2, -(p * g + g * q); -(2 * b * p), 1 - (2 * b * g + q ^ 2)] := by
      ext i j; fin_cases i <;> fin_cases j <;> simp
    rw [hsub, Matrix.det_fin_two_of]
    ring
  · intro hp0 hp1 hq0 hq1 hgb
    have hpq_pos : 0 < 1 + p * q := by nlinarith
    have h1p2 : 0 < 1 - p ^ 2 := by nlinarith
    have h1q2 : 0 < 1 - q ^ 2 := by nlinarith
    have hdiag : p ^ 2 + 2 * g * b = 1 - q * (p + q) * (1 - p ^ 2) / (1 + p * q) := by
      rw [hgb]
      field_simp [hpq_pos.ne']
      ring
    have h2gb_pos : 0 < 2 * g * b := by
      rw [hgb]
      exact div_pos (mul_pos h1p2 h1q2) hpq_pos
    have hdiag_pos : 0 < p ^ 2 + 2 * g * b := by nlinarith
    have hfrac_nn : 0 ≤ q * (p + q) * (1 - p ^ 2) / (1 + p * q) := by
      apply div_nonneg _ hpq_pos.le
      exact mul_nonneg (mul_nonneg hq0 (by linarith)) h1p2.le
    have hdiag_le : p ^ 2 + 2 * g * b ≤ 1 := by linarith [hdiag, hfrac_nn]
    have hdiag_lt : 0 < q → p ^ 2 + 2 * g * b < 1 := by
      intro hqpos
      have hfrac_pos : 0 < q * (p + q) * (1 - p ^ 2) / (1 + p * q) :=
        div_pos (mul_pos (mul_pos hqpos (by linarith)) h1p2) hpq_pos
      linarith [hdiag]
    have htr : (stepMat2 p q b g 0 * stepMat2 p q b g 2).trace = p ^ 2 + 2 * g * b + q ^ 2 := by
      rw [hprod02]
      simp [Matrix.trace, Fin.sum_univ_two]
    have habs : |(stepMat2 p q b g 0 * stepMat2 p q b g 2).trace - 1| < 1 := by
      rw [htr, abs_lt]
      have hq2_nn : 0 ≤ q ^ 2 := sq_nonneg q
      have hq2_lt1 : q ^ 2 < 1 := by nlinarith
      constructor <;> linarith
    exact ⟨hdiag, hdiag_pos, hdiag_le, hdiag_lt, habs⟩

/-- **O2 (`a13_proof.md` §1 Proposition 1, `(1.2)`).**
For `a > 1` and `s ≥ 1` (`b = 1 / a`, `p = 1 − b`, `ε = 1 / s`, `q = 1 − ε`):
1. Numerator minus denominator of `cAlt(a, s)` equals `−1`, and `0 < cAlt(a, s) < 1`;
2. `det(I − A(0) A(2)) = 0 ↔ c = cAlt(a, s)` (for `c : ℝ`), and for `c = exp(D)`,
   `det(I − A(0) A(2)) = 0 ↔ D = DAlt(a, s)`;
3. `DAlt(a, s) < 0` for all `a > 1` and `s ≥ 1`. -/
theorem O2_period2_root_and_neg (a s : ℝ) (ha : 1 < a) (hs : 1 ≤ s) :
    let b := 1 / a
    let p := 1 - b
    let ε := 1 / s
    let q := 1 - ε
    ((2 * a - 1) * (2 * s - 1) - 2 * ((2 * a - 1) * s - a + 1) = -1) ∧
    (0 < cAlt a s ∧ cAlt a s < 1) ∧
    (∀ c : ℝ,
      ((1 : Matrix (Fin 2) (Fin 2) ℝ) -
        stepMat2 p q b (ε * c) 0 * stepMat2 p q b (ε * c) 2).det = 0 ↔ c = cAlt a s) ∧
    (∀ D : ℝ,
      ((1 : Matrix (Fin 2) (Fin 2) ℝ) -
        stepMat2 p q b (ε * Real.exp D) 0 * stepMat2 p q b (ε * Real.exp D) 2).det = 0 ↔
          D = DAlt a s) ∧
    DAlt a s < 0 := by
  intro b p ε q
  have ha0 : 0 < a := zero_lt_one.trans ha
  have hs0 : 0 < s := lt_of_lt_of_le zero_lt_one hs
  have h2a1 : 0 < 2 * a - 1 := by linarith
  have h2s1 : 0 < 2 * s - 1 := by linarith
  have hden_inner : 0 < (2 * a - 1) * s - a + 1 := by nlinarith
  have hnum_pos : 0 < (2 * a - 1) * (2 * s - 1) := mul_pos h2a1 h2s1
  have hden_pos : 0 < 2 * ((2 * a - 1) * s - a + 1) := by linarith
  have hdiff : (2 * a - 1) * (2 * s - 1) - 2 * ((2 * a - 1) * s - a + 1) = -1 := by ring
  have hc_pos : 0 < cAlt a s := by
    unfold cAlt
    exact div_pos hnum_pos hden_pos
  have hc_lt1 : cAlt a s < 1 := by
    unfold cAlt
    rw [div_lt_one hden_pos]
    linarith
  have hdet_rewrite : ∀ c : ℝ,
      ((1 : Matrix (Fin 2) (Fin 2) ℝ) -
        stepMat2 p q b (ε * c) 0 * stepMat2 p q b (ε * c) 2).det =
      ((2 * a - 1) * (2 * s - 1) - 2 * ((2 * a - 1) * s - a + 1) * c) / (a ^ 2 * s ^ 2) := by
    intro c
    rw [(O1_period2_prod_det_and_perron p q b (ε * c)).2.1]
    dsimp [b, p, ε, q]
    field_simp [ha0.ne', hs0.ne']
    ring
  have hc_iff : ∀ c : ℝ,
      ((1 : Matrix (Fin 2) (Fin 2) ℝ) -
        stepMat2 p q b (ε * c) 0 * stepMat2 p q b (ε * c) 2).det = 0 ↔ c = cAlt a s := by
    intro c
    rw [hdet_rewrite c, div_eq_zero_iff]
    have has2 : a ^ 2 * s ^ 2 ≠ 0 := by positivity
    unfold cAlt
    constructor
    · intro h
      have h0 := h.resolve_right has2
      rw [eq_div_iff hden_pos.ne']
      linarith
    · intro hc_eq
      left
      have hmul : c * (2 * ((2 * a - 1) * s - a + 1)) = (2 * a - 1) * (2 * s - 1) :=
        (eq_div_iff hden_pos.ne').mp hc_eq
      linarith
  refine ⟨hdiff, ⟨hc_pos, hc_lt1⟩, hc_iff, ?_, ?_⟩
  · intro D
    rw [hc_iff (Real.exp D)]
    unfold DAlt
    constructor
    · intro h
      have := congr_arg Real.log h
      simpa using this
    · intro h
      rw [h, Real.exp_log hc_pos]
  · unfold DAlt
    exact Real.log_neg hc_pos hc_lt1

/-- **O3 (`a13_proof.md` §1 Proposition 1, `(1.2)`).**
For every `a > 1`:
1. On `s ≥ 1`, `DAlt a` has derivative `1 / [(2s − 1)((2a − 1)s − a + 1)] > 0`;
2. `DAlt a` is strictly increasing on `[1, ∞)` (`StrictMonoOn (DAlt a) (Set.Ici 1)`);
3. As `s → ∞`, `DAlt a s → 0` and `s * DAlt a s → −1 / [2 (2a − 1)]`. -/
theorem O3_period2_deriv_strictMono_and_asymptotic (a : ℝ) (ha : 1 < a) :
    (∀ s : ℝ, 1 ≤ s →
      HasDerivAt (DAlt a) (1 / ((2 * s - 1) * ((2 * a - 1) * s - a + 1))) s ∧
      0 < 1 / ((2 * s - 1) * ((2 * a - 1) * s - a + 1))) ∧
    StrictMonoOn (DAlt a) (Set.Ici 1) ∧
    Filter.Tendsto (DAlt a) Filter.atTop (nhds 0) ∧
    Filter.Tendsto (fun s : ℝ => s * DAlt a s) Filter.atTop (nhds (-1 / (2 * (2 * a - 1)))) := by
  have h2a1 : 0 < 2 * a - 1 := by linarith
  have hrew : ∀ s : ℝ, 1 ≤ s → cAlt a s = 1 - 1 / (2 * ((2 * a - 1) * s - a + 1)) := by
    intro s hs
    have hd : 2 * ((2 * a - 1) * s - a + 1) ≠ 0 := by nlinarith
    unfold cAlt
    rw [eq_sub_iff_add_eq, ← add_div, div_eq_one_iff_eq hd]
    ring
  have hderiv_all : ∀ s : ℝ, 1 ≤ s →
      HasDerivAt (DAlt a) (1 / ((2 * s - 1) * ((2 * a - 1) * s - a + 1))) s ∧
      0 < 1 / ((2 * s - 1) * ((2 * a - 1) * s - a + 1)) := by
    intro s hs
    have h2s1 : 0 < 2 * s - 1 := by linarith
    have hden_in : 0 < (2 * a - 1) * s - a + 1 := by nlinarith
    have hf1 : HasDerivAt (fun x : ℝ => (2 * a - 1) * (2 * x - 1)) ((2 * a - 1) * 2) s := by
      have hlin : HasDerivAt (fun x : ℝ => 2 * x - 1) 2 s := by
        simpa using ((hasDerivAt_id' s).const_mul 2).sub_const 1
      simpa using hlin.const_mul (2 * a - 1)
    have hf2 : HasDerivAt (fun x : ℝ => 2 * ((2 * a - 1) * x - a + 1)) (2 * (2 * a - 1)) s := by
      have hlin : HasDerivAt (fun x : ℝ => (2 * a - 1) * x - a + 1) (2 * a - 1) s := by
        simpa using (((hasDerivAt_id' s).const_mul (2 * a - 1)).sub_const a).add_const 1
      simpa using hlin.const_mul 2
    have hden_pos : 0 < 2 * ((2 * a - 1) * s - a + 1) := by linarith
    have hquot : HasDerivAt (cAlt a)
        (((2 * a - 1) * 2 * (2 * ((2 * a - 1) * s - a + 1)) -
          (2 * a - 1) * (2 * s - 1) * (2 * (2 * a - 1))) /
          (2 * ((2 * a - 1) * s - a + 1)) ^ 2) s :=
      hf1.div hf2 hden_pos.ne'
    have hc_pos : 0 < cAlt a s := (O2_period2_root_and_neg a s ha hs).2.1.1
    have hlog : HasDerivAt (DAlt a)
        ((((2 * a - 1) * 2 * (2 * ((2 * a - 1) * s - a + 1)) -
          (2 * a - 1) * (2 * s - 1) * (2 * (2 * a - 1))) /
          (2 * ((2 * a - 1) * s - a + 1)) ^ 2) / cAlt a s) s :=
      hquot.log hc_pos.ne'
    have heq : (((2 * a - 1) * 2 * (2 * ((2 * a - 1) * s - a + 1)) -
          (2 * a - 1) * (2 * s - 1) * (2 * (2 * a - 1))) /
        (2 * ((2 * a - 1) * s - a + 1)) ^ 2) / cAlt a s =
        1 / ((2 * s - 1) * ((2 * a - 1) * s - a + 1)) := by
      unfold cAlt
      field_simp [h2a1.ne', h2s1.ne', hden_in.ne']
      ring
    rw [heq] at hlog
    exact ⟨hlog, div_pos one_pos (mul_pos h2s1 hden_in)⟩
  have hmono : StrictMonoOn (DAlt a) (Set.Ici 1) := by
    intro s₁ hs₁ s₂ hs₂ hlt
    have hs₁' : 1 ≤ s₁ := hs₁
    have hs₂' : 1 ≤ s₂ := hs₂
    have hc₁_pos : 0 < cAlt a s₁ := (O2_period2_root_and_neg a s₁ ha hs₁').2.1.1
    have hden₁ : 0 < 2 * ((2 * a - 1) * s₁ - a + 1) := by nlinarith
    have hc_lt : cAlt a s₁ < cAlt a s₂ := by
      rw [hrew s₁ hs₁', hrew s₂ hs₂']
      have hden_lt : 2 * ((2 * a - 1) * s₁ - a + 1) < 2 * ((2 * a - 1) * s₂ - a + 1) := by
        nlinarith
      have hinv := one_div_lt_one_div_of_lt hden₁ hden_lt
      linarith
    exact Real.log_lt_log hc₁_pos hc_lt
  -- Limits as `s → ∞`: write `qFun s = 2((2a - 1)s - a + 1) → +∞` and `cAlt a s = 1 - 1 / qFun s`.
  let qFun : ℝ → ℝ := fun s => 2 * ((2 * a - 1) * s - a + 1)
  have hq_atTop : Filter.Tendsto qFun Filter.atTop Filter.atTop := by
    have h1 : Filter.Tendsto (fun s : ℝ => (2 * a - 1) * s) Filter.atTop Filter.atTop :=
      Filter.Tendsto.const_mul_atTop h2a1 Filter.tendsto_id
    have h2 : Filter.Tendsto (fun s : ℝ => (2 * a - 1) * s + (-a + 1)) Filter.atTop Filter.atTop :=
      Filter.Tendsto.atTop_add h1 tendsto_const_nhds
    have h3 : Filter.Tendsto (fun s : ℝ => 2 * ((2 * a - 1) * s + (-a + 1))) Filter.atTop Filter.atTop :=
      Filter.Tendsto.const_mul_atTop (by norm_num : (0 : ℝ) < 2) h2
    refine h3.congr fun s => by dsimp [qFun]; ring
  have hc_tendsto : Filter.Tendsto (cAlt a) Filter.atTop (nhds 1) := by
    have hinv0 : Filter.Tendsto (fun s : ℝ => 1 / qFun s) Filter.atTop (nhds 0) :=
      tendsto_const_nhds.div_atTop hq_atTop
    have h1sub : Filter.Tendsto (fun s : ℝ => 1 - 1 / qFun s) Filter.atTop (nhds (1 - 0)) :=
      tendsto_const_nhds.sub hinv0
    rw [sub_zero] at h1sub
    refine h1sub.congr' ?_
    filter_upwards [Filter.eventually_ge_atTop 1] with s hs
    exact (hrew s hs).symm
  have hD_tendsto : Filter.Tendsto (DAlt a) Filter.atTop (nhds 0) := by
    have hlog := (Real.continuousAt_log one_ne_zero).tendsto.comp hc_tendsto
    rw [Real.log_one] at hlog
    exact hlog
  -- Scaled limit `s * DAlt a s → -1 / [2 (2a - 1)]`:
  have hsD_tendsto :
      Filter.Tendsto (fun s : ℝ => s * DAlt a s) Filter.atTop
        (nhds (-1 / (2 * (2 * a - 1)))) := by
    have hdlog1 : HasDerivAt Real.log 1 1 := by simpa using Real.hasDerivAt_log one_ne_zero
    have hslope : Filter.Tendsto (slope Real.log 1) (nhdsWithin 1 {(1 : ℝ)}ᶜ) (nhds 1) :=
      hasDerivAt_iff_tendsto_slope.mp hdlog1
    have hc_nhdsWithin : Filter.Tendsto (cAlt a) Filter.atTop (nhdsWithin 1 {(1 : ℝ)}ᶜ) := by
      rw [tendsto_nhdsWithin_iff]
      refine ⟨hc_tendsto, ?_⟩
      filter_upwards [Filter.eventually_ge_atTop 1] with s hs
      exact (O2_period2_root_and_neg a s ha hs).2.1.2.ne
    have hlog_div := hslope.comp hc_nhdsWithin
    have hrat : Filter.Tendsto (fun s : ℝ => s * (cAlt a s - 1)) Filter.atTop
        (nhds (-1 / (2 * (2 * a - 1)))) := by
      have hinv_s : Filter.Tendsto (fun s : ℝ => 2 * (1 - a) / s) Filter.atTop (nhds 0) :=
        tendsto_const_nhds.div_atTop Filter.tendsto_id
      have hden_lim : Filter.Tendsto (fun s : ℝ => 2 * (2 * a - 1) + 2 * (1 - a) / s)
          Filter.atTop (nhds (2 * (2 * a - 1) + 0)) :=
        tendsto_const_nhds.add hinv_s
      rw [add_zero] at hden_lim
      have hquot : Filter.Tendsto (fun s : ℝ => (-1 : ℝ) / (2 * (2 * a - 1) + 2 * (1 - a) / s))
          Filter.atTop (nhds (-1 / (2 * (2 * a - 1)))) :=
        tendsto_const_nhds.div hden_lim (by positivity)
      refine hquot.congr' ?_
      filter_upwards [Filter.eventually_ge_atTop 1] with s hs
      have hs0 : 0 < s := lt_of_lt_of_le zero_lt_one hs
      have hd : 2 * ((2 * a - 1) * s - a + 1) ≠ 0 := by nlinarith
      have hd_eq : s * (2 * (2 * a - 1) + 2 * (1 - a) / s) = 2 * ((2 * a - 1) * s - a + 1) := by
        calc s * (2 * (2 * a - 1) + 2 * (1 - a) / s)
            = s * (2 * (2 * a - 1)) + s * (2 * (1 - a) / s) := mul_add _ _ _
          _ = s * (2 * (2 * a - 1)) + 2 * (1 - a) := by rw [mul_div_cancel₀ _ hs0.ne']
          _ = 2 * ((2 * a - 1) * s - a + 1) := by ring
      have hd2 : 2 * (2 * a - 1) + 2 * (1 - a) / s ≠ 0 := by
        intro h0; rw [h0, mul_zero] at hd_eq; exact hd hd_eq.symm
      rw [hrew s hs]
      have hsub1 : 1 - 1 / (2 * ((2 * a - 1) * s - a + 1)) - 1 =
          -1 / (2 * ((2 * a - 1) * s - a + 1)) := by ring
      rw [hsub1, ← hd_eq]
      field_simp [hs0.ne', hd2]
    have hprod := hrat.mul hlog_div
    rw [mul_one] at hprod
    refine hprod.congr' ?_
    filter_upwards [Filter.eventually_ge_atTop 1] with s hs
    have hc_ne1 : cAlt a s - 1 ≠ 0 :=
      sub_ne_zero.mpr (O2_period2_root_and_neg a s ha hs).2.1.2.ne
    dsimp [Function.comp]
    rw [slope_def_field, Real.log_one, sub_zero]
    unfold DAlt
    field_simp [hc_ne1]
  exact ⟨hderiv_all, hmono, hD_tendsto, hsD_tendsto⟩

/-! ## Section 2: Algebraic and Inequality Core of the Proved Part ii Bound (`O4`–`O8`) -/

/-- First standard basis vector `e₀ = (1, 0, ..., 0)ᵀ` in `ℝ^k`. -/
def eFirst (k : ℕ) : Fin k → ℝ :=
  fun i => if (i : ℕ) = 0 then 1 else 0

/-- Last standard basis vector `e_{k-1} = (0, ..., 0, 1)ᵀ` in `ℝ^k`. -/
def eLast (k : ℕ) : Fin k → ℝ :=
  fun i => if (i : ℕ) = k - 1 then 1 else 0

/-- Lower bidiagonal matrix `C` with staying diagonal `(p_0, ..., p_{k-1})` and subdiagonal `1`s. -/
def bidiagC (k : ℕ) (p : Fin k → ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  Matrix.of fun i j =>
    (if i = j then p i else 0) + (if (i : ℕ) = (j : ℕ) + 1 then 1 else 0)

/-- Rank-one host transfer matrix `E_{0, k-1}` (`1` in row `0`, column `k − 1`). -/
def hostRankOne (k : ℕ) : Matrix (Fin k) (Fin k) ℝ :=
  Matrix.of fun i j => if (i : ℕ) = 0 ∧ (j : ℕ) = k - 1 then 1 else 0

/-- Rescaled `k`-stage ring step matrix `A(h) = C + z h e_1 e_kᵀ` (`zh = z * h`). -/
def outageRingMat (k : ℕ) (p : Fin k → ℝ) (zh : ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  bidiagC k p + zh • hostRankOne k

/-- `m`-th Krylov vector `(C ^ m) e₀`. -/
def krylovVec (k : ℕ) (p : Fin k → ℝ) (m : ℕ) : Fin k → ℝ :=
  ((bidiagC k p) ^ m).mulVec (eFirst k)

/-- Krylov matrix `K = [e₀, C e₀, ..., C^{k-1} e₀]`. -/
def krylovMat (k : ℕ) (p : Fin k → ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  Matrix.of fun i j => krylovVec k p (j : ℕ) i

/-- Companion matrix associated with a monic degree-`k` polynomial `χ`:
subdiagonal `1`s and last column `−coeff χ i`. -/
noncomputable def companionOfPoly (k : ℕ) (χ : Polynomial ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  Matrix.of fun i j =>
    (if (i : ℕ) = (j : ℕ) + 1 then 1 else 0) +
      (if (j : ℕ) = k - 1 then -χ.coeff (i : ℕ) else 0)

/-- Ordered matrix product `A(h_{T-1}) ⋯ A(h_0)` (`1` when `T = 0`). -/
def prodAll (k : ℕ) (p : Fin k → ℝ) (z : ℝ) (h : ℕ → ℝ) : ℕ → Matrix (Fin k) (Fin k) ℝ
  | 0 => 1
  | T + 1 => outageRingMat k p (z * h T) * prodAll k p z h T

/-- Ordered positive-step matrix product `Q_T` obtained by deleting steps with `h t = 0`. -/
noncomputable def prodPos (k : ℕ) (p : Fin k → ℝ) (z : ℝ) (h : ℕ → ℝ) : ℕ → Matrix (Fin k) (Fin k) ℝ
  | 0 => 1
  | T + 1 => if h T = 0 then prodPos k p z h T else outageRingMat k p (z * h T) * prodPos k p z h T

/-- Number of zero-host steps `n₀` in `{0, ..., T − 1}`. -/
noncomputable def nZeros (h : ℕ → ℝ) : ℕ → ℕ
  | 0 => 0
  | T + 1 => if h T = 0 then nZeros h T + 1 else nZeros h T

/-- Invariant cone `Ccone = {x ≥ 0 : (C − pmax I) x ≥ 0}` of `a13_proof.md` §6 Step 2. -/
def inOutageCone (k : ℕ) (p : Fin k → ℝ) (pmax : ℝ) (x : Fin k → ℝ) : Prop :=
  (∀ i : Fin k, 0 ≤ x i) ∧ (∀ i : Fin k, pmax * x i ≤ ((bidiagC k p).mulVec x) i)

private theorem bidiagC_mulVec_apply (k : ℕ) (p : Fin k → ℝ) (v : Fin k → ℝ) (i : Fin k) :
    ((bidiagC k p).mulVec v) i =
      if _h0 : (i : ℕ) = 0 then
        p i * v i
      else
        p i * v i + v ⟨(i : ℕ) - 1, by omega⟩ := by
  unfold bidiagC
  simp only [Matrix.mulVec, dotProduct, Matrix.of_apply]
  split_ifs with h0
  · rw [Finset.sum_eq_single i]
    · simp [h0]
    · intro j _ hji
      have h1 : i ≠ j := hji.symm
      have h2 : (i : ℕ) ≠ (j : ℕ) + 1 := by omega
      simp [h1, h2]
    · intro hi; exact False.elim (hi (Finset.mem_univ i))
  · let iprev : Fin k := ⟨(i : ℕ) - 1, by omega⟩
    have hne : i ≠ iprev := by
      intro heq
      have := congr_arg Fin.val heq
      dsimp [iprev] at this
      omega
    rw [Fintype.sum_eq_add i iprev hne]
    · have hip2 : (i : ℕ) = (iprev : ℕ) + 1 := by dsimp [iprev]; omega
      simp [hne, hip2]
    · intro c hc
      have h1 : i ≠ c := hc.1.symm
      have h2 : (i : ℕ) ≠ (c : ℕ) + 1 := by
        intro heq
        apply hc.2
        ext
        dsimp [iprev]
        omega
      simp [h1, h2]

private theorem krylovVec_support_and_diag (k : ℕ) (_hk : 1 ≤ k) (p : Fin k → ℝ) :
    ∀ m : ℕ,
      (∀ i : Fin k, m < (i : ℕ) → krylovVec k p m i = 0) ∧
      (∀ i : Fin k, (i : ℕ) = m → krylovVec k p m i = 1) := by
  intro m
  induction m with
  | zero =>
    unfold krylovVec eFirst
    simp only [pow_zero, Matrix.one_mulVec]
    exact ⟨fun i hi => by simp [Nat.ne_of_gt hi], fun i hi => by simp [hi]⟩
  | succ m ih =>
    have hstep : krylovVec k p (m + 1) = (bidiagC k p).mulVec (krylovVec k p m) := by
      unfold krylovVec
      rw [pow_succ', Matrix.mulVec_mulVec]
    rw [hstep]
    refine ⟨fun i hi => ?_, fun i hi => ?_⟩
    · rw [bidiagC_mulVec_apply]
      have hi0 : (i : ℕ) ≠ 0 := by omega
      simp only [hi0, ↓reduceDIte]
      rw [ih.1 i (by omega), ih.1 ⟨(i : ℕ) - 1, by omega⟩ (by simp; omega)]
      ring
    · rw [bidiagC_mulVec_apply]
      have hi0 : (i : ℕ) ≠ 0 := by omega
      simp only [hi0, ↓reduceDIte]
      rw [ih.1 i (by omega), ih.2 ⟨(i : ℕ) - 1, by omega⟩ (by simp; omega)]
      ring

private theorem bidiagC_charpoly (k : ℕ) (p : Fin k → ℝ) :
    (bidiagC k p).charpoly = ∏ i : Fin k, (Polynomial.X - Polynomial.C (p i)) := by
  have hlow : (bidiagC k p).IsLowerTriangular := by
    intro i j hij
    have hij' : (i : ℕ) < (j : ℕ) := OrderDual.toDual_lt_toDual.mp hij
    unfold bidiagC
    simp only [Matrix.of_apply]
    have h1 : i ≠ j := fun h => by subst h; omega
    have h2 : (i : ℕ) ≠ (j : ℕ) + 1 := by omega
    simp [h1, h2]
  have hupT : (bidiagC k p)ᵀ.IsUpperTriangular := hlow.transpose
  rw [← Matrix.charpoly_transpose, Matrix.charpoly_of_isUpperTriangular _ hupT]
  refine Finset.prod_congr rfl fun i _ => ?_
  have hdiag : (bidiagC k p)ᵀ i i = p i := by
    unfold bidiagC
    simp
  rw [hdiag]

/-- **O4 (`a13_proof.md` §6 Step 1 & dated fix lines 399–404; `a13-check.md` lines 42, 140).**
Simultaneous Krylov companion similarity for reordering stage stays:
1. `krylovMat k p` is **upper triangular** (`IsUpperTriangular`, incorporating the referee's
   directional fix) with unit diagonal, `det = 1`, `IsUnit`, `K⁻¹ e₀ = e₀`, and `e_{k-1}ᵀ K = e_{k-1}ᵀ`;
2. `K⁻¹ * outageRingMat k p zh * K = companionOfPoly k (bidiagC k p).charpoly + zh • hostRankOne k`
   simultaneously for all `zh : ℝ`;
3. For every permutation `σ : Equiv.Perm (Fin k)`, there exists an invertible matrix `S` (`IsUnit S`,
   independent of `z` and `h`) such that `S⁻¹ * outageRingMat k p zh * S = outageRingMat k (p ∘ σ) zh`
   for all `zh : ℝ`, and `S⁻¹ * prodAll k p z h T * S = prodAll k (p ∘ σ) z h T` for every horizon
   `T : ℕ` and host trajectory `h : ℕ → ℝ`. -/
theorem O4_krylov_upperTriangular_and_perm_conj (k : ℕ) (hk : 1 ≤ k) (p : Fin k → ℝ) :
    let K := krylovMat k p
    K.IsUpperTriangular ∧
    (∀ i : Fin k, K i i = 1) ∧
    K.det = 1 ∧
    IsUnit K ∧
    K⁻¹.mulVec (eFirst k) = eFirst k ∧
    K.vecMul (eLast k) = eLast k ∧
    (∀ zh : ℝ,
      K⁻¹ * outageRingMat k p zh * K =
        companionOfPoly k (bidiagC k p).charpoly + zh • hostRankOne k) ∧
    (∀ σ : Equiv.Perm (Fin k),
      ∃ S : Matrix (Fin k) (Fin k) ℝ, IsUnit S ∧
        (∀ zh : ℝ, S⁻¹ * outageRingMat k p zh * S = outageRingMat k (p ∘ σ) zh) ∧
        (∀ (z : ℝ) (h : ℕ → ℝ) (T : ℕ),
          S⁻¹ * prodAll k p z h T * S = prodAll k (p ∘ σ) z h T)) := by
  have hK_up : ∀ p' : Fin k → ℝ, (krylovMat k p').IsUpperTriangular := by
    intro p' i j hij
    exact (krylovVec_support_and_diag k hk p' (j : ℕ)).1 i hij
  have hK_diag : ∀ p' : Fin k → ℝ, ∀ i : Fin k, krylovMat k p' i i = 1 := by
    intro p' i
    exact (krylovVec_support_and_diag k hk p' (i : ℕ)).2 i rfl
  have hK_det : ∀ p' : Fin k → ℝ, (krylovMat k p').det = 1 := by
    intro p'
    rw [Matrix.det_of_isUpperTriangular (hK_up p')]
    simp [hK_diag p']
  have hK_unit : ∀ p' : Fin k → ℝ, IsUnit (krylovMat k p') := by
    intro p'
    rw [Matrix.isUnit_iff_isUnit_det, hK_det p']
    exact isUnit_one
  have hKe0 : ∀ p' : Fin k → ℝ, (krylovMat k p').mulVec (eFirst k) = eFirst k := by
    intro p'
    ext i
    unfold krylovMat eFirst
    simp only [Matrix.mulVec, dotProduct, Matrix.of_apply]
    let z0 : Fin k := ⟨0, by omega⟩
    rw [Finset.sum_eq_single z0]
    · have hz0 : krylovVec k p' 0 = eFirst k := by
        unfold krylovVec; simp
      change krylovVec k p' (z0 : ℕ) i * (if (z0 : ℕ) = 0 then 1 else 0) = _
      simp only [z0, ↓reduceIte, mul_one]
      rw [hz0]; rfl
    · intro j _ hj
      have hj0 : (j : ℕ) ≠ 0 := fun h => hj (Fin.ext h)
      simp [hj0]
    · intro hz; exact False.elim (hz (Finset.mem_univ z0))
  have hKinve0 : ∀ p' : Fin k → ℝ, (krylovMat k p')⁻¹.mulVec (eFirst k) = eFirst k := by
    intro p'
    have hdetu : IsUnit (krylovMat k p').det := by rw [hK_det p']; exact isUnit_one
    calc (krylovMat k p')⁻¹.mulVec (eFirst k)
        = (krylovMat k p')⁻¹.mulVec ((krylovMat k p').mulVec (eFirst k)) := by rw [hKe0 p']
      _ = ((krylovMat k p')⁻¹ * krylovMat k p').mulVec (eFirst k) := by rw [Matrix.mulVec_mulVec]
      _ = (1 : Matrix (Fin k) (Fin k) ℝ).mulVec (eFirst k) := by
          rw [Matrix.nonsing_inv_mul _ hdetu]
      _ = eFirst k := Matrix.one_mulVec _
  have heLastK : ∀ p' : Fin k → ℝ, (krylovMat k p').vecMul (eLast k) = eLast k := by
    intro p'
    ext j
    unfold krylovMat eLast
    simp only [Matrix.vecMul, dotProduct, Matrix.of_apply]
    let zlast : Fin k := ⟨k - 1, by omega⟩
    rw [Finset.sum_eq_single zlast]
    · change (if (zlast : ℕ) = k - 1 then 1 else 0) * krylovVec k p' (j : ℕ) zlast = _
      simp only [zlast, ↓reduceIte, one_mul]
      by_cases hj : (j : ℕ) = k - 1
      · simp only [hj, ↓reduceIte]
        exact (krylovVec_support_and_diag k hk p' (k - 1)).2 zlast (by simp [zlast])
      · simp only [hj, ↓reduceIte]
        have hjlt : (j : ℕ) < (zlast : ℕ) := by dsimp [zlast]; omega
        exact (krylovVec_support_and_diag k hk p' (j : ℕ)).1 zlast hjlt
    · intro i _ hi
      have hi_ne : (i : ℕ) ≠ k - 1 := fun h => hi (Fin.ext h)
      simp [hi_ne]
    · intro hz; exact False.elim (hz (Finset.mem_univ zlast))
  -- Companion conjugation for any `p' : Fin k → ℝ`:
  have hcomp_all : ∀ (p' : Fin k → ℝ) (zh : ℝ),
      (krylovMat k p')⁻¹ * outageRingMat k p' zh * krylovMat k p' =
        companionOfPoly k (bidiagC k p').charpoly + zh • hostRankOne k := by
    intro p' zh
    let K' := krylovMat k p'
    let C' := bidiagC k p'
    let χ := C'.charpoly
    have hdetu : IsUnit K'.det := by dsimp [K']; rw [hK_det p']; exact isUnit_one
    have hCK : C' * K' = K' * companionOfPoly k χ := by
      have hχ_eq : χ = ∏ i : Fin k, (Polynomial.X - Polynomial.C (p' i)) :=
        bidiagC_charpoly k p'
      have hmonic : χ.Monic := by
        rw [hχ_eq]
        exact Polynomial.monic_prod_of_monic _ _ fun i _ => Polynomial.monic_X_sub_C (p' i)
      have hdeg : χ.natDegree = k := by
        rw [hχ_eq, Polynomial.natDegree_prod_of_monic _ _ fun i _ => Polynomial.monic_X_sub_C (p' i)]
        simp
      have hpoly_sum : (∑ r ∈ Finset.range (k + 1), Polynomial.C (χ.coeff r) * Polynomial.X ^ r) = χ := by
        have hs := Polynomial.as_sum_range_C_mul_X_pow χ
        rw [hdeg] at hs
        exact hs.symm
      have hlead : χ.coeff k = 1 := by
        rw [← hdeg]
        exact hmonic.leadingCoeff
      have haeval0 : Polynomial.aeval C' χ = 0 := Matrix.aeval_self_charpoly C'
      have hCk_sum : C' ^ k = -∑ r : Fin k, χ.coeff (r : ℕ) • (C' ^ (r : ℕ)) := by
        rw [← hpoly_sum, Finset.sum_range_succ, hlead, map_one, one_mul, map_add, map_sum,
          Polynomial.aeval_X_pow] at haeval0
        have hterm_eval : ∀ r : ℕ,
            (Polynomial.aeval C') (Polynomial.C (χ.coeff r) * Polynomial.X ^ r) = χ.coeff r • (C' ^ r) := by
          intro r
          rw [map_mul, Polynomial.aeval_C, Polynomial.aeval_X_pow,
            Algebra.algebraMap_eq_smul_one, smul_mul_assoc, one_mul]
        simp_rw [hterm_eval] at haeval0
        have hfin : ∑ r ∈ Finset.range k, χ.coeff r • (C' ^ r) =
            ∑ r : Fin k, χ.coeff (r : ℕ) • (C' ^ (r : ℕ)) :=
          (Fin.sum_univ_eq_sum_range (fun r => χ.coeff r • (C' ^ r)) k).symm
        rw [hfin] at haeval0
        exact eq_neg_of_add_eq_zero_right haeval0
      ext i j
      have hlhs : (C' * K') i j = krylovVec k p' ((j : ℕ) + 1) i := by
        dsimp [K', C']
        unfold krylovMat krylovVec
        rw [pow_succ', ← Matrix.mulVec_mulVec]
        simp [Matrix.mul_apply, Matrix.mulVec, dotProduct]
      rw [hlhs]
      simp only [K', krylovMat, companionOfPoly, Matrix.mul_apply, Matrix.of_apply]
      by_cases hj : (j : ℕ) = k - 1
      · have hrhs : ∑ r : Fin k, krylovVec k p' (r : ℕ) i *
            ((if (r : ℕ) = (j : ℕ) + 1 then 1 else 0) +
              (if (j : ℕ) = k - 1 then -χ.coeff (r : ℕ) else 0)) =
            -∑ r : Fin k, χ.coeff (r : ℕ) * krylovVec k p' (r : ℕ) i := by
          rw [← Finset.sum_neg_distrib]
          refine Finset.sum_congr rfl fun r _ => ?_
          have hrk : (r : ℕ) ≠ (j : ℕ) + 1 := by omega
          rw [ite_eq_right_iff.mpr fun h => (hrk h).elim, ite_eq_left_iff.mpr fun h => (h hj).elim]
          ring
        rw [hrhs]
        have hj1 : (j : ℕ) + 1 = k := by omega
        rw [hj1]
        unfold krylovVec
        change ((C' ^ k).mulVec (eFirst k)) i = _
        rw [hCk_sum, Matrix.neg_mulVec, Pi.neg_apply]
        congr 1
        simp only [C', Matrix.mulVec, dotProduct, Matrix.sum_apply, Matrix.smul_apply,
          smul_eq_mul, Finset.sum_mul]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun r _ => ?_
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun s _ => by ring
      · let jnext : Fin k := ⟨(j : ℕ) + 1, by omega⟩
        rw [Finset.sum_eq_single jnext]
        · simp [jnext, hj]
        · intro r _ hr
          have hr1 : (r : ℕ) ≠ (j : ℕ) + 1 := fun h => hr (Fin.ext h)
          simp [hr1, hj]
        · intro hmem; exact False.elim (hmem (Finset.mem_univ jnext))
    have hHK : hostRankOne k * K' = K' * hostRankOne k := by
      have hleft : hostRankOne k * K' = hostRankOne k := by
        ext i j
        have hrow := congr_fun (heLastK p') j
        dsimp [krylovMat, eLast, Matrix.vecMul, dotProduct] at hrow
        dsimp [K', krylovMat]
        unfold hostRankOne
        simp only [Matrix.mul_apply, Matrix.of_apply]
        by_cases hi0 : (i : ℕ) = 0
        · simp only [hi0, true_and]
          rw [← hrow]
        · simp [hi0]
      have hright : K' * hostRankOne k = hostRankOne k := by
        ext i j
        have hcol := congr_fun (hKe0 p') i
        dsimp [krylovMat, eFirst, Matrix.mulVec, dotProduct] at hcol
        dsimp [K', krylovMat]
        unfold hostRankOne
        simp only [Matrix.mul_apply, Matrix.of_apply]
        by_cases hjk : (j : ℕ) = k - 1
        · simp only [hjk, and_true]
          rw [← hcol]
        · simp [hjk]
      rw [hleft, hright]
    -- Combine into `outageRingMat k p' zh * K' = K' * (companion + zh • hostRankOne)`
    have hAK : outageRingMat k p' zh * K' =
        K' * (companionOfPoly k χ + zh • hostRankOne k) := by
      unfold outageRingMat
      rw [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul, hCK, hHK]
    calc K'⁻¹ * outageRingMat k p' zh * K'
        = K'⁻¹ * (outageRingMat k p' zh * K') := Matrix.mul_assoc _ _ _
      _ = K'⁻¹ * (K' * (companionOfPoly k χ + zh • hostRankOne k)) := by rw [hAK]
      _ = (K'⁻¹ * K') * (companionOfPoly k χ + zh • hostRankOne k) :=
          (Matrix.mul_assoc _ _ _).symm
      _ = companionOfPoly k χ + zh • hostRankOne k := by
          rw [Matrix.nonsing_inv_mul _ hdetu, Matrix.one_mul]
  refine ⟨hK_up p, hK_diag p, hK_det p, hK_unit p, hKinve0 p, heLastK p, hcomp_all p, ?_⟩
  intro σ
  -- Put `S = K(p) * (K(p ∘ σ))⁻¹`, which has `det S = 1`!
  let K₁ := krylovMat k p
  let K₂ := krylovMat k (p ∘ σ)
  let S := K₁ * K₂⁻¹
  have hdet1 : IsUnit K₁.det := by dsimp [K₁]; rw [hK_det p]; exact isUnit_one
  have hdet2 : IsUnit K₂.det := by dsimp [K₂]; rw [hK_det (p ∘ σ)]; exact isUnit_one
  have hK2inv_det : K₂⁻¹.det = 1 := by
    rw [Matrix.det_nonsing_inv, hK_det (p ∘ σ)]
    simp
  have hS_det : S.det = 1 := by
    dsimp [S, K₁]
    rw [Matrix.det_mul, hK_det p, hK2inv_det, mul_one]
  have hS_unit : IsUnit S := by
    rw [Matrix.isUnit_iff_isUnit_det, hS_det]
    exact isUnit_one
  have hS_detu : IsUnit S.det := by rw [hS_det]; exact isUnit_one
  -- Note `S⁻¹ = K₂ * K₁⁻¹`!
  have hSinv : S⁻¹ = K₂ * K₁⁻¹ := by
    apply Matrix.inv_eq_left_inv
    dsimp [S]
    calc (K₂ * K₁⁻¹) * (K₁ * K₂⁻¹)
        = K₂ * ((K₁⁻¹ * K₁) * K₂⁻¹) := by simp [Matrix.mul_assoc]
      _ = K₂ * K₂⁻¹ := by rw [Matrix.nonsing_inv_mul _ hdet1, Matrix.one_mul]
      _ = 1 := Matrix.mul_nonsing_inv _ hdet2
  have hchar_eq : (bidiagC k p).charpoly = (bidiagC k (p ∘ σ)).charpoly := by
    rw [bidiagC_charpoly k p, bidiagC_charpoly k (p ∘ σ)]
    exact (Equiv.prod_comp σ fun x => Polynomial.X - Polynomial.C (p x)).symm
  have hstep_conj : ∀ zh : ℝ, S⁻¹ * outageRingMat k p zh * S = outageRingMat k (p ∘ σ) zh := by
    intro zh
    rw [hSinv]
    dsimp [S, K₁, K₂]
    calc (krylovMat k (p ∘ σ) * (krylovMat k p)⁻¹) * outageRingMat k p zh *
            (krylovMat k p * (krylovMat k (p ∘ σ))⁻¹)
        = krylovMat k (p ∘ σ) *
            ((krylovMat k p)⁻¹ * outageRingMat k p zh * krylovMat k p) *
            (krylovMat k (p ∘ σ))⁻¹ := by simp [Matrix.mul_assoc]
      _ = krylovMat k (p ∘ σ) *
            ((krylovMat k (p ∘ σ))⁻¹ * outageRingMat k (p ∘ σ) zh * krylovMat k (p ∘ σ)) *
            (krylovMat k (p ∘ σ))⁻¹ := by
          rw [hcomp_all p zh, hcomp_all (p ∘ σ) zh, hchar_eq]
      _ = (krylovMat k (p ∘ σ) * (krylovMat k (p ∘ σ))⁻¹) *
            outageRingMat k (p ∘ σ) zh *
            (krylovMat k (p ∘ σ) * (krylovMat k (p ∘ σ))⁻¹) := by
          simp [Matrix.mul_assoc]
      _ = outageRingMat k (p ∘ σ) zh := by
          rw [Matrix.mul_nonsing_inv _ hdet2, Matrix.one_mul, Matrix.mul_one]
  refine ⟨S, hS_unit, hstep_conj, ?_⟩
  intro z h T
  induction T with
  | zero =>
    simp [prodAll, Matrix.nonsing_inv_mul _ hS_detu]
  | succ t ih =>
    simp only [prodAll]
    calc S⁻¹ * (outageRingMat k p (z * h t) * prodAll k p z h t) * S
        = (S⁻¹ * outageRingMat k p (z * h t) * S) * (S⁻¹ * prodAll k p z h t * S) := by
          calc S⁻¹ * (outageRingMat k p (z * h t) * prodAll k p z h t) * S
              = (S⁻¹ * outageRingMat k p (z * h t)) * ((S * S⁻¹) * prodAll k p z h t * S) := by
                rw [Matrix.mul_nonsing_inv _ hS_detu, Matrix.one_mul]
                simp [Matrix.mul_assoc]
            _ = (S⁻¹ * outageRingMat k p (z * h t) * S) * (S⁻¹ * prodAll k p z h t * S) := by
                simp [Matrix.mul_assoc]
      _ = outageRingMat k (p ∘ σ) (z * h t) * prodAll k (p ∘ σ) z h t := by
          rw [hstep_conj (z * h t), ih]

/-- **O5 (`a13_proof.md` §6 Step 2, `(6.3)`).**
Invariant cone `Ccone`, zero-deletion matrix-product inequality `(6.3)`, and `(k − 1)`-step
positivity under `p_0 = pmax`:
1. `e₀ ∈ Ccone`, and for all `zh ≥ 0`, `outageRingMat k p zh` maps `Ccone` into `Ccone`;
2. For every `z ≥ 0`, horizon `T : ℕ`, and nonnegative host sequence `h` (`∀ t < T, 0 ≤ h t`),
   `prodPos k p z h T e₀ ∈ Ccone` and entrywise
   `pmax ^ (nZeros h T) * (prodPos k p z h T e₀)_i ≤ (prodAll k p z h T e₀)_i`;
3. When `0 < pmax`, for every `i : Fin k`, `0 < pmax ^ (k − 1 − i)` and
   `pmax ^ (k − 1 − i) ≤ ((bidiagC k p ^ (k − 1)) e₀)_i`. -/
theorem O5_invariant_cone_and_zero_deletion (k : ℕ) (hk : 1 ≤ k) (p : Fin k → ℝ)
    (hp_nn : ∀ j : Fin k, 0 ≤ p j) (pmax : ℝ) (hpmax_nn : 0 ≤ pmax)
    (hp0 : p ⟨0, by omega⟩ = pmax) (_hp_le : ∀ j : Fin k, p j ≤ pmax) :
    inOutageCone k p pmax (eFirst k) ∧
    (∀ (zh : ℝ), 0 ≤ zh → ∀ x : Fin k → ℝ,
      inOutageCone k p pmax x → inOutageCone k p pmax ((outageRingMat k p zh).mulVec x)) ∧
    (∀ (z : ℝ), 0 ≤ z → ∀ (h : ℕ → ℝ) (T : ℕ), (∀ t : ℕ, t < T → 0 ≤ h t) →
      inOutageCone k p pmax ((prodPos k p z h T).mulVec (eFirst k)) ∧
      ∀ i : Fin k,
        pmax ^ (nZeros h T) * (((prodPos k p z h T).mulVec (eFirst k)) i) ≤
          (((prodAll k p z h T).mulVec (eFirst k)) i)) ∧
    (0 < pmax → ∀ i : Fin k,
      0 < pmax ^ (k - 1 - (i : ℕ)) ∧
      pmax ^ (k - 1 - (i : ℕ)) ≤ (((bidiagC k p) ^ (k - 1)).mulVec (eFirst k)) i) := by
  have hC_nn : ∀ i j : Fin k, 0 ≤ bidiagC k p i j := by
    intro i j
    unfold bidiagC
    simp only [Matrix.of_apply]
    have h1 : 0 ≤ if i = j then p i else 0 := by split_ifs <;> [exact hp_nn i; exact le_rfl]
    have h2 : 0 ≤ if (i : ℕ) = (j : ℕ) + 1 then (1 : ℝ) else 0 := by split_ifs <;> norm_num
    linarith
  have hA_nn : ∀ zh : ℝ, 0 ≤ zh → ∀ i j : Fin k, 0 ≤ outageRingMat k p zh i j := by
    intro zh hzh i j
    unfold outageRingMat hostRankOne
    simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul]
    have h1 := hC_nn i j
    have h2 : 0 ≤ zh * (if (i : ℕ) = 0 ∧ (j : ℕ) = k - 1 then 1 else 0) := by
      split_ifs <;> [linarith; simp]
    linarith
  have hmat_mono : ∀ (M : Matrix (Fin k) (Fin k) ℝ), (∀ i j, 0 ≤ M i j) →
      ∀ u v : Fin k → ℝ, (∀ j, u j ≤ v j) → ∀ i, (M.mulVec u) i ≤ (M.mulVec v) i := by
    intro M hM u v huv i
    simp only [Matrix.mulVec, dotProduct]
    exact Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (huv j) (hM i j)
  -- Part 1a: `eFirst k ∈ inOutageCone`
  have he0_cone : inOutageCone k p pmax (eFirst k) := by
    refine ⟨fun i => by unfold eFirst; split_ifs <;> norm_num, fun i => ?_⟩
    rw [bidiagC_mulVec_apply]
    split_ifs with hi0
    · have hi_eq : i = ⟨0, by omega⟩ := Fin.ext hi0
      rw [hi_eq, hp0]
    · unfold eFirst
      simp only [hi0, ↓reduceIte, mul_zero]
      have hnn : 0 ≤ if (i : ℕ) - 1 = 0 then (1 : ℝ) else 0 := by split_ifs <;> norm_num
      linarith
  -- Part 1b: `outageRingMat k p zh` preserves `inOutageCone`
  have hA_cone : ∀ (zh : ℝ), 0 ≤ zh → ∀ x : Fin k → ℝ,
      inOutageCone k p pmax x → inOutageCone k p pmax ((outageRingMat k p zh).mulVec x) := by
    intro zh hzh x ⟨hx_nn, hx_cone⟩
    have hAx_nn : ∀ i : Fin k, 0 ≤ ((outageRingMat k p zh).mulVec x) i := by
      intro i
      have h0 : (0 : ℝ) = ((outageRingMat k p zh).mulVec 0) i := by simp
      rw [h0]
      exact hmat_mono _ (hA_nn zh hzh) 0 x hx_nn i
    refine ⟨hAx_nn, fun i => ?_⟩
    -- `(A x) = C x + (zh * x(k-1)) • e₀`
    have hAx_split : (outageRingMat k p zh).mulVec x =
        (bidiagC k p).mulVec x + (zh * x ⟨k - 1, by omega⟩) • (eFirst k) := by
      ext r
      unfold outageRingMat hostRankOne eFirst
      simp only [Matrix.add_mulVec, Matrix.smul_mulVec, Pi.add_apply, Pi.smul_apply,
        smul_eq_mul]
      congr 1
      simp only [Matrix.mulVec, dotProduct, Matrix.of_apply]
      by_cases hr0 : (r : ℕ) = 0
      · simp only [hr0, true_and, ↓reduceIte, mul_one]
        let zlast : Fin k := ⟨k - 1, by omega⟩
        rw [Finset.sum_eq_single zlast]
        · simp [zlast]
        · intro j _ hj
          have hjn : (j : ℕ) ≠ k - 1 := fun h => hj (Fin.ext h)
          simp [hjn]
        · intro hz; exact False.elim (hz (Finset.mem_univ zlast))
      · simp [hr0]
    rw [hAx_split]
    simp only [Matrix.mulVec_add, Matrix.mulVec_smul, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    have h1 : pmax * ((bidiagC k p).mulVec x) i ≤
        ((bidiagC k p).mulVec ((bidiagC k p).mulVec x)) i := by
      have hscale : pmax * ((bidiagC k p).mulVec x) i =
          ((bidiagC k p).mulVec (pmax • x)) i := by
        rw [Matrix.mulVec_smul, Pi.smul_apply, smul_eq_mul]
      rw [hscale]
      exact hmat_mono (bidiagC k p) hC_nn (pmax • x) ((bidiagC k p).mulVec x) hx_cone i
    have hcoef_nn : 0 ≤ zh * x ⟨k - 1, by omega⟩ := mul_nonneg hzh (hx_nn _)
    have h2 : pmax * ((zh * x ⟨k - 1, by omega⟩) * eFirst k i) ≤
        (zh * x ⟨k - 1, by omega⟩) * ((bidiagC k p).mulVec (eFirst k)) i := by
      have he0i := he0_cone.2 i
      have hmul := mul_le_mul_of_nonneg_left he0i hcoef_nn
      linarith
    linarith
  -- Part 2: Zero-deletion induction `(6.3)`
  have hdel : ∀ (z : ℝ), 0 ≤ z → ∀ (h : ℕ → ℝ) (T : ℕ), (∀ t : ℕ, t < T → 0 ≤ h t) →
      inOutageCone k p pmax ((prodPos k p z h T).mulVec (eFirst k)) ∧
      ∀ i : Fin k,
        pmax ^ (nZeros h T) * (((prodPos k p z h T).mulVec (eFirst k)) i) ≤
          (((prodAll k p z h T).mulVec (eFirst k)) i) := by
    intro z hz h T
    induction T with
    | zero =>
      intro _
      simp only [prodPos, prodAll, nZeros, pow_zero, one_mul, Matrix.one_mulVec]
      exact ⟨he0_cone, fun _ => le_rfl⟩
    | succ t ih =>
      intro h_nn
      obtain ⟨hcone_t, hle_t⟩ := ih (fun s hs => h_nn s (by omega))
      have hht : 0 ≤ h t := h_nn t (by omega)
      by_cases hzero : h t = 0
      · simp only [prodPos, prodAll, nZeros, hzero, ↓reduceIte, mul_zero]
        have hA0 : outageRingMat k p 0 = bidiagC k p := by
          unfold outageRingMat; simp
        refine ⟨hcone_t, fun i => ?_⟩
        rw [hA0, ← Matrix.mulVec_mulVec]
        have hmono_step := hmat_mono (bidiagC k p) hC_nn
          ((pmax ^ nZeros h t) • ((prodPos k p z h t).mulVec (eFirst k)))
          ((prodAll k p z h t).mulVec (eFirst k)) hle_t i
        rw [Matrix.mulVec_smul, Pi.smul_apply, smul_eq_mul] at hmono_step
        have hcone_step : pmax ^ (nZeros h t + 1) * (((prodPos k p z h t).mulVec (eFirst k)) i) ≤
            pmax ^ nZeros h t * ((bidiagC k p).mulVec ((prodPos k p z h t).mulVec (eFirst k))) i := by
          rw [pow_succ]
          have hc := hcone_t.2 i
          have hp_pow : 0 ≤ pmax ^ nZeros h t := pow_nonneg hpmax_nn _
          have hm := mul_le_mul_of_nonneg_left hc hp_pow
          linarith
        exact hcone_step.trans hmono_step
      · simp only [prodPos, prodAll, nZeros, hzero, ↓reduceIte]
        have hzh : 0 ≤ z * h t := mul_nonneg hz hht
        rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
        refine ⟨hA_cone (z * h t) hzh _ hcone_t, fun i => ?_⟩
        have hmono_step := hmat_mono (outageRingMat k p (z * h t)) (hA_nn (z * h t) hzh)
          ((pmax ^ nZeros h t) • ((prodPos k p z h t).mulVec (eFirst k)))
          ((prodAll k p z h t).mulVec (eFirst k)) hle_t i
        simpa [Matrix.mulVec_smul, Pi.smul_apply, smul_eq_mul] using hmono_step
  -- Part 3: `(k - 1)`-step positivity `(C ^ m e₀)_i ≥ pmax ^ (m - i)` for `(i : ℕ) ≤ m`
  have hpow_lb : ∀ m : ℕ, ∀ i : Fin k, (i : ℕ) ≤ m →
      pmax ^ (m - (i : ℕ)) ≤ (((bidiagC k p) ^ m).mulVec (eFirst k)) i := by
    intro m
    induction m with
    | zero =>
      intro i hi
      have hi0 : (i : ℕ) = 0 := by omega
      simp [hi0, eFirst]
    | succ m ih =>
      -- Note `(C ^ m e₀) ≥ 0` entrywise
      have hvec_nn : ∀ r : Fin k, 0 ≤ (((bidiagC k p) ^ m).mulVec (eFirst k)) r := by
        clear ih
        induction m with
        | zero =>
          intro r
          simp only [pow_zero, Matrix.one_mulVec]
          unfold eFirst
          split_ifs <;> norm_num
        | succ s ihs =>
          intro r
          rw [pow_succ', ← Matrix.mulVec_mulVec]
          have h0 : (0 : ℝ) = ((bidiagC k p).mulVec 0) r := by simp
          rw [h0]
          exact hmat_mono (bidiagC k p) hC_nn 0 _ ihs r
      intro i hi
      rw [pow_succ', ← Matrix.mulVec_mulVec, bidiagC_mulVec_apply]
      split_ifs with hi0
      · have hi_eq : i = ⟨0, by omega⟩ := Fin.ext hi0
        have hp_i : p i = pmax := by rw [hi_eq, hp0]
        have hih0 := ih i (by omega)
        simp only [hi0, Nat.sub_zero] at hih0 ⊢
        rw [hp_i, pow_succ']
        exact mul_le_mul_of_nonneg_left hih0 hpmax_nn
      · have hiprev_le : ((⟨(i : ℕ) - 1, by omega⟩ : Fin k) : ℕ) ≤ m := by simp; omega
        have hih_prev := ih ⟨(i : ℕ) - 1, by omega⟩ hiprev_le
        have hexp_eq : m - ((⟨(i : ℕ) - 1, by omega⟩ : Fin k) : ℕ) = m + 1 - (i : ℕ) := by
          simp; omega
        rw [hexp_eq] at hih_prev
        have hterm1_nn : 0 ≤ p i * (((bidiagC k p) ^ m).mulVec (eFirst k)) i :=
          mul_nonneg (hp_nn i) (hvec_nn i)
        linarith
  refine ⟨he0_cone, hA_cone, hdel, fun hpmax_pos i => ?_⟩
  exact ⟨pow_pos hpmax_pos _, hpow_lb (k - 1) i (by omega)⟩

/-- Recursive binary edge-choice word of length `m` (`2^m` choices; at `k = 1`, `false` and `true`
distinguish the staying self-edge from the host-transfer self-edge). -/
def PathChoice : ℕ → Type
  | 0 => Unit
  | m + 1 => PathChoice m × Bool

instance instFintypePathChoice : ∀ m : ℕ, Fintype (PathChoice m)
  | 0 => inferInstanceAs (Fintype Unit)
  | m + 1 => @instFintypeProd (PathChoice m) Bool (instFintypePathChoice m) inferInstance

/-- Target stage of a labelled outgoing edge (`false` = stay, `true` = advance / host transfer). -/
def nextStage (k : ℕ) (hk : 1 ≤ k) (j : Fin k) (β : Bool) : Fin k :=
  if β then
    if h : (j : ℕ) + 1 < k then ⟨(j : ℕ) + 1, h⟩ else ⟨0, by omega⟩
  else j

/-- Weight of a labelled outgoing edge (`p j` for `false`; `1` or `zh` for `true`). -/
def stepWeight (k : ℕ) (p : Fin k → ℝ) (zh : ℝ) (j : Fin k) (β : Bool) : ℝ :=
  if β then
    if (j : ℕ) + 1 < k then 1 else zh
  else p j

/-- Indicator of the labelled host-transfer edge (`β = true` from stage `k − 1` to stage `0`). -/
def isHostEdge (k : ℕ) (j : Fin k) (β : Bool) : Bool :=
  β && decide ((j : ℕ) = k - 1)

/-- Ending stage of an `m`-step labelled path `(j₀, γ)`. -/
def pathEnd (k : ℕ) (hk : 1 ≤ k) : ∀ m : ℕ, Fin k → PathChoice m → Fin k
  | 0, j₀, _ => j₀
  | m + 1, j₀, γ => nextStage k hk (pathEnd k hk m j₀ γ.1) γ.2

/-- Product of edge weights along an `m`-step labelled path `(j₀, γ)` under sequence `H`. -/
def pathProdWeight (k : ℕ) (hk : 1 ≤ k) (p : Fin k → ℝ) (z : ℝ) (H : ℕ → ℝ) :
    ∀ m : ℕ, Fin k → PathChoice m → ℝ
  | 0, _, _ => 1
  | m + 1, j₀, γ =>
      pathProdWeight k hk p z H m j₀ γ.1 *
        stepWeight k p (z * H m) (pathEnd k hk m j₀ γ.1) γ.2

/-- Product of `(H t / Gplus)` over host-edge traversals along `(j₀, γ)`. -/
noncomputable def pathHostRatio (k : ℕ) (hk : 1 ≤ k) (Gplus : ℝ) (H : ℕ → ℝ) :
    ∀ m : ℕ, Fin k → PathChoice m → ℝ
  | 0, _, _ => 1
  | m + 1, j₀, γ =>
      pathHostRatio k hk Gplus H m j₀ γ.1 *
        (if isHostEdge k (pathEnd k hk m j₀ γ.1) γ.2 then H m / Gplus else 1)

/-- Indicator (`0` or `1`) of whether step `t` of `(j₀, γ)` traverses the host edge. -/
def pathUsesHostAt (k : ℕ) (hk : 1 ≤ k) : ∀ m : ℕ, Fin k → PathChoice m → ℕ → ℝ
  | 0, _, _, _ => 0
  | m + 1, j₀, γ, t =>
      if t = m then
        if isHostEdge k (pathEnd k hk m j₀ γ.1) γ.2 then 1 else 0
      else
        pathUsesHostAt k hk m j₀ γ.1 t

/-- Normalized `m`-step Perron reference path probability on `Fin k × PathChoice m`. -/
noncomputable def refPathProb (k : ℕ) (hk : 1 ≤ k) (p : Fin k → ℝ) (z Gplus lam : ℝ)
    (w v : Fin k → ℝ) (m : ℕ) (ω : Fin k × PathChoice m) : ℝ :=
  lam ^ (-(m : ℤ)) * w (pathEnd k hk m ω.1 ω.2) *
    pathProdWeight k hk p z (fun _ => Gplus) m ω.1 ω.2 * v ω.1

private theorem outageRingMat_apply_eq_sum_bool (k : ℕ) (hk : 1 ≤ k) (p : Fin k → ℝ)
    (zh : ℝ) (i j : Fin k) :
    outageRingMat k p zh i j =
      ∑ β : Bool, if nextStage k hk j β = i then stepWeight k p zh j β else 0 := by
  rw [Fintype.sum_bool]
  unfold outageRingMat bidiagC hostRankOne nextStage stepWeight
  simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul,
    Bool.false_eq_true, ↓reduceIte]
  have hdiag : (if j = i then p j else 0) = (if i = j then p i else 0) := by
    by_cases hij : i = j
    · subst hij; simp
    · have hji : j ≠ i := fun h => hij h.symm
      simp [hij, hji]
  rw [hdiag]
  by_cases hjlt : (j : ℕ) + 1 < k
  · have hj_ne : (j : ℕ) ≠ k - 1 := by omega
    have hnot_host : ¬((i : ℕ) = 0 ∧ (j : ℕ) = k - 1) := fun h => hj_ne h.2
    simp only [hjlt, hnot_host, ↓reduceDIte, ↓reduceIte]
    have heq_idx : (⟨(j : ℕ) + 1, hjlt⟩ : Fin k) = i ↔ (i : ℕ) = (j : ℕ) + 1 := by
      constructor
      · intro h; rw [← h]
      · intro h; ext; exact h.symm
    by_cases hi_succ : (i : ℕ) = (j : ℕ) + 1
    · have hidx_true : (⟨(j : ℕ) + 1, hjlt⟩ : Fin k) = i := heq_idx.mpr hi_succ
      simp only [hi_succ, hidx_true, ↓reduceIte]; ring
    · have hidx_false : ¬((⟨(j : ℕ) + 1, hjlt⟩ : Fin k) = i) := fun h => hi_succ (heq_idx.mp h)
      simp only [hi_succ, hidx_false, ↓reduceIte]; ring
  · have hj_eq : (j : ℕ) = k - 1 := by omega
    have hi_ne_succ : (i : ℕ) ≠ (j : ℕ) + 1 := by omega
    simp only [hjlt, hi_ne_succ, ↓reduceDIte, ↓reduceIte]
    have heq_zero : (⟨0, by omega⟩ : Fin k) = i ↔ (i : ℕ) = 0 := by
      constructor
      · intro h; rw [← h]
      · intro h; ext; exact h.symm
    by_cases hi0 : (i : ℕ) = 0
    · have hzero_true : (⟨0, by omega⟩ : Fin k) = i := heq_zero.mpr hi0
      simp only [hi0, hj_eq, and_self, hzero_true, ↓reduceIte]; ring
    · have hzero_false : ¬((⟨0, by omega⟩ : Fin k) = i) := fun h => hi0 (heq_zero.mp h)
      simp only [hi0, false_and, hzero_false, ↓reduceIte]; ring

private theorem prodAll_mulVec_eq_path_sum (k : ℕ) (hk : 1 ≤ k) (p : Fin k → ℝ)
    (z : ℝ) (H : ℕ → ℝ) (v : Fin k → ℝ) :
    ∀ (m : ℕ) (i : Fin k),
      ((prodAll k p z H m).mulVec v) i =
        ∑ j₀ : Fin k, ∑ γ : PathChoice m,
          if pathEnd k hk m j₀ γ = i then
            pathProdWeight k hk p z H m j₀ γ * v j₀
          else 0 := by
  intro m
  induction m with
  | zero =>
    intro i
    simp only [prodAll, Matrix.one_mulVec, pathEnd, pathProdWeight, one_mul]
    rw [Finset.sum_eq_single i]
    · change v i = ∑ _ : Unit, if i = i then v i else 0
      simp
    · intro j₀ _ hj₀
      change (∑ _ : Unit, if j₀ = i then v j₀ else 0) = 0
      simp [hj₀]
    · intro hi; exact False.elim (hi (Finset.mem_univ i))
  | succ m ih =>
    intro i
    rw [prodAll, ← Matrix.mulVec_mulVec]
    change ∑ r : Fin k, outageRingMat k p (z * H m) i r * ((prodAll k p z H m).mulVec v) r = _
    have hterm : ∀ r : Fin k,
        outageRingMat k p (z * H m) i r * ((prodAll k p z H m).mulVec v) r =
          ∑ j₀ : Fin k, ∑ b₀ : PathChoice m, ∑ β : Bool,
            if pathEnd k hk m j₀ b₀ = r then
              (if nextStage k hk r β = i then
                stepWeight k p (z * H m) r β * (pathProdWeight k hk p z H m j₀ b₀ * v j₀)
              else 0)
            else 0 := by
      intro r
      rw [outageRingMat_apply_eq_sum_bool k hk p (z * H m) i r, ih r, Finset.sum_mul]
      simp_rw [Finset.mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun j₀ _ => ?_
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun b₀ _ => ?_
      refine Finset.sum_congr rfl fun β _ => ?_
      split_ifs <;> ring
    rw [Finset.sum_congr rfl fun r _ => hterm r]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j₀ _ => ?_
    change (∑ r : Fin k, ∑ b₀ : PathChoice m, ∑ β : Bool, _) =
      ∑ γ : PathChoice m × Bool, _
    rw [Fintype.sum_prod_type, Finset.sum_comm]
    refine Finset.sum_congr rfl fun b₀ _ => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun β _ => ?_
    rw [Finset.sum_eq_single (pathEnd k hk m j₀ b₀)]
    · simp only [↓reduceIte]
      change (if nextStage k hk (pathEnd k hk m j₀ b₀) β = i then
          stepWeight k p (z * H m) (pathEnd k hk m j₀ b₀) β *
            (pathProdWeight k hk p z H m j₀ b₀ * v j₀) else 0) =
        (if nextStage k hk (pathEnd k hk m j₀ b₀) β = i then
          (pathProdWeight k hk p z H m j₀ b₀ *
            stepWeight k p (z * H m) (pathEnd k hk m j₀ b₀) β) * v j₀ else 0)
      split_ifs <;> ring
    · intro r _ hr
      simp [hr.symm]
    · intro hm; exact False.elim (hm (Finset.mem_univ _))

private theorem pathProdWeight_eq_const_mul_ratio (k : ℕ) (hk : 1 ≤ k) (p : Fin k → ℝ)
    (z Gplus : ℝ) (hG : Gplus ≠ 0) (H : ℕ → ℝ) :
    ∀ (m : ℕ) (j₀ : Fin k) (γ : PathChoice m),
      pathProdWeight k hk p z H m j₀ γ =
        pathProdWeight k hk p z (fun _ => Gplus) m j₀ γ *
          pathHostRatio k hk Gplus H m j₀ γ := by
  intro m
  induction m with
  | zero => intro j₀ γ; simp [pathProdWeight, pathHostRatio]
  | succ m ih =>
    intro j₀ γ
    simp only [pathProdWeight, pathHostRatio]
    rw [ih j₀ γ.1]
    set j := pathEnd k hk m j₀ γ.1
    set β := γ.2
    unfold stepWeight isHostEdge
    cases β with
    | false => simp; ring
    | true =>
      simp only [↓reduceIte, Bool.true_and]
      by_cases hjlt : (j : ℕ) + 1 < k
      · have hj_ne : ¬((j : ℕ) = k - 1) := by omega
        simp [hjlt, hj_ne]
      · have hj_eq : (j : ℕ) = k - 1 := by omega
        simp only [hjlt, ↓reduceIte]
        simp only [hj_eq, decide_true, ↓reduceIte]
        field_simp [hG]

theorem O6_reference_path_jensen (k : ℕ) (hk : 1 ≤ k) (p : Fin k → ℝ)
    (hp_nn : ∀ j : Fin k, 0 ≤ p j) (z Gplus lam : ℝ)
    (hz : 0 < z) (hG : 0 < Gplus) (hlam : 0 < lam)
    (w v : Fin k → ℝ) (hw : ∀ i : Fin k, 0 < w i) (hv : ∀ j : Fin k, 0 < v j)
    (hwv : dotProduct w v = 1)
    (hAv : (outageRingMat k p (z * Gplus)).mulVec v = lam • v)
    (hwA : (outageRingMat k p (z * Gplus)).vecMul w = lam • w) :
    let ν := w ⟨0, by omega⟩ * z * Gplus * v ⟨k - 1, by omega⟩ / lam
    (∀ (m : ℕ) (ω : Fin k × PathChoice m),
      0 ≤ refPathProb k hk p z Gplus lam w v m ω) ∧
    (∀ m : ℕ,
      ∑ ω : Fin k × PathChoice m, refPathProb k hk p z Gplus lam w v m ω = 1) ∧
    (∀ (m : ℕ) (H : ℕ → ℝ),
      lam ^ (-(m : ℤ)) * dotProduct w ((prodAll k p z H m).mulVec v) =
        ∑ ω : Fin k × PathChoice m,
          refPathProb k hk p z Gplus lam w v m ω * pathHostRatio k hk Gplus H m ω.1 ω.2) ∧
    (∀ (m : ℕ) (H : ℕ → ℝ), (∀ t : ℕ, t < m → 0 < H t) →
      0 < dotProduct w ((prodAll k p z H m).mulVec v) ∧
      (m : ℝ) * Real.log lam + ν * (∑ t ∈ Finset.range m, Real.log (H t / Gplus)) ≤
        Real.log (dotProduct w ((prodAll k p z H m).mulVec v))) := by
  intro ν
  have hsw_nn : ∀ (j : Fin k) (β : Bool), 0 ≤ stepWeight k p (z * Gplus) j β := by
    intro j β
    unfold stepWeight
    cases β with
    | false => exact hp_nn j
    | true =>
      simp only [↓reduceIte]
      split_ifs
      · norm_num
      · exact (mul_pos hz hG).le
  have hpw_nn : ∀ (m : ℕ) (j₀ : Fin k) (γ : PathChoice m),
      0 ≤ pathProdWeight k hk p z (fun _ => Gplus) m j₀ γ := by
    intro m
    induction m with
    | zero => intro _ _; simp [pathProdWeight]
    | succ m ih =>
      intro j₀ γ
      exact mul_nonneg (ih j₀ γ.1) (hsw_nn _ _)
  have hprob_nn : ∀ (m : ℕ) (ω : Fin k × PathChoice m),
      0 ≤ refPathProb k hk p z Gplus lam w v m ω := by
    intro m ω
    unfold refPathProb
    have h1 : 0 ≤ lam ^ (-(m : ℤ)) := zpow_nonneg hlam.le _
    exact mul_nonneg (mul_nonneg (mul_nonneg h1 (hw _).le) (hpw_nn m ω.1 ω.2)) (hv _).le
  have hpath_exp : ∀ (m : ℕ) (H : ℕ → ℝ),
      lam ^ (-(m : ℤ)) * dotProduct w ((prodAll k p z H m).mulVec v) =
        ∑ ω : Fin k × PathChoice m,
          refPathProb k hk p z Gplus lam w v m ω * pathHostRatio k hk Gplus H m ω.1 ω.2 := by
    intro m H
    have hdot : dotProduct w ((prodAll k p z H m).mulVec v) =
        ∑ j₀ : Fin k, ∑ γ : PathChoice m,
          w (pathEnd k hk m j₀ γ) * pathProdWeight k hk p z H m j₀ γ * v j₀ := by
      unfold dotProduct
      simp_rw [prodAll_mulVec_eq_path_sum k hk p z H v m, Finset.mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun j₀ _ => ?_
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun γ _ => ?_
      rw [Finset.sum_eq_single (pathEnd k hk m j₀ γ)]
      · simp [mul_assoc]
      · intro i _ hi; simp [hi.symm]
      · intro hm; exact False.elim (hm (Finset.mem_univ _))
    rw [hdot, Finset.mul_sum, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun j₀ _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun γ _ => ?_
    rw [pathProdWeight_eq_const_mul_ratio k hk p z Gplus hG.ne' H m j₀ γ]
    unfold refPathProb
    ring
  have hconst_pow : ∀ m : ℕ,
      (prodAll k p z (fun _ => Gplus) m).mulVec v = (lam ^ m) • v := by
    intro m
    induction m with
    | zero => simp [prodAll]
    | succ m ih =>
      rw [prodAll, ← Matrix.mulVec_mulVec, ih, Matrix.mulVec_smul, hAv, smul_smul, pow_succ]
  have hprob_sum : ∀ m : ℕ,
      ∑ ω : Fin k × PathChoice m, refPathProb k hk p z Gplus lam w v m ω = 1 := by
    intro m
    have hratio_one : ∀ (j₀ : Fin k) (γ : PathChoice m),
        pathHostRatio k hk Gplus (fun _ => Gplus) m j₀ γ = 1 := by
      clear hpath_exp
      induction m with
      | zero => intro _ _; rfl
      | succ m ih =>
        intro j₀ γ
        simp [pathHostRatio, ih j₀ γ.1, div_self hG.ne']
    have hexp := hpath_exp m (fun _ => Gplus)
    simp_rw [hratio_one, mul_one] at hexp
    rw [← hexp, hconst_pow m, dotProduct_smul, smul_eq_mul, hwv, mul_one,
      zpow_neg, zpow_natCast, inv_mul_cancel₀ (pow_pos hlam m).ne']
  have hfreq : ∀ (m : ℕ) (t : ℕ), t < m →
      ∑ ω : Fin k × PathChoice m,
        refPathProb k hk p z Gplus lam w v m ω * pathUsesHostAt k hk m ω.1 ω.2 t = ν := by
    intro m t htm
    let H_pert : ℕ → ℝ := fun s => if s = t then 2 * Gplus else Gplus
    have hrat_pert : ∀ (r : ℕ),
        (∀ j₀ : Fin k, ∀ γ : PathChoice r,
          (t < r → pathHostRatio k hk Gplus H_pert r j₀ γ = 1 + pathUsesHostAt k hk r j₀ γ t) ∧
          (r ≤ t → pathHostRatio k hk Gplus H_pert r j₀ γ = 1)) := by
      intro r
      induction r with
      | zero =>
        intro j₀ γ
        exact ⟨fun h => by omega, fun _ => rfl⟩
      | succ r ihr =>
        intro j₀ γ
        refine ⟨fun htr => ?_, fun hrt => ?_⟩
        · simp only [pathHostRatio, pathUsesHostAt]
          by_cases hteq : t = r
          · simp only [hteq, ↓reduceIte]
            have hr_le : r ≤ t := by omega
            rw [(ihr j₀ γ.1).2 hr_le]
            have hH : H_pert r / Gplus = 2 := by
              dsimp [H_pert]; simp only [hteq, ↓reduceIte]; field_simp [hG.ne']
            rw [hH]
            split_ifs <;> ring
          · simp only [hteq, ↓reduceIte]
            have hlt' : t < r := by omega
            rw [(ihr j₀ γ.1).1 hlt']
            have hrt_ne : r ≠ t := Ne.symm hteq
            have hH : H_pert r / Gplus = 1 := by
              dsimp [H_pert]; simp only [hrt_ne, ↓reduceIte]; exact div_self hG.ne'
            rw [hH]
            split_ifs <;> ring
        · simp only [pathHostRatio]
          rw [(ihr j₀ γ.1).2 (by omega)]
          have hne : r ≠ t := by omega
          have hH : H_pert r / Gplus = 1 := by
            dsimp [H_pert]; simp only [hne, ↓reduceIte]; exact div_self hG.ne'
          rw [hH]
          split_ifs <;> ring
    have hvec_before : ∀ r : ℕ, r ≤ t →
        (prodAll k p z H_pert r).mulVec v = (lam ^ r) • v := by
      intro r hr
      induction r with
      | zero => simp [prodAll]
      | succ r ihr =>
        have hne : r ≠ t := by omega
        have hHr : H_pert r = Gplus := by dsimp [H_pert]; simp [hne]
        rw [prodAll, hHr, ← Matrix.mulVec_mulVec, ihr (by omega), Matrix.mulVec_smul, hAv,
          smul_smul, pow_succ]
    have hw_step : ∀ u : Fin k → ℝ,
        dotProduct w ((outageRingMat k p (z * Gplus)).mulVec u) = lam * dotProduct w u := by
      intro u
      have hcomm : dotProduct w ((outageRingMat k p (z * Gplus)).mulVec u) =
          dotProduct ((outageRingMat k p (z * Gplus)).vecMul w) u := by
        simp only [dotProduct, Matrix.mulVec, Matrix.vecMul, Finset.mul_sum, Finset.sum_mul]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring
      rw [hcomm, hwA, smul_dotProduct, smul_eq_mul]
    have hdot_after : ∀ d : ℕ,
        dotProduct w ((prodAll k p z H_pert (t + 1 + d)).mulVec v) =
          lam ^ (t + 1 + d) * (1 + ν) := by
      intro d
      induction d with
      | zero =>
        simp only [Nat.add_zero, prodAll, ← Matrix.mulVec_mulVec]
        rw [hvec_before t le_rfl]
        have hHt : H_pert t = 2 * Gplus := by dsimp [H_pert]; simp
        rw [hHt, Matrix.mulVec_smul, dotProduct_smul, smul_eq_mul]
        have hsplit : outageRingMat k p (z * (2 * Gplus)) =
            outageRingMat k p (z * Gplus) + (z * Gplus) • hostRankOne k := by
          unfold outageRingMat
          ext i j
          simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
          ring
        rw [hsplit, Matrix.add_mulVec, Matrix.smul_mulVec, dotProduct_add, dotProduct_smul,
          hw_step v, hwv, mul_one, smul_eq_mul]
        have hrank_dot : dotProduct w ((hostRankOne k).mulVec v) =
            w ⟨0, by omega⟩ * v ⟨k - 1, by omega⟩ := by
          unfold dotProduct Matrix.mulVec hostRankOne
          let z0 : Fin k := ⟨0, by omega⟩
          let zlast : Fin k := ⟨k - 1, by omega⟩
          rw [Finset.sum_eq_single z0]
          · simp only [Matrix.of_apply, z0, true_and]
            congr 1
            unfold dotProduct
            rw [Finset.sum_eq_single zlast]
            · simp [zlast]
            · intro j _ hj
              have hjn : (j : ℕ) ≠ k - 1 := fun h => hj (Fin.ext h)
              simp [hjn]
            · intro hz0; exact False.elim (hz0 (Finset.mem_univ zlast))
          · intro i _ hi
            have hin : (i : ℕ) ≠ 0 := fun h => hi (Fin.ext h)
            simp [Matrix.of_apply, dotProduct, hin]
          · intro hz0; exact False.elim (hz0 (Finset.mem_univ z0))
        rw [hrank_dot]
        dsimp [ν]
        rw [pow_succ]
        field_simp [hlam.ne']
      | succ d ihd =>
        have hidx : t + 1 + (d + 1) = (t + 1 + d) + 1 := by omega
        rw [hidx, prodAll, ← Matrix.mulVec_mulVec]
        have hne : t + 1 + d ≠ t := by omega
        have hHr : H_pert (t + 1 + d) = Gplus := by dsimp [H_pert]; simp [hne]
        rw [hHr, hw_step, ihd, pow_succ]
        ring
    have hexp := hpath_exp m H_pert
    have hm_eq : m = t + 1 + (m - 1 - t) := by omega
    have hdot_m : dotProduct w ((prodAll k p z H_pert m).mulVec v) = lam ^ m * (1 + ν) := by
      conv_lhs => rw [hm_eq]
      rw [hdot_after (m - 1 - t), ← hm_eq]
    rw [hdot_m, zpow_neg, zpow_natCast, ← mul_assoc,
      inv_mul_cancel₀ (pow_pos hlam m).ne', one_mul] at hexp
    have hrhs_eq : (∑ ω : Fin k × PathChoice m,
        refPathProb k hk p z Gplus lam w v m ω * pathHostRatio k hk Gplus H_pert m ω.1 ω.2) =
        (∑ ω : Fin k × PathChoice m, refPathProb k hk p z Gplus lam w v m ω) +
        (∑ ω : Fin k × PathChoice m,
          refPathProb k hk p z Gplus lam w v m ω * pathUsesHostAt k hk m ω.1 ω.2 t) := by
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun ω _ => ?_
      rw [(hrat_pert m ω.1 ω.2).1 htm]
      ring
    rw [hrhs_eq, hprob_sum m] at hexp
    linarith
  refine ⟨hprob_nn, hprob_sum, hpath_exp, ?_⟩
  intro m H hH_pos
  have hrat_pos_and_log : ∀ (r : ℕ), r ≤ m →
      ∀ (j₀ : Fin k) (γ : PathChoice r),
        0 < pathHostRatio k hk Gplus H r j₀ γ ∧
        Real.log (pathHostRatio k hk Gplus H r j₀ γ) =
          ∑ t ∈ Finset.range r, pathUsesHostAt k hk r j₀ γ t * Real.log (H t / Gplus) := by
    intro r
    induction r with
    | zero =>
      intro _ j₀ γ
      simp [pathHostRatio]
    | succ r ihr =>
      intro hrm j₀ γ
      obtain ⟨hpos_prev, hlog_prev⟩ := ihr (by omega) j₀ γ.1
      have hHr : 0 < H r / Gplus := div_pos (hH_pos r (by omega)) hG
      have hstep_pos : 0 < (if isHostEdge k (pathEnd k hk r j₀ γ.1) γ.2 then H r / Gplus else 1) := by
        split_ifs <;> [exact hHr; exact one_pos]
      have htot_pos : 0 < pathHostRatio k hk Gplus H (r + 1) j₀ γ :=
        mul_pos hpos_prev hstep_pos
      refine ⟨htot_pos, ?_⟩
      simp only [pathHostRatio]
      rw [Real.log_mul hpos_prev.ne' hstep_pos.ne', hlog_prev, Finset.sum_range_succ]
      congr 1
      · refine Finset.sum_congr rfl fun t ht => ?_
        have htr : t ≠ r := Nat.ne_of_lt (Finset.mem_range.mp ht)
        simp [pathUsesHostAt, htr]
      · simp only [pathUsesHostAt, ↓reduceIte]
        split_ifs <;> simp
  have hsum_pos : 0 < ∑ ω : Fin k × PathChoice m,
      refPathProb k hk p z Gplus lam w v m ω * pathHostRatio k hk Gplus H m ω.1 ω.2 := by
    have hexists : ∃ ω₀ ∈ (Finset.univ : Finset (Fin k × PathChoice m)),
        0 < refPathProb k hk p z Gplus lam w v m ω₀ := by
      by_contra hall
      push Not at hall
      have h0 : ∑ ω : Fin k × PathChoice m, refPathProb k hk p z Gplus lam w v m ω ≤ 0 :=
        Finset.sum_nonpos hall
      linarith [hprob_sum m]
    obtain ⟨ω₀, hω₀_mem, hω₀_pos⟩ := hexists
    exact Finset.sum_pos'
      (fun ω _ => mul_nonneg (hprob_nn m ω) (hrat_pos_and_log m le_rfl ω.1 ω.2).1.le)
      ⟨ω₀, hω₀_mem, mul_pos hω₀_pos (hrat_pos_and_log m le_rfl ω₀.1 ω₀.2).1⟩
  have hdot_pos : 0 < dotProduct w ((prodAll k p z H m).mulVec v) := by
    have hexp := hpath_exp m H
    have hlam_pow : 0 < lam ^ (-(m : ℤ)) := zpow_pos hlam _
    nlinarith
  have hjensen := strictConcaveOn_log_Ioi.concaveOn.le_map_sum
    (t := (Finset.univ : Finset (Fin k × PathChoice m)))
    (w := refPathProb k hk p z Gplus lam w v m)
    (p := fun ω => pathHostRatio k hk Gplus H m ω.1 ω.2)
    (fun ω _ => hprob_nn m ω)
    (hprob_sum m)
    (fun ω _ => Set.mem_Ioi.mpr (hrat_pos_and_log m le_rfl ω.1 ω.2).1)
  simp only [smul_eq_mul] at hjensen
  rw [← hpath_exp m H] at hjensen
  have hlog_lhs : Real.log (lam ^ (-(m : ℤ)) * dotProduct w ((prodAll k p z H m).mulVec v)) =
      -(m : ℝ) * Real.log lam + Real.log (dotProduct w ((prodAll k p z H m).mulVec v)) := by
    rw [Real.log_mul (zpow_pos hlam _).ne' hdot_pos.ne', Real.log_zpow]
    simp
  have hsum_rhs : (∑ ω : Fin k × PathChoice m,
      refPathProb k hk p z Gplus lam w v m ω *
        Real.log (pathHostRatio k hk Gplus H m ω.1 ω.2)) =
      ν * (∑ t ∈ Finset.range m, Real.log (H t / Gplus)) := by
    have h1 : ∀ ω : Fin k × PathChoice m,
        refPathProb k hk p z Gplus lam w v m ω *
          Real.log (pathHostRatio k hk Gplus H m ω.1 ω.2) =
        ∑ t ∈ Finset.range m,
          (refPathProb k hk p z Gplus lam w v m ω * pathUsesHostAt k hk m ω.1 ω.2 t) *
            Real.log (H t / Gplus) := by
      intro ω
      rw [(hrat_pos_and_log m le_rfl ω.1 ω.2).2, Finset.mul_sum]
      refine Finset.sum_congr rfl fun t _ => by ring
    simp_rw [h1]
    rw [Finset.sum_comm, Finset.mul_sum]
    refine Finset.sum_congr rfl fun t ht => ?_
    rw [← Finset.sum_mul, hfreq m t (Finset.mem_range.mp ht)]
  rw [hlog_lhs, hsum_rhs] at hjensen
  exact ⟨hdot_pos, by linarith⟩


/-- General-`k` characteristic determinant `det(λ I − A(h)) = ∏_j (λ − p_j) − z h`. -/
theorem O7_ring_characteristic_determinant (k : ℕ) (hk : 1 ≤ k) (p : Fin k → ℝ) (lam zh : ℝ) :
    (lam • (1 : Matrix (Fin k) (Fin k) ℝ) - outageRingMat k p zh).det =
      (∏ j : Fin k, (lam - p j)) - zh := by
  obtain _ | _ | n := k
  · omega
  · rw [Matrix.det_fin_one, Fin.prod_univ_one]
    simp [outageRingMat, bidiagC, hostRankOne]
    ring
  · let M := lam • (1 : Matrix (Fin (n + 2)) (Fin (n + 2)) ℝ) - outageRingMat (n + 2) p zh
    have hM_apply : ∀ i j : Fin (n + 2),
        M i j = (if i = j then lam - p i else 0) -
          (if (i : ℕ) = (j : ℕ) + 1 then 1 else 0) -
          (if (i : ℕ) = 0 ∧ (j : ℕ) = n + 1 then zh else 0) := by
      intro i j
      dsimp [M]
      unfold outageRingMat bidiagC hostRankOne
      simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply, Matrix.add_apply,
        Matrix.of_apply, smul_eq_mul]
      have hts : n + 2 - 1 = n + 1 := by omega
      rw [hts]
      by_cases hij : i = j
      · subst hij; simp; ring
      · simp [hij]; split_ifs <;> ring
    have hrow0 : M 0 0 = lam - p 0 := by rw [hM_apply]; simp
    have hrowlast : M 0 (Fin.last (n + 1)) = -zh := by
      rw [hM_apply]
      have hne : (0 : Fin (n + 2)) ≠ Fin.last (n + 1) := by simp [Fin.ext_iff]
      simp [hne]
    have hrow_mid : ∀ c : Fin (n + 2), c ≠ 0 → c ≠ Fin.last (n + 1) → M 0 c = 0 := by
      intro c hc0 hcl
      rw [hM_apply]
      have h1 : (0 : Fin (n + 2)) ≠ c := hc0.symm
      have h2 : (c : ℕ) ≠ n + 1 := fun h => hcl (Fin.ext h)
      simp [h1, h2]
    have hdetL : (M.submatrix Fin.succ Fin.succ).det = ∏ i : Fin (n + 1), (lam - p i.succ) := by
      rw [Matrix.det_of_isLowerTriangular]
      · refine Finset.prod_congr rfl fun i _ => ?_
        simp only [Matrix.submatrix_apply]
        rw [hM_apply]
        simp [Fin.val_succ]
      · intro i j hij
        have hij' : (i : ℕ) < (j : ℕ) := OrderDual.toDual_lt_toDual.mp hij
        simp only [Matrix.submatrix_apply]
        rw [hM_apply]
        have h1 : i.succ ≠ j.succ := fun h => by have := Fin.succ_inj.mp h; subst this; omega
        have h2 : (i.succ : ℕ) ≠ (j.succ : ℕ) + 1 := by simp [Fin.val_succ]; omega
        simp only [h1, h2, ↓reduceIte]
        simp
    have hdetU : (M.submatrix Fin.succ Fin.castSucc).det = (-1 : ℝ) ^ (n + 1) := by
      rw [Matrix.det_of_isUpperTriangular]
      · simp only [Matrix.submatrix_apply]
        have hdiag : ∀ i : Fin (n + 1), M i.succ i.castSucc = -1 := by
          intro i
          rw [hM_apply]
          have h1 : i.succ ≠ i.castSucc := fun h => by
            have := congr_arg Fin.val h; simp [Fin.val_succ, Fin.val_castSucc] at this
          simp [h1, Fin.val_succ, Fin.val_castSucc]
        simp_rw [hdiag]
        rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
      · intro i j hij
        have hij' : (j : ℕ) < (i : ℕ) := hij
        simp only [Matrix.submatrix_apply]
        rw [hM_apply]
        have h1 : i.succ ≠ j.castSucc := fun h => by
          have := congr_arg Fin.val h; simp [Fin.val_succ, Fin.val_castSucc] at this; omega
        have h2 : (i.succ : ℕ) ≠ (j.castSucc : ℕ) + 1 := by
          simp [Fin.val_succ, Fin.val_castSucc]; omega
        simp only [h1, h2, ↓reduceIte]
        simp
    have hsq : ((-1 : ℝ) ^ (n + 1)) * ((-1 : ℝ) ^ (n + 1)) = 1 := by
      rw [← mul_pow]; simp
    have hprod : (∏ j : Fin (n + 2), (lam - p j)) =
        (lam - p 0) * ∏ i : Fin (n + 1), (lam - p i.succ) := by
      rw [Fin.prod_univ_succ]
    change M.det = _
    rw [Matrix.det_succ_row_zero,
      Finset.sum_eq_add (0 : Fin (n + 2)) (Fin.last (n + 1)) (by simp [Fin.ext_iff]) ?_
        (by simp) (by simp)]
    · rw [Fin.succAbove_zero, Fin.succAbove_last, hdetL, hdetU, hrow0, hrowlast, hprod]
      simp only [Fin.val_zero, Fin.val_last, pow_zero]
      linear_combination (-zh) * hsq
    · intro c _ hc
      rw [hrow_mid c hc.1 hc.2]
      ring


/-! ## O7 and O8 -/

/-- Critical reference eigenvalue `λ_* = pmax ^ (−q₀ / (1 − q₀))` of `a13_proof.md` `(6.1)`. -/
noncomputable def lamStar (pmax q0 : ℝ) : ℝ :=
  pmax ^ (-q0 / (1 - q0))

/-- Universal single-fluctuating-transfer outage upper bound `Dsup` of `a13_proof.md` `(6.1)`. -/
noncomputable def Dsup (k : ℕ) (τ : Fin k → ℝ) (pmax q0 Gplus : ℝ) : ℝ :=
  Real.log ((∏ j : Fin k, τ j * (lamStar pmax q0 - (1 - 1 / τ j))) / Gplus)

/-- Explicit Hypothesis H1 (Part ii subadditive convexity & monotonicity of `y ↦ r(y)`,
standing in for Kingman's subadditive ergodic theorem + Hölder convexity on `ℝ`, `a13_proof.md` §6 Step 4). -/
def H1_SubadditiveConvexMono (r : ℝ → ℝ) : Prop :=
  ConvexOn ℝ Set.univ r ∧ Monotone r

/-- Explicit Hypothesis H2 (Part ii negative tilt limit `lim_{y → -∞} r(y) = log pmax < 0`,
witnessed by some `y_neg` with `r(y_neg) < 0`, `a13_proof.md` §6 Step 4). -/
def H2_NegativeTiltPoint (r : ℝ → ℝ) : Prop :=
  ∃ y_neg : ℝ, r y_neg < 0

/-- Explicit Hypothesis H3 (Part ii Birkhoff ergodic passage from finite-horizon `O5` + `O6` to
`r(y) ≥ q₀ log pmax + (1 − q₀) log λ`, `a13_proof.md` `(6.2)`). -/
def H3_ErgodicOutageLowerBound (k : ℕ) (τ : Fin k → ℝ) (pmax q0 Gplus : ℝ) (r : ℝ → ℝ) : Prop :=
  ∀ (y lam : ℝ), pmax < lam →
    ∏ j : Fin k, (lam - (1 - 1 / τ j)) = Real.exp y * Gplus / (∏ j : Fin k, τ j) →
      q0 * Real.log pmax + (1 - q0) * Real.log lam ≤ r y

theorem O7_part_ii_characteristic_and_bound (k : ℕ) (_hk : 1 ≤ k) (τ : Fin k → ℝ)
    (hτ : ∀ j : Fin k, 1 ≤ τ j) (pmax q0 Gplus : ℝ)
    (hpmax0 : 0 < pmax) (hpmax1 : pmax < 1)
    (hp_le : ∀ j : Fin k, 1 - 1 / τ j ≤ pmax)
    (hq0_pos : 0 < q0) (hq0_lt1 : q0 < 1) (hG : 0 < Gplus) :
    (1 < lamStar pmax q0 ∧
      (∀ j : Fin k, 0 < τ j * (lamStar pmax q0 - (1 - 1 / τ j))) ∧
      ∏ j : Fin k, (lamStar pmax q0 - (1 - 1 / τ j)) =
        Real.exp (Dsup k τ pmax q0 Gplus) * Gplus / (∏ j : Fin k, τ j) ∧
      q0 * Real.log pmax + (1 - q0) * Real.log (lamStar pmax q0) = 0) ∧
    (∀ r : ℝ → ℝ,
      H1_SubadditiveConvexMono r →
      H2_NegativeTiltPoint r →
      H3_ErgodicOutageLowerBound k τ pmax q0 Gplus r →
        0 ≤ r (Dsup k τ pmax q0 Gplus) ∧
        (∃ D : ℝ, r D = 0 ∧ D ≤ Dsup k τ pmax q0 Gplus ∧
          ∀ D' : ℝ, r D' = 0 → D' = D)) := by
  have h1q : 0 < 1 - q0 := by linarith
  have hexp_neg : -q0 / (1 - q0) < 0 := div_neg_of_neg_of_pos (by linarith) h1q
  have hlam_gt1 : 1 < lamStar pmax q0 := by
    unfold lamStar
    exact Real.one_lt_rpow_of_pos_of_lt_one_of_neg hpmax0 hpmax1 hexp_neg
  have hlam_pos : 0 < lamStar pmax q0 := zero_lt_one.trans hlam_gt1
  have hfac_pos : ∀ j : Fin k, 0 < τ j * (lamStar pmax q0 - (1 - 1 / τ j)) := by
    intro j
    have hτj : 0 < τ j := lt_of_lt_of_le zero_lt_one (hτ j)
    have hpj_lt : 1 - 1 / τ j < lamStar pmax q0 :=
      lt_of_le_of_lt (hp_le j) (hpmax1.trans hlam_gt1)
    exact mul_pos hτj (sub_pos.mpr hpj_lt)
  have hτ_prod_pos : 0 < ∏ j : Fin k, τ j :=
    Finset.prod_pos fun j _ => lt_of_lt_of_le zero_lt_one (hτ j)
  have hnum_pos : 0 < (∏ j : Fin k, τ j * (lamStar pmax q0 - (1 - 1 / τ j))) / Gplus :=
    div_pos (Finset.prod_pos fun j _ => hfac_pos j) hG
  have hchar : ∏ j : Fin k, (lamStar pmax q0 - (1 - 1 / τ j)) =
      Real.exp (Dsup k τ pmax q0 Gplus) * Gplus / (∏ j : Fin k, τ j) := by
    unfold Dsup
    rw [Real.exp_log hnum_pos, div_mul_cancel₀ _ hG.ne', Finset.prod_mul_distrib,
      mul_div_cancel_left₀ _ hτ_prod_pos.ne']
  have hcancel : q0 * Real.log pmax + (1 - q0) * Real.log (lamStar pmax q0) = 0 := by
    unfold lamStar
    rw [Real.log_rpow hpmax0]
    field_simp [h1q.ne']
    ring
  refine ⟨⟨hlam_gt1, hfac_pos, hchar, hcancel⟩, ?_⟩
  intro r ⟨hconv, hmono⟩ ⟨y_neg, hy_neg⟩ hlower
  have hr_Dsup : 0 ≤ r (Dsup k τ pmax q0 Gplus) := by
    have hl := hlower (Dsup k τ pmax q0 Gplus) (lamStar pmax q0) (hpmax1.trans hlam_gt1) hchar
    linarith [hcancel]
  -- Continuity of `r` on `ℝ` from `ConvexOn ℝ Set.univ r`:
  have hcont : Continuous r :=
    continuousOn_univ.mp (hconv.continuousOn isOpen_univ)
  -- `y_neg < Dsup` since `r y_neg < 0 ≤ r Dsup` and `r` is monotone:
  have hy_lt : y_neg ≤ Dsup k τ pmax q0 Gplus := by
    by_contra hgt
    push Not at hgt
    have hle := hmono hgt.le
    linarith
  -- Existence on `[y_neg, Dsup]` via Intermediate Value Theorem:
  have hmem0 : (0 : ℝ) ∈ Set.Icc (r y_neg) (r (Dsup k τ pmax q0 Gplus)) :=
    ⟨hy_neg.le, hr_Dsup⟩
  obtain ⟨D, hD_mem, hD_zero⟩ :=
    intermediate_value_Icc hy_lt hcont.continuousOn hmem0
  -- Uniqueness: any two zeros `D₁ < D₂` violate `ConvexOn` against `y_neg` (`r y_neg < 0`)!
  have huniq : ∀ y₁ y₂ : ℝ, r y₁ = 0 → r y₂ = 0 → y₁ < y₂ → False := by
    intro y₁ y₂ hz₁ hz₂ hlt12
    have hy_lt1 : y_neg < y₁ := by
      by_contra hle
      push Not at hle
      have := hmono hle
      linarith
    have hden : 0 < y₂ - y_neg := by linarith
    let a := (y₂ - y₁) / (y₂ - y_neg)
    let b := (y₁ - y_neg) / (y₂ - y_neg)
    have ha_pos : 0 < a := div_pos (by linarith) hden
    have hb_nn : 0 ≤ b := div_nonneg (by linarith) hden.le
    have hab : a + b = 1 := by dsimp [a, b]; field_simp [hden.ne']; ring
    have hcomb : a • y_neg + b • y₂ = y₁ := by
      simp only [smul_eq_mul, a, b]
      field_simp [hden.ne']
      ring
    have hconv_ineq := hconv.2 (Set.mem_univ y_neg) (Set.mem_univ y₂) ha_pos.le hb_nn hab
    rw [hcomb, hz₁, hz₂] at hconv_ineq
    simp only [smul_eq_mul, mul_zero, add_zero] at hconv_ineq
    have hneg_prod : a * r y_neg < 0 := mul_neg_of_pos_of_neg ha_pos hy_neg
    linarith
  refine ⟨hr_Dsup, D, hD_zero, hD_mem.2, ?_⟩
  intro D' hD'_zero
  rcases lt_trichotomy D' D with hlt | heq | hgt
  · exact False.elim (huniq D' D hD'_zero hD_zero hlt)
  · exact heq
  · exact False.elim (huniq D D' hD_zero hD'_zero hgt)

theorem O8_Dsup_large_stay_limit_and_strict_jensen (k_prev : ℕ) (τ_fixed : Fin k_prev → ℝ)
    (hτ : ∀ j : Fin k_prev, 1 ≤ τ_fixed j) (q0 Gplus : ℝ)
    (_hq0_pos : 0 < q0) (hq0_lt1 : q0 < 1) (hG : 0 < Gplus) :
    (Filter.Tendsto
        (fun s : ℝ => s * (lamStar (1 - 1 / s) q0 - (1 - 1 / s)))
        Filter.atTop (nhds (1 / (1 - q0))) ∧
      (∀ j : Fin k_prev,
        Filter.Tendsto
          (fun s : ℝ => τ_fixed j * (lamStar (1 - 1 / s) q0 - (1 - 1 / τ_fixed j)))
          Filter.atTop (nhds 1)) ∧
      Filter.Tendsto
        (fun s : ℝ => Dsup (k_prev + 1) (Fin.snoc τ_fixed s) (1 - 1 / s) q0 Gplus)
        Filter.atTop (nhds (-Real.log ((1 - q0) * Gplus)))) ∧
    (∀ {ι : Type*} (t : Finset ι) (w h : ι → ℝ),
      (∀ i ∈ t, 0 < w i) →
      ∑ i ∈ t, w i = 1 - q0 →
      (∀ i ∈ t, 0 < h i) →
      ∑ i ∈ t, w i * h i = 1 →
      Gplus = Real.exp ((∑ i ∈ t, w i * Real.log (h i)) / (1 - q0)) →
        0 ≤ -Real.log ((1 - q0) * Gplus) ∧
        (-Real.log ((1 - q0) * Gplus) = 0 ↔ ∀ i ∈ t, h i = 1 / (1 - q0))) := by
  have h1q : 0 < 1 - q0 := by linarith
  let α := -q0 / (1 - q0)
  -- `1 - 1 / s → 1` through `≠ 1`:
  have hinv_s : Filter.Tendsto (fun s : ℝ => (1 : ℝ) / s) Filter.atTop (nhds 0) :=
    tendsto_const_nhds.div_atTop Filter.tendsto_id
  have hp_lim : Filter.Tendsto (fun s : ℝ => 1 - 1 / s) Filter.atTop (nhds 1) := by
    simpa using tendsto_const_nhds.sub hinv_s
  have hp_nhdsWithin : Filter.Tendsto (fun s : ℝ => 1 - 1 / s) Filter.atTop
      (nhdsWithin 1 {(1 : ℝ)}ᶜ) := by
    rw [tendsto_nhdsWithin_iff]
    refine ⟨hp_lim, ?_⟩
    filter_upwards [Filter.eventually_gt_atTop 0] with s hs0
    simp only [Set.mem_compl_iff, Set.mem_singleton_iff]
    have hpos : 0 < 1 / s := one_div_pos.mpr hs0
    linarith
  -- Derivative of `x ↦ x ^ α` at `x = 1`:
  have hdrpow : HasDerivAt (fun x : ℝ => x ^ α) α 1 := by
    simpa using Real.hasDerivAt_rpow_const (x := (1 : ℝ)) (p := α) (Or.inl one_ne_zero)
  have hslope := (hasDerivAt_iff_tendsto_slope.mp hdrpow).comp hp_nhdsWithin
  -- Conclude `s * (lamStar (1 - 1 / s) q0 - (1 - 1 / s)) → -α + 1 = 1 / (1 - q0)`:
  have hslow_lim : Filter.Tendsto
      (fun s : ℝ => s * (lamStar (1 - 1 / s) q0 - (1 - 1 / s)))
      Filter.atTop (nhds (1 / (1 - q0))) := by
    have hlin : Filter.Tendsto (fun s : ℝ => 1 - (slope (fun x : ℝ => x ^ α) 1 (1 - 1 / s)))
        Filter.atTop (nhds (1 - α)) :=
      tendsto_const_nhds.sub hslope
    have hval : 1 - α = 1 / (1 - q0) := by
      dsimp [α]; field_simp [h1q.ne']; ring
    rw [hval] at hlin
    refine hlin.congr' ?_
    filter_upwards [Filter.eventually_gt_atTop 1] with s hs1
    have hs0 : 0 < s := zero_lt_one.trans hs1
    have hne : (1 - 1 / s) - 1 ≠ 0 := by
      have : 0 < 1 / s := one_div_pos.mpr hs0
      linarith
    rw [slope_def_field]
    simp only [Real.one_rpow, lamStar, α]
    have hden_sub : 1 - 1 / s - 1 = -(1 / s) := by ring
    rw [hden_sub]
    field_simp [hs0.ne']
    ring
  -- Also `lamStar (1 - 1 / s) q0 → 1`:
  have hlam_lim : Filter.Tendsto (fun s : ℝ => lamStar (1 - 1 / s) q0)
      Filter.atTop (nhds 1) := by
    have hcont_rpow := Real.continuousAt_rpow_const (1 : ℝ) α (Or.inl one_ne_zero)
    have hcomp := hcont_rpow.tendsto.comp hp_lim
    simpa [lamStar, α, Function.comp_def] using hcomp
  have hfixed_lim : ∀ j : Fin k_prev,
      Filter.Tendsto
        (fun s : ℝ => τ_fixed j * (lamStar (1 - 1 / s) q0 - (1 - 1 / τ_fixed j)))
        Filter.atTop (nhds 1) := by
    intro j
    have hτj_pos : 0 < τ_fixed j := lt_of_lt_of_le zero_lt_one (hτ j)
    have h1 := (tendsto_const_nhds (x := τ_fixed j)).mul (hlam_lim.sub (tendsto_const_nhds (x := 1 - 1 / τ_fixed j)))
    have hval : τ_fixed j * (1 - (1 - 1 / τ_fixed j)) = 1 := by
      field_simp [hτj_pos.ne']; ring
    rwa [hval] at h1
  -- Combine into `Dsup → -log((1 - q0) * Gplus)`:
  have hDsup_lim : Filter.Tendsto
      (fun s : ℝ => Dsup (k_prev + 1) (Fin.snoc τ_fixed s) (1 - 1 / s) q0 Gplus)
      Filter.atTop (nhds (-Real.log ((1 - q0) * Gplus))) := by
    have hprod_fixed : Filter.Tendsto
        (fun s : ℝ => ∏ j : Fin k_prev,
          τ_fixed j * (lamStar (1 - 1 / s) q0 - (1 - 1 / τ_fixed j)))
        Filter.atTop (nhds (∏ _j : Fin k_prev, (1 : ℝ))) :=
      tendsto_finsetProd Finset.univ fun j _ => hfixed_lim j
    simp only [Finset.prod_const_one] at hprod_fixed
    have hprod_all : Filter.Tendsto
        (fun s : ℝ => ((∏ j : Fin (k_prev + 1),
          ((Fin.snoc τ_fixed s : Fin (k_prev + 1) → ℝ) j) *
            (lamStar (1 - 1 / s) q0 - (1 - 1 / ((Fin.snoc τ_fixed s : Fin (k_prev + 1) → ℝ) j)))) / Gplus))
        Filter.atTop (nhds ((1 * (1 / (1 - q0))) / Gplus)) := by
      have hmul := (hprod_fixed.mul hslow_lim).div_const Gplus
      refine hmul.congr fun s => ?_
      rw [Fin.prod_univ_castSucc]
      simp [Fin.snoc_castSucc, Fin.snoc_last]
    have hpos_lim : 0 < (1 * (1 / (1 - q0))) / Gplus := by positivity
    have hlog := (Real.continuousAt_log hpos_lim.ne').tendsto.comp hprod_all
    have hlog_eq : Real.log ((1 * (1 / (1 - q0))) / Gplus) = -Real.log ((1 - q0) * Gplus) := by
      have hrew : (1 * (1 / (1 - q0))) / Gplus = ((1 - q0) * Gplus)⁻¹ := by
        field_simp [h1q.ne', hG.ne']
      rw [hrew, Real.log_inv]
    rw [hlog_eq] at hlog
    exact hlog
  -- Strict Jensen part on any finite positive-host law:
  refine ⟨⟨hslow_lim, hfixed_lim, hDsup_lim⟩, ?_⟩
  intro ι t w h hw_pos hw_sum hh_pos hwh hG_def
  let w' : ι → ℝ := fun i => w i / (1 - q0)
  have hw'_pos : ∀ i ∈ t, 0 < w' i := fun i hi => div_pos (hw_pos i hi) h1q
  have hw'_sum : ∑ i ∈ t, w' i = 1 := by
    dsimp [w']
    rw [← Finset.sum_div, hw_sum, div_self h1q.ne']
  have hmean_cond : ∑ i ∈ t, w' i * h i = 1 / (1 - q0) := by
    have h1 : ∀ i ∈ t, w' i * h i = (w i * h i) / (1 - q0) := fun i _ => by dsimp [w']; ring
    rw [Finset.sum_congr rfl h1, ← Finset.sum_div, hwh]
  have hlog_sum : (∑ i ∈ t, w i * Real.log (h i)) / (1 - q0) =
      ∑ i ∈ t, w' i * Real.log (h i) := by
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun i _ => by dsimp [w']; ring
  have hgap_id : -Real.log ((1 - q0) * Gplus) =
      Real.log (∑ i ∈ t, w' i * h i) - ∑ i ∈ t, w' i * Real.log (h i) := by
    rw [hmean_cond, hG_def, hlog_sum, Real.log_mul h1q.ne' (Real.exp_pos _).ne',
      Real.log_exp, one_div, Real.log_inv]
    ring
  have hjensen_le := strictConcaveOn_log_Ioi.concaveOn.le_map_sum
    (t := t) (w := w') (p := h)
    (fun i hi => (hw'_pos i hi).le) hw'_sum (fun i hi => Set.mem_Ioi.mpr (hh_pos i hi))
  simp only [smul_eq_mul] at hjensen_le
  have hjensen_eq := strictConcaveOn_log_Ioi.map_sum_eq_iff
    (t := t) (w := w') (p := h)
    hw'_pos hw'_sum (fun i hi => Set.mem_Ioi.mpr (hh_pos i hi))
  simp only [smul_eq_mul] at hjensen_eq
  refine ⟨by linarith [hgap_id, hjensen_le], ?_⟩
  rw [hgap_id, sub_eq_zero, hjensen_eq, hmean_cond]

/-- Fast `tsum` comparison without `whnf` norm timeouts. -/
private theorem abs_tsum_le_of_le {f g : ℕ → ℝ} (hf : Summable f) (hg : Summable g)
    (hle : ∀ m : ℕ, |f m| ≤ g m) :
    |∑' m : ℕ, f m| ≤ ∑' m : ℕ, g m := by
  rw [abs_le]
  refine ⟨?_, ?_⟩
  · have hneg : HasSum (fun m => -g m) (-(∑' m, g m)) := hg.hasSum.neg
    exact hasSum_le (fun m => (abs_le.mp (hle m)).1) hneg hf.hasSum
  · exact hasSum_le (fun m => (abs_le.mp (hle m)).2) hf.hasSum hg.hasSum

/-! ## Section 3: Sign Criterion of Part i, Where Proved (`O9`–`O13`) -/

/-- Raw-covariance long-stay coefficient `A = V / 2 + ∑_{m ≥ 0} p^m C_{m + 1}` (`a13_proof.md` `(4.1)`). -/
noncomputable def Acoeff (p V : ℝ) (C : ℕ → ℝ) : ℝ :=
  V / 2 + ∑' m : ℕ, p ^ m * C (m + 1)

/-- **O9 (`a13_proof.md` §4 `(4.1)`; `a13-check.md` §1.2 & §4.1 item 3).**
Raw-covariance bracket `A`:
1. Absolute summability and bound `|A| ≤ V * (1 / 2 + 1 / (1 − p))` for `0 ≤ p < 1` and
   `∀ ℓ ≥ 1, |C ℓ| ≤ V`;
2. Normalized autocorrelation factorization `A = V * (1 / 2 + ∑_{m ≥ 0} p^m ρ_{m + 1})` and
   sign equivalence `sign(A) = sign(1 / 2 + ∑_{m ≥ 0} p^m ρ_{m + 1})` when `V > 0`;
3. Exact evaluation on period-2 alternating outages `h = (2, 0)` (`V = 1`, `C ℓ = (−1)^ℓ`,
   `p = 1 − 1 / a`, `a > 1`): `A = −1 / [2 (2a − 1)] < 0` (matching `O3`). -/
theorem O9_raw_covariance_bracket_and_alt (p V : ℝ) (C : ℕ → ℝ)
    (hp0 : 0 ≤ p) (hp1 : p < 1) (hV : 0 ≤ V) (hC : ∀ m : ℕ, |C (m + 1)| ≤ V) :
    Summable (fun m : ℕ => |p ^ m * C (m + 1)|) ∧
    |Acoeff p V C| ≤ V * (1 / 2 + 1 / (1 - p)) ∧
    (∀ ρ : ℕ → ℝ, 0 < V → (∀ m : ℕ, C (m + 1) = V * ρ (m + 1)) →
      Acoeff p V C = V * (1 / 2 + ∑' m : ℕ, p ^ m * ρ (m + 1)) ∧
      Real.sign (Acoeff p V C) = Real.sign (1 / 2 + ∑' m : ℕ, p ^ m * ρ (m + 1))) ∧
    (∀ a : ℝ, 1 < a →
      Acoeff (1 - 1 / a) 1 (fun ℓ => (-1 : ℝ) ^ ℓ) = -1 / (2 * (2 * a - 1)) ∧
      Acoeff (1 - 1 / a) 1 (fun ℓ => (-1 : ℝ) ^ ℓ) < 0) := by
  have hp_norm : ‖p‖ < 1 := by
    rw [Real.norm_eq_abs, abs_of_nonneg hp0]
    exact hp1
  have hgeom_sum : Summable (fun m : ℕ => V * p ^ m) :=
    (summable_geometric_of_norm_lt_one hp_norm).mul_left V
  have hterm_le : ∀ m : ℕ, |p ^ m * C (m + 1)| ≤ V * p ^ m := by
    intro m
    rw [abs_mul, abs_of_nonneg (pow_nonneg hp0 m), mul_comm]
    exact mul_le_mul_of_nonneg_right (hC m) (pow_nonneg hp0 m)
  have hsum_abs : Summable (fun m : ℕ => |p ^ m * C (m + 1)|) :=
    Summable.of_nonneg_of_le (fun _ => abs_nonneg _) hterm_le hgeom_sum
  have hsum : Summable (fun m : ℕ => p ^ m * C (m + 1)) := hsum_abs.of_abs
  have htsum_le : |∑' m : ℕ, p ^ m * C (m + 1)| ≤ V * (1 / (1 - p)) := by
    have h := abs_tsum_le_of_le hsum hgeom_sum hterm_le
    simpa only [tsum_mul_left, tsum_geometric_of_norm_lt_one hp_norm, one_div] using h
  have hA_bound : |Acoeff p V C| ≤ V * (1 / 2 + 1 / (1 - p)) := by
    unfold Acoeff
    have htri := abs_add_le (V / 2) (∑' m : ℕ, p ^ m * C (m + 1))
    have hV2 : |V / 2| = V / 2 := abs_of_nonneg (by linarith)
    linarith [htri, hV2, htsum_le]
  refine ⟨hsum_abs, hA_bound, ?_, ?_⟩
  · intro ρ hVpos hCρ
    have hfac : Acoeff p V C = V * (1 / 2 + ∑' m : ℕ, p ^ m * ρ (m + 1)) := by
      unfold Acoeff
      have hrew : (fun m : ℕ => p ^ m * C (m + 1)) = (fun m : ℕ => V * (p ^ m * ρ (m + 1))) := by
        ext m; rw [hCρ m]; ring
      rw [hrew, tsum_mul_left]
      ring
    refine ⟨hfac, ?_⟩
    rw [hfac]
    set S := 1 / 2 + ∑' m : ℕ, p ^ m * ρ (m + 1)
    rcases lt_trichotomy S 0 with hneg | hzero | hpos
    · rw [Real.sign_of_neg (mul_neg_of_pos_of_neg hVpos hneg), Real.sign_of_neg hneg]
    · rw [hzero, mul_zero, Real.sign_zero]
    · rw [Real.sign_of_pos (mul_pos hVpos hpos), Real.sign_of_pos hpos]
  · intro a ha
    have ha0 : 0 < a := zero_lt_one.trans ha
    have h2a1 : 0 < 2 * a - 1 := by linarith
    have hpa0 : 0 < 1 - 1 / a := by
      rw [sub_pos, div_lt_one ha0]; exact ha
    have hpa1 : 1 - 1 / a < 1 := by
      have : 0 < 1 / a := one_div_pos.mpr ha0
      linarith
    have hnorm_neg : ‖-(1 - 1 / a)‖ < 1 := by
      rw [norm_neg, Real.norm_eq_abs, abs_of_pos hpa0]
      exact hpa1
    have htsum_alt : (∑' m : ℕ, (1 - 1 / a) ^ m * (-1 : ℝ) ^ (m + 1)) = -a / (2 * a - 1) := by
      have hfn : (fun m : ℕ => (1 - 1 / a) ^ m * (-1 : ℝ) ^ (m + 1)) =
          (fun m : ℕ => -(1 * (-(1 - 1 / a)) ^ m)) := by
        ext m
        rw [pow_succ, neg_eq_neg_one_mul (1 - 1 / a), mul_pow]
        ring
      rw [hfn, tsum_neg]
      simp only [one_mul]
      rw [tsum_geometric_of_norm_lt_one hnorm_neg]
      have hden : 1 - -(1 - 1 / a) = (2 * a - 1) / a := by field_simp [ha0.ne']; ring
      rw [hden]
      field_simp [ha0.ne', h2a1.ne']
    have hval : Acoeff (1 - 1 / a) 1 (fun ℓ => (-1 : ℝ) ^ ℓ) = -1 / (2 * (2 * a - 1)) := by
      unfold Acoeff
      rw [htsum_alt]
      field_simp [h2a1.ne']
      ring
    exact ⟨hval, by rw [hval]; exact div_neg_of_neg_of_pos (by norm_num) (by positivity)⟩

/-- Explicit Hypothesis H4 (Part i stationary pullback rate expansion on `c ∈ [1/2, 2]`,
`a13_proof.md` `(4.7)`). -/
def H4_PullbackRateExpansion (r B : ℝ → ℝ) (ε Rg : ℝ) : Prop :=
  ∀ c : ℝ, c ∈ Set.Icc (1 / 2 : ℝ) 2 →
    |r (Real.log c) / ε - (c - 1) + ε * B c| ≤ Rg * ε ^ 2

/-- Explicit Hypothesis H5 (Part i monotone growth exponent `r` with threshold root `r D = 0`,
`a13_proof.md` Proposition 3 & §4 line 182). -/
def H5_MonotoneRoot (r : ℝ → ℝ) (D : ℝ) : Prop :=
  Monotone r ∧ r D = 0

/-- Helper: `|log c - (c - 1)| ≤ 2 * (c - 1) ^ 2` for all `c ≥ 1 / 2` (`a13_proof.md` line 185). -/
private theorem abs_log_sub_sub_one_le_two_sq (c : ℝ) (hc : 1 / 2 ≤ c) :
    |Real.log c - (c - 1)| ≤ 2 * (c - 1) ^ 2 := by
  have hc_pos : 0 < c := by linarith
  have hlog_le : Real.log c ≤ c - 1 := Real.log_le_sub_one_of_pos hc_pos
  have habs : |Real.log c - (c - 1)| = (c - 1) - Real.log c := by
    rw [abs_of_nonpos (by linarith)]
    ring
  rw [habs]
  -- Show `0 ≤ g c` where `g x = log x - (x - 1) + 2 * (x - 1)^2` on `[1/2, ∞)`:
  let g : ℝ → ℝ := fun x => Real.log x - (x - 1) + 2 * (x - 1) ^ 2
  have hg1 : g 1 = 0 := by dsimp [g]; simp
  have hg_deriv : ∀ x : ℝ, 0 < x → HasDerivAt g (((x - 1) * (4 * x - 1)) / x) x := by
    intro x hx
    have hd1 : HasDerivAt Real.log x⁻¹ x := Real.hasDerivAt_log hx.ne'
    have hd2 : HasDerivAt (fun u : ℝ => u - 1) 1 x := (hasDerivAt_id' x).sub_const 1
    have hd3 : HasDerivAt (fun u : ℝ => 2 * (u - 1) ^ 2) (2 * (2 * (x - 1) ^ 1 * 1)) x := by
      simpa using (hd2.pow 2).const_mul 2
    have hcomb := (hd1.sub hd2).add hd3
    have heq : x⁻¹ - 1 + 2 * (2 * (x - 1) ^ 1 * 1) = ((x - 1) * (4 * x - 1)) / x := by
      field_simp [hx.ne']
      ring
    rwa [heq] at hcomb
  suffices h0 : 0 ≤ g c by dsimp [g] at h0; linarith
  rcases le_or_gt c 1 with hc_le1 | hc_gt1
  · -- Antitone on `[1/2, 1]`:
    have hanti : AntitoneOn g (Set.Icc (1 / 2 : ℝ) 1) := by
      refine antitoneOn_of_deriv_nonpos (convex_Icc _ _) ?_ ?_ ?_
      · intro x hx
        exact (hg_deriv x (by linarith [hx.1])).continuousAt.continuousWithinAt
      · intro x hx
        rw [interior_Icc] at hx
        exact (hg_deriv x (by linarith [hx.1])).differentiableAt.differentiableWithinAt
      · intro x hx
        rw [interior_Icc] at hx
        have hx_pos : 0 < x := by linarith [hx.1]
        rw [(hg_deriv x hx_pos).deriv]
        exact div_nonpos_of_nonpos_of_nonneg (mul_nonpos_of_nonpos_of_nonneg (by linarith [hx.2])
          (by linarith [hx.1])) hx_pos.le
    have hge := hanti ⟨hc, hc_le1⟩ ⟨by norm_num, le_rfl⟩ hc_le1
    rwa [hg1] at hge
  · -- Monotone on `[1, ∞)`:
    have hmono : MonotoneOn g (Set.Ici 1) := by
      refine monotoneOn_of_deriv_nonneg (convex_Ici 1) ?_ ?_ ?_
      · intro x hx
        have hx1 : 1 ≤ x := hx
        exact (hg_deriv x (zero_lt_one.trans_le hx1)).continuousAt.continuousWithinAt
      · intro x hx
        rw [interior_Ici] at hx
        change 1 < x at hx
        exact (hg_deriv x (zero_lt_one.trans hx)).differentiableAt.differentiableWithinAt
      · intro x hx
        rw [interior_Ici] at hx
        change 1 < x at hx
        have hx_pos : 0 < x := zero_lt_one.trans hx
        rw [(hg_deriv x hx_pos).deriv]
        exact div_nonneg (mul_nonneg (by linarith [hx.le]) (by linarith [hx.le])) hx_pos.le
    have hge := hmono (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hc_gt1.le) hc_gt1.le
    rwa [hg1] at hge

/-! Dated correction, 2026-11-16 (leanowed): the actual covariance polynomial now
supplies B(1), its compact-interval bound, and its Lipschitz bound. Earlier audit
comments above record the former generic interface. Constants below use the actual
variance V; the fourth-moment assumption of the physical pullback remains in H4. -/
noncomputable def outageB (b V : ℝ) (C : ℕ → ℝ) (c : ℝ) : ℝ :=
  c^2*((∑' m : ℕ, (1-b)^m*C (m+1))+1/b)-c/b+(c^2*(V+1)-2*c+1)/2
noncomputable def outageB0 (b V : ℝ) : ℝ := 4*(V+1)/b+2/b+2*(V+1)+3
noncomputable def outageJ (b V : ℝ) : ℝ := 4*(V+1)/b+2*(V+1)+1/b+1

theorem outage_coefficient_bounds (b V : ℝ) (C : ℕ → ℝ)
    (hb : 0 < b) (hb1 : b < 1) (hV : 0 ≤ V)
    (hC : ∀ m : ℕ, |C (m+1)| ≤ V) :
    outageB b V C 1 = Acoeff (1-b) V C ∧
    0 ≤ outageB0 b V ∧ 0 ≤ outageJ b V ∧
    (∀ c ∈ Set.Icc (1/2:ℝ) 2, |outageB b V C c| ≤ outageB0 b V) ∧
    (∀ c ∈ Set.Icc (1/2:ℝ) 2,
      |outageB b V C c-outageB b V C 1| ≤ outageJ b V*|c-1|) := by
  have hp : 0 ≤ 1-b := by linarith
  have hp1 : 1-b < 1 := by linarith
  have hnorm : ‖1-b‖ < 1 := by rw [Real.norm_eq_abs, abs_of_nonneg hp]; exact hp1
  have hsum := (O9_raw_covariance_bracket_and_alt (1-b) V C hp hp1 hV hC).1.of_abs
  have hg := (summable_geometric_of_norm_lt_one hnorm).mul_left V
  have hs : |∑' m : ℕ, (1-b)^m*C (m+1)| ≤ V/b := by
    have hle : ∀ m : ℕ, |(1-b)^m*C (m+1)| ≤ V*(1-b)^m := by
      intro m
      rw [abs_mul, abs_of_nonneg (pow_nonneg hp _), mul_comm V]
      exact mul_le_mul_of_nonneg_left (hC m) (pow_nonneg hp _)
    have ht : |∑' m : ℕ, (1-b)^m*C (m+1)| ≤ ∑' m : ℕ, V*(1-b)^m := by
      rw [abs_le]
      exact ⟨hasSum_le (fun m => (abs_le.mp (hle m)).1) hg.hasSum.neg hsum.hasSum,
        hasSum_le (fun m => (abs_le.mp (hle m)).2) hsum.hasSum hg.hasSum⟩
    simpa [div_eq_mul_inv, tsum_mul_left, tsum_geometric_of_norm_lt_one hnorm] using ht
  have hS : |(∑' m : ℕ, (1-b)^m*C (m+1))+1/b| ≤ (V+1)/b := by
    have ht := abs_add_le (∑' m : ℕ, (1-b)^m*C (m+1)) (1/b)
    rw [abs_of_pos (one_div_pos.mpr hb)] at ht
    calc
      _ ≤ V/b+1/b := ht.trans (add_le_add hs le_rfl)
      _ = (V+1)/b := by ring
  have h0 : 0 ≤ outageB0 b V := by dsimp [outageB0]; positivity
  have hJ : 0 ≤ outageJ b V := by dsimp [outageJ]; positivity
  refine ⟨by dsimp [outageB, Acoeff]; ring, h0, hJ, ?_, ?_⟩
  · intro c hc
    have hc0 : 0 ≤ c := by linarith [hc.1]
    have hc2 : c^2 ≤ 4 := by nlinarith [hc.2]
    have hterm : |c^2*((∑' m : ℕ, (1-b)^m*C (m+1))+1/b)| ≤ 4*((V+1)/b) := by
      rw [abs_mul, abs_of_nonneg (sq_nonneg _)]
      exact mul_le_mul hc2 hS (abs_nonneg _) (by positivity)
    have hcdiv : |c/b| ≤ 2/b := by rw [abs_of_nonneg (div_nonneg hc0 hb.le)]; gcongr; exact hc.2
    have hquad : |(c^2*(V+1)-2*c+1)/2| ≤ 2*(V+1)+3 := by
      rw [abs_div, abs_of_pos (by norm_num : (0:ℝ)<2)]
      have ht := (abs_add_le (c^2*(V+1)-2*c) 1).trans
        (add_le_add (abs_sub (c^2*(V+1)) (2*c)) (le_refl |(1:ℝ)|))
      rw [abs_one, abs_of_nonneg (by positivity : 0 ≤ c^2*(V+1)),
        abs_of_nonneg (by positivity : 0 ≤ 2*c)] at ht
      have hp := mul_le_mul_of_nonneg_right hc2 (by linarith : 0 ≤ V+1)
      apply (div_le_iff₀ (by norm_num : (0:ℝ)<2)).mpr
      nlinarith [hc.2]
    dsimp [outageB, outageB0]
    have ht := abs_add_le
      (c^2*((∑' m : ℕ, (1-b)^m*C (m+1))+1/b)-c/b)
      ((c^2*(V+1)-2*c+1)/2)
    have ht2 := abs_sub (c^2*((∑' m : ℕ, (1-b)^m*C (m+1))+1/b)) (c/b)
    have hscale : 4*((V+1)/b) = 4*(V+1)/b := by ring
    linarith
  · intro c hc
    have hc0 : 0 ≤ c+1 := by linarith [hc.1]
    have hc3 : c+1 ≤ 3 := by linarith [hc.2]
    have hfac : outageB b V C c-outageB b V C 1 =
      (c-1)*((c+1)*((∑' m : ℕ, (1-b)^m*C (m+1))+1/b+(V+1)/2)-(1/b+1)) := by
      dsimp [outageB]; ring
    have hS2 : |(∑' m : ℕ, (1-b)^m*C (m+1))+1/b+(V+1)/2| ≤ (V+1)/b+(V+1)/2 := by
      apply (abs_add_le _ _).trans
      rw [abs_of_nonneg (by positivity : 0 ≤ (V+1)/2)]
      exact add_le_add hS le_rfl
    have hinner : |(c+1)*((∑' m : ℕ, (1-b)^m*C (m+1))+1/b+(V+1)/2)-(1/b+1)| ≤ outageJ b V := by
      have ht := abs_sub ((c+1)*((∑' m : ℕ, (1-b)^m*C (m+1))+1/b+(V+1)/2)) (1/b+1)
      rw [abs_mul, abs_of_nonneg hc0, abs_of_pos (by positivity : 0 < 1/b+1)] at ht
      have hm := mul_le_mul hc3 hS2 (abs_nonneg _) (by norm_num : (0:ℝ) ≤ 3)
      dsimp [outageJ]
      have hpos : 0 ≤ (V+1)/b := by positivity
      have hscale : 4*((V+1)/b) = 4*(V+1)/b := by ring
      nlinarith
    rw [hfac, abs_mul, mul_comm]
    exact mul_le_mul_of_nonneg_right hinner (abs_nonneg _)

/-- **O10 (`a13_proof.md` §4 Proposition 4, `(4.3)`–`(4.7)`; `a13-check.md` §1.2).**
Algebraic and inequality core of Proposition 4:
1. `[0, Vmax]` invariance of `(4.4)` (`Vmax = b / (b − ε) ≤ 2`) for all `h ≥ 0`;
2. Exact `(δ, η)` algebraic recursions `(4.5)`;
3. `B(1) = A` identity in `(4.7)`;
4. Conditional root localization `|D − A ε| ≤ C ε²` `(4.3)` under explicit hypotheses `H4` and `H5`. -/
theorem O10_part_i_pullback_algebra_and_root_bound (b : ℝ) (hb0 : 0 < b) (hb1 : b < 1) :
    let p := 1 - b
    -- (1) `[0, Vmax]` invariance `(4.4)`
    (∀ (ε c h v : ℝ), 0 < ε → ε ≤ b / 2 → 1 / 2 ≤ c → c ≤ 2 → 0 ≤ h →
      0 ≤ v → v ≤ b / (b - ε) →
        b / (b - ε) ≤ 2 ∧
        0 < 1 - ε ∧
        1 - ε ≤ 1 + ε * (c * h * v - 1) ∧
        0 ≤ (p * v + b) / (1 + ε * (c * h * v - 1)) ∧
        (p * v + b) / (1 + ε * (c * h * v - 1)) ≤ b / (b - ε)) ∧
    -- (2) Exact `(δ, η)` recursions `(4.5)`
    (∀ (ε c h v u : ℝ), 1 + ε * (c * h * v - 1) ≠ 0 →
      let W := c * h * v - 1
      let v_next := (p * v + b) / (1 + ε * W)
      let u_next := p * u - (c * h - 1)
      v_next - 1 = p * (v - 1) - ε * v_next * W ∧
      (v_next - 1 - ε * u_next) =
        p * (v - 1 - ε * u) - ε * ((v_next - 1) * W + c * h * (v - 1))) ∧
    -- (3) `B(1) = Acoeff p V C` `(4.7)`
    (∀ (V : ℝ) (C : ℕ → ℝ), Summable (fun m : ℕ => p ^ m * C (m + 1)) →
      let S := ∑' m : ℕ, p ^ m * (C (m + 1) + 1)
      (1 : ℝ) ^ 2 * S - 1 / b + ((1 : ℝ) ^ 2 * (V + 1) - 2 * 1 + 1) / 2 = Acoeff p V C) ∧
    -- (4) Root localization `|D - A * ε| ≤ C_rem * ε ^ 2` `(4.3)` under `H4` and `H5`
    (∀ (V Rg ε D : ℝ) (C : ℕ → ℝ) (r : ℝ → ℝ),
      0 ≤ V → (∀ m : ℕ, |C (m+1)| ≤ V) →
      let B := outageB b V C
      let B0 := outageB0 b V
      let J := outageJ b V
      let A := Acoeff p V C
      0 ≤ Rg →
      0 < ε → ε ≤ min (b / 2) (1 / (4 * (B0 + Rg + 1))) →
      H4_PullbackRateExpansion r B ε Rg →
      H5_MonotoneRoot r D →
        let K0 := B0 + Rg
        let C_rem := J * K0 + Rg + 2 * K0 ^ 2
        Real.exp D ∈ Set.Ioo (1 / 2 : ℝ) 2 ∧
        |Real.exp D - 1| ≤ K0 * ε ∧
        |Real.exp D - 1 - A * ε| ≤ (J * K0 + Rg) * ε ^ 2 ∧
        |D - A * ε| ≤ C_rem * ε ^ 2) := by
  intro p
  have hp_pos : 0 < p := by dsimp [p]; linarith
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- Part (1): `[0, Vmax]` invariance
    intro ε c h v hε0 hεb hc_lo _hc_hi hh0 hv0 hv_hi
    have hbε : 0 < b - ε := by linarith
    have hVmax_le2 : b / (b - ε) ≤ 2 := by
      rw [div_le_iff₀ hbε]; linarith
    have h1ε_pos : 0 < 1 - ε := by linarith
    have hchv_nn : 0 ≤ c * h * v := mul_nonneg (mul_nonneg (by linarith) hh0) hv0
    have hden_ge : 1 - ε ≤ 1 + ε * (c * h * v - 1) := by nlinarith
    have hden_pos : 0 < 1 + ε * (c * h * v - 1) := lt_of_lt_of_le h1ε_pos hden_ge
    have hnum_nn : 0 ≤ p * v + b := by nlinarith
    have hnext_nn : 0 ≤ (p * v + b) / (1 + ε * (c * h * v - 1)) :=
      div_nonneg hnum_nn hden_pos.le
    have hnext_le : (p * v + b) / (1 + ε * (c * h * v - 1)) ≤ b / (b - ε) := by
      calc (p * v + b) / (1 + ε * (c * h * v - 1))
          ≤ (p * (b / (b - ε)) + b) / (1 - ε) := by
            have h1 : p * v + b ≤ p * (b / (b - ε)) + b := by nlinarith
            have h2 : (p * v + b) / (1 + ε * (c * h * v - 1)) ≤ (p * v + b) / (1 - ε) :=
              div_le_div_of_nonneg_left hnum_nn h1ε_pos hden_ge
            have h3 : (p * v + b) / (1 - ε) ≤ (p * (b / (b - ε)) + b) / (1 - ε) :=
              div_le_div_of_nonneg_right h1 h1ε_pos.le
            exact h2.trans h3
        _ = b / (b - ε) := by
            dsimp [p]
            field_simp [hbε.ne', h1ε_pos.ne']
            ring
    exact ⟨hVmax_le2, h1ε_pos, hden_ge, hnext_nn, hnext_le⟩
  · -- Part (2): Exact `(δ, η)` recursions `(4.5)`
    intro ε c h v u hden_ne
    dsimp only
    set W := c * h * v - 1
    set v_next := (p * v + b) / (1 + ε * W)
    have hmul : v_next * (1 + ε * W) = p * v + b := div_mul_cancel₀ (p * v + b) hden_ne
    have hδ : v_next - 1 = p * (v - 1) - ε * v_next * W := by
      dsimp [p] at hmul ⊢
      linear_combination hmul
    refine ⟨hδ, ?_⟩
    have hW_id : W = c * h - 1 + c * h * (v - 1) := by dsimp [W]; ring
    linear_combination hδ + ε * v_next * hW_id - ε * hW_id
  · -- Part (3): `B(1) = Acoeff p V C`
    intro V C hsumC
    dsimp only
    have hp_norm : ‖p‖ < 1 := by
      rw [Real.norm_eq_abs, abs_of_pos hp_pos]
      dsimp [p]; linarith
    have hgeom : Summable (fun m : ℕ => p ^ m) := summable_geometric_of_norm_lt_one hp_norm
    have hsplit : (fun m : ℕ => p ^ m * (C (m + 1) + 1)) =
        (fun m : ℕ => p ^ m * C (m + 1) + p ^ m) := by ext m; ring
    rw [hsplit, hsumC.tsum_add hgeom, tsum_geometric_of_norm_lt_one hp_norm]
    have h1p : 1 - p = b := by dsimp [p]; ring
    rw [h1p, ← one_div]
    unfold Acoeff
    ring
  · -- Part (4): Root localization `(4.3)`
    intro V Rg ε D C r hV hC B B0 J A hRg hε0 hε_max hrate ⟨hmono, hrD⟩
    have hbds := outage_coefficient_bounds b V C hb0 hb1 hV hC
    have hB1 : B 1 = A := hbds.1
    have hB0 : 0 ≤ B0 := hbds.2.1
    have hJ : 0 ≤ J := hbds.2.2.1
    have hB_bd : ∀ c ∈ Set.Icc (1/2:ℝ) 2, |B c| ≤ B0 := hbds.2.2.2.1
    have hB_lip : ∀ c ∈ Set.Icc (1/2:ℝ) 2, |B c-B 1| ≤ J*|c-1| := hbds.2.2.2.2
    dsimp only
    set K0 := B0 + Rg
    have hdenom_pos : 0 < 4 * (B0 + Rg + 1) := by positivity
    have hε_bound : ε ≤ 1 / (4 * (B0 + Rg + 1)) := le_trans hε_max (min_le_right _ _)
    have hε_prod : ε * (B0 + Rg + 1) ≤ 1 / 4 := by
      have hm := mul_le_mul_of_nonneg_right hε_bound (by linarith : 0 ≤ B0 + Rg + 1)
      have h1 : 1 / (4 * (B0 + Rg + 1)) * (B0 + Rg + 1) = (1 / 4 : ℝ) := by
        field_simp [(by linarith : B0 + Rg + 1 ≠ 0)]
      linarith
    have hε_lt1 : ε < 1 := by nlinarith
    -- Check signs of `r (log (1/2))` and `r (log 2)`:
    have hhalf_mem : (1 / 2 : ℝ) ∈ Set.Icc (1 / 2 : ℝ) 2 := ⟨le_rfl, by norm_num⟩
    have htwo_mem : (2 : ℝ) ∈ Set.Icc (1 / 2 : ℝ) 2 := ⟨by norm_num, le_rfl⟩
    have hr_half_neg : r (Real.log (1 / 2)) < 0 := by
      have h1 := (abs_le.mp (hrate (1 / 2) hhalf_mem)).2
      have hB_half : -B0 ≤ B (1 / 2) := (abs_le.mp (hB_bd (1 / 2) hhalf_mem)).1
      -- `r(log(1/2)) / ε ≤ -1/2 + ε * B0 + Rg * ε ^ 2 ≤ -1/2 + ε * (B0 + Rg) < -1/4 < 0`
      have hq : r (Real.log (1 / 2)) / ε < 0 := by nlinarith
      exact ((div_neg_iff.mp hq).resolve_left (fun h => by linarith [h.2])).1
    have hr_two_pos : 0 < r (Real.log 2) := by
      have h1 := (abs_le.mp (hrate 2 htwo_mem)).1
      have hB_two : B 2 ≤ B0 := (abs_le.mp (hB_bd 2 htwo_mem)).2
      have hq : 0 < r (Real.log 2) / ε := by nlinarith
      exact ((div_pos_iff.mp hq).resolve_right (fun h => by linarith [h.2])).1
    -- By `Monotone r` and `r D = 0`, `log (1/2) < D < log 2`, hence `exp D ∈ (1/2, 2)`:
    have hD_gt : Real.log (1 / 2) < D := by
      by_contra hle
      push Not at hle
      have := hmono hle
      linarith
    have hD_lt : D < Real.log 2 := by
      by_contra hge
      push Not at hge
      have := hmono hge
      linarith
    set c_star := Real.exp D
    have hc_gt : 1 / 2 < c_star := by
      have h := Real.exp_lt_exp.mpr hD_gt
      rwa [Real.exp_log (by norm_num)] at h
    have hc_lt : c_star < 2 := by
      have h := Real.exp_lt_exp.mpr hD_lt
      rwa [Real.exp_log (by norm_num)] at h
    have hc_mem : c_star ∈ Set.Icc (1 / 2 : ℝ) 2 := ⟨hc_gt.le, hc_lt.le⟩
    -- Evaluate `hrate` at `c_star` (`log c_star = D`, `r D = 0`):
    have hr_at_c : |-(c_star - 1) + ε * B c_star| ≤ Rg * ε ^ 2 := by
      have h := hrate c_star hc_mem
      rw [Real.log_exp, hrD, zero_div, zero_sub] at h
      exact h
    have hc_sub_eq : |(c_star - 1) - ε * B c_star| ≤ Rg * ε ^ 2 := by
      have heq : (c_star - 1) - ε * B c_star = -(-(c_star - 1) + ε * B c_star) := by ring
      rw [heq, abs_neg]
      exact hr_at_c
    have hBc_le : |B c_star| ≤ B0 := hB_bd c_star hc_mem
    have hc_K0 : |c_star - 1| ≤ K0 * ε := by
      have htri : |c_star - 1| ≤ |(c_star - 1) - ε * B c_star| + |ε * B c_star| := by
        simpa using abs_add_le ((c_star - 1) - ε * B c_star) (ε * B c_star)
      rw [abs_mul, abs_of_pos hε0] at htri
      have hmulB : ε * |B c_star| ≤ ε * B0 := mul_le_mul_of_nonneg_left hBc_le hε0.le
      have hε2_le : Rg * ε ^ 2 ≤ Rg * ε :=
        mul_le_mul_of_nonneg_left (by nlinarith : ε ^ 2 ≤ ε) hRg
      dsimp [K0]
      linarith
    have hc_A_bound : |c_star - 1 - A * ε| ≤ (J * K0 + Rg) * ε ^ 2 := by
      have hsplit : c_star - 1 - A * ε =
          ((c_star - 1) - ε * B c_star) + ε * (B c_star - B 1) := by
        rw [hB1]; ring
      rw [hsplit]
      have htri := abs_add_le ((c_star - 1) - ε * B c_star) (ε * (B c_star - B 1))
      rw [abs_mul, abs_of_pos hε0] at htri
      have hlip := hB_lip c_star hc_mem
      have hlip_K0 : |B c_star - B 1| ≤ J * (K0 * ε) :=
        hlip.trans (mul_le_mul_of_nonneg_left hc_K0 hJ)
      have hmul_ε : ε * |B c_star - B 1| ≤ J * K0 * ε ^ 2 := by
        have hm := mul_le_mul_of_nonneg_left hlip_K0 hε0.le
        linarith
      linarith
    -- Finally combine with `|log c_star - (c_star - 1)| ≤ 2 * (c_star - 1)^2`:
    have hlog_quad : |D - (c_star - 1)| ≤ 2 * K0 ^ 2 * ε ^ 2 := by
      have h1 : |Real.log c_star - (c_star - 1)| ≤ 2 * (c_star - 1) ^ 2 :=
        abs_log_sub_sub_one_le_two_sq c_star hc_gt.le
      rw [Real.log_exp] at h1
      have hsq : (c_star - 1) ^ 2 ≤ (K0 * ε) ^ 2 := by
        rw [← sq_abs (c_star - 1), ← sq_abs (K0 * ε)]
        exact pow_le_pow_left₀ (abs_nonneg _) (hc_K0.trans (le_abs_self _)) 2
      linarith [h1, hsq]
    have hD_bound : |D - A * ε| ≤ (J * K0 + Rg + 2 * K0 ^ 2) * ε ^ 2 := by
      have hsplit : D - A * ε = (D - (c_star - 1)) + (c_star - 1 - A * ε) := by ring
      rw [hsplit]
      have htri := abs_add_le (D - (c_star - 1)) (c_star - 1 - A * ε)
      linarith
    exact ⟨⟨hc_gt, hc_lt⟩, hc_K0, hc_A_bound, hD_bound⟩

/-- **O11 (`a13_proof.md` §4 Proposition 5, `(4.8)`; `a13-check.md` §1.2 & §4.1 item 3).**
Unconditional algebraic sign and fixed-factor stay-reversal criterion:
Suppose `|D s − A / s| ≤ C / s²` for all `s` with `0 < 1 / s ≤ ε₀`. Whenever `A ≠ 0`:
1. If `0 < 1 / s ≤ ε₀` and `C * (1 / s) < |A|`, then `sign(D s) = sign(A)`;
2. For every fixed `κ > 1` and `s` satisfying `0 < 1 / s ≤ ε₀` and
   `C * (1 + 1 / κ ^ 2) * (1 / s) < |A| * (1 − 1 / κ)` (equivalently
   `1 / s < |A| * (1 − 1 / κ) / (C * (1 + 1 / κ ^ 2))` when `C > 0`),
   `sign(D (κ * s) − D s) = −sign(A)` (`A < 0 → D s < D (κ * s)` and `A > 0 → D (κ * s) < D s`). -/
theorem O11_part_i_sign_and_fixed_factor_reversal (D : ℝ → ℝ) (A C ε0 : ℝ)
    (hA : A ≠ 0) (_hC : 0 ≤ C)
    (hrem : ∀ s : ℝ, 0 < 1 / s → 1 / s ≤ ε0 → |D s - A / s| ≤ C / s ^ 2) :
    (∀ s : ℝ, 0 < 1 / s → 1 / s ≤ ε0 → C * (1 / s) < |A| →
      Real.sign (D s) = Real.sign A ∧ (0 < A → 0 < D s) ∧ (A < 0 → D s < 0)) ∧
    (∀ (κ s : ℝ), 1 < κ → 0 < 1 / s → 1 / s ≤ ε0 →
      C * (1 + 1 / κ ^ 2) * (1 / s) < |A| * (1 - 1 / κ) →
        Real.sign (D (κ * s) - D s) = -Real.sign A ∧
        (A < 0 → D s < D (κ * s)) ∧
        (0 < A → D (κ * s) < D s)) ∧
    (0 < C → ∀ (κ s : ℝ), 1 < κ → 0 < 1 / s → 1 / s ≤ ε0 →
      1 / s < |A| * (1 - 1 / κ) / (C * (1 + 1 / κ ^ 2)) →
        Real.sign (D (κ * s) - D s) = -Real.sign A) := by
  have hmain : (∀ s : ℝ, 0 < 1 / s → 1 / s ≤ ε0 → C * (1 / s) < |A| →
      Real.sign (D s) = Real.sign A ∧ (0 < A → 0 < D s) ∧ (A < 0 → D s < 0)) ∧
    (∀ (κ s : ℝ), 1 < κ → 0 < 1 / s → 1 / s ≤ ε0 →
      C * (1 + 1 / κ ^ 2) * (1 / s) < |A| * (1 - 1 / κ) →
        Real.sign (D (κ * s) - D s) = -Real.sign A ∧
        (A < 0 → D s < D (κ * s)) ∧
        (0 < A → D (κ * s) < D s)) := by
    constructor
    · intro s hs_inv hs_le hCA
      have hs0 : 0 < s := one_div_pos.mp hs_inv
      have hbd := abs_le.mp (hrem s hs_inv hs_le)
      have herr_lt : C / s ^ 2 < |A| / s := by
        have hm := mul_lt_mul_of_pos_right hCA hs_inv
        have hl : C * (1 / s) * (1 / s) = C / s ^ 2 := by field_simp [hs0.ne']
        have hr : |A| * (1 / s) = |A| / s := by ring
        linarith
      rcases lt_or_gt_of_ne hA with hA_neg | hA_pos
      · have habsA : |A| = -A := abs_of_neg hA_neg
        have hDs_neg : D s < 0 := by
          have h1 : -(A / s) = |A| / s := by rw [habsA, neg_div]
          linarith [hbd.2, herr_lt, h1]
        refine ⟨by rw [Real.sign_of_neg hDs_neg, Real.sign_of_neg hA_neg],
          fun h => by linarith, fun _ => hDs_neg⟩
      · have habsA : |A| = A := abs_of_pos hA_pos
        have hDs_pos : 0 < D s := by
          have h1 : A / s = |A| / s := by rw [habsA]
          linarith [hbd.1, herr_lt, h1]
        refine ⟨by rw [Real.sign_of_pos hDs_pos, Real.sign_of_pos hA_pos],
          fun _ => hDs_pos, fun h => by linarith⟩
    · intro κ s hκ hs_inv hs_le hcond
      have hs0 : 0 < s := one_div_pos.mp hs_inv
      have hκ0 : 0 < κ := zero_lt_one.trans hκ
      have hκs_pos : 0 < κ * s := mul_pos hκ0 hs0
      have hκs_inv_pos : 0 < 1 / (κ * s) := one_div_pos.mpr hκs_pos
      have hκs_le : 1 / (κ * s) ≤ ε0 := by
        have hlt : 1 / (κ * s) < 1 / s := one_div_lt_one_div_of_lt hs0 (by nlinarith)
        exact hlt.le.trans hs_le
      have hbd1 := hrem s hs_inv hs_le
      have hbd2 := hrem (κ * s) hκs_inv_pos hκs_le
      -- `(D (κ * s) - D s) - (-A * (1 - 1 / κ) / s) = (D (κ * s) - A / (κ * s)) - (D s - A / s)`
      have hdiff_id : (D (κ * s) - D s) - (-A * (1 - 1 / κ) * (1 / s)) =
          (D (κ * s) - A / (κ * s)) - (D s - A / s) := by
        field_simp [hs0.ne', hκ0.ne']
        ring
      have herr_sum : |(D (κ * s) - D s) - (-A * (1 - 1 / κ) * (1 / s))| ≤
          C * (1 + 1 / κ ^ 2) * (1 / s) * (1 / s) := by
        rw [hdiff_id]
        have htri := abs_sub (D (κ * s) - A / (κ * s)) (D s - A / s)
        have hsum_eq : C / (κ * s) ^ 2 + C / s ^ 2 =
            C * (1 + 1 / κ ^ 2) * (1 / s) * (1 / s) := by
          field_simp [hs0.ne', hκ0.ne']
          ring
        linarith [htri, hbd1, hbd2, hsum_eq]
      have herr_lt : C * (1 + 1 / κ ^ 2) * (1 / s) * (1 / s) <
          |A| * (1 - 1 / κ) * (1 / s) :=
        mul_lt_mul_of_pos_right hcond hs_inv
      have hbd_diff := abs_le.mp herr_sum
      have h1κ_pos : 0 < 1 - 1 / κ := by
        rw [sub_pos, div_lt_one hκ0]; exact hκ
      rcases lt_or_gt_of_ne hA with hA_neg | hA_pos
      · have habsA : |A| = -A := abs_of_neg hA_neg
        have hpos_diff : 0 < D (κ * s) - D s := by
          have h1 : -A * (1 - 1 / κ) * (1 / s) = |A| * (1 - 1 / κ) * (1 / s) := by
            rw [habsA]
          linarith [hbd_diff.1, herr_lt, h1]
        refine ⟨by rw [Real.sign_of_pos hpos_diff, Real.sign_of_neg hA_neg]; norm_num,
          fun _ => sub_pos.mp hpos_diff, fun h => by linarith⟩
      · have habsA : |A| = A := abs_of_pos hA_pos
        have hneg_diff : D (κ * s) - D s < 0 := by
          have h1 : -A * (1 - 1 / κ) * (1 / s) = -(|A| * (1 - 1 / κ) * (1 / s)) := by
            rw [habsA]; ring
          linarith [hbd_diff.2, herr_lt, h1]
        refine ⟨by simp [Real.sign_of_neg hneg_diff, Real.sign_of_pos hA_pos],
          fun h => by linarith, fun _ => sub_neg.mp hneg_diff⟩
  refine ⟨hmain.1, hmain.2, ?_⟩
  intro hCpos κ s hκ hs_inv hs_le hquot
  have hκ0 : 0 < κ := zero_lt_one.trans hκ
  have hden_pos : 0 < C * (1 + 1 / κ ^ 2) := by positivity
  have hcond : C * (1 + 1 / κ ^ 2) * (1 / s) < |A| * (1 - 1 / κ) := by
    have hm := (lt_div_iff₀ hden_pos).mp hquot
    linarith
  exact (hmain.2 κ s hκ hs_inv hs_le hcond).1

/-- Helper: `Real.exp (1 / 4) < 129 / 100`. -/
private theorem exp_quarter_lt : Real.exp (1 / 4 : ℝ) < 129 / 100 := by
  have h := Real.exp_bound' (x := (1 / 4 : ℝ)) (by norm_num) (by norm_num) (n := 4) (by norm_num)
  norm_num [Finset.sum_range_succ, Nat.factorial] at h
  linarith

/-- Helper: `3 / 2 - Real.exp (1 / 4) ≤ ‖Complex.exp D - 1‖` on `‖D‖ = 1 / 4`. -/
private theorem rouche_circle_lower (D : ℂ) (hD : ‖D‖ = 1 / 4) :
    3 / 2 - Real.exp (1 / 4 : ℝ) ≤ ‖Complex.exp D - 1‖ := by
  have hsumC : Summable (fun n : ℕ => D ^ n / (n.factorial : ℂ)) := by
    refine Summable.of_norm ?_
    have hnorm : (fun n : ℕ => ‖D ^ n / (n.factorial : ℂ)‖) =
        (fun n : ℕ => ‖D‖ ^ n / (n.factorial : ℝ)) := by
      ext n; simp [norm_div]
    rw [hnorm]
    exact Real.summable_pow_div_factorial ‖D‖
  have hhasC : HasSum (fun n : ℕ => D ^ n / (n.factorial : ℂ)) (Complex.exp D) := by
    rw [Complex.exp_eq_exp_ℂ, NormedSpace.exp_eq_tsum_div]
    exact hsumC.hasSum
  have hhasR : HasSum (fun n : ℕ => ‖D‖ ^ n / (n.factorial : ℝ)) (Real.exp ‖D‖) := by
    rw [Real.exp_eq_exp_ℝ, NormedSpace.exp_eq_tsum_div]
    exact (Real.summable_pow_div_factorial ‖D‖).hasSum
  have hhasC2 := (hasSum_nat_add_iff' 2).mpr hhasC
  have hhasR2 := (hasSum_nat_add_iff' 2).mpr hhasR
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one,
    Nat.factorial_zero, Nat.factorial_one, Nat.cast_one, div_one] at hhasC2 hhasR2
  have htail_le : ‖Complex.exp D - (1 + D)‖ ≤ Real.exp ‖D‖ - (1 + ‖D‖) := by
    refine le_of_tendsto_of_tendsto
      (continuous_norm.continuousAt.tendsto.comp hhasC2.tendsto_sum_nat)
      hhasR2.tendsto_sum_nat ?_
    exact Filter.Eventually.of_forall fun m => by
      calc ‖∑ k ∈ Finset.range m, D ^ (k + 2) / ((k + 2).factorial : ℂ)‖
          ≤ ∑ k ∈ Finset.range m, ‖D ^ (k + 2) / ((k + 2).factorial : ℂ)‖ := norm_sum_le _ _
        _ = ∑ k ∈ Finset.range m, ‖D‖ ^ (k + 2) / ((k + 2).factorial : ℝ) := by
            refine Finset.sum_congr rfl fun k _ => by simp [norm_div]
  have htri : ‖D‖ ≤ ‖Complex.exp D - 1‖ + ‖Complex.exp D - (1 + D)‖ := by
    have heq : D = (Complex.exp D - 1) - (Complex.exp D - (1 + D)) := by ring
    calc ‖D‖ = ‖(Complex.exp D - 1) - (Complex.exp D - (1 + D))‖ := congr_arg norm heq
      _ ≤ ‖Complex.exp D - 1‖ + ‖Complex.exp D - (1 + D)‖ := norm_sub_le _ _
  rw [hD] at htail_le htri
  linarith

/-- **O12 (`a13_proof.md` §5 Proposition 6, `(5.3)` & Rouché separation; `a13-check.md` §1.2).** -/
theorem O12_bounded_host_disk_and_rouche (b M : ℝ) (hb0 : 0 < b) (hb1 : b < 1) (hM : 1 ≤ M) :
    let p := 1 - b
    let R := b / (256 * M ^ 2)
    let K := 8 * M / b
    0 < R ∧
    K * R ≤ 1 / 32 ∧
    0 < 1 - 4 * M * R ∧
    (p * K + 4 * M) / (1 - 4 * M * R) ≤ K ∧
    (p + R * (p + 2 * b * M)) / (1 - 4 * M * R) ^ 2 ≤ (1 - 61 * b / 64) / (1 - b / 8) ∧
    (1 - 61 * b / 64) / (1 - b / 8) < 1 ∧
    ((25 * M ^ 2 / b) * R = 25 / 256 ∧
      (25 / 256 : ℝ) < 3 / 2 - Real.exp (1 / 4 : ℝ) ∧
      ∀ D : ℂ, ‖D‖ = 1 / 4 → 3 / 2 - Real.exp (1 / 4 : ℝ) ≤ ‖Complex.exp D - 1‖) ∧
    (∀ (ε D z : ℂ) (h : ℝ),
      ‖ε‖ ≤ R → ‖D‖ ≤ 1 / 4 → 0 ≤ h → h ≤ M → ‖z‖ ≤ K * ‖ε‖ →
        let W : ℂ := Complex.exp D * (h : ℂ) * (1 + z) - 1
        ‖W‖ < 4 * M ∧
        1 + ε * W ≠ 0 ∧
        ‖((p : ℂ) * z - ε * W) / (1 + ε * W)‖ ≤ K * ‖ε‖ ∧
        ‖((p : ℂ) * (1 - ε) - ε * (b : ℂ) * Complex.exp D * (h : ℂ)) / (1 + ε * W) ^ 2‖ ≤
          (p + R * (p + 2 * b * M)) / (1 - 4 * M * R) ^ 2) := by
  intro p R K
  have hM_pos : 0 < M := lt_of_lt_of_le zero_lt_one hM
  have hp_pos : 0 < p := by dsimp [p]; linarith
  have hR_pos : 0 < R := by dsimp [R]; positivity
  have hK_pos : 0 < K := by dsimp [K]; positivity
  have hKR_eq : K * R = 1 / (32 * M) := by
    dsimp [K, R]; field_simp [hb0.ne', hM_pos.ne']; ring
  have hKR_le : K * R ≤ 1 / 32 := by
    rw [hKR_eq]
    exact one_div_le_one_div_of_le (by norm_num) (by linarith)
  have h4MR_eq : 4 * M * R = b / (64 * M) := by
    dsimp [R]; field_simp [hM_pos.ne']; ring
  have h4MR_le : 4 * M * R ≤ b / 64 := by
    rw [h4MR_eq]
    exact div_le_div_of_nonneg_left hb0.le (by norm_num) (by linarith)
  have h1_4MR : 0 < 1 - 4 * M * R := by linarith
  have hball_scalar : (p * K + 4 * M) / (1 - 4 * M * R) ≤ K := by
    rw [div_le_iff₀ h1_4MR]
    have hbK : b * K = 8 * M := by dsimp [K]; field_simp [hb0.ne']
    have hpK : p * K = K - 8 * M := by dsimp [p]; linear_combination -hbK
    have h4MKR : 4 * M * (K * R) ≤ M / 8 := by nlinarith [hKR_le]
    nlinarith [hpK, h4MKR]
  have h2bMR : R * (p + 2 * b * M) ≤ 3 * b / 64 := by
    have hRp : R * p ≤ b / 256 := by
      have hR_le : R ≤ b / 256 := by
        dsimp [R]
        exact div_le_div_of_nonneg_left hb0.le (by norm_num) (by nlinarith)
      have hp1 : p ≤ 1 := by dsimp [p]; linarith
      nlinarith
    have hR2bM : R * (2 * b * M) ≤ b / 128 := by
      have h1 : R * (2 * b * M) = b ^ 2 / (128 * M) := by
        dsimp [R]; field_simp [hM_pos.ne']; ring
      rw [h1]
      rw [div_le_div_iff₀ (by positivity) (by norm_num)]
      nlinarith
    linarith
  have h1_b8 : 0 < 1 - b / 8 := by linarith
  have hsq_den : 1 - b / 8 ≤ (1 - 4 * M * R) ^ 2 := by
    have h1 : 1 - b / 32 ≤ (1 - b / 64) ^ 2 := by nlinarith [sq_nonneg (b / 64)]
    have h2 : (1 - b / 64) ^ 2 ≤ (1 - 4 * M * R) ^ 2 :=
      pow_le_pow_left₀ (by linarith) (by linarith) 2
    linarith
  have hcontr_le : (p + R * (p + 2 * b * M)) / (1 - 4 * M * R) ^ 2 ≤
      (1 - 61 * b / 64) / (1 - b / 8) := by
    have hnum_le : p + R * (p + 2 * b * M) ≤ 1 - 61 * b / 64 := by
      dsimp [p]; linarith
    have h1 : (p + R * (p + 2 * b * M)) / (1 - 4 * M * R) ^ 2 ≤
        (1 - 61 * b / 64) / (1 - 4 * M * R) ^ 2 :=
      div_le_div_of_nonneg_right hnum_le (sq_nonneg _)
    have hnum2_nn : 0 ≤ 1 - 61 * b / 64 := by linarith
    have h2 : (1 - 61 * b / 64) / (1 - 4 * M * R) ^ 2 ≤ (1 - 61 * b / 64) / (1 - b / 8) :=
      div_le_div_of_nonneg_left hnum2_nn h1_b8 hsq_den
    exact h1.trans h2
  have hcontr_lt1 : (1 - 61 * b / 64) / (1 - b / 8) < 1 := by
    rw [div_lt_one h1_b8]; linarith
  have hrouche_coeff : (25 * M ^ 2 / b) * R = 25 / 256 := by
    dsimp [R]; field_simp [hb0.ne', hM_pos.ne']
  have hrouche_num : (25 / 256 : ℝ) < 3 / 2 - Real.exp (1 / 4 : ℝ) := by
    linarith [exp_quarter_lt]
  refine ⟨hR_pos, hKR_le, h1_4MR, hball_scalar, hcontr_le, hcontr_lt1,
    ⟨hrouche_coeff, hrouche_num, rouche_circle_lower⟩, ?_⟩
  intro ε D z h hε hD hh0 hhM hz W
  have hexpD : ‖Complex.exp D‖ ≤ Real.exp (1 / 4 : ℝ) := by
    rw [Complex.norm_exp]
    exact Real.exp_le_exp.mpr ((Complex.re_le_norm D).trans hD)
  have hexp_lt : Real.exp (1 / 4 : ℝ) < 129 / 100 := exp_quarter_lt
  have hz_bound : ‖z‖ ≤ 1 / 32 := by
    calc ‖z‖ ≤ K * ‖ε‖ := hz
      _ ≤ K * R := mul_le_mul_of_nonneg_left hε hK_pos.le
      _ ≤ 1 / 32 := hKR_le
  have h1z : ‖1 + z‖ ≤ 33 / 32 := by
    have ht := norm_add_le (1 : ℂ) z
    rw [norm_one] at ht
    linarith
  have hh_norm : ‖(h : ℂ)‖ = h := by simp [abs_of_nonneg hh0]
  have hW_lt : ‖W‖ < 4 * M := by
    dsimp [W]
    have ht := norm_sub_le (Complex.exp D * (h : ℂ) * (1 + z)) 1
    rw [norm_one, norm_mul, norm_mul, hh_norm] at ht
    have hprod : ‖Complex.exp D‖ * h * ‖1 + z‖ ≤ (129 / 100) * M * (33 / 32) := by
      have h1 : ‖Complex.exp D‖ * h ≤ (129 / 100) * M :=
        mul_le_mul (hexpD.trans hexp_lt.le) hhM hh0 (by norm_num)
      exact mul_le_mul h1 h1z (norm_nonneg _) (by positivity)
    linarith
  have hεW_le : ‖ε * W‖ ≤ 4 * M * R := by
    rw [norm_mul]
    calc ‖ε‖ * ‖W‖ ≤ R * (4 * M) :=
          mul_le_mul hε hW_lt.le (norm_nonneg _) hR_pos.le
      _ = 4 * M * R := by ring
  have hden_lb : 1 - 4 * M * R ≤ ‖1 + ε * W‖ := by
    have ht := norm_sub_norm_le (1 : ℂ) (-(ε * W))
    simp only [norm_one, norm_neg, sub_neg_eq_add] at ht
    linarith
  have hden_pos : 0 < ‖1 + ε * W‖ := lt_of_lt_of_le h1_4MR hden_lb
  have hden_ne : 1 + ε * W ≠ 0 := norm_ne_zero_iff.mp hden_pos.ne'
  have hp_norm : ‖(p : ℂ)‖ = p := by simp [abs_of_pos hp_pos]
  have hb_norm : ‖(b : ℂ)‖ = b := by simp [abs_of_pos hb0]
  have hmap_le : ‖((p : ℂ) * z - ε * W) / (1 + ε * W)‖ ≤ K * ‖ε‖ := by
    rw [norm_div]
    have hnum : ‖(p : ℂ) * z - ε * W‖ ≤ (p * K + 4 * M) * ‖ε‖ := by
      have ht := norm_sub_le ((p : ℂ) * z) (ε * W)
      rw [norm_mul, hp_norm, norm_mul] at ht
      have h1 : p * ‖z‖ ≤ p * (K * ‖ε‖) := mul_le_mul_of_nonneg_left hz hp_pos.le
      have h2 : ‖ε‖ * ‖W‖ ≤ 4 * M * ‖ε‖ := by
        nlinarith [hW_lt.le, norm_nonneg ε]
      linarith
    calc ‖(p : ℂ) * z - ε * W‖ / ‖1 + ε * W‖
        ≤ ((p * K + 4 * M) * ‖ε‖) / (1 - 4 * M * R) :=
          div_le_div₀ (by positivity) hnum h1_4MR hden_lb
      _ = ((p * K + 4 * M) / (1 - 4 * M * R)) * ‖ε‖ := by ring
      _ ≤ K * ‖ε‖ := mul_le_mul_of_nonneg_right hball_scalar (norm_nonneg ε)
  have hderiv_le :
      ‖((p : ℂ) * (1 - ε) - ε * (b : ℂ) * Complex.exp D * (h : ℂ)) / (1 + ε * W) ^ 2‖ ≤
        (p + R * (p + 2 * b * M)) / (1 - 4 * M * R) ^ 2 := by
    rw [norm_div, norm_pow]
    have hnum : ‖(p : ℂ) * (1 - ε) - ε * (b : ℂ) * Complex.exp D * (h : ℂ)‖ ≤
        p + R * (p + 2 * b * M) := by
      have heq : (p : ℂ) * (1 - ε) - ε * (b : ℂ) * Complex.exp D * (h : ℂ) =
          (p : ℂ) - ε * ((p : ℂ) + (b : ℂ) * Complex.exp D * (h : ℂ)) := by ring
      rw [heq]
      have ht := norm_sub_le (p : ℂ) (ε * ((p : ℂ) + (b : ℂ) * Complex.exp D * (h : ℂ)))
      rw [hp_norm, norm_mul] at ht
      have hin : ‖(p : ℂ) + (b : ℂ) * Complex.exp D * (h : ℂ)‖ ≤ p + 2 * b * M := by
        have ht2 := norm_add_le (p : ℂ) ((b : ℂ) * Complex.exp D * (h : ℂ))
        rw [hp_norm, norm_mul, norm_mul, hb_norm, hh_norm] at ht2
        have hexp2 : ‖Complex.exp D‖ ≤ 2 := by linarith
        have hbh : b * ‖Complex.exp D‖ * h ≤ b * 2 * M := by
          have h1 : b * ‖Complex.exp D‖ ≤ b * 2 := mul_le_mul_of_nonneg_left hexp2 hb0.le
          exact mul_le_mul h1 hhM hh0 (by positivity)
        linarith
      have hmul : ‖ε‖ * ‖(p : ℂ) + (b : ℂ) * Complex.exp D * (h : ℂ)‖ ≤ R * (p + 2 * b * M) :=
        mul_le_mul hε hin (norm_nonneg _) hR_pos.le
      linarith
    exact div_le_div₀ (by positivity) hnum (sq_pos_of_pos h1_4MR)
      (pow_le_pow_left₀ h1_4MR.le hden_lb 2)
  exact ⟨hW_lt, hden_ne, hmap_le, hderiv_le⟩

/-- Explicit Hypothesis H6 (Part i bounded-host analytic Cauchy series representation on `(0, R/2]`,
`a13_proof.md` Proposition 6 lines 250–261; `appendices/C-mechanisms.tex` lines 843–848). -/
def H6_BoundedHostAnalyticSeries (D_eps : ℝ → ℝ) (A R : ℝ) : Prop :=
  ∃ d : ℕ → ℝ,
    (∀ n : ℕ, 2 ≤ n → |d n| ≤ (1 / 4 : ℝ) * R ^ (-(n : ℤ))) ∧
    ∀ ε : ℝ, 0 < ε → ε ≤ R / 2 →
      D_eps ε = A * ε + ∑' k : ℕ, d (k + 2) * ε ^ (k + 2) ∧
      HasDerivAt D_eps (A + ∑' k : ℕ, ((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)) ε

/-- **O13 (`a13_proof.md` §5 Proposition 6, `(5.1)`–`(5.2)`; `a13-check.md` §1.2 & §4.1 item 4).** -/
theorem O13_bounded_host_cauchy_and_deriv_sign (R : ℝ) (hR : 0 < R) :
    -- (1) Unconditional Cauchy geometric & derivative tail bounds
    (∀ (d : ℕ → ℝ) (ε : ℝ),
      (∀ n : ℕ, 2 ≤ n → |d n| ≤ (1 / 4 : ℝ) * R ^ (-(n : ℤ))) →
      0 < ε → ε ≤ R / 2 →
        Summable (fun k : ℕ => d (k + 2) * ε ^ (k + 2)) ∧
        Summable (fun k : ℕ => ((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)) ∧
        |∑' k : ℕ, d (k + 2) * ε ^ (k + 2)| ≤ ε ^ 2 / (2 * R ^ 2) ∧
        |∑' k : ℕ, ((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)| ≤ 2 * ε / R ^ 2) ∧
    -- (2) Conditional on `H6`, derive `(5.1)` and `(5.2)` (`sign(dD/ds) = -sign(A)` and `sign(D) = sign(A)`)
    (∀ (D_eps : ℝ → ℝ) (A : ℝ),
      H6_BoundedHostAnalyticSeries D_eps A R →
        (∀ ε : ℝ, 0 < ε → ε ≤ R / 2 →
          |D_eps ε - A * ε| ≤ ε ^ 2 / (2 * R ^ 2) ∧
          |deriv D_eps ε - A| ≤ 2 * ε / R ^ 2) ∧
        (∀ s : ℝ, 0 < 1 / s → 1 / s ≤ R / 2 →
          DifferentiableAt ℝ (fun u : ℝ => D_eps (1 / u)) s ∧
          |s ^ 2 * deriv (fun u : ℝ => D_eps (1 / u)) s + A| ≤ 2 / (s * R ^ 2)) ∧
        (A ≠ 0 → ∀ s : ℝ, 0 < 1 / s → 1 / s < min (R / 2) (|A| * R ^ 2 / 4) →
          Real.sign (deriv (fun u : ℝ => D_eps (1 / u)) s) = -Real.sign A ∧
          Real.sign (D_eps (1 / s)) = Real.sign A ∧
          (A < 0 → 0 < deriv (fun u : ℝ => D_eps (1 / u)) s ∧ D_eps (1 / s) < 0) ∧
          (0 < A → deriv (fun u : ℝ => D_eps (1 / u)) s < 0 ∧ 0 < D_eps (1 / s)))) := by
  have hk2_le : ∀ k : ℕ, ((k + 2 : ℕ) : ℝ) ≤ 2 * (3 / 2 : ℝ) ^ k := by
    intro k
    induction k with
    | zero => norm_num
    | succ k ih =>
      have hk_nn : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg k
      have hstep : ((k + 1 + 2 : ℕ) : ℝ) ≤ (3 / 2 : ℝ) * ((k + 2 : ℕ) : ℝ) := by
        push_cast; linarith
      calc ((k + 1 + 2 : ℕ) : ℝ)
          ≤ (3 / 2 : ℝ) * ((k + 2 : ℕ) : ℝ) := hstep
        _ ≤ (3 / 2 : ℝ) * (2 * (3 / 2 : ℝ) ^ k) := mul_le_mul_of_nonneg_left ih (by norm_num)
        _ = 2 * (3 / 2 : ℝ) ^ (k + 1) := by rw [pow_succ]; ring
  have htails : ∀ (d : ℕ → ℝ) (ε : ℝ),
      (∀ n : ℕ, 2 ≤ n → |d n| ≤ (1 / 4 : ℝ) * R ^ (-(n : ℤ))) →
      0 < ε → ε ≤ R / 2 →
        Summable (fun k : ℕ => d (k + 2) * ε ^ (k + 2)) ∧
        Summable (fun k : ℕ => ((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)) ∧
        |∑' k : ℕ, d (k + 2) * ε ^ (k + 2)| ≤ ε ^ 2 / (2 * R ^ 2) ∧
        |∑' k : ℕ, ((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)| ≤ 2 * ε / R ^ 2 := by
    intro d ε hd hε0 hεR
    let x := ε / R
    have hx0 : 0 < x := div_pos hε0 hR
    have hx_half : x ≤ 1 / 2 := by
      dsimp [x]; rw [div_le_iff₀ hR]; linarith
    have hx_norm : ‖x‖ < 1 := by
      rw [Real.norm_eq_abs, abs_of_pos hx0]; linarith
    -- Value series majorant `(1/4 * x^2) * x^k`:
    have hmaj1_sum : Summable (fun k : ℕ => ((1 / 4 : ℝ) * x ^ 2) * x ^ k) :=
      (summable_geometric_of_norm_lt_one hx_norm).mul_left ((1 / 4 : ℝ) * x ^ 2)
    have hterm1 : ∀ k : ℕ, |d (k + 2) * ε ^ (k + 2)| ≤ ((1 / 4 : ℝ) * x ^ 2) * x ^ k := by
      intro k
      rw [abs_mul, abs_of_pos (pow_pos hε0 _)]
      have hdk := hd (k + 2) (by omega)
      have hpowR : R ^ (-(k + 2 : ℕ) : ℤ) * ε ^ (k + 2) = x ^ (k + 2) := by
        rw [zpow_neg, zpow_natCast, div_pow]
        field_simp [hR.ne']
      calc |d (k + 2)| * ε ^ (k + 2)
          ≤ ((1 / 4 : ℝ) * R ^ (-(k + 2 : ℕ) : ℤ)) * ε ^ (k + 2) :=
            mul_le_mul_of_nonneg_right hdk (pow_nonneg hε0.le _)
        _ = (1 / 4 : ℝ) * (R ^ (-(k + 2 : ℕ) : ℤ) * ε ^ (k + 2)) := by ring
        _ = (1 / 4 : ℝ) * x ^ (k + 2) := by rw [hpowR]
        _ = ((1 / 4 : ℝ) * x ^ 2) * x ^ k := by rw [pow_add]; ring
    have hsum1_abs : Summable (fun k : ℕ => |d (k + 2) * ε ^ (k + 2)|) :=
      Summable.of_nonneg_of_le (fun _ => abs_nonneg _) hterm1 hmaj1_sum
    have hsum1 : Summable (fun k : ℕ => d (k + 2) * ε ^ (k + 2)) := hsum1_abs.of_abs
    have hbd1 : |∑' k : ℕ, d (k + 2) * ε ^ (k + 2)| ≤ ε ^ 2 / (2 * R ^ 2) := by
      have hle := abs_tsum_le_of_le hsum1 hmaj1_sum hterm1
      rw [tsum_mul_left, tsum_geometric_of_norm_lt_one hx_norm] at hle
      have h1x : 1 / 2 ≤ 1 - x := by linarith
      have hinv : (1 - x)⁻¹ ≤ 2 := by
        rw [← one_div]
        exact le_trans (one_div_le_one_div_of_le (by norm_num) h1x) (by norm_num)
      have hx2_eq : (1 / 4 : ℝ) * x ^ 2 * 2 = ε ^ 2 / (2 * R ^ 2) := by
        dsimp [x]; field_simp [hR.ne']; ring
      have hmul : (1 / 4 : ℝ) * x ^ 2 * (1 - x)⁻¹ ≤ (1 / 4 : ℝ) * x ^ 2 * 2 :=
        mul_le_mul_of_nonneg_left hinv (by positivity)
      linarith
    -- Derivative series majorant `(ε / (2 * R^2)) * (3/4)^k`:
    have h34_norm : ‖(3 / 4 : ℝ)‖ < 1 := by norm_num
    have hmaj2_sum : Summable (fun k : ℕ => (ε / (2 * R ^ 2)) * (3 / 4 : ℝ) ^ k) :=
      (summable_geometric_of_norm_lt_one h34_norm).mul_left (ε / (2 * R ^ 2))
    have hterm2 : ∀ k : ℕ,
        |((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)| ≤
          (ε / (2 * R ^ 2)) * (3 / 4 : ℝ) ^ k := by
      intro k
      have hk_pos : (0 : ℝ) < ((k + 2 : ℕ) : ℝ) := by positivity
      rw [abs_mul, abs_mul, abs_of_pos hk_pos, abs_of_pos (pow_pos hε0 _)]
      have hdk := hd (k + 2) (by omega)
      have hpow_eq : (1 / 4 : ℝ) * R ^ (-(k + 2 : ℕ) : ℤ) * ε ^ (k + 1) =
          (ε / (4 * R ^ 2)) * x ^ k := by
        change (1 / 4 : ℝ) * R ^ (-(↑(k + 2 : ℕ) : ℤ)) * ε ^ (k + 1) =
          (ε / (4 * R ^ 2)) * (ε / R) ^ k
        rw [zpow_neg, zpow_natCast, div_pow, pow_add]
        simp only [pow_succ]
        field_simp [hR.ne']
      have hxk_le : ((k + 2 : ℕ) : ℝ) * x ^ k ≤ 2 * (3 / 4 : ℝ) ^ k := by
        have hxpow : x ^ k ≤ (1 / 2 : ℝ) ^ k := pow_le_pow_left₀ hx0.le hx_half k
        calc ((k + 2 : ℕ) : ℝ) * x ^ k
            ≤ (2 * (3 / 2 : ℝ) ^ k) * (1 / 2 : ℝ) ^ k :=
              mul_le_mul (hk2_le k) hxpow (pow_nonneg hx0.le k) (by positivity)
          _ = 2 * ((3 / 2 : ℝ) * (1 / 2 : ℝ)) ^ k := by rw [mul_assoc, ← mul_pow]
          _ = 2 * (3 / 4 : ℝ) ^ k := by norm_num
      calc ((k + 2 : ℕ) : ℝ) * |d (k + 2)| * ε ^ (k + 1)
          ≤ ((k + 2 : ℕ) : ℝ) * ((1 / 4 : ℝ) * R ^ (-(k + 2 : ℕ) : ℤ)) * ε ^ (k + 1) := by
            gcongr
        _ = (ε / (4 * R ^ 2)) * (((k + 2 : ℕ) : ℝ) * x ^ k) := by linear_combination ((k + 2 : ℕ) : ℝ) * hpow_eq
        _ ≤ (ε / (4 * R ^ 2)) * (2 * (3 / 4 : ℝ) ^ k) :=
            mul_le_mul_of_nonneg_left hxk_le (by positivity)
        _ = (ε / (2 * R ^ 2)) * (3 / 4 : ℝ) ^ k := by ring
    have hsum2_abs : Summable (fun k : ℕ => |((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)|) :=
      Summable.of_nonneg_of_le (fun _ => abs_nonneg _) hterm2 hmaj2_sum
    have hsum2 : Summable (fun k : ℕ => ((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)) :=
      hsum2_abs.of_abs
    have hbd2 : |∑' k : ℕ, ((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)| ≤ 2 * ε / R ^ 2 := by
      have hle := abs_tsum_le_of_le hsum2 hmaj2_sum hterm2
      rw [tsum_mul_left, tsum_geometric_of_norm_lt_one h34_norm] at hle
      have heval : (ε / (2 * R ^ 2)) * (1 - (3 / 4 : ℝ))⁻¹ = 2 * ε / R ^ 2 := by
        norm_num; ring
      linarith
    exact ⟨hsum1, hsum2, hbd1, hbd2⟩
  refine ⟨htails, ?_⟩
  intro D_eps A ⟨d, hd, hser⟩
  have h51_eps : ∀ ε : ℝ, 0 < ε → ε ≤ R / 2 →
      |D_eps ε - A * ε| ≤ ε ^ 2 / (2 * R ^ 2) ∧
      |deriv D_eps ε - A| ≤ 2 * ε / R ^ 2 := by
    intro ε hε0 hεR
    obtain ⟨hval_eq, hder⟩ := hser ε hε0 hεR
    obtain ⟨_, _, hbd1, hbd2⟩ := htails d ε hd hε0 hεR
    refine ⟨?_, ?_⟩
    · have hsub : D_eps ε - A * ε = ∑' k : ℕ, d (k + 2) * ε ^ (k + 2) := by linarith
      rwa [hsub]
    · rw [hder.deriv]
      have hsub : A + (∑' k : ℕ, ((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1)) - A =
          ∑' k : ℕ, ((k + 2 : ℕ) : ℝ) * d (k + 2) * ε ^ (k + 1) := by ring
      rwa [hsub]
  have h51_s : ∀ s : ℝ, 0 < 1 / s → 1 / s ≤ R / 2 →
      DifferentiableAt ℝ (fun u : ℝ => D_eps (1 / u)) s ∧
      |s ^ 2 * deriv (fun u : ℝ => D_eps (1 / u)) s + A| ≤ 2 / (s * R ^ 2) := by
    intro s hs_inv hs_le
    have hs0 : 0 < s := one_div_pos.mp hs_inv
    obtain ⟨_, hder_eps⟩ := hser (1 / s) hs_inv hs_le
    have hinv_der : HasDerivAt (fun u : ℝ => 1 / u) (-(s ^ 2)⁻¹) s := by
      simpa [one_div] using hasDerivAt_inv hs0.ne'
    have hcomp : HasDerivAt (fun u : ℝ => D_eps (1 / u))
        ((deriv D_eps (1 / s)) * (-(s ^ 2)⁻¹)) s := by
      rw [← hder_eps.deriv] at hder_eps
      exact hder_eps.comp s hinv_der
    refine ⟨hcomp.differentiableAt, ?_⟩
    rw [hcomp.deriv]
    have hrew : s ^ 2 * (deriv D_eps (1 / s) * -(s ^ 2)⁻¹) + A =
        -(deriv D_eps (1 / s) - A) := by
      field_simp [hs0.ne']
      ring
    rw [hrew, abs_neg]
    have hb := (h51_eps (1 / s) hs_inv hs_le).2
    have hrhs : 2 * (1 / s) / R ^ 2 = 2 / (s * R ^ 2) := by
      field_simp [hs0.ne', hR.ne']
    rwa [hrhs] at hb
  refine ⟨h51_eps, h51_s, ?_⟩
  intro hA s hs_inv hs_lt
  have hs0 : 0 < s := one_div_pos.mp hs_inv
  have hs_R2 : 1 / s < R / 2 := lt_of_lt_of_le hs_lt (min_le_left _ _)
  have hs_AR : 1 / s < |A| * R ^ 2 / 4 := lt_of_lt_of_le hs_lt (min_le_right _ _)
  have hder_bd := (h51_s s hs_inv hs_R2.le).2
  have hval_bd := (h51_eps (1 / s) hs_inv hs_R2.le).1
  -- Derivative error `2 / (s * R ^ 2) < |A| / 2 < |A|`:
  have hder_err_lt : 2 / (s * R ^ 2) < |A| / 2 := by
    have hm := mul_lt_mul_of_pos_right hs_AR (show (0 : ℝ) < 2 / R ^ 2 by positivity)
    have hl : (1 / s) * (2 / R ^ 2) = 2 / (s * R ^ 2) := by field_simp [hs0.ne', hR.ne']
    have hr : (|A| * R ^ 2 / 4) * (2 / R ^ 2) = |A| / 2 := by field_simp [hR.ne']; ring
    linarith
  -- Value error `(1 / s) ^ 2 / (2 * R ^ 2) < |A| * (1 / s) / 8 < |A| * (1 / s)`:
  have hval_err_lt : (1 / s) ^ 2 / (2 * R ^ 2) < |A| * (1 / s) := by
    have h1 : (1 / s) / (2 * R ^ 2) < |A| / 8 := by
      have hm := mul_lt_mul_of_pos_right hs_AR (show (0 : ℝ) < 1 / (2 * R ^ 2) by positivity)
      have hl : (1 / s) * (1 / (2 * R ^ 2)) = (1 / s) / (2 * R ^ 2) := by ring
      have hr : (|A| * R ^ 2 / 4) * (1 / (2 * R ^ 2)) = |A| / 8 := by field_simp [hR.ne']; ring
      linarith
    have hm2 := mul_lt_mul_of_pos_right h1 hs_inv
    have hl2 : ((1 / s) / (2 * R ^ 2)) * (1 / s) = (1 / s) ^ 2 / (2 * R ^ 2) := by ring
    have hA_pos : 0 < |A| := abs_pos.mpr hA
    have hr2 : (|A| / 8) * (1 / s) < |A| * (1 / s) := by nlinarith
    linarith
  set Ds' := deriv (fun u : ℝ => D_eps (1 / u)) s
  have hder_abs := abs_le.mp hder_bd
  have hval_abs := abs_le.mp hval_bd
  have hA_abs_pos : 0 < |A| := abs_pos.mpr hA
  rcases lt_or_gt_of_ne hA with hA_neg | hA_pos
  · have habsA : |A| = -A := abs_of_neg hA_neg
    have hs2_Ds'_pos : 0 < s ^ 2 * Ds' := by linarith
    have hDs'_pos : 0 < Ds' := by nlinarith [sq_pos_of_pos hs0]
    have hval_neg : D_eps (1 / s) < 0 := by
      have h1 : A * (1 / s) = -(|A| * (1 / s)) := by rw [habsA]; ring
      linarith
    refine ⟨?_, ?_, fun _ => ⟨hDs'_pos, hval_neg⟩, fun h => by linarith⟩
    · rw [Real.sign_of_pos hDs'_pos, Real.sign_of_neg hA_neg]; norm_num
    · rw [Real.sign_of_neg hval_neg, Real.sign_of_neg hA_neg]
  · have habsA : |A| = A := abs_of_pos hA_pos
    have hs2_Ds'_neg : s ^ 2 * Ds' < 0 := by linarith
    have hDs'_neg : Ds' < 0 := by
      by_contra hnn
      have : 0 ≤ s ^ 2 * Ds' := mul_nonneg (sq_nonneg s) (not_lt.mp hnn)
      linarith
    have hval_pos : 0 < D_eps (1 / s) := by
      have h1 : A * (1 / s) = |A| * (1 / s) := by rw [habsA]
      linarith
    refine ⟨?_, ?_, fun h => by linarith, fun _ => ⟨hDs'_neg, hval_pos⟩⟩
    · simp [Real.sign_of_neg hDs'_neg, Real.sign_of_pos hA_pos]
    · rw [Real.sign_of_pos hval_pos, Real.sign_of_pos hA_pos]

end Dobson.HostOutages

#print axioms Dobson.HostOutages.outage_coefficient_bounds
