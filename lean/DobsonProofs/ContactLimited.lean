import Mathlib

/-!
# The contact-limited coefficient (items K1 to K6)

All variables are real. `A` stands for `(1 + 2θ)/(2T)`.

* K1 to K4 are stated on variables. In K3 the derived quantities `K` and `Kc` are variables
  carrying their defining equations as hypotheses.
* K5 uses the definitions `qk`, `coeff`, `Kcoef` and `Kc` below, which are copied from the
  statement: `Kc b` is the left side of K1 with `θ = 1`, waiting stay `3`, `d = 1/10` and a
  second stage of stay `b`.
* K6 uses the definition `f p h = p·h / (p·h + 1 − p)`; derivatives are Mathlib's `deriv` and
  `HasDerivAt`, the second derivative is `deriv (deriv (f p))`, and strict concavity is
  `StrictConcaveOn ℝ (Set.Ioi 0) (f p)`.
-/

namespace Dobson.ContactLimited

/-! ## K1 to K4 -/

/-- **K1 (two forms of one coefficient).** For real `A`, `R`, `q`, `d` with `R ≠ 1`:
`d²·A − d·(R − (1−d)·q)/(1−R) = d²·(A − R/(1−R)) + d·(1−d)·(q − R)/(1−R)`. -/
theorem K1 (A R q d : ℝ) (hR : R ≠ 1) :
    d ^ 2 * A - d * (R - (1 - d) * q) / (1 - R)
      = d ^ 2 * (A - R / (1 - R)) + d * (1 - d) * (q - R) / (1 - R) := by
  have h : 1 - R ≠ 0 := sub_ne_zero.mpr (Ne.symm hR)
  field_simp
  ring

/-- **K2 (positivity with non-negative memory).** If `0 < d ≤ 1`, `0 ≤ R ≤ q < 1` and
`K > 0`, then `d²·K + d·(1−d)·(q−R)/(1−R) > 0`.
(The hypothesis `0 ≤ R` is part of the statement but is not needed by the proof.) -/
theorem K2 (d R q K : ℝ) (hd0 : 0 < d) (hd1 : d ≤ 1) (_hR0 : 0 ≤ R) (hRq : R ≤ q)
    (hq1 : q < 1) (hK : K > 0) :
    d ^ 2 * K + d * (1 - d) * (q - R) / (1 - R) > 0 := by
  -- `0 ≤ R` is not needed. It is cleared here so that no later step can use it.
  clear _hR0
  have h1 : 0 < 1 - R := by linarith
  have h2 : 0 ≤ d * (1 - d) * (q - R) / (1 - R) :=
    div_nonneg (mul_nonneg (mul_nonneg hd0.le (by linarith)) (by linarith)) h1.le
  have h3 : 0 < d ^ 2 * K := mul_pos (pow_pos hd0 2) hK
  linarith

/-- **K3 (it can exceed the gain-only coefficient, exactly when).** With
`K = A − R/(1−R)` and `Kc = d²·K + d·(1−d)·(q−R)/(1−R)`, for `R < 1` and `0 ≤ d < 1`:
`Kc > K ↔ (1+d)·A < (R + d·q)/(1−R)`.
(The hypothesis `0 ≤ d` is part of the statement but is not needed by the proof.)

Audit correction, 2026-11-16 (auditfix1): removed the unused `0 ≤ R` premise, which
excluded negative-memory odd-stage rings covered by the manuscript. The same algebra
proves the comparison for every `R < 1`, including negative `R`. -/
theorem K3 (A R q d K Kc : ℝ) (hK : K = A - R / (1 - R))
    (hKc : Kc = d ^ 2 * K + d * (1 - d) * (q - R) / (1 - R))
    (hR1 : R < 1) (_hd0 : 0 ≤ d) (hd1 : d < 1) :
    Kc > K ↔ (1 + d) * A < (R + d * q) / (1 - R) := by
  -- Retain the audited stronger-than-needed model restriction on `d`, as requested.
  clear _hd0
  have h1 : 1 - R ≠ 0 := by linarith
  have h2 : 0 < 1 - d := by linarith
  have key : Kc - K = (1 - d) * ((R + d * q) / (1 - R) - (1 + d) * A) := by
    rw [hKc, hK]
    field_simp
    ring
  rw [gt_iff_lt, ← sub_pos, key, mul_pos_iff_of_pos_left h2, sub_pos]

/-- **K4 (the long-stage limit and its size).** For `0 ≤ d ≤ 1` and `0 ≤ q ≤ 1`:
`0 ≤ d·(1−d)·q` and `d·(1−d)·q ≤ 1/4`. -/
theorem K4 (d q : ℝ) (hd0 : 0 ≤ d) (hd1 : d ≤ 1) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    0 ≤ d * (1 - d) * q ∧ d * (1 - d) * q ≤ 1 / 4 := by
  have h1 : 0 ≤ d * (1 - d) := mul_nonneg hd0 (by linarith)
  refine ⟨mul_nonneg h1 hq0, ?_⟩
  have h2 : d * (1 - d) ≤ 1 / 4 := by nlinarith [sq_nonneg (d - 1 / 2)]
  calc d * (1 - d) * q ≤ d * (1 - d) * 1 := mul_le_mul_of_nonneg_left hq1 h1
    _ ≤ 1 / 4 := by linarith

/-! ## K5: the concrete case -/

/-- `q_k(θ, τ) = 1 / (1 + τ/θ)`. -/
noncomputable def qk (θ τ : ℝ) : ℝ := 1 / (1 + τ / θ)

/-- The left side of K1: `d²·A − d·(R − (1−d)·q)/(1−R)`. -/
noncomputable def coeff (A R q d : ℝ) : ℝ :=
  d ^ 2 * A - d * (R - (1 - d) * q) / (1 - R)

/-- The coefficient for memory parameter `θ`, a waiting stage of stay `a`, contact
probability `d` and a second stage of stay `b`: the left side of K1 with
`A = (1 + 2θ)/(2·(a + b))`, `q = q_k(θ, a)` and `R = q_k(θ, a)·q_k(θ, b)`. -/
noncomputable def Kcoef (θ a d b : ℝ) : ℝ :=
  coeff ((1 + 2 * θ) / (2 * (a + b))) (qk θ a * qk θ b) (qk θ a) d

/-- `Kc(b)`: `θ = 1`, waiting stay `3`, `d = 1/10`, second stage of stay `b`. -/
noncomputable def Kc (b : ℝ) : ℝ := Kcoef 1 3 (1 / 10) b

/-- The ingredients of `Kc(b)` as written in the statement: `q_k(1, 3) = 1/4` and
`q_k(1, b) = 1/(1 + b)`, so `A = 3/(2·(3 + b))`, `q = 1/4`, `R = (1/4)·(1/(1 + b))`. -/
theorem Kc_eq (b : ℝ) :
    Kc b = (1 / 10 : ℝ) ^ 2 * ((1 + 2 * 1) / (2 * (3 + b)))
      - (1 / 10) * ((1 / 4) * (1 / (1 + b)) - (1 - 1 / 10) * (1 / 4))
        / (1 - (1 / 4) * (1 / (1 + b))) := by
  unfold Kc Kcoef coeff qk
  norm_num

/-- Exact value: `Kc(1) = 17/1120 = 0.015178571428…`. -/
theorem K5_value_one : Kc 1 = 17 / 1120 := by
  unfold Kc Kcoef coeff qk
  norm_num

/-- Exact value: `Kc(10) = 2443/111800 = 0.021851520572…`. -/
theorem K5_value_ten : Kc 10 = 2443 / 111800 := by
  unfold Kc Kcoef coeff qk
  norm_num

/-- The limit `d·(1−d)·q` with `d = 1/10` and `q = q_k(1, 3)` is `9/400 = 0.0225`. -/
theorem K5_limit : (1 / 10 : ℝ) * (1 - 1 / 10) * qk 1 3 = 9 / 400 := by
  unfold qk
  norm_num

/-- **K5 (the limit is not a lower bound: the concrete case).** `Kc(1) < Kc(10)` and
`Kc(10) < d·(1−d)·q = 9/400`, with `d = 1/10` and `q = q_k(1, 3) = 1/4`: the coefficient
rises from `b = 1` to `b = 10` and stays below the limit.

Exact values: `Kc(1) = 17/1120 = 0.015178571428…` and `Kc(10) = 2443/111800 = 0.021851520572…`
(`K5_value_one`, `K5_value_ten`); they match the decimals `0.01517857…` and `0.02185152…`
(`K5_decimals`). -/
theorem K5 :
    Kc 1 < Kc 10 ∧ Kc 10 < (1 / 10 : ℝ) * (1 - 1 / 10) * qk 1 3
      ∧ (1 / 10 : ℝ) * (1 - 1 / 10) * qk 1 3 = 9 / 400 := by
  rw [K5_value_one, K5_value_ten, K5_limit]
  norm_num

/-- The exact values agree with the quoted decimals to the eight places quoted:
`0.01517857 < Kc(1) < 0.01517858` and `0.02185152 < Kc(10) < 0.02185153`. -/
theorem K5_decimals :
    (0.01517857 : ℝ) < Kc 1 ∧ Kc 1 < 0.01517858
      ∧ (0.02185152 : ℝ) < Kc 10 ∧ Kc 10 < 0.02185153 := by
  rw [K5_value_one, K5_value_ten]
  norm_num

/-! ## K6: a waiting individual in a frozen host -/

/-- `f(h) = p·h / (p·h + 1 − p)`. -/
noncomputable def f (p h : ℝ) : ℝ := p * h / (p * h + 1 - p)

/-- First derivative of `f` at any `h` where the denominator `p·h + 1 − p` is not zero:
`f'(h) = p·(1−p)/(p·h + 1 − p)²`. -/
theorem hasDerivAt_f (p h : ℝ) (hD : p * h + 1 - p ≠ 0) :
    HasDerivAt (f p) (p * (1 - p) / (p * h + 1 - p) ^ 2) h := by
  have hnum : HasDerivAt (fun x : ℝ => p * x) p h := hasDerivAt_const_mul p
  have hden : HasDerivAt (fun x : ℝ => p * x + 1 - p) p h := (hnum.add_const 1).sub_const p
  have h3 : HasDerivAt (fun x : ℝ => p * x / (p * x + 1 - p))
      ((p * (p * h + 1 - p) - p * h * p) / (p * h + 1 - p) ^ 2) h := hnum.fun_div hden hD
  have e : (p * (p * h + 1 - p) - p * h * p) / (p * h + 1 - p) ^ 2
      = p * (1 - p) / (p * h + 1 - p) ^ 2 := by ring
  rw [← e]
  exact h3

/-- `deriv` form of `hasDerivAt_f`. -/
theorem deriv_f (p h : ℝ) (hD : p * h + 1 - p ≠ 0) :
    deriv (f p) h = p * (1 - p) / (p * h + 1 - p) ^ 2 :=
  (hasDerivAt_f p h hD).deriv

/-- Second derivative of `f` at any `h` where the denominator is not zero:
`f''(h) = −2·p²·(1−p)/(p·h + 1 − p)³`. -/
theorem hasDerivAt_deriv_f (p h : ℝ) (hD : p * h + 1 - p ≠ 0) :
    HasDerivAt (deriv (f p)) (-2 * p ^ 2 * (1 - p) / (p * h + 1 - p) ^ 3) h := by
  have hden : HasDerivAt (fun x : ℝ => p * x + 1 - p) p h :=
    ((hasDerivAt_const_mul p).add_const 1).sub_const p
  have hpow : HasDerivAt (fun x : ℝ => (p * x + 1 - p) ^ 2)
      ((2 : ℕ) * (p * h + 1 - p) ^ (2 - 1) * p) h := hden.fun_pow 2
  have hc : HasDerivAt (fun _ : ℝ => p * (1 - p)) 0 h := hasDerivAt_const h _
  have hg : HasDerivAt (fun x : ℝ => p * (1 - p) / (p * x + 1 - p) ^ 2)
      ((0 * (p * h + 1 - p) ^ 2 - p * (1 - p) * ((2 : ℕ) * (p * h + 1 - p) ^ (2 - 1) * p))
        / ((p * h + 1 - p) ^ 2) ^ 2) h := hc.fun_div hpow (pow_ne_zero 2 hD)
  have e : (0 * (p * h + 1 - p) ^ 2 - p * (1 - p) * ((2 : ℕ) * (p * h + 1 - p) ^ (2 - 1) * p))
        / ((p * h + 1 - p) ^ 2) ^ 2 = -2 * p ^ 2 * (1 - p) / (p * h + 1 - p) ^ 3 := by
    push_cast
    field_simp
    ring
  rw [e] at hg
  have hopen : IsOpen {x : ℝ | p * x + 1 - p ≠ 0} :=
    isOpen_ne_fun (by fun_prop) continuous_const
  have hev : deriv (f p) =ᶠ[nhds h] fun x : ℝ => p * (1 - p) / (p * x + 1 - p) ^ 2 := by
    filter_upwards [hopen.mem_nhds hD] with x hx
    exact deriv_f p x hx
  exact hg.congr_of_eventuallyEq hev

/-- `deriv` form of `hasDerivAt_deriv_f`. -/
theorem deriv_deriv_f (p h : ℝ) (hD : p * h + 1 - p ≠ 0) :
    deriv (deriv (f p)) h = -2 * p ^ 2 * (1 - p) / (p * h + 1 - p) ^ 3 :=
  (hasDerivAt_deriv_f p h hD).deriv

/-- **K6, value.** `f(1) = p`.
(The hypotheses `0 < p < 1` are part of the statement but are not needed for this
equality.) -/
theorem K6_value (p : ℝ) (_hp0 : 0 < p) (_hp1 : p < 1) : f p 1 = p := by
  clear _hp0 _hp1
  simp [f]

/-- **K6, first derivative.** For `0 < p < 1` the derivative of `f` at `1` is `p·(1−p)`.
(The hypotheses `0 < p < 1` are part of the statement but are not needed for this
equality.) -/
theorem K6_hasDerivAt (p : ℝ) (_hp0 : 0 < p) (_hp1 : p < 1) :
    HasDerivAt (f p) (p * (1 - p)) 1 := by
  clear _hp0 _hp1
  have h := hasDerivAt_f p 1 (by simp)
  simpa using h

/-- **K6, second derivative.** For `0 < p < 1` the second derivative of `f` at `1` is
`−2·p²·(1−p)`. (The hypotheses `0 < p < 1` are part of the statement but are not needed for
this equality.) -/
theorem K6_hasDerivAt_deriv (p : ℝ) (_hp0 : 0 < p) (_hp1 : p < 1) :
    HasDerivAt (deriv (f p)) (-2 * p ^ 2 * (1 - p)) 1 := by
  clear _hp0 _hp1
  have h := hasDerivAt_deriv_f p 1 (by simp)
  simpa using h

/-- **K6, sign of the second derivative.** For `0 < p < 1` and every `h > 0`: `f''(h) < 0`. -/
theorem K6_deriv_deriv_neg (p : ℝ) (hp0 : 0 < p) (hp1 : p < 1) (h : ℝ) (hh : 0 < h) :
    deriv (deriv (f p)) h < 0 := by
  have hD : 0 < p * h + 1 - p := by nlinarith [mul_pos hp0 hh]
  rw [deriv_deriv_f p h hD.ne']
  have hnum : -2 * p ^ 2 * (1 - p) < 0 := by
    nlinarith [mul_pos (pow_pos hp0 2) (sub_pos.mpr hp1)]
  exact div_neg_of_neg_of_pos hnum (pow_pos hD 3)

/-- **K6, strict concavity.** For `0 < p < 1`, `f` is strictly concave on `(0, ∞)`. -/
theorem K6_strictConcaveOn (p : ℝ) (hp0 : 0 < p) (hp1 : p < 1) :
    StrictConcaveOn ℝ (Set.Ioi 0) (f p) := by
  refine strictConcaveOn_of_deriv2_neg' (convex_Ioi 0) ?_ ?_
  · intro x hx
    have hx0 : 0 < x := hx
    have hD : 0 < p * x + 1 - p := by nlinarith [mul_pos hp0 hx0]
    exact (hasDerivAt_f p x hD.ne').continuousAt.continuousWithinAt
  · intro x hx
    simpa using K6_deriv_deriv_neg p hp0 hp1 x hx

/-- **K6.** For `0 < p < 1` and `f(h) = p·h/(p·h + 1 − p)`: `f(1) = p`; the derivative of `f`
at `1` is `p·(1−p)`; the second derivative at `1` is `−2·p²·(1−p)`; and `f` is strictly
concave on `(0, ∞)`. -/
theorem K6 (p : ℝ) (hp0 : 0 < p) (hp1 : p < 1) :
    f p 1 = p ∧ deriv (f p) 1 = p * (1 - p)
      ∧ deriv (deriv (f p)) 1 = -2 * p ^ 2 * (1 - p)
      ∧ StrictConcaveOn ℝ (Set.Ioi 0) (f p) :=
  ⟨K6_value p hp0 hp1, (K6_hasDerivAt p hp0 hp1).deriv, (K6_hasDerivAt_deriv p hp0 hp1).deriv,
    K6_strictConcaveOn p hp0 hp1⟩

end Dobson.ContactLimited

#print axioms Dobson.ContactLimited.K1
#print axioms Dobson.ContactLimited.K2
#print axioms Dobson.ContactLimited.K3
#print axioms Dobson.ContactLimited.K4
#print axioms Dobson.ContactLimited.K5
#print axioms Dobson.ContactLimited.K5_value_one
#print axioms Dobson.ContactLimited.K5_value_ten
#print axioms Dobson.ContactLimited.K5_decimals
#print axioms Dobson.ContactLimited.K6
