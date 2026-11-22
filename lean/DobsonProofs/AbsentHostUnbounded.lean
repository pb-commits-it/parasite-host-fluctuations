import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

/-!
# A13i: explicit derivative interval for unbounded hosts

Constants are exactly (6), (12), (15), (21) of `a13i/a13i.md`.
`RootInputs` lists the analytic handoff; no stationary ergodic process is
constructed here. Fourth-moment filters, growth differentiability and its
intervals, coefficient estimates and implicit differentiation remain inputs.
No bounded-host, inverse-moment or higher-moment premise is introduced.

## Audit correction, 2026-11-16 (auditfix1)

The nonstandard inputs are project-prose results, not model assumptions.
`hf` in the convex-secant lemma requires the stationary growth regularity
proof of `a13i/a13i.md` section 2; `hc`, `hl`, `hr` require the fourth-order
value estimate and endpoint-domain checks in sections 3–4. The eleven
original `RootInputs` fields remain unproved for the stochastic model in
Lean: `root` is the expansion evaluated at the restoring root; `B_bound`,
`C_bound`, `A_bound` are coefficient bounds; `B_difference` is a coefficient
Lipschitz bound; `B_derivative`, `C_derivative` are coefficient derivative
bounds; `value_error` is the fourth-order growth remainder; `e_derivative`,
`D_derivative` are actual growth derivative estimates; `implicit_derivative`
requires model regularity and root identification. Their prose sources are
sections 3–5, with section 2 for regularity. Every theorem receiving
`RootInputs`, including either uniform family, is partial as coverage of
the stochastic model. The scalar consequences remain proved.

`value_link` is the missing model coordinate definition `c = exp D`. With
that definition the separate zero-limit premise is proved by `root_zero_limit`
and removed from `eventual_value_sign`. It cannot be deduced from the original
eleven fields, which constrain only D's derivative. The three audited
stronger-than-needed quotient hypotheses stay unchanged.
-/
namespace Dobson.AbsentHostUnbounded
open Set Filter
open scoped Topology

noncomputable def T (H : ℝ) : ℝ := 2 * H + 1
noncomputable def Q (H : ℝ) : ℝ := 4 * H + 1
noncomputable def U (b H : ℝ) : ℝ := T H / b
noncomputable def L (b H : ℝ) : ℝ := 2 * Q H / b
noncomputable def N (b H : ℝ) : ℝ := L b H * (Q H + 2 * H) / b
noncomputable def J (b H : ℝ) : ℝ := U b H * Q H / b
noncomputable def Z (b H : ℝ) : ℝ := (Q H * N b H + 2 * H * L b H ^ 2) / b
noncomputable def B0 (b H : ℝ) : ℝ := 4 * H ^ 2 / b + 2 / b + 2 * H ^ 2 + 3
noncomputable def C0 (b H : ℝ) : ℝ :=
  2 * H * J b H + 2 * H * T H * U b H + T H ^ 3 / 3
noncomputable def R4 (b H : ℝ) : ℝ :=
  2 * H * Z b H + 2 * H * T H * N b H + 2 * H ^ 2 * L b H ^ 2 +
  (2 * H * L b H / 3) * (Q H ^ 2 + Q H * T H + T H ^ 2) + 4 * Q H ^ 4
noncomputable def K (b H : ℝ) : ℝ := B0 b H + C0 b H + R4 b H
noncomputable def e0 (b H : ℝ) : ℝ := min (b / 4) (min (1 / 16) (1 / (8 * (K b H + 1))))
noncomputable def Mx (b H : ℝ) : ℝ := 1 + 4 * B0 b H + 9 * C0 b H
noncomputable def My (b H : ℝ) : ℝ := 2 + 4 * B0 b H + 9 * C0 b H
noncomputable def Ex (b H : ℝ) : ℝ := Mx b H / 2 + 3 * R4 b H
noncomputable def Ey (b H : ℝ) : ℝ := My b H / 2 + 2 * R4 b H
noncomputable def E0 (b H : ℝ) : ℝ := 2 * Ex b H + Ey b H
noncomputable def J0 (b H : ℝ) : ℝ := K b H + 2 * B0 b H + 3 * C0 b H + Ey b H
noncomputable def N0 (b H : ℝ) : ℝ := 4 * B0 b H * K b H + 2 * C0 b H + R4 b H + E0 b H
noncomputable def estar (b H : ℝ) : ℝ := min (e0 b H) (1 / (2 * J0 b H))
noncomputable def Cstar (b H : ℝ) : ℝ := 2 * (N0 b H + B0 b H * J0 b H)

/-- All constants are positive for `b=1/a>0`, `H=‖h‖₄≥1`. -/
theorem constants_positive (b H : ℝ) (hb : 0 < b) (hH : 1 ≤ H) :
    0 < B0 b H ∧ 0 < C0 b H ∧ 0 < R4 b H ∧ 0 < K b H ∧
    0 < Ey b H ∧ 0 < E0 b H ∧ 0 < J0 b H ∧ 0 < N0 b H ∧
    0 < estar b H ∧ 0 < Cstar b H := by
  have hH0 : 0 < H := lt_of_lt_of_le zero_lt_one hH
  have hT : 0 < T H := by unfold T; positivity
  have hQ : 0 < Q H := by unfold Q; positivity
  have hU : 0 < U b H := div_pos hT hb
  have hL : 0 < L b H := by unfold L; positivity
  have hN : 0 < N b H := by unfold N; positivity
  have hJ : 0 < J b H := by unfold J; positivity
  have hZ : 0 < Z b H := by unfold Z; positivity
  have hB : 0 < B0 b H := by unfold B0; positivity
  have hC : 0 < C0 b H := by unfold C0; positivity
  have hR : 0 < R4 b H := by unfold R4; positivity
  have hK : 0 < K b H := by unfold K; positivity
  have hMx : 0 < Mx b H := by unfold Mx; positivity
  have hMy : 0 < My b H := by unfold My; positivity
  have hEx : 0 < Ex b H := by unfold Ex; positivity
  have hEy : 0 < Ey b H := by unfold Ey; positivity
  have hE : 0 < E0 b H := by unfold E0; positivity
  have hJ0 : 0 < J0 b H := by unfold J0; positivity
  have hN0 : 0 < N0 b H := by unfold N0; positivity
  have he0 : 0 < e0 b H := by unfold e0; positivity
  have he : 0 < estar b H := by unfold estar; positivity
  exact ⟨hB, hC, hR, hK, hEy, hE, hJ0, hN0, he, by unfold Cstar; positivity⟩

/-- The registered cutoff gives the root and denominator margins. -/
theorem cutoff_bounds (b H e : ℝ) (hb : 0 < b) (hH : 1 ≤ H)
    (he : 0 < e) (he' : e ≤ estar b H) :
    e ≤ b / 4 ∧ e ≤ 1 / 16 ∧ e ≤ 1 ∧ K b H * e ≤ 1 / 8 ∧ J0 b H * e ≤ 1 / 2 := by
  obtain ⟨_, _, _, hK, _, _, hJ, _⟩ := constants_positive b H hb hH
  have h0 : e ≤ e0 b H := le_trans he' (min_le_left _ _)
  have h1 : e ≤ b / 4 := le_trans h0 (min_le_left _ _)
  have h2 : e ≤ 1 / 16 := le_trans h0 (le_trans (min_le_right _ _) (min_le_left _ _))
  have h3 : e ≤ 1 / (8 * (K b H + 1)) :=
    le_trans h0 (le_trans (min_le_right _ _) (min_le_right _ _))
  have h4 : e ≤ 1 / (2 * J0 b H) := le_trans he' (min_le_right _ _)
  have h3' := (le_div_iff₀ (show 0 < 8 * (K b H + 1) by positivity)).mp h3
  have h4' := (le_div_iff₀ (show 0 < 2 * J0 b H by positivity)).mp h4
  exact ⟨h1, h2, by linarith, by nlinarith, by nlinarith⟩

/-- Three-point value errors and Taylor secant bounds control a convex derivative.
The two comparison-function secant bounds are explicit hypotheses; deriving them
from a curvature bound is not included in this lemma. -/
theorem convex_value_error_to_derivative_interval
    (f P : ℝ → ℝ) (z d fp pp M ec ee : ℝ) (hd : 0 < d)
    (hconv : ConvexOn ℝ (Icc (z - d) (z + d)) f) (hf : HasDerivAt f fp z)
    (hc : |f z - P z| ≤ ec) (hl : |f (z - d) - P (z - d)| ≤ ee)
    (hr : |f (z + d) - P (z + d)| ≤ ee)
    (hPl : |(P z - P (z - d)) / d - pp| ≤ M * d / 2)
    (hPr : |(P (z + d) - P z) / d - pp| ≤ M * d / 2) :
    |fp - pp| ≤ M * d / 2 + (ec + ee) / d := by
  have hz : z ∈ Icc (z-d) (z+d) := ⟨by linarith, by linarith⟩
  have hlz : z-d ∈ Icc (z-d) (z+d) := ⟨le_rfl, by linarith⟩
  have hrz : z+d ∈ Icc (z-d) (z+d) := ⟨by linarith, le_rfl⟩
  have hleft := hconv.slope_le_of_hasDerivAt hlz hz (by linarith) hf
  have hright := hconv.le_slope_of_hasDerivAt hz hrz (by linarith) hf
  simp only [slope_def_field, show z - (z-d) = d by ring, show z+d-z = d by ring] at hleft hright
  have hleft' := (div_le_iff₀ hd).mp hleft
  have hright' := (le_div_iff₀ hd).mp hright
  rcases abs_le.mp hc with ⟨hc1,hc2⟩
  rcases abs_le.mp hl with ⟨hl1,hl2⟩
  rcases abs_le.mp hr with ⟨hr1,hr2⟩
  have hPl' := (le_div_iff₀ hd).mp (show pp - M*d/2 ≤ (P z-P (z-d))/d by linarith [(abs_le.mp hPl).1])
  have hPr' := (div_le_iff₀ hd).mp (show (P (z+d)-P z)/d ≤ pp + M*d/2 by linarith [(abs_le.mp hPr).2])
  have hcancel : (ec + ee) / d * d = ec + ee := div_mul_cancel₀ _ hd.ne'
  apply abs_le.mpr
  constructor
  · apply (mul_le_mul_iff_left₀ hd).mp
    nlinarith
  · apply (mul_le_mul_iff_left₀ hd).mp
    nlinarith

structure RootData where
  c : ℝ
  B : ℝ
  C : ℝ
  cBp : ℝ
  cCp : ℝ
  rho : ℝ
  re : ℝ
  rD : ℝ

/-- Eleven external analytic inputs, plus the model coordinate definition
`value_link`; `cBp,cCp` mean `c B'(c),c C'(c)`. The desired derivative interval,
zero limit and sign are conclusions, never inputs. -/
structure RootInputs (b H A e : ℝ) (D : ℝ → ℝ) (v : RootData) : Prop where
  /-- Modelling definition: the restoring multiplier is the exponential loss. -/
  value_link : v.c = Real.exp (D e)
  root : e * (v.c-1) - e^2*v.B + e^3*v.C + v.rho = 0
  B_bound : |v.B| ≤ B0 b H
  C_bound : |v.C| ≤ C0 b H
  A_bound : |A| ≤ B0 b H
  B_difference : |v.B-A| ≤ 4 * B0 b H * |v.c-1|
  B_derivative : |v.cBp| ≤ 2 * B0 b H
  C_derivative : |v.cCp| ≤ 3 * C0 b H
  value_error : |v.rho| ≤ R4 b H * e^4
  e_derivative : |v.re - (v.c-1-2*e*v.B+3*e^2*v.C)| ≤ E0 b H * e * Real.sqrt e
  D_derivative : |v.rD - (e*v.c-e^2*v.cBp+e^3*v.cCp)| ≤ Ey b H * e^2 * Real.sqrt e
  implicit_derivative : HasDerivAt D (-v.re / v.rD) e

private theorem abs_div_bound {x a e : ℝ} (he : 0 < e) (h : |x| ≤ a*e) : |x/e| ≤ a := by
  rw [abs_div, abs_of_pos he, div_le_iff₀ he]
  exact h

/-- Root localization is derived from the exact root equation. -/
theorem root_localization (b H A e : ℝ) (D : ℝ → ℝ) (v : RootData)
    (hb : 0 < b) (hH : 1 ≤ H) (he : 0 < e) (he1 : e ≤ 1)
    (h : RootInputs b H A e D v) : |v.c-1| ≤ K b H * e := by
  obtain ⟨_, hC, hR, _⟩ := constants_positive b H hb hH
  have he2 : e^2 ≤ e := by nlinarith
  have he3 : e^3 ≤ e := by nlinarith [mul_le_mul_of_nonneg_left he2 he.le]
  have hid : v.c-1 = e*v.B-e^2*v.C-v.rho/e := by
    field_simp
    nlinarith [h.root]
  have hrho : |v.rho/e| ≤ R4 b H * e^3 := by
    apply abs_div_bound he
    convert h.value_error using 1; ring
  rw [hid]
  calc
    _ ≤ |e*v.B| + |e^2*v.C| + |v.rho/e| :=
      (abs_sub _ _).trans (add_le_add (abs_sub _ _) le_rfl)
    _ ≤ e * B0 b H + e^2 * C0 b H + R4 b H * e^3 := by
      rw [abs_mul, abs_mul, abs_of_pos he, abs_of_nonneg (sq_nonneg e)]
      exact add_le_add (add_le_add (mul_le_mul_of_nonneg_left h.B_bound he.le)
        (mul_le_mul_of_nonneg_left h.C_bound (sq_nonneg e))) hrho
    _ ≤ K b H * e := by
      unfold K
      nlinarith [mul_le_mul_of_nonneg_left he2 hC.le, mul_le_mul_of_nonneg_left he3 hR.le]

/-- Root localization and the model coordinate definition imply the zero limit;
no value-asymptotic or limit hypothesis is needed. -/
theorem root_zero_limit (b H A : ℝ) (D : ℝ → ℝ)
    (hb : 0 < b) (hH : 1 ≤ H)
    (hinputs : ∀ e, 0 < e → e ≤ estar b H → ∃ v, RootInputs b H A e D v) :
    Tendsto (fun s => D (1/s)) atTop (𝓝 0) := by
  have hepos : 0 < estar b H := (constants_positive b H hb hH).2.2.2.2.2.2.2.2.1
  have hinv : Tendsto (fun s : ℝ => 1/s) atTop (𝓝 0) := by
    simpa only [one_div] using tendsto_inv_atTop_zero
  have hsmall : ∀ᶠ s : ℝ in atTop, 1/s < estar b H :=
    hinv.eventually (gt_mem_nhds hepos)
  have hbound : ∀ᶠ s : ℝ in atTop,
      ‖Real.exp (D (1/s)) - 1‖ ≤ K b H * (1/s) := by
    filter_upwards [hsmall, eventually_gt_atTop (0 : ℝ)] with s he hs
    have he0 : 0 < 1/s := one_div_pos.mpr hs
    obtain ⟨v, hv⟩ := hinputs (1/s) he0 he.le
    have he1 := (cutoff_bounds b H (1/s) hb hH he0 he.le).2.2.1
    simpa only [Real.norm_eq_abs, hv.value_link] using
      root_localization b H A (1/s) D v hb hH he0 he1 hv
  have hzero : Tendsto (fun s => Real.exp (D (1/s)) - 1) atTop (𝓝 0) :=
    squeeze_zero_norm' hbound (by simpa using hinv.const_mul (K b H))
  have hexp : Tendsto (fun s => Real.exp (D (1/s))) atTop (𝓝 1) := by
    simpa using hzero.add_const 1
  simpa [Function.comp_def] using (Real.continuousAt_log one_ne_zero).tendsto.comp hexp

private theorem sqrt_controls (e : ℝ) (he : 0 < e) (he1 : e ≤ 1) :
    0 < Real.sqrt e ∧ Real.sqrt e ≤ 1 ∧ e ≤ Real.sqrt e ∧ e^2 ≤ Real.sqrt e := by
  have hsq := Real.sq_sqrt he.le
  have hp := Real.sqrt_pos.mpr he
  have hu : Real.sqrt e ≤ 1 := by nlinarith
  exact ⟨hp, hu, by nlinarith, by nlinarith⟩

/-- Root cancellation proves (23) without differentiating the value remainder. -/
theorem root_numerator_interval (b H A e : ℝ) (D : ℝ → ℝ) (v : RootData)
    (hb : 0 < b) (hH : 1 ≤ H) (he : 0 < e) (he1 : e ≤ 1)
    (h : RootInputs b H A e D v) : |-v.re/e-A| ≤ N0 b H * Real.sqrt e := by
  obtain ⟨hB, hC, hR, hK, _, hE, _⟩ := constants_positive b H hb hH
  obtain ⟨_, _, hesq, he2sq⟩ := sqrt_controls e he he1
  have hloc := root_localization b H A e D v hb hH he he1 h
  have hBA : |v.B-A| ≤ 4 * B0 b H * K b H * e :=
    h.B_difference.trans (by nlinarith [mul_le_mul_of_nonneg_left hloc (show 0 ≤ 4 * B0 b H by positivity)])
  have hrho : |v.rho/e^2| ≤ R4 b H * e^2 := by
    apply abs_div_bound (sq_pos_of_pos he)
    convert h.value_error using 1; ring
  have herr : |(v.re-(v.c-1-2*e*v.B+3*e^2*v.C))/e| ≤ E0 b H * Real.sqrt e := by
    apply abs_div_bound he
    convert h.e_derivative using 1; ring
  have hid : -v.re/e-A = (v.B-A)-2*e*v.C+v.rho/e^2 -
      (v.re-(v.c-1-2*e*v.B+3*e^2*v.C))/e := by
    field_simp
    nlinarith [h.root]
  rw [hid]
  calc
    _ ≤ |v.B-A| + |2*e*v.C| + |v.rho/e^2| + |(v.re-(v.c-1-2*e*v.B+3*e^2*v.C))/e| :=
      (abs_sub _ _).trans (add_le_add ((abs_add_le _ _).trans
        (add_le_add (abs_sub _ _) le_rfl)) le_rfl)
    _ ≤ 4 * B0 b H * K b H * e + 2*e*C0 b H + R4 b H * e^2 + E0 b H * Real.sqrt e := by
      rw [abs_mul, abs_of_pos (show 0 < 2*e by positivity)]
      exact add_le_add (add_le_add (add_le_add hBA
        (mul_le_mul_of_nonneg_left h.C_bound (by positivity))) hrho) herr
    _ ≤ N0 b H * Real.sqrt e := by
      unfold N0
      nlinarith [mul_le_mul_of_nonneg_left hesq (show 0 ≤ 4 * B0 b H * K b H by positivity),
        mul_le_mul_of_nonneg_left hesq (show 0 ≤ 2 * C0 b H by positivity),
        mul_le_mul_of_nonneg_left he2sq hR.le]

/-- The scaled implicit denominator stays at least one half. -/
theorem root_denominator_interval (b H A e : ℝ) (D : ℝ → ℝ) (v : RootData)
    (hb : 0 < b) (hH : 1 ≤ H) (he : 0 < e) (he' : e ≤ estar b H)
    (h : RootInputs b H A e D v) : |v.rD/e-1| ≤ J0 b H * e ∧ 1/2 ≤ v.rD/e := by
  obtain ⟨_, _, he1, _, hJe⟩ := cutoff_bounds b H e hb hH he he'
  obtain ⟨hB, hC, _, _, hEy, _⟩ := constants_positive b H hb hH
  obtain ⟨_, hsq1, _, _⟩ := sqrt_controls e he he1
  have hloc := root_localization b H A e D v hb hH he he1 h
  have herr : |(v.rD-(e*v.c-e^2*v.cBp+e^3*v.cCp))/e| ≤ Ey b H * e := by
    apply abs_div_bound he
    calc
      _ ≤ Ey b H * e^2 * Real.sqrt e := h.D_derivative
      _ ≤ (Ey b H * e)*e := by
        nlinarith [mul_le_mul_of_nonneg_left hsq1 (show 0 ≤ Ey b H * e^2 by positivity)]
  have hid : v.rD/e-1 = (v.c-1)-e*v.cBp+e^2*v.cCp +
      (v.rD-(e*v.c-e^2*v.cBp+e^3*v.cCp))/e := by field_simp; ring
  have hbound : |v.rD/e-1| ≤ J0 b H * e := by
    rw [hid]
    calc
      _ ≤ |v.c-1| + |e*v.cBp| + |e^2*v.cCp| + |(v.rD-(e*v.c-e^2*v.cBp+e^3*v.cCp))/e| :=
        (abs_add_le _ _).trans (add_le_add ((abs_add_le _ _).trans
          (add_le_add (abs_sub _ _) le_rfl)) le_rfl)
      _ ≤ K b H * e + e*(2*B0 b H) + e^2*(3*C0 b H) + Ey b H * e := by
        rw [abs_mul, abs_mul, abs_of_pos he, abs_of_nonneg (sq_nonneg e)]
        exact add_le_add (add_le_add (add_le_add hloc (mul_le_mul_of_nonneg_left h.B_derivative he.le))
          (mul_le_mul_of_nonneg_left h.C_derivative (sq_nonneg e))) herr
      _ ≤ J0 b H * e := by
        unfold J0
        have he2 : e^2 ≤ e := by nlinarith
        nlinarith [mul_le_mul_of_nonneg_left he2 (show 0 ≤ 3*C0 b H by positivity)]
  exact ⟨hbound, by have := (abs_le.mp hbound).1; linarith⟩

/-- Quotient error with the exact factor two in (24). -/
theorem quotient_interval (x y A B J N e : ℝ) (he : 0 < e) (he1 : e ≤ 1)
    (hB : 0 ≤ B) (hJ : 0 ≤ J) (hN : 0 ≤ N) (hy : 1/2 ≤ y)
    (hx : |x-A| ≤ N*Real.sqrt e) (hy1 : |y-1| ≤ J*e) (hA : |A| ≤ B) :
    |x/y-A| ≤ 2*(N+B*J)*Real.sqrt e := by
  have hy0 : 0 < y := by linarith
  have hsq := (sqrt_controls e he he1).2.2.1
  rw [show x/y-A = ((x-A)-A*(y-1))/y by field_simp; ring]
  rw [abs_div, abs_of_pos hy0, div_le_iff₀ hy0]
  calc
    _ ≤ |x-A| + |A| * |y-1| := by simpa [abs_mul] using abs_sub (x-A) (A*(y-1))
    _ ≤ N*Real.sqrt e+B*(J*e) := add_le_add hx (mul_le_mul hA hy1 (abs_nonneg _) hB)
    _ ≤ (N+B*J)*Real.sqrt e := by nlinarith [mul_le_mul_of_nonneg_left hsq (mul_nonneg hB hJ)]
    _ ≤ 2*(N+B*J)*Real.sqrt e*y := by
      nlinarith [mul_nonneg (show 0 ≤ (N+B*J)*Real.sqrt e by positivity) (show 0 ≤ 2*y-1 by linarith)]

/-- Conditional threshold derivative interval, with the paper's explicit constants. -/
theorem threshold_derivative_interval (b H A e : ℝ) (D : ℝ → ℝ) (v : RootData)
    (hb : 0 < b) (hH : 1 ≤ H) (he : 0 < e) (he' : e ≤ estar b H)
    (h : RootInputs b H A e D v) : |deriv D e-A| ≤ Cstar b H * Real.sqrt e := by
  obtain ⟨hB, _, _, _, _, _, hJ, hN, _⟩ := constants_positive b H hb hH
  have he1 := (cutoff_bounds b H e hb hH he he').2.2.1
  have hn := root_numerator_interval b H A e D v hb hH he he1 h
  obtain ⟨hd, hy⟩ := root_denominator_interval b H A e D v hb hH he he' h
  have hq := quotient_interval (-v.re/e) (v.rD/e) A (B0 b H) (J0 b H)
    (N0 b H) e he he1 hB.le hJ.le hN.le hy hn hd h.A_bound
  rw [h.implicit_derivative.deriv]
  convert hq using 1 <;> simp [Cstar, div_div_div_cancel_right₀ he.ne']

/-- Actual reciprocal chain rule. -/
theorem reciprocal_derivative (D : ℝ → ℝ) (s d : ℝ) (hs : 0 < s)
    (hD : HasDerivAt D d (1/s)) : HasDerivAt (fun t => D (1/t)) (-d/s^2) s := by
  have hi : HasDerivAt (fun t : ℝ => 1/t) (-(1:ℝ)/s^2) s := by
    simpa [div_eq_mul_inv] using hasDerivAt_inv hs.ne'
  convert hD.comp s hi using 1 <;> simp [Function.comp_def, div_eq_mul_inv]

/-- Equation (2), conditional on the root inputs. -/
theorem eventual_reversal_fourth_moment (b H A s : ℝ) (D : ℝ → ℝ) (v : RootData)
    (hb : 0 < b) (hH : 1 ≤ H) (hs : 0 < s) (he' : 1/s ≤ estar b H)
    (h : RootInputs b H A (1/s) D v) :
    |s^2 * deriv (fun t => D (1/t)) s + A| ≤ Cstar b H / Real.sqrt s := by
  have he : 0 < 1/s := one_div_pos.mpr hs
  have hbound := threshold_derivative_interval b H A (1/s) D v hb hH he he' h
  have hd := reciprocal_derivative D s (-v.re/v.rD) hs h.implicit_derivative
  rw [hd.deriv]
  rw [h.implicit_derivative.deriv] at hbound
  have hid : s^2 * (-(-v.re/v.rD)/s^2)+A = -((-v.re/v.rD)-A) := by field_simp; ring
  rw [hid, abs_neg]
  simpa [one_div, Real.sqrt_inv, div_eq_mul_inv] using hbound

/-- Strict cutoff (3); the zero-coefficient case intentionally gives no sign. -/
theorem sign_of_strict_cutoff (A C s d : ℝ) (hA : A ≠ 0) (hC : 0 < C)
    (hs : 0 < s) (hcut : 1/s < (|A|/(2*C))^2)
    (hbound : |s^2*d+A| ≤ C/Real.sqrt s) : (0 < A → d < 0) ∧ (A < 0 → 0 < d) := by
  have hsqrt : 0 < Real.sqrt s := Real.sqrt_pos.mpr hs
  have hsqrt2 := Real.sq_sqrt hs.le
  have hAabs : 0 < |A| := abs_pos.mpr hA
  have hcut' := (div_lt_iff₀ hs).mp hcut
  have hratio : 0 < |A|/(2*C) := by positivity
  have hprod : 1 < (|A|/(2*C))*Real.sqrt s := by
    have heq : ((|A|/(2*C))*Real.sqrt s)^2 = (|A|/(2*C))^2*s := by rw [mul_pow, hsqrt2]
    have hnon : 0 ≤ (|A|/(2*C))*Real.sqrt s := by positivity
    nlinarith [sq_nonneg ((|A|/(2*C))*Real.sqrt s - 1)]
  have hhalf : C/Real.sqrt s < |A|/2 := by
    apply (div_lt_iff₀ hsqrt).mpr
    have hid : (|A|/(2*C)*Real.sqrt s)*(2*C) = |A| * Real.sqrt s := by field_simp
    have hh := mul_lt_mul_of_pos_right hprod (show 0 < 2*C by positivity)
    rw [hid] at hh
    nlinarith
  have habs : |s^2*d+A| < |A|/2 := lt_of_le_of_lt hbound hhalf
  rcases abs_lt.mp habs with ⟨hl, hu⟩
  constructor
  · intro hAp
    rw [abs_of_pos hAp] at hu
    have hsd : s^2*d < 0 := by linarith
    nlinarith [sq_pos_of_pos hs]
  · intro hAn
    rw [abs_of_neg hAn] at hl
    exact (mul_pos_iff_of_pos_left (sq_pos_of_pos hs)).mp (by linarith : 0 < s^2*d)


/-- Combine the interval and strict cutoff in one threshold theorem. -/
theorem tail_derivative_sign (b H A s : ℝ) (D : ℝ → ℝ) (v : RootData)
    (hb : 0 < b) (hH : 1 ≤ H) (hA : A ≠ 0) (hs : 0 < s)
    (hcut : 1/s < min (estar b H) ((|A|/(2*Cstar b H))^2))
    (h : RootInputs b H A (1/s) D v) :
    (0 < A → deriv (fun t => D (1/t)) s < 0) ∧
    (A < 0 → 0 < deriv (fun t => D (1/t)) s) := by
  have hc := (constants_positive b H hb hH).2.2.2.2.2.2.2.2.2
  have he := lt_of_lt_of_le hcut (min_le_left _ _)
  have hcut' := lt_of_lt_of_le hcut (min_le_right _ _)
  exact sign_of_strict_cutoff A (Cstar b H) s _ hA hc hs hcut'
    (eventual_reversal_fourth_moment b H A s D v hb hH hs he.le h)

noncomputable def tailStart (b H A : ℝ) : ℝ :=
  1 / min (estar b H) ((|A|/(2*Cstar b H))^2)

/-- The displayed strict reciprocal cutoff is a nonempty ordinary tail interval. -/
theorem tail_domain (b H A s : ℝ) (hb : 0 < b) (hH : 1 ≤ H) (hA : A ≠ 0)
    (hs : tailStart b H A < s) :
    0 < s ∧ 1/s < min (estar b H) ((|A|/(2*Cstar b H))^2) := by
  obtain ⟨_, _, _, _, _, _, _, _, he, hc⟩ := constants_positive b H hb hH
  have hAp : 0 < |A| := abs_pos.mpr hA
  have hm : 0 < min (estar b H) ((|A|/(2*Cstar b H))^2) := by positivity
  have ht : 0 < tailStart b H A := one_div_pos.mpr hm
  refine ⟨ht.trans hs, ?_⟩
  have hi := one_div_lt_one_div_of_lt ht hs
  simpa [tailStart] using hi

/-- Actual derivatives give strict monotonicity on the explicit tail by the
mean value theorem. The eleven root inputs are required at each small `e`. -/
theorem eventual_monotonicity (b H A : ℝ) (D : ℝ → ℝ)
    (hb : 0 < b) (hH : 1 ≤ H) (hA : A ≠ 0)
    (hinputs : ∀ e, 0 < e → e ≤ estar b H → ∃ v, RootInputs b H A e D v) :
    (0 < A → StrictAntiOn (fun s => D (1/s)) (Ioi (tailStart b H A))) ∧
    (A < 0 → StrictMonoOn (fun s => D (1/s)) (Ioi (tailStart b H A))) := by
  have hsign : ∀ s ∈ Ioi (tailStart b H A),
      (0 < A → deriv (fun t => D (1/t)) s < 0) ∧
      (A < 0 → 0 < deriv (fun t => D (1/t)) s) := by
    intro s hs
    obtain ⟨hs0, hcut⟩ := tail_domain b H A s hb hH hA hs
    have he : 1/s ≤ estar b H := (lt_of_lt_of_le hcut (min_le_left _ _)).le
    obtain ⟨v, hv⟩ := hinputs (1/s) (one_div_pos.mpr hs0) he
    exact tail_derivative_sign b H A s D v hb hH hA hs0 hcut hv
  constructor
  · intro hAp
    apply strictAntiOn_of_deriv_neg (convex_Ioi _)
    · intro s hs
      exact (differentiableAt_of_deriv_ne_zero ((hsign s hs).1 hAp).ne).continuousAt.continuousWithinAt
    · intro s hs
      exact (hsign s (interior_subset hs)).1 hAp
  · intro hAn
    apply strictMonoOn_of_deriv_pos (convex_Ioi _)
    · intro s hs
      exact (differentiableAt_of_deriv_ne_zero ((hsign s hs).2 hAn).ne').continuousAt.continuousWithinAt
    · intro s hs
      exact (hsign s (interior_subset hs)).2 hAn

/-- A strictly monotone tail approaching zero has the appropriate strict value sign. -/
theorem value_sign_from_zero_limit (f : ℝ → ℝ) (S : ℝ)
    (hlim : Tendsto f atTop (𝓝 0)) :
    (StrictAntiOn f (Ioi S) → ∀ s, S < s → 0 < f s) ∧
    (StrictMonoOn f (Ioi S) → ∀ s, S < s → f s < 0) := by
  constructor
  · intro hanti s hs
    have hs1 : s+1 ∈ Ioi S := by change S < s+1; linarith
    have hnon : 0 ≤ f (s+1) := le_of_tendsto hlim (by
      filter_upwards [eventually_gt_atTop (s+1)] with t ht
      have htS : t ∈ Ioi S := by change S < t; linarith
      exact (hanti hs1 htS ht).le)
    exact lt_of_le_of_lt hnon (hanti hs hs1 (by linarith))
  · intro hmono s hs
    have hs1 : s+1 ∈ Ioi S := by change S < s+1; linarith
    have hnon : f (s+1) ≤ 0 := ge_of_tendsto hlim (by
      filter_upwards [eventually_gt_atTop (s+1)] with t ht
      have htS : t ∈ Ioi S := by change S < t; linarith
      exact (hmono hs1 htS ht).le)
    exact lt_of_lt_of_le (hmono hs hs1 (by linarith)) hnon

/-- The value sign uses the zero limit proved from localization and `value_link`.
The eleven analytic inputs remain conditional model steps. -/
theorem eventual_value_sign (b H A : ℝ) (D : ℝ → ℝ)
    (hb : 0 < b) (hH : 1 ≤ H) (hA : A ≠ 0)
    (hinputs : ∀ e, 0 < e → e ≤ estar b H → ∃ v, RootInputs b H A e D v) :
    (0 < A → ∀ s, tailStart b H A < s → 0 < D (1/s)) ∧
    (A < 0 → ∀ s, tailStart b H A < s → D (1/s) < 0) := by
  have hm := eventual_monotonicity b H A D hb hH hA hinputs
  have hlim := root_zero_limit b H A D hb hH hinputs
  have hv := value_sign_from_zero_limit (fun s => D (1/s)) (tailStart b H A) hlim
  exact ⟨fun ha => hv.1 (hm.1 ha), fun ha => hv.2 (hm.2 ha)⟩

end Dobson.AbsentHostUnbounded

#print axioms Dobson.AbsentHostUnbounded.constants_positive

#print axioms Dobson.AbsentHostUnbounded.cutoff_bounds

#print axioms Dobson.AbsentHostUnbounded.convex_value_error_to_derivative_interval

#print axioms Dobson.AbsentHostUnbounded.root_localization

#print axioms Dobson.AbsentHostUnbounded.root_zero_limit

#print axioms Dobson.AbsentHostUnbounded.root_numerator_interval

#print axioms Dobson.AbsentHostUnbounded.root_denominator_interval

#print axioms Dobson.AbsentHostUnbounded.quotient_interval

#print axioms Dobson.AbsentHostUnbounded.threshold_derivative_interval

#print axioms Dobson.AbsentHostUnbounded.reciprocal_derivative

#print axioms Dobson.AbsentHostUnbounded.eventual_reversal_fourth_moment

#print axioms Dobson.AbsentHostUnbounded.sign_of_strict_cutoff

#print axioms Dobson.AbsentHostUnbounded.tail_derivative_sign

#print axioms Dobson.AbsentHostUnbounded.tail_domain

#print axioms Dobson.AbsentHostUnbounded.eventual_monotonicity

#print axioms Dobson.AbsentHostUnbounded.value_sign_from_zero_limit

#print axioms Dobson.AbsentHostUnbounded.eventual_value_sign
