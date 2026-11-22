import DobsonProofs.GapOneTail
import DobsonProofs.GapTwoCorrection
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# GAP 1: fixed-amplitude near-zero interval

Algebraic and inequality cores of gap1zero targets 1, 2, 4--6 and 11.
Dated closure, 2026-11-16 (`leanp1`): stationary value bounds are now proved from
primitive stationary laws by GAP 2 Q8. The physical coordinate comparison is
proved by Q10; convexity and smoothness remain explicit project-specific inputs.
-/
namespace Dobson.GapOneNearZero
open Filter Set MeasureTheory ProbabilityTheory
open scoped Topology

noncomputable def staying (q : ℝ → ℝ) (L : ℝ) : ℝ := Real.exp (-(L + q L / 2))
noncomputable def endpoint (T b : ℝ) : ℝ := Real.exp (-T-b-1)

/-- The actual parameter map has nonzero derivative, enabling local inversion. -/
theorem threshold_parametric_derivative (q : ℝ → ℝ) (L p : ℝ)
    (hq : HasStrictDerivAt q (-p) L) (hp : p < 2) :
    HasStrictDerivAt (fun x => x + q x / 2) (1-p/2) L ∧
    HasStrictDerivAt (staying q) (-staying q L * (1-p/2)) L ∧
    0 < 1-p/2 ∧ -staying q L * (1-p/2) < 0 := by
  have ht : HasStrictDerivAt (fun x => x + q x / 2) (1-p/2) L := by
    convert (hasStrictDerivAt_id L).add (hq.div_const 2) using 1
    · ext x; rfl
    · ring
  have hs : HasStrictDerivAt (staying q) (-staying q L * (1-p/2)) L := by
    convert ht.neg.exp using 1
    · rfl
    · dsimp [staying]; ring
  exact ⟨ht, hs, by linarith, mul_neg_of_neg_of_pos (by exact neg_neg_of_pos (Real.exp_pos _)) (by linarith)⟩

/-- A genuine local inverse and its derivative, not a field-arithmetic proxy.
It inverts the physical s(L) near this L. -/
theorem threshold_local_inverse (q : ℝ → ℝ) (L p : ℝ)
    (hq : HasStrictDerivAt q (-p) L) (hp : p < 2) :
    ∃ inv : ℝ → ℝ,
      (∀ᶠ x in 𝓝 L, inv (staying q x) = x) ∧
      HasDerivAt inv (-staying q L * (1-p/2))⁻¹ (staying q L) := by
  obtain ⟨_, hs, _, hneg⟩ := threshold_parametric_derivative q L p hq hp
  exact ⟨hs.localInverse _ _ _ hneg.ne, hs.eventually_left_inverse hneg.ne,
    (hs.to_localInverse hneg.ne).hasDerivAt⟩

/-- Differentiate the physical threshold through the constructed inverse.
The inverse hypotheses are obtained in `threshold_local_inverse`; transferring
this function to the ecological D additionally requires the local model identity. -/
theorem threshold_inverse_derivative (q inv : ℝ → ℝ) (L p ell : ℝ)
    (hq : HasDerivAt q (-p) L) (hp : p < 2) (hs1 : staying q L < 1)
    (hi : inv (staying q L)=L)
    (hd : HasDerivAt inv (-staying q L*(1-p/2))⁻¹ (staying q L)) :
    HasDerivAt (fun s => ell-q (inv s)-Real.log (1-s))
      (-(p/(1-p/2)-staying q L/(1-staying q L))/staying q L) (staying q L) := by
  have hs : 0 < staying q L := Real.exp_pos _
  have hden : 1-p/2 ≠ 0 := by linarith
  have hsden : 1-staying q L ≠ 0 := by linarith
  have hq' : HasDerivAt q (-p) (inv (staying q L)) := by rwa [hi]
  have hlog : HasDerivAt (fun s : ℝ => Real.log (1-s))
      ((1-staying q L)⁻¹*(-1)) (staying q L) := by
    simpa only [staying, id_eq, zero_sub, div_eq_mul_inv, mul_comm] using
      ((hasDerivAt_const _ (1:ℝ)).fun_sub (hasDerivAt_id _)).log hsden
  convert ((hasDerivAt_const _ ell).fun_sub (hq'.comp _ hd)).fun_sub hlog using 1
  · rfl
  · field_simp [hs.ne', hden, hsden]
    <;> ring

/-- Exact derivative sign equivalence once the physical derivative identity is
supplied. The denominator positivity is proved from s in (0,1), p<2. -/
theorem derivative_sign_iff (s p D' : ℝ) (hs0 : 0 < s) (hs1 : s < 1)
    (hp : p < 2) (hD : -s*D' = p/(1-p/2)-s/(1-s)) :
    D' < 0 ↔ 2*s/(2-s) < p := by
  have hd : 0 < 1-p/2 := by linarith
  have hs : 0 < 1-s := by linarith
  have hs2 : 0 < 2-s := by linarith
  have hid : (p/(1-p/2)-s/(1-s))*((1-p/2)*(1-s)) = p*(1-s/2)-s := by
    field_simp
    ring
  rw [div_lt_iff₀ hs2]
  constructor
  · intro h
    have hpos : 0 < p/(1-p/2)-s/(1-s) := by nlinarith
    have hmul := mul_pos hpos (mul_pos hd hs)
    nlinarith
  · intro h
    have hpos : 0 < p/(1-p/2)-s/(1-s) := by
      apply (mul_pos_iff_of_pos_right (mul_pos hd hs)).mp
      nlinarith
    nlinarith

/-- Actual convexity gives both secants, including both shifted arguments. -/
theorem convex_secant_bounds (q : ℝ → ℝ) (L p h : ℝ) (hh : 0 < h)
    (hc : ConvexOn ℝ univ q) (hd : HasDerivAt q (-p) L) :
    (q L-q (L+h))/h ≤ p ∧ p ≤ (q (L-h)-q L)/h := by
  have hr := hc.le_slope_of_hasDerivAt (mem_univ L) (mem_univ (L+h)) (by linarith) hd
  have hl := hc.slope_le_of_hasDerivAt (mem_univ (L-h)) (mem_univ L) (by linarith) hd
  simp only [slope_def_field, show L+h-L=h by ring, show L-(L-h)=h by ring] at hr hl
  constructor
  · rw [div_le_iff₀ hh]
    have := (le_div_iff₀ hh).mp hr
    nlinarith
  · rw [le_div_iff₀ hh]
    have := (div_le_iff₀ hh).mp hl
    nlinarith

theorem negative_derivative_antitone (q : ℝ → ℝ) (hc : ConvexOn ℝ univ q)
    (hd : Differentiable ℝ q) : Antitone (fun L => -deriv q L) := by
  intro x y hxy
  have h := hc.monotoneOn_deriv (fun x _ => hd x) (mem_univ x) (mem_univ y) hxy
  linarith

/-- The whole-cell certificate uses monotonicity of the actual staying map.
Enclosures at u,v,v+h and convexity control every point between u and v. -/
theorem whole_cell_secant_certificate (q : ℝ → ℝ) (u v L h qlo qhi shi p : ℝ)
    (hc : ConvexOn ℝ univ q) (hq : Differentiable ℝ q)
    (huL : u ≤ L) (hLv : L ≤ v) (hh : 0 < h)
    (hlo : qlo ≤ q v) (hhi : q (v+h) ≤ qhi)
    (hp : p = -deriv q L) (hsmono : AntitoneOn (staying q) (Icc u v))
    (hshi : staying q u ≤ shi) :
    (qlo-qhi)/h ≤ p ∧ staying q L ≤ shi := by
  have hs := (convex_secant_bounds q v (-deriv q v) h hh hc (by simpa using (hq v).hasDerivAt)).1
  have hm := negative_derivative_antitone q hc hq hLv
  refine ⟨?_, ?_⟩
  · rw [hp]
    exact (div_le_div_of_nonneg_right (by linarith : qlo-qhi ≤ q v-q (v+h)) hh.le).trans (hs.trans hm)
  · exact (hsmono ⟨le_rfl, huL.trans hLv⟩ ⟨huL,hLv⟩ huL).trans hshi

/-- The exact rational margin in (12). -/
theorem rational_margin : (4/9:ℝ)-9/28 = 31/252 ∧ (1/9:ℝ) < 31/252 := by norm_num

/-- Lower value estimate (7) implies the first endpoint bound. -/
theorem value_lower_margin (q V L A : ℝ) (hV : 0 ≤ V) (hL : 0 < L)
    (hA0 : 0 ≤ A) (hA : A ≤ L/4) (hq : V/(2*L+A) ≤ q) :
    4*V/(9*L) ≤ q := by
  apply le_trans _ hq
  apply (div_le_div_iff₀ (by positivity : 0 < 9*L) (by positivity : 0 < 2*L+A)).mpr
  nlinarith [mul_le_mul_of_nonneg_left hA hV]

/-- Upper value estimate at 2L, including its tail and denominator checks. -/
theorem value_upper_margin (q V L M B tail : ℝ) (hV : 0 ≤ V) (hL : 1 ≤ L)
    (hM : 0 ≤ M) (hB : 0 ≤ B) (hMB : 2*M+B ≤ L/2)
    (ht : tail ≤ V/(4*L^2)) (hq : q ≤ (V+2*M*tail)/(4*L-2*M-B)) :
    q ≤ 9*V/(28*L) := by
  have hL0 : 0 < L := by linarith
  have hd : 0 < 4*L-2*M-B := by linarith
  apply hq.trans
  apply (div_le_div_iff₀ hd (by positivity : 0 < 28*L)).mpr
  have ht' := (le_div_iff₀ (by positivity : 0 < 4*L^2)).mp ht
  have hm : 2*M ≤ L/2 := by linarith
  have htail : 2*M*tail ≤ V/(8*L) := by
    calc
      2*M*tail ≤ 2*M*(V/(4*L^2)) := mul_le_mul_of_nonneg_left ht (by positivity)
      _ ≤ (L/2)*(V/(4*L^2)) := mul_le_mul_of_nonneg_right hm (by positivity)
      _ = V/(8*L) := by field_simp; ring
  have htail' := (le_div_iff₀ (by positivity : 0 < 8*L)).mp htail
  nlinarith [mul_nonneg hV (show 0 ≤ L-1 by linarith),
    mul_le_mul_of_nonneg_left hMB (show 0 ≤ 9*V by positivity)]

/-- Convexity and two values yield a strict derivative margin, with no
differentiation of the value remainder. -/
theorem explicit_sign_margin (q : ℝ → ℝ) (V L p : ℝ) (hV : 0 < V) (hL : 0 < L)
    (hc : ConvexOn ℝ univ q) (hd : HasDerivAt q (-p) L)
    (hlo : 4*V/(9*L) ≤ q L) (hhi : q (2*L) ≤ 9*V/(28*L)) :
    31*V/(252*L^2) ≤ p ∧ V/(9*L^2) < p := by
  have hsec := (convex_secant_bounds q L p L hL hc hd).1
  rw [show L+L=2*L by ring] at hsec
  have hmargin : 31*V/(252*L^2) ≤ p := by
    calc
      31*V/(252*L^2) = (4*V/(9*L)-9*V/(28*L))/L := by field_simp; ring
      _ ≤ (q L-q (2*L))/L := div_le_div_of_nonneg_right (by linarith) hL.le
      _ ≤ p := hsec
  refine ⟨hmargin, lt_of_lt_of_le ?_ hmargin⟩
  apply (div_lt_div_iff₀ (by positivity : 0 < 9*L^2) (by positivity : 0 < 252*L^2)).mpr
  nlinarith [mul_pos hV (sq_pos_of_pos hL)]

/-- The exponential estimate used to separate the physical threshold term. -/
private theorem exponential_sign_margin_of_bound (V L s : ℝ) (hL : 0 < L)
    (hs0 : 0 < s) (hs1 : s < 1) (hs : s ≤ Real.exp (-L))
    (hpoly : L^2 ≤ Real.exp (L/2)) (htail : Real.exp (-L/2) ≤ V/36) :
    2*s/(2-s) < V/(18*L^2) := by
  have he : Real.exp (-L)*L^2 ≤ V/36 := by
    calc
      Real.exp (-L)*L^2 ≤ Real.exp (-L)*Real.exp (L/2) :=
        mul_le_mul_of_nonneg_left hpoly (Real.exp_pos _).le
      _ = Real.exp (-L/2) := by rw [← Real.exp_add]; congr 1; ring
      _ ≤ V/36 := htail
  have hsmall : 2*Real.exp (-L) ≤ V/(18*L^2) := by
    apply (le_div_iff₀ (by positivity : 0 < 18*L^2)).mpr
    nlinarith
  apply lt_of_lt_of_le _ hsmall
  apply (div_lt_iff₀ (show 0 < 2-s by linarith)).mpr
  nlinarith [Real.exp_pos (-L)]

theorem endpoint_positive (T b : ℝ) : 0 < endpoint T b := Real.exp_pos _

/-- The actual physical-coordinate comparison places the entire closed
endpoint interval beyond T. It is explicitly supplied, not inferred from s. -/
private theorem endpoint_places_L_of_coord (T b L s : ℝ) (hb : 0 ≤ b) (hs0 : 0 < s)
    (hs : s ≤ endpoint T b)
    (hcoord : -Real.log s-(b+Real.log 2)/2 ≤ L) : T < L := by
  have hlog : Real.log s ≤ -T-b-1 := by
    simpa [endpoint] using Real.log_le_log hs0 hs
  have htwo : Real.log 2 ≤ 1 := by
    have := Real.log_le_sub_one_of_pos (by norm_num : (0:ℝ)<2)
    norm_num at this
    exact this
  linarith

/-- The explicit endpoint places the actual stationary threshold coordinate
beyond T. The coordinate comparison is proved by the constant-host bracket Q10.
A positive transfer candidate and its stationary zero-growth law are primitive
model data; existence or positivity of a separately selected threshold is not asserted. -/
theorem endpoint_places_L {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T δ s c : ℝ)
    (hT : 0 < T) (hδ0 : 0 < δ) (hδ1 : δ < 1) (hs0 : 0 < s)
    (hs : s ≤ endpoint T (Real.artanh δ)) (hc : 0 < c)
    (X Zprev Zcurr : Ω → ℝ)
    (hstat : IdentDistrib
      (fun ω =>
        (GapTwoCorrection.Fstep
          (Real.log (1 / s) + (Real.log c - GapTwoCorrection.ell δ) / 2)
          (Real.artanh δ * Zcurr ω) (X ω), Zcurr ω))
      (fun ω => (X ω, Zprev ω)) μ μ)
    (hbinary : ∀ᵐ ω ∂μ, Zcurr ω = 1 ∨ Zcurr ω = -1)
    (hzero : (Real.log c - GapTwoCorrection.ell δ) + 2 * (∫ ω, X ω ∂μ) = 0) :
    T < Real.log (1 / s) + (Real.log c - GapTwoCorrection.ell δ) / 2 := by
  have hb : 0 ≤ Real.artanh δ := (Real.artanh_pos (mem_Ioo.mpr ⟨hδ0, hδ1⟩)).le
  have hlog2 : Real.log 2 ≤ 1 := by
    have h := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2)
    norm_num at h
    exact h
  have hhalf : s ≤ 1 / 2 := by
    calc
      s ≤ endpoint T (Real.artanh δ) := hs
      _ ≤ Real.exp (-Real.log 2) := by
        apply Real.exp_le_exp.mpr
        linarith
      _ = 1 / 2 := by rw [Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 2)]; norm_num
  have hs1 : s < 1 := by linarith
  have hbracket := (GapTwoCorrection.Q10 μ δ s c hδ0 hδ1 hs0 hs1 hc
    X Zprev Zcurr hstat hbinary hzero).2.1.1
  have hlogstay : -Real.log 2 ≤ Real.log (1 - s) := by
    have h := Real.log_le_log (by norm_num : (0 : ℝ) < 1 / 2)
      (show 1 / 2 ≤ 1 - s by linarith)
    simpa only [Real.log_div (by norm_num : (1 : ℝ) ≠ 0)
      (by norm_num : (2 : ℝ) ≠ 0), Real.log_one, zero_sub] using h
  apply endpoint_places_L_of_coord T (Real.artanh δ) _ s hb hs0 hs
  rw [Real.log_div one_ne_zero hs0.ne', Real.log_one]
  linarith

/-- Tangent-line bound for log, with a positive reference value. -/
theorem log_tangent_bound (x c : ℝ) (hx : 0 < x) (hc : 0 < c) :
    Real.log x ≤ Real.log c+x/c-1 := by
  have h := Real.log_le_sub_one_of_pos (div_pos hx hc)
  rw [Real.log_div hx.ne' hc.ne'] at h
  linarith

theorem log_scalar_bounds (L : ℝ) (hL : 128 ≤ L) :
    4*Real.log (2*L) ≤ L/4 ∧ L^2 ≤ Real.exp (L/2) := by
  have hL0 : 0 < L := by linarith
  have hlog256 : Real.log (256:ℝ) ≤ 8 := by
    have he : (256:ℝ)=2^8 := by norm_num
    rw [he, Real.log_pow]
    norm_num only [Nat.cast_ofNat]
    linarith [Real.log_two_lt_d9]
  have h := log_tangent_bound (2*L) 256 (by positivity) (by norm_num)
  have hlog : 4*Real.log (2*L) ≤ L/4 := by linarith
  refine ⟨hlog, ?_⟩
  have hmono : Real.log L ≤ Real.log (2*L) := Real.log_le_log hL0 (by linarith)
  have hex : Real.exp (2*Real.log L) = L^2 := by
    rw [show 2*Real.log L = Real.log L+Real.log L by ring, Real.exp_add, Real.exp_log hL0]
    ring
  rw [← hex]
  exact Real.exp_le_exp.mpr (by linarith)

noncomputable def kConst (b V : ℝ) : ℝ := b+Real.log (1+1/V)
noncomputable def jConst (b V B : ℝ) : ℝ := 2*kConst b V+B
noncomputable def cutoff (b V A B : ℝ) : ℝ :=
  max (max 128 (4*A)) (max (4*jConst b V B) (2*Real.log (36/V)))

/-- The displayed exponential tail identity, with all denominators nonzero. -/
theorem split_tail_identity (b V x : ℝ) (hV : 0 < V) (hx : 0 < x) :
    Real.exp (b-(2*Real.log x+kConst b V)) = V/((1+V)*x^2) := by
  have harg : 0 < 1+1/V := by positivity
  rw [show b-(2*Real.log x+kConst b V) = -(Real.log (1+1/V)) -
    (Real.log x+Real.log x) by dsimp [kConst]; ring]
  rw [Real.exp_sub, Real.exp_neg, Real.exp_log harg, Real.exp_add, Real.exp_log hx]
  field_simp
  ring

theorem cutoff_bounds (b V A B L : ℝ) (hV : 0 < V) (hL : cutoff b V A B ≤ L) :
    128 ≤ L ∧ A ≤ L/4 ∧ jConst b V B ≤ L/4 ∧
    L^2 ≤ Real.exp (L/2) ∧ Real.exp (-L/2) ≤ V/36 := by
  have h := (max_le_iff.mp hL)
  have h128 := (max_le_iff.mp h.1).1
  have hA := (max_le_iff.mp h.1).2
  have hJ := (max_le_iff.mp h.2).1
  have ht := (max_le_iff.mp h.2).2
  refine ⟨h128, by linarith, by linarith, (log_scalar_bounds L h128).2, ?_⟩
  have he : Real.exp (-L/2) ≤ Real.exp (-Real.log (36/V)) :=
    Real.exp_le_exp.mpr (by linarith)
  rw [Real.exp_neg, Real.exp_log (by positivity : 0 < 36/V)] at he
  simpa using he

/-- Source estimate (7) at L and 2L gives both rational margins at the exact
registered cutoff. The boundary-split estimate itself is the probability input. -/
private theorem value_margins_of_bounds (q : ℝ → ℝ) (b V A B L : ℝ)
    (hb : 0 ≤ b) (hV : 0 < V) (hA : 0 ≤ A) (hB : 0 ≤ B)
    (hL : cutoff b V A B ≤ L)
    (hlower : V/(2*L+A) ≤ q L)
    (hupper : ∀ M : ℝ, 1 ≤ M → 0 < 4*L-2*M-B →
      q (2*L) ≤ (V+2*M*Real.exp (b-M))/(4*L-2*M-B)) :
    4*V/(9*L) ≤ q L ∧ q (2*L) ≤ 9*V/(28*L) := by
  obtain ⟨h128,hAc,hJ,_,_⟩ := cutoff_bounds b V A B L hV hL
  have hL0 : 0 < L := by linarith
  have hk : 0 ≤ kConst b V := by
    unfold kConst
    exact add_nonneg hb (Real.log_nonneg (by linarith [div_pos (by norm_num : (0:ℝ)<1) hV]))
  have hlog : 1 ≤ Real.log (2*L) := by
    have hm := Real.log_le_log (by norm_num : (0:ℝ)<4) (show (4:ℝ)≤2*L by linarith)
    have he : Real.log (4:ℝ)=2*Real.log 2 := by
      rw [show (4:ℝ)=2^2 by norm_num, Real.log_pow]; norm_num
    rw [he] at hm
    linarith [Real.log_two_gt_d9]
  let M := 2*Real.log (2*L)+kConst b V
  have hM : 1 ≤ M := by dsimp [M]; linarith
  have hMB : 2*M+B ≤ L/2 := by
    have hl := (log_scalar_bounds L h128).1
    dsimp [M, jConst] at *
    linarith
  have htail : Real.exp (b-M) ≤ V/(4*L^2) := by
    dsimp [M]
    rw [split_tail_identity b V (2*L) hV (by positivity)]
    apply (div_le_div_iff₀ (by positivity : 0 < (1+V)*(2*L)^2)
      (by positivity : 0 < 4*L^2)).mpr
    nlinarith [mul_nonneg (sq_nonneg V) (sq_nonneg L)]
  exact ⟨value_lower_margin (q L) V L A hV.le hL0 hA hAc hlower,
    value_upper_margin (q (2*L)) V L M B (Real.exp (b-M)) hV.le (by linarith)
      (by linarith) hB hMB htail (hupper M hM (by linarith))⟩

/-- Primitive stationary normalized host laws, indexed by the real coordinate `L`.
No boundary estimate, moment identity, smoothness or convexity is stored here. -/
structure StationaryFamily (Ω : Type*) [MeasurableSpace Ω] (b φ : ℝ) where
  μ : ℝ → Measure Ω
  probability : ∀ L, IsProbabilityMeasure (μ L)
  X : ℝ → Ω → ℝ
  Zprev : ℝ → Ω → ℝ
  Zcurr : ℝ → Ω → ℝ
  stationary : ∀ L, IdentDistrib
    (fun ω => (GapTwoCorrection.Fstep L (b * Zcurr L ω) (X L ω), Zcurr L ω))
    (fun ω => (X L ω, Zprev L ω)) (μ L) (μ L)
  binary : ∀ L, ∀ᵐ ω ∂μ L, Zcurr L ω = 1 ∨ Zcurr L ω = -1
  host_mean : ∀ L, ∫ ω, Zcurr L ω ∂μ L = 0
  markov : ∀ L, ∫ ω, X L ω * Zcurr L ω ∂μ L =
    φ * ∫ ω, X L ω * Zprev L ω ∂μ L

/-- The actual stationary reflection expectation in the manuscript. -/
noncomputable def stationaryValue {Ω : Type*} [MeasurableSpace Ω] {b φ : ℝ}
    (law : StationaryFamily Ω b φ) (L : ℝ) : ℝ :=
  ∫ ω, GapTwoCorrection.Qstep L (law.X L ω) (b * law.Zcurr L ω) ∂law.μ L

/-- Alternating-sum variance of the normalized binary host. -/
noncomputable def stationaryVariance (b φ : ℝ) : ℝ := b ^ 2 * (1 - φ) / (1 + φ)

/-- Q8 supplies the full boundary split for the actual stationary expectation. -/
theorem stationary_boundary_bounds {Ω : Type*} [MeasurableSpace Ω] {b φ : ℝ}
    (law : StationaryFamily Ω b φ) (hb : 0 ≤ b) (hφlo : -1 < φ) (hφhi : φ < 1)
    (L : ℝ) (hL : 1 ≤ L) :
    stationaryVariance b φ / (2 * L + GapTwoCorrection.Cplus b φ) ≤ stationaryValue law L ∧
      ∀ M : ℝ, 1 ≤ M → 0 < 2 * L - 2 * M - GapTwoCorrection.Cminus b φ →
        stationaryValue law L ≤
          (stationaryVariance b φ + 2 * M * Real.exp (b - M)) /
            (2 * L - 2 * M - GapTwoCorrection.Cminus b φ) := by
  have : IsProbabilityMeasure (law.μ L) := law.probability L
  obtain ⟨_, hlo, hhi⟩ := GapTwoCorrection.Q8 (law.μ L) L b φ hb hφlo hφhi hL
    (law.X L) (law.Zprev L) (law.Zcurr L) (law.stationary L) (law.binary L)
    (law.host_mean L) (law.markov L)
  exact ⟨hlo.2, fun M hM hd => (hhi M hM).2 hd⟩

/-- Signs of the explicit manuscript constants. -/
theorem stationary_constants_positive (b φ : ℝ) (hb : 0 < b)
    (hφlo : -1 < φ) (hφhi : φ < 1) :
    0 < stationaryVariance b φ ∧ 0 ≤ GapTwoCorrection.Cplus b φ ∧
      0 ≤ GapTwoCorrection.Cminus b φ := by
  have hp : 0 < 1 + φ := by linarith
  have hv : 0 < stationaryVariance b φ := by
    exact div_pos (mul_pos (sq_pos_of_pos hb) (by linarith)) hp
  have hq : 0 ≤ GapTwoCorrection.Qstar b := by
    exact Real.log_nonneg (by linarith [Real.exp_pos (2 * b + 1)])
  refine ⟨hv, ?_, ?_⟩
  · unfold GapTwoCorrection.Cplus
    positivity
  · unfold GapTwoCorrection.Cminus
    exact add_nonneg (div_nonneg (by positivity) hp.le) hq

/-- The actual stationary soft-reflection expectation is nonnegative. -/
theorem stationary_value_nonneg {Ω : Type*} [MeasurableSpace Ω] {b φ : ℝ}
    (law : StationaryFamily Ω b φ) (L : ℝ) : 0 ≤ stationaryValue law L := by
  apply integral_nonneg
  intro ω
  exact Real.log_nonneg (by linarith [Real.exp_pos (law.X L ω - b * law.Zcurr L ω - L)])

/-- The physical normalization `s(L)=exp(-L-q(L)/2)` supplies the comparison
with `exp(-L)` directly from the stationary reflection observable. -/
theorem stationary_staying_le {Ω : Type*} [MeasurableSpace Ω] {b φ : ℝ}
    (law : StationaryFamily Ω b φ) (L : ℝ) :
    staying (stationaryValue law) L ≤ Real.exp (-L) := by
  apply Real.exp_le_exp.mpr
  linarith [stationary_value_nonneg law L]

/-- The exponential separation for the actual physical parameter map.
The comparison with `exp(-L)` is proved, not an input. -/
theorem exponential_sign_margin {Ω : Type*} [MeasurableSpace Ω] {b φ : ℝ}
    (law : StationaryFamily Ω b φ) (L : ℝ) (hL : 0 < L)
    (hpoly : L ^ 2 ≤ Real.exp (L / 2))
    (htail : Real.exp (-L / 2) ≤ stationaryVariance b φ / 36) :
    2 * staying (stationaryValue law) L / (2 - staying (stationaryValue law) L) <
      stationaryVariance b φ / (18 * L ^ 2) := by
  have hs := stationary_staying_le law L
  have hs1 : staying (stationaryValue law) L < 1 :=
    hs.trans_lt (Real.exp_lt_one_iff.mpr (by linarith))
  exact exponential_sign_margin_of_bound (stationaryVariance b φ) L
    (staying (stationaryValue law) L) hL (Real.exp_pos _) hs1 hs hpoly htail

/-- Both registered margins follow from Q8 for the actual stationary model. -/
theorem registered_value_margins {Ω : Type*} [MeasurableSpace Ω] {b φ : ℝ}
    (law : StationaryFamily Ω b φ) (hb : 0 < b) (hφlo : -1 < φ) (hφhi : φ < 1)
    (L : ℝ)
    (hL : cutoff b (stationaryVariance b φ) (GapTwoCorrection.Cplus b φ)
      (GapTwoCorrection.Cminus b φ) ≤ L)
 :
    4 * stationaryVariance b φ / (9 * L) ≤ stationaryValue law L ∧
      stationaryValue law (2 * L) ≤ 9 * stationaryVariance b φ / (28 * L) := by
  obtain ⟨hV, hA, hB⟩ := stationary_constants_positive b φ hb hφlo hφhi
  have h128 := (cutoff_bounds b (stationaryVariance b φ) (GapTwoCorrection.Cplus b φ)
    (GapTwoCorrection.Cminus b φ) L hV hL).1
  have hupper : ∀ M : ℝ, 1 ≤ M → 0 < 4 * L - 2 * M - GapTwoCorrection.Cminus b φ →
      stationaryValue law (2 * L) ≤
        (stationaryVariance b φ + 2 * M * Real.exp (b - M)) /
          (4 * L - 2 * M - GapTwoCorrection.Cminus b φ) := by
    intro M hM hd
    simpa only [show (2 : ℝ) * (2 * L) = 4 * L by ring] using
      (stationary_boundary_bounds law hb.le hφlo hφhi (2 * L) (by linarith)).2 M hM
        (by nlinarith)
  exact value_margins_of_bounds (stationaryValue law) b (stationaryVariance b φ)
    (GapTwoCorrection.Cplus b φ) (GapTwoCorrection.Cminus b φ) L hb.le hV hA hB hL
    (stationary_boundary_bounds law hb.le hφlo hφhi L (by linarith)).1 hupper

/-- Conditional fixed-amplitude interval. Every bridge is quantified on its
actual domain. The endpoint is explicit and positive; no exceptional s is used. -/
theorem fixed_amplitude_near_zero (q : ℝ → ℝ) (D' Lof pof : ℝ → ℝ) (T b V : ℝ)
    (hT : 0 < T) (hb : 0 ≤ b) (hV : 0 < V) (hc : ConvexOn ℝ univ q)
    (hmodel : ∀ s, 0 < s → s ≤ endpoint T b →
      s < 1 ∧ -Real.log s-(b+Real.log 2)/2 ≤ Lof s ∧
      pof s < 2 ∧ HasDerivAt q (-pof s) (Lof s) ∧
      -s*D' s = pof s/(1-pof s/2)-s/(1-s) ∧ s ≤ Real.exp (-Lof s))
    (hvalues : ∀ L, T ≤ L → 4*V/(9*L) ≤ q L ∧ q (2*L) ≤ 9*V/(28*L))
    (hexp : ∀ L, T ≤ L → L^2 ≤ Real.exp (L/2) ∧ Real.exp (-L/2) ≤ V/36) :
    0 < endpoint T b ∧ ∀ s, 0 < s → s ≤ endpoint T b → D' s < 0 := by
  refine ⟨endpoint_positive T b, ?_⟩
  intro s hs0 hs
  obtain ⟨hs1, hcoord, hp, hd, hD, hsexp⟩ := hmodel s hs0 hs
  have hL : T < Lof s := endpoint_places_L_of_coord T b (Lof s) s hb hs0 hs hcoord
  have hL0 : 0 < Lof s := hT.trans hL
  have hm := (explicit_sign_margin q V (Lof s) (pof s) hV hL0 hc hd
    (hvalues _ hL.le).1 (hvalues _ hL.le).2).2
  have he := exponential_sign_margin_of_bound V (Lof s) s hL0 hs0 hs1 hsexp
    (hexp _ hL.le).1 (hexp _ hL.le).2
  apply (derivative_sign_iff s (pof s) (D' s) hs0 hs1 hp hD).mpr
  apply he.trans (lt_of_le_of_lt _ hm)
  apply div_le_div_of_nonneg_left hV.le (by positivity)
  nlinarith [sq_nonneg (Lof s)]

/-- The manuscript's explicit T and s₀ for the actual stationary model.
The boundary split is proved by Q8; convexity and the physical differential model
remain the explicit project-specific inputs shown in the statement. -/
theorem registered_near_zero {Ω : Type*} [MeasurableSpace Ω] {b φ : ℝ}
    (law : StationaryFamily Ω b φ) (D' Lof pof : ℝ → ℝ)
    (hb : 0 < b) (hφlo : -1 < φ) (hφhi : φ < 1)
    (hc : ConvexOn ℝ univ (stationaryValue law))
    (hmodel : ∀ s, 0 < s →
      s ≤ endpoint (cutoff b (stationaryVariance b φ) (GapTwoCorrection.Cplus b φ)
        (GapTwoCorrection.Cminus b φ)) b →
      s < 1 ∧ -Real.log s - (b + Real.log 2) / 2 ≤ Lof s ∧
      pof s < 2 ∧ HasDerivAt (stationaryValue law) (-pof s) (Lof s) ∧
      -s * D' s = pof s / (1 - pof s / 2) - s / (1 - s) ∧ s ≤ Real.exp (-Lof s)) :
    0 < endpoint (cutoff b (stationaryVariance b φ) (GapTwoCorrection.Cplus b φ)
        (GapTwoCorrection.Cminus b φ)) b ∧
      ∀ s, 0 < s →
        s ≤ endpoint (cutoff b (stationaryVariance b φ) (GapTwoCorrection.Cplus b φ)
          (GapTwoCorrection.Cminus b φ)) b → D' s < 0 := by
  let V := stationaryVariance b φ
  let A := GapTwoCorrection.Cplus b φ
  let B := GapTwoCorrection.Cminus b φ
  obtain ⟨hV, _, _⟩ := stationary_constants_positive b φ hb hφlo hφhi
  have hT : 0 < cutoff b V A B := by
    have h := (cutoff_bounds b V A B (cutoff b V A B) hV le_rfl).1
    linarith
  apply fixed_amplitude_near_zero (stationaryValue law) D' Lof pof (cutoff b V A B)
    b V hT hb.le hV hc hmodel
  · intro L hL
    exact registered_value_margins law hb hφlo hφhi L hL
  · intro L hL
    exact (cutoff_bounds b V A B L hV hL).2.2.2

end Dobson.GapOneNearZero

#print axioms Dobson.GapOneNearZero.threshold_parametric_derivative
#print axioms Dobson.GapOneNearZero.threshold_local_inverse
#print axioms Dobson.GapOneNearZero.threshold_inverse_derivative
#print axioms Dobson.GapOneNearZero.derivative_sign_iff
#print axioms Dobson.GapOneNearZero.convex_secant_bounds
#print axioms Dobson.GapOneNearZero.negative_derivative_antitone
#print axioms Dobson.GapOneNearZero.whole_cell_secant_certificate
#print axioms Dobson.GapOneNearZero.rational_margin
#print axioms Dobson.GapOneNearZero.value_lower_margin
#print axioms Dobson.GapOneNearZero.value_upper_margin
#print axioms Dobson.GapOneNearZero.explicit_sign_margin
#print axioms Dobson.GapOneNearZero.exponential_sign_margin
#print axioms Dobson.GapOneNearZero.endpoint_positive
#print axioms Dobson.GapOneNearZero.endpoint_places_L
#print axioms Dobson.GapOneNearZero.log_tangent_bound
#print axioms Dobson.GapOneNearZero.log_scalar_bounds
#print axioms Dobson.GapOneNearZero.split_tail_identity
#print axioms Dobson.GapOneNearZero.cutoff_bounds
#print axioms Dobson.GapOneNearZero.stationary_value_nonneg
#print axioms Dobson.GapOneNearZero.stationary_staying_le
#print axioms Dobson.GapOneNearZero.stationary_boundary_bounds
#print axioms Dobson.GapOneNearZero.stationary_constants_positive
#print axioms Dobson.GapOneNearZero.registered_value_margins
#print axioms Dobson.GapOneNearZero.fixed_amplitude_near_zero
#print axioms Dobson.GapOneNearZero.registered_near_zero
