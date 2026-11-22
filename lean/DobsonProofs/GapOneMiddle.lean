import DobsonProofs.GapOneTail
import Mathlib.Analysis.Calculus.MeanValue

/-!
# GAP 1: interval certificates and the persistence comparison

Finite cores of gap1mid targets 2, 4, 5, 7, 8 and 11. A box supplies sound
rate/tangent enclosures; no binary64 arithmetic or 196,665-box cover is trusted.
Stationary normalization and the two run-comparison expectation estimates are
explicit analytic inputs. See the lean5 STATUS ledger.
-/
namespace Dobson.GapOneMiddle
open Dobson.GapOneTail (Gfun)
open scoped BigOperators

/-- The quadratic characterizes both invariant endpoints; positivity selects
the intended root, without trusting a floating-point square root. -/
theorem invariant_endpoints (A B l : ℝ) (hA : 0 < A) (hAB : A ≤ B)
    (hl : 0 < l) (hroot : l ^ 2 + (B - A - 1) * l - B = 0) :
    1 < l ∧ l ≤ l + B - A ∧
      l = 1 + A / (l + B - A) ∧ l + B - A = 1 + B / l := by
  have hu : 0 < l + B - A := by linarith
  have hid : (l - 1) * (l + B - A) = A := by nlinarith
  have hl1 : 1 < l := by nlinarith
  refine ⟨hl1, by linarith, ?_, ?_⟩
  · field_simp
    nlinarith
  · field_simp
    nlinarith

/-- Every parameter and every entering ratio in the closed box are covered. -/
theorem ratio_interval (A B l u a y : ℝ) (hl : 0 < l) (hy : l ≤ y)
    (hyu : y ≤ u) (hA : 0 ≤ A) (ha : A ≤ a) (haB : a ≤ B)
    (hlo : l = 1 + A / u) (hhi : u = 1 + B / l) :
    l ≤ 1 + a / y ∧ 1 + a / y ≤ u := by
  have hy0 : 0 < y := hl.trans_le hy
  have hu0 : 0 < u := hy0.trans_le hyu
  constructor
  · rw [hlo]
    gcongr; linarith
  · rw [hhi]
    gcongr; linarith

/-- Both arguments of the tangent update have their correct monotonicity. -/
theorem tangent_interval (alo ahi a elo ehi e : ℝ)
    (hal : 0 ≤ alo) (ha : alo ≤ a) (hah : a ≤ ahi)
    (hel : elo ≤ e) (heh : e ≤ ehi) (he1 : ehi ≤ 1) :
    alo * (1 - ehi) ≤ a * (1 - e) ∧
      a * (1 - e) ≤ ahi * (1 - elo) := by
  constructor <;> gcongr <;> linarith

/-- Finite conditional covers need no independence of the entering history. -/
theorem weighted_enclosure {ι : Type*} (S : Finset ι) (w f lo hi : ι → ℝ)
    (hw : ∀ i ∈ S, 0 ≤ w i) (hf : ∀ i ∈ S, lo i ≤ f i ∧ f i ≤ hi i) :
    (∑ i ∈ S, w i * lo i) ≤ ∑ i ∈ S, w i * f i ∧
      (∑ i ∈ S, w i * f i) ≤ ∑ i ∈ S, w i * hi i := by
  constructor
  · exact Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (hf i hi).1 (hw i hi)
  · exact Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (hf i hi).2 (hw i hi)

/-- Convex hull bound used after each exact Bernstein subdivision. -/
theorem convex_hull_bound {ι : Type*} (S : Finset ι) (w f : ι → ℝ) (lo hi : ℝ)
    (hw : ∀ i ∈ S, 0 ≤ w i) (hsum : ∑ i ∈ S, w i = 1)
    (hf : ∀ i ∈ S, lo ≤ f i ∧ f i ≤ hi) :
    lo ≤ ∑ i ∈ S, w i * f i ∧ (∑ i ∈ S, w i * f i) ≤ hi := by
  simpa [← Finset.sum_mul, hsum] using
    weighted_enclosure S w f (fun _ => lo) (fun _ => hi) hw hf

/-- G increases with R on the full closed tangent interval. -/
theorem G_mono_R (R₀ R V : ℝ) (hR : R₀ ≤ R) (hV : V ≤ 1) :
    Gfun R₀ V ≤ Gfun R V := by
  have he : Real.exp (-R) ≤ Real.exp (-R₀) := Real.exp_le_exp.mpr (by linarith)
  unfold Gfun
  nlinarith [mul_nonneg (sub_nonneg.mpr he) (sub_nonneg.mpr hV)]

/-- G decreases with V whenever the rate is nonnegative. -/
theorem G_antitone_V (R V₀ V : ℝ) (hR : 0 ≤ R) (hV : V₀ ≤ V) :
    Gfun R V ≤ Gfun R V₀ := by
  have he : Real.exp (-R) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  unfold Gfun
  nlinarith [mul_nonneg (sub_nonneg.mpr hV) (show 0 ≤ 2 - Real.exp (-R) by linarith)]

/-- The per-box certificate, with closed enclosure inequalities. Its only
model input is the actual derivative identity; enclosures are explicit. -/
theorem box_certificate (Rlo R V Vhi s D' : ℝ)
    (hRlo : 0 ≤ Rlo) (hR : Rlo ≤ R) (hV : 0 < V)
    (hVhi : V ≤ Vhi) (hVhi1 : Vhi ≤ 1)
    (hs0 : 0 < s) (hs1 : s < 1)
    (hD : D' = -Gfun R V / (s * (1 - s) * V))
    (hcert : 0 < Gfun Rlo Vhi) : D' < 0 := by
  have hG : 0 < Gfun R V := hcert.trans_le
    ((G_mono_R Rlo R Vhi hR hVhi1).trans (G_antitone_V R V Vhi (hRlo.trans hR) hVhi))
  rw [hD]
  exact div_neg_of_neg_of_pos (neg_neg_of_pos hG) (by positivity)

/-- A box certificate applied to an actual derivative of a real threshold. -/
theorem box_derivative_neg (D : ℝ → ℝ) (Rlo R V Vhi s : ℝ)
    (hRlo : 0 ≤ Rlo) (hR : Rlo ≤ R) (hV : 0 < V)
    (hVhi : V ≤ Vhi) (hVhi1 : Vhi ≤ 1) (hs0 : 0 < s) (hs1 : s < 1)
    (hD : HasDerivAt D (-Gfun R V/(s*(1-s)*V)) s)
    (hcert : 0 < Gfun Rlo Vhi) : deriv D s < 0 := by
  rw [hD.deriv]
  exact box_certificate Rlo R V Vhi s _ hRlo hR hV hVhi hVhi1 hs0 hs1 rfl hcert

/-- Algebraic strict mixture gap. Writing a,b for exp(-r₋/2), exp(-r₊/2)
avoids any unproved strict-concavity premise. -/
theorem mixture_gap_identity (a b : ℝ) (ha : a ^ 2 ≠ 2) (hb : b ^ 2 ≠ 2) :
    1 - a*b - (2-a*b) * (((1-a^2)/(2-a^2)+(1-b^2)/(2-b^2))/2) =
      (a-b)^2 * (2+a*b) / (2*(2-a^2)*(2-b^2)) := by
  have ha' : 2-a^2 ≠ 0 := by exact sub_ne_zero.mpr (Ne.symm ha)
  have hb' : 2-b^2 ≠ 0 := sub_ne_zero.mpr (Ne.symm hb)
  field_simp
  ring

theorem mixture_gap_positive (a b : ℝ) (ha0 : 0 < a) (hb0 : 0 < b)
    (ha1 : a ≤ 1) (hb1 : b ≤ 1) (hab : a ≠ b) :
    0 < 1 - a*b - (2-a*b) * (((1-a^2)/(2-a^2)+(1-b^2)/(2-b^2))/2) := by
  have ha2 : a^2 < 2 := by nlinarith
  have hb2 : b^2 < 2 := by nlinarith
  rw [mixture_gap_identity a b (ne_of_lt ha2) (ne_of_lt hb2)]
  exact div_pos (mul_pos (sq_pos_of_ne_zero (sub_ne_zero.mpr hab)) (by positivity))
    (by positivity)

/-- The explicit comparison constant is positive throughout its open domain. -/
theorem persistence_constant_pos (M κ : ℝ) (hM : 0 ≤ M) (hκ : 0 < κ)
    (hκ1 : κ < 1) :
    0 < M*κ/(1-κ) + 2*(κ/(1-κ)+M*κ/(1-κ)^2) := by positivity

theorem persistence_interval_nonempty (G₀ C : ℝ) (hG : 0 < G₀) (hC : 0 < C) :
    ∃ φ : ℝ, 0 ≤ φ ∧ φ < 1 ∧ 1-φ ≤ G₀/C := by
  let ε := min (1/2:ℝ) (G₀/(2*C))
  have he0 : 0 < ε := lt_min (by norm_num) (by positivity)
  have he1 : ε ≤ 1/2 := min_le_left _ _
  have heG : ε ≤ G₀/(2*C) := min_le_right _ _
  refine ⟨1-ε, by linarith, by linarith, ?_⟩
  have hhalf : G₀/(2*C) ≤ G₀/C := by
    exact div_le_div_of_nonneg_left hG.le hC (by linarith)
  linarith

/-- Finite scalar form of the two geometric-run expectation bounds. -/
theorem geometric_denominator_bounds (ε κ : ℝ) (hε : 0 ≤ ε)
    (hκ : 0 ≤ κ) (hκ1 : κ < 1) :
    ε*κ/(1-(1-ε)*κ) ≤ ε*κ/(1-κ) ∧
    ε*κ/(1-(1-ε)*κ)^2 ≤ ε*κ/(1-κ)^2 := by
  have hd : 1-κ ≤ 1-(1-ε)*κ := by nlinarith
  have hd0 : 0 < 1-κ := by linarith
  constructor
  · exact div_le_div_of_nonneg_left (by positivity) hd0 hd
  · exact div_le_div_of_nonneg_left (by positivity) (by positivity)
      (pow_le_pow_left₀ hd0.le hd 2)

/-- Persistence criterion (6), including its equality boundary. The comparison
estimate is an explicit analytic handoff from the geometric run calculation. -/
theorem high_persistence_criterion (G G₀ C φ : ℝ) (hG₀ : 0 < G₀) (hC : 0 < C)
    (hcomparison : G₀ - ((1-φ)/2)*C ≤ G)
    (hφ : 1-φ ≤ G₀/C) : G₀/2 ≤ G ∧ 0 < G := by
  have h := (le_div_iff₀ hC).mp hφ
  constructor <;> nlinarith

/-- The exponential coordinate is 1-Lipschitz on nonnegative rates. -/
theorem exp_neg_lipschitz (R R₀ : ℝ) (hR : 0 ≤ R) (hR₀ : 0 ≤ R₀) :
    |Real.exp (-R)-Real.exp (-R₀)| ≤ |R-R₀| := by
  have hd : ∀ x ∈ Set.Ici (0:ℝ), HasDerivWithinAt (fun x : ℝ => Real.exp (-x))
      (-Real.exp (-x)) (Set.Ici 0) x := by
    intro x _
    convert ((hasDerivAt_id x).neg.exp).hasDerivWithinAt using 1 <;> simp
  have hb : ∀ x ∈ Set.Ici (0:ℝ), ‖-Real.exp (-x)‖ ≤ (1:ℝ) := by
    intro x hx
    simpa using Real.exp_le_one_iff.mpr (show -x ≤ 0 by simpa using hx)
  simpa [Real.norm_eq_abs] using Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    hd hb (convex_Ici (0:ℝ)) hR₀ hR

/-- The exact constants one and two in the comparison error. -/
theorem G_lipschitz (R R₀ V V₀ : ℝ) (hR : 0 ≤ R) (hR₀ : 0 ≤ R₀)
    (hV₀ : 0 ≤ V₀) (hV₀1 : V₀ ≤ 1) :
    |Gfun R V-Gfun R₀ V₀| ≤ |R-R₀| +2*|V-V₀| := by
  have he := exp_neg_lipschitz R R₀ hR hR₀
  have hex : Real.exp (-R) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hex0 := Real.exp_pos (-R)
  have hid : Gfun R V-Gfun R₀ V₀ =
      (Real.exp (-R₀)-Real.exp (-R))*(1-V₀)-(2-Real.exp (-R))*(V-V₀) := by
    unfold Gfun; ring
  rw [hid]
  calc
    _ ≤ |(Real.exp (-R₀)-Real.exp (-R))*(1-V₀)| +
        |(2-Real.exp (-R))*(V-V₀)| := abs_sub _ _
    _ = |Real.exp (-R)-Real.exp (-R₀)| *(1-V₀)+(2-Real.exp (-R))*|V-V₀| := by
      rw [abs_mul, abs_mul, abs_sub_comm (Real.exp (-R₀)),
        abs_of_nonneg (by linarith : 0 ≤ 1-V₀), abs_of_pos (by linarith : 0 < 2-Real.exp (-R))]
    _ ≤ |R-R₀| +2*|V-V₀| := by
      nlinarith [abs_nonneg (R-R₀), abs_nonneg (V-V₀),
        mul_le_mul_of_nonneg_right he (show 0 ≤ 1-V₀ by linarith)]

/-- Assembly from the two run-comparison expectation bounds. Their probability
derivation is the only additional handoff beyond the stationary model. -/
theorem high_persistence_from_errors (R R₀ V V₀ CR CV φ : ℝ)
    (hR : 0 ≤ R) (hR₀ : 0 ≤ R₀) (hV₀ : 0 ≤ V₀) (hV₀1 : V₀ ≤ 1)
    (hG₀ : 0 < Gfun R₀ V₀) (hC : 0 < CR+2*CV)
    (herrorR : |R-R₀| ≤ ((1-φ)/2)*CR)
    (herrorV : |V-V₀| ≤ ((1-φ)/2)*CV)
    (hφ : 1-φ ≤ Gfun R₀ V₀/(CR+2*CV)) :
    Gfun R₀ V₀/2 ≤ Gfun R V ∧ 0 < Gfun R V := by
  apply high_persistence_criterion _ _ _ _ hG₀ hC _ hφ
  have h := (abs_le.mp (G_lipschitz R R₀ V V₀ hR hR₀ hV₀ hV₀1)).1
  nlinarith

end Dobson.GapOneMiddle

#print axioms Dobson.GapOneMiddle.invariant_endpoints
#print axioms Dobson.GapOneMiddle.ratio_interval
#print axioms Dobson.GapOneMiddle.tangent_interval
#print axioms Dobson.GapOneMiddle.weighted_enclosure
#print axioms Dobson.GapOneMiddle.convex_hull_bound
#print axioms Dobson.GapOneMiddle.G_mono_R
#print axioms Dobson.GapOneMiddle.G_antitone_V
#print axioms Dobson.GapOneMiddle.box_certificate
#print axioms Dobson.GapOneMiddle.box_derivative_neg
#print axioms Dobson.GapOneMiddle.mixture_gap_identity
#print axioms Dobson.GapOneMiddle.mixture_gap_positive
#print axioms Dobson.GapOneMiddle.persistence_constant_pos
#print axioms Dobson.GapOneMiddle.persistence_interval_nonempty
#print axioms Dobson.GapOneMiddle.geometric_denominator_bounds
#print axioms Dobson.GapOneMiddle.high_persistence_criterion
#print axioms Dobson.GapOneMiddle.exp_neg_lipschitz
#print axioms Dobson.GapOneMiddle.G_lipschitz
#print axioms Dobson.GapOneMiddle.high_persistence_from_errors
