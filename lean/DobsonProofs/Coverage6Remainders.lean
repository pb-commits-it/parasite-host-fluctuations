import DobsonProofs.Coverage1Lineage
import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Topology.Order.Compact
import Mathlib.Tactic

/-! # The analytic last step of the remainder theorems

Targets: `thm:ring-rem-ar1`, `thm:ring-rem-lin`, `lem:mech-bounded`,
and `lem:ring-rem-uniform`. The public finite-path results prove the second
variation and coefficient convergence. Their derivative and common-radius
bounds remain displayed premises; no Taylor remainder is assumed.

Coverage correction, 2026-11-22: the Gaussian and uniform ring rows remain
partial-P1. For bounded additive reference paths, `P1bB_MechBounded` proves
uniform derivative and remainder bounds, giving `lem:mech-bounded` core
coverage. General-matrix reduction and exponent identification remain outside
that credit. A finite-path bound does not by itself identify physical growth. -/

open Set Filter
open scoped Topology

namespace Dobson.FinitePathCoefficient
open MeasureTheory Filter Dobson.FiniteLineageVariance
open scoped BigOperators Topology
variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

/-- The finite quadratic coefficient from the actual within-environment path variance. -/
noncomputable def finiteCoefficient (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (N : ℕ) : ℝ :=
  (∑ e, K.initial e*lagCovariance μ T f e e 0)/2-
    withinVariance μ K T f (N+1)/(2*(N+1:ℕ))

/-- The manuscript's directed stationary covariance series. -/
noncomputable def covarianceCoefficient (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) : ℝ :=
  (1/2:ℝ)*(∑ e, ∑ e', K.initial e*K.initial e'*lagCovariance μ T f e e' 0)-
    ∑' m : ℕ, ∑ e, ∑ e', lagCovariance μ T f e e' (m+1)*K.initial e'*
      ((K.prob^(m+1)) e' e-K.initial e)

/-- Finite actual path coefficients converge by pair counting and the geometric Cesàro bound. -/
theorem finiteCoefficient_tendsto (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (B Ku β : ℝ) (hB : 0 ≤ B) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hcov : ∀ e e' m, |lagCovariance μ T f e e' m| ≤ B)
    (hmix : ∀ e e' m, |(K.prob^(m+1)) e' e-K.initial e| ≤ Ku*β^m) :
    Tendsto (finiteCoefficient μ K T f) atTop (𝓝 (covarianceCoefficient μ K T f)) := by
  have h := Dobson.Coverage1Lineage.lineage_variance_identity_and_remainder μ K T hT f hf
    B Ku β hB hKu hβ0 hβ1 hcov hmix 0 (fun _ => 0) 0 0 2 (by
      intro σ hσ
      have : σ=0 := by have hh := abs_le.mp hσ; linarith
      subst σ
      simp)
  dsimp only at h
  have hlim := h.2.1.comp (tendsto_add_atTop_nat 1)
  have hfinal := (tendsto_const_nhds (x := (1/2:ℝ)*(∑ e, K.initial e*lagCovariance μ T f e e 0))).sub
    (hlim.const_mul (1/2:ℝ))
  rw [h.2.2.1] at hfinal
  convert hfinal using 1
  · funext N
    dsimp [finiteCoefficient, Function.comp_def]
    ring
  · rfl
end Dobson.FinitePathCoefficient
#print axioms Dobson.FinitePathCoefficient.finiteCoefficient_tendsto

namespace Dobson.FinitePathJet
open MeasureTheory Filter Dobson.FiniteLineageVariance
open scoped BigOperators Topology
variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

lemma pathExpect_hasDerivAt (K : Chain E) (n : ℕ) (i : E)
    (F D : ℝ → (ℕ → E) → ℝ) (x : ℝ)
    (h : ∀ p, HasDerivAt (fun y => F y p) (D x p) x) :
    HasDerivAt (fun y => pathExpect K n i (F y)) (pathExpect K n i (D x)) x := by
  induction n generalizing i F D with
  | zero => exact h _
  | succ n ih =>
    exact HasDerivAt.fun_sum fun j _ =>
      (ih j (fun y p => F y (prepend i p)) (fun y p => D y (prepend i p))
        (fun p => h _)).const_mul (K.prob i j)

lemma expect_hasDerivAt (K : Chain E) (n : ℕ)
    (F D : ℝ → (ℕ → E) → ℝ) (x : ℝ)
    (h : ∀ p, HasDerivAt (fun y => F y p) (D x p) x) :
    HasDerivAt (fun y => expect K n (F y)) (expect K n (D x)) x :=
  HasDerivAt.fun_sum fun i _ => (pathExpect_hasDerivAt K n i F D x h).const_mul _

lemma pathExpect_meas (K : Chain E) (n : ℕ) (i : E)
    (F : Ω → (ℕ → E) → ℝ) (μ : Measure Ω)
    (hF : ∀ p, AEStronglyMeasurable (fun ω => F ω p) μ) :
    AEStronglyMeasurable (fun ω => pathExpect K n i (F ω)) μ := by
  induction n generalizing i F with
  | zero => exact hF _
  | succ n ih =>
    convert Finset.aestronglyMeasurable_sum Finset.univ (fun j _ =>
      (ih j (fun ω p => F ω (prepend i p)) (fun p => hF _)).const_mul (K.prob i j)) using 1
    funext ω
    simp only [pathExpect, Finset.sum_apply]

lemma expect_meas (K : Chain E) (n : ℕ)
    (F : Ω → (ℕ → E) → ℝ) (μ : Measure Ω)
    (hF : ∀ p, AEStronglyMeasurable (fun ω => F ω p) μ) :
    AEStronglyMeasurable (fun ω => expect K n (F ω)) μ :=
by
  convert Finset.aestronglyMeasurable_sum Finset.univ
    (fun i (_ : i ∈ Finset.univ) => (pathExpect_meas K n i F μ hF).const_mul (K.initial i)) using 1
  funext ω
  simp only [expect, Finset.sum_apply]


lemma expect_mono (K : Chain E) (n : ℕ) (F G : (ℕ → E) → ℝ)
    (h : ∀ p, F p ≤ G p) : expect K n F ≤ expect K n G := by
  have hp := expect_nonneg K n (fun p => G p+(-1)*F p) (fun p => by linarith [h p])
  rw [expect_add, expect_const_mul] at hp
  linarith

lemma expect_pos (K : Chain E) (n : ℕ) (F : (ℕ → E) → ℝ)
    (h : ∀ p, 0 < F p) : 0 < expect K n F := by
  have hpath : ∀ n i (F : (ℕ → E) → ℝ), (∀ p, 0 < F p) → 0 < pathExpect K n i F := by
    intro n
    induction n with
    | zero => exact fun i F hF => hF _
    | succ n ih =>
      intro i F hF
      have hex : ∃ j, 0 < K.prob i j := by
        by_contra hc
        push Not at hc
        have := Finset.sum_nonpos (s := Finset.univ) (fun j _ => hc j)
        rw [K.total i] at this
        norm_num at this
      obtain ⟨j,hj⟩ := hex
      exact Finset.sum_pos' (fun j _ => mul_nonneg (K.nonneg i j)
        (ih j _ (fun p => hF _)).le)
        ⟨j,Finset.mem_univ _,mul_pos hj (ih j _ (fun p => hF _))⟩
  have hex : ∃ i, 0 < K.initial i := by
    by_contra hc
    push Not at hc
    have := Finset.sum_nonpos (s := Finset.univ) (fun i _ => hc i)
    rw [K.initial_total] at this
    norm_num at this
  obtain ⟨i,hi⟩ := hex
  exact Finset.sum_pos' (fun j _ => mul_nonneg (K.initial_nonneg j) (hpath n j F h).le)
    ⟨i,Finset.mem_univ _,mul_pos hi (hpath n i F h)⟩

lemma expect_abs_le (K : Chain E) (n : ℕ) (F G : (ℕ → E) → ℝ)
    (h : ∀ p, |F p| ≤ G p) : |expect K n F| ≤ expect K n G := by
  apply abs_le.mpr
  constructor
  · have hh := expect_mono K n (fun p => (-1)*G p) F (fun p => by
      have := (abs_le.mp (h p)).1; linarith)
    rw [expect_const_mul] at hh
    linarith
  · exact expect_mono K n F G (fun p => (abs_le.mp (h p)).2)

noncomputable def partition (K : Chain E) (n : ℕ)
    (S : ℝ → (ℕ → E) → ℝ) (x : ℝ) : ℝ := expect K n (fun p => Real.exp (S x p))
noncomputable def tilted (K : Chain E) (n : ℕ)
    (S : ℝ → (ℕ → E) → ℝ) (x : ℝ) (F : (ℕ → E) → ℝ) : ℝ :=
  expect K n (fun p => Real.exp (S x p)*F p)/partition K n S x
noncomputable def logPartition (K : Chain E) (n : ℕ)
    (S : ℝ → (ℕ → E) → ℝ) (x : ℝ) : ℝ := Real.log (partition K n S x)
noncomputable def logFirst (K : Chain E) (n : ℕ)
    (S D : ℝ → (ℕ → E) → ℝ) (x : ℝ) : ℝ := tilted K n S x (D x)
noncomputable def logSecond (K : Chain E) (n : ℕ)
    (S D DD : ℝ → (ℕ → E) → ℝ) (x : ℝ) : ℝ :=
  tilted K n S x (fun p => D x p^2+DD x p)-(logFirst K n S D x)^2

lemma partition_pos (K : Chain E) (n : ℕ) (S : ℝ → (ℕ → E) → ℝ) (x : ℝ) :
    0 < partition K n S x := expect_pos K n _ (fun p => Real.exp_pos _)

lemma tilted_abs_le (K : Chain E) (n : ℕ) (S : ℝ → (ℕ → E) → ℝ)
    (x : ℝ) (F : (ℕ → E) → ℝ) (B : ℝ) (hB : ∀ p, |F p| ≤ B) :
    |tilted K n S x F| ≤ B := by
  rw [tilted, abs_div, abs_of_pos (partition_pos K n S x), div_le_iff₀ (partition_pos K n S x)]
  have h := expect_abs_le K n (fun p => Real.exp (S x p)*F p)
    (fun p => B*Real.exp (S x p)) (fun p => by
      rw [abs_mul, abs_of_pos (Real.exp_pos _)]; nlinarith [Real.exp_pos (S x p), hB p])
  rwa [expect_const_mul] at h

lemma logPartition_hasDerivAt (K : Chain E) (n : ℕ)
    (S D : ℝ → (ℕ → E) → ℝ) (x : ℝ)
    (hS : ∀ p, HasDerivAt (fun y => S y p) (D x p) x) :
    HasDerivAt (logPartition K n S) (logFirst K n S D x) x := by
  have hz := expect_hasDerivAt K n (fun y p => Real.exp (S y p))
    (fun y p => Real.exp (S y p)*D y p) x (fun p => (hS p).exp)
  exact hz.log (partition_pos K n S x).ne'

lemma logFirst_hasDerivAt (K : Chain E) (n : ℕ)
    (S D DD : ℝ → (ℕ → E) → ℝ) (x : ℝ)
    (hS : ∀ p, HasDerivAt (fun y => S y p) (D x p) x)
    (hD : ∀ p, HasDerivAt (fun y => D y p) (DD x p) x) :
    HasDerivAt (logFirst K n S D) (logSecond K n S D DD x) x := by
  have hz := expect_hasDerivAt K n (fun y p => Real.exp (S y p))
    (fun y p => Real.exp (S y p)*D y p) x (fun p => (hS p).exp)
  have hd := expect_hasDerivAt K n (fun y p => Real.exp (S y p)*D y p)
    (fun y p => Real.exp (S y p)*(D y p^2+DD y p)) x (fun p => by
      convert ((hS p).exp.mul (hD p)) using 1 <;> ring)
  have heq : logSecond K n S D DD x =
      (expect K n (fun p => Real.exp (S x p)*(D x p^2+DD x p))*partition K n S x-
        expect K n (fun p => Real.exp (S x p)*D x p)*
        expect K n (fun p => Real.exp (S x p)*D x p))/(partition K n S x)^2 := by
    unfold logSecond logFirst tilted
    field_simp [(partition_pos K n S x).ne']
  rw [heq]
  exact hd.div hz (partition_pos K n S x).ne'


lemma logSecond_abs_le (K : Chain E) (n : ℕ)
    (S D DD : ℝ → (ℕ → E) → ℝ) (x A B : ℝ)
    (hD : ∀ p, |D x p| ≤ A) (hDD : ∀ p, |DD x p| ≤ B) :
    |logSecond K n S D DD x| ≤ 2*A^2+B := by
  have hfirst := tilted_abs_le K n S x (D x) A hD
  have hcomb := tilted_abs_le K n S x (fun p => D x p^2+DD x p) (A^2+B) (fun p => by
    calc
      |D x p^2+DD x p| ≤ |D x p^2|+|DD x p| := abs_add_le _ _
      _ ≤ A^2+B := by
        rw [abs_of_nonneg (sq_nonneg _)]
        have := sq_le_sq₀ (abs_nonneg (D x p)) ((abs_nonneg _).trans (hD p)) |>.2 (hD p)
        nlinarith [sq_abs (D x p),hDD p])
  calc
    |logSecond K n S D DD x| ≤
        |tilted K n S x (fun p => D x p^2+DD x p)|+|(logFirst K n S D x)^2| := by simpa only [logSecond, sub_zero, zero_sub, abs_neg] using abs_sub_le (tilted K n S x (fun p => D x p^2+DD x p)) 0 ((logFirst K n S D x)^2)
    _ ≤ 2*A^2+B := by
      rw [abs_of_nonneg (sq_nonneg (logFirst K n S D x))]
      have := sq_le_sq₀ (abs_nonneg _) ((abs_nonneg _).trans hfirst) |>.2 hfirst
      dsimp [logFirst]
      nlinarith [sq_abs (tilted K n S x (D x))]

lemma logPartition_meas (K : Chain E) (n : ℕ) (μ : Measure Ω)
    (S : ℝ → Ω → (ℕ → E) → ℝ) (x : ℝ)
    (hS : ∀ p, AEStronglyMeasurable (fun ω => S x ω p) μ) :
    AEStronglyMeasurable (fun ω => logPartition K n (fun y => S y ω) x) μ :=
  ((expect_meas K n _ μ (fun p =>
    (Real.measurable_exp.comp_aemeasurable (hS p).aemeasurable).aestronglyMeasurable)).aemeasurable.log).aestronglyMeasurable

lemma logFirst_meas (K : Chain E) (n : ℕ) (μ : Measure Ω)
    (S D : ℝ → Ω → (ℕ → E) → ℝ) (x : ℝ)
    (hS : ∀ p, AEStronglyMeasurable (fun ω => S x ω p) μ)
    (hD : ∀ p, AEStronglyMeasurable (fun ω => D x ω p) μ) :
    AEStronglyMeasurable (fun ω => logFirst K n (fun y => S y ω) (fun y => D y ω) x) μ := by
  have he (p : ℕ → E) :=
    (Real.measurable_exp.comp_aemeasurable (hS p).aemeasurable).aestronglyMeasurable
  exact ((expect_meas K n _ μ (fun p => (he p).mul (hD p))).aemeasurable.div
    (expect_meas K n _ μ he).aemeasurable).aestronglyMeasurable

lemma logSecond_meas (K : Chain E) (n : ℕ) (μ : Measure Ω)
    (S D DD : ℝ → Ω → (ℕ → E) → ℝ) (x : ℝ)
    (hS : ∀ p, AEStronglyMeasurable (fun ω => S x ω p) μ)
    (hD : ∀ p, AEStronglyMeasurable (fun ω => D x ω p) μ)
    (hDD : ∀ p, AEStronglyMeasurable (fun ω => DD x ω p) μ) :
    AEStronglyMeasurable (fun ω => logSecond K n (fun y => S y ω)
      (fun y => D y ω) (fun y => DD y ω) x) μ := by
  have he (p : ℕ → E) :=
    (Real.measurable_exp.comp_aemeasurable (hS p).aemeasurable).aestronglyMeasurable
  exact (((expect_meas K n _ μ (fun p => (he p).mul ((hD p).pow 2 |>.add (hDD p)))).aemeasurable.div
    (expect_meas K n _ μ he).aemeasurable).aestronglyMeasurable).sub ((logFirst_meas K n μ S D x hS hD).pow 2)

/-- The two integral exchanges use square-integrable envelopes, not exponential moments. -/
lemma expectedLog_second (K : Chain E) (n : ℕ) (μ : Measure Ω)
    (S D DD : ℝ → Ω → (ℕ → E) → ℝ) (δ : ℝ) (hδ : 0 < δ)
    (A B : Ω → ℝ) (hA : Integrable A μ) (hAA : Integrable (fun ω => A ω^2) μ)
    (hB : Integrable B μ)
    (hSmeas : ∀ x p, AEStronglyMeasurable (fun ω => S x ω p) μ)
    (hDmeas : ∀ x p, AEStronglyMeasurable (fun ω => D x ω p) μ)
    (hDDmeas : ∀ x p, AEStronglyMeasurable (fun ω => DD x ω p) μ)
    (hPint : ∀ x ∈ Set.Ioo (-δ) δ, Integrable
      (fun ω => logPartition K n (fun y => S y ω) x) μ)
    (hS : ∀ ω, ∀ x ∈ Set.Ioo (-δ) δ, ∀ p, HasDerivAt (fun y => S y ω p) (D x ω p) x)
    (hD : ∀ ω, ∀ x ∈ Set.Ioo (-δ) δ, ∀ p, HasDerivAt (fun y => D y ω p) (DD x ω p) x)
    (hAbound : ∀ ω, ∀ x ∈ Set.Ioo (-δ) δ, ∀ p, |D x ω p| ≤ A ω)
    (hBbound : ∀ ω, ∀ x ∈ Set.Ioo (-δ) δ, ∀ p, |DD x ω p| ≤ B ω) :
    iteratedDeriv 2 (fun x => ∫ ω, logPartition K n (fun y => S y ω) x ∂μ) 0 =
      ∫ ω, logSecond K n (fun y => S y ω) (fun y => D y ω) (fun y => DD y ω) 0 ∂μ := by
  have hzero : (0 : ℝ) ∈ Set.Ioo (-δ) δ := ⟨by linarith,hδ⟩
  have hfirst (x : ℝ) (hx : x ∈ Set.Ioo (-δ) δ) :=
    hasDerivAt_integral_of_dominated_loc_of_deriv_le
      (μ := μ) (bound := A) (isOpen_Ioo.mem_nhds hx)
      (Eventually.of_forall (fun y => logPartition_meas K n μ S y (hSmeas y)))
      (hPint x hx) (logFirst_meas K n μ S D x (hSmeas x) (hDmeas x))
      (Eventually.of_forall (fun ω y hy => by
        simpa only [Real.norm_eq_abs, logFirst] using
          tilted_abs_le K n (fun z => S z ω) y (D y ω) (A ω) (hAbound ω y hy)))
      hA (Eventually.of_forall (fun ω y hy =>
        logPartition_hasDerivAt K n (fun z => S z ω) (fun z => D z ω) y (hS ω y hy)))
  have hsecond := hasDerivAt_integral_of_dominated_loc_of_deriv_le
      (μ := μ) (bound := fun ω => 2*A ω^2+B ω) (isOpen_Ioo.mem_nhds hzero)
      (Eventually.of_forall (fun y => logFirst_meas K n μ S D y (hSmeas y) (hDmeas y)))
      (hfirst 0 hzero).1 (logSecond_meas K n μ S D DD 0 (hSmeas 0) (hDmeas 0) (hDDmeas 0))
      (Eventually.of_forall (fun ω y hy => by
        simpa only [Real.norm_eq_abs] using logSecond_abs_le K n
          (fun z => S z ω) (fun z => D z ω) (fun z => DD z ω) y (A ω) (B ω)
          (hAbound ω y hy) (hBbound ω y hy)))
      ((hAA.const_mul 2).add hB)
      (Eventually.of_forall (fun ω y hy => logFirst_hasDerivAt K n
        (fun z => S z ω) (fun z => D z ω) (fun z => DD z ω) y (hS ω y hy) (hD ω y hy)))
  have heq : deriv (fun x => ∫ ω, logPartition K n (fun y => S y ω) x ∂μ) =ᶠ[𝓝 0]
      (fun x => ∫ ω, logFirst K n (fun y => S y ω) (fun y => D y ω) x ∂μ) := by
    filter_upwards [isOpen_Ioo.mem_nhds hzero] with x hx
    exact (hfirst x hx).2.deriv
  simpa only [iteratedDeriv_succ, iteratedDeriv_zero] using
    (hsecond.2.congr_of_eventuallyEq heq).deriv

lemma logPartition_abs_le (K : Chain E) (n : ℕ) (S : ℝ → (ℕ → E) → ℝ)
    (x C : ℝ) (hS : ∀ p, |S x p| ≤ C) : |logPartition K n S x| ≤ C := by
  have hlo := expect_mono K n (fun _ => Real.exp (-C)) (fun p => Real.exp (S x p))
    (fun p => Real.exp_le_exp.mpr (abs_le.mp (hS p)).1)
  have hhi := expect_mono K n (fun p => Real.exp (S x p)) (fun _ => Real.exp C)
    (fun p => Real.exp_le_exp.mpr (abs_le.mp (hS p)).2)
  rw [expect_const] at hlo hhi
  apply abs_le.mpr
  constructor
  · simpa only [Real.log_exp, logPartition, partition] using Real.log_le_log (Real.exp_pos (-C)) hlo
  · simpa only [Real.log_exp, logPartition, partition] using Real.log_le_log (partition_pos K n S x) hhi

end Dobson.FinitePathJet

namespace Dobson.FinitePathJet
open MeasureTheory Filter Dobson.FiniteLineageVariance
open scoped BigOperators Topology
variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

noncomputable def quadraticScore (Y : Ω → (ℕ → E) → ℝ) (V : (ℕ → E) → ℝ)
    (x : ℝ) (ω : Ω) (p : ℕ → E) : ℝ := x*Y ω p-x^2*V p/2
noncomputable def quadraticFirst (Y : Ω → (ℕ → E) → ℝ) (V : (ℕ → E) → ℝ)
    (x : ℝ) (ω : Ω) (p : ℕ → E) : ℝ := Y ω p-x*V p
noncomputable def quadraticSecond (V : (ℕ → E) → ℝ)
    (_x : ℝ) (_ω : Ω) (p : ℕ → E) : ℝ := -V p

lemma quadraticScore_deriv (Y : Ω → (ℕ → E) → ℝ) (V : (ℕ → E) → ℝ)
    (ω : Ω) (p : ℕ → E) (x : ℝ) :
    HasDerivAt (fun y => quadraticScore Y V y ω p) (quadraticFirst Y V x ω p) x := by
  convert ((hasDerivAt_id x).mul_const (Y ω p)).sub
    ((((hasDerivAt_id x).pow 2).mul_const (V p)).div_const 2) using 1
  · funext y
    simp only [quadraticScore, Pi.sub_apply, Pi.pow_apply, id_eq]
  · simp only [quadraticFirst, id_eq] <;> ring

lemma quadraticFirst_deriv (Y : Ω → (ℕ → E) → ℝ) (V : (ℕ → E) → ℝ)
    (ω : Ω) (p : ℕ → E) (x : ℝ) :
    HasDerivAt (fun y => quadraticFirst Y V y ω p) (quadraticSecond V x ω p) x := by
  simpa only [quadraticFirst, quadraticSecond, Pi.sub_def, Pi.sub_apply, id_eq, one_mul, zero_sub] using
    (hasDerivAt_const x (Y ω p)).sub ((hasDerivAt_id x).mul_const (V p))

lemma quadratic_expected_second (K : Chain E) (n : ℕ) (μ : Measure Ω) [IsProbabilityMeasure μ]
    (Y : Ω → (ℕ → E) → ℝ) (V : (ℕ → E) → ℝ) (A : Ω → ℝ) (C : ℝ)
    (hA : MemLp A 2 μ) (hYmeas : ∀ p, AEStronglyMeasurable (fun ω => Y ω p) μ)
    (hY : ∀ ω p, |Y ω p| ≤ A ω) (hC : 0 ≤ C) (hV : ∀ p, |V p| ≤ C) :
    iteratedDeriv 2 (fun x => ∫ ω, logPartition K n (fun y => quadraticScore Y V y ω) x ∂μ) 0 =
      ∫ ω, expect K n (fun p => Y ω p^2-V p)-(expect K n (Y ω))^2 ∂μ := by
  have hsm (x : ℝ) (p : ℕ → E) :
      AEStronglyMeasurable (fun ω => quadraticScore Y V x ω p) μ := by
    exact ((hYmeas p).const_mul x).sub aestronglyMeasurable_const
  have hdm (x : ℝ) (p : ℕ → E) :
      AEStronglyMeasurable (fun ω => quadraticFirst Y V x ω p) μ := by
    exact (hYmeas p).sub aestronglyMeasurable_const
  have hddm (x : ℝ) (p : ℕ → E) :
      AEStronglyMeasurable (fun ω => quadraticSecond V x ω p) μ := aestronglyMeasurable_const
  have henv : MemLp (fun ω => A ω+C) 2 μ := hA.add (memLp_const C)
  have hpint (x : ℝ) (_hx : x ∈ Set.Ioo (-1) 1) : Integrable
      (fun ω => logPartition K n (fun y => quadraticScore Y V y ω) x) μ := by
    apply Integrable.mono' (((hA.integrable (by norm_num)).const_mul |x|).add
      (integrable_const (x^2*C/2))) (logPartition_meas K n μ (quadraticScore Y V) x (hsm x))
    exact Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs]
      apply logPartition_abs_le
      intro p
      calc
        |quadraticScore Y V x ω p| ≤ |x*Y ω p|+|x^2*V p/2| := by
          simpa only [quadraticScore, sub_zero, zero_sub, abs_neg] using
            abs_sub_le (x*Y ω p) 0 (x^2*V p/2)
        _ ≤ |x| *A ω+x^2*C/2 := by
          rw [abs_mul, abs_div, abs_mul, abs_of_nonneg (sq_nonneg x)]
          norm_num
          have hy := mul_le_mul_of_nonneg_left (hY ω p) (abs_nonneg x)
          have hv := mul_le_mul_of_nonneg_left (hV p) (sq_nonneg x)
          linarith
  have hAb (ω : Ω) (x : ℝ) (hx : x ∈ Set.Ioo (-1) 1) (p : ℕ → E) :
      |quadraticFirst Y V x ω p| ≤ A ω+C := by
    have hx' : |x| ≤ 1 := abs_le.mpr ⟨hx.1.le,hx.2.le⟩
    have h := abs_sub_le (Y ω p) 0 (x*V p)
    simp only [sub_zero, zero_sub, abs_neg, abs_mul] at h
    have hv := mul_le_mul (hV p) hx' (abs_nonneg x) hC
    dsimp [quadraticFirst]
    nlinarith [hY ω p]
  have h := expectedLog_second K n μ (quadraticScore Y V) (quadraticFirst Y V)
    (quadraticSecond V) 1 (by norm_num) (fun ω => A ω+C) (fun _ => C)
    (henv.integrable (by norm_num)) henv.integrable_sq (integrable_const C)
    hsm hdm hddm hpint
    (fun ω x _ p => quadraticScore_deriv Y V ω p x)
    (fun ω x _ p => quadraticFirst_deriv Y V ω p x) hAb
    (fun ω x _ p => by simpa [quadraticSecond] using hV p)
  convert h using 1
  congr 1
  funext ω
  simp only [logSecond, logFirst, tilted, partition, quadraticScore, quadraticFirst,
    quadraticSecond, zero_mul, zero_pow, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
    zero_div, sub_self, Real.exp_zero, one_mul, sub_zero, expect_const, div_one, sub_eq_add_neg, neg_zero, add_zero]

end Dobson.FinitePathJet

namespace Dobson.FinitePathJet
open MeasureTheory Filter Dobson.FiniteLineageVariance
open scoped BigOperators Topology
variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

lemma pathExpect_memLp (K : Chain E) (n : ℕ) (i : E)
    (F : Ω → (ℕ → E) → ℝ) (μ : Measure Ω) (q : ENNReal)
    (hF : ∀ p, MemLp (fun ω => F ω p) q μ) :
    MemLp (fun ω => pathExpect K n i (F ω)) q μ := by
  induction n generalizing i F with
  | zero => exact hF _
  | succ n ih =>
    exact memLp_finsetSum _ fun j _ =>
      (ih j (fun ω p => F ω (prepend i p)) (fun p => hF _)).const_mul (K.prob i j)

lemma expect_memLp (K : Chain E) (n : ℕ)
    (F : Ω → (ℕ → E) → ℝ) (μ : Measure Ω) (q : ENNReal)
    (hF : ∀ p, MemLp (fun ω => F ω p) q μ) :
    MemLp (fun ω => expect K n (F ω)) q μ :=
  memLp_finsetSum _ fun i _ => (pathExpect_memLp K n i F μ q hF).const_mul (K.initial i)

noncomputable def rewardEnvelope (T : Ω → Ω) (f : E → Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ∑ t ∈ Finset.range n, ∑ e, |f e (T^[t] ω)|

lemma rewardEnvelope_memLp (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ) (n : ℕ) :
    MemLp (rewardEnvelope T f n) 2 μ := by
  exact memLp_finsetSum _ fun t _ => memLp_finsetSum _ fun e _ => by
    simpa only [Real.norm_eq_abs, Function.comp_apply] using ((hf e).comp_measurePreserving (hT.iterate t)).norm

lemma reward_memLp (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ) (n : ℕ) (p : ℕ → E) :
    MemLp (fun ω => reward n (timeReward T f ω) p) 2 μ :=
  memLp_finsetSum _ fun t _ => (hf (p t)).comp_measurePreserving (hT.iterate t)

lemma reward_abs_le (T : Ω → Ω) (f : E → Ω → ℝ) (n : ℕ) (ω : Ω) (p : ℕ → E) :
    |reward n (timeReward T f ω) p| ≤ rewardEnvelope T f n ω := by
  calc
    _ ≤ ∑ t ∈ Finset.range n, |f (p t) (T^[t] ω)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ _ := Finset.sum_le_sum fun t _ =>
      Finset.single_le_sum (fun e _ => abs_nonneg (f e (T^[t] ω))) (Finset.mem_univ (p t))

lemma constant_reward_abs_le (v : E → ℝ) (n : ℕ) (p : ℕ → E) :
    |reward n (fun _ e => v e) p| ≤ (n:ℝ)*(∑ e, |v e|) := by
  calc
    _ ≤ ∑ t ∈ Finset.range n, |v (p t)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _t ∈ Finset.range n, ∑ e, |v e| := Finset.sum_le_sum fun t _ =>
      Finset.single_le_sum (fun e _ => abs_nonneg (v e)) (Finset.mem_univ (p t))
    _ = _ := by simp

lemma expect_constant_reward (K : Chain E) (v : E → ℝ) (n : ℕ) :
    expect K n (reward n (fun _ e => v e)) = (n:ℝ)*(∑ e, K.initial e*v e) := by
  change expect K n (fun p => ∑ t ∈ Finset.range n, v (p t))=_
  rw [expect_sum]
  have he : ∀ t ∈ Finset.range n, expect K n (fun p => v (p t))=∑ e, K.initial e*v e := by
    intro t ht
    exact expect_coordinate K n t (Nat.le_of_lt (Finset.mem_range.mp ht)) v
  simp only [Finset.sum_congr rfl he, Finset.sum_const, Finset.card_range, nsmul_eq_mul]

lemma pathVariance_integrable (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ) (n : ℕ) :
    Integrable (fun ω => pathVariance K n (timeReward T f ω)) μ := by
  have hA := rewardEnvelope_memLp μ T hT f hf n
  have hY (p : ℕ → E) := reward_memLp μ T hT f hf n p
  have hsq : Integrable (fun ω => expect K n (fun p => (reward n (timeReward T f ω) p)^2)) μ := by
    apply Integrable.mono' hA.integrable_sq
      (expect_meas K n _ μ (fun p => (hY p).aestronglyMeasurable.pow 2))
    exact Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs]
      have := expect_abs_le K n (fun p => (reward n (timeReward T f ω) p)^2)
        (fun _ => rewardEnvelope T f n ω^2) (fun p => by
          rw [abs_of_nonneg (sq_nonneg _)]
          have hh := sq_le_sq₀ (abs_nonneg _) ((abs_nonneg _).trans (reward_abs_le T f n ω p)) |>.2
            (reward_abs_le T f n ω p)
          nlinarith [sq_abs (reward n (timeReward T f ω) p)])
      simpa only [expect_const, Pi.pow_apply] using this
  exact hsq.sub ((expect_memLp K n _ μ 2 hY).integrable_sq)

/-- Expected finite growth of the actual multiplicative lognormal reference paths. -/
noncomputable def lognormalGrowth (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (v : E → ℝ) (L : ℝ) (N : ℕ) (x : ℝ) : ℝ :=
  L+(∫ ω, logPartition K (N+1) (fun y => quadraticScore
    (fun ω p => reward (N+1) (timeReward T f ω) p)
    (reward (N+1) (fun _ e => v e)) y ω) x ∂μ)/(N+1:ℕ)

noncomputable def finiteCoefficient (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (v : E → ℝ) (N : ℕ) : ℝ :=
  (∑ e, K.initial e*v e)/2-withinVariance μ K T f (N+1)/(2*(N+1:ℕ))

lemma lognormalGrowth_second (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ) (v : E → ℝ) (L : ℝ) (N : ℕ) :
    iteratedDeriv 2 (lognormalGrowth μ K T f v L N) 0 = -2*finiteCoefficient μ K T f v N := by
  have h := quadratic_expected_second K (N+1) μ
    (fun ω p => reward (N+1) (timeReward T f ω) p)
    (reward (N+1) (fun _ e => v e)) (rewardEnvelope T f (N+1))
    ((N+1:ℕ)*(∑ e, |v e|)) (rewardEnvelope_memLp μ T hT f hf (N+1))
    (fun p => (reward_memLp μ T hT f hf (N+1) p).aestronglyMeasurable)
    (reward_abs_le T f (N+1)) (by positivity) (constant_reward_abs_le v (N+1))
  have he (ω : Ω) :
      expect K (N+1) (fun p => reward (N+1) (timeReward T f ω) p^2-
        reward (N+1) (fun _ e => v e) p)-
      (expect K (N+1) (reward (N+1) (timeReward T f ω)))^2 =
      pathVariance K (N+1) (timeReward T f ω)-(N+1:ℕ)*(∑ e, K.initial e*v e) := by
    simp only [sub_eq_add_neg, ← neg_one_mul (reward (N+1) (fun _ e => v e) _),
      expect_add, expect_const_mul, expect_constant_reward, pathVariance]
    ring
  simp_rw [he] at h
  rw [integral_sub (pathVariance_integrable μ K T hT f hf (N+1)) (integrable_const _),
    integral_const] at h
  simp only [probReal_univ, smul_eq_mul, one_mul] at h
  change iteratedDeriv 2 (fun x => L+_/(N+1:ℕ)) 0 = _
  rw [iteratedDeriv_const_add (by norm_num : (0:ℕ) < 2), iteratedDeriv_div_const]
  rw [h]
  unfold finiteCoefficient withinVariance
  field_simp
  <;> ring

end Dobson.FinitePathJet

namespace Dobson.FinitePathJet
open MeasureTheory Filter Dobson.FiniteLineageVariance
open scoped BigOperators Topology
variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

noncomputable def additiveScore (T : Ω → Ω) (f : E → Ω → ℝ) (n : ℕ)
    (x : ℝ) (ω : Ω) (p : ℕ → E) : ℝ :=
  ∑ t ∈ Finset.range n, Real.log (1+x*timeReward T f ω t (p t))
noncomputable def additiveFirst (T : Ω → Ω) (f : E → Ω → ℝ) (n : ℕ)
    (x : ℝ) (ω : Ω) (p : ℕ → E) : ℝ :=
  ∑ t ∈ Finset.range n, timeReward T f ω t (p t)/(1+x*timeReward T f ω t (p t))
noncomputable def additiveSecond (T : Ω → Ω) (f : E → Ω → ℝ) (n : ℕ)
    (x : ℝ) (ω : Ω) (p : ℕ → E) : ℝ :=
  ∑ t ∈ Finset.range n, -(timeReward T f ω t (p t))^2/(1+x*timeReward T f ω t (p t))^2

lemma additive_factor_bounds (a x B : ℝ) (hB : 0 < B) (ha : |a| ≤ B)
    (hx : x ∈ Set.Ioo (-(1/(2*B))) (1/(2*B))) :
    1/2 < 1+x*a ∧ 1+x*a < 2 := by
  have hx' : |x| < 1/(2*B) := abs_lt.mpr hx
  have hxb := (lt_div_iff₀ (by positivity : 0 < 2*B)).mp hx'
  have hmul : |x*a| ≤ |x| * B := by rw [abs_mul]; gcongr
  have hl := neg_abs_le (x*a)
  have hu := le_abs_self (x*a)
  constructor <;> nlinarith

lemma additive_expected_second (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (B : ℝ) (hB : 0 < B) (hbound : ∀ e ω, |f e ω| ≤ B) (n : ℕ) :
    iteratedDeriv 2 (fun x => ∫ ω, logPartition K n (fun y => additiveScore T f n y ω) x ∂μ) 0 =
      ∫ ω, pathVariance K n (timeReward T f ω)-
        expect K n (reward n (fun t e => (timeReward T f ω t e)^2)) ∂μ := by
  have hm (t : ℕ) (e : E) : AEStronglyMeasurable (fun ω => timeReward T f ω t e) μ :=
    ((hf e).comp_measurePreserving (hT.iterate t)).aestronglyMeasurable
  have hfac (x : ℝ) (t : ℕ) (e : E) : AEStronglyMeasurable
      (fun ω => 1+x*timeReward T f ω t e) μ := aestronglyMeasurable_const.add ((hm t e).const_mul x)
  have hsm (x : ℝ) (p : ℕ → E) : AEStronglyMeasurable (fun ω => additiveScore T f n x ω p) μ := by
    convert Finset.aestronglyMeasurable_sum (Finset.range n) (fun t _ =>
      ((hfac x t (p t)).aemeasurable.log).aestronglyMeasurable) using 1
    funext ω
    simp only [additiveScore, Finset.sum_apply]
  have hdm (x : ℝ) (p : ℕ → E) : AEStronglyMeasurable (fun ω => additiveFirst T f n x ω p) μ := by
    convert Finset.aestronglyMeasurable_sum (Finset.range n) (fun t _ =>
      ((hm t (p t)).aemeasurable.div (hfac x t (p t)).aemeasurable).aestronglyMeasurable) using 1
    funext ω
    simp only [additiveFirst, Finset.sum_apply, Pi.div_apply]
  have hddm (x : ℝ) (p : ℕ → E) : AEStronglyMeasurable (fun ω => additiveSecond T f n x ω p) μ := by
    convert Finset.aestronglyMeasurable_sum (Finset.range n) (fun t _ =>
      ((((hm t (p t)).pow 2).neg.aemeasurable).div
        ((hfac x t (p t)).pow 2).aemeasurable).aestronglyMeasurable) using 1
    funext ω
    simp only [additiveSecond, Finset.sum_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]
  have hb (ω : Ω) (x : ℝ) (hx : x ∈ Set.Ioo (-(1/(2*B))) (1/(2*B))) (t : ℕ) (e : E) :=
    additive_factor_bounds (timeReward T f ω t e) x B hB (hbound e _) hx
  have hder (ω : Ω) (x : ℝ) (hx : x ∈ Set.Ioo (-(1/(2*B))) (1/(2*B))) (p : ℕ → E) :
      HasDerivAt (fun y => additiveScore T f n y ω p) (additiveFirst T f n x ω p) x := by
    apply HasDerivAt.fun_sum
    intro t _
    have hp : 0 < 1+x*timeReward T f ω t (p t) := by linarith [(hb ω x hx t (p t)).1]
    simpa only [Pi.add_def, id_eq, one_mul, zero_add] using
      ((hasDerivAt_const x 1).add ((hasDerivAt_id x).mul_const (timeReward T f ω t (p t)))).log hp.ne'
  have hdder (ω : Ω) (x : ℝ) (hx : x ∈ Set.Ioo (-(1/(2*B))) (1/(2*B))) (p : ℕ → E) :
      HasDerivAt (fun y => additiveFirst T f n y ω p) (additiveSecond T f n x ω p) x := by
    apply HasDerivAt.fun_sum
    intro t _
    have hp : 0 < 1+x*timeReward T f ω t (p t) := by linarith [(hb ω x hx t (p t)).1]
    convert (hasDerivAt_const x (timeReward T f ω t (p t))).div
      ((hasDerivAt_const x 1).add ((hasDerivAt_id x).mul_const (timeReward T f ω t (p t)))) hp.ne' using 1
    · rfl
    · simp only [id_eq, zero_add, one_mul, zero_mul, zero_sub, neg_mul, Pi.add_apply]
      ring
  have hab (ω : Ω) (x : ℝ) (hx : x ∈ Set.Ioo (-(1/(2*B))) (1/(2*B))) (p : ℕ → E) :
      |additiveFirst T f n x ω p| ≤ (n:ℝ)*(2*B) := by
    calc
      _ ≤ ∑ t ∈ Finset.range n, |timeReward T f ω t (p t)/(1+x*timeReward T f ω t (p t))| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _t ∈ Finset.range n, 2*B := Finset.sum_le_sum fun t _ => by
        have hh := hb ω x hx t (p t)
        have hp : 0 < 1+x*timeReward T f ω t (p t) := by linarith
        rw [abs_div, abs_of_pos hp, div_le_iff₀ hp]
        have hbp : |timeReward T f ω t (p t)| ≤ B := hbound (p t) (T^[t] ω)
        nlinarith
      _ = _ := by simp
  have habb (ω : Ω) (x : ℝ) (hx : x ∈ Set.Ioo (-(1/(2*B))) (1/(2*B))) (p : ℕ → E) :
      |additiveSecond T f n x ω p| ≤ (n:ℝ)*(4*B^2) := by
    calc
      _ ≤ ∑ t ∈ Finset.range n, |-(timeReward T f ω t (p t))^2/(1+x*timeReward T f ω t (p t))^2| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _t ∈ Finset.range n, 4*B^2 := Finset.sum_le_sum fun t _ => by
        have hh := hb ω x hx t (p t)
        have hp : 0 < 1+x*timeReward T f ω t (p t) := by linarith
        rw [abs_div, abs_neg, abs_of_nonneg (sq_nonneg _), abs_of_nonneg (sq_nonneg _), div_le_iff₀ (sq_pos_of_pos hp)]
        have hh2 : (1/4:ℝ) ≤ (1+x*timeReward T f ω t (p t))^2 := by nlinarith
        have hs := sq_le_sq₀ (abs_nonneg (timeReward T f ω t (p t))) hB.le |>.2 (hbound (p t) _)
        nlinarith [sq_abs (timeReward T f ω t (p t)), sq_nonneg B,
          mul_nonneg (sq_nonneg B) (sub_nonneg.mpr hh2)]
      _ = _ := by simp
  have hpint (x : ℝ) (hx : x ∈ Set.Ioo (-(1/(2*B))) (1/(2*B))) : Integrable
      (fun ω => logPartition K n (fun y => additiveScore T f n y ω) x) μ := by
    apply Integrable.mono' (integrable_const ((n:ℝ)*(|Real.log (1/2)|+|Real.log 2|)))
      (logPartition_meas K n μ (additiveScore T f n) x (hsm x))
    exact Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs]
      apply logPartition_abs_le
      intro p
      calc
        _ ≤ ∑ t ∈ Finset.range n, |Real.log (1+x*timeReward T f ω t (p t))| := Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _t ∈ Finset.range n, (|Real.log (1/2)|+|Real.log 2|) := Finset.sum_le_sum fun t _ => by
          have hh := hb ω x hx t (p t)
          have hp : 0 < 1+x*timeReward T f ω t (p t) := by linarith
          have hl := Real.log_le_log (by norm_num : (0:ℝ)<1/2) hh.1.le
          have hu := Real.log_le_log hp hh.2.le
          apply abs_le.mpr
          constructor <;> linarith [neg_abs_le (Real.log (1/2)), le_abs_self (Real.log 2),
            abs_nonneg (Real.log (1/2)), abs_nonneg (Real.log 2)]
        _ = _ := by simp <;> ring
  have h := expectedLog_second K n μ (additiveScore T f n) (additiveFirst T f n)
    (additiveSecond T f n) (1/(2*B)) (by positivity) (fun _ => (n:ℝ)*(2*B)) (fun _ => (n:ℝ)*(4*B^2))
    (integrable_const _) (integrable_const _) (integrable_const _)
    hsm hdm hddm hpint hder hdder hab habb
  convert h using 1
  congr 1
  funext ω
  simp only [logSecond, logFirst, tilted, partition, additiveScore, additiveFirst, additiveSecond,
    zero_mul, add_zero, Real.log_one, Finset.sum_const_zero, Real.exp_zero, one_mul, one_pow,
    div_one, expect_const]
  simp only [Finset.sum_neg_distrib, pathVariance]
  change expect K n (fun p => (reward n (timeReward T f ω) p)^2)-
      (expect K n (reward n (timeReward T f ω)))^2-
      expect K n (reward n (fun t e => (timeReward T f ω t e)^2)) =
    expect K n (fun p => (reward n (timeReward T f ω) p)^2+
      -reward n (fun t e => (timeReward T f ω t e)^2) p)-
      (expect K n (reward n (timeReward T f ω)))^2
  simp only [← neg_one_mul (reward n (fun t e => (timeReward T f ω t e)^2) _),
    expect_add, expect_const_mul]
  ring

end Dobson.FinitePathJet

namespace Dobson.FinitePathJet
open MeasureTheory Filter Dobson.FiniteLineageVariance
open scoped BigOperators Topology
variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

lemma expect_squared_reward (K : Chain E) (n : ℕ) (g : ℕ → E → ℝ) :
    expect K n (reward n (fun t e => (g t e)^2)) =
      ∑ t ∈ Finset.range n, ∑ e, K.initial e*(g t e)^2 := by
  unfold reward
  rw [expect_sum]
  exact Finset.sum_congr rfl fun t ht =>
    expect_coordinate K n t (Nat.le_of_lt (Finset.mem_range.mp ht)) (fun e => (g t e)^2)

lemma squared_cost_integrable (μ : Measure Ω) (K : Chain E)
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ) (n : ℕ) :
    Integrable (fun ω => expect K n (reward n (fun t e => (timeReward T f ω t e)^2))) μ := by
  simp_rw [expect_squared_reward]
  exact integrable_finsetSum _ fun t _ => integrable_finsetSum _ fun e _ =>
    (((hf e).comp_measurePreserving (hT.iterate t)).integrable_sq).const_mul (K.initial e)

lemma squared_cost_integral (μ : Measure Ω) (K : Chain E)
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ) (n : ℕ) :
    (∫ ω, expect K n (reward n (fun t e => (timeReward T f ω t e)^2)) ∂μ) =
      (n:ℝ)*(∑ e, K.initial e*lagCovariance μ T f e e 0) := by
  simp_rw [expect_squared_reward]
  have hint (t : ℕ) (e : E) : Integrable (fun ω => K.initial e*(timeReward T f ω t e)^2) μ :=
    (((hf e).comp_measurePreserving (hT.iterate t)).integrable_sq).const_mul (K.initial e)
  rw [integral_finsetSum _ (fun t _ => integrable_finsetSum _ fun e _ => hint t e)]
  have he (t : ℕ) :
      (∫ ω, ∑ e, K.initial e*(timeReward T f ω t e)^2 ∂μ)=∑ e, K.initial e*lagCovariance μ T f e e 0 := by
    rw [integral_finsetSum _ (fun e _ => hint t e)]
    simp_rw [integral_const_mul]
    apply Finset.sum_congr rfl
    intro e _
    congr 1
    simpa only [pow_two, Nat.add_zero] using integral_timeReward_pair μ T hT f e e t 0
  simp only [he, Finset.sum_const, Finset.card_range, nsmul_eq_mul]

noncomputable def additiveLogGrowth (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (L : ℝ) (N : ℕ) (x : ℝ) : ℝ :=
  L+(∫ ω, logPartition K (N+1) (fun y => additiveScore T f (N+1) y ω) x ∂μ)/(N+1:ℕ)

lemma additiveLogGrowth_second (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (B : ℝ) (hB : 0 < B) (hbound : ∀ e ω, |f e ω| ≤ B) (L : ℝ) (N : ℕ) :
    iteratedDeriv 2 (additiveLogGrowth μ K T f L N) 0 =
      -2*finiteCoefficient μ K T f (fun e => lagCovariance μ T f e e 0) N := by
  have h := additive_expected_second μ K T hT f hf B hB hbound (N+1)
  rw [integral_sub (pathVariance_integrable μ K T hT f hf (N+1))
    (squared_cost_integrable μ K T hT f hf (N+1)), squared_cost_integral μ K T hT f hf (N+1)] at h
  change iteratedDeriv 2 (fun x => L+_/(N+1:ℕ)) 0 = _
  rw [iteratedDeriv_const_add (by norm_num : (0:ℕ)<2), iteratedDeriv_div_const, h]
  unfold finiteCoefficient withinVariance
  field_simp
  <;> ring

/-- The additive finite partition uses the actual product of the affine edge factors. -/
noncomputable def additiveGrowth (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (L : ℝ) (N : ℕ) (x : ℝ) : ℝ :=
  L+(∫ ω, Real.log (expect K (N+1) (fun p =>
    ∏ t ∈ Finset.range (N+1), (1+x*timeReward T f ω t (p t)))) ∂μ)/(N+1:ℕ)

lemma additiveGrowth_eventuallyEq (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (B : ℝ) (hB : 0 < B) (hbound : ∀ e ω, |f e ω| ≤ B)
    (L : ℝ) (N : ℕ) :
    additiveGrowth μ K T f L N =ᶠ[𝓝 0] additiveLogGrowth μ K T f L N := by
  have hδ : (0:ℝ)<1/(2*B) := by positivity
  filter_upwards [isOpen_Ioo.mem_nhds (show (0:ℝ) ∈ Set.Ioo (-(1/(2*B))) (1/(2*B)) from ⟨by linarith,hδ⟩)] with x hx
  unfold additiveGrowth additiveLogGrowth
  congr 2
  apply integral_congr_ae
  exact Eventually.of_forall fun ω => by
    unfold logPartition partition
    apply congrArg Real.log
    apply congrArg (expect K (N+1))
    funext p
    unfold additiveScore
    rw [Real.exp_sum]
    apply Finset.prod_congr rfl
    intro t _
    have hh := additive_factor_bounds (timeReward T f ω t (p t)) x B hB (hbound (p t) _) hx
    exact (Real.exp_log (by linarith : 0 < 1+x*timeReward T f ω t (p t))).symm

lemma additiveGrowth_second (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (B : ℝ) (hB : 0 < B) (hbound : ∀ e ω, |f e ω| ≤ B) (L : ℝ) (N : ℕ) :
    iteratedDeriv 2 (additiveGrowth μ K T f L N) 0 =
      -2*finiteCoefficient μ K T f (fun e => lagCovariance μ T f e e 0) N := by
  rw [((additiveGrowth_eventuallyEq μ K T f B hB hbound L N).iteratedDeriv 2).eq_of_nhds]
  exact additiveLogGrowth_second μ K T hT f hf B hB hbound L N

inductive NoiseMode where
  | lognormal
  | additive
  deriving DecidableEq

/-- Exactly the two manuscript edge factors: exp(σf−σ²E[f²]/2), or 1+σf.
The sum over paths is the actual stationary finite Markov path probability law. -/
noncomputable def finiteGrowth (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (L : ℝ) (mode : NoiseMode) (N : ℕ) : ℝ → ℝ :=
  match mode with
  | .lognormal => lognormalGrowth μ K T f (fun e => lagCovariance μ T f e e 0) L N
  | .additive => additiveGrowth μ K T f L N

lemma finiteGrowth_second (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ) (L : ℝ) (mode : NoiseMode)
    (hbounded : mode = .additive → ∃ B : ℝ, 0 < B ∧ ∀ e ω, |f e ω| ≤ B) (N : ℕ) :
    iteratedDeriv 2 (finiteGrowth μ K T f L mode N) 0 =
      -2*Dobson.FinitePathCoefficient.finiteCoefficient μ K T f N := by
  cases mode with
  | lognormal =>
    exact lognormalGrowth_second μ K T hT f hf (fun e => lagCovariance μ T f e e 0) L N
  | additive =>
    obtain ⟨B,hB,hb⟩ := hbounded rfl
    exact additiveGrowth_second μ K T hT f hf B hB hb L N

lemma finiteGrowth_zero (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω → Ω) (f : E → Ω → ℝ) (L : ℝ) (mode : NoiseMode) (N : ℕ) :
    finiteGrowth μ K T f L mode N 0=L := by
  cases mode <;> simp [finiteGrowth, lognormalGrowth, additiveGrowth, logPartition, partition,
    quadraticScore, additiveScore, expect_const]

end Dobson.FinitePathJet
#print axioms Dobson.FinitePathJet.finiteGrowth_second

namespace Dobson.Coverage6Remainders

/-- Taylor's cubic polynomial at zero, with ordinary derivatives. -/
noncomputable def cubicTaylor (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f 0 + deriv f 0 * x + iteratedDeriv 2 f 0 * x ^ 2 / 2 +
    iteratedDeriv 3 f 0 * x ^ 3 / 6

/-- A local fourth-derivative bound gives an actual quartic remainder,
including negative noise amplitudes and the exact constant 1/24. -/
theorem taylor_three_bound (f : ℝ → ℝ) (δ M x : ℝ) (hδ : 0 < δ)
    (hf : ContDiffOn ℝ 4 f (Icc (-δ) δ))
    (hM : ∀ y ∈ Icc (-δ) δ, |iteratedDeriv 4 f y| ≤ M)
    (hx : |x| ≤ δ) :
    |f x - cubicTaylor f x| ≤ M / 24 * x ^ 4 := by
  by_cases hx0 : x = 0
  · simp [hx0, cubicTaylor]
  have hn : (0 : ℝ) ≠ x := Ne.symm hx0
  have hseg : uIcc 0 x ⊆ Icc (-δ) δ := by
    apply uIcc_subset_Icc
    · exact ⟨by linarith, hδ.le⟩
    · exact abs_le.mp hx
  have hc : ContDiffAt ℝ 4 f 0 := hf.contDiffAt (Icc_mem_nhds (by linarith) hδ)
  obtain ⟨y, hy, he⟩ := taylor_mean_remainder_lagrange_iteratedDeriv
    (n := 3) hn (hf.mono hseg)
  have hd (k : ℕ) (hk : k ≤ 4) :
      iteratedDerivWithin k f (uIcc 0 x) 0 = iteratedDeriv k f 0 :=
    iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_uIcc hn)
      (hc.of_le (by exact_mod_cast hk)) (left_mem_uIcc)
  have ht : taylorWithinEval f 3 (uIcc 0 x) 0 x = cubicTaylor f x := by
    norm_num [show 3 = 2 + 1 from rfl, taylorWithinEval_succ,
      show 2 = 1 + 1 from rfl, show 1 = 0 + 1 from rfl,
      hd 1 (by omega), hd 2 (by omega), hd 3 (by omega),
      cubicTaylor, iteratedDeriv_one, smul_eq_mul]
    ; ring
  rw [ht] at he
  rw [he]
  norm_num [abs_div, abs_mul, abs_pow]
  have hy' : y ∈ Icc (-δ) δ := hseg ⟨hy.1.le, hy.2.le⟩
  calc
    |iteratedDeriv 4 f y| * |x| ^ 4 / 24 ≤ M * |x| ^ 4 / 24 := by
      gcongr
      exact hM y hy'
    _ = M / 24 * x ^ 4 := by rw [← abs_pow, abs_of_nonneg (by positivity : 0 ≤ x ^ 4)]; ring

/-- Whole-function sign symmetry cancels the linear and cubic Taylor terms.
This is stronger than marginal symmetry of one noise variable. -/
theorem even_quartic_bound (f : ℝ → ℝ) (δ M x : ℝ) (hδ : 0 < δ)
    (hf : ContDiffOn ℝ 4 f (Icc (-δ) δ))
    (hM : ∀ y ∈ Icc (-δ) δ, |iteratedDeriv 4 f y| ≤ M)
    (heven : ∀ y ∈ Icc (-δ) δ, f (-y) = f y)
    (hx : |x| ≤ δ) :
    |f x - f 0 - iteratedDeriv 2 f 0 / 2 * x ^ 2| ≤ M / 24 * x ^ 4 := by
  have hp := taylor_three_bound f δ M x hδ hf hM hx
  have hm := taylor_three_bound f δ M (-x) hδ hf hM (by simpa using hx)
  rw [heven x (abs_le.mp hx)] at hm
  have hid : 2 * (f x - f 0 - iteratedDeriv 2 f 0 / 2 * x ^ 2) =
      (f x - cubicTaylor f x) + (f x - cubicTaylor f (-x)) := by
    dsimp [cubicTaylor]; ring
  have h := abs_add_le (f x - cubicTaylor f x) (f x - cubicTaylor f (-x))
  rw [← hid, abs_mul] at h
  norm_num at h hm
  nlinarith

/-- The bound survives the infinite-time limit. `f N` is expected normalized
finite-time log growth (or a Cesàro average of one-step log growth), and `c N`
is its computed quadratic loss. Convergence, local smoothness, parity and
uniform fourth-derivative control are the unproved probability inputs.
Neither a finite-time nor an infinite-time remainder is assumed. -/
private theorem growth_limit_quartic_of_coefficient_limit (f : ℕ → ℝ → ℝ) (c : ℕ → ℝ)
    (r : ℝ → ℝ) (L C δ M : ℝ) (hδ : 0 < δ)
    (hf : ∀ N, ContDiffOn ℝ 4 (f N) (Icc (-δ) δ))
    (hM : ∀ N y, y ∈ Icc (-δ) δ → |iteratedDeriv 4 (f N) y| ≤ M)
    (heven : ∀ N y, y ∈ Icc (-δ) δ → f N (-y) = f N y)
    (hzero : ∀ N, f N 0 = L)
    (hsecond : ∀ N, iteratedDeriv 2 (f N) 0 = -2 * c N)
    (hc : Tendsto c atTop (𝓝 C))
    (hr : ∀ x, |x| ≤ δ → Tendsto (fun N => f N x) atTop (𝓝 (r x))) :
    ∀ x, |x| ≤ δ → |L - r x - C * x ^ 2| ≤ M / 24 * x ^ 4 := by
  intro x hx
  have hb (N : ℕ) : |L - f N x - c N * x ^ 2| ≤ M / 24 * x ^ 4 := by
    have h := even_quartic_bound (f N) δ M x hδ (hf N) (hM N) (heven N) hx
    rw [hzero N, hsecond N] at h
    have heq : f N x - L - -2 * c N / 2 * x ^ 2 =
        -(L - f N x - c N * x ^ 2) := by ring
    rwa [heq, abs_neg] at h
  exact le_of_tendsto ((tendsto_const_nhds.sub (hr x hx)).sub
    (hc.mul_const (x ^ 2))).abs (Eventually.of_forall hb)

/-- Quartic transfer for the actual finite path partition. The second variation
is proved from the lognormal or bounded additive edge factors and the stationary
path law. Uniform fourth derivatives remain an explicit project hypothesis. -/
theorem growth_limit_quartic {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (K : Dobson.FiniteLineageVariance.Chain E) (T : Ω ≃ᵐ Ω)
    (hT : MeasureTheory.MeasurePreserving T μ μ) (ξ : E → Ω → ℝ)
    (hξ : ∀ e, MeasureTheory.MemLp (ξ e) 2 μ)
    (B Ku β : ℝ) (hB : 0 ≤ B) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hcov : ∀ e e' m, |Dobson.FiniteLineageVariance.lagCovariance μ T ξ e e' m| ≤ B)
    (hmix : ∀ e e' m, |(K.prob^(m+1)) e' e-K.initial e| ≤ Ku*β^m)
    (mode : Dobson.FinitePathJet.NoiseMode)
    (hbounded : mode = .additive → ∃ A : ℝ, 0 < A ∧ ∀ e ω, |ξ e ω| ≤ A)
    (r : ℝ → ℝ) (L δ M : ℝ) (hδ : 0 < δ)
    (hf : ∀ N, ContDiffOn ℝ 4 (Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N) (Icc (-δ) δ))
    (hM : ∀ N y, y ∈ Icc (-δ) δ →
      |iteratedDeriv 4 (Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N) y| ≤ M)
    (heven : ∀ N y, y ∈ Icc (-δ) δ →
      Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N (-y)=
        Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N y)
    (hr : ∀ x, |x| ≤ δ → Tendsto
      (fun N => Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N x) atTop (𝓝 (r x))) :
    ∀ x, |x| ≤ δ →
      |L-r x-Dobson.FinitePathCoefficient.covarianceCoefficient μ K T ξ*x^2| ≤ M/24*x^4 := by
  exact growth_limit_quartic_of_coefficient_limit
    (Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode)
    (Dobson.FinitePathCoefficient.finiteCoefficient μ K T ξ) r L
    (Dobson.FinitePathCoefficient.covarianceCoefficient μ K T ξ) δ M hδ hf hM heven
    (Dobson.FinitePathJet.finiteGrowth_zero μ K T ξ L mode)
    (Dobson.FinitePathJet.finiteGrowth_second μ K T hT ξ hξ L mode hbounded)
    (Dobson.FinitePathCoefficient.finiteCoefficient_tendsto μ K T hT ξ hξ B Ku β hB hKu hβ0 hβ1 hcov hmix) hr

/-- Compactness turns continuous positive radii and continuous derivative
majorants into one radius and one constant. The controls must already exist
locally: this does not derive them from primitive stochastic matrices. -/
theorem compact_controls {α : Type*} [TopologicalSpace α]
    (S : Set α) (hS : IsCompact S) (radius bound : α → ℝ)
    (hr : ContinuousOn radius S) (hpos : ∀ a ∈ S, 0 < radius a)
    (hB : ContinuousOn bound S) :
    ∃ δ M : ℝ, 0 < δ ∧ 0 ≤ M ∧
      ∀ a ∈ S, δ ≤ radius a ∧ bound a ≤ M := by
  by_cases hne : S.Nonempty
  · obtain ⟨a, ha, hmin⟩ := hS.exists_isMinOn hne hr
    obtain ⟨b, hb, hmax⟩ := hS.exists_isMaxOn hne hB
    refine ⟨radius a, max 0 (bound b), hpos a ha, le_max_left _ _, ?_⟩
    intro x hx
    exact ⟨hmin hx, (hmax hx).trans (le_max_right _ _)⟩
  · refine ⟨1, 0, by norm_num, le_rfl, ?_⟩
    intro x hx
    exact (hne ⟨x, hx⟩).elim

/-- A compact family has a common noise interval and a common quartic
constant, provided continuous local derivative controls are supplied. All
finite-time bounds are derived by Taylor's theorem before taking the limit. -/
private theorem compact_growth_quartic_of_coefficient_limits {α : Type*} [TopologicalSpace α]
    (S : Set α) (hS : IsCompact S) (radius bound L C : α → ℝ)
    (f : α → ℕ → ℝ → ℝ) (c : α → ℕ → ℝ) (r : α → ℝ → ℝ)
    (hrad : ContinuousOn radius S) (hpos : ∀ a ∈ S, 0 < radius a)
    (hB : ContinuousOn bound S)
    (hf : ∀ a ∈ S, ∀ N, ContDiffOn ℝ 4 (f a N) (Icc (-radius a) (radius a)))
    (hM : ∀ a ∈ S, ∀ N y, y ∈ Icc (-radius a) (radius a) →
      |iteratedDeriv 4 (f a N) y| ≤ bound a)
    (heven : ∀ a ∈ S, ∀ N y, y ∈ Icc (-radius a) (radius a) →
      f a N (-y) = f a N y)
    (hzero : ∀ a ∈ S, ∀ N, f a N 0 = L a)
    (hsecond : ∀ a ∈ S, ∀ N, iteratedDeriv 2 (f a N) 0 = -2 * c a N)
    (hc : ∀ a ∈ S, Tendsto (c a) atTop (𝓝 (C a)))
    (hr : ∀ a ∈ S, ∀ x, |x| ≤ radius a →
      Tendsto (fun N => f a N x) atTop (𝓝 (r a x))) :
    ∃ δ K : ℝ, 0 < δ ∧ 0 ≤ K ∧
      ∀ a ∈ S, ∀ x, |x| ≤ δ → |L a - r a x - C a * x ^ 2| ≤ K * x ^ 4 := by
  obtain ⟨δ, M, hδ, hM0, hcontrols⟩ := compact_controls S hS radius bound hrad hpos hB
  refine ⟨δ, M / 24, hδ, by positivity, ?_⟩
  intro a ha x hx
  have h := growth_limit_quartic_of_coefficient_limit (f a) (c a) (r a) (L a) (C a)
    (radius a) (bound a) (hpos a ha) (hf a ha) (hM a ha)
    (heven a ha) (hzero a ha) (hsecond a ha) (hc a ha) (hr a ha)
    x (hx.trans (hcontrols a ha).1)
  exact h.trans (by gcongr; exact (hcontrols a ha).2)

/-- Compact-family transfer for actual lognormal or bounded additive path products.
The common-radius and common-fourth-derivative estimates remain explicit. -/
theorem compact_growth_quartic {α E Ω : Type*} [TopologicalSpace α]
    [Fintype E] [DecidableEq E] [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasureTheory.MeasurePreserving T μ μ)
    (K : α → Dobson.FiniteLineageVariance.Chain E) (ξ : α → E → Ω → ℝ)
    (S : Set α) (hS : IsCompact S)
    (hξ : ∀ a ∈ S, ∀ e, MeasureTheory.MemLp (ξ a e) 2 μ)
    (B Ku β : α → ℝ)
    (hBcov : ∀ a ∈ S, 0 ≤ B a) (hKu : ∀ a ∈ S, 0 ≤ Ku a)
    (hβ0 : ∀ a ∈ S, 0 ≤ β a) (hβ1 : ∀ a ∈ S, β a < 1)
    (hcov : ∀ a ∈ S, ∀ e e' m, |Dobson.FiniteLineageVariance.lagCovariance μ T (ξ a) e e' m| ≤ B a)
    (hmix : ∀ a ∈ S, ∀ e e' m, |((K a).prob^(m+1)) e' e-(K a).initial e| ≤ Ku a*(β a)^m)
    (radius bound L : α → ℝ) (mode : α → Dobson.FinitePathJet.NoiseMode)
    (hbounded : ∀ a ∈ S, mode a = .additive → ∃ A : ℝ, 0 < A ∧ ∀ e ω, |ξ a e ω| ≤ A)
    (r : α → ℝ → ℝ)
    (hrad : ContinuousOn radius S) (hpos : ∀ a ∈ S, 0 < radius a)
    (hB : ContinuousOn bound S)
    (hf : ∀ a ∈ S, ∀ N, ContDiffOn ℝ 4
      (Dobson.FinitePathJet.finiteGrowth μ (K a) T (ξ a) (L a) (mode a) N) (Icc (-radius a) (radius a)))
    (hM : ∀ a ∈ S, ∀ N y, y ∈ Icc (-radius a) (radius a) →
      |iteratedDeriv 4 (Dobson.FinitePathJet.finiteGrowth μ (K a) T (ξ a) (L a) (mode a) N) y| ≤ bound a)
    (heven : ∀ a ∈ S, ∀ N y, y ∈ Icc (-radius a) (radius a) →
      Dobson.FinitePathJet.finiteGrowth μ (K a) T (ξ a) (L a) (mode a) N (-y)=
        Dobson.FinitePathJet.finiteGrowth μ (K a) T (ξ a) (L a) (mode a) N y)
    (hr : ∀ a ∈ S, ∀ x, |x| ≤ radius a → Tendsto
      (fun N => Dobson.FinitePathJet.finiteGrowth μ (K a) T (ξ a) (L a) (mode a) N x)
      atTop (𝓝 (r a x))) :
    ∃ δ M : ℝ, 0 < δ ∧ 0 ≤ M ∧ ∀ a ∈ S, ∀ x, |x| ≤ δ →
      |L a-r a x-Dobson.FinitePathCoefficient.covarianceCoefficient μ (K a) T (ξ a)*x^2| ≤ M*x^4 := by
  exact compact_growth_quartic_of_coefficient_limits S hS radius bound L
    (fun a => Dobson.FinitePathCoefficient.covarianceCoefficient μ (K a) T (ξ a))
    (fun a => Dobson.FinitePathJet.finiteGrowth μ (K a) T (ξ a) (L a) (mode a))
    (fun a => Dobson.FinitePathCoefficient.finiteCoefficient μ (K a) T (ξ a)) r
    hrad hpos hB hf hM heven
    (fun a _ => Dobson.FinitePathJet.finiteGrowth_zero μ (K a) T (ξ a) (L a) (mode a))
    (fun a ha => Dobson.FinitePathJet.finiteGrowth_second μ (K a) T hT (ξ a) (hξ a ha)
      (L a) (mode a) (hbounded a ha))
    (fun a ha => Dobson.FinitePathCoefficient.finiteCoefficient_tendsto μ (K a) T hT (ξ a)
      (hξ a ha) (B a) (Ku a) (β a) (hBcov a ha) (hKu a ha) (hβ0 a ha) (hβ1 a ha)
      (hcov a ha) (hmix a ha)) hr

/-- Without whole-process sign symmetry, the cubic term is retained.
The bounded-forcing proof supplies four derivatives, so this form uses those
same inputs and an explicit bound on the third derivative at zero. -/
theorem quadratic_cubic_bound (f : ℝ → ℝ) (δ M B x : ℝ) (hδ : 0 < δ)
    (hf : ContDiffOn ℝ 4 f (Icc (-δ) δ))
    (hM : ∀ y ∈ Icc (-δ) δ, |iteratedDeriv 4 f y| ≤ M)
    (hfirst : deriv f 0 = 0) (hthird : |iteratedDeriv 3 f 0| ≤ B)
    (hx : |x| ≤ δ) :
    |f x - f 0 - iteratedDeriv 2 f 0 / 2 * x ^ 2| ≤
      (B / 6 + M * δ / 24) * |x| ^ 3 := by
  have ht := taylor_three_bound f δ M x hδ hf hM hx
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0 ⟨by linarith, hδ.le⟩)
  have hid : f x - f 0 - iteratedDeriv 2 f 0 / 2 * x ^ 2 =
      (f x - cubicTaylor f x) + iteratedDeriv 3 f 0 * x ^ 3 / 6 := by
    simp only [cubicTaylor, hfirst]; ring
  rw [hid]
  calc
    |(f x - cubicTaylor f x) + iteratedDeriv 3 f 0 * x ^ 3 / 6| ≤
        |f x - cubicTaylor f x| + |iteratedDeriv 3 f 0 * x ^ 3 / 6| := abs_add_le _ _
    _ ≤ M / 24 * x ^ 4 + B * |x| ^ 3 / 6 := by
      apply add_le_add ht
      simpa [abs_div, abs_mul, abs_pow] using
          (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hthird
            (by positivity : 0 ≤ |x| ^ 3)) (by norm_num : (0 : ℝ) ≤ 6))
    _ ≤ (B / 6 + M * δ / 24) * |x| ^ 3 := by
      have hp : x ^ 4 = |x| ^ 3 * |x| := by
        rw [← pow_succ, ← abs_pow, abs_of_nonneg (by positivity : 0 ≤ x ^ 4)]
      rw [hp]
      nlinarith [mul_nonneg (mul_nonneg hM0 (by positivity : 0 ≤ |x| ^ 3))
        (sub_nonneg.mpr hx)]

/-- The bounded additive model's cubic conclusion follows from centered
first variation and uniform derivative controls, without sign symmetry.
The probability inputs are stated separately from the proved error estimate. -/
private theorem growth_limit_cubic_of_coefficient_limit (f : ℕ → ℝ → ℝ) (c : ℕ → ℝ)
    (r : ℝ → ℝ) (L C δ M B : ℝ) (hδ : 0 < δ)
    (hf : ∀ N, ContDiffOn ℝ 4 (f N) (Icc (-δ) δ))
    (hM : ∀ N y, y ∈ Icc (-δ) δ → |iteratedDeriv 4 (f N) y| ≤ M)
    (hfirst : ∀ N, deriv (f N) 0 = 0)
    (hthird : ∀ N, |iteratedDeriv 3 (f N) 0| ≤ B)
    (hzero : ∀ N, f N 0 = L)
    (hsecond : ∀ N, iteratedDeriv 2 (f N) 0 = -2 * c N)
    (hc : Tendsto c atTop (𝓝 C))
    (hr : ∀ x, |x| ≤ δ → Tendsto (fun N => f N x) atTop (𝓝 (r x))) :
    ∀ x, |x| ≤ δ → |L - r x - C * x ^ 2| ≤
      (B / 6 + M * δ / 24) * |x| ^ 3 := by
  intro x hx
  have hb (N : ℕ) : |L - f N x - c N * x ^ 2| ≤
      (B / 6 + M * δ / 24) * |x| ^ 3 := by
    have h := quadratic_cubic_bound (f N) δ M B x hδ (hf N) (hM N)
      (hfirst N) (hthird N) hx
    rw [hzero N, hsecond N] at h
    have heq : f N x - L - -2 * c N / 2 * x ^ 2 =
        -(L - f N x - c N * x ^ 2) := by ring
    rwa [heq, abs_neg] at h
  exact le_of_tendsto ((tendsto_const_nhds.sub (hr x hx)).sub
    (hc.mul_const (x ^ 2))).abs (Eventually.of_forall hb)

/-- Cubic transfer for the actual multiplicative path model. The second variation
and its coefficient limit are proved; third/fourth-derivative controls remain owed. -/
theorem growth_limit_cubic {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (K : Dobson.FiniteLineageVariance.Chain E) (T : Ω ≃ᵐ Ω)
    (hT : MeasureTheory.MeasurePreserving T μ μ) (ξ : E → Ω → ℝ)
    (hξ : ∀ e, MeasureTheory.MemLp (ξ e) 2 μ)
    (Bcov Ku β : ℝ) (hBcov : 0 ≤ Bcov) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hcov : ∀ e e' m, |Dobson.FiniteLineageVariance.lagCovariance μ T ξ e e' m| ≤ Bcov)
    (hmix : ∀ e e' m, |(K.prob^(m+1)) e' e-K.initial e| ≤ Ku*β^m)
    (mode : Dobson.FinitePathJet.NoiseMode)
    (hbounded : mode = .additive → ∃ A : ℝ, 0 < A ∧ ∀ e ω, |ξ e ω| ≤ A)
    (r : ℝ → ℝ) (L δ M B : ℝ) (hδ : 0 < δ)
    (hf : ∀ N, ContDiffOn ℝ 4 (Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N) (Icc (-δ) δ))
    (hM : ∀ N y, y ∈ Icc (-δ) δ →
      |iteratedDeriv 4 (Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N) y| ≤ M)
    (hfirst : ∀ N, deriv (Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N) 0=0)
    (hthird : ∀ N, |iteratedDeriv 3 (Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N) 0| ≤ B)
    (hr : ∀ x, |x| ≤ δ → Tendsto
      (fun N => Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode N x) atTop (𝓝 (r x))) :
    ∀ x, |x| ≤ δ →
      |L-r x-Dobson.FinitePathCoefficient.covarianceCoefficient μ K T ξ*x^2| ≤
        (B/6+M*δ/24)*|x|^3 := by
  exact growth_limit_cubic_of_coefficient_limit (Dobson.FinitePathJet.finiteGrowth μ K T ξ L mode)
    (Dobson.FinitePathCoefficient.finiteCoefficient μ K T ξ) r L
    (Dobson.FinitePathCoefficient.covarianceCoefficient μ K T ξ) δ M B hδ hf hM hfirst hthird
    (Dobson.FinitePathJet.finiteGrowth_zero μ K T ξ L mode)
    (Dobson.FinitePathJet.finiteGrowth_second μ K T hT ξ hξ L mode hbounded)
    (Dobson.FinitePathCoefficient.finiteCoefficient_tendsto μ K T hT ξ hξ Bcov Ku β
      hBcov hKu hβ0 hβ1 hcov hmix) hr

end Dobson.Coverage6Remainders

#print axioms Dobson.Coverage6Remainders.growth_limit_quartic

#print axioms Dobson.Coverage6Remainders.compact_growth_quartic

#print axioms Dobson.Coverage6Remainders.growth_limit_cubic
