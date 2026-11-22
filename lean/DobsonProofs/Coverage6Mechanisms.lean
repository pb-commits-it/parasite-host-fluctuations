import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.Normed.Algebra.MatrixExponential
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals
import Mathlib.MeasureTheory.Integral.IntegralEqImproper
import Mathlib.Tactic

import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Batch 6: remaining mechanism components

Exact components of lem:mech-contact-ct, lem:mech-thermal-full,
lem:mech-common, lem:mech-common-exact and lem:mech-open.

For CT contact only the lower-level renewal and first-wait equations are inputs;
all finite coefficient reductions and matrix-product identities are proved directly.
No Gaussian concentration, ergodic theorem, stationary-projective construction,
perturbation remainder or strict stochastic threshold is claimed.
-/
/-! ## Coverage correction — 2026-11-22

The former `contact_first_wait` and `contact_integral_assembly` renewal inputs
are proved for actual finite-state transition integrals, with a named textbook
finite-Markov-semigroup hypothesis. `lem:mech-contact-ct` is core; stochastic
growth and remainder identification remain outside credit. The other rows in
this module retain their finite thermal, common-vector and scalar open-network
cores, not general stochastic threshold conclusions. -/

open Set Filter MeasureTheory
open scoped Topology

namespace Dobson.ContactCT

noncomputable def laplace (δ : ℝ) (f : ℝ → ℝ) : ℝ :=
  ∫ t in Ioi 0, Real.exp (-δ*t)*f t

lemma integrable_laplace (δ : ℝ) (hδ : 0 < δ) (f : ℝ → ℝ)
    (hf : Continuous f) (hb : ∀ t, 0 ≤ t → |f t| ≤ 1) :
    IntegrableOn (fun t => Real.exp (-δ*t)*f t) (Ioi 0) := by
  have he := integrableOn_exp_mul_Ioi (show -δ<0 by linarith) 0
  refine he.mono' ((by fun_prop : Continuous (fun t => Real.exp (-δ*t)*f t)).aestronglyMeasurable) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
  rw [Real.norm_eq_abs, abs_mul, abs_of_pos (Real.exp_pos _)]
  exact mul_le_of_le_one_right (Real.exp_pos _).le (hb t ht.le)

lemma tendsto_laplace_product (δ : ℝ) (hδ : 0 < δ) (f : ℝ → ℝ)
    (hb : ∀ t, 0 ≤ t → |f t| ≤ 1) :
    Tendsto (fun t => Real.exp (-δ*t)*f t) atTop (𝓝 0) := by
  have he : Tendsto (fun t : ℝ => Real.exp (-δ*t)) atTop (𝓝 0) :=
    by simpa only [Function.comp_def, neg_mul, id_eq] using
      Real.tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp (tendsto_id.const_mul_atTop hδ))
  apply squeeze_zero_norm' ?_ he
  filter_upwards [eventually_ge_atTop (0:ℝ)] with t ht
  rw [Real.norm_eq_abs, abs_mul, abs_of_pos (Real.exp_pos _)]
  exact mul_le_of_le_one_right (Real.exp_pos _).le (hb t ht)

/-- Discounted occupation integrates the actual scalar backward equation. -/
theorem laplace_backward (δ τ : ℝ) (hδ : 0 < δ) (hτ : 0 < τ)
    (f g : ℝ → ℝ) (hf : Continuous f) (hg : Continuous g)
    (hfb : ∀ t, 0 ≤ t → |f t| ≤ 1) (hgb : ∀ t, 0 ≤ t → |g t| ≤ 1)
    (hd : ∀ t, 0 ≤ t → HasDerivAt f ((g t-f t)/τ) t) :
    (δ+1/τ)*laplace δ f = f 0+laplace δ g/τ := by
  have hi := integrable_laplace δ hδ f hf hfb
  have hj := integrable_laplace δ hδ g hg hgb
  let fp : ℝ → ℝ := fun t => -(δ+1/τ)*(Real.exp (-δ*t)*f t)+
    (1/τ)*(Real.exp (-δ*t)*g t)
  have hfp : IntegrableOn fp (Ioi 0) := (hi.const_mul _).add (hj.const_mul _)
  have hdw (t : ℝ) (ht : t ∈ Ici 0) :
      HasDerivAt (fun t => Real.exp (-δ*t)*f t) (fp t) t := by
    convert (((hasDerivAt_id t).const_mul (-δ)).exp.mul (hd t ht)) using 1
    · rfl
    · dsimp [fp]; ring
  have he := integral_Ioi_of_hasDerivAt_of_tendsto' hdw hfp (tendsto_laplace_product δ hδ f hfb)
  dsimp only [fp] at he
  rw [integral_add (hi.const_mul _) (hj.const_mul _), integral_const_mul, integral_const_mul] at he
  simp only [mul_zero, Real.exp_zero, one_mul, zero_sub] at he
  dsimp [laplace]
  linear_combination -he


section Matrix
open scoped Matrix.Norms.Operator

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- A column generator for deterministic successive stages with exponential waits. -/
noncomputable def generator (τ : ι → ℝ) (next : ι → ι) : Matrix ι ι ℝ :=
  fun i j => (if i=next j then 1/τ j else 0)-(if i=j then 1/τ j else 0)

noncomputable def transition (τ : ι → ℝ) (next : ι → ι) (t : ℝ) : Matrix ι ι ℝ :=
  NormedSpace.exp (t • generator τ next)

lemma mul_generator (τ : ι → ℝ) (next : ι → ι) (M : Matrix ι ι ℝ) (i j : ι) :
    (M*generator τ next) i j = (M i (next j)-M i j)/τ j := by
  simp only [Matrix.mul_apply, generator, mul_sub, Finset.sum_sub_distrib,
    mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, if_true]
  ring

lemma transition_derivative (τ : ι → ℝ) (next : ι → ι) (i j : ι) (t : ℝ) :
    HasDerivAt (fun t => transition τ next t i j)
      ((transition τ next t i (next j)-transition τ next t i j)/τ j) t := by
  have he := hasDerivAt_exp_smul_const (generator τ next) t
  have hei := (hasDerivAt_pi.mp (hasDerivAt_pi.mp he i)) j
  simpa only [transition, mul_generator] using! hei

/-- The universal textbook finite-state Markov semigroup stochasticity theorem.
This says only that exponentiating an arbitrary finite Markov generator gives
transition probabilities; it contains no ring renewal or occupation conclusion. -/
def TextbookMarkovSemigroup : Prop :=
  ∀ (Q : Matrix ι ι ℝ),
    (∀ i j, i ≠ j → 0 ≤ Q i j) →
    (∀ j, ∑ i, Q i j = 0) →
    ∀ t : ℝ, 0 ≤ t → ∀ i j, 0 ≤ NormedSpace.exp (t • Q) i j ∧ NormedSpace.exp (t • Q) i j ≤ 1

lemma generator_offdiag (τ : ι → ℝ) (hτ : ∀ i, 0 < τ i) (next : ι → ι)
    (i j : ι) (hij : i ≠ j) : 0 ≤ generator τ next i j := by
  simp only [generator, if_neg hij, sub_zero]
  split_ifs
  · exact one_div_nonneg.mpr (hτ j).le
  · exact le_rfl

lemma generator_colsum (τ : ι → ℝ) (next : ι → ι) (j : ι) :
    ∑ i, generator τ next i j = 0 := by
  simp [generator, Finset.sum_sub_distrib]

lemma transition_bounded (book : TextbookMarkovSemigroup (ι := ι))
    (τ : ι → ℝ) (hτ : ∀ i, 0 < τ i) (next : ι → ι) (i j : ι) :
    ∀ t, 0 ≤ t → |transition τ next t i j| ≤ 1 := by
  intro t ht
  obtain ⟨hp,hb⟩ := book (generator τ next) (generator_offdiag τ hτ next)
    (generator_colsum τ next) t ht i j
  simpa only [transition, abs_of_nonneg hp] using hb

lemma transition_initial (τ : ι → ℝ) (next : ι → ι) (i j : ι) :
    transition τ next 0 i j = if i=j then 1 else 0 := by
  simp [transition, Matrix.one_apply]

lemma occupation_backward (book : TextbookMarkovSemigroup (ι := ι))
    (δ : ℝ) (hδ : 0 < δ) (τ : ι → ℝ) (hτ : ∀ i, 0 < τ i) (next : ι → ι) (i j : ι) :
    (δ+1/τ j)*laplace δ (fun t => transition τ next t i j) =
      (if i=j then 1 else 0)+laplace δ (fun t => transition τ next t i (next j))/τ j := by
  rw [← transition_initial τ next i j]
  apply laplace_backward δ (τ j) hδ (hτ j)
  · exact continuous_iff_continuousAt.mpr fun t => (transition_derivative τ next i j t).continuousAt
  · exact continuous_iff_continuousAt.mpr fun t => (transition_derivative τ next i (next j) t).continuousAt
  · exact transition_bounded book τ hτ next i j
  · exact transition_bounded book τ hτ next i (next j)
  · intro t ht; exact transition_derivative τ next i j t

end Matrix


section Cycle
open scoped BigOperators

def previous {n : ℕ} : Fin (n+1) → Fin (n+1) :=
  Fin.cases (Fin.last n) Fin.castSucc

noncomputable def waitTransform {n : ℕ} (θ : ℝ) (τ : Fin (n+1) → ℝ) (i : Fin (n+1)) : ℝ :=
  θ/(θ+τ i)
noncomputable def circuitTransform {n : ℕ} (θ : ℝ) (τ : Fin (n+1) → ℝ) : ℝ :=
  ∏ i, waitTransform θ τ i
noncomputable def occupation {n : ℕ} (θ : ℝ) (τ : Fin (n+1) → ℝ) (i : Fin (n+1)) : ℝ :=
  laplace (1/θ) (fun t => transition τ previous t 0 i)
noncomputable def returnIntegral {n : ℕ} (θ : ℝ) (τ : Fin (n+1) → ℝ) : ℝ :=
  occupation θ τ (Fin.last n)
noncomputable def stayIntegral {n : ℕ} (θ : ℝ) (τ : Fin (n+1) → ℝ) : ℝ :=
  occupation θ τ 0

lemma occupation_step {n : ℕ} (book : TextbookMarkovSemigroup (ι := Fin (n+1)))
    (θ : ℝ) (hθ : 0 < θ) (τ : Fin (n+1) → ℝ) (hτ : ∀ i, 0 < τ i) (i : Fin n) :
    occupation θ τ i.succ = waitTransform θ τ i.succ * occupation θ τ i.castSucc := by
  have he := occupation_backward book (1/θ) (one_div_pos.mpr hθ) τ hτ previous 0 i.succ
  have hz : (0 : Fin (n+1)) ≠ i.succ := (Fin.succ_ne_zero i).symm
  simp only [previous, Fin.cases_succ, if_neg hz, zero_add] at he
  dsimp only [occupation, waitTransform]
  have hτi := hτ i.succ
  have hs : θ+τ i.succ ≠ 0 := ne_of_gt (add_pos hθ hτi)
  field_simp at he ⊢
  nlinarith [he]

lemma occupation_forced {n : ℕ} (book : TextbookMarkovSemigroup (ι := Fin (n+1)))
    (θ : ℝ) (hθ : 0 < θ) (τ : Fin (n+1) → ℝ) (hτ : ∀ i, 0 < τ i) :
    occupation θ τ 0 = τ 0*waitTransform θ τ 0+
      waitTransform θ τ 0*occupation θ τ (Fin.last n) := by
  have he := occupation_backward book (1/θ) (one_div_pos.mpr hθ) τ hτ previous 0 0
  simp only [previous, Fin.cases_zero, ↓reduceIte] at he
  dsimp only [occupation, waitTransform]
  have hτi := hτ 0
  have hs : θ+τ 0 ≠ 0 := ne_of_gt (add_pos hθ hτi)
  field_simp at he ⊢
  nlinarith [he]

noncomputable def partialProduct {n : ℕ} (q : Fin (n+1) → ℝ) (k : Fin (n+1)) : ℝ :=
  ∏ i : Fin k.val, q ⟨i.val+1, by omega⟩

lemma partialProduct_zero {n : ℕ} (q : Fin (n+1) → ℝ) : partialProduct q 0 = 1 := by
  exact Fin.prod_univ_zero _
lemma partialProduct_succ {n : ℕ} (q : Fin (n+1) → ℝ) (k : Fin n) :
    partialProduct q k.succ = partialProduct q k.castSucc*q k.succ := by
  simp only [partialProduct, Fin.val_succ, Fin.val_castSucc, Fin.prod_univ_castSucc]
  rfl
lemma cycle_propagate {n : ℕ} (q x : Fin (n+1) → ℝ)
    (hstep : ∀ i : Fin n, x i.succ=q i.succ*x i.castSucc) (k : Fin (n+1)) :
    x k=partialProduct q k*x 0 := by
  induction k using Fin.induction with
  | zero => simp [partialProduct_zero]
  | succ k ih => rw [hstep k, ih, partialProduct_succ]; ring

lemma circuit_lt_one {n : ℕ} (θ : ℝ) (hθ : 0 < θ)
    (τ : Fin (n+1) → ℝ) (hτ : ∀ i, 0 < τ i) : circuitTransform θ τ < 1 := by
  have hq (i : Fin (n+1)) : 0 < waitTransform θ τ i ∧ waitTransform θ τ i < 1 := by
    dsimp [waitTransform]
    exact ⟨div_pos hθ (add_pos hθ (hτ i)),
      (div_lt_one (add_pos hθ (hτ i))).mpr (by linarith [hτ i])⟩
  unfold circuitTransform
  simpa using Finset.prod_lt_prod_of_nonempty₀ (s := Finset.univ)
    (fun i _ => (hq i).1) (fun i _ => (hq i).2) (Finset.univ_nonempty)

/-- Solve the actual discounted occupation cycle, including a single stage. -/
theorem occupation_cycle {n : ℕ} (book : TextbookMarkovSemigroup (ι := Fin (n+1)))
    (θ : ℝ) (hθ : 0 < θ) (τ : Fin (n+1) → ℝ) (hτ : ∀ i, 0 < τ i) :
    returnIntegral θ τ/τ 0=circuitTransform θ τ/(1-circuitTransform θ τ) ∧
    stayIntegral θ τ=τ 0*waitTransform θ τ 0/(1-circuitTransform θ τ) := by
  let q := waitTransform θ τ
  let x := occupation θ τ
  have hp := cycle_propagate q x (occupation_step book θ hθ τ hτ) (Fin.last n)
  have hf := occupation_forced book θ hθ τ hτ
  change x 0=τ 0*q 0+q 0*x (Fin.last n) at hf
  have hr : circuitTransform θ τ=q 0*partialProduct q (Fin.last n) := by
    rw [circuitTransform, Fin.prod_univ_succ]
    rfl
  have hden : 1-circuitTransform θ τ ≠ 0 := ne_of_gt (sub_pos.mpr (circuit_lt_one θ hθ τ hτ))
  have hx : x 0=τ 0*q 0/(1-circuitTransform θ τ) := by
    rw [hp] at hf
    apply (eq_div_iff hden).mpr
    rw [hr]
    dsimp only [q, x] at hf ⊢
    linear_combination hf
  constructor
  · change x (Fin.last n)/τ 0=_
    rw [hp, hx, hr]
    have hτ0 : τ 0 ≠ 0 := ne_of_gt (hτ 0)
    have hden' : 1-q 0*partialProduct q (Fin.last n) ≠ 0 := by rwa [hr] at hden
    field_simp
  · exact hx


/-- First-wait identity for the actual two transition Laplace integrals. -/
theorem occupation_first_wait {n : ℕ} (book : TextbookMarkovSemigroup (ι := Fin (n+1)))
    (θ : ℝ) (hθ : 0 < θ) (τ : Fin (n+1) → ℝ) (hτ : ∀ i, 0 < τ i) :
    stayIntegral θ τ=(1+returnIntegral θ τ/τ 0)/(1/θ+1/τ 0) := by
  have he := occupation_backward book (1/θ) (one_div_pos.mpr hθ) τ hτ previous 0 0
  simp only [previous, Fin.cases_zero, ↓reduceIte] at he
  have hs : 1/θ+1/τ 0 ≠ 0 := ne_of_gt (add_pos (one_div_pos.mpr hθ) (one_div_pos.mpr (hτ 0)))
  apply (eq_div_iff hs).mpr
  dsimp [stayIntegral, returnIntegral, occupation]
  linear_combination he

end Cycle

end Dobson.ContactCT

namespace Dobson.Coverage6Mechanisms
open scoped BigOperators
noncomputable section

def contactCT (θ T d q R : ℝ) : ℝ :=
  d^2*θ/T-d*(R-(1-d)*q)/(1-R)

/-- Solve the first-wait equation from the renewal equation. F and H represent
Laplace integrals; their transition-probability interpretation is not proved. -/
private theorem contact_first_wait_of_identities (θ τ R F H : ℝ) (hθ : 0 < θ) (hτ : 0 < τ)
    (hR : R ≠ 1) (hF : F/τ = R/(1-R))
    (hH : H = (1+F/τ)/(1/θ+1/τ)) :
    H = τ*(θ/(θ+τ))/(1-R) := by
  rw [hH, hF]
  have hs : θ+τ ≠ 0 := ne_of_gt (add_pos hθ hτ)
  have hr : 1-R ≠ 0 := sub_ne_zero.mpr (Ne.symm hR)
  field_simp
  <;> ring

/-- Substitute transition-integral identities into the projected covariance
contraction. A claimed growth expansion is not an input. -/
private theorem contact_integral_assembly_of_identities (θ τ T R F H d : ℝ)
    (hθ : 0 < θ) (hτ : 0 < τ) (hT : T ≠ 0) (hR : R ≠ 1)
    (hF : F/τ = R/(1-R)) (hH : H = (1+F/τ)/(1/θ+1/τ)) :
    T*(-(d/(T*τ))*(F-(1-d)*H-d*(θ*τ/T))) =
      contactCT θ T d (θ/(θ+τ)) R := by
  have hf : F = τ*(R/(1-R)) := by
    calc
      F = (F/τ)*τ := by field_simp
      _ = τ*(R/(1-R)) := by rw [hF]; ring
  rw [hf, contact_first_wait_of_identities θ τ R F H hθ hτ hR hF hH]
  unfold contactCT
  field_simp
  <;> ring

/-!
## Actual continuous-time reference occupations — 2026-11-16 (leanp1)

The contact identities below use the actual finite ring generator, its matrix
exponential and Laplace integrals of transition entries. The ring's local
backward equations, finite loop closure and discounted occupation identities
are proved. A named universal textbook finite-Markov-semigroup stochasticity
hypothesis supplies bounded entries of exp(tQ) for an arbitrary generator;
it contains no ring-specific renewal, occupation or perturbation conclusion.
The ecological Perron conjugation, perturbation expansion and its remainder
remain outside this finite reference-chain component.
-/
open Dobson.ContactCT

/-- The closed first-wait formula for actual reference transition integrals. -/
theorem contact_first_wait {n : ℕ} (book : TextbookMarkovSemigroup (ι := Fin (n+1)))
    (θ : ℝ) (τ : Fin (n+1) → ℝ) (hθ : 0 < θ) (hτ : ∀ i, 0 < τ i) :
    stayIntegral θ τ=τ 0*(θ/(θ+τ 0))/(1-circuitTransform θ τ) := by
  exact (occupation_cycle book θ hθ τ hτ).2

/-- Projected contact contraction from the actual transition Laplace integrals. -/
theorem contact_integral_assembly {n : ℕ}
    (book : TextbookMarkovSemigroup (ι := Fin (n+1)))
    (θ : ℝ) (τ : Fin (n+1) → ℝ) (hθ : 0 < θ) (hτ : ∀ i, 0 < τ i) (d : ℝ) :
    (∑ i, τ i)*(-(d/((∑ i, τ i)*τ 0))*(returnIntegral θ τ-(1-d)*stayIntegral θ τ-
      d*(θ*τ 0/(∑ i, τ i)))) = contactCT θ (∑ i, τ i) d (θ/(θ+τ 0)) (circuitTransform θ τ) := by
  have hT : (∑ i, τ i) ≠ 0 := ne_of_gt (Finset.sum_pos (fun i _ => hτ i) Finset.univ_nonempty)
  exact contact_integral_assembly_of_identities θ (τ 0) (∑ i, τ i) (circuitTransform θ τ)
    (returnIntegral θ τ) (stayIntegral θ τ) d hθ (hτ 0) hT (ne_of_lt (circuit_lt_one θ hθ τ hτ))
    (occupation_cycle book θ hθ τ hτ).1 (occupation_first_wait book θ hθ τ hτ)

/-- Exact gain/clock split. -/
theorem contact_split (θ T d q R : ℝ) :
    contactCT θ T d q R =
      d^2*(θ/T-R/(1-R))+d*(1-d)*((q-R)/(1-R)) := by
  unfold contactCT
  ring

/-- Scalar continuous-time contact has zero displayed coefficient. -/
theorem contact_scalar (θ τ d : ℝ) (hθ : 0 < θ) (hτ : 0 < τ) :
    contactCT θ τ d (θ/(θ+τ)) (θ/(θ+τ)) = 0 := by
  have hs : θ+τ ≠ 0 := ne_of_gt (add_pos hθ hτ)
  have hr : 1-θ/(θ+τ) = τ/(θ+τ) := by field_simp; ring
  unfold contactCT
  rw [hr]
  field_simp
  <;> ring

/-- Two stages give a memory-independent contact/gain multiplicative ratio. -/
theorem contact_two_stage (θ a b d : ℝ) (hθ : 0 < θ) (ha : 0 < a) (hb : 0 < b) :
    contactCT θ (a+b) d (θ/(θ+a)) ((θ/(θ+a))*(θ/(θ+b))) =
      (d^2+d*(1-d)*(a+b)/a) *
      (θ/(a+b)-((θ/(θ+a))*(θ/(θ+b)))/(1-(θ/(θ+a))*(θ/(θ+b)))) := by
  have hsa : θ+a ≠ 0 := ne_of_gt (add_pos hθ ha)
  have hsb : θ+b ≠ 0 := ne_of_gt (add_pos hθ hb)
  have hab : a+b ≠ 0 := ne_of_gt (add_pos ha hb)
  have hd : a*b+θ*(a+b) ≠ 0 := ne_of_gt (by positivity)
  have hr : 1-(θ/(θ+a))*(θ/(θ+b)) =
      (a*b+θ*(a+b))/((θ+a)*(θ+b)) := by field_simp; ring
  unfold contactCT
  rw [hr]
  field_simp
  <;> ring

/-- Ratio of the actual eliminated mobile-stage incoming factors, including k=0. -/
theorem retention_factor (g s k : ℝ) (hg : g ≠ 0) (hs : s ≠ 1) :
    (k*g/(1-k*s))/(g/(1-s)) = k*(1-s)/(1-k*s) := by
  have hs' : 1-s ≠ 0 := sub_ne_zero.mpr (Ne.symm hs)
  by_cases hk : 1-k*s = 0
  · simp [hk]
  · field_simp
    <;> ring

/-- The full cycle product is multiplied by the claimed retention factor. -/
theorem retention_reproduction (other g s k : ℝ) (hs : s ≠ 1) :
    other*(k*g/(1-k*s)) =
      (other*(g/(1-s)))*(k*(1-s)/(1-k*s)) := by
  have hs' : 1-s ≠ 0 := sub_ne_zero.mpr (Ne.symm hs)
  by_cases hk : 1-k*s = 0
  · simp [hk]
  · field_simp
    <;> ring

/-- Circle transit loss and open-line nonemigration give the stated factors. -/
theorem retention_specializations (m ℓ s : ℝ) :
    (1-m)+m*(1-ℓ) = 1-m*ℓ ∧
    (1-m)*(1-s)/(1-(1-m)*s) = (1-m)*(1-s)/(1-s+m*s) := by
  constructor <;> ring

/-- Retention between zero and one cannot increase mean reproduction. -/
theorem retention_bounds (s k : ℝ) (hs0 : 0 ≤ s) (hs1 : s < 1)
    (hk0 : 0 ≤ k) (hk1 : k ≤ 1) :
    0 ≤ k*(1-s)/(1-k*s) ∧ k*(1-s)/(1-k*s) ≤ 1 := by
  have hks : k*s < 1 := lt_of_le_of_lt (mul_le_of_le_one_left hs0 hk1) hs1
  have hd : 0 < 1-k*s := sub_pos.mpr hks
  constructor
  · positivity
  · rw [div_le_iff₀ hd]
    nlinarith

/-- Ordered completion pairs collect to the square of the sensitivity sum. -/
theorem common_ordered_collection {ι : Type*} (S : Finset ι)
    (κ : ι → ℝ) (F : ι → ι → ℝ) (U R : ℝ) :
    (∑ i ∈ S, ∑ j ∈ S, κ i*κ j*(U-F j i/(1-R))) =
      U*(∑ i ∈ S, κ i)^2-(∑ i ∈ S, ∑ j ∈ S, κ i*κ j*F j i)/(1-R) := by
  calc
    _ = ∑ i ∈ S, ∑ j ∈ S,
        (κ i*κ j*U-κ i*κ j*F j i/(1-R)) := by
      apply Finset.sum_congr rfl
      intro i hi
      apply Finset.sum_congr rfl
      intro j hj
      ring
    _ = (∑ i ∈ S, ∑ j ∈ S, κ i*κ j)*U-
        (∑ i ∈ S, ∑ j ∈ S, κ i*κ j*F j i)/(1-R) := by
      simp only [Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.sum_div]
    _ = _ := by
      simp only [← Finset.mul_sum, ← Finset.sum_mul]
      ring

/-- Vanishing zero-memory first-use transforms give the exact all-stage square. -/
theorem common_zero_memory {ι : Type*} (S : Finset ι) (κ : ι → ℝ) (T : ℝ) :
    (∑ i ∈ S, ∑ j ∈ S, κ i*κ j/(2*T)) = (∑ i ∈ S, κ i)^2/(2*T) := by
  simp only [← Finset.sum_div, ← Finset.mul_sum, ← Finset.sum_mul]
  rw [pow_two]

/-- Arbitrary signed sensitivities: reduction of four directed first-use terms. -/
theorem common_two_stage (θ a b x y : ℝ) (hθ : 0 < θ) (ha : 0 < a) (hb : 0 < b) :
    (1+2*θ)/(2*(a+b))*(x+y)^2-
      ((x^2+y^2)*(θ/(θ+a))*(θ/(θ+b))+x*y*(θ/(θ+a)+θ/(θ+b)))/
        (1-(θ/(θ+a))*(θ/(θ+b))) =
      (x+y)^2/(2*(a+b))+
      θ*((x+y)^2*a*b-x*y*(a+b)^2)/((a+b)*(a*b+θ*(a+b))) := by
  have hsa : θ+a ≠ 0 := ne_of_gt (add_pos hθ ha)
  have hsb : θ+b ≠ 0 := ne_of_gt (add_pos hθ hb)
  have hab : a+b ≠ 0 := ne_of_gt (add_pos ha hb)
  have hd : a*b+θ*(a+b) ≠ 0 := ne_of_gt (by positivity)
  have hr : 1-(θ/(θ+a))*(θ/(θ+b)) =
      (a*b+θ*(a+b))/((θ+a)*(θ+b)) := by field_simp; ring
  rw [hr]
  field_simp
  <;> ring

/-- Equal stays with same/opposite sensitivities, also for negative memory. -/
theorem common_equal_stays (a φ : ℝ) (ha : 1 < a) (hφ0 : -1 < φ) (hφ1 : φ < 1) :
    let q := φ/(a-(a-1)*φ)
    (1+φ)/(a*(1-φ))-(2*q^2+2*q)/(1-q^2) = 1/a ∧
    (2*q-2*q^2)/(1-q^2) = (2*φ/a)/(1-φ*(1-2/a)) := by
  dsimp
  have ha0 : 0 < a := by linarith
  have hd : 0 < a-(a-1)*φ := by nlinarith
  have he : 0 < a-(a-2)*φ := by
    by_cases hp : 0 ≤ φ
    · nlinarith
    · nlinarith
  set q : ℝ := φ/(a-(a-1)*φ) with hq
  have hrel : q*(a-(a-1)*φ) = φ := (eq_div_iff (ne_of_gt hd)).mp hq
  have hqm : 0 < 1-q := by
    have : q < 1 := by rw [hq, div_lt_iff₀ hd]; nlinarith
    linarith
  have hqp : 0 < 1+q := by
    have : -1 < q := by rw [hq, lt_div_iff₀ hd]; nlinarith
    linarith
  have hf : 1-q^2 = (1-q)*(1+q) := by ring
  have hp : 1-φ ≠ 0 := ne_of_gt (by linarith)
  have ha0ne : a ≠ 0 := ne_of_gt ha0
  have hqmn : 1-q ≠ 0 := ne_of_gt hqm
  have hqpn : 1+q ≠ 0 := ne_of_gt hqp
  have hsame : (2*q^2+2*q)/(1-q^2) = 2*q/(1-q) := by
    rw [hf]
    field_simp
    <;> ring
  have hopp : (2*q-2*q^2)/(1-q^2) = 2*q/(1+q) := by
    rw [hf]
    field_simp
    <;> ring
  rw [hsame, hopp]
  constructor
  · field_simp
    nlinarith [hrel]
  · have hr : 1-φ*(1-2/a) = (a-(a-2)*φ)/a := by field_simp <;> ring
    rw [hr, div_div_div_cancel_right₀ ha0ne]
    have hen : a-(a-2)*φ ≠ 0 := ne_of_gt he
    apply (div_eq_div_iff hqpn hen).2
    nlinarith [hrel]

/-- Full two-stage collection of completion and completion-clock contractions. -/
theorem thermal_two_stage (U R ρ₁ ρ₂ π₁ π₂ χ₁ χ₂ : ℝ) :
    π₁^2*(U-R/(1-R))+π₂^2*(U-R/(1-R))+
      π₁*π₂*(U-ρ₁/(1-R))+π₂*π₁*(U-ρ₂/(1-R))-
      π₁*χ₁*(R-ρ₁)/(1-R)-π₂*χ₂*(R-ρ₂)/(1-R)-
      π₁*χ₂*ρ₁*(1-ρ₂)/(1-R)-π₂*χ₁*ρ₂*(1-ρ₁)/(1-R) =
    π₁^2*(U-R/(1-R))+π₂^2*(U-R/(1-R))+
      π₁*π₂*(2*U-(ρ₁+ρ₂)/(1-R))+
      π₁*χ₁*((ρ₁-R)/(1-R))+π₂*χ₂*((ρ₂-R)/(1-R))-
      π₁*χ₂*(ρ₁*(1-ρ₂)/(1-R))-π₂*χ₁*(ρ₂*(1-ρ₁)/(1-R)) := by
  ring

/-- Full ordered thermal coefficient, before collecting unordered pairs.
The arguments remain finite deterministic quantities, not a stochastic model. -/
def thermalOrdered {ι : Type*} [DecidableEq ι] (S : Finset ι)
    (π χ ρ : ι → ℝ) (F : ι → ι → ℝ) (U R : ℝ) : ℝ :=
  (∑ j ∈ S, ∑ k ∈ S, π j*π k*(U-F k j/(1-R)))+
    (∑ k ∈ S, π k*χ k*((ρ k-R)/(1-R)))-
    ∑ j ∈ S, ∑ k ∈ S,
      if j=k then 0 else π j*χ k*F k j*(1-ρ k)/(1-R)

/-- All-stage zero-memory thermal coefficient: clock terms vanish and the
completion sensitivities form a single square, with no sign restrictions. -/
theorem thermal_zero_memory {ι : Type*} [DecidableEq ι] (S : Finset ι)
    (π χ : ι → ℝ) (T : ℝ) :
    thermalOrdered S π χ (fun _ => 0) (fun _ _ => 0) (1/(2*T)) 0 =
      (∑ j ∈ S, π j)^2/(2*T) := by
  simp only [thermalOrdered, zero_div, sub_zero, mul_zero, zero_sub,
    one_mul, zero_mul, div_one, ite_self, Finset.sum_const_zero, add_zero]
  simpa only [mul_one_div] using common_zero_memory S π T

/-- Completion and clock derivatives in one actual two-stage matrix column. -/
def completionClock (g q π χ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![-χ*q, 0; (π+χ)*g, 0]

/-- Clock derivatives are killed by the actual reproductive row vector. -/
theorem clock_left_cancellation (g q w₁ w₂ χ : ℝ) (hw : w₂*g = w₁*q) :
    Matrix.vecMul ![w₁,w₂] (completionClock g q 0 χ) = 0 := by
  ext i
  fin_cases i <;>
    simp [completionClock, Matrix.vecMul, dotProduct, Fin.sum_univ_two]
  calc
    _ = χ*(w₂*g-w₁*q) := by ring
    _ = 0 := by rw [hw]; ring

/-- Actual same-driver equal-stay matrix after distributing the birth tilt. -/
def commonStep (s g h : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![s,g*h;g*h,s]

/-- Every step shares the positive equal-coordinate eigenvector. -/
theorem common_step_vector (s g h : ℝ) :
    Matrix.mulVec (commonStep s g h) ![1,1] = (s+g*h) • (![1,1] : Fin 2 → ℝ) := by
  ext i
  fin_cases i <;>
    simp [commonStep, Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> ring

/-- The original possibly unequal transfer entries share an explicit vector
when their product is c squared. This is the actual diagonal-balancing bridge. -/
theorem common_original_vector (s a b c h : ℝ) (hab : a*b=c^2) :
    Matrix.mulVec (!![s,a*h;b*h,s] : Matrix (Fin 2) (Fin 2) ℝ) ![a,c] =
      (s+c*h) • (![a,c] : Fin 2 → ℝ) := by
  ext i
  fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  · ring
  · calc
      b*h*a+s*c = h*(a*b)+s*c := by ring
      _ = (s+c*h)*c := by rw [hab]; ring

/-- Chronological products of the actual same-driver matrices. -/
def commonProduct (s g : ℝ) (h : ℕ → ℝ) : ℕ → Matrix (Fin 2) (Fin 2) ℝ
  | 0 => 1
  | n+1 => commonStep s g (h n)*commonProduct s g h n

/-- Every finite product reduces exactly on the common positive eigenvector. -/
theorem common_product_vector (s g : ℝ) (h : ℕ → ℝ) (n : ℕ) :
    Matrix.mulVec (commonProduct s g h n) ![1,1] =
      (∏ t ∈ Finset.range n, (s+g*h t)) • (![1,1] : Fin 2 → ℝ) := by
  induction n with
  | zero => simp [commonProduct]
  | succ n ih =>
    rw [commonProduct, ← Matrix.mulVec_mulVec, ih, Matrix.mulVec_smul,
      common_step_vector, smul_smul, Finset.prod_range_succ]

/-- Exact finite-time logarithm; the ergodic infinite-time passage remains open. -/
theorem common_product_log (s g : ℝ) (h : ℕ → ℝ) (n : ℕ)
    (hp : ∀ t, 0 < s+g*h t) :
    Real.log (∏ t ∈ Finset.range n, (s+g*h t)) =
      ∑ t ∈ Finset.range n, Real.log (s+g*h t) := by
  exact Real.log_prod (fun t _ => ne_of_gt (hp t))

end
end Dobson.Coverage6Mechanisms

#print axioms Dobson.Coverage6Mechanisms.contact_first_wait
#print axioms Dobson.Coverage6Mechanisms.contact_integral_assembly
