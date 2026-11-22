import DobsonProofs.Coverage6Remainders
import DobsonProofs.ReferenceRing
import DobsonProofs.Coverage4ThermalFixed
import Mathlib.Dynamics.Ergodic.Ergodic
import Mathlib.Analysis.SpecialFunctions.Complex.Analytic

/-!
# Identification of the quadratic stochastic growth coefficient

The only unformalized input below is `TextbookRuelle`: the general uniform
holomorphic convergence theorem for bounded matrix cocycles near a constant
matrix with a simple dominant eigenvalue. It is a proposition passed explicitly,
not an axiom. It contains no ecological coefficient or covariance formula.

Source: D. Ruelle, Advances in Mathematics 32 (1979), 68–80,
"Analyticity properties of the characteristic exponents of random matrix
products", Theorem 3.1, Proposition 4.2, proof of Proposition 4.8, and §4.9.
Author's full preprint: https://www.ihes.fr/~ruelle/PUBLICATIONS/%5B57%5D.pdf
The common complex neighborhood and uniform convergence in the proof of 4.8
give convergence of every derivative by Cauchy's formula. §4.9 gives the
bounded measurable version. The almost-sure norm limit is the usual ergodic
Furstenberg–Kesten conclusion, also recalled in §1 of that source.

The project convention is COST: the signed quadratic growth coefficient is
minus `FinitePathCoefficient.covarianceCoefficient`. Boundedness is essential
to this use of Ruelle. No unbounded Gaussian claim is made.
-/

open MeasureTheory Filter Set
open scoped Topology BigOperators

namespace Dobson.Ident

variable {E Ω : Type*} [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

/-- Ordered product. Row vectors evolve from the left; transposing gives the
usual chronological product acting on column vectors. -/
noncomputable def matrixProduct (T : Ω → Ω) (A : Ω → Matrix E E ℝ) :
    ℕ → Ω → Matrix E E ℝ
  | 0, _ => 1
  | n + 1, ω => A ω * matrixProduct T A n (T ω)

noncomputable def projectedProduct (T : Ω → Ω) (A : Ω → Matrix E E ℝ)
    (p : E → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ∑ i, p i * (matrixProduct T A n ω).mulVec (fun _ => 1) i

/-- Maximum absolute entry, a norm on the finite matrix space. -/
noncomputable def matrixSize (A : Matrix E E ℝ) : ℝ := ‖fun i j => A i j‖

/-- Expected log of the actual matrix product, normalized by its length N+1. -/
noncomputable def matrixGrowth (μ : Measure Ω) (T : Ω → Ω)
    (A : ℝ → Ω → Matrix E E ℝ) (p : E → ℝ) (N : ℕ) (x : ℝ) : ℝ :=
  (∫ ω, Real.log (projectedProduct T (A x) p (N+1) ω) ∂μ) / (N+1 : ℕ)

/-- A generic textbook hypothesis, universally quantified over matrix families.
The base matrix is stochastic, with prescribed stationary left vector p and
explicit exponential convergence to the rank-one projection 1 ⊗ p. This is
a concrete sufficient dominated-splitting condition, not a stochastic
derivative bound. The perturbation is measurable, uniformly bounded on a
complex disk, holomorphic entry by entry, and real on the real axis.

The output refers to actual log products and their actual derivatives, and to
the almost-sure log matrix norm limit. No candidate second derivative appears.
The smoothness/jet conclusion is the Cauchy consequence of the uniform
holomorphic convergence in Ruelle 4.8, not merely analyticity of the limit.
The row-product convention is obtained by transposition in that theorem.
-/
def TextbookRuelle (μ : Measure Ω) (T : Ω ≃ᵐ Ω) : Prop :=
  IsProbabilityMeasure μ → MeasurePreserving T μ μ →
  ∀ (P : Matrix E E ℝ) (p : E → ℝ),
    (∀ i j, 0 ≤ P i j) → (∀ i, ∑ j, P i j = 1) →
    (∀ i, 0 ≤ p i) → (∑ i, p i = 1) → P.vecMul p = p →
    (∃ K β : ℝ, 0 ≤ K ∧ 0 ≤ β ∧ β < 1 ∧
      ∀ n i j, |(P^(n+1)) i j - p j| ≤ K * β^n) →
    ∀ (A : ℂ → Ω → Matrix E E ℂ) (R : ℝ), 0 < R →
      (∀ z i j, AEStronglyMeasurable (fun ω => A z ω i j) μ) →
      (∀ ω i j, DifferentiableOn ℂ (fun z => A z ω i j) (Metric.ball 0 R)) →
      (∃ B : ℝ, ∀ z ∈ Metric.ball (0 : ℂ) R, ∀ ω i j, ‖A z ω i j‖ ≤ B) →
      (∀ ω i j, A 0 ω i j = (P i j : ℂ)) →
      (∀ x : ℝ, ∀ ω i j, A x ω i j = ((A x ω i j).re : ℂ)) →
      ∃ (r : ℝ → ℝ) (δ : ℝ), 0 < δ ∧ δ < R ∧
        ContDiffOn ℝ 4 r (Ioo (-δ) δ) ∧
        (∀ x ∈ Ioo (-δ) δ, Tendsto
          (fun N => matrixGrowth μ T (fun y ω i j => (A y ω i j).re) p N x)
          atTop (𝓝 (r x))) ∧
        (∀ k : ℕ, Tendsto
          (fun N => iteratedDeriv k
            (matrixGrowth μ T (fun y ω i j => (A y ω i j).re) p N) 0)
          atTop (𝓝 (iteratedDeriv k r 0))) ∧
        (Ergodic T μ → ∀ x ∈ Ioo (-δ) δ, ∀ᵐ ω ∂μ,
          Tendsto (fun N => Real.log (matrixSize (matrixProduct T
            (fun ω i j => (A x ω i j).re) (N+1) ω)) / (N+1 : ℕ))
            atTop (𝓝 (r x)))

open Dobson.FiniteLineageVariance

/-- A weighted transition matrix uses the reward on the departure edge-state.
The last transition disappears against the terminal vector of ones. -/
noncomputable def weightedMatrix (K : Chain E) (h : Ω → E → ℝ) (ω : Ω) :
    Matrix E E ℝ := fun i j => h ω i * K.prob i j

lemma path_product_matrix (K : Chain E) (T : Ω → Ω) (h : Ω → E → ℝ)
    (n : ℕ) (ω : Ω) (i : E) :
    pathExpect K n i (fun p => ∏ t ∈ Finset.range n, h (T^[t] ω) (p t)) =
      (matrixProduct T (weightedMatrix K h) n ω).mulVec (fun _ => 1) i := by
  induction n generalizing ω i with
  | zero => simp [pathExpect, matrixProduct]
  | succ n ih =>
    simp only [pathExpect, matrixProduct]
    rw [← Matrix.mulVec_mulVec]
    simp only [Matrix.mulVec, dotProduct, weightedMatrix]
    apply Finset.sum_congr rfl
    intro j _
    have he : (fun p : ℕ → E =>
        ∏ t ∈ Finset.range (n+1), h (T^[t] ω) (prepend i p t)) =
        (fun p => h ω i * ∏ t ∈ Finset.range n, h (T^[t] (T ω)) (p t)) := by
      funext p
      rw [Finset.prod_range_succ']
      simp only [Function.iterate_zero, id_eq, prepend, Function.iterate_succ_apply]
      ring
    rw [he, pathExpect_const_mul, ih]
    simp only [Matrix.mulVec, dotProduct]
    ring

lemma expect_product_matrix (K : Chain E) (T : Ω → Ω) (h : Ω → E → ℝ)
    (n : ℕ) (ω : Ω) :
    expect K n (fun p => ∏ t ∈ Finset.range n, h (T^[t] ω) (p t)) =
      projectedProduct T (weightedMatrix K h) K.initial n ω := by
  simp only [expect, projectedProduct, path_product_matrix]

noncomputable def complexFactor (mode : FinitePathJet.NoiseMode)
    (v : E → ℝ) (f : E → Ω → ℝ) (z : ℂ) (ω : Ω) (i : E) : ℂ :=
  match mode with
  | .additive => 1 + z * f i ω
  | .lognormal => Complex.exp (z * f i ω - z^2 * v i / 2)

noncomputable def complexMatrix (K : Chain E) (mode : FinitePathJet.NoiseMode)
    (v : E → ℝ) (f : E → Ω → ℝ) (z : ℂ) (ω : Ω) : Matrix E E ℂ :=
  fun i j => complexFactor mode v f z ω i * K.prob i j

lemma complexMatrix_zero (K : Chain E) (mode : FinitePathJet.NoiseMode)
    (v : E → ℝ) (f : E → Ω → ℝ) (ω : Ω) (i j : E) :
    complexMatrix K mode v f 0 ω i j = (K.prob i j : ℂ) := by
  cases mode <;> simp [complexMatrix, complexFactor]

noncomputable def realFactor (mode : FinitePathJet.NoiseMode)
    (v : E → ℝ) (f : E → Ω → ℝ) (x : ℝ) (ω : Ω) (i : E) : ℝ :=
  match mode with
  | .additive => 1 + x * f i ω
  | .lognormal => Real.exp (x * f i ω - x^2 * v i / 2)

lemma complexFactor_ofReal (mode : FinitePathJet.NoiseMode)
    (v : E → ℝ) (f : E → Ω → ℝ) (x : ℝ) (ω : Ω) (i : E) :
    complexFactor mode v f x ω i = (realFactor mode v f x ω i : ℂ) := by
  cases mode <;> simp only [complexFactor, realFactor]
  all_goals push_cast; rfl

lemma complexMatrix_real (K : Chain E) (mode : FinitePathJet.NoiseMode)
    (v : E → ℝ) (f : E → Ω → ℝ) (x : ℝ) (ω : Ω) (i j : E) :
    complexMatrix K mode v f x ω i j =
      ((complexMatrix K mode v f x ω i j).re : ℂ) := by
  simp only [complexMatrix, complexFactor_ofReal, ← Complex.ofReal_mul, Complex.ofReal_re]

lemma complexMatrix_holomorphic (K : Chain E) (mode : FinitePathJet.NoiseMode)
    (v : E → ℝ) (f : E → Ω → ℝ) (ω : Ω) (i j : E) (R : ℝ) :
    DifferentiableOn ℂ (fun z => complexMatrix K mode v f z ω i j)
      (Metric.ball 0 R) := by
  have hid : Differentiable ℂ (fun z : ℂ => z) := differentiable_id
  cases mode with
  | additive => exact (((hid.mul_const _).const_add 1).mul_const _).differentiableOn
  | lognormal =>
    exact ((((hid.mul_const _).sub (((hid.pow 2).mul_const _).div_const 2)).cexp).mul_const _).differentiableOn

lemma complexMatrix_measurable (μ : Measure Ω) (K : Chain E)
    (mode : FinitePathJet.NoiseMode) (v : E → ℝ) (f : E → Ω → ℝ)
    (hf : ∀ i, AEStronglyMeasurable (f i) μ) (z : ℂ) (i j : E) :
    AEStronglyMeasurable (fun ω => complexMatrix K mode v f z ω i j) μ := by
  have hi : AEStronglyMeasurable (fun ω => (f i ω : ℂ)) μ :=
    Complex.continuous_ofReal.comp_aestronglyMeasurable (hf i)
  cases mode with
  | additive => exact (aestronglyMeasurable_const.add (hi.const_mul z)).mul_const _
  | lognormal =>
    exact (Complex.continuous_exp.comp_aestronglyMeasurable
      ((hi.const_mul z).sub aestronglyMeasurable_const)).mul_const _

lemma prob_le_one (K : Chain E) (i j : E) : K.prob i j ≤ 1 := by
  rw [← K.total i]
  exact Finset.single_le_sum (fun k _ => K.nonneg i k) (Finset.mem_univ j)

/-- The uniform complex bound follows from bounded host noise, before Ruelle
is used. It is independent of the product length. -/
lemma complexMatrix_bounded (K : Chain E) (mode : FinitePathJet.NoiseMode)
    (v : E → ℝ) (f : E → Ω → ℝ) (B : ℝ) (hB : 0 ≤ B)
    (hf : ∀ i ω, |f i ω| ≤ B) :
    ∃ D : ℝ, ∀ z ∈ Metric.ball (0 : ℂ) 1, ∀ ω i j,
      ‖complexMatrix K mode v f z ω i j‖ ≤ D := by
  let V := ∑ i, |v i|
  have hV : 0 ≤ V := Finset.sum_nonneg (fun i _ => abs_nonneg _)
  have hv (i : E) : |v i| ≤ V :=
    Finset.single_le_sum (fun j _ => abs_nonneg (v j)) (Finset.mem_univ i)
  refine ⟨max (1+B) (Real.exp (B+V/2)), ?_⟩
  intro z hz ω i j
  have hz' : ‖z‖ ≤ 1 := (by simpa using hz : ‖z‖ < 1).le
  have hzf : ‖z * (f i ω : ℂ)‖ ≤ B := by
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs]
    exact (mul_le_mul hz' (hf i ω) (abs_nonneg _) (by norm_num)).trans_eq (one_mul B)
  have hfac : ‖complexFactor mode v f z ω i‖ ≤ max (1+B) (Real.exp (B+V/2)) := by
    cases mode with
    | additive =>
      exact ((norm_add_le _ _).trans (by simpa using add_le_add_left hzf 1)).trans
        (le_max_left _ _)
    | lognormal =>
      apply le_trans (Complex.norm_exp_le_exp_norm _) (le_trans ?_ (le_max_right _ _))
      apply Real.exp_le_exp.mpr
      apply (norm_sub_le _ _).trans
      have hq : ‖z^2 * (v i : ℂ) / 2‖ ≤ V/2 := by
        rw [norm_div, norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs]
        norm_num
        have hz2 : ‖z‖^2 ≤ 1 := by nlinarith [norm_nonneg z]
        have hh := mul_le_mul hz2 (hv i) (abs_nonneg (v i)) (by norm_num : (0:ℝ) ≤ 1)
        linarith
      exact add_le_add hzf hq
  calc
    ‖complexMatrix K mode v f z ω i j‖ =
        ‖complexFactor mode v f z ω i‖ * K.prob i j := by
      simp [complexMatrix, norm_mul, Complex.norm_real, Real.norm_eq_abs,
        abs_of_nonneg (K.nonneg i j)]
    _ ≤ ‖complexFactor mode v f z ω i‖ :=
      mul_le_of_le_one_right (norm_nonneg _) (prob_le_one K i j)
    _ ≤ _ := hfac

open Dobson.FinitePathJet

lemma quadratic_exp_product (T : Ω → Ω) (f : E → Ω → ℝ) (v : E → ℝ)
    (n : ℕ) (x : ℝ) (ω : Ω) (p : ℕ → E) :
    Real.exp (quadraticScore (fun ω p => reward n (timeReward T f ω) p)
      (reward n (fun _ e => v e)) x ω p) =
      ∏ t ∈ Finset.range n, realFactor .lognormal v f x (T^[t] ω) (p t) := by
  simp only [realFactor, ← Real.exp_sum]
  congr 1
  simp only [quadraticScore, reward, timeReward, Finset.sum_sub_distrib,
    ← Finset.sum_div, ← Finset.mul_sum]

/-- Exact finite-horizon identification with the actual matrix product. -/
theorem finiteGrowth_eq_matrix (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (L : ℝ) (mode : NoiseMode) (N : ℕ) (x : ℝ) :
    finiteGrowth μ K T f L mode N x = L +
      matrixGrowth μ T (fun y => weightedMatrix K
        (realFactor mode (fun e => lagCovariance μ T f e e 0) f y)) K.initial N x := by
  cases mode with
  | additive =>
    simp only [finiteGrowth, additiveGrowth, matrixGrowth]
    congr 2
    apply integral_congr_ae
    exact Eventually.of_forall (fun ω => congrArg Real.log
      (expect_product_matrix K T (realFactor .additive _ f x) (N+1) ω))
  | lognormal =>
    simp only [finiteGrowth, lognormalGrowth, logPartition, partition,
      quadratic_exp_product, expect_product_matrix, matrixGrowth]

lemma re_complexMatrix (K : Chain E) (mode : NoiseMode) (v : E → ℝ)
    (f : E → Ω → ℝ) (x : ℝ) :
    (fun ω i j => (complexMatrix K mode v f x ω i j).re) =
      weightedMatrix K (realFactor mode v f x) := by
  funext ω i j
  simp [complexMatrix, complexFactor_ofReal, weightedMatrix]

/-- A first derivative can be exchanged with the finite-horizon expectation
using a bounded score derivative. This is proved, not a textbook hypothesis. -/
lemma expectedLog_first (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (n : ℕ) (S D : ℝ → Ω → (ℕ → E) → ℝ)
    (δ B : ℝ) (hδ : 0 < δ)
    (hSmeas : ∀ x p, AEStronglyMeasurable (fun ω => S x ω p) μ)
    (hDmeas : ∀ x p, AEStronglyMeasurable (fun ω => D x ω p) μ)
    (hzero : ∀ ω p, S 0 ω p = 0)
    (hder : ∀ ω x, x ∈ Ioo (-δ) δ → ∀ p,
      HasDerivAt (fun y => S y ω p) (D x ω p) x)
    (hbound : ∀ ω x, x ∈ Ioo (-δ) δ → ∀ p, |D x ω p| ≤ B) :
    deriv (fun x => ∫ ω, logPartition K n (fun y => S y ω) x ∂μ) 0 =
      ∫ ω, expect K n (D 0 ω) ∂μ := by
  have hz : (0 : ℝ) ∈ Ioo (-δ) δ := ⟨by linarith, hδ⟩
  have hp : Integrable (fun ω => logPartition K n (fun y => S y ω) 0) μ := by
    simpa [logPartition, partition, hzero, expect_const] using integrable_zero Ω ℝ μ
  have hd := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := μ) (bound := fun _ => B) (isOpen_Ioo.mem_nhds hz)
    (Eventually.of_forall (fun y => logPartition_meas K n μ S y (hSmeas y))) hp
    (logFirst_meas K n μ S D 0 (hSmeas 0) (hDmeas 0))
    (Eventually.of_forall (fun ω y hy => by
      simpa only [Real.norm_eq_abs, logFirst] using
        tilted_abs_le K n (fun z => S z ω) y (D y ω) B (hbound ω y hy)))
    (integrable_const B) (Eventually.of_forall (fun ω y hy =>
      logPartition_hasDerivAt K n (fun z => S z ω) (fun z => D z ω) y (hder ω y hy)))
  simpa [logFirst, tilted, partition, hzero, expect_const] using hd.2.deriv

lemma centered_reward_integral (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, Integrable (f e) μ)
    (hc : ∀ e, ∫ ω, f e ω ∂μ = 0) (n : ℕ) :
    (∫ ω, expect K n (reward n (timeReward T f ω)) ∂μ) = 0 := by
  have he (ω : Ω) : expect K n (reward n (timeReward T f ω)) =
      ∑ t ∈ Finset.range n, ∑ i, K.initial i * f i (T^[t] ω) := by
    unfold reward
    rw [expect_sum]
    apply Finset.sum_congr rfl
    intro t ht
    exact expect_coordinate K n t (Nat.le_of_lt (Finset.mem_range.mp ht)) _
  simp_rw [he]
  have hint (t : ℕ) (i : E) : Integrable (fun ω => K.initial i * f i (T^[t] ω)) μ :=
    ((hT.iterate t).integrable_comp_of_integrable (hf i)).const_mul _
  rw [integral_finsetSum _ (fun t _ => integrable_finsetSum _ (fun i _ => hint t i))]
  apply Finset.sum_eq_zero
  intro t _
  rw [integral_finsetSum _ (fun i _ => hint t i)]
  simp_rw [integral_const_mul]
  apply Finset.sum_eq_zero
  intro i _
  have hti : (∫ ω, f i (T^[t] ω) ∂μ) = ∫ ω, f i ω ∂μ := by
    clear! n
    induction t with
    | zero => rfl
    | succ t ih =>
      simp only [Function.iterate_succ_apply]
      rw [hT.integral_comp' (fun ω => f i (T^[t] ω)), ih]
  rw [hti, hc, mul_zero]

lemma lognormalGrowth_first (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (hc : ∀ e, ∫ ω, f e ω ∂μ = 0)
    (B : ℝ) (hB : 0 ≤ B) (hb : ∀ e ω, |f e ω| ≤ B)
    (v : E → ℝ) (L : ℝ) (N : ℕ) :
    deriv (lognormalGrowth μ K T f v L N) 0 = 0 := by
  let n := N+1
  let Y := fun ω p => reward n (timeReward T f ω) p
  let V := reward n (fun _ e => v e)
  let C := (n : ℝ) * (∑ e, |v e|)
  have hy (ω : Ω) (p : ℕ → E) : |Y ω p| ≤ n*B := by
    exact (Finset.abs_sum_le_sum_abs _ _).trans
      ((Finset.sum_le_sum (fun t _ => hb (p t) (T^[t] ω))).trans_eq (by simp))
  have hv (p : ℕ → E) : |V p| ≤ C := constant_reward_abs_le v n p
  have hC : 0 ≤ C := mul_nonneg (Nat.cast_nonneg _) (Finset.sum_nonneg (fun _ _ => abs_nonneg _))
  have hYm (p : ℕ → E) : AEStronglyMeasurable (fun ω => Y ω p) μ :=
    (reward_memLp μ T hT f hf n p).aestronglyMeasurable
  have hd := expectedLog_first μ K n (quadraticScore Y V) (quadraticFirst Y V)
    1 (n*B+C) (by norm_num)
    (fun x p => ((hYm p).const_mul x).sub aestronglyMeasurable_const)
    (fun x p => (hYm p).sub aestronglyMeasurable_const)
    (by intro ω p; simp [quadraticScore])
    (fun ω x _ p => quadraticScore_deriv Y V ω p x)
    (by
      intro ω x hx p
      have hx' : |x| ≤ 1 := abs_le.mpr ⟨hx.1.le, hx.2.le⟩
      calc
        |quadraticFirst Y V x ω p| ≤ |Y ω p| + |x| * |V p| := by
          simpa [quadraticFirst, abs_mul] using abs_sub (Y ω p) (x*V p)
        _ ≤ n*B+1*C := add_le_add (hy ω p) (mul_le_mul hx' (hv p) (abs_nonneg _) (by norm_num))
        _ = _ := by ring)
  have hcent := centered_reward_integral μ K T hT f
    (fun e => (hf e).integrable (by norm_num)) hc n
  have hd0 : quadraticFirst Y V 0 = Y := by funext ω p; simp [quadraticFirst]
  rw [hd0] at hd
  rw [hcent] at hd
  change deriv (fun x => L + (∫ ω, logPartition K n
    (fun y => quadraticScore Y V y ω) x ∂μ)/(n : ℝ)) 0 = 0
  rw [deriv_const_add, deriv_div_const, hd]
  simp

lemma additiveGrowth_first (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (hc : ∀ e, ∫ ω, f e ω ∂μ = 0)
    (B : ℝ) (hB : 0 < B) (hb : ∀ e ω, |f e ω| ≤ B)
    (L : ℝ) (N : ℕ) : deriv (additiveGrowth μ K T f L N) 0 = 0 := by
  let n := N+1
  have hm (t : ℕ) (e : E) : AEStronglyMeasurable (fun ω => timeReward T f ω t e) μ :=
    ((hf e).comp_measurePreserving (hT.iterate t)).aestronglyMeasurable
  have hfac (x : ℝ) (t : ℕ) (e : E) : AEStronglyMeasurable
      (fun ω => 1+x*timeReward T f ω t e) μ :=
    aestronglyMeasurable_const.add ((hm t e).const_mul x)
  have hsm (x : ℝ) (p : ℕ → E) : AEStronglyMeasurable
      (fun ω => additiveScore T f n x ω p) μ := by
    convert Finset.aestronglyMeasurable_sum (Finset.range n) (fun t _ =>
      ((hfac x t (p t)).aemeasurable.log).aestronglyMeasurable) using 1
    funext ω
    simp only [additiveScore, Finset.sum_apply]
  have hdm (x : ℝ) (p : ℕ → E) : AEStronglyMeasurable
      (fun ω => additiveFirst T f n x ω p) μ := by
    convert Finset.aestronglyMeasurable_sum (Finset.range n) (fun t _ =>
      ((hm t (p t)).aemeasurable.div (hfac x t (p t)).aemeasurable).aestronglyMeasurable) using 1
    funext ω
    simp only [additiveFirst, Finset.sum_apply, Pi.div_apply]
  have hfacb (ω : Ω) (x : ℝ) (hx : x ∈ Ioo (-(1/(2*B))) (1/(2*B))) (t : ℕ) (e : E) :=
    additive_factor_bounds (timeReward T f ω t e) x B hB (hb e _) hx
  have hd := expectedLog_first μ K n (additiveScore T f n) (additiveFirst T f n)
    (1/(2*B)) (n*(2*B)) (by positivity) hsm hdm
    (by intro ω p; simp [additiveScore])
    (by
      intro ω x hx p
      apply HasDerivAt.fun_sum
      intro t _
      have hp : 0 < 1+x*timeReward T f ω t (p t) := by
        linarith [(hfacb ω x hx t (p t)).1]
      simpa only [Pi.add_def, id_eq, one_mul, zero_add] using
        ((hasDerivAt_const x 1).add
          ((hasDerivAt_id x).mul_const (timeReward T f ω t (p t)))).log hp.ne')
    (by
      intro ω x hx p
      calc
        |additiveFirst T f n x ω p| ≤
            ∑ t ∈ Finset.range n, |timeReward T f ω t (p t)/(1+x*timeReward T f ω t (p t))| :=
          Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _t ∈ Finset.range n, 2*B := Finset.sum_le_sum (fun t _ => by
          have hh := hfacb ω x hx t (p t)
          have hp : 0 < 1+x*timeReward T f ω t (p t) := by linarith
          rw [abs_div, abs_of_pos hp, div_le_iff₀ hp]
          have hbp := hb (p t) (T^[t] ω)
          change |timeReward T f ω t (p t)| ≤ B at hbp
          nlinarith)
        _ = _ := by simp)
  have hd0 : additiveFirst T f n 0 = (fun ω => reward n (timeReward T f ω)) := by
    funext ω p
    simp [additiveFirst, reward]
  rw [hd0] at hd
  rw [centered_reward_integral μ K T hT f
    (fun e => (hf e).integrable (by norm_num)) hc n] at hd
  rw [((additiveGrowth_eventuallyEq μ K T f B hB hb L N).deriv).eq_of_nhds]
  change deriv (fun x => L + (∫ ω, logPartition K n
    (fun y => additiveScore T f n y ω) x ∂μ)/(n : ℝ)) 0 = 0
  rw [deriv_const_add, deriv_div_const, hd]
  simp

lemma finiteGrowth_first (μ : Measure Ω) [IsProbabilityMeasure μ]
    (K : Chain E) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (hc : ∀ e, ∫ ω, f e ω ∂μ = 0)
    (B : ℝ) (hB : 0 < B) (hb : ∀ e ω, |f e ω| ≤ B)
    (L : ℝ) (mode : NoiseMode) (N : ℕ) :
    deriv (finiteGrowth μ K T f L mode N) 0 = 0 := by
  cases mode
  · exact lognormalGrowth_first μ K T hT f hf hc B hB.le hb _ L N
  · exact additiveGrowth_first μ K T hT f hf hc B hB hb L N

lemma covariance_bound (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω → Ω) (f : E → Ω → ℝ) (B : ℝ) (hB : 0 ≤ B)
    (hb : ∀ e ω, |f e ω| ≤ B) (e e' : E) (m : ℕ) :
    |lagCovariance μ T f e e' m| ≤ B^2 := by
  unfold lagCovariance
  have hh : ∀ᵐ ω ∂μ, ‖f e (T^[m] ω) * f e' ω‖ ≤ B^2 := Eventually.of_forall (fun ω => by
    rw [Real.norm_eq_abs, abs_mul]
    simpa [pow_two] using mul_le_mul (hb e _) (hb e' _) (abs_nonneg _) hB)
  simpa [Real.norm_eq_abs] using norm_integral_le_of_norm_le_const hh

/-- The precise two-sided Peano expansion, in the project's cost convention. -/
lemma quadratic_expansion (r : ℝ → ℝ) (L C δ : ℝ) (hδ : 0 < δ)
    (hs : ContDiffOn ℝ 4 r (Ioo (-δ) δ))
    (hzero : r 0 = L) (hfirst : deriv r 0 = 0)
    (hsecond : iteratedDeriv 2 r 0 = -2*C) :
    (fun x => r x - L + C*x^2) =o[𝓝 0] (fun x : ℝ => x^2) := by
  have hz : (0 : ℝ) ∈ Ioo (-δ) δ := ⟨by linarith, hδ⟩
  have hs2 : ContDiffOn ℝ 2 r (Ioo (-δ) δ) := hs.of_le (by norm_num)
  have ha := hs2.contDiffAt (isOpen_Ioo.mem_nhds hz)
  have hd (k : ℕ) (hk : k ≤ 2) :
      iteratedDerivWithin k r (Ioo (-δ) δ) 0 = iteratedDeriv k r 0 :=
    iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_Ioo _ _)
      (ha.of_le (by exact_mod_cast hk)) hz
  have ht (x : ℝ) : taylorWithinEval r 2 (Ioo (-δ) δ) 0 x = L-C*x^2 := by
    norm_num [show 2 = 1+1 from rfl, show 1 = 0+1 from rfl,
      taylorWithinEval_succ, hd 1 (by omega), hd 2 (by omega),
      iteratedDeriv_one, hzero, hfirst, hsecond, smul_eq_mul]
    ring
  have h := taylor_isLittleO (n := 2) (convex_Ioo (-δ) δ) hz hs2
  rw [nhdsWithin_eq_nhds.mpr (isOpen_Ioo.mem_nhds hz)] at h
  have he (x : ℝ) : r x - taylorWithinEval r 2 (Ioo (-δ) δ) 0 x =
      r x - L + C*x^2 := by rw [ht]; ring
  simpa only [he, sub_zero] using h

/-- Bounded stationary centered drivers and a deterministic spectral gap
identify the covariance series with the quadratic cost of the infinite-time
matrix growth. All project-specific bridges are proved. `hRuelle` is the
explicit generic textbook hypothesis above; no uniform derivative or
remainder estimate for this model is an input.

The last clause identifies the same r with the almost-sure top Lyapunov
exponent (plus deterministic baseline L) when the noise shift is ergodic.
No temporal mixing of the noise is required. -/
theorem covariance_is_growth_coefficient
    (μ : Measure Ω) [IsProbabilityMeasure μ] (K : Chain E)
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ)
    (hRuelle : TextbookRuelle (E := E) μ T)
    (f : E → Ω → ℝ) (hf : ∀ e, AEStronglyMeasurable (f e) μ)
    (hcenter : ∀ e, ∫ ω, f e ω ∂μ = 0)
    (B : ℝ) (hB : 0 < B) (hbounded : ∀ e ω, |f e ω| ≤ B)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : ∀ e e' m, |(K.prob^(m+1)) e' e-K.initial e| ≤ Ku*β^m)
    (L : ℝ) (mode : NoiseMode) :
    ∃ (r : ℝ → ℝ) (δ : ℝ), 0 < δ ∧
      ContDiffOn ℝ 4 r (Ioo (-δ) δ) ∧ r 0 = L ∧ deriv r 0 = 0 ∧
      iteratedDeriv 2 r 0 = -2 * FinitePathCoefficient.covarianceCoefficient μ K T f ∧
      ((fun x => r x - L + FinitePathCoefficient.covarianceCoefficient μ K T f*x^2)
        =o[𝓝 0] (fun x : ℝ => x^2)) ∧
      (∀ x ∈ Ioo (-δ) δ, Tendsto (fun N => finiteGrowth μ K T f L mode N x)
        atTop (𝓝 (r x))) ∧
      (Ergodic T μ → ∀ x ∈ Ioo (-δ) δ, ∀ᵐ ω ∂μ,
        Tendsto (fun N => L + Real.log (matrixSize (matrixProduct T
          (weightedMatrix K (realFactor mode (fun e => lagCovariance μ T f e e 0) f x))
          (N+1) ω)) / (N+1 : ℕ)) atTop (𝓝 (r x))) := by
  have hf2 (e : E) : MemLp (f e) 2 μ :=
    MemLp.of_bound (hf e) B (Eventually.of_forall (fun ω => by
      simpa only [Real.norm_eq_abs] using hbounded e ω))
  let v := fun e => lagCovariance μ T f e e 0
  let A := complexMatrix K mode v f
  obtain ⟨q, δ, hδ, _hδ1, hq, hlim, hjets, htop⟩ := hRuelle inferInstance hT
    K.prob K.initial K.nonneg K.total K.initial_nonneg K.initial_total K.stationary
    ⟨Ku, β, hKu, hβ0, hβ1, fun n i j => hmix j i n⟩ A 1 (by norm_num)
    (complexMatrix_measurable μ K mode v f hf)
    (fun ω i j => complexMatrix_holomorphic K mode v f ω i j 1)
    (complexMatrix_bounded K mode v f B hB.le hbounded)
    (complexMatrix_zero K mode v f) (complexMatrix_real K mode v f)
  let r := fun x => L + q x
  have heq (N : ℕ) : finiteGrowth μ K T f L mode N =
      (fun x => L + matrixGrowth μ T (fun y ω i j => (A y ω i j).re) K.initial N x) := by
    funext x
    have hA : (fun (y : ℝ) ω i j => (A y ω i j).re) =
        (fun y => weightedMatrix K (realFactor mode v f y)) := by
      funext y
      exact re_complexMatrix K mode v f y
    rw [hA]
    exact finiteGrowth_eq_matrix μ K T f L mode N x
  have hr (x : ℝ) (hx : x ∈ Ioo (-δ) δ) :
      Tendsto (fun N => finiteGrowth μ K T f L mode N x) atTop (𝓝 (r x)) := by
    simp_rw [heq]
    exact (hlim x hx).const_add L
  have hj (k : ℕ) (hk : 0 < k) :
      Tendsto (fun N => iteratedDeriv k (finiteGrowth μ K T f L mode N) 0)
        atTop (𝓝 (iteratedDeriv k r 0)) := by
    simp_rw [heq, iteratedDeriv_const_add hk]
    simpa only [r, iteratedDeriv_const_add hk] using hjets k
  have hz : (0 : ℝ) ∈ Ioo (-δ) δ := ⟨by linarith, hδ⟩
  have hzero : r 0 = L := by
    have hh := hr 0 hz
    simp_rw [finiteGrowth_zero] at hh
    exact tendsto_nhds_unique hh tendsto_const_nhds
  have hfirst : deriv r 0 = 0 := by
    have hh := hj 1 (by norm_num)
    simp only [iteratedDeriv_one, finiteGrowth_first μ K T hT f hf2 hcenter B hB hbounded] at hh
    exact tendsto_nhds_unique hh tendsto_const_nhds
  have hc := FinitePathCoefficient.finiteCoefficient_tendsto μ K T hT f hf2
    (B^2) Ku β (sq_nonneg B) hKu hβ0 hβ1
    (covariance_bound μ T f B hB.le hbounded) hmix
  have hsecond : iteratedDeriv 2 r 0 =
      -2*FinitePathCoefficient.covarianceCoefficient μ K T f := by
    have hh := hj 2 (by norm_num)
    simp_rw [finiteGrowth_second μ K T hT f hf2 L mode
      (fun _ => ⟨B, hB, hbounded⟩)] at hh
    exact tendsto_nhds_unique hh (hc.const_mul (-2))
  have hs : ContDiffOn ℝ 4 r (Ioo (-δ) δ) := contDiffOn_const.add hq
  refine ⟨r, δ, hδ, hs, hzero, hfirst, hsecond,
    quadratic_expansion r L _ δ hδ hs hzero hfirst hsecond, hr, ?_⟩
  intro herg x hx
  filter_upwards [htop herg x hx] with ω hω
  simpa only [r, A, re_complexMatrix] using hω.const_add L

end Dobson.Ident

#print axioms Dobson.Ident.finiteGrowth_eq_matrix
#print axioms Dobson.Ident.finiteGrowth_first
#print axioms Dobson.Ident.covariance_is_growth_coefficient

namespace Dobson.Ident
open Dobson.FiniteLineageVariance Dobson.FinitePathJet
variable {ι E Ω : Type*} [Fintype ι] [DecidableEq ι]
  [Fintype E] [DecidableEq E] [MeasurableSpace Ω]

/-- The stage matrix for a reference kernel, with arbitrary edge multipliers. -/
noncomputable def kernelMatrix (K : ReferencePaths.Kernel ι E)
    (h : Ω → E → ℝ) (ω : Ω) : Matrix ι ι ℝ :=
  fun i j => ∑ e, K.prob i e * h ω e * (if K.dest e = j then 1 else 0)

lemma kernelMatrix_mulVec (K : ReferencePaths.Kernel ι E)
    (h : Ω → E → ℝ) (ω : Ω) (v : ι → ℝ) (i : ι) :
    (kernelMatrix K h ω).mulVec v i = ∑ e, K.prob i e * h ω e * v (K.dest e) := by
  simp only [Matrix.mulVec, dotProduct, kernelMatrix, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro e _
  simp [mul_ite, ite_mul]

lemma edge_stage_value (K : ReferencePaths.Kernel ι E) (T : Ω → Ω)
    (h : Ω → E → ℝ) (n : ℕ) (ω : Ω) (e : E) :
    (matrixProduct T (weightedMatrix (Chain.ofKernel K) h) (n+1) ω).mulVec (fun _ => 1) e =
      h ω e * (matrixProduct T (kernelMatrix K h) n (T ω)).mulVec (fun _ => 1) (K.dest e) := by
  induction n generalizing ω e with
  | zero =>
    simp only [matrixProduct, mul_one, Matrix.one_mulVec, Matrix.mulVec, dotProduct,
      weightedMatrix, Chain.ofKernel, mul_one, ← Finset.mul_sum, K.total]
    simp [Matrix.one_apply]
  | succ n ih =>
    rw [matrixProduct, ← Matrix.mulVec_mulVec]
    change (∑ e', (h ω e * K.prob (K.dest e) e') *
      (matrixProduct T (weightedMatrix (Chain.ofKernel K) h) (n+1) (T ω)).mulVec (fun _ => 1) e') = _
    simp_rw [ih]
    rw [matrixProduct, ← Matrix.mulVec_mulVec, kernelMatrix_mulVec]
    simp only [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro e' _
    ring

/-- Exact equality of edge-chain and physical stage projections, for every
positive horizon. Parallel contributions and unused zero-mass labels work too. -/
theorem edge_stage_projected (K : ReferencePaths.Kernel ι E) (T : Ω → Ω)
    (h : Ω → E → ℝ) (n : ℕ) (ω : Ω) :
    projectedProduct T (weightedMatrix (Chain.ofKernel K) h) (Chain.ofKernel K).initial (n+1) ω =
      projectedProduct T (kernelMatrix K h) K.initial (n+1) ω := by
  unfold projectedProduct
  simp_rw [edge_stage_value]
  simp only [Chain.ofKernel, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  rw [matrixProduct, ← Matrix.mulVec_mulVec, kernelMatrix_mulVec, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro e _
  ring

variable [Nonempty ι]

def ringDriver (ξ : ι → Ω → ℝ) (e : ι × Bool) (ω : Ω) : ℝ :=
  if e.2 then ξ e.1 ω else 0

/-- Bounded transfer drivers on the physical geometric ring; stays are fixed.
The mean ring here has Perron root one, stays 1-1/τ and transfers 1/τ. -/
lemma ring_kernelMatrix (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i)
    (next : Equiv.Perm ι) (birth : ι) (μ : Measure Ω) (T : Ω → Ω)
    (ξ : ι → Ω → ℝ) (mode : NoiseMode) (x : ℝ) :
    kernelMatrix (ReferenceRing.kernel τ hτ next)
      (realFactor mode (fun e => lagCovariance μ T (ringDriver ξ) e e 0) (ringDriver ξ) x) =
    Coverage4ThermalFixed.physicalRingStep next
      (fun ω i => (1/τ i) * realFactor mode
        (fun i => ∫ ω, ξ i ω * ξ i ω ∂μ) ξ x ω i)
      (fun _ i => 1-1/τ i) 1 birth := by
  funext ω i j
  unfold kernelMatrix
  simp only [ReferenceRing.kernel, mul_assoc]
  rw [ReferenceRing.outgoing_sum]
  cases mode <;> by_cases hi : i = j <;> by_cases hn : next i = j <;>
    simp [ReferenceRing.destination, realFactor, ringDriver, lagCovariance,
      Coverage4ThermalFixed.physicalRingStep, hi, hn]

lemma matrixProduct_physical (next : Equiv.Perm ι) (g s : Ω → ι → ℝ)
    (birth : ι) (T : Ω → Ω) (n : ℕ) (ω : Ω) :
    matrixProduct T (Coverage4ThermalFixed.physicalRingStep next g s 1 birth) n ω =
      Coverage4ThermalFixed.physicalRingProduct next g s 1 birth T n ω := by
  induction n generalizing ω with
  | zero => rfl
  | succ n ih => simp only [matrixProduct, Coverage4ThermalFixed.physicalRingProduct, ih]

/-- The finite growth in the coefficient theorem is exactly the existing
physical-ring growth, not a newly defined proxy for that growth. -/
theorem finiteGrowth_eq_physicalRing (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i)
    (next : Equiv.Perm ι) (birth : ι) (μ : Measure Ω) (T : Ω → Ω)
    (ξ : ι → Ω → ℝ) (mode : NoiseMode) (N : ℕ) (x : ℝ) :
    finiteGrowth μ (Chain.ofKernel (ReferenceRing.kernel τ hτ next)) T
      (ringDriver ξ) 0 mode N x =
    Coverage4ThermalFixed.physicalRingGrowth τ next
      (fun ω i => (1/τ i) * realFactor mode
        (fun i => ∫ ω, ξ i ω * ξ i ω ∂μ) ξ x ω i)
      (fun _ i => 1-1/τ i) 1 birth μ T N := by
  rw [finiteGrowth_eq_matrix, zero_add]
  unfold matrixGrowth
  simp_rw [edge_stage_projected, ring_kernelMatrix τ hτ next birth μ T ξ mode x]
  simp only [projectedProduct, matrixProduct_physical, ReferenceRing.kernel,
    Coverage4ThermalFixed.physicalRingGrowth]

end Dobson.Ident

#print axioms Dobson.Ident.finiteGrowth_eq_physicalRing

namespace Dobson.Ident
open Dobson.FiniteLineageVariance Dobson.FinitePathJet

/-- Local regularity gives a cubic error on a smaller closed noise interval.
The derivative bound is obtained by compactness, not supplied as a premise. -/
lemma local_cubic_remainder (r : ℝ → ℝ) (L C δ : ℝ) (hδ : 0 < δ)
    (hs : ContDiffOn ℝ 4 r (Ioo (-δ) δ)) (hzero : r 0 = L)
    (hfirst : deriv r 0 = 0) (hsecond : iteratedDeriv 2 r 0 = -2*C) :
    ∃ M : ℝ, 0 ≤ M ∧ ∀ x, |x| ≤ δ/2 →
      |L-r x-C*x^2| ≤ M*|x|^3 := by
  have hsub : Icc (-(δ/2)) (δ/2) ⊆ Ioo (-δ) δ := by
    intro x hx
    constructor <;> linarith [hx.1, hx.2]
  have hc4 : ContinuousOn (iteratedDeriv 4 r) (Ioo (-δ) δ) := by
    apply (hs.continuousOn_iteratedDerivWithin (m := 4) (by norm_num) (uniqueDiffOn_Ioo _ _)).congr
    intro x hx
    exact (iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_Ioo _ _)
      (hs.contDiffAt (isOpen_Ioo.mem_nhds hx)) hx).symm
  obtain ⟨M, hM⟩ := isCompact_Icc.exists_bound_of_continuousOn (hc4.mono hsub)
  have hm : ∀ x ∈ Icc (-(δ/2)) (δ/2), |iteratedDeriv 4 r x| ≤ M := by
    simpa only [Real.norm_eq_abs] using hM
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hm 0 ⟨by linarith, by linarith⟩)
  refine ⟨|iteratedDeriv 3 r 0|/6+M*(δ/2)/24, by positivity, ?_⟩
  intro x hx
  have hh := Coverage6Remainders.quadratic_cubic_bound r (δ/2) M
    |iteratedDeriv 3 r 0| x (by positivity) (hs.mono hsub) hm hfirst le_rfl hx
  rw [hzero, hsecond] at hh
  have he : r x-L-(-2*C)/2*x^2 = -(L-r x-C*x^2) := by ring
  simpa only [he, abs_neg] using hh

variable {ι Ω : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι] [MeasurableSpace Ω]

/-- Actual physical geometric-ring specialization. The only matrix hypothesis
is the explicitly displayed deterministic spectral gap of its reference edge
chain. Stays may be zero, hence zero-mass stay labels are allowed.

This proves the bounded-noise identification and cubic remainder for actual
physicalRingGrowth. It is not the Gaussian AR(1) quartic theorem. -/
theorem ring_covariance_is_growth_coefficient
    (τ : ι → ℝ) (hτ : ∀ i, 1 ≤ τ i) (next : Equiv.Perm ι) (birth : ι)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ)
    (hRuelle : TextbookRuelle (E := ι × Bool) μ T)
    (ξ : ι → Ω → ℝ) (hξ : ∀ i, AEStronglyMeasurable (ξ i) μ)
    (hcenter : ∀ i, ∫ ω, ξ i ω ∂μ = 0)
    (B : ℝ) (hB : 0 < B) (hbounded : ∀ i ω, |ξ i ω| ≤ B)
    (Ku β : ℝ) (hKu : 0 ≤ Ku) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hmix : let K := Chain.ofKernel (ReferenceRing.kernel τ hτ next)
      ∀ e e' m, |(K.prob^(m+1)) e' e-K.initial e| ≤ Ku*β^m)
    (mode : NoiseMode) :
    let K := Chain.ofKernel (ReferenceRing.kernel τ hτ next)
    let C := FinitePathCoefficient.covarianceCoefficient μ K T (ringDriver ξ)
    ∃ (r : ℝ → ℝ) (η M : ℝ), 0 < η ∧ 0 ≤ M ∧ r 0 = 0 ∧
      deriv r 0 = 0 ∧ iteratedDeriv 2 r 0 = -2*C ∧
      ((fun x => r x+C*x^2) =o[𝓝 0] (fun x : ℝ => x^2)) ∧
      ∀ x, |x| ≤ η → |r x+C*x^2| ≤ M*|x|^3 ∧
        Tendsto (Coverage4ThermalFixed.physicalRingGrowth τ next
          (fun ω i => (1/τ i) * realFactor mode
            (fun i => ∫ ω, ξ i ω * ξ i ω ∂μ) ξ x ω i)
          (fun _ i => 1-1/τ i) 1 birth μ T) atTop (𝓝 (r x)) := by
  let K := Chain.ofKernel (ReferenceRing.kernel τ hτ next)
  have hf : ∀ e, AEStronglyMeasurable (ringDriver ξ e) μ := by
    rintro ⟨i, b⟩
    cases b
    · exact aestronglyMeasurable_const
    · exact hξ i
  have hc : ∀ e, ∫ ω, ringDriver ξ e ω ∂μ = 0 := by
    rintro ⟨i, b⟩
    cases b <;> simp [ringDriver, hcenter]
  have hb : ∀ e ω, |ringDriver ξ e ω| ≤ B := by
    rintro ⟨i, b⟩ ω
    cases b <;> simp only [ringDriver, Bool.false_eq_true, ↓reduceIte]
    · simpa using hB.le
    · exact hbounded i ω
  obtain ⟨r, δ, hδ, hs, hz, hd, hdd, hsmall, hlim, _htop⟩ :=
    covariance_is_growth_coefficient μ K T hT hRuelle (ringDriver ξ) hf hc
      B hB hb Ku β hKu hβ0 hβ1 hmix 0 mode
  obtain ⟨M, hM, hrem⟩ := local_cubic_remainder r 0 _ δ hδ hs hz hd hdd
  refine ⟨r, δ/2, M, by positivity, hM, hz, hd, hdd, ?_, ?_⟩
  · simpa only [sub_zero] using hsmall
  · intro x hx
    constructor
    · have he : 0-r x-FinitePathCoefficient.covarianceCoefficient μ K T (ringDriver ξ)*x^2 =
          -(r x+FinitePathCoefficient.covarianceCoefficient μ K T (ringDriver ξ)*x^2) := by ring
      simpa only [he, abs_neg] using hrem x hx
    · have hi : x ∈ Ioo (-δ) δ := by
        have hh := abs_le.mp hx
        constructor <;> linarith [hh.1, hh.2]
      have hh := hlim x hi
      simpa only [K, finiteGrowth_eq_physicalRing τ hτ next birth μ T ξ mode] using hh

end Dobson.Ident

#print axioms Dobson.Ident.ring_covariance_is_growth_coefficient
