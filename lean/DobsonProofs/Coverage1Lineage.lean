import Mathlib
import DobsonProofs.ReferencePaths

/-!
# Coverage batch 1 — Item 8: Second-order lineage-variance identity (`thm:bounds-lineage`)

Formalises the matrix-perturbation, Cesàro variance-limit, and remainder-transfer core of
`thm:bounds-lineage` (`sections/04-bounds.tex`, lines 100–113, and `appendices/B-bounds.tex`,
lines 66–89):

1. **Transverse projector power identity (`M^m Q = T_⊥^m Q`) and directed edge-reuse contraction:**
   For any finite state set `V`, mean-normalized matrix `M : Matrix V V ℝ` with right/left
   eigenvectors `v, w` satisfying `M v = v`, `wᵀ M = wᵀ`, `wᵀ v = 1`, transverse projector
   `Q = I − v wᵀ`, `T_⊥ = M − v wᵀ`, and labeled edges `e : j_e → i_e` with `B_e = b_e E_{i_e, j_e}`,
   stationary edge frequency `c_e = wᵀ B_e v = b_e w(i_e) v(j_e)`, and reference transition chain
   `P(j → i) = M(i, j) w(i) / w(j)` (`w > 0`):
   - `M^m * Q = T_⊥^m * Q` for all `m : ℕ`;
   - `c_{e'} * (u_{e' → e}(m + 1) − c_e) = wᵀ B_e T_⊥^m Q B_{e'} v` for all `m : ℕ`, where
     `u_{e' → e}(m + 1) = (b_e w(i_e) / w(j_e)) * ((M^m)(j_e, i_{e'}) w(j_e) / w(i_{e'}))`.
2. **Absolute convergence and Cesàro total-variance limit (`V₂`):**
   For the private helper, under the stationary-path total-variance expansion `(H_totvar)` at horizon `T ≥ 1`,
   uniform lag-covariance bound `(H_cov)` `|γ(e, e', ℓ)| ≤ B_γ`, and primitive geometric mixing
   `(H_mix)` `|u_{e' → e}(ℓ) − c_e| ≤ K_u β^ℓ` (`0 ≤ β < 1`):
   - the lag series `∑_{m = 0}^∞ ∑_{e, e'} γ(e, e', m + 1) c_{e'} (u_{e' → e}(m + 1) − c_e)` is
     absolutely summable;
   - the Cesàro within-environment variance rate `W_T / T` converges as `T → ∞` to `V₂`;
   - `J₂ − V₂ / 2 = C''` holds identically, canceling all diagonal self-terms `∑_e c_e γ(e, e, 0)`
     and matching Tuljapurkar's matrix lag-sum coefficient `C''`.
The public actual-path theorem below proves the total-variance expansion from the finite law.
3. **Remainder transfer (`eq:bounds-lineage`):**
   Under an explicit growth-rate remainder bound `|log λ − r(σ) − C'' σ²| ≤ K |σ|^p` (`p = 3` or
   `p = 4`) on `|σ| ≤ σ₀`, putting `J(σ) = σ² J₂` and `V_between(σ) = σ² V₂` yields
   `|log λ − r(σ) − (J(σ) − V_between(σ) / 2)| ≤ K |σ|^p`.
-/

/-! ## Coverage correction — 2026-11-22

`thm:bounds-lineage` remains partial-P1. The public theorem constructs the
finite stationary path law, its variance and lag-pair counts; it no longer
takes the former `h_totvar` premise. The growth perturbation remainder
`h_rem` remains supplied. Finite variance identities alone do not establish
that infinite-time growth expansion. -/

namespace Dobson.FiniteLineageVariance
open MeasureTheory Filter
open scoped BigOperators Topology

variable {E : Type*} [Fintype E] [DecidableEq E]

/-- A primitive finite stationary edge-chain law. -/
structure Chain (E : Type*) [Fintype E] where
  prob : Matrix E E ℝ
  nonneg : ∀ i j, 0 ≤ prob i j
  total : ∀ i, ∑ j, prob i j = 1
  initial : E → ℝ
  initial_nonneg : ∀ i, 0 ≤ initial i
  initial_total : ∑ i, initial i = 1
  stationary : prob.vecMul initial = initial


/-- The actual edge chain of a reference kernel, including parallel edges. -/
noncomputable def Chain.ofKernel {ι : Type*} [Fintype ι]
    (K : Dobson.ReferencePaths.Kernel ι E) : Chain E where
  prob := fun e e' => K.prob (K.dest e) e'
  nonneg := fun e e' => K.nonneg (K.dest e) e'
  total := fun e => K.total (K.dest e)
  initial := fun e => ∑ i, K.initial i*K.prob i e
  initial_nonneg := fun e => Finset.sum_nonneg fun i _ => mul_nonneg (K.initial_nonneg i) (K.nonneg i e)
  initial_total := by
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, K.total, mul_one, K.initial_total]
  stationary := by
    funext e
    change (∑ e', (∑ i, K.initial i*K.prob i e')*K.prob (K.dest e') e)=
      ∑ i, K.initial i*K.prob i e
    simp only [Finset.sum_mul]
    rw [Finset.sum_comm]
    simpa only [Finset.mul_sum, mul_assoc] using K.stationary (fun i => K.prob i e)

def prepend (i : E) (p : ℕ → E) : ℕ → E
  | 0 => i
  | n+1 => p n

/-- Actual finite Markov path expectation, starting in the prescribed edge.
There are n transitions; coordinates after n repeat the terminal edge. -/
noncomputable def pathExpect (K : Chain E) : ℕ → E → ((ℕ → E) → ℝ) → ℝ
  | 0, i, F => F (fun _ => i)
  | n+1, i, F => ∑ j, K.prob i j * pathExpect K n j (fun p => F (prepend i p))

noncomputable def expect (K : Chain E) (n : ℕ) (F : (ℕ → E) → ℝ) : ℝ :=
  ∑ i, K.initial i * pathExpect K n i F

lemma pathExpect_const (K : Chain E) (n : ℕ) (i : E) (a : ℝ) :
    pathExpect K n i (fun _ => a)=a := by
  induction n generalizing i with
  | zero => rfl
  | succ n ih => simp only [pathExpect, ih, ← Finset.sum_mul, K.total, one_mul]

lemma expect_const (K : Chain E) (n : ℕ) (a : ℝ) : expect K n (fun _ => a)=a := by
  simp only [expect, pathExpect_const, ← Finset.sum_mul, K.initial_total, one_mul]

lemma pathExpect_add (K : Chain E) (n : ℕ) (i : E) (F G : (ℕ → E) → ℝ) :
    pathExpect K n i (fun p => F p+G p)=pathExpect K n i F+pathExpect K n i G := by
  induction n generalizing i F G with
  | zero => rfl
  | succ n ih => simp only [pathExpect, ih, mul_add, Finset.sum_add_distrib]

lemma pathExpect_mul_const (K : Chain E) (n : ℕ) (i : E) (F : (ℕ → E) → ℝ) (a : ℝ) :
    pathExpect K n i (fun p => F p*a)=pathExpect K n i F*a := by
  induction n generalizing i F with
  | zero => rfl
  | succ n ih => simp only [pathExpect, ih, ← mul_assoc, ← Finset.sum_mul]

lemma pathExpect_const_mul (K : Chain E) (n : ℕ) (i : E) (a : ℝ) (F : (ℕ → E) → ℝ) :
    pathExpect K n i (fun p => a*F p)=a*pathExpect K n i F := by
  simp only [mul_comm a, pathExpect_mul_const]

lemma pathExpect_sum {α : Type*} (K : Chain E) (n : ℕ) (i : E)
    (s : Finset α) (F : α → (ℕ → E) → ℝ) :
    pathExpect K n i (fun p => ∑ t ∈ s, F t p)=∑ t ∈ s, pathExpect K n i (F t) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [pathExpect_const]
  | @insert a s ha ih => simp only [Finset.sum_insert ha, pathExpect_add, ih]

lemma expect_sum {α : Type*} (K : Chain E) (n : ℕ) (s : Finset α) (F : α → (ℕ → E) → ℝ) :
    expect K n (fun p => ∑ t ∈ s, F t p)=∑ t ∈ s, expect K n (F t) := by
  simp only [expect, pathExpect_sum, Finset.mul_sum]
  exact Finset.sum_comm

lemma pathExpect_nonneg (K : Chain E) (n : ℕ) (i : E) (F : (ℕ → E) → ℝ)
    (hF : ∀ p, 0 ≤ F p) : 0 ≤ pathExpect K n i F := by
  induction n generalizing i F with
  | zero => exact hF _
  | succ n ih =>
      exact Finset.sum_nonneg fun j _ =>
        mul_nonneg (K.nonneg i j) (ih j _ (fun p => hF _))

lemma expect_nonneg (K : Chain E) (n : ℕ) (F : (ℕ → E) → ℝ)
    (hF : ∀ p, 0 ≤ F p) : 0 ≤ expect K n F :=
  Finset.sum_nonneg fun i _ => mul_nonneg (K.initial_nonneg i) (pathExpect_nonneg K n i F hF)

lemma pathExpect_coordinate (K : Chain E) (n t : ℕ) (ht : t ≤ n) (i : E) (f : E → ℝ) :
    pathExpect K n i (fun p => f (p t))=(K.prob^t).mulVec f i := by
  induction n generalizing t i with
  | zero =>
      have : t=0 := by omega
      subst t
      simp [pathExpect]
  | succ n ih =>
    cases t with
    | zero => simp [pathExpect, prepend, pathExpect_const, ← Finset.sum_mul, K.total]
    | succ t =>
      simp only [pathExpect, prepend, ih t (by omega), pow_succ', ← Matrix.mulVec_mulVec]
      rfl

lemma pathExpect_pair (K : Chain E) (n s t : ℕ) (hst : s+t ≤ n)
    (i : E) (f g : E → ℝ) :
    pathExpect K n i (fun p => f (p s)*g (p (s+t))) =
      (K.prob^s).mulVec (fun j => f j*(K.prob^t).mulVec g j) i := by
  induction n generalizing s i with
  | zero =>
    have hs : s=0 := by omega
    have ht : t=0 := by omega
    subst s; subst t; simp [pathExpect]
  | succ n ih =>
    cases s with
    | zero =>
      simp only [Nat.zero_add, pow_zero, Matrix.one_mulVec]
      cases t with
      | zero => simp [pathExpect, prepend, pathExpect_const, ← Finset.sum_mul, K.total]
      | succ t =>
        simp only [pathExpect, prepend, pathExpect_const_mul,
          pathExpect_coordinate K n t (by omega), pow_succ', ← Matrix.mulVec_mulVec]
        change (∑ j, K.prob i j*(f i*(K.prob^t).mulVec g j)) =
          f i*(∑ j, K.prob i j*(K.prob^t).mulVec g j)
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl; intro j hj; ring
    | succ s =>
      simp only [pathExpect, Nat.succ_add, prepend, ih s (by omega), pow_succ', ← Matrix.mulVec_mulVec]
      rfl

lemma stationary_power (K : Chain E) (t : ℕ) : (K.prob^t).vecMul K.initial=K.initial := by
  induction t with
  | zero => simp
  | succ t ih => rw [pow_succ, ← Matrix.vecMul_vecMul, ih, K.stationary]


lemma stationary_expectation (K : Chain E) (t : ℕ) (f : E → ℝ) :
    (∑ i, K.initial i*(K.prob^t).mulVec f i)=∑ i, K.initial i*f i := by
  change dotProduct K.initial ((K.prob^t).mulVec f)=dotProduct K.initial f
  rw [Matrix.dotProduct_mulVec, stationary_power]

lemma expect_coordinate (K : Chain E) (n t : ℕ) (ht : t ≤ n) (f : E → ℝ) :
    expect K n (fun p => f (p t))=∑ i, K.initial i*f i := by
  simp only [expect, pathExpect_coordinate K n t ht]
  exact stationary_expectation K t f

lemma expect_pair (K : Chain E) (n s t : ℕ) (hst : s+t ≤ n) (f g : E → ℝ) :
    expect K n (fun p => f (p s)*g (p (s+t))) =
      ∑ i, ∑ j, K.initial i*(K.prob^t) i j*f i*g j := by
  simp only [expect, pathExpect_pair K n s t hst]
  rw [stationary_expectation]
  simp only [Matrix.mulVec, dotProduct, Finset.mul_sum]
  apply Finset.sum_congr rfl; intro i hi
  apply Finset.sum_congr rfl; intro j hj
  ring



noncomputable def reward (n : ℕ) (f : ℕ → E → ℝ) (p : ℕ → E) : ℝ :=
  ∑ t ∈ Finset.range n, f t (p t)

/-- Variance under the actual finite Markov path law, at fixed environment. -/
noncomputable def pathVariance (K : Chain E) (n : ℕ) (f : ℕ → E → ℝ) : ℝ :=
  expect K n (fun p => (reward n f p)^2)-(expect K n (reward n f))^2

noncomputable def pathCovariance (K : Chain E) (n : ℕ) (f : ℕ → E → ℝ) (s t : ℕ) : ℝ :=
  expect K n (fun p => f s (p s)*f t (p t))-
    (∑ i, K.initial i*f s i)*(∑ j, K.initial j*f t j)

lemma pathCovariance_symm (K : Chain E) (n : ℕ) (f : ℕ → E → ℝ) (s t : ℕ) :
    pathCovariance K n f s t=pathCovariance K n f t s := by
  simp only [pathCovariance, mul_comm]


lemma expect_add (K : Chain E) (n : ℕ) (F G : (ℕ → E) → ℝ) :
    expect K n (fun p => F p+G p)=expect K n F+expect K n G := by
  simp only [expect, pathExpect_add, mul_add, Finset.sum_add_distrib]
lemma expect_const_mul (K : Chain E) (n : ℕ) (a : ℝ) (F : (ℕ → E) → ℝ) :
    expect K n (fun p => a*F p)=a*expect K n F := by
  simp only [expect, pathExpect_const_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl; intro i hi; ring

/-- The definition is exactly the expected squared centered path reward. -/
lemma pathVariance_eq_centered (K : Chain E) (n : ℕ) (f : ℕ → E → ℝ) :
    pathVariance K n f=expect K n (fun p =>
      (reward n f p-expect K n (reward n f))^2) := by
  have he : (fun p => (reward n f p-expect K n (reward n f))^2) =
      (fun p => (reward n f p)^2+(-2*expect K n (reward n f))*reward n f p+
        (expect K n (reward n f))^2) := by funext p; ring
  rw [he, expect_add, expect_add, expect_const_mul, expect_const]
  unfold pathVariance
  ring

lemma pathVariance_nonneg (K : Chain E) (n : ℕ) (f : ℕ → E → ℝ) : 0 ≤ pathVariance K n f := by
  rw [pathVariance_eq_centered]
  exact expect_nonneg K n _ (fun p => sq_nonneg _)

lemma pathVariance_eq_sum (K : Chain E) (n : ℕ) (f : ℕ → E → ℝ) :
    pathVariance K n f=∑ s ∈ Finset.range n, ∑ t ∈ Finset.range n, pathCovariance K n f s t := by
  have hmean : expect K n (reward n f)=∑ t ∈ Finset.range n, ∑ i, K.initial i*f t i := by
    change expect K n (fun p => ∑ t ∈ Finset.range n, f t (p t))=_
    rw [expect_sum]
    exact Finset.sum_congr rfl fun t ht => expect_coordinate K n t (Nat.le_of_lt (Finset.mem_range.mp ht)) _
  have he (p : ℕ → E) : (reward n f p)^2 =
      ∑ s ∈ Finset.range n, ∑ t ∈ Finset.range n, f s (p s)*f t (p t) := by
    rw [reward, pow_two, Finset.sum_mul_sum]
  have hmeansquare : (expect K n (reward n f))^2 =
      ∑ s ∈ Finset.range n, ∑ t ∈ Finset.range n,
        (∑ i, K.initial i*f s i)*(∑ j, K.initial j*f t j) := by
    rw [hmean, pow_two, Finset.sum_mul_sum]
  unfold pathVariance
  simp only [he, expect_sum, hmeansquare]
  simp only [pathCovariance, Finset.sum_sub_distrib]

lemma pathCovariance_lag (K : Chain E) (n s t : ℕ) (hst : s+t ≤ n) (f : ℕ → E → ℝ) :
    pathCovariance K n f s (s+t) =
      ∑ i, ∑ j, K.initial i*((K.prob^t) i j-K.initial j)*f s i*f (s+t) j := by
  rw [pathCovariance, expect_pair K n s t hst]
  rw [Fintype.sum_mul_sum]
  simp only [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl; intro i hi
  apply Finset.sum_congr rfl; intro j hj
  ring

/-- Count every diagonal and every ordered lag pair once in a symmetric array. -/
lemma sum_toeplitz (b : ℕ → ℝ) (n : ℕ) :
    (∑ s ∈ Finset.range n, ∑ t ∈ Finset.range n, (b (s-t)+b (t-s)-b 0)) =
      (n:ℝ)*b 0+2*∑ k ∈ Finset.range n, ((n:ℝ)-(k+1))*b (k+1) := by
  have hrow (n : ℕ) : (∑ k ∈ Finset.range n, b (n-k)) = ∑ k ∈ Finset.range n, b (k+1) := by
    rw [← Finset.sum_range_reflect (fun k => b (k+1)) n]
    apply Finset.sum_congr rfl
    intro k hk
    congr 1
    have hk := Finset.mem_range.mp hk
    omega
  induction n with
  | zero => simp
  | succ n ih =>
    simp only [Finset.sum_range_succ]
    have hc : (∑ s ∈ Finset.range n, (b (s-n)+b (n-s)-b 0)) =
        ∑ k ∈ Finset.range n, b (k+1) := by
      convert hrow n using 1
      apply Finset.sum_congr rfl
      intro s hs
      have hsn : s-n=0 := Nat.sub_eq_zero_of_le (Nat.le_of_lt (Finset.mem_range.mp hs))
      rw [hsn]; ring
    have hd : (∑ t ∈ Finset.range n, (b (n-t)+b (t-n)-b 0)) =
        ∑ k ∈ Finset.range n, b (k+1) := by
      convert hrow n using 1
      apply Finset.sum_congr rfl
      intro t ht
      have htn : t-n=0 := Nat.sub_eq_zero_of_le (Nat.le_of_lt (Finset.mem_range.mp ht))
      rw [htn]; ring
    rw [Finset.sum_add_distrib, ih, hc, hd]
    simp only [Nat.sub_self, Nat.cast_add, Nat.cast_one]
    have hsum : (∑ k ∈ Finset.range n, (((n:ℝ)+1)-(k+1))*b (k+1)) =
        (∑ k ∈ Finset.range n, ((n:ℝ)-(k+1))*b (k+1))+
          ∑ k ∈ Finset.range n, b (k+1) := by
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl; intro k hk; ring
    rw [hsum]
    ring

lemma sum_stationary_pairs (a : ℕ → ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ)
    (hsymm : ∀ s t, a s t=a t s)
    (hlag : ∀ s t, s+t<n → a s (s+t)=b t) :
    (∑ s ∈ Finset.range n, ∑ t ∈ Finset.range n, a s t) =
      (n:ℝ)*b 0+2*∑ k ∈ Finset.range n, ((n:ℝ)-(k+1))*b (k+1) := by
  rw [← sum_toeplitz]
  apply Finset.sum_congr rfl; intro s hs
  apply Finset.sum_congr rfl; intro t ht
  by_cases hst : s ≤ t
  · have hh := hlag s (t-s) (by simpa [Nat.add_sub_of_le hst] using Finset.mem_range.mp ht)
    rw [Nat.add_sub_of_le hst] at hh
    rw [hh, Nat.sub_eq_zero_of_le hst]; ring
  · have hts : t ≤ s := by omega
    have hh := hlag t (s-t) (by simpa [Nat.add_sub_of_le hts] using Finset.mem_range.mp hs)
    rw [Nat.add_sub_of_le hts] at hh
    rw [hsymm, hh, Nat.sub_eq_zero_of_le hts]; ring

section Environment
variable {Ω : Type*} [MeasurableSpace Ω]

noncomputable def timeReward (T : Ω → Ω) (f : E → Ω → ℝ) (ω : Ω) (t : ℕ) (e : E) : ℝ :=
  f e (T^[t] ω)

noncomputable def lagCovariance (μ : Measure Ω) (T : Ω → Ω) (f : E → Ω → ℝ)
    (e e' : E) (t : ℕ) : ℝ := ∫ ω, f e (T^[t] ω)*f e' ω ∂μ

noncomputable def withinVariance (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (n : ℕ) : ℝ := ∫ ω, pathVariance K n (timeReward T f ω) ∂μ

noncomputable def varianceLag (μ : Measure Ω) (K : Chain E) (T : Ω → Ω)
    (f : E → Ω → ℝ) (t : ℕ) : ℝ :=
  ∑ i, ∑ j, K.initial i*((K.prob^t) i j-K.initial j)*lagCovariance μ T f j i t

private lemma integral_shift (μ : Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (g : Ω → ℝ) (n : ℕ) :
    (∫ ω, g ((T : Ω → Ω)^[n] ω) ∂μ)=∫ ω, g ω ∂μ := by
  induction n with
  | zero => simp
  | succ n ih =>
    simpa only [Function.iterate_succ_apply] using
      (hT.integral_comp' (fun ω => g ((T : Ω → Ω)^[n] ω))).trans ih

lemma timeReward_product_integrable (μ : Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (i j : E) (s t : ℕ) :
    Integrable (fun ω => timeReward T f ω s i*timeReward T f ω t j) μ :=
  ((hf i).comp_measurePreserving (hT.iterate s)).integrable_mul
    ((hf j).comp_measurePreserving (hT.iterate t))

lemma integral_timeReward_pair (μ : Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (f : E → Ω → ℝ) (i j : E) (s t : ℕ) :
    (∫ ω, timeReward T f ω s i*timeReward T f ω (s+t) j ∂μ)=lagCovariance μ T f j i t := by
  have he := integral_shift μ T hT (fun ω => f j ((T : Ω → Ω)^[t] ω)*f i ω) s
  dsimp only [timeReward, lagCovariance]
  rw [Nat.add_comm s t]
  simpa only [Function.iterate_add_apply, mul_comm] using he

lemma pathCovariance_lag_integrable (μ : Measure Ω) (K : Chain E) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (n s t : ℕ) (hst : s+t ≤ n) :
    Integrable (fun ω => pathCovariance K n (timeReward T f ω) s (s+t)) μ := by
  simp_rw [pathCovariance_lag K n s t hst, mul_assoc]
  exact integrable_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ =>
    ((timeReward_product_integrable μ T hT f hf i j s (s+t)).const_mul _).const_mul _

lemma pathCovariance_integrable (μ : Measure Ω) (K : Chain E) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (n s t : ℕ) (hs : s ≤ n) (ht : t ≤ n) :
    Integrable (fun ω => pathCovariance K n (timeReward T f ω) s t) μ := by
  by_cases hst : s ≤ t
  · have hh := pathCovariance_lag_integrable μ K T hT f hf n s (t-s) (by omega)
    simpa only [Nat.add_sub_of_le hst] using hh
  · have hts : t ≤ s := by omega
    have hh := pathCovariance_lag_integrable μ K T hT f hf n t (s-t) (by omega)
    simpa only [Nat.add_sub_of_le hts, pathCovariance_symm K n _ t s] using hh

lemma integral_pathCovariance_lag (μ : Measure Ω) (K : Chain E) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (n s t : ℕ) (hst : s+t ≤ n) :
    (∫ ω, pathCovariance K n (timeReward T f ω) s (s+t) ∂μ)=varianceLag μ K T f t := by
  have hi (i j : E) := timeReward_product_integrable μ T hT f hf i j s (s+t)
  simp_rw [pathCovariance_lag K n s t hst, mul_assoc]
  rw [integral_finsetSum _ (fun i _ => integrable_finsetSum _ fun j _ => ((hi i j).const_mul _).const_mul _)]
  unfold varianceLag
  apply Finset.sum_congr rfl; intro i hi'
  rw [integral_finsetSum _ (fun j _ => ((hi i j).const_mul _).const_mul _)]
  apply Finset.sum_congr rfl; intro j hj
  rw [integral_const_mul, integral_const_mul, integral_timeReward_pair μ T hT]
  ring

/-- The finite-time within-environment variance identity before division by time. -/
theorem withinVariance_finite (μ : Measure Ω) (K : Chain E) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (n : ℕ) :
    withinVariance μ K T f n=(n:ℝ)*varianceLag μ K T f 0+
      2*∑ k ∈ Finset.range n, ((n:ℝ)-(k+1))*varianceLag μ K T f (k+1) := by
  have hi (s t : ℕ) (hs : s<n) (ht : t<n) :=
    pathCovariance_integrable μ K T hT f hf n s t hs.le ht.le
  unfold withinVariance
  simp_rw [pathVariance_eq_sum]
  rw [integral_finsetSum _ (fun s hs => integrable_finsetSum _ fun t ht =>
    hi s t (Finset.mem_range.mp hs) (Finset.mem_range.mp ht))]
  have hex : (∑ s ∈ Finset.range n, ∫ ω, ∑ t ∈ Finset.range n,
      pathCovariance K n (timeReward T f ω) s t ∂μ) =
      ∑ s ∈ Finset.range n, ∑ t ∈ Finset.range n,
        ∫ ω, pathCovariance K n (timeReward T f ω) s t ∂μ := by
    apply Finset.sum_congr rfl; intro s hs
    exact integral_finsetSum _ (fun t ht => hi s t (Finset.mem_range.mp hs) (Finset.mem_range.mp ht))
  rw [hex]
  apply sum_stationary_pairs
  · intro s t
    simp only [pathCovariance_symm K n _ s t]
  · intro s t hst
    exact integral_pathCovariance_lag μ K T hT f hf n s t hst.le

lemma varianceLag_zero (μ : Measure Ω) (K : Chain E) (T : Ω → Ω) (f : E → Ω → ℝ) :
    varianceLag μ K T f 0 =
      (∑ e, K.initial e*lagCovariance μ T f e e 0)-
        ∑ e, ∑ e', K.initial e*K.initial e'*lagCovariance μ T f e e' 0 := by
  unfold varianceLag
  simp only [pow_zero, mul_sub, sub_mul, Finset.sum_sub_distrib]
  congr 1
  · apply Finset.sum_congr rfl; intro i hi
    simp [Matrix.one_apply, mul_ite, ite_mul]
  · apply Finset.sum_congr rfl; intro i hi
    apply Finset.sum_congr rfl; intro j hj
    have hsym : lagCovariance μ T f j i 0=lagCovariance μ T f i j 0 := by
      simp only [lagCovariance, Function.iterate_zero_apply, mul_comm]
    rw [hsym]

lemma varianceLag_pos (μ : Measure Ω) (K : Chain E) (T : Ω → Ω) (f : E → Ω → ℝ) (k : ℕ) :
    varianceLag μ K T f (k+1)=
      ∑ e, ∑ e', lagCovariance μ T f e e' (k+1)*K.initial e'*
        ((K.prob^(k+1)) e' e-K.initial e) := by
  rw [varianceLag, Finset.sum_comm]
  apply Finset.sum_congr rfl; intro e he
  apply Finset.sum_congr rfl; intro e' he'
  ring

/-- The exact finite-horizon total-variance subtraction for the actual path law. -/
theorem withinVariance_div (μ : Measure Ω) (K : Chain E) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (f : E → Ω → ℝ) (hf : ∀ e, MemLp (f e) 2 μ)
    (n : ℕ) (hn : 1 ≤ n) :
    withinVariance μ K T f n/(n:ℝ) =
      (∑ e, K.initial e*lagCovariance μ T f e e 0)-
        (∑ e, ∑ e', K.initial e*K.initial e'*lagCovariance μ T f e e' 0)+
      2*∑ k ∈ Finset.range (n-1), (1-((k+1:ℕ):ℝ)/(n:ℝ))*
        ∑ e, ∑ e', lagCovariance μ T f e e' (k+1)*K.initial e'*
          ((K.prob^(k+1)) e' e-K.initial e) := by
  have hn0 : n≠0 := by omega
  have hnR : (n:ℝ)≠0 := Nat.cast_ne_zero.mpr hn0
  have hs : (∑ k ∈ Finset.range n, ((n:ℝ)-(k+1))*varianceLag μ K T f (k+1)) =
      ∑ k ∈ Finset.range (n-1), ((n:ℝ)-(k+1))*varianceLag μ K T f (k+1) := by
    obtain ⟨m,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn0
    simp [Finset.sum_range_succ]
  rw [withinVariance_finite μ K T hT f hf n, hs, add_div,
    mul_div_cancel_left₀ _ hnR, varianceLag_zero]
  congr 1
  rw [mul_div_assoc, Finset.sum_div, Finset.mul_sum, Finset.mul_sum]
  apply Finset.sum_congr rfl; intro k hk
  rw [varianceLag_pos]
  push_cast
  field_simp

end Environment

end Dobson.FiniteLineageVariance


namespace Dobson.Coverage1Lineage

open scoped BigOperators

/-- Rank-one outer product matrix `v wᵀ` on a finite type `V`. -/
def rankOne {V : Type*} (v w : V → ℝ) : Matrix V V ℝ :=
  Matrix.of fun i j => v i * w j

/-- Single-entry edge matrix `b · e_{i_e} e_{j_e}ᵀ` on `V`. -/
def edgeMat {V : Type*} [DecidableEq V] (i_e j_e : V) (b_e : ℝ) : Matrix V V ℝ :=
  Matrix.of fun i j => if i = i_e ∧ j = j_e then b_e else 0

/-- Bilinear scalar contraction `wᵀ A v = ∑_{i, j} w i * A i j * v j`. -/
def quadForm {V : Type*} [Fintype V] (w : V → ℝ) (A : Matrix V V ℝ) (v : V → ℝ) : ℝ :=
  ∑ i : V, ∑ j : V, w i * A i j * v j

/-- Stationary edge use frequency `c_e = b_e * w(i_e) * v(j_e)`. -/
def edgeFreq {V : Type*} (w v : V → ℝ) (i_e j_e : V) (b_e : ℝ) : ℝ :=
  b_e * w i_e * v j_e

/-- Reference-lineage lag-`(m + 1)` reuse probability `u_{e' → e}(m + 1)`:
starting right after edge `e'` lands in `i_{e'}`, moves `m` steps along `P` to `j_e` with
probability `(M^m)(j_e, i_{e'}) * w(j_e) / w(i_{e'})`, then takes edge `e` with probability
`b_e * w(i_e) / w(j_e)`. -/
noncomputable def edgeReuse {V : Type*} [Fintype V] [DecidableEq V]
    (M : Matrix V V ℝ) (w : V → ℝ) (i_e j_e i_e' : V) (b_e : ℝ) (m : ℕ) : ℝ :=
  (b_e * w i_e / w j_e) * ((M ^ m) j_e i_e' * w j_e / w i_e')

section MatrixReduction

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- `quadForm w (edgeMat i_e j_e b_e) v = edgeFreq w v i_e j_e b_e`. -/
theorem quadForm_edgeMat (w v : V → ℝ) (i_e j_e : V) (b_e : ℝ) :
    quadForm w (edgeMat i_e j_e b_e) v = edgeFreq w v i_e j_e b_e := by
  unfold quadForm edgeMat edgeFreq
  simp only [Matrix.of_apply]
  rw [Finset.sum_eq_single i_e]
  · rw [Finset.sum_eq_single j_e]
    · simp only [and_self, ↓reduceIte]
      ring
    · intro j _ hj; simp [hj]
    · intro hj; exact False.elim (hj (Finset.mem_univ j_e))
  · intro i _ hi
    exact Finset.sum_eq_zero fun j _ => by simp [hi]
  · intro hi; exact False.elim (hi (Finset.mem_univ i_e))

/-- **Part 1a (transverse projector power identity `(M - v wᵀ)^m (I - v wᵀ) = M^m (I - v wᵀ)`).** -/
theorem transMat_pow_mul_proj (M : Matrix V V ℝ) (v w : V → ℝ)
    (hv : M.mulVec v = v) (hw : M.vecMul w = w) (hwv : ∑ k : V, w k * v k = 1) :
    let P := rankOne v w
    let Q := (1 : Matrix V V ℝ) - P
    let T_perp := M - P
    ∀ m : ℕ, T_perp ^ m * Q = M ^ m * Q := by
  intro P Q T_perp
  -- Key rank-one identities: `P * P = P`, `M * P = P`, `P * M = P`, hence `P * Q = 0`,
  -- `M * Q = Q * M`? At least `P * (M^m * Q) = 0`!
  have hPP : P * P = P := by
    ext i j
    simp only [P, rankOne, Matrix.mul_apply, Matrix.of_apply]
    have hsum : ∑ k : V, v i * w k * (v k * w j) =
        v i * w j * ∑ k : V, w k * v k := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun k _ => by ring
    rw [hsum, hwv, mul_one]
  have hMP : M * P = P := by
    ext i j
    simp only [P, rankOne, Matrix.mul_apply, Matrix.of_apply]
    have hvi : ∑ k : V, M i k * v k = v i := congr_fun hv i
    have hsum : ∑ k : V, M i k * (v k * w j) = (∑ k : V, M i k * v k) * w j := by
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl fun k _ => by ring
    rw [hsum, hvi]
  have hPM : P * M = P := by
    ext i j
    simp only [P, rankOne, Matrix.mul_apply, Matrix.of_apply]
    have hwj : ∑ k : V, w k * M k j = w j := congr_fun hw j
    have hsum : ∑ k : V, v i * w k * M k j = v i * ∑ k : V, w k * M k j := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun k _ => by ring
    rw [hsum, hwj]
  have hPMpow : ∀ m : ℕ, P * M ^ m = P := by
    intro m
    induction m with
    | zero => simp
    | succ k ih =>
      rw [pow_succ', ← Matrix.mul_assoc, hPM, ih]
  have hPQ : P * Q = 0 := by
    dsimp [Q]
    rw [Matrix.mul_sub, Matrix.mul_one, hPP, sub_self]
  intro m
  induction m with
  | zero => simp
  | succ k ih =>
    rw [pow_succ', pow_succ', Matrix.mul_assoc, ih]
    dsimp [T_perp]
    rw [Matrix.sub_mul, Matrix.mul_assoc, ← Matrix.mul_assoc P (M ^ k) Q, hPMpow k, hPQ, sub_zero]

/-- **Part 1b (exact identity between lineage reuse covariance and matrix contraction).**
For edges `e : j_e → i_e` (`b_e`) and `e' : j_{e'} → i_{e'}` (`b_{e'}`) and strictly positive
reproductive values `w_k > 0`:
`c_{e'} * (u_{e' → e}(m + 1) − c_e) = wᵀ (B_e * T_⊥^m * Q * B_{e'}) v`. -/
theorem reuse_cov_eq_quadForm_trans (M : Matrix V V ℝ) (v w : V → ℝ)
    (hv : M.mulVec v = v) (hw : M.vecMul w = w) (hwv : ∑ k : V, w k * v k = 1)
    (hw_pos : ∀ k : V, 0 < w k)
    (i_e j_e i_e' j_e' : V) (b_e b_e' : ℝ) (m : ℕ) :
    let Q := (1 : Matrix V V ℝ) - rankOne v w
    let T_perp := M - rankOne v w
    let B_e := edgeMat i_e j_e b_e
    let B_e' := edgeMat i_e' j_e' b_e'
    edgeFreq w v i_e' j_e' b_e' *
        (edgeReuse M w i_e j_e i_e' b_e m - edgeFreq w v i_e j_e b_e) =
      quadForm w (B_e * (T_perp ^ m * Q) * B_e') v := by
  intro Q T_perp B_e B_e'
  rw [transMat_pow_mul_proj M v w hv hw hwv m]
  -- Evaluate `quadForm w (B_e * A * B_e') v = w(i_e) * b_e * A(j_e, i_e') * b_e' * v(j_e')`
  -- for any matrix `A`.
  have hquad_edge : ∀ A : Matrix V V ℝ,
      quadForm w (B_e * A * B_e') v =
        w i_e * b_e * A j_e i_e' * b_e' * v j_e' := by
    intro A
    have hentry : ∀ i j : V, (B_e * A * B_e') i j =
        if i = i_e ∧ j = j_e' then b_e * A j_e i_e' * b_e' else 0 := by
      intro i j
      simp only [B_e, B_e', edgeMat, Matrix.mul_apply, Matrix.of_apply]
      by_cases hij : i = i_e ∧ j = j_e'
      · rcases hij with ⟨rfl, rfl⟩
        simp only [and_self, ↓reduceIte]
        rw [Finset.sum_eq_single i_e']
        · simp only [true_and, ↓reduceIte]
          rw [Finset.sum_eq_single j_e]
          · simp only [↓reduceIte]
          · intro k _ hk; simp [hk]
          · intro hk; exact False.elim (hk (Finset.mem_univ j_e))
        · intro k _ hk; simp [hk]
        · intro hk; exact False.elim (hk (Finset.mem_univ i_e'))
      · rw [ite_eq_right hij]
        apply Finset.sum_eq_zero
        intro k _
        by_cases hj : j = j_e'
        · have hi : i ≠ i_e := fun hi => hij ⟨hi, hj⟩
          have h0 : ∑ x : V, (if i = i_e ∧ x = j_e then b_e else 0) * A x k = 0 :=
            Finset.sum_eq_zero fun x _ => by simp [hi]
          rw [h0, zero_mul]
        · simp [hj]
    unfold quadForm
    simp_rw [hentry]
    rw [Finset.sum_eq_single i_e]
    · rw [Finset.sum_eq_single j_e']
      · simp only [and_self, ↓reduceIte]
        ring
      · intro j _ hj; simp [hj]
      · intro hj; exact False.elim (hj (Finset.mem_univ j_e'))
    · intro i _ hi
      exact Finset.sum_eq_zero fun j _ => by simp [hi]
    · intro hi; exact False.elim (hi (Finset.mem_univ i_e))
  rw [hquad_edge (M ^ m * Q)]
  -- Compute `(M^m * Q)(j_e, i_e') = (M^m)(j_e, i_e') - v(j_e) * w(i_e')` using `M^m v = v`.
  have hMpow_v : ∀ k : ℕ, (M ^ k).mulVec v = v := by
    intro k
    induction k with
    | zero => simp
    | succ r ihr =>
      rw [pow_succ', ← Matrix.mulVec_mulVec, ihr, hv]
  have hMQ_entry : (M ^ m * Q) j_e i_e' = (M ^ m) j_e i_e' - v j_e * w i_e' := by
    dsimp [Q]
    rw [Matrix.mul_sub, Matrix.mul_one]
    have hMP : (M ^ m * rankOne v w) j_e i_e' = v j_e * w i_e' := by
      simp only [rankOne, Matrix.mul_apply, Matrix.of_apply]
      have hvj : ∑ k : V, (M ^ m) j_e k * v k = v j_e := congr_fun (hMpow_v m) j_e
      have hsum : ∑ k : V, (M ^ m) j_e k * (v k * w i_e') =
          (∑ k : V, (M ^ m) j_e k * v k) * w i_e' := by
        rw [Finset.sum_mul]
        refine Finset.sum_congr rfl fun k _ => by ring
      rw [hsum, hvj]
    simp [hMP]
  rw [hMQ_entry]
  unfold edgeFreq edgeReuse
  have hwj_ne : w j_e ≠ 0 := (hw_pos j_e).ne'
  have hwi'_ne : w i_e' ≠ 0 := (hw_pos i_e').ne'
  field_simp [hwj_ne, hwi'_ne]

end MatrixReduction

section CesaroAndVarianceIdentity

/-- **Helper (Cesàro convergence of weighted partial sums `∑_{m=0}^{T-2} (1 - (m+1)/T) a_m → ∑' m, a_m`
for any absolutely summable sequence `a` with geometric bound `|a m| ≤ C * β^m`, `0 ≤ β < 1`).** -/
theorem tendsto_cesaro_lag_sum (a : ℕ → ℝ) (C β : ℝ) (_hC : 0 ≤ C) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (ha : ∀ m : ℕ, |a m| ≤ C * β ^ m) :
    Summable a ∧
      Filter.Tendsto
        (fun T : ℕ => ∑ m ∈ Finset.range (T - 1), (1 - ((m + 1 : ℕ) : ℝ) / (T : ℝ)) * a m)
        Filter.atTop (nhds (∑' m : ℕ, a m)) := by
  have hgeom : Summable (fun m : ℕ => C * β ^ m) :=
    (summable_geometric_of_lt_one hβ0 hβ1).mul_left C
  have hsum_abs : Summable (fun m : ℕ => ‖a m‖) :=
    Summable.of_nonneg_of_le (fun _ => norm_nonneg _)
      (fun m => by simpa [Real.norm_eq_abs] using ha m) hgeom
  have hsum : Summable a := hsum_abs.of_norm
  refine ⟨hsum, ?_⟩
  -- Split `(1 - (m+1)/T) * a m = a m - (1/T) * ((m+1) * a m)`.
  -- Note `∑' m, (m+1) * |a m| < ∞` since `(m+1) * β^m` is summable!
  have hlin_summable : Summable (fun m : ℕ => ((m + 1 : ℕ) : ℝ) * |a m|) := by
    have hnormβ : ‖β‖ < 1 := by rw [Real.norm_eq_abs, abs_of_nonneg hβ0]; exact hβ1
    have hm1 : Summable (fun m : ℕ => (m : ℝ) * β ^ m) := by
      have h := summable_pow_mul_geometric_of_norm_lt_one 1 hnormβ
      simpa using h
    have hg2 : Summable (fun m : ℕ => C * ((m : ℝ) * β ^ m + β ^ m)) :=
      (hm1.add (summable_geometric_of_lt_one hβ0 hβ1)).mul_left C
    refine Summable.of_nonneg_of_le (fun m => by positivity) (fun m => ?_) hg2
    calc ((m + 1 : ℕ) : ℝ) * |a m|
        ≤ ((m + 1 : ℕ) : ℝ) * (C * β ^ m) :=
          mul_le_mul_of_nonneg_left (ha m) (Nat.cast_nonneg _)
      _ = C * ((m : ℝ) * β ^ m + β ^ m) := by push_cast; ring
  let S_lin : ℝ := ∑' m : ℕ, ((m + 1 : ℕ) : ℝ) * |a m|
  have hpartial : Filter.Tendsto (fun T : ℕ => ∑ m ∈ Finset.range (T - 1), a m)
      Filter.atTop (nhds (∑' m : ℕ, a m)) := by
    have hsub : Filter.Tendsto (fun T : ℕ => T - 1) Filter.atTop Filter.atTop :=
      Filter.tendsto_atTop_atTop.mpr fun b => ⟨b + 1, fun n hn => by omega⟩
    exact hsum.hasSum.tendsto_sum_nat.comp hsub
  have hrem_zero : Filter.Tendsto
      (fun T : ℕ => ∑ m ∈ Finset.range (T - 1), (((m + 1 : ℕ) : ℝ) / (T : ℝ)) * a m)
      Filter.atTop (nhds 0) := by
    apply squeeze_zero_norm (a := fun T : ℕ => (1 / (T : ℝ)) * S_lin)
    · intro T
      rw [Real.norm_eq_abs]
      by_cases hT0 : T = 0
      · subst hT0; simp
      · have hTpos : (0 : ℝ) < (T : ℝ) := Nat.cast_pos.mpr (Nat.pos_of_ne_zero hT0)
        calc |∑ m ∈ Finset.range (T - 1), (((m + 1 : ℕ) : ℝ) / (T : ℝ)) * a m|
            ≤ ∑ m ∈ Finset.range (T - 1), |(((m + 1 : ℕ) : ℝ) / (T : ℝ)) * a m| :=
              Finset.abs_sum_le_sum_abs _ _
          _ = (1 / (T : ℝ)) * ∑ m ∈ Finset.range (T - 1), ((m + 1 : ℕ) : ℝ) * |a m| := by
              rw [Finset.mul_sum]
              refine Finset.sum_congr rfl fun m _ => ?_
              rw [abs_mul, abs_of_nonneg (div_nonneg (Nat.cast_nonneg _) hTpos.le)]
              ring
          _ ≤ (1 / (T : ℝ)) * S_lin := by
              apply mul_le_mul_of_nonneg_left _ (by positivity)
              exact hlin_summable.sum_le_tsum (Finset.range (T - 1)) (fun m _ => by positivity)
    · have hinv : Filter.Tendsto (fun T : ℕ => (1 : ℝ) / (T : ℝ)) Filter.atTop (nhds 0) :=
        tendsto_one_div_atTop_nhds_zero_nat
      have hmul := hinv.mul_const S_lin
      rwa [zero_mul] at hmul
  have hdiff := hpartial.sub hrem_zero
  rw [sub_zero] at hdiff
  refine hdiff.congr fun T => ?_
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun m _ => by ring

/-- Private conditional variance/remainder helper. Its hypotheses supply
the finite-horizon variance expansion, covariance and mixing bounds, and
the growth remainder. The public actual-path theorem below derives the
variance expansion; it still takes the growth remainder explicitly. -/
private theorem lineage_variance_identity_and_remainder_of_total_variance {E : Type*} [Fintype E]
    (c : E → ℝ) (hc_nn : ∀ e, 0 ≤ c e)
    (γ : E → E → ℕ → ℝ) (u : E → E → ℕ → ℝ)
    (W : ℕ → ℝ)
    (B_γ K_u β : ℝ) (hB : 0 ≤ B_γ) (hKu : 0 ≤ K_u) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (h_totvar : ∀ T : ℕ, 1 ≤ T →
      W T / (T : ℝ) =
        (∑ e : E, c e * γ e e 0) - (∑ e : E, ∑ e' : E, c e * c e' * γ e e' 0) +
          2 * ∑ m ∈ Finset.range (T - 1),
            (1 - ((m + 1 : ℕ) : ℝ) / (T : ℝ)) *
              ∑ e : E, ∑ e' : E, γ e e' (m + 1) * c e' * (u e' e (m + 1) - c e))
    (h_cov : ∀ e e' : E, ∀ ℓ : ℕ, |γ e e' ℓ| ≤ B_γ)
    (h_mix : ∀ e e' : E, ∀ m : ℕ, |u e' e (m + 1) - c e| ≤ K_u * β ^ m)
    (logLam : ℝ) (r : ℝ → ℝ) (σ₀ K_rem : ℝ) (p : ℕ)
    (h_rem : ∀ σ : ℝ, |σ| ≤ σ₀ →
      |logLam - r σ -
          ((1 / 2 : ℝ) * (∑ e : E, ∑ e' : E, c e * c e' * γ e e' 0) -
            ∑' m : ℕ, ∑ e : E, ∑ e' : E,
              γ e e' (m + 1) * c e' * (u e' e (m + 1) - c e)) * σ ^ 2| ≤
        K_rem * |σ| ^ p) :
    let a : ℕ → ℝ := fun m =>
      ∑ e : E, ∑ e' : E, γ e e' (m + 1) * c e' * (u e' e (m + 1) - c e)
    let J₂ : ℝ := (1 / 2 : ℝ) * ∑ e : E, c e * γ e e 0
    let V₂ : ℝ := (∑ e : E, c e * γ e e 0) - (∑ e : E, ∑ e' : E, c e * c e' * γ e e' 0) +
      2 * ∑' m : ℕ, a m
    let C_pp : ℝ := (1 / 2 : ℝ) * (∑ e : E, ∑ e' : E, c e * c e' * γ e e' 0) - ∑' m : ℕ, a m
    Summable a ∧
      Filter.Tendsto (fun T : ℕ => W T / (T : ℝ)) Filter.atTop (nhds V₂) ∧
      J₂ - (1 / 2 : ℝ) * V₂ = C_pp ∧
      (∀ σ : ℝ, |σ| ≤ σ₀ →
        |logLam - r σ - (σ ^ 2 * J₂ - (1 / 2 : ℝ) * (σ ^ 2 * V₂))| ≤ K_rem * |σ| ^ p) := by
  intro a J₂ V₂ C_pp
  -- Geometric bound on `|a m|`
  let C_sum : ℝ := (Fintype.card E : ℝ) * (∑ e' : E, c e') * B_γ * K_u
  have hC_sum_nn : 0 ≤ C_sum := by
    dsimp [C_sum]
    exact mul_nonneg (mul_nonneg (mul_nonneg (Nat.cast_nonneg _)
      (Finset.sum_nonneg fun e' _ => hc_nn e')) hB) hKu
  have ha_bound : ∀ m : ℕ, |a m| ≤ C_sum * β ^ m := by
    intro m
    dsimp [a]
    calc |∑ e : E, ∑ e' : E, γ e e' (m + 1) * c e' * (u e' e (m + 1) - c e)|
        ≤ ∑ e : E, |∑ e' : E, γ e e' (m + 1) * c e' * (u e' e (m + 1) - c e)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ e : E, ∑ e' : E, |γ e e' (m + 1) * c e' * (u e' e (m + 1) - c e)| :=
          Finset.sum_le_sum fun e _ => Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _e : E, ∑ e' : E, B_γ * c e' * (K_u * β ^ m) := by
          refine Finset.sum_le_sum fun e _ => Finset.sum_le_sum fun e' _ => ?_
          rw [abs_mul, abs_mul, abs_of_nonneg (hc_nn e')]
          have h1 : |γ e e' (m + 1)| * c e' ≤ B_γ * c e' :=
            mul_le_mul_of_nonneg_right (h_cov e e' (m + 1)) (hc_nn e')
          exact mul_le_mul h1 (h_mix e e' m) (abs_nonneg _) (mul_nonneg hB (hc_nn e'))
      _ = C_sum * β ^ m := by
          simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
          dsimp [C_sum]
          have hpull : ∑ e' : E, B_γ * c e' * (K_u * β ^ m) =
              (∑ e' : E, c e') * (B_γ * K_u * β ^ m) := by
            rw [Finset.sum_mul]
            refine Finset.sum_congr rfl fun e' _ => by ring
          rw [hpull]
          ring
  obtain ⟨hsum_a, hcesaro⟩ := tendsto_cesaro_lag_sum a C_sum β hC_sum_nn hβ0 hβ1 ha_bound
  have htend_W : Filter.Tendsto (fun T : ℕ => W T / (T : ℝ)) Filter.atTop (nhds V₂) := by
    have hbase : Filter.Tendsto
        (fun T : ℕ => (∑ e : E, c e * γ e e 0) - (∑ e : E, ∑ e' : E, c e * c e' * γ e e' 0) +
          2 * ∑ m ∈ Finset.range (T - 1), (1 - ((m + 1 : ℕ) : ℝ) / (T : ℝ)) * a m)
        Filter.atTop (nhds V₂) := by
      dsimp [V₂]
      exact tendsto_const_nhds.add (hcesaro.const_mul 2)
    refine hbase.congr' ?_
    filter_upwards [Filter.eventually_ge_atTop 1] with T hT
    exact (h_totvar T hT).symm
  have hident : J₂ - (1 / 2 : ℝ) * V₂ = C_pp := by
    dsimp [J₂, V₂, C_pp]
    ring
  refine ⟨hsum_a, htend_W, hident, fun σ hσ => ?_⟩
  have hrew : σ ^ 2 * J₂ - (1 / 2 : ℝ) * (σ ^ 2 * V₂) = C_pp * σ ^ 2 := by
    linear_combination (σ ^ 2) * hident
  rw [hrew]
  exact h_rem σ hσ

/-!
## Actual finite lineage variance — 2026-11-16 (leanp1)

The public theorem now uses an actual stationary finite edge-chain law and
actual stationary L² environment rewards. Its within-environment variance W,
lag covariance γ and reuse probabilities u are defined from those objects.
The finite path law is normalized and nonnegative; its one- and two-time
marginals, path variance expansion, environment integration and lag-pair
counts are proved. `Chain.ofKernel` constructs this law from the existing
reference kernel, including parallel labeled edges. No total-variance
identity is assumed. The growth perturbation remainder remains explicit
project-specific debt (`h_rem`, row38); this is not full manuscript coverage.
-/

/-- Lineage variance and remainder transfer for the actual finite path law.
The separately owed growth expansion is retained as `h_rem`. -/
theorem lineage_variance_identity_and_remainder {E : Type*} [Fintype E] [DecidableEq E]
    {Ω : Type*} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (K : Dobson.FiniteLineageVariance.Chain E) (T : Ω ≃ᵐ Ω)
    (hT : MeasureTheory.MeasurePreserving T μ μ) (f : E → Ω → ℝ)
    (hf : ∀ e, MeasureTheory.MemLp (f e) 2 μ)
    (B_γ K_u β : ℝ) (hB : 0 ≤ B_γ) (hKu : 0 ≤ K_u) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (h_cov : ∀ e e' : E, ∀ ℓ : ℕ, |(Dobson.FiniteLineageVariance.lagCovariance μ T f) e e' ℓ| ≤ B_γ)
    (h_mix : ∀ e e' : E, ∀ m : ℕ, |(fun e' e ℓ => (K.prob^ℓ) e' e) e' e (m + 1) - K.initial e| ≤ K_u * β ^ m)
    (logLam : ℝ) (r : ℝ → ℝ) (σ₀ K_rem : ℝ) (p : ℕ)
    (h_rem : ∀ σ : ℝ, |σ| ≤ σ₀ →
      |logLam - r σ -
          ((1 / 2 : ℝ) * (∑ e : E, ∑ e' : E, K.initial e * K.initial e' * (Dobson.FiniteLineageVariance.lagCovariance μ T f) e e' 0) -
            ∑' m : ℕ, ∑ e : E, ∑ e' : E,
              (Dobson.FiniteLineageVariance.lagCovariance μ T f) e e' (m + 1) * K.initial e' * ((fun e' e ℓ => (K.prob^ℓ) e' e) e' e (m + 1) - K.initial e)) * σ ^ 2| ≤
        K_rem * |σ| ^ p) :
    let a : ℕ → ℝ := fun m =>
      ∑ e : E, ∑ e' : E, (Dobson.FiniteLineageVariance.lagCovariance μ T f) e e' (m + 1) * K.initial e' * ((fun e' e ℓ => (K.prob^ℓ) e' e) e' e (m + 1) - K.initial e)
    let J₂ : ℝ := (1 / 2 : ℝ) * ∑ e : E, K.initial e * (Dobson.FiniteLineageVariance.lagCovariance μ T f) e e 0
    let V₂ : ℝ := (∑ e : E, K.initial e * (Dobson.FiniteLineageVariance.lagCovariance μ T f) e e 0) - (∑ e : E, ∑ e' : E, K.initial e * K.initial e' * (Dobson.FiniteLineageVariance.lagCovariance μ T f) e e' 0) +
      2 * ∑' m : ℕ, a m
    let C_pp : ℝ := (1 / 2 : ℝ) * (∑ e : E, ∑ e' : E, K.initial e * K.initial e' * (Dobson.FiniteLineageVariance.lagCovariance μ T f) e e' 0) - ∑' m : ℕ, a m
    Summable a ∧
      Filter.Tendsto (fun N : ℕ => (Dobson.FiniteLineageVariance.withinVariance μ K T f) N / (N : ℝ)) Filter.atTop (nhds V₂) ∧
      J₂ - (1 / 2 : ℝ) * V₂ = C_pp ∧
      (∀ σ : ℝ, |σ| ≤ σ₀ →
        |logLam - r σ - (σ ^ 2 * J₂ - (1 / 2 : ℝ) * (σ ^ 2 * V₂))| ≤ K_rem * |σ| ^ p) := by
  exact lineage_variance_identity_and_remainder_of_total_variance K.initial K.initial_nonneg
    (Dobson.FiniteLineageVariance.lagCovariance μ T f) (fun e' e ℓ => (K.prob^ℓ) e' e)
    (Dobson.FiniteLineageVariance.withinVariance μ K T f)
    B_γ K_u β hB hKu hβ0 hβ1
    (Dobson.FiniteLineageVariance.withinVariance_div μ K T hT f hf)
    h_cov h_mix logLam r σ₀ K_rem p h_rem

end CesaroAndVarianceIdentity

end Dobson.Coverage1Lineage

#print axioms Dobson.Coverage1Lineage.quadForm_edgeMat
#print axioms Dobson.Coverage1Lineage.transMat_pow_mul_proj
#print axioms Dobson.Coverage1Lineage.reuse_cov_eq_quadForm_trans
#print axioms Dobson.Coverage1Lineage.tendsto_cesaro_lag_sum
#print axioms Dobson.Coverage1Lineage.lineage_variance_identity_and_remainder
