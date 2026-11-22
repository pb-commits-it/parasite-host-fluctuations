import DobsonProofs.Coverage6ExactZeroRadius
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Analysis.MeanInequalitiesPow
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Function.L2Space

/-! # Part B: finite Jacobi trace bounds and conditional series divergence

Coverage correction, 2026-11-22: this module is uncredited in the version-2
inventory; `prop:exact-mem-zero-radius` remains partial-P1. The independent
fidelity audit (2026-11-16) did not confirm closure of `part1:67` or `part1:68`.

Finite trace domination and host moment-tail bounds are proved. The final
series-divergence theorem assumes integrability of the finite traces and
full-sequence convergence of their normalized expectations to `μ`.
Construction of the spectral law, identification with physical Taylor
coefficients, and analytic inversion to the threshold remain outside credit. -/

namespace Dobson.P1bB_ExactMemZeroRadius

open Matrix Finset Set Filter MeasureTheory
open scoped BigOperators Topology

theorem matrix_pow_nonneg {ι : Type*} [Fintype ι] [DecidableEq ι]
    (A : Matrix ι ι ℝ) (hA : ∀ i j, 0 ≤ A i j) (n : ℕ) :
    ∀ i j, 0 ≤ (A ^ n) i j := by
  induction n with
  | zero =>
    intro i j
    simp only [pow_zero, Matrix.one_apply]
    split_ifs <;> norm_num
  | succ n ih =>
    intro i j
    rw [pow_succ, Matrix.mul_apply]
    exact Finset.sum_nonneg fun k _ => mul_nonneg (ih i k) (hA k j)

theorem matrix_diagonal_pow_le {ι : Type*} [Fintype ι] [DecidableEq ι]
    (A : Matrix ι ι ℝ) (hA : ∀ i j, 0 ≤ A i j) (n : ℕ) (i : ι) :
    (A i i) ^ n ≤ (A ^ n) i i := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [pow_succ, pow_succ, Matrix.mul_apply]
    calc
      (A i i) ^ n * A i i ≤ (A ^ n) i i * A i i := mul_le_mul_of_nonneg_right ih (hA i i)
      _ ≤ ∑ j, (A ^ n) i j * A j i := Finset.single_le_sum
        (fun j _ => mul_nonneg (matrix_pow_nonneg A hA n i j) (hA j i)) (mem_univ i)

theorem sum_pow_le_pow_sum {ι : Type*} (S : Finset ι) (f : ι → ℝ)
    (hf : ∀ i ∈ S, 0 ≤ f i) (k : ℕ) (hk : 0 < k) :
    (∑ i ∈ S, (f i) ^ k) ≤ (∑ i ∈ S, f i) ^ k := by
  classical
  induction S using Finset.induction_on with
  | empty => simp [hk.ne']
  | @insert i S hi ih =>
    rw [Finset.sum_insert hi, Finset.sum_insert hi]
    have hfi := hf i (mem_insert_self i S)
    have hfS : ∀ j ∈ S, 0 ≤ f j := fun j hj => hf j (mem_insert_of_mem hj)
    have hb := pow_add_pow_le hfi (Finset.sum_nonneg hfS) hk.ne'
    have ht := ih hfS
    linarith

/-- Every oriented repeated-edge walk contributes to the even trace `tr(A^(2k))`. -/
theorem edge_power_sum_le_trace {ι : Type*} [Fintype ι] [DecidableEq ι]
    (A : Matrix ι ι ℝ) (hA : ∀ i j, 0 ≤ A i j) (k : ℕ) (hk : 0 < k) :
    (∑ i, ∑ j, (A i j * A j i) ^ k) ≤ Matrix.trace (A ^ (2 * k)) := by
  rw [Matrix.trace, pow_mul]
  apply Finset.sum_le_sum
  intro i _
  have hsum : (∑ j, (A i j * A j i) ^ k) ≤ (∑ j, A i j * A j i) ^ k :=
    sum_pow_le_pow_sum univ (fun j => A i j * A j i)
      (fun j _ => mul_nonneg (hA i j) (hA j i)) k hk
  apply hsum.trans
  simpa only [pow_two, Matrix.mul_apply, Matrix.diag_apply] using
    matrix_diagonal_pow_le (A ^ 2) (matrix_pow_nonneg A hA 2) k i

/-- Finite symmetric tridiagonal Jacobi matrix on `N + 1` vertices with off-diagonals `√(h_i)`. -/
noncomputable def jacobi (N : ℕ) (h : ℕ → ℝ) : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ :=
  fun i j => if i.val + 1 = j.val then Real.sqrt (h i.val)
    else if j.val + 1 = i.val then Real.sqrt (h j.val) else 0

theorem jacobi_nonneg (N : ℕ) (h : ℕ → ℝ) :
    ∀ i j, 0 ≤ jacobi N h i j := by
  intro i j
  simp only [jacobi]
  split_ifs <;> positivity

theorem jacobi_forward (N : ℕ) (h : ℕ → ℝ) (i : Fin N) :
    jacobi N h i.castSucc i.succ = Real.sqrt (h i.val) := by
  simp [jacobi]

theorem jacobi_backward (N : ℕ) (h : ℕ → ℝ) (i : Fin N) :
    jacobi N h i.succ i.castSucc = Real.sqrt (h i.val) := by
  simp [jacobi, show i.val + 1 + 1 ≠ i.val by omega]

def orientedEdge (N : ℕ) : Fin N × Bool → Fin (N + 1) × Fin (N + 1)
  | (i, false) => (i.castSucc, i.succ)
  | (i, true) => (i.succ, i.castSucc)

theorem orientedEdge_injective (N : ℕ) : Function.Injective (orientedEdge N) := by
  intro ⟨i, b⟩ ⟨j, c⟩ he
  cases b <;> cases c <;> simp only [orientedEdge, Prod.mk.injEq] at he
  · have hij : i = j := Fin.ext (congrArg (fun x : Fin (N + 1) => x.val) he.1)
    simp [hij]
  · have h1 := congrArg Fin.val he.1
    have h2 := congrArg Fin.val he.2
    simp only [Fin.val_castSucc, Fin.val_succ] at h1 h2
    omega
  · have h1 := congrArg Fin.val he.1
    have h2 := congrArg Fin.val he.2
    simp only [Fin.val_castSucc, Fin.val_succ] at h1 h2
    omega
  · have hij : i = j := Fin.ext (congrArg (fun x : Fin (N + 1) => x.val) he.2)
    simp [hij]

/-- Manuscript trace lower bound `(N / (N + 1))` pathwise ingredient (`eq:exact-mem-trace`):
`2 * ∑_{i < N} h_i^k ≤ tr((jacobi N h)^(2k))`. -/
theorem jacobi_trace_lower (N : ℕ) (h : ℕ → ℝ) (hpos : ∀ i, 0 ≤ h i)
    (k : ℕ) (hk : 0 < k) :
    2 * (∑ i : Fin N, (h i.val) ^ k) ≤ Matrix.trace ((jacobi N h) ^ (2 * k)) := by
  have hb := edge_power_sum_le_trace (jacobi N h) (jacobi_nonneg N h) k hk
  have he : ∀ q : Fin N × Bool,
      (jacobi N h (orientedEdge N q).1 (orientedEdge N q).2 *
       jacobi N h (orientedEdge N q).2 (orientedEdge N q).1) ^ k = (h q.1.val) ^ k := by
    rintro ⟨i, b⟩
    cases b <;> simp only [orientedEdge, jacobi_forward, jacobi_backward]
      <;> rw [← sq, Real.sq_sqrt (hpos _)]
  have hs : (∑ q : Fin N × Bool, (h q.1.val) ^ k) ≤
      ∑ q : Fin (N + 1) × Fin (N + 1), (jacobi N h q.1 q.2 * jacobi N h q.2 q.1) ^ k := by
    apply Finset.sum_le_sum_of_injOn (orientedEdge N) (orientedEdge_injective N).injOn
      (by simp)
    · intro q _
      rw [he]
    · intro q _ _
      exact pow_nonneg (mul_nonneg (jacobi_nonneg N h _ _) (jacobi_nonneg N h _ _)) k
  rw [Fintype.sum_prod_type] at hs
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_bool, nsmul_eq_mul,
    Nat.cast_ofNat] at hs
  rw [← Finset.mul_sum] at hs
  rw [Fintype.sum_prod_type] at hs
  exact hs.trans hb

private theorem integral_shift {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T P P) (f : Ω → ℝ) (n : ℕ) :
    (∫ ω, f ((T : Ω → Ω)^[n] ω) ∂P) = ∫ ω, f ω ∂P := by
  induction n with
  | zero => simp
  | succ n ih =>
    simpa only [Function.iterate_succ_apply] using
      (hT.integral_comp' (fun ω => f ((T : Ω → Ω)^[n] ω))).trans ih

/-- Expected finite-volume trace bound:
`(N / (N + 1)) E[h^k] ≤ (1 / (2(N+1))) E[tr J^(2k)]`, under the displayed
integrability assumptions. No infinite-volume spectral law is constructed. -/
theorem expected_jacobi_moment_lower {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T P P)
    (h : Ω → ℝ) (hpos : ∀ ω, 0 ≤ h ω) (N k : ℕ) (hk : 0 < k)
    (hpow : Integrable (fun ω => (h ω) ^ k) P)
    (htr : Integrable (fun ω =>
      Matrix.trace ((jacobi N (fun j => h ((T : Ω → Ω)^[j] ω))) ^ (2 * k))) P) :
    ((N : ℝ) / ((N : ℝ) + 1)) * (∫ ω, (h ω) ^ k ∂P) ≤
      (∫ ω, Matrix.trace ((jacobi N (fun j => h ((T : Ω → Ω)^[j] ω))) ^ (2 * k)) ∂P) /
        (2 * ((N : ℝ) + 1)) := by
  have hshift (i : Fin N) : Integrable (fun ω => (h ((T : Ω → Ω)^[i.val] ω)) ^ k) P :=
    (hT.iterate i.val).integrable_comp_of_integrable hpow
  have hsum_int : Integrable (fun ω => 2 * ∑ i : Fin N, (h ((T : Ω → Ω)^[i.val] ω)) ^ k) P :=
    (integrable_finsetSum _ (fun i _ => hshift i)).const_mul 2
  have hmono := integral_mono hsum_int htr (fun ω =>
    jacobi_trace_lower N (fun j => h ((T : Ω → Ω)^[j] ω)) (fun j => hpos _) k hk)
  rw [integral_const_mul, integral_finsetSum _ (fun i _ => hshift i)] at hmono
  have heq : ∀ i : Fin N, (∫ ω, (h ((T : Ω → Ω)^[i.val] ω)) ^ k ∂P) = ∫ ω, (h ω) ^ k ∂P :=
    fun i => integral_shift P T hT (fun ω => (h ω) ^ k) i.val
  simp_rw [heq] at hmono
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hmono
  have hden : 0 < 2 * ((N : ℝ) + 1) := by positivity
  rw [le_div_iff₀ hden]
  calc
    (N : ℝ) / ((N : ℝ) + 1) * (∫ ω, (h ω) ^ k ∂P) * (2 * ((N : ℝ) + 1))
        = 2 * ((N : ℝ) * ∫ ω, (h ω) ^ k ∂P) := by
          have hN1 : (N : ℝ) + 1 ≠ 0 := by positivity
          field_simp [hN1]
    _ ≤ ∫ ω, Matrix.trace ((jacobi N (fun j => h ((T : Ω → Ω)^[j] ω))) ^ (2 * k)) ∂P := hmono

/-- Full-sequence convergence of the normalized expected finite Jacobi
traces to the supplied `μ k` implies `E[h^k] ≤ μ k`. Trace integrability
and that convergence are hypotheses, not a constructed spectral limit. -/
theorem spectral_moment_ge_host_moment {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T P P)
    (h : Ω → ℝ) (hpos : ∀ ω, 0 ≤ h ω) (μ : ℕ → ℝ) (k : ℕ) (hk : 0 < k)
    (hpow : Integrable (fun ω => (h ω) ^ k) P)
    (htr : ∀ N, Integrable (fun ω =>
      Matrix.trace ((jacobi N (fun j => h ((T : Ω → Ω)^[j] ω))) ^ (2 * k))) P)
    (hlim : Tendsto (fun N : ℕ =>
      (∫ ω, Matrix.trace ((jacobi N (fun j => h ((T : Ω → Ω)^[j] ω))) ^ (2 * k)) ∂P) /
        (2 * ((N : ℝ) + 1))) atTop (𝓝 (μ k))) :
    (∫ ω, (h ω) ^ k ∂P) ≤ μ k := by
  have hrat : Tendsto (fun N : ℕ => (N : ℝ) / ((N : ℝ) + 1)) atTop (𝓝 1) := by
    have hinv : Tendsto (fun N : ℕ => 1 - 1 / ((N : ℝ) + 1)) atTop (𝓝 (1 - 0)) :=
      tendsto_const_nhds.sub
        (tendsto_const_nhds.div_atTop
          (tendsto_natCast_atTop_atTop.atTop_add tendsto_const_nhds))
    simp only [sub_zero] at hinv
    refine hinv.congr' ?_
    filter_upwards with N
    have hN1 : (N : ℝ) + 1 ≠ 0 := by positivity
    field_simp [hN1]
    ring
  have hlow_lim : Tendsto (fun N : ℕ =>
      ((N : ℝ) / ((N : ℝ) + 1)) * (∫ ω, (h ω) ^ k ∂P)) atTop (𝓝 (∫ ω, (h ω) ^ k ∂P)) := by
    simpa only [one_mul] using hrat.mul_const (∫ ω, (h ω) ^ k ∂P)
  exact le_of_tendsto_of_tendsto hlow_lim hlim (Eventually.of_forall fun N =>
    expected_jacobi_moment_lower P T hT h hpos N k hk hpow (htr N))

/-- Unbounded-above nonnegative host Markov tail inequality: if `h ≥ 0` is unbounded above
(`∀ M > 0, P {ω | M < h ω} ≠ 0`) with finite moments, then for every `M > 0` there exists
`p > 0` (`p = P.real {ω | M < h ω}`) such that `p * M^(n + 1) ≤ E[h^(n + 1)]` for all `n`. -/
lemma unbounded_host_moment_tail {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P]
    (h : Ω → ℝ) (hmeas : Measurable h) (hpos : ∀ ω, 0 ≤ h ω)
    (hunb : ∀ M : ℝ, 0 < M → P {ω | M < h ω} ≠ 0)
    (hmom : ∀ k : ℕ, 0 < k → Integrable (fun ω => (h ω) ^ k) P) :
    ∀ M : ℝ, 0 < M → ∃ p : ℝ, 0 < p ∧
      ∀ n : ℕ, p * M ^ (n + 1) ≤ ∫ ω, (h ω) ^ (n + 1) ∂P := by
  intro M hM
  let S : Set Ω := {ω | M < h ω}
  have hS_meas : MeasurableSet S := measurableSet_lt measurable_const hmeas
  have hS_ne : P S ≠ 0 := hunb M hM
  have hS_lt_top : P S < ⊤ := measure_lt_top P S
  have hp_pos : 0 < P.real S := ENNReal.toReal_pos hS_ne hS_lt_top.ne
  refine ⟨P.real S, hp_pos, ?_⟩
  intro n
  have hk : 0 < n + 1 := Nat.succ_pos n
  have hint := hmom (n + 1) hk
  have hind_int : Integrable (S.indicator (fun _ : Ω => M ^ (n + 1))) P :=
    (integrable_const (M ^ (n + 1))).indicator hS_meas
  have hle : ∀ ω, S.indicator (fun _ : Ω => M ^ (n + 1)) ω ≤ (h ω) ^ (n + 1) := by
    intro ω
    by_cases hω : ω ∈ S
    · simp only [Set.indicator_of_mem hω]
      exact pow_le_pow_left₀ hM.le (le_of_lt hω) (n + 1)
    · simp only [Set.indicator_of_notMem hω]
      exact pow_nonneg (hpos ω) (n + 1)
  have hint_le := integral_mono hind_int hint hle
  rw [integral_indicator_const _ hS_meas] at hint_le
  simpa only [smul_eq_mul] using hint_le

/-- Conditional moment-tail bound for `μ`: finite trace domination and
unbounded host support give `htail` when integrability of the traces and
full-sequence convergence of their normalized expectations are supplied. -/
lemma spectral_tail_of_unbounded_host {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T P P)
    (h : Ω → ℝ) (hmeas : Measurable h) (hpos : ∀ ω, 0 ≤ h ω)
    (hunb : ∀ M : ℝ, 0 < M → P {ω | M < h ω} ≠ 0)
    (hmom : ∀ k : ℕ, 0 < k → Integrable (fun ω => (h ω) ^ k) P)
    (μ : ℕ → ℝ)
    (htr : ∀ k : ℕ, 0 < k → ∀ N : ℕ, Integrable (fun ω =>
      Matrix.trace ((jacobi N (fun j => h ((T : Ω → Ω)^[j] ω))) ^ (2 * k))) P)
    (hlim : ∀ k : ℕ, 0 < k → Tendsto (fun N : ℕ =>
      (∫ ω, Matrix.trace ((jacobi N (fun j => h ((T : Ω → Ω)^[j] ω))) ^ (2 * k)) ∂P) /
        (2 * ((N : ℝ) + 1))) atTop (𝓝 (μ k))) :
    ∀ M : ℝ, 0 < M → ∃ p : ℝ, 0 < p ∧
      ∀ n : ℕ, p * M ^ (n + 1) ≤ μ (n + 1) := by
  intro M hM
  obtain ⟨p, hp, hhost⟩ := unbounded_host_moment_tail P h hmeas hpos hunb hmom M hM
  refine ⟨p, hp, fun n => ?_⟩
  exact (hhost n).trans (spectral_moment_ge_host_moment P T hT h hpos μ (n + 1)
    (Nat.succ_pos n) (hmom (n + 1) (Nat.succ_pos n))
    (htr (n + 1) (Nat.succ_pos n)) (hlim (n + 1) (Nat.succ_pos n)))

/-- Conditional divergence of the numerical logarithm series formed from
`μ`, at every nonzero argument. The theorem assumes `htr` (trace integrability)
and `hlim` (full-sequence trace-moment convergence). It does not identify this
series with the Taylor series of physical growth or of the threshold. -/
theorem exact_mem_zero_radius_unconditional {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T P P)
    (h : Ω → ℝ) (hmeas : Measurable h) (hpos : ∀ ω, 0 ≤ h ω)
    (hunb : ∀ M : ℝ, 0 < M → P {ω | M < h ω} ≠ 0)
    (hmom : ∀ k : ℕ, 0 < k → Integrable (fun ω => (h ω) ^ k) P)
    (μ : ℕ → ℝ)
    (htr : ∀ k : ℕ, 0 < k → ∀ N : ℕ, Integrable (fun ω =>
      Matrix.trace ((jacobi N (fun j => h ((T : Ω → Ω)^[j] ω))) ^ (2 * k))) P)
    (hlim : ∀ k : ℕ, 0 < k → Tendsto (fun N : ℕ =>
      (∫ ω, Matrix.trace ((jacobi N (fun j => h ((T : Ω → Ω)^[j] ω))) ^ (2 * k)) ∂P) /
        (2 * ((N : ℝ) + 1))) atTop (𝓝 (μ k))) :
    (∀ z : ℝ, z ≠ 0 →
      ¬ Summable (fun n : ℕ => (-1 : ℝ) ^ n * μ (n + 1) / ((n : ℝ) + 1) * z ^ (n + 1))) ∧
    ¬ ∃ R : ℝ, 0 < R ∧ ∀ z : ℝ, |z| < R →
      Summable (fun n : ℕ => (-1 : ℝ) ^ n * μ (n + 1) / ((n : ℝ) + 1) * z ^ (n + 1)) := by
  have htail := spectral_tail_of_unbounded_host P T hT h hmeas hpos hunb hmom μ htr hlim
  exact ⟨fun z hz => Dobson.Coverage6ExactZeroRadius.logarithm_series_not_summable μ htail z hz,
    Dobson.Coverage6ExactZeroRadius.no_positive_convergence_radius μ htail⟩

end Dobson.P1bB_ExactMemZeroRadius

#print axioms Dobson.P1bB_ExactMemZeroRadius.jacobi_trace_lower
#print axioms Dobson.P1bB_ExactMemZeroRadius.spectral_tail_of_unbounded_host
#print axioms Dobson.P1bB_ExactMemZeroRadius.exact_mem_zero_radius_unconditional
