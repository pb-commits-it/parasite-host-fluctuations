import DobsonProofs.Coverage6ExactLongStay
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Analysis.Calculus.Taylor

/-! # Part B: Jacobi moments and conditional moving-root coefficients

Coverage correction, 2026-11-22: this module is uncredited in the version-2
inventory; `prop:exact-mem-unbounded-long` remains partial-P1. The independent
fidelity audit (2026-11-16) did not confirm the six claimed closures.

The local Jacobi walk moments and centered expansions are proved. The
spectral coefficient equations assume a root of the supplied integral equation
and existence of its first two right jets. The centered corollary also assumes
identification of the spectral moments with the local moments. Neither spectral
growth identification nor an `O(ε³)` threshold-root error is proved here. -/

namespace Dobson.P1bB_ExactMemLong

open scoped ENNReal BigOperators Topology
open Set MeasureTheory Filter
open Dobson.Coverage6ExactLongStay

/-! ## Section 1: Jacobi walk enumeration and centered stationary moments (`part2:54`, `part2:55`) -/

section JacobiWalkMoments

/-- Diagonal entries of powers of the nearest-neighbor Jacobi operator via its two-step recurrence. -/
noncomputable def walk (h : ℤ → ℝ) : ℕ → ℤ → ℤ → ℝ
  | 0, i, j => if i = j then 1 else 0
  | n + 1, i, j => Real.sqrt (h i) * walk h n (i + 1) j +
      Real.sqrt (h (i - 1)) * walk h n (i - 1) j

lemma walk_four (h : ℤ → ℝ) (hpos : ∀ i, 0 ≤ h i) :
    walk h 4 3 3 = h 1 * h 2 + (h 2) ^ 2 + 2 * h 2 * h 3 + (h 3) ^ 2 + h 3 * h 4 := by
  have hp4 (i : ℤ) : (Real.sqrt (h i)) ^ 4 = (h i) ^ 2 := by
    rw [show 4 = 2 * 2 by omega, pow_mul, Real.sq_sqrt (hpos i)]
  norm_num [walk]
  ring_nf
  simp only [Real.sq_sqrt (hpos 1), Real.sq_sqrt (hpos 2), Real.sq_sqrt (hpos 3),
    Real.sq_sqrt (hpos 4), hp4]
  ring

lemma walk_six (h : ℤ → ℝ) (hpos : ∀ i, 0 ≤ h i) :
    walk h 6 3 3 =
      h 0 * h 1 * h 2 + (h 1) ^ 2 * h 2 + 2 * h 1 * (h 2) ^ 2 + 2 * h 1 * h 2 * h 3 +
      (h 2) ^ 3 + 3 * (h 2) ^ 2 * h 3 + 3 * h 2 * (h 3) ^ 2 + 2 * h 2 * h 3 * h 4 +
      (h 3) ^ 3 + 2 * (h 3) ^ 2 * h 4 + h 3 * (h 4) ^ 2 + h 3 * h 4 * h 5 := by
  have hp4 (i : ℤ) : (Real.sqrt (h i)) ^ 4 = (h i) ^ 2 := by
    rw [show 4 = 2 * 2 by omega, pow_mul, Real.sq_sqrt (hpos i)]
  have hp6 (i : ℤ) : (Real.sqrt (h i)) ^ 6 = (h i) ^ 3 := by
    rw [show 6 = 2 * 3 by omega, pow_mul, Real.sq_sqrt (hpos i)]
  norm_num [walk]
  ring_nf
  simp only [Real.sq_sqrt (hpos 0), Real.sq_sqrt (hpos 1), Real.sq_sqrt (hpos 2),
    Real.sq_sqrt (hpos 3), Real.sq_sqrt (hpos 4), Real.sq_sqrt (hpos 5), hp4, hp6]
  ring

variable {Ω : Type*} [MeasurableSpace Ω]

def orbit (T : Ω ≃ᵐ Ω) : ℤ → Ω → Ω
  | Int.ofNat n => (T : Ω → Ω)^[n]
  | Int.negSucc n => (T.symm : Ω → Ω)^[n + 1]

noncomputable def hostAt (T : Ω ≃ᵐ Ω) (h : Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  h ((T : Ω → Ω)^[n] ω)

/-- Expected local Jacobi walk moment `(1 / 2) E[walk_{2k}(3, 3)]`.
Identification with an infinite-volume spectral moment is a separate premise. -/
noncomputable def localMoment (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (h : Ω → ℝ) (k : ℕ) : ℝ :=
  (∫ ω, walk (fun j => h (orbit T j ω)) (2 * k) 3 3 ∂μ) / 2

private theorem integral_shift (μ : Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (f : Ω → ℝ) (n : ℕ) :
    (∫ ω, f ((T : Ω → Ω)^[n] ω) ∂μ) = ∫ ω, f ω ∂μ := by
  induction n with
  | zero => simp
  | succ n ih =>
    simpa only [Function.iterate_succ_apply] using
      (hT.integral_comp' (fun ω => f ((T : Ω → Ω)^[n] ω))).trans ih

private theorem integral_pair_shift (μ : Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (h : Ω → ℝ) (i j n : ℕ) :
    (∫ ω, hostAt T h (i + n) ω * hostAt T h (j + n) ω ∂μ) =
      ∫ ω, hostAt T h i ω * hostAt T h j ω ∂μ := by
  simpa only [hostAt, Function.iterate_add_apply] using
    integral_shift μ T hT (fun ω => hostAt T h i ω * hostAt T h j ω) n

private theorem integral_triple_shift (μ : Measure Ω) (T : Ω ≃ᵐ Ω)
    (hT : MeasurePreserving T μ μ) (h : Ω → ℝ) (i j k n : ℕ) :
    (∫ ω, hostAt T h (i + n) ω * hostAt T h (j + n) ω * hostAt T h (k + n) ω ∂μ) =
      ∫ ω, hostAt T h i ω * hostAt T h j ω * hostAt T h k ω ∂μ := by
  simpa only [hostAt, Function.iterate_add_apply] using
    integral_shift μ T hT (fun ω => hostAt T h i ω * hostAt T h j ω * hostAt T h k ω) n

private theorem integrable_pair (μ : Measure Ω) [IsProbabilityMeasure μ]
    (f g : Ω → ℝ) (hf : MemLp f 4 μ) (hg : MemLp g 4 μ) :
    Integrable (fun ω => f ω * g ω) μ := by
  have hf2 : MemLp f 2 μ := hf.mono_exponent (by norm_num)
  have hg2 : MemLp g 2 μ := hg.mono_exponent (by norm_num)
  exact hf2.integrable_mul hg2

private theorem integrable_triple (μ : Measure Ω) [IsProbabilityMeasure μ]
    (f g h : Ω → ℝ) (hf : MemLp f 4 μ) (hg : MemLp g 4 μ) (hh : MemLp h 4 μ) :
    Integrable (fun ω => f ω * g ω * h ω) μ := by
  let : ENNReal.HolderTriple 4 4 2 := ⟨by
    rw [← two_mul, show (4 : ℝ≥0∞) = 2 * 2 by norm_num,
      ENNReal.mul_inv (by simp) (by simp), ← mul_assoc,
      ENNReal.mul_inv_cancel (by norm_num) (by norm_num), one_mul]⟩
  have hfg : MemLp (fun ω => f ω * g ω) 2 μ := hf.mul hg
  have hh2 : MemLp h 2 μ := hh.mono_exponent (by norm_num)
  exact hfg.integrable_mul hh2

theorem localMoment_two (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (h : Ω → ℝ)
    (hh : MemLp h 4 μ) (hpos : ∀ ω, 0 ≤ h ω) :
    localMoment μ T h 2 = (∫ ω, (h ω) ^ 2 ∂μ) + 2 * (∫ ω, h ω * h (T ω) ∂μ) := by
  have hs (n : ℕ) : MemLp (hostAt T h n) 4 μ := hh.comp_measurePreserving (hT.iterate n)
  have hi (i j : ℕ) : Integrable (fun ω => hostAt T h i ω * hostAt T h j ω) μ :=
    integrable_pair μ _ _ (hs i) (hs j)
  have hi2 (i : ℕ) : Integrable (fun ω => (hostAt T h i ω) ^ 2) μ := by
    simpa only [pow_two] using hi i i
  have hp (ω : Ω) := walk_four (fun j => h (orbit T j ω)) (fun _ => hpos _)
  have hsquare (i : ℕ) : (∫ ω, (hostAt T h i ω) ^ 2 ∂μ) = ∫ ω, (h ω) ^ 2 ∂μ :=
    integral_shift μ T hT (fun ω => (h ω) ^ 2) i
  have hadj (i : ℕ) : (∫ ω, hostAt T h i ω * hostAt T h (i + 1) ω ∂μ) =
      ∫ ω, h ω * h (T ω) ∂μ := by
    simpa [hostAt, Function.iterate_one, Nat.add_comm] using integral_pair_shift μ T hT h 0 1 i
  unfold localMoment
  change (∫ ω, walk (fun j => h (orbit T j ω)) 4 3 3 ∂μ) / 2 = _
  conv_lhs => arg 1; arg 2; ext ω; rw [hp ω]
  change (∫ ω, hostAt T h 1 ω * hostAt T h 2 ω + (hostAt T h 2 ω) ^ 2 +
    2 * hostAt T h 2 ω * hostAt T h 3 ω + (hostAt T h 3 ω) ^ 2 +
    hostAt T h 3 ω * hostAt T h 4 ω ∂μ) / 2 = _
  simp_rw [mul_assoc]
  rw [integral_add (by fun_prop) (hi 3 4), integral_add (by fun_prop) (hi2 3),
    integral_add (by fun_prop) (by fun_prop), integral_add (hi 1 2) (hi2 2), integral_const_mul,
    hsquare, hsquare, hadj 1, hadj 2, hadj 3]
  ring

theorem localMoment_three (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (h : Ω → ℝ)
    (hh : MemLp h 4 μ) (hpos : ∀ ω, 0 ≤ h ω) :
    localMoment μ T h 3 = (∫ ω, (h ω) ^ 3 ∂μ) +
      3 * ((∫ ω, (h ω) ^ 2 * h (T ω) ∂μ) + (∫ ω, h ω * (h (T ω)) ^ 2 ∂μ) +
        (∫ ω, h ω * h (T ω) * h (T (T ω)) ∂μ)) := by
  have hs (n : ℕ) : MemLp (hostAt T h n) 4 μ := hh.comp_measurePreserving (hT.iterate n)
  have hi (i j k : ℕ) : Integrable (fun ω => hostAt T h i ω * hostAt T h j ω *
      hostAt T h k ω) μ := integrable_triple μ _ _ _ (hs i) (hs j) (hs k)
  have hit (i j k : ℕ) : Integrable (fun ω => hostAt T h i ω *
      (hostAt T h j ω * hostAt T h k ω)) μ := by simpa only [mul_assoc] using hi i j k
  have hia (i j : ℕ) : Integrable (fun ω => (hostAt T h i ω) ^ 2 * hostAt T h j ω) μ := by
    simpa only [pow_two] using hi i i j
  have hib (i j : ℕ) : Integrable (fun ω => hostAt T h i ω * (hostAt T h j ω) ^ 2) μ := by
    simpa only [pow_two, mul_assoc] using hi i j j
  have hi3 (i : ℕ) : Integrable (fun ω => (hostAt T h i ω) ^ 3) μ := by
    simpa [pow_succ, pow_two] using hi i i i
  have hc (i : ℕ) : (∫ ω, (hostAt T h i ω) ^ 3 ∂μ) = ∫ ω, (h ω) ^ 3 ∂μ :=
    integral_shift μ T hT (fun ω => (h ω) ^ 3) i
  have ha (i : ℕ) : (∫ ω, (hostAt T h i ω) ^ 2 * hostAt T h (i + 1) ω ∂μ) =
      ∫ ω, (h ω) ^ 2 * h (T ω) ∂μ := by
    simpa [hostAt, pow_two, Nat.add_comm] using integral_triple_shift μ T hT h 0 0 1 i
  have hb (i : ℕ) : (∫ ω, hostAt T h i ω * (hostAt T h (i + 1) ω) ^ 2 ∂μ) =
      ∫ ω, h ω * (h (T ω)) ^ 2 ∂μ := by
    simpa [hostAt, pow_two, Nat.add_comm, mul_assoc] using integral_triple_shift μ T hT h 0 1 1 i
  have ht (i : ℕ) : (∫ ω, hostAt T h i ω * (hostAt T h (i + 1) ω * hostAt T h (i + 2) ω) ∂μ) =
      ∫ ω, h ω * h (T ω) * h (T (T ω)) ∂μ := by
    simpa [hostAt, Nat.add_comm, Function.iterate_succ_apply, mul_assoc] using
      integral_triple_shift μ T hT h 0 1 2 i
  have hp (ω : Ω) := walk_six (fun j => h (orbit T j ω)) (fun _ => hpos _)
  unfold localMoment
  change (∫ ω, walk (fun j => h (orbit T j ω)) 6 3 3 ∂μ) / 2 = _
  conv_lhs => arg 1; arg 2; ext ω; rw [hp ω]
  change (∫ ω, hostAt T h 0 ω * hostAt T h 1 ω * hostAt T h 2 ω +
    (hostAt T h 1 ω) ^ 2 * hostAt T h 2 ω + 2 * hostAt T h 1 ω * (hostAt T h 2 ω) ^ 2 +
    2 * hostAt T h 1 ω * hostAt T h 2 ω * hostAt T h 3 ω + (hostAt T h 2 ω) ^ 3 +
    3 * (hostAt T h 2 ω) ^ 2 * hostAt T h 3 ω + 3 * hostAt T h 2 ω * (hostAt T h 3 ω) ^ 2 +
    2 * hostAt T h 2 ω * hostAt T h 3 ω * hostAt T h 4 ω + (hostAt T h 3 ω) ^ 3 +
    2 * (hostAt T h 3 ω) ^ 2 * hostAt T h 4 ω + hostAt T h 3 ω * (hostAt T h 4 ω) ^ 2 +
    hostAt T h 3 ω * hostAt T h 4 ω * hostAt T h 5 ω ∂μ) / 2 = _
  simp_rw [mul_assoc]
  repeat rw [integral_add (by fun_prop) (by fun_prop)]
  simp only [integral_const_mul]
  rw [ht 0, ht 1, ht 2, ht 3, ha 1, ha 2, ha 3, hb 1, hb 2, hb 3, hc 2, hc 3]
  simp only [mul_assoc]
  ring

private theorem integral_one_add_pair (μ : Measure Ω) [IsProbabilityMeasure μ]
    (f g : Ω → ℝ) (hf : MemLp f 4 μ) (hg : MemLp g 4 μ)
    (hf0 : (∫ ω, f ω ∂μ) = 0) (hg0 : (∫ ω, g ω ∂μ) = 0) :
    (∫ ω, (1 + f ω) * (1 + g ω) ∂μ) = 1 + (∫ ω, f ω * g ω ∂μ) := by
  have hfi : Integrable f μ := hf.integrable (by norm_num)
  have hgi : Integrable g μ := hg.integrable (by norm_num)
  have hp := integrable_pair μ f g hf hg
  have he : (fun ω => (1 + f ω) * (1 + g ω)) =
      (fun ω => 1 + f ω + g ω + f ω * g ω) := by funext ω; ring
  rw [he]
  repeat rw [integral_add (by fun_prop) (by fun_prop)]
  simp [hf0, hg0]

private theorem integral_one_add_triple (μ : Measure Ω) [IsProbabilityMeasure μ]
    (f g h : Ω → ℝ) (hf : MemLp f 4 μ) (hg : MemLp g 4 μ) (hh : MemLp h 4 μ)
    (hf0 : (∫ ω, f ω ∂μ) = 0) (hg0 : (∫ ω, g ω ∂μ) = 0)
    (hh0 : (∫ ω, h ω ∂μ) = 0) :
    (∫ ω, (1 + f ω) * (1 + g ω) * (1 + h ω) ∂μ) =
      1 + (∫ ω, f ω * g ω ∂μ) + (∫ ω, f ω * h ω ∂μ) + (∫ ω, g ω * h ω ∂μ) +
        (∫ ω, f ω * g ω * h ω ∂μ) := by
  have hfi : Integrable f μ := hf.integrable (by norm_num)
  have hgi : Integrable g μ := hg.integrable (by norm_num)
  have hhi : Integrable h μ := hh.integrable (by norm_num)
  have hfg := integrable_pair μ f g hf hg
  have hfh := integrable_pair μ f h hf hh
  have hgh := integrable_pair μ g h hg hh
  have ht := integrable_triple μ f g h hf hg hh
  have he : (fun ω => (1 + f ω) * (1 + g ω) * (1 + h ω)) =
      (fun ω => 1 + f ω + g ω + h ω + f ω * g ω + f ω * h ω + g ω * h ω + f ω * g ω * h ω) := by
    funext ω; ring
  rw [he]
  repeat rw [integral_add (by fun_prop) (by fun_prop)]
  simp [hf0, hg0, hh0]

/-- Centered stationary moments for the shift `T`. If `T` is the forward
shift, `triple12 = E[b₀ b₁²]` and `triple21 = E[b₀² b₁]` have the opposite
labels to the manuscript's backward-lag convention. Their symmetric sum in
the coefficient formulas agrees; time reversibility is not assumed. -/
noncomputable def variance (μ : Measure Ω) (b : Ω → ℝ) : ℝ := ∫ ω, (b ω) ^ 2 ∂μ
noncomputable def correlation (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (b : Ω → ℝ) (n : ℕ) : ℝ :=
  ∫ ω, b ω * hostAt T b n ω ∂μ
noncomputable def thirdMoment (μ : Measure Ω) (b : Ω → ℝ) : ℝ := ∫ ω, (b ω) ^ 3 ∂μ
noncomputable def triple12 (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (b : Ω → ℝ) : ℝ :=
  ∫ ω, b ω * (b (T ω)) ^ 2 ∂μ
noncomputable def triple21 (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (b : Ω → ℝ) : ℝ :=
  ∫ ω, (b ω) ^ 2 * b (T ω) ∂μ
noncomputable def triple111 (μ : Measure Ω) (T : Ω ≃ᵐ Ω) (b : Ω → ℝ) : ℝ :=
  ∫ ω, b ω * b (T ω) * b (T (T ω)) ∂μ

/-- For the local Jacobi walk moment, `localMoment ... 2 = 3 + V + 2 C₁`.
This does not identify a separately supplied spectral second moment. -/
lemma localMoment_two_centered (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (b : Ω → ℝ)
    (hb : MemLp b 4 μ) (hb0 : (∫ ω, b ω ∂μ) = 0) (hpos : ∀ ω, 0 ≤ 1 + b ω) :
    localMoment μ T (fun ω => 1 + b ω) 2 =
      3 + variance μ b + 2 * correlation μ T b 1 := by
  have hh : MemLp (fun ω => 1 + b ω) 4 μ := (memLp_const 1).add hb
  have hb1 : MemLp (fun ω => b (T ω)) 4 μ := hb.comp_measurePreserving hT
  have hb10 : (∫ ω, b (T ω) ∂μ) = 0 := (hT.integral_comp' b).trans hb0
  have hs := integral_one_add_pair μ b b hb hb hb0 hb0
  have ha := integral_one_add_pair μ b (fun ω => b (T ω)) hb hb1 hb0 hb10
  rw [localMoment_two μ T hT _ hh hpos]
  simp only [← pow_two] at hs
  rw [hs, ha]
  simp only [variance, correlation, hostAt, Function.iterate_one]
  ring

/-- The local third Jacobi walk moment equals
`10 + 9 V + 18 C₁ + 3 C₂ + 3 T₁₂ + 3 T₂₁ + 3 T₁₁₁ + m₃`, without time
reversibility. Identification with a spectral third moment remains separate. -/
lemma localMoment_three_centered (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T μ μ) (b : Ω → ℝ)
    (hb : MemLp b 4 μ) (hb0 : (∫ ω, b ω ∂μ) = 0) (hpos : ∀ ω, 0 ≤ 1 + b ω) :
    localMoment μ T (fun ω => 1 + b ω) 3 =
      10 + 9 * variance μ b + 18 * correlation μ T b 1 + 3 * correlation μ T b 2 +
        3 * triple12 μ T b + 3 * triple21 μ T b + 3 * triple111 μ T b + thirdMoment μ b := by
  have hh : MemLp (fun ω => 1 + b ω) 4 μ := (memLp_const 1).add hb
  have hb1 : MemLp (fun ω => b (T ω)) 4 μ := hb.comp_measurePreserving hT
  have hb2 : MemLp (fun ω => b (T (T ω))) 4 μ := hb1.comp_measurePreserving hT
  have hb10 : (∫ ω, b (T ω) ∂μ) = 0 := (hT.integral_comp' b).trans hb0
  have hb20 : (∫ ω, b (T (T ω)) ∂μ) = 0 :=
    (hT.integral_comp' (fun ω => b (T ω))).trans hb10
  have hc := integral_one_add_triple μ b b b hb hb hb hb0 hb0 hb0
  have ha := integral_one_add_triple μ b b (fun ω => b (T ω)) hb hb hb1 hb0 hb0 hb10
  have hd := integral_one_add_triple μ b (fun ω => b (T ω)) (fun ω => b (T ω))
    hb hb1 hb1 hb0 hb10 hb10
  have ht := integral_one_add_triple μ b (fun ω => b (T ω)) (fun ω => b (T (T ω)))
    hb hb1 hb2 hb0 hb10 hb20
  have hv : (∫ ω, b (T ω) * b (T ω) ∂μ) = ∫ ω, b ω * b ω ∂μ :=
    hT.integral_comp' (fun ω => b ω * b ω)
  have hadj : (∫ ω, b (T ω) * b (T (T ω)) ∂μ) = ∫ ω, b ω * b (T ω) ∂μ :=
    hT.integral_comp' (fun ω => b ω * b (T ω))
  have hcube : (fun ω => (1 + b ω) ^ 3) = (fun ω => (1 + b ω) * (1 + b ω) * (1 + b ω)) := by
    funext ω; ring
  rw [localMoment_three μ T hT _ hh hpos, hcube]
  simp only [pow_two, ← mul_assoc]
  rw [hc, ha, hd, ht, hv, hadj]
  simp only [variance, correlation, thirdMoment, triple12, triple21, triple111,
    hostAt, Function.iterate_succ_apply, Function.iterate_zero_apply, pow_succ, pow_zero, one_mul]
  ring

end JacobiWalkMoments

/-! ## Section 2: Conditional moving-root jet matching

The integral remainder is controlled, but no cubic threshold-root error bound
is concluded. -/

section SpectralMovingRoot

/-- Scalar logarithm cubic remainder on `(-1/2, 1/2)` for `log(1 - ε)`:
`|log(1 - ε) + ε + ε² / 2 + ε³ / 3| ≤ ε⁴` for `0 ≤ ε ≤ 1 / 2`. -/
lemma log_one_sub_cubic_bound (ε : ℝ) (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1 / 2) :
    |Real.log (1 - ε) + ε + ε ^ 2 / 2 + ε ^ 3 / 3| ≤ ε ^ 4 := by
  let g : ℝ → ℝ := fun x => -(Real.log (1 - x) + x + x ^ 2 / 2 + x ^ 3 / 3)
  have hg_deriv : ∀ x ∈ Icc (0 : ℝ) (1 / 2), HasDerivAt g (x ^ 3 / (1 - x)) x := by
    intro x hx
    have h1x : 1 - x ≠ 0 := by linarith [hx.2]
    have hd := (((((hasDerivAt_id x).const_sub 1).log h1x).add (hasDerivAt_id x)).add
      (((hasDerivAt_id x).pow 2).div_const 2)).add (((hasDerivAt_id x).pow 3).div_const 3)
    convert hd.neg using 1
    · ext t; rfl
    · dsimp
      field_simp [h1x]
      ring
  have hup_deriv : ∀ x ∈ Icc (0 : ℝ) (1 / 2),
      HasDerivAt (fun t => t ^ 4 - g t) (x ^ 3 * (3 - 4 * x) / (1 - x)) x := by
    intro x hx
    have h1x : 1 - x ≠ 0 := by linarith [hx.2]
    convert (((hasDerivAt_id x).pow 4).sub (hg_deriv x hx)) using 1
    · ext t; rfl
    · dsimp
      field_simp [h1x]
      ring
  have hmono1 : MonotoneOn g (Icc 0 (1 / 2)) := by
    apply monotoneOn_of_deriv_nonneg (convex_Icc 0 (1 / 2))
    · intro x hx; exact (hg_deriv x hx).continuousAt.continuousWithinAt
    · intro x hx; exact (hg_deriv x (interior_subset hx)).differentiableAt.differentiableWithinAt
    · intro x hx
      have hx' := interior_subset hx
      rw [(hg_deriv x hx').deriv]
      exact div_nonneg (pow_nonneg hx'.1 3) (by linarith [hx'.2])
  have hmono2 : MonotoneOn (fun t => t ^ 4 - g t) (Icc 0 (1 / 2)) := by
    apply monotoneOn_of_deriv_nonneg (convex_Icc 0 (1 / 2))
    · intro x hx; exact (hup_deriv x hx).continuousAt.continuousWithinAt
    · intro x hx; exact (hup_deriv x (interior_subset hx)).differentiableAt.differentiableWithinAt
    · intro x hx
      have hx' := interior_subset hx
      rw [(hup_deriv x hx').deriv]
      exact div_nonneg (mul_nonneg (pow_nonneg hx'.1 3) (by linarith [hx'.2])) (by linarith [hx'.2])
  have h0_mem : (0 : ℝ) ∈ Icc (0 : ℝ) (1 / 2) := ⟨le_rfl, by norm_num⟩
  have hε_mem : ε ∈ Icc (0 : ℝ) (1 / 2) := ⟨hε0, hε1⟩
  have hlo := hmono1 h0_mem hε_mem hε0
  have hhi := hmono2 h0_mem hε_mem hε0
  have hg0 : g 0 = 0 := by simp [g]
  rw [hg0] at hlo
  simp only [zero_pow (by norm_num : 4 ≠ 0), hg0, sub_zero] at hhi
  have heq : Real.log (1 - ε) + ε + ε ^ 2 / 2 + ε ^ 3 / 3 = -g ε := by dsimp [g]; ring
  rw [heq, abs_neg, abs_of_nonneg hlo]
  linarith

/-- Moving-root coordinate `z(ε, D) = ε * exp(D) / (1 - ε)²`. -/
noncomputable def spectralZ (ε D : ℝ) : ℝ :=
  ε * Real.exp D / (1 - ε) ^ 2

/-- Elementary limit of `(z(ε, D(ε)) - ε - (A + 2) ε²) / ε³` when `D(ε) / ε → A` and
`(D(ε) - A ε) / ε² → B`: equals `B + A² / 2 + 2 A + 3`. -/
lemma spectralZ_cubic_jet (D : ℝ → ℝ) (A B : ℝ)
    (hD1 : Tendsto (fun ε : ℝ => D ε / ε) (𝓝[>] 0) (𝓝 A))
    (hD2 : Tendsto (fun ε : ℝ => (D ε - A * ε) / ε ^ 2) (𝓝[>] 0) (𝓝 B)) :
    Tendsto (fun ε : ℝ => spectralZ ε (D ε) / ε) (𝓝[>] 0) (𝓝 1) ∧
    Tendsto (fun ε : ℝ => (spectralZ ε (D ε) - ε) / ε ^ 2) (𝓝[>] 0) (𝓝 (A + 2)) ∧
    Tendsto (fun ε : ℝ => (spectralZ ε (D ε) - ε - (A + 2) * ε ^ 2) / ε ^ 3)
      (𝓝[>] 0) (𝓝 (B + A ^ 2 / 2 + 2 * A + 3)) := by
  have hid_gt : Tendsto (fun ε : ℝ => ε) (𝓝[>] 0) (𝓝 0) :=
    tendsto_id.mono_left nhdsWithin_le_nhds
  have hD0 : Tendsto D (𝓝[>] 0) (𝓝 0) := by
    have ht := hD1.mul hid_gt
    simp only [mul_zero] at ht
    apply ht.congr'
    filter_upwards [self_mem_nhdsWithin] with ε hε
    change 0 < ε at hε
    field_simp [hε.ne']
  have hexp0 : Tendsto (fun ε : ℝ => Real.exp (D ε)) (𝓝[>] 0) (𝓝 1) := by
    simpa only [Function.comp_def, Real.exp_zero] using (Real.continuous_exp.tendsto 0).comp hD0
  have hden0 : Tendsto (fun ε : ℝ => (1 - ε) ^ 2) (𝓝[>] 0) (𝓝 1) := by
    have hc : ContinuousAt (fun ε : ℝ => (1 - ε) ^ 2) 0 := by fun_prop
    simpa using hc.tendsto.mono_left nhdsWithin_le_nhds
  have hz1 : Tendsto (fun ε : ℝ => spectralZ ε (D ε) / ε) (𝓝[>] 0) (𝓝 1) := by
    have ht := hexp0.div hden0 one_ne_zero
    simp only [div_one] at ht
    apply ht.congr'
    filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with ε hε
    change Real.exp (D ε) / (1 - ε) ^ 2 = (ε * Real.exp (D ε) / (1 - ε) ^ 2) / ε
    have hs1 : 1 - ε ≠ 0 := by linarith [hε.2]
    field_simp [hε.1.ne', hs1]
  let g1 : ℝ → ℝ := fun x => if x = 0 then 1 else (Real.exp x - 1) / x
  have hg1_tend : Tendsto g1 (𝓝 0) (𝓝 1) := by
    have hd : HasDerivAt Real.exp 1 0 := by simpa using Real.hasDerivAt_exp 0
    have hsl := hd.tendsto_slope
    simp only [slope_fun_def_field, sub_zero, Real.exp_zero] at hsl
    have hne : Tendsto g1 (𝓝[≠] 0) (𝓝 1) := by
      apply hsl.congr'
      filter_upwards [self_mem_nhdsWithin] with x hx
      have hx' : x ≠ 0 := hx
      simp [g1, hx']
    have hcont : ContinuousAt g1 0 := by
      rw [continuousAt_iff_punctured_nhds, show g1 0 = 1 by simp [g1]]
      exact hne
    simpa [g1] using hcont.tendsto
  let g2 : ℝ → ℝ := fun x => if x = 0 then 1 / 2 else (Real.exp x - 1 - x) / x ^ 2
  have hg2_tend : Tendsto g2 (𝓝 0) (𝓝 (1 / 2)) := by
    have hc : ContDiff ℝ 2 Real.exp := Real.contDiff_exp
    have ht := Real.taylor_tendsto convex_univ (mem_univ (0 : ℝ)) hc.contDiffOn
    rw [nhdsWithin_univ] at ht
    have hh := (ht.mono_left (nhdsWithin_le_nhds (s := ({0}ᶜ : Set ℝ)))).add_const (1 / 2 : ℝ)
    simp only [zero_add] at hh
    have hne : Tendsto g2 (𝓝[≠] 0) (𝓝 (1 / 2)) := by
      apply hh.congr'
      filter_upwards [self_mem_nhdsWithin] with x hx
      have hx' : x ≠ 0 := hx
      have hteval : taylorWithinEval Real.exp 2 univ 0 x = 1 + x + x ^ 2 / 2 := by
        rw [taylorWithinEval_succ (n := 1), taylorWithinEval_succ (n := 0)]
        have hd1 : iteratedDeriv 1 Real.exp 0 = 1 := by simp [iteratedDeriv_one]
        have hd2 : iteratedDeriv 2 Real.exp 0 = 1 := by
          rw [iteratedDeriv_succ, iteratedDeriv_one, Real.deriv_exp, Real.deriv_exp, Real.exp_zero]
        simp [taylor_within_zero_eval, iteratedDerivWithin_univ, hd1, hd2]
        ring
      rw [hteval]
      simp only [g2, ite_eq_right_iff.mpr fun h => (hx' h).elim, sub_zero]
      field_simp [hx']
      ring
    have hcont : ContinuousAt g2 0 := by
      rw [continuousAt_iff_punctured_nhds, show g2 0 = 1 / 2 by simp [g2]]
      exact hne
    simpa [g2] using hcont.tendsto
  have hexpD1 : Tendsto (fun ε : ℝ => (Real.exp (D ε) - 1) / ε) (𝓝[>] 0) (𝓝 A) := by
    have ht := (hg1_tend.comp hD0).mul hD1
    simp only [one_mul] at ht
    apply ht.congr'
    filter_upwards [self_mem_nhdsWithin] with ε hε
    change 0 < ε at hε
    change g1 (D ε) * (D ε / ε) = (Real.exp (D ε) - 1) / ε
    dsimp only [g1]
    by_cases hDε : D ε = 0
    · simp [hDε]
    · rw [ite_eq_right_iff.mpr fun h => (hDε h).elim]
      field_simp [hDε, hε.ne']
  have hexpD2 : Tendsto (fun ε : ℝ => (Real.exp (D ε) - 1 - A * ε) / ε ^ 2)
      (𝓝[>] 0) (𝓝 (B + A ^ 2 / 2)) := by
    have ht := hD2.add ((hg2_tend.comp hD0).mul (hD1.pow 2))
    have hval : B + 1 / 2 * A ^ 2 = B + A ^ 2 / 2 := by ring
    rw [hval] at ht
    apply ht.congr'
    filter_upwards [self_mem_nhdsWithin] with ε hε
    change 0 < ε at hε
    change (D ε - A * ε) / ε ^ 2 + g2 (D ε) * (D ε / ε) ^ 2 = (Real.exp (D ε) - 1 - A * ε) / ε ^ 2
    dsimp only [g2]
    by_cases hDε : D ε = 0
    · simp [hDε]
    · rw [ite_eq_right_iff.mpr fun h => (hDε h).elim]
      field_simp [hDε, hε.ne']
      ring
  have hz2 : Tendsto (fun ε : ℝ => (spectralZ ε (D ε) - ε) / ε ^ 2) (𝓝[>] 0) (𝓝 (A + 2)) := by
    have hnum := hexpD1.add ((tendsto_const_nhds (x := (2 : ℝ))).sub hid_gt)
    have ht := hnum.div hden0 one_ne_zero
    have hval : (A + (2 - 0)) / 1 = A + 2 := by ring
    rw [hval] at ht
    apply ht.congr'
    filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with ε hε
    change ((Real.exp (D ε) - 1) / ε + (2 - ε)) / (1 - ε) ^ 2 =
      (ε * Real.exp (D ε) / (1 - ε) ^ 2 - ε) / ε ^ 2
    have hs1 : 1 - ε ≠ 0 := by linarith [hε.2]
    field_simp [hε.1.ne', hs1]
    ring
  refine ⟨hz1, hz2, ?_⟩
  have hnum3 := (hexpD2.add (tendsto_const_nhds (x := 2 * A + 3))).sub
    (hid_gt.const_mul (A + 2))
  have ht3 := hnum3.div hden0 one_ne_zero
  have hval3 : (B + A ^ 2 / 2 + (2 * A + 3) - (A + 2) * 0) / 1 = B + A ^ 2 / 2 + 2 * A + 3 := by
    ring
  rw [hval3] at ht3
  apply ht3.congr'
  filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with ε hε
  change ((Real.exp (D ε) - 1 - A * ε) / ε ^ 2 + (2 * A + 3) - (A + 2) * ε) / (1 - ε) ^ 2 =
    (ε * Real.exp (D ε) / (1 - ε) ^ 2 - ε - (A + 2) * ε ^ 2) / ε ^ 3
  have hs1 : 1 - ε ≠ 0 := by linarith [hε.2]
  field_simp [hε.1.ne', hs1]
  ring

/-- Conditional coefficient matching for the supplied spectral integral.
Assuming the root equation and existence of the first two right jets of `D`,
`spectral_integral_quartic` gives the order-`ε²` and order-`ε³` equations.
This does not construct those jets or identify the physical growth function. -/
lemma spectral_threshold_jet_equations {α : Type*} [MeasurableSpace α]
    (ν : Measure α) (X : α → ℝ)
    (hX : ∀ᵐ x ∂ν, 0 ≤ X x)
    (h1 : Integrable X ν) (h2 : Integrable (fun x => X x ^ 2) ν)
    (h3 : Integrable (fun x => X x ^ 3) ν) (h4 : Integrable (fun x => X x ^ 4) ν)
    (hlog : ∀ z : ℝ, 0 ≤ z → Integrable (fun x => Real.log (1 + z * X x)) ν)
    (hμ1 : (∫ x, X x ∂ν) = 1)
    (D : ℝ → ℝ) (A B : ℝ)
    (hroot : ∀ᶠ ε in 𝓝[>] (0 : ℝ),
      Real.log (1 - ε) + (∫ x, Real.log (1 + spectralZ ε (D ε) * X x) ∂ν) = 0)
    (hD1 : Tendsto (fun ε : ℝ => D ε / ε) (𝓝[>] 0) (𝓝 A))
    (hD2 : Tendsto (fun ε : ℝ => (D ε - A * ε) / ε ^ 2) (𝓝[>] 0) (𝓝 B)) :
    let μ₂ := ∫ x, X x ^ 2 ∂ν
    let μ₃ := ∫ x, X x ^ 3 ∂ν
    (A + 2 - μ₂ / 2 - 1 / 2 = 0) ∧
    (B + A ^ 2 / 2 + 2 * A + 3 - μ₂ * (A + 2) + μ₃ / 3 - 1 / 3 = 0) := by
  intro μ₂ μ₃
  let μ₄ := ∫ x, X x ^ 4 ∂ν
  obtain ⟨hz1, hz2, hz3⟩ := spectralZ_cubic_jet D A B hD1 hD2
  let Rlog : ℝ → ℝ := fun ε => Real.log (1 - ε) + ε + ε ^ 2 / 2 + ε ^ 3 / 3
  let Rspec : ℝ → ℝ := fun ε =>
    (∫ x, Real.log (1 + spectralZ ε (D ε) * X x) ∂ν) -
      spectralZ ε (D ε) * (∫ x, X x ∂ν) +
      (spectralZ ε (D ε)) ^ 2 / 2 * μ₂ -
      (spectralZ ε (D ε)) ^ 3 / 3 * μ₃
  have hRlog3 : Tendsto (fun ε : ℝ => Rlog ε / ε ^ 3) (𝓝[>] 0) (𝓝 0) := by
    have hhi : Tendsto (fun ε : ℝ => ε) (𝓝[>] 0) (𝓝 0) :=
      tendsto_id.mono_left nhdsWithin_le_nhds
    have hlo : Tendsto (fun ε : ℝ => -ε) (𝓝[>] 0) (𝓝 0) := by
      simpa only [neg_zero] using hhi.neg
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' hlo hhi
    · filter_upwards [Ioc_mem_nhdsGT (show (0 : ℝ) < 1 / 2 by norm_num)] with ε hε
      have hb := (abs_le.mp (log_one_sub_cubic_bound ε hε.1.le hε.2)).1
      rw [le_div_iff₀ (pow_pos hε.1 3)]
      dsimp only [Rlog]
      nlinarith
    · filter_upwards [Ioc_mem_nhdsGT (show (0 : ℝ) < 1 / 2 by norm_num)] with ε hε
      have hb := (abs_le.mp (log_one_sub_cubic_bound ε hε.1.le hε.2)).2
      rw [div_le_iff₀ (pow_pos hε.1 3)]
      dsimp only [Rlog]
      nlinarith
  have hRspec3 : Tendsto (fun ε : ℝ => Rspec ε / ε ^ 3) (𝓝[>] 0) (𝓝 0) := by
    have hbound : Tendsto (fun ε : ℝ =>
        (spectralZ ε (D ε) / ε) ^ 4 * (μ₄ / 4) * ε) (𝓝[>] 0) (𝓝 0) := by
      have ht := ((hz1.pow 4).mul_const (μ₄ / 4)).mul
        (tendsto_id.mono_left nhdsWithin_le_nhds)
      simpa using ht
    have hbound_neg : Tendsto (fun ε : ℝ =>
        -((spectralZ ε (D ε) / ε) ^ 4 * (μ₄ / 4) * ε)) (𝓝[>] 0) (𝓝 0) := by
      simpa only [neg_zero] using hbound.neg
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' hbound_neg hbound
    · filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with ε hε
      have hz_nonneg : 0 ≤ spectralZ ε (D ε) := by
        unfold spectralZ
        exact div_nonneg (mul_nonneg hε.1.le (Real.exp_pos _).le) (sq_nonneg _)
      have hq := spectral_integral_quartic ν X (spectralZ ε (D ε)) hz_nonneg
        hX h1 h2 h3 h4 (hlog _ hz_nonneg)
      have hlo := (abs_le.mp hq).1
      rw [le_div_iff₀ (pow_pos hε.1 3)]
      have hid : -((spectralZ ε (D ε) / ε) ^ 4 * (μ₄ / 4) * ε) * ε ^ 3 =
          -((spectralZ ε (D ε)) ^ 4 / 4 * μ₄) := by
        field_simp [hε.1.ne']
      rw [hid]
      exact hlo
    · filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 1 by norm_num)] with ε hε
      have hz_nonneg : 0 ≤ spectralZ ε (D ε) := by
        unfold spectralZ
        exact div_nonneg (mul_nonneg hε.1.le (Real.exp_pos _).le) (sq_nonneg _)
      have hq := spectral_integral_quartic ν X (spectralZ ε (D ε)) hz_nonneg
        hX h1 h2 h3 h4 (hlog _ hz_nonneg)
      have hhi := (abs_le.mp hq).2
      rw [div_le_iff₀ (pow_pos hε.1 3)]
      have hid : (spectralZ ε (D ε) / ε) ^ 4 * (μ₄ / 4) * ε * ε ^ 3 =
          (spectralZ ε (D ε)) ^ 4 / 4 * μ₄ := by
        field_simp [hε.1.ne']
      rw [hid]
      exact hhi
  have hRsum3 : Tendsto (fun ε : ℝ => (Rlog ε + Rspec ε) / ε ^ 3) (𝓝[>] 0) (𝓝 0) := by
    simpa [add_div] using hRlog3.add hRspec3
  have hRsum2 : Tendsto (fun ε : ℝ => (Rlog ε + Rspec ε) / ε ^ 2) (𝓝[>] 0) (𝓝 0) := by
    have ht := hRsum3.mul (tendsto_id.mono_left nhdsWithin_le_nhds :
      Tendsto (fun ε : ℝ => ε) (𝓝[>] 0) (𝓝 0))
    simp only [mul_zero] at ht
    apply ht.congr'
    filter_upwards [self_mem_nhdsWithin] with ε hε
    change 0 < ε at hε
    change (Rlog ε + Rspec ε) / ε ^ 3 * ε = (Rlog ε + Rspec ε) / ε ^ 2
    field_simp [hε.ne']
  -- Order ε² identity
  have h2_lim : Tendsto (fun ε : ℝ =>
      (Rlog ε + Rspec ε) / ε ^ 2 +
        ((spectralZ ε (D ε) - ε) / ε ^ 2 -
          (spectralZ ε (D ε) / ε) ^ 2 * (μ₂ / 2) - 1 / 2 +
          (spectralZ ε (D ε) / ε) ^ 3 * (μ₃ / 3) * ε - ε / 3))
      (𝓝[>] 0)
      (𝓝 (0 + ((A + 2) - 1 ^ 2 * (μ₂ / 2) - 1 / 2 + 1 ^ 3 * (μ₃ / 3) * 0 - 0 / 3))) := by
    refine hRsum2.add ?_
    refine (((hz2.sub ((hz1.pow 2).mul_const (μ₂ / 2))).sub tendsto_const_nhds).add ?_).sub ?_
    · exact ((hz1.pow 3).mul_const (μ₃ / 3)).mul (tendsto_id.mono_left nhdsWithin_le_nhds)
    · exact (tendsto_id.mono_left nhdsWithin_le_nhds).div_const 3
  have h2_zero : Tendsto (fun ε : ℝ =>
      (Rlog ε + Rspec ε) / ε ^ 2 +
        ((spectralZ ε (D ε) - ε) / ε ^ 2 -
          (spectralZ ε (D ε) / ε) ^ 2 * (μ₂ / 2) - 1 / 2 +
          (spectralZ ε (D ε) / ε) ^ 3 * (μ₃ / 3) * ε - ε / 3))
      (𝓝[>] 0) (𝓝 0) := by
    apply tendsto_const_nhds.congr'
    filter_upwards [hroot, self_mem_nhdsWithin] with ε hε_root hε
    change 0 < ε at hε
    dsimp [Rlog, Rspec]
    rw [hμ1]
    have hsum_num : Real.log (1 - ε) + ε + ε ^ 2 / 2 + ε ^ 3 / 3 +
        ((∫ x, Real.log (1 + spectralZ ε (D ε) * X x) ∂ν) - spectralZ ε (D ε) * 1 +
          spectralZ ε (D ε) ^ 2 / 2 * μ₂ - spectralZ ε (D ε) ^ 3 / 3 * μ₃) =
        -(spectralZ ε (D ε) - ε) + (spectralZ ε (D ε) ^ 2 * (μ₂ / 2) + ε ^ 2 / 2) -
          (spectralZ ε (D ε) ^ 3 * (μ₃ / 3) - ε ^ 3 / 3) := by
      linear_combination hε_root
    rw [hsum_num]
    field_simp [hε.ne']
    ring
  have heq2_raw := tendsto_nhds_unique h2_lim h2_zero
  have heq2 : A + 2 - μ₂ / 2 - 1 / 2 = 0 := by linarith
  -- Order ε³ identity using heq2
  have h3_lim : Tendsto (fun ε : ℝ =>
      (Rlog ε + Rspec ε) / ε ^ 3 +
        ((spectralZ ε (D ε) - ε - (A + 2) * ε ^ 2) / ε ^ 3 -
          μ₂ / 2 * ((spectralZ ε (D ε) / ε + 1) *
            ((spectralZ ε (D ε) - ε) / ε ^ 2)) +
          (spectralZ ε (D ε) / ε) ^ 3 * (μ₃ / 3) - 1 / 3))
      (𝓝[>] 0)
      (𝓝 (0 + ((B + A ^ 2 / 2 + 2 * A + 3) -
        μ₂ / 2 * ((1 + 1) * (A + 2)) + 1 ^ 3 * (μ₃ / 3) - 1 / 3))) := by
    refine hRsum3.add ?_
    refine ((hz3.sub ?_).add ((hz1.pow 3).mul_const (μ₃ / 3))).sub tendsto_const_nhds
    exact ((hz1.add tendsto_const_nhds).mul hz2).const_mul (μ₂ / 2)
  have h3_zero : Tendsto (fun ε : ℝ =>
      (Rlog ε + Rspec ε) / ε ^ 3 +
        ((spectralZ ε (D ε) - ε - (A + 2) * ε ^ 2) / ε ^ 3 -
          μ₂ / 2 * ((spectralZ ε (D ε) / ε + 1) *
            ((spectralZ ε (D ε) - ε) / ε ^ 2)) +
          (spectralZ ε (D ε) / ε) ^ 3 * (μ₃ / 3) - 1 / 3))
      (𝓝[>] 0) (𝓝 0) := by
    apply tendsto_const_nhds.congr'
    filter_upwards [hroot, self_mem_nhdsWithin] with ε hε_root hε
    change 0 < ε at hε
    dsimp [Rlog, Rspec]
    rw [hμ1]
    have hsum_num3 : Real.log (1 - ε) + ε + ε ^ 2 / 2 + ε ^ 3 / 3 +
        ((∫ x, Real.log (1 + spectralZ ε (D ε) * X x) ∂ν) - spectralZ ε (D ε) * 1 +
          spectralZ ε (D ε) ^ 2 / 2 * μ₂ - spectralZ ε (D ε) ^ 3 / 3 * μ₃) =
        -(spectralZ ε (D ε) - ε - (A + 2) * ε ^ 2) +
          μ₂ / 2 * (spectralZ ε (D ε) ^ 2 - ε ^ 2) -
          (spectralZ ε (D ε) ^ 3 * (μ₃ / 3) - ε ^ 3 / 3) := by
      linear_combination hε_root - ε ^ 2 * heq2
    rw [hsum_num3]
    field_simp [hε.ne']
    ring
  have heq3_raw := tendsto_nhds_unique h3_lim h3_zero
  have heq3 : B + A ^ 2 / 2 + 2 * A + 3 - μ₂ * (A + 2) + μ₃ / 3 - 1 / 3 = 0 := by linarith
  exact ⟨heq2, heq3⟩

end SpectralMovingRoot

/-! ## Section 3: Conditional coefficient corollaries -/

section Corollaries

/-- For a supplied positive integral law with unit first moment and
integrable powers through order four, the root equation and assumed right
jet limits imply `A = (μ₂ - 3) / 2` and
`B = 3 A² / 2 + 5 A + 10 / 3 - μ₃ / 3`. Jet existence is not proved. -/
theorem threshold_coefficients_unconditional {α : Type*} [MeasurableSpace α]
    (ν : Measure α) (X : α → ℝ)
    (hX : ∀ᵐ x ∂ν, 0 ≤ X x)
    (h1 : Integrable X ν) (h2 : Integrable (fun x => X x ^ 2) ν)
    (h3 : Integrable (fun x => X x ^ 3) ν) (h4 : Integrable (fun x => X x ^ 4) ν)
    (hlog : ∀ z : ℝ, 0 ≤ z → Integrable (fun x => Real.log (1 + z * X x)) ν)
    (hμ1 : (∫ x, X x ∂ν) = 1)
    (D : ℝ → ℝ) (A B : ℝ)
    (hroot : ∀ᶠ ε in 𝓝[>] (0 : ℝ),
      Real.log (1 - ε) + (∫ x, Real.log (1 + spectralZ ε (D ε) * X x) ∂ν) = 0)
    (hD1 : Tendsto (fun ε : ℝ => D ε / ε) (𝓝[>] 0) (𝓝 A))
    (hD2 : Tendsto (fun ε : ℝ => (D ε - A * ε) / ε ^ 2) (𝓝[>] 0) (𝓝 B)) :
    let μ₂ := ∫ x, X x ^ 2 ∂ν
    let μ₃ := ∫ x, X x ^ 3 ∂ν
    A = (μ₂ - 3) / 2 ∧ B = 3 * A ^ 2 / 2 + 5 * A + 10 / 3 - μ₃ / 3 := by
  intro μ₂ μ₃
  obtain ⟨h₂, h₃⟩ := spectral_threshold_jet_equations ν X hX h1 h2 h3 h4 hlog hμ1 D A B
    hroot hD1 hD2
  exact threshold_coefficients μ₂ μ₃ A B h₂ h₃

/-- Centered coefficient formula conditional on `hμ2_id`, `hμ3_id`,
the spectral root equation, and the jet-existence limits `hD1`, `hD2`.
The local moment expansions yield `A = V / 2 + C₁` and
`B = -A - C₂ - T₁₂ - T₂₁ - T₁₁₁ - m₃ / 3 + 3 A² / 2`.
These premises do not establish the physical spectral/growth identification. -/
theorem centered_threshold_coefficients_unconditional
    {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α]
    (P : Measure Ω) [IsProbabilityMeasure P] (T : Ω ≃ᵐ Ω) (hT : MeasurePreserving T P P)
    (b : Ω → ℝ) (hb : MemLp b 4 P) (hb0 : (∫ ω, b ω ∂P) = 0) (hpos : ∀ ω, 0 ≤ 1 + b ω)
    (ν : Measure α) (X : α → ℝ)
    (hX : ∀ᵐ x ∂ν, 0 ≤ X x)
    (h1 : Integrable X ν) (h2 : Integrable (fun x => X x ^ 2) ν)
    (h3 : Integrable (fun x => X x ^ 3) ν) (h4 : Integrable (fun x => X x ^ 4) ν)
    (hlog : ∀ z : ℝ, 0 ≤ z → Integrable (fun x => Real.log (1 + z * X x)) ν)
    (hμ1 : (∫ x, X x ∂ν) = 1)
    (hμ2_id : (∫ x, X x ^ 2 ∂ν) = localMoment P T (fun ω => 1 + b ω) 2)
    (hμ3_id : (∫ x, X x ^ 3 ∂ν) = localMoment P T (fun ω => 1 + b ω) 3)
    (D : ℝ → ℝ) (A B : ℝ)
    (hroot : ∀ᶠ ε in 𝓝[>] (0 : ℝ),
      Real.log (1 - ε) + (∫ x, Real.log (1 + spectralZ ε (D ε) * X x) ∂ν) = 0)
    (hD1 : Tendsto (fun ε : ℝ => D ε / ε) (𝓝[>] 0) (𝓝 A))
    (hD2 : Tendsto (fun ε : ℝ => (D ε - A * ε) / ε ^ 2) (𝓝[>] 0) (𝓝 B)) :
    A = variance P b / 2 + correlation P T b 1 ∧
      B = -A - correlation P T b 2 - triple12 P T b - triple21 P T b -
        triple111 P T b - thirdMoment P b / 3 + 3 * A ^ 2 / 2 := by
  obtain ⟨h₂, h₃⟩ := spectral_threshold_jet_equations ν X hX h1 h2 h3 h4 hlog hμ1 D A B
    hroot hD1 hD2
  have hm₂ : (∫ x, X x ^ 2 ∂ν) = 3 + variance P b + 2 * correlation P T b 1 :=
    hμ2_id.trans (localMoment_two_centered P T hT b hb hb0 hpos)
  have hm₃ : (∫ x, X x ^ 3 ∂ν) =
      10 + 9 * variance P b + 18 * correlation P T b 1 + 3 * correlation P T b 2 +
        3 * triple12 P T b + 3 * triple21 P T b + 3 * triple111 P T b + thirdMoment P b :=
    hμ3_id.trans (localMoment_three_centered P T hT b hb hb0 hpos)
  exact centered_threshold_coefficients
    (variance P b) (correlation P T b 1) (correlation P T b 2) (thirdMoment P b)
    (triple12 P T b) (triple21 P T b) (triple111 P T b)
    (∫ x, X x ^ 2 ∂ν) (∫ x, X x ^ 3 ∂ν) A B hm₂ hm₃ h₂ h₃

end Corollaries

end Dobson.P1bB_ExactMemLong

#print axioms Dobson.P1bB_ExactMemLong.localMoment_two_centered
#print axioms Dobson.P1bB_ExactMemLong.localMoment_three_centered
#print axioms Dobson.P1bB_ExactMemLong.spectral_threshold_jet_equations
#print axioms Dobson.P1bB_ExactMemLong.threshold_coefficients_unconditional
#print axioms Dobson.P1bB_ExactMemLong.centered_threshold_coefficients_unconditional
