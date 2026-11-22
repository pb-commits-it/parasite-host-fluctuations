import DobsonProofs.GapOneTail
import DobsonProofs.GapFourConstants

/-!
# GAP 1: the extended interval and a small-amplitude region

Finite recurrence (6)-(7), exact coefficient bounds through degree twelve,
the uniform endpoint sqrt(85/98), and the squared small-amplitude condition.
The probability and holomorphic pullback inputs remain explicit hypotheses.
In particular the finite recurrence alone does not identify actual Taylor
coefficients: that identification is included in the two remainder hypotheses.
The real inverse-function calculation is reused from `GapOneTail`.

2026-11-16, auditfix1, IDs 94-102: the degree-12 remainders (94, 97),
stationary chord bounds (95, 96), amplitude remainders (98, 100, 102), and
strict threshold gaps (99, 101) are results rather than model definitions.
Written proofs are in gap1rest/gap1rest.md, equations (5), (11), (14), and
D-exact.tex, prop:exact-mem-bounds. Finite coefficient calculations do not
construct the analytic pullback or identify its Taylor coefficients. The
strict Jensen result in Coverage6ExactBounds also requires a supplied
stationary threshold model and does not instantiate this module's abstract
functions. These nine bridges remain unproved here; the dependent extended
interval and physical small-amplitude conclusions have partial coverage.
-/
namespace Dobson.GapOneExtended
open Filter Set
open scoped Topology
open Dobson.GapOneTail (Gfun derivative_identity)

noncomputable def zSign (j : Bool) : ℝ := if j then 1 else -1

/-- Conditional moments in the deterministic b normalization, equation (6). -/
noncomputable def M_rec (δ φ : ℝ) : ℕ → ℕ → Bool → ℝ
  | 0, 0, _ => 1
  | 0, _ + 1, _ => 0
  | k + 1, n, j =>
      if k + 1 ≤ n then
        ∑ r ∈ Finset.range (k + 2), ∑ m ∈ Finset.range (n - k),
          if k + 1 - r + m ≤ n - (k + 1) then
            (Nat.choose (k + 1) r : ℝ) * (-1 : ℝ) ^ (k + 1 - r + m) *
              (Nat.choose (k + m) m : ℝ) * (δ * zSign j) ^ r *
                (((1 + φ) / 2) * M_rec δ φ (k + 1 - r + m) (n - (k + 1)) j +
                 ((1 - φ) / 2) * M_rec δ φ (k + 1 - r + m) (n - (k + 1)) (!j))
          else 0
      else 0
termination_by _ n _ => n

noncomputable def r_rec (δ φ : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0 else 1 / (n : ℝ) + (1 / 2 : ℝ) *
    ∑ j ∈ Finset.range n, ((-1 : ℝ) ^ (j + 2) / ((j + 1 : ℕ) : ℝ)) *
      (M_rec δ φ (j + 1) n false + M_rec δ φ (j + 1) n true)

noncomputable def e_rec (δ φ : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => -(1 / ((n + 1 : ℕ) : ℝ)) * ∑ j ∈ Finset.range (n + 1),
      ((j + 1 : ℕ) : ℝ) * r_rec δ φ (j + 1) * e_rec δ φ (n - j)
termination_by n => n

noncomputable def v_rec (δ φ : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * r_rec δ φ n + 2 * ∑ k ∈ Finset.range n,
    (-1 : ℝ) ^ (n - k) * (k : ℝ) * r_rec δ φ k

noncomputable def g_rec (δ φ : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0 else -e_rec δ φ n - 2 * v_rec δ φ n +
    ∑ j ∈ Finset.range n, e_rec δ φ (n - 1 - j) * v_rec δ φ (j + 1)

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000


private lemma M_0_0_false (δ φ : ℝ) : M_rec δ φ 0 0 false = 1 := by rw [M_rec]

private lemma M_0_0_true (δ φ : ℝ) : M_rec δ φ 0 0 true = 1 := by rw [M_rec]

private lemma M_1_0_false (δ φ : ℝ) : M_rec δ φ 0 1 false = 0 := by rw [M_rec]

private lemma M_1_0_true (δ φ : ℝ) : M_rec δ φ 0 1 true = 0 := by rw [M_rec]

private lemma M_2_0_false (δ φ : ℝ) : M_rec δ φ 0 2 false = 0 := by rw [M_rec]

private lemma M_2_0_true (δ φ : ℝ) : M_rec δ φ 0 2 true = 0 := by rw [M_rec]

private lemma M_3_0_false (δ φ : ℝ) : M_rec δ φ 0 3 false = 0 := by rw [M_rec]

private lemma M_3_0_true (δ φ : ℝ) : M_rec δ φ 0 3 true = 0 := by rw [M_rec]

private lemma M_4_0_false (δ φ : ℝ) : M_rec δ φ 0 4 false = 0 := by rw [M_rec]

private lemma M_4_0_true (δ φ : ℝ) : M_rec δ φ 0 4 true = 0 := by rw [M_rec]

private lemma M_5_0_false (δ φ : ℝ) : M_rec δ φ 0 5 false = 0 := by rw [M_rec]

private lemma M_5_0_true (δ φ : ℝ) : M_rec δ φ 0 5 true = 0 := by rw [M_rec]

private lemma M_6_0_false (δ φ : ℝ) : M_rec δ φ 0 6 false = 0 := by rw [M_rec]

private lemma M_6_0_true (δ φ : ℝ) : M_rec δ φ 0 6 true = 0 := by rw [M_rec]

private lemma M_7_0_false (δ φ : ℝ) : M_rec δ φ 0 7 false = 0 := by rw [M_rec]

private lemma M_7_0_true (δ φ : ℝ) : M_rec δ φ 0 7 true = 0 := by rw [M_rec]

private lemma M_8_0_false (δ φ : ℝ) : M_rec δ φ 0 8 false = 0 := by rw [M_rec]

private lemma M_8_0_true (δ φ : ℝ) : M_rec δ φ 0 8 true = 0 := by rw [M_rec]

private lemma M_9_0_false (δ φ : ℝ) : M_rec δ φ 0 9 false = 0 := by rw [M_rec]

private lemma M_9_0_true (δ φ : ℝ) : M_rec δ φ 0 9 true = 0 := by rw [M_rec]

private lemma M_10_0_false (δ φ : ℝ) : M_rec δ φ 0 10 false = 0 := by rw [M_rec]

private lemma M_10_0_true (δ φ : ℝ) : M_rec δ φ 0 10 true = 0 := by rw [M_rec]

private lemma M_11_0_false (δ φ : ℝ) : M_rec δ φ 0 11 false = 0 := by rw [M_rec]

private lemma M_11_0_true (δ φ : ℝ) : M_rec δ φ 0 11 true = 0 := by rw [M_rec]

private lemma M_1_1_false (δ φ : ℝ) :
    M_rec δ φ 1 1 false =
      (-1 : ℝ) * δ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_1_1_true (δ φ : ℝ) :
    M_rec δ φ 1 1 true =
      (1 : ℝ) * δ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_2_1_false (δ φ : ℝ) :
    M_rec δ φ 1 2 false =
      (-1 : ℝ) * δ ^ 2 * φ + (1 : ℝ) * δ * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_2_1_true (δ φ : ℝ) :
    M_rec δ φ 1 2 true =
      (-1 : ℝ) * δ ^ 2 * φ + (-1 : ℝ) * δ * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_2_2_false (δ φ : ℝ) :
    M_rec δ φ 2 2 false =
      (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_2_2_true (δ φ : ℝ) :
    M_rec δ φ 2 2 true =
      (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_3_1_false (δ φ : ℝ) :
    M_rec δ φ 1 3 false =
      (-1 : ℝ) * δ ^ 3 * φ + (-1 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 2 + (1 : ℝ) * δ ^ 2 * φ
      + (1 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_3_1_true (δ φ : ℝ) :
    M_rec δ φ 1 3 true =
      (1 : ℝ) * δ ^ 3 * φ + (1 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 2 + (1 : ℝ) * δ ^ 2 * φ
      + (1 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_3_2_false (δ φ : ℝ) :
    M_rec δ φ 2 3 false =
      (2 : ℝ) * δ ^ 3 * φ + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_3_2_true (δ φ : ℝ) :
    M_rec δ φ 2 3 true =
      (-2 : ℝ) * δ ^ 3 * φ + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_3_3_false (δ φ : ℝ) :
    M_rec δ φ 3 3 false =
      (-1 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_3_3_true (δ φ : ℝ) :
    M_rec δ φ 3 3 true =
      (1 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_4_1_false (δ φ : ℝ) :
    M_rec δ φ 1 4 false =
      (-3 : ℝ) * δ ^ 4 * φ ^ 2 + (-2 : ℝ) * δ ^ 4 * φ + (4 : ℝ) * δ ^ 3 * φ ^ 2
      + (5 : ℝ) * δ ^ 3 * φ + (1 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 3
      + (-1 : ℝ) * δ ^ 2 * φ ^ 2 + (-3 : ℝ) * δ ^ 2 * φ + (-1 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_4_1_true (δ φ : ℝ) :
    M_rec δ φ 1 4 true =
      (-3 : ℝ) * δ ^ 4 * φ ^ 2 + (-2 : ℝ) * δ ^ 4 * φ + (-4 : ℝ) * δ ^ 3 * φ ^ 2
      + (-5 : ℝ) * δ ^ 3 * φ + (-1 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 3
      + (-1 : ℝ) * δ ^ 2 * φ ^ 2 + (-3 : ℝ) * δ ^ 2 * φ + (-1 : ℝ) * δ ^ 2
      + (-1 : ℝ) * δ * φ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_4_2_false (δ φ : ℝ) :
    M_rec δ φ 2 4 false =
      (2 : ℝ) * δ ^ 4 * φ + (3 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 2 + (-2 : ℝ) * δ ^ 3 * φ
      + (-4 : ℝ) * δ ^ 3 + (2 : ℝ) * δ ^ 2 * φ ^ 2 + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_4_2_true (δ φ : ℝ) :
    M_rec δ φ 2 4 true =
      (2 : ℝ) * δ ^ 4 * φ + (3 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 2 + (2 : ℝ) * δ ^ 3 * φ
      + (4 : ℝ) * δ ^ 3 + (2 : ℝ) * δ ^ 2 * φ ^ 2 + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_4_3_false (δ φ : ℝ) :
    M_rec δ φ 3 4 false =
      (-3 : ℝ) * δ ^ 4 * φ + (3 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_4_3_true (δ φ : ℝ) :
    M_rec δ φ 3 4 true =
      (-3 : ℝ) * δ ^ 4 * φ + (-3 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_4_4_false (δ φ : ℝ) :
    M_rec δ φ 4 4 false =
      (1 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_4_4_true (δ φ : ℝ) :
    M_rec δ φ 4 4 true =
      (1 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_5_1_false (δ φ : ℝ) :
    M_rec δ φ 1 5 false =
      (-3 : ℝ) * δ ^ 5 * φ ^ 2 + (-7 : ℝ) * δ ^ 5 * φ + (-4 : ℝ) * δ ^ 5
      + (6 : ℝ) * δ ^ 4 * φ ^ 3 + (13 : ℝ) * δ ^ 4 * φ ^ 2 + (12 : ℝ) * δ ^ 4 * φ
      + (4 : ℝ) * δ ^ 4 + (-7 : ℝ) * δ ^ 3 * φ ^ 3 + (-13 : ℝ) * δ ^ 3 * φ ^ 2
      + (-8 : ℝ) * δ ^ 3 * φ + (-2 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 4
      + (1 : ℝ) * δ ^ 2 * φ ^ 3 + (3 : ℝ) * δ ^ 2 * φ ^ 2 + (3 : ℝ) * δ ^ 2 * φ
      + (2 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_5_1_true (δ φ : ℝ) :
    M_rec δ φ 1 5 true =
      (3 : ℝ) * δ ^ 5 * φ ^ 2 + (7 : ℝ) * δ ^ 5 * φ + (4 : ℝ) * δ ^ 5 + (6 : ℝ) * δ ^ 4 * φ ^ 3
      + (13 : ℝ) * δ ^ 4 * φ ^ 2 + (12 : ℝ) * δ ^ 4 * φ + (4 : ℝ) * δ ^ 4
      + (7 : ℝ) * δ ^ 3 * φ ^ 3 + (13 : ℝ) * δ ^ 3 * φ ^ 2 + (8 : ℝ) * δ ^ 3 * φ
      + (2 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 3
      + (3 : ℝ) * δ ^ 2 * φ ^ 2 + (3 : ℝ) * δ ^ 2 * φ + (2 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_5_2_false (δ φ : ℝ) :
    M_rec δ φ 2 5 false =
      (8 : ℝ) * δ ^ 5 * φ ^ 2 + (6 : ℝ) * δ ^ 5 * φ + (-12 : ℝ) * δ ^ 4 * φ ^ 2
      + (-16 : ℝ) * δ ^ 4 * φ + (-2 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 3
      + (4 : ℝ) * δ ^ 3 * φ ^ 2 + (12 : ℝ) * δ ^ 3 * φ + (2 : ℝ) * δ ^ 3
      + (-2 : ℝ) * δ ^ 2 * φ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_5_2_true (δ φ : ℝ) :
    M_rec δ φ 2 5 true =
      (-8 : ℝ) * δ ^ 5 * φ ^ 2 + (-6 : ℝ) * δ ^ 5 * φ + (-12 : ℝ) * δ ^ 4 * φ ^ 2
      + (-16 : ℝ) * δ ^ 4 * φ + (-2 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 3
      + (-4 : ℝ) * δ ^ 3 * φ ^ 2 + (-12 : ℝ) * δ ^ 3 * φ + (-2 : ℝ) * δ ^ 3
      + (-2 : ℝ) * δ ^ 2 * φ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_5_3_false (δ φ : ℝ) :
    M_rec δ φ 3 5 false =
      (-3 : ℝ) * δ ^ 5 * φ + (-6 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 2 + (3 : ℝ) * δ ^ 4 * φ
      + (9 : ℝ) * δ ^ 4 + (-3 : ℝ) * δ ^ 3 * φ ^ 2 + (-3 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_5_3_true (δ φ : ℝ) :
    M_rec δ φ 3 5 true =
      (3 : ℝ) * δ ^ 5 * φ + (6 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 2 + (3 : ℝ) * δ ^ 4 * φ
      + (9 : ℝ) * δ ^ 4 + (3 : ℝ) * δ ^ 3 * φ ^ 2 + (3 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_5_4_false (δ φ : ℝ) :
    M_rec δ φ 4 5 false =
      (4 : ℝ) * δ ^ 5 * φ + (-4 : ℝ) * δ ^ 4 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_5_4_true (δ φ : ℝ) :
    M_rec δ φ 4 5 true =
      (-4 : ℝ) * δ ^ 5 * φ + (-4 : ℝ) * δ ^ 4 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_5_5_false (δ φ : ℝ) :
    M_rec δ φ 5 5 false =
      (-1 : ℝ) * δ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_5_5_true (δ φ : ℝ) :
    M_rec δ φ 5 5 true =
      (1 : ℝ) * δ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_6_1_false (δ φ : ℝ) :
    M_rec δ φ 1 6 false =
      (-11 : ℝ) * δ ^ 6 * φ ^ 3 + (-20 : ℝ) * δ ^ 6 * φ ^ 2 + (-11 : ℝ) * δ ^ 6 * φ
      + (17 : ℝ) * δ ^ 5 * φ ^ 3 + (48 : ℝ) * δ ^ 5 * φ ^ 2 + (46 : ℝ) * δ ^ 5 * φ
      + (15 : ℝ) * δ ^ 5 + (-9 : ℝ) * δ ^ 4 * φ ^ 4 + (-26 : ℝ) * δ ^ 4 * φ ^ 3
      + (-48 : ℝ) * δ ^ 4 * φ ^ 2 + (-42 : ℝ) * δ ^ 4 * φ + (-15 : ℝ) * δ ^ 4
      + (10 : ℝ) * δ ^ 3 * φ ^ 4 + (23 : ℝ) * δ ^ 3 * φ ^ 3 + (23 : ℝ) * δ ^ 3 * φ ^ 2
      + (12 : ℝ) * δ ^ 3 * φ + (2 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 5
      + (-1 : ℝ) * δ ^ 2 * φ ^ 4 + (-3 : ℝ) * δ ^ 2 * φ ^ 3 + (-3 : ℝ) * δ ^ 2 * φ ^ 2
      + (-5 : ℝ) * δ ^ 2 * φ + (-2 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_6_1_true (δ φ : ℝ) :
    M_rec δ φ 1 6 true =
      (-11 : ℝ) * δ ^ 6 * φ ^ 3 + (-20 : ℝ) * δ ^ 6 * φ ^ 2 + (-11 : ℝ) * δ ^ 6 * φ
      + (-17 : ℝ) * δ ^ 5 * φ ^ 3 + (-48 : ℝ) * δ ^ 5 * φ ^ 2 + (-46 : ℝ) * δ ^ 5 * φ
      + (-15 : ℝ) * δ ^ 5 + (-9 : ℝ) * δ ^ 4 * φ ^ 4 + (-26 : ℝ) * δ ^ 4 * φ ^ 3
      + (-48 : ℝ) * δ ^ 4 * φ ^ 2 + (-42 : ℝ) * δ ^ 4 * φ + (-15 : ℝ) * δ ^ 4
      + (-10 : ℝ) * δ ^ 3 * φ ^ 4 + (-23 : ℝ) * δ ^ 3 * φ ^ 3 + (-23 : ℝ) * δ ^ 3 * φ ^ 2
      + (-12 : ℝ) * δ ^ 3 * φ + (-2 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 5
      + (-1 : ℝ) * δ ^ 2 * φ ^ 4 + (-3 : ℝ) * δ ^ 2 * φ ^ 3 + (-3 : ℝ) * δ ^ 2 * φ ^ 2
      + (-5 : ℝ) * δ ^ 2 * φ + (-2 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_6_2_false (δ φ : ℝ) :
    M_rec δ φ 2 6 false =
      (6 : ℝ) * δ ^ 6 * φ ^ 2 + (22 : ℝ) * δ ^ 6 * φ + (14 : ℝ) * δ ^ 6
      + (-14 : ℝ) * δ ^ 5 * φ ^ 3 + (-34 : ℝ) * δ ^ 5 * φ ^ 2 + (-44 : ℝ) * δ ^ 5 * φ
      + (-20 : ℝ) * δ ^ 5 + (18 : ℝ) * δ ^ 4 * φ ^ 3 + (44 : ℝ) * δ ^ 4 * φ ^ 2
      + (32 : ℝ) * δ ^ 4 * φ + (11 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 4
      + (-4 : ℝ) * δ ^ 3 * φ ^ 3 + (-18 : ℝ) * δ ^ 3 * φ ^ 2 + (-10 : ℝ) * δ ^ 3 * φ
      + (-6 : ℝ) * δ ^ 3 + (2 : ℝ) * δ ^ 2 * φ ^ 4 + (2 : ℝ) * δ ^ 2 * φ ^ 2 + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_6_2_true (δ φ : ℝ) :
    M_rec δ φ 2 6 true =
      (6 : ℝ) * δ ^ 6 * φ ^ 2 + (22 : ℝ) * δ ^ 6 * φ + (14 : ℝ) * δ ^ 6
      + (14 : ℝ) * δ ^ 5 * φ ^ 3 + (34 : ℝ) * δ ^ 5 * φ ^ 2 + (44 : ℝ) * δ ^ 5 * φ
      + (20 : ℝ) * δ ^ 5 + (18 : ℝ) * δ ^ 4 * φ ^ 3 + (44 : ℝ) * δ ^ 4 * φ ^ 2
      + (32 : ℝ) * δ ^ 4 * φ + (11 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 4
      + (4 : ℝ) * δ ^ 3 * φ ^ 3 + (18 : ℝ) * δ ^ 3 * φ ^ 2 + (10 : ℝ) * δ ^ 3 * φ
      + (6 : ℝ) * δ ^ 3 + (2 : ℝ) * δ ^ 2 * φ ^ 4 + (2 : ℝ) * δ ^ 2 * φ ^ 2 + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_6_3_false (δ φ : ℝ) :
    M_rec δ φ 3 6 false =
      (-15 : ℝ) * δ ^ 6 * φ ^ 2 + (-13 : ℝ) * δ ^ 6 * φ + (24 : ℝ) * δ ^ 5 * φ ^ 2
      + (36 : ℝ) * δ ^ 5 * φ + (3 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 3
      + (-9 : ℝ) * δ ^ 4 * φ ^ 2 + (-30 : ℝ) * δ ^ 4 * φ + (-3 : ℝ) * δ ^ 4
      + (3 : ℝ) * δ ^ 3 * φ ^ 3 + (7 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_6_3_true (δ φ : ℝ) :
    M_rec δ φ 3 6 true =
      (-15 : ℝ) * δ ^ 6 * φ ^ 2 + (-13 : ℝ) * δ ^ 6 * φ + (-24 : ℝ) * δ ^ 5 * φ ^ 2
      + (-36 : ℝ) * δ ^ 5 * φ + (-3 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 3
      + (-9 : ℝ) * δ ^ 4 * φ ^ 2 + (-30 : ℝ) * δ ^ 4 * φ + (-3 : ℝ) * δ ^ 4
      + (-3 : ℝ) * δ ^ 3 * φ ^ 3 + (-7 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_6_4_false (δ φ : ℝ) :
    M_rec δ φ 4 6 false =
      (4 : ℝ) * δ ^ 6 * φ + (10 : ℝ) * δ ^ 6 + (-4 : ℝ) * δ ^ 5 * φ ^ 2 + (-4 : ℝ) * δ ^ 5 * φ
      + (-16 : ℝ) * δ ^ 5 + (4 : ℝ) * δ ^ 4 * φ ^ 2 + (6 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_6_4_true (δ φ : ℝ) :
    M_rec δ φ 4 6 true =
      (4 : ℝ) * δ ^ 6 * φ + (10 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 2 + (4 : ℝ) * δ ^ 5 * φ
      + (16 : ℝ) * δ ^ 5 + (4 : ℝ) * δ ^ 4 * φ ^ 2 + (6 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_6_5_false (δ φ : ℝ) :
    M_rec δ φ 5 6 false =
      (-5 : ℝ) * δ ^ 6 * φ + (5 : ℝ) * δ ^ 5 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_6_5_true (δ φ : ℝ) :
    M_rec δ φ 5 6 true =
      (-5 : ℝ) * δ ^ 6 * φ + (-5 : ℝ) * δ ^ 5 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_6_6_false (δ φ : ℝ) :
    M_rec δ φ 6 6 false =
      (1 : ℝ) * δ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_6_6_true (δ φ : ℝ) :
    M_rec δ φ 6 6 true =
      (1 : ℝ) * δ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_7_1_false (δ φ : ℝ) :
    M_rec δ φ 1 7 false =
      (-11 : ℝ) * δ ^ 7 * φ ^ 3 + (-41 : ℝ) * δ ^ 7 * φ ^ 2 + (-55 : ℝ) * δ ^ 7 * φ
      + (-25 : ℝ) * δ ^ 7 + (31 : ℝ) * δ ^ 6 * φ ^ 4 + (121 : ℝ) * δ ^ 6 * φ ^ 3
      + (176 : ℝ) * δ ^ 6 * φ ^ 2 + (109 : ℝ) * δ ^ 6 * φ + (25 : ℝ) * δ ^ 6
      + (-40 : ℝ) * δ ^ 5 * φ ^ 4 + (-157 : ℝ) * δ ^ 5 * φ ^ 3 + (-240 : ℝ) * δ ^ 5 * φ ^ 2
      + (-158 : ℝ) * δ ^ 5 * φ + (-35 : ℝ) * δ ^ 5 + (12 : ℝ) * δ ^ 4 * φ ^ 5
      + (39 : ℝ) * δ ^ 4 * φ ^ 4 + (88 : ℝ) * δ ^ 4 * φ ^ 3 + (134 : ℝ) * δ ^ 4 * φ ^ 2
      + (112 : ℝ) * δ ^ 4 * φ + (35 : ℝ) * δ ^ 4 + (-13 : ℝ) * δ ^ 3 * φ ^ 5
      + (-33 : ℝ) * δ ^ 3 * φ ^ 4 + (-44 : ℝ) * δ ^ 3 * φ ^ 3 + (-34 : ℝ) * δ ^ 3 * φ ^ 2
      + (-13 : ℝ) * δ ^ 3 * φ + (-3 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 6
      + (1 : ℝ) * δ ^ 2 * φ ^ 5 + (3 : ℝ) * δ ^ 2 * φ ^ 4 + (3 : ℝ) * δ ^ 2 * φ ^ 3
      + (5 : ℝ) * δ ^ 2 * φ ^ 2 + (5 : ℝ) * δ ^ 2 * φ + (3 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_7_1_true (δ φ : ℝ) :
    M_rec δ φ 1 7 true =
      (11 : ℝ) * δ ^ 7 * φ ^ 3 + (41 : ℝ) * δ ^ 7 * φ ^ 2 + (55 : ℝ) * δ ^ 7 * φ
      + (25 : ℝ) * δ ^ 7 + (31 : ℝ) * δ ^ 6 * φ ^ 4 + (121 : ℝ) * δ ^ 6 * φ ^ 3
      + (176 : ℝ) * δ ^ 6 * φ ^ 2 + (109 : ℝ) * δ ^ 6 * φ + (25 : ℝ) * δ ^ 6
      + (40 : ℝ) * δ ^ 5 * φ ^ 4 + (157 : ℝ) * δ ^ 5 * φ ^ 3 + (240 : ℝ) * δ ^ 5 * φ ^ 2
      + (158 : ℝ) * δ ^ 5 * φ + (35 : ℝ) * δ ^ 5 + (12 : ℝ) * δ ^ 4 * φ ^ 5
      + (39 : ℝ) * δ ^ 4 * φ ^ 4 + (88 : ℝ) * δ ^ 4 * φ ^ 3 + (134 : ℝ) * δ ^ 4 * φ ^ 2
      + (112 : ℝ) * δ ^ 4 * φ + (35 : ℝ) * δ ^ 4 + (13 : ℝ) * δ ^ 3 * φ ^ 5
      + (33 : ℝ) * δ ^ 3 * φ ^ 4 + (44 : ℝ) * δ ^ 3 * φ ^ 3 + (34 : ℝ) * δ ^ 3 * φ ^ 2
      + (13 : ℝ) * δ ^ 3 * φ + (3 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 6
      + (1 : ℝ) * δ ^ 2 * φ ^ 5 + (3 : ℝ) * δ ^ 2 * φ ^ 4 + (3 : ℝ) * δ ^ 2 * φ ^ 3
      + (5 : ℝ) * δ ^ 2 * φ ^ 2 + (5 : ℝ) * δ ^ 2 * φ + (3 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_7_2_false (δ φ : ℝ) :
    M_rec δ φ 2 7 false =
      (30 : ℝ) * δ ^ 7 * φ ^ 3 + (64 : ℝ) * δ ^ 7 * φ ^ 2 + (38 : ℝ) * δ ^ 7 * φ
      + (-50 : ℝ) * δ ^ 6 * φ ^ 3 + (-162 : ℝ) * δ ^ 6 * φ ^ 2 + (-158 : ℝ) * δ ^ 6 * φ
      + (-50 : ℝ) * δ ^ 6 + (20 : ℝ) * δ ^ 5 * φ ^ 4 + (70 : ℝ) * δ ^ 5 * φ ^ 3
      + (168 : ℝ) * δ ^ 5 * φ ^ 2 + (176 : ℝ) * δ ^ 5 * φ + (70 : ℝ) * δ ^ 5
      + (-24 : ℝ) * δ ^ 4 * φ ^ 4 + (-68 : ℝ) * δ ^ 4 * φ ^ 3 + (-88 : ℝ) * δ ^ 4 * φ ^ 2
      + (-76 : ℝ) * δ ^ 4 * φ + (-24 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 5
      + (4 : ℝ) * δ ^ 3 * φ ^ 4 + (20 : ℝ) * δ ^ 3 * φ ^ 3 + (18 : ℝ) * δ ^ 3 * φ ^ 2
      + (22 : ℝ) * δ ^ 3 * φ + (4 : ℝ) * δ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ ^ 5
      + (-2 : ℝ) * δ ^ 2 * φ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_7_2_true (δ φ : ℝ) :
    M_rec δ φ 2 7 true =
      (-30 : ℝ) * δ ^ 7 * φ ^ 3 + (-64 : ℝ) * δ ^ 7 * φ ^ 2 + (-38 : ℝ) * δ ^ 7 * φ
      + (-50 : ℝ) * δ ^ 6 * φ ^ 3 + (-162 : ℝ) * δ ^ 6 * φ ^ 2 + (-158 : ℝ) * δ ^ 6 * φ
      + (-50 : ℝ) * δ ^ 6 + (-20 : ℝ) * δ ^ 5 * φ ^ 4 + (-70 : ℝ) * δ ^ 5 * φ ^ 3
      + (-168 : ℝ) * δ ^ 5 * φ ^ 2 + (-176 : ℝ) * δ ^ 5 * φ + (-70 : ℝ) * δ ^ 5
      + (-24 : ℝ) * δ ^ 4 * φ ^ 4 + (-68 : ℝ) * δ ^ 4 * φ ^ 3 + (-88 : ℝ) * δ ^ 4 * φ ^ 2
      + (-76 : ℝ) * δ ^ 4 * φ + (-24 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 5
      + (-4 : ℝ) * δ ^ 3 * φ ^ 4 + (-20 : ℝ) * δ ^ 3 * φ ^ 3 + (-18 : ℝ) * δ ^ 3 * φ ^ 2
      + (-22 : ℝ) * δ ^ 3 * φ + (-4 : ℝ) * δ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ ^ 5
      + (-2 : ℝ) * δ ^ 2 * φ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_7_3_false (δ φ : ℝ) :
    M_rec δ φ 3 7 false =
      (-9 : ℝ) * δ ^ 7 * φ ^ 2 + (-48 : ℝ) * δ ^ 7 * φ + (-33 : ℝ) * δ ^ 7
      + (24 : ℝ) * δ ^ 6 * φ ^ 3 + (66 : ℝ) * δ ^ 6 * φ ^ 2 + (105 : ℝ) * δ ^ 6 * φ
      + (57 : ℝ) * δ ^ 6 + (-33 : ℝ) * δ ^ 5 * φ ^ 3 + (-102 : ℝ) * δ ^ 5 * φ ^ 2
      + (-81 : ℝ) * δ ^ 5 * φ + (-36 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 4
      + (9 : ℝ) * δ ^ 4 * φ ^ 3 + (54 : ℝ) * δ ^ 4 * φ ^ 2 + (24 : ℝ) * δ ^ 4 * φ
      + (15 : ℝ) * δ ^ 4 + (-3 : ℝ) * δ ^ 3 * φ ^ 4 + (-9 : ℝ) * δ ^ 3 * φ ^ 2
      + (-3 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_7_3_true (δ φ : ℝ) :
    M_rec δ φ 3 7 true =
      (9 : ℝ) * δ ^ 7 * φ ^ 2 + (48 : ℝ) * δ ^ 7 * φ + (33 : ℝ) * δ ^ 7
      + (24 : ℝ) * δ ^ 6 * φ ^ 3 + (66 : ℝ) * δ ^ 6 * φ ^ 2 + (105 : ℝ) * δ ^ 6 * φ
      + (57 : ℝ) * δ ^ 6 + (33 : ℝ) * δ ^ 5 * φ ^ 3 + (102 : ℝ) * δ ^ 5 * φ ^ 2
      + (81 : ℝ) * δ ^ 5 * φ + (36 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 4
      + (9 : ℝ) * δ ^ 4 * φ ^ 3 + (54 : ℝ) * δ ^ 4 * φ ^ 2 + (24 : ℝ) * δ ^ 4 * φ
      + (15 : ℝ) * δ ^ 4 + (3 : ℝ) * δ ^ 3 * φ ^ 4 + (9 : ℝ) * δ ^ 3 * φ ^ 2 + (3 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_7_4_false (δ φ : ℝ) :
    M_rec δ φ 4 7 false =
      (24 : ℝ) * δ ^ 7 * φ ^ 2 + (24 : ℝ) * δ ^ 7 * φ + (-40 : ℝ) * δ ^ 6 * φ ^ 2
      + (-68 : ℝ) * δ ^ 6 * φ + (-4 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 3
      + (16 : ℝ) * δ ^ 5 * φ ^ 2 + (60 : ℝ) * δ ^ 5 * φ + (4 : ℝ) * δ ^ 5
      + (-4 : ℝ) * δ ^ 4 * φ ^ 3 + (-16 : ℝ) * δ ^ 4 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_7_4_true (δ φ : ℝ) :
    M_rec δ φ 4 7 true =
      (-24 : ℝ) * δ ^ 7 * φ ^ 2 + (-24 : ℝ) * δ ^ 7 * φ + (-40 : ℝ) * δ ^ 6 * φ ^ 2
      + (-68 : ℝ) * δ ^ 6 * φ + (-4 : ℝ) * δ ^ 6 + (-4 : ℝ) * δ ^ 5 * φ ^ 3
      + (-16 : ℝ) * δ ^ 5 * φ ^ 2 + (-60 : ℝ) * δ ^ 5 * φ + (-4 : ℝ) * δ ^ 5
      + (-4 : ℝ) * δ ^ 4 * φ ^ 3 + (-16 : ℝ) * δ ^ 4 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_7_5_false (δ φ : ℝ) :
    M_rec δ φ 5 7 false =
      (-5 : ℝ) * δ ^ 7 * φ + (-15 : ℝ) * δ ^ 7 + (5 : ℝ) * δ ^ 6 * φ ^ 2 + (5 : ℝ) * δ ^ 6 * φ
      + (25 : ℝ) * δ ^ 6 + (-5 : ℝ) * δ ^ 5 * φ ^ 2 + (-10 : ℝ) * δ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_7_5_true (δ φ : ℝ) :
    M_rec δ φ 5 7 true =
      (5 : ℝ) * δ ^ 7 * φ + (15 : ℝ) * δ ^ 7 + (5 : ℝ) * δ ^ 6 * φ ^ 2 + (5 : ℝ) * δ ^ 6 * φ
      + (25 : ℝ) * δ ^ 6 + (5 : ℝ) * δ ^ 5 * φ ^ 2 + (10 : ℝ) * δ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_7_6_false (δ φ : ℝ) :
    M_rec δ φ 6 7 false =
      (6 : ℝ) * δ ^ 7 * φ + (-6 : ℝ) * δ ^ 6 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_7_6_true (δ φ : ℝ) :
    M_rec δ φ 6 7 true =
      (-6 : ℝ) * δ ^ 7 * φ + (-6 : ℝ) * δ ^ 6 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_7_7_false (δ φ : ℝ) :
    M_rec δ φ 7 7 false =
      (-1 : ℝ) * δ ^ 7 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_7_7_true (δ φ : ℝ) :
    M_rec δ φ 7 7 true =
      (1 : ℝ) * δ ^ 7 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_8_1_false (δ φ : ℝ) :
    M_rec δ φ 1 8 false =
      (-41 : ℝ) * δ ^ 8 * φ ^ 4 + (-138 : ℝ) * δ ^ 8 * φ ^ 3 + (-176 : ℝ) * δ ^ 8 * φ ^ 2
      + (-74 : ℝ) * δ ^ 8 * φ + (72 : ℝ) * δ ^ 7 * φ ^ 4 + (333 : ℝ) * δ ^ 7 * φ ^ 3
      + (625 : ℝ) * δ ^ 7 * φ ^ 2 + (525 : ℝ) * δ ^ 7 * φ + (161 : ℝ) * δ ^ 7
      + (-60 : ℝ) * δ ^ 6 * φ ^ 5 + (-295 : ℝ) * δ ^ 6 * φ ^ 4 + (-726 : ℝ) * δ ^ 6 * φ ^ 3
      + (-924 : ℝ) * δ ^ 6 * φ ^ 2 + (-606 : ℝ) * δ ^ 6 * φ + (-161 : ℝ) * δ ^ 6
      + (72 : ℝ) * δ ^ 5 * φ ^ 5 + (330 : ℝ) * δ ^ 5 * φ ^ 4 + (700 : ℝ) * δ ^ 5 * φ ^ 3
      + (751 : ℝ) * δ ^ 5 * φ ^ 2 + (383 : ℝ) * δ ^ 5 * φ + (74 : ℝ) * δ ^ 5
      + (-15 : ℝ) * δ ^ 4 * φ ^ 6 + (-52 : ℝ) * δ ^ 4 * φ ^ 5 + (-130 : ℝ) * δ ^ 4 * φ ^ 4
      + (-230 : ℝ) * δ ^ 4 * φ ^ 3 + (-311 : ℝ) * δ ^ 4 * φ ^ 2 + (-238 : ℝ) * δ ^ 4 * φ
      + (-74 : ℝ) * δ ^ 4 + (16 : ℝ) * δ ^ 3 * φ ^ 6 + (43 : ℝ) * δ ^ 3 * φ ^ 5
      + (67 : ℝ) * δ ^ 3 * φ ^ 4 + (66 : ℝ) * δ ^ 3 * φ ^ 3 + (40 : ℝ) * δ ^ 3 * φ ^ 2
      + (17 : ℝ) * δ ^ 3 * φ + (3 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 7
      + (-1 : ℝ) * δ ^ 2 * φ ^ 6 + (-3 : ℝ) * δ ^ 2 * φ ^ 5 + (-3 : ℝ) * δ ^ 2 * φ ^ 4
      + (-5 : ℝ) * δ ^ 2 * φ ^ 3 + (-5 : ℝ) * δ ^ 2 * φ ^ 2 + (-7 : ℝ) * δ ^ 2 * φ
      + (-3 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 7 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_8_1_true (δ φ : ℝ) :
    M_rec δ φ 1 8 true =
      (-41 : ℝ) * δ ^ 8 * φ ^ 4 + (-138 : ℝ) * δ ^ 8 * φ ^ 3 + (-176 : ℝ) * δ ^ 8 * φ ^ 2
      + (-74 : ℝ) * δ ^ 8 * φ + (-72 : ℝ) * δ ^ 7 * φ ^ 4 + (-333 : ℝ) * δ ^ 7 * φ ^ 3
      + (-625 : ℝ) * δ ^ 7 * φ ^ 2 + (-525 : ℝ) * δ ^ 7 * φ + (-161 : ℝ) * δ ^ 7
      + (-60 : ℝ) * δ ^ 6 * φ ^ 5 + (-295 : ℝ) * δ ^ 6 * φ ^ 4 + (-726 : ℝ) * δ ^ 6 * φ ^ 3
      + (-924 : ℝ) * δ ^ 6 * φ ^ 2 + (-606 : ℝ) * δ ^ 6 * φ + (-161 : ℝ) * δ ^ 6
      + (-72 : ℝ) * δ ^ 5 * φ ^ 5 + (-330 : ℝ) * δ ^ 5 * φ ^ 4 + (-700 : ℝ) * δ ^ 5 * φ ^ 3
      + (-751 : ℝ) * δ ^ 5 * φ ^ 2 + (-383 : ℝ) * δ ^ 5 * φ + (-74 : ℝ) * δ ^ 5
      + (-15 : ℝ) * δ ^ 4 * φ ^ 6 + (-52 : ℝ) * δ ^ 4 * φ ^ 5 + (-130 : ℝ) * δ ^ 4 * φ ^ 4
      + (-230 : ℝ) * δ ^ 4 * φ ^ 3 + (-311 : ℝ) * δ ^ 4 * φ ^ 2 + (-238 : ℝ) * δ ^ 4 * φ
      + (-74 : ℝ) * δ ^ 4 + (-16 : ℝ) * δ ^ 3 * φ ^ 6 + (-43 : ℝ) * δ ^ 3 * φ ^ 5
      + (-67 : ℝ) * δ ^ 3 * φ ^ 4 + (-66 : ℝ) * δ ^ 3 * φ ^ 3 + (-40 : ℝ) * δ ^ 3 * φ ^ 2
      + (-17 : ℝ) * δ ^ 3 * φ + (-3 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 7
      + (-1 : ℝ) * δ ^ 2 * φ ^ 6 + (-3 : ℝ) * δ ^ 2 * φ ^ 5 + (-3 : ℝ) * δ ^ 2 * φ ^ 4
      + (-5 : ℝ) * δ ^ 2 * φ ^ 3 + (-5 : ℝ) * δ ^ 2 * φ ^ 2 + (-7 : ℝ) * δ ^ 2 * φ
      + (-3 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 7 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_8_2_false (δ φ : ℝ) :
    M_rec δ φ 2 8 false =
      (22 : ℝ) * δ ^ 8 * φ ^ 3 + (118 : ℝ) * δ ^ 8 * φ ^ 2 + (190 : ℝ) * δ ^ 8 * φ
      + (99 : ℝ) * δ ^ 8 + (-76 : ℝ) * δ ^ 7 * φ ^ 4 + (-336 : ℝ) * δ ^ 7 * φ ^ 3
      + (-572 : ℝ) * δ ^ 7 * φ ^ 2 + (-452 : ℝ) * δ ^ 7 * φ + (-148 : ℝ) * δ ^ 7
      + (108 : ℝ) * δ ^ 6 * φ ^ 4 + (526 : ℝ) * δ ^ 6 * φ ^ 3 + (886 : ℝ) * δ ^ 6 * φ ^ 2
      + (636 : ℝ) * δ ^ 6 * φ + (154 : ℝ) * δ ^ 6 + (-26 : ℝ) * δ ^ 5 * φ ^ 5
      + (-102 : ℝ) * δ ^ 5 * φ ^ 4 + (-336 : ℝ) * δ ^ 5 * φ ^ 3 + (-588 : ℝ) * δ ^ 5 * φ ^ 2
      + (-488 : ℝ) * δ ^ 5 * φ + (-140 : ℝ) * δ ^ 5 + (30 : ℝ) * δ ^ 4 * φ ^ 5
      + (88 : ℝ) * δ ^ 4 * φ ^ 4 + (148 : ℝ) * δ ^ 4 * φ ^ 3 + (192 : ℝ) * δ ^ 4 * φ ^ 2
      + (130 : ℝ) * δ ^ 4 * φ + (42 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 6
      + (-4 : ℝ) * δ ^ 3 * φ ^ 5 + (-20 : ℝ) * δ ^ 3 * φ ^ 4 + (-24 : ℝ) * δ ^ 3 * φ ^ 3
      + (-38 : ℝ) * δ ^ 3 * φ ^ 2 + (-16 : ℝ) * δ ^ 3 * φ + (-8 : ℝ) * δ ^ 3
      + (2 : ℝ) * δ ^ 2 * φ ^ 6 + (2 : ℝ) * δ ^ 2 * φ ^ 4 + (2 : ℝ) * δ ^ 2 * φ ^ 2
      + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_8_2_true (δ φ : ℝ) :
    M_rec δ φ 2 8 true =
      (22 : ℝ) * δ ^ 8 * φ ^ 3 + (118 : ℝ) * δ ^ 8 * φ ^ 2 + (190 : ℝ) * δ ^ 8 * φ
      + (99 : ℝ) * δ ^ 8 + (76 : ℝ) * δ ^ 7 * φ ^ 4 + (336 : ℝ) * δ ^ 7 * φ ^ 3
      + (572 : ℝ) * δ ^ 7 * φ ^ 2 + (452 : ℝ) * δ ^ 7 * φ + (148 : ℝ) * δ ^ 7
      + (108 : ℝ) * δ ^ 6 * φ ^ 4 + (526 : ℝ) * δ ^ 6 * φ ^ 3 + (886 : ℝ) * δ ^ 6 * φ ^ 2
      + (636 : ℝ) * δ ^ 6 * φ + (154 : ℝ) * δ ^ 6 + (26 : ℝ) * δ ^ 5 * φ ^ 5
      + (102 : ℝ) * δ ^ 5 * φ ^ 4 + (336 : ℝ) * δ ^ 5 * φ ^ 3 + (588 : ℝ) * δ ^ 5 * φ ^ 2
      + (488 : ℝ) * δ ^ 5 * φ + (140 : ℝ) * δ ^ 5 + (30 : ℝ) * δ ^ 4 * φ ^ 5
      + (88 : ℝ) * δ ^ 4 * φ ^ 4 + (148 : ℝ) * δ ^ 4 * φ ^ 3 + (192 : ℝ) * δ ^ 4 * φ ^ 2
      + (130 : ℝ) * δ ^ 4 * φ + (42 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 6
      + (4 : ℝ) * δ ^ 3 * φ ^ 5 + (20 : ℝ) * δ ^ 3 * φ ^ 4 + (24 : ℝ) * δ ^ 3 * φ ^ 3
      + (38 : ℝ) * δ ^ 3 * φ ^ 2 + (16 : ℝ) * δ ^ 3 * φ + (8 : ℝ) * δ ^ 3
      + (2 : ℝ) * δ ^ 2 * φ ^ 6 + (2 : ℝ) * δ ^ 2 * φ ^ 4 + (2 : ℝ) * δ ^ 2 * φ ^ 2
      + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_8_3_false (δ φ : ℝ) :
    M_rec δ φ 3 8 false =
      (-57 : ℝ) * δ ^ 8 * φ ^ 3 + (-147 : ℝ) * δ ^ 8 * φ ^ 2 + (-93 : ℝ) * δ ^ 8 * φ
      + (99 : ℝ) * δ ^ 7 * φ ^ 3 + (390 : ℝ) * δ ^ 7 * φ ^ 2 + (387 : ℝ) * δ ^ 7 * φ
      + (114 : ℝ) * δ ^ 7 + (-33 : ℝ) * δ ^ 6 * φ ^ 4 + (-135 : ℝ) * δ ^ 6 * φ ^ 3
      + (-414 : ℝ) * δ ^ 6 * φ ^ 2 + (-486 : ℝ) * δ ^ 6 * φ + (-192 : ℝ) * δ ^ 6
      + (42 : ℝ) * δ ^ 5 * φ ^ 4 + (144 : ℝ) * δ ^ 5 * φ ^ 3 + (219 : ℝ) * δ ^ 5 * φ ^ 2
      + (258 : ℝ) * δ ^ 5 * φ + (93 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 5
      + (-9 : ℝ) * δ ^ 4 * φ ^ 4 + (-60 : ℝ) * δ ^ 4 * φ ^ 3 + (-48 : ℝ) * δ ^ 4 * φ ^ 2
      + (-75 : ℝ) * δ ^ 4 * φ + (-15 : ℝ) * δ ^ 4 + (3 : ℝ) * δ ^ 3 * φ ^ 5
      + (9 : ℝ) * δ ^ 3 * φ ^ 3 + (9 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_8_3_true (δ φ : ℝ) :
    M_rec δ φ 3 8 true =
      (-57 : ℝ) * δ ^ 8 * φ ^ 3 + (-147 : ℝ) * δ ^ 8 * φ ^ 2 + (-93 : ℝ) * δ ^ 8 * φ
      + (-99 : ℝ) * δ ^ 7 * φ ^ 3 + (-390 : ℝ) * δ ^ 7 * φ ^ 2 + (-387 : ℝ) * δ ^ 7 * φ
      + (-114 : ℝ) * δ ^ 7 + (-33 : ℝ) * δ ^ 6 * φ ^ 4 + (-135 : ℝ) * δ ^ 6 * φ ^ 3
      + (-414 : ℝ) * δ ^ 6 * φ ^ 2 + (-486 : ℝ) * δ ^ 6 * φ + (-192 : ℝ) * δ ^ 6
      + (-42 : ℝ) * δ ^ 5 * φ ^ 4 + (-144 : ℝ) * δ ^ 5 * φ ^ 3 + (-219 : ℝ) * δ ^ 5 * φ ^ 2
      + (-258 : ℝ) * δ ^ 5 * φ + (-93 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 5
      + (-9 : ℝ) * δ ^ 4 * φ ^ 4 + (-60 : ℝ) * δ ^ 4 * φ ^ 3 + (-48 : ℝ) * δ ^ 4 * φ ^ 2
      + (-75 : ℝ) * δ ^ 4 * φ + (-15 : ℝ) * δ ^ 4 + (-3 : ℝ) * δ ^ 3 * φ ^ 5
      + (-9 : ℝ) * δ ^ 3 * φ ^ 3 + (-9 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_8_4_false (δ φ : ℝ) :
    M_rec δ φ 4 8 false =
      (12 : ℝ) * δ ^ 8 * φ ^ 2 + (88 : ℝ) * δ ^ 8 * φ + (65 : ℝ) * δ ^ 8
      + (-36 : ℝ) * δ ^ 7 * φ ^ 3 + (-112 : ℝ) * δ ^ 7 * φ ^ 2 + (-204 : ℝ) * δ ^ 7 * φ
      + (-128 : ℝ) * δ ^ 7 + (52 : ℝ) * δ ^ 6 * φ ^ 3 + (196 : ℝ) * δ ^ 6 * φ ^ 2
      + (164 : ℝ) * δ ^ 6 * φ + (92 : ℝ) * δ ^ 6 + (-4 : ℝ) * δ ^ 5 * φ ^ 4
      + (-16 : ℝ) * δ ^ 5 * φ ^ 3 + (-120 : ℝ) * δ ^ 5 * φ ^ 2 + (-48 : ℝ) * δ ^ 5 * φ
      + (-36 : ℝ) * δ ^ 5 + (4 : ℝ) * δ ^ 4 * φ ^ 4 + (24 : ℝ) * δ ^ 4 * φ ^ 2 + (7 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_8_4_true (δ φ : ℝ) :
    M_rec δ φ 4 8 true =
      (12 : ℝ) * δ ^ 8 * φ ^ 2 + (88 : ℝ) * δ ^ 8 * φ + (65 : ℝ) * δ ^ 8
      + (36 : ℝ) * δ ^ 7 * φ ^ 3 + (112 : ℝ) * δ ^ 7 * φ ^ 2 + (204 : ℝ) * δ ^ 7 * φ
      + (128 : ℝ) * δ ^ 7 + (52 : ℝ) * δ ^ 6 * φ ^ 3 + (196 : ℝ) * δ ^ 6 * φ ^ 2
      + (164 : ℝ) * δ ^ 6 * φ + (92 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 4
      + (16 : ℝ) * δ ^ 5 * φ ^ 3 + (120 : ℝ) * δ ^ 5 * φ ^ 2 + (48 : ℝ) * δ ^ 5 * φ
      + (36 : ℝ) * δ ^ 5 + (4 : ℝ) * δ ^ 4 * φ ^ 4 + (24 : ℝ) * δ ^ 4 * φ ^ 2 + (7 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_8_5_false (δ φ : ℝ) :
    M_rec δ φ 5 8 false =
      (-35 : ℝ) * δ ^ 8 * φ ^ 2 + (-40 : ℝ) * δ ^ 8 * φ + (60 : ℝ) * δ ^ 7 * φ ^ 2
      + (115 : ℝ) * δ ^ 7 * φ + (5 : ℝ) * δ ^ 7 + (-5 : ℝ) * δ ^ 6 * φ ^ 3
      + (-25 : ℝ) * δ ^ 6 * φ ^ 2 + (-105 : ℝ) * δ ^ 6 * φ + (-5 : ℝ) * δ ^ 6
      + (5 : ℝ) * δ ^ 5 * φ ^ 3 + (30 : ℝ) * δ ^ 5 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_8_5_true (δ φ : ℝ) :
    M_rec δ φ 5 8 true =
      (-35 : ℝ) * δ ^ 8 * φ ^ 2 + (-40 : ℝ) * δ ^ 8 * φ + (-60 : ℝ) * δ ^ 7 * φ ^ 2
      + (-115 : ℝ) * δ ^ 7 * φ + (-5 : ℝ) * δ ^ 7 + (-5 : ℝ) * δ ^ 6 * φ ^ 3
      + (-25 : ℝ) * δ ^ 6 * φ ^ 2 + (-105 : ℝ) * δ ^ 6 * φ + (-5 : ℝ) * δ ^ 6
      + (-5 : ℝ) * δ ^ 5 * φ ^ 3 + (-30 : ℝ) * δ ^ 5 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_8_6_false (δ φ : ℝ) :
    M_rec δ φ 6 8 false =
      (6 : ℝ) * δ ^ 8 * φ + (21 : ℝ) * δ ^ 8 + (-6 : ℝ) * δ ^ 7 * φ ^ 2 + (-6 : ℝ) * δ ^ 7 * φ
      + (-36 : ℝ) * δ ^ 7 + (6 : ℝ) * δ ^ 6 * φ ^ 2 + (15 : ℝ) * δ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_8_6_true (δ φ : ℝ) :
    M_rec δ φ 6 8 true =
      (6 : ℝ) * δ ^ 8 * φ + (21 : ℝ) * δ ^ 8 + (6 : ℝ) * δ ^ 7 * φ ^ 2 + (6 : ℝ) * δ ^ 7 * φ
      + (36 : ℝ) * δ ^ 7 + (6 : ℝ) * δ ^ 6 * φ ^ 2 + (15 : ℝ) * δ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_8_7_false (δ φ : ℝ) :
    M_rec δ φ 7 8 false =
      (-7 : ℝ) * δ ^ 8 * φ + (7 : ℝ) * δ ^ 7 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_8_7_true (δ φ : ℝ) :
    M_rec δ φ 7 8 true =
      (-7 : ℝ) * δ ^ 8 * φ + (-7 : ℝ) * δ ^ 7 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_8_8_false (δ φ : ℝ) :
    M_rec δ φ 8 8 false =
      (1 : ℝ) * δ ^ 8 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_8_8_true (δ φ : ℝ) :
    M_rec δ φ 8 8 true =
      (1 : ℝ) * δ ^ 8 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_9_1_false (δ φ : ℝ) :
    M_rec δ φ 1 9 false =
      (-41 : ℝ) * δ ^ 9 * φ ^ 4 + (-217 : ℝ) * δ ^ 9 * φ ^ 3 + (-488 : ℝ) * δ ^ 9 * φ ^ 2
      + (-498 : ℝ) * δ ^ 9 * φ + (-186 : ℝ) * δ ^ 9 + (148 : ℝ) * δ ^ 8 * φ ^ 5
      + (845 : ℝ) * δ ^ 8 * φ ^ 4 + (1982 : ℝ) * δ ^ 8 * φ ^ 3 + (2184 : ℝ) * δ ^ 8 * φ ^ 2
      + (1090 : ℝ) * δ ^ 8 * φ + (186 : ℝ) * δ ^ 8 + (-208 : ℝ) * δ ^ 7 * φ ^ 5
      + (-1240 : ℝ) * δ ^ 7 * φ ^ 4 + (-3209 : ℝ) * δ ^ 7 * φ ^ 3 + (-4147 : ℝ) * δ ^ 7 * φ ^ 2
      + (-2589 : ℝ) * δ ^ 7 * φ + (-619 : ℝ) * δ ^ 7 + (98 : ℝ) * δ ^ 6 * φ ^ 6
      + (538 : ℝ) * δ ^ 6 * φ ^ 5 + (1637 : ℝ) * δ ^ 6 * φ ^ 4 + (3122 : ℝ) * δ ^ 6 * φ ^ 3
      + (3658 : ℝ) * δ ^ 6 * φ ^ 2 + (2340 : ℝ) * δ ^ 6 * φ + (619 : ℝ) * δ ^ 6
      + (-113 : ℝ) * δ ^ 5 * φ ^ 6 + (-563 : ℝ) * δ ^ 5 * φ ^ 5 + (-1432 : ℝ) * δ ^ 5 * φ ^ 4
      + (-2116 : ℝ) * δ ^ 5 * φ ^ 3 + (-1782 : ℝ) * δ ^ 5 * φ ^ 2 + (-786 : ℝ) * δ ^ 5 * φ
      + (-138 : ℝ) * δ ^ 5 + (18 : ℝ) * δ ^ 4 * φ ^ 7 + (65 : ℝ) * δ ^ 4 * φ ^ 6
      + (172 : ℝ) * δ ^ 4 * φ ^ 5 + (330 : ℝ) * δ ^ 4 * φ ^ 4 + (516 : ℝ) * δ ^ 4 * φ ^ 3
      + (617 : ℝ) * δ ^ 4 * φ ^ 2 + (454 : ℝ) * δ ^ 4 * φ + (138 : ℝ) * δ ^ 4
      + (-19 : ℝ) * δ ^ 3 * φ ^ 7 + (-53 : ℝ) * δ ^ 3 * φ ^ 6 + (-90 : ℝ) * δ ^ 3 * φ ^ 5
      + (-104 : ℝ) * δ ^ 3 * φ ^ 4 + (-83 : ℝ) * δ ^ 3 * φ ^ 3 + (-49 : ℝ) * δ ^ 3 * φ ^ 2
      + (-18 : ℝ) * δ ^ 3 * φ + (-4 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 8
      + (1 : ℝ) * δ ^ 2 * φ ^ 7 + (3 : ℝ) * δ ^ 2 * φ ^ 6 + (3 : ℝ) * δ ^ 2 * φ ^ 5
      + (5 : ℝ) * δ ^ 2 * φ ^ 4 + (5 : ℝ) * δ ^ 2 * φ ^ 3 + (7 : ℝ) * δ ^ 2 * φ ^ 2
      + (7 : ℝ) * δ ^ 2 * φ + (4 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 8 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_8_0_false, M_8_0_true, M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false,
    M_8_3_true, M_8_4_false, M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true,
    M_8_7_false, M_8_7_true, M_8_8_false, M_8_8_true]
  ring

private lemma M_9_1_true (δ φ : ℝ) :
    M_rec δ φ 1 9 true =
      (41 : ℝ) * δ ^ 9 * φ ^ 4 + (217 : ℝ) * δ ^ 9 * φ ^ 3 + (488 : ℝ) * δ ^ 9 * φ ^ 2
      + (498 : ℝ) * δ ^ 9 * φ + (186 : ℝ) * δ ^ 9 + (148 : ℝ) * δ ^ 8 * φ ^ 5
      + (845 : ℝ) * δ ^ 8 * φ ^ 4 + (1982 : ℝ) * δ ^ 8 * φ ^ 3 + (2184 : ℝ) * δ ^ 8 * φ ^ 2
      + (1090 : ℝ) * δ ^ 8 * φ + (186 : ℝ) * δ ^ 8 + (208 : ℝ) * δ ^ 7 * φ ^ 5
      + (1240 : ℝ) * δ ^ 7 * φ ^ 4 + (3209 : ℝ) * δ ^ 7 * φ ^ 3 + (4147 : ℝ) * δ ^ 7 * φ ^ 2
      + (2589 : ℝ) * δ ^ 7 * φ + (619 : ℝ) * δ ^ 7 + (98 : ℝ) * δ ^ 6 * φ ^ 6
      + (538 : ℝ) * δ ^ 6 * φ ^ 5 + (1637 : ℝ) * δ ^ 6 * φ ^ 4 + (3122 : ℝ) * δ ^ 6 * φ ^ 3
      + (3658 : ℝ) * δ ^ 6 * φ ^ 2 + (2340 : ℝ) * δ ^ 6 * φ + (619 : ℝ) * δ ^ 6
      + (113 : ℝ) * δ ^ 5 * φ ^ 6 + (563 : ℝ) * δ ^ 5 * φ ^ 5 + (1432 : ℝ) * δ ^ 5 * φ ^ 4
      + (2116 : ℝ) * δ ^ 5 * φ ^ 3 + (1782 : ℝ) * δ ^ 5 * φ ^ 2 + (786 : ℝ) * δ ^ 5 * φ
      + (138 : ℝ) * δ ^ 5 + (18 : ℝ) * δ ^ 4 * φ ^ 7 + (65 : ℝ) * δ ^ 4 * φ ^ 6
      + (172 : ℝ) * δ ^ 4 * φ ^ 5 + (330 : ℝ) * δ ^ 4 * φ ^ 4 + (516 : ℝ) * δ ^ 4 * φ ^ 3
      + (617 : ℝ) * δ ^ 4 * φ ^ 2 + (454 : ℝ) * δ ^ 4 * φ + (138 : ℝ) * δ ^ 4
      + (19 : ℝ) * δ ^ 3 * φ ^ 7 + (53 : ℝ) * δ ^ 3 * φ ^ 6 + (90 : ℝ) * δ ^ 3 * φ ^ 5
      + (104 : ℝ) * δ ^ 3 * φ ^ 4 + (83 : ℝ) * δ ^ 3 * φ ^ 3 + (49 : ℝ) * δ ^ 3 * φ ^ 2
      + (18 : ℝ) * δ ^ 3 * φ + (4 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 8
      + (1 : ℝ) * δ ^ 2 * φ ^ 7 + (3 : ℝ) * δ ^ 2 * φ ^ 6 + (3 : ℝ) * δ ^ 2 * φ ^ 5
      + (5 : ℝ) * δ ^ 2 * φ ^ 4 + (5 : ℝ) * δ ^ 2 * φ ^ 3 + (7 : ℝ) * δ ^ 2 * φ ^ 2
      + (7 : ℝ) * δ ^ 2 * φ + (4 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 8 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_8_0_false, M_8_0_true, M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false,
    M_8_3_true, M_8_4_false, M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true,
    M_8_7_false, M_8_7_true, M_8_8_false, M_8_8_true]
  ring

private lemma M_9_2_false (δ φ : ℝ) :
    M_rec δ φ 2 9 false =
      (112 : ℝ) * δ ^ 9 * φ ^ 4 + (430 : ℝ) * δ ^ 9 * φ ^ 3 + (608 : ℝ) * δ ^ 9 * φ ^ 2
      + (280 : ℝ) * δ ^ 9 * φ + (-204 : ℝ) * δ ^ 8 * φ ^ 4 + (-1072 : ℝ) * δ ^ 8 * φ ^ 3
      + (-2196 : ℝ) * δ ^ 8 * φ ^ 2 + (-1936 : ℝ) * δ ^ 8 * φ + (-598 : ℝ) * δ ^ 8
      + (140 : ℝ) * δ ^ 7 * φ ^ 5 + (768 : ℝ) * δ ^ 7 * φ ^ 4 + (2242 : ℝ) * δ ^ 7 * φ ^ 3
      + (3490 : ℝ) * δ ^ 7 * φ ^ 2 + (2782 : ℝ) * δ ^ 7 * φ + (874 : ℝ) * δ ^ 7
      + (-184 : ℝ) * δ ^ 6 * φ ^ 5 + (-986 : ℝ) * δ ^ 6 * φ ^ 4 + (-2476 : ℝ) * δ ^ 6 * φ ^ 3
      + (-3168 : ℝ) * δ ^ 6 * φ ^ 2 + (-1948 : ℝ) * δ ^ 6 * φ + (-478 : ℝ) * δ ^ 6
      + (32 : ℝ) * δ ^ 5 * φ ^ 6 + (134 : ℝ) * δ ^ 5 * φ ^ 5 + (488 : ℝ) * δ ^ 5 * φ ^ 4
      + (1132 : ℝ) * δ ^ 5 * φ ^ 3 + (1554 : ℝ) * δ ^ 5 * φ ^ 2 + (1024 : ℝ) * δ ^ 5 * φ
      + (256 : ℝ) * δ ^ 5 + (-36 : ℝ) * δ ^ 4 * φ ^ 6 + (-108 : ℝ) * δ ^ 4 * φ ^ 5
      + (-204 : ℝ) * δ ^ 4 * φ ^ 4 + (-304 : ℝ) * δ ^ 4 * φ ^ 3 + (-320 : ℝ) * δ ^ 4 * φ ^ 2
      + (-228 : ℝ) * δ ^ 4 * φ + (-60 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 7
      + (4 : ℝ) * δ ^ 3 * φ ^ 6 + (20 : ℝ) * δ ^ 3 * φ ^ 5 + (26 : ℝ) * δ ^ 3 * φ ^ 4
      + (50 : ℝ) * δ ^ 3 * φ ^ 3 + (32 : ℝ) * δ ^ 3 * φ ^ 2 + (28 : ℝ) * δ ^ 3 * φ
      + (6 : ℝ) * δ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ ^ 7 + (-2 : ℝ) * δ ^ 2 * φ ^ 5
      + (-2 : ℝ) * δ ^ 2 * φ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_9_2_true (δ φ : ℝ) :
    M_rec δ φ 2 9 true =
      (-112 : ℝ) * δ ^ 9 * φ ^ 4 + (-430 : ℝ) * δ ^ 9 * φ ^ 3 + (-608 : ℝ) * δ ^ 9 * φ ^ 2
      + (-280 : ℝ) * δ ^ 9 * φ + (-204 : ℝ) * δ ^ 8 * φ ^ 4 + (-1072 : ℝ) * δ ^ 8 * φ ^ 3
      + (-2196 : ℝ) * δ ^ 8 * φ ^ 2 + (-1936 : ℝ) * δ ^ 8 * φ + (-598 : ℝ) * δ ^ 8
      + (-140 : ℝ) * δ ^ 7 * φ ^ 5 + (-768 : ℝ) * δ ^ 7 * φ ^ 4 + (-2242 : ℝ) * δ ^ 7 * φ ^ 3
      + (-3490 : ℝ) * δ ^ 7 * φ ^ 2 + (-2782 : ℝ) * δ ^ 7 * φ + (-874 : ℝ) * δ ^ 7
      + (-184 : ℝ) * δ ^ 6 * φ ^ 5 + (-986 : ℝ) * δ ^ 6 * φ ^ 4 + (-2476 : ℝ) * δ ^ 6 * φ ^ 3
      + (-3168 : ℝ) * δ ^ 6 * φ ^ 2 + (-1948 : ℝ) * δ ^ 6 * φ + (-478 : ℝ) * δ ^ 6
      + (-32 : ℝ) * δ ^ 5 * φ ^ 6 + (-134 : ℝ) * δ ^ 5 * φ ^ 5 + (-488 : ℝ) * δ ^ 5 * φ ^ 4
      + (-1132 : ℝ) * δ ^ 5 * φ ^ 3 + (-1554 : ℝ) * δ ^ 5 * φ ^ 2 + (-1024 : ℝ) * δ ^ 5 * φ
      + (-256 : ℝ) * δ ^ 5 + (-36 : ℝ) * δ ^ 4 * φ ^ 6 + (-108 : ℝ) * δ ^ 4 * φ ^ 5
      + (-204 : ℝ) * δ ^ 4 * φ ^ 4 + (-304 : ℝ) * δ ^ 4 * φ ^ 3 + (-320 : ℝ) * δ ^ 4 * φ ^ 2
      + (-228 : ℝ) * δ ^ 4 * φ + (-60 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 7
      + (-4 : ℝ) * δ ^ 3 * φ ^ 6 + (-20 : ℝ) * δ ^ 3 * φ ^ 5 + (-26 : ℝ) * δ ^ 3 * φ ^ 4
      + (-50 : ℝ) * δ ^ 3 * φ ^ 3 + (-32 : ℝ) * δ ^ 3 * φ ^ 2 + (-28 : ℝ) * δ ^ 3 * φ
      + (-6 : ℝ) * δ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ ^ 7 + (-2 : ℝ) * δ ^ 2 * φ ^ 5
      + (-2 : ℝ) * δ ^ 2 * φ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_9_3_false (δ φ : ℝ) :
    M_rec δ φ 3 9 false =
      (-33 : ℝ) * δ ^ 9 * φ ^ 3 + (-246 : ℝ) * δ ^ 9 * φ ^ 2 + (-460 : ℝ) * δ ^ 9 * φ
      + (-262 : ℝ) * δ ^ 9 + (135 : ℝ) * δ ^ 8 * φ ^ 4 + (681 : ℝ) * δ ^ 8 * φ ^ 3
      + (1311 : ℝ) * δ ^ 8 * φ ^ 2 + (1245 : ℝ) * δ ^ 8 * φ + (489 : ℝ) * δ ^ 8
      + (-204 : ℝ) * δ ^ 7 * φ ^ 4 + (-1218 : ℝ) * δ ^ 7 * φ ^ 3 + (-2238 : ℝ) * δ ^ 7 * φ ^ 2
      + (-1782 : ℝ) * δ ^ 7 * φ + (-498 : ℝ) * δ ^ 7 + (42 : ℝ) * δ ^ 6 * φ ^ 5
      + (192 : ℝ) * δ ^ 6 * φ ^ 4 + (861 : ℝ) * δ ^ 6 * φ ^ 3 + (1681 : ℝ) * δ ^ 6 * φ ^ 2
      + (1426 : ℝ) * δ ^ 6 * φ + (418 : ℝ) * δ ^ 6 + (-51 : ℝ) * δ ^ 5 * φ ^ 5
      + (-174 : ℝ) * δ ^ 5 * φ ^ 4 + (-357 : ℝ) * δ ^ 5 * φ ^ 3 + (-636 : ℝ) * δ ^ 5 * φ ^ 2
      + (-492 : ℝ) * δ ^ 5 * φ + (-180 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 6
      + (9 : ℝ) * δ ^ 4 * φ ^ 5 + (60 : ℝ) * δ ^ 4 * φ ^ 4 + (66 : ℝ) * δ ^ 4 * φ ^ 3
      + (141 : ℝ) * δ ^ 4 * φ ^ 2 + (63 : ℝ) * δ ^ 4 * φ + (36 : ℝ) * δ ^ 4
      + (-3 : ℝ) * δ ^ 3 * φ ^ 6 + (-9 : ℝ) * δ ^ 3 * φ ^ 4 + (-13 : ℝ) * δ ^ 3 * φ ^ 2
      + (-3 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_9_3_true (δ φ : ℝ) :
    M_rec δ φ 3 9 true =
      (33 : ℝ) * δ ^ 9 * φ ^ 3 + (246 : ℝ) * δ ^ 9 * φ ^ 2 + (460 : ℝ) * δ ^ 9 * φ
      + (262 : ℝ) * δ ^ 9 + (135 : ℝ) * δ ^ 8 * φ ^ 4 + (681 : ℝ) * δ ^ 8 * φ ^ 3
      + (1311 : ℝ) * δ ^ 8 * φ ^ 2 + (1245 : ℝ) * δ ^ 8 * φ + (489 : ℝ) * δ ^ 8
      + (204 : ℝ) * δ ^ 7 * φ ^ 4 + (1218 : ℝ) * δ ^ 7 * φ ^ 3 + (2238 : ℝ) * δ ^ 7 * φ ^ 2
      + (1782 : ℝ) * δ ^ 7 * φ + (498 : ℝ) * δ ^ 7 + (42 : ℝ) * δ ^ 6 * φ ^ 5
      + (192 : ℝ) * δ ^ 6 * φ ^ 4 + (861 : ℝ) * δ ^ 6 * φ ^ 3 + (1681 : ℝ) * δ ^ 6 * φ ^ 2
      + (1426 : ℝ) * δ ^ 6 * φ + (418 : ℝ) * δ ^ 6 + (51 : ℝ) * δ ^ 5 * φ ^ 5
      + (174 : ℝ) * δ ^ 5 * φ ^ 4 + (357 : ℝ) * δ ^ 5 * φ ^ 3 + (636 : ℝ) * δ ^ 5 * φ ^ 2
      + (492 : ℝ) * δ ^ 5 * φ + (180 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 6
      + (9 : ℝ) * δ ^ 4 * φ ^ 5 + (60 : ℝ) * δ ^ 4 * φ ^ 4 + (66 : ℝ) * δ ^ 4 * φ ^ 3
      + (141 : ℝ) * δ ^ 4 * φ ^ 2 + (63 : ℝ) * δ ^ 4 * φ + (36 : ℝ) * δ ^ 4
      + (3 : ℝ) * δ ^ 3 * φ ^ 6 + (9 : ℝ) * δ ^ 3 * φ ^ 4 + (13 : ℝ) * δ ^ 3 * φ ^ 2
      + (3 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_9_4_false (δ φ : ℝ) :
    M_rec δ φ 4 9 false =
      (92 : ℝ) * δ ^ 9 * φ ^ 3 + (288 : ℝ) * δ ^ 9 * φ ^ 2 + (192 : ℝ) * δ ^ 9 * φ
      + (-164 : ℝ) * δ ^ 8 * φ ^ 3 + (-796 : ℝ) * δ ^ 8 * φ ^ 2 + (-804 : ℝ) * δ ^ 8 * φ
      + (-216 : ℝ) * δ ^ 8 + (48 : ℝ) * δ ^ 7 * φ ^ 4 + (224 : ℝ) * δ ^ 7 * φ ^ 3
      + (864 : ℝ) * δ ^ 7 * φ ^ 2 + (1096 : ℝ) * δ ^ 7 * φ + (408 : ℝ) * δ ^ 7
      + (-64 : ℝ) * δ ^ 6 * φ ^ 4 + (-260 : ℝ) * δ ^ 6 * φ ^ 3 + (-456 : ℝ) * δ ^ 6 * φ ^ 2
      + (-664 : ℝ) * δ ^ 6 * φ + (-236 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 5
      + (16 : ℝ) * δ ^ 5 * φ ^ 4 + (132 : ℝ) * δ ^ 5 * φ ^ 3 + (100 : ℝ) * δ ^ 5 * φ ^ 2
      + (208 : ℝ) * δ ^ 5 * φ + (44 : ℝ) * δ ^ 5 + (-4 : ℝ) * δ ^ 4 * φ ^ 5
      + (-24 : ℝ) * δ ^ 4 * φ ^ 3 + (-28 : ℝ) * δ ^ 4 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_9_4_true (δ φ : ℝ) :
    M_rec δ φ 4 9 true =
      (-92 : ℝ) * δ ^ 9 * φ ^ 3 + (-288 : ℝ) * δ ^ 9 * φ ^ 2 + (-192 : ℝ) * δ ^ 9 * φ
      + (-164 : ℝ) * δ ^ 8 * φ ^ 3 + (-796 : ℝ) * δ ^ 8 * φ ^ 2 + (-804 : ℝ) * δ ^ 8 * φ
      + (-216 : ℝ) * δ ^ 8 + (-48 : ℝ) * δ ^ 7 * φ ^ 4 + (-224 : ℝ) * δ ^ 7 * φ ^ 3
      + (-864 : ℝ) * δ ^ 7 * φ ^ 2 + (-1096 : ℝ) * δ ^ 7 * φ + (-408 : ℝ) * δ ^ 7
      + (-64 : ℝ) * δ ^ 6 * φ ^ 4 + (-260 : ℝ) * δ ^ 6 * φ ^ 3 + (-456 : ℝ) * δ ^ 6 * φ ^ 2
      + (-664 : ℝ) * δ ^ 6 * φ + (-236 : ℝ) * δ ^ 6 + (-4 : ℝ) * δ ^ 5 * φ ^ 5
      + (-16 : ℝ) * δ ^ 5 * φ ^ 4 + (-132 : ℝ) * δ ^ 5 * φ ^ 3 + (-100 : ℝ) * δ ^ 5 * φ ^ 2
      + (-208 : ℝ) * δ ^ 5 * φ + (-44 : ℝ) * δ ^ 5 + (-4 : ℝ) * δ ^ 4 * φ ^ 5
      + (-24 : ℝ) * δ ^ 4 * φ ^ 3 + (-28 : ℝ) * δ ^ 4 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_9_5_false (δ φ : ℝ) :
    M_rec δ φ 5 9 false =
      (-15 : ℝ) * δ ^ 9 * φ ^ 2 + (-145 : ℝ) * δ ^ 9 * φ + (-115 : ℝ) * δ ^ 9
      + (50 : ℝ) * δ ^ 8 * φ ^ 3 + (175 : ℝ) * δ ^ 8 * φ ^ 2 + (350 : ℝ) * δ ^ 8 * φ
      + (250 : ℝ) * δ ^ 8 + (-75 : ℝ) * δ ^ 7 * φ ^ 3 + (-335 : ℝ) * δ ^ 7 * φ ^ 2
      + (-290 : ℝ) * δ ^ 7 * φ + (-200 : ℝ) * δ ^ 7 + (5 : ℝ) * δ ^ 6 * φ ^ 4
      + (25 : ℝ) * δ ^ 6 * φ ^ 3 + (225 : ℝ) * δ ^ 6 * φ ^ 2 + (85 : ℝ) * δ ^ 6 * φ
      + (80 : ℝ) * δ ^ 6 + (-5 : ℝ) * δ ^ 5 * φ ^ 4 + (-50 : ℝ) * δ ^ 5 * φ ^ 2
      + (-15 : ℝ) * δ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_9_5_true (δ φ : ℝ) :
    M_rec δ φ 5 9 true =
      (15 : ℝ) * δ ^ 9 * φ ^ 2 + (145 : ℝ) * δ ^ 9 * φ + (115 : ℝ) * δ ^ 9
      + (50 : ℝ) * δ ^ 8 * φ ^ 3 + (175 : ℝ) * δ ^ 8 * φ ^ 2 + (350 : ℝ) * δ ^ 8 * φ
      + (250 : ℝ) * δ ^ 8 + (75 : ℝ) * δ ^ 7 * φ ^ 3 + (335 : ℝ) * δ ^ 7 * φ ^ 2
      + (290 : ℝ) * δ ^ 7 * φ + (200 : ℝ) * δ ^ 7 + (5 : ℝ) * δ ^ 6 * φ ^ 4
      + (25 : ℝ) * δ ^ 6 * φ ^ 3 + (225 : ℝ) * δ ^ 6 * φ ^ 2 + (85 : ℝ) * δ ^ 6 * φ
      + (80 : ℝ) * δ ^ 6 + (5 : ℝ) * δ ^ 5 * φ ^ 4 + (50 : ℝ) * δ ^ 5 * φ ^ 2 + (15 : ℝ) * δ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_9_6_false (δ φ : ℝ) :
    M_rec δ φ 6 9 false =
      (48 : ℝ) * δ ^ 9 * φ ^ 2 + (62 : ℝ) * δ ^ 9 * φ + (-84 : ℝ) * δ ^ 8 * φ ^ 2
      + (-180 : ℝ) * δ ^ 8 * φ + (-6 : ℝ) * δ ^ 8 + (6 : ℝ) * δ ^ 7 * φ ^ 3
      + (36 : ℝ) * δ ^ 7 * φ ^ 2 + (168 : ℝ) * δ ^ 7 * φ + (6 : ℝ) * δ ^ 7
      + (-6 : ℝ) * δ ^ 6 * φ ^ 3 + (-50 : ℝ) * δ ^ 6 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_9_6_true (δ φ : ℝ) :
    M_rec δ φ 6 9 true =
      (-48 : ℝ) * δ ^ 9 * φ ^ 2 + (-62 : ℝ) * δ ^ 9 * φ + (-84 : ℝ) * δ ^ 8 * φ ^ 2
      + (-180 : ℝ) * δ ^ 8 * φ + (-6 : ℝ) * δ ^ 8 + (-6 : ℝ) * δ ^ 7 * φ ^ 3
      + (-36 : ℝ) * δ ^ 7 * φ ^ 2 + (-168 : ℝ) * δ ^ 7 * φ + (-6 : ℝ) * δ ^ 7
      + (-6 : ℝ) * δ ^ 6 * φ ^ 3 + (-50 : ℝ) * δ ^ 6 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_9_7_false (δ φ : ℝ) :
    M_rec δ φ 7 9 false =
      (-7 : ℝ) * δ ^ 9 * φ + (-28 : ℝ) * δ ^ 9 + (7 : ℝ) * δ ^ 8 * φ ^ 2 + (7 : ℝ) * δ ^ 8 * φ
      + (49 : ℝ) * δ ^ 8 + (-7 : ℝ) * δ ^ 7 * φ ^ 2 + (-21 : ℝ) * δ ^ 7 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_9_7_true (δ φ : ℝ) :
    M_rec δ φ 7 9 true =
      (7 : ℝ) * δ ^ 9 * φ + (28 : ℝ) * δ ^ 9 + (7 : ℝ) * δ ^ 8 * φ ^ 2 + (7 : ℝ) * δ ^ 8 * φ
      + (49 : ℝ) * δ ^ 8 + (7 : ℝ) * δ ^ 7 * φ ^ 2 + (21 : ℝ) * δ ^ 7 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_9_8_false (δ φ : ℝ) :
    M_rec δ φ 8 9 false =
      (8 : ℝ) * δ ^ 9 * φ + (-8 : ℝ) * δ ^ 8 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_9_8_true (δ φ : ℝ) :
    M_rec δ φ 8 9 true =
      (-8 : ℝ) * δ ^ 9 * φ + (-8 : ℝ) * δ ^ 8 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_9_9_false (δ φ : ℝ) :
    M_rec δ φ 9 9 false =
      (-1 : ℝ) * δ ^ 9 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_9_9_true (δ φ : ℝ) :
    M_rec δ φ 9 9 true =
      (1 : ℝ) * δ ^ 9 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_10_1_false (δ φ : ℝ) :
    M_rec δ φ 1 10 false =
      (-153 : ℝ) * δ ^ 10 * φ ^ 5 + (-772 : ℝ) * δ ^ 10 * φ ^ 4 + (-1693 : ℝ) * δ ^ 10 * φ ^ 3
      + (-1652 : ℝ) * δ ^ 10 * φ ^ 2 + (-592 : ℝ) * δ ^ 10 * φ + (301 : ℝ) * δ ^ 9 * φ ^ 5
      + (1956 : ℝ) * δ ^ 9 * φ ^ 4 + (5642 : ℝ) * δ ^ 9 * φ ^ 3 + (8405 : ℝ) * δ ^ 9 * φ ^ 2
      + (6212 : ℝ) * δ ^ 9 * φ + (1794 : ℝ) * δ ^ 9 + (-348 : ℝ) * δ ^ 8 * φ ^ 6
      + (-2408 : ℝ) * δ ^ 8 * φ ^ 5 + (-8158 : ℝ) * δ ^ 8 * φ ^ 4 + (-15066 : ℝ) * δ ^ 8 * φ ^ 3
      + (-15460 : ℝ) * δ ^ 8 * φ ^ 2 + (-8246 : ℝ) * δ ^ 8 * φ + (-1794 : ℝ) * δ ^ 8
      + (446 : ℝ) * δ ^ 7 * φ ^ 6 + (3024 : ℝ) * δ ^ 7 * φ ^ 5 + (9858 : ℝ) * δ ^ 7 * φ ^ 4
      + (17867 : ℝ) * δ ^ 7 * φ ^ 3 + (17895 : ℝ) * δ ^ 7 * φ ^ 2 + (9139 : ℝ) * δ ^ 7 * φ
      + (1831 : ℝ) * δ ^ 7 + (-145 : ℝ) * δ ^ 6 * φ ^ 7 + (-850 : ℝ) * δ ^ 6 * φ ^ 6
      + (-2879 : ℝ) * δ ^ 6 * φ ^ 5 + (-6621 : ℝ) * δ ^ 6 * φ ^ 4 + (-10872 : ℝ) * δ ^ 6 * φ ^ 3
      + (-11698 : ℝ) * δ ^ 6 * φ ^ 2 + (-7146 : ℝ) * δ ^ 6 * φ + (-1831 : ℝ) * δ ^ 6
      + (163 : ℝ) * δ ^ 5 * φ ^ 7 + (856 : ℝ) * δ ^ 5 * φ ^ 6 + (2408 : ℝ) * δ ^ 5 * φ ^ 5
      + (4331 : ℝ) * δ ^ 5 * φ ^ 4 + (5032 : ℝ) * δ ^ 5 * φ ^ 3 + (3588 : ℝ) * δ ^ 5 * φ ^ 2
      + (1406 : ℝ) * δ ^ 5 * φ + (234 : ℝ) * δ ^ 5 + (-21 : ℝ) * δ ^ 4 * φ ^ 8
      + (-78 : ℝ) * δ ^ 4 * φ ^ 7 + (-214 : ℝ) * δ ^ 4 * φ ^ 6 + (-432 : ℝ) * δ ^ 4 * φ ^ 5
      + (-727 : ℝ) * δ ^ 4 * φ ^ 4 + (-1004 : ℝ) * δ ^ 4 * φ ^ 3 + (-1124 : ℝ) * δ ^ 4 * φ ^ 2
      + (-786 : ℝ) * δ ^ 4 * φ + (-234 : ℝ) * δ ^ 4 + (22 : ℝ) * δ ^ 3 * φ ^ 8
      + (63 : ℝ) * δ ^ 3 * φ ^ 7 + (113 : ℝ) * δ ^ 3 * φ ^ 6 + (144 : ℝ) * δ ^ 3 * φ ^ 5
      + (138 : ℝ) * δ ^ 3 * φ ^ 4 + (101 : ℝ) * δ ^ 3 * φ ^ 3 + (53 : ℝ) * δ ^ 3 * φ ^ 2
      + (22 : ℝ) * δ ^ 3 * φ + (4 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 9
      + (-1 : ℝ) * δ ^ 2 * φ ^ 8 + (-3 : ℝ) * δ ^ 2 * φ ^ 7 + (-3 : ℝ) * δ ^ 2 * φ ^ 6
      + (-5 : ℝ) * δ ^ 2 * φ ^ 5 + (-5 : ℝ) * δ ^ 2 * φ ^ 4 + (-7 : ℝ) * δ ^ 2 * φ ^ 3
      + (-7 : ℝ) * δ ^ 2 * φ ^ 2 + (-9 : ℝ) * δ ^ 2 * φ + (-4 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 9 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_9_0_false, M_9_0_true, M_9_1_false, M_9_1_true, M_9_2_false, M_9_2_true, M_9_3_false,
    M_9_3_true, M_9_4_false, M_9_4_true, M_9_5_false, M_9_5_true, M_9_6_false, M_9_6_true,
    M_9_7_false, M_9_7_true, M_9_8_false, M_9_8_true, M_9_9_false, M_9_9_true]
  ring

private lemma M_10_1_true (δ φ : ℝ) :
    M_rec δ φ 1 10 true =
      (-153 : ℝ) * δ ^ 10 * φ ^ 5 + (-772 : ℝ) * δ ^ 10 * φ ^ 4 + (-1693 : ℝ) * δ ^ 10 * φ ^ 3
      + (-1652 : ℝ) * δ ^ 10 * φ ^ 2 + (-592 : ℝ) * δ ^ 10 * φ + (-301 : ℝ) * δ ^ 9 * φ ^ 5
      + (-1956 : ℝ) * δ ^ 9 * φ ^ 4 + (-5642 : ℝ) * δ ^ 9 * φ ^ 3 + (-8405 : ℝ) * δ ^ 9 * φ ^ 2
      + (-6212 : ℝ) * δ ^ 9 * φ + (-1794 : ℝ) * δ ^ 9 + (-348 : ℝ) * δ ^ 8 * φ ^ 6
      + (-2408 : ℝ) * δ ^ 8 * φ ^ 5 + (-8158 : ℝ) * δ ^ 8 * φ ^ 4 + (-15066 : ℝ) * δ ^ 8 * φ ^ 3
      + (-15460 : ℝ) * δ ^ 8 * φ ^ 2 + (-8246 : ℝ) * δ ^ 8 * φ + (-1794 : ℝ) * δ ^ 8
      + (-446 : ℝ) * δ ^ 7 * φ ^ 6 + (-3024 : ℝ) * δ ^ 7 * φ ^ 5 + (-9858 : ℝ) * δ ^ 7 * φ ^ 4
      + (-17867 : ℝ) * δ ^ 7 * φ ^ 3 + (-17895 : ℝ) * δ ^ 7 * φ ^ 2 + (-9139 : ℝ) * δ ^ 7 * φ
      + (-1831 : ℝ) * δ ^ 7 + (-145 : ℝ) * δ ^ 6 * φ ^ 7 + (-850 : ℝ) * δ ^ 6 * φ ^ 6
      + (-2879 : ℝ) * δ ^ 6 * φ ^ 5 + (-6621 : ℝ) * δ ^ 6 * φ ^ 4 + (-10872 : ℝ) * δ ^ 6 * φ ^ 3
      + (-11698 : ℝ) * δ ^ 6 * φ ^ 2 + (-7146 : ℝ) * δ ^ 6 * φ + (-1831 : ℝ) * δ ^ 6
      + (-163 : ℝ) * δ ^ 5 * φ ^ 7 + (-856 : ℝ) * δ ^ 5 * φ ^ 6 + (-2408 : ℝ) * δ ^ 5 * φ ^ 5
      + (-4331 : ℝ) * δ ^ 5 * φ ^ 4 + (-5032 : ℝ) * δ ^ 5 * φ ^ 3 + (-3588 : ℝ) * δ ^ 5 * φ ^ 2
      + (-1406 : ℝ) * δ ^ 5 * φ + (-234 : ℝ) * δ ^ 5 + (-21 : ℝ) * δ ^ 4 * φ ^ 8
      + (-78 : ℝ) * δ ^ 4 * φ ^ 7 + (-214 : ℝ) * δ ^ 4 * φ ^ 6 + (-432 : ℝ) * δ ^ 4 * φ ^ 5
      + (-727 : ℝ) * δ ^ 4 * φ ^ 4 + (-1004 : ℝ) * δ ^ 4 * φ ^ 3 + (-1124 : ℝ) * δ ^ 4 * φ ^ 2
      + (-786 : ℝ) * δ ^ 4 * φ + (-234 : ℝ) * δ ^ 4 + (-22 : ℝ) * δ ^ 3 * φ ^ 8
      + (-63 : ℝ) * δ ^ 3 * φ ^ 7 + (-113 : ℝ) * δ ^ 3 * φ ^ 6 + (-144 : ℝ) * δ ^ 3 * φ ^ 5
      + (-138 : ℝ) * δ ^ 3 * φ ^ 4 + (-101 : ℝ) * δ ^ 3 * φ ^ 3 + (-53 : ℝ) * δ ^ 3 * φ ^ 2
      + (-22 : ℝ) * δ ^ 3 * φ + (-4 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 9
      + (-1 : ℝ) * δ ^ 2 * φ ^ 8 + (-3 : ℝ) * δ ^ 2 * φ ^ 7 + (-3 : ℝ) * δ ^ 2 * φ ^ 6
      + (-5 : ℝ) * δ ^ 2 * φ ^ 5 + (-5 : ℝ) * δ ^ 2 * φ ^ 4 + (-7 : ℝ) * δ ^ 2 * φ ^ 3
      + (-7 : ℝ) * δ ^ 2 * φ ^ 2 + (-9 : ℝ) * δ ^ 2 * φ + (-4 : ℝ) * δ ^ 2
      + (-1 : ℝ) * δ * φ ^ 9 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_9_0_false, M_9_0_true, M_9_1_false, M_9_1_true, M_9_2_false, M_9_2_true, M_9_3_false,
    M_9_3_true, M_9_4_false, M_9_4_true, M_9_5_false, M_9_5_true, M_9_6_false, M_9_6_true,
    M_9_7_false, M_9_7_true, M_9_8_false, M_9_8_true, M_9_9_false, M_9_9_true]
  ring

private lemma M_10_2_false (δ φ : ℝ) :
    M_rec δ φ 2 10 false =
      (82 : ℝ) * δ ^ 10 * φ ^ 4 + (570 : ℝ) * δ ^ 10 * φ ^ 3 + (1564 : ℝ) * δ ^ 10 * φ ^ 2
      + (1868 : ℝ) * δ ^ 10 * φ + (778 : ℝ) * δ ^ 10 + (-372 : ℝ) * δ ^ 9 * φ ^ 5
      + (-2332 : ℝ) * δ ^ 9 * φ ^ 4 + (-6194 : ℝ) * δ ^ 9 * φ ^ 3 + (-7914 : ℝ) * δ ^ 9 * φ ^ 2
      + (-4884 : ℝ) * δ ^ 9 * φ + (-1184 : ℝ) * δ ^ 9 + (568 : ℝ) * δ ^ 8 * φ ^ 5
      + (3938 : ℝ) * δ ^ 8 * φ ^ 4 + (11442 : ℝ) * δ ^ 8 * φ ^ 3 + (16054 : ℝ) * δ ^ 8 * φ ^ 2
      + (10490 : ℝ) * δ ^ 8 * φ + (2553 : ℝ) * δ ^ 8 + (-222 : ℝ) * δ ^ 7 * φ ^ 6
      + (-1350 : ℝ) * δ ^ 7 * φ ^ 5 + (-4956 : ℝ) * δ ^ 7 * φ ^ 4 + (-11532 : ℝ) * δ ^ 7 * φ ^ 3
      + (-15818 : ℝ) * δ ^ 7 * φ ^ 2 + (-11114 : ℝ) * δ ^ 7 * φ + (-3056 : ℝ) * δ ^ 7
      + (278 : ℝ) * δ ^ 6 * φ ^ 6 + (1558 : ℝ) * δ ^ 6 * φ ^ 5 + (4540 : ℝ) * δ ^ 6 * φ ^ 4
      + (8244 : ℝ) * δ ^ 6 * φ ^ 3 + (8892 : ℝ) * δ ^ 6 * φ ^ 2 + (5240 : ℝ) * δ ^ 6 * φ
      + (1278 : ℝ) * δ ^ 6 + (-38 : ℝ) * δ ^ 5 * φ ^ 7 + (-166 : ℝ) * δ ^ 5 * φ ^ 6
      + (-634 : ℝ) * δ ^ 5 * φ ^ 5 + (-1630 : ℝ) * δ ^ 5 * φ ^ 4 + (-2992 : ℝ) * δ ^ 5 * φ ^ 3
      + (-3256 : ℝ) * δ ^ 5 * φ ^ 2 + (-1910 : ℝ) * δ ^ 5 * φ + (-462 : ℝ) * δ ^ 5
      + (42 : ℝ) * δ ^ 4 * φ ^ 7 + (128 : ℝ) * δ ^ 4 * φ ^ 6 + (256 : ℝ) * δ ^ 4 * φ ^ 5
      + (412 : ℝ) * δ ^ 4 * φ ^ 4 + (510 : ℝ) * δ ^ 4 * φ ^ 3 + (528 : ℝ) * δ ^ 4 * φ ^ 2
      + (332 : ℝ) * δ ^ 4 * φ + (102 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 8
      + (-4 : ℝ) * δ ^ 3 * φ ^ 7 + (-20 : ℝ) * δ ^ 3 * φ ^ 6 + (-26 : ℝ) * δ ^ 3 * φ ^ 5
      + (-56 : ℝ) * δ ^ 3 * φ ^ 4 + (-48 : ℝ) * δ ^ 3 * φ ^ 3 + (-52 : ℝ) * δ ^ 3 * φ ^ 2
      + (-22 : ℝ) * δ ^ 3 * φ + (-10 : ℝ) * δ ^ 3 + (2 : ℝ) * δ ^ 2 * φ ^ 8
      + (2 : ℝ) * δ ^ 2 * φ ^ 6 + (2 : ℝ) * δ ^ 2 * φ ^ 4 + (2 : ℝ) * δ ^ 2 * φ ^ 2
      + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_8_0_false, M_8_0_true, M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false,
    M_8_3_true, M_8_4_false, M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true,
    M_8_7_false, M_8_7_true, M_8_8_false, M_8_8_true]
  ring

private lemma M_10_2_true (δ φ : ℝ) :
    M_rec δ φ 2 10 true =
      (82 : ℝ) * δ ^ 10 * φ ^ 4 + (570 : ℝ) * δ ^ 10 * φ ^ 3 + (1564 : ℝ) * δ ^ 10 * φ ^ 2
      + (1868 : ℝ) * δ ^ 10 * φ + (778 : ℝ) * δ ^ 10 + (372 : ℝ) * δ ^ 9 * φ ^ 5
      + (2332 : ℝ) * δ ^ 9 * φ ^ 4 + (6194 : ℝ) * δ ^ 9 * φ ^ 3 + (7914 : ℝ) * δ ^ 9 * φ ^ 2
      + (4884 : ℝ) * δ ^ 9 * φ + (1184 : ℝ) * δ ^ 9 + (568 : ℝ) * δ ^ 8 * φ ^ 5
      + (3938 : ℝ) * δ ^ 8 * φ ^ 4 + (11442 : ℝ) * δ ^ 8 * φ ^ 3 + (16054 : ℝ) * δ ^ 8 * φ ^ 2
      + (10490 : ℝ) * δ ^ 8 * φ + (2553 : ℝ) * δ ^ 8 + (222 : ℝ) * δ ^ 7 * φ ^ 6
      + (1350 : ℝ) * δ ^ 7 * φ ^ 5 + (4956 : ℝ) * δ ^ 7 * φ ^ 4 + (11532 : ℝ) * δ ^ 7 * φ ^ 3
      + (15818 : ℝ) * δ ^ 7 * φ ^ 2 + (11114 : ℝ) * δ ^ 7 * φ + (3056 : ℝ) * δ ^ 7
      + (278 : ℝ) * δ ^ 6 * φ ^ 6 + (1558 : ℝ) * δ ^ 6 * φ ^ 5 + (4540 : ℝ) * δ ^ 6 * φ ^ 4
      + (8244 : ℝ) * δ ^ 6 * φ ^ 3 + (8892 : ℝ) * δ ^ 6 * φ ^ 2 + (5240 : ℝ) * δ ^ 6 * φ
      + (1278 : ℝ) * δ ^ 6 + (38 : ℝ) * δ ^ 5 * φ ^ 7 + (166 : ℝ) * δ ^ 5 * φ ^ 6
      + (634 : ℝ) * δ ^ 5 * φ ^ 5 + (1630 : ℝ) * δ ^ 5 * φ ^ 4 + (2992 : ℝ) * δ ^ 5 * φ ^ 3
      + (3256 : ℝ) * δ ^ 5 * φ ^ 2 + (1910 : ℝ) * δ ^ 5 * φ + (462 : ℝ) * δ ^ 5
      + (42 : ℝ) * δ ^ 4 * φ ^ 7 + (128 : ℝ) * δ ^ 4 * φ ^ 6 + (256 : ℝ) * δ ^ 4 * φ ^ 5
      + (412 : ℝ) * δ ^ 4 * φ ^ 4 + (510 : ℝ) * δ ^ 4 * φ ^ 3 + (528 : ℝ) * δ ^ 4 * φ ^ 2
      + (332 : ℝ) * δ ^ 4 * φ + (102 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 8
      + (4 : ℝ) * δ ^ 3 * φ ^ 7 + (20 : ℝ) * δ ^ 3 * φ ^ 6 + (26 : ℝ) * δ ^ 3 * φ ^ 5
      + (56 : ℝ) * δ ^ 3 * φ ^ 4 + (48 : ℝ) * δ ^ 3 * φ ^ 3 + (52 : ℝ) * δ ^ 3 * φ ^ 2
      + (22 : ℝ) * δ ^ 3 * φ + (10 : ℝ) * δ ^ 3 + (2 : ℝ) * δ ^ 2 * φ ^ 8
      + (2 : ℝ) * δ ^ 2 * φ ^ 6 + (2 : ℝ) * δ ^ 2 * φ ^ 4 + (2 : ℝ) * δ ^ 2 * φ ^ 2
      + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_8_0_false, M_8_0_true, M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false,
    M_8_3_true, M_8_4_false, M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true,
    M_8_7_false, M_8_7_true, M_8_8_false, M_8_8_true]
  ring

private lemma M_10_3_false (δ φ : ℝ) :
    M_rec δ φ 3 10 false =
      (-213 : ℝ) * δ ^ 10 * φ ^ 4 + (-957 : ℝ) * δ ^ 10 * φ ^ 3 + (-1506 : ℝ) * δ ^ 10 * φ ^ 2
      + (-756 : ℝ) * δ ^ 10 * φ + (396 : ℝ) * δ ^ 9 * φ ^ 4 + (2484 : ℝ) * δ ^ 9 * φ ^ 3
      + (5559 : ℝ) * δ ^ 9 * φ ^ 2 + (5046 : ℝ) * δ ^ 9 * φ + (1530 : ℝ) * δ ^ 9
      + (-240 : ℝ) * δ ^ 8 * φ ^ 5 + (-1464 : ℝ) * δ ^ 8 * φ ^ 4 + (-5043 : ℝ) * δ ^ 8 * φ ^ 3
      + (-9237 : ℝ) * δ ^ 8 * φ ^ 2 + (-8247 : ℝ) * δ ^ 8 * φ + (-2796 : ℝ) * δ ^ 8
      + (336 : ℝ) * δ ^ 7 * φ ^ 5 + (2106 : ℝ) * δ ^ 7 * φ ^ 4 + (6042 : ℝ) * δ ^ 7 * φ ^ 3
      + (8946 : ℝ) * δ ^ 7 * φ ^ 2 + (6426 : ℝ) * δ ^ 7 * φ + (1884 : ℝ) * δ ^ 7
      + (-51 : ℝ) * δ ^ 6 * φ ^ 6 + (-249 : ℝ) * δ ^ 6 * φ ^ 5 + (-1218 : ℝ) * δ ^ 6 * φ ^ 4
      + (-3342 : ℝ) * δ ^ 6 * φ ^ 3 + (-4890 : ℝ) * δ ^ 6 * φ ^ 2 + (-3300 : ℝ) * δ ^ 6 * φ
      + (-810 : ℝ) * δ ^ 6 + (60 : ℝ) * δ ^ 5 * φ ^ 6 + (204 : ℝ) * δ ^ 5 * φ ^ 5
      + (465 : ℝ) * δ ^ 5 * φ ^ 4 + (984 : ℝ) * δ ^ 5 * φ ^ 3 + (1263 : ℝ) * δ ^ 5 * φ ^ 2
      + (966 : ℝ) * δ ^ 5 * φ + (216 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 7
      + (-9 : ℝ) * δ ^ 4 * φ ^ 6 + (-60 : ℝ) * δ ^ 4 * φ ^ 5 + (-72 : ℝ) * δ ^ 4 * φ ^ 4
      + (-183 : ℝ) * δ ^ 4 * φ ^ 3 + (-135 : ℝ) * δ ^ 4 * φ ^ 2 + (-144 : ℝ) * δ ^ 4 * φ
      + (-24 : ℝ) * δ ^ 4 + (3 : ℝ) * δ ^ 3 * φ ^ 7 + (9 : ℝ) * δ ^ 3 * φ ^ 5
      + (15 : ℝ) * δ ^ 3 * φ ^ 3 + (9 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_10_3_true (δ φ : ℝ) :
    M_rec δ φ 3 10 true =
      (-213 : ℝ) * δ ^ 10 * φ ^ 4 + (-957 : ℝ) * δ ^ 10 * φ ^ 3 + (-1506 : ℝ) * δ ^ 10 * φ ^ 2
      + (-756 : ℝ) * δ ^ 10 * φ + (-396 : ℝ) * δ ^ 9 * φ ^ 4 + (-2484 : ℝ) * δ ^ 9 * φ ^ 3
      + (-5559 : ℝ) * δ ^ 9 * φ ^ 2 + (-5046 : ℝ) * δ ^ 9 * φ + (-1530 : ℝ) * δ ^ 9
      + (-240 : ℝ) * δ ^ 8 * φ ^ 5 + (-1464 : ℝ) * δ ^ 8 * φ ^ 4 + (-5043 : ℝ) * δ ^ 8 * φ ^ 3
      + (-9237 : ℝ) * δ ^ 8 * φ ^ 2 + (-8247 : ℝ) * δ ^ 8 * φ + (-2796 : ℝ) * δ ^ 8
      + (-336 : ℝ) * δ ^ 7 * φ ^ 5 + (-2106 : ℝ) * δ ^ 7 * φ ^ 4 + (-6042 : ℝ) * δ ^ 7 * φ ^ 3
      + (-8946 : ℝ) * δ ^ 7 * φ ^ 2 + (-6426 : ℝ) * δ ^ 7 * φ + (-1884 : ℝ) * δ ^ 7
      + (-51 : ℝ) * δ ^ 6 * φ ^ 6 + (-249 : ℝ) * δ ^ 6 * φ ^ 5 + (-1218 : ℝ) * δ ^ 6 * φ ^ 4
      + (-3342 : ℝ) * δ ^ 6 * φ ^ 3 + (-4890 : ℝ) * δ ^ 6 * φ ^ 2 + (-3300 : ℝ) * δ ^ 6 * φ
      + (-810 : ℝ) * δ ^ 6 + (-60 : ℝ) * δ ^ 5 * φ ^ 6 + (-204 : ℝ) * δ ^ 5 * φ ^ 5
      + (-465 : ℝ) * δ ^ 5 * φ ^ 4 + (-984 : ℝ) * δ ^ 5 * φ ^ 3 + (-1263 : ℝ) * δ ^ 5 * φ ^ 2
      + (-966 : ℝ) * δ ^ 5 * φ + (-216 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 7
      + (-9 : ℝ) * δ ^ 4 * φ ^ 6 + (-60 : ℝ) * δ ^ 4 * φ ^ 5 + (-72 : ℝ) * δ ^ 4 * φ ^ 4
      + (-183 : ℝ) * δ ^ 4 * φ ^ 3 + (-135 : ℝ) * δ ^ 4 * φ ^ 2 + (-144 : ℝ) * δ ^ 4 * φ
      + (-24 : ℝ) * δ ^ 4 + (-3 : ℝ) * δ ^ 3 * φ ^ 7 + (-9 : ℝ) * δ ^ 3 * φ ^ 5
      + (-15 : ℝ) * δ ^ 3 * φ ^ 3 + (-9 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_10_4_false (δ φ : ℝ) :
    M_rec δ φ 4 10 false =
      (44 : ℝ) * δ ^ 10 * φ ^ 3 + (440 : ℝ) * δ ^ 10 * φ ^ 2 + (944 : ℝ) * δ ^ 10 * φ
      + (574 : ℝ) * δ ^ 10 + (-208 : ℝ) * δ ^ 9 * φ ^ 4 + (-1196 : ℝ) * δ ^ 9 * φ ^ 3
      + (-2540 : ℝ) * δ ^ 9 * φ ^ 2 + (-2816 : ℝ) * δ ^ 9 * φ + (-1248 : ℝ) * δ ^ 9
      + (328 : ℝ) * δ ^ 8 * φ ^ 4 + (2360 : ℝ) * δ ^ 8 * φ ^ 3 + (4696 : ℝ) * δ ^ 8 * φ ^ 2
      + (4152 : ℝ) * δ ^ 8 * φ + (1334 : ℝ) * δ ^ 8 + (-60 : ℝ) * δ ^ 7 * φ ^ 5
      + (-312 : ℝ) * δ ^ 7 * φ ^ 4 + (-1804 : ℝ) * δ ^ 7 * φ ^ 3 + (-3892 : ℝ) * δ ^ 7 * φ ^ 2
      + (-3416 : ℝ) * δ ^ 7 * φ + (-1076 : ℝ) * δ ^ 7 + (76 : ℝ) * δ ^ 6 * φ ^ 5
      + (300 : ℝ) * δ ^ 6 * φ ^ 4 + (732 : ℝ) * δ ^ 6 * φ ^ 3 + (1648 : ℝ) * δ ^ 6 * φ ^ 2
      + (1328 : ℝ) * δ ^ 6 * φ + (536 : ℝ) * δ ^ 6 + (-4 : ℝ) * δ ^ 5 * φ ^ 6
      + (-16 : ℝ) * δ ^ 5 * φ ^ 5 + (-132 : ℝ) * δ ^ 5 * φ ^ 4 + (-136 : ℝ) * δ ^ 5 * φ ^ 3
      + (-396 : ℝ) * δ ^ 5 * φ ^ 2 + (-192 : ℝ) * δ ^ 5 * φ + (-132 : ℝ) * δ ^ 5
      + (4 : ℝ) * δ ^ 4 * φ ^ 6 + (24 : ℝ) * δ ^ 4 * φ ^ 4 + (44 : ℝ) * δ ^ 4 * φ ^ 2
      + (12 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_10_4_true (δ φ : ℝ) :
    M_rec δ φ 4 10 true =
      (44 : ℝ) * δ ^ 10 * φ ^ 3 + (440 : ℝ) * δ ^ 10 * φ ^ 2 + (944 : ℝ) * δ ^ 10 * φ
      + (574 : ℝ) * δ ^ 10 + (208 : ℝ) * δ ^ 9 * φ ^ 4 + (1196 : ℝ) * δ ^ 9 * φ ^ 3
      + (2540 : ℝ) * δ ^ 9 * φ ^ 2 + (2816 : ℝ) * δ ^ 9 * φ + (1248 : ℝ) * δ ^ 9
      + (328 : ℝ) * δ ^ 8 * φ ^ 4 + (2360 : ℝ) * δ ^ 8 * φ ^ 3 + (4696 : ℝ) * δ ^ 8 * φ ^ 2
      + (4152 : ℝ) * δ ^ 8 * φ + (1334 : ℝ) * δ ^ 8 + (60 : ℝ) * δ ^ 7 * φ ^ 5
      + (312 : ℝ) * δ ^ 7 * φ ^ 4 + (1804 : ℝ) * δ ^ 7 * φ ^ 3 + (3892 : ℝ) * δ ^ 7 * φ ^ 2
      + (3416 : ℝ) * δ ^ 7 * φ + (1076 : ℝ) * δ ^ 7 + (76 : ℝ) * δ ^ 6 * φ ^ 5
      + (300 : ℝ) * δ ^ 6 * φ ^ 4 + (732 : ℝ) * δ ^ 6 * φ ^ 3 + (1648 : ℝ) * δ ^ 6 * φ ^ 2
      + (1328 : ℝ) * δ ^ 6 * φ + (536 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 6
      + (16 : ℝ) * δ ^ 5 * φ ^ 5 + (132 : ℝ) * δ ^ 5 * φ ^ 4 + (136 : ℝ) * δ ^ 5 * φ ^ 3
      + (396 : ℝ) * δ ^ 5 * φ ^ 2 + (192 : ℝ) * δ ^ 5 * φ + (132 : ℝ) * δ ^ 5
      + (4 : ℝ) * δ ^ 4 * φ ^ 6 + (24 : ℝ) * δ ^ 4 * φ ^ 4 + (44 : ℝ) * δ ^ 4 * φ ^ 2
      + (12 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_10_5_false (δ φ : ℝ) :
    M_rec δ φ 5 10 false =
      (-135 : ℝ) * δ ^ 10 * φ ^ 3 + (-510 : ℝ) * δ ^ 10 * φ ^ 2 + (-356 : ℝ) * δ ^ 10 * φ
      + (245 : ℝ) * δ ^ 9 * φ ^ 3 + (1460 : ℝ) * δ ^ 9 * φ ^ 2 + (1505 : ℝ) * δ ^ 9 * φ
      + (365 : ℝ) * δ ^ 9 + (-65 : ℝ) * δ ^ 8 * φ ^ 4 + (-340 : ℝ) * δ ^ 8 * φ ^ 3
      + (-1620 : ℝ) * δ ^ 8 * φ ^ 2 + (-2180 : ℝ) * δ ^ 8 * φ + (-745 : ℝ) * δ ^ 8
      + (90 : ℝ) * δ ^ 7 * φ ^ 4 + (425 : ℝ) * δ ^ 7 * φ ^ 3 + (855 : ℝ) * δ ^ 7 * φ ^ 2
      + (1450 : ℝ) * δ ^ 7 * φ + (480 : ℝ) * δ ^ 7 + (-5 : ℝ) * δ ^ 6 * φ ^ 5
      + (-25 : ℝ) * δ ^ 6 * φ ^ 4 + (-245 : ℝ) * δ ^ 6 * φ ^ 3 + (-185 : ℝ) * δ ^ 6 * φ ^ 2
      + (-490 : ℝ) * δ ^ 6 * φ + (-100 : ℝ) * δ ^ 6 + (5 : ℝ) * δ ^ 5 * φ ^ 5
      + (50 : ℝ) * δ ^ 5 * φ ^ 3 + (71 : ℝ) * δ ^ 5 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_10_5_true (δ φ : ℝ) :
    M_rec δ φ 5 10 true =
      (-135 : ℝ) * δ ^ 10 * φ ^ 3 + (-510 : ℝ) * δ ^ 10 * φ ^ 2 + (-356 : ℝ) * δ ^ 10 * φ
      + (-245 : ℝ) * δ ^ 9 * φ ^ 3 + (-1460 : ℝ) * δ ^ 9 * φ ^ 2 + (-1505 : ℝ) * δ ^ 9 * φ
      + (-365 : ℝ) * δ ^ 9 + (-65 : ℝ) * δ ^ 8 * φ ^ 4 + (-340 : ℝ) * δ ^ 8 * φ ^ 3
      + (-1620 : ℝ) * δ ^ 8 * φ ^ 2 + (-2180 : ℝ) * δ ^ 8 * φ + (-745 : ℝ) * δ ^ 8
      + (-90 : ℝ) * δ ^ 7 * φ ^ 4 + (-425 : ℝ) * δ ^ 7 * φ ^ 3 + (-855 : ℝ) * δ ^ 7 * φ ^ 2
      + (-1450 : ℝ) * δ ^ 7 * φ + (-480 : ℝ) * δ ^ 7 + (-5 : ℝ) * δ ^ 6 * φ ^ 5
      + (-25 : ℝ) * δ ^ 6 * φ ^ 4 + (-245 : ℝ) * δ ^ 6 * φ ^ 3 + (-185 : ℝ) * δ ^ 6 * φ ^ 2
      + (-490 : ℝ) * δ ^ 6 * φ + (-100 : ℝ) * δ ^ 6 + (-5 : ℝ) * δ ^ 5 * φ ^ 5
      + (-50 : ℝ) * δ ^ 5 * φ ^ 3 + (-71 : ℝ) * δ ^ 5 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_10_6_false (δ φ : ℝ) :
    M_rec δ φ 6 10 false =
      (18 : ℝ) * δ ^ 10 * φ ^ 2 + (222 : ℝ) * δ ^ 10 * φ + (189 : ℝ) * δ ^ 10
      + (-66 : ℝ) * δ ^ 9 * φ ^ 3 + (-258 : ℝ) * δ ^ 9 * φ ^ 2 + (-552 : ℝ) * δ ^ 9 * φ
      + (-444 : ℝ) * δ ^ 9 + (102 : ℝ) * δ ^ 8 * φ ^ 3 + (528 : ℝ) * δ ^ 8 * φ ^ 2
      + (468 : ℝ) * δ ^ 8 * φ + (387 : ℝ) * δ ^ 8 + (-6 : ℝ) * δ ^ 7 * φ ^ 4
      + (-36 : ℝ) * δ ^ 7 * φ ^ 3 + (-378 : ℝ) * δ ^ 7 * φ ^ 2 + (-138 : ℝ) * δ ^ 7 * φ
      + (-162 : ℝ) * δ ^ 7 + (6 : ℝ) * δ ^ 6 * φ ^ 4 + (90 : ℝ) * δ ^ 6 * φ ^ 2
      + (30 : ℝ) * δ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_10_6_true (δ φ : ℝ) :
    M_rec δ φ 6 10 true =
      (18 : ℝ) * δ ^ 10 * φ ^ 2 + (222 : ℝ) * δ ^ 10 * φ + (189 : ℝ) * δ ^ 10
      + (66 : ℝ) * δ ^ 9 * φ ^ 3 + (258 : ℝ) * δ ^ 9 * φ ^ 2 + (552 : ℝ) * δ ^ 9 * φ
      + (444 : ℝ) * δ ^ 9 + (102 : ℝ) * δ ^ 8 * φ ^ 3 + (528 : ℝ) * δ ^ 8 * φ ^ 2
      + (468 : ℝ) * δ ^ 8 * φ + (387 : ℝ) * δ ^ 8 + (6 : ℝ) * δ ^ 7 * φ ^ 4
      + (36 : ℝ) * δ ^ 7 * φ ^ 3 + (378 : ℝ) * δ ^ 7 * φ ^ 2 + (138 : ℝ) * δ ^ 7 * φ
      + (162 : ℝ) * δ ^ 7 + (6 : ℝ) * δ ^ 6 * φ ^ 4 + (90 : ℝ) * δ ^ 6 * φ ^ 2
      + (30 : ℝ) * δ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_10_7_false (δ φ : ℝ) :
    M_rec δ φ 7 10 false =
      (-63 : ℝ) * δ ^ 10 * φ ^ 2 + (-91 : ℝ) * δ ^ 10 * φ + (112 : ℝ) * δ ^ 9 * φ ^ 2
      + (266 : ℝ) * δ ^ 9 * φ + (7 : ℝ) * δ ^ 9 + (-7 : ℝ) * δ ^ 8 * φ ^ 3
      + (-49 : ℝ) * δ ^ 8 * φ ^ 2 + (-252 : ℝ) * δ ^ 8 * φ + (-7 : ℝ) * δ ^ 8
      + (7 : ℝ) * δ ^ 7 * φ ^ 3 + (77 : ℝ) * δ ^ 7 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_10_7_true (δ φ : ℝ) :
    M_rec δ φ 7 10 true =
      (-63 : ℝ) * δ ^ 10 * φ ^ 2 + (-91 : ℝ) * δ ^ 10 * φ + (-112 : ℝ) * δ ^ 9 * φ ^ 2
      + (-266 : ℝ) * δ ^ 9 * φ + (-7 : ℝ) * δ ^ 9 + (-7 : ℝ) * δ ^ 8 * φ ^ 3
      + (-49 : ℝ) * δ ^ 8 * φ ^ 2 + (-252 : ℝ) * δ ^ 8 * φ + (-7 : ℝ) * δ ^ 8
      + (-7 : ℝ) * δ ^ 7 * φ ^ 3 + (-77 : ℝ) * δ ^ 7 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_10_8_false (δ φ : ℝ) :
    M_rec δ φ 8 10 false =
      (8 : ℝ) * δ ^ 10 * φ + (36 : ℝ) * δ ^ 10 + (-8 : ℝ) * δ ^ 9 * φ ^ 2 + (-8 : ℝ) * δ ^ 9 * φ
      + (-64 : ℝ) * δ ^ 9 + (8 : ℝ) * δ ^ 8 * φ ^ 2 + (28 : ℝ) * δ ^ 8 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_10_8_true (δ φ : ℝ) :
    M_rec δ φ 8 10 true =
      (8 : ℝ) * δ ^ 10 * φ + (36 : ℝ) * δ ^ 10 + (8 : ℝ) * δ ^ 9 * φ ^ 2 + (8 : ℝ) * δ ^ 9 * φ
      + (64 : ℝ) * δ ^ 9 + (8 : ℝ) * δ ^ 8 * φ ^ 2 + (28 : ℝ) * δ ^ 8 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_10_9_false (δ φ : ℝ) :
    M_rec δ φ 9 10 false =
      (-9 : ℝ) * δ ^ 10 * φ + (9 : ℝ) * δ ^ 9 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_10_9_true (δ φ : ℝ) :
    M_rec δ φ 9 10 true =
      (-9 : ℝ) * δ ^ 10 * φ + (-9 : ℝ) * δ ^ 9 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_10_10_false (δ φ : ℝ) :
    M_rec δ φ 10 10 false =
      (1 : ℝ) * δ ^ 10 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_10_10_true (δ φ : ℝ) :
    M_rec δ φ 10 10 true =
      (1 : ℝ) * δ ^ 10 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_11_1_false (δ φ : ℝ) :
    M_rec δ φ 1 11 false =
      (-153 : ℝ) * δ ^ 11 * φ ^ 5 + (-1067 : ℝ) * δ ^ 11 * φ ^ 4 + (-3399 : ℝ) * δ ^ 11 * φ ^ 3
      + (-5753 : ℝ) * δ ^ 11 * φ ^ 2 + (-4846 : ℝ) * δ ^ 11 * φ + (-1578 : ℝ) * δ ^ 11
      + (673 : ℝ) * δ ^ 10 * φ ^ 6 + (5045 : ℝ) * δ ^ 10 * φ ^ 5 + (16894 : ℝ) * δ ^ 10 * φ ^ 4
      + (29655 : ℝ) * δ ^ 10 * φ ^ 3 + (27051 : ℝ) * δ ^ 10 * φ ^ 2 + (11482 : ℝ) * δ ^ 10 * φ
      + (1578 : ℝ) * δ ^ 10 + (-1021 : ℝ) * δ ^ 9 * φ ^ 6 + (-8108 : ℝ) * δ ^ 9 * φ ^ 5
      + (-29780 : ℝ) * δ ^ 9 * φ ^ 4 + (-60616 : ℝ) * δ ^ 9 * φ ^ 3
      + (-68950 : ℝ) * δ ^ 9 * φ ^ 2 + (-40671 : ℝ) * δ ^ 9 * φ + (-9644 : ℝ) * δ ^ 9
      + (668 : ℝ) * δ ^ 8 * φ ^ 7 + (5118 : ℝ) * δ ^ 8 * φ ^ 6 + (20544 : ℝ) * δ ^ 8 * φ ^ 5
      + (51666 : ℝ) * δ ^ 8 * φ ^ 4 + (82144 : ℝ) * δ ^ 8 * φ ^ 3 + (79412 : ℝ) * δ ^ 8 * φ ^ 2
      + (42524 : ℝ) * δ ^ 8 * φ + (9644 : ℝ) * δ ^ 8 + (-813 : ℝ) * δ ^ 7 * φ ^ 7
      + (-5949 : ℝ) * δ ^ 7 * φ ^ 6 + (-22095 : ℝ) * δ ^ 7 * φ ^ 5
      + (-50423 : ℝ) * δ ^ 7 * φ ^ 4 + (-71219 : ℝ) * δ ^ 7 * φ ^ 3
      + (-59163 : ℝ) * δ ^ 7 * φ ^ 2 + (-25993 : ℝ) * δ ^ 7 * φ + (-4585 : ℝ) * δ ^ 7
      + (201 : ℝ) * δ ^ 6 * φ ^ 8 + (1231 : ℝ) * δ ^ 6 * φ ^ 7 + (4446 : ℝ) * δ ^ 6 * φ ^ 6
      + (11325 : ℝ) * δ ^ 6 * φ ^ 5 + (21904 : ℝ) * δ ^ 6 * φ ^ 4 + (31938 : ℝ) * δ ^ 6 * φ ^ 3
      + (31948 : ℝ) * δ ^ 6 * φ ^ 2 + (18548 : ℝ) * δ ^ 6 * φ + (4585 : ℝ) * δ ^ 6
      + (-222 : ℝ) * δ ^ 5 * φ ^ 8 + (-1209 : ℝ) * δ ^ 5 * φ ^ 7 + (-3622 : ℝ) * δ ^ 5 * φ ^ 6
      + (-7306 : ℝ) * δ ^ 5 * φ ^ 5 + (-10429 : ℝ) * δ ^ 5 * φ ^ 4
      + (-10200 : ℝ) * δ ^ 5 * φ ^ 3 + (-6376 : ℝ) * δ ^ 5 * φ ^ 2 + (-2306 : ℝ) * δ ^ 5 * φ
      + (-372 : ℝ) * δ ^ 5 + (24 : ℝ) * δ ^ 4 * φ ^ 9 + (91 : ℝ) * δ ^ 4 * φ ^ 8
      + (256 : ℝ) * δ ^ 4 * φ ^ 7 + (534 : ℝ) * δ ^ 4 * φ ^ 6 + (942 : ℝ) * δ ^ 4 * φ ^ 5
      + (1399 : ℝ) * δ ^ 4 * φ ^ 4 + (1802 : ℝ) * δ ^ 4 * φ ^ 3 + (1884 : ℝ) * δ ^ 4 * φ ^ 2
      + (1276 : ℝ) * δ ^ 4 * φ + (372 : ℝ) * δ ^ 4 + (-25 : ℝ) * δ ^ 3 * φ ^ 9
      + (-73 : ℝ) * δ ^ 3 * φ ^ 8 + (-136 : ℝ) * δ ^ 3 * φ ^ 7 + (-184 : ℝ) * δ ^ 3 * φ ^ 6
      + (-199 : ℝ) * δ ^ 3 * φ ^ 5 + (-171 : ℝ) * δ ^ 3 * φ ^ 4 + (-112 : ℝ) * δ ^ 3 * φ ^ 3
      + (-62 : ℝ) * δ ^ 3 * φ ^ 2 + (-23 : ℝ) * δ ^ 3 * φ + (-5 : ℝ) * δ ^ 3
      + (1 : ℝ) * δ ^ 2 * φ ^ 10 + (1 : ℝ) * δ ^ 2 * φ ^ 9 + (3 : ℝ) * δ ^ 2 * φ ^ 8
      + (3 : ℝ) * δ ^ 2 * φ ^ 7 + (5 : ℝ) * δ ^ 2 * φ ^ 6 + (5 : ℝ) * δ ^ 2 * φ ^ 5
      + (7 : ℝ) * δ ^ 2 * φ ^ 4 + (7 : ℝ) * δ ^ 2 * φ ^ 3 + (9 : ℝ) * δ ^ 2 * φ ^ 2
      + (9 : ℝ) * δ ^ 2 * φ + (5 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 10 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_10_0_false, M_10_0_true, M_10_1_false, M_10_1_true, M_10_2_false, M_10_2_true,
    M_10_3_false, M_10_3_true, M_10_4_false, M_10_4_true, M_10_5_false, M_10_5_true,
    M_10_6_false, M_10_6_true, M_10_7_false, M_10_7_true, M_10_8_false, M_10_8_true,
    M_10_9_false, M_10_9_true, M_10_10_false, M_10_10_true]
  ring

private lemma M_11_1_true (δ φ : ℝ) :
    M_rec δ φ 1 11 true =
      (153 : ℝ) * δ ^ 11 * φ ^ 5 + (1067 : ℝ) * δ ^ 11 * φ ^ 4 + (3399 : ℝ) * δ ^ 11 * φ ^ 3
      + (5753 : ℝ) * δ ^ 11 * φ ^ 2 + (4846 : ℝ) * δ ^ 11 * φ + (1578 : ℝ) * δ ^ 11
      + (673 : ℝ) * δ ^ 10 * φ ^ 6 + (5045 : ℝ) * δ ^ 10 * φ ^ 5 + (16894 : ℝ) * δ ^ 10 * φ ^ 4
      + (29655 : ℝ) * δ ^ 10 * φ ^ 3 + (27051 : ℝ) * δ ^ 10 * φ ^ 2 + (11482 : ℝ) * δ ^ 10 * φ
      + (1578 : ℝ) * δ ^ 10 + (1021 : ℝ) * δ ^ 9 * φ ^ 6 + (8108 : ℝ) * δ ^ 9 * φ ^ 5
      + (29780 : ℝ) * δ ^ 9 * φ ^ 4 + (60616 : ℝ) * δ ^ 9 * φ ^ 3 + (68950 : ℝ) * δ ^ 9 * φ ^ 2
      + (40671 : ℝ) * δ ^ 9 * φ + (9644 : ℝ) * δ ^ 9 + (668 : ℝ) * δ ^ 8 * φ ^ 7
      + (5118 : ℝ) * δ ^ 8 * φ ^ 6 + (20544 : ℝ) * δ ^ 8 * φ ^ 5 + (51666 : ℝ) * δ ^ 8 * φ ^ 4
      + (82144 : ℝ) * δ ^ 8 * φ ^ 3 + (79412 : ℝ) * δ ^ 8 * φ ^ 2 + (42524 : ℝ) * δ ^ 8 * φ
      + (9644 : ℝ) * δ ^ 8 + (813 : ℝ) * δ ^ 7 * φ ^ 7 + (5949 : ℝ) * δ ^ 7 * φ ^ 6
      + (22095 : ℝ) * δ ^ 7 * φ ^ 5 + (50423 : ℝ) * δ ^ 7 * φ ^ 4 + (71219 : ℝ) * δ ^ 7 * φ ^ 3
      + (59163 : ℝ) * δ ^ 7 * φ ^ 2 + (25993 : ℝ) * δ ^ 7 * φ + (4585 : ℝ) * δ ^ 7
      + (201 : ℝ) * δ ^ 6 * φ ^ 8 + (1231 : ℝ) * δ ^ 6 * φ ^ 7 + (4446 : ℝ) * δ ^ 6 * φ ^ 6
      + (11325 : ℝ) * δ ^ 6 * φ ^ 5 + (21904 : ℝ) * δ ^ 6 * φ ^ 4 + (31938 : ℝ) * δ ^ 6 * φ ^ 3
      + (31948 : ℝ) * δ ^ 6 * φ ^ 2 + (18548 : ℝ) * δ ^ 6 * φ + (4585 : ℝ) * δ ^ 6
      + (222 : ℝ) * δ ^ 5 * φ ^ 8 + (1209 : ℝ) * δ ^ 5 * φ ^ 7 + (3622 : ℝ) * δ ^ 5 * φ ^ 6
      + (7306 : ℝ) * δ ^ 5 * φ ^ 5 + (10429 : ℝ) * δ ^ 5 * φ ^ 4 + (10200 : ℝ) * δ ^ 5 * φ ^ 3
      + (6376 : ℝ) * δ ^ 5 * φ ^ 2 + (2306 : ℝ) * δ ^ 5 * φ + (372 : ℝ) * δ ^ 5
      + (24 : ℝ) * δ ^ 4 * φ ^ 9 + (91 : ℝ) * δ ^ 4 * φ ^ 8 + (256 : ℝ) * δ ^ 4 * φ ^ 7
      + (534 : ℝ) * δ ^ 4 * φ ^ 6 + (942 : ℝ) * δ ^ 4 * φ ^ 5 + (1399 : ℝ) * δ ^ 4 * φ ^ 4
      + (1802 : ℝ) * δ ^ 4 * φ ^ 3 + (1884 : ℝ) * δ ^ 4 * φ ^ 2 + (1276 : ℝ) * δ ^ 4 * φ
      + (372 : ℝ) * δ ^ 4 + (25 : ℝ) * δ ^ 3 * φ ^ 9 + (73 : ℝ) * δ ^ 3 * φ ^ 8
      + (136 : ℝ) * δ ^ 3 * φ ^ 7 + (184 : ℝ) * δ ^ 3 * φ ^ 6 + (199 : ℝ) * δ ^ 3 * φ ^ 5
      + (171 : ℝ) * δ ^ 3 * φ ^ 4 + (112 : ℝ) * δ ^ 3 * φ ^ 3 + (62 : ℝ) * δ ^ 3 * φ ^ 2
      + (23 : ℝ) * δ ^ 3 * φ + (5 : ℝ) * δ ^ 3 + (1 : ℝ) * δ ^ 2 * φ ^ 10
      + (1 : ℝ) * δ ^ 2 * φ ^ 9 + (3 : ℝ) * δ ^ 2 * φ ^ 8 + (3 : ℝ) * δ ^ 2 * φ ^ 7
      + (5 : ℝ) * δ ^ 2 * φ ^ 6 + (5 : ℝ) * δ ^ 2 * φ ^ 5 + (7 : ℝ) * δ ^ 2 * φ ^ 4
      + (7 : ℝ) * δ ^ 2 * φ ^ 3 + (9 : ℝ) * δ ^ 2 * φ ^ 2 + (9 : ℝ) * δ ^ 2 * φ
      + (5 : ℝ) * δ ^ 2 + (1 : ℝ) * δ * φ ^ 10 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_10_0_false, M_10_0_true, M_10_1_false, M_10_1_true, M_10_2_false, M_10_2_true,
    M_10_3_false, M_10_3_true, M_10_4_false, M_10_4_true, M_10_5_false, M_10_5_true,
    M_10_6_false, M_10_6_true, M_10_7_false, M_10_7_true, M_10_8_false, M_10_8_true,
    M_10_9_false, M_10_9_true, M_10_10_false, M_10_10_true]
  ring

private lemma M_11_2_false (δ φ : ℝ) :
    M_rec δ φ 2 11 false =
      (418 : ℝ) * δ ^ 11 * φ ^ 5 + (2316 : ℝ) * δ ^ 11 * φ ^ 4 + (5650 : ℝ) * δ ^ 11 * φ ^ 3
      + (6068 : ℝ) * δ ^ 11 * φ ^ 2 + (2344 : ℝ) * δ ^ 11 * φ + (-826 : ℝ) * δ ^ 10 * φ ^ 5
      + (-5930 : ℝ) * δ ^ 10 * φ ^ 4 + (-18938 : ℝ) * δ ^ 10 * φ ^ 3
      + (-30706 : ℝ) * δ ^ 10 * φ ^ 2 + (-23980 : ℝ) * δ ^ 10 * φ + (-7136 : ℝ) * δ ^ 10
      + (836 : ℝ) * δ ^ 9 * φ ^ 6 + (6248 : ℝ) * δ ^ 9 * φ ^ 5 + (23716 : ℝ) * δ ^ 9 * φ ^ 4
      + (50768 : ℝ) * δ ^ 9 * φ ^ 3 + (62054 : ℝ) * δ ^ 9 * φ ^ 2 + (40174 : ℝ) * δ ^ 9 * φ
      + (10684 : ℝ) * δ ^ 9 + (-1172 : ℝ) * δ ^ 8 * φ ^ 6 + (-8956 : ℝ) * δ ^ 8 * φ ^ 5
      + (-33132 : ℝ) * δ ^ 8 * φ ^ 4 + (-68208 : ℝ) * δ ^ 8 * φ ^ 3
      + (-76224 : ℝ) * δ ^ 8 * φ ^ 2 + (-42996 : ℝ) * δ ^ 8 * φ + (-9552 : ℝ) * δ ^ 8
      + (322 : ℝ) * δ ^ 7 * φ ^ 7 + (2088 : ℝ) * δ ^ 7 * φ ^ 6 + (8518 : ℝ) * δ ^ 7 * φ ^ 5
      + (24328 : ℝ) * δ ^ 7 * φ ^ 4 + (47406 : ℝ) * δ ^ 7 * φ ^ 3 + (55258 : ℝ) * δ ^ 7 * φ ^ 2
      + (33914 : ℝ) * δ ^ 7 * φ + (8346 : ℝ) * δ ^ 7 + (-390 : ℝ) * δ ^ 6 * φ ^ 7
      + (-2250 : ℝ) * δ ^ 6 * φ ^ 6 + (-7030 : ℝ) * δ ^ 6 * φ ^ 5 + (-15052 : ℝ) * δ ^ 6 * φ ^ 4
      + (-22332 : ℝ) * δ ^ 6 * φ ^ 3 + (-21840 : ℝ) * δ ^ 6 * φ ^ 2 + (-12248 : ℝ) * δ ^ 6 * φ
      + (-2942 : ℝ) * δ ^ 6 + (44 : ℝ) * δ ^ 5 * φ ^ 8 + (198 : ℝ) * δ ^ 5 * φ ^ 7
      + (780 : ℝ) * δ ^ 5 * φ ^ 6 + (2088 : ℝ) * δ ^ 5 * φ ^ 5 + (4394 : ℝ) * δ ^ 5 * φ ^ 4
      + (6408 : ℝ) * δ ^ 5 * φ ^ 3 + (6112 : ℝ) * δ ^ 5 * φ ^ 2 + (3268 : ℝ) * δ ^ 5 * φ
      + (732 : ℝ) * δ ^ 5 + (-48 : ℝ) * δ ^ 4 * φ ^ 8 + (-148 : ℝ) * δ ^ 4 * φ ^ 7
      + (-308 : ℝ) * δ ^ 4 * φ ^ 6 + (-516 : ℝ) * δ ^ 4 * φ ^ 5 + (-700 : ℝ) * δ ^ 4 * φ ^ 4
      + (-828 : ℝ) * δ ^ 4 * φ ^ 3 + (-764 : ℝ) * δ ^ 4 * φ ^ 2 + (-508 : ℝ) * δ ^ 4 * φ
      + (-140 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 9 + (4 : ℝ) * δ ^ 3 * φ ^ 8
      + (20 : ℝ) * δ ^ 3 * φ ^ 7 + (26 : ℝ) * δ ^ 3 * φ ^ 6 + (58 : ℝ) * δ ^ 3 * φ ^ 5
      + (60 : ℝ) * δ ^ 3 * φ ^ 4 + (76 : ℝ) * δ ^ 3 * φ ^ 3 + (42 : ℝ) * δ ^ 3 * φ ^ 2
      + (34 : ℝ) * δ ^ 3 * φ + (8 : ℝ) * δ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ ^ 9
      + (-2 : ℝ) * δ ^ 2 * φ ^ 7 + (-2 : ℝ) * δ ^ 2 * φ ^ 5 + (-2 : ℝ) * δ ^ 2 * φ ^ 3
      + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_9_0_false, M_9_0_true, M_9_1_false, M_9_1_true, M_9_2_false, M_9_2_true, M_9_3_false,
    M_9_3_true, M_9_4_false, M_9_4_true, M_9_5_false, M_9_5_true, M_9_6_false, M_9_6_true,
    M_9_7_false, M_9_7_true, M_9_8_false, M_9_8_true, M_9_9_false, M_9_9_true]
  ring

private lemma M_11_2_true (δ φ : ℝ) :
    M_rec δ φ 2 11 true =
      (-418 : ℝ) * δ ^ 11 * φ ^ 5 + (-2316 : ℝ) * δ ^ 11 * φ ^ 4 + (-5650 : ℝ) * δ ^ 11 * φ ^ 3
      + (-6068 : ℝ) * δ ^ 11 * φ ^ 2 + (-2344 : ℝ) * δ ^ 11 * φ + (-826 : ℝ) * δ ^ 10 * φ ^ 5
      + (-5930 : ℝ) * δ ^ 10 * φ ^ 4 + (-18938 : ℝ) * δ ^ 10 * φ ^ 3
      + (-30706 : ℝ) * δ ^ 10 * φ ^ 2 + (-23980 : ℝ) * δ ^ 10 * φ + (-7136 : ℝ) * δ ^ 10
      + (-836 : ℝ) * δ ^ 9 * φ ^ 6 + (-6248 : ℝ) * δ ^ 9 * φ ^ 5 + (-23716 : ℝ) * δ ^ 9 * φ ^ 4
      + (-50768 : ℝ) * δ ^ 9 * φ ^ 3 + (-62054 : ℝ) * δ ^ 9 * φ ^ 2 + (-40174 : ℝ) * δ ^ 9 * φ
      + (-10684 : ℝ) * δ ^ 9 + (-1172 : ℝ) * δ ^ 8 * φ ^ 6 + (-8956 : ℝ) * δ ^ 8 * φ ^ 5
      + (-33132 : ℝ) * δ ^ 8 * φ ^ 4 + (-68208 : ℝ) * δ ^ 8 * φ ^ 3
      + (-76224 : ℝ) * δ ^ 8 * φ ^ 2 + (-42996 : ℝ) * δ ^ 8 * φ + (-9552 : ℝ) * δ ^ 8
      + (-322 : ℝ) * δ ^ 7 * φ ^ 7 + (-2088 : ℝ) * δ ^ 7 * φ ^ 6 + (-8518 : ℝ) * δ ^ 7 * φ ^ 5
      + (-24328 : ℝ) * δ ^ 7 * φ ^ 4 + (-47406 : ℝ) * δ ^ 7 * φ ^ 3
      + (-55258 : ℝ) * δ ^ 7 * φ ^ 2 + (-33914 : ℝ) * δ ^ 7 * φ + (-8346 : ℝ) * δ ^ 7
      + (-390 : ℝ) * δ ^ 6 * φ ^ 7 + (-2250 : ℝ) * δ ^ 6 * φ ^ 6 + (-7030 : ℝ) * δ ^ 6 * φ ^ 5
      + (-15052 : ℝ) * δ ^ 6 * φ ^ 4 + (-22332 : ℝ) * δ ^ 6 * φ ^ 3
      + (-21840 : ℝ) * δ ^ 6 * φ ^ 2 + (-12248 : ℝ) * δ ^ 6 * φ + (-2942 : ℝ) * δ ^ 6
      + (-44 : ℝ) * δ ^ 5 * φ ^ 8 + (-198 : ℝ) * δ ^ 5 * φ ^ 7 + (-780 : ℝ) * δ ^ 5 * φ ^ 6
      + (-2088 : ℝ) * δ ^ 5 * φ ^ 5 + (-4394 : ℝ) * δ ^ 5 * φ ^ 4 + (-6408 : ℝ) * δ ^ 5 * φ ^ 3
      + (-6112 : ℝ) * δ ^ 5 * φ ^ 2 + (-3268 : ℝ) * δ ^ 5 * φ + (-732 : ℝ) * δ ^ 5
      + (-48 : ℝ) * δ ^ 4 * φ ^ 8 + (-148 : ℝ) * δ ^ 4 * φ ^ 7 + (-308 : ℝ) * δ ^ 4 * φ ^ 6
      + (-516 : ℝ) * δ ^ 4 * φ ^ 5 + (-700 : ℝ) * δ ^ 4 * φ ^ 4 + (-828 : ℝ) * δ ^ 4 * φ ^ 3
      + (-764 : ℝ) * δ ^ 4 * φ ^ 2 + (-508 : ℝ) * δ ^ 4 * φ + (-140 : ℝ) * δ ^ 4
      + (-2 : ℝ) * δ ^ 3 * φ ^ 9 + (-4 : ℝ) * δ ^ 3 * φ ^ 8 + (-20 : ℝ) * δ ^ 3 * φ ^ 7
      + (-26 : ℝ) * δ ^ 3 * φ ^ 6 + (-58 : ℝ) * δ ^ 3 * φ ^ 5 + (-60 : ℝ) * δ ^ 3 * φ ^ 4
      + (-76 : ℝ) * δ ^ 3 * φ ^ 3 + (-42 : ℝ) * δ ^ 3 * φ ^ 2 + (-34 : ℝ) * δ ^ 3 * φ
      + (-8 : ℝ) * δ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ ^ 9 + (-2 : ℝ) * δ ^ 2 * φ ^ 7
      + (-2 : ℝ) * δ ^ 2 * φ ^ 5 + (-2 : ℝ) * δ ^ 2 * φ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_9_0_false, M_9_0_true, M_9_1_false, M_9_1_true, M_9_2_false, M_9_2_true, M_9_3_false,
    M_9_3_true, M_9_4_false, M_9_4_true, M_9_5_false, M_9_5_true, M_9_6_false, M_9_6_true,
    M_9_7_false, M_9_7_true, M_9_8_false, M_9_8_true, M_9_9_false, M_9_9_true]
  ring

private lemma M_11_3_false (δ φ : ℝ) :
    M_rec δ φ 3 11 false =
      (-123 : ℝ) * δ ^ 11 * φ ^ 4 + (-1116 : ℝ) * δ ^ 11 * φ ^ 3 + (-3621 : ℝ) * δ ^ 11 * φ ^ 2
      + (-4872 : ℝ) * δ ^ 11 * φ + (-2202 : ℝ) * δ ^ 11 + (672 : ℝ) * δ ^ 10 * φ ^ 5
      + (4668 : ℝ) * δ ^ 10 * φ ^ 4 + (13953 : ℝ) * δ ^ 10 * φ ^ 3
      + (20223 : ℝ) * δ ^ 10 * φ ^ 2 + (14556 : ℝ) * δ ^ 10 * φ + (4272 : ℝ) * δ ^ 10
      + (-1080 : ℝ) * δ ^ 9 * φ ^ 5 + (-8748 : ℝ) * δ ^ 9 * φ ^ 4 + (-28629 : ℝ) * δ ^ 9 * φ ^ 3
      + (-43674 : ℝ) * δ ^ 9 * φ ^ 2 + (-30261 : ℝ) * δ ^ 9 * φ + (-7728 : ℝ) * δ ^ 9
      + (372 : ℝ) * δ ^ 8 * φ ^ 6 + (2490 : ℝ) * δ ^ 8 * φ ^ 5 + (10899 : ℝ) * δ ^ 8 * φ ^ 4
      + (30267 : ℝ) * δ ^ 8 * φ ^ 3 + (46653 : ℝ) * δ ^ 8 * φ ^ 2 + (34773 : ℝ) * δ ^ 8 * φ
      + (9681 : ℝ) * δ ^ 8 + (-495 : ℝ) * δ ^ 7 * φ ^ 6 + (-3150 : ℝ) * δ ^ 7 * φ ^ 5
      + (-10317 : ℝ) * δ ^ 7 * φ ^ 4 + (-22290 : ℝ) * δ ^ 7 * φ ^ 3
      + (-28686 : ℝ) * δ ^ 7 * φ ^ 2 + (-19752 : ℝ) * δ ^ 7 * φ + (-5400 : ℝ) * δ ^ 7
      + (60 : ℝ) * δ ^ 6 * φ ^ 7 + (306 : ℝ) * δ ^ 6 * φ ^ 6 + (1557 : ℝ) * δ ^ 6 * φ ^ 5
      + (4671 : ℝ) * δ ^ 6 * φ ^ 4 + (9666 : ℝ) * δ ^ 6 * φ ^ 3 + (11244 : ℝ) * δ ^ 6 * φ ^ 2
      + (6804 : ℝ) * δ ^ 6 * φ + (1728 : ℝ) * δ ^ 6 + (-69 : ℝ) * δ ^ 5 * φ ^ 7
      + (-234 : ℝ) * δ ^ 5 * φ ^ 6 + (-561 : ℝ) * δ ^ 5 * φ ^ 5 + (-1236 : ℝ) * δ ^ 5 * φ ^ 4
      + (-2040 : ℝ) * δ ^ 5 * φ ^ 3 + (-2406 : ℝ) * δ ^ 5 * φ ^ 2 + (-1368 : ℝ) * δ ^ 5 * φ
      + (-402 : ℝ) * δ ^ 5 + (3 : ℝ) * δ ^ 4 * φ ^ 8 + (9 : ℝ) * δ ^ 4 * φ ^ 7
      + (60 : ℝ) * δ ^ 4 * φ ^ 6 + (72 : ℝ) * δ ^ 4 * φ ^ 5 + (201 : ℝ) * δ ^ 4 * φ ^ 4
      + (189 : ℝ) * δ ^ 4 * φ ^ 3 + (282 : ℝ) * δ ^ 4 * φ ^ 2 + (120 : ℝ) * δ ^ 4 * φ
      + (54 : ℝ) * δ ^ 4 + (-3 : ℝ) * δ ^ 3 * φ ^ 8 + (-9 : ℝ) * δ ^ 3 * φ ^ 6
      + (-15 : ℝ) * δ ^ 3 * φ ^ 4 + (-15 : ℝ) * δ ^ 3 * φ ^ 2 + (-3 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_8_0_false, M_8_0_true, M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false,
    M_8_3_true, M_8_4_false, M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true,
    M_8_7_false, M_8_7_true, M_8_8_false, M_8_8_true]
  ring

private lemma M_11_3_true (δ φ : ℝ) :
    M_rec δ φ 3 11 true =
      (123 : ℝ) * δ ^ 11 * φ ^ 4 + (1116 : ℝ) * δ ^ 11 * φ ^ 3 + (3621 : ℝ) * δ ^ 11 * φ ^ 2
      + (4872 : ℝ) * δ ^ 11 * φ + (2202 : ℝ) * δ ^ 11 + (672 : ℝ) * δ ^ 10 * φ ^ 5
      + (4668 : ℝ) * δ ^ 10 * φ ^ 4 + (13953 : ℝ) * δ ^ 10 * φ ^ 3
      + (20223 : ℝ) * δ ^ 10 * φ ^ 2 + (14556 : ℝ) * δ ^ 10 * φ + (4272 : ℝ) * δ ^ 10
      + (1080 : ℝ) * δ ^ 9 * φ ^ 5 + (8748 : ℝ) * δ ^ 9 * φ ^ 4 + (28629 : ℝ) * δ ^ 9 * φ ^ 3
      + (43674 : ℝ) * δ ^ 9 * φ ^ 2 + (30261 : ℝ) * δ ^ 9 * φ + (7728 : ℝ) * δ ^ 9
      + (372 : ℝ) * δ ^ 8 * φ ^ 6 + (2490 : ℝ) * δ ^ 8 * φ ^ 5 + (10899 : ℝ) * δ ^ 8 * φ ^ 4
      + (30267 : ℝ) * δ ^ 8 * φ ^ 3 + (46653 : ℝ) * δ ^ 8 * φ ^ 2 + (34773 : ℝ) * δ ^ 8 * φ
      + (9681 : ℝ) * δ ^ 8 + (495 : ℝ) * δ ^ 7 * φ ^ 6 + (3150 : ℝ) * δ ^ 7 * φ ^ 5
      + (10317 : ℝ) * δ ^ 7 * φ ^ 4 + (22290 : ℝ) * δ ^ 7 * φ ^ 3 + (28686 : ℝ) * δ ^ 7 * φ ^ 2
      + (19752 : ℝ) * δ ^ 7 * φ + (5400 : ℝ) * δ ^ 7 + (60 : ℝ) * δ ^ 6 * φ ^ 7
      + (306 : ℝ) * δ ^ 6 * φ ^ 6 + (1557 : ℝ) * δ ^ 6 * φ ^ 5 + (4671 : ℝ) * δ ^ 6 * φ ^ 4
      + (9666 : ℝ) * δ ^ 6 * φ ^ 3 + (11244 : ℝ) * δ ^ 6 * φ ^ 2 + (6804 : ℝ) * δ ^ 6 * φ
      + (1728 : ℝ) * δ ^ 6 + (69 : ℝ) * δ ^ 5 * φ ^ 7 + (234 : ℝ) * δ ^ 5 * φ ^ 6
      + (561 : ℝ) * δ ^ 5 * φ ^ 5 + (1236 : ℝ) * δ ^ 5 * φ ^ 4 + (2040 : ℝ) * δ ^ 5 * φ ^ 3
      + (2406 : ℝ) * δ ^ 5 * φ ^ 2 + (1368 : ℝ) * δ ^ 5 * φ + (402 : ℝ) * δ ^ 5
      + (3 : ℝ) * δ ^ 4 * φ ^ 8 + (9 : ℝ) * δ ^ 4 * φ ^ 7 + (60 : ℝ) * δ ^ 4 * φ ^ 6
      + (72 : ℝ) * δ ^ 4 * φ ^ 5 + (201 : ℝ) * δ ^ 4 * φ ^ 4 + (189 : ℝ) * δ ^ 4 * φ ^ 3
      + (282 : ℝ) * δ ^ 4 * φ ^ 2 + (120 : ℝ) * δ ^ 4 * φ + (54 : ℝ) * δ ^ 4
      + (3 : ℝ) * δ ^ 3 * φ ^ 8 + (9 : ℝ) * δ ^ 3 * φ ^ 6 + (15 : ℝ) * δ ^ 3 * φ ^ 4
      + (15 : ℝ) * δ ^ 3 * φ ^ 2 + (3 : ℝ) * δ ^ 3 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_8_0_false, M_8_0_true, M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false,
    M_8_3_true, M_8_4_false, M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true,
    M_8_7_false, M_8_7_true, M_8_8_false, M_8_8_true]
  ring

private lemma M_11_4_false (δ φ : ℝ) :
    M_rec δ φ 4 11 false =
      (344 : ℝ) * δ ^ 11 * φ ^ 4 + (1824 : ℝ) * δ ^ 11 * φ ^ 3 + (3184 : ℝ) * δ ^ 11 * φ ^ 2
      + (1720 : ℝ) * δ ^ 11 * φ + (-648 : ℝ) * δ ^ 10 * φ ^ 4 + (-4932 : ℝ) * δ ^ 10 * φ ^ 3
      + (-12036 : ℝ) * δ ^ 10 * φ ^ 2 + (-11136 : ℝ) * δ ^ 10 * φ + (-3280 : ℝ) * δ ^ 10
      + (360 : ℝ) * δ ^ 9 * φ ^ 5 + (2432 : ℝ) * δ ^ 9 * φ ^ 4 + (9804 : ℝ) * δ ^ 9 * φ ^ 3
      + (20628 : ℝ) * δ ^ 9 * φ ^ 2 + (19836 : ℝ) * δ ^ 9 * φ + (7000 : ℝ) * δ ^ 9
      + (-528 : ℝ) * δ ^ 8 * φ ^ 5 + (-3844 : ℝ) * δ ^ 8 * φ ^ 4 + (-12356 : ℝ) * δ ^ 8 * φ ^ 3
      + (-20796 : ℝ) * δ ^ 8 * φ ^ 2 + (-16948 : ℝ) * δ ^ 8 * φ + (-5588 : ℝ) * δ ^ 8
      + (72 : ℝ) * δ ^ 7 * φ ^ 6 + (400 : ℝ) * δ ^ 7 * φ ^ 5 + (2488 : ℝ) * δ ^ 7 * φ ^ 4
      + (7752 : ℝ) * δ ^ 7 * φ ^ 3 + (12184 : ℝ) * δ ^ 7 * φ ^ 2 + (9008 : ℝ) * δ ^ 7 * φ
      + (2416 : ℝ) * δ ^ 7 + (-88 : ℝ) * δ ^ 6 * φ ^ 6 + (-340 : ℝ) * δ ^ 6 * φ ^ 5
      + (-920 : ℝ) * δ ^ 6 * φ ^ 4 + (-2544 : ℝ) * δ ^ 6 * φ ^ 3 + (-3592 : ℝ) * δ ^ 6 * φ ^ 2
      + (-2984 : ℝ) * δ ^ 6 * φ + (-620 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 7
      + (16 : ℝ) * δ ^ 5 * φ ^ 6 + (132 : ℝ) * δ ^ 5 * φ ^ 5 + (148 : ℝ) * δ ^ 5 * φ ^ 4
      + (504 : ℝ) * δ ^ 5 * φ ^ 3 + (428 : ℝ) * δ ^ 5 * φ ^ 2 + (544 : ℝ) * δ ^ 5 * φ
      + (72 : ℝ) * δ ^ 5 + (-4 : ℝ) * δ ^ 4 * φ ^ 7 + (-24 : ℝ) * δ ^ 4 * φ ^ 5
      + (-52 : ℝ) * δ ^ 4 * φ ^ 3 + (-40 : ℝ) * δ ^ 4 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_11_4_true (δ φ : ℝ) :
    M_rec δ φ 4 11 true =
      (-344 : ℝ) * δ ^ 11 * φ ^ 4 + (-1824 : ℝ) * δ ^ 11 * φ ^ 3 + (-3184 : ℝ) * δ ^ 11 * φ ^ 2
      + (-1720 : ℝ) * δ ^ 11 * φ + (-648 : ℝ) * δ ^ 10 * φ ^ 4 + (-4932 : ℝ) * δ ^ 10 * φ ^ 3
      + (-12036 : ℝ) * δ ^ 10 * φ ^ 2 + (-11136 : ℝ) * δ ^ 10 * φ + (-3280 : ℝ) * δ ^ 10
      + (-360 : ℝ) * δ ^ 9 * φ ^ 5 + (-2432 : ℝ) * δ ^ 9 * φ ^ 4 + (-9804 : ℝ) * δ ^ 9 * φ ^ 3
      + (-20628 : ℝ) * δ ^ 9 * φ ^ 2 + (-19836 : ℝ) * δ ^ 9 * φ + (-7000 : ℝ) * δ ^ 9
      + (-528 : ℝ) * δ ^ 8 * φ ^ 5 + (-3844 : ℝ) * δ ^ 8 * φ ^ 4 + (-12356 : ℝ) * δ ^ 8 * φ ^ 3
      + (-20796 : ℝ) * δ ^ 8 * φ ^ 2 + (-16948 : ℝ) * δ ^ 8 * φ + (-5588 : ℝ) * δ ^ 8
      + (-72 : ℝ) * δ ^ 7 * φ ^ 6 + (-400 : ℝ) * δ ^ 7 * φ ^ 5 + (-2488 : ℝ) * δ ^ 7 * φ ^ 4
      + (-7752 : ℝ) * δ ^ 7 * φ ^ 3 + (-12184 : ℝ) * δ ^ 7 * φ ^ 2 + (-9008 : ℝ) * δ ^ 7 * φ
      + (-2416 : ℝ) * δ ^ 7 + (-88 : ℝ) * δ ^ 6 * φ ^ 6 + (-340 : ℝ) * δ ^ 6 * φ ^ 5
      + (-920 : ℝ) * δ ^ 6 * φ ^ 4 + (-2544 : ℝ) * δ ^ 6 * φ ^ 3 + (-3592 : ℝ) * δ ^ 6 * φ ^ 2
      + (-2984 : ℝ) * δ ^ 6 * φ + (-620 : ℝ) * δ ^ 6 + (-4 : ℝ) * δ ^ 5 * φ ^ 7
      + (-16 : ℝ) * δ ^ 5 * φ ^ 6 + (-132 : ℝ) * δ ^ 5 * φ ^ 5 + (-148 : ℝ) * δ ^ 5 * φ ^ 4
      + (-504 : ℝ) * δ ^ 5 * φ ^ 3 + (-428 : ℝ) * δ ^ 5 * φ ^ 2 + (-544 : ℝ) * δ ^ 5 * φ
      + (-72 : ℝ) * δ ^ 5 + (-4 : ℝ) * δ ^ 4 * φ ^ 7 + (-24 : ℝ) * δ ^ 4 * φ ^ 5
      + (-52 : ℝ) * δ ^ 4 * φ ^ 3 + (-40 : ℝ) * δ ^ 4 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_11_5_false (δ φ : ℝ) :
    M_rec δ φ 5 11 false =
      (-55 : ℝ) * δ ^ 11 * φ ^ 3 + (-715 : ℝ) * δ ^ 11 * φ ^ 2 + (-1750 : ℝ) * δ ^ 11 * φ
      + (-1120 : ℝ) * δ ^ 11 + (295 : ℝ) * δ ^ 10 * φ ^ 4 + (1925 : ℝ) * δ ^ 10 * φ ^ 3
      + (4435 : ℝ) * δ ^ 10 * φ ^ 2 + (5630 : ℝ) * δ ^ 10 * φ + (2730 : ℝ) * δ ^ 10
      + (-480 : ℝ) * δ ^ 9 * φ ^ 4 + (-4095 : ℝ) * δ ^ 9 * φ ^ 3 + (-8785 : ℝ) * δ ^ 9 * φ ^ 2
      + (-8560 : ℝ) * δ ^ 9 * φ + (-3105 : ℝ) * δ ^ 9 + (80 : ℝ) * δ ^ 8 * φ ^ 5
      + (465 : ℝ) * δ ^ 8 * φ ^ 4 + (3330 : ℝ) * δ ^ 8 * φ ^ 3 + (7910 : ℝ) * δ ^ 8 * φ ^ 2
      + (7190 : ℝ) * δ ^ 8 * φ + (2475 : ℝ) * δ ^ 8 + (-105 : ℝ) * δ ^ 7 * φ ^ 5
      + (-475 : ℝ) * δ ^ 7 * φ ^ 4 + (-1350 : ℝ) * δ ^ 7 * φ ^ 3 + (-3680 : ℝ) * δ ^ 7 * φ ^ 2
      + (-2980 : ℝ) * δ ^ 7 * φ + (-1310 : ℝ) * δ ^ 7 + (5 : ℝ) * δ ^ 6 * φ ^ 6
      + (25 : ℝ) * δ ^ 6 * φ ^ 5 + (245 : ℝ) * δ ^ 6 * φ ^ 4 + (245 : ℝ) * δ ^ 6 * φ ^ 3
      + (950 : ℝ) * δ ^ 6 * φ ^ 2 + (470 : ℝ) * δ ^ 6 * φ + (370 : ℝ) * δ ^ 6
      + (-5 : ℝ) * δ ^ 5 * φ ^ 6 + (-50 : ℝ) * δ ^ 5 * φ ^ 4 + (-115 : ℝ) * δ ^ 5 * φ ^ 2
      + (-40 : ℝ) * δ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_11_5_true (δ φ : ℝ) :
    M_rec δ φ 5 11 true =
      (55 : ℝ) * δ ^ 11 * φ ^ 3 + (715 : ℝ) * δ ^ 11 * φ ^ 2 + (1750 : ℝ) * δ ^ 11 * φ
      + (1120 : ℝ) * δ ^ 11 + (295 : ℝ) * δ ^ 10 * φ ^ 4 + (1925 : ℝ) * δ ^ 10 * φ ^ 3
      + (4435 : ℝ) * δ ^ 10 * φ ^ 2 + (5630 : ℝ) * δ ^ 10 * φ + (2730 : ℝ) * δ ^ 10
      + (480 : ℝ) * δ ^ 9 * φ ^ 4 + (4095 : ℝ) * δ ^ 9 * φ ^ 3 + (8785 : ℝ) * δ ^ 9 * φ ^ 2
      + (8560 : ℝ) * δ ^ 9 * φ + (3105 : ℝ) * δ ^ 9 + (80 : ℝ) * δ ^ 8 * φ ^ 5
      + (465 : ℝ) * δ ^ 8 * φ ^ 4 + (3330 : ℝ) * δ ^ 8 * φ ^ 3 + (7910 : ℝ) * δ ^ 8 * φ ^ 2
      + (7190 : ℝ) * δ ^ 8 * φ + (2475 : ℝ) * δ ^ 8 + (105 : ℝ) * δ ^ 7 * φ ^ 5
      + (475 : ℝ) * δ ^ 7 * φ ^ 4 + (1350 : ℝ) * δ ^ 7 * φ ^ 3 + (3680 : ℝ) * δ ^ 7 * φ ^ 2
      + (2980 : ℝ) * δ ^ 7 * φ + (1310 : ℝ) * δ ^ 7 + (5 : ℝ) * δ ^ 6 * φ ^ 6
      + (25 : ℝ) * δ ^ 6 * φ ^ 5 + (245 : ℝ) * δ ^ 6 * φ ^ 4 + (245 : ℝ) * δ ^ 6 * φ ^ 3
      + (950 : ℝ) * δ ^ 6 * φ ^ 2 + (470 : ℝ) * δ ^ 6 * φ + (370 : ℝ) * δ ^ 6
      + (5 : ℝ) * δ ^ 5 * φ ^ 6 + (50 : ℝ) * δ ^ 5 * φ ^ 4 + (115 : ℝ) * δ ^ 5 * φ ^ 2
      + (40 : ℝ) * δ ^ 5 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_11_6_false (δ φ : ℝ) :
    M_rec δ φ 6 11 false =
      (186 : ℝ) * δ ^ 11 * φ ^ 3 + (840 : ℝ) * δ ^ 11 * φ ^ 2 + (612 : ℝ) * δ ^ 11 * φ
      + (-342 : ℝ) * δ ^ 10 * φ ^ 3 + (-2478 : ℝ) * δ ^ 10 * φ ^ 2 + (-2616 : ℝ) * δ ^ 10 * φ
      + (-570 : ℝ) * δ ^ 10 + (84 : ℝ) * δ ^ 9 * φ ^ 4 + (486 : ℝ) * δ ^ 9 * φ ^ 3
      + (2808 : ℝ) * δ ^ 9 * φ ^ 2 + (3972 : ℝ) * δ ^ 9 * φ + (1230 : ℝ) * δ ^ 9
      + (-120 : ℝ) * δ ^ 8 * φ ^ 4 + (-648 : ℝ) * δ ^ 8 * φ ^ 3 + (-1488 : ℝ) * δ ^ 8 * φ ^ 2
      + (-2832 : ℝ) * δ ^ 8 * φ + (-852 : ℝ) * δ ^ 8 + (6 : ℝ) * δ ^ 7 * φ ^ 5
      + (36 : ℝ) * δ ^ 7 * φ ^ 4 + (408 : ℝ) * δ ^ 7 * φ ^ 3 + (318 : ℝ) * δ ^ 7 * φ ^ 2
      + (1020 : ℝ) * δ ^ 7 * φ + (192 : ℝ) * δ ^ 7 + (-6 : ℝ) * δ ^ 6 * φ ^ 5
      + (-90 : ℝ) * δ ^ 6 * φ ^ 3 + (-156 : ℝ) * δ ^ 6 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_11_6_true (δ φ : ℝ) :
    M_rec δ φ 6 11 true =
      (-186 : ℝ) * δ ^ 11 * φ ^ 3 + (-840 : ℝ) * δ ^ 11 * φ ^ 2 + (-612 : ℝ) * δ ^ 11 * φ
      + (-342 : ℝ) * δ ^ 10 * φ ^ 3 + (-2478 : ℝ) * δ ^ 10 * φ ^ 2 + (-2616 : ℝ) * δ ^ 10 * φ
      + (-570 : ℝ) * δ ^ 10 + (-84 : ℝ) * δ ^ 9 * φ ^ 4 + (-486 : ℝ) * δ ^ 9 * φ ^ 3
      + (-2808 : ℝ) * δ ^ 9 * φ ^ 2 + (-3972 : ℝ) * δ ^ 9 * φ + (-1230 : ℝ) * δ ^ 9
      + (-120 : ℝ) * δ ^ 8 * φ ^ 4 + (-648 : ℝ) * δ ^ 8 * φ ^ 3 + (-1488 : ℝ) * δ ^ 8 * φ ^ 2
      + (-2832 : ℝ) * δ ^ 8 * φ + (-852 : ℝ) * δ ^ 8 + (-6 : ℝ) * δ ^ 7 * φ ^ 5
      + (-36 : ℝ) * δ ^ 7 * φ ^ 4 + (-408 : ℝ) * δ ^ 7 * φ ^ 3 + (-318 : ℝ) * δ ^ 7 * φ ^ 2
      + (-1020 : ℝ) * δ ^ 7 * φ + (-192 : ℝ) * δ ^ 7 + (-6 : ℝ) * δ ^ 6 * φ ^ 5
      + (-90 : ℝ) * δ ^ 6 * φ ^ 3 + (-156 : ℝ) * δ ^ 6 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_11_7_false (δ φ : ℝ) :
    M_rec δ φ 7 11 false =
      (-21 : ℝ) * δ ^ 11 * φ ^ 2 + (-322 : ℝ) * δ ^ 11 * φ + (-294 : ℝ) * δ ^ 11
      + (84 : ℝ) * δ ^ 10 * φ ^ 3 + (364 : ℝ) * δ ^ 10 * φ ^ 2 + (819 : ℝ) * δ ^ 10 * φ
      + (735 : ℝ) * δ ^ 10 + (-133 : ℝ) * δ ^ 9 * φ ^ 3 + (-784 : ℝ) * δ ^ 9 * φ ^ 2
      + (-707 : ℝ) * δ ^ 9 * φ + (-686 : ℝ) * δ ^ 9 + (7 : ℝ) * δ ^ 8 * φ ^ 4
      + (49 : ℝ) * δ ^ 8 * φ ^ 3 + (588 : ℝ) * δ ^ 8 * φ ^ 2 + (210 : ℝ) * δ ^ 8 * φ
      + (301 : ℝ) * δ ^ 8 + (-7 : ℝ) * δ ^ 7 * φ ^ 4 + (-147 : ℝ) * δ ^ 7 * φ ^ 2
      + (-56 : ℝ) * δ ^ 7 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_11_7_true (δ φ : ℝ) :
    M_rec δ φ 7 11 true =
      (21 : ℝ) * δ ^ 11 * φ ^ 2 + (322 : ℝ) * δ ^ 11 * φ + (294 : ℝ) * δ ^ 11
      + (84 : ℝ) * δ ^ 10 * φ ^ 3 + (364 : ℝ) * δ ^ 10 * φ ^ 2 + (819 : ℝ) * δ ^ 10 * φ
      + (735 : ℝ) * δ ^ 10 + (133 : ℝ) * δ ^ 9 * φ ^ 3 + (784 : ℝ) * δ ^ 9 * φ ^ 2
      + (707 : ℝ) * δ ^ 9 * φ + (686 : ℝ) * δ ^ 9 + (7 : ℝ) * δ ^ 8 * φ ^ 4
      + (49 : ℝ) * δ ^ 8 * φ ^ 3 + (588 : ℝ) * δ ^ 8 * φ ^ 2 + (210 : ℝ) * δ ^ 8 * φ
      + (301 : ℝ) * δ ^ 8 + (7 : ℝ) * δ ^ 7 * φ ^ 4 + (147 : ℝ) * δ ^ 7 * φ ^ 2
      + (56 : ℝ) * δ ^ 7 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_11_8_false (δ φ : ℝ) :
    M_rec δ φ 8 11 false =
      (80 : ℝ) * δ ^ 11 * φ ^ 2 + (128 : ℝ) * δ ^ 11 * φ + (-144 : ℝ) * δ ^ 10 * φ ^ 2
      + (-376 : ℝ) * δ ^ 10 * φ + (-8 : ℝ) * δ ^ 10 + (8 : ℝ) * δ ^ 9 * φ ^ 3
      + (64 : ℝ) * δ ^ 9 * φ ^ 2 + (360 : ℝ) * δ ^ 9 * φ + (8 : ℝ) * δ ^ 9
      + (-8 : ℝ) * δ ^ 8 * φ ^ 3 + (-112 : ℝ) * δ ^ 8 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_11_8_true (δ φ : ℝ) :
    M_rec δ φ 8 11 true =
      (-80 : ℝ) * δ ^ 11 * φ ^ 2 + (-128 : ℝ) * δ ^ 11 * φ + (-144 : ℝ) * δ ^ 10 * φ ^ 2
      + (-376 : ℝ) * δ ^ 10 * φ + (-8 : ℝ) * δ ^ 10 + (-8 : ℝ) * δ ^ 9 * φ ^ 3
      + (-64 : ℝ) * δ ^ 9 * φ ^ 2 + (-360 : ℝ) * δ ^ 9 * φ + (-8 : ℝ) * δ ^ 9
      + (-8 : ℝ) * δ ^ 8 * φ ^ 3 + (-112 : ℝ) * δ ^ 8 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_11_9_false (δ φ : ℝ) :
    M_rec δ φ 9 11 false =
      (-9 : ℝ) * δ ^ 11 * φ + (-45 : ℝ) * δ ^ 11 + (9 : ℝ) * δ ^ 10 * φ ^ 2
      + (9 : ℝ) * δ ^ 10 * φ + (81 : ℝ) * δ ^ 10 + (-9 : ℝ) * δ ^ 9 * φ ^ 2 + (-36 : ℝ) * δ ^ 9 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_11_9_true (δ φ : ℝ) :
    M_rec δ φ 9 11 true =
      (9 : ℝ) * δ ^ 11 * φ + (45 : ℝ) * δ ^ 11 + (9 : ℝ) * δ ^ 10 * φ ^ 2 + (9 : ℝ) * δ ^ 10 * φ
      + (81 : ℝ) * δ ^ 10 + (9 : ℝ) * δ ^ 9 * φ ^ 2 + (36 : ℝ) * δ ^ 9 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_11_10_false (δ φ : ℝ) :
    M_rec δ φ 10 11 false =
      (10 : ℝ) * δ ^ 11 * φ + (-10 : ℝ) * δ ^ 10 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_11_10_true (δ φ : ℝ) :
    M_rec δ φ 10 11 true =
      (-10 : ℝ) * δ ^ 11 * φ + (-10 : ℝ) * δ ^ 10 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_11_11_false (δ φ : ℝ) :
    M_rec δ φ 11 11 false =
      (-1 : ℝ) * δ ^ 11 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_11_11_true (δ φ : ℝ) :
    M_rec δ φ 11 11 true =
      (1 : ℝ) * δ ^ 11 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_12_1_false (δ φ : ℝ) :
    M_rec δ φ 1 12 false =
      (-571 : ℝ) * δ ^ 12 * φ ^ 6 + (-3850 : ℝ) * δ ^ 12 * φ ^ 5 + (-12230 : ℝ) * δ ^ 12 * φ ^ 4
      + (-20282 : ℝ) * δ ^ 12 * φ ^ 3 + (-16613 : ℝ) * δ ^ 12 * φ ^ 2 + (-5240 : ℝ) * δ ^ 12 * φ
      + (1244 : ℝ) * δ ^ 11 * φ ^ 6 + (10393 : ℝ) * δ ^ 11 * φ ^ 5
      + (40665 : ℝ) * δ ^ 11 * φ ^ 4 + (90111 : ℝ) * δ ^ 11 * φ ^ 3
      + (114059 : ℝ) * δ ^ 11 * φ ^ 2 + (75854 : ℝ) * δ ^ 11 * φ + (20390 : ℝ) * δ ^ 11
      + (-1857 : ℝ) * δ ^ 10 * φ ^ 7 + (-16469 : ℝ) * δ ^ 10 * φ ^ 6
      + (-71783 : ℝ) * δ ^ 10 * φ ^ 5 + (-182974 : ℝ) * δ ^ 10 * φ ^ 4
      + (-277585 : ℝ) * δ ^ 10 * φ ^ 3 + (-241987 : ℝ) * δ ^ 10 * φ ^ 2
      + (-110735 : ℝ) * δ ^ 10 * φ + (-20390 : ℝ) * δ ^ 10 + (2525 : ℝ) * δ ^ 9 * φ ^ 7
      + (22458 : ℝ) * δ ^ 9 * φ ^ 6 + (97838 : ℝ) * δ ^ 9 * φ ^ 5 + (254672 : ℝ) * δ ^ 9 * φ ^ 4
      + (404766 : ℝ) * δ ^ 9 * φ ^ 3 + (377612 : ℝ) * δ ^ 9 * φ ^ 2 + (187706 : ℝ) * δ ^ 9 * φ
      + (38093 : ℝ) * δ ^ 9 + (-1135 : ℝ) * δ ^ 8 * φ ^ 8 + (-9272 : ℝ) * δ ^ 8 * φ ^ 7
      + (-40936 : ℝ) * δ ^ 8 * φ ^ 6 + (-120672 : ℝ) * δ ^ 8 * φ ^ 5
      + (-250558 : ℝ) * δ ^ 8 * φ ^ 4 + (-356446 : ℝ) * δ ^ 8 * φ ^ 3
      + (-325738 : ℝ) * δ ^ 8 * φ ^ 2 + (-169890 : ℝ) * δ ^ 8 * φ + (-38093 : ℝ) * δ ^ 8
      + (1336 : ℝ) * δ ^ 7 * φ ^ 8 + (10285 : ℝ) * δ ^ 7 * φ ^ 7 + (41369 : ℝ) * δ ^ 7 * φ ^ 6
      + (108357 : ℝ) * δ ^ 7 * φ ^ 5 + (193217 : ℝ) * δ ^ 7 * φ ^ 4
      + (226251 : ℝ) * δ ^ 7 * φ ^ 3 + (162241 : ℝ) * δ ^ 7 * φ ^ 2 + (63515 : ℝ) * δ ^ 7 * φ
      + (10245 : ℝ) * δ ^ 7 + (-266 : ℝ) * δ ^ 6 * φ ^ 9 + (-1681 : ℝ) * δ ^ 6 * φ ^ 8
      + (-6338 : ℝ) * δ ^ 6 * φ ^ 7 + (-17182 : ℝ) * δ ^ 6 * φ ^ 6
      + (-36540 : ℝ) * δ ^ 6 * φ ^ 5 + (-61944 : ℝ) * δ ^ 6 * φ ^ 4
      + (-82252 : ℝ) * δ ^ 6 * φ ^ 3 + (-77060 : ℝ) * δ ^ 6 * φ ^ 2 + (-42828 : ℝ) * δ ^ 6 * φ
      + (-10245 : ℝ) * δ ^ 6 + (290 : ℝ) * δ ^ 5 * φ ^ 9 + (1622 : ℝ) * δ ^ 5 * φ ^ 8
      + (5074 : ℝ) * δ ^ 5 * φ ^ 7 + (10989 : ℝ) * δ ^ 5 * φ ^ 6 + (17811 : ℝ) * δ ^ 5 * φ ^ 5
      + (21452 : ℝ) * δ ^ 5 * φ ^ 4 + (18308 : ℝ) * δ ^ 5 * φ ^ 3 + (10416 : ℝ) * δ ^ 5 * φ ^ 2
      + (3562 : ℝ) * δ ^ 5 * φ + (566 : ℝ) * δ ^ 5 + (-27 : ℝ) * δ ^ 4 * φ ^ 10
      + (-104 : ℝ) * δ ^ 4 * φ ^ 9 + (-298 : ℝ) * δ ^ 4 * φ ^ 8 + (-636 : ℝ) * δ ^ 4 * φ ^ 7
      + (-1159 : ℝ) * δ ^ 4 * φ ^ 6 + (-1800 : ℝ) * δ ^ 4 * φ ^ 5 + (-2488 : ℝ) * δ ^ 4 * φ ^ 4
      + (-2990 : ℝ) * δ ^ 4 * φ ^ 3 + (-2987 : ℝ) * δ ^ 4 * φ ^ 2 + (-1960 : ℝ) * δ ^ 4 * φ
      + (-566 : ℝ) * δ ^ 4 + (28 : ℝ) * δ ^ 3 * φ ^ 10 + (83 : ℝ) * δ ^ 3 * φ ^ 9
      + (159 : ℝ) * δ ^ 3 * φ ^ 8 + (224 : ℝ) * δ ^ 3 * φ ^ 7 + (262 : ℝ) * δ ^ 3 * φ ^ 6
      + (253 : ℝ) * δ ^ 3 * φ ^ 5 + (195 : ℝ) * δ ^ 3 * φ ^ 4 + (128 : ℝ) * δ ^ 3 * φ ^ 3
      + (66 : ℝ) * δ ^ 3 * φ ^ 2 + (27 : ℝ) * δ ^ 3 * φ + (5 : ℝ) * δ ^ 3
      + (-1 : ℝ) * δ ^ 2 * φ ^ 11 + (-1 : ℝ) * δ ^ 2 * φ ^ 10 + (-3 : ℝ) * δ ^ 2 * φ ^ 9
      + (-3 : ℝ) * δ ^ 2 * φ ^ 8 + (-5 : ℝ) * δ ^ 2 * φ ^ 7 + (-5 : ℝ) * δ ^ 2 * φ ^ 6
      + (-7 : ℝ) * δ ^ 2 * φ ^ 5 + (-7 : ℝ) * δ ^ 2 * φ ^ 4 + (-9 : ℝ) * δ ^ 2 * φ ^ 3
      + (-9 : ℝ) * δ ^ 2 * φ ^ 2 + (-11 : ℝ) * δ ^ 2 * φ + (-5 : ℝ) * δ ^ 2
      + (1 : ℝ) * δ * φ ^ 11 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_11_0_false, M_11_0_true, M_11_1_false, M_11_1_true, M_11_2_false, M_11_2_true,
    M_11_3_false, M_11_3_true, M_11_4_false, M_11_4_true, M_11_5_false, M_11_5_true,
    M_11_6_false, M_11_6_true, M_11_7_false, M_11_7_true, M_11_8_false, M_11_8_true,
    M_11_9_false, M_11_9_true, M_11_10_false, M_11_10_true, M_11_11_false, M_11_11_true]
  ring

private lemma M_12_1_true (δ φ : ℝ) :
    M_rec δ φ 1 12 true =
      (-571 : ℝ) * δ ^ 12 * φ ^ 6 + (-3850 : ℝ) * δ ^ 12 * φ ^ 5 + (-12230 : ℝ) * δ ^ 12 * φ ^ 4
      + (-20282 : ℝ) * δ ^ 12 * φ ^ 3 + (-16613 : ℝ) * δ ^ 12 * φ ^ 2 + (-5240 : ℝ) * δ ^ 12 * φ
      + (-1244 : ℝ) * δ ^ 11 * φ ^ 6 + (-10393 : ℝ) * δ ^ 11 * φ ^ 5
      + (-40665 : ℝ) * δ ^ 11 * φ ^ 4 + (-90111 : ℝ) * δ ^ 11 * φ ^ 3
      + (-114059 : ℝ) * δ ^ 11 * φ ^ 2 + (-75854 : ℝ) * δ ^ 11 * φ + (-20390 : ℝ) * δ ^ 11
      + (-1857 : ℝ) * δ ^ 10 * φ ^ 7 + (-16469 : ℝ) * δ ^ 10 * φ ^ 6
      + (-71783 : ℝ) * δ ^ 10 * φ ^ 5 + (-182974 : ℝ) * δ ^ 10 * φ ^ 4
      + (-277585 : ℝ) * δ ^ 10 * φ ^ 3 + (-241987 : ℝ) * δ ^ 10 * φ ^ 2
      + (-110735 : ℝ) * δ ^ 10 * φ + (-20390 : ℝ) * δ ^ 10 + (-2525 : ℝ) * δ ^ 9 * φ ^ 7
      + (-22458 : ℝ) * δ ^ 9 * φ ^ 6 + (-97838 : ℝ) * δ ^ 9 * φ ^ 5
      + (-254672 : ℝ) * δ ^ 9 * φ ^ 4 + (-404766 : ℝ) * δ ^ 9 * φ ^ 3
      + (-377612 : ℝ) * δ ^ 9 * φ ^ 2 + (-187706 : ℝ) * δ ^ 9 * φ + (-38093 : ℝ) * δ ^ 9
      + (-1135 : ℝ) * δ ^ 8 * φ ^ 8 + (-9272 : ℝ) * δ ^ 8 * φ ^ 7 + (-40936 : ℝ) * δ ^ 8 * φ ^ 6
      + (-120672 : ℝ) * δ ^ 8 * φ ^ 5 + (-250558 : ℝ) * δ ^ 8 * φ ^ 4
      + (-356446 : ℝ) * δ ^ 8 * φ ^ 3 + (-325738 : ℝ) * δ ^ 8 * φ ^ 2
      + (-169890 : ℝ) * δ ^ 8 * φ + (-38093 : ℝ) * δ ^ 8 + (-1336 : ℝ) * δ ^ 7 * φ ^ 8
      + (-10285 : ℝ) * δ ^ 7 * φ ^ 7 + (-41369 : ℝ) * δ ^ 7 * φ ^ 6
      + (-108357 : ℝ) * δ ^ 7 * φ ^ 5 + (-193217 : ℝ) * δ ^ 7 * φ ^ 4
      + (-226251 : ℝ) * δ ^ 7 * φ ^ 3 + (-162241 : ℝ) * δ ^ 7 * φ ^ 2 + (-63515 : ℝ) * δ ^ 7 * φ
      + (-10245 : ℝ) * δ ^ 7 + (-266 : ℝ) * δ ^ 6 * φ ^ 9 + (-1681 : ℝ) * δ ^ 6 * φ ^ 8
      + (-6338 : ℝ) * δ ^ 6 * φ ^ 7 + (-17182 : ℝ) * δ ^ 6 * φ ^ 6
      + (-36540 : ℝ) * δ ^ 6 * φ ^ 5 + (-61944 : ℝ) * δ ^ 6 * φ ^ 4
      + (-82252 : ℝ) * δ ^ 6 * φ ^ 3 + (-77060 : ℝ) * δ ^ 6 * φ ^ 2 + (-42828 : ℝ) * δ ^ 6 * φ
      + (-10245 : ℝ) * δ ^ 6 + (-290 : ℝ) * δ ^ 5 * φ ^ 9 + (-1622 : ℝ) * δ ^ 5 * φ ^ 8
      + (-5074 : ℝ) * δ ^ 5 * φ ^ 7 + (-10989 : ℝ) * δ ^ 5 * φ ^ 6
      + (-17811 : ℝ) * δ ^ 5 * φ ^ 5 + (-21452 : ℝ) * δ ^ 5 * φ ^ 4
      + (-18308 : ℝ) * δ ^ 5 * φ ^ 3 + (-10416 : ℝ) * δ ^ 5 * φ ^ 2 + (-3562 : ℝ) * δ ^ 5 * φ
      + (-566 : ℝ) * δ ^ 5 + (-27 : ℝ) * δ ^ 4 * φ ^ 10 + (-104 : ℝ) * δ ^ 4 * φ ^ 9
      + (-298 : ℝ) * δ ^ 4 * φ ^ 8 + (-636 : ℝ) * δ ^ 4 * φ ^ 7 + (-1159 : ℝ) * δ ^ 4 * φ ^ 6
      + (-1800 : ℝ) * δ ^ 4 * φ ^ 5 + (-2488 : ℝ) * δ ^ 4 * φ ^ 4 + (-2990 : ℝ) * δ ^ 4 * φ ^ 3
      + (-2987 : ℝ) * δ ^ 4 * φ ^ 2 + (-1960 : ℝ) * δ ^ 4 * φ + (-566 : ℝ) * δ ^ 4
      + (-28 : ℝ) * δ ^ 3 * φ ^ 10 + (-83 : ℝ) * δ ^ 3 * φ ^ 9 + (-159 : ℝ) * δ ^ 3 * φ ^ 8
      + (-224 : ℝ) * δ ^ 3 * φ ^ 7 + (-262 : ℝ) * δ ^ 3 * φ ^ 6 + (-253 : ℝ) * δ ^ 3 * φ ^ 5
      + (-195 : ℝ) * δ ^ 3 * φ ^ 4 + (-128 : ℝ) * δ ^ 3 * φ ^ 3 + (-66 : ℝ) * δ ^ 3 * φ ^ 2
      + (-27 : ℝ) * δ ^ 3 * φ + (-5 : ℝ) * δ ^ 3 + (-1 : ℝ) * δ ^ 2 * φ ^ 11
      + (-1 : ℝ) * δ ^ 2 * φ ^ 10 + (-3 : ℝ) * δ ^ 2 * φ ^ 9 + (-3 : ℝ) * δ ^ 2 * φ ^ 8
      + (-5 : ℝ) * δ ^ 2 * φ ^ 7 + (-5 : ℝ) * δ ^ 2 * φ ^ 6 + (-7 : ℝ) * δ ^ 2 * φ ^ 5
      + (-7 : ℝ) * δ ^ 2 * φ ^ 4 + (-9 : ℝ) * δ ^ 2 * φ ^ 3 + (-9 : ℝ) * δ ^ 2 * φ ^ 2
      + (-11 : ℝ) * δ ^ 2 * φ + (-5 : ℝ) * δ ^ 2 + (-1 : ℝ) * δ * φ ^ 11 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_11_0_false, M_11_0_true, M_11_1_false, M_11_1_true, M_11_2_false, M_11_2_true,
    M_11_3_false, M_11_3_true, M_11_4_false, M_11_4_true, M_11_5_false, M_11_5_true,
    M_11_6_false, M_11_6_true, M_11_7_false, M_11_7_true, M_11_8_false, M_11_8_true,
    M_11_9_false, M_11_9_true, M_11_10_false, M_11_10_true, M_11_11_false, M_11_11_true]
  ring

private lemma M_12_2_false (δ φ : ℝ) :
    M_rec δ φ 2 12 false =
      (306 : ℝ) * δ ^ 12 * φ ^ 5 + (2642 : ℝ) * δ ^ 12 * φ ^ 4 + (9954 : ℝ) * δ ^ 12 * φ ^ 3
      + (19910 : ℝ) * δ ^ 12 * φ ^ 2 + (19112 : ℝ) * δ ^ 12 * φ + (6862 : ℝ) * δ ^ 12
      + (-1718 : ℝ) * δ ^ 11 * φ ^ 6 + (-13838 : ℝ) * δ ^ 11 * φ ^ 5
      + (-50864 : ℝ) * δ ^ 11 * φ ^ 4 + (-100132 : ℝ) * δ ^ 11 * φ ^ 3
      + (-104838 : ℝ) * δ ^ 11 * φ ^ 2 + (-53962 : ℝ) * δ ^ 11 * φ + (-10568 : ℝ) * δ ^ 11
      + (2786 : ℝ) * δ ^ 10 * φ ^ 6 + (24760 : ℝ) * δ ^ 10 * φ ^ 5
      + (100298 : ℝ) * δ ^ 10 * φ ^ 4 + (223928 : ℝ) * δ ^ 10 * φ ^ 3
      + (272246 : ℝ) * δ ^ 10 * φ ^ 2 + (167090 : ℝ) * δ ^ 10 * φ + (40294 : ℝ) * δ ^ 10
      + (-1558 : ℝ) * δ ^ 9 * φ ^ 7 + (-12810 : ℝ) * δ ^ 9 * φ ^ 6
      + (-57426 : ℝ) * δ ^ 9 * φ ^ 5 + (-166562 : ℝ) * δ ^ 9 * φ ^ 4
      + (-310534 : ℝ) * δ ^ 9 * φ ^ 3 + (-350090 : ℝ) * δ ^ 9 * φ ^ 2
      + (-214012 : ℝ) * δ ^ 9 * φ + (-53888 : ℝ) * δ ^ 9 + (2070 : ℝ) * δ ^ 8 * φ ^ 7
      + (16682 : ℝ) * δ ^ 8 * φ ^ 6 + (68530 : ℝ) * δ ^ 8 * φ ^ 5 + (178862 : ℝ) * δ ^ 8 * φ ^ 4
      + (292454 : ℝ) * δ ^ 8 * φ ^ 3 + (283550 : ℝ) * δ ^ 8 * φ ^ 2 + (147346 : ℝ) * δ ^ 8 * φ
      + (31526 : ℝ) * δ ^ 8 + (-440 : ℝ) * δ ^ 7 * φ ^ 8 + (-2982 : ℝ) * δ ^ 7 * φ ^ 7
      + (-12964 : ℝ) * δ ^ 7 * φ ^ 6 + (-40708 : ℝ) * δ ^ 7 * φ ^ 5
      + (-96338 : ℝ) * δ ^ 7 * φ ^ 4 + (-157750 : ℝ) * δ ^ 7 * φ ^ 3
      + (-159376 : ℝ) * δ ^ 7 * φ ^ 2 + (-86736 : ℝ) * δ ^ 7 * φ + (-19282 : ℝ) * δ ^ 7
      + (520 : ℝ) * δ ^ 6 * φ ^ 8 + (3062 : ℝ) * δ ^ 6 * φ ^ 7 + (10002 : ℝ) * δ ^ 6 * φ ^ 6
      + (23204 : ℝ) * δ ^ 6 * φ ^ 5 + (40628 : ℝ) * δ ^ 6 * φ ^ 4 + (53320 : ℝ) * δ ^ 6 * φ ^ 3
      + (47892 : ℝ) * δ ^ 6 * φ ^ 2 + (25596 : ℝ) * δ ^ 6 * φ + (5986 : ℝ) * δ ^ 6
      + (-50 : ℝ) * δ ^ 5 * φ ^ 9 + (-230 : ℝ) * δ ^ 5 * φ ^ 8 + (-926 : ℝ) * δ ^ 5 * φ ^ 7
      + (-2538 : ℝ) * δ ^ 5 * φ ^ 6 + (-5648 : ℝ) * δ ^ 5 * φ ^ 5 + (-9700 : ℝ) * δ ^ 5 * φ ^ 4
      + (-12378 : ℝ) * δ ^ 5 * φ ^ 3 + (-10370 : ℝ) * δ ^ 5 * φ ^ 2 + (-5092 : ℝ) * δ ^ 5 * φ
      + (-1116 : ℝ) * δ ^ 5 + (54 : ℝ) * δ ^ 4 * φ ^ 9 + (168 : ℝ) * δ ^ 4 * φ ^ 8
      + (360 : ℝ) * δ ^ 4 * φ ^ 7 + (616 : ℝ) * δ ^ 4 * φ ^ 6 + (886 : ℝ) * δ ^ 4 * φ ^ 5
      + (1128 : ℝ) * δ ^ 4 * φ ^ 4 + (1204 : ℝ) * δ ^ 4 * φ ^ 3 + (1136 : ℝ) * δ ^ 4 * φ ^ 2
      + (686 : ℝ) * δ ^ 4 * φ + (197 : ℝ) * δ ^ 4 + (-2 : ℝ) * δ ^ 3 * φ ^ 10
      + (-4 : ℝ) * δ ^ 3 * φ ^ 9 + (-20 : ℝ) * δ ^ 3 * φ ^ 8 + (-26 : ℝ) * δ ^ 3 * φ ^ 7
      + (-58 : ℝ) * δ ^ 3 * φ ^ 6 + (-66 : ℝ) * δ ^ 3 * φ ^ 5 + (-96 : ℝ) * δ ^ 3 * φ ^ 4
      + (-66 : ℝ) * δ ^ 3 * φ ^ 3 + (-62 : ℝ) * δ ^ 3 * φ ^ 2 + (-28 : ℝ) * δ ^ 3 * φ
      + (-12 : ℝ) * δ ^ 3 + (2 : ℝ) * δ ^ 2 * φ ^ 10 + (2 : ℝ) * δ ^ 2 * φ ^ 8
      + (2 : ℝ) * δ ^ 2 * φ ^ 6 + (2 : ℝ) * δ ^ 2 * φ ^ 4 + (2 : ℝ) * δ ^ 2 * φ ^ 2
      + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_10_0_false, M_10_0_true, M_10_1_false, M_10_1_true, M_10_2_false, M_10_2_true,
    M_10_3_false, M_10_3_true, M_10_4_false, M_10_4_true, M_10_5_false, M_10_5_true,
    M_10_6_false, M_10_6_true, M_10_7_false, M_10_7_true, M_10_8_false, M_10_8_true,
    M_10_9_false, M_10_9_true, M_10_10_false, M_10_10_true]
  ring

private lemma M_12_2_true (δ φ : ℝ) :
    M_rec δ φ 2 12 true =
      (306 : ℝ) * δ ^ 12 * φ ^ 5 + (2642 : ℝ) * δ ^ 12 * φ ^ 4 + (9954 : ℝ) * δ ^ 12 * φ ^ 3
      + (19910 : ℝ) * δ ^ 12 * φ ^ 2 + (19112 : ℝ) * δ ^ 12 * φ + (6862 : ℝ) * δ ^ 12
      + (1718 : ℝ) * δ ^ 11 * φ ^ 6 + (13838 : ℝ) * δ ^ 11 * φ ^ 5
      + (50864 : ℝ) * δ ^ 11 * φ ^ 4 + (100132 : ℝ) * δ ^ 11 * φ ^ 3
      + (104838 : ℝ) * δ ^ 11 * φ ^ 2 + (53962 : ℝ) * δ ^ 11 * φ + (10568 : ℝ) * δ ^ 11
      + (2786 : ℝ) * δ ^ 10 * φ ^ 6 + (24760 : ℝ) * δ ^ 10 * φ ^ 5
      + (100298 : ℝ) * δ ^ 10 * φ ^ 4 + (223928 : ℝ) * δ ^ 10 * φ ^ 3
      + (272246 : ℝ) * δ ^ 10 * φ ^ 2 + (167090 : ℝ) * δ ^ 10 * φ + (40294 : ℝ) * δ ^ 10
      + (1558 : ℝ) * δ ^ 9 * φ ^ 7 + (12810 : ℝ) * δ ^ 9 * φ ^ 6 + (57426 : ℝ) * δ ^ 9 * φ ^ 5
      + (166562 : ℝ) * δ ^ 9 * φ ^ 4 + (310534 : ℝ) * δ ^ 9 * φ ^ 3
      + (350090 : ℝ) * δ ^ 9 * φ ^ 2 + (214012 : ℝ) * δ ^ 9 * φ + (53888 : ℝ) * δ ^ 9
      + (2070 : ℝ) * δ ^ 8 * φ ^ 7 + (16682 : ℝ) * δ ^ 8 * φ ^ 6 + (68530 : ℝ) * δ ^ 8 * φ ^ 5
      + (178862 : ℝ) * δ ^ 8 * φ ^ 4 + (292454 : ℝ) * δ ^ 8 * φ ^ 3
      + (283550 : ℝ) * δ ^ 8 * φ ^ 2 + (147346 : ℝ) * δ ^ 8 * φ + (31526 : ℝ) * δ ^ 8
      + (440 : ℝ) * δ ^ 7 * φ ^ 8 + (2982 : ℝ) * δ ^ 7 * φ ^ 7 + (12964 : ℝ) * δ ^ 7 * φ ^ 6
      + (40708 : ℝ) * δ ^ 7 * φ ^ 5 + (96338 : ℝ) * δ ^ 7 * φ ^ 4 + (157750 : ℝ) * δ ^ 7 * φ ^ 3
      + (159376 : ℝ) * δ ^ 7 * φ ^ 2 + (86736 : ℝ) * δ ^ 7 * φ + (19282 : ℝ) * δ ^ 7
      + (520 : ℝ) * δ ^ 6 * φ ^ 8 + (3062 : ℝ) * δ ^ 6 * φ ^ 7 + (10002 : ℝ) * δ ^ 6 * φ ^ 6
      + (23204 : ℝ) * δ ^ 6 * φ ^ 5 + (40628 : ℝ) * δ ^ 6 * φ ^ 4 + (53320 : ℝ) * δ ^ 6 * φ ^ 3
      + (47892 : ℝ) * δ ^ 6 * φ ^ 2 + (25596 : ℝ) * δ ^ 6 * φ + (5986 : ℝ) * δ ^ 6
      + (50 : ℝ) * δ ^ 5 * φ ^ 9 + (230 : ℝ) * δ ^ 5 * φ ^ 8 + (926 : ℝ) * δ ^ 5 * φ ^ 7
      + (2538 : ℝ) * δ ^ 5 * φ ^ 6 + (5648 : ℝ) * δ ^ 5 * φ ^ 5 + (9700 : ℝ) * δ ^ 5 * φ ^ 4
      + (12378 : ℝ) * δ ^ 5 * φ ^ 3 + (10370 : ℝ) * δ ^ 5 * φ ^ 2 + (5092 : ℝ) * δ ^ 5 * φ
      + (1116 : ℝ) * δ ^ 5 + (54 : ℝ) * δ ^ 4 * φ ^ 9 + (168 : ℝ) * δ ^ 4 * φ ^ 8
      + (360 : ℝ) * δ ^ 4 * φ ^ 7 + (616 : ℝ) * δ ^ 4 * φ ^ 6 + (886 : ℝ) * δ ^ 4 * φ ^ 5
      + (1128 : ℝ) * δ ^ 4 * φ ^ 4 + (1204 : ℝ) * δ ^ 4 * φ ^ 3 + (1136 : ℝ) * δ ^ 4 * φ ^ 2
      + (686 : ℝ) * δ ^ 4 * φ + (197 : ℝ) * δ ^ 4 + (2 : ℝ) * δ ^ 3 * φ ^ 10
      + (4 : ℝ) * δ ^ 3 * φ ^ 9 + (20 : ℝ) * δ ^ 3 * φ ^ 8 + (26 : ℝ) * δ ^ 3 * φ ^ 7
      + (58 : ℝ) * δ ^ 3 * φ ^ 6 + (66 : ℝ) * δ ^ 3 * φ ^ 5 + (96 : ℝ) * δ ^ 3 * φ ^ 4
      + (66 : ℝ) * δ ^ 3 * φ ^ 3 + (62 : ℝ) * δ ^ 3 * φ ^ 2 + (28 : ℝ) * δ ^ 3 * φ
      + (12 : ℝ) * δ ^ 3 + (2 : ℝ) * δ ^ 2 * φ ^ 10 + (2 : ℝ) * δ ^ 2 * φ ^ 8
      + (2 : ℝ) * δ ^ 2 * φ ^ 6 + (2 : ℝ) * δ ^ 2 * φ ^ 4 + (2 : ℝ) * δ ^ 2 * φ ^ 2
      + (1 : ℝ) * δ ^ 2 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_10_0_false, M_10_0_true, M_10_1_false, M_10_1_true, M_10_2_false, M_10_2_true,
    M_10_3_false, M_10_3_true, M_10_4_false, M_10_4_true, M_10_5_false, M_10_5_true,
    M_10_6_false, M_10_6_true, M_10_7_false, M_10_7_true, M_10_8_false, M_10_8_true,
    M_10_9_false, M_10_9_true, M_10_10_false, M_10_10_true]
  ring

private lemma M_12_3_false (δ φ : ℝ) :
    M_rec δ φ 3 12 false =
      (-795 : ℝ) * δ ^ 12 * φ ^ 5 + (-4941 : ℝ) * δ ^ 12 * φ ^ 4 + (-13551 : ℝ) * δ ^ 12 * φ ^ 3
      + (-16047 : ℝ) * δ ^ 12 * φ ^ 2 + (-6656 : ℝ) * δ ^ 12 * φ + (1575 : ℝ) * δ ^ 11 * φ ^ 5
      + (12984 : ℝ) * δ ^ 11 * φ ^ 4 + (46401 : ℝ) * δ ^ 11 * φ ^ 3
      + (80994 : ℝ) * δ ^ 11 * φ ^ 2 + (65334 : ℝ) * δ ^ 11 * φ + (19458 : ℝ) * δ ^ 11
      + (-1464 : ℝ) * δ ^ 10 * φ ^ 6 + (-11868 : ℝ) * δ ^ 10 * φ ^ 5
      + (-50406 : ℝ) * δ ^ 10 * φ ^ 4 + (-123870 : ℝ) * δ ^ 10 * φ ^ 3
      + (-173751 : ℝ) * δ ^ 10 * φ ^ 2 + (-126771 : ℝ) * δ ^ 10 * φ + (-36966 : ℝ) * δ ^ 10
      + (2178 : ℝ) * δ ^ 9 * φ ^ 6 + (18882 : ℝ) * δ ^ 9 * φ ^ 5 + (78150 : ℝ) * δ ^ 9 * φ ^ 4
      + (180888 : ℝ) * δ ^ 9 * φ ^ 3 + (224737 : ℝ) * δ ^ 9 * φ ^ 2 + (140780 : ℝ) * δ ^ 9 * φ
      + (35065 : ℝ) * δ ^ 9 + (-531 : ℝ) * δ ^ 8 * φ ^ 7 + (-3777 : ℝ) * δ ^ 8 * φ ^ 6
      + (-18339 : ℝ) * δ ^ 8 * φ ^ 5 + (-62451 : ℝ) * δ ^ 8 * φ ^ 4
      + (-139779 : ℝ) * δ ^ 8 * φ ^ 3 + (-176157 : ℝ) * δ ^ 8 * φ ^ 2
      + (-111711 : ℝ) * δ ^ 8 * φ + (-27795 : ℝ) * δ ^ 8 + (681 : ℝ) * δ ^ 7 * φ ^ 7
      + (4374 : ℝ) * δ ^ 7 * φ ^ 6 + (15117 : ℝ) * δ ^ 7 * φ ^ 5 + (38124 : ℝ) * δ ^ 7 * φ ^ 4
      + (68604 : ℝ) * δ ^ 7 * φ ^ 3 + (79932 : ℝ) * δ ^ 7 * φ ^ 2 + (50460 : ℝ) * δ ^ 7 * φ
      + (12978 : ℝ) * δ ^ 7 + (-69 : ℝ) * δ ^ 6 * φ ^ 8 + (-363 : ℝ) * δ ^ 6 * φ ^ 7
      + (-1896 : ℝ) * δ ^ 6 * φ ^ 6 + (-5838 : ℝ) * δ ^ 6 * φ ^ 5 + (-13854 : ℝ) * δ ^ 6 * φ ^ 4
      + (-22198 : ℝ) * δ ^ 6 * φ ^ 3 + (-23134 : ℝ) * δ ^ 6 * φ ^ 2 + (-13524 : ℝ) * δ ^ 6 * φ
      + (-3208 : ℝ) * δ ^ 6 + (78 : ℝ) * δ ^ 5 * φ ^ 8 + (264 : ℝ) * δ ^ 5 * φ ^ 7
      + (657 : ℝ) * δ ^ 5 * φ ^ 6 + (1458 : ℝ) * δ ^ 5 * φ ^ 5 + (2619 : ℝ) * δ ^ 5 * φ ^ 4
      + (3870 : ℝ) * δ ^ 5 * φ ^ 3 + (3672 : ℝ) * δ ^ 5 * φ ^ 2 + (2310 : ℝ) * δ ^ 5 * φ
      + (516 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 9 + (-9 : ℝ) * δ ^ 4 * φ ^ 8
      + (-60 : ℝ) * δ ^ 4 * φ ^ 7 + (-72 : ℝ) * δ ^ 4 * φ ^ 6 + (-207 : ℝ) * δ ^ 4 * φ ^ 5
      + (-225 : ℝ) * δ ^ 4 * φ ^ 4 + (-384 : ℝ) * δ ^ 4 * φ ^ 3 + (-246 : ℝ) * δ ^ 4 * φ ^ 2
      + (-231 : ℝ) * δ ^ 4 * φ + (-48 : ℝ) * δ ^ 4 + (3 : ℝ) * δ ^ 3 * φ ^ 9
      + (9 : ℝ) * δ ^ 3 * φ ^ 7 + (15 : ℝ) * δ ^ 3 * φ ^ 5 + (19 : ℝ) * δ ^ 3 * φ ^ 3
      + (9 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_9_0_false, M_9_0_true, M_9_1_false, M_9_1_true, M_9_2_false, M_9_2_true, M_9_3_false,
    M_9_3_true, M_9_4_false, M_9_4_true, M_9_5_false, M_9_5_true, M_9_6_false, M_9_6_true,
    M_9_7_false, M_9_7_true, M_9_8_false, M_9_8_true, M_9_9_false, M_9_9_true]
  ring

private lemma M_12_3_true (δ φ : ℝ) :
    M_rec δ φ 3 12 true =
      (-795 : ℝ) * δ ^ 12 * φ ^ 5 + (-4941 : ℝ) * δ ^ 12 * φ ^ 4 + (-13551 : ℝ) * δ ^ 12 * φ ^ 3
      + (-16047 : ℝ) * δ ^ 12 * φ ^ 2 + (-6656 : ℝ) * δ ^ 12 * φ + (-1575 : ℝ) * δ ^ 11 * φ ^ 5
      + (-12984 : ℝ) * δ ^ 11 * φ ^ 4 + (-46401 : ℝ) * δ ^ 11 * φ ^ 3
      + (-80994 : ℝ) * δ ^ 11 * φ ^ 2 + (-65334 : ℝ) * δ ^ 11 * φ + (-19458 : ℝ) * δ ^ 11
      + (-1464 : ℝ) * δ ^ 10 * φ ^ 6 + (-11868 : ℝ) * δ ^ 10 * φ ^ 5
      + (-50406 : ℝ) * δ ^ 10 * φ ^ 4 + (-123870 : ℝ) * δ ^ 10 * φ ^ 3
      + (-173751 : ℝ) * δ ^ 10 * φ ^ 2 + (-126771 : ℝ) * δ ^ 10 * φ + (-36966 : ℝ) * δ ^ 10
      + (-2178 : ℝ) * δ ^ 9 * φ ^ 6 + (-18882 : ℝ) * δ ^ 9 * φ ^ 5
      + (-78150 : ℝ) * δ ^ 9 * φ ^ 4 + (-180888 : ℝ) * δ ^ 9 * φ ^ 3
      + (-224737 : ℝ) * δ ^ 9 * φ ^ 2 + (-140780 : ℝ) * δ ^ 9 * φ + (-35065 : ℝ) * δ ^ 9
      + (-531 : ℝ) * δ ^ 8 * φ ^ 7 + (-3777 : ℝ) * δ ^ 8 * φ ^ 6 + (-18339 : ℝ) * δ ^ 8 * φ ^ 5
      + (-62451 : ℝ) * δ ^ 8 * φ ^ 4 + (-139779 : ℝ) * δ ^ 8 * φ ^ 3
      + (-176157 : ℝ) * δ ^ 8 * φ ^ 2 + (-111711 : ℝ) * δ ^ 8 * φ + (-27795 : ℝ) * δ ^ 8
      + (-681 : ℝ) * δ ^ 7 * φ ^ 7 + (-4374 : ℝ) * δ ^ 7 * φ ^ 6 + (-15117 : ℝ) * δ ^ 7 * φ ^ 5
      + (-38124 : ℝ) * δ ^ 7 * φ ^ 4 + (-68604 : ℝ) * δ ^ 7 * φ ^ 3
      + (-79932 : ℝ) * δ ^ 7 * φ ^ 2 + (-50460 : ℝ) * δ ^ 7 * φ + (-12978 : ℝ) * δ ^ 7
      + (-69 : ℝ) * δ ^ 6 * φ ^ 8 + (-363 : ℝ) * δ ^ 6 * φ ^ 7 + (-1896 : ℝ) * δ ^ 6 * φ ^ 6
      + (-5838 : ℝ) * δ ^ 6 * φ ^ 5 + (-13854 : ℝ) * δ ^ 6 * φ ^ 4
      + (-22198 : ℝ) * δ ^ 6 * φ ^ 3 + (-23134 : ℝ) * δ ^ 6 * φ ^ 2 + (-13524 : ℝ) * δ ^ 6 * φ
      + (-3208 : ℝ) * δ ^ 6 + (-78 : ℝ) * δ ^ 5 * φ ^ 8 + (-264 : ℝ) * δ ^ 5 * φ ^ 7
      + (-657 : ℝ) * δ ^ 5 * φ ^ 6 + (-1458 : ℝ) * δ ^ 5 * φ ^ 5 + (-2619 : ℝ) * δ ^ 5 * φ ^ 4
      + (-3870 : ℝ) * δ ^ 5 * φ ^ 3 + (-3672 : ℝ) * δ ^ 5 * φ ^ 2 + (-2310 : ℝ) * δ ^ 5 * φ
      + (-516 : ℝ) * δ ^ 5 + (-3 : ℝ) * δ ^ 4 * φ ^ 9 + (-9 : ℝ) * δ ^ 4 * φ ^ 8
      + (-60 : ℝ) * δ ^ 4 * φ ^ 7 + (-72 : ℝ) * δ ^ 4 * φ ^ 6 + (-207 : ℝ) * δ ^ 4 * φ ^ 5
      + (-225 : ℝ) * δ ^ 4 * φ ^ 4 + (-384 : ℝ) * δ ^ 4 * φ ^ 3 + (-246 : ℝ) * δ ^ 4 * φ ^ 2
      + (-231 : ℝ) * δ ^ 4 * φ + (-48 : ℝ) * δ ^ 4 + (-3 : ℝ) * δ ^ 3 * φ ^ 9
      + (-9 : ℝ) * δ ^ 3 * φ ^ 7 + (-15 : ℝ) * δ ^ 3 * φ ^ 5 + (-19 : ℝ) * δ ^ 3 * φ ^ 3
      + (-9 : ℝ) * δ ^ 3 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_9_0_false, M_9_0_true, M_9_1_false, M_9_1_true, M_9_2_false, M_9_2_true, M_9_3_false,
    M_9_3_true, M_9_4_false, M_9_4_true, M_9_5_false, M_9_5_true, M_9_6_false, M_9_6_true,
    M_9_7_false, M_9_7_true, M_9_8_false, M_9_8_true, M_9_9_false, M_9_9_true]
  ring

private lemma M_12_4_false (δ φ : ℝ) :
    M_rec δ φ 4 12 false =
      (164 : ℝ) * δ ^ 12 * φ ^ 4 + (1912 : ℝ) * δ ^ 12 * φ ^ 3 + (7204 : ℝ) * δ ^ 12 * φ ^ 2
      + (10720 : ℝ) * δ ^ 12 * φ + (5194 : ℝ) * δ ^ 12 + (-1048 : ℝ) * δ ^ 11 * φ ^ 5
      + (-8096 : ℝ) * δ ^ 11 * φ ^ 4 + (-26988 : ℝ) * δ ^ 11 * φ ^ 3
      + (-43616 : ℝ) * δ ^ 11 * φ ^ 2 + (-35580 : ℝ) * δ ^ 11 * φ + (-11968 : ℝ) * δ ^ 11
      + (1744 : ℝ) * δ ^ 10 * φ ^ 5 + (16468 : ℝ) * δ ^ 10 * φ ^ 4
      + (60420 : ℝ) * δ ^ 10 * φ ^ 3 + (99684 : ℝ) * δ ^ 10 * φ ^ 2 + (73764 : ℝ) * δ ^ 10 * φ
      + (20192 : ℝ) * δ ^ 10 + (-548 : ℝ) * δ ^ 9 * φ ^ 6 + (-4016 : ℝ) * δ ^ 9 * φ ^ 5
      + (-20660 : ℝ) * δ ^ 9 * φ ^ 4 + (-67244 : ℝ) * δ ^ 9 * φ ^ 3
      + (-113720 : ℝ) * δ ^ 9 * φ ^ 2 + (-88980 : ℝ) * δ ^ 9 * φ + (-25152 : ℝ) * δ ^ 9
      + (764 : ℝ) * δ ^ 8 * φ ^ 6 + (5520 : ℝ) * δ ^ 8 * φ ^ 5 + (20048 : ℝ) * δ ^ 8 * φ ^ 4
      + (50720 : ℝ) * δ ^ 8 * φ ^ 3 + (74960 : ℝ) * δ ^ 8 * φ ^ 2 + (56840 : ℝ) * δ ^ 8 * φ
      + (16373 : ℝ) * δ ^ 8 + (-84 : ℝ) * δ ^ 7 * φ ^ 7 + (-488 : ℝ) * δ ^ 7 * φ ^ 6
      + (-3136 : ℝ) * δ ^ 7 * φ ^ 5 + (-10492 : ℝ) * δ ^ 7 * φ ^ 4
      + (-24196 : ℝ) * δ ^ 7 * φ ^ 3 + (-31300 : ℝ) * δ ^ 7 * φ ^ 2 + (-20632 : ℝ) * δ ^ 7 * φ
      + (-5768 : ℝ) * δ ^ 7 + (100 : ℝ) * δ ^ 6 * φ ^ 7 + (380 : ℝ) * δ ^ 6 * φ ^ 6
      + (1084 : ℝ) * δ ^ 6 * φ ^ 5 + (3056 : ℝ) * δ ^ 6 * φ ^ 4 + (5984 : ℝ) * δ ^ 6 * φ ^ 3
      + (7824 : ℝ) * δ ^ 6 * φ ^ 2 + (4280 : ℝ) * δ ^ 6 * φ + (1316 : ℝ) * δ ^ 6
      + (-4 : ℝ) * δ ^ 5 * φ ^ 8 + (-16 : ℝ) * δ ^ 5 * φ ^ 7 + (-132 : ℝ) * δ ^ 5 * φ ^ 6
      + (-148 : ℝ) * δ ^ 5 * φ ^ 5 + (-540 : ℝ) * δ ^ 5 * φ ^ 4 + (-608 : ℝ) * δ ^ 5 * φ ^ 3
      + (-1108 : ℝ) * δ ^ 5 * φ ^ 2 + (-412 : ℝ) * δ ^ 5 * φ + (-200 : ℝ) * δ ^ 5
      + (4 : ℝ) * δ ^ 4 * φ ^ 8 + (24 : ℝ) * δ ^ 4 * φ ^ 6 + (52 : ℝ) * δ ^ 4 * φ ^ 4
      + (72 : ℝ) * δ ^ 4 * φ ^ 2 + (13 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_8_0_false, M_8_0_true, M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false,
    M_8_3_true, M_8_4_false, M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true,
    M_8_7_false, M_8_7_true, M_8_8_false, M_8_8_true]
  ring

private lemma M_12_4_true (δ φ : ℝ) :
    M_rec δ φ 4 12 true =
      (164 : ℝ) * δ ^ 12 * φ ^ 4 + (1912 : ℝ) * δ ^ 12 * φ ^ 3 + (7204 : ℝ) * δ ^ 12 * φ ^ 2
      + (10720 : ℝ) * δ ^ 12 * φ + (5194 : ℝ) * δ ^ 12 + (1048 : ℝ) * δ ^ 11 * φ ^ 5
      + (8096 : ℝ) * δ ^ 11 * φ ^ 4 + (26988 : ℝ) * δ ^ 11 * φ ^ 3
      + (43616 : ℝ) * δ ^ 11 * φ ^ 2 + (35580 : ℝ) * δ ^ 11 * φ + (11968 : ℝ) * δ ^ 11
      + (1744 : ℝ) * δ ^ 10 * φ ^ 5 + (16468 : ℝ) * δ ^ 10 * φ ^ 4
      + (60420 : ℝ) * δ ^ 10 * φ ^ 3 + (99684 : ℝ) * δ ^ 10 * φ ^ 2 + (73764 : ℝ) * δ ^ 10 * φ
      + (20192 : ℝ) * δ ^ 10 + (548 : ℝ) * δ ^ 9 * φ ^ 6 + (4016 : ℝ) * δ ^ 9 * φ ^ 5
      + (20660 : ℝ) * δ ^ 9 * φ ^ 4 + (67244 : ℝ) * δ ^ 9 * φ ^ 3 + (113720 : ℝ) * δ ^ 9 * φ ^ 2
      + (88980 : ℝ) * δ ^ 9 * φ + (25152 : ℝ) * δ ^ 9 + (764 : ℝ) * δ ^ 8 * φ ^ 6
      + (5520 : ℝ) * δ ^ 8 * φ ^ 5 + (20048 : ℝ) * δ ^ 8 * φ ^ 4 + (50720 : ℝ) * δ ^ 8 * φ ^ 3
      + (74960 : ℝ) * δ ^ 8 * φ ^ 2 + (56840 : ℝ) * δ ^ 8 * φ + (16373 : ℝ) * δ ^ 8
      + (84 : ℝ) * δ ^ 7 * φ ^ 7 + (488 : ℝ) * δ ^ 7 * φ ^ 6 + (3136 : ℝ) * δ ^ 7 * φ ^ 5
      + (10492 : ℝ) * δ ^ 7 * φ ^ 4 + (24196 : ℝ) * δ ^ 7 * φ ^ 3 + (31300 : ℝ) * δ ^ 7 * φ ^ 2
      + (20632 : ℝ) * δ ^ 7 * φ + (5768 : ℝ) * δ ^ 7 + (100 : ℝ) * δ ^ 6 * φ ^ 7
      + (380 : ℝ) * δ ^ 6 * φ ^ 6 + (1084 : ℝ) * δ ^ 6 * φ ^ 5 + (3056 : ℝ) * δ ^ 6 * φ ^ 4
      + (5984 : ℝ) * δ ^ 6 * φ ^ 3 + (7824 : ℝ) * δ ^ 6 * φ ^ 2 + (4280 : ℝ) * δ ^ 6 * φ
      + (1316 : ℝ) * δ ^ 6 + (4 : ℝ) * δ ^ 5 * φ ^ 8 + (16 : ℝ) * δ ^ 5 * φ ^ 7
      + (132 : ℝ) * δ ^ 5 * φ ^ 6 + (148 : ℝ) * δ ^ 5 * φ ^ 5 + (540 : ℝ) * δ ^ 5 * φ ^ 4
      + (608 : ℝ) * δ ^ 5 * φ ^ 3 + (1108 : ℝ) * δ ^ 5 * φ ^ 2 + (412 : ℝ) * δ ^ 5 * φ
      + (200 : ℝ) * δ ^ 5 + (4 : ℝ) * δ ^ 4 * φ ^ 8 + (24 : ℝ) * δ ^ 4 * φ ^ 6
      + (52 : ℝ) * δ ^ 4 * φ ^ 4 + (72 : ℝ) * δ ^ 4 * φ ^ 2 + (13 : ℝ) * δ ^ 4 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_8_0_false, M_8_0_true, M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false,
    M_8_3_true, M_8_4_false, M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true,
    M_8_7_false, M_8_7_true, M_8_8_false, M_8_8_true]
  ring

private lemma M_12_5_false (δ φ : ℝ) :
    M_rec δ φ 5 12 false =
      (-505 : ℝ) * δ ^ 12 * φ ^ 4 + (-3160 : ℝ) * δ ^ 12 * φ ^ 3 + (-6095 : ℝ) * δ ^ 12 * φ ^ 2
      + (-3500 : ℝ) * δ ^ 12 * φ + (960 : ℝ) * δ ^ 11 * φ ^ 4 + (8875 : ℝ) * δ ^ 11 * φ ^ 3
      + (23605 : ℝ) * δ ^ 11 * φ ^ 2 + (22140 : ℝ) * δ ^ 11 * φ + (6300 : ℝ) * δ ^ 11
      + (-500 : ℝ) * δ ^ 10 * φ ^ 5 + (-3725 : ℝ) * δ ^ 10 * φ ^ 4
      + (-17385 : ℝ) * δ ^ 10 * φ ^ 3 + (-41425 : ℝ) * δ ^ 10 * φ ^ 2
      + (-41985 : ℝ) * δ ^ 10 * φ + (-15100 : ℝ) * δ ^ 10 + (760 : ℝ) * δ ^ 9 * φ ^ 5
      + (6370 : ℝ) * δ ^ 9 * φ ^ 4 + (22645 : ℝ) * δ ^ 9 * φ ^ 3 + (42845 : ℝ) * δ ^ 9 * φ ^ 2
      + (38745 : ℝ) * δ ^ 9 * φ + (13760 : ℝ) * δ ^ 9 + (-95 : ℝ) * δ ^ 8 * φ ^ 6
      + (-590 : ℝ) * δ ^ 8 * φ ^ 5 + (-4490 : ℝ) * δ ^ 8 * φ ^ 4 + (-15560 : ℝ) * δ ^ 8 * φ ^ 3
      + (-26230 : ℝ) * δ ^ 8 * φ ^ 2 + (-21690 : ℝ) * δ ^ 8 * φ + (-6420 : ℝ) * δ ^ 8
      + (120 : ℝ) * δ ^ 7 * φ ^ 6 + (525 : ℝ) * δ ^ 7 * φ ^ 5 + (1655 : ℝ) * δ ^ 7 * φ ^ 4
      + (5640 : ℝ) * δ ^ 7 * φ ^ 3 + (8390 : ℝ) * δ ^ 7 * φ ^ 2 + (7760 : ℝ) * δ ^ 7 * φ
      + (1650 : ℝ) * δ ^ 7 + (-5 : ℝ) * δ ^ 6 * φ ^ 7 + (-25 : ℝ) * δ ^ 6 * φ ^ 6
      + (-245 : ℝ) * δ ^ 6 * φ ^ 5 + (-265 : ℝ) * δ ^ 6 * φ ^ 4 + (-1190 : ℝ) * δ ^ 6 * φ ^ 3
      + (-1090 : ℝ) * δ ^ 6 * φ ^ 2 + (-1610 : ℝ) * δ ^ 6 * φ + (-190 : ℝ) * δ ^ 6
      + (5 : ℝ) * δ ^ 5 * φ ^ 7 + (50 : ℝ) * δ ^ 5 * φ ^ 5 + (135 : ℝ) * δ ^ 5 * φ ^ 3
      + (140 : ℝ) * δ ^ 5 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_12_5_true (δ φ : ℝ) :
    M_rec δ φ 5 12 true =
      (-505 : ℝ) * δ ^ 12 * φ ^ 4 + (-3160 : ℝ) * δ ^ 12 * φ ^ 3 + (-6095 : ℝ) * δ ^ 12 * φ ^ 2
      + (-3500 : ℝ) * δ ^ 12 * φ + (-960 : ℝ) * δ ^ 11 * φ ^ 4 + (-8875 : ℝ) * δ ^ 11 * φ ^ 3
      + (-23605 : ℝ) * δ ^ 11 * φ ^ 2 + (-22140 : ℝ) * δ ^ 11 * φ + (-6300 : ℝ) * δ ^ 11
      + (-500 : ℝ) * δ ^ 10 * φ ^ 5 + (-3725 : ℝ) * δ ^ 10 * φ ^ 4
      + (-17385 : ℝ) * δ ^ 10 * φ ^ 3 + (-41425 : ℝ) * δ ^ 10 * φ ^ 2
      + (-41985 : ℝ) * δ ^ 10 * φ + (-15100 : ℝ) * δ ^ 10 + (-760 : ℝ) * δ ^ 9 * φ ^ 5
      + (-6370 : ℝ) * δ ^ 9 * φ ^ 4 + (-22645 : ℝ) * δ ^ 9 * φ ^ 3
      + (-42845 : ℝ) * δ ^ 9 * φ ^ 2 + (-38745 : ℝ) * δ ^ 9 * φ + (-13760 : ℝ) * δ ^ 9
      + (-95 : ℝ) * δ ^ 8 * φ ^ 6 + (-590 : ℝ) * δ ^ 8 * φ ^ 5 + (-4490 : ℝ) * δ ^ 8 * φ ^ 4
      + (-15560 : ℝ) * δ ^ 8 * φ ^ 3 + (-26230 : ℝ) * δ ^ 8 * φ ^ 2 + (-21690 : ℝ) * δ ^ 8 * φ
      + (-6420 : ℝ) * δ ^ 8 + (-120 : ℝ) * δ ^ 7 * φ ^ 6 + (-525 : ℝ) * δ ^ 7 * φ ^ 5
      + (-1655 : ℝ) * δ ^ 7 * φ ^ 4 + (-5640 : ℝ) * δ ^ 7 * φ ^ 3 + (-8390 : ℝ) * δ ^ 7 * φ ^ 2
      + (-7760 : ℝ) * δ ^ 7 * φ + (-1650 : ℝ) * δ ^ 7 + (-5 : ℝ) * δ ^ 6 * φ ^ 7
      + (-25 : ℝ) * δ ^ 6 * φ ^ 6 + (-245 : ℝ) * δ ^ 6 * φ ^ 5 + (-265 : ℝ) * δ ^ 6 * φ ^ 4
      + (-1190 : ℝ) * δ ^ 6 * φ ^ 3 + (-1090 : ℝ) * δ ^ 6 * φ ^ 2 + (-1610 : ℝ) * δ ^ 6 * φ
      + (-190 : ℝ) * δ ^ 6 + (-5 : ℝ) * δ ^ 5 * φ ^ 7 + (-50 : ℝ) * δ ^ 5 * φ ^ 5
      + (-135 : ℝ) * δ ^ 5 * φ ^ 3 + (-140 : ℝ) * δ ^ 5 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_7_0_false, M_7_0_true, M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false,
    M_7_3_true, M_7_4_false, M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true,
    M_7_7_false, M_7_7_true]
  ring

private lemma M_12_6_false (δ φ : ℝ) :
    M_rec δ φ 6 12 false =
      (66 : ℝ) * δ ^ 12 * φ ^ 3 + (1086 : ℝ) * δ ^ 12 * φ ^ 2 + (3020 : ℝ) * δ ^ 12 * φ
      + (2016 : ℝ) * δ ^ 12 + (-396 : ℝ) * δ ^ 11 * φ ^ 4 + (-2916 : ℝ) * δ ^ 11 * φ ^ 3
      + (-7206 : ℝ) * δ ^ 11 * φ ^ 2 + (-10314 : ℝ) * δ ^ 11 * φ + (-5376 : ℝ) * δ ^ 11
      + (660 : ℝ) * δ ^ 10 * φ ^ 4 + (6582 : ℝ) * δ ^ 10 * φ ^ 3 + (15180 : ℝ) * δ ^ 10 * φ ^ 2
      + (16128 : ℝ) * δ ^ 10 * φ + (6495 : ℝ) * δ ^ 10 + (-102 : ℝ) * δ ^ 9 * φ ^ 5
      + (-654 : ℝ) * δ ^ 9 * φ ^ 4 + (-5628 : ℝ) * δ ^ 9 * φ ^ 3 + (-14684 : ℝ) * δ ^ 9 * φ ^ 2
      + (-13772 : ℝ) * δ ^ 9 * φ + (-5200 : ℝ) * δ ^ 9 + (138 : ℝ) * δ ^ 8 * φ ^ 5
      + (708 : ℝ) * δ ^ 8 * φ ^ 4 + (2304 : ℝ) * δ ^ 8 * φ ^ 3 + (7404 : ℝ) * δ ^ 8 * φ ^ 2
      + (5928 : ℝ) * δ ^ 8 * φ + (2823 : ℝ) * δ ^ 8 + (-6 : ℝ) * δ ^ 7 * φ ^ 6
      + (-36 : ℝ) * δ ^ 7 * φ ^ 5 + (-408 : ℝ) * δ ^ 7 * φ ^ 4 + (-408 : ℝ) * δ ^ 7 * φ ^ 3
      + (-2040 : ℝ) * δ ^ 7 * φ ^ 2 + (-990 : ℝ) * δ ^ 7 * φ + (-864 : ℝ) * δ ^ 7
      + (6 : ℝ) * δ ^ 6 * φ ^ 6 + (90 : ℝ) * δ ^ 6 * φ ^ 4 + (260 : ℝ) * δ ^ 6 * φ ^ 2
      + (106 : ℝ) * δ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_12_6_true (δ φ : ℝ) :
    M_rec δ φ 6 12 true =
      (66 : ℝ) * δ ^ 12 * φ ^ 3 + (1086 : ℝ) * δ ^ 12 * φ ^ 2 + (3020 : ℝ) * δ ^ 12 * φ
      + (2016 : ℝ) * δ ^ 12 + (396 : ℝ) * δ ^ 11 * φ ^ 4 + (2916 : ℝ) * δ ^ 11 * φ ^ 3
      + (7206 : ℝ) * δ ^ 11 * φ ^ 2 + (10314 : ℝ) * δ ^ 11 * φ + (5376 : ℝ) * δ ^ 11
      + (660 : ℝ) * δ ^ 10 * φ ^ 4 + (6582 : ℝ) * δ ^ 10 * φ ^ 3 + (15180 : ℝ) * δ ^ 10 * φ ^ 2
      + (16128 : ℝ) * δ ^ 10 * φ + (6495 : ℝ) * δ ^ 10 + (102 : ℝ) * δ ^ 9 * φ ^ 5
      + (654 : ℝ) * δ ^ 9 * φ ^ 4 + (5628 : ℝ) * δ ^ 9 * φ ^ 3 + (14684 : ℝ) * δ ^ 9 * φ ^ 2
      + (13772 : ℝ) * δ ^ 9 * φ + (5200 : ℝ) * δ ^ 9 + (138 : ℝ) * δ ^ 8 * φ ^ 5
      + (708 : ℝ) * δ ^ 8 * φ ^ 4 + (2304 : ℝ) * δ ^ 8 * φ ^ 3 + (7404 : ℝ) * δ ^ 8 * φ ^ 2
      + (5928 : ℝ) * δ ^ 8 * φ + (2823 : ℝ) * δ ^ 8 + (6 : ℝ) * δ ^ 7 * φ ^ 6
      + (36 : ℝ) * δ ^ 7 * φ ^ 5 + (408 : ℝ) * δ ^ 7 * φ ^ 4 + (408 : ℝ) * δ ^ 7 * φ ^ 3
      + (2040 : ℝ) * δ ^ 7 * φ ^ 2 + (990 : ℝ) * δ ^ 7 * φ + (864 : ℝ) * δ ^ 7
      + (6 : ℝ) * δ ^ 6 * φ ^ 6 + (90 : ℝ) * δ ^ 6 * φ ^ 4 + (260 : ℝ) * δ ^ 6 * φ ^ 2
      + (106 : ℝ) * δ ^ 6 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_6_0_false, M_6_0_true, M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false,
    M_6_3_true, M_6_4_false, M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma M_12_7_false (δ φ : ℝ) :
    M_rec δ φ 7 12 false =
      (-245 : ℝ) * δ ^ 12 * φ ^ 3 + (-1309 : ℝ) * δ ^ 12 * φ ^ 2 + (-994 : ℝ) * δ ^ 12 * φ
      + (455 : ℝ) * δ ^ 11 * φ ^ 3 + (3962 : ℝ) * δ ^ 11 * φ ^ 2 + (4298 : ℝ) * δ ^ 11 * φ
      + (840 : ℝ) * δ ^ 11 + (-105 : ℝ) * δ ^ 10 * φ ^ 4 + (-665 : ℝ) * δ ^ 10 * φ ^ 3
      + (-4578 : ℝ) * δ ^ 10 * φ ^ 2 + (-6776 : ℝ) * δ ^ 10 * φ + (-1890 : ℝ) * δ ^ 10
      + (154 : ℝ) * δ ^ 9 * φ ^ 4 + (938 : ℝ) * δ ^ 9 * φ ^ 3 + (2443 : ℝ) * δ ^ 9 * φ ^ 2
      + (5096 : ℝ) * δ ^ 9 * φ + (1379 : ℝ) * δ ^ 9 + (-7 : ℝ) * δ ^ 8 * φ ^ 5
      + (-49 : ℝ) * δ ^ 8 * φ ^ 4 + (-630 : ℝ) * δ ^ 8 * φ ^ 3 + (-518 : ℝ) * δ ^ 8 * φ ^ 2
      + (-1932 : ℝ) * δ ^ 8 * φ + (-329 : ℝ) * δ ^ 8 + (7 : ℝ) * δ ^ 7 * φ ^ 5
      + (147 : ℝ) * δ ^ 7 * φ ^ 3 + (308 : ℝ) * δ ^ 7 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_12_7_true (δ φ : ℝ) :
    M_rec δ φ 7 12 true =
      (-245 : ℝ) * δ ^ 12 * φ ^ 3 + (-1309 : ℝ) * δ ^ 12 * φ ^ 2 + (-994 : ℝ) * δ ^ 12 * φ
      + (-455 : ℝ) * δ ^ 11 * φ ^ 3 + (-3962 : ℝ) * δ ^ 11 * φ ^ 2 + (-4298 : ℝ) * δ ^ 11 * φ
      + (-840 : ℝ) * δ ^ 11 + (-105 : ℝ) * δ ^ 10 * φ ^ 4 + (-665 : ℝ) * δ ^ 10 * φ ^ 3
      + (-4578 : ℝ) * δ ^ 10 * φ ^ 2 + (-6776 : ℝ) * δ ^ 10 * φ + (-1890 : ℝ) * δ ^ 10
      + (-154 : ℝ) * δ ^ 9 * φ ^ 4 + (-938 : ℝ) * δ ^ 9 * φ ^ 3 + (-2443 : ℝ) * δ ^ 9 * φ ^ 2
      + (-5096 : ℝ) * δ ^ 9 * φ + (-1379 : ℝ) * δ ^ 9 + (-7 : ℝ) * δ ^ 8 * φ ^ 5
      + (-49 : ℝ) * δ ^ 8 * φ ^ 4 + (-630 : ℝ) * δ ^ 8 * φ ^ 3 + (-518 : ℝ) * δ ^ 8 * φ ^ 2
      + (-1932 : ℝ) * δ ^ 8 * φ + (-329 : ℝ) * δ ^ 8 + (-7 : ℝ) * δ ^ 7 * φ ^ 5
      + (-147 : ℝ) * δ ^ 7 * φ ^ 3 + (-308 : ℝ) * δ ^ 7 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_5_0_false, M_5_0_true, M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false,
    M_5_3_true, M_5_4_false, M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma M_12_8_false (δ φ : ℝ) :
    M_rec δ φ 8 12 false =
      (24 : ℝ) * δ ^ 12 * φ ^ 2 + (448 : ℝ) * δ ^ 12 * φ + (438 : ℝ) * δ ^ 12
      + (-104 : ℝ) * δ ^ 11 * φ ^ 3 + (-496 : ℝ) * δ ^ 11 * φ ^ 2 + (-1160 : ℝ) * δ ^ 11 * φ
      + (-1152 : ℝ) * δ ^ 11 + (168 : ℝ) * δ ^ 10 * φ ^ 3 + (1112 : ℝ) * δ ^ 10 * φ ^ 2
      + (1016 : ℝ) * δ ^ 10 * φ + (1136 : ℝ) * δ ^ 10 + (-8 : ℝ) * δ ^ 9 * φ ^ 4
      + (-64 : ℝ) * δ ^ 9 * φ ^ 3 + (-864 : ℝ) * δ ^ 9 * φ ^ 2 + (-304 : ℝ) * δ ^ 9 * φ
      + (-520 : ℝ) * δ ^ 9 + (8 : ℝ) * δ ^ 8 * φ ^ 4 + (224 : ℝ) * δ ^ 8 * φ ^ 2
      + (98 : ℝ) * δ ^ 8 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_12_8_true (δ φ : ℝ) :
    M_rec δ φ 8 12 true =
      (24 : ℝ) * δ ^ 12 * φ ^ 2 + (448 : ℝ) * δ ^ 12 * φ + (438 : ℝ) * δ ^ 12
      + (104 : ℝ) * δ ^ 11 * φ ^ 3 + (496 : ℝ) * δ ^ 11 * φ ^ 2 + (1160 : ℝ) * δ ^ 11 * φ
      + (1152 : ℝ) * δ ^ 11 + (168 : ℝ) * δ ^ 10 * φ ^ 3 + (1112 : ℝ) * δ ^ 10 * φ ^ 2
      + (1016 : ℝ) * δ ^ 10 * φ + (1136 : ℝ) * δ ^ 10 + (8 : ℝ) * δ ^ 9 * φ ^ 4
      + (64 : ℝ) * δ ^ 9 * φ ^ 3 + (864 : ℝ) * δ ^ 9 * φ ^ 2 + (304 : ℝ) * δ ^ 9 * φ
      + (520 : ℝ) * δ ^ 9 + (8 : ℝ) * δ ^ 8 * φ ^ 4 + (224 : ℝ) * δ ^ 8 * φ ^ 2
      + (98 : ℝ) * δ ^ 8 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_4_0_false, M_4_0_true, M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false,
    M_4_3_true, M_4_4_false, M_4_4_true]
  ring

private lemma M_12_9_false (δ φ : ℝ) :
    M_rec δ φ 9 12 false =
      (-99 : ℝ) * δ ^ 12 * φ ^ 2 + (-174 : ℝ) * δ ^ 12 * φ + (180 : ℝ) * δ ^ 11 * φ ^ 2
      + (513 : ℝ) * δ ^ 11 * φ + (9 : ℝ) * δ ^ 11 + (-9 : ℝ) * δ ^ 10 * φ ^ 3
      + (-81 : ℝ) * δ ^ 10 * φ ^ 2 + (-495 : ℝ) * δ ^ 10 * φ + (-9 : ℝ) * δ ^ 10
      + (9 : ℝ) * δ ^ 9 * φ ^ 3 + (156 : ℝ) * δ ^ 9 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_12_9_true (δ φ : ℝ) :
    M_rec δ φ 9 12 true =
      (-99 : ℝ) * δ ^ 12 * φ ^ 2 + (-174 : ℝ) * δ ^ 12 * φ + (-180 : ℝ) * δ ^ 11 * φ ^ 2
      + (-513 : ℝ) * δ ^ 11 * φ + (-9 : ℝ) * δ ^ 11 + (-9 : ℝ) * δ ^ 10 * φ ^ 3
      + (-81 : ℝ) * δ ^ 10 * φ ^ 2 + (-495 : ℝ) * δ ^ 10 * φ + (-9 : ℝ) * δ ^ 10
      + (-9 : ℝ) * δ ^ 9 * φ ^ 3 + (-156 : ℝ) * δ ^ 9 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_3_0_false, M_3_0_true, M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false,
    M_3_3_true]
  ring

private lemma M_12_10_false (δ φ : ℝ) :
    M_rec δ φ 10 12 false =
      (10 : ℝ) * δ ^ 12 * φ + (55 : ℝ) * δ ^ 12 + (-10 : ℝ) * δ ^ 11 * φ ^ 2
      + (-10 : ℝ) * δ ^ 11 * φ + (-100 : ℝ) * δ ^ 11 + (10 : ℝ) * δ ^ 10 * φ ^ 2
      + (45 : ℝ) * δ ^ 10 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_12_10_true (δ φ : ℝ) :
    M_rec δ φ 10 12 true =
      (10 : ℝ) * δ ^ 12 * φ + (55 : ℝ) * δ ^ 12 + (10 : ℝ) * δ ^ 11 * φ ^ 2
      + (10 : ℝ) * δ ^ 11 * φ + (100 : ℝ) * δ ^ 11 + (10 : ℝ) * δ ^ 10 * φ ^ 2
      + (45 : ℝ) * δ ^ 10 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_2_0_false, M_2_0_true, M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma M_12_11_false (δ φ : ℝ) :
    M_rec δ φ 11 12 false =
      (-11 : ℝ) * δ ^ 12 * φ + (11 : ℝ) * δ ^ 11 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_12_11_true (δ φ : ℝ) :
    M_rec δ φ 11 12 true =
      (-11 : ℝ) * δ ^ 12 * φ + (-11 : ℝ) * δ ^ 11 * φ := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_1_0_false, M_1_0_true, M_1_1_false, M_1_1_true]
  ring

private lemma M_12_12_false (δ φ : ℝ) :
    M_rec δ φ 12 12 false =
      (1 : ℝ) * δ ^ 12 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma M_12_12_true (δ φ : ℝ) :
    M_rec δ φ 12 12 true =
      (1 : ℝ) * δ ^ 12 := by
  rw [M_rec]; norm_num [zSign, Finset.sum_range_succ, Nat.choose]
  rw [M_0_0_false, M_0_0_true]
  ring

private lemma r_val_1 (δ φ : ℝ) : r_rec δ φ 1 =
      (1 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_1_1_false, M_1_1_true]
  ring

private lemma r_val_2 (δ φ : ℝ) : r_rec δ φ 2 =
      (-1 : ℝ) * δ ^ 2 * φ + (-1 / 2 : ℝ) * δ ^ 2 + (1 / 2 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_2_1_false, M_2_1_true, M_2_2_false, M_2_2_true]
  ring

private lemma r_val_3 (δ φ : ℝ) : r_rec δ φ 3 =
      (1 : ℝ) * δ ^ 2 * φ ^ 2 + (2 : ℝ) * δ ^ 2 * φ + (1 : ℝ) * δ ^ 2 + (1 / 3 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_3_1_false, M_3_1_true, M_3_2_false, M_3_2_true, M_3_3_false, M_3_3_true]
  ring

private lemma r_val_4 (δ φ : ℝ) : r_rec δ φ 4 =
      (-3 : ℝ) * δ ^ 4 * φ ^ 2 + (-4 : ℝ) * δ ^ 4 * φ + (-7 / 4 : ℝ) * δ ^ 4
      + (-1 : ℝ) * δ ^ 2 * φ ^ 3 + (-2 : ℝ) * δ ^ 2 * φ ^ 2 + (-3 : ℝ) * δ ^ 2 * φ
      + (-3 / 2 : ℝ) * δ ^ 2 + (1 / 4 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_4_1_false, M_4_1_true, M_4_2_false, M_4_2_true, M_4_3_false, M_4_3_true, M_4_4_false,
    M_4_4_true]
  ring

private lemma r_val_5 (δ φ : ℝ) : r_rec δ φ 5 =
      (6 : ℝ) * δ ^ 4 * φ ^ 3 + (20 : ℝ) * δ ^ 4 * φ ^ 2 + (22 : ℝ) * δ ^ 4 * φ
      + (8 : ℝ) * δ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 4 + (2 : ℝ) * δ ^ 2 * φ ^ 3
      + (3 : ℝ) * δ ^ 2 * φ ^ 2 + (4 : ℝ) * δ ^ 2 * φ + (2 : ℝ) * δ ^ 2 + (1 / 5 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_5_1_false, M_5_1_true, M_5_2_false, M_5_2_true, M_5_3_false, M_5_3_true, M_5_4_false,
    M_5_4_true, M_5_5_false, M_5_5_true]
  ring

private lemma r_val_6 (δ φ : ℝ) : r_rec δ φ 6 =
      (-11 : ℝ) * δ ^ 6 * φ ^ 3 + (-28 : ℝ) * δ ^ 6 * φ ^ 2 + (-85 / 3 : ℝ) * δ ^ 6 * φ
      + (-29 / 3 : ℝ) * δ ^ 6 + (-9 : ℝ) * δ ^ 4 * φ ^ 4 + (-36 : ℝ) * δ ^ 4 * φ ^ 3
      + (-74 : ℝ) * δ ^ 4 * φ ^ 2 + (-68 : ℝ) * δ ^ 4 * φ + (-23 : ℝ) * δ ^ 4
      + (-1 : ℝ) * δ ^ 2 * φ ^ 5 + (-2 : ℝ) * δ ^ 2 * φ ^ 4 + (-3 : ℝ) * δ ^ 2 * φ ^ 3
      + (-4 : ℝ) * δ ^ 2 * φ ^ 2 + (-5 : ℝ) * δ ^ 2 * φ + (-5 / 2 : ℝ) * δ ^ 2 + (1 / 6 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_6_1_false, M_6_1_true, M_6_2_false, M_6_2_true, M_6_3_false, M_6_3_true, M_6_4_false,
    M_6_4_true, M_6_5_false, M_6_5_true, M_6_6_false, M_6_6_true]
  ring

private lemma r_val_7 (δ φ : ℝ) : r_rec δ φ 7 =
      (31 : ℝ) * δ ^ 6 * φ ^ 4 + (154 : ℝ) * δ ^ 6 * φ ^ 3 + (290 : ℝ) * δ ^ 6 * φ ^ 2
      + (242 : ℝ) * δ ^ 6 * φ + (75 : ℝ) * δ ^ 6 + (12 : ℝ) * δ ^ 4 * φ ^ 5
      + (52 : ℝ) * δ ^ 4 * φ ^ 4 + (126 : ℝ) * δ ^ 4 * φ ^ 3 + (196 : ℝ) * δ ^ 4 * φ ^ 2
      + (162 : ℝ) * δ ^ 4 * φ + (52 : ℝ) * δ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 6
      + (2 : ℝ) * δ ^ 2 * φ ^ 5 + (3 : ℝ) * δ ^ 2 * φ ^ 4 + (4 : ℝ) * δ ^ 2 * φ ^ 3
      + (5 : ℝ) * δ ^ 2 * φ ^ 2 + (6 : ℝ) * δ ^ 2 * φ + (3 : ℝ) * δ ^ 2 + (1 / 7 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_7_1_false, M_7_1_true, M_7_2_false, M_7_2_true, M_7_3_false, M_7_3_true, M_7_4_false,
    M_7_4_true, M_7_5_false, M_7_5_true, M_7_6_false, M_7_6_true, M_7_7_false, M_7_7_true]
  ring

private lemma r_val_8 (δ φ : ℝ) : r_rec δ φ 8 =
      (-41 : ℝ) * δ ^ 8 * φ ^ 4 + (-168 : ℝ) * δ ^ 8 * φ ^ 3 + (-294 : ℝ) * δ ^ 8 * φ ^ 2
      + (-232 : ℝ) * δ ^ 8 * φ + (-555 / 8 : ℝ) * δ ^ 8 + (-60 : ℝ) * δ ^ 6 * φ ^ 5
      + (-360 : ℝ) * δ ^ 6 * φ ^ 4 + (-1048 : ℝ) * δ ^ 6 * φ ^ 3 + (-1560 : ℝ) * δ ^ 6 * φ ^ 2
      + (-1148 : ℝ) * δ ^ 6 * φ + (-657 / 2 : ℝ) * δ ^ 6 + (-15 : ℝ) * δ ^ 4 * φ ^ 6
      + (-68 : ℝ) * δ ^ 4 * φ ^ 5 + (-178 : ℝ) * δ ^ 4 * φ ^ 4 + (-324 : ℝ) * δ ^ 4 * φ ^ 3
      + (-429 : ℝ) * δ ^ 4 * φ ^ 2 + (-328 : ℝ) * δ ^ 4 * φ + (-407 / 4 : ℝ) * δ ^ 4
      + (-1 : ℝ) * δ ^ 2 * φ ^ 7 + (-2 : ℝ) * δ ^ 2 * φ ^ 6 + (-3 : ℝ) * δ ^ 2 * φ ^ 5
      + (-4 : ℝ) * δ ^ 2 * φ ^ 4 + (-5 : ℝ) * δ ^ 2 * φ ^ 3 + (-6 : ℝ) * δ ^ 2 * φ ^ 2
      + (-7 : ℝ) * δ ^ 2 * φ + (-7 / 2 : ℝ) * δ ^ 2 + (1 / 8 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_8_1_false, M_8_1_true, M_8_2_false, M_8_2_true, M_8_3_false, M_8_3_true, M_8_4_false,
    M_8_4_true, M_8_5_false, M_8_5_true, M_8_6_false, M_8_6_true, M_8_7_false, M_8_7_true,
    M_8_8_false, M_8_8_true]
  ring

private lemma r_val_9 (δ φ : ℝ) : r_rec δ φ 9 =
      (148 : ℝ) * δ ^ 8 * φ ^ 5 + (992 : ℝ) * δ ^ 8 * φ ^ 4 + (2796 : ℝ) * δ ^ 8 * φ ^ 3
      + (3968 : ℝ) * δ ^ 8 * φ ^ 2 + (2776 : ℝ) * δ ^ 8 * φ + (760 : ℝ) * δ ^ 8
      + (98 : ℝ) * δ ^ 6 * φ ^ 6 + (644 : ℝ) * δ ^ 6 * φ ^ 5 + (2211 : ℝ) * δ ^ 6 * φ ^ 4
      + (4718 : ℝ) * δ ^ 6 * φ ^ 3 + (17884 / 3 : ℝ) * δ ^ 6 * φ ^ 2
      + (11942 / 3 : ℝ) * δ ^ 6 * φ + (3217 / 3 : ℝ) * δ ^ 6 + (18 : ℝ) * δ ^ 4 * φ ^ 7
      + (84 : ℝ) * δ ^ 4 * φ ^ 6 + (230 : ℝ) * δ ^ 4 * φ ^ 5 + (452 : ℝ) * δ ^ 4 * φ ^ 4
      + (696 : ℝ) * δ ^ 4 * φ ^ 3 + (824 : ℝ) * δ ^ 4 * φ ^ 2 + (596 : ℝ) * δ ^ 4 * φ
      + (180 : ℝ) * δ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 8 + (2 : ℝ) * δ ^ 2 * φ ^ 7
      + (3 : ℝ) * δ ^ 2 * φ ^ 6 + (4 : ℝ) * δ ^ 2 * φ ^ 5 + (5 : ℝ) * δ ^ 2 * φ ^ 4
      + (6 : ℝ) * δ ^ 2 * φ ^ 3 + (7 : ℝ) * δ ^ 2 * φ ^ 2 + (8 : ℝ) * δ ^ 2 * φ
      + (4 : ℝ) * δ ^ 2 + (1 / 9 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_9_1_false, M_9_1_true, M_9_2_false, M_9_2_true, M_9_3_false, M_9_3_true, M_9_4_false,
    M_9_4_true, M_9_5_false, M_9_5_true, M_9_6_false, M_9_6_true, M_9_7_false, M_9_7_true,
    M_9_8_false, M_9_8_true, M_9_9_false, M_9_9_true]
  ring

private lemma r_val_10 (δ φ : ℝ) : r_rec δ φ 10 =
      (-153 : ℝ) * δ ^ 10 * φ ^ 5 + (-884 : ℝ) * δ ^ 10 * φ ^ 4 + (-2335 : ℝ) * δ ^ 10 * φ ^ 3
      + (-3160 : ℝ) * δ ^ 10 * φ ^ 2 + (-10686 / 5 : ℝ) * δ ^ 10 * φ + (-2843 / 5 : ℝ) * δ ^ 10
      + (-348 : ℝ) * δ ^ 8 * φ ^ 6 + (-2772 : ℝ) * δ ^ 8 * φ ^ 5 + (-10710 : ℝ) * δ ^ 8 * φ ^ 4
      + (-23144 : ℝ) * δ ^ 8 * φ ^ 3 + (-28160 : ℝ) * δ ^ 8 * φ ^ 2 + (-17828 : ℝ) * δ ^ 8 * φ
      + (-4554 : ℝ) * δ ^ 8 + (-145 : ℝ) * δ ^ 6 * φ ^ 7 + (-1006 : ℝ) * δ ^ 6 * φ ^ 6
      + (-3761 : ℝ) * δ ^ 6 * φ ^ 5 + (-9378 : ℝ) * δ ^ 6 * φ ^ 4 + (-16340 : ℝ) * δ ^ 6 * φ ^ 3
      + (-18238 : ℝ) * δ ^ 6 * φ ^ 2 + (-11296 : ℝ) * δ ^ 6 * φ + (-2899 : ℝ) * δ ^ 6
      + (-21 : ℝ) * δ ^ 4 * φ ^ 8 + (-100 : ℝ) * δ ^ 4 * φ ^ 7 + (-282 : ℝ) * δ ^ 4 * φ ^ 6
      + (-580 : ℝ) * δ ^ 4 * φ ^ 5 + (-963 : ℝ) * δ ^ 4 * φ ^ 4 + (-1320 : ℝ) * δ ^ 4 * φ ^ 3
      + (-1444 : ℝ) * δ ^ 4 * φ ^ 2 + (-1000 : ℝ) * δ ^ 4 * φ + (-296 : ℝ) * δ ^ 4
      + (-1 : ℝ) * δ ^ 2 * φ ^ 9 + (-2 : ℝ) * δ ^ 2 * φ ^ 8 + (-3 : ℝ) * δ ^ 2 * φ ^ 7
      + (-4 : ℝ) * δ ^ 2 * φ ^ 6 + (-5 : ℝ) * δ ^ 2 * φ ^ 5 + (-6 : ℝ) * δ ^ 2 * φ ^ 4
      + (-7 : ℝ) * δ ^ 2 * φ ^ 3 + (-8 : ℝ) * δ ^ 2 * φ ^ 2 + (-9 : ℝ) * δ ^ 2 * φ
      + (-9 / 2 : ℝ) * δ ^ 2 + (1 / 10 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_10_1_false, M_10_1_true, M_10_2_false, M_10_2_true, M_10_3_false, M_10_3_true,
    M_10_4_false, M_10_4_true, M_10_5_false, M_10_5_true, M_10_6_false, M_10_6_true,
    M_10_7_false, M_10_7_true, M_10_8_false, M_10_8_true, M_10_9_false, M_10_9_true,
    M_10_10_false, M_10_10_true]
  ring

private lemma r_val_11 (δ φ : ℝ) : r_rec δ φ 11 =
      (673 : ℝ) * δ ^ 10 * φ ^ 6 + (5682 : ℝ) * δ ^ 10 * φ ^ 5 + (21636 : ℝ) * δ ^ 10 * φ ^ 4
      + (45462 : ℝ) * δ ^ 10 * φ ^ 3 + (53525 : ℝ) * δ ^ 10 * φ ^ 2 + (32836 : ℝ) * δ ^ 10 * φ
      + (8146 : ℝ) * δ ^ 10 + (668 : ℝ) * δ ^ 8 * φ ^ 7 + (5828 : ℝ) * δ ^ 8 * φ ^ 6
      + (26000 : ℝ) * δ ^ 8 * φ ^ 5 + (72940 : ℝ) * δ ^ 8 * φ ^ 4 + (130208 : ℝ) * δ ^ 8 * φ ^ 3
      + (140188 : ℝ) * δ ^ 8 * φ ^ 2 + (81804 : ℝ) * δ ^ 8 * φ + (19724 : ℝ) * δ ^ 8
      + (201 : ℝ) * δ ^ 6 * φ ^ 8 + (1446 : ℝ) * δ ^ 6 * φ ^ 7 + (5696 : ℝ) * δ ^ 6 * φ ^ 6
      + (15450 : ℝ) * δ ^ 6 * φ ^ 5 + (31266 : ℝ) * δ ^ 6 * φ ^ 4 + (47026 : ℝ) * δ ^ 6 * φ ^ 3
      + (47704 : ℝ) * δ ^ 6 * φ ^ 2 + (27806 : ℝ) * δ ^ 6 * φ + (6861 : ℝ) * δ ^ 6
      + (24 : ℝ) * δ ^ 4 * φ ^ 9 + (116 : ℝ) * δ ^ 4 * φ ^ 8 + (334 : ℝ) * δ ^ 4 * φ ^ 7
      + (708 : ℝ) * δ ^ 4 * φ ^ 6 + (1230 : ℝ) * δ ^ 4 * φ ^ 5 + (1816 : ℝ) * δ ^ 4 * φ ^ 4
      + (2292 : ℝ) * δ ^ 4 * φ ^ 3 + (2360 : ℝ) * δ ^ 4 * φ ^ 2 + (1580 : ℝ) * δ ^ 4 * φ
      + (460 : ℝ) * δ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 10 + (2 : ℝ) * δ ^ 2 * φ ^ 9
      + (3 : ℝ) * δ ^ 2 * φ ^ 8 + (4 : ℝ) * δ ^ 2 * φ ^ 7 + (5 : ℝ) * δ ^ 2 * φ ^ 6
      + (6 : ℝ) * δ ^ 2 * φ ^ 5 + (7 : ℝ) * δ ^ 2 * φ ^ 4 + (8 : ℝ) * δ ^ 2 * φ ^ 3
      + (9 : ℝ) * δ ^ 2 * φ ^ 2 + (10 : ℝ) * δ ^ 2 * φ + (5 : ℝ) * δ ^ 2 + (1 / 11 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_11_1_false, M_11_1_true, M_11_2_false, M_11_2_true, M_11_3_false, M_11_3_true,
    M_11_4_false, M_11_4_true, M_11_5_false, M_11_5_true, M_11_6_false, M_11_6_true,
    M_11_7_false, M_11_7_true, M_11_8_false, M_11_8_true, M_11_9_false, M_11_9_true,
    M_11_10_false, M_11_10_true, M_11_11_false, M_11_11_true]
  ring

private lemma r_val_12 (δ φ : ℝ) : r_rec δ φ 12 =
      (-571 : ℝ) * δ ^ 12 * φ ^ 6 + (-4268 : ℝ) * δ ^ 12 * φ ^ 5 + (-15340 : ℝ) * δ ^ 12 * φ ^ 4
      + (-30932 : ℝ) * δ ^ 12 * φ ^ 3 + (-35319 : ℝ) * δ ^ 12 * φ ^ 2
      + (-63352 / 3 : ℝ) * δ ^ 12 * φ + (-30755 / 6 : ℝ) * δ ^ 12 + (-1857 : ℝ) * δ ^ 10 * φ ^ 7
      + (-18350 : ℝ) * δ ^ 10 * φ ^ 6 + (-88655 : ℝ) * δ ^ 10 * φ ^ 5
      + (-254912 : ℝ) * δ ^ 10 * φ ^ 4 + (-450635 : ℝ) * δ ^ 10 * φ ^ 3
      + (-472566 : ℝ) * δ ^ 10 * φ ^ 2 + (-267213 : ℝ) * δ ^ 10 * φ + (-62427 : ℝ) * δ ^ 10
      + (-1135 : ℝ) * δ ^ 8 * φ ^ 8 + (-10484 : ℝ) * δ ^ 8 * φ ^ 7
      + (-50746 : ℝ) * δ ^ 8 * φ ^ 6 + (-162572 : ℝ) * δ ^ 8 * φ ^ 5
      + (-366842 : ℝ) * δ ^ 8 * φ ^ 4 + (-565532 : ℝ) * δ ^ 8 * φ ^ 3
      + (-551554 : ℝ) * δ ^ 8 * φ ^ 2 + (-300612 : ℝ) * δ ^ 8 * φ + (-69028 : ℝ) * δ ^ 8
      + (-266 : ℝ) * δ ^ 6 * φ ^ 9 + (-1964 : ℝ) * δ ^ 6 * φ ^ 8 + (-8016 : ℝ) * δ ^ 6 * φ ^ 7
      + (-22916 : ℝ) * δ ^ 6 * φ ^ 6 + (-50408 : ℝ) * δ ^ 6 * φ ^ 5
      + (-87708 : ℝ) * δ ^ 6 * φ ^ 4 + (-354136 / 3 : ℝ) * δ ^ 6 * φ ^ 3
      + (-332804 / 3 : ℝ) * δ ^ 6 * φ ^ 2 + (-61526 : ℝ) * δ ^ 6 * φ + (-14692 : ℝ) * δ ^ 6
      + (-27 : ℝ) * δ ^ 4 * φ ^ 10 + (-132 : ℝ) * δ ^ 4 * φ ^ 9 + (-386 : ℝ) * δ ^ 4 * φ ^ 8
      + (-836 : ℝ) * δ ^ 4 * φ ^ 7 + (-1497 : ℝ) * δ ^ 4 * φ ^ 6 + (-2312 : ℝ) * δ ^ 4 * φ ^ 5
      + (-3140 : ℝ) * δ ^ 4 * φ ^ 4 + (-3720 : ℝ) * δ ^ 4 * φ ^ 3 + (-3655 : ℝ) * δ ^ 4 * φ ^ 2
      + (-2380 : ℝ) * δ ^ 4 * φ + (-2735 / 4 : ℝ) * δ ^ 4 + (-1 : ℝ) * δ ^ 2 * φ ^ 11
      + (-2 : ℝ) * δ ^ 2 * φ ^ 10 + (-3 : ℝ) * δ ^ 2 * φ ^ 9 + (-4 : ℝ) * δ ^ 2 * φ ^ 8
      + (-5 : ℝ) * δ ^ 2 * φ ^ 7 + (-6 : ℝ) * δ ^ 2 * φ ^ 6 + (-7 : ℝ) * δ ^ 2 * φ ^ 5
      + (-8 : ℝ) * δ ^ 2 * φ ^ 4 + (-9 : ℝ) * δ ^ 2 * φ ^ 3 + (-10 : ℝ) * δ ^ 2 * φ ^ 2
      + (-11 : ℝ) * δ ^ 2 * φ + (-11 / 2 : ℝ) * δ ^ 2 + (1 / 12 : ℝ) := by
  unfold r_rec; norm_num [Finset.sum_range_succ]
  rw [M_12_1_false, M_12_1_true, M_12_2_false, M_12_2_true, M_12_3_false, M_12_3_true,
    M_12_4_false, M_12_4_true, M_12_5_false, M_12_5_true, M_12_6_false, M_12_6_true,
    M_12_7_false, M_12_7_true, M_12_8_false, M_12_8_true, M_12_9_false, M_12_9_true,
    M_12_10_false, M_12_10_true, M_12_11_false, M_12_11_true, M_12_12_false, M_12_12_true]
  ring

private lemma e_val_0 (δ φ : ℝ) : e_rec δ φ 0 = 1 := by rw [e_rec]

private lemma e_val_1 (δ φ : ℝ) : e_rec δ φ 1 =
      (-1 : ℝ) := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, e_val_0]
  ring

private lemma e_val_2 (δ φ : ℝ) : e_rec δ φ 2 =
      (1 : ℝ) * δ ^ 2 * φ + (1 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, e_val_0, e_val_1]
  ring

private lemma e_val_3 (δ φ : ℝ) : e_rec δ φ 3 =
      (-1 : ℝ) * δ ^ 2 * φ ^ 2 + (-3 : ℝ) * δ ^ 2 * φ + (-3 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, e_val_0, e_val_1, e_val_2]
  ring

private lemma e_val_4 (δ φ : ℝ) : e_rec δ φ 4 =
      (7 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (9 / 2 : ℝ) * δ ^ 4 * φ + (15 / 8 : ℝ) * δ ^ 4
      + (1 : ℝ) * δ ^ 2 * φ ^ 3 + (3 : ℝ) * δ ^ 2 * φ ^ 2 + (5 : ℝ) * δ ^ 2 * φ
      + (5 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, e_val_0, e_val_1, e_val_2, e_val_3]
  ring

private lemma e_val_5 (δ φ : ℝ) : e_rec δ φ 5 =
      (-7 : ℝ) * δ ^ 4 * φ ^ 3 + (-26 : ℝ) * δ ^ 4 * φ ^ 2 + (-57 / 2 : ℝ) * δ ^ 4 * φ
      + (-83 / 8 : ℝ) * δ ^ 4 + (-1 : ℝ) * δ ^ 2 * φ ^ 4 + (-3 : ℝ) * δ ^ 2 * φ ^ 3
      + (-5 : ℝ) * δ ^ 2 * φ ^ 2 + (-7 : ℝ) * δ ^ 2 * φ + (-7 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4]
  ring

private lemma e_val_6 (δ φ : ℝ) : e_rec δ φ 6 =
      (85 / 6 : ℝ) * δ ^ 6 * φ ^ 3 + (135 / 4 : ℝ) * δ ^ 6 * φ ^ 2 + (773 / 24 : ℝ) * δ ^ 6 * φ
      + (169 / 16 : ℝ) * δ ^ 6 + (21 / 2 : ℝ) * δ ^ 4 * φ ^ 4 + (95 / 2 : ℝ) * δ ^ 4 * φ ^ 3
      + (207 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (97 : ℝ) * δ ^ 4 * φ + (131 / 4 : ℝ) * δ ^ 4
      + (1 : ℝ) * δ ^ 2 * φ ^ 5 + (3 : ℝ) * δ ^ 2 * φ ^ 4 + (5 : ℝ) * δ ^ 2 * φ ^ 3
      + (7 : ℝ) * δ ^ 2 * φ ^ 2 + (9 : ℝ) * δ ^ 2 * φ + (9 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, e_val_0, e_val_1, e_val_2, e_val_3,
    e_val_4, e_val_5]
  ring

private lemma e_val_7 (δ φ : ℝ) : e_rec δ φ 7 =
      (-81 / 2 : ℝ) * δ ^ 6 * φ ^ 4 + (-608 / 3 : ℝ) * δ ^ 6 * φ ^ 3
      + (-2961 / 8 : ℝ) * δ ^ 6 * φ ^ 2 + (-7235 / 24 : ℝ) * δ ^ 6 * φ
      + (-1463 / 16 : ℝ) * δ ^ 6 + (-14 : ℝ) * δ ^ 4 * φ ^ 5 + (-69 : ℝ) * δ ^ 4 * φ ^ 4
      + (-357 / 2 : ℝ) * δ ^ 4 * φ ^ 3 + (-292 : ℝ) * δ ^ 4 * φ ^ 2 + (-245 : ℝ) * δ ^ 4 * φ
      + (-315 / 4 : ℝ) * δ ^ 4 + (-1 : ℝ) * δ ^ 2 * φ ^ 6 + (-3 : ℝ) * δ ^ 2 * φ ^ 5
      + (-5 : ℝ) * δ ^ 2 * φ ^ 4 + (-7 : ℝ) * δ ^ 2 * φ ^ 3 + (-9 : ℝ) * δ ^ 2 * φ ^ 2
      + (-11 : ℝ) * δ ^ 2 * φ + (-11 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, e_val_0, e_val_1, e_val_2,
    e_val_3, e_val_4, e_val_5, e_val_6]
  ring

private lemma e_val_8 (δ φ : ℝ) : e_rec δ φ 8 =
      (1393 / 24 : ℝ) * δ ^ 8 * φ ^ 4 + (2605 / 12 : ℝ) * δ ^ 8 * φ ^ 3
      + (16939 / 48 : ℝ) * δ ^ 8 * φ ^ 2 + (12683 / 48 : ℝ) * δ ^ 8 * φ
      + (9723 / 128 : ℝ) * δ ^ 8 + (79 : ℝ) * δ ^ 6 * φ ^ 5 + (1947 / 4 : ℝ) * δ ^ 6 * φ ^ 4
      + (11375 / 8 : ℝ) * δ ^ 6 * φ ^ 3 + (16795 / 8 : ℝ) * δ ^ 6 * φ ^ 2
      + (12209 / 8 : ℝ) * δ ^ 6 * φ + (6911 / 16 : ℝ) * δ ^ 6 + (35 / 2 : ℝ) * δ ^ 4 * φ ^ 6
      + (181 / 2 : ℝ) * δ ^ 4 * φ ^ 5 + (507 / 2 : ℝ) * δ ^ 4 * φ ^ 4
      + (487 : ℝ) * δ ^ 4 * φ ^ 3 + (1335 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (1035 / 2 : ℝ) * δ ^ 4 * φ
      + (1285 / 8 : ℝ) * δ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 7 + (3 : ℝ) * δ ^ 2 * φ ^ 6
      + (5 : ℝ) * δ ^ 2 * φ ^ 5 + (7 : ℝ) * δ ^ 2 * φ ^ 4 + (9 : ℝ) * δ ^ 2 * φ ^ 3
      + (11 : ℝ) * δ ^ 2 * φ ^ 2 + (13 : ℝ) * δ ^ 2 * φ + (13 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, e_val_0, e_val_1,
    e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7]
  ring

private lemma e_val_9 (δ φ : ℝ) : e_rec δ φ 9 =
      (-1285 / 6 : ℝ) * δ ^ 8 * φ ^ 5 + (-11029 / 8 : ℝ) * δ ^ 8 * φ ^ 4
      + (-88133 / 24 : ℝ) * δ ^ 8 * φ ^ 3 + (-39849 / 8 : ℝ) * δ ^ 8 * φ ^ 2
      + (-53869 / 16 : ℝ) * δ ^ 8 * φ + (-115075 / 128 : ℝ) * δ ^ 8
      + (-389 / 3 : ℝ) * δ ^ 6 * φ ^ 6 + (-884 : ℝ) * δ ^ 6 * φ ^ 5
      + (-24625 / 8 : ℝ) * δ ^ 6 * φ ^ 4 + (-159155 / 24 : ℝ) * δ ^ 6 * φ ^ 3
      + (-200957 / 24 : ℝ) * δ ^ 6 * φ ^ 2 + (-133525 / 24 : ℝ) * δ ^ 6 * φ
      + (-23825 / 16 : ℝ) * δ ^ 6 + (-21 : ℝ) * δ ^ 4 * φ ^ 7 + (-112 : ℝ) * δ ^ 4 * φ ^ 6
      + (-657 / 2 : ℝ) * δ ^ 4 * φ ^ 5 + (-682 : ℝ) * δ ^ 4 * φ ^ 4
      + (-1090 : ℝ) * δ ^ 4 * φ ^ 3 + (-1326 : ℝ) * δ ^ 4 * φ ^ 2 + (-1939 / 2 : ℝ) * δ ^ 4 * φ
      + (-2345 / 8 : ℝ) * δ ^ 4 + (-1 : ℝ) * δ ^ 2 * φ ^ 8 + (-3 : ℝ) * δ ^ 2 * φ ^ 7
      + (-5 : ℝ) * δ ^ 2 * φ ^ 6 + (-7 : ℝ) * δ ^ 2 * φ ^ 5 + (-9 : ℝ) * δ ^ 2 * φ ^ 4
      + (-11 : ℝ) * δ ^ 2 * φ ^ 3 + (-13 : ℝ) * δ ^ 2 * φ ^ 2 + (-15 : ℝ) * δ ^ 2 * φ
      + (-15 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, r_val_9, e_val_0,
    e_val_1, e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7, e_val_8]
  ring

private lemma e_val_10 (δ φ : ℝ) : e_rec δ φ 10 =
      (28501 / 120 : ℝ) * δ ^ 10 * φ ^ 5 + (19771 / 16 : ℝ) * δ ^ 10 * φ ^ 4
      + (143027 / 48 : ℝ) * δ ^ 10 * φ ^ 3 + (120559 / 32 : ℝ) * δ ^ 10 * φ ^ 2
      + (1551703 / 640 : ℝ) * δ ^ 10 * φ + (159287 / 256 : ℝ) * δ ^ 10
      + (6125 / 12 : ℝ) * δ ^ 8 * φ ^ 6 + (3985 : ℝ) * δ ^ 8 * φ ^ 5
      + (715397 / 48 : ℝ) * δ ^ 8 * φ ^ 4 + (1493441 / 48 : ℝ) * δ ^ 8 * φ ^ 3
      + (1764059 / 48 : ℝ) * δ ^ 8 * φ ^ 2 + (272941 / 12 : ℝ) * δ ^ 8 * φ
      + (182791 / 32 : ℝ) * δ ^ 8 + (385 / 2 : ℝ) * δ ^ 6 * φ ^ 7
      + (16733 / 12 : ℝ) * δ ^ 6 * φ ^ 6 + (42551 / 8 : ℝ) * δ ^ 6 * φ ^ 5
      + (107747 / 8 : ℝ) * δ ^ 6 * φ ^ 4 + (568943 / 24 : ℝ) * δ ^ 6 * φ ^ 3
      + (637169 / 24 : ℝ) * δ ^ 6 * φ ^ 2 + (196991 / 12 : ℝ) * δ ^ 6 * φ
      + (33579 / 8 : ℝ) * δ ^ 6 + (49 / 2 : ℝ) * δ ^ 4 * φ ^ 8 + (267 / 2 : ℝ) * δ ^ 4 * φ ^ 7
      + (807 / 2 : ℝ) * δ ^ 4 * φ ^ 6 + (877 : ℝ) * δ ^ 4 * φ ^ 5
      + (3025 / 2 : ℝ) * δ ^ 4 * φ ^ 4 + (4269 / 2 : ℝ) * δ ^ 4 * φ ^ 3
      + (4767 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (1666 : ℝ) * δ ^ 4 * φ + (987 / 2 : ℝ) * δ ^ 4
      + (1 : ℝ) * δ ^ 2 * φ ^ 9 + (3 : ℝ) * δ ^ 2 * φ ^ 8 + (5 : ℝ) * δ ^ 2 * φ ^ 7
      + (7 : ℝ) * δ ^ 2 * φ ^ 6 + (9 : ℝ) * δ ^ 2 * φ ^ 5 + (11 : ℝ) * δ ^ 2 * φ ^ 4
      + (13 : ℝ) * δ ^ 2 * φ ^ 3 + (15 : ℝ) * δ ^ 2 * φ ^ 2 + (17 : ℝ) * δ ^ 2 * φ
      + (17 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, r_val_9, r_val_10,
    e_val_0, e_val_1, e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7, e_val_8, e_val_9]
  ring

private lemma e_val_11 (δ φ : ℝ) : e_rec δ φ 11 =
      (-25741 / 24 : ℝ) * δ ^ 10 * φ ^ 6 + (-1017961 / 120 : ℝ) * δ ^ 10 * φ ^ 5
      + (-179729 / 6 : ℝ) * δ ^ 10 * φ ^ 4 + (-1415975 / 24 : ℝ) * δ ^ 10 * φ ^ 3
      + (-8460143 / 128 : ℝ) * δ ^ 10 * φ ^ 2 + (-75083479 / 1920 : ℝ) * δ ^ 10 * φ
      + (-2419021 / 256 : ℝ) * δ ^ 10 + (-5933 / 6 : ℝ) * δ ^ 8 * φ ^ 7
      + (-102679 / 12 : ℝ) * δ ^ 8 * φ ^ 6 + (-449069 / 12 : ℝ) * δ ^ 8 * φ ^ 5
      + (-410883 / 4 : ℝ) * δ ^ 8 * φ ^ 4 + (-2868517 / 16 : ℝ) * δ ^ 8 * φ ^ 3
      + (-1135777 / 6 : ℝ) * δ ^ 8 * φ ^ 2 + (-435051 / 4 : ℝ) * δ ^ 8 * φ
      + (-829355 / 32 : ℝ) * δ ^ 8 + (-535 / 2 : ℝ) * δ ^ 6 * φ ^ 8
      + (-2018 : ℝ) * δ ^ 6 * φ ^ 7 + (-65137 / 8 : ℝ) * δ ^ 6 * φ ^ 6
      + (-180017 / 8 : ℝ) * δ ^ 6 * φ ^ 5 + (-369347 / 8 : ℝ) * δ ^ 6 * φ ^ 4
      + (-561367 / 8 : ℝ) * δ ^ 6 * φ ^ 3 + (-286091 / 4 : ℝ) * δ ^ 6 * φ ^ 2
      + (-166803 / 4 : ℝ) * δ ^ 6 * φ + (-82133 / 8 : ℝ) * δ ^ 6 + (-28 : ℝ) * δ ^ 4 * φ ^ 9
      + (-155 : ℝ) * δ ^ 4 * φ ^ 8 + (-957 / 2 : ℝ) * δ ^ 4 * φ ^ 7
      + (-1072 : ℝ) * δ ^ 4 * φ ^ 6 + (-1935 : ℝ) * δ ^ 4 * φ ^ 5 + (-2943 : ℝ) * δ ^ 4 * φ ^ 4
      + (-7595 / 2 : ℝ) * δ ^ 4 * φ ^ 3 + (-3976 : ℝ) * δ ^ 4 * φ ^ 2 + (-2682 : ℝ) * δ ^ 4 * φ
      + (-1563 / 2 : ℝ) * δ ^ 4 + (-1 : ℝ) * δ ^ 2 * φ ^ 10 + (-3 : ℝ) * δ ^ 2 * φ ^ 9
      + (-5 : ℝ) * δ ^ 2 * φ ^ 8 + (-7 : ℝ) * δ ^ 2 * φ ^ 7 + (-9 : ℝ) * δ ^ 2 * φ ^ 6
      + (-11 : ℝ) * δ ^ 2 * φ ^ 5 + (-13 : ℝ) * δ ^ 2 * φ ^ 4 + (-15 : ℝ) * δ ^ 2 * φ ^ 3
      + (-17 : ℝ) * δ ^ 2 * φ ^ 2 + (-19 : ℝ) * δ ^ 2 * φ + (-19 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, r_val_9, r_val_10,
    r_val_11, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7, e_val_8,
    e_val_9, e_val_10]
  ring

private lemma e_val_12 (δ φ : ℝ) : e_rec δ φ 12 =
      (698191 / 720 : ℝ) * δ ^ 12 * φ ^ 6 + (519007 / 80 : ℝ) * δ ^ 12 * φ ^ 5
      + (12108413 / 576 : ℝ) * δ ^ 12 * φ ^ 4 + (3740621 / 96 : ℝ) * δ ^ 12 * φ ^ 3
      + (481092409 / 11520 : ℝ) * δ ^ 12 * φ ^ 2 + (91569649 / 3840 : ℝ) * δ ^ 12 * φ
      + (5731003 / 1024 : ℝ) * δ ^ 12 + (24093 / 8 : ℝ) * δ ^ 10 * φ ^ 7
      + (170533 / 6 : ℝ) * δ ^ 10 * φ ^ 6 + (519959 / 4 : ℝ) * δ ^ 10 * φ ^ 5
      + (11315413 / 32 : ℝ) * δ ^ 10 * φ ^ 4 + (228838001 / 384 : ℝ) * δ ^ 10 * φ ^ 3
      + (231294191 / 384 : ℝ) * δ ^ 10 * φ ^ 2 + (127293673 / 384 : ℝ) * δ ^ 10 * φ
      + (19440047 / 256 : ℝ) * δ ^ 10 + (40595 / 24 : ℝ) * δ ^ 8 * φ ^ 8
      + (62471 / 4 : ℝ) * δ ^ 8 * φ ^ 7 + (3586271 / 48 : ℝ) * δ ^ 8 * φ ^ 6
      + (3776141 / 16 : ℝ) * δ ^ 8 * φ ^ 5 + (8394109 / 16 : ℝ) * δ ^ 8 * φ ^ 4
      + (3187723 / 4 : ℝ) * δ ^ 8 * φ ^ 3 + (6136911 / 8 : ℝ) * δ ^ 8 * φ ^ 2
      + (3309337 / 8 : ℝ) * δ ^ 8 * φ + (6028767 / 64 : ℝ) * δ ^ 8
      + (1064 / 3 : ℝ) * δ ^ 6 * φ ^ 9 + (11019 / 4 : ℝ) * δ ^ 6 * φ ^ 8
      + (92383 / 8 : ℝ) * δ ^ 6 * φ ^ 7 + (809105 / 24 : ℝ) * δ ^ 6 * φ ^ 6
      + (602713 / 8 : ℝ) * δ ^ 6 * φ ^ 5 + (1062203 / 8 : ℝ) * δ ^ 6 * φ ^ 4
      + (2165551 / 12 : ℝ) * δ ^ 6 * φ ^ 3 + (2046107 / 12 : ℝ) * δ ^ 6 * φ ^ 2
      + (378727 / 4 : ℝ) * δ ^ 6 * φ + (180649 / 8 : ℝ) * δ ^ 6 + (63 / 2 : ℝ) * δ ^ 4 * φ ^ 10
      + (353 / 2 : ℝ) * δ ^ 4 * φ ^ 9 + (1107 / 2 : ℝ) * δ ^ 4 * φ ^ 8
      + (1267 : ℝ) * δ ^ 4 * φ ^ 7 + (4715 / 2 : ℝ) * δ ^ 4 * φ ^ 6
      + (7503 / 2 : ℝ) * δ ^ 4 * φ ^ 5 + (10423 / 2 : ℝ) * δ ^ 4 * φ ^ 4
      + (6286 : ℝ) * δ ^ 4 * φ ^ 3 + (12519 / 2 : ℝ) * δ ^ 4 * φ ^ 2
      + (8205 / 2 : ℝ) * δ ^ 4 * φ + (9435 / 8 : ℝ) * δ ^ 4 + (1 : ℝ) * δ ^ 2 * φ ^ 11
      + (3 : ℝ) * δ ^ 2 * φ ^ 10 + (5 : ℝ) * δ ^ 2 * φ ^ 9 + (7 : ℝ) * δ ^ 2 * φ ^ 8
      + (9 : ℝ) * δ ^ 2 * φ ^ 7 + (11 : ℝ) * δ ^ 2 * φ ^ 6 + (13 : ℝ) * δ ^ 2 * φ ^ 5
      + (15 : ℝ) * δ ^ 2 * φ ^ 4 + (17 : ℝ) * δ ^ 2 * φ ^ 3 + (19 : ℝ) * δ ^ 2 * φ ^ 2
      + (21 : ℝ) * δ ^ 2 * φ + (21 / 2 : ℝ) * δ ^ 2 := by
  rw [e_rec]; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, r_val_9, r_val_10,
    r_val_11, r_val_12, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7,
    e_val_8, e_val_9, e_val_10, e_val_11]
  ring

private lemma v_val_1 (δ φ : ℝ) : v_rec δ φ 1 =
      (1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1]

private lemma v_val_2 (δ φ : ℝ) : v_rec δ φ 2 =
      (-2 : ℝ) * δ ^ 2 * φ + (-1 : ℝ) * δ ^ 2 + (-1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2]
  ring

private lemma v_val_3 (δ φ : ℝ) : v_rec δ φ 3 =
      (3 : ℝ) * δ ^ 2 * φ ^ 2 + (10 : ℝ) * δ ^ 2 * φ + (5 : ℝ) * δ ^ 2 + (1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3]
  ring

private lemma v_val_4 (δ φ : ℝ) : v_rec δ φ 4 =
      (-12 : ℝ) * δ ^ 4 * φ ^ 2 + (-16 : ℝ) * δ ^ 4 * φ + (-7 : ℝ) * δ ^ 4
      + (-4 : ℝ) * δ ^ 2 * φ ^ 3 + (-14 : ℝ) * δ ^ 2 * φ ^ 2 + (-28 : ℝ) * δ ^ 2 * φ
      + (-14 : ℝ) * δ ^ 2 + (-1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4]
  ring

private lemma v_val_5 (δ φ : ℝ) : v_rec δ φ 5 =
      (30 : ℝ) * δ ^ 4 * φ ^ 3 + (124 : ℝ) * δ ^ 4 * φ ^ 2 + (142 : ℝ) * δ ^ 4 * φ
      + (54 : ℝ) * δ ^ 4 + (5 : ℝ) * δ ^ 2 * φ ^ 4 + (18 : ℝ) * δ ^ 2 * φ ^ 3
      + (37 : ℝ) * δ ^ 2 * φ ^ 2 + (60 : ℝ) * δ ^ 2 * φ + (30 : ℝ) * δ ^ 2 + (1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5]
  ring

private lemma v_val_6 (δ φ : ℝ) : v_rec δ φ 6 =
      (-66 : ℝ) * δ ^ 6 * φ ^ 3 + (-168 : ℝ) * δ ^ 6 * φ ^ 2 + (-170 : ℝ) * δ ^ 6 * φ
      + (-58 : ℝ) * δ ^ 6 + (-54 : ℝ) * δ ^ 4 * φ ^ 4 + (-276 : ℝ) * δ ^ 4 * φ ^ 3
      + (-668 : ℝ) * δ ^ 4 * φ ^ 2 + (-660 : ℝ) * δ ^ 4 * φ + (-232 : ℝ) * δ ^ 4
      + (-6 : ℝ) * δ ^ 2 * φ ^ 5 + (-22 : ℝ) * δ ^ 2 * φ ^ 4 + (-46 : ℝ) * δ ^ 2 * φ ^ 3
      + (-76 : ℝ) * δ ^ 2 * φ ^ 2 + (-110 : ℝ) * δ ^ 2 * φ + (-55 : ℝ) * δ ^ 2 + (-1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6]
  ring

private lemma v_val_7 (δ φ : ℝ) : v_rec δ φ 7 =
      (217 : ℝ) * δ ^ 6 * φ ^ 4 + (1210 : ℝ) * δ ^ 6 * φ ^ 3 + (2366 : ℝ) * δ ^ 6 * φ ^ 2
      + (2034 : ℝ) * δ ^ 6 * φ + (641 : ℝ) * δ ^ 6 + (84 : ℝ) * δ ^ 4 * φ ^ 5
      + (472 : ℝ) * δ ^ 4 * φ ^ 4 + (1374 : ℝ) * δ ^ 4 * φ ^ 3 + (2484 : ℝ) * δ ^ 4 * φ ^ 2
      + (2202 : ℝ) * δ ^ 4 * φ + (734 : ℝ) * δ ^ 4 + (7 : ℝ) * δ ^ 2 * φ ^ 6
      + (26 : ℝ) * δ ^ 2 * φ ^ 5 + (55 : ℝ) * δ ^ 2 * φ ^ 4 + (92 : ℝ) * δ ^ 2 * φ ^ 3
      + (135 : ℝ) * δ ^ 2 * φ ^ 2 + (182 : ℝ) * δ ^ 2 * φ + (91 : ℝ) * δ ^ 2 + (1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7]
  ring

private lemma v_val_8 (δ φ : ℝ) : v_rec δ φ 8 =
      (-328 : ℝ) * δ ^ 8 * φ ^ 4 + (-1344 : ℝ) * δ ^ 8 * φ ^ 3 + (-2352 : ℝ) * δ ^ 8 * φ ^ 2
      + (-1856 : ℝ) * δ ^ 8 * φ + (-555 : ℝ) * δ ^ 8 + (-480 : ℝ) * δ ^ 6 * φ ^ 5
      + (-3314 : ℝ) * δ ^ 6 * φ ^ 4 + (-10672 : ℝ) * δ ^ 6 * φ ^ 3
      + (-16876 : ℝ) * δ ^ 6 * φ ^ 2 + (-12912 : ℝ) * δ ^ 6 * φ + (-3794 : ℝ) * δ ^ 6
      + (-120 : ℝ) * δ ^ 4 * φ ^ 6 + (-712 : ℝ) * δ ^ 4 * φ ^ 5 + (-2260 : ℝ) * δ ^ 4 * φ ^ 4
      + (-4848 : ℝ) * δ ^ 4 * φ ^ 3 + (-7288 : ℝ) * δ ^ 4 * φ ^ 2 + (-5960 : ℝ) * δ ^ 4 * φ
      + (-1912 : ℝ) * δ ^ 4 + (-8 : ℝ) * δ ^ 2 * φ ^ 7 + (-30 : ℝ) * δ ^ 2 * φ ^ 6
      + (-64 : ℝ) * δ ^ 2 * φ ^ 5 + (-108 : ℝ) * δ ^ 2 * φ ^ 4 + (-160 : ℝ) * δ ^ 2 * φ ^ 3
      + (-218 : ℝ) * δ ^ 2 * φ ^ 2 + (-280 : ℝ) * δ ^ 2 * φ + (-140 : ℝ) * δ ^ 2 + (-1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8]
  ring

private lemma v_val_9 (δ φ : ℝ) : v_rec δ φ 9 =
      (1332 : ℝ) * δ ^ 8 * φ ^ 5 + (9584 : ℝ) * δ ^ 8 * φ ^ 4 + (27852 : ℝ) * δ ^ 8 * φ ^ 3
      + (40416 : ℝ) * δ ^ 8 * φ ^ 2 + (28696 : ℝ) * δ ^ 8 * φ + (7950 : ℝ) * δ ^ 8
      + (882 : ℝ) * δ ^ 6 * φ ^ 6 + (6756 : ℝ) * δ ^ 6 * φ ^ 5 + (26093 : ℝ) * δ ^ 6 * φ ^ 4
      + (61518 : ℝ) * δ ^ 6 * φ ^ 3 + (83008 : ℝ) * δ ^ 6 * φ ^ 2 + (57922 : ℝ) * δ ^ 6 * φ
      + (16073 : ℝ) * δ ^ 6 + (162 : ℝ) * δ ^ 4 * φ ^ 7 + (996 : ℝ) * δ ^ 4 * φ ^ 6
      + (3326 : ℝ) * δ ^ 4 * φ ^ 5 + (7752 : ℝ) * δ ^ 4 * φ ^ 4 + (13704 : ℝ) * δ ^ 4 * φ ^ 3
      + (18136 : ℝ) * δ ^ 4 * φ ^ 2 + (13948 : ℝ) * δ ^ 4 * φ + (4346 : ℝ) * δ ^ 4
      + (9 : ℝ) * δ ^ 2 * φ ^ 8 + (34 : ℝ) * δ ^ 2 * φ ^ 7 + (73 : ℝ) * δ ^ 2 * φ ^ 6
      + (124 : ℝ) * δ ^ 2 * φ ^ 5 + (185 : ℝ) * δ ^ 2 * φ ^ 4 + (254 : ℝ) * δ ^ 2 * φ ^ 3
      + (329 : ℝ) * δ ^ 2 * φ ^ 2 + (408 : ℝ) * δ ^ 2 * φ + (204 : ℝ) * δ ^ 2 + (1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, r_val_9]
  ring

private lemma v_val_10 (δ φ : ℝ) : v_rec δ φ 10 =
      (-1530 : ℝ) * δ ^ 10 * φ ^ 5 + (-8840 : ℝ) * δ ^ 10 * φ ^ 4
      + (-23350 : ℝ) * δ ^ 10 * φ ^ 3 + (-31600 : ℝ) * δ ^ 10 * φ ^ 2
      + (-21372 : ℝ) * δ ^ 10 * φ + (-5686 : ℝ) * δ ^ 10 + (-3480 : ℝ) * δ ^ 8 * φ ^ 6
      + (-30384 : ℝ) * δ ^ 8 * φ ^ 5 + (-125612 : ℝ) * δ ^ 8 * φ ^ 4
      + (-284456 : ℝ) * δ ^ 8 * φ ^ 3 + (-357728 : ℝ) * δ ^ 8 * φ ^ 2
      + (-231960 : ℝ) * δ ^ 8 * φ + (-60330 : ℝ) * δ ^ 8 + (-1450 : ℝ) * δ ^ 6 * φ ^ 7
      + (-11824 : ℝ) * δ ^ 6 * φ ^ 6 + (-50162 : ℝ) * δ ^ 6 * φ ^ 5
      + (-139772 : ℝ) * δ ^ 6 * φ ^ 4 + (-267380 : ℝ) * δ ^ 6 * φ ^ 3
      + (-319040 : ℝ) * δ ^ 6 * φ ^ 2 + (-206708 : ℝ) * δ ^ 6 * φ + (-54714 : ℝ) * δ ^ 6
      + (-210 : ℝ) * δ ^ 4 * φ ^ 8 + (-1324 : ℝ) * δ ^ 4 * φ ^ 7 + (-4572 : ℝ) * δ ^ 4 * φ ^ 6
      + (-11196 : ℝ) * δ ^ 4 * φ ^ 5 + (-21450 : ℝ) * δ ^ 4 * φ ^ 4
      + (-33168 : ℝ) * δ ^ 4 * φ ^ 3 + (-39992 : ℝ) * δ ^ 4 * φ ^ 2 + (-29312 : ℝ) * δ ^ 4 * φ
      + (-8926 : ℝ) * δ ^ 4 + (-10 : ℝ) * δ ^ 2 * φ ^ 9 + (-38 : ℝ) * δ ^ 2 * φ ^ 8
      + (-82 : ℝ) * δ ^ 2 * φ ^ 7 + (-140 : ℝ) * δ ^ 2 * φ ^ 6 + (-210 : ℝ) * δ ^ 2 * φ ^ 5
      + (-290 : ℝ) * δ ^ 2 * φ ^ 4 + (-378 : ℝ) * δ ^ 2 * φ ^ 3 + (-472 : ℝ) * δ ^ 2 * φ ^ 2
      + (-570 : ℝ) * δ ^ 2 * φ + (-285 : ℝ) * δ ^ 2 + (-1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, r_val_9, r_val_10]
  ring

private lemma v_val_11 (δ φ : ℝ) : v_rec δ φ 11 =
      (7403 : ℝ) * δ ^ 10 * φ ^ 6 + (65562 : ℝ) * δ ^ 10 * φ ^ 5 + (255676 : ℝ) * δ ^ 10 * φ ^ 4
      + (546782 : ℝ) * δ ^ 10 * φ ^ 3 + (651975 : ℝ) * δ ^ 10 * φ ^ 2
      + (403940 : ℝ) * δ ^ 10 * φ + (100978 : ℝ) * δ ^ 10 + (7348 : ℝ) * δ ^ 8 * φ ^ 7
      + (71068 : ℝ) * δ ^ 8 * φ ^ 6 + (344104 : ℝ) * δ ^ 8 * φ ^ 5
      + (1035052 : ℝ) * δ ^ 8 * φ ^ 4 + (1948184 : ℝ) * δ ^ 8 * φ ^ 3
      + (2181396 : ℝ) * δ ^ 8 * φ ^ 2 + (1310084 : ℝ) * δ ^ 8 * φ + (322834 : ℝ) * δ ^ 8
      + (2211 : ℝ) * δ ^ 6 * φ ^ 8 + (18806 : ℝ) * δ ^ 6 * φ ^ 7 + (84540 : ℝ) * δ ^ 6 * φ ^ 6
      + (257722 : ℝ) * δ ^ 6 * φ ^ 5 + (577478 : ℝ) * δ ^ 6 * φ ^ 4
      + (948066 : ℝ) * δ ^ 6 * φ ^ 3 + (1026164 : ℝ) * δ ^ 6 * φ ^ 2 + (625534 : ℝ) * δ ^ 6 * φ
      + (159175 : ℝ) * δ ^ 6 + (264 : ℝ) * δ ^ 4 * φ ^ 9 + (1696 : ℝ) * δ ^ 4 * φ ^ 8
      + (5998 : ℝ) * δ ^ 4 * φ ^ 7 + (15180 : ℝ) * δ ^ 4 * φ ^ 6 + (30526 : ℝ) * δ ^ 4 * φ ^ 5
      + (51056 : ℝ) * δ ^ 4 * φ ^ 4 + (71580 : ℝ) * δ ^ 4 * φ ^ 3 + (80392 : ℝ) * δ ^ 4 * φ ^ 2
      + (56692 : ℝ) * δ ^ 4 * φ + (16946 : ℝ) * δ ^ 4 + (11 : ℝ) * δ ^ 2 * φ ^ 10
      + (42 : ℝ) * δ ^ 2 * φ ^ 9 + (91 : ℝ) * δ ^ 2 * φ ^ 8 + (156 : ℝ) * δ ^ 2 * φ ^ 7
      + (235 : ℝ) * δ ^ 2 * φ ^ 6 + (326 : ℝ) * δ ^ 2 * φ ^ 5 + (427 : ℝ) * δ ^ 2 * φ ^ 4
      + (536 : ℝ) * δ ^ 2 * φ ^ 3 + (651 : ℝ) * δ ^ 2 * φ ^ 2 + (770 : ℝ) * δ ^ 2 * φ
      + (385 : ℝ) * δ ^ 2 + (1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, r_val_9, r_val_10,
    r_val_11]
  ring

private lemma v_val_12 (δ φ : ℝ) : v_rec δ φ 12 =
      (-6852 : ℝ) * δ ^ 12 * φ ^ 6 + (-51216 : ℝ) * δ ^ 12 * φ ^ 5
      + (-184080 : ℝ) * δ ^ 12 * φ ^ 4 + (-371184 : ℝ) * δ ^ 12 * φ ^ 3
      + (-423828 : ℝ) * δ ^ 12 * φ ^ 2 + (-253408 : ℝ) * δ ^ 12 * φ + (-61510 : ℝ) * δ ^ 12
      + (-22284 : ℝ) * δ ^ 10 * φ ^ 7 + (-235006 : ℝ) * δ ^ 10 * φ ^ 6
      + (-1191924 : ℝ) * δ ^ 10 * φ ^ 5 + (-3552616 : ℝ) * δ ^ 10 * φ ^ 4
      + (-6454484 : ℝ) * δ ^ 10 * φ ^ 3 + (-6911542 : ℝ) * δ ^ 10 * φ ^ 2
      + (-3971692 : ℝ) * δ ^ 10 * φ + (-939708 : ℝ) * δ ^ 10 + (-13620 : ℝ) * δ ^ 8 * φ ^ 8
      + (-140504 : ℝ) * δ ^ 8 * φ ^ 7 + (-744128 : ℝ) * δ ^ 8 * φ ^ 6
      + (-2580968 : ℝ) * δ ^ 8 * φ ^ 5 + (-6239496 : ℝ) * δ ^ 8 * φ ^ 4
      + (-10166856 : ℝ) * δ ^ 8 * φ ^ 3 + (-10342112 : ℝ) * δ ^ 8 * φ ^ 2
      + (-5817272 : ℝ) * δ ^ 8 * φ + (-1368134 : ℝ) * δ ^ 8 + (-3192 : ℝ) * δ ^ 6 * φ ^ 9
      + (-27990 : ℝ) * δ ^ 6 * φ ^ 8 + (-130904 : ℝ) * δ ^ 6 * φ ^ 7
      + (-422188 : ℝ) * δ ^ 6 * φ ^ 6 + (-1032568 : ℝ) * δ ^ 6 * φ ^ 5
      + (-1973900 : ℝ) * δ ^ 6 * φ ^ 4 + (-2881896 : ℝ) * δ ^ 6 * φ ^ 3
      + (-2882124 : ℝ) * δ ^ 6 * φ ^ 2 + (-1669712 : ℝ) * δ ^ 6 * φ + (-410950 : ℝ) * δ ^ 6
      + (-324 : ℝ) * δ ^ 4 * φ ^ 10 + (-2112 : ℝ) * δ ^ 4 * φ ^ 9 + (-7604 : ℝ) * δ ^ 4 * φ ^ 8
      + (-19704 : ℝ) * δ ^ 4 * φ ^ 7 + (-40932 : ℝ) * δ ^ 4 * φ ^ 6
      + (-71800 : ℝ) * δ ^ 4 * φ ^ 5 + (-108712 : ℝ) * δ ^ 4 * φ ^ 4
      + (-141432 : ℝ) * δ ^ 4 * φ ^ 3 + (-150212 : ℝ) * δ ^ 4 * φ ^ 2
      + (-102632 : ℝ) * δ ^ 4 * φ + (-30211 : ℝ) * δ ^ 4 + (-12 : ℝ) * δ ^ 2 * φ ^ 11
      + (-46 : ℝ) * δ ^ 2 * φ ^ 10 + (-100 : ℝ) * δ ^ 2 * φ ^ 9 + (-172 : ℝ) * δ ^ 2 * φ ^ 8
      + (-260 : ℝ) * δ ^ 2 * φ ^ 7 + (-362 : ℝ) * δ ^ 2 * φ ^ 6 + (-476 : ℝ) * δ ^ 2 * φ ^ 5
      + (-600 : ℝ) * δ ^ 2 * φ ^ 4 + (-732 : ℝ) * δ ^ 2 * φ ^ 3 + (-870 : ℝ) * δ ^ 2 * φ ^ 2
      + (-1012 : ℝ) * δ ^ 2 * φ + (-506 : ℝ) * δ ^ 2 + (-1 : ℝ) := by
  unfold v_rec; norm_num [Finset.sum_range_succ]
  rw [r_val_1, r_val_2, r_val_3, r_val_4, r_val_5, r_val_6, r_val_7, r_val_8, r_val_9, r_val_10,
    r_val_11, r_val_12]
  ring

private lemma g_val_0 (δ φ : ℝ) : g_rec δ φ 0 = 0 := by unfold g_rec; simp

private lemma g_val_1 (δ φ : ℝ) : g_rec δ φ 1 =
      0 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, e_val_0, e_val_1]
  ring

private lemma g_val_2 (δ φ : ℝ) : g_rec δ φ 2 =
      (1 : ℝ) * δ ^ 2 * φ + (1 / 2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, e_val_0, e_val_1, e_val_2]
  ring

private lemma g_val_3 (δ φ : ℝ) : g_rec δ φ 3 =
      (-2 : ℝ) * δ ^ 2 * φ ^ 2 + (-4 : ℝ) * δ ^ 2 * φ + (-2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, e_val_0, e_val_1, e_val_2, e_val_3]
  ring

private lemma g_val_4 (δ φ : ℝ) : g_rec δ φ 4 =
      (13 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (19 / 2 : ℝ) * δ ^ 4 * φ + (37 / 8 : ℝ) * δ ^ 4
      + (3 : ℝ) * δ ^ 2 * φ ^ 3 + (7 : ℝ) * δ ^ 2 * φ ^ 2 + (9 : ℝ) * δ ^ 2 * φ
      + (9 / 2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, v_val_4, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4]
  ring

private lemma g_val_5 (δ φ : ℝ) : g_rec δ φ 5 =
      (-18 : ℝ) * δ ^ 4 * φ ^ 3 + (-64 : ℝ) * δ ^ 4 * φ ^ 2 + (-77 : ℝ) * δ ^ 4 * φ
      + (-123 / 4 : ℝ) * δ ^ 4 + (-4 : ℝ) * δ ^ 2 * φ ^ 4 + (-10 : ℝ) * δ ^ 2 * φ ^ 3
      + (-14 : ℝ) * δ ^ 2 * φ ^ 2 + (-16 : ℝ) * δ ^ 2 * φ + (-8 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, v_val_4, v_val_5, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4,
    e_val_5]
  ring

private lemma g_val_6 (δ φ : ℝ) : g_rec δ φ 6 =
      (197 / 6 : ℝ) * δ ^ 6 * φ ^ 3 + (399 / 4 : ℝ) * δ ^ 6 * φ ^ 2
      + (2749 / 24 : ℝ) * δ ^ 6 * φ + (673 / 16 : ℝ) * δ ^ 6 + (69 / 2 : ℝ) * δ ^ 4 * φ ^ 4
      + (299 / 2 : ℝ) * δ ^ 4 * φ ^ 3 + (647 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (320 : ℝ) * δ ^ 4 * φ
      + (116 : ℝ) * δ ^ 4 + (5 : ℝ) * δ ^ 2 * φ ^ 5 + (13 : ℝ) * δ ^ 2 * φ ^ 4
      + (19 : ℝ) * δ ^ 2 * φ ^ 3 + (23 : ℝ) * δ ^ 2 * φ ^ 2 + (25 : ℝ) * δ ^ 2 * φ
      + (25 / 2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, v_val_4, v_val_5, v_val_6, e_val_0, e_val_1, e_val_2, e_val_3,
    e_val_4, e_val_5, e_val_6]
  ring

private lemma g_val_7 (δ φ : ℝ) : g_rec δ φ 7 =
      (-110 : ℝ) * δ ^ 6 * φ ^ 4 + (-1886 / 3 : ℝ) * δ ^ 6 * φ ^ 3 + (-1366 : ℝ) * δ ^ 6 * φ ^ 2
      + (-7619 / 6 : ℝ) * δ ^ 6 * φ + (-1695 / 4 : ℝ) * δ ^ 6 + (-56 : ℝ) * δ ^ 4 * φ ^ 5
      + (-266 : ℝ) * δ ^ 4 * φ ^ 4 + (-680 : ℝ) * δ ^ 4 * φ ^ 3 + (-1102 : ℝ) * δ ^ 4 * φ ^ 2
      + (-959 : ℝ) * δ ^ 4 * φ + (-1305 / 4 : ℝ) * δ ^ 4 + (-6 : ℝ) * δ ^ 2 * φ ^ 6
      + (-16 : ℝ) * δ ^ 2 * φ ^ 5 + (-24 : ℝ) * δ ^ 2 * φ ^ 4 + (-30 : ℝ) * δ ^ 2 * φ ^ 3
      + (-34 : ℝ) * δ ^ 2 * φ ^ 2 + (-36 : ℝ) * δ ^ 2 * φ + (-18 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, v_val_4, v_val_5, v_val_6, v_val_7, e_val_0, e_val_1, e_val_2,
    e_val_3, e_val_4, e_val_5, e_val_6, e_val_7]
  ring

private lemma g_val_8 (δ φ : ℝ) : g_rec δ φ 8 =
      (1069 / 8 : ℝ) * δ ^ 8 * φ ^ 4 + (2937 / 4 : ℝ) * δ ^ 8 * φ ^ 3
      + (24447 / 16 : ℝ) * δ ^ 8 * φ ^ 2 + (21343 / 16 : ℝ) * δ ^ 8 * φ
      + (54573 / 128 : ℝ) * δ ^ 8 + (249 : ℝ) * δ ^ 6 * φ ^ 5 + (6721 / 4 : ℝ) * δ ^ 6 * φ ^ 4
      + (132955 / 24 : ℝ) * δ ^ 6 * φ ^ 3 + (74167 / 8 : ℝ) * δ ^ 6 * φ ^ 2
      + (179567 / 24 : ℝ) * δ ^ 6 * φ + (36699 / 16 : ℝ) * δ ^ 6 + (165 / 2 : ℝ) * δ ^ 4 * φ ^ 6
      + (827 / 2 : ℝ) * δ ^ 4 * φ ^ 5 + (2293 / 2 : ℝ) * δ ^ 4 * φ ^ 4
      + (2174 : ℝ) * δ ^ 4 * φ ^ 3 + (5901 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (4695 / 2 : ℝ) * δ ^ 4 * φ
      + (6113 / 8 : ℝ) * δ ^ 4 + (7 : ℝ) * δ ^ 2 * φ ^ 7 + (19 : ℝ) * δ ^ 2 * φ ^ 6
      + (29 : ℝ) * δ ^ 2 * φ ^ 5 + (37 : ℝ) * δ ^ 2 * φ ^ 4 + (43 : ℝ) * δ ^ 2 * φ ^ 3
      + (47 : ℝ) * δ ^ 2 * φ ^ 2 + (49 : ℝ) * δ ^ 2 * φ + (49 / 2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, v_val_4, v_val_5, v_val_6, v_val_7, v_val_8, e_val_0, e_val_1,
    e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7, e_val_8]
  ring

private lemma g_val_9 (δ φ : ℝ) : g_rec δ φ 9 =
      (-1567 / 3 : ℝ) * δ ^ 8 * φ ^ 5 + (-53437 / 12 : ℝ) * δ ^ 8 * φ ^ 4
      + (-186103 / 12 : ℝ) * δ ^ 8 * φ ^ 3 + (-304493 / 12 : ℝ) * δ ^ 8 * φ ^ 2
      + (-467909 / 24 : ℝ) * δ ^ 8 * φ + (-364441 / 64 : ℝ) * δ ^ 8
      + (-1402 / 3 : ℝ) * δ ^ 6 * φ ^ 6 + (-3408 : ℝ) * δ ^ 6 * φ ^ 5
      + (-52199 / 4 : ℝ) * δ ^ 6 * φ ^ 4 + (-371569 / 12 : ℝ) * δ ^ 6 * φ ^ 3
      + (-258643 / 6 : ℝ) * δ ^ 6 * φ ^ 2 + (-93551 / 3 : ℝ) * δ ^ 6 * φ
      + (-17875 / 2 : ℝ) * δ ^ 6 + (-114 : ℝ) * δ ^ 4 * φ ^ 7 + (-592 : ℝ) * δ ^ 4 * φ ^ 6
      + (-1723 : ℝ) * δ ^ 4 * φ ^ 5 + (-3536 : ℝ) * δ ^ 4 * φ ^ 4 + (-5577 : ℝ) * δ ^ 4 * φ ^ 3
      + (-6714 : ℝ) * δ ^ 4 * φ ^ 2 + (-5006 : ℝ) * δ ^ 4 * φ + (-3153 / 2 : ℝ) * δ ^ 4
      + (-8 : ℝ) * δ ^ 2 * φ ^ 8 + (-22 : ℝ) * δ ^ 2 * φ ^ 7 + (-34 : ℝ) * δ ^ 2 * φ ^ 6
      + (-44 : ℝ) * δ ^ 2 * φ ^ 5 + (-52 : ℝ) * δ ^ 2 * φ ^ 4 + (-58 : ℝ) * δ ^ 2 * φ ^ 3
      + (-62 : ℝ) * δ ^ 2 * φ ^ 2 + (-64 : ℝ) * δ ^ 2 * φ + (-32 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, v_val_4, v_val_5, v_val_6, v_val_7, v_val_8, v_val_9, e_val_0,
    e_val_1, e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7, e_val_8, e_val_9]
  ring

private lemma g_val_10 (δ φ : ℝ) : g_rec δ φ 10 =
      (53689 / 120 : ℝ) * δ ^ 10 * φ ^ 5 + (65399 / 16 : ℝ) * δ ^ 10 * φ ^ 4
      + (668303 / 48 : ℝ) * δ ^ 10 * φ ^ 3 + (696251 / 32 : ℝ) * δ ^ 10 * φ ^ 2
      + (10287427 / 640 : ℝ) * δ ^ 10 * φ + (1159075 / 256 : ℝ) * δ ^ 10
      + (16145 / 12 : ℝ) * δ ^ 8 * φ ^ 6 + (13078 : ℝ) * δ ^ 8 * φ ^ 5
      + (984559 / 16 : ℝ) * δ ^ 8 * φ ^ 4 + (7540417 / 48 : ℝ) * δ ^ 8 * φ ^ 3
      + (10362523 / 48 : ℝ) * δ ^ 8 * φ ^ 2 + (3573451 / 24 : ℝ) * δ ^ 8 * φ
      + (2587379 / 64 : ℝ) * δ ^ 8 + (1565 / 2 : ℝ) * δ ^ 6 * φ ^ 7
      + (71705 / 12 : ℝ) * δ ^ 6 * φ ^ 6 + (197123 / 8 : ℝ) * δ ^ 6 * φ ^ 5
      + (541045 / 8 : ℝ) * δ ^ 6 * φ ^ 4 + (1030967 / 8 : ℝ) * δ ^ 6 * φ ^ 3
      + (3747593 / 24 : ℝ) * δ ^ 6 * φ ^ 2 + (414689 / 4 : ℝ) * δ ^ 6 * φ
      + (225087 / 8 : ℝ) * δ ^ 6 + (301 / 2 : ℝ) * δ ^ 4 * φ ^ 8
      + (1603 / 2 : ℝ) * δ ^ 4 * φ ^ 7 + (4819 / 2 : ℝ) * δ ^ 4 * φ ^ 6
      + (5188 : ℝ) * δ ^ 4 * φ ^ 5 + (17677 / 2 : ℝ) * δ ^ 4 * φ ^ 4
      + (24611 / 2 : ℝ) * δ ^ 4 * φ ^ 3 + (27199 / 2 : ℝ) * δ ^ 4 * φ ^ 2
      + (9652 : ℝ) * δ ^ 4 * φ + (2963 : ℝ) * δ ^ 4 + (9 : ℝ) * δ ^ 2 * φ ^ 9
      + (25 : ℝ) * δ ^ 2 * φ ^ 8 + (39 : ℝ) * δ ^ 2 * φ ^ 7 + (51 : ℝ) * δ ^ 2 * φ ^ 6
      + (61 : ℝ) * δ ^ 2 * φ ^ 5 + (69 : ℝ) * δ ^ 2 * φ ^ 4 + (75 : ℝ) * δ ^ 2 * φ ^ 3
      + (79 : ℝ) * δ ^ 2 * φ ^ 2 + (81 : ℝ) * δ ^ 2 * φ + (81 / 2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, v_val_4, v_val_5, v_val_6, v_val_7, v_val_8, v_val_9, v_val_10,
    e_val_0, e_val_1, e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7, e_val_8, e_val_9,
    e_val_10]
  ring

private lemma g_val_11 (δ φ : ℝ) : g_rec δ φ 11 =
      (-3871 / 2 : ℝ) * δ ^ 10 * φ ^ 6 + (-368666 / 15 : ℝ) * δ ^ 10 * φ ^ 5
      + (-508721 / 4 : ℝ) * δ ^ 10 * φ ^ 4 + (-3911783 / 12 : ℝ) * δ ^ 10 * φ ^ 3
      + (-13934791 / 32 : ℝ) * δ ^ 10 * φ ^ 2 + (-23209403 / 80 : ℝ) * δ ^ 10 * φ
      + (-2435843 / 32 : ℝ) * δ ^ 10 + (-2830 : ℝ) * δ ^ 8 * φ ^ 7
      + (-87746 / 3 : ℝ) * δ ^ 8 * φ ^ 6 + (-155584 : ℝ) * δ ^ 8 * φ ^ 5
      + (-513629 : ℝ) * δ ^ 8 * φ ^ 4 + (-12637681 / 12 : ℝ) * δ ^ 8 * φ ^ 3
      + (-7565441 / 6 : ℝ) * δ ^ 8 * φ ^ 2 + (-19086793 / 24 : ℝ) * δ ^ 8 * φ
      + (-12993133 / 64 : ℝ) * δ ^ 8 + (-1212 : ℝ) * δ ^ 6 * φ ^ 8 + (-9546 : ℝ) * δ ^ 6 * φ ^ 7
      + (-246913 / 6 : ℝ) * δ ^ 6 * φ ^ 6 + (-121865 : ℝ) * δ ^ 6 * φ ^ 5
      + (-1071227 / 4 : ℝ) * δ ^ 6 * φ ^ 4 + (-1739355 / 4 : ℝ) * δ ^ 6 * φ ^ 3
      + (-1419559 / 3 : ℝ) * δ ^ 6 * φ ^ 2 + (-293172 : ℝ) * δ ^ 6 * φ + (-76082 : ℝ) * δ ^ 6
      + (-192 : ℝ) * δ ^ 4 * φ ^ 9 + (-1042 : ℝ) * δ ^ 4 * φ ^ 8 + (-3206 : ℝ) * δ ^ 4 * φ ^ 7
      + (-7130 : ℝ) * δ ^ 4 * φ ^ 6 + (-12735 : ℝ) * δ ^ 4 * φ ^ 5
      + (-19122 : ℝ) * δ ^ 4 * φ ^ 4 + (-24349 : ℝ) * δ ^ 4 * φ ^ 3
      + (-25236 : ℝ) * δ ^ 4 * φ ^ 2 + (-17234 : ℝ) * δ ^ 4 * φ + (-10367 / 2 : ℝ) * δ ^ 4
      + (-10 : ℝ) * δ ^ 2 * φ ^ 10 + (-28 : ℝ) * δ ^ 2 * φ ^ 9 + (-44 : ℝ) * δ ^ 2 * φ ^ 8
      + (-58 : ℝ) * δ ^ 2 * φ ^ 7 + (-70 : ℝ) * δ ^ 2 * φ ^ 6 + (-80 : ℝ) * δ ^ 2 * φ ^ 5
      + (-88 : ℝ) * δ ^ 2 * φ ^ 4 + (-94 : ℝ) * δ ^ 2 * φ ^ 3 + (-98 : ℝ) * δ ^ 2 * φ ^ 2
      + (-100 : ℝ) * δ ^ 2 * φ + (-50 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, v_val_4, v_val_5, v_val_6, v_val_7, v_val_8, v_val_9, v_val_10,
    v_val_11, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7, e_val_8,
    e_val_9, e_val_10, e_val_11]
  ring

private lemma g_val_12 (δ φ : ℝ) : g_rec δ φ 12 =
      (790397 / 720 : ℝ) * δ ^ 12 * φ ^ 6 + (1447469 / 80 : ℝ) * δ ^ 12 * φ ^ 5
      + (54650791 / 576 : ℝ) * δ ^ 12 * φ ^ 4 + (22639255 / 96 : ℝ) * δ ^ 12 * φ ^ 3
      + (3510795803 / 11520 : ℝ) * δ ^ 12 * φ ^ 2 + (755768003 / 3840 : ℝ) * δ ^ 12 * φ
      + (51469441 / 1024 : ℝ) * δ ^ 12 + (43335 / 8 : ℝ) * δ ^ 10 * φ ^ 7
      + (923135 / 12 : ℝ) * δ ^ 10 * φ ^ 6 + (5026049 / 10 : ℝ) * δ ^ 10 * φ ^ 5
      + (175164037 / 96 : ℝ) * δ ^ 10 * φ ^ 4 + (486947393 / 128 : ℝ) * δ ^ 10 * φ ^ 3
      + (1713936175 / 384 : ℝ) * δ ^ 10 * φ ^ 2 + (1744089939 / 640 : ℝ) * δ ^ 10 * φ
      + (172005683 / 256 : ℝ) * δ ^ 10 + (126025 / 24 : ℝ) * δ ^ 8 * φ ^ 8
      + (674675 / 12 : ℝ) * δ ^ 8 * φ ^ 7 + (15289013 / 48 : ℝ) * δ ^ 8 * φ ^ 6
      + (56974717 / 48 : ℝ) * δ ^ 8 * φ ^ 5 + (49201535 / 16 : ℝ) * δ ^ 8 * φ ^ 4
      + (128101469 / 24 : ℝ) * δ ^ 8 * φ ^ 3 + (137205523 / 24 : ℝ) * δ ^ 8 * φ ^ 2
      + (40132307 / 12 : ℝ) * δ ^ 8 * φ + (25915231 / 32 : ℝ) * δ ^ 8
      + (5320 / 3 : ℝ) * δ ^ 6 * φ ^ 9 + (57133 / 4 : ℝ) * δ ^ 6 * φ ^ 8
      + (507517 / 8 : ℝ) * δ ^ 6 * φ ^ 7 + (4722281 / 24 : ℝ) * δ ^ 6 * φ ^ 6
      + (3737017 / 8 : ℝ) * δ ^ 6 * φ ^ 5 + (6983097 / 8 : ℝ) * δ ^ 6 * φ ^ 4
      + (15057317 / 12 : ℝ) * δ ^ 6 * φ ^ 3 + (15039155 / 12 : ℝ) * δ ^ 6 * φ ^ 2
      + (2934631 / 4 : ℝ) * δ ^ 6 * φ + (1467289 / 8 : ℝ) * δ ^ 6
      + (477 / 2 : ℝ) * δ ^ 4 * φ ^ 10 + (2627 / 2 : ℝ) * δ ^ 4 * φ ^ 9
      + (8225 / 2 : ℝ) * δ ^ 4 * φ ^ 8 + (9362 : ℝ) * δ ^ 4 * φ ^ 7
      + (34533 / 2 : ℝ) * δ ^ 4 * φ ^ 6 + (54327 / 2 : ℝ) * δ ^ 4 * φ ^ 5
      + (74509 / 2 : ℝ) * δ ^ 4 * φ ^ 4 + (44360 : ℝ) * δ ^ 4 * φ ^ 3
      + (87485 / 2 : ℝ) * δ ^ 4 * φ ^ 2 + (57923 / 2 : ℝ) * δ ^ 4 * φ + (68533 / 8 : ℝ) * δ ^ 4
      + (11 : ℝ) * δ ^ 2 * φ ^ 11 + (31 : ℝ) * δ ^ 2 * φ ^ 10 + (49 : ℝ) * δ ^ 2 * φ ^ 9
      + (65 : ℝ) * δ ^ 2 * φ ^ 8 + (79 : ℝ) * δ ^ 2 * φ ^ 7 + (91 : ℝ) * δ ^ 2 * φ ^ 6
      + (101 : ℝ) * δ ^ 2 * φ ^ 5 + (109 : ℝ) * δ ^ 2 * φ ^ 4 + (115 : ℝ) * δ ^ 2 * φ ^ 3
      + (119 : ℝ) * δ ^ 2 * φ ^ 2 + (121 : ℝ) * δ ^ 2 * φ + (121 / 2 : ℝ) * δ ^ 2 := by
  unfold g_rec; norm_num [Finset.sum_range_succ]
  rw [v_val_1, v_val_2, v_val_3, v_val_4, v_val_5, v_val_6, v_val_7, v_val_8, v_val_9, v_val_10,
    v_val_11, v_val_12, e_val_0, e_val_1, e_val_2, e_val_3, e_val_4, e_val_5, e_val_6, e_val_7,
    e_val_8, e_val_9, e_val_10, e_val_11, e_val_12]
  ring

private lemma monom_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ)
    (c : ℝ) (hc : 0 ≤ c) (a b : ℕ) : 0 ≤ c * (δ ^ a * φ ^ b) :=
  mul_nonneg hc (mul_nonneg (pow_nonneg hδ0 a) (pow_nonneg hφ0 b))

private lemma monom_le_one {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (a b : ℕ) : 0 ≤ 1 - δ ^ a * φ ^ b := by
  have h1 : δ ^ a ≤ 1 := pow_le_one₀ hδ0 hδ1
  have h2 : φ ^ b ≤ 1 := pow_le_one₀ hφ0 hφ1
  have h3 := mul_le_mul h1 h2 (pow_nonneg hφ0 b) zero_le_one
  linarith

private lemma g4_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ g_rec δ φ 4 := by
  have hid : g_rec δ φ 4 =
      (13 / 2 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (19 / 2 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (37 / 8 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (3 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (7 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (9 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (9 / 2 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_4]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (13 / 2 : ℝ) (by norm_num) 4 2)
    (monom_nonneg hδ0 hφ0 (19 / 2 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (37 / 8 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (3 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (7 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (9 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (9 / 2 : ℝ) (by norm_num) 2 0)

private lemma g6_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ g_rec δ φ 6 := by
  have hid : g_rec δ φ 6 =
      (197 / 6 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (399 / 4 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (2749 / 24 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (673 / 16 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (69 / 2 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (299 / 2 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (647 / 2 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (320 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (116 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (5 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (13 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (19 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (23 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (25 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (25 / 2 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_6]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (197 / 6 : ℝ) (by norm_num) 6 3)
    (monom_nonneg hδ0 hφ0 (399 / 4 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (2749 / 24 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (673 / 16 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (69 / 2 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (299 / 2 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (647 / 2 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (320 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (116 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (5 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (13 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (19 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (23 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (25 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (25 / 2 : ℝ) (by norm_num) 2 0)

private lemma g8_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ g_rec δ φ 8 := by
  have hid : g_rec δ φ 8 =
      (1069 / 8 : ℝ) * (δ ^ 8 * φ ^ 4)
      + (2937 / 4 : ℝ) * (δ ^ 8 * φ ^ 3)
      + (24447 / 16 : ℝ) * (δ ^ 8 * φ ^ 2)
      + (21343 / 16 : ℝ) * (δ ^ 8 * φ ^ 1)
      + (54573 / 128 : ℝ) * (δ ^ 8 * φ ^ 0)
      + (249 : ℝ) * (δ ^ 6 * φ ^ 5)
      + (6721 / 4 : ℝ) * (δ ^ 6 * φ ^ 4)
      + (132955 / 24 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (74167 / 8 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (179567 / 24 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (36699 / 16 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (165 / 2 : ℝ) * (δ ^ 4 * φ ^ 6)
      + (827 / 2 : ℝ) * (δ ^ 4 * φ ^ 5)
      + (2293 / 2 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (2174 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (5901 / 2 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (4695 / 2 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (6113 / 8 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (7 : ℝ) * (δ ^ 2 * φ ^ 7)
      + (19 : ℝ) * (δ ^ 2 * φ ^ 6)
      + (29 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (37 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (43 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (47 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (49 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (49 / 2 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_8]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (1069 / 8 : ℝ) (by norm_num) 8 4)
    (monom_nonneg hδ0 hφ0 (2937 / 4 : ℝ) (by norm_num) 8 3))
    (monom_nonneg hδ0 hφ0 (24447 / 16 : ℝ) (by norm_num) 8 2))
    (monom_nonneg hδ0 hφ0 (21343 / 16 : ℝ) (by norm_num) 8 1))
    (monom_nonneg hδ0 hφ0 (54573 / 128 : ℝ) (by norm_num) 8 0))
    (monom_nonneg hδ0 hφ0 (249 : ℝ) (by norm_num) 6 5))
    (monom_nonneg hδ0 hφ0 (6721 / 4 : ℝ) (by norm_num) 6 4))
    (monom_nonneg hδ0 hφ0 (132955 / 24 : ℝ) (by norm_num) 6 3))
    (monom_nonneg hδ0 hφ0 (74167 / 8 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (179567 / 24 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (36699 / 16 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (165 / 2 : ℝ) (by norm_num) 4 6))
    (monom_nonneg hδ0 hφ0 (827 / 2 : ℝ) (by norm_num) 4 5))
    (monom_nonneg hδ0 hφ0 (2293 / 2 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (2174 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (5901 / 2 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (4695 / 2 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (6113 / 8 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (7 : ℝ) (by norm_num) 2 7))
    (monom_nonneg hδ0 hφ0 (19 : ℝ) (by norm_num) 2 6))
    (monom_nonneg hδ0 hφ0 (29 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (37 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (43 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (47 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (49 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (49 / 2 : ℝ) (by norm_num) 2 0)

private lemma g10_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ g_rec δ φ 10 := by
  have hid : g_rec δ φ 10 =
      (53689 / 120 : ℝ) * (δ ^ 10 * φ ^ 5)
      + (65399 / 16 : ℝ) * (δ ^ 10 * φ ^ 4)
      + (668303 / 48 : ℝ) * (δ ^ 10 * φ ^ 3)
      + (696251 / 32 : ℝ) * (δ ^ 10 * φ ^ 2)
      + (10287427 / 640 : ℝ) * (δ ^ 10 * φ ^ 1)
      + (1159075 / 256 : ℝ) * (δ ^ 10 * φ ^ 0)
      + (16145 / 12 : ℝ) * (δ ^ 8 * φ ^ 6)
      + (13078 : ℝ) * (δ ^ 8 * φ ^ 5)
      + (984559 / 16 : ℝ) * (δ ^ 8 * φ ^ 4)
      + (7540417 / 48 : ℝ) * (δ ^ 8 * φ ^ 3)
      + (10362523 / 48 : ℝ) * (δ ^ 8 * φ ^ 2)
      + (3573451 / 24 : ℝ) * (δ ^ 8 * φ ^ 1)
      + (2587379 / 64 : ℝ) * (δ ^ 8 * φ ^ 0)
      + (1565 / 2 : ℝ) * (δ ^ 6 * φ ^ 7)
      + (71705 / 12 : ℝ) * (δ ^ 6 * φ ^ 6)
      + (197123 / 8 : ℝ) * (δ ^ 6 * φ ^ 5)
      + (541045 / 8 : ℝ) * (δ ^ 6 * φ ^ 4)
      + (1030967 / 8 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (3747593 / 24 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (414689 / 4 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (225087 / 8 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (301 / 2 : ℝ) * (δ ^ 4 * φ ^ 8)
      + (1603 / 2 : ℝ) * (δ ^ 4 * φ ^ 7)
      + (4819 / 2 : ℝ) * (δ ^ 4 * φ ^ 6)
      + (5188 : ℝ) * (δ ^ 4 * φ ^ 5)
      + (17677 / 2 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (24611 / 2 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (27199 / 2 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (9652 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (2963 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (9 : ℝ) * (δ ^ 2 * φ ^ 9)
      + (25 : ℝ) * (δ ^ 2 * φ ^ 8)
      + (39 : ℝ) * (δ ^ 2 * φ ^ 7)
      + (51 : ℝ) * (δ ^ 2 * φ ^ 6)
      + (61 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (69 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (75 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (79 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (81 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (81 / 2 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_10]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (53689 / 120 : ℝ) (by norm_num) 10 5)
    (monom_nonneg hδ0 hφ0 (65399 / 16 : ℝ) (by norm_num) 10 4))
    (monom_nonneg hδ0 hφ0 (668303 / 48 : ℝ) (by norm_num) 10 3))
    (monom_nonneg hδ0 hφ0 (696251 / 32 : ℝ) (by norm_num) 10 2))
    (monom_nonneg hδ0 hφ0 (10287427 / 640 : ℝ) (by norm_num) 10 1))
    (monom_nonneg hδ0 hφ0 (1159075 / 256 : ℝ) (by norm_num) 10 0))
    (monom_nonneg hδ0 hφ0 (16145 / 12 : ℝ) (by norm_num) 8 6))
    (monom_nonneg hδ0 hφ0 (13078 : ℝ) (by norm_num) 8 5))
    (monom_nonneg hδ0 hφ0 (984559 / 16 : ℝ) (by norm_num) 8 4))
    (monom_nonneg hδ0 hφ0 (7540417 / 48 : ℝ) (by norm_num) 8 3))
    (monom_nonneg hδ0 hφ0 (10362523 / 48 : ℝ) (by norm_num) 8 2))
    (monom_nonneg hδ0 hφ0 (3573451 / 24 : ℝ) (by norm_num) 8 1))
    (monom_nonneg hδ0 hφ0 (2587379 / 64 : ℝ) (by norm_num) 8 0))
    (monom_nonneg hδ0 hφ0 (1565 / 2 : ℝ) (by norm_num) 6 7))
    (monom_nonneg hδ0 hφ0 (71705 / 12 : ℝ) (by norm_num) 6 6))
    (monom_nonneg hδ0 hφ0 (197123 / 8 : ℝ) (by norm_num) 6 5))
    (monom_nonneg hδ0 hφ0 (541045 / 8 : ℝ) (by norm_num) 6 4))
    (monom_nonneg hδ0 hφ0 (1030967 / 8 : ℝ) (by norm_num) 6 3))
    (monom_nonneg hδ0 hφ0 (3747593 / 24 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (414689 / 4 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (225087 / 8 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (301 / 2 : ℝ) (by norm_num) 4 8))
    (monom_nonneg hδ0 hφ0 (1603 / 2 : ℝ) (by norm_num) 4 7))
    (monom_nonneg hδ0 hφ0 (4819 / 2 : ℝ) (by norm_num) 4 6))
    (monom_nonneg hδ0 hφ0 (5188 : ℝ) (by norm_num) 4 5))
    (monom_nonneg hδ0 hφ0 (17677 / 2 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (24611 / 2 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (27199 / 2 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (9652 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (2963 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (9 : ℝ) (by norm_num) 2 9))
    (monom_nonneg hδ0 hφ0 (25 : ℝ) (by norm_num) 2 8))
    (monom_nonneg hδ0 hφ0 (39 : ℝ) (by norm_num) 2 7))
    (monom_nonneg hδ0 hφ0 (51 : ℝ) (by norm_num) 2 6))
    (monom_nonneg hδ0 hφ0 (61 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (69 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (75 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (79 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (81 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (81 / 2 : ℝ) (by norm_num) 2 0)

private lemma g12_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ g_rec δ φ 12 := by
  have hid : g_rec δ φ 12 =
      (790397 / 720 : ℝ) * (δ ^ 12 * φ ^ 6)
      + (1447469 / 80 : ℝ) * (δ ^ 12 * φ ^ 5)
      + (54650791 / 576 : ℝ) * (δ ^ 12 * φ ^ 4)
      + (22639255 / 96 : ℝ) * (δ ^ 12 * φ ^ 3)
      + (3510795803 / 11520 : ℝ) * (δ ^ 12 * φ ^ 2)
      + (755768003 / 3840 : ℝ) * (δ ^ 12 * φ ^ 1)
      + (51469441 / 1024 : ℝ) * (δ ^ 12 * φ ^ 0)
      + (43335 / 8 : ℝ) * (δ ^ 10 * φ ^ 7)
      + (923135 / 12 : ℝ) * (δ ^ 10 * φ ^ 6)
      + (5026049 / 10 : ℝ) * (δ ^ 10 * φ ^ 5)
      + (175164037 / 96 : ℝ) * (δ ^ 10 * φ ^ 4)
      + (486947393 / 128 : ℝ) * (δ ^ 10 * φ ^ 3)
      + (1713936175 / 384 : ℝ) * (δ ^ 10 * φ ^ 2)
      + (1744089939 / 640 : ℝ) * (δ ^ 10 * φ ^ 1)
      + (172005683 / 256 : ℝ) * (δ ^ 10 * φ ^ 0)
      + (126025 / 24 : ℝ) * (δ ^ 8 * φ ^ 8)
      + (674675 / 12 : ℝ) * (δ ^ 8 * φ ^ 7)
      + (15289013 / 48 : ℝ) * (δ ^ 8 * φ ^ 6)
      + (56974717 / 48 : ℝ) * (δ ^ 8 * φ ^ 5)
      + (49201535 / 16 : ℝ) * (δ ^ 8 * φ ^ 4)
      + (128101469 / 24 : ℝ) * (δ ^ 8 * φ ^ 3)
      + (137205523 / 24 : ℝ) * (δ ^ 8 * φ ^ 2)
      + (40132307 / 12 : ℝ) * (δ ^ 8 * φ ^ 1)
      + (25915231 / 32 : ℝ) * (δ ^ 8 * φ ^ 0)
      + (5320 / 3 : ℝ) * (δ ^ 6 * φ ^ 9)
      + (57133 / 4 : ℝ) * (δ ^ 6 * φ ^ 8)
      + (507517 / 8 : ℝ) * (δ ^ 6 * φ ^ 7)
      + (4722281 / 24 : ℝ) * (δ ^ 6 * φ ^ 6)
      + (3737017 / 8 : ℝ) * (δ ^ 6 * φ ^ 5)
      + (6983097 / 8 : ℝ) * (δ ^ 6 * φ ^ 4)
      + (15057317 / 12 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (15039155 / 12 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (2934631 / 4 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (1467289 / 8 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (477 / 2 : ℝ) * (δ ^ 4 * φ ^ 10)
      + (2627 / 2 : ℝ) * (δ ^ 4 * φ ^ 9)
      + (8225 / 2 : ℝ) * (δ ^ 4 * φ ^ 8)
      + (9362 : ℝ) * (δ ^ 4 * φ ^ 7)
      + (34533 / 2 : ℝ) * (δ ^ 4 * φ ^ 6)
      + (54327 / 2 : ℝ) * (δ ^ 4 * φ ^ 5)
      + (74509 / 2 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (44360 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (87485 / 2 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (57923 / 2 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (68533 / 8 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (11 : ℝ) * (δ ^ 2 * φ ^ 11)
      + (31 : ℝ) * (δ ^ 2 * φ ^ 10)
      + (49 : ℝ) * (δ ^ 2 * φ ^ 9)
      + (65 : ℝ) * (δ ^ 2 * φ ^ 8)
      + (79 : ℝ) * (δ ^ 2 * φ ^ 7)
      + (91 : ℝ) * (δ ^ 2 * φ ^ 6)
      + (101 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (109 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (115 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (119 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (121 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (121 / 2 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_12]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (790397 / 720 : ℝ) (by norm_num) 12 6)
    (monom_nonneg hδ0 hφ0 (1447469 / 80 : ℝ) (by norm_num) 12 5))
    (monom_nonneg hδ0 hφ0 (54650791 / 576 : ℝ) (by norm_num) 12 4))
    (monom_nonneg hδ0 hφ0 (22639255 / 96 : ℝ) (by norm_num) 12 3))
    (monom_nonneg hδ0 hφ0 (3510795803 / 11520 : ℝ) (by norm_num) 12 2))
    (monom_nonneg hδ0 hφ0 (755768003 / 3840 : ℝ) (by norm_num) 12 1))
    (monom_nonneg hδ0 hφ0 (51469441 / 1024 : ℝ) (by norm_num) 12 0))
    (monom_nonneg hδ0 hφ0 (43335 / 8 : ℝ) (by norm_num) 10 7))
    (monom_nonneg hδ0 hφ0 (923135 / 12 : ℝ) (by norm_num) 10 6))
    (monom_nonneg hδ0 hφ0 (5026049 / 10 : ℝ) (by norm_num) 10 5))
    (monom_nonneg hδ0 hφ0 (175164037 / 96 : ℝ) (by norm_num) 10 4))
    (monom_nonneg hδ0 hφ0 (486947393 / 128 : ℝ) (by norm_num) 10 3))
    (monom_nonneg hδ0 hφ0 (1713936175 / 384 : ℝ) (by norm_num) 10 2))
    (monom_nonneg hδ0 hφ0 (1744089939 / 640 : ℝ) (by norm_num) 10 1))
    (monom_nonneg hδ0 hφ0 (172005683 / 256 : ℝ) (by norm_num) 10 0))
    (monom_nonneg hδ0 hφ0 (126025 / 24 : ℝ) (by norm_num) 8 8))
    (monom_nonneg hδ0 hφ0 (674675 / 12 : ℝ) (by norm_num) 8 7))
    (monom_nonneg hδ0 hφ0 (15289013 / 48 : ℝ) (by norm_num) 8 6))
    (monom_nonneg hδ0 hφ0 (56974717 / 48 : ℝ) (by norm_num) 8 5))
    (monom_nonneg hδ0 hφ0 (49201535 / 16 : ℝ) (by norm_num) 8 4))
    (monom_nonneg hδ0 hφ0 (128101469 / 24 : ℝ) (by norm_num) 8 3))
    (monom_nonneg hδ0 hφ0 (137205523 / 24 : ℝ) (by norm_num) 8 2))
    (monom_nonneg hδ0 hφ0 (40132307 / 12 : ℝ) (by norm_num) 8 1))
    (monom_nonneg hδ0 hφ0 (25915231 / 32 : ℝ) (by norm_num) 8 0))
    (monom_nonneg hδ0 hφ0 (5320 / 3 : ℝ) (by norm_num) 6 9))
    (monom_nonneg hδ0 hφ0 (57133 / 4 : ℝ) (by norm_num) 6 8))
    (monom_nonneg hδ0 hφ0 (507517 / 8 : ℝ) (by norm_num) 6 7))
    (monom_nonneg hδ0 hφ0 (4722281 / 24 : ℝ) (by norm_num) 6 6))
    (monom_nonneg hδ0 hφ0 (3737017 / 8 : ℝ) (by norm_num) 6 5))
    (monom_nonneg hδ0 hφ0 (6983097 / 8 : ℝ) (by norm_num) 6 4))
    (monom_nonneg hδ0 hφ0 (15057317 / 12 : ℝ) (by norm_num) 6 3))
    (monom_nonneg hδ0 hφ0 (15039155 / 12 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (2934631 / 4 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (1467289 / 8 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (477 / 2 : ℝ) (by norm_num) 4 10))
    (monom_nonneg hδ0 hφ0 (2627 / 2 : ℝ) (by norm_num) 4 9))
    (monom_nonneg hδ0 hφ0 (8225 / 2 : ℝ) (by norm_num) 4 8))
    (monom_nonneg hδ0 hφ0 (9362 : ℝ) (by norm_num) 4 7))
    (monom_nonneg hδ0 hφ0 (34533 / 2 : ℝ) (by norm_num) 4 6))
    (monom_nonneg hδ0 hφ0 (54327 / 2 : ℝ) (by norm_num) 4 5))
    (monom_nonneg hδ0 hφ0 (74509 / 2 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (44360 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (87485 / 2 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (57923 / 2 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (68533 / 8 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (11 : ℝ) (by norm_num) 2 11))
    (monom_nonneg hδ0 hφ0 (31 : ℝ) (by norm_num) 2 10))
    (monom_nonneg hδ0 hφ0 (49 : ℝ) (by norm_num) 2 9))
    (monom_nonneg hδ0 hφ0 (65 : ℝ) (by norm_num) 2 8))
    (monom_nonneg hδ0 hφ0 (79 : ℝ) (by norm_num) 2 7))
    (monom_nonneg hδ0 hφ0 (91 : ℝ) (by norm_num) 2 6))
    (monom_nonneg hδ0 hφ0 (101 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (109 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (115 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (119 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (121 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (121 / 2 : ℝ) (by norm_num) 2 0)

private lemma neg_g5_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ -g_rec δ φ 5 := by
  have hid : -g_rec δ φ 5 =
      (18 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (64 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (77 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (123 / 4 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (4 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (10 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (14 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (16 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (8 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_5]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (18 : ℝ) (by norm_num) 4 3)
    (monom_nonneg hδ0 hφ0 (64 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (77 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (123 / 4 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (4 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (10 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (14 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (16 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (8 : ℝ) (by norm_num) 2 0)

private lemma g5_l1_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    0 ≤ (967 / 4 : ℝ) * δ ^ 2 + g_rec δ φ 5 := by
  have hid : (967 / 4 : ℝ) * δ ^ 2 + g_rec δ φ 5 =
      (18 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 3)
      + (64 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 2)
      + (77 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 1)
      + (123 / 4 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 0)
      + (4 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 4)
      + (10 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 3)
      + (14 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 2)
      + (16 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 1) := by
    rw [g_val_5]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 3))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 1))

private lemma neg_g7_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ -g_rec δ φ 7 := by
  have hid : -g_rec δ φ 7 =
      (110 : ℝ) * (δ ^ 6 * φ ^ 4)
      + (1886 / 3 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (1366 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (7619 / 6 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (1695 / 4 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (56 : ℝ) * (δ ^ 4 * φ ^ 5)
      + (266 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (680 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (1102 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (959 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (1305 / 4 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (6 : ℝ) * (δ ^ 2 * φ ^ 6)
      + (16 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (24 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (30 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (34 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (36 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (18 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_7]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (110 : ℝ) (by norm_num) 6 4)
    (monom_nonneg hδ0 hφ0 (1886 / 3 : ℝ) (by norm_num) 6 3))
    (monom_nonneg hδ0 hφ0 (1366 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (7619 / 6 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (1695 / 4 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (56 : ℝ) (by norm_num) 4 5))
    (monom_nonneg hδ0 hφ0 (266 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (680 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (1102 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (959 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (1305 / 4 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (6 : ℝ) (by norm_num) 2 6))
    (monom_nonneg hδ0 hφ0 (16 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (24 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (30 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (34 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (36 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (18 : ℝ) (by norm_num) 2 0)

private lemma g7_l1_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    0 ≤ (14703 / 2 : ℝ) * δ ^ 2 + g_rec δ φ 7 := by
  have hid : (14703 / 2 : ℝ) * δ ^ 2 + g_rec δ φ 7 =
      (110 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 4)
      + (1886 / 3 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 3)
      + (1366 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 2)
      + (7619 / 6 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 1)
      + (1695 / 4 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 0)
      + (56 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 5)
      + (266 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 4)
      + (680 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 3)
      + (1102 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 2)
      + (959 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 1)
      + (1305 / 4 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 0)
      + (6 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 6)
      + (16 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 5)
      + (24 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 4)
      + (30 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 3)
      + (34 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 2)
      + (36 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 1) := by
    rw [g_val_7]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 4))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 6)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 1))

private lemma neg_g9_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ -g_rec δ φ 9 := by
  have hid : -g_rec δ φ 9 =
      (1567 / 3 : ℝ) * (δ ^ 8 * φ ^ 5)
      + (53437 / 12 : ℝ) * (δ ^ 8 * φ ^ 4)
      + (186103 / 12 : ℝ) * (δ ^ 8 * φ ^ 3)
      + (304493 / 12 : ℝ) * (δ ^ 8 * φ ^ 2)
      + (467909 / 24 : ℝ) * (δ ^ 8 * φ ^ 1)
      + (364441 / 64 : ℝ) * (δ ^ 8 * φ ^ 0)
      + (1402 / 3 : ℝ) * (δ ^ 6 * φ ^ 6)
      + (3408 : ℝ) * (δ ^ 6 * φ ^ 5)
      + (52199 / 4 : ℝ) * (δ ^ 6 * φ ^ 4)
      + (371569 / 12 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (258643 / 6 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (93551 / 3 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (17875 / 2 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (114 : ℝ) * (δ ^ 4 * φ ^ 7)
      + (592 : ℝ) * (δ ^ 4 * φ ^ 6)
      + (1723 : ℝ) * (δ ^ 4 * φ ^ 5)
      + (3536 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (5577 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (6714 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (5006 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (3153 / 2 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (8 : ℝ) * (δ ^ 2 * φ ^ 8)
      + (22 : ℝ) * (δ ^ 2 * φ ^ 7)
      + (34 : ℝ) * (δ ^ 2 * φ ^ 6)
      + (44 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (52 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (58 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (62 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (64 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (32 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_9]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (1567 / 3 : ℝ) (by norm_num) 8 5)
    (monom_nonneg hδ0 hφ0 (53437 / 12 : ℝ) (by norm_num) 8 4))
    (monom_nonneg hδ0 hφ0 (186103 / 12 : ℝ) (by norm_num) 8 3))
    (monom_nonneg hδ0 hφ0 (304493 / 12 : ℝ) (by norm_num) 8 2))
    (monom_nonneg hδ0 hφ0 (467909 / 24 : ℝ) (by norm_num) 8 1))
    (monom_nonneg hδ0 hφ0 (364441 / 64 : ℝ) (by norm_num) 8 0))
    (monom_nonneg hδ0 hφ0 (1402 / 3 : ℝ) (by norm_num) 6 6))
    (monom_nonneg hδ0 hφ0 (3408 : ℝ) (by norm_num) 6 5))
    (monom_nonneg hδ0 hφ0 (52199 / 4 : ℝ) (by norm_num) 6 4))
    (monom_nonneg hδ0 hφ0 (371569 / 12 : ℝ) (by norm_num) 6 3))
    (monom_nonneg hδ0 hφ0 (258643 / 6 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (93551 / 3 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (17875 / 2 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (114 : ℝ) (by norm_num) 4 7))
    (monom_nonneg hδ0 hφ0 (592 : ℝ) (by norm_num) 4 6))
    (monom_nonneg hδ0 hφ0 (1723 : ℝ) (by norm_num) 4 5))
    (monom_nonneg hδ0 hφ0 (3536 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (5577 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (6714 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (5006 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (3153 / 2 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (8 : ℝ) (by norm_num) 2 8))
    (monom_nonneg hδ0 hφ0 (22 : ℝ) (by norm_num) 2 7))
    (monom_nonneg hδ0 hφ0 (34 : ℝ) (by norm_num) 2 6))
    (monom_nonneg hδ0 hφ0 (44 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (52 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (58 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (62 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (64 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (32 : ℝ) (by norm_num) 2 0)

private lemma g9_l1_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    0 ≤ (14552385 / 64 : ℝ) * δ ^ 2 + g_rec δ φ 9 := by
  have hid : (14552385 / 64 : ℝ) * δ ^ 2 + g_rec δ φ 9 =
      (1567 / 3 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 5)
      + (53437 / 12 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 4)
      + (186103 / 12 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 3)
      + (304493 / 12 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 2)
      + (467909 / 24 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 1)
      + (364441 / 64 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 0)
      + (1402 / 3 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 6)
      + (3408 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 5)
      + (52199 / 4 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 4)
      + (371569 / 12 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 3)
      + (258643 / 6 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 2)
      + (93551 / 3 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 1)
      + (17875 / 2 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 0)
      + (114 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 7)
      + (592 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 6)
      + (1723 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 5)
      + (3536 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 4)
      + (5577 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 3)
      + (6714 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 2)
      + (5006 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 1)
      + (3153 / 2 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 0)
      + (8 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 8)
      + (22 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 7)
      + (34 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 6)
      + (44 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 5)
      + (52 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 4)
      + (58 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 3)
      + (62 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 2)
      + (64 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 1) := by
    rw [g_val_9]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 5))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 6)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 7)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 6)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 8)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 7)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 6)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 1))

private lemma neg_g11_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hφ0 : 0 ≤ φ) :
    0 ≤ -g_rec δ φ 11 := by
  have hid : -g_rec δ φ 11 =
      (3871 / 2 : ℝ) * (δ ^ 10 * φ ^ 6)
      + (368666 / 15 : ℝ) * (δ ^ 10 * φ ^ 5)
      + (508721 / 4 : ℝ) * (δ ^ 10 * φ ^ 4)
      + (3911783 / 12 : ℝ) * (δ ^ 10 * φ ^ 3)
      + (13934791 / 32 : ℝ) * (δ ^ 10 * φ ^ 2)
      + (23209403 / 80 : ℝ) * (δ ^ 10 * φ ^ 1)
      + (2435843 / 32 : ℝ) * (δ ^ 10 * φ ^ 0)
      + (2830 : ℝ) * (δ ^ 8 * φ ^ 7)
      + (87746 / 3 : ℝ) * (δ ^ 8 * φ ^ 6)
      + (155584 : ℝ) * (δ ^ 8 * φ ^ 5)
      + (513629 : ℝ) * (δ ^ 8 * φ ^ 4)
      + (12637681 / 12 : ℝ) * (δ ^ 8 * φ ^ 3)
      + (7565441 / 6 : ℝ) * (δ ^ 8 * φ ^ 2)
      + (19086793 / 24 : ℝ) * (δ ^ 8 * φ ^ 1)
      + (12993133 / 64 : ℝ) * (δ ^ 8 * φ ^ 0)
      + (1212 : ℝ) * (δ ^ 6 * φ ^ 8)
      + (9546 : ℝ) * (δ ^ 6 * φ ^ 7)
      + (246913 / 6 : ℝ) * (δ ^ 6 * φ ^ 6)
      + (121865 : ℝ) * (δ ^ 6 * φ ^ 5)
      + (1071227 / 4 : ℝ) * (δ ^ 6 * φ ^ 4)
      + (1739355 / 4 : ℝ) * (δ ^ 6 * φ ^ 3)
      + (1419559 / 3 : ℝ) * (δ ^ 6 * φ ^ 2)
      + (293172 : ℝ) * (δ ^ 6 * φ ^ 1)
      + (76082 : ℝ) * (δ ^ 6 * φ ^ 0)
      + (192 : ℝ) * (δ ^ 4 * φ ^ 9)
      + (1042 : ℝ) * (δ ^ 4 * φ ^ 8)
      + (3206 : ℝ) * (δ ^ 4 * φ ^ 7)
      + (7130 : ℝ) * (δ ^ 4 * φ ^ 6)
      + (12735 : ℝ) * (δ ^ 4 * φ ^ 5)
      + (19122 : ℝ) * (δ ^ 4 * φ ^ 4)
      + (24349 : ℝ) * (δ ^ 4 * φ ^ 3)
      + (25236 : ℝ) * (δ ^ 4 * φ ^ 2)
      + (17234 : ℝ) * (δ ^ 4 * φ ^ 1)
      + (10367 / 2 : ℝ) * (δ ^ 4 * φ ^ 0)
      + (10 : ℝ) * (δ ^ 2 * φ ^ 10)
      + (28 : ℝ) * (δ ^ 2 * φ ^ 9)
      + (44 : ℝ) * (δ ^ 2 * φ ^ 8)
      + (58 : ℝ) * (δ ^ 2 * φ ^ 7)
      + (70 : ℝ) * (δ ^ 2 * φ ^ 6)
      + (80 : ℝ) * (δ ^ 2 * φ ^ 5)
      + (88 : ℝ) * (δ ^ 2 * φ ^ 4)
      + (94 : ℝ) * (δ ^ 2 * φ ^ 3)
      + (98 : ℝ) * (δ ^ 2 * φ ^ 2)
      + (100 : ℝ) * (δ ^ 2 * φ ^ 1)
      + (50 : ℝ) * (δ ^ 2 * φ ^ 0) := by
    rw [g_val_11]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (monom_nonneg hδ0 hφ0 (3871 / 2 : ℝ) (by norm_num) 10 6)
    (monom_nonneg hδ0 hφ0 (368666 / 15 : ℝ) (by norm_num) 10 5))
    (monom_nonneg hδ0 hφ0 (508721 / 4 : ℝ) (by norm_num) 10 4))
    (monom_nonneg hδ0 hφ0 (3911783 / 12 : ℝ) (by norm_num) 10 3))
    (monom_nonneg hδ0 hφ0 (13934791 / 32 : ℝ) (by norm_num) 10 2))
    (monom_nonneg hδ0 hφ0 (23209403 / 80 : ℝ) (by norm_num) 10 1))
    (monom_nonneg hδ0 hφ0 (2435843 / 32 : ℝ) (by norm_num) 10 0))
    (monom_nonneg hδ0 hφ0 (2830 : ℝ) (by norm_num) 8 7))
    (monom_nonneg hδ0 hφ0 (87746 / 3 : ℝ) (by norm_num) 8 6))
    (monom_nonneg hδ0 hφ0 (155584 : ℝ) (by norm_num) 8 5))
    (monom_nonneg hδ0 hφ0 (513629 : ℝ) (by norm_num) 8 4))
    (monom_nonneg hδ0 hφ0 (12637681 / 12 : ℝ) (by norm_num) 8 3))
    (monom_nonneg hδ0 hφ0 (7565441 / 6 : ℝ) (by norm_num) 8 2))
    (monom_nonneg hδ0 hφ0 (19086793 / 24 : ℝ) (by norm_num) 8 1))
    (monom_nonneg hδ0 hφ0 (12993133 / 64 : ℝ) (by norm_num) 8 0))
    (monom_nonneg hδ0 hφ0 (1212 : ℝ) (by norm_num) 6 8))
    (monom_nonneg hδ0 hφ0 (9546 : ℝ) (by norm_num) 6 7))
    (monom_nonneg hδ0 hφ0 (246913 / 6 : ℝ) (by norm_num) 6 6))
    (monom_nonneg hδ0 hφ0 (121865 : ℝ) (by norm_num) 6 5))
    (monom_nonneg hδ0 hφ0 (1071227 / 4 : ℝ) (by norm_num) 6 4))
    (monom_nonneg hδ0 hφ0 (1739355 / 4 : ℝ) (by norm_num) 6 3))
    (monom_nonneg hδ0 hφ0 (1419559 / 3 : ℝ) (by norm_num) 6 2))
    (monom_nonneg hδ0 hφ0 (293172 : ℝ) (by norm_num) 6 1))
    (monom_nonneg hδ0 hφ0 (76082 : ℝ) (by norm_num) 6 0))
    (monom_nonneg hδ0 hφ0 (192 : ℝ) (by norm_num) 4 9))
    (monom_nonneg hδ0 hφ0 (1042 : ℝ) (by norm_num) 4 8))
    (monom_nonneg hδ0 hφ0 (3206 : ℝ) (by norm_num) 4 7))
    (monom_nonneg hδ0 hφ0 (7130 : ℝ) (by norm_num) 4 6))
    (monom_nonneg hδ0 hφ0 (12735 : ℝ) (by norm_num) 4 5))
    (monom_nonneg hδ0 hφ0 (19122 : ℝ) (by norm_num) 4 4))
    (monom_nonneg hδ0 hφ0 (24349 : ℝ) (by norm_num) 4 3))
    (monom_nonneg hδ0 hφ0 (25236 : ℝ) (by norm_num) 4 2))
    (monom_nonneg hδ0 hφ0 (17234 : ℝ) (by norm_num) 4 1))
    (monom_nonneg hδ0 hφ0 (10367 / 2 : ℝ) (by norm_num) 4 0))
    (monom_nonneg hδ0 hφ0 (10 : ℝ) (by norm_num) 2 10))
    (monom_nonneg hδ0 hφ0 (28 : ℝ) (by norm_num) 2 9))
    (monom_nonneg hδ0 hφ0 (44 : ℝ) (by norm_num) 2 8))
    (monom_nonneg hδ0 hφ0 (58 : ℝ) (by norm_num) 2 7))
    (monom_nonneg hδ0 hφ0 (70 : ℝ) (by norm_num) 2 6))
    (monom_nonneg hδ0 hφ0 (80 : ℝ) (by norm_num) 2 5))
    (monom_nonneg hδ0 hφ0 (88 : ℝ) (by norm_num) 2 4))
    (monom_nonneg hδ0 hφ0 (94 : ℝ) (by norm_num) 2 3))
    (monom_nonneg hδ0 hφ0 (98 : ℝ) (by norm_num) 2 2))
    (monom_nonneg hδ0 hφ0 (100 : ℝ) (by norm_num) 2 1))
    (monom_nonneg hδ0 hφ0 (50 : ℝ) (by norm_num) 2 0)

private lemma g11_l1_nonneg {δ φ : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    0 ≤ (456321605 / 64 : ℝ) * δ ^ 2 + g_rec δ φ 11 := by
  have hid : (456321605 / 64 : ℝ) * δ ^ 2 + g_rec δ φ 11 =
      (3871 / 2 : ℝ) * δ ^ 2 * (1 - δ ^ 8 * φ ^ 6)
      + (368666 / 15 : ℝ) * δ ^ 2 * (1 - δ ^ 8 * φ ^ 5)
      + (508721 / 4 : ℝ) * δ ^ 2 * (1 - δ ^ 8 * φ ^ 4)
      + (3911783 / 12 : ℝ) * δ ^ 2 * (1 - δ ^ 8 * φ ^ 3)
      + (13934791 / 32 : ℝ) * δ ^ 2 * (1 - δ ^ 8 * φ ^ 2)
      + (23209403 / 80 : ℝ) * δ ^ 2 * (1 - δ ^ 8 * φ ^ 1)
      + (2435843 / 32 : ℝ) * δ ^ 2 * (1 - δ ^ 8 * φ ^ 0)
      + (2830 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 7)
      + (87746 / 3 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 6)
      + (155584 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 5)
      + (513629 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 4)
      + (12637681 / 12 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 3)
      + (7565441 / 6 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 2)
      + (19086793 / 24 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 1)
      + (12993133 / 64 : ℝ) * δ ^ 2 * (1 - δ ^ 6 * φ ^ 0)
      + (1212 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 8)
      + (9546 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 7)
      + (246913 / 6 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 6)
      + (121865 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 5)
      + (1071227 / 4 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 4)
      + (1739355 / 4 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 3)
      + (1419559 / 3 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 2)
      + (293172 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 1)
      + (76082 : ℝ) * δ ^ 2 * (1 - δ ^ 4 * φ ^ 0)
      + (192 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 9)
      + (1042 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 8)
      + (3206 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 7)
      + (7130 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 6)
      + (12735 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 5)
      + (19122 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 4)
      + (24349 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 3)
      + (25236 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 2)
      + (17234 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 1)
      + (10367 / 2 : ℝ) * δ ^ 2 * (1 - δ ^ 2 * φ ^ 0)
      + (10 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 10)
      + (28 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 9)
      + (44 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 8)
      + (58 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 7)
      + (70 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 6)
      + (80 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 5)
      + (88 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 4)
      + (94 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 3)
      + (98 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 2)
      + (100 : ℝ) * δ ^ 2 * (1 - δ ^ 0 * φ ^ 1) := by
    rw [g_val_11]
    ring
  rw [hid]
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 8 6))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 8 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 8 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 8 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 8 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 8 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 8 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 7)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 6)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 6 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 8)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 7)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 6)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 4 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 9)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 8)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 7)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 6)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 1)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 2 0)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 10)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 9)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 8)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 7)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 6)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 5)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 4)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 3)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 2)))
    (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg δ)) (monom_le_one hδ0 hδ1 hφ0 hφ1 0 1))

/-- Nonpositivity of all four odd coefficient polynomials, from their
termwise rational certificates rather than parameter samples. -/
theorem odd_coefficient_nonpos (δ φ : ℝ) (hδ : 0 ≤ δ) (hφ : 0 ≤ φ) :
    g_rec δ φ 5 ≤ 0 ∧ g_rec δ φ 7 ≤ 0 ∧
    g_rec δ φ 9 ≤ 0 ∧ g_rec δ φ 11 ≤ 0 := by
  exact ⟨by linarith [neg_g5_nonneg hδ hφ], by linarith [neg_g7_nonneg hδ hφ],
    by linarith [neg_g9_nonneg hδ hφ], by linarith [neg_g11_nonneg hδ hφ]⟩

/-- Exact sums of absolute coefficients, evaluated at delta = phi = 1. -/
theorem odd_coefficient_sums :
    g_rec 1 1 5 = -(967/4:ℝ) ∧ g_rec 1 1 7 = -(14703/2:ℝ) ∧
    g_rec 1 1 9 = -(14552385/64:ℝ) ∧ g_rec 1 1 11 = -(456321605/64:ℝ) := by
  norm_num [g_val_5,g_val_7,g_val_9,g_val_11]


/-- The exact kernel-checked recurrence certificate through degree twelve. -/
theorem coefficient_bounds (δ φ : ℝ) (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    g_rec δ φ 0 = 0 ∧ g_rec δ φ 1 = 0 ∧
    g_rec δ φ 2 = δ ^ 2 * (1 / 2 + φ) ∧
    g_rec δ φ 3 = -2 * δ ^ 2 * (1 + φ) ^ 2 ∧
    0 ≤ g_rec δ φ 4 ∧ 0 ≤ g_rec δ φ 6 ∧ 0 ≤ g_rec δ φ 8 ∧
    0 ≤ g_rec δ φ 10 ∧ 0 ≤ g_rec δ φ 12 ∧
    -(967 / 4) * δ ^ 2 ≤ g_rec δ φ 5 ∧
    -(14703 / 2) * δ ^ 2 ≤ g_rec δ φ 7 ∧
    -(14552385 / 64) * δ ^ 2 ≤ g_rec δ φ 9 ∧
    -(456321605 / 64) * δ ^ 2 ≤ g_rec δ φ 11 := by
  refine ⟨g_val_0 δ φ, g_val_1 δ φ, ?_, ?_, g4_nonneg hδ0 hφ0,
    g6_nonneg hδ0 hφ0, g8_nonneg hδ0 hφ0, g10_nonneg hδ0 hφ0,
    g12_nonneg hδ0 hφ0, ?_, ?_, ?_, ?_⟩
  · rw [g_val_2]; ring
  · rw [g_val_3]; ring
  · linarith [g5_l1_nonneg hδ0 hδ1 hφ0 hφ1]
  · linarith [g7_l1_nonneg hδ0 hδ1 hφ0 hφ1]
  · linarith [g9_l1_nonneg hδ0 hδ1 hφ0 hφ1]
  · linarith [g11_l1_nonneg hδ0 hδ1 hφ0 hφ1]

noncomputable def partialSum (δ φ b : ℝ) : ℝ :=
  ∑ n ∈ Finset.range 13, g_rec δ φ n * b ^ n

noncomputable def lowerMargin (b : ℝ) : ℝ :=
  1 / 2 - 2 * b - (967 / 4) * b ^ 3 - (14703 / 2) * b ^ 5 -
    (14552385 / 64) * b ^ 7 - (456321605 / 64) * b ^ 9 -
    12 * 6 ^ 13 * b ^ 11 / (1 - 6 * b)

/-- All even contributions are nonnegative; all four negative odd terms are included. -/
theorem partialSum_lower (δ φ b : ℝ) (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (hb0 : 0 ≤ b) (hb1 : b ≤ 1 / 13) :
    δ ^ 2 * b ^ 2 * (1 / 2 - 2 * b - (967 / 4) * b ^ 3 -
      (14703 / 2) * b ^ 5 - (14552385 / 64) * b ^ 7 -
      (456321605 / 64) * b ^ 9) ≤ partialSum δ φ b := by
  obtain ⟨h0,h1,h2,h3,h4,h6,h8,h10,h12,h5,h7,h9,h11⟩ :=
    coefficient_bounds δ φ hδ0 hδ1 hφ0 hφ1
  have hlead : δ ^ 2 * (1 / 2 - 2 * b) ≤ g_rec δ φ 2 + b * g_rec δ φ 3 := by
    rw [h2,h3]
    have ha : 0 ≤ φ * (1 - 6 * b) := mul_nonneg hφ0 (by linarith)
    have hb : 0 ≤ 2 * b * φ * (1 - φ) := by positivity
    nlinarith only [mul_nonneg (sq_nonneg δ) ha, mul_nonneg (sq_nonneg δ) hb]
  unfold partialSum
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, h0, h1, zero_mul, zero_add]
  nlinarith only [mul_le_mul_of_nonneg_right hlead (sq_nonneg b),
    mul_le_mul_of_nonneg_right h5 (pow_nonneg hb0 5),
    mul_le_mul_of_nonneg_right h7 (pow_nonneg hb0 7),
    mul_le_mul_of_nonneg_right h9 (pow_nonneg hb0 9),
    mul_le_mul_of_nonneg_right h11 (pow_nonneg hb0 11),
    mul_nonneg h4 (pow_nonneg hb0 4), mul_nonneg h6 (pow_nonneg hb0 6),
    mul_nonneg h8 (pow_nonneg hb0 8), mul_nonneg h10 (pow_nonneg hb0 10),
    mul_nonneg h12 (pow_nonneg hb0 12)]

/-- The exact rational margin in equation (10). -/
theorem lowerMargin_endpoint :
    lowerMargin (1 / 13) = (1531986426123 / 30880302174176 : ℝ) ∧
    (1 / 25 : ℝ) < 1531986426123 / 30880302174176 := by
  norm_num [lowerMargin]

theorem lowerMargin_lower (b : ℝ) (hb0 : 0 ≤ b) (hb1 : b ≤ 1 / 13) :
    (1531986426123 / 30880302174176 : ℝ) ≤ lowerMargin b := by
  have h3 : b ^ 3 ≤ (1 / 13 : ℝ) ^ 3 := by gcongr
  have h5 : b ^ 5 ≤ (1 / 13 : ℝ) ^ 5 := by gcongr
  have h7 : b ^ 7 ≤ (1 / 13 : ℝ) ^ 7 := by gcongr
  have h9 : b ^ 9 ≤ (1 / 13 : ℝ) ^ 9 := by gcongr
  have ht : (12 : ℝ) * 6 ^ 13 * b ^ 11 / (1 - 6 * b) ≤
      12 * 6 ^ 13 * (1 / 13) ^ 11 / (1 - 6 * (1 / 13)) := by gcongr
  unfold lowerMargin
  norm_num at h3 h5 h7 h9 ht
  linarith

/-- The entire Cauchy remainder, with its amplitude factor, is an explicit input. -/
theorem sign_of_remainder (δ φ b G : ℝ) (hδ0 : 0 < δ) (hδ1 : δ ≤ 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (hb0 : 0 < b) (hb1 : b ≤ 1 / 13)
    (htail : |G - partialSum δ φ b| ≤ 12 * δ ^ 2 * (6 * b) ^ 13 / (1 - 6 * b)) :
    δ ^ 2 * b ^ 2 / 25 ≤ G ∧ 0 < G := by
  have hden : 1 - 6 * b ≠ 0 := by linarith
  have hp := partialSum_lower δ φ b hδ0.le hδ1 hφ0 hφ1 hb0.le hb1
  have ht := (abs_le.mp htail).1
  have hid : δ ^ 2 * b ^ 2 * lowerMargin b =
      δ ^ 2 * b ^ 2 * (1 / 2 - 2 * b - (967 / 4) * b ^ 3 -
        (14703 / 2) * b ^ 5 - (14552385 / 64) * b ^ 7 - (456321605 / 64) * b ^ 9) -
      12 * δ ^ 2 * (6 * b) ^ 13 / (1 - 6 * b) := by
    unfold lowerMargin
    field_simp
  have hg : δ ^ 2 * b ^ 2 * lowerMargin b ≤ G := by linarith
  have hm := mul_le_mul_of_nonneg_left (lowerMargin_lower b hb0.le hb1)
    (by positivity : 0 ≤ δ ^ 2 * b ^ 2)
  have hweak : δ ^ 2 * b ^ 2 / 25 ≤ G := by nlinarith only [hg,hm, sq_nonneg (δ*b)]
  exact ⟨hweak, lt_of_lt_of_le (by positivity) hweak⟩

/-- Normalization and exact endpoint conversion. -/
noncomputable def qOfB (b : ℝ) : ℝ := b / (1 - b) ^ 2

theorem normalization_derivative (b : ℝ) (hb : b < 1) :
    HasDerivAt qOfB ((1+b)/(1-b)^3) b ∧
    qOfB b / ((1+b)/(1-b)^3) = b*(1-b)/(1+b) := by
  have hd : 1-b ≠ 0 := ne_of_gt (sub_pos.mpr hb)
  have h := (hasDerivAt_id b).div (((hasDerivAt_const b (1 : ℝ)).sub
    (hasDerivAt_id b)).pow 2) (pow_ne_zero 2 hd)
  constructor
  · convert h using 1
    · rfl
    · dsimp
      field_simp
      ring
  · unfold qOfB
    field_simp

theorem normalization_strictMono : StrictMonoOn qOfB (Ioo (0 : ℝ) 1) := by
  intro a ha b hb hab
  have hda : 0 < (1-a)^2 := sq_pos_of_pos (by linarith [ha.2])
  have hdb : 0 < (1-b)^2 := sq_pos_of_pos (by linarith [hb.2])
  unfold qOfB
  rw [div_lt_div_iff₀ hda hdb]
  have hp : 0 < (b-a)*(1-a*b) := by
    apply mul_pos (sub_pos.mpr hab)
    have hprod : a*b < 1 := by nlinarith [ha.1,ha.2,hb.1,hb.2]
    linarith
  nlinarith only [hp]

theorem endpoint_conversion :
    qOfB (1 / 13) = (13 / 144 : ℝ) ∧
    (1 + 2 * (13 / 144 : ℝ)) / (1 + 4 * (13 / 144)) = 85 / 98 := by
  norm_num [qOfB]

/-- Every positive q has the normalized b used in the two final theorems. -/
theorem normalization_exists (q : ℝ) (hq : 0 < q) :
    ∃ b : ℝ, 0 < b ∧ b < 1 ∧ qOfB b = q := by
  let t := Real.sqrt (1+4*q)
  have ht0 : 0 ≤ t := Real.sqrt_nonneg _
  have ht2 : t^2 = 1+4*q := Real.sq_sqrt (by positivity)
  have ht : 1 < t := by nlinarith
  have htp : 0 < t+1 := by linarith
  refine ⟨(t-1)/(t+1), div_pos (by linarith) htp,
    (div_lt_one htp).mpr (by linarith), ?_⟩
  have hid : 1-(t-1)/(t+1) = 2/(t+1) := by field_simp; ring
  unfold qOfB
  rw [hid]
  field_simp
  nlinarith only [ht2]

/-- Equation (11) is supplied by invariant ratios and the concave-log chord. -/
theorem q_le_of_chord (q R s : ℝ) (hq : 0 ≤ q)
    (hs : Real.sqrt (85 / 98) ≤ s) (hthreshold : s = Real.exp (-R))
    (hchord : (1 / 2 : ℝ) * Real.log ((1 + 4*q)/(1 + 2*q)) ≤ R) :
    q ≤ 13 / 144 := by
  have hsq : (85/98 : ℝ) ≤ s^2 := by
    have hh := pow_le_pow_left₀ (Real.sqrt_nonneg (85/98)) hs 2
    rw [Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 85/98)] at hh
    exact hh
  have hden : 0 < 1 + 2*q := by positivity
  have hnum : 0 < 1 + 4*q := by positivity
  have he : (1+4*q)/(1+2*q) ≤ Real.exp (2*R) := by
    rw [← Real.exp_log (div_pos hnum hden)]
    apply Real.exp_le_exp.mpr
    linarith
  have hid : Real.exp (2*R)*s^2=1 := by
    rw [hthreshold,sq,← Real.exp_add,← Real.exp_add]
    rw [show 2*R+(-R + -R)=0 by ring,Real.exp_zero]
  have hm := mul_le_mul_of_nonneg_right he (sq_nonneg s)
  rw [hid] at hm
  have hprod : (1+4*q)*s^2 ≤ 1+2*q := by
    apply (div_le_one hden).mp
    simpa [div_mul_eq_mul_div] using hm
  have hh := mul_le_mul_of_nonneg_left hsq hnum.le
  nlinarith only [hh,hprod]

theorem b_le_of_q (b : ℝ) (hb0 : 0 < b) (hb1 : b < 1)
    (hq : qOfB b ≤ 13/144) : b ≤ 1/13 := by
  by_contra! hh
  have hm := normalization_strictMono (by norm_num : (1/13:ℝ) ∈ Ioo 0 1)
    ⟨hb0,hb1⟩ hh
  rw [endpoint_conversion.1] at hm
  linarith

/-- Uniform derivative sign on the new closed-left interval. The seven analytic
inputs are the five inherited real-threshold conditions, the chord inequality,
and the Cauchy remainder including identification of its recurrence coefficients. -/
theorem extended_derivative_neg (δ φ s b R' : ℝ) (c_star R_real : ℝ → ℝ)
    (hδ0 : 0 < δ) (hδ1 : δ < 1) (hφ0 : 0 ≤ φ) (hφ1 : φ < 1)
    (hs : Real.sqrt (85/98) ≤ s) (hs1 : s < 1)
    (hb0 : 0 < b) (hb1 : b < 1) (hbq : qOfB b = c_star s / s^2)
    (hc_pos : 0 < c_star s) (hc_cont : ContinuousAt c_star s)
    (hR_sderiv : HasStrictDerivAt R_real R' (c_star s / s^2))
    (hV_pos : 0 < (c_star s / s^2)*R')
    (h_thresh : ∀ᶠ u in 𝓝 s, Real.log u + R_real (c_star u / u^2) = 0)
    (hchord : (1/2:ℝ)*Real.log ((1+4*(c_star s/s^2))/(1+2*(c_star s/s^2))) ≤
      R_real (c_star s/s^2))
    (htail : |Gfun (R_real (c_star s/s^2)) ((c_star s/s^2)*R') - partialSum δ φ b| ≤
      12*δ^2*(6*b)^13/(1-6*b)) :
    deriv (fun u => Real.log (c_star u / (1-u))) s < 0 := by
  have hs0 : 0 < s := lt_of_lt_of_le (Real.sqrt_pos.2 (by norm_num)) hs
  obtain ⟨_,hq,he,hD,hsign⟩ :=
    derivative_identity s R' c_star R_real hs0 hs1 hc_pos hc_cont hR_sderiv hV_pos h_thresh
  have hq1 := q_le_of_chord (c_star s/s^2) (R_real (c_star s/s^2)) s hq.le hs he hchord
  have hb := b_le_of_q b hb0 hb1 (by rwa [hbq])
  have hg := (sign_of_remainder δ φ b _ hδ0 hδ1.le hφ0 hφ1.le hb0 hb htail).2
  rw [hD.deriv]
  exact hsign.mpr hg


noncomputable def eta (b : ℝ) : ℝ := (1-b)^2/(8*b)
noncomputable def K (b φ : ℝ) : ℝ :=
  b^2*(1-b)*(1+2*φ+2*b*φ-b^2*φ^2)/(2*(1+b)^3*(1+b*φ)^2)

/-- Scalar complex-disk constants. Analytic pullback and Cauchy implications
are not concluded from these numerical inequalities alone. -/
theorem complex_disk_constants :
    (1/6:ℝ)*(1+1/3)/(1-1/3)=1/3 ∧
    (1/6:ℝ)*2/(1-1/3)^2=3/4 ∧
    (1/3:ℝ)/(1-3/4)=4/3 ∧
    (1/5:ℝ)+(7/5)*(4/3)/(2/3)=3 ∧
    (3/4:ℝ)+(2+7/4)*3=12 ∧
    (167/1000:ℝ)*(1001/1000+17/50)/(1-17/50)<17/50 ∧
    (167/1000:ℝ)*(1+1001/1000)/(1-17/50)^2<1 := by norm_num

/-- Strict amplitude-disk invariance and contraction reuse the checked P7 disk. -/
theorem amplitude_disk (b : ℝ) (hb0 : 0 < b) (hb1 : b < 1) (δ w : ℂ)
    (hδ : ‖δ‖ ≤ eta b) (hw : ‖w‖ ≤ (1-b)/2) :
    ‖(b:ℂ)*(δ-w)/(1+w)‖ < (1-b)/2 ∧
    ‖-(b:ℂ)*(1+δ)/(1+w)^2‖ < 1 :=
  Dobson.GapFourConstants.p7_disk b hb0 hb1 δ w hδ hw

/-- Both bounds for the quadratic amplitude coefficient, with positive radius. -/
theorem amplitude_constants (b φ : ℝ) (hb0 : 0 < b) (hb1 : b < 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    0 < eta b ∧ 0 < K b φ ∧ b^2*(1-b)/16 ≤ K b φ ∧ K b φ * (eta b)^2 ≤ 3/128 := by
  have ht : 0 < 1-b := by linarith
  have hbp : 0 < 1+b := by linarith
  have hx : 0 < 1+b*φ := by positivity
  have hb2 : b^2 ≤ 1 := pow_le_one₀ hb0.le hb1.le
  have hprod : b^2*φ ≤ 1 := by nlinarith [mul_nonneg (sub_nonneg.mpr hb2) hφ0]
  have hH : (1+b*φ)^2 ≤ 1+2*φ+2*b*φ-b^2*φ^2 := by
    nlinarith only [mul_nonneg hφ0 (sub_nonneg.mpr hprod)]
  have hH3 : 1+2*φ+2*b*φ-b^2*φ^2 ≤ 3*(1+b*φ)^2 := by
    nlinarith [mul_nonneg hb0.le hφ0, sq_nonneg (b*φ)]
  let Q := (1+2*φ+2*b*φ-b^2*φ^2)/(1+b*φ)^2
  have hQ1 : 1 ≤ Q := (le_div_iff₀ (sq_pos_of_pos hx)).mpr (by simpa using hH)
  have hQ3 : Q ≤ 3 := (div_le_iff₀ (sq_pos_of_pos hx)).mpr hH3
  have hK : K b φ = b^2*(1-b)/(2*(1+b)^3)*Q := by
    unfold K Q
    field_simp
  have hbase : b^2*(1-b)/16 ≤ b^2*(1-b)/(2*(1+b)^3) := by
    apply div_le_div_of_nonneg_left (by positivity) (by positivity)
    have hc : (1+b)^3 ≤ (2:ℝ)^3 := by gcongr; linarith
    linarith
  have hlow : b^2*(1-b)/16 ≤ K b φ := by
    rw [hK]
    calc b^2*(1-b)/16 ≤ b^2*(1-b)/(2*(1+b)^3) := hbase
      _ ≤ b^2*(1-b)/(2*(1+b)^3)*Q := by nlinarith [mul_nonneg (by positivity : 0 ≤ b^2*(1-b)/(2*(1+b)^3)) (sub_nonneg.mpr hQ1)]
  refine ⟨by unfold eta; positivity, lt_of_lt_of_le (by positivity) hlow, hlow, ?_⟩
  have hid : K b φ*(eta b)^2 = (1-b)^5/(128*(1+b)^3)*Q := by
    rw [hK]; unfold eta
    field_simp
    ring
  rw [hid]
  have hsmall : (1-b)^5/(128*(1+b)^3) ≤ 1/128 := by
    rw [div_le_iff₀ (by positivity : 0 < 128*(1+b)^3)]
    have h1 : (1-b)^5 ≤ (1:ℝ)^5 := by gcongr; linarith
    have h2 : (1:ℝ)^3 ≤ (1+b)^3 := by gcongr; linarith
    norm_num at h1 h2 ⊢
    linarith
  calc (1-b)^5/(128*(1+b)^3)*Q ≤ (1/128:ℝ)*3 := by
        exact mul_le_mul hsmall hQ3 (by linarith) (by norm_num)
    _ = 3/128 := by norm_num

/-- Remainder (14) gives strict positivity even at the amplitude boundary. -/
theorem amplitude_sign (δ η κ G : ℝ) (hδ : 0 < δ) (hη : 0 < η) (hκ : 0 < κ)
    (hsmall : δ^2 ≤ κ*η^4/(22+κ*η^2))
    (htail : |G-κ*δ^2| ≤ 11*(δ/η)^4/(1-(δ/η)^2)) :
    δ < η ∧ κ*δ^2/2 ≤ G ∧ 0 < G := by
  have hden : 0 < 22+κ*η^2 := by positivity
  have hs := (le_div_iff₀ hden).mp hsmall
  have hsq : δ^2 < η^2 := by
    by_contra! hh
    have hp := mul_le_mul_of_nonneg_left hh (by positivity : 0 ≤ κ*η^2)
    nlinarith [sq_pos_of_pos hδ]
  have hde : δ < η := by nlinarith
  have hdiff : 0 < η^2-δ^2 := sub_pos.mpr hsq
  have hE : 0 < η^2 := sq_pos_of_pos hη
  have hid : 11*(δ/η)^4/(1-(δ/η)^2) = 11*δ^4/(η^2*(η^2-δ^2)) := by
    field_simp
  have hrem : 11*δ^4/(η^2*(η^2-δ^2)) ≤ κ*δ^2/2 := by
    rw [div_le_iff₀ (mul_pos hE hdiff)]
    have hm := mul_le_mul_of_nonneg_right hs (sq_nonneg δ)
    nlinarith only [hm]
  rw [hid] at htail
  have hlow : κ*δ^2/2 ≤ G := by linarith [(abs_le.mp htail).1]
  exact ⟨hde,hlow,lt_of_lt_of_le (by positivity) hlow⟩

/-- Equation (15), including the conservative constant 23. -/
theorem amplitude_radius_lower (b φ : ℝ) (hb0 : 0 < b) (hb1 : b < 1)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    (1-b)^9/(23*65536) ≤ K b φ*(eta b)^4/(22+K b φ*(eta b)^2) := by
  obtain ⟨hη,hK,hlo,hup⟩ := amplitude_constants b φ hb0 hb1 hφ0 hφ1
  have ht : 0 < 1-b := by linarith
  have hden : 0 < 22+K b φ*(eta b)^2 := by positivity
  have hid : (b^2*(1-b)/16)*(eta b)^4 = (1-b)^9/(65536*b^2) := by
    unfold eta
    field_simp
    ring
  have hnum : (1-b)^9/(65536*b^2) ≤ K b φ*(eta b)^4 := by
    rw [← hid]
    exact mul_le_mul_of_nonneg_right hlo (by positivity)
  have hb2 : b^2 ≤ 1 := pow_le_one₀ hb0.le hb1.le
  have hnum2 : (1-b)^9/65536 ≤ (1-b)^9/(65536*b^2) := by
    apply div_le_div_of_nonneg_left (by positivity) (by positivity)
    nlinarith
  have hnum3 := hnum2.trans hnum
  apply (le_div_iff₀ hden).mpr
  have hm := mul_le_mul_of_nonneg_left (show 22+K b φ*(eta b)^2 ≤ 23 by linarith)
    (by positivity : 0 ≤ (1-b)^9/(23*65536))
  nlinarith only [hm,hnum3]

/-- The squared physical condition implies the threshold amplitude condition
whenever the threshold normalization has `1-b ≥ s/2`. -/
theorem physical_amplitude_bound (δ s b φ : ℝ) (hs0 : 0 < s)
    (hb0 : 0 < b) (hb1 : b < 1) (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1)
    (ht : s/2 ≤ 1-b) (hsmall : δ^2 ≤ s^9/2^30) :
    δ^2 ≤ K b φ*(eta b)^4/(22+K b φ*(eta b)^2) := by
  have hp : (s/2)^9 ≤ (1-b)^9 := by gcongr
  have hc : s^9/2^30 ≤ (s/2)^9/(23*65536) := by
    have hs : 0 ≤ s^9 := by positivity
    nlinarith only [hs]
  have hm : (s/2)^9/(23*65536) ≤ (1-b)^9/(23*65536) := by gcongr
  exact hsmall.trans (hc.trans (hm.trans (amplitude_radius_lower b φ hb0 hb1 hφ0 hφ1)))



/-- The P4 gap bound implies the simple upper bound on the normalized threshold.
The P4 inequality itself is a probability-model input. -/
theorem q_upper_of_gap (δ s c : ℝ) (hs0 : 0 < s) (hs1 : s < 1)
    (hc : 0 < c) (hδ : δ^2 ≤ 1/4)
    (hgap : Real.log (c/(1-s)) < -Real.log (1-δ^2)/2) :
    c/s^2 < 2*(1-s)/s^2 := by
  have ht : 0 < 1-s := sub_pos.mpr hs1
  have hd : 0 < 1-δ^2 := by linarith
  have hlog : Real.log (1/4:ℝ) < Real.log (1-δ^2) :=
    Real.log_lt_log (by norm_num) (by linarith)
  have hid : Real.log (1/4:ℝ) = -2*Real.log 2 := by
    have h4 : (4:ℝ) = 2^2 := by norm_num
    rw [one_div,Real.log_inv,h4,Real.log_pow]
    ring
  rw [hid] at hlog
  have hclog : Real.log (c/(1-s)) < Real.log 2 := by linarith
  have hc2 : c/(1-s) < 2 := (Real.log_lt_log_iff (div_pos hc ht) (by norm_num)).mp hclog
  have hc2' : c < 2*(1-s) := (div_lt_iff₀ ht).mp hc2
  exact div_lt_div_of_pos_right hc2' (sq_pos_of_pos hs0)

/-- Monotonicity of q(b) converts the threshold upper bound into `1-b ≥ s/2`. -/
theorem stay_lower (s b : ℝ) (hs0 : 0 < s) (hs1 : s < 1)
    (hb0 : 0 < b) (hb1 : b < 1) (hq : qOfB b < 2*(1-s)/s^2) :
    s/2 ≤ 1-b := by
  have hmid0 : 0 < 1-s/2 := by linarith
  have hmid1 : 1-s/2 < 1 := by linarith
  have hid : qOfB (1-s/2) = (4-2*s)/s^2 := by
    unfold qOfB
    field_simp [hs0.ne']
    ring
  have hcomp : 2*(1-s)/s^2 < qOfB (1-s/2) := by
    rw [hid]
    apply div_lt_div_of_pos_right _ (sq_pos_of_pos hs0)
    linarith
  have hbm : b < 1-s/2 := by
    by_contra! hh
    rcases eq_or_lt_of_le hh with he | he
    · rw [← he] at hq
      linarith
    · have hm := normalization_strictMono ⟨hmid0,hmid1⟩ ⟨hb0,hb1⟩ he
      linarith
  linarith

/-- The exact amplitude criterion, conditional on real-threshold inputs and
the amplitude Cauchy remainder with quadratic coefficient K. -/
theorem amplitude_derivative_neg (δ φ s b R' : ℝ) (c_star R_real : ℝ → ℝ)
    (hδ0 : 0 < δ) (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1)
    (hs0 : 0 < s) (hs1 : s < 1) (hb0 : 0 < b) (hb1 : b < 1)
    (hsmall : δ^2 ≤ K b φ*(eta b)^4/(22+K b φ*(eta b)^2))
    (hc_pos : 0 < c_star s) (hc_cont : ContinuousAt c_star s)
    (hR_sderiv : HasStrictDerivAt R_real R' (c_star s/s^2))
    (hV_pos : 0 < (c_star s/s^2)*R')
    (h_thresh : ∀ᶠ u in 𝓝 s, Real.log u + R_real (c_star u/u^2)=0)
    (htail : |Gfun (R_real (c_star s/s^2)) ((c_star s/s^2)*R')-K b φ*δ^2| ≤
      11*(δ/eta b)^4/(1-(δ/eta b)^2)) :
    deriv (fun u => Real.log (c_star u/(1-u))) s < 0 := by
  obtain ⟨hη,hK,_,_⟩ := amplitude_constants b φ hb0 hb1 hφ0 hφ1
  have hg := (amplitude_sign δ (eta b) (K b φ) _ hδ0 hη hK hsmall htail).2.2
  obtain ⟨_,_,_,hD,hsign⟩ := derivative_identity s R' c_star R_real hs0 hs1
    hc_pos hc_cont hR_sderiv hV_pos h_thresh
  rw [hD.deriv]
  exact hsign.mpr hg

/-- Squared form of `δ ≤ s^(9/2)/32768`. Seven analytic inputs: the five
real-threshold inputs, P4's upper gap bound, and the amplitude Cauchy remainder.
Neither the final sign nor a finite parameter grid is among the assumptions. -/
theorem physical_derivative_neg (δ φ s b R' : ℝ) (c_star R_real : ℝ → ℝ)
    (hδ0 : 0 < δ) (hφ0 : 0 ≤ φ) (hφ1 : φ < 1)
    (hs0 : 0 < s) (hs1 : s < 1) (hb0 : 0 < b) (hb1 : b < 1)
    (hbq : qOfB b = c_star s/s^2) (hsmall : δ^2 ≤ s^9/2^30)
    (hc_pos : 0 < c_star s) (hc_cont : ContinuousAt c_star s)
    (hR_sderiv : HasStrictDerivAt R_real R' (c_star s/s^2))
    (hV_pos : 0 < (c_star s/s^2)*R')
    (h_thresh : ∀ᶠ u in 𝓝 s, Real.log u + R_real (c_star u/u^2)=0)
    (hgap : Real.log (c_star s/(1-s)) < -Real.log (1-δ^2)/2)
    (htail : |Gfun (R_real (c_star s/s^2)) ((c_star s/s^2)*R')-K b φ*δ^2| ≤
      11*(δ/eta b)^4/(1-(δ/eta b)^2)) :
    deriv (fun u => Real.log (c_star u/(1-u))) s < 0 := by
  have hδ : δ^2 ≤ 1/4 := by
    have hp : s^9 ≤ 1 := pow_le_one₀ hs0.le hs1.le
    nlinarith only [hp,hsmall]
  have hq := q_upper_of_gap δ s (c_star s) hs0 hs1 hc_pos hδ hgap
  rw [← hbq] at hq
  have ht := stay_lower s b hs0 hs1 hb0 hb1 hq
  have ha := physical_amplitude_bound δ s b φ hs0 hb0 hb1 hφ0 hφ1.le ht hsmall
  exact amplitude_derivative_neg δ φ s b R' c_star R_real hδ0 hφ0 hφ1.le
    hs0 hs1 hb0 hb1 ha hc_pos hc_cont hR_sderiv hV_pos h_thresh htail


end Dobson.GapOneExtended

#print axioms Dobson.GapOneExtended.odd_coefficient_nonpos

#print axioms Dobson.GapOneExtended.odd_coefficient_sums

#print axioms Dobson.GapOneExtended.coefficient_bounds

#print axioms Dobson.GapOneExtended.partialSum_lower

#print axioms Dobson.GapOneExtended.lowerMargin_endpoint

#print axioms Dobson.GapOneExtended.lowerMargin_lower

#print axioms Dobson.GapOneExtended.sign_of_remainder

#print axioms Dobson.GapOneExtended.normalization_derivative

#print axioms Dobson.GapOneExtended.normalization_strictMono

#print axioms Dobson.GapOneExtended.endpoint_conversion

#print axioms Dobson.GapOneExtended.normalization_exists

#print axioms Dobson.GapOneExtended.q_le_of_chord

#print axioms Dobson.GapOneExtended.b_le_of_q

#print axioms Dobson.GapOneExtended.extended_derivative_neg

#print axioms Dobson.GapOneExtended.complex_disk_constants

#print axioms Dobson.GapOneExtended.amplitude_disk

#print axioms Dobson.GapOneExtended.amplitude_constants

#print axioms Dobson.GapOneExtended.amplitude_sign

#print axioms Dobson.GapOneExtended.amplitude_radius_lower

#print axioms Dobson.GapOneExtended.physical_amplitude_bound

#print axioms Dobson.GapOneExtended.q_upper_of_gap

#print axioms Dobson.GapOneExtended.stay_lower

#print axioms Dobson.GapOneExtended.amplitude_derivative_neg

#print axioms Dobson.GapOneExtended.physical_derivative_neg
