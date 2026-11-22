import Mathlib.Analysis.Polynomial.Basic
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic

/-!
# One slow L stage: exact polynomial alternatives

Core of `prop:dyn-k6`. The characteristic polynomial is defined literally as
`(X + C a) * D - C a * N`. The denominator is monic, its value at zero is
positive, and the numerator degree is at most its degree. These are the
algebraic consequences of the manuscript's internally Hurwitz block models;
their derivation from the block Jacobians is not supplied here. No slow-root
existence or continuity of the other roots is assumed or claimed.
-/

open Polynomial Set Filter
open scoped Topology

namespace Dobson.Coverage6Dynamics

noncomputable def characteristic (D N : ℝ[X]) (a : ℝ) : ℝ[X] :=
  (X + C a) * D - C a * N

/-- Exact evaluation, including coincident zeros of the unreduced factors. -/
theorem characteristic_eval (D N : ℝ[X]) (a z : ℝ) :
    (characteristic D N a).eval z = (z + a) * D.eval z - a * N.eval z := by
  simp [characteristic]

/-- Equal static gain gives an exact zero root for every rate. -/
theorem zero_root (D N : ℝ[X]) (a : ℝ) (hD : D.eval 0 ≠ 0)
    (hQ : N.eval 0 / D.eval 0 = 1) :
    (characteristic D N a).IsRoot 0 := by
  have hN : N.eval 0 = D.eval 0 := (div_eq_one_iff_eq hD).mp hQ
  simp [IsRoot, characteristic_eval, hN]

/-- The polynomial retains its positive leading coefficient and degree. -/
theorem characteristic_monic (D N : ℝ[X]) (a : ℝ) (hD : D.Monic)
    (hdeg : N.natDegree ≤ D.natDegree) :
    (characteristic D N a).Monic ∧
      (characteristic D N a).natDegree = D.natDegree + 1 := by
  have hP : ((X + C a) * D).Monic := (monic_X_add_C a).mul hD
  have hdP : ((X + C a) * D).natDegree = D.natDegree + 1 := by
    rw [(monic_X_add_C a).natDegree_mul hD, natDegree_X_add_C]
    omega
  have hdN : (C a * N).natDegree < ((X + C a) * D).natDegree := by
    have h := natDegree_C_mul_le a N
    omega
  have hd : (C a * N).degree < ((X + C a) * D).degree :=
    degree_lt_degree hdN
  exact ⟨hP.sub_of_left hd, (natDegree_sub_eq_left_of_natDegree_lt hdN).trans hdP⟩

/-- Static gain above one gives a positive real root for every positive rate,
using the intermediate value theorem on the actual characteristic polynomial. -/
theorem positive_root_all_rates (D N : ℝ[X]) (a : ℝ) (ha : 0 < a)
    (hD : D.Monic) (hdeg : N.natDegree ≤ D.natDegree)
    (hD0 : 0 < D.eval 0) (hQ : 1 < N.eval 0 / D.eval 0) :
    ∃ z : ℝ, 0 < z ∧ (characteristic D N a).IsRoot z := by
  let P := characteristic D N a
  obtain ⟨hmon, hnd⟩ := characteristic_monic D N a hD hdeg
  have hdegree : 0 < P.degree := by
    rw [degree_eq_natDegree hmon.ne_zero]
    exact_mod_cast (show 0 < P.natDegree by dsimp [P]; omega)
  have hlim : Tendsto (fun x => P.eval x) atTop atTop :=
    P.tendsto_atTop_of_leadingCoeff_nonneg hdegree (by rw [hmon]; norm_num)
  have hneg : P.eval 0 < 0 := by
    dsimp [P]
    rw [characteristic_eval]
    have h := (lt_div_iff₀ hD0).mp hQ
    nlinarith [mul_pos ha (sub_pos.mpr (show D.eval 0 < N.eval 0 by linarith))]
  obtain ⟨b, hb, hp⟩ := ((eventually_gt_atTop (0 : ℝ)).and
    (hlim.eventually_gt_atTop 0)).exists
  obtain ⟨z, hz, he⟩ := intermediate_value_Icc hb.le P.continuous.continuousOn
    (show (0 : ℝ) ∈ Icc (P.eval 0) (P.eval b) from ⟨hneg.le, hp.le⟩)
  refine ⟨z, lt_of_le_of_ne hz.1 ?_, he⟩
  intro heq
  have : P.eval 0 = 0 := by simpa [← heq] using he
  linarith

/-- The slow-root coefficient with a quantitative second-order error.
`hroot` identifies an actual root, and `hLip` is the local Lipschitz control
on the rational response. Existence of such a root near zero is not asserted.
Unlike assuming an asymptotic expansion, these inputs imply its error bound. -/
theorem slow_root_error (Q : ℝ → ℝ) (a z K : ℝ) (hK : 0 ≤ K)
    (hroot : z = a * (Q z - 1))
    (hLip : |Q z - Q 0| ≤ K * |z|) (hsmall : |a| * K ≤ 1 / 2) :
    |z - a * (Q 0 - 1)| ≤ 2 * K * |Q 0 - 1| * a ^ 2 := by
  have htri : |Q z - 1| ≤ |Q z - Q 0| + |Q 0 - 1| := by
    calc
      |Q z - 1| = |(Q z - Q 0) + (Q 0 - 1)| := by congr 1; ring
      _ ≤ |Q z - Q 0| + |Q 0 - 1| := abs_add_le _ _
  have hz : |z| ≤ |a| * (K * |z| + |Q 0 - 1|) := by
    calc
      |z| = |a| * |Q z - 1| := by conv_lhs => rw [hroot]; rw [abs_mul]
      _ ≤ |a| * (K * |z| + |Q 0 - 1|) := by
        gcongr
        exact htri.trans (add_le_add hLip le_rfl)
  have hzbound : |z| ≤ 2 * |a| * |Q 0 - 1| := by
    have hh := mul_le_mul_of_nonneg_right hsmall (abs_nonneg z)
    nlinarith
  have heq : z - a * (Q 0 - 1) = a * (Q z - Q 0) := by
    conv_lhs => lhs; rw [hroot]
    ring
  rw [heq, abs_mul]
  calc
    |a| * |Q z - Q 0| ≤ |a| * (K * |z|) := by gcongr
    _ ≤ |a| * (K * (2 * |a| * |Q 0 - 1|)) := by gcongr
    _ = 2 * K * |Q 0 - 1| * |a| ^ 2 := by ring
    _ = 2 * K * |Q 0 - 1| * a ^ 2 := by rw [sq_abs]

end Dobson.Coverage6Dynamics
