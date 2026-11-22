import DobsonProofs.Full4_Cascade
import DobsonProofs.Full4_TwoSlow
import DobsonProofs.Full4_SelfRegulation

/-! Stability regimes, cascade instability and self-regulation asymptotics
for the defined host/L matrices. Coverage correction, 2026-11-22:
`prop:dyn-k5` is core. The host block is entered explicitly; its derivation
from the nonlinear model remains outside credit. -/
namespace Dobson.Full4.UnregulatedHost
open Filter
open scoped Topology
open Dobson.Full1.DelayStability Dobson.Full1.BlockQualification

/-- All conclusions for one V/M host with h>0, rho=-h and positive coupling q.
A=d-h and J=A-q are expanded to make the domain and boundary cases explicit. -/
def ManuscriptK5 (d q h : ℝ) : Prop :=
  (A d (-h) = d-h ∧ J d q (-h) = d-h-q ∧ B d h (-h) = 0 ∧ C d q h (-h) = q*h) ∧
  (∀ a : ℝ, 0 < a →
    characteristic d q h (-h) a = cubic (a+d-h) (a*(d-h-q)) (a*q*h) ∧
    (Hurwitz (characteristic d q h (-h) a) ↔
      0 < d-h-q ∧ q*h < (a+d-h)*(d-h-q))) ∧
  (0 < d-h-q → q*h ≤ (d-h)*(d-h-q) →
    ∀ a : ℝ, 0 < a → Hurwitz (characteristic d q h (-h) a)) ∧
  (0 < d-h-q → (d-h)*(d-h-q) < q*h →
    0 < q*h/(d-h-q)-(d-h) ∧
      ∀ a : ℝ, 0 < a → (Hurwitz (characteristic d q h (-h) a) ↔
        q*h/(d-h-q)-(d-h) < a)) ∧
  (d-h-q ≤ 0 → ∀ a : ℝ, 0 < a → ¬ Hurwitz (characteristic d q h (-h) a)) ∧
  (∀ n : ℕ, 1 ≤ n → ∀ a : Fin n → ℝ, (∀ i, 0 < a i) → d-h-q ≤ 0 →
    ∃ z : ℂ, z ∈ spectrum ℂ (Cascade.jacobian d q h a) ∧ 0 < z.re) ∧
  (h < d → ∃ ε : ℝ, 0 < ε ∧ ∀ a : ℝ, 0 < a → a < ε →
    ∃ z : ℂ, z ∈ spectrum ℂ (Cascade.jacobian d q h (fun _ : Fin 2 => a)) ∧ 0 < z.re) ∧
  (h < d → (d-h)^2 < q*d →
    (∀ᶠ δ : ℝ in 𝓝[>] 0,
      SelfRegulation.IsLowerBoundary d q h δ (SelfRegulation.lowerBoundary d q h δ)) ∧
    0 < (d-h)*d/(q*d-(d-h)^2) ∧
    Tendsto (fun δ : ℝ => SelfRegulation.lowerBoundary d q h δ/δ) (𝓝[>] 0)
      (𝓝 ((d-h)*d/(q*d-(d-h)^2))))

theorem host_conclusions (d q h : ℝ) (hq : 0 < q) (hh : 0 < h) :
    ManuscriptK5 d q h := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_,SelfRegulation.small_self_regulation d q h hh hq⟩
  · unfold A J B C
    constructor
    · ring
    constructor
    · ring
    constructor <;> ring
  · intro a ha
    constructor
    · rw [characteristic_cubic]
      funext z
      simp only [cubic, A, B, C, J]
      push_cast
      ring
    · simpa [add_sub_assoc] using unregulated_single_delay_iff a d q h ha hq hh
  · intro hJ hslack a ha
    apply (unregulated_single_delay_iff a d q h ha hq hh).mpr
    exact ⟨hJ,by nlinarith [mul_pos ha hJ]⟩
  · intro hJ hslack
    constructor
    · exact sub_pos.mpr ((lt_div_iff₀ hJ).mpr hslack)
    · intro a ha
      rw [unregulated_single_delay_iff a d q h ha hq hh, and_iff_right hJ]
      exact Dobson.Coverage2Dynamics.rate_threshold a (d-h) (d-h-q) q h hJ
  · intro hJ a ha hs
    exact (not_lt_of_ge hJ) ((unregulated_single_delay_iff a d q h ha hq hh).mp hs).1
  · intro n hn a ha hJ
    exact Cascade.cascade_unstable hn d q h a hq hh ha hJ
  · intro hdh
    obtain ⟨ε,hε,hu⟩ := TwoSlow.two_slow_unstable (d-h) q h (sub_pos.mpr hdh) hq hh
    refine ⟨ε,hε,fun a ha hae => ?_⟩
    obtain ⟨z,hz,hpos⟩ := hu a ha hae
    refine ⟨z,?_,hpos⟩
    rw [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot,
      Cascade.characteristic_polynomial, Cascade.eval_polynomial]
    simpa [Fin.prod_univ_two, pow_two, mul_assoc, mul_left_comm, mul_comm] using hz

/-- The M substitution and epsilon>=2 corollary are part of the manuscript,
not optional examples. The identities hold before restricting epsilon. -/
def MConclusions : Prop :=
  (∀ ξ ε h : ℝ,
    (ξ+2*ε*h)-h-(ξ+ε*h) = (ε-1)*h ∧
    ((ξ+2*ε*h)-h)*((ξ+2*ε*h)-h-(ξ+ε*h))-(ξ+ε*h)*h =
      h*((ε-2)*ξ+(2*(ε-1)^2-1)*h)) ∧
  (∀ ξ ε h a : ℝ, 0 ≤ ξ → 2 ≤ ε → 0 < h → 0 < a →
    Hurwitz (characteristic (ξ+2*ε*h) (ξ+ε*h) h (-h) a))

theorem m_conclusions : MConclusions := by
  constructor
  · intro ξ ε h
    constructor <;> ring
  · intro ξ ε h a hξ hε hh ha
    have hq : 0 < ξ+ε*h :=
      add_pos_of_nonneg_of_pos hξ (mul_pos (by linarith) hh)
    rw [unregulated_single_delay_iff a (ξ+2*ε*h) (ξ+ε*h) h ha hq hh]
    have hid : ξ+2*ε*h-h-(ξ+ε*h) = (ε-1)*h := by ring
    rw [hid]
    exact Dobson.Coverage2Dynamics.m_block_all_rates ξ ε h a hξ hε hh ha

/-- The complete proposition, including all one-delay cases, M specialization,
both strict cascade-instability clauses, and the actual lower-boundary
asymptotic. No textbook probability hypotheses are needed. -/
theorem prop_dyn_k5 :
    (∀ d q h : ℝ, 0 < q → 0 < h → ManuscriptK5 d q h) ∧ MConclusions :=
  ⟨host_conclusions,m_conclusions⟩

#print axioms prop_dyn_k5
end Dobson.Full4.UnregulatedHost
