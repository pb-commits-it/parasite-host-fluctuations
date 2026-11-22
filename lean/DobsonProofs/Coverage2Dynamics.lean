import DobsonProofs.OneHostOneDelay

/-!
# Coverage batch 2: host without self-regulation
Core of `prop:dyn-k5`: cubic coefficients, exact Routh inequalities, stability
sufficiency, M-block identity and epsilon >= 2. Necessity of Hurwitz stability,
finite cascades and the singular root asymptotic are not claimed.
-/
namespace Dobson.Coverage2Dynamics

/-- Expanding the one-host/one-delay characteristic expression at rho = -h. -/
theorem cubic_identity (z a d q h : ℂ) :
    z * (z + (d - h)) * (z + a) - q * a * (z - h) =
      z ^ 3 + (a + (d - h)) * z ^ 2 + a * (d - h - q) * z + a * q * h := by
  ring

/-- All three strict Routh inequalities reduce to the two manuscript inequalities. -/
theorem routh_reduction (a A J q h : ℝ) (ha : 0 < a) (hq : 0 < q) (hh : 0 < h) :
    (0 < a + A ∧ 0 < a * J ∧ a * q * h < (a + A) * (a * J)) ↔
      (0 < J ∧ q * h < (a + A) * J) := by
  have hqh : 0 < q * h := mul_pos hq hh
  constructor
  · rintro ⟨_, hJ, hRH⟩
    refine ⟨(mul_pos_iff_of_pos_left ha).mp hJ, ?_⟩
    nlinarith
  · rintro ⟨hJ, hRH⟩
    have hA : 0 < a + A := by nlinarith
    refine ⟨hA, mul_pos ha hJ, ?_⟩
    nlinarith [mul_pos ha (sub_pos.mpr hRH)]

/-- Proves the Hurwitz direction without assuming a cubic stability criterion. -/
theorem cubic_stable (a A J q h : ℝ) (ha : 0 < a) (hq : 0 < q) (hh : 0 < h)
    (hJ : 0 < J) (hRH : q * h < (a + A) * J) (z : ℂ)
    (hz : z ^ 3 + ((a + A : ℝ) : ℂ) * z ^ 2 +
      ((a * J : ℝ) : ℂ) * z + ((a * q * h : ℝ) : ℂ) = 0) : z.re < 0 := by
  obtain ⟨h2, h1, hr⟩ := (routh_reduction a A J q h ha hq hh).mpr ⟨hJ, hRH⟩
  apply Dobson.OneHostOneDelay.D3 1 (a + A) (a * J) (a * q * h)
    (by norm_num) h2 h1 (by positivity) (by simpa using hr) z
  simpa using hz

/-- The critical-rate inequality has no asymptotic approximation. -/
theorem rate_threshold (a A J q h : ℝ) (hJ : 0 < J) :
    q * h < (a + A) * J ↔ q * h / J - A < a := by
  rw [sub_lt_iff_lt_add, div_lt_iff₀ hJ]

/-- Substitute the M-block parameters. -/
theorem m_block_identity (ξ ε h : ℝ) :
    (ξ + 2 * ε * h - h) - (ξ + ε * h) = (ε - 1) * h ∧
    (ξ + 2 * ε * h - h) * ((ε - 1) * h) - (ξ + ε * h) * h =
      h * ((ε - 2) * ξ + (2 * (ε - 1) ^ 2 - 1) * h) := by
  constructor <;> ring

/-- The M-block has strictly positive Routh slack for every positive delay rate. -/
theorem m_block_all_rates (ξ ε h a : ℝ)
    (hξ : 0 ≤ ξ) (hε : 2 ≤ ε) (hh : 0 < h) (ha : 0 < a) :
    0 < (ε - 1) * h ∧
    (ξ + ε * h) * h < (a + (ξ + 2 * ε * h - h)) * ((ε - 1) * h) := by
  have he : 0 < ε - 1 := by linarith
  have hJ := mul_pos he hh
  have hs : 0 < 2 * (ε - 1) ^ 2 - 1 := by nlinarith
  have hx : 0 ≤ (ε - 2) * ξ := mul_nonneg (by linarith) hξ
  have hid := (m_block_identity ξ ε h).2
  have hp : 0 < h * ((ε - 2) * ξ + (2 * (ε - 1) ^ 2 - 1) * h) := by positivity
  exact ⟨hJ, by nlinarith [mul_pos ha hJ]⟩

end Dobson.Coverage2Dynamics
