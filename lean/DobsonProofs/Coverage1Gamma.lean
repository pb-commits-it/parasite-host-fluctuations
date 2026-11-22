import Mathlib
import DobsonProofs.Band
import DobsonProofs.GainCondition

/-!
# Coverage batch 1 — Item 7: One fluctuating transfer in a geometric ring (`prop:ring-gamma`)

Formalises the analytic core of `prop:ring-gamma` (`sections/03-ring.tex`, lines 26–33, and
`appendices/A-ring.tex`, lines 28–45):

1. Each geometric stay on `{1, 2, ...}` with staying probability `m_k = 1 − 1 / τ_k` (`τ_k ≥ 1`)
   has probability generating function series
   `∑_{ℓ = 0}^∞ (1 − m_k) m_k^ℓ φ^{ℓ + 1} = φ / (τ_k − (τ_k − 1) φ)`,
   which equals `0` at `φ = 0` and `1 / (1 + τ_k / θ)` (`θ = φ / (1 − φ)`) for `0 < φ < 1`.
2. The full-circuit return transform `R(φ, τ) = ∏_{k = 1}^n E[φ^{L_k}]` satisfies
   `R(0, τ) = 0`, `R(φ, τ) = ∏_k (1 + τ_k / θ)⁻¹` for `0 < φ < 1`, and `0 ≤ R(φ, τ) < 1`.
3. Under the explicit Markov-renewal generating-function hypothesis `(H_renewal)`
   `∑_{ℓ = 0}^∞ φ^{ℓ + 1} u(ℓ + 1) = R(φ, τ) / (1 − R(φ, τ))` on the reuse probabilities
   `u(ℓ + 1)` with use frequency `c = 1 / 𝒯` (`𝒯 = ∑_k τ_k`), the renewal per-step cost
   coefficient `C = c² / 2 − c ∑_{ℓ ≥ 1} φ^ℓ (u(ℓ) − c)` equals
   `(1 + 2 θ) / (2 𝒯²) − 𝒯⁻¹ R / (1 − R)` across the entire range `0 ≤ φ < 1`, and the
   per-generation loss `Γ = 𝒯 C σ²` equals `σ² [ (1 + 2 θ) / (2 𝒯) − R / (1 − R) ]`.
4. `C` and `Γ` are invariant under any permutation `π : Equiv.Perm (Fin n)` of stage stays and
   independent of which transfer `k₀ : Fin n` fluctuates.
5. Under the explicit fourth-order stochastic Lyapunov remainder hypothesis `(H_rem)`
   `|log λ − r(σ) − σ² ∑_j C_j| ≤ K σ⁴` for independent `AR(1)` hosts on `|σ| ≤ σ₀`, the
   per-generation loss satisfies `|𝒯 (log λ − r(σ)) − ∑_j Γ_j(σ)| ≤ 𝒯 K σ⁴`.
-/

/-! ## Coverage correction — 2026-11-22

`prop:ring-gamma` remains partial-P1. The independent hypothesis audit
(2026-11-16) identifies the supplied stochastic expansion and quartic remainder.
The credited renewal HasSum identity linking reuse probabilities to the circuit
product also remains supplied. Stage permutations and transfer-index independence
are properties of the displayed coefficient; they do not identify the physical
reference-chain return law. -/

namespace Dobson.Coverage1Gamma

open scoped BigOperators

/-- Memory scale `θ(φ) = φ / (1 − φ)`. -/
noncomputable def thetaOfPhi (φ : ℝ) : ℝ :=
  φ / (1 - φ)

/-- Single-stage geometric stay probability generating function `E[φ^{L_k}]` for mean stay `τ ≥ 1`
(`m = 1 − 1 / τ`), written smoothly across `0 ≤ φ < 1` as `φ / (τ − (τ − 1) φ)`. -/
noncomputable def stayPGF (φ τ : ℝ) : ℝ :=
  φ / (τ - (τ - 1) * φ)

/-- Circuit return transform `R(φ, τ) = ∏_{k : Fin n} stayPGF φ (τ k)`. -/
noncomputable def circuitR (n : ℕ) (φ : ℝ) (τ : Fin n → ℝ) : ℝ :=
  ∏ k : Fin n, stayPGF φ (τ k)

/-- Lineage generation time `𝒯 = ∑_{k : Fin n} τ k`. -/
noncomputable def genTime (n : ℕ) (τ : Fin n → ℝ) : ℝ :=
  ∑ k : Fin n, τ k

/-- Closed-form per-step cost coefficient `C(φ, τ) = (1 + 2 θ)/(2 𝒯²) − 𝒯⁻¹ R/(1 − R)`
(`0 ≤ φ < 1`). -/
noncomputable def stepCostCoeff (n : ℕ) (φ : ℝ) (τ : Fin n → ℝ) : ℝ :=
  let θ := thetaOfPhi φ
  let T := genTime n τ
  let R := circuitR n φ τ
  (1 + 2 * θ) / (2 * T ^ 2) - R / (T * (1 - R))

/-- Closed-form per-generation second-order loss `Γ(σ², φ, τ) = σ² [(1 + 2 θ)/(2 𝒯) − R/(1 − R)]`
(`0 ≤ φ < 1`). -/
noncomputable def ringGamma (n : ℕ) (v φ : ℝ) (τ : Fin n → ℝ) : ℝ :=
  let θ := thetaOfPhi φ
  let T := genTime n τ
  let R := circuitR n φ τ
  v * ((1 + 2 * θ) / (2 * T) - R / (1 - R))

/-- **Part 1a (geometric stay PGF series).**
For `τ ≥ 1` (`m = 1 − 1 / τ ∈ [0, 1)`) and `|φ| < 1`, the geometric stay law `(1 − m) m^ℓ` on
`ℓ + 1 ∈ {1, 2, ...}` sums to `stayPGF φ τ = φ / (τ − (τ − 1) φ)`. -/
theorem hasSum_geometric_stayPGF (φ τ : ℝ) (hτ : 1 ≤ τ) (hφ : |φ| < 1) :
    HasSum (fun ℓ : ℕ => (1 / τ) * (1 - 1 / τ) ^ ℓ * φ ^ (ℓ + 1)) (stayPGF φ τ) := by
  have hτ_pos : 0 < τ := lt_of_lt_of_le zero_lt_one hτ
  have hm_nn : 0 ≤ 1 - 1 / τ := by
    rw [sub_nonneg, div_le_one hτ_pos]
    exact hτ
  have hm_le : 1 - 1 / τ ≤ 1 := sub_le_self 1 (by positivity)
  have hnorm : ‖(1 - 1 / τ) * φ‖ < 1 := by
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg hm_nn]
    calc (1 - 1 / τ) * |φ| ≤ 1 * |φ| :=
          mul_le_mul_of_nonneg_right hm_le (abs_nonneg φ)
      _ = |φ| := one_mul _
      _ < 1 := hφ
  have hgeom : HasSum (fun ℓ : ℕ => ((1 - 1 / τ) * φ) ^ ℓ)
      (1 - (1 - 1 / τ) * φ)⁻¹ := hasSum_geometric_of_norm_lt_one hnorm
  have hmul := hgeom.mul_left ((1 / τ) * φ)
  have hterm : (fun ℓ : ℕ => (1 / τ) * φ * ((1 - 1 / τ) * φ) ^ ℓ) =
      (fun ℓ : ℕ => (1 / τ) * (1 - 1 / τ) ^ ℓ * φ ^ (ℓ + 1)) := by
    ext ℓ
    rw [mul_pow, pow_succ]
    ring
  rw [hterm] at hmul
  have hden_ne : τ - (τ - 1) * φ ≠ 0 := by
    have h1 : 0 < 1 - (1 - 1 / τ) * φ := by
      have habs := abs_lt.mp hnorm
      linarith
    have heq : τ * (1 - (1 - 1 / τ) * φ) = τ - (τ - 1) * φ := by
      field_simp [hτ_pos.ne']
    intro h0
    have : τ * (1 - (1 - 1 / τ) * φ) = 0 := by rw [heq, h0]
    nlinarith
  have hval : (1 / τ) * φ * (1 - (1 - 1 / τ) * φ)⁻¹ = stayPGF φ τ := by
    unfold stayPGF
    have h1ne : 1 - (1 - 1 / τ) * φ ≠ 0 := by
      intro h0
      apply hden_ne
      have heq : τ - (τ - 1) * φ = τ * (1 - (1 - 1 / τ) * φ) := by
        field_simp [hτ_pos.ne']
      rw [heq, h0, mul_zero]
    rw [← div_eq_mul_inv]
    field_simp [hτ_pos.ne', h1ne, hden_ne]
  rwa [hval] at hmul

/-- **Part 1b (stage PGF bounds and θ-form).**
For `0 ≤ φ < 1` and `τ > 0` with `τ ≥ 1` (or more generally `τ > 0` and `0 ≤ φ < 1`):
- at `φ = 0`, `stayPGF 0 τ = 0`;
- for `0 < φ < 1`, `stayPGF φ τ = 1 / (1 + τ / θ(φ))` and `0 < stayPGF φ τ < 1`. -/
theorem stayPGF_zero_and_pos (φ τ : ℝ) (hτ : 0 < τ) :
    stayPGF 0 τ = 0 ∧
      (0 < φ → φ < 1 →
        stayPGF φ τ = 1 / (1 + τ / thetaOfPhi φ) ∧
          0 < stayPGF φ τ ∧ stayPGF φ τ < 1) := by
  refine ⟨by simp [stayPGF], fun hφ0 hφ1 => ?_⟩
  have h1φ : 0 < 1 - φ := sub_pos.mpr hφ1
  have hθ : 0 < thetaOfPhi φ := div_pos hφ0 h1φ
  have hden : τ - (τ - 1) * φ = τ * (1 - φ) + φ := by ring
  have hden_pos : 0 < τ - (τ - 1) * φ := by rw [hden]; positivity
  refine ⟨?_, div_pos hφ0 hden_pos, ?_⟩
  · unfold stayPGF thetaOfPhi
    have hsum_pos : 0 < 1 + τ / (φ / (1 - φ)) := add_pos one_pos (div_pos hτ hθ)
    rw [div_eq_div_iff hden_pos.ne' hsum_pos.ne']
    field_simp [hφ0.ne', h1φ.ne']
    ring
  · unfold stayPGF
    rw [div_lt_one hden_pos]
    nlinarith

/-- **Part 2 (circuit return transform `R`: zero at `φ = 0`, `Band.R` at `φ > 0`, and `0 ≤ R < 1`).** -/
theorem circuitR_properties (n : ℕ) (hn : 1 ≤ n) (φ : ℝ) (hφ0 : 0 ≤ φ) (hφ1 : φ < 1)
    (τ : Fin n → ℝ) (hτ : ∀ k, 0 < τ k) :
    (φ = 0 → circuitR n φ τ = 0) ∧
      (0 < φ → circuitR n φ τ = Dobson.Band.R (thetaOfPhi φ) Finset.univ τ) ∧
      0 ≤ circuitR n φ τ ∧ circuitR n φ τ < 1 := by
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro h0
    subst h0
    unfold circuitR
    obtain ⟨k₀, hk₀⟩ := hne
    exact Finset.prod_eq_zero hk₀ (by simp [stayPGF])
  · intro hpos
    unfold circuitR Dobson.Band.R Dobson.Band.P
    rw [one_div, ← Finset.prod_inv_distrib]
    refine Finset.prod_congr rfl fun k _ => ?_
    rw [(stayPGF_zero_and_pos φ (τ k) (hτ k)).2 hpos hφ1 |>.1, one_div]
  · rcases hφ0.eq_or_lt with rfl | hpos
    · obtain ⟨k₀, hk₀⟩ := hne
      unfold circuitR
      rw [Finset.prod_eq_zero hk₀ (by simp [stayPGF])]
    · unfold circuitR
      exact Finset.prod_nonneg fun k _ =>
        ((stayPGF_zero_and_pos φ (τ k) (hτ k)).2 hpos hφ1).2.1.le
  · rcases hφ0.eq_or_lt with rfl | hpos
    · obtain ⟨k₀, hk₀⟩ := hne
      unfold circuitR
      rw [Finset.prod_eq_zero hk₀ (by simp [stayPGF])]
      exact zero_lt_one
    · have hθ : 0 < thetaOfPhi φ := div_pos hpos (sub_pos.mpr hφ1)
      have hB2 := Dobson.Band.R_pos_and_R_lt_one_and_ratio_le (thetaOfPhi φ) hθ
        Finset.univ hne τ (fun k _ => hτ k)
      have heq : circuitR n φ τ = Dobson.Band.R (thetaOfPhi φ) Finset.univ τ := by
        unfold circuitR Dobson.Band.R Dobson.Band.P
        rw [one_div, ← Finset.prod_inv_distrib]
        exact Finset.prod_congr rfl fun k _ => by
          rw [(stayPGF_zero_and_pos φ (τ k) (hτ k)).2 hpos hφ1 |>.1, one_div]
      rw [heq]
      exact hB2.2.1

/-- **Part 3 (derivation of `C` and `Γ` from the renewal lag sum across `0 ≤ φ < 1`).**
Given reuse probabilities `u : ℕ → ℝ` satisfying the explicit renewal generating-function
hypothesis `(H_renewal)` `HasSum (fun m => φ^(m+1) * u (m+1)) (R / (1 − R))` with use frequency
`c = 1 / 𝒯`, the lag-sum cost `c²/2 − c ∑_{m ≥ 0} φ^{m+1} (u(m+1) − c)` converges to
`stepCostCoeff n φ τ`, and `𝒯 C σ² = ringGamma n σ² φ τ`. -/
theorem renewal_lag_sum_to_ringGamma (n : ℕ) (hn : 1 ≤ n) (v φ : ℝ)
    (hφ0 : 0 ≤ φ) (hφ1 : φ < 1) (τ : Fin n → ℝ) (hτ : ∀ k, 0 < τ k)
    (u : ℕ → ℝ)
    (h_renewal : HasSum (fun m : ℕ => φ ^ (m + 1) * u (m + 1))
      (circuitR n φ τ / (1 - circuitR n φ τ))) :
    HasSum (fun m : ℕ => φ ^ (m + 1) * (u (m + 1) - 1 / genTime n τ))
        (circuitR n φ τ / (1 - circuitR n φ τ) - thetaOfPhi φ / genTime n τ) ∧
      ((1 / genTime n τ) ^ 2 / 2 -
          (1 / genTime n τ) *
            (circuitR n φ τ / (1 - circuitR n φ τ) - thetaOfPhi φ / genTime n τ) =
        stepCostCoeff n φ τ) ∧
      genTime n τ * stepCostCoeff n φ τ * v = ringGamma n v φ τ := by
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  have hT : 0 < genTime n τ := Finset.sum_pos (fun k _ => hτ k) hne
  have h1φ : 0 < 1 - φ := sub_pos.mpr hφ1
  have hnormφ : ‖φ‖ < 1 := by
    rw [Real.norm_eq_abs, abs_of_nonneg hφ0]
    exact hφ1
  have hgeomφ : HasSum (fun m : ℕ => φ ^ (m + 1)) (thetaOfPhi φ) := by
    have hg := (hasSum_geometric_of_norm_lt_one hnormφ).mul_left φ
    have hfun : (fun m : ℕ => φ * φ ^ m) = (fun m : ℕ => φ ^ (m + 1)) := by
      ext m; rw [pow_succ, mul_comm]
    rw [hfun] at hg
    unfold thetaOfPhi
    rwa [div_eq_mul_inv]
  have hconst := hgeomφ.mul_right (1 / genTime n τ)
  have hsub := h_renewal.sub hconst
  have hfun2 : (fun m : ℕ => φ ^ (m + 1) * u (m + 1) - φ ^ (m + 1) * (1 / genTime n τ)) =
      (fun m : ℕ => φ ^ (m + 1) * (u (m + 1) - 1 / genTime n τ)) := by
    ext m; ring
  rw [hfun2] at hsub
  have hval : thetaOfPhi φ * (1 / genTime n τ) = thetaOfPhi φ / genTime n τ := by ring
  rw [hval] at hsub
  have hRlt1 : circuitR n φ τ < 1 :=
    (circuitR_properties n hn φ hφ0 hφ1 τ hτ).2.2.2
  have h1R : 1 - circuitR n φ τ ≠ 0 := (sub_pos.mpr hRlt1).ne'
  refine ⟨hsub, ?_, ?_⟩
  · unfold stepCostCoeff
    field_simp [hT.ne', h1R]
    ring
  · unfold stepCostCoeff ringGamma
    field_simp [hT.ne', h1R]

/-- **Agreement with `Dobson.Band.Gamma` for `0 < φ < 1`, and `φ = 0` evaluation.** -/
theorem ringGamma_zero_and_band_eq (n : ℕ) (hn : 1 ≤ n) (v φ : ℝ)
    (τ : Fin n → ℝ) (hτ : ∀ k, 0 < τ k) :
    ringGamma n v 0 τ = v / (2 * genTime n τ) ∧
      (0 < φ → φ < 1 →
        ringGamma n v φ τ = Dobson.Band.Gamma v (thetaOfPhi φ) Finset.univ τ) := by
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  have hT : 0 < genTime n τ := Finset.sum_pos (fun k _ => hτ k) hne
  refine ⟨?_, fun hφ0 hφ1 => ?_⟩
  · have hR0 : circuitR n 0 τ = 0 :=
      (circuitR_properties n hn 0 le_rfl zero_lt_one τ hτ).1 rfl
    unfold ringGamma thetaOfPhi
    rw [hR0]
    simp only [zero_div, mul_zero, add_zero, sub_zero]
    ring
  · have hReq : circuitR n φ τ = Dobson.Band.R (thetaOfPhi φ) Finset.univ τ :=
      (circuitR_properties n hn φ hφ0.le hφ1 τ hτ).2.1 hφ0
    unfold ringGamma Dobson.Band.Gamma genTime Dobson.Band.total
    rw [hReq]

/-- **Part 4 (invariance under stage permutation and choice of fluctuating transfer).**
Permuting `(τ_1, ..., τ_n)` by any bijection `π : Equiv.Perm (Fin n)` leaves `genTime`, `circuitR`,
`stepCostCoeff`, and `ringGamma` unchanged; likewise the formula has no dependence on the transfer
index `k₀ : Fin n`. -/
theorem ringGamma_perm_and_transfer_invariant (n : ℕ) (v φ : ℝ) (τ : Fin n → ℝ)
    (π : Equiv.Perm (Fin n)) (k₁ k₂ : Fin n) :
    genTime n (τ ∘ π) = genTime n τ ∧
      circuitR n φ (τ ∘ π) = circuitR n φ τ ∧
      stepCostCoeff n φ (τ ∘ π) = stepCostCoeff n φ τ ∧
      ringGamma n v φ (τ ∘ π) = ringGamma n v φ τ ∧
      (fun _ : Fin n => ringGamma n v φ τ) k₁ = (fun _ : Fin n => ringGamma n v φ τ) k₂ := by
  have hT : genTime n (τ ∘ π) = genTime n τ := by
    unfold genTime
    exact Equiv.sum_comp π τ
  have hR : circuitR n φ (τ ∘ π) = circuitR n φ τ := by
    unfold circuitR
    exact Equiv.prod_comp π (fun x => stayPGF φ (τ x))
  refine ⟨hT, hR, ?_, ?_, rfl⟩
  · unfold stepCostCoeff
    simp only [hT, hR]
  · unfold ringGamma
    simp only [hT, hR]

/-- **Part 5 (independent-host additivity and per-generation fourth-order remainder).**
Given finitely many independent `AR(1)` hosts `j ∈ J` with variances `(σ · η j)²`, memories
`0 ≤ φ j < 1`, and explicit fourth-order stochastic remainder hypothesis `(H_rem)`
`|log λ − r(σ) − σ² ∑_{j ∈ J} η_j² C(φ_j, τ)| ≤ K σ⁴` on `|σ| ≤ σ₀`, the per-generation loss
`𝒯 (log λ − r(σ))` equals `∑_{j ∈ J} Γ_j(σ)` up to `𝒯 K σ⁴`. -/
theorem independent_hosts_generation_remainder {ι : Type*} (J : Finset ι)
    (n : ℕ) (hn : 1 ≤ n) (τ : Fin n → ℝ) (hτ : ∀ k, 0 < τ k)
    (η φ : ι → ℝ) (hφ0 : ∀ j ∈ J, 0 ≤ φ j) (hφ1 : ∀ j ∈ J, φ j < 1)
    (logLam : ℝ) (r : ℝ → ℝ) (σ₀ K : ℝ)
    (h_rem : ∀ σ : ℝ, |σ| ≤ σ₀ →
      |logLam - r σ - σ ^ 2 * ∑ j ∈ J, η j ^ 2 * stepCostCoeff n (φ j) τ| ≤ K * σ ^ 4) :
    ∀ σ : ℝ, |σ| ≤ σ₀ →
      |genTime n τ * (logLam - r σ) -
          ∑ j ∈ J, ringGamma n ((σ * η j) ^ 2) (φ j) τ| ≤
        genTime n τ * K * σ ^ 4 := by
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  have hT : 0 < genTime n τ := Finset.sum_pos (fun k _ => hτ k) hne
  intro σ hσ
  have hsum_eq : genTime n τ * (σ ^ 2 * ∑ j ∈ J, η j ^ 2 * stepCostCoeff n (φ j) τ) =
      ∑ j ∈ J, ringGamma n ((σ * η j) ^ 2) (φ j) τ := by
    rw [Finset.mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j hj => ?_
    have hRlt1 : circuitR n (φ j) τ < 1 :=
      (circuitR_properties n hn (φ j) (hφ0 j hj) (hφ1 j hj) τ hτ).2.2.2
    have h1R : 1 - circuitR n (φ j) τ ≠ 0 := (sub_pos.mpr hRlt1).ne'
    unfold stepCostCoeff ringGamma
    field_simp [hT.ne', h1R]
  have hdiff : genTime n τ * (logLam - r σ) -
      ∑ j ∈ J, ringGamma n ((σ * η j) ^ 2) (φ j) τ =
      genTime n τ * (logLam - r σ - σ ^ 2 * ∑ j ∈ J, η j ^ 2 * stepCostCoeff n (φ j) τ) := by
    linarith [hsum_eq]
  rw [hdiff, abs_mul, abs_of_pos hT, mul_assoc]
  exact mul_le_mul_of_nonneg_left (h_rem σ hσ) hT.le

end Dobson.Coverage1Gamma

#print axioms Dobson.Coverage1Gamma.hasSum_geometric_stayPGF
#print axioms Dobson.Coverage1Gamma.stayPGF_zero_and_pos
#print axioms Dobson.Coverage1Gamma.circuitR_properties
#print axioms Dobson.Coverage1Gamma.renewal_lag_sum_to_ringGamma
#print axioms Dobson.Coverage1Gamma.ringGamma_zero_and_band_eq
#print axioms Dobson.Coverage1Gamma.ringGamma_perm_and_transfer_invariant
#print axioms Dobson.Coverage1Gamma.independent_hosts_generation_remainder
