import DobsonProofs.Full1_DelayStability

/-! The lower stability boundary as first-order host self-regulation tends to
zero. The boundary is an actual root of the actual three-state characteristic
Routh margin, and stability below it is proved. -/
namespace Dobson.Full4.SelfRegulation
open Filter
open scoped Topology
open Dobson.Full1.DelayStability Dobson.Full1.BlockQualification

def j (d q h δ : ℝ) := d-h-q+δ
def c (d q h δ : ℝ) := (d-h)^2-q*d+2*(d-h)*δ+δ^2
def k (d h δ : ℝ) := (d-h+δ)*d*δ
def discriminant (d q h δ : ℝ) := (c d q h δ)^2-4*j d q h δ*k d h δ
noncomputable def lowerBoundary (d q h δ : ℝ) :=
  2*k d h δ / (-c d q h δ + Real.sqrt (discriminant d q h δ))

theorem margin_expansion (d q h δ a : ℝ) :
    margin d q h (-h+δ) a = j d q h δ*a^2+c d q h δ*a+k d h δ := by
  unfold margin A B J j c k
  ring

/-- Rationalized smaller root, including a vanishing quadratic coefficient. -/
theorem smaller_root (j c k : ℝ) (hc : c < 0) (hk : 0 < k)
    (hdisc : 0 < c^2-4*j*k) :
    let r := 2*k/(-c+Real.sqrt (c^2-4*j*k))
    0 < r ∧ j*r^2+c*r+k = 0 ∧ 2*j*r+c = -Real.sqrt (c^2-4*j*k) ∧
      (∀ a : ℝ, 0 ≤ a → a < r → 0 < j*a^2+c*a+k) := by
  dsimp
  let t := Real.sqrt (c^2-4*j*k)
  have ht : 0 < t := Real.sqrt_pos.mpr hdisc
  have ht2 : t^2 = c^2-4*j*k := Real.sq_sqrt hdisc.le
  have hden : 0 < -c+t := by linarith
  have hr : 0 < 2*k/(-c+t) := div_pos (by positivity) hden
  have hder : 2*j*(2*k/(-c+t))+c = -t := by
    field_simp
    nlinarith [ht2]
  have hroot : j*(2*k/(-c+t))^2+c*(2*k/(-c+t))+k = 0 := by
    field_simp
    nlinarith [ht2]
  refine ⟨hr,hroot,hder,fun a ha har => ?_⟩
  let r := 2*k/(-c+t)
  change 0 < j*a^2+c*a+k
  have he : j*a^2+c*a+k = (r-a)*(t+j*(r-a)) := by
    change 2*j*r+c = -t at hder
    change j*r^2+c*r+k = 0 at hroot
    nlinarith [hroot,hder]
  have hpos : 0 < t+j*(r-a) := by
    by_cases hj : 0 ≤ j
    · have : 0 ≤ j*(r-a) := mul_nonneg hj (sub_pos.mpr har).le
      linarith
    · have hja : j*a ≤ 0 := mul_nonpos_of_nonpos_of_nonneg (le_of_not_ge hj) ha
      have hkr : 0 < k/r := div_pos hk hr
      have he2 : t+j*r = k/r := by
        apply (eq_div_iff hr.ne').mpr
        change 2*j*r+c = -t at hder
        change j*r^2+c*r+k = 0 at hroot
        nlinarith [hroot,hder]
      rw [← he2] at hkr
      nlinarith
  rw [he]
  exact mul_pos (sub_pos.mpr har) hpos

/-- The exact cubic criterion needs no positivity assumption on J: positive
leading and constant coefficients make the Routh margin decisive. -/
theorem stability_iff_margin (d q h δ a : ℝ)
    (h2 : 0 < a+A d (-h+δ)) (h0 : 0 < a*C d q h (-h+δ)) :
    Hurwitz (characteristic d q h (-h+δ) a) ↔ 0 < margin d q h (-h+δ) a := by
  rw [characteristic_cubic, cubic_hurwitz_iff]
  have hid := margin_identity d q h (-h+δ) a
  constructor
  · rintro ⟨_,_,_,hm⟩
    linarith
  · intro hm
    have h1 : 0 < B d h (-h+δ)+a*J d q (-h+δ) := by
      by_contra hn
      have := mul_nonpos_of_nonneg_of_nonpos h2.le (le_of_not_gt hn)
      linarith
    exact ⟨h2,h1,h0,by linarith⟩

/-- This is an actual limit of the boundary, not just a formal balance of
polynomial coefficients. The positive side is the manuscript's δ down to zero. -/
theorem boundary_ratio_limit (d q h : ℝ) (hgap : (d-h)^2 < q*d) :
    Tendsto (fun δ : ℝ => lowerBoundary d q h δ / δ) (𝓝[>] 0)
      (𝓝 ((d-h)*d/(q*d-(d-h)^2))) := by
  have hpos : 0 < q*d-(d-h)^2 := sub_pos.mpr hgap
  have hsqrt : Real.sqrt (((d-h)^2-q*d)^2) = q*d-(d-h)^2 := by
    rw [Real.sqrt_sq_eq_abs, abs_of_neg (sub_neg.mpr hgap)]
    ring
  have hden : -c d q h 0 + Real.sqrt (discriminant d q h 0) ≠ 0 := by
    simp only [c, discriminant, k, mul_zero, zero_add, add_zero, sub_zero, zero_pow,
      OfNat.ofNat_ne_zero, ne_eq, not_false_eq_true, mul_one]
    rw [hsqrt]
    linarith
  have hc : ContinuousAt (fun δ : ℝ =>
      2*(d-h+δ)*d/(-c d q h δ+Real.sqrt (discriminant d q h δ))) 0 := by
    apply ContinuousAt.div
    · fun_prop
    · simp only [discriminant, c, j, k]; fun_prop
    · exact hden
  have heq : ∀ᶠ δ : ℝ in 𝓝[>] 0, lowerBoundary d q h δ/δ =
      2*(d-h+δ)*d/(-c d q h δ+Real.sqrt (discriminant d q h δ)) := by
    filter_upwards [self_mem_nhdsWithin (s := Set.Ioi (0 : ℝ))] with δ hδ
    unfold lowerBoundary k
    have hd0 : δ ≠ 0 := ne_of_gt hδ
    field_simp [hd0]
  have ht := hc.tendsto.mono_left (show 𝓝[>] (0 : ℝ) ≤ 𝓝 (0 : ℝ) from nhdsWithin_le_nhds)
  have hv : 2*(d-h+0)*d/(-c d q h 0+Real.sqrt (discriminant d q h 0)) =
      (d-h)*d/(q*d-(d-h)^2) := by
    simp [c, discriminant, k, hsqrt]
    field_simp
    ring
  rw [hv] at ht
  exact ht.congr' (heq.mono (fun _ hx => hx.symm))

/-- A lower boundary means strict stability at every positive smaller rate,
loss of strict stability at the boundary, and strict instability just above. -/
def IsLowerBoundary (d q h δ r : ℝ) : Prop :=
  0 < r ∧ margin d q h (-h+δ) r = 0 ∧
    (∀ a : ℝ, 0 < a → a < r → Hurwitz (characteristic d q h (-h+δ) a)) ∧
    (¬ Hurwitz (characteristic d q h (-h+δ) r)) ∧
    (∃ η : ℝ, 0 < η ∧ ∀ a : ℝ, r < a → a < r+η →
      ∃ z : ℂ, characteristic d q h (-h+δ) a z = 0 ∧ 0 < z.re)

theorem local_boundary (d q h δ : ℝ) (hc : c d q h δ < 0)
    (hk : 0 < k d h δ) (hdisc : 0 < discriminant d q h δ)
    (hA : 0 < A d (-h+δ)) (hC : 0 < C d q h (-h+δ)) :
    IsLowerBoundary d q h δ (lowerBoundary d q h δ) := by
  let r := lowerBoundary d q h δ
  let t := Real.sqrt (discriminant d q h δ)
  have ht : 0 < t := Real.sqrt_pos.mpr hdisc
  obtain ⟨hr,heq,hder,hbefore⟩ := smaller_root (j d q h δ) (c d q h δ)
    (k d h δ) hc hk hdisc
  change 0 < r at hr
  change j d q h δ*r^2+c d q h δ*r+k d h δ = 0 at heq
  change 2*j d q h δ*r+c d q h δ = -t at hder
  have hmargin : margin d q h (-h+δ) r = 0 := by rw [margin_expansion]; exact heq
  have h2 : 0 < r+A d (-h+δ) := add_pos hr hA
  have h0 : 0 < r*C d q h (-h+δ) := mul_pos hr hC
  have h1 : 0 < B d h (-h+δ)+r*J d q (-h+δ) := by
    have hi := margin_identity d q h (-h+δ) r
    by_contra hn
    have := mul_nonpos_of_nonneg_of_nonpos h2.le (le_of_not_gt hn)
    linarith
  refine ⟨hr,hmargin,?_,?_,?_⟩
  · intro a ha har
    apply (stability_iff_margin d q h δ a (add_pos ha hA) (mul_pos ha hC)).mpr
    rw [margin_expansion]
    exact hbefore a ha.le har
  · rw [stability_iff_margin d q h δ r h2 h0, hmargin]
    exact lt_irrefl 0
  · have hcont : Continuous (fun a : ℝ => B d h (-h+δ)+a*J d q (-h+δ)) := by fun_prop
    obtain ⟨ε,hε,hnear⟩ := Metric.eventually_nhds_iff.mp
      (hcont.continuousAt.eventually (lt_mem_nhds h1))
    let η := min ε (t/(|j d q h δ|+1))
    have hη : 0 < η := lt_min hε (div_pos ht (by positivity))
    refine ⟨η,hη,fun a har haη => ?_⟩
    have hdist : dist a r < ε := by
      rw [Real.dist_eq, abs_of_pos (sub_pos.mpr har)]
      have := min_le_left ε (t/(|j d q h δ|+1))
      dsimp [η] at haη
      linarith
    have h1a := hnear hdist
    have ha : 0 < a := hr.trans har
    have hx : a-r < t/(|j d q h δ|+1) := by
      have := min_le_right ε (t/(|j d q h δ|+1))
      dsimp [η] at haη
      linarith
    have hx' := (lt_div_iff₀ (by positivity : 0 < |j d q h δ|+1)).mp hx
    have hjx : j d q h δ*(a-r) < t := by
      have hx0 : 0 < a-r := sub_pos.mpr har
      have := mul_le_mul_of_nonneg_right (le_abs_self (j d q h δ)) hx0.le
      nlinarith
    have hm : margin d q h (-h+δ) a < 0 := by
      rw [margin_expansion]
      have hid : j d q h δ*a^2+c d q h δ*a+k d h δ =
          (a-r)*(j d q h δ*(a-r)-t) := by nlinarith [heq,hder]
      rw [hid]
      exact mul_neg_of_pos_of_neg (sub_pos.mpr har) (sub_neg.mpr hjx)
    have hi := margin_identity d q h (-h+δ) a
    obtain ⟨z,hz,hzpos⟩ := cubic_unstable_of_margin (a+A d (-h+δ))
      (B d h (-h+δ)+a*J d q (-h+δ)) (a*C d q h (-h+δ))
      (add_pos ha hA) h1a (mul_pos ha hC) (by linarith)
    exact ⟨z,by rw [characteristic_cubic]; exact hz,hzpos⟩

/-- Full small-self-regulation clause of dyn-k5, including identification as
an actual stability boundary and the positive asymptotic constant. -/
theorem small_self_regulation (d q h : ℝ) (hh : 0 < h) (hq : 0 < q)
    (hdh : h < d) (hgap : (d-h)^2 < q*d) :
    (∀ᶠ δ : ℝ in 𝓝[>] 0, IsLowerBoundary d q h δ (lowerBoundary d q h δ)) ∧
    0 < (d-h)*d/(q*d-(d-h)^2) ∧
    Tendsto (fun δ : ℝ => lowerBoundary d q h δ/δ) (𝓝[>] 0)
      (𝓝 ((d-h)*d/(q*d-(d-h)^2))) := by
  have hd : 0 < d := hh.trans hdh
  have hc0 : c d q h 0 < 0 := by simp [c]; linarith
  have hdisc0 : 0 < discriminant d q h 0 := by
    simpa [discriminant,k] using sq_pos_of_neg hc0
  have hA0 : 0 < A d (-h+0) := by unfold A; linarith
  have hC0 : 0 < C d q h (-h+0) := by
    have he : C d q h (-h+0) = q*h := by unfold C; ring
    rw [he]; exact mul_pos hq hh
  have hcc : Continuous (fun δ : ℝ => c d q h δ) := by unfold c; fun_prop
  have hdc : Continuous (fun δ : ℝ => discriminant d q h δ) := by
    simp only [discriminant,c,j,k]; fun_prop
  have hAc : Continuous (fun δ : ℝ => A d (-h+δ)) := by unfold A; fun_prop
  have hCc : Continuous (fun δ : ℝ => C d q h (-h+δ)) := by unfold C; fun_prop
  refine ⟨?_,div_pos (mul_pos (sub_pos.mpr hdh) hd) (sub_pos.mpr hgap),
    boundary_ratio_limit d q h hgap⟩
  filter_upwards [self_mem_nhdsWithin (s := Set.Ioi (0 : ℝ)),
    (hcc.continuousAt.eventually (gt_mem_nhds hc0)).filter_mono nhdsWithin_le_nhds,
    (hdc.continuousAt.eventually (lt_mem_nhds hdisc0)).filter_mono nhdsWithin_le_nhds,
    (hAc.continuousAt.eventually (lt_mem_nhds hA0)).filter_mono nhdsWithin_le_nhds,
    (hCc.continuousAt.eventually (lt_mem_nhds hC0)).filter_mono nhdsWithin_le_nhds]
    with δ hδ hc hdisc hA hC
  apply local_boundary d q h δ hc _ hdisc hA hC
  unfold k
  exact mul_pos (mul_pos (add_pos (sub_pos.mpr hdh) hδ) hd) hδ

end Dobson.Full4.SelfRegulation
