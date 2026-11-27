/-!
# Discrete energy identity of the lossless periodic Yee scheme

This file proves, in core Lean 4 (no Mathlib), that the leapfrog Yee update
conserves a discrete energy on a periodic grid, in 1D and in 2D TMz.

## Assumptions (all explicit)

* The field values lie in an arbitrary commutative ring `R`
  (`Lean.Grind.CommRing R`). The proof is exact algebra. It says nothing about
  floating-point arithmetic, and it does not cover the f64 CPU or CUDA code of
  photon-fdtd.
* Uniform, lossless medium: one update coefficient per axis (`c` in 1D;
  `cx`, `cy` in 2D), the same at every cell. In natural units with
  eps = mu = 1 these are the Courant numbers dt/dx and dt/dy. No sources, no
  conductivity, no CPML, no material contrast.
* Periodic boundaries: the fields are functions `Int → R` (1D) or
  `Int → Int → R` (2D) with period `n` (`nx`, `ny`) in each index.

## The quantity

The state is `E` at time step `n` and `H` at step `n - 1/2`. One step is
`H⁺ = H + c D⁺E` and then `E' = E + c D⁻H⁺`. The energy is

  W(E, H) = Σ E_i² + Σ H_i H⁺_i

over one period. The theorem says W(E', H⁺) = W(E, H). This W is the
standard discrete energy of the leapfrog scheme. It is a positive quadratic
form only under the CFL condition, which this file does not prove or use: the
identity holds for every coefficient.

## Proof idea

The change of the energy density at a cell is a difference of a flux at two
neighbouring cells (`density_step` and `density_step_2d`, checked by `grind`).
A sum of such differences over one period telescopes to zero.
-/

namespace YeeEnergy

variable {R : Type} [Lean.Grind.CommRing R]

/-- `sumN n f = f 0 + f 1 + ... + f (n - 1)`. -/
def sumN : Nat → (Nat → R) → R
  | 0, _ => 0
  | k + 1, f => sumN k f + f k

theorem sumN_sub (n : Nat) (f g : Nat → R) :
    sumN n (fun k => f k - g k) = sumN n f - sumN n g := by
  induction n with
  | zero => simp only [sumN]; grind
  | succ k ih => simp only [sumN, ih]; grind

theorem sumN_add (n : Nat) (f g : Nat → R) :
    sumN n (fun k => f k + g k) = sumN n f + sumN n g := by
  induction n with
  | zero => simp only [sumN]; grind
  | succ k ih => simp only [sumN, ih]; grind

theorem sumN_congr (n : Nat) (f g : Nat → R) (h : ∀ k, f k = g k) :
    sumN n f = sumN n g := by
  induction n with
  | zero => rfl
  | succ k ih => simp only [sumN, ih, h]

theorem sumN_zero (n : Nat) : sumN n (fun _ => (0 : R)) = 0 := by
  induction n with
  | zero => rfl
  | succ k ih => simp only [sumN, ih]; grind

/-- A sum of neighbour differences telescopes. -/
theorem sumN_telescope (n : Nat) (G : Int → R) :
    sumN n (fun k => G ((k : Int) + 1) - G (k : Int)) = G (n : Int) - G 0 := by
  induction n with
  | zero => simp only [sumN, Int.natCast_zero]; grind
  | succ k ih =>
    simp only [sumN, ih]
    have : ((k + 1 : Nat) : Int) = (k : Int) + 1 := by omega
    rw [this]; grind

/-- `f` has period `n`. -/
def Periodic (n : Nat) (f : Int → R) : Prop := ∀ i, f (i + n) = f i

theorem telescope_periodic (n : Nat) (G : Int → R) (hG : Periodic n G) :
    sumN n (fun k => G ((k : Int) + 1) - G (k : Int)) = 0 := by
  rw [sumN_telescope]
  have := hG 0
  rw [Int.zero_add] at this
  rw [this]; grind

/-! ## 1D -/

/-- H at step n + 1/2: `H⁺_i = H_i + c (E_{i+1} - E_i)`. -/
def hNext (c : R) (e h : Int → R) : Int → R := fun i => h i + c * (e (i + 1) - e i)

/-- E at step n + 1: `E'_i = E_i + c (H⁺_i - H⁺_{i-1})`. -/
def eNext (c : R) (e h : Int → R) : Int → R :=
  fun i => e i + c * (hNext c e h i - hNext c e h (i - 1))

/-- Energy density `E_i² + H_i H⁺_i`. -/
def density (c : R) (e h : Int → R) (i : Int) : R := e i * e i + h i * hNext c e h i

/-- Energy over one period of length `n`. -/
def energy (n : Nat) (c : R) (e h : Int → R) : R := sumN n (fun k => density c e h k)

/-- The flux of the 1D identity: `G_i = H⁺_{i-1} (2 c E_i + c (E'_i - E_i))`. -/
def flux (c : R) (e h : Int → R) (i : Int) : R :=
  hNext c e h (i - 1) * (2 * c * e i + c * (c * (hNext c e h i - hNext c e h (i - 1))))

/-- Pointwise: the change of the density is a flux difference. -/
theorem density_step (c : R) (e h : Int → R) (i : Int) :
    density c (eNext c e h) (hNext c e h) i - density c e h i
      = flux c e h (i + 1) - flux c e h i := by
  simp only [density, eNext, hNext, flux, Int.add_sub_cancel]
  grind

theorem hNext_periodic (n : Nat) (c : R) (e h : Int → R)
    (pe : Periodic n e) (ph : Periodic n h) : Periodic n (hNext c e h) := by
  intro i
  simp only [hNext]
  rw [show i + (n : Int) + 1 = (i + 1) + n by omega, pe, pe, ph]

theorem flux_periodic (n : Nat) (c : R) (e h : Int → R)
    (pe : Periodic n e) (ph : Periodic n h) : Periodic n (flux c e h) := by
  intro i
  have hp := hNext_periodic n c e h pe ph
  simp only [flux]
  rw [show i + (n : Int) - 1 = (i - 1) + n by omega, hp, hp, pe]

/-- **Energy identity, 1D.** One Yee step conserves the discrete energy on a
periodic grid, for every coefficient `c`. -/
theorem energy_conserved (n : Nat) (c : R) (e h : Int → R)
    (pe : Periodic n e) (ph : Periodic n h) :
    energy n c (eNext c e h) (hNext c e h) = energy n c e h := by
  have step : energy n c (eNext c e h) (hNext c e h) - energy n c e h = 0 := by
    simp only [energy]
    rw [← sumN_sub]
    rw [sumN_congr n _ (fun k => flux c e h ((k : Int) + 1) - flux c e h (k : Int))
      (fun k => density_step c e h k)]
    exact telescope_periodic n (flux c e h) (flux_periodic n c e h pe ph)
  grind

/-! ## 2D TMz -/

/-- `Hx` at step n + 1/2: `Hx⁺ = Hx - cy (E_{i,j+1} - E_{i,j})`. -/
def hxNext (cy : R) (e hx : Int → Int → R) : Int → Int → R :=
  fun i j => hx i j - cy * (e i (j + 1) - e i j)

/-- `Hy` at step n + 1/2: `Hy⁺ = Hy + cx (E_{i+1,j} - E_{i,j})`. -/
def hyNext (cx : R) (e hy : Int → Int → R) : Int → Int → R :=
  fun i j => hy i j + cx * (e (i + 1) j - e i j)

/-- The E increment: `cx (Hy⁺_{i,j} - Hy⁺_{i-1,j}) - cy (Hx⁺_{i,j} - Hx⁺_{i,j-1})`. -/
def incr (cx cy : R) (e hx hy : Int → Int → R) (i j : Int) : R :=
  cx * (hyNext cx e hy i j - hyNext cx e hy (i - 1) j)
    - cy * (hxNext cy e hx i j - hxNext cy e hx i (j - 1))

/-- `Ez` at step n + 1. -/
def eNext2 (cx cy : R) (e hx hy : Int → Int → R) : Int → Int → R :=
  fun i j => e i j + incr cx cy e hx hy i j

/-- Energy density `E² + Hx Hx⁺ + Hy Hy⁺`. -/
def density2 (cx cy : R) (e hx hy : Int → Int → R) (i j : Int) : R :=
  e i j * e i j + hx i j * hxNext cy e hx i j + hy i j * hyNext cx e hy i j

/-- Energy over one `nx` by `ny` period. -/
def energy2 (nx ny : Nat) (cx cy : R) (e hx hy : Int → Int → R) : R :=
  sumN ny (fun l => sumN nx (fun k => density2 cx cy e hx hy k l))

/-- x flux: `G = Hy⁺_{i-1,j} (2 cx E + cx s)`. -/
def fluxX (cx cy : R) (e hx hy : Int → Int → R) (i j : Int) : R :=
  hyNext cx e hy (i - 1) j * (2 * cx * e i j + cx * incr cx cy e hx hy i j)

/-- y flux: `K = -Hx⁺_{i,j-1} (2 cy E + cy s)`. -/
def fluxY (cx cy : R) (e hx hy : Int → Int → R) (i j : Int) : R :=
  -(hxNext cy e hx i (j - 1) * (2 * cy * e i j + cy * incr cx cy e hx hy i j))

/-- Pointwise: the change of the density is a sum of an x and a y flux difference. -/
theorem density_step_2d (cx cy : R) (e hx hy : Int → Int → R) (i j : Int) :
    density2 cx cy (eNext2 cx cy e hx hy) (hxNext cy e hx) (hyNext cx e hy) i j
        - density2 cx cy e hx hy i j
      = (fluxX cx cy e hx hy (i + 1) j - fluxX cx cy e hx hy i j)
        + (fluxY cx cy e hx hy i (j + 1) - fluxY cx cy e hx hy i j) := by
  simp only [density2, eNext2, incr, hxNext, hyNext, fluxX, fluxY, Int.add_sub_cancel]
  grind

/-- Period `nx` in the first index and `ny` in the second. -/
def Periodic2 (nx ny : Nat) (f : Int → Int → R) : Prop :=
  (∀ i j, f (i + nx) j = f i j) ∧ (∀ i j, f i (j + ny) = f i j)

theorem fields_periodic (nx ny : Nat) (cx cy : R) (e hx hy : Int → Int → R)
    (pe : Periodic2 nx ny e) (px : Periodic2 nx ny hx) (py : Periodic2 nx ny hy) :
    Periodic2 nx ny (hxNext cy e hx) ∧ Periodic2 nx ny (hyNext cx e hy) := by
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> intro i j <;> simp only [hxNext, hyNext]
  · rw [pe.1, pe.1, px.1]
  · rw [show j + (ny : Int) + 1 = (j + 1) + ny by omega, pe.2, pe.2, px.2]
  · rw [show i + (nx : Int) + 1 = (i + 1) + nx by omega, pe.1, pe.1, py.1]
  · rw [pe.2, pe.2, py.2]

theorem incr_periodic (nx ny : Nat) (cx cy : R) (e hx hy : Int → Int → R)
    (pe : Periodic2 nx ny e) (px : Periodic2 nx ny hx) (py : Periodic2 nx ny hy) :
    Periodic2 nx ny (incr cx cy e hx hy) := by
  obtain ⟨⟨hx1, hx2⟩, ⟨hy1, hy2⟩⟩ := fields_periodic nx ny cx cy e hx hy pe px py
  refine ⟨?_, ?_⟩ <;> intro i j <;> simp only [incr]
  · rw [show i + (nx : Int) - 1 = (i - 1) + nx by omega, hy1, hy1, hx1, hx1]
  · rw [show j + (ny : Int) - 1 = (j - 1) + ny by omega, hy2, hy2, hx2, hx2]

/-- **Energy identity, 2D TMz.** One Yee step conserves the discrete energy on
an `nx` by `ny` periodic grid, for all coefficients `cx`, `cy`. -/
theorem energy2_conserved (nx ny : Nat) (cx cy : R) (e hx hy : Int → Int → R)
    (pe : Periodic2 nx ny e) (px : Periodic2 nx ny hx) (py : Periodic2 nx ny hy) :
    energy2 nx ny cx cy (eNext2 cx cy e hx hy) (hxNext cy e hx) (hyNext cx e hy)
      = energy2 nx ny cx cy e hx hy := by
  obtain ⟨⟨hx1, hx2⟩, ⟨hy1, hy2⟩⟩ := fields_periodic nx ny cx cy e hx hy pe px py
  have ps := incr_periodic nx ny cx cy e hx hy pe px py
  -- Each x row telescopes to zero.
  have gx : ∀ l : Nat, Periodic nx (fun i => fluxX cx cy e hx hy i (l : Int)) := by
    intro l i
    simp only [fluxX]
    rw [show i + (nx : Int) - 1 = (i - 1) + nx by omega, hy1, pe.1, ps.1]
  -- The row sums of the y flux are periodic in j.
  have gy : Periodic ny (fun j => sumN nx (fun k => fluxY cx cy e hx hy (k : Int) j)) := by
    intro j
    apply sumN_congr
    intro k
    simp only [fluxY]
    rw [show j + (ny : Int) - 1 = (j - 1) + ny by omega, hx2, pe.2, ps.2]
  have step : energy2 nx ny cx cy (eNext2 cx cy e hx hy) (hxNext cy e hx) (hyNext cx e hy)
      - energy2 nx ny cx cy e hx hy = 0 := by
    simp only [energy2]
    rw [← sumN_sub]
    have row : ∀ l : Nat,
        sumN nx (fun k => density2 cx cy (eNext2 cx cy e hx hy) (hxNext cy e hx)
            (hyNext cx e hy) (k : Int) (l : Int))
          - sumN nx (fun k => density2 cx cy e hx hy (k : Int) (l : Int))
        = sumN nx (fun k => fluxY cx cy e hx hy (k : Int) ((l : Int) + 1))
          - sumN nx (fun k => fluxY cx cy e hx hy (k : Int) (l : Int)) := by
      intro l
      rw [← sumN_sub, sumN_congr nx _ _ (fun k => density_step_2d cx cy e hx hy k l),
        sumN_add, telescope_periodic nx _ (gx l), ← sumN_sub]
      grind
    rw [sumN_congr ny _ _ row]
    exact telescope_periodic ny _ gy
  grind

end YeeEnergy
