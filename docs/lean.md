# Lean 4 proof: energy identity of the lossless periodic Yee scheme

`lean/YeeEnergy.lean` proves, in core Lean 4 (no Mathlib), that one leapfrog
Yee step conserves a discrete energy on a periodic grid, in 1D and in 2D TMz.

## What is proved

| Theorem | Statement |
|---|---|
| `YeeEnergy.energy_conserved` | 1D: `energy n c (eNext c e h) (hNext c e h) = energy n c e h` for fields of period `n` |
| `YeeEnergy.energy2_conserved` | 2D TMz: `energy2 nx ny cx cy (eNext2 ..) (hxNext ..) (hyNext ..) = energy2 nx ny cx cy e hx hy` for fields of period `nx` by `ny` |

The state is E at step n and H at step n - 1/2. The energy is
W = Σ E² + Σ H·H⁺, where H⁺ is H at step n + 1/2 (in 2D, both Hx and Hy
terms). The proof shows that the change of the energy density at each cell is
a difference of a flux at two neighbouring cells (`density_step`,
`density_step_2d`), and a sum of such differences over one period is zero.

## Assumptions

- The field values are in any commutative ring (`Lean.Grind.CommRing R`).
  The arithmetic is exact.
- Uniform, lossless medium: one coefficient per axis (the Courant number in
  natural units), the same at every cell.
- Periodic boundaries.

## What it does not show

- Nothing about the f64 CPU or CUDA code of photon-fdtd, or about rounding.
- Nothing about the CPML, sources, material contrast, or the S-parameter
  ports. The solver uses all of them, so the proof does not certify the solver.
- Not stability: W is a positive quadratic form only under the CFL condition,
  which the file does not prove or use. The identity holds for every
  coefficient.
- The proof does not show that the Lean definitions match the Rust kernels. The
  2D definitions restate lines 31 to 33 of the update in `docs/methodology.md`
  with eps_r = 1, the CPML scale factors equal to 1 and no psi terms, and with
  periodic indices instead of the PEC ring of the solver.

## Evidence

`docs/results/lean/print_axioms.txt` (Lean 4.36.0, Lake 5.0.0):

- `lake build`: success.
- `#print axioms` for the four main theorems: `propext`, `Classical.choice`,
  `Quot.sound` only (the standard axioms of Lean; no `sorryAx`).
- No `sorry`, `admit` or `axiom` in the source.
- A concrete check over `Int`, with arbitrary integer fields and coefficients
  3 and 2 (far outside the CFL limit): the energy over 10 steps stays at 525
  (1D, period 7) and 1710 (2D, 5 x 4).
- Mutation (`docs/results/lean/mutation.txt`): a wrong flux coefficient makes
  the build fail.

Reproduce:

	curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y
	cd lean && lake build && lake env lean PrintAxioms.lean
