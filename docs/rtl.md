# Fixed-point 1D Yee stencil in SystemVerilog

This is a standalone RTL artifact. It is not connected to the 2D solver, the
CPML, or the CUDA path, and it shows none of their properties.

## Files

| File | Content |
|---|---|
| `rtl/yee1d.sv` | The stencil: N cells, parallel H phase then parallel E phase, sticky overflow flag |
| `rtl/tb_yee1d.sv` | Self-checking testbench: every E and H word and the flag, after every step, bit for bit |
| `src/fixed1d.rs` | Bit-exact software reference (the same integer operations) |
| `src/bin/yee1d_vectors.rs` | Writes the golden vectors in `rtl/vectors/{nominal,saturate}` |
| `tests/verification.rs`, `fixed_point_yee1d_reference_and_vectors` | Reference checks (below) |

## Update

PEC ends: E[0] and E[N-1] stay 0. One time step:

	H[i] <- sat(H[i] + rnd(CH * (E[i+1] - E[i])))            0 <= i <= N-2
	E[i] <- sat(E[i] + rnd(CB[i] * (H[i] - H[i-1])) + s[i])   1 <= i <= N-2

`s[i]` is the source sample at cell `SRC` and 0 elsewhere.

## Word lengths and arithmetic

| Quantity | Format | Range | LSB |
|---|---|---|---|
| E, H, source sample | 18-bit two's complement, 15 fraction bits | [-4, 4) | 2^-15 |
| CH, CB | 18-bit two's complement, 16 fraction bits | [-2, 2) | 2^-16 |
| Difference | 19 bits | exact | 2^-15 |
| Product | 37 bits, 31 fraction bits | exact | 2^-31 |
| Sum before saturation | 22 bits | no overflow | 2^-15 |

- Rounding: `rnd(p) = (p + 2^15) >>> 16`, an arithmetic shift. This is round
  half up (towards plus infinity). The result has 15 fraction bits.
- Overflow: each sum is clamped to [-2^17, 2^17 - 1], that is [-4, 4 - 2^-15].
  A clamp sets the sticky flag `ovf`. Only `rst` clears it. There is no
  wrap-around.
- Coefficient quantisation: CB = S / eps is rounded to 16 fraction bits. For
  S = 0.5 and eps = 12.11, CB = 0.041288 has a relative error of at most
  1.2e-4.

## Test case

N = 64, Courant number S = 0.5, a dielectric slab (eps 12.11) on cells 40 to
55, and a Gaussian-sine source at cell 8, for T = 400 steps:

- `nominal`: source peak 0.5. No saturation.
- `saturate`: source peak 3.9. The first clamp is at step 59.

## Evidence

Measured with `cargo test --release --test verification -- fixed_point`:

| Check | Result |
|---|---|
| Committed vectors equal a fresh run of `yee1d_vectors` | yes, all 14 files |
| Local error of one step against f64 (same quantised coefficients) | E 0.750 LSB (bound 1), H 0.500 LSB (bound 0.5) |
| Drift after 400 steps against f64 (same coefficients) | 110.6 LSB (measured; a closed lossless cavity, so rounding errors do not decay) |
| Drift after 400 steps against f64 (exact coefficients) | 144.1 LSB (`yee1d_vectors` output) |
| Saturate case | flag set at step 59, all words inside 18 bits |

## Simulation (Verilator)

Measured with Verilator 5.046 (`docs/results/rtl/`):

	verilator --binary -j 0 -Wall --top-module tb_yee1d rtl/yee1d.sv rtl/tb_yee1d.sv
	obj_dir/Vtb_yee1d +VEC=rtl/vectors/nominal
	obj_dir/Vtb_yee1d +VEC=rtl/vectors/saturate

| Run | Result |
|---|---|
| Lint (`-Wall`) | 0 warnings |
| `nominal` | 400 steps, 51600 words checked, 0 mismatches, final ovf 0, PASS, exit 0 |
| `saturate` | 400 steps, 51600 words checked, 0 mismatches, final ovf 1, PASS, exit 0 |
| Mutant: `rnd` without the rounding constant (round down) | 42241 mismatches, first at step 2 (E[7] = 00000, expected 00001), `$fatal`, exit 1 |

Thus `rtl/yee1d.sv` is bit-exact with `src/fixed1d.rs` on both test cases, and
the testbench detects a 1-LSB rounding change.

## Not verified

- Only the two committed test cases were simulated (N = 64, 400 steps).
- No synthesis was run (no synthesis tool is installed), so this document
  gives no area, timing, or power figures.
