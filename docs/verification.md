# Verification and reproduction

All verification code is in `src/verify.rs`. The `waveguide` binary prints the results of these scenarios. `tests/verification.rs` asserts on the same results. Thus the printed values and the test results come from one implementation.

## Run the suite

	cargo test --release

Use `--release`. In a debug build, the CPU runs are very slow. To see the measured values, add `-- --nocapture`.

At commit `160cfbb29b01fb1f351493c305fc157c57bd4038`, 7 tests pass. 0 tests fail, and 0 tests are ignored. The test binary runs for 27.8 s. `docs/results/test.log` has the full output.

## Tests

| Test | What it proves | Pass criterion |
|---|---|---|
| `cfl_guard_rejects_courant_above_2d_limit` | Both constructors reject S = 1.001/sqrt(2). S = 1/sqrt(2) is accepted. | `Err` for S above the limit |
| `confinement_and_parity_1200x600` | The silicon strip holds the pulse. The all-air control spreads it. The GPU run matches the CPU run bit for bit. | See [Confinement criterion](#confinement-criterion), and 0 differing bits |
| `confinement_and_parity_600x1200` | The same as the previous test, but the grid is taller than it is wide. A swapped nx or ny stride fails this test. | The same |
| `bit_parity_serial_parallel_gpu_512x256` | After 500 steps, the serial CPU, rayon CPU and GPU fields are identical. | 0 differing bits for each pair |
| `absorber_reflection_db` | Regression limit for the sponge reflection | normal < -25 dB, corner < -14 dB |
| `large_grid_16384x20000_gpu_corners_match_cpu` | GPU offsets above 2^31 bytes are correct | 0 differing bits in two 25 x 25 corner patches |
| `tall_grid_past_cuda_grid_y_limit_matches_cpu` | GPU rows above row 65535 get updates | 0 differing bits on all 4.48e6 cells |

### Confinement criterion

`trapped` in `src/verify.rs` needs all four of these conditions:

- The silicon run has more than 90 % of the grid energy in the band at t_end.
- The silicon run has more than 90 % of the time-integrated Ez^2 at x = 400 in the band.
- The all-air run has less than 20 % of the grid energy in the band.
- The all-air run has less than 20 % of the time-integrated Ez^2 at x = 400 in the band.

The band is the strip plus half a strip width of air on each side: rows 284..316 for ny = 600.

Measured values:

| Grid | Si band energy | Si probe | Air band energy | Air probe |
|---|---|---|---|---|
| 1200 x 600 | 0.9999 | 0.9939 | 0.0979 | 0.0704 |
| 600 x 1200 | 0.9999 | 0.9796 | 0.0603 | 0.0454 |

### Bit parity

The CUDA kernels compile with `--fmad=false`. Thus NVRTC does not contract `a * b + c` into a fused multiply-add. Rust does not contract floating-point operations. Each kernel has the same order of operations as the CPU code, and the source amplitude comes from one host function. With these conditions, IEEE 754 arithmetic gives the same bits on both devices. The tests compare `f64::to_bits` and allow no difference.

### Large GPU grid

The grid has 16384 x 20000 cells. This is 3.28e8 cells. Each f64 array has 2.62 GB, and the 6 arrays use 15.7 GB of VRAM. `GpuYee::new` builds the coefficients in row blocks of about 32 MiB. Thus the host never holds a full-size array.

The test puts an Ez impulse near each corner and runs 8 steps. The test then compares 25 x 25 patches with the same impulses on a 64 x 64 CPU grid. Near a corner, the coefficients of the two grids are the same, because the sponge depth uses only the distance to the nearest edge. The high-corner impulse is at a byte offset above 2^31, and the test asserts this condition. The test also counts the nonzero cells: the patches hold all 290 nonzero cells of the CPU grid.

The suggested grid of 16384 x 16385 does not cross 2^31 bytes in the updated cells. Only its last row is past 2^31 bytes, and this row is a PEC row that the update does not change. For this reason, the test uses ny = 20000.

### Rows above the CUDA gridDim.y limit

CUDA limits gridDim.y to 65535. The kernels launch min(ny, 65535) rows of blocks and use a loop with a grid stride over the rows. The test uses a 64 x 70000 grid with a second source near the top row, and it compares all cells with the CPU. A mutation run confirmed that the test can fail: a loop that stops after one pass made the test fail, as expected.

## Absorber reflection

Run the measurement:

	cargo run --release --bin waveguide -- --check-pml

### Method

The measurement uses two all-air grids. Each grid has a point source at its centre and the same waveform: lambda0 = 40 cells, t0 = 90 and tau = 30. Each run has 1200 steps.

- The test grid has 512 x 512 cells and the sponge under test.
- The reference grid has 2048 x 2048 cells. The centre offset (768, 768) maps each test cell to a reference cell.

Each probe is 5 cells inside the sponge interface, on the interior side. One probe is on the centre row near the +x wall (normal incidence). The other probe is on the diagonal near the corner. The result for each probe is:

	R_dB    = 20 log10( max_t |Ez_test(t) - Ez_ref(t)| / max_t |Ez_ref(t)| )
	R_f0_dB = 20 log10( |DFT(Ez_test - Ez_ref)(f0)| / |DFT(Ez_ref)(f0)| )

The reference must be exact. In the Yee stencil, a change moves at most one cell per step. This speed is faster than c, which is 0.7 cells per step. `reference_is_clean` checks each probe. For each probe, the path from the source to the reference sponge and back to the probe must be longer than 1200 cells. Thus the reference probe values do not depend on the reference walls. This independence is exact, not only below a tolerance.

### Results at commit 160cfbb

| Sponge (cells) | Normal, time max | Normal, at f0 | Corner, time max | Corner, at f0 |
|---|---|---|---|---|
| 10 | -21.3 dB | -21.2 dB | -5.8 dB | -5.5 dB |
| 20 | -20.6 dB | -17.1 dB | -8.3 dB | -7.6 dB |
| 40 | -26.3 dB | -26.4 dB | -15.1 dB | -15.1 dB |
| 80 | -45.4 dB | -57.0 dB | -36.4 dB | -47.9 dB |

The scenarios use a 40-cell sponge. At 40 cells, the sponge reflects -26 dB at normal incidence and -15 dB near a corner. The request that started this measurement gives about -40 dB for simple grading or Mur, and below -80 dB for CPML. The sponge is below both of these levels. The waveguide confinement result is not sensitive to this reflection: at t_end, the energy of the all-air control is 4.54e-5 of its peak.

The first measurement found a defect. The H update did not apply the self coefficient ca, so the sponge had electric loss only and was not matched. With this defect, the 40-cell sponge gave -18.0 dB at normal incidence and -12.5 dB at the corner. Commit 160cfbb includes the correction.

Hypothesis, not tested: most of the remaining normal-incidence reflection comes from the Hx and Hy coefficients. They use the cell centre, not the half-cell positions. Oblique incidence on an isotropic conductor causes the corner reflection. A CPML removes both causes.
