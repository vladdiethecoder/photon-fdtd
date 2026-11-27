# Verification and reproduction

All verification code is in `src/verify.rs`. The `waveguide` binary prints the results of these scenarios. `tests/verification.rs` asserts on the same results. Thus the printed values and the test results come from one implementation.

## Run the suite

	cargo test --release

Use `--release`. In a debug build, the CPU runs are very slow. To see the measured values, add `-- --nocapture`.

At commit `d6e23e8c9a3528007c2c537044ffc191b51796b5`, 11 tests pass. 0 tests fail, and 0 tests are ignored. The test binary runs for 9.2 s. `docs/results/test.log` has the full output. The CPML and throughput measurements below come from commit `921ccb3`. The solver kernels did not change between the two commits.

## Tests

| Test | What it proves | Pass criterion |
|---|---|---|
| `cfl_guard_rejects_courant_above_2d_limit` | Both constructors reject S = 1.001/sqrt(2). S = 1/sqrt(2) is accepted. | `Err` for S above the limit |
| `confinement_and_parity_1200x600` | The silicon strip holds the pulse. The all-air control spreads it. The GPU run matches the CPU run bit for bit. | See [Confinement criterion](#confinement-criterion), and 0 differing bits |
| `confinement_and_parity_600x1200` | The same as the previous test, but the grid is taller than it is wide. A swapped nx or ny stride fails this test. | The same |
| `bit_parity_serial_parallel_gpu_512x256` | After 500 steps, the serial CPU, rayon CPU and GPU fields are identical. | 0 differing bits for each pair |
| `bit_parity_cpml_corner_non_square_300x170` | A point source near a corner of a 300 x 170 grid puts field into both strip types and their overlap. After 800 steps, the three backends agree. | 0 differing bits in Ez. Also 0 in the four psi arrays (rayon vs serial). |
| `absorber_reflection_db` | CPML gates at N = 10 for pulses A and B | normal <= -60 dB, corner <= -50 dB |
| `absorber_reflection_grazing_db` | CPML gate at about 81 degrees, N = 10, pulses A and B | <= -50 dB |
| `absorber_metric_detects_bare_pec` | With N = 0, the metric reports the reflection of bare PEC walls | normal and corner > -6 dB |
| `large_grid_16384x20000_gpu_corners_match_cpu` | GPU offsets above 2^31 bytes are correct | 0 differing bits in two 25 x 25 corner patches |
| `tall_grid_past_cuda_grid_y_limit_matches_cpu` | GPU rows above row 65535 get updates | 0 differing bits on all 4.48e6 cells |
| `gpu_viewer_kernels_paint_render_reset` | The brush sets exactly the disk cells to the constructor's `cb` bits. The heat map puts row j at image row ny-1-j, with +Ez red, -Ez blue and silicon grey. Reset clears the fields and keeps `cb`. | exact bits and pixel values |

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
| 1200 x 600 | 0.9999 | 0.9933 | 0.0545 | 0.0677 |
| 600 x 1200 | 0.9999 | 0.9791 | 0.0000 | 0.0451 |

With the CPML, the all-air grid keeps only 6.44e-11 of its peak energy at t_end (1200 x 600). The air band-energy fraction is thus a ratio of residual values, and the 600 x 1200 value of 0.0000 has no physical meaning. The probe metric integrates over the full run and is the reliable air-control metric.

### Bit parity

The CUDA kernels compile with `--fmad=false`. Thus NVRTC does not contract `a * b + c` into a fused multiply-add. Rust does not contract floating-point operations. Each kernel has the same order of operations as the CPU code, and the source amplitude comes from one host function.

The CPML tables and strip index maps come from one `CpmlLayout`, which both backends receive. With these conditions, IEEE 754 arithmetic gives the same bits on both devices. The tests compare `f64::to_bits` and allow no difference.

### Mutation runs

Each mutation below was applied to the code and then removed. The table shows the tests that failed.

| Mutation | Failed tests |
|---|---|
| GPU `psi_e_y`: `Ez -= cb psi` changed to `Ez += cb psi` | all parity tests, `absorber_reflection_db`, `absorber_reflection_grazing_db`, both corner and tall-grid tests |
| GPU `psi_h_x`: a times 0.5 | all parity tests, `absorber_reflection_db`, both corner and tall-grid tests |
| CPU `e_row`: y-strip Ez correction times 0 | all parity tests (6 failed) |
| GPU `psi_e_x`: right-strip map shifted by one column | none |

The last mutation is equivalent to the original code. The strip is W = N + 1 cells wide, and the extra column has a = 0. Thus a shift by one column applies the same nonzero corrections. The suite cannot detect an equivalent mutation, and no test needs to.

### Large GPU grid

The grid has 16384 x 20000 cells. This is 3.28e8 cells. Each f64 array has 2.62 GB, and the 4 full-grid arrays use 10.5 GB of VRAM. `GpuYee::new` builds `cb` in row blocks of about 32 MiB. Thus the host never holds a full-size array.

The test puts an Ez impulse near each corner and runs 8 steps with a 16-cell CPML. The test then compares 25 x 25 patches with the same impulses on a 64 x 64 CPU grid. Near a corner, the CPML tables and strips of the two grids are the same, because they depend only on the distance to the nearest walls. The high-corner impulse is at a byte offset above 2^31, and the test asserts this condition. The test also counts the nonzero cells: the patches hold all 290 nonzero cells of the CPU grid.

The suggested grid of 16384 x 16385 does not cross 2^31 bytes in the updated cells. Only its last row is past 2^31 bytes, and this row is a PEC row that the update does not change. For this reason, the test uses ny = 20000.

### Rows above the CUDA gridDim.y limit

CUDA limits gridDim.y to 65535. The main kernels and the x-strip kernels launch min(ny, 65535) rows of blocks and use a loop with a grid stride over the rows. The test uses a 64 x 70000 grid with a second source near the top row, and it compares all cells with the CPU. In Stage 1, a mutation run confirmed that the test can fail: a loop that stops after one pass made the test fail.

## Viewer

The viewer needs a display, so `cargo test` does not run it. These runs verified it on the RTX 5090:

	cargo run --release --features gui --bin viewer -- --script examples/viewer_demo.script
	cargo run --release --features gui --bin viewer -- --capture out/a.ppm --steps 6000
	cargo run --release --features gui --bin viewer -- --host-copy --capture out/b.ppm --steps 6000

The script sends egui pointer events through `raw_input_hook`. The events go through the same widgets and hit tests as a mouse. `docs/results/viewer_demo.log` shows each effect:

- **Pause** stopped the run, two **Step** clicks advanced the step counter from 2540 to 2542, and **Run** continued.
- The silicon brush painted from the press cell (300, 400), with 61 disks. The erase brush started at cell (585, 380), with 21 disks.
- A click with **Place source** moved the source to (320, 400).
- Two clicks on the gain slider set the gain to 10.000 and then 0.250.
- **Clear fields** cleared the fields at step 3922.

`docs/images/viewer_edited.png` is the second screenshot of the script. It shows the new strip, the cut in the vertical arm, and the source on the new strip.

`docs/results/viewer_paths.log` shows that the viewer used CUDA-GL interop by default and the host copy with `--host-copy`.

### Pointer input on KDE Plasma Wayland

The viewer ran as a native Wayland window. `ydotoold` created a uinput virtual mouse, so the events went through the kernel, libinput and KWin, like the events of a physical mouse. `kdotool getmouselocation` read the pointer position back. Pointer acceleration made absolute moves wrong by a factor of about 2. Thus each move was a closed loop of small relative steps until the read-back position was within 1 pixel of the target. `docs/results/viewer_real_mouse.log` shows the effects:

- A silicon drag that started at screen pixel (849, 676) started at cell (299, 102). The planned cell was (300, 100), with an estimated 28-pixel title bar.
- An erase drag across the vertical arm started at cell (583, 380).
- A click with **Place source** moved the source to (399, 100).
- Clicks on **Pause** and **Run** set `running = false` and `running = true`. F12 saved `docs/images/viewer_real_mouse.png`.

Two earlier attempts did not reach the viewer. Under XWayland, the XTest drags of the desktop automation tool had no effect. The first uinput drag used an absolute move, which acceleration put at about (1699, 1079), at the bottom edge of the screen, outside the viewer. That press and drag went to the desktop panel or to the window under the pointer at that position.

## Absorber reflection

Run the measurement:

	cargo run --release --bin waveguide -- --check-pml
	cargo run --release --bin waveguide -- --check-pml --sweep

### Method

Each case compares a test grid with a larger reference grid. Both grids have a point source and the same waveform. The two grids are the same, except for the boundary that the wave reaches in the test grid.

| Geometry | Test grid | Source | Reference grid | Steps | Probes |
|---|---|---|---|---|---|
| Square | 512 x 512 | (256, 256) | 2048 x 2048, offset 768 | 1200 | normal: (nx-1-N-5, 256). Corner: (nx-1-N-5, ny-1-N-5). |
| Grazing | 1024 x 256 | (150, 128) | 3824 x 3056, offset 1400 | 1400 | (850, N+5) |

Each probe is 5 cells inside the CPML interface, on the interior side. At the grazing probe, the direct wave arrives at about 81 degrees from the wall normal (atan(700 / 113) at N = 10). The probe also receives the waves from the other walls, so the result is the total boundary error at that cell.

The result for each probe is:

	R_dB    = 20 log10( max_t |Ez_test(t) - Ez_ref(t)| / max_t |Ez_ref(t)| )
	R_f0_dB = 20 log10( |DFT(Ez_test - Ez_ref)(f0)| / |DFT(Ez_ref)(f0)| ),  f0 = 1/40

The reference must be exact. In the Yee stencil, a change moves at most one cell per step. This speed is faster than c, which is 0.7 cells per step. `reference_is_clean` checks each probe. For each probe, the path from the source to the reference CPML strip and back to the probe must be longer than the number of steps. Thus the reference probe values do not depend on the reference walls. This independence is exact, not only below a tolerance.

Pulse A is the Gaussian sine of Stage 1. Pulse B is a differentiated Gaussian with no DC content. Its spectral peak is at 1/40, so it has tau = sqrt(2) * 40 / (2 pi) = 9.00 and t0 = 5 tau. Pulse B checks that a DC component does not change the result.

The GPU runs all traces. The parity tests show that the GPU result is bitwise equal to the CPU result.

### Results against thickness

`docs/results/check_pml.log` has the full table with both metrics. The README gives the time-domain values. Summary, with the default profile:

- N = 0 (bare PEC walls) gives -0.2 dB to +7.0 dB. Thus the metric detects a reflection.
- N = 6 is the thinnest PML that meets -60 dB at normal incidence and -50 dB at the corner for both pulses.
- N = 8 is the thinnest PML that also meets -50 dB at grazing incidence.
- N = 10 (the default) gives -87.7 dB or lower at normal incidence and at the corner, and -66.8 dB or lower at grazing incidence.
- From N = 8 to N = 14, pulse A and pulse B differ by less than 4 dB. At N >= 16 the difference is larger, but all values are below -100 dB. The DC component of pulse A does not change the conclusion.

### Parameter sweep

The sweep (`docs/results/check_pml_sweep.log`) has 160 profiles: N in {8, 10, 12, 16}, kappa_max in {1, 3, 5, 7, 11}, alpha_max in {0, 0.05, 0.1, 0.2} and m in {3, 4}. The table shows the effect of kappa_max at m = 3, alpha_max = 0.05. Each value is the worst case of pulses A and B.

| N | kappa_max | Worst of normal and corner | Grazing |
|---|---|---|---|
| 10 | 1 | -87.7 dB | -66.8 dB |
| 10 | 3 | -88.1 dB | -43.3 dB |
| 10 | 5 | -87.6 dB | -40.4 dB |
| 10 | 7 | -84.0 dB | -40.2 dB |
| 10 | 11 | -64.8 dB | -41.9 dB |
| 12 | 1 | -92.1 dB | -80.2 dB |
| 12 | 7 | -91.7 dB | -50.9 dB |

In the rows of this table, kappa_max > 1 makes the grazing reflection 23 to 30 dB worse. At the other probes, it improves the worst case by 0.4 dB or less. With alpha_max = 0.2, kappa_max > 1 improves the worst case of normal and corner by up to 17.7 dB. Those profiles are worse than the default at every probe. At each N in the sweep, the best profile (worst case of all probes and both pulses) has kappa_max = 1.

The usual advice for kappa_max (5 to 11) is to absorb evanescent waves. The point source in this check is hundreds of cells from the walls, so its evanescent field at the walls is negligible.

Hypothesis, not tested: kappa_max > 1 shortens the effective wavelength inside the CPML by the factor kappa. With 10 cells, the stretched field is then sampled too coarsely, and the discretization reflection increases. The effect is strongest at grazing incidence, where the wave crosses many cells along the wall.

alpha_max at N = 10, kappa_max = 1 (worst of normal and corner, then grazing): 0 gives -88.6 dB and -63.8 dB. 0.05 gives -87.7 dB and -66.8 dB. 0.1 gives -71.9 dB and -67.5 dB. 0.2 gives -55.3 dB and -57.2 dB. The default 0.05 is the best compromise of these values.

The default profile has kappa_max = 1. A problem with strong evanescent fields at the boundary (for example, a waveguide below cutoff that ends in the CPML) can need kappa_max > 1. No test measures that case.

### History: the Stage 1 sponge

v0.1.0 used a graded conductive sponge. At 40 cells, it gave -26.3 dB at normal incidence and -15.1 dB at the corner. The first measurement of the sponge found a defect: the H update did not apply the loss term. With this defect, the result was -18.0 dB and -12.5 dB. The CPML replaced the sponge and its per-cell coefficients `ca` and `hb`.
