# photon-fdtd

photon-fdtd is a minimal 2D TM_z Yee FDTD solver in Rust. It has a CPU backend (one thread or a rayon thread pool) and a CUDA backend for the RTX 5090 (sm_120). Both backends use flat, row-major `Vec<f64>` arrays and natural units (c = eps0 = mu0 = dx = 1). The two backends give bitwise identical fields. A CFS-CPML (Roden and Gedney, 2000) absorbs the outgoing waves. Its auxiliary arrays exist only in the boundary strips.

## Reproduce every claim

Run the verification suite:

	cargo test --release

Run the demonstration binaries:

	cargo run --release --bin waveguide                          # silicon strip confinement, GPU parity
	cargo run --release --bin waveguide -- --check-pml           # CPML reflection in dB against thickness
	cargo run --release --bin waveguide -- --check-pml --sweep   # CPML parameter sweep (160 profiles)
	cargo run --release --bin bench -- --cpu 2048,4096 --gpu 4096,8192

The GPU tests and the binaries need CUDA device 0. They compile the kernels for `sm_120` at run time with NVRTC.

## Status by claim

All results come from commit `921ccb35809a5872e2981425fb169f3d2dd35507` on the machine in [Environment](#environment). The raw output of each run is in `docs/results/`.

| Capability | Status | Evidence (entry point) |
|---|---|---|
| CFL enforcement (S <= 1/sqrt(2)) | Verified | `cfl_guard_rejects_courant_above_2d_limit` |
| 2D TM_z Yee update | Partial. The update is self-consistent and gives the expected physics. There is no comparison with an analytic solution. | `confinement_and_parity_*`, docs/methodology.md |
| Silicon strip confinement (eps_r = 12.11 in air) | Verified on 1200 x 600 and 600 x 1200 | `confinement_and_parity_1200x600`, `confinement_and_parity_600x1200`, `waveguide` |
| CPU serial, CPU rayon and GPU bit parity, with the CPML strips and corners | Verified, 0 differing bits | `bit_parity_cpml_corner_non_square_300x170`, `bit_parity_serial_parallel_gpu_512x256`, `confinement_and_parity_*` |
| CPML reflection at normal incidence and at the corner (10 cells) | Verified: -87.7 dB or lower for both pulses. Gates: -60 dB (normal) and -50 dB (corner). | `absorber_reflection_db`, `waveguide --check-pml` |
| CPML reflection at grazing incidence, about 81 degrees (10 cells) | Verified: -66.8 dB or lower. Gate: -50 dB. | `absorber_reflection_grazing_db` |
| The reflection metric detects a reflection | Verified: bare PEC walls give -0.2 dB (normal) and +4.9 dB (corner) | `absorber_metric_detects_bare_pec` |
| GPU byte offsets above 2^31 | Verified on 16384 x 20000 (2.62 GB per array) | `large_grid_16384x20000_gpu_corners_match_cpu` |
| GPU rows above the CUDA gridDim.y limit (65535) | Verified on 64 x 70000 | `tall_grid_past_cuda_grid_y_limit_matches_cpu` |
| Throughput (Mcells/s) | Measured | `bench`, docs/architecture.md |
| Flux, S-parameters | Open. Not implemented. | none |
| CPML absorption of evanescent waves | Open. Not measured. | none |
| Device selection, multi-GPU | Open. Device 0 is hard-coded. | none |

Test suite result at that commit: 10 tests pass, 0 fail, 0 are ignored. The test binary ran for 12.4 s. The wall time from `cargo clean` to the end was 21.5 s, including the build.

## CPML thickness

The default profile is m = 3, sigma_max = 0.8 (m + 1) = 3.2, kappa_max = 1, alpha_max = 0.05 and m_alpha = 1. The scenarios use N = 10 cells. The table gives the reflection in dB (time-domain maximum), from `docs/results/check_pml.log`:

| N (cells) | Normal, A | Normal, B | Corner, A | Corner, B | Grazing, A | Grazing, B |
|---|---|---|---|---|---|---|
| 0 (bare PEC) | -0.2 | 1.3 | 4.9 | 7.0 | -0.0 | -0.0 |
| 4 | -43.0 | -44.0 | -38.3 | -38.7 | -23.0 | -22.5 |
| 6 | -65.7 | -63.3 | -61.6 | -61.0 | -36.9 | -37.5 |
| 8 | -81.1 | -82.0 | -80.2 | -76.9 | -56.2 | -52.4 |
| 10 | -90.5 | -88.4 | -89.9 | -87.7 | -70.2 | -66.8 |
| 12 | -96.0 | -95.5 | -93.5 | -92.1 | -80.7 | -80.2 |
| 16 | -102.9 | -102.6 | -100.2 | -100.9 | -103.9 | -109.3 |

Pulse A is a Gaussian-modulated sine (lambda0 = 40 cells, t0 = 90, tau = 30). Pulse B is a differentiated Gaussian with no DC content (tau = 9.00, t0 = 45.02).

- N = 6 is the thinnest PML that meets -60 dB at normal incidence and -50 dB at the corner for both pulses.
- N = 8 is the thinnest PML that also meets -50 dB at grazing incidence.

The parameter sweep does not agree with the usual advice of kappa_max = 5 to 11. At N = 10, kappa_max = 7 changes the grazing reflection from -66.8 dB to -40.2 dB. docs/verification.md gives the data and a hypothesis.

## Verified numbers

These are the results of the waveguide run. The grid is 1200 x 600 with a 10-cell CPML and S = 0.700036 (0.99 x 1/sqrt(2)). The run has 2572 steps (t = 1800). The strip occupies rows 292..308 (16 cells). The source is a soft line source at x = 100 across the strip: a Gaussian-modulated sine with lambda0 = 40 cells, t0 = 300 and tau = 80.

| Metric | Silicon strip | All-air control |
|---|---|---|
| Grid energy in rows 284..316 at t_end | 0.9999 | 0.0545 |
| Time-integrated Ez^2 in rows 284..316 at x = 400 | 0.9933 | 0.0677 |
| RMS transverse width at x = 400 | 17.9 cells | 155.8 cells |
| Grid energy at t_end / peak grid energy | 0.496 | 6.44e-11 |
| GPU vs CPU serial, differing Ez values | 0 of 720000 | not run |

In the silicon run, the pulse centre on the axis is at x = 507 at t_end. The pulse moves at a group velocity of about 0.271 c, so n_g is about 3.69.

These are the throughput results, f64, median of 3 trials (from `docs/results/bench.log`):

| Backend | Grid | Mcells/s | Effective GB/s (80 B/cell) |
|---|---|---|---|
| CPU, 1 thread | 4104^2 | 438.5 | 35 |
| CPU, 1 thread | 4096^2 | 225.4 | 18 |
| CPU, rayon 24 threads | 4096^2 | 622.8 | 50 |
| RTX 5090 | 4096^2 | 15561.7 | 1245 |
| RTX 5090 | 8192^2 | 15621.1 | 1250 |
| RTX 5090 | 16384^2 | 15654.2 | 1252 |

One cell update is one H update plus one E update of one cell, with the CPML strips. Desktop processes shared the GPU during the benchmark (kwin_wayland, gamescope, Chrome, Discord, a MetaBonk viewer and the CUDA MPS server).

v0.1.0 used a conductive sponge. At 4096^2, it gave 12959.7 Mcells/s on the GPU and 498.1 Mcells/s on 24 CPU threads. The CPML removed the per-cell loss coefficients, so the minimum traffic decreased from 104 to 80 bytes per cell.

## Environment

- CPU: AMD Ryzen 9 7900X, 12 cores, 24 threads. 32 MiB L3 per CCD, 2 CCDs.
- RAM: 31.9 GB, DDR5 at 6000 MT/s, 2 modules.
- GPU: NVIDIA GeForce RTX 5090, 32 GB, driver 615.78.08, CUDA 13.1 (NVRTC).
- OS: Fedora, Linux 7.2.12-200.fc44.x86_64.
- Rust: rustc 1.96.0. Release profile with `lto = "fat"`, `codegen-units = 1` and `-C target-cpu=native`.

## Limits that are not verified

- The suite does not run CPU grids larger than 4104^2. Host RAM (31.9 GB) limits the CPU solver.
- No test crosses 2^32 bytes per array. The largest test array is 2.62 GB.
- The GPU backend uses CUDA device 0 only. The kernels compile for `sm_120` only.
- The CPML is measured for propagating waves in air at three probe cells. Evanescent waves and the reflection of a guided mode in the silicon strip are not measured.
- No test compares the numerical dispersion with the analytic Yee dispersion relation.
- In the air control, the band-energy fraction at t_end is a ratio of residual energies near 6e-11 of the peak. For the air control, the probe metric is the reliable metric.

## Documentation

- [docs/methodology.md](docs/methodology.md): the discrete equations, the CFL derivation, the CPML and the sources.
- [docs/verification.md](docs/verification.md): what each test does, how the reflection is measured, and the parameter sweep.
- [docs/architecture.md](docs/architecture.md): the memory layout, the strip layout, the CUDA mapping and the throughput analysis.
