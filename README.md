# photon-fdtd

photon-fdtd is a minimal 2D TM_z Yee FDTD solver in Rust. It has a CPU backend (one thread or a rayon thread pool) and a CUDA backend for the RTX 5090 (sm_120). Both backends use flat, row-major `Vec<f64>` arrays and natural units (c = eps0 = mu0 = dx = 1). The two backends give bitwise identical fields.

## Reproduce every claim

Run the verification suite:

	cargo test --release

Run the demonstration binaries:

	cargo run --release --bin waveguide              # silicon strip confinement, GPU parity
	cargo run --release --bin waveguide -- --check-pml   # absorber reflection in dB
	cargo run --release --bin bench -- --cpu 2048,4096 --gpu 4096,8192

The GPU tests and the binaries need CUDA device 0. They compile the kernels for `sm_120` at run time with NVRTC.

## Status by claim

All results come from commit `160cfbb29b01fb1f351493c305fc157c57bd4038` on the machine in [Environment](#environment).

| Capability | Status | Evidence (entry point) |
|---|---|---|
| CFL enforcement (S <= 1/sqrt(2)) | Verified | `cfl_guard_rejects_courant_above_2d_limit` |
| 2D TM_z Yee update | Partial. The update is self-consistent and gives the expected physics. There is no comparison with an analytic solution. | `confinement_and_parity_*`, docs/methodology.md |
| Silicon strip confinement (eps_r = 12.11 in air) | Verified on 1200 x 600 and 600 x 1200 | `confinement_and_parity_1200x600`, `confinement_and_parity_600x1200`, `waveguide` |
| CPU serial, CPU rayon and GPU bit parity | Verified, 0 differing bits | `bit_parity_serial_parallel_gpu_512x256`, `confinement_and_parity_*` |
| Absorbing boundary | Measured. The matched sponge is simple and is not a PML. 40 cells give -26.3 dB at normal incidence and -15.1 dB at the corner. | `absorber_reflection_db`, `waveguide --check-pml` |
| GPU byte offsets above 2^31 | Verified on 16384 x 20000 (2.62 GB per array) | `large_grid_16384x20000_gpu_corners_match_cpu` |
| GPU rows above the CUDA gridDim.y limit (65535) | Verified on 64 x 70000 | `tall_grid_past_cuda_grid_y_limit_matches_cpu` |
| Throughput (Mcells/s) | Measured | `bench`, docs/architecture.md |
| Flux, S-parameters | Open. Not implemented. | none |
| CPML | Open. Not implemented. | none |
| Device selection, multi-GPU | Open. Device 0 is hard-coded. | none |

Test suite result at that commit: 7 tests pass, 0 fail, 0 are ignored. The test binary ran for 27.8 s. The wall time from `cargo clean` to the end was 36.4 s, including the build.

## Verified numbers

The raw output of each run is in `docs/results/`.


These are the results of the waveguide run. The grid is 1200 x 600 with a 40-cell sponge and S = 0.700036 (0.99 x 1/sqrt(2)). The run has 2572 steps (t = 1800). The strip occupies rows 292..308 (16 cells). The source is a soft line source at x = 100 across the strip: a Gaussian-modulated sine with lambda0 = 40 cells, t0 = 300 and tau = 80. The CPU silicon run takes 2.29 s.

| Metric | Silicon strip | All-air control |
|---|---|---|
| Grid energy in rows 284..316 at t_end | 0.9999 | 0.0979 |
| Time-integrated Ez^2 in rows 284..316 at x = 400 | 0.9939 | 0.0704 |
| RMS transverse width at x = 400 | 16.1 cells | 147.7 cells |
| Grid energy at t_end / peak grid energy | 0.496 | 4.54e-5 |
| GPU vs CPU serial, differing Ez values | 0 of 720000 | not run |

In the silicon run, the pulse centre on the axis is at x = 507 at t_end. The pulse moves at a group velocity of about 0.271 c, so n_g is about 3.69.

These are the throughput results, f64, median of 3 trials (from `docs/results/bench.log`):

| Backend | Grid | Mcells/s | Effective GB/s (104 B/cell) |
|---|---|---|---|
| CPU, 1 thread | 4104^2 | 422.0 | 44 |
| CPU, 1 thread | 4096^2 | 221.2 | 23 |
| CPU, rayon 24 threads | 4096^2 | 498.1 | 52 |
| RTX 5090 | 4096^2 | 12959.7 | 1348 |
| RTX 5090 | 8192^2 | 12951.0 | 1347 |
| RTX 5090 | 16384^2 | 12927.4 | 1344 |

One cell update is one H update plus one E update of one cell. Desktop processes shared the GPU during the benchmark (kwin_wayland, gamescope, Chrome, Discord, a MetaBonk viewer and the CUDA MPS server). GPU utilization before the run was 9 %.

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
- The sponge reflection depends on frequency and on the angle. The suite measures one pulse at two probe cells.
- No test compares the numerical dispersion with the analytic Yee dispersion relation.

## Documentation

- [docs/methodology.md](docs/methodology.md): the discrete equations, the CFL derivation, the sponge and the source.
- [docs/verification.md](docs/verification.md): what each test does and how the absorber is measured.
- [docs/architecture.md](docs/architecture.md): the memory layout, the CUDA mapping and the throughput analysis.
