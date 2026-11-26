# Architecture

This document describes the data layout, the CPU and CUDA execution, and the measured throughput.

## Modules

| File | Content |
|---|---|
| `src/lib.rs` | `Yee2d` (CPU solver), `check_setup`, `cell_coeffs`, `gaussian_sine` |
| `src/gpu.rs` | `GpuYee` (CUDA solver) and the kernel source |
| `src/verify.rs` | The scenarios for the binaries and the tests: `Strip`, `AbsorberCheck` |
| `src/bin/waveguide.rs` | The silicon strip demonstration and `--check-pml` |
| `src/bin/bench.rs` | The throughput measurement |
| `tests/verification.rs` | The verification suite |

## Memory layout

Each field and each coefficient is one flat `Vec<f64>` (CPU) or `CudaSlice<f64>` (GPU) of nx * ny values. The layout is row-major with k = j * nx + i, so each row of x values is contiguous. The solver has no structs of fields and no padding.

| Array | Read by | Written by |
|---|---|---|
| `ez` | H pass, E pass | E pass, source |
| `hx`, `hy` | H pass, E pass | H pass |
| `ca` | H pass, E pass | constructor |
| `cb` | E pass | constructor |
| `hb` | H pass | constructor |
| `eps_r` (CPU only) | energy diagnostics | constructor |

The solver stores coefficients for each cell and does not branch on the material in the update. One step thus reads 10 arrays and writes 3 arrays. Each array is read and written once, if the neighbour values come from the cache. The minimum DRAM traffic is 13 x 8 = 104 bytes per cell per step. `bench` uses this value for its effective GB/s figure.

## CPU execution

`step_serial` and `step_parallel` call the same row functions, `h_row` and `e_row`. Each row function receives row slices and cuts them to one length first. The compiler can then remove the bounds checks and vectorize the inner loop. The build uses `-C target-cpu=native` (in `.cargo/config.toml`), which enables AVX-512 on the Ryzen 9 7900X.

`step_parallel` uses rayon `par_chunks_mut(nx)`: one task gets at least 4 rows. The H pass writes `hx` and `hy` and reads `ez`. The E pass writes `ez` and reads `hx` and `hy`. Each pass has one writer for each row, so the borrow checker accepts the code without locks or `unsafe`. Each cell gets the same operations in the same order, so the thread count does not change the result.

## CUDA execution

The kernels have one thread for each cell. Blocks are 256 x 1 threads along x, so a warp reads 32 contiguous f64 values from each array. The grid is ceil(nx / 256) x min(ny, 65535) blocks. Each thread loops over the rows with a stride of gridDim.y, so ny can be larger than 65535. Offsets are `size_t`, calculated as `(size_t)j * (size_t)nx + (size_t)i`.

One step is two kernel launches, `update_h` and `update_e`, in one stream. The launch order prevents the race between the H and E passes. The kernels use no shared memory and no tiling. The L1 and L2 caches supply the neighbour values (`ez[k + 1]`, `ez[k + nx]`, `hy[k - 1]` and `hx[k - nx]`).

## Throughput

The data comes from `docs/results/bench.log` (commit 160cfbb, f64, median of 3 trials).

| Grid | Footprint (6 arrays) | CPU 1 thread | CPU 24 threads | RTX 5090 |
|---|---|---|---|---|
| 512^2 | 12 MiB | 1195.9 | 966.6 | not run |
| 1024^2 | 48 MiB | 464.7 | 687.7 | 41552.9 |
| 2048^2 | 192 MiB | 304.3 | 491.5 | not run |
| 2056^2 | 194 MiB | 430.5 | 511.7 | not run |
| 4096^2 | 768 MiB | 221.2 | 498.1 | 12959.7 |
| 4104^2 | 771 MiB | 422.0 | 506.5 | not run |
| 8192^2 | 3 GiB | not run | not run | 12951.0 |
| 16384^2 | 12 GiB | not run | not run | 12927.4 |

All values are in million cell updates per second (Mcells/s).

### GPU: memory bandwidth limits the update

For grids of 4096^2 and larger, the RTX 5090 gives 12.93 to 12.96 Gcells/s. This rate is 1344 to 1348 GB/s of the 104-byte minimum traffic. The NVIDIA specification gives 1792 GB/s for the GDDR7 bus, so the kernels use about 75 % of the peak bandwidth. The 1024^2 grid (48 MiB) fits in the L2 cache, and its effective rate of 4322 GB/s is larger than the DRAM bandwidth. A rate above the DRAM bandwidth is possible only from the cache, so the 1024^2 value is not a DRAM figure.

The f64 arithmetic rate of the RTX 5090 is low, but each cell needs only 14 floating-point operations for 104 bytes. Thus the arithmetic rate does not limit the update.

### CPU with 24 threads: memory bandwidth limits the update

From 2048^2 to 4104^2, the rayon rate is 491.5 to 511.7 Mcells/s. This rate is 51 to 53 GB/s of the minimum traffic. Each store also reads its cache line first (read for ownership). Thus the real traffic is about 16 x 8 = 128 bytes per cell, or about 64 GB/s. The DDR5-6000 dual-channel peak is 96 GB/s.

The 2048^2 and 4096^2 rates are almost the same (491.5 and 498.1 Mcells/s). An earlier measurement gave 547 and 517 Mcells/s. That measurement used one trial each and the kernel without the H loss term (96 bytes per cell). The earlier difference is thus not a property of the grid size.

At 512^2, the arrays (12 MiB) fit in the caches. There, 24 threads give 966.6 Mcells/s, which is less than the 1195.9 Mcells/s of one thread. The hypothesis is that the work per task is small relative to the rayon scheduling cost and the wait at the end of each pass. This hypothesis is not measured.

### CPU with one thread: row strides that are powers of two are slow

With one thread, the row stride has a large effect:

- 2048^2 gives 304.3 Mcells/s, and 2056^2 gives 430.5 Mcells/s.
- 4096^2 gives 221.2 Mcells/s, and 4104^2 gives 422.0 Mcells/s.

Eight more cells in each row (64 bytes) increase the serial rate by 1.4 times at 2048 and by 1.9 times at 4096. The footprint changes by less than 1 %. Thus the footprint does not cause the difference. The row stride causes it.

Hypothesis, not measured with hardware counters: if nx is a multiple of 512, the row stride is a multiple of 4096 bytes. Then the streams `ez[k]`, `ez[k + nx]`, `hx[k]` and `hx[k - nx]` map to the same L1 cache sets. They also have the same low 12 address bits, which causes false load-store dependencies (4K aliasing). A row stride with padding removes both effects. The parallel rate does not show the effect, because DRAM bandwidth limits it first.

The solver does not pad rows. To get the best single-thread rate, select an nx that is not a multiple of 512. Padding the row stride inside `Yee2d` is a possible change. It does not improve the multi-thread rate, which DRAM bandwidth limits.
