# Architecture

This document describes the data layout, the CPU and CUDA execution, and the measured throughput.

## Modules

| File | Content |
|---|---|
| `src/lib.rs` | `Yee2d` (CPU solver), `check_setup`, `cb_coeff`, `gaussian_sine`, `gaussian_derivative` |
| `src/cpml.rs` | `Pml` (parameters), `CpmlLayout` (1D tables and strip index maps), `Profile` |
| `src/gpu.rs` | `GpuYee` (CUDA solver) and the kernel source |
| `src/verify.rs` | The scenarios for the binaries and the tests: `Strip`, `AbsorberCheck` |
| `src/bin/waveguide.rs` | The silicon strip demonstration and `--check-pml` |
| `src/bin/bench.rs` | The throughput measurement |
| `src/bin/viewer.rs` | The interactive viewer. It builds only with `--features gui`. |
| `examples/viewer_demo.script` | A scripted viewer session that uses every control |
| `tests/verification.rs` | The verification suite |

## Memory layout

Each full-grid array is one flat `Vec<f64>` (CPU) or `CudaSlice<f64>` (GPU) of nx * ny values. The layout is row-major with k = j * nx + i, so each row of x values is contiguous. The solver has no structs of fields and no padding.

| Array | Size | Read by | Written by |
|---|---|---|---|
| `ez` | nx * ny | H pass, E pass | E pass, source |
| `hx`, `hy` | nx * ny | H pass, E pass | H pass |
| `cb` | nx * ny | E pass | constructor |
| `eps_r` (CPU only) | nx * ny | energy diagnostics | constructor |
| `psi_ezx`, `psi_hyx` | ny * 2W | x-strip passes | x-strip passes |
| `psi_ezy`, `psi_hxy` | 2W * nx | y-strip passes | y-strip passes |
| CPML tables (1 / kappa, a, b) | nx or ny each | all passes | constructor |

W = N + 1 is the strip width. For a 4096^2 grid with N = 10, the four psi arrays have 4 * 4096 * 22 = 360448 values. This is 2.1 % of one full-grid array. The 12 tables have 12 * 4096 values.

The main passes read `ez`, `hx`, `hy` and `cb` and write `ez`, `hx` and `hy`. The 1D tables are small and stay in the cache. Outside the strips, the H pass has 3 reads and 2 writes, and the E pass has 4 reads and 1 write. The minimum DRAM traffic is thus 10 x 8 = 80 bytes per cell per step. `bench` uses this value for its effective GB/s figure. The Stage 1 sponge needed 104 bytes per cell, because it also had the per-cell arrays `ca` and `hb`.

## One layout for both backends

`CpmlLayout::new` calculates all CPML data on the host: four 1D profiles (x and y, integer and half positions) and the strip width W. `strip_i(c)`, `strip_j(r)` and `strip_row(j)` map between strip indices and grid indices. The CPU solver holds the layout. `GpuYee` receives a copy, uploads the same tables, and passes nx, ny and W to the kernels. The kernels use the same index formula as `strip_i` and `strip_j`. Thus the two backends cannot disagree on a strip cell or a corner cell.

## CPU execution

`step_serial` and `step_parallel` call the same row functions, `h_row` and `e_row`. Each row function does the main update of its row first. Then it applies the x-strip correction for the 2W strip columns of that row. If the row is in a y strip, it then applies the y-strip correction for the whole row.

Each row function cuts its slices to one length first. The compiler can then remove the bounds checks and vectorize the main loop. The build uses `-C target-cpu=native` (in `.cargo/config.toml`), which enables AVX-512 on the Ryzen 9 7900X.

`step_parallel` uses rayon `par_chunks_mut(nx)`: one task gets at least 4 rows. Each task also gets the x-strip psi row of its grid row and, in a y strip, the y-strip psi row (`y_strip_rows`). Each pass has one writer for each row, so the borrow checker accepts the code without locks or `unsafe`. Each cell gets the same operations in the same order, so the thread count does not change the result.

## CUDA execution

One step is six kernel launches in one stream: `update_h`, `psi_h_x`, `psi_h_y`, `update_e`, `psi_e_x` and `psi_e_y`. The launch order gives each cell the same operation order as the CPU code. It also prevents races between the passes.

- The main kernels have one thread for each cell, in blocks of 256 x 1 threads along x. A warp reads 32 contiguous f64 values from each array. The grid is ceil(nx / 256) x min(ny, 65535) blocks, with a loop over the rows that has a stride of gridDim.y.
- The x-strip kernels have one thread for each strip column, in blocks of 32 threads, with the same row loop.
- The y-strip kernels have one block row for each of the 2W strip rows.

Offsets are `size_t`, calculated as `(size_t)j * (size_t)nx + (size_t)i`. The kernels use no shared memory and no tiling. The L1 and L2 caches supply the neighbour values.

## Viewer boundary

The library (`src/lib.rs`, `src/cpml.rs`, `src/gpu.rs`, `src/verify.rs`) has no GUI dependency. The `gui` feature adds `eframe` and `libloading`, and only `src/bin/viewer.rs` uses them (`required-features = ["gui"]`). `cargo tree -e normal` without the feature lists no eframe, egui, winit or glow crate. Thus `cargo test --release` needs no display server.

The library gives the viewer five headless functions on `GpuYee`:

| Function | Kernel | Effect |
|---|---|---|
| `paint_disk(cx, cy, r, eps)` | `paint_disk` | Sets `cb` in a disk to `cb_coeff(dt, eps)`, calculated on the host |
| `render_rgba_to_ptr(ptr, gain)` | `render_rgba` | Writes the RGBA8 heat map to any device address |
| `render_rgba(&mut buf, gain)` | `render_rgba` | The same, into a `CudaSlice<u32>` |
| `reset_fields()` | memset | Sets the fields and the psi arrays to zero. `cb` stays. |
| `stream()` | none | The CUDA stream, for the interop calls |

A painted cell holds the same `cb` bits as a cell that a constructor builds. Thus the CPU and GPU parity rules also apply to painted geometry.

### Viewer frame

One frame has these steps:

1. `advance` enqueues the solver steps and the source updates.
2. The viewer waits for the stream. Then `cuGraphicsMapResources` maps the GL pixel buffer, `render_rgba_to_ptr` writes the image into it, and `cuGraphicsUnmapResources` unmaps it. The unmap orders the CUDA work before the GL use.
3. `glTexSubImage2D` copies the pixel buffer into the texture on the GPU.
4. egui draws the texture as a native glow texture, and the panel widgets.

The mapping needs a GL context on the same GPU as CUDA. The host has two GPUs (the RTX 5090 and a Radeon 610M). The GL context of the viewer runs on the RTX 5090 (GL renderer `NVIDIA GeForce RTX 5090/PCIe/SSE2`). If `cuGraphicsGLRegisterBuffer` fails, the viewer copies the image through the host instead.

## Throughput

The data comes from `docs/results/bench.log` (commit 921ccb3, f64, median of 3 trials).

| Grid | Footprint (4 arrays) | CPU 1 thread | CPU 24 threads | RTX 5090 |
|---|---|---|---|---|
| 512^2 | 8 MiB | 1192.6 | 837.3 | not run |
| 1024^2 | 32 MiB | 631.9 | 921.5 | 28629.5 |
| 2048^2 | 128 MiB | 368.9 | 661.2 | not run |
| 2056^2 | 129 MiB | 450.3 | 665.1 | not run |
| 4096^2 | 512 MiB | 225.4 | 622.8 | 15561.7 |
| 4104^2 | 514 MiB | 438.5 | 651.1 | not run |
| 8192^2 | 2 GiB | not run | not run | 15621.1 |
| 16384^2 | 8 GiB | not run | not run | 15654.2 |

All values are in million cell updates per second (Mcells/s).

### GPU: memory bandwidth limits the update

For grids of 4096^2 and larger, the RTX 5090 gives 15.56 to 15.65 Gcells/s. This rate is 1245 to 1252 GB/s of the 80-byte minimum traffic. The NVIDIA specification gives 1792 GB/s for the GDDR7 bus, so the kernels use about 70 % of the peak bandwidth. With the Stage 1 sponge, the kernels used about 75 % of the peak. The four strip kernels add launches and strided accesses. This probably causes the lower percentage, but it is not measured.

The 1024^2 grid (32 MiB) fits in the L2 cache, and its effective rate of 2290 GB/s is larger than the DRAM bandwidth. The rate decreased from 41552.9 Mcells/s (Stage 1, two launches per step) to 28629.5 Mcells/s. At this size, one step takes about 37 us, so the cost of six launches is a large part of the step. This explanation is a hypothesis.

The f64 arithmetic rate of the RTX 5090 is low, but each cell needs only about 15 floating-point operations for 80 bytes. Thus the arithmetic rate does not limit the update.

### CPU with 24 threads: memory bandwidth limits the update

From 2048^2 to 4104^2, the rayon rate is 622.8 to 665.1 Mcells/s. This rate is 50 to 53 GB/s of the minimum traffic. Each store also reads its cache line first (read for ownership). Thus the real traffic is about 13 x 8 = 104 bytes per cell, or about 65 to 69 GB/s. The DDR5-6000 dual-channel peak is 96 GB/s.

The 2048^2 and 4096^2 rates are almost the same (661.2 and 622.8 Mcells/s). The grid size thus has only a small effect on the parallel rate when the arrays do not fit in the caches.

At 512^2, the arrays (8 MiB) fit in the caches. There, 24 threads give 837.3 Mcells/s, which is less than the 1192.6 Mcells/s of one thread. The hypothesis is that the work per task is small relative to the rayon scheduling cost and the wait at the end of each pass. This hypothesis is not measured.

### CPU with one thread: row strides that are powers of two are slow

With one thread, the row stride has a large effect:

- 2048^2 gives 368.9 Mcells/s, and 2056^2 gives 450.3 Mcells/s.
- 4096^2 gives 225.4 Mcells/s, and 4104^2 gives 438.5 Mcells/s.

Eight more cells in each row (64 bytes) increase the serial rate by 1.2 times at 2048 and by 1.9 times at 4096. The footprint changes by less than 1 %. Thus the footprint does not cause the difference. The row stride causes it.

Hypothesis, not measured with hardware counters: if nx is a multiple of 512, the row stride is a multiple of 4096 bytes. Then the streams `ez[k]`, `ez[k + nx]`, `hx[k]` and `hx[k - nx]` map to the same L1 cache sets. They also have the same low 12 address bits, which causes false load-store dependencies (4K aliasing). A row stride with padding removes both effects. The parallel rate does not show the effect, because DRAM bandwidth limits it first.

The solver does not pad rows. To get the best single-thread rate, select an nx that is not a multiple of 512. Padding the row stride inside `Yee2d` is a possible change. It does not improve the multi-thread rate, which DRAM bandwidth limits.
