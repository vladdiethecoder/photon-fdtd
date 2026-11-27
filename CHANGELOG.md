# Changelog

All results are for the 2D TMz model in natural units. No entry claims 3D
silicon-on-insulator performance or fabrication readiness.

## v0.6.0-rc1 (release candidate, not yet tagged)

Robust 80 x 80 bend, layout export with an independent check, optimiser
benchmark, project page, and a fixed-point RTL stencil. The protocol
`docs/stage6-protocol.md` (v2.1) was committed before any production run.

Added
- Port-1 reflection adjoint (`gradient_r`), verified against finite
  differences: cosine 1.00000000, magnitude ratio 1.000000, 16/16 signs,
  directional ratios within 5e-8 (`adjoint_reflection_gradient_matches_finite_differences`).
- Shared robust evaluator (`src/robust.rs`): frozen +-1-cell bias rule,
  connectivity, 5-cell width and gap rules, robust binary score, relaxed
  three-field objective with T and R terms.
- Optimisers on one evaluator: discrete adjoint (full or 20 x 20 grid),
  CMA-ES, one-step PPO (`stage6_adjoint`, `stage6_bb`), with cost accounting
  that logs forward and adjoint solves separately.
- Held-out validation (`stage6_validate`): 61 wavelengths and the half-cell
  grid, every variant, measured grid change, win test.
- Layered GDSII and SVG export (`export_layout`) and the independent checker
  and certificate (`tools/check_layout.py`).
- Fixed-point 1D Yee stencil in SystemVerilog (`rtl/`) with a bit-exact Rust
  reference, golden vectors and a self-checking testbench (not yet simulated).
- Plots, field movie, project page and release tooling (`tools/`).

Results
- Phase 1 design: robust worst-case IL 0.2737 dB, RL 22.31 dB on 61 held-out
  wavelengths (0.2724 dB, 22.59 dB on the half-cell grid); passes every gate.
- No win: the gain over the best tested conventional bend (Bezier k = 0.45,
  0.3306 dB) is 0.057 dB, below the pre-registered 0.10 dB.
- Failure mode: 23 of 25 scored masks of the phase 1 run broke the 5-cell rule;
  the final relaxed iterate (0.0877 dB) is not feasible.
- Layout certificate: passes ("layout-verified for the stated 2D rules").
- Benchmark: BENCHMARK_CHANGELOG

Not verified
- The RTL has not been simulated (no HDL simulator installed).
- No Lean proof (Lean is not installed; optional).

## v0.5.0 (2026-11-26)

Adjoint inverse design of the 48 x 48 bend. DFT adjoint of the modal bend
transmission, verified against finite differences (per cell, directional over
4 step sizes, full chain). Optimised worst-case IL 0.2586 dB, RL 23.36 dB over
61 frequencies; best conventional baseline (Bezier k = 0.40) 0.5933 dB; gain
0.335 dB, grid uncertainty 0.0073 dB. 1-cell erosion: IL 0.311 dB, RL 20.40 dB.
`cargo test --release`: 20 passed.

## v0.4.0 (2026-11-26)

Discrete Yee eigenmode source, on-GPU DFT port monitors, flux and modal
S-parameters. Straight guide: modal |S11|^2 + |S21|^2 = 1 within 4.3e-6.
Guided mode into the 10-cell CPML: -83.4 dB or lower. Sharp 90-degree bend at
lambda0 = 146 cells: modal IL 3.748 dB, RL 6.69 dB. 14 tests.

## v0.3.0 (2026-11-26)

Optional interactive viewer (`--features gui`) with CUDA-GL interop
(0.266 ms per frame). The core library has no GUI dependency. 11 tests.

## v0.2.0 (2026-11-26)

CFS-CPML replaces the conductive sponge. At 10 cells: normal and corner
reflection -87.7 dB or lower, grazing -66.8 dB or lower. 10 tests.

## v0.1.0 (2026-11-26)

2D TMz Yee FDTD in f64, natural units; CPU (rayon) and CUDA sm_120 backends
with bitwise identical fields. 7 tests.
