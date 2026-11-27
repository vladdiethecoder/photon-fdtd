## v0.6.0-rc1 (release candidate)

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
  reference, golden vectors and a self-checking testbench; Verilator 5.046:
  0 mismatches on both cases, a rounding mutant fails.
- Plots, field movie, project page and release tooling (`tools/`).

Results
- Release design (adjoint, shared 20 x 20 representation, seed 5): robust
  worst-case IL 0.1718 dB, RL 22.99 dB on 61 held-out wavelengths (0.1688 dB,
  23.24 dB on the half-cell grid); passes every gate and beats the best tested
  conventional bend (Bezier k = 0.45, 0.3306 dB) by 0.1595 / 0.1588 / 0.1611 dB
  (training / held-out / half-cell grid). The protocol did not name the release
  design; it is the lowest training-set J among the designs that pass every gate.
- Full-resolution phase 1 design (`bend80_phase1.gds`, kept as the record of
  the unconstrained-grid failure mode): robust IL 0.2737 dB, RL 22.31 dB
  held-out; passes every gate; no win (gain 0.057 dB, below the
  pre-registered 0.10 dB).
- Failure mode: 23 of 25 scored masks of the full-resolution run broke the
  5-cell rule; the final relaxed iterate (0.0877 dB) is not feasible.
- Benchmark (5 seeds, cap 1200 charged): adjoint median best J 0.2166 dB
  (0.1711 to 0.2949), T1 5/5, T2 3/5; CMA-ES median 2.0659 dB (0.8128 to
  3.9520), T1 0/5; one-step PPO median 0.3766 dB (0.3261 to 0.5371), T1 4/5,
  T2 0/5, but every PPO run stopped at the 20000-candidate rule after 153 to
  396 charged. Adjoint leads CMA-ES (ranges do not overlap); comparisons with
  PPO are inconclusive at equal budget.
- Layout certificates (release design and phase 1 design): pass
  ("layout-verified for the stated 2D rules"); the biased masks of the release
  design have 4 width (eroded) and 2 width + 2 gap (dilated) violations, which
  are reported, not gated.
- Clean-checkout receipts: `release/receipts/receipts.json`.

Not verified
- No synthesis figures (no synthesis tool).
- No Lean proof in this release candidate (optional).


Assets
- `bend80.gds`, `bend80.svg`, `bend80_certificate.json`: release design (adjoint, 20 x 20 grid, seed 5).
- `bend80_phase1.gds`, `bend80_phase1_certificate.json`: full-resolution phase 1 design, the record of the unconstrained-grid failure mode.
- `spectra_heldout61.png`, `tradeoff.png`: the two lead plots. `SHA256SUMS`: checksums of the release files.

Disclosures
- The protocol did not name the release design in advance. It was chosen after the results by the lowest training-set robust J among the designs that pass every gate; the choice uses no held-out data.
- Every one-step PPO run ended at the 20000-candidate stop rule (153 to 396 of 1200 charged solves), so comparisons with PPO are inconclusive.
- 2D TMz model in natural units. Not a 3D silicon-on-insulator result, not fabrication-ready, not a PDK signoff; the layout scale (1 cell = 10 nm) is illustrative.
