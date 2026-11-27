# Stage 6 protocol (version 2.1)

Version 2.1 corrects the definition of target T1 (section 8) and the sentence
about J in section 5. Both changes were made before any production run.

This protocol is fixed before any Stage 6 production trial. It replaces
`docs/stage6-benchmark-protocol.md` (version 1), which no production run used.
All results are for the 2D TMz model in natural units. They are not 3D
silicon-on-insulator results, and no design is fabrication-ready.

The gates and rules below do not change after a production result is seen. A
change makes a new protocol version, committed before the runs that use it.

## 1. Shared evaluator

- `PortSetup::standard(Geometry::Bend, scale)` with `extra_time = 6000 * scale`.
- Design box: `DesignBox::for_bend(setup, 80 * scale, 8 * scale)`.
- Training spectrum: the 15 Stage 5 band samples (lambda0 = 115 to 200 native
  cells, uniform in frequency), scale 1.
- Held-out spectrum: 61 samples, uniform in frequency over the same band, scale 1.
- Held-out grid: scale 2 (the half-cell grid), 15 samples.
- One reference run (`Run::Reference`) for each spectrum and grid. It is not
  charged to any method.
- IL and RL are the modal values of `PortSetup::sparams` (`s21_mode`, `s11_mode`).

Code: `src/robust.rs` (`Scorer`, `spectrum`, `variants`, `geometry`).

## 2. Bias rule (frozen)

- Three variants of a binary mask: eroded, nominal, and dilated.
- The eroded and dilated masks are the nominal box mask eroded or dilated with
  the L1 ball of radius `scale` cells. At scale 1 this is the 4-neighbour cross,
  which is the Stage 5 +-1 native cell check. At scale 2 the radius is 2 fine
  cells, which is the same physical bias.
- Only box cells change. The fixed exterior (the two slabs and the cladding) is
  the padding and is not biased. The ports and the reference run do not change.

## 3. Geometry rules (frozen)

All rules use 4-connectivity on the box with the fixed exterior attached.

- Binary: the scored mask is the 0.5 level set of the filtered latent density.
  For gradient designs, the binary fraction of the projected density before the
  threshold is also reported.
- Connected: the input slab connects to the output slab through silicon in the
  box, and no silicon component floats (touches neither slab).
- Width: an opening with a disk of diameter `5 * scale` cells changes no box
  cell (the second step uses the reflected disk).
- Gap: a closing with the same disk changes no box cell.
- No 1-cell islands of either phase.
- Enclosed air holes are allowed. Their number is reported.

The gates apply to the nominal mask at scale 1. For the eroded and dilated
masks, and at scale 2, width, gap, and connectivity are reported separately
but are not gates. The reported minimum width and minimum gap are the largest
disk diameters (1 to 12 * scale cells) for which the opening, and the closing,
change no box cell.

## 4. Performance gates

For the release design, on the training spectrum, the held-out spectrum, and
the held-out grid:

- worst-case IL over the three variants and all samples < 0.50 dB;
- worst-case RL over the three variants and all samples >= 22 dB.

The measured grid change is the scale-2 value minus the scale-1 value of the
worst-case IL and of the worst-case RL, for each variant. It is reported as a
measurement, not as an uncertainty bound.

## 5. Robust score

For a binary nominal mask that meets the geometry gates:

	J = max over variants (worst IL) + max(0, 22 - min over variants (worst RL))   [dB]

on the training spectrum. A mask that fails a geometry gate gets
J = 100 + (number of violations) and no FDTD run. Lower J is better. A mask
that meets the performance gates on the training spectrum has J < 0.50, but
J < 0.50 alone does not show RL >= 22 dB (for example, IL 0.30 dB with RL
21.9 dB gives J = 0.40). The performance gates are therefore tested on the
spectra, not on J.

## 6. Conventional baselines (phase 1)

These baselines are re-evaluated with sections 1 to 5, before any phase 1
optimiser run:

- the circular arc (`DesignBox::circular_arc`, centreline radius 64 cells);
- Euler bends, p = 0.1, 0.2, ..., 1.0 (`euler_centreline`);
- cubic Bezier bends, k = 0.30, 0.35, ..., 0.70 (`bezier_centreline`).

The best tested baseline is the baseline with the lowest J among the baselines
that meet the geometry gates. A win of the optimised design over the
conventional bends needs all of these:

- it meets every gate of sections 3 and 4;
- its robust worst-case IL is at least 0.10 dB lower than the robust
  worst-case IL of the best tested baseline, on the training spectrum, the
  held-out spectrum, and the held-out grid;
- that gain is larger than the absolute measured grid change of the
  worst-case IL of both designs (all three variants).

## 7. Phase 1 optimiser (practical, full resolution)

- Representation `Rep::Full`: one latent value for each of the 80 x 80 box cells.
- Conic filter radius r_f = 6 cells. Start: the circular arc mask.
- Relaxed objective (`robust::relaxed`): projection thresholds
  [eta_e, 0.5, eta_d]. eta_e and eta_d are calibrated once on the start design
  (`eta_for_area`): the filtered arc thresholded at eta_e (eta_d) has the same
  box area as the eroded (dilated) arc mask of section 2. For variant v, the
  loss terms are (1 - T_k) and c R_k, with c = 0.1 / 10^-2.2. F_v is their
  p-norm (p = 8) over the 15 samples, and F = max over v of F_v. The gradient is
  the gradient of the worst variant.
- Adam (beta1 0.9, beta2 0.999), learning rate 0.05, latent clipped to [0, 1].
- Beta continuation 1, 2, 4, 8, 16, 32, 64, 128, with 12 iterations each.
- Every 4 iterations, and after the last iteration, the binary nominal mask is
  scored with `Scorer` (section 5).
- The phase 1 design is the scored mask with the lowest J. Only the training
  spectrum takes part in this choice.
- Fallback (fixed now): if that design fails a gate of section 3 or 4 on the
  training spectrum, one more run uses r_f = 8 with no other change. Both runs
  are reported. A failure on the held-out spectrum or grid does not start a
  new run.

## 8. Phase 2 benchmark

### Shared parts

- Representation `Rep::Grid(20)`: 20 x 20 control values in [0, 1], bilinear
  interpolation to the 80 x 80 latent density, conic filter r_f = 6.
- Robust objective and binary score: section 5, with the `Scorer` cache.
- Initialisation set, encoded as 4 x 4 block means of the mask:
  I1 arc, I2 Bezier k = 0.40, I3 Euler p = 0.5, I4 Bezier k = 0.60,
  I5 Euler p = 1.0.
- Seeds 1, 2, 3, 4, 5. Seed i starts from init Ii, for every method.
- Cap: 1200 charged forward-equivalents for each run. Checkpoints at 150,
  300, 600 and 1200.
- Runs are sequential on one RTX 5090. Wall time is measured for each run.

### Methods

- Discrete adjoint. The relaxed objective of section 7 on `Rep::Grid(20)`,
  with eta_e and eta_d of section 7. Beta = 2, 4, 8, 16, 32, 64 with 6
  iterations each, then 64 until the cap. Adam with the tuned learning rate.
  The binary nominal mask is scored every 2 iterations.
- CMA-ES (Hansen defaults): n = 400, population 21, mu = 10, tuned sigma0. The
  full covariance is decomposed every generation.
- One-step PPO (a contextual bandit with no state). Gaussian policy over the
  400 controls, with a learned mean and a learned shared log standard
  deviation (start 0.10). 16 samples for each update. The advantage is the
  centred rank of -J. Clipped surrogate (epsilon 0.2), 4 Adam epochs for each
  batch, tuned learning rate. With one step and no state, this policy gradient
  is an evolution strategy. Multi-step PPO and GRPO are not part of this
  protocol.

### Cost accounting

- One forward solve is one FDTD run of the scale-1 setup. One adjoint solve is
  one FDTD run with a source on a port line.
- Charge: 12 forward-equivalents for each adjoint gradient, in addition to
  each forward solve that is not part of a gradient.
- One adjoint iteration takes 3 forward solves (one for each variant) and
  11 + 12 adjoint solves (the T gradient and the R gradient of the worst
  variant). Charge: 12 (T gradient, which includes the forward solve of the
  worst variant) + 12 (R gradient: 12 adjoint solves, and it uses the same
  forward solve, which is not charged again) + 2 (the forward solves of the
  other two variants) = 26. This is equal to the number of physical solves,
  so no solve is counted twice.
- A binary score costs 3 forward solves. A cache hit or a mask that fails a
  geometry gate costs 0.
- Forward solves and adjoint solves are also logged separately, with their
  wall time. A sensitivity view recomputes the adjoint cost with the measured
  ratio of adjoint-solve wall time to forward-solve wall time.

### Tuning allowance

Each method has two configurations. Each configuration has one tuning run:
seed 0, start = the sharp L (`DesignBox::sharp_l`, which is not in the
initialisation set), cap 300. The configuration with the lower best J at 300
is used for all production runs (if equal, the first one).

- Adjoint: learning rate 0.05 or 0.10.
- CMA-ES: sigma0 0.15 or 0.30.
- PPO: learning rate 0.02 or 0.05.

Tuning cost is reported separately from production cost.

### Reported values

- For each run: best feasible J against charged cost and against wall time,
  the fraction of candidates that meet the geometry gates, and the fraction
  that meet the performance gates.
- For each method: the five runs, the median, and the range.
- Time to target, as charged cost and wall time, for two targets:
  T1 = the first scored mask that meets the geometry gates and the performance
  gates of section 4 on the training spectrum, and
  T2 = J <= (J of the best tested baseline) - 0.10. A run that does not reach
  a target is reported as "not reached".
- Failed and infeasible runs stay in every plot and table.
- Finalists: the lowest-J design of each method over its five runs goes
  through the held-out spectrum and grid.
- No method is declared a winner unless its range at the cap does not
  overlap the range of the other method. If no method reaches T1 in its
  median run, the comparison is reported as inconclusive for this budget.

## 9. Compute and storage caps

- Phase 1: at most 2 optimiser runs of at most 3000 charged forward-equivalents.
- Phase 2: tuning 6 x 300, production 15 x 1200.
- Total GPU wall time for phases 1 and 2: at most 10 hours. If a cap stops the
  work, the evidence that exists is reported and the comparison is marked
  incomplete.
- Storage: at most 3 GB under `out/stage6`. Committed raw logs: at most 100 MB.

## 10. Layout export

- Manhattan cell-edge contours, exact integer coordinates in native cells.
- Layers: 1/0 nominal silicon, 2/0 eroded silicon, 3/0 dilated silicon,
  10/0 design-box outline. Each layer has the box and a 16-cell halo of the
  fixed slabs.
- Origin: the lower-left corner of the halo. x to the right, y up. Database
  unit 1 nm, user unit 1 um. The mapping 1 native cell = 10 nm is
  illustrative and not calibrated to a physical process.
- An independent reader (`tools/check_layout.py`, numpy only, no code shared
  with the Rust writer) re-imports the GDSII. It checks the exact union area,
  zero symmetric difference against the source mask, and the width, gap, and
  connectivity rules on the imported layout. A negative control (the same mask
  shifted by one cell, equal area) must fail the symmetric-difference check.
- A JSON certificate records the file hashes, the mask and solver revisions,
  the units, the rule definitions, the check results, and the command. Its
  claim is "layout-verified for the stated 2D rules". It is not a PDK signoff.
