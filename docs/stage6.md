# Stage 6: robust 80 x 80 bend, layout export, optimiser benchmark

All results are for the 2D TMz model in natural units. They are not 3D
silicon-on-insulator results, and no design is fabrication-ready. The rules,
gates, seeds, budgets and caps are in [stage6-protocol.md](stage6-protocol.md)
(version 2.1, committed before the production runs). The step-by-step record
is in [stage6-ledger.md](stage6-ledger.md).

## Phase 1: the layout loop

### Result

The release design passes every gate, but it does not beat the best
conventional bend by the required margin.

| Design | Robust IL / RL, training (15) | Robust IL / RL, held-out (61) | Robust IL / RL, half-cell grid | Gates |
|---|---|---|---|---|
| Phase 1 design (r_f 6) | 0.2737 / 22.50 dB | 0.2737 / 22.31 dB | 0.2724 / 22.59 dB | pass |
| Best tested baseline: Bezier k = 0.45 | 0.3306 / 23.21 dB | 0.3306 / 23.00 dB | 0.3299 / 23.16 dB | pass |
| Circular arc (radius 64) | 0.3625 / 25.25 dB | 0.3625 / 24.99 dB | 0.3614 / 25.96 dB | pass |

"Robust" is the worst case over the eroded, nominal and dilated masks and over
all wavelengths of the set. The gates are IL < 0.50 dB and RL >= 22 dB.

Win test (protocol section 6): the gain over Bezier k = 0.45 is 0.0569 dB
(training), 0.0569 dB (held-out) and 0.0575 dB (half-cell grid). The protocol
requires at least 0.10 dB on all three. Result: **no win**.

Source: `docs/results/stage6/phase1/validate/validate.json`, `validate.log`.

### Each variant and the measured grid change

| Design | Variant | Training IL / RL | Held-out IL / RL | Half-cell IL / RL | Grid change IL | Grid change RL |
|---|---|---|---|---|---|---|
| Phase 1 design | eroded | 0.2737 / 27.98 | 0.2737 / 27.86 | 0.2724 / 26.67 | -0.0013 dB | -1.31 dB |
| | nominal | 0.2006 / 28.78 | 0.2006 / 28.60 | 0.1999 / 29.05 | -0.0007 dB | +0.27 dB |
| | dilated | 0.1799 / 22.50 | 0.1799 / 22.31 | 0.1795 / 22.59 | -0.0004 dB | +0.09 dB |
| Bezier k = 0.45 | eroded | 0.3306 / 25.76 | 0.3306 / 25.75 | 0.3299 / 27.93 | -0.0007 dB | +2.18 dB |
| | nominal | 0.2586 / 31.10 | 0.2586 / 31.07 | 0.2589 / 32.83 | +0.0002 dB | +1.74 dB |
| | dilated | 0.2445 / 23.21 | 0.2445 / 23.00 | 0.2471 / 23.16 | +0.0026 dB | -0.05 dB |
| Circular arc | eroded | 0.3625 / 25.25 | 0.3625 / 24.99 | 0.3614 / 26.13 | -0.0012 dB | +0.88 dB |
| | nominal | 0.2580 / 40.91 | 0.2580 / 40.89 | 0.2578 / 41.81 | -0.0001 dB | +0.91 dB |
| | dilated | 0.2166 / 25.73 | 0.2166 / 25.73 | 0.2180 / 25.96 | +0.0014 dB | +0.22 dB |

The grid change is the half-cell value minus the native-grid value. It is a
measurement, not an uncertainty bound. The worst-case IL of every design is at
lambda0 = 200 cells, the long-wavelength band edge. That point is in both the
training and the held-out sets, so their worst-case IL values are equal. The
nominal arc gives 0.2580 dB, which agrees with the reported 0.258 dB of the
Stage 5 arc at 80 x 80.

### Geometry

All three designs meet the geometry gates on the nominal mask at scale 1: the
ports are connected, no silicon floats, there are no 1-cell islands, and the
5-cell width and gap checks change no cell. The width, gap and connectivity of
every variant at both scales are in `docs/results/stage6/phase1/validate/geometry.csv`.

### Failure mode: the length-scale control did not hold

The phase 1 design is the scored mask of iteration 3 of 96. 23 of the 25
scored masks of the run failed the geometry gates, by 7 to 36 violations, and the
ports stayed connected in all of them. The relaxed objective kept improving
(F fell from 0.047 to 0.014), and the projection reached a binary fraction of
0.980 at beta 128. But the 0.5 level set of the filtered density had features
thinner than 5 cells.

A post-hoc diagnostic of the final relaxed iterate shows the size of the gap.
It is not a candidate and is not part of any gate.

| | Robust IL / RL, held-out | Nominal geometry |
|---|---|---|
| Final relaxed iterate (diagnostic) | 0.0877 / 24.41 dB | fails: 1 floating silicon component, 8 width and 11 gap violations, minimum width 2 cells, minimum gap 3 cells |
| Phase 1 design | 0.2737 / 22.31 dB | passes |

The three-field robust formulation, with the thresholds calibrated to a
1-cell bias, did not enforce the 5-cell rule at r_f = 6 in this setup. An
explicit length-scale constraint is the next step to try. It is not part of
this protocol.

Source: `docs/results/stage6/phase1/rf6/events.csv`, `docs/results/stage6/phase1/diagnostic_final/`.

### Cost

| | Value |
|---|---|
| Charged forward-equivalents | 2502 (cap 3000) |
| Forward solves | 294 (288 for the relaxed variants, 3 for each of 96 iterations; 6 for the 2 binary scores that met the geometry gates) |
| Adjoint solves | 2208 (96 iterations x 23) |
| Wall time | 1596.8 s on one RTX 5090 |
| Measured wall time for each solve | forward 0.72 s, adjoint 0.63 s |

## Layout export and certificate

`release/bend80.gds` and `release/bend80.svg` hold the phase 1 design with a
16-cell halo of the fixed slabs.

| Item | Value |
|---|---|
| Layers | 1/0 nominal silicon, 2/0 eroded silicon, 3/0 dilated silicon, 10/0 design-box outline |
| Frame | origin at the lower-left corner of the halo, x right, y up |
| Units | database unit 1 nm, user unit 1 um, 1 cell = 10 nm (illustrative, not calibrated) |
| Contours | Manhattan, on cell edges, exact integer coordinates; holes by zero-width keyhole cuts |

The independent reader `tools/check_layout.py` (numpy only, no code shared
with the Rust writer) re-imports the GDSII file:

| Layer | Union area (cells) | Source cells | Symmetric difference | Width viol. | Gap viol. | Ports connected | Floating Si |
|---|---|---|---|---|---|---|---|
| 1 nominal | 2218 | 2218 | 0 | 0 | 0 | yes | 0 |
| 2 eroded | 2037 | 2037 | 0 | 0 | 0 | yes | 0 |
| 3 dilated | 2399 | 2399 | 0 | 0 | 0 | yes | 0 |

Negative control: the nominal source mask shifted by one cell (equal area,
2218 cells) gives a symmetric difference of 176 cells, and the check fails as
it must. A GDSII file with one corrupted coordinate also fails (27 differing
cells). The certificate `release/bend80_certificate.json` records the file
hashes, the revisions, the units, the rule definitions, the check results and
the command. Its claim is "layout-verified for the stated 2D rules". It is not
a PDK signoff.

Regenerate:

	cargo run --release --bin export_layout -- --mask docs/results/stage6/phase1/validate/phase1_r_f_6_s1.mask --out release/bend80
	tools/check_layout.py release/bend80 --design-revision REV --cert release/bend80_certificate.json

## Phase 2: optimiser benchmark

All methods use the shared parts of protocol section 8: the 20 x 20 control grid, the initialisation set I1 to I5 (seed i starts from Ii), the robust binary score, the geometry gates, the `Scorer` cache, and a cap of 1200 charged forward-equivalents. The runs ran one after another on one RTX 5090.

![Optimiser trade-offs](../release/tradeoff.svg)

### Tuning allowance (reported separately)

| Configuration | Best J at 300 (dB) | Charged | Wall time |
|---|---|---|---|
| adjoint_lr0.05 | 18.0850 | 289 | 184 s |
| adjoint_lr0.10 | 19.6125 | 289 | 185 s |
| cma_s0.15 | 21.2300 | 300 | 256 s |
| cma_s0.30 | 15.0296 | 300 | 257 s |
| ppo_lr0.02 | 21.9393 | 300 | 247 s |
| ppo_lr0.05 | 20.0089 | 300 | 247 s |

Chosen by the rule of section 8: adjoint learning rate 0.05, CMA-ES sigma0 0.3, PPO learning rate 0.05. Tuning cost: 1778 charged, 1377 s.

### Production runs

| Run | Best J (dB) | Charged | Forward solves | Adjoint solves | Cache hits | Wall time | Geometry-feasible candidates | Gate-passing candidates | T1 (charged / min) | T2 (charged / min) |
|---|---|---|---|---|---|---|---|---|---|---|
| adjoint 1 | 0.2074 | 1181 | 192 | 989 | 0 | 762 s | 21 / 22 | 15 | 110 / 1.2 | 822 / 8.8 |
| adjoint 2 | 0.2166 | 1181 | 192 | 989 | 0 | 766 s | 21 / 22 | 20 | 55 / 0.6 | 932 / 10.1 |
| adjoint 3 | 0.2612 | 1194 | 159 | 1035 | 0 | 770 s | 8 / 23 | 4 | 110 / 1.2 | - |
| adjoint 4 | 0.2949 | 1188 | 153 | 1035 | 0 | 761 s | 6 / 23 | 5 | 55 / 0.6 | - |
| adjoint 5 | 0.1711 | 1200 | 165 | 1035 | 0 | 774 s | 10 / 22 | 8 | 110 / 1.2 | 807 / 8.7 |
| cma 1 | 2.9650 | 1200 | 1200 | 0 | 0 | 1144 s | 400 / 5130 | 0 | - | - |
| cma 2 | 3.9520 | 1200 | 1200 | 0 | 0 | 1186 s | 400 / 6545 | 0 | - | - |
| cma 3 | 1.8578 | 1200 | 1200 | 0 | 0 | 1048 s | 400 / 1882 | 0 | - | - |
| cma 4 | 2.0659 | 1200 | 1200 | 0 | 0 | 1077 s | 400 / 2890 | 0 | - | - |
| cma 5 | 0.8128 | 1200 | 1200 | 0 | 0 | 1058 s | 400 / 2222 | 0 | - | - |
| ppo 1 | 0.3261 | 213 | 213 | 0 | 0 | 206 s | 71 / 20000 | 9 | 3 / 0.0 | - |
| ppo 2 | 0.3310 | 153 | 153 | 0 | 0 | 156 s | 51 / 20000 | 1 | 18 / 0.2 | - |
| ppo 3 | 0.4770 | 396 | 396 | 0 | 0 | 356 s | 132 / 20000 | 4 | 24 / 0.3 | - |
| ppo 4 | 0.3766 | 261 | 261 | 0 | 0 | 245 s | 87 / 20000 | 9 | 3 / 0.0 | - |
| ppo 5 | 0.5371 | 309 | 309 | 0 | 0 | 285 s | 103 / 20000 | 0 | - | - |

Production total: 13276 charged, 10594 s (2.94 h). T1: the first scored mask that meets every gate on the training spectrum. T2: J <= 0.2306 dB (best tested baseline J 0.3306 dB minus 0.10 dB).

| Method | Median best J (dB) | Range (dB) | Charged per run | T1 | T2 |
|---|---|---|---|---|---|
| discrete adjoint | 0.2166 | 0.1711 to 0.2949 | 1181 to 1200 | 5/5 | 3/5 |
| CMA-ES | 2.0659 | 0.8128 to 3.9520 | 1200 to 1200 | 0/5 | 0/5 |
| one-step PPO | 0.3766 | 0.3261 to 0.5371 | 153 to 396 | 4/5 | 0/5 |

**Verdict.** discrete adjoint vs CMA-ES: discrete adjoint leads (the ranges of best J at the cap do not overlap); discrete adjoint vs one-step PPO: inconclusive (one-step PPO stopped below 90 % of the cap in at least one run); CMA-ES vs one-step PPO: inconclusive (one-step PPO stopped below 90 % of the cap in at least one run). one-step PPO runs ended at the 20000-candidate stop rule of stage6_bb, because most samples failed the geometry gates (free to score); that rule is in the committed binary but not in the protocol.

### Cost accounting, cache, sensitivity

- One adjoint iteration: 3 forward solves (one for each variant) and 11 + 12 adjoint solves (the T and R gradients of the worst variant). Charge: 12 (T gradient, which includes the forward solve of the worst variant) + 12 (R gradient, 12 adjoint solves; it reuses that forward solve, which is not charged again) + 2 (the other two forward solves) = 26, equal to the physical solves. A binary score costs 3 forward solves; a cache hit or a mask that fails a geometry gate costs 0.
- Cache: the `Scorer` cache is keyed by the binary nominal mask and is the same for all methods. It had 0 hits in the production runs.
- Sensitivity: the measured wall time of an adjoint solve is 0.852 times that of a forward solve (median over the five adjoint runs). The dotted line in panel A re-weights the adjoint solves with this ratio, which moves the adjoint curve to about 22.6 forward-equivalents for each iteration instead of 26.
- The CMA-ES and PPO runs also spend wall time outside the FDTD solves, so their wall time is not proportional to the charged cost. Measured (wall time minus forward-solve time, divided by the candidates that failed the geometry gates): 1.5 ms for each such candidate for PPO, and 31 to 39 ms for CMA-ES, where this time also holds the covariance update and the eigen-decomposition of every generation.
- Every PPO run stopped at the 20000-candidate stop rule of `stage6_bb` (153 to 396 charged): the policy moved to masks that fail the geometry gates, which cost nothing to score. That stop rule was in the binary committed before the production runs (`c786c38`), but not in the protocol. Comparisons with PPO are inconclusive at equal budget.

### Finalists on the held-out sweep and grid

| Finalist | Robust IL / RL training | Held-out 61 | Half-cell grid | Gates | Gain over Bezier k = 0.45 (train / 61 / half-cell) | Win |
|---|---|---|---|---|---|---|
| adjoint seed 5 | 0.1711 / 22.99 | 0.1718 / 22.99 | 0.1688 / 23.24 | pass | 0.1595 / 0.1588 / 0.1611 | yes |
| cma seed 5 | 0.5780 / 21.77 | 0.5780 / 21.76 | 0.5860 / 19.75 | fail | -0.2474 / -0.2474 / -0.2561 | no |
| ppo seed 1 | 0.3261 / 22.75 | 0.3261 / 22.75 | 0.3262 / 22.81 | pass | 0.0045 / 0.0045 / 0.0037 | no |
| phase1 r_f 6 | 0.2737 / 22.50 | 0.2737 / 22.31 | 0.2724 / 22.59 | pass | 0.0569 / 0.0569 / 0.0575 | no |

The finalist of each method is its lowest-J run on the training spectrum. Held-out validation cost: 125 s for the phase 2 finalists (`docs/results/stage6/phase2/validate.log`), 113 s for phase 1, and 155 s for the combined release validation, whose results are identical to both (`docs/results/stage6/release_validate/`).

Release artifacts: `release/bend80.gds` is the release design; `release/bend80_phase1.gds` is the full-resolution phase 1 design, kept as the record of the unconstrained-grid failure mode (it passes the gates but has no win, and its run produced mostly infeasible masks). The release design is `adjoint seed 5`: the lowest training-set robust J among the designs that meet every gate. The protocol did not name the release design in advance. This rule was stated after the results were known, and it does not use held-out data. Its biased masks have 4 width violations (eroded) and 2 width and 2 gap violations (dilated); the protocol reports these and does not gate them.

Regenerate the plots and tables:

	tools/plot_stage6.py tradeoff docs/results/stage6/phase2/runs release/tradeoff.svg release/tradeoff.json --t2 0.2306
	tools/plot_stage6.py spectra docs/results/stage6/release_validate release/spectra_heldout61.svg release/spectra_heldout61.json --designs "adjoint seed 5,phase1 r_f 6,baseline bezier:0.45,baseline arc"
	tools/make_release.sh v0.6.0-rc1

## Lean 4 proof (range item)

`lean/YeeEnergy.lean` proves in core Lean 4 (no Mathlib) that one leapfrog Yee
step conserves the discrete energy W = Σ E² + Σ H·H⁺ on a periodic grid, in 1D
(`energy_conserved`) and 2D TMz (`energy2_conserved`), over any commutative
ring and for every Courant coefficient. `#print axioms` lists only `propext`,
`Classical.choice` and `Quot.sound`; a concrete integer instance keeps the
energy at 525 (1D) and 1710 (2D) over 10 steps; a wrong flux coefficient makes
the build fail. The proof does not cover the CPML, sources, material contrast,
floating-point arithmetic or the CUDA code. Details: [lean.md](lean.md);
evidence: `docs/results/lean/`.

