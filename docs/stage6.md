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

PHASE2_PLACEHOLDER
