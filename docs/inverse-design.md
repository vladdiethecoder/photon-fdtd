# Inverse design of the 90-degree bend (Stage 5)

All results are for a 2D TMz model: Ez out of plane, a slab of eps_r 12.11 in a cladding of eps_r 1, natural units. They are not 3D silicon-on-insulator results, and the designs are not fabrication-ready.

The comparison rules are in [docs/stage5-protocol.md](stage5-protocol.md). They were committed (`1d438bc`) before the optimiser existed. The best conventional baseline was measured and committed (`6996ac9`) before the optimiser ran.

The code is at commit `07e727e`. The raw output is in `docs/results/stage5/` and `docs/results/stage5_baselines.log`.

	cargo run --release --bin adjoint_check -- --extra-time 6000
	cargo run --release --bin stage5_baselines
	cargo run --release --bin optimize_bend -- --iters 20 --out out/opt
	cargo run --release --bin optimize_bend -- --iters 20 --betas 64,128 --init out/opt/rho_final.txt --out out/opt2
	cargo run --release --bin stage5_validate -- out/opt2/rho_final.txt --beta 128

## Result (primary benchmark, 48 x 48 box)

| Design | Worst IL, 15 freqs | Worst RL, 15 freqs | Worst IL, 61 freqs | Worst RL, 61 freqs | Worst IL, scale 2 | Worst RL, scale 2 |
|---|---|---|---|---|---|---|
| Optimised | 0.2586 dB | 23.41 dB | 0.2586 dB | 23.36 dB | 0.2659 dB | 23.53 dB |
| Bezier k = 0.40 (best baseline) | 0.5933 dB | 25.35 dB | 0.5933 dB | 25.34 dB | 0.5869 dB | 26.08 dB |
| Circular arc | 0.6926 dB | 27.52 dB | 0.6926 dB | 27.52 dB | 0.6967 dB | 27.75 dB |

The numerical uncertainty is U = 0.0073 dB: the larger change of worst-case IL between scale 1 and scale 2, for the optimised design (0.0073 dB) and for the best baseline (0.0064 dB). The gain over the best baseline is:

| Spectrum | Gain | Required |
|---|---|---|
| 15 optimisation frequencies | 0.3347 dB | >= 0.10 dB and > U |
| 61 validation frequencies | 0.3347 dB | >= 0.10 dB and > U |
| scale 2 | 0.3210 dB | >= 0.10 dB and > U |

Thus the result meets the protocol's win condition. The acceptance gates also pass:

| Gate | Value |
|---|---|
| Binary fraction before the threshold (beta 128) | 0.9965 (>= 0.99) |
| Isolated 1-cell islands | 0 |
| L_min = 5 cells (opening and closing with a disk of diameter 5) | 0 and 0 changed cells |
| Worst IL < 0.50 dB, worst RL > 20 dB (15 and 61 frequencies) | 0.2586 dB, 23.36 dB |

![Optimised, Bezier k = 0.40 and arc (white = silicon, 48 x 48 cells)](images/stage5_designs_48.png)

The optimised design is one connected silicon region. It is wider than a single waveguide in the turn, and it has no holes, gratings or isolated parts.

### Spectra (61 frequencies)

`docs/results/stage5/spectra_61.csv` has the modal IL and RL of the three designs at 61 frequencies.

| Design | IL range | Worst IL at | RL range | Worst RL at | IL at lambda0 = 146.0 |
|---|---|---|---|---|---|
| Optimised | 0.1387 to 0.2586 dB | lambda0 = 200.0 | 23.36 to 40.35 dB | lambda0 = 154.4 | 0.1552 dB |
| Bezier k = 0.40 | 0.2886 to 0.5933 dB | lambda0 = 200.0 | 25.34 to 36.45 dB | lambda0 = 167.1 | 0.2963 dB |
| Circular arc | 0.2540 to 0.6926 dB | lambda0 = 200.0 | 27.52 to 65.51 dB | lambda0 = 200.0 | 0.3585 dB |

The worst IL of every design is at the long-wavelength band edge, which is also an optimisation frequency. Thus the 15-point and the 61-point worst cases are the same.

### Erosion and dilation by 1 cell (scale 1, 15 frequencies)

| Design | Eroded: worst IL, RL | Dilated: worst IL, RL |
|---|---|---|
| Optimised | 0.3114 dB, 20.40 dB | 0.2730 dB, 20.69 dB |
| Bezier k = 0.40 | 0.6949 dB, 29.45 dB | 0.5903 dB, 19.86 dB |
| Circular arc | 0.8226 dB, 22.40 dB | 0.6096 dB, 25.83 dB |

The erosion and the dilation act on the box only. The ports stay fixed. The optimised design keeps IL < 0.5 dB under both changes. Its RL falls to about 20.4 dB, so it has almost no RL margin under a 1-cell change.

## Method

### Design region and parameterization

`DesignBox::for_bend(setup, 48, 8)` is a 48 x 48 box at the corner of the Stage 4 bend. The slabs enter through the left side and leave through the top side, 8 cells from the box corner, and the ports stay outside the box. The chain from the latent density to eps:

1. latent rho in [0, 1], one value for each box cell;
2. conic filter, radius 5 cells, with the fixed exterior geometry as the halo;
3. tanh projection, eta = 0.5, beta 1, 2, 4, 8, 16, 32, 64, 128 (20 iterations each);
4. eps = 1 + rho_bar (12.11 - 1).

The final binary design is the 0.5 level set of the filtered density.

### Adjoint gradient

`src/adjoint.rs` gives the derivation. In short, the DFT fields of the leapfrog update obey (K - eps Omega^2) E = b exactly, with Omega = (2 / dt) sin(omega dt / 2). Thus dT_k / deps_r = 2 Omega_k^2 Re(conj(a_k) E_r lambda_r) / |a_inc,k|^2. Here lambda is the field of an adjoint run with a source at the port-2 row, normalised by i Omega e^{i omega dt / 2} G_hat / dt. The 15 port-2 mode profiles are expanded in 11 orthonormal vectors (largest residual 7.05e-7). Thus one gradient needs 1 forward run and 11 adjoint runs, about 7 s on the RTX 5090.

The objective is F = (mean_k (1 - T_k)^8)^(1/8), a smooth form of the worst-case loss. T_k is the port-2 mode amplitude from Ez on the port row, divided by the incident amplitude of Stage 4. The optimiser is Adam on rho (learning rate 0.05), and rho is clipped to [0, 1].

### Gradient verification

From `docs/results/adjoint_check_t0.log` and `docs/results/adjoint_check_t6000.log`, on a grey design (filtered arc, beta 4):

| Check | Stage 4 run length (11688 steps) | Stage 5 run length (20259 steps) |
|---|---|---|
| 4 x 4 block, cosine at lambda0 = 200, 165, 146, 131, 115 | 1.0000000 at all five | 1.0000000 at all five |
| 4 x 4 block, per-cell adjoint / FD ratio | 0.99805 to 1.00038 | 1.00000 at all cells |
| 4 x 4 block, signs | 16/16 | 16/16 |
| Directional, random direction, lambda0 = 200, steps 1e-1 to 1e-4 | 5.227 to 5.230 | 0.99998 to 1.00001 |
| Directional, lambda0 = 146 and 115 | 0.99973 and 1.00036 | 1.00000 within 4.3e-6 |
| Full chain rho -> F, steps 1e-2 to 1e-5 | 1.00036 to 1.00037 | 0.99999 to 1.00000 |

With the Stage 4 run length, the cosine was 1.0000000, but the random-direction derivative at lambda0 = 200 was wrong by a factor of 5.2. Most of the per-cell derivative cancels in that direction, so a 0.2 % per-cell error became a large error of the sum. The cause was DFT truncation at the weak band edge of the pulse. The extra 6000 time units remove it. Thus every Stage 5 run, baselines included, uses the longer run (`extra_time = 6000`).

## Stretch comparison (80 x 80 box)

One run of the same optimiser in the 80 x 80 box (r_f 5, beta 1 to 128, 20 iterations each, `docs/results/stage5/optimize_80.log`). It is a stretch comparison, not a gate, and it has no validation runs (no 61-point sweep, no scale 2, no erosion or dilation).

| Design | Worst IL, 15 freqs | Worst RL, 15 freqs | Binary before threshold | L_min violations (open, close) |
|---|---|---|---|---|
| Optimised, 80 x 80 | 0.0376 dB | 24.87 dB | 0.9877 | 68, 0 |
| Circular arc, 80 x 80 (Stage 4 run length) | 0.258 dB | 40.91 dB | 1.0 | 0, 0 |

The 80 x 80 result has a lower IL than the arc, but it fails two fabrication rules: it is less than 99 % binary, and an opening with the 5-cell disk changes 68 cells, so it has silicon features thinner than L_min. A larger filter radius or a feature constraint is necessary before this result can count.

## Tests

| Test | Gate |
|---|---|
| `adjoint_gradient_matches_finite_differences` | 4 x 4 block at 3 wavelengths: cosine > 0.999, magnitude within 1e-3, 16/16 signs; directional (steps 1e-2, 1e-3) within 1e-3; full chain within 1e-4 |
| `stage5_feature_check_and_best_baseline` | the L_min check detects a 3-cell line and a 3-cell gap; Bezier k = 0.40 worst IL 0.5933 dB |
| `stage5_optimised_design_meets_gates` | the committed design: >= 99 % binary, 0 islands, L_min, worst IL < 0.50 dB, RL > 20 dB, IL 0.2586 +- 0.002 dB |
| `design_filter_projection_and_pullback`, `design_metrics_binary_fraction_and_islands`, `design_box_forward_evaluation_and_baselines` | see unit 1 below |

## Unit 1: design region and box baselines

The sharp L in the box gives bit-identical modal S-parameters to the Stage 4 bend. Circular arcs at the Stage 4 run length (`docs/results/bend_design.log`):

| Box | Arc centreline radius | Arc: band max IL | Arc: band min RL |
|---|---|---|---|
| 48 x 48 | 32 cells | 0.693 dB | 27.52 dB |
| 56 x 56 | 40 cells | 0.538 dB | 36.05 dB |
| 64 x 64 | 48 cells | 0.406 dB | 37.94 dB |
| 72 x 72 | 56 cells | 0.314 dB | 38.27 dB |
| 80 x 80 | 64 cells | 0.258 dB | 40.91 dB |
| 146 x 146 | 130 cells | 0.039 dB | 37.77 dB |

Hypothesis, not tested: the 2D slab has a high index contrast (3.48 to 1) and no substrate, so tight bends radiate less than in published 3D silicon-on-insulator devices.

## Limits

- 2D TMz only. No 3D, substrate or out-of-plane loss.
- The L_min check is a pixel morphology test at scale 1. It is not a design-rule check for a fabrication process.
- The numerical uncertainty U measures only the grid change from scale 1 to scale 2. The CPML reflection (-83 dB or lower) and the mode split are smaller error sources and are not part of U.
- The optimised design has almost no RL margin under a 1-cell erosion (20.40 dB).
- The SVG and GDSII export, and the RL and evolutionary optimisers, are not implemented.
