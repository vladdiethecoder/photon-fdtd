# Inverse design of the 90-degree bend (Stage 5)

This document gives the state of Stage 5. Unit 1 (the design region and the forward evaluation) is complete. The adjoint gradient and the optimisers are not implemented yet. All values come from commit `abe9cd2faad1dafa6dff8a5c6682c414b0d62a46`. The raw output is in `docs/results/bend_design.log`.

	cargo run --release --bin bend_design -- 48 56 64 72 80 146

## Design region

`DesignBox::for_bend(setup, n, margin)` places an n x n box at the corner of the Stage 4 bend. The horizontal slab enters through the left side and the vertical slab leaves through the top side, `margin` cells from the box corner (margin = 8). The ports stay outside the box, so the modal decomposition of Stage 4 does not change. If the box holds the sharp L, the grid is identical to the Stage 4 bend.

The parameterization has three steps:

1. The latent density rho in [0, 1] has one value for each box cell.
2. `ConicFilter` (radius r_f, weight max(0, r_f - distance), normalised by the weight sum) smooths rho. Outside the box, the filter halo takes the fixed geometry: 1 in the two slabs and 0 in the cladding. Thus the slabs continue into the box without a step, and the filter is affine (x_f = W x + c).
3. `project` (tanh projection, eta = 0.5) binarises the filtered density. eps = 1 + rho_bar (12.11 - 1).

`pullback` applies the chain rule back to rho: the projection derivative, then W^T.

## Fabrication metrics

- `binary_fraction`: the fraction of cells with rho_bar < 0.01 or > 0.99.
- `single_cell_islands`: the number of 4-connected components of 1 cell, of silicon or of air, that do not touch the same phase outside the box.

## Baselines

| Box | Arc centreline radius | Sharp L: IL, RL at lambda0 = 146 | Arc: IL at lambda0 = 146 | Arc: band max IL | Arc: band min RL |
|---|---|---|---|---|---|
| 48 x 48 | 32 cells (0.22 lambda0) | 3.748 dB, 6.69 dB | 0.359 dB | 0.693 dB | 27.52 dB |
| 56 x 56 | 40 cells | 3.748 dB, 6.69 dB | 0.235 dB | 0.538 dB | 36.05 dB |
| 64 x 64 | 48 cells | 3.748 dB, 6.69 dB | 0.161 dB | 0.406 dB | 37.94 dB |
| 72 x 72 | 56 cells | 3.748 dB, 6.69 dB | 0.113 dB | 0.314 dB | 38.27 dB |
| 80 x 80 | 64 cells (0.44 lambda0) | 3.748 dB, 6.69 dB | 0.071 dB | 0.258 dB | 40.91 dB |
| 146 x 146 | 130 cells (0.89 lambda0) | 3.748 dB, 6.69 dB | 0.015 dB | 0.039 dB | 37.77 dB |

The arc is a rasterised annulus of width 16 cells, centred at the top-left box corner and tangent to both slabs. IL and RL are modal values over the 15 band frequencies (lambda0 = 115 to 200 cells).

The sharp L gives bit-identical modal S-parameters to the Stage 4 bend in every box. The circular arc already meets the proposed Stage 5 gates (IL < 0.5 dB and RL > 20 dB over the band) in every box of 64 x 64 cells or larger. In the 80 x 80 box, it has a band maximum IL of 0.258 dB without any optimisation.

Hypothesis, not tested: the 2D slab has a high index contrast (3.48 to 1) and no substrate, so tight bends radiate less than in the 3D silicon-on-insulator devices of the published comparisons. Those comparisons are not reproduced here.

## Proposed gate change

The proposed gates do not separate an optimiser from the analytic arc at 80 x 80 cells. These gates separate them:

- **Box 48 x 48 cells**: the arc has a band maximum IL of 0.693 dB and a band minimum RL of 27.52 dB. The optimised design must have a band maximum IL < 0.5 dB and RL > 20 dB, and it must beat the arc at each band frequency.
- **Box 80 x 80 cells**: the optimised design must beat the arc's band maximum IL of 0.258 dB.

The fabrication gates stay: >= 99 % binary and no isolated 1-cell islands.

## Tests

| Test | Gate |
|---|---|
| `design_filter_projection_and_pullback` | partition of unity; filtered sharp L > 0.99 at the slab centre on the box edge; <W x, y> = <x, W^T y> within 1e-10; projection derivative within 1e-6; pullback vs central differences cosine > 0.999999 |
| `design_metrics_binary_fraction_and_islands` | sharp L and arc: binary 1.0, 0 islands; one silicon and one air pixel give 2 islands |
| `design_box_forward_evaluation_and_baselines` | the sharp L in the box is bit-identical to the Stage 4 bend; arc baselines at 80 x 80 and 48 x 48 within 0.01 dB (IL) and 0.1 dB (RL) |

## Next units

1. The adjoint gradient of the modal |S21|^2 at the 15 frequencies, from DFT fields of one forward and one adjoint run. The gate is a finite-difference check on a 4 x 4 block (cosine > 0.999).
2. A gradient optimiser with beta continuation (1 to 32), against the gates above.
3. The GPU environment for RL and evolutionary optimisers.
4. SVG and GDSII export of the binarised contour.
