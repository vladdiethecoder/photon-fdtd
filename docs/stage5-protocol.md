# Stage 5 comparison protocol

This protocol was committed before the optimiser existed and before any optimiser tuning. Later commits must not change it. If a rule must change, a new protocol file supersedes it, and the report names both files.

All results are for a 2D TMz model (Ez out of plane, eps_r 12.11 slab in eps_r 1 cladding). They are not 3D silicon-on-insulator results, and the designs are not fabrication-ready.

## Benchmark

- Primary: the 48 x 48-cell design box of `DesignBox::for_bend(PortSetup::standard(Geometry::Bend, 1), 48, 8)`. The ports, the source, the CPML (10 cells, alpha_max 0.01), the band (lambda0 = 115 to 200 cells) and the modal S-parameters are those of Stage 4 (docs/sparams.md).
- Stretch comparison, not a release gate: the 80 x 80-cell box against the circular arc (band maximum IL 0.258 dB).

## Designs under comparison

All designs use the same box, ports and minimum feature size.

1. **Optimised**: the binarised output of the adjoint optimiser.
2. **Circular arc**: `DesignBox::circular_arc` (centreline radius n - margin - core_w / 2).
3. **Euler bends**: a 90-degree bend with the same end points and end directions as the arc. The curvature rises linearly from 0, stays constant, then falls linearly to 0. The Euler fraction p (the fraction of the bend angle in the two clothoid parts) takes the values 0.1, 0.2, ..., 1.0. p = 0 is the arc.
4. **Cubic Bezier bends**: control points P0 = start, P1 = P0 + k L (+x), P2 = P3 - k L (+y), P3 = end, with L the straight distance per axis and k = 0.30, 0.35, ..., 0.70.

The waveguide width of baselines 2 to 4 is 16 cells. A box cell is silicon if its centre is within 8 cells of the centreline.

**Best conventional baseline**: the design among 2 to 4 with the lowest worst-case modal IL over the 15 optimisation frequencies, at scale 1. It is selected from measurements before the optimiser runs.

## Minimum feature size

L_min = 5 cells at scale 1, for silicon width and for air gaps. A binary design passes if a morphological opening and a morphological closing with a discrete disk of diameter 5 cells (offsets with dx^2 + dy^2 <= 4) change no box cell. The fixed geometry outside the box pads the operations. All compared designs must pass.

## Acceptance gates (optimised design, binary)

1. The design is >= 99 % binary before the final threshold. After the threshold, it has 0 isolated 1-cell islands and passes the L_min check.
2. Worst-case modal IL < 0.50 dB and worst-case modal RL > 20 dB over the 15 optimisation frequencies.
3. The same holds on the dense validation sweep (61 frequencies over the same band).

## Optimiser win

The optimiser wins only if all of these conditions are true:

- Its worst-case IL is at least 0.10 dB lower than the worst-case IL of the best conventional baseline. This must hold on the 15 optimisation frequencies and on the 61 validation frequencies.
- The gain is larger than the numerical uncertainty U. U is the larger of the two changes of worst-case IL (optimised design and best baseline) between scale 1 and scale 2.
- At scale 2, the gain is also >= 0.10 dB and > U.

## Validation

- Dense sweep: 61 frequencies over lambda0 = 115 to 200 cells, at scale 1.
- Finer grid: the same physical designs on the scale-2 grid (cells half as large). The optimised design is the 0.5 level set of its filtered density, interpolated bilinearly at the fine cell centres. The baselines are rasterised from their centrelines.
- Sensitivity: erosion and dilation of each binary design by 1 cell (4-neighbour structuring element) at scale 1. Report the worst-case IL and RL.

## Gradient verification (before the optimiser)

- 4 x 4 cell block: the adjoint dT/deps against central finite differences at each cell. Cosine > 0.999 at several wavelengths. The magnitude ratio and the sign are reported for each cell.
- Directional derivatives along random directions with step sizes 1e-1 to 1e-4. The ratio to the adjoint value must approach 1.
- The full chain from the latent density to the objective is tested in the same way.
