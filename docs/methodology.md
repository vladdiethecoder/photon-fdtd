# Methodology

This document gives the discrete equations that `src/lib.rs` and `src/gpu.rs` use. It also gives the stability limit, the absorbing sponge and the source.

## Units

The solver uses natural units: c = eps0 = mu0 = 1 and dx = dy = 1. Then dt = S, where S = c dt / dx is the Courant number. A material has only one parameter, the relative permittivity eps_r. mu_r is 1 everywhere.

In these units, the update contains no physical constants. For example, eps0 = 8.854e-12 and mu0 = 1.257e-6 do not occur. Thus no values of very different scales go into one sum. To convert to SI units, select a physical cell size dx. Then one time unit is dx / c, and one field unit for H is one field unit for E divided by the vacuum impedance (376.73 ohm).

## Fields and grid

The TM_z mode has three field components: Ez, Hx and Hy. On the Yee grid, they have these positions:

- Ez(i, j) at (i, j)
- Hx(i, j) at (i, j + 1/2)
- Hy(i, j) at (i + 1/2, j)

Each component is one flat array with index k = j * nx + i. The x index i is the contiguous index.

## Continuous equations

	dHx/dt = -(1/mu) dEz/dy
	dHy/dt =  (1/mu) dEz/dx
	dEz/dt =  (1/eps) (dHy/dx - dHx/dy)

## Discrete update

One step updates H first and then Ez. With k = j * nx + i:

	Hx[k] = ca[k] * Hx[k] - hb[k] * (Ez[k + nx] - Ez[k])
	Hy[k] = ca[k] * Hy[k] + hb[k] * (Ez[k + 1]  - Ez[k])
	Ez[k] = ca[k] * Ez[k] + cb[k] * ((Hy[k] - Hy[k - 1]) - (Hx[k] - Hx[k - nx]))

The H update covers i = 0..nx-2 and j = 0..ny-2. The Ez update covers i = 1..nx-2 and j = 1..ny-2. The outer ring of Ez stays at 0, which is a perfect electric conductor (PEC).

Outside the sponge, ca = 1, hb = S and cb = S / eps_r. Thus the update is the standard lossless Yee update.

## Stability limit (CFL)

Put a plane wave exp(i(omega n dt - kx i dx - ky j dy)) into the lossless update. The result is the numerical dispersion relation of the 2D Yee scheme:

	(1/(c dt))^2 sin^2(omega dt / 2) = (1/dx^2) sin^2(kx dx / 2) + (1/dy^2) sin^2(ky dy / 2)

A real omega needs a left side of 1 / (c dt)^2 or less for every kx and ky. The right side has its largest value when both sines are 1. Thus:

	c dt <= 1 / sqrt(1/dx^2 + 1/dy^2)

For dx = dy, the limit is S = c dt / dx <= 1/sqrt(2) = 0.7071. If S is larger, some wave numbers have a complex omega and grow exponentially. Vacuum is the fastest medium, so the vacuum limit applies to all materials with eps_r >= 1.

`check_setup` returns an error if S is not in (0, 1/sqrt(2)]. `Yee2d::new` also returns an error for an eps_r value that is NaN or smaller than 1. All scenarios use S = 0.99 / sqrt(2) = 0.700036.

## Absorbing sponge

The sponge is a graded conductor in the outer `sponge` cells. The variable d is the distance in cells from the cell to the nearest outer edge. The depth is (sponge - d) / sponge if d < sponge, otherwise 0. `cell_coeffs` calculates:

	sigma_max = (m + 1) ln(1e6) / (2 * sponge),   m = 3
	sigma     = sigma_max * depth^m * sqrt(eps_r)
	loss      = sigma * dt / (2 eps_r)
	ca = (1 - loss) / (1 + loss)
	cb = dt / (eps_r (1 + loss))
	hb = dt / (1 + loss)

The magnetic conductivity sigma_m obeys sigma_m / mu = sigma / eps. With this match, the wave impedance in the sponge is the same as in the medium. H and Ez thus share the loss term and the self coefficient ca. The factor sqrt(eps_r) gives the same attenuation per cell in silicon and in air.

The sponge has these limits:

- The impedance match is exact only at normal incidence in the continuous medium. An isotropic conductor is not a perfectly matched layer (PML). It reflects oblique waves.
- Hx and Hy use the coefficient of cell (i, j), not of their half-cell positions. This approximation adds discretization error where the grading is steep.

docs/verification.md gives the measured reflection.

## Source

The scenarios use a soft source. The source adds a value to Ez after each step and does not replace Ez:

	Ez[j * nx + i] += g((n + 1) dt),   g(t) = exp(-((t - t0) / tau)^2) sin(2 pi f0 t)

f0 = 1 / lambda0 in natural units. The silicon scenario uses a line source across the strip rows. The absorber check uses a point source.

## Discrete energy

The scenarios measure confinement with this sum, calculated by `energy_rows`:

	W = 0.5 * sum over cells of (eps_r Ez^2 + Hx^2 + Hy^2)

The sum mixes the field values of half time steps. Thus it is an approximation of the conserved discrete energy, which is sufficient for a confinement ratio.
