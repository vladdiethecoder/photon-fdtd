# Methodology

This document gives the discrete equations that `src/lib.rs` and `src/gpu.rs` use. It also gives the stability limit, the CPML and the sources.

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

One step updates H first and then Ez. With k = j * nx + i, dt = S and cb[k] = dt / eps_r[k]:

	Hx[k] = Hx[k] - dt * ((Ez[k + nx] - Ez[k]) * iky_h[j])
	Hy[k] = Hy[k] + dt * ((Ez[k + 1]  - Ez[k]) * ikx_h[i])
	Ez[k] = Ez[k] + cb[k] * ((Hy[k] - Hy[k - 1]) * ikx_e[i] - (Hx[k] - Hx[k - nx]) * iky_e[j])

The values ikx and iky are 1 / kappa of the CPML. The suffix `_h` is the half-cell position and `_e` is the integer position. Outside the CPML, all four tables are exactly 1.0. Thus the update is the standard lossless Yee update there. In the CPML strips, the CPML corrections follow the main update (see [CPML](#cpml-cfs-cpml-roden-and-gedney-2000)).

The H update covers i = 0..nx-2 and j = 0..ny-2. The Ez update covers i = 1..nx-2 and j = 1..ny-2. The outer ring of Ez stays at 0, which is a perfect electric conductor (PEC).

## Stability limit (CFL)

Put a plane wave exp(i(omega n dt - kx i dx - ky j dy)) into the lossless update. The result is the numerical dispersion relation of the 2D Yee scheme:

	(1/(c dt))^2 sin^2(omega dt / 2) = (1/dx^2) sin^2(kx dx / 2) + (1/dy^2) sin^2(ky dy / 2)

A real omega needs a left side of 1 / (c dt)^2 or less for every kx and ky. The right side has its largest value when both sines are 1. Thus:

	c dt <= 1 / sqrt(1/dx^2 + 1/dy^2)

For dx = dy, the limit is S = c dt / dx <= 1/sqrt(2) = 0.7071. If S is larger, some wave numbers have a complex omega and grow exponentially. Vacuum is the fastest medium, so the vacuum limit applies to all materials with eps_r >= 1.

`check_setup` returns an error if S is not in (0, 1/sqrt(2)]. `Yee2d::new` also returns an error for an eps_r value that is NaN or smaller than 1. All scenarios use S = 0.99 / sqrt(2) = 0.700036.

## CPML (CFS-CPML, Roden and Gedney 2000)

The CPML stretches the coordinates in the boundary layer with s = kappa + sigma / (alpha + j omega), in natural units. `src/cpml.rs` calculates the profiles. N is the thickness in cells. The PML depth of a position p is:

	rho = max((N - p) / N, (p - (n - 1 - N)) / N, 0)

Thus rho = 0 at the inner interface and rho = 1 at the PEC wall. p is i for Ez and i + 1/2 for H. Each axis has two tables, one for each staggering. For rho > 0:

	sigma = sigma_max * rho^m
	kappa = 1 + (kappa_max - 1) * rho^m
	alpha = alpha_max * (1 - rho)^m_alpha
	b     = exp(-(sigma / kappa + alpha) * dt)
	a     = sigma / (sigma * kappa + kappa^2 * alpha) * (b - 1)

For rho <= 0, the tables hold 1 / kappa = 1, a = 0 and b = 0. If the denominator of a is 0, a is 0.

Four auxiliary arrays (psi) hold the recursive convolutions. Each psi value is updated after the main update of its field, in this order:

	psi_hyx = b_x * psi_hyx + a_x * (Ez[k + 1] - Ez[k]);     Hy[k] += dt * psi_hyx     (x strips)
	psi_hxy = b_y * psi_hxy + a_y * (Ez[k + nx] - Ez[k]);    Hx[k] -= dt * psi_hxy     (y strips)
	psi_ezx = b_x * psi_ezx + a_x * (Hy[k] - Hy[k - 1]);     Ez[k] += cb[k] * psi_ezx  (x strips)
	psi_ezy = b_y * psi_ezy + a_y * (Hx[k] - Hx[k - nx]);    Ez[k] -= cb[k] * psi_ezy  (y strips)

A corner cell is in one x strip and in one y strip, so it gets both corrections. The x correction comes first.

The coordinate stretch does not depend on the medium. The material enters only through cb = dt / eps_r. Thus the CPML also absorbs a wave in a dielectric that crosses the layer, such as the silicon strip.

### Default profile

`Pml::new(cells)` uses m = 3, sigma_max = 0.8 (m + 1) = 3.2, kappa_max = 1, alpha_max = 0.05 and m_alpha = 1. sigma_max = 0.8 (m + 1) / (eta dx) is the usual optimum, and eta = 1 in natural units. The parameter sweep in docs/verification.md selected kappa_max and alpha_max.

## Source

The scenarios use a soft source. The source adds a value to Ez after each step and does not replace Ez:

	Ez[j * nx + i] += g((n + 1) dt)

The scenarios use two waveforms:

	Gaussian sine (pulse A):           g(t) = exp(-((t - t0) / tau)^2) sin(2 pi f0 t),  f0 = 1 / lambda0
	Gaussian derivative (pulse B):     g(t) = -a exp(-a^2) sqrt(2) exp(1/2),  a = (t - t0) / tau

Pulse A has a small DC component. Pulse B has none, and its peak value is 1. The silicon scenario uses a line source across the strip rows with pulse A. The absorber check uses a point source with pulse A and pulse B.

## Discrete energy

The scenarios measure confinement with this sum, calculated by `energy_rows`:

	W = 0.5 * sum over cells of (eps_r Ez^2 + Hx^2 + Hy^2)

The sum mixes the field values of half time steps. Thus it is an approximation of the conserved discrete energy, which is sufficient for a confinement ratio.
