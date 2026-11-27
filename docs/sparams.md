# Slab modes, ports and S-parameters

This document describes the Stage 4 measurement pipeline in `src/sparams.rs` and gives its results. The pipeline has four parts: slab modes, the mode source, the port monitors and the S-parameters. All values come from commit `702b85c4f3eb6df12fb90a400267d46e03953548`. The raw output is in `docs/results/sparams*.log`.

Run the measurement:

	cargo run --release --bin sparams                               # scale 1: straight guide and 90-degree bend
	cargo run --release --bin sparams -- --scale 2 --straight-only  # twice the resolution

## Setup

| Item | Value (scale 1) |
|---|---|
| Slab | 16 cells of silicon (eps_r 12.11) in air, walls at half-cell positions |
| Band | lambda0 = 115 to 200 cells, 15 frequencies. The guide has one guided mode over the band (it has one mode for lambda0 > 107 cells). |
| Cladding | 150 cells between the slab and the CPML |
| CPML | 10 cells, kappa_max = 1, alpha_max = 0.01 (see [CPML at long wavelengths](#cpml-at-long-wavelengths)) |
| Source | soft Ez line source at x = 50 with the discrete eigenmode profile at the band centre, Gaussian sine (tau = 243.6, t0 = 4.5 tau = 1096.3) |
| Port 1 | Ez at x = 400 and 401, Hy at x = 400.5, full interior height |
| Port 2, straight | the same lines at x = 610 |
| Port 2, bend | Ez at y = 426 and 427, Hx at y = 426.5, full interior width. The vertical arm starts at x = 600. |
| Steps | straight 8891, bend 11688. At the end, the DUT field energy is below 3.1e-10 of its peak. |

At scale 2, all lengths double and all frequencies halve.

## Slab mode

### Analytic mode

Ez is out of plane and the slab walls lie along x, so Ez and dEz/dy are continuous across the walls. The fundamental even mode obeys:

	kappa = k0 sqrt(n1^2 - neff^2),  gamma = k0 sqrt(neff^2 - n2^2),  tan(kappa d / 2) = gamma / kappa

`slab_even_mode` solves this equation by bisection on neff, on the branch kappa d / 2 < pi / 2. The profile is cos(kappa y) in the core and cos(kappa d / 2) exp(-gamma (|y| - d / 2)) in the cladding.

### Discrete mode

The leapfrog update has an exact eigenproblem for a mode Ez = u_j exp(i (omega t - beta x)). With Omega = (2 / dt) sin(omega dt / 2) and B = 2 sin(beta / 2):

	u_{j+1} - 2 u_j + u_{j-1} + eps_j Omega^2 u_j = B^2 u_j

`discrete_mode` finds the largest eigenvalue of this tridiagonal matrix by Sturm-sequence bisection, and its eigenvector by inverse iteration. The eigenvector is the source profile and the projection basis of the ports. The count of eigenvalues above the cladding light line (eps_clad Omega^2) is the number of guided modes.

### Results

From `docs/results/sparams.log` and `docs/results/sparams_scale2.log`:

| lambda0 (scale 1) | neff (analytic) | beta measured / beta grid - 1 | beta analytic / beta grid - 1, scale 1 | the same, scale 2 | 1 - profile overlap |
|---|---|---|---|---|---|
| 200.0 | 2.27796 | -8.28e-5 | -7.68e-4 | -1.92e-4 | 6.24e-7 |
| 146.0 | 2.59098 | 1.65e-6 | -1.20e-3 | -3.00e-4 | 1.17e-6 |
| 115.0 | 2.80098 | 5.08e-7 | -1.70e-3 | -4.25e-4 | 1.62e-6 |

- The measured beta comes from the DFT phase of Ez along the slab centre over 300 cells, with a least-squares slope. It agrees with the grid eigenvalue to within 8.3e-5 at every frequency.
- The analytic beta differs from the grid beta by 7.7e-4 to 1.7e-3. At twice the resolution, this difference is 4.0 times smaller at every frequency. The difference is thus the second-order grid dispersion, and the analytic root is the continuum limit of the grid eigenvalue.
- The analytic profile, sampled at the Ez rows, overlaps the grid eigenmode to 1 - 1.6e-6 or better.

## Ports and S-parameters

### Runs

Each measurement has two runs on the GPU with the same source and port 1:

- **Reference**: a straight guide with the height of the DUT. The grid is extended to the right, so that no wave from its right CPML can reach port 1 within the run. `reference_is_clean` checks this with the 1-cell-per-step bound. The DFTs at port 1 are the incident fields.
- **DUT**: the straight guide that ends in the right CPML, or the 90-degree bend.

The DFT monitors accumulate on the GPU at each step (`GpuYee::dft_line`, `accumulate`). Ez samples belong to t = (n + 1) dt and H samples to t = (n + 1/2) dt.

### Flux S-parameters

Port 1 subtracts the incident fields from the DUT fields: E_ref = E_tot - E_inc and H_ref = H_tot - H_inc. The flux through a port line is:

	P = 0.5 Re sum( E_avg conj(H) ) * s,   s = -1 for an x port (S_x = -Ez Hy), s = +1 for a y port (S_y = Ez Hx)

E_avg is the mean of the two Ez lines that bracket the H line. |S11|^2 = P_ref / P_in (P_ref counts the flux toward -x) and |S21|^2 = P_out / P_in.

### Modal S-parameters

The flux counts all power that crosses a port line, including radiation from the source. The modal S-parameters count only the fundamental mode. `mode_amplitudes` projects E_avg and H on the grid eigenmode u:

	e = <u, E_avg> / <u, u>,   h = <u, H> / <u, u>,   Y = 2 tan(beta / 2) / Omega
	forward = (e - s' h / Y) / 2,   backward = (e + s' h / Y) / 2,   s' = +1 (x port), -1 (y port)

For the leapfrog update, a forward mode has H = -s' Y E_avg exactly. Then |S11|^2 = |backward(E_ref, H_ref)|^2 / |forward(E_inc, H_inc)|^2 and |S21|^2 = |forward(port 2)|^2 / |forward(E_inc, H_inc)|^2. All ports have the same cross-section, so the modal power constant cancels.

### Straight guide

From `docs/results/sparams.log`:

| lambda0 | |S11|^2 (CPML modal reflection) | modal sum - 1 | flux sum - 1 | incident backward |
|---|---|---|---|---|
| 200.0 | -84.9 dB | 4.25e-6 | -9.29e-4 | -122.8 dB |
| 146.0 | -84.6 dB | -2.97e-9 | -6.12e-4 | -178.4 dB |
| 115.0 | -83.4 dB | 4.54e-9 | -4.10e-4 | -235.1 dB |

- The modal |S11|^2 + |S21|^2 equals 1 within 4.3e-6 at every frequency.
- The flux sum equals 1 within 9.3e-4, but only because port 1 is 350 cells from the source. `docs/results/sparams_port1.log` moves port 1. At 70, 150, 250 and 350 cells from the source, the flux sum - 1 at lambda0 = 200 is -2.18e-2, -8.12e-3, -2.61e-3 and -9.29e-4. The modal sum stays within 5.9e-6 at all four positions. Thus the flux deficit is source radiation that crosses port 1 and leaves through the top and bottom CPML before port 2. The modal S-parameters exclude it.
- The guided mode that runs into the 10-cell CPML reflects -83.4 dB or less. This closes the open item of v0.2.0 (a guided mode in the CPML).
- The forward and backward split of the incident wave puts -122.8 dB or less into the backward amplitude.

### Sharp 90-degree bend

From `docs/results/sparams.log`. IL = -10 log10 |S21|^2 and RL = -10 log10 |S11|^2:

| lambda0 | IL, mode | RL, mode | radiated, mode | IL, flux | RL, flux |
|---|---|---|---|---|---|
| 200.0 | 2.458 dB | 7.46 dB | 0.2526 | 2.179 dB | 7.42 dB |
| 146.0 | 3.748 dB | 6.69 dB | 0.3640 | 3.424 dB | 6.60 dB |
| 115.0 | 7.620 dB | 6.27 dB | 0.5911 | 5.585 dB | 5.68 dB |

At the band centre (lambda0 = 146 cells), the sharp bend transmits 42.2 % of the mode power and reflects 21.4 %. It radiates 36.4 %. The flux values are lower losses, because some of the corner radiation crosses the port lines. `bend_90_baseline_insertion_and_return_loss` fixes the centre values (IL 3.748 dB, RL 6.69 dB, +- 0.05 dB). The Stage 5 optimiser must beat this benchmark.

## CPML at long wavelengths

The CFS term alpha removes absorption at frequencies with omega < alpha. The Stage 2 default alpha_max = 0.05 came from a sweep at lambda0 = 40 cells (omega = 0.157). In this band, omega is 0.031 to 0.055. `docs/results/sparams_alpha.log` gives the CPML modal reflection at lambda0 = 200 and 115:

| CPML | lambda0 = 200 | lambda0 = 115 |
|---|---|---|
| 10 cells, alpha_max = 0.05 | -61.5 dB | -76.4 dB |
| 10 cells, alpha_max = 0.01 | -84.9 dB | -83.4 dB |
| 10 cells, alpha_max = 0 | -84.8 dB | -83.0 dB |
| 16 cells, alpha_max = 0.05 | -98.3 dB | -97.1 dB |

The Stage 4 setup thus uses alpha_max = 0.01 / scale. The library default stays at 0.05, and the documentation of `Pml::new` gives the rule: keep alpha_max below the lowest angular frequency of the run.

## Gates

| Test | Gate |
|---|---|
| `slab_mode_grid_dispersion_is_second_order` | gap ratio (scale 1 / scale 2) in 3.8 to 4.2 at all 15 frequencies; gap < 2e-3; one guided mode |
| `straight_guide_mode_energy_balance_and_cpml_reflection` | measured beta / grid beta - 1 within 1e-4; measured vs analytic within the grid gap + 1e-4; overlap > 1 - 1e-5; modal and flux sum within 1e-3; CPML modal reflection <= -70 dB; incident backward <= -60 dB |
| `bend_90_baseline_insertion_and_return_loss` | IL 3.748 +- 0.05 dB and RL 6.69 +- 0.05 dB at lambda0 = 146; 0 < S11, S21 and S11 + S21 < 1 at all frequencies |

Mutation runs:

| Mutation | Result |
|---|---|
| Sign of the mode split swapped (x and y ports) | the straight-guide and bend tests fail |
| H sample time (n + 1) dt instead of (n + 1/2) dt | the straight-guide test fails (incident backward -45.2 dB) |
| Inverse-iteration shift 0.97 lambda instead of lambda | no test fails. The iteration converges to the same mode, so the mutation is equivalent. |

Before the incident-backward gate existed, the H-time mutation passed all tests. The phase error cancels in every power ratio when the fields at a port travel in one direction only. The gate on the incident split detects it.

## Limits

- The ports measure the fundamental mode. Multimode ports are not implemented.
- The source is a soft line source. It also radiates backward, and the left CPML absorbs that wave. The flux S-parameters include its radiation that crosses the port lines.
- The bend is one geometry with a sharp corner. No other device is measured.
- The S-parameters are power ratios. The code does not report the phase of S11 or S21.
