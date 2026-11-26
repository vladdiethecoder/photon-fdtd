//! Verification scenarios. The `waveguide` binary prints them; `cargo test`
//! (tests/verification.rs) asserts on them. One implementation, two entry points.

use std::error::Error;

use crate::gpu::GpuYee;
use crate::{COURANT_MAX_2D, Yee2d, gaussian_sine};

/// Courant number used by every scenario: 99 % of the 2D limit.
pub fn courant() -> f64 {
    0.99 * COURANT_MAX_2D
}

pub const EPS_SI: f64 = 12.11;
pub const EPS_AIR: f64 = 1.0;

/// A dielectric strip of `core_w` rows across the middle of the grid (along x),
/// with a soft line source across the strip at x = `src_x`.
#[derive(Clone, Debug)]
pub struct Strip {
    pub nx: usize,
    pub ny: usize,
    pub sponge: usize,
    pub core_w: usize,
    pub src_x: usize,
    pub probe_x: usize,
    /// Free-space wavelength of the source carrier, in cells.
    pub lambda0: f64,
    /// Gaussian envelope centre and 1/e half-width, natural time units.
    pub t0: f64,
    pub tau: f64,
    pub t_end: f64,
    pub eps_core: f64,
}

pub struct StripReport {
    pub sim: Yee2d,
    /// Time-integrated Ez^2 at x = probe_x, per row.
    pub probe: Vec<f64>,
    pub peak_energy: f64,
    /// Grid energy at t_end / peak grid energy.
    pub energy_retained: f64,
    /// Grid energy inside the band rows / grid energy, at t_end.
    pub band_fraction: f64,
    /// Time-integrated Ez^2 inside the band / column total, at x = probe_x.
    pub probe_band_fraction: f64,
    /// RMS distance of the probe profile from the strip centre, cells.
    pub probe_rms_width: f64,
}

impl Strip {
    /// The reference case: 1200 x 600 grid, 16-cell strip, lambda0 = 40 cells.
    pub fn standard(eps_core: f64) -> Self {
        Self {
            nx: 1200,
            ny: 600,
            sponge: 40,
            core_w: 16,
            src_x: 100,
            probe_x: 400,
            lambda0: 40.0,
            t0: 300.0,
            tau: 80.0,
            t_end: 1800.0,
            eps_core,
        }
    }

    pub fn with_grid(mut self, nx: usize, ny: usize) -> Self {
        self.nx = nx;
        self.ny = ny;
        self
    }

    pub fn yc(&self) -> usize {
        self.ny / 2
    }

    /// Strip rows j0..j1.
    pub fn core(&self) -> (usize, usize) {
        (self.yc() - self.core_w / 2, self.yc() + self.core_w / 2)
    }

    /// Band = strip plus half a strip width of cladding on each side.
    pub fn band(&self) -> (usize, usize) {
        let (j0, j1) = self.core();
        (j0 - self.core_w / 2, j1 + self.core_w / 2)
    }

    pub fn steps(&self) -> usize {
        (self.t_end / courant()).ceil() as usize
    }

    pub fn build(&self) -> Yee2d {
        let mut eps = vec![EPS_AIR; self.nx * self.ny];
        let (j0, j1) = self.core();
        for j in j0..j1 {
            eps[j * self.nx..(j + 1) * self.nx].fill(self.eps_core);
        }
        Yee2d::new(self.nx, self.ny, eps, self.sponge, courant()).expect("valid setup")
    }

    /// Source value added after step n (time (n + 1) dt).
    pub fn source_amp(&self, n: usize) -> f64 {
        gaussian_sine((n + 1) as f64 * courant(), 1.0 / self.lambda0, self.t0, self.tau)
    }

    /// Full run on the CPU thread pool, with confinement metrics.
    pub fn run_cpu(&self) -> StripReport {
        let mut sim = self.build();
        let dt = sim.dt();
        let (j0, j1) = self.core();
        let (nx, ny) = (self.nx, self.ny);
        let mut probe = vec![0.0; ny];
        let mut peak_energy: f64 = 0.0;
        for n in 0..self.steps() {
            sim.step_parallel();
            sim.add_source_line(self.src_x, j0, j1, self.source_amp(n));
            for (j, p) in probe.iter_mut().enumerate() {
                let e = sim.ez[j * nx + self.probe_x];
                *p += e * e * dt;
            }
            if n % 20 == 0 {
                peak_energy = peak_energy.max(sim.energy());
            }
        }
        let (b0, b1) = self.band();
        let total = sim.energy();
        let probe_total: f64 = probe.iter().sum();
        let yc = self.yc() as f64 - 0.5;
        let var = probe.iter().enumerate().map(|(j, &p)| p * (j as f64 - yc).powi(2)).sum::<f64>() / probe_total;
        StripReport {
            energy_retained: total / peak_energy,
            band_fraction: sim.energy_rows(b0, b1) / total,
            probe_band_fraction: probe[b0..b1].iter().sum::<f64>() / probe_total,
            probe_rms_width: var.sqrt(),
            sim,
            probe,
            peak_energy,
        }
    }

    /// Ez after `steps` steps, CPU, one thread.
    pub fn run_serial_ez(&self, steps: usize) -> Vec<f64> {
        let mut sim = self.build();
        let (j0, j1) = self.core();
        for n in 0..steps {
            sim.step_serial();
            sim.add_source_line(self.src_x, j0, j1, self.source_amp(n));
        }
        sim.ez
    }

    /// Ez after `steps` steps, CPU thread pool.
    pub fn run_parallel_ez(&self, steps: usize) -> Vec<f64> {
        let mut sim = self.build();
        let (j0, j1) = self.core();
        for n in 0..steps {
            sim.step_parallel();
            sim.add_source_line(self.src_x, j0, j1, self.source_amp(n));
        }
        sim.ez
    }

    /// Ez after `steps` steps, CUDA device 0.
    pub fn run_gpu_ez(&self, steps: usize) -> Result<Vec<f64>, Box<dyn Error>> {
        let mut gpu = GpuYee::from_cpu(&self.build())?;
        let (j0, j1) = self.core();
        for n in 0..steps {
            gpu.step()?;
            gpu.add_source_line(self.src_x, j0, j1, self.source_amp(n))?;
        }
        gpu.download_ez()
    }
}

impl StripReport {
    /// x of the strongest 16-cell-smoothed Ez^2 on the centre row.
    pub fn pulse_x(&self, s: &Strip) -> usize {
        let row = &self.sim.ez[s.yc() * s.nx..(s.yc() + 1) * s.nx];
        let e2: Vec<f64> = row.iter().map(|e| e * e).collect();
        let win = |x: usize| e2[x - 8..x + 8].iter().sum::<f64>();
        (s.src_x + 20..s.nx - s.sponge).max_by(|&a, &b| win(a).total_cmp(&win(b))).unwrap()
    }
}

/// The confinement criterion used by the binary and the tests.
pub fn trapped(si: &StripReport, air: &StripReport) -> bool {
    si.band_fraction > 0.9 && si.probe_band_fraction > 0.9 && air.band_fraction < 0.2 && air.probe_band_fraction < 0.2
}

/// Number of values whose f64 bit patterns differ.
pub fn bit_mismatches(a: &[f64], b: &[f64]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// Absorber reflection measured against a reference grid whose walls the
/// wave never reaches (see docs/verification.md).
#[derive(Clone, Debug)]
pub struct AbsorberCheck {
    /// Test grid edge (square), sponge thickness, steps.
    pub n: usize,
    pub sponge: usize,
    pub n_ref: usize,
    pub steps: usize,
    pub lambda0: f64,
    pub t0: f64,
    pub tau: f64,
    /// Probe distance from the absorber interface, interior side, cells.
    pub gap: usize,
}

#[derive(Clone, Debug)]
pub struct ProbeReflection {
    pub name: &'static str,
    pub i: usize,
    pub j: usize,
    pub incident_peak: f64,
    pub max_diff: f64,
    /// Time-domain metric: 20 log10(max|test - ref| / max|ref|).
    pub r_db: f64,
    /// Same ratio of single-frequency DFT magnitudes at the carrier f0 = 1 / lambda0.
    pub r_f0_db: f64,
}

impl AbsorberCheck {
    pub fn standard(sponge: usize) -> Self {
        Self { n: 512, sponge, n_ref: 2048, steps: 1200, lambda0: 40.0, t0: 90.0, tau: 30.0, gap: 5 }
    }

    /// Test-grid probe cells: normal incidence on the +x wall, and 45 degrees toward the corner.
    pub fn probes(&self) -> [(&'static str, usize, usize); 2] {
        let c = self.n / 2;
        let p = self.n - 1 - self.sponge - self.gap;
        [("normal (+x wall)", p, c), ("diagonal (corner)", p, p)]
    }

    fn amp(&self, n: usize) -> f64 {
        gaussian_sine((n + 1) as f64 * courant(), 1.0 / self.lambda0, self.t0, self.tau)
    }

    /// Ez(t) at the given cells of an n x n air grid with a point source at the centre.
    fn trace(&self, n: usize, sponge: usize, cells: &[(usize, usize)]) -> Vec<Vec<f64>> {
        let mut sim = Yee2d::new(n, n, vec![EPS_AIR; n * n], sponge, courant()).expect("valid setup");
        let c = n / 2;
        let mut out = vec![Vec::with_capacity(self.steps); cells.len()];
        for s in 0..self.steps {
            sim.step_parallel();
            sim.add_source_line(c, c, c + 1, self.amp(s));
            for (o, &(i, j)) in out.iter_mut().zip(cells) {
                o.push(sim.ez[j * n + i]);
            }
        }
        out
    }

    /// Reference probe traces; independent of the sponge thickness of the test grid.
    pub fn reference(&self) -> Vec<Vec<f64>> {
        // The reference keeps its own sponge, but no wave reaches it in time:
        // checked by `reference_is_clean`.
        let off = (self.n_ref - self.n) / 2;
        let cells: Vec<_> = self.probes().iter().map(|&(_, i, j)| (i + off, j + off)).collect();
        self.trace(self.n_ref, self.sponge, &cells)
    }

    /// True if the reference probes are exactly independent of the reference
    /// absorber. The Yee stencil moves information at most one cell per step
    /// (faster than c, which moves 0.7 cells per step), so the bound is in
    /// steps: source -> absorber -> probe must take more than `steps` cells.
    pub fn reference_is_clean(&self) -> bool {
        let off = (self.n_ref - self.n) / 2;
        let edge = |i: usize, j: usize| i.min(self.n_ref - 1 - i).min(j).min(self.n_ref - 1 - j);
        let c = self.n_ref / 2;
        let d_src = edge(c, c).saturating_sub(self.sponge);
        self.probes().iter().all(|&(_, i, j)| d_src + edge(i + off, j + off).saturating_sub(self.sponge) > self.steps)
    }

    pub fn measure(&self, reference: &[Vec<f64>]) -> Vec<ProbeReflection> {
        let probes = self.probes();
        let cells: Vec<_> = probes.iter().map(|&(_, i, j)| (i, j)).collect();
        let test = self.trace(self.n, self.sponge, &cells);
        probes
            .iter()
            .zip(test.iter().zip(reference))
            .map(|(&(name, i, j), (t, r))| {
                let incident_peak = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let max_diff = t.iter().zip(r).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
                let r_db = 20.0 * (max_diff / incident_peak).log10();
                let diff: Vec<f64> = t.iter().zip(r).map(|(a, b)| a - b).collect();
                let f0 = 1.0 / self.lambda0;
                let r_f0_db = 20.0 * (dft_mag(&diff, f0) / dft_mag(r, f0)).log10();
                ProbeReflection { name, i, j, incident_peak, max_diff, r_db, r_f0_db }
            })
            .collect()
    }
}

/// |sum_n x[n] exp(-2 pi i f t_n)| with t_n = (n + 1) dt.
fn dft_mag(x: &[f64], f: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * f * courant();
    let (mut re, mut im) = (0.0, 0.0);
    for (n, v) in x.iter().enumerate() {
        let ph = w * (n + 1) as f64;
        re += v * ph.cos();
        im -= v * ph.sin();
    }
    re.hypot(im)
}
