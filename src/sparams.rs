//! Slab waveguide modes, eigenmode injection, Poynting-flux ports and
//! S-parameters (Stage 4).
//!
//! Natural units as in the rest of the crate: c = eps0 = mu0 = dx = 1, so a
//! frequency f is in cycles per time unit and k0 = 2 pi f per cell.
//!
//! Polarization. Ez is out of plane and the slab walls lie along x, so Ez and
//! dEz/dy are continuous across the interfaces. The fundamental even mode of a
//! symmetric slab of width d obeys
//!   kappa = k0 sqrt(n1^2 - neff^2),  gamma = k0 sqrt(neff^2 - n2^2),
//!   tan(kappa d / 2) = gamma / kappa.
//!
//! The Yee grid has its own (discrete) eigenproblem. With Omega =
//! (2 / dt) sin(omega dt / 2) and B = 2 sin(beta / 2), a mode
//! Ez = u_j exp(i (omega t - beta x)) of the leapfrog update obeys exactly
//!   u_{j+1} - 2 u_j + u_{j-1} + eps_j Omega^2 u_j = B^2 u_j.
//! `discrete_mode` solves this tridiagonal problem. Its fundamental
//! eigenvector is the injected source profile, and its beta is the exact
//! propagation constant of the grid. The analytic root is the physical
//! reference; the difference of the two is the grid-dispersion error.

use std::error::Error;
use std::f64::consts::PI;

use crate::gpu::{DftLine, Field, GpuYee};
use crate::verify::{EPS_AIR, EPS_SI, courant};
use crate::{Pml, gaussian_sine};

/// Fundamental even mode of the continuous symmetric slab.
#[derive(Clone, Copy, Debug)]
pub struct SlabMode {
    pub neff: f64,
    pub kappa: f64,
    pub gamma: f64,
    pub beta: f64,
}

/// Solve tan(kappa d / 2) = gamma / kappa for the fundamental even mode by
/// bisection on neff. `d` is the slab width in cells and `freq` the frequency.
pub fn slab_even_mode(d: f64, n1: f64, n2: f64, freq: f64) -> SlabMode {
    let k0 = 2.0 * PI * freq;
    let parts = |neff: f64| {
        let kappa = k0 * (n1 * n1 - neff * neff).max(0.0).sqrt();
        let gamma = k0 * (neff * neff - n2 * n2).max(0.0).sqrt();
        (kappa, gamma)
    };
    // kappa d / 2 < pi / 2 on the fundamental branch.
    let lo_sq = (n1 * n1 - (PI / (k0 * d)).powi(2)).max(n2 * n2);
    let (mut lo, mut hi) = (lo_sq.sqrt(), n1);
    // F(neff) = kappa tan(kappa d / 2) - gamma decreases from > 0 at lo to < 0 at hi.
    let f = |neff: f64| {
        let (kappa, gamma) = parts(neff);
        kappa * (kappa * d / 2.0).tan() - gamma
    };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 { lo = mid } else { hi = mid }
    }
    let neff = 0.5 * (lo + hi);
    let (kappa, gamma) = parts(neff);
    SlabMode { neff, kappa, gamma, beta: k0 * neff }
}

impl SlabMode {
    /// Ez profile at distance y from the slab centre: cos(kappa y) in the core,
    /// cos(kappa d / 2) exp(-gamma (|y| - d / 2)) in the cladding.
    pub fn profile(&self, d: f64, y: f64) -> f64 {
        let a = y.abs();
        if a <= d / 2.0 { (self.kappa * a).cos() } else { (self.kappa * d / 2.0).cos() * (-self.gamma * (a - d / 2.0)).exp() }
    }
}

/// Fundamental eigenmode of the discrete Yee transverse operator.
#[derive(Clone, Debug)]
pub struct DiscreteMode {
    /// Exact propagation constant of the grid (per cell).
    pub beta: f64,
    /// Ez profile on rows 0..ny (rows 0 and ny - 1 are PEC), max = 1.
    pub profile: Vec<f64>,
    /// Number of guided eigenmodes (eigenvalue above the cladding light line).
    pub guided: usize,
}

/// Number of eigenvalues of the tridiagonal matrix (diag a, off-diagonal 1)
/// that are smaller than x (Sturm sequence).
fn count_below(a: &[f64], x: f64) -> usize {
    let mut q = 1.0;
    let mut n = 0;
    for (k, &ak) in a.iter().enumerate() {
        q = (ak - x) - if k == 0 { 0.0 } else { 1.0 / q };
        if q == 0.0 {
            q = -1e-300;
        }
        if q < 0.0 {
            n += 1;
        }
    }
    n
}

/// Solve (tridiag(1, a - shift, 1)) x = b (Thomas algorithm).
fn solve_tridiag(a: &[f64], shift: f64, b: &[f64]) -> Vec<f64> {
    let m = a.len();
    let (mut c, mut d) = (vec![0.0; m], vec![0.0; m]);
    let mut w = a[0] - shift;
    c[0] = 1.0 / w;
    d[0] = b[0] / w;
    for k in 1..m {
        w = (a[k] - shift) - c[k - 1];
        c[k] = 1.0 / w;
        d[k] = (b[k] - d[k - 1]) / w;
    }
    let mut x = vec![0.0; m];
    x[m - 1] = d[m - 1];
    for k in (0..m - 1).rev() {
        x[k] = d[k] - c[k] * x[k + 1];
    }
    x
}

/// Fundamental mode of the Yee grid for the eps column `eps` (rows 0..ny).
pub fn discrete_mode(eps: &[f64], freq: f64, dt: f64) -> DiscreteMode {
    let omega_d = 2.0 / dt * (PI * freq * dt).sin();
    let w2 = omega_d * omega_d;
    // Unknowns: rows 1..ny-1 (rows 0 and ny-1 are PEC).
    let a: Vec<f64> = eps[1..eps.len() - 1].iter().map(|&e| -2.0 + e * w2).collect();
    let m = a.len();
    let (mut lo, mut hi) =
        (a.iter().cloned().fold(f64::INFINITY, f64::min) - 2.0, a.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 2.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if count_below(&a, mid) == m { hi = mid } else { lo = mid }
    }
    let lambda = 0.5 * (lo + hi);
    let clad = eps.iter().cloned().fold(f64::INFINITY, f64::min) * w2;
    let guided = m - count_below(&a, clad);
    // Inverse iteration with a shift just above the eigenvalue.
    let shift = lambda + 1e-9 * lambda.abs().max(1e-6);
    let mut v = vec![1.0; m];
    for _ in 0..6 {
        v = solve_tridiag(&a, shift, &v);
        let n = v.iter().fold(0.0f64, |s, x| s.max(x.abs()));
        v.iter_mut().for_each(|x| *x /= n);
    }
    let sign = if v[m / 2] < 0.0 { -1.0 } else { 1.0 };
    let mut profile = vec![0.0; eps.len()];
    for (k, x) in v.iter().enumerate() {
        profile[k + 1] = sign * x;
    }
    let b = lambda.max(0.0).sqrt();
    DiscreteMode { beta: 2.0 * (b / 2.0).min(1.0).asin(), profile, guided }
}

/// Port test geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geometry {
    /// Straight guide that runs into the right CPML.
    Straight,
    /// Sharp 90-degree bend: the guide turns up at x = corner and runs into the top CPML.
    Bend,
}

/// Which run of a port measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Run {
    /// Straight guide, extended to the right so that no wave from its right
    /// wall can reach port 1 within the run (incident-field calibration).
    Reference,
    /// The device under test.
    Dut,
}

/// Port measurement setup. All lengths scale with `scale`; frequencies scale with 1 / scale.
#[derive(Clone, Debug)]
pub struct PortSetup {
    pub geometry: Geometry,
    pub scale: usize,
    pub pml: Pml,
    /// Slab width in cells.
    pub core_w: usize,
    /// Cladding between the slab and the CPML (cells).
    pub clad: usize,
    /// Band in natural units; single-mode for the 16-cell slab when lambda0 > 107 cells.
    pub f_min: f64,
    pub f_max: f64,
    pub nf: usize,
    pub src_x: usize,
    pub port1_x: usize,
    /// Bend DUT only: eps of the design box cells (row-major, b * n + a).
    pub design: Option<std::sync::Arc<(crate::design::DesignBox, Vec<f64>)>>,
    /// Extra settling time added to `steps` (natural time units).
    pub extra_time: f64,
}

/// Raw port data of one run.
pub struct PortData {
    pub freqs: Vec<f64>,
    /// Port 1 (x = port1_x): Ez at x1 and x1 + 1, Hy at x1 + 1/2, rows p0..p1.
    pub p1: [Vec<Vec<(f64, f64)>>; 3],
    /// Port 2 (DUT only): Ez at the port and the next line, H across.
    pub p2: Option<[Vec<Vec<(f64, f64)>>; 3]>,
    /// Ez along the slab centre row from port1_x (for beta), one row per frequency.
    pub axis: Vec<Vec<(f64, f64)>>,
    pub steps: usize,
    /// max over the last 5 % of steps of sum(Ez^2) / its max over the run.
    pub tail_energy: f64,
}

#[derive(Clone, Debug)]
pub struct SParams {
    pub freqs: Vec<f64>,
    pub p_in: Vec<f64>,
    /// |S11|^2 and |S21|^2 from the Poynting flux through the full port lines
    /// (power ratios). Radiation that crosses a port line counts as well.
    pub s11: Vec<f64>,
    pub s21: Vec<f64>,
    /// |S11|^2 and |S21|^2 of the fundamental mode: forward and backward mode
    /// amplitudes from the projection of Ez and H on the grid eigenmode.
    pub s11_mode: Vec<f64>,
    pub s21_mode: Vec<f64>,
    /// |backward|^2 / |forward|^2 of the incident field at port 1 (no subtraction).
    /// The incident wave travels forward only, so this checks the mode split.
    pub inc_backward: Vec<f64>,
}

/// Forward and backward amplitudes of the fundamental mode at a port.
///
/// `u` is the discrete eigenmode on the port line and `beta` its propagation
/// constant at frequency `freq`. `ea`, `eb` are the DFTs of Ez on the two lines
/// that bracket the H line `h`. For the leapfrog update, a mode travelling in
/// the +normal direction has H = -s Y E_avg with Y = 2 tan(beta / 2) / Omega,
/// where E_avg is the mean of the two Ez lines, s = 1 for an x port (H = Hy)
/// and s = -1 for a y port (H = Hx).
/// Returns (forward, backward) as complex amplitudes of the normalised profile.
pub fn mode_amplitudes(
    u: &[f64],
    beta: f64,
    freq: f64,
    ea: &[(f64, f64)],
    eb: &[(f64, f64)],
    h: &[(f64, f64)],
    x_port: bool,
) -> ((f64, f64), (f64, f64)) {
    let dt = courant();
    let omega_d = 2.0 / dt * (PI * freq * dt).sin();
    let y = 2.0 * (beta / 2.0).tan() / omega_d;
    let uu: f64 = u.iter().map(|v| v * v).sum();
    let (mut e, mut hh) = ((0.0, 0.0), (0.0, 0.0));
    for k in 0..u.len() {
        let m = c_add(ea[k], eb[k]);
        e = c_add(e, (0.5 * m.0 * u[k], 0.5 * m.1 * u[k]));
        hh = c_add(hh, (h[k].0 * u[k], h[k].1 * u[k]));
    }
    let (e, hh) = ((e.0 / uu, e.1 / uu), (hh.0 / uu, hh.1 / uu));
    let s = if x_port { 1.0 } else { -1.0 };
    // forward: h = -s Y e_f ; backward: h = +s Y e_b
    let hy = (hh.0 * s / y, hh.1 * s / y);
    (((e.0 - hy.0) * 0.5, (e.1 - hy.1) * 0.5), ((e.0 + hy.0) * 0.5, (e.1 + hy.1) * 0.5))
}

fn norm2(a: (f64, f64)) -> f64 {
    a.0 * a.0 + a.1 * a.1
}

fn c_add(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}
fn c_sub(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}
/// Re(a conj(b))
fn re_mul_conj(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1
}

/// Time-averaged flux through a port line: 0.5 Re sum(E_mid conj(H)) * sign,
/// with E_mid the average of the two Ez lines that bracket the H line.
/// x port: S_x = -Ez Hy (sign = -1). y port: S_y = Ez Hx (sign = +1).
pub fn port_flux(ea: &[(f64, f64)], eb: &[(f64, f64)], h: &[(f64, f64)], sign: f64) -> f64 {
    let mut p = 0.0;
    for k in 0..h.len() {
        let e = c_add(ea[k], eb[k]);
        p += re_mul_conj((e.0 * 0.5, e.1 * 0.5), h[k]);
    }
    0.5 * sign * p
}

impl PortSetup {
    /// The bend with a design box: `rho_bar` per box cell (0 = air, 1 = silicon).
    pub fn with_design(mut self, bx: &crate::design::DesignBox, rho_bar: &[f64]) -> Self {
        let eps = rho_bar.iter().map(|&r| crate::design::DesignBox::eps(r)).collect();
        self.design = Some(std::sync::Arc::new((bx.clone(), eps)));
        self
    }

    /// The standard setup at resolution `scale` (1: 16-cell slab).
    pub fn standard(geometry: Geometry, scale: usize) -> Self {
        let s = scale;
        Self {
            geometry,
            scale,
            // alpha_max must stay below the lowest band frequency omega = 2 pi f_min
            // (0.031 at scale 1), or the CFS term switches the absorption off.
            pml: Pml { alpha_max: 0.01 / s as f64, ..Pml::new(10) },
            core_w: 16 * s,
            clad: 150 * s,
            f_min: 1.0 / (200.0 * s as f64),
            f_max: 1.0 / (115.0 * s as f64),
            nf: 15,
            src_x: 10 + 40 * s,
            port1_x: 10 + 390 * s,
            design: None,
            extra_time: 0.0,
        }
    }

    pub fn freqs(&self) -> Vec<f64> {
        (0..self.nf).map(|k| self.f_min + (self.f_max - self.f_min) * k as f64 / (self.nf - 1) as f64).collect()
    }

    pub fn fc(&self) -> f64 {
        0.5 * (self.f_min + self.f_max)
    }

    /// Gaussian envelope: amplitude e^-2 at the band edges.
    pub fn tau(&self) -> f64 {
        std::f64::consts::SQRT_2 / (PI * 0.5 * (self.f_max - self.f_min))
    }

    pub fn t0(&self) -> f64 {
        4.5 * self.tau()
    }

    /// First slab row.
    pub fn j0(&self) -> usize {
        self.pml.cells + self.clad
    }

    /// Corner column of the bend (left edge of the vertical arm).
    pub fn corner_x(&self) -> usize {
        self.port1_x + 200 * self.scale
    }

    /// Port 2: x line (straight) or y line (bend).
    pub fn port2(&self) -> usize {
        match self.geometry {
            Geometry::Straight => self.dut_grid().0 - self.pml.cells - 80 * self.scale,
            Geometry::Bend => self.j0() + self.core_w + 250 * self.scale,
        }
    }

    /// DUT grid size.
    pub fn dut_grid(&self) -> (usize, usize) {
        let s = self.scale;
        let p = self.pml.cells;
        match self.geometry {
            Geometry::Straight => (700 * s, 2 * (p + self.clad) + self.core_w),
            Geometry::Bend => {
                let nx = self.corner_x() + self.core_w + self.clad + p;
                let ny = self.j0() + self.core_w + 250 * s + 80 * s + p;
                (nx, ny)
            }
        }
    }

    /// Smallest group velocity over the band, from the discrete beta(f).
    pub fn vg_min(&self) -> f64 {
        let eps = self.eps_column();
        let df = 1e-3 * (self.f_max - self.f_min);
        self.freqs()
            .iter()
            .map(|&f| {
                let b1 = discrete_mode(&eps, f - df, courant()).beta;
                let b2 = discrete_mode(&eps, f + df, courant()).beta;
                2.0 * PI * 2.0 * df / (b2 - b1)
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Steps: pulse, the longest guided path (source - port 2, and source -
    /// bend - port 1 for the reflection), and settling time.
    pub fn steps(&self) -> usize {
        let path = match self.geometry {
            Geometry::Straight => (self.port2() - self.src_x) as f64,
            Geometry::Bend => ((self.corner_x() - self.src_x) + (self.port2() - self.j0())).max(2 * (self.corner_x() - self.src_x)) as f64,
        };
        let t = self.t0() + 4.5 * self.tau() + path / self.vg_min() + 2000.0 * self.scale as f64 + self.extra_time;
        (t / courant()).ceil() as usize
    }

    /// Reference grid: the DUT height, extended to the right so that
    /// source -> right CPML strip -> port 1 is longer than `steps` cells.
    pub fn reference_grid(&self) -> (usize, usize) {
        let (_, ny) = self.dut_grid();
        let w = self.pml.cells + 1;
        let need = (self.steps() + self.src_x + self.port1_x) / 2 + w + 2;
        (need.max(self.dut_grid().0), ny)
    }

    /// True if no wave from the reference grid's right CPML can reach port 1 (1 cell per step bound).
    pub fn reference_is_clean(&self) -> bool {
        let (nx, _) = self.reference_grid();
        let strip = nx - 1 - (self.pml.cells + 1);
        (strip - self.src_x) + (strip - self.port1_x) > self.steps()
    }

    pub fn eps_at(&self, run: Run, i: usize, j: usize) -> f64 {
        let (j0, j1) = (self.j0(), self.j0() + self.core_w);
        let horizontal = (j0..j1).contains(&j);
        let si = match (run, self.geometry) {
            (Run::Reference, _) | (Run::Dut, Geometry::Straight) => horizontal,
            (Run::Dut, Geometry::Bend) => {
                if let Some(d) = &self.design
                    && d.0.contains(i, j)
                {
                    return d.1[(j - d.0.y0) * d.0.n + (i - d.0.x0)];
                }
                let (x0, x1) = (self.corner_x(), self.corner_x() + self.core_w);
                (horizontal && i < x1) || ((x0..x1).contains(&i) && j >= j0)
            }
        };
        if si { EPS_SI } else { EPS_AIR }
    }

    /// eps column of the straight input guide.
    pub fn eps_column(&self) -> Vec<f64> {
        let (_, ny) = self.dut_grid();
        (0..ny).map(|j| self.eps_at(Run::Reference, 0, j)).collect()
    }

    /// Run one simulation on the GPU and return the port DFTs.
    pub fn run(&self, run: Run) -> Result<PortData, Box<dyn Error>> {
        let (nx, ny) = match run {
            Run::Reference => self.reference_grid(),
            Run::Dut => self.dut_grid(),
        };
        let mut g = GpuYee::new(nx, ny, self.pml, courant(), &|i, j| self.eps_at(run, i, j))?;
        let dt = courant();
        let freqs = self.freqs();
        let mode = discrete_mode(&self.eps_column(), self.fc(), dt);
        // Source rows: the slab +- 6 decay lengths, clipped to the interior.
        let p = self.pml.cells;
        let (r0, r1) = (p + 1, ny - p - 1);
        let prof = g.upload_profile(&mode.profile[r0..r1])?;
        // Port 1 spans the full interior height.
        let x1 = self.port1_x;
        let n1 = r1 - r0;
        let mut m1 = [
            g.dft_line(Field::Ez, (x1, r0), n1, false, &freqs)?,
            g.dft_line(Field::Ez, (x1 + 1, r0), n1, false, &freqs)?,
            g.dft_line(Field::Hy, (x1, r0), n1, false, &freqs)?,
        ];
        let mut m2: Option<[DftLine; 3]> = match (run, self.geometry) {
            (Run::Reference, _) => None,
            (Run::Dut, Geometry::Straight) => {
                let x2 = self.port2();
                Some([
                    g.dft_line(Field::Ez, (x2, r0), n1, false, &freqs)?,
                    g.dft_line(Field::Ez, (x2 + 1, r0), n1, false, &freqs)?,
                    g.dft_line(Field::Hy, (x2, r0), n1, false, &freqs)?,
                ])
            }
            (Run::Dut, Geometry::Bend) => {
                let y2 = self.port2();
                let (c0, c1) = (p + 1, nx - p - 1);
                Some([
                    g.dft_line(Field::Ez, (c0, y2), c1 - c0, true, &freqs)?,
                    g.dft_line(Field::Ez, (c0, y2 + 1), c1 - c0, true, &freqs)?,
                    g.dft_line(Field::Hx, (c0, y2), c1 - c0, true, &freqs)?,
                ])
            }
        };
        let jc = self.j0() + self.core_w / 2;
        let axis_n = 300 * self.scale;
        let mut axis = match run {
            Run::Reference => Some(g.dft_line(Field::Ez, (x1, jc), axis_n, true, &freqs)?),
            Run::Dut => None,
        };
        let steps = self.steps();
        let (mut peak, mut tail) = (0.0f64, 0.0f64);
        let probe_every = (steps / 100).max(1);
        for n in 0..steps {
            g.step()?;
            let t_e = (n + 1) as f64 * dt;
            g.add_source_profile(self.src_x, r0, &prof, gaussian_sine(t_e, self.fc(), self.t0(), self.tau()))?;
            let t_h = (n as f64 + 0.5) * dt;
            g.accumulate(&mut m1[0], t_e)?;
            g.accumulate(&mut m1[1], t_e)?;
            g.accumulate(&mut m1[2], t_h)?;
            if let Some(m) = m2.as_mut() {
                g.accumulate(&mut m[0], t_e)?;
                g.accumulate(&mut m[1], t_e)?;
                g.accumulate(&mut m[2], t_h)?;
            }
            if let Some(a) = axis.as_mut() {
                g.accumulate(a, t_e)?;
            }
            if n % probe_every == 0 {
                let e: f64 = g.download(Field::Ez)?.iter().map(|v| v * v).sum();
                peak = peak.max(e);
                if n >= steps - steps / 20 {
                    tail = tail.max(e);
                }
            }
        }
        let s = g.stream();
        let dl = |m: &DftLine| m.download(s);
        Ok(PortData {
            freqs,
            p1: [dl(&m1[0])?, dl(&m1[1])?, dl(&m1[2])?],
            p2: match &m2 {
                Some(m) => Some([dl(&m[0])?, dl(&m[1])?, dl(&m[2])?]),
                None => None,
            },
            axis: match &axis {
                Some(a) => dl(a)?,
                None => Vec::new(),
            },
            steps,
            tail_energy: tail / peak,
        })
    }

    /// S-parameters from a reference run and a DUT run with the same source and port 1.
    pub fn sparams(&self, reference: &PortData, dut: &PortData) -> SParams {
        let nf = reference.freqs.len();
        let sign2 = match self.geometry {
            Geometry::Straight => -1.0,
            Geometry::Bend => 1.0,
        };
        let mut out = SParams {
            freqs: reference.freqs.clone(),
            p_in: vec![],
            s11: vec![],
            s21: vec![],
            s11_mode: vec![],
            s21_mode: vec![],
            inc_backward: vec![],
        };
        let p = self.pml.cells;
        let (nx, ny) = self.dut_grid();
        let col = self.eps_column();
        let row: Vec<f64> = match self.geometry {
            Geometry::Straight => col.clone(),
            Geometry::Bend => (0..nx).map(|i| self.eps_at(Run::Dut, i, self.port2())).collect(),
        };
        let (r0, r1) = (p + 1, ny - p - 1);
        let (c0, c1) = match self.geometry {
            Geometry::Straight => (r0, r1),
            Geometry::Bend => (p + 1, nx - p - 1),
        };
        for f in 0..nf {
            let [ia, ib, ih] = [&reference.p1[0][f], &reference.p1[1][f], &reference.p1[2][f]];
            let [ta, tb, th] = [&dut.p1[0][f], &dut.p1[1][f], &dut.p1[2][f]];
            let p_in = port_flux(ia, ib, ih, -1.0);
            let ra: Vec<_> = ta.iter().zip(ia.iter()).map(|(&t, &i)| c_sub(t, i)).collect();
            let rb: Vec<_> = tb.iter().zip(ib.iter()).map(|(&t, &i)| c_sub(t, i)).collect();
            let rh: Vec<_> = th.iter().zip(ih.iter()).map(|(&t, &i)| c_sub(t, i)).collect();
            let p_ref = -port_flux(&ra, &rb, &rh, -1.0);
            let p2 = dut.p2.as_ref().expect("DUT port 2");
            let p_out = port_flux(&p2[0][f], &p2[1][f], &p2[2][f], sign2);
            out.p_in.push(p_in);
            out.s11.push(p_ref / p_in);
            out.s21.push(p_out / p_in);
            let freq = reference.freqs[f];
            let m1 = discrete_mode(&col, freq, courant());
            let m2 = discrete_mode(&row, freq, courant());
            let (inc_f, inc_b) = mode_amplitudes(&m1.profile[r0..r1], m1.beta, freq, ia, ib, ih, true);
            out.inc_backward.push(norm2(inc_b) / norm2(inc_f));
            let (_, ref_b) = mode_amplitudes(&m1.profile[r0..r1], m1.beta, freq, &ra, &rb, &rh, true);
            let x2 = self.geometry == Geometry::Straight;
            let (out_f, _) = mode_amplitudes(&m2.profile[c0..c1], m2.beta, freq, &p2[0][f], &p2[1][f], &p2[2][f], x2);
            out.s11_mode.push(norm2(ref_b) / norm2(inc_f));
            out.s21_mode.push(norm2(out_f) / norm2(inc_f));
        }
        out
    }
}

/// beta measured from the DFT phase of Ez along the slab centre (least-squares slope).
pub fn measured_beta(axis_row: &[(f64, f64)]) -> f64 {
    let mut phase: Vec<f64> = axis_row.iter().map(|&(re, im)| im.atan2(re)).collect();
    for k in 1..phase.len() {
        while phase[k] - phase[k - 1] > PI {
            phase[k] -= 2.0 * PI;
        }
        while phase[k] - phase[k - 1] < -PI {
            phase[k] += 2.0 * PI;
        }
    }
    let n = phase.len() as f64;
    let mx = (n - 1.0) / 2.0;
    let my = phase.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (k, p) in phase.iter().enumerate() {
        sxy += (k as f64 - mx) * (p - my);
        sxx += (k as f64 - mx).powi(2);
    }
    // e^{-i beta x} under the e^{-i omega t} DFT: phase = -beta x.
    -sxy / sxx
}

pub fn db(x: f64) -> f64 {
    10.0 * x.log10()
}

/// Normalised overlap |<a, u>|^2 / (<a, a> <u, u>) of the analytic slab profile
/// `a` (sampled at the Ez rows) and the discrete eigenmode `u` of the setup.
pub fn analytic_overlap(setup: &PortSetup, freq: f64) -> f64 {
    let eps = setup.eps_column();
    let u = discrete_mode(&eps, freq, courant()).profile;
    let d = setup.core_w as f64;
    // The slab occupies rows j0..j0+d, so its walls are at j0 - 1/2 and j0 + d - 1/2.
    let yc = setup.j0() as f64 + (d - 1.0) / 2.0;
    let m = slab_even_mode(d, EPS_SI.sqrt(), 1.0, freq);
    let a: Vec<f64> = (0..eps.len()).map(|j| m.profile(d, j as f64 - yc)).collect();
    let (au, aa, uu) = a.iter().zip(&u).fold((0.0, 0.0, 0.0), |(x, y, z), (p, q)| (x + p * q, y + p * p, z + q * q));
    au * au / (aa * uu)
}
