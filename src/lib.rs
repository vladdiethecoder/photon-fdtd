//! Minimal 2D TM_z Yee FDTD solver.
//!
//! Natural units: c = eps0 = mu0 = 1 and dx = dy = 1, so dt = S (the Courant
//! number). No physical constants enter the update, so no large or small
//! scale factors enter the floating-point arithmetic.
//!
//! Fields (TM_z): Ez at (i, j), Hx at (i, j + 1/2), Hy at (i + 1/2, j).
//! Every field and coefficient is one flat row-major `Vec<f64>`:
//! index k = j * nx + i, so i (x) is the contiguous, fast index.
//!
//! The outer ring of Ez stays 0 (PEC). A graded, impedance-matched
//! conductive sponge of `sponge` cells absorbs outgoing waves before
//! they reach the PEC wall.

pub mod gpu;
pub mod verify;

use rayon::prelude::*;

/// 2D Yee stability limit: S = c dt / dx <= 1 / sqrt(2).
pub const COURANT_MAX_2D: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Reject a Courant number that lets light outrun the grid, and degenerate grids.
pub fn check_setup(nx: usize, ny: usize, courant: f64) -> Result<(), String> {
    if !(courant > 0.0 && courant <= COURANT_MAX_2D) {
        return Err(format!(
            "CFL violated: S = {courant} but the 2D Yee limit is 1/sqrt(2) = {COURANT_MAX_2D}"
        ));
    }
    if nx < 3 || ny < 3 {
        return Err(format!("bad grid: {nx} x {ny}"));
    }
    Ok(())
}

/// Update coefficients (ca, cb, hb) of cell (i, j). Shared by the CPU and GPU
/// constructors so both backends hold bitwise identical coefficients.
///
/// Sponge: conductivity graded as depth^3 over `sponge` cells, measured from
/// the nearest outer edge. sigma_max targets ~1e-6 normal-incidence reflection
/// from the PEC wall behind the sponge (round trip).
pub fn cell_coeffs(
    i: usize,
    j: usize,
    nx: usize,
    ny: usize,
    sponge: usize,
    courant: f64,
    eps: f64,
) -> (f64, f64, f64) {
    let dt = courant;
    let m = 3.0;
    let edge = i.min(nx - 1 - i).min(j).min(ny - 1 - j);
    let depth = if sponge > 0 && edge < sponge { (sponge - edge) as f64 / sponge as f64 } else { 0.0 };
    let sigma_max = if sponge > 0 { (m + 1.0) * (1e6f64).ln() / (2.0 * sponge as f64) } else { 0.0 };
    // sigma scaled by sqrt(eps) gives the same attenuation per cell in every medium.
    let sigma = sigma_max * depth.powf(m) * eps.sqrt();
    // Matched sponge: sigma_m / mu = sigma / eps, so E and H share one loss term.
    let loss = sigma * dt / (2.0 * eps);
    ((1.0 - loss) / (1.0 + loss), dt / (eps * (1.0 + loss)), dt / (1.0 + loss))
}

pub struct Yee2d {
    pub nx: usize,
    pub ny: usize,
    /// Courant number S. In natural units dt = S.
    pub courant: f64,
    pub ez: Vec<f64>,
    pub hx: Vec<f64>,
    pub hy: Vec<f64>,
    /// Ez self coefficient (loss).
    pub ca: Vec<f64>,
    /// Ez curl coefficient: dt / (eps_r (1 + loss)).
    pub cb: Vec<f64>,
    /// H curl coefficient: dt / (1 + loss). Matched sponge: the H self coefficient is `ca`.
    pub hb: Vec<f64>,
    pub eps_r: Vec<f64>,
}

impl Yee2d {
    /// `eps_r` is row-major, length nx * ny. Returns an error if the
    /// Courant number would let light outrun the grid.
    pub fn new(
        nx: usize,
        ny: usize,
        eps_r: Vec<f64>,
        sponge: usize,
        courant: f64,
    ) -> Result<Self, String> {
        check_setup(nx, ny, courant)?;
        if eps_r.len() != nx * ny {
            return Err(format!("bad grid: {nx} x {ny}, eps_r.len() = {}", eps_r.len()));
        }
        if eps_r.iter().any(|&e| !(e >= 1.0)) {
            return Err("eps_r must be >= 1 (CFL is set by the fastest medium, vacuum)".into());
        }
        let n = nx * ny;
        let (mut ca, mut cb, mut hb) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                (ca[k], cb[k], hb[k]) = cell_coeffs(i, j, nx, ny, sponge, courant, eps_r[k]);
            }
        }
        Ok(Self {
            nx,
            ny,
            courant,
            ez: vec![0.0; n],
            hx: vec![0.0; n],
            hy: vec![0.0; n],
            ca,
            cb,
            hb,
            eps_r,
        })
    }

    pub fn dt(&self) -> f64 {
        self.courant
    }

    /// One Yee step on one thread.
    pub fn step_serial(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let Self { ez, hx, hy, ca, cb, hb, .. } = self;
        for j in 0..ny - 1 {
            let s = j * nx;
            h_row(
                &mut hx[s..s + nx],
                &mut hy[s..s + nx],
                &ca[s..s + nx],
                &hb[s..s + nx],
                &ez[s..s + nx],
                &ez[s + nx..s + 2 * nx],
            );
        }
        for j in 1..ny - 1 {
            let s = j * nx;
            e_row(
                &mut ez[s..s + nx],
                &ca[s..s + nx],
                &cb[s..s + nx],
                &hx[s..s + nx],
                &hx[s - nx..s],
                &hy[s..s + nx],
            );
        }
    }

    /// One Yee step, rows split across the rayon thread pool.
    pub fn step_parallel(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let Self { ez, hx, hy, ca, cb, hb, .. } = self;
        {
            let ez: &[f64] = ez;
            hx.par_chunks_mut(nx)
                .zip(hy.par_chunks_mut(nx))
                .zip(ca.par_chunks(nx))
                .zip(hb.par_chunks(nx))
                .enumerate()
                .with_min_len(4)
                .for_each(|(j, (((hx_r, hy_r), ca_r), hb_r))| {
                    if j < ny - 1 {
                        let s = j * nx;
                        h_row(hx_r, hy_r, ca_r, hb_r, &ez[s..s + nx], &ez[s + nx..s + 2 * nx]);
                    }
                });
        }
        let (hx, hy): (&[f64], &[f64]) = (hx, hy);
        ez.par_chunks_mut(nx)
            .zip(ca.par_chunks(nx))
            .zip(cb.par_chunks(nx))
            .enumerate()
            .with_min_len(4)
            .for_each(|(j, ((ez_r, ca_r), cb_r))| {
                if j > 0 && j < ny - 1 {
                    let s = j * nx;
                    e_row(ez_r, ca_r, cb_r, &hx[s..s + nx], &hx[s - nx..s], &hy[s..s + nx]);
                }
            });
    }

    /// Soft source: add `amp` to Ez at column i, rows j0..j1.
    pub fn add_source_line(&mut self, i: usize, j0: usize, j1: usize, amp: f64) {
        for j in j0..j1 {
            let k = j * self.nx + i;
            self.ez[k] = self.ez[k] + amp;
        }
    }

    /// Discrete field energy 0.5 * (eps Ez^2 + Hx^2 + Hy^2) in rows j0..j1.
    pub fn energy_rows(&self, j0: usize, j1: usize) -> f64 {
        let mut w = 0.0;
        for k in j0 * self.nx..j1 * self.nx {
            w += self.eps_r[k] * self.ez[k] * self.ez[k] + self.hx[k] * self.hx[k] + self.hy[k] * self.hy[k];
        }
        0.5 * w
    }

    pub fn energy(&self) -> f64 {
        self.energy_rows(0, self.ny)
    }
}

/// Hx = ca Hx - hb dEz/dy ; Hy = ca Hy + hb dEz/dx  (one row, i = 0..nx-2).
/// The matched sponge gives H the same self coefficient `ca` as Ez.
#[inline(always)]
fn h_row(hx: &mut [f64], hy: &mut [f64], ca: &[f64], hb: &[f64], ez: &[f64], ez_up: &[f64]) {
    let n = hx.len() - 1;
    let (hx, hy, ca, hb, ez_up) = (&mut hx[..n], &mut hy[..n], &ca[..n], &hb[..n], &ez_up[..n]);
    let ez = &ez[..n + 1];
    for i in 0..n {
        let e = ez[i];
        hx[i] = ca[i] * hx[i] - hb[i] * (ez_up[i] - e);
        hy[i] = ca[i] * hy[i] + hb[i] * (ez[i + 1] - e);
    }
}

/// Ez = ca Ez + cb (dHy/dx - dHx/dy)  (one row, i = 1..nx-2).
#[inline(always)]
fn e_row(ez: &mut [f64], ca: &[f64], cb: &[f64], hx: &[f64], hx_dn: &[f64], hy: &[f64]) {
    let n = ez.len();
    let (ca, cb, hx, hx_dn, hy) = (&ca[..n], &cb[..n], &hx[..n], &hx_dn[..n], &hy[..n]);
    for i in 1..n - 1 {
        ez[i] = ca[i] * ez[i] + cb[i] * ((hy[i] - hy[i - 1]) - (hx[i] - hx_dn[i]));
    }
}

/// Gaussian-modulated sine, the source waveform used by the demos.
pub fn gaussian_sine(t: f64, f0: f64, t0: f64, tau: f64) -> f64 {
    let a = (t - t0) / tau;
    (-a * a).exp() * (2.0 * std::f64::consts::PI * f0 * t).sin()
}
