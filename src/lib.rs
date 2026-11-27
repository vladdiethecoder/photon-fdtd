//! Minimal 2D TM_z Yee FDTD solver with a CFS-CPML boundary.
//!
//! Natural units: c = eps0 = mu0 = 1 and dx = dy = 1, so dt = S (the Courant
//! number). No physical constants enter the update, so no large or small
//! scale factors enter the floating-point arithmetic.
//!
//! Fields (TM_z): Ez at (i, j), Hx at (i, j + 1/2), Hy at (i + 1/2, j).
//! Every field is one flat row-major `Vec<f64>`: index k = j * nx + i,
//! so i (x) is the contiguous, fast index.
//!
//! The outer ring of Ez stays 0 (PEC). A CFS-CPML (see `cpml`) absorbs
//! outgoing waves before they reach the PEC wall. Its auxiliary psi arrays
//! exist only in the boundary strips.

pub mod adjoint;
pub mod cpml;
pub mod design;
pub mod export;
pub mod fixed1d;
pub mod gpu;
pub mod robust;
pub mod sparams;
pub mod stage6;
pub mod verify;

use rayon::prelude::*;

pub use cpml::{CpmlLayout, Pml};

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

/// Ez curl coefficient dt / eps_r. Shared by both backends.
#[inline(always)]
pub fn cb_coeff(courant: f64, eps: f64) -> f64 {
    courant / eps
}

pub struct Yee2d {
    pub nx: usize,
    pub ny: usize,
    /// Courant number S. In natural units dt = S.
    pub courant: f64,
    pub ez: Vec<f64>,
    pub hx: Vec<f64>,
    pub hy: Vec<f64>,
    /// Ez curl coefficient dt / eps_r, per cell.
    pub cb: Vec<f64>,
    pub eps_r: Vec<f64>,
    pub layout: CpmlLayout,
    /// x strips (ny rows of 2W): psi for dHy/dx in Ez, psi for dEz/dx in Hy.
    pub psi_ezx: Vec<f64>,
    pub psi_hyx: Vec<f64>,
    /// y strips (2W rows of nx): psi for dHx/dy in Ez, psi for dEz/dy in Hx.
    pub psi_ezy: Vec<f64>,
    pub psi_hxy: Vec<f64>,
}

impl Yee2d {
    /// `eps_r` is row-major, length nx * ny. Returns an error if the
    /// Courant number would let light outrun the grid.
    pub fn new(nx: usize, ny: usize, eps_r: Vec<f64>, pml: Pml, courant: f64) -> Result<Self, String> {
        check_setup(nx, ny, courant)?;
        if eps_r.len() != nx * ny {
            return Err(format!("bad grid: {nx} x {ny}, eps_r.len() = {}", eps_r.len()));
        }
        if eps_r.iter().any(|&e| !(e >= 1.0)) {
            return Err("eps_r must be >= 1 (CFL is set by the fastest medium, vacuum)".into());
        }
        let layout = CpmlLayout::new(nx, ny, pml, courant)?;
        let n = nx * ny;
        Ok(Self {
            nx,
            ny,
            courant,
            ez: vec![0.0; n],
            hx: vec![0.0; n],
            hy: vec![0.0; n],
            cb: eps_r.iter().map(|&e| cb_coeff(courant, e)).collect(),
            eps_r,
            psi_ezx: vec![0.0; layout.x_strip_len()],
            psi_hyx: vec![0.0; layout.x_strip_len()],
            psi_ezy: vec![0.0; layout.y_strip_len()],
            psi_hxy: vec![0.0; layout.y_strip_len()],
            layout,
        })
    }

    pub fn dt(&self) -> f64 {
        self.courant
    }

    /// One Yee step on one thread.
    pub fn step_serial(&mut self) {
        let (nx, ny, dt) = (self.nx, self.ny, self.courant);
        let w2 = 2 * self.layout.w;
        let Self { ez, hx, hy, cb, layout, psi_ezx, psi_hyx, psi_ezy, psi_hxy, .. } = self;
        for j in 0..ny - 1 {
            let s = j * nx;
            let py = layout.strip_row(j).map(|r| &mut psi_hxy[r * nx..(r + 1) * nx]);
            h_row(
                j,
                &mut hx[s..s + nx],
                &mut hy[s..s + nx],
                &ez[s..s + nx],
                &ez[s + nx..s + 2 * nx],
                &mut psi_hyx[j * w2..(j + 1) * w2],
                py,
                layout,
                dt,
            );
        }
        for j in 1..ny - 1 {
            let s = j * nx;
            let py = layout.strip_row(j).map(|r| &mut psi_ezy[r * nx..(r + 1) * nx]);
            e_row(
                j,
                &mut ez[s..s + nx],
                &hx[s..s + nx],
                &hx[s - nx..s],
                &hy[s..s + nx],
                &cb[s..s + nx],
                &mut psi_ezx[j * w2..(j + 1) * w2],
                py,
                layout,
            );
        }
    }

    /// One Yee step, rows split across the rayon thread pool.
    pub fn step_parallel(&mut self) {
        let (nx, ny, dt) = (self.nx, self.ny, self.courant);
        let w2 = 2 * self.layout.w;
        let Self { ez, hx, hy, cb, layout, psi_ezx, psi_hyx, psi_ezy, psi_hxy, .. } = self;
        let layout: &CpmlLayout = layout;
        {
            let ez: &[f64] = ez;
            hx.par_chunks_mut(nx)
                .zip(hy.par_chunks_mut(nx))
                .zip(psi_hyx.par_chunks_mut(w2))
                .zip(y_strip_rows(layout, psi_hxy).into_par_iter())
                .enumerate()
                .with_min_len(4)
                .for_each(|(j, (((hx_r, hy_r), px), py))| {
                    if j < ny - 1 {
                        let s = j * nx;
                        h_row(j, hx_r, hy_r, &ez[s..s + nx], &ez[s + nx..s + 2 * nx], px, py, layout, dt);
                    }
                });
        }
        let (hx, hy): (&[f64], &[f64]) = (hx, hy);
        ez.par_chunks_mut(nx)
            .zip(cb.par_chunks(nx))
            .zip(psi_ezx.par_chunks_mut(w2))
            .zip(y_strip_rows(layout, psi_ezy).into_par_iter())
            .enumerate()
            .with_min_len(4)
            .for_each(|(j, (((ez_r, cb_r), px), py))| {
                if j > 0 && j < ny - 1 {
                    let s = j * nx;
                    e_row(j, ez_r, &hx[s..s + nx], &hx[s - nx..s], &hy[s..s + nx], cb_r, px, py, layout);
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

/// One `Option` per grid row: the y-strip psi row of that grid row, if any.
fn y_strip_rows<'a>(layout: &CpmlLayout, psi: &'a mut [f64]) -> Vec<Option<&'a mut [f64]>> {
    let mut rows: Vec<Option<&'a mut [f64]>> = (0..layout.ny).map(|_| None).collect();
    for (r, ch) in psi.chunks_mut(layout.nx).enumerate() {
        rows[layout.strip_j(r)] = Some(ch);
    }
    rows
}

/// H update of row j (i = 0..nx-2), then the CPML psi corrections:
///   Hx = Hx - dt (dEz/dy / kappa_y) ;  Hy = Hy + dt (dEz/dx / kappa_x)
///   x strip: psi_hyx = b psi_hyx + a dEz/dx ; Hy += dt psi_hyx
///   y strip: psi_hxy = b psi_hxy + a dEz/dy ; Hx -= dt psi_hxy
/// The CUDA kernels in gpu.rs use the same expressions in the same order.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn h_row(
    j: usize,
    hx: &mut [f64],
    hy: &mut [f64],
    ez: &[f64],
    ez_up: &[f64],
    psi_x: &mut [f64],
    psi_y: Option<&mut [f64]>,
    lay: &CpmlLayout,
    dt: f64,
) {
    let n = hx.len() - 1;
    let iky = lay.y_half.ik[j];
    {
        let (hx, hy, ez_up, ikx) = (&mut hx[..n], &mut hy[..n], &ez_up[..n], &lay.x_half.ik[..n]);
        let ez = &ez[..n + 1];
        for i in 0..n {
            let e = ez[i];
            hx[i] = hx[i] - dt * ((ez_up[i] - e) * iky);
            hy[i] = hy[i] + dt * ((ez[i + 1] - e) * ikx[i]);
        }
    }
    for (c, p) in psi_x.iter_mut().enumerate() {
        let i = lay.strip_i(c);
        if i >= n {
            continue;
        }
        *p = lay.x_half.b[i] * *p + lay.x_half.a[i] * (ez[i + 1] - ez[i]);
        hy[i] = hy[i] + dt * *p;
    }
    if let Some(psi_y) = psi_y {
        let (a, b) = (lay.y_half.a[j], lay.y_half.b[j]);
        for i in 0..n {
            psi_y[i] = b * psi_y[i] + a * (ez_up[i] - ez[i]);
            hx[i] = hx[i] - dt * psi_y[i];
        }
    }
}

/// Ez update of row j (i = 1..nx-2), then the CPML psi corrections:
///   Ez = Ez + cb (dHy/dx / kappa_x - dHx/dy / kappa_y)
///   x strip: psi_ezx = b psi_ezx + a dHy/dx ; Ez += cb psi_ezx
///   y strip: psi_ezy = b psi_ezy + a dHx/dy ; Ez -= cb psi_ezy
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn e_row(
    j: usize,
    ez: &mut [f64],
    hx: &[f64],
    hx_dn: &[f64],
    hy: &[f64],
    cb: &[f64],
    psi_x: &mut [f64],
    psi_y: Option<&mut [f64]>,
    lay: &CpmlLayout,
) {
    let n = ez.len();
    let iky = lay.y_int.ik[j];
    {
        let (cb, hx, hx_dn, hy, ikx) = (&cb[..n], &hx[..n], &hx_dn[..n], &hy[..n], &lay.x_int.ik[..n]);
        for i in 1..n - 1 {
            ez[i] = ez[i] + cb[i] * ((hy[i] - hy[i - 1]) * ikx[i] - (hx[i] - hx_dn[i]) * iky);
        }
    }
    for (c, p) in psi_x.iter_mut().enumerate() {
        let i = lay.strip_i(c);
        if i < 1 || i >= n - 1 {
            continue;
        }
        *p = lay.x_int.b[i] * *p + lay.x_int.a[i] * (hy[i] - hy[i - 1]);
        ez[i] = ez[i] + cb[i] * *p;
    }
    if let Some(psi_y) = psi_y {
        let (a, b) = (lay.y_int.a[j], lay.y_int.b[j]);
        for i in 1..n - 1 {
            psi_y[i] = b * psi_y[i] + a * (hx[i] - hx_dn[i]);
            ez[i] = ez[i] - cb[i] * psi_y[i];
        }
    }
}

/// Gaussian-modulated sine, the source waveform used by the demos.
pub fn gaussian_sine(t: f64, f0: f64, t0: f64, tau: f64) -> f64 {
    let a = (t - t0) / tau;
    (-a * a).exp() * (2.0 * std::f64::consts::PI * f0 * t).sin()
}

/// First derivative of a Gaussian (zero mean, no DC content), peak value 1.
pub fn gaussian_derivative(t: f64, t0: f64, tau: f64) -> f64 {
    let a = (t - t0) / tau;
    -a * (-a * a).exp() * (2.0f64).sqrt() * (0.5f64).exp()
}
