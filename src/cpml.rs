//! CFS-CPML (Roden and Gedney, 2000) profiles and the strip layout.
//!
//! The layout is the single source of the PML geometry. The CPU solver and the
//! CUDA solver both read their 1D coefficient tables and their strip index maps
//! from one `CpmlLayout`, so the two backends cannot disagree on a cell.
//!
//! Geometry. `cells` = N. Each wall is a PEC at Ez index 0 or n - 1. The PML
//! depth of a position p (integer for Ez, p + 1/2 for H) is
//! rho = max((N - p) / N, (p - (n - 1 - N)) / N, 0), so rho = 0 at the inner
//! interface and rho = 1 at the PEC wall.
//!
//! Strips. The auxiliary arrays exist only in strips of W = N + 1 cells on
//! each side (left/right for the x strips, bottom/top for the y strips).
//! W = N + 1 covers both the Ez and the H positions with rho > 0.
//!   x strips: psi index s = j * 2W + c, c in 0..2W, i = c (c < W) or nx - 2W + c
//!   y strips: psi index s = r * nx + i, r in 0..2W, j = r (r < W) or ny - 2W + r
//! Corner cells lie in one x strip and one y strip; each strip has its own
//! arrays, so the corners need no special case.

/// CPML parameters in natural units (eps0 = mu0 = c = dx = 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pml {
    /// Thickness N in cells.
    pub cells: usize,
    /// Polynomial grading order of sigma and kappa.
    pub m: f64,
    pub sigma_max: f64,
    pub kappa_max: f64,
    pub alpha_max: f64,
    /// Grading order of alpha, which is alpha_max at the interface and 0 at the wall.
    pub m_alpha: f64,
}

impl Pml {
    /// Default profile. sigma_max = 0.8 (m + 1) / dx is the usual optimum for
    /// eta = 1. kappa_max = 1 and alpha_max = 0.05 come from the sweep in
    /// docs/verification.md: kappa_max > 1 made the grazing reflection worse.
    /// The sweep used lambda0 = 40 cells (omega = 0.157). Keep alpha_max below
    /// the lowest angular frequency of the run: at omega < alpha_max the CFS
    /// term removes most of the absorption (Stage 4 uses 0.01 for lambda0 <= 200).
    pub fn new(cells: usize) -> Self {
        let m = 3.0;
        Self { cells, m, sigma_max: 0.8 * (m + 1.0), kappa_max: 1.0, alpha_max: 0.05, m_alpha: 1.0 }
    }
}

/// 1D CPML coefficients along one axis at one staggering (integer or half).
/// Outside the PML: ik = 1, a = 0, b = 0.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    /// 1 / kappa
    pub ik: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct CpmlLayout {
    pub nx: usize,
    pub ny: usize,
    pub pml: Pml,
    /// Strip width W = N + 1.
    pub w: usize,
    /// Ez (and the Ez update of dHy/dx) at integer x.
    pub x_int: Profile,
    /// Hy at x = i + 1/2.
    pub x_half: Profile,
    /// Ez (and the Ez update of dHx/dy) at integer y.
    pub y_int: Profile,
    /// Hx at y = j + 1/2.
    pub y_half: Profile,
}

fn profile(n: usize, pml: &Pml, dt: f64, half: bool) -> Profile {
    let mut p = Profile { ik: vec![1.0; n], a: vec![0.0; n], b: vec![0.0; n] };
    if pml.cells == 0 {
        return p;
    }
    let big_n = pml.cells as f64;
    let far = (n - 1) as f64 - big_n;
    for i in 0..n {
        let pos = i as f64 + if half { 0.5 } else { 0.0 };
        let rho = ((big_n - pos) / big_n).max((pos - far) / big_n).max(0.0);
        if rho <= 0.0 {
            continue;
        }
        let g = rho.powf(pml.m);
        let sigma = pml.sigma_max * g;
        let kappa = 1.0 + (pml.kappa_max - 1.0) * g;
        let alpha = pml.alpha_max * (1.0 - rho).powf(pml.m_alpha);
        let b = (-(sigma / kappa + alpha) * dt).exp();
        let den = sigma * kappa + kappa * kappa * alpha;
        p.ik[i] = 1.0 / kappa;
        p.b[i] = b;
        p.a[i] = if den > 0.0 { sigma / den * (b - 1.0) } else { 0.0 };
    }
    p
}

impl CpmlLayout {
    pub fn new(nx: usize, ny: usize, pml: Pml, courant: f64) -> Result<Self, String> {
        let w = pml.cells + 1;
        if nx < 2 * w + 1 || ny < 2 * w + 1 {
            return Err(format!("grid {nx} x {ny} too small for a {}-cell PML", pml.cells));
        }
        if !(pml.kappa_max >= 1.0 && pml.sigma_max >= 0.0 && pml.alpha_max >= 0.0) {
            return Err(format!("bad PML parameters {pml:?}"));
        }
        Ok(Self {
            nx,
            ny,
            pml,
            w,
            x_int: profile(nx, &pml, courant, false),
            x_half: profile(nx, &pml, courant, true),
            y_int: profile(ny, &pml, courant, false),
            y_half: profile(ny, &pml, courant, true),
        })
    }

    /// x index of x-strip column c.
    #[inline(always)]
    pub fn strip_i(&self, c: usize) -> usize {
        if c < self.w { c } else { self.nx - 2 * self.w + c }
    }

    /// y index of y-strip row r.
    #[inline(always)]
    pub fn strip_j(&self, r: usize) -> usize {
        if r < self.w { r } else { self.ny - 2 * self.w + r }
    }

    /// y-strip row of y index j, if j lies in a y strip.
    #[inline(always)]
    pub fn strip_row(&self, j: usize) -> Option<usize> {
        if j < self.w {
            Some(j)
        } else if j >= self.ny - self.w {
            Some(j + 2 * self.w - self.ny)
        } else {
            None
        }
    }

    /// Length of one x-strip psi array (ny rows of 2W).
    pub fn x_strip_len(&self) -> usize {
        self.ny * 2 * self.w
    }

    /// Length of one y-strip psi array (2W rows of nx).
    pub fn y_strip_len(&self) -> usize {
        2 * self.w * self.nx
    }
}
