//! Design region for the 90-degree bend (Stage 5): parameterization, filter,
//! projection, fabrication metrics and reference shapes.
//!
//! The box is n x n cells at the corner of the `PortSetup` bend. The
//! horizontal slab enters through the left side and the vertical slab leaves
//! through the top side, both `margin` cells from the box corner, so the port
//! cross-sections stay outside the box. Cell (a, b) of the box is grid cell
//! (x0 + a, y0 + b); arrays are row-major with index b * n + a.
//!
//! Chain: latent rho in [0, 1] -> conic filter (radius r_f) -> tanh projection
//! (beta, eta) -> eps = eps_clad + rho_bar (eps_core - eps_clad). The filter
//! halo outside the box uses the fixed exterior geometry (1 in the two slabs,
//! 0 in the cladding), so the slabs continue into the box without a step.

use crate::sparams::PortSetup;
use crate::verify::{EPS_AIR, EPS_SI};

#[derive(Clone, Debug)]
pub struct DesignBox {
    pub x0: usize,
    pub y0: usize,
    pub n: usize,
    /// Distance from the box bottom to the horizontal slab, and from the box
    /// right side to the vertical slab (cells).
    pub margin: usize,
    pub core_w: usize,
}

impl DesignBox {
    /// Box of n x n cells placed so that its sharp-L fill is exactly the
    /// Stage 4 bend: vertical slab at the setup's corner column, horizontal
    /// slab at the setup's slab rows.
    pub fn for_bend(setup: &PortSetup, n: usize, margin: usize) -> Self {
        let w = setup.core_w;
        assert!(n >= 2 * margin + 2 * w, "box too small");
        Self { x0: setup.corner_x() + w + margin - n, y0: setup.j0() - margin, n, margin, core_w: w }
    }

    pub fn contains(&self, i: usize, j: usize) -> bool {
        (self.x0..self.x0 + self.n).contains(&i) && (self.y0..self.y0 + self.n).contains(&j)
    }

    /// Fixed geometry outside the box (1 = core, 0 = cladding), in box coordinates
    /// (may be negative or >= n). The slabs extend to infinity away from the box.
    pub fn exterior(&self, a: i64, b: i64) -> f64 {
        let (n, m, w) = (self.n as i64, self.margin as i64, self.core_w as i64);
        let horizontal = a < 0 && (m..m + w).contains(&b);
        let vertical = b >= n && (n - m - w..n - m).contains(&a);
        if horizontal || vertical { 1.0 } else { 0.0 }
    }

    /// Sharp L: the Stage 4 corner.
    pub fn sharp_l(&self) -> Vec<f64> {
        let (n, m, w) = (self.n, self.margin, self.core_w);
        let mut r = vec![0.0; n * n];
        for b in 0..n {
            for a in 0..n {
                let horizontal = (m..m + w).contains(&b) && a < n - m;
                let vertical = (n - m - w..n - m).contains(&a) && b >= m;
                if horizontal || vertical {
                    r[b * n + a] = 1.0;
                }
            }
        }
        r
    }

    /// Circular arc of width core_w whose centreline has radius
    /// R = n - margin - core_w / 2, centred at the top-left box corner. A cell
    /// is core if its centre lies in the annulus.
    pub fn circular_arc(&self) -> Vec<f64> {
        let (n, w) = (self.n, self.core_w as f64);
        let r = self.arc_radius() as f64;
        // Centre of curvature: the top-left box corner (0, n). The arc is tangent
        // to the horizontal slab at the left side and to the vertical slab at the top.
        let mut out = vec![0.0; n * n];
        for b in 0..n {
            for a in 0..n {
                let (x, y) = (a as f64 + 0.5, b as f64 + 0.5 - n as f64);
                if (x.hypot(y) - r).abs() <= w / 2.0 {
                    out[b * n + a] = 1.0;
                }
            }
        }
        out
    }

    /// Centreline radius of the circular arc.
    pub fn arc_radius(&self) -> usize {
        self.n - self.margin - self.core_w / 2
    }

    /// eps of a box cell for the projected density.
    pub fn eps(rho_bar: f64) -> f64 {
        EPS_AIR + rho_bar * (EPS_SI - EPS_AIR)
    }
}

/// Conic density filter with radius `r` (cells): weight max(0, r - distance),
/// each output normalised by its weight sum. Cells outside the box take the
/// fixed exterior value, so the filter is affine: x_f = W x + c.
#[derive(Clone, Debug)]
pub struct ConicFilter {
    pub r: f64,
    /// Kernel offsets and weights.
    taps: Vec<(i64, i64, f64)>,
    total: f64,
}

impl ConicFilter {
    pub fn new(r: f64) -> Self {
        let k = r.ceil() as i64;
        let mut taps = Vec::new();
        for db in -k..=k {
            for da in -k..=k {
                let w = r - ((da * da + db * db) as f64).sqrt();
                if w > 0.0 {
                    taps.push((da, db, w));
                }
            }
        }
        let total = taps.iter().map(|t| t.2).sum();
        Self { r, taps, total }
    }

    /// Filtered density on the box (exterior halo from `bx`).
    pub fn apply(&self, bx: &DesignBox, rho: &[f64]) -> Vec<f64> {
        let n = bx.n as i64;
        let mut out = vec![0.0; rho.len()];
        for b in 0..n {
            for a in 0..n {
                let mut s = 0.0;
                for &(da, db, w) in &self.taps {
                    let (p, q) = (a + da, b + db);
                    let v = if (0..n).contains(&p) && (0..n).contains(&q) { rho[(q * n + p) as usize] } else { bx.exterior(p, q) };
                    s += w * v;
                }
                out[(b * n + a) as usize] = s / self.total;
            }
        }
        out
    }

    /// Transpose of the linear part: g_rho = W^T g_filtered.
    pub fn pullback(&self, bx: &DesignBox, g: &[f64]) -> Vec<f64> {
        let n = bx.n as i64;
        let mut out = vec![0.0; g.len()];
        for b in 0..n {
            for a in 0..n {
                let gv = g[(b * n + a) as usize] / self.total;
                for &(da, db, w) in &self.taps {
                    let (p, q) = (a + da, b + db);
                    if (0..n).contains(&p) && (0..n).contains(&q) {
                        out[(q * n + p) as usize] += w * gv;
                    }
                }
            }
        }
        out
    }
}

/// tanh projection with threshold eta and sharpness beta.
pub fn project(x: f64, beta: f64, eta: f64) -> f64 {
    let d = (beta * eta).tanh() + (beta * (1.0 - eta)).tanh();
    ((beta * eta).tanh() + (beta * (x - eta)).tanh()) / d
}

/// d project / dx.
pub fn project_grad(x: f64, beta: f64, eta: f64) -> f64 {
    let d = (beta * eta).tanh() + (beta * (1.0 - eta)).tanh();
    let t = (beta * (x - eta)).tanh();
    beta * (1.0 - t * t) / d
}

/// Full chain: rho -> filtered -> projected.
pub fn densities(bx: &DesignBox, f: &ConicFilter, rho: &[f64], beta: f64) -> Vec<f64> {
    f.apply(bx, rho).into_iter().map(|x| project(x, beta, 0.5)).collect()
}

/// Pull a gradient with respect to the projected density back to rho.
pub fn pullback(bx: &DesignBox, f: &ConicFilter, rho: &[f64], beta: f64, g_bar: &[f64]) -> Vec<f64> {
    let xf = f.apply(bx, rho);
    let g: Vec<f64> = xf.iter().zip(g_bar).map(|(&x, &g)| g * project_grad(x, beta, 0.5)).collect();
    f.pullback(bx, &g)
}

/// Fraction of cells with rho_bar < 0.01 or > 0.99.
pub fn binary_fraction(rho_bar: &[f64]) -> f64 {
    rho_bar.iter().filter(|&&v| !(0.01..=0.99).contains(&v)).count() as f64 / rho_bar.len() as f64
}

/// Sizes of the 4-connected components of solid (>= 0.5) and of void cells in
/// the box, with the fixed exterior attached: a component that touches the
/// exterior of the same phase is reported with size usize::MAX (anchored).
pub fn components(bx: &DesignBox, rho_bar: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = bx.n;
    let solid: Vec<bool> = rho_bar.iter().map(|&v| v >= 0.5).collect();
    let mut seen = vec![false; n * n];
    let (mut s_sizes, mut v_sizes) = (Vec::new(), Vec::new());
    for start in 0..n * n {
        if seen[start] {
            continue;
        }
        let phase = solid[start];
        let (mut stack, mut size, mut anchored) = (vec![start], 0usize, false);
        seen[start] = true;
        while let Some(k) = stack.pop() {
            size += 1;
            let (a, b) = ((k % n) as i64, (k / n) as i64);
            for (p, q) in [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)] {
                if (0..n as i64).contains(&p) && (0..n as i64).contains(&q) {
                    let kk = (q as usize) * n + p as usize;
                    if !seen[kk] && solid[kk] == phase {
                        seen[kk] = true;
                        stack.push(kk);
                    }
                } else if (bx.exterior(p, q) >= 0.5) == phase {
                    anchored = true;
                }
            }
        }
        let s = if anchored { usize::MAX } else { size };
        if phase { s_sizes.push(s) } else { v_sizes.push(s) }
    }
    (s_sizes, v_sizes)
}

/// Number of isolated 1-cell islands (solid or void) that are not attached to the exterior.
pub fn single_cell_islands(bx: &DesignBox, rho_bar: &[f64]) -> usize {
    let (s, v) = components(bx, rho_bar);
    s.iter().chain(&v).filter(|&&k| k == 1).count()
}
