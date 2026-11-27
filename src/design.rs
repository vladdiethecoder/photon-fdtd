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

/// A centreline in box coordinates (continuous, cell (a, b) spans [a, a+1) x [b, b+1)).
pub type Polyline = Vec<(f64, f64)>;

impl DesignBox {
    /// Start and end of the bend centreline and the per-axis distance L.
    fn bend_ends(&self) -> ((f64, f64), (f64, f64), f64) {
        let (n, m, w) = (self.n as f64, self.margin as f64, self.core_w as f64);
        let p0 = (0.0, m + w / 2.0);
        let p3 = (n - m - w / 2.0, n);
        (p0, p3, p3.0 - p0.0)
    }

    /// Euler bend: curvature ramps up linearly, stays constant, ramps down;
    /// `p` is the fraction of the 90-degree angle in the two ramps (p = 0: arc).
    pub fn euler_centreline(&self, p: f64) -> Polyline {
        let (p0, _, l) = self.bend_ends();
        let (s1, s2) = (p * std::f64::consts::FRAC_PI_2, (1.0 - p) * std::f64::consts::FRAC_PI_2);
        let total = 2.0 * s1 + s2;
        let kappa = |s: f64| {
            if s < s1 {
                s / s1
            } else if s < s1 + s2 {
                1.0
            } else {
                (total - s) / s1
            }
        };
        let steps = 20000;
        let ds = total / steps as f64;
        let (mut x, mut y, mut th) = (0.0, 0.0, 0.0);
        let mut pts = vec![(0.0, 0.0)];
        for k in 0..steps {
            let s = (k as f64 + 0.5) * ds;
            let thm = th + 0.5 * kappa(s) * ds;
            x += thm.cos() * ds;
            y += thm.sin() * ds;
            th += kappa(s) * ds;
            pts.push((x, y));
        }
        let scale = l / x;
        pts.into_iter().map(|(x, y)| (p0.0 + scale * x, p0.1 + scale * y)).collect()
    }

    /// Cubic Bezier bend with handle length k L along the end directions.
    pub fn bezier_centreline(&self, k: f64) -> Polyline {
        let (p0, p3, l) = self.bend_ends();
        let p1 = (p0.0 + k * l, p0.1);
        let p2 = (p3.0, p3.1 - k * l);
        (0..=4000)
            .map(|q| {
                let t = q as f64 / 4000.0;
                let u = 1.0 - t;
                let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                (a * p0.0 + b * p1.0 + c * p2.0 + d * p3.0, a * p0.1 + b * p1.1 + c * p2.1 + d * p3.1)
            })
            .collect()
    }

    /// Rasterise a tube of width core_w around a centreline. The centreline is
    /// extended straight beyond both ends, so the slabs continue across the box edges.
    pub fn rasterise(&self, line: &Polyline) -> Vec<f64> {
        let (n, half) = (self.n, self.core_w as f64 / 2.0);
        let (first, last) = (line[0], line[line.len() - 1]);
        let mut pts = vec![(first.0 - 2.0 * n as f64, first.1)];
        pts.extend_from_slice(line);
        pts.push((last.0, last.1 + 2.0 * n as f64));
        let mut out = vec![0.0; n * n];
        for b in 0..n {
            for a in 0..n {
                let (x, y) = (a as f64 + 0.5, b as f64 + 0.5);
                let mut d2 = f64::INFINITY;
                for s in pts.windows(2) {
                    let ((x0, y0), (x1, y1)) = (s[0], s[1]);
                    let (dx, dy) = (x1 - x0, y1 - y0);
                    let t = (((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
                    let (px, py) = (x0 + t * dx - x, y0 + t * dy - y);
                    d2 = d2.min(px * px + py * py);
                }
                if d2 <= half * half {
                    out[b * n + a] = 1.0;
                }
            }
        }
        out
    }
}

/// Binary morphology on the box with the fixed exterior as padding.
/// `offsets`: the structuring element. `dilate = true`: max, otherwise min.
pub fn morph(bx: &DesignBox, s: &[f64], offsets: &[(i64, i64)], dilate: bool) -> Vec<f64> {
    let n = bx.n as i64;
    let at = |p: i64, q: i64| if (0..n).contains(&p) && (0..n).contains(&q) { s[(q * n + p) as usize] >= 0.5 } else { bx.exterior(p, q) >= 0.5 };
    let mut out = vec![0.0; s.len()];
    for b in 0..n {
        for a in 0..n {
            let v = if dilate { offsets.iter().any(|&(dx, dy)| at(a + dx, b + dy)) } else { offsets.iter().all(|&(dx, dy)| at(a + dx, b + dy)) };
            out[(b * n + a) as usize] = if v { 1.0 } else { 0.0 };
        }
    }
    out
}

/// Disk offsets dx^2 + dy^2 <= r2 (r2 = 4: diameter 5 cells).
pub fn disk(r2: i64) -> Vec<(i64, i64)> {
    let r = (r2 as f64).sqrt().floor() as i64;
    (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| (dx, dy))).filter(|&(dx, dy)| dx * dx + dy * dy <= r2).collect()
}

/// 4-neighbour cross (1-cell erosion and dilation).
pub fn cross() -> Vec<(i64, i64)> {
    vec![(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
}

/// Minimum feature check: number of box cells changed by an opening (thin
/// silicon) and by a closing (narrow air gaps) with a symmetric `element`.
/// The operations run on a canvas that holds the box and a halo of the fixed
/// exterior, so the exterior slabs are eroded and dilated like the box.
pub fn feature_violations(bx: &DesignBox, s: &[f64], element: &[(i64, i64)]) -> (usize, usize) {
    let n = bx.n as i64;
    let r = element.iter().map(|&(x, y)| x.abs().max(y.abs())).max().unwrap_or(0);
    let pad = 2 * r + 2;
    let m = n + 2 * pad;
    let base: Vec<bool> = (0..m * m)
        .map(|k| {
            let (a, b) = (k % m - pad, k / m - pad);
            if (0..n).contains(&a) && (0..n).contains(&b) { s[(b * n + a) as usize] >= 0.5 } else { bx.exterior(a, b) >= 0.5 }
        })
        .collect();
    let op = |c: &[bool], dilate: bool| -> Vec<bool> {
        (0..m * m)
            .map(|k| {
                let (a, b) = (k % m, k / m);
                let get = |p: i64, q: i64| {
                    if (0..m).contains(&p) && (0..m).contains(&q) { c[(q * m + p) as usize] } else { bx.exterior(p - pad, q - pad) >= 0.5 }
                };
                if dilate { element.iter().any(|&(dx, dy)| get(a + dx, b + dy)) } else { element.iter().all(|&(dx, dy)| get(a + dx, b + dy)) }
            })
            .collect()
    };
    let open = op(&op(&base, false), true);
    let close = op(&op(&base, true), false);
    let diff = |t: &[bool]| (0..n * n).filter(|&k| base[((k / n + pad) * m + k % n + pad) as usize] != t[((k / n + pad) * m + k % n + pad) as usize]).count();
    (diff(&open), diff(&close))
}
