//! Stage 6 robust evaluation (docs/stage6-protocol.md). Every Stage 6
//! optimiser and baseline uses this module, so they share one evaluator.
//!
//! Bias rule (frozen). A biased mask is the nominal box mask eroded or dilated
//! with the L1 ball of radius `scale` cells (the 4-neighbour cross at scale 1,
//! the same +-1 native cell as the Stage 5 check). Only box cells change. The
//! fixed exterior (the two slabs and the cladding) is the padding and is not
//! biased, so the ports and the S-parameter reference stay the same.
//!
//! Geometry rules. L_min = 5 native cells for silicon width and air gap: an
//! opening (width) and a closing (gap) with a disk of diameter 5 * scale cells
//! must change no box cell. Connectivity: 4-connectivity; the input slab must
//! connect to the output slab through the box; no silicon component may float
//! (touch neither slab); enclosed air holes are allowed and counted.

use std::collections::HashMap;
use std::error::Error;

use crate::adjoint::{Objective, forward, gradient, gradient_r};
use crate::design::{ConicFilter, DesignBox, project, project_grad, single_cell_islands};
use crate::sparams::{PortData, PortSetup, Run, db};
use crate::verify::{EPS_AIR, EPS_SI};

pub const IL_MAX_DB: f64 = 0.50;
pub const RL_MIN_DB: f64 = 22.0;
pub const L_MIN_NATIVE: usize = 5;

/// The L1 ball of radius r (the cross for r = 1).
pub fn l1_ball(r: i64) -> Vec<(i64, i64)> {
    (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| (dx, dy))).filter(|&(dx, dy)| dx.abs() + dy.abs() <= r).collect()
}

/// A disk that spans d cells. Odd d: the cell centres within (d - 1) / 2 of
/// a cell centre (d = 5 is the Stage 5 L_min disk, `design::disk(4)`).
/// Even d: the cell centres within sqrt(((d - 1) / 2)^2 + 1/4) of a cell
/// corner, which includes the cells at (+-(d - 1) / 2, +-1/2) (d = 2: the
/// 2 x 2 block).
pub fn disk_d(d: usize) -> Vec<(i64, i64)> {
    let h = (d as f64 - 1.0) / 2.0;
    let (c, r2) = if d % 2 == 1 { (0.0, h * h) } else { (0.5, h * h + 0.25) };
    let m = d as i64;
    (-m..=m)
        .flat_map(|dy| (-m..=m).map(move |dx| (dx, dy)))
        .filter(|&(dx, dy)| (dx as f64 + c).powi(2) + (dy as f64 + c).powi(2) <= r2 + 1e-9)
        .collect()
}

/// Canvas: the box with a halo of the fixed exterior.
struct Canvas {
    n: i64,
    pad: i64,
    m: i64,
    on: Vec<bool>,
}

impl Canvas {
    fn new(bx: &DesignBox, s: &[f64], pad: i64) -> Self {
        let n = bx.n as i64;
        let m = n + 2 * pad;
        let on = (0..m * m)
            .map(|k| {
                let (a, b) = (k % m - pad, k / m - pad);
                if (0..n).contains(&a) && (0..n).contains(&b) { s[(b * n + a) as usize] >= 0.5 } else { bx.exterior(a, b) >= 0.5 }
            })
            .collect();
        Canvas { n, pad, m, on }
    }
    fn box_cell(&self, k: i64) -> usize {
        (((k / self.n) + self.pad) * self.m + k % self.n + self.pad) as usize
    }
}

/// Binary morphology on a canvas. Outside the canvas, the exterior rule holds.
fn op(bx: &DesignBox, c: &Canvas, src: &[bool], e: &[(i64, i64)], dilate: bool) -> Vec<bool> {
    let m = c.m;
    (0..m * m)
        .map(|k| {
            let (a, b) = (k % m, k / m);
            let get = |p: i64, q: i64| {
                if (0..m).contains(&p) && (0..m).contains(&q) { src[(q * m + p) as usize] } else { bx.exterior(p - c.pad, q - c.pad) >= 0.5 }
            };
            if dilate { e.iter().any(|&(dx, dy)| get(a + dx, b + dy)) } else { e.iter().all(|&(dx, dy)| get(a + dx, b + dy)) }
        })
        .collect()
}

/// Box cells changed by an opening (thin silicon) and by a closing (narrow
/// air) with element `e`. The second step uses the reflected element, so the
/// test is exact for elements that are not symmetric. The slabs are on the
/// canvas, so they are opened and closed with the box.
pub fn width_gap_violations(bx: &DesignBox, s: &[f64], e: &[(i64, i64)]) -> (usize, usize) {
    let r = e.iter().map(|&(x, y)| x.abs().max(y.abs())).max().unwrap_or(0);
    let c = Canvas::new(bx, s, 2 * r + 2);
    let refl: Vec<(i64, i64)> = e.iter().map(|&(x, y)| (-x, -y)).collect();
    let open = op(bx, &c, &op(bx, &c, &c.on, e, false), &refl, true);
    let close = op(bx, &c, &op(bx, &c, &c.on, e, true), &refl, false);
    let diff = |t: &[bool]| (0..c.n * c.n).filter(|&k| c.on[c.box_cell(k)] != t[c.box_cell(k)]).count();
    (diff(&open), diff(&close))
}

/// Biased box mask: erode (dilate = false) or dilate with the L1 ball of
/// radius `scale`. Only box cells change; the exterior is the fixed padding.
pub fn bias(bx: &DesignBox, s: &[f64], scale: usize, dilate: bool) -> Vec<f64> {
    let e = l1_ball(scale as i64);
    let n = bx.n as i64;
    let at = |p: i64, q: i64| if (0..n).contains(&p) && (0..n).contains(&q) { s[(q * n + p) as usize] >= 0.5 } else { bx.exterior(p, q) >= 0.5 };
    (0..n * n)
        .map(|k| {
            let (a, b) = (k % n, k / n);
            let v = if dilate { e.iter().any(|&(dx, dy)| at(a + dx, b + dy)) } else { e.iter().all(|&(dx, dy)| at(a + dx, b + dy)) };
            if v { 1.0 } else { 0.0 }
        })
        .collect()
}

/// [eroded, nominal, dilated].
pub fn variants(bx: &DesignBox, nominal: &[f64], scale: usize) -> [Vec<f64>; 3] {
    [bias(bx, nominal, scale, false), nominal.to_vec(), bias(bx, nominal, scale, true)]
}

#[derive(Clone, Debug, Default)]
pub struct Geometry {
    pub silicon_cells: usize,
    pub ports_connected: bool,
    pub floating_silicon: usize,
    pub enclosed_air: usize,
    pub single_cell_islands: usize,
    /// L_min violations (opening, closing) with the 5-native-cell disk.
    pub width_violations: usize,
    pub gap_violations: usize,
    /// Largest disk diameter (cells of this grid) with no opening change,
    /// and with no closing change; tested 1 ..= 12 * scale.
    pub min_width: usize,
    pub min_gap: usize,
}

impl Geometry {
    pub fn feasible(&self) -> bool {
        self.ports_connected && self.floating_silicon == 0 && self.single_cell_islands == 0 && self.width_violations == 0 && self.gap_violations == 0
    }
}

/// Connectivity counts: (ports connected, floating silicon components, enclosed air components).
pub fn connectivity(bx: &DesignBox, s: &[f64]) -> (bool, usize, usize) {
    let n = bx.n;
    let (m, w) = (bx.margin, bx.core_w);
    let solid: Vec<bool> = s.iter().map(|&v| v >= 0.5).collect();
    let mut label = vec![usize::MAX; n * n];
    let mut comps: Vec<(bool, bool, bool)> = Vec::new(); // (phase, touches input/exterior of same phase, touches output)
    for start in 0..n * n {
        if label[start] != usize::MAX {
            continue;
        }
        let id = comps.len();
        let phase = solid[start];
        let (mut t_in, mut t_out) = (false, false);
        let mut stack = vec![start];
        label[start] = id;
        while let Some(k) = stack.pop() {
            let (a, b) = ((k % n) as i64, (k / n) as i64);
            for (p, q) in [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)] {
                if (0..n as i64).contains(&p) && (0..n as i64).contains(&q) {
                    let kk = q as usize * n + p as usize;
                    if label[kk] == usize::MAX && solid[kk] == phase {
                        label[kk] = id;
                        stack.push(kk);
                    }
                } else if (bx.exterior(p, q) >= 0.5) == phase {
                    if phase {
                        // Input slab: left side, rows m..m+w. Output slab: top side.
                        if p < 0 && (m as i64..(m + w) as i64).contains(&q) {
                            t_in = true;
                        } else {
                            t_out = true;
                        }
                    } else {
                        t_in = true; // air: touches the cladding
                    }
                }
            }
        }
        comps.push((phase, t_in, t_out));
    }
    let connected = comps.iter().any(|&(ph, i, o)| ph && i && o);
    let floating = comps.iter().filter(|&&(ph, i, o)| ph && !i && !o).count();
    let enclosed = comps.iter().filter(|&&(ph, i, _)| !ph && !i).count();
    (connected, floating, enclosed)
}

pub fn geometry(bx: &DesignBox, s: &[f64], scale: usize) -> Geometry {
    let (ports_connected, floating_silicon, enclosed_air) = connectivity(bx, s);
    let (width_violations, gap_violations) = width_gap_violations(bx, s, &disk_d(L_MIN_NATIVE * scale));
    let largest = |f: &dyn Fn((usize, usize)) -> usize| (1..=12 * scale).take_while(|&d| f(width_gap_violations(bx, s, &disk_d(d))) == 0).last().unwrap_or(0);
    Geometry {
        silicon_cells: s.iter().filter(|&&v| v >= 0.5).count(),
        ports_connected,
        floating_silicon,
        enclosed_air,
        single_cell_islands: single_cell_islands(bx, s),
        width_violations,
        gap_violations,
        min_width: largest(&|(o, _)| o),
        min_gap: largest(&|(_, c)| c),
    }
}

/// Modal spectra of one binary mask.
#[derive(Clone, Debug)]
pub struct Spectrum {
    pub lambda0: Vec<f64>,
    pub il_db: Vec<f64>,
    pub rl_db: Vec<f64>,
}

impl Spectrum {
    pub fn worst_il(&self) -> f64 {
        self.il_db.iter().cloned().fold(0.0, f64::max)
    }
    pub fn worst_rl(&self) -> f64 {
        self.rl_db.iter().cloned().fold(f64::INFINITY, f64::min)
    }
}

pub fn spectrum(setup: &PortSetup, bx: &DesignBox, reference: &PortData, s: &[f64]) -> Result<Spectrum, Box<dyn Error>> {
    let st = setup.clone().with_design(bx, s);
    let sp = st.sparams(reference, &st.run(Run::Dut)?);
    Ok(Spectrum {
        lambda0: sp.freqs.iter().map(|f| 1.0 / f).collect(),
        il_db: sp.s21_mode.iter().map(|v| -db(*v)).collect(),
        rl_db: sp.s11_mode.iter().map(|v| -db(*v)).collect(),
    })
}

/// Robust binary score J = max_v worst IL + max(0, 22 - min_v worst RL), in dB.
pub fn robust_j(sp: &[Spectrum; 3]) -> f64 {
    let il = sp.iter().map(|s| s.worst_il()).fold(0.0, f64::max);
    let rl = sp.iter().map(|s| s.worst_rl()).fold(f64::INFINITY, f64::min);
    il + (RL_MIN_DB - rl).max(0.0)
}

/// Cached robust scorer of binary nominal masks. Infeasible nominal masks
/// (geometry gates) get J = 100 + violations and no FDTD run. A cache hit
/// costs no run. Counts the forward solves it runs.
pub struct Scorer {
    pub base: PortSetup,
    pub bx: DesignBox,
    pub reference: PortData,
    pub cache: HashMap<Vec<u8>, (f64, Option<[Spectrum; 3]>)>,
    pub forward_solves: usize,
    pub cache_hits: usize,
    pub forward_seconds: f64,
}

pub struct Scored {
    pub j: f64,
    pub feasible: bool,
    pub solves: usize,
    pub cached: bool,
    pub spectra: Option<[Spectrum; 3]>,
}

impl Scorer {
    pub fn new(base: PortSetup, bx: DesignBox, reference: PortData) -> Self {
        Scorer { base, bx, reference, cache: HashMap::new(), forward_solves: 0, cache_hits: 0, forward_seconds: 0.0 }
    }

    pub fn score(&mut self, nominal: &[f64]) -> Result<Scored, Box<dyn Error>> {
        let key: Vec<u8> = nominal.iter().map(|&v| (v >= 0.5) as u8).collect();
        if let Some((j, sp)) = self.cache.get(&key) {
            self.cache_hits += 1;
            return Ok(Scored { j: *j, feasible: sp.is_some(), solves: 0, cached: true, spectra: sp.clone() });
        }
        let g = geometry(&self.bx, nominal, 1);
        if !g.feasible() {
            let v = g.width_violations + g.gap_violations + g.single_cell_islands + g.floating_silicon + (!g.ports_connected) as usize * 100;
            let j = 100.0 + v as f64;
            self.cache.insert(key, (j, None));
            return Ok(Scored { j, feasible: false, solves: 0, cached: false, spectra: None });
        }
        let t0 = std::time::Instant::now();
        let [e, n, d] = variants(&self.bx, nominal, 1);
        let sp = [spectrum(&self.base, &self.bx, &self.reference, &e)?, spectrum(&self.base, &self.bx, &self.reference, &n)?, spectrum(&self.base, &self.bx, &self.reference, &d)?];
        self.forward_seconds += t0.elapsed().as_secs_f64();
        self.forward_solves += 3;
        let j = robust_j(&sp);
        self.cache.insert(key, (j, Some(sp.clone())));
        Ok(Scored { j, feasible: true, solves: 3, cached: false, spectra: Some(sp) })
    }
}

/// Design representation shared by the Stage 6 benchmark (`Grid(20)` on the
/// 80 x 80 box) and the full-resolution practical run (`Full`).
#[derive(Clone, Copy, Debug)]
pub enum Rep {
    Full,
    /// nc x nc control values; bilinear interpolation, controls at block centres.
    Grid(usize),
}

impl Rep {
    pub fn dim(&self, n: usize) -> usize {
        match self {
            Rep::Full => n * n,
            Rep::Grid(nc) => nc * nc,
        }
    }

    fn weights(nc: usize, n: usize, k: usize) -> [(usize, f64); 4] {
        let s = n as f64 / nc as f64;
        let u = (((k % n) as f64 + 0.5) / s - 0.5).clamp(0.0, (nc - 1) as f64);
        let v = (((k / n) as f64 + 0.5) / s - 0.5).clamp(0.0, (nc - 1) as f64);
        let (i, j) = ((u.floor() as usize).min(nc - 2), (v.floor() as usize).min(nc - 2));
        let (fu, fv) = (u - i as f64, v - j as f64);
        [(j * nc + i, (1.0 - fu) * (1.0 - fv)), (j * nc + i + 1, fu * (1.0 - fv)), ((j + 1) * nc + i, (1.0 - fu) * fv), ((j + 1) * nc + i + 1, fu * fv)]
    }

    /// Latent density on the n x n box. Values are clipped to [0, 1] first.
    pub fn latent(&self, x: &[f64], n: usize) -> Vec<f64> {
        match *self {
            Rep::Full => x.iter().map(|v| v.clamp(0.0, 1.0)).collect(),
            Rep::Grid(nc) => (0..n * n).map(|k| Self::weights(nc, n, k).iter().map(|&(c, w)| w * x[c].clamp(0.0, 1.0)).sum()).collect(),
        }
    }

    /// Transpose of `latent` (for x inside [0, 1]).
    pub fn pullback(&self, g: &[f64], n: usize) -> Vec<f64> {
        match *self {
            Rep::Full => g.to_vec(),
            Rep::Grid(nc) => {
                let mut out = vec![0.0; nc * nc];
                for (k, gk) in g.iter().enumerate() {
                    for (c, w) in Self::weights(nc, n, k) {
                        out[c] += w * gk;
                    }
                }
                out
            }
        }
    }

    /// Representation of a box mask: the mask itself (Full) or its block means (Grid).
    pub fn encode(&self, mask: &[f64], n: usize) -> Vec<f64> {
        match *self {
            Rep::Full => mask.to_vec(),
            Rep::Grid(nc) => {
                let s = n / nc;
                (0..nc * nc)
                    .map(|k| {
                        let (i, j) = (k % nc, k / nc);
                        (0..s * s).map(|q| mask[(j * s + q / s) * n + i * s + q % s]).sum::<f64>() / (s * s) as f64
                    })
                    .collect()
            }
        }
    }
}

/// Binary nominal mask of a representation value: 0.5 level set of the filtered latent.
pub fn nominal_mask(rep: Rep, filt: &ConicFilter, bx: &DesignBox, x: &[f64]) -> Vec<f64> {
    filt.apply(bx, &rep.latent(x, bx.n)).iter().map(|&v| if v >= 0.5 { 1.0 } else { 0.0 }).collect()
}

/// Threshold eta such that the filtered field thresholded at eta has the
/// same box silicon area as `target_area` (bisection; area falls with eta).
pub fn eta_for_area(filtered: &[f64], target_area: usize) -> f64 {
    let area = |eta: f64| filtered.iter().filter(|&&v| v >= eta).count();
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if area(mid) > target_area { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// Relaxed robust objective of the adjoint method.
/// Variants: projection thresholds [eta_e, 0.5, eta_d]. For variant v, the
/// loss terms are (1 - T_k) and c R_k (c = 0.1 / R_max, R_max = 10^(-2.2)),
/// k over the 15 samples; F_v is their p-norm (p = 8). F = max_v F_v.
pub struct Relaxed {
    pub eta: [f64; 3],
    pub beta: f64,
    pub p: f64,
}

pub struct RelaxedEval {
    pub f: f64,
    pub f_v: [f64; 3],
    pub worst: usize,
    /// Gradient with respect to the representation.
    pub grad: Vec<f64>,
    pub forward_solves: usize,
    pub adjoint_solves: usize,
    pub forward_seconds: f64,
    pub adjoint_seconds: f64,
}

pub fn reflection_weight() -> f64 {
    0.1 / 10f64.powf(-RL_MIN_DB / 10.0)
}

fn pnorm(l: &[f64], p: f64) -> (f64, Vec<f64>) {
    let n = l.len() as f64;
    let mean: f64 = l.iter().map(|&x| x.max(1e-300).powf(p)).sum::<f64>() / n;
    let f = mean.powf(1.0 / p);
    (f, l.iter().map(|&x| mean.powf(1.0 / p - 1.0) * x.max(1e-300).powf(p - 1.0) / n).collect())
}

/// Evaluate F and dF/dx: 3 forward solves (one per variant) and the T and R
/// adjoint solves of the worst variant only (a subgradient of the max).
pub fn relaxed(
    rx: &Relaxed,
    rep: Rep,
    filt: &ConicFilter,
    base: &PortSetup,
    bx: &DesignBox,
    obj: &Objective,
    x: &[f64],
) -> Result<RelaxedEval, Box<dyn Error>> {
    let n = bx.n;
    let lat = rep.latent(x, n);
    let xf = filt.apply(bx, &lat);
    let c = reflection_weight();
    let mut fwds = Vec::new();
    let mut fv = [0.0; 3];
    let t0 = std::time::Instant::now();
    for v in 0..3 {
        let rb: Vec<f64> = xf.iter().map(|&x| project(x, rx.beta, rx.eta[v])).collect();
        let st = base.clone().with_design(bx, &rb);
        let fw = forward(&st, obj, bx)?;
        let l: Vec<f64> = fw.t.iter().map(|t| 1.0 - t).chain(fw.r.iter().map(|r| c * r)).collect();
        fv[v] = pnorm(&l, rx.p).0;
        fwds.push((st, fw));
    }
    let forward_seconds = t0.elapsed().as_secs_f64();
    let worst = (0..3).max_by(|&a, &b| fv[a].total_cmp(&fv[b])).unwrap();
    let (st, fw) = &fwds[worst];
    let t1 = std::time::Instant::now();
    let gt = gradient(st, obj, bx, fw)?;
    let gr = gradient_r(st, obj, bx, fw)?;
    let adjoint_seconds = t1.elapsed().as_secs_f64();
    let nf = fw.t.len();
    let l: Vec<f64> = fw.t.iter().map(|t| 1.0 - t).chain(fw.r.iter().map(|r| c * r)).collect();
    let (_, dl) = pnorm(&l, rx.p);
    let mut g_bar = vec![0.0; n * n];
    for k in 0..nf {
        let (wt, wr) = (-dl[k], c * dl[nf + k]);
        for (q, gb) in g_bar.iter_mut().enumerate() {
            *gb += (wt * gt[k][q] + wr * gr[k][q]) * (EPS_SI - EPS_AIR);
        }
    }
    let g_f: Vec<f64> = xf.iter().zip(&g_bar).map(|(&x, &g)| g * project_grad(x, rx.beta, rx.eta[worst])).collect();
    let g_lat = filt.pullback(bx, &g_f);
    let grad = rep.pullback(&g_lat, n);
    Ok(RelaxedEval {
        f: fv[worst],
        f_v: fv,
        worst,
        grad,
        forward_solves: 3,
        adjoint_solves: obj.basis.len() + obj.p1.as_ref().map(|p| p.basis.len()).unwrap_or(0),
        forward_seconds,
        adjoint_seconds,
    })
}
