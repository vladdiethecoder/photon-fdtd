//! Adjoint gradient of the modal bend transmission (Stage 5).
//!
//! Frequency-domain sensitivity of the time-stepped scheme. With DFTs at the
//! sample times of the leapfrog update (Ez at (n + 1) dt), the steady-state
//! fields obey A(eps) E = b with A = K - eps Omega^2, Omega = (2 / dt)
//! sin(omega dt / 2). K holds the curls and the CPML and does not depend on
//! eps, so dA / deps_r = -Omega^2 at cell r, and for a linear functional
//! a = c^T E:
//!   da / deps_r = Omega^2 E_r (G^T c)_r,   G = A^{-1}.
//! Outside the CPML, G^T = G (reciprocity of the stretched-coordinate
//! operator, which is symmetric after scaling by s_x s_y = 1 there), so
//! G^T c is the field of an adjoint run with a source at the port-2 cells.
//!
//! Soft source normalisation: adding s^n to Ez after step n gives the
//! right-hand side b_q = i Omega (eps_q / dt) e^{i omega dt / 2} S_q(omega),
//! S_q = sum_n s_q^n e^{-i omega (n + 1) dt}. The adjoint run therefore injects
//! phi_q / eps_q with the pulse g(t), and divides its DFT by
//! N = i Omega e^{i omega dt / 2} G_hat(omega) / dt.
//!
//! Objective. a_k = <u_k, Ez(y2)> / <u_k, u_k> is the port-2 mode amplitude
//! from Ez on the port row (the backward wave at port 2 is the top-CPML
//! reflection, below -80 dB). T_k = |a_k|^2 / |a_inc,k|^2 with the incident
//! forward amplitude of the Stage 4 reference run. The 15 port-2 profiles u_k
//! are expanded in an orthonormal basis phi_m (`basis`), so the adjoint needs
//! one run per basis vector, not one per frequency.

use std::error::Error;
use std::f64::consts::PI;

use crate::design::DesignBox;
use crate::gaussian_sine;
use crate::gpu::{Field, GpuYee};
use crate::sparams::{PortData, PortSetup, Run, discrete_mode, mode_amplitudes};
use crate::verify::courant;

type C = (f64, f64);

fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn cdiv(a: C, b: C) -> C {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
fn cabs2(a: C) -> f64 {
    a.0 * a.0 + a.1 * a.1
}

/// Eigen-decomposition of a small symmetric matrix (cyclic Jacobi).
fn jacobi(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    for _ in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-30 * (0..n).map(|i| a[i][i] * a[i][i]).sum::<f64>() {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

/// Orthonormal basis of the profiles u_k (eigenvectors of the Gram matrix,
/// largest first, until the largest relative residual is below 1e-10), the
/// coefficients <u_k, phi_m> and that residual.
fn orthonormal_basis(u: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64) {
    let nf = u.len();
    let gram: Vec<Vec<f64>> =
        (0..nf).map(|i| (0..nf).map(|j| u[i].iter().zip(&u[j]).map(|(a, b)| a * b).sum()).collect()).collect();
    let (vals, vecs) = jacobi(gram);
    let mut order: Vec<usize> = (0..nf).collect();
    order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut residual = f64::INFINITY;
    for &m in &order {
        if residual < 1e-10 || vals[m] <= 0.0 {
            break;
        }
        let mut phi = vec![0.0; u[0].len()];
        for k in 0..nf {
            for (x, uk) in phi.iter_mut().zip(&u[k]) {
                *x += vecs[k][m] * uk;
            }
        }
        let norm = phi.iter().map(|x| x * x).sum::<f64>().sqrt();
        phi.iter_mut().for_each(|x| *x /= norm);
        basis.push(phi);
        residual = u
            .iter()
            .map(|uk| {
                let mut r = uk.clone();
                for b in &basis {
                    let c: f64 = uk.iter().zip(b).map(|(a, b)| a * b).sum();
                    r.iter_mut().zip(b).for_each(|(x, y)| *x -= c * y);
                }
                (r.iter().map(|x| x * x).sum::<f64>() / uk.iter().map(|x| x * x).sum::<f64>()).sqrt()
            })
            .fold(0.0, f64::max);
    }
    let coeff = u.iter().map(|uk| basis.iter().map(|b| uk.iter().zip(b).map(|(a, b)| a * b).sum()).collect()).collect();
    (basis, coeff, residual)
}

/// The fixed parts of the bend objective: ports, profiles, adjoint basis.
pub struct Objective {
    pub freqs: Vec<f64>,
    /// Incident forward amplitude at port 1 (reference run).
    pub a_inc: Vec<C>,
    /// Port-2 row and its columns c0..c1.
    pub y2: usize,
    pub c0: usize,
    pub c1: usize,
    /// Port-2 mode profile at each frequency (on c0..c1).
    pub u: Vec<Vec<f64>>,
    /// Orthonormal basis of the profiles and the coefficients b[k][m] = <u_k, phi_m>.
    pub basis: Vec<Vec<f64>>,
    pub coeff: Vec<Vec<f64>>,
    /// Largest relative residual of a profile outside the basis.
    pub basis_residual: f64,
    /// eps on the port-2 row (adjoint source weights are divided by it).
    pub eps_row: Vec<f64>,
    /// Port-1 reflection term, if requested (`Objective::with_reflection`).
    pub p1: Option<Port1>,
}

/// Port-1 reflection. P_k = <v_k, Ez(x1, r0..r1)> / <v_k, v_k> with the
/// port-1 mode profile v_k. The incident wave and the reference run do not
/// depend on the design, so the reflected amplitude is b_k = P_k - P_ref,k and
/// R_k = |b_k|^2 / |a_inc,k|^2. Radiation that reaches the port-1 column
/// enters P_k only through its overlap with v_k.
pub struct Port1 {
    pub x1: usize,
    pub r0: usize,
    pub v: Vec<Vec<f64>>,
    pub basis: Vec<Vec<f64>>,
    pub coeff: Vec<Vec<f64>>,
    pub basis_residual: f64,
    pub eps_col: Vec<f64>,
    pub p_ref: Vec<C>,
}

impl Objective {
    pub fn new(setup: &PortSetup, reference: &PortData) -> Self {
        let p = setup.pml.cells;
        let (nx, ny) = setup.dut_grid();
        let y2 = setup.port2();
        let (c0, c1) = (p + 1, nx - p - 1);
        let (r0, r1) = (p + 1, ny - p - 1);
        let col = setup.eps_column();
        let row: Vec<f64> = (0..nx).map(|i| setup.eps_at(Run::Dut, i, y2)).collect();
        let freqs = reference.freqs.clone();
        let mut a_inc = Vec::new();
        let mut u = Vec::new();
        for (k, &f) in freqs.iter().enumerate() {
            let m1 = discrete_mode(&col, f, courant());
            let (fw, _) = mode_amplitudes(
                &m1.profile[r0..r1],
                m1.beta,
                f,
                &reference.p1[0][k],
                &reference.p1[1][k],
                &reference.p1[2][k],
                true,
            );
            a_inc.push(fw);
            u.push(discrete_mode(&row, f, courant()).profile[c0..c1].to_vec());
        }
        let (basis, coeff, residual) = orthonormal_basis(&u);
        Self { freqs, a_inc, y2, c0, c1, u, basis, coeff, basis_residual: residual, eps_row: row[c0..c1].to_vec(), p1: None }
    }

    /// As `new`, with the port-1 reflection term.
    pub fn with_reflection(setup: &PortSetup, reference: &PortData) -> Self {
        let mut o = Self::new(setup, reference);
        let p = setup.pml.cells;
        let (_, ny) = setup.dut_grid();
        let (r0, r1) = (p + 1, ny - p - 1);
        let col = setup.eps_column();
        let v: Vec<Vec<f64>> = o.freqs.iter().map(|&f| discrete_mode(&col, f, courant()).profile[r0..r1].to_vec()).collect();
        let (basis, coeff, basis_residual) = orthonormal_basis(&v);
        let p_ref = (0..o.freqs.len()).map(|k| project_profile(&v[k], &reference.p1[0][k])).collect();
        o.p1 = Some(Port1 { x1: setup.port1_x, r0, v, basis, coeff, basis_residual, eps_col: col[r0..r1].to_vec(), p_ref });
        o
    }

    /// Port-1 reflected amplitudes and reflections from the DFT of Ez on the port-1 column.
    pub fn reflection(&self, column: &[Vec<C>]) -> (Vec<C>, Vec<f64>) {
        let p1 = self.p1.as_ref().expect("objective without port 1");
        let mut b = Vec::new();
        let mut r = Vec::new();
        for k in 0..self.freqs.len() {
            let pk = project_profile(&p1.v[k], &column[k]);
            let bk = (pk.0 - p1.p_ref[k].0, pk.1 - p1.p_ref[k].1);
            r.push(cabs2(bk) / cabs2(self.a_inc[k]));
            b.push(bk);
        }
        (b, r)
    }

    /// Port-2 amplitudes and transmissions from the DFT of Ez on the port row.
    pub fn transmission(&self, port_row: &[Vec<C>]) -> (Vec<C>, Vec<f64>) {
        let mut a = Vec::new();
        let mut t = Vec::new();
        for k in 0..self.freqs.len() {
            let uu: f64 = self.u[k].iter().map(|x| x * x).sum();
            let s = self.u[k].iter().zip(&port_row[k]).fold((0.0, 0.0), |acc, (w, e)| (acc.0 + w * e.0, acc.1 + w * e.1));
            let ak = (s.0 / uu, s.1 / uu);
            t.push(cabs2(ak) / cabs2(self.a_inc[k]));
            a.push(ak);
        }
        (a, t)
    }
}

fn project_profile(v: &[f64], e: &[C]) -> C {
    let vv: f64 = v.iter().map(|x| x * x).sum();
    let s = v.iter().zip(e).fold((0.0, 0.0), |acc, (w, e)| (acc.0 + w * e.0, acc.1 + w * e.1));
    (s.0 / vv, s.1 / vv)
}

/// Result of a forward run: amplitudes, transmissions and the box field.
pub struct Forward {
    pub a: Vec<C>,
    pub t: Vec<f64>,
    /// Port-1 reflected amplitudes and reflections (empty without port 1).
    pub b: Vec<C>,
    pub r: Vec<f64>,
    /// DFT of Ez on the box cells, per frequency (index b * n + a).
    pub e_box: Vec<Vec<C>>,
}

fn source_pulse(setup: &PortSetup, n: usize) -> f64 {
    gaussian_sine((n + 1) as f64 * courant(), setup.fc(), setup.t0(), setup.tau())
}

/// Forward run of the bend DUT (with its design) on the GPU.
pub fn forward(setup: &PortSetup, obj: &Objective, bx: &DesignBox) -> Result<Forward, Box<dyn Error>> {
    let (nx, ny) = setup.dut_grid();
    let mut g = GpuYee::new(nx, ny, setup.pml, courant(), &|i, j| setup.eps_at(Run::Dut, i, j))?;
    let p = setup.pml.cells;
    let (r0, r1) = (p + 1, ny - p - 1);
    let mode = discrete_mode(&setup.eps_column(), setup.fc(), courant());
    let prof = g.upload_profile(&mode.profile[r0..r1])?;
    let mut port = g.dft_line(Field::Ez, (obj.c0, obj.y2), obj.c1 - obj.c0, true, &obj.freqs)?;
    let mut region = g.dft_rect(Field::Ez, (bx.x0, bx.y0), bx.n, bx.n, &obj.freqs)?;
    let mut col = match &obj.p1 {
        Some(p1) => Some(g.dft_line(Field::Ez, (p1.x1, r0), r1 - r0, false, &obj.freqs)?),
        None => None,
    };
    for n in 0..setup.steps() {
        g.step()?;
        g.add_source_profile(setup.src_x, r0, &prof, source_pulse(setup, n))?;
        let t = (n + 1) as f64 * courant();
        g.accumulate(&mut port, t)?;
        g.accumulate(&mut region, t)?;
        if let Some(c) = col.as_mut() {
            g.accumulate(c, t)?;
        }
    }
    let port_row = port.download(g.stream())?;
    let (a, t) = obj.transmission(&port_row);
    let (b, r) = match &col {
        Some(c) => obj.reflection(&c.download(g.stream())?),
        None => (Vec::new(), Vec::new()),
    };
    Ok(Forward { a, t, b, r, e_box: region.download(g.stream())? })
}

/// G_hat(omega_k) of the pulse at the Ez sample times.
fn pulse_dft(setup: &PortSetup, freqs: &[f64]) -> Vec<C> {
    let (dt, steps) = (courant(), setup.steps());
    freqs
        .iter()
        .map(|&f| {
            let w = 2.0 * PI * f;
            (0..steps).fold((0.0, 0.0), |acc, n| {
                let (t, v) = ((n + 1) as f64 * dt, source_pulse(setup, n));
                (acc.0 + v * (w * t).cos(), acc.1 - v * (w * t).sin())
            })
        })
        .collect()
}

/// lambda_k = G^T c_k on the box cells, for the functional
/// c_k^T E = <u_k, E(line)> / <u_k, u_k> on the line start + i stride.
/// One adjoint run per basis vector.
#[allow(clippy::too_many_arguments)]
fn adjoint_field(
    setup: &PortSetup,
    bx: &DesignBox,
    freqs: &[f64],
    u: &[Vec<f64>],
    basis: &[Vec<f64>],
    coeff: &[Vec<f64>],
    eps_line: &[f64],
    (start, stride): (usize, usize),
) -> Result<Vec<Vec<C>>, Box<dyn Error>> {
    let (nx, ny) = setup.dut_grid();
    let dt = courant();
    let nf = freqs.len();
    let steps = setup.steps();
    let ghat = pulse_dft(setup, freqs);
    let mut lam: Vec<Vec<C>> = vec![vec![(0.0, 0.0); bx.n * bx.n]; nf];
    for (m, phi) in basis.iter().enumerate() {
        let mut g = GpuYee::new(nx, ny, setup.pml, courant(), &|i, j| setup.eps_at(Run::Dut, i, j))?;
        let w: Vec<f64> = phi.iter().zip(eps_line).map(|(p, e)| p / e).collect();
        let prof = g.upload_profile(&w)?;
        let mut region = g.dft_rect(Field::Ez, (bx.x0, bx.y0), bx.n, bx.n, freqs)?;
        for n in 0..steps {
            g.step()?;
            g.add_source_strided(start, stride, &prof, source_pulse(setup, n))?;
            g.accumulate(&mut region, (n + 1) as f64 * dt)?;
        }
        let resp = region.download(g.stream())?;
        for k in 0..nf {
            let w = 2.0 * PI * freqs[k];
            let om = 2.0 / dt * (w * dt / 2.0).sin();
            let norm = cmul(cmul((0.0, om), ((w * dt / 2.0).cos(), (w * dt / 2.0).sin())), (ghat[k].0 / dt, ghat[k].1 / dt));
            let uu: f64 = u[k].iter().map(|x| x * x).sum();
            let c = coeff[k][m] / uu;
            for (l, r) in lam[k].iter_mut().zip(&resp[k]) {
                let v = cdiv(*r, norm);
                *l = (l.0 + c * v.0, l.1 + c * v.1);
            }
        }
    }
    Ok(lam)
}

/// d|amp_k|^2 / |a_inc,k|^2 / d eps_r = 2 Omega^2 Re(conj(amp_k) E_r lambda_r) / |a_inc,k|^2.
fn power_gradient(obj: &Objective, fwd: &Forward, amp: &[C], lam: &[Vec<C>]) -> Vec<Vec<f64>> {
    let dt = courant();
    (0..obj.freqs.len())
        .map(|k| {
            let w = 2.0 * PI * obj.freqs[k];
            let om2 = (2.0 / dt * (w * dt / 2.0).sin()).powi(2);
            let inc = cabs2(obj.a_inc[k]);
            let conj_a = (amp[k].0, -amp[k].1);
            fwd.e_box[k]
                .iter()
                .zip(&lam[k])
                .map(|(e, l)| {
                    let da = cmul(*e, *l);
                    2.0 * om2 * cmul(conj_a, da).0 / inc
                })
                .collect()
        })
        .collect()
}

/// dT_k / d eps_r for every box cell r and every frequency k
/// (1 forward run, done by the caller, and one adjoint run per port-2 basis vector).
pub fn gradient(setup: &PortSetup, obj: &Objective, bx: &DesignBox, fwd: &Forward) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let (nx, _) = setup.dut_grid();
    let lam = adjoint_field(setup, bx, &obj.freqs, &obj.u, &obj.basis, &obj.coeff, &obj.eps_row, (obj.y2 * nx + obj.c0, 1))?;
    Ok(power_gradient(obj, fwd, &fwd.a, &lam))
}

/// dR_k / d eps_r (port-1 reflection) for every box cell r and every frequency k
/// (one adjoint run per port-1 basis vector; reuses the caller's forward run).
pub fn gradient_r(setup: &PortSetup, obj: &Objective, bx: &DesignBox, fwd: &Forward) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let (nx, _) = setup.dut_grid();
    let p1 = obj.p1.as_ref().expect("objective without port 1");
    let lam = adjoint_field(setup, bx, &obj.freqs, &p1.v, &p1.basis, &p1.coeff, &p1.eps_col, (p1.r0 * nx + p1.x1, nx))?;
    Ok(power_gradient(obj, fwd, &fwd.b, &lam))
}

/// Smooth worst-case loss: F = (mean_k (1 - T_k)^p)^(1/p), and dF/dT_k.
pub fn pnorm_loss(t: &[f64], p: f64) -> (f64, Vec<f64>) {
    let n = t.len() as f64;
    let mean: f64 = t.iter().map(|&x| (1.0 - x).max(1e-300).powf(p)).sum::<f64>() / n;
    let f = mean.powf(1.0 / p);
    let d = t.iter().map(|&x| -mean.powf(1.0 / p - 1.0) * (1.0 - x).max(1e-300).powf(p - 1.0) / n).collect();
    (f, d)
}
