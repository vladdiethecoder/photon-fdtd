//! Gradient verification for the Stage 5 adjoint (before any optimiser).
//!
//! 1. 4 x 4 block: adjoint dT_k/deps against central differences per cell,
//!    at several wavelengths: cosine, magnitude ratio, sign.
//! 2. Directional derivatives of T_k along a random block direction,
//!    step sizes 1e-1 .. 1e-4.
//! 3. Full chain latent rho -> filter -> projection -> eps -> F (p-norm of
//!    the losses): directional derivative over the whole box.

use photon_fdtd::adjoint::{Objective, forward, gradient, pnorm_loss};
use photon_fdtd::design::{ConicFilter, DesignBox, densities, pullback};
use photon_fdtd::sparams::{Geometry, PortSetup, Run};
use photon_fdtd::verify::{EPS_AIR, EPS_SI};

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut base = PortSetup::standard(Geometry::Bend, 1);
    if let Some(p) = std::env::args().position(|a| a == "--extra-time") {
        base.extra_time = std::env::args().nth(p + 1).unwrap().parse()?;
    }
    println!("steps {}", base.steps());
    let bx = DesignBox::for_bend(&base, 48, 8);
    let n = bx.n;
    let filt = ConicFilter::new(5.0);
    let beta = 4.0;
    // Grey latent density (0.15 .. 0.85) so that +-h steps stay inside [0, 1].
    let rho0: Vec<f64> = bx.circular_arc().iter().map(|v| 0.15 + 0.7 * v).collect();
    let rho_bar = densities(&bx, &filt, &rho0, beta);
    let reference = base.run(Run::Reference)?;
    let setup = base.clone().with_design(&bx, &rho_bar);
    let obj = Objective::new(&setup, &reference);
    println!("adjoint basis: {} vectors, max residual {:.2e}", obj.basis.len(), obj.basis_residual);
    let t0 = std::time::Instant::now();
    let fwd = forward(&setup, &obj, &bx)?;
    let grad = gradient(&setup, &obj, &bx, &fwd)?;
    println!("forward + adjoint: {:.1} s; T at k=0,7,14: {:.5} {:.5} {:.5}", t0.elapsed().as_secs_f64(), fwd.t[0], fwd.t[7], fwd.t[14]);
    let eps0: Vec<f64> = rho_bar.iter().map(|&r| DesignBox::eps(r)).collect();
    let eval = |eps: &[f64]| -> Vec<f64> {
        let rb: Vec<f64> = eps.iter().map(|&e| (e - EPS_AIR) / (EPS_SI - EPS_AIR)).collect();
        let st = base.clone().with_design(&bx, &rb);
        forward(&st, &obj, &bx).unwrap().t
    };

    // 1. 4 x 4 block where the field is strong: around the arc centreline at 45 degrees.
    let r = bx.arc_radius() as f64;
    let (ca, cb) = ((r * 0.7071) as usize - 2, (n as f64 - r * 0.7071) as usize - 2);
    let cells: Vec<usize> = (0..16).map(|q| (cb + q / 4) * n + ca + q % 4).collect();
    let h = 1e-3;
    let mut fd = vec![vec![0.0; 16]; obj.freqs.len()];
    for (q, &c) in cells.iter().enumerate() {
        let (mut p, mut m) = (eps0.clone(), eps0.clone());
        p[c] += h;
        m[c] -= h;
        let (tp, tm) = (eval(&p), eval(&m));
        for k in 0..obj.freqs.len() {
            fd[k][q] = (tp[k] - tm[k]) / (2.0 * h);
        }
    }
    println!("block cells (a, b) from ({ca}, {cb}), h = {h}");
    for &k in &[0usize, 4, 7, 10, 14] {
        let ad: Vec<f64> = cells.iter().map(|&c| grad[k][c]).collect();
        let dot: f64 = ad.iter().zip(&fd[k]).map(|(a, b)| a * b).sum();
        let na = ad.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = fd[k].iter().map(|x| x * x).sum::<f64>().sqrt();
        let ratios: Vec<f64> = ad.iter().zip(&fd[k]).map(|(a, b)| a / b).collect();
        let signs = ad.iter().zip(&fd[k]).filter(|(a, b)| a.signum() == b.signum()).count();
        println!(
            "  lambda0 {:>6.1}: cosine {:.7}, |adj|/|fd| {:.6}, per-cell ratio {:.5}..{:.5}, signs {}/16",
            1.0 / obj.freqs[k],
            dot / (na * nb),
            na / nb,
            ratios.iter().cloned().fold(f64::INFINITY, f64::min),
            ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            signs
        );
    }

    // 2. Directional derivative along a random block direction.
    let mut seed = 11u64;
    let d: Vec<f64> = (0..16).map(|_| lcg(&mut seed) - 0.5).collect();
    println!("directional (block), ratio adjoint / central difference:");
    for &hs in &[1e-1, 1e-2, 1e-3, 1e-4] {
        let (mut p, mut m) = (eps0.clone(), eps0.clone());
        for (q, &c) in cells.iter().enumerate() {
            p[c] += hs * d[q];
            m[c] -= hs * d[q];
        }
        let (tp, tm) = (eval(&p), eval(&m));
        let line: Vec<String> = [0usize, 7, 14]
            .iter()
            .map(|&k| {
                let adj: f64 = cells.iter().zip(&d).map(|(&c, dq)| grad[k][c] * dq).sum();
                format!("{:.1}: {:.8}", 1.0 / obj.freqs[k], adj / ((tp[k] - tm[k]) / (2.0 * hs)))
            })
            .collect();
        println!("  h {hs:.0e}: {}", line.join("  "));
    }

    // 3. Full chain from the latent density to F.
    let p_norm = 8.0;
    let (f0, df_dt) = pnorm_loss(&fwd.t, p_norm);
    let mut g_bar = vec![0.0; n * n];
    for (k, w) in df_dt.iter().enumerate() {
        for (gb, gk) in g_bar.iter_mut().zip(&grad[k]) {
            *gb += w * gk * (EPS_SI - EPS_AIR);
        }
    }
    let g_rho = pullback(&bx, &filt, &rho0, beta, &g_bar);
    let dr: Vec<f64> = (0..n * n).map(|_| lcg(&mut seed) - 0.5).collect();
    let adj: f64 = g_rho.iter().zip(&dr).map(|(a, b)| a * b).sum();
    println!("full chain rho -> F (F = {f0:.6}), ratio adjoint / central difference:");
    for &hs in &[1e-2, 1e-3, 1e-4, 1e-5] {
        let fp = |sgn: f64| {
            let r: Vec<f64> = rho0.iter().zip(&dr).map(|(a, b)| a + sgn * hs * b).collect();
            let st = base.clone().with_design(&bx, &densities(&bx, &filt, &r, beta));
            pnorm_loss(&forward(&st, &obj, &bx).unwrap().t, p_norm).0
        };
        let fdv = (fp(1.0) - fp(-1.0)) / (2.0 * hs);
        println!("  h {hs:.0e}: adjoint {adj:.8e}, fd {fdv:.8e}, ratio {:.8}", adj / fdv);
    }
    Ok(())
}
