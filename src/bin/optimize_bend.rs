//! Stage 5 adjoint optimiser for the 48 x 48 bend (docs/stage5-protocol.md).
//!
//! Latent rho -> conic filter -> tanh projection (beta 1 .. 32) -> eps.
//! Objective: F = (mean_k (1 - T_k)^p)^(1/p) over the 15 band samples
//! (smooth worst-case loss). Adam on rho, clipped to [0, 1].
//! Writes the latent density after each beta stage to OUT/rho_beta_B.txt and
//! the final binary design to OUT/binary.txt.
//!
//! Usage: optimize_bend [--n 48] [--rf 5] [--iters 20] [--lr 0.05] [--p 8] [--out out/opt]
//!                      [--betas 1,2,4,8,16,32] [--init RHO.txt]

use std::fs;

use photon_fdtd::adjoint::{Objective, forward, gradient, pnorm_loss};
use photon_fdtd::design::{ConicFilter, DesignBox, binary_fraction, densities, disk, feature_violations, pullback, single_cell_islands};
use photon_fdtd::sparams::{Geometry, PortSetup, Run, db};
use photon_fdtd::verify::{EPS_AIR, EPS_SI};

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).map(|p| a[p + 1].parse().ok().unwrap()).unwrap_or(default)
}

fn save(path: &str, v: &[f64]) {
    fs::write(path, v.iter().map(|x| format!("{x:.17e}")).collect::<Vec<_>>().join("\n")).unwrap();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (n, rf, iters, lr, p): (usize, f64, usize, f64, f64) =
        (arg("--n", 48), arg("--rf", 5.0), arg("--iters", 20), arg("--lr", 0.05), arg("--p", 8.0));
    let out: String = arg("--out", "out/opt".to_string());
    fs::create_dir_all(&out)?;
    let mut base = PortSetup::standard(Geometry::Bend, 1);
    base.extra_time = 6000.0;
    let bx = DesignBox::for_bend(&base, n, 8);
    let filt = ConicFilter::new(rf);
    let reference = base.run(Run::Reference)?;
    let obj = Objective::new(&base.clone().with_design(&bx, &bx.circular_arc()), &reference);
    println!("box {n}, r_f {rf}, iters/beta {iters}, lr {lr}, p {p}, steps {}, adjoint basis {}", base.steps(), obj.basis.len());
    let betas: Vec<f64> = arg("--betas", "1,2,4,8,16,32".to_string()).split(',').map(|b| b.parse().unwrap()).collect();
    let mut rho = match std::env::args().position(|a| a == "--init") {
        Some(p) => fs::read_to_string(std::env::args().nth(p + 1).unwrap())?.lines().map(|l| l.trim().parse().unwrap()).collect(),
        None => bx.circular_arc(),
    };
    let (mut m1, mut m2) = (vec![0.0; n * n], vec![0.0; n * n]);
    let mut t_adam = 0;
    let t_start = std::time::Instant::now();
    for &beta in &betas {
        for it in 0..iters {
            let rho_bar = densities(&bx, &filt, &rho, beta);
            let setup = base.clone().with_design(&bx, &rho_bar);
            let fwd = forward(&setup, &obj, &bx)?;
            let grad = gradient(&setup, &obj, &bx, &fwd)?;
            let (f, dfdt) = pnorm_loss(&fwd.t, p);
            let mut g_bar = vec![0.0; n * n];
            for (k, w) in dfdt.iter().enumerate() {
                for (gb, gk) in g_bar.iter_mut().zip(&grad[k]) {
                    *gb += w * gk * (EPS_SI - EPS_AIR);
                }
            }
            let g = pullback(&bx, &filt, &rho, beta, &g_bar);
            t_adam += 1;
            let (b1, b2) = (0.9f64, 0.999f64);
            for q in 0..n * n {
                m1[q] = b1 * m1[q] + (1.0 - b1) * g[q];
                m2[q] = b2 * m2[q] + (1.0 - b2) * g[q] * g[q];
                let (mh, vh) = (m1[q] / (1.0 - b1.powi(t_adam)), m2[q] / (1.0 - b2.powi(t_adam)));
                rho[q] = (rho[q] - lr * mh / (vh.sqrt() + 1e-12)).clamp(0.0, 1.0);
            }
            let worst = fwd.t.iter().cloned().fold(f64::INFINITY, f64::min);
            println!(
                "beta {beta:>4} it {it:>3}  F {f:.6}  worst IL {:.4} dB  mean T {:.5}  binary {:.3}  {:.0} s",
                -db(worst),
                fwd.t.iter().sum::<f64>() / fwd.t.len() as f64,
                binary_fraction(&rho_bar),
                t_start.elapsed().as_secs_f64()
            );
        }
        save(&format!("{out}/rho_beta_{beta}.txt"), &rho);
    }
    // Final binary design: the 0.5 level set of the filtered density.
    let beta_final = *betas.last().unwrap();
    let rho_bar = densities(&bx, &filt, &rho, beta_final);
    let bin: Vec<f64> = rho_bar.iter().map(|&v| if v >= 0.5 { 1.0 } else { 0.0 }).collect();
    save(&format!("{out}/binary.txt"), &bin);
    save(&format!("{out}/rho_final.txt"), &rho);
    let st = base.clone().with_design(&bx, &bin);
    let s = st.sparams(&reference, &st.run(Run::Dut)?);
    let il = s.s21_mode.iter().map(|v| -db(*v)).fold(0.0, f64::max);
    let rl = s.s11_mode.iter().map(|v| -db(*v)).fold(f64::INFINITY, f64::min);
    let (vo, vc) = feature_violations(&bx, &bin, &disk(4));
    println!(
        "final: binary before threshold {:.4}, islands {}, L_min violations open {vo} close {vc}, worst IL {il:.4} dB, worst RL {rl:.2} dB",
        binary_fraction(&rho_bar),
        single_cell_islands(&bx, &bin)
    );
    for k in 0..s.freqs.len() {
        println!("  lambda0 {:>6.1}  IL {:.4}  RL {:.2}", 1.0 / s.freqs[k], -db(s.s21_mode[k]), -db(s.s11_mode[k]));
    }
    Ok(())
}
