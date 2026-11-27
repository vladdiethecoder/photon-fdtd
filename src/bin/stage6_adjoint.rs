//! Stage 6 adjoint optimiser (docs/stage6-protocol.md, sections 7 and 8).
//!
//! Phase 1 (practical, full resolution):
//!   stage6_adjoint --rep full --rf 6 --lr 0.05 --betas 1,2,4,8,16,32,64,128 --iters 12 --score-every 4 --cap 3000 --init arc
//! Phase 2 (benchmark, shared 20 x 20 grid):
//!   stage6_adjoint --rep grid20 --rf 6 --lr LR --betas 2,4,8,16,32,64 --iters 6 --hold-last --score-every 2 --cap 1200 --init I1
//!
//! eta_e and eta_d come from the calibration of section 7 (the arc at r_f).
//! Every iteration is charged 12 (T gradient, with the worst variant's
//! forward solve) + 12 (R gradient) + 2 (the other variants' forward solves).

use photon_fdtd::adjoint::Objective;
use photon_fdtd::design::{ConicFilter, DesignBox, binary_fraction, project};
use photon_fdtd::robust::{Relaxed, Rep, Scorer, bias, eta_for_area, nominal_mask, relaxed};
use photon_fdtd::sparams::{Geometry as G, PortSetup, Run};
use photon_fdtd::stage6::{RunLog, arg, named_mask};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rep = match arg("--rep", "full".to_string()).as_str() {
        "full" => Rep::Full,
        "grid20" => Rep::Grid(20),
        r => return Err(format!("unknown --rep {r}").into()),
    };
    let rf: f64 = arg("--rf", 6.0);
    let lr: f64 = arg("--lr", 0.05);
    let betas: Vec<f64> = arg("--betas", "1,2,4,8,16,32,64,128".to_string()).split(',').map(|b| b.parse().unwrap()).collect();
    let iters: usize = arg("--iters", 12);
    let hold_last = std::env::args().any(|a| a == "--hold-last");
    let score_every: usize = arg("--score-every", 4);
    let cap: usize = arg("--cap", 3000);
    let init: String = arg("--init", "arc".to_string());
    let out: String = arg("--out", "out/stage6/adjoint".to_string());

    let mut base = PortSetup::standard(G::Bend, 1);
    base.extra_time = 6000.0;
    let bx = DesignBox::for_bend(&base, 80, 8);
    let n = bx.n;
    let filt = ConicFilter::new(rf);
    let reference = base.run(Run::Reference)?;
    let arc = bx.circular_arc();
    let fa = filt.apply(&bx, &arc);
    let area = |s: &[f64]| s.iter().filter(|&&v| v >= 0.5).count();
    let eta = [eta_for_area(&fa, area(&bias(&bx, &arc, 1, false))), 0.5, eta_for_area(&fa, area(&bias(&bx, &arc, 1, true)))];
    let obj = Objective::with_reflection(&base.clone().with_design(&bx, &arc), &reference);
    let (nb2, nb1) = (obj.basis.len(), obj.p1.as_ref().unwrap().basis.len());
    let charge = 12 + 12 + 2;
    println!(
        "rep {rep:?}, r_f {rf}, lr {lr}, betas {betas:?} x {iters}{}, score every {score_every}, cap {cap}, init {init}, eta {eta:?}",
        if hold_last { " then hold" } else { "" }
    );
    println!("adjoint basis: port 2 {nb2}, port 1 {nb1}; physical solves per iteration {}, charged {charge}", 3 + nb2 + nb1);
    let mut log = RunLog::new(&out, n, cap, &[150, 300, 600, 1200, 3000])?;
    let mut scorer = Scorer::new(base.clone(), bx.clone(), reference);
    let mut x = rep.encode(&named_mask(&bx, &init), n);
    let dim = x.len();
    let (mut m1, mut m2) = (vec![0.0; dim], vec![0.0; dim]);
    let mut t = 0;
    let mut schedule: Vec<f64> = betas.iter().flat_map(|&b| std::iter::repeat_n(b, iters)).collect();
    if hold_last {
        schedule.extend(std::iter::repeat_n(*betas.last().unwrap(), 100_000));
    }
    let score = |log: &mut RunLog, scorer: &mut Scorer, x: &[f64], note: &str| -> Result<bool, Box<dyn std::error::Error>> {
        if !log.fits(3) {
            return Ok(false);
        }
        let mask = nominal_mask(rep, &filt, &bx, x);
        let s0 = scorer.forward_seconds;
        let s = scorer.score(&mask)?;
        let ds = scorer.forward_seconds - s0;
        log.score(&s, &mask, x, ds, note);
        println!("  score {note}: J {:.4}  best {:.4}  charged {}", s.j, log.best_j(), log.charged);
        Ok(true)
    };
    let mut last_beta = schedule[0];
    for (it, &beta) in schedule.iter().enumerate() {
        if !log.fits(charge) {
            break;
        }
        last_beta = beta;
        let rx = Relaxed { eta, beta, p: 8.0 };
        let ev = relaxed(&rx, rep, &filt, &base, &bx, &obj, &x)?;
        assert_eq!(ev.forward_solves + ev.adjoint_solves, charge, "charge differs from the physical solves");
        log.iteration(ev.forward_solves, ev.adjoint_solves, charge, ev.forward_seconds, ev.adjoint_seconds, &format!("beta {beta} F {:.6} worst {}", ev.f, ev.worst));
        println!(
            "it {it:>3} beta {beta:>4}  F {:.6}  F_v [{:.5} {:.5} {:.5}]  worst {}  charged {}  fwd {:.1} s adj {:.1} s",
            ev.f, ev.f_v[0], ev.f_v[1], ev.f_v[2], ev.worst, log.charged, ev.forward_seconds, ev.adjoint_seconds
        );
        t += 1;
        let (b1, b2) = (0.9f64, 0.999f64);
        for q in 0..dim {
            m1[q] = b1 * m1[q] + (1.0 - b1) * ev.grad[q];
            m2[q] = b2 * m2[q] + (1.0 - b2) * ev.grad[q] * ev.grad[q];
            let (mh, vh) = (m1[q] / (1.0 - b1.powi(t)), m2[q] / (1.0 - b2.powi(t)));
            x[q] = (x[q] - lr * mh / (vh.sqrt() + 1e-12)).clamp(0.0, 1.0);
        }
        if (it + 1) % score_every == 0 && !score(&mut log, &mut scorer, &x, &format!("it {it}"))? {
            break;
        }
    }
    score(&mut log, &mut scorer, &x, "final")?;
    let rb: Vec<f64> = filt.apply(&bx, &rep.latent(&x, n)).iter().map(|&v| project(v, last_beta, 0.5)).collect();
    let extra = format!(
        ",\n  \"method\": \"adjoint\",\n  \"rep\": \"{rep:?}\",\n  \"rf\": {rf},\n  \"lr\": {lr},\n  \"init\": \"{init}\",\n  \"eta\": [{:.9}, 0.5, {:.9}],\n  \"last_beta\": {last_beta},\n  \"binary_fraction_last_beta\": {:.6},\n  \"iterations\": {t}",
        eta[0],
        eta[2],
        binary_fraction(&rb)
    );
    photon_fdtd::stage6::write_values(&format!("{out}/x_final.txt"), &x)?;
    log.finish(&extra)?;
    println!("done: best J {:.4}, charged {}, forward {} adjoint {} solves, {} cache hits", log.best_j(), log.charged, log.forward_solves, log.adjoint_solves, log.cache_hits);
    Ok(())
}
