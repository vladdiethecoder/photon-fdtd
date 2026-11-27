//! Stage 6 black-box optimisers on the shared 20 x 20 representation
//! (docs/stage6-protocol.md, section 8): CMA-ES and one-step PPO.
//!
//!   stage6_bb --method cma --sigma0 S --seed i --init Ii --cap 1200 --out DIR
//!   stage6_bb --method ppo --lr L     --seed i --init Ii --cap 1200 --out DIR
//!
//! Each candidate is the binary nominal mask of its control values (clipped
//! to [0, 1]), scored with the shared `Scorer`: 3 forward solves, or 0 for a
//! cache hit or a mask that fails a geometry gate. The run stops when the next
//! score could pass the cap, or after 20000 candidates.

use photon_fdtd::design::{ConicFilter, DesignBox};
use photon_fdtd::robust::{Rep, Scorer, nominal_mask};
use photon_fdtd::sparams::{Geometry as G, PortSetup, Run};
use photon_fdtd::stage6::{Rng, RunLog, arg, named_mask, write_values};

const MAX_CANDIDATES: usize = 20_000;

struct Eval<'a> {
    rep: Rep,
    filt: &'a ConicFilter,
    bx: &'a DesignBox,
    scorer: Scorer,
    log: RunLog,
    candidates: usize,
}

impl Eval<'_> {
    /// J of control values x, or None when the run must stop.
    fn j(&mut self, x: &[f64], note: &str) -> Result<Option<f64>, Box<dyn std::error::Error>> {
        if !self.log.fits(3) || self.candidates >= MAX_CANDIDATES {
            return Ok(None);
        }
        self.candidates += 1;
        let mask = nominal_mask(self.rep, self.filt, self.bx, x);
        let s0 = self.scorer.forward_seconds;
        let s = self.scorer.score(&mask)?;
        let ds = self.scorer.forward_seconds - s0;
        let xc: Vec<f64> = x.iter().map(|v| v.clamp(0.0, 1.0)).collect();
        self.log.score(&s, &mask, &xc, ds, note);
        Ok(Some(s.j))
    }
}

/// Cyclic Jacobi eigen-decomposition of a symmetric matrix (row-major):
/// eigenvalues and eigenvectors (columns).
fn eigh(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..50 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i == j { diag += a[i * n + j] * a[i * n + j] } else { off += a[i * n + j] * a[i * n + j] }
            }
        }
        if off <= 1e-22 * diag {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = if theta == 0.0 { 1.0 } else { theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt()) };
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

fn cma(ev: &mut Eval, x0: &[f64], sigma0: f64, rng: &mut Rng) -> Result<String, Box<dyn std::error::Error>> {
    let n = x0.len();
    let nf = n as f64;
    let lambda = 4 + (3.0 * nf.ln()).floor() as usize;
    let mu = lambda / 2;
    let w0: Vec<f64> = (0..mu).map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).collect();
    let ws: f64 = w0.iter().sum();
    let w: Vec<f64> = w0.iter().map(|x| x / ws).collect();
    let mueff = 1.0 / w.iter().map(|x| x * x).sum::<f64>();
    let cs = (mueff + 2.0) / (nf + mueff + 5.0);
    let ds = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
    let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
    let c1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
    let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
    let chin = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
    println!("CMA-ES: n {n}, lambda {lambda}, mu {mu}, mueff {mueff:.3}, cs {cs:.4}, ds {ds:.4}, cc {cc:.4}, c1 {c1:.3e}, cmu {cmu:.3e}, sigma0 {sigma0}");
    let (mut m, mut sigma) = (x0.to_vec(), sigma0);
    let (mut ps, mut pc) = (vec![0.0; n], vec![0.0; n]);
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        c[i * n + i] = 1.0;
    }
    let (mut d, mut b) = (vec![1.0; n], c.clone());
    for g in 0.. {
        let mut pop: Vec<(f64, Vec<f64>)> = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            let z: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let y: Vec<f64> = (0..n).map(|i| (0..n).map(|k| b[i * n + k] * d[k] * z[k]).sum()).collect();
            let x: Vec<f64> = (0..n).map(|i| m[i] + sigma * y[i]).collect();
            match ev.j(&x, &format!("gen {g}"))? {
                Some(j) => pop.push((j, y)),
                None => return Ok(format!(",\n  \"generations\": {g},\n  \"sigma_final\": {sigma:.6}")),
            }
        }
        pop.sort_by(|p, q| p.0.total_cmp(&q.0));
        let yw: Vec<f64> = (0..n).map(|i| (0..mu).map(|k| w[k] * pop[k].1[i]).sum()).collect();
        for i in 0..n {
            m[i] += sigma * yw[i];
        }
        let bt: Vec<f64> = (0..n).map(|k| (0..n).map(|i| b[i * n + k] * yw[i]).sum::<f64>() / d[k]).collect();
        let cinv: Vec<f64> = (0..n).map(|i| (0..n).map(|k| b[i * n + k] * bt[k]).sum()).collect();
        let a = (cs * (2.0 - cs) * mueff).sqrt();
        for i in 0..n {
            ps[i] = (1.0 - cs) * ps[i] + a * cinv[i];
        }
        let psn = ps.iter().map(|x| x * x).sum::<f64>().sqrt();
        let hs = psn / (1.0 - (1.0 - cs).powi(2 * (g as i32 + 1))).sqrt() < (1.4 + 2.0 / (nf + 1.0)) * chin;
        let hsf = if hs { 1.0 } else { 0.0 };
        let ac = (cc * (2.0 - cc) * mueff).sqrt();
        for i in 0..n {
            pc[i] = (1.0 - cc) * pc[i] + hsf * ac * yw[i];
        }
        let keep = 1.0 - c1 - cmu + (1.0 - hsf) * c1 * cc * (2.0 - cc);
        for i in 0..n {
            for j in 0..n {
                let rank_mu: f64 = (0..mu).map(|k| w[k] * pop[k].1[i] * pop[k].1[j]).sum();
                c[i * n + j] = keep * c[i * n + j] + c1 * pc[i] * pc[j] + cmu * rank_mu;
            }
        }
        sigma *= ((cs / ds) * (psn / chin - 1.0)).exp();
        let (ev_, bv) = eigh(&c, n);
        d = ev_.iter().map(|x| x.max(1e-20).sqrt()).collect();
        b = bv;
        println!("gen {g:>3} charged {:>5} best J {:.4} gen best {:.4} sigma {sigma:.4}", ev.log.charged, ev.log.best_j(), pop[0].0);
    }
    unreachable!()
}

fn ppo(ev: &mut Eval, x0: &[f64], lr: f64, rng: &mut Rng) -> Result<String, Box<dyn std::error::Error>> {
    let n = x0.len();
    let (batch, clip, epochs) = (16usize, 0.2f64, 4usize);
    let mut mu = x0.to_vec();
    let mut logsd = 0.10f64.ln();
    let (mut am1, mut am2, mut sm1, mut sm2) = (vec![0.0; n], vec![0.0; n], 0.0f64, 0.0f64);
    let mut t = 0;
    for u in 0.. {
        let sd_old = logsd.exp();
        let mut acts = Vec::with_capacity(batch);
        let mut js = Vec::with_capacity(batch);
        for _ in 0..batch {
            let a: Vec<f64> = mu.iter().map(|m| m + sd_old * rng.normal()).collect();
            match ev.j(&a, &format!("update {u}"))? {
                Some(j) => js.push(j),
                None => return Ok(format!(",\n  \"updates\": {u},\n  \"policy_sd_final\": {:.6}", logsd.exp())),
            }
            acts.push(a);
        }
        // Advantage: centred rank of -J in [-0.5, 0.5].
        let mut idx: Vec<usize> = (0..batch).collect();
        idx.sort_by(|&p, &q| js[q].total_cmp(&js[p]));
        let mut adv = vec![0.0; batch];
        for (rank, &i) in idx.iter().enumerate() {
            adv[i] = rank as f64 / (batch - 1) as f64 - 0.5;
        }
        let mu_old = mu.clone();
        let logp = |a: &[f64], m: &[f64], ls: f64| -> f64 {
            let sd = ls.exp();
            a.iter().zip(m).map(|(a, m)| -(a - m) * (a - m) / (2.0 * sd * sd)).sum::<f64>() - n as f64 * ls
        };
        let lp_old: Vec<f64> = acts.iter().map(|a| logp(a, &mu_old, sd_old.ln())).collect();
        let mut active = 0;
        for _ in 0..epochs {
            let sd = logsd.exp();
            let (mut gm, mut gs) = (vec![0.0; n], 0.0f64);
            for i in 0..batch {
                let r = (logp(&acts[i], &mu, logsd) - lp_old[i]).exp();
                let unclipped_active = if adv[i] >= 0.0 { r < 1.0 + clip } else { r > 1.0 - clip };
                if !unclipped_active {
                    continue;
                }
                active += 1;
                let k = adv[i] * r / batch as f64;
                let mut sq = 0.0;
                for q in 0..n {
                    let dv = acts[i][q] - mu[q];
                    gm[q] += k * dv / (sd * sd);
                    sq += dv * dv;
                }
                gs += k * (sq / (sd * sd) - n as f64);
            }
            t += 1;
            let (b1, b2) = (0.9f64, 0.999f64);
            for q in 0..n {
                am1[q] = b1 * am1[q] + (1.0 - b1) * gm[q];
                am2[q] = b2 * am2[q] + (1.0 - b2) * gm[q] * gm[q];
                let (mh, vh) = (am1[q] / (1.0 - b1.powi(t)), am2[q] / (1.0 - b2.powi(t)));
                mu[q] = (mu[q] + lr * mh / (vh.sqrt() + 1e-12)).clamp(0.0, 1.0);
            }
            sm1 = b1 * sm1 + (1.0 - b1) * gs;
            sm2 = b2 * sm2 + (1.0 - b2) * gs * gs;
            let (mh, vh) = (sm1 / (1.0 - b1.powi(t)), sm2 / (1.0 - b2.powi(t)));
            logsd = (logsd + lr * mh / (vh.sqrt() + 1e-12)).clamp(0.01f64.ln(), 0.5f64.ln());
        }
        let batch_best = js.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "update {u:>3} charged {:>5} best J {:.4} batch best {batch_best:.4} sd {:.4} active {active}/{}",
            ev.log.charged,
            ev.log.best_j(),
            logsd.exp(),
            batch * epochs
        );
    }
    unreachable!()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let method: String = arg("--method", "cma".to_string());
    let seed: u64 = arg("--seed", 1);
    let init: String = arg("--init", "I1".to_string());
    let cap: usize = arg("--cap", 1200);
    let sigma0: f64 = arg("--sigma0", 0.15);
    let lr: f64 = arg("--lr", 0.02);
    let rf: f64 = arg("--rf", 6.0);
    let out: String = arg("--out", format!("out/stage6/{method}_{seed}"));
    let mut base = PortSetup::standard(G::Bend, 1);
    base.extra_time = 6000.0;
    let bx = DesignBox::for_bend(&base, 80, 8);
    let filt = ConicFilter::new(rf);
    let rep = Rep::Grid(20);
    let reference = base.run(Run::Reference)?;
    let x0 = rep.encode(&named_mask(&bx, &init), bx.n);
    let log = RunLog::new(&out, bx.n, cap, &[150, 300, 600, 1200])?;
    let scorer = Scorer::new(base.clone(), bx.clone(), reference);
    let mut ev = Eval { rep, filt: &filt, bx: &bx, scorer, log, candidates: 0 };
    let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x5EED);
    println!("method {method}, seed {seed}, init {init}, cap {cap}, rf {rf}");
    let extra = match method.as_str() {
        "cma" => cma(&mut ev, &x0, sigma0, &mut rng)?,
        "ppo" => ppo(&mut ev, &x0, lr, &mut rng)?,
        m => return Err(format!("unknown method {m}").into()),
    };
    let extra = format!(
        ",\n  \"method\": \"{method}\",\n  \"seed\": {seed},\n  \"init\": \"{init}\",\n  \"rf\": {rf},\n  \"sigma0\": {sigma0},\n  \"lr\": {lr},\n  \"candidates\": {}{extra}",
        ev.candidates
    );
    ev.log.finish(&extra)?;
    let best = ev.log.best.as_ref().map(|b| b.x.clone()).unwrap_or_default();
    write_values(&format!("{out}/best_x.txt"), &best)?;
    println!("done: best J {:.4}, charged {}, candidates {}, cache hits {}", ev.log.best_j(), ev.log.charged, ev.candidates, ev.log.cache_hits);
    Ok(())
}
