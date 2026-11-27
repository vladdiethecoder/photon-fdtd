//! Stage 5 validation (docs/stage5-protocol.md) of the optimised bend against
//! the best conventional baseline (Bezier k = 0.40) and the circular arc.
//!
//! 1. scale 1, the 15 optimisation frequencies;
//! 2. scale 1, a dense sweep of 61 frequencies over the same band;
//! 3. scale 2: the same physical designs re-rasterised on the finer grid;
//! 4. erosion and dilation of each binary design by 1 cell (scale 1, 15 frequencies);
//! 5. numerical uncertainty U and the protocol's win test and acceptance gates.
//!
//! Usage: stage5_validate RHO_FINAL.txt [--rf 5] [--beta 32] [--n 48] [--bezier-k 0.40] [--out out/validate]

use std::fs;

use photon_fdtd::design::{
    ConicFilter, DesignBox, binary_fraction, cross, densities, disk, feature_violations, morph, project, single_cell_islands,
};
use photon_fdtd::sparams::{Geometry, PortSetup, Run, SParams, db};

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).map(|p| a[p + 1].parse().ok().unwrap()).unwrap_or(default)
}

fn setup(scale: usize, nf: usize) -> PortSetup {
    let mut s = PortSetup::standard(Geometry::Bend, scale);
    s.extra_time = 6000.0 * scale as f64;
    s.nf = nf;
    s
}

fn worst(s: &SParams) -> (f64, f64) {
    (s.s21_mode.iter().map(|v| -db(*v)).fold(0.0, f64::max), s.s11_mode.iter().map(|v| -db(*v)).fold(f64::INFINITY, f64::min))
}

/// Optimised design on the scale-2 box: the 0.5 level set of the scale-1
/// filtered density, bilinearly interpolated at the fine cell centres.
fn upsample(bx1: &DesignBox, filtered: &[f64], bx2: &DesignBox) -> Vec<f64> {
    let n1 = bx1.n as i64;
    let at = |a: i64, b: i64| {
        if (0..n1).contains(&a) && (0..n1).contains(&b) { filtered[(b * n1 + a) as usize] } else { bx1.exterior(a, b) }
    };
    let n2 = bx2.n;
    let mut out = vec![0.0; n2 * n2];
    for bb in 0..n2 {
        for aa in 0..n2 {
            // Fine cell centre in scale-1 cell units, relative to scale-1 cell centres.
            let (x, y) = ((aa as f64 + 0.5) / 2.0 - 0.5, (bb as f64 + 0.5) / 2.0 - 0.5);
            let (a0, b0) = (x.floor() as i64, y.floor() as i64);
            let (fx, fy) = (x - a0 as f64, y - b0 as f64);
            let v = (1.0 - fx) * (1.0 - fy) * at(a0, b0)
                + fx * (1.0 - fy) * at(a0 + 1, b0)
                + (1.0 - fx) * fy * at(a0, b0 + 1)
                + fx * fy * at(a0 + 1, b0 + 1);
            out[bb * n2 + aa] = if v >= 0.5 { 1.0 } else { 0.0 };
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("RHO_FINAL.txt");
    let rf: f64 = arg("--rf", 5.0);
    let beta: f64 = arg("--beta", 32.0);
    let n: usize = arg("--n", 48);
    let k: f64 = arg("--bezier-k", 0.40);
    let bez = format!("bezier k={k:.2}");
    let out: String = arg("--out", "out/validate".to_string());
    fs::create_dir_all(&out)?;
    let rho: Vec<f64> = fs::read_to_string(&path)?.lines().map(|l| l.trim().parse().unwrap()).collect();

    let s15 = setup(1, 15);
    let bx1 = DesignBox::for_bend(&s15, n, 8);
    let filt = ConicFilter::new(rf);
    let filtered = filt.apply(&bx1, &rho);
    let rho_bar = densities(&bx1, &filt, &rho, beta);
    let opt1: Vec<f64> = filtered.iter().map(|&v| if project(v, beta, 0.5) >= 0.5 { 1.0 } else { 0.0 }).collect();
    let designs1 = [
        ("optimised", opt1.clone()),
        (bez.as_str(), bx1.rasterise(&bx1.bezier_centreline(k))),
        ("arc", bx1.circular_arc()),
    ];
    let s2 = setup(2, 15);
    let bx2 = DesignBox::for_bend(&s2, 2 * n, 16);
    let designs2 = [
        ("optimised", upsample(&bx1, &filtered, &bx2)),
        (bez.as_str(), bx2.rasterise(&bx2.bezier_centreline(k))),
        ("arc", bx2.circular_arc()),
    ];

    println!("fabrication (scale 1):");
    for (name, d) in &designs1 {
        let (vo, vc) = feature_violations(&bx1, d, &disk(4));
        println!("  {name:<14} islands {}  L_min violations open {vo} close {vc}", single_cell_islands(&bx1, d));
    }
    println!("  optimised: binary fraction before the threshold {:.4}", binary_fraction(&rho_bar));

    let mut res: Vec<(String, [(f64, f64); 3])> = Vec::new();
    let s61 = setup(1, 61);
    let (r15, r61, r2) = (s15.run(Run::Reference)?, s61.run(Run::Reference)?, s2.run(Run::Reference)?);
    let mut spectra = String::from("design,lambda0,IL_dB,RL_dB\n");
    for ((name, d1), (_, d2)) in designs1.iter().zip(&designs2) {
        let a = s15.clone().with_design(&bx1, d1);
        let w15 = worst(&a.sparams(&r15, &a.run(Run::Dut)?));
        let b = s61.clone().with_design(&bx1, d1);
        let sp61 = b.sparams(&r61, &b.run(Run::Dut)?);
        for k in 0..sp61.freqs.len() {
            spectra += &format!("{name},{:.3},{:.5},{:.3}\n", 1.0 / sp61.freqs[k], -db(sp61.s21_mode[k]), -db(sp61.s11_mode[k]));
        }
        let w61 = worst(&sp61);
        let c = s2.clone().with_design(&bx2, d2);
        let w2 = worst(&c.sparams(&r2, &c.run(Run::Dut)?));
        println!(
            "{name:<14} worst IL / RL: 15 freqs {:.4} / {:.2}  61 freqs {:.4} / {:.2}  scale 2 {:.4} / {:.2} dB",
            w15.0, w15.1, w61.0, w61.1, w2.0, w2.1
        );
        res.push((name.to_string(), [w15, w61, w2]));
    }
    fs::write(format!("{out}/spectra_61.csv"), spectra)?;

    println!("erosion / dilation by 1 cell (scale 1, 15 freqs):");
    for (name, d) in &designs1 {
        let mut line = format!("  {name:<14}");
        for (label, dil) in [("eroded", false), ("dilated", true)] {
            let e = morph(&bx1, d, &cross(), dil);
            let st = s15.clone().with_design(&bx1, &e);
            let w = worst(&st.sparams(&r15, &st.run(Run::Dut)?));
            line += &format!("  {label}: IL {:.4} RL {:.2}", w.0, w.1);
        }
        println!("{line}");
    }

    let (opt, base) = (&res[0].1, &res[1].1);
    let u = (opt[2].0 - opt[0].0).abs().max((base[2].0 - base[0].0).abs());
    let gains = [base[0].0 - opt[0].0, base[1].0 - opt[1].0, base[2].0 - opt[2].0];
    println!("numerical uncertainty U = {u:.4} dB (worst-IL change, scale 1 -> 2)");
    println!("gain over the best baseline: 15 freqs {:.4}, 61 freqs {:.4}, scale 2 {:.4} dB", gains[0], gains[1], gains[2]);
    let win = gains.iter().all(|&g| g >= 0.10 && g > u);
    let gates = opt[0].0 < 0.5 && opt[0].1 > 20.0 && opt[1].0 < 0.5 && opt[1].1 > 20.0;
    println!("acceptance (IL < 0.50 dB and RL > 20 dB on 15 and 61 freqs): {}", if gates { "PASS" } else { "FAIL" });
    println!("optimiser win (gain >= 0.10 dB and > U on 15, 61 freqs and scale 2): {}", if win { "YES" } else { "NO" });
    // Box images: the three designs at scale 1 (white = silicon).
    for (name, d) in &designs1 {
        let mut ppm = format!("P5\n{0} {0}\n255\n", bx1.n).into_bytes();
        for b in (0..bx1.n).rev() {
            for a in 0..bx1.n {
                ppm.push(if d[b * bx1.n + a] >= 0.5 { 255 } else { 0 });
            }
        }
        fs::write(format!("{out}/{}.pgm", name.replace([' ', '='], "_")), ppm)?;
    }
    Ok(())
}
