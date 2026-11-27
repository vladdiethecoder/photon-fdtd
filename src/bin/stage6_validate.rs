//! Stage 6 held-out validation (docs/stage6-protocol.md, sections 4 and 6).
//!
//!   stage6_validate --opt LABEL:REP:RF:XFILE ... --baseline NAME ... --out DIR
//!
//! REP is full or grid20; XFILE holds the representation value. The first
//! --baseline is the best tested baseline of the win test. For each design:
//! the three variants on the training spectrum (15), the held-out spectrum
//! (61) and the held-out grid (scale 2, 15 samples, bias radius 2), with the
//! geometry of every variant at both scales. Optimised designs are re-rasterised
//! at scale 2 from the bilinear interpolation of their scale-1 filtered density;
//! baselines are re-rasterised from their centreline.

use std::fmt::Write as _;
use std::fs;

use photon_fdtd::design::{ConicFilter, DesignBox};
use photon_fdtd::robust::{IL_MAX_DB, RL_MIN_DB, Rep, Spectrum, geometry, spectrum, variants};
use photon_fdtd::sparams::{Geometry as G, PortData, PortSetup, Run};
use photon_fdtd::stage6::{named_mask, read_values, write_mask};

struct Design {
    label: String,
    m1: Vec<f64>,
    m2: Vec<f64>,
}

fn setup(scale: usize, nf: usize) -> PortSetup {
    let mut s = PortSetup::standard(G::Bend, scale);
    s.extra_time = 6000.0 * scale as f64;
    s.nf = nf;
    s
}

/// Bilinear interpolation of the scale-1 filtered density (exterior as padding)
/// at the scale-2 cell centres, thresholded at 0.5.
fn upsample(bx1: &DesignBox, filtered: &[f64], bx2: &DesignBox) -> Vec<f64> {
    let n1 = bx1.n as i64;
    let at = |a: i64, b: i64| if (0..n1).contains(&a) && (0..n1).contains(&b) { filtered[(b * n1 + a) as usize] } else { bx1.exterior(a, b) };
    let n2 = bx2.n;
    (0..n2 * n2)
        .map(|k| {
            let (x, y) = (((k % n2) as f64 + 0.5) / 2.0 - 0.5, ((k / n2) as f64 + 0.5) / 2.0 - 0.5);
            let (a0, b0) = (x.floor() as i64, y.floor() as i64);
            let (fx, fy) = (x - a0 as f64, y - b0 as f64);
            let v = (1.0 - fx) * (1.0 - fy) * at(a0, b0) + fx * (1.0 - fy) * at(a0 + 1, b0) + (1.0 - fx) * fy * at(a0, b0 + 1) + fx * fy * at(a0 + 1, b0 + 1);
            if v >= 0.5 { 1.0 } else { 0.0 }
        })
        .collect()
}

fn worst(sp: &[Spectrum]) -> (f64, f64) {
    (sp.iter().map(|s| s.worst_il()).fold(0.0, f64::max), sp.iter().map(|s| s.worst_rl()).fold(f64::INFINITY, f64::min))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let get_all = |flag: &str| -> Vec<String> { args.windows(2).filter(|w| w[0] == flag).map(|w| w[1].clone()).collect() };
    let out = get_all("--out").pop().unwrap_or_else(|| "out/stage6/validate".into());
    fs::create_dir_all(&out)?;
    let (s15, s61, s2) = (setup(1, 15), setup(1, 61), setup(2, 15));
    let (bx1, bx2) = (DesignBox::for_bend(&s15, 80, 8), DesignBox::for_bend(&s2, 160, 16));
    let mut designs: Vec<Design> = Vec::new();
    for spec in get_all("--opt") {
        let p: Vec<&str> = spec.splitn(4, ':').collect();
        let rep = match p[1] {
            "full" => Rep::Full,
            "grid20" => Rep::Grid(20),
            r => return Err(format!("unknown rep {r}").into()),
        };
        let filt = ConicFilter::new(p[2].parse()?);
        let x = read_values(p[3])?;
        let filtered = filt.apply(&bx1, &rep.latent(&x, bx1.n));
        let m1: Vec<f64> = filtered.iter().map(|&v| if v >= 0.5 { 1.0 } else { 0.0 }).collect();
        designs.push(Design { label: p[0].to_string(), m2: upsample(&bx1, &filtered, &bx2), m1 });
    }
    let baselines = get_all("--baseline");
    for name in &baselines {
        designs.push(Design { label: format!("baseline {name}"), m1: named_mask(&bx1, name), m2: named_mask(&bx2, name) });
    }
    let t0 = std::time::Instant::now();
    let refs: [PortData; 3] = [s15.run(Run::Reference)?, s61.run(Run::Reference)?, s2.run(Run::Reference)?];
    let mut spectra = String::from("design,set,variant,lambda0,il_db,rl_db\n");
    let mut geo_csv = String::from("design,scale,variant,silicon_cells,ports_connected,floating_silicon,enclosed_air,islands,width_violations,gap_violations,min_width,min_gap,feasible\n");
    let mut json = String::from("{\n  \"designs\": [\n");
    let mut results: Vec<(String, [(f64, f64); 3], [[(f64, f64); 3]; 3], bool)> = Vec::new();
    for d in &designs {
        let tag = d.label.replace([' ', ':', '='], "_");
        write_mask(&format!("{out}/{tag}_s1.mask"), bx1.n, &d.m1)?;
        write_mask(&format!("{out}/{tag}_s2.mask"), bx2.n, &d.m2)?;
        let v1 = variants(&bx1, &d.m1, 1);
        let v2 = variants(&bx2, &d.m2, 2);
        let mut geo_ok = false;
        for (scale, vs, bx) in [(1usize, &v1, &bx1), (2, &v2, &bx2)] {
            for (v, label) in vs.iter().zip(["eroded", "nominal", "dilated"]) {
                let g = geometry(bx, v, scale);
                if scale == 1 && label == "nominal" {
                    geo_ok = g.feasible();
                }
                writeln!(
                    geo_csv,
                    "{},{scale},{label},{},{},{},{},{},{},{},{},{},{}",
                    d.label, g.silicon_cells, g.ports_connected, g.floating_silicon, g.enclosed_air, g.single_cell_islands, g.width_violations, g.gap_violations, g.min_width, g.min_gap, g.feasible()
                )?;
            }
        }
        let mut per_set = [(0.0, 0.0); 3];
        let mut per_var = [[(0.0, 0.0); 3]; 3];
        for (si, (set, st, bx, vs, r)) in [("train15", &s15, &bx1, &v1, &refs[0]), ("heldout61", &s61, &bx1, &v1, &refs[1]), ("scale2", &s2, &bx2, &v2, &refs[2])].into_iter().enumerate() {
            let mut sps = Vec::new();
            for (vi, (v, label)) in vs.iter().zip(["eroded", "nominal", "dilated"]).enumerate() {
                let sp = spectrum(st, bx, r, v)?;
                for k in 0..sp.lambda0.len() {
                    writeln!(spectra, "{},{set},{label},{:.4},{:.6},{:.4}", d.label, sp.lambda0[k], sp.il_db[k], sp.rl_db[k])?;
                }
                per_var[si][vi] = (sp.worst_il(), sp.worst_rl());
                sps.push(sp);
            }
            per_set[si] = worst(&sps);
        }
        let gates = geo_ok && per_set.iter().all(|&(il, rl)| il < IL_MAX_DB && rl >= RL_MIN_DB);
        println!(
            "{:<26} robust IL / RL: train {:.4} / {:.2}  held-out 61 {:.4} / {:.2}  scale 2 {:.4} / {:.2}  geometry {}  gates {}",
            d.label, per_set[0].0, per_set[0].1, per_set[1].0, per_set[1].1, per_set[2].0, per_set[2].1, geo_ok, if gates { "PASS" } else { "FAIL" }
        );
        for (vi, label) in ["eroded", "nominal", "dilated"].iter().enumerate() {
            println!(
                "    {label:<8} train {:.4} / {:.2}  61 {:.4} / {:.2}  scale 2 {:.4} / {:.2}  grid change IL {:+.4} dB RL {:+.2} dB",
                per_var[0][vi].0, per_var[0][vi].1, per_var[1][vi].0, per_var[1][vi].1, per_var[2][vi].0, per_var[2][vi].1,
                per_var[2][vi].0 - per_var[0][vi].0, per_var[2][vi].1 - per_var[0][vi].1
            );
        }
        let vjson = |s: usize| -> String {
            (0..3).map(|v| format!("[{:.6}, {:.4}]", per_var[s][v].0, per_var[s][v].1)).collect::<Vec<_>>().join(", ")
        };
        writeln!(
            json,
            "    {{\"label\": \"{}\", \"geometry_ok\": {geo_ok}, \"gates\": {gates}, \"robust\": {{\"train15\": [{:.6}, {:.4}], \"heldout61\": [{:.6}, {:.4}], \"scale2\": [{:.6}, {:.4}]}}, \"variants_il_rl\": {{\"train15\": [{}], \"heldout61\": [{}], \"scale2\": [{}]}}, \"grid_change_il_db\": [{:.6}, {:.6}, {:.6}], \"grid_change_rl_db\": [{:.4}, {:.4}, {:.4}]}},",
            d.label, per_set[0].0, per_set[0].1, per_set[1].0, per_set[1].1, per_set[2].0, per_set[2].1, vjson(0), vjson(1), vjson(2),
            per_var[2][0].0 - per_var[0][0].0, per_var[2][1].0 - per_var[0][1].0, per_var[2][2].0 - per_var[0][2].0,
            per_var[2][0].1 - per_var[0][0].1, per_var[2][1].1 - per_var[0][1].1, per_var[2][2].1 - per_var[0][2].1
        )?;
        results.push((d.label.clone(), per_set, per_var, gates));
    }
    json.pop();
    json.pop();
    json.push_str("\n  ],\n  \"win_tests\": [\n");
    // Win test of section 6: each optimised design against the best tested baseline.
    if let Some(bi) = results.iter().position(|r| r.0.starts_with("baseline")) {
        let b = results[bi].clone();
        let change = |pv: &[[(f64, f64); 3]; 3]| (0..3).map(|v| (pv[2][v].0 - pv[0][v].0).abs()).fold(0.0, f64::max);
        for r in results.iter().filter(|r| !r.0.starts_with("baseline")) {
            let gains = [b.1[0].0 - r.1[0].0, b.1[1].0 - r.1[1].0, b.1[2].0 - r.1[2].0];
            let gc = change(&r.2).max(change(&b.2));
            let win = r.3 && gains.iter().all(|&g| g >= 0.10 && g > gc);
            println!(
                "win test {} vs {}: gains {:.4} / {:.4} / {:.4} dB (train / 61 / scale 2), largest |grid change| {gc:.4} dB, gates {} -> {}",
                r.0, b.0, gains[0], gains[1], gains[2], r.3, if win { "WIN" } else { "NO WIN" }
            );
            writeln!(json, "    {{\"design\": \"{}\", \"baseline\": \"{}\", \"gains_db\": [{:.6}, {:.6}, {:.6}], \"largest_abs_grid_change_db\": {gc:.6}, \"gates\": {}, \"win\": {win}}},", r.0, b.0, gains[0], gains[1], gains[2], r.3)?;
        }
    }
    if json.ends_with(",\n") {
        json.pop();
        json.pop();
        json.push('\n');
    }
    json.push_str("  ]\n}\n");
    fs::write(format!("{out}/spectra.csv"), spectra)?;
    fs::write(format!("{out}/geometry.csv"), geo_csv)?;
    fs::write(format!("{out}/validate.json"), json)?;
    println!("{:.0} s", t0.elapsed().as_secs_f64());
    Ok(())
}
