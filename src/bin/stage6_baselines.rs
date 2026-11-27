//! Stage 6 phase 1 baselines (docs/stage6-protocol.md, section 6) and the
//! eta calibration of section 7. Spectra are computed for every baseline,
//! also for baselines that fail a geometry gate, so that none is hidden.
//!
//! Usage: stage6_baselines [--out out/stage6/baselines] [--nf 15] [--scale 1]

use std::fmt::Write as _;
use std::fs;

use photon_fdtd::design::{ConicFilter, DesignBox};
use photon_fdtd::robust::{Geometry, Spectrum, bias, eta_for_area, geometry, robust_j, spectrum, variants};
use photon_fdtd::sparams::{Geometry as G, PortSetup, Run};
use photon_fdtd::stage6::{arg, named_mask, write_mask};

fn geo_json(g: &Geometry) -> String {
    format!(
        "{{\"ports_connected\": {}, \"floating_silicon\": {}, \"enclosed_air\": {}, \"single_cell_islands\": {}, \"width_violations\": {}, \"gap_violations\": {}, \"min_width\": {}, \"min_gap\": {}, \"feasible\": {}}}",
        g.ports_connected, g.floating_silicon, g.enclosed_air, g.single_cell_islands, g.width_violations, g.gap_violations, g.min_width, g.min_gap, g.feasible()
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out: String = arg("--out", "out/stage6/baselines".to_string());
    let nf: usize = arg("--nf", 15);
    fs::create_dir_all(&out)?;
    let mut base = PortSetup::standard(G::Bend, 1);
    base.extra_time = 6000.0;
    base.nf = nf;
    let bx = DesignBox::for_bend(&base, 80, 8);
    let t0 = std::time::Instant::now();
    let reference = base.run(Run::Reference)?;

    // eta calibration on the start design (section 7).
    let arc = bx.circular_arc();
    let (ae, ad) = (bias(&bx, &arc, 1, false), bias(&bx, &arc, 1, true));
    let area = |s: &[f64]| s.iter().filter(|&&v| v >= 0.5).count();
    let mut cal = String::from("{\n");
    for rf in [6.0, 8.0] {
        let f = ConicFilter::new(rf).apply(&bx, &arc);
        let (eta_e, eta_d) = (eta_for_area(&f, area(&ae)), eta_for_area(&f, area(&ad)));
        let a05 = f.iter().filter(|&&v| v >= 0.5).count();
        println!("calibration r_f {rf}: arc area {} (filtered at 0.5: {a05}), eroded {}, dilated {}, eta_e {eta_e:.6}, eta_d {eta_d:.6}", area(&arc), area(&ae), area(&ad));
        writeln!(cal, "  \"rf{rf}\": {{\"eta_e\": {eta_e:.9}, \"eta_d\": {eta_d:.9}, \"arc_area\": {}, \"eroded_area\": {}, \"dilated_area\": {}, \"filtered_area_at_0.5\": {a05}}},", area(&arc), area(&ae), area(&ad))?;
    }
    cal.pop();
    cal.pop();
    cal.push_str("\n}\n");
    fs::write(format!("{out}/eta_calibration.json"), cal)?;

    let mut names = vec!["arc".to_string()];
    names.extend((1..=10).map(|q| format!("euler:{:.1}", q as f64 / 10.0)));
    names.extend((0..=8).map(|q| format!("bezier:{:.2}", 0.30 + 0.05 * q as f64)));
    let mut csv = String::from("design,variant,worst_il_db,worst_rl_db,silicon_cells,ports_connected,floating_silicon,enclosed_air,islands,width_violations,gap_violations,min_width,min_gap\n");
    let mut spectra_csv = String::from("design,variant,lambda0,il_db,rl_db\n");
    let mut json = String::from("[\n");
    let mut best: Option<(String, f64)> = None;
    for name in &names {
        let nominal = named_mask(&bx, name);
        write_mask(&format!("{out}/{}.mask", name.replace(':', "_")), bx.n, &nominal)?;
        let vs = variants(&bx, &nominal, 1);
        let mut sps: Vec<Spectrum> = Vec::new();
        let mut geos = Vec::new();
        for (v, label) in vs.iter().zip(["eroded", "nominal", "dilated"]) {
            let g = geometry(&bx, v, 1);
            let sp = spectrum(&base, &bx, &reference, v)?;
            writeln!(
                csv,
                "{name},{label},{:.5},{:.3},{},{},{},{},{},{},{},{},{}",
                sp.worst_il(),
                sp.worst_rl(),
                g.silicon_cells,
                g.ports_connected,
                g.floating_silicon,
                g.enclosed_air,
                g.single_cell_islands,
                g.width_violations,
                g.gap_violations,
                g.min_width,
                g.min_gap
            )?;
            for k in 0..sp.lambda0.len() {
                writeln!(spectra_csv, "{name},{label},{:.4},{:.6},{:.4}", sp.lambda0[k], sp.il_db[k], sp.rl_db[k])?;
            }
            geos.push(g);
            sps.push(sp);
        }
        let sp3 = [sps[0].clone(), sps[1].clone(), sps[2].clone()];
        let j = robust_j(&sp3);
        let il = sp3.iter().map(|s| s.worst_il()).fold(0.0, f64::max);
        let rl = sp3.iter().map(|s| s.worst_rl()).fold(f64::INFINITY, f64::min);
        let feasible = geos[1].feasible();
        println!(
            "{name:<12} J {j:.4}  robust IL {il:.4} dB  robust RL {rl:.2} dB  nominal geometry ok {feasible}  (min width {} gap {}; eroded width {}, dilated gap {})",
            geos[1].min_width, geos[1].min_gap, geos[0].min_width, geos[2].min_gap
        );
        writeln!(
            json,
            "  {{\"design\": \"{name}\", \"j\": {j:.6}, \"robust_il_db\": {il:.6}, \"robust_rl_db\": {rl:.4}, \"geometry_ok\": {feasible}, \"variants\": [{}, {}, {}]}},",
            geo_json(&geos[0]),
            geo_json(&geos[1]),
            geo_json(&geos[2])
        )?;
        if feasible && best.as_ref().is_none_or(|b| j < b.1) {
            best = Some((name.clone(), j));
        }
    }
    json.pop();
    json.pop();
    json.push_str("\n]\n");
    fs::write(format!("{out}/baselines.csv"), csv)?;
    fs::write(format!("{out}/spectra.csv"), spectra_csv)?;
    fs::write(format!("{out}/baselines.json"), json)?;
    match &best {
        Some((n, j)) => println!("best tested baseline: {n} (J {j:.4} dB)"),
        None => println!("best tested baseline: none meets the geometry gates"),
    }
    fs::write(format!("{out}/best_baseline.txt"), best.map(|b| format!("{} {:.6}\n", b.0, b.1)).unwrap_or_else(|| "none\n".into()))?;
    println!("{} FDTD runs + 1 reference, {:.0} s", 3 * names.len(), t0.elapsed().as_secs_f64());
    Ok(())
}
