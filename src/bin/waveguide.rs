//! Proof run: a silicon strip (eps_r = 12.11) in air (eps_r = 1.0) across the
//! middle of the grid. A pulse is launched inside the strip. The same pulse is
//! also launched in an all-air grid as a control. Then the same silicon run is
//! repeated on the GPU and compared bit-for-bit with the CPU result.
//!
//! `waveguide --check-pml` instead measures the absorbing-boundary reflection in dB.
//!
//! Output: metrics on stdout, field snapshots as PPM files in out/.
//! Exit code 0 only if every check passes. The same checks run in `cargo test`.

use std::fs;
use std::io::Write;

use std::collections::HashMap;

use photon_fdtd::verify::{
    AbsorberCheck, EPS_AIR, EPS_SI, Geometry, ProbeReflection, Pulse, STANDARD_PML, Strip, StripReport, bit_mismatches, courant,
    trapped,
};
use photon_fdtd::{COURANT_MAX_2D, Pml, Yee2d};

fn write_ppm(path: &str, s: &Strip, sim: &Yee2d) {
    let (nx, ny) = (s.nx, s.ny);
    let max = sim.ez.iter().fold(0.0f64, |m, e| m.max(e.abs())).max(1e-300);
    let mut buf = Vec::with_capacity(nx * ny * 3 + 32);
    write!(buf, "P6\n{nx} {ny}\n255\n").unwrap();
    for j in (0..ny).rev() {
        for i in 0..nx {
            let k = j * nx + i;
            let v = sim.ez[k] / max;
            let px = (v.abs().sqrt() * 255.0) as u8;
            let bg = if sim.eps_r[k] > 1.0 { 45u8 } else { 0 };
            let edge = i.min(nx - 1 - i).min(j).min(ny - 1 - j) < s.pml.cells;
            let bg = if edge { bg.max(20) } else { bg };
            let rgb = if v >= 0.0 { [px.max(bg), bg, bg] } else { [bg, bg, px.max(bg)] };
            buf.extend_from_slice(&rgb);
        }
    }
    fs::write(path, buf).unwrap();
}

fn report(name: &str, s: &Strip, o: &StripReport) {
    let (b0, b1) = s.band();
    println!("{name}:");
    println!("  energy left in grid / peak energy      = {:.2e}", o.energy_retained);
    println!("  final energy inside band rows {b0}..{b1}   = {:.4} of grid energy", o.band_fraction);
    println!("  probe x={}: time-integrated Ez^2 in band = {:.4} of column total", s.probe_x, o.probe_band_fraction);
    println!("  probe x={}: RMS transverse width        = {:.1} cells", s.probe_x, o.probe_rms_width);
    if s.eps_core > 1.0 {
        let x = o.pulse_x(s);
        let vg = (x - s.src_x) as f64 / (s.t_end - s.t0);
        println!("  pulse centre on axis at t_end          = x {x} (group velocity ~{vg:.3} c, n_g ~{:.2})", 1.0 / vg);
    }
}

fn waveguide() -> bool {
    fs::create_dir_all("out").unwrap();
    let si = Strip::standard(EPS_SI);
    let air = Strip::standard(EPS_AIR);
    let (j0, j1) = si.core();
    println!(
        "grid {} x {}, CPML {:?}, strip rows {j0}..{j1} (width {}), lambda0 {} cells, pulse t0 {} tau {}",
        si.nx, si.ny, si.pml, si.core_w, si.lambda0, si.t0, si.tau
    );
    println!("Courant S = {:.6} (limit 1/sqrt(2) = {COURANT_MAX_2D:.6}), steps = {}", courant(), si.steps());
    let cfl_ok = match Yee2d::new(8, 8, vec![1.0; 64], Pml::new(0), 1.001 * COURANT_MAX_2D) {
        Err(e) => {
            println!("CFL guard: S = 1.001/sqrt(2) rejected -> {e}");
            true
        }
        Ok(_) => {
            println!("CFL guard FAILED: an unstable Courant number was accepted");
            false
        }
    };
    println!();

    let t = std::time::Instant::now();
    let si_r = si.run_cpu();
    println!("[cpu silicon run {:.2} s]", t.elapsed().as_secs_f64());
    let air_r = air.run_cpu();
    report("silicon strip eps_r = 12.11 in air", &si, &si_r);
    report("control: all air", &air, &air_r);
    write_ppm("out/waveguide_si.ppm", &si, &si_r.sim);
    write_ppm("out/waveguide_air.ppm", &air, &air_r.sim);
    let ok_trap = trapped(&si_r, &air_r);
    println!("\nconfinement verdict: {}", if ok_trap { "TRAPPED in the silicon lane" } else { "NOT confined" });

    let steps = si.steps();
    let ez_gpu = si.run_gpu_ez(steps).expect("CUDA run");
    let ez_serial = si.run_serial_ez(steps);
    let gpu_mis = bit_mismatches(&ez_gpu, &ez_serial);
    let par_mis = bit_mismatches(&si_r.sim.ez, &ez_serial);
    println!("\nGPU vs CPU serial after {steps} steps: {gpu_mis} of {} Ez values differ in bits", ez_gpu.len());
    println!("CPU parallel vs CPU serial: {par_mis} Ez values differ in bits");
    println!("snapshots: out/waveguide_si.ppm out/waveguide_air.ppm");
    cfl_ok && ok_trap && gpu_mis == 0 && par_mis == 0
}

type Measured = (AbsorberCheck, Vec<ProbeReflection>);

/// Reflection of every case. One reference run per (geometry, pulse) covers
/// the probe cells of every PML thickness.
fn reflections(cases: &[AbsorberCheck]) -> Vec<Measured> {
    let mut refs: HashMap<(Geometry, bool, usize, usize), Vec<f64>> = HashMap::new();
    for geometry in [Geometry::Square, Geometry::Grazing] {
        for pulse in [Pulse::GaussSine, Pulse::GaussDerivative] {
            let mine: Vec<&AbsorberCheck> = cases.iter().filter(|c| c.pulse == pulse && c.geometry == geometry).collect();
            let Some(first) = mine.first() else { continue };
            assert!(mine.iter().all(|c| c.reference_is_clean()), "reference reached by its own walls");
            assert!(mine.iter().all(|c| c.reference_frame() == first.reference_frame() && c.steps == first.steps));
            let mut cells: Vec<(usize, usize)> = mine.iter().flat_map(|c| c.reference_cells()).collect();
            cells.sort();
            cells.dedup();
            let traces = first.reference_at(&cells).expect("CUDA reference");
            for (cell, t) in cells.into_iter().zip(traces) {
                refs.insert((geometry, pulse == Pulse::GaussSine, cell.0, cell.1), t);
            }
        }
    }
    cases
        .iter()
        .map(|c| {
            let key = |&(i, j): &(usize, usize)| (c.geometry, c.pulse == Pulse::GaussSine, i, j);
            let r: Vec<Vec<f64>> = c.reference_cells().iter().map(|p| refs[&key(p)].clone()).collect();
            (c.clone(), c.measure(&r).expect("CUDA test run"))
        })
        .collect()
}

fn check_pml(sweep: bool) -> bool {
    let sq = AbsorberCheck::standard(Pml::new(STANDARD_PML));
    let gr = AbsorberCheck::grazing(Pml::new(STANDARD_PML));
    let zd = sq.clone().zero_dc();
    println!("square:  test 512 x 512, reference 2048 x 2048, source (256,256), {} steps", sq.steps);
    println!("grazing: test 1024 x 256, reference {:?}, source (150,128), {} steps", gr.reference_frame().1, gr.steps);
    println!("probes {} cells inside the PML interface", sq.gap);
    println!("pulse A: Gaussian sine, lambda0 {} cells, t0 {} tau {}", sq.lambda0, sq.t0, sq.tau);
    println!("pulse B: Gaussian derivative (no DC), t0 {:.2} tau {:.2}; DFT metric at 1/{} cells^-1", zd.t0, zd.tau, zd.lambda0);
    let mut pmls = Vec::new();
    if sweep {
        for cells in [8, 10, 12, 16] {
            for kappa_max in [1.0, 3.0, 5.0, 7.0, 11.0] {
                for alpha_max in [0.0, 0.05, 0.1, 0.2] {
                    for m in [3.0, 4.0] {
                        pmls.push(Pml { kappa_max, alpha_max, m, sigma_max: 0.8 * (m + 1.0), ..Pml::new(cells) });
                    }
                }
            }
        }
    } else {
        // N = 0 is the control: a bare PEC wall must reflect about 0 dB.
        for cells in [0, 4, 6, 7, 8, 9, 10, 12, 14, 16, 20] {
            pmls.push(Pml::new(cells));
        }
    }
    let mut cases = Vec::new();
    for &p in &pmls {
        for c in [AbsorberCheck::standard(p), AbsorberCheck::grazing(p)] {
            cases.push(c.clone());
            cases.push(c.zero_dc());
        }
    }
    let results = reflections(&cases);
    println!(
        "{:>3} {:>3} {:>5} {:>5} {:>5} | {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}   (dB, time max / DFT at f0)",
        "N", "m", "kmax", "amax", "pulse", "norm t", "norm f0", "corn t", "corn f0", "graz t", "graz f0"
    );
    let mut ok = true;
    for &p in &pmls {
        for pulse in [Pulse::GaussSine, Pulse::GaussDerivative] {
            let pick = |g: Geometry| {
                results.iter().find(|(c, _)| c.pml == p && c.pulse == pulse && c.geometry == g).map(|(_, r)| r).unwrap()
            };
            let (s, g) = (pick(Geometry::Square), pick(Geometry::Grazing));
            println!(
                "{:>3} {:>3} {:>5} {:>5} {:>5} | {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
                p.cells,
                p.m,
                p.kappa_max,
                p.alpha_max,
                if pulse == Pulse::GaussSine { "A" } else { "B" },
                s[0].r_db,
                s[0].r_f0_db,
                s[1].r_db,
                s[1].r_f0_db,
                g[0].r_db,
                g[0].r_f0_db
            );
            ok &= s.iter().chain(g).all(|x| x.r_db.is_finite());
        }
    }
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ok = if args.iter().any(|a| a == "--check-pml") {
        check_pml(args.iter().any(|a| a == "--sweep"))
    } else {
        waveguide()
    };
    std::process::exit(if ok { 0 } else { 1 });
}
