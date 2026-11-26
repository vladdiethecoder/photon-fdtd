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

use photon_fdtd::verify::{AbsorberCheck, EPS_AIR, EPS_SI, Strip, StripReport, bit_mismatches, courant, trapped};
use photon_fdtd::{COURANT_MAX_2D, Yee2d};

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
            let edge = i.min(nx - 1 - i).min(j).min(ny - 1 - j) < s.sponge;
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
        "grid {} x {}, sponge {}, strip rows {j0}..{j1} (width {}), lambda0 {} cells, pulse t0 {} tau {}",
        si.nx, si.ny, si.sponge, si.core_w, si.lambda0, si.t0, si.tau
    );
    println!("Courant S = {:.6} (limit 1/sqrt(2) = {COURANT_MAX_2D:.6}), steps = {}", courant(), si.steps());
    let cfl_ok = match Yee2d::new(8, 8, vec![1.0; 64], 0, 1.001 * COURANT_MAX_2D) {
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

fn check_pml() -> bool {
    let base = AbsorberCheck::standard(40);
    println!(
        "absorber reflection: test {0} x {0}, reference {1} x {1}, {2} steps, point source at centre,",
        base.n, base.n_ref, base.steps
    );
    println!(
        "Gaussian-sine lambda0 {} cells, t0 {} tau {}; probes {} cells inside the absorber interface",
        base.lambda0, base.t0, base.tau, base.gap
    );
    let mut ok = true;
    for sponge in [10, 20, 40, 80] {
        let c = AbsorberCheck { sponge, ..base.clone() };
        // Probe cells move with the sponge thickness, so each case gets its own reference run.
        let clean = c.reference_is_clean();
        ok &= clean;
        println!("sponge {sponge} cells: reference exactly independent of its own walls (1 cell/step bound): {clean}");
        for p in c.measure(&c.reference()) {
            println!(
                "  {:<18} cell ({:>3},{:>3})  incident {:.3e}  max|diff| {:.3e}  R = {:>6.1} dB (time max)  {:>6.1} dB (at f0)",
                p.name, p.i, p.j, p.incident_peak, p.max_diff, p.r_db, p.r_f0_db
            );
            ok &= p.r_db.is_finite();
        }
    }
    ok
}

fn main() {
    let ok = if std::env::args().any(|a| a == "--check-pml") { check_pml() } else { waveguide() };
    std::process::exit(if ok { 0 } else { 1 });
}
