//! Stage 4 measurement: slab mode (analytic vs Yee grid vs measured), straight
//! guide calibration (energy balance, CPML modal reflection) and the sharp
//! 90-degree bend (insertion and return loss).
//!
//! Usage: sparams [--scale 1]

use photon_fdtd::sparams::{Geometry, PortSetup, Run, analytic_overlap, db, discrete_mode, measured_beta, slab_even_mode};
use photon_fdtd::verify::{EPS_SI, courant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let scale: usize = args.iter().position(|a| a == "--scale").map(|p| args[p + 1].parse().unwrap()).unwrap_or(1);
    let mut st = PortSetup::standard(Geometry::Straight, scale);
    let opt = |flag: &str| args.iter().position(|a| a == flag).map(|p| args[p + 1].clone());
    if let Some(v) = opt("--port1") {
        st.port1_x = v.parse().unwrap();
    }
    if let Some(v) = opt("--pml") {
        st.pml.cells = v.parse().unwrap();
    }
    if let Some(v) = opt("--alpha") {
        st.pml.alpha_max = v.parse().unwrap();
    }
    let only_straight = args.iter().any(|a| a == "--straight-only");
    let n1 = EPS_SI.sqrt();
    let eps = st.eps_column();
    println!(
        "scale {scale}: slab {} cells, band lambda0 {:.0}..{:.0} cells, {} freqs, CPML {} cells",
        st.core_w,
        1.0 / st.f_max,
        1.0 / st.f_min,
        st.nf,
        st.pml.cells
    );
    println!("guided modes at f_max: {}", discrete_mode(&eps, st.f_max, courant()).guided);
    let t = std::time::Instant::now();
    let reference = st.run(Run::Reference)?;
    let dut = st.run(Run::Dut)?;
    println!(
        "straight: grid {:?}, reference {:?} (clean: {}), {} steps, tail energy ref {:.1e} dut {:.1e}, {:.1} s",
        st.dut_grid(),
        st.reference_grid(),
        st.reference_is_clean(),
        dut.steps,
        reference.tail_energy,
        dut.tail_energy,
        t.elapsed().as_secs_f64()
    );
    println!(
        "{:>9} {:>9} {:>11} {:>11} {:>11} {:>11} {:>10} {:>12}",
        "lambda0", "neff_an", "beta_an", "beta_grid", "beta_meas", "meas/grid-1", "an/grid-1", "1-overlap"
    );
    for (k, &f) in reference.freqs.iter().enumerate() {
        let an = slab_even_mode(st.core_w as f64, n1, 1.0, f);
        let gr = discrete_mode(&eps, f, courant());
        let me = measured_beta(&reference.axis[k]);
        println!(
            "{:>9.1} {:>9.5} {:>11.7} {:>11.7} {:>11.7} {:>11.2e} {:>10.2e} {:>12.2e}",
            1.0 / f,
            an.neff,
            an.beta,
            gr.beta,
            me,
            me / gr.beta - 1.0,
            an.beta / gr.beta - 1.0,
            1.0 - analytic_overlap(&st, f)
        );
    }
    let s = st.sparams(&reference, &dut);
    println!(
        "{:>9} | {:>10} {:>10} {:>11} | {:>10} {:>10} {:>11} | {:>8}   (flux | mode | incident backward dB)",
        "lambda0", "S11^2 dB", "S21^2", "sum-1", "S11^2 dB", "S21^2", "sum-1", "inc bw"
    );
    for k in 0..s.freqs.len() {
        println!(
            "{:>9.1} | {:>10.1} {:>10.6} {:>11.2e} | {:>10.1} {:>10.6} {:>11.2e} | {:>8.1}",
            1.0 / s.freqs[k],
            db(s.s11[k]),
            s.s21[k],
            s.s11[k] + s.s21[k] - 1.0,
            db(s.s11_mode[k]),
            s.s21_mode[k],
            s.s11_mode[k] + s.s21_mode[k] - 1.0,
            db(s.inc_backward[k])
        );
    }
    if only_straight {
        return Ok(());
    }
    let mut bd = PortSetup::standard(Geometry::Bend, scale);
    bd.pml = st.pml;
    bd.port1_x = st.port1_x;
    let t = std::time::Instant::now();
    let reference = bd.run(Run::Reference)?;
    let dut = bd.run(Run::Dut)?;
    println!(
        "bend: grid {:?}, reference {:?} (clean: {}), {} steps, tail energy ref {:.1e} dut {:.1e}, {:.1} s",
        bd.dut_grid(),
        bd.reference_grid(),
        bd.reference_is_clean(),
        dut.steps,
        reference.tail_energy,
        dut.tail_energy,
        t.elapsed().as_secs_f64()
    );
    let s = bd.sparams(&reference, &dut);
    println!("{:>9} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}   (flux | mode: IL dB, RL dB, 1-S11^2-S21^2)", "lambda0", "IL", "RL", "loss", "IL", "RL", "loss");
    for k in 0..s.freqs.len() {
        println!(
            "{:>9.1} | {:>8.3} {:>8.2} {:>8.4} | {:>8.3} {:>8.2} {:>8.4}",
            1.0 / s.freqs[k],
            -db(s.s21[k]),
            -db(s.s11[k]),
            1.0 - s.s11[k] - s.s21[k],
            -db(s.s21_mode[k]),
            -db(s.s11_mode[k]),
            1.0 - s.s11_mode[k] - s.s21_mode[k]
        );
    }
    Ok(())
}
