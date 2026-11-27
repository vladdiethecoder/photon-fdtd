//! Verification suite: `cargo test --release`.
//!
//! Every test calls the same scenario code as the `waveguide` binary
//! (src/verify.rs). GPU tests need CUDA device 0 (an sm_120 card).

use photon_fdtd::gpu::GpuYee;
use photon_fdtd::verify::{AbsorberCheck, EPS_AIR, EPS_SI, STANDARD_PML, Strip, bit_mismatches, courant, trapped};
use photon_fdtd::{COURANT_MAX_2D, Pml, Yee2d, check_setup, gaussian_sine};

#[test]
fn cfl_guard_rejects_courant_above_2d_limit() {
    assert!(check_setup(8, 8, COURANT_MAX_2D).is_ok());
    assert!(check_setup(8, 8, 1.001 * COURANT_MAX_2D).is_err());
    assert!(Yee2d::new(8, 8, vec![1.0; 64], Pml::new(0), 1.001 * COURANT_MAX_2D).is_err());
    assert!(GpuYee::new(8, 8, Pml::new(0), 1.001 * COURANT_MAX_2D, &|_, _| 1.0).is_err());
}

/// Confinement in silicon vs an all-air control, then GPU vs CPU bit parity
/// on the same silicon run. Non-square grids catch swapped nx / ny strides.
fn confinement_and_parity(nx: usize, ny: usize) {
    let si = Strip::standard(EPS_SI).with_grid(nx, ny);
    let air = Strip::standard(EPS_AIR).with_grid(nx, ny);
    let (si_r, air_r) = (si.run_cpu(), air.run_cpu());
    println!(
        "{nx}x{ny}: si band {:.4} probe {:.4} | air band {:.4} probe {:.4}",
        si_r.band_fraction, si_r.probe_band_fraction, air_r.band_fraction, air_r.probe_band_fraction
    );
    assert!(trapped(&si_r, &air_r), "pulse not confined to the silicon strip");
    let ez_gpu = si.run_gpu_ez(si.steps()).expect("CUDA run");
    assert_eq!(bit_mismatches(&ez_gpu, &si_r.sim.ez), 0, "GPU differs from CPU");
}

#[test]
fn confinement_and_parity_1200x600() {
    confinement_and_parity(1200, 600);
}

#[test]
fn confinement_and_parity_600x1200() {
    confinement_and_parity(600, 1200);
}

#[test]
fn bit_parity_serial_parallel_gpu_512x256() {
    let s = Strip::standard(EPS_SI).with_grid(512, 256);
    let serial = s.run_serial_ez(500);
    assert!(serial.iter().any(|&e| e != 0.0));
    assert_eq!(bit_mismatches(&s.run_parallel_ez(500), &serial), 0, "rayon differs from serial");
    assert_eq!(bit_mismatches(&s.run_gpu_ez(500).expect("CUDA run"), &serial), 0, "GPU differs from serial");
}

/// Corner-heavy CPML parity: point source near the low corner of a
/// non-square grid, so the x strips, the y strips and their overlap all carry
/// field. Serial CPU, rayon CPU and GPU must agree bit for bit.
#[test]
fn bit_parity_cpml_corner_non_square_300x170() {
    let (nx, ny, steps) = (300usize, 170usize, 800usize);
    let mut serial = Yee2d::new(nx, ny, vec![EPS_AIR; nx * ny], Pml::new(STANDARD_PML), courant()).unwrap();
    let mut par = Yee2d::new(nx, ny, vec![EPS_AIR; nx * ny], Pml::new(STANDARD_PML), courant()).unwrap();
    let mut gpu = GpuYee::from_cpu(&serial).expect("CUDA init");
    let (si, sj) = (20usize, 18usize);
    for n in 0..steps {
        let a = gaussian_sine((n + 1) as f64 * courant(), 1.0 / 20.0, 60.0, 20.0);
        serial.step_serial();
        serial.add_source_line(si, sj, sj + 1, a);
        par.step_parallel();
        par.add_source_line(si, sj, sj + 1, a);
        gpu.step().unwrap();
        gpu.add_source_line(si, sj, sj + 1, a).unwrap();
    }
    assert!(serial.psi_ezx.iter().any(|&p| p != 0.0) && serial.psi_ezy.iter().any(|&p| p != 0.0));
    assert_eq!(bit_mismatches(&par.ez, &serial.ez), 0, "rayon differs from serial");
    for (a, b) in [(&par.psi_ezx, &serial.psi_ezx), (&par.psi_ezy, &serial.psi_ezy), (&par.psi_hxy, &serial.psi_hxy), (&par.psi_hyx, &serial.psi_hyx)] {
        assert_eq!(bit_mismatches(a, b), 0, "rayon psi differs from serial");
    }
    assert_eq!(bit_mismatches(&gpu.download_ez().unwrap(), &serial.ez), 0, "GPU differs from serial");
}

/// CPML reflection gates: <= -60 dB at normal incidence and <= -50 dB at the
/// corner, for both source pulses, at the standard thickness.
#[test]
fn absorber_reflection_db() {
    for c in [AbsorberCheck::standard(Pml::new(STANDARD_PML)), AbsorberCheck::standard(Pml::new(STANDARD_PML)).zero_dc()] {
        assert!(c.reference_is_clean());
        let r = c.measure(&c.reference().unwrap()).unwrap();
        for p in &r {
            println!("{:?} N={}: {}: {:.1} dB time max, {:.1} dB at f0", c.pulse, c.pml.cells, p.name, p.r_db, p.r_f0_db);
        }
        assert!(r[0].r_db <= -60.0, "{:?} normal-incidence reflection {:.1} dB", c.pulse, r[0].r_db);
        assert!(r[1].r_db <= -50.0, "{:?} corner reflection {:.1} dB", c.pulse, r[1].r_db);
    }
}

/// Grazing incidence (about 81 degrees from the wall normal) is the hardest
/// case for a PML. Gate: <= -50 dB for both pulses at the standard thickness.
#[test]
fn absorber_reflection_grazing_db() {
    for c in [AbsorberCheck::grazing(Pml::new(STANDARD_PML)), AbsorberCheck::grazing(Pml::new(STANDARD_PML)).zero_dc()] {
        assert!(c.reference_is_clean());
        let r = c.measure(&c.reference().unwrap()).unwrap();
        println!("{:?} N={}: {}: {:.1} dB time max, {:.1} dB at f0", c.pulse, c.pml.cells, r[0].name, r[0].r_db, r[0].r_f0_db);
        assert!(r[0].r_db <= -50.0, "{:?} grazing reflection {:.1} dB", c.pulse, r[0].r_db);
    }
}

/// Control for the metric: with no PML (bare PEC walls) the same measurement
/// must report a strong reflection, so a low dB value cannot come from a
/// broken probe or reference.
#[test]
fn absorber_metric_detects_bare_pec() {
    let c = AbsorberCheck::standard(Pml::new(0));
    assert!(c.reference_is_clean());
    let r = c.measure(&c.reference().unwrap()).unwrap();
    println!("PEC control: normal {:.1} dB, corner {:.1} dB", r[0].r_db, r[1].r_db);
    assert!(r[0].r_db > -6.0 && r[1].r_db > -6.0);
}

/// 16384 x 20000 grid (3.28e8 cells, 2.62 GB per f64 array, 10.5 GB of VRAM):
/// byte offsets of updated cells pass 2^31 (16384 x 16385 would not: only its
/// last row, a PEC row that is never updated, lies past 2^31 bytes).
/// Impulses near both corners must evolve bitwise like the same impulses on a
/// 64 x 64 CPU grid, whose coefficients match near the corners because the
/// CPML profiles and strips depend only on the distance to the nearest walls.
#[test]
fn large_grid_16384x20000_gpu_corners_match_cpu() {
    let (nx, ny, steps, n, p) = (16384usize, 20000usize, 8usize, 64usize, 25usize);
    let pml = Pml::new(16);
    let lo = (12usize, 12usize);
    let hi_big = (nx - 13, ny - 13);
    let hi_small = (n - 13, n - 13);
    let k_hi = hi_big.1 * nx + hi_big.0;
    assert!(k_hi * 8 > i32::MAX as usize, "test must cross the 2^31 byte offset");

    let mut gpu = GpuYee::new(nx, ny, pml, courant(), &|_, _| 1.0).expect("16384 x 20000 GPU grid");
    gpu.set_ez(lo.1 * nx + lo.0, 1.0).unwrap();
    gpu.set_ez(k_hi, 1.0).unwrap();
    let mut cpu = Yee2d::new(n, n, vec![1.0; n * n], pml, courant()).unwrap();
    cpu.ez[lo.1 * n + lo.0] = 1.0;
    cpu.ez[hi_small.1 * n + hi_small.0] = 1.0;
    for _ in 0..steps {
        gpu.step().unwrap();
        cpu.step_serial();
    }
    let mut nonzero = 0;
    for dj in 0..p {
        let g_lo = gpu.download_ez_range(dj * nx..dj * nx + p).unwrap();
        let rb = (ny - p + dj) * nx + (nx - p);
        let g_hi = gpu.download_ez_range(rb..rb + p).unwrap();
        let c_lo = &cpu.ez[dj * n..dj * n + p];
        let rs = (n - p + dj) * n + (n - p);
        let c_hi = &cpu.ez[rs..rs + p];
        assert_eq!(bit_mismatches(&g_lo, c_lo), 0, "low corner row {dj}");
        assert_eq!(bit_mismatches(&g_hi, c_hi), 0, "high corner row {dj}");
        nonzero += g_lo.iter().chain(&g_hi).filter(|&&e| e != 0.0).count();
    }
    // Everything the two impulses touched lies inside the compared patches.
    let cpu_nonzero = cpu.ez.iter().filter(|&&e| e != 0.0).count();
    println!("nonzero cells compared: {nonzero} (CPU grid total {cpu_nonzero})");
    assert!(nonzero > 0);
    assert_eq!(nonzero, cpu_nonzero);
}

/// ny = 70000 > 65535, the CUDA gridDim.y limit: rows past the first 65535
/// are reached only by the grid-stride loop in the kernels.
#[test]
fn tall_grid_past_cuda_grid_y_limit_matches_cpu() {
    let s = Strip { src_x: 20, probe_x: 40, pml: Pml::new(8), ..Strip::standard(EPS_SI) }.with_grid(64, 70000);
    let (nx, ny) = (s.nx, s.ny);
    // Second source far past row 65535 so those rows carry real field values.
    let mut cpu = s.build();
    let mut gpu = GpuYee::from_cpu(&cpu).expect("CUDA init");
    let (j0, j1) = s.core();
    let (k0, k1) = (ny - 100, ny - 100 + s.core_w);
    for n in 0..300 {
        let a = s.source_amp(n);
        cpu.step_serial();
        cpu.add_source_line(s.src_x, j0, j1, a);
        cpu.add_source_line(s.src_x, k0, k1, a);
        gpu.step().unwrap();
        gpu.add_source_line(s.src_x, j0, j1, a).unwrap();
        gpu.add_source_line(s.src_x, k0, k1, a).unwrap();
    }
    let ez = gpu.download_ez().unwrap();
    assert!(ez[65535 * nx..].iter().any(|&e| e != 0.0), "no field past row 65535");
    assert_eq!(bit_mismatches(&ez, &cpu.ez), 0);
}

/// Kernels behind the viewer, without a display: the brush sets exactly the
/// cells of the disk to the constructor's cb bits, the heat map puts grid row
/// j at image row ny-1-j with red for +Ez and blue for -Ez, and reset clears
/// the fields but keeps the geometry.
#[test]
fn gpu_viewer_kernels_paint_render_reset() {
    let (nx, ny) = (64usize, 48usize);
    let mut g = GpuYee::new(nx, ny, Pml::new(4), courant(), &|_, _| EPS_AIR).unwrap();
    g.paint_disk(30, 20, 5, EPS_SI).unwrap();
    let cb = g.download_cb().unwrap();
    let (si, air) = (photon_fdtd::cb_coeff(courant(), EPS_SI), photon_fdtd::cb_coeff(courant(), EPS_AIR));
    for j in 0..ny {
        for i in 0..nx {
            let inside = (i as i64 - 30).pow(2) + (j as i64 - 20).pow(2) <= 25;
            let want = if inside { si } else { air };
            assert_eq!(cb[j * nx + i].to_bits(), want.to_bits(), "cell ({i}, {j})");
        }
    }
    g.set_ez(10 * nx + 40, 1.0).unwrap();
    g.set_ez(30 * nx + 12, -1.0).unwrap();
    let mut img = g.stream().alloc_zeros::<u32>(nx * ny).unwrap();
    g.render_rgba(&mut img, 1.0).unwrap();
    let img = g.stream().clone_dtoh(&img).unwrap();
    let px = |i: usize, j: usize| img[(ny - 1 - j) * nx + i];
    assert_eq!(px(40, 10), 0xFF0000FF, "+Ez is pure red (RGBA8 little endian)");
    assert_eq!(px(12, 30), 0xFFFF0000, "-Ez is pure blue");
    let grey = px(30, 20) & 0xFF;
    assert!(grey > 0 && px(30, 20) == 0xFF000000 | grey << 16 | grey << 8 | grey, "silicon is grey");
    assert_eq!(px(20, 24), 0xFF000000, "air with Ez = 0 is black");
    for _ in 0..20 {
        g.step().unwrap();
    }
    assert!(g.download_ez().unwrap().iter().any(|&e| e != 0.0));
    g.reset_fields().unwrap();
    assert!(g.download_ez().unwrap().iter().all(|&e| e == 0.0));
    assert_eq!(bit_mismatches(&g.download_cb().unwrap(), &cb), 0, "reset keeps the geometry");
}

// ---- Stage 4: slab mode, ports and S-parameters (src/sparams.rs) ----

use photon_fdtd::sparams::{
    Geometry, PortSetup, Run, analytic_overlap, db, discrete_mode, measured_beta, slab_even_mode,
};

/// The grid-dispersion error |beta_analytic / beta_grid - 1| decreases by 4x
/// when the resolution doubles (second order), so the analytic root is the
/// continuum limit of the Yee eigenvalue. Single mode over the whole band.
#[test]
fn slab_mode_grid_dispersion_is_second_order() {
    let (s1, s2) = (PortSetup::standard(Geometry::Straight, 1), PortSetup::standard(Geometry::Straight, 2));
    let (e1, e2) = (s1.eps_column(), s2.eps_column());
    for (f1, f2) in s1.freqs().into_iter().zip(s2.freqs()) {
        let gap = |s: &PortSetup, e: &[f64], f: f64| {
            slab_even_mode(s.core_w as f64, EPS_SI.sqrt(), 1.0, f).beta / discrete_mode(e, f, courant()).beta - 1.0
        };
        let (g1, g2) = (gap(&s1, &e1, f1), gap(&s2, &e2, f2));
        assert!((3.8..4.2).contains(&(g1 / g2)), "lambda0 {:.1}: gap {g1:.3e} -> {g2:.3e}", 1.0 / f1);
        assert!(g1.abs() < 2e-3);
        assert_eq!(discrete_mode(&e1, f1, courant()).guided, 1, "not single mode at lambda0 {:.1}", 1.0 / f1);
    }
}

/// Straight guide into the right CPML. Gates:
/// - the measured beta (DFT phase along the slab) equals the grid eigenvalue within 1e-4,
///   and the analytic beta within the grid-dispersion error + 1e-4;
/// - the analytic profile overlaps the grid eigenmode to better than 1 - 1e-5;
/// - |S11|^2 + |S21|^2 = 1 +- 1e-3 for the modal and for the flux S-parameters;
/// - modal back-reflection from the 10-cell CPML <= -70 dB;
/// - the forward/backward mode split puts <= -60 dB of the incident wave into backward.
#[test]
fn straight_guide_mode_energy_balance_and_cpml_reflection() {
    let st = PortSetup::standard(Geometry::Straight, 1);
    assert!(st.reference_is_clean());
    let (reference, dut) = (st.run(Run::Reference).unwrap(), st.run(Run::Dut).unwrap());
    let eps = st.eps_column();
    let s = st.sparams(&reference, &dut);
    for (k, &f) in reference.freqs.iter().enumerate() {
        let grid = discrete_mode(&eps, f, courant()).beta;
        let an = slab_even_mode(st.core_w as f64, EPS_SI.sqrt(), 1.0, f).beta;
        let me = measured_beta(&reference.axis[k]);
        let l = 1.0 / f;
        assert!((me / grid - 1.0).abs() <= 1e-4, "lambda0 {l:.1}: measured {me} grid {grid}");
        assert!((me / an - 1.0).abs() <= (an / grid - 1.0).abs() + 1e-4, "lambda0 {l:.1}: measured {me} analytic {an}");
        assert!(1.0 - analytic_overlap(&st, f) < 1e-5);
        let (mode_sum, flux_sum) = (s.s11_mode[k] + s.s21_mode[k], s.s11[k] + s.s21[k]);
        println!("lambda0 {l:.1}: S11 {:.1} dB, mode sum-1 {:.2e}, flux sum-1 {:.2e}", db(s.s11_mode[k]), mode_sum - 1.0, flux_sum - 1.0);
        assert!((mode_sum - 1.0).abs() <= 1e-3 && (flux_sum - 1.0).abs() <= 1e-3);
        assert!(db(s.s11_mode[k]) <= -70.0, "CPML modal reflection {:.1} dB", db(s.s11_mode[k]));
        assert!(db(s.inc_backward[k]) <= -60.0, "mode split: incident backward {:.1} dB", db(s.inc_backward[k]));
    }
}

/// Baseline of the sharp 90-degree bend at the band centre (lambda0 = 146 cells):
/// modal insertion loss 3.748 dB and return loss 6.69 dB (+- 0.05 dB). The
/// run is deterministic; a change of these values means a change of the solver
/// or of the setup. The Stage 5 optimiser must beat this benchmark.
#[test]
fn bend_90_baseline_insertion_and_return_loss() {
    let bd = PortSetup::standard(Geometry::Bend, 1);
    assert!(bd.reference_is_clean());
    let s = bd.sparams(&bd.run(Run::Reference).unwrap(), &bd.run(Run::Dut).unwrap());
    let k = bd.nf / 2;
    let (il, rl) = (-db(s.s21_mode[k]), -db(s.s11_mode[k]));
    println!("lambda0 {:.1}: IL {il:.3} dB, RL {rl:.2} dB, radiated {:.4}", 1.0 / s.freqs[k], 1.0 - s.s11_mode[k] - s.s21_mode[k]);
    assert!((il - 3.748).abs() < 0.05 && (rl - 6.69).abs() < 0.05);
    for k in 0..s.freqs.len() {
        assert!(s.s11_mode[k] + s.s21_mode[k] < 1.0 && s.s11_mode[k] > 0.0 && s.s21_mode[k] > 0.0);
    }
}
