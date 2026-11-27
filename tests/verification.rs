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

// ---- Stage 5: design region (src/design.rs) ----

use photon_fdtd::design::{
    ConicFilter, DesignBox, binary_fraction, densities, project, project_grad, pullback, single_cell_islands,
};

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

/// Filter and projection: partition of unity, exterior padding continues the
/// slabs into the box, W^T is the transpose of W, and the chain-rule pullback
/// matches central finite differences.
#[test]
fn design_filter_projection_and_pullback() {
    let base = PortSetup::standard(Geometry::Bend, 1);
    let bx = DesignBox::for_bend(&base, 48, 8);
    let n = bx.n;
    let f = ConicFilter::new(4.0);
    // Partition of unity: a box full of silicon, with exterior silicon only in the slabs,
    // stays 1 where the whole kernel lies in silicon.
    let ones = f.apply(&bx, &vec![1.0; n * n]);
    assert!((ones[(n / 2) * n + n / 2] - 1.0).abs() < 1e-12);
    // The slab continues across the left box edge: filtered sharp L at the slab centre on a = 0.
    let l = f.apply(&bx, &bx.sharp_l());
    assert!(l[(bx.margin + bx.core_w / 2) * n] > 0.99, "slab pinched at the box edge");
    // Transpose: <W x, y> = <x, W^T y> for the linear part.
    let mut seed = 7u64;
    let x: Vec<f64> = (0..n * n).map(|_| lcg(&mut seed)).collect();
    let y: Vec<f64> = (0..n * n).map(|_| lcg(&mut seed)).collect();
    let c = f.apply(&bx, &vec![0.0; n * n]);
    let wx: Vec<f64> = f.apply(&bx, &x).iter().zip(&c).map(|(a, b)| a - b).collect();
    let lhs: f64 = wx.iter().zip(&y).map(|(a, b)| a * b).sum();
    let rhs: f64 = x.iter().zip(f.pullback(&bx, &y)).map(|(a, b)| a * b).sum();
    assert!((lhs - rhs).abs() < 1e-10 * lhs.abs());
    // Projection derivative and the full chain against finite differences (J = sum g_k rho_bar_k).
    for &xv in &[0.1, 0.45, 0.5, 0.8] {
        let h = 1e-6;
        let fd = (project(xv + h, 8.0, 0.5) - project(xv - h, 8.0, 0.5)) / (2.0 * h);
        assert!((fd - project_grad(xv, 8.0, 0.5)).abs() < 1e-6);
    }
    let g: Vec<f64> = (0..n * n).map(|_| lcg(&mut seed) - 0.5).collect();
    let rho: Vec<f64> = (0..n * n).map(|_| lcg(&mut seed)).collect();
    let analytic = pullback(&bx, &f, &rho, 8.0, &g);
    let j = |r: &[f64]| densities(&bx, &f, r, 8.0).iter().zip(&g).map(|(a, b)| a * b).sum::<f64>();
    let (mut dot, mut na, mut nf) = (0.0, 0.0, 0.0);
    for k in (0..n * n).step_by(97) {
        let (mut p, mut m) = (rho.clone(), rho.clone());
        p[k] += 1e-6;
        m[k] -= 1e-6;
        let fd = (j(&p) - j(&m)) / 2e-6;
        dot += fd * analytic[k];
        na += analytic[k] * analytic[k];
        nf += fd * fd;
    }
    assert!(dot / (na * nf).sqrt() > 0.999999, "pullback vs finite differences");
}

/// Fabrication metrics: binary fraction and isolated 1-cell islands.
#[test]
fn design_metrics_binary_fraction_and_islands() {
    let base = PortSetup::standard(Geometry::Bend, 1);
    let bx = DesignBox::for_bend(&base, 48, 8);
    let n = bx.n;
    for shape in [bx.sharp_l(), bx.circular_arc()] {
        assert_eq!(binary_fraction(&shape), 1.0);
        assert_eq!(single_cell_islands(&bx, &shape), 0);
    }
    let mut s = bx.sharp_l();
    s[40 * n + 4] = 1.0; // isolated silicon pixel in the cladding
    s[(bx.margin + 8) * n + 10] = 0.0; // isolated air pixel in the slab
    assert_eq!(single_cell_islands(&bx, &s), 2);
    let mut g = s.clone();
    g[0] = 0.5;
    assert!(binary_fraction(&g) < 1.0);
}

/// Forward evaluation: the box filled with the sharp L gives bit-identical
/// modal S-parameters to the Stage 4 bend. The circular-arc baselines are
/// recorded: in an 80 x 80 box (R = 64 cells) the band maximum IL is 0.258 dB
/// and the band minimum RL 40.9 dB; in a 48 x 48 box (R = 32) 0.693 dB and 27.5 dB.
#[test]
fn design_box_forward_evaluation_and_baselines() {
    let base = PortSetup::standard(Geometry::Bend, 1);
    let reference = base.run(Run::Reference).unwrap();
    let plain = base.sparams(&reference, &base.run(Run::Dut).unwrap());
    let bx80 = DesignBox::for_bend(&base, 80, 8);
    let l = base.clone().with_design(&bx80, &bx80.sharp_l());
    let sl = l.sparams(&reference, &l.run(Run::Dut).unwrap());
    assert_eq!(bit_mismatches(&sl.s21_mode, &plain.s21_mode), 0);
    assert_eq!(bit_mismatches(&sl.s11_mode, &plain.s11_mode), 0);
    for (n, il_max, rl_min) in [(80usize, 0.258, 40.91), (48, 0.693, 27.52)] {
        let bx = DesignBox::for_bend(&base, n, 8);
        let st = base.clone().with_design(&bx, &bx.circular_arc());
        let s = st.sparams(&reference, &st.run(Run::Dut).unwrap());
        let il = s.s21_mode.iter().map(|v| -db(*v)).fold(0.0, f64::max);
        let rl = s.s11_mode.iter().map(|v| -db(*v)).fold(f64::INFINITY, f64::min);
        println!("arc in {n} x {n}: band max IL {il:.3} dB, band min RL {rl:.2} dB");
        assert!((il - il_max).abs() < 0.01 && (rl - rl_min).abs() < 0.1);
    }
}

// ---- Stage 5: adjoint gradient and baselines ----

use photon_fdtd::adjoint::{Objective, forward, gradient, pnorm_loss};
use photon_fdtd::design::{disk, feature_violations};

/// Adjoint dT_k/deps against finite differences on a grey design (filtered
/// arc, beta 4, latent 0.15..0.85) in the 48 x 48 box, Stage 5 run length:
/// 4 x 4 block per cell (cosine > 0.999, |adj|/|fd| within 1e-3, 16/16 signs),
/// directional derivatives (steps 1e-2 and 1e-3, ratio within 1e-3) at three
/// wavelengths, and the full chain latent rho -> F (ratio within 1e-4).
#[test]
fn adjoint_gradient_matches_finite_differences() {
    let mut base = PortSetup::standard(Geometry::Bend, 1);
    base.extra_time = 6000.0;
    let bx = DesignBox::for_bend(&base, 48, 8);
    let n = bx.n;
    let filt = ConicFilter::new(5.0);
    let rho0: Vec<f64> = bx.circular_arc().iter().map(|v| 0.15 + 0.7 * v).collect();
    let rho_bar = densities(&bx, &filt, &rho0, 4.0);
    let reference = base.run(Run::Reference).unwrap();
    let setup = base.clone().with_design(&bx, &rho_bar);
    let obj = Objective::new(&setup, &reference);
    let fwd = forward(&setup, &obj, &bx).unwrap();
    let grad = gradient(&setup, &obj, &bx, &fwd).unwrap();
    let eps0: Vec<f64> = rho_bar.iter().map(|&r| DesignBox::eps(r)).collect();
    let eval = |eps: &[f64]| {
        let rb: Vec<f64> = eps.iter().map(|&e| (e - EPS_AIR) / (EPS_SI - EPS_AIR)).collect();
        forward(&base.clone().with_design(&bx, &rb), &obj, &bx).unwrap().t
    };
    let r = bx.arc_radius() as f64;
    let (ca, cb) = ((r * 0.7071) as usize - 2, (n as f64 - r * 0.7071) as usize - 2);
    let cells: Vec<usize> = (0..16).map(|q| (cb + q / 4) * n + ca + q % 4).collect();
    let ks = [0usize, 7, 14];
    let mut fd = vec![vec![0.0; 16]; 15];
    for (q, &c) in cells.iter().enumerate() {
        let (mut p, mut m) = (eps0.clone(), eps0.clone());
        p[c] += 1e-3;
        m[c] -= 1e-3;
        let (tp, tm) = (eval(&p), eval(&m));
        for k in 0..15 {
            fd[k][q] = (tp[k] - tm[k]) / 2e-3;
        }
    }
    for &k in &ks {
        let ad: Vec<f64> = cells.iter().map(|&c| grad[k][c]).collect();
        let dot: f64 = ad.iter().zip(&fd[k]).map(|(a, b)| a * b).sum();
        let (na, nb) = (ad.iter().map(|x| x * x).sum::<f64>().sqrt(), fd[k].iter().map(|x| x * x).sum::<f64>().sqrt());
        println!("lambda0 {:.1}: cosine {:.8}, |adj|/|fd| {:.6}", 1.0 / obj.freqs[k], dot / (na * nb), na / nb);
        assert!(dot / (na * nb) > 0.999);
        assert!((na / nb - 1.0).abs() < 1e-3);
        assert!(ad.iter().zip(&fd[k]).all(|(a, b)| a.signum() == b.signum()));
    }
    let mut seed = 11u64;
    let d: Vec<f64> = (0..16).map(|_| lcg(&mut seed) - 0.5).collect();
    for &h in &[1e-2, 1e-3] {
        let (mut p, mut m) = (eps0.clone(), eps0.clone());
        for (q, &c) in cells.iter().enumerate() {
            p[c] += h * d[q];
            m[c] -= h * d[q];
        }
        let (tp, tm) = (eval(&p), eval(&m));
        for &k in &ks {
            let adj: f64 = cells.iter().zip(&d).map(|(&c, dq)| grad[k][c] * dq).sum();
            let ratio = adj / ((tp[k] - tm[k]) / (2.0 * h));
            println!("directional h {h:.0e} lambda0 {:.1}: ratio {ratio:.8}", 1.0 / obj.freqs[k]);
            assert!((ratio - 1.0).abs() < 1e-3);
        }
    }
    let (_, dfdt) = pnorm_loss(&fwd.t, 8.0);
    let mut g_bar = vec![0.0; n * n];
    for (k, w) in dfdt.iter().enumerate() {
        for (gb, gk) in g_bar.iter_mut().zip(&grad[k]) {
            *gb += w * gk * (EPS_SI - EPS_AIR);
        }
    }
    let g_rho = pullback(&bx, &filt, &rho0, 4.0, &g_bar);
    let dr: Vec<f64> = (0..n * n).map(|_| lcg(&mut seed) - 0.5).collect();
    let adj: f64 = g_rho.iter().zip(&dr).map(|(a, b)| a * b).sum();
    let f_at = |s: f64| {
        let r: Vec<f64> = rho0.iter().zip(&dr).map(|(a, b)| a + s * 1e-3 * b).collect();
        pnorm_loss(&forward(&base.clone().with_design(&bx, &densities(&bx, &filt, &r, 4.0)), &obj, &bx).unwrap().t, 8.0).0
    };
    let ratio = adj / ((f_at(1.0) - f_at(-1.0)) / 2e-3);
    println!("full chain: ratio {ratio:.8}");
    assert!((ratio - 1.0).abs() < 1e-4);
}

/// The L_min check detects a 3-cell silicon line and a 3-cell air gap, and
/// passes the arc. The best conventional baseline of the protocol (Bezier
/// k = 0.40) keeps its measured worst-case IL of 0.5933 dB.
#[test]
fn stage5_feature_check_and_best_baseline() {
    let mut base = PortSetup::standard(Geometry::Bend, 1);
    base.extra_time = 6000.0;
    let bx = DesignBox::for_bend(&base, 48, 8);
    let n = bx.n;
    assert_eq!(feature_violations(&bx, &bx.circular_arc(), &disk(4)), (0, 0));
    let mut thin = bx.sharp_l();
    for a in 5..20 {
        for b in 30..33 {
            thin[b * n + a] = 1.0;
        }
    }
    assert!(feature_violations(&bx, &thin, &disk(4)).0 > 0, "thin silicon not detected");
    let mut gap = bx.sharp_l();
    for b in bx.margin..bx.margin + bx.core_w {
        for a in 10..13 {
            gap[b * n + a] = 0.0;
        }
    }
    assert!(feature_violations(&bx, &gap, &disk(4)).1 > 0, "narrow air gap not detected");
    let reference = base.run(Run::Reference).unwrap();
    let st = base.clone().with_design(&bx, &bx.rasterise(&bx.bezier_centreline(0.40)));
    let s = st.sparams(&reference, &st.run(Run::Dut).unwrap());
    let il = s.s21_mode.iter().map(|v| -db(*v)).fold(0.0, f64::max);
    assert!((il - 0.5933).abs() < 0.0005, "Bezier k=0.40 worst IL {il}");
}

/// The committed optimised 48 x 48 design (docs/results/stage5/rho_final.txt,
/// threshold of the filtered density, r_f = 5) meets the Stage 5 gates on the
/// 15 frequencies: worst-case IL < 0.50 dB and RL > 20 dB, L_min and islands,
/// with the recorded worst-case IL 0.2586 dB.
#[test]
fn stage5_optimised_design_meets_gates() {
    let mut base = PortSetup::standard(Geometry::Bend, 1);
    base.extra_time = 6000.0;
    let bx = DesignBox::for_bend(&base, 48, 8);
    let rho: Vec<f64> = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/results/stage5/rho_final.txt"))
        .unwrap()
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let filt = ConicFilter::new(5.0);
    assert!(binary_fraction(&densities(&bx, &filt, &rho, 128.0)) >= 0.99);
    let bin: Vec<f64> = filt.apply(&bx, &rho).iter().map(|&v| if v >= 0.5 { 1.0 } else { 0.0 }).collect();
    assert_eq!(single_cell_islands(&bx, &bin), 0);
    assert_eq!(feature_violations(&bx, &bin, &disk(4)), (0, 0));
    let reference = base.run(Run::Reference).unwrap();
    let st = base.clone().with_design(&bx, &bin);
    let s = st.sparams(&reference, &st.run(Run::Dut).unwrap());
    let il = s.s21_mode.iter().map(|v| -db(*v)).fold(0.0, f64::max);
    let rl = s.s11_mode.iter().map(|v| -db(*v)).fold(f64::INFINITY, f64::min);
    println!("optimised: worst IL {il:.4} dB, worst RL {rl:.2} dB");
    assert!(il < 0.5 && rl > 20.0 && (il - 0.2586).abs() < 0.002);
}

// ---- Stage 6: GDSII / SVG export ----

use photon_fdtd::export::{Mask, area2, gds, group, rasterise, read_gds, trace};

fn export_round_trip(mask: &Mask) -> (usize, usize) {
    let groups = group(trace(mask));
    let area: i64 = groups.iter().map(|g| area2(&g.0) + g.1.iter().map(area2).sum::<i64>()).sum::<i64>() / 2;
    assert_eq!(area, mask.cells() as i64, "traced area differs from the cell count");
    for (o, hs) in &groups {
        for r in std::iter::once(o).chain(hs) {
            for i in 0..r.len() {
                let (p, q) = (r[i], r[(i + 1) % r.len()]);
                assert!(p.0 == q.0 || p.1 == q.1, "non-Manhattan edge {p:?} -> {q:?}");
            }
        }
    }
    let bytes = gds(mask, 10, "T", 1);
    let (polys, units) = read_gds(&bytes);
    assert_eq!(units, (1e-3, 1e-9));
    assert_eq!(polys.iter().map(area2).sum::<i64>() / 2, mask.cells() as i64 * 100);
    assert_eq!(rasterise(&polys, 10, mask.nx, mask.ny), mask.on, "GDSII round trip differs from the mask");
    (groups.len(), groups.iter().map(|g| g.1.len()).sum())
}

/// Export: a synthetic mask with two holes, an island inside a hole, and two
/// diagonal saddles; then the committed 48 x 48 design. Each must give
/// Manhattan loops whose area equals the cell count, and a GDSII file whose
/// polygons rasterise back to the same mask, cell for cell.
#[test]
fn export_gds_round_trip_matches_mask() {
    let rows = [
        "..............",
        ".##########...",
        ".#........#...",
        ".#.####...#...",
        ".#.#..#...#.#.",
        ".#.#.##...##..",
        ".#.####...#...",
        ".#........#...",
        ".##########...",
        ".#.#.#........",
        ".###.#.#......",
        "......#.......",
    ];
    let (nx, ny) = (rows[0].len(), rows.len());
    let on = (0..nx * ny).map(|k| rows[ny - 1 - k / nx].as_bytes()[k % nx] == b'#').collect();
    let (outer, holes) = export_round_trip(&Mask { nx, ny, on });
    println!("synthetic: {outer} outer boundaries, {holes} holes");
    assert_eq!((outer, holes), (5, 3));

    let bx = DesignBox::for_bend(&PortSetup::standard(Geometry::Bend, 1), 48, 8);
    let rho: Vec<f64> = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/results/stage5/rho_final.txt"))
        .unwrap()
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let bin: Vec<f64> = ConicFilter::new(5.0).apply(&bx, &rho).iter().map(|&v| if v >= 0.5 { 1.0 } else { 0.0 }).collect();
    let (outer, holes) = export_round_trip(&Mask::from_box(&bx, &bin, 16));
    println!("optimised 48 x 48: {outer} outer boundaries, {holes} holes");
}

/// Port-1 reflection gradient dR_k/deps against finite differences, on the
/// same grey design as `adjoint_gradient_matches_finite_differences`:
/// 4 x 4 block at 3 wavelengths (cosine > 0.999, magnitude within 1e-3,
/// 16/16 signs) and directional derivatives (steps 1e-2, 1e-3, within 1e-3).
/// The single-line R estimate agrees with the two-line modal |S11|^2 of
/// `sparams` to within 2 % (relative) at every band sample.
#[test]
fn adjoint_reflection_gradient_matches_finite_differences() {
    use photon_fdtd::adjoint::gradient_r;
    let mut base = PortSetup::standard(Geometry::Bend, 1);
    base.extra_time = 6000.0;
    let bx = DesignBox::for_bend(&base, 48, 8);
    let n = bx.n;
    let filt = ConicFilter::new(5.0);
    let rho0: Vec<f64> = bx.circular_arc().iter().map(|v| 0.15 + 0.7 * v).collect();
    let rho_bar = densities(&bx, &filt, &rho0, 4.0);
    let reference = base.run(Run::Reference).unwrap();
    let setup = base.clone().with_design(&bx, &rho_bar);
    let obj = Objective::with_reflection(&setup, &reference);
    println!("port-1 basis {} vectors, residual {:.2e}", obj.p1.as_ref().unwrap().basis.len(), obj.p1.as_ref().unwrap().basis_residual);
    let fwd = forward(&setup, &obj, &bx).unwrap();
    let s = setup.sparams(&reference, &setup.run(Run::Dut).unwrap());
    for k in 0..15 {
        let rel = (fwd.r[k] - s.s11_mode[k]).abs() / s.s11_mode[k];
        println!("lambda0 {:.1}: R adjoint-line {:.6e}, |S11|^2 {:.6e}, rel {rel:.2e}", 1.0 / obj.freqs[k], fwd.r[k], s.s11_mode[k]);
        assert!(rel < 0.02);
    }
    let grad = gradient_r(&setup, &obj, &bx, &fwd).unwrap();
    let eps0: Vec<f64> = rho_bar.iter().map(|&r| DesignBox::eps(r)).collect();
    let eval = |eps: &[f64]| {
        let rb: Vec<f64> = eps.iter().map(|&e| (e - EPS_AIR) / (EPS_SI - EPS_AIR)).collect();
        forward(&base.clone().with_design(&bx, &rb), &obj, &bx).unwrap().r
    };
    let r = bx.arc_radius() as f64;
    let (ca, cb) = ((r * 0.7071) as usize - 2, (n as f64 - r * 0.7071) as usize - 2);
    let cells: Vec<usize> = (0..16).map(|q| (cb + q / 4) * n + ca + q % 4).collect();
    let ks = [0usize, 7, 14];
    let mut fd = vec![vec![0.0; 16]; 15];
    for (q, &c) in cells.iter().enumerate() {
        let (mut p, mut m) = (eps0.clone(), eps0.clone());
        p[c] += 1e-3;
        m[c] -= 1e-3;
        let (rp, rm) = (eval(&p), eval(&m));
        for k in 0..15 {
            fd[k][q] = (rp[k] - rm[k]) / 2e-3;
        }
    }
    for &k in &ks {
        let ad: Vec<f64> = cells.iter().map(|&c| grad[k][c]).collect();
        let dot: f64 = ad.iter().zip(&fd[k]).map(|(a, b)| a * b).sum();
        let (na, nb) = (ad.iter().map(|x| x * x).sum::<f64>().sqrt(), fd[k].iter().map(|x| x * x).sum::<f64>().sqrt());
        let signs = ad.iter().zip(&fd[k]).filter(|(a, b)| a.signum() == b.signum()).count();
        println!("R lambda0 {:.1}: cosine {:.8}, |adj|/|fd| {:.6}, signs {signs}/16", 1.0 / obj.freqs[k], dot / (na * nb), na / nb);
        assert!(dot / (na * nb) > 0.999);
        assert!((na / nb - 1.0).abs() < 1e-3);
        assert_eq!(signs, 16);
    }
    let mut seed = 17u64;
    let d: Vec<f64> = (0..16).map(|_| lcg(&mut seed) - 0.5).collect();
    for &h in &[1e-2, 1e-3] {
        let (mut p, mut m) = (eps0.clone(), eps0.clone());
        for (q, &c) in cells.iter().enumerate() {
            p[c] += h * d[q];
            m[c] -= h * d[q];
        }
        let (rp, rm) = (eval(&p), eval(&m));
        for &k in &ks {
            let adj: f64 = cells.iter().zip(&d).map(|(&c, dq)| grad[k][c] * dq).sum();
            let ratio = adj / ((rp[k] - rm[k]) / (2.0 * h));
            println!("R directional h {h:.0e} lambda0 {:.1}: ratio {ratio:.8}", 1.0 / obj.freqs[k]);
            assert!((ratio - 1.0).abs() < 1e-3);
        }
    }
}

// ---- Stage 6: robust evaluator rules ----

/// Geometry and bias rules of docs/stage6-protocol.md (sections 2 and 3) on
/// cases with a known answer, at scale 1 and scale 2.
#[test]
fn stage6_geometry_and_bias_rules() {
    use photon_fdtd::robust::{bias, connectivity, disk_d, geometry, l1_ball, width_gap_violations};
    assert_eq!(l1_ball(1).len(), 5);
    assert_eq!(l1_ball(2).len(), 13);
    let mut a = disk_d(5);
    let mut b = disk(4);
    a.sort();
    b.sort();
    assert_eq!(a, b, "the 5-cell disk is the Stage 5 L_min disk");
    let d10 = disk_d(10);
    assert!(d10.iter().all(|&(x, y)| d10.contains(&(-1 - x, -1 - y))), "even disk is symmetric about its corner centre");
    for scale in [1usize, 2] {
        let base = PortSetup::standard(Geometry::Bend, scale);
        let bx = DesignBox::for_bend(&base, 80 * scale, 8 * scale);
        let n = bx.n;
        let arc = bx.circular_arc();
        let g = geometry(&bx, &arc, scale);
        println!("scale {scale}: arc {g:?}");
        assert!(g.feasible() && g.min_width == 12 * scale && g.enclosed_air == 0, "arc: width at the scan cap");
        // Bias: eroded < nominal < dilated; the closing (erosion of the dilation) contains the arc.
        let (e, d) = (bias(&bx, &arc, scale, false), bias(&bx, &arc, scale, true));
        let area = |s: &[f64]| s.iter().filter(|&&v| v >= 0.5).count();
        assert!(area(&e) < area(&arc) && area(&arc) < area(&d));
        let closed = bias(&bx, &d, scale, false);
        let added = closed.iter().zip(&arc).filter(|(c, a)| **c >= 0.5 && **a < 0.5).count();
        // The exterior padding is not biased (protocol section 2), so the
        // closing is extensive only away from the box edge.
        let lost: Vec<usize> = (0..n * n).filter(|&k| arc[k] >= 0.5 && closed[k] < 0.5).collect();
        let interior = |k: usize| (scale..n - scale).contains(&(k % n)) && (scale..n - scale).contains(&(k / n));
        assert!(lost.iter().all(|&k| !interior(k)), "a closing keeps every interior arc cell");
        println!("scale {scale}: closing adds {added} cells, drops {} edge cells next to the unbiased padding", lost.len());
        assert!(added < 2 * n);
        // A floating silicon square and a 3-native-cell line.
        let mut s = arc.clone();
        for q in 0..36 * scale * scale {
            s[(5 * scale + q / (6 * scale)) * n + 50 * scale + q % (6 * scale)] = 1.0;
        }
        let (conn, floating, _) = connectivity(&bx, &s);
        assert!(conn && floating == 1);
        let mut thin = bx.sharp_l();
        for a in 5 * scale..25 * scale {
            for b in 40 * scale..43 * scale {
                thin[b * n + a] = 1.0;
            }
        }
        assert!(width_gap_violations(&bx, &thin, &disk_d(5 * scale)).0 > 0);
        // A 3-native-cell air gap across the input slab breaks the ports and the gap rule.
        let mut gap = bx.sharp_l();
        for b in bx.margin..bx.margin + bx.core_w {
            for a in 10 * scale..13 * scale {
                gap[b * n + a] = 0.0;
            }
        }
        let gg = geometry(&bx, &gap, scale);
        assert!(!gg.ports_connected && gg.gap_violations > 0 && !gg.feasible());
    }
}

// ---- RTL reference: fixed-point 1D Yee stencil ----

/// src/fixed1d.rs (the bit-exact reference of rtl/yee1d.sv): the committed
/// golden vectors equal a fresh run of yee1d_vectors. Local error: one fixed
/// step from a state, against one f64 step (same quantised coefficients) from
/// the same state, differs by at most 0.5 LSB in H (one rounding) and
/// 0.5 + |CB| LSB <= 1 LSB in E (its own rounding plus the H rounding through
/// the difference), at every step of the nominal case, which never saturates.
/// The accumulated drift over 400 steps is printed (a lossless closed cavity,
/// so rounding errors do not decay). The saturate case sets the sticky flag at
/// step 59 and keeps every word inside 18 bits.
#[test]
fn fixed_point_yee1d_reference_and_vectors() {
    use photon_fdtd::fixed1d::{CF, MAX, MIN, case, step_f64};
    let dir = std::env::temp_dir().join(format!("yee1d_vectors_{}", std::process::id()));
    let st = std::process::Command::new(env!("CARGO_BIN_EXE_yee1d_vectors")).arg("--out").arg(&dir).status().unwrap();
    assert!(st.success());
    for case_name in ["nominal", "saturate"] {
        for f in ["cb.hex", "ch.hex", "src.hex", "e.hex", "h.hex", "ovf.hex", "meta.txt"] {
            let a = std::fs::read(dir.join(case_name).join(f)).unwrap();
            let b = std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("rtl/vectors").join(case_name).join(f)).unwrap();
            assert!(a == b, "rtl/vectors/{case_name}/{f} differs from the reference");
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    let (mut y, _, src) = case(0.5);
    let n = y.e.len();
    let cbq: Vec<f64> = y.cb.iter().map(|&c| c as f64 / (1u64 << CF) as f64).collect();
    let chq = y.ch as f64 / (1u64 << CF) as f64;
    let (mut ef, mut hf) = (vec![0.0; n], vec![0.0; n]);
    let (mut drift, mut local_e, mut local_h) = (0.0f64, 0.0f64, 0.0f64);
    let lsb = 1.0 / 32768.0;
    for &s in &src {
        let (mut le, mut lh): (Vec<f64>, Vec<f64>) = (y.e.iter().map(|&v| v as f64 * lsb).collect(), y.h.iter().map(|&v| v as f64 * lsb).collect());
        step_f64(&mut le, &mut lh, chq, &cbq, y.src, s as f64 * lsb);
        y.step(s);
        step_f64(&mut ef, &mut hf, chq, &cbq, y.src, s as f64 * lsb);
        for i in 0..n {
            local_e = local_e.max((y.e[i] as f64 * lsb - le[i]).abs() / lsb);
            local_h = local_h.max((y.h[i] as f64 * lsb - lh[i]).abs() / lsb);
            drift = drift.max((y.e[i] as f64 * lsb - ef[i]).abs().max((y.h[i] as f64 * lsb - hf[i]).abs()) / lsb);
        }
    }
    println!("nominal: local error E {local_e:.3} LSB, H {local_h:.3} LSB; drift after 400 steps {drift:.1} LSB");
    assert!(!y.ovf && local_h <= 0.5 && local_e <= 1.0);
    let (mut y, _, src) = case(3.9);
    let mut first = None;
    for (k, &s) in src.iter().enumerate() {
        y.step(s);
        assert!(y.e.iter().chain(&y.h).all(|v| (MIN..=MAX).contains(v)));
        if y.ovf && first.is_none() {
            first = Some(k);
        }
    }
    println!("saturate: first overflow at step {first:?}");
    assert_eq!(first, Some(59));
}
