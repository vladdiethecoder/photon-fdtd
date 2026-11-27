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
