//! Throughput: million Yee cell updates per second (one H + E update of one cell = 1).
//!
//! Usage: bench [--cpu 2048,4096] [--gpu 4096,8192] [--trials 3]
//! Square grids. Each line reports the median of the trials and the min..max spread.
//! Step counts scale with 1/n^2 so every trial runs for roughly 0.5-2 s.

use std::time::Instant;

use photon_fdtd::gpu::GpuYee;
use photon_fdtd::{COURANT_MAX_2D, Yee2d};

/// Minimum DRAM traffic per cell per step: H pass 5 reads (ez, hx, hy, ca, hb) + 2 writes,
/// E pass 5 reads (ez, hx, hy, ca, cb) + 1 write.
const BYTES_PER_CELL: f64 = 13.0 * 8.0;

fn make(n: usize) -> Yee2d {
    let mut sim = Yee2d::new(n, n, vec![1.0; n * n], 32, 0.99 * COURANT_MAX_2D).unwrap();
    // Non-zero O(1) field so no step runs on trivial zeros.
    let mut s = 0x9E3779B97F4A7C15u64;
    for e in sim.ez.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *e = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    }
    sim
}

fn steps_for(n: usize, cell_updates: f64, min: usize) -> usize {
    ((cell_updates / (n * n) as f64).ceil() as usize).max(min)
}

fn print(label: &str, n: usize, steps: usize, mut rates: Vec<f64>) {
    rates.sort_by(f64::total_cmp);
    let med = rates[rates.len() / 2];
    println!(
        "{label:<26} {n:>5}^2 {:>7.0} MiB {steps:>5} steps x{}  {med:>8.1} Mcells/s  [{:.1}..{:.1}]  {:>5.0} GB/s eff.",
        (n * n * 6 * 8) as f64 / (1u64 << 20) as f64,
        rates.len(),
        rates[0],
        rates[rates.len() - 1],
        med * 1e6 * BYTES_PER_CELL / 1e9
    );
}

fn cpu(n: usize, parallel: bool, trials: usize) {
    let mut sim = make(n);
    let step = |s: &mut Yee2d| if parallel { s.step_parallel() } else { s.step_serial() };
    let steps = if parallel { steps_for(n, 5e8, 3) } else { steps_for(n, 1.5e8, 3) };
    for _ in 0..3 {
        step(&mut sim);
    }
    let rates = (0..trials)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..steps {
                step(&mut sim);
            }
            (n * n * steps) as f64 / t.elapsed().as_secs_f64() / 1e6
        })
        .collect();
    let label =
        if parallel { format!("CPU rayon {} threads", rayon::current_num_threads()) } else { "CPU 1 thread".into() };
    print(&label, n, steps, rates);
    assert!(sim.ez.iter().all(|e| e.is_finite()), "non-finite field");
}

fn gpu(n: usize, trials: usize) {
    let sim = make(n);
    let mut g = GpuYee::from_cpu(&sim).expect("CUDA init");
    drop(sim);
    let steps = steps_for(n, 1.6e10, 100);
    g.timed_steps(20).unwrap(); // warm-up
    let rates = (0..trials)
        .map(|_| (n * n * steps) as f64 / (g.timed_steps(steps).unwrap() / 1e3) / 1e6)
        .collect();
    print(&format!("GPU {}", g.device_name()), n, steps, rates);
    assert!(g.download_ez().unwrap().iter().all(|e| e.is_finite()), "non-finite field");
}

fn list(args: &[String], flag: &str, default: &[usize]) -> Vec<usize> {
    match args.iter().position(|a| a == flag) {
        Some(p) => args[p + 1].split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect(),
        None => default.to_vec(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cpu_ns = list(&args, "--cpu", &[2048, 4096]);
    let gpu_ns = list(&args, "--gpu", &[4096, 8192]);
    let trials = list(&args, "--trials", &[3])[0];
    println!("f64, TM_z Yee, {BYTES_PER_CELL} B/cell/step minimum traffic; MiB = 6 f64 arrays");
    for &n in &cpu_ns {
        cpu(n, false, trials);
    }
    for &n in &cpu_ns {
        cpu(n, true, trials);
    }
    for n in gpu_ns {
        gpu(n, trials);
    }
}
