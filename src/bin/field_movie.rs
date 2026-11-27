//! Ez field frames of a bend design, from the solver (for the project page).
//!
//!   field_movie --mask DESIGN_s1.mask [--lambda0 146] [--frames 150] [--every 40] --out DIR
//!
//! The same port-1 mode source as the S-parameter runs, driven by a sine at
//! lambda0 with a 3-period raised-cosine ramp, on the Stage 6 scale-1 grid
//! (CPML on all sides). Writes DIR/frame_NNNN.ppm: red = +Ez, blue = -Ez,
//! grey = silicon, with one fixed colour scale for all frames (the 99.5th
//! percentile of |Ez| over the last frame). Also writes DIR/movie.json with
//! the frame times and the scale.

use std::f64::consts::PI;
use std::fmt::Write as _;

use photon_fdtd::design::DesignBox;
use photon_fdtd::gpu::{Field, GpuYee};
use photon_fdtd::sparams::{Geometry as G, PortSetup, Run, discrete_mode};
use photon_fdtd::stage6::{arg, read_mask};
use photon_fdtd::verify::courant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path: String = arg("--mask", String::new());
    let lambda0: f64 = arg("--lambda0", 146.0);
    let frames: usize = arg("--frames", 150);
    let every: usize = arg("--every", 40);
    let out: String = arg("--out", "out/stage6/movie".to_string());
    std::fs::create_dir_all(&out)?;
    let (n, mask) = read_mask(&path)?;
    let base = PortSetup::standard(G::Bend, 1);
    let bx = DesignBox::for_bend(&base, n, 8);
    let setup = base.clone().with_design(&bx, &mask);
    let (nx, ny) = setup.dut_grid();
    let dt = courant();
    let eps = |i: usize, j: usize| setup.eps_at(Run::Dut, i, j);
    let mut g = GpuYee::new(nx, ny, setup.pml, dt, &eps)?;
    let p = setup.pml.cells;
    let (r0, r1) = (p + 1, ny - p - 1);
    let f = 1.0 / lambda0;
    let mode = discrete_mode(&setup.eps_column(), f, dt);
    let prof = g.upload_profile(&mode.profile[r0..r1])?;
    // Crop: from the source to the right CPML, from the bottom CPML to the top CPML.
    let (x0, x1, y0, y1) = (setup.src_x - 20, nx - p, p, ny - p);
    let (w, h) = (x1 - x0, y1 - y0);
    let ramp = 3.0 / f;
    let mut snaps: Vec<Vec<f64>> = Vec::with_capacity(frames);
    let mut times = Vec::new();
    for k in 0..frames * every {
        g.step()?;
        let t = (k + 1) as f64 * dt;
        let env = if t < ramp { 0.5 * (1.0 - (PI * t / ramp).cos()) } else { 1.0 };
        g.add_source_profile(setup.src_x, r0, &prof, env * (2.0 * PI * f * t).sin())?;
        if (k + 1) % every == 0 {
            let ez = g.download(Field::Ez)?;
            snaps.push((0..w * h).map(|q| ez[(y0 + q / w) * nx + x0 + q % w]).collect());
            times.push(t);
        }
    }
    let mut last: Vec<f64> = snaps.last().unwrap().iter().map(|v| v.abs()).collect();
    last.sort_by(f64::total_cmp);
    let scale = last[(0.995 * (last.len() - 1) as f64) as usize];
    for (fi, s) in snaps.iter().enumerate() {
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        for row in (0..h).rev() {
            for col in 0..w {
                let v = (s[row * w + col] / scale).clamp(-1.0, 1.0);
                let si = eps(x0 + col, y0 + row) > 1.5;
                let bg = if si { 200.0 } else { 255.0 };
                let (r, gg, b) = if v >= 0.0 { (bg, bg * (1.0 - v), bg * (1.0 - v)) } else { (bg * (1.0 + v), bg * (1.0 + v), bg) };
                // Silicon outline: darker where a neighbour is air.
                // The crop is inside the CPML frame, so every neighbour is on the grid.
                let (ci, cj) = ((x0 + col) as i64, (y0 + row) as i64);
                let edge = si && [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| eps((ci + dx) as usize, (cj + dy) as usize) < 1.5);
                let k = if edge { 0.35 } else { 1.0 };
                ppm.extend([(r * k) as u8, (gg * k) as u8, (b * k) as u8]);
            }
        }
        std::fs::write(format!("{out}/frame_{fi:04}.ppm"), ppm)?;
    }
    let mut meta = String::new();
    write!(
        meta,
        "{{\"lambda0_cells\": {lambda0}, \"frames\": {frames}, \"steps_per_frame\": {every}, \"dt\": {dt}, \"crop\": [{x0}, {y0}, {x1}, {y1}], \"colour_scale_abs_ez\": {scale:e}, \"source\": \"port-1 discrete mode, sine with a 3-period raised-cosine ramp\", \"times\": [{}]}}",
        times.iter().map(|t| format!("{t:.3}")).collect::<Vec<_>>().join(", ")
    )?;
    std::fs::write(format!("{out}/movie.json"), meta)?;
    println!("{frames} frames of {w} x {h}, every {every} steps; |Ez| scale {scale:.4e}; wrote {out}");
    Ok(())
}
