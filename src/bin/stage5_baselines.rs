//! Stage 5 conventional baselines, measured before the optimiser exists
//! (docs/stage5-protocol.md): circular arc, Euler bends p = 0.1 .. 1.0,
//! cubic Bezier k = 0.30 .. 0.70, in the 48 x 48 box, Stage 5 run length.

use photon_fdtd::design::{DesignBox, binary_fraction, disk, feature_violations, single_cell_islands};
use photon_fdtd::sparams::{Geometry, PortSetup, Run, db};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args().nth(1).map(|a| a.parse().unwrap()).unwrap_or(48);
    let mut base = PortSetup::standard(Geometry::Bend, 1);
    base.extra_time = 6000.0;
    let bx = DesignBox::for_bend(&base, n, 8);
    let reference = base.run(Run::Reference)?;
    let mut designs: Vec<(String, Vec<f64>)> = vec![("arc".into(), bx.circular_arc())];
    for q in 1..=10 {
        let p = q as f64 / 10.0;
        designs.push((format!("euler p={p:.1}"), bx.rasterise(&bx.euler_centreline(p))));
    }
    for q in 0..=8 {
        let k = 0.30 + 0.05 * q as f64;
        designs.push((format!("bezier k={k:.2}"), bx.rasterise(&bx.bezier_centreline(k))));
    }
    println!("box {n} x {n}, steps {}, L_min disk diameter 5", base.steps());
    let mut best = (String::new(), f64::INFINITY);
    for (name, shape) in &designs {
        let st = base.clone().with_design(&bx, shape);
        let s = st.sparams(&reference, &st.run(Run::Dut)?);
        let il = s.s21_mode.iter().map(|v| -db(*v)).fold(0.0, f64::max);
        let rl = s.s11_mode.iter().map(|v| -db(*v)).fold(f64::INFINITY, f64::min);
        let (vo, vc) = feature_violations(&bx, shape, &disk(4));
        println!(
            "{name:<14} worst IL {il:.4} dB  worst RL {rl:.2} dB  binary {:.3}  islands {}  L_min violations open {vo} close {vc}",
            binary_fraction(shape),
            single_cell_islands(&bx, shape)
        );
        if vo == 0 && vc == 0 && il < best.1 {
            best = (name.clone(), il);
        }
    }
    println!("best conventional baseline: {} (worst IL {:.4} dB)", best.0, best.1);
    Ok(())
}
