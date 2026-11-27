//! Stage 5 baselines: the design box filled with the sharp L (must reproduce
//! the Stage 4 benchmark) and with a circular arc, for box sizes given on the
//! command line (default 80 and 146 cells, margin 8).
//!
//! Usage: bend_design [n ...]

use photon_fdtd::design::{DesignBox, binary_fraction, single_cell_islands};
use photon_fdtd::sparams::{Geometry, PortSetup, Run, db};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sizes: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let sizes = if sizes.is_empty() { vec![80, 146] } else { sizes };
    let base = PortSetup::standard(Geometry::Bend, 1);
    let reference = base.run(Run::Reference)?;
    let k = base.nf / 2;
    for n in sizes {
        let bx = DesignBox::for_bend(&base, n, 8);
        println!(
            "box {n} x {n} at ({}, {}), margin {}, arc centreline radius {} cells ({:.2} lambda0 at 146)",
            bx.x0,
            bx.y0,
            bx.margin,
            bx.arc_radius(),
            bx.arc_radius() as f64 * base.fc()
        );
        for (name, shape) in [("sharp L", bx.sharp_l()), ("circular arc", bx.circular_arc())] {
            let st = base.clone().with_design(&bx, &shape);
            let s = st.sparams(&reference, &st.run(Run::Dut)?);
            let il: Vec<f64> = s.s21_mode.iter().map(|v| -db(*v)).collect();
            let rl: Vec<f64> = s.s11_mode.iter().map(|v| -db(*v)).collect();
            println!(
                "  {name:<13} binary {:.3}, 1-cell islands {} | centre IL {:.3} dB RL {:.2} dB | band IL {:.3}..{:.3} dB, RL {:.2}..{:.2} dB",
                binary_fraction(&shape),
                single_cell_islands(&bx, &shape),
                il[k],
                rl[k],
                il.iter().cloned().fold(f64::INFINITY, f64::min),
                il.iter().cloned().fold(0.0, f64::max),
                rl.iter().cloned().fold(f64::INFINITY, f64::min),
                rl.iter().cloned().fold(0.0, f64::max)
            );
            for (q, f) in s.freqs.iter().enumerate() {
                println!("    lambda0 {:>6.1}  IL {:>7.3}  RL {:>6.2}  radiated {:.4}", 1.0 / f, il[q], rl[q], 1.0 - s.s11_mode[q] - s.s21_mode[q]);
            }
        }
    }
    Ok(())
}
