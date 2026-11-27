//! Golden vectors of the fixed-point 1D Yee stencil for rtl/tb_yee1d.sv.
//!
//!   yee1d_vectors [--out rtl/vectors]
//!
//! Two cases: nominal (source peak 0.5, no saturation) and saturate (source
//! peak 3.9, saturation expected). Each directory holds, as $readmemh files of
//! 18-bit two's-complement words (5 hex digits): cb.hex (N), ch.hex (1),
//! src.hex (T), e.hex and h.hex (T x N, after each step), and ovf.hex (T, the
//! sticky flag after each step, 1 hex digit). meta.txt gives N, T, SRC.

use std::fmt::Write as _;

use photon_fdtd::fixed1d::{W, case, step_f64};
use photon_fdtd::stage6::arg;

fn hex(v: i64) -> String {
    format!("{:05x}", (v as u64) & ((1 << W) - 1))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out: String = arg("--out", "rtl/vectors".to_string());
    for (name, amp) in [("nominal", 0.5), ("saturate", 3.9)] {
        let dir = format!("{out}/{name}");
        std::fs::create_dir_all(&dir)?;
        let (mut y, cb, src) = case(amp);
        let n = y.e.len();
        let (mut ef, mut hf) = (vec![0.0; n], vec![0.0; n]);
        let (mut e_txt, mut h_txt, mut o_txt) = (String::new(), String::new(), String::new());
        let mut err = 0.0f64;
        let mut first_ovf = None;
        for (k, &s) in src.iter().enumerate() {
            y.step(s);
            step_f64(&mut ef, &mut hf, 0.5, &cb, y.src, s as f64 / (1 << 15) as f64);
            for i in 0..n {
                writeln!(e_txt, "{}", hex(y.e[i]))?;
                writeln!(h_txt, "{}", hex(y.h[i]))?;
                err = err.max((y.e[i] as f64 / 32768.0 - ef[i]).abs()).max((y.h[i] as f64 / 32768.0 - hf[i]).abs());
            }
            writeln!(o_txt, "{:x}", y.ovf as u8)?;
            if y.ovf && first_ovf.is_none() {
                first_ovf = Some(k);
            }
        }
        std::fs::write(format!("{dir}/cb.hex"), y.cb.iter().map(|&c| hex(c) + "\n").collect::<String>())?;
        std::fs::write(format!("{dir}/ch.hex"), hex(y.ch) + "\n")?;
        std::fs::write(format!("{dir}/src.hex"), src.iter().map(|&s| hex(s) + "\n").collect::<String>())?;
        std::fs::write(format!("{dir}/e.hex"), e_txt)?;
        std::fs::write(format!("{dir}/h.hex"), h_txt)?;
        std::fs::write(format!("{dir}/ovf.hex"), o_txt)?;
        std::fs::write(format!("{dir}/meta.txt"), format!("N {n}\nT {}\nSRC {}\n", src.len(), y.src))?;
        println!("{name}: N {n}, T {}, max |fixed - f64| {err:.3e} ({:.1} LSB), first overflow step {first_ovf:?}", src.len(), err * 32768.0);
    }
    Ok(())
}
