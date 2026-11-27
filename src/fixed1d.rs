//! Fixed-point 1D Yee stencil: the bit-exact software reference of
//! `rtl/yee1d.sv` (docs/rtl.md).
//!
//! Grid: N cells, E at integer points, H between points. PEC ends: E[0] and
//! E[N-1] stay 0. H[N-1] is not used. One time step:
//!   H[i] <- sat(H[i] + rnd(CH * (E[i+1] - E[i])))                 0 <= i <= N-2
//!   E[i] <- sat(E[i] + rnd(CB[i] * (H[i] - H[i-1])) + s[i])        1 <= i <= N-2
//! s[i] = src for i == SRC, else 0.
//!
//! Word lengths (two's complement):
//!   E, H, src: W = 18 bits, F = 15 fraction bits (range [-4, 4), LSB 2^-15);
//!   CH, CB:    CW = 18 bits, CF = 16 fraction bits (range [-2, 2));
//!   difference: 19 bits (exact); product: 37 bits with F + CF = 31 fraction bits.
//! rnd(p) = (p + 2^(CF-1)) >> CF with an arithmetic shift: round half up
//! (towards +infinity), result with F fraction bits.
//! sat(v): v is formed in at least 21 bits (no overflow before saturation) and
//! clamped to [-2^17, 2^17 - 1]; a clamp sets the sticky overflow flag.

pub const W: u32 = 18;
pub const F: u32 = 15;
pub const CW: u32 = 18;
pub const CF: u32 = 16;
pub const MAX: i64 = (1 << (W - 1)) - 1;
pub const MIN: i64 = -(1 << (W - 1));

pub fn to_fixed(v: f64, frac: u32) -> i64 {
    (v * (1u64 << frac) as f64).round() as i64
}

#[derive(Clone, Debug)]
pub struct Yee1d {
    pub e: Vec<i64>,
    pub h: Vec<i64>,
    pub ch: i64,
    pub cb: Vec<i64>,
    pub src: usize,
    pub ovf: bool,
}

fn rnd(p: i64) -> i64 {
    (p + (1 << (CF - 1))) >> CF
}

impl Yee1d {
    /// `ch` and `cb` are the real coefficients; they are rounded to CF bits
    /// and must fit in CW bits.
    pub fn new(n: usize, ch: f64, cb: &[f64], src: usize) -> Self {
        let q = |v: f64| {
            let c = to_fixed(v, CF);
            assert!((-(1 << (CW - 1))..(1 << (CW - 1))).contains(&c), "coefficient outside {CW} bits");
            c
        };
        Yee1d { e: vec![0; n], h: vec![0; n], ch: q(ch), cb: cb.iter().map(|&c| q(c)).collect(), src, ovf: false }
    }

    fn sat(&mut self, v: i64) -> i64 {
        if v > MAX {
            self.ovf = true;
            MAX
        } else if v < MIN {
            self.ovf = true;
            MIN
        } else {
            v
        }
    }

    /// One time step with source sample `s` (W bits, F fraction bits).
    pub fn step(&mut self, s: i64) {
        assert!((MIN..=MAX).contains(&s), "source sample outside {W} bits");
        let n = self.e.len();
        for i in 0..n - 1 {
            let v = self.h[i] + rnd(self.ch * (self.e[i + 1] - self.e[i]));
            self.h[i] = self.sat(v);
        }
        for i in 1..n - 1 {
            let v = self.e[i] + rnd(self.cb[i] * (self.h[i] - self.h[i - 1])) + if i == self.src { s } else { 0 };
            self.e[i] = self.sat(v);
        }
    }
}

/// The same scheme in f64 (no rounding, no saturation), for the error check.
pub fn step_f64(e: &mut [f64], h: &mut [f64], ch: f64, cb: &[f64], src: usize, s: f64) {
    let n = e.len();
    for i in 0..n - 1 {
        h[i] += ch * (e[i + 1] - e[i]);
    }
    for i in 1..n - 1 {
        e[i] += cb[i] * (h[i] - h[i - 1]) + if i == src { s } else { 0.0 };
    }
}

/// Test case of the RTL testbench: N = 64 cells, Courant number 0.5,
/// a dielectric slab (eps 12.11) on cells 40..56, a Gaussian-sine source at
/// cell 8 with peak `amp`.
pub fn case(amp: f64) -> (Yee1d, Vec<f64>, Vec<i64>) {
    let n = 64;
    let s = 0.5;
    let cb: Vec<f64> = (0..n).map(|i| if (40..56).contains(&i) { s / 12.11 } else { s }).collect();
    let y = Yee1d::new(n, s, &cb, 8);
    let src: Vec<i64> = (0..400)
        .map(|k| {
            let t = k as f64;
            to_fixed(amp * (-((t - 60.0) / 20.0).powi(2)).exp() * (2.0 * std::f64::consts::PI * t / 24.0).sin(), F)
        })
        .collect();
    (y, cb, src)
}
