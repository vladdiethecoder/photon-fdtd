//! Stage 6 run harness (docs/stage6-protocol.md): named start designs, mask
//! files, and the event log that every optimiser writes.
//!
//! Event log (CSV, one line per scored candidate or optimiser iteration):
//! event, kind, charged, forward_solves, adjoint_solves, cache_hits, wall_s,
//! forward_s, adjoint_s, j, geometry_ok, gates_ok, best_j, best_il, best_rl,
//! note. `charged` is the forward-equivalent cost of the protocol; the solve
//! counts are the physical runs. Both are cumulative.

use std::fs;
use std::io::Write;
use std::time::Instant;

use crate::design::DesignBox;
use crate::robust::{IL_MAX_DB, RL_MIN_DB, Scored};

/// Box mask of a named start design: arc, sharpL, bezier:K, euler:P,
/// and the protocol aliases I1 .. I5.
pub fn named_mask(bx: &DesignBox, name: &str) -> Vec<f64> {
    let name = match name {
        "I1" => "arc",
        "I2" => "bezier:0.40",
        "I3" => "euler:0.5",
        "I4" => "bezier:0.60",
        "I5" => "euler:1.0",
        other => other,
    };
    if name == "arc" {
        return bx.circular_arc();
    }
    if name == "sharpL" {
        return bx.sharp_l();
    }
    if let Some(k) = name.strip_prefix("bezier:") {
        return bx.rasterise(&bx.bezier_centreline(k.parse().unwrap()));
    }
    if let Some(p) = name.strip_prefix("euler:") {
        return bx.rasterise(&bx.euler_centreline(p.parse().unwrap()));
    }
    panic!("unknown design {name}");
}

/// Write a box mask: first line n, then n rows of '0'/'1' from b = n - 1 (top) to 0.
pub fn write_mask(path: &str, n: usize, s: &[f64]) -> std::io::Result<()> {
    let mut out = format!("{n}\n");
    for b in (0..n).rev() {
        out.extend((0..n).map(|a| if s[b * n + a] >= 0.5 { '1' } else { '0' }));
        out.push('\n');
    }
    fs::write(path, out)
}

pub fn read_mask(path: &str) -> std::io::Result<(usize, Vec<f64>)> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let rows: Vec<&str> = lines.take(n).collect();
    let mut s = vec![0.0; n * n];
    for (r, row) in rows.iter().enumerate() {
        let b = n - 1 - r;
        for (a, c) in row.bytes().enumerate() {
            s[b * n + a] = if c == b'1' { 1.0 } else { 0.0 };
        }
    }
    Ok((n, s))
}

pub fn write_values(path: &str, v: &[f64]) -> std::io::Result<()> {
    fs::write(path, v.iter().map(|x| format!("{x:.17e}")).collect::<Vec<_>>().join("\n"))
}

pub fn read_values(path: &str) -> std::io::Result<Vec<f64>> {
    Ok(fs::read_to_string(path)?.lines().map(|l| l.trim().parse().unwrap()).collect())
}

pub struct Best {
    pub j: f64,
    pub il: f64,
    pub rl: f64,
    pub mask: Vec<f64>,
    pub x: Vec<f64>,
}

/// Event log, cost counters and checkpoints of one run.
pub struct RunLog {
    file: fs::File,
    dir: String,
    n: usize,
    t0: Instant,
    pub events: usize,
    pub charged: usize,
    pub forward_solves: usize,
    pub adjoint_solves: usize,
    pub cache_hits: usize,
    pub forward_s: f64,
    pub adjoint_s: f64,
    pub cap: usize,
    pub scored: usize,
    pub geometry_ok: usize,
    pub gates_ok: usize,
    pub best: Option<Best>,
    checkpoints: Vec<usize>,
}

impl RunLog {
    pub fn new(dir: &str, n: usize, cap: usize, checkpoints: &[usize]) -> std::io::Result<Self> {
        fs::create_dir_all(dir)?;
        let mut file = fs::File::create(format!("{dir}/events.csv"))?;
        writeln!(file, "event,kind,charged,forward_solves,adjoint_solves,cache_hits,wall_s,forward_s,adjoint_s,j,geometry_ok,gates_ok,best_j,best_il,best_rl,note")?;
        Ok(RunLog {
            file,
            dir: dir.to_string(),
            n,
            t0: Instant::now(),
            events: 0,
            charged: 0,
            forward_solves: 0,
            adjoint_solves: 0,
            cache_hits: 0,
            forward_s: 0.0,
            adjoint_s: 0.0,
            cap,
            scored: 0,
            geometry_ok: 0,
            gates_ok: 0,
            best: None,
            checkpoints: checkpoints.to_vec(),
        })
    }

    /// True if `cost` more forward-equivalents fit under the cap.
    pub fn fits(&self, cost: usize) -> bool {
        self.charged + cost <= self.cap
    }

    pub fn best_j(&self) -> f64 {
        self.best.as_ref().map(|b| b.j).unwrap_or(f64::INFINITY)
    }

    fn line(&mut self, kind: &str, j: f64, geo: bool, gates: bool, note: &str) {
        self.events += 1;
        let (bj, bi, br) = self.best.as_ref().map(|b| (b.j, b.il, b.rl)).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
        writeln!(
            self.file,
            "{},{kind},{},{},{},{},{:.3},{:.3},{:.3},{j:.6},{},{},{bj:.6},{bi:.6},{br:.4},{note}",
            self.events,
            self.charged,
            self.forward_solves,
            self.adjoint_solves,
            self.cache_hits,
            self.t0.elapsed().as_secs_f64(),
            self.forward_s,
            self.adjoint_s,
            geo as u8,
            gates as u8
        )
        .unwrap();
        self.file.flush().unwrap();
    }

    fn checkpoint(&mut self, before: usize) {
        let cps: Vec<usize> = self.checkpoints.iter().copied().filter(|&c| before < c && self.charged >= c).collect();
        for c in cps {
            if let Some(b) = &self.best {
                write_mask(&format!("{}/best_at_{c}.mask", self.dir), self.n, &b.mask).unwrap();
            }
        }
    }

    /// Record a binary score (3 forward solves, or 0 for a cache hit or a
    /// mask that fails a geometry gate).
    pub fn score(&mut self, s: &Scored, mask: &[f64], x: &[f64], forward_s: f64, note: &str) {
        let before = self.charged;
        self.charged += s.solves;
        self.forward_solves += s.solves;
        self.forward_s += forward_s;
        if s.cached {
            self.cache_hits += 1;
        }
        self.scored += 1;
        let (mut il, mut rl, mut gates) = (f64::NAN, f64::NAN, false);
        if let Some(sp) = &s.spectra {
            self.geometry_ok += 1;
            il = sp.iter().map(|v| v.worst_il()).fold(0.0, f64::max);
            rl = sp.iter().map(|v| v.worst_rl()).fold(f64::INFINITY, f64::min);
            gates = il < IL_MAX_DB && rl >= RL_MIN_DB;
            if gates {
                self.gates_ok += 1;
            }
            if s.j < self.best_j() {
                self.best = Some(Best { j: s.j, il, rl, mask: mask.to_vec(), x: x.to_vec() });
            }
        }
        let _ = (il, rl);
        self.line("score", s.j, s.feasible, gates, note);
        self.checkpoint(before);
    }

    /// Record an adjoint iteration: `fwd` forward and `adj` adjoint solves,
    /// charged `charge` forward-equivalents.
    pub fn iteration(&mut self, fwd: usize, adj: usize, charge: usize, forward_s: f64, adjoint_s: f64, note: &str) {
        let before = self.charged;
        self.charged += charge;
        self.forward_solves += fwd;
        self.adjoint_solves += adj;
        self.forward_s += forward_s;
        self.adjoint_s += adjoint_s;
        self.line("iter", f64::NAN, false, false, note);
        self.checkpoint(before);
    }

    /// Summary JSON and the best mask and representation value.
    pub fn finish(&mut self, extra: &str) -> std::io::Result<()> {
        if let Some(b) = &self.best {
            write_mask(&format!("{}/best.mask", self.dir), self.n, &b.mask)?;
            write_values(&format!("{}/best_x.txt", self.dir), &b.x)?;
        }
        let (bj, bi, br) = self.best.as_ref().map(|b| (b.j, b.il, b.rl)).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
        let js = |v: f64| if v.is_finite() { format!("{v:.6}") } else { "null".into() };
        let summary = format!(
            "{{\n  \"charged\": {},\n  \"cap\": {},\n  \"forward_solves\": {},\n  \"adjoint_solves\": {},\n  \"cache_hits\": {},\n  \"wall_s\": {:.1},\n  \"forward_s\": {:.1},\n  \"adjoint_s\": {:.1},\n  \"scored\": {},\n  \"geometry_ok\": {},\n  \"gates_ok\": {},\n  \"best_j\": {},\n  \"best_il_db\": {},\n  \"best_rl_db\": {}{extra}\n}}\n",
            self.charged,
            self.cap,
            self.forward_solves,
            self.adjoint_solves,
            self.cache_hits,
            self.t0.elapsed().as_secs_f64(),
            self.forward_s,
            self.adjoint_s,
            self.scored,
            self.geometry_ok,
            self.gates_ok,
            js(bj),
            js(bi),
            js(br)
        );
        fs::write(format!("{}/summary.json", self.dir), summary)
    }
}

/// Command-line value `--name V`, or the default.
pub fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|p| a.get(p + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// splitmix64 uniform and Box-Muller normal.
pub struct Rng(pub u64);
impl Rng {
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}
