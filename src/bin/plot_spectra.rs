//! Plot a spectra CSV from stage5_validate (design,lambda0,IL_dB,RL_dB) as an
//! SVG with two panels: insertion loss and return loss against lambda0.
//! Dashed lines mark the Stage 5 gates (IL 0.50 dB, RL 20 dB).
//!
//! Usage: plot_spectra SPECTRA.csv OUT.svg [TITLE]

use std::collections::BTreeMap;
use std::fmt::Write;

const W: f64 = 900.0;
const PANEL_H: f64 = 300.0;
const LEFT: f64 = 80.0;
const RIGHT: f64 = 200.0;
const TOP: f64 = 50.0;
const GAP: f64 = 70.0;
const COLOURS: [&str; 6] = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"];

struct Panel {
    y0: f64,
    lo: f64,
    hi: f64,
    label: &'static str,
    gate: f64,
}

fn nice_step(range: f64) -> f64 {
    let raw = range / 6.0;
    let p = 10f64.powf(raw.log10().floor());
    [1.0, 2.0, 5.0, 10.0].iter().map(|m| m * p).find(|s| *s >= raw).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let (csv, out) = (&a[1], &a[2]);
    let title = a.get(3).cloned().unwrap_or_else(|| "Modal spectra (2D TMz)".into());
    let mut data: BTreeMap<String, Vec<(f64, f64, f64)>> = BTreeMap::new();
    let mut order = Vec::new();
    for line in std::fs::read_to_string(csv)?.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if !data.contains_key(f[0]) {
            order.push(f[0].to_string());
        }
        data.entry(f[0].to_string()).or_default().push((f[1].parse()?, f[2].parse()?, f[3].parse()?));
    }
    let all: Vec<&(f64, f64, f64)> = data.values().flatten().collect();
    let (xlo, xhi) = all.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| (l.min(p.0), h.max(p.0)));
    let il_hi = all.iter().map(|p| p.1).fold(0.5, f64::max) * 1.1;
    let rl_lo = all.iter().map(|p| p.2).fold(20.0, f64::min).min(20.0) - 2.0;
    let rl_hi = all.iter().map(|p| p.2).fold(0.0, f64::max) + 2.0;
    let panels = [
        Panel { y0: TOP, lo: 0.0, hi: il_hi, label: "insertion loss (dB)", gate: 0.5 },
        Panel { y0: TOP + PANEL_H + GAP, lo: rl_lo, hi: rl_hi, label: "return loss (dB)", gate: 20.0 },
    ];
    let pw = W - LEFT - RIGHT;
    let x = |v: f64| LEFT + (v - xlo) / (xhi - xlo) * pw;
    let total_h = TOP + 2.0 * PANEL_H + GAP + 60.0;
    let mut s = String::new();
    writeln!(s, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" font-family="sans-serif" font-size="13">"#)?;
    writeln!(s, r#"<rect width="100%" height="100%" fill="white"/>"#)?;
    writeln!(s, r#"<text x="{}" y="28" font-size="16" text-anchor="middle">{title}</text>"#, LEFT + pw / 2.0)?;
    for p in &panels {
        let y = |v: f64| p.y0 + PANEL_H - (v - p.lo) / (p.hi - p.lo) * PANEL_H;
        writeln!(s, r#"<rect x="{LEFT}" y="{}" width="{pw}" height="{PANEL_H}" fill="none" stroke="black"/>"#, p.y0)?;
        let st = nice_step(p.hi - p.lo);
        let mut t = (p.lo / st).ceil() * st;
        while t <= p.hi + 1e-12 {
            writeln!(s, r##"<line x1="{LEFT}" x2="{0}" y1="{1}" y2="{1}" stroke="#ddd"/><text x="{2}" y="{3}" text-anchor="end">{4}</text>"##, LEFT + pw, y(t), LEFT - 6.0, y(t) + 4.0, (t * 1000.0).round() / 1000.0)?;
            t += st;
        }
        let mut lx = (xlo / 10.0).ceil() * 10.0;
        while lx <= xhi + 1e-9 {
            writeln!(s, r#"<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke="black"/><text x="{0}" y="{3}" text-anchor="middle">{4}</text>"#, x(lx), p.y0 + PANEL_H, p.y0 + PANEL_H + 5.0, p.y0 + PANEL_H + 20.0, lx)?;
            lx += 10.0;
        }
        writeln!(s, r#"<line x1="{LEFT}" x2="{0}" y1="{1}" y2="{1}" stroke="black" stroke-dasharray="6,4"/><text x="{2}" y="{3}">gate {4}</text>"#, LEFT + pw, y(p.gate), LEFT + pw + 6.0, y(p.gate) + 4.0, p.gate)?;
        writeln!(s, r#"<text transform="translate(22,{}) rotate(-90)" text-anchor="middle">{}</text>"#, p.y0 + PANEL_H / 2.0, p.label)?;
        writeln!(s, r#"<text x="{}" y="{}" text-anchor="middle">lambda0 (cells)</text>"#, LEFT + pw / 2.0, p.y0 + PANEL_H + 38.0)?;
        for (i, name) in order.iter().enumerate() {
            let v = &data[name];
            let pts: Vec<String> = v.iter().map(|q| format!("{:.2},{:.2}", x(q.0), y(if p.gate < 1.0 { q.1 } else { q.2 }))).collect();
            writeln!(s, r#"<polyline points="{}" fill="none" stroke="{}" stroke-width="2"/>"#, pts.join(" "), COLOURS[i % 6])?;
        }
    }
    for (i, name) in order.iter().enumerate() {
        let ly = TOP + 20.0 + 22.0 * i as f64;
        writeln!(s, r#"<line x1="{0}" x2="{1}" y1="{ly}" y2="{ly}" stroke="{2}" stroke-width="3"/><text x="{3}" y="{4}">{name}</text>"#, LEFT + pw + 60.0, LEFT + pw + 85.0, COLOURS[i % 6], LEFT + pw + 90.0, ly + 4.0)?;
    }
    s.push_str("</svg>\n");
    std::fs::write(out, s)?;
    println!("wrote {out}: {} designs, {} points each", order.len(), data[&order[0]].len());
    Ok(())
}
