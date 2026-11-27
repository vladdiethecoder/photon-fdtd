//! GDSII and SVG export of a binary design.
//!
//! The mask is a grid of cells (cell (a, b) spans [a, a+1] x [b, b+1], b up).
//! The boundary of the silicon is traced along the cell edges, so every edge
//! is horizontal or vertical (Manhattan) and every vertex is on the cell grid.
//! Each loop keeps the silicon on its left: outer boundaries are
//! counter-clockwise (positive area), holes are clockwise (negative area).
//! Silicon cells that touch only at a corner are separate (4-connectivity).
//!
//! GDSII has no polygons with holes, so each hole is joined to its outer
//! boundary by a zero-width horizontal cut (a "keyhole"). SVG uses one path
//! per outer boundary with its holes and fill-rule evenodd.
//!
//! The scale is a free choice: the solver uses natural units (1 cell = 1).
//! `cell_nm` sets the cell size in the file; the database unit is 1 nm.

use std::collections::HashMap;

/// A binary mask with `nx` columns and `ny` rows, row 0 at the bottom.
pub struct Mask {
    pub nx: usize,
    pub ny: usize,
    pub on: Vec<bool>,
}

pub type Ring = Vec<(i64, i64)>;

impl Mask {
    fn at(&self, a: i64, b: i64) -> bool {
        a >= 0 && b >= 0 && (a as usize) < self.nx && (b as usize) < self.ny && self.on[b as usize * self.nx + a as usize]
    }

    pub fn cells(&self) -> usize {
        self.on.iter().filter(|v| **v).count()
    }

    /// The design box with a halo of `halo` cells of the fixed exterior
    /// (the input and output slabs), from box densities (>= 0.5 = silicon).
    pub fn from_box(bx: &crate::design::DesignBox, s: &[f64], halo: usize) -> Mask {
        let (n, h) = (bx.n as i64, halo as i64);
        let m = (n + 2 * h) as usize;
        let on = (0..m * m)
            .map(|k| {
                let (a, b) = ((k % m) as i64 - h, (k / m) as i64 - h);
                if (0..n).contains(&a) && (0..n).contains(&b) { s[(b * n + a) as usize] >= 0.5 } else { bx.exterior(a, b) >= 0.5 }
            })
            .collect();
        Mask { nx: m, ny: m, on }
    }
}

/// Twice the signed area (shoelace), exact in integers.
pub fn area2(r: &Ring) -> i64 {
    (0..r.len()).map(|i| {
        let (p, q) = (r[i], r[(i + 1) % r.len()]);
        p.0 * q.1 - q.0 * p.1
    }).sum()
}

/// Trace every boundary loop. Collinear vertices are removed.
pub fn trace(mask: &Mask) -> Vec<Ring> {
    // Directed edges with silicon on the left, keyed by start vertex.
    let mut out: HashMap<(i64, i64), Vec<(i64, i64)>> = HashMap::new();
    for b in 0..mask.ny as i64 {
        for a in 0..mask.nx as i64 {
            if !mask.at(a, b) {
                continue;
            }
            let mut add = |p: (i64, i64), q: (i64, i64)| {
                out.entry(p).or_default().push(q);
            };
            if !mask.at(a, b - 1) { add((a, b), (a + 1, b)); }
            if !mask.at(a + 1, b) { add((a + 1, b), (a + 1, b + 1)); }
            if !mask.at(a, b + 1) { add((a + 1, b + 1), (a, b + 1)); }
            if !mask.at(a - 1, b) { add((a, b + 1), (a, b)); }
        }
    }
    let mut starts: Vec<(i64, i64)> = out.keys().copied().collect();
    starts.sort();
    let mut rings = Vec::new();
    for s in starts {
        while let Some(first) = out.get_mut(&s).and_then(|v| v.pop()) {
            let mut pts = vec![s];
            let (mut prev, mut cur) = (s, first);
            loop {
                let d = (cur.0 - prev.0, cur.1 - prev.1);
                // At a corner where two silicon cells touch diagonally there are
                // two exits: take the left turn, so the loop hugs its own cell.
                let left = (cur.0 - d.1, cur.1 + d.0);
                if cur == s && left == first {
                    break;
                }
                pts.push(cur);
                let cand = out.get_mut(&cur).unwrap();
                let k = cand.iter().position(|q| *q == left).unwrap_or(0);
                let next = cand.swap_remove(k);
                (prev, cur) = (cur, next);
                if cur == s && out[&s].is_empty() {
                    break;
                }
            }
            rings.push(simplify(pts));
        }
    }
    rings
}

fn simplify(pts: Ring) -> Ring {
    let n = pts.len();
    (0..n)
        .filter(|&i| {
            let (p, c, q) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
            (c.0 - p.0) * (q.1 - c.1) - (c.1 - p.1) * (q.0 - c.0) != 0
        })
        .map(|i| pts[i])
        .collect()
}

/// Even-odd point-in-polygon test for a point off the grid lines.
fn inside(r: &Ring, x: f64, y: f64) -> bool {
    let mut c = false;
    for i in 0..r.len() {
        let (p, q) = (r[i], r[(i + 1) % r.len()]);
        let (py, qy) = (p.1 as f64, q.1 as f64);
        if (py > y) != (qy > y) {
            let xi = p.0 as f64 + (y - py) / (qy - py) * (q.0 - p.0) as f64;
            if x < xi {
                c = !c;
            }
        }
    }
    c
}

/// Outer boundaries, each with the holes inside it.
pub fn group(rings: Vec<Ring>) -> Vec<(Ring, Vec<Ring>)> {
    let (outer, holes): (Vec<Ring>, Vec<Ring>) = rings.into_iter().partition(|r| area2(r) > 0);
    let mut groups: Vec<(Ring, Vec<Ring>)> = outer.into_iter().map(|r| (r, Vec::new())).collect();
    for h in holes {
        // A point just inside the hole: next to its lowest-left vertex.
        let v = *h.iter().min_by_key(|p| (p.0, p.1)).unwrap();
        let (x, y) = (v.0 as f64 + 0.5, v.1 as f64 + 0.5);
        // The innermost outer boundary that contains it (smallest area).
        let g = groups
            .iter_mut()
            .filter(|(o, _)| inside(o, x, y))
            .min_by_key(|(o, _)| area2(o))
            .expect("hole without an outer boundary");
        g.1.push(h);
    }
    groups
}

/// Join the holes to the outer boundary with zero-width horizontal cuts.
/// Holes are joined in order of their leftmost x, so the cut to the left
/// from a hole always ends on the boundary that is already merged.
pub fn keyhole(outer: &Ring, holes: &[Ring]) -> Ring {
    let mut ring = outer.clone();
    let mut hs: Vec<&Ring> = holes.iter().collect();
    hs.sort_by_key(|h| h.iter().map(|p| (p.0, p.1)).min().unwrap());
    for h in hs {
        let vi = (0..h.len()).min_by_key(|&i| (h[i].0, h[i].1)).unwrap();
        let v = h[vi];
        // Nearest vertical edge to the left that spans y = v.1 (half-open).
        let mut best: Option<(i64, usize)> = None;
        for i in 0..ring.len() {
            let (p, q) = (ring[i], ring[(i + 1) % ring.len()]);
            if p.0 == q.0 && p.0 < v.0 && p.1.min(q.1) <= v.1 && v.1 < p.1.max(q.1) && best.is_none_or(|(x, _)| p.0 > x) {
                best = Some((p.0, i));
            }
        }
        let (x, i) = best.expect("no boundary to the left of a hole");
        let cut = (x, v.1);
        let mut joined: Ring = ring[..=i].to_vec();
        if joined.last() != Some(&cut) {
            joined.push(cut);
        }
        joined.extend((0..=h.len()).map(|k| h[(vi + k) % h.len()]));
        joined.push(cut);
        joined.extend_from_slice(&ring[i + 1..]);
        ring = joined;
    }
    ring
}

// ---- GDSII ----

fn rec(buf: &mut Vec<u8>, kind: u16, data: &[u8]) {
    let len = 4 + data.len();
    assert!(len <= 65534, "GDSII record too long ({len} bytes)");
    buf.extend_from_slice(&(len as u16).to_be_bytes());
    buf.extend_from_slice(&kind.to_be_bytes());
    buf.extend_from_slice(data);
}

fn ascii(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    if v.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// GDSII 8-byte real: sign, excess-64 base-16 exponent, 56-bit mantissa.
pub fn gds_real(v: f64) -> [u8; 8] {
    if v == 0.0 {
        return [0; 8];
    }
    let (sign, mut m) = (if v < 0.0 { 0x80u8 } else { 0 }, v.abs());
    let mut e = 64i32;
    while m >= 1.0 {
        m /= 16.0;
        e += 1;
    }
    while m < 1.0 / 16.0 {
        m *= 16.0;
        e -= 1;
    }
    let mant = (m * 2f64.powi(56)).round() as u64;
    let mut out = [0u8; 8];
    out[0] = sign | e as u8;
    out[1..].copy_from_slice(&mant.to_be_bytes()[1..]);
    out
}

pub fn gds_real_decode(b: &[u8]) -> f64 {
    let e = (b[0] & 0x7f) as i32 - 64;
    let mut m = 0u64;
    for &x in &b[1..8] {
        m = (m << 8) | x as u64;
    }
    let v = m as f64 / 2f64.powi(56) * 16f64.powi(e);
    if b[0] & 0x80 != 0 { -v } else { v }
}

/// GDSII library with one structure; one BOUNDARY per outer boundary (holes
/// keyholed), on `layer`/datatype 0. Coordinates in nm. The dates are fixed,
/// so the output is deterministic.
pub fn gds(mask: &Mask, cell_nm: i64, name: &str, layer: i16) -> Vec<u8> {
    let date: Vec<u8> = [2026i16, 11, 26, 0, 0, 0, 2026, 11, 26, 0, 0, 0].iter().flat_map(|v| v.to_be_bytes()).collect();
    let mut b = Vec::new();
    rec(&mut b, 0x0002, &600i16.to_be_bytes());
    rec(&mut b, 0x0102, &date);
    rec(&mut b, 0x0206, &ascii("PHOTON_FDTD"));
    let mut units = gds_real(1e-3).to_vec(); // database unit in user units (um)
    units.extend_from_slice(&gds_real(1e-9)); // database unit in metres
    rec(&mut b, 0x0305, &units);
    rec(&mut b, 0x0502, &date);
    rec(&mut b, 0x0606, &ascii(name));
    for (outer, holes) in group(trace(mask)) {
        let ring = keyhole(&outer, &holes);
        let mut xy = Vec::with_capacity(8 * (ring.len() + 1));
        for p in ring.iter().chain(std::iter::once(&ring[0])) {
            xy.extend_from_slice(&((p.0 * cell_nm) as i32).to_be_bytes());
            xy.extend_from_slice(&((p.1 * cell_nm) as i32).to_be_bytes());
        }
        assert!(ring.len() < 8191, "polygon has {} vertices (GDSII limit 8191)", ring.len());
        rec(&mut b, 0x0800, &[]);
        rec(&mut b, 0x0D02, &layer.to_be_bytes());
        rec(&mut b, 0x0E02, &0i16.to_be_bytes());
        rec(&mut b, 0x1003, &xy);
        rec(&mut b, 0x1100, &[]);
    }
    rec(&mut b, 0x0700, &[]);
    rec(&mut b, 0x0400, &[]);
    b
}

/// Read back the BOUNDARY polygons (closing vertex removed) and the two UNITS
/// values from a GDSII stream.
pub fn read_gds(bytes: &[u8]) -> (Vec<Ring>, (f64, f64)) {
    let (mut i, mut polys, mut units) = (0, Vec::new(), (0.0, 0.0));
    while i + 4 <= bytes.len() {
        let len = u16::from_be_bytes([bytes[i], bytes[i + 1]]) as usize;
        let kind = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]);
        let d = &bytes[i + 4..i + len];
        match kind {
            0x0305 => units = (gds_real_decode(&d[..8]), gds_real_decode(&d[8..16])),
            0x1003 => {
                let mut r: Ring = d
                    .chunks(8)
                    .map(|c| (i32::from_be_bytes(c[..4].try_into().unwrap()) as i64, i32::from_be_bytes(c[4..].try_into().unwrap()) as i64))
                    .collect();
                assert_eq!(r.first(), r.last(), "GDSII boundary not closed");
                r.pop();
                polys.push(r);
            }
            0x0400 => break,
            _ => {}
        }
        i += len;
    }
    (polys, units)
}

/// Rasterise polygons at the cell centres (even-odd), for the round-trip check.
pub fn rasterise(polys: &[Ring], cell_nm: i64, nx: usize, ny: usize) -> Vec<bool> {
    let c = cell_nm as f64;
    (0..nx * ny)
        .map(|k| {
            let (x, y) = (((k % nx) as f64 + 0.5) * c, ((k / nx) as f64 + 0.5) * c);
            polys.iter().filter(|r| inside(r, x, y)).count() % 2 == 1
        })
        .collect()
}

/// SVG: one path per outer boundary with its holes, fill-rule evenodd,
/// y flipped so that the image shows b up. Units: cells.
pub fn svg(mask: &Mask, title: &str) -> String {
    let (w, h) = (mask.nx as i64, mask.ny as i64);
    let mut s = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{}\" height=\"{}\">\n<title>{title}</title>\n<rect width=\"{w}\" height=\"{h}\" fill=\"black\"/>\n",
        8 * w,
        8 * h
    );
    for (outer, holes) in group(trace(mask)) {
        let mut d = String::new();
        for r in std::iter::once(&outer).chain(&holes) {
            d += &format!("M{} {}", r[0].0, h - r[0].1);
            for p in &r[1..] {
                d += &format!(" L{} {}", p.0, h - p.1);
            }
            d += " Z ";
        }
        s += &format!("<path d=\"{}\" fill=\"white\" fill-rule=\"evenodd\"/>\n", d.trim_end());
    }
    s + "</svg>\n"
}
