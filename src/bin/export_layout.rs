//! Layout export of a Stage 6 design (docs/stage6-protocol.md, section 10).
//!
//!   export_layout --mask DESIGN_s1.mask [--halo 16] [--cell-nm 10] --out PREFIX
//!
//! Layers (datatype 0): 1 nominal silicon, 2 eroded silicon, 3 dilated
//! silicon (the bias rule of section 2 at scale 1), 10 design-box outline.
//! Each silicon layer holds the box and a halo of the fixed slabs. Origin: the
//! lower-left corner of the halo; x right, y up; integer cell coordinates
//! times `cell_nm`; database unit 1 nm, user unit 1 um. The cell size is an
//! illustrative mapping, not a calibrated physical scale.
//!
//! Writes PREFIX.gds, PREFIX.svg, PREFIX_layerL.mask (the source canvas of each
//! silicon layer, the reference of tools/check_layout.py) and PREFIX_meta.json.

use std::fmt::Write as _;

use photon_fdtd::design::DesignBox;
use photon_fdtd::export::{Mask, area2, gds_layers, group, rasterise, read_gds, svg, trace};
use photon_fdtd::robust::{geometry, variants};
use photon_fdtd::sparams::{Geometry as G, PortSetup};
use photon_fdtd::stage6::{arg, read_mask};

fn canvas_text(m: &Mask) -> String {
    let mut s = format!("{} {}\n", m.nx, m.ny);
    for b in (0..m.ny).rev() {
        s.extend((0..m.nx).map(|a| if m.on[b * m.nx + a] { '1' } else { '0' }));
        s.push('\n');
    }
    s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path: String = arg("--mask", String::new());
    let halo: usize = arg("--halo", 16);
    let cell_nm: i64 = arg("--cell-nm", 10);
    let out: String = arg("--out", "out/stage6/layout/bend80".to_string());
    if let Some(d) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(d)?;
    }
    let (n, nominal) = read_mask(&path)?;
    let bx = DesignBox::for_bend(&PortSetup::standard(G::Bend, 1), n, 8);
    let vs = variants(&bx, &nominal, 1);
    let masks: Vec<Mask> = [&vs[1], &vs[0], &vs[2]].iter().map(|v| Mask::from_box(&bx, v, halo)).collect();
    let layers: [i16; 3] = [1, 2, 3];
    let (h, nn) = (halo as i64, n as i64);
    let bytes = gds_layers(&[(1, &masks[0]), (2, &masks[1]), (3, &masks[2])], &[(10, (h, h, h + nn, h + nn))], cell_nm, "BEND80");
    std::fs::write(format!("{out}.gds"), &bytes)?;
    std::fs::write(format!("{out}.svg"), svg(&masks[0], "photon-fdtd 80 x 80 bend, nominal silicon (2D TMz design; white = silicon; illustrative scale)"))?;
    // Rust-side checks (the independent check is tools/check_layout.py).
    let (polys, units) = read_gds(&bytes);
    let mut meta = String::new();
    writeln!(meta, "{{")?;
    writeln!(meta, "  \"source_mask\": \"{path}\",")?;
    writeln!(meta, "  \"box_cells\": {n}, \"halo_cells\": {halo}, \"canvas_cells\": [{}, {}],", masks[0].nx, masks[0].ny)?;
    writeln!(meta, "  \"origin\": \"lower-left corner of the halo\", \"x_axis\": \"right\", \"y_axis\": \"up\",")?;
    writeln!(meta, "  \"database_unit_m\": {:e}, \"user_unit_per_database_unit\": {:e}, \"cell_nm\": {cell_nm}, \"scale_note\": \"illustrative mapping, not calibrated\",", units.1, units.0)?;
    writeln!(meta, "  \"layers\": {{\"1/0\": \"nominal silicon\", \"2/0\": \"eroded silicon (L1 radius 1)\", \"3/0\": \"dilated silicon (L1 radius 1)\", \"10/0\": \"design-box outline\"}},")?;
    writeln!(meta, "  \"box_outline_cells\": [{h}, {h}, {}, {}],", h + nn, h + nn)?;
    writeln!(meta, "  \"rust_checks\": {{")?;
    let names = ["nominal", "eroded", "dilated"];
    for (i, (m, layer)) in masks.iter().zip(layers).enumerate() {
        std::fs::write(format!("{out}_layer{layer}.mask"), canvas_text(m))?;
        let groups = group(trace(m));
        let holes: usize = groups.iter().map(|g| g.1.len()).sum();
        let traced: i64 = groups.iter().map(|g| area2(&g.0) + g.1.iter().map(area2).sum::<i64>()).sum::<i64>() / 2;
        let g = geometry(&bx, [&vs[1], &vs[0], &vs[2]][i], 1);
        println!(
            "layer {layer} ({}): cells {}, outer boundaries {}, holes {holes}, traced area {traced}; geometry {g:?}",
            names[i],
            m.cells(),
            groups.len()
        );
        writeln!(
            meta,
            "    \"{layer}\": {{\"variant\": \"{}\", \"cells\": {}, \"outer_boundaries\": {}, \"holes\": {holes}, \"traced_area_cells\": {traced}, \"ports_connected\": {}, \"floating_silicon\": {}, \"enclosed_air\": {}, \"width_violations\": {}, \"gap_violations\": {}, \"min_width\": {}, \"min_gap\": {}}}{}",
            names[i], m.cells(), groups.len(), g.ports_connected, g.floating_silicon, g.enclosed_air, g.width_violations, g.gap_violations, g.min_width, g.min_gap,
            if i < 2 { "," } else { "" }
        )?;
        if traced != m.cells() as i64 {
            return Err(format!("layer {layer}: traced area differs from the cell count").into());
        }
    }
    writeln!(meta, "  }}")?;
    writeln!(meta, "}}")?;
    std::fs::write(format!("{out}_meta.json"), meta)?;
    // Own round trip of layer 1 (polygons are read in layer order).
    let n1 = group(trace(&masks[0])).len();
    let back = rasterise(&polys[..n1], cell_nm, masks[0].nx, masks[0].ny);
    let mismatch = back.iter().zip(&masks[0].on).filter(|(a, b)| a != b).count();
    println!("own round trip, layer 1: {mismatch} differing cells; {} polygons in the file; wrote {out}.gds ({} bytes), {out}.svg", polys.len(), bytes.len());
    if mismatch != 0 {
        return Err("own round trip failed".into());
    }
    Ok(())
}
