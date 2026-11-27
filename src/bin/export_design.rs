//! Export a Stage 5 design (latent density file) as GDSII and SVG.
//! The mask is the 0.5 level set of the filtered density, with a halo of the
//! fixed input and output slabs. Prints the area and round-trip checks.
//!
//! Usage: export_design RHO.txt [--n 48] [--rf 5] [--halo 16] [--cell-nm 10] [--out out/export/bend48]

use photon_fdtd::design::{ConicFilter, DesignBox};
use photon_fdtd::export::{Mask, area2, gds, group, rasterise, read_gds, svg, trace};
use photon_fdtd::sparams::{Geometry, PortSetup};

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).map(|p| a[p + 1].parse().ok().unwrap()).unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("RHO.txt");
    let (n, rf, halo, cell_nm): (usize, f64, usize, i64) = (arg("--n", 48), arg("--rf", 5.0), arg("--halo", 16), arg("--cell-nm", 10));
    let out: String = arg("--out", "out/export/bend48".to_string());
    if let Some(d) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(d)?;
    }
    let rho: Vec<f64> = std::fs::read_to_string(&path)?.lines().map(|l| l.trim().parse().unwrap()).collect();
    let bx = DesignBox::for_bend(&PortSetup::standard(Geometry::Bend, 1), n, 8);
    let filtered = ConicFilter::new(rf).apply(&bx, &rho);
    let bin: Vec<f64> = filtered.iter().map(|&v| if v >= 0.5 { 1.0 } else { 0.0 }).collect();
    let mask = Mask::from_box(&bx, &bin, halo);
    let groups = group(trace(&mask));
    let holes: usize = groups.iter().map(|g| g.1.len()).sum();
    let verts: usize = groups.iter().map(|g| g.0.len() + g.1.iter().map(|h| h.len()).sum::<usize>()).sum();
    let area: i64 = groups.iter().map(|g| area2(&g.0) + g.1.iter().map(area2).sum::<i64>()).sum::<i64>() / 2;
    let bytes = gds(&mask, cell_nm, "BEND", 1);
    std::fs::write(format!("{out}.gds"), &bytes)?;
    std::fs::write(format!("{out}.svg"), svg(&mask, "photon-fdtd bend (2D TMz design, white = silicon)"))?;
    let (polys, units) = read_gds(&bytes);
    let gds_area: i64 = polys.iter().map(area2).sum::<i64>() / 2;
    let back = rasterise(&polys, cell_nm, mask.nx, mask.ny);
    let mismatch = back.iter().zip(&mask.on).filter(|(a, b)| a != b).count();
    println!("canvas {} x {} cells (box {n}, halo {halo}), cell {cell_nm} nm, units {:e} um / {:e} m", mask.nx, mask.ny, units.0, units.1);
    println!("silicon cells {}, outer boundaries {}, holes {holes}, vertices {verts}", mask.cells(), groups.len());
    println!("traced area {area} cells, GDSII area {} cells, GDSII polygons {}", gds_area as f64 / (cell_nm * cell_nm) as f64, polys.len());
    println!("round trip (GDSII rasterised at cell centres vs mask): {mismatch} differing cells");
    println!("wrote {out}.gds ({} bytes), {out}.svg", bytes.len());
    if area != mask.cells() as i64 || gds_area != mask.cells() as i64 * cell_nm * cell_nm || mismatch != 0 {
        return Err("export check failed".into());
    }
    Ok(())
}
