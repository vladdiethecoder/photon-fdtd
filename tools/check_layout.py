#!/usr/bin/env python3
"""Independent layout check and certificate (docs/stage6-protocol.md, section 10).

This reader shares no code with the Rust writer (src/export.rs). It parses
the GDSII stream, rasterises each layer exactly with integer arithmetic,
and compares it with the source canvas masks that export_layout wrote.

Checks, for layers 1, 2 and 3:
  - every BOUNDARY is closed, Manhattan, and on the cell grid;
  - exact union area (cells) = sum of polygon areas (no overlap) = source cell count;
  - zero symmetric difference against the source mask;
  - width and gap: an opening and a closing with the 5-cell disk
    (dx^2 + dy^2 <= 4) change no cell of the design box;
  - connectivity: the input slab connects to the output slab, no silicon
    component floats, and enclosed air holes are counted.
Negative control: the source mask shifted by one cell (equal area) must give
a non-zero symmetric difference against the imported layer.

Usage:
  tools/check_layout.py PREFIX --design-revision REV [--extra FILE ...] [--cert OUT.json]

Only numpy and the Python standard library are used.
"""

import argparse
import hashlib
import json
import platform
import shlex
import struct
import subprocess
import sys
from collections import deque

import numpy as np

CLAIM = "layout-verified for the stated 2D rules"
NOT_CLAIMED = ["PDK signoff", "3D SOI performance", "fabrication readiness", "a calibrated physical scale"]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def real8(b):
    """GDSII 8-byte real: sign bit, excess-64 base-16 exponent, 56-bit mantissa."""
    sign = -1.0 if b[0] & 0x80 else 1.0
    exponent = (b[0] & 0x7F) - 64
    mantissa = int.from_bytes(b[1:8], "big") / float(1 << 56)
    return sign * mantissa * 16.0 ** exponent


def parse_gds(data):
    units, polys, cur, i = None, [], None, 0
    names = {0x0002: "HEADER", 0x0102: "BGNLIB", 0x0206: "LIBNAME", 0x0305: "UNITS", 0x0502: "BGNSTR",
             0x0606: "STRNAME", 0x0800: "BOUNDARY", 0x0D02: "LAYER", 0x0E02: "DATATYPE", 0x1003: "XY",
             0x1100: "ENDEL", 0x0700: "ENDSTR", 0x0400: "ENDLIB"}
    seen = []
    while i + 4 <= len(data):
        length, rtype = struct.unpack(">HH", data[i:i + 4])
        if length < 4:
            raise ValueError(f"bad record length {length} at byte {i}")
        body = data[i + 4:i + length]
        seen.append(names.get(rtype, hex(rtype)))
        if rtype == 0x0305:
            units = (real8(body[0:8]), real8(body[8:16]))
        elif rtype == 0x0800:
            cur = {"layer": None, "datatype": None, "xy": None}
        elif rtype == 0x0D02:
            cur["layer"] = struct.unpack(">h", body)[0]
        elif rtype == 0x0E02:
            cur["datatype"] = struct.unpack(">h", body)[0]
        elif rtype == 0x1003:
            n = len(body) // 8
            cur["xy"] = [struct.unpack(">ii", body[8 * k:8 * k + 8]) for k in range(n)]
        elif rtype == 0x1100:
            polys.append(cur)
            cur = None
        elif rtype == 0x0400:
            break
        i += length
    return units, polys, seen


def read_canvas(path):
    with open(path) as f:
        nx, ny = map(int, f.readline().split())
        rows = [f.readline().strip() for _ in range(ny)]
    m = np.zeros((ny, nx), dtype=bool)  # m[y, x], y up
    for r, row in enumerate(rows):
        m[ny - 1 - r, :] = [c == "1" for c in row]
    return m


def raster(poly_cells, nx, ny):
    """Even-odd raster at cell centres. A vertical edge at x = X from y0 to y1
    crosses the rightward ray from the centre (a + 1/2, b + 1/2) if and only if
    X > a and min(y0, y1) <= b < max(y0, y1). Exact in integers."""
    inside = np.zeros((ny, nx), dtype=bool)
    n = len(poly_cells)
    for k in range(n):
        (x0, y0), (x1, y1) = poly_cells[k], poly_cells[(k + 1) % n]
        if x0 == x1 and y0 != y1:
            lo, hi = min(y0, y1), max(y0, y1)
            lo, hi, xe = max(lo, 0), min(hi, ny), min(max(x0, 0), nx)
            if lo < hi and xe > 0:
                inside[lo:hi, :xe] ^= True
    return inside


def shoelace2(p):
    return sum(p[k][0] * p[(k + 1) % len(p)][1] - p[(k + 1) % len(p)][0] * p[k][1] for k in range(len(p)))


DISK5 = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3) if dx * dx + dy * dy <= 4]


def morph(m, offsets, dilate):
    """Erosion (all) or dilation (any) over the offsets. Outside the canvas,
    each row and column continues its edge value (the slabs are straight)."""
    r = max(max(abs(dx), abs(dy)) for dx, dy in offsets)
    p = np.pad(m, r, mode="edge")
    ny, nx = m.shape
    acc = np.zeros_like(m) if dilate else np.ones_like(m)
    for dx, dy in offsets:
        sh = p[r + dy:r + dy + ny, r + dx:r + dx + nx]
        acc = (acc | sh) if dilate else (acc & sh)
    return acc


def width_gap(m, box):
    refl = [(-dx, -dy) for dx, dy in DISK5]
    opened = morph(morph(m, DISK5, False), refl, True)
    closed = morph(morph(m, DISK5, True), refl, False)
    (x0, y0, x1, y1) = box
    sl = (slice(y0, y1), slice(x0, x1))
    return int((opened[sl] != m[sl]).sum()), int((closed[sl] != m[sl]).sum())


def components(m, phase):
    ny, nx = m.shape
    label = -np.ones((ny, nx), dtype=int)
    comps = []
    for y in range(ny):
        for x in range(nx):
            if m[y, x] != phase or label[y, x] >= 0:
                continue
            cid, q, cells, edge = len(comps), deque([(y, x)]), [], set()
            label[y, x] = cid
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                if cx == 0:
                    edge.add("left")
                if cy == ny - 1:
                    edge.add("top")
                if cx == nx - 1 or cy == 0:
                    edge.add("other")
                for yy, xx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= yy < ny and 0 <= xx < nx and label[yy, xx] < 0 and m[yy, xx] == phase:
                        label[yy, xx] = cid
                        q.append((yy, xx))
            comps.append({"cells": len(cells), "edges": edge})
    return comps


def connectivity(m):
    si = components(m, True)
    air = components(m, False)
    connected = any({"left", "top"} <= c["edges"] for c in si)
    floating = sum(1 for c in si if not c["edges"])
    enclosed = sum(1 for c in air if not c["edges"])
    return connected, floating, enclosed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--design-revision", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--cert", default=None)
    args = ap.parse_args()
    pre = args.prefix
    meta = json.load(open(pre + "_meta.json"))
    cell = meta["cell_nm"]
    data = open(pre + ".gds", "rb").read()
    units, polys, seen = parse_gds(data)
    ok = True
    report = {"records": {"first": seen[0], "last": seen[-1], "boundaries": len(polys)}}
    units_ok = units is not None and abs(units[0] - 1e-3) < 1e-15 and abs(units[1] - 1e-9) < 1e-21
    report["units"] = {"user_unit_per_db_unit": units[0], "db_unit_m": units[1], "ok": units_ok}
    ok &= units_ok
    nx, ny = meta["canvas_cells"]
    bx0, by0, bx1, by1 = meta["box_outline_cells"]
    by_layer = {}
    grid_ok = True
    for p in polys:
        xy = p["xy"]
        closed = xy[0] == xy[-1]
        pts = xy[:-1]
        manhattan = all(pts[k][0] == pts[(k + 1) % len(pts)][0] or pts[k][1] == pts[(k + 1) % len(pts)][1] for k in range(len(pts)))
        on_grid = all(x % cell == 0 and y % cell == 0 for x, y in pts)
        grid_ok &= closed and manhattan and on_grid and p["datatype"] == 0
        by_layer.setdefault(p["layer"], []).append([(x // cell, y // cell) for x, y in pts])
    report["polygons_closed_manhattan_on_grid"] = grid_ok
    ok &= grid_ok
    outline = by_layer.get(10, [])
    outline_ok = len(outline) == 1 and sorted(outline[0]) == sorted([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)])
    report["box_outline_ok"] = outline_ok
    ok &= outline_ok
    layers = {}
    for layer in (1, 2, 3):
        src = read_canvas(f"{pre}_layer{layer}.mask")
        rasters = [raster(p, nx, ny) for p in by_layer.get(layer, [])]
        union = np.zeros((ny, nx), dtype=bool)
        for r in rasters:
            union |= r
        poly_sum = int(sum(r.sum() for r in rasters))
        shoelace = sum(abs(shoelace2(p)) for p in by_layer.get(layer, [])) // 2
        sym = int((union ^ src).sum())
        w, g = width_gap(union, (bx0, by0, bx1, by1))
        conn, floating, enclosed = connectivity(union)
        res = {
            "polygons": len(rasters),
            "union_area_cells": int(union.sum()),
            "union_area_nm2": int(union.sum()) * cell * cell,
            "sum_polygon_raster_cells": poly_sum,
            "sum_polygon_shoelace_cells": shoelace,
            "source_cells": int(src.sum()),
            "symmetric_difference_cells": sym,
            "width_violations": w,
            "gap_violations": g,
            "ports_connected": conn,
            "floating_silicon": floating,
            "enclosed_air": enclosed,
            "rust_meta": meta["rust_checks"][str(layer)],
        }
        res["area_ok"] = res["union_area_cells"] == poly_sum == shoelace == res["source_cells"]
        res["symmetric_difference_ok"] = sym == 0
        res["agrees_with_rust"] = (w, g, conn, floating, enclosed) == (
            res["rust_meta"]["width_violations"], res["rust_meta"]["gap_violations"], res["rust_meta"]["ports_connected"],
            res["rust_meta"]["floating_silicon"], res["rust_meta"]["enclosed_air"])
        res["rules_ok"] = w == 0 and g == 0 and conn and floating == 0
        ok &= res["area_ok"] and res["symmetric_difference_ok"] and res["agrees_with_rust"]
        if layer == 1:
            ok &= res["rules_ok"]
            shifted = np.roll(src, 1, axis=1)
            neg = int((union ^ shifted).sum())
            report["negative_control"] = {
                "description": "layer-1 source mask shifted by +1 cell in x (np.roll, equal area)",
                "shifted_cells": int(shifted.sum()),
                "equal_area": int(shifted.sum()) == int(src.sum()),
                "symmetric_difference_cells": neg,
                "detected": neg > 0,
            }
            ok &= report["negative_control"]["equal_area"] and report["negative_control"]["detected"]
        layers[str(layer)] = res
    report["layers"] = layers
    files = {f: sha256(f) for f in [pre + ".gds", pre + ".svg", pre + "_meta.json"] + [f"{pre}_layer{l}.mask" for l in (1, 2, 3)] + args.extra}
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src", "tools", "Cargo.toml", "Cargo.lock"]) != 0
    except Exception:
        rev, dirty = "unknown", True
    cert = {
        "claim": CLAIM,
        "passed": bool(ok),
        "not_claimed": NOT_CLAIMED,
        "model": "2D TMz, natural units",
        "solver_revision": rev,
        "solver_tree_dirty": dirty,
        "design_revision": args.design_revision,
        "files_sha256": files,
        "units": {"database_unit_m": 1e-9, "user_unit_m": 1e-6, "cell_nm": cell, "scale_note": meta["scale_note"]},
        "geometry_frame": {"origin": meta["origin"], "x_axis": meta["x_axis"], "y_axis": meta["y_axis"],
                           "canvas_cells": meta["canvas_cells"], "box_outline_cells": meta["box_outline_cells"]},
        "layers": meta["layers"],
        "rules": {
            "contours": "Manhattan cell-edge contours, closed, integer coordinates = cell index x cell_nm",
            "holes": "each hole joined to its outer boundary by a zero-width horizontal cut (keyhole); islands are separate polygons",
            "area": "exact union area in cells equals the sum of polygon areas and the source cell count",
            "parity": "zero symmetric difference between the imported raster and the source mask, at cell centres",
            "width_gap": "opening and closing with the disk dx^2 + dy^2 <= 4 (5 cells across) change no design-box cell",
            "connectivity": "4-connectivity; input slab (left edge) and output slab (top edge) in one silicon component; no floating silicon",
            "bias": "layers 2 and 3: the nominal box mask eroded and dilated with the L1 ball of radius 1; slabs not biased",
            "gates": "rules_ok is required on layer 1 (nominal); layers 2 and 3 are reported",
        },
        "checks": report,
        "command": shlex.join(["tools/check_layout.py"] + sys.argv[1:]),
        "tools": {"python": platform.python_version(), "numpy": np.__version__},
    }
    text = json.dumps(cert, indent=2, sort_keys=True) + "\n"
    if args.cert:
        open(args.cert, "w").write(text)
    print(json.dumps({"passed": cert["passed"], "layers": {k: {kk: v[kk] for kk in ("union_area_cells", "source_cells", "symmetric_difference_cells", "width_violations", "gap_violations", "ports_connected", "floating_silicon", "enclosed_air", "agrees_with_rust")} for k, v in layers.items()}, "negative_control": report["negative_control"]}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
