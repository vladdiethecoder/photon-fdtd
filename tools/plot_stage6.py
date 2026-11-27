#!/usr/bin/env python3
"""Stage 6 plots from the raw logs (SVG, numpy only; no plotting library).

  tools/plot_stage6.py spectra VALIDATE_DIR OUT.svg OUT.json [--designs A,B,...]
      Held-out 61-sample spectra: for each design, the worst of the three bias
      variants at each wavelength (solid) and the nominal variant (dashed);
      IL and RL panels with the gates (IL 0.50 dB, RL 22 dB).

  tools/plot_stage6.py tradeoff RUNS_DIR OUT.svg OUT.json [--t2 J]
      Best feasible J so far against charged forward-equivalents and against
      wall time, for every run (thin) and the median of each method (thick,
      with the min-max band); the adjoint median also with the adjoint solves
      re-weighted by the measured adjoint/forward wall-time ratio (dotted).
      A run without a feasible design so far has no line at that point, and a
      run's line ends where the run ended; the median and the band are drawn
      only where all five runs of the method are still running.
      Bars: fraction of scored candidates that meet the geometry gates, and
      that meet the performance gates. Table: time to the targets.
"""

import csv
import json
import math
import sys
from collections import defaultdict

import numpy as np

COL = {"adjoint": "#d62728", "cma": "#1f77b4", "ppo": "#2ca02c"}
DCOL = ["#d62728", "#1f77b4", "#7f7f7f", "#9467bd", "#ff7f0e"]


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, s):
        self.items.append(s)

    def text(self, x, y, s, size=13, anchor="start", rot=None, weight="normal"):
        tr = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}"{tr}>{s}</text>')

    def done(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'font-family="Helvetica, Arial, sans-serif">\n<rect width="100%" height="100%" fill="white"/>\n' + "\n".join(self.items) + "\n</svg>\n")


def ticks(lo, hi, n=6):
    raw = (hi - lo) / n
    p = 10 ** math.floor(math.log10(raw))
    step = next(m * p for m in (1, 2, 2.5, 5, 10) if m * p >= raw)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out


class Panel:
    def __init__(self, svg, x, y, w, h, xr, yr, xlabel, ylabel, title=None, logx=False, xticks=True):
        self.s, self.x, self.y, self.w, self.h, self.xr, self.yr, self.logx = svg, x, y, w, h, xr, yr, logx
        svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#222"/>')
        for t in ticks(*yr):
            py = self.py(t)
            svg.add(f'<line x1="{x}" x2="{x + w}" y1="{py:.1f}" y2="{py:.1f}" stroke="#e5e5e5"/>')
            svg.text(x - 6, py + 4, f"{t:g}", 12, "end")
        xt = [] if not xticks else [10 ** k for k in range(int(math.floor(math.log10(xr[0]))), int(math.ceil(math.log10(xr[1]))) + 1)] if logx else ticks(*xr)
        for t in xt:
            if xr[0] <= t <= xr[1]:
                px = self.px(t)
                svg.add(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{y + h}" y2="{y + h + 5}" stroke="#222"/>')
                svg.text(px, y + h + 19, f"{t:g}", 12, "middle")
        svg.text(x + w / 2, y + h + 38, xlabel, 13, "middle")
        svg.text(x - 48, y + h / 2, ylabel, 13, "middle", rot=-90)
        if title:
            svg.text(x, y - 10, title, 14, weight="bold")

    def px(self, v):
        if self.logx:
            v = math.log10(max(v, 1e-9))
            lo, hi = math.log10(self.xr[0]), math.log10(self.xr[1])
        else:
            lo, hi = self.xr
        return self.x + (v - lo) / (hi - lo) * self.w

    def py(self, v):
        lo, hi = self.yr
        v = min(max(v, lo), hi)
        return self.y + self.h - (v - lo) / (hi - lo) * self.h

    def line(self, xs, ys, color, width=2.0, dash=None, opacity=1.0):
        seg, segs = [], []
        for a, b in zip(xs, ys):
            if b is None or not math.isfinite(b):
                if seg:
                    segs.append(seg)
                seg = []
            else:
                seg.append(f"{self.px(a):.1f},{self.py(b):.1f}")
        if seg:
            segs.append(seg)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        for s in segs:
            self.s.add(f'<polyline points="{" ".join(s)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}"{d}/>')

    def band(self, xs, lo, hi, color):
        pts = [(a, l, h) for a, l, h in zip(xs, lo, hi) if math.isfinite(l) and math.isfinite(h)]
        if len(pts) < 2:
            return
        up = " ".join(f"{self.px(a):.1f},{self.py(h):.1f}" for a, _, h in pts)
        dn = " ".join(f"{self.px(a):.1f},{self.py(l):.1f}" for a, l, _ in reversed(pts))
        self.s.add(f'<polygon points="{up} {dn}" fill="{color}" fill-opacity="0.15" stroke="none"/>')

    def hline(self, v, label, color="#000"):
        py = self.py(v)
        self.s.add(f'<line x1="{self.x}" x2="{self.x + self.w}" y1="{py:.1f}" y2="{py:.1f}" stroke="{color}" stroke-dasharray="6,4"/>')
        self.s.text(self.x + 4, py - 5, label, 11, "start")


def spectra(vdir, out_svg, out_json, designs=None):
    rows = list(csv.DictReader(open(f"{vdir}/spectra.csv")))
    data = defaultdict(lambda: defaultdict(dict))  # design -> variant -> lambda -> (il, rl)
    for r in rows:
        if r["set"] != "heldout61":
            continue
        data[r["design"]][r["variant"]][float(r["lambda0"])] = (float(r["il_db"]), float(r["rl_db"]))
    names = designs or list(data.keys())
    lam = sorted(next(iter(data[names[0]].values())).keys())
    series = {}
    for n in names:
        il_w = [max(data[n][v][l][0] for v in data[n]) for l in lam]
        rl_w = [min(data[n][v][l][1] for v in data[n]) for l in lam]
        series[n] = {"lambda0": lam, "il_worst": il_w, "rl_worst": rl_w,
                     "il_nominal": [data[n]["nominal"][l][0] for l in lam], "rl_nominal": [data[n]["nominal"][l][1] for l in lam],
                     "variants": {v: {"il": [data[n][v][l][0] for l in lam], "rl": [data[n][v][l][1] for l in lam]} for v in data[n]}}
    il_hi = max(0.55, max(max(s["il_worst"]) for s in series.values()) * 1.08)
    rl_vals = [x for s in series.values() for x in s["rl_worst"] + s["rl_nominal"]]
    rl_lo, rl_hi = min(18.0, min(rl_vals) - 1), min(max(rl_vals) + 2, 60)
    svg = Svg(900, 760)
    svg.text(450, 28, "Held-out spectra, 61 wavelengths, three bias variants (2D TMz)", 16, "middle", weight="bold")
    top = Panel(svg, 90, 60, 560, 280, (min(lam), max(lam)), (0, il_hi), "lambda0 (native cells)", "insertion loss (dB)")
    bot = Panel(svg, 90, 420, 560, 280, (min(lam), max(lam)), (rl_lo, rl_hi), "lambda0 (native cells)", "return loss (dB)")
    top.hline(0.5, "gate 0.50 dB")
    bot.hline(22.0, "gate 22 dB")
    for i, n in enumerate(names):
        c = DCOL[i % len(DCOL)]
        top.line(lam, series[n]["il_worst"], c, 2.4)
        top.line(lam, series[n]["il_nominal"], c, 1.4, "5,4")
        bot.line(lam, series[n]["rl_worst"], c, 2.4)
        bot.line(lam, series[n]["rl_nominal"], c, 1.4, "5,4")
        ly = 80 + 24 * i
        svg.add(f'<line x1="670" x2="700" y1="{ly}" y2="{ly}" stroke="{c}" stroke-width="3"/>')
        svg.text(706, ly + 4, n, 12)
    ly = 80 + 24 * len(names) + 10
    svg.add(f'<line x1="670" x2="700" y1="{ly}" y2="{ly}" stroke="#444" stroke-width="2.4"/>')
    svg.text(706, ly + 4, "worst of eroded, nominal, dilated", 11)
    svg.add(f'<line x1="670" x2="700" y1="{ly + 20}" y2="{ly + 20}" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4"/>')
    svg.text(706, ly + 24, "nominal only", 11)
    open(out_svg, "w").write(svg.done())
    json.dump({"set": "heldout61", "designs": series}, open(out_json, "w"))
    print(f"wrote {out_svg}, {out_json}: {len(names)} designs, {len(lam)} wavelengths")


def load_runs(rdir):
    import glob
    import os
    runs = []
    for d in sorted(glob.glob(f"{rdir}/*_*/")):
        name = os.path.basename(d.rstrip("/"))
        method, seed = name.rsplit("_", 1)
        ev = list(csv.DictReader(open(d + "events.csv")))
        summ = json.load(open(d + "summary.json")) if os.path.exists(d + "summary.json") else None
        runs.append({"method": method, "seed": seed, "events": ev, "summary": summ, "complete": summ is not None})
    return runs


def curve(ev, xkey, rho=None):
    xs, ys = [0.0], [math.nan]
    for e in ev:
        if rho is not None:
            x = float(e["forward_solves"]) + rho * float(e["adjoint_solves"])
        else:
            x = float(e[xkey])
        b = float(e["best_j"])
        xs.append(x)
        ys.append(b if math.isfinite(b) else math.nan)
    return np.array(xs), np.array(ys)


ENDED = -1.0  # marker: the run had ended at this grid point


def at(xs, ys, grid):
    """Best J so far at each grid point: NaN before the first feasible design,
    ENDED after the run ended (a run that stopped early does not count as
    having used the rest of the budget)."""
    idx = np.searchsorted(xs, grid, side="right") - 1
    v = np.where(idx >= 0, ys[np.clip(idx, 0, None)], math.nan)
    return np.where(grid > xs[-1], ENDED, v)


def tradeoff(rdir, out_svg, out_json, t2=None):
    runs = load_runs(rdir)
    methods = [m for m in ("adjoint", "cma", "ppo") if any(r["method"] == m for r in runs)]
    cap = max(float(r["events"][-1]["charged"]) for r in runs if r["events"])
    wall = max(float(r["events"][-1]["wall_s"]) for r in runs if r["events"])
    allj = [float(e["best_j"]) for r in runs for e in r["events"] if math.isfinite(float(e["best_j"]))]
    ylo, yhi = (min(allj + [0.5]) * 0.95, max(allj) * 1.03) if allj else (0, 1)
    yhi = min(max(yhi, 0.52), 1.2)  # values above 1.2 dB are drawn at the top edge
    svg = Svg(1180, 900 + max(0, len(runs) - 15) * 17)
    svg.text(590, 28, "Best feasible robust J so far (lower is better), shared 20 x 20 representation, 2D TMz", 16, "middle", weight="bold")
    pa = Panel(svg, 90, 70, 460, 320, (0, cap), (ylo, yhi), "charged forward-equivalent solves", "best feasible J (dB, clipped at 1.2)", "A. against charged cost")
    pb = Panel(svg, 670, 70, 460, 320, (0, wall / 60), (ylo, yhi), "wall time (min), one RTX 5090", "best feasible J (dB, clipped at 1.2)", "B. against wall time")
    for p in (pa, pb):
        p.hline(0.5, "T1 gates: J < 0.50")
        if t2 is not None:
            p.hline(t2, f"T2: J <= {t2:.4f}", "#555")
    out = {"methods": {}, "runs": []}
    grid = np.linspace(0, cap, 241)
    tgrid = np.linspace(0, wall, 241)
    for m in methods:
        mr = [r for r in runs if r["method"] == m]
        c = COL[m]
        curves_c, curves_t = [], []
        for r in mr:
            xs, ys = curve(r["events"], "charged")
            ts, _ = curve(r["events"], "wall_s")
            pa.line(xs, ys, c, 1.0, opacity=0.45)
            pb.line(ts / 60, ys, c, 1.0, opacity=0.45)
            curves_c.append(at(xs, ys, grid))
            curves_t.append(at(ts, ys, tgrid))
            ev = r["events"]
            scored = [e for e in ev if e["kind"] == "score"]
            def first(cond):
                for e in ev:
                    if cond(e):
                        return {"charged": int(e["charged"]), "wall_s": float(e["wall_s"])}
                return None
            fs = float(ev[-1]["forward_s"]) / max(int(ev[-1]["forward_solves"]), 1)
            asv = float(ev[-1]["adjoint_s"]) / max(int(ev[-1]["adjoint_solves"]), 1) if int(ev[-1]["adjoint_solves"]) else None
            out["runs"].append({
                "method": m, "seed": r["seed"], "complete": r["complete"],
                "charged": int(ev[-1]["charged"]), "forward_solves": int(ev[-1]["forward_solves"]), "adjoint_solves": int(ev[-1]["adjoint_solves"]),
                "cache_hits": int(ev[-1]["cache_hits"]), "wall_s": float(ev[-1]["wall_s"]),
                "best_j": None if not math.isfinite(float(ev[-1]["best_j"])) else float(ev[-1]["best_j"]),
                "scored": len(scored), "geometry_ok": sum(int(e["geometry_ok"]) for e in scored), "gates_ok": sum(int(e["gates_ok"]) for e in scored),
                "t1": first(lambda e: e["gates_ok"] == "1"),
                "t2": first(lambda e: t2 is not None and math.isfinite(float(e["best_j"])) and float(e["best_j"]) <= t2),
                "forward_s_per_solve": fs, "adjoint_s_per_solve": asv,
                "curve_charged": [[float(a), None if not math.isfinite(b) else float(b)] for a, b in zip(*curve(ev, "charged"))],
                "curve_wall_s": [[float(a), None if not math.isfinite(b) else float(b)] for a, b in zip(curve(ev, "wall_s")[0], curve(ev, "charged")[1])],
            })
        A = np.array(curves_c)
        B = np.array(curves_t)
        def stats(M):
            # Drawn only where all runs are still running; "no feasible design
            # yet" counts as the worst value.
            running = (M != ENDED).all(axis=0)
            Mi = np.where(np.isnan(M), np.inf, M)
            Mi = np.where(running[None, :], Mi, np.inf)
            med = np.median(Mi, axis=0)
            return np.where(np.isinf(med), np.nan, med), np.where(np.isinf(Mi.min(0)), np.nan, Mi.min(0)), np.where(np.isinf(Mi.max(0)), np.nan, Mi.max(0))
        med, lo, hi = stats(A)
        pa.band(grid, lo, hi, c)
        pa.line(grid, med, c, 3.0)
        medt, lot, hit = stats(B)
        pb.band(tgrid / 60, lot, hit, c)
        pb.line(tgrid / 60, medt, c, 3.0)
        # Final best J of each run (at the end of the run, whatever budget it used).
        fin = np.array([x["best_j"] if x["best_j"] is not None else np.inf for x in out["runs"] if x["method"] == m])
        used = [x["charged"] for x in out["runs"] if x["method"] == m]
        f = lambda v: None if not math.isfinite(v) else float(v)
        out["methods"][m] = {"runs": len(mr), "median_at_cap": f(np.median(fin)), "min_at_cap": f(fin.min()), "max_at_cap": f(fin.max()),
                             "charged_min": min(used), "charged_max": max(used)}
        if m == "adjoint":
            rhos = [x["adjoint_s_per_solve"] / x["forward_s_per_solve"] for x in out["runs"] if x["method"] == "adjoint" and x["adjoint_s_per_solve"]]
            rho = float(np.median(rhos)) if rhos else 1.0
            S = np.array([at(*curve(r["events"], None, rho), grid) for r in mr])
            meds, _, _ = stats(S)
            pa.line(grid, meds, c, 2.0, "2,3")
            out["methods"][m]["measured_adjoint_to_forward_ratio"] = rho
    ly = 460
    for i, m in enumerate(methods):
        svg.add(f'<line x1="90" x2="120" y1="{ly + 22 * i}" y2="{ly + 22 * i}" stroke="{COL[m]}" stroke-width="3"/>')
        svg.text(126, ly + 22 * i + 4, {"adjoint": "discrete adjoint", "cma": "CMA-ES", "ppo": "one-step PPO (bandit)"}[m] + " - median, min-max band, thin = single runs", 12)
    svg.add(f'<line x1="90" x2="120" y1="{ly + 66}" y2="{ly + 66}" stroke="{COL["adjoint"]}" stroke-width="2" stroke-dasharray="2,3"/>')
    svg.text(126, ly + 70, "adjoint median with adjoint solves re-weighted by the measured wall-time ratio", 12)
    # C: feasibility rates; D: time to target.
    pc = Panel(svg, 90, 580, 460, 240, (0, 3 * len(methods)), (0, 1), "", "fraction of scored candidates", "C. feasibility rates (all runs)", xticks=False)
    for i, m in enumerate(methods):
        rr = [x for x in out["runs"] if x["method"] == m]
        sc = sum(x["scored"] for x in rr)
        for k, (key, op) in enumerate((("geometry_ok", 0.9), ("gates_ok", 0.45))):
            frac = sum(x[key] for x in rr) / sc if sc else 0
            x0 = pc.px(3 * i + 0.3 + 1.2 * k)
            x1 = pc.px(3 * i + 1.3 + 1.2 * k)
            y = pc.py(frac)
            svg.add(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{x1 - x0:.1f}" height="{pc.py(0) - y:.1f}" fill="{COL[m]}" fill-opacity="{op}"/>')
            svg.text((x0 + x1) / 2, y - 4, f"{frac:.3f}" if frac < 0.01 else f"{frac:.2f}", 11, "middle")
        svg.text(pc.px(3 * i + 1.5), 840, {"adjoint": "adjoint", "cma": "CMA-ES", "ppo": "PPO"}[m], 12, "middle")
    svg.text(90, 862, "dark: geometry gates met; light: geometry and performance gates met", 11)
    svg.text(670, 570, "D. time to target (charged solves / wall min); - = not reached", 14, weight="bold")
    y = 596
    svg.text(670, y, "run", 12, weight="bold")
    svg.text(790, y, "best J", 12, weight="bold")
    svg.text(870, y, "T1", 12, weight="bold")
    svg.text(990, y, "T2", 12, weight="bold")
    for x in out["runs"]:
        y += 17
        fmt = lambda t: "-" if t is None else f'{t["charged"]} / {t["wall_s"] / 60:.1f}'
        svg.text(670, y, f'{x["method"]} {x["seed"]}' + ("" if x["complete"] else " (incomplete)"), 11)
        svg.text(790, y, "none" if x["best_j"] is None else f'{x["best_j"]:.4f}', 11)
        svg.text(870, y, fmt(x["t1"]), 11)
        svg.text(990, y, fmt(x["t2"]), 11)
    open(out_svg, "w").write(svg.done())
    json.dump(out, open(out_json, "w"))
    print(f"wrote {out_svg}, {out_json}: {len(runs)} runs")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = {a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--")}
    if a[0] == "spectra":
        spectra(a[1], a[2], a[3], opt.get("--designs", "").split(",") if "--designs" in opt else None)
    elif a[0] == "tradeoff":
        tradeoff(a[1], a[2], a[3], float(opt["--t2"]) if "--t2" in opt else None)
    else:
        sys.exit(__doc__)
