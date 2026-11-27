#!/usr/bin/env python3
"""Write release/release.json: the file names of the release artifacts and the
result sentences of the project page, all computed from the result files.

  tools/release_summary.py --phase1 DIR --diagnostic DIR --release DIR [--tradeoff JSON] [--tests LOG]

Rules (docs/stage6-protocol.md):
  - the release design is the phase 1 design that meets the gates (r_f 6
    first, then the r_f 8 fallback); if none does, the r_f 6 design, marked
    as failing;
  - benchmark: a method "leads" only if its range at the cap does not overlap
    the range of every other method; if no method reaches T1 in its median
    run, the comparison is inconclusive for this budget.
"""

import argparse
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", required=True)
    ap.add_argument("--diagnostic", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--tradeoff", default=None)
    ap.add_argument("--tests", default=None)
    a = ap.parse_args()
    val = json.load(open(os.path.join(a.phase1, "validate", "validate.json")))
    des = {d["label"]: d for d in val["designs"]}
    opt = [l for l in des if not l.startswith("baseline")]
    passing = [l for l in opt if des[l]["gates"]]
    label = passing[0] if passing else opt[0]
    win = next(w for w in val["win_tests"] if w["design"] == label)
    base = win["baseline"]
    d, b = des[label], des[base]
    r, rb = d["robust"], b["robust"]
    gate_txt = "meets" if d["gates"] else "fails"
    win_txt = (f"and beats the best tested conventional bend ({base.replace('baseline ', '')}, robust IL {rb['heldout61'][0]:.4f} dB) by "
               f"{win['gains_db'][1]:.4f} dB" if win["win"] else
               f"but does not beat the best tested conventional bend ({base.replace('baseline ', '')}, robust IL {rb['heldout61'][0]:.4f} dB) "
               f"by the required 0.10 dB: the gain is {win['gains_db'][0]:.4f} dB (training), {win['gains_db'][1]:.4f} dB (61 held-out wavelengths) "
               f"and {win['gains_db'][2]:.4f} dB (half-cell grid)")
    headline = (f"The robust 80 x 80 inverse-designed bend {gate_txt} the release gates on the training spectrum, the 61-wavelength held-out "
                f"spectrum and the half-cell grid (worst case over eroded, nominal and dilated masks: IL {r['heldout61'][0]:.4f} dB, "
                f"RL {r['heldout61'][1]:.2f} dB held-out; gates IL < 0.50 dB, RL >= 22 dB), {win_txt}.")
    diag = json.load(open(os.path.join(a.diagnostic, "validate.json")))["designs"][0]
    geo = {}
    import csv
    for row in csv.DictReader(open(os.path.join(a.diagnostic, "geometry.csv"))):
        if row["scale"] == "1" and row["variant"] == "nominal":
            geo = row
    ev = list(csv.DictReader(open(os.path.join(a.phase1, "rf6", "events.csv"))))
    scored = [e for e in ev if e["kind"] == "score"]
    bad = sum(1 for e in scored if e["geometry_ok"] == "0")
    best_j = min(float(e["j"]) for e in scored if e["geometry_ok"] == "1")
    chosen_it = next(e["note"] for e in scored if e["geometry_ok"] == "1" and float(e["j"]) == best_j)
    failures = (f"Phase 1: {bad} of {len(scored)} scored masks of the r_f 6 run failed the geometry gates; the chosen design is the scored mask "
                f"of {chosen_it}, of 96 iterations. "
                f"The final relaxed iterate (post-hoc diagnostic, not a candidate) reaches a robust IL of {diag['robust']['heldout61'][0]:.4f} dB and RL "
                f"{diag['robust']['heldout61'][1]:.2f} dB held-out, but its nominal mask has {geo.get('floating_silicon', '?')} floating silicon component, "
                f"{geo.get('width_violations', '?')} width and {geo.get('gap_violations', '?')} gap violations (minimum width {geo.get('min_width', '?')} cells, "
                f"minimum gap {geo.get('min_gap', '?')} cells): the projection-based length-scale control did not hold. "
                "Earlier, a nominal-only 80 x 80 run (Stage 5 stretch, 0.0376 dB) also failed the 5-cell width rule; it is not a release target.")
    out = {
        "design_label": label,
        "baseline_label": base,
        "gds": "bend80.gds",
        "svg": "bend80.svg",
        "certificate": "bend80_certificate.json",
        "spectra_plot": "spectra_heldout61.svg",
        "spectra_plot_png": "spectra_heldout61.png",
        "headline": headline,
        "gates": d["gates"],
        "win": win["win"],
        "failures": failures,
        "unverified": ("RTL: the SystemVerilog stencil has not been simulated (no HDL simulator installed); only its software reference is tested. "
                       "No Lean proof (Lean is not installed). The layout scale is illustrative. The independent layout reader is a second "
                       "implementation by the same author, not a commercial DRC tool."),
    }
    if a.tests and os.path.exists(a.tests):
        m = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed", open(a.tests).read())
        if m:
            out["tests"] = f"{sum(int(x[0]) for x in m)} passed, {sum(int(x[1]) for x in m)} failed"
    if a.tradeoff and os.path.exists(a.tradeoff):
        t = json.load(open(a.tradeoff))
        out["tradeoff_plot"] = "tradeoff.svg"
        out["tradeoff_plot_png"] = "tradeoff.png"
        meth = t["methods"]
        names = {"adjoint": "discrete adjoint", "cma": "CMA-ES", "ppo": "one-step PPO"}
        parts = []
        for m, s in meth.items():
            rr = [x for x in t["runs"] if x["method"] == m]
            parts.append(f"{names[m]}: median best J {('none' if s['median_at_cap'] is None else format(s['median_at_cap'], '.4f'))} dB "
                         f"(range {s['min_at_cap'] if s['min_at_cap'] is None else format(s['min_at_cap'], '.4f')} to "
                         f"{s['max_at_cap'] if s['max_at_cap'] is None else format(s['max_at_cap'], '.4f')}), "
                         f"T1 in {sum(1 for x in rr if x['t1'])}/{len(rr)} runs, T2 in {sum(1 for x in rr if x['t2'])}/{len(rr)} runs")
        incomplete = any(not x["complete"] for x in t["runs"]) or any(len([x for x in t["runs"] if x["method"] == m]) < 5 for m in meth)
        def rng(m):
            s = meth[m]
            return (s["min_at_cap"] if s["min_at_cap"] is not None else 1e9, s["max_at_cap"] if s["max_at_cap"] is not None else 1e9)
        lead = [m for m in meth if all(rng(m)[1] < rng(o)[0] for o in meth if o != m)]
        t1_median = [m for m in meth if sum(1 for x in t["runs"] if x["method"] == m and x["t1"]) >= 3]
        if incomplete:
            verdict = "The benchmark is incomplete (not every planned run finished), so no ordering is claimed."
        elif not t1_median:
            verdict = "No method reaches T1 in its median run, so the comparison is inconclusive for this budget."
        elif lead:
            verdict = f"{names[lead[0]]} leads: its range of best J at the cap does not overlap the ranges of the other methods."
        else:
            verdict = "The ranges of best J at the cap overlap, so no method is declared the winner."
        out["benchmark_verdict"] = "Benchmark (5 seeds per method, cap 1200 charged solves): " + "; ".join(parts) + ". " + verdict
        neg = [f"{names[m]} never reaches T2 (J <= best baseline - 0.10 dB) in any run" for m in meth
               if not any(x["t2"] for x in t["runs"] if x["method"] == m)]
        out["negative_results"] = ("; ".join(neg) + ". " if neg else "") + ("Phase 1 does not beat the best conventional bend by 0.10 dB." if not win["win"] else "")
    else:
        out["benchmark_verdict"] = "The benchmark has not finished."
        out["negative_results"] = "Phase 1 does not beat the best conventional bend by 0.10 dB." if not win["win"] else ""
    json.dump(out, open(os.path.join(a.release, "release.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
