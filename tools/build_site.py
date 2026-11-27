#!/usr/bin/env python3
"""Build the static project page and the portfolio index (no network, no
dependencies beyond the Python standard library).

  tools/build_site.py --release DIR --validate DIR --movie DIR --out DIR [--tag v0.6.0-rc1]

Reads only result files; every result sentence on the page comes from them:
  DIR(validate)/validate.json, and from DIR(release): release.json, the spectra
  and trade-off JSON of tools/plot_stage6.py, the layouts, the certificates
  and the plots; the movie files (movie.mp4, movie.webm, poster.png, movie.json).
Writes OUT/index.html (portfolio index) and OUT/photon-fdtd/index.html.
"""

import argparse
import hashlib
import html
import json
import os
import shutil

REPO = "https://github.com/vladdiethecoder/photon-fdtd"

CSS = """
:root{--fg:#1a1a1a;--muted:#555;--line:#ddd;--bg:#fff;--accent:#b3261e;--ok:#1e7d32;--bad:#b3261e;--card:#f6f6f4}
*{box-sizing:border-box}
body{margin:0;font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg)}
main{max-width:980px;margin:0 auto;padding:20px 18px 60px}
h1{font-size:1.9rem;line-height:1.2;margin:.4em 0 .2em}
h2{font-size:1.35rem;margin:2em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.2em}
h3{font-size:1.1rem;margin:1.4em 0 .4em}
p{margin:.5em 0}
a{color:#0b57d0}
code,pre{font:13.5px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--card);padding:12px;overflow-x:auto;border-radius:6px}
.hero{background:var(--card);border-radius:10px;padding:16px 18px;margin:14px 0}
.scope{font-size:.95rem;color:var(--muted)}
.stat{display:inline-block;margin-right:14px}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.85rem;font-weight:600;color:#fff}
.pass{background:var(--ok)}.fail{background:var(--bad)}.neutral{background:#666}
table{border-collapse:collapse;width:100%;font-size:.93rem;margin:.6em 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}
th{background:var(--card)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.grid2{grid-template-columns:1fr}h1{font-size:1.5rem}}
figure{margin:12px 0}
figure img,figure video,figure svg{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:.9rem;color:var(--muted)}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:6px 0}
.controls button{font:inherit;padding:4px 12px;border:1px solid #999;border-radius:6px;background:#fff;cursor:pointer}
.controls input[type=range]{flex:1;min-width:160px}
.limits{border:2px solid var(--accent);border-radius:10px;padding:6px 18px 12px;margin-top:2em}
.limits h2{border:none;margin-top:.6em;color:var(--accent)}
#spec-tip{font-size:.9rem;min-height:1.4em;color:var(--muted)}
.toggles label{margin-right:14px;white-space:nowrap}
footer{margin-top:3em;font-size:.85rem;color:var(--muted)}
"""

# innerHTML below only receives numbers and design labels from the release's own
# spectra JSON (written by tools/plot_stage6.py); no user input reaches it.
SPECTRA_JS = r"""
(function(){
 const D = JSON.parse(document.getElementById('spec-data').textContent).designs;
 const names = Object.keys(D), cols = ['#d62728','#1f77b4','#7f7f7f','#9467bd','#ff7f0e'];
 const svg = document.getElementById('spec-svg'), tip = document.getElementById('spec-tip');
 const W = 860, H = 520, L = 70, R = 20, T = 20, PH = 200, GAP = 70;
 const lam = D[names[0]].lambda0, x0 = Math.min(...lam), x1 = Math.max(...lam);
 const on = {}; names.forEach(n => on[n] = true);
 let mode = 'worst';
 const box = document.getElementById('spec-toggles');
 names.forEach((n,i) => { const l = document.createElement('label'); l.innerHTML = `<input type=checkbox checked> <span style="color:${cols[i%5]};font-weight:600">${n}</span>`; l.firstChild.onchange = e => { on[n] = e.target.checked; draw(); }; box.appendChild(l); });
 document.querySelectorAll('input[name=spec-mode]').forEach(r => r.onchange = e => { mode = e.target.value; draw(); });
 function series(n, q){ const d = D[n]; if (mode === 'worst') return q === 'il' ? d.il_worst : d.rl_worst; return d.variants[mode][q]; }
 function draw(){
  const vis = names.filter(n => on[n]);
  let ilmax = 0.55, rlmin = 21, rlmax = 24;
  vis.forEach(n => { ilmax = Math.max(ilmax, ...series(n,'il')); const r = series(n,'rl'); rlmin = Math.min(rlmin, ...r); rlmax = Math.max(rlmax, ...r); });
  rlmax = Math.min(rlmax + 1, 60); rlmin = rlmin - 1; ilmax *= 1.06;
  const px = v => L + (v - x0) / (x1 - x0) * (W - L - R);
  const py = (v, lo, hi, top) => top + PH - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo) * PH;
  let s = '';
  [['il', 0, ilmax, T, 'insertion loss (dB)', 0.5], ['rl', rlmin, rlmax, T + PH + GAP, 'return loss (dB)', 22]].forEach(([q, lo, hi, top, lab, gate]) => {
   s += `<rect x="${L}" y="${top}" width="${W-L-R}" height="${PH}" fill="none" stroke="#222"/>`;
   for (let k = 0; k <= 5; k++){ const v = lo + (hi - lo) * k / 5, y = py(v, lo, hi, top); s += `<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="#eee"/><text x="${L-6}" y="${y+4}" font-size="12" text-anchor="end">${v.toFixed(q==='il'?2:1)}</text>`; }
   const gy = py(gate, lo, hi, top); s += `<line x1="${L}" x2="${W-R}" y1="${gy}" y2="${gy}" stroke="#000" stroke-dasharray="6,4"/><text x="${L+4}" y="${gy-4}" font-size="11">gate ${gate}${q==='il'?' dB (max)':' dB (min)'}</text>`;
   s += `<text x="16" y="${top+PH/2}" font-size="13" transform="rotate(-90 16 ${top+PH/2})" text-anchor="middle">${lab}</text>`;
   for (let v = Math.ceil(x0/10)*10; v <= x1; v += 10) s += `<text x="${px(v)}" y="${top+PH+16}" font-size="12" text-anchor="middle">${v}</text>`;
   vis.forEach(n => { const i = names.indexOf(n), ys = series(n, q); s += `<polyline fill="none" stroke="${cols[i%5]}" stroke-width="2.2" points="${lam.map((l,k)=>px(l)+','+py(ys[k],lo,hi,top)).join(' ')}"/>`; });
  });
  s += `<text x="${(L+W-R)/2}" y="${H-4}" font-size="13" text-anchor="middle">lambda0 (native cells)</text>`;
  s += `<line id="spec-cursor" x1="0" x2="0" y1="${T}" y2="${T+2*PH+GAP}" stroke="#888" visibility="hidden"/>`;
  svg.innerHTML = s;
  svg.onmousemove = svg.ontouchmove = ev => {
   const p = ev.touches ? ev.touches[0] : ev, rc = svg.getBoundingClientRect(), x = (p.clientX - rc.left) * W / rc.width;
   let k = 0, best = 1e9; lam.forEach((l,j) => { const d = Math.abs(px(l) - x); if (d < best){ best = d; k = j; } });
   const c = svg.querySelector('#spec-cursor'); c.setAttribute('x1', px(lam[k])); c.setAttribute('x2', px(lam[k])); c.setAttribute('visibility','visible');
   tip.textContent = `lambda0 = ${lam[k].toFixed(1)} cells: ` + vis.map(n => `${n}: IL ${series(n,'il')[k].toFixed(4)} dB, RL ${series(n,'rl')[k].toFixed(2)} dB`).join(' | ');
  };
 }
 draw();
})();
"""

MOVIE_JS = r"""
(function(){
 const v = document.getElementById('movie'), b = document.getElementById('movie-play'), r = document.getElementById('movie-scrub'), t = document.getElementById('movie-time');
 if (!v || !v.canPlayType) return;
 document.getElementById('movie-controls').hidden = false;
 const fmt = () => { t.textContent = `frame ${Math.round(v.currentTime * FPS)} / ${Math.round((v.duration || 0) * FPS)}`; };
 b.onclick = () => { if (v.paused) { v.play(); } else { v.pause(); } };
 v.onplay = () => b.textContent = 'Pause'; v.onpause = () => b.textContent = 'Play';
 v.ontimeupdate = () => { if (v.duration) r.value = 1000 * v.currentTime / v.duration; fmt(); };
 r.oninput = () => { v.pause(); if (v.duration) v.currentTime = r.value / 1000 * v.duration; fmt(); };
 v.onloadedmetadata = fmt;
})();
"""


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def esc(s):
    return html.escape(str(s))


def badge(ok, yes="PASS", no="FAIL"):
    return f'<span class="badge {"pass" if ok else "fail"}">{yes if ok else no}</span>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--validate", required=True)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="v0.6.0-rc1")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--portfolio-href", default="../index.html", help="back link; empty: link to the source repository instead")
    a = ap.parse_args()
    out = os.path.join(a.out, "photon-fdtd")
    os.makedirs(os.path.join(out, "files"), exist_ok=True)
    rel = json.load(open(os.path.join(a.release, "release.json")))
    val = json.load(open(os.path.join(a.validate, "validate.json")))
    spec = json.load(open(os.path.join(a.release, "spectra_heldout61.json")))
    trade = json.load(open(os.path.join(a.release, "tradeoff.json"))) if os.path.exists(os.path.join(a.release, "tradeoff.json")) else None
    cert = json.load(open(os.path.join(a.release, rel["certificate"])))
    files = []
    for key in ("gds", "svg", "certificate", "phase1_gds", "phase1_svg", "phase1_certificate", "spectra_plot", "tradeoff_plot", "spectra_plot_png", "tradeoff_plot_png"):
        f = rel.get(key)
        if f and os.path.exists(os.path.join(a.release, f)):
            shutil.copy(os.path.join(a.release, f), os.path.join(out, "files", f))
            files.append((key, f, os.path.getsize(os.path.join(a.release, f)), sha(os.path.join(a.release, f))))
    for f in ("movie.mp4", "movie.webm", "poster.png"):
        shutil.copy(os.path.join(a.movie, f), os.path.join(out, "files", f))
    movie = json.load(open(os.path.join(a.movie, "movie.json")))
    designs = {d["label"]: d for d in val["designs"]}
    chosen = designs[rel["design_label"]]
    base = designs[rel["baseline_label"]]
    win = next((w for w in val["win_tests"] if w["design"] == rel["design_label"]), None)
    r = chosen["robust"]
    gates = chosen["gates"]
    gch = chosen["grid_change_il_db"]

    def row(lbl, d):
        rr = d["robust"]
        return (f"<tr><td>{esc(lbl)}</td><td>{rr['train15'][0]:.4f}</td><td>{rr['train15'][1]:.2f}</td><td>{rr['heldout61'][0]:.4f}</td>"
                f"<td>{rr['heldout61'][1]:.2f}</td><td>{rr['scale2'][0]:.4f}</td><td>{rr['scale2'][1]:.2f}</td><td>{badge(d['gates'])}</td></tr>")

    result_rows = "".join(row(l, d) for l, d in designs.items())
    variant_rows = ""
    for l, d in designs.items():
        for vi, vn in enumerate(("eroded", "nominal", "dilated")):
            vv = d["variants_il_rl"]
            variant_rows += (f"<tr><td>{esc(l)}</td><td>{vn}</td><td>{vv['train15'][vi][0]:.4f} / {vv['train15'][vi][1]:.2f}</td>"
                             f"<td>{vv['heldout61'][vi][0]:.4f} / {vv['heldout61'][vi][1]:.2f}</td><td>{vv['scale2'][vi][0]:.4f} / {vv['scale2'][vi][1]:.2f}</td>"
                             f"<td>{d['grid_change_il_db'][vi]:+.4f}</td><td>{d['grid_change_rl_db'][vi]:+.2f}</td></tr>")
    if win:
        win_text = (f"Against the best tested conventional bend ({esc(win['baseline'])}), the robust worst-case insertion-loss gain is "
                    f"{win['gains_db'][0]:.4f} dB on the 15 training samples, {win['gains_db'][1]:.4f} dB on the 61 held-out samples and "
                    f"{win['gains_db'][2]:.4f} dB on the half-cell grid; the protocol requires at least 0.10 dB on all three, and more than the "
                    f"largest measured grid change ({win['largest_abs_grid_change_db']:.4f} dB). Result: {badge(win['win'], 'WIN', 'NO WIN')}")
    else:
        win_text = "No win test was run."
    lay = cert["checks"]["layers"]
    neg = cert["checks"]["negative_control"]
    layer_rows = "".join(
        f"<tr><td>{k}</td><td>{esc(cert['layers'][k + '/0'])}</td><td>{v['union_area_cells']}</td><td>{v['source_cells']}</td><td>{v['symmetric_difference_cells']}</td>"
        f"<td>{v['width_violations']}</td><td>{v['gap_violations']}</td><td>{v['ports_connected']}</td><td>{v['floating_silicon']}</td><td>{v['enclosed_air']}</td></tr>"
        for k, v in lay.items())
    file_rows = "".join(f'<tr><td><a href="files/{esc(f)}" download>{esc(f)}</a></td><td>{n:,} B</td><td><code>{h[:16]}…</code></td></tr>' for _, f, n, h in files)
    trade_html = ""
    if trade:
        trows = ""
        for m, s in trade["methods"].items():
            rr = [x for x in trade["runs"] if x["method"] == m]
            t1 = sum(1 for x in rr if x["t1"])
            t2 = sum(1 for x in rr if x["t2"])
            fmt = lambda v: "none" if v is None else f"{v:.4f}"
            trows += (f"<tr><td>{esc(m)}</td><td>{len(rr)}</td><td>{sum(1 for x in rr if x['complete'])}</td><td>{fmt(s['median_at_cap'])}</td>"
                      f"<td>{fmt(s['min_at_cap'])} – {fmt(s['max_at_cap'])}</td><td>{s.get('charged_min')}–{s.get('charged_max')}</td><td>{t1} / {len(rr)}</td><td>{t2} / {len(rr)}</td>"
                      f"<td>{sum(x['geometry_ok'] for x in rr)} / {sum(x['scored'] for x in rr)}</td></tr>")
        trade_html = f"""
<figure><img src="files/{esc(rel['tradeoff_plot'])}" alt="Optimizer trade-off plot: best feasible robust J against charged cost and wall time for adjoint, CMA-ES and one-step PPO"><figcaption>Every run is drawn, including runs without a feasible design (no line until the first feasible design). Thick lines: median of five runs; bands: min–max.</figcaption></figure>
<table><tr><th>method</th><th>runs</th><th>complete</th><th>median best J (dB)</th><th>range (dB)</th><th>charged per run</th><th>T1 runs</th><th>T2 runs</th><th>geometry-feasible / scored</th></tr>{trows}</table>
<p>{esc(rel.get('benchmark_verdict', ''))}</p>"""
    poster_note = f"Ez at lambda0 = {movie['lambda0_cells']:g} cells, {movie['frames']} frames, {movie['steps_per_frame']} steps per frame; red +Ez, blue −Ez, grey silicon."
    back = f'<a href="{esc(a.portfolio_href)}">← portfolio</a>' if a.portfolio_href else f'<a href="{REPO}">source on GitHub</a>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>photon-fdtd: robust inverse-designed waveguide bend (2D TMz)</title><style>{CSS}</style></head><body><main>
<p>{back}</p>
<h1>photon-fdtd: a verified 2D FDTD solver and a robust inverse-designed bend</h1>
<div class="hero">
<p><strong>Result.</strong> {esc(rel['headline'])}</p>
<p><span class="stat">Release gates (robust IL &lt; 0.50 dB and RL ≥ 22 dB over eroded, nominal and dilated masks; binary, connected, 5-cell width and gap): {badge(gates)}</span> <span class="stat">Win over the best tested conventional bend: {badge(bool(win and win['win']), 'WIN', 'NO WIN')}</span> <span class="stat">Layout certificate: {badge(cert['passed'])}</span></p>
<p class="scope"><strong>Scope.</strong> 2D TMz model in natural units (Rust CPU and CUDA f64). Not a 3D silicon-on-insulator result, not fabrication-ready, and not a PDK signoff. The layout scale (1 cell = {cert['units']['cell_nm']} nm) is illustrative, not calibrated. Release candidate {esc(a.tag)}, solver revision <code>{esc(cert['solver_revision'][:12])}</code>.</p>
</div>
<div class="grid2">
<figure><img src="files/{esc(rel['svg'])}" alt="Optimised 80 by 80 bend, nominal silicon mask"><figcaption>Nominal mask with a 16-cell halo of the input and output slabs (white = silicon). Manhattan cell-edge contours.</figcaption></figure>
<figure><video id="movie" poster="files/poster.png" muted playsinline loop preload="metadata" width="{movie['crop'][2]-movie['crop'][0]}" height="{movie['crop'][3]-movie['crop'][1]}">
<source src="files/movie.webm" type="video/webm"><source src="files/movie.mp4" type="video/mp4">
<img src="files/poster.png" alt="Static frame of the Ez field in the optimised bend"></video>
<div class="controls" id="movie-controls" hidden><button id="movie-play" type="button">Play</button><input id="movie-scrub" type="range" min="0" max="1000" value="0" aria-label="scrub"><span id="movie-time"></span></div>
<noscript><p><img src="files/poster.png" alt="Static frame of the Ez field"></p></noscript>
<figcaption>Solver output: {esc(poster_note)}</figcaption></figure>
</div>

<h2>Validated spectra</h2>
<figure><img src="files/{esc(rel['spectra_plot'])}" alt="Held-out 61-wavelength insertion and return loss spectra"></figure>
<p>Interactive view of the same data (held-out 61 wavelengths). Move the pointer over the plot for values.</p>
<div class="toggles" id="spec-toggles"></div>
<div class="toggles"><label><input type=radio name=spec-mode value=worst checked> worst of the three variants</label><label><input type=radio name=spec-mode value=eroded> eroded</label><label><input type=radio name=spec-mode value=nominal> nominal</label><label><input type=radio name=spec-mode value=dilated> dilated</label></div>
<svg id="spec-svg" viewBox="0 0 860 520" width="100%" role="img" aria-label="interactive spectra"></svg>
<div id="spec-tip"></div>
<script id="spec-data" type="application/json">{json.dumps(spec)}</script>

<h3>Robust results (worst over the three bias variants)</h3>
<table><tr><th>design</th><th>IL train (dB)</th><th>RL train (dB)</th><th>IL 61 (dB)</th><th>RL 61 (dB)</th><th>IL half-cell (dB)</th><th>RL half-cell (dB)</th><th>gates</th></tr>{result_rows}</table>
<p>{win_text}</p>
<h3>Every variant, and the measured grid change (half-cell minus native grid)</h3>
<table><tr><th>design</th><th>variant</th><th>train IL / RL</th><th>61 IL / RL</th><th>half-cell IL / RL</th><th>grid change IL (dB)</th><th>grid change RL (dB)</th></tr>{variant_rows}</table>

<h2>Optimizer benchmark</h2>
{trade_html or '<p>The benchmark did not run.</p>'}

<h2>Layout and certificate</h2>
<p>The certificate says: <em>{esc(cert['claim'])}</em> — {badge(cert['passed'])}. It was checked by an independent reader (<code>tools/check_layout.py</code>, numpy only, no code shared with the Rust writer).</p>
<table><tr><th>layer</th><th>content</th><th>union area (cells)</th><th>source cells</th><th>symmetric difference</th><th>width viol.</th><th>gap viol.</th><th>ports connected</th><th>floating Si</th><th>enclosed air</th></tr>{layer_rows}</table>
<p>Negative control: {esc(neg['description'])}: symmetric difference {neg['symmetric_difference_cells']} cells, detected {badge(neg['detected'], 'YES', 'NO')}.</p>
<p>Frame: origin at the {esc(cert['geometry_frame']['origin'])}, x {esc(cert['geometry_frame']['x_axis'])}, y {esc(cert['geometry_frame']['y_axis'])}; database unit 1 nm, user unit 1 µm.</p>
<h3>Downloads</h3>
<table><tr><th>file</th><th>size</th><th>SHA-256</th></tr>{file_rows}</table>

<h2>Reproduce</h2>
<pre>git clone {REPO} && cd photon-fdtd && git checkout {esc(a.tag)}
cargo test --release                      # {esc(rel.get('tests', ''))}
cargo run --release --bin stage6_baselines -- --out out/stage6/baselines
tools/stage6_phase1.sh                    # phase 1 optimiser and held-out validation
tools/stage6_phase2.sh                    # phase 2 benchmark (about 4 to 5 hours on one RTX 5090)
tools/make_release.sh                     # layout export, certificate, plots, this page</pre>
<p>Protocol (fixed before the production runs): <a href="{REPO}/blob/{esc(a.tag)}/docs/stage6-protocol.md">docs/stage6-protocol.md</a>. Progress ledger: <a href="{REPO}/blob/{esc(a.tag)}/docs/stage6-ledger.md">docs/stage6-ledger.md</a>.</p>

<section class="limits" id="limitations"><h2>Limitations &amp; Failure Modes</h2>
<ul>
<li><strong>2D TMz versus 3D.</strong> The model has one field component (Ez), an infinitely tall slab and no substrate. A 3D silicon-on-insulator bend has a finite height, substrate leakage and polarisation coupling; none of them is modelled. The 3.48:1 index contrast without a substrate probably makes tight bends look better than in 3D (not tested).</li>
<li><strong>Stair-stepping.</strong> The geometry is a binary pixel mask on the solver grid, and the exported layout keeps the same cell-edge steps. The half-cell grid re-rasterises the same design and gives the measured grid change above; it does not remove stair-stepping.</li>
<li><strong>Modelled bias versus real process variation.</strong> The eroded and dilated masks are a uniform ±1-native-cell morphological change of the design box only. The slabs are not biased. Real processes have non-uniform edge bias, corner rounding, sidewall angle, line-edge roughness and thickness variation; none of them is in the model.</li>
<li><strong>Negative benchmark results.</strong> {esc(rel.get('negative_results', ''))}</li>
<li><strong>Known failures kept in the record.</strong> {esc(rel.get('failures', ''))}</li>
<li><strong>Not verified.</strong> {esc(rel.get('unverified', ''))}</li>
</ul></section>
<footer>photon-fdtd {esc(a.tag)} · generated by tools/build_site.py from the release files · <a href="{REPO}">source</a></footer>
</main>
<script>const FPS = {a.fps};</script><script>{MOVIE_JS}</script><script>{SPECTRA_JS}</script></body></html>
"""
    open(os.path.join(out, "index.html"), "w").write(page)
    index = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio</title><style>{CSS}</style></head><body><main>
<h1>Projects</h1>
<div class="hero"><h3 style="margin-top:0"><a href="photon-fdtd/index.html">photon-fdtd</a></h3>
<p>{esc(rel['headline'])}</p><p class="scope">2D TMz, Rust and CUDA. Not a 3D or fabrication result.</p>
<p><img src="photon-fdtd/files/poster.png" alt="Ez field in the optimised bend" style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px"></p></div>
<footer>Static page, built locally.</footer></main></body></html>
"""
    open(os.path.join(a.out, "index.html"), "w").write(index)
    print(f"wrote {a.out}/index.html and {out}/index.html ({len(files)} files + movie)")


if __name__ == "__main__":
    main()
