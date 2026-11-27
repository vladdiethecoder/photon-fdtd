#!/usr/bin/env bash
# Assemble the Stage 6 release candidate from the committed results
# (docs/stage6-protocol.md, docs/stage6-ledger.md). Run after
# tools/stage6_phase1.sh and tools/stage6_phase2.sh, from the repository root.
#
#   tools/make_release.sh [TAG]
#
# Writes release/ (layout, certificate, lead plots, release.json, SHA256SUMS)
# and dist/site/ (project page and portfolio index).
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-v0.6.0-rc1}
B=./target/release
R=release
P1=docs/results/stage6/phase1
P2=docs/results/stage6/phase2
V=docs/results/stage6/release_validate
mkdir -p $R
cargo build --release --bins

# Release design: lowest training-set robust J among the designs that meet every gate.
tools/release_summary.py --phase1 $P1 --validate $V --diagnostic $P1/diagnostic_final --release $R > /dev/null
MASK=$V/$(python3 -c "import json; print(json.load(open('$R/release.json'))['design_mask'])")
P1MASK=$V/phase1_r_f_6_s1.mask

# Layout export and the independent check, for the release design and the phase 1 design.
$B/export_layout --mask "$MASK" --out $R/bend80
tools/check_layout.py $R/bend80 --design-revision "$(git rev-parse HEAD) $MASK" --extra "$MASK" --cert $R/bend80_certificate.json
$B/export_layout --mask "$P1MASK" --out $R/bend80_phase1
tools/check_layout.py $R/bend80_phase1 --design-revision "$(git rev-parse HEAD) $P1MASK" --extra "$P1MASK" --cert $R/bend80_phase1_certificate.json

# Lead plots (SVG, with PNG copies for e-mail and README).
tools/plot_stage6.py spectra $V $R/spectra_heldout61.svg $R/spectra_heldout61.json \
  --designs "$(python3 -c "import json; r=json.load(open('$R/release.json')); print(r['design_label'] + ',' + r['phase1_label'])"),baseline bezier:0.45,baseline arc"
t2=$(python3 -c "print(round(float(open('docs/results/stage6/baselines/best_baseline.txt').read().split()[1]) - 0.10, 6))")
tools/plot_stage6.py tradeoff $P2/runs $R/tradeoff.svg $R/tradeoff.json --t2 "$t2"
magick -density 110 $R/spectra_heldout61.svg $R/spectra_heldout61.png
magick -density 96 $R/tradeoff.svg $R/tradeoff.png

# Field movie of the release design (solver output).
$B/field_movie --mask "$MASK" --frames 150 --every 40 --out out/stage6/movie
ffmpeg -hide_banner -loglevel error -y -framerate 25 -i out/stage6/movie/frame_%04d.ppm -c:v libvpx-vp9 -b:v 0 -crf 34 -pix_fmt yuv420p -row-mt 1 out/stage6/movie/movie.webm
ffmpeg -hide_banner -loglevel error -y -framerate 25 -i out/stage6/movie/frame_%04d.ppm -c:v libopenh264 -b:v 1500k -pix_fmt yuv420p out/stage6/movie/movie.mp4
magick out/stage6/movie/frame_0149.ppm out/stage6/movie/poster.png

# Result sentences from the result files, then the site.
tools/release_summary.py --phase1 $P1 --validate $V --diagnostic $P1/diagnostic_final --release $R --tradeoff $R/tradeoff.json --tests $R/receipts/test.log > /dev/null
rm -rf dist/site
tools/build_site.py --release $R --validate $V --movie out/stage6/movie --out dist/site --tag "$TAG"
(cd $R && sha256sum bend80*.gds bend80*.svg bend80*_certificate.json bend80*_meta.json bend80*_layer*.mask spectra_heldout61.* tradeoff.* release.json > SHA256SUMS)
echo "release candidate $TAG assembled in $R and dist/site"
