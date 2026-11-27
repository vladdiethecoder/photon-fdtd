#!/usr/bin/env bash
# Build and test receipts from a clean checkout of one commit.
#
#   tools/clean_checkout_check.sh [COMMIT] [OUT_DIR]
#
# Clones this repository into a scratch directory, checks out COMMIT (default
# HEAD), and records: the toolchain, cargo build --release, cargo test
# --release, the GUI viewer build, the Stage 5 fixture regression, and a
# byte-for-byte rebuild of the release layout files from the committed mask.
# Uses CUDA device 0. Writes OUT_DIR/*.log and OUT_DIR/receipts.json.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMMIT=$(git rev-parse "${1:-HEAD}")
OUT=$(realpath -m "${2:-release/receipts}")
SCRATCH=${TMPDIR:-/tmp}/photon-fdtd-clean-$COMMIT
rm -rf "$SCRATCH"
mkdir -p "$OUT"
git clone -q --no-hardlinks "$REPO" "$SCRATCH"
cd "$SCRATCH"
git checkout -q "$COMMIT"
test -z "$(git status --porcelain)"
{ rustc --version; cargo --version; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; uname -sr; } > "$OUT/toolchain.log"
run() { # name command...
  local name=$1; shift
  local t0=$(date +%s)
  if "$@" > "$OUT/$name.log" 2>&1; then echo "$name ok $(( $(date +%s) - t0 ))"; else echo "$name FAILED"; return 1; fi
}
run build cargo build --release
run test cargo test --release
run viewer_build cargo build --release --features gui --bin viewer
run stage5_fixture ./target/release/stage5_validate docs/results/stage5/rho_final.txt --beta 128 --out "$SCRATCH/out/s5"
diff "$OUT/stage5_fixture.log" docs/results/stage5/validate.log > "$OUT/stage5_fixture.diff" || true
MASK=docs/results/stage6/release_validate/$(python3 -c "import json; print(json.load(open('release/release.json'))['design_mask'])")
./target/release/export_layout --mask "$MASK" --out "$SCRATCH/out/bend80" > "$OUT/layout_rebuild.log"
./target/release/export_layout --mask docs/results/stage6/release_validate/phase1_r_f_6_s1.mask --out "$SCRATCH/out/bend80_phase1" >> "$OUT/layout_rebuild.log"
layout_same=true
for p in bend80 bend80_phase1; do
  for f in $p.gds $p.svg ${p}_layer1.mask ${p}_layer2.mask ${p}_layer3.mask; do
    cmp -s "$SCRATCH/out/$f" "release/$f" || layout_same=false
  done
done
python3 tools/check_layout.py release/bend80 --design-revision clean-checkout > "$OUT/layout_check.log" && python3 tools/check_layout.py release/bend80_phase1 --design-revision clean-checkout >> "$OUT/layout_check.log" || layout_same=false
rtl=skipped
if command -v verilator > /dev/null; then
  verilator --version >> "$OUT/toolchain.log"
  run rtl_build verilator --binary -j 0 -Wall --top-module tb_yee1d --Mdir "$SCRATCH/obj_dir" rtl/yee1d.sv rtl/tb_yee1d.sv
  run rtl_nominal "$SCRATCH/obj_dir/Vtb_yee1d" +VEC=rtl/vectors/nominal
  run rtl_saturate "$SCRATCH/obj_dir/Vtb_yee1d" +VEC=rtl/vectors/saturate
  rtl=$(grep -h 'tb_yee1d rtl' "$OUT/rtl_nominal.log" "$OUT/rtl_saturate.log" | grep -c 'PASS')
fi
passed=$(grep -o 'test result: ok\. [0-9]* passed' "$OUT/test.log" | awk '{s+=$4} END {print s+0}')
failed=$(grep -o '[0-9]* failed' "$OUT/test.log" | awk '{s+=$1} END {print s+0}')
cat > "$OUT/receipts.json" <<JSON
{
  "commit": "$COMMIT",
  "clean_checkout": true,
  "toolchain": "$(head -2 "$OUT/toolchain.log" | tr '\n' ';')",
  "gpu": "$(sed -n 3p "$OUT/toolchain.log")",
  "tests_passed": $passed,
  "tests_failed": $failed,
  "viewer_build": true,
  "stage5_fixture_identical": $( [ -s "$OUT/stage5_fixture.diff" ] && echo false || echo true ),
  "release_layout_rebuilt_byte_identical": $layout_same,
  "rtl_testbench_cases_passed": "$rtl of 2"
}
JSON
cat "$OUT/receipts.json"
