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
./target/release/export_layout --mask docs/results/stage6/phase1/validate/phase1_r_f_6_s1.mask --out "$SCRATCH/out/bend80" > "$OUT/layout_rebuild.log"
layout_same=true
for f in bend80.gds bend80.svg bend80_layer1.mask bend80_layer2.mask bend80_layer3.mask; do
  cmp -s "$SCRATCH/out/$f" "release/$f" || layout_same=false
done
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
  "release_layout_rebuilt_byte_identical": $layout_same
}
JSON
cat "$OUT/receipts.json"
