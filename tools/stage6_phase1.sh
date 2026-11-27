#!/usr/bin/env bash
# Stage 6 phase 1 (docs/stage6-protocol.md, sections 6 and 7): baselines are
# in out/stage6/baselines (stage6_baselines). This script runs the r_f = 6
# full-resolution adjoint run, the pre-declared r_f = 8 fallback if the
# r_f = 6 design fails a training gate, and the held-out validation.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=out/stage6
B=./target/release
best_baseline=$(cut -d' ' -f1 $OUT/baselines/best_baseline.txt)
run() { # rf dir
  $B/stage6_adjoint --rep full --rf "$1" --lr 0.05 --betas 1,2,4,8,16,32,64,128 --iters 12 --score-every 4 --cap 3000 --init arc --out "$2" > "$2.log" 2>&1
}
passes() { # dir -> exit 0 if the chosen design meets the training gates
  python3 -c "import json,sys; s=json.load(open('$1/summary.json')); sys.exit(0 if s['best_il_db'] is not None and s['best_il_db'] < 0.5 and s['best_rl_db'] >= 22.0 else 1)"
}
mkdir -p $OUT/phase1
git rev-parse HEAD > $OUT/phase1/revision.txt
run 6 $OUT/phase1/rf6
opts=(--opt "phase1 r_f 6:full:6:$OUT/phase1/rf6/best_x.txt")
if passes $OUT/phase1/rf6; then
  echo "r_f 6 design meets the training gates; no fallback" | tee $OUT/phase1/decision.txt
else
  echo "r_f 6 design fails a training gate; fallback r_f 8" | tee $OUT/phase1/decision.txt
  run 8 $OUT/phase1/rf8
  opts+=(--opt "phase1 r_f 8:full:8:$OUT/phase1/rf8/best_x.txt")
fi
$B/stage6_validate "${opts[@]}" --baseline "$best_baseline" --baseline arc --out $OUT/phase1/validate > $OUT/phase1/validate.log 2>&1
echo done
