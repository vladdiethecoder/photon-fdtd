#!/usr/bin/env bash
# Stage 6 phase 2 benchmark (docs/stage6-protocol.md, section 8).
# 1. Tuning allowance: 2 configurations per method, seed 0, start sharpL, cap 300.
# 2. Production: seeds 1..5 (seed i starts from Ii), cap 1200, the chosen configuration.
# 3. Finalists (lowest J of each method over its 5 runs): held-out validation.
# Runs are sequential on one GPU. Re-running skips runs that have a summary.json.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=out/stage6/phase2
B=./target/release
mkdir -p $OUT/tuning $OUT/runs
git rev-parse HEAD > $OUT/revision.txt

adjoint() { # lr seed init cap dir
  $B/stage6_adjoint --rep grid20 --rf 6 --lr "$1" --betas 2,4,8,16,32,64 --iters 6 --hold-last --score-every 2 --cap "$4" --init "$3" --out "$5"
}
bb() { # method param value seed init cap dir
  $B/stage6_bb --method "$1" "$2" "$3" --seed "$4" --init "$5" --cap "$6" --out "$7"
}
once() { # dir command...
  local dir=$1; shift
  if [ -f "$dir/summary.json" ]; then echo "skip $dir"; return; fi
  mkdir -p "$dir"
  "$@" > "$dir.log" 2>&1
}
best_j() { python3 -c "import json; v=json.load(open('$1/summary.json'))['best_j']; print(v if v is not None else 1e9)"; }
pick() { # dirA valueA dirB valueB -> the value with the lower best J (A if equal)
  python3 -c "import sys; a,b=float('$(best_j $1)'),float('$(best_j $3)'); print('$2' if a<=b else '$4')"
}

# 1. Tuning
once $OUT/tuning/adjoint_lr0.05 adjoint 0.05 0 sharpL 300 $OUT/tuning/adjoint_lr0.05
once $OUT/tuning/adjoint_lr0.10 adjoint 0.10 0 sharpL 300 $OUT/tuning/adjoint_lr0.10
once $OUT/tuning/cma_s0.15 bb cma --sigma0 0.15 0 sharpL 300 $OUT/tuning/cma_s0.15
once $OUT/tuning/cma_s0.30 bb cma --sigma0 0.30 0 sharpL 300 $OUT/tuning/cma_s0.30
once $OUT/tuning/ppo_lr0.02 bb ppo --lr 0.02 0 sharpL 300 $OUT/tuning/ppo_lr0.02
once $OUT/tuning/ppo_lr0.05 bb ppo --lr 0.05 0 sharpL 300 $OUT/tuning/ppo_lr0.05
A_LR=$(pick $OUT/tuning/adjoint_lr0.05 0.05 $OUT/tuning/adjoint_lr0.10 0.10)
C_S0=$(pick $OUT/tuning/cma_s0.15 0.15 $OUT/tuning/cma_s0.30 0.30)
P_LR=$(pick $OUT/tuning/ppo_lr0.02 0.02 $OUT/tuning/ppo_lr0.05 0.05)
printf '{"adjoint_lr": %s, "cma_sigma0": %s, "ppo_lr": %s}\n' "$A_LR" "$C_S0" "$P_LR" | tee $OUT/tuning/chosen.json

# 2. Production
for i in 1 2 3 4 5; do
  once $OUT/runs/adjoint_$i adjoint "$A_LR" $i I$i 1200 $OUT/runs/adjoint_$i
  once $OUT/runs/cma_$i bb cma --sigma0 "$C_S0" $i I$i 1200 $OUT/runs/cma_$i
  once $OUT/runs/ppo_$i bb ppo --lr "$P_LR" $i I$i 1200 $OUT/runs/ppo_$i
done

# 3. Finalists
opts=()
for m in adjoint cma ppo; do
  best=$(python3 -c "
import json
r=[(json.load(open(f'$OUT/runs/${m}_{i}/summary.json'))['best_j'] or 1e9, i) for i in range(1,6)]
print(min(r)[1] if min(r)[0] < 1e9 else 'none')")
  if [ "$best" = none ]; then echo "$m: no feasible design in any run; no finalist" | tee -a $OUT/finalists.txt; continue; fi
  echo "$m finalist: seed $best" | tee -a $OUT/finalists.txt
  opts+=(--opt "$m seed $best:grid20:6:$OUT/runs/${m}_$best/best_x.txt")
done
best_baseline=$(cut -d' ' -f1 out/stage6/baselines/best_baseline.txt)
$B/stage6_validate "${opts[@]}" --baseline "$best_baseline" --out $OUT/validate > $OUT/validate.log 2>&1
echo done
