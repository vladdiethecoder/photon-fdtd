# Stage 6 progress ledger

Each line is one gate or step, the artifact that holds the evidence, and the
command or check that supports it. Status: done, failed, blocked, open.
Paths under `out/` are local working outputs. The release copies are under
`docs/results/stage6/` and `release/`.

## Phase 0: baseline

| Gate | Status | Artifact | Evidence |
|---|---|---|---|
| Branch, tags and worktree checked | done | `git branch -a`, `git tag -l -n1` | `v0.5.0` = `master` = `60197eb`; work on the local branch `stage6-rc`, from `4177246` |
| `1187512` is the Stage 5 result commit | done | `git merge-base --is-ancestor 1187512 v0.5.0`; `git diff --stat 1187512 v0.5.0` | ancestor of `v0.5.0`; the only difference is the version bump in `Cargo.toml` and `Cargo.lock` |
| Stage 5 fixture preserved | done | `docs/results/stage5/rho_final.txt` | SHA-256 `ac8ee8de…df40a7` at `v0.5.0` and at HEAD; `git diff --quiet v0.5.0 -- docs/results/stage5` |
| Stage 5 regression rerun | done | `out/phase0/test.log`, `out/phase0/validate.log` | `cargo test --release`: 21 passed; `stage5_validate` on the fixture is identical to `docs/results/stage5/validate.log` |
| Pre-protocol 80 x 80 runs stopped | done | `out/opt_80_rf8.log`, `out/opt_80_rf10.log` | nominal-only runs started before any Stage 6 protocol; stopped at beta 128 iteration 14 and not used for any gate or choice |
| Protocol committed before production | done | `docs/stage6-protocol.md` | v2 `16fe4cc`, v2.1 `ffc572f` (T1 definition corrected), both before the first production run; v1 (`docs/stage6-benchmark-protocol.md`) superseded, no production run |
| Harness smoke tests, separate from production | done | `docs/results/stage6/smoke/` | adjoint (cap 60), CMA-ES and PPO (cap 45), validation; charge = physical solves (26 per adjoint iteration), cache hits charged 0 |

## Phase 1: layout loop (80 x 80)

| Gate | Status | Artifact | Evidence |
|---|---|---|---|
| Reflection adjoint verified | done | `adjoint_reflection_gradient_matches_finite_differences` | cosine 1.00000000, magnitude ratio 1.000000, 16/16 signs at 3 wavelengths; directional ratios within 5e-8; single-line R within 0.6 % of the modal S11 |
| Geometry and bias rules tested | done | `stage6_geometry_and_bias_rules` | known cases at scale 1 and 2; disk d = 5 equals the Stage 5 disk |
| Baselines re-evaluated (21 designs) | done | `out/stage6/baselines/` | best tested baseline: Bezier k = 0.45, J = 0.3306 dB (robust IL 0.3306 dB, robust RL 23.21 dB); T2 = 0.2306 dB |
| eta calibration | done | `out/stage6/baselines/eta_calibration.json` | r_f 6: eta_e 0.641366, eta_d 0.358634 |

## Phase 2, 3, 4

Filled in as the steps finish.
