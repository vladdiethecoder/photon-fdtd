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

| Phase 1 run (r_f 6, full resolution) | done | `docs/results/stage6/phase1/rf6/` | 96 iterations, 2502 charged (cap 3000), 1596.8 s; best scored mask: iteration 3, J 0.2737 dB |
| Fallback rule | done | `docs/results/stage6/phase1/decision.txt` | the r_f 6 design meets the training gates, so the r_f 8 fallback did not run |
| Release gates (training, 61 held-out, half-cell grid) | done (pass) | `docs/results/stage6/phase1/validate/validate.json` | robust IL 0.2737 / 0.2737 / 0.2724 dB; robust RL 22.50 / 22.31 / 22.59 dB; geometry gates pass |
| Win over the best tested baseline | failed | same file, `win_tests` | gain 0.0569 / 0.0569 / 0.0575 dB, required >= 0.10 dB; best result preserved (phase 1 design) |
| Measured grid change reported | done | same file, `grid_change_il_db`, `grid_change_rl_db` | IL within +-0.0026 dB; RL -1.31 to +2.18 dB |
| Length-scale failure recorded | done | `docs/results/stage6/phase1/diagnostic_final/` | 23/25 scored masks fail the 5-cell rule; final iterate 0.0877 dB, not feasible (post-hoc diagnostic) |
| Layout export (SVG, GDSII, 4 layers) | done | `release/bend80.gds`, `release/bend80.svg`, `release/bend80_meta.json` | byte-identical on a second run |
| Independent re-import, area, symmetric difference, rules | done (pass) | `release/bend80_certificate.json` | `tools/check_layout.py`: 0 symmetric difference on layers 1 to 3; width, gap and connectivity agree with Rust |
| Negative controls | done | same certificate; mutated-GDSII check | shifted mask: 176 cells detected; corrupted coordinate: check fails (exit 1) |

## Phase 2: benchmark

| Gate | Status | Artifact | Evidence |
|---|---|---|---|
| Tuning allowance (6 runs, cap 300, seed 0, sharp L) | done | `out/stage6/phase2/tuning/chosen.json` | best J at 300: adjoint lr 0.05 18.09 vs 0.10 19.61; CMA-ES sigma0 0.30 15.03 vs 0.15 21.23; PPO lr 0.05 20.01 vs 0.02 21.94 (no tuning run reached a feasible design below J 15); tuning cost 1778 charged, 1376 s wall |
| Production (15 runs, cap 1200) | done | `docs/results/stage6/phase2/runs/` | 15/15 complete; adjoint median J 0.2166 dB (0.1711 to 0.2949); CMA-ES median 2.0659 dB, no run meets the gates; PPO median 0.3766 dB, all 5 runs stopped at the 20000-candidate rule after 153 to 396 charged (inconclusive) |
| Cost accounting | done | `events.csv` and `summary.json` of each run | charged = physical solves (26 per adjoint iteration, 3 per score); forward and adjoint solves logged separately; cache hits 0; measured adjoint/forward wall-time ratio 0.852 |
| Finalists on held-out sweep and grid | done | `docs/results/stage6/phase2/validate/` | adjoint seed 5: pass, win (0.1595 / 0.1588 / 0.1611 dB); CMA-ES seed 5: fail (RL 19.75 dB half-cell); PPO seed 1: pass, no win (0.0045 dB) |
| Combined release validation | done | `docs/results/stage6/release_validate/` | identical to the phase 1 and phase 2 validation files (7/7 design entries) |
| Release design choice | done (disclosed) | `release/release.json` | lowest training-set J among designs that pass every gate: adjoint seed 5; the rule was stated after the results; it does not use held-out data |
| Lead plot: optimiser trade-offs | done | `release/tradeoff.svg`, `.png` | all 15 runs drawn; curves end where runs end; medians only while all 5 runs run; checked by eye |

## Phase 3: package and showcase

| Gate | Status | Artifact | Evidence |
|---|---|---|---|
| Lead plot: validated spectra | done | `release/spectra_heldout61.svg`, `.png` | `tools/plot_stage6.py spectra`; checked by eye: no curve crosses a gate |
| Field movie from solver output | done | `out/stage6/movie/` (`field_movie`) | 150 frames, VP9 and H.264, poster; checked by eye: guided field through the bend |
| Project page and portfolio index | done (draft) | `dist/site/` (`tools/build_site.py`) | 15 local links resolve; downloads byte-identical; JS renders 6 curves; play, pause, scrub (1.5 / 3.0 / 5.4 s) and MP4 fallback work with a Range-capable server (`tools/serve_site.py`); first screen correct at 1280 and 390 px |
| Clean-checkout receipts | done | `release/receipts/receipts.json` | see the release-candidate commit; 24 tests pass, viewer builds, Stage 5 fixture identical, layouts rebuild byte-identical |
| Layout of the release design | done (pass) | `release/bend80_certificate.json` | nominal: 0 violations, 0 symmetric difference; eroded: 4 width; dilated: 2 width, 2 gap (reported); negative control 178 cells |

## Phase 4: range

| Gate | Status | Artifact | Evidence |
|---|---|---|---|
| Fixed-point 1D Yee RTL, reference, testbench | done (RTL not simulated) | `rtl/`, `src/fixed1d.rs`, `docs/rtl.md` | `fixed_point_yee1d_reference_and_vectors`: vectors match the reference; local error E 0.750 LSB (bound 1), H 0.500 LSB (bound 0.5); saturation at step 59 |
| RTL simulation | blocked | none | no HDL simulator installed; install needs approval |
| Synthesis figures | not measured | none | no synthesis tool |
| Lean proof (optional) | blocked | none | Lean 4 not installed; install needs approval |
| Outreach drafts (5, including Waterloo) | done (drafts, not sent) | `out/outreach/emails.md` (not committed) | contacts and research connections checked on official pages |
