# Stage 6 benchmark protocol: black-box optimisers against the adjoint

Superseded by [docs/stage6-protocol.md](stage6-protocol.md) (version 2) before any production run. Only two smoke runs of 20 solves used this version.

This protocol is fixed before any black-box optimiser code exists. It compares
CMA-ES and a bandit policy gradient with the `v0.5.0` adjoint result on the
same 48 x 48 evaluator. All results are for the 2D TMz model.

## Evaluator

The evaluator is the Stage 5 evaluator without changes:

- `PortSetup::standard(Geometry::Bend, 1)` with `extra_time = 6000`;
- `DesignBox::for_bend(setup, 48, 8)`, conic filter radius 5;
- the binary design is the 0.5 level set of the filtered density;
- the 15 band samples, lambda0 = 115 to 200 cells;
- one shared reference run (`Run::Reference`), not counted.

## Cost unit

The cost unit is one FDTD run of the 48 x 48 setup.

- One adjoint gradient costs 12 runs: 1 forward run and 11 adjoint runs.
- The `v0.5.0` adjoint result cost 1921 runs: 8 beta values x 20 iterations x
  12 runs, and 1 run for the final binary design.
- A black-box candidate that fails the fabrication gates gets no FDTD run and
  costs 0 runs.

## Black-box parameterization

The latent density has 12 x 12 control values in [0, 1]. Bilinear
interpolation gives the 48 x 48 latent density. The conic filter and the 0.5
threshold then give the binary design. The start point is the circular arc
latent, averaged over each 4 x 4 block. Values outside [0, 1] are clipped
before evaluation.

The 12 x 12 grid cannot represent the adjoint design exactly. The comparison
thus measures optimiser and parameterization together. A full 2304-value
black-box search is not part of this protocol.

## Objective

For one candidate:

1. If the binary design has isolated 1-cell islands or L_min violations
   (`feature_violations` with `disk(4)`), J = 100 + islands + violations, and
   no FDTD run occurs.
2. Otherwise, run the design. J = worst-case IL (dB) over the 15 samples, plus
   (20 - worst-case RL) if the worst-case RL is less than 20 dB.

Lower J is better. A feasible design with RL >= 20 dB has J = worst-case IL.

## Optimisers

- CMA-ES (Hansen, default strategy parameters), population 18
  (4 + floor(3 ln 144)), sigma0 = 0.25.
- Bandit policy gradient: one-step Gaussian policy over the 144 control values
  (REINFORCE with antithetic samples, a mean baseline and rank-normalised
  returns), 16 antithetic pairs per update, policy standard deviation 0.10,
  Adam on the policy mean with learning rate 0.02. A one-step problem without
  state has no credit assignment, so this method is an evolution strategy.
  PPO, GRPO and multi-step policies are not part of this protocol.

## Budget and seeds

- Budget: 3842 runs for each optimiser and seed (2 x the adjoint cost).
- Seeds: 3 for each optimiser (seeds 1, 2 and 3).
- No tuning of the values in this file after the first black-box run. A
  change makes a new protocol version with a new commit before its runs.

## Reported values

For each optimiser and seed:

- the best J and its worst-case IL and RL;
- the run count at which the best J first reaches 0.2659 dB, which is the
  adjoint result plus the Stage 5 numerical uncertainty U = 0.0073 dB
  (the adjoint is "matched"), or "not reached";
- the curve of best J against the run count.

The best black-box design over all seeds goes through `stage5_validate`: 61
frequencies, scale 2, and 1-cell erosion and dilation.

## Claims

- A black-box optimiser matches the adjoint if its best feasible design has a
  worst-case IL <= 0.2659 dB within the budget.
- A black-box optimiser beats the adjoint only if its validated design is at
  least 0.10 dB lower than 0.2586 dB on the 15 samples, the 61 frequencies and
  scale 2, and the gain is larger than U.
- Otherwise, the result is reported as the measured gap and the run counts.
