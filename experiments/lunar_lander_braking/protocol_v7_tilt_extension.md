# v7 initial-tilt extension

This is a post-result exploratory extension of the amended v7 study. The original
held-out results have already been inspected. This is not a new independent
confirmatory replication and does not replace the original grid or results.

## Design

Cross each of the 18 original v7 held-out scenes with initial angles -4, 0, and
+4 degrees (stored in radians). Preserve initial angular velocity at zero and
all other initial-state values, terrain seeds, and engine-noise seeds. Initial
tilts lie within the training range of +/-0.08 radians. Shared random seeds
provide matched initial conditions, not identical future states under different
policies. The zero-angle arm is a rerun baseline, not additional independent data.

Reuse all 60 checkpoints from the hash-locked amended selection, without further
selection or training. Preserve rewards and detectors. Evaluate both existing
speed interventions: original and half initial descent speed. Total: 54 scenes
x 60 models x 2 interventions = 6,480 episodes.

The question is whether the timing contrast persists when initial posture must
also be controlled. Neither onset ordering nor landing quality may be used to
choose a different angle after results are observed.

## Outputs and interpretation

Retain the standard episode outcomes, all onset detectors, rejection candidates,
and full stepwise trajectories (angle, angular velocity, position, velocity,
action, and contact). Add precontact side-action count, peak absolute angle,
final precontact angle, and horizontal range. Stop posture diagnostics at first
contact to avoid conflating landing bounces with airborne attitude control.

Report landing/crash/timeout/out-of-bounds/technical-error counts and observed
onset denominators separately by angle, condition, and speed intervention before
interpreting timing. For descriptive timing comparisons, compute paired scene
ECON-minus-DESC differences within each angle and intervention, then medians per
training seed and across seeds. Report joint-event counts; do not impute absent
events or treat the three angles as independent training seeds. Also compare each
tilted scene with its matched zero-angle baseline within condition. No new
confirmatory significance claim is specified by this extension.

Side-engine use alone is not evidence that the policy acts specifically to
correct angle: side thrust also changes horizontal motion. Persistent-thrust
ONSET measures sustained main-engine action, not guaranteed net deceleration.

## Running

From the repository root, on the machine holding the selected model files:

```sh
python scripts/lunar_v7_tilt_extension.py prepare
qsub scripts/katana_lunar_v7_tilt_eval.pbs
```

For one model, use `python scripts/lunar_v7_tilt_extension.py evaluate --condition
DESC --seed 2001`. Preparation locks the config, scenario grid, protocol, runner,
original freeze, amendment, and selected-model manifests before evaluation.
Checkpoint resolution verifies the original selection and model hashes. Results
go to `outputs/lunar_lander_braking_formal_v3_persistent_onset/v7_tilt_extension/`.
Preparation and evaluation refuse existing destinations. A partial failed run
must be inspected before deliberately moving its output aside and retrying.
