# v7 nine-angle extension

This is a second post-result exploratory extension. The original amended v7
held-out result and the first three-angle extension have already been inspected.
This analysis cannot create a new independent confirmatory claim.

## Design

Preserve the completed -4, 0, and +4 degree evaluations. Add -3, -2, -1, +1,
+2, and +3 degrees, crossing each angle with all 18 original held-out scenes.
All angles lie within the training distribution of +/-0.08 radians. Preserve
initial angular velocity at zero, all other scenario values, terrain and engine
noise seeds, selected checkpoints, rewards, detectors, and the original and
half-speed interventions.

The new run contains 18 scenes x 6 angles x 60 models x 2 interventions =
12,960 episodes. Combined with the prior 6,480 episodes, the nine-angle report
contains 19,440 episodes. The fixed symmetric grid provides uniform coverage and
matched comparisons without random-angle sampling variability.

## Analysis

For each angle and intervention, calculate matched-scene ECON-minus-DESC onset
differences, take the median of 18 differences within each training seed, then
the median across 20 seeds. For the combined estimate, take the median of all
18 x 9 matched scene-angle differences within each seed before summarizing the
20 seed medians. Bootstrap whole seed medians with 10,000 replicates and seed
260916. Angles and scenes are repeated conditions, not independent sample units.

Always report outcome and primary-event denominators by angle, condition, and
intervention. Do not impute absent events. Preserve the posture diagnostics from
the first extension and record the same compact precontact diagnostics for the
six additional angles. Side-engine use is not uniquely attributable to angle
correction because it also changes horizontal motion.

## Running

After the completed `v7_tilt_extension` is present, run from the repository root:

```sh
bash scripts/submit_katana_v7_angle_grid.sh
```

The submission script prepares and locks the six-angle manifest, submits the
60-model evaluation array, and schedules aggregation only after every array task
succeeds. Outputs are written under
`outputs/lunar_lander_braking_formal_v3_persistent_onset/v7_angle_grid_extension/`.
The combined machine-readable result and report are `aggregate/summary.json` and
`aggregate/report.md`. Preparation, model evaluation, and aggregation refuse to
overwrite existing destinations.
