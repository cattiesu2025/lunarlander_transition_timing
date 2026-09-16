# LunarLander v7 amended held-out results

Analysis date: 2026-09-17 AEST. Source: the 60 amended held-out `episodes.jsonl` files, `aggregate_amended_original_only/summary.json`, and the locked amended selection under `outputs/lunar_lander_braking_formal_v3_persistent_onset/`. The supplementary calculations can be reproduced with:

```bash
conda run -n lunar-lander python scripts/summarize_lunar_v7_amended.py \
  --output outputs/lunar_lander_braking_formal_v3_persistent_onset/aggregate_amended_original_only/supplementary_checks.json
```

The combined SHA-256 digest of the sorted episode-file paths and their SHA-256 hashes is `4e06642ea96b02b4698c98d69dbb76288e934a44b4f2cfc3a612f9088805f494`. The original aggregate summary hash is `24fad3f1c19eb8001262983346f56ec63f3165d83b45cd7f417800900628d197`; the amended selection manifest hash is `63adc22bed8dd5019fbfa24d3cc7b56c0fab8e680678dff3b2b14f984c4a86c9`.

## Analysis population and primary result

All 60 condition-by-seed models have one selected checkpoint. The held-out grid contains 18 matched scenes under each of two interventions for each model: 2,160 unique episodes in total, with no missing, duplicate, unexpected, or technical-error records. All 20 seeds have 18/18 jointly observed original-scenario DESC/ECON persistent-firing onsets, exceeding the declared 12/18 minimum. The aggregate and an independent calculation from the episode rows agree.

The amended v7 primary estimate is the median across 20 seed-level medians of matched-scene `tau_ECON - tau_DESC`: **+0.31 s**, whole-seed bootstrap 95% CI **[+0.24, +0.35] s**, with **20 positive, 0 negative** seed medians. The supplementary two-sided exact sign-test value is `p = 1.9073486328125e-6`. At the LunarLander action interval of 0.02 s, the point estimate corresponds to 15.5 action steps; the half step arises from medians.

## Detector and confirmation-window checks

The table applies the same paired-scene, then seed-median calculation to each original-scenario detector. Every listed detector was observed in 360/360 episodes per condition, giving 18/18 jointly observed DESC/ECON events per seed. Intervals outside the primary row are **descriptive**, unadjusted seed-bootstrap intervals and are not additional confirmation tests.

| Original-scenario endpoint | Median ECON − DESC (s) | Descriptive 95% interval (s) | Positive / negative / tied seeds |
| --- | ---: | ---: | ---: |
| First main-engine firing | +0.340 | [+0.235, +0.365] | 20 / 0 / 0 |
| Sustained firing, 0.20 s window | +0.330 | [+0.240, +0.360] | 20 / 0 / 0 |
| **Sustained firing, 0.30 s (primary)** | **+0.310** | **[+0.240, +0.350]** | **20 / 0 / 0** |
| Sustained firing, 0.40 s window | +0.300 | [+0.250, +0.335] | 20 / 0 / 0 |
| Effective braking, 0.20 s window | +0.010 | [−0.100, +0.090] | 10 / 9 / 1 |
| Effective braking, 0.30 s window | +0.005 | [−0.115, +0.165] | 10 / 8 / 2 |
| Effective braking, 0.40 s window | +0.070 | [−0.075, +0.120] | 13 / 7 / 0 |

The sustained-action ordering is stable across the prespecified 0.20 and 0.40 s windows. The effective-braking endpoint does **not** show a consistent later ECON onset. Its interval includes both directions; this is not an equivalence result. The main claim should refer to the start of **sustained engine action**, not to the start of net descent-speed reduction.

As a post-result diagnostic, effective braking and sustained firing began at the same candidate step in 37/360 DESC episodes, 165/360 BAL episodes, and 241/360 ECON episodes under the original scenario. The median reduction in downward-speed magnitude over the first sustained-firing window was 0.162, 0.365, and 0.746 world-units/s, respectively. This explains how earlier DESC engine use can coexist with a similar effective-braking onset; it does not establish why the policies learned that behavior.

## Halved initial descent speed

The `low_descent_speed` intervention preserved persistent-firing occurrence in all 360/360 episodes per condition: every original-to-low-speed event transition was observed-to-observed. For each seed, the table takes the median of 18 matched-scene `tau_low − tau_original` differences, then the median across 20 seeds. Intervals are descriptive.

| Condition | Median delay (s) | Descriptive 95% interval (s) | Positive seed medians |
| --- | ---: | ---: | ---: |
| DESC | +0.080 | [+0.060, +0.085] | 20/20 |
| BAL | +0.080 | [+0.080, +0.085] | 20/20 |
| ECON | +0.085 | [+0.080, +0.090] | 20/20 |

The low-speed `tau_ECON − tau_DESC` sustained-firing contrast was +0.320 s (descriptive interval [+0.235, +0.340] s; 20/20 positive seed medians). Effective-braking contrast under low speed was +0.005 s (descriptive interval [−0.055, +0.155] s). These intervention results support event responsiveness and timing-direction stability; they do not replace the original-scenario primary estimate.

## Terminal outcomes

All entries are counts out of 360 episodes per condition and intervention. Terminal outcome is reported separately from whether an onset was observed; all onset detectors above had 360/360 events even when landing failed.

| Condition | Intervention | Landed | Crash | Timeout |
| --- | --- | ---: | ---: | ---: |
| DESC | Original | 352 | 1 | 7 |
| DESC | Low descent speed | 352 | 1 | 7 |
| BAL | Original | 355 | 0 | 5 |
| BAL | Low descent speed | 353 | 3 | 4 |
| ECON | Original | 345 | 13 | 2 |
| ECON | Low descent speed | 342 | 15 | 3 |

ECON has more crashes and fewer landings than DESC in this held-out grid. The timing finding therefore should not be described as improved task performance or safety.

## Protocol status and manuscript wording

The frozen v7 gate required at least 15/18 development landings and primary events under **both** interventions. It selected 59/60 models; ECON seed 2011 had no eligible checkpoint, so sealed held-out evaluation did not start. The dated `amendment_v7_original_only.md` changed the gate uniformly to require those counts only under original development scenes, while still requiring complete, technically valid low-speed records. The amended gate selected 60/60 models before the sealed v7 held-out evaluation. Compared with the failed original selection, ECON seed 2011 gained its 600k checkpoint and ECON seed 2018 moved from 700k to 900k; the other 58 choices were unchanged. The paper must state that this was a **post-freeze, pre-held-out** amendment based on development task-quality counts. The result is not a wholly preregistered v7 result.

Possible Results wording:

> After a documented pre-held-out amendment to the checkpoint eligibility gate, ECON initiated sustained main-engine firing 0.31 s later than DESC on matched original held-out scenes (20 seed-level median contrasts; seed-bootstrap 95% CI [0.24, 0.35] s; 20/20 positive seeds). The ordering persisted with 0.20 and 0.40 s confirmation windows and under halved initial descent speed. In contrast, effective-braking onset showed no consistent ECON–DESC separation (median +0.005 s; descriptive 95% CI [−0.115, +0.165] s). The result concerns the timing of persistent engine action; terminal outcomes and the development-gate amendment are reported separately.
