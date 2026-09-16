# Confirmatory Replication Protocol — persistent action ONSET

Status: **prepared for freeze before any replication training or held-out evaluation**. This replication was motivated by aligning the ONSET construct across the Highway and LunarLander domains. Its endpoint and directional hypothesis were fixed from that construct-level decision; the previously observed LunarLander held-out data are not part of this replication.

## Claim and estimand

The common paper-level construct is the start of a persistent target behavior after a matched exposure. For LunarLander, the target behavior is sustained main-engine firing. The primary ONSET is the earliest descending, pre-contact main-engine firing step whose 0.30 s forward window contains a main-engine firing fraction of at least 0.60 and contains no contact, terminal, out-of-bounds, or technical-error step. ONSET is dated at the candidate action time; confirmation occurs after the forward window.

For each training seed, the primary contrast is the median `tau_ECON - tau_DESC` over matched sealed scenarios with both persistent-thrust events. The directional hypothesis is `tau_ECON - tau_DESC > 0`: ECON begins sustained thrust later than DESC.

The former primary endpoint is retained under the explicit name **effective braking ONSET**. It adds the requirement that downward-speed magnitude fall by at least 0.40 world-units/s during the same window. It is a predeclared physical-effect validation endpoint and cannot replace the persistent-thrust primary result. First main-engine firing is descriptive. Window sensitivities are 0.20 and 0.40 s at the same 0.60 firing fraction.

## Independence from the previous study

- Training uses 20 new seeds, 2001–2020, under DESC, BAL, and ECON.
- The sealed replication grid uses new height, descent-speed, horizontal-offset, terrain-seed, and engine-noise-seed values. None appeared on the previous development or held-out grids.
- Previous trained policies and previous held-out episodes are excluded from the replication estimate.
- The development grid is used only for the task-quality checkpoint gate. It cannot be used for onset-time contrasts or directional selection.

## Training, selection, and sealed evaluation

The environment, reward conditions, Double DQN implementation, 1,000,000-step maximum budget, checkpoint schedule, and high-height training distribution are unchanged from formal v2. For each condition and seed, checkpoints are inspected newest to oldest. Eligibility requires, separately for original and low-speed development episodes, at least 15/18 landed outcomes, at least 15/18 primary persistent-thrust events, complete records, matching hashes, and zero technical errors. Selection does not read onset times or cross-condition contrasts.

All 60 models must have a selected checkpoint and the selection manifest must be SHA-256 locked before the sealed replication grid is evaluated. Each model is evaluated once on all 18 sealed scenarios under the original initial speed and the predeclared `|vy| x 0.5` intervention.

## Confirmatory inference

Every seed must have at least 12/18 jointly observed DESC/ECON primary events. The study estimate is the median of 20 seed-level paired medians. A 10,000-replicate percentile bootstrap over whole seeds uses seed 260916. The directional claim is supported only when the two-sided 95% interval for `tau_ECON - tau_DESC` lies wholly above zero. A two-sided exact sign test is supplementary.

No-event episodes receive no artificial latency. Event denominators, terminal outcomes, technical errors, low-speed intervention transitions, first-fire timing, effective-braking timing, and both sensitivity windows are always reported. Failure of completeness or estimability produces no confirmatory timing result.

## Freeze rule

Before training, freeze this protocol, `formal_v7_persistent_onset.yaml`, `development_high.json`, `held_out_persistent_v7.json`, environment specifications, and dependency locks into an immutable bundle. After freeze, do not change seeds, scenarios, rewards, endpoint definitions, detector thresholds, training budget, checkpoint schedule, selection gate, hypothesis direction, or inference settings. A failed task may be rerun only for documented technical incompleteness with identical frozen inputs.
