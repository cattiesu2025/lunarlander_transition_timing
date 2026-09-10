# Frozen Protocol — candidate v3

Status: **pilot candidate, not confirmation-ready**. Run `freeze` only after fixed-action calibration and pilot task-quality review. The resulting bundle and checksums define a formal run; editing this file does not alter an already frozen bundle.

## Environment and timing

- Gymnasium `LunarLander-v3`, discrete actions, gravity `-10`, wind disabled.
- Controlled reset changes the Box2D body assembly, not only the observation.
- One action per native step: `dt = 1/50 = 0.02` simulated seconds.
- Position/height: Box2D world units; velocities: Box2D world units/s.
- Terrain and engine-dispersion seeds are separately recorded.

## Reward

Common shaping removes the native velocity term and main-engine charge. It retains position, attitude, leg-contact changes, side-engine cost, and native `+100/-100` terminal rewards. It applies a common `time_scale=5.0` reward-units/s penalty: `-0.1` per 0.02 s step and `-100` over the 1000-step horizon. Pilot v2 also applies a common `settling_actuation_scale=5.0` reward-units/s penalty (`-0.1` per step) only when both legs were already in contact before a nonzero action. This settling term is zero before contact and for action 0. Replacement downward-speed-squared and main-engine-use costs are integrated per simulated second. Both added terms are identical across conditions; only the declared downward-speed/main-engine weights differ. The weights and scales are in the frozen YAML; native return is diagnostic only.

## Endpoints

Primary ONSET is the earliest descending, pre-contact main-engine firing step whose 0.30 s forward window has at least 0.60 firing fraction, at least 0.40 world-units/s reduction in downward-speed magnitude, and no contact/terminal/out-of-bounds step. ONSET time is the candidate time; confirmation time is the final window record. Candidate rejections are retained.

Auxiliary endpoints are first main-engine firing and firing-fraction-only sustained firing. Sensitivity windows are 0.20 and 0.40 s with the same fraction and motion threshold. Fixed-action calibration over all 18 development scenarios found first-window downward-speed-magnitude changes of −2.993 (no fire), −2.788 to −2.486 (single pulse), and +0.800 to +2.400 world-units/s (sustained main). `epsilon_v=0.40` is the nonnegative midpoint to the weakest sustained-probe change. This remains a pilot freeze candidate until policy event semantics are reviewed.

## Training and evaluation

- Reference-controller screening landed all 18/18 development and 18/18 held-out scenarios, with 18/18 non-immediate primary events in each grid. This is a recoverability check, not an optimality proof.
- Pilot v3: seeds 101/202/303, each condition 500,000 steps. Training uses Double DQN implemented by inheriting Stable-Baselines3 `DQN` and overriding `train()` so the online network selects the next action and the target network evaluates it. Pilot v0 failed the all-condition task-quality gate because timeout policies were not charged the planned common time term; v1 improved DESC/ECON aggregate landing but exposed post-contact actuation loops; v2 fixed most BAL settling failures but ECON did not learn the task reliably.
- Formal: seeds 1001–1020, each condition 500,000 steps; final checkpoint only.
- Evaluation: all 18 held-out scenes, original and `|vy| × 0.5` intervention.
- A successful sleep termination is `landed`; body contact/game-over is `crash`; horizontal state beyond bounds is `out_of_bounds`; TimeLimit is `timeout`.

## Confirmatory inference

For each seed, calculate the median `tau_ECON - tau_DESC` over matched held-out scenes with both primary events. All 20 seeds must have at least 12/18 joint events. The study estimate is the median of seed estimates with a 10,000-replicate seed bootstrap percentile 95% CI using seed 260910. A two-sided exact sign test is supplementary. Event denominators, terminal outcomes, technical errors, intervention transition counts and window sensitivity are always reported. No-event episodes receive no artificial time.

## Freeze gate

Freeze only if reset/reward/event tests pass, fixed-action probes calibrate `epsilon_v`, development exposures are recoverable, all three pilot conditions learn the task, and sustained braking is a meaningful behavior. Do not use the anticipated DESC–ECON ordering as a gate.
