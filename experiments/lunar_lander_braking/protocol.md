# Formal Protocol — v6 high-height latest-eligible

Status: **approved for formal freeze after the v6 pilot task-quality review**. No confirmation data have been opened. The resulting bundle and checksums define the formal run; editing this file does not alter an already frozen bundle.

## Environment and timing

- Gymnasium `LunarLander-v3`, discrete actions, gravity `-10`, wind disabled.
- Controlled reset changes the Box2D body assembly, not only the observation.
- One action per native step: `dt = 1/50 = 0.02` simulated seconds.
- Position/height: Box2D world units; velocities: Box2D world units/s.
- Terrain and engine-dispersion seeds are separately recorded.

## Reward

The custom per-step reward is the sum of the following components. Here,
`dt=0.02` s, `v_down=max(0, -vy)`, and `I(·)` is an indicator function.

| Component | Per-step value | Purpose |
| --- | --- | --- |
| Common shaping | Change in `-100 hypot(x_norm, y_norm) - 100 abs(angle) + 10 left_contact + 10 right_contact` | Rewards proximity to the pad, upright attitude, and leg contact. The native velocity term is removed. |
| Side-engine cost | `-0.03 I(action in {1, 3})` | Discourages unnecessary lateral thrust. |
| Downward-speed cost | `-w_downward * downward_scale * (v_down / downward_reference_speed)^2 * dt` | Penalizes downward speed quadratically. |
| Main-engine cost | `-w_main * main_engine_scale * I(action = 2) * dt` | Penalizes main-engine use. The native main-engine charge is removed. |
| Time cost | `-time_scale * dt` | Discourages hovering and timeouts. With `time_scale=5.0`, this is `-0.1` per step and `-100` over the 1000-step horizon. |
| Settling-actuation cost | `-settling_actuation_scale * I(action != 0 and both legs contacted before the action) * dt` | Discourages post-contact actuation loops. With `settling_actuation_scale=5.0`, this is `-0.1` for an applicable step and zero before two-leg contact or for action 0. |
| Terminal reward | `+100` for landing; `-100` for other native terminal outcomes | Preserves the native terminal reward. |

The reward-condition weights are:

| Condition | `w_downward` | `w_main` | Intended trade-off |
| --- | ---: | ---: | --- |
| `DESC` | 1.000 | 0.250 | Places more cost on descending quickly and less on main-engine use. |
| `BAL` | 0.625 | 0.625 | Balances downward-speed and main-engine costs. |
| `ECON` | 0.250 | 1.000 | Places less cost on descending quickly and more on main-engine use. |

At the reference downward speed (`v_down=2.0`) with continuous main-engine
firing, the condition-dependent costs per simulated second are:

| Condition | Downward-speed cost/s | Main-engine cost/s | Total/s |
| --- | ---: | ---: | ---: |
| `DESC` | `-10.00` | `-2.50` | `-12.50` |
| `BAL` | `-6.25` | `-6.25` | `-12.50` |
| `ECON` | `-2.50` | `-10.00` | `-12.50` |

Thus, the total condition-dependent cost is matched at the reference speed;
only its allocation between downward speed and main-engine use changes. The
weights and scales are recorded in the frozen YAML. All added terms are
integrated per simulated second, and native return is diagnostic only.

## Endpoints

Primary ONSET is the earliest descending, pre-contact main-engine firing step whose 0.30 s forward window has at least 0.60 firing fraction, at least 0.40 world-units/s reduction in downward-speed magnitude, and no contact/terminal/out-of-bounds step. ONSET time is the candidate time; confirmation time is the final window record. Candidate rejections are retained.

Auxiliary endpoints are first main-engine firing and firing-fraction-only sustained firing. Sensitivity windows are 0.20 and 0.40 s with the same fraction and motion threshold. Fixed-action calibration found first-window downward-speed-magnitude changes of −2.993 (no fire), −2.788 to −2.486 (single pulse), and +0.800 to +2.400 world-units/s (sustained main). `epsilon_v=0.40` is the nonnegative midpoint to the weakest sustained-probe change. The v6 pilot retained primary events at the task-valid selected checkpoints; neither onset times nor condition contrasts were used for model selection.

## Training and evaluation

- Training uses Double DQN implemented by inheriting Stable-Baselines3 `DQN` and overriding `train()` so the online network selects the next action and the target network evaluates it. All other replay, exploration and serialization machinery remains SB3 2.9.0.
- The v6 pilot used seeds 101/202/303 under all three conditions. Every model had at least one eligible checkpoint on the high development grid, so the fixed task-quality gate permits formal freeze. Pilot models and pilot onset contrasts are excluded from confirmation.
- Formal training uses seeds 1001–1020 under DESC/BAL/ECON. Every model trains to a maximum of 1,000,000 environment steps and saves fixed checkpoints at 600k, 700k, 800k, 900k and 1M.
- The continuous training height range is 7.8–12.2 world-units. Development heights are 8/10/12. The sealed held-out heights are 8.5/10.5/11.5, with non-overlapping speed/offset axes and independent terrain/noise seeds.
- For each formal `condition × seed`, inspect checkpoints newest to oldest on development data and select the latest checkpoint for which original and low-speed intervention each have at least 15/18 landed outcomes, at least 15/18 primary events, 0 technical errors and complete matching hashes. Selection does not read onset time, condition contrasts, significance or training return.
- Held-out evaluation is forbidden until all 60 models have a selected checkpoint and the selected-model manifest is SHA-256 locked. Each selected model is then evaluated once on all 18 sealed scenes under original and `|vy| × 0.5` intervention.
- A successful sleep termination is `landed`; body contact/game-over is `crash`; horizontal state beyond bounds is `out_of_bounds`; TimeLimit is `timeout`.

## Confirmatory inference

For each seed, calculate the median `tau_ECON - tau_DESC` over matched held-out scenes with both primary events. All 20 seeds must have at least 12/18 joint events. The study estimate is the median of seed estimates with a 10,000-replicate seed bootstrap percentile 95% CI using seed 260910. A two-sided exact sign test is supplementary. Event denominators, terminal outcomes, technical errors, intervention transition counts and window sensitivity are always reported. No-event episodes receive no artificial time.

## Freeze gate

The v6 pilot satisfied the predeclared task-validity gate for all nine pilot models, with 0 integrity errors. Formal freeze fixes `formal_v6_high_latest.yaml`, `development_high.json`, `held_out_high.json`, this protocol, environment specifications and dependency locks. After freeze, do not change seeds, heights, reward, detector, budget, checkpoint schedule or gate. Do not use the anticipated DESC–ECON ordering as a gate. Held-out results may be rerun only for documented technical incompleteness under identical frozen inputs.
