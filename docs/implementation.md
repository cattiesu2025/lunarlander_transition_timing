# Implementation Contract — LunarLander Powered-Braking Onset

> Build mode: from scratch. The scientific design in `lunar_lander_plan.md` is authoritative.

## Complete Tree

```text
README.md
environment.yml
requirements.txt
lunar_lander_plan.md
docs/{user_requirements.md,implementation.md,dev_log.md}
experiments/lunar_lander_braking/
  __init__.py config.py env.py events.py agent.py run.py protocol.md
  configs/{pilot,formal}.yaml
  grids/{development,held_out}.json
scripts/{aggregate_lunar_lander.py,katana_lunar_pilot_train.pbs,katana_lunar_pilot_eval.pbs,katana_lunar_train.pbs,katana_lunar_eval.pbs}
tests/{test_lunar_lander_env.py,test_lunar_lander_events.py,test_lunar_lander_aggregate.py}
notebooks/.gitkeep
outputs/.gitkeep
```

## File Responsibilities

| File | Responsibility |
|---|---|
| `config.py` | Typed YAML configuration loading, canonical JSON and SHA-256 hashes. |
| `env.py` | Controlled physical reset, reward decomposition, trajectory/outcome records. |
| `events.py` | ONSET and auxiliary detectors with candidate rejection reasons. |
| `agent.py` | MLP Double DQN, replay buffer, checkpoints and deterministic training loop. |
| `run.py` | `probe`, `recoverability`, `train`, `evaluate`, and `freeze` CLI modes. |
| `protocol.md` | Frozen endpoint, units, seeds, completeness and inference rules. |
| `configs/*.yaml` | Pilot/formal budgets, reward conditions and endpoint constants. |
| `grids/*.json` | Explicit 18-scenario development/held-out manifests. |
| `aggregate_lunar_lander.py` | Integrity checks, matched seed effects, bootstrap CI, sign test and intervention transitions. |
| `katana_lunar_*.pbs` | Katana pilot/formal array training and post-training evaluation. |
| `tests/*` | Endpoint, aggregation and Box2D reset/reward contracts. |

## Function-Level Design

### `config.py`

- `load_config(path: Path) -> dict`: parse YAML and validate required keys.
- `canonical_hash(value: Any) -> str`: stable SHA-256 of canonical JSON.
- `load_manifest(path: Path) -> list[dict]`: validate unique scenario IDs and physical fields.
- `write_freeze_bundle(config_path, manifest_paths, output_dir) -> dict`: copy immutable inputs and write checksums; formal callers also supply Conda/pip locks.

### `env.py`

- `Scenario`: physical Box2D-native state (`x`, height above pad, velocities, angle, angular velocity, terrain/noise seeds).
- `RewardWeights`: `downward_speed` and `main_engine` weights.
- `ControlledLunarLander.reset(seed, options)`: build terrain using `terrain_seed`, move lander and both joined legs as one rigid assembly, clear contact/terminal/reward history, reseed the time-indexed engine-dispersion stream, and return a synchronized observation.
- `ControlledLunarLander.step(action)`: invoke native physics, return the custom reward plus `reward_components` and physical state in `info`.
- `classify_outcome(...) -> str`: distinguish landed, crash, out-of-bounds, timeout and technical error.
- `rollout(...) -> dict`: record every step and keep endpoint separate from terminal outcome.

Observations remain the native normalized 8-vector. Logged positions and velocities are explicitly labelled Box2D world units and world-units/s. One action is one native physics step (`dt=1/50 s`); action repeat defaults to one.

### `events.py`

- `DetectorConfig(window_s, firing_fraction, epsilon_v, dt)` validates exact integer window steps.
- `detect_onset(records, config) -> DetectionResult`: earliest pre-contact descending main-engine candidate whose forward window has sufficient firing, reduced downward-speed magnitude by `epsilon_v`, and no invalid state.
- `detect_first_fire(records)`, `detect_sustained_fire(records, config)`: auxiliary detectors.
- Every main-engine candidate receives `confirmed` or rejection reasons; an incomplete terminal window is rejected.

### `agent.py`

- `QNetwork(obs_dim, actions, hidden_sizes)` implements the shared MLP.
- `ReplayBuffer` stores `(s,a,r,s',done)` arrays.
- `DoubleDQNAgent.act`, `update`, `save`, and `load` implement online action selection, Double-DQN targets, gradient clipping and checkpoint metadata.
- `train_condition(config, condition, seed, output_dir)` uses an episode-indexed scenario RNG shared across reward conditions and always saves the final checkpoint.

### `run.py`

- `probe`: fixed no-fire, single-fire and sustained-fire policies for calibration artifacts.
- `recoverability`: apply a fixed reference heuristic to every manifest scene and report outcomes/onsets without treating it as an optimality proof.
- `freeze`: checksum config and both manifests into a versioned output directory.
- `train`: verify frozen inputs for formal runs and train exactly one condition/seed (PBS array unit).
- `evaluate`: validate frozen inputs and checkpoint hash, evaluate a fixed checkpoint over all held-out scenarios at original and 0.5 downward-speed intervention, emit JSONL trajectories, episodes and detector records; persist technical errors separately and exit nonzero.

### `aggregate_lunar_lander.py`

- `read_jsonl`, `validate_completeness`, `seed_contrasts`, `bootstrap_seed_median`, `exact_sign_test`, and `aggregate` implement the predeclared estimand. No-event episodes never receive an artificial horizon time. A seed is estimable only with at least 12/18 joint DESC/ECON events.

## Data and Result Formats

Each manifest row contains `scenario_id`, native physical state, `terrain_seed`, and `noise_seed`. Evaluation writes:

- `episodes.jsonl`: one row per model/scenario/intervention with hashes, detector times and terminal outcome.
- `trajectories/*.jsonl.gz`: step records including time, action, position/velocity, contacts and reward components.
- `candidates.jsonl`: all ignition candidates and rejection reasons.
- `summary.json`, `seed_contrasts.csv`: event denominators, outcome counts, seed effects, bootstrap CI and sign-test result.

Models and all frozen inputs carry SHA-256 hashes. Technical errors are never merged with behavioral outcomes.

## Data Preparation

No external dataset is required. Gymnasium creates deterministic terrain from each manifest `terrain_seed`; controlled state and noise streams come from the remaining manifest fields.

## Implementation Order

1. Requirements, environment and frozen configs/manifests.
2. Configuration/hash utilities and controlled environment.
3. Event detectors and fixed-action probes.
4. Double DQN and train/evaluate CLI.
5. Aggregation, PBS scripts, tests and README.

## Pre-coding Validation

- ✅ Experiment coverage: main comparison, intervention, detector comparison, window sensitivity and probes all have explicit entry points.
- ✅ Logic consistency: environment and agent exchange native `(8,) float32` observations; actions are integers in `[0,3]`; timing is derived from `1/50 s` records.
- ✅ Completeness: every file in the tree has a responsibility and function-level implementation section or is a declarative artifact.
