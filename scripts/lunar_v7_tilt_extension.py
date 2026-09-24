#!/usr/bin/env python3
"""Prepare and evaluate the post-result v7 matched initial-tilt extension."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = Path('outputs/lunar_lander_braking_formal_v3_persistent_onset')
EXTENSION = STUDY / 'v7_tilt_extension'
CONFIG = Path('experiments/lunar_lander_braking/configs/formal_v7_persistent_onset.yaml')
BASE_GRID = Path('experiments/lunar_lander_braking/grids/held_out_persistent_v7.json')
PROTOCOL = Path('experiments/lunar_lander_braking/protocol_v7_tilt_extension.md')
SELECTION = STUDY / 'selection_amended_original_only/selected_checkpoints.json'
SELECTION_LOCK = SELECTION.with_name('selected_checkpoints.sha256.json')
FREEZE = STUDY / 'frozen/manifest.sha256.json'
AMENDMENT = Path('experiments/lunar_lander_braking/amendment_v7_original_only.md')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_grid(base):
    if len(base) != 18 or len({r['scenario_id'] for r in base}) != 18:
        raise ValueError('Expected 18 unique base scenarios')
    result = []
    for row in base:
        if row['angle'] != 0 or row['angular_velocity'] != 0:
            raise ValueError('Expected upright base scenarios with zero angular velocity')
        for degrees, suffix in ((-4, 'm04'), (0, 'z00'), (4, 'p04')):
            result.append({**row, 'scenario_id': row['scenario_id'] + '_tilt_' + suffix,
                           'angle': math.radians(degrees)})
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def prepare():
    inputs = [CONFIG, BASE_GRID, PROTOCOL, SELECTION, SELECTION_LOCK, FREEZE,
              AMENDMENT, Path(__file__).relative_to(ROOT)]
    hashes = {str(p): sha(p) for p in inputs}
    grid = build_grid(json.loads(BASE_GRID.read_text()))
    EXTENSION.mkdir(parents=True, exist_ok=False)
    manifest = EXTENSION / 'tilt_scenarios.json'
    write_json(manifest, grid)
    hashes[str(manifest)] = sha(manifest)
    write_json(EXTENSION / 'manifest.sha256.json', {
        'study_version': 'v7',
        'status': 'post-result exploratory v7 extension; not the original held-out study',
        'files': {CONFIG.name: sha(CONFIG), manifest.name: sha(manifest)},
        'source_files': hashes,
    })
    print(f'Prepared {len(grid)} matched scenarios at {EXTENSION}')


def posture_summary(records):
    airborne = []
    for row in records:
        if row.get('contact_before') or row.get('contact'):
            break
        airborne.append(row)
    if not airborne:
        return {'precontact_steps': 0}
    return {
        'precontact_steps': len(airborne),
        'precontact_side_actions': sum(r['action'] in (1, 3) for r in airborne),
        'precontact_peak_abs_angle_deg': max(abs(math.degrees(r['angle'])) for r in airborne),
        'precontact_final_angle_deg': math.degrees(airborne[-1]['angle']),
        'precontact_x_range': max([airborne[0]['x_before']] + [r['x'] for r in airborne])
                              - min([airborne[0]['x_before']] + [r['x'] for r in airborne]),
    }


def evaluate(condition, seed):
    lock_path = EXTENSION / 'manifest.sha256.json'
    lock = json.loads(lock_path.read_text())
    for path, expected in lock['source_files'].items():
        if sha(path) != expected:
            raise ValueError(f'Extension input changed: {path}')
    # Import the simulator only for actual evaluation; preparation is dependency-free.
    from scripts.resolve_lunar_checkpoint import resolve_checkpoint
    from experiments.lunar_lander_braking.run import command_evaluate
    checkpoint = resolve_checkpoint(CONFIG, SELECTION, SELECTION_LOCK, FREEZE,
                                    condition, seed, AMENDMENT)
    output = EXTENSION / 'eval' / condition / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'provenance.json', {'extension_lock_sha256': sha(lock_path),
                                          'checkpoint_sha256': sha(checkpoint)})
    command_evaluate(argparse.Namespace(
        config=CONFIG, manifest=EXTENSION / 'tilt_scenarios.json',
        freeze_manifest=lock_path, checkpoint=checkpoint,
        condition=condition, seed=seed, output=output))
    summaries = []
    for line in (output / 'episodes.jsonl').read_text().splitlines():
        episode = json.loads(line)
        with gzip.open(episode['trajectory'], 'rt') as handle:
            records = [json.loads(line) for line in handle]
        summaries.append({**episode, **posture_summary(records)})
    write_json(output / 'posture_diagnostics.json', summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--condition', choices=['DESC', 'BAL', 'ECON'], required=True)
    evaluation.add_argument('--seed', type=int, choices=range(2001, 2021), required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        evaluate(args.condition, args.seed)


if __name__ == '__main__':
    main()
