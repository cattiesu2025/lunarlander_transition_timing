#!/usr/bin/env python3
"""Reselect all v7 models under the documented pre-held-out amendment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.lunar_lander_braking.config import (  # noqa: E402
    canonical_hash,
    file_hash,
    load_config,
    load_manifest,
)
from experiments.lunar_lander_braking.run import verify_frozen_inputs  # noqa: E402
from scripts.select_lunar_checkpoints import (  # noqa: E402
    CONDITIONS,
    select_models,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite amended selection: {output}")
    amendment = Path(args.amendment)
    if amendment.name != "amendment_v7_original_only.md" or not amendment.is_file():
        raise SystemExit("Expected the documented v7 original-only amendment")
    config = load_config(args.config)
    if config["experiment"]["name"] != "lunar_lander_braking_formal_v3_persistent_onset":
        raise SystemExit("Amendment applies only to the frozen v7 replication")
    verify_frozen_inputs(args.freeze_manifest, [args.config, args.manifest])
    manifest = load_manifest(args.manifest)

    audit, selected = select_models(
        config,
        manifest,
        Path(args.input),
        Path(args.train_root),
        gate_interventions=("original",),
    )
    expected = {
        (condition, int(seed))
        for condition in CONDITIONS
        for seed in config["experiment"]["seeds"]
    }
    actual = {(row["condition"], int(row["seed"])) for row in selected}
    if len(selected) != len(expected) or actual != expected:
        raise SystemExit(
            f"Amended selection still incomplete; missing: {sorted(expected - actual)}. "
            "Held-out remains sealed."
        )

    selection = config["selection"]
    write_outputs(
        output,
        audit,
        selected,
        int(selection["minimum_landed_per_intervention"]),
        int(selection["minimum_primary_events_per_intervention"]),
        canonical_hash(config),
        file_hash(args.manifest),
        file_hash(args.freeze_manifest),
        gate_interventions=("original",),
        amendment_sha256=file_hash(amendment),
    )
    print(f"Amended selection locked: {len(selected)}/{len(expected)} models")
    print(output / "selected_checkpoints.json")


if __name__ == "__main__":
    main()
