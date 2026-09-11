import json
from pathlib import Path

import pytest

from experiments.lunar_lander_braking.config import canonical_hash, file_hash, load_config
from scripts.resolve_lunar_checkpoint import resolve_checkpoint
from scripts.select_lunar_checkpoints import write_outputs


CONFIG = Path(
    "experiments/lunar_lander_braking/configs/formal_v6_high_latest.yaml"
)
DEVELOPMENT = Path(
    "experiments/lunar_lander_braking/grids/development_high.json"
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_selection_outputs_hash_locked_schema(tmp_path):
    output = tmp_path / "selection"
    write_outputs(
        output,
        [{"condition": "DESC", "seed": 1001, "eligible": True}],
        [],
        15,
        15,
        "config-hash",
        "development-hash",
        "freeze-hash",
    )
    selected = output / "selected_checkpoints.json"
    payload = json.loads(selected.read_text(encoding="utf-8"))
    lock = json.loads(
        (output / "selected_checkpoints.sha256.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 2
    assert payload["config_hash"] == "config-hash"
    assert lock == {
        "schema_version": 1,
        "file": selected.name,
        "sha256": file_hash(selected),
    }


def test_resolver_requires_locked_complete_selection_and_matching_checkpoint(tmp_path):
    config = load_config(CONFIG)
    freeze = tmp_path / "manifest.sha256.json"
    write_json(
        freeze,
        {
            "schema_version": 1,
            "files": {
                CONFIG.name: file_hash(CONFIG),
                DEVELOPMENT.name: file_hash(DEVELOPMENT),
            },
        },
    )
    checkpoint = tmp_path / "train" / "DESC" / "seed_1001" / "model.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixed selected model")
    checkpoint_hash = file_hash(checkpoint)
    write_json(
        checkpoint.with_name("metadata.json"),
        {
            "checkpoint_sha256": checkpoint_hash,
            "config_hash": canonical_hash(config),
            "condition": "DESC",
            "seed": 1001,
            "training_steps": 800_000,
        },
    )
    selected_models = []
    for condition in ("DESC", "BAL", "ECON"):
        for seed in config["experiment"]["seeds"]:
            selected_models.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "checkpoint_step": 800_000,
                    "checkpoint_path": str(checkpoint) if (condition, seed) == ("DESC", 1001) else "unused",
                    "checkpoint_sha256": checkpoint_hash if (condition, seed) == ("DESC", 1001) else "unused",
                }
            )
    selection = tmp_path / "selected_checkpoints.json"
    write_json(
        selection,
        {
            "schema_version": 2,
            "selection_rule": "latest_eligible_checkpoint",
            "config_hash": canonical_hash(config),
            "development_manifest_sha256": file_hash(DEVELOPMENT),
            "freeze_manifest_sha256": file_hash(freeze),
            "gate": {
                "interventions": ["original", "low_descent_speed"],
                "minimum_landed_per_intervention": 15,
                "minimum_primary_events_per_intervention": 15,
                "uses_onset_time": False,
            },
            "selected_models": selected_models,
        },
    )
    lock = tmp_path / "selected_checkpoints.sha256.json"
    write_json(
        lock,
        {"schema_version": 1, "file": selection.name, "sha256": file_hash(selection)},
    )

    assert resolve_checkpoint(
        CONFIG, selection, lock, freeze, "DESC", 1001
    ) == checkpoint
    selection.write_text(selection.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        resolve_checkpoint(CONFIG, selection, lock, freeze, "DESC", 1001)


def test_formal_pbs_task_counts_and_sealed_ordering_contract():
    train = Path("scripts/katana_lunar_v6_formal_train.pbs").read_text()
    development = Path("scripts/katana_lunar_v6_formal_dev_eval.pbs").read_text()
    held_out = Path("scripts/katana_lunar_v6_formal_heldout_eval.pbs").read_text()
    assert "#PBS -J 0-59" in train
    assert "#PBS -J 0-299" in development
    assert "#PBS -J 0-59" in held_out
    assert "development_high.json" in development
    assert "resolve_lunar_checkpoint.py" in held_out
    assert "held_out_high.json" in held_out
    assert "Refusing to overwrite sealed held-out output" in held_out


def test_every_compute_pbs_activates_guarded_python311_environment():
    setup = Path("scripts/katana_lunar_setup.pbs")
    compute_jobs = [
        path
        for path in Path("scripts").glob("katana_lunar_*.pbs")
        if path != setup
    ]
    assert compute_jobs
    for path in compute_jobs:
        assert "source scripts/katana_env.sh" in path.read_text(), path

    helper = Path("scripts/katana_env.sh").read_text()
    setup_helper = Path("scripts/setup_katana_venv.sh").read_text()
    setup_job = setup.read_text()
    for text in (helper, setup_helper, setup_job):
        assert "lunar-lander-py311" in text
        assert "3.11" in text
    assert "python -m venv" in setup_helper
    assert "python3 -m venv" not in setup_helper
    assert "Wrong Python after venv activation" in helper
    assert 'os.environ["VIRTUAL_ENV"]' in helper
