#!/bin/bash
set -euo pipefail

# Run on a Katana compute node, normally through katana_lunar_setup.pbs.
python_module="${PYTHON_MODULE:-python/3.11.3}"
required_python="${LUNAR_PYTHON_VERSION:-3.11}"
venv_dir="${LUNAR_VENV_DIR:-/srv/scratch/$USER/environments/lunar-lander-py311}"

module load "$python_module"
python -c '
import sys

expected = tuple(map(int, sys.argv[1].split(".")))
actual = sys.version_info[:2]
if actual != expected:
    raise SystemExit(
        f"Loaded module has wrong Python: expected {sys.argv[1]}, "
        f"got {actual[0]}.{actual[1]} at {sys.executable}"
    )
' "$required_python"
mkdir -p "$(dirname "$venv_dir")"
if [[ -x "$venv_dir/bin/python" ]]; then
  "$venv_dir/bin/python" -c '
import sys

expected = tuple(map(int, sys.argv[1].split(".")))
actual = sys.version_info[:2]
if actual != expected:
    raise SystemExit(
        f"Existing venv has wrong Python: expected {sys.argv[1]}, "
        f"got {actual[0]}.{actual[1]} at {sys.executable}; use a new LUNAR_VENV_DIR"
    )
' "$required_python"
fi
python -m venv "$venv_dir"
source "$venv_dir/bin/activate"
python -c '
import os
from pathlib import Path
import sys

expected = tuple(map(int, sys.argv[1].split(".")))
actual = sys.version_info[:2]
if actual != expected:
    raise SystemExit(
        f"Created venv has wrong Python: expected {sys.argv[1]}, "
        f"got {actual[0]}.{actual[1]} at {sys.executable}"
    )
if Path(sys.prefix).resolve() != Path(os.environ["VIRTUAL_ENV"]).resolve():
    raise SystemExit("Created venv is not the active Python prefix")
' "$required_python"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -c 'import Box2D, gymnasium, stable_baselines3, torch; print("Box2D", Box2D.__version__); print("Gymnasium", gymnasium.__version__); print("Stable-Baselines3", stable_baselines3.__version__); print("PyTorch", torch.__version__)'

echo "Katana Python $required_python environment ready: $venv_dir"
