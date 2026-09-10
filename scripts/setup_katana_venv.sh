#!/bin/bash
set -euo pipefail

# Run inside a Katana interactive compute job, not on a login node.
python_module="${PYTHON_MODULE:-python/3.10.8}"
venv_dir="${LUNAR_VENV_DIR:-/srv/scratch/$USER/environments/lunar-lander}"

module load "$python_module"
mkdir -p "$(dirname "$venv_dir")"
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -c 'import Box2D, gymnasium, torch; print("Box2D", Box2D.__version__); print("Gymnasium", gymnasium.__version__); print("PyTorch", torch.__version__)'

echo "Katana environment ready: $venv_dir"
