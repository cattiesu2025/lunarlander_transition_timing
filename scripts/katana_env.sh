#!/bin/bash

# Source this file from a Katana PBS job after changing to PBS_O_WORKDIR.
python_module="${PYTHON_MODULE:-python/3.11.3}"
required_python="${LUNAR_PYTHON_VERSION:-3.11}"
venv_dir="${LUNAR_VENV_DIR:-/srv/scratch/$USER/environments/lunar-lander-py311}"

module load "$python_module"
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  echo "Missing $venv_dir/bin/activate" >&2
  echo "Submit scripts/katana_lunar_setup.pbs before this job." >&2
  exit 1
fi
source "$venv_dir/bin/activate"
python -c '
import os
from pathlib import Path
import sys

expected = tuple(map(int, sys.argv[1].split(".")))
actual = sys.version_info[:2]
if actual != expected:
    raise SystemExit(
        f"Wrong Python after venv activation: expected {sys.argv[1]}, "
        f"got {actual[0]}.{actual[1]} at {sys.executable}"
    )
venv = Path(os.environ["VIRTUAL_ENV"]).resolve()
prefix = Path(sys.prefix).resolve()
if prefix != venv:
    raise SystemExit(f"Wrong active venv: expected {venv}, got {prefix}")
print(f"Active Katana environment: Python {sys.version.split()[0]} at {sys.executable}")
' "$required_python"

export PYTHONPATH="${PBS_O_WORKDIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
job_tmp="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="$job_tmp/lunar-lander-matplotlib"
export XDG_CACHE_HOME="$job_tmp/lunar-lander-cache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
