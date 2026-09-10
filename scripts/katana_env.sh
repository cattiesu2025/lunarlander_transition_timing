#!/bin/bash

# Source this file from a Katana PBS job after changing to PBS_O_WORKDIR.
python_module="${PYTHON_MODULE:-python/3.11.3}"
venv_dir="${LUNAR_VENV_DIR:-/srv/scratch/$USER/environments/lunar-lander}"

module load "$python_module"
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  echo "Missing $venv_dir/bin/activate" >&2
  echo "Submit scripts/katana_lunar_setup.pbs before this job." >&2
  exit 1
fi
source "$venv_dir/bin/activate"

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
