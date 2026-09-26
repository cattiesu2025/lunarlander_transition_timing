#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/lunar_v7_angle_grid_extension.py prepare
evaluation_job=$(qsub scripts/katana_lunar_v7_angle_grid_eval.pbs)
aggregate_job=$(qsub -W "depend=afterok:${evaluation_job}" scripts/katana_lunar_v7_angle_grid_aggregate.pbs)

echo "v7 nine-angle extension submitted"
echo "evaluation: $evaluation_job"
echo "aggregate:  $aggregate_job"
