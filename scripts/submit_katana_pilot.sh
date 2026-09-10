#!/bin/bash
set -euo pipefail

setup_job="$(qsub scripts/katana_lunar_setup.pbs)"
train_job="$(qsub -W "depend=afterok:${setup_job}" scripts/katana_lunar_pilot_train.pbs)"

echo "Environment setup job: $setup_job"
echo "Pilot training array: $train_job (starts after setup succeeds)"
