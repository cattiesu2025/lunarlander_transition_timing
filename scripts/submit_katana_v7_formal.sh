#!/bin/bash
set -euo pipefail

# Submit the complete formal-v3 replication as one PBS dependency chain. Once
# qsub returns the six job IDs, the chain lives in the scheduler and does not
# depend on this login shell remaining connected.
command -v qsub >/dev/null 2>&1 || {
  echo "qsub is unavailable; run this launcher on a Katana login node." >&2
  exit 1
}

freeze_job="$(qsub scripts/katana_lunar_v7_formal_freeze.pbs)"
train_job="$(qsub -W "depend=afterok:${freeze_job}" scripts/katana_lunar_v7_formal_train.pbs)"
development_job="$(qsub -W "depend=afterok:${train_job}" scripts/katana_lunar_v7_formal_dev_eval.pbs)"
selection_job="$(qsub -W "depend=afterok:${development_job}" scripts/katana_lunar_v7_formal_select.pbs)"
held_out_job="$(qsub -W "depend=afterok:${selection_job}" scripts/katana_lunar_v7_formal_heldout_eval.pbs)"
aggregate_job="$(qsub -W "depend=afterok:${held_out_job}" scripts/katana_lunar_v7_formal_aggregate.pbs)"

cat <<EOF
Persistent-action ONSET replication submitted as one dependency chain.

freeze:      $freeze_job
train:       $train_job
development: $development_job
selection:   $selection_job
held-out:    $held_out_job
aggregate:   $aggregate_job

You may disconnect now. PBS retains the dependency chain. A failed stage keeps
every dependent stage from starting, including the sealed held-out evaluation.
Check the whole chain later with: qstat -u "$USER"
EOF
