#!/bin/bash
set -euo pipefail

command -v qsub >/dev/null 2>&1 || {
  echo "qsub is unavailable; run this launcher on a Katana login node." >&2
  exit 1
}

root="outputs/lunar_lander_braking_formal_v3_persistent_onset"
for required in "$root/frozen/manifest.sha256.json" "$root/development_eval" "$root/train" "$root/selection/model_selection.csv"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing original v7 artifact: $required" >&2
    exit 1
  fi
done
for forbidden in "$root/held_out_eval" "$root/selection_amended_original_only" "$root/held_out_eval_amended_original_only" "$root/aggregate_amended_original_only"; do
  if [[ -e "$forbidden" ]]; then
    echo "Refusing to submit: output already exists: $forbidden" >&2
    exit 1
  fi
done

selection_job="$(qsub scripts/katana_lunar_v7_amended_select.pbs)"
held_out_job="$(qsub -W "depend=afterok:${selection_job}" scripts/katana_lunar_v7_amended_heldout.pbs)"
aggregate_job="$(qsub -W "depend=afterok:${held_out_job}" scripts/katana_lunar_v7_amended_aggregate.pbs)"

cat <<EOF
Post-freeze v7 amendment submitted; the original failed selection is preserved.
selection: $selection_job
held-out:  $held_out_job
aggregate: $aggregate_job

PBS retains this dependency chain after logout. Held-out starts only if the
amended selector locks all 60 models under the same original-only gate.
Check later with: qstat -u "$USER"
EOF
