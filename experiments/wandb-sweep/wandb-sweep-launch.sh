#!/bin/bash
set -euo pipefail

# ./scripts/launch_sweep.sh sweep.yaml my-first-sweep 400 20
# sweep.yaml, project name, total runs, max parallel agents

SWEEP_YAML="${1:-sweep.yaml}"
PROJECT="${2:-my-first-sweep}"
TOTAL_RUNS="${3:-100}"
MAX_PARALLEL="${4:-10}"

if ! command -v wandb >/dev/null 2>&1; then
  echo "ERROR: wandb CLI not found in PATH...?"
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH. You'll need to run that in the cluster!"
  exit 1
fi

# You really should set the WANDB key :)
if [[ -z "${WANDB_KEY:-}" ]]; then
  echo "WARNING: WANDB_KEY is not set. If you're not already logged in, set it:"
  echo '  export WANDB_KEY="..."'
fi

echo "Creating sweep from: $SWEEP_YAML (project=$PROJECT)"
OUT="$(wandb sweep --project "$PROJECT" "$SWEEP_YAML" 2>&1 | tee /dev/stderr)"

# We then parse the sweep id from output lines, e.g.:
# "Create sweep with ID: entity/project/somethingsomething"
SWEEP_ID="$(echo "$OUT" | sed -n 's/.*Create sweep with ID:[[:space:]]*\([^[:space:]]*\).*/\1/p' | tail -n 1)"

#if [[ -z "$SWEEP_ID" ]]; then
  # if output is different?
#  SWEEP_ID="$(echo "$OUT" | grep -Eo '[^[:space:]]+/[^[:space:]]+/[^[:space:]]+' | tail -n 1)"
#fi

if [[ -z "$SWEEP_ID" ]]; then
  echo "ERROR: Could not parse SWEEP_ID from wandb output."
  echo "Raw output:"
  echo "$OUT"
  exit 1
fi

echo "Sweep ID: $SWEEP_ID"
echo "Submitting Slurm array: total=$TOTAL_RUNS, max_parallel=$MAX_PARALLEL"

ARRAY_SPEC="0-$((TOTAL_RUNS-1))%${MAX_PARALLEL}"
sbatch --array="$ARRAY_SPEC" experiments/wandb-sweep/wandb_agent.sbatch "$SWEEP_ID"

echo "Done."
