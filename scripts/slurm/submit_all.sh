#!/bin/bash
# Submit eval.sbatch once per model in a models CSV (header repo,sha), all with one config.
#
#   scripts/slurm/submit_all.sh configs/harness/<config>.yaml [configs/models.csv] [extra sbatch --export vars]
#
# e.g. a smoke run of every model:
#   scripts/slurm/submit_all.sh configs/harness/<config>.yaml configs/models.csv LIMIT=20,OUT=/scratch/$USER/eval_smoke
set -euo pipefail

CONFIG=${1:?usage: submit_all.sh CONFIG [MODELS_CSV] [EXTRA_EXPORTS]}
MODELS=${2:-configs/models.csv}
EXTRA=${3:+,$3}
cd "$(dirname "$0")/../.."  # repo root, so CONFIG and MODELS are repo-relative

[ -f "$CONFIG" ] || { echo "no config $CONFIG"; exit 1; }
[ "$(head -n1 "$MODELS" | tr -d '\r')" = "repo,sha" ] || { echo "$MODELS must start with the header repo,sha"; exit 1; }

tail -n +2 "$MODELS" | tr -d '\r' | while IFS=, read -r repo sha || [ -n "$repo" ]; do  # last line may lack a newline
  [ -z "$repo" ] && continue  # blank line
  echo -n "$repo@${sha:0:12}: "
  sbatch --export=ALL,MODEL_REPO=$repo,MODEL_SHA=$sha,CONFIG=$CONFIG$EXTRA scripts/slurm/eval.sbatch
done