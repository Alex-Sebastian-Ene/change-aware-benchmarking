#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/eval_logs/%x-%j.log
set -euo pipefail

# Inputs, passed with sbatch --export: MODEL_REPO, MODEL_SHA, CONFIG (path relative to the
# repo), optional LIMIT (smoke runs) and OUT. Submit from anywhere; the job runs from $REPO.
: "${MODEL_REPO:?}" "${MODEL_SHA:?}" "${CONFIG:?}"
SCR=/scratch/$USER
OUT=${OUT:-$SCR/eval_runs}
REPO=${REPO:-$HOME/gdp/change-aware-benchmarking}
BOX=$SCR/vllm_sandbox
LIMIT_ARG=()
[ -n "${LIMIT:-}" ] && LIMIT_ARG=(--limit "$LIMIT")

module load apptainer

# The sandbox must come from the .sif the config pins.
want=$(awk '/apptainer_digest:/ {gsub(/"/, "", $2); print $2}' "$REPO/$CONFIG")
have=$(cat "$BOX/.source_sif_sha256")
[ "$want" = "$have" ] || { echo "sandbox is from $have, config pins $want"; exit 1; }

echo "job $SLURM_JOB_ID on $(hostname): $MODEL_REPO@$MODEL_SHA, $CONFIG, limit=${LIMIT:-none}"
cd "$REPO"
apptainer exec --nv --cleanenv --bind /scratch \
  --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$SCR/git-env/bin \
  --env PYTHONPATH=$REPO/src \
  --env HF_HOME=$SCR/hf_cache --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
  --env XDG_CACHE_HOME=$SCR/.cache \
  --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  --env SLURM_JOB_ID=$SLURM_JOB_ID \
  --env USER=$USER \
  "$BOX" "$SCR/ctr-venv/bin/python" scripts/run_eval.py \
    --config "$CONFIG" --model "$MODEL_REPO" --sha "$MODEL_SHA" --out "$OUT" "${LIMIT_ARG[@]}"
