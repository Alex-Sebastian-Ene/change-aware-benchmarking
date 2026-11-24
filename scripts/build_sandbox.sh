#!/bin/bash
#SBATCH --job-name=build-sandbox
#SBATCH --partition=amd_student
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=build_sandbox-%j.log
set -euo pipefail
module load apptainer
export APPTAINER_TMPDIR=/scratch/$USER/apptainer/tmp

cd /scratch/$USER
SIF=vllm.sif
BOX=vllm_sandbox
[ -e "$BOX" ] && { echo "$BOX already exists; remove it first"; exit 1; }

apptainer build --sandbox "$BOX" "$SIF"
sha256sum "$SIF" | cut -d' ' -f1 > "$BOX/.source_sif_sha256"
chmod -R a-w "$BOX"
echo "built $BOX from $SIF ($(cat $BOX/.source_sif_sha256))"