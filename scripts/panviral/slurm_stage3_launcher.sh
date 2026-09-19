#!/bin/bash
#SBATCH --job-name=pv_s3launch
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=logs/panviral/stage3_launch_%j.log

# Submitted with afterok on the stage-2 fetch array. Builds the stage-3
# worklist from manifests that landed, then fans out the tree-building array.

set -euo pipefail
REPO="${TREESBM_ROOT:-$HOME/DiscreteTreeFlows}"
cd "$REPO"
mkdir -p logs/panviral

PY=${TREESBM_PY:-}
if [ -z "$PY" ]; then
  for cand in "${HOME}/.conda/envs/treesbm-data/bin/python" "${HOME}/.conda/envs/treesbm/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then PY="$cand"; break; fi
  done
fi
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
  echo "set TREESBM_PY to a python that has biopython + mafft/FastTree/augur on PATH" >&2
  exit 1
fi
export PATH="$(dirname "$PY"):${PATH}"
CONCURRENCY=${STAGE3_CONCURRENCY:-20}

$PY scripts/panviral/make_stage3_worklist.py

N=$(wc -l < data/panviral/stage3_worklist.tsv | tr -d ' ')
if [ "${N:-0}" -eq 0 ]; then
    echo "stage3 worklist empty; nothing to submit"
    exit 0
fi

echo "submitting stage-3 array 1-${N}%${CONCURRENCY}"
jid=$(sbatch --parsable --array="1-${N}%${CONCURRENCY}" \
      --export=ALL,TREESBM_ROOT="$REPO",TREESBM_PY="$PY",TREESBM_CLOCK_RATE="${TREESBM_CLOCK_RATE:-0.001}",WORKERS="${WORKERS:-4}" \
      scripts/panviral/slurm_stage3_array.sh)
echo "stage 3 array job: $jid"
echo "$jid" > data/panviral/stage3_array_jobid.txt
