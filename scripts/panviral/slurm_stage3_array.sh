#!/bin/bash
#SBATCH --job-name=pv_trees
#SBATCH --partition=genoa-std-mem
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=logs/panviral/trees_%A_%a.log

# Stage 3 fan-out: one array task per (virus, split).
# Runs align / FastTree / augur refine / ancestral / translate on the groups
# stage 2 wrote. Reference-anchored CDS slices use the aligned translator with
# cds_start=0 so the ATG search cannot walk off the start of the window.

set -euo pipefail
REPO="${TREESBM_ROOT:-$HOME/DiscreteTreeFlows}"
cd "$REPO"

WORKLIST=${WORKLIST:-data/panviral/stage3_worklist.tsv}
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
WORKERS=${WORKERS:-4}

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")
if [ -z "${line:-}" ]; then
    echo "no work at index ${SLURM_ARRAY_TASK_ID}"
    exit 0
fi

slug=$(echo "$line" | cut -f1)
split=$(echo "$line" | cut -f2)
ng=$(echo "$line" | cut -f3)
data_dir=$(echo "$line" | cut -f4)

echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID}: $slug/$split ($ng groups)"

# New panviral CDS windows open on the reference start codon (see
# data/TRANSLATION_MODES.md). Do not inherit flu/HIV "ungapped" mode here.
export TREESBM_TRANSLATE_MODE=aligned
export TREESBM_CDS_START=0
export TREESBM_CLOCK_RATE="${TREESBM_CLOCK_RATE:-0.001}"

$PY scripts/run_all_groups.py \
    --data-dir "$data_dir" \
    --prefix "$slug" \
    --workers "$WORKERS" \
    --stop-after translate

echo "[$(date -Is)] done $slug/$split"
