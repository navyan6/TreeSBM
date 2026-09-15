#!/bin/bash
#SBATCH --job-name=pv_inventory
#SBATCH --partition=genoa-std-mem
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=logs/panviral/inventory_%j.log

# Stage 1: count NCBI genomes for every eukaryotic virus species.
# Network-bound, not CPU-bound: ~53k taxa at the unauthenticated NCBI rate of
# 2.5 req/s is roughly 6 hours. Set NCBI_API_KEY to run at 9 req/s (~1.6 h).
# The counts cache makes this resumable, so a requeue costs nothing.

set -euo pipefail
REPO="${TREESBM_ROOT:-$HOME/DiscreteTreeFlows}"
cd "$REPO"
mkdir -p logs/panviral data/panviral

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

echo "host=$(hostname)  start=$(date -Is)  repo=$REPO"
$PY scripts/panviral/build_virus_inventory.py \
    --min-count "${MIN_COUNT:-150}" \
    --exclude-covid-flu "${EXCLUDE_COVID_FLU:-}" \
    --out data/panviral/virus_inventory.json
echo "done=$(date -Is)"
