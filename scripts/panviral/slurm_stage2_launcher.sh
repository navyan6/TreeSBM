#!/bin/bash
#SBATCH --job-name=pv_launch
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=logs/panviral/launch_%j.log

# Submitted with a dependency on the stage-1 inventory job, so stage 2 fans out
# on its own the moment the counts land. An array's size has to be known at
# submission time and we do not know how many viruses will qualify until stage 1
# finishes -- hence a launcher rather than a pre-sized array.

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
MIN_COUNT=${MIN_COUNT:-150}
KEYFILE=${NCBI_API_KEY_FILE:-$HOME/.ncbi_api_key}
CHAIN_STAGE3=${CHAIN_STAGE3:-1}

# NCBI rate-limits per source IP and every array task shares that one budget,
# so concurrency and the per-task sleep are chosen together: aggregate request
# rate is roughly CONCURRENCY / SLEEP, and it has to stay under the limit.
# Unauthenticated that limit is 3 req/s; with a key it is 10 req/s.
#
# The key is only ever tested for existence -- never read, printed, or passed
# on a command line. Adding the file later is enough; nothing else changes.
if [ -s "$KEYFILE" ]; then
    CONCURRENCY=${CONCURRENCY:-8}
    SLEEP=${SLEEP:-1.0}          # ~8 req/s, under the authenticated limit of 10
    echo "NCBI API key present -> concurrency ${CONCURRENCY}, sleep ${SLEEP}"
else
    CONCURRENCY=${CONCURRENCY:-3}
    SLEEP=${SLEEP:-1.2}          # ~2.5 req/s, under the unauthenticated limit of 3
    echo "no API key -> concurrency ${CONCURRENCY}, sleep ${SLEEP}"
fi

$PY scripts/panviral/make_worklist.py --min-count "$MIN_COUNT" --skip-existing

N=$(wc -l < data/panviral/worklist.tsv)
if [ "$N" -eq 0 ]; then
    echo "worklist empty; nothing to submit"
    exit 0
fi

echo "submitting array 1-${N}%${CONCURRENCY}"
jid=$(sbatch --parsable --array="1-${N}%${CONCURRENCY}" \
      --export=ALL,SLEEP="$SLEEP",TREESBM_ROOT="$REPO",TREESBM_PY="$PY" \
      scripts/panviral/slurm_fetch_array.sh)
echo "stage 2 array job: $jid"
echo "$jid" > data/panviral/stage2_array_jobid.txt

# Fan out stage 3 only after every fetch task has finished. Use afterany (not
# afterok) so one failed array element does not block the whole pipeline;
# stage 3 only consumes viruses that wrote manifests.
# CHAIN_STAGE3=0 to stop after the CDS pull.
if [ "$CHAIN_STAGE3" = "1" ]; then
  s3=$(sbatch --parsable --dependency=afterany:"$jid" \
          --export=ALL,TREESBM_ROOT="$REPO",TREESBM_PY="$PY" \
          scripts/panviral/slurm_stage3_launcher.sh)
    echo "stage 3 launcher: $s3 (waits on array $jid)"
    echo "$s3" > data/panviral/stage3_launcher_jobid.txt
fi
