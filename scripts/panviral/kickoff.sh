#!/usr/bin/env bash
# One entrypoint for the pan-viral data pipeline on Betty (or any SLURM host).
#
#   bash scripts/panviral/kickoff.sh
#
# Submits stage 1 (inventory) → stage 2 (fetch + CDS extract) → stage 3
# (mafft / FastTree / augur / translate). Later stages wait on earlier ones via
# SLURM dependencies, so this is fire-and-forget.
#
# Useful knobs (all optional):
#   TREESBM_ROOT=/path/to/repo
#   TREESBM_PY=/path/to/python          # must see biopython + mafft/FastTree/augur on PATH
#   MIN_COUNT=150
#   CHAIN_STAGE3=1                      # set 0 to stop after the CDS pull
#   NCBI_API_KEY_FILE=~/.ncbi_api_key
#
# Usage:
#   bash scripts/panviral/kickoff.sh [env-name] [exclude-covid-flu] [path-to-repo-root]
#   bash scripts/panviral/kickoff.sh
#
# Resume is free: inventory caches counts, stage 2 skips viruses with a
# manifest, stage 3 skips splits that already have rooted trees.
set -euo pipefail

ENV_NAME="${1:-${TREESBM_CONDA_ENV:-treesbm-data}}"
EXCLUDE_COVID_FLU="0"
if [ "${2:-}" = "exclude-covid-flu" ]; then
    EXCLUDE_COVID_FLU="1"
fi
REPO="${3:-${TREESBM_ROOT:-$HOME/DiscreteTreeFlows}}"
export TREESBM_CONDA_ENV="$ENV_NAME"
export TREESBM_ROOT="$REPO"
cd "$REPO"
mkdir -p logs/panviral data/panviral

if ! command -v sbatch >/dev/null 2>&1; then
    export PATH="/vast/parcc/sw/slurm/bin:${PATH}"
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found; run this on a SLURM login node" >&2
    exit 1
fi

resolve_py() {
    if [ -n "${TREESBM_PY:-}" ]; then
        echo "$TREESBM_PY"
        return
    fi
    env_name="${TREESBM_CONDA_ENV:-treesbm-data}"
    for cand in \
        "${HOME}/.conda/envs/${env_name}/bin/python" \
        "${HOME}/.conda/envs/treesbm-data/bin/python" \
        "${HOME}/.conda/envs/treesbm/bin/python" \
        "$(command -v python3 || true)" \
        "$(command -v python || true)"
    do
        if [ -n "$cand" ] && [ -x "$cand" ]; then
            echo "$cand"
            return
        fi
    done
    echo "No python found. Create the env first:" >&2
    echo "  conda env create -f scripts/panviral/environment.yml" >&2
    echo "  conda activate ${env_name}" >&2
    echo "  TREESBM_PY=\$(which python) bash scripts/panviral/kickoff.sh ${env_name}" >&2
    exit 1
}

export TREESBM_PY="$(resolve_py)"
export PATH="$(dirname "$TREESBM_PY"):${PATH}"
export MIN_COUNT="${MIN_COUNT:-150}"
export CHAIN_STAGE3="${CHAIN_STAGE3:-1}"

bash scripts/panviral/check_deps.sh

echo "repo=$REPO"
echo "python=$TREESBM_PY"
echo "min_count=$MIN_COUNT  chain_stage3=$CHAIN_STAGE3  exclude_covid_flu=$EXCLUDE_COVID_FLU"

inv=$(sbatch --parsable --export=ALL,EXCLUDE_COVID_FLU="$EXCLUDE_COVID_FLU" \
      scripts/panviral/slurm_inventory.sh)
echo "stage 1 inventory: $inv"
echo "$inv" > data/panviral/stage1_jobid.txt

s2=$(sbatch --parsable --dependency=afterok:"$inv" --export=ALL \
      scripts/panviral/slurm_stage2_launcher.sh)
echo "stage 2 launcher:  $s2  (waits on $inv)"
echo "$s2" > data/panviral/stage2_launcher_jobid.txt

echo
echo "queued. track with:  squeue -u \$USER"
echo "logs under:          logs/panviral/"
if [ "$CHAIN_STAGE3" = "1" ]; then
    echo "stage 3 submits itself once the stage-2 fetch array finishes."
fi
