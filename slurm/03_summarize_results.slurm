#!/usr/bin/env bash
#SBATCH --job-name=rule_summary
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/RulE}"
EXPERIMENT_KEY="${EXPERIMENT_KEY:-wn18rr_apples}"

BASE_OUT="${PROJECT_DIR}/outputs/experiments/${EXPERIMENT_KEY}"
SUMMARY_JSON="${BASE_OUT}/summaries/unified_summary.json"

module purge
module load Python/3.10.4-GCCcore-11.3.0
source "${PROJECT_DIR}/rule_env/bin/activate"

python3 -u "${PROJECT_DIR}/scripts/summarize_unified_results.py" \
  --experiment_dir "${BASE_OUT}" \
  --output_json "${SUMMARY_JSON}"

echo "Summary written to ${SUMMARY_JSON}"
