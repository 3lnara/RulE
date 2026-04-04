#!/usr/bin/env bash
# sync_to_hpc.sh — transfer RulE project files to TU/e HPC
# Usage: ./sync_to_hpc.sh [-n] [-r REMOTE_DIR]
#   -n            dry-run (preview only, nothing transferred)
#   -r REMOTE_DIR remote base directory (default: ~/RulE)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
HPC_USER="20252643"
HPC_HOST="hpc.tue.nl"
REMOTE_DIR="~/RulE"
LOG_FILE="$(dirname "$0")/sync_to_hpc.log"
DRY_RUN=false

# ── Argument parsing ───────────────────────────────────────────────────────────
while getopts ":nr:" opt; do
  case $opt in
    n) DRY_RUN=true ;;
    r) REMOTE_DIR="$OPTARG" ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# ── Resolve project root ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── SSH ControlMaster — one password prompt for the whole script ───────────────
# A shared socket is opened now; all ssh/rsync calls reuse it automatically.
SSH_SOCKET="/tmp/hpc_sync_$$"
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$SSH_SOCKET" -o ControlPersist=10m)

cleanup() {
  ssh "${SSH_OPTS[@]}" -O exit "${HPC_USER}@${HPC_HOST}" 2>/dev/null || true
  rm -f "$SSH_SOCKET"
}
trap cleanup EXIT

if ! $DRY_RUN; then
  echo "Opening SSH connection to ${HPC_HOST} (one password prompt for all transfers)..."
  ssh "${SSH_OPTS[@]}" -fN "${HPC_USER}@${HPC_HOST}"
fi

# ── Build base rsync flags ─────────────────────────────────────────────────────
RSYNC_FLAGS=(
  --archive        # preserve permissions, timestamps, symlinks, etc.
  --compress       # compress during transfer
  --progress       # per-file progress
  --stats          # summary stats at the end
  --human-readable # human-readable sizes
  -e "ssh ${SSH_OPTS[*]}"   # reuse the ControlMaster socket
)

if $DRY_RUN; then
  RSYNC_FLAGS+=(--dry-run)
  echo "[DRY-RUN] No files will be transferred."
fi

DEST_BASE="${HPC_USER}@${HPC_HOST}:${REMOTE_DIR}"

# ── Log header ─────────────────────────────────────────────────────────────────
{
  echo "========================================"
  echo "Sync started : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Destination  : ${DEST_BASE}"
  $DRY_RUN && echo "Mode         : DRY-RUN" || echo "Mode         : LIVE"
  echo "========================================"
} | tee -a "$LOG_FILE"

# ── Helper: ensure a remote directory exists (no-op in dry-run) ───────────────
ensure_remote_dir() {
  local remote_path="$1"
  if $DRY_RUN; then
    echo "[DRY-RUN] Would ssh mkdir -p ${remote_path}" | tee -a "$LOG_FILE"
    return 0
  fi
  ssh "${SSH_OPTS[@]}" "${HPC_USER}@${HPC_HOST}" "mkdir -p ${remote_path}" 2>&1 | tee -a "$LOG_FILE"
}

# ── Helper: run one rsync group and log it ────────────────────────────────────
run_rsync() {
  local dest="$1"; shift
  local sources=("$@")

  echo "" | tee -a "$LOG_FILE"
  echo "--- Syncing to ${dest} ---" | tee -a "$LOG_FILE"

  rsync "${RSYNC_FLAGS[@]}" \
    --log-file="$LOG_FILE" \
    "${sources[@]}" \
    "$dest" \
    2>&1 | tee -a "$LOG_FILE"

  return "${PIPESTATUS[0]}"
}

# ── Transfer groups ────────────────────────────────────────────────────────────
OVERALL_EXIT=0

# 1. src/ → remote src/
run_rsync "${DEST_BASE}/src/" \
  "${SCRIPT_DIR}/src/data.py" \
  "${SCRIPT_DIR}/src/layers.py" \
  "${SCRIPT_DIR}/src/main.py" \
  "${SCRIPT_DIR}/src/model.py" \
  "${SCRIPT_DIR}/src/trainer.py" \
  "${SCRIPT_DIR}/src/utils.py" \
  "${SCRIPT_DIR}/src/run_comparison.py" \
  || OVERALL_EXIT=$?

# 1b. RulE_original/src/ → remote RulE_original/src/
ensure_remote_dir "${REMOTE_DIR}/RulE_original/src" || OVERALL_EXIT=$?
run_rsync "${DEST_BASE}/RulE_original/src/" \
  "${SCRIPT_DIR}/RulE_original/src/" \
  || OVERALL_EXIT=$?

# 2. data/family/ → remote data/family/
run_rsync "${DEST_BASE}/data/family/" \
  "${SCRIPT_DIR}/data/family/" \
  || OVERALL_EXIT=$?

# 3. config files → remote config/
run_rsync "${DEST_BASE}/config/" \
  "${SCRIPT_DIR}/config/family_config.json" \
  "${SCRIPT_DIR}/config/family_comparison_hpc.json" \
  || OVERALL_EXIT=$?

# 4. SLURM job script + setup script → remote ~/
run_rsync "${HPC_USER}@${HPC_HOST}:~/" \
  "${SCRIPT_DIR}/rule_comparison_gpu.slurm" \
  "${SCRIPT_DIR}/setup_env_hpc.sh" \
  || OVERALL_EXIT=$?

# ── Log footer ─────────────────────────────────────────────────────────────────
{
  echo ""
  echo "========================================"
  if [ "$OVERALL_EXIT" -eq 0 ]; then
    $DRY_RUN \
      && echo "Result       : Dry-run completed successfully" \
      || echo "Result       : Transfer completed successfully"
  else
    echo "Result       : FAILED (one or more groups had errors)"
  fi
  echo "Sync ended   : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================"
  echo ""
} | tee -a "$LOG_FILE"

exit "$OVERALL_EXIT"
