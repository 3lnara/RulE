#!/usr/bin/env bash
# setup_hpc_dirs.sh — create RulE project directory structure on TU/e HPC
# Usage: ./setup_hpc_dirs.sh [-r REMOTE_DIR]
#   -r REMOTE_DIR  remote base directory (default: ~/RulE)

set -euo pipefail

HPC_USER="20252643"
HPC_HOST="hpc.tue.nl"
REMOTE_DIR="~/RulE"

while getopts ":r:" opt; do
  case $opt in
    r) REMOTE_DIR="$OPTARG" ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
  esac
done

echo "Creating directory structure at ${HPC_USER}@${HPC_HOST}:${REMOTE_DIR} ..."

ssh "${HPC_USER}@${HPC_HOST}" bash <<EOF
set -e
mkdir -p ${REMOTE_DIR}/{src_additive}
echo "Directories created:"
ls -la ${REMOTE_DIR}/
EOF

echo "Done."
