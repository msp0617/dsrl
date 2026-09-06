#!/usr/bin/env bash
# Copy the project folder between Google Drive and the VM disk with rclone.
#
#   bash gce/sync_drive.sh pull          # Drive -> VM: env_cache, dppo_log, offline, logs/pretrain, base_policy_eval*.csv
#   bash gce/sync_drive.sh pull-logs     # Drive -> VM: every logs/<exp>/ (CSV, .out, checkpoints) for resuming old runs
#   bash gce/sync_drive.sh push          # VM -> Drive: logs/ without checkpoints (CSV + .out), so local analysis works as before
#   bash gce/sync_drive.sh push-all      # VM -> Drive: logs/ including checkpoints
#
# One-time: `rclone config` -> n (new remote) -> name: gdrive -> storage: drive ->
# leave client id/secret empty -> scope 1 (full access) -> "Use auto config?" n ->
# run the printed `rclone authorize "drive"` on the Windows machine
# (winget install Rclone.Rclone), paste the token back. Test: rclone lsd gdrive:dsrl_project
set -euo pipefail

DATA=${DATA:-/data/dsrl_project}
REMOTE=${REMOTE:-gdrive:dsrl_project}
FLAGS=(--progress --transfers 8 --checkers 16 --drive-chunk-size 64M)

case "${1:-}" in
  pull)
    for d in env_cache dppo_log offline logs/pretrain; do
      echo "=== pull $d"; rclone copy "$REMOTE/$d" "$DATA/$d" "${FLAGS[@]}"
    done
    rclone copy "$REMOTE/logs" "$DATA/logs" --include "base_policy_eval*.csv" "${FLAGS[@]}"
    ;;
  pull-logs)
    echo "=== pull logs (with checkpoints)"; rclone copy "$REMOTE/logs" "$DATA/logs" "${FLAGS[@]}"
    ;;
  push)
    echo "=== push logs (no checkpoints)"
    rclone copy "$DATA/logs" "$REMOTE/logs" --exclude "*/checkpoint/**" "${FLAGS[@]}"
    ;;
  push-all)
    echo "=== push logs (with checkpoints)"; rclone copy "$DATA/logs" "$REMOTE/logs" "${FLAGS[@]}"
    ;;
  *)
    sed -n 2,14p "$0"; exit 1 ;;
esac
echo "done"
