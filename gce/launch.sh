#!/usr/bin/env bash
# Launch (or resume) a batch of runs from a runs file, the way the Colab launch
# cells did, then optionally wait for them, push the CSVs to Drive and stop the VM.
#
#   bash gce/launch.sh RUNS_FILE            # start every run in the file that is not [done]
#   bash gce/launch.sh RUNS_FILE --wait     # ... then wait, `sync_drive.sh push`, and `sudo shutdown -h now`
#   bash gce/launch.sh --status             # what is running, last eval of every run in logs/
#
# RUNS_FILE: one run per line, `#` comments allowed:
#   <exp_id>  <can|square>  <hydra overrides...>
# e.g.
#   can_tent6_s1   can     seed=1 train.target_ent=6 train.total_env_steps=100000
#   square_tent6_s1 square seed=1 train.target_ent=6 train.total_env_steps=150000
#
# Defaults added to every run (override in the file if needed):
#   log_dir=$PROJ/logs variant=baseline offline_mix.mode=none load_offline_data=False
# A run is skipped when its .out already contains [done]; a run with a checkpoint
# resumes from it (resume=True is the config default). Re-running this script after
# a Spot preemption is therefore enough to continue.
set -euo pipefail

PROJ=/content/drive/MyDrive/dsrl_project
cd /content/dsrl
# shellcheck disable=SC1091
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
# shellcheck disable=SC1091
source /content/env.sh

status () {
  echo "running: $(pgrep -fc '[t]rain_dsrl.py' || true)  $(pgrep -fa 'train_dsrl.py' | grep -o 'exp_id=[a-z_0-9]*' | sed 's/exp_id=//' | tr '\n' ' ')"
  free -g | awk 'NR==2{print "ram", $3"/"$2}'
  for f in "$PROJ"/logs/*/eval_log.csv; do
    E=$(basename "$(dirname "$f")")
    printf "%-24s done=%s last_eval=%s\n" "$E" "$(grep -c '\[done\]' "$PROJ/logs/$E.out" 2>/dev/null || echo 0)" "$(tail -n 1 "$f" | cut -d, -f2,5)"
  done
}

if [ "${1:-}" = "--status" ]; then status; exit 0; fi
RUNS=${1:?runs file}
WAIT=0; [ "${2:-}" = "--wait" ] && WAIT=1

git pull origin o2o | tail -n 1
started=0
while read -r EXP TASK REST; do
  [[ -z "$EXP" || "$EXP" == \#* ]] && continue
  case "$TASK" in
    can)    CFG="--config-path=cfg/robomimic --config-name=dsrl_can.yaml" ;;
    square) CFG="--config-path=cfg/robomimic --config-name=dsrl_square.yaml" ;;
    *) echo "unknown task '$TASK' for $EXP"; exit 1 ;;
  esac
  if grep -q '\[done\]' "$PROJ/logs/$EXP.out" 2>/dev/null; then echo "skip $EXP (done)"; continue; fi
  if pgrep -f "exp_id=$EXP " >/dev/null; then echo "skip $EXP (running)"; continue; fi
  COMMON="log_dir=$PROJ/logs variant=baseline offline_mix.mode=none load_offline_data=False"
  # shellcheck disable=SC2086
  nohup python train_dsrl.py $CFG exp_id="$EXP" $COMMON $REST >> "$PROJ/logs/$EXP.out" 2>&1 &
  echo "started $EXP (pid $!): $REST"
  started=$((started + 1))
  sleep 2
done < "$RUNS"
echo "$started run(s) started"

if [ "$WAIT" = 1 ]; then
  while [ "$(pgrep -fc '[t]rain_dsrl.py' || true)" -gt 0 ]; do
    echo "$(date +%H:%M) $(pgrep -fa 'train_dsrl.py' | grep -o 'exp_id=[a-z_0-9]*' | sed 's/exp_id=//' | tr '\n' ' ')"
    sleep 600
  done
  status
  bash /content/dsrl/gce/sync_drive.sh push || echo "push failed (rclone not configured?) - results are on the VM disk"
  echo "all runs finished; stopping the VM in 60 s"
  sleep 60
  sudo shutdown -h now
fi
