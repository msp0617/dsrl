#!/usr/bin/env bash
# One-time setup of a Compute Engine VM so that every command from the Colab
# notebook runs verbatim: the repo lives in /content/dsrl, the conda env in
# /usr/local/envs/dsrl, and /content/drive/MyDrive/dsrl_project is a symlink
# to a directory on the VM's disk ($DATA, default /data/dsrl_project).
#
#   bash gce/setup_gce.sh            # restore the env from $DATA/env_cache/dsrl_env.tar.gz
#   bash gce/setup_gce.sh --build    # no cache: build the env from scratch (15 min)
#
# Run gce/sync_drive.sh pull first (or copy the Drive folder some other way) so
# that $DATA holds env_cache/, dppo_log/, offline/ and logs/pretrain/.
# Tested layout: Deep Learning VM image (Ubuntu 22.04, NVIDIA driver preinstalled).
set -euo pipefail

DATA=${DATA:-/data/dsrl_project}
BUILD=0
[ "${1:-}" = "--build" ] && BUILD=1

echo "=== 1. directories and the Drive-compatible symlink ==="
sudo mkdir -p /content/drive/MyDrive "$DATA"
sudo chown -R "$USER":"$USER" /content "$DATA"
ln -sfn "$DATA" /content/drive/MyDrive/dsrl_project
mkdir -p "$DATA"/{logs,offline,dppo_log,env_cache,ckpt,cfg_backup,figures,videos}
ls -la /content/drive/MyDrive/

echo "=== 2. system packages (EGL for headless MuJoCo, build tools) ==="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  git tmux rclone curl zip unzip patchelf build-essential \
  libegl1 libgl1 libglew-dev libosmesa6-dev libglfw3 libgl1-mesa-dev ffmpeg
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

echo "=== 3. miniconda at /usr/local (same layout as condacolab) ==="
if [ ! -x /usr/local/bin/conda ]; then
  curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -o /tmp/miniconda.sh
  sudo bash /tmp/miniconda.sh -b -u -p /usr/local
  sudo chown -R "$USER":"$USER" /usr/local/envs /usr/local/pkgs 2>/dev/null || true
fi
sudo mkdir -p /usr/local/envs && sudo chown "$USER":"$USER" /usr/local/envs
# shellcheck disable=SC1091
source /usr/local/etc/profile.d/conda.sh

echo "=== 4. repository ==="
git config --global url."https://github.com/".insteadOf "git@github.com:"
if [ ! -d /content/dsrl/.git ]; then
  git clone --recurse-submodules -b o2o https://github.com/msp0617/dsrl.git /content/dsrl
fi
cd /content/dsrl && git pull origin o2o && git log --oneline -n 1

echo "=== 5. conda env ==="
CACHE="$DATA/env_cache/dsrl_env.tar.gz"
if [ "$BUILD" = 0 ] && [ -f "$CACHE" ]; then
  echo "restoring from $CACHE"
  cd /usr/local/envs && rm -rf dsrl && tar -xzf "$CACHE"
else
  echo "building from scratch (notebook cells 4-5)"
  conda env remove -n dsrl -y 2>/dev/null || true
  conda create -n dsrl python=3.10 -y -q
  conda activate dsrl
  (cd /content/dsrl/dppo && pip install -q -e . && pip install -q -e ".[robomimic]")
  (cd /content/dsrl/stable-baselines3 && pip install -q -e .)
  pip install -q gdown
  python -m pip install -q "mujoco==3.1.6"
  python -m pip install -q "cython<3" patchelf
  python -m pip install -q "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@v1.4.1"
  python -m pip install -q "cmake==3.31.6"
  python -m pip install -q --no-build-isolation "egl_probe==1.0.2"
  python -m pip install -q "robomimic==0.3.0"
  python -m pip install -q --force-reinstall "numpy==1.26.4" "opencv-python==4.9.0.80"
  # cu128 wheels run on every current GPU (T4/L4/A100/Blackwell); the driver must be >= 525.
  python -m pip install -q "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
  conda deactivate
fi
conda activate dsrl
python /usr/local/envs/dsrl/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py || true
python /content/dsrl/colab/patch_env.py

echo "=== 6. env.sh (notebook section 7) ==="
cat > /content/env.sh <<'EOS'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled
EOS

echo "=== 7. pi_dp checkpoints into dppo/log (notebook section 6, Can + Square) ==="
RUNTIME=/content/dsrl/dppo/log
DRIVE="$DATA/dppo_log"
mkdir -p "$RUNTIME"
if [ -z "$(ls -A "$DRIVE" 2>/dev/null)" ]; then
  echo "dppo_log is empty: downloading the public checkpoints (once)"
  (cd "$RUNTIME" && gdown --folder https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1)
  cp -r "$RUNTIME"/. "$DRIVE"/
fi
place () {  # place <relative path under dppo/log>
  local REL=$1 SRC
  SRC=$(find "$RUNTIME" "$DRIVE" -type f -path "*/$REL" -print -quit 2>/dev/null || true)
  if [ -z "$SRC" ]; then echo "MISSING: $REL"; return 0; fi
  mkdir -p "$RUNTIME/$(dirname "$REL")"
  [ "$SRC" = "$RUNTIME/$REL" ] || cp -f "$SRC" "$RUNTIME/$REL"
  echo "OK: $REL"
}
place robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt
place robomimic/can/normalization.npz
place robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td100_ddim-100steps/2025-04-11_19-13-26_44/checkpoint/state_3000.pt
place robomimic/square/normalization.npz

echo "=== 8. verification (notebook cell 20) ==="
# shellcheck disable=SC1091
source /content/env.sh
python - <<'PY'
import torch, mujoco, robomimic, robosuite, stable_baselines3 as sb3
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device="cuda"); print("matmul ok", (x @ x).sum().item() != 0)
print("mujoco", mujoco.__version__, "robomimic", robomimic.__version__, "robosuite", robosuite.__version__, "sb3", sb3.__version__)
PY
echo
echo "data under $DATA:"; du -sh "$DATA"/* 2>/dev/null || true
echo
echo "setup done. Next: bash gce/launch.sh gce/runs_example.txt  (or write your own runs file)"
