"""Success rate of the frozen diffusion policy driven by pure N(0, I) noise.

The step-0 point of a run measures pi_dp fed by a *random* pi_W, whose bias
differs from seed to seed (0.34 to 0.66 on Can in practice). The regret of a
run is better measured against pi_dp with the noise it was trained for, which
is what this script evaluates, with the same episode protocol as
LoggingCallback.evaluate.

  python scripts/eval_base_policy.py num_evals=200 seed=1 log_dir=$PROJ/logs

(no --config-path: Hydra resolves that flag relative to this script's own
directory; the decorator below already points at the repo's cfg/robomimic.)

Appends one row to ${log_dir}/base_policy_eval.csv (Can) or
base_policy_eval_<env>.csv. For another task pass --config-name=dsrl_square.yaml
(config-name is fine on the command line; only --config-path is not).
"""

import csv
import os
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('./dppo')

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env_utils import ActionChunkWrapper, ObservationWrapperRobomimic, make_robomimic_env
from utils import load_base_policy

OmegaConf.register_new_resolver("eval", eval, replace=True)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    def make_env():
        env = make_robomimic_env(
            env=cfg.env_name, normalization_path=cfg.normalization_path,
            low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path,
        )
        env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        return ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)

    n_envs = int(cfg.env.n_eval_envs)
    episodes = int(cfg.num_evals) // n_envs
    max_steps = int(cfg.env.max_episode_steps // cfg.act_steps)
    base_policy = load_base_policy(cfg)
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env.seed(cfg.seed + int(cfg.env.n_envs) + 1)  # same seed the run's eval env gets

    success = []
    started = time.time()
    for i in range(episodes):
        obs = env.reset()
        success_i = np.zeros(obs.shape[0])
        for _ in range(max_steps):
            noise = torch.randn(n_envs, cfg.act_steps, cfg.action_dim, device=cfg.device)
            action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
            action = action.reshape(n_envs, -1)
            obs, reward, done, info = env.step(action)
            success_i[reward > -cfg.env.reward_offset] = 1
        success.append(success_i.mean())
        print("episode batch %d/%d  success so far %.3f" % (i + 1, episodes, float(np.mean(success))), flush=True)
    env.close()

    rate = float(np.mean(success))
    n = episodes * n_envs
    row = {
        "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(cfg.seed),
        "episodes": n,
        "success_rate": rate,
        "stderr": float(np.sqrt(rate * (1 - rate) / n)),
    }
    # Can keeps the original file name; other tasks get their own file so the
    # rows never mix (the plotter picks the file by task).
    name = "base_policy_eval.csv" if str(cfg.env_name) == "can" else "base_policy_eval_%s.csv" % cfg.env_name
    out = os.path.join(str(cfg.log_dir), name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_header = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print("[base] pi_dp with N(0,I) noise: success %.3f +- %.3f over %d episodes in %.0fs -> %s"
          % (rate, row["stderr"], n, time.time() - started, out), flush=True)


if __name__ == "__main__":
    main()
