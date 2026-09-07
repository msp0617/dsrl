"""Offline pass criteria for a pre-trained critic, before it goes online.

    python scripts/check_pretrain.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \\
        pretrain_path=$PROJ/logs/pretrain/calql_can_s1.pt offline_data_path=$PROJ/offline/can_train_offline.npz

Loads Q_A from the .pt, samples states from the chunk file and reports, per
state and averaged: Q_A(s, a_data), E_w Q_A(s, pi_dp(s, w)) over K prior
noises, the demonstration's return-to-go G(s), and from them

  Q_data - G          how far the data action's value sits from what the data earned
  calibration gap     E_w Q_ood - G   (Cal-QL: should be near 0 or above)
  Pr[E_w Q_ood < G]   share of states whose unseen actions are valued below the return
  Q_ood - Q_data      the conservative gap (CQL: negative, same order of magnitude)

Rule of thumb on Can (G about -100): Q_data within a few tens of G and
Q_ood between -120 and -200 is fine; Q_data > 0 or Q_ood < -1,000 is not.
No simulator is needed (SpacesOnlyEnv), only the diffusion policy and a GPU.
"""

import os
import sys

import hydra
import numpy as np
import torch as th
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from o2o_utils import OfflineBuffer, SpacesOnlyEnv, build_agent  # noqa: E402
from offline_pretrain import combine_q, diffusion_actions, prior_noise  # noqa: E402
from utils import load_base_policy  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    path = str(cfg.pretrain_path)
    n_states = int(cfg.get("check_states", 4096))
    k = int(cfg.pretrain.cql_n_samples)
    payload = th.load(path, map_location="cuda" if th.cuda.is_available() else "cpu", weights_only=False)
    meta = payload.get("meta") or {}
    env = DummyVecEnv([lambda: SpacesOnlyEnv(int(cfg.obs_dim), int(cfg.act_steps) * int(cfg.action_dim))])
    model = build_agent(cfg, env, load_base_policy(cfg), buffer_size=1024, tensorboard_log=None, verbose=0)
    model.set_logger(configure(None, []))
    model.critic.load_state_dict(payload["critic"])
    model.critic.set_training_mode(False)
    buf = OfflineBuffer(str(cfg.offline_data_path), model.device)
    combine = str(cfg.train.critic_backup_combine_type)

    th.manual_seed(0)
    q_data, q_ood, ret = [], [], []
    with th.no_grad():
        for _ in range(0, n_states, 512):
            batch, returns = buf.sample(512, with_returns=True)
            obs, actions = batch.observations, batch.actions
            q_data.append(combine_q(model.critic(obs, actions), combine))
            obs_rep = obs.repeat_interleave(k, dim=0)
            a_ood = diffusion_actions(model, obs_rep, prior_noise(512 * k, model, float(cfg.pretrain.cql_noise_clip)))
            q_ood.append(combine_q(model.critic(obs_rep, a_ood), combine).reshape(512, k).mean(dim=1, keepdim=True))
            ret.append(returns if returns is not None else th.full_like(q_data[-1], float("nan")))
    q_data, q_ood, ret = (th.cat(x).squeeze(1).cpu().numpy() for x in (q_data, q_ood, ret))

    print("== %s  (method %s, steps %s, cql_alpha %s, created %s)" % (
        os.path.basename(path), meta.get("method"), meta.get("steps"), meta.get("cql_alpha"), meta.get("created")))
    print("states %d, %d prior noises each" % (len(q_data), k))
    print("Q_data          mean %8.1f  [p10 %8.1f  p90 %8.1f]" % (q_data.mean(), *np.percentile(q_data, [10, 90])))
    print("E_w Q_ood       mean %8.1f  [p10 %8.1f  p90 %8.1f]" % (q_ood.mean(), *np.percentile(q_ood, [10, 90])))
    if np.isfinite(ret).all():
        print("G (return)      mean %8.1f  [p10 %8.1f  p90 %8.1f]" % (ret.mean(), *np.percentile(ret, [10, 90])))
        print("Q_data - G      mean %8.1f   (|.| < a few tens of G's scale is fine)" % (q_data - ret).mean())
        print("calibration gap E_w Q_ood - G   mean %8.1f   Pr[E_w Q_ood < G] %.2f" % ((q_ood - ret).mean(), (q_ood < ret).mean()))
    print("Q_ood - Q_data  mean %8.1f   Pr[Q_ood < Q_data] %.2f   (CQL: negative, same order of magnitude)"
          % ((q_ood - q_data).mean(), (q_ood < q_data).mean()))
    verdict = "OK"
    if q_data.mean() > 0 or q_ood.mean() < 10 * min(ret.mean() if np.isfinite(ret).all() else -100, -100):
        verdict = "REJECT (Q_data positive or Q_ood an order of magnitude below the returns)"
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
