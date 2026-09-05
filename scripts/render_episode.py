"""Render evaluation episodes of pi_dp, or of a trained run, to a video file.

  # the frozen diffusion policy under N(0, I) noise
  python scripts/render_episode.py +policy=pi_dp +episodes=2 +out=$PROJ/videos

  # a trained run, from its latest checkpoint in ${log_dir}/<exp_id>/checkpoint
  python scripts/render_episode.py +policy=can_baseline_s1 +episodes=2 +out=$PROJ/videos log_dir=$PROJ/logs

Other tasks: add --config-name=dsrl_square.yaml (not --config-path). Frames
come from robosuite's offscreen renderer (MUJOCO_GL=egl), one per environment
step, so the clip runs at the controller's 20 Hz. Writes <out>/<policy>_ep<i>_<ok|fail>.mp4
when imageio-ffmpeg is available, otherwise a .gif.
"""

import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('./dppo')

from stable_baselines3.common.vec_env import DummyVecEnv

from env_utils import ObservationWrapperRobomimic, make_robomimic_env
from o2o_utils import SpacesOnlyEnv, build_agent, read_run_states, restore_agent
from utils import load_base_policy

OmegaConf.register_new_resolver("eval", eval, replace=True)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_video(path_no_ext, frames, fps):
    import imageio

    try:
        with imageio.get_writer(path_no_ext + ".mp4", fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)
        return path_no_ext + ".mp4"
    except Exception as exc:  # no imageio-ffmpeg on this VM
        print("mp4 writer unavailable (%s), writing gif" % exc)
        imageio.mimsave(path_no_ext + ".gif", frames, duration=int(1000 / fps), loop=0)
        return path_no_ext + ".gif"


@hydra.main(config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    policy_name = str(cfg.get("policy", "pi_dp"))
    episodes = int(cfg.get("episodes", 2))
    fps = int(cfg.get("fps", 20))
    out_dir = str(cfg.get("out", os.path.join(str(cfg.log_dir), "videos")))
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    inner = make_robomimic_env(
        render=True, env=cfg.env_name, normalization_path=cfg.normalization_path,
        low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path,
    )
    env = ObservationWrapperRobomimic(inner, reward_offset=cfg.env.reward_offset)
    env.seed(cfg.seed + 100)
    base_policy = load_base_policy(cfg)
    act_steps, action_dim = int(cfg.act_steps), int(cfg.action_dim)

    model = None
    if policy_name != "pi_dp":
        ckpt_dir = os.path.join(str(cfg.log_dir), policy_name, "checkpoint")
        candidates = read_run_states(ckpt_dir)
        if not candidates:
            raise SystemExit("no checkpoint under %s" % ckpt_dir)
        spaces_env = DummyVecEnv([lambda: SpacesOnlyEnv(int(cfg.obs_dim), act_steps * action_dim)])
        model = build_agent(cfg, spaces_env, base_policy, buffer_size=1024, tensorboard_log=None, verbose=0)
        restore_agent(model, candidates[0]["model_path"], device=cfg.device)
        print("loaded %s at %s env steps" % (candidates[0]["model_path"], candidates[0].get("env_steps")), flush=True)

    def chunk(obs):
        obs_t = torch.as_tensor(obs[None], device=cfg.device, dtype=torch.float32)
        if model is None:
            noise = torch.randn(1, act_steps, action_dim, device=cfg.device)
            return base_policy(obs_t, noise)[0]
        action, _ = model.predict_diffused(obs[None], deterministic=False)
        return np.asarray(action).reshape(act_steps, action_dim)

    max_chunks = int(cfg.env.max_episode_steps) // act_steps
    for ep in range(episodes):
        obs = env.reset()
        frames, success = [inner.render(mode="rgb_array")], False
        for _ in range(max_chunks):
            actions = chunk(obs)
            for a in actions:
                obs, reward, done, info = env.step(a)
                frames.append(inner.render(mode="rgb_array"))
                if reward > -cfg.env.reward_offset:
                    success = True
            if success and len(frames) > 40:
                break  # a short tail after the first success is enough to see it
        name = "%s_%s_ep%d_%s" % (cfg.env_name, policy_name, ep, "ok" if success else "fail")
        path = write_video(os.path.join(out_dir, name), frames, fps)
        print("[video] %s  %d frames  success=%s" % (path, len(frames), success), flush=True)


if __name__ == "__main__":
    main()
