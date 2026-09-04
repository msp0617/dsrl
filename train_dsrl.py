import os
import warnings

warnings.filterwarnings("ignore")
import math
import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import gym

sys.path.append('./dppo')

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env_utils import (
    DiffusionPolicyEnvWrapper,
    ObservationWrapperRobomimic,
    ObservationWrapperGym,
    ActionChunkWrapper,
    make_robomimic_env,
)
from o2o_utils import (
    VARIANTS,
    ResumeCheckpointCallback,
    build_agent,
    check_buffer_capacity,
    check_fingerprint,
    config_fingerprint,
    load_pretrained_weights,
    read_run_states,
    resolve_ckpt_dir,
    restore_agent,
    restore_rng_state,
)
from utils import load_base_policy, load_offline_data, collect_rollouts, LoggingCallback

try:  # wandb is optional
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))

GYM_ENVS = ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']
ROBOMIMIC_ENVS = ['lift', 'can', 'square', 'transport']


@hydra.main(
    config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    # Resuming needs a log directory that is the same in the next session, so a
    # run started with exp_id=<name> writes to ${log_dir}/<name> instead of a
    # timestamped folder.
    exp_id = str(cfg.get("exp_id", "") or "")
    if exp_id:
        cfg.logdir = os.path.join(cfg.log_dir, exp_id)
    os.makedirs(cfg.logdir, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.use_wandb:
        if wandb is None:
            raise ImportError("use_wandb=True but wandb is not installed")
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name,
            group=cfg.wandb.group,
            monitor_gym=True,
            save_code=True,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

    num_env = cfg.env.n_envs

    # The variant decides only where the initial critic weights come from:
    # baseline starts random, warmup and iql load a file from offline_pretrain.py.
    variant = str(cfg.get("variant", "baseline") or "baseline")
    pretrain_path = str(cfg.get("pretrain_path", "") or "")
    if variant not in VARIANTS:
        raise ValueError("variant must be one of %s, got %r" % (", ".join(VARIANTS), variant))
    if variant != "baseline" and not pretrain_path:
        raise ValueError("variant=%s needs pretrain_path=<file written by offline_pretrain.py>" % variant)
    if variant != "baseline" and not os.path.exists(pretrain_path):
        raise FileNotFoundError("pretrain_path does not exist: %s" % pretrain_path)

    # A 10M-transition buffer costs several GB and makes checkpointing the
    # buffer impractical; size it to the run instead.
    default_buffer_size = 20000000 if cfg.algorithm == 'dsrl_sac' else 10000000
    buffer_size = int(cfg.train.get("buffer_size", default_buffer_size))
    offline_mix_ratio = float(cfg.train.get("offline_mix_ratio", -1))
    replay_buffer_kwargs = None
    if offline_mix_ratio > 0:
        replay_buffer_kwargs = dict(offline_mix_ratio=offline_mix_ratio)
    check_buffer_capacity(cfg, buffer_size)

    # Everything that can rule out this run is checked before the environments
    # and the diffusion policy are built, which costs minutes.
    ckpt_dir = resolve_ckpt_dir(cfg)
    candidates = read_run_states(ckpt_dir) if cfg.get("resume", True) else []
    if candidates:
        check_fingerprint(candidates[0], cfg)
        if all(c["replay_buffer_path"] is None for c in candidates):
            raise RuntimeError(
                "There is a checkpoint in %s but no replay buffer beside it. An off-policy "
                "agent resumed with an empty buffer would train on almost nothing and quietly "
                "destroy its critics. Re-run with resume=False to start over, or remove the "
                "directory. Runs meant to be resumed need save_replay_buffer=True." % ckpt_dir
            )

    def make_env():
        if cfg.env_name in GYM_ENVS:
            import d4rl  # noqa: F401  (only the gym tasks need d4rl)
            import d4rl.gym_mujoco  # noqa: F401

            env = gym.make(cfg.env_name)
            env = ObservationWrapperGym(env, cfg.normalization_path)
        elif cfg.env_name in ROBOMIMIC_ENVS:
            env = make_robomimic_env(env=cfg.env_name, normalization_path=cfg.normalization_path, low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
            env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env

    base_policy = load_base_policy(cfg)
    env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
    if cfg.algorithm == 'dsrl_sac':
        env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
    env.seed(cfg.seed + 1)
    model = build_agent(
        cfg, env, base_policy, buffer_size,
        replay_buffer_kwargs=replay_buffer_kwargs,
        tensorboard_log=cfg.logdir,
        verbose=1,
    )

    num_env_eval = cfg.env.n_eval_envs
    eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
    if cfg.algorithm == 'dsrl_sac':
        eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
    eval_env.seed(cfg.seed + num_env + 1)

    # Evaluation schedule. When the cfg carries an `eval` block the schedule is
    # expressed in environment steps (the paper's unit) and can be denser over
    # the early phase; otherwise the upstream callback-count schedule is kept.
    eval_cfg = cfg.get("eval_schedule", None)
    eval_every_env = int(eval_cfg.get("every_env", 0)) if eval_cfg else 0
    eval_every_env_early = int(eval_cfg.get("every_env_early", 0)) if eval_cfg else 0
    eval_early_until_env = int(eval_cfg.get("early_until_env", 0)) if eval_cfg else 0
    num_evals_early = int(eval_cfg.get("num_evals_early", 0)) if eval_cfg else 0

    logging_callback = LoggingCallback(
        action_chunk = cfg.act_steps,
        eval_episodes = int(cfg.num_evals / num_env_eval),
        log_freq=MAX_STEPS,
        use_wandb=cfg.use_wandb,
        eval_env=eval_env,
        eval_freq=cfg.eval_interval,
        num_train_env=num_env,
        num_eval_env=num_env_eval,
        rew_offset=cfg.env.reward_offset,
        algorithm=cfg.algorithm,
        max_steps=MAX_STEPS,
        deterministic_eval=cfg.deterministic_eval,
        csv_dir=cfg.logdir,
        eval_every_env=eval_every_env,
        eval_every_env_early=eval_every_env_early,
        eval_early_until_env=eval_early_until_env,
        eval_episodes_early=int(num_evals_early / num_env_eval) if num_evals_early else 0,
    )

    state = None
    for candidate in candidates:
        try:
            restore_agent(model, candidate["model_path"], device=cfg.device)
            model.load_replay_buffer(candidate["replay_buffer_path"])
        except Exception as exc:  # truncated archive, interrupted save, bad pickle
            print(
                "[resume] slot %s did not load (%s: %s), falling back"
                % (candidate["slot"], type(exc).__name__, exc),
                flush=True,
            )
            continue
        state = candidate
        break

    if candidates and state is None:
        raise RuntimeError(
            "Found checkpoints in %s but none of them could be loaded." % ckpt_dir
        )

    if state is not None:
        if state["rng_path"] is not None:
            restore_rng_state(state["rng_path"])
        model.num_timesteps = int(state["num_timesteps"])
        model._episode_num = int(state.get("episode_num", 0))
        model._n_updates = int(state.get("n_updates", 0))
        logging_callback.load_state(state)
        print(
            "[resume] %s at %d env steps (%d sb3 steps), buffer %d transitions"
            % (
                state["model_path"],
                logging_callback.total_timesteps,
                model.num_timesteps,
                model.replay_buffer.size(),
            ),
            flush=True,
        )
    else:
        if variant != "baseline":
            # Loaded before the first evaluation so that the step-0 point in the
            # curve measures the policy this run actually starts from.
            pretrain_cfg = cfg.get("pretrain", None)
            load_ent_coef = bool(pretrain_cfg.get("load_ent_coef", False)) if pretrain_cfg else False
            load_actor = bool(pretrain_cfg.get("load_actor", True)) if pretrain_cfg else True
            meta, loaded = load_pretrained_weights(
                model, pretrain_path, cfg, variant, load_ent_coef=load_ent_coef, load_actor=load_actor
            )
            print(
                "[pretrain] %s: loaded %s from %s (%s steps on %s)"
                % (variant, ", ".join(loaded), pretrain_path, meta.get("steps"), meta.get("offline_data_path")),
                flush=True,
            )

        logging_callback.evaluate(model, deterministic=False)
        if cfg.deterministic_eval:
            logging_callback.evaluate(model, deterministic=True)
        logging_callback.log_count += 1

        if cfg.load_offline_data:
            load_offline_data(model, cfg.offline_data_path, num_env)
        if cfg.train.init_rollout_steps > 0:
            collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)
            # One rollout step is one action chunk in every parallel env.
            logging_callback.set_timesteps(
                cfg.train.init_rollout_steps * num_env * cfg.act_steps
            )

    # Checkpoint period in environment steps; falls back to the upstream key,
    # which counts callback calls.
    ckpt_every_env_steps = int(
        cfg.get("ckpt_every_env_steps", 0)
        or cfg.save_model_interval * num_env * cfg.act_steps
    )
    checkpoint_callback = ResumeCheckpointCallback(
        save_dir=ckpt_dir,
        every_env_steps=ckpt_every_env_steps,
        logging_callback=logging_callback,
        save_replay_buffer=cfg.save_replay_buffer,
        fingerprint=config_fingerprint(cfg),
    )

    # Budget. `total_env_steps` is in original-environment steps and covers the
    # initial rollout as well, so that runs are comparable to the paper's x axis.
    total_env_steps = int(cfg.train.get("total_env_steps", 0))
    if total_env_steps > 0:
        remaining_env_steps = max(total_env_steps - logging_callback.total_timesteps, 0)
        learn_timesteps = int(math.ceil(remaining_env_steps / cfg.act_steps))
    else:
        remaining_env_steps = -1
        learn_timesteps = 20000000

    print(
        "[budget] env steps done %d, target %s, learning for %d sb3 timesteps"
        % (logging_callback.total_timesteps, total_env_steps or "unbounded", learn_timesteps),
        flush=True,
    )

    callbacks = [checkpoint_callback, logging_callback]
    # Train the agent
    if learn_timesteps > 0:
        model.learn(
            total_timesteps=learn_timesteps,
            callback=callbacks,
            reset_num_timesteps=state is None,
        )

    # The rolling checkpoint writes on training end, so there is no separate
    # final archive to save.
    print("[done] %d env steps, checkpoint in %s" % (logging_callback.total_timesteps, ckpt_dir), flush=True)

    # Close environment and wandb
    env.close()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
