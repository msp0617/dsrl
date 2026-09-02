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

from stable_baselines3 import SAC
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
    DSRLResumable,
    ResumeCheckpointCallback,
    read_run_state,
    resolve_ckpt_dir,
    restore_agent,
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
    post_linear_modules = None
    if cfg.train.use_layer_norm:
        post_linear_modules = [torch.nn.LayerNorm]

    net_arch = []
    for _ in range(cfg.train.num_layers):
        net_arch.append(cfg.train.layer_size)
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )

    # A 10M-transition buffer costs several GB and makes checkpointing the
    # buffer impractical; size it to the run instead.
    buffer_size = int(cfg.train.get("buffer_size", 10000000))
    offline_mix_ratio = float(cfg.train.get("offline_mix_ratio", -1))
    replay_buffer_kwargs = None
    if offline_mix_ratio > 0:
        replay_buffer_kwargs = dict(offline_mix_ratio=offline_mix_ratio)

    if cfg.algorithm == 'dsrl_sac':
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=buffer_size,      # Replay buffer size
            learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
            batch_size=cfg.train.batch_size,
            tau=cfg.train.tau,                # Target network update rate
            gamma=cfg.train.discount,               # Discount factor
            train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
            gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
            action_noise=None,        # No additional action noise
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=False,
            ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
            target_update_interval=1, # Update target network every interval
            target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=cfg.logdir,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    elif cfg.algorithm == 'dsrl_na':
        model = DSRLResumable(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=buffer_size,      # Replay buffer size
            learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
            batch_size=cfg.train.batch_size,
            tau=cfg.train.tau,                # Target network update rate
            gamma=cfg.train.discount,               # Discount factor
            train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
            gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
            action_noise=None,        # No additional action noise
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=False,
            ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
            target_update_interval=1, # Update target network every interval
            target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=cfg.logdir,
            verbose=1,
            policy_kwargs=policy_kwargs,
            diffusion_policy=base_policy,
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
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
    )

    ckpt_dir = resolve_ckpt_dir(cfg)
    state = read_run_state(ckpt_dir) if cfg.get("resume", True) else None

    if state is not None:
        restore_agent(model, state["model_path"], device=cfg.device)
        if state["replay_buffer_path"] is not None:
            model.load_replay_buffer(state["replay_buffer_path"])
        else:
            warnings.warn(
                "Resuming without a replay buffer: an off-policy run restarted "
                "with an empty buffer is not a continuation of the same run."
            )
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

    # Save the final model
    if len(cfg.name) > 0:
        model.save(os.path.join(ckpt_dir, "final"))

    # Close environment and wandb
    env.close()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
