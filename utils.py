import csv
import os
import time

import hydra
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

try:  # wandb is optional: runs on Colab often log to CSV only
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class DPPOBasePolicyWrapper:
    def __init__(self, base_policy):
        self.base_policy = base_policy

    def __call__(self, obs, initial_noise, return_numpy=True):
        cond = {
            "state": obs,
            "noise_action": initial_noise,
        }
        with torch.no_grad():
            samples = self.base_policy(cond=cond, deterministic=True)
        diffused_actions = (samples.trajectories.detach())
        if return_numpy:
            diffused_actions = diffused_actions.cpu().numpy()
        return diffused_actions


def load_base_policy(cfg):
    base_policy = hydra.utils.instantiate(cfg.model)
    base_policy = base_policy.eval()
    return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
    """Upstream logging callback, extended for the offline-to-online study.

    Changes relative to ajwagen/dsrl:
      * every metric is also appended to CSV, so a killed Colab session still
        leaves a usable learning curve behind;
      * evaluation can be scheduled on environment steps rather than callback
        calls, which keeps the period meaningful across ``n_envs`` and allows a
        denser schedule over the early phase where the initial dip lives;
      * the counters survive a resume via ``state_dict`` / ``load_state``.

    ``total_timesteps`` counts steps in the original environment (the unit used
    by the paper), i.e. ``act_steps * n_envs`` per callback call.
    """

    def __init__(
        self,
        action_chunk=4,
        log_freq=1000,
        use_wandb=True,
        eval_env=None,
        eval_freq=70,
        eval_episodes=2,
        verbose=0,
        rew_offset=0,
        num_train_env=1,
        num_eval_env=1,
        algorithm='dsrl_sac',
        max_steps=-1,
        deterministic_eval=False,
        csv_dir=None,
        eval_every_env=None,
        eval_every_env_early=None,
        eval_early_until_env=0,
        eval_episodes_early=0,
    ):
        super().__init__(verbose)
        self.action_chunk = action_chunk
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.use_wandb = use_wandb and wandb is not None
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.log_count = 0
        self.total_reward = 0
        self.rew_offset = rew_offset
        self.total_timesteps = 0
        self.num_train_env = num_train_env
        self.num_eval_env = num_eval_env
        self.episode_success = np.zeros(self.num_train_env)
        self.episode_completed = np.zeros(self.num_train_env)
        self.algorithm = algorithm
        self.max_steps = max_steps
        self.deterministic_eval = deterministic_eval

        self.csv_dir = csv_dir
        if self.csv_dir is not None:
            os.makedirs(self.csv_dir, exist_ok=True)
        self.eval_every_env = int(eval_every_env) if eval_every_env else 0
        self.eval_every_env_early = int(eval_every_env_early or eval_every_env or 0)
        self.eval_early_until_env = int(eval_early_until_env or 0)
        self.eval_episodes_early = int(eval_episodes_early or 0)
        self.next_eval_at = None

    # ------------------------------------------------------------------ csv
    def _csv_append(self, filename, row):
        if self.csv_dir is None:
            return
        path = os.path.join(self.csv_dir, filename)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _now():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------- schedule
    def eval_interval_env(self):
        if self.eval_every_env_early and self.total_timesteps < self.eval_early_until_env:
            return self.eval_every_env_early
        return self.eval_every_env

    def episodes_now(self):
        """Episodes per evaluation, which may be smaller in the dense early phase.

        One evaluation costs eval_episodes x n_eval_envs episodes of simulation,
        which is a real share of the wall clock when evaluations are frequent.
        Trading episodes for points is the better deal while resolving the dip:
        the curve can be smoothed across points afterwards.
        """
        if self.eval_episodes_early and self.total_timesteps < self.eval_early_until_env:
            return self.eval_episodes_early
        return self.eval_episodes

    def arm_eval_schedule(self):
        if self.eval_every_env:
            self.next_eval_at = self.total_timesteps + self.eval_interval_env()

    def set_timesteps(self, timesteps):
        self.total_timesteps = int(timesteps)
        self.arm_eval_schedule()

    def state_dict(self):
        return {
            "env_steps": int(self.total_timesteps),
            "log_count": int(self.log_count),
            "next_eval_at": int(self.next_eval_at or 0),
        }

    def load_state(self, state):
        self.total_timesteps = int(state.get("env_steps", 0))
        self.log_count = int(state.get("log_count", 0))
        next_eval_at = int(state.get("next_eval_at", 0))
        if next_eval_at > 0:
            self.next_eval_at = next_eval_at
        else:
            self.arm_eval_schedule()

    # ----------------------------------------------------------------- step
    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        rew = self.locals['rewards']
        self.total_reward += np.mean(rew)
        self.episode_success[rew > -self.rew_offset] = 1
        self.episode_completed[self.locals['dones']] = 1
        self.total_timesteps += self.action_chunk * self.model.n_envs

        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                self.log_train()
                self.episode_rewards = []
                self.episode_lengths = []
                self.total_reward = 0
                self.episode_success = np.zeros(self.num_train_env)
                self.episode_completed = np.zeros(self.num_train_env)

        if self.eval_every_env:
            if self.next_eval_at is None:
                self.arm_eval_schedule()
            if self.total_timesteps >= self.next_eval_at:
                self.evaluate(self.locals['self'], deterministic=False)
                if self.deterministic_eval:
                    self.evaluate(self.locals['self'], deterministic=True)
                self.arm_eval_schedule()
        elif self.n_calls % self.eval_freq == 0:
            self.evaluate(self.locals['self'], deterministic=False)
            if self.deterministic_eval:
                self.evaluate(self.locals['self'], deterministic=True)
        return True

    def log_train(self):
        agent = self.locals['self']
        values = agent.logger.name_to_value
        completed = float(np.sum(self.episode_completed))
        success_rate = float(np.sum(self.episode_success) / completed) if completed > 0 else float('nan')
        row = {
            "wall_time": self._now(),
            "env_steps": int(self.total_timesteps),
            "sb3_timesteps": int(getattr(agent, "num_timesteps", 0)),
            "success_rate": success_rate,
            "ep_rew_mean": float(np.mean(self.episode_rewards)),
            "ep_len_mean": float(np.mean(self.episode_lengths)),
            "rew_mean": float(np.mean(self.total_reward)),
            "actor_loss": float(values.get('train/actor_loss', float('nan'))),
            "critic_loss": float(values.get('train/critic_loss', float('nan'))),
            "noise_critic_loss": float(values.get('train/noise_critic_loss', float('nan'))),
            "ent_coef": float(values.get('train/ent_coef', float('nan'))),
            "ent_coef_loss": float(values.get('train/ent_coef_loss', float('nan'))),
        }
        self._csv_append("train_log.csv", row)

        if self.use_wandb:
            self.log_count += 1
            wandb.log({
                "train/ep_len_mean": row["ep_len_mean"],
                "train/ep_rew_mean": row["ep_rew_mean"],
                "train/rew_mean": row["rew_mean"],
                "train/timesteps": row["env_steps"],
                "train/ent_coef": row["ent_coef"],
                "train/actor_loss": row["actor_loss"],
                "train/critic_loss": row["critic_loss"],
                "train/ent_coef_loss": row["ent_coef_loss"],
            }, step=self.log_count)
            if completed > 0:
                wandb.log({
                    "train/success_rate": row["success_rate"],
                }, step=self.log_count)
            if self.algorithm == 'dsrl_na':
                wandb.log({
                    "train/noise_critic_loss": row["noise_critic_loss"],
                }, step=self.log_count)

    # ----------------------------------------------------------------- eval
    def evaluate(self, agent, deterministic=False):
        episodes = self.episodes_now()
        if episodes <= 0:
            return
        env = self.eval_env
        with torch.no_grad():
            success, rews = [], []
            rew_total, total_ep = 0, 0
            rew_ep = np.zeros(self.num_eval_env)
            for i in range(episodes):
                obs = env.reset()
                success_i = np.zeros(obs.shape[0])
                r = []
                for _ in range(self.max_steps):
                    if self.algorithm == 'dsrl_sac':
                        action, _ = agent.predict(obs, deterministic=deterministic)
                    elif self.algorithm == 'dsrl_na':
                        action, _ = agent.predict_diffused(obs, deterministic=deterministic)
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    rew_ep += reward
                    rew_total += sum(rew_ep[done])
                    rew_ep[done] = 0
                    total_ep += np.sum(done)
                    success_i[reward > -self.rew_offset] = 1
                    r.append(reward)
                success.append(success_i.mean())
                rews.append(np.mean(np.array(r)))
                print(f'eval episode {i} at timestep {self.total_timesteps}', flush=True)
            success_rate = float(np.mean(success))
            avg_rew = float(rew_total / total_ep) if total_ep > 0 else 0.0

            row = {
                "wall_time": self._now(),
                "env_steps": int(self.total_timesteps),
                "sb3_timesteps": int(getattr(agent, "num_timesteps", 0)),
                "deterministic": int(bool(deterministic)),
                "success_rate": success_rate,
                "avg_reward": avg_rew,
                "episodes": int(episodes * self.num_eval_env),
            }
            self._csv_append("eval_log.csv", row)
            print(
                "[eval] env_steps=%d success_rate=%.3f reward=%.2f%s"
                % (self.total_timesteps, success_rate, avg_rew, " (deterministic)" if deterministic else ""),
                flush=True,
            )

            if self.use_wandb:
                name = 'eval'
                if deterministic:
                    wandb.log({
                        f"{name}/success_rate_deterministic": success_rate,
                        f"{name}/reward_deterministic": avg_rew,
                    }, step=self.log_count)
                else:
                    wandb.log({
                        f"{name}/success_rate": success_rate,
                        f"{name}/reward": avg_rew,
                        f"{name}/timesteps": self.total_timesteps,
                    }, step=self.log_count)


def collect_rollouts(model, env, num_steps, base_policy, cfg):
    obs = env.reset()
    for i in range(num_steps):
        noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
        if cfg.algorithm == 'dsrl_sac':
            noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
            noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
        action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
        next_obs, reward, done, info = env.step(action)
        if cfg.algorithm == 'dsrl_na':
            action_store = action
        elif cfg.algorithm == 'dsrl_sac':
            action_store = noise.detach().cpu().numpy()
        action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
        if cfg.algorithm == 'dsrl_sac':
            action_store = model.policy.scale_action(action_store)
        model.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action_store,
            reward=reward,
            done=done,
            infos=info,
        )
        obs = next_obs
    model.replay_buffer.final_offline_step()


def load_offline_data(model, offline_data_path, n_env):
    # this function should only be applied with dsrl_na
    offline_data = np.load(offline_data_path)
    obs = offline_data['states']
    next_obs = offline_data['states_next']
    actions = offline_data['actions']
    rewards = offline_data['rewards']
    terminals = offline_data['terminals']
    for i in range(int(obs.shape[0] / n_env)):
        model.replay_buffer.add(
            obs=obs[n_env * i:n_env * i + n_env],
            next_obs=next_obs[n_env * i:n_env * i + n_env],
            action=actions[n_env * i:n_env * i + n_env],
            reward=rewards[n_env * i:n_env * i + n_env],
            done=terminals[n_env * i:n_env * i + n_env],
            infos=[{}] * n_env,
        )
    model.replay_buffer.final_offline_step()
