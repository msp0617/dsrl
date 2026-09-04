"""Additions for the offline-to-online study.

Kept in a separate module so the upstream files stay close to ajwagen/dsrl and
future upstream changes stay easy to merge.

What lives here:
  * ``DSRLResumable``  - DSRL-NA whose checkpoints are complete enough to resume.
  * ``ResumeCheckpointCallback`` - crash-safe rolling checkpoint (weights,
    replay buffer, random state and counters) written to a single directory.
  * helpers to find, validate and load that checkpoint.
"""

import json
import math
import os
import pickle
import random
import time
import zipfile

import gymnasium
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3 import DSRL, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr

try:
    from stable_baselines3.common.type_aliases import ReplayBufferSamples
except ImportError:  # torch-free tests stub stable_baselines3
    from collections import namedtuple

    ReplayBufferSamples = namedtuple(
        "ReplayBufferSamples", "observations actions next_observations dones rewards"
    )

VARIANTS = ("baseline", "warmup", "iql")
MIX_MODES = ("none", "prefill", "fixed", "linear")

STATE_FILE = "run_state.json"
SLOTS = ("a", "b")


class DSRLResumable(DSRL):
    """DSRL-NA with a checkpoint that can actually be resumed from.

    Two upstream gaps are closed:
      * the noise critic's optimizer state was not saved, so a resumed run
        restarted Adam's moments for the distilled critic;
      * the frozen diffusion policy was pickled into every checkpoint, which is
        both wasteful and fragile across sessions.
    """

    # Offline replay mix (offline_mix.mode). Class defaults so that a model
    # built without the feature behaves exactly like upstream.
    offline_buf = None  # OfflineBuffer when mode is fixed/linear
    offline_p = 0.0  # share of each batch drawn from offline_buf
    offline_prefill_slots = 0  # buffer slots D_off occupies when mode is prefill

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["diffusion_policy", "offline_buf"]

    def _get_torch_save_params(self):
        state_dicts, pytorch_variables = super()._get_torch_save_params()
        if "critic_noise.optimizer" not in state_dicts:
            state_dicts = state_dicts + ["critic_noise.optimizer"]
        return state_dicts, pytorch_variables

    def _sample_batch(self, batch_size):
        if self.offline_buf is None:
            return self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        return mixed_sample(
            self.replay_buffer, self.offline_buf, batch_size, self.offline_p, env=self._vec_normalize_env
        )

    def current_offline_share(self):
        """Share of a batch that comes from the demonstrations right now.

        Explicit for fixed/linear. For prefill it is the share D_off holds in
        the uniformly sampled buffer, which decays as online data arrives.
        """
        if self.offline_buf is not None:
            return float(self.offline_p)
        if self.offline_prefill_slots:
            return float(self.offline_prefill_slots) / max(float(self.replay_buffer.size()), 1.0)
        return 0.0

    def train(self, gradient_steps, batch_size=64):
        """Upstream DSRL.train with two additions.

        Batches come from ``_sample_batch`` (offline mix), and the last actor
        step records the diagnostics that LoggingCallback writes to
        train_log.csv. With offline_mix.mode=none the computation, and the
        random stream, are identical to upstream.
        """
        from torch.nn import functional as F
        from stable_baselines3.common.utils import polyak_update

        self.policy.set_training_mode(True)
        self.critic_noise.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.critic_noise.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, noise_critic_losses = [], [], []
        diag = {}

        if self.actor_gradient_steps < 0:
            actor_gradient_idx = np.linspace(0, gradient_steps - 1, gradient_steps, dtype=int)
        else:
            actor_gradient_idx = np.linspace(
                int(gradient_steps / self.actor_gradient_steps) - 1, gradient_steps - 1,
                self.actor_gradient_steps, dtype=int,
            )

        for gradient_step in range(gradient_steps):
            replay_data = self._sample_batch(batch_size)
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_actions = th.tensor(self.policy.unscale_action(next_actions.cpu().numpy())).to(self.device)
                next_actions = self.diffusion_policy(
                    replay_data.next_observations,
                    next_actions.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim),
                    return_numpy=False,
                )
                next_actions = next_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                if self.critic_backup_combine_type == "min":
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                elif self.critic_backup_combine_type == "mean":
                    next_q_values = th.mean(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if gradient_step in actor_gradient_idx:
                q_values_pi = th.cat(self.critic_noise(replay_data.observations, actions_pi), dim=1)
                if self.critic_backup_combine_type == "min":
                    min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                elif self.critic_backup_combine_type == "mean":
                    min_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                if gradient_step == gradient_steps - 1:
                    # Diagnostics for the dip: where pi_W sits relative to the
                    # N(0, I) prior pi_dp was trained with. w is tanh-bounded to
                    # [-1, 1], so the pre-tanh mean and log-std are the direct
                    # measure, and |w| > 0.9 is saturation, not a large noise.
                    with th.no_grad():
                        mean_actions, log_std, _ = self.actor.get_action_dist_params(replay_data.observations)
                        diag = {
                            "w_absmean": actions_pi.abs().mean().item(),
                            "w_std": actions_pi.std(dim=0).mean().item(),
                            "w_frac_sat": (actions_pi.abs() > 0.9).float().mean().item(),
                            "mu_absmean": mean_actions.abs().mean().item(),
                            "log_std_mean": log_std.mean().item(),
                            "logp_mean": log_prob.mean().item(),
                            "qw_mean": min_qf_pi.mean().item(),
                        }

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        for gradient_step in range(self.noise_critic_grad_steps):
            replay_data = self._sample_batch(batch_size)
            critic_distill_loss = self.update_noise_critic(replay_data)
            noise_critic_losses.append(critic_distill_loss.item())
            self.critic_noise.optimizer.zero_grad()
            critic_distill_loss.backward()
            self.critic_noise.optimizer.step()

        self.critic_noise.set_training_mode(False)
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/noise_critic_loss", np.mean(noise_critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("train/offline_p", self.current_offline_share())
        for key, value in diag.items():
            self.logger.record("train/" + key, value)


# --------------------------------------------------------------- config check

def network_fingerprint(cfg):
    """What decides whether saved weights fit this run's networks at all."""
    return {
        "algorithm": str(cfg.algorithm),
        "env_name": str(cfg.env_name),
        "obs_dim": int(cfg.obs_dim),
        "action_dim": int(cfg.action_dim),
        "act_steps": int(cfg.act_steps),
        "layer_size": int(cfg.train.layer_size),
        "num_layers": int(cfg.train.num_layers),
        "n_critics": int(cfg.train.n_critics),
        "use_layer_norm": bool(cfg.train.use_layer_norm),
    }


def config_fingerprint(cfg):
    """The settings a run checkpoint cannot be reinterpreted under.

    On top of the network shape: the number of envs and the buffer size decide
    what the saved replay buffer means, and the variant decides which
    experiment the run belongs to.
    """
    fingerprint = network_fingerprint(cfg)
    fingerprint.update({
        "n_envs": int(cfg.env.n_envs),
        "buffer_size": int(cfg.train.get("buffer_size", 0)),
        "variant": str(cfg.get("variant", "baseline") or "baseline"),
        # auto (-1) vs a fixed alpha is a different experiment; the budget
        # (total_env_steps) is deliberately left out so a run can be extended.
        "ent_coef": float(cfg.train.get("ent_coef", -1)),
        "target_ent": float(cfg.train.get("target_ent", -1)),
    })
    pretrain = cfg.get("pretrain", None)
    if pretrain is not None:
        fingerprint["load_actor"] = bool(pretrain.get("load_actor", True))
        fingerprint["load_ent_coef"] = bool(pretrain.get("load_ent_coef", False))
    mix = cfg.get("offline_mix", None)
    if mix is not None:
        mode = str(mix.get("mode", "none") or "none")
        if mode in ("fixed", "linear"):
            fingerprint["offline_mix"] = "%s:%s:%s:%s" % (
                mode, mix.get("p0"), mix.get("p1"), mix.get("until_env"))
        else:
            fingerprint["offline_mix"] = mode
    return fingerprint


def check_fingerprint(state, cfg):
    """Raise if the checkpoint was produced by a differently shaped run."""
    saved = state.get("config") or {}
    if not saved:
        return
    current = config_fingerprint(cfg)
    mismatch = {k: (saved[k], v) for k, v in current.items() if k in saved and saved[k] != v}
    if mismatch:
        lines = ["  %s: checkpoint %r, config %r" % (k, a, b) for k, (a, b) in sorted(mismatch.items())]
        raise RuntimeError(
            "The checkpoint was written by a run with different settings, so resuming it "
            "would not continue the same experiment:\n%s\n"
            "Use a different exp_id, pass resume=False, or restore the original settings."
            % "\n".join(lines)
        )


def check_buffer_capacity(cfg, buffer_size):
    """Refuse a run that would fill the replay buffer.

    The fork samples with an explicit probability vector of length ``pos`` while
    drawing indices over the whole buffer, so a buffer that fills raises deep
    inside training, hours in. One slot holds one action chunk per env.
    """
    n_envs = int(cfg.env.n_envs)
    act_steps = int(cfg.act_steps)
    slots = max(buffer_size // n_envs, 1)

    needed = int(cfg.train.init_rollout_steps)
    if cfg.load_offline_data:
        path = str(cfg.offline_data_path)
        if os.path.exists(path):
            with np.load(path) as data:
                needed += int(data["states"].shape[0]) // n_envs

    total_env_steps = int(cfg.train.get("total_env_steps", 0))
    if total_env_steps <= 0:
        return  # unbounded run, as upstream
    needed += int(math.ceil(total_env_steps / (act_steps * n_envs)))

    if needed >= slots:
        raise RuntimeError(
            "train.buffer_size=%d gives %d buffer slots at n_envs=%d, but this run needs %d. "
            "Raise train.buffer_size to at least %d."
            % (buffer_size, slots, n_envs, needed, (needed + 1) * n_envs * 2)
        )
    if needed > 0.9 * slots:
        print(
            "[buffer] warning: %d of %d slots will be used, little headroom" % (needed, slots),
            flush=True,
        )


# ------------------------------------------------------------ agent building

def build_agent(cfg, env, base_policy, buffer_size, replay_buffer_kwargs=None,
                tensorboard_log=None, verbose=1):
    """The one place the networks are shaped.

    Both the online run and the offline pre-training build their agent here, so
    weights saved by one always fit the other.
    """
    post_linear_modules = [th.nn.LayerNorm] if cfg.train.use_layer_norm else None
    net_arch = [int(cfg.train.layer_size)] * int(cfg.train.num_layers)
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=th.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )
    common = dict(
        learning_rate=cfg.train.actor_lr,
        buffer_size=int(buffer_size),
        learning_starts=1,
        batch_size=cfg.train.batch_size,
        tau=cfg.train.tau,
        gamma=cfg.train.discount,
        train_freq=cfg.train.train_freq,
        gradient_steps=cfg.train.utd,
        action_noise=None,
        replay_buffer_kwargs=replay_buffer_kwargs,
        optimize_memory_usage=False,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
        target_update_interval=1,
        target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
    )
    if cfg.algorithm == "dsrl_sac":
        return SAC("MlpPolicy", env, **common)
    if cfg.algorithm == "dsrl_na":
        return DSRLResumable(
            "MlpPolicy",
            env,
            diffusion_policy=base_policy,
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
            **common,
        )
    raise ValueError("unknown algorithm %r" % cfg.algorithm)


class SpacesOnlyEnv(gymnasium.Env):
    """Carries the task's observation and action spaces and nothing else.

    Stable-Baselines3 needs an env to shape its networks and replay buffer.
    Offline pre-training never steps one, so this stands in for the robosuite
    env and lets the offline stage run without mujoco.
    """

    metadata = {}

    def __init__(self, obs_dim, action_dim):
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32), high=np.ones(obs_dim, dtype=np.float32), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.ones(action_dim, dtype=np.float32), high=np.ones(action_dim, dtype=np.float32), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError("SpacesOnlyEnv cannot be stepped; it only exists to build the agent offline")


# --------------------------------------------------------- pretrained weights

def check_pretrain_meta(meta, cfg, variant):
    """Refuse weights that were made for another variant or another network."""
    method = meta.get("method")
    if method != variant:
        raise RuntimeError(
            "pretrain_path holds %r weights but this run is variant=%s" % (method, variant)
        )
    saved = meta.get("network") or {}
    current = network_fingerprint(cfg)
    mismatch = {k: (saved[k], v) for k, v in current.items() if k in saved and saved[k] != v}
    if mismatch:
        lines = ["  %s: pretrained %r, config %r" % (k, a, b) for k, (a, b) in sorted(mismatch.items())]
        raise RuntimeError("The pretrained weights do not fit this run's networks:\n%s" % "\n".join(lines))


def load_pretrained_weights(model, path, cfg, variant, load_ent_coef=False, load_actor=True):
    """Copy offline-pretrained networks into a freshly built agent.

    The critics are always loaded, the actor when the offline stage trained it
    and ``load_actor`` is set. The entropy coefficient is loaded only when
    ``load_ent_coef`` is set: the
    warm-up stage anneals it, and starting online with an already small alpha
    is an advantage that has nothing to do with the critic. The optimizers
    start fresh, as they would for any newly initialised network.
    """
    payload = th.load(path, map_location=model.device, weights_only=False)
    meta = payload.get("meta") or {}
    check_pretrain_meta(meta, cfg, variant)

    loaded = []
    for name in ("critic", "critic_target", "critic_noise", "actor"):
        if name == "actor" and not load_actor:
            continue
        if name in payload:
            getattr(model, name).load_state_dict(payload[name])
            loaded.append(name)
    if "critic" in loaded and "critic_target" not in loaded:
        model.critic_target.load_state_dict(model.critic.state_dict())
        loaded.append("critic_target<-critic")
    if load_ent_coef and "log_ent_coef" in payload and getattr(model, "log_ent_coef", None) is not None:
        model.log_ent_coef.data.copy_(payload["log_ent_coef"].to(model.device))
        loaded.append("log_ent_coef")
    return meta, loaded


# --------------------------------------------------------------- offline mix

def ratio_at(mix, env_steps_since_train_start):
    """Share of each batch to draw from the demonstrations at this point.

    ``mix`` is cfg.offline_mix. t is counted from the start of training, i.e.
    after the initial rollout, so that the schedule lines up with the dip.
    none and prefill return 0: prefill puts D_off into the online buffer
    instead, and its share then decays on its own as online data arrives.
    """
    mode = str(mix.get("mode", "none") or "none") if mix is not None else "none"
    if mode not in MIX_MODES:
        raise ValueError("offline_mix.mode must be one of %s, got %r" % (", ".join(MIX_MODES), mode))
    if mode in ("none", "prefill"):
        return 0.0
    p0 = float(mix.get("p0", 0.0))
    if mode == "fixed":
        return min(max(p0, 0.0), 1.0)
    p1 = float(mix.get("p1", p0))
    until = float(mix.get("until_env", 0) or 0)
    t = max(float(env_steps_since_train_start), 0.0)
    frac = 1.0 if until <= 0 else min(t / until, 1.0)
    return min(max(p0 + (p1 - p0) * frac, 0.0), 1.0)


def mixed_sample(online_buf, offline_buf, batch_size, p, env=None):
    """A batch with round(p * batch_size) rows from the demonstrations.

    Field by field concatenation; order inside a batch does not matter to
    any of the losses, so there is no shuffle.
    """
    n_off = int(round(float(p) * batch_size))
    n_off = min(max(n_off, 0), batch_size)
    n_on = batch_size - n_off
    if n_off == 0:
        return online_buf.sample(batch_size, env=env)
    if n_on == 0:
        return offline_buf.sample(batch_size)
    on = online_buf.sample(n_on, env=env)
    off = offline_buf.sample(n_off)
    return ReplayBufferSamples(*[th.cat([a, b], dim=0) for a, b in zip(on, off)])


class OfflineBuffer:
    """The chunked demonstrations as tensors, sampled uniformly.

    Static and small (61,856 chunks, about 20 MB as float32), so it lives on
    the device whole. Fields, dtypes and shapes match what the online
    ReplayBuffer returns, so the two can be concatenated.
    """

    def __init__(self, npz_path, device):
        with np.load(npz_path) as data:
            states = np.asarray(data["states"], dtype=np.float32)
            self.n = int(states.shape[0])
            self.observations = th.as_tensor(states, device=device)
            self.actions = th.as_tensor(np.asarray(data["actions"], dtype=np.float32), device=device)
            self.next_observations = th.as_tensor(np.asarray(data["states_next"], dtype=np.float32), device=device)
            self.dones = th.as_tensor(np.asarray(data["terminals"], dtype=np.float32).reshape(-1, 1), device=device)
            self.rewards = th.as_tensor(np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1), device=device)
        self.device = device

    def sample(self, n):
        idx = th.randint(0, self.n, (int(n),), device=self.device)
        return ReplayBufferSamples(
            self.observations[idx], self.actions[idx], self.next_observations[idx], self.dones[idx], self.rewards[idx]
        )


class OfflineRatioCallback(BaseCallback):
    """Keeps model.offline_p on the schedule, in environment steps.

    p is a pure function of the env-step count, so nothing needs saving for a
    resume; it is also set on training start so the first update after a
    resume does not run with the class default of 0.
    """

    def __init__(self, mix, logging_callback, train_start_env_steps, verbose=0):
        super().__init__(verbose)
        self.mix = mix
        self.logging_callback = logging_callback
        self.train_start_env_steps = int(train_start_env_steps)

    def _update(self):
        t = int(self.logging_callback.total_timesteps) - self.train_start_env_steps
        self.model.offline_p = ratio_at(self.mix, t)

    def _on_training_start(self):
        self._update()

    def _on_step(self):
        self._update()
        return True


# ------------------------------------------------------------- reading state

def _read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def resolve_ckpt_dir(cfg):
    """Directory holding the rolling checkpoint for this run."""
    ckpt_dir = str(cfg.get("ckpt_dir", "") or "")
    if not ckpt_dir:
        ckpt_dir = os.path.join(cfg.logdir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def read_run_states(ckpt_dir):
    """Candidate checkpoints, best first.

    The state file names the slot written last, but on a Drive mount a save can
    be interrupted after the small state file lands and before the large archive
    is durable. Both slots are therefore offered, newest first, so the caller can
    fall back to the older one when the newest fails to load.
    """
    raw = _read_json(os.path.join(ckpt_dir, STATE_FILE))
    if raw is None:
        return []

    slots = raw.get("slots")
    if not slots:  # state written by an earlier version
        slots = {raw.get("slot", "a"): raw}

    order = sorted(slots, key=lambda s: slots[s].get("env_steps", -1), reverse=True)
    latest = raw.get("slot")
    if latest in order:
        order.remove(latest)
        order.insert(0, latest)

    candidates = []
    for slot in order:
        model_path = os.path.join(ckpt_dir, "model_%s.zip" % slot)
        if not (os.path.exists(model_path) and zipfile.is_zipfile(model_path)):
            continue
        entry = dict(slots[slot])
        entry["slot"] = slot
        entry["model_path"] = model_path
        for key, name in (("replay_buffer_path", "replay_buffer_%s.pkl"), ("rng_path", "rng_%s.pkl")):
            path = os.path.join(ckpt_dir, name % slot)
            entry[key] = path if os.path.exists(path) else None
        entry["config"] = raw.get("config") or {}
        candidates.append(entry)
    return candidates


def restore_agent(model, path, device="auto"):
    """Load weights, optimizer state and loose tensors into an existing agent.

    ``set_parameters`` alone silently skips the "pytorch variables", which for
    SAC-family agents holds ``log_ent_coef``. Restoring the optimizer moments
    without the coefficient itself would resume with a temperature reset to its
    initial value, so both are restored here.
    """
    _, params, pytorch_variables = load_from_zip_file(path, device=device, load_data=False)
    model.set_parameters(params, exact_match=True, device=device)
    if pytorch_variables is not None:
        for name, value in pytorch_variables.items():
            if value is None:
                continue
            recursive_setattr(model, "%s.data" % name, value.data)


# -------------------------------------------------------------- random state

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": th.get_rng_state(),
    }
    if th.cuda.is_available():
        state["torch_cuda"] = th.cuda.get_rng_state_all()
    return state


def restore_rng_state(path):
    """Continue the random stream instead of replaying it from the seed.

    Without this every resumed session repeats the same exploration noise and
    the same replay sampling order from the top, so a run's trajectory would
    depend on how often Colab killed it.
    """
    with open(path, "rb") as f:
        state = pickle.load(f)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    th.set_rng_state(state["torch"])
    if "torch_cuda" in state and th.cuda.is_available():
        try:
            th.cuda.set_rng_state_all(state["torch_cuda"])
        except (RuntimeError, ValueError):
            pass  # a different GPU count: the CPU streams still continue


# ------------------------------------------------------------------ callback

def _atomic_write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


class ResumeCheckpointCallback(BaseCallback):
    """Save everything needed to continue this run in a new Colab session.

    Alternates between two slots so an interrupted save never destroys the last
    good checkpoint. Saving is scheduled on environment steps (not on callback
    calls) so the period means the same thing regardless of ``n_envs``.
    """

    def __init__(
        self,
        save_dir,
        every_env_steps,
        logging_callback,
        save_replay_buffer=True,
        fingerprint=None,
        verbose=1,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.every_env_steps = int(every_env_steps)
        self.logging_callback = logging_callback
        self.save_replay_buffer = save_replay_buffer
        self.fingerprint = fingerprint or {}
        self.next_save_at = 0

    def _env_steps(self):
        return int(self.logging_callback.total_timesteps)

    def _on_training_start(self):
        self.next_save_at = self._env_steps() + self.every_env_steps

    def _on_step(self):
        if self.every_env_steps > 0 and self._env_steps() >= self.next_save_at:
            self.save()
            self.next_save_at = self._env_steps() + self.every_env_steps
        return True

    def _on_training_end(self):
        self.save()

    def save(self):
        state_path = os.path.join(self.save_dir, STATE_FILE)
        raw = _read_json(state_path) or {}
        # Alternate on what the state file says, not on which files exist, so a
        # missing archive can never make us overwrite the surviving slot.
        slot = "b" if raw.get("slot") == "a" else "a"

        started = time.time()
        written = []
        model_path = os.path.join(self.save_dir, "model_%s.zip" % slot)
        self.model.save(model_path)
        written.append(model_path)

        if self.save_replay_buffer:
            buffer_path = os.path.join(self.save_dir, "replay_buffer_%s.pkl" % slot)
            self.model.save_replay_buffer(buffer_path)
            written.append(buffer_path)

        rng_path = os.path.join(self.save_dir, "rng_%s.pkl" % slot)
        with open(rng_path, "wb") as f:
            pickle.dump(capture_rng_state(), f)
        written.append(rng_path)

        entry = {
            "num_timesteps": int(self.model.num_timesteps),
            "episode_num": int(getattr(self.model, "_episode_num", 0)),
            "n_updates": int(getattr(self.model, "_n_updates", 0)),
            "has_replay_buffer": bool(self.save_replay_buffer),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        entry.update(self.logging_callback.state_dict())

        slots = dict(raw.get("slots") or {})
        slots[slot] = entry
        payload = dict(entry)
        payload["slot"] = slot
        payload["slots"] = slots
        payload["config"] = self.fingerprint
        _atomic_write_json(state_path, payload)

        if self.verbose:
            megabytes = sum(os.path.getsize(p) for p in written) / 1e6
            print(
                "[ckpt] slot %s at %d env steps, %.0f MB in %.1fs -> %s"
                % (slot, entry.get("env_steps", -1), megabytes, time.time() - started, self.save_dir),
                flush=True,
            )
