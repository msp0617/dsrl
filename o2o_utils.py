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

import numpy as np
import torch as th

from stable_baselines3 import DSRL
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr

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

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["diffusion_policy"]

    def _get_torch_save_params(self):
        state_dicts, pytorch_variables = super()._get_torch_save_params()
        if "critic_noise.optimizer" not in state_dicts:
            state_dicts = state_dicts + ["critic_noise.optimizer"]
        return state_dicts, pytorch_variables


# --------------------------------------------------------------- config check

def config_fingerprint(cfg):
    """The settings a checkpoint cannot be reinterpreted under.

    Network shape decides whether the weights even load; the action chunk,
    number of envs and buffer size decide what the saved replay buffer means.
    """
    return {
        "algorithm": str(cfg.algorithm),
        "env_name": str(cfg.env_name),
        "obs_dim": int(cfg.obs_dim),
        "action_dim": int(cfg.action_dim),
        "act_steps": int(cfg.act_steps),
        "n_envs": int(cfg.env.n_envs),
        "layer_size": int(cfg.train.layer_size),
        "num_layers": int(cfg.train.num_layers),
        "n_critics": int(cfg.train.n_critics),
        "use_layer_norm": bool(cfg.train.use_layer_norm),
        "buffer_size": int(cfg.train.get("buffer_size", 0)),
    }


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
