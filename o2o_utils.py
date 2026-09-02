"""Additions for the offline-to-online study.

Kept in a separate module so the upstream files stay close to ajwagen/dsrl and
future upstream changes stay easy to merge.

What lives here:
  * ``DSRLResumable``  - DSRL-NA whose checkpoints are complete enough to resume.
  * ``ResumeCheckpointCallback`` - crash-safe rolling checkpoint (weights +
    replay buffer + counters) written to a single directory.
  * helpers to find and read that checkpoint.
"""

import json
import os
import time

from stable_baselines3 import DSRL
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import recursive_setattr

STATE_FILE = "run_state.json"


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


def resolve_ckpt_dir(cfg):
    """Directory holding the rolling checkpoint for this run."""
    ckpt_dir = str(cfg.get("ckpt_dir", "") or "")
    if not ckpt_dir:
        ckpt_dir = os.path.join(cfg.logdir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def read_run_state(ckpt_dir):
    """Return the latest committed checkpoint state, or None if there is none.

    The state file is the commit marker: it is written last and names the slot
    holding a complete model. A session killed mid-save therefore leaves the
    previous slot intact and still pointed at.
    """
    path = os.path.join(ckpt_dir, STATE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    slot = state.get("slot", "a")
    model_path = os.path.join(ckpt_dir, "model_%s.zip" % slot)
    if not os.path.exists(model_path):
        return None
    state["model_path"] = model_path
    buffer_path = os.path.join(ckpt_dir, "replay_buffer_%s.pkl" % slot)
    state["replay_buffer_path"] = buffer_path if os.path.exists(buffer_path) else None
    return state


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
        verbose=1,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.every_env_steps = int(every_env_steps)
        self.logging_callback = logging_callback
        self.save_replay_buffer = save_replay_buffer
        self.next_save_at = None

    def _env_steps(self):
        return int(self.logging_callback.total_timesteps)

    def _on_training_start(self):
        if self.next_save_at is None:
            self.next_save_at = self._env_steps() + self.every_env_steps

    def _on_step(self):
        if self.every_env_steps > 0 and self._env_steps() >= self.next_save_at:
            self.save()
            self.next_save_at = self._env_steps() + self.every_env_steps
        return True

    def _on_training_end(self):
        self.save()

    def save(self):
        previous = read_run_state(self.save_dir)
        used_slot = (previous or {}).get("slot", "b")
        slot = "b" if used_slot == "a" else "a"

        started = time.time()
        self.model.save(os.path.join(self.save_dir, "model_%s.zip" % slot))
        if self.save_replay_buffer:
            self.model.save_replay_buffer(
                os.path.join(self.save_dir, "replay_buffer_%s.pkl" % slot)
            )

        state = {
            "slot": slot,
            "num_timesteps": int(self.model.num_timesteps),
            "episode_num": int(getattr(self.model, "_episode_num", 0)),
            "n_updates": int(getattr(self.model, "_n_updates", 0)),
            "has_replay_buffer": bool(self.save_replay_buffer),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        state.update(self.logging_callback.state_dict())
        _atomic_write_json(os.path.join(self.save_dir, STATE_FILE), state)

        if self.verbose:
            print(
                "[ckpt] slot %s at %d env steps, %.1fs -> %s"
                % (slot, state.get("env_steps", -1), time.time() - started, self.save_dir),
                flush=True,
            )


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
