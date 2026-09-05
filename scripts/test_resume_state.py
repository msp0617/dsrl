"""Checks for the crash-recovery logic, runnable without torch or mujoco.

    python scripts/test_resume_state.py

The checkpoint bookkeeping only matters when a session dies at a bad moment,
which is exactly when it is never exercised on purpose. Stub modules stand in
for torch and stable-baselines3 so the slot alternation, the fallback to the
older slot and the config fingerprint can be tested on a laptop.
"""

import json
import os
import shutil
import sys
import tempfile
import types
import zipfile


def install_stubs():
    torch = types.ModuleType("torch")
    torch.get_rng_state = lambda: b"cpu-rng"
    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_rng_state_all=lambda: [],
        set_rng_state_all=lambda state: None,
    )
    torch.cuda = cuda
    torch.set_rng_state = lambda state: None
    sys.modules["torch"] = torch

    class BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.model = None

    class DSRL:
        def _excluded_save_params(self):
            return ["policy", "env"]

        def _get_torch_save_params(self):
            return ["policy", "critic_noise"], ["log_ent_coef"]

    class SAC:
        pass

    sb3 = types.ModuleType("stable_baselines3")
    sb3.DSRL = DSRL
    sb3.SAC = SAC
    common = types.ModuleType("stable_baselines3.common")
    callbacks = types.ModuleType("stable_baselines3.common.callbacks")
    callbacks.BaseCallback = BaseCallback
    save_util = types.ModuleType("stable_baselines3.common.save_util")
    save_util.load_from_zip_file = lambda *a, **k: (None, {}, {})
    save_util.recursive_setattr = lambda obj, attr, value: None
    sb3.common = common
    common.callbacks = callbacks
    common.save_util = save_util
    for name, module in [
        ("stable_baselines3", sb3),
        ("stable_baselines3.common", common),
        ("stable_baselines3.common.callbacks", callbacks),
        ("stable_baselines3.common.save_util", save_util),
    ]:
        sys.modules[name] = module


install_stubs()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import o2o_utils  # noqa: E402


class FakeModel:
    """Writes real zip files so zipfile.is_zipfile means something."""

    def __init__(self):
        self.num_timesteps = 0
        self._episode_num = 0
        self._n_updates = 0
        self.gate = None

    def gate_state(self):
        return self.gate.state_dict() if self.gate is not None else None

    def save(self, path):
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("policy.pth", b"weights %d" % self.num_timesteps)

    def save_replay_buffer(self, path):
        with open(path, "wb") as f:
            f.write(b"buffer %d" % self.num_timesteps)


class FakeLoggingCallback:
    def __init__(self):
        self.total_timesteps = 0

    def state_dict(self):
        return {"env_steps": int(self.total_timesteps), "log_count": 3, "next_eval_at": 999}


def make_callback(save_dir, fingerprint=None):
    logging_callback = FakeLoggingCallback()
    callback = o2o_utils.ResumeCheckpointCallback(
        save_dir=save_dir,
        every_env_steps=1000,
        logging_callback=logging_callback,
        save_replay_buffer=True,
        fingerprint=fingerprint or {"layer_size": 2048, "n_envs": 4},
        verbose=0,
    )
    callback.model = FakeModel()
    return callback, logging_callback


def test_empty_dir_has_no_candidates():
    d = tempfile.mkdtemp()
    assert o2o_utils.read_run_states(d) == []


def test_slots_alternate_and_newest_comes_first():
    d = tempfile.mkdtemp()
    cb, log = make_callback(d)

    log.total_timesteps, cb.model.num_timesteps = 1000, 250
    cb.save()
    log.total_timesteps, cb.model.num_timesteps = 2000, 500
    cb.save()

    assert sorted(os.listdir(d)) == sorted([
        "run_state.json",
        "model_a.zip", "model_b.zip",
        "replay_buffer_a.pkl", "replay_buffer_b.pkl",
        "rng_a.pkl", "rng_b.pkl",
    ]), os.listdir(d)

    candidates = o2o_utils.read_run_states(d)
    assert [c["slot"] for c in candidates] == ["b", "a"], "newest slot first"
    assert candidates[0]["env_steps"] == 2000 and candidates[1]["env_steps"] == 1000
    assert candidates[0]["num_timesteps"] == 500
    assert candidates[0]["log_count"] == 3, "the logging counters travel with the slot"
    assert candidates[0]["replay_buffer_path"].endswith("replay_buffer_b.pkl")
    assert candidates[0]["rng_path"].endswith("rng_b.pkl")


def test_truncated_newest_slot_falls_back_to_the_older_one():
    d = tempfile.mkdtemp()
    cb, log = make_callback(d)
    log.total_timesteps, cb.model.num_timesteps = 1000, 250
    cb.save()
    log.total_timesteps, cb.model.num_timesteps = 2000, 500
    cb.save()

    # a save killed by the session dying: the archive never finished
    with open(os.path.join(d, "model_b.zip"), "wb") as f:
        f.write(b"PK\x03\x04 half a file")

    candidates = o2o_utils.read_run_states(d)
    assert [c["slot"] for c in candidates] == ["a"], "the good older slot survives"
    assert candidates[0]["env_steps"] == 1000


def test_missing_archive_does_not_make_the_next_save_overwrite_the_good_slot():
    d = tempfile.mkdtemp()
    cb, log = make_callback(d)
    log.total_timesteps = 1000
    cb.save()          # -> slot a
    log.total_timesteps = 2000
    cb.save()          # -> slot b, state names b
    os.remove(os.path.join(d, "model_b.zip"))

    log.total_timesteps = 3000
    cb.save()
    state = json.load(open(os.path.join(d, "run_state.json")))
    assert state["slot"] == "a", "alternation follows the state file, not the files present"

    # and slot a is now the fresh one
    candidates = o2o_utils.read_run_states(d)
    assert candidates[0]["slot"] == "a" and candidates[0]["env_steps"] == 3000


class Cfg(dict):
    """Stands in for a DictConfig: attribute access plus dict.get with a default."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def make_cfg(layer_size=2048, n_envs=4):
    return Cfg(
        algorithm="dsrl_na", env_name="can", obs_dim=23, action_dim=7, act_steps=4,
        env=Cfg(n_envs=n_envs),
        train=Cfg(layer_size=layer_size, num_layers=3, n_critics=2, use_layer_norm=True,
                  buffer_size=200000, init_rollout_steps=1501, total_env_steps=300000),
        load_offline_data=False,
    )


def test_fingerprint_mismatch_raises():
    d = tempfile.mkdtemp()
    fingerprint = o2o_utils.config_fingerprint(make_cfg())
    cb, log = make_callback(d, fingerprint=fingerprint)
    log.total_timesteps = 1000
    cb.save()
    state = o2o_utils.read_run_states(d)[0]

    o2o_utils.check_fingerprint(state, make_cfg())

    try:
        o2o_utils.check_fingerprint(state, make_cfg(layer_size=256))
    except RuntimeError as exc:
        assert "layer_size" in str(exc)
    else:
        raise AssertionError("a different network shape must refuse to resume")

    try:
        o2o_utils.check_fingerprint(state, make_cfg(n_envs=1))
    except RuntimeError as exc:
        assert "n_envs" in str(exc), "a different env count reshapes the replay buffer"
    else:
        raise AssertionError("a different n_envs must refuse to resume")


def test_buffer_capacity_check():
    o2o_utils.check_buffer_capacity(make_cfg(), 200000)

    try:
        o2o_utils.check_buffer_capacity(make_cfg(), 40000)
    except RuntimeError as exc:
        assert "buffer_size" in str(exc)
    else:
        raise AssertionError("a buffer that would fill must be refused up front")

    # n_envs=1 quarters the slots for the same buffer_size
    try:
        o2o_utils.check_buffer_capacity(make_cfg(n_envs=1), 60000)
    except RuntimeError:
        pass
    else:
        raise AssertionError("fewer envs means more slots are needed, not fewer")


def test_pretrain_meta_checks():
    cfg = make_cfg()
    meta = {"method": "iql", "network": o2o_utils.network_fingerprint(cfg)}
    o2o_utils.check_pretrain_meta(meta, cfg, "iql")

    try:
        o2o_utils.check_pretrain_meta(meta, cfg, "warmup")
    except RuntimeError as exc:
        assert "iql" in str(exc) and "warmup" in str(exc)
    else:
        raise AssertionError("iql weights must not be loaded into a warmup run")

    try:
        o2o_utils.check_pretrain_meta(meta, make_cfg(layer_size=512), "iql")
    except RuntimeError as exc:
        assert "layer_size" in str(exc)
    else:
        raise AssertionError("weights for another network shape must be refused")

    # n_envs is a run property, not a network property: pretraining used one env
    o2o_utils.check_pretrain_meta(meta, make_cfg(n_envs=1), "iql")


def test_run_fingerprint_includes_variant():
    baseline = o2o_utils.config_fingerprint(make_cfg())
    assert baseline["variant"] == "baseline"
    cfg = make_cfg()
    cfg["variant"] = "iql"
    assert o2o_utils.config_fingerprint(cfg)["variant"] == "iql"


def test_gate_state_is_written_to_the_run_state():
    d = tempfile.mkdtemp()
    cb, log = make_callback(d)
    cb.model.gate = o2o_utils.GateController(signal="ratio", tau=1.0, K=2)
    cb.model.gate.update(0.5)
    cb.save()
    state = o2o_utils.read_run_states(d)[0]
    assert state["gate"] == {"calls": 1, "streak": 1, "open": False, "open_call": -1}
    cb.model.gate = None
    cb.save()
    assert "gate" not in o2o_utils.read_run_states(d)[0]


def test_unreadable_state_file_is_not_a_crash():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "run_state.json"), "w") as f:
        f.write("{ truncated")
    assert o2o_utils.read_run_states(d) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("%d checks passed" % len(tests))
