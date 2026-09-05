"""Checks for the offline replay mix, runnable without torch or mujoco.

    python scripts/test_offline_mix.py

The ratio schedule and the batch mixing are the parts that fail silently: a
wrong p or a half-applied mix shows up as a slightly different learning curve
and nothing else. Stub modules stand in for torch (numpy arrays play tensors)
and stable-baselines3 so they can be tested on a laptop.
"""

import os
import sys
import types

import numpy as np


def install_stubs():
    torch = types.ModuleType("torch")
    torch.cat = lambda parts, dim=0: np.concatenate(parts, axis=dim)
    torch.get_rng_state = lambda: b"cpu-rng"
    torch.set_rng_state = lambda state: None
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, get_rng_state_all=lambda: [], set_rng_state_all=lambda state: None
    )
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

    sb3 = types.ModuleType("stable_baselines3")
    sb3.DSRL = DSRL
    sb3.SAC = type("SAC", (), {})
    common = types.ModuleType("stable_baselines3.common")
    callbacks = types.ModuleType("stable_baselines3.common.callbacks")
    callbacks.BaseCallback = BaseCallback
    save_util = types.ModuleType("stable_baselines3.common.save_util")
    save_util.load_from_zip_file = lambda *a, **k: (None, {}, {})
    save_util.recursive_setattr = lambda obj, attr, value: None
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

OBS, ACT = 23, 28


class FakeBuffer:
    """Rows are filled with `marker` so a mixed batch can be told apart."""

    def __init__(self, marker):
        self.marker = float(marker)
        self.calls = []

    def _rows(self, n):
        m = self.marker
        return o2o_utils.ReplayBufferSamples(
            np.full((n, OBS), m, np.float32), np.full((n, ACT), m, np.float32),
            np.full((n, OBS), m, np.float32), np.full((n, 1), m, np.float32), np.full((n, 1), m, np.float32),
        )

    def sample(self, n, env=None):
        self.calls.append(int(n))
        return self._rows(int(n))


class OfflineFake(FakeBuffer):
    def sample(self, n):  # the offline buffer takes no env argument
        return FakeBuffer.sample(self, n)


def linear(p0=0.8, p1=0.1, until=100000):
    return {"mode": "linear", "p0": p0, "p1": p1, "until_env": until}


def test_ratio_none_and_prefill_are_zero():
    assert o2o_utils.ratio_at({"mode": "none", "p0": 0.9}, 5000) == 0.0
    assert o2o_utils.ratio_at({"mode": "prefill", "p0": 0.9}, 5000) == 0.0
    assert o2o_utils.ratio_at(None, 5000) == 0.0


def test_ratio_fixed_is_p0_clipped():
    assert o2o_utils.ratio_at({"mode": "fixed", "p0": 0.5}, 0) == 0.5
    assert o2o_utils.ratio_at({"mode": "fixed", "p0": 0.5}, 10**9) == 0.5
    assert o2o_utils.ratio_at({"mode": "fixed", "p0": 1.7}, 0) == 1.0


def test_ratio_linear_endpoints_and_midpoint():
    assert abs(o2o_utils.ratio_at(linear(), 0) - 0.8) < 1e-9
    assert abs(o2o_utils.ratio_at(linear(), 50000) - 0.45) < 1e-9
    assert abs(o2o_utils.ratio_at(linear(), 100000) - 0.1) < 1e-9
    assert abs(o2o_utils.ratio_at(linear(), 200000) - 0.1) < 1e-9
    assert abs(o2o_utils.ratio_at(linear(), -3000) - 0.8) < 1e-9  # before training starts


def test_ratio_linear_until_zero_means_p1_at_once():
    assert abs(o2o_utils.ratio_at(linear(until=0), 0) - 0.1) < 1e-9


def test_ratio_rejects_unknown_mode():
    try:
        o2o_utils.ratio_at({"mode": "cosine"}, 0)
    except ValueError:
        return
    raise AssertionError("unknown mode accepted")


def test_mixed_sample_sizes_add_up():
    on, off = FakeBuffer(1), OfflineFake(2)
    batch = o2o_utils.mixed_sample(on, off, 256, 0.3)
    assert on.calls == [179] and off.calls == [77], (on.calls, off.calls)
    for field in batch:
        assert field.shape[0] == 256, field.shape
    assert batch.observations.shape == (256, OBS)
    assert batch.actions.shape == (256, ACT)
    assert batch.dones.shape == (256, 1) and batch.rewards.shape == (256, 1)
    # the first n_on rows are online, the rest offline
    assert np.all(batch.observations[:179] == 1) and np.all(batch.observations[179:] == 2)


def test_mixed_sample_p_zero_never_touches_offline():
    on, off = FakeBuffer(1), OfflineFake(2)
    batch = o2o_utils.mixed_sample(on, off, 64, 0.0)
    assert on.calls == [64] and off.calls == []
    assert np.all(batch.rewards == 1)


def test_mixed_sample_p_one_never_touches_online():
    on, off = FakeBuffer(1), OfflineFake(2)
    batch = o2o_utils.mixed_sample(on, off, 64, 1.0)
    assert on.calls == [] and off.calls == [64]
    assert np.all(batch.rewards == 2)


def test_mixed_sample_tiny_p_rounds_to_zero_offline_rows():
    on, off = FakeBuffer(1), OfflineFake(2)
    o2o_utils.mixed_sample(on, off, 64, 0.001)
    assert off.calls == [] and on.calls == [64]


def test_callback_follows_env_steps_from_training_start():
    logging = types.SimpleNamespace(total_timesteps=24016)
    cb = o2o_utils.OfflineRatioCallback(linear(), logging, train_start_env_steps=24016)
    cb.model = types.SimpleNamespace(offline_p=0.0)
    cb._on_training_start()
    assert abs(cb.model.offline_p - 0.8) < 1e-9
    logging.total_timesteps = 24016 + 50000
    cb._on_step()
    assert abs(cb.model.offline_p - 0.45) < 1e-9
    logging.total_timesteps = 24016 + 300000
    cb._on_step()
    assert abs(cb.model.offline_p - 0.1) < 1e-9


def test_prefill_share_decays_with_buffer_size():
    model = o2o_utils.DSRLResumable.__new__(o2o_utils.DSRLResumable)
    model.offline_prefill_slots = 15464
    model.replay_buffer = types.SimpleNamespace(size=lambda: 15464 + 1501)
    assert abs(model.current_offline_share() - 15464 / 16965) < 1e-9
    model.replay_buffer = types.SimpleNamespace(size=lambda: 35715)
    assert abs(model.current_offline_share() - 15464 / 35715) < 1e-9
    fresh = o2o_utils.DSRLResumable.__new__(o2o_utils.DSRLResumable)
    fresh.replay_buffer = types.SimpleNamespace(size=lambda: 1000)
    assert fresh.current_offline_share() == 0.0
    fresh.offline_buf = object()
    fresh.offline_p = 0.37
    assert fresh.current_offline_share() == 0.37


class Cfg(dict):
    __getattr__ = dict.__getitem__


def make_cfg(mode="none", ent_coef_lr=-1, reward_scale=1.0, critic_entropy_scale=1.0):
    return Cfg(
        algorithm="dsrl_na", env_name="can", obs_dim=23, action_dim=7, act_steps=4,
        train=Cfg(layer_size=2048, num_layers=3, n_critics=2, use_layer_norm=True, buffer_size=200000,
                  ent_coef=-1, target_ent=0.0, ent_coef_lr=ent_coef_lr,
                  reward_scale=reward_scale, critic_entropy_scale=critic_entropy_scale),
        env=Cfg(n_envs=4), variant="baseline",
        offline_mix=Cfg(mode=mode, p0=0.8, p1=0.1, until_env=100000),
    )


def test_fingerprint_carries_the_alpha_learning_rate():
    assert o2o_utils.config_fingerprint(make_cfg())["ent_coef_lr"] == -1
    assert o2o_utils.config_fingerprint(make_cfg(ent_coef_lr=1.5e-4))["ent_coef_lr"] == 1.5e-4
    old = {"config": o2o_utils.config_fingerprint(make_cfg(ent_coef_lr=1.5e-4))}
    try:
        o2o_utils.check_fingerprint(old, make_cfg())
    except RuntimeError:
        return
    raise AssertionError("a half-rate checkpoint resumed under the shared rate")


def test_decoupled_alpha_optimizer_stays_out_of_the_lr_reset():
    # The class default means "shared", which is what upstream does.
    assert o2o_utils.DSRLResumable.ent_coef_lr is None


def test_gate_ratio_opens_after_k_consecutive_updates_below_tau():
    g = o2o_utils.GateController(signal="ratio", tau=1.0, K=3)
    for r in (5.0, 0.5, 0.5):
        assert g.update(r) is False
    assert g.update(0.5) is True and g.open_call == 4
    assert g.update(9.0) is True, "the gate never closes again"


def test_gate_streak_resets_on_a_violation_and_ignores_missing_ratio():
    g = o2o_utils.GateController(signal="ratio", tau=1.0, K=3)
    g.update(0.5); g.update(0.5); g.update(2.0)
    assert g.streak == 0 and not g.open
    g.update(0.5); g.update(None); g.update(0.5); g.update(0.5)
    assert not g.open  # None broke the streak
    g.update(0.5)
    assert g.open and g.open_call == 8


def test_gate_clock_opens_on_schedule_regardless_of_ratio():
    g = o2o_utils.GateController(signal="clock", clock_calls=3)
    assert g.update(0.0) is False and g.update(0.0) is False
    assert g.update(0.0) is True and g.open_call == 3


def test_gate_state_roundtrip_and_fingerprint():
    g = o2o_utils.GateController(signal="ratio", tau=0.8, K=2)
    g.update(0.1)
    h = o2o_utils.GateController(signal="ratio", tau=0.8, K=2)
    h.load_state_dict(g.state_dict())
    assert h.update(0.1) is True and h.open_call == 2
    cfg = make_cfg()
    cfg["gate"] = Cfg(enabled=True, signal="ratio", actuator="hard_backup", tau=1.0, K=50, clock_calls=0, alpha_hi=0.3)
    assert o2o_utils.config_fingerprint(cfg)["gate"] == "ratio:hard_backup:1.0:50:0:0.3"
    assert "gate" not in o2o_utils.config_fingerprint(make_cfg())
    try:
        o2o_utils.GateController(signal="oracle")
    except ValueError:
        return
    raise AssertionError("unknown signal accepted")


def test_critic_target_levers_default_to_upstream_and_enter_the_fingerprint():
    assert o2o_utils.DSRLResumable.reward_scale == 1.0
    assert o2o_utils.DSRLResumable.critic_entropy_scale == 1.0
    fp = o2o_utils.config_fingerprint(make_cfg())
    assert fp["reward_scale"] == 1.0 and fp["critic_entropy_scale"] == 1.0
    old = {"config": o2o_utils.config_fingerprint(make_cfg(reward_scale=2.0))}
    try:
        o2o_utils.check_fingerprint(old, make_cfg())
    except RuntimeError:
        pass
    else:
        raise AssertionError("a reward_scale=2 checkpoint resumed under scale 1")
    old = {"config": o2o_utils.config_fingerprint(make_cfg(critic_entropy_scale=0.0))}
    try:
        o2o_utils.check_fingerprint(old, make_cfg())
    except RuntimeError:
        return
    raise AssertionError("a hard-backup checkpoint resumed under the soft backup")


def test_fingerprint_carries_the_schedule_only_when_it_matters():
    assert o2o_utils.config_fingerprint(make_cfg("none"))["offline_mix"] == "none"
    assert o2o_utils.config_fingerprint(make_cfg("prefill"))["offline_mix"] == "prefill"
    assert o2o_utils.config_fingerprint(make_cfg("linear"))["offline_mix"] == "linear:0.8:0.1:100000"
    old = {"config": o2o_utils.config_fingerprint(make_cfg("linear"))}
    try:
        o2o_utils.check_fingerprint(old, make_cfg("fixed"))
    except RuntimeError:
        return
    raise AssertionError("a linear checkpoint resumed as fixed")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("%d checks passed" % len(tests))
