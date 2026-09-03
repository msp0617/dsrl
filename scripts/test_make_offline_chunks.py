"""Checks for the offline chunking, runnable without mujoco or torch.

    python scripts/test_make_offline_chunks.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_offline_chunks import build_chunks, check_against, load_hdf5, normalize, trim_to_multiple


def fake_dataset():
    """Two demonstrations, solved over their last two steps."""
    lengths = [10, 6]
    n = sum(lengths)
    rng = np.random.default_rng(0)
    rewards = np.zeros(n, dtype=np.float32)
    rewards[8:10] = 1.0
    rewards[14:16] = 1.0
    return {
        "states": rng.uniform(-1, 1, size=(n, 23)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(n, 7)).astype(np.float32),
        "rewards": rewards,
        "traj_lengths": np.array(lengths),
    }


def test_overlapping_chunks():
    data = fake_dataset()
    out = build_chunks(data, act_steps=4, stride=1, reward_offset=1, terminal_at_traj_end=True)
    states = data["states"]

    assert out["states"].shape[0] == 10, "7 chunks from a 10-step demo, 3 from a 6-step one"
    assert out["actions"].shape[1] == 28

    assert np.allclose(out["states"][0], states[0])
    assert np.allclose(out["states_next"][0], states[4]), "next state is act_steps later"

    assert not out["terminals"][5]
    assert out["terminals"][6], "last chunk of a demo ends it"
    assert np.allclose(out["states_next"][6], states[9]), "clamped, and masked out by the terminal"

    assert abs(out["rewards"][6] - (-2.0)) < 1e-5, "steps 6..9 pay 0,0,1,1 shifted by -1 each"
    assert abs(out["rewards"][7] - (-4.0)) < 1e-5, "first chunk of the second demo solves nothing"
    assert np.allclose(out["states"][7], states[10]), "chunks never cross a demo boundary"
    assert out["terminals"][9]


def test_disjoint_chunks():
    data = fake_dataset()
    out = build_chunks(data, act_steps=4, stride=4, reward_offset=1, terminal_at_traj_end=True)
    assert out["states"].shape[0] == 3, "t=0,4 in the 10-step demo and t=0 in the 6-step one"


def test_short_demo_is_skipped():
    data = fake_dataset()
    data["traj_lengths"] = np.array([13, 3])
    out = build_chunks(data, act_steps=4, stride=1, reward_offset=1, terminal_at_traj_end=True)
    assert out["states"].shape[0] == 10


def test_trim_to_multiple():
    data = fake_dataset()
    out = build_chunks(data, act_steps=4, stride=1, reward_offset=1, terminal_at_traj_end=True)
    trimmed = trim_to_multiple(out, n_envs=4)
    assert trimmed["states"].shape[0] == 8
    assert trimmed["terminals"].shape[0] == 8


def test_length_mismatch_raises():
    data = fake_dataset()
    data["traj_lengths"] = np.array([10, 5])
    try:
        build_chunks(data, act_steps=4, stride=1, reward_offset=1, terminal_at_traj_end=True)
    except ValueError:
        return
    raise AssertionError("a traj_lengths sum that disagrees with the rows must raise")


def test_hdf5_source_matches_published_normalisation():
    """States rebuilt from the hdf5 with the published stats equal train.npz."""
    import tempfile

    import h5py

    d = tempfile.mkdtemp()
    rng = np.random.default_rng(1)
    keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    dims = [3, 4, 2, 14]
    lengths = [12, 9, 15]

    raw_obs, raw_act, rewards = [], [], []
    with h5py.File(os.path.join(d, "low_dim.hdf5"), "w") as f:
        data = f.create_group("data")
        mask = f.create_group("mask")
        mask.create_dataset("better", data=np.array([b"demo_0"]))
        mask.create_dataset("worse", data=np.array([b"demo_2"]))
        # written out of order on purpose: demo_10 must sort after demo_2
        for name, n in zip(["demo_2", "demo_0", "demo_10"], [15, 12, 9]):
            g = data.create_group(name)
            g.attrs["num_samples"] = n
            obs = g.create_group("obs")
            parts = []
            for key, dim in zip(keys, dims):
                x = rng.uniform(-2, 3, size=(n, dim))
                obs.create_dataset(key, data=x)
                parts.append(x)
            act = rng.uniform(-1, 1, size=(n, 7))
            rew = np.zeros(n); rew[-2:] = 1.0
            g.create_dataset("actions", data=act)
            g.create_dataset("rewards", data=rew)
            raw_obs.append((name, np.concatenate(parts, axis=1)))
            raw_act.append((name, act)); rewards.append((name, rew))

    order = ["demo_0", "demo_2", "demo_10"]
    raw_obs = np.concatenate([dict(raw_obs)[k] for k in order])
    raw_act = np.concatenate([dict(raw_act)[k] for k in order])
    obs_min, obs_max = raw_obs.min(0), raw_obs.max(0)
    act_min, act_max = raw_act.min(0), raw_act.max(0)
    np.savez(os.path.join(d, "normalization.npz"), obs_min=obs_min, obs_max=obs_max,
             action_min=act_min, action_max=act_max)
    # the "published" train.npz, produced by DPPO's own formula
    np.savez(os.path.join(d, "train.npz"),
             states=(2 * (raw_obs - obs_min) / (obs_max - obs_min + 1e-6) - 1).astype(np.float32),
             actions=(2 * (raw_act - act_min) / (act_max - act_min + 1e-6) - 1).astype(np.float32),
             traj_lengths=np.array([12, 15, 9]))

    data = load_hdf5(os.path.join(d, "low_dim.hdf5"), os.path.join(d, "normalization.npz"), keys)
    assert data["states"].shape == (36, 23) and data["actions"].shape == (36, 7)
    assert list(data["traj_lengths"]) == [12, 15, 9], "demos are ordered numerically"
    assert data["rewards"].sum() == 6
    assert data["quality"][0] == 2 and data["quality"][12] == 0 and data["quality"][-1] == -1
    check_against(data, os.path.join(d, "train.npz"))  # raises on any mismatch

    out = build_chunks(data, act_steps=4, stride=1, reward_offset=1, terminal_at_traj_end=True)
    assert out["quality"].shape[0] == out["states"].shape[0]
    assert out["quality"][0] == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("%d checks passed" % len(tests))
