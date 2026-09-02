"""Checks for the offline chunking, runnable without mujoco or torch.

    python scripts/test_make_offline_chunks.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_offline_chunks import build_chunks, trim_to_multiple


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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("%d checks passed" % len(tests))
