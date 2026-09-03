"""Turn the robomimic demonstrations into the chunked form DSRL loads.

    python scripts/make_offline_chunks.py \
        --load_path robomimic_raw/can/mh/low_dim_v141.hdf5 \
        --normalization_path dppo/log/robomimic/can/normalization.npz \
        --check_against dppo/log/robomimic/can/train.npz \
        --save_path dppo/log/robomimic/can/train_offline.npz

The published DPPO `train.npz` has no rewards (it only served diffusion-policy
pre-training), so the source is robomimic's own hdf5, which carries the sparse
reward per step. Observations and actions are normalised with the published
`normalization.npz`, the very statistics the online environment wrapper uses,
so the offline critic sees the same input space as the online one.
`--check_against` proves it: the states rebuilt here must equal the published
`train.npz` states to floating-point precision.

The hdf5 stores one row per environment step. DSRL's replay buffer stores one
row per action chunk, with the reward summed over the chunk and the observation
taken at the chunk boundary, so the demonstrations are regrouped before
`load_offline_data` can push them into the buffer.

Conventions follow what the online rollouts put in the buffer:

  * observations and actions are already normalised to [-1, 1] by the DPPO
    preprocessing, the same normalisation the environment wrapper applies;
  * the chunk reward is the sum of `reward_offset`-shifted step rewards, which
    is what ActionChunkWrapper accumulates and ObservationWrapperRobomimic
    shifts (Robomimic pays +1 per step while the task is solved, so a shifted
    step reward is 0 on success and -1 otherwise);
  * the last chunk of a demonstration is marked terminal. Online, a solved task
    keeps paying a shifted reward of 0 forever, so bootstrapping from zero at
    the end of a successful demonstration is the consistent choice.
"""

import argparse
import os

import numpy as np

# Same keys and order as cfg env.wrappers.robomimic_lowdim.low_dim_keys.
DEFAULT_LOW_DIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
QUALITY_LABELS = {"worse": 0, "okay": 1, "better": 2}


def normalize(x, low, high):
    """The formula DPPO's preprocessing and the env wrapper both use."""
    return 2 * (x - low) / (high - low + 1e-6) - 1


def load_hdf5(path, normalization_path, low_dim_keys):
    """Read robomimic's low_dim hdf5 into the flat per-step layout of train.npz.

    Returns states, actions (normalised), rewards, traj_lengths, and a per-step
    operator-quality label for the multi-human sets (-1 when unknown).
    """
    import h5py

    stats = np.load(normalization_path)
    obs_min, obs_max = stats["obs_min"], stats["obs_max"]
    act_min, act_max = stats["action_min"], stats["action_max"]

    states, actions, rewards, lengths, quality = [], [], [], [], []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda name: int(name.split("_")[1]))
        demo_quality = {}
        if "mask" in f:
            for label, value in QUALITY_LABELS.items():
                if label in f["mask"]:
                    for name in f["mask"][label][()]:
                        demo_quality[name.decode() if isinstance(name, bytes) else str(name)] = value
        for name in demos:
            group = f["data"][name]
            obs = np.concatenate([group["obs"][key][()] for key in low_dim_keys], axis=1)
            act = group["actions"][()]
            rew = group["rewards"][()]
            n = int(group.attrs["num_samples"])
            if not (obs.shape[0] == act.shape[0] == rew.shape[0] == n):
                raise ValueError("%s: obs %d, actions %d, rewards %d, num_samples %d"
                                 % (name, obs.shape[0], act.shape[0], rew.shape[0], n))
            states.append(normalize(obs, obs_min, obs_max))
            actions.append(normalize(act, act_min, act_max))
            rewards.append(rew)
            lengths.append(n)
            quality.append(np.full(n, demo_quality.get(name, -1), dtype=np.int8))

    return {
        "states": np.concatenate(states).astype(np.float32),
        "actions": np.concatenate(actions).astype(np.float32),
        "rewards": np.concatenate(rewards).astype(np.float32),
        "traj_lengths": np.asarray(lengths, dtype=np.int64),
        "quality": np.concatenate(quality),
    }


def load_dataset(path, normalization_path=None, low_dim_keys=None):
    if path.endswith(".hdf5") or path.endswith(".h5"):
        if not normalization_path:
            raise SystemExit("--normalization_path is required with an hdf5 source")
        return load_hdf5(path, normalization_path, low_dim_keys or DEFAULT_LOW_DIM_KEYS)
    data = dict(np.load(path))
    if "rewards" not in data:
        raise SystemExit(
            "%s has no 'rewards' (the published DPPO train.npz only served policy "
            "pre-training). Point --load_path at robomimic's low_dim hdf5 instead, "
            "with --normalization_path." % path
        )
    return data


def check_against(data, reference_path):
    """The rebuilt states must equal the published train.npz to fp precision.

    That single comparison covers demo order, key order, and normalisation.
    """
    ref = np.load(reference_path)
    ref_states = np.asarray(ref["states"], dtype=np.float32)
    if ref_states.shape != data["states"].shape:
        raise SystemExit("state shape %s differs from %s in %s"
                         % (data["states"].shape, ref_states.shape, reference_path))
    if "traj_lengths" in ref and not np.array_equal(ref["traj_lengths"], data["traj_lengths"]):
        raise SystemExit("traj_lengths differ from %s" % reference_path)
    gap = float(np.max(np.abs(ref_states - data["states"])))
    if gap > 1e-4:
        raise SystemExit("states differ from %s by up to %.3g" % (reference_path, gap))
    act_gap = None
    if "actions" in ref:
        act_gap = float(np.max(np.abs(np.asarray(ref["actions"], dtype=np.float32) - data["actions"])))
    print("check against %s: states match (max |diff| %.2g%s)"
          % (os.path.basename(reference_path), gap,
             "" if act_gap is None else ", actions %.2g" % act_gap))


def report_rewards(data):
    rewards, lengths = data["rewards"], data["traj_lengths"]
    solved = rewards > 0
    first_solved = []
    start = 0
    for n in lengths:
        r = solved[start:start + n]
        first_solved.append(int(np.argmax(r)) / n if r.any() else float("nan"))
        start += n
    first_solved = np.asarray(first_solved)
    print("source steps     %d in %d demos" % (rewards.shape[0], len(lengths)))
    print("rewarded steps   %.1f%%; demos with any reward %d"
          % (100.0 * solved.mean(), int(np.isfinite(first_solved).sum())))
    print("first reward at  %.0f%% of the demo on average" % (100.0 * np.nanmean(first_solved)))
    if "quality" in data and (data["quality"] >= 0).any():
        counts = {label: int((data["quality"] == v).sum()) for label, v in QUALITY_LABELS.items()}
        print("operator quality %s (steps)" % counts)


def build_chunks(data, act_steps, stride, reward_offset, terminal_at_traj_end):
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    traj_lengths = np.asarray(data["traj_lengths"], dtype=np.int64)
    quality = np.asarray(data["quality"], dtype=np.int8) if "quality" in data else None

    total = int(traj_lengths.sum())
    if total != states.shape[0]:
        raise ValueError(
            "traj_lengths sum to %d but there are %d rows" % (total, states.shape[0])
        )

    obs_out, next_obs_out, act_out, rew_out, term_out, qual_out = [], [], [], [], [], []
    start = 0
    skipped = 0
    for length in traj_lengths:
        length = int(length)
        s = states[start:start + length]
        a = actions[start:start + length]
        r = rewards[start:start + length]
        q = int(quality[start]) if quality is not None else -1
        start += length

        if length < act_steps + 1:
            skipped += 1
            continue

        for t in range(0, length - act_steps + 1, stride):
            next_index = t + act_steps
            if next_index <= length - 1:
                next_state = s[next_index]
                terminal = False
            else:
                # No state left in this demonstration: the transition ends it.
                next_state = s[length - 1]
                terminal = terminal_at_traj_end

            obs_out.append(s[t])
            next_obs_out.append(next_state)
            act_out.append(a[t:t + act_steps].reshape(-1))
            rew_out.append(r[t:t + act_steps].sum() - act_steps * reward_offset)
            term_out.append(terminal)
            qual_out.append(q)

    if skipped:
        print("skipped %d demonstrations shorter than %d steps" % (skipped, act_steps + 1))

    return {
        "states": np.asarray(obs_out, dtype=np.float32),
        "states_next": np.asarray(next_obs_out, dtype=np.float32),
        "actions": np.asarray(act_out, dtype=np.float32),
        "rewards": np.asarray(rew_out, dtype=np.float32),
        "terminals": np.asarray(term_out, dtype=bool),
        "quality": np.asarray(qual_out, dtype=np.int8),
    }


def trim_to_multiple(out, n_envs):
    """`load_offline_data` pushes n_envs rows per buffer slot and drops the rest."""
    n = out["states"].shape[0]
    keep = n - (n % n_envs)
    if keep != n:
        print("trimming %d rows to a multiple of n_envs=%d" % (n - keep, n_envs))
        out = {k: v[:keep] for k, v in out.items()}
    return out


def report(out, act_steps, n_envs):
    n = out["states"].shape[0]
    rewards = out["rewards"]
    print("transitions      %d (%d buffer slots at n_envs=%d)" % (n, n // n_envs, n_envs))
    print("obs dim          %d" % out["states"].shape[1])
    print("action dim       %d (%d steps x %d)" % (out["actions"].shape[1], act_steps, out["actions"].shape[1] // act_steps))
    print("chunk reward     min %.1f  max %.1f  mean %.3f" % (rewards.min(), rewards.max(), rewards.mean()))
    print("rewarded chunks  %.1f%% (a chunk paying more than -%d)"
          % (100.0 * float((rewards > -act_steps).mean()), act_steps))
    print("terminals        %d" % int(out["terminals"].sum()))
    print("obs range        [%.2f, %.2f]" % (out["states"].min(), out["states"].max()))
    print("action range     [%.2f, %.2f]" % (out["actions"].min(), out["actions"].max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", required=True,
                        help="robomimic low_dim hdf5 (recommended) or a DPPO npz that has rewards")
    parser.add_argument("--normalization_path", default="",
                        help="published normalization.npz; required with an hdf5 source")
    parser.add_argument("--check_against", default="",
                        help="published train.npz; the rebuilt states must match it exactly")
    parser.add_argument("--save_path", required=True, help="where to write the chunked npz")
    parser.add_argument("--act_steps", type=int, default=4, help="action chunk size of the base policy")
    parser.add_argument("--stride", type=int, default=1,
                        help="1 keeps a chunk starting at every step, act_steps keeps disjoint chunks")
    parser.add_argument("--reward_offset", type=int, default=1, help="env.reward_offset from the run config")
    parser.add_argument("--n_envs", type=int, default=4, help="env.n_envs of the run that will load this")
    parser.add_argument("--keep_traj_end_nonterminal", action="store_true",
                        help="do not mark the last chunk of a demonstration terminal")
    args = parser.parse_args()

    data = load_dataset(args.load_path, args.normalization_path)
    if args.check_against:
        check_against(data, args.check_against)
    report_rewards(data)
    out = build_chunks(
        data,
        act_steps=args.act_steps,
        stride=args.stride,
        reward_offset=args.reward_offset,
        terminal_at_traj_end=not args.keep_traj_end_nonterminal,
    )
    out = trim_to_multiple(out, args.n_envs)
    report(out, args.act_steps, args.n_envs)
    np.savez_compressed(args.save_path, **out)
    print("wrote", args.save_path)


if __name__ == "__main__":
    main()


