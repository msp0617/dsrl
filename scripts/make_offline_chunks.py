"""Turn a DPPO-processed robomimic dataset into the chunked form DSRL loads.

    python scripts/make_offline_chunks.py \
        --load_path dppo/log/robomimic/can/train.npz \
        --save_path dppo/log/robomimic/can/train_offline.npz

The published `train.npz` stores one row per environment step. DSRL's replay
buffer stores one row per action chunk, with the reward summed over the chunk
and the observation taken at the chunk boundary, so the offline demonstrations
have to be regrouped before `load_offline_data` can push them into the buffer.

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

import numpy as np


def build_chunks(data, act_steps, stride, reward_offset, terminal_at_traj_end):
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    traj_lengths = np.asarray(data["traj_lengths"], dtype=np.int64)

    total = int(traj_lengths.sum())
    if total != states.shape[0]:
        raise ValueError(
            "traj_lengths sum to %d but there are %d rows" % (total, states.shape[0])
        )

    obs_out, next_obs_out, act_out, rew_out, term_out = [], [], [], [], []
    start = 0
    skipped = 0
    for length in traj_lengths:
        length = int(length)
        s = states[start:start + length]
        a = actions[start:start + length]
        r = rewards[start:start + length]
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

    if skipped:
        print("skipped %d demonstrations shorter than %d steps" % (skipped, act_steps + 1))

    return {
        "states": np.asarray(obs_out, dtype=np.float32),
        "states_next": np.asarray(next_obs_out, dtype=np.float32),
        "actions": np.asarray(act_out, dtype=np.float32),
        "rewards": np.asarray(rew_out, dtype=np.float32),
        "terminals": np.asarray(term_out, dtype=bool),
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
    parser.add_argument("--load_path", required=True, help="DPPO-processed train.npz")
    parser.add_argument("--save_path", required=True, help="where to write the chunked npz")
    parser.add_argument("--act_steps", type=int, default=4, help="action chunk size of the base policy")
    parser.add_argument("--stride", type=int, default=1,
                        help="1 keeps a chunk starting at every step, act_steps keeps disjoint chunks")
    parser.add_argument("--reward_offset", type=int, default=1, help="env.reward_offset from the run config")
    parser.add_argument("--n_envs", type=int, default=4, help="env.n_envs of the run that will load this")
    parser.add_argument("--keep_traj_end_nonterminal", action="store_true",
                        help="do not mark the last chunk of a demonstration terminal")
    args = parser.parse_args()

    data = np.load(args.load_path)
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


