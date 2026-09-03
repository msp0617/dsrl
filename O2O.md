# Offline-to-online study on top of DSRL

This fork adds the infrastructure needed to run DSRL-NA on robomimic Can from
Google Colab, where sessions die without warning, and to compare the baseline
against variants that initialise the critic from offline data.

Upstream code is left as close to `ajwagen/dsrl` as possible. Everything new
lives in `o2o_utils.py`, plus small marked changes in `utils.py` and
`train_dsrl.py`.

## What changed

**Portability.** `d4rl` is imported only for the Gym tasks and `wandb` only when
`use_wandb=True`, so the robomimic tasks run in an environment where neither is
installed. Both used to be hard imports at module level.

**Resume.** `ResumeCheckpointCallback` writes weights, the replay buffer, the
random state and the step counters on a period measured in environment steps.
It alternates between two slots and writes `run_state.json` last, so a session
killed mid-save still leaves the previous checkpoint intact, and the reader
falls back to the older slot when the newest archive turns out to be truncated.
On start, `train_dsrl.py` continues from the newest usable slot: it restores the
noise critic's optimizer and `log_ent_coef`, which upstream `save` and
`set_parameters` would drop, continues the random stream instead of replaying it
from the seed, skips the initial rollout, and passes only the remaining budget
to `learn`.

Resume needs a log directory that survives the session. Pass `exp_id=<name>` and
the run writes to `${log_dir}/<name>` instead of a timestamped folder.

Three things are refused up front, before the environments and the diffusion
policy are built: a checkpoint written by a differently shaped run, a checkpoint
with no replay buffer beside it, and a buffer too small for the requested
budget. Each of those otherwise fails hours in, or worse, does not fail at all.

Two limitations worth knowing. The vectorised environment is reset on resume, so
each restart truncates the episodes in flight. And an evaluation that a session
dies during is simply lost, since only whole evaluations reach the CSV.

**Units.** `train.total_env_steps` and the eval and checkpoint periods are all in
original-environment steps, the unit the paper's x axis uses. One SB3 timestep is
one action chunk, so an environment step is `act_steps * n_envs` per callback
call. The initial rollout used to be counted without `act_steps`, which
understated the logged timestep by 4x on Can; that is fixed.

**Logging.** Every train and eval metric is appended to `train_log.csv` and
`eval_log.csv` in the log directory, whether or not W&B is enabled, so a lost
session still leaves a usable learning curve. Success rate now appears there.
The eval schedule can be dense early and sparse later, which is where the
initial dip lives.

**Buffer size.** The buffer was hard-coded to 10M transitions, several GB of
preallocated arrays, which makes checkpointing it impractical. It is now
`train.buffer_size`. Slots are `buffer_size / n_envs` and one slot holds one
action chunk per env, so a run needs
`init_rollout_steps + offline_adds + total_env_steps / (act_steps * n_envs)` of
them. `check_buffer_capacity` enforces this at startup.

## Disk

A checkpoint is dominated by the optimizer state of a 3x2048 critic pair:

| part | size |
|---|---|
| policy, including the target critic | 170 MB |
| actor and critic optimizers | 204 MB |
| noise critic and its optimizer | 204 MB |
| replay buffer | 60 MB |

That is about 640 MB per slot and 1.3 GB per run with both slots. Nine runs do
not fit in a 15 GB Drive, so delete a finished run's `checkpoint/` directory once
its CSVs are safe. The CSVs, not the weights, are the result.

## Evaluation cost

One evaluation is `num_evals` episodes of 300 environment steps, so 200 episodes
cost 60,000 environment steps of simulation. Evaluating every 5,000 training
steps therefore spends more simulation on measurement than on training. The
early phase compensates with `num_evals_early`, which halves the episodes while
the schedule is dense.

## Running

```bash
python colab/patch_env.py        # once per Colab session, patches site-packages
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
  exp_id=can_baseline_s1 seed=1 log_dir=/content/drive/MyDrive/dsrl_project/logs
```

Re-running the same command after a session dies picks up where it stopped.

## Variants

`variant` changes one thing: where the initial critic weights come from.

| variant | Q_A, Q_W at step 0 | actor at step 0 |
|---|---|---|
| `baseline` | random (upstream DSRL-NA) | random |
| `warmup` | DSRL's own update run on the offline data for `pretrain.steps` | trained alongside, loaded |
| `iql` | IQL on the offline data, then distilled into Q_W | random (unless `pretrain.actor_steps > 0`) |

Both pretrained variants are produced by `offline_pretrain.py`, which needs no
simulator: the agent is built on `SpacesOnlyEnv`, which carries the task's
spaces and nothing else, and the diffusion policy only needs torch. It runs on
any GPU and its output is reused by every seed of the online run.

```bash
python offline_pretrain.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
  pretrain.method=iql seed=1 offline_data_path=<chunked npz>
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
  variant=iql pretrain_path=<the .pt above> exp_id=can_iql_s1 seed=1
```

The difference the study is after sits in the Q_A target. Warm-up uses DSRL's
own, `r + γ Q̄_A(s', π_dp(s', π_W(s')))`, which leans on an actor that is
random at that point. IQL fits `V(s)` by expectile regression to `Q_A(s, a)` on
actions present in the data and uses `r + γ V(s')`: no actor, nothing queried
outside the data. Distillation into Q_W is then the online loop's own
`update_noise_critic`, unchanged.

Whether the offline data also stays in the online replay buffer is a separate
switch, `load_offline_data`, deliberately independent of the variant so that
the critic-initialisation effect can be measured on its own.

The `.pt` carries a `meta` block with the method and the network fingerprint;
a file made for another variant or another network shape is refused on load,
and the run fingerprint includes the variant so a baseline checkpoint cannot be
resumed as an iql run.

## Offline data

The config points `offline_data_path` at `can_test/train_offline.npz`, which the
authors did not publish. What is published is `robomimic/can/train.npz`, the
DPPO-processed Multi-Human demonstrations, one row per environment step.
`scripts/make_offline_chunks.py` regroups those into the chunked rows the replay
buffer stores:

```bash
python scripts/make_offline_chunks.py \
  --load_path dppo/log/robomimic/can/train.npz \
  --save_path dppo/log/robomimic/can/train_offline.npz
```

`python scripts/test_make_offline_chunks.py` checks the regrouping without
needing mujoco or torch.

## Checks

`python scripts/test_resume_state.py` covers the crash-recovery bookkeeping with
stub modules, so slot alternation, the fallback to the older slot and the config
fingerprint can be exercised without torch or mujoco.
`python scripts/test_make_offline_chunks.py` covers the data conversion.

## Known upstream issue

`ReplayBuffer.sample` in the stable-baselines3 submodule builds its sampling
probabilities with length `self.pos` but draws indices over `buffer_size` once
the buffer is full, so a full buffer raises. Sizing the buffer above the run
length avoids it, which is what the startup check enforces.
