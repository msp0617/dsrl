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

**Resume.** `ResumeCheckpointCallback` writes weights, the replay buffer and the
step counters on a period measured in environment steps. It alternates between
two slots and writes `run_state.json` last, so a session killed mid-save still
leaves the previous checkpoint intact. On start, `train_dsrl.py` reads that file
and continues: it restores the noise critic's optimizer and `log_ent_coef`,
which upstream `save`/`set_parameters` would drop, skips the initial rollout,
and passes only the remaining budget to `learn`.

Resume needs a log directory that survives the session. Pass `exp_id=<name>` and
the run writes to `${log_dir}/<name>` instead of a timestamped folder.

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
them.

## Running

```bash
python colab/patch_env.py        # once per Colab session, patches site-packages
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
  exp_id=can_baseline_s1 seed=1 log_dir=/content/drive/MyDrive/dsrl_project/logs
```

Re-running the same command after a session dies picks up where it stopped.

## Known upstream issue

`ReplayBuffer.sample` in the stable-baselines3 submodule builds its sampling
probabilities with length `self.pos` but draws indices over `buffer_size` once
the buffer is full, so a full buffer raises. Sizing the buffer above the run
length avoids it.
