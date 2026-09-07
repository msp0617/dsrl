"""Pre-train DSRL's critics on the offline demonstrations, without a simulator.

Five methods, one script, so an online run differs from the baseline only in
where its initial critic weights come from:

  warmup  runs DSRL's own update (Algorithm 1) on the offline data for k steps.
          Its Q_A target is r + gamma * Q_A_target(s', pi_dp(s', pi_W(s'))): the
          target depends on the actor pi_W, which is random when this starts.

  iql     runs implicit Q-learning. V(s) is fitted by expectile regression to
          Q_A(s, a) over actions that are in the data, and the Q_A target is
          r + gamma * V(s'). No actor is involved and nothing is evaluated
          outside the data. Q_A is then distilled into Q_W exactly as the
          online loop does it (Algorithm 1, line 5).

  td      plain TD: Q_A <- r + gamma * Q_A_target(s', pi_dp(s', w')), w' ~ N(0, I),
          the value of the diffusion policy under its prior noise, with no
          conservatism and no calibration. The rung between no pre-training
          and cql/calql. Distilled into Q_W like iql.

  cql     iql's critic plus CQL's regulariser: the same expectile V(s) and
          the same target r + gamma * V(s'), and on top of the TD loss
          cql_alpha * (E_w Q_A(s, pi_dp(s, w)) - Q_A(s, a_data)) pushes down
          the actions pi_dp can produce but the data does not hold, and pushes
          up the ones it does, in CQL(H)'s log-sum-exp form with the data
          action inside the set (so the push stops once unseen actions sit
          below the data action). The target never sees the sampled actions,
          so the penalty cannot feed back through the bootstrap. cql_alpha=0
          is iql. Distilled into Q_W like iql.

  calql   the same with Cal-QL's floor: in the penalty, Q_A of a sampled action
          is replaced by max(Q_A, G), G the demonstration's return-to-go from s
          (the 'returns' column make_offline_chunks.py --gamma writes), so the
          penalty cannot drive Q below what the data actually earned. Identical
          to cql in target, weight and sampling; the floor is the only change.

Usage, same Hydra config and overrides as train_dsrl.py:

  python offline_pretrain.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \\
      pretrain.method=iql pretrain.steps=50000
  python offline_pretrain.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \\
      pretrain.method=calql pretrain.cql_alpha=5.0

Writes a .pt holding the state dicts plus a meta block. An online run picks it
up with `variant=iql pretrain_path=<file>` (or variant=warmup, cql, calql).

Nothing here touches robosuite or mujoco: the agent is built on
SpacesOnlyEnv, which carries the task's spaces and nothing else, and the
diffusion policy needs only torch. This stage can run on a small GPU.
"""

import csv
import math
import os
import random
import sys
import time

import hydra
import numpy as np
import torch as th
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

sys.path.append('./dppo')

from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv

from o2o_utils import OfflineBuffer, SpacesOnlyEnv, build_agent, network_fingerprint
from utils import load_base_policy, load_offline_data

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------- IQL

def expectile_loss(diff, expectile):
    """rho_tau(u) = |tau - 1{u < 0}| * u^2, applied to u = Q(s, a) - V(s).

    With tau > 0.5 a sample whose Q sits above V is weighted more than one
    below it, so V is pulled up toward the best actions the data contains
    rather than toward their average. That makes V a soft in-sample maximum of
    Q: the "max over actions" that a Q target needs, without ever querying Q on
    an action the data does not have.
    """
    weight = th.abs(expectile - (diff < 0).float())
    return weight * diff.pow(2)


def combine_q(q_values, combine_type):
    stacked = th.cat(q_values, dim=1)
    if combine_type == "mean":
        return th.mean(stacked, dim=1, keepdim=True)
    return th.min(stacked, dim=1, keepdim=True)[0]


def make_value_net(cfg, device):
    """Same MLP recipe as the critics, minus the action input."""
    post_linear_modules = [nn.LayerNorm] if cfg.train.use_layer_norm else None
    net_arch = [int(cfg.train.layer_size)] * int(cfg.train.num_layers)
    layers = create_mlp(int(cfg.obs_dim), 1, net_arch, nn.Tanh, post_linear_modules=post_linear_modules)
    return nn.Sequential(*layers).to(device)


def run_iql(model, cfg, pre, log):
    """Fit V and Q_A on the data alone. Returns the value network."""
    device = model.device
    gamma = float(cfg.train.discount)
    expectile = float(pre.expectile)
    batch_size = int(pre.batch_size)
    combine = str(cfg.train.critic_backup_combine_type)

    value_net = make_value_net(cfg, device)
    value_optimizer = th.optim.Adam(value_net.parameters(), lr=float(cfg.train.actor_lr))
    critic, critic_target = model.critic, model.critic_target
    critic.set_training_mode(True)

    for step in range(1, int(pre.steps) + 1):
        batch = model.replay_buffer.sample(batch_size)
        obs, actions, next_obs = batch.observations, batch.actions, batch.next_observations
        rewards, dones = batch.rewards, batch.dones

        # V(s) <- expectile regression on Q_target(s, a), a from the data
        with th.no_grad():
            q_data = combine_q(critic_target(obs, actions), combine)
        value = value_net(obs)
        value_loss = expectile_loss(q_data - value, expectile).mean()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Q_A(s, a) <- r + gamma V(s'). No action is sampled for s'.
        with th.no_grad():
            target = rewards + gamma * (1.0 - dones) * value_net(next_obs)
        q_values = critic(obs, actions)
        critic_loss = 0.5 * sum(F.mse_loss(q, target) for q in q_values)
        critic.optimizer.zero_grad()
        critic_loss.backward()
        critic.optimizer.step()
        polyak_update(critic.parameters(), critic_target.parameters(), model.tau)

        if step % int(pre.log_every) == 0 or step == int(pre.steps):
            log("iql", step, {
                "value_loss": value_loss.item(),
                "critic_loss": critic_loss.item(),
                "q_mean": q_data.mean().item(),
                "v_mean": value.mean().item(),
            })
    return value_net


def run_distill(model, pre, log):
    """Q_W(s, w) <- Q_A(s, pi_dp(s, w)), w ~ N(0, I). Algorithm 1, line 5."""
    batch_size = int(pre.batch_size)
    model.critic_noise.set_training_mode(True)
    for step in range(1, int(pre.distill_steps) + 1):
        batch = model.replay_buffer.sample(batch_size)
        loss = model.update_noise_critic(batch)
        model.critic_noise.optimizer.zero_grad()
        loss.backward()
        model.critic_noise.optimizer.step()
        if step % int(pre.log_every) == 0 or step == int(pre.distill_steps):
            log("distill", step, {"noise_critic_loss": loss.item()})
    model.critic_noise.set_training_mode(False)


def run_actor(model, cfg, pre, log):
    """Optional: move pi_W toward argmax_w Q_W before going online.

    Off by default so that the iql variant changes only the critics. The
    entropy coefficient is held fixed here.
    """
    batch_size = int(pre.batch_size)
    combine = str(cfg.train.critic_backup_combine_type)
    for step in range(1, int(pre.actor_steps) + 1):
        batch = model.replay_buffer.sample(batch_size)
        actions_pi, log_prob = model.actor.action_log_prob(batch.observations)
        q_pi = combine_q(model.critic_noise(batch.observations, actions_pi), combine)
        if model.log_ent_coef is not None:
            ent_coef = th.exp(model.log_ent_coef.detach())
        else:
            ent_coef = model.ent_coef_tensor
        actor_loss = (ent_coef * log_prob.reshape(-1, 1) - q_pi).mean()
        model.actor.optimizer.zero_grad()
        actor_loss.backward()
        model.actor.optimizer.step()
        if step % int(pre.log_every) == 0 or step == int(pre.actor_steps):
            log("actor", step, {"actor_loss": actor_loss.item()})


# ------------------------------------------------------------------ warm-up

def run_warmup(model, cfg, pre, log):
    """Algorithm 1 as implemented online, minus the environment.

    One online update is `utd` critic/actor steps followed by
    `noise_critic_grad_steps` distillation steps; calling `train` with the
    same `gradient_steps` keeps that ratio instead of doing all distillation
    at the end.
    """
    utd = int(cfg.train.utd)
    batch_size = int(pre.batch_size)
    calls = max(int(pre.steps) // utd, 1)
    for call in range(1, calls + 1):
        model.train(gradient_steps=utd, batch_size=batch_size)
        step = call * utd
        if step % int(pre.log_every) < utd or call == calls:
            values = model.logger.name_to_value
            log("warmup", step, {
                "critic_loss": float(values.get("train/critic_loss", float("nan"))),
                "actor_loss": float(values.get("train/actor_loss", float("nan"))),
                "noise_critic_loss": float(values.get("train/noise_critic_loss", float("nan"))),
                "ent_coef": float(values.get("train/ent_coef", float("nan"))),
            })


# ------------------------------------------------------------ CQL / Cal-QL

def prior_noise(n, model, clip=0.0):
    """w ~ N(0, I) in the shape pi_dp takes: the noise it was trained with and
    the noise update_noise_critic distils over. Clipped to +-clip when clip > 0
    (the online noise space is +-train.action_magnitude)."""
    noise = th.randn(n, model.diffusion_act_chunk, model.diffusion_act_dim, device=model.device)
    if clip > 0:
        noise = noise.clamp(-clip, clip)
    return noise


def diffusion_actions(model, obs, noise):
    """a = pi_dp(s, w), flattened to the chunk the critics take (as in o2o_utils' train)."""
    actions = model.diffusion_policy(obs, noise, return_numpy=False)
    return actions.reshape(obs.shape[0], model.diffusion_act_chunk * model.diffusion_act_dim)


def conservative_penalty(q_data, q_ood, alpha, returns=None, temp=1.0):
    """CQL(H)'s regulariser on one critic head; Cal-QL's floor when returns is given.

    q_data (B, 1): Q of the action in the data. q_ood (B, n): Q of n actions
    pi_dp produces from prior noise, which the data does not contain.
      penalty = alpha * (temp * logsumexp({q_ood_1..n, q_data} / temp) - q_data)
    The data action sits inside the log-sum-exp, so the penalty is never
    negative and it is self-limiting: an unseen action is pushed down only
    while it is valued above the data action (its softmax weight vanishes
    below), and the data action is pushed up only while something else is
    the maximum. (A plain alpha * (mean q_ood - q_data) has a constant
    gradient: with no actor tying the sampled actions to the target, q_ood
    fell to -8,000 and q_data rose by alpha/(1-gamma) = +500 on Can.)
    Cal-QL replaces q_ood by max(q_ood, G), G (B, 1) the demonstration's
    return-to-go from s: the push on an unseen action stops once it is
    valued below what the data actually earned, and a data action valued
    below G is pulled up to it.
    Returns the penalty and the fraction of q_ood entries the floor replaced.
    """
    if returns is not None:
        floored = (q_ood < returns).float().mean()
        q_ood = th.maximum(q_ood, returns.expand_as(q_ood))
    else:
        floored = th.zeros((), device=q_ood.device)
    cat = th.cat([q_ood, q_data], dim=1)
    lse = temp * th.logsumexp(cat / temp, dim=1, keepdim=True)
    penalty = alpha * (lse - q_data).mean()
    return penalty, floored


def require_returns(buf, gamma, path):
    """Cal-QL needs the return-to-go column, discounted with the run's gamma."""
    if buf.returns is None:
        raise ValueError(
            "%s has no 'returns'; rebuild it with scripts/make_offline_chunks.py --gamma %g" % (path, gamma)
        )
    if buf.returns_gamma is None or abs(float(buf.returns_gamma) - gamma) > 1e-6:
        raise ValueError(
            "'returns' in %s were discounted with gamma=%r but train.discount is %g; rebuild with --gamma %g"
            % (path, buf.returns_gamma, gamma, gamma)
        )


def run_cql(model, cfg, pre, log, buf, calibrate):
    """Q_A with IQL's in-sample target plus CQL's penalty; calibrate=True is Cal-QL.

    V(s) is fitted by expectile regression to Q_target(s, a_data) and the
    Q_A target is r + gamma V(s'), exactly as run_iql does; on top, the
    penalty acts on Q_A of actions pi_dp produces from prior noise. The
    target never sees those sampled actions, so the penalty cannot feed
    back through the bootstrap. (A first version bootstrapped from
    Q_target(s', pi_dp(s', w')), the very distribution the penalty pushes
    down; with no actor to push back, Q_A spiralled to -9,500 on Can within
    40k steps while the returns are about -100. Real CQL relies on the actor
    maximising Q to hold that side up.) With cql_alpha=0 this is run_iql.
    Returns the value network.
    """
    device = model.device
    gamma = float(cfg.train.discount)
    expectile = float(pre.expectile)
    alpha = float(pre.cql_alpha)
    n_samples = int(pre.cql_n_samples)
    clip = float(pre.cql_noise_clip)
    batch_size = int(pre.batch_size)
    combine = str(cfg.train.critic_backup_combine_type)
    critic, critic_target = model.critic, model.critic_target
    critic.set_training_mode(True)
    value_net = make_value_net(cfg, device)
    value_optimizer = th.optim.Adam(value_net.parameters(), lr=float(cfg.train.actor_lr))
    phase = "calql" if calibrate else "cql"

    for step in range(1, int(pre.steps) + 1):
        batch, returns = buf.sample(batch_size, with_returns=True)
        obs, actions, next_obs = batch.observations, batch.actions, batch.next_observations
        rewards, dones = batch.rewards, batch.dones

        # V(s) <- expectile regression on Q_target(s, a), a from the data
        with th.no_grad():
            q_data_target = combine_q(critic_target(obs, actions), combine)
        value = value_net(obs)
        value_loss = expectile_loss(q_data_target - value, expectile).mean()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        logging = step % int(pre.log_every) == 0 or step == int(pre.steps)
        # With cql_alpha=0 the penalty is zero whatever the sampled actions
        # are worth, so pi_dp is only run on them at log steps (diagnostic).
        need_ood = alpha > 0 or logging
        with th.no_grad():
            # Q_A(s, a) <- r + gamma V(s'). No action is sampled for s'.
            target = rewards + gamma * (1.0 - dones) * value_net(next_obs)
            if need_ood:
                # actions the online actor could play but the data does not contain
                obs_rep = obs.repeat_interleave(n_samples, dim=0)
                ood_actions = diffusion_actions(model, obs_rep, prior_noise(batch_size * n_samples, model, clip))

        q_values = critic(obs, actions)
        td_loss = 0.5 * sum(F.mse_loss(q, target) for q in q_values)
        penalty, floored = 0.0, 0.0
        q_ood_heads = None
        if need_ood:
            q_ood_heads = critic(obs_rep, ood_actions)
            for q_d, q_o in zip(q_values, q_ood_heads):
                p, f = conservative_penalty(
                    q_d, q_o.reshape(batch_size, n_samples), alpha, returns if calibrate else None
                )
                penalty = penalty + p
                floored = floored + f / len(q_values)
        loss = td_loss + penalty
        critic.optimizer.zero_grad()
        loss.backward()
        critic.optimizer.step()
        polyak_update(critic.parameters(), critic_target.parameters(), model.tau)

        if logging:
            log(phase, step, {
                "value_loss": value_loss.item(),
                "critic_loss": td_loss.item(),
                "penalty": float(penalty),
                "q_mean": th.cat(q_values, dim=1).mean().item(),
                "v_mean": value.mean().item(),
                "q_ood_mean": th.cat(q_ood_heads, dim=1).mean().item(),
                "floored_frac": float(floored),
                "return_mean": returns.mean().item() if returns is not None else float("nan"),
            })
    return value_net


def run_td(model, cfg, pre, log, buf):
    """Plain TD pre-training of Q_A: r + gamma Q_A_target(s', pi_dp(s', w')), w' ~ N(0, I).

    The value of the diffusion policy under its prior noise, fitted by
    ordinary TD with no conservatism and no calibration: the rung between
    "no pre-training" and cql/calql, so that pre-training itself can be told
    apart from what cql and calql add. Unlike iql it evaluates actions the
    data does not contain (pi_dp's), so it can over-estimate them, which is
    the point of the comparison. Same sampled-action diagnostic as cql.
    """
    gamma = float(cfg.train.discount)
    n_samples = int(pre.cql_n_samples)
    clip = float(pre.cql_noise_clip)
    batch_size = int(pre.batch_size)
    combine = str(cfg.train.critic_backup_combine_type)
    critic, critic_target = model.critic, model.critic_target
    critic.set_training_mode(True)
    for step in range(1, int(pre.steps) + 1):
        batch, returns = buf.sample(batch_size, with_returns=True)
        obs, actions, next_obs = batch.observations, batch.actions, batch.next_observations
        rewards, dones = batch.rewards, batch.dones
        logging = step % int(pre.log_every) == 0 or step == int(pre.steps)
        with th.no_grad():
            next_actions = diffusion_actions(model, next_obs, prior_noise(batch_size, model, clip))
            next_q = combine_q(critic_target(next_obs, next_actions), combine)
            target = rewards + gamma * (1.0 - dones) * next_q
            if logging:
                obs_rep = obs.repeat_interleave(n_samples, dim=0)
                ood_actions = diffusion_actions(model, obs_rep, prior_noise(batch_size * n_samples, model, clip))
        q_values = critic(obs, actions)
        td_loss = 0.5 * sum(F.mse_loss(q, target) for q in q_values)
        critic.optimizer.zero_grad()
        td_loss.backward()
        critic.optimizer.step()
        polyak_update(critic.parameters(), critic_target.parameters(), model.tau)
        if logging:
            with th.no_grad():
                q_ood = th.cat(critic(obs_rep, ood_actions), dim=1).mean().item()
            log("td", step, {
                "critic_loss": td_loss.item(),
                "q_mean": th.cat(q_values, dim=1).mean().item(),
                "q_ood_mean": q_ood,
                "return_mean": returns.mean().item() if returns is not None else float("nan"),
            })


# --------------------------------------------------------------------- main

@hydra.main(
    config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    pre = cfg.pretrain
    method = str(pre.method)
    if method not in ("iql", "warmup", "td", "cql", "calql"):
        raise ValueError("pretrain.method must be iql, warmup, td, cql or calql, got %r" % method)
    if method in ("td", "cql", "calql") and int(pre.actor_steps) > 0:
        raise ValueError("pretrain.actor_steps is iql-only: td, cql and calql go online with a random actor")
    if cfg.algorithm != "dsrl_na":
        raise NotImplementedError("offline pre-training is written for dsrl_na (it needs Q_A and Q_W)")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    data_path = str(cfg.offline_data_path)
    with np.load(data_path) as data:
        n_rows = int(data["states"].shape[0])
        obs_dim, action_dim = int(data["states"].shape[1]), int(data["actions"].shape[1])
        if "states_next" not in data:
            raise ValueError("%s has no states_next; build it with scripts/make_offline_chunks.py" % data_path)
    expected_action_dim = int(cfg.act_steps) * int(cfg.action_dim)
    if obs_dim != int(cfg.obs_dim) or action_dim != expected_action_dim:
        raise ValueError(
            "offline data is (%d, %d) but the config expects obs %d, action %d x %d"
            % (obs_dim, action_dim, cfg.obs_dim, cfg.act_steps, cfg.action_dim)
        )

    out_path = str(pre.get("out_path", "") or "")
    if not out_path:
        out_path = os.path.join(str(cfg.log_dir), "pretrain", "%s_%s_s%d.pt" % (method, cfg.env_name, cfg.seed))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = os.path.splitext(out_path)[0] + "_log.csv"
    if os.path.exists(log_path):
        os.remove(log_path)

    def log(phase, step, values):
        row = {"wall_time": time.strftime("%Y-%m-%d %H:%M:%S"), "phase": phase, "step": step}
        row.update({k: float(v) for k, v in values.items()})
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wall_time", "phase", "step", "value_loss", "critic_loss",
                                                    "noise_critic_loss", "actor_loss", "ent_coef", "q_mean", "v_mean",
                                                    "penalty", "q_ood_mean", "floored_frac", "return_mean"])
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print("[%s] step %d  %s" % (phase, step, "  ".join("%s=%.4f" % kv for kv in values.items())), flush=True)

    base_policy = load_base_policy(cfg)
    env = DummyVecEnv([lambda: SpacesOnlyEnv(int(cfg.obs_dim), expected_action_dim)])
    model = build_agent(cfg, env, base_policy, buffer_size=n_rows + 1024, tensorboard_log=None, verbose=0)
    model.set_logger(configure(None, []))
    load_offline_data(model, data_path, 1)
    print("[data] %d transitions from %s" % (model.replay_buffer.size(), data_path), flush=True)
    buf = None
    if method in ("td", "cql", "calql"):
        # sampled by index so that Cal-QL's return-to-go lines up with the rows
        buf = OfflineBuffer(data_path, model.device)
        if method == "calql":
            require_returns(buf, float(cfg.train.discount), data_path)

    started = time.time()
    payload = {"meta": {
        "method": method,
        "steps": int(pre.steps),
        "distill_steps": int(pre.distill_steps) if method != "warmup" else 0,
        "actor_steps": int(pre.actor_steps) if method != "warmup" else int(pre.steps),
        "expectile": float(pre.expectile) if method in ("iql", "cql", "calql") else None,
        "cql_alpha": float(pre.cql_alpha) if method in ("cql", "calql") else None,
        "cql_n_samples": int(pre.cql_n_samples) if method in ("cql", "calql") else None,
        "cql_noise_clip": float(pre.cql_noise_clip) if method in ("cql", "calql") else None,
        "calibrated": method == "calql",
        "returns_gamma": float(buf.returns_gamma) if method == "calql" else None,
        "batch_size": int(pre.batch_size),
        "offline_data_path": data_path,
        "n_transitions": n_rows,
        "seed": int(cfg.seed),
        "network": network_fingerprint(cfg),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}

    if method == "warmup":
        run_warmup(model, cfg, pre, log)
        payload["actor"] = model.actor.state_dict()
        if model.log_ent_coef is not None:
            payload["log_ent_coef"] = model.log_ent_coef.detach().cpu()
    elif method == "iql":
        value_net = run_iql(model, cfg, pre, log)
        run_distill(model, pre, log)
        payload["value"] = value_net.state_dict()
        if int(pre.actor_steps) > 0:
            run_actor(model, cfg, pre, log)
            payload["actor"] = model.actor.state_dict()
    elif method == "td":
        run_td(model, cfg, pre, log, buf)
        run_distill(model, pre, log)
    else:
        value_net = run_cql(model, cfg, pre, log, buf, calibrate=(method == "calql"))
        run_distill(model, pre, log)
        payload["value"] = value_net.state_dict()

    payload["critic"] = model.critic.state_dict()
    payload["critic_target"] = model.critic_target.state_dict()
    payload["critic_noise"] = model.critic_noise.state_dict()
    payload["meta"]["seconds"] = round(time.time() - started, 1)

    th.save(payload, out_path)
    print("[done] %s in %.0fs -> %s (%.0f MB)" % (
        method, time.time() - started, out_path, os.path.getsize(out_path) / 1e6), flush=True)
    print("       launch online with: variant=%s pretrain_path=%s" % (method, out_path), flush=True)


if __name__ == "__main__":
    main()
