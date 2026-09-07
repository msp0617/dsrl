"""Checks for the CQL / Cal-QL pieces of offline_pretrain.py.

    .venv/bin/python scripts/test_offline_pretrain.py

Needs torch (the penalty is tensor arithmetic) but not mujoco, hydra or the
stable-baselines3 fork: those imports are stubbed so the module loads on a
laptop. The diffusion policy is a stub too; what is tested is the algebra the
online run inherits: the penalty's value and gradient direction, Cal-QL's
floor, the noise sampler and the returns check.
"""

import os
import sys
import types

import numpy as np
import torch as th


def install_stubs():
    hydra = types.ModuleType("hydra")
    hydra.main = lambda **kwargs: (lambda f: f)
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = types.SimpleNamespace(register_new_resolver=lambda *a, **k: None, resolve=lambda cfg: None)
    sb3 = types.ModuleType("stable_baselines3")
    common = types.ModuleType("stable_baselines3.common")
    logger = types.ModuleType("stable_baselines3.common.logger")
    logger.configure = lambda *a, **k: None
    torch_layers = types.ModuleType("stable_baselines3.common.torch_layers")
    torch_layers.create_mlp = lambda *a, **k: []
    utils_mod = types.ModuleType("stable_baselines3.common.utils")
    utils_mod.polyak_update = lambda *a, **k: None
    vec_env = types.ModuleType("stable_baselines3.common.vec_env")
    vec_env.DummyVecEnv = object
    o2o = types.ModuleType("o2o_utils")
    o2o.OfflineBuffer = object
    o2o.SpacesOnlyEnv = object
    o2o.build_agent = lambda *a, **k: None
    o2o.network_fingerprint = lambda cfg: {}
    utils = types.ModuleType("utils")
    utils.load_base_policy = lambda cfg: None
    utils.load_offline_data = lambda *a, **k: None
    for name, module in [
        ("hydra", hydra), ("omegaconf", omegaconf), ("stable_baselines3", sb3),
        ("stable_baselines3.common", common), ("stable_baselines3.common.logger", logger),
        ("stable_baselines3.common.torch_layers", torch_layers), ("stable_baselines3.common.utils", utils_mod),
        ("stable_baselines3.common.vec_env", vec_env), ("o2o_utils", o2o), ("utils", utils),
    ]:
        sys.modules[name] = module


install_stubs()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import offline_pretrain as op  # noqa: E402


class FakeModel:
    """Only what prior_noise and diffusion_actions touch."""

    diffusion_act_chunk, diffusion_act_dim = 4, 7
    device = th.device("cpu")

    def diffusion_policy(self, obs, noise, return_numpy=False):
        assert noise.shape[1:] == (4, 7)
        return noise + obs[:, :1, None]  # a marker so the row order can be checked


def test_penalty_is_logsumexp_over_unseen_and_data_minus_data():
    q_data = th.tensor([[1.0], [2.0]])
    q_ood = th.tensor([[3.0, 5.0], [0.0, 2.0]])
    penalty, floored = op.conservative_penalty(q_data, q_ood, alpha=2.0)
    expect = 2.0 * (th.logsumexp(th.tensor([[3.0, 5.0, 1.0], [0.0, 2.0, 2.0]]), dim=1) - q_data[:, 0]).mean()
    assert abs(penalty.item() - expect.item()) < 1e-5
    assert penalty.item() > 0 and floored.item() == 0.0


def test_penalty_vanishes_once_the_data_action_is_the_maximum():
    q_data = th.full((4, 1), 100.0, requires_grad=True)
    q_ood = th.zeros(4, 3, requires_grad=True)
    penalty, _ = op.conservative_penalty(q_data, q_ood, alpha=5.0)
    assert penalty.item() < 1e-3, "nothing unseen is valued above the data action"
    penalty.backward()
    assert q_data.grad.abs().max() < 1e-6 and q_ood.grad.abs().max() < 1e-6, "no push either way"


def test_penalty_gradient_pushes_unseen_down_and_data_up_only_while_unseen_is_higher():
    q_data = th.tensor([[1.0]], requires_grad=True)
    q_ood = th.tensor([[50.0, -50.0]], requires_grad=True)
    penalty, _ = op.conservative_penalty(q_data, q_ood, alpha=2.0)
    penalty.backward()
    assert q_ood.grad[0, 0] > 1.9, "the unseen action above the data action takes the push"
    assert q_ood.grad[0, 1] < 1e-6, "the one far below gets none"
    assert q_data.grad[0, 0] < -1.9, "and the data action is pushed up"


def test_calql_floor_stops_the_push_below_the_return_and_lifts_data_to_it():
    q_data = th.tensor([[1.0], [2.0]], requires_grad=True)
    q_ood = th.tensor([[3.0, -50.0], [-50.0, -50.0]], requires_grad=True)
    returns = th.tensor([[-10.0], [20.0]])
    penalty, floored = op.conservative_penalty(q_data, q_ood, alpha=2.0, returns=returns)
    assert abs(floored.item() - 0.75) < 1e-6, "three of the four unseen values sat below their return"
    penalty.backward()
    assert q_ood.grad[0, 0] > 0 and q_ood.grad[0, 1] == 0 and (q_ood.grad[1] == 0).all(), "floored entries get no push"
    assert q_data.grad[1, 0] < -0.9, "row 2: the return (20) is above the data action (2), so it is pulled up (alpha / batch)"


def test_prior_noise_shape_and_clip():
    th.manual_seed(0)
    model = FakeModel()
    noise = op.prior_noise(512, model)
    assert noise.shape == (512, 4, 7)
    assert noise.abs().max() > 1.5, "unclipped N(0, I) reaches past the online noise box"
    clipped = op.prior_noise(512, model, clip=1.5)
    assert clipped.abs().max() <= 1.5


def test_diffusion_actions_keep_row_order_and_flatten():
    model = FakeModel()
    obs = th.arange(3, dtype=th.float32).reshape(3, 1).repeat(1, 23)
    noise = th.zeros(3, 4, 7)
    actions = op.diffusion_actions(model, obs, noise)
    assert actions.shape == (3, 28)
    assert th.allclose(actions[2], th.full((28,), 2.0)), "row i came from observation i"


def test_require_returns():
    ok = types.SimpleNamespace(returns=th.zeros(4, 1), returns_gamma=0.99)
    op.require_returns(ok, 0.99, "x.npz")
    for bad in (types.SimpleNamespace(returns=None, returns_gamma=None),
                types.SimpleNamespace(returns=th.zeros(4, 1), returns_gamma=0.999),
                types.SimpleNamespace(returns=th.zeros(4, 1), returns_gamma=None)):
        try:
            op.require_returns(bad, 0.99, "x.npz")
        except ValueError as e:
            assert "--gamma 0.99" in str(e)
        else:
            raise AssertionError("missing or mismatched returns must be refused before training")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("%d checks passed" % len(tests))
