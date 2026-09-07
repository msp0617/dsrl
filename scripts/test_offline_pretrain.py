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


def test_penalty_is_alpha_times_ood_minus_data():
    q_data = th.tensor([[1.0], [2.0]])
    q_ood = th.tensor([[3.0, 5.0], [0.0, 2.0]])
    penalty, floored = op.conservative_penalty(q_data, q_ood, alpha=2.0)
    # rows: (4 - 1) and (1 - 2), mean 1, times alpha
    assert abs(penalty.item() - 2.0) < 1e-6
    assert floored.item() == 0.0


def test_calql_floor_lifts_ood_values_to_the_return():
    q_data = th.tensor([[1.0], [2.0]])
    q_ood = th.tensor([[3.0, 5.0], [0.0, 2.0]])
    returns = th.tensor([[4.5], [1.5]])
    penalty, floored = op.conservative_penalty(q_data, q_ood, alpha=2.0, returns=returns)
    # rows: mean(4.5, 5) - 1 = 3.75 and mean(1.5, 2) - 2 = -0.25, mean 1.75, times alpha
    assert abs(penalty.item() - 3.5) < 1e-6
    assert abs(floored.item() - 0.5) < 1e-6, "3 and 0 were below their returns, 5 and 2 were not"


def test_penalty_gradient_pushes_data_up_and_unseen_down_but_not_below_the_floor():
    q_data = th.tensor([[1.0], [2.0]], requires_grad=True)
    q_ood = th.tensor([[3.0, 5.0], [0.0, 2.0]], requires_grad=True)
    returns = th.tensor([[4.5], [1.5]])
    penalty, _ = op.conservative_penalty(q_data, q_ood, alpha=2.0, returns=returns)
    penalty.backward()
    assert (q_data.grad < 0).all(), "descending the penalty raises Q on data actions"
    assert q_ood.grad[0, 1] > 0 and q_ood.grad[1, 1] > 0, "and lowers Q on unseen actions above the floor"
    assert q_ood.grad[0, 0] == 0 and q_ood.grad[1, 0] == 0, "floored entries get no push"


def test_without_returns_every_unseen_action_is_pushed():
    q_ood = th.tensor([[3.0, 5.0], [0.0, 2.0]], requires_grad=True)
    penalty, _ = op.conservative_penalty(th.zeros(2, 1), q_ood, alpha=1.0)
    penalty.backward()
    assert (q_ood.grad > 0).all()


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
