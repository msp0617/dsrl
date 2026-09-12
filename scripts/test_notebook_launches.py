"""Regression checks for TD pre-training and online notebook mappings."""

import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def notebook_code(name):
    path = os.path.join(ROOT, "colab", name)
    with open(path, encoding="utf-8") as f:
        notebook = json.load(f)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def notebook_text(name):
    path = os.path.join(ROOT, "colab", name)
    with open(path, encoding="utf-8") as f:
        notebook = json.load(f)
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_can_notebooks_map_td_to_td():
    for name in ("vm2_can_td_cql.ipynb", "dsrl_colab_calql_axisA.ipynb"):
        code = notebook_code(name)
        assert 'td) A="pretrain.method=td"' in code, name
        assert "launch can_${M}_s$S seed=$S variant=$M" in code, name
        assert "td) V=cql" not in code, name
        assert 'td) A="pretrain.method=cql pretrain.cql_alpha=0' not in code, name


def test_square_notebooks_map_td_to_td():
    for name in ("vm3_square_cql.ipynb", "dsrl_colab_calql_axisA_square.ipynb"):
        code = notebook_code(name)
        assert 'td) A="pretrain.method=td"' in code, name
        assert "launch square_${M}_s$S seed=$S variant=$M" in code, name
        assert "td) V=cql" not in code, name
        assert 'td) A="pretrain.method=cql pretrain.cql_alpha=0' not in code, name


def test_new_vm1_is_a_self_contained_can_launch_plan():
    code = notebook_code("vm1_new.ipynb")
    assert "launch can_${M}_s$S seed=$S variant=$M" in code
    assert "train.total_env_steps=150000" in code
    assert "can_{m}_s{s}" in code
    assert "td) V=cql" not in code
    assert "runtime.unassign()" in code


def test_new_vm2_launches_only_t12i_and_prefill():
    code = notebook_code("vm2_new.ipynb")
    assert "launch can_calql_t12i_s$S" in code
    assert "launch can_calql_prefill_s$S" in code
    assert code.count("variant=calql") == 2
    assert "offline_mix.mode=prefill" in code
    assert "train.target_ent=12" in code
    assert "runtime.unassign()" in code


def test_new_vm3_is_a_self_contained_square_launch_plan():
    code = notebook_code("vm3_new.ipynb")
    assert "launch square_${M}_s$S seed=$S variant=$M" in code
    assert "train.total_env_steps=100000" in code
    assert "--config-name=dsrl_square.yaml" in code
    assert "scripts/check_pretrain.py --config-path" not in code
    assert "+check_states=4096" in code
    assert " offline_data_path=$PROJ/offline/square_train_offline.npz check_states=4096" not in code
    assert "np.isclose(gamma, 0.999, rtol=0.0, atol=1e-6)" in code
    assert "< 1e-9" not in code
    assert "online launch is not blocked" in code
    assert "td) V=cql" not in code
    assert "runtime.unassign()" in code


def test_prefill_alpha_launches_fixed_03_and_auto_015_target_12():
    code = notebook_code("vm_prefill_alpha.ipynb")
    assert "launch can_prefill_fixa03_s$S" in code
    assert "launch can_prefill_t12i_a015_s$S" in code
    assert "$PREFILL train.ent_coef=0.3\n" in code
    assert "$PREFILL train.ent_coef=auto_0.15 train.target_ent=12\n" in code
    assert code.count("variant=baseline") == 2
    assert "offline_mix.mode=prefill offline_data_path=$PROJ/offline/can_train_offline.npz" in code
    assert "for kind in ('prefill_fixa03', 'prefill_t12i_a015')" in code
    assert "--only can_prefill_fixa03_s,can_prefill_t12i_a015_s" in code
    assert "runtime.unassign()" in code


def test_prefill_fixa015_is_a_single_fixed_alpha_arm():
    code = notebook_code("vm_prefill_fixa015.ipynb")
    assert "launch can_prefill_fixa015_s$S" in code
    assert code.count("launch can_") == 1
    assert "$PREFILL train.ent_coef=0.15\n" in code
    assert "target_ent" not in code
    assert "auto_" not in code
    assert "variant=baseline" in code
    assert "offline_mix.mode=prefill offline_data_path=$PROJ/offline/can_train_offline.npz" in code
    assert "EXPECTED = [f'can_prefill_fixa015_s{s}' for s in (1, 2, 3)]" in code
    assert "--only can_prefill_fixa015_s" in code
    assert "runtime.unassign()" in code


def test_new_notebooks_have_numbered_zero_to_nine_workflow():
    for name in (
        "vm1_new.ipynb",
        "vm2_new.ipynb",
        "vm3_new.ipynb",
        "vm_hq_can.ipynb",
        "vm_prefill_alpha.ipynb",
        "vm_prefill_fixa015.ipynb",
    ):
        text = notebook_text(name)
        for section in range(10):
            assert f"## {section}." in text, (name, section)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("%d checks passed" % len(tests))
