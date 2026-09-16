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


def notebook_cells(name):
    path = os.path.join(ROOT, "colab", name)
    with open(path, encoding="utf-8") as f:
        return [c for c in json.load(f)["cells"] if c.get("cell_type") == "code"]


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


def test_square_rs_launches_two_rescaled_arms_and_the_t12i_control():
    code = notebook_code("vm_square_rs.ipynb")
    assert "--config-name=dsrl_square.yaml" in code
    assert 'T12I="train.ent_coef=auto_0.3 train.target_ent=12"' in code
    assert "launch square_tent12i_rs02_s$S seed=$S $T12I train.reward_scale=0.2\n" in code
    assert "launch square_fixa015_rs02_s$S seed=$S train.ent_coef=0.15 train.reward_scale=0.2\n" in code
    assert "launch square_tent12i_s$S      seed=$S $T12I\n" in code
    assert code.count("launch square_") == 3 and "launch can_" not in code
    assert "critic_alpha_cap" not in code and "critic_entropy_scale" not in code
    assert "train.total_env_steps=150000" in code
    assert "for kind in ('tent12i_rs02', 'fixa015_rs02', 'tent12i')" in code
    assert "--only square_tent12i_rs02_s,square_fixa015_rs02_s,square_tent12i_s" in code
    assert "runtime.unassign()" in code


def test_square_cap_launches_two_caps_and_the_can_regression():
    code = notebook_code("vm_square_cap.ipynb")
    assert "--config-name=dsrl_square.yaml" in code and "--config-name=dsrl_can.yaml" in code
    assert 'T12I="train.ent_coef=auto_0.3 train.target_ent=12"' in code
    assert 'launch square_tent12i_cap03_s$S "$CFG_SQ"  seed=$S $T12I train.critic_alpha_cap=0.3\n' in code
    assert 'launch square_tent12i_cap1_s$S  "$CFG_SQ"  seed=$S $T12I train.critic_alpha_cap=1.0\n' in code
    assert 'launch can_tent12i_cap03_s$S    "$CFG_CAN" seed=$S $T12I train.critic_alpha_cap=0.3\n' in code
    assert code.count("launch square_") == 2 and code.count("launch can_") == 1
    assert "train.reward_scale" not in code and "critic_entropy_scale" not in code
    assert "train.total_env_steps=150000" in code
    # the unit tests and the cap smoke gate the launch cell
    assert "scripts/test_offline_mix.py" in code and "scripts/test_notebook_launches.py" in code
    assert "train.critic_alpha_cap=$CAP" in code and "for CAP in 0.3 -1; do" in code
    assert "'critic_ent_coef' in rows[0]" in code
    assert "for kind in ('cap03', 'cap1')" in code
    assert "--only square_tent12i_cap03_s,square_tent12i_cap1_s,can_tent12i_cap03_s" in code
    assert "runtime.unassign()" in code


def test_square_a0_reruns_tent12_under_a_new_id_and_adds_seeds_4_5():
    code = notebook_code("vm_square_a0.ipynb")
    assert "--config-name=dsrl_square.yaml" in code and "launch can_" not in code
    assert 'T12I="train.ent_coef=auto_0.3 train.target_ent=12"' in code
    # the re-run keeps tent12's original override (auto alpha, SB3 init 1.0) under a new exp_id:
    # the same name would resume the finished 9/6 checkpoints
    assert "launch square_tent12r_s$S       seed=$S train.target_ent=12\n" in code
    assert "launch square_tent12_s" not in code
    assert "launch square_tent12i_s$S       seed=$S $T12I\n" in code
    assert "launch square_tent12i_cap03_s$S seed=$S $T12I train.critic_alpha_cap=0.3\n" in code
    assert "for S in 1 2 3; do\n  launch square_tent12r_s$S" in code
    assert "for S in 4 5; do\n  launch square_tent12i_s$S" in code
    assert code.count("launch square_") == 3
    assert "critic_alpha_fixed=" not in code and "train.discount" not in code and "train.reward_scale" not in code
    assert "train.total_env_steps=150000" in code
    assert "[f'square_tent12r_s{s}' for s in (1, 2, 3)] + [f'square_tent12i_s{s}' for s in (4, 5)] + [f'square_tent12i_cap03_s{s}' for s in (4, 5)]" in code
    assert "--only square_tent12r_s,square_tent12i_s4,square_tent12i_s5,square_tent12i_cap03_s4,square_tent12i_cap03_s5" in code
    assert "runtime.unassign()" in code


def test_square_cfix_launches_fixed_critic_temperatures_and_the_gamma_arm():
    code = notebook_code("vm_square_cfix.ipynb")
    assert "--config-name=dsrl_square.yaml" in code and "launch can_" not in code
    assert 'T12I="train.ent_coef=auto_0.3 train.target_ent=12"' in code
    assert "launch square_tent12i_cfix03_s$S seed=$S $T12I train.critic_alpha_fixed=0.3\n" in code
    assert "launch square_tent12i_cfix1_s$S  seed=$S $T12I train.critic_alpha_fixed=1.0\n" in code
    assert "launch square_tent12i_g099_s$S   seed=$S $T12I train.discount=0.99\n" in code
    assert code.count("launch square_") == 3
    # the cap appears only in the smoke that checks cap+fixed is refused, never in a launch line
    assert "$T12I train.critic_alpha_cap" not in code and "train.reward_scale" not in code
    assert "train.total_env_steps=150000" in code
    # the unit tests and the fixed-temperature smoke gate the launch cell; the smoke must
    # prove "constant", not "min": fixed 2.0 above the actor's fixed alpha 1.0
    assert "scripts/test_offline_mix.py" in code and "scripts/test_notebook_launches.py" in code
    assert "for FIXED in 2.0 -1; do" in code and "train.critic_alpha_fixed=$FIXED" in code
    assert "want = [fixed if fixed > 0 else x for x in a]" in code
    assert "train.critic_alpha_cap=0.3 train.critic_alpha_fixed=1.0" in code and "mutually exclusive" in code
    assert "for kind in ('cfix03', 'cfix1', 'g099')" in code
    assert "--only square_tent12i_cfix03_s,square_tent12i_cfix1_s,square_tent12i_g099_s" in code
    assert "runtime.unassign()" in code


def test_new_notebooks_have_numbered_zero_to_nine_workflow():
    for name in (
        "vm1_new.ipynb",
        "vm2_new.ipynb",
        "vm3_new.ipynb",
        "vm_hq_can.ipynb",
        "vm_prefill_alpha.ipynb",
        "vm_prefill_fixa015.ipynb",
        "vm_square_rs.ipynb",
        "vm_square_cap.ipynb",
        "vm_square_a0.ipynb",
        "vm_square_cfix.ipynb",
    ):
        text = notebook_text(name)
        for section in range(10):
            assert f"## {section}." in text, (name, section)
        # a shell cell must use the cell magic: "%bash" is not a line magic and the
        # cell dies with a SyntaxError before running anything (caught 9/16)
        for cell in notebook_cells(name):
            first = "".join(cell.get("source", [])).split(chr(10), 1)[0]
            if first.startswith("%"):
                assert first.strip() == "%%bash", (name, first)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("%d checks passed" % len(tests))
