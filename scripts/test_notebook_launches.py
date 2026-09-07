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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("%d checks passed" % len(tests))
