"""Runtime patches for the installed packages on Colab.

These touch site-packages, not this repository, so they have to be re-applied in
every new Colab session (run this once after installing the environment).

  python colab/patch_env.py

1. robomimic imports mujoco_py unconditionally, which is not installed (and does
   not build) under Python 3.10 with mujoco 3.x. The import is made optional.
"""

import site
import sys
from pathlib import Path


def patch_robomimic_env_robosuite():
    candidates = [Path(p) / "robomimic/envs/env_robosuite.py" for p in site.getsitepackages()]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("robomimic/envs/env_robosuite.py not found in site-packages")

    text = path.read_text()
    if "MUJOCO_EXCEPTIONS" in text:
        print("already patched:", path)
        return

    old_import = "import mujoco_py\n"
    new_import = (
        "try:\n"
        "    import mujoco_py\n"
        "    MUJOCO_EXCEPTIONS = [mujoco_py.builder.MujocoException]\n"
        "except ImportError:\n"
        "    MUJOCO_EXCEPTIONS = []\n\n"
    )
    if old_import not in text:
        raise RuntimeError("expected 'import mujoco_py' in %s" % path)
    text = text.replace(old_import, new_import, 1)

    old_return = "return (mujoco_py.builder.MujocoException)"
    new_return = "return tuple(MUJOCO_EXCEPTIONS)"
    if old_return in text:
        text = text.replace(old_return, new_return, 1)
    elif new_return not in text:
        raise RuntimeError("rollout_exceptions not found in %s" % path)

    path.write_text(text)
    print("patched:", path)


def main():
    patch_robomimic_env_robosuite()
    import robomimic.envs.env_robosuite as module  # noqa: F401

    print("robomimic env import OK")


if __name__ == "__main__":
    sys.exit(main())
