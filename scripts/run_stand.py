"""Unitree's go2_stand_example.py, unchanged, against the sim.

  echo | python scripts/run_stand.py [interface]

Two things differ from running it on a real Go2:
  - the macOS timerfd shim (the SDK's timer is Linux-only);
  - MotionSwitcher is answered here. On the real dog the example first asks
    the dog's own service to switch its built-in walking controller off, so
    the 12 motors take low-level commands. The sim has no such service and
    no built-in controller, so the answer is "nothing active".
Never use this launcher against the real dog.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2_world import _macos_timerfd  # noqa: E402,F401  (before unitree_sdk2py)

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (  # noqa: E402
    MotionSwitcherClient)

MotionSwitcherClient.CheckMode = lambda self: (0, {"name": ""})

EXAMPLE = os.path.expanduser(
    "~/Work/unitree/unitree_sdk2_python/example/go2/low_level/go2_stand_example.py")
sys.path.insert(0, os.path.dirname(EXAMPLE))   # its unitree_legged_const.py
sys.argv = [EXAMPLE, sys.argv[1] if len(sys.argv) > 1 else "lo0"]
runpy.run_path(EXAMPLE, run_name="__main__")
