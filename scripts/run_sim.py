"""Run unitree_mujoco's Go2 sim, flat floor, on this Mac without editing Unitree's files.

Overrides four settings from simulate_python/config.py:
  DOMAIN_ID 1 -> 0   the real Go2 and Unitree's examples use domain 0
  INTERFACE lo -> lo0 (macOS loopback), or the name given as argv[1]
  USE_JOYSTICK 1 -> 0 no gamepad attached
  SIMULATE_DT 5 -> 2 ms  at 5 ms the stand gains chatter and the dog walks off

run_world.py is the same with Cyberdog's building in place of the floor.
Run with mjpython (the MuJoCo viewer needs it on macOS):
  mjpython scripts/run_sim.py [interface]
"""
import os
import runpy
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2_world import _macos_timerfd  # noqa: E402,F401  (before unitree_sdk2py)
from go2_world.scene import SIMULATE_DT, UNITREE_MUJOCO  # noqa: E402

SIM = os.path.join(UNITREE_MUJOCO, "simulate_python")
os.chdir(SIM)                      # config.ROBOT_SCENE is a relative path
sys.path.insert(0, SIM)
import config  # noqa: E402  (the sim's own module, cached for its import)

config.DOMAIN_ID = 0
config.INTERFACE = sys.argv[1] if len(sys.argv) > 1 else "lo0"
config.USE_JOYSTICK = 0
config.SIMULATE_DT = SIMULATE_DT
print(f"sim: robot {config.ROBOT}, domain {config.DOMAIN_ID}, interface {config.INTERFACE}")

runpy.run_path(os.path.join(SIM, "unitree_mujoco.py"), run_name="__main__")

# mjpython closes the viewer when this script returns, and unitree_mujoco's
# __main__ only starts its threads. Wait for them, so the sim keeps running
# until the viewer window is closed.
for t in threading.enumerate():
    if t is not threading.current_thread() and not t.daemon:
        t.join()
