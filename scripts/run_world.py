"""The physics: unitree_mujoco's Go2 in the building, on DDS domain 0.

Same as ~/Work/unitree/scripts/run_sim.py, with the building in place of the
flat floor: rt/lowcmd in, rt/lowstate and rt/sportmodestate out, so go2_board
and the LiDAR emulator cannot tell the two apart.

  mjpython scripts/run_world.py [interface]
"""
import os
import runpy
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2_world import _macos_timerfd  # noqa: E402,F401  (before unitree_sdk2py)
from go2_world.scene import SIMULATE_DT, UNITREE_MUJOCO, write_scene  # noqa: E402

SIM = os.path.join(UNITREE_MUJOCO, "simulate_python")
os.chdir(SIM)
sys.path.insert(0, SIM)
import config  # noqa: E402  (the sim's own module, cached for its import)

config.ROBOT_SCENE = write_scene()
config.DOMAIN_ID = 0
config.INTERFACE = sys.argv[1] if len(sys.argv) > 1 else "lo0"
config.USE_JOYSTICK = 0
config.PRINT_SCENE_INFORMATION = False
config.SIMULATE_DT = SIMULATE_DT
print(f"world: {config.ROBOT_SCENE}, domain {config.DOMAIN_ID}, interface {config.INTERFACE},"
      f" step {SIMULATE_DT * 1000:.0f} ms")

runpy.run_path(os.path.join(SIM, "unitree_mujoco.py"), run_name="__main__")

# mjpython closes the viewer when this script returns: wait for the sim's threads.
for t in threading.enumerate():
    if t is not threading.current_thread() and not t.daemon:
        t.join()
