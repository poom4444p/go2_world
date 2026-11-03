"""The physics: unitree_mujoco's Go2 in the building, on DDS domain 0.

Same as scripts/run_sim.py, with the building in place of the
flat floor: rt/lowcmd in, rt/lowstate and rt/sportmodestate out, so go2_board
and the LiDAR emulator cannot tell the two apart.

  mjpython scripts/run_world.py [interface]
  python scripts/run_world.py --headless [interface]

--headless: no viewer, plain python. unitree_mujoco's physics thread waits
on the viewer's lock while a frame is drawn; with the GPU busy (VAMOS on
MPS) that held the physics 50-130 ms at a time, long enough for go2_board's
stale-state guard to trip. Headless, the same runs stay under 20 ms.
"""
import os
import runpy
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2_world import _macos_timerfd  # noqa: E402,F401  (before unitree_sdk2py)
from go2_world.scene import SIMULATE_DT, UNITREE_MUJOCO, write_scene  # noqa: E402

HEADLESS = "--headless" in sys.argv[1:]
ARGS = [a for a in sys.argv[1:] if a != "--headless"]

if HEADLESS:
    import mujoco.viewer

    class _NoViewer:
        """Runs until the process is killed; draws nothing."""
        def is_running(self):
            return True

        def sync(self):
            pass

    mujoco.viewer.launch_passive = lambda *a, **k: _NoViewer()

SIM = os.path.join(UNITREE_MUJOCO, "simulate_python")
os.chdir(SIM)
sys.path.insert(0, SIM)
import config  # noqa: E402  (the sim's own module, cached for its import)

config.ROBOT_SCENE = write_scene()
config.DOMAIN_ID = 0
config.INTERFACE = ARGS[0] if ARGS else "lo0"
config.USE_JOYSTICK = 0
config.PRINT_SCENE_INFORMATION = False
config.SIMULATE_DT = SIMULATE_DT
print(f"world: {config.ROBOT_SCENE}, domain {config.DOMAIN_ID}, interface {config.INTERFACE},"
      f" step {SIMULATE_DT * 1000:.0f} ms{', headless' if HEADLESS else ''}")

runpy.run_path(os.path.join(SIM, "unitree_mujoco.py"), run_name="__main__")

# mjpython closes the viewer when this script returns: wait for the sim's threads.
for t in threading.enumerate():
    if t is not threading.current_thread() and not t.daemon:
        t.join()
