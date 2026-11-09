"""The physics: unitree_mujoco's Go2 in the building, on DDS domain 0.

Same as scripts/run_sim.py, with the building in place of the
flat floor: rt/lowcmd in, rt/lowstate and rt/sportmodestate out, so go2_board
and the LiDAR emulator cannot tell the two apart.

  mjpython scripts/run_world.py [interface]
  python scripts/run_world.py --headless [interface]
  python scripts/run_world.py --headless --spawn=24.0,0.0,0 en9    # the dog at the lift doors
  python scripts/run_world.py --headless --record=run.mp4 en9      # and a video of it (go2_world/recorder.py)

--held-pd: unitree_mujoco's own motor model, the torque worked out once per
LowCmd and held until the next. By default the PD runs every physics step
on the last command, as the dog's drivers do (go2_world/motor_pd.py).

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
from go2_world.crowd import MOVERS_PORT, Movers  # noqa: E402
from go2_world.lift import LIFT_PORT, LiftServer  # noqa: E402
from go2_world.scene import SIMULATE_DT, UNITREE_MUJOCO, write_scene  # noqa: E402

HEADLESS = "--headless" in sys.argv[1:]
HELD_PD = "--held-pd" in sys.argv[1:]  # unitree_mujoco's own: torque once per LowCmd (go2_world/motor_pd.py)
SPAWN = None                 # --spawn X,Y,YAW: the dog elsewhere than the origin (this frame, rad)
PEOPLE = 0                   # --people=N: people walking the corridors (go2_world/crowd.py)
LIFT_LISTEN = "127.0.0.1"    # --lift-listen=IP: take lift calls there (the brain on another machine)
RECORD = None               # --record=PATH: the world to an MP4 while it runs
ARGS = []
for a in sys.argv[1:]:
    if a in ("--headless", "--held-pd"):
        continue
    if a.startswith("--lift-listen="):
        LIFT_LISTEN = a.split("=", 1)[1]
        continue
    if a.startswith("--people="):
        PEOPLE = int(a.split("=", 1)[1])
        continue
    if a.startswith("--record="):
        RECORD = os.path.abspath(a.split("=", 1)[1])
        continue
    if a.startswith("--spawn="):
        SPAWN = tuple(float(v) for v in a.split("=", 1)[1].split(","))
        continue
    ARGS.append(a)

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
      f" step {SIMULATE_DT * 1000:.0f} ms{', headless' if HEADLESS else ''},"
      f" motor PD {'once per LowCmd (held)' if HELD_PD else 'every step'}")

if not HELD_PD:
    # The motors' PD every physics step from the last command, as the dog's
    # drivers do -- not once per LowCmd, held. See go2_world/motor_pd.py.
    import unitree_sdk2py_bridge  # noqa: E402  (unitree_mujoco's, from SIM)

    from go2_world import motor_pd  # noqa: E402
    motor_pd.install(unitree_sdk2py_bridge.UnitreeSdk2Bridge)

sim = runpy.run_path(os.path.join(SIM, "unitree_mujoco.py"), run_name="__main__")

if SPAWN is not None:
    import math

    import mujoco
    x, y, yaw = SPAWN
    with sim["locker"]:
        d = sim["mj_data"]
        d.qpos[0:2] = (x, y)
        d.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qvel[:] = 0.0
        mujoco.mj_forward(sim["mj_model"], d)
    print(f"world: dog spawned at ({x:.2f}, {y:.2f}), yaw {math.degrees(yaw):.0f} deg", flush=True)

# The lift car: moved by its own thread, called by whoever presses the button
# (the brain stands in for the person, over UDP on LIFT_PORT).
lift = LiftServer(sim["mj_model"], sim["mj_data"], sim["locker"], host=LIFT_LISTEN).start()
print(f"lift: car at floor 1, call it on udp {LIFT_LISTEN}:{LIFT_PORT}", flush=True)
# The people, and the moving parts' positions for the LiDAR emulator.
Movers(sim["mj_model"], sim["mj_data"], sim["locker"], PEOPLE, lift=lift.lift).start()
print(f"world: {PEOPLE} people walking; movers to udp 127.0.0.1:{MOVERS_PORT}", flush=True)

if RECORD:
    import signal

    from go2_world.recorder import Recorder
    rec = Recorder(sim["mj_model"], sim["mj_data"], sim["locker"], RECORD).start()

    def _finish(signum, frame):
        # Stopped with pkill (SIGTERM) or Ctrl-C: finish the file, then go, as
        # the signal would have. The sim's threads are not daemons, so exit hard.
        rec.close()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, _finish)
    signal.signal(signal.SIGINT, _finish)
    print(f"recorder: {RECORD}", flush=True)

# mjpython closes the viewer when this script returns: wait for the sim's threads.
for t in threading.enumerate():
    if t is not threading.current_thread() and not t.daemon:
        t.join()
