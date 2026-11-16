"""The physics: unitree_mujoco's Go2 in the building, on DDS domain 0.

Same as scripts/run_sim.py, with the building in place of the
flat floor: rt/lowcmd in, rt/lowstate and rt/sportmodestate out, so go2_board
and the LiDAR emulator cannot tell the two apart.

  mjpython scripts/run_world.py [interface]
  python scripts/run_world.py --headless [interface]
  python scripts/run_world.py --headless --spawn=24.0,0.0,0 en9    # the dog at the lift doors
  python scripts/run_world.py --headless --record=run.mp4 en9      # and a video of it (go2_world/recorder.py)
  python scripts/run_world.py --headless --stand=3.0,0.8 en9        # somebody standing in the trolley's north gap
  python scripts/run_world.py --headless --walker=6.0,0.5 en9       # somebody slower walking ahead in the dog's lane
  python scripts/run_world.py --headless --record=run.mp4 --puddle=9.5,-0.6,0.34 en9   # a puddle drawn in the video

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
import signal
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2_world import _macos_timerfd  # noqa: E402,F401  (before unitree_sdk2py)
from go2_world.crowd import APPEAR_M as crowd_appear  # noqa: E402
from go2_world.crowd import MOVERS_PORT, Movers, SlowWalker, Stander  # noqa: E402
from go2_world.lift import LIFT_PORT, LiftServer  # noqa: E402
from go2_world.scene import SIMULATE_DT, UNITREE_MUJOCO, write_scene  # noqa: E402

HEADLESS = "--headless" in sys.argv[1:]
HELD_PD = "--held-pd" in sys.argv[1:]  # unitree_mujoco's own: torque once per LowCmd (go2_world/motor_pd.py)
SPAWN = None                 # --spawn X,Y,YAW: the dog elsewhere than the origin (this frame, rad)
PEOPLE = 0                   # --people=N: people walking the corridors (go2_world/crowd.py)
LIFT_LISTEN = "127.0.0.1"    # --lift-listen=IP: take lift calls there (the brain on another machine)
RECORD = None               # --record=PATH: the world to an MP4 while it runs
STAND = []                   # --stand=X,Y[,SECONDS[,FLOOR]]: somebody standing still (crowd.Stander),
                             # for SECONDS of wall clock then gone (default: for good), floor 1
WALK = []                    # --walker=X0,V[,X1[,Y[,FLOOR]]]: somebody walking from X0 to X1
                             # (default 18) at V m/s of simulated time, along Y (default
                             # -0.6, the right-hand lane going +x), floor 1 (crowd.SlowWalker)
PUDDLES = []                 # --puddle=X,Y,R[,FLOOR]: a puddle, drawn in --record's video only
                             # (no physics, no LiDAR: recorder.add_puddles), floor 1
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
    if a.startswith("--stand="):
        v = [float(p) for p in a.split("=", 1)[1].split(",")]
        STAND.append((v[0], v[1], int(v[3]) if len(v) > 3 else 1, v[2] if len(v) > 2 else None))
        continue
    if a.startswith("--walker="):
        v = [float(p) for p in a.split("=", 1)[1].split(",")]
        WALK.append((v[0], v[1], v[2] if len(v) > 2 else 18.0, v[3] if len(v) > 3 else -0.6,
                     int(v[4]) if len(v) > 4 else 1))
        continue
    if a.startswith("--puddle="):
        v = [float(p) for p in a.split("=", 1)[1].split(",")]
        PUDDLES.append((v[0], v[1], v[2], int(v[3]) if len(v) > 3 else 1))
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

# rt/lowstate's tick: the simulated time, for the Mid-360's IMU (pose.install_tick)
import unitree_sdk2py_bridge  # noqa: E402,F811  (unitree_mujoco's, from SIM)

from go2_world.pose import install_tick  # noqa: E402
install_tick(unitree_sdk2py_bridge.UnitreeSdk2Bridge)

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
movers = Movers(sim["mj_model"], sim["mj_data"], sim["locker"], PEOPLE, lift=lift.lift,
                standers=[Stander(x, y, floor, stay) for x, y, floor, stay in STAND]
                + [SlowWalker(x0, v, x1, y, floor) for x0, v, x1, y, floor in WALK]).start()
print(f"world: {PEOPLE} people walking; movers to udp 127.0.0.1:{MOVERS_PORT}", flush=True)
for x, y, floor, stay in STAND:
    print(f"world: somebody standing at ({x}, {y}) floor {floor}"
          f"{'' if stay is None else f' for {stay:.0f} s'}", flush=True)
for x0, v, x1, y, floor in WALK:
    print(f"world: somebody walking {x0} -> {x1} at {v} m/s along y {y}, floor {floor},"
          f" setting off when the dog is {crowd_appear} m behind", flush=True)

rec = None
if RECORD:
    from go2_world.recorder import Recorder
    rec = Recorder(sim["mj_model"], sim["mj_data"], sim["locker"], RECORD, puddles=PUDDLES).start()
    print(f"recorder: {RECORD}" + (f", {len(PUDDLES)} puddle(s) drawn" if PUDDLES else ""), flush=True)


def _finish(signum, frame):
    # Stopped with pkill (SIGTERM) or Ctrl-C: the run's truth (go2_world/score.py)
    # into the log, finish the video if one is being made, then go, as the
    # signal would have. The sim's threads are not daemons, so exit hard.
    for line in movers.score.summary():
        print(line, flush=True)
    if rec is not None:
        rec.close()
    os._exit(128 + signum)


signal.signal(signal.SIGTERM, _finish)
signal.signal(signal.SIGINT, _finish)

# mjpython closes the viewer when this script returns: wait for the sim's threads.
for t in threading.enumerate():
    if t is not threading.current_thread() and not t.daemon:
        t.join()
