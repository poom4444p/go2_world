"""The fake Mid-360: rays into the building from the pose on DDS, out as Livox UDP.

  python scripts/run_lidar.py                       # pose from run_world.py
  python scripts/run_lidar.py --static              # no physics: a dog standing at the origin
  python scripts/run_lidar.py --dense --noise 0.02 --drop 0.01 --delay-ms 5 --jitter-ms 3
  python scripts/run_lidar.py --host <J4012 IP>      # to the J4012 over Ethernet
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2_world.emulator import Emulator, Impairments, UdpOut  # noqa: E402
from go2_world.raycast import Raycaster  # noqa: E402
from go2_world.scene import BUILDING  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="where the points go (the board)")
    ap.add_argument("--dense", action="store_true", help="the real unit's ~200k points/s")
    ap.add_argument("--static", action="store_true", help="no DDS: stand at the origin")
    ap.add_argument("--interface", default="lo0" if sys.platform == "darwin" else "lo")
    ap.add_argument("--noise", type=float, default=0.0, help="range noise, m (1 sigma)")
    ap.add_argument("--drop", type=float, default=0.0, help="packet drop probability")
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=float("inf"))
    a = ap.parse_args()

    ray = Raycaster(BUILDING, "dense" if a.dense else "cyberdog")
    if a.static:
        from go2_world.pose import StaticPose
        pose = StaticPose()
    else:
        from go2_world.pose import DdsPose
        pose = DdsPose(a.interface)
        pose.wait()
    emu = Emulator(ray, pose, UdpOut(a.host),
                   Impairments(a.noise, a.drop, a.delay_ms, a.jitter_ms, a.seed))
    print(f"mid360: {ray.n} rays/frame -> {a.host}:56301 points, :56401 imu")
    t = time.monotonic()
    try:
        emu.run(a.seconds)
    except KeyboardInterrupt:
        pass
    dt = time.monotonic() - t
    print(f"mid360: {emu.sent} packets in {dt:.1f} s, {emu.dropped} dropped on purpose")


if __name__ == "__main__":
    main()
