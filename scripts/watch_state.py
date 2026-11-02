"""Print the Go2's body height and a few joint angles from DDS, once a second.

  python scripts/watch_state.py [interface] [seconds]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2_world import _macos_timerfd  # noqa: E402,F401  (before unitree_sdk2py)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_  # noqa: E402

iface = sys.argv[1] if len(sys.argv) > 1 else "lo0"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 15
ChannelFactoryInitialize(0, iface)

last = {}
low = ChannelSubscriber("rt/lowstate", LowState_)
low.Init(lambda m: last.__setitem__("low", m), 10)
high = ChannelSubscriber("rt/sportmodestate", SportModeState_)
high.Init(lambda m: last.__setitem__("high", m), 10)

t0 = time.time()
while time.time() - t0 < seconds:
    time.sleep(1)
    h, l = last.get("high"), last.get("low")
    if h is None or l is None:
        print(f"t={time.time() - t0:4.1f}s  no state yet (high={h is not None}, low={l is not None})")
        continue
    q = [round(l.motor_state[i].q, 2) for i in (0, 1, 2)]
    print(f"t={time.time() - t0:4.1f}s  body z={h.position[2]:.3f} m  FR leg q={q}")
