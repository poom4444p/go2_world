"""Where the dog is, for the raycaster: one interface, two sources.

DdsPose reads the physics over DDS, the same topics go2_board's SimBus reads:
position from rt/sportmodestate, orientation and the IMU from rt/lowstate.
unitree_mujoco fills only position and velocity in rt/sportmodestate, not its
imu_state, so the attitude has to come from rt/lowstate.

StaticPose is a dog that stands still, for tests and for running the emulator
without the physics.

The physics runs slower than real time on a busy Mac (~0.83x), and its gyro
is in radians per simulated second, while the emulator -- and so the board's
IMU integration -- runs on the wall clock: a board turning the gyro into an
angle over-turned by 1/0.83, and the levelled floor rocked with it (1 deg of
the 2 deg left on HIL 11-10 11:05). run_world stamps rt/lowstate's `tick`
with the simulated time (install_tick), and Pose.rtf is simulated seconds per
wall second, measured over RTF_WINDOW_S, so the emulator can give the board
what a real-time IMU would measure.
"""
import collections
import sys
import threading
import time

import numpy as np

UPRIGHT = (1.0, 0.0, 0.0, 0.0)
G = 9.81
RTF_WINDOW_S = 1.0           # s of wall clock the physics' speed is measured over


def install_tick(bridge_cls):
    """unitree_mujoco's bridge: rt/lowstate's tick is the simulated time, ms."""
    publish = bridge_cls.PublishLowState

    def stamped(self):
        if self.mj_data is not None:
            self.low_state.tick = int(self.mj_data.time * 1000) & 0xFFFFFFFF
        publish(self)
    bridge_cls.PublishLowState = stamped


class Rate:
    """Simulated seconds per wall second, from (tick ms, wall s) pairs."""

    def __init__(self, window=RTF_WINDOW_S):
        self.window, self.seen = window, collections.deque()

    def add(self, tick_ms, wall):
        if self.seen and tick_ms < self.seen[-1][0]:     # the physics restarted
            self.seen.clear()
        self.seen.append((tick_ms, wall))
        while len(self.seen) > 2 and wall - self.seen[0][1] > self.window:
            self.seen.popleft()

    def __call__(self):
        if len(self.seen) < 2:
            return 1.0
        (k0, w0), (k1, w1) = self.seen[0], self.seen[-1]
        if w1 - w0 < 0.2 or k1 == k0:        # too little to tell, or no tick at all
            return 1.0
        return (k1 - k0) / 1000.0 / (w1 - w0)


class Pose:
    def __init__(self, pos, quat, gyro=(0.0, 0.0, 0.0), acc=(0.0, 0.0, G), stamp=0.0, rtf=1.0):
        self.pos = np.asarray(pos, float)        # world, m
        self.quat = np.asarray(quat, float)      # (w, x, y, z), trunk to world
        self.gyro = np.asarray(gyro, float)      # trunk frame, rad/s
        self.acc = np.asarray(acc, float)        # trunk frame, m/s^2, specific force
        self.stamp = stamp
        self.rtf = rtf                           # simulated s per wall s (Rate)


class StaticPose:
    def __init__(self, pos=(0.0, 0.0, 0.30), quat=UPRIGHT):
        self.pose = Pose(pos, quat, stamp=time.monotonic())

    def read(self):
        return self.pose


class DdsPose:
    def __init__(self, interface="lo0" if sys.platform == "darwin" else "lo", domain=0):
        if sys.platform == "darwin":
            from go2_world import _macos_timerfd  # noqa: F401  (before unitree_sdk2py)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

        ChannelFactoryInitialize(domain, interface)
        self._lock = threading.Lock()
        self._pos = None
        self._pose = None
        self._high = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self._high.Init(self._on_high, 10)
        self._rate = Rate()
        self._low = ChannelSubscriber("rt/lowstate", LowState_)
        self._low.Init(self._on_low, 10)

    def _on_high(self, msg):
        self._pos = np.array(msg.position, float)

    def _on_low(self, msg):
        if self._pos is None:
            return
        imu = msg.imu_state
        now = time.monotonic()
        self._rate.add(msg.tick, now)
        pose = Pose(self._pos, imu.quaternion, imu.gyroscope, imu.accelerometer,
                    now, self._rate())
        with self._lock:
            self._pose = pose

    def read(self):
        with self._lock:
            return self._pose

    def wait(self, timeout=5.0):
        end = time.monotonic() + timeout
        while self.read() is None:
            if time.monotonic() > end:
                raise TimeoutError("no rt/sportmodestate + rt/lowstate: is run_world.py running?")
            time.sleep(0.01)
        return self.read()
