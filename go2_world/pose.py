"""Where the dog is, for the raycaster: one interface, two sources.

DdsPose reads the physics over DDS, the same topics go2_board's SimBus reads:
position from rt/sportmodestate, orientation and the IMU from rt/lowstate.
unitree_mujoco fills only position and velocity in rt/sportmodestate, not its
imu_state, so the attitude has to come from rt/lowstate.

StaticPose is a dog that stands still, for tests and for running the emulator
without the physics.
"""
import sys
import threading
import time

import numpy as np

UPRIGHT = (1.0, 0.0, 0.0, 0.0)
G = 9.81


class Pose:
    def __init__(self, pos, quat, gyro=(0.0, 0.0, 0.0), acc=(0.0, 0.0, G), stamp=0.0):
        self.pos = np.asarray(pos, float)        # world, m
        self.quat = np.asarray(quat, float)      # (w, x, y, z), trunk to world
        self.gyro = np.asarray(gyro, float)      # trunk frame, rad/s
        self.acc = np.asarray(acc, float)        # trunk frame, m/s^2, specific force
        self.stamp = stamp


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
        self._low = ChannelSubscriber("rt/lowstate", LowState_)
        self._low.Init(self._on_low, 10)

    def _on_high(self, msg):
        self._pos = np.array(msg.position, float)

    def _on_low(self, msg):
        if self._pos is None:
            return
        imu = msg.imu_state
        pose = Pose(self._pos, imu.quaternion, imu.gyroscope, imu.accelerometer,
                    time.monotonic())
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
