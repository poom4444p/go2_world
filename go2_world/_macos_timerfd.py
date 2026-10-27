"""macOS stand-in for unitree_sdk2py.utils.timerfd (Linux-only timerfd(2)).

unitree_sdk2py's RecurrentThread reads 8 bytes from a timerfd each period. Here
the "timerfd" is the read end of a pipe, and a daemon thread writes 8 bytes to
it on a monotonic schedule. Import this before anything from unitree_sdk2py.
On Linux (the J4012) it does nothing and the SDK's real timerfd is used.
"""
import ctypes
import os
import sys
import threading
import time
import types

if sys.platform == "darwin":
    class timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    class itimerspec(ctypes.Structure):
        _fields_ = [("it_interval", timespec), ("it_value", timespec)]

        @classmethod
        def from_seconds(cls, interval, value):
            spec = cls()
            spec.it_interval.tv_sec = int(interval)
            spec.it_interval.tv_nsec = int((interval % 1) * 1e9)
            spec.it_value.tv_sec = int(value)
            spec.it_value.tv_nsec = int((value % 1) * 1e9)
            return spec

    _writers = {}

    def timerfd_create(clockid, flags):
        r, w = os.pipe()
        _writers[r] = w
        return r

    def timerfd_settime(fd, flags, new_value, old_value):
        spec = new_value._obj if hasattr(new_value, "_obj") else new_value
        period = spec.it_interval.tv_sec + spec.it_interval.tv_nsec / 1e9
        w = _writers[fd]

        def tick():
            nxt = time.monotonic() + period
            while True:
                time.sleep(max(0.0, nxt - time.monotonic()))
                nxt += period
                try:
                    os.write(w, b"\0" * 8)
                except OSError:          # reader closed: the loop has quit
                    return

        threading.Thread(target=tick, daemon=True).start()
        return 0

    shim = types.ModuleType("unitree_sdk2py.utils.timerfd")
    shim.timespec, shim.itimerspec = timespec, itimerspec
    shim.timerfd_create, shim.timerfd_settime = timerfd_create, timerfd_settime
    shim.__all__ = ["timespec", "itimerspec", "timerfd_create", "timerfd_settime"]
    sys.modules["unitree_sdk2py.utils.timerfd"] = shim
