"""The lift car, moving: what carries the dog between floors in the sim.

In the building the car is a real thing the dog stands in while it moves; the
dog does nothing but stand. Here it is a mocap body (scene.py takes the car's
floor and roof out of the frozen building and puts them on one), moved in z
at RIDE_V by a thread, so the dog is lifted by contact through the physics,
as on the real lift, not placed.

Who presses the button is outside the robot: in the building a person (or
the building's lift API), here a UDP datagram on LIFT_PORT, one JSON object:
    {"call": n}   send the car to floor n
    {}            just report
and every datagram gets the car's state back:
    {"z": m, "floor": n or null (between floors), "target": n, "moving": bool}
"""
import json
import socket
import threading
import time

# Cyberdog's building (scene/levels.py, scene/lift.py): the same numbers, so
# the car stops where the slabs are.
FLOOR_HEIGHT = 2.2
FLOORS = (1, 2, 3)
RIDE_V = 0.6                 # m/s
# A lift eases in and out. Starting down at full speed dropped the car away
# from the dog's feet for an instant and tipped it 13 deg; at 0.5 m/s^2 the
# ride is the gentle thing a passenger expects, up or down.
ACCEL = 0.5                  # m/s^2
LIFT_PORT = 9880
STEP_S = 0.002
AT_FLOOR_M = 0.005           # nearer than this to a floor's height: at that floor


def floor_z(n):
    return (n - 1) * FLOOR_HEIGHT


class Lift:
    """The car's height and where it is going. Pure: tick(dt) moves it."""

    def __init__(self, floor=1):
        self.z = floor_z(floor)
        self.v = 0.0
        self.target = floor

    def call(self, floor):
        if floor not in FLOORS:
            raise ValueError(f"no floor {floor}: the lift serves {FLOORS}")
        self.target = floor

    def tick(self, dt):
        """Toward the target floor: speed limited to RIDE_V, changed at most
        ACCEL, and slowed in time to stop on the floor (v^2 = 2 a d)."""
        to_go = floor_z(self.target) - self.z
        if abs(to_go) < 1e-6 and abs(self.v) < 1e-6:
            self.z, self.v = floor_z(self.target), 0.0
            return self.z
        sign = 1.0 if to_go > 0 else -1.0
        want = sign * min(RIDE_V, (2 * ACCEL * abs(to_go)) ** 0.5)
        dv = max(-ACCEL * dt, min(ACCEL * dt, want - self.v))
        self.v += dv
        step = self.v * dt
        if abs(step) >= abs(to_go) and step * to_go >= 0:      # arrives this tick
            self.z, self.v = floor_z(self.target), 0.0
        else:
            self.z += step
        return self.z

    @property
    def moving(self):
        return abs(floor_z(self.target) - self.z) > 1e-9 or abs(self.v) > 1e-9

    @property
    def floor(self):
        for n in FLOORS:
            if abs(floor_z(n) - self.z) < AT_FLOOR_M:
                return n
        return None

    def state(self):
        return {"z": round(self.z, 4), "floor": self.floor, "target": self.target,
                "moving": self.moving}


class LiftServer:
    """Moves the mocap car in the running sim and answers on LIFT_PORT."""

    def __init__(self, model, data, lock, host="127.0.0.1", port=LIFT_PORT, body="lift_car"):
        self.data, self.lock = data, lock
        self.mocap = int(model.body(body).mocapid[0])
        self.base = data.mocap_pos[self.mocap].copy()       # the car at floor 1
        self.lift = Lift()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(STEP_S)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        last = time.monotonic()
        while True:
            try:
                data, peer = self.sock.recvfrom(1024)
                try:
                    msg = json.loads(data)
                    if "call" in msg:
                        self.lift.call(int(msg["call"]))
                        print(f"lift: called to floor {self.lift.target}", flush=True)
                    reply = self.lift.state()
                except (ValueError, TypeError) as e:
                    reply = {"error": str(e), **self.lift.state()}
                self.sock.sendto(json.dumps(reply).encode(), peer)
            except socket.timeout:
                pass
            now = time.monotonic()
            # A late wake-up is not made up: the physics sees only the newest
            # height, so catching up would be a jump into the dog's feet. The
            # car moves at most 2 * STEP_S worth (2.4 mm) a wake-up, and a ride
            # on a busy machine takes a little longer instead.
            dt, last = now - last, now
            z = self.lift.tick(min(dt, 2 * STEP_S))
            with self.lock:
                self.data.mocap_pos[self.mocap] = self.base + (0.0, 0.0, z)
