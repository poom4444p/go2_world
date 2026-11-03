"""People walking in the building: solid to the dog, visible to the Mid-360.

Each is a capsule on a mocap body, walked back and forth along a lane of a
corridor by a thread, so the dog can bump into them in the physics and the
LiDAR emulator (which gets their positions, and the lift car's height, from
run_world over UDP on MOVERS_PORT) sees them as a real unit would.

There are always MAX_PEOPLE bodies in the scene, physics and raycaster alike,
so the two models match whatever the count: the ones not walking are parked
under the ground.
"""
import json
import math
import socket
import threading
import time

MAX_PEOPLE = 6
PARKED = (0.0, 0.0, -20.0)
RADIUS, HEIGHT = 0.22, 1.70            # a shoulder's half-width, standing height
MOVERS_PORT = 9882                     # run_world -> run_lidar: people and the lift car
FLOOR_HEIGHT = 2.2

# Lanes in this (go2_world) frame: floor, y, x from, x to, walking speed.
# Floor 1's corridor runs along x at y = 0 (walls at about +-1.4); the
# trolley stands at x 2.4..3.6 on the south half; the lift is at the east end.
ROUTES = [
    (1, +0.70, 6.0, 16.0, 1.0),
    (1, -0.70, 8.0, 20.0, 0.9),
    (1, +0.60, 12.0, 22.0, 0.8),
    (3, -0.60, 15.0, 23.0, 0.8),
    (2, +0.60, 6.0, 18.0, 1.0),
    (1, -0.60, -12.0, -2.0, 1.0),
]


def people_xml(n=MAX_PEOPLE):
    """Mocap bodies person_0.. (parked), for the building's worldbody."""
    out = []
    for k in range(n):
        out.append(f'    <body name="person_{k}" mocap="true" pos="{PARKED[0]} {PARKED[1]} {PARKED[2]}">\n'
                   f'      <geom name="person_{k}" type="capsule" '
                   f'fromto="0 0 {RADIUS} 0 0 {HEIGHT - RADIUS}" size="{RADIUS}" '
                   f'rgba="0.30 0.45 0.75 1"/>\n    </body>')
    return "\n".join(out)


class Walker:
    """Back and forth along one lane: x(t) as a triangle wave, at a floor."""

    def __init__(self, floor, y, x0, x1, speed, phase=0.0):
        self.z = (floor - 1) * FLOOR_HEIGHT
        self.y, self.x0, self.x1, self.speed = y, x0, x1, speed
        self.s = phase * 2 * abs(x1 - x0)     # distance walked along the out-and-back

    def tick(self, dt):
        self.s = (self.s + self.speed * dt) % (2 * abs(self.x1 - self.x0))
        return self.pos()

    def pos(self):
        span = abs(self.x1 - self.x0)
        d = self.s if self.s <= span else 2 * span - self.s
        return (self.x0 + math.copysign(d, self.x1 - self.x0), self.y, self.z)


class Movers:
    """Walks the people in the physics and tells the emulator where they and
    the lift car are, 20 times a second."""

    def __init__(self, model, data, lock, n_people, lift=None, emulator=("127.0.0.1", MOVERS_PORT)):
        if n_people > MAX_PEOPLE:
            raise ValueError(f"at most {MAX_PEOPLE} people")
        self.data, self.lock, self.lift = data, lock, lift
        self.mocap = [int(model.body(f"person_{k}").mocapid[0]) for k in range(MAX_PEOPLE)]
        self.walkers = [Walker(*ROUTES[k], phase=0.37 * k) for k in range(n_people)]
        self.emulator = emulator
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def state(self):
        return {"people": [list(w.pos()) for w in self.walkers],
                "lift_z": None if self.lift is None else self.lift.z}

    def _run(self):
        last, sent = time.monotonic(), 0.0
        while True:
            time.sleep(0.01)
            now = time.monotonic()
            dt, last = now - last, now
            with self.lock:
                for k, w in enumerate(self.walkers):
                    self.data.mocap_pos[self.mocap[k]] = w.tick(dt)
            if now - sent >= 0.05:
                sent = now
                try:
                    self.sock.sendto(json.dumps(self.state()).encode(), self.emulator)
                except OSError:
                    pass
