"""People walking in the building: solid to the dog, visible to the Mid-360.

Each is a capsule on a mocap body, walked back and forth along a lane of a
corridor by a thread, so the dog can bump into them in the physics and the
LiDAR emulator (which gets their positions, and the lift car's height, from
run_world over UDP on MOVERS_PORT) sees them as a real unit would.

There are always MAX_PEOPLE bodies in the scene, physics and raycaster alike,
so the two models match whatever the count: the ones not walking are parked
under the ground.

They make way for the dog, as people in a corridor do. A mocap body is
infinitely strong: one that walks on regardless pushes the dog and pins it
on the walls, and the dog's brain waits it out (about 115 s of a HIL route
went on that). So a person who sees the dog ahead (NOTICE_M, same floor,
near their line) steps away from it (up to SIDE_M, as far as the room on
that side allows), and walks past slowly if that leaves PASS_SEP between
them; else stops short and waits, and after GIVE_UP_S turns back. Never
closer than MIN_GAP to the dog, and back to the lane once past it. Speed
and the side step change at ACCEL and SIDE_V: nobody jumps.
"""
import json
import math
import socket
import threading
import time

import numpy as np

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


# Making way for the dog (see the module's doc).
DOG_R = 0.35                 # m, the Go2's half-length: its reach from the trunk centre
DOG_FLOOR_Z = (0.0, 1.5)     # trunk height above a person's floor that is "this floor"
NOTICE_M = 4.0               # m ahead along the lane at which the dog is seen
NOTICE_LAT = 1.6             # m either side of the person: the corridor, roughly
BEHIND_M = 0.5               # m behind them: passed, back to the lane
SIDE_M = 0.45                # m, the most a person steps aside
SIDE_V = 0.4                 # m/s sideways
WALL_KEEP = 0.10             # m kept from whatever is on that side
ROOM_H = (0.5, 1.0)          # m above their floor the room aside is looked for at
PASS_SEP = RADIUS + DOG_R + 0.25   # m across, centre to centre, to walk past the dog
PASS_V = 0.4                 # m/s walking past it
WAIT_M = 1.2                 # m, centre to centre, at which one who cannot pass stops
MIN_GAP = 0.10               # m, surface to surface: never nearer the dog than this
ACCEL = 0.8                  # m/s^2, speeding up and slowing down
GIVE_UP_S = 8.0              # s stopped for the dog before turning back


def _toward(a, b, step):
    return b if abs(b - a) <= step else a + math.copysign(step, b - a)


class Walker:
    """Back and forth along one lane at a floor, making way for the dog."""

    def __init__(self, floor, y, x0, x1, speed, phase=0.0):
        self.z = (floor - 1) * FLOOR_HEIGHT
        self.y, self.x0, self.x1, self.speed = y, x0, x1, speed
        self.lo, self.hi = min(x0, x1), max(x0, x1)
        span = self.hi - self.lo
        s = (phase * 2 * span) % (2 * span) if span else 0.0   # distance walked, out and back
        out = math.copysign(1.0, x1 - x0)                        # x0 -> x1 first
        if s <= span:
            self.x, self.u = x0 + out * s, out
        else:
            self.x, self.u = x1 - out * (s - span), -out
        self.v = speed               # m/s along the lane, now
        self.o = 0.0                 # m stepped aside from the lane, now
        self.waited = 0.0            # s stopped for the dog

    def pos(self):
        return (self.x, self.y + self.o, self.z)

    def aim(self, dog, room):
        """(speed, side step) wanted with the dog at `dog` (x, y, z), or None
        for no dog. room(side) -> metres free that way (+1: +y) from the lane."""
        if dog is None or not DOG_FLOOR_Z[0] <= dog[2] - self.z < DOG_FLOOR_Z[1]:
            return self.speed, 0.0
        ahead = (dog[0] - self.x) * self.u
        lat = dog[1] - (self.y + self.o)
        if not (-BEHIND_M < ahead < NOTICE_M and abs(lat) < NOTICE_LAT):
            return self.speed, 0.0
        side = -1.0 if lat > 0 else 1.0 if lat < 0 else math.copysign(1.0, self.y or 1.0)
        free = math.inf if room is None else room(side)
        o = side * max(0.0, min(SIDE_M, free - RADIUS - WALL_KEEP))
        if ahead <= 0 or abs(dog[1] - (self.y + o)) >= PASS_SEP:
            return min(self.speed, PASS_V), o
        near = math.hypot(dog[0] - self.x, dog[1] - (self.y + self.o))
        return (0.0 if near < WAIT_M + DOG_R else min(self.speed, PASS_V)), o

    def tick(self, dt, dog=None, room=None):
        """Walk dt seconds. dog: its trunk (x, y, z) or None; room: as in aim()."""
        want_v, want_o = self.aim(dog, room)
        self.v = _toward(self.v, want_v, ACCEL * dt)
        o = _toward(self.o, want_o, SIDE_V * dt)
        x = self.x + self.u * self.v * dt
        if x > self.hi or x < self.lo:              # the lane's end: turn round there
            x = min(max(x, self.lo), self.hi)
            self.u = -self.u
        if dog is not None and DOG_FLOOR_Z[0] <= dog[2] - self.z < DOG_FLOOR_Z[1]:
            touch = RADIUS + DOG_R + MIN_GAP
            now = math.hypot(dog[0] - self.x, dog[1] - (self.y + self.o))
            if math.hypot(dog[0] - x, dog[1] - (self.y + self.o)) < min(touch, now):
                x, self.v = self.x, 0.0             # a step along would close on it: stand
            if math.hypot(dog[0] - x, dog[1] - (self.y + o)) < min(touch, now):
                o = self.o                          # so would the step aside
        self.x, self.o = x, o
        if want_v == 0.0 and self.v < 0.05:
            self.waited += dt
            if self.waited > GIVE_UP_S:             # it is not moving: go back the other way
                self.u, self.waited = -self.u, 0.0
        else:
            self.waited = 0.0
        return self.pos()


class Movers:
    """Walks the people in the physics and tells the emulator where they and
    the lift car are, 20 times a second."""

    def __init__(self, model, data, lock, n_people, lift=None, emulator=("127.0.0.1", MOVERS_PORT)):
        if n_people > MAX_PEOPLE:
            raise ValueError(f"at most {MAX_PEOPLE} people")
        self.model, self.data, self.lock, self.lift = model, data, lock, lift
        self.mocap = [int(model.body(f"person_{k}").mocapid[0]) for k in range(MAX_PEOPLE)]
        self.body = [int(model.body(f"person_{k}").id) for k in range(MAX_PEOPLE)]
        self._geom = np.zeros(1, np.int32)
        self.walkers = [Walker(*ROUTES[k], phase=0.37 * k) for k in range(n_people)]
        self.emulator = emulator
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def state(self):
        return {"people": [list(w.pos()) for w in self.walkers],
                "lift_z": None if self.lift is None else self.lift.z}

    def dog(self):
        """The dog's trunk (x, y, z): the scene's first joint is its free joint."""
        return tuple(float(v) for v in self.data.qpos[0:3])

    def room(self, k, w, side):
        """Metres free from walker k's lane sideways (+1: +y), at ROOM_H:
        walls, the trolley, the dog and other people all count."""
        import mujoco
        best = math.inf
        for h in ROOM_H:
            d = mujoco.mj_ray(self.model, self.data, np.array([w.x, w.y, w.z + h]),
                              np.array([0.0, side, 0.0]), None, 1, self.body[k], self._geom)
            if d >= 0:
                best = min(best, d)
        return best

    def _run(self):
        last, sent = time.monotonic(), 0.0
        while True:
            time.sleep(0.01)
            now = time.monotonic()
            dt, last = now - last, now
            with self.lock:
                dog = self.dog()
                for k, w in enumerate(self.walkers):
                    self.data.mocap_pos[self.mocap[k]] = w.tick(
                        dt, dog, lambda side, k=k, w=w: self.room(k, w, side))
            if now - sent >= 0.05:
                sent = now
                try:
                    self.sock.sendto(json.dumps(self.state()).encode(), self.emulator)
                except OSError:
                    pass
