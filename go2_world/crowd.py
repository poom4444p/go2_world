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
closer than MIN_GAP to the dog, and back to the lane once past it. "The
dog" is the party when its heading is known: the dog and the person on its
handle, HANDLER_BEHIND behind it -- people make room for both. Speed
and the side step change at ACCEL and SIDE_V: nobody jumps.

And they keep to the right of the corridor (KEEP_RIGHT), as people in Denmark
do: a lane's y is how far off the corridor's middle they walk, on their right
whichever way they are going, so somebody coming the other way passes on the
dog's left when it keeps right too. Turning round at a lane's end, they cross
over at SIDE_V as they walk back.
"""
import json
import math
import socket
import threading
import time

import numpy as np

PARKED = (0.0, 0.0, -20.0)
RADIUS, HEIGHT = 0.22, 1.70            # a shoulder's half-width, standing height
MOVERS_PORT = 9882                     # run_world -> run_lidar: people and the lift car
FLOOR_HEIGHT = 2.2
KEEP_RIGHT = True                      # people walk on their right (Denmark); False: fixed lanes

# Lanes in this (go2_world) frame: floor, y, x from, x to, walking speed.
# With KEEP_RIGHT the sign of y is ignored: it is how far off the middle.
# Each floor's corridor runs along x at y = 0 (walls at about +-1.4, from
# x -18 to 24); the lift is at the east end. Crates on the south half
# (floor 1 the trolley at x 2.4..3.6, floor 2 x -4.5 and 13..14, floor 3
# x 20.5) are gone round (BOX_LEAD). --people=N walks the first N lanes of
# EVERY floor: the dog meets N on whichever floor it is on (it was N in the
# whole building, most of them on floor 1: room 201 met one, HIL 11-19).
# Floor 1's lanes keep clear of the dog's spawn (0, 0) and its user behind it.
LANES = {
    1: [(1, 0.70, 6.0, 16.0, 1.0),
        (1, 0.70, 8.0, 20.0, 0.9),
        (1, 0.60, 12.0, 22.0, 0.8),
        (1, 0.60, -12.0, -3.0, 1.0),
        (1, 0.65, -17.0, -6.0, 0.9),
        (1, 0.65, 4.0, 14.0, 0.85)],
    2: [(2, 0.60, 6.0, 18.0, 1.0),
        (2, 0.65, -16.0, -4.0, 0.9),
        (2, 0.70, -8.0, 6.0, 1.0),
        (2, 0.60, 10.0, 22.0, 0.85),
        (2, 0.65, -14.0, 2.0, 0.8),
        (2, 0.70, 0.0, 14.0, 0.95)],
    3: [(3, 0.60, 15.0, 22.0, 0.8),
        (3, 0.65, -16.0, -4.0, 1.0),
        (3, 0.70, -8.0, 6.0, 0.9),
        (3, 0.60, 0.0, 12.0, 1.0),
        (3, 0.65, 6.0, 20.0, 0.85),
        (3, 0.70, -12.0, 2.0, 0.8)],
}
PER_FLOOR = min(len(v) for v in LANES.values())   # --people=N: at most this many a floor
EXTRA = 2                                         # bodies for --stand / --walker on top
MAX_PEOPLE = PER_FLOOR * len(LANES) + EXTRA       # bodies in the scene, all of them parked


def lanes(per_floor):
    """The lanes walked with `per_floor` people on every floor, floor by floor."""
    if not 0 <= per_floor <= PER_FLOOR:
        raise ValueError(f"--people is per floor, 0 to {PER_FLOOR}, not {per_floor}")
    return [lane for f in sorted(LANES) for lane in LANES[f][:per_floor]]


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
SIDE_M = 0.90                # m, the most a person steps aside (as the twin's SIDESTEP)
SIDE_V = 0.4                 # m/s sideways
WALL_KEEP = 0.10             # m kept from whatever is on that side
ROOM_H = (0.5, 1.0)          # m above their floor the room aside is looked for at
# How much room. People give a guide dog and its user what room the
# corridor has -- up to SIDE_M aside -- and never less than MIN_GAP. They
# stepped aside 0.45 m at most, passed at 0.25 m, closed to 0.10 m, and on
# the dog alone, its user not there at all: they walked along beside the
# user and pressed in on the dog, and the dog stood for them 16-23 s at a
# time (HIL PEOPLE=6, 11-17). Insisting on 0.5 m always is not a corridor
# either: in 2.8 m with the dog in the middle nobody could pass at all.
MIN_GAP = 0.30               # m, surface to surface: never nearer the party than this
PASS_SEP = RADIUS + DOG_R + MIN_GAP   # m across, centre to centre, to walk past the party
PASS_V = 0.4                 # m/s walking past it
WAIT_M = 1.2                 # m, centre to centre, at which one who cannot pass stops
# The party is the dog and the person on its handle, on one line: the user's
# centre HANDLER_BEHIND behind the dog's (cyberdog run_building.HANDLER_BEHIND).
# Given the dog's heading, people keep their room from the nearest point of
# that line, not from the dog alone.
HANDLER_BEHIND = 1.1         # m
ACCEL = 0.8                  # m/s^2, speeding up and slowing down
SAME_WAY_V = 0.2             # m/s of the dog along their way: it is going their way.
                             # Then they overtake it at their own pace, as people do,
                             # or, with no room to, follow at its pace: not hover
                             # behind it at PASS_V, which held them there until the
                             # dog stopped, and the dog stopped for them (HIL 11-10)
RETURN_M = 2.0               # m past the dog before an overtaker steps back into
                             # its lane: at BEHIND_M they cut in 1 m in front of it
ONCOMING_M = 6.0             # m ahead in the other half with somebody coming: no
                             # stepping out to overtake, follow instead
OVERTAKE_SIDE_M = 1.0        # m stepped out to overtake: over into the other half
OVERTAKE_TIE = 0.3           # m: a dog this near their line is passed on their left
GIVE_UP_S = 3.0              # s stopped for the dog before turning back: a person
                             # who cannot get past goes another way soon, rather than
                             # stand there while the dog stands for them (8 s was a
                             # stand-off, the dog asking "Excuse me" at them, HIL 11-17)


# Walls and crates. People only ever looked sideways, at the dog: they walked
# straight through the floor-3 cartons (the lane at y -0.6 crosses them), and
# stepping aside by a doorway, where the room reads as the whole room beyond,
# they walked on into the wall past it (HIL 11-17, and on main before it: a
# capsule up to 0.22 m into a wall in a headless run). So, given `wall` -- the
# metres free from a point in a direction, among walls and crates only:
BOX_LEAD = 1.2               # m ahead at which something static in their way is gone round
WALL_H = 0.5                 # m above their floor the walls and crates are looked for at
ROOM_AHEAD = (0.0, 0.6, 1.2)  # m along their way at which the room aside is measured


def _toward(a, b, step):
    return b if abs(b - a) <= step else a + math.copysign(step, b - a)


def party_point(dog, yaw, x, y):
    """The point of the party -- the dog, and its user HANDLER_BEHIND behind it
    -- nearest (x, y), as (x, y, z) with the dog's z. Without a heading
    (yaw None) the dog alone."""
    if dog is None or yaw is None:
        return dog
    bx, by = -math.cos(yaw), -math.sin(yaw)               # the way to the user
    s = (x - dog[0]) * bx + (y - dog[1]) * by
    s = min(max(s, 0.0), HANDLER_BEHIND)
    return (dog[0] + bx * s, dog[1] + by * s, dog[2])


class Walker:
    """Back and forth along one lane at a floor, making way for the dog."""

    def __init__(self, floor, y, x0, x1, speed, phase=0.0, keep_right=False):
        self.z = (floor - 1) * FLOOR_HEIGHT
        self.keep_right, self.off = keep_right, abs(y)
        self.y, self.x0, self.x1, self.speed = y, x0, x1, speed
        self.lo, self.hi = min(x0, x1), max(x0, x1)
        span = self.hi - self.lo
        s = (phase * 2 * span) % (2 * span) if span else 0.0   # distance walked, out and back
        out = math.copysign(1.0, x1 - x0)                        # x0 -> x1 first
        if s <= span:
            self.x, self.u = x0 + out * s, out
        else:
            self.x, self.u = x1 - out * (s - span), -out
        if keep_right:
            self.y = self.lane_y()
        self.v = speed               # m/s along the lane, now
        self.o = 0.0                 # m stepped aside from the lane, now
        self.waited = 0.0            # s stopped for the dog
        self.round_side = None       # the side a crate ahead is being gone round on
        self.round_front = None      # ...and the x of its near face

    def pos(self):
        return (self.x, self.y + self.o, self.z)

    def lane_y(self):
        """Their lane, on the right of their way along x: -y going +x."""
        return -self.u * self.off

    def aim(self, dog, room, dog_v=None, oncoming=None):
        """(speed, side step) wanted with the dog at `dog` (x, y, z), or None
        for no dog. room(side) -> metres free that way (+1: +y) from the lane.
        dog_v: its (vx, vy), m/s, or None for a dog standing still.
        oncoming(side) -> somebody coming the other way on that side, within
        ONCOMING_M: no stepping out to overtake into them."""
        if dog is None or not DOG_FLOOR_Z[0] <= dog[2] - self.z < DOG_FLOOR_Z[1]:
            return self.speed, 0.0
        ahead = (dog[0] - self.x) * self.u
        lat = dog[1] - (self.y + self.o)
        along = 0.0 if dog_v is None else dog_v[0] * self.u    # the lanes run along x
        same_way = along >= SAME_WAY_V
        passed = RETURN_M if same_way and self.o else BEHIND_M   # overtaken: clear of it first
        if not (-passed < ahead < NOTICE_M and abs(lat) < NOTICE_LAT):
            return self.speed, 0.0
        side = -1.0 if lat > 0 else 1.0 if lat < 0 else math.copysign(1.0, self.y or 1.0)
        if same_way and abs(lat) < OVERTAKE_TIE:
            side = self.u             # in its lane: overtake on their left (+y going +x)
        free = math.inf if room is None else room(side)
        reach = OVERTAKE_SIDE_M if same_way else SIDE_M
        if same_way and ahead > 0 and not self.o and oncoming is not None and oncoming(side):
            reach = 0.0               # not out into somebody coming: behind the dog
        o = side * max(0.0, min(reach, free - RADIUS - WALL_KEEP))
        if ahead <= 0 or abs(dog[1] - (self.y + o)) >= PASS_SEP:
            return (self.speed if same_way else min(self.speed, PASS_V)), o
        near = math.hypot(dog[0] - self.x, dog[1] - (self.y + self.o))
        if same_way:                  # no room to pass it: behind it, at its pace
            return (min(self.speed, along) if near < WAIT_M + DOG_R else self.speed), o
        return (0.0 if near < WAIT_M + DOG_R else min(self.speed, PASS_V)), o

    def tick(self, dt, dog=None, room=None, dog_v=None, oncoming=None, sim_dt=None,
             dog_yaw=None, wall=None):
        """Walk dt seconds -- or sim_dt, simulated ones, when given: the
        physics runs ~0.83x real time and the dog walks on it, so people on
        the wall clock walked, sped up, stepped aside and gave up ~20 % quick
        against it, and were judged against its qvel (simulated m/s) on the
        wrong clock. dog: its trunk (x, y, z) or None; room, dog_v,
        oncoming: as in aim(). dog_yaw: its heading, for the party (see
        HANDLER_BEHIND); None, the dog alone. wall(x, y, dx, dy): metres free
        from (x, y) that way among walls and crates (see BOX_LEAD); None,
        none looked for."""
        dt = dt if sim_dt is None else sim_dt
        dog = party_point(dog, dog_yaw, self.x, self.y + self.o)
        want_v, want_o = self.aim(dog, room, dog_v, oncoming)
        y0, o0 = self.y, self.o
        stepping_round = False
        if wall is not None:
            # A crate in their way, shoulder to shoulder: round it, stepping
            # over on one side (the roomier, then kept) until clear of it, at
            # PASS_V, standing short of it while they cannot yet get by.
            here = self.y + self.o
            ahead = min(wall(self.x, here + s * RADIUS, self.u, 0.0) for s in (-1.0, 0.0, 1.0))
            if ahead < BOX_LEAD:
                self.round_front = self.x + self.u * ahead   # where its near face is
                if self.round_side is None:
                    rooms = {s: wall(self.x, here, 0.0, s) for s in (-1.0, 1.0)}
                    self.round_side = max(rooms, key=rooms.get)
                s = self.round_side
                step = min(0.3, wall(self.x, here, 0.0, s) - RADIUS - WALL_KEEP)
                if step > 0.0:
                    want_o, stepping_round = self.o + s * step, True
                    want_v = min(want_v, PASS_V)
                if ahead < RADIUS + WALL_KEEP + 0.05:
                    want_v = 0.0
            elif (self.round_front is not None
                  and (self.x - self.round_front) * self.u < RADIUS + WALL_KEEP):
                # Clear of it only just: hold the side step until level with
                # it -- back to the lane now walks straight back into it. Past
                # its face, the guard below keeps them off its side.
                want_o, stepping_round = self.o, True
            else:
                self.round_side = self.round_front = None
        self.v = _toward(self.v, want_v, ACCEL * dt)
        o = _toward(self.o, want_o, SIDE_V * dt)
        x = self.x + self.u * self.v * dt
        if x > self.hi or x < self.lo:              # the lane's end: turn round there
            x = min(max(x, self.lo), self.hi)
            self.u = -self.u
        if self.keep_right:                         # over to the right of the way back
            self.y = _toward(self.y, self.lane_y(), SIDE_V * dt)
        if wall is not None:
            # Never into a wall or a crate: not sideways, and not along.
            dy = (self.y + o) - (y0 + o0)
            if dy and min(wall(self.x + s * RADIUS, y0 + o0, 0.0, math.copysign(1.0, dy))
                          for s in (-1.0, 0.0, 1.0)) < RADIUS + abs(dy):
                self.y, o = y0, o0                  # (front and back too: a crate's corner)
            dx = x - self.x
            if dx and min(wall(self.x, self.y + o + s * RADIUS, math.copysign(1.0, dx), 0.0)
                          for s in (-1.0, 0.0, 1.0)) < RADIUS + abs(dx):
                x, self.v = self.x, 0.0
        if dog is not None and DOG_FLOOR_Z[0] <= dog[2] - self.z < DOG_FLOOR_Z[1]:
            touch = RADIUS + DOG_R + MIN_GAP
            now = math.hypot(dog[0] - self.x, dog[1] - (self.y + self.o))
            if math.hypot(dog[0] - x, dog[1] - (self.y + self.o)) < min(touch, now):
                x, self.v = self.x, 0.0             # a step along would close on it: stand
            if math.hypot(dog[0] - x, dog[1] - (self.y + o)) < min(touch, now):
                o = self.o                          # so would the step aside
        self.x, self.o = x, o
        if want_v == 0.0 and self.v < 0.05 and not stepping_round:
            self.waited += dt
            if self.waited > GIVE_UP_S:             # it is not moving: go back the other way
                self.u, self.waited = -self.u, 0.0
        else:
            self.waited = 0.0
        return self.pos()


class Stander:
    """Somebody standing still where they are put -- on the phone, reading a
    sign -- who does not make way for the dog, for `stay` seconds (None: for
    good) and then is gone. A test of what the dog does about a person who
    will not move, which the walkers, who always do, never are (--stand)."""

    def __init__(self, x, y, floor=1, stay=None):
        self.x, self.y, self.z = x, y, (floor - 1) * FLOOR_HEIGHT
        self.stay, self.t = stay, 0.0

    def pos(self):
        if self.stay is not None and self.t > self.stay:
            return PARKED
        return (self.x, self.y, self.z)

    def tick(self, dt, *_, **__):
        self.t += dt
        return self.pos()


APPEAR_M = 6.0               # m behind a slow walker's start the dog is when they set off


class SlowWalker:
    """Somebody walking the dog's way ahead of it, slower than it, in its lane:
    for the brain to follow or overtake (--walker). The crowd's walkers are
    0.8-1.0 m/s, faster than the dog, and overtake it instead.

    Parked until the dog, on their floor, is within APPEAR_M behind x0 (so
    they set off ahead of it whenever it gets there); then along y from x0 to
    x1 at v, and gone at x1. They do not make way -- people walking ahead do
    not see the dog behind them -- but never step into it (MIN_GAP), should
    it cut in front. On simulated time (sim_dt), as the crowd: the physics
    runs ~0.83x real time, and a speed is set to be slower than the dog by
    so much."""

    def __init__(self, x0, v, x1=18.0, y=-0.6, floor=1):
        self.x0, self.x1, self.y, self.v = x0, x1, y, v
        self.z = (floor - 1) * FLOOR_HEIGHT
        self.u = math.copysign(1.0, x1 - x0)
        self.x, self.state = x0, "waiting"           # waiting -> walking -> gone

    def pos(self):
        return (self.x, self.y, self.z) if self.state == "walking" else PARKED

    def tick(self, dt, dog=None, sim_dt=None):
        """dt: wall seconds; sim_dt: simulated ones, used when given.
        dog: its trunk (x, y, z) or None."""
        here = dog is not None and DOG_FLOOR_Z[0] <= dog[2] - self.z < DOG_FLOOR_Z[1]
        if self.state == "waiting":
            if here and 0.0 < (self.x0 - dog[0]) * self.u < APPEAR_M:
                self.state = "walking"
            return self.pos()
        if self.state == "walking":
            x = self.x + self.u * self.v * (dt if sim_dt is None else sim_dt)
            if here:
                touch = RADIUS + DOG_R + MIN_GAP
                now = math.hypot(dog[0] - self.x, dog[1] - self.y)
                if math.hypot(dog[0] - x, dog[1] - self.y) < min(touch, now):
                    x = self.x                       # a step would close on it: stand
            self.x = x
            if (self.x - self.x1) * self.u >= 0.0:
                self.state = "gone"
        return self.pos()


class Movers:
    """Walks the people in the physics and tells the emulator where they and
    the lift car are, 20 times a second; scores the run's truth as it goes
    (.score, go2_world/score.py)."""

    def __init__(self, model, data, lock, per_floor, lift=None, emulator=("127.0.0.1", MOVERS_PORT),
                 standers=(), walls_xml=None):
        walk = lanes(per_floor)                 # per_floor on every floor (see LANES)
        if len(walk) + len(standers) > MAX_PEOPLE:
            raise ValueError(f"at most {MAX_PEOPLE} people, walking and standing")
        self.model, self.data, self.lock, self.lift = model, data, lock, lift
        # Walls and crates alone, for wall(): the building as written, no dog,
        # no people, no car. Its own model, so nothing moving is ever in it.
        import mujoco
        from go2_world.scene import BUILDING
        self.walls = mujoco.MjModel.from_xml_path(walls_xml or BUILDING)
        self.walls_data = mujoco.MjData(self.walls)
        mujoco.mj_forward(self.walls, self.walls_data)
        self._wgeom = np.zeros(1, np.int32)
        self.mocap = [int(model.body(f"person_{k}").mocapid[0]) for k in range(MAX_PEOPLE)]
        self.body = [int(model.body(f"person_{k}").id) for k in range(MAX_PEOPLE)]
        self._geom = np.zeros(1, np.int32)
        self.walkers = [Walker(*lane, phase=0.37 * k, keep_right=KEEP_RIGHT)
                        for k, lane in enumerate(walk)]
        self.standers = list(standers)          # bodies after the walkers' (and
                                                # slow walkers: anything with tick/pos)
        self.emulator = emulator
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        from go2_world.score import Score
        self.score = Score(model, log=lambda line: print(line, flush=True))

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def state(self):
        return {"people": [list(w.pos()) for w in self.walkers + self.standers],
                "lift_z": None if self.lift is None else self.lift.z}

    def dog(self):
        """The dog's trunk (x, y, z): the scene's first joint is its free joint."""
        return tuple(float(v) for v in self.data.qpos[0:3])

    def dog_v(self):
        """Its (vx, vy), world frame, m/s: the free joint's linear velocity."""
        return tuple(float(v) for v in self.data.qvel[0:2])

    def dog_yaw(self):
        """Its heading, rad: from the free joint's quaternion (w, x, y, z)."""
        w, x, y, z = (float(v) for v in self.data.qpos[3:7])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def room(self, k, w, side):
        """Metres free from walker k's lane sideways (+1: +y), at ROOM_H:
        walls, the trolley, the dog and other people all count."""
        import mujoco
        best = math.inf
        # Here and ahead along their way (ROOM_AHEAD): a doorway beside them is
        # not room to step into when the wall is back a step further on.
        for a in ROOM_AHEAD:
            for h in ROOM_H:
                d = mujoco.mj_ray(self.model, self.data, np.array([w.x + a * w.u, w.y, w.z + h]),
                                  np.array([0.0, side, 0.0]), None, 1, self.body[k], self._geom)
                if d >= 0:
                    best = min(best, d)
        return best

    def wall(self, w, x, y, dx, dy):
        """Metres free from (x, y) at walker w's floor, towards (dx, dy), among
        the walls and crates alone (see BOX_LEAD); inf for nothing."""
        import mujoco
        d = mujoco.mj_ray(self.walls, self.walls_data, np.array([x, y, w.z + WALL_H]),
                          np.array([dx, dy, 0.0]), None, 1, -1, self._wgeom)
        return math.inf if d < 0 else d

    def oncoming(self, k, w, side):
        """Somebody walking against walker k, on its `side` (+1: +y) of it,
        within ONCOMING_M ahead of it."""
        for j, v in enumerate(self.walkers):
            if j == k or v.z != w.z or v.u != -w.u:
                continue
            if 0.0 < (v.x - w.x) * w.u < ONCOMING_M and (v.y + v.o - w.y) * side > 0.3:
                return True
        return False

    def _run(self):
        last, sent = time.monotonic(), 0.0
        last_sim = None
        while True:
            time.sleep(0.01)
            now = time.monotonic()
            dt, last = now - last, now
            with self.lock:
                t_sim = float(self.data.time)
                sim_dt = 0.0 if last_sim is None else max(0.0, t_sim - last_sim)
                last_sim = t_sim
                # The truth first, on the physics as it stands (go2_world/score.py).
                self.score.tick(self.data, sim_dt, [self.data.mocap_pos[m] for m in self.mocap])
                dog, dog_v, yaw = self.dog(), self.dog_v(), self.dog_yaw()
                for k, w in enumerate(self.walkers):
                    self.data.mocap_pos[self.mocap[k]] = w.tick(
                        dt, dog, lambda side, k=k, w=w: self.room(k, w, side), dog_v,
                        lambda side, k=k, w=w: self.oncoming(k, w, side), sim_dt=sim_dt,
                        dog_yaw=yaw,
                        wall=lambda x, y, dx, dy, w=w: self.wall(w, x, y, dx, dy))
                for j, s in enumerate(self.standers):
                    self.data.mocap_pos[self.mocap[len(self.walkers) + j]] = s.tick(
                        dt, dog, sim_dt=sim_dt)
            if now - sent >= 0.05:
                sent = now
                try:
                    self.sock.sendto(json.dumps(self.state()).encode(), self.emulator)
                except OSError:
                    pass
