"""People making way for the dog (go2_world/crowd.py)."""
import math
import threading

import mujoco
import pytest

from go2_world import crowd
from go2_world.crowd import DOG_R, MIN_GAP, RADIUS, Movers, Walker

DT = 0.01
DOG_Z = 0.30                     # the trunk standing on floor 1


def walk(w, seconds, dog=None, room=None):
    """Tick w for `seconds`; the positions it went through."""
    return [w.tick(DT, dog, room) for _ in range(round(seconds / DT))]


def gap(p, dog):
    return math.hypot(p[0] - dog[0], p[1] - dog[1]) - RADIUS - DOG_R


def test_without_the_dog_still_back_and_forth_at_its_speed():
    w = Walker(1, 0.7, 6.0, 16.0, 1.0)
    xs = [p[0] for p in walk(w, 25.0)]
    assert min(xs) >= 6.0 and max(xs) <= 16.0 and max(xs) > 15.9 and min(xs) < 6.1
    assert all(abs(b - a) <= 1.0 * DT + 1e-9 for a, b in zip(xs, xs[1:]))
    assert {p[1] for p in walk(w, 1.0)} == {0.7}


def test_phase_starts_part_way_along_and_on_the_way_back():
    w = Walker(1, 0.0, 0.0, 10.0, 1.0, phase=0.75)     # 15 m of 20: on the way back
    assert w.pos()[0] == pytest.approx(5.0) and w.u == -1.0


def test_steps_away_from_the_dog_and_walks_past_slowly():
    dog = (10.0, 0.0, DOG_Z)                          # standing in the middle of the corridor
    w = Walker(1, 0.5, 6.0, 16.0, 1.0)
    path = walk(w, 12.0, dog, room=lambda side: 0.9)  # 0.9 m free either way
    alongside = [p for p in path if abs(p[0] - dog[0]) < 0.3]
    assert alongside, "it never got past the dog"
    assert min(p[1] for p in alongside) >= 0.5 + crowd.SIDE_M - 1e-6     # stepped away (+y)
    assert all(gap(p, dog) >= MIN_GAP - 1e-6 for p in path)
    speeds = [abs(b[0] - a[0]) / DT for a, b in zip(path, path[1:]) if abs(a[0] - dog[0]) < 0.3]
    assert max(speeds) <= crowd.PASS_V + 1e-6
    assert path[-1][1] == pytest.approx(0.5)          # past it: back on the lane


def test_the_side_step_is_only_as_far_as_the_room():
    dog = (10.0, 0.0, DOG_Z)
    w = Walker(1, 0.7, 6.0, 16.0, 1.0)
    path = walk(w, 4.0, dog, room=lambda side: 0.5 if side > 0 else 3.0)
    assert max(p[1] for p in path) <= 0.7 + 0.5 - RADIUS - crowd.WALL_KEEP + 1e-9


def test_no_room_to_pass_stops_short_then_turns_back():
    dog = (10.0, 0.3, DOG_Z)                          # the dog on the person's line, no room
    w = Walker(1, 0.3, 6.0, 16.0, 1.0)
    path = walk(w, 7.0, dog, room=lambda side: RADIUS)    # slows, stops, stands
    stop = path[-1]
    assert stop[0] < dog[0] and gap(stop, dog) >= MIN_GAP
    assert walk(w, 0.5, dog, lambda side: RADIUS)[-1][0] == pytest.approx(stop[0], abs=1e-6)
    later = walk(w, crowd.GIVE_UP_S + 3.0, dog, lambda side: RADIUS)
    assert min(p[0] for p in later) < stop[0] - 0.5                 # went back the way it came
    assert all(gap(p, dog) >= MIN_GAP - 1e-6 for p in path + later)


def test_never_closes_on_the_dog_even_when_it_steps_in():
    w = Walker(1, 0.0, 6.0, 16.0, 1.0)
    walk(w, 2.0)                                      # at x 8, walking +x
    dog = (w.x + RADIUS + DOG_R + MIN_GAP + 0.02, 0.0, DOG_Z)   # right in front, suddenly
    path = walk(w, 3.0, dog, room=lambda side: 0.0)
    assert all(gap(p, dog) >= MIN_GAP - 1e-6 for p in path)


def test_the_dog_on_another_floor_or_behind_is_no_reason_to_stop():
    w = Walker(1, 0.0, 6.0, 16.0, 1.0)
    upstairs = (9.0, 0.0, crowd.FLOOR_HEIGHT + DOG_Z)
    assert w.aim(upstairs, None) == (1.0, 0.0)
    behind = (w.x - 2.0, 0.0, DOG_Z)
    assert w.aim(behind, None) == (1.0, 0.0)


def test_speed_and_side_step_change_smoothly():
    dog = (10.0, 0.0, DOG_Z)
    w = Walker(1, 0.5, 6.0, 16.0, 1.0)
    path = walk(w, 12.0, dog, room=lambda side: 0.9)
    for a, b in zip(path, path[1:]):
        assert abs(b[0] - a[0]) <= 1.0 * DT + 1e-9
        assert abs(b[1] - a[1]) <= crowd.SIDE_V * DT + 1e-9


WALLED = """<mujoco><worldbody>
  <geom name="floor" type="plane" size="20 20 0.1"/>
  <geom name="wall_n" type="box" pos="0 1.4 1" size="20 0.05 1"/>
  <body name="dog" pos="5 0 0.3"><freejoint/><geom type="box" size="0.35 0.15 0.1"/></body>
  {people}
</worldbody></mujoco>"""


def test_movers_find_the_wall_and_the_dog_in_the_scene():
    m = mujoco.MjModel.from_xml_string(WALLED.format(people=crowd.people_xml()))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    movers = Movers(m, d, threading.Lock(), 1)
    assert movers.dog() == pytest.approx((5.0, 0.0, 0.3))
    w = movers.walkers[0]
    w.x, w.y = 0.0, 0.7
    d.mocap_pos[movers.mocap[0]] = w.pos()
    mujoco.mj_forward(m, d)
    assert movers.room(0, w, +1.0) == pytest.approx(1.4 - 0.05 - 0.7, abs=1e-6)   # its own capsule skipped
    assert movers.room(0, w, -1.0) == math.inf
