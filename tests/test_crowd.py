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


def test_keeping_right_they_walk_on_the_right_of_their_way():
    east = Walker(1, 0.7, 6.0, 16.0, 1.0, keep_right=True)            # going +x first
    assert east.u > 0 and east.y == -0.7                              # +x: right is -y
    west = Walker(1, 0.7, 16.0, 6.0, 1.0, keep_right=True)            # going -x first
    assert west.u < 0 and west.y == 0.7


def test_keeping_right_they_cross_over_after_turning_round():
    w = Walker(1, 0.7, 6.0, 16.0, 1.0, keep_right=True)
    w.x = 15.95
    for _ in range(400):                                             # 4 s at 10 ms
        w.tick(0.01)
    assert w.u < 0 and abs(w.y - 0.7) < 1e-9                         # back west, on the north side


def test_fixed_lanes_unless_asked():
    w = Walker(1, 0.7, 16.0, 6.0, 1.0)
    assert w.y == 0.7
    for _ in range(100):
        w.tick(0.01)
    assert w.y == 0.7


def walk_with_dog(w, seconds, dog0, dog_v, room):
    """Tick w beside a dog walking at dog_v from dog0; (person, dog) positions."""
    out, (x, y, z) = [], dog0
    for _ in range(round(seconds / DT)):
        x, y = x + dog_v[0] * DT, y + dog_v[1] * DT
        out.append((w.tick(DT, (x, y, z), room, dog_v), (x, y)))
    return out


def test_overtakes_a_dog_going_its_way_at_its_own_pace():
    w = Walker(1, -0.6, 6.0, 30.0, 1.0)                 # going +x, on the right (-y)
    # the corridor's other half free: 2 m from the lane to the far wall
    path = walk_with_dog(w, 25.0, (9.0, -0.6, DOG_Z), (0.5, 0.0), room=lambda side: 2.0)
    passed = [k for k, (p, d) in enumerate(path) if p[0] > d[0] + 1.0]
    assert passed, "it never got past the dog"
    alongside = [p for p, d in path if abs(p[0] - d[0]) < 0.3]
    assert min(p[1] for p in alongside) >= -0.6 + crowd.SIDE_M - 1e-6   # stepped out (+y)
    speeds = [abs(b[0][0] - a[0][0]) / DT for a, b in zip(path, path[1:])
              if abs(a[0][0] - a[1][0]) < 0.3]
    assert max(speeds) > crowd.PASS_V + 0.3            # its own pace, not PASS_V
    assert all(gap(p, (d[0], d[1])) >= MIN_GAP - 1e-6 for p, d in path)


def test_no_room_to_overtake_follows_at_the_dogs_pace():
    w = Walker(1, -0.3, 6.0, 30.0, 1.0)
    path = walk_with_dog(w, 15.0, (9.0, -0.3, DOG_Z), (0.5, 0.0), room=lambda side: RADIUS)
    late = path[len(path) // 2:]
    v = [(b[0][0] - a[0][0]) / DT for a, b in zip(late, late[1:])]
    assert sum(v) / len(v) == pytest.approx(0.5, abs=0.05)   # behind it, not stopped
    assert all(gap(p, (d[0], d[1])) >= MIN_GAP - 1e-6 for p, d in path)


def test_a_dog_coming_the_other_way_is_still_passed_slowly():
    w = Walker(1, -0.6, 6.0, 30.0, 1.0)
    path = walk_with_dog(w, 12.0, (16.0, 0.3, DOG_Z), (-0.5, 0.0), room=lambda side: 0.9)
    speeds = [abs(b[0][0] - a[0][0]) / DT for a, b in zip(path, path[1:])
              if abs(a[0][0] - a[1][0]) < 0.3]
    assert speeds and max(speeds) <= crowd.PASS_V + 1e-6


def test_an_overtaker_steps_back_in_only_well_past_the_dog():
    w = Walker(1, -0.6, 6.0, 40.0, 1.0)
    path = walk_with_dog(w, 30.0, (9.0, -0.6, DOG_Z), (0.5, 0.0), room=lambda side: 2.0)
    back = [(p, d) for p, d in path if p[0] > d[0] and abs(p[1] - (-0.6)) < 0.05]
    assert back, "it never got back into its lane"
    assert min(p[0] - d[0] for p, d in back) >= crowd.RETURN_M - 0.05


def test_no_overtaking_into_somebody_coming():
    w = Walker(1, -0.6, 6.0, 40.0, 1.0)
    path = walk_with_dog(w, 10.0, (9.0, -0.6, DOG_Z), (0.5, 0.0), room=lambda side: 2.0)
    w2 = Walker(1, -0.6, 6.0, 40.0, 1.0)
    out = []
    x, y = 9.0, -0.6
    for _ in range(round(10.0 / DT)):
        x += 0.5 * DT
        out.append((w2.tick(DT, (x, y, DOG_Z), lambda side: 2.0, (0.5, 0.0),
                            oncoming=lambda side: True), (x, y)))
    assert max(p[1] for p, _ in out) <= -0.6 + 1e-6          # stayed in its lane
    assert max(p[0] for p, _ in out) < max(d[0] for _, d in out)   # behind the dog
    assert all(gap(p, d) >= MIN_GAP - 1e-6 for p, d in out)
    assert max(p[1] for p, _ in path) > -0.6 + 0.5          # without anybody coming, it went out


def test_a_stander_stays_put_whatever_the_dog_does():
    s = crowd.Stander(3.0, 0.8)
    for _ in range(500):
        p = s.tick(DT, (3.5, 0.8, DOG_Z), lambda side: 0.0, (0.5, 0.0))
    assert p == (3.0, 0.8, 0.0)


def test_a_stander_leaves_after_its_time():
    s = crowd.Stander(3.0, 0.8, floor=3, stay=2.0)
    assert s.tick(1.0) == (3.0, 0.8, 2 * crowd.FLOOR_HEIGHT)
    s.tick(1.5)
    assert s.pos() == crowd.PARKED


def test_movers_put_standers_after_the_walkers():
    m = mujoco.MjModel.from_xml_string(WALLED.format(people=crowd.people_xml()))
    d = mujoco.MjData(m)
    movers = Movers(m, d, threading.Lock(), 2, standers=[crowd.Stander(3.0, 0.8)])
    assert movers.state()["people"][2] == [3.0, 0.8, 0.0]
    with pytest.raises(ValueError):
        Movers(m, d, threading.Lock(), crowd.MAX_PEOPLE, standers=[crowd.Stander(0, 0)])


def test_a_slow_walker_waits_parked_until_the_dog_comes_near():
    w = crowd.SlowWalker(6.0, 0.5)
    assert w.tick(DT, (-2.0, -0.6, DOG_Z), sim_dt=DT) == crowd.PARKED     # 8 m behind
    assert w.tick(DT, (6.0, -0.6, DOG_Z + 2 * crowd.FLOOR_HEIGHT), sim_dt=DT) == crowd.PARKED  # floor 3
    assert w.tick(DT, (6.0 - crowd.APPEAR_M + 0.5, -0.6, DOG_Z), sim_dt=DT) == (6.0, -0.6, 0.0)


def test_a_slow_walker_walks_on_simulated_time_and_is_gone_at_its_end():
    w = crowd.SlowWalker(6.0, 0.5, x1=8.0)
    w.tick(DT, (2.0, -0.6, DOG_Z), sim_dt=0.0)                    # set off
    for _ in range(100):                                         # 1 s of wall, 0.8 s simulated
        p = w.tick(DT, None, sim_dt=0.8 * DT)
    assert abs(p[0] - (6.0 + 0.5 * 0.8)) < 1e-9 and p[1] == -0.6
    for _ in range(1000):
        p = w.tick(DT, None, sim_dt=DT)
    assert p == crowd.PARKED


def test_a_slow_walker_never_steps_into_a_dog_cut_in_ahead():
    w = crowd.SlowWalker(6.0, 0.5)
    w.tick(DT, (2.0, -0.6, DOG_Z), sim_dt=0.0)
    dog = (6.0 + crowd.RADIUS + crowd.DOG_R + crowd.MIN_GAP + 0.02, -0.6, DOG_Z)
    for _ in range(100):
        p = w.tick(DT, dog, sim_dt=DT)
    assert math.hypot(dog[0] - p[0], dog[1] - p[1]) >= crowd.RADIUS + crowd.DOG_R + crowd.MIN_GAP


def test_movers_put_slow_walkers_with_the_standers():
    m = mujoco.MjModel.from_xml_string(WALLED.format(people=crowd.people_xml()))
    d = mujoco.MjData(m)
    w = crowd.SlowWalker(6.0, 0.5)
    movers = Movers(m, d, threading.Lock(), 2, standers=[crowd.Stander(3.0, 0.8), w])
    assert movers.state()["people"][3] == list(crowd.PARKED)
