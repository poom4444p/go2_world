"""The run's truth, scored in the world (go2_world/score.py)."""
import threading

import mujoco
import pytest

from go2_world import crowd, score
from go2_world.crowd import Movers

# A dog (free joint, a trunk and a foot under it), the floor it stands on, a
# wall, a trolley and the lift car's floor; the people as the building has them.
SCENE = """<mujoco><worldbody>
  <geom name="ground" type="plane" size="20 20 0.1"/>
  <geom name="slab1_0" type="box" pos="0 0 -0.05" size="20 20 0.05"/>
  <geom name="f1w0" type="box" pos="0 1.4 1" size="20 0.05 1"/>
  <geom name="obs1_trolley" type="box" pos="-3 0 0.4" size="0.3 0.3 0.4"/>
  <geom name="lift_plate" type="box" pos="8 0 -0.1" size="1 1 0.1"/>
  <body name="dog" pos="{x} {y} 0.3"><freejoint/>
    <geom name="trunk" type="box" size="0.35 0.15 0.1"/>
    <body name="foot" pos="0 0 -0.28"><geom name="foot" type="sphere" size="0.022"/></body>
  </body>
  {people}
</worldbody></mujoco>"""


def world(x=0.0, y=0.0):
    m = mujoco.MjModel.from_xml_string(SCENE.format(x=x, y=y, people=crowd.people_xml()))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def quiet(m):
    lines = []
    return score.Score(m, log=lines.append), lines


def test_the_dog_its_geoms_and_what_it_may_touch():
    m, _ = world()
    s, _ = quiet(m)
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in s.dog}
    assert names == {"trunk", "foot"}
    scene = {n: k for n, k in s.scene.values()}
    assert scene == {"f1w0": "wall", "obs1_trolley": "trolley"}    # not ground, slab, lift_plate, people


def test_standing_on_the_floor_is_no_collision():
    m, d = world()
    s, lines = quiet(m)
    d.qpos[2] = 0.3 - 0.28 + 0.02 - 0.001        # the foot just into the floor
    mujoco.mj_forward(m, d)
    assert d.ncon > 0
    s.tick(d, 0.01, [])
    assert s.touch_s == {} and not lines
    assert "collisions: none" in s.summary()[1]


def test_a_trunk_in_the_wall_is_scored_and_said_when():
    m, d = world(y=1.4 - 0.05 - 0.15 + 0.01)      # trunk 1 cm into the wall
    s, lines = quiet(m)
    for _ in range(3):
        s.tick(d, 0.01, [])
    assert s.touch_s == {"wall": pytest.approx(0.03)}
    assert len(lines) == 1 and "COLLISION starts" in lines[0] and "wall f1w0" in lines[0]
    d.qpos[1] = 0.0
    mujoco.mj_forward(m, d)
    s.tick(d, 0.01, [])
    assert "COLLISION ends" in lines[-1]
    assert s.summary()[1] == "world truth: COLLISIONS: 0.0s touching wall 0.0s"


def test_an_obstacle_is_named_as_the_brain_names_it():
    m, d = world(x=-3 + 0.3 + 0.35 - 0.01)
    s, _ = quiet(m)
    s.tick(d, 0.5, [])
    assert s.touch_s == {"trolley": 0.5}
    assert "trolley 0.5s" in s.summary()[1]


def test_people_scored_by_distance_on_the_dogs_floor_only():
    m, d = world()
    s, lines = quiet(m)
    upstairs = (0.3, 0.0, crowd.FLOOR_HEIGHT)
    s.tick(d, 0.1, [crowd.PARKED, upstairs])
    assert s.min_d == float("inf") and "nobody was ever on its floor" in s.summary()[2]
    s.tick(d, 0.1, [(0.5, 0.0, 0.0)])
    assert s.near_s == pytest.approx(0.1) and s.min_d == pytest.approx(0.5)
    assert "CONTACT starts" in lines[-1] and "person 0" in lines[-1]
    s.tick(d, 0.1, [(2.0, 0.0, 0.0)])
    assert "CONTACT ends" in lines[-1]
    assert s.summary()[2] == ("world truth: CONTACT: 0.1s within 0.55 m of a person"
                              " (closest 0.50 m)")


def test_nobody_near_says_how_near():
    m, d = world()
    s, _ = quiet(m)
    s.tick(d, 0.1, [(1.2, 0.3, 0.0)])
    assert s.summary()[2] == "world truth: contact: none -- closest it came to anybody was 1.24 m"


def test_movers_score_on_the_physics_with_the_people_where_the_physics_has_them():
    m, d = world()
    movers = Movers(m, d, threading.Lock(), 1)
    movers.score.log = lambda line: None
    d.mocap_pos[movers.mocap[0]] = (0.4, 0.0, 0.0)
    movers.score.tick(d, 0.01, [d.mocap_pos[k] for k in movers.mocap])
    assert movers.score.min_d == pytest.approx(0.4)
