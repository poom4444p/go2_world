import math

import mujoco

from go2_world.lift import floor_z
from go2_world.recorder import (CHASE_D, CHASE_EL, LIFT_CAM, MIN_D, WALL_GAP, add_puddles, back_dir,
                                 camera_for, ease_angle, floor_of, pulled_in, yaw_of)


def test_floor_of_trunk_heights():
    assert floor_of(0.3) == 1
    assert floor_of(2.2 + 0.3) == 2
    assert floor_of(4.4 + 0.3) == 3


def test_yaw_of_quaternion():
    q = (math.cos(0.5 / 2), 0.0, 0.0, math.sin(0.5 / 2))
    assert abs(yaw_of(q) - 0.5) < 1e-9


def test_ease_angle_goes_the_short_way_round():
    assert ease_angle(None, 30.0, 0.1) == 30.0
    a = ease_angle(170.0, -170.0, 0.5)          # 20 deg apart across +-180
    assert abs(((a - 180.0 + 180.0) % 360.0) - 180.0) < 1e-9


def test_chase_camera_behind_the_dog():
    lookat, az, dist, el, in_lift = camera_for((10.0, 9.0, 0.3), math.radians(90), (45.0, 9.5), None)
    assert not in_lift
    assert (az, dist, el) == (90.0, CHASE_D, CHASE_EL)
    assert lookat == (10.0, 9.0, 0.6)


def test_lift_camera_when_the_dog_is_in_the_car():
    _, az, dist, el, in_lift = camera_for((45.2, 9.6, 2.5), 0.0, (45.0, 9.5), 10.0)
    assert in_lift
    assert (az, dist, el) == LIFT_CAM


def test_back_dir_is_behind_and_above_for_a_downward_camera():
    v = back_dir(0.0, -11.0)                    # looking along +x, a little down
    assert v[0] < 0 and v[2] > 0 and abs(v[1]) < 1e-9
    assert abs(sum(c * c for c in v) - 1.0) < 1e-9


def test_pulled_in_front_of_a_wall():
    assert pulled_in(CHASE_D, -1) == CHASE_D               # nothing behind
    assert pulled_in(CHASE_D, 10.0) == CHASE_D             # wall beyond the camera
    assert abs(pulled_in(CHASE_D, 2.0) - (2.0 - WALL_GAP)) < 1e-9
    assert pulled_in(CHASE_D, 0.5) == MIN_D                # never closer than this


def test_puddles_are_drawn_on_their_floor_and_not_in_the_model():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='plane' size='5 5 0.1'/></worldbody></mujoco>")
    data = mujoco.MjData(model)
    scn = mujoco.MjvScene(model, maxgeom=10)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(),
                           mujoco.mjtCatBit.mjCAT_ALL, scn)
    before = scn.ngeom
    assert add_puddles(scn, [(1.0, 2.0, 0.4, 1), (3.0, -1.0, 0.3, 3)]) == 2
    assert scn.ngeom == before + 2
    g = scn.geoms[before + 1]
    assert abs(g.pos[2] - (floor_z(3) + 0.004)) < 1e-6 and abs(g.size[0] - 0.3) < 1e-6
    assert model.ngeom == 1                         # nothing the physics or a ray could hit


def test_puddles_stop_when_the_scene_is_full():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scn = mujoco.MjvScene(model, maxgeom=1)
    assert add_puddles(scn, [(0.0, 0.0, 0.4, 1)] * 3) == 1
