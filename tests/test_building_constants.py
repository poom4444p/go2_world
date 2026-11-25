"""The numbers that live in more than one place, held together by the building.

go2_world copies the storey height (lift.py, crowd.py), the corridor the people
walk (crowd.LANES) and the lift's stops from Cyberdog's building, which is
frozen into assets/building.xml. Nothing else ties the copies to it, so a
re-export or an edit in one place would drift silently. These tests read the
building itself and fail when a copy no longer matches.
"""
import mujoco
import numpy as np
import pytest

from go2_world import crowd, lift
from go2_world.scene import CAR_BODY, movers_building, write_scene

TOL = 1e-6
FOOT_INSET = 0.05      # m: the car's edge may touch a slab piece, not overlap it


def geoms(model, prefix):
    """(name, centre xyz, half-size xyz) of every static geom whose name starts with prefix."""
    out = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and name.startswith(prefix):
            out.append((name, model.geom_pos[i].copy(), model.geom_size[i].copy()))
    return out


def slab_top(model, floor):
    tops = {round(p[2] + s[2], 4) for _, p, s in geoms(model, f"slab{floor}_")}
    assert len(tops) == 1, f"floor {floor}'s slab pieces do not share one top: {sorted(tops)}"
    return tops.pop()


@pytest.fixture(scope="module")
def scene():
    return mujoco.MjModel.from_xml_path(write_scene())


@pytest.fixture(scope="module")
def movers():
    m = mujoco.MjModel.from_xml_path(movers_building())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_storey_height_is_one_number_and_the_slabs_are_where_it_says(scene):
    assert crowd.FLOOR_HEIGHT == lift.FLOOR_HEIGHT
    for n in lift.FLOORS:
        assert slab_top(scene, n) == pytest.approx(lift.floor_z(n), abs=TOL), \
            f"floor {n}'s slab is not at the height the lift stops at"


def test_the_car_floor_starts_level_with_floor_one_and_the_lift_stops_match_the_slabs(scene):
    body = scene.body(CAR_BODY)
    car_z = scene.body_pos[body.id][2]
    plate = [g for g in range(scene.ngeom)
             if mujoco.mj_id2name(scene, mujoco.mjtObj.mjOBJ_GEOM, g) == "lift_plate"]
    assert len(plate) == 1
    g = plate[0]
    top = car_z + scene.geom_pos[g][2] + scene.geom_size[g][2]
    assert top == pytest.approx(lift.floor_z(1), abs=TOL)
    # riding to floor n puts that top at floor n's slab
    for n in lift.FLOORS:
        assert top + lift.floor_z(n) == pytest.approx(slab_top(scene, n), abs=TOL)


def test_the_lift_shaft_is_open_in_every_upper_slab(scene):
    body = scene.body(CAR_BODY)
    cx, cy = scene.body_pos[body.id][:2]
    plate = next(g for g in range(scene.ngeom)
                 if mujoco.mj_id2name(scene, mujoco.mjtObj.mjOBJ_GEOM, g) == "lift_plate")
    hx, hy = scene.geom_size[plate][:2]
    for n in lift.FLOORS[1:]:
        for name, p, s in geoms(scene, f"slab{n}_"):
            x_in = abs(p[0] - cx) < s[0] + hx - FOOT_INSET
            y_in = abs(p[1] - cy) < s[1] + hy - FOOT_INSET
            assert not (x_in and y_in), f"{name} is in the car's way on floor {n}"


def test_every_crowd_lane_has_room_for_a_person_between_the_corridor_walls(movers):
    m, d = movers
    clear = crowd.RADIUS + 0.05
    for floor, lanes in crowd.LANES.items():
        z = lift.floor_z(floor) + 0.5
        for (_f, off, x0, x1, _v) in lanes:
            for x in np.linspace(min(x0, x1), max(x0, x1), 12):
                for y in (off, -off):                       # keep-right puts them on either side
                    for side in (1.0, -1.0):
                        gid = np.zeros(1, np.int32)
                        dist = mujoco.mj_ray(m, d, np.array([x, y, z]),
                                             np.array([0.0, side, 0.0]), None, 1, -1, gid)
                        if dist < 0:
                            pytest.fail(f"floor {floor} lane at x={x:.1f}, y={y}: no wall at all "
                                        f"on the {'+' if side > 0 else '-'}y side")
                        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or ""
                        if name.startswith("obs"):
                            continue                        # crates are walked round, not through
                        assert dist >= clear, (
                            f"floor {floor} lane at x={x:.1f}, y={y}: {name} is {dist:.2f} m away "
                            f"on the {'+' if side > 0 else '-'}y side, less than a person needs")
