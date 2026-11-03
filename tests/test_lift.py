"""The lift car: motion, the scene's mocap body, and the UDP call."""
import json
import socket
import threading
import time

import mujoco
import pytest

from go2_world.lift import ACCEL, FLOORS, RIDE_V, Lift, LiftServer, floor_z
from go2_world.scene import CAR_BODY, write_scene


def ride(lift, dt=0.002, limit=60.0):
    t, vmax = 0.0, 0.0
    while lift.moving and t < limit:
        lift.tick(dt)
        t += dt
        vmax = max(vmax, abs(lift.v))
    return t, vmax


def test_rides_to_each_floor_and_stops_on_it():
    lift = Lift()
    for n in (3, 1, 2, 3, 1):
        lift.call(n)
        t, vmax = ride(lift)
        assert lift.floor == n and lift.z == pytest.approx(floor_z(n)) and lift.v == 0.0
        assert vmax <= RIDE_V + 1e-9
    # one storey takes its distance at RIDE_V plus the time lost easing in and out
    lift.call(2)
    t, _ = ride(lift)
    assert floor_z(2) / RIDE_V < t < floor_z(2) / RIDE_V + RIDE_V / ACCEL + 0.1


def test_between_floors_is_no_floor_and_bad_calls_are_refused():
    lift = Lift()
    lift.call(2)
    for _ in range(500):
        lift.tick(0.002)
    assert lift.floor is None and lift.moving
    with pytest.raises(ValueError):
        lift.call(4)
    assert set(FLOORS) == {1, 2, 3}


def test_scene_puts_the_car_on_a_mocap_body():
    m = mujoco.MjModel.from_xml_path(write_scene())
    car = m.body(CAR_BODY)
    assert car.mocapid[0] >= 0
    names = {m.geom(i).name for i in range(m.ngeom) if m.geom_bodyid[i] == car.id}
    assert names == {"lift_plate", "lift_roof"}


def test_called_over_udp_and_moves_the_car():
    m = mujoco.MjModel.from_xml_path(write_scene())
    d = mujoco.MjData(m)
    server = LiftServer(m, d, threading.Lock(), port=19880).start()
    c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    c.settimeout(1.0)
    try:
        c.sendto(json.dumps({"call": 2}).encode(), ("127.0.0.1", 19880))
        assert json.loads(c.recvfrom(1024)[0])["target"] == 2
        time.sleep(0.5)
        assert d.mocap_pos[server.mocap][2] > server.base[2]       # on its way up
        c.sendto(b'{"call": 9}', ("127.0.0.1", 19880))
        assert "error" in json.loads(c.recvfrom(1024)[0])
    finally:
        c.close()
