"""Packets, rays and the emulator's schedule: no DDS, no physics, no network."""
import math
import struct
import zlib

import numpy as np
import pytest

from go2_world import livox_packets as lp
from go2_world import mid360
from go2_world.emulator import Emulator, Impairments
from go2_world.pose import G, StaticPose
from go2_world.raycast import Raycaster
from go2_world.scene import BUILDING

# A room: floor at z = 0, a wall whose face is at x = 3 and one whose face is at y = 3.
ROOM = """<mujoco><worldbody>
  <geom type="plane" size="20 20 0.1"/>
  <geom type="box" pos="3.5 0 1" size="0.5 20 1"/>
  <geom type="box" pos="0 3.5 1" size="20 0.5 1"/>
</worldbody></mujoco>"""


def yaw(rad):
    return (math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2))


@pytest.fixture
def room(tmp_path):
    p = tmp_path / "room.xml"
    p.write_text(ROOM)
    return str(p)


class Out:
    def __init__(self):
        self.got = []

    def __call__(self, data, port):
        self.got.append((data, port))

    def points(self):
        return [d for d, p in self.got if p == mid360.POINT_HOST_PORT]

    def imu(self):
        return [d for d, p in self.got if p == mid360.IMU_HOST_PORT]


def header(b):
    """Read back by the doc's offsets, not by the encoder's struct."""
    u16 = lambda o: struct.unpack_from("<H", b, o)[0]   # noqa: E731
    return dict(version=b[0], length=u16(1), interval=u16(3), dot_num=u16(5), udp_cnt=u16(7),
                frame_cnt=b[9], data_type=b[10], time_type=b[11],
                crc=struct.unpack_from("<I", b, 24)[0], stamp=struct.unpack_from("<Q", b, 28)[0])


# -- packets ----------------------------------------------------------------

def test_point_packet_layout():
    xyz = np.zeros((96, 3))
    xyz[0] = (1.5, -0.25, 0.04)
    b = lp.points_packet(xyz, np.full(96, 7, np.uint8), 123456789, 5000.0, 3, 300)
    h = header(b)
    assert len(b) == 36 + 96 * 14 == h["length"] == 1380
    assert (h["version"], h["dot_num"], h["udp_cnt"], h["frame_cnt"]) == (0, 96, 3, 300 & 0xFF)
    assert (h["data_type"], h["time_type"], h["stamp"]) == (1, 0, 123456789)
    assert h["interval"] == 5000 * 96 // 100                 # driver: interval*100/dot_num = 5000 ns
    assert h["crc"] == zlib.crc32(b[28:]) and b[12:24] == bytes(12)
    assert struct.unpack_from("<iiiBB", b, 36) == (1500, -250, 40, 7, 0)
    assert struct.unpack_from("<iii", b, 36 + 14) == (0, 0, 0)


def test_imu_packet_layout():
    b = lp.imu_packet((0.1, 0.2, 0.3), (0.0, 0.0, 1.0), 42, 7)
    h = header(b)
    assert len(b) == h["length"] == 60 and (h["data_type"], h["dot_num"], h["udp_cnt"]) == (0, 1, 7)
    assert np.allclose(struct.unpack_from("<6f", b, 36), (0.1, 0.2, 0.3, 0, 0, 1))


# -- rays -------------------------------------------------------------------

def test_points_are_in_the_sensor_frame(room):
    r = Raycaster(room, mount_xyz=(0, 0, 0), mount_pitch=0.0)
    pts, cos = r.cast((0, 0, 0.5), yaw(0.0))
    ahead = pts[:mid360.PATTERNS["cyberdog"][1]]              # azimuth 0: the whole column
    hit = ahead[np.any(ahead != 0, axis=1)]
    assert len(hit) and np.allclose(hit[:, 0], 3.0) and np.allclose(hit[:, 1], 0.0)
    # Turned left 90 deg, the sensor's x now faces the y wall.
    pts, _ = r.cast((0, 0, 0.5), yaw(math.pi / 2))
    ahead = pts[:mid360.PATTERNS["cyberdog"][1]]
    hit = ahead[np.any(ahead != 0, axis=1)]
    assert len(hit) and np.allclose(hit[:, 0], 3.0)


def test_pitched_mount_sees_the_floor_closer(room):
    lo = math.radians(mid360.EL_MIN)
    for pitch in (0.0, 20.0):
        r = Raycaster(room, mount_xyz=(0, 0, 0), mount_pitch=pitch)
        pts, _ = r.cast((0, 0, 0.5), yaw(math.pi))            # facing away from both walls
        first = pts[0]                                       # azimuth 0, the lowest ray
        expect = 0.5 / math.sin(-lo + math.radians(pitch))
        assert np.isclose(np.linalg.norm(first), expect, atol=2e-3)


def test_misses_and_blind_zone_are_zero(room):
    r = Raycaster(room, "cyberdog", mount_xyz=(0, 0, 0), mount_pitch=0.0)
    pts, cos = r.cast((-10, -10, 0.5), yaw(math.pi))          # nothing within 12 m behind
    up = pts.reshape(180, 11, 3)[:, -1]                      # 52 deg up: open sky
    assert np.all(up == 0)
    near, _ = Raycaster(room, mount_xyz=(0, 0, 0), mount_pitch=0.0).cast((2.8, 0, 0.5), yaw(0.0))
    assert np.all(np.linalg.norm(near, axis=1)[np.any(near != 0, axis=1)] >= 0.35)


def test_building_corridor_has_the_trolley_ahead():
    r = Raycaster(BUILDING)
    pts, _ = r.cast((0, 0, 0.30), yaw(0.0))
    col = pts[:11]
    R = mid360.mount_rotation()
    world_x = (col @ R.T)[:, 0]
    assert np.any(np.isclose(world_x, 2.4, atol=0.02))       # its face, 2.4 m on


# -- the emulator -----------------------------------------------------------

def run(room, seconds, impair=None, pattern="cyberdog", pose=None):
    out = Out()
    emu = Emulator(Raycaster(room, pattern), pose or StaticPose(), out, impair, clock=lambda: 0.0)
    t = 0.0
    while t < seconds - 1e-9:
        emu.step(t)
        t += 0.001
    return emu, out


def test_frames_packets_and_times(room):
    emu, out = run(room, 0.3)
    hs = [header(b) for b in out.points()]
    per_frame = math.ceil(1980 / 96)                          # 21, the last one padded
    frames = [[h for h in hs if h["frame_cnt"] == k] for k in range(2)]
    for k, f in enumerate(frames):
        assert [h["udp_cnt"] for h in f] == list(range(per_frame))
        assert f[0]["stamp"] == k * 100_000_000
        assert all(b["stamp"] > a["stamp"] for a, b in zip(f, f[1:]))
        assert f[-1]["stamp"] < (k + 1) * 100_000_000
    assert all(h["dot_num"] == 96 for h in hs)


def test_imu_at_200_hz_in_g_in_the_sensor_frame(room):
    emu, out = run(room, 0.1)
    hs = [header(b) for b in out.imu()]
    assert len(hs) == 20 and [h["stamp"] for h in hs[:2]] == [0, 5_000_000]
    acc = np.array(struct.unpack_from("<6f", out.imu()[0], 36)[3:])
    # Standing still: 1 g up in the trunk, seen by a sensor pitched 20 deg down.
    p = math.radians(mid360.MOUNT_PITCH)
    assert np.allclose(acc, [-math.sin(p), 0.0, math.cos(p)], atol=1e-6)
    assert np.isclose(np.linalg.norm(acc), 1.0) and G > 9


def test_drop_delay_and_noise_switches(room):
    clean, a = run(room, 1.0)
    dropped, b = run(room, 1.0, Impairments(drop=0.2, seed=1))
    assert 0.1 < dropped.dropped / (dropped.dropped + dropped.sent) < 0.3
    late, c = run(room, 1.0, Impairments(delay_ms=20, jitter_ms=10, seed=1))
    assert len(c.got) < len(a.got)                           # the last 20-30 ms still in flight
    order = [header(x)["stamp"] for x in c.points()]
    assert order != sorted(order)                            # jitter reorders
    noisy, d = run(room, 0.1, Impairments(noise_m=0.02, seed=1))
    x0 = np.frombuffer(a.points()[0][36:], lp.POINT_HIGH)
    x1 = np.frombuffer(d.points()[0][36:], lp.POINT_HIGH)
    diff = (x1["x"].astype(int) - x0["x"])[x0["x"] != 0]
    assert 5 < np.std(diff) < 40                              # mm, about 2 cm of noise


def test_dense_is_the_real_rate(room):
    emu, out = run(room, 0.1, pattern="dense")
    assert len(out.points()) == math.ceil(20160 / 96) == 210   # ~2100 packets/s, as a real unit


def test_round_trip_through_go2_board():
    livox = pytest.importorskip("go2_board.livox")
    out = Out()
    emu = Emulator(Raycaster(BUILDING), StaticPose(), out,
                   Impairments(drop=0.05, delay_ms=3, jitter_ms=4, seed=2), clock=lambda: 0.0)
    for i in range(1000):
        emu.step(i * 0.001)
    a = livox.FrameAssembler()
    frames = [f for f in (a.feed(livox.parse(b)) for b in out.points()) if f is not None]
    assert len(frames) >= 8
    # every lost packet in a closed frame is found; drops in the open one are not yet
    assert 0 < a.lost <= emu.dropped_points
    f = frames[1]
    assert len(f.xyz) > 1000 and np.all(np.linalg.norm(f.xyz, axis=1) >= 0.35)
    samples = [livox.imu(livox.parse(b)) for b in out.imu()]
    assert np.allclose(np.linalg.norm(samples[0].acc), 9.81, atol=1e-3)
