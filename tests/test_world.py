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


def test_go2_stands_still_at_the_physics_step():
    """The bridge's PD, computed per lowcmd and held between steps, at the
    stand gains: the dog must stay put (at 5 ms it walks off metres)."""
    import mujoco
    from go2_world.scene import SIMULATE_DT, write_scene

    m = mujoco.MjModel.from_xml_path(write_scene())
    m.opt.timestep = SIMULATE_DT
    d = mujoco.MjData(m)
    stand = np.array([0.0, 0.67, -1.3] * 4)
    d.qpos[2], d.qpos[7:19] = 0.33, stand
    kp, kd, cmd_dt = 60.0, 5.0, 0.002
    t, next_cmd, xy0 = 0.0, 0.0, None
    while t < 3.0:
        if t >= next_cmd:
            d.ctrl[:12] = kp * (stand - d.sensordata[:12]) - kd * d.sensordata[12:24]
            next_cmd += cmd_dt
        mujoco.mj_step(m, d)
        t += SIMULATE_DT
        if xy0 is None and t > 1.0:
            xy0 = d.qpos[:2].copy()
    assert np.linalg.norm(d.qpos[:2] - xy0) < 0.01
    assert 0.30 < d.qpos[2] < 0.40


# -- keeping up (lidar-pacing) ------------------------------------------------

def test_behind_at_a_frame_start_whole_frames_are_skipped_not_burst(room):
    # HIL 11-11 14:51 (dense): behind, it cast and sent every overdue slice at
    # once, the send buffer overflowed, and the board got a third of a sweep.
    out = Out()
    emu = Emulator(Raycaster(room, "cyberdog"), StaticPose(), out, clock=lambda: 0.0)
    for t in np.arange(0.0, 0.1, 0.001):                       # frame 0, on time
        emu.step(t)
    emu.step(0.35)                                             # then 250 ms late
    for t in np.arange(0.351, 0.5, 0.001):
        emu.step(t)
    hs = [header(b) for b in out.points()]
    per_frame = math.ceil(1980 / 96)
    cnts = sorted({h["frame_cnt"] for h in hs})
    assert emu.frames_skipped == 2 and cnts[:3] == [0, 3, 4]  # frames 1 and 2 never sent
    for k in (0, 3, 4):                                       # the finished ones (5 has begun)
        f = [h for h in hs if h["frame_cnt"] == k]
        assert [h["udp_cnt"] for h in f] == list(range(per_frame))   # every frame sent is whole
        assert f[0]["stamp"] == k * 100_000_000               # and on the sensor's own 100 ms grid


def test_a_late_slice_in_a_frame_already_begun_is_counted_and_sent(room):
    out = Out()
    emu = Emulator(Raycaster(room, "cyberdog"), StaticPose(), out, clock=lambda: 0.0)
    emu.step(0.0)
    emu.step(0.05)                                             # slices 1-5 due, one tick late
    assert emu.frames_skipped == 0 and emu.slices_late >= 3


def test_udp_out_counts_what_it_could_not_send():
    import errno
    from go2_world.emulator import UdpOut

    class Full:
        def sendto(self, data, addr):
            raise OSError(errno.ENOBUFS, "No buffer space available")

    u = UdpOut.__new__(UdpOut)                                 # no real sockets for this
    u.host, u.failed, u.socks = "127.0.0.1", {}, {56301: Full()}
    u(b"x", 56301)
    u(b"x", 56301)
    assert u.failed == {"ENOBUFS": 2}
