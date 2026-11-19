import io
import math
import socket
import threading

import mujoco
import numpy as np
import pytest

from go2_world.camera import (CAM_FORWARD, CAM_HEIGHT, CAM_NAME, FOVY, HEAD, MAGIC, TRUNK_H, Camera,
                              encode, lens_pose, pack)

PIL = pytest.importorskip("PIL")

# A trunk, a red wall 3 m ahead of it (+x), a person-like blue box to its left.
XML = f"""<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall" type="box" pos="3 0 1" size="0.05 3 1" rgba="1 0 0 1"/>
    <body name="base_link" pos="0 0 {TRUNK_H}"><freejoint/><geom type="box" size="0.1 0.05 0.05" contype="0" conaffinity="0"/></body>
    <body name="person" mocap="true" pos="0 5 0"><geom type="box" size="0.2 0.2 0.8" pos="0 0 0.8" rgba="0 0 1 1"/></body>
    <camera name="{CAM_NAME}" pos="0 0 0" fovy="{FOVY}"/>
  </worldbody>
</mujoco>"""


def test_lens_is_level_ahead_and_at_camera_height():
    pos, axes = lens_pose((1.0, 2.0, TRUNK_H), math.pi / 2)
    assert np.allclose(pos, (1.0, 2.0 + CAM_FORWARD, CAM_HEIGHT))
    assert np.allclose(-axes[:, 2], (0.0, 1.0, 0.0), atol=1e-12)      # looks along the heading
    assert np.allclose(axes[:, 1], (0.0, 0.0, 1.0))                    # level: up is up
    assert np.allclose(axes[:, 0], (1.0, 0.0, 0.0), atol=1e-12)       # right of +y is +x


def test_lens_rides_with_the_trunk():
    pos, _ = lens_pose((0.0, 0.0, 2.2 + TRUNK_H + 0.02), 0.0)            # floor 2, trunk 2 cm up
    assert abs(pos[2] - (2.2 + CAM_HEIGHT + 0.02)) < 1e-12


def test_message_head_carries_no_position():
    msg = pack(7, 12.5, b"jpegbytes", sent=100.0)
    magic, seq, t, sent, n = HEAD.unpack_from(msg)
    assert (magic, seq, t, sent, n) == (MAGIC, 7, 12.5, 100.0, 9)
    assert msg[HEAD.size:] == b"jpegbytes"
    assert HEAD.size == 28                                               # magic, seq, 2 times, length


def _recv(sock, n):
    out = b""
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        assert chunk, "server closed"
        out += chunk
    return out


def test_served_frame_sees_what_the_dog_faces():
    from PIL import Image
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = Camera(model, data, threading.Lock(), port=0, fps=20).start()
    try:
        s = socket.create_connection(("127.0.0.1", cam.port), timeout=5)
        magic, seq, t, sent, n = HEAD.unpack(_recv(s, HEAD.size))
        assert magic == MAGIC and seq >= 1
        img = np.asarray(Image.open(io.BytesIO(_recv(s, n))))
        assert img.shape == (480, 640, 3)
        r, g, b = img[240, 320].astype(int)              # the centre: the red wall ahead
        assert r > 120 and r > 2 * max(g, b)
        # The person, moved in front of the wall: the next frames see it.
        with cam.lock:
            data.mocap_pos[0] = (2.0, 0.0, 0.0)
        for _ in range(3):
            _, _, _, _, n = HEAD.unpack(_recv(s, HEAD.size))
            img = np.asarray(Image.open(io.BytesIO(_recv(s, n))))
        r, g, b = img[240, 320].astype(int)
        assert b > 120 and b > 2 * max(r, g)
        s.close()
    finally:
        cam.close()
    assert cam.frames >= 4


def test_encode_is_a_jpeg():
    assert encode(np.zeros((480, 640, 3), np.uint8))[:2] == b"\xff\xd8"
