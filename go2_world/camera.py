"""The dog's camera, rendered in the world and streamed to the brain.

In HIL the brain's own twin scene has no people (they reach it only as
Mid-360 points), so a camera rendered there showed VAMOS an empty building.
This renders the camera here, where the people and the lift car are, and
sends the pixels to whoever connects (the brain on the J4012), as a USB
camera would hand them over: JPEG frames, nothing about where anything is.

Same lens and mount as Cyberdog's twin camera ("dogcam", MujocoRobot):
vertical FOV from its camera_config.yaml, CAM_FORWARD ahead of the trunk,
level, turned with the dog's heading only -- the projector that draws
VAMOS's paths assumes a level lens. Its height follows the trunk, as a lens
fixed to the dog would, CAM_HEIGHT above the floor when the dog stands.

The lens is the scene's fixed camera "dogcam" (scene.py, fovy FOVY); its
pose is set on the rendered copy only, never in the physics. Like the recorder, a thread copies where things are under the physics' lock
(microseconds), then renders that copy with the lock released. It renders
only while somebody is connected.

Wire format, over TCP, one message per frame:
  HEAD (magic, seq, world time s, sender wall time s, jpeg length) + JPEG bytes
"""
import io
import math
import socket
import struct
import threading
import time

import mujoco
import numpy as np

from go2_world.lift import FLOORS, floor_z

CAM_PORT = 9883
FPS = 10                        # VAMOS answers every ~1.8 s; 10 a second is plenty fresh
W, H = 640, 480                 # MujocoRobot.CAM_W, CAM_H
FOVY = 64.55                    # deg: 2 atan((480 / 2) / fy), Cyberdog's camera_config.yaml
CAM_FORWARD = 0.34              # m ahead of the trunk centre: MujocoRobot.CAM_FORWARD
CAM_HEIGHT = 0.32               # m above the floor: camera_config.yaml camera_height
TRUNK_H = 0.30                  # m: the trunk above the floor, standing (pose.py's start)
JPEG_Q = 90
CAM_NAME = "dogcam"              # in scene.SCENE, with fovy FOVY
MAGIC = b"GCAM"
HEAD = struct.Struct("<4sIddI")


def floor_of(z):
    """The storey a trunk height is on (as recorder.floor_of)."""
    return max([n for n in FLOORS if z >= floor_z(n) - 0.5] or [1])


def yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def lens_pose(trunk_xyz, yaw):
    """(position, xyaxes) of the lens for a trunk at trunk_xyz heading yaw:
    level, CAM_FORWARD ahead, CAM_HEIGHT above the floor under the trunk."""
    x, y, z = trunk_xyz
    c, s = math.cos(yaw), math.sin(yaw)
    pos = np.array([x + CAM_FORWARD * c, y + CAM_FORWARD * s, z - TRUNK_H + CAM_HEIGHT])
    # MuJoCo cameras look down their -z; x is right, y up.
    fwd = np.array([c, s, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    return pos, np.column_stack([right, up, -fwd])


def pack(seq, t_world, jpeg, sent=None):
    return HEAD.pack(MAGIC, seq & 0xFFFFFFFF, t_world, time.time() if sent is None else sent,
                     len(jpeg)) + jpeg


def encode(rgb, quality=JPEG_Q):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class Camera:
    def __init__(self, model, data, lock, host="127.0.0.1", port=CAM_PORT, fps=FPS):
        self.model, self.live, self.lock = model, data, lock
        self.data = mujoco.MjData(model)             # the copy that is rendered
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        if self.cam < 0:
            raise ValueError(f"no camera {CAM_NAME!r} in the scene (go2_world/scene.py)")
        self.addr, self.fps = (host, port), fps
        self.stop_ev = threading.Event()
        self.cond = threading.Condition()
        self.latest = None                           # (seq, message)
        self.clients = 0
        self.frames = self.sent = self.skipped = 0
        self.bytes = 0
        self.render_ms = []
        self.trunk_h = []                            # trunk above its floor, a frame: is TRUNK_H right?

    def start(self):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(self.addr)
        self.port = self.srv.getsockname()[1]       # port 0: the one the system gave
        self.srv.listen(2)
        self.srv.settimeout(0.5)
        threading.Thread(target=self._accept, name="camera-accept", daemon=True).start()
        threading.Thread(target=self._render_loop, name="camera", daemon=True).start()
        return self

    def close(self):
        self.stop_ev.set()
        with self.cond:
            self.cond.notify_all()
        try:
            self.srv.close()
        except OSError:
            pass

    def summary(self):
        ms = sorted(self.render_ms) or [0.0]
        kb = self.bytes / max(self.frames, 1) / 1024
        return (f"camera: {self.frames} frames rendered ({self.skipped} late), {self.sent} sent, "
                f"{kb:.0f} KB a frame, render+jpeg median {ms[len(ms) // 2]:.1f} ms, "
                f"max {ms[-1]:.1f} ms; trunk above its floor median "
                f"{float(np.median(self.trunk_h)) if self.trunk_h else float('nan'):.3f} m "
                f"(lens assumes {TRUNK_H:.2f})")

    # -- the frames -------------------------------------------------------
    def _copy(self):
        with self.lock:
            self.data.qpos[:] = self.live.qpos
            self.data.mocap_pos[:] = self.live.mocap_pos
            self.data.mocap_quat[:] = self.live.mocap_quat
            self.data.time = self.live.time
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def _render_loop(self):
        renderer = mujoco.Renderer(self.model, H, W)
        step, nxt, seq = 1.0 / self.fps, time.monotonic(), 0
        try:
            while not self.stop_ev.is_set():
                now = time.monotonic()
                if now < nxt:
                    self.stop_ev.wait(nxt - now)
                    continue
                if now - nxt > step:
                    self.skipped += int((now - nxt) / step)
                    nxt = now
                nxt += step
                if not self.clients:
                    continue
                t0 = time.perf_counter()
                self._copy()
                rgb = self.render(renderer)
                jpeg = encode(rgb)
                self.render_ms.append((time.perf_counter() - t0) * 1000)
                seq += 1
                self.frames += 1
                self.bytes += len(jpeg)
                with self.cond:
                    self.latest = (seq, pack(seq, self.data.time, jpeg))
                    self.cond.notify_all()
        finally:
            renderer.close()

    def render(self, renderer):
        """The lens's view of self.data: the scene's "dogcam", put where the
        lens is on this copy only (its cam_xpos/xmat, after kinematics)."""
        trunk = self.data.xpos[self.trunk]
        self.trunk_h.append(trunk[2] - floor_z(floor_of(trunk[2])))
        pos, axes = lens_pose(trunk, yaw_of(self.data.xquat[self.trunk]))
        self.data.cam_xpos[self.cam] = pos
        self.data.cam_xmat[self.cam] = axes.flatten()
        renderer.update_scene(self.data, camera=self.cam)
        return renderer.render()

    # -- the connections ----------------------------------------------------
    def _accept(self):
        while not self.stop_ev.is_set():
            try:
                conn, peer = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"camera: {peer[0]}:{peer[1]} connected", flush=True)
            threading.Thread(target=self._serve, args=(conn, peer), name="camera-send",
                             daemon=True).start()

    def _serve(self, conn, peer):
        """Send the newest frame each time there is a new one. A slow link
        skips frames rather than queueing them: only the newest matters."""
        self.clients += 1
        last = 0
        try:
            while not self.stop_ev.is_set():
                with self.cond:
                    while not self.stop_ev.is_set() and (self.latest is None or self.latest[0] == last):
                        self.cond.wait(0.5)
                    if self.stop_ev.is_set():
                        return
                    last, msg = self.latest
                conn.sendall(msg)
                self.sent += 1
        except OSError as e:
            print(f"camera: {peer[0]}:{peer[1]} gone ({e.__class__.__name__})", flush=True)
        finally:
            self.clients -= 1
            conn.close()

