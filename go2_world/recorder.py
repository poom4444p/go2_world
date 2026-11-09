"""The world to an MP4, as it happens: the dog, the people and the lift.

A thread takes the physics' lock only to copy where things are (qpos and the
mocap bodies, microseconds), then renders that copy offscreen and hands the
frame to ffmpeg with the lock released, so the physics never waits on a
frame. The camera chases the dog as Cyberdog's own video does (3.2 m behind,
11 deg down, under the ceiling of the floor above), and watches the lift car
from the corridor while the dog is in it: chasing from behind there puts the
camera inside the shaft wall.

Each frame is stamped with the world's time and the dog's position, to set
against a brain log. The MP4 is fragmented, so a run that is killed still
leaves a file that plays; `close()` finishes it properly.

Needs imageio[ffmpeg] and Pillow (`pip install -e .[record]`).
"""
import math
import threading
import time

import mujoco
import numpy as np

from go2_world.lift import FLOORS, floor_z

FPS = 15
W, H = 640, 480                  # the scene's offscreen buffer is 640x480
CHASE_D, CHASE_EL = 3.2, -11     # Cyberdog run_building's CHASE_D, CHASE_EL
LIFT_CAM = (0.0, 3.4, -6.0)      # azimuth, distance, elevation: from the corridor, west of the car
IN_CAR_M = 1.0                   # the dog's trunk this near the car's centre (xy): in the lift
AZ_SMOOTH = 0.15                 # per frame: the trunk yaws with every step; the camera should not
WALL_GAP, MIN_D = 0.25, 0.8      # m: a camera that would be in a wall comes in front of it, this near at most


def floor_of(z):
    """The storey a trunk height is on."""
    return max(n for n in FLOORS if z >= floor_z(n) - 0.5)


def yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def ease_angle(prev, target, k):
    """prev moved k of the way to target, the short way round (degrees)."""
    if prev is None:
        return target
    d = (target - prev + 180.0) % 360.0 - 180.0
    return prev + k * d


def back_dir(az_deg, el_deg):
    """Unit vector from the look-at point to a free camera (MuJoCo's convention:
    the camera looks along azimuth/elevation, from distance behind the point)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([-math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), -math.sin(e)])


def pulled_in(dist, hit):
    """The camera distance with a wall `hit` m behind the look-at point (-1: none)."""
    if hit < 0 or hit - WALL_GAP >= dist:
        return dist
    return max(MIN_D, hit - WALL_GAP)


def camera_for(dog_xyz, yaw, car_xy, prev_az):
    """(lookat, azimuth, distance, elevation, in_lift) for one frame."""
    x, y, z = dog_xyz
    in_lift = car_xy is not None and math.hypot(x - car_xy[0], y - car_xy[1]) < IN_CAR_M
    if in_lift:
        az, dist, el = LIFT_CAM
    else:
        # Azimuth is the direction the camera looks along: the dog's heading
        # puts the camera behind it, looking where it walks.
        az, dist, el = ease_angle(prev_az, math.degrees(yaw), AZ_SMOOTH), CHASE_D, CHASE_EL
    return (x, y, z + 0.3), az, dist, el, in_lift


class Recorder:
    def __init__(self, model, data, lock, path, fps=FPS):
        self.model, self.live, self.lock = model, data, lock
        self.path, self.fps = path, fps
        self.data = mujoco.MjData(model)            # the copy that is rendered
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.car = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lift_car")
        self.stop_ev = threading.Event()
        self.thread = None
        self.frames = 0
        self.skipped = 0
        self.slowest_lock_ms = 0.0

    def start(self):
        self.thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self.thread.start()
        return self

    def close(self):
        """Stop and finish the file. Safe to call twice."""
        self.stop_ev.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=5)

    def _copy(self):
        t0 = time.perf_counter()
        with self.lock:
            self.data.qpos[:] = self.live.qpos
            self.data.mocap_pos[:] = self.live.mocap_pos
            self.data.mocap_quat[:] = self.live.mocap_quat
            self.data.time = self.live.time
        self.slowest_lock_ms = max(self.slowest_lock_ms, (time.perf_counter() - t0) * 1000)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def _stamp(self, frame, dog_xyz, in_lift):
        from PIL import Image, ImageDraw
        img = Image.fromarray(frame)
        x, y, z = dog_xyz
        text = (f"world t={self.data.time:7.1f}s  floor {floor_of(z)}  dog ({x:5.1f}, {y:5.1f})"
                f"{'  in lift' if in_lift else ''}")
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, W, 20), fill=(0, 0, 0))
        d.text((6, 4), text, fill=(255, 255, 255))
        return np.asarray(img)

    def _run(self):
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(self.model, H, W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        writer = imageio.get_writer(
            self.path, fps=self.fps, codec="libx264", quality=7, macro_block_size=1,
            output_params=["-movflags", "frag_keyframe+empty_moov"])
        az = None
        step = 1.0 / self.fps
        nxt = time.monotonic()
        try:
            while not self.stop_ev.is_set():
                now = time.monotonic()
                if now < nxt:
                    self.stop_ev.wait(nxt - now)
                    continue
                if now - nxt > step:                 # behind: drop frames, keep real time
                    self.skipped += int((now - nxt) / step)
                    nxt = now
                nxt += step
                self._copy()
                dog = self.data.xpos[self.trunk].copy()
                yaw = yaw_of(self.data.xquat[self.trunk])
                car = self.data.xpos[self.car][:2] if self.car >= 0 else None
                lookat, az_now, dist, el, in_lift = camera_for(dog, yaw, car, az)
                if not in_lift:
                    az = az_now
                if not in_lift:
                    # A corridor is narrower than the chase distance when the dog
                    # turns: come in front of the wall rather than render its inside.
                    geomid = np.zeros(1, np.int32)
                    hit = mujoco.mj_ray(self.model, self.data, np.array(lookat), back_dir(az_now, el),
                                        None, 1, self.trunk, geomid)
                    dist = pulled_in(dist, hit)
                cam.lookat[:] = lookat
                cam.azimuth, cam.distance, cam.elevation = az_now, dist, el
                renderer.update_scene(self.data, cam)
                writer.append_data(self._stamp(renderer.render(), dog, in_lift))
                self.frames += 1
        finally:
            writer.close()
            renderer.close()
            print(f"recorder: {self.frames} frames to {self.path} ({self.skipped} dropped,"
                  f" lock held at most {self.slowest_lock_ms:.2f} ms)", flush=True)
