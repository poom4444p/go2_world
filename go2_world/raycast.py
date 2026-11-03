"""Rays from the Mid-360 into the building, in the sensor's own frame.

The raycaster loads the building alone, not the physics model: the dog is not
in it, so no ray can return the dog's own trunk, and min_r stands in for the
real unit's blind zone instead. Only the pose comes from the physics, over DDS.
"""
import mujoco
import numpy as np

from go2_world import mid360


def quat_to_rot(q):
    """(w, x, y, z) -> 3x3, the order Unitree's IMU state uses."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Raycaster:
    def __init__(self, xml_path, pattern="cyberdog", mount_xyz=mid360.MOUNT_XYZ,
                 mount_pitch=mid360.MOUNT_PITCH):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)       # geom poses, once: all static
        nh, nv, self.min_r, self.max_r = mid360.PATTERNS[pattern]
        self.rays = mid360.ray_pattern(nh, nv)
        self.n = len(self.rays)
        self.R_bs = mid360.mount_rotation(mount_pitch)
        self.t_bs = np.array(mount_xyz, float)
        # The moving parts, if this building has them (scene.movers_building):
        # the lift car and the people, placed by move() from run_world's word.
        self._car = self._people = None
        names = {self.model.body(i).name for i in range(self.model.nbody)}
        if "lift_car" in names:
            car = self.model.body("lift_car")
            self._car = (int(car.mocapid[0]), self.data.mocap_pos[car.mocapid[0]].copy())
            self._people = [int(self.model.body(n).mocapid[0])
                            for n in sorted(x for x in names if x.startswith("person_"))]

    def move(self, people=(), lift_z=None):
        """Put the lift car at height lift_z and the people where run_world
        says they are; the rest of the people stay parked underground."""
        if self._car is None:
            return
        from go2_world.crowd import PARKED
        if lift_z is not None:
            m, base = self._car
            self.data.mocap_pos[m] = base + (0.0, 0.0, lift_z)
        for k, m in enumerate(self._people):
            self.data.mocap_pos[m] = people[k] if k < len(people) else PARKED
        mujoco.mj_forward(self.model, self.data)

    def listen(self, port=None):
        """Take run_world's movers (people, lift car) from UDP in a thread;
        cast() applies the newest before its rays, on its own thread."""
        import json
        import socket
        import threading
        from go2_world.crowd import MOVERS_PORT
        self._pending = None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port or MOVERS_PORT))

        def run():
            while True:
                try:
                    self._pending = json.loads(sock.recvfrom(65536)[0])
                except (OSError, ValueError):
                    continue
        threading.Thread(target=run, daemon=True).start()
        return self

    def cast(self, pos, quat, lo=0, hi=None):
        """Rays lo..hi from a trunk at pos (world) with orientation quat (w, x, y, z).

        Returns (points (n, 3) in the sensor frame, zero where nothing came
        back, and cos of the incidence angle, zero likewise)."""
        pending = getattr(self, "_pending", None)
        if pending is not None:
            self._pending = None
            self.move([tuple(p) for p in pending.get("people", [])], pending.get("lift_z"))
        hi = self.n if hi is None else hi
        rays = self.rays[lo:hi]
        n = len(rays)
        R_ws = quat_to_rot(np.asarray(quat, float)) @ self.R_bs
        origin = np.asarray(pos, float) + quat_to_rot(np.asarray(quat, float)) @ self.t_bs
        # einsum, not @: numpy on Apple's Accelerate warns spuriously on this matmul
        vec = np.ascontiguousarray(np.einsum("ij,kj->ik", rays, R_ws).ravel())
        geomid = np.zeros(n, np.int32)
        dist = np.zeros(n)
        normal = np.zeros(3 * n)
        mujoco.mj_multiRay(self.model, self.data, origin, vec, None, True, -1,
                           geomid, dist, normal, n, self.max_r)
        hit = (geomid >= 0) & (dist >= self.min_r) & (dist <= self.max_r)
        pts = np.where(hit[:, None], rays * dist[:, None], 0.0)
        cos_inc = np.abs(np.einsum("ij,ij->i", normal.reshape(-1, 3), vec.reshape(-1, 3)))
        return pts, np.where(hit, cos_inc, 0.0)


def reflectivity(cos_inc, rng=None):
    """0..255, brighter face-on. The building has one material, so this is
    all there is to tell surfaces apart by, and that is fine for a guide dog."""
    r = 30.0 + 170.0 * cos_inc
    if rng is not None:
        r = r + rng.normal(0.0, 5.0, len(r))
    return np.where(cos_inc > 0, np.clip(r, 1, 255), 0).astype(np.uint8)

