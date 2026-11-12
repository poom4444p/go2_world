"""Ground truth for a run: what the dog really touched, and how near it came
to the people -- scored in the world, where the truth is.

The brain scores the same things itself (run_building's score_collision and
score_people), but in a HIL run it can only score them on the pose the board
reports and against its own crowd, which is empty there: with the board's
odometry (POSE=lio) that pose is up to ~0.4 m off, and the people walk here,
not in the brain. So the world scores both on its own physics, and run_hil.sh
prints this next to the brain's summary.

Same rules as the brain, so the two can be compared on a simulated run:
- collisions: any MuJoCo contact between a geom under the dog's free joint
  (trunk, hips, legs, feet) and the scene, except what it walks on (`ground`,
  the `slab`s, the lift car's `lift_plate`) and the people, who are scored by
  distance instead;
- people: the trunk's centre within PED_NEAR of a person's, on the same floor.

Sampled when the people are moved (~100 Hz, the physics steps at 500 Hz): a
touch shorter than ~10 ms can be missed. Coordinates are go2_world's frame
(the dog spawns at (0, 0)); the brain's map frame is this + (21.0, 9.5).
"""
import math

PED_NEAR = 0.55              # m, centre to centre: run_building's PED_NEAR
SAME_FLOOR_Z = (0.0, 1.5)    # trunk height above a person's floor: their floor (crowd.DOG_FLOOR_Z)
WALKED_ON = ("ground", "lift_plate")


def what(name):
    """What a scene geom is, for the summary: as run_building's body_contacts."""
    if name.startswith("obs"):
        return name.split("_", 1)[1] if "_" in name else name
    if name.startswith("step"):
        return "stairs"
    if name.startswith("lift"):
        return "lift"
    return "wall"


class Score:
    """Counts touching and nearness over a run. tick() under the physics lock."""

    def __init__(self, model, log=print):
        import mujoco
        self.log = log
        root = next(model.jnt_bodyid[j] for j in range(model.njnt)
                    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)

        def ours(b):
            while b:
                if b == root:
                    return True
                b = model.body_parentid[b]
            return False

        self.dog, self.scene = set(), {}
        for g in range(model.ngeom):
            if ours(model.geom_bodyid[g]):
                self.dog.add(g)
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if (name in WALKED_ON or name.startswith("slab") or name.startswith("person_")
                    or not model.geom_contype[g] and not model.geom_conaffinity[g]):
                continue
            self.scene[g] = (name, what(name))
        self.touch_s = {}            # what -> seconds touching it
        self.touching = None         # (since, names) while touching
        self.near_s, self.min_d = 0.0, math.inf
        self.near = None             # since, while within PED_NEAR of somebody
        self.t = 0.0

    def tick(self, data, dt, people):
        """dt: simulated seconds since the last tick; people: their (x, y, z)."""
        self.t = float(data.time)
        x, y, z = (float(v) for v in data.qpos[0:3])
        names = set()
        for c in data.contact[:data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 in self.dog else g1 if g2 in self.dog else None
            if other in self.scene:
                names.add(self.scene[other])
        for kind in {k for _, k in names}:
            self.touch_s[kind] = self.touch_s.get(kind, 0.0) + dt
        if names and self.touching is None:
            self.touching = self.t
            self.log(f"world: t={self.t:6.1f}s  COLLISION starts: dog ({x:.2f}, {y:.2f}) touching "
                     + ", ".join(sorted(f"{k} {n}" for n, k in names)))
        elif not names and self.touching is not None:
            self.log(f"world: t={self.t:6.1f}s  COLLISION ends after {self.t - self.touching:.1f}s:"
                     f" dog ({x:.2f}, {y:.2f})")
            self.touching = None

        close = None
        for k, p in enumerate(people):
            if not SAME_FLOOR_Z[0] <= z - p[2] < SAME_FLOOR_Z[1]:
                continue                       # another floor, or parked under the ground
            d = math.hypot(p[0] - x, p[1] - y)
            self.min_d = min(self.min_d, d)
            if d < PED_NEAR and (close is None or d < close[0]):
                close = (d, k, p)
        if close is not None:
            self.near_s += dt
            if self.near is None:
                self.near = self.t
                d, k, p = close
                self.log(f"world: t={self.t:6.1f}s  CONTACT starts: dog ({x:.2f}, {y:.2f}),"
                         f" person {k} ({p[0]:.2f}, {p[1]:.2f}), {d:.2f} m")
        elif self.near is not None:
            self.log(f"world: t={self.t:6.1f}s  CONTACT ends after {self.t - self.near:.1f}s:"
                     f" dog ({x:.2f}, {y:.2f})")
            self.near = None

    def summary(self):
        """The two lines run_hil.sh prints, in the brain's words."""
        if self.touch_s:
            hit = (f"world truth: COLLISIONS: {sum(self.touch_s.values()):.1f}s touching "
                   + ", ".join(f"{k} {s:.1f}s" for k, s in sorted(self.touch_s.items())))
        else:
            hit = "world truth: collisions: none -- no part of the dog touched a wall or an obstacle"
        if self.near_s:
            ppl = (f"world truth: CONTACT: {self.near_s:.1f}s within {PED_NEAR} m of a person"
                   f" (closest {self.min_d:.2f} m)")
        elif self.min_d == math.inf:
            ppl = "world truth: contact: none -- nobody was ever on its floor"
        else:
            ppl = f"world truth: contact: none -- closest it came to anybody was {self.min_d:.2f} m"
        return [f"world truth: {self.t:.0f}s of simulated time scored", hit, ppl]
