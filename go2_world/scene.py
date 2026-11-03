"""The physics scene: unitree_mujoco's Go2 standing in the exported building.

unitree_mujoco loads one XML path, and its go2.xml finds its meshes relative to
whichever file includes it, so the scene is written out at run time with
absolute paths into build/ (not committed) rather than kept as a file here.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDING = os.path.join(ROOT, "assets", "building.xml")
UNITREE_MUJOCO = os.path.expanduser(
    os.environ.get("UNITREE_MUJOCO", "~/Work/unitree/unitree_mujoco"))
GO2 = os.path.join(UNITREE_MUJOCO, "unitree_robots", "go2")

# Physics step. unitree_mujoco's default 5 ms holds each PD torque too long
# for the stand gains (kd 5): the legs chatter and the dog walks off. A real
# Go2's motors close the PD at kHz, so the sim, not the gains, is what changes.
SIMULATE_DT = 0.002

SCENE = """<mujoco model="go2_world">
  <include file="{go2}/go2.xml"/>
  <compiler meshdir="{go2}/assets"/>
  <include file="{building}"/>

  <visual>
    <headlight ambient="0.42 0.42 0.45" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <global azimuth="-130" elevation="-20"/>
    <map znear="0.002" zfar="60"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.62 0.72" rgb2="0.82 0.85 0.9"
             width="512" height="512"/>
  </asset>
</mujoco>
"""


# The car's floor and roof, frozen at floor 1 in the export, become one mocap
# body the lift thread moves (go2_world/lift.py): the dog rides on contact.
CAR_GEOMS = ("lift_plate", "lift_roof")
CAR_FLOOR_DEPTH = 0.30
CAR_BODY = "lift_car"


def building_with_car(text):
    """The building XML with the car's geoms moved onto a mocap body at the
    car's floor-1 position, their positions made relative to it."""
    found = {}
    for name in CAR_GEOMS:
        m = re.search(rf'\n(\s*)<geom name="{name}"[^\n]*/>', text)
        if m is None:
            raise ValueError(f"no {name} in the building: re-export it with the lift car")
        found[name] = m.group(0).strip()
        text = text.replace(m.group(0), "")
    px, py, _ = (float(v) for v in re.search(r'pos="([^"]+)"', found["lift_plate"]).group(1).split())
    geoms = []
    for name, g in found.items():
        x, y, z = (float(v) for v in re.search(r'pos="([^"]+)"', g).group(1).split())
        if name == "lift_plate":
            # A car floor is a structure, not a sheet: 0.30 m deep, its top where
            # the export put it. At 0.06 m a foot (r 0.022) went through it when
            # the thread moving the car woke late and the car rose a few cm in
            # one step, and the dog rode up standing in it, 0.15 m too low.
            sx, sy, sz = (float(v) for v in re.search(r'size="([^"]+)"', g).group(1).split())
            top = z + sz
            z, sz = top - CAR_FLOOR_DEPTH / 2, CAR_FLOOR_DEPTH / 2
            g = re.sub(r'size="[^"]+"', f'size="{sx:.3f} {sy:.3f} {sz:.3f}"', g)
        geoms.append(re.sub(r'pos="[^"]+"', f'pos="{x - px:.3f} {y - py:.3f} {z:.3f}"', g))
    from go2_world.crowd import people_xml
    body = (f'    <body name="{CAR_BODY}" mocap="true" pos="{px:.3f} {py:.3f} 0">\n      '
            + "\n      ".join(geoms) + "\n    </body>\n" + people_xml() + "\n  </worldbody>")
    return text.replace("</worldbody>", body, 1)


def movers_building(out_dir=os.path.join(ROOT, "build")):
    """The building with its moving parts (the lift car, the people), alone:
    what the LiDAR emulator casts into. Written next to the scene."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "building_movers.xml")
    with open(BUILDING) as f:
        inner = building_with_car(f.read())
    with open(path, "w") as out:
        out.write(inner)
    return path


def write_scene(out_dir=os.path.join(ROOT, "build")):
    os.makedirs(out_dir, exist_ok=True)
    building = os.path.join(out_dir, "building_lift.xml")
    with open(BUILDING) as f, open(building, "w") as out:
        out.write(building_with_car(f.read()))
    path = os.path.join(out_dir, "scene.xml")
    with open(path, "w") as f:
        f.write(SCENE.format(go2=GO2, building=building))
    return path
