"""The physics scene: unitree_mujoco's Go2 standing in the exported building.

unitree_mujoco loads one XML path, and its go2.xml finds its meshes relative to
whichever file includes it, so the scene is written out at run time with
absolute paths into build/ (not committed) rather than kept as a file here.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDING = os.path.join(ROOT, "assets", "building.xml")
UNITREE_MUJOCO = os.path.expanduser(
    os.environ.get("UNITREE_MUJOCO", "~/Work/unitree/unitree_mujoco"))
GO2 = os.path.join(UNITREE_MUJOCO, "unitree_robots", "go2")

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


def write_scene(out_dir=os.path.join(ROOT, "build")):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "scene.xml")
    with open(path, "w") as f:
        f.write(SCENE.format(go2=GO2, building=BUILDING))
    return path
