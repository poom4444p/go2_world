"""The motors' PD every physics step (go2_world/motor_pd.py), on a one-joint model."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from go2_world import motor_pd

ARM = """<mujoco><option timestep="0.002"/><worldbody><body>
  <joint name="j" type="hinge" axis="0 1 0" damping="0.01"/>
  <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
</body></worldbody>
<actuator><motor name="j" joint="j"/></actuator>
<sensor><jointpos joint="j"/><jointvel joint="j"/><jointactuatorfrc joint="j"/></sensor>
</mujoco>"""


def lowcmd(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0):
    return SimpleNamespace(motor_cmd=[SimpleNamespace(q=q, dq=dq, kp=kp, kd=kd, tau=tau)])


class Bridge:
    """Stands in for unitree_mujoco's UnitreeSdk2Bridge: one motor."""
    num_motor = 1

    def LowCmdHandler(self, msg):
        raise AssertionError("the bridge's own handler should be replaced")


@pytest.fixture
def arm():
    model = mujoco.MjModel.from_xml_string(ARM)
    data = mujoco.MjData(model)
    yield model, data
    mujoco.set_mjcb_control(None)        # the callback is global: leave none behind


def test_models_still_load_after_install():
    # run_world installs before unitree_mujoco loads the scene; a control
    # callback set then made MuJoCo's compile fail ("Python exception raised").
    motor_pd.install(Bridge)
    try:
        mujoco.MjModel.from_xml_string(ARM)
    finally:
        mujoco.set_mjcb_control(None)


def test_no_command_leaves_the_motor_off(arm):
    model, data = arm
    motor_pd.install(Bridge)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.ctrl[0] == 0.0


def test_the_pd_follows_the_joint_every_step_not_once_per_command(arm):
    model, data = arm
    motor_pd.install(Bridge)
    Bridge().LowCmdHandler(lowcmd(q=0.5, kp=20.0, kd=0.5))    # one command, then none
    assert data.ctrl[0] == 0.0           # keeping it does not touch the controls
    seen = []
    for _ in range(200):                 # 0.4 s on the one command
        q, dq = data.qpos[0], data.qvel[0]     # the joint now (sensordata is last step's)
        mujoco.mj_step(model, data)
        seen.append((data.ctrl[0], 20.0 * (0.5 - q) + 0.5 * (0.0 - dq)))
    ctrl = np.array([c for c, _ in seen])
    want = np.array([w for _, w in seen])
    assert len(np.unique(ctrl.round(9))) > 100          # recomputed as the joint moved
    np.testing.assert_allclose(ctrl, want, atol=1e-9)   # from the joint as it was at that step
    # and it got there, short by what holding the arm up against gravity
    # costs a PD: m g (l/2) / kp = 0.5 * 9.81 * 0.1 / 20 = 0.025 rad
    assert data.qpos[0] == pytest.approx(0.5, abs=0.04)


def test_a_new_command_replaces_the_last_whole(arm):
    model, data = arm
    motor_pd.install(Bridge)
    b = Bridge()
    b.LowCmdHandler(lowcmd(q=0.5, kp=20.0, kd=0.5))
    for _ in range(50):
        mujoco.mj_step(model, data)
    b.LowCmdHandler(lowcmd(tau=0.3))                     # torque only: kp, kd now zero
    mujoco.mj_step(model, data)
    assert data.ctrl[0] == pytest.approx(0.3)
