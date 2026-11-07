"""The motors' PD loop, every physics step, as the Go2's motor drivers run it.

unitree_mujoco's bridge works the torque out of a LowCmd once, when it
arrives (`tau + kp (q - q_now) + kd (dq - dq_now)`), and the physics then
holds that torque until the next one. With go2_board on this Mac its
commands come at 500 Hz, one per 2 ms step, and nothing is held. With the
board on the J4012 they came at 415-420 Hz and up to 15 ms apart: for that
long the torque stayed what it was when the joint was somewhere else, and
the damping did nothing. The walker that walks with the board here fell,
every time, within two seconds of starting with the board there.

The dog's own drivers keep the last command and close the loop on it
thousands of times a second. So here: the bridge's handler only keeps the
last command, and MuJoCo's control callback (run inside every mj_step, after
the joint sensors) works the torque out of it and the joints as they are.
"""
import mujoco
import numpy as np


class MotorPD:
    """The last LowCmd, and the PD on it at every step."""

    def __init__(self):
        self.n = 0                         # motors: the bridge's num_motor, from the first command
        self.cmd = None                    # (q, dq, kp, kd, tau), replaced whole

    def keep(self, msg, n):
        """The bridge's LowCmd handler: keep the command, nothing more --
        and on the first, hand MuJoCo the callback. Not before: MuJoCo calls
        it while compiling a model too, and the bindings cannot hand Python
        a model that is half made ("engine error: Python exception raised")."""
        if self.cmd is None:
            mujoco.set_mjcb_control(self.control)
        m = msg.motor_cmd
        self.n = n
        self.cmd = (np.array([m[i].q for i in range(n)]),
                    np.array([m[i].dq for i in range(n)]),
                    np.array([m[i].kp for i in range(n)]),
                    np.array([m[i].kd for i in range(n)]),
                    np.array([m[i].tau for i in range(n)]))

    def control(self, model, data):
        """mjcb_control: the torque from the last command and the joints now.
        Before any command, the controls are left as they are (zero)."""
        cmd = self.cmd
        if cmd is None:
            return
        q, dq, kp, kd, tau = cmd
        n = len(q)
        data.ctrl[:n] = tau + kp * (q - data.sensordata[:n]) + kd * (dq - data.sensordata[n:2 * n])


def install(bridge_cls):
    """Make unitree_mujoco's bridge keep commands and MuJoCo run the PD.
    Call before the bridge is made: it subscribes its LowCmdHandler then.
    The control callback goes in with the first command (MotorPD.keep)."""
    pd = MotorPD()
    bridge_cls.LowCmdHandler = lambda self, msg: pd.keep(msg, self.num_motor)
    return pd
