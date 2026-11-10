"""The emulated Mid-360's IMU: exact by default, a real unit's with ImuNoise.realistic()."""
import math

import numpy as np
import pytest

from go2_world.emulator import ImuNoise

DT = 1 / 200
G = np.array([0.0, 0.0, 9.81])


def test_off_is_the_truth():
    n = ImuNoise()
    g, a = n(np.array([0.1, -0.2, 0.3]), G, DT)
    assert np.allclose(g, [0.1, -0.2, 0.3]) and np.allclose(a, G)


def test_realistic_bias_and_noise():
    n = ImuNoise.realistic(seed=3)
    gs, accs = zip(*(n(np.zeros(3), G, DT) for _ in range(4000)))   # 20 s still
    gs, accs = np.array(gs), np.array(accs)
    bias = np.degrees(gs.mean(0))
    assert np.allclose(np.abs(bias), ImuNoise.GYRO_BIAS, atol=0.05)     # +-0.5 deg/s, drifted a little
    assert np.degrees(gs[:200].std(0)) == pytest.approx([ImuNoise.GYRO_NOISE] * 3, rel=0.25)
    assert np.allclose(np.abs(accs.mean(0) - G), ImuNoise.ACC_BIAS, atol=0.005)


def test_gyro_bias_can_be_set():
    n = ImuNoise.realistic(seed=1, gyro_bias=2.0)
    assert np.allclose(np.abs(np.degrees(n.gyro_b)), 2.0)
    assert math.isclose(ImuNoise.realistic(seed=1, gyro_bias=0.0).gyro_b[0], 0.0, abs_tol=1e-12)


# -- the physics' speed: gyro per wall second ------------------------------------

def test_rate_is_simulated_seconds_per_wall_second():
    from go2_world.pose import Rate
    r = Rate()
    for k in range(500):                    # 1 s of wall clock, 0.83 s simulated
        r.add(int(k * 2 * 0.83), k * 0.002)
    assert r() == pytest.approx(0.83, abs=0.01)


def test_rate_without_a_tick_is_real_time():
    from go2_world.pose import Rate
    r = Rate()
    for k in range(500):
        r.add(0, k * 0.002)                 # a world that does not stamp tick
    assert r() == 1.0


def test_install_tick_stamps_simulated_ms():
    from types import SimpleNamespace

    from go2_world.pose import install_tick

    class Bridge:
        def PublishLowState(self):
            self.sent = self.low_state.tick
    install_tick(Bridge)
    b = Bridge()
    b.mj_data, b.low_state = SimpleNamespace(time=12.3456), SimpleNamespace(tick=0)
    b.PublishLowState()
    assert b.sent == 12345
