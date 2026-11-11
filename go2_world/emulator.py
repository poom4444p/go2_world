"""A Mid-360 on UDP: points 56300 -> 56301, IMU 56400 -> 56401, 10 Hz frames.

Each frame is cast in SLICES azimuth sectors, one every 10 ms at the pose of
that moment, so a dog that turns mid-frame bends its own scan the way a real
one does and anything that undoes motion distortion (FAST-LIO) has real work
to do. Points are 96 to a packet; a ray with no return is a (0, 0, 0) point,
as on the real unit, so every frame has the same packet count. The frame's
last packet is padded with such points. udp_cnt starts again at 0 each frame.

Time is time since power-on (time_type 0), on an ideal clock: packet times are
where the schedule says they are, not when Python got round to them.

Impairments, all off by default: range noise, packet drop, and delay with
jitter (which also reorders, as a real network can).
"""
import errno
import heapq
import math
import socket
import time

import numpy as np

from go2_world import livox_packets as lp
from go2_world import mid360
from go2_world.pose import G
from go2_world.raycast import reflectivity


class Impairments:
    def __init__(self, noise_m=0.0, drop=0.0, delay_ms=0.0, jitter_ms=0.0, seed=0):
        self.noise_m, self.drop = noise_m, drop
        self.delay_s, self.jitter_s = delay_ms / 1e3, jitter_ms / 1e3
        self.rng = np.random.default_rng(seed)


class ImuNoise:
    """What a real Mid-360's IMU adds to the truth, per axis, in its own frame.

    The defaults (realistic()) are a little worse than the ICM-40609 the unit
    carries, as best known without one on the bench: white noise from its
    noise densities at 200 Hz, a turn-on gyro bias of +-GYRO_BIAS (each axis
    a random sign), drifting as a random walk, and the accelerometer's offset
    left after a level calibration (an uncalibrated one tilts "level" by about
    1 deg, which the board cannot learn: it would be calibrated on the dog).
    Off -- ImuNoise() -- the IMU is exactly the physics, as before."""

    GYRO_NOISE = 0.05        # deg/s, 1 sigma per sample (0.0038 deg/s/sqrt(Hz) * 10)
    GYRO_BIAS = 0.5          # deg/s, turn-on bias per axis
    GYRO_WALK = 0.002        # deg/s/sqrt(s), bias random walk
    ACC_NOISE = 0.01         # m/s^2, 1 sigma per sample (70 ug/sqrt(Hz) * 10)
    ACC_BIAS = 0.03          # m/s^2 per axis left after calibration (~0.2 deg of tilt)

    def __init__(self, gyro_noise=0.0, gyro_bias=0.0, gyro_walk=0.0, acc_noise=0.0,
                 acc_bias=0.0, seed=0):
        self.rng = np.random.default_rng(seed)
        r = math.radians
        self.gyro_noise, self.gyro_walk = r(gyro_noise), r(gyro_walk)
        self.gyro_b = r(gyro_bias) * self.rng.choice([-1.0, 1.0], 3)
        self.acc_noise = acc_noise
        self.acc_b = acc_bias * self.rng.choice([-1.0, 1.0], 3)

    @classmethod
    def realistic(cls, seed=0, gyro_bias=None):
        return cls(cls.GYRO_NOISE, cls.GYRO_BIAS if gyro_bias is None else gyro_bias,
                   cls.GYRO_WALK, cls.ACC_NOISE, cls.ACC_BIAS, seed)

    def __call__(self, gyro, acc, dt):
        """(gyro rad/s, acc m/s^2) as measured, dt s after the last sample."""
        if self.gyro_walk:
            self.gyro_b = self.gyro_b + self.rng.normal(0.0, self.gyro_walk * math.sqrt(dt), 3)
        g = gyro + self.gyro_b
        a = acc + self.acc_b
        if self.gyro_noise:
            g = g + self.rng.normal(0.0, self.gyro_noise, 3)
        if self.acc_noise:
            a = a + self.rng.normal(0.0, self.acc_noise, 3)
        return g, a


class UdpOut:
    """Sends from the sensor's own ports, as the real unit does.

    A failed send is counted, not raised (`failed`, by errno name): nobody
    listening is no concern of a sensor's, but a full send buffer is packets
    lost on this side, and it was silent. With the dense pattern in a full HIL
    run, a burst of catching-up slices overflowed it and the board got frames
    of a third of a sweep (HIL 11-11 14:51)."""

    SNDBUF = 4 << 20                          # bytes: room for a few frames' bursts

    def __init__(self, host="127.0.0.1", bind="0.0.0.0"):
        self.host = host
        self.socks = {}
        self.failed = {}
        for src, dst in ((mid360.POINT_PORT, mid360.POINT_HOST_PORT),
                         (mid360.IMU_PORT, mid360.IMU_HOST_PORT)):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.SNDBUF)
            except OSError:
                pass                          # the OS's own limit, then
            s.bind((bind, src))
            self.socks[dst] = s

    def __call__(self, data, port):
        try:
            self.socks[port].sendto(data, (self.host, port))
        except OSError as e:
            name = errno.errorcode.get(e.errno, str(e.errno))
            self.failed[name] = self.failed.get(name, 0) + 1

    def close(self):
        for s in self.socks.values():
            s.close()


class Emulator:
    def __init__(self, raycaster, pose, out, impair=None, clock=time.monotonic, imu_noise=None):
        self.ray, self.pose, self.out = raycaster, pose, out
        self.imu_noise = imu_noise or ImuNoise()
        self.imp = impair or Impairments()
        self.clock = clock
        self.t0 = clock()
        self.frame_s = 1.0 / mid360.FRAME_HZ
        self.slice_s = self.frame_s / mid360.SLICES
        self.imu_s = 1.0 / mid360.IMU_HZ
        self.point_dt_ns = self.frame_s * 1e9 / raycaster.n
        self.bounds = [len(a) for a in np.array_split(np.arange(raycaster.n), mid360.SLICES)]
        self.bounds = np.concatenate([[0], np.cumsum(self.bounds)])
        self.frame_cnt = self.udp_cnt = self.imu_cnt = 0
        self.slice = 0
        self.frame_t = 0.0                     # this frame's start, s since power-on
        self.next_slice = self.next_imu = 0.0
        self.buf = []                          # (xyz, refl) waiting to fill a packet
        self.late = []                         # (release, n, data, port): delayed packets
        self.sent = self.dropped = self.dropped_points = self.queued = 0
        self.frames_sent = self.frames_skipped = self.slices_late = 0

    # -- what goes out ---------------------------------------------------------

    def _send(self, data, port, now):
        imp = self.imp
        if imp.drop and imp.rng.random() < imp.drop:
            self.dropped += 1
            self.dropped_points += port == mid360.POINT_HOST_PORT
            return
        if imp.delay_s or imp.jitter_s:
            release = now + imp.delay_s + imp.rng.uniform(0.0, imp.jitter_s)
            self.queued += 1
            heapq.heappush(self.late, (release, self.queued, data, port))
            return
        self.out(data, port)
        self.sent += 1

    def _release(self, now):
        while self.late and self.late[0][0] <= now:
            _, _, data, port = heapq.heappop(self.late)
            self.out(data, port)
            self.sent += 1

    def _packets(self, now, pad=False):
        """Emit the buffered points 96 at a time; pad=True flushes the rest."""
        n = mid360.POINTS_PER_PACKET
        xyz = np.concatenate([b[0] for b in self.buf]) if self.buf else np.zeros((0, 3))
        refl = np.concatenate([b[1] for b in self.buf]) if self.buf else np.zeros(0, np.uint8)
        if pad and len(xyz) % n:
            k = n - len(xyz) % n
            xyz = np.vstack([xyz, np.zeros((k, 3))])
            refl = np.concatenate([refl, np.zeros(k, np.uint8)])
        full = len(xyz) // n * n
        for i in range(0, full, n):
            first = self.udp_cnt * n                       # index of its first point in the frame
            stamp = round((self.frame_t * 1e9) + first * self.point_dt_ns)
            pkt = lp.points_packet(xyz[i:i + n], refl[i:i + n], stamp, self.point_dt_ns,
                                   self.udp_cnt, self.frame_cnt)
            self._send(pkt, mid360.POINT_HOST_PORT, now)
            self.udp_cnt += 1
        self.buf = [(xyz[full:], refl[full:])] if full < len(xyz) else []

    # -- the schedule ----------------------------------------------------------

    def _do_slice(self, now):
        p = self.pose.read()
        lo, hi = self.bounds[self.slice], self.bounds[self.slice + 1]
        if p is None:                          # no pose yet: a sensor still sees nothing
            pts, cos = np.zeros((hi - lo, 3)), np.zeros(hi - lo)
        else:
            pts, cos = self.ray.cast(p.pos, p.quat, lo, hi)
        if self.imp.noise_m:
            hit = cos > 0
            r = np.linalg.norm(pts[hit], axis=1)
            noisy = r + self.imp.rng.normal(0.0, self.imp.noise_m, len(r))
            pts[hit] *= (np.maximum(noisy, 0.0) / r)[:, None]
        self.buf.append((pts, reflectivity(cos, self.imp.rng if self.imp.noise_m else None)))
        last = self.slice == mid360.SLICES - 1
        self._packets(now, pad=last)
        self.slice += 1
        if last:
            self.frames_sent += 1
            self.slice, self.udp_cnt = 0, 0
            self.frame_cnt = (self.frame_cnt + 1) & 0xFF
            self.frame_t += self.frame_s

    def _do_imu(self, now, t):
        p = self.pose.read()
        if p is None:
            return
        R_sb = self.ray.R_bs.T                 # the IMU sits in the sensor, axes as the points
        # per wall second, as a real-time IMU turns: see pose.Rate
        gyro, acc = self.imu_noise(R_sb @ p.gyro * getattr(p, "rtf", 1.0), R_sb @ p.acc, self.imu_s)
        pkt = lp.imu_packet(gyro, acc / G, round(t * 1e9), self.imu_cnt)
        self.imu_cnt += 1
        self._send(pkt, mid360.IMU_HOST_PORT, now)

    def step(self, now=None):
        """Do whatever is due. Returns the time of the next thing that will be."""
        now = (self.clock() if now is None else now) - self.t0
        while self.next_slice <= now:
            if self.slice == 0 and now - self.next_slice >= self.frame_s:
                # A whole frame or more behind as a new one is due: skip whole
                # frames, as a sensor that missed them would (frame_cnt moves
                # on), rather than cast them all now and send them in a burst.
                # A frame already begun is finished, so whatever is sent is a
                # whole sweep.
                k = int((now - self.next_slice) // self.frame_s)
                self.next_slice += k * self.frame_s
                self.frame_t += k * self.frame_s
                self.frame_cnt = (self.frame_cnt + k) & 0xFF
                self.frames_skipped += k
                continue
            if now - self.next_slice > self.slice_s:
                self.slices_late += 1
            self._do_slice(now)
            self.next_slice += self.slice_s
        while self.next_imu <= now:
            self._do_imu(now, self.next_imu)
            self.next_imu += self.imu_s
        self._release(now)
        nxt = min(self.next_slice, self.next_imu, self.late[0][0] if self.late else math.inf)
        return nxt + self.t0

    def status(self):
        """One line of how it is keeping up, for the log."""
        failed = getattr(self.out, "failed", {})
        return (f"mid360: {self.frames_sent} frames sent, {self.frames_skipped} skipped late, "
                f"{self.slices_late} slices late, send errors "
                + (", ".join(f"{k} {v}" for k, v in sorted(failed.items())) or "none"))

    def run(self, seconds=math.inf):
        end = self.clock() + seconds
        while self.clock() < end:
            due = self.step()
            time.sleep(max(0.0, min(due, end) - self.clock()))
