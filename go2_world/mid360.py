"""The Mid-360 as the emulator plays it: ports, rates, field of view, mount.

Wire format numbers are from Livox's protocol doc (livox_eth_protocol_mid360)
and Livox-SDK2's livox_lidar_def.h; see livox_packets.py. Where the emulator
departs from the real unit on purpose, it says so here.
"""
import math

import numpy as np

# Ports, sensor side -> host side (Livox defaults).
POINT_PORT, POINT_HOST_PORT = 56300, 56301
IMU_PORT, IMU_HOST_PORT = 56400, 56401
CMD_PORT = 56100                   # control handshake: not answered yet

FRAME_HZ = 10                      # the driver's default publish_freq
SLICES = 10                        # each frame is cast in 10 azimuth sectors, 10 ms apart
IMU_HZ = 200
POINTS_PER_PACKET = 96             # data types 1, 2, 3

EL_MIN, EL_MAX = -7.0, 52.0        # vertical FOV, degrees

# Ray patterns. The real unit is non-repetitive at ~200k points/s; "dense"
# matches its point and packet rate, "cyberdog" is Cyberdog's own virtual
# Mid-360 (sim/sensing/lidar.py) so both twins see the same thing.
PATTERNS = {
    #           NH   NV   min_r  max_r
    "cyberdog": (180, 11, 0.35, 12.0),
    "dense":    (720, 28, 0.10, 40.0),
}

# Mount on the Go2's trunk, sensor frame -> body frame. The Livox frame is x
# forward (the logo side), y left, z up. 20 deg nose-down and 0.5 m off the
# floor at standing height, as Cyberdog's: until the real bracket exists these
# are placeholders, and nothing below them cares what they are.
MOUNT_XYZ = (0.0, 0.0, 0.20)       # from the trunk origin (rt/sportmodestate position)
MOUNT_PITCH = 20.0                 # degrees, nose-down


def mount_rotation(pitch_deg=MOUNT_PITCH):
    """R_body_sensor: positive pitch tips the sensor's x towards the floor."""
    p = math.radians(pitch_deg)
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def ray_pattern(nh, nv):
    """(nh*nv, 3) unit rays in the sensor frame, azimuth-major so that any
    contiguous run of them is one sector of the sweep."""
    az = np.linspace(0.0, 2 * math.pi, nh, endpoint=False)
    el = np.radians(np.linspace(EL_MIN, EL_MAX, nv))
    A, E = np.meshgrid(az, el, indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)],
                    axis=-1).reshape(-1, 3)
