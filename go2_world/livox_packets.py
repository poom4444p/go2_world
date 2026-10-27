"""Livox SDK2 data packets, as a Mid-360 puts them on the wire.

From Livox's protocol doc for the Mid-360 and Livox-SDK2's livox_lidar_def.h
(LivoxLidarEthernetPacket). Little-endian, packed, 36-byte header:

  off size field
   0   1   version         0
   1   2   length          the whole UDP payload, header included
   3   2   time_interval   first point to last, in 0.1 us
   5   2   dot_num         points in this packet (96; 1 for IMU)
   7   2   udp_cnt         packet count in the frame, 0 at frame start
   9   1   frame_cnt       +1 a frame (Livox: not valid for non-repetitive scan)
  10   1   data_type       0 IMU, 1 xyz int32 mm, 2 xyz int16 cm, 3 spherical
  11   1   time_type       0 time since power-on, 1 PTP/gPTP, 2 GPS
  12  12   reserved
  24   4   crc32           over timestamp + data: CRC-32, init and xorout
                           0xFFFFFFFF, reflected -- zlib's
  28   8   timestamp       ns, of the first point
  36       data

The real driver (livox_ros_driver2) times point i of a packet as
timestamp + i * time_interval * 100 / dot_num ns, so that is how the
emulator fills time_interval.

This is the sending side and deliberately does not share code with
go2_board/livox.py: two readings of the same spec that agree are worth more
than one reading tested against itself.
"""
import struct
import zlib

import numpy as np

HEADER = struct.Struct("<BHHHHBBB12sI")       # up to crc32; the timestamp follows
HEADER_SIZE = 36
VERSION = 0
IMU, CARTESIAN_HIGH, CARTESIAN_LOW = 0, 1, 2
TIME_POWER_ON = 0

POINT_HIGH = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                       ("reflectivity", "u1"), ("tag", "u1")])   # 14 bytes
IMU_SAMPLE = struct.Struct("<6f")             # gyro xyz rad/s, acc xyz g


def packet(data_type, payload, timestamp_ns, dot_num, udp_cnt, frame_cnt=0,
           time_interval=0, time_type=TIME_POWER_ON):
    body = struct.pack("<Q", timestamp_ns) + payload
    head = HEADER.pack(VERSION, HEADER_SIZE + len(payload), time_interval, dot_num,
                       udp_cnt, frame_cnt, data_type, time_type, bytes(12),
                       zlib.crc32(body))
    return head + body


def points_packet(xyz_m, reflectivity, timestamp_ns, point_dt_ns, udp_cnt, frame_cnt,
                  tag=None):
    """One data type 1 packet. xyz_m is (n, 3) metres, sensor frame; a point
    with no return is (0, 0, 0), as on a real Livox."""
    n = len(xyz_m)
    pts = np.zeros(n, POINT_HIGH)
    mm = np.round(np.asarray(xyz_m) * 1000.0).astype("<i4")
    pts["x"], pts["y"], pts["z"] = mm[:, 0], mm[:, 1], mm[:, 2]
    pts["reflectivity"] = reflectivity
    pts["tag"] = 0 if tag is None else tag
    interval = min(0xFFFF, round(point_dt_ns * n / 100))
    return packet(CARTESIAN_HIGH, pts.tobytes(), timestamp_ns, n, udp_cnt & 0xFFFF,
                  frame_cnt & 0xFF, interval)


def imu_packet(gyro, acc_g, timestamp_ns, udp_cnt):
    """One data type 0 packet: gyro in rad/s, acceleration in g."""
    return packet(IMU, IMU_SAMPLE.pack(*gyro, *acc_g), timestamp_ns, 1, udp_cnt & 0xFFFF)
