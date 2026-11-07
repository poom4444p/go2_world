# go2_world

The Go2's world, on the Mac only: it never goes on the dog. Physics in
Cyberdog's building, and a fake Mid-360 that speaks Livox SDK2's UDP, so
go2_board reads it with the same code it will use for the real sensor.

```
Mac                                                    J4012 (or the Mac)
  run_world.py   unitree_mujoco + building    --DDS rt/lowcmd, rt/lowstate-->   go2_board
                   |  rt/sportmodestate (position), rt/lowstate (attitude, IMU)
                   v
  run_lidar.py   raycast the building at that pose
                 -> Livox packets: points 56300->56301, IMU 56400->56401 --UDP-->   go2_board/livox.py
```

| File | What |
|---|---|
| `assets/building.xml` | Cyberdog's building, frozen: static geoms, shifted so the Go2's start is a free spot in the floor 1 corridor |
| `tools/export_building.py` | Makes `assets/building.xml` from Cyberdog's scene cache, once (reads Cyberdog, never writes it) |
| `go2_world/scene.py` | unitree_mujoco's Go2 + the building, written to `build/scene.xml` at run time |
| `go2_world/mid360.py` | The sensor: ports, rates, FOV, ray patterns, the mount on the trunk |
| `go2_world/raycast.py` | Rays into the building at a trunk pose; points in the sensor frame, (0, 0, 0) for no return |
| `go2_world/livox_packets.py` | Livox SDK2 point (type 1) and IMU (type 0) packets, CRC-32 and all |
| `go2_world/emulator.py` | 10 Hz frames cast in 10 sectors at the pose of the moment, 200 Hz IMU, noise/drop/delay |
| `go2_world/pose.py` | The pose from DDS, or a dog standing still |
| `go2_world/lift.py` | The lift car, moving: a mocap body the dog rides on contact, eased at 0.5 m/s², called to a floor over UDP 9880 (`{"call": 3}`) by whoever presses the button |
| `go2_world/crowd.py` | People walking lanes of the corridors (`--people=N`, up to 6): solid capsules the dog can bump, their positions and the lift car's sent to the LiDAR emulator (UDP 9882) so the Mid-360 sees them |
| `scripts/run_world.py` | The physics (mjpython, or `--headless` with plain python), with the lift running; `--people=N`, `--spawn=X,Y,YAW` |
| `scripts/run_lidar.py` | The fake Mid-360 |
| `scripts/run_sim.py` | The physics in unitree_mujoco's own go2 scene, no building (mjpython). Not flat: 8 cm bars at x = 1.2 and 1.6 m, stairs from 2.3 m, in front of the dog |
| `scripts/run_stand.py` | Unitree's `go2_stand_example.py`, unchanged, against the sim. Never against the real dog |
| `scripts/watch_state.py` | Body height and FR leg angles from DDS, once a second |

## Run (Mac)

```
conda activate unitree_sim
pip install -e ~/Work/unitree/go2_world              # once
mjpython scripts/run_world.py                        # terminal 1: physics in the building
python scripts/run_world.py --headless               # the same without the viewer: steadier timing
python scripts/run_lidar.py                          # terminal 2: the Mid-360
python ~/Work/unitree/go2_board/scripts/livox_watch.py   # terminal 3: what the board receives
python scripts/run_lidar.py --static                 # without the physics
mjpython scripts/run_sim.py                          # unitree_mujoco's scene instead (bars from x = 1.1 m)
python scripts/watch_state.py                        # body height from DDS
python -m pytest -q
```

Switches: `--dense` (the real unit's ~200k points/s, 210 packets a frame),
`--noise 0.02` (m), `--drop 0.01`, `--delay-ms 5 --jitter-ms 3`,
`--host <J4012 IP>` to send over Ethernet instead of loopback.

## What it does and does not copy

- Copied: packet format byte for byte, ports, 96 points a packet, (0, 0, 0) for
  no return, udp_cnt from 0 each frame, time since power-on (time_type 0), IMU
  in g and rad/s in the sensor frame, motion distortion within a frame.
- Not copied: the non-repetitive pattern (a raster is cast instead), PTP time,
  the IMU's offset from the optical centre, and reflectivity, which comes only
  from the angle of incidence.
- Not yet: the control handshake on 56100 (discovery 0x0000, parameters
  0x0100/0x0101), which livox_ros_driver2 needs before it will listen. go2_board
  does not need it.

The mount (0.20 m above the trunk origin, 20 deg nose-down) is Cyberdog's
placeholder until the real bracket exists: `mid360.MOUNT_XYZ`, `MOUNT_PITCH`.
