# orbbec_dual_sync

ROS 2 package for three Orbbec cameras + UR arm overlay.

## How to read the code

Start here if you want the live window:

1. `orbbec_dual_sync/dual_view_node.py` — composes cam0 / cam1 / cam2 and draws overlays
2. `orbbec_dual_sync/online_cam_pose.py` — rolling camera pose from ArUco tags (+ optional blues)
3. `orbbec_dual_sync/ur_skeleton_draw.py` — projects FK joints into the image
4. `orbbec_dual_sync/blue_joint_tracks.py` — tracks baby-blue joint caps
5. `orbbec_dual_sync/baby_blue_draw.py` — HSV masks, depth sampling, proximity
6. `orbbec_dual_sync/cell_frames.py` — saved camera poses (`cell_frames.json`)
7. `orbbec_dual_sync/camera_publisher.py` — opens one Orbbec and publishes color / depth / camera_info

Robotics helpers (outside this package, on `PYTHONPATH`):

- `robotics/ur_skeleton.py` — live joint angles + FK
- `robotics/aruco_marker.py` — joint ArUco tags
- `robotics/depthcamera.py` — Orbbec open / vial detect (used by publishers and sync_detect)

## Other nodes

| Module | Role |
| --- | --- |
| `sync_detect_node.py` | Approximate-time sync + vial detect preview |
| `detect_bridge.py` | Spawned YOLO / cap worker for sync_detect |
| `furnace_overlay_*.py` | Furnace slot boxes |
| `rack_overlay_*.py` | Rack OPEN / CLOSED |
| `approx_time.py` | 2-stream stamp sync (no message_filters dependency) |
| `preview_gui.py` | tkinter / optional HighGUI preview window |

## Constraints

This machine’s NumPy GEMM aborts the process. Do not use `@`, `np.dot`, or `np.matmul` in the overlay / pose path. Prefer OpenCV (`projectPoints`, `solvePnP*`, `Rodrigues`) or explicit loops.

## Install copies

After editing package modules, copy them into both:

- `ros2_ws/install/Lib/orbbec_dual_sync/`
- `ros2_ws/install/Lib/site-packages/orbbec_dual_sync/`

Restart the viewer; it does not hot-reload.
