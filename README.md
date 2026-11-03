# Computer vision lab (simplified)

This branch keeps the programs that actually run and removes the comparison experiments, one-off debug scripts, and label dumps that were checked in beside them. Detection behavior is unchanged. The old copies are still on `main`.

## What it does

Two Orbbec depth cameras look for a clear plastic vial with a white screw cap. A YOLO model proposes boxes. Gates then require a white cap, a see-through cup in the depth image, and a sane size. Distance is measured on the cap. The two cameras never average that distance.

## Run the cameras

From `robotics/`:

```powershell
python depthcamera.py --list-cameras
python depthcamera.py --enable-vial-detect --dual-independent
```

`--dual-independent` starts one process per camera. They compare distances over localhost UDP and still show each camera's own range. `--dual` is the older single-process mode. Free-form tracking stays off unless you pass `--enable-vial-detect`.

Weights resolve to `runs/vial-live/weights/best.pt` when that file is present (those weights are not in git). Thresholds live in `camera_gates.json`. Camera 0 is the permissive view. Camera 1 is stricter.

Train after you have labeled stills:

```powershell
python train_vial_model.py label
python train_vial_model.py split
python train_vial_model.py train
```

## Where to read the code

`robotics/depthcamera.py` is the live tracker. Read it in this order:

| Section | What it decides |
|---|---|
| `CameraGateProfile` | Per-camera confidence and clear-cup thresholds |
| `locate_pair` / `locate_standalone` | Coupled dual vs one camera alone |
| `detect_vial_by_cap`, `VialDetector` | White-cap fallback and YOLO proposals |
| `accepts_vial_box`, `looks_like_clear_cup` | Keep or reject a box |
| `measure_cap_mm` | Distance from the opaque cap |
| `OrbbecDepthCamera` | Open the camera, track, draw |
| `main` | Command-line modes |

Sideways vials use `sideways_race/methods/method_i_hard_robust.py` (built on method B2, which is built on method B). The other sideways variants were race entries and are not used by the live path.

Rack and furnace tools stay as their own scripts: `rack_lid_state.py`, `rack_capture_agents.py`, `furnace_agents.py`, `arm_agents.py`.

## ROS

`ros2_ws` still has the two packages the launch scripts build:

- `orbbec_dual_sync` — publish both cameras and detect
- `r2_10_fusion_multiorient` — upright YOLO plus the sideways detector

```powershell
cd ros2_ws
.\build.cmd
.\launch_dual_sync.cmd
```

Do not run `launch_dual_sync.cmd` and `launch_r2_10.cmd` at the same time. They both open the same USB cameras.

Round-1 `agent_*` packages and round-2 `r2_01`–`r2_09` were alternate detectors. They are not imported by the live path, so they are not on this branch.
