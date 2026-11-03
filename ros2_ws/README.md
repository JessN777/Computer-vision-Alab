# Orbbec dual sync (ROS 2 workspace)

This simplified branch keeps `orbbec_dual_sync` and `r2_10_fusion_multiorient`. The older `agent_*` and `r2_01`–`r2_09` experiments are on `main` only.

Native **Windows** workflow (RoboStack / Pixi). Full details: [../ROS_SYNC.md](../ROS_SYNC.md).

## PowerShell (recommended)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
.\build.cmd
.\launch_dual_sync.cmd
```

- Prefix **`.\`** for every script in this folder (`source_install.cmd` alone is not recognized).
- **`launch_dual_sync.cmd`** / **`run_dual_sync.ps1`** are one-shot: Pixi + `install\setup.bat` + `ORBBEC_CAMERA_ROOT` + `ros2 launch` — no separate source step.

Manual env (same PowerShell session):

```powershell
.\source_install.ps1
ros2 launch orbbec_dual_sync dual_sync_detect.launch.py
```

Headless (no tkinter window): `.\launch_dual_sync.cmd preview:=false`

## Logs: preview

| Node | Default | Meaning |
|------|---------|---------|
| `orbbec_cam0_pub` / `orbbec_cam1_pub` | `preview=off` | Raw USB window disabled (normal) |
| `orbbec_sync_detect` | `Preview ON` | Side-by-side annotated sync view (~12s after launch) |

## CMD

```bat
cd /d E:\Desktop\Jessica\Camera\ros2_ws
build.cmd
launch_dual_sync.cmd
```

See [SESSION_NOTES.md](SESSION_NOTES.md) for crash-hardening and prompt guidance.

## r2_10_fusion_multiorient (vial fusion node)

Do **not** run bare `colcon` or `ros2` in PowerShell until Pixi/RoboStack and the workspace overlay are active. Use the wrappers (prefix **`.\`**):

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
.\build_r2_10.cmd
.\launch_r2_10.cmd
```

Live camera (subscribe to `/camera/color/image_raw`):

```powershell
.\launch_r2_10.cmd run_offline:=false
```

One-time RoboStack install if `build_r2_10.cmd` reports missing Pixi/env:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install_windows_ros2.ps1
```

Manual env in the same session (after build):

```powershell
.\source_install.ps1
ros2 launch r2_10_fusion_multiorient vial_detect.launch.py run_offline:=false
```

Package docs: [src/r2_10_fusion_multiorient/README.md](src/r2_10_fusion_multiorient/README.md).
