# UR teach-pendant / FK stick figure

Draws the robot from **real joint angles**, confirms against vision blue joints, and alerts on a **rack keep-out** even when blues leave the FOV.

## Quick start

1. Edit (or create) `ur_joints.json` with the six joint angles from the teach pendant:

```json
{
  "q_rad": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
  "model": "ur5e",
  "source": "teach_pendant"
}
```

2. Optional: edit `ur_fk_config.json` — model (`ur5` / `ur5e` / `ur10e`), rack keep-out AABB in `ur_base` mm, confirm gate, RTDE `host`.

3. Launch dual_view as usual. Overlay shows:
   - Green/cyan **FK0..FK6** stick figure
   - `J#->FK#` when a blue track agrees with FK (within `confirm_gate_mm`)
   - Red **RACK** alert when an FK joint enters the keep-out (works offline from vision)

## Live RTDE (optional)

```text
pip install ur_rtde
```

Set `"host": "192.168.x.x"` in `ur_fk_config.json`.

## ROS params (`orbbec_dual_view`)

| Param | Default | Meaning |
|-------|---------|---------|
| `enable_ur_fk` | true | Master switch |
| `ur_fk_draw` | true | Stick figure |
| `ur_fk_confirm` | true | Vision vs FK labels |
| `ur_fk_rack_alert` | true | Keep-out from FK |

## Align pendant base to cameras

If the skeleton is shifted, set `R_ur_to_vision` / `t_ur_to_vision_mm` in `ur_fk_config.json` after asymmetric-board / cell_frames calib so UR base matches vision `ur_base`.

An identity `R_ur_to_vision` means “not calibrated”: the graph stands the arm up (UR Z → camera up) instead of drawing it along the camera’s depth axis. Joint angles come from RTDE or `ur_joints.json`. When blue joints are visible they are a **check** — the chain is refit so those points land on the kinematic joints, and the HUD says `KIN CHECK ok` or `KIN CHECK off`. The corner inset is the upright kinematic stick, not the raw blue blobs.
