"""UR forward kinematics + joint-angle sources for stick-figure / crash assist.

Joint sources (first that works):
  1. Optional RTDE (``ur_rtde`` or ``rtde_receive``) when ``host`` is set
  2. JSON file ``ur_joints.json`` (teach-pendant / external writer)
  3. Mock zeros (draw at home) for offline overlay tests

    python -c "from ur_skeleton import fk_joint_positions_mm, load_joints; print(fk_joint_positions_mm(load_joints()[0]))"
"""

from __future__ import annotations

import json
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Drawn joint = (W_FK * FK_pixel + W_BLUE * blue_pixel) / (W_FK + W_BLUE).
# About 1:1. Raise W_FK to trust the teach pendant more, W_BLUE to trust the mask.
W_FK = 1.0
W_BLUE = 1.0
BLEND_MIN_BLUES = 2


def _mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Small matrix product without BLAS (this machine's GEMM aborts)."""
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    n, m = A.shape
    p = B.shape[1]
    out = np.zeros((n, p), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            aik = float(A[i, k])
            if aik == 0.0:
                continue
            for j in range(p):
                out[i, j] += aik * float(B[k, j])
    return out


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gaussian elimination for the 7-parameter kinematic step."""
    M = np.asarray(a, dtype=np.float64).copy()
    x = np.asarray(b, dtype=np.float64).reshape(-1).copy()
    n = int(x.size)
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(float(M[pivot, col])) < 1e-12 or not math.isfinite(float(M[pivot, col])):
            x[col] = 0.0
            continue
        if pivot != col:
            row = M[col].copy()
            M[col] = M[pivot]
            M[pivot] = row
            x[col], x[pivot] = float(x[pivot]), float(x[col])
        div = float(M[col, col])
        M[col, :] /= div
        x[col] = float(x[col]) / div
        for r in range(n):
            if r == col:
                continue
            factor = float(M[r, col])
            if factor == 0.0:
                continue
            M[r, :] -= factor * M[col, :]
            x[r] = float(x[r]) - factor * float(x[col])
    for i in range(n):
        if not math.isfinite(float(x[i])):
            x[i] = 0.0
    return x

ROOT = Path(__file__).resolve().parent
JOINTS_PATH = ROOT / "ur_joints.json"
CONFIG_PATH = ROOT / "ur_fk_config.json"

# DH (meters) — Universal Robots published parameters (approx by model family).
UR_MODELS_M = {
    "ur5": {
        "d": [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823],
        "a": [0.0, -0.42500, -0.39225, 0.0, 0.0, 0.0],
    },
    "ur5e": {
        "d": [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996],
        "a": [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0],
    },
    # UR7e is this same chain. UR7e.step (Downloads) puts the shoulder axis
    # 162.5 mm up, the elbow 425 mm out, and the wrist housings at radius 37.5 mm.
    "ur7e": {
        "d": [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996],
        "a": [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0],
    },
    "ur10e": {
        "d": [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655],
        "a": [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0],
    },
}
UR_ALPHAS = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]


@dataclass
class UrFkConfig:
    model: str = "ur5e"
    host: str = ""  # controller IP; empty = file/mock only
    port: int = 30003  # UR real-time stream (actual q). RTDE 30004 needs ur_rtde.
    joints_path: Path = JOINTS_PATH
    # Offset: p_vision_base = R @ p_ur_base + t_mm (align pendant base to cell ur_base)
    R_ur_to_vision: np.ndarray | None = None
    t_ur_to_vision_mm: np.ndarray | None = None
    rack_keepout_min_mm: tuple[float, float, float] = (200.0, -400.0, 0.0)
    rack_keepout_max_mm: tuple[float, float, float] = (700.0, 400.0, 500.0)
    confirm_gate_mm: float = 80.0
    proximity_mm: float = 100.0


def load_fk_config(path: Path | None = None) -> UrFkConfig:
    path = path or CONFIG_PATH
    cfg = UrFkConfig()
    if not path.is_file():
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return cfg
    cfg.model = str(data.get("model", cfg.model))
    cfg.host = str(data.get("host", cfg.host) or "")
    try:
        cfg.port = int(data.get("port", cfg.port) or cfg.port)
    except (TypeError, ValueError):
        cfg.port = 30003
    jp = data.get("joints_path")
    if jp:
        cfg.joints_path = Path(jp)
        if not cfg.joints_path.is_absolute():
            cfg.joints_path = ROOT / cfg.joints_path
    if data.get("R_ur_to_vision") is not None:
        cfg.R_ur_to_vision = np.asarray(data["R_ur_to_vision"], dtype=np.float64).reshape(3, 3)
    if data.get("t_ur_to_vision_mm") is not None:
        cfg.t_ur_to_vision_mm = np.asarray(
            data["t_ur_to_vision_mm"], dtype=np.float64
        ).reshape(3)
    if data.get("rack_keepout_min_mm") is not None:
        cfg.rack_keepout_min_mm = tuple(float(v) for v in data["rack_keepout_min_mm"])  # type: ignore
    if data.get("rack_keepout_max_mm") is not None:
        cfg.rack_keepout_max_mm = tuple(float(v) for v in data["rack_keepout_max_mm"])  # type: ignore
    cfg.confirm_gate_mm = float(data.get("confirm_gate_mm", cfg.confirm_gate_mm))
    cfg.proximity_mm = float(data.get("proximity_mm", cfg.proximity_mm))
    return cfg


def write_example_files() -> None:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "model": "ur5e",
                    "host": "",
                    "joints_path": "ur_joints.json",
                    "R_ur_to_vision": np.eye(3).tolist(),
                    "t_ur_to_vision_mm": [0.0, 0.0, 0.0],
                    "rack_keepout_min_mm": [200.0, -400.0, 0.0],
                    "rack_keepout_max_mm": [700.0, 400.0, 500.0],
                    "confirm_gate_mm": 80.0,
                    "proximity_mm": 100.0,
                    "notes": (
                        "Write q_rad into ur_joints.json from teach pendant / script, "
                        "or set host to robot IP for RTDE when ur_rtde is installed."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not JOINTS_PATH.exists():
        JOINTS_PATH.write_text(
            json.dumps(
                {
                    "q_rad": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
                    "model": "ur5e",
                    "source": "example",
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _validation_kinematics():
    """Load Useful Validation Scripts/ur5e_kinematics without executing its GEMM.

    ``ur_fk`` / ``ur_fk_T`` multiply with ``@``, which aborts this interpreter.
    Callers use that module's ``_dh`` and the UR5e length arrays only.
    """
    import sys

    root = Path(r"C:\Users\Ceder-ALAB\Desktop\Useful Validation Scripts")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ur5e_kinematics as uk  # noqa: WPS433

    return uk


def fk_transforms(q_rad: np.ndarray, model: str = "ur5e") -> list[np.ndarray]:
    """Return T_0_i (4x4) for i=0..6 (base + 6 joints) in UR base meters.

    UR5e uses ``ur5e_kinematics._dh`` and ``UR5E_D`` / ``UR5E_A`` / ``UR5E_ALPHA``.
    The chain product is a loop, not ``ur_fk``, because that function's ``@``
    aborts here.
    """
    q = np.asarray(q_rad, dtype=np.float64).reshape(6)
    use_validation = model in ("ur5e", "ur7e", "ur5e_validation", "")
    uk = _validation_kinematics() if use_validation else None
    T = np.eye(4, dtype=np.float64)
    out = [T.copy()]
    for i in range(6):
        if uk is not None:
            # ur5e_kinematics._dh(q, a, d, alpha) — their argument order.
            Ti = uk._dh(
                float(q[i]),
                float(uk.UR5E_A[i]),
                float(uk.UR5E_D[i]),
                float(uk.UR5E_ALPHA[i]),
            )
        else:
            key = model if model in UR_MODELS_M else "ur5"
            Ti = _dh(
                float(q[i]),
                float(UR_MODELS_M[key]["d"][i]),
                float(UR_MODELS_M[key]["a"][i]),
                float(UR_ALPHAS[i]),
            )
        T = _mm(T, np.asarray(Ti, dtype=np.float64))
        out.append(T.copy())
    return out


# urp_viewer.UrpViewer / ProgramSimulator world placement.
# Home rail puts the base at the UR origin. Tool offset is zero: TCP is the flange.
RAIL_HOME_UNITS = 3584.0
RAIL_M_PER_UNIT = 0.001
TOOL_OFFSET_M = (0.0, 0.0, 0.0)
# Gripper reach past the wrist flange, along the flange approach (+Z local).
# The chain's last drawn point lands on the gripper, not the bare flange.
# Past the flange, along the tool approach, so the last cam0 point reaches
# the gripper fingers. 180 mm stopped about 90 px short of the fingers.
GRIPPER_LEN_MM = 340.0
# Outer radius of each link, measured from the solid vertices and the r=58 / r=37.5
# cylinders in UR7e.step. Index i is the tube between joint i and joint i+1.
# 0 base column (footprint diameter 149 mm), 1 shoulder-to-elbow, 2 forearm,
# 3..5 wrist housings.
UR7E_LINK_RADIUS_MM = (74.0, 58.0, 58.0, 37.5, 37.5, 37.5)
# ProgramSimulator start pose when no .urp / RTDE log is loaded.
VIEWER_DEMO_Q = np.array([-0.05, -1.2, -1.6, -1.9, -4.7, 0.1], dtype=np.float64)


def rail_shift_m(rail_units: float | None = None) -> float:
    """Meters along UR +X, matching ``UrpViewer._rail_m`` at the default scale."""
    units = RAIL_HOME_UNITS if rail_units is None else float(rail_units)
    return (units - RAIL_HOME_UNITS) * RAIL_M_PER_UNIT


def fk_joint_positions_mm(
    q_rad: np.ndarray, model: str = "ur5e", rail_units: float | None = None
) -> np.ndarray:
    """(7,3) link origins in the validation viewer's world frame, millimeters.

    Same points as ``ur_fk`` (meters there), then ``links[:, 0] += rail`` from
    ``urp_viewer.redraw``. Home rail (3584) adds 0. No tool point is appended:
    ``ur5e_kinematics`` treats the flange as the TCP.
    """
    Ts = fk_transforms(q_rad, model=model)
    pts = np.zeros((7, 3), dtype=np.float64)
    dx = rail_shift_m(rail_units) * 1000.0
    for i, T in enumerate(Ts):
        pts[i, 0] = float(T[0, 3]) * 1000.0 + dx
        pts[i, 1] = float(T[1, 3]) * 1000.0
        pts[i, 2] = float(T[2, 3]) * 1000.0
    return pts


# UR base is Z-up. Cam0's ur_base is the optical frame (X right, Y down, Z forward)
# until a real mount rotation is saved. Identity R therefore graphs the arm on its side.
R_UR_Z_UP_TO_CAM_Y_DOWN = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


# File example is an L. The validation viewer does not draw that L.
# Its simulator starts at VIEWER_DEMO_Q (urp_viewer.ProgramSimulator).
FILE_L_Q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)
# Photographed cell pose, used only while the teach pendant stream is down.
PHOTO_Q = np.array([0.0, -2.618, 1.5708, 0.0, 0.0, 0.0], dtype=np.float64)
PHOTO_YAW = -math.pi / 2
# Live overlay heading. Encoder q stays locked. yaw spins the arm about
# vertical before the mount; img_deg / img_scale seat that projection on the
# photograph (upper arm up-right, forearm back along the arm). Joint 0 stays
# on the pedestal anchor. Blues still only blend pixels 1:1.
LIVE_VIEW_ALIGN = {
    "cam0": {"yaw": math.radians(34.0), "img_deg": 44.0, "img_scale": 0.62},
    "cam1": {"yaw": math.radians(-100.0), "img_deg": 40.0, "img_scale": 0.45},
}


def is_example_l(q_rad: np.ndarray) -> bool:
    q = np.asarray(q_rad, dtype=np.float64).reshape(6)
    return float(np.max(np.abs(q - FILE_L_Q))) < 0.08


def seed_photo_if_example(
    q_rad: np.ndarray, yaw: float, *, locked: bool = False
) -> tuple[np.ndarray, float]:
    """Replace the canned L with the validation viewer's start pose.

    RTDE q is left alone. Yaw is not changed: their redraw applies no extra
    spin, only the rail shift along UR +X (0 at the home rail).
    """
    q = np.asarray(q_rad, dtype=np.float64).reshape(6).copy()
    if locked or not is_example_l(q):
        return q, float(yaw)
    return VIEWER_DEMO_Q.copy(), float(yaw)


def mount_R(cfg: UrFkConfig) -> np.ndarray:
    """Rotation from UR base (Z up) into the vision frame the blue points use."""
    R = cfg.R_ur_to_vision
    if R is None or np.allclose(R, np.eye(3), atol=1e-3):
        return R_UR_Z_UP_TO_CAM_Y_DOWN.copy()
    return np.asarray(R, dtype=np.float64).reshape(3, 3)


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def ur_base_to_vision_mm(
    pts_ur_mm: np.ndarray, cfg: UrFkConfig, *, yaw: float = 0.0
) -> np.ndarray:
    """Map UR-base points into vision ur_base (mm).

    ``yaw`` spins the arm about vertical before the mount rotation, so the
    kinematic chain can line up with the cell even when pendant X is unknown.
    """
    P = np.asarray(pts_ur_mm, dtype=np.float64).reshape(-1, 3)
    R = _mm(mount_R(cfg), _rz(float(yaw)))
    t = (
        cfg.t_ur_to_vision_mm
        if cfg.t_ur_to_vision_mm is not None
        else np.zeros(3)
    )
    return _mm(P, R.T) + t.reshape(3)


def upright_points_mm(q_rad: np.ndarray, yaw: float, model: str = "ur5e") -> np.ndarray:
    """(7,3) kinematic joints in a Z-up frame after base yaw. For the stick graph."""
    pts = fk_joint_positions_mm(q_rad, model=model)
    return _mm(pts, _rz(float(yaw)).T)


def tool_point_upright_mm(
    q_rad: np.ndarray,
    yaw: float,
    model: str = "ur5e",
    *,
    tool_len_mm: float = GRIPPER_LEN_MM,
    rail_units: float | None = None,
) -> np.ndarray:
    """Gripper tip (3,) in the same Z-up frame as ``upright_points_mm``.

    The tip is the flange origin pushed ``tool_len_mm`` along the flange's
    local +Z (the tool approach axis), so the last bone reaches the gripper.
    """
    Ts = fk_transforms(q_rad, model=model)
    T6 = Ts[6]
    dx = rail_shift_m(rail_units) * 1000.0
    px = float(T6[0, 3]) * 1000.0 + dx
    py = float(T6[1, 3]) * 1000.0
    pz = float(T6[2, 3]) * 1000.0
    zx, zy, zz = float(T6[0, 2]), float(T6[1, 2]), float(T6[2, 2])
    tip = np.array(
        [
            [px + zx * float(tool_len_mm), py + zy * float(tool_len_mm), pz + zz * float(tool_len_mm)]
        ],
        dtype=np.float64,
    )
    return _mm(tip, _rz(float(yaw)).T)[0]


def append_tool_points(
    pts_upright_mm: np.ndarray,
    pts_vision_mm: np.ndarray,
    q_rad: np.ndarray,
    yaw: float,
    cfg: "UrFkConfig",
    *,
    model: str = "ur5e",
    tool_len_mm: float = GRIPPER_LEN_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the chain with a gripper tip appended (upright, vision) in mm.

    The tip extends past the flange so the final drawn point ends at the gripper.
    FK joints 0..6 are unchanged; only an 8th endpoint is added.
    """
    up = np.asarray(pts_upright_mm, dtype=np.float64).reshape(-1, 3)
    vis = np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 3)
    if up.shape[0] < 7 or vis.shape[0] < 7:
        return up, vis
    tip_up = tool_point_upright_mm(q_rad, yaw, model=model, tool_len_mm=tool_len_mm)
    R = mount_R(cfg)
    t = (
        cfg.t_ur_to_vision_mm
        if cfg.t_ur_to_vision_mm is not None
        else np.zeros(3, dtype=np.float64)
    )
    tip_vis = _mm(tip_up.reshape(1, 3), R.T)[0] + np.asarray(t, dtype=np.float64).reshape(3)
    up8 = np.vstack([up, tip_up.reshape(1, 3)])
    vis8 = np.vstack([vis, tip_vis.reshape(1, 3)])
    return up8, vis8


def _assign_blues(
    obs: np.ndarray,
    pts: np.ndarray,
    *,
    max_mm: float = 450.0,
    min_joint: int = 1,
    origin: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """Match blues outward from the robot base.

    Joint 0 is the base and is not claimed by a free blue detection.
    Remaining detections are walked near-to-far from the base and paired
    with the next kinematic joint along the chain.
    """
    if obs.size == 0 or pts.size == 0:
        return []
    if origin is None:
        origin = np.asarray(pts[0], dtype=np.float64)
    else:
        origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    if not np.isfinite(origin).all():
        origin = np.zeros(pts.shape[1], dtype=np.float64)
    order = sorted(
        range(obs.shape[0]),
        key=lambda i: float(np.sum((obs[i] - origin) ** 2)),
    )
    pairs: list[tuple[int, int]] = []
    next_j = max(0, int(min_joint))
    for i in order:
        best_j: Optional[int] = None
        best_d = float(max_mm)
        for j in range(next_j, pts.shape[0]):
            if not np.isfinite(pts[j]).all():
                continue
            delta = obs[i] - pts[j]
            dist = float(np.sqrt(np.sum(delta * delta)))
            if dist < best_d:
                best_d = dist
                best_j = j
        if best_j is None:
            continue
        pairs.append((i, int(best_j)))
        next_j = int(best_j) + 1
    return pairs


def blend_fk_blue_pixels(
    fk_uv: np.ndarray,
    blue_uv: np.ndarray,
    *,
    min_joint: int = 1,
    max_px: float = 900.0,
) -> tuple[np.ndarray, int, Optional[float]]:
    """Equal-weight compromise between the encoder projection and blue pixels.

    Joint 0 is never a blue (the pedestal tape owns it). A joint with no valid
    blue stays on the FK pixel, so a dropped blob cannot collapse the arm.
    Fewer than BLEND_MIN_BLUES matched blues returns the pure FK projection.
    """
    fk = np.asarray(fk_uv, dtype=np.float64).reshape(-1, 2).copy()
    for i in range(fk.shape[0]):
        if not np.isfinite(fk[i]).all():
            fk[i, 0] = 0.0
            fk[i, 1] = 0.0
    raw = np.asarray(blue_uv, dtype=np.float64).reshape(-1, 2)
    kept: list[np.ndarray] = []
    for i in range(raw.shape[0]):
        if np.isfinite(raw[i]).all():
            kept.append(raw[i].copy())
    if len(kept) < BLEND_MIN_BLUES or fk.shape[0] < 2:
        return fk, 0, None
    obs = np.stack(kept, axis=0)
    origin = fk[0].copy()
    pairs = _assign_blues(
        obs, fk, max_mm=float(max_px), min_joint=int(min_joint), origin=origin
    )
    pairs = [
        (int(i), int(j))
        for i, j in pairs
        if int(j) >= int(min_joint) and np.isfinite(obs[int(i)]).all()
    ]
    if len(pairs) < BLEND_MIN_BLUES:
        return fk, 0, None
    w_sum = float(W_FK) + float(W_BLUE)
    if (not math.isfinite(w_sum)) or w_sum <= 1e-9:
        return fk, 0, None
    acc = 0.0
    used = 0
    for i, j in pairs:
        blue = obs[i]
        pred = fk[j]
        u = (float(W_FK) * float(pred[0]) + float(W_BLUE) * float(blue[0])) / w_sum
        v = (float(W_FK) * float(pred[1]) + float(W_BLUE) * float(blue[1])) / w_sum
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        fk[j, 0] = u
        fk[j, 1] = v
        du = u - float(blue[0])
        dv = v - float(blue[1])
        acc += math.sqrt(du * du + dv * dv)
        used += 1
    fk[0] = origin
    if used < BLEND_MIN_BLUES:
        return np.asarray(fk_uv, dtype=np.float64).reshape(-1, 2).copy(), 0, None
    return fk, used, acc / float(used)


def _pair_rms(obs: np.ndarray, pts: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 1.0e6
    err = [float(np.linalg.norm(obs[i] - pts[j])) for i, j in pairs]
    return float(np.sqrt(np.mean(np.square(err))))


def _coverage_rms(
    obs: np.ndarray,
    pts: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    miss_px: float = 220.0,
) -> float:
    """RMS of paired joints, with a miss cost for every blue the chain ignores.

    A pose that lands on two joints and leaves the elbow unexplained must not
    look better than one that follows the whole arm.
    """
    if obs.size == 0:
        return 1.0e6
    paired = {int(i) for i, _j in pairs}
    err = [float(np.linalg.norm(obs[i] - pts[j])) for i, j in pairs]
    n_obs = int(obs.shape[0])
    for i in range(n_obs):
        if i not in paired:
            err.append(float(miss_px))
    if not err:
        return 1.0e6
    acc = 0.0
    for e in err:
        acc += e * e
    return float(math.sqrt(acc / float(len(err))))


def _wrap_pi(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def fit_arm_to_blues(
    observations_mm: np.ndarray,
    q_prior: np.ndarray,
    yaw_prior: float,
    *,
    model: str = "ur5e",
    cfg: UrFkConfig | None = None,
    lock_q: bool = False,
    search: bool = True,
) -> dict:
    """Fit UR joint angles (and base yaw) so the kinematic chain meets blue points.

    Blue points are a measurement of orientation, not the skeleton itself.
    Returns q, yaw, rms_mm, n_pairs, pts_upright_mm, pts_vision_mm.
    """
    cfg = cfg or load_fk_config()
    R = mount_R(cfg)
    t = (
        cfg.t_ur_to_vision_mm
        if cfg.t_ur_to_vision_mm is not None
        else np.zeros(3, dtype=np.float64)
    )
    obs = np.asarray(observations_mm, dtype=np.float64).reshape(-1, 3)
    q_prior = np.asarray(q_prior, dtype=np.float64).reshape(6).copy()
    yaw_prior = _wrap_pi(float(yaw_prior))

    def realize(q: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        up = upright_points_mm(q, yaw, model=model)
        vis = _mm(up, R.T) + t.reshape(3)
        return up, vis

    if obs.shape[0] == 0:
        up, vis = realize(q_prior, yaw_prior)
        return {
            "q": q_prior,
            "yaw": yaw_prior,
            "rms_mm": None,
            "n_pairs": 0,
            "pts_upright_mm": up,
            "pts_vision_mm": vis,
            "check_ok": None,
        }

    wq = 400.0 if lock_q else 18.0

    def residual(q: np.ndarray, yaw: float, pairs: list[tuple[int, int]]) -> np.ndarray:
        _up, vis = realize(q, yaw)
        chunks = [obs[i] - vis[j] for i, j in pairs]
        geom = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
        prior_q = wq * (q - q_prior)
        prior_y = np.array([12.0 * _wrap_pi(yaw - yaw_prior)], dtype=np.float64)
        return np.concatenate([geom, prior_q, prior_y])

    def lm(q0: np.ndarray, y0: float) -> tuple[np.ndarray, float, float, int]:
        q = np.clip(np.asarray(q0, dtype=np.float64).reshape(6), -math.tau, math.tau)
        yaw = _wrap_pi(y0)
        _up, vis = realize(q, yaw)
        pairs = _assign_blues(obs, vis)
        if not pairs:
            return q, yaw, 1.0e6, 0
        r = residual(q, yaw, pairs)
        lam = 1e-2
        for _outer in range(5):
            _up, vis = realize(q, yaw)
            pairs = _assign_blues(obs, vis)
            if not pairs:
                break
            r = residual(q, yaw, pairs)
            for _it in range(12):
                J = np.zeros((r.size, 7), dtype=np.float64)
                eps = 1e-3
                base_ss = float(np.sum(r * r))
                for k in range(7):
                    qq = q.copy()
                    yy = yaw
                    if k < 6:
                        qq[k] += eps
                    else:
                        yy += eps
                    rk = residual(qq, yy, pairs)
                    J[:, k] = (rk - r) / eps
                hess = _mm(J.T, J)
                diag = np.maximum(np.diag(hess), 1e-3)
                for i in range(hess.shape[0]):
                    hess[i, i] += lam * float(diag[i])
                grad = _mm(J.T, r.reshape(-1, 1)).reshape(-1)
                step = _solve(hess, -grad)
                if lock_q:
                    step[:6] = 0.0
                q_new = np.clip(q + step[:6], -math.tau, math.tau)
                yaw_new = _wrap_pi(yaw + float(step[6]))
                r_new = residual(q_new, yaw_new, pairs)
                if float(np.sum(r_new * r_new)) < base_ss:
                    q, yaw, r = q_new, yaw_new, r_new
                    lam = max(lam * 0.4, 1e-6)
                else:
                    lam = min(lam * 5.0, 1e6)
        _up, vis = realize(q, yaw)
        pairs = _assign_blues(obs, vis)
        return q, yaw, _pair_rms(obs, vis, pairs), len(pairs)

    _up0, vis0 = realize(q_prior, yaw_prior)
    pairs0 = _assign_blues(obs, vis0)
    rms_prior = _pair_rms(obs, vis0, pairs0)
    best_q, best_yaw, best_rms, best_n = lm(q_prior, yaw_prior)
    if search and (best_rms > float(cfg.confirm_gate_mm)) and not lock_q:
        seeds_q = [
            np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
            np.zeros(6),
            np.array([0.4, -1.2, 1.4, -1.6, -1.4, 0.3]),
        ]
        seeds_y = [0.0, math.pi / 2, -math.pi / 2, math.pi]
        for q0 in seeds_q:
            for y0 in seeds_y:
                qf, yf, rms, n = lm(q0, y0)
                if rms < best_rms:
                    best_q, best_yaw, best_rms, best_n = qf, yf, rms, n
    # Keep the pendant/file pose when the blue fit does not actually improve it.
    if best_n < 2 or best_rms + 20.0 >= rms_prior:
        best_q, best_yaw, best_rms, best_n = q_prior, yaw_prior, rms_prior, len(pairs0)
    up, vis = realize(best_q, best_yaw)
    gate = float(cfg.confirm_gate_mm)
    check_ok = best_n >= 2 and best_rms <= gate
    return {
        "q": best_q,
        "yaw": best_yaw,
        "rms_mm": None if best_n == 0 else best_rms,
        "n_pairs": best_n,
        "pts_upright_mm": up,
        "pts_vision_mm": vis,
        "check_ok": check_ok,
        "check_unit": "mm",
    }


def _rotate_scale_uv(
    uv: np.ndarray, origin: np.ndarray, deg: float, scale: float
) -> np.ndarray:
    """In-image rotation and scale about the pedestal pixel. Joint 0 stays put."""
    c = math.cos(math.radians(float(deg)))
    s = math.sin(math.radians(float(deg)))
    out = np.asarray(uv, dtype=np.float64).copy()
    ox, oy = float(origin[0]), float(origin[1])
    for i in range(out.shape[0]):
        if not np.isfinite(out[i]).all():
            continue
        dx = float(out[i, 0]) - ox
        dy = float(out[i, 1]) - oy
        out[i, 0] = ox + float(scale) * (c * dx - s * dy)
        out[i, 1] = oy + float(scale) * (s * dx + c * dy)
    return out


def _paired_mean_px(obs: np.ndarray, uv: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 1.0e6
    acc = 0.0
    n = 0
    for i, j in pairs:
        if j == 0 or not np.isfinite(uv[j]).all() or not np.isfinite(obs[i]).all():
            continue
        dx = float(obs[i, 0]) - float(uv[j, 0])
        dy = float(obs[i, 1]) - float(uv[j, 1])
        acc += math.hypot(dx, dy)
        n += 1
    if n == 0:
        return 1.0e6
    return acc / float(n)


def rigid_pixel_align(
    uv: np.ndarray, obs: np.ndarray, origin: np.ndarray
) -> tuple[float, float, float, int]:
    """Small in-image rotation and scale about the pedestal.

    Encoder joint angles are not an input. Returns (deg, scale, mean_px, n_pairs).
    Identity is kept when the similarity does not reduce the error.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1, 2)
    origin = np.asarray(origin, dtype=np.float64).reshape(2)
    if not np.isfinite(origin).all() or obs.shape[0] < 2:
        return 0.0, 1.0, 1.0e6, 0
    pairs = _assign_blues(obs, uv, max_mm=2500.0, origin=origin)
    before = _paired_mean_px(obs, uv, pairs)
    if len(pairs) < 2:
        return 0.0, 1.0, before, len(pairs)
    num_re = 0.0
    num_im = 0.0
    den = 0.0
    for i, j in pairs:
        if not np.isfinite(uv[j]).all():
            continue
        vx = float(uv[j, 0]) - float(origin[0])
        vy = float(uv[j, 1]) - float(origin[1])
        bx = float(obs[i, 0]) - float(origin[0])
        by = float(obs[i, 1]) - float(origin[1])
        num_re += vx * bx + vy * by
        num_im += vx * by - vy * bx
        den += vx * vx + vy * vy
    if den < 1.0 or not math.isfinite(den):
        return 0.0, 1.0, before, len(pairs)
    deg = max(-15.0, min(15.0, math.degrees(math.atan2(num_im, num_re))))
    scale = max(0.85, min(1.15, math.hypot(num_re, num_im) / den))
    moved = _rotate_scale_uv(uv, origin, deg, scale)
    pairs2 = _assign_blues(obs, moved, max_mm=2500.0, origin=origin)
    after = _paired_mean_px(obs, moved, pairs2)
    if not math.isfinite(after) or after + 1.0 >= before:
        return 0.0, 1.0, before, len(pairs)
    return float(deg), float(scale), float(after), len(pairs2)


def fit_arm_to_pixels(
    pixels_uv: np.ndarray,
    q_prior: np.ndarray,
    yaw_prior: float,
    project_fn,
    *,
    model: str = "ur5e",
    cfg: UrFkConfig | None = None,
    lock_q: bool = False,
    search: bool = True,
    base_uv: Optional[np.ndarray] = None,
    prior_w: Optional[np.ndarray] = None,
    yaw_w: float = 4.0,
    image_align: bool = True,
) -> dict:
    """Fit joint angles so the chain projects onto the blue joints in the image.

    ``project_fn(pts_vision_mm) -> (7, 2)`` uses the camera. Invalid joints are NaN.
    When ``base_uv`` is set, every finite joint is shifted so joint 0 lands on
    that pixel before the residual is scored. That is the same translation the
    overlay uses to put BASE on the purple tape, so yaw and q are fit in the
    image the operator sees. Link lengths stay the UR model; blues only check
    orientation.
    """
    cfg = cfg or load_fk_config()
    R = mount_R(cfg)
    t = (
        cfg.t_ur_to_vision_mm
        if cfg.t_ur_to_vision_mm is not None
        else np.zeros(3, dtype=np.float64)
    )
    obs = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    if obs.size:
        obs = obs[np.isfinite(obs).all(axis=1)]
    q_prior = np.asarray(q_prior, dtype=np.float64).reshape(6).copy()
    yaw_prior = _wrap_pi(float(yaw_prior))
    gate_px = 36.0
    anchor = None if base_uv is None else np.asarray(base_uv, dtype=np.float64).reshape(2)
    # Display slides the chain so joint 0 sits on the pedestal. Fit that same
    # image, otherwise blues near the tape never meet the uncalibrated projection
    # and the file pose (an L) is what gets drawn.
    # Off-screen joints of a wrong pose can sit >1000px away; they still have to
    # pull, or a couple of near coincidences look like a finished orientation.
    match_px = 2500.0

    def _anchor_uv(uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(7, 2).copy()
        if anchor is None or not np.isfinite(uv[0]).all():
            return uv
        du = float(anchor[0]) - float(uv[0, 0])
        dv = float(anchor[1]) - float(uv[0, 1])
        finite = np.isfinite(uv).all(axis=1)
        uv[finite, 0] = uv[finite, 0] + du
        uv[finite, 1] = uv[finite, 1] + dv
        return uv

    def realize(q: np.ndarray, yaw: float):
        up = upright_points_mm(q, yaw, model=model)
        vis = _mm(up, R.T) + t.reshape(3)
        uv = _anchor_uv(np.asarray(project_fn(vis), dtype=np.float64).reshape(7, 2))
        return up, vis, uv

    if obs.shape[0] < 2:
        up, vis, _uv = realize(q_prior, yaw_prior)
        return {
            "q": q_prior,
            "yaw": yaw_prior,
            "rms_mm": None,
            "n_pairs": int(obs.shape[0]),
            "pts_upright_mm": up,
            "pts_vision_mm": vis,
            "check_ok": None,
            "check_unit": "px",
            "img_deg": 0.0,
            "img_scale": 1.0,
        }

    if lock_q and not image_align:
        # This camera already sits on the arm (cam1). Score the blues, do not move it.
        up, vis, uv = realize(q_prior, yaw_prior)
        origin = anchor if anchor is not None and np.isfinite(anchor).all() else uv[0]
        pairs = _assign_blues(obs, uv, max_mm=match_px, origin=origin)
        err = _paired_mean_px(obs, uv, pairs)
        n = len(pairs)
        return {
            "q": q_prior.copy(),
            "yaw": float(yaw_prior),
            "rms_mm": None if n < 2 or not math.isfinite(err) else float(err),
            "n_pairs": int(n),
            "pts_upright_mm": up,
            "pts_vision_mm": vis,
            "check_ok": None if n < 2 or not math.isfinite(err) else bool(err <= gate_px),
            "check_unit": "px",
            "img_deg": 0.0,
            "img_scale": 1.0,
        }

    if lock_q:
        # Encoder angles stay the FK half of the residual. Yaw is per camera:
        # spin the whole chain about the pedestal and keep the heading whose
        # projection is closest to the blues. The 1:1 pixel blend does the rest.
        # No in-image scale, so cam1 (image_align off) is not touched.
        def _yaw_pack(yaw: float):
            _up, _vis, uv = realize(q_prior, yaw)
            origin = anchor if anchor is not None else uv[0]
            pairs = _assign_blues(obs, uv, max_mm=match_px, origin=origin)
            err = _paired_mean_px(obs, uv, pairs)
            share = float(W_BLUE) / (float(W_FK) + float(W_BLUE))
            if math.isfinite(err):
                err = err * share
            return _up, _vis, float(yaw), float(err), int(len(pairs))

        base_up, base_vis, _by, berr, bn = _yaw_pack(yaw_prior)
        best = (berr, yaw_prior, bn, base_up, base_vis)
        for step_i in range(-21, 22):
            if step_i == 0:
                continue
            yaw_try = _wrap_pi(yaw_prior + step_i * 0.15)
            up_i, vis_i, yaw_i, err_i, n_i = _yaw_pack(yaw_try)
            if n_i >= 2 and math.isfinite(err_i) and err_i + 1.0 < best[0]:
                best = (err_i, yaw_i, n_i, up_i, vis_i)
        err, yaw_b, n_b, up_b, vis_b = best
        if not math.isfinite(err) or n_b < 2 or err + 1.0 >= berr:
            err, yaw_b, n_b = berr, yaw_prior, bn
            up_b, vis_b = base_up, base_vis
        return {
            "q": q_prior.copy(),
            "yaw": float(yaw_b),
            "rms_mm": None if n_b < 2 else float(err),
            "n_pairs": int(n_b),
            "pts_upright_mm": up_b,
            "pts_vision_mm": vis_b,
            "check_ok": None if n_b < 2 else bool(err <= gate_px),
            "check_unit": "px",
            "img_deg": 0.0,
            "img_scale": 1.0,
        }

    def residual(q, yaw, pairs):
        _up, _vis, uv = realize(q, yaw)
        chunks = []
        # The chain root is the robot base, not whichever blue blob is closest.
        if anchor is not None:
            if np.isfinite(uv[0]).all():
                chunks.append((uv[0] - anchor) * 6.0)
            else:
                chunks.append(np.array([400.0, 400.0], dtype=np.float64))
        for i, j in pairs:
            if j == 0 or not np.isfinite(uv[j]).all():
                continue
            chunks.append(obs[i] - uv[j])
        geom = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
        if prior_w is None:
            wq_vec = np.full(6, 80.0 if lock_q else 6.0, dtype=np.float64)
        else:
            wq_vec = np.asarray(prior_w, dtype=np.float64).reshape(6)
        prior_q = wq_vec * (q - q_prior)
        prior_y = np.array([float(yaw_w) * _wrap_pi(yaw - yaw_prior)], dtype=np.float64)
        return np.concatenate([geom, prior_q, prior_y])

    def lm(q0, y0):
        q = np.clip(np.asarray(q0, dtype=np.float64).reshape(6), -math.tau, math.tau)
        yaw = _wrap_pi(y0)
        _up, _vis, uv = realize(q, yaw)
        finite = np.isfinite(uv).all(axis=1)
        pts = np.where(finite[:, None], uv, 1.0e6)
        pairs = _assign_blues(obs, pts, max_mm=match_px, origin=anchor)
        if not pairs and anchor is None:
            return q, yaw, 1.0e6, 0
        r = residual(q, yaw, pairs)
        lam = 1e-2
        for _outer in range(4):
            _up, _vis, uv = realize(q, yaw)
            finite = np.isfinite(uv).all(axis=1)
            pts = np.where(finite[:, None], uv, 1.0e6)
            pairs = _assign_blues(obs, pts, max_mm=match_px, origin=anchor)
            if not pairs:
                break
            r = residual(q, yaw, pairs)
            for _it in range(8):
                J = np.zeros((r.size, 7), dtype=np.float64)
                eps = 1e-3
                base_ss = float(np.sum(r * r))
                for k in range(7):
                    qq = q.copy()
                    yy = yaw
                    if k < 6:
                        qq[k] += eps
                    else:
                        yy += eps
                    rk = residual(qq, yy, pairs)
                    if rk.size != r.size:
                        rk = np.resize(rk, r.size)
                    J[:, k] = (rk - r) / eps
                hess = _mm(J.T, J)
                diag = np.maximum(np.diag(hess), 1e-3)
                for i in range(hess.shape[0]):
                    hess[i, i] += lam * float(diag[i])
                grad = _mm(J.T, r.reshape(-1, 1)).reshape(-1)
                step = _solve(hess, -grad)
                if lock_q:
                    step[:6] = 0.0
                q_new = np.clip(q + step[:6], -math.tau, math.tau)
                yaw_new = _wrap_pi(yaw + float(step[6]))
                r_new = residual(q_new, yaw_new, pairs)
                if r_new.size == r.size and float(np.sum(r_new * r_new)) < base_ss:
                    q, yaw, r = q_new, yaw_new, r_new
                    lam = max(lam * 0.4, 1e-6)
                else:
                    lam = min(lam * 5.0, 1e6)
        _up, _vis, uv = realize(q, yaw)
        finite = np.isfinite(uv).all(axis=1)
        pts = np.where(finite[:, None], uv, 1.0e6)
        pairs = _assign_blues(obs, pts, max_mm=match_px, origin=anchor)
        return q, yaw, _coverage_rms(obs, pts, pairs), len(pairs)

    def pose_score(q: np.ndarray, yaw: float) -> tuple[float, int]:
        _up, _vis, uv = realize(q, yaw)
        finite = np.isfinite(uv).all(axis=1)
        pts = np.where(finite[:, None], uv, 1.0e6)
        pairs = _assign_blues(obs, pts, max_mm=match_px, origin=anchor)
        return _coverage_rms(obs, pts, pairs), len(pairs)

    _up, _vis, uv0 = realize(q_prior, yaw_prior)
    finite0 = np.isfinite(uv0).all(axis=1)
    pts0 = np.where(finite0[:, None], uv0, 1.0e6)
    pairs0 = _assign_blues(obs, pts0, max_mm=match_px, origin=anchor)
    rms_prior = _coverage_rms(obs, pts0, pairs0)
    best_q, best_yaw, best_rms, best_n = lm(q_prior, yaw_prior)
    covered = best_n * 2 >= int(obs.shape[0]) and best_n >= 2
    if search and (best_rms > gate_px or not covered) and not lock_q:
        # q1 spins the arm about vertical (same axis as yaw). q2 leans the
        # upper arm, q3 folds the forearm. The example file pose is an L;
        # a coarse net finds the image triangle before the local step.
        wrist = q_prior[3:].copy()
        ranked: list[tuple[float, np.ndarray]] = []
        for q1 in (0.0, 0.9, -0.9, 1.8, -1.8, 2.7, -2.7):
            for q2 in (-0.7, -1.2, -1.57, -2.1, -2.6):
                for q3 in (0.9, 1.57, 2.2, 2.8):
                    q_try = np.array(
                        [q1, q2, q3, float(wrist[0]), float(wrist[1]), float(wrist[2])],
                        dtype=np.float64,
                    )
                    rms_try, _n_try = pose_score(q_try, yaw_prior)
                    ranked.append((rms_try, q_try))
        ranked.sort(key=lambda item: item[0])
        tried = 0
        for rms_try, q_try in ranked:
            if tried >= 3:
                break
            if rms_try > best_rms and tried > 0:
                break
            tried += 1
            qf, yf, rms, n = lm(q_try, yaw_prior)
            if rms < best_rms:
                best_q, best_yaw, best_rms, best_n = qf, yf, rms, n
    if best_n < 2 or best_rms + 4.0 >= rms_prior:
        best_q, best_yaw, best_rms, best_n = q_prior, yaw_prior, rms_prior, len(pairs0)
    up, vis, uv = realize(best_q, best_yaw)
    origin = anchor if anchor is not None else uv[0]
    deg, scale, err, n_img = rigid_pixel_align(uv, obs, origin)
    if n_img >= 2 and math.isfinite(err):
        best_rms = float(err)
        best_n = int(n_img)
    covered = best_n * 2 >= int(obs.shape[0]) and best_n >= 2
    return {
        "q": best_q,
        "yaw": best_yaw,
        "rms_mm": None if best_n == 0 else float(best_rms),
        "n_pairs": best_n,
        "pts_upright_mm": up,
        "pts_vision_mm": vis,
        "check_ok": covered and best_rms <= gate_px,
        "check_unit": "px",
        "img_deg": float(deg),
        "img_scale": float(scale),
    }


def _read_joints_file(path: Path) -> Optional[tuple[np.ndarray, str]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    q = data.get("q_rad") or data.get("q")
    if q is None or len(q) < 6:
        return None
    model = str(data.get("model", "ur5e"))
    return np.asarray(q[:6], dtype=np.float64), model


# UR real-time interface (port 30003). Message size is a big-endian int, then
# doubles. q_actual is six doubles at byte 252 (after time and the five
# target blocks). Same angles RTDEReceiveInterface.getActualQ returns.
_Q_ACTUAL_OFFSET = 252


def q_from_realtime_packet(packet: bytes) -> Optional[np.ndarray]:
    """Six joint angles (rad) from one UR 30003 state packet, or None."""
    if len(packet) < _Q_ACTUAL_OFFSET + 48:
        return None
    q = struct.unpack(">6d", packet[_Q_ACTUAL_OFFSET : _Q_ACTUAL_OFFSET + 48])
    if not all(math.isfinite(v) and abs(v) < 8.0 for v in q):
        return None
    return np.asarray(q, dtype=np.float64)


class TeachPadCache:
    """Background reader. The viewer only copies the last q."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.q: Optional[np.ndarray] = None
        self.mode = ""
        self.host = ""
        self._started = False
        self._stop = False

    def note(self, q: np.ndarray, *, live: bool, host: str) -> None:
        with self._lock:
            self.q = np.asarray(q, dtype=np.float64).reshape(6).copy()
            self.mode = "live" if live else "hold"
            self.host = host
            self._started = True

    def mark_hold(self) -> None:
        with self._lock:
            if self.q is not None:
                self.mode = "hold"

    def snapshot(self, host: str = "") -> Optional[tuple[np.ndarray, str]]:
        with self._lock:
            if self.q is None:
                return None
            if host and self.host and self.host != host:
                return None
            tag = "rtde-live" if self.mode == "live" else "rtde-hold"
            return self.q.copy(), f"{tag}:{self.host or host}"

    def ensure(self, host: str, port: int = 30003) -> None:
        with self._lock:
            if self._started and self.host == host:
                return
            self.host = host
            self._started = True
            self._stop = False
        threading.Thread(
            target=self._run, args=(host, int(port)), daemon=True
        ).start()

    def _run(self, host: str, port: int) -> None:
        while not self._stop:
            sock = None
            try:
                sock = socket.create_connection((host, port), 1.0)
                sock.settimeout(0.5)
                buf = b""
                while not self._stop:
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("teach pad closed")
                    buf += chunk
                    if len(buf) > 20000:
                        buf = buf[-4000:]
                    while len(buf) >= 4:
                        n = struct.unpack(">i", buf[:4])[0]
                        if n < 300 or n > 4000:
                            buf = buf[1:]
                            continue
                        if len(buf) < n:
                            break
                        q = q_from_realtime_packet(buf[:n])
                        buf = buf[n:]
                        if q is None:
                            continue
                        with self._lock:
                            self.q = q
                            self.mode = "live"
                            self.host = host
            except Exception:
                self.mark_hold()
                time.sleep(0.4)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass


_TEACH = TeachPadCache()



def load_joints(
    cfg: UrFkConfig | None = None,
) -> tuple[np.ndarray, str, str]:
    """Return (q_rad, model, source_name)."""
    cfg = cfg or load_fk_config()
    if cfg.host:
        snap = _TEACH.snapshot(cfg.host)
        if snap is None:
            _TEACH.ensure(cfg.host, cfg.port)
            snap = _TEACH.snapshot(cfg.host)
        if snap is not None:
            return snap[0], cfg.model, snap[1]
        # Stream not up yet. Do not draw the file L. Photographed pose until
        # the first actual-q packet, then the cache holds the last encoders.
        return PHOTO_Q.copy(), cfg.model, "photo:offline"
    got = _read_joints_file(cfg.joints_path)
    if got is not None:
        return got[0], str(got[1] or cfg.model), f"file:{cfg.joints_path.name}"
    # Mock “elbow up” pose so the stick figure is visible offline.
    q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)
    return q, cfg.model, "mock"


def point_to_aabb_gap_mm(
    p_mm: np.ndarray,
    mn: tuple[float, float, float],
    mx: tuple[float, float, float],
) -> float:
    """0 if inside AABB; else Euclidean distance to box surface (mm)."""
    p = np.asarray(p_mm, dtype=np.float64).reshape(3)
    mn_a = np.asarray(mn, dtype=np.float64)
    mx_a = np.asarray(mx, dtype=np.float64)
    closest = np.minimum(np.maximum(p, mn_a), mx_a)
    if np.all(p >= mn_a) and np.all(p <= mx_a):
        return 0.0
    return float(np.linalg.norm(p - closest))


def fk_rack_proximity(
    pts_vision_mm: np.ndarray,
    cfg: UrFkConfig,
) -> list[tuple[int, float]]:
    """Return list of (joint_index, gap_mm) within cfg.proximity_mm of rack keep-out.

    joint_index: 0=base ... 6=tool.
    """
    hits: list[tuple[int, float]] = []
    for i, p in enumerate(np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 3)):
        if i == 0:
            continue  # skip base origin
        gap = point_to_aabb_gap_mm(p, cfg.rack_keepout_min_mm, cfg.rack_keepout_max_mm)
        if gap <= float(cfg.proximity_mm):
            hits.append((i, gap))
    return hits
