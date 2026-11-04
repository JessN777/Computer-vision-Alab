"""Draw UR FK stick figure, confirm vs vision blues, FK rack keep-out alerts.

Called from ``dual_view_node._overlay_ur_fk``. When ``calibrated=True`` (online
pose ready), joints are projected with ``cv2.projectPoints(K, dist)`` and the
legacy tag-fit / img_scale seating stages are skipped.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Calibrated draw: a projected joint within this many pixels of its tag is "ok".
CALIB_TAG_OK_PX = 25.0


def _ensure_robotics(robotics: Optional[Path] = None) -> Optional[Path]:
    if robotics is not None and (robotics / "ur_skeleton.py").is_file():
        if str(robotics) not in sys.path:
            sys.path.insert(0, str(robotics))
        return robotics
    import os

    env = os.environ.get("ORBBEC_CAMERA_ROOT", "").strip()
    for root in (
        Path(env) if env else None,
        Path(__file__).resolve().parents[4] / "robotics",
        Path(__file__).resolve().parents[3] / "robotics",
    ):
        if root is not None and (root / "ur_skeleton.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    return None


def base_to_cam(p_base_mm: np.ndarray, pose) -> np.ndarray:
    """Inverse of cell_frames.cam_to_base: p_cam = R.T @ (p_base - t)."""
    p = np.asarray(p_base_mm, dtype=np.float64).reshape(3)
    R = np.asarray(pose.R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(pose.t_mm, dtype=np.float64).reshape(3)
    d = p - t
    out = np.zeros(3, dtype=np.float64)
    # p_cam = R.T @ (p_base - t), written out so a broken BLAS gemm cannot kill the overlay.
    for i in range(3):
        out[i] = (
            float(R[0, i]) * float(d[0])
            + float(R[1, i]) * float(d[1])
            + float(R[2, i]) * float(d[2])
        )
    return out


def project_cam_to_pixel(
    p_cam_mm: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> Optional[tuple[int, int]]:
    x, y, z = [float(v) for v in np.asarray(p_cam_mm, dtype=np.float64).reshape(3)]
    if z <= 1e-3:
        return None
    u = int(round(fx * x / z + cx))
    v = int(round(fy * y / z + cy))
    return u, v


def _project_points_calibrated(pts, pose, K, dist):
    """Same model as OnlineCamPose: cv2.projectPoints(K, dist). No ``@``."""
    from .online_cam_pose import _cv_from_cell

    R = [[float(pose.R[i, j]) for j in range(3)] for i in range(3)]
    t = (float(pose.t_mm[0]), float(pose.t_mm[1]), float(pose.t_mm[2]))
    rvec, tvec = _cv_from_cell(R, t)
    obj = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 3)
    Km = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if dist is None:
        d = np.zeros((5, 1), dtype=np.float64)
    else:
        d = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, Km, d)
    flat = proj.reshape(-1, 2)
    pix: list[Optional[tuple[int, int]]] = []
    for i in range(flat.shape[0]):
        u, v = float(flat[i, 0]), float(flat[i, 1])
        if math.isfinite(u) and math.isfinite(v):
            pix.append((int(round(u)), int(round(v))))
        else:
            pix.append(None)
    return pix


def _clip_line_to_rect(
    x0: float, y0: float, x1: float, y1: float, w: int, h: int
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Cohen–Sutherland clip. Returns pixel endpoints inside the frame."""
    left, right, bottom, top = 1, 2, 4, 8

    def code(x: float, y: float) -> int:
        c = 0
        if x < 0:
            c |= left
        elif x >= w:
            c |= right
        if y < 0:
            c |= top
        elif y >= h:
            c |= bottom
        return c

    c0, c1 = code(x0, y0), code(x1, y1)
    for _ in range(12):
        if c0 == 0 and c1 == 0:
            return (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))
        if c0 & c1:
            return None
        c = c0 or c1
        if c & bottom:
            y = h - 1
            x = x0 + (x1 - x0) * (y - y0) / (y1 - y0 if y1 != y0 else 1e-6)
        elif c & top:
            y = 0
            x = x0 + (x1 - x0) * (y - y0) / (y1 - y0 if y1 != y0 else 1e-6)
        elif c & right:
            x = w - 1
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0 if x1 != x0 else 1e-6)
        else:
            x = 0
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0 if x1 != x0 else 1e-6)
        if c == c0:
            x0, y0, c0 = x, y, code(x, y)
        else:
            x1, y1, c1 = x, y, code(x, y)
    return None


def draw_ur_monitor_inset(
    bgr: np.ndarray,
    pts_vision_mm: np.ndarray,
    *,
    rack_hits: Optional[list[tuple[int, float]]] = None,
    source: str = "",
    model: str = "",
    cam_pix: Optional[list] = None,
) -> np.ndarray:
    """Corner stick figure in the camera view, with true angles (uniform scale)."""
    out = bgr
    h, w = out.shape[:2]
    bw, bh = min(210, max(120, w // 4)), min(168, max(110, h // 4))
    x0 = max(0, w - bw - 8)
    y0 = 8
    y1, x1 = min(h, y0 + bh), min(w, x0 + bw)
    if y1 - y0 < 40 or x1 - x0 < 40:
        return out
    roi = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = (roi.astype(np.float32) * 0.40).astype(np.uint8)
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (80, 220, 80), 1, cv2.LINE_AA)
    pad_x, pad_top, pad_bot = 16, 22, 14
    avail_w = max(1.0, float((x1 - x0) - 2 * pad_x))
    avail_h = max(1.0, float((y1 - y0) - pad_top - pad_bot))

    use_cam = False
    xy = None
    if cam_pix is not None:
        rows = []
        for uv in cam_pix:
            if uv is None:
                rows.append([np.nan, np.nan])
            else:
                rows.append([float(uv[0]), float(uv[1])])
        cand = np.asarray(rows, dtype=np.float64)
        if np.isfinite(cand).all(axis=1).sum() >= 3:
            xy = cand
            use_cam = True
    if xy is None:
        pts = np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 3)
        # Side view of whichever horizontal axis the arm actually extends along.
        span_x = float(pts[:, 0].max() - pts[:, 0].min())
        span_y = float(pts[:, 1].max() - pts[:, 1].min())
        horiz = 0 if span_x >= span_y else 1
        xy = pts[:, [horiz, 2]].copy()
        xy[:, 1] = -xy[:, 1]  # Z up → image y down

    finite = np.isfinite(xy).all(axis=1)
    if use_cam:
        # Joints that project far above the frame must not shrink the visible
        # chain. Scale the inset from pixels that are actually on this image.
        inside = (
            finite
            & (xy[:, 0] >= 0.0)
            & (xy[:, 0] < float(w))
            & (xy[:, 1] >= 0.0)
            & (xy[:, 1] < float(h))
        )
        if int(inside.sum()) >= 2:
            finite = inside
    if finite.sum() < 2:
        return out
    mn = xy[finite].min(axis=0)
    mx = xy[finite].max(axis=0)
    span = np.maximum(mx - mn, 1.0)
    scale = min(avail_w / float(span[0]), avail_h / float(span[1]))
    drawn_w = float(span[0]) * scale
    drawn_h = float(span[1]) * scale
    origin_u = x0 + pad_x + 0.5 * (avail_w - drawn_w)
    origin_v = y0 + pad_top + 0.5 * (avail_h - drawn_h)

    def to_px(p: np.ndarray) -> Optional[tuple[int, int]]:
        if not np.isfinite(p).all():
            return None
        u = origin_u + (float(p[0]) - mn[0]) * scale
        v = origin_v + (float(p[1]) - mn[1]) * scale
        return int(round(u)), int(round(v))

    pix = [to_px(p) for p in xy]
    hit_idx = {i for i, _ in (rack_hits or [])}
    for i in range(len(pix) - 1):
        if pix[i] is None or pix[i + 1] is None:
            continue
        color = (0, 0, 255) if (i + 1) in hit_idx else (80, 255, 120)
        cv2.line(out, pix[i], pix[i + 1], color, 2, cv2.LINE_AA)
    for i, uv in enumerate(pix):
        if uv is None:
            continue
        color = (0, 0, 255) if i in hit_idx else (0, 220, 255)
        cv2.circle(out, uv, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            str(i) if i else "B",
            (uv[0] + 4, uv[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    label = _ur7e_name(model)
    label += " cam" if use_cam else " side"
    if source:
        bit = source.split(":")[0].split()[0]
        if bit not in ("file", "mock"):
            label += f" {bit}"
    cv2.putText(
        out,
        label,
        (x0 + 6, y0 + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (80, 255, 120),
        1,
        cv2.LINE_AA,
    )
    return out


def load_ur_state(robotics: Optional[Path] = None):
    root = _ensure_robotics(robotics)
    if root is None:
        return None
    from ur_skeleton import (  # noqa: WPS433
        fk_joint_positions_mm,
        fk_rack_proximity,
        load_fk_config,
        load_joints,
        ur_base_to_vision_mm,
        write_example_files,
    )

    write_example_files()
    cfg = load_fk_config()
    q, model, source = load_joints(cfg)
    pts_ur = fk_joint_positions_mm(q, model=model)
    pts_vis = ur_base_to_vision_mm(pts_ur, cfg)
    return {
        "cfg": cfg,
        "q": q,
        "model": model,
        "source": source,
        "pts_vision_mm": pts_vis,
        "rack_hits": fk_rack_proximity(pts_vis, cfg),
    }


def confirm_vision_vs_fk(
    views: list,
    pts_vision_mm: np.ndarray,
    *,
    gate_mm: float = 80.0,
) -> list[tuple[int, int, float]]:
    """Match vision TrackView (with p_base) to nearest FK joint.

    Returns list of (track_id, fk_index, dist_mm) for matches within gate.
    """
    matches: list[tuple[int, int, float]] = []
    pts = np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 3)
    for v in views:
        pb = getattr(v, "p_base", None)
        if pb is None:
            continue
        p = np.asarray(pb, dtype=np.float64).reshape(3)
        # Skip base FK index 0 for moving joints; allow for is_base tracks.
        start = 0 if getattr(v, "is_base", False) else 1
        best_i, best_d = -1, 1e9
        for i in range(start, pts.shape[0]):
            d = float(np.linalg.norm(p - pts[i]))
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= float(gate_mm):
            matches.append((int(v.track_id), best_i, best_d))
    return matches


def align_state_to_blues(
    state: dict,
    observations_mm: np.ndarray,
    *,
    q_prev: Optional[np.ndarray] = None,
    yaw_prev: float = 0.0,
    search: bool = True,
    pixels_uv: Optional[np.ndarray] = None,
    project_fn=None,
    base_uv: Optional[np.ndarray] = None,
    image_align: bool = True,
    with_tool: bool = False,
) -> dict:
    """Place the stick figure with UR kinematics; blue points only check orientation."""
    from ur_skeleton import (  # noqa: WPS433
        append_tool_points,
        fit_arm_to_blues,
        fit_arm_to_pixels,
        fk_rack_proximity,
        is_example_l,
        seed_photo_if_example,
    )

    src = str(state.get("source", ""))
    locked = src.startswith("rtde")
    q_prior = state["q"] if q_prev is None else q_prev
    if src.startswith("photo"):
        from ur_skeleton import PHOTO_Q, PHOTO_YAW

        q_prior = PHOTO_Q.copy()
        if abs(float(yaw_prev)) < 1e-6:
            yaw_prev = float(PHOTO_YAW)
        seeded = False
    else:
        # File L only. A live/held teach-pad q is locked and is not replaced.
        seeded = (not locked) and is_example_l(np.asarray(q_prior, dtype=np.float64))
        q_prior, yaw_prev = seed_photo_if_example(q_prior, yaw_prev, locked=locked)
    prior_w = None
    yaw_w = 4.0
    if seeded:
        prior_w = np.array([40.0, 80.0, 80.0, 4.0, 4.0, 4.0], dtype=np.float64)
        yaw_w = 18.0
        search = False
    use_pixels = (
        project_fn is not None
        and pixels_uv is not None
        and np.asarray(pixels_uv).reshape(-1, 2).shape[0] >= 2
    )
    if use_pixels:
        fit = fit_arm_to_pixels(
            pixels_uv,
            q_prior,
            yaw_prev,
            project_fn,
            model=str(state.get("model") or "ur5e"),
            cfg=state["cfg"],
            lock_q=locked,
            search=search and not locked,
            base_uv=base_uv,
            prior_w=prior_w,
            yaw_w=yaw_w,
            image_align=image_align,
        )
    else:
        fit = fit_arm_to_blues(
            observations_mm,
            q_prior,
            yaw_prev,
            model=str(state.get("model") or "ur5e"),
            cfg=state["cfg"],
            lock_q=locked,
            search=search and not locked,
        )
    out = dict(state)
    out["q"] = fit["q"]
    out["yaw"] = fit["yaw"]
    pts_vis = fit["pts_vision_mm"]
    pts_up = fit["pts_upright_mm"]
    if with_tool:
        pts_up, pts_vis = append_tool_points(
            pts_up,
            pts_vis,
            fit["q"],
            fit["yaw"],
            state["cfg"],
            model=str(state.get("model") or "ur5e"),
        )
    out["pts_vision_mm"] = pts_vis
    out["pts_upright_mm"] = pts_up
    out["has_tool"] = bool(with_tool)
    out["check_ok"] = fit["check_ok"]
    out["check_mm"] = fit["rms_mm"]
    out["check_unit"] = fit.get("check_unit", "mm")
    out["n_pairs"] = fit["n_pairs"]
    out["img_deg"] = float(fit.get("img_deg") or 0.0)
    out["img_scale"] = float(fit.get("img_scale") or 1.0)
    if pixels_uv is not None:
        out["blue_uv"] = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2).copy()
    out["rack_hits"] = fk_rack_proximity(fit["pts_vision_mm"], state["cfg"])
    if fit["check_ok"] is True:
        out["source"] = f"{src} blue-ok"
    elif fit["check_ok"] is False:
        out["source"] = f"{src} blue-off"
    return out


def project_ur7e_from_tags(
    pts_mm: np.ndarray,
    tag_uv,
    fx: float,
    width: int,
    height: int,
):
    """Place the UR7e chain in the image from tag centers and the CAD joint lengths.

    Object points are the kinematic joints (mm). Image points are ArUco ids 0..5.
    Returns one pixel per joint when three or more tags agree within 35 px.
    The base correspondence is tag 0, on the base joint, not the taped preset.
    """
    pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 3)
    if tag_uv is None or float(fx) <= 1.0 or width <= 0 or height <= 0:
        return None
    obj: list[list[float]] = []
    img: list[list[float]] = []
    idxs: list[int] = []
    n = min(len(tag_uv), int(pts.shape[0]))
    for i in range(n):
        uv = tag_uv[i]
        if uv is None:
            continue
        u, v = float(uv[0]), float(uv[1])
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        p = (float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
        if not all(math.isfinite(c) for c in p):
            continue
        obj.append([p[0], p[1], p[2]])
        img.append([u, v])
        idxs.append(i)
    if len(obj) < 3:
        return None
    # ITERATIVE's DLT wants 6 correspondences. Three tags use SQPnP, four or five use EPnP.
    if len(obj) >= 6:
        flag = cv2.SOLVEPNP_ITERATIVE
    elif len(obj) == 3:
        flag = getattr(cv2, "SOLVEPNP_SQPNP", getattr(cv2, "SOLVEPNP_EPNP", cv2.SOLVEPNP_ITERATIVE))
    else:
        flag = getattr(cv2, "SOLVEPNP_EPNP", cv2.SOLVEPNP_ITERATIVE)
    k = np.array(
        [
            [float(fx), 0.0, 0.5 * float(width)],
            [0.0, float(fx), 0.5 * float(height)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        ok, rvec, tvec = cv2.solvePnP(
            np.asarray(obj, dtype=np.float64),
            np.asarray(img, dtype=np.float64),
            k,
            None,
            flags=flag,
        )
    except Exception:
        return None
    if not ok:
        return None
    try:
        rot, _ = cv2.Rodrigues(rvec)
        tz = tvec.reshape(3)
    except Exception:
        return None
    for i in range(int(pts.shape[0])):
        z_cam = (
            float(rot[2, 0]) * float(pts[i, 0])
            + float(rot[2, 1]) * float(pts[i, 1])
            + float(rot[2, 2]) * float(pts[i, 2])
            + float(tz[2])
        )
        if z_cam < 80.0:
            return None
    try:
        proj, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, k, None)
    except Exception:
        return None
    proj = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
    err = 0.0
    for k_i, i in enumerate(idxs):
        du = float(proj[i, 0]) - img[k_i][0]
        dv = float(proj[i, 1]) - img[k_i][1]
        err += math.sqrt(du * du + dv * dv)
    if err / float(len(idxs)) > 35.0:
        return None
    out_pix: list[Optional[tuple[int, int]]] = []
    for i in range(int(proj.shape[0])):
        u, v = float(proj[i, 0]), float(proj[i, 1])
        if not (math.isfinite(u) and math.isfinite(v)):
            out_pix.append(None)
        else:
            out_pix.append((int(round(u)), int(round(v))))
    return out_pix


def align_chain_about_base_tag(pix, tag_uv):
    """Turn the projected STEP chain about the base tag so the other tags fit.

    Tag 0 is the base joint. One other tag fixes a translation onto that tag.
    Two or more other tags are not required: one other tag sets the in-plane
    rotation and scale of the STEP chain. Returns ``(pixels, kind)`` or
    ``(None, None)``. ``kind`` is ``similarity`` or ``translate``.
    """
    if not tag_uv or not pix or pix[0] is None or tag_uv[0] is None:
        return None, None
    bu, bv = float(tag_uv[0][0]), float(tag_uv[0][1])
    if not (math.isfinite(bu) and math.isfinite(bv)):
        return None, None
    acc = 0.0
    a_num = 0.0
    b_num = 0.0
    n_obs = 0
    n = min(len(pix), len(tag_uv))
    for i in range(1, n):
        if pix[i] is None or tag_uv[i] is None:
            continue
        mx = float(pix[i][0]) - float(pix[0][0])
        my = float(pix[i][1]) - float(pix[0][1])
        ox = float(tag_uv[i][0]) - bu
        oy = float(tag_uv[i][1]) - bv
        if not all(math.isfinite(v) for v in (mx, my, ox, oy)):
            continue
        acc += mx * mx + my * my
        a_num += mx * ox + my * oy
        b_num += mx * oy - my * ox
        n_obs += 1
    kind = "translate"
    a, b = 1.0, 0.0
    if n_obs >= 1 and acc >= 25.0:
        a = a_num / acc
        b = b_num / acc
        scale = math.sqrt(a * a + b * b)
        if scale < 0.25 or scale > 4.0:
            if scale > 1e-6:
                a /= scale
                b /= scale
        kind = "similarity"
    out: list[Optional[tuple[int, int]]] = []
    ox0, oy0 = float(pix[0][0]), float(pix[0][1])
    for uv in pix:
        if uv is None:
            out.append(None)
            continue
        dx = float(uv[0]) - ox0
        dy = float(uv[1]) - oy0
        u = bu + a * dx - b * dy
        v = bv + b * dx + a * dy
        if not (math.isfinite(u) and math.isfinite(v)):
            out.append(None)
        else:
            out.append((int(round(u)), int(round(v))))
    return out, kind


# Last accepted twist about the shoulder axis, per camera. A one-frame
# tag blip must not spin the arm to the other pose.
_ARM_HOLD: dict[str, tuple[float, float, list]] = {}


def _tag_map(tag_uv, n_pts: int) -> dict[int, tuple[float, float]]:
    tags: dict[int, tuple[float, float]] = {}
    if not tag_uv:
        return tags
    n = min(len(tag_uv), n_pts)
    for i in range(n):
        uv = tag_uv[i]
        if uv is None:
            continue
        u, v = float(uv[0]), float(uv[1])
        if math.isfinite(u) and math.isfinite(v):
            tags[i] = (u, v)
    return tags


def place_validation_joints_sized(
    pts_mm, tag_uv, fx, width, height, hint_pix=None, *, cam: str = "", blue_uv=None
):
    """Validation-script joint coordinates, with the shoulder tag fixed.

    ``pts_mm`` is ``fk_joint_positions_mm`` (the same points as ``ur_fk``).
    Joint 1 is the shoulder, 162.5 mm up the base axis. When that tag is
    seen with the base tag, the drawn shoulder is that coordinate. The twist
    about that axis stays on the last good pose until another tag clearly
    prefers a different one.
    """
    if tag_uv is None or pts_mm is None:
        return None
    pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return None
    tags = _tag_map(tag_uv, int(pts.shape[0]))
    # Base + shoulder own the chain. A third tag is only a vote for the
    # twist. solvePnP on a one-frame extra tag was flipping the whole arm.
    if 0 in tags and 1 in tags:
        return _shoulder_frame(pts, tags, hint_pix, cam=cam, blue_uv=blue_uv)
    if len(tags) >= 3:
        return project_ur7e_from_tags(pts, tag_uv, fx, width, height)
    return None


def _follow_held_chain(held: list, tag_uv) -> list:
    """Slide the last good chain onto the base or shoulder tag still in view."""
    src = None
    dst = None
    if tag_uv and held and held[0] is not None and tag_uv[0] is not None:
        src, dst = held[0], tag_uv[0]
    elif (
        tag_uv
        and held
        and len(held) > 1
        and len(tag_uv) > 1
        and held[1] is not None
        and tag_uv[1] is not None
    ):
        src, dst = held[1], tag_uv[1]
    if src is None or dst is None:
        return list(held)
    du = float(dst[0]) - float(src[0])
    dv = float(dst[1]) - float(src[1])
    out = []
    for uv in held:
        if uv is None:
            out.append(None)
        else:
            out.append((int(round(float(uv[0]) + du)), int(round(float(uv[1]) + dv))))
    return out


def _shoulder_frame(pts, tags, hint_pix, *, cam: str = "", blue_uv=None):
    """Image frame whose +Z runs from the base tag to the shoulder tag."""
    b = tags[0]
    s = tags[1]
    dx = float(pts[1, 0]) - float(pts[0, 0])
    dy = float(pts[1, 1]) - float(pts[0, 1])
    dz = float(pts[1, 2]) - float(pts[0, 2])
    reach = math.sqrt(dx * dx + dy * dy + dz * dz)
    if reach < 1.0:
        return None
    ezx = (s[0] - b[0]) / reach
    ezy = (s[1] - b[1]) / reach
    elen = math.sqrt(ezx * ezx + ezy * ezy)
    if elen < 1e-6:
        return None
    px = -ezy / elen
    py = ezx / elen

    def project(theta: float):
        # Orthographic view of the validation chain. Sideways motion stays
        # perpendicular to the shoulder axis, so a link cannot stretch or
        # fold through the elbow the way a sheared frame can.
        c, sn = math.cos(theta), math.sin(theta)
        out = []
        for i in range(int(pts.shape[0])):
            x = float(pts[i, 0]) - float(pts[0, 0])
            y = float(pts[i, 1]) - float(pts[0, 1])
            z = float(pts[i, 2]) - float(pts[0, 2])
            side = c * x + sn * y
            u = b[0] + z * ezx + side * px * elen
            v = b[1] + z * ezy + side * py * elen
            out.append((u, v))
        return out

    def contorted(proj) -> bool:
        """True when the drawing bends a joint the validation chain does not."""
        n_pts = int(pts.shape[0])
        for i in range(n_pts - 1):
            dmm = math.sqrt(
                (float(pts[i + 1, 0]) - float(pts[i, 0])) ** 2
                + (float(pts[i + 1, 1]) - float(pts[i, 1])) ** 2
                + (float(pts[i + 1, 2]) - float(pts[i, 2])) ** 2
            )
            dpx = math.hypot(proj[i + 1][0] - proj[i][0], proj[i + 1][1] - proj[i][1])
            if dmm > 20.0 and dpx > elen * dmm * 1.2:
                return True
        for i in range(1, n_pts - 1):
            ax = float(pts[i, 0]) - float(pts[i - 1, 0])
            ay = float(pts[i, 1]) - float(pts[i - 1, 1])
            az = float(pts[i, 2]) - float(pts[i - 1, 2])
            bx = float(pts[i + 1, 0]) - float(pts[i, 0])
            by = float(pts[i + 1, 1]) - float(pts[i, 1])
            bz = float(pts[i + 1, 2]) - float(pts[i, 2])
            na = math.sqrt(ax * ax + ay * ay + az * az)
            nb = math.sqrt(bx * bx + by * by + bz * bz)
            if na < 20.0 or nb < 20.0:
                continue
            d3 = (ax * bx + ay * by + az * bz) / (na * nb)
            ux = proj[i][0] - proj[i - 1][0]
            uy = proj[i][1] - proj[i - 1][1]
            vx = proj[i + 1][0] - proj[i][0]
            vy = proj[i + 1][1] - proj[i][1]
            nua = math.hypot(ux, uy)
            nvb = math.hypot(vx, vy)
            if nua < 2.0 or nvb < 2.0:
                continue
            d2 = (ux * vx + uy * vy) / (nua * nvb)
            # Open in the validation model, folded backward on screen.
            if d3 > 0.35 and d2 < -0.45:
                return True
        def _side(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        for i in range(n_pts - 1):
            for j in range(i + 2, n_pts - 1):
                a, bseg = proj[i], proj[i + 1]
                cpt, dpt = proj[j], proj[j + 1]
                if _side(a, bseg, cpt) * _side(a, bseg, dpt) < 0 and _side(cpt, dpt, a) * _side(cpt, dpt, bseg) < 0:
                    return True
        return False

    blues = []
    if blue_uv is not None:
        arr = np.asarray(blue_uv, dtype=np.float64).reshape(-1, 2)
        for row in arr:
            u, v = float(row[0]), float(row[1])
            if math.isfinite(u) and math.isfinite(v):
                blues.append((u, v))

    def score(proj, *, tags_only: bool):
        cost = 0.0
        n_obs = 0
        for i, uv in tags.items():
            if i < 2 or i >= len(proj):
                continue
            du = proj[i][0] - uv[0]
            dv = proj[i][1] - uv[1]
            cost += du * du + dv * dv
            n_obs += 1
        if tags_only:
            return cost, n_obs
        if n_obs == 0 and blues:
            for i in (2, 3, 4):
                if i >= len(proj):
                    continue
                best_d = None
                for bu, bv in blues:
                    d = (proj[i][0] - bu) ** 2 + (proj[i][1] - bv) ** 2
                    if best_d is None or d < best_d:
                        best_d = d
                if best_d is not None:
                    cost += best_d
                    n_obs += 1
        return cost, n_obs

    def _unit_dir(a, b):
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        n = math.hypot(dx, dy)
        if n < 12.0:
            return None
        return (dx / n, dy / n)

    # Camera heading of the live FK, after the view is seated.
    # ur_ik stays on one elbow. The other image side is the bend it refuses.
    hint_arm = None
    hint_sign = None
    if hint_pix is not None and len(hint_pix) > 2 and hint_pix[2] is not None:
        origin = None
        if len(hint_pix) > 1 and hint_pix[1] is not None:
            origin = hint_pix[1]
        elif hint_pix[0] is not None:
            origin = hint_pix[0]
        if origin is not None:
            hint_arm = _unit_dir(origin, hint_pix[2])
            ax = float(s[0]) - float(b[0])
            ay = float(s[1]) - float(b[1])
            hint_sign = ax * (float(hint_pix[2][1]) - float(origin[1])) - ay * (
                float(hint_pix[2][0]) - float(origin[0])
            )
            if abs(hint_sign) < 200.0:
                hint_sign = None

    def align_cost(proj):
        if hint_arm is None or len(proj) < 3:
            return None
        direction = _unit_dir(proj[1], proj[2])
        if direction is None:
            return None
        return 1.0 - (hint_arm[0] * direction[0] + hint_arm[1] * direction[1])

    def elbow_sign(proj) -> float:
        ax = float(s[0]) - float(b[0])
        ay = float(s[1]) - float(b[1])
        return ax * (proj[2][1] - proj[1][1]) - ay * (proj[2][0] - proj[1][0])

    use_tags = any(i >= 2 for i in tags)
    now = time.monotonic()
    held = _ARM_HOLD.get(cam) if cam else None
    held_theta = None
    if held is not None and now - float(held[0]) < 2.0:
        cand = float(held[1])
        if not contorted(project(cand)):
            held_theta = cand

    best = None
    best_theta = 0.0
    best_cost = 1e18
    best_n = 0
    # A held twist, or the camera's seated heading, owns the elbow side.
    # Blue blobs are not allowed to pick the mirrored bend.
    locked_side = False
    if held_theta is not None and not use_tags:
        best = project(held_theta)
        best_theta = held_theta
        locked_side = True
    elif hint_sign is not None and not use_tags:
        # Same side of the shoulder as the camera heading. A steep upper arm
        # still only leans a little, so a direction match alone cannot see it.
        best_align = 1e9
        for deg in range(0, 360, 3):
            theta = math.radians(deg)
            proj = project(theta)
            if contorted(proj):
                continue
            if elbow_sign(proj) * hint_sign <= 0.0:
                continue
            cost = align_cost(proj)
            if cost is None:
                cost = 0.0
            if cost < best_align:
                best_align = cost
                best = proj
                best_theta = theta
                locked_side = True
    if not locked_side:
        for deg in range(0, 360, 3):
            theta = math.radians(deg)
            proj = project(theta)
            if contorted(proj):
                continue
            cost, n_obs = score(proj, tags_only=use_tags)
            if n_obs == 0:
                continue
            if cost < best_cost:
                best_cost = cost
                best = proj
                best_theta = theta
                best_n = n_obs
        if held_theta is not None:
            held_proj = project(held_theta)
            held_cost, held_n = score(held_proj, tags_only=True)
            jump = abs((best_theta - held_theta + math.pi) % (2.0 * math.pi) - math.pi)
            keep = best is None or best_n == 0 or jump > math.radians(20.0)
            if (
                keep
                and best is not None
                and best_n > 0
                and held_n > 0
                and held_cost > 2500.0
                and best_cost < held_cost * 0.55
            ):
                keep = False
            if keep:
                best = held_proj
                best_theta = held_theta
    if best is None:
        # First look. Use the first rigid view that does not fold a joint backward.
        best_theta = 0.0
        best = project(0.0)
        for deg in range(0, 360, 3):
            theta = math.radians(deg)
            proj = project(theta)
            if not contorted(proj):
                best = proj
                best_theta = theta
                break
    if cam:
        rounded = []
        for u, v in best:
            rounded.append((int(round(u)), int(round(v))) if math.isfinite(u) and math.isfinite(v) else None)
        rounded[0] = (int(round(b[0])), int(round(b[1])))
        rounded[1] = (int(round(s[0])), int(round(s[1])))
        _ARM_HOLD[cam] = (now, best_theta, rounded)
    if best is None:
        return None
    out: list[Optional[tuple[int, int]]] = []
    for u, v in best:
        if not (math.isfinite(u) and math.isfinite(v)):
            out.append(None)
        else:
            out.append((int(round(u)), int(round(v))))
    # The shoulder coordinate from the validation script lands on its tag.
    out[0] = (int(round(b[0])), int(round(b[1])))
    out[1] = (int(round(s[0])), int(round(s[1])))
    return out


def _ur7e_name(model: str) -> str:
    key = str(model or "").lower()
    if key in ("", "ur5e", "ur7e", "ur5e_validation"):
        return "UR7e"
    return f"UR {model}"


def draw_ur_skeleton(
    bgr: np.ndarray,
    pts_vision_mm: np.ndarray,
    pose,
    *,
    fx: float = 520.0,
    matches: Optional[list[tuple[int, int, float]]] = None,
    rack_hits: Optional[list[tuple[int, float]]] = None,
    source: str = "",
    model: str = "",
    pts_upright_mm: Optional[np.ndarray] = None,
    check_ok: Optional[bool] = None,
    check_mm: Optional[float] = None,
    check_unit: str = "mm",
    base_anchor_uv: Optional[tuple[float, float]] = None,
    img_deg: float = 0.0,
    img_scale: float = 1.0,
    blue_uv: Optional[np.ndarray] = None,
    has_tool: bool = False,
    tag_uv: Optional[list] = None,
    pts_fk_mm: Optional[np.ndarray] = None,
    cam: str = "",
    calibrated: bool = False,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    K=None,
    dist=None,
) -> np.ndarray:
    """Project the kinematic chain into the camera and keep an upright stick figure.

    When ``calibrated`` is True the pixels are the pure projection through the
    saved camera pose. Every tag-fit / hold / blue-seat stage is skipped; the
    tags and blue caps are then used only for the match check and the HUD.
    """
    out = bgr
    h, w = out.shape[:2]
    # Real per-camera intrinsics when the viewer has them (from camera_info);
    # otherwise fall back to square pixels through the image center.
    fy = float(fx) if fy is None else float(fy)
    cx = 0.5 * w if cx is None else float(cx)
    cy = 0.5 * h if cy is None else float(cy)
    pts = np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 3)
    cams: list[Optional[np.ndarray]] = []
    pix: list[Optional[tuple[int, int]]] = []
    if pose is not None:
        if calibrated and K is not None:
            pix = _project_points_calibrated(pts, pose, K, dist)
            for p in pts:
                cams.append(base_to_cam(p, pose))
        else:
            for p in pts:
                p_cam = base_to_cam(p, pose)
                cams.append(p_cam)
                pix.append(project_cam_to_pixel(p_cam, fx, fy, cx, cy))
    else:
        cams = [None] * len(pts)
        pix = [None] * len(pts)

    # Validation-script coordinates (ur_fk). The shoulder tag fixes joint 1,
    # 162.5 mm up the base axis. The seated heading only picks which side
    # the elbow is on. It does not move the base or the shoulder.
    posed_from_tags = False
    fk_pts = pts_fk_mm if pts_fk_mm is not None else pts_upright_mm
    hint_for_twist = pix
    if pix and pix[0] is not None and (
        abs(float(img_deg)) > 1e-6 or abs(float(img_scale) - 1.0) > 1e-6
    ):
        ox, oy = float(pix[0][0]), float(pix[0][1])
        c = math.cos(math.radians(float(img_deg)))
        s = math.sin(math.radians(float(img_deg)))
        sc = float(img_scale)
        seated_hint = []
        for uv in pix:
            if uv is None:
                seated_hint.append(None)
            else:
                dx = float(uv[0]) - ox
                dy = float(uv[1]) - oy
                seated_hint.append(
                    (ox + sc * (c * dx - s * dy), oy + sc * (s * dx + c * dy))
                )
        hint_for_twist = seated_hint
    if (not calibrated) and fk_pts is not None and tag_uv:
        solved = place_validation_joints_sized(
            fk_pts,
            tag_uv,
            fx,
            w,
            h,
            hint_pix=hint_for_twist,
            cam=cam,
            blue_uv=blue_uv,
        )
        if solved is not None and len(solved) == len(pix):
            pix = solved
            posed_from_tags = True
    if (not calibrated) and not posed_from_tags and cam and cam in _ARM_HOLD:
        held_t, _held_theta, held_pix = _ARM_HOLD[cam]
        if time.monotonic() - float(held_t) < 0.8 and held_pix and len(held_pix) == len(pix):
            pix = _follow_held_chain(held_pix, tag_uv)
            posed_from_tags = True

    aligned_to_tags = False
    translated_to_tag = False
    if (not calibrated) and not posed_from_tags and tag_uv and pix and pix[0] is not None:
        fitted, kind = align_chain_about_base_tag(pix, tag_uv)
        if fitted is not None and len(fitted) == len(pix):
            pix = fitted
            if kind == "similarity":
                aligned_to_tags = True
            else:
                translated_to_tag = True

    # Slide the chain onto the base tag. Never onto the preset tape.
    shift_u, shift_v = 0, 0
    use_hold = (
        not calibrated
        and not posed_from_tags
        and not aligned_to_tags
        and not translated_to_tag
        and base_anchor_uv is not None
    )
    if use_hold and pix and pix[0] is not None:
        shift_u = int(round(float(base_anchor_uv[0]) - pix[0][0]))
        shift_v = int(round(float(base_anchor_uv[1]) - pix[0][1]))
        pix = [
            None if uv is None else (int(uv[0] + shift_u), int(uv[1] + shift_v))
            for uv in pix
        ]
    elif use_hold:
        if not pix:
            pix = [None]
        pix[0] = (int(round(float(base_anchor_uv[0]))), int(round(float(base_anchor_uv[1]))))

    # Seat the whole chain on the blue detections. Joint 0 stays on the tape.
    # Encoder angles are already in pts; this is only rotation and scale in the image.
    def _seat(u: float, v: float) -> tuple[float, float]:
        if (
            not pix
            or pix[0] is None
            or (abs(float(img_deg)) < 1e-6 and abs(float(img_scale) - 1.0) < 1e-6)
        ):
            return float(u), float(v)
        ox, oy = pix[0]
        dx, dy = float(u) - float(ox), float(v) - float(oy)
        c = math.cos(math.radians(float(img_deg)))
        s = math.sin(math.radians(float(img_deg)))
        return (
            float(ox) + float(img_scale) * (c * dx - s * dy),
            float(oy) + float(img_scale) * (s * dx + c * dy),
        )

    if (not calibrated) and (not posed_from_tags) and (not aligned_to_tags) and pix and pix[0] is not None:
        seated = []
        for uv in pix:
            if uv is None:
                seated.append(None)
            else:
                su, sv = _seat(uv[0], uv[1])
                seated.append((int(round(su)), int(round(sv))))
        pix = seated

    # 1:1 pull toward this camera's blue mask. Joint 0 stays on the tape.
    # A joint with no blue keeps the encoder projection.
    if (not calibrated) and (not posed_from_tags) and (not aligned_to_tags) and blue_uv is not None and pix:
        from ur_skeleton import blend_fk_blue_pixels

        fk_rows = []
        for uv in pix:
            if uv is None:
                fk_rows.append([np.nan, np.nan])
            else:
                fk_rows.append([float(uv[0]), float(uv[1])])
        blended, n_blend, mean_px = blend_fk_blue_pixels(
            np.asarray(fk_rows, dtype=np.float64),
            np.asarray(blue_uv, dtype=np.float64),
        )
        if n_blend >= 2 and mean_px is not None and math.isfinite(float(mean_px)):
            new_pix = []
            for i, uv in enumerate(pix):
                if i == 0 and uv is not None:
                    new_pix.append(uv)
                    continue
                u, v = float(blended[i, 0]), float(blended[i, 1])
                if not (math.isfinite(u) and math.isfinite(v)):
                    new_pix.append(uv)
                else:
                    new_pix.append((int(round(u)), int(round(v))))
            pix = new_pix
            check_mm = float(mean_px)
            check_unit = "px"
            check_ok = bool(mean_px <= 36.0)

    # Calibrated: pixels are the pure projection. Measure how far the seen tags
    # land from the projected joints, for the match/mismatch HUD only. Nothing
    # here moves the drawing.
    if calibrated:
        if tag_uv is not None and any(t is not None for t in tag_uv) and pix:
            errs = []
            for i, uv in enumerate(tag_uv):
                if uv is None or i >= len(pix) or pix[i] is None:
                    continue
                u, v = float(uv[0]), float(uv[1])
                if not (math.isfinite(u) and math.isfinite(v)):
                    continue
                du = u - float(pix[i][0])
                dv = v - float(pix[i][1])
                errs.append((du * du + dv * dv) ** 0.5)
            if errs:
                mean_tag = sum(errs) / float(len(errs))
                check_mm = float(mean_tag)
                check_unit = "px"
                check_ok = bool(mean_tag <= CALIB_TAG_OK_PX)

    # A seen 40 mm tag owns that joint. The flange (ur_fk's last point) keeps
    # the validation-script step off the last tag, so it cannot jump back to the base.
    fk_pix = list(pix)
    n_tags = 0
    if (not calibrated) and tag_uv and pix:
        for i, uv in enumerate(tag_uv):
            if uv is None or i >= len(pix):
                continue
            u, v = float(uv[0]), float(uv[1])
            if not (math.isfinite(u) and math.isfinite(v)):
                continue
            pix[i] = (int(round(u)), int(round(v)))
            n_tags += 1
        if n_tags:
            check_ok = True
            check_mm = float(n_tags)
            check_unit = "tag"
            # A validation placement already has the shoulder on its tag and
            # the other joints on the ur_fk coordinates. Rebase only when the
            # chain still comes from the old image heading.
            if not posed_from_tags:
                for i in range(1, len(pix)):
                    tagged = i < len(tag_uv) and tag_uv[i] is not None
                    if tagged or pix[i - 1] is None:
                        continue
                    if fk_pix[i] is None or fk_pix[i - 1] is None:
                        continue
                    du = int(fk_pix[i][0]) - int(fk_pix[i - 1][0])
                    dv = int(fk_pix[i][1]) - int(fk_pix[i - 1][1])
                    pix[i] = (int(pix[i - 1][0]) + du, int(pix[i - 1][1]) + dv)

    grip_hit = None

    # UR7e body: tube thickness from the STEP file, scaled by this bone's pixel length.
    try:
        from ur_skeleton import UR7E_LINK_RADIUS_MM
    except Exception:
        UR7E_LINK_RADIUS_MM = ()
    length_src = pts
    if pts_upright_mm is not None:
        upright = np.asarray(pts_upright_mm, dtype=np.float64).reshape(-1, 3)
        if upright.shape[0] == len(pix):
            length_src = upright
    for i in range(len(pix) - 1):
        if i >= len(UR7E_LINK_RADIUS_MM):
            break
        if pix[i] is None or pix[i + 1] is None:
            continue
        if i + 1 >= len(length_src):
            continue
        dmm = math.sqrt(
            (float(length_src[i + 1, 0]) - float(length_src[i, 0])) ** 2
            + (float(length_src[i + 1, 1]) - float(length_src[i, 1])) ** 2
            + (float(length_src[i + 1, 2]) - float(length_src[i, 2])) ** 2
        )
        du = float(pix[i + 1][0]) - float(pix[i][0])
        dv = float(pix[i + 1][1]) - float(pix[i][1])
        dpx = math.sqrt(du * du + dv * dv)
        if dmm < 20.0 or dpx < 2.0:
            continue
        radius_px = float(UR7E_LINK_RADIUS_MM[i]) * dpx / dmm
        thick = int(round(2.0 * radius_px))
        thick = max(4, min(thick, 72))
        seg = _clip_line_to_rect(
            float(pix[i][0]),
            float(pix[i][1]),
            float(pix[i + 1][0]),
            float(pix[i + 1][1]),
            w,
            h,
        )
        if seg is None:
            continue
        cv2.line(out, seg[0], seg[1], (168, 158, 148), thick, cv2.LINE_AA)
        cv2.circle(out, pix[i], max(3, thick // 2), (168, 158, 148), -1, cv2.LINE_AA)
        cv2.circle(out, pix[i + 1], max(3, thick // 2), (168, 158, 148), -1, cv2.LINE_AA)

    # Bones follow the blended joints, clipped when one leaves the frame.
    for i in range(len(pix) - 1):
        if pix[i] is None or pix[i + 1] is None:
            continue
        seg = _clip_line_to_rect(
            float(pix[i][0]),
            float(pix[i][1]),
            float(pix[i + 1][0]),
            float(pix[i + 1][1]),
            w,
            h,
        )
        if seg is None:
            continue
        if rack_hits and any(hi == i + 1 for hi, _ in rack_hits):
            color = (0, 0, 255)
        elif check_ok is False:
            color = (0, 140, 255)
        elif check_ok is None:
            color = (200, 180, 0)
        else:
            color = (80, 220, 80)
        cv2.line(out, seg[0], seg[1], color, 3, cv2.LINE_AA)

    agree_fk = {m[1] for m in (matches or [])}
    for i, uv in enumerate(pix):
        if uv is None:
            continue
        u, v = uv
        if u < 0 or v < 0 or u >= w or v >= h:
            continue
        ok = i in agree_fk
        hit = rack_hits and any(hi == i for hi, _ in rack_hits)
        # Index 6 is the flange, which ur_fk treats as the TCP.
        is_tool = i == 6 or (has_tool and i == len(pix) - 1)
        col = (0, 0, 255) if hit else ((0, 255, 0) if ok else (0, 200, 255))
        if is_tool:
            col = (0, 255, 255)
        cv2.circle(out, (u, v), 6, col, -1, cv2.LINE_AA)
        label = "GRIP" if is_tool else ("BASE" if i == 0 else ("SHOULDER" if i == 1 else ("ELBOW" if i == 2 else f"FK{i}")))
        if is_tool and grip_hit is not None:
            label = "GRIP" if not grip_hit.get("coasting") else "GRIP~"
        cv2.putText(
            out,
            label,
            (u + 6, max(14, v - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            col,
            1,
            cv2.LINE_AA,
        )

    # Vision confirm ticks on matched tracks
    if matches:
        for tid, fki, dist in matches:
            cv2.putText(
                out,
                f"J{tid}->FK{fki} {dist:.0f}mm",
                (12, 40 + 14 * tid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    if check_unit == "tag":
        hud = f"KIN on {int(check_mm or 0)} tags · UR7e"
        hud_col = (80, 220, 80)
    elif check_ok is True:
        unit = "px" if check_unit == "px" else "mm"
        hud = (
            f"KIN CHECK ok {check_mm:.0f}{unit}"
            if check_mm is not None
            else "KIN CHECK ok"
        )
        hud_col = (80, 220, 80)
    elif check_ok is False:
        unit = "px" if check_unit == "px" else "mm"
        extra = f" {check_mm:.0f}{unit}" if check_mm is not None else ""
        hud = f"KIN CHECK off{extra} — blue disagrees"
        hud_col = (0, 140, 255)
    else:
        hud = f"UR {model} kinematics (no blue check)"
        hud_col = (200, 180, 0)
    if source:
        token = source.split(":")[0].split()[0]
        if token.startswith("rtde-live"):
            token = "LIVE"
        elif token.startswith("rtde-hold"):
            token = "HOLD"
        elif token.startswith("photo"):
            token = "PHOTO"
        hud = f"{hud} [{token}]"
    if rack_hits:
        worst = min(rack_hits, key=lambda x: x[1])
        hud += f"  RACK FK{worst[0]} gap={worst[1]:.0f}mm"
        hud_col = (0, 0, 255)
    cv2.putText(
        out,
        hud,
        (12, h - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        hud_col,
        1,
        cv2.LINE_AA,
    )
    # Corner figure is how this camera sees the chain, angles preserved.
    graph = pts if pts_upright_mm is None else np.asarray(pts_upright_mm, dtype=np.float64)
    return draw_ur_monitor_inset(
        out,
        graph,
        rack_hits=rack_hits,
        source=source,
        model=model,
        cam_pix=pix,
    )


def fk_alerts_as_proximity(
    rack_hits: list[tuple[int, float]],
) -> list[tuple[int, int, int, int, float, str]]:
    """Synthetic proximity alert tuples for draw_proximity_error compatibility.

    Uses a dummy box in the corner; gap_mm and label carry the meaning.
    """
    alerts = []
    for idx, gap in rack_hits:
        alerts.append((20, 20 + 30 * idx, 40, 24, float(gap), f"fk_rack_{idx}"))
    return alerts
