"""Track pale baby-blue / cyan UR joint plastic via HSV color mask (no NN).

Used by ``blue_joint_tracks`` and ``dual_view_node``. Color boxes stay in color
pixels; sample depth after ``color_uv_to_depth_uv`` / ``scale_color_box_to_depth``,
then unproject with the color camera matrix from camera_info.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Seeded from user baby-blue joint crops (OpenCV HSV: H 0-179).
# The pedestal cap is paler (mean S about 38) than the upper joints.
# Mean-S gate still rejects white panels.
DEFAULT_HSV_LO = (88, 24, 140)
DEFAULT_HSV_HI = (104, 180, 255)
DEFAULT_MIN_AREA = 220
DEFAULT_MIN_MEAN_S = 36.0
DEFAULT_MIN_FILL = 0.18
DEFAULT_MAX_BOXES = 12
DEFAULT_PROXIMITY_MM = 100.0  # 10 cm
# Lab equipment obstacle colors (mask is never drawn):
# - matte black / near-black
# - cool light-grey / aluminum extrusion (seeded from user crop)
DEFAULT_BLACK_V_MAX = 58
DEFAULT_BLACK_S_MAX = 255
DEFAULT_SILVER_HSV_LO = (85, 22, 105)
DEFAULT_SILVER_HSV_HI = (115, 100, 220)
# Physical height of the cam1 jar rack (black face) for similar-triangles range.
DEFAULT_RACK_HEIGHT_MM = 400.0  # 40 cm
# Purple tape sits on the pedestal next to J0. Hue stops before red (H~170+).
# Do not require B>R — the tape is magenta (R slightly above B, G lowest).
# Cam1 often sees a smaller / less saturated strip — keep S a bit loose.
DEFAULT_PURPLE_HSV_LO = (150, 45, 80)
DEFAULT_PURPLE_HSV_HI = (168, 255, 255)
BOX_COLOR = (255, 220, 80)
MASK_TINT = (255, 200, 60)
ALERT_COLOR = (0, 0, 255)  # BGR red


def baby_blue_mask(
    bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int] = DEFAULT_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_HSV_HI,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    # Prefer slightly open to kill speckles on pale equipment.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def purple_tape_mask(
    bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int] = DEFAULT_PURPLE_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_PURPLE_HSV_HI,
) -> np.ndarray:
    """Mask for the purple tape strip. That strip is the cell origin (0,0)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    # Magenta tape: green is the darkest channel. Strong red has R >> B.
    b, g, r = cv2.split(bgr)
    bi = b.astype(np.int16)
    gi = g.astype(np.int16)
    ri = r.astype(np.int16)
    magenta = ((gi + 12) < bi) & ((gi + 12) < ri) & ((ri - bi) < 60)
    mask = cv2.bitwise_and(mask, magenta.astype(np.uint8) * 255)
    # Soft close only — OPEN erases the tiny strip seen by the other camera.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 4), np.uint8))
    return mask

def color_uv_to_depth_uv(
    u: float,
    v: float,
    color_hw: tuple[int, int],
    depth_hw: tuple[int, int],
) -> tuple[float, float]:
    """Map a color-pixel (u, v) into the depth frame.

    CAP_OBSENSOR publishes native-resolution depth (not D2C). When the two
    frames share the same optical model and only differ in size, this is a
    uniform scale. It is not a substitute for Orbbec AlignFilter.
    """
    ch, cw = int(color_hw[0]), int(color_hw[1])
    dh, dw = int(depth_hw[0]), int(depth_hw[1])
    if ch <= 0 or cw <= 0 or (ch, cw) == (dh, dw):
        return float(u), float(v)
    return float(u) * dw / float(cw), float(v) * dh / float(ch)


def depth_uv_to_color_uv(
    u: float,
    v: float,
    color_hw: tuple[int, int],
    depth_hw: tuple[int, int],
) -> tuple[float, float]:
    """Inverse of ``color_uv_to_depth_uv``."""
    ch, cw = int(color_hw[0]), int(color_hw[1])
    dh, dw = int(depth_hw[0]), int(depth_hw[1])
    if dh <= 0 or dw <= 0 or (ch, cw) == (dh, dw):
        return float(u), float(v)
    return float(u) * cw / float(dw), float(v) * ch / float(dh)


def scale_color_box_to_depth(
    x: float,
    y: float,
    w: float,
    h: float,
    color_hw: tuple[int, int],
    depth_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map a color-pixel box into the depth frame (resolution scale only)."""
    u0, v0 = color_uv_to_depth_uv(x, y, color_hw, depth_hw)
    u1, v1 = color_uv_to_depth_uv(x + w, y + h, color_hw, depth_hw)
    return (
        int(round(u0)),
        int(round(v0)),
        max(1, int(round(u1 - u0))),
        max(1, int(round(v1 - v0))),
    )


def scale_depth_box_to_color(
    x: float,
    y: float,
    w: float,
    h: float,
    color_hw: tuple[int, int],
    depth_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map a depth-pixel box back into the color frame."""
    u0, v0 = depth_uv_to_color_uv(x, y, color_hw, depth_hw)
    u1, v1 = depth_uv_to_color_uv(x + w, y + h, color_hw, depth_hw)
    return (
        int(round(u0)),
        int(round(v0)),
        max(1, int(round(u1 - u0))),
        max(1, int(round(v1 - v0))),
    )


def depth_frame_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color_hw: tuple[int, int],
    depth_hw: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Color intrinsics scaled into the depth frame (same optical model,
    different resolution). Returns (fx, fy, cx, cy) in depth pixels."""
    ch, cw = int(color_hw[0]), int(color_hw[1])
    dh, dw = int(depth_hw[0]), int(depth_hw[1])
    if ch <= 0 or cw <= 0 or (ch, cw) == (dh, dw):
        return float(fx), float(fy), float(cx), float(cy)
    sx, sy = dw / float(cw), dh / float(ch)
    return float(fx) * sx, float(fy) * sy, float(cx) * sx, float(cy) * sy


def median_depth_mm(
    depth_mm: Optional[np.ndarray],
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    mask: Optional[np.ndarray] = None,
    color_hw: Optional[tuple[int, int]] = None,
) -> Optional[float]:
    """Median depth (mm) in box; ignores zeros / invalid. Depth must be mm.

    If ``color_hw`` is given and the depth frame is a different resolution, the
    box (in color pixels) is mapped into depth pixels first; the color-resolution
    ``mask`` no longer matches that patch, so a stable center crop is used.
    """
    if depth_mm is None or depth_mm.size == 0:
        return None
    dh, dw = depth_mm.shape[:2]
    if color_hw is not None:
        ch, cw = int(color_hw[0]), int(color_hw[1])
        if ch > 0 and cw > 0 and (ch, cw) != (dh, dw):
            x, y, w, h = scale_color_box_to_depth(x, y, w, h, color_hw, (dh, dw))
            mask = None
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dw, x + w), min(dh, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth_mm[y0:y1, x0:x1].astype(np.float32)
    if mask is not None:
        m = mask[y0:y1, x0:x1]
        if m.shape[:2] == patch.shape[:2]:
            vals = patch[m > 0]
        else:
            vals = patch.reshape(-1)
    else:
        # Center crop is more stable than full box edges
        ch, cw = patch.shape[:2]
        vals = patch[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4].reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 50.0) & (vals < 8000.0)]
    if vals.size < 8:
        return None
    return float(np.median(vals))


def format_distance_mm(depth_mm: Optional[float]) -> str:
    """Human label: centimetres if < 1000 mm, else metres."""
    if depth_mm is None or depth_mm <= 0:
        return "?"
    if depth_mm < 1000.0:
        return f"{depth_mm / 10.0:.0f}cm"
    return f"{depth_mm / 1000.0:.2f}m"


def black_equipment_mask(
    bgr: np.ndarray,
    *,
    v_max: int = DEFAULT_BLACK_V_MAX,
    s_max: int = DEFAULT_BLACK_S_MAX,
) -> np.ndarray:
    """HSV mask for matte black / near-black lab equipment (not drawn on screen)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0], np.uint8),
        np.array([180, int(s_max), int(v_max)], np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def silver_equipment_mask(
    bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int] = DEFAULT_SILVER_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_SILVER_HSV_HI,
) -> np.ndarray:
    """Cool light-grey / aluminum lab frames (not drawn). Seeded from user crop."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def gray_equipment_mask(
    bgr: np.ndarray,
    blue_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Neutral gray and aluminum that is not the robot arm or the table.

    The arm is the gray component touching the baby-blue joint caps. The table
    is the wide gray component along the bottom edge. Everything else (rack,
    extrusions, machine frames) stays, and a blue joint within 10 cm of it
    raises the proximity error.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    neutral = cv2.inRange(
        hsv, np.array([0, 0, 45], np.uint8), np.array([179, 48, 200], np.uint8)
    )
    gray = cv2.bitwise_or(neutral, silver_equipment_mask(bgr))
    if blue_mask is not None and blue_mask.shape[:2] == gray.shape[:2]:
        gray = cv2.bitwise_and(gray, cv2.bitwise_not(blue_mask))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pieces = cv2.erode(gray, np.ones((5, 5), np.uint8), iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(pieces)
    h, w = gray.shape[:2]
    drop_lut = np.zeros(int(count), np.uint8)
    blue_touch = None
    if blue_mask is not None and blue_mask.shape[:2] == gray.shape[:2]:
        blue_touch = cv2.dilate(blue_mask, np.ones((31, 31), np.uint8), iterations=1)
        touched = labels[blue_touch > 0].ravel()
        touch_counts = np.bincount(touched, minlength=int(count))
        for idx in range(1, int(count)):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 40 or int(touch_counts[idx]) == 0:
                continue
            # A small piece stuck to a blue cap is the arm. A rack or machine
            # that only meets the cap along one edge is kept.
            frac = float(touch_counts[idx]) / float(area)
            if area < 8000 or frac > 0.45:
                drop_lut[idx] = 1
    for idx in range(1, int(count)):
        if int(stats[idx, cv2.CC_STAT_AREA]) < 40:
            continue
        if drop_lut[idx]:
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        top = int(stats[idx, cv2.CC_STAT_TOP])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        wide = bw > int(0.35 * w)
        low = float(centroids[idx, 1]) > 0.72 * h and (top + bh) >= h - 8
        if wide and low:
            drop_lut[idx] = 1
    if int(drop_lut.sum()) > 0:
        dropped = np.zeros(pieces.shape, np.uint8)
        dropped[drop_lut[labels] > 0] = 255
        dropped = cv2.dilate(dropped, np.ones((15, 15), np.uint8), iterations=1)
        gray = cv2.bitwise_and(gray, cv2.bitwise_not(dropped))
    if blue_touch is not None:
        gray = cv2.bitwise_and(gray, cv2.bitwise_not(blue_touch))
    return gray


def equipment_obstacle_mask(
    bgr: np.ndarray,
    *,
    black_v_max: int = DEFAULT_BLACK_V_MAX,
) -> np.ndarray:
    """Union of black + silver/aluminum equipment colors (hidden; for proximity)."""
    mask = cv2.bitwise_or(
        black_equipment_mask(bgr, v_max=black_v_max),
        silver_equipment_mask(bgr),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def _robot_body_mask(bgr: np.ndarray) -> np.ndarray:
    """Rough UR grey / dark metal so we don't treat the arm as an obstacle."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Desaturated grey / charcoal arm links (+ slightly wider S for brushed metal)
    grey = cv2.inRange(
        hsv, np.array([0, 0, 30], np.uint8), np.array([180, 70, 220], np.uint8)
    )
    grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    grey = cv2.dilate(grey, np.ones((9, 9), np.uint8), iterations=1)
    return grey


def is_base_blue_box(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_h: int,
    *,
    y_frac: float = 0.62,
) -> bool:
    """True for the pedestal / base joint (lower part of the frame).

    That joint sits on the table forever — never use it for proximity alarms.
    """
    if frame_h <= 0:
        return False
    cy = y + 0.5 * h
    return cy >= float(y_frac) * float(frame_h)


def _unproject_mm(
    u: float,
    v: float,
    z_mm: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    """Pixel + depth → camera-frame XYZ (mm) via similar triangles / pinhole."""
    z = float(z_mm)
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def _dist3d_mm(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return float(
        np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    )


def _blue_point_mm(
    depth_mm: np.ndarray,
    blue_mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    d_blue: Optional[float],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color_hw: Optional[tuple[int, int]] = None,
) -> Optional[tuple[tuple[float, float, float], float]]:
    """Color-pixel box + depth frame → camera-frame XYZ (mm).

    ``x,y,w,h`` and ``fx,fy,cx,cy`` are in the **color** frame. Depth is
    sampled after an explicit color→depth scale (native Orbbec depth is not
    the same resolution as color). Unprojection uses the color ray.
    """
    if d_blue is None:
        d_blue = median_depth_mm(
            depth_mm, x, y, w, h, mask=blue_mask, color_hw=color_hw
        )
    if d_blue is None:
        return None
    u = x + 0.5 * w
    v = y + 0.5 * h
    return _unproject_mm(u, v, float(d_blue), fx=fx, fy=fy, cx=cx, cy=cy), float(
        d_blue
    )


def depth_from_known_height_mm(
    pixel_h: float,
    real_h_mm: float = DEFAULT_RACK_HEIGHT_MM,
    *,
    fy_px: float = 520.0,
) -> Optional[float]:
    """Range (mm) from similar triangles: Z = fy * H_real / H_pixels."""
    if pixel_h < 8.0 or real_h_mm <= 0 or fy_px <= 0:
        return None
    return float(fy_px) * float(real_h_mm) / float(pixel_h)


def _chain_from_base(
    boxes: list,
    frame_h: int,
) -> list:
    """Blue joints ordered from the pedestal out. Each step is one link."""
    if not boxes:
        return []
    base = None
    for b in boxes:
        if is_base_blue_box(b[0], b[1], b[2], b[3], frame_h):
            base = b
            break
    if base is None:
        base = max(boxes, key=lambda b: b[1] + 0.5 * b[3])
    chain = [base]
    rest = [b for b in boxes if b is not base]
    while rest:
        ax = chain[-1][0] + 0.5 * chain[-1][2]
        ay = chain[-1][1] + 0.5 * chain[-1][3]
        nxt = min(
            rest,
            key=lambda b: (b[0] + 0.5 * b[2] - ax) ** 2
            + (b[1] + 0.5 * b[3] - ay) ** 2,
        )
        chain.append(nxt)
        rest = [b for b in rest if b is not nxt]
    return chain


def _carve_link_tube(gray: np.ndarray, chain: list, fx: float) -> np.ndarray:
    """Remove the silver tube between blue joints so a cap is not measured against its own link."""
    if gray is None or len(chain) < 2:
        return gray
    arm = np.zeros(gray.shape, np.uint8)
    for a, b in zip(chain, chain[1:]):
        ax, ay = int(a[0] + 0.5 * a[2]), int(a[1] + 0.5 * a[3])
        bx, by = int(b[0] + 0.5 * b[2]), int(b[1] + 0.5 * b[3])
        za = float(a[4]) if a[4] else 600.0
        zb = float(b[4]) if b[4] else za
        z = max(200.0, 0.5 * (za + zb))
        rad = int(round(55.0 * float(fx) / z))
        thick = max(18, min(96, 2 * rad))
        cv2.line(arm, (ax, ay), (bx, by), 255, thick)
        cv2.circle(arm, (ax, ay), max(10, rad), 255, -1)
        cv2.circle(arm, (bx, by), max(10, rad), 255, -1)
    return cv2.bitwise_and(gray, cv2.bitwise_not(arm))


def _nearest_gray_gap_mm(
    gray: np.ndarray,
    depth_mm: np.ndarray,
    u: float,
    v: float,
    origin: tuple[float, float, float],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    search_px: int = 160,
    ignore_px: int = 18,
) -> Optional[float]:
    """4th-closest stable gray pixel, in mm. Ignores the pixel under the sample."""
    dh, dw = depth_mm.shape[:2]
    x, y = int(round(u)), int(round(v))
    rad = max(40, int(search_px))
    x0, y0 = max(0, x - rad), max(0, y - rad)
    x1, y1 = min(dw, x + rad), min(dh, y + rad)
    g_roi = gray[y0:y1, x0:x1]
    if g_roi.size == 0 or cv2.countNonZero(g_roi) < 8:
        return None
    gd = depth_mm[y0:y1, x0:x1].astype(np.float32)
    med = cv2.medianBlur(np.clip(gd, 0, 65535).astype(np.uint16), 5).astype(np.float32)
    ys, xs = np.where(
        (g_roi > 0)
        & (np.abs(gd - med) < 40.0)
        & np.isfinite(gd)
        & (gd > 50.0)
        & (gd < 8000.0)
    )
    if xs.size < 8:
        return None
    step = max(1, int(xs.size // 200))
    gaps: list[float] = []
    for i in range(0, int(xs.size), step):
        pu = float(xs[i] + x0)
        pv = float(ys[i] + y0)
        if ((pu - u) ** 2 + (pv - v) ** 2) ** 0.5 < float(ignore_px):
            continue
        gz = float(gd[ys[i], xs[i]])
        # Same depth as the link is the arm itself, not a separate machine.
        if abs(gz - float(origin[2])) < 80.0:
            continue
        gp = _unproject_mm(pu, pv, gz, fx=fx, fy=fy, cx=cx, cy=cy)
        gaps.append(_dist3d_mm(origin, gp))
    if len(gaps) < 12:
        return None
    gaps.sort()
    return gaps[3]


def proximity_alerts(
    depth_mm: Optional[np.ndarray],
    boxes: list[tuple[int, int, int, int, Optional[float]]],
    blue_mask: np.ndarray,
    *,
    bgr: Optional[np.ndarray] = None,
    thresh_mm: float = DEFAULT_PROXIMITY_MM,
    fx_px: float = 520.0,
    fy_px: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    skip_base: bool = True,
    black_v_max: int = DEFAULT_BLACK_V_MAX,
    search_px: int = 180,
    rack_xywh: Optional[tuple[int, int, int, int]] = None,
    rack_height_mm: float = DEFAULT_RACK_HEIGHT_MM,
    gray_mask: Optional[np.ndarray] = None,
) -> list[tuple[int, int, int, int, float, str]]:
    """Alert when a non-base blue joint is within ``thresh_mm`` of equipment.

    Hidden obstacle mask = black + cool silver/aluminum. Distance uses each
    pixel's own depth (similar triangles / pinhole). Arm body is carved out.

    For the cam1 jar rack, pass ``rack_xywh`` + known physical height (40 cm).
    Rack-plane depth is then Z = fy * 400mm / rack_pixel_h so an arm that only
    *looks* in front of the rack in 2D does not false-alarm when the rack is
    much farther in Z.

    Returns (x, y, w, h, gap_mm, label).
    """
    if depth_mm is None or depth_mm.size == 0 or not boxes or bgr is None:
        return []
    dh, dw = depth_mm.shape[:2]
    if bgr.shape[:2] != (dh, dw):
        return []

    fx = max(50.0, float(fx_px))
    fy = fx if fy_px is None else max(50.0, float(fy_px))
    cx_f = 0.5 * dw if cx is None else float(cx)
    cy_f = 0.5 * dh if cy is None else float(cy)

    z_rack: Optional[float] = None
    rack_box: Optional[tuple[int, int, int, int]] = None
    if rack_xywh is not None and len(rack_xywh) == 4:
        rx, ry, rw, rh = (int(v) for v in rack_xywh)
        if rw > 0 and rh > 0:
            rack_box = (rx, ry, rw, rh)
            z_rack = depth_from_known_height_mm(
                float(rh), float(rack_height_mm), fy_px=fy
            )

    equip = equipment_obstacle_mask(bgr, black_v_max=black_v_max)
    body = _robot_body_mask(bgr)
    # Carve out the arm (grey + blue) so cables / joint plastic don't alarm
    arm = cv2.bitwise_or(body, blue_mask)
    arm = cv2.dilate(arm, np.ones((21, 21), np.uint8), iterations=1)
    obstacle = cv2.bitwise_and(equip, cv2.bitwise_not(arm))
    gray = None
    if gray_mask is not None and gray_mask.shape[:2] == (dh, dw):
        gray = gray_mask
    chain = _chain_from_base(boxes, dh)
    if gray is not None and len(chain) >= 2:
        # L1's own silver tube is not equipment. Leave the rack beside it.
        gray = _carve_link_tube(gray, chain, fx)
    if cv2.countNonZero(obstacle) < 40 and z_rack is None and gray is None:
        return []

    alerts: list[tuple[int, int, int, int, float, str]] = []
    rad = max(40, int(search_px))
    for x, y, w, h, d_blue in boxes:
        bp = _blue_point_mm(
            depth_mm, blue_mask, x, y, w, h, d_blue, fx=fx, fy=fy, cx=cx_f, cy=cy_f
        )
        if bp is None:
            continue
        p_blue, d_blue_f = bp
        best: Optional[float] = None
        best_label = "equip"
        is_base = skip_base and is_base_blue_box(x, y, w, h, dh)

        # Gray equipment (rack, extrusions) uses the depth camera directly.
        # The arm and the table are already out of this mask.
        if gray is not None:
            gx0 = max(0, x - rad)
            gy0 = max(0, y - rad)
            gx1 = min(dw, x + w + rad)
            gy1 = min(dh, y + h + rad)
            g_roi = gray[gy0:gy1, gx0:gx1].copy()
            # The silver link under a blue cap is not equipment. Ignore gray
            # on the joint itself.
            margin = 28
            ix0 = max(0, int(x - margin - gx0))
            iy0 = max(0, int(y - margin - gy0))
            ix1 = min(g_roi.shape[1], int(x + w + margin - gx0))
            iy1 = min(g_roi.shape[0], int(y + h + margin - gy0))
            if ix1 > ix0 and iy1 > iy0:
                g_roi[iy0:iy1, ix0:ix1] = 0
            gd_roi = depth_mm[gy0:gy1, gx0:gx1].astype(np.float32)
            depth_u = np.clip(gd_roi, 0, 65535).astype(np.uint16)
            depth_med = cv2.medianBlur(depth_u, 5).astype(np.float32)
            stable = np.abs(gd_roi - depth_med) < 40.0
            gys, gxs = np.where(
                (g_roi > 0)
                & stable
                & np.isfinite(gd_roi)
                & (gd_roi > 50.0)
                & (gd_roi < 8000.0)
            )
            gaps: list[float] = []
            if gxs.size >= 8:
                gstep = max(1, int(gxs.size // 200))
                for i in range(0, int(gxs.size), gstep):
                    gz = float(gd_roi[gys[i], gxs[i]])
                    # The cap's own housing and the surface it sits on share
                    # its depth. A separate machine is in front of or behind that.
                    if abs(gz - d_blue_f) < 80.0:
                        continue
                    pu = float(gxs[i] + gx0)
                    pv = float(gys[i] + gy0)
                    gp = _unproject_mm(
                        pu, pv, gz, fx=fx, fy=fy, cx=cx_f, cy=cy_f
                    )
                    gaps.append(_dist3d_mm(p_blue, gp))
            if len(gaps) >= 12:
                gaps.sort()
                # A few closest stable pixels, so one depth speck does not
                # count and a far wall in the same window does not hide the rack.
                gap_g = gaps[3]
                if best is None or gap_g < best:
                    best = gap_g
                    best_label = "gray"

        # The pedestal cap sits on the table. It is allowed to be near gray.
        if is_base:
            continue

        # Explicit blue ↔ rack plane check using similar-triangles rack depth
        if rack_box is not None and z_rack is not None:
            rx, ry, rw, rh = rack_box
            # Sample a few points on the rack face at fixed Z_rack
            rack_pts = (
                (rx + 0.5 * rw, ry + 0.5 * rh),
                (rx + 0.15 * rw, ry + 0.2 * rh),
                (rx + 0.85 * rw, ry + 0.2 * rh),
                (rx + 0.15 * rw, ry + 0.8 * rh),
                (rx + 0.85 * rw, ry + 0.8 * rh),
            )
            for uu, vv in rack_pts:
                p_rack = _unproject_mm(
                    float(uu), float(vv), float(z_rack), fx=fx, fy=fy, cx=cx_f, cy=cy_f
                )
                gap_r = _dist3d_mm(p_blue, p_rack)
                if best is None or gap_r < best:
                    best = gap_r
                    best_label = "rack"

        x0 = max(0, x - rad)
        y0 = max(0, y - rad)
        x1 = min(dw, x + w + rad)
        y1 = min(dh, y + h + rad)
        d_roi = depth_mm[y0:y1, x0:x1].astype(np.float32)
        o_roi = obstacle[y0:y1, x0:x1] if obstacle.size else None
        if o_roi is not None:
            ys, xs = np.where(
                (o_roi > 0)
                & np.isfinite(d_roi)
                & (d_roi > 50.0)
                & (d_roi < 8000.0)
            )
        else:
            ys = np.array([], dtype=np.int32)
            xs = np.array([], dtype=np.int32)
        if xs.size >= 12:
            step = max(1, xs.size // 250)
            xs_s = xs[::step]
            ys_s = ys[::step]
            for i in range(xs_s.size):
                u = float(xs_s[i] + x0)
                v = float(ys_s[i] + y0)
                z = float(d_roi[ys_s[i], xs_s[i]])
                # Inside the rack ROI: ignore depth-camera hits on the arm and
                # use the similar-triangles rack range instead.
                if (
                    rack_box is not None
                    and z_rack is not None
                    and rack_box[0] <= u < rack_box[0] + rack_box[2]
                    and rack_box[1] <= v < rack_box[1] + rack_box[3]
                ):
                    z = float(z_rack)
                    label = "rack"
                else:
                    label = "equip"
                    # Depths nearly identical to the blue joint are usually the
                    # same foreground object (arm / cable), not the distant rack.
                    if abs(z - d_blue_f) < 40.0:
                        continue
                p_obs = _unproject_mm(u, v, z, fx=fx, fy=fy, cx=cx_f, cy=cy_f)
                gap = _dist3d_mm(p_blue, p_obs)
                if best is None or gap < best:
                    best = gap
                    best_label = label
        if best is not None and best <= float(thresh_mm):
            alerts.append(
                (int(x), int(y), int(w), int(h), float(best), best_label)
            )

    # Link L1 is the tube from the pedestal cap to the next blue joint.
    # The cap can sit ~12 cm off the rack while the tube itself is in it.
    if gray is not None and len(chain) >= 2:
        a, b = chain[0], chain[1]
        pa = _blue_point_mm(
            depth_mm, blue_mask, a[0], a[1], a[2], a[3], a[4],
            fx=fx, fy=fy, cx=cx_f, cy=cy_f,
        )
        pb = _blue_point_mm(
            depth_mm, blue_mask, b[0], b[1], b[2], b[3], b[4],
            fx=fx, fy=fy, cx=cx_f, cy=cy_f,
        )
        if pa is not None and pb is not None:
            best_l1 = None
            best_uv = None
            for t in (0.35, 0.5, 0.65):
                u = (1.0 - t) * (a[0] + 0.5 * a[2]) + t * (b[0] + 0.5 * b[2])
                v = (1.0 - t) * (a[1] + 0.5 * a[3]) + t * (b[1] + 0.5 * b[3])
                origin = (
                    (1.0 - t) * pa[0][0] + t * pb[0][0],
                    (1.0 - t) * pa[0][1] + t * pb[0][1],
                    (1.0 - t) * pa[0][2] + t * pb[0][2],
                )
                gap_l1 = _nearest_gray_gap_mm(
                    gray, depth_mm, u, v, origin,
                    fx=fx, fy=fy, cx=cx_f, cy=cy_f,
                )
                if gap_l1 is None:
                    continue
                if best_l1 is None or gap_l1 < best_l1:
                    best_l1 = gap_l1
                    best_uv = (u, v)
            if best_l1 is not None and best_l1 <= float(thresh_mm) and best_uv is not None:
                alerts.append(
                    (
                        int(best_uv[0]) - 20,
                        int(best_uv[1]) - 20,
                        40,
                        40,
                        float(best_l1),
                        "L1",
                    )
                )
    return alerts


def retreat_guidance_toward_base(
    depth_mm: Optional[np.ndarray],
    boxes: list[tuple[int, int, int, int, Optional[float]]],
    blue_mask: np.ndarray,
    alerts: list,
    *,
    fx_px: float = 520.0,
) -> Optional[str]:
    """Human instruction: move the alerting joint toward the robot base (zero).

    Uses the lowest blue blob (pedestal / base joint) as a live landmark for
    ``ur_base`` in this camera until full T_base_cam transforms exist.
    Camera axes: +X right in image, +Y down in image, +Z away from camera.
    """
    if depth_mm is None or depth_mm.size == 0 or not boxes or not alerts:
        return None
    dh, dw = depth_mm.shape[:2]
    fx = max(50.0, float(fx_px))
    fy = fx
    cx_f, cy_f = 0.5 * dw, 0.5 * dh

    base_box = None
    for b in boxes:
        if is_base_blue_box(b[0], b[1], b[2], b[3], dh):
            base_box = b
            break
    if base_box is None:
        # Fallback: lowest box in the frame
        base_box = max(boxes, key=lambda b: b[1] + 0.5 * b[3])

    bp_base = _blue_point_mm(
        depth_mm,
        blue_mask,
        base_box[0],
        base_box[1],
        base_box[2],
        base_box[3],
        base_box[4],
        fx=fx,
        fy=fy,
        cx=cx_f,
        cy=cy_f,
    )
    if bp_base is None:
        return None
    p_base, _ = bp_base

    # Closest / worst alert joint
    alert = min(alerts, key=lambda a: float(a[4]))
    ax, ay, aw, ah = int(alert[0]), int(alert[1]), int(alert[2]), int(alert[3])
    # Prefer matching box depth from the blue list
    d_alert = None
    for bx, by, bw, bh, bd in boxes:
        if abs(bx - ax) < 8 and abs(by - ay) < 8:
            d_alert = bd
            break
    bp_j = _blue_point_mm(
        depth_mm,
        blue_mask,
        ax,
        ay,
        aw,
        ah,
        d_alert,
        fx=fx,
        fy=fy,
        cx=cx_f,
        cy=cy_f,
    )
    if bp_j is None:
        return None
    p_joint, _ = bp_j

    # Vector from joint → base (retreat direction)
    dx = p_base[0] - p_joint[0]
    dy = p_base[1] - p_joint[1]
    dz = p_base[2] - p_joint[2]
    dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if dist < 15.0:
        return (
            "PROXIMITY RETREAT: joint already near BASE landmark - "
            "pull arm gently toward pedestal / home"
        )

    parts: list[str] = []
    # Image-horizontal (camera X)
    if abs(dx) >= 30.0:
        parts.append(
            ("camera-RIGHT" if dx > 0 else "camera-LEFT")
            + f" ~{abs(dx) / 10.0:.0f}cm"
        )
    # Image-vertical (camera Y; +Y is down)
    if abs(dy) >= 30.0:
        parts.append(
            ("camera-DOWN" if dy > 0 else "camera-UP")
            + f" ~{abs(dy) / 10.0:.0f}cm"
        )
    # Depth (camera Z; +Z away from camera)
    if abs(dz) >= 30.0:
        parts.append(
            ("FARTHER from camera" if dz > 0 else "CLOSER to camera")
            + f" ~{abs(dz) / 10.0:.0f}cm"
        )
    if not parts:
        parts.append(f"toward BASE (~{dist / 10.0:.0f}cm)")

    return (
        "PROXIMITY RETREAT -> move arm toward ROBOT BASE (0,0,0): "
        + ", ".join(parts)
        + f"  |  remaining ~{dist / 10.0:.0f}cm to base landmark"
    )


def draw_proximity_error(
    bgr: np.ndarray,
    alerts: list,
    *,
    thresh_cm: float = 10.0,
) -> np.ndarray:
    """Paint a big red error banner + highlight offending blue boxes."""
    if not alerts:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), ALERT_COLOR, 14)
    cv2.rectangle(out, (10, 10), (w - 11, h - 11), ALERT_COLOR, 4)
    banner_h = max(72, h // 7)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), ALERT_COLOR, -1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
    gap_cm = min(float(a[4]) for a in alerts) / 10.0
    lines = [
        "ERROR: PROXIMITY",
        f"BLUE < {thresh_cm:.0f}cm OF GRAY / EQUIPMENT",
        f"closest gap ~ {gap_cm:.1f}cm",
    ]
    y = 28
    for i, text in enumerate(lines):
        scale = 1.15 if i == 0 else 0.75
        cv2.putText(
            out,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            3 if i == 0 else 2,
            cv2.LINE_AA,
        )
        y += 28 if i == 0 else 24
    for item in alerts:
        x, y0, bw, bh, gap = item[0], item[1], item[2], item[3], item[4]
        cv2.rectangle(out, (x, y0), (x + bw, y0 + bh), ALERT_COLOR, 3)
        cv2.putText(
            out,
            f"NEAR {item[5] if len(item) > 5 else 'EQUIP'} {gap / 10.0:.1f}cm",
            (x, max(banner_h + 20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ALERT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return out


def find_baby_blue_boxes(
    bgr: np.ndarray,
    *,
    min_area: int = DEFAULT_MIN_AREA,
    max_boxes: int = DEFAULT_MAX_BOXES,
    hsv_lo: tuple[int, int, int] = DEFAULT_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_HSV_HI,
    min_mean_s: float = DEFAULT_MIN_MEAN_S,
    min_fill: float = DEFAULT_MIN_FILL,
    depth_mm: Optional[np.ndarray] = None,
) -> list[tuple[int, int, int, int, Optional[float]]]:
    """Return list of (x, y, w, h, depth_mm_or_None) for strict baby-blue blobs."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = baby_blue_mask(bgr, hsv_lo=hsv_lo, hsv_hi=hsv_hi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int, Optional[float], float]] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < float(min_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        # Reject long thin pale strips (cabinet edges)
        aspect = max(w, h) / max(1.0, float(min(w, h)))
        if aspect > 4.5:
            continue
        roi_m = mask[y : y + h, x : x + w]
        fill = float(cv2.countNonZero(roi_m)) / max(1.0, float(w * h))
        if fill < float(min_fill):
            continue
        roi_s = hsv[y : y + h, x : x + w, 1]
        mean_s = float(cv2.mean(roi_s, mask=roi_m)[0]) if roi_m is not None else 0.0
        if mean_s < float(min_mean_s):
            continue
        quality = min(0.99, 0.45 * fill + 0.55 * (mean_s / 120.0))
        if quality < 0.35:
            continue
        # bgr is the color frame; depth may be the sensor's native resolution.
        d_mm = median_depth_mm(
            depth_mm, x, y, w, h, mask=mask, color_hw=bgr.shape[:2]
        )
        boxes.append((int(x), int(y), int(w), int(h), d_mm, quality))
    boxes.sort(key=lambda b: (b[5], -(b[2] * b[3])), reverse=True)
    return [(x, y, w, h, d) for x, y, w, h, d, _q in boxes[: max(1, int(max_boxes))]]

