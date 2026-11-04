"""One 40 mm ArUco tag on each UR joint.

The tags on the arm are 5x5, ids 0 through 5. A 4x4 pass remains so the older
table tag still decodes. Ids are fixed:

    0 base, 1 shoulder, 2 elbow, 3 wrist 1, 4 wrist 2, 5 wrist 3

A tag on a joint cap can be only ~10 px in the 640 image, so each blue-cap
box is enlarged before a second detect. Range uses the known 40 mm side
(solvePnP, then similar triangles). Matrix products in this file are loops:
this machine's NumPy GEMM aborts the viewer.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

MARKER_ID = 0
MARKER_MM = 40.0
JOINT_IDS = (0, 1, 2, 3, 4, 5)
JOINT_NAMES = {
    0: "base",
    1: "shoulder",
    2: "elbow",
    3: "wrist 1",
    4: "wrist 2",
    5: "wrist 3",
}
_DETECTORS: Optional[list] = None
_ZOOM: Optional[object] = None
_COAST: dict[str, tuple[float, dict]] = {}
COAST_S = 1.0


def _make_detector(dict_id: int, *, zoom: bool = False):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeStep = 6
    if zoom:
        # Local cap crops. The bits are soft, so allow a bit more correction.
        params.minMarkerPerimeterRate = 0.03
        params.adaptiveThreshWinSizeMax = 63
        params.perspectiveRemovePixelPerCell = 12
        params.errorCorrectionRate = 0.8
    else:
        # A tag that is obvious to the eye can still be only ~10 px, or washed
        # out. Accept a small perimeter and a softer threshold.
        params.minMarkerPerimeterRate = 0.008
        params.adaptiveThreshWinSizeMax = 63
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 5
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.01
        params.perspectiveRemovePixelPerCell = 8
        params.errorCorrectionRate = 0.75
    return cv2.aruco.ArucoDetector(dictionary, params)


def _detectors():
    """5x5 first (the joint tags), then 4x4 (the older table tag)."""
    global _DETECTORS
    if _DETECTORS is None:
        _DETECTORS = [
            _make_detector(cv2.aruco.DICT_5X5_50),
            _make_detector(cv2.aruco.DICT_4X4_50),
        ]
    return _DETECTORS


def _zoom_detector():
    global _ZOOM
    if _ZOOM is None:
        _ZOOM = _make_detector(cv2.aruco.DICT_5X5_50, zoom=True)
    return _ZOOM


def _quad_ok(corners: np.ndarray) -> Optional[float]:
    sides = []
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        du = float(a[0] - b[0])
        dv = float(a[1] - b[1])
        sides.append((du * du + dv * dv) ** 0.5)
    short = min(sides)
    long = max(sides)
    if short < 5.0 or long > 280.0:
        return None
    if long / short > 3.5:
        return None
    return 0.25 * sum(sides)


def range_mm_from_size(side_px: float, fx: float, *, marker_mm: float = MARKER_MM) -> Optional[float]:
    """Z = fx * real_side / pixel_side. None when the tag is too small to trust."""
    if side_px < 8.0 or fx <= 1.0 or marker_mm <= 0:
        return None
    z = float(fx) * float(marker_mm) / float(side_px)
    if z < 80.0 or z > 4000.0:
        return None
    return z


def range_mm_from_pose(
    corners: np.ndarray,
    fx: float,
    width: int,
    height: int,
    *,
    marker_mm: float = MARKER_MM,
) -> Optional[float]:
    """Distance to a 40 mm tag from its image quad. OpenCV pose, no NumPy GEMM."""
    if width <= 0 or height <= 0 or fx <= 1.0:
        return None
    half = 0.5 * float(marker_mm)
    obj = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )
    img = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    k = np.array(
        [[float(fx), 0.0, 0.5 * float(width)], [0.0, float(fx), 0.5 * float(height)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    try:
        ok, _rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=flag)
    except Exception:
        return None
    if not ok:
        return None
    z = float(tvec.reshape(3)[2])
    if z < 80.0 or z > 4000.0:
        return None
    return z


def _absorb(found: dict[int, dict], corners, ids, *, scale: float, origin: tuple[float, float], replace: bool) -> None:
    if ids is None:
        return
    ox, oy = float(origin[0]), float(origin[1])
    inv = 1.0 / float(scale) if scale else 1.0
    for k, mid in enumerate(ids.flatten()):
        mid_i = int(mid)
        if mid_i not in JOINT_NAMES:
            continue
        if mid_i in found and not replace:
            continue
        quad = np.asarray(corners[k][0], dtype=np.float64) * inv
        quad[:, 0] += ox
        quad[:, 1] += oy
        side = _quad_ok(quad)
        if side is None:
            continue
        prev = found.get(mid_i)
        if prev is not None and float(prev["side_px"]) >= float(side):
            continue
        found[mid_i] = {
            "id": mid_i,
            "name": JOINT_NAMES[mid_i],
            "corners": quad,
            "center": (float(quad[:, 0].mean()), float(quad[:, 1].mean())),
            "side_px": float(side),
            "coasting": False,
        }


def _scan_rois(gray: np.ndarray, found: dict[int, dict], rois) -> None:
    """Enlarge each joint-cap box. A 40 mm tag there can be ~10 px in the full frame."""
    h, w = gray.shape[:2]
    det = _zoom_detector()
    for roi in rois or ():
        if len(roi) < 4:
            continue
        x, y, rw, rh = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        if rw < 8 or rh < 8:
            continue
        pad = int(0.5 * max(rw, rh)) + 8
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + rw + pad), min(h, y + rh + pad)
        if x1 - x0 < 12 or y1 - y0 < 12:
            continue
        crop = gray[y0:y1, x0:x1]
        long_side = max(crop.shape[:2])
        scale = int(max(3, min(6, round(280.0 / float(long_side)))))
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(big, (0, 0), 2)
        sharp = cv2.addWeighted(big, 1.4, blur, -0.4, 0)
        corners, ids, _ = det.detectMarkers(sharp)
        _absorb(found, corners, ids, scale=float(scale), origin=(x0, y0), replace=True)


def _scan_enlarged(gray: np.ndarray, found: dict[int, dict]) -> None:
    """Read the whole frame at 2x. A tag does not have to sit in a blue box."""
    if all(i in found for i in JOINT_IDS):
        return
    det = _detectors()[0]
    big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = det.detectMarkers(big)
    _absorb(found, corners, ids, scale=2.0, origin=(0.0, 0.0), replace=True)
    if all(i in found for i in JOINT_IDS):
        return
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corners, ids, _ = det.detectMarkers(clahe.apply(big))
    _absorb(found, corners, ids, scale=2.0, origin=(0.0, 0.0), replace=True)


def find_joint_tags(bgr: np.ndarray, *, cam: str = "", rois=None) -> dict[int, dict]:
    """Tags 0..5. Each hit has corners, center, side_px, name, coasting.

    ``rois`` are (x, y, w, h) boxes, usually the baby-blue caps. Each one is
    enlarged so a tag that is only a few cells wide in the full frame still decodes.
    """
    found: dict[int, dict] = {}
    if bgr is not None and bgr.size != 0 and bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # 5x5 is the joint-tag dictionary. 4x4 runs only for the old table
        # frames, which call this with no cap boxes.
        five, four = _detectors()
        corners, ids, _ = five.detectMarkers(gray)
        _absorb(found, corners, ids, scale=1.0, origin=(0.0, 0.0), replace=True)
        _scan_enlarged(gray, found)
        if rois is None and not found:
            corners, ids, _ = four.detectMarkers(gray)
            _absorb(found, corners, ids, scale=1.0, origin=(0.0, 0.0), replace=False)
        _scan_rois(gray, found, rois)
        if cam:
            now = time.monotonic()
            for mid_i, hit in found.items():
                _COAST[f"{cam}:{mid_i}"] = (now, hit)
    if cam:
        now = time.monotonic()
        for mid_i in JOINT_IDS:
            if mid_i in found:
                continue
            key = f"{cam}:{mid_i}"
            if key not in _COAST:
                continue
            seen, hit = _COAST[key]
            if now - seen > COAST_S:
                continue
            held = dict(hit)
            held["coasting"] = True
            found[mid_i] = held
    return found


def find_marker0(bgr: np.ndarray, *, cam: str = "") -> Optional[dict]:
    """Tag 0 only. Kept so the original table-tag frames still decode."""
    return find_joint_tags(bgr, cam=cam).get(MARKER_ID)


def _coast(cam: str) -> Optional[dict]:
    return find_marker0(None, cam=cam)





_TAG_COLOR = (
    (0, 220, 0),
    (0, 200, 255),
    (255, 180, 0),
    (255, 80, 180),
    (180, 80, 255),
    (0, 140, 255),
)


def draw_joint_tag(
    bgr: np.ndarray,
    hit: dict,
    *,
    z_mm: Optional[float] = None,
    scale_uv: tuple[float, float] = (1.0, 1.0),
) -> None:
    """Draw one joint tag and the range from its 40 mm side."""
    sx, sy = float(scale_uv[0]), float(scale_uv[1])
    mid = int(hit.get("id") or 0)
    color = _TAG_COLOR[mid % len(_TAG_COLOR)]
    pts = []
    for p in hit["corners"]:
        pts.append([int(round(float(p[0]) * sx)), int(round(float(p[1]) * sy))])
    cv2.polylines(bgr, [np.asarray(pts, dtype=np.int32)], True, color, 2, cv2.LINE_AA)
    cu = int(round(float(hit["center"][0]) * sx))
    cv = int(round(float(hit["center"][1]) * sy))
    cv2.circle(bgr, (cu, cv), 4, color, -1, cv2.LINE_AA)
    name = str(hit.get("name") or JOINT_NAMES.get(mid, str(mid)))
    label = f"{name}~" if hit.get("coasting") else name
    if z_mm is not None and z_mm > 0:
        label = f"{label} {z_mm / 10.0:.0f}cm"
    cv2.putText(
        bgr,
        label,
        (cu + 8, max(16, cv - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )

