"""6-slot blue joint bank: HSV detections + Kalman coast in ur_base (no NN).

``JointBank.ingest_cam`` turns color-pixel blue boxes into tracks. Coasting
keeps IDs when a cap leaves view; ``draw_tracked_blue`` only paints live blobs.
Purple tape (``low_purple_tape_uv``) is the legacy origin marker.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .baby_blue_draw import (
    BOX_COLOR,
    MASK_TINT,
    _blue_point_mm,
    _unproject_mm,
    baby_blue_mask,
    format_distance_mm,
    is_base_blue_box,
    median_depth_mm,
)
from .cell_frames import CamPose, cam_to_base


DEFAULT_NUM_JOINTS = 6
DEFAULT_COAST_S = 10.0  # keep predicted labels/highlights long enough to be useful
DEFAULT_MERGE_MM = 80.0
DEFAULT_MATCH_MM = 120.0
DEFAULT_NMS_PX = 40.0
DEFAULT_DET_STALE_S = 0.35
DEFAULT_TAPE_HOLD_S = 3.0  # keep last purple origin if the mask flickers
DEFAULT_NN_HOLD_S = 12.0


def _lowest_blue_cap_uv(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Center of the lowest large light-blue joint cap in the frame."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (85, 40, 120), (110, 180, 255))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh = bgr.shape[0]
    best_y = -1.0
    best: Optional[np.ndarray] = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) < 800.0 or w < 24 or h < 24:
            continue
        cy = y + 0.5 * h
        if cy < 0.35 * fh:
            continue
        if cy > best_y:
            best_y = cy
            best = np.array([x + 0.5 * w, cy], dtype=np.float64)
    return best


def low_purple_tape_uv(bgr: np.ndarray, cam: str | None = None) -> Optional[np.ndarray]:
    """Center of the purple strip on the pedestal, beside the lower blue cap."""
    from .baby_blue_draw import purple_tape_mask

    mask = purple_tape_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = bgr.shape[:2]
    cands: list[tuple[float, float, float]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = float(max(cv2.contourArea(c), w * h * 0.6))
        if area < 12.0 or w < 5 or h < 2 or w < h:
            continue
        cy = y + 0.5 * h
        cx = x + 0.5 * w
        # Bottom edge of cam0 is the table, not the tape on the pedestal.
        if cy < 0.55 * fh or cy > 0.88 * fh:
            continue
        if cam == "cam0" and (cx < 0.15 * fw or cx > 0.70 * fw):
            continue
        cands.append((cx, cy, area))
    if not cands:
        return None
    cap = _lowest_blue_cap_uv(bgr)
    if cap is not None:
        near = []
        for cx, cy, area in cands:
            du = cx - float(cap[0])
            dv = cy - float(cap[1])
            dist = (du * du + dv * dv) ** 0.5
            if dist < 160.0:
                near.append((dist, -area, cx, cy))
        if near:
            near.sort()
            return np.array([near[0][2], near[0][3]], dtype=np.float64)
    cands.sort(key=lambda item: item[2], reverse=True)
    return np.array([cands[0][0], cands[0][1]], dtype=np.float64)


@dataclass
class Detection:
    cam: str
    x: int
    y: int
    w: int
    h: int
    d_mm: Optional[float]
    p_cam: Optional[np.ndarray]
    p_base: Optional[np.ndarray]
    is_base: bool
    area: int


@dataclass
class TrackView:
    """One track projected for drawing / proximity on a camera pane."""

    track_id: int
    x: int
    y: int
    w: int
    h: int
    d_mm: Optional[float]
    coasting: bool
    is_base: bool
    p_base: Optional[np.ndarray]
    p_cam: Optional[np.ndarray] = None  # camera-frame XYZ (mm) when depth is valid
    # Position relative to purple tape (0,0): radius mm + yaw degrees in XY.
    r_mm: Optional[float] = None
    ang_deg: Optional[float] = None
    nn_out: bool = False


def _polar_from_origin(p_base: Optional[np.ndarray]) -> tuple[Optional[float], Optional[float]]:
    """Distance (mm) and angle (deg) from purple-tape origin in the XY plane."""
    if p_base is None:
        return None, None
    p = np.asarray(p_base, dtype=np.float64).reshape(3)
    r = float(np.linalg.norm(p[:2]))
    if r < 1.0:
        return 0.0, 0.0
    ang = float(np.degrees(np.arctan2(p[1], p[0])))
    return r, ang


@dataclass
class JointTrack:
    track_id: int
    state: np.ndarray  # 6: x,y,z,vx,vy,vz relative to J0 origin (mm)
    P: np.ndarray
    last_seen_t: float
    last_cam: str
    boxes: dict = field(default_factory=dict)  # cam -> (x,y,w,h,d)
    is_base: bool = False
    coasting: bool = False
    seen_this_frame: bool = False
    has_base_frame: bool = False  # True once we have p_base for Kalman
    p_cam: Optional[np.ndarray] = None  # last camera-frame point (mm)
    hist: list = field(default_factory=list)  # recent xyz vs purple origin
    nn_out: bool = False  # True while the motion net is carrying this joint
    last_r_mm: Optional[float] = None
    last_ang_deg: Optional[float] = None


def _nms_image(
    dets: list[Detection], nms_px: float
) -> list[Detection]:
    if len(dets) <= 1:
        return dets
    ordered = sorted(dets, key=lambda d: d.area, reverse=True)
    kept: list[Detection] = []
    for d in ordered:
        cx, cy = d.x + 0.5 * d.w, d.y + 0.5 * d.h
        clash = False
        for k in kept:
            kx, ky = k.x + 0.5 * k.w, k.y + 0.5 * k.h
            if (cx - kx) ** 2 + (cy - ky) ** 2 <= nms_px**2:
                clash = True
                break
        if not clash:
            kept.append(d)
    return kept


def _merge_3d(dets: list[Detection], merge_mm: float) -> list[Detection]:
    """Merge detections whose ur_base points are within merge_mm."""
    with_base = [d for d in dets if d.p_base is not None]
    without = [d for d in dets if d.p_base is None]
    if len(with_base) <= 1:
        return dets
    ordered = sorted(with_base, key=lambda d: d.area, reverse=True)
    kept: list[Detection] = []
    for d in ordered:
        merged = False
        for i, k in enumerate(kept):
            if k.p_base is None or d.p_base is None:
                continue
            dist = float(np.linalg.norm(d.p_base - k.p_base))
            if dist <= merge_mm:
                # Prefer larger blob / finite depth; average p_base
                if d.area > k.area or (d.d_mm is not None and k.d_mm is None):
                    p = 0.5 * (d.p_base + k.p_base)
                    kept[i] = Detection(
                        cam=d.cam,
                        x=d.x,
                        y=d.y,
                        w=d.w,
                        h=d.h,
                        d_mm=d.d_mm if d.d_mm is not None else k.d_mm,
                        p_cam=d.p_cam if d.p_cam is not None else k.p_cam,
                        p_base=p,
                        is_base=d.is_base or k.is_base,
                        area=max(d.area, k.area),
                    )
                    # Keep both cam boxes by stuffing into boxes via cam field
                    # of the survivor; bank will also store from raw cam ingest.
                else:
                    kept[i] = Detection(
                        cam=k.cam,
                        x=k.x,
                        y=k.y,
                        w=k.w,
                        h=k.h,
                        d_mm=k.d_mm if k.d_mm is not None else d.d_mm,
                        p_cam=k.p_cam if k.p_cam is not None else d.p_cam,
                        p_base=0.5 * (d.p_base + k.p_base),
                        is_base=d.is_base or k.is_base,
                        area=max(d.area, k.area),
                    )
                merged = True
                break
        if not merged:
            kept.append(d)
    return kept + without


def _new_kalman(p_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros(6, dtype=np.float64)
    state[:3] = np.asarray(p_base, dtype=np.float64).reshape(3)
    P = np.eye(6, dtype=np.float64)
    P[:3, :3] *= 400.0  # mm^2
    P[3:, 3:] *= 2500.0  # (mm/s)^2
    return state, P


_NAN_LOGGED: set[str] = set()


def finite_blue_box(x, y, w, h) -> bool:
    """A blob the bank can place. Zero area or a non-finite corner is skipped."""
    try:
        vals = (float(x), float(y), float(w), float(h))
    except (TypeError, ValueError):
        _skip_nan("blue box not numeric")
        return False
    if not all(np.isfinite(v) for v in vals):
        _skip_nan("blue box coordinate")
        return False
    if vals[2] * vals[3] < 1.0:
        _skip_nan("blue box zero area")
        return False
    return True


def _skip_nan(where: str) -> None:
    if where in _NAN_LOGGED:
        return
    _NAN_LOGGED.add(where)
    print(f"blue joint NaN skipped: {where}", flush=True)


def _mm_small(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            acc = 0.0
            for k in range(A.shape[1]):
                acc += float(A[i, k]) * float(B[k, j])
            out[i, j] = acc
    return out


def _solve3(S: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Solve S x = b. A zero pivot returns None instead of NaN."""
    M = np.asarray(S, dtype=np.float64).copy()
    x = np.asarray(b, dtype=np.float64).reshape(3).copy()
    for col in range(3):
        pivot = col
        best = abs(float(M[col, col]))
        for r in range(col + 1, 3):
            v = abs(float(M[r, col]))
            if v > best:
                best = v
                pivot = r
        if best < 1e-8 or not np.isfinite(best):
            return None
        if pivot != col:
            row = M[col].copy()
            M[col] = M[pivot]
            M[pivot] = row
            x[col], x[pivot] = float(x[pivot]), float(x[col])
        div = float(M[col, col])
        for j in range(3):
            M[col, j] = float(M[col, j]) / div
        x[col] = float(x[col]) / div
        for r in range(3):
            if r == col:
                continue
            factor = float(M[r, col])
            if factor == 0.0:
                continue
            for j in range(3):
                M[r, j] = float(M[r, j]) - factor * float(M[col, j])
            x[r] = float(x[r]) - factor * float(x[col])
    if not np.isfinite(x).all():
        return None
    return x


def _kalman_predict(state: np.ndarray, P: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    dt = max(1e-3, float(dt))
    state = np.asarray(state, dtype=np.float64).reshape(6).copy()
    P = np.asarray(P, dtype=np.float64).reshape(6, 6).copy()
    if not np.isfinite(state).all() or not np.isfinite(P).all():
        _skip_nan("kalman predict state")
        return _new_kalman(np.zeros(3))
    for i in range(3):
        state[i] = float(state[i]) + dt * float(state[i + 3])
    F = np.eye(6, dtype=np.float64)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    PF = _mm_small(P, F.T)
    P = _mm_small(F, PF)
    q_pos = 50.0 * dt
    q_vel = 200.0 * dt
    for i in range(3):
        P[i, i] += q_pos
        P[i + 3, i + 3] += q_vel
    return state, P


def _kalman_update(
    state: np.ndarray, P: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Position update. K = P[:, :3] inv(S) with S = P[:3,:3] + 900 I, looped."""
    state = np.asarray(state, dtype=np.float64).reshape(6).copy()
    P = np.asarray(P, dtype=np.float64).reshape(6, 6).copy()
    z = np.asarray(z, dtype=np.float64).reshape(3)
    if not np.isfinite(z).all() or not np.isfinite(state).all() or not np.isfinite(P).all():
        _skip_nan("kalman update measurement")
        return state, P
    S = P[:3, :3].copy()
    for i in range(3):
        S[i, i] = float(S[i, i]) + 900.0
    eye = np.eye(3, dtype=np.float64)
    inv_cols = []
    for col in range(3):
        solved = _solve3(S, eye[col])
        if solved is None:
            _skip_nan("kalman update singular")
            return state, P
        inv_cols.append(solved)
    K = np.zeros((6, 3), dtype=np.float64)
    for i in range(6):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += float(P[i, k]) * float(inv_cols[j][k])
            K[i, j] = acc
    for i in range(6):
        acc = float(state[i])
        for k in range(3):
            acc += float(K[i, k]) * (float(z[k]) - float(state[k]))
        state[i] = acc
    P_new = P.copy()
    for r in range(6):
        for c in range(6):
            acc = float(P[r, c])
            for k in range(3):
                acc -= float(K[r, k]) * float(P[k, c])
            P_new[r, c] = acc
    if not np.isfinite(state).all() or not np.isfinite(P_new).all():
        _skip_nan("kalman update result")
        return state, P
    return state, P_new


class JointBank:
    """Fixed-size bank of blue joint tracks with Kalman coast in ur_base."""

    def __init__(
        self,
        *,
        num_joints: int = DEFAULT_NUM_JOINTS,
        coast_s: float = DEFAULT_COAST_S,
        merge_mm: float = DEFAULT_MERGE_MM,
        match_mm: float = DEFAULT_MATCH_MM,
        nms_px: float = DEFAULT_NMS_PX,
    ) -> None:
        self.num_joints = max(1, int(num_joints))
        self.coast_s = max(0.1, float(coast_s))
        self.merge_mm = max(10.0, float(merge_mm))
        self.match_mm = max(20.0, float(match_mm))
        self.nms_px = max(5.0, float(nms_px))
        self.tracks: list[Optional[JointTrack]] = [None] * self.num_joints
        self._cam_dets: dict[str, tuple[float, list[Detection]]] = {}
        self._last_tick = time.monotonic()
        self._lock_cam_boxes: dict[str, list[Detection]] = {"cam0": [], "cam1": []}
        # Pedestal blue joint in camera frame — cell origin (0,0,0) when set.
        self.j0_p_cam: Optional[np.ndarray] = None
        self.j0_cam: Optional[str] = None
        self.origin_ready: bool = False
        # Survivor det id -> peer Detection (other camera box for the same joint)
        self._peer_boxes: dict[int, list[Detection]] = {}
        # Purple tape is the shared origin (0,0), visible to both cameras.
        self.tape_seen: bool = False
        self._tape_box: dict[str, tuple[int, int, int, int]] = {}
        self._tape_p_cam: dict[str, np.ndarray] = {}
        self._tape_last_t: dict[str, float] = {}
        self._base_xy: dict[str, tuple[float, float]] = {}
        self._origin_lock: dict[str, dict] = {}
        self._robotics_root: Optional[Path] = None
        self.motion = None
        self.nn_steps = 0
        try:
            from .joint_motion_net import JointMotionNet
            from .cell_frames import default_cell_frames_path

            root = default_cell_frames_path().parent
            self._robotics_root = root
            self.motion = JointMotionNet(root / "joint_motion_mlp.npz")
            self.nn_steps = int(self.motion.steps)
        except Exception:
            self.motion = None
        if self._robotics_root is None:
            try:
                from .cell_frames import default_cell_frames_path

                self._robotics_root = default_cell_frames_path().parent
            except Exception:
                self._robotics_root = None
        self._load_origin_lock()

    def hud_counts(self) -> tuple[int, int, int]:
        """Return (live, predicted/coasting, capacity)."""
        live = sum(1 for t in self.tracks if t is not None and not t.coasting)
        coast = sum(1 for t in self.tracks if t is not None and t.coasting)
        return live, coast, self.num_joints

    def _origin_json_path(self) -> Optional[Path]:
        if self._robotics_root is None:
            return None
        return Path(self._robotics_root) / "base_origin.json"

    def _origin_reset_path(self) -> Optional[Path]:
        if self._robotics_root is None:
            return None
        return Path(self._robotics_root) / "RESET_BASE_ORIGIN"

    def _load_origin_lock(self) -> None:
        path = self._origin_json_path()
        self._origin_lock = {}
        if path is None or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if data.get("reset"):
            return
        for cam, rec in (data.get("cams") or {}).items():
            if not isinstance(rec, dict):
                continue
            if "u" not in rec or "v" not in rec:
                continue
            self._origin_lock[str(cam)] = rec

    def _save_origin_lock(self) -> None:
        path = self._origin_json_path()
        if path is None:
            return
        payload = {
            "note": (
                "Locked (0,0) is the purple tape on the robot pedestal. "
                "Both cameras store that same physical point. "
                "After you bump a camera, create RESET_BASE_ORIGIN in this "
                "folder and the next tape sighting stores the new offset."
            ),
            "cams": self._origin_lock,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def consume_origin_reset(self) -> bool:
        """Drop the locked base if the operator asked for a new offset."""
        flag = self._origin_reset_path()
        path = self._origin_json_path()
        reset = flag is not None and flag.is_file()
        if path is not None and path.is_file():
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("reset"):
                    reset = True
            except Exception:
                pass
        if not reset:
            return False
        self._origin_lock.clear()
        self._tape_box.clear()
        self._tape_p_cam.clear()
        self._tape_last_t.clear()
        self.j0_p_cam = None
        self.origin_ready = False
        if flag is not None and flag.is_file():
            try:
                flag.unlink()
            except Exception:
                pass
        self._save_origin_lock()
        print("base origin reset — next purple-tape sighting locks (0,0)")
        return True

    def locked_uv(self, cam: str | None) -> Optional[np.ndarray]:
        if not cam:
            return None
        rec = self._origin_lock.get(cam)
        if not rec:
            return None
        return np.array([float(rec["u"]), float(rec["v"])], dtype=np.float64)

    def lock_origin(
        self,
        cam: str,
        u: float,
        v: float,
        p_cam: Optional[np.ndarray] = None,
    ) -> bool:
        """Freeze this camera's (0,0) pixel. Returns True only the first time."""
        if cam in self._origin_lock:
            return False
        rec: dict = {"u": float(u), "v": float(v)}
        if p_cam is not None:
            rec["p_cam_mm"] = [float(x) for x in np.asarray(p_cam, dtype=np.float64).reshape(3)]
        self._origin_lock[cam] = rec
        self._save_origin_lock()
        print(f"base origin locked {cam} uv=({u:.0f},{v:.0f})")
        return True

    def lock_origin_depth(self, cam: str, p_cam: np.ndarray) -> None:
        rec = self._origin_lock.get(cam)
        if rec is None or rec.get("p_cam_mm"):
            return
        rec["p_cam_mm"] = [float(x) for x in np.asarray(p_cam, dtype=np.float64).reshape(3)]
        self._save_origin_lock()

    def ingest_cam(
        self,
        cam: str,
        boxes: list[tuple[int, int, int, int, Optional[float]]],
        depth: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        *,
        pose: Optional[CamPose],
        fx: float = 520.0,
        frame_h: int,
        frame_w: int,
        now: Optional[float] = None,
        bgr: Optional[np.ndarray] = None,
        cam_K: Optional[np.ndarray] = None,
        color_hw: Optional[tuple[int, int]] = None,
    ) -> list[TrackView]:
        """Ingest one camera's HSV boxes; return views for that cam pane.

        ``boxes``/``mask``/``bgr`` stay in **color** pixels. Depth is the
        sensor's native frame (often a different size). ``cam_K`` is the real
        color intrinsics; unprojection uses that K and samples depth after an
        explicit color→depth scale.
        """
        now = time.monotonic() if now is None else float(now)
        hw = color_hw or (bgr.shape[:2] if bgr is not None else (frame_h, frame_w))
        if cam_K is not None:
            K = np.asarray(cam_K, dtype=np.float64).reshape(3, 3)
            fxc, fyc, cxc, cyc = (
                float(K[0, 0]),
                float(K[1, 1]),
                float(K[0, 2]),
                float(K[1, 2]),
            )
        else:
            fxc, fyc, cxc, cyc = (
                float(fx),
                float(fx),
                0.5 * float(hw[1]),
                0.5 * float(hw[0]),
            )
        dets = self._boxes_to_dets(
            cam,
            boxes,
            depth,
            mask,
            pose=pose,
            fx=fxc,
            fy=fyc,
            cx=cxc,
            cy=cyc,
            frame_h=frame_h,
            frame_w=frame_w,
            color_hw=hw,
        )
        dets = _nms_image(dets, self.nms_px)
        self._remember_base_xy(
            cam,
            dets,
            color_hw=bgr.shape[:2] if bgr is not None else (frame_h, frame_w),
            det_hw=(frame_h, frame_w),
        )
        self._store_tape_box(
            cam, bgr, depth, fx=fxc, fy=fyc, cx=cxc, cy=cyc, color_hw=hw
        )
        self._cam_dets[cam] = (now, dets)
        self._lock_cam_boxes[cam] = dets
        self._align_tape_heading(
            cam, dets, fx=fxc, fy=fyc, cx=cxc, cy=cyc, frame_w=frame_w, frame_h=frame_h
        )
        self._associate(now)
        return self.views_for_cam(cam, frame_w=frame_w, frame_h=frame_h)

    def _remember_base_xy(
        self,
        cam: str,
        dets: list[Detection],
        *,
        color_hw: tuple[int, int],
        det_hw: tuple[int, int],
    ) -> None:
        """Pedestal blue center in the color image (purple tape must sit beside it)."""
        bi = self._pick_pedestal_for_cam(dets, cam)
        if bi is None:
            return
        d = dets[bi]
        cx, cy = d.x + 0.5 * d.w, d.y + 0.5 * d.h
        dh, dw = det_hw
        ch, cw = color_hw
        if dh > 0 and dw > 0 and (ch, cw) != (dh, dw):
            cx *= cw / float(dw)
            cy *= ch / float(dh)
        self._base_xy[cam] = (float(cx), float(cy))

    def _store_tape_box(
        self,
        cam: str,
        bgr: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        *,
        fx: float = 520.0,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        color_hw: Optional[tuple[int, int]] = None,
    ) -> None:
        if bgr is None:
            return
        fy_use = float(fx) if fy is None else float(fy)
        cx_use = 0.5 * float(bgr.shape[1]) if cx is None else float(cx)
        cy_use = 0.5 * float(bgr.shape[0]) if cy is None else float(cy)
        hw = color_hw or bgr.shape[:2]

        def _tape_xyz(u: float, v: float):
            if depth is None:
                return None
            d_mm = median_depth_mm(
                depth,
                max(0, int(u - 6)),
                max(0, int(v - 4)),
                12,
                8,
                color_hw=hw,
            )
            if d_mm is None or d_mm <= 50.0:
                return None
            return np.asarray(
                _unproject_mm(u, v, float(d_mm), fx=fx, fy=fy_use, cx=cx_use, cy=cy_use),
                dtype=np.float64,
            )

        self.consume_origin_reset()
        locked = self.locked_uv(cam)
        if locked is not None:
            u, v = float(locked[0]), float(locked[1])
            self._tape_box[cam] = (int(u - 10), int(v - 4), 20, 8)
            self._tape_last_t[cam] = time.monotonic()
            self.tape_seen = True
            rec = self._origin_lock.get(cam) or {}
            if rec.get("p_cam_mm"):
                self._tape_p_cam[cam] = np.asarray(rec["p_cam_mm"], dtype=np.float64)
            else:
                p = _tape_xyz(u, v)
                if p is not None:
                    self._tape_p_cam[cam] = p
                    self.lock_origin_depth(cam, p)
            return
        spot = low_purple_tape_uv(bgr, cam)
        if spot is None:
            age = time.monotonic() - self._tape_last_t.get(cam, 0.0)
            if cam in self._tape_box and age <= DEFAULT_TAPE_HOLD_S:
                return
            self._tape_box.pop(cam, None)
            self._tape_p_cam.pop(cam, None)
            self._tape_last_t.pop(cam, None)
            return
        u, v = float(spot[0]), float(spot[1])
        self._tape_box[cam] = (int(u - 10), int(v - 4), 20, 8)
        self._tape_last_t[cam] = time.monotonic()
        self.tape_seen = True
        p_cam = _tape_xyz(u, v)
        if p_cam is not None:
            self._tape_p_cam[cam] = p_cam
        self.lock_origin(cam, u, v, p_cam)

    def _align_tape_heading(
        self,
        cam: str,
        dets: list[Detection],
        *,
        fx: float,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Shift this camera so the purple tape is exactly (0,0,0)."""
        fy_use = float(fx) if fy is None else float(fy)
        cx_use = 0.5 * float(frame_w) if cx is None else float(cx)
        cy_use = 0.5 * float(frame_h) if cy is None else float(cy)
        origin = self._tape_p_cam.get(cam)
        if origin is None:
            box = self._tape_box.get(cam)
            bi = self._pick_pedestal_for_cam(dets, cam)
            if box is None or bi is None or dets[bi].p_cam is None:
                return
            x, y, w, h = box
            z = float(dets[bi].p_cam[2])
            if z < 50.0:
                return
            origin = np.asarray(
                _unproject_mm(
                    x + 0.5 * w,
                    y + 0.5 * h,
                    z,
                    fx=fx,
                    fy=fy_use,
                    cx=cx_use,
                    cy=cy_use,
                ),
                dtype=np.float64,
            )
            self._tape_p_cam[cam] = origin
        self.j0_p_cam = np.asarray(origin, dtype=np.float64).reshape(3)
        self.j0_cam = cam
        self.origin_ready = True
        for d in dets:
            if d.cam != cam or d.p_cam is None:
                continue
            d.p_base = np.asarray(d.p_cam, dtype=np.float64).reshape(3) - self.j0_p_cam
            # Tape itself (or the pedestal blob sitting on it) stays the origin.
            d.is_base = float(np.linalg.norm(d.p_base)) < 40.0
            if d.is_base:
                d.p_base = np.zeros(3, dtype=np.float64)

    def _boxes_to_dets(
        self,
        cam: str,
        boxes: list[tuple[int, int, int, int, Optional[float]]],
        depth: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        *,
        pose: Optional[CamPose],
        fx: float,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        frame_h: int,
        frame_w: int,
        color_hw: Optional[tuple[int, int]] = None,
    ) -> list[Detection]:
        out: list[Detection] = []
        fy_use = float(fx) if fy is None else float(fy)
        cx_f = 0.5 * float(frame_w) if cx is None else float(cx)
        cy_f = 0.5 * float(frame_h) if cy is None else float(cy)
        hw = color_hw or (frame_h, frame_w)
        for x, y, w, h, d in boxes:
            if not finite_blue_box(x, y, w, h):
                continue
            p_cam = None
            p_base = None
            d_use = d
            if depth is not None and mask is not None:
                bp = _blue_point_mm(
                    depth,
                    mask,
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                    d,
                    fx=fx,
                    fy=fy_use,
                    cx=cx_f,
                    cy=cy_f,
                    color_hw=hw,
                )
                if bp is not None and np.isfinite(np.asarray(bp[0], dtype=np.float64)).all():
                    p_cam = np.asarray(bp[0], dtype=np.float64)
                    d_use = bp[1]
                    if d_use is not None and not np.isfinite(float(d_use)):
                        _skip_nan("blue depth")
                        p_cam = None
                        d_use = None
                    # Tentative; _rebase_to_j0 overwrites once pedestal is known.
                    if pose is not None and pose.ready:
                        p_base = cam_to_base(p_cam, pose)
                        if p_base is not None and not np.isfinite(np.asarray(p_base)).all():
                            _skip_nan("p_base")
                            p_base = None
            is_base = is_base_blue_box(
                int(x), int(y), int(w), int(h), frame_h, y_frac=0.45
            )
            out.append(
                Detection(
                    cam=cam,
                    x=int(x),
                    y=int(y),
                    w=int(w),
                    h=int(h),
                    d_mm=d_use,
                    p_cam=p_cam,
                    p_base=p_base,
                    is_base=is_base,
                    area=int(w) * int(h),
                )
            )
        return out

    def _fresh_dets(self, now: float) -> list[Detection]:
        all_dets: list[Detection] = []
        for _cam, (t, dets) in self._cam_dets.items():
            if now - t <= DEFAULT_DET_STALE_S:
                all_dets.extend(dets)
        return all_dets

    def _pick_pedestal_for_cam(self, dets: list[Detection], cam: str) -> Optional[int]:
        """Pedestal index within one camera (lowest blue blob)."""
        idxs = [i for i, d in enumerate(dets) if d.cam == cam]
        if not idxs:
            return None
        flagged = [i for i in idxs if dets[i].is_base]
        pool = flagged or idxs
        return max(pool, key=lambda i: dets[i].y + 0.5 * dets[i].h)

    def _rebase_each_camera(self, dets: list[Detection]) -> None:
        """Per camera: pedestal = (0,0,0). Other joints get radius from that origin.

        Both cams then share a distance-from-base coordinate so the same joint
        matches across the perpendicular views without double-counting.
        """
        for cam in ("cam0", "cam1"):
            bi = self._pick_pedestal_for_cam(dets, cam)
            if bi is None:
                continue
            base = dets[bi]
            if base.p_cam is None:
                continue
            p0 = np.asarray(base.p_cam, dtype=np.float64).reshape(3)
            if cam == "cam0" or self.j0_p_cam is None:
                self.j0_p_cam = p0.copy()
                self.j0_cam = cam
            self.origin_ready = True
            # Purple tape, when seen by this camera, is the origin — not the blue blob.
            origin = self._tape_p_cam.get(cam)
            p0 = (
                np.asarray(origin, dtype=np.float64).reshape(3)
                if origin is not None
                else p0
            )
            for d in dets:
                if d.cam != cam or d.p_cam is None:
                    continue
                d.p_base = np.asarray(d.p_cam, dtype=np.float64).reshape(3) - p0
                d.is_base = float(np.linalg.norm(d.p_base)) < 45.0
                if d.is_base:
                    d.p_base = np.zeros(3, dtype=np.float64)
            if origin is None:
                base.is_base = True
                base.p_base = np.zeros(3, dtype=np.float64)
        # Mark every camera's pedestal in the raw pool so J0 can draw on both panes.
        for cam, pool in self._lock_cam_boxes.items():
            if not pool:
                continue
            flagged = [i for i, d in enumerate(pool) if d.is_base]
            pick = max(
                flagged or list(range(len(pool))),
                key=lambda i: pool[i].y + 0.5 * pool[i].h,
            )
            for i, d in enumerate(pool):
                d.is_base = i == pick
                if d.is_base:
                    d.p_base = np.zeros(3, dtype=np.float64)

    def _fuse_cross_cam(self, dets: list[Detection]) -> list[Detection]:
        """One detection per physical joint. Pair by (r, angle) from purple (0,0)."""
        bases = [d for d in dets if d.is_base]
        others = [d for d in dets if not d.is_base and d.p_base is not None]
        no_depth = [d for d in dets if not d.is_base and d.p_base is None]
        fused: list[Detection] = []
        if bases:
            primary = next((d for d in bases if d.cam == "cam0"), bases[0])
            fused.append(primary)
        used = [False] * len(others)
        a_idx = [i for i, d in enumerate(others) if d.cam == "cam0"]
        b_idx = [i for i, d in enumerate(others) if d.cam == "cam1"]
        pairs: list[tuple[float, int, int]] = []
        for i in a_idx:
            ra, aa = _polar_from_origin(others[i].p_base)
            if ra is None:
                continue
            for j in b_idx:
                rb, ab = _polar_from_origin(others[j].p_base)
                if rb is None:
                    continue
                # Same joint ≈ same distance from purple tape; angle can differ
                # a bit across perpendicular views, so weight radius more.
                dr = abs(ra - rb)
                da = abs(((aa or 0.0) - (ab or 0.0) + 180.0) % 360.0 - 180.0)
                if dr > self.merge_mm * 1.4:
                    continue
                cost = dr + 0.8 * da  # mm + deg
                if cost <= self.merge_mm * 1.6 + 35.0:
                    pairs.append((cost, i, j))
        pairs.sort(key=lambda x: x[0])
        used_b: set[int] = set()
        for _cost, i, j in pairs:
            if used[i] or j in used_b:
                continue
            used[i] = True
            used_b.add(j)
            a = others[i]
            b = others[j]
            a.area = max(a.area, b.area)
            # Average relative position from purple origin (helps Kalman).
            if a.p_base is not None and b.p_base is not None:
                a.p_base = 0.5 * (
                    np.asarray(a.p_base, dtype=np.float64).reshape(3)
                    + np.asarray(b.p_base, dtype=np.float64).reshape(3)
                )
            fused.append(a)
            self._peer_boxes.setdefault(id(a), []).append(b)
        for i, d in enumerate(others):
            if used[i] or i in used_b:
                continue
            fused.append(d)
        fused.extend(no_depth)
        return fused

    def _associate(self, now: float) -> None:
        dt = now - self._last_tick
        self._last_tick = now
        for i, tr in enumerate(self.tracks):
            if tr is None:
                continue
            tr.seen_this_frame = False
            if tr.has_base_frame and i != 0:
                tr.state, tr.P = _kalman_predict(tr.state, tr.P, dt)
            age = now - tr.last_seen_t
            tr.coasting = age > 0.05
            tr.nn_out = False
            # Keep Kalman-predicted joints (and their r/@ labels) for the full coast window.
            hold_s = self.coast_s
            if (
                i != 0
                and tr.coasting
                and tr.has_base_frame
                and self.motion is not None
                and self.motion.ready
                and len(tr.hist) >= 6
            ):
                window = np.asarray(tr.hist[-6:], dtype=np.float64).reshape(6, 3)
                nxt = self.motion.predict_next(window)
                nxt = np.asarray(nxt, dtype=np.float64).reshape(3)
                if not np.isfinite(nxt).all() or not np.isfinite(tr.state[:3]).all():
                    _skip_nan("motion net coast")
                else:
                    tr.state[:3] = 0.7 * nxt + 0.3 * tr.state[:3]
                tr.hist.append(tr.state[:3].copy())
                tr.hist = tr.hist[-24:]
                tr.nn_out = True
                hold_s = max(self.coast_s, DEFAULT_NN_HOLD_S)
            # Refresh sticky polar labels from the current Kalman state while coasting.
            if tr.has_base_frame:
                rr, aa = _polar_from_origin(tr.state[:3])
                if rr is not None:
                    tr.last_r_mm = rr
                if aa is not None:
                    tr.last_ang_deg = aa
            # Never delete J0. Others keep highlighting until hold_s expires.
            if i != 0 and age > hold_s:
                self.tracks[i] = None

        raw = self._fresh_dets(now)
        self._rebase_each_camera(raw)
        self._peer_boxes = {}
        dets = self._fuse_cross_cam(raw)
        # Also fold same-cam near-duplicates
        dets = _merge_3d(dets, self.merge_mm)
        base_i = next((i for i, d in enumerate(dets) if d.is_base), None)

        cam_box_pool = {
            "cam0": list(self._lock_cam_boxes.get("cam0") or []),
            "cam1": list(self._lock_cam_boxes.get("cam1") or []),
        }

        used_det = [False] * len(dets)

        # 1) Pedestal → J0 at exactly (0,0,0)
        if base_i is not None:
            d = dets[base_i]
            used_det[base_i] = True
            tr = self.tracks[0]
            if tr is None:
                st, P = _new_kalman(np.zeros(3))
                tr = JointTrack(
                    track_id=0,
                    state=st,
                    P=P,
                    last_seen_t=now,
                    last_cam=d.cam,
                    is_base=True,
                    has_base_frame=True,
                )
                self.tracks[0] = tr
            self._apply_det(tr, d, now, cam_box_pool, force_origin=True)

        # 2) Greedy match remaining to non-base tracks (distance from J0 frame)
        candidates = [
            (i, d)
            for i, d in enumerate(dets)
            if not used_det[i] and not d.is_base and d.p_base is not None
        ]
        active = [
            (i, t)
            for i, t in enumerate(self.tracks)
            if t is not None and i != 0 and t.has_base_frame
        ]
        pairs: list[tuple[float, int, int]] = []
        for di, d in candidates:
            for si, t in active:
                dist = float(np.linalg.norm(d.p_base - t.state[:3]))
                if dist <= self.match_mm:
                    pairs.append((dist, di, si))
        pairs.sort(key=lambda x: x[0])
        used_slots: set[int] = set()
        for _dist, di, si in pairs:
            if used_det[di] or si in used_slots:
                continue
            used_det[di] = True
            used_slots.add(si)
            self._apply_det(self.tracks[si], dets[di], now, cam_box_pool)

        # 3) Spawn by increasing radius from J0 (closest free → lowest free id)
        unmatched = [
            (i, d)
            for i, d in enumerate(dets)
            if not used_det[i] and not d.is_base and d.p_base is not None
        ]
        unmatched.sort(key=lambda idd: float(np.linalg.norm(idd[1].p_base)))
        for di, d in unmatched:
            slot = next(
                (i for i in range(1, self.num_joints) if self.tracks[i] is None),
                None,
            )
            if slot is None:
                break
            st, P = _new_kalman(d.p_base)
            tr = JointTrack(
                track_id=slot,
                state=st,
                P=P,
                last_seen_t=now,
                last_cam=d.cam,
                is_base=False,
                has_base_frame=True,
            )
            self.tracks[slot] = tr
            used_det[di] = True
            self._apply_det(tr, d, now, cam_box_pool)

        # 4) Pixel-only match leftovers
        for di, d in enumerate(dets):
            if used_det[di] or d.is_base:
                continue
            best_si = None
            best_d2 = (self.nms_px * 2.5) ** 2
            for si, t in enumerate(self.tracks):
                if t is None or si == 0:
                    continue
                box = t.boxes.get(d.cam)
                if box is None:
                    continue
                bx, by, bw, bh, _bd = box
                d2 = (d.x + 0.5 * d.w - (bx + 0.5 * bw)) ** 2 + (
                    d.y + 0.5 * d.h - (by + 0.5 * bh)
                ) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_si = si
            if best_si is not None:
                used_det[di] = True
                self._apply_det(self.tracks[best_si], d, now, cam_box_pool)

        # 5) Display IDs: J0 = origin; J1..J5 = nearest → farthest from J0
        self._assign_labels_by_radius()

    def _assign_labels_by_radius(self) -> None:
        """Label by distance from purple tape (0,0): closest → J1, then J2…"""
        j0 = self.tracks[0]
        if j0 is not None:
            j0.track_id = 0
            j0.is_base = True
            if j0.has_base_frame:
                j0.state[:3] = 0.0
                j0.state[3:] = 0.0
        others = [
            (i, t)
            for i, t in enumerate(self.tracks)
            if t is not None and i != 0 and t.has_base_frame
        ]
        # Primary: XY radius from purple origin; secondary: |Z|; tertiary: angle.
        def _key(it: tuple[int, JointTrack]) -> tuple[float, float, float]:
            p = it[1].state[:3]
            r, ang = _polar_from_origin(p)
            return (float(r or 0.0), float(abs(p[2])), float(ang or 0.0))

        others.sort(key=_key)
        for rank, (_slot, t) in enumerate(others, start=1):
            t.track_id = rank
            t.is_base = False

    def _apply_det(
        self,
        tr: JointTrack,
        d: Detection,
        now: float,
        cam_box_pool: dict[str, list[Detection]],
        *,
        force_origin: bool = False,
    ) -> None:
        tr.last_seen_t = now
        tr.last_cam = d.cam
        tr.coasting = False
        tr.seen_this_frame = True
        tr.is_base = tr.is_base or d.is_base or force_origin
        tr.boxes[d.cam] = (d.x, d.y, d.w, d.h, d.d_mm)
        if d.p_cam is not None:
            tr.p_cam = np.asarray(d.p_cam, dtype=np.float64).reshape(3)
        # Attach the other camera's box for this same physical joint.
        for peer in self._peer_boxes.get(id(d), []):
            tr.boxes[peer.cam] = (peer.x, peer.y, peer.w, peer.h, peer.d_mm)
        if d.is_base or force_origin:
            for cam, pool in cam_box_pool.items():
                if cam == d.cam:
                    continue
                bases = [od for od in pool if od.is_base]
                if bases:
                    od = bases[0]
                    tr.boxes[cam] = (od.x, od.y, od.w, od.h, od.d_mm)
        if force_origin:
            tr.state, tr.P = _new_kalman(np.zeros(3))
            tr.has_base_frame = True
            tr.state[:3] = 0.0
            tr.state[3:] = 0.0
            return
        if d.p_base is not None:
            if tr.has_base_frame:
                tr.state, tr.P = _kalman_update(tr.state, tr.P, d.p_base)
            else:
                tr.state, tr.P = _new_kalman(d.p_base)
                tr.has_base_frame = True
            self._learn_motion(tr)
            rr, aa = _polar_from_origin(tr.state[:3])
            if rr is not None:
                tr.last_r_mm = rr
            if aa is not None:
                tr.last_ang_deg = aa
        tr.nn_out = False

    def _learn_motion(self, tr: JointTrack) -> None:
        """Teach the net the next position while the joint is actually seen."""
        if not tr.has_base_frame or self.motion is None or tr.is_base:
            return
        pos = tr.state[:3].copy()
        if len(tr.hist) >= 6:
            window = np.asarray(tr.hist[-6:], dtype=np.float64).reshape(6, 3)
            self.motion.train(window, pos)
            self.nn_steps = int(self.motion.steps)
        tr.hist.append(pos)
        tr.hist = tr.hist[-24:]

    def views_for_cam(
        self, cam: str, *, frame_w: int, frame_h: int
    ) -> list[TrackView]:
        views: list[TrackView] = []
        for tr in self.tracks:
            if tr is None:
                continue
            box = tr.boxes.get(cam)
            live = box is not None and not tr.coasting
            if box is None:
                if tr.boxes:
                    box = next(iter(tr.boxes.values()))
                elif tr.has_base_frame:
                    # Still draw a ghost near image center-ish from last polar.
                    box = (max(0, frame_w // 2 - 20), max(0, frame_h // 2 - 20), 40, 40, None)
                else:
                    continue
            x, y, w, h, d = box
            coasting = not live
            if tr.is_base and tr.track_id == 0:
                # Origin joint: if we have any box this frame, never show as pred
                if tr.seen_this_frame:
                    coasting = False
            if coasting and tr.has_base_frame:
                if not np.isfinite(tr.state).all() or not finite_blue_box(x, y, w, h):
                    _skip_nan("coast highlight")
                else:
                    z_state = float(tr.state[2])
                    z = 800.0 if abs(z_state) <= 1.0 else max(200.0, abs(z_state))
                    age = time.monotonic() - tr.last_seen_t
                    if tr.nn_out:
                        sx = float(tr.state[0]) * 520.0 / z * 0.02
                        sy = float(tr.state[1]) * 520.0 / z * 0.02
                    else:
                        sx = float(tr.state[3]) * age * 520.0 / z
                        sy = float(tr.state[4]) * age * 520.0 / z
                    if np.isfinite(sx) and np.isfinite(sy):
                        x = int(np.clip(float(x) + sx, 0, max(0, frame_w - w)))
                        y = int(np.clip(float(y) + sy, 0, max(0, frame_h - h)))
                        w = max(28, int(w))
                        h = max(28, int(h))
            p_base = tr.state[:3].copy() if tr.has_base_frame else None
            r_mm, ang_deg = _polar_from_origin(p_base)
            if r_mm is None:
                r_mm = tr.last_r_mm
            if ang_deg is None:
                ang_deg = tr.last_ang_deg
            views.append(
                TrackView(
                    track_id=tr.track_id,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    d_mm=d,
                    coasting=coasting,
                    is_base=bool(tr.is_base or tr.track_id == 0),
                    p_base=p_base,
                    p_cam=None if tr.p_cam is None else np.asarray(tr.p_cam, dtype=np.float64).reshape(3),
                    r_mm=r_mm,
                    ang_deg=ang_deg,
                    nn_out=bool(tr.nn_out),
                )
            )
        views.sort(key=lambda v: v.track_id)
        return views



def views_to_prox_boxes(
    views: list[TrackView],
    *,
    include_base: bool = False,
) -> list[tuple[int, int, int, int, Optional[float]]]:
    """One box per track id for proximity. Coasting ghosts are skipped.

    The pedestal cap is included when checking the gray equipment mask.
    """
    seen: set[int] = set()
    out = []
    for v in views:
        if v.coasting or v.track_id in seen:
            continue
        if v.is_base and not include_base:
            continue
        seen.add(v.track_id)
        out.append((v.x, v.y, v.w, v.h, v.d_mm))
    return out


def draw_tracked_blue(
    bgr: np.ndarray,
    views: list[TrackView],
    *,
    show_mask: bool = True,
    gray_mask: Optional[np.ndarray] = None,
    depth_mm: Optional[np.ndarray] = None,
    hud_live: int = 0,
    hud_coast: int = 0,
    hud_cap: int = 6,
    tape_box: Optional[tuple[int, int, int, int]] = None,
    show_origin: bool = True,
) -> np.ndarray:
    """Draw J0..J5 with strong highlights — live and predicted / out-of-view."""
    out = bgr.copy()
    mask = baby_blue_mask(bgr)
    # Live: tint real blue pixels inside boxes.
    live_views = [v for v in views if not v.coasting]
    if show_mask and live_views:
        keep = np.zeros_like(mask)
        for v in live_views:
            y0, y1 = max(0, v.y), min(mask.shape[0], v.y + v.h)
            x0, x1 = max(0, v.x), min(mask.shape[1], v.x + v.w)
            if y1 > y0 and x1 > x0:
                keep[y0:y1, x0:x1] = cv2.bitwise_and(mask[y0:y1, x0:x1], 255)
        tint = np.zeros_like(out)
        tint[:] = MASK_TINT
        overlay = cv2.addWeighted(out, 1.0, tint, 0.40, 0)
        out = np.where(keep[..., None] > 0, overlay, out)

    if show_mask and gray_mask is not None and gray_mask.shape[:2] == out.shape[:2]:
        tint = np.zeros_like(out)
        tint[:] = (170, 170, 170)
        overlay = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
        out = np.where(gray_mask[..., None] > 0, overlay, out)

    # Draw spokes from purple tape (0,0) → each joint so relative layout is clear.
    tape_c: Optional[tuple[int, int]] = None
    if show_origin and tape_box is not None:
        tx, ty, tw, th = tape_box
        tape_c = (int(tx + 0.5 * tw), int(ty + 0.5 * th))
        cv2.circle(out, tape_c, 6, (180, 40, 180), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            "0,0",
            (tape_c[0] + 8, max(16, tape_c[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 40, 180),
            2,
            cv2.LINE_AA,
        )

    # Coasting tracks stay in the bank for association, but are not drawn
    # (dashed boxes were landing off the arm).
    for v in views:
        if v.coasting:
            continue
        d_mm = v.d_mm
        if depth_mm is not None:
            d_mm = median_depth_mm(
                depth_mm,
                v.x,
                v.y,
                v.w,
                v.h,
                mask=mask,
                color_hw=bgr.shape[:2],
            )
        if v.nn_out:
            color = (0, 200, 255)
            thickness = 3
        else:
            color = BOX_COLOR
            thickness = 2
        jcx, jcy = int(v.x + 0.5 * v.w), int(v.y + 0.5 * v.h)
        if tape_c is not None and not (v.is_base or v.track_id == 0):
            cv2.line(out, tape_c, (jcx, jcy), (200, 120, 200), 2, cv2.LINE_AA)
        cv2.rectangle(out, (v.x, v.y), (v.x + v.w, v.y + v.h), color, thickness)
        cv2.circle(out, (jcx, jcy), 4, color, -1, cv2.LINE_AA)
        if v.is_base or v.track_id == 0:
            tag = "blue"
        else:
            tag = f"J{v.track_id}"
            if v.r_mm is not None:
                tag += f" r={v.r_mm / 10.0:.0f}cm"
            if v.ang_deg is not None:
                tag += f" @{v.ang_deg:.0f}deg"
        tag += f" {format_distance_mm(d_mm)}"
        cv2.putText(
            out,
            tag,
            (v.x, max(16, v.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"BLUE TRACKS {hud_live}/{hud_cap} live + {hud_coast} predicted",
        (12, out.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 80),
        2,
        cv2.LINE_AA,
    )
    if show_origin:
        cv2.putText(
            out,
            "ORIGIN purple (0,0)  |  NN predicts joints out of view",
            (12, out.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 230, 255),
            1,
            cv2.LINE_AA,
        )
    if show_origin and tape_box is not None:
        tx, ty, tw, th = tape_box
        cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), (180, 0, 180), 2)
        cv2.putText(
            out,
            "PURPLE (0,0)",
            (tx, max(16, ty - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 0, 180),
            2,
            cv2.LINE_AA,
        )
    return out
