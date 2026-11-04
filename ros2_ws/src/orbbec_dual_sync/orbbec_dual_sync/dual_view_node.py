"""Live side-by-side preview: cam0 furnace CV | cam1 rack OPEN/CLOSED CV.

Subscribes only — does not open CAP_OBSENSOR. Furnace overlay on the left pane
(same helpers as furnace_agents / orbbec_furnace_overlay). Rack overlay on the
right pane (same as orbbec_rack_overlay).

Reading map (this file is the viewer hub):
  - __init__ / subscriptions: ROS params and topics
  - _paint_cam*: per-camera overlays (furnace, rack, blues, tags, UR FK)
  - _overlay_ur_fk / online pose: stick figure + OnlineCamPose
  - _on_timer: compose the three panes + depth row and show the preview

Related modules: online_cam_pose.py, ur_skeleton_draw.py, blue_joint_tracks.py,
baby_blue_draw.py, cell_frames.py, camera_publisher.py.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image

from .camera_path import ensure_camera_on_path
from .furnace_overlay_draw import (
    default_furnace_json,
    paint_furnace_overlay,
)
from .image_codec import imgmsg_to_numpy
from .preview_gui import PreviewWindow
from .rack_overlay_draw import (
    classify_whole_rack,
    draw_rack_overlay,
    load_rack_rois,
)


def _label(bgr: np.ndarray, text: str) -> np.ndarray:
    out = bgr.copy()
    cv2.putText(
        out,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _placeholder(lines: list[str], h: int = 480, w: int = 640) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y0 = h // 2 - 12 * (len(lines) - 1)
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (24, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def compose_side_by_side(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
    *,
    max_height: int = 480,
    left_status: str = "",
    right_status: str = "",
    left_already_labeled: bool = False,
    right_already_labeled: bool = False,
) -> np.ndarray:
    if left is not None:
        a = left if left_already_labeled else _label(left, "cam0")
    else:
        a = _placeholder(["waiting cam0…", left_status] if left_status else ["waiting cam0…"])
    if right is not None:
        b = right if right_already_labeled else _label(right, "cam1")
    else:
        b = _placeholder(
            ["waiting cam1…", right_status] if right_status else ["waiting cam1…"]
        )
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb, max_height) if max_height > 0 else min(ha, hb)
    a_s = cv2.resize(a, (max(1, int(wa * (h / float(ha)))), h))
    b_s = cv2.resize(b, (max(1, int(wb * (h / float(hb)))), h))
    return cv2.hconcat([a_s, b_s])


def _attach_pane(row: np.ndarray, pane: np.ndarray) -> np.ndarray:
    """Add one picture on the right, scaled to the row height."""
    h = row.shape[0]
    ph, pw = pane.shape[:2]
    scaled = cv2.resize(pane, (max(1, int(pw * (h / float(ph)))), h))
    return cv2.hconcat([row, scaled])


def _default_rack_json() -> Path:
    root = ensure_camera_on_path()
    return root / "rack_lid_state" / "rack.json"


class DualViewNode(Node):
    def __init__(self) -> None:
        super().__init__("orbbec_dual_view")
        self.declare_parameter("cam0_color_topic", "/orbbec/cam0/color/image_raw")
        self.declare_parameter("cam1_color_topic", "/orbbec/cam1/color/image_raw")
        self.declare_parameter("cam2_color_topic", "/orbbec/cam2/color/image_raw")
        self.declare_parameter("cam0_depth_topic", "/orbbec/cam0/depth/image_raw")
        self.declare_parameter("cam1_depth_topic", "/orbbec/cam1/depth/image_raw")
        self.declare_parameter("cam2_depth_topic", "/orbbec/cam2/depth/image_raw")
        self.declare_parameter("max_height", 480)
        self.declare_parameter("refresh_hz", 15.0)
        # Cam1 rack OPEN/CLOSED CV (same as orbbec_rack_overlay).
        self.declare_parameter("enable_rack_cv", True)
        self.declare_parameter("rack_json", "")
        self.declare_parameter("state_mode", "classify")
        self.declare_parameter("force_state", "CLOSED")
        self.declare_parameter("classify_every_n", 2)
        # Cam0 furnace CV (same helpers as furnace_agents / orbbec_furnace_overlay).
        self.declare_parameter("enable_furnace_cv", True)
        self.declare_parameter("furnace_json", "")
        self.declare_parameter("furnace_match_live", True)
        self.declare_parameter("furnace_match_every_n", 2)
        # YOLO arm detector (arm_state/runs/arm_detect/weights/best.pt)
        self.declare_parameter("enable_arm_cv", True)
        self.declare_parameter("arm_weights", "")
        self.declare_parameter("arm_conf", 0.25)
        self.declare_parameter("arm_every_n", 3)
        # Pale baby-blue color mask tracker (no NN) — UR joint plastic.
        self.declare_parameter("enable_blue_cv", True)
        self.declare_parameter("blue_show_mask", True)
        self.declare_parameter("blue_min_area", 280)
        self.declare_parameter("blue_every_n", 1)
        self.declare_parameter("blue_proximity_mm", 100.0)  # 10 cm red alert
        # 6-slot JointBank + Kalman coast (HSV detector unchanged).
        self.declare_parameter("blue_num_joints", 6)
        self.declare_parameter("blue_coast_s", 10.0)
        self.declare_parameter("blue_merge_mm", 80.0)
        self.declare_parameter("blue_match_mm", 120.0)
        # Teach-pendant / UR FK stick figure + vision confirm + FK rack keep-out.
        self.declare_parameter("enable_ur_fk", True)
        self.declare_parameter("ur_fk_draw", True)
        self.declare_parameter("ur_fk_confirm", True)
        self.declare_parameter("ur_fk_rack_alert", True)
        # Online per-camera pose from arm tags + blue caps over many frames.
        # Legacy tag-fitting stays available behind this flag (default off).
        self.declare_parameter("use_legacy_tag_fit", False)
        self.declare_parameter("online_use_blue", True)
        self.declare_parameter("online_use_blue_in_pnp", False)
        self.declare_parameter("online_save_cell", False)
        self.declare_parameter("online_ignore_saved_pose", True)
        self.declare_parameter("online_reset", False)
        self.declare_parameter("cam0_info_topic", "/orbbec/cam0/color/camera_info")
        self.declare_parameter("cam1_info_topic", "/orbbec/cam1/color/camera_info")
        self.declare_parameter("cam2_info_topic", "/orbbec/cam2/color/camera_info")

        t0 = str(self.get_parameter("cam0_color_topic").value)
        t1 = str(self.get_parameter("cam1_color_topic").value)
        t2 = str(self.get_parameter("cam2_color_topic").value)
        self.cam0_depth_topic = str(self.get_parameter("cam0_depth_topic").value)
        self.cam1_depth_topic = str(self.get_parameter("cam1_depth_topic").value)
        self.cam2_depth_topic = str(self.get_parameter("cam2_depth_topic").value)
        self.max_height = int(self.get_parameter("max_height").value)
        hz = float(self.get_parameter("refresh_hz").value)
        period = 1.0 / max(hz, 1.0)

        self.enable_rack_cv = bool(self.get_parameter("enable_rack_cv").value)
        self.state_mode = str(self.get_parameter("state_mode").value).strip().lower()
        self.force_state = str(self.get_parameter("force_state").value).strip().upper()
        self.classify_every_n = max(1, int(self.get_parameter("classify_every_n").value))

        self.enable_furnace_cv = bool(self.get_parameter("enable_furnace_cv").value)
        self.furnace_match_live = bool(self.get_parameter("furnace_match_live").value)
        self.furnace_match_every_n = max(
            1, int(self.get_parameter("furnace_match_every_n").value)
        )
        self.furnace_json: Optional[Path] = None

        self.enable_arm_cv = bool(self.get_parameter("enable_arm_cv").value)
        arm_w = str(self.get_parameter("arm_weights").value).strip()
        self.arm_weights: Optional[Path] = Path(arm_w) if arm_w else None
        self.arm_conf = float(self.get_parameter("arm_conf").value)
        self.arm_every_n = max(1, int(self.get_parameter("arm_every_n").value))
        self._arm_cache0: Optional[np.ndarray] = None
        self._arm_cache1: Optional[np.ndarray] = None

        self.enable_blue_cv = bool(self.get_parameter("enable_blue_cv").value)
        self.blue_show_mask = bool(self.get_parameter("blue_show_mask").value)
        self.blue_min_area = max(20, int(self.get_parameter("blue_min_area").value))
        self.blue_every_n = max(1, int(self.get_parameter("blue_every_n").value))
        self.blue_proximity_mm = max(
            10.0, float(self.get_parameter("blue_proximity_mm").value)
        )
        self.blue_num_joints = max(
            1, int(self.get_parameter("blue_num_joints").value)
        )
        self.blue_coast_s = max(0.1, float(self.get_parameter("blue_coast_s").value))
        self.blue_merge_mm = max(
            10.0, float(self.get_parameter("blue_merge_mm").value)
        )
        self.blue_match_mm = max(
            20.0, float(self.get_parameter("blue_match_mm").value)
        )
        self.enable_ur_fk = bool(self.get_parameter("enable_ur_fk").value)
        self.ur_fk_draw = bool(self.get_parameter("ur_fk_draw").value)
        self.ur_fk_confirm = bool(self.get_parameter("ur_fk_confirm").value)
        self.ur_fk_rack_alert = bool(self.get_parameter("ur_fk_rack_alert").value)
        self.use_legacy_tag_fit = bool(self.get_parameter("use_legacy_tag_fit").value)
        self.online_use_blue = bool(self.get_parameter("online_use_blue").value)
        self.online_use_blue_in_pnp = bool(
            self.get_parameter("online_use_blue_in_pnp").value
        )
        self.online_save_cell = bool(self.get_parameter("online_save_cell").value)
        self.online_ignore_saved_pose = bool(
            self.get_parameter("online_ignore_saved_pose").value
        )
        # Raw-image intrinsics per camera, filled from camera_info.
        self._cam_info: dict = {}
        # One OnlineCamPose per camera (built lazily once intrinsics arrive).
        self._online_pose: dict = {}
        self._last_q = None
        self._last_q_t = 0.0
        self._arm_speed = None
        self._ur_state = None
        self._ur_state_t = 0.0
        self._ur_fit = None
        self._ur_fit_t = 0.0
        self._last_fk_alert_log = 0.0
        self._blue_boxes0 = None
        self._blue_boxes1 = None
        self._blue_alert0 = False
        self._blue_alert1 = False
        self._last_retreat_log0 = 0.0
        self._last_retreat_log1 = 0.0
        self._last_fuse_log = 0.0
        self._fuse0: Optional[dict] = None
        self._fuse1: Optional[dict] = None
        self._joint_bank = None
        self._cell_frames = None
        self._furnace_slots: list[tuple[int, int, int, int, int]] = []
        self._latest_depth0: Optional[np.ndarray] = None
        self._latest_depth1: Optional[np.ndarray] = None

        self._latest0: Optional[np.ndarray] = None
        self._latest0_cv: Optional[np.ndarray] = None
        self._latest1: Optional[np.ndarray] = None
        self._latest1_cv: Optional[np.ndarray] = None
        self._latest2: Optional[np.ndarray] = None
        self._latest2_cv: Optional[np.ndarray] = None
        self._latest_depth2: Optional[np.ndarray] = None
        self._recv0 = 0
        self._recv1 = 0
        self._recv2 = 0
        self._cam0_seq = 0
        self._cam1_seq = 0
        self._last_state = "CLOSED"
        self._started = time.monotonic()
        self._warned_empty = False
        self._preview_error_logged = False
        self._should_exit = False
        self._robotics: Optional[Path] = None
        self.classify_xywh = (226, 55, 137, 152)
        self.display_xywh = (186, 30, 220, 260)
        self.holes: list = []

        need_robotics = (
            self.enable_rack_cv
            or self.enable_furnace_cv
            or self.enable_arm_cv
            or self.enable_blue_cv
            or self.enable_ur_fk
        )
        if need_robotics:
            try:
                self._robotics = ensure_camera_on_path()
            except Exception as exc:
                self.get_logger().error(f"Could not resolve robotics root: {exc}")
                self._robotics = None

        if self.enable_arm_cv:
            if self._robotics is None:
                self.enable_arm_cv = False
                self.get_logger().error("Arm CV disabled — robotics root unavailable")
            else:
                w = self.arm_weights or (
                    self._robotics
                    / "arm_state"
                    / "runs"
                    / "arm_detect"
                    / "weights"
                    / "best.pt"
                )
                if not w.is_file():
                    self.enable_arm_cv = False
                    self.get_logger().error(f"Arm CV disabled — missing weights {w}")
                else:
                    self.arm_weights = w
                    self.get_logger().info(
                        f"Arm YOLO ON both panes: conf={self.arm_conf} "
                        f"every_n={self.arm_every_n} weights={w}"
                    )

        if self.enable_blue_cv:
            self.get_logger().info(
                f"Baby-blue color mask ON both panes: min_area={self.blue_min_area} "
                f"show_mask={self.blue_show_mask}; RED alert if blue < "
                f"{self.blue_proximity_mm:.0f}mm of black/silver equipment "
                f"(mask hidden; 3D/depth; base joint ignored); "
                f"JointBank n={self.blue_num_joints} coast={self.blue_coast_s:.1f}s "
                f"merge={self.blue_merge_mm:.0f}mm match={self.blue_match_mm:.0f}mm"
            )
            try:
                from .cell_frames import (
                    LAYOUT_PERPENDICULAR,
                    apply_perpendicular_layout,
                    default_cell_frames_path,
                    load_cell_frames,
                )

                cpath = default_cell_frames_path(self._robotics)
                self._cell_frames = load_cell_frames(cpath)

                if self._cell_frames.layout == LAYOUT_PERPENDICULAR:
                    apply_perpendicular_layout(self._cell_frames)
                self.get_logger().info(
                    f"ur_base cell_frames: {cpath} "
                    f"layout={self._cell_frames.layout} "
                    f"bootstrap={self._cell_frames.bootstrap_from_blue_base} "
                    f"cam0_ready={self._cell_frames.cam0.ready} "
                    f"cam1_ready={self._cell_frames.cam1.ready} "
                    f"(cam0⊥cam1 90deg — pedestal blue = origin)"
                )
            except Exception as exc:
                self._cell_frames = None
                self.get_logger().warn(f"cell_frames not loaded: {exc}")
            try:
                from .blue_joint_tracks import JointBank

                self._joint_bank = JointBank(
                    num_joints=self.blue_num_joints,
                    coast_s=self.blue_coast_s,
                    merge_mm=self.blue_merge_mm,
                    match_mm=self.blue_match_mm,
                )
            except Exception as exc:
                self._joint_bank = None
                self.get_logger().warn(f"JointBank init failed: {exc}")

        if self.enable_ur_fk:
            self.get_logger().info(
                "UR FK stick-figure ON: write robotics/ur_joints.json from teach "
                "pendant (q_rad[6]) or set ur_fk_config.json host for RTDE; "
                f"draw={self.ur_fk_draw} confirm={self.ur_fk_confirm} "
                f"rack_alert={self.ur_fk_rack_alert}"
            )

        if self.enable_rack_cv:
            try:
                if self._robotics is None:
                    raise FileNotFoundError("robotics root unavailable")
                rack_raw = str(self.get_parameter("rack_json").value).strip()
                rack_path = Path(rack_raw) if rack_raw else _default_rack_json()
                if not rack_path.is_file():
                    raise FileNotFoundError(f"rack.json not found: {rack_path}")
                self.classify_xywh, self.display_xywh, self.holes = load_rack_rois(
                    rack_path
                )
                self.get_logger().info(
                    f"Rack CV ON cam1 pane: mode={self.state_mode} "
                    f"classify={self.classify_xywh} display={self.display_xywh} "
                    f"json={rack_path}"
                )
            except Exception as exc:
                self.enable_rack_cv = False
                self.get_logger().error(
                    f"Rack CV disabled (could not load ROI/templates): {exc}"
                )

        if self.enable_furnace_cv:
            try:
                if self._robotics is None:
                    raise FileNotFoundError("robotics root unavailable")
                furnace_raw = str(self.get_parameter("furnace_json").value).strip()
                self.furnace_json = (
                    Path(furnace_raw) if furnace_raw else default_furnace_json(self._robotics)
                )
                if not self.furnace_json.is_file():
                    raise FileNotFoundError(
                        f"furnace.json not found: {self.furnace_json}"
                    )
                self.get_logger().info(
                    f"Furnace CV ON cam0 pane: match_live={self.furnace_match_live} "
                    f"json={self.furnace_json}"
                )
            except Exception as exc:
                self.enable_furnace_cv = False
                self.furnace_json = None
                self.get_logger().error(
                    f"Furnace CV disabled (could not load furnace.json): {exc}"
                )

        title_bits = ["cam0"]
        if self.enable_furnace_cv:
            title_bits[0] = "cam0 + furnace CV"
        title_bits.append("cam1 + rack CV" if self.enable_rack_cv else "cam1")
        self._preview = PreviewWindow(
            f"orbbec_dual_view ({' | '.join(title_bits)})",
            max_height=max(self.max_height, 480),
        )
        extras = []
        if self.enable_furnace_cv:
            extras.append("furnace boxes on cam0")
        if self.enable_rack_cv:
            extras.append("rack OPEN/CLOSED on cam1")
        if self.enable_arm_cv:
            extras.append("arm YOLO on both panes")
        if self.enable_blue_cv:
            extras.append(
                f"baby-blue vs black/silver equipment ({self.blue_proximity_mm:.0f}mm 3D alert)"
            )

        extra = (" + " + "; ".join(extras) + ".") if extras else "."
        self.get_logger().info(
            f"Dual view: {t0} | {t1} | {t2}{extra} Q / Esc / close exits. R resets online pose."
        )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, t0, self._on_cam0, image_qos)
        self.create_subscription(Image, t1, self._on_cam1, image_qos)
        self.create_subscription(Image, t2, self._on_cam2, image_qos)
        self.create_subscription(
            Image, self.cam0_depth_topic, self._on_depth0, image_qos
        )
        self.create_subscription(
            Image, self.cam1_depth_topic, self._on_depth1, image_qos
        )
        self.create_subscription(
            Image, self.cam2_depth_topic, self._on_depth2, image_qos
        )
        # Real intrinsics for the online pose / projection.
        info_topics = {
            "cam0": str(self.get_parameter("cam0_info_topic").value),
            "cam1": str(self.get_parameter("cam1_info_topic").value),
            "cam2": str(self.get_parameter("cam2_info_topic").value),
        }
        for cam, topic in info_topics.items():
            if topic:
                self.create_subscription(
                    CameraInfo,
                    topic,
                    (lambda m, c=cam: self._on_cam_info(c, m)),
                    image_qos,
                )
        self.create_timer(period, self._on_timer)

    def _on_cam_info(self, cam: str, msg: CameraInfo) -> None:
        """Store this camera's raw-image intrinsics (fx, fy, cx, cy, distortion)."""
        try:
            k = list(msg.k)
            if len(k) < 9 or float(k[0]) <= 0.0:
                return
            dist = [float(v) for v in (msg.d or [])]
            self._cam_info[cam] = {
                "fx": float(k[0]),
                "fy": float(k[4]),
                "cx": float(k[2]),
                "cy": float(k[5]),
                "dist": dist,
                "width": int(msg.width),
                "height": int(msg.height),
            }
        except Exception as exc:
            self.get_logger().warn(f"{cam} camera_info decode failed: {exc}")

    def _color_K(self, cam: str, color_hw):
        """3x3 color-camera K for this frame size, or None until camera_info."""
        intr = self._scaled_intrinsics(cam, color_hw, color_hw)
        if intr is None:
            return None
        return np.array(
            [
                [intr["fx"], 0.0, intr["cx"]],
                [0.0, intr["fy"], intr["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _scaled_intrinsics(self, cam: str, out_hw, raw_hw):
        """Raw camera_info intrinsics scaled to the drawn image. None if unknown."""
        info = self._cam_info.get(cam)
        if not info:
            return None
        raw_w = float(raw_hw[1]) if raw_hw else float(info["width"] or 0.0)
        raw_h = float(raw_hw[0]) if raw_hw else float(info["height"] or 0.0)
        if raw_w <= 0.0 or raw_h <= 0.0:
            return None
        sx = float(out_hw[1]) / raw_w
        sy = float(out_hw[0]) / raw_h
        return {
            "fx": info["fx"] * sx,
            "fy": info["fy"] * sy,
            "cx": info["cx"] * sx,
            "cy": info["cy"] * sy,
            "dist": info["dist"],
        }

    def _cam_info_known(self, cam: str) -> bool:
        """True when this camera has a camera_info entry and a non-zero K."""
        info = self._cam_info.get(cam)
        if not info:
            return False
        try:
            fx = float(info.get("fx") or 0.0)
            fy = float(info.get("fy") or 0.0)
        except (TypeError, ValueError):
            return False
        return fx > 0.0 and fy > 0.0

    def _warn_missing_cam_info(self, cam: str) -> None:
        warned = getattr(self, "_cam_info_warned", None)
        if warned is None:
            warned = set()
            self._cam_info_warned = warned
        if cam in warned:
            return
        warned.add(cam)
        self.get_logger().warn(f"online {cam}: NO camera_info (fx unknown)")

    def _draw_no_cam_info_hud(self, out: np.ndarray, cam: str) -> np.ndarray:
        cv2.putText(
            out,
            f"online {cam}: NO camera_info (fx unknown)",
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def _get_online_pose(self, cam: str, intr: dict):
        """Lazily build / refresh the OnlineCamPose for this camera."""
        import numpy as _np

        from .online_cam_pose import OnlineCamPose

        K = _np.array(
            [
                [intr["fx"], 0.0, intr["cx"]],
                [0.0, intr["fy"], intr["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=_np.float64,
        )
        dist = _np.asarray(intr["dist"], dtype=_np.float64) if intr["dist"] else None
        op = self._online_pose.get(cam)
        if op is None:
            R0 = t0 = None
            if (not self.online_ignore_saved_pose) and self._cell_frames is not None:
                p0 = self._cell_frames.pose(cam)
                if getattr(p0, "ready", False):
                    R0, t0 = p0.R, p0.t_mm
            op = OnlineCamPose(
                K,
                dist,
                R0,
                t0,
                cam=cam,
                use_blue=self.online_use_blue,
                use_blue_in_pnp=self.online_use_blue_in_pnp,
                assign_px=25.0,
            )
            self._online_pose[cam] = op
        else:
            op.set_intrinsics(K, dist)
        return op

    def _resolve_state(self, bgr: np.ndarray) -> str:
        if self.state_mode == "force_open":
            return "OPEN"
        if self.state_mode == "force_closed":
            return "CLOSED"
        if self.state_mode in ("force", "fixed"):
            return self.force_state or "CLOSED"
        try:
            return classify_whole_rack(
                bgr,
                self.classify_xywh,
                self.holes,
                robotics_root=self._robotics,
            )
        except Exception as exc:
            self.get_logger().warn(f"classify failed, using last={self._last_state}: {exc}")
            return self._last_state

    def _depth_map_bgr(self, depth, label: str) -> np.ndarray:
        """Color depth map. Red is near, blue is far, missing depth is black."""
        if depth is None:
            return _placeholder([label, "waiting for depth"])
        z = np.asarray(depth, dtype=np.float32)
        if z.ndim != 2 or z.size == 0:
            return _placeholder([label, "no depth"])
        near_mm, far_mm = 300.0, 2000.0
        valid = (z > 80.0) & (z < 4000.0)
        span = (np.clip(z, near_mm, far_mm) - near_mm) / (far_mm - near_mm)
        idx = np.clip((1.0 - span) * 255.0, 0, 255).astype(np.uint8)
        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        out = cv2.applyColorMap(idx, cmap)
        out[~valid] = 0
        h, w = out.shape[:2]
        cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        bar_x, bar_y0, bar_h = 8, 36, min(120, max(40, h - 48))
        for i in range(bar_h):
            dist = near_mm + (far_mm - near_mm) * (i / float(max(1, bar_h - 1)))
            tone = int(round((1.0 - (dist - near_mm) / (far_mm - near_mm)) * 255.0))
            color = cv2.applyColorMap(np.array([[tone]], dtype=np.uint8), cmap)[0, 0]
            cv2.line(
                out,
                (bar_x, bar_y0 + i),
                (bar_x + 12, bar_y0 + i),
                (int(color[0]), int(color[1]), int(color[2])),
                1,
            )
        for cm, frac in ((30, 0.0), (100, 0.41), (200, 1.0)):
            y = bar_y0 + int(round(frac * (bar_h - 1)))
            cv2.putText(
                out,
                f"{cm}cm",
                (bar_x + 16, min(h - 4, y + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cy, cx = h // 2, w // 2
        patch = z[max(0, cy - 2) : cy + 3, max(0, cx - 2) : cx + 3]
        good = patch[(patch > 80.0) & (patch < 4000.0)]
        cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
        if good.size:
            cv2.putText(
                out,
                f"center {float(np.median(good)) / 10.0:.0f}cm",
                (cx + 10, max(18, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return out

    def _paint_cam0_cv(self, bgr: np.ndarray) -> np.ndarray:
        self._cam0_seq += 1
        if self.enable_furnace_cv:
            do_match = self.furnace_match_live and (
                self._cam0_seq % self.furnace_match_every_n == 0
                or self._latest0_cv is None
            )
            try:
                from .furnace_overlay_draw import (
                    draw_furnace_overlay_on_frame,
                    resolve_slots_for_frame,
                )

                slots = resolve_slots_for_frame(
                    bgr,
                    robotics_root=self._robotics,
                    furnace_json=self.furnace_json,
                    match_live=do_match,
                )
                self._furnace_slots = [
                    (int(s.id), int(s.x), int(s.y), int(s.w), int(s.h)) for s in slots
                ]
                view = draw_furnace_overlay_on_frame(
                    bgr,
                    slots,
                    robotics_root=self._robotics,
                    corner_tag="cam0 + furnace + arm",
                )
            except Exception as exc:
                self.get_logger().warn(f"furnace resolve failed: {exc}")
                self._furnace_slots = []
                view = paint_furnace_overlay(
                    bgr,
                    robotics_root=self._robotics,
                    furnace_json=self.furnace_json,
                    match_live=do_match,
                    corner_tag="cam0 + furnace + arm",
                )
        else:
            self._furnace_slots = []
            view = _label(bgr, "cam0")
        view = self._overlay_arm_on(view, bgr, cam="cam0")
        return self._overlay_blue_on(view, bgr, cam="cam0")

    def _overlay_blue_on(
        self, decorated: np.ndarray, raw: np.ndarray, *, cam: str
    ) -> np.ndarray:
        """HSV baby-blue → JointBank tracks (J0..J5) with Kalman coast."""
        alert_attr = "_blue_alert0" if cam == "cam0" else "_blue_alert1"
        if not self.enable_blue_cv:
            setattr(self, alert_attr, False)
            return self._draw_fk_monitor(decorated, cam)
        seq = self._cam0_seq if cam == "cam0" else self._cam1_seq
        cache_attr = "_blue_boxes0" if cam == "cam0" else "_blue_boxes1"
        depth = self._latest_depth0 if cam == "cam0" else self._latest_depth1
        if seq % self.blue_every_n == 0 or getattr(self, cache_attr) is None:
            try:
                from .baby_blue_draw import find_baby_blue_boxes

                setattr(
                    self,
                    cache_attr,
                    find_baby_blue_boxes(
                        raw, min_area=self.blue_min_area, depth_mm=depth
                    ),
                )
            except Exception as exc:
                self.get_logger().warn(f"blue mask {cam}: {exc}")
                setattr(self, cache_attr, [])
        try:
            from .baby_blue_draw import (
                baby_blue_mask,
                draw_proximity_error,
                gray_equipment_mask,
                proximity_alerts,
                retreat_guidance_toward_base,
            )
            from .blue_joint_tracks import (
                draw_tracked_blue,
                views_to_prox_boxes,
            )

            boxes = getattr(self, cache_attr) or []
            rh, rw = raw.shape[:2]
            mask = baby_blue_mask(raw)
            gray = gray_equipment_mask(raw, mask)
            pose = None
            fx = 520.0
            if self._cell_frames is not None:
                pose = self._cell_frames.pose(cam)
                fx = float(self._cell_frames.fx_px)
            cam_K = self._color_K(cam, (rh, rw))

            # Boxes stay in color pixels. Depth is the native Orbbec frame
            # (often a different size); ingest maps color→depth when it samples Z.
            views = []
            if self._joint_bank is not None:
                try:
                    views = self._joint_bank.ingest_cam(
                    cam,
                    boxes,
                    depth,
                    mask,
                    pose=pose,
                    fx=fx,
                    frame_h=rh,
                    frame_w=rw,
                    bgr=raw,
                    cam_K=cam_K,
                    color_hw=(rh, rw),
                    )
                except Exception as exc:
                    self.get_logger().warn(f"blue bank {cam}: {exc}")
                    views = []
                # Pedestal blue = ur_base (0,0,0) for both cameras.
                try:
                    self._bootstrap_origin_from_j0(cam)
                except Exception:
                    pass
                live, coast, cap = self._joint_bank.hud_counts()
            else:
                live = coast = 0
                cap = self.blue_num_joints
                from .blue_joint_tracks import TrackView

                views = [
                    TrackView(
                        track_id=i,
                        x=b[0],
                        y=b[1],
                        w=b[2],
                        h=b[3],
                        d_mm=b[4],
                        coasting=False,
                        is_base=False,
                        p_base=None,
                    )
                    for i, b in enumerate(boxes)
                ]

            try:
                out = draw_tracked_blue(
                    decorated,
                    views,
                    show_mask=self.blue_show_mask,
                    gray_mask=None,
                    depth_mm=depth,
                    hud_live=live,
                    hud_coast=coast,
                    hud_cap=cap,
                    tape_box=(
                        self._joint_bank._tape_box.get(cam)
                        if self._joint_bank is not None and self.use_legacy_tag_fit
                        else None
                    ),
                    show_origin=self.use_legacy_tag_fit,
                )
            except Exception as exc:
                self.get_logger().warn(f"blue paint {cam}: {exc}")
                out = decorated

            alerts = []
            guide = None
            prox_boxes = views_to_prox_boxes(views, include_base=True)
            if depth is not None and prox_boxes:
                from .baby_blue_draw import (
                    depth_frame_intrinsics,
                    scale_color_box_to_depth,
                )

                # Proximity walks a pixel mask, so color/mask must sit on the
                # same grid as depth. That resize is for the mask only — it is
                # not D2C alignment. Boxes are remapped with the same scale.
                dh, dw = depth.shape[:2]
                if mask.shape[:2] != (dh, dw):
                    mask_d = cv2.resize(
                        mask, (dw, dh), interpolation=cv2.INTER_NEAREST
                    )
                    raw_for_prox = cv2.resize(
                        raw, (dw, dh), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    mask_d = mask
                    raw_for_prox = raw
                if (dh, dw) != (rh, rw) and rh > 0 and rw > 0:
                    boxes_d = [
                        (*scale_color_box_to_depth(x, y, w, h, (rh, rw), (dh, dw)), d)
                        for x, y, w, h, d in prox_boxes
                        if all(np.isfinite(float(v)) for v in (x, y, w, h))
                    ]
                else:
                    boxes_d = prox_boxes
                rack_for_prox = None
                if cam == "cam1" and self.classify_xywh:
                    rcx, rcy, rcw, rch = self.classify_xywh
                    if (dh, dw) != (rh, rw) and rh > 0 and rw > 0:
                        rack_for_prox = scale_color_box_to_depth(
                            rcx, rcy, rcw, rch, (rh, rw), (dh, dw)
                        )
                    else:
                        rack_for_prox = (int(rcx), int(rcy), int(rcw), int(rch))
                gray_for_prox = gray
                if gray.shape[:2] != (dh, dw):
                    gray_for_prox = cv2.resize(
                        gray, (dw, dh), interpolation=cv2.INTER_NEAREST
                    )
                if cam_K is not None:
                    fxd, fyd, cxd, cyd = depth_frame_intrinsics(
                        float(cam_K[0, 0]),
                        float(cam_K[1, 1]),
                        float(cam_K[0, 2]),
                        float(cam_K[1, 2]),
                        (rh, rw),
                        (dh, dw),
                    )
                else:
                    fxd, fyd, cxd, cyd = fx, fx, 0.5 * dw, 0.5 * dh
                alerts = proximity_alerts(
                    depth,
                    boxes_d,
                    mask_d,
                    bgr=raw_for_prox,
                    thresh_mm=self.blue_proximity_mm,
                    fx_px=fxd,
                    fy_px=fyd,
                    cx=cxd,
                    cy=cyd,
                    skip_base=True,
                    rack_xywh=rack_for_prox,
                    rack_height_mm=400.0,
                    gray_mask=gray_for_prox,
                )
                if alerts:
                    try:
                        guide = retreat_guidance_toward_base(
                            depth, boxes_d, mask_d, alerts
                        )
                    except Exception:
                        guide = None
                try:
                    self._stash_fuse_points(cam, depth, boxes_d, mask_d, alerts)
                except Exception:
                    pass
                if alerts and (dh, dw) != out.shape[:2]:
                    ox, oy = out.shape[1] / float(dw), out.shape[0] / float(dh)
                    alerts = [
                        (
                            int(a[0] * ox),
                            int(a[1] * oy),
                            max(1, int(a[2] * ox)),
                            max(1, int(a[3] * oy)),
                            a[4],
                            a[5] if len(a) > 5 else "",
                        )
                        for a in alerts
                        if all(np.isfinite(float(a[i])) for i in range(4))
                    ]

            # 40 mm tags: 0 base, 1 shoulder, 2 elbow, 3 wrist 1, 4 wrist 2, 5 wrist 3.
            # Detected on the raw frame so the kinematic line can sit on them.
            joint_tags: dict = {}
            try:
                from aruco_marker import (
                    find_joint_tags,
                    range_mm_from_pose,
                    range_mm_from_size,
                )

                joint_tags = find_joint_tags(raw, cam=cam, rois=boxes)
                for mid, hit in joint_tags.items():
                    z_pose = range_mm_from_pose(
                        hit["corners"], fx, raw.shape[1], raw.shape[0]
                    )
                    z_size = range_mm_from_size(float(hit["side_px"]), fx)
                    hit["z_mm"] = z_pose if z_pose is not None else z_size
            except Exception as exc:
                self.get_logger().warn(f"aruco {cam}: {exc}")
                joint_tags = {}

            # Teach-pendant / UR FK stick figure + confirm + out-of-view rack alert.
            # Drawn every frame, including when the arm is outside this camera.
            if self.enable_ur_fk:
                # Online mode has no focal length until camera_info arrives.
                # Do not seat the stick figure on cell_frames / fx_px / img_scale.
                if (not self.use_legacy_tag_fit) and not self._cam_info_known(cam):
                    self._warn_missing_cam_info(cam)
                    out = self._draw_no_cam_info_hud(out, cam)
                else:
                    if pose is None and self._cell_frames is not None:
                        pose = self._cell_frames.pose(cam)
                        fx = float(self._cell_frames.fx_px)
                    try:
                        out, fk_alerts, fk_guide = self._overlay_ur_fk(
                            out,
                            views,
                            pose=pose,
                            fx=fx,
                            cam=cam,
                            tape_bgr=raw,
                            joint_tags=joint_tags,
                        )
                        if fk_alerts:
                            alerts = list(alerts) + list(fk_alerts)
                            if fk_guide and not guide:
                                guide = fk_guide
                    except Exception as exc:
                        self.get_logger().warn(f"ur_fk {cam}: {exc}")

            # Tag quads sit on top of the kinematic line.
            try:
                from aruco_marker import draw_joint_tag

                sx = out.shape[1] / float(raw.shape[1])
                sy = out.shape[0] / float(raw.shape[0])
                for hit in joint_tags.values():
                    draw_joint_tag(
                        out,
                        hit,
                        z_mm=hit.get("z_mm"),
                        scale_uv=(sx, sy),
                    )
            except Exception as exc:
                self.get_logger().warn(f"aruco draw {cam}: {exc}")

            # Dedup pane alerts: only the tighter-gap cam raises fused flag if
            # both see the same track; still draw red on the alerting pane.
            setattr(self, alert_attr, bool(alerts))
            if alerts:
                out = draw_proximity_error(
                    out, alerts, thresh_cm=self.blue_proximity_mm / 10.0
                )
                if guide:
                    log_attr = (
                        "_last_retreat_log0"
                        if cam == "cam0"
                        else "_last_retreat_log1"
                    )
                    now = time.monotonic()
                    if now - float(getattr(self, log_attr, 0.0)) >= 1.0:
                        setattr(self, log_attr, now)
                        self.get_logger().error(f"[{cam}] {guide}")
                        print(f"\n*** [{cam}] {guide} ***\n", flush=True)
                    cv2.putText(
                        out,
                        "RETREAT -> BASE (see terminal)",
                        (18, out.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            return out
        except Exception as exc:
            self.get_logger().warn(f"blue draw {cam}: {exc}")
            setattr(self, alert_attr, False)
            return self._draw_fk_monitor(decorated, cam, tape_bgr=raw)

    def _draw_fk_monitor(self, out: np.ndarray, cam: str, tape_bgr=None) -> np.ndarray:
        """Keep the UR stick figure up even if the blue overlay did not run."""
        if not self.enable_ur_fk or not self.ur_fk_draw:
            return out
        if (not self.use_legacy_tag_fit) and not self._cam_info_known(cam):
            self._warn_missing_cam_info(cam)
            return self._draw_no_cam_info_hud(out, cam)
        pose = None
        fx = 520.0
        if self._cell_frames is not None:
            pose = self._cell_frames.pose(cam)
            fx = float(self._cell_frames.fx_px)
        try:
            painted, _alerts, _guide = self._overlay_ur_fk(
                out, [], pose=pose, fx=fx, cam=cam, tape_bgr=tape_bgr
            )
            return painted
        except Exception as exc:
            self.get_logger().warn(f"ur_fk {cam}: {exc}")
            return out

    def _refresh_ur_state(self):
        now = time.monotonic()
        if self._ur_state is not None and (now - self._ur_state_t) < 0.05:
            return self._ur_state
        from .ur_skeleton_draw import load_ur_state

        self._ur_state = load_ur_state(self._robotics)
        self._ur_state_t = now
        q = None
        if self._ur_state is not None:
            q = self._ur_state.get("q")
        self._update_arm_speed(q)
        return self._ur_state

    def _remember_tape_uv(self, cam: str | None, uv: np.ndarray) -> np.ndarray:
        hold = getattr(self, "_tape_uv_hold", None)
        if hold is None:
            hold = {}
            self._tape_uv_hold = hold
        if cam:
            hold[cam] = np.asarray(uv, dtype=np.float64).reshape(2).copy()
        return hold[cam] if cam else np.asarray(uv, dtype=np.float64).reshape(2)

    def _pedestal_uv(self, views: list, cam: str | None = None, bgr=None):
        """Purple-tape pixel. Never a blue cap, even when the tape flickers.

        ``views`` is ignored on purpose: joint 0 is not a blue detection.
        """
        del views  # blues do not place the base
        bank = getattr(self, "_joint_bank", None)
        if bank is not None:
            if bank.consume_origin_reset():
                self._origin_bootstrapped = set()
            locked = bank.locked_uv(cam)
            if locked is not None:
                return locked
        fh = int(bgr.shape[0]) if bgr is not None else 0
        if bank is not None and cam and fh > 0:
            box = getattr(bank, "_tape_box", {}).get(cam)
            if box is not None:
                x, y, w, h = box
                cy = y + 0.5 * h
                cx = x + 0.5 * w
                fw = int(bgr.shape[1]) if bgr is not None else 0
                cam0_ok = cam != "cam0" or (
                    fw > 0 and 0.15 * fw <= cx <= 0.70 * fw and cy >= 0.70 * fh
                )
                if cy >= 0.62 * fh and cam0_ok:
                    return self._remember_tape_uv(
                        cam, np.array([cx, cy], dtype=np.float64)
                    )
        if bgr is not None:
            try:
                from .blue_joint_tracks import low_purple_tape_uv

                found = low_purple_tape_uv(bgr, cam)
            except Exception:
                found = None
            if found is not None:
                if bank is not None and cam:
                    bank.lock_origin(cam, float(found[0]), float(found[1]))
                return self._remember_tape_uv(cam, found)
        hold = getattr(self, "_tape_uv_hold", {})
        if cam and cam in hold:
            return np.asarray(hold[cam], dtype=np.float64).reshape(2).copy()
        return None

    def _align_ur_to_blues(
        self,
        state: dict,
        views: list,
        *,
        pose=None,
        fx: float = 520.0,
        frame_hw: tuple[int, int] = (480, 640),
        cam: str | None = None,
        bgr=None,
    ) -> dict:
        """Kinematic chain is the arm graph; blue pixels check that camera view."""
        now = time.monotonic()
        fits = getattr(self, "_ur_fit_by_cam", None)
        if fits is None:
            fits = {}
            self._ur_fit_by_cam = fits
        cam_key = cam or "cam"
        cached = fits.get(cam_key)
        fit_key = (int(frame_hw[0]), int(frame_hw[1]), id(pose), cam_key)
        if (
            cached is not None
            and cached.get("_fit_key") == fit_key
            and (now - float(cached.get("_fit_t") or 0.0)) < 0.15
        ):
            return cached
        obs_live = []
        obs_coast = []
        pix_live = []
        pix_coast = []
        xyz_live = []
        xyz_coast = []
        seen: set[int] = set()
        for v in views or []:
            tid = int(getattr(v, "track_id", -1))
            if tid in seen:
                continue
            seen.add(tid)
            cx_b = float(getattr(v, "x", 0) + 0.5 * getattr(v, "w", 0))
            cy_b = float(getattr(v, "y", 0) + 0.5 * getattr(v, "h", 0))
            if not (np.isfinite(cx_b) and np.isfinite(cy_b)):
                continue
            if float(getattr(v, "w", 0)) * float(getattr(v, "h", 0)) < 1.0:
                continue
            uv = [cx_b, cy_b]
            pc = getattr(v, "p_cam", None)
            if pc is not None:
                pc = np.asarray(pc, dtype=np.float64).reshape(3)
                if not np.isfinite(pc).all():
                    pc = None
            pb = getattr(v, "p_base", None)
            # Zero-depth unprojects are not a joint. Skip them before the blend.
            if pb is not None:
                pb_arr = np.asarray(pb, dtype=np.float64).reshape(3)
                if (not np.isfinite(pb_arr).all()) or float(
                    pb_arr[0] ** 2 + pb_arr[1] ** 2 + pb_arr[2] ** 2
                ) < 1.0:
                    continue
            elif getattr(v, "coasting", False):
                continue
            if getattr(v, "coasting", False):
                pix_coast.append(uv)
                xyz_coast.append(pc)
                if pb is not None:
                    obs_coast.append(np.asarray(pb, dtype=np.float64).reshape(3))
            else:
                pix_live.append(uv)
                xyz_live.append(pc)
                if pb is not None:
                    obs_live.append(np.asarray(pb, dtype=np.float64).reshape(3))
        chosen = obs_live if len(obs_live) >= 2 else obs_live + obs_coast
        obs = (
            np.stack(chosen, axis=0)
            if chosen
            else np.zeros((0, 3), dtype=np.float64)
        )
        chosen_uv = pix_live if len(pix_live) >= 2 else pix_live + pix_coast
        base_uv = self._pedestal_uv(views, cam=cam, bgr=bgr)
        if base_uv is not None and chosen_uv:
            chosen_uv = [
                p
                for p in chosen_uv
                if abs(p[0] - float(base_uv[0])) > 3.0
                or abs(p[1] - float(base_uv[1])) > 3.0
            ]
        pixels = (
            np.asarray(chosen_uv, dtype=np.float64).reshape(-1, 2)
            if chosen_uv
            else None
        )
        fh, fw = int(frame_hw[0]), int(frame_hw[1])
        cx, cy = 0.5 * fw, 0.5 * fh

        def project_fn(pts_vis):
            from .ur_skeleton_draw import base_to_cam, project_cam_to_pixel

            out_uv = np.full((len(pts_vis), 2), np.nan, dtype=np.float64)
            if pose is None:
                return out_uv
            for i, p in enumerate(pts_vis):
                uv = project_cam_to_pixel(
                    base_to_cam(p, pose), float(fx), float(fx), cx, cy
                )
                if uv is not None:
                    out_uv[i, 0] = uv[0]
                    out_uv[i, 1] = uv[1]
            return out_uv

        prev = cached
        src = str(state.get("source", ""))
        if src.startswith("rtde"):
            q_prev = state["q"]
        elif prev is not None:
            q_prev = prev["q"]
        else:
            q_prev = None
        yaw_prev = 0.0 if prev is None else float(prev.get("yaw") or 0.0)
        from ur_skeleton import LIVE_VIEW_ALIGN

        view_align = LIVE_VIEW_ALIGN.get(cam_key)
        if view_align is not None:
            # Fixed heading for this camera. Blues must not spin a locked q.
            yaw_prev = float(view_align["yaw"])
        search = prev is None or (
            prev.get("check_ok") is not True
            and (now - float(getattr(self, "_ur_search_t", 0.0))) > 2.0
        )
        from .ur_skeleton_draw import align_state_to_blues

        aligned = align_state_to_blues(
            state,
            obs,
            q_prev=q_prev,
            yaw_prev=yaw_prev,
            search=search and view_align is None,
            pixels_uv=pixels if pose is not None else None,
            project_fn=project_fn if pose is not None else None,
            base_uv=base_uv,
            image_align=(view_align is None and cam_key == "cam0"),
            # ur5e_kinematics.ur_fk: seven points, flange is the TCP, no tool offset.
            with_tool=False,
        )
        if view_align is not None:
            aligned["yaw"] = float(view_align["yaw"])
            aligned["img_deg"] = float(view_align["img_deg"])
            aligned["img_scale"] = float(view_align["img_scale"])
        if pixels is not None:
            aligned["blue_uv"] = pixels
            chosen_xyz = xyz_live if len(pix_live) >= 2 else xyz_live + xyz_coast
            if base_uv is not None and chosen_uv:
                # Drop the same pedestal blob that was filtered from chosen_uv.
                kept_xyz = []
                src_uv = pix_live if len(pix_live) >= 2 else pix_live + pix_coast
                src_xyz = xyz_live if len(pix_live) >= 2 else xyz_live + xyz_coast
                for p, xyz in zip(src_uv, src_xyz):
                    if abs(p[0] - float(base_uv[0])) > 3.0 or abs(
                        p[1] - float(base_uv[1])
                    ) > 3.0:
                        kept_xyz.append(xyz)
                aligned["blue_xyz"] = kept_xyz
            else:
                aligned["blue_xyz"] = chosen_xyz
            # Live-only (no coast) for OnlineCamPose.
            live_uv = list(pix_live)
            live_xyz = list(xyz_live)
            if base_uv is not None and live_uv:
                kept_uv, kept_xyz_live = [], []
                for p, xyz in zip(live_uv, live_xyz):
                    if abs(p[0] - float(base_uv[0])) > 3.0 or abs(
                        p[1] - float(base_uv[1])
                    ) > 3.0:
                        kept_uv.append(p)
                        kept_xyz_live.append(xyz)
                live_uv, live_xyz = kept_uv, kept_xyz_live
            aligned["blue_uv_live"] = (
                np.asarray(live_uv, dtype=np.float64).reshape(-1, 2)
                if live_uv
                else None
            )
            aligned["blue_xyz_live"] = live_xyz
        aligned["_fit_key"] = fit_key
        aligned["_fit_t"] = now
        if search:
            self._ur_search_t = now
        fits[cam_key] = aligned
        self._ur_fit = aligned
        self._ur_fit_t = now
        return aligned

    def _overlay_ur_fk(
        self,
        out: np.ndarray,
        views: list,
        *,
        pose,
        fx: float,
        cam: str,
        tape_bgr=None,
        joint_tags: Optional[dict] = None,
    ):
        """Draw FK skeleton, confirm vs vision, return extra rack alerts."""
        from .ur_skeleton_draw import (
            confirm_vision_vs_fk,
            draw_ur_skeleton,
            fk_alerts_as_proximity,
        )

        if (not self.use_legacy_tag_fit) and not self._cam_info_known(cam):
            self._warn_missing_cam_info(cam)
            return self._draw_no_cam_info_hud(out, cam), [], None

        state = self._refresh_ur_state()
        if not state:
            return out, [], None
        state = self._align_ur_to_blues(
            state,
            views,
            pose=pose,
            fx=fx,
            frame_hw=out.shape[:2],
            cam=cam,
            bgr=tape_bgr,
        )
        cfg = state["cfg"]
        pts = state["pts_vision_mm"]
        matches = []
        if self.ur_fk_confirm:
            matches = confirm_vision_vs_fk(
                views, pts, gate_mm=float(cfg.confirm_gate_mm)
            )
        rack_hits = state["rack_hits"] if self.ur_fk_rack_alert else []
        if self.ur_fk_draw:
            tag_uv = [None] * 6
            tags = joint_tags or {}
            # Tags were found on the raw frame. The stick figure is on `out`.
            raw_hw = None if tape_bgr is None else tape_bgr.shape[:2]
            if raw_hw and raw_hw[0] > 0 and raw_hw[1] > 0:
                sx = out.shape[1] / float(raw_hw[1])
                sy = out.shape[0] / float(raw_hw[0])
            else:
                sx, sy = 1.0, 1.0
            for mid, hit in tags.items():
                mid_i = int(mid)
                if mid_i < 0 or mid_i > 5:
                    continue
                # Joint tags are live. The base tag may coast: it does not
                # translate when the arm moves, and it replaces the taped preset.
                if hit.get("coasting") and mid_i not in (0, 1):
                    continue
                cu, cv = hit["center"]
                tag_uv[mid_i] = (float(cu) * sx, float(cv) * sy)
            # OnlineCamPose gets only tags detected this frame (no 10 s hold,
            # no coasting base/shoulder). Draw/legacy still use hold[] below.
            live_tag_uv = [None] * 6
            for mid, hit in tags.items():
                mid_i = int(mid)
                if mid_i < 0 or mid_i > 5:
                    continue
                if hit.get("coasting"):
                    continue
                cu, cv = hit["center"]
                live_tag_uv[mid_i] = (float(cu) * sx, float(cv) * sy)
            hold = getattr(self, "_base_tag_uv", None)
            hold_t = getattr(self, "_base_tag_t", None)
            if hold is None:
                hold = {}
                self._base_tag_uv = hold
            if hold_t is None:
                hold_t = {}
                self._base_tag_t = hold_t
            now_tag = time.monotonic()
            if tag_uv[0] is not None:
                hold[cam] = tag_uv[0]
                hold_t[cam] = now_tag
            elif cam in hold and now_tag - float(hold_t.get(cam) or 0.0) < 10.0:
                tag_uv[0] = hold[cam]
            # Base of the STEP chain is the base ArUco tag, not the preset tape.
            anchor = tag_uv[0]
            pts_fk = None
            try:
                from ur_skeleton import fk_joint_positions_mm

                pts_fk = fk_joint_positions_mm(
                    state["q"], model=str(state.get("model") or "ur5e")
                )
            except Exception:
                pts_fk = None
            # Arm-tag calibration is solved in plain robot-base coordinates
            # (Z up, no mount swap, no per-camera yaw). Project those same
            # points. The Z-up→Y-down swap and LIVE_VIEW_ALIGN yaw are only
            # the seating used when the pose is still the 90° guess.
            calibrated = False
            if self._cell_frames is not None and pts_fk is not None:
                from .cell_frames import LAYOUT_ARM_TAGS

                calibrated = getattr(self._cell_frames, "layout", "") == LAYOUT_ARM_TAGS

            # Online per-camera pose (default). It estimates R,t from tags + blue
            # caps over many paused poses and then draws by pure projection. The
            # old tag-fitting heuristics stay behind use_legacy_tag_fit.
            pose_draw = pose
            fy_draw = cx_draw = cy_draw = None
            K_draw = None
            dist_draw = None
            online_pose = None
            if (not self.use_legacy_tag_fit) and pts_fk is not None:
                intr = self._scaled_intrinsics(cam, out.shape[:2], raw_hw)
                if intr is not None:
                    arm_speed = self._arm_joint_speed(state.get("q"))
                    online_pose = self._get_online_pose(cam, intr)
                    online_pose.add_observations(
                        pts_fk,
                        live_tag_uv,
                        state.get("blue_uv_live") or [],
                        arm_speed,
                        blue_xyz=state.get("blue_xyz_live"),
                    )
                    online_pose.update()
                    if online_pose.have_pose:
                        from .cell_frames import CamPose

                        pose_draw = CamPose(
                            frame_id=f"online_{cam}",
                            R=online_pose.R.copy(),
                            t_mm=online_pose.t.copy(),
                            ready=True,
                        )
                        fx = float(intr["fx"])
                        fy_draw = float(intr["fy"])
                        cx_draw = float(intr["cx"])
                        cy_draw = float(intr["cy"])
                        K_draw = online_pose.K
                        dist_draw = online_pose.dist
                        calibrated = True
                        if self.online_save_cell and online_pose.last_accepted:
                            try:
                                online_pose.save()
                            except Exception:
                                pass

            img_deg = 0.0 if calibrated else float(state.get("img_deg") or 0.0)
            img_scale = 1.0 if calibrated else float(state.get("img_scale") or 1.0)
            out = draw_ur_skeleton(
                out,
                pts_fk if calibrated else pts,
                pose_draw,
                fx=fx,
                matches=matches,
                rack_hits=rack_hits,
                source=state["source"],
                model=state["model"],
                pts_upright_mm=pts_fk if calibrated else state.get("pts_upright_mm"),
                check_ok=state.get("check_ok"),
                check_mm=state.get("check_mm"),
                check_unit=str(state.get("check_unit") or "mm"),
                base_anchor_uv=anchor,
                img_deg=img_deg,
                img_scale=img_scale,
                blue_uv=state.get("blue_uv"),
                has_tool=bool(state.get("has_tool")),
                tag_uv=tag_uv,
                pts_fk_mm=pts_fk,
                cam=cam,
                calibrated=calibrated,
                fy=fy_draw,
                cx=cx_draw,
                cy=cy_draw,
                K=K_draw,
                dist=dist_draw,
            )
            if online_pose is not None:
                out = self._draw_online_pose_hud(out, cam, online_pose)
        fk_alerts = fk_alerts_as_proximity(rack_hits) if rack_hits else []
        guide = None
        if rack_hits:
            worst = min(rack_hits, key=lambda x: x[1])
            guide = (
                f"FK RACK KEEP-OUT: joint FK{worst[0]} gap={worst[1]:.0f}mm "
                f"(teach-pad / UR angles — works even if blue joint left FOV)"
            )
            now = time.monotonic()
            if now - float(self._last_fk_alert_log) >= 1.0:
                self._last_fk_alert_log = now
                self.get_logger().error(f"[{cam}] {guide}")
        return out, fk_alerts, guide

    def _update_arm_speed(self, q) -> None:
        """Compute joint speed once per new robot reading; cameras only read it."""
        if q is None:
            return
        try:
            q_now = np.asarray(q, dtype=np.float64).reshape(-1)
        except Exception:
            return
        now = time.monotonic()
        speed = None
        if (
            self._last_q is not None
            and self._last_q.shape == q_now.shape
            and (now - self._last_q_t) > 1e-3
        ):
            peak = 0.0
            dt = now - self._last_q_t
            for i in range(q_now.shape[0]):
                d = abs(float(q_now[i]) - float(self._last_q[i])) / dt
                if d > peak:
                    peak = d
            speed = peak
        self._last_q = q_now
        self._last_q_t = now
        self._arm_speed = speed

    def _arm_joint_speed(self, q) -> Optional[float]:
        """Stored max joint speed (rad/s). Does not update _last_q."""
        _ = q
        return getattr(self, "_arm_speed", None)

    def _maybe_reset_online(self) -> None:
        """Clear every OnlineCamPose when R is pressed or online_reset is True."""
        want = False
        try:
            want = bool(self.get_parameter("online_reset").value)
        except Exception:
            want = False
        prev = getattr(self, "_preview", None)
        if prev is not None and getattr(prev, "reset_requested", False):
            want = True
            prev.reset_requested = False
        if not want:
            return
        for op in self._online_pose.values():
            try:
                op.reset()
            except Exception:
                pass
        try:
            from rclpy.parameter import Parameter

            self.set_parameters(
                [Parameter("online_reset", Parameter.Type.BOOL, False)]
            )
        except Exception:
            pass
        self.get_logger().info("OnlineCamPose reset (buffer + pose cleared)")

    def _draw_online_pose_hud(self, out: np.ndarray, cam: str, op) -> np.ndarray:
        """Per-camera online-pose status: buffer pairs, distinct poses, error."""
        err = op.mean_px
        err_txt = "--" if not (err == err) else f"{err:.1f}px"  # NaN-safe
        err3 = getattr(op, "mean_3d_mm", float("nan"))
        err3_txt = "--" if not (err3 == err3) else f"{err3:.0f}mm"
        n3 = int(getattr(op, "n_3d", 0))
        state_txt = "ACC" if op.last_accepted else ("pose" if op.have_pose else "solving")
        line = (
            f"online {cam}: pairs {op.n_pairs} poses {op.n_poses} "
            f"3d {n3} err {err_txt}/{err3_txt} [{state_txt}]"
        )
        color = (0, 220, 0) if op.last_accepted else (0, 210, 210)
        y = out.shape[0] - 40
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
        return out

    def _overlay_arm_on(
        self, decorated: np.ndarray, raw: np.ndarray, *, cam: str
    ) -> np.ndarray:
        """Detect on raw frame, then draw boxes onto the decorated overlay."""
        if not self.enable_arm_cv or self._robotics is None:
            return decorated
        seq = self._cam0_seq if cam == "cam0" else self._cam1_seq
        cache_attr = "_arm_boxes0" if cam == "cam0" else "_arm_boxes1"
        if not hasattr(self, "_arm_boxes0"):
            self._arm_boxes0 = None
            self._arm_boxes1 = None
        if seq % self.arm_every_n == 0 or getattr(self, cache_attr) is None:
            try:
                from .arm_overlay_draw import default_arm_weights, get_worker

                wpath = self.arm_weights or default_arm_weights(self._robotics)
                boxes = get_worker(wpath, conf=self.arm_conf).predict(
                    raw, conf=self.arm_conf, cam=cam, refine=True
                )
                setattr(self, cache_attr, boxes)
            except Exception as exc:
                self.get_logger().warn(f"arm detect {cam}: {exc}")
                setattr(self, cache_attr, [])
        out = decorated.copy()
        for xyxy, conf in getattr(self, cache_attr) or []:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 0), 2)
            cv2.putText(
                out,
                f"arm {conf:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 0),
                2,
                cv2.LINE_AA,
            )
        return out

    def _paint_cam1_cv(self, bgr: np.ndarray) -> np.ndarray:
        self._cam1_seq += 1
        if self.enable_rack_cv:
            if self._cam1_seq % self.classify_every_n == 0 or self._latest1_cv is None:
                self._last_state = self._resolve_state(bgr)
            view = draw_rack_overlay(
                bgr,
                self.display_xywh,
                self._last_state,
                classify_xywh=self.classify_xywh,
            )
            cv2.putText(
                view,
                "cam1 + rack + arm",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            view = _label(bgr, "cam1")
        view = self._overlay_arm_on(view, bgr, cam="cam1")
        return self._overlay_blue_on(view, bgr, cam="cam1")

    def _paint_cam2_cv(self, bgr: np.ndarray) -> np.ndarray:
        return _label(bgr, "cam2")

    def _on_depth0(self, msg: Image) -> None:
        try:
            d = imgmsg_to_numpy(msg)
            if d.ndim == 2:
                self._latest_depth0 = d
        except Exception as exc:
            self.get_logger().warn(f"cam0 depth decode failed: {exc}")

    def _on_depth1(self, msg: Image) -> None:
        try:
            d = imgmsg_to_numpy(msg)
            if d.ndim == 2:
                self._latest_depth1 = d
        except Exception as exc:
            self.get_logger().warn(f"cam1 depth decode failed: {exc}")

    def _on_depth2(self, msg: Image) -> None:
        try:
            d = imgmsg_to_numpy(msg)
            if d.ndim == 2:
                self._latest_depth2 = d
        except Exception as exc:
            self.get_logger().warn(f"cam2 depth decode failed: {exc}")

    def _on_cam0(self, msg: Image) -> None:
        try:
            img = imgmsg_to_numpy(msg)
            if img.ndim == 3:
                self._latest0 = img
                self._latest0_cv = self._paint_cam0_cv(img)
                self._recv0 += 1
                if self._recv0 == 1:
                    self.get_logger().info(
                        f"cam0 first frame {img.shape[1]}x{img.shape[0]} {msg.encoding}"
                    )
        except Exception as exc:
            self.get_logger().warn(f"cam0 decode/CV failed: {exc}")

    def _on_cam1(self, msg: Image) -> None:
        try:
            img = imgmsg_to_numpy(msg)
            if img.ndim == 3:
                self._latest1 = img
                self._latest1_cv = self._paint_cam1_cv(img)
                self._recv1 += 1
                if self._recv1 == 1:
                    self.get_logger().info(
                        f"cam1 first frame {img.shape[1]}x{img.shape[0]} {msg.encoding}"
                    )
        except Exception as exc:
            self.get_logger().warn(f"cam1 decode/CV failed: {exc}")

    def _on_cam2(self, msg: Image) -> None:
        try:
            img = imgmsg_to_numpy(msg)
            if img.ndim == 3:
                self._latest2 = img
                self._latest2_cv = self._paint_cam2_cv(img)
                self._recv2 += 1
                if self._recv2 == 1:
                    self.get_logger().info(
                        f"cam2 first frame {img.shape[1]}x{img.shape[0]} {msg.encoding}"
                    )
        except Exception as exc:
            self.get_logger().warn(f"cam2 decode/CV failed: {exc}")

    def _status_line(self, recv: int) -> str:
        age = int(time.monotonic() - self._started)
        if recv > 0:
            return f"ok ({recv})"
        return f"no ROS frames {age}s — is publisher up?"

    def _bootstrap_origin_from_j0(self, cam: str) -> None:
        """Lock cell_frames so the pedestal blue joint is ur_base (0,0,0)."""
        if self._joint_bank is None or self._cell_frames is None:
            return
        done = getattr(self, "_origin_bootstrapped", None)
        if done is None:
            done = set()
            self._origin_bootstrapped = done
        if cam in done:
            return
        rec = getattr(self._joint_bank, "_origin_lock", {}).get(cam) or {}
        p_mm = rec.get("p_cam_mm")
        if not p_mm:
            return
        from .cell_frames import CamPose

        pose = self._cell_frames.pose(cam)
        p = np.asarray(p_mm, dtype=np.float64).reshape(3)
        t = np.zeros(3, dtype=np.float64)
        for i in range(3):
            acc = 0.0
            for j in range(3):
                acc += float(pose.R[i, j]) * float(p[j])
            t[i] = -acc
        new_pose = CamPose(
            frame_id=pose.frame_id, R=pose.R.copy(), t_mm=t, ready=True
        )
        done.add(cam)
        if cam == "cam0":
            self._cell_frames.cam0 = new_pose
        else:
            self._cell_frames.cam1 = new_pose

    def _stash_fuse_points(
        self,
        cam: str,
        depth: Optional[np.ndarray],
        boxes: list,
        mask: np.ndarray,
        alerts: list,
    ) -> None:
        """Cache cam-frame 3D base + alerting joint for fused ur_base retreat."""
        from .baby_blue_draw import (
            _blue_point_mm,
            is_base_blue_box,
        )
        from .cell_frames import bootstrap_pose_from_base_landmark, cam_to_base

        fuse_attr = "_fuse0" if cam == "cam0" else "_fuse1"
        if depth is None or not boxes:
            setattr(self, fuse_attr, None)
            return
        dh, dw = depth.shape[:2]
        fx = 520.0
        fy = 520.0
        cx_f, cy_f = 0.5 * dw, 0.5 * dh
        info = self._cam_info.get(cam)
        if info:
            from .baby_blue_draw import depth_frame_intrinsics

            ch = int(info.get("height") or 0)
            cw = int(info.get("width") or 0)
            if ch > 0 and cw > 0:
                fx, fy, cx_f, cy_f = depth_frame_intrinsics(
                    float(info["fx"]),
                    float(info["fy"]),
                    float(info["cx"]),
                    float(info["cy"]),
                    (ch, cw),
                    (dh, dw),
                )
        elif self._cell_frames is not None:
            fx = float(self._cell_frames.fx_px)
            fy = fx
        base_box = None
        for b in boxes:
            if is_base_blue_box(b[0], b[1], b[2], b[3], dh):
                base_box = b
                break
        if base_box is None:
            base_box = max(boxes, key=lambda b: b[1] + 0.5 * b[3])
        bp_base = _blue_point_mm(
            depth,
            mask,
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
        p_joint_cam = None
        if alerts:
            alert = min(alerts, key=lambda a: float(a[4]))
            ax, ay, aw, ah = (
                int(alert[0]),
                int(alert[1]),
                int(alert[2]),
                int(alert[3]),
            )
            d_alert = None
            for bx, by, bw, bh, bd in boxes:
                if abs(bx - ax) < 8 and abs(by - ay) < 8:
                    d_alert = bd
                    break
            bp_j = _blue_point_mm(
                depth,
                mask,
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
            if bp_j is not None:
                p_joint_cam = bp_j[0]

        p_base_cam = bp_base[0] if bp_base is not None else None
        # Bootstrap / refresh T_base_cam from pedestal landmark
        origin_locked = cam in getattr(
            getattr(self, "_joint_bank", None), "_origin_lock", {}
        )
        if (
            self._cell_frames is not None
            and self._cell_frames.bootstrap_from_blue_base
            and p_base_cam is not None
            and not origin_locked
        ):
            pose = self._cell_frames.pose(cam)
            # Keep configured R; set t so pedestal → ur_base origin
            new_pose = bootstrap_pose_from_base_landmark(
                p_base_cam, R=pose.R, frame_id=pose.frame_id
            )
            if cam == "cam0":
                self._cell_frames.cam0 = new_pose
            else:
                self._cell_frames.cam1 = new_pose

        p_joint_base = None
        if (
            p_joint_cam is not None
            and self._cell_frames is not None
            and self._cell_frames.pose(cam).ready
        ):
            p_joint_base = cam_to_base(p_joint_cam, self._cell_frames.pose(cam))

        setattr(
            self,
            fuse_attr,
            {
                "alerted": bool(alerts),
                "p_joint_cam": p_joint_cam,
                "p_base_cam": p_base_cam,
                "p_joint_base": p_joint_base,
                "gap_mm": float(min(a[4] for a in alerts)) if alerts else None,
            },
        )

    def _fuse_base_retreat(self) -> None:
        """One terminal instruction in ur_base when either cam proximity-alerts."""
        from .cell_frames import format_base_retreat, save_cell_frames

        if not (self._blue_alert0 or self._blue_alert1):
            return
        now = time.monotonic()
        if now - float(self._last_fuse_log) < 1.0:
            return
        self._last_fuse_log = now

        # Prefer the camera with the tighter gap
        candidates = []
        for cam, fuse in (("cam0", self._fuse0), ("cam1", self._fuse1)):
            if not fuse or not fuse.get("alerted"):
                continue
            pjb = fuse.get("p_joint_base")
            if pjb is None:
                continue
            candidates.append((fuse.get("gap_mm") or 1e9, cam, pjb))
        if not candidates:
            return
        candidates.sort(key=lambda x: x[0])
        _gap, cam, p_joint_base = candidates[0]
        msg = format_base_retreat(p_joint_base)
        src = []
        if self._blue_alert0:
            src.append("cam0")
        if self._blue_alert1:
            src.append("cam1")
        full = f"[{'+'.join(src)} via {cam}] {msg}"
        self.get_logger().error(full)
        print(f"\n*** {full} ***\n", flush=True)
        # Persist bootstrapped poses occasionally
        if self._cell_frames is not None:
            try:
                save_cell_frames(self._cell_frames)
            except Exception:
                pass

    def _on_timer(self) -> None:
        if self._should_exit:
            return
        self._maybe_reset_online()
        if self.enable_ur_fk:
            self._refresh_ur_state()
        age = time.monotonic() - self._started
        if (
            not self._warned_empty
            and age >= 8.0
            and (self._recv0 == 0 or self._recv1 == 0)
        ):
            self._warned_empty = True
            self.get_logger().error(
                f"Still empty after {age:.0f}s (cam0={self._recv0} cam1={self._recv1}). "
                "Publishers must be streaming; close USB dual viewers first."
            )
        try:
            left = self._latest0_cv if self._latest0_cv is not None else self._latest0
            right = self._latest1_cv if self._latest1_cv is not None else self._latest1
            frame = compose_side_by_side(
                left,
                right,
                max_height=self.max_height,
                left_status=self._status_line(self._recv0),
                right_status=self._status_line(self._recv1),
                left_already_labeled=self._latest0_cv is not None,
                right_already_labeled=self._latest1_cv is not None,
            )
            depth_row = compose_side_by_side(
                self._depth_map_bgr(self._latest_depth0, "cam0 depth"),
                self._depth_map_bgr(self._latest_depth1, "cam1 depth"),
                max_height=max(160, int(self.max_height) // 2),
                left_already_labeled=True,
                right_already_labeled=True,
            )
            third = self._latest2_cv if self._latest2_cv is not None else self._latest2
            if third is None:
                third = _placeholder(
                    ["waiting cam2…", self._status_line(self._recv2)]
                )
            frame = _attach_pane(frame, third)
            depth_row = _attach_pane(
                depth_row, self._depth_map_bgr(self._latest_depth2, "cam2 depth")
            )
            if depth_row.shape[1] != frame.shape[1]:
                depth_row = cv2.resize(depth_row, (frame.shape[1], depth_row.shape[0]))
            frame = cv2.vconcat([frame, depth_row])
            if self._blue_alert0 or self._blue_alert1:
                # Extra full-window red signal so it's impossible to miss
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 18)
                cv2.putText(
                    frame,
                    "ERROR: BLUE WITHIN 10cm",
                    (24, max(40, h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "FUSED RETREAT -> ur_base (0,0,0)  see terminal",
                    (24, max(80, h // 2 + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                try:
                    self._fuse_base_retreat()
                except Exception as exc:
                    self.get_logger().warn(f"ur_base fuse failed: {exc}")
            ok = self._preview.show(frame)
            if not ok and self._preview.user_closed:
                self.get_logger().info("Preview closed — exiting dual view.")
                self._should_exit = True
            elif (
                not ok
                and self._preview.last_error
                and not self._preview_error_logged
            ):
                self._preview_error_logged = True
                self.get_logger().error(
                    f"Preview backend failed: {self._preview.last_error}"
                )
        except Exception as exc:
            self.get_logger().error(
                f"dual view refresh failed: {exc}\n{traceback.format_exc()}"
            )

    def destroy_node(self) -> bool:
        try:
            self._preview.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DualViewNode()
        # Single-threaded: tkinter preview must run on the main thread.
        while rclpy.ok() and not node._should_exit:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
