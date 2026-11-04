"""Track a vial on an Orbbec depth camera and show how far away it is.

The clear cup does not return a steady depth reading. Distance is taken from
the white cap, which is opaque. A trained YOLO model is used when weights
exist. Otherwise the white cap is found directly and the box is extended
down over the cup.

Single-camera mode is fully independent. Classic ``--dual`` still uses
``locate_pair`` (sticky peer soft assist). Preferred dual UX is
``--dual-independent``: two processes, each detects alone; localhost IPC
only corroborates distances (never averages or replaces local range).

    cd E:\\Desktop\\Jessica\\Camera
    python depthcamera.py
    python depthcamera.py --camera 0
    python depthcamera.py --camera 1
    python depthcamera.py --dual-independent
    python depthcamera.py --dual-independent --3d-reproject-assist
    python depthcamera.py --dual-independent --no-edge-assist
    python depthcamera.py --dual-independent --orb-assist
    python depthcamera.py --dual
    python depthcamera.py --camera 0 --camera2 1
    python depthcamera.py --list-cameras
    python depthcamera.py --model runs\\vial-live\\weights\\best.pt

Live 3D size-score / lid@Z prior stays off by default. Opt-in cross-cam
``--3d-reproject-assist`` works when ``stereo_extrinsics.json`` is real
(not PLACEHOLDER); local clear-vial / soft-clear / cap gates must still accept.
Offline tools remain in ``vial_3d_model.py`` / ``vial_3d_identify.py``. After YOLO
and cap-first miss, optional dx/dy edge assist (``detect_vial_by_edges``, default on)
may propose / re-rank local ROIs that still must pass clear-vial / scale gates.
Tiny-weight ORB assist is **off by default** (``--orb-assist``); when on it only soft-
ties among non-upright candidates — never invents boxes and never overrides a pool
that already has an upright full-vial box. Far vials allowed when the box is small
enough in mm at Z. Pose selection is **upright-first**: if any candidate passes
full-vial aspect (~1.0–1.65), that wins; otherwise squat top-down lid (~0.72–0.99)
or clearly horizontal sideways (~0.40–0.71) with lid-on-end geometry.
``--sideways-assist`` (default on) grows the clear cup horizontally from the
white end-cap and expands wide YOLO boxes; upright lids never use that path.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

try:
    from sideways_race.methods.method_i_hard_robust import (
        detect_sideways as _detect_sideways_race_winner,
    )
except ImportError:
    # Keep the camera module importable in minimal deployments that omit race assets.
    _detect_sideways_race_winner = None

ROOT = Path(__file__).resolve().parent
# Prefer the exported live checkpoint, then Attempt-E lineage; never the newest random run.
# Train-race R1 winner is copied to vial-live (and archived under train-race/r1-winner).
LIVE_WEIGHTS = ROOT / "runs" / "vial-live" / "weights" / "best.pt"
R1_WINNER_WEIGHTS = ROOT / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt"
BON_WINNER_WEIGHTS = ROOT / "runs" / "bon-winner" / "weights" / "best.pt"
WINDOW = "Vial distance"
MIN_DEPTH_MM = 0
MAX_DEPTH_MM = 5000
# Legacy display/docs hint only — live accept no longer hard-cuts at this range.
# Far vials pass when vial_scale says the pixel box is small enough at Z.
MAX_RANGE_MM = 1500
HOLD_FRAMES = 15
# Dual cam: soft white-cap gates stay available this many frames after a strict clear hit.
PEER_MEMORY_FRAMES = 15
# Independent dual: UDP IPC on localhost (base + camera_index).
IPC_BASE_PORT = 47800
PEER_STALE_S = 1.0
# Corroboration: agree when |Δcm| ≤ abs OR relative ≤ rel (local distance kept).
CORROBORATE_ABS_CM = 3.0
CORROBORATE_REL = 0.15
SNAPSHOT_INTERVAL_S = 3.0
SNAPSHOT_MAX_PER_RUN = 50
SNAPSHOT_MIN_MOVE_IOU = 0.85
# Keep distinct vials while merging duplicate proposals for the same object.
MAX_VIALS_PER_CAMERA = 8
VIAL_NMS_IOU = 0.45
# Re-check the whole image often enough to discover newly-arrived vials. Between
# those scans, small padded crops around every accepted vial keep stationary scenes
# cheap without turning the tracker into a permanent spatial prior.
ROI_FULL_SEARCH_INTERVAL = 5
# Full-frame discovery is deliberately a little more permissive than ROI refresh.
# The depth/clear-vial/scale gates below still decide acceptance, so this recovers a
# lower-scoring second YOLO proposal without weakening the tracked vial.
DISCOVERY_YOLO_CONF_DELTA = 0.03
MIN_DISCOVERY_YOLO_CONF = 0.05
ROI_PAD_FRACTION = 0.65
ROI_MIN_SIZE = 96
ROI_PREDICT_IMGSZ = 256
ROI_FORGET_MISSES = 2
# A track owns its ROI until it has missed this many complete frame updates.
TRACK_MAX_MISSES = ROI_FORGET_MISSES
TRACK_ASSOC_IOU = 0.08
REVIEW_PENDING = ROOT / "vial_dataset" / "review" / "pending"
REVIEW_WRONG = ROOT / "vial_dataset" / "review" / "wrong"
REVIEW_INCORRECT = ROOT / "vial_dataset" / "review" / "incorrect"
RAW_DATASET = ROOT / "vial_dataset" / "raw"
# Full vial (lid + clear cup) is roughly square-to-tall; lid-only boxes are flatter.
FULL_VIAL_ASPECT_MIN = 1.0
FULL_VIAL_ASPECT_MAX = 1.65
# Sideways / fallen: clearly horizontal lid+cup (w≫h). Kept below top-down band
# so squat lids (0.72–0.99) and upright (1.0–1.65) are not stolen.
# Slightly wider than the old thin 0.45–0.70 band; live path still requires
# lid-on-left/right geometry so squat upright mistakes do not steal.
SIDEWAYS_ASPECT_MIN = 0.40
SIDEWAYS_ASPECT_MAX = 0.71  # keep below top-down band start (0.72)
# Cap width times this is the max fitted height. Was 1.2 (too short for lid+cup).
FIT_MAX_HEIGHT_CAP_RATIO = 1.55
# Horizontal cup span relative to short-axis (Ø) for sideways growth.
FIT_SIDEWAYS_LENGTH_RATIO = 1.65
# Clear cup: depth farther than the lid by this much counts as "see-through".
CLEAR_FAR_DELTA_MM = 200
# Cup must show at least this much more hole/far depth than the opaque lid.
CLEAR_HOLE_MARGIN = 0.12
# Opaque mason-jar bodies keep this much near-depth under the lid.
OPAQUE_CUP_NEAR_FRAC = 0.55
# Default YOLO conf for the strong / reference camera profile.
# Below the old 0.25 race/eval floor to cut independent-dual false negatives
# when YOLO scores a real white-capped vial just under that threshold.
DEFAULT_YOLO_CONF = 0.15
# Live asymmetric gates snapshot (BoN #3). Weak cam conf/floors prefer recall.
CAMERA_GATES_JSON = ROOT / "camera_gates.json"
_GATES_FILE_CACHE: dict | None = None
# Live 3D size-score / rank / lid@Z prior stays off. Cross-cam reproject is a
# separate narrow opt-in (--3d-reproject-assist) when extrinsics are real.
LIVE_3D_VIAL_ENABLED = False
# Free-form vial YOLO / locate is OFF for default entrypoints. Rack tools only.
# Opt-in for tests: python depthcamera.py --enable-vial-detect ...
VIAL_DETECT_ENABLED_DEFAULT = False
RACK_TOOL_HINT = (
    "Free-form vial detection is DISABLED (rack tools only).\n"
    "  cd C:\\Users\\Ceder-ALAB\\Desktop\\Jessica\\Jessica\\robotics\n"
    "  python rack_lid_state.py focus\n"
    "  python rack_capture_agents.py run --camera 1 --seconds 30\n"
    "Legacy vial trackers require an explicit --enable-vial-detect."
)
LIVE_3D_REPROJECT_ASSIST_ALLOWED = True
DEFAULT_USE_3D_MODEL_SCORE = False
DEFAULT_3D_MODEL_SCORE_MIN = 0.05
# Cap-first top-down: accept white blobs whose width@Z ≈ fitted lid Ø (unused while 3D off).
DEFAULT_LID_WIDTH_TOL = 0.45
# Image-derivative assist after YOLO/cap miss (explicit Gx/Gy + rim). Local only.
DEFAULT_USE_EDGE_ASSIST = True
# Tiny-weight ORB assist: soft dual agree / choose_box boost only (not a detector).
# Default OFF; when on, never overrides an upright full-vial candidate pool.
DEFAULT_USE_ORB_ASSIST = False
DEFAULT_ORB_ASSIST_WEIGHT = 0.05
ORB_MAX_FEATURES = 32
ORB_IPC_MAX_FEATURES = 16  # keep UDP PeerReport payloads small
ORB_RATIO_TEST = 0.75
ORB_AGREE_GOOD_MATCHES = 5
ORB_AGREE_SCORE_MIN = 0.22
# Sideways / fallen live path — on by default; upright-first ranking protects upright.
DEFAULT_ALLOW_SIDEWAYS = True
_ALLOW_SIDEWAYS = DEFAULT_ALLOW_SIDEWAYS
# Improved horizontal cap→cup growth + wide-YOLO expand (default on with sideways).
DEFAULT_SIDEWAYS_ASSIST = True
_SIDEWAYS_ASSIST = DEFAULT_SIDEWAYS_ASSIST
# Surface nearer than this is treated as junk (keep); far end uses vial_scale, not a hard m cutoff.
MIN_SURFACE_MM = 80.0
# Live range-band labels (overlay + primary ranking). Close covers rack/gripper work.
CLOSE_RANGE_MM = 500.0
MID_RANGE_MM = 1000.0
_3D_IDENTIFY_STACK = None
_3D_IDENTIFY_DIMS = None
_LIVE_3D_STATUS_PRINTED = False
_PHYSICAL_CHECKLIST_PRINTED = False
USB_TIP = (
    "Each Orbbec needs its own USB 3 port. "
    "Close Orbbec Viewer and other apps using the camera, then try again. "
    "Two cameras are indexes 0 and 1 — use: python depthcamera.py --dual-independent "
    "(preferred) or --dual / --camera 0 --camera2 1. Index 2 is not a second Orbbec."
)
PHYSICAL_DUAL_CHECKLIST = (
    "Physical dual checklist (once per session) — full notes: DUAL_PHYSICAL_CHECKLIST.md\n"
    "  1. Each Orbbec on its own USB 3 port (not a shared hub / USB2).\n"
    "  2. Close Orbbec Viewer / other apps holding the cameras.\n"
    "  3. Aim both FOVs so the vial (lid + cup) is in both views; reduce glare on the clear cup.\n"
    "  4. Weak angle is camera index 0 by default (--weak-camera 0); swap if cam1 is the hard view.\n"
    "  5. Stereo reproject needs real extrinsics: python stereo_calib_stub.py status\n"
    "     Capture board pairs: python stereo_calib_stub.py capture"
)


@dataclass(frozen=True)
class CameraGateProfile:
    """Per-camera clear-vial gates (asymmetric for a weak dual-cam viewpoint).

    Weak profiles lower YOLO conf and relax hole / aspect / edge floors slightly
    so a harder angle still IDs the clear vial+white cap, while opaque walls and
    mason jars still fail lid-solidity / opaque-near / structure checks.
    """

    name: str = "strong"
    yolo_conf: float = DEFAULT_YOLO_CONF
    clear_hole_margin: float = CLEAR_HOLE_MARGIN
    lid_hole_max: float = 0.55
    cup_hole_min: float = 0.25
    opaque_cup_near_frac: float = OPAQUE_CUP_NEAR_FRAC
    clear_far_delta_mm: float = CLEAR_FAR_DELTA_MM
    aspect_min: float = FULL_VIAL_ASPECT_MIN
    aspect_max: float = FULL_VIAL_ASPECT_MAX
    usable_aspect_max: float = 1.85
    interior_edge_min: float = 0.012

    def is_full_vial_aspect(self, box) -> bool:
        aspect = box_aspect_hw(box)
        return self.aspect_min <= aspect <= self.aspect_max


def load_camera_gates_file(path: Path | None = None, *, reload: bool = False) -> dict:
    """Load ``camera_gates.json`` (cached). Missing/invalid file → empty dict."""
    global _GATES_FILE_CACHE
    gate_path = Path(path) if path is not None else CAMERA_GATES_JSON
    if path is None and _GATES_FILE_CACHE is not None and not reload:
        return _GATES_FILE_CACHE
    data: dict = {}
    if gate_path.exists():
        try:
            loaded = json.loads(gate_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError) as error:
            print(f"Warning: could not read {gate_path.name}: {error}")
    if path is None:
        _GATES_FILE_CACHE = data
    return data


def _gate_profile_from_mapping(
    data: dict | None, *, fallback: CameraGateProfile
) -> CameraGateProfile:
    """Overlay JSON keys onto a hardcoded fallback profile."""
    if not isinstance(data, dict):
        return fallback
    return CameraGateProfile(
        name=str(data.get("name", fallback.name)),
        yolo_conf=float(data.get("yolo_conf", fallback.yolo_conf)),
        clear_hole_margin=float(data.get("clear_hole_margin", fallback.clear_hole_margin)),
        lid_hole_max=float(data.get("lid_hole_max", fallback.lid_hole_max)),
        cup_hole_min=float(data.get("cup_hole_min", fallback.cup_hole_min)),
        opaque_cup_near_frac=float(
            data.get("opaque_cup_near_frac", fallback.opaque_cup_near_frac)
        ),
        clear_far_delta_mm=float(data.get("clear_far_delta_mm", fallback.clear_far_delta_mm)),
        aspect_min=float(data.get("aspect_min", fallback.aspect_min)),
        aspect_max=float(data.get("aspect_max", fallback.aspect_max)),
        usable_aspect_max=float(data.get("usable_aspect_max", fallback.usable_aspect_max)),
        interior_edge_min=float(data.get("interior_edge_min", fallback.interior_edge_min)),
    )


def _hardcoded_strong_profile() -> CameraGateProfile:
    return CameraGateProfile(name="strong", yolo_conf=DEFAULT_YOLO_CONF)


def _hardcoded_weak_profile() -> CameraGateProfile:
    """Code fallback matching camera_gates.json cam1_weak (recall-oriented)."""
    return CameraGateProfile(
        name="weak",
        yolo_conf=0.05,
        clear_hole_margin=0.05,
        lid_hole_max=0.68,
        cup_hole_min=0.15,
        opaque_cup_near_frac=0.68,
        clear_far_delta_mm=CLEAR_FAR_DELTA_MM,
        aspect_min=0.88,
        aspect_max=1.85,
        usable_aspect_max=2.10,
        interior_edge_min=0.004,
    )


def strong_gate_profile(*, yolo_conf: float | None = None) -> CameraGateProfile:
    file_data = load_camera_gates_file()
    profile = _gate_profile_from_mapping(
        file_data.get("cam0_strong"), fallback=_hardcoded_strong_profile()
    )
    if yolo_conf is None:
        return profile
    return replace(profile, yolo_conf=float(yolo_conf))


def weak_gate_profile(*, yolo_conf: float | None = None, base_conf: float = DEFAULT_YOLO_CONF) -> CameraGateProfile:
    """More permissive structure gates + lower YOLO conf for the failing dual cam.

    Defaults come from ``camera_gates.json`` ``cam1_weak`` when present so the
    weak camera actually uses the tuned low conf (0.05), not a stale hardcode.
    """
    file_data = load_camera_gates_file()
    profile = _gate_profile_from_mapping(
        file_data.get("cam1_weak"), fallback=_hardcoded_weak_profile()
    )
    if yolo_conf is not None:
        conf = float(yolo_conf)
    elif abs(float(base_conf) - DEFAULT_YOLO_CONF) < 1e-9:
        # Prefer the JSON/file weak conf when CLI still uses the default strong base.
        conf = float(profile.yolo_conf)
    else:
        # Non-default --conf: keep the historical offset, floored at the file weak conf.
        conf = max(float(profile.yolo_conf), float(base_conf) - 0.20)
    return replace(profile, yolo_conf=max(0.05, float(conf)))


def build_camera_gate_map(
    *,
    weak_camera: int | None = None,
    base_conf: float = DEFAULT_YOLO_CONF,
    cam0_conf: float | None = None,
    cam1_conf: float | None = None,
) -> dict[int, CameraGateProfile]:
    """Map OpenCV camera index → gate profile. Unknown indexes use the strong profile."""
    file_data = load_camera_gates_file()
    weak_idx = int(file_data.get("weak_camera", 0) if weak_camera is None else weak_camera)
    profiles: dict[int, CameraGateProfile] = {}
    for idx in (0, 1):
        explicit = cam0_conf if idx == 0 else cam1_conf
        if idx == weak_idx:
            profiles[idx] = weak_gate_profile(
                yolo_conf=explicit if explicit is not None else None,
                base_conf=base_conf,
            )
        else:
            profiles[idx] = strong_gate_profile(
                yolo_conf=float(explicit) if explicit is not None else float(base_conf)
            )
    return profiles


def gate_profile_for(camera_index: int, gate_map: dict[int, CameraGateProfile] | None = None) -> CameraGateProfile:
    if gate_map and camera_index in gate_map:
        return gate_map[camera_index]
    return strong_gate_profile()


@dataclass
class PeerMemory:
    """Sticky dual-cam memory: recent strict clear-vial confirm unlocks soft gates.

    Soft-gate assists do not refresh the timer. Memory clears once both cameras
    miss (no strict clear hit) for longer than ``window`` frames.
    """

    window: int = PEER_MEMORY_FRAMES
    frames_since_strict: int | None = None

    @property
    def active(self) -> bool:
        return (
            self.frames_since_strict is not None
            and self.frames_since_strict <= self.window
        )

    def observe(self, any_strict_hit: bool) -> bool:
        """Update from this frame's strict clears. Returns whether soft assist is on."""
        if any_strict_hit:
            self.frames_since_strict = 0
        elif self.frames_since_strict is not None:
            self.frames_since_strict += 1
            if self.frames_since_strict > self.window:
                self.frames_since_strict = None
        return self.active


@dataclass
class PairLocateResult:
    """Matched dual-camera fixes from one ``locate_pair`` call (same UI tick)."""

    fix_a: VialFix | None
    fix_b: VialFix | None
    peer_assist: bool = False


def locate_pair(
    cam_a: "OrbbecDepthCamera",
    frame_a: tuple | None,
    cam_b: "OrbbecDepthCamera",
    frame_b: tuple | None,
    *,
    peer_memory: PeerMemory | None = None,
) -> PairLocateResult:
    """Synchronized dual locate: both fixes with peer soft applied before return.

    Runs the strict clear-vial gate on each camera, updates sticky peer memory,
    then soft white-cap retries for misses — all inside this function so the UI
    always receives a paired result for the same frame tick. Distances stay
    per-camera (no averaging).

    ``frame_*`` is ``(color, depth_mm)`` or ``None`` when a camera has no frame.
    """
    if peer_memory is None:
        peer_memory = PeerMemory()

    def _strict(cam: "OrbbecDepthCamera", frame: tuple | None) -> VialFix | None:
        if frame is None:
            return None
        color, depth_mm = frame
        if color is None or depth_mm is None:
            return None
        return cam.locate(color, depth_mm, peer_confirmed=False)

    fix_a = _strict(cam_a, frame_a)
    fix_b = _strict(cam_b, frame_b)
    peer_assist = peer_memory.observe(fix_a is not None or fix_b is not None)

    def _soft_if_needed(
        cam: "OrbbecDepthCamera", frame: tuple | None, fix: VialFix | None
    ) -> VialFix | None:
        if fix is not None or not peer_assist:
            return fix
        if frame is None:
            return None
        color, depth_mm = frame
        if color is None or depth_mm is None:
            return None
        return cam.locate(color, depth_mm, peer_confirmed=True)

    return PairLocateResult(
        fix_a=_soft_if_needed(cam_a, frame_a, fix_a),
        fix_b=_soft_if_needed(cam_b, frame_b, fix_b),
        peer_assist=peer_assist,
    )


def locate_standalone(
    cam: "OrbbecDepthCamera", color, depth_mm: np.ndarray
) -> "VialFix | None":
    """Detect from this camera's RGB-D alone (single-cam / independent dual).

    Never uses sticky peer soft assist from the sibling camera. Enables local
    soft-clear; ``locate`` also runs cap-first fallback (``detect_vial_by_cap``)
    when YOLO/gates miss — no peer-invented boxes.
    """
    return cam.locate(color, depth_mm, peer_confirmed=False, soft_clear=True)


def format_gate_assignment_line(gate_map: dict[int, CameraGateProfile]) -> str:
    """One startup line: which index is weak (permissive) vs strong (tighter)."""
    weak_idxs = sorted(i for i, g in gate_map.items() if g.name == "weak")
    strong_idxs = sorted(i for i, g in gate_map.items() if g.name != "weak")
    weak_s = ",".join(f"cam{i}" for i in weak_idxs) if weak_idxs else "none"
    strong_s = ",".join(f"cam{i}" for i in strong_idxs) if strong_idxs else "none"
    return (
        f"Asymmetric gates: weak={weak_s} (permissive), strong={strong_s} (tighter). "
        f"Override: --weak-camera N"
    )


def print_camera_gate_assignment(gate_map: dict[int, CameraGateProfile]) -> None:
    print(format_gate_assignment_line(gate_map))


def print_physical_dual_checklist(*, force: bool = False) -> None:
    """Print USB / aim / glare / weak-cam checklist once per process (dual startup)."""
    global _PHYSICAL_CHECKLIST_PRINTED
    if _PHYSICAL_CHECKLIST_PRINTED and not force:
        return
    _PHYSICAL_CHECKLIST_PRINTED = True
    print(PHYSICAL_DUAL_CHECKLIST)


def _note_live_3d_status(*, want_reproject: bool = False, want_score: bool = False) -> None:
    """Print once: size-score off; reproject status if requested."""
    global _LIVE_3D_STATUS_PRINTED
    if _LIVE_3D_STATUS_PRINTED:
        return
    _LIVE_3D_STATUS_PRINTED = True
    print(format_3d_reproject_status_line(want_assist=want_reproject))
    if want_score and not LIVE_3D_VIAL_ENABLED:
        print("Live 3D size-score: off (default). Offline: python vial_3d_identify.py score-stills")



def _load_3d_assist_stack():
    """Lazy import for opt-in dual reproject (size-score stays separately gated)."""
    if not LIVE_3D_REPROJECT_ASSIST_ALLOWED:
        return None
    try:
        from vial_3d_model import (  # type: ignore
            assist_box_from_peer,
            default_intrinsics,
            extrinsics_reproject_state,
            is_extrinsics_placeholder,
            load_dims,
            load_extrinsics,
        )
    except ImportError:
        return None
    return (
        assist_box_from_peer,
        default_intrinsics,
        load_dims,
        load_extrinsics,
        is_extrinsics_placeholder,
        extrinsics_reproject_state,
    )


def format_3d_reproject_status_line(want_assist: bool = False) -> str:
    """Startup one-liner for size-score + optional reproject assist."""
    score_bit = "size-score off"
    if not want_assist:
        return f"Live 3D: {score_bit}; reproject assist off (pass --3d-reproject-assist)."
    if not LIVE_3D_REPROJECT_ASSIST_ALLOWED:
        return f"Live 3D: {score_bit}; reproject assist blocked in code."
    stack = _load_3d_assist_stack()
    if stack is None:
        return f"Live 3D: {score_bit}; reproject assist unavailable (import failed)."
    state_fn = stack[5]
    state = state_fn()
    if state == "ready":
        return (
            f"Live 3D: {score_bit}; reproject assist ON "
            "(real stereo_extrinsics.json; local gates still apply)."
        )
    if state == "placeholder":
        return (
            f"Live 3D: {score_bit}; reproject assist requested but "
            "stereo_extrinsics.json is PLACEHOLDER — run stereo_calib_stub.py capture/calibrate."
        )
    return (
        f"Live 3D: {score_bit}; reproject assist requested but "
        "stereo_extrinsics.json missing — run stereo_calib_stub.py howto."
    )


def _load_3d_identify_stack():
    """Lazy import for live score-vs-model — returns None while size-score is off."""
    global _3D_IDENTIFY_STACK, _3D_IDENTIFY_DIMS
    if not LIVE_3D_VIAL_ENABLED:
        return None
    if _3D_IDENTIFY_STACK is not None:
        return _3D_IDENTIFY_STACK if _3D_IDENTIFY_STACK is not False else None
    try:
        from vial_3d_identify import (  # type: ignore
            DEFAULT_LIVE_SCORE_MIN,
            filter_boxes_by_3d_model,
            lid_diameter_match_at_z,
            rank_key_3d_model,
            score_detection_vs_3d,
        )
        from vial_3d_model import (  # type: ignore
            is_topdown_box_aspect,
            load_dims,
        )
    except ImportError:
        _3D_IDENTIFY_STACK = False
        return None
    _3D_IDENTIFY_DIMS = load_dims()
    # Order (7): filter, rank, score, default_min, load_dims, lid@Z, topdown_aspect
    _3D_IDENTIFY_STACK = (
        filter_boxes_by_3d_model,
        rank_key_3d_model,
        score_detection_vs_3d,
        DEFAULT_LIVE_SCORE_MIN,
        load_dims,
        lid_diameter_match_at_z,
        is_topdown_box_aspect,
    )
    return _3D_IDENTIFY_STACK


def _lid_match_at_z(width_px, distance_mm, intrinsics, *, tol: float = DEFAULT_LID_WIDTH_TOL):
    """(ok, score) for lid width@Z vs fitted Ø; False if 3D stack unavailable."""
    stack = _load_3d_identify_stack()
    if not stack or intrinsics is None:
        return False, 0.0
    match_fn = stack[5]
    return match_fn(
        width_px,
        distance_mm,
        intrinsics,
        _dims_for_3d_score(),
        tol=tol,
    )


def set_allow_sideways(enabled: bool) -> None:
    """Enable/disable live sideways/fallen pose exceptions (process-wide)."""
    global _ALLOW_SIDEWAYS
    _ALLOW_SIDEWAYS = bool(enabled)


def allow_sideways_enabled() -> bool:
    return bool(_ALLOW_SIDEWAYS)


def set_sideways_assist(enabled: bool) -> None:
    """Enable/disable horizontal cap→cup growth + wide-YOLO sideways expand."""
    global _SIDEWAYS_ASSIST
    _SIDEWAYS_ASSIST = bool(enabled)


def sideways_assist_enabled() -> bool:
    return bool(_SIDEWAYS_ASSIST) and allow_sideways_enabled()


def detect_vial_by_sideways_winner(
    image, depth_mm, intrinsics=None
) -> list[tuple[int, int, int, int]]:
    """Run the promoted cap-first + depth-hole sideways detector when enabled."""
    if not sideways_assist_enabled() or _detect_sideways_race_winner is None:
        return []
    return [
        tuple(int(v) for v in box)
        for box in (_detect_sideways_race_winner(image, depth_mm, intrinsics) or [])
    ]


def _is_topdown_box(box) -> bool:
    stack = _load_3d_identify_stack()
    if stack:
        return bool(stack[6](box))
    _, _, w, h = box
    if w < 1:
        return False
    aspect = h / float(w)
    return 0.72 <= aspect <= 1.28


def _is_sideways_box(box) -> bool:
    """Clearly horizontal lid+cup (fallen / on-side). h/w ~0.40–0.71.

    Pure geometry. Callers should prefer upright (1.0–1.65), then squat top-down
    (0.72–0.99), and sideways only when w≫h (and usually lid-on-left/right).
    """
    _, _, w, h = box
    if w < 1 or h < 1:
        return False
    aspect = h / float(w)
    return SIDEWAYS_ASPECT_MIN <= aspect <= SIDEWAYS_ASPECT_MAX


def _is_squat_topdown_box(box) -> bool:
    """Top-down lid face below upright aspect min (does not steal upright 1.0–1.65)."""
    if is_full_vial_aspect(box):
        return False
    if not _is_topdown_box(box):
        return False
    return box_aspect_hw(box) < FULL_VIAL_ASPECT_MIN


def _end_cap_lr_score(image, box) -> float:
    """How clearly a white lid sits on the left or right end (0..1), not top/bottom."""
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    if w < 12 or h < 8:
        return 0.0
    hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    vband = max(3, int(w * 0.34))
    if w <= vband * 2:
        return 0.0
    left = float(white[:, :vband].mean())
    right = float(white[:, -vband:].mean())
    mid = float(white[:, vband : w - vband].mean())
    hband = max(3, int(h * 0.34))
    top = float(white[:hband].mean())
    bot = float(white[-hband:].mean())
    end = max(left, right)
    # Need a bright end, darker mid, and end brighter than top/bottom lid cue.
    if end < 0.28 or mid > end - 0.10:
        return 0.0
    return float(max(0.0, min(1.0, end - max(mid, top, bot) + 0.15)))


def _sideways_pose(
    box,
    gates: CameraGateProfile | None = None,
    image=None,
    *,
    require_end_cap: bool = False,
) -> bool:
    """Live sideways: flag on, clearly horizontal, not upright.

    When ``image`` is set (or ``require_end_cap``), also need lid-on-left/right
    geometry so squat upright-ish mistakes do not take the sideways path.
    """
    if not allow_sideways_enabled():
        return False
    if is_full_vial_aspect(box, gates=gates):
        return False
    if not _is_sideways_box(box):
        return False
    if image is not None or require_end_cap:
        if image is None:
            return False
        return _end_cap_lr_score(image, box) >= 0.12
    return True


def _pose_rank(box, gates: CameraGateProfile | None = None) -> int:
    """Higher = prefer in choose_box. Upright ≫ squat top-down ≫ sideways."""
    if is_full_vial_aspect(box, gates=gates):
        return 2
    if _is_squat_topdown_box(box):
        return 1
    if _sideways_pose(box, gates=gates):
        return 0
    return 0


def _dims_for_3d_score():
    stack = _load_3d_identify_stack()
    if not stack:
        return None
    global _3D_IDENTIFY_DIMS
    if _3D_IDENTIFY_DIMS is None:
        _3D_IDENTIFY_DIMS = stack[4]()
    return _3D_IDENTIFY_DIMS


def score_box_vs_3d_prior(
    box,
    depth_mm,
    image,
    intrinsics,
    *,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    """0..1 match to fitted vial at measured white-cap Z; None if unscored."""
    stack = _load_3d_identify_stack()
    if not stack or intrinsics is None:
        return None
    (
        _filter,
        _rank,
        score_fn,
        _min,
        _load,
        _lid_match,
        _is_topdown,
    ) = stack
    distance, _ = measure_cap_mm(depth_mm, image, box)
    return score_fn(
        box,
        distance,
        intrinsics,
        _dims_for_3d_score(),
        image_size=image_size,
    )


def apply_3d_model_score_filter(
    boxes,
    depth_mm,
    image,
    intrinsics,
    *,
    enabled: bool = True,
    min_score: float = DEFAULT_3D_MODEL_SCORE_MIN,
    image_size: tuple[int, int] | None = None,
):
    """Mild gate: drop size outliers vs 3D prior; never wipe to empty if all fail."""
    if not enabled or not boxes or intrinsics is None:
        return list(boxes)
    stack = _load_3d_identify_stack()
    if not stack:
        return list(boxes)
    (
        filter_fn,
        _rank,
        _score,
        default_min,
        _load,
        _lid_match,
        _is_topdown,
    ) = stack
    floor = float(min_score if min_score is not None else default_min)

    def distance_for_box(box):
        distance, _ = measure_cap_mm(depth_mm, image, box)
        return distance

    return filter_fn(
        boxes,
        distance_for_box=distance_for_box,
        K=intrinsics,
        dims=_dims_for_3d_score(),
        min_score=floor,
        image_size=image_size,
    )


def accept_proposed_box(
    cam: "OrbbecDepthCamera",
    color,
    depth_mm: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    soft_clear: bool = True,
) -> VialFix | None:
    """Gate a geometrically proposed box with local clear-vial rules (no YOLO invent).

    Used only by opt-in ``--3d-reproject-assist``. Distances are measured on this
    camera's white cap — never copied from the peer.
    """
    height, width = color.shape[:2]
    depth_mm = align_depth(depth_mm, width, height)
    if cam.intrinsics is None and cam.capture is not None:
        cam.intrinsics = intrinsics_from_capture(cam.capture, width, height)
    fitted = fit_vial_shape(color, depth_mm, box)
    fitted = clip_box(fitted, width, height)
    if not (
        accepts_vial_box(
            depth_mm,
            color,
            fitted,
            peer_confirmed=False,
            soft_clear=soft_clear,
            gates=cam.gates,
            intrinsics=cam.intrinsics,
        )
        and has_interior_structure(color, fitted, gates=cam.gates)
        and vial_scale(depth_mm, color, fitted, cam.intrinsics)
        and within_reach(depth_mm, color, fitted)
    ):
        return None
    kept = apply_3d_model_score_filter(
        [fitted],
        depth_mm,
        color,
        cam.intrinsics,
        enabled=cam.use_3d_model_score,
        min_score=cam.model_score_min,
        image_size=(width, height),
    )
    if not kept:
        return None
    fitted = kept[0]
    cam.misses = 0
    cam.miss_reason = "no_vial"
    cam.box = smooth_box(cam.box, fitted)
    box = clip_box(cam.box, width, height)
    measured, sample_box = measure_cap_mm(depth_mm, color, box)
    if measured is not None:
        cam.depth_samples.append(measured)
        del cam.depth_samples[:-5]
        cam.distance_mm = float(np.median(cam.depth_samples))
    model_score = None
    if cam.use_3d_model_score:
        model_score = score_box_vs_3d_prior(
            box, depth_mm, color, cam.intrinsics, image_size=(width, height)
        )
        if cam.distance_mm is not None and cam.intrinsics is not None:
            stack = _load_3d_identify_stack()
            if stack:
                model_score = stack[2](
                    box,
                    cam.distance_mm,
                    cam.intrinsics,
                    _dims_for_3d_score(),
                    image_size=(width, height),
                )
    cam.last_model_score = model_score
    if sample_box is not None:
        sx, sy, sw, sh = sample_box
        center = (sx + sw // 2, sy + sh // 2)
    else:
        center = (box[0] + box[2] // 2, box[1] + box[3] // 2)
    x_mm, y_mm, _slant = position_mm(center, cam.distance_mm, cam.intrinsics)
    return VialFix(
        box, cam.distance_mm, x_mm, y_mm, cam.distance_mm, False, sample_box, model_score
    )


def try_3d_reproject_assist(
    miss_cam: "OrbbecDepthCamera",
    miss_frame: tuple | None,
    peer_cam: "OrbbecDepthCamera",
    peer_fix: VialFix | None,
    *,
    soft_clear: bool = True,
) -> VialFix | None:
    """If peer has a vial, reproject parametric 3D model into miss_cam and gate it.

    Requires ``stereo_extrinsics.json`` with real calib. PLACEHOLDER R,t is refused
    (would mis-aim). Returns None when missing/placeholder extrinsics / gates reject.
    """
    stack = _load_3d_assist_stack()
    if stack is None or peer_fix is None or peer_fix.distance_mm is None:
        return None
    if miss_frame is None:
        return None
    color, depth_mm = miss_frame
    if color is None or depth_mm is None:
        return None
    (
        assist_box_from_peer,
        default_intrinsics,
        load_dims,
        load_extrinsics,
        is_placeholder,
        _state_fn,
    ) = stack
    if is_placeholder():
        return None
    extrinsics = load_extrinsics()
    if extrinsics is None:
        return None
    height, width = color.shape[:2]
    if miss_cam.intrinsics is None and miss_cam.capture is not None:
        miss_cam.intrinsics = intrinsics_from_capture(miss_cam.capture, width, height)
    if peer_cam.intrinsics is None:
        # Peer may be same process (classic dual) or unknown (independent) —
        # fall back to FOV prior when OpenCV props are unavailable.
        peer_K = default_intrinsics(width, height)
    else:
        peer_K = peer_cam.intrinsics
    local_K = miss_cam.intrinsics or default_intrinsics(width, height)
    proposed = assist_box_from_peer(
        tuple(int(v) for v in peer_fix.box),
        float(peer_fix.distance_mm),
        peer_K,
        local_K,
        (width, height),
        peer_camera=int(peer_cam.camera_index),
        local_camera=int(miss_cam.camera_index),
        dims=load_dims(),
        extrinsics=extrinsics,
    )
    if proposed is None:
        return None
    return accept_proposed_box(
        miss_cam, color, depth_mm, proposed, soft_clear=soft_clear
    )


def apply_3d_reproject_pair(
    cam_a: "OrbbecDepthCamera",
    frame_a: tuple | None,
    fix_a: VialFix | None,
    cam_b: "OrbbecDepthCamera",
    frame_b: tuple | None,
    fix_b: VialFix | None,
) -> tuple[VialFix | None, VialFix | None, bool]:
    """Fill one missing fix via opt-in 3D reproject; returns (fix_a, fix_b, used)."""
    used = False
    if fix_a is None and fix_b is not None:
        helped = try_3d_reproject_assist(
            cam_a, frame_a, cam_b, fix_b, soft_clear=True
        )
        if helped is not None:
            fix_a = helped
            used = True
    if fix_b is None and fix_a is not None:
        helped = try_3d_reproject_assist(
            cam_b, frame_b, cam_a, fix_a, soft_clear=True
        )
        if helped is not None:
            fix_b = helped
            used = True
    return fix_a, fix_b, used


@dataclass
class _PeerCamStub:
    """Minimal peer stand-in so IPC reproject can reuse ``try_3d_reproject_assist``."""

    camera_index: int
    intrinsics: tuple | None = None
    capture: object | None = None


def try_3d_reproject_from_peer_report(
    miss_cam: "OrbbecDepthCamera",
    miss_frame: tuple | None,
    peer: "PeerReport | None",
    *,
    soft_clear: bool = True,
    stale_s: float = PEER_STALE_S,
    now: float | None = None,
) -> VialFix | None:
    """Independent-dual path: reproject from peer IPC box+cm when local locate missed.

    Not classic sticky soft-locate — requires real extrinsics + local gate accept.
    """
    if peer is None or not peer.has_vial or peer.box is None or peer.distance_cm is None:
        return None
    stamp = time.time() if now is None else float(now)
    if stale_s > 0 and (stamp - float(peer.timestamp)) > stale_s:
        return None
    peer_fix = VialFix(
        box=tuple(int(v) for v in peer.box),
        distance_mm=float(peer.distance_cm) * 10.0,
        x_mm=None,
        y_mm=None,
        z_mm=float(peer.distance_cm) * 10.0,
        held=False,
    )
    peer_cam = _PeerCamStub(camera_index=int(peer.camera_id))
    return try_3d_reproject_assist(
        miss_cam, miss_frame, peer_cam, peer_fix, soft_clear=soft_clear
    )


@dataclass
class PeerReport:
    """One camera's published state for distance corroboration (+ optional ORB)."""

    timestamp: float
    camera_id: int
    has_vial: bool
    box: tuple[int, int, int, int] | None = None
    distance_cm: float | None = None
    conf: float | None = None
    # Compact ORB descriptors from white-cap crop (list of 32-byte rows as ints).
    orb_desc: list[list[int]] | None = None

    def to_payload(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "camera_id": int(self.camera_id),
            "has_vial": bool(self.has_vial),
            "box": list(self.box) if self.box is not None else None,
            "distance_cm": None if self.distance_cm is None else float(self.distance_cm),
            "conf": None if self.conf is None else float(self.conf),
            "orb_desc": self.orb_desc,
        }


def encode_peer_report(report: PeerReport) -> bytes:
    return json.dumps(report.to_payload(), separators=(",", ":")).encode("utf-8")


def parse_peer_report(data: bytes | str | dict | None) -> PeerReport | None:
    """Parse a peer IPC payload. Returns None on empty/invalid input."""
    if data is None:
        return None
    try:
        if isinstance(data, dict):
            payload = data
        else:
            text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
            text = text.strip()
            if not text:
                return None
            payload = json.loads(text)
        if not isinstance(payload, dict):
            return None
        camera_id = int(payload["camera_id"])
        has_vial = bool(payload["has_vial"])
        box_raw = payload.get("box")
        box = None
        if box_raw is not None:
            if not isinstance(box_raw, (list, tuple)) or len(box_raw) != 4:
                return None
            box = tuple(int(round(float(v))) for v in box_raw)
        dist = payload.get("distance_cm")
        conf = payload.get("conf")
        orb_raw = payload.get("orb_desc")
        orb_desc = None
        if orb_raw is not None:
            if not isinstance(orb_raw, list) or len(orb_raw) > ORB_IPC_MAX_FEATURES:
                return None
            orb_desc = []
            for row in orb_raw:
                if not isinstance(row, (list, tuple)) or len(row) != 32:
                    return None
                orb_desc.append([int(v) & 0xFF for v in row])
        return PeerReport(
            timestamp=float(payload.get("timestamp", 0.0)),
            camera_id=camera_id,
            has_vial=has_vial,
            box=box,
            distance_cm=None if dist is None else float(dist),
            conf=None if conf is None else float(conf),
            orb_desc=orb_desc,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class CorroborationResult:
    """Compare local vs peer distances — never mutates either measurement."""

    status: str  # agree | disagree | missing_peer | peer_no_vial | local_no_vial | incomplete
    local_cm: float | None = None
    peer_cm: float | None = None
    delta_cm: float | None = None
    label: str = ""

    @property
    def overlay_line(self) -> str:
        return self.label or f"peer {self.status.replace('_', ' ')}"


def corroborate_distances(
    local: PeerReport | None,
    peer: PeerReport | None,
    *,
    abs_cm: float = CORROBORATE_ABS_CM,
    rel: float = CORROBORATE_REL,
    now: float | None = None,
    stale_s: float = PEER_STALE_S,
) -> CorroborationResult:
    """Validate peer distance against local. Does not average or replace local."""
    local_cm = None if local is None or not local.has_vial else local.distance_cm
    if peer is None:
        return CorroborationResult(
            status="missing_peer",
            local_cm=local_cm,
            label="no peer",
        )
    stamp = time.time() if now is None else float(now)
    if stale_s > 0 and (stamp - float(peer.timestamp)) > stale_s:
        return CorroborationResult(
            status="missing_peer",
            local_cm=local_cm,
            label="no peer",
        )
    if local is None or not local.has_vial:
        return CorroborationResult(
            status="local_no_vial",
            peer_cm=peer.distance_cm if peer.has_vial else None,
            label="peer ok" if peer.has_vial else "peer no vial",
        )
    if not peer.has_vial:
        return CorroborationResult(
            status="peer_no_vial",
            local_cm=local.distance_cm,
            label="peer no vial",
        )
    if local.distance_cm is None or peer.distance_cm is None:
        return CorroborationResult(
            status="incomplete",
            local_cm=local.distance_cm,
            peer_cm=peer.distance_cm,
            label="peer incomplete",
        )
    delta = abs(float(local.distance_cm) - float(peer.distance_cm))
    scale = max(abs(float(local.distance_cm)), abs(float(peer.distance_cm)), 1e-6)
    relative = delta / scale
    agree = delta <= float(abs_cm) or relative <= float(rel)
    if agree:
        return CorroborationResult(
            status="agree",
            local_cm=float(local.distance_cm),
            peer_cm=float(peer.distance_cm),
            delta_cm=delta,
            label=f"peer agree Δ{delta:.1f}cm",
        )
    return CorroborationResult(
        status="disagree",
        local_cm=float(local.distance_cm),
        peer_cm=float(peer.distance_cm),
        delta_cm=delta,
        label=f"peer disagree Δ{delta:.1f}cm",
    )


def peer_report_from_fix(
    camera_id: int,
    fix: "VialFix | None",
    *,
    conf: float | None = None,
    now: float | None = None,
    image=None,
    use_orb: bool = False,
) -> PeerReport:
    has = fix is not None
    dist_cm = None
    box = None
    orb_desc = None
    if fix is not None:
        box = tuple(int(v) for v in fix.box)
        if fix.distance_mm is not None:
            dist_cm = float(fix.distance_mm) / 10.0
        if use_orb and image is not None:
            orb_desc = pack_orb_descriptors(
                extract_orb_descriptors(image, box)[1]
            )
    return PeerReport(
        timestamp=time.time() if now is None else float(now),
        camera_id=int(camera_id),
        has_vial=has,
        box=box,
        distance_cm=dist_cm,
        conf=None if not has else conf,
        orb_desc=orb_desc,
    )


def extract_orb_descriptors(
    image,
    box=None,
    *,
    max_features: int = ORB_MAX_FEATURES,
) -> tuple[list | None, np.ndarray | None]:
    """ORB on white-cap ROI (preferred) or full frame. Soft-fail → (None, None).

    Clear plastic is texture-poor — few keypoints is normal. Never raises; never
    invents a detection box from matches alone.
    """
    try:
        if image is None or image.size == 0:
            return None, None
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image
        )
        roi = gray
        if box is not None:
            height, width = gray.shape[:2]
            x, y, w, h = clip_box(box, width, height)
            # Prefer the brighter end (white cap) as a tighter crop.
            hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
            white = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 160)
            if white.any():
                if _sideways_pose((x, y, w, h)):
                    band = max(4, int(w * 0.38))
                    left = float(white[:, :band].mean())
                    right = float(white[:, -band:].mean())
                    if left >= right:
                        x1, x2 = x, x + band
                    else:
                        x1, x2 = x + w - band, x + w
                    y1, y2 = y, y + h
                else:
                    band = max(4, int(h * 0.38))
                    top = float(white[:band].mean())
                    bot = float(white[-band:].mean())
                    if top >= bot:
                        y1, y2 = y, y + band
                    else:
                        y1, y2 = y + h - band, y + h
                    x1, x2 = x, x + w
                # Pad slightly so ORB has rim texture.
                pad = max(2, min(w, h) // 10)
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    roi = gray[y1:y2, x1:x2]
            else:
                roi = gray[y : y + h, x : x + w]
        if roi.size == 0 or min(roi.shape[:2]) < 8:
            return None, None
        orb = cv2.ORB_create(nfeatures=int(max_features))
        keypoints, descriptors = orb.detectAndCompute(roi, None)
        if descriptors is None or len(keypoints) < 2:
            return None, None
        return keypoints, descriptors
    except cv2.error:
        return None, None


def pack_orb_descriptors(descriptors: np.ndarray | None) -> list[list[int]] | None:
    """Serialize ORB uint8 descriptors for PeerReport IPC (tiny payload)."""
    if descriptors is None or getattr(descriptors, "size", 0) == 0:
        return None
    rows = np.asarray(descriptors, dtype=np.uint8)
    if rows.ndim != 2 or rows.shape[1] != 32:
        return None
    return rows[:ORB_IPC_MAX_FEATURES].tolist()


def unpack_orb_descriptors(payload: list[list[int]] | None) -> np.ndarray | None:
    if not payload:
        return None
    try:
        arr = np.asarray(payload, dtype=np.uint8)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 32 or arr.shape[0] == 0:
        return None
    return arr


def match_orb_descriptors(
    desc_a: np.ndarray | None,
    desc_b: np.ndarray | None,
    *,
    ratio: float = ORB_RATIO_TEST,
) -> tuple[int, float]:
    """BFMatcher Hamming + Lowe ratio test. Returns (n_good, score 0..1).

    Empty / weak matches → (0, 0.0). Soft fail only — never invents boxes.
    """
    if desc_a is None or desc_b is None:
        return 0, 0.0
    if len(desc_a) < 2 or len(desc_b) < 2:
        return 0, 0.0
    try:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return 0, 0.0
    good = 0
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < float(ratio) * second.distance:
            good += 1
    # Soft score: saturates around ~12 good matches (clear plastic rarely has more).
    score = min(1.0, good / 12.0)
    return good, float(score)



def orb_agree_label(
    local_desc: np.ndarray | None,
    peer_desc: np.ndarray | None,
) -> str | None:
    """Overlay fragment when peer ORB agrees; None when weak (behave as today)."""
    n_good, score = match_orb_descriptors(local_desc, peer_desc)
    if n_good >= ORB_AGREE_GOOD_MATCHES and score >= ORB_AGREE_SCORE_MIN:
        return f"orb agree ({n_good})"
    return None


class PeerIpc:
    """Localhost UDP publish/subscribe for dual-independent corroboration only."""

    def __init__(
        self,
        camera_id: int,
        peer_id: int,
        *,
        base_port: int = IPC_BASE_PORT,
        host: str = "127.0.0.1",
    ):
        self.camera_id = int(camera_id)
        self.peer_id = int(peer_id)
        self.host = host
        self.listen_port = int(base_port) + self.camera_id
        self.peer_port = int(base_port) + self.peer_id
        self._latest: PeerReport | None = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, self.listen_port))
        self.sock.setblocking(False)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def publish(self, report: PeerReport) -> None:
        payload = encode_peer_report(report)
        try:
            self.sock.sendto(payload, (self.host, self.peer_port))
        except OSError:
            pass

    def poll(self) -> PeerReport | None:
        """Drain the socket; keep the newest valid peer report."""
        while True:
            try:
                data, _addr = self.sock.recvfrom(16384)
            except BlockingIOError:
                break
            except OSError:
                break
            parsed = parse_peer_report(data)
            if parsed is None:
                continue
            if parsed.camera_id != self.peer_id:
                continue
            self._latest = parsed
        return self._latest

    def latest(self, *, now: float | None = None, stale_s: float = PEER_STALE_S) -> PeerReport | None:
        report = self.poll()
        if report is None:
            return None
        stamp = time.time() if now is None else float(now)
        if stale_s > 0 and (stamp - float(report.timestamp)) > stale_s:
            return None
        return report


def box_aspect_hw(box) -> float:
    """Height / width of an xywh box."""
    _, _, bw, bh = box
    return float(bh) / max(float(bw), 1.0)


def is_full_vial_aspect(box, gates: CameraGateProfile | None = None) -> bool:
    """True when the box looks like lid through cup bottom, not lid-only."""
    if gates is not None:
        return gates.is_full_vial_aspect(box)
    aspect = box_aspect_hw(box)
    return FULL_VIAL_ASPECT_MIN <= aspect <= FULL_VIAL_ASPECT_MAX


def is_review_acceptable_box(box, gates: CameraGateProfile | None = None) -> bool:
    """True for a review R / I save: upright, top-down lid, or sideways lid+cup.

    Review labeling accepts sideways geometrically for dataset capture. Live detect
    is upright-first and only takes sideways when clearly horizontal. Flat strips out.
    """
    if is_full_vial_aspect(box, gates=gates):
        return True
    return _is_topdown_box(box) or _is_sideways_box(box)


def review_box_reject_message(box) -> str:
    """Plain-language why R / I save refused this box."""
    aspect = box_aspect_hw(box)
    if (
        aspect < FULL_VIAL_ASPECT_MIN
        and not _is_topdown_box(box)
        and not _is_sideways_box(box)
    ):
        return (
            f"Box is almost flat (h/w={aspect:.2f}) — looks like a lid-only crop, "
            f"not a full side-view vial (need ~{FULL_VIAL_ASPECT_MIN:.1f}–"
            f"{FULL_VIAL_ASPECT_MAX:.1f}), sideways lid+cup "
            f"(~{SIDEWAYS_ASPECT_MIN:.2f}–{SIDEWAYS_ASPECT_MAX:.2f}), or a "
            f"near-square top-down lid (~0.72–1.28). Press I and drag the full vial."
        )
    if aspect > FULL_VIAL_ASPECT_MAX:
        return (
            f"Box is too tall (h/w={aspect:.2f}; need ≤~{FULL_VIAL_ASPECT_MAX:.1f}). "
            "Tighten the box around the vial, then save again."
        )
    # Near-square that somehow still failed (shouldn't happen if gate is consistent).
    return (
        f"Box looks wrong (h/w={aspect:.2f}). "
        f"Side view: lid through cup (~{FULL_VIAL_ASPECT_MIN:.1f}–"
        f"{FULL_VIAL_ASPECT_MAX:.1f}). Top-down: near-square lid (~0.72–1.28). "
        f"Sideways: elongated lid+cup (~{SIDEWAYS_ASPECT_MIN:.2f}–"
        f"{SIDEWAYS_ASPECT_MAX:.2f})."
    )


@dataclass
class VialFix:
    box: tuple[int, int, int, int]
    distance_mm: float | None
    x_mm: float | None
    y_mm: float | None
    z_mm: float | None
    held: bool = False
    sample_box: tuple[int, int, int, int] | None = None
    # 0..1 match to vial_3d_dims.json at measured Z (None = unscored / disabled).
    model_score: float | None = None
    track_id: int | None = None

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2


@dataclass
class VialTrack:
    """Persistent local identity, ROI, miss count, and depth history for one vial."""

    track_id: int
    box: tuple[float, float, float, float]
    misses: int = 0
    depth_samples: list[float] = field(default_factory=list)
    distance_mm: float | None = None
    sample_box: tuple[int, int, int, int] | None = None


def find_weights(explicit: Path | None) -> Path | None:
    """Resolve YOLO weights for live detection.

    Explicit paths win. Otherwise prefer ``runs/vial-live`` (train-race /
    fine-tune export), then ``runs/train-race/r1-winner``, then
    ``runs/bon-winner`` (Attempt E). Do **not** pick the newest ``best.pt``
    under ``runs/`` — BoN scratch / nano experiments are often newer and worse
    on this tiny clear-vial set.
    """
    if explicit is not None:
        return explicit if explicit.exists() else None
    for preferred in (LIVE_WEIGHTS, R1_WINNER_WEIGHTS, BON_WINNER_WEIGHTS):
        if preferred.exists():
            return preferred
    runs = ROOT / "runs"
    if not runs.exists():
        return None
    candidates = list(runs.rglob("best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x = round(x)
    y = round(y)
    w = round(max(1, w))
    h = round(max(1, h))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return x, y, w, h


def _white_mask(value, saturation, minimum_value: float) -> np.ndarray:
    mask = ((value >= minimum_value) & (saturation <= 55)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _cap_mask(value, saturation) -> np.ndarray:
    """Keep the white cap without swallowing a bright background."""
    best = _white_mask(value, saturation, 200)
    best_fraction = float(best.mean() / 255)
    if 0.002 <= best_fraction <= 0.22:
        return best
    for minimum in (185, 170, 210, 220, 230):
        trial = _white_mask(value, saturation, minimum)
        fraction = float(trial.mean() / 255)
        if 0.002 <= fraction <= 0.22 and abs(fraction - 0.06) < abs(best_fraction - 0.06):
            best = trial
            best_fraction = fraction
    return best


def skin_mask(image) -> np.ndarray:
    """Skin is pink in YCrCb. The white lid stays near neutral gray."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    return (
        (cr >= 137)
        & (cr <= 175)
        & (cb >= 85)
        & (cb <= 130)
        & (sat >= 30)
        & (val >= 50)
        & ((hue <= 22) | (hue >= 168))
    )


def _region_skin(skin: np.ndarray, box) -> float:
    height, width = skin.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    roi = skin[y : y + h, x : x + w]
    if roi.size == 0:
        return 1.0
    return float(roi.mean())


def _beside_white_background(value, saturation, x, y, bw, bh, width) -> bool:
    """A lid has edges. A table or wall stays white on both sides of the blob."""
    row = y + max(0, bh // 2)
    if row >= value.shape[0]:
        return False
    margin = max(6, bw // 2)
    sides = []
    if x - margin >= 0:
        side_v = value[row, x - margin : x]
        side_s = saturation[row, x - margin : x]
        sides.append(float(((side_v >= 170) & (side_s <= 55)).mean()))
    if x + bw + margin <= width:
        side_v = value[row, x + bw : x + bw + margin]
        side_s = saturation[row, x + bw : x + bw + margin]
        sides.append(float(((side_v >= 170) & (side_s <= 55)).mean()))
    return len(sides) == 2 and min(sides) > 0.6


def _expand_lid(value, saturation, x, y, bw, bh, width, height):
    """Grow only across the white lid. Stop where the white area keeps going, that is the background."""
    max_h = max(bh + 2, int(max(bw, 1) * 0.55))

    def row_is_lid(row, left, right) -> bool:
        if row < 0 or row >= height or right <= left:
            return False
        row_v = value[row, left:right]
        row_s = saturation[row, left:right]
        return float(((row_v >= 165) & (row_s <= 55)).mean()) >= 0.45

    def outside_is_background(row, left, right) -> bool:
        margin = max(4, (right - left) // 3)
        strips = []
        if left - margin >= 0:
            strips.append((value[row, left - margin : left], saturation[row, left - margin : left]))
        if right + margin <= width:
            strips.append((value[row, right : right + margin], saturation[row, right : right + margin]))
        if not strips:
            return False
        white_sides = [
            float(((strip_v >= 170) & (strip_s <= 55)).mean())
            for strip_v, strip_s in strips
            if strip_v.size
        ]
        return bool(white_sides) and max(white_sides) > 0.65

    top = y
    while top > 0 and (y + bh - top) < max_h and row_is_lid(top - 1, x, x + bw):
        if outside_is_background(top - 1, x, x + bw):
            break
        top -= 1
    bottom = y + bh
    while bottom < height and (bottom - top) < max_h and row_is_lid(bottom, x, x + bw):
        if outside_is_background(bottom, x, x + bw):
            break
        bottom += 1
    return x, top, bw, max(1, bottom - top)


def _cup_rows(value, saturation, edges, x, bw, start, step, max_rows, height, width, skin=None) -> int:
    """Follow the clear cup until its side walls end. Do not continue onto the table."""
    traveled = 0
    gap = 0
    side = max(2, bw // 8)
    for index in range(max_rows):
        row = start + index * step
        if row < 0 or row >= height:
            break
        segment_v = value[row, x : x + bw]
        segment_s = saturation[row, x : x + bw]
        if segment_v.size == 0:
            break
        if skin is not None:
            skin_row = skin[row, x : x + bw]
            if skin_row.size and float(skin_row.mean()) > 0.35:
                break
        margin = max(4, bw // 3)
        outside_white = []
        if x - margin >= 0:
            outside_v = value[row, x - margin : x]
            outside_s = saturation[row, x - margin : x]
            outside_white.append(float(((outside_v >= 170) & (outside_s <= 55)).mean()))
        if x + bw + margin <= width:
            outside_v = value[row, x + bw : x + bw + margin]
            outside_s = saturation[row, x + bw : x + bw + margin]
            outside_white.append(float(((outside_v >= 170) & (outside_s <= 55)).mean()))
        walls = False
        if edges is not None:
            left_wall = edges[row, x : x + side]
            right_wall = edges[row, x + bw - side : x + bw]
            walls = bool(left_wall.any() or right_wall.any())
        center_bright = float(segment_v.mean()) >= 90
        if outside_white and max(outside_white) > 0.7 and not walls and not center_bright:
            gap += 1
            if gap > 3:
                break
            continue
        if float(segment_v.mean()) < 45 and not walls:
            gap += 1
            if gap > 2:
                break
            continue
        if not walls and traveled > 6:
            gap += 1
            if gap > 3:
                break
            continue
        gap = 0
        traveled += 1
    return traveled


def _cup_cols(
    value,
    saturation,
    edges,
    y,
    bh,
    start,
    step,
    max_cols,
    height,
    width,
    skin=None,
    *,
    depth_mm=None,
    cap_depth: float | None = None,
    gy=None,
) -> int:
    """Follow the clear cup left/right of a sideways lid until side walls end.

    Mirrors upright ``_cup_rows`` / ``fit_vial_shape`` cup_side: Canny walls,
    optional Sobel |Gy| on the horizontal rim, intensity continuity, and depth
    holes through clear plastic when ``depth_mm`` + ``cap_depth`` are set.
    """
    traveled = 0
    gap = 0
    side = max(2, bh // 8)
    for index in range(max_cols):
        col = start + index * step
        if col < 0 or col >= width:
            break
        segment_v = value[y : y + bh, col]
        segment_s = saturation[y : y + bh, col]
        if segment_v.size == 0:
            break
        if skin is not None:
            skin_col = skin[y : y + bh, col]
            if skin_col.size and float(skin_col.mean()) > 0.35:
                break
        # Far table behind the cup: stop when depth jumps well past the lid.
        if depth_mm is not None and cap_depth is not None:
            strip = depth_mm[y : y + bh, col]
            near = strip[(strip > 80) & (strip < MAX_DEPTH_MM)]
            if near.size >= 4 and float(np.median(near)) > float(cap_depth) + 350:
                break
        margin = max(4, bh // 3)
        outside_white = []
        if y - margin >= 0:
            outside_v = value[y - margin : y, col]
            outside_s = saturation[y - margin : y, col]
            outside_white.append(float(((outside_v >= 170) & (outside_s <= 55)).mean()))
        if y + bh + margin <= height:
            outside_v = value[y + bh : y + bh + margin, col]
            outside_s = saturation[y + bh : y + bh + margin, col]
            outside_white.append(float(((outside_v >= 170) & (outside_s <= 55)).mean()))
        walls = False
        if edges is not None:
            top_wall = edges[y : y + side, col]
            bot_wall = edges[y + bh - side : y + bh, col]
            walls = bool(top_wall.any() or bot_wall.any())
        # Soft |Gy| cue when Canny is thin on clear plastic.
        if not walls and gy is not None:
            top_gy = float(np.mean(np.abs(gy[y : y + side, col]))) if side > 0 else 0.0
            bot_gy = (
                float(np.mean(np.abs(gy[y + bh - side : y + bh, col]))) if side > 0 else 0.0
            )
            walls = top_gy >= 12.0 or bot_gy >= 12.0
        mean_v = float(segment_v.mean())
        # Clear cup body: mid intensity, low-ish sat (not opaque white slab).
        bodyish = mean_v >= 55 and mean_v <= 210 and float(segment_s.mean()) <= 90
        # Depth holes through cup count as continuing the body.
        depth_body = False
        if depth_mm is not None and cap_depth is not None:
            strip = depth_mm[y : y + bh, col]
            finite = np.isfinite(strip)
            valid = finite & (strip > 80) & (strip < MAX_DEPTH_MM)
            invalid = ~valid
            far = valid & (strip > float(cap_depth) + CLEAR_FAR_DELTA_MM)
            hole = float(np.count_nonzero(invalid | far)) / max(float(strip.size), 1.0)
            depth_body = hole >= 0.35
        center_bright = mean_v >= 90
        if outside_white and max(outside_white) > 0.7 and not walls and not center_bright:
            gap += 1
            if gap > 3:
                break
            continue
        if mean_v < 40 and not walls and not depth_body:
            gap += 1
            if gap > 2:
                break
            continue
        # Clear plastic often lacks strong edges — allow bodyish / depth continuation.
        if not walls and not bodyish and not depth_body and traveled > 4:
            gap += 1
            if gap > 4:
                break
            continue
        if not walls and traveled > 10 and not (bodyish or depth_body):
            gap += 1
            if gap > 3:
                break
            continue
        gap = 0
        traveled += 1
    return traveled


def _lid_end_side(image, box) -> str | None:
    """Which horizontal end holds the white lid: ``\"left\"``, ``\"right\"``, or None."""
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    if w < 10 or h < 8:
        return None
    hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    band = max(3, int(w * 0.34))
    if w <= band * 2:
        return None
    left = float(white[:, :band].mean())
    right = float(white[:, -band:].mean())
    mid = float(white[:, band : w - band].mean())
    if left >= right and left >= 0.28 and mid <= left - 0.08:
        return "left"
    if right > left and right >= 0.28 and mid <= right - 0.08:
        return "right"
    return None


def grow_sideways_from_lid(
    image,
    lid_box,
    depth_mm=None,
    *,
    prefer_side: str | None = None,
) -> tuple[int, int, int, int] | None:
    """Grow a clear cup horizontally from a white end-cap lid (sideways assist).

    Only for non-upright lids (not wide top/bottom upright strips). Returns a
    sideways-shaped box or None. Distances still come from the opaque lid later.
    """
    if not sideways_assist_enabled():
        return None
    height, width = image.shape[:2]
    x, y, bw, bh = clip_box(lid_box, width, height)
    if bw < 8 or bh < 8:
        return None
    # Upright lid strips (w≫h): never grow horizontally — that corrupts upright.
    lid_aspect = bw / max(bh, 1)
    if lid_aspect >= 1.35:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 100)
    _, gy, _ = _image_derivatives(gray)
    skin = skin_mask(image)
    if depth_mm is not None and depth_mm.shape[:2] != (height, width):
        depth_mm = align_depth(depth_mm, width, height)
    cap_depth = None
    if depth_mm is not None:
        cap_depth = _blob_depth_mm(depth_mm, x, y, bw, bh)
    inset_y = max(1, bh // 5)
    core_y = y + inset_y
    core_h = max(1, bh - 2 * inset_y)
    max_cup_x = int(max(bh, bw) * FIT_SIDEWAYS_LENGTH_RATIO)
    left = _cup_cols(
        value,
        saturation,
        edges,
        core_y,
        core_h,
        x - 1,
        -1,
        max_cup_x,
        height,
        width,
        skin,
        depth_mm=depth_mm,
        cap_depth=cap_depth,
        gy=gy,
    )
    right = _cup_cols(
        value,
        saturation,
        edges,
        core_y,
        core_h,
        x + bw,
        1,
        max_cup_x,
        height,
        width,
        skin,
        depth_mm=depth_mm,
        cap_depth=cap_depth,
        gy=gy,
    )
    pad = max(2, int(max(bw, bh) * 0.04))
    min_cup = max(8, int(max(bh, bw) * 0.22))
    side = prefer_side
    if side not in ("left", "right"):
        if left >= right and left >= min_cup:
            side = "left"
        elif right > left and right >= min_cup:
            side = "right"
        else:
            return None
    if side == "left" and left < min_cup:
        return None
    if side == "right" and right < min_cup:
        return None
    if side == "left":
        grown = (x - left, y - pad, bw + left + pad, bh + pad * 2)
    else:
        grown = (x - pad, y - pad, bw + right + pad, bh + pad * 2)
    grown = clip_box(grown, width, height)
    if is_full_vial_aspect(grown):
        return None
    if not _is_sideways_box(grown):
        return None
    if _end_cap_lr_score(image, grown) < 0.10:
        return None
    return grown


def expand_wide_proposal_sideways(
    image, depth_mm, box
) -> tuple[int, int, int, int] | None:
    """Expand a wide YOLO/proposal box along its long axis into a sideways lid+cup.

    Does not reject wide boxes — grows from the white end using horizontal cup
    cues. Upright full-vial boxes are left alone.
    """
    if not sideways_assist_enabled():
        return None
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    if w < 16 or h < 10:
        return None
    if is_full_vial_aspect((x, y, w, h)):
        return None
    aspect = h / float(w)
    # Need clearly wider than tall (fallen), not upright / top-down.
    if aspect >= 0.95 or w < h * 1.05:
        return None
    # Already a good sideways box — keep as-is.
    if _is_sideways_box((x, y, w, h)) and _end_cap_lr_score(image, (x, y, w, h)) >= 0.12:
        return (x, y, w, h)
    end = _lid_end_side(image, (x, y, w, h))
    if end is None:
        # Lid may fill most of a short proposal — try both ends as seed lids.
        band = max(8, int(min(w, h) * 0.55))
        left_lid = (x, y, min(band, w), h)
        right_lid = (x + max(0, w - band), y, min(band, w), h)
        for seed, prefer in ((left_lid, "right"), (right_lid, "left")):
            grown = grow_sideways_from_lid(
                image, seed, depth_mm, prefer_side=prefer
            )
            if grown is not None:
                return grown
        return None
    # Seed from the white end band, grow into the body.
    band = max(8, int(w * 0.32))
    if end == "left":
        lid = (x, y, min(band, w), h)
        prefer = "right"
    else:
        lid = (x + max(0, w - band), y, min(band, w), h)
        prefer = "left"
    grown = grow_sideways_from_lid(image, lid, depth_mm, prefer_side=prefer)
    if grown is not None:
        return grown
    # Soft fallback: keep original if it already looks sideways with an end lid.
    if _is_sideways_box((x, y, w, h)) and _end_cap_lr_score(image, (x, y, w, h)) >= 0.10:
        return (x, y, w, h)
    return None


def _sideways_soft_rank(
    box, image, depth_mm=None, intrinsics=None
) -> float:
    """Soft ranking among sideways candidates: lid-end clarity + body length@Z.

    Not a hard gate and not the live 3D size-score path — only breaks ties when
    no upright candidate exists.
    """
    if not _sideways_pose(box, image=image):
        return 0.0
    end = _end_cap_lr_score(image, box)
    length_score = 0.0
    if depth_mm is not None and intrinsics is not None:
        distance, _ = measure_cap_mm(depth_mm, image, box)
        if distance is not None and distance > MIN_SURFACE_MM:
            fx, fy, _, _ = intrinsics
            if fx > 1 and fy > 1:
                _, _, bw, bh = box
                long_mm = max(bw * distance / fx, bh * distance / fy)
                short_mm = min(bw * distance / fx, bh * distance / fy)
                # Soft peak near fitted H≈73 mm, Ø≈54 mm (no hard reject here).
                length_score = max(
                    0.0,
                    1.0 - abs(long_mm - 73.0) / 80.0,
                ) * 0.5 + max(0.0, 1.0 - abs(short_mm - 54.0) / 60.0) * 0.5
    return float(end * 0.65 + length_score * 0.35)


def _longest_white_run(columns) -> tuple[int, int]:
    best_start = best_end = -1
    run_start = None
    for index, bright in enumerate(columns):
        if bright and run_start is None:
            run_start = index
        elif not bright and run_start is not None:
            if run_start > best_start and index - run_start > best_end - best_start:
                best_start, best_end = run_start, index
            run_start = None
    if run_start is not None and len(columns) - run_start > best_end - best_start:
        best_start, best_end = run_start, len(columns)
    return best_start, best_end


def fit_vial_shape(image, depth_mm, box):
    """Shrink a loose box to the lid width and the clear cup, leaving out the background.

    Top-down / near-straight-up lid faces (nearly square) are left alone — shrinking
    them into a thin band destroys the lid@Z diameter cue.
    """
    height, width = image.shape[:2]
    x, y, bw, bh = clip_box(box, width, height)
    if bw < 16 or bh < 16:
        return box
    # Squat top-down / live sideways: do not shrink into a thin band.
    # Upright aspects (~1.0–1.65) that overlap the top-down band still get fitted.
    if _is_squat_topdown_box((x, y, bw, bh)) or _sideways_pose((x, y, bw, bh)):
        return (x, y, bw, bh)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 35, 100)
    band_h = max(3, min(bh // 5, max(8, bw // 8)))
    best = None
    step = max(1, band_h // 2)
    for row in range(y, y + bh - band_h + 1, step):
        region = white[row : row + band_h, x : x + bw]
        if region.size == 0:
            continue
        columns = region.mean(axis=0) >= 0.5
        start, end = _longest_white_run(columns)
        run = end - start
        if run < max(12, int(bw * 0.28)):
            continue
        score = run
        if best is None or score > best[0]:
            best = (score, row, start, end)
    if best is None:
        return box
    _, cap_row, start, end = best
    cap_x = x + start
    cap_w = max(8, end - start)
    cap_y = cap_row
    cap_h = band_h
    while cap_y > y and float(white[cap_y - 1, cap_x : cap_x + cap_w].mean()) >= 0.45:
        cap_y -= 1
        cap_h += 1
    bottom = cap_row + band_h
    while bottom < y + bh and float(white[bottom, cap_x : cap_x + cap_w].mean()) >= 0.45:
        bottom += 1
        cap_h = bottom - cap_y

    cap_depth = None
    if depth_mm is not None:
        cap_roi = depth_mm[cap_y : cap_y + max(1, cap_h), cap_x : cap_x + cap_w]
        valid = cap_roi[(cap_roi > 80) & (cap_roi < MAX_DEPTH_MM)]
        if valid.size >= 8:
            cap_depth = float(np.median(valid))

    def cup_side(start_row, step_row) -> int:
        traveled = 0
        gap = 0
        side = max(2, cap_w // 8)
        # Search far enough for lid+cup (~1.55x width); was 1.35 with a 1.2 clamp.
        limit = int(cap_w * FIT_MAX_HEIGHT_CAP_RATIO)
        for _ in range(limit):
            row = start_row + (traveled + gap) * step_row
            # A local fit must not jump from a vial into a tripod, monitor, or
            # another object outside the detector proposal. This previously
            # moved the right-vial box ~50 px above the physical vial.
            if row < y or row >= y + bh:
                break
            if depth_mm is not None and cap_depth is not None:
                strip = depth_mm[row, cap_x : cap_x + cap_w]
                near = strip[(strip > 80) & (strip < MAX_DEPTH_MM)]
                if near.size >= 6 and float(np.median(near)) > cap_depth + 350:
                    break
            walls = bool(
                edges[row, cap_x : cap_x + side].any()
                or edges[row, cap_x + cap_w - side : cap_x + cap_w].any()
            )
            if not walls and traveled > 3:
                gap += 1
                if gap > 1:
                    break
                continue
            gap = 0
            traveled += 1
        return traveled

    above = cup_side(cap_y - 1, -1)
    below = cup_side(cap_y + cap_h, 1)
    pad = max(1, int(cap_w * 0.04))
    if above >= below:
        fitted = (cap_x - pad, cap_y - above, cap_w + pad * 2, cap_h + above + pad)
    else:
        fitted = (cap_x - pad, cap_y - pad, cap_w + pad * 2, cap_h + below + pad)
    fitted = clip_box(fitted, width, height)
    fx, fy, fw, fh = fitted
    max_h = int(cap_w * FIT_MAX_HEIGHT_CAP_RATIO)
    if fh > max_h:
        if above >= below:
            fy = fy + fh - max_h
        fh = max_h
        fitted = (fx, fy, fw, fh)
    fitted = clip_box(fitted, width, height)
    fx, fy, fw, fh = fitted
    # Clear plastic often has weak edges, so cup_side can stop at the lid. If YOLO
    # already proposed a full-vial box, keep that vertical span with the fitted width.
    if is_full_vial_aspect((x, y, bw, bh)) and fh < bh * 0.85:
        fy = y
        fh = bh
        fitted = clip_box((fx, fy, fw, fh), width, height)
        fx, fy, fw, fh = fitted
        if fh > max_h:
            if above >= below:
                fy = fy + fh - max_h
            fh = max_h
            fitted = clip_box((fx, fy, fw, fh), width, height)
    if fitted[2] < 16 or fitted[3] < 20:
        return box
    return fitted


def tighten_yolo_box_to_lid(image, depth_mm, box):
    """Tighten only a loose YOLO box's width before clear-vial gating.

    YOLO commonly pads these transparent vials by 10-20 px on each side. That
    dilutes the otherwise-solid white lid below ``cap_run``'s threshold. Reuse
    the lid localization from ``fit_vial_shape``, but preserve the detector's
    vertical span so a weak clear body cannot be cropped away before gating.
    """
    height, width = image.shape[:2]
    x, y, box_w, box_h = clip_box(box, width, height)
    fitted = fit_vial_shape(image, depth_mm, (x, y, box_w, box_h))
    fit_x, _fit_y, fit_w, _fit_h = clip_box(fitted, width, height)
    if fit_w >= box_w or fit_w < 20:
        return (x, y, box_w, box_h)
    # Remove a little residual side padding while keeping upright aspect valid.
    inset = max(2, int(round(fit_w * 0.05)))
    tightened_w = fit_w - inset * 2
    if tightened_w < 18:
        return (x, y, box_w, box_h)
    return clip_box((fit_x + inset, y, tightened_w, box_h), width, height)


def _blob_depth_mm(depth_mm: np.ndarray, x: int, y: int, bw: int, bh: int) -> float | None:
    """Median near depth under a white-cap blob (center-weighted)."""
    height, width = depth_mm.shape[:2]
    cx = x + bw // 2
    cy = y + max(1, bh // 2)
    radius = max(3, min(bw, bh) // 3)
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(width, cx + radius + 1)
    y1 = min(height, cy + radius + 1)
    patch = depth_mm[y0:y1, x0:x1]
    return _nearest_surface_mm(patch)


def detect_vial_by_cap(
    image,
    depth_mm=None,
    intrinsics=None,
    *,
    lid_width_tol: float = DEFAULT_LID_WIDTH_TOL,
) -> list[tuple[int, int, int, int]]:
    """Find the white lid, then cover the clear cup on whichever side it sits.

    With depth + intrinsics, also accepts **near-round top-down lids** whose
    width at measured Z matches the fitted lid diameter (``vial_3d_dims.json``).
    Those proposals stay lid-face sized — they do **not** invent a tall upright
    lid→cup box from the 3D model.
    """
    height, width = image.shape[:2]
    if depth_mm is not None and depth_mm.shape[:2] != (height, width):
        depth_mm = align_depth(depth_mm, width, height)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    cap_mask = _cap_mask(value, saturation)
    cap_mask = cv2.morphologyEx(
        cap_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
    )
    # Round top-down lids need an isotropic close as well as the wide kernel.
    if depth_mm is not None and intrinsics is not None:
        cap_mask = cv2.morphologyEx(
            cap_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 100)
    _, gy, _ = _image_derivatives(gray)
    skin = skin_mask(image)
    # Also close vertically so end-cap (sideways) lids survive as compact blobs.
    if sideways_assist_enabled():
        cap_mask = cv2.bitwise_or(
            cap_mask,
            cv2.morphologyEx(
                cap_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)),
            ),
        )
    contours, _ = cv2.findContours(cap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Slightly smaller floor helps distant / weak-angle lids without swallowing walls.
    min_width = max(16, int(width * 0.035))
    min_end = max(10, int(min(width, height) * 0.025))
    found = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw > 0.72 * width or bh > 0.72 * height:
            continue
        if bw * bh > 0.4 * width * height:
            continue
        aspect = bw / max(bh, 1)
        # Side-on lids are wide strips; top-down / end-cap lids are near-round.
        side_on = bw >= min_width and 1.6 <= aspect <= 12.0 and bh >= 3
        top_downish = (
            bw >= min_end
            and bh >= min_end
            and 0.75 <= aspect < 1.6
            and bh >= max(8, int(bw * 0.55))
        )
        # Sideways end-cap: compact white blob — often taller than wide (Ø×lid depth).
        end_cap = (
            sideways_assist_enabled()
            and bw >= min_end
            and bh >= min_end
            and 0.22 <= aspect <= 1.55
            and max(bw, bh) <= min(width, height) * 0.40
            and min(bw, bh) >= min_end
        )
        if not (side_on or top_downish or end_cap):
            continue
        if _region_skin(skin, (x, y, bw, bh)) > 0.18:
            continue
        if _beside_white_background(value, saturation, x, y, bw, bh, width):
            continue
        x, y, bw, bh = _expand_lid(value, saturation, x, y, bw, bh, width, height)
        aspect = bw / max(bh, 1)
        pad = max(2, int(bw * 0.03))

        # Prefer classic upright cup span when the lid looks side-on / elongated.
        # Do NOT fall through to left/right growth — that corrupts upright boxes.
        cup_box = None
        if aspect >= 1.35:
            inset = max(1, bw // 5)
            core_x = x + inset
            core_w = max(1, bw - 2 * inset)
            max_cup = int(bw * FIT_MAX_HEIGHT_CAP_RATIO)
            above = _cup_rows(
                value, saturation, edges, core_x, core_w, y - 1, -1, max_cup, height, width, skin
            )
            below = _cup_rows(
                value, saturation, edges, core_x, core_w, y + bh, 1, max_cup, height, width, skin
            )
            cup = max(above, below)
            if cup >= max(8, int(bw * 0.28)):
                if above >= below:
                    cup_box = (x - pad, y - above, bw + pad * 2, bh + above + pad)
                else:
                    cup_box = (x - pad, y - pad, bw + pad * 2, bh + below + pad)
                found.append((bw * cup, cup_box, False))

        # Sideways / fallen: cup left/right of an end-cap lid only (never after wide upright lid).
        elif allow_sideways_enabled() and aspect < 1.35:
            grown = None
            if sideways_assist_enabled():
                grown = grow_sideways_from_lid(image, (x, y, bw, bh), depth_mm)
            if grown is None:
                inset_y = max(1, bh // 5)
                core_y = y + inset_y
                core_h = max(1, bh - 2 * inset_y)
                max_cup_x = int(max(bh, bw) * FIT_SIDEWAYS_LENGTH_RATIO)
                cap_z = _blob_depth_mm(depth_mm, x, y, bw, bh) if depth_mm is not None else None
                left = _cup_cols(
                    value,
                    saturation,
                    edges,
                    core_y,
                    core_h,
                    x - 1,
                    -1,
                    max_cup_x,
                    height,
                    width,
                    skin,
                    depth_mm=depth_mm,
                    cap_depth=cap_z,
                    gy=gy,
                )
                right = _cup_cols(
                    value,
                    saturation,
                    edges,
                    core_y,
                    core_h,
                    x + bw,
                    1,
                    max_cup_x,
                    height,
                    width,
                    skin,
                    depth_mm=depth_mm,
                    cap_depth=cap_z,
                    gy=gy,
                )
                cup_x = max(left, right)
                if cup_x >= max(8, int(max(bh, bw) * 0.22)):
                    if left >= right:
                        grown = (x - left, y - pad, bw + left + pad, bh + pad * 2)
                    else:
                        grown = (x - pad, y - pad, bw + right + pad, bh + pad * 2)
            if grown is not None:
                grown = clip_box(grown, width, height)
                if _is_sideways_box(grown) and not is_full_vial_aspect(grown):
                    cup_box = grown
                    found.append((grown[2] * grown[3], cup_box, False))

        # Lid-diameter@Z assist: keep round / cup-less lids that match fitted Ø.
        if cup_box is None and depth_mm is not None and intrinsics is not None:
            z = _blob_depth_mm(depth_mm, x, y, bw, bh)
            ok_lid, lid_score = _lid_match_at_z(
                bw, z, intrinsics, tol=lid_width_tol
            )
            if ok_lid and z is not None:
                # Expand slightly but do not invent full upright cup height.
                face = clip_box((x - pad, y - pad, bw + pad * 2, bh + pad * 2), width, height)
                # Prefer more circular, tighter Ø matches.
                roundness = 1.0 - min(1.0, abs(1.0 - (bh / max(bw, 1))))
                found.append((1000.0 * lid_score + 50.0 * roundness, face, True))
    if not found:
        return []
    found.sort(key=lambda item: item[0], reverse=True)
    return [clip_box(box, width, height) for _, box, _ in found[:8]]


def _rim_edge_score(edge_u8: np.ndarray, x: int, y: int, w: int, h: int, *, pad: int = 3) -> float:
    """Fraction of strong edge pixels in a thin ring around the box (0..1)."""
    height, width = edge_u8.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    outer = edge_u8[y0:y1, x0:x1]
    ring = np.ones(outer.shape, dtype=bool)
    ix0, iy0 = x - x0, y - y0
    ix1, iy1 = ix0 + w, iy0 + h
    shrink = 2
    if iy0 + shrink < iy1 - shrink and ix0 + shrink < ix1 - shrink:
        ring[iy0 + shrink : iy1 - shrink, ix0 + shrink : ix1 - shrink] = False
    samples = outer[ring]
    if samples.size == 0:
        return 0.0
    return float((samples > 0).mean())


def _image_derivatives(gray: np.ndarray):
    """Explicit horizontal (Gx/dx) and vertical (Gy/dy) Sobel responses + magnitude."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def _mean_abs(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def _vial_dxdy_structure_score(
    gx: np.ndarray,
    gy: np.ndarray,
    mag: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    top_downish: bool,
    sidewaysish: bool = False,
    gray: np.ndarray | None = None,
) -> float:
    """Score dx/dy energy vs expected vial structure (roughly 0..1+).

    Upright: strong opposing |Gx| on cup sides + closed-ish lid rim.
    Sideways / fallen: opposing |Gy| on top/bottom + lid on left/right end.
    Top-down: more isotropic rim (|Gx|≈|Gy|) around a brighter lid face.
    White-cap + darker body contrast down-weights flat bright slabs/walls.
    """
    height, width = mag.shape[:2]
    x, y, w, h = clip_box((x, y, w, h), width, height)
    if w < 6 or h < 6:
        return 0.0
    roi_gx = gx[y : y + h, x : x + w]
    roi_gy = gy[y : y + h, x : x + w]
    roi_mag = mag[y : y + h, x : x + w]
    if roi_mag.size == 0:
        return 0.0

    side = max(1, min(3, w // 5))
    left_gx = _mean_abs(roi_gx[:, :side])
    right_gx = _mean_abs(roi_gx[:, -side:])
    mid_gx = _mean_abs(roi_gx[:, side : max(side + 1, w - side)])
    side_pair = min(left_gx, right_gx)
    opposing = side_pair / max(mid_gx, 1.0)
    # Both sides present (not a single vertical scratch).
    balance = min(left_gx, right_gx) / max(max(left_gx, right_gx), 1.0)

    # Horizontal cup walls (sideways): opposing |Gy| on top/bottom edges.
    band_h = max(1, min(3, h // 5))
    top_gy = _mean_abs(roi_gy[:band_h, :])
    bot_gy = _mean_abs(roi_gy[-band_h:, :])
    mid_gy = _mean_abs(roi_gy[band_h : max(band_h + 1, h - band_h), :])
    opposing_h = min(top_gy, bot_gy) / max(mid_gy, 1.0)
    balance_h = min(top_gy, bot_gy) / max(max(top_gy, bot_gy), 1.0)

    # Thin outer ring vs soft interior — lid rim / cup silhouette.
    pad = max(1, min(3, min(w, h) // 6))
    ring = np.ones((h, w), dtype=bool)
    if h > 2 * pad and w > 2 * pad:
        ring[pad : h - pad, pad : w - pad] = False
    ring_mag = float(roi_mag[ring].mean()) if ring.any() else 0.0
    inner = ~ring
    inner_mag = float(roi_mag[inner].mean()) if inner.any() else 0.0
    rim_contrast = ring_mag / max(inner_mag, 1.0)

    gx_ring = _mean_abs(roi_gx[ring]) if ring.any() else 0.0
    gy_ring = _mean_abs(roi_gy[ring]) if ring.any() else 0.0
    iso = min(gx_ring, gy_ring) / max(max(gx_ring, gy_ring), 1.0)

    # Lid band (top/bottom or left/right): closed-ish rim often has both Gx and Gy.
    band = max(2, h // 4)
    top_iso = min(_mean_abs(roi_gx[:band]), _mean_abs(roi_gy[:band])) / max(
        max(_mean_abs(roi_gx[:band]), _mean_abs(roi_gy[:band])), 1.0
    )
    bot_iso = min(_mean_abs(roi_gx[-band:]), _mean_abs(roi_gy[-band:])) / max(
        max(_mean_abs(roi_gx[-band:]), _mean_abs(roi_gy[-band:])), 1.0
    )
    vband = max(2, w // 4)
    left_iso = min(_mean_abs(roi_gx[:, :vband]), _mean_abs(roi_gy[:, :vband])) / max(
        max(_mean_abs(roi_gx[:, :vband]), _mean_abs(roi_gy[:, :vband])), 1.0
    )
    right_iso = min(_mean_abs(roi_gx[:, -vband:]), _mean_abs(roi_gy[:, -vband:])) / max(
        max(_mean_abs(roi_gx[:, -vband:]), _mean_abs(roi_gy[:, -vband:])), 1.0
    )
    lid_closed = max(top_iso, bot_iso)
    if sidewaysish:
        lid_closed = max(lid_closed, left_iso, right_iso)

    # White lid + darker cup body (flat bright slabs score ~0 here).
    lid_body = 0.0
    if gray is not None and gray.shape[:2] == mag.shape[:2]:
        g = gray[y : y + h, x : x + w].astype(np.float32)
        end_bright = max(float(g[:band].mean()), float(g[-band:].mean()))
        mid_h0, mid_h1 = band, max(band + 1, h - band)
        mid_bright = float(g[mid_h0:mid_h1].mean())
        lid_body = max(0.0, (end_bright - mid_bright) / max(end_bright, 1.0))
        if sidewaysish:
            end_lr = max(float(g[:, :vband].mean()), float(g[:, -vband:].mean()))
            mid_w0, mid_w1 = vband, max(vband + 1, w - vband)
            mid_lr = float(g[:, mid_w0:mid_w1].mean())
            lid_body = max(
                lid_body, max(0.0, (end_lr - mid_lr) / max(end_lr, 1.0))
            )

    if top_downish and not sidewaysish:
        # Circular rim: isotropic ring, quieter center; prefer some lid brightness cue.
        score = (
            0.35 * iso
            + 0.25 * min(1.5, rim_contrast) / 1.5
            + 0.15 * lid_closed
            + 0.10 * min(1.0, ring_mag / 40.0)
            + 0.15 * lid_body
        )
    elif sidewaysish:
        # Horizontal cup walls (high |Gy| top/bottom) + end lid + body split.
        score = (
            0.30 * min(1.5, opposing_h) / 1.5
            + 0.15 * balance_h
            + 0.20 * lid_closed
            + 0.10 * min(1.5, rim_contrast) / 1.5
            + 0.25 * lid_body
        )
    else:
        # Vertical cup walls (high |Gx| left/right) + rim closure + lid/body split.
        score = (
            0.30 * min(1.5, opposing) / 1.5
            + 0.15 * balance
            + 0.20 * lid_closed
            + 0.10 * min(1.5, rim_contrast) / 1.5
            + 0.25 * lid_body
        )
    return float(score)


def vial_gradient_match_score(image, box, *, gx=None, gy=None, mag=None, gray=None) -> float:
    """Public dx/dy vial-structure score for ranking existing proposals (0..~1+)."""
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    if w < 6 or h < 6:
        return 0.0
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gx is None or gy is None or mag is None:
        gx, gy, mag = _image_derivatives(gray)
    aspect = h / max(w, 1)
    sidewaysish = _sideways_pose((x, y, w, h)) and aspect < 0.90
    top_downish = (not sidewaysish) and (
        _is_squat_topdown_box((x, y, w, h))
        or (0.75 <= (w / max(h, 1)) <= 1.35 and aspect < FULL_VIAL_ASPECT_MIN)
    )
    return _vial_dxdy_structure_score(
        gx,
        gy,
        mag,
        x,
        y,
        w,
        h,
        top_downish=top_downish,
        sidewaysish=sidewaysish,
        gray=gray,
    )


def detect_vial_by_edges(
    image,
    *,
    max_proposals: int = 6,
    gates: CameraGateProfile | None = None,
) -> list[tuple[int, int, int, int]]:
    """Propose lid+cup ROIs from bright low-sat blobs whose dx/dy match a vial.

    Supporting cue when YOLO / cap-first miss. Uses explicit Sobel Gx/Gy (plus
    Canny), scores rim circularity / opposing side edges, then ranks by that
    match. Not a standalone detector — locate still requires clear-vial /
    soft_clear / scale / within_reach. Local only (no peer invent).

    Weak gate profiles use slightly looser structure/rim floors for recall.
    """
    height, width = image.shape[:2]
    gates = gates or strong_gate_profile()
    weak = gates.name == "weak"
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx, gy, mag = _image_derivatives(gray)
    thr = max(22.0, float(np.percentile(mag, 68)))
    strong = (mag >= thr).astype(np.uint8) * 255
    canny = cv2.Canny(gray, 35, 100)
    edge = cv2.bitwise_or(strong, canny)

    # Slightly looser white than `_cap_mask`; structure score keeps blank walls out.
    white = ((value >= 155) & (saturation <= 70)).astype(np.uint8) * 255
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    skin = skin_mask(image)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Allow slightly smaller lids so far / tiny vials can still propose.
    min_width = max(10, int(width * 0.02))
    struct_floor = 0.14 if weak else 0.18
    rim_floor = 0.05 if weak else 0.07
    struct_soft = 0.09 if weak else 0.12
    rim_soft = 0.09 if weak else 0.12
    found: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < min_width or bh < 3 or bw > 0.72 * width:
            continue
        if bw * bh > 0.4 * width * height:
            continue
        aspect = bw / max(bh, 1)
        side_on = 1.5 <= aspect <= 12.0
        top_downish = 0.7 <= aspect < 1.6 and bh >= max(8, int(bw * 0.5))
        end_cap = (
            sideways_assist_enabled()
            and 0.22 <= aspect <= 1.55
            and bw >= max(8, int(min_width * 0.7))
            and bh >= max(8, int(min_width * 0.7))
        )
        if not (side_on or top_downish or end_cap):
            continue
        if _region_skin(skin, (x, y, bw, bh)) > 0.18:
            continue
        if _beside_white_background(value, saturation, x, y, bw, bh, width):
            continue
        rim = _rim_edge_score(edge, x, y, bw, bh)
        structure = _vial_dxdy_structure_score(
            gx, gy, mag, x, y, bw, bh, top_downish=top_downish, gray=gray
        )
        # Prefer structure match; keep a soft rim floor so blank slabs drop out.
        if structure < struct_floor and rim < rim_floor:
            continue
        if structure < struct_soft and rim < rim_soft:
            continue
        x, y, bw, bh = _expand_lid(value, saturation, x, y, bw, bh, width, height)
        aspect = bw / max(bh, 1)
        pad = max(2, int(bw * 0.03))
        cup_box = None
        if aspect >= 1.25:
            inset = max(1, bw // 5)
            core_x = x + inset
            core_w = max(1, bw - 2 * inset)
            max_cup = int(bw * FIT_MAX_HEIGHT_CAP_RATIO)
            above = _cup_rows(
                value, saturation, canny, core_x, core_w, y - 1, -1, max_cup, height, width, skin
            )
            below = _cup_rows(
                value, saturation, canny, core_x, core_w, y + bh, 1, max_cup, height, width, skin
            )
            cup = max(above, below)
            if cup >= max(8, int(bw * 0.25)):
                if above >= below:
                    cup_box = (x - pad, y - above, bw + pad * 2, bh + above + pad)
                else:
                    cup_box = (x - pad, y - pad, bw + pad * 2, bh + below + pad)
        elif allow_sideways_enabled() and aspect < 1.25:
            grown = None
            if sideways_assist_enabled():
                grown = grow_sideways_from_lid(image, (x, y, bw, bh), depth_mm=None)
            if grown is None:
                inset_y = max(1, bh // 5)
                core_y = y + inset_y
                core_h = max(1, bh - 2 * inset_y)
                max_cup_x = int(max(bh, bw) * FIT_SIDEWAYS_LENGTH_RATIO)
                left = _cup_cols(
                    value,
                    saturation,
                    canny,
                    core_y,
                    core_h,
                    x - 1,
                    -1,
                    max_cup_x,
                    height,
                    width,
                    skin,
                    gy=gy,
                )
                right = _cup_cols(
                    value,
                    saturation,
                    canny,
                    core_y,
                    core_h,
                    x + bw,
                    1,
                    max_cup_x,
                    height,
                    width,
                    skin,
                    gy=gy,
                )
                cup_x = max(left, right)
                if cup_x >= max(8, int(max(bh, bw) * 0.22)):
                    grown = (
                        (x - left, y - pad, bw + left + pad, bh + pad * 2)
                        if left >= right
                        else (x - pad, y - pad, bw + right + pad, bh + pad * 2)
                    )
            if grown is not None and _is_sideways_box(grown) and not is_full_vial_aspect(grown):
                cup_box = grown
        if cup_box is not None:
            full_structure = vial_gradient_match_score(
                image, cup_box, gx=gx, gy=gy, mag=mag, gray=gray
            )
            score = (
                1200.0 * full_structure
                + 400.0 * structure
                + 200.0 * rim
                + float(bw * cup_box[3]) * 0.02
            )
            found.append((score, cup_box))
        else:
            # Strong rim / structure alone: padded lid-face / short proposal.
            face = (x - pad, y - pad, bw + pad * 2, max(bh + pad * 2, int(bw * 1.05)))
            face_structure = vial_gradient_match_score(
                image, face, gx=gx, gy=gy, mag=mag, gray=gray
            )
            found.append((1000.0 * face_structure + 300.0 * structure + 150.0 * rim, face))
    if not found:
        return []
    found.sort(key=lambda item: item[0], reverse=True)
    return [clip_box(box, width, height) for _, box in found[:max_proposals]]


def accepts_topdown_lid_box(
    depth_mm,
    image,
    box,
    intrinsics,
    *,
    gates: CameraGateProfile | None = None,
    lid_width_tol: float = DEFAULT_LID_WIDTH_TOL,
) -> bool:
    """Accept a nearly round white lid face whose width@Z ≈ fitted lid Ø.

    Top-down recovery only — no tall cup invent. Walls/hands still fail.
    """
    if intrinsics is None or depth_mm is None:
        return False
    gates = gates or strong_gate_profile()
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    if w < 14 or h < 14:
        return False
    if w > 0.5 * width or h > 0.5 * height:
        return False
    if not _is_topdown_box((x, y, w, h)):
        return False
    if _region_skin(skin_mask(image), (x, y, w, h)) > 0.18:
        return False
    distance, _ = measure_cap_mm(depth_mm, image, (x, y, w, h))
    if distance is None:
        distance = _blob_depth_mm(depth_mm, x, y, w, h)
    ok_lid, _score = _lid_match_at_z(w, distance, intrinsics, tol=lid_width_tol)
    if not ok_lid:
        return False
    hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 165)
    # Lid face should be mostly white; blank walls often are, but Ø@Z rejects most.
    if float(white.mean()) < 0.35:
        return False
    # Mild edge presence (rim / threads); blank walls are near-zero.
    edge_floor = max(0.004, (gates.interior_edge_min or 0.012) * 0.45)
    return interior_edge_density(image, (x, y, w, h)) >= edge_floor



class VialDetector:
    def __init__(self, model_path: Path | None, confidence: float):
        self.confidence = confidence
        self.model = None
        self.name = "white cap"
        if model_path is None:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ultralytics is not installed. Tracking the white cap instead.")
            return
        self.model = YOLO(str(model_path))
        self.name = model_path.name
        print(f"Loaded YOLO weights: {model_path}")

    def detect(
        self,
        image,
        conf: float | None = None,
        *,
        imgsz: int | None = None,
    ) -> list[tuple[int, int, int, int]]:
        if self.model is None:
            return detect_vial_by_cap(image)
        height, width = image.shape[:2]
        threshold = self.confidence if conf is None else float(conf)
        # Leave room for gated candidates; the final per-camera set is NMS-capped below.
        predict_kwargs = {
            "verbose": False,
            "conf": threshold,
            "max_det": MAX_VIALS_PER_CAMERA * 4,
            # Ultralytics NMS is deliberately less aggressive than our final,
            # vial-aware NMS. Nearby real vials must reach the local gates.
            "iou": 0.80,
        }
        if imgsz is not None:
            predict_kwargs["imgsz"] = int(imgsz)
        predicted = self.model.predict(image, **predict_kwargs)
        boxes = []
        if predicted is None:
            return boxes
        for result in predicted:
            detected = getattr(result, "boxes", None)
            if detected is None:
                continue
            ranked = []
            for box in detected:
                xyxy = getattr(box, "xyxy", None)
                if xyxy is None:
                    continue
                x1, y1, x2, y2 = xyxy[0].tolist()
                confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
                ranked.append((confidence, clip_box((x1, y1, x2 - x1, y2 - y1), width, height)))
            ranked.sort(key=lambda item: item[0], reverse=True)
            boxes.extend(item[1] for item in ranked)
            break
        return boxes

    def detect_in_rois(
        self,
        image,
        tracked_boxes,
        conf: float | None = None,
    ) -> list[list[tuple[int, int, int, int]]]:
        """Run inexpensive crop inference for each prior vial and map boxes globally."""
        height, width = image.shape[:2]
        groups: list[list[tuple[int, int, int, int]]] = []
        for tracked in list(tracked_boxes)[:MAX_VIALS_PER_CAMERA]:
            tx, ty, tw, th = clip_box(tracked, width, height)
            pad_x = max(int(tw * ROI_PAD_FRACTION), (ROI_MIN_SIZE - tw) // 2)
            pad_y = max(int(th * ROI_PAD_FRACTION), (ROI_MIN_SIZE - th) // 2)
            x0 = max(0, tx - pad_x)
            y0 = max(0, ty - pad_y)
            x1 = min(width, tx + tw + pad_x)
            y1 = min(height, ty + th + pad_y)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                groups.append([])
                continue
            local = self.detect(
                crop,
                conf=conf,
                imgsz=ROI_PREDICT_IMGSZ,
            )
            groups.append(
                [
                    clip_box((x0 + x, y0 + y, box_w, box_h), width, height)
                    for x, y, box_w, box_h in local
                ]
            )
        return groups


def depth_to_mm(depth) -> np.ndarray:
    values = depth.astype(np.float32)
    if values.ndim == 3:
        values = values[:, :, 0]
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmax(finite)) <= 20:
        values = values * 1000.0
    return values


def align_depth(depth_mm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize depth to ``(height, width)``. This is NOT Orbbec D2C alignment.

    CAP_OBSENSOR returns the depth sensor's native frame. A nearest-neighbor
    resize only matches resolution; it does not remove the RGB–IR baseline.
    Prefer sampling depth at mapped color coordinates, or the SDK AlignFilter.
    """
    if depth_mm.shape[:2] == (height, width):
        return depth_mm
    return cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)


def focus_near(color, depth_mm, near_mm: float = 6000):
    """Blur and dim depth farther than 600 cm. Holes next to a closer object stay sharp."""
    valid = (depth_mm > 80) & np.isfinite(depth_mm)
    if float(valid.mean()) < 1:
        return color
    near = ((depth_mm > 80) & (depth_mm <= near_mm)).astype(np.uint8) * 255
    fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, fill)
    near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    blurred = cv2.GaussianBlur(color, (51, 51), 0)
    dimmed = (blurred.astype(np.float32) * 0.25).astype(np.uint8)
    focused = color.copy()
    focused[near == 0] = dimmed[near == 0]
    return focused


def within_reach(depth_mm, image, box) -> bool:
    """Keep a box when it has a valid surface depth (not absurdly near junk).

    Absolute far cutoffs are intentionally loose — a distant vial is allowed when
    ``vial_scale`` says the box is small enough in real mm at that Z (far + huge
    wall/window boxes still fail). Sensor invalid / beyond ``MAX_DEPTH_MM`` still
    reject. Legacy ``MAX_RANGE_MM`` is no longer a hard accept gate.
    """
    distance, _ = measure_cap_mm(depth_mm, image, box)
    if distance is None:
        height, width = depth_mm.shape[:2]
        x, y, w, h = clip_box(box, width, height)
        roi = depth_mm[y : y + h, x : x + w]
        valid = roi[(roi > MIN_SURFACE_MM) & (roi < MAX_DEPTH_MM)]
        if valid.size < 12:
            return False
        distance = float(np.median(valid))
    return MIN_SURFACE_MM < float(distance) < float(MAX_DEPTH_MM)


def _nearest_surface_mm(values) -> float | None:
    """Depth of the closest solid surface. The wall behind a clear cup is farther."""
    usable = values[(values > 80) & (values < MAX_DEPTH_MM) & np.isfinite(values)]
    if usable.size < 8:
        return None
    front = float(np.percentile(usable, 20))
    cluster = usable[(usable >= front - 15) & (usable <= front + 35)]
    if cluster.size < 8:
        order = np.sort(usable)
        cluster = order[: max(8, int(order.size * 0.35))]
    return float(np.median(cluster))


def measure_cap_mm(depth_mm: np.ndarray, image, box):
    """Depth of the white cap only. Background seen through the clear cup is ignored.

    Upright: solid white band near top/bottom. Sideways/fallen: white band on
    left or right end. Distances always come from the opaque lid, never the cup.
    """
    height, width = depth_mm.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    if w < 4 or h < 4:
        return None, None
    hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 170)
    best = None
    band_h = max(3, min(h // 4, max(4, w // 8)))
    for row in range(0, max(1, h - band_h + 1), max(1, band_h // 2)):
        columns = white[row : row + band_h].mean(axis=0) >= 0.55
        start, end = _longest_white_run(columns)
        run = end - start
        if run < max(6, int(w * 0.4)):
            continue
        inset = max(1, run // 8)
        left = x + start + inset
        top = y + row
        cap_w = max(2, run - 2 * inset)
        cap = clip_box((left, top, cap_w, band_h), width, height)
        reading = _nearest_surface_mm(depth_mm[cap[1] : cap[1] + cap[3], cap[0] : cap[0] + cap[2]])
        if reading is None:
            continue
        if best is None or reading < best[0]:
            best = (reading, cap)
    # Sideways: scan vertical bands on left/right ends (cap is a short-axis strip).
    # Never replace a good top/bottom upright cap with a sideways band.
    if best is None and _sideways_pose((x, y, w, h), image=image):
        band_w = max(3, min(w // 4, max(4, h // 8)))
        for col in range(0, max(1, w - band_w + 1), max(1, band_w // 2)):
            rows = white[:, col : col + band_w].mean(axis=1) >= 0.55
            start, end = _longest_white_run(rows)
            run = end - start
            if run < max(6, int(h * 0.4)):
                continue
            inset = max(1, run // 8)
            left = x + col
            top = y + start + inset
            cap_h = max(2, run - 2 * inset)
            cap = clip_box((left, top, band_w, cap_h), width, height)
            reading = _nearest_surface_mm(
                depth_mm[cap[1] : cap[1] + cap[3], cap[0] : cap[0] + cap[2]]
            )
            if reading is None:
                continue
            if best is None or reading < best[0]:
                best = (reading, cap)
    if best is None:
        return None, None
    return best


def intrinsics_from_capture(cap, width: int, height: int):
    fx = float(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FX))
    fy = float(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FY))
    cx = float(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_CX))
    cy = float(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_CY))
    if fx <= 1 or fy <= 1 or cx <= 1 or cy <= 1:
        return None
    if cx > width or cy > height:
        scale_x = width / (cx * 2)
        scale_y = height / (cy * 2)
        fx *= scale_x
        cx *= scale_x
        fy *= scale_y
        cy *= scale_y
    return fx, fy, cx, cy


def position_mm(pixel, distance_mm: float | None, intrinsics):
    if intrinsics is None or distance_mm is None:
        return None, None, distance_mm
    fx, fy, cx, cy = intrinsics
    u, v = pixel
    z_mm = distance_mm
    x_mm = (u - cx) * z_mm / fx
    y_mm = (v - cy) * z_mm / fy
    slant = math.hypot(x_mm, y_mm, z_mm)
    return x_mm, y_mm, slant


def cap_run(image, box) -> float:
    """How much of one end is a solid white lid (top/bottom or left/right)."""
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    if w < 8 or h < 8:
        return 0.0
    hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    band = max(3, int(h * 0.34))

    def run_fraction_h(region) -> float:
        columns = region.mean(axis=0) >= 0.45
        best = run = 0
        for bright in columns:
            run = run + 1 if bright else 0
            best = max(best, run)
        return best / max(len(columns), 1)

    def run_fraction_v(region) -> float:
        rows = region.mean(axis=1) >= 0.45
        best = run = 0
        for bright in rows:
            run = run + 1 if bright else 0
            best = max(best, run)
        return best / max(len(rows), 1)

    top_bot = max(run_fraction_h(white[:band]), run_fraction_h(white[-band:]))
    if _sideways_pose((x, y, w, h)):
        vband = max(3, int(w * 0.34))
        left_right = max(
            run_fraction_v(white[:, :vband]), run_fraction_v(white[:, -vband:])
        )
        return max(top_bot, left_right)
    return top_bot


def has_white_cap(image, box, gates: CameraGateProfile | None = None) -> bool:
    """The lid is a solid white band. A logo or a wall does not span the box that way.

    Pass the camera's ``gates`` so weak-cam aspect floors are not silently
    overridden by the strong ``usable_detection`` defaults (false not-in-view).
    Sideways boxes look for the white band on the left or right end.
    """
    if not usable_detection(image, box, gates=gates):
        return False
    if cap_run(image, box) < 0.75:
        return False
    # Reject uniform bright slabs (walls/windows): a real vial has a white lid
    # at one end and a much less-white clear/plastic body in the middle.
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    sideways = _sideways_pose((x, y, w, h), gates=gates, image=image)
    if sideways:
        band = max(3, int(w * 0.34))
        if w <= band * 2:
            return True
        left = float(white[:, :band].mean())
        right = float(white[:, -band:].mean())
        mid = float(white[:, band : w - band].mean())
        lid = max(left, right)
        if mid > 0.5 and min(left, right) > 0.4:
            return False
        return mid <= lid - 0.12
    band = max(3, int(h * 0.34))
    if h <= band * 2:
        return True
    top = float(white[:band].mean())
    bottom = float(white[-band:].mean())
    mid = float(white[band : h - band].mean())
    lid = max(top, bottom)
    # Whole box white-ish → background, not lid + cup.
    if mid > 0.5 and min(top, bottom) > 0.4:
        return False
    return mid <= lid - 0.12


def _lid_cup_splits(image, box):
    """Split a box into lid/cup ranges.

    Returns ``("rows", x, w, lid_y, cup_y)`` for upright or
    ``("cols", y, h, lid_x, cup_x)`` for sideways.
    """
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    sideways = _sideways_pose((x, y, w, h), image=image)
    if sideways:
        if w < 12 or h < 8:
            return None
        band = max(3, int(w * 0.28))
        left = float(white[:, :band].mean())
        right = float(white[:, -band:].mean())
        if left >= right:
            lid = (x, x + band)
            cup = (x + band, x + w)
        else:
            lid = (x + w - band, x + w)
            cup = (x, x + w - band)
        if cup[1] - cup[0] < 6:
            return None
        return "cols", y, h, lid, cup
    if w < 8 or h < 12:
        return None
    band = max(3, int(h * 0.28))
    top = float(white[:band].mean())
    bottom = float(white[-band:].mean())
    if top >= bottom:
        lid = (y, y + band)
        cup = (y + band, y + h)
    else:
        lid = (y + h - band, y + h)
        cup = (y, y + h - band)
    if cup[1] - cup[0] < 6:
        return None
    return "rows", x, w, lid, cup


def _depth_hole_or_far_frac(
    roi, cap_depth: float | None, *, far_delta_mm: float = CLEAR_FAR_DELTA_MM
) -> float:
    """Fraction of pixels that are invalid or farther than the opaque lid."""
    total = int(roi.size)
    if total <= 0:
        return 1.0
    finite = np.isfinite(roi)
    valid = finite & (roi > 80) & (roi < MAX_DEPTH_MM)
    invalid = ~valid
    if cap_depth is None:
        far = np.zeros(roi.shape, dtype=bool)
    else:
        far = valid & (roi > float(cap_depth) + float(far_delta_mm))
    return float(np.count_nonzero(invalid | far)) / float(total)


def looks_like_clear_cup(depth_mm, image, box, gates: CameraGateProfile | None = None) -> bool:
    """True when the body under/above a white lid is clear (depth holes / far wall).

    Opaque jars and walls keep dense near-depth in the cup region. Clear plastic
    lets background through, so invalid or far depth is much higher than on the lid.
    """
    gates = gates or strong_gate_profile()
    if not has_white_cap(image, box, gates=gates):
        return False
    if depth_mm is None:
        # Color-only fallback: mid-band contrast already required by has_white_cap.
        return looks_like_capped_vial(image, box)

    height, width = image.shape[:2]
    depth_mm = align_depth(depth_mm, width, height)
    split = _lid_cup_splits(image, box)
    if split is None:
        return False
    mode, a0, a1, (lid0, lid1), (cup0, cup1) = split
    if mode == "cols":
        # Sideways: a0/a1 are y/h; lid/cup are absolute x ranges.
        lid_roi = depth_mm[a0 : a0 + a1, lid0:lid1]
        cup_roi = depth_mm[a0 : a0 + a1, cup0:cup1]
    else:
        # Upright: a0/a1 are x/w; lid/cup are absolute y ranges.
        lid_roi = depth_mm[lid0:lid1, a0 : a0 + a1]
        cup_roi = depth_mm[cup0:cup1, a0 : a0 + a1]
    lid_valid = lid_roi[np.isfinite(lid_roi) & (lid_roi > 80) & (lid_roi < MAX_DEPTH_MM)]
    if lid_valid.size < 6:
        return False
    cap_depth = float(np.median(lid_valid))
    lid_hole = _depth_hole_or_far_frac(lid_roi, cap_depth, far_delta_mm=gates.clear_far_delta_mm)
    cup_hole = _depth_hole_or_far_frac(cup_roi, cap_depth, far_delta_mm=gates.clear_far_delta_mm)
    # Screw cap is opaque: mostly solid near-depth.
    if lid_hole > gates.lid_hole_max:
        return False
    # Clear cup: more holes / farther readings than the lid.
    if cup_hole < lid_hole + gates.clear_hole_margin:
        return False
    if cup_hole < gates.cup_hole_min:
        return False
    # Mason-jar / opaque blob: dense surface near the lid distance fills the cup.
    cup_near = cup_roi[
        np.isfinite(cup_roi)
        & (cup_roi > 80)
        & (cup_roi < MAX_DEPTH_MM)
        & (cup_roi <= cap_depth + 120)
    ]
    if cup_roi.size > 0 and (cup_near.size / float(cup_roi.size)) >= gates.opaque_cup_near_frac:
        return False
    return True


def looks_like_capped_clear_vial(
    depth_mm, image, box, gates: CameraGateProfile | None = None
) -> bool:
    """Primary accept gate: white screw cap + clear cup depth holes, not a jar/wall."""
    gates = gates or strong_gate_profile()
    return (
        has_white_cap(image, box, gates=gates)
        and looks_like_capped_vial(image, box)
        and looks_like_clear_cup(depth_mm, image, box, gates=gates)
    )


def accepts_cap_fallback_box(
    image, box, *, gates: CameraGateProfile | None = None,
    depth_mm=None, intrinsics=None,
) -> bool:
    """Cap-first FN recovery: white lid + plausible body (no clear-depth required).

    Only for boxes from ``detect_vial_by_cap`` (lid blob + cup span, or top-down
    lid@Z match). Does not invent a box from the peer camera. Blank walls /
    empty benches still fail.
    """
    gates = gates or strong_gate_profile()
    if (
        depth_mm is not None
        and intrinsics is not None
        and accepts_topdown_lid_box(
            depth_mm, image, box, intrinsics, gates=gates
        )
    ):
        return True
    if not usable_detection(image, box, gates=gates):
        return False
    return (
        has_white_cap(image, box, gates=gates)
        and looks_like_capped_vial(image, box)
        and has_interior_structure(image, box, gates=gates)
    )


def accepts_vial_box(
    depth_mm,
    image,
    box,
    *,
    peer_confirmed: bool = False,
    soft_clear: bool = False,
    cap_fallback: bool = False,
    gates: CameraGateProfile | None = None,
    intrinsics=None,
) -> bool:
    """Strict clear-vial gate, or soft white-cap+structure when allowed.

    Soft path unlocks when sticky peer memory is warm (``peer_confirmed``), when
    local soft-clear is on (standalone / independent dual), or when cap-first
    fallback found a lid+body (``cap_fallback``). Soft path still requires a solid
    white lid, capped look, and interior edges — blank walls and empty benches fail.
    Top-down lid-face boxes that match fitted Ø@Z are accepted on soft/cap paths.
    """
    gates = gates or strong_gate_profile()
    if looks_like_capped_clear_vial(depth_mm, image, box, gates=gates):
        return True
    if cap_fallback or soft_clear or peer_confirmed:
        if accepts_topdown_lid_box(
            depth_mm, image, box, intrinsics, gates=gates
        ):
            return True
    if cap_fallback:
        return accepts_cap_fallback_box(
            image, box, gates=gates, depth_mm=depth_mm, intrinsics=intrinsics
        )
    if not (peer_confirmed or soft_clear):
        return False
    # Worse angle / filled-in clear-cup depth: accept white lid + structure.
    return (
        has_white_cap(image, box, gates=gates)
        and looks_like_capped_vial(image, box)
        and has_interior_structure(image, box, gates=gates)
    )


def interior_edge_density(image, box) -> float:
    """Fraction of Canny edge pixels inside the box (0–1)."""
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    if w < 8 or h < 8:
        return 0.0
    gray = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 100)
    return float(edges.mean()) / 255.0


def has_interior_structure(image, box, gates: CameraGateProfile | None = None) -> bool:
    """Walls/windows are near-uniform; a capped vial has lid and rim edges."""
    floor = (gates or strong_gate_profile()).interior_edge_min
    # Mild floor: clear cups are edge-light, but blank walls sit near zero.
    return interior_edge_density(image, box) >= floor


def looks_like_capped_vial(image, box) -> bool:
    """A white lid at either end of the box (top/bottom or left/right), not a hand."""
    height, width = image.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    sideways = _sideways_pose((x, y, w, h), image=image)
    if sideways:
        if w < 22 or h < 14:
            return False
        inset = max(1, int(h * 0.16))
        center = image[y + inset : y + h - inset, x : x + w]
        if center.size == 0 or float(skin_mask(center).mean()) > 0.22:
            return False
        hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
        white = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 165)
        band = max(3, int(w * 0.4))
        return max(float(white[:, :band].mean()), float(white[:, -band:].mean())) >= 0.3
    if w < 16 or h < 20:
        return False
    inset = max(1, int(w * 0.16))
    center = image[y : y + h, x + inset : x + w - inset]
    if center.size == 0 or float(skin_mask(center).mean()) > 0.22:
        return False
    hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 165)
    band = max(3, int(h * 0.4))
    return max(float(white[:band].mean()), float(white[-band:].mean())) >= 0.3


def usable_detection(image, box, gates: CameraGateProfile | None = None) -> bool:
    """A vial is a small cup, squat top-down lid face, or sideways lid+cup.

    Upright full-vial aspect (~1.0–1.65) is checked first. Squat top-down and
    clearly horizontal sideways are exceptions for non-upright poses only.
    """
    gates = gates or strong_gate_profile()
    height, width = image.shape[:2]
    x, y, box_w, box_h = clip_box(box, width, height)
    aspect = box_h / max(box_w, 1)
    upright = gates.aspect_min <= aspect <= max(gates.aspect_max, gates.usable_aspect_max)
    squat_td = (not upright) and _is_squat_topdown_box((x, y, box_w, box_h))
    sideways = (not upright) and _sideways_pose((x, y, box_w, box_h), gates=gates)
    if upright:
        if box_w < 18 or box_h < 22:
            return False
        if box_w > 0.42 * width or box_h > 0.55 * height:
            return False
    elif squat_td:
        if box_w < 14 or box_h < 14:
            return False
        if box_w > 0.5 * width or box_h > 0.5 * height:
            return False
    elif sideways:
        if box_w < 22 or box_h < 12:
            return False
        if box_w > 0.55 * width or box_h > 0.42 * height:
            return False
    else:
        return False
    touches = sum(
        (
            x <= 1,
            y <= 1,
            x + box_w >= width - 1,
            y + box_h >= height - 1,
        )
    )
    if touches >= 2:
        return False
    return True


def vial_scale(depth_mm, image, box, intrinsics) -> bool:
    """Geometric scale gate: box width/height in mm must match a small vial at Z.

    Far vials are allowed when the pixel box is correspondingly tiny. Far + huge
    (wall/window) fails. Upright uses width/height bands; squat top-down uses
    diameter-like bounds; sideways (long≈H, short≈Ø) only for non-upright boxes.
    """
    if intrinsics is None:
        return True
    distance, _ = measure_cap_mm(depth_mm, image, box)
    if distance is None or distance <= 0:
        return False
    if float(distance) <= MIN_SURFACE_MM:
        return False
    fx, fy, _, _ = intrinsics
    if fx <= 1 or fy <= 1:
        return False
    _, _, box_w, box_h = box
    width_mm = box_w * distance / fx
    height_mm = box_h * distance / fy
    # Upright first — never apply sideways long/short bands to full-vial boxes.
    if is_full_vial_aspect(box):
        return 14 <= width_mm <= 110 and 22 <= height_mm <= 170
    if _is_squat_topdown_box(box):
        # Lid face ~Ø54 mm; allow generous tol for foreshortening / pad.
        return 18 <= width_mm <= 110 and 18 <= height_mm <= 110
    if _sideways_pose(box, image=image):
        long_mm = max(width_mm, height_mm)
        short_mm = min(width_mm, height_mm)
        # long ≈ total H (~73 mm), short ≈ Ø (~54 mm); keep walls out.
        return 35 <= long_mm <= 155 and 14 <= short_mm <= 110
    # Non-pose leftover: upright-ish geometric band.
    return 14 <= width_mm <= 110 and 22 <= height_mm <= 170


def drop_huge(boxes, depth_mm, image, intrinsics):
    """Throw out a box when the depth reading says it is much bigger than a vial."""
    return [box for box in boxes if vial_scale(depth_mm, image, box, intrinsics)]



def box_iou(first, second) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def nms_vial_boxes(
    boxes, *, iou_threshold: float = VIAL_NMS_IOU, max_vials: int = MAX_VIALS_PER_CAMERA
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping proposals while retaining distinct vials in input rank order."""
    kept: list[tuple[int, int, int, int]] = []
    for raw in boxes:
        box = tuple(int(round(v)) for v in raw)
        if box[2] <= 0 or box[3] <= 0:
            continue
        if any(box_iou(box, prior) >= iou_threshold for prior in kept):
            continue
        kept.append(box)
        if len(kept) >= max_vials:
            break
    return kept


def associate_vial_tracks(
    tracks: list[VialTrack],
    detections: list[tuple[int, int, int, int]],
    *,
    iou_threshold: float = TRACK_ASSOC_IOU,
) -> tuple[dict[int, int], list[int], list[int]]:
    """Greedy one-to-one IoU association without collapsing distinct detections."""
    ranked = sorted(
        (
            (box_iou(track.box, detection), track_index, detection_index)
            for track_index, track in enumerate(tracks)
            for detection_index, detection in enumerate(detections)
        ),
        reverse=True,
    )
    matches: dict[int, int] = {}
    used_detections: set[int] = set()
    for overlap, track_index, detection_index in ranked:
        if overlap < iou_threshold:
            break
        if track_index in matches or detection_index in used_detections:
            continue
        matches[track_index] = detection_index
        used_detections.add(detection_index)
    unmatched_tracks = [index for index in range(len(tracks)) if index not in matches]
    unmatched_detections = [
        index for index in range(len(detections)) if index not in used_detections
    ]
    return matches, unmatched_tracks, unmatched_detections


def choose_box(
    boxes,
    previous,
    image,
    depth_mm=None,
    *,
    peer_confirmed: bool = False,
    soft_clear: bool = False,
    cap_fallback: bool = False,
    gates: CameraGateProfile | None = None,
    intrinsics=None,
    use_3d_model_score: bool = False,
    use_edge_rank: bool = False,
    peer_orb_desc: np.ndarray | None = None,
    orb_assist_weight: float = DEFAULT_ORB_ASSIST_WEIGHT,
):
    """Prefer a white-capped clear vial. Do not lock onto background once the hold is bad.

    **Upright-first:** if any pool candidate is full-vial aspect (~1.0–1.65), only
    those compete — sideways/top-down cannot steal. ORB never participates when an
    upright candidate exists (dual overlay / soft ties among non-upright only).
    **Nearest-first:** among remaining candidates, prefer the nearer white-cap depth
    so a gripped/foreground vial beats a rack of farther lids.
    When ``use_3d_model_score`` is on and depth+intrinsics exist, prefer boxes whose
    projected size at white-cap Z matches ``vial_3d_dims.json``.
    When ``use_edge_rank`` is on, dx/dy vial-structure score breaks ties among
    close YOLO/cap candidates (does not invent new boxes).
    """
    if not boxes:
        return None

    def is_ok(box) -> bool:
        if depth_mm is None:
            return has_white_cap(image, box, gates=gates)
        return accepts_vial_box(
            depth_mm,
            image,
            box,
            peer_confirmed=peer_confirmed,
            soft_clear=soft_clear,
            cap_fallback=cap_fallback,
            gates=gates,
            intrinsics=intrinsics,
        )

    capped = [box for box in boxes if is_ok(box)]
    pool = capped if capped else [box for box in boxes if has_white_cap(image, box, gates=gates)]
    if not pool:
        return None

    # Upright-first: a good lid→cup box always beats sideways/top-down alternatives.
    upright_pool = [box for box in pool if is_full_vial_aspect(box, gates=gates)]
    if upright_pool:
        pool = upright_pool

    stack = _load_3d_identify_stack() if use_3d_model_score else None
    rank_fn = stack[1] if stack else None
    dims = _dims_for_3d_score() if stack else None
    height, width = image.shape[:2]
    edge_gx = edge_gy = edge_mag = None
    edge_gray = None
    if use_edge_rank and len(pool) > 1:
        edge_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_gx, edge_gy, edge_mag = _image_derivatives(edge_gray)
    # ORB is dual-overlay only among non-upright pools — never overrides upright YOLO.
    use_orb = (
        peer_orb_desc is not None
        and float(orb_assist_weight) > 0.0
        and not upright_pool
    )
    orb_cache: dict[tuple[int, int, int, int], float] = {}
    near_cache: dict[tuple[int, int, int, int], float] = {}

    def model_rank(box) -> float:
        if rank_fn is None or depth_mm is None or intrinsics is None:
            return 0.0
        distance, _ = measure_cap_mm(depth_mm, image, box)
        return float(
            rank_fn(
                box,
                distance_mm=distance,
                K=intrinsics,
                dims=dims,
                image_size=(width, height),
            )
        )

    def nearness_rank(box) -> float:
        """Higher is nearer. Unknown depth sorts below any measured surface."""
        if depth_mm is None:
            return 0.0
        key = tuple(int(v) for v in box)
        if key not in near_cache:
            distance, _ = measure_cap_mm(depth_mm, image, box)
            near_cache[key] = (
                -1.0e9 if distance is None else -float(distance)
            )
        return near_cache[key]

    def edge_rank(box) -> float:
        if not use_edge_rank:
            return 0.0
        return vial_gradient_match_score(
            image, box, gx=edge_gx, gy=edge_gy, mag=edge_mag, gray=edge_gray
        )

    def orb_rank(box) -> float:
        if not use_orb:
            return 0.0
        key = tuple(int(v) for v in box)
        if key not in orb_cache:
            _, local = extract_orb_descriptors(image, box)
            _n, score = match_orb_descriptors(local, peer_orb_desc)
            orb_cache[key] = float(score)
        return orb_cache[key]

    def sideways_rank(box) -> float:
        # Soft lid-end + body length@Z among non-upright only (no live 3D gate).
        if upright_pool or not sideways_assist_enabled():
            return 0.0
        return _sideways_soft_rank(box, image, depth_mm, intrinsics)

    def best_of(candidates):
        return max(
            candidates,
            key=lambda box: (
                _pose_rank(box, gates=gates),
                nearness_rank(box),
                model_rank(box),
                sideways_rank(box),
                cap_run(image, box),
                edge_rank(box) + float(orb_assist_weight) * orb_rank(box),
                # Larger pixel area ≈ nearer when depth is missing.
                box[2] * box[3],
            ),
        )

    # Previous was not a real clear vial — jump to the best capped candidate immediately.
    if previous is None or not is_ok(previous):
        return best_of(pool)

    px, py, pw, ph = previous
    pcx, pcy = px + pw / 2, py + ph / 2
    reach = max(pw, ph, 1) * 1.4
    nearby = []
    for box in pool:
        x, y, w, h = box
        distance = math.hypot((x + w / 2) - pcx, (y + h / 2) - pcy)
        if box_iou(previous, box) >= 0.08 or distance <= reach:
            nearby.append(box)
    if not nearby:
        # Better a real vial elsewhere than sticking to empty space.
        return best_of(pool)
    return max(
        nearby,
        key=lambda box: (
            _pose_rank(box, gates=gates),
            nearness_rank(box),
            model_rank(box),
            sideways_rank(box),
            box_iou(previous, box),
            cap_run(image, box),
            edge_rank(box) + float(orb_assist_weight) * orb_rank(box),
            box[2] * box[3],
        ),
    )


def smooth_box(previous, current, keep: float = 0.78):
    if previous is None:
        return tuple(float(v) for v in current)
    return tuple(keep * old + (1 - keep) * new for old, new in zip(previous, current))


def draw_label(image, origin, lines, color) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(line, font, 0.55, 1)[0] for line in lines]
    width = max(size[0] for size in sizes) + 12
    height = sum(size[1] + 8 for size in sizes) + 8
    x = max(0, min(x, image.shape[1] - width))
    top = max(0, y - height)
    cv2.rectangle(image, (x, top), (x + width, top + height), (0, 0, 0), -1)
    cursor = top + 6
    for line, (_, text_h) in zip(lines, sizes):
        cursor += text_h
        cv2.putText(image, line, (x + 6, cursor), font, 0.55, color, 1, cv2.LINE_AA)
        cursor += 8



def range_band_label(distance_mm: float | None) -> str:
    """Coarse proximity tag for overlay: close / mid / far."""
    if distance_mm is None:
        return "unknown"
    if float(distance_mm) <= CLOSE_RANGE_MM:
        return "close"
    if float(distance_mm) <= MID_RANGE_MM:
        return "mid"
    return "far"


def empty_view_message(miss_reason: str | None = None) -> str:
    """Honest empty-scene banner: only claim not-in-view when no candidates exist."""
    if miss_reason == "low_confidence":
        return "Low confidence"
    return "No vial in view"


def log_found_vials(camera_index: int, fixes: list[VialFix]) -> bool:
    """Emit an unmistakable live-worker event whenever multiple vials are present."""
    if len(fixes) < 2:
        return False
    ids = ",".join(
        "?" if fix.track_id is None else str(fix.track_id)
        for fix in fixes
    )
    print(
        f"[cam{camera_index}] FOUND {len(fixes)} vials ids={ids}",
        flush=True,
    )
    return True


def annotate(
    image,
    fix: VialFix | None,
    *,
    fixes: list[VialFix] | None = None,
    camera_label: str | None = None,
    help_lines: list[str] | None = None,
    miss_reason: str | None = None,
    show_vial_labels: bool = False,
):
    """Draw detection boxes. Vial/close/mid text is off unless show_vial_labels=True."""
    view = image.copy()
    tip = help_lines or ["Q quits"]
    visible = fixes if fixes else ([] if fix is None else [fix])
    if camera_label:
        draw_label(view, (12, view.shape[0] - 8), [camera_label, *tip], (220, 220, 220))
    if not visible:
        if show_vial_labels:
            lines = [empty_view_message(miss_reason), *tip]
            draw_label(view, (12, 78), lines, (0, 180, 255))
        return view

    if not show_vial_labels:
        for item in visible:
            x, y, w, h = item.box
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 160, 0), 1)
        return view

    height, width = view.shape[:2]
    cv2.drawMarker(view, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 1)
    for item in visible:
        x, y, w, h = item.box
        color = (0, 220, 0) if item.distance_mm is not None else (0, 165, 255)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 3)
        cx, cy = item.center
        cv2.drawMarker(view, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
        cv2.line(view, (width // 2, height // 2), (cx, cy), (255, 255, 0), 1)

        if item.sample_box is not None:
            sx, sy, sw, sh = item.sample_box
            cv2.rectangle(view, (sx, sy), (sx + sw, sy + sh), (255, 220, 0), 1)

        # Opt-in overlay only: class, range band (close/mid/far), distance.
        identity = [] if item.track_id is None else [f"track {item.track_id}"]
        if item.distance_mm is None:
            lines = ["vial", *identity, "No depth on the cap"]
        else:
            shown = item.z_mm if item.z_mm is not None else item.distance_mm
            band = range_band_label(shown)
            lines = ["vial", band, *identity, f"{shown / 10:.1f} cm away"]
        if item.held:
            lines.append("holding last view")
        draw_label(view, (x, max(8, y - 4)), lines, color)
    return view


def window_title(camera_index: int, dual: bool) -> str:
    if dual:
        return f"Vial distance — camera {camera_index}"
    return WINDOW


def next_raw_index() -> int:
    RAW_DATASET.mkdir(parents=True, exist_ok=True)
    nums = []
    for path in RAW_DATASET.glob("vial_*.jpg"):
        parts = path.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return max(nums, default=0) + 1


def box_to_yolo(box, width: int, height: int) -> str:
    x, y, w, h = box
    cx = min(1.0, max(0.0, (x + w / 2) / width))
    cy = min(1.0, max(0.0, (y + h / 2) / height))
    bw = min(1.0, max(0.0, w / width))
    bh = min(1.0, max(0.0, h / height))
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def load_sidecar(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


class SnapshotRecorder:
    """Save clean stills (no overlay) with box/distance sidecars. Cap one run at 50."""

    def __init__(self, pending_dir: Path | None = None, max_saves: int = SNAPSHOT_MAX_PER_RUN):
        self.pending_dir = pending_dir or REVIEW_PENDING
        self.max_saves = max_saves
        self.saved = 0
        self._last_box: dict[int, tuple[int, int, int, int]] = {}
        self._last_time: dict[int, float] = {}
        self.session_files: list[Path] = []

    @property
    def full(self) -> bool:
        return self.saved >= self.max_saves

    def maybe_save(self, camera_index: int, color, fix: VialFix | None, now: float | None = None) -> Path | None:
        if fix is None or fix.held:
            return None
        if self.full:
            return None
        now = time.time() if now is None else now
        last_t = self._last_time.get(camera_index, 0.0)
        if now - last_t < SNAPSHOT_INTERVAL_S:
            return None
        previous = self._last_box.get(camera_index)
        if previous is not None and box_iou(previous, fix.box) >= SNAPSHOT_MIN_MOVE_IOU:
            return None
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"cam{camera_index}_{stamp}_{self.saved + 1:04d}"
        image_path = self.pending_dir / f"{stem}.jpg"
        meta_path = self.pending_dir / f"{stem}.json"
        height, width = color.shape[:2]
        ok = cv2.imwrite(str(image_path), color, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            print(f"Could not save still {image_path}")
            return None
        payload = {
            "camera": camera_index,
            "box": [int(v) for v in fix.box],
            "distance_mm": None if fix.distance_mm is None else float(fix.distance_mm),
            "x_mm": None if fix.x_mm is None else float(fix.x_mm),
            "y_mm": None if fix.y_mm is None else float(fix.y_mm),
            "image_width": width,
            "image_height": height,
            "timestamp": stamp,
        }
        meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.saved += 1
        self._last_box[camera_index] = tuple(int(v) for v in fix.box)
        self._last_time[camera_index] = now
        self.session_files.append(image_path)
        print(f"Saved review still {self.saved}/{self.max_saves}: {image_path.name}")
        return image_path


def draw_review_help(view, index: int, total: int, meta: dict) -> None:
    cam = meta.get("camera", "?")
    dist = meta.get("distance_mm")
    dist_line = "no distance" if dist is None else f"{float(dist) / 10:.1f} cm on cam {cam}"
    box = meta.get("box")
    short_note = ""
    if box is not None and len(box) == 4:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 0), 2)
        box_t = (x, y, w, h)
        if not is_review_acceptable_box(box_t):
            short_note = (
                f"BOX TOO SHORT (h/w={box_aspect_hw(box_t):.2f}) — "
                "use I, drag lid through cup (side view)"
            )
        elif _is_sideways_box(box_t) and not is_full_vial_aspect(box_t) and not _is_topdown_box(box_t):
            short_note = (
                f"Sideways lid+cup (h/w={box_aspect_hw(box_t):.2f}) — R ok"
            )
        elif _is_topdown_box(box_t) and not is_full_vial_aspect(box_t):
            short_note = (
                f"Near-square top-down lid (h/w={box_aspect_hw(box_t):.2f}) — R ok"
            )
    lines = [
        f"Review {index + 1}/{total}   camera {cam}   {dist_line}",
        "R correct (upright lid→cup, top-down lid, or sideways lid+cup)",
        "W not a vial (empty-label negative)",
        "I incorrect box -> then drag the right box",
        "N skip (leave pending)   Q end review",
    ]
    if short_note:
        lines.insert(1, short_note)
    draw_label(
        view,
        (12, 24),
        lines,
        (255, 255, 255),
    )


def _move_pair(image_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / image_path.name
    if dest.exists():
        dest = dest_dir / f"{image_path.stem}_moved{image_path.suffix}"
    shutil.move(str(image_path), str(dest))
    sidecar = image_path.with_suffix(".json")
    if sidecar.exists():
        side_dest = dest.with_suffix(".json")
        shutil.move(str(sidecar), str(side_dest))
    return dest


def _copy_to_raw_with_label(
    image_path: Path, label_text: str, meta: dict | None = None
) -> Path:
    """Copy still into raw/ as vial_NNNN; keep a tiny camera sidecar when known.

    Review used to drop cam0/cam1 identity (stem rename + JSON delete), which
    made train/val/test unstratifiable by Orbbec angle. Preserving ``camera``
    on accept enables weak-angle oversample / per-cam eval later.
    """
    RAW_DATASET.mkdir(parents=True, exist_ok=True)
    dest = RAW_DATASET / f"vial_{next_raw_index():04d}.jpg"
    shutil.copy2(image_path, dest)
    dest.with_suffix(".txt").write_text(label_text, encoding="utf-8")
    camera = None
    if meta and meta.get("camera") is not None:
        try:
            camera = int(meta["camera"])
        except (TypeError, ValueError):
            camera = meta.get("camera")
    if camera is None:
        # Recover from pending stem cam0_* / cam1_* when sidecar was incomplete.
        stem = image_path.stem
        if stem.startswith("cam") and len(stem) > 3 and stem[3].isdigit():
            try:
                camera = int(stem[3])
            except ValueError:
                camera = None
    if camera is not None or meta:
        side = {
            "camera": camera,
            "source_stem": image_path.stem,
            "raw_name": dest.name,
        }
        # Keep metric fields for 3D vial template fit (vial_3d_model.py).
        if meta:
            for key in ("box", "distance_mm", "x_mm", "y_mm", "image_width", "image_height"):
                if meta.get(key) is not None:
                    side[key] = meta[key]
        dest.with_suffix(".json").write_text(
            json.dumps(side, indent=2) + "\n", encoding="utf-8"
        )
    return dest


MIN_DRAW_BOX_PX = 4


@dataclass
class BoxDrawState:
    """Mouse-drag state for drawing one full-vial box (image-pixel coords)."""

    scale: float = 1.0
    boxes: list[tuple[float, float, float, float]] | None = None
    drawing: bool = False
    start: tuple[float, float] = (0.0, 0.0)
    current: tuple[float, float] = (0.0, 0.0)
    status: str = ""

    def __post_init__(self) -> None:
        if self.boxes is None:
            self.boxes = []

    def to_image(self, x: int, y: int) -> tuple[float, float]:
        return x / self.scale, y / self.scale

    def on_mouse(self, event, x, y, flags, param=None) -> None:
        """Accept any drag >= MIN_DRAW_BOX_PX; aspect is checked only on save."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = self.to_image(x, y)
            self.current = self.start
            self.status = ""
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current = self.to_image(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            end = self.to_image(x, y)
            left, right = sorted((self.start[0], end[0]))
            top, bottom = sorted((self.start[1], end[1]))
            if right - left >= MIN_DRAW_BOX_PX and bottom - top >= MIN_DRAW_BOX_PX:
                assert self.boxes is not None
                self.boxes.clear()
                self.boxes.append((left, top, right, bottom))
                box = (left, top, right - left, bottom - top)
                if is_review_acceptable_box(box):
                    if is_full_vial_aspect(box):
                        kind = "full vial"
                    elif _is_sideways_box(box) and not _is_topdown_box(box):
                        kind = "sideways vial"
                    elif _is_topdown_box(box):
                        kind = "top-down lid"
                    else:
                        kind = "full vial"
                    self.status = (
                        f"Box ok ({kind}, h/w={box_aspect_hw(box):.2f}) — press Q to save"
                    )
                else:
                    self.status = review_box_reject_message(box)
            else:
                self.status = "Drag a larger box around the full vial."

    def xywh(self) -> tuple[float, float, float, float] | None:
        if not self.boxes:
            return None
        x1, y1, x2, y2 = self.boxes[0]
        return (x1, y1, x2 - x1, y2 - y1)

    def try_save_label(self, width: int, height: int) -> tuple[str | None, str]:
        """Return (yolo_line_or_None, status_message). Never drops the drawn box."""
        box = self.xywh()
        if box is None:
            return None, "Draw a box before saving, or press Esc to cancel."
        if not is_review_acceptable_box(box):
            msg = review_box_reject_message(box)
            self.status = msg
            return None, msg
        return box_to_yolo(box, width, height) + "\n", ""


def _fit_review_scale(width: int, height: int) -> float:
    return min(1400 / max(width, 1), 900 / max(height, 1), 3.0)


def relabel_incorrect(image_path: Path, window: str | None = None) -> Path | None:
    """Interactive draw-a-box on the review (or a dedicated) window; save into raw/.

    Uses the same OpenCV window the user already sees when possible so mouse
    drags are not lost to a hidden second window behind the review UI.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not open {image_path} for relabeling.")
        return None
    height, width = image.shape[:2]
    scale = _fit_review_scale(width, height)
    state = BoxDrawState(scale=scale)
    win = window or "Incorrect vial — draw the tight box"
    own_window = window is None

    if own_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Show first, then bind mouse — required on some Windows OpenCV builds.
    display0 = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
    cv2.imshow(win, display0)
    cv2.setMouseCallback(win, state.on_mouse)
    # Drain any leftover key (e.g. the 'I' that opened this mode).
    cv2.waitKey(1)

    print("Draw from TOP OF LID through BOTTOM OF CLEAR CUP.")
    print("Q / Enter / S saves to training; Esc cancels; Z undo.")
    saved = None
    while True:
        display = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        assert state.boxes is not None
        for x1, y1, x2, y2 in state.boxes:
            box = (x1, y1, x2 - x1, y2 - y1)
            color = (0, 220, 0) if is_review_acceptable_box(box) else (0, 0, 220)
            p1 = (int(x1 * scale), int(y1 * scale))
            p2 = (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(display, p1, p2, color, 2)
        if state.drawing:
            p1 = (int(state.start[0] * scale), int(state.start[1] * scale))
            p2 = (int(state.current[0] * scale), int(state.current[1] * scale))
            cv2.rectangle(display, p1, p2, (0, 220, 255), 2)
        lines = [
            "I: vial is here, but the tracked box was wrong",
            "Drag lid→cup (side) or near-square lid (top-down)",
            "Q / Enter / S save   Esc cancel   Z undo",
        ]
        if state.status:
            lines.insert(1, state.status)
        bad_status = (
            state.status.upper().startswith("TOO")
            or "wrong" in state.status.lower()
            or "almost flat" in state.status.lower()
            or "too tall" in state.status.lower()
        )
        banner_color = (0, 0, 255) if bad_status else (255, 255, 255)
        if state.xywh() is not None and is_review_acceptable_box(state.xywh()):
            banner_color = (180, 255, 180)
        draw_label(display, (12, 24), lines, banner_color)
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("z"), ord("Z")) and state.boxes:
            state.boxes.pop()
            state.status = "Box cleared — drag again."
        elif key == 27:
            print("I relabel cancelled; still archived under review/incorrect/.")
            break
        elif key in (ord("q"), ord("Q"), ord("s"), ord("S"), 13):
            label, msg = state.try_save_label(width, height)
            if label is None:
                print(msg)
                continue
            meta = load_sidecar(image_path.with_suffix(".json"))
            saved = _copy_to_raw_with_label(image_path, label, meta=meta)
            print(f"Corrected box saved to {saved.name} for the next train.")
            break
    cv2.setMouseCallback(win, lambda *args: None)
    if own_window:
        cv2.destroyWindow(win)
    return saved


def accept_correct(image_path: Path, meta: dict) -> Path | None:
    box = meta.get("box")
    width = int(meta.get("image_width") or 0)
    height = int(meta.get("image_height") or 0)
    if not box or width <= 0 or height <= 0:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read {image_path}")
            return None
        height, width = image.shape[:2]
    if not box or len(box) != 4:
        print(f"No box in sidecar for {image_path.name}; skipping R.")
        return None
    box_t = tuple(int(v) for v in box)
    if not is_review_acceptable_box(box_t):
        print(f"R rejected: {review_box_reject_message(box_t)}")
        return None
    label = box_to_yolo(box_t, width, height) + "\n"
    dest = _copy_to_raw_with_label(image_path, label, meta=meta)
    image_path.unlink(missing_ok=True)
    image_path.with_suffix(".json").unlink(missing_ok=True)
    if is_full_vial_aspect(box_t):
        kind = "full vial"
    elif _is_sideways_box(box_t) and not _is_topdown_box(box_t):
        kind = "sideways vial"
    elif _is_topdown_box(box_t):
        kind = "top-down lid"
    else:
        kind = "full vial"
    print(f"R: copied correct {kind} box into training as {dest.name}")
    return dest


def mark_wrong(image_path: Path) -> Path:
    """Not a vial: empty label for training, and keep a copy under review/wrong/."""
    meta = load_sidecar(image_path.with_suffix(".json"))
    wrong = _move_pair(image_path, REVIEW_WRONG)
    dest = _copy_to_raw_with_label(wrong, "", meta=meta)
    wrong.with_suffix(".txt").write_text("", encoding="utf-8")
    print(f"W: empty-label negative -> {dest.name}; archive {wrong.name}")
    return dest



def mark_incorrect(image_path: Path, window: str | None = None) -> Path | None:
    moved = _move_pair(image_path, REVIEW_INCORRECT)
    print(f"I: moved to {moved} — draw the real vial box.")
    return relabel_incorrect(moved, window=window)


def review_pending(files: list[Path] | None = None) -> None:
    pending = REVIEW_PENDING
    if files is None:
        if not pending.exists():
            print("No review stills waiting.")
            return
        files = sorted(pending.glob("*.jpg"))
    else:
        files = [path for path in files if path.exists()]
    if not files:
        print("No review stills waiting.")
        return

    window = "Review vial stills"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print()
    print("Review keys (also shown on screen):")
    print("  R  correct FULL vial box (lid through cup) — copy into training")
    print("  W  not a vial — empty-label negative")
    print("  I  incorrect / too-short box — draw a new full-vial box")
    print("  N  skip — leave pending")
    print("  Q  end review")
    index = 0
    while 0 <= index < len(files):
        path = files[index]
        if not path.exists():
            index += 1
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"Could not read {path}")
            index += 1
            continue
        meta = load_sidecar(path.with_suffix(".json"))
        view = image.copy()
        draw_review_help(view, index, len(files), meta)
        cv2.imshow(window, view)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("n"), ord("N")):
            print(f"N: left {path.name} pending")
            index += 1
            continue
        if key in (ord("r"), ord("R")):
            if accept_correct(path, meta) is None:
                continue
            index += 1
            continue
        if key in (ord("w"), ord("W")):
            mark_wrong(path)
            index += 1
            continue
        if key in (ord("i"), ord("I")):
            # Reuse this window so the drag lands on the visible image
            # (a second "Incorrect vial" window often sits behind and gets no mouse).
            mark_incorrect(path, window=window)
            cv2.setMouseCallback(window, lambda *args: None)
            index += 1
            continue
    cv2.destroyWindow(window)
    left = len(list(pending.glob("*.jpg"))) if pending.exists() else 0
    print(f"Review finished. Pending stills left: {left}")
    print("When ready: python train_vial_model.py train")


def list_orbbec_indexes(max_index: int = 6) -> list[int]:
    """Return Orbbec OBSENSOR indexes that open on this PC.

    Probing empty indexes makes OpenCV print C++ 'index out of range' lines on
    stderr. Those are muted for the probe only.
    """
    import os

    found: list[int] = []
    devnull = open(os.devnull, "w")
    try:
        old_err = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            for index in range(max_index):
                cap = cv2.VideoCapture(index, cv2.CAP_OBSENSOR)
                opened = cap.isOpened()
                cap.release()
                if opened:
                    found.append(index)
                elif found:
                    break
        finally:
            os.dup2(old_err, 2)
            os.close(old_err)
    finally:
        devnull.close()
    return found


def print_orbbec_cameras(max_index: int = 6) -> list[int]:
    found = list_orbbec_indexes(max_index)
    if not found:
        print("No Orbbec cameras detected by OpenCV.")
        print(USB_TIP)
        return found
    print(f"Orbbec cameras ready at index(es): {', '.join(str(i) for i in found)}")
    if len(found) >= 2:
        print(f"Dual cam: python depthcamera.py --camera {found[0]} --camera2 {found[1]}")
        print("Or shorter: python depthcamera.py --dual")
    else:
        print(f"Only one Orbbec seen. Single cam: python depthcamera.py --camera {found[0]}")
    return found


def warm_capture(cap: cv2.VideoCapture, tries: int = 45) -> tuple[object, object] | tuple[None, None]:
    """First Orbbec frames are often empty — wait until color and depth both arrive."""
    for _ in range(tries):
        if not cap.grab():
            time.sleep(0.03)
            continue
        ok_color, color = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
        ok_depth, depth = cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
        if ok_color and color is not None and ok_depth and depth is not None:
            return color, depth
        time.sleep(0.03)
    return None, None


def waiting_frame(message: str, width: int = 640, height: int = 480):
    view = np.zeros((height, width, 3), dtype=np.uint8)
    view[:] = (36, 36, 36)
    cv2.putText(view, message, (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    return view


class OrbbecDepthCamera:
    def __init__(
        self,
        model_path: str | Path | None = None,
        camera_index: int = 0,
        confidence: float = DEFAULT_YOLO_CONF,
        detector: VialDetector | None = None,
        window: str | None = None,
        gates: CameraGateProfile | None = None,
        *,
        use_3d_model_score: bool = DEFAULT_USE_3D_MODEL_SCORE,
        model_score_min: float = DEFAULT_3D_MODEL_SCORE_MIN,
        use_edge_assist: bool = DEFAULT_USE_EDGE_ASSIST,
        use_orb_assist: bool = DEFAULT_USE_ORB_ASSIST,
        orb_assist_weight: float = DEFAULT_ORB_ASSIST_WEIGHT,
    ):
        self.camera_index = camera_index
        self.window = window or window_title(camera_index, dual=False)
        self.gates = gates or strong_gate_profile(yolo_conf=confidence)
        # Live size-score stays off unless LIVE_3D_VIAL_ENABLED is flipped on.
        self.use_3d_model_score = bool(use_3d_model_score) and LIVE_3D_VIAL_ENABLED
        self.model_score_min = float(model_score_min)
        self.use_edge_assist = bool(use_edge_assist)
        self.use_orb_assist = bool(use_orb_assist)
        self.orb_assist_weight = float(orb_assist_weight)
        self._peer_orb_desc: np.ndarray | None = None
        if detector is not None:
            self.detector = detector
        else:
            explicit = None if model_path is None else Path(model_path)
            weights = find_weights(explicit)
            if explicit is not None and weights is None:
                print(f"Model not found: {explicit}")
                print("Tracking the white cap until a trained model is available.")
            self.detector = VialDetector(weights, confidence)
        self.capture = None
        self.intrinsics = None
        self.box = None
        self.misses = 0
        # "no_vial" = truly empty proposals; "low_confidence" = proposals failed gates.
        self.miss_reason: str = "no_vial"
        self.distance_mm = None
        self.depth_samples: list[float] = []
        self.preview = None
        # All current local detections. ``box``/the locate return remain the primary
        # tracked fix for IPC and callers that expect one distance.
        self.fixes: list[VialFix] = []
        self._tracked_boxes: list[tuple[int, int, int, int]] = []
        self.tracks: list[VialTrack] = []
        self._next_track_id = 1
        self._primary_track_id: int | None = None
        self._roi_frame = 0
        self._roi_misses = 0
        self.last_model_score: float | None = None
        self.last_detect_source: str | None = None

    def open(
        self,
        *,
        require_frame: bool = True,
        retries: int = 3,
        warm_tries: int = 45,
    ) -> None:
        last_error = None
        for attempt in range(1, retries + 1):
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_OBSENSOR)
            if not self.capture.isOpened():
                last_error = f"Could not open Orbbec camera index {self.camera_index}."
                time.sleep(0.35 * attempt)
                continue
            edge_note = "edge-assist on" if self.use_edge_assist else "edge-assist off"
            orb_note = (
                f"orb-assist on w={self.orb_assist_weight:.2f}"
                if self.use_orb_assist
                else "orb-assist off"
            )
            if not require_frame:
                print(
                    f"Camera {self.camera_index}: detector {self.detector.name} "
                    f"gates={self.gates.name} conf={self.gates.yolo_conf:.2f} "
                    f"{edge_note} {orb_note}"
                )
                return
            color, depth = warm_capture(self.capture, tries=warm_tries)
            if color is not None and depth is not None:
                print(
                    f"Camera {self.camera_index}: detector {self.detector.name} "
                    f"gates={self.gates.name} conf={self.gates.yolo_conf:.2f} "
                    f"{edge_note} {orb_note} ({color.shape[1]}x{color.shape[0]})"
                )
                return
            last_error = (
                f"Orbbec index {self.camera_index} opened but never returned color+depth."
            )
            self.capture.release()
            self.capture = None
            time.sleep(0.35 * attempt)
        raise RuntimeError(f"{last_error} {USB_TIP}")

    def close(self) -> None:
        """Release this camera only. Does not call destroyAllWindows (safe for multi-process)."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        # Only this process's window — never destroyAllWindows here (other processes keep theirs).
        try:
            cv2.destroyWindow(self.window)
        except cv2.error:
            pass

    def read_images(self):
        if self.capture is None or not self.capture.grab():
            return None, None
        ok_color, color = self.capture.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
        ok_depth, depth = self.capture.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
        if not ok_color or color is None or not ok_depth or depth is None:
            return None, None
        depth_mm = depth_to_mm(depth)
        if not getattr(self, "_logged_stream_shapes", False):
            self._logged_stream_shapes = True
            ch, cw = color.shape[:2]
            dh, dw = depth_mm.shape[:2]
            if (ch, cw) == (dh, dw):
                note = (
                    "same size as color — treat as already D2C-aligned "
                    "(confirm a hand outline overlaps in both panes)"
                )
            else:
                note = (
                    "native depth size, not D2C. CAP_OBSENSOR does not run "
                    "Orbbec AlignFilter; viewers must map color pixels into "
                    "this frame. Do not treat cv2.resize as alignment."
                )
            print(
                f"Camera {self.camera_index}: color {cw}x{ch} depth {dw}x{dh} — {note}"
            )
        return color, depth_mm

    def locate(
        self,
        color,
        depth_mm: np.ndarray,
        *,
        peer_confirmed: bool = False,
        soft_clear: bool = False,
    ) -> VialFix | None:
        height, width = color.shape[:2]
        depth_mm = align_depth(depth_mm, width, height)
        if self.intrinsics is None and self.capture is not None:
            self.intrinsics = intrinsics_from_capture(self.capture, width, height)
        gates = self.gates

        def ok(box, *, use_cap_fallback: bool = False) -> bool:
            return accepts_vial_box(
                depth_mm,
                color,
                box,
                peer_confirmed=peer_confirmed,
                soft_clear=soft_clear,
                cap_fallback=use_cap_fallback,
                gates=gates,
                intrinsics=self.intrinsics,
            )

        self.preview = focus_near(color, depth_mm)
        raw_yolo: list[tuple[int, int, int, int]] = []
        accepted: list[tuple[tuple[int, int, int, int], bool]] = []

        def validate_yolo(proposals):
            """Return every independently accepted YOLO proposal and its gate mode."""
            prepared: list[tuple[int, int, int, int]] = []
            seen: set[tuple[int, int, int, int]] = set()
            for proposal in proposals:
                box = tuple(proposal)
                if (
                    usable_detection(self.preview, box, gates=gates)
                    and within_reach(depth_mm, color, box)
                    and vial_scale(depth_mm, color, box, self.intrinsics)
                ):
                    prepared.append(box)
                    seen.add(box)
                    continue
                # A wide raw proposal may be a sideways vial whose body needs growing.
                if sideways_assist_enabled():
                    grown = expand_wide_proposal_sideways(color, depth_mm, box)
                    if (
                        grown is not None
                        and tuple(grown) not in seen
                        and usable_detection(self.preview, grown, gates=gates)
                        and within_reach(depth_mm, color, grown)
                        and vial_scale(depth_mm, color, grown, self.intrinsics)
                    ):
                        prepared.append(tuple(grown))
                        seen.add(tuple(grown))
            result: list[tuple[tuple[int, int, int, int], bool]] = []
            for box in prepared:
                # Gate the original first, then a lid-width-tightened variant.
                # Fitting only after acceptance made a padded second vial
                # impossible to recover even when YOLO proposed it strongly.
                candidates = [box]
                tightened = tighten_yolo_box_to_lid(color, depth_mm, box)
                if tightened != box:
                    candidates.append(tightened)
                for candidate in candidates:
                    if not (
                        usable_detection(self.preview, candidate, gates=gates)
                        and within_reach(depth_mm, color, candidate)
                        and vial_scale(depth_mm, color, candidate, self.intrinsics)
                    ):
                        continue
                    if ok(candidate):
                        result.append((candidate, False))
                        break
                    if ok(candidate, use_cap_fallback=True):
                        result.append((candidate, True))
                        break
            return result

        if self.detector.model is not None:
            self._roi_frame += 1
            roi_groups = self.detector.detect_in_rois(
                self.preview, self._tracked_boxes, conf=gates.yolo_conf
            )
            roi_missed = False
            for group in roi_groups:
                raw_yolo.extend(group)
                group_accepted = validate_yolo(group)
                accepted.extend(group_accepted)
                if not group_accepted:
                    roi_missed = True
            periodic_full = self._roi_frame % ROI_FULL_SEARCH_INTERVAL == 0
            # Full search discovers new vials even while every existing ROI remains
            # healthy. Use a slightly lower proposal threshold here; all proposals
            # still pass the normal per-vial depth, scale, and clear-body gates.
            if not self._tracked_boxes or roi_missed or periodic_full:
                discovery_conf = max(
                    MIN_DISCOVERY_YOLO_CONF,
                    gates.yolo_conf - DISCOVERY_YOLO_CONF_DELTA,
                )
                full_yolo = self.detector.detect(self.preview, conf=discovery_conf)
                raw_yolo.extend(full_yolo)
                accepted.extend(validate_yolo(full_yolo))

        # Keep gate mode per proposal. A frame-wide fallback flag used to re-gate all
        # objects as if they were one vial, then a second global structure gate could
        # discard the weaker sibling. Each proposal now survives on its own evidence.
        boxes = [box for box, _fallback in accepted]
        cap_raw: list[tuple[int, int, int, int]] = []
        sideways_raw: list[tuple[int, int, int, int]] = []
        edge_raw: list[tuple[int, int, int, int]] = []
        used_cap_fallback = any(fallback for _box, fallback in accepted)
        used_sideways_winner = False
        used_edge_assist = False
        # Cap proposals also augment YOLO instead of running only when the whole
        # frame has no accepted box. NMS below merges duplicate proposals.
        cap_raw = detect_vial_by_cap(color, depth_mm, self.intrinsics)
        strict_cap = [
            box
            for box in cap_raw
            if ok(box)
            and within_reach(depth_mm, color, box)
            and vial_scale(depth_mm, color, box, self.intrinsics)
        ]
        fallback_cap = [
            box
            for box in cap_raw
            if box not in strict_cap
            and ok(box, use_cap_fallback=True)
            and within_reach(depth_mm, color, box)
            and vial_scale(depth_mm, color, box, self.intrinsics)
        ]
        boxes.extend(strict_cap)
        boxes.extend(fallback_cap)
        accepted.extend((box, False) for box in strict_cap)
        accepted.extend((box, True) for box in fallback_cap)
        used_cap_fallback = used_cap_fallback or bool(fallback_cap)
        # Promoted race winner: only run after YOLO/cap paths miss, so its
        # horizontal proposals can never displace an accepted upright detection.
        if not boxes and sideways_assist_enabled():
            sideways_raw = detect_vial_by_sideways_winner(
                color, depth_mm, self.intrinsics
            )
            sideways_raw = [
                box
                for box in sideways_raw
                if _sideways_pose(box, gates=gates, image=color)
            ]
            boxes = [
                box
                for box in sideways_raw
                if ok(box)
                and within_reach(depth_mm, color, box)
                and vial_scale(depth_mm, color, box, self.intrinsics)
            ]
            sideways_fallback = False
            if not boxes:
                boxes = [
                    box
                    for box in sideways_raw
                    if ok(box, use_cap_fallback=True)
                    and within_reach(depth_mm, color, box)
                    and vial_scale(depth_mm, color, box, self.intrinsics)
                ]
                sideways_fallback = bool(boxes)
                used_cap_fallback = bool(boxes)
            accepted.extend((box, sideways_fallback) for box in boxes)
            used_sideways_winner = bool(boxes)
        if not boxes and self.use_edge_assist:
            # Weak cam: slightly looser edge floors + more proposals when YOLO misses.
            edge_n = 8 if gates.name == "weak" else 6
            edge_raw = detect_vial_by_edges(color, max_proposals=edge_n, gates=gates)
            boxes = [
                box
                for box in edge_raw
                if ok(box)
                and within_reach(depth_mm, color, box)
                and vial_scale(depth_mm, color, box, self.intrinsics)
            ]
            edge_fallback = False
            if not boxes:
                boxes = [
                    box
                    for box in edge_raw
                    if ok(box, use_cap_fallback=True)
                    and within_reach(depth_mm, color, box)
                    and vial_scale(depth_mm, color, box, self.intrinsics)
                ]
                edge_fallback = bool(boxes)
                used_cap_fallback = bool(boxes)
            accepted.extend((box, edge_fallback) for box in boxes)
            used_edge_assist = bool(boxes)

        fallback_by_box: dict[tuple[int, int, int, int], bool] = {}
        for accepted_box, fallback in accepted:
            fallback_by_box.setdefault(tuple(accepted_box), bool(fallback))

        def ok_final(box) -> bool:
            return ok(
                box,
                use_cap_fallback=fallback_by_box.get(tuple(box), used_cap_fallback),
            )

        def keeps_gate(box, *, use_cap_fallback: bool) -> bool:
            return (
                ok(box, use_cap_fallback=use_cap_fallback)
                and vial_scale(depth_mm, color, box, self.intrinsics)
                and within_reach(depth_mm, color, box)
            )

        # Any proposal that existed before clear-vial acceptance — used for honest miss text.
        had_candidates = (
            bool(raw_yolo)
            or bool(cap_raw)
            or bool(sideways_raw)
            or bool(edge_raw)
            or bool(accepted)
        )
        # fit_vial_shape can over-shrink; keep the pre-fit box when fit fails gates.
        kept: list[tuple[int, int, int, int]] = []
        for box in boxes:
            use_fallback = fallback_by_box.get(tuple(box), False)
            fitted = fit_vial_shape(color, depth_mm, box)
            if keeps_gate(fitted, use_cap_fallback=use_fallback):
                kept.append(fitted)
                fallback_by_box.setdefault(tuple(fitted), use_fallback)
            else:
                # The original proposal was already independently accepted. Do not
                # impose a second edge/structure veto that disproportionately drops
                # a dimmer second vial after a brighter sibling has passed.
                kept.append(box)
        boxes = kept
        # Live 3D score filter is a no-op while LIVE_3D_VIAL_ENABLED is False.
        boxes = apply_3d_model_score_filter(
            boxes,
            depth_mm,
            color,
            self.intrinsics,
            enabled=self.use_3d_model_score,
            min_score=self.model_score_min,
            image_size=(width, height),
        )
        # fit/cap/sideways paths can produce several boxes for one object. Merge
        # those here, but retain separate vials and cap the final per-camera set.
        boxes = nms_vial_boxes(boxes)
        chosen = choose_box(
            boxes,
            self.box,
            color,
            depth_mm,
            peer_confirmed=peer_confirmed,
            soft_clear=soft_clear,
            cap_fallback=used_cap_fallback,
            gates=gates,
            intrinsics=self.intrinsics,
            use_3d_model_score=False,
            use_edge_rank=self.use_edge_assist,
            peer_orb_desc=self._peer_orb_desc if self.use_orb_assist else None,
            orb_assist_weight=self.orb_assist_weight if self.use_orb_assist else 0.0,
        )
        held = False
        if chosen is not None and not (
            ok_final(chosen)
            and within_reach(depth_mm, color, chosen)
            and vial_scale(depth_mm, color, chosen, self.intrinsics)
        ):
            chosen = None

        # Associate every accepted box with persistent state. Previously only
        # ``chosen`` was smoothed/held; all sibling fixes were display-only and had
        # no miss counter or ROI identity, which made live tracking serial-looking.
        matches, unmatched_tracks, unmatched_detections = associate_vial_tracks(
            self.tracks, boxes
        )
        detection_to_track: dict[int, VialTrack] = {}
        for track_index, detection_index in matches.items():
            track = self.tracks[track_index]
            track.box = smooth_box(track.box, boxes[detection_index])
            track.misses = 0
            detection_to_track[detection_index] = track
        for track_index in unmatched_tracks:
            self.tracks[track_index].misses += 1
        for detection_index in unmatched_detections:
            if len(self.tracks) >= MAX_VIALS_PER_CAMERA:
                break
            track = VialTrack(self._next_track_id, tuple(float(v) for v in boxes[detection_index]))
            self._next_track_id += 1
            self.tracks.append(track)
            detection_to_track[detection_index] = track
        self.tracks = [
            track for track in self.tracks if track.misses <= TRACK_MAX_MISSES
        ]

        chosen_track = None
        if chosen is not None:
            try:
                chosen_index = boxes.index(chosen)
            except ValueError:
                chosen_index = -1
            chosen_track = detection_to_track.get(chosen_index)
        live_ids = {track.track_id for track in self.tracks}
        if chosen_track is not None:
            self._primary_track_id = chosen_track.track_id
        elif self._primary_track_id not in live_ids:
            self._primary_track_id = self.tracks[0].track_id if self.tracks else None

        fixes_by_id: dict[int, VialFix] = {}
        for track in self.tracks:
            box = clip_box(track.box, width, height)
            held = track.misses > 0
            if not held:
                measured, sample_box = measure_cap_mm(depth_mm, color, box)
                if measured is not None:
                    track.depth_samples.append(measured)
                    del track.depth_samples[:-5]
                    track.distance_mm = float(np.median(track.depth_samples))
                track.sample_box = sample_box
            else:
                sample_box = track.sample_box
            if sample_box is not None:
                sx, sy, sw, sh = sample_box
                center = (sx + sw // 2, sy + sh // 2)
            else:
                center = (box[0] + box[2] // 2, box[1] + box[3] // 2)
            x_mm, y_mm, _slant = position_mm(
                center, track.distance_mm, self.intrinsics
            )
            fixes_by_id[track.track_id] = VialFix(
                box,
                track.distance_mm,
                x_mm,
                y_mm,
                track.distance_mm,
                held,
                sample_box,
                None,
                track.track_id,
            )

        ordered_ids = []
        if self._primary_track_id in fixes_by_id:
            ordered_ids.append(self._primary_track_id)
        ordered_ids.extend(
            track.track_id
            for track in self.tracks
            if track.track_id not in ordered_ids
        )
        self.fixes = [fixes_by_id[track_id] for track_id in ordered_ids]
        self._tracked_boxes = [
            clip_box(track.box, width, height)
            for track in self.tracks[:MAX_VIALS_PER_CAMERA]
        ]

        if not self.fixes:
            self.misses += 1
            self._roi_misses += 1
            self.box = None
            self.distance_mm = None
            self.depth_samples.clear()
            self.last_model_score = None
            self.last_detect_source = None
            self.miss_reason = "low_confidence" if had_candidates else "no_vial"
            return None

        any_current = any(not fix.held for fix in self.fixes)
        self.misses = 0 if any_current else self.misses + 1
        self._roi_misses = 0 if any_current else self._roi_misses + 1
        self.miss_reason = "no_vial"
        if used_sideways_winner and any_current:
            self.last_detect_source = "sideways-race-winner"
        elif used_edge_assist and any_current:
            self.last_detect_source = "edges"
        else:
            self.last_detect_source = None
        primary = self.fixes[0]
        primary_track = next(
            track for track in self.tracks if track.track_id == primary.track_id
        )
        self.box = primary_track.box
        self.distance_mm = primary.distance_mm
        self.depth_samples = primary_track.depth_samples
        self.last_model_score = None
        return primary

    def run(
        self,
        snapshot: Path | None = None,
        *,
        peer_ipc: PeerIpc | None = None,
        corroborate_abs_cm: float = CORROBORATE_ABS_CM,
        corroborate_rel: float = CORROBORATE_REL,
        reproject_assist: bool = False,
    ) -> VialFix | None:
        """Single-camera path. Optional peer_ipc corroborates cm; opt-in 3D reproject.

        Classic sticky soft peer invent stays off. ``reproject_assist`` may propose a
        geometrically reprojected box from the peer IPC report when extrinsics are real;
        local clear-vial / soft-clear gates must still accept it.
        """
        want_reproject = bool(reproject_assist) and LIVE_3D_REPROJECT_ASSIST_ALLOWED
        recorder = SnapshotRecorder()
        self.open()
        independent = peer_ipc is not None
        if independent:
            print(
                f"Independent camera {self.camera_index}: detect alone; "
                f"IPC peer={peer_ipc.peer_id} port={peer_ipc.listen_port}→{peer_ipc.peer_port} "
                f"(corroborate cm; no sticky soft-locate)."
            )
            _note_live_3d_status(want_reproject=want_reproject)
        print("Q quits. Stills auto-save every few seconds when a vial is in view.")
        last_print = 0.0
        latest = None
        try:
            while True:
                color, depth_mm = self.read_images()
                if color is None or depth_mm is None:
                    if peer_ipc is not None:
                        peer_ipc.publish(
                            peer_report_from_fix(
                                self.camera_index, None, conf=self.gates.yolo_conf
                            )
                        )
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
                    continue
                # Poll peer ORB first so choose_box can apply a tiny match boost.
                used_reproject = False
                corr: CorroborationResult | None = None
                orb_line: str | None = None
                if peer_ipc is not None:
                    peer = peer_ipc.latest()
                    if self.use_orb_assist and peer is not None and peer.orb_desc:
                        self._peer_orb_desc = unpack_orb_descriptors(peer.orb_desc)
                    else:
                        self._peer_orb_desc = None
                # Standalone locate — soft-clear local only; never sticky peer assist.
                latest = locate_standalone(self, color, depth_mm)
                log_found_vials(self.camera_index, self.fixes)
                if peer_ipc is not None:
                    peer = peer_ipc.latest()
                    if latest is None and want_reproject:
                        helped = try_3d_reproject_from_peer_report(
                            self, (color, depth_mm), peer, soft_clear=True
                        )
                        if helped is not None:
                            latest = helped
                            used_reproject = True
                            self.last_detect_source = "3d_reproject"
                    report = peer_report_from_fix(
                        self.camera_index,
                        latest,
                        conf=self.gates.yolo_conf,
                        image=color,
                        use_orb=self.use_orb_assist,
                    )
                    peer_ipc.publish(report)
                    peer = peer_ipc.latest()
                    corr = corroborate_distances(
                        report,
                        peer,
                        abs_cm=corroborate_abs_cm,
                        rel=corroborate_rel,
                    )
                    if (
                        self.use_orb_assist
                        and report.orb_desc
                        and peer is not None
                        and peer.orb_desc
                    ):
                        orb_line = orb_agree_label(
                            unpack_orb_descriptors(report.orb_desc),
                            unpack_orb_descriptors(peer.orb_desc),
                        )
                if snapshot is None:
                    recorder.maybe_save(self.camera_index, color, latest)
                help_lines = [
                    "Q quits",
                    f"stills {recorder.saved}/{recorder.max_saves}",
                ]
                if corr is not None:
                    help_lines.append(corr.overlay_line)
                if orb_line:
                    help_lines.append(orb_line)
                if used_reproject:
                    help_lines.append("3d reproject")
                if latest is not None and self.last_detect_source == "edges":
                    help_lines.append("edges")
                view = annotate(
                    self.preview if self.preview is not None else color,
                    latest,
                    fixes=self.fixes,
                    camera_label=f"camera {self.camera_index}",
                    help_lines=help_lines,
                    miss_reason=None if latest is not None else self.miss_reason,
                )
                now = time.time()
                if now - last_print > 1:
                    last_print = now
                    if latest is None or latest.distance_mm is None:
                        msg = f"Camera {self.camera_index}: no reading"
                    else:
                        away = (latest.z_mm or latest.distance_mm) / 10
                        cx, cy = latest.center
                        msg = f"Camera {self.camera_index}: {away:.1f} cm at ({cx}, {cy})"
                    if self.fixes:
                        track_ids = ",".join(
                            str(fix.track_id)
                            for fix in self.fixes
                            if fix.track_id is not None
                        )
                        suffix = f" [{track_ids}]" if track_ids else ""
                        msg = f"{msg} | {len(self.fixes)} vials{suffix}"
                    if corr is not None:
                        msg = f"{msg} | {corr.overlay_line}"
                    if orb_line:
                        msg = f"{msg} | {orb_line}"
                    print(msg)
                if snapshot is not None:
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snapshot), view)
                    print(f"Saved {snapshot}")
                    return latest
                cv2.imshow(self.window, view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
        finally:
            if peer_ipc is not None:
                peer_ipc.close()
            self.close()
        if snapshot is None and recorder.session_files:
            review_pending(recorder.session_files)
        return latest


def build_shared_detector(model_path: Path | None, confidence: float) -> VialDetector:
    explicit = None if model_path is None else Path(model_path)
    weights = find_weights(explicit)
    if explicit is not None and weights is None:
        print(f"Model not found: {explicit}")
        print("Tracking the white cap until a trained model is available.")
    return VialDetector(weights, confidence)


def open_cameras(primary: OrbbecDepthCamera, secondary: OrbbecDepthCamera | None) -> list[OrbbecDepthCamera]:
    available = list_orbbec_indexes()
    print(f"Detected Orbbec index(es): {available if available else 'none'}")
    if available and primary.camera_index not in available:
        raise RuntimeError(
            f"Primary index {primary.camera_index} is not among detected Orbbecs {available}. "
            f"Try --camera {available[0]}."
        )
    opened: list[OrbbecDepthCamera] = []
    primary.open()
    opened.append(primary)
    if secondary is None:
        return opened
    if available and secondary.camera_index not in available:
        print(
            f"Camera {secondary.camera_index} is not among detected Orbbecs {available}. "
            f"For two devices use --camera {available[0]} --camera2 {available[1] if len(available) > 1 else '?'}."
        )
        print(USB_TIP)
        return opened
    # Stagger second CAP_OBSENSOR open — simultaneous dual USB warm-up often fails.
    time.sleep(2.0)
    try:
        secondary.open()
        opened.append(secondary)
    except RuntimeError as error:
        print(f"Camera {secondary.camera_index} did not open — continuing with camera {primary.camera_index} only.")
        print(error)
        if len(available) >= 2:
            other = [i for i in available if i != primary.camera_index]
            if other:
                print(f"Hint: try --camera {primary.camera_index} --camera2 {other[0]}")
        print(USB_TIP)
    return opened


def run_cameras(
    cameras: list[OrbbecDepthCamera],
    snapshot: Path | None = None,
    *,
    reproject_assist: bool = False,
) -> None:
    want_reproject = bool(reproject_assist) and LIVE_3D_REPROJECT_ASSIST_ALLOWED
    if not cameras:
        raise RuntimeError(f"No Orbbec cameras opened. {USB_TIP}")
    dual = len(cameras) > 1
    for cam in cameras:
        cam.window = window_title(cam.camera_index, dual=dual)
        cv2.namedWindow(cam.window, cv2.WINDOW_NORMAL)
        cv2.imshow(cam.window, waiting_frame(f"Starting camera {cam.camera_index}..."))
    cv2.waitKey(1)
    if dual:
        print(f"Dual mode: windows for camera {cameras[0].camera_index} and camera {cameras[1].camera_index}.")
        print(
            f"Synchronized locate_pair + sticky peer memory ({PEER_MEMORY_FRAMES} frames): "
            "both cameras get strict+soft in one tick so the UI always sees paired results."
        )
        print_physical_dual_checklist()
        _note_live_3d_status(want_reproject=want_reproject)
        for cam in cameras:
            g = cam.gates
            print(
                f"  cam {cam.camera_index} gates={g.name} conf={g.yolo_conf:.2f} "
                f"hole_margin={g.clear_hole_margin:.2f} lid_hole_max={g.lid_hole_max:.2f} "
                f"aspect=[{g.aspect_min:.2f},{g.aspect_max:.2f}]"
            )
    else:
        print(f"Single mode: camera {cameras[0].camera_index} only.")
    recorder = SnapshotRecorder()
    print("Q quits all windows. Distances stay per camera (not averaged).")
    print(f"Auto-saves up to {recorder.max_saves} clean stills under {REVIEW_PENDING}")
    print(USB_TIP)
    last_print = 0.0
    miss_streak = {cam.camera_index: 0 for cam in cameras}
    peer_memory = PeerMemory(window=PEER_MEMORY_FRAMES)
    try:
        while True:
            now = time.time()
            quit_requested = False
            frames: dict[int, tuple] = {}
            latest_by_cam: dict[int, VialFix | None] = {}
            peer_assist = False
            for cam in cameras:
                color, depth_mm = cam.read_images()
                frames[cam.camera_index] = (color, depth_mm)
                if color is None or depth_mm is None:
                    miss_streak[cam.camera_index] += 1
                    if miss_streak[cam.camera_index] == 30:
                        print(
                            f"Camera {cam.camera_index}: still waiting for frames. "
                            "Check the USB 3 cable/port."
                        )
                    cv2.imshow(
                        cam.window,
                        waiting_frame(f"Camera {cam.camera_index}: waiting for frames..."),
                    )
                else:
                    miss_streak[cam.camera_index] = 0

            used_reproject = False
            if dual and len(cameras) >= 2:
                # Peer soft is applied inside locate_pair before either fix is returned.
                cam_a, cam_b = cameras[0], cameras[1]
                pair = locate_pair(
                    cam_a,
                    frames.get(cam_a.camera_index),
                    cam_b,
                    frames.get(cam_b.camera_index),
                    peer_memory=peer_memory,
                )
                fix_a, fix_b = pair.fix_a, pair.fix_b
                if want_reproject:
                    fix_a, fix_b, used_reproject = apply_3d_reproject_pair(
                        cam_a,
                        frames.get(cam_a.camera_index),
                        fix_a,
                        cam_b,
                        frames.get(cam_b.camera_index),
                        fix_b,
                    )
                latest_by_cam[cam_a.camera_index] = fix_a
                latest_by_cam[cam_b.camera_index] = fix_b
                peer_assist = pair.peer_assist
            else:
                for cam in cameras:
                    color, depth_mm = frames.get(cam.camera_index, (None, None))
                    if color is None or depth_mm is None:
                        latest_by_cam[cam.camera_index] = None
                        continue
                    latest_by_cam[cam.camera_index] = cam.locate(
                        color, depth_mm, peer_confirmed=False
                    )

            classic_orb_line: str | None = None
            if dual and len(cameras) >= 2:
                ca, cb = cameras[0], cameras[1]
                fa = latest_by_cam.get(ca.camera_index)
                fb = latest_by_cam.get(cb.camera_index)
                frame_a = frames.get(ca.camera_index, (None, None))[0]
                frame_b = frames.get(cb.camera_index, (None, None))[0]
                if (
                    ca.use_orb_assist
                    and fa is not None
                    and fb is not None
                    and frame_a is not None
                    and frame_b is not None
                ):
                    _, da = extract_orb_descriptors(frame_a, fa.box)
                    _, db = extract_orb_descriptors(frame_b, fb.box)
                    classic_orb_line = orb_agree_label(da, db)

            for cam in cameras:
                color, depth_mm = frames.get(cam.camera_index, (None, None))
                latest = latest_by_cam.get(cam.camera_index)
                if color is None or depth_mm is None:
                    continue
                if snapshot is None:
                    recorder.maybe_save(cam.camera_index, color, latest, now=now)
                help_lines = ["Q quits", f"stills {recorder.saved}/{recorder.max_saves}"]
                if dual and peer_assist and latest is not None:
                    help_lines.append("peer assist")
                if dual and classic_orb_line and latest is not None:
                    help_lines.append(classic_orb_line)
                if dual and used_reproject and latest is not None:
                    help_lines.append("3d reproject")
                if latest is not None and cam.last_detect_source == "edges":
                    help_lines.append("edges")
                view = annotate(
                    cam.preview if cam.preview is not None else color,
                    latest,
                    fixes=cam.fixes,
                    camera_label=f"camera {cam.camera_index}",
                    help_lines=help_lines,
                    miss_reason=None if latest is not None else cam.miss_reason,
                )
                if snapshot is not None and cam is cameras[0]:
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snapshot), view)
                    print(f"Saved {snapshot}")
                    return
                cv2.imshow(cam.window, view)
                if now - last_print > 1:
                    if latest is None or latest.distance_mm is None:
                        print(f"Camera {cam.camera_index}: no reading")
                    else:
                        away = (latest.z_mm or latest.distance_mm) / 10
                        cx, cy = latest.center
                        print(f"Camera {cam.camera_index}: {away:.1f} cm at ({cx}, {cy})")
            if now - last_print > 1:
                last_print = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                quit_requested = True
            if quit_requested:
                break
            if snapshot is not None:
                break
    finally:
        for cam in cameras:
            cam.close()
        # Single-process dual only — independent dual uses separate processes / close().
        cv2.destroyAllWindows()
    if snapshot is None and recorder.session_files:
        review_pending(recorder.session_files)


def _worker_argv_from_args(
    args: argparse.Namespace,
    camera: int,
    peer_camera: int,
) -> list[str]:
    """Build argv for one independent dual worker process."""
    cmd = [
        sys.executable,
        str(ROOT / "depthcamera.py"),
        "--independent-worker",
        "--camera",
        str(camera),
        "--peer-camera",
        str(peer_camera),
        "--ipc-base-port",
        str(args.ipc_base_port),
        "--corroborate-abs-cm",
        str(args.corroborate_abs_cm),
        "--corroborate-rel",
        str(args.corroborate_rel),
        "--conf",
        str(args.conf),
        "--weak-camera",
        str(args.weak_camera),
    ]
    if args.model is not None:
        cmd.extend(["--model", str(args.model)])
    if args.cam0_conf is not None:
        cmd.extend(["--cam0-conf", str(args.cam0_conf)])
    if args.cam1_conf is not None:
        cmd.extend(["--cam1-conf", str(args.cam1_conf)])
    if args.snapshot is not None and camera == args.camera:
        cmd.extend(["--snapshot", str(args.snapshot)])
    if not getattr(args, "use_edge_assist", DEFAULT_USE_EDGE_ASSIST):
        cmd.append("--no-edge-assist")
    if getattr(args, "use_orb_assist", DEFAULT_USE_ORB_ASSIST):
        cmd.append("--orb-assist")
    if getattr(args, "orb_assist_weight", DEFAULT_ORB_ASSIST_WEIGHT) != DEFAULT_ORB_ASSIST_WEIGHT:
        cmd.extend(["--orb-assist-weight", str(args.orb_assist_weight)])
    if not getattr(args, "allow_sideways", DEFAULT_ALLOW_SIDEWAYS):
        cmd.append("--no-sideways")
    if not getattr(args, "sideways_assist", DEFAULT_SIDEWAYS_ASSIST):
        cmd.append("--no-sideways-assist")
    if getattr(args, "reproject_assist_3d", False):
        cmd.append("--3d-reproject-assist")
    # Workers inherit the same hard gate as main().
    if getattr(args, "enable_vial_detect", False):
        cmd.append("--enable-vial-detect")
    return cmd


def launch_dual_independent(args: argparse.Namespace) -> int:
    """Spawn two standalone camera processes; IPC corroborates cm (+ opt-in reproject)."""
    if not getattr(args, "enable_vial_detect", False):
        print(RACK_TOOL_HINT)
        return 2
    found = list_orbbec_indexes()
    if len(found) < 2:
        print(f"Need two Orbbecs for --dual-independent; detected: {found if found else 'none'}")
        print(USB_TIP)
        print_physical_dual_checklist(force=True)
        return 1
    cam0, cam1 = found[0], found[1]
    want_reproject = bool(getattr(args, "reproject_assist_3d", False))
    want_score = bool(getattr(args, "use_3d_model_score", False))
    print(
        f"Independent dual: launching camera {cam0} and camera {cam1} as separate processes "
        f"(IPC base port {args.ipc_base_port}). Each finds the vial alone; peer corroborates cm"
        f"{'; + opt-in 3D reproject' if want_reproject else ''}."
    )
    parent_gates = build_camera_gate_map(
        weak_camera=args.weak_camera,
        base_conf=args.conf,
        cam0_conf=args.cam0_conf,
        cam1_conf=args.cam1_conf,
    )
    print_camera_gate_assignment(parent_gates)
    print_physical_dual_checklist()
    _note_live_3d_status(want_reproject=want_reproject, want_score=want_score)
    print("Classic coupled dual remains: python depthcamera.py --dual")
    workers = [
        subprocess.Popen(_worker_argv_from_args(args, cam0, cam1)),
        subprocess.Popen(_worker_argv_from_args(args, cam1, cam0)),
    ]
    exit_code = 0
    try:
        while True:
            alive = [p for p in workers if p.poll() is None]
            if not alive:
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("Stopping independent dual workers...")
    finally:
        for proc in workers:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 5.0
        for proc in workers:
            remaining = max(0.05, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
            if proc.returncode not in (0, None):
                exit_code = max(exit_code, int(proc.returncode))
    return exit_code


def run_independent_worker(args: argparse.Namespace) -> None:
    """One process / one camera: standalone locate + optional peer distance IPC."""
    camera = int(args.camera)
    peer_camera = args.peer_camera
    if peer_camera is None:
        found = list_orbbec_indexes()
        others = [i for i in found if i != camera]
        peer_camera = others[0] if others else (1 if camera == 0 else 0)
    peer_camera = int(peer_camera)
    if peer_camera == camera:
        print("--peer-camera must differ from --camera.")
        raise SystemExit(2)

    gate_map = build_camera_gate_map(
        weak_camera=args.weak_camera,
        base_conf=args.conf,
        cam0_conf=args.cam0_conf,
        cam1_conf=args.cam1_conf,
    )
    print_camera_gate_assignment(gate_map)
    gates = gate_profile_for(camera, gate_map)
    print(f"Worker cam{camera}: profile={gates.name}, yolo_conf={gates.yolo_conf:.3f}")
    detector = build_shared_detector(args.model, gates.yolo_conf)
    cam = OrbbecDepthCamera(
        args.model,
        camera,
        gates.yolo_conf,
        detector=detector,
        window=window_title(camera, dual=True),
        gates=gates,
        use_edge_assist=bool(getattr(args, "use_edge_assist", DEFAULT_USE_EDGE_ASSIST)),
        use_orb_assist=bool(getattr(args, "use_orb_assist", DEFAULT_USE_ORB_ASSIST)),
        orb_assist_weight=float(
            getattr(args, "orb_assist_weight", DEFAULT_ORB_ASSIST_WEIGHT)
        ),
    )
    ipc = PeerIpc(camera, peer_camera, base_port=int(args.ipc_base_port))
    try:
        cam.run(
            args.snapshot,
            peer_ipc=ipc,
            corroborate_abs_cm=float(args.corroborate_abs_cm),
            corroborate_rel=float(args.corroborate_rel),
            reproject_assist=bool(getattr(args, "reproject_assist_3d", False)),
        )
    except RuntimeError as error:
        print(error)
        print(USB_TIP)
        raise SystemExit(1) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Orbbec helpers. Free-form vial detect is DISABLED by default — "
            "use rack_lid_state.py / rack_capture_agents.py. "
            "Legacy vial trackers require --enable-vial-detect."
        )
    )
    parser.add_argument(
        "--enable-vial-detect",
        action="store_true",
        default=VIAL_DETECT_ENABLED_DEFAULT,
        help=(
            "OPT-IN: allow YOLO / free-form vial locate. Default is OFF; "
            "use rack_lid_state.py focus or rack_capture_agents.py instead."
        ),
    )
    parser.add_argument("--camera", type=int, default=0, help="Primary Orbbec camera index (usually 0)")
    parser.add_argument(
        "--camera2",
        type=int,
        nargs="?",
        const=-1,
        default=None,
        help="Second Orbbec index (usually 1). Use --camera2 or --camera2 1. Not --camera 2.",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Classic one-process dual with locate_pair sticky peer soft assist",
    )
    parser.add_argument(
        "--dual-independent",
        action="store_true",
        help="Preferred: spawn two processes; each detects alone; IPC corroborates distances only",
    )
    parser.add_argument(
        "--independent",
        action="store_true",
        help="With --dual, same as --dual-independent. Alone: single-cam worker with peer IPC",
    )
    parser.add_argument(
        "--independent-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--peer-camera",
        type=int,
        default=None,
        help="Peer camera index for independent IPC corroboration",
    )
    parser.add_argument(
        "--ipc-base-port",
        type=int,
        default=IPC_BASE_PORT,
        help=f"UDP base port for independent dual IPC (default {IPC_BASE_PORT})",
    )
    parser.add_argument(
        "--corroborate-abs-cm",
        type=float,
        default=CORROBORATE_ABS_CM,
        help=f"Agree when |Δcm| ≤ this (default {CORROBORATE_ABS_CM})",
    )
    parser.add_argument(
        "--corroborate-rel",
        type=float,
        default=CORROBORATE_REL,
        help=f"Agree when |Δ|/max ≤ this (default {CORROBORATE_REL})",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print which Orbbec indexes OpenCV can open, then exit",
    )
    parser.add_argument("--model", type=Path, default=None, help="Trained YOLO weights")
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_YOLO_CONF,
        help=f"Base YOLO confidence for strong cam (default {DEFAULT_YOLO_CONF})",
    )
    parser.add_argument(
        "--weak-camera",
        type=int,
        default=0,
        help=(
            "Camera index with permissive gates / lower YOLO conf (default: 0). "
            "Swap to 1 if the hard physical view is cam1."
        ),
    )
    parser.add_argument("--cam0-conf", type=float, default=None, help="Override YOLO conf for camera 0")
    parser.add_argument("--cam1-conf", type=float, default=None, help="Override YOLO conf for camera 1")
    parser.add_argument("--snapshot", type=Path, default=None, help="Save one frame and exit")
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Skip the cameras and review pending stills under vial_dataset/review/pending/",
    )
    parser.add_argument(
        "--3d-reproject-assist",
        dest="reproject_assist_3d",
        action="store_true",
        help=(
            "Opt-in: when peer has a vial and stereo_extrinsics.json is real (not "
            "PLACEHOLDER), reproject parametric 3D into the misser; local gates must "
            "still accept. Size-score stays off. Prefer with --dual-independent."
        ),
    )
    parser.add_argument(
        "--no-3d-model-score",
        dest="use_3d_model_score",
        action="store_false",
        help="Keep live 3D size-score off (default; score path remains disabled).",
    )
    parser.add_argument(
        "--3d-model-score-min",
        dest="model_score_min",
        type=float,
        default=DEFAULT_3D_MODEL_SCORE_MIN,
        help="Live size-score threshold (unused while size-score stays disabled).",
    )
    parser.add_argument(
        "--no-edge-assist",
        dest="use_edge_assist",
        action="store_false",
        help=(
            "Disable Gx/Gy (Sobel) vial-structure edge assist after YOLO/cap miss "
            "(default on; also skips dx/dy re-rank of close candidates)."
        ),
    )
    parser.add_argument(
        "--orb-assist",
        dest="use_orb_assist",
        action="store_true",
        help=(
            "Enable tiny-weight ORB white-cap match assist (default off). "
            "Never invents boxes; never overrides an upright full-vial pool; "
            "soft dual agree / choose_box boost among non-upright only."
        ),
    )
    parser.add_argument(
        "--no-orb-assist",
        dest="use_orb_assist",
        action="store_false",
        help="Disable ORB assist (default; kept for compatibility).",
    )
    parser.add_argument(
        "--orb-assist-weight",
        type=float,
        default=DEFAULT_ORB_ASSIST_WEIGHT,
        help=(
            f"ORB match weight in choose_box ranking (default {DEFAULT_ORB_ASSIST_WEIGHT}; "
            "only applied when no upright candidate is in the pool)."
        ),
    )
    parser.add_argument(
        "--allow-sideways",
        dest="allow_sideways",
        action="store_true",
        help="Allow clearly horizontal sideways/fallen vials (default on).",
    )
    parser.add_argument(
        "--no-sideways",
        dest="allow_sideways",
        action="store_false",
        help="Disable live sideways/fallen pose path (upright + top-down only).",
    )
    parser.add_argument(
        "--sideways-assist",
        dest="sideways_assist",
        action="store_true",
        help=(
            "Grow cup horizontally from white end-cap + expand wide YOLO boxes "
            "(default on; requires sideways allowed)."
        ),
    )
    parser.add_argument(
        "--no-sideways-assist",
        dest="sideways_assist",
        action="store_false",
        help="Disable horizontal sideways growth / wide-YOLO expand (aspect band only).",
    )
    parser.set_defaults(
        use_3d_model_score=False,
        use_edge_assist=DEFAULT_USE_EDGE_ASSIST,
        use_orb_assist=DEFAULT_USE_ORB_ASSIST,
        allow_sideways=DEFAULT_ALLOW_SIDEWAYS,
        sideways_assist=DEFAULT_SIDEWAYS_ASSIST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_allow_sideways(bool(getattr(args, "allow_sideways", DEFAULT_ALLOW_SIDEWAYS)))
    set_sideways_assist(bool(getattr(args, "sideways_assist", DEFAULT_SIDEWAYS_ASSIST)))
    if args.list_cameras:
        found = print_orbbec_cameras()
        raise SystemExit(0 if found else 1)
    if args.review_only:
        review_pending()
        return

    # Hard off: no YOLO / locate / vial boxes unless explicitly opted in.
    if not bool(getattr(args, "enable_vial_detect", False)):
        print(RACK_TOOL_HINT)
        raise SystemExit(2)

    want_independent_dual = bool(args.dual_independent or (args.dual and args.independent))
    if want_independent_dual and not args.independent_worker:
        raise SystemExit(launch_dual_independent(args))

    if args.independent_worker or (args.independent and args.camera2 is None and not args.dual):
        run_independent_worker(args)
        return

    camera = args.camera
    camera2 = args.camera2
    if args.dual:
        found = list_orbbec_indexes()
        if len(found) < 2:
            print(f"Need two Orbbecs for --dual; detected: {found if found else 'none'}")
            print(USB_TIP)
            raise SystemExit(1)
        camera, camera2 = found[0], found[1]
        print(f"--dual selected camera {camera} and camera2 {camera2}")
        print("Note: for independent dual (recommended), use --dual-independent")
    elif camera2 == -1:
        found = list_orbbec_indexes()
        others = [i for i in found if i != camera]
        if not others:
            print(f"--camera2 needs a second Orbbec. Detected: {found if found else 'none'}")
            print(USB_TIP)
            raise SystemExit(1)
        camera2 = others[0]
        print(f"--camera2 auto-selected index {camera2}")

    if camera2 is not None and camera2 == camera:
        print("--camera2 must be a different index than --camera.")
        raise SystemExit(2)

    gate_map = build_camera_gate_map(
        weak_camera=args.weak_camera,
        base_conf=args.conf,
        cam0_conf=args.cam0_conf,
        cam1_conf=args.cam1_conf,
    )
    print_camera_gate_assignment(gate_map)
    # Shared YOLO weights; per-cam conf applied inside locate via CameraGateProfile.
    shared_conf = min(g.yolo_conf for g in gate_map.values())
    detector = build_shared_detector(args.model, shared_conf)
    use_edge = bool(getattr(args, "use_edge_assist", DEFAULT_USE_EDGE_ASSIST))
    use_orb = bool(getattr(args, "use_orb_assist", DEFAULT_USE_ORB_ASSIST))
    orb_w = float(getattr(args, "orb_assist_weight", DEFAULT_ORB_ASSIST_WEIGHT))
    want_reproject = bool(getattr(args, "reproject_assist_3d", False))
    want_score = bool(getattr(args, "use_3d_model_score", False))
    if want_reproject or want_score:
        _note_live_3d_status(want_reproject=want_reproject, want_score=want_score)
    primary = OrbbecDepthCamera(
        args.model,
        camera,
        gate_map.get(camera, strong_gate_profile(yolo_conf=args.conf)).yolo_conf,
        detector=detector,
        window=window_title(camera, dual=camera2 is not None),
        gates=gate_profile_for(camera, gate_map),
        use_edge_assist=use_edge,
        use_orb_assist=use_orb,
        orb_assist_weight=orb_w,
    )

    if camera2 is None:
        try:
            primary.run(
                args.snapshot,
                reproject_assist=want_reproject,
            )
        except RuntimeError as error:
            print(error)
            print(USB_TIP)
            raise SystemExit(1) from error
        return

    secondary = OrbbecDepthCamera(
        args.model,
        camera2,
        gate_map.get(camera2, strong_gate_profile(yolo_conf=args.conf)).yolo_conf,
        detector=detector,
        window=window_title(camera2, dual=True),
        gates=gate_profile_for(camera2, gate_map),
        use_edge_assist=use_edge,
        use_orb_assist=use_orb,
        orb_assist_weight=orb_w,
    )
    try:
        cameras = open_cameras(primary, secondary)
        if len(cameras) < 2:
            print(
                "Only one camera is running. Fix USB / indexes, then use "
                "--dual-independent or --dual / --camera 0 --camera2 1."
            )
            print_physical_dual_checklist(force=True)
        run_cameras(cameras, args.snapshot, reproject_assist=want_reproject)
    except RuntimeError as error:
        print(error)
        print(USB_TIP)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
