"""Pure helpers for cam0 furnace overlay (no rclpy).

Imports match/draw from robotics/furnace_agents.py via camera_path.
Fixed slots from furnace_state/furnace.json stay drawn when occluded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def default_furnace_json(robotics_root: Path) -> Path:
    return robotics_root / "furnace_state" / "furnace.json"



def _ensure_robotics(robotics_root: Optional[Path]) -> Path:
    if robotics_root is None:
        from .camera_path import ensure_camera_on_path  # noqa: PLC0415

        return ensure_camera_on_path()
    root_s = str(robotics_root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return robotics_root


def load_furnace_agents(robotics_root: Optional[Path] = None) -> Any:
    """Import furnace_agents with robotics on sys.path."""
    root = _ensure_robotics(robotics_root)
    import furnace_agents as fa  # noqa: PLC0415

    return fa


def _slot_occluded_by_arm(
    bgr: np.ndarray,
    slot: Any,
    *,
    blue_frac: float = 0.10,
    grey_frac: float = 0.35,
) -> bool:
    """True if baby-blue joint plastic or thick grey arm metal covers this slot."""
    x, y, w, h = int(slot.x), int(slot.y), int(slot.w), int(slot.h)
    hs, ws = bgr.shape[:2]
    if w < 4 or h < 4 or x >= ws or y >= hs:
        return False
    x2, y2 = min(ws, x + w), min(hs, y + h)
    roi = bgr[y:y2, x:x2]
    if roi.size == 0:
        return False
    area = float(roi.shape[0] * roi.shape[1])
    try:
        from .baby_blue_draw import baby_blue_mask  # noqa: PLC0415

        blue = baby_blue_mask(roi)
        if float(cv2.countNonZero(blue)) / area >= float(blue_frac):
            return True
    except Exception:
        pass
    # Grey UR links (low saturation metal) covering the door ROI
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    grey = cv2.inRange(hsv, (0, 0, 40), (179, 55, 210))
    if float(cv2.countNonZero(grey)) / area >= float(grey_frac):
        # Prefer grey+blue combo so empty closed doors (beige) don't count
        try:
            from .baby_blue_draw import baby_blue_mask  # noqa: PLC0415

            blue = baby_blue_mask(roi)
            if float(cv2.countNonZero(blue)) / area >= 0.03:
                return True
        except Exception:
            pass
    return False


def resolve_slots_for_frame(
    bgr: np.ndarray,
    *,
    robotics_root: Optional[Path] = None,
    furnace_json: Optional[Path] = None,
    match_live: bool = True,
) -> list[Any]:
    """Return FurnaceSlot list at *fixed* furnace.json positions only.

    Furnaces never move in the image — we do not freestyle-detect new boxes.
    ``match_live`` only controls whether OPEN/CLOSED is reclassified each frame
    (template match inside each fixed ROI). Slots covered by the arm (baby-blue
    / grey links) are marked occluded so overlays can hide them.
    """
    fa = load_furnace_agents(robotics_root)
    root = _ensure_robotics(robotics_root)
    cfg_path = Path(furnace_json) if furnace_json else default_furnace_json(root)
    cfg = fa.load_furnace_config(cfg_path)
    fixed = fa.slots_from_config(cfg)
    if not fixed:
        return []

    # Always pin geometry to furnace.json; only refresh door labels.
    pinned = [
        fa.FurnaceSlot(
            id=s.id,
            x=s.x,
            y=s.y,
            w=s.w,
            h=s.h,
            score=s.score,
            occluded=_slot_occluded_by_arm(bgr, s),
            label=s.label,
        )
        for s in fixed
    ]
    # Only classify slots that are visible; arm-covered slots keep last label.
    visible = [s for s in pinned if not s.occluded]
    if match_live and visible:
        labeled = fa.classify_slots_door_state(bgr, visible, cfg=cfg)
        by_id = {s.id: s for s in labeled}
        pinned = [
            by_id.get(
                s.id,
                s,
            )
            if not s.occluded
            else fa.FurnaceSlot(
                id=s.id,
                x=s.x,
                y=s.y,
                w=s.w,
                h=s.h,
                score=s.score,
                occluded=True,
                label=s.label,
            )
            for s in pinned
        ]
        # Merge classified visible slots back
        pinned = [
            by_id[s.id]
            if (not s.occluded and s.id in by_id)
            else s
            for s in pinned
        ]
    return pinned


def draw_furnace_overlay_on_frame(
    bgr: np.ndarray,
    slots: list[Any],
    *,
    robotics_root: Optional[Path] = None,
    corner_tag: str = "cam0 + furnace CV",
) -> np.ndarray:
    """Draw furnace boxes/labels and a corner tag for dual_view layout."""
    fa = load_furnace_agents(robotics_root)
    # Hide boxes when the arm covers that furnace (occluded=True).
    view = fa.draw_furnace_overlay(bgr, slots, hide_occluded=True)
    cv2.putText(
        view,
        corner_tag,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return view


def paint_furnace_overlay(
    bgr: np.ndarray,
    *,
    robotics_root: Optional[Path] = None,
    furnace_json: Optional[Path] = None,
    match_live: bool = True,
    corner_tag: str = "cam0 + furnace CV",
) -> np.ndarray:
    """End-to-end: resolve slots then draw overlay on a copy of bgr."""
    slots = resolve_slots_for_frame(
        bgr,
        robotics_root=robotics_root,
        furnace_json=furnace_json,
        match_live=match_live,
    )
    return draw_furnace_overlay_on_frame(
        bgr, slots, robotics_root=robotics_root, corner_tag=corner_tag
    )
