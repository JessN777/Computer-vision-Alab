"""Desktop preview helper for ROS nodes.

RoboStack/conda ``opencv`` on Windows is often built without a usable
HighGUI backend.  In a ``ros2 launch`` child process, probing HighGUI can
terminate the native process instead of raising a catchable Python exception.
Use tkinter + Pillow by default and only try HighGUI when explicitly opted in.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def probe_cv2_gui() -> bool:
    """Return True if cv2.imshow / namedWindow are usable."""
    try:
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.namedWindow("__orbbec_gui_probe__", cv2.WINDOW_NORMAL)
        cv2.imshow("__orbbec_gui_probe__", img)
        cv2.waitKey(1)
        cv2.destroyWindow("__orbbec_gui_probe__")
        return True
    except Exception:
        try:
            cv2.destroyWindow("__orbbec_gui_probe__")
        except Exception:
            pass
        return False


class PreviewWindow:
    """Show BGR frames; Q / window-close disables preview (does not exit process)."""

    def __init__(
        self,
        title: str,
        *,
        max_height: int = 480,
        allow_cv2: Optional[bool] = None,
    ) -> None:
        self.title = title
        self.max_height = int(max_height)
        if allow_cv2 is None:
            allow_cv2 = os.environ.get("ORBBEC_PREVIEW_CV2", "").strip() == "1"
        self.allow_cv2 = bool(allow_cv2)
        self.enabled = True
        self.user_closed = False
        self.reset_requested = False
        self.last_error = ""
        self.backend: Optional[str] = None  # "cv2" | "tk"
        self._tk_root = None
        self._tk_label = None
        self._tk_photo = None
        self._cv2_ok: Optional[bool] = None

    def show(self, bgr: np.ndarray) -> bool:
        """Display one BGR frame. Returns False if preview was disabled."""
        if not self.enabled:
            return False
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
            return True
        frame = self._fit(bgr)
        if self.backend is None:
            self._pick_backend()
        if not self.enabled or self.backend is None:
            return False
        try:
            if self.backend == "cv2":
                return self._show_cv2(frame)
            return self._show_tk(frame)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.enabled = False
            self.backend = None
            return False

    def close(self) -> None:
        self.enabled = False
        if self.backend == "cv2":
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
        root = self._tk_root
        self._tk_root = None
        self._tk_label = None
        self._tk_photo = None
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    def _fit(self, bgr: np.ndarray) -> np.ndarray:
        h = int(bgr.shape[0])
        if h <= self.max_height or self.max_height <= 0:
            return bgr
        scale = self.max_height / float(h)
        w = max(1, int(bgr.shape[1] * scale))
        return cv2.resize(bgr, (w, self.max_height))

    def _pick_backend(self) -> None:
        try:
            import tkinter  # noqa: F401
            from PIL import Image, ImageTk  # noqa: F401
        except Exception as exc:
            # HighGUI probing is unsafe in Windows ros2-launch children. It is
            # available only as an explicit escape hatch for known-good builds.
            if self.allow_cv2:
                if self._cv2_ok is None:
                    self._cv2_ok = probe_cv2_gui()
                if self._cv2_ok:
                    self.backend = "cv2"
                    return
            self.last_error = (
                "tkinter/Pillow preview dependencies unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
            self.enabled = False
            self.backend = None
            return
        self.backend = "tk"

    def _disable(self, reason: str = "") -> None:
        _ = reason
        self.user_closed = True
        self.close()
        self.user_closed = True

    def _show_cv2(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self._disable("key")
            return False
        if key in (ord("r"), ord("R")):
            self.reset_requested = True
        return True

    def _show_tk(self, frame: np.ndarray) -> bool:
        import tkinter as tk
        from PIL import Image, ImageTk

        if self._tk_root is None:
            root = tk.Tk()
            root.title(self.title)
            root.protocol("WM_DELETE_WINDOW", lambda: self._disable("close"))
            for seq in ("<q>", "<Q>", "<Escape>"):
                root.bind(seq, lambda _e: self._disable("key"))
            for seq in ("<r>", "<R>"):
                root.bind(seq, lambda _e: setattr(self, "reset_requested", True))
            label = tk.Label(root)
            label.pack()
            self._tk_root = root
            self._tk_label = label

        if not self.enabled or self._tk_root is None:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=im)
        self._tk_photo = photo  # prevent GC
        self._tk_label.configure(image=photo)
        self._tk_root.update_idletasks()
        self._tk_root.update()
        return self.enabled
