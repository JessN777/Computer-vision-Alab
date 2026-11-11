"""YOLO arm detections via a child process (avoids torch DLL clash with ROS)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_WORKER: Optional["ArmInferWorker"] = None


def default_arm_weights(robotics_root: Path) -> Path:
    cfg = robotics_root / "arm_state" / "arm.json"
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8-sig"))
            w = str(data.get("weights") or "").strip()
            if w and Path(w).is_file():
                return Path(w)
        except Exception:
            pass
    return robotics_root / "arm_state" / "runs" / "arm_detect" / "weights" / "best.pt"


def _prefer_python() -> str:
    """Prefer Anaconda if present (stable torch), else current interpreter."""
    ana = Path(r"C:\Users\Ceder-ALAB\anaconda3\python.exe")
    if ana.is_file():
        return str(ana)
    return sys.executable


def _worker_env(py: str) -> dict[str, str]:
    """Sanitized env for the YOLO child — drop ROS/pixi PYTHONPATH that breaks Anaconda."""
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Anaconda must not import pixi/ROS site-packages (numpy/OpenCV clash).
    if "anaconda" in py.lower() or "conda" in py.lower():
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONNOUSERSITE",
            "CONDA_PREFIX",
            "VIRTUAL_ENV",
        ):
            env.pop(key, None)
        # Ensure conda python uses its own libs only.
        env["PYTHONNOUSERSITE"] = "1"
    return env


_WORKER_SCRIPT = r'''
import json, os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import cv2
from ultralytics import YOLO

weights = sys.argv[1]
conf = float(sys.argv[2])
model = YOLO(weights)
print("READY", flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line or line == "QUIT":
        break
    path, conf_s = line.split("|", 1)
    try:
        conf_f = float(conf_s)
        im = cv2.imread(path)
        if im is None:
            print(json.dumps({"ok": False, "boxes": []}), flush=True)
            continue
        r = model.predict(im, conf=conf_f, imgsz=640, verbose=False)[0]
        boxes = []
        if r.boxes is not None:
            for b in r.boxes:
                xyxy = [float(x) for x in b.xyxy[0].tolist()]
                c = float(b.conf[0]) if b.conf is not None else 0.0
                boxes.append({"xyxy": xyxy, "conf": c})
        print(json.dumps({"ok": True, "boxes": boxes}), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "boxes": []}), flush=True)
'''


class ArmInferWorker:
    def __init__(self, weights: Path, conf: float = 0.25) -> None:
        self.weights = Path(weights)
        self.conf = float(conf)
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._start()

    def _start(self) -> None:
        py = _prefer_python()
        env = _worker_env(py)
        self._proc = subprocess.Popen(
            [py, "-u", "-c", _WORKER_SCRIPT, str(self.weights), str(self.conf)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline().strip()
        if not line.startswith("READY"):
            err = ""
            if self._proc.stderr is not None:
                try:
                    err = self._proc.stderr.read(800)
                except Exception:
                    err = ""
            raise RuntimeError(f"arm worker failed to start: {line!r} {err}")

    def predict(
        self,
        bgr: np.ndarray,
        conf: Optional[float] = None,
        *,
        cam: str = "cam0",
        refine: bool = True,
    ) -> list[tuple[list[float], float]]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                cv2.imwrite(path, bgr)
                self._proc.stdin.write(f"{path}|{float(conf if conf is not None else self.conf)}\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                data = json.loads(line.strip() or "{}")
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            out: list[tuple[list[float], float]] = []
            for b in data.get("boxes") or []:
                xyxy = list(b["xyxy"])
                if refine:
                    xyxy = refine_arm_xyxy(bgr, xyxy, cam=cam)
                out.append((xyxy, float(b["conf"])))
            # If YOLO missed but color prior sees the arm, still draw a box (cam0).
            if not out and refine and cam == "cam0":
                try:
                    import arm_agents as aa  # noqa: PLC0415

                    box = aa._arm_color_box(bgr, pad=48, cam="cam0")
                    if box is not None:
                        x, y, w, h = box
                        out.append(([float(x), float(y), float(x + w), float(y + h)], 0.40))
                except Exception:
                    pass
            return out

    def close(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    if self._proc.stdin:
                        self._proc.stdin.write("QUIT\n")
                        self._proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            self._proc = None


def refine_arm_xyxy(
    bgr: np.ndarray, xyxy: list[float], *, cam: str = "cam0"
) -> list[float]:
    """Expand YOLO arm box using robotics/arm_agents color+grey silhouette."""
    try:
        import arm_agents as aa  # noqa: PLC0415

        return aa.refine_arm_xyxy(bgr, xyxy, cam=cam)
    except Exception:
        return list(xyxy)


def get_worker(weights: Path, conf: float = 0.25) -> ArmInferWorker:
    global _WORKER
    if _WORKER is None or Path(_WORKER.weights) != Path(weights):
        if _WORKER is not None:
            _WORKER.close()
        _WORKER = ArmInferWorker(weights, conf=conf)
    return _WORKER


