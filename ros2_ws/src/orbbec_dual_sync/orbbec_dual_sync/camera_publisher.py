"""Publish one Orbbec's color (+ depth) as stamped sensor_msgs/Image.

Color and depth from the same grab share an identical header stamp so
intra-camera association is exact; cross-camera sync is ApproximateTime.

Publishers open CAP_OBSENSOR only — YOLO loads in sync_detect, not here.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import threading
import time
import traceback

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image

from .approx_time import seconds_to_stamp_parts
from .camera_path import ensure_camera_on_path
from .image_codec import bgr8_to_imgmsg, depth_mm_to_imgmsg
from .preview_gui import PreviewWindow

# Dual USB Orbbecs need longer warm-up than a single desktop open. Keep the
# mutex for the complete retry sequence: a peer must not initialize OBSENSOR
# until this camera either has color+depth or has cleanly released its handle.
_PUBLISH_OPEN_RETRIES = 3
_PUBLISH_OPEN_CYCLES = 4
_PUBLISH_WARM_TRIES = 120
_USB_OPEN_LOCK_TIMEOUT_S = 300.0
_REOPEN_AFTER_MISSES = 20


@contextmanager
def _exclusive_usb_open(logger, stop_event: threading.Event):
    """Serialize CAP_OBSENSOR initialization across publisher processes.

    A Windows kernel mutex is abandoned automatically on *any* process exit,
    including a native OpenCV abort that never runs Python cleanup. A waiter
    receives WAIT_ABANDONED and immediately takes ownership. There is no lock
    file on disk, so Ctrl+C/relaunch cannot leave a sticky lock behind.
    """
    if os.name != "nt":
        yield
        return

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_mutex = kernel32.CreateMutexW
    create_mutex.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    create_mutex.restype = wintypes.HANDLE
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait_for_single_object.restype = wintypes.DWORD
    release_mutex = kernel32.ReleaseMutex
    release_mutex.argtypes = (wintypes.HANDLE,)
    release_mutex.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    wait_object_0 = 0x00000000
    wait_abandoned = 0x00000080
    wait_timeout = 0x00000102
    wait_failed = 0xFFFFFFFF
    mutex = create_mutex(
        None, False, "Local\\JessicaOrbbecCapObsensorUsbOpen"
    )
    if not mutex:
        raise OSError(
            ctypes.get_last_error(), "CreateMutexW failed for Orbbec USB lock"
        )

    acquired = False
    try:
        deadline = time.monotonic() + _USB_OPEN_LOCK_TIMEOUT_S
        wait_started = time.monotonic()
        next_progress_log = wait_started
        while True:
            if stop_event.is_set() or not rclpy.ok():
                raise InterruptedError("Shutdown requested while waiting for USB open lock")
            result = wait_for_single_object(mutex, 250)
            if result == wait_object_0:
                acquired = True
                break
            if result == wait_abandoned:
                acquired = True
                logger.warn(
                    "Previous Orbbec publisher died while initializing "
                    "CAP_OBSENSOR; reclaimed its abandoned USB lock and retrying"
                )
                break
            if result == wait_timeout:
                now = time.monotonic()
                if now >= next_progress_log:
                    logger.info(
                        "Waiting for the other Orbbec publisher to finish "
                        f"CAP_OBSENSOR initialization ({now - wait_started:.0f}s "
                        f"elapsed; timeout {_USB_OPEN_LOCK_TIMEOUT_S:.0f}s). "
                        "The lock is released automatically when its process exits."
                    )
                    next_progress_log = now + 15.0
                if now >= deadline:
                    raise TimeoutError(
                        f"Timed out after {_USB_OPEN_LOCK_TIMEOUT_S:.0f}s waiting "
                        "for the other Orbbec publisher to finish CAP_OBSENSOR "
                        "initialization. A publisher process is still alive or "
                        "the USB driver is wedged; stop leftover publisher "
                        "processes, then unplug/replug both cameras."
                    )
                continue
            if result == wait_failed:
                raise OSError(
                    ctypes.get_last_error(),
                    "WaitForSingleObject failed for Orbbec USB lock",
                )
            raise RuntimeError(f"Unexpected USB mutex wait result: 0x{result:08x}")
        try:
            yield
        finally:
            if acquired and not release_mutex(mutex):
                logger.warn(
                    "ReleaseMutex failed for Orbbec USB lock "
                    f"(Windows error {ctypes.get_last_error()})"
                )
    finally:
        close_handle(mutex)


class OrbbecCameraPublisher(Node):
    def __init__(self) -> None:
        super().__init__("orbbec_camera_publisher")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("frame_id", "orbbec_cam0")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("publish_depth", True)
        self.declare_parameter("publish_camera_info", True)
        self.declare_parameter("show_preview", False)
        self.declare_parameter("color_topic", "")
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("info_topic", "")
        # Extra wait before VideoCapture (cam1 YAML / launch stagger).
        self.declare_parameter("open_delay_s", 0.0)

        self._stop_event = threading.Event()
        self._cam = None
        self.context.on_shutdown(self._request_shutdown)
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        # Optional K if CAP_OBSENSOR reports no intrinsics. Left unset.
        ncam = self.camera_index
        for _axis in ("fx", "fy", "cx", "cy"):
            self.declare_parameter(f"cam{ncam}_{_axis}")
        self._intrinsics_warn_t = 0.0
        rate = float(self.get_parameter("publish_rate_hz").value)
        self.publish_depth = bool(self.get_parameter("publish_depth").value)
        self.publish_camera_info = bool(self.get_parameter("publish_camera_info").value)
        show_preview = bool(self.get_parameter("show_preview").value)
        open_delay_s = float(self.get_parameter("open_delay_s").value)

        color_topic = str(self.get_parameter("color_topic").value).strip()
        depth_topic = str(self.get_parameter("depth_topic").value).strip()
        info_topic = str(self.get_parameter("info_topic").value).strip()
        if not color_topic:
            color_topic = f"/orbbec/cam{self.camera_index}/color/image_raw"
        if not depth_topic:
            depth_topic = f"/orbbec/cam{self.camera_index}/depth/image_raw"
        if not info_topic:
            info_topic = f"/orbbec/cam{self.camera_index}/color/camera_info"

        # Large RGB-D frames must not accumulate in reliable DDS queues. A
        # latest-sensor-frame profile avoids native middleware exhaustion when
        # detection is slower than capture.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.color_pub = self.create_publisher(Image, color_topic, image_qos)
        self.depth_pub = (
            self.create_publisher(Image, depth_topic, image_qos)
            if self.publish_depth
            else None
        )
        self.info_pub = (
            self.create_publisher(CameraInfo, info_topic, image_qos)
            if self.publish_camera_info
            else None
        )

        ensure_camera_on_path()
        import depthcamera as dc

        self._dc = dc
        # No YOLO in publishers — VialDetector(None) skips ultralytics load.
        stream_only = dc.VialDetector(None, dc.DEFAULT_YOLO_CONF)
        self._cam = dc.OrbbecDepthCamera(
            camera_index=self.camera_index,
            detector=stream_only,
            window=f"ros_pub_cam{self.camera_index}",
        )
        if open_delay_s > 0.0:
            self.get_logger().info(
                f"Waiting {open_delay_s:.1f}s before opening camera_index="
                f"{self.camera_index} (USB stagger)"
            )
            if self._stop_event.wait(open_delay_s):
                raise SystemExit(0)
        try:
            self._open_camera_with_backoff("initial open")
        except InterruptedError:
            self._release_capture()
            self._cam = None
            raise SystemExit(0)
        except Exception as exc:
            # OpenCV CAP_OBSENSOR prints "Camera index out of range" when USB
            # cams are missing; surface a clear ROS error before exiting.
            self._release_capture()
            self._cam = None
            tip = getattr(dc, "USB_TIP", "")
            msg = (
                f"Failed to open Orbbec camera_index={self.camera_index} "
                f"(OpenCV CAP_OBSENSOR). Close Orbbec Viewer / other apps, "
                f"use separate USB3 ports, unplug/replug if needed. "
                f"Detail: {exc}"
            )
            if tip:
                msg = f"{msg} {tip}"
            self.get_logger().error(msg)
            print(f"[orbbec_camera_publisher] ERROR: {msg}", flush=True)
            raise SystemExit(1) from exc
        self._intrinsics = None
        self._intrinsics_initialized = False
        self._read_misses = 0
        self._recovering = False
        self._preview = (
            PreviewWindow(f"orbbec_raw_cam{self.camera_index}")
            if show_preview
            else None
        )
        self.get_logger().info(
            f"Publishing Orbbec index {self.camera_index} "
            f"color={color_topic} depth={depth_topic} @ {rate:.1f} Hz "
            f"preview={'on' if self._preview else 'off'} (no YOLO in publisher)"
        )
        period = 1.0 / max(rate, 1.0)
        self._timer = self.create_timer(period, self._tick)
        self._frames = 0

    def _stamp_now(self):
        # Prefer wall time at grab for cross-process USB cams (clocks shared).
        now_s = time.time()
        sec, nanosec = seconds_to_stamp_parts(now_s)
        stamp = self.get_clock().now().to_msg()
        stamp.sec = sec
        stamp.nanosec = nanosec
        return stamp

    def _tick(self) -> None:
        try:
            color, depth_mm = self._cam.read_images()
            if color is None or depth_mm is None:
                self._read_misses += 1
                if self._read_misses >= _REOPEN_AFTER_MISSES:
                    self._recover_camera()
                return
            self._read_misses = 0
            stamp = self._stamp_now()
            self.color_pub.publish(
                bgr8_to_imgmsg(color, frame_id=self.frame_id, stamp=stamp)
            )
            if self.depth_pub is not None:
                self.depth_pub.publish(
                    depth_mm_to_imgmsg(depth_mm, frame_id=self.frame_id, stamp=stamp)
                )
            if self.info_pub is not None:
                self.info_pub.publish(self._camera_info(color, stamp))
            if self._preview is not None and self._preview.enabled:
                ok = self._preview.show(color)
                if not ok:
                    if self._preview.user_closed:
                        self.get_logger().info(
                            f"cam{self.camera_index} preview closed; "
                            "publisher still running."
                        )
                    else:
                        self.get_logger().error(
                            f"cam{self.camera_index} preview unavailable; "
                            f"publisher still running. Detail: "
                            f"{self._preview.last_error or 'unknown UI error'}"
                        )
                    self._preview = None
            self._frames += 1
            if self._frames % 60 == 0:
                self.get_logger().info(
                    f"cam{self.camera_index}: published {self._frames} frames"
                )
        except Exception as exc:
            self.get_logger().error(
                f"publish tick failed (node stays up): {exc}\n{traceback.format_exc()}"
            )
            self._read_misses += 1

    def _release_capture(self) -> None:
        """Release OBSENSOR without calling the unsafe HighGUI cleanup path."""
        cam = getattr(self, "_cam", None)
        capture = None if cam is None else cam.capture
        if cam is not None:
            cam.capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception as exc:
                self.get_logger().warn(f"Camera release failed: {exc}")

    def _request_shutdown(self) -> None:
        """Cancel waits and release native USB promptly on SIGINT/shutdown."""
        self._stop_event.set()
        self._release_capture()

    def _open_camera_with_backoff(self, reason: str) -> None:
        """Open while holding the cross-process mutex through all retries."""
        with _exclusive_usb_open(self.get_logger(), self._stop_event):
            last_error = None
            for cycle in range(1, _PUBLISH_OPEN_CYCLES + 1):
                if self._stop_event.is_set() or not rclpy.ok():
                    raise InterruptedError("Shutdown requested during camera open")
                try:
                    self.get_logger().info(
                        f"cam{self.camera_index}: {reason}, OBSENSOR attempt "
                        f"{cycle}/{_PUBLISH_OPEN_CYCLES}"
                    )
                    self._cam.open(
                        require_frame=True,
                        retries=_PUBLISH_OPEN_RETRIES,
                        warm_tries=_PUBLISH_WARM_TRIES,
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    self._release_capture()
                    if cycle == _PUBLISH_OPEN_CYCLES:
                        break
                    delay = min(2.0 * (2 ** (cycle - 1)), 8.0)
                    self.get_logger().warn(
                        f"cam{self.camera_index}: OBSENSOR attempt {cycle} "
                        f"failed ({exc}); releasing USB and retrying in "
                        f"{delay:.1f}s"
                    )
                    if self._stop_event.wait(delay):
                        raise InterruptedError(
                            "Shutdown requested during camera-open backoff"
                        )
            raise RuntimeError(
                f"cam{self.camera_index}: all {_PUBLISH_OPEN_CYCLES} OBSENSOR "
                f"open cycles failed. Last error: {last_error}"
            )

    def _recover_camera(self) -> None:
        if self._recovering:
            return
        self._recovering = True
        self.get_logger().error(
            f"cam{self.camera_index}: {_REOPEN_AFTER_MISSES} consecutive read "
            "failures; reopening OBSENSOR without stopping the ROS node"
        )
        try:
            self._release_capture()
            if self._stop_event.wait(2.0):
                return
            self._open_camera_with_backoff("stream recovery")
            self._intrinsics = None
            self._intrinsics_initialized = False
            self.get_logger().info(f"cam{self.camera_index}: stream recovered")
        except Exception as exc:
            self.get_logger().error(
                f"cam{self.camera_index}: reopen failed; will retry: {exc}"
            )
            self._release_capture()
        finally:
            self._read_misses = 0
            self._recovering = False

    def _log_intrinsics_none(self, text: str) -> None:
        now = time.monotonic()
        if now - float(self._intrinsics_warn_t) < 10.0:
            return
        self._intrinsics_warn_t = now
        self.get_logger().warn(text)

    def _param_intrinsics(self):
        """camN_fx/fy/cx/cy when set. None if any of the four is unset."""
        ncam = int(self.camera_index)
        vals = []
        for axis in ("fx", "fy", "cx", "cy"):
            param = self.get_parameter(f"cam{ncam}_{axis}")
            if param.type_ == Parameter.Type.NOT_SET or param.value is None:
                return None
            try:
                vals.append(float(param.value))
            except (TypeError, ValueError):
                return None
        if vals[0] <= 0.0 or vals[1] <= 0.0:
            return None
        return tuple(vals)

    def _camera_info(self, color: np.ndarray, stamp) -> CameraInfo:
        h, w = color.shape[:2]
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.height = int(h)
        info.width = int(w)
        if self._intrinsics is None and self._cam is not None and self._cam.capture is not None:
            got = None
            try:
                got = self._dc.intrinsics_from_capture(self._cam.capture, w, h)
            except Exception as exc:
                got = None
                self._log_intrinsics_none(f"Camera intrinsics unavailable: {exc}")
            else:
                if got is None:
                    self._log_intrinsics_none(
                        f"cam{self.camera_index}: intrinsics_from_capture returned None"
                    )
                else:
                    self._intrinsics = got
                    self._intrinsics_initialized = True
        fx = fy = cx = cy = None
        if self._intrinsics is not None:
            fx, fy, cx, cy = self._intrinsics
        else:
            fallback = self._param_intrinsics()
            if fallback is not None:
                fx, fy, cx, cy = fallback
        if fx is not None:
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return info

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._preview is not None:
            try:
                self._preview.close()
            except Exception:
                pass
            self._preview = None
        if self._cam is not None:
            self._release_capture()
            self._cam = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = OrbbecCameraPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        raise
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
