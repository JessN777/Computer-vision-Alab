"""Color↔depth mapping and the 3D-3D Kabsch path."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

_PKG = Path(__file__).resolve().parents[1] / "orbbec_dual_sync"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from orbbec_dual_sync.baby_blue_draw import (  # noqa: E402
    color_uv_to_depth_uv,
    median_depth_mm,
    scale_color_box_to_depth,
    scale_depth_box_to_color,
)
from orbbec_dual_sync.online_cam_pose import OnlineCamPose, umeyama_cell  # noqa: E402


def _rot_y(deg: float):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


class TestColorDepthMap(unittest.TestCase):
    def test_scale_round_trips_a_box(self):
        color_hw = (720, 1280)
        depth_hw = (400, 640)
        box = scale_color_box_to_depth(200, 100, 80, 40, color_hw, depth_hw)
        back = scale_depth_box_to_color(*box, color_hw, depth_hw)
        self.assertAlmostEqual(back[0], 200, delta=2)
        self.assertAlmostEqual(back[1], 100, delta=2)
        self.assertAlmostEqual(back[2], 80, delta=3)
        self.assertAlmostEqual(back[3], 40, delta=3)

    def test_same_size_is_identity(self):
        hw = (480, 640)
        u, v = color_uv_to_depth_uv(100.0, 50.0, hw, hw)
        self.assertEqual((u, v), (100.0, 50.0))
        self.assertEqual(scale_color_box_to_depth(10, 20, 30, 40, hw, hw), (10, 20, 30, 40))

    def test_median_maps_color_box_into_smaller_depth(self):
        depth = np.zeros((100, 160), dtype=np.float32)
        # Color 200x320, so scale is 1/2. Color box (40,20,20,20) → depth (20,10,10,10).
        depth[10:20, 20:30] = 800.0
        z = median_depth_mm(depth, 40, 20, 20, 20, color_hw=(200, 320))
        self.assertIsNotNone(z)
        self.assertAlmostEqual(float(z), 800.0, delta=1.0)


class TestUmeyama(unittest.TestCase):
    def test_recovers_known_pose(self):
        R = _rot_y(20.0)
        t = (40.0, -120.0, 800.0)
        src = [
            (0.0, 0.0, 500.0),
            (80.0, -30.0, 620.0),
            (-50.0, 90.0, 710.0),
            (120.0, 40.0, 540.0),
            (-20.0, -70.0, 880.0),
        ]
        dst = []
        for p in src:
            dst.append(
                (
                    R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0],
                    R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1],
                    R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2],
                )
            )
        solved = umeyama_cell(src, dst)
        self.assertIsNotNone(solved)
        Rs, ts, rms = solved
        self.assertLess(rms, 1e-4)
        for i in range(3):
            self.assertAlmostEqual(float(ts[i]), t[i], delta=1e-3)
            for j in range(3):
                self.assertAlmostEqual(float(Rs[i, j]), R[i][j], delta=1e-4)

    def test_online_cam_pose_uses_3d_pairs(self):
        K = np.array([[600.0, 0.0, 330.0], [0.0, 600.0, 250.0], [0.0, 0.0, 1.0]])
        R = np.asarray(_rot_y(15.0), dtype=np.float64)
        t = np.array([90.0, -350.0, 820.0], dtype=np.float64)
        op = OnlineCamPose(K, None, R, t, cam="cam0", update_period_s=0.0, same_pose_mm=1.0)
        fk = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 162.5],
                [320.0, 40.0, 300.0],
                [480.0, 180.0, 440.0],
                [400.0, 260.0, 610.0],
                [340.0, -80.0, 690.0],
                [350.0, -90.0, 750.0],
            ],
            dtype=np.float64,
        )
        # Camera-frame points: invert p_base = R p_cam + t  →  p_cam = R.T (p_base - t)
        Rt = [[float(R[j, i]) for j in range(3)] for i in range(3)]

        def _cam_of(pts):
            out_uv, out_xyz = [], []
            rvec, tvec = op._cv_pose()
            for i in range(6):
                d = [float(pts[i, k]) - float(t[k]) for k in range(3)]
                pcam = [
                    Rt[0][0] * d[0] + Rt[0][1] * d[1] + Rt[0][2] * d[2],
                    Rt[1][0] * d[0] + Rt[1][1] * d[1] + Rt[1][2] * d[2],
                    Rt[2][0] * d[0] + Rt[2][1] * d[1] + Rt[2][2] * d[2],
                ]
                proj, _ = cv2.projectPoints(
                    np.asarray([pts[i]], dtype=np.float32), rvec, tvec, K, None
                )
                out_uv.append((float(proj[0, 0, 0]), float(proj[0, 0, 1])))
                out_xyz.append(pcam)
            return out_uv, out_xyz

        for k in range(4):
            shifted = fk.copy()
            shifted[:, 0] += 25.0 * k
            blue_uv, blue_xyz = _cam_of(shifted)
            tag_uv = list(blue_uv)
            op.add_observations(
                shifted, tag_uv, blue_uv, arm_speed=0.0, blue_xyz=blue_xyz
            )
            op.update()
        self.assertGreaterEqual(op.n_3d, 3)
        self.assertTrue(math.isfinite(op.mean_3d_mm))
        self.assertLess(op.mean_3d_mm, 1.0)
        op.reset()
        self.assertFalse(op.have_pose)
        self.assertEqual(len(op.buffer), 0)


if __name__ == "__main__":
    unittest.main()
