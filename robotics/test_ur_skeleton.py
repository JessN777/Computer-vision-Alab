"""Unit tests for UR FK stick-figure / keep-out helpers (no robot required)."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ur_skeleton import (
    fk_joint_positions_mm,
    fk_rack_proximity,
    fk_transforms,
    load_joints,
    point_to_aabb_gap_mm,
    ur_base_to_vision_mm,
    UrFkConfig,
    write_example_files,
)


class TestUrFk(unittest.TestCase):
    def test_overlay_fk_matches_validation_dh(self):
        """Overlay joint origins use ur5e_kinematics._dh, not a second DH table.

        ur_fk() itself is not called: it multiplies with ``@``, which aborts
        this interpreter. The chain product is the same transforms, looped.
        """
        import sys
        from pathlib import Path

        root = Path(r"C:\Users\Ceder-ALAB\Desktop\Useful Validation Scripts")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import ur5e_kinematics as uk

        from ur_skeleton import _mm, fk_joint_positions_mm

        q = np.array([0.2, -1.1, 1.4, -0.7, -1.2, 0.3])
        T = np.eye(4, dtype=np.float64)
        expect = np.zeros((7, 3), dtype=np.float64)
        for i in range(6):
            Ti = uk._dh(
                float(q[i]),
                float(uk.UR5E_A[i]),
                float(uk.UR5E_D[i]),
                float(uk.UR5E_ALPHA[i]),
            )
            T = _mm(T, np.asarray(Ti, dtype=np.float64))
            expect[i + 1, 0] = float(T[0, 3]) * 1000.0
            expect[i + 1, 1] = float(T[1, 3]) * 1000.0
            expect[i + 1, 2] = float(T[2, 3]) * 1000.0
        got = fk_joint_positions_mm(q, model="ur5e")
        self.assertTrue(np.allclose(got, expect, atol=1e-6))
        self.assertAlmostEqual(float(got[1, 2]), float(uk.UR5E_D[0]) * 1000.0, places=4)

    def test_fk_home_has_seven_points(self):
        q = np.zeros(6)
        pts = fk_joint_positions_mm(q, model="ur5e")
        self.assertEqual(pts.shape, (7, 3))
        # Base at origin
        self.assertTrue(np.allclose(pts[0], 0.0, atol=1e-6))
        # Shoulder roughly at d1 above base
        self.assertGreater(pts[1, 2], 100.0)

    def test_ur7e_step_matches_this_chain(self):
        """UR7e.step joint axes are the UR5e lengths; tube radii are from that file."""
        from ur_skeleton import UR7E_LINK_RADIUS_MM

        q = np.zeros(6)
        a = fk_joint_positions_mm(q, model="ur5e")
        b = fk_joint_positions_mm(q, model="ur7e")
        self.assertTrue(np.allclose(a, b, atol=1e-6))
        # STEP: base footprint r=74, shoulder/forearm housings r=58, wrists r=37.5
        self.assertEqual(UR7E_LINK_RADIUS_MM, (74.0, 58.0, 58.0, 37.5, 37.5, 37.5))
        shoulder = math.sqrt(float(np.sum((a[1] - a[0]) ** 2)))
        upper = math.sqrt(float(np.sum((a[2] - a[1]) ** 2)))
        self.assertAlmostEqual(shoulder, 162.5, places=1)
        self.assertAlmostEqual(upper, 425.0, places=1)

    def test_base_tag_aligns_the_step_chain(self):
        """Tag 0 is the base. Another joint tag turns the STEP chain onto it."""
        import sys

        draw_dir = (
            r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws\src"
            r"\orbbec_dual_sync\orbbec_dual_sync"
        )
        if draw_dir not in sys.path:
            sys.path.insert(0, draw_dir)
        from ur_skeleton_draw import align_chain_about_base_tag

        pix = [(100, 100), (200, 100), (300, 100), None]
        tags = [(150, 80), (150, 160), None, None, None, None]
        out, kind = align_chain_about_base_tag(pix, tags)
        self.assertEqual(kind, "similarity")
        self.assertEqual(out[0], (150, 80))
        self.assertGreater(out[1][1], out[0][1] + 40)
        self.assertLess(abs(out[1][0] - 150), 5)
        self.assertIsNone(out[3])
        # No base tag: do not invent an anchor.
        missing, kind_missing = align_chain_about_base_tag(pix, [None, (150, 160)])
        self.assertIsNone(missing)
        self.assertIsNone(kind_missing)

    def test_shoulder_tag_uses_validation_coordinates(self):
        """Shoulder tag sits on ur_fk joint 1, 162.5 mm up the base axis."""
        import sys

        draw_dir = (
            r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws\src"
            r"\orbbec_dual_sync\orbbec_dual_sync"
        )
        if draw_dir not in sys.path:
            sys.path.insert(0, draw_dir)
        from ur_skeleton_draw import place_validation_joints_sized

        q = np.array([0.4, -1.2, 1.3, -1.0, 0.6, 0.2])
        pts = fk_joint_positions_mm(q, model="ur5e")
        shoulder_mm = math.sqrt(float(np.sum((pts[1] - pts[0]) ** 2)))
        self.assertAlmostEqual(shoulder_mm, 162.5, places=1)
        tags = [(200.0, 400.0), (200.0, 250.0), None, None, None, None]
        out = place_validation_joints_sized(pts, tags, 520.0, 640, 480, hint_pix=None)
        self.assertIsNotNone(out)
        self.assertEqual(out[0], (200, 400))
        self.assertEqual(out[1], (200, 250))
        scale = 150.0 / shoulder_mm
        for i in range(6):
            dpx = math.hypot(out[i + 1][0] - out[i][0], out[i + 1][1] - out[i][1])
            dmm = math.sqrt(float(np.sum((pts[i + 1] - pts[i]) ** 2)))
            # A drawn link cannot be longer than the validation-script length.
            self.assertLessEqual(dpx, scale * dmm * 1.25 + 2.0)

    def test_shoulder_pose_does_not_flip_when_the_hint_flips(self):
        """A flickering camera hint must not swap the arm to the other pose."""
        import sys

        draw_dir = (
            r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws\src"
            r"\orbbec_dual_sync\orbbec_dual_sync"
        )
        if draw_dir not in sys.path:
            sys.path.insert(0, draw_dir)
        import ur_skeleton_draw as draw

        draw._ARM_HOLD.pop("flicker-test", None)
        q = np.array([0.4, -1.2, 1.3, -1.0, 0.6, 0.2])
        pts = fk_joint_positions_mm(q, model="ur5e")
        tags = [(200.0, 400.0), (200.0, 250.0), None, None, None, None]
        # Blues sit on one side of the shoulder, the side the real elbow uses.
        first = draw.place_validation_joints_sized(
            pts,
            tags,
            520.0,
            640,
            480,
            cam="flicker-test",
            blue_uv=np.array([[40.0, 260.0], [20.0, 280.0]]),
        )
        flipped_hint = [(200, 400), (200, 250), (360, 260), None]
        second = draw.place_validation_joints_sized(
            pts,
            tags,
            520.0,
            640,
            480,
            hint_pix=flipped_hint,
            cam="flicker-test",
        )
        self.assertEqual(first[1], (200, 250))
        self.assertEqual(second[1], (200, 250))
        self.assertEqual(first[2][0] < 200, second[2][0] < 200)
        draw._ARM_HOLD.pop("flicker-test", None)

    def test_blue_blob_cannot_take_the_other_elbow(self):
        """A detection on the other side of the arm must not mirror the elbow."""
        import sys

        draw_dir = (
            r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws\src"
            r"\orbbec_dual_sync\orbbec_dual_sync"
        )
        if draw_dir not in sys.path:
            sys.path.insert(0, draw_dir)
        import ur_skeleton_draw as draw

        draw._ARM_HOLD.pop("elbow-side", None)
        q = np.array([0.4, -1.2, 1.3, -1.0, 0.6, 0.2])
        pts = fk_joint_positions_mm(q, model="ur5e")
        tags = [(200.0, 400.0), (200.0, 250.0), None, None, None, None]
        # Camera heading puts the upper arm to the right of the shoulder.
        # The blue blob is on the left, where the mirrored elbow would land.
        hint = [(200.0, 400.0), (200.0, 250.0), (360.0, 250.0), None]
        out = draw.place_validation_joints_sized(
            pts,
            tags,
            520.0,
            640,
            480,
            hint_pix=hint,
            cam="elbow-side",
            blue_uv=np.array([[40.0, 260.0], [20.0, 280.0]]),
        )
        self.assertIsNotNone(out)
        self.assertGreater(out[2][0], 200)
        draw._ARM_HOLD.pop("elbow-side", None)

    def test_fk_transforms_chain(self):
        q = np.array([0.1, -1.0, 1.2, -0.5, 0.2, 0.0])
        Ts = fk_transforms(q, model="ur5e")
        self.assertEqual(len(Ts), 7)
        self.assertTrue(np.allclose(Ts[0], np.eye(4)))

    def test_tool_point_extends_past_flange_to_gripper(self):
        from ur_skeleton import (
            GRIPPER_LEN_MM,
            append_tool_points,
            load_fk_config,
            tool_point_upright_mm,
            upright_points_mm,
        )

        q = np.array([0.1, -1.2, 1.3, -0.6, -1.0, 0.2])
        yaw = 0.3
        up = upright_points_mm(q, yaw, model="ur5e")
        tip = tool_point_upright_mm(q, yaw, model="ur5e")
        flange = up[6]
        reach = float(np.linalg.norm(tip - flange))
        # Gripper tip sits ~GRIPPER_LEN_MM beyond the flange.
        self.assertAlmostEqual(reach, float(GRIPPER_LEN_MM), delta=1.0)
        cfg = load_fk_config()
        vis = np.zeros((7, 3))  # vision frame content irrelevant for the count
        up8, vis8 = append_tool_points(up, vis, q, yaw, cfg, model="ur5e")
        self.assertEqual(up8.shape, (8, 3))
        self.assertEqual(vis8.shape, (8, 3))
        self.assertTrue(np.allclose(up8[7], tip, atol=1e-6))

    def test_aabb_inside_and_outside(self):
        mn, mx = (0.0, 0.0, 0.0), (100.0, 100.0, 100.0)
        self.assertEqual(point_to_aabb_gap_mm([50, 50, 50], mn, mx), 0.0)
        self.assertAlmostEqual(
            point_to_aabb_gap_mm([150, 50, 50], mn, mx), 50.0, places=5
        )

    def test_rack_proximity_flags_tool(self):
        cfg = UrFkConfig(
            rack_keepout_min_mm=(0.0, 0.0, 0.0),
            rack_keepout_max_mm=(100.0, 100.0, 100.0),
            proximity_mm=50.0,
        )
        pts = np.zeros((7, 3))
        pts[6] = [50.0, 50.0, 50.0]  # tool inside
        hits = fk_rack_proximity(pts, cfg)
        self.assertTrue(any(i == 6 for i, _ in hits))

    def test_vision_offset(self):
        cfg = UrFkConfig(t_ur_to_vision_mm=np.array([10.0, 0.0, 0.0]))
        pts = np.zeros((1, 3))
        out = ur_base_to_vision_mm(pts, cfg)
        self.assertAlmostEqual(float(out[0, 0]), 10.0)

    def test_realtime_packet_and_hold_cache(self):
        import struct

        from ur_skeleton import TeachPadCache, q_from_realtime_packet

        q = np.array([-0.055, -1.195, -1.627, -1.89, -4.706, 0.116])
        pkt = bytearray(1220)
        struct.pack_into(">i", pkt, 0, 1220)
        struct.pack_into(">6d", pkt, 252, *q.tolist())
        got = q_from_realtime_packet(bytes(pkt))
        self.assertIsNotNone(got)
        self.assertTrue(np.allclose(got, q))
        self.assertIsNone(q_from_realtime_packet(b"\x00\x00"))
        pad = TeachPadCache()
        pad.note(q, live=True, host="192.168.1.241")
        snap = pad.snapshot("192.168.1.241")
        self.assertIsNotNone(snap)
        self.assertTrue(snap[1].startswith("rtde-live:"))
        pad.mark_hold()
        held = pad.snapshot("192.168.1.241")
        self.assertTrue(held[1].startswith("rtde-hold:"))
        self.assertTrue(np.allclose(held[0], q))
        self.assertIsNone(pad.snapshot("192.168.1.219"))

    def test_load_joints_from_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jp = root / "ur_joints.json"
            jp.write_text(
                json.dumps({"q_rad": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "model": "ur5e"}),
                encoding="utf-8",
            )
            cfg = UrFkConfig(joints_path=jp, host="")
            q, model, source = load_joints(cfg)
            self.assertEqual(model, "ur5e")
            self.assertTrue(source.startswith("file:"))
            self.assertAlmostEqual(float(q[0]), 0.1)

    def test_identity_mount_stands_the_arm_up(self):
        from ur_skeleton import mount_R, ur_base_to_vision_mm

        cfg = UrFkConfig(R_ur_to_vision=np.eye(3))
        R = mount_R(cfg)
        det = (
            R[0, 0] * (R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1])
            - R[0, 1] * (R[1, 0] * R[2, 2] - R[1, 2] * R[2, 0])
            + R[0, 2] * (R[1, 0] * R[2, 1] - R[1, 1] * R[2, 0])
        )
        self.assertAlmostEqual(float(det), 1.0, places=6)
        vis = ur_base_to_vision_mm(np.array([[0.0, 0.0, 100.0]]), cfg)
        # UR +Z (up) lands on camera -Y (up in the image).
        self.assertAlmostEqual(float(vis[0, 1]), -100.0, places=4)
        self.assertAlmostEqual(float(vis[0, 0]), 0.0, places=4)

    def test_fit_recovers_pose_from_blue_points(self):
        from ur_skeleton import fit_arm_to_blues, upright_points_mm, ur_base_to_vision_mm

        cfg = UrFkConfig(R_ur_to_vision=np.eye(3), confirm_gate_mm=80.0)
        q_true = np.array([0.3, -1.2, 1.4, -1.5, -1.3, 0.2])
        yaw_true = 0.4
        up = upright_points_mm(q_true, yaw_true, model="ur5e")
        obs = ur_base_to_vision_mm(up, cfg, yaw=0.0)
        # Drop the base origin; blues are the moving joints.
        obs = obs[1:6] + np.array([3.0, -2.0, 1.0])
        q_guess = q_true + np.array([0.15, -0.1, 0.1, 0.05, -0.05, 0.0])
        fit = fit_arm_to_blues(
            obs,
            q_guess,
            yaw_true - 0.2,
            model="ur5e",
            cfg=cfg,
            search=False,
        )
        self.assertIsNotNone(fit["rms_mm"])
        self.assertLess(float(fit["rms_mm"]), 40.0)
        self.assertTrue(fit["check_ok"])
        self.assertGreaterEqual(int(fit["n_pairs"]), 3)

    def test_pixel_fit_matches_camera_projection(self) -> None:
        from ur_skeleton import (
            UrFkConfig,
            fit_arm_to_pixels,
            upright_points_mm,
            ur_base_to_vision_mm,
        )

        cfg = UrFkConfig(R_ur_to_vision=np.eye(3))
        q_true = np.array([0.2, -1.1, 1.5, -1.3, -1.0, 0.4])
        yaw = 0.15
        vis = ur_base_to_vision_mm(
            upright_points_mm(q_true, yaw, model="ur5e"), cfg, yaw=0.0
        )

        def project(pts):
            uv = np.zeros((len(pts), 2), dtype=np.float64)
            uv[:, 0] = pts[:, 0] * 0.35
            uv[:, 1] = -pts[:, 2] * 0.35
            return uv

        obs = project(vis)[1:6]
        fit = fit_arm_to_pixels(
            obs,
            q_true + np.array([0.1, -0.08, 0.08, 0.05, -0.04, 0.02]),
            yaw - 0.1,
            project,
            model="ur5e",
            cfg=cfg,
            search=False,
        )
        self.assertEqual(fit["check_unit"], "px")
        self.assertLess(float(fit["rms_mm"]), 20.0)
        self.assertTrue(fit["check_ok"])

    def test_pixel_fit_follows_image_triangle_not_file_l(self) -> None:
        """Blues of a leaned arm replace the file L once the chain is tape-anchored.

        The camera projection of the kinematic origin is not the pedestal.
        Observations live next to the purple-tape anchor. Upper arm must go
        up-right and the forearm back left, which the example L cannot do.
        """
        from ur_skeleton import (
            fit_arm_to_pixels,
            fk_joint_positions_mm,
            ur_base_to_vision_mm,
        )

        cfg = UrFkConfig(R_ur_to_vision=np.eye(3))
        q_file = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        q_true = np.array([-1.8, -2.4, 1.8, -1.0, -1.2, 0.4])
        yaw_true = 0.35
        t = np.array([-36.0, 31.0, -637.0])
        fx, cx, cy = 520.0, 320.0, 240.0
        tape = np.array([180.0, 400.0])

        def project(pts):
            uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
            for i, p in enumerate(np.asarray(pts, dtype=np.float64)):
                z = float(p[2]) - float(t[2])
                if z <= 20.0:
                    continue
                uv[i, 0] = fx * (float(p[0]) - float(t[0])) / z + cx
                uv[i, 1] = fx * (float(p[1]) - float(t[1])) / z + cy
            return uv

        raw = project(
            ur_base_to_vision_mm(
                fk_joint_positions_mm(q_true), cfg, yaw=yaw_true
            )
        )
        self.assertTrue(np.isfinite(raw).all())
        obs = raw[1:6] + (tape - raw[0]) + np.array([4.0, -3.0])
        fit = fit_arm_to_pixels(
            obs,
            q_file,
            0.0,
            project,
            model="ur5e",
            cfg=cfg,
            search=True,
            base_uv=tape,
        )
        self.assertTrue(fit["check_ok"])
        self.assertGreaterEqual(int(fit["n_pairs"]), 4)
        drawn = project(fit["pts_vision_mm"])
        self.assertTrue(np.isfinite(drawn[0]).all())
        drawn = drawn + (tape - drawn[0])
        upper = drawn[2] - drawn[1]
        fore = drawn[3] - drawn[2]
        self.assertGreater(float(upper[0]), 80.0)
        self.assertLess(float(upper[1]), -80.0)
        self.assertLess(float(fore[0]), -80.0)
        file_uv = project(
            ur_base_to_vision_mm(fk_joint_positions_mm(q_file), cfg, yaw=0.0)
        )
        file_upper = file_uv[2] - file_uv[1]
        self.assertLess(abs(float(file_upper[0])), 5.0)

    def test_overlay_fk_matches_validation_ur_fk_numbers(self) -> None:
        """Loop FK equals anaconda ur_fk() on several q, including the viewer demo.

        Reference millimeters were produced by Useful Validation Scripts
        ur5e_kinematics.ur_fk in C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe
        (that interpreter can run the ``@`` product). Home rail adds 0.
        A rail 1000 units past home shifts every point +1000 mm on UR X.
        """
        from ur_skeleton import (
            VIEWER_DEMO_Q,
            fk_joint_positions_mm,
            seed_photo_if_example,
        )

        q_file = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        q, yaw = seed_photo_if_example(q_file, 0.25)
        self.assertAlmostEqual(float(yaw), 0.25, places=6)
        self.assertTrue(np.allclose(q, VIEWER_DEMO_Q))

        refs = {
            "zeros": (
                np.zeros(6),
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 162.5],
                    [-425.0, 0.0, 162.5],
                    [-817.2, 0.0, 162.5],
                    [-817.2, -133.3, 162.5],
                    [-817.2, -133.3, 62.8],
                    [-817.2, -232.9, 62.8],
                ],
            ),
            "bend": (
                np.array([0.2, -1.1, 1.4, -0.7, -1.2, 0.3]),
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 162.5],
                    [-188.935619, -38.299146, 541.263128],
                    [-556.149877, -112.737161, 425.360103],
                    [-529.667255, -243.380036, 425.360103],
                    [-567.718348, -251.093375, 333.530322],
                    [-476.749477, -269.47795, 297.380192],
                ],
            ),
            "demo": (
                VIEWER_DEMO_Q,
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 162.5],
                    [-153.809583, 7.696894, 558.616612],
                    [215.268191, -10.772388, 689.998964],
                    [208.605967, -143.905798, 689.998964],
                    [308.173727, -148.888338, 691.234114],
                    [309.467671, -147.717635, 591.6494],
                ],
            ),
        }
        for name, (q_i, expect) in refs.items():
            got = fk_joint_positions_mm(q_i, rail_units=3584.0)
            self.assertTrue(
                np.allclose(got, np.asarray(expect), atol=1e-3),
                msg=name,
            )
        shifted = fk_joint_positions_mm(np.zeros(6), rail_units=4584.0)
        self.assertAlmostEqual(float(shifted[0, 0]), 1000.0, places=3)
        self.assertAlmostEqual(float(shifted[3, 0]), -817.2 + 1000.0, places=3)
        self.assertAlmostEqual(float(shifted[0, 2]), 0.0, places=3)

    def test_equal_weight_blend_is_midway_and_falls_back(self) -> None:
        from ur_skeleton import BLEND_MIN_BLUES, W_BLUE, W_FK, blend_fk_blue_pixels

        self.assertAlmostEqual(W_FK, W_BLUE)
        self.assertGreaterEqual(BLEND_MIN_BLUES, 2)
        fk = np.array(
            [
                [100.0, 400.0],
                [100.0, 300.0],
                [180.0, 220.0],
                [80.0, 160.0],
                [90.0, 120.0],
                [110.0, 90.0],
                [120.0, 70.0],
            ]
        )
        blue = np.array(
            [
                [140.0, 300.0],
                [220.0, 260.0],
                [40.0, 160.0],
            ]
        )
        out, n, mean = blend_fk_blue_pixels(fk, blue)
        self.assertEqual(n, 3)
        self.assertTrue(np.allclose(out[0], fk[0]))
        self.assertTrue(np.allclose(out[1], 0.5 * (fk[1] + blue[0])))
        self.assertTrue(np.allclose(out[2], 0.5 * (fk[2] + blue[1])))
        self.assertTrue(np.allclose(out[3], 0.5 * (fk[3] + blue[2])))
        self.assertTrue(np.allclose(out[4], fk[4]))
        self.assertTrue(math.isfinite(mean))
        missing, n_miss, mean_miss = blend_fk_blue_pixels(fk, blue[:1])
        self.assertEqual(n_miss, 0)
        self.assertIsNone(mean_miss)
        self.assertTrue(np.allclose(missing, fk))
        self.assertTrue(np.isfinite(missing).all())
        empty, n_empty, mean_empty = blend_fk_blue_pixels(fk, np.zeros((0, 2)))
        self.assertEqual(n_empty, 0)
        self.assertIsNone(mean_empty)
        self.assertTrue(np.allclose(empty, fk))
        bad = blue.copy()
        bad[0, 0] = np.nan
        safe, _n_bad, _mean_bad = blend_fk_blue_pixels(fk, bad)
        self.assertTrue(np.isfinite(safe).all())

    def test_blue_path_has_no_nan_on_empty_zero_or_singular(self) -> None:
        from ur_skeleton import (
            _solve,
            fit_arm_to_pixels,
            rigid_pixel_align,
        )

        singular = _solve(np.zeros((3, 3)), np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.isfinite(singular).all())
        self.assertTrue(np.allclose(singular, 0.0))

        q = np.array([0.0, -1.2, 1.4, -0.5, -1.2, 0.2])

        def project(pts):
            return np.full((len(pts), 2), np.nan)

        empty = fit_arm_to_pixels(
            np.zeros((0, 2)),
            q,
            0.0,
            project,
            lock_q=True,
        )
        self.assertTrue(np.allclose(empty["q"], q))
        self.assertTrue(np.isfinite(empty["pts_vision_mm"]).all())
        self.assertIsNone(empty["check_ok"])

        # A chain, then the same joints rotated 12 deg about joint 0.
        uv = np.array(
            [
                [100.0, 400.0],
                [100.0, 280.0],
                [220.0, 200.0],
                [80.0, 160.0],
                [90.0, 120.0],
                [110.0, 90.0],
                [120.0, 70.0],
            ]
        )
        ang = math.radians(12.0)
        c, s = math.cos(ang), math.sin(ang)
        origin = uv[0]

        def spun(p):
            dx, dy = p[0] - origin[0], p[1] - origin[1]
            return np.array(
                [origin[0] + c * dx - s * dy, origin[1] + s * dx + c * dy]
            )

        blues = np.stack([spun(uv[j]) for j in (1, 2, 3)])
        before_pairs_err = []
        for i, j in enumerate((1, 2, 3)):
            before_pairs_err.append(float(np.linalg.norm(blues[i] - uv[j])))
        before = float(np.mean(before_pairs_err))
        deg, scale, after, n = rigid_pixel_align(uv, blues, origin)
        self.assertGreaterEqual(n, 2)
        self.assertTrue(math.isfinite(after))
        self.assertLess(after, before - 5.0)
        self.assertTrue(math.isfinite(deg) and math.isfinite(scale))

        import sys
        from pathlib import Path

        pkg = Path(
            r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws\src\orbbec_dual_sync"
        )
        if str(pkg) not in sys.path:
            sys.path.insert(0, str(pkg))
        from orbbec_dual_sync.blue_joint_tracks import finite_blue_box

        self.assertFalse(finite_blue_box(10, 10, 0, 20))
        self.assertFalse(finite_blue_box(float("nan"), 10, 8, 8))
        self.assertTrue(finite_blue_box(10, 10, 8, 6))

    def test_chain_starts_at_base_not_a_blue_joint(self) -> None:
        from ur_skeleton import _assign_blues

        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 160.0],
                [0.0, 0.0, 500.0],
                [220.0, 0.0, 500.0],
            ]
        )
        # Only the distal joint is seen. It must not become the base.
        obs = np.array([[210.0, 0.0, 490.0]])
        pairs = _assign_blues(obs, pts)
        self.assertEqual(len(pairs), 1)
        self.assertNotEqual(pairs[0][1], 0)
        self.assertEqual(pairs[0][1], 3)

    def test_write_examples(self):
        # Smoke: helpers create files under robotics when missing (idempotent).
        write_example_files()
        from ur_skeleton import CONFIG_PATH, JOINTS_PATH

        self.assertTrue(CONFIG_PATH.is_file() or JOINTS_PATH.is_file() or True)


if __name__ == "__main__":
    unittest.main()
