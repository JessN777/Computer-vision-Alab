"""Continuous camera-pose estimate from arm stickers and blue joint caps.

One ``OnlineCamPose`` per camera. Called from ``dual_view_node._overlay_ur_fk``.

Flow:
  add_observations(fk, live tags, live blues) -> rolling buffer
  update() -> solvePnPRansac on tags (blues optional) + gated Kabsch 3D-3D
  project() / draw uses the same K, dist as the solve

Rules of thumb:
  - Kabsch is accepted only if its tag reprojection beats the current pose
  - Blues require mean tag error < 10 px; assign_px default 25
  - Blues stay out of solvePnPRansac unless use_blue_in_pnp is True
  - No ``@`` / ``np.dot`` / ``np.matmul`` (broken BLAS on this PC)

cell_frames convention: p_base = R p_cam + t_mm.
OpenCV solvePnP: p_cam = R_cv p_base + t_cv, so R = R_cv^T, t = -R_cv^T t_cv.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

# Tag id i is FK joint i: 0 base, 1 shoulder, 2 elbow, 3 wrist1, 4 wrist2, 5 wrist3.


def _mat3_mul(a, b):
    out = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(3):
        for j in range(3):
            out[i][j] = (
                float(a[i][0]) * float(b[0][j])
                + float(a[i][1]) * float(b[1][j])
                + float(a[i][2]) * float(b[2][j])
            )
    return out


def _mat3_vec(a, v):
    out = [0.0, 0.0, 0.0]
    for i in range(3):
        out[i] = (
            float(a[i][0]) * float(v[0])
            + float(a[i][1]) * float(v[1])
            + float(a[i][2]) * float(v[2])
        )
    return out


def _transpose3(a):
    return [[float(a[j][i]) for j in range(3)] for i in range(3)]


def _cv_from_cell(R_cell, t_cell):
    """cell (R,t) -> OpenCV (rvec, tvec). p_cam = R_cv p_base + t_cv."""
    R_cv = _transpose3(R_cell)
    t_cv = _mat3_vec(R_cv, (-float(t_cell[0]), -float(t_cell[1]), -float(t_cell[2])))
    rvec, _ = cv2.Rodrigues(np.asarray(R_cv, dtype=np.float64))
    return rvec, np.asarray(t_cv, dtype=np.float64).reshape(3, 1)


def _cell_from_cv(rvec, tvec):
    """OpenCV (rvec, tvec) -> cell (R, t). p_base = R p_cam + t."""
    R_cv, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    R_cv = [[float(R_cv[i, j]) for j in range(3)] for i in range(3)]
    t_cv = [float(v) for v in np.asarray(tvec, dtype=np.float64).reshape(3)]
    R = _transpose3(R_cv)
    t = _mat3_vec(R, (-t_cv[0], -t_cv[1], -t_cv[2]))
    return (
        np.asarray(R, dtype=np.float64),
        np.asarray(t, dtype=np.float64).reshape(3),
    )


def _rot_angle_deg(R_a, R_b):
    """Angle between two rotations, in degrees, with no matmul."""
    rel = _mat3_mul(_transpose3(R_a), R_b)
    trace = float(rel[0][0]) + float(rel[1][1]) + float(rel[2][2])
    c = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(c))


def _det3(R) -> float:
    return (
        float(R[0][0]) * (float(R[1][1]) * float(R[2][2]) - float(R[1][2]) * float(R[2][1]))
        - float(R[0][1]) * (float(R[1][0]) * float(R[2][2]) - float(R[1][2]) * float(R[2][0]))
        + float(R[0][2]) * (float(R[1][0]) * float(R[2][1]) - float(R[1][1]) * float(R[2][0]))
    )


def umeyama_cell(p_cam, p_base):
    """Rigid p_base = R p_cam + t from 3D-3D pairs (Kabsch / Umeyama).

    3x3 SVD is ``cv2.SVDecomp``; the products are explicit loops (no ``@``).
    Returns ``(R, t_mm, rms_mm)`` or None.
    """
    n = len(p_cam)
    if n < 3 or n != len(p_base):
        return None
    src = [[float(p[0]), float(p[1]), float(p[2])] for p in p_cam]
    dst = [[float(p[0]), float(p[1]), float(p[2])] for p in p_base]
    cs = [0.0, 0.0, 0.0]
    cd = [0.0, 0.0, 0.0]
    for i in range(n):
        for k in range(3):
            cs[k] += src[i][k]
            cd[k] += dst[i][k]
    inv = 1.0 / float(n)
    for k in range(3):
        cs[k] *= inv
        cd[k] *= inv
    H = np.zeros((3, 3), dtype=np.float64)
    for i in range(n):
        for r in range(3):
            for c in range(3):
                H[r, c] += (src[i][r] - cs[r]) * (dst[i][c] - cd[c])
    _w, u, vt = cv2.SVDecomp(H)
    U = [[float(u[i, j]) for j in range(3)] for i in range(3)]
    Vt = [[float(vt[i, j]) for j in range(3)] for i in range(3)]
    V = _transpose3(Vt)
    R = _mat3_mul(V, _transpose3(U))
    if _det3(R) < 0.0:
        V[0][2] = -V[0][2]
        V[1][2] = -V[1][2]
        V[2][2] = -V[2][2]
        R = _mat3_mul(V, _transpose3(U))
    Rc = _mat3_vec(R, cs)
    t = [cd[0] - Rc[0], cd[1] - Rc[1], cd[2] - Rc[2]]
    err = 0.0
    for i in range(n):
        pred = _mat3_vec(R, src[i])
        dx = pred[0] + t[0] - dst[i][0]
        dy = pred[1] + t[1] - dst[i][1]
        dz = pred[2] + t[2] - dst[i][2]
        err += (dx * dx + dy * dy + dz * dz) ** 0.5
    return (
        np.asarray(R, dtype=np.float64),
        np.asarray(t, dtype=np.float64).reshape(3),
        err / float(n),
    )


class OnlineCamPose:
    def __init__(
        self,
        K,
        dist=None,
        R=None,
        t_mm=None,
        *,
        cam: str = "",
        buffer_size: int = 200,
        min_pairs: int = 8,
        min_poses: int = 3,
        same_pose_mm: float = 20.0,
        fast_speed: float = 0.15,
        assign_px: float = 25.0,
        max_jump_mm: float = 150.0,
        max_jump_deg: float = 15.0,
        update_period_s: float = 1.0,
        use_blue: bool = True,
        use_blue_in_pnp: bool = False,
        min_3d_pairs: int = 3,
    ):
        self.cam = cam
        self.set_intrinsics(K, dist)
        self.R: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None
        if R is not None and t_mm is not None:
            self.R = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
            self.t = np.asarray(t_mm, dtype=np.float64).reshape(3).copy()
        self.buffer: list[dict] = []
        self.buffer_size = int(buffer_size)
        self.min_pairs = int(min_pairs)
        self.min_poses = int(min_poses)
        self.same_pose_mm = float(same_pose_mm)
        self.fast_speed = float(fast_speed)
        self.assign_px = float(assign_px)
        self.max_jump_mm = float(max_jump_mm)
        self.max_jump_deg = float(max_jump_deg)
        self.update_period_s = float(update_period_s)
        self.use_blue = bool(use_blue)
        self.use_blue_in_pnp = bool(use_blue_in_pnp)
        self.min_3d_pairs = int(min_3d_pairs)
        # Max-jump check stays off until a pose is accepted with tag error < 10 px.
        self._jump_armed = False
        self._last_fk: Optional[np.ndarray] = None
        self._next_pose_id = 0
        self._last_update_t = 0.0
        # HUD / status
        self.mean_px = float("nan")
        self.mean_3d_mm = float("nan")
        self.last_accepted = False
        self.n_pairs = 0
        self.n_poses = 0
        self.n_3d = 0

    # -- intrinsics ---------------------------------------------------------
    def set_intrinsics(self, K, dist=None):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
        self.dist = (
            None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        )

    @property
    def have_pose(self) -> bool:
        return self.R is not None and self.t is not None

    def reset(self) -> None:
        """Clear the buffer and pose so the next solve is tags-only / fresh."""
        self.buffer.clear()
        self.R = None
        self.t = None
        self._last_fk = None
        self._last_update_t = 0.0
        self._next_pose_id = 0
        self.n_poses = 0
        self.n_pairs = 0
        self.n_3d = 0
        self.mean_px = float("nan")
        self.mean_3d_mm = float("nan")
        self.last_accepted = False
        self._jump_armed = False

    # -- ingest -------------------------------------------------------------
    def add_observations(
        self, fk_points_mm, tag_uv, blue_uv, arm_speed=None, blue_xyz=None
    ) -> None:
        """Pair sticker/blue pixels with FK joints and push to the buffer.

        ``fk_points_mm`` is (7,3) in the robot base. ``tag_uv`` is a list of 6
        (u,v) or None. ``blue_uv`` is a list of unlabeled pixel points.
        ``blue_xyz`` is an optional parallel list of camera-frame 3D points
        (mm) from a valid depth sample; None entries are 2D-only. The joint
        reading and the image are not captured at the same instant, so a
        fast-moving arm is skipped, as is a pose almost identical to the last.
        """
        if arm_speed is not None and float(arm_speed) > self.fast_speed:
            return
        fk = np.asarray(fk_points_mm, dtype=np.float64).reshape(-1, 3)
        if fk.shape[0] < 6 or not np.isfinite(fk).all():
            return
        if self._last_fk is not None and self._last_fk.shape == fk.shape:
            moved = 0.0
            for i in range(fk.shape[0]):
                d = (
                    (fk[i, 0] - self._last_fk[i, 0]) ** 2
                    + (fk[i, 1] - self._last_fk[i, 1]) ** 2
                    + (fk[i, 2] - self._last_fk[i, 2]) ** 2
                ) ** 0.5
                if d > moved:
                    moved = d
            if moved < self.same_pose_mm:
                return

        pairs: list[dict] = []
        # Tags: id i -> FK joint i.
        if tag_uv is not None:
            for i, uv in enumerate(tag_uv):
                if uv is None or i >= fk.shape[0]:
                    continue
                u, v = float(uv[0]), float(uv[1])
                if not (math.isfinite(u) and math.isfinite(v)):
                    continue
                pairs.append(
                    {
                        "obj": (float(fk[i, 0]), float(fk[i, 1]), float(fk[i, 2])),
                        "img": (u, v),
                        "weight": 1.0,
                        "is_blue": False,
                        "cam_xyz": None,
                    }
                )

        # Blue caps: only when the current pose's mean tag reproj is < 10 px.
        # Caller must pass live blobs only (no coasting).
        xyz_list = []
        if blue_xyz is not None:
            xyz_list = list(blue_xyz)
        allow_blue = False
        if self.use_blue and blue_uv is not None and self.have_pose:
            tag_for_err = [pr for pr in pairs if not pr["is_blue"]]
            if tag_for_err:
                tag_err = self._mean_reproj(*self._cv_pose(), tag_for_err)
                allow_blue = math.isfinite(tag_err) and tag_err < 10.0
        if allow_blue:
            proj = self.project(fk)
            blue_list = blue_uv
            if hasattr(blue_uv, "reshape"):
                arr = np.asarray(blue_uv, dtype=np.float64).reshape(-1, 2)
                blue_list = [(float(p[0]), float(p[1])) for p in arr]
            for bi, b in enumerate(blue_list):
                if b is None:
                    continue
                bu, bv = float(b[0]), float(b[1])
                if not (math.isfinite(bu) and math.isfinite(bv)):
                    continue
                best_i = -1
                best_d = self.assign_px
                for i, p in enumerate(proj):
                    if p is None:
                        continue
                    d = math.hypot(bu - float(p[0]), bv - float(p[1]))
                    if d < best_d:
                        best_d = d
                        best_i = i
                if best_i < 0:
                    continue
                cam_xyz = None
                if bi < len(xyz_list) and xyz_list[bi] is not None:
                    xyz = xyz_list[bi]
                    cam_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                    if not all(math.isfinite(v) for v in cam_xyz):
                        cam_xyz = None
                pairs.append(
                    {
                        "obj": (
                            float(fk[best_i, 0]),
                            float(fk[best_i, 1]),
                            float(fk[best_i, 2]),
                        ),
                        "img": (bu, bv),
                        "weight": 0.3,
                        "is_blue": True,
                        "cam_xyz": cam_xyz,
                    }
                )

        if not pairs:
            return
        pose_id = self._next_pose_id
        self._next_pose_id += 1
        for pr in pairs:
            pr["pose_id"] = pose_id
            self.buffer.append(pr)
        while len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self._last_fk = fk.copy()

    # -- status helpers -----------------------------------------------------
    def _distinct_poses(self, use_blue: bool) -> int:
        ids = set()
        for pr in self.buffer:
            if pr["is_blue"] and not use_blue:
                continue
            ids.add(pr["pose_id"])
        return len(ids)

    def _mean_reproj(self, rvec, tvec, rows) -> float:
        if not rows:
            return float("inf")
        obj = np.asarray([r["obj"] for r in rows], dtype=np.float32).reshape(-1, 3)
        img = [r["img"] for r in rows]
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        flat = proj.reshape(-1, 2)
        err = 0.0
        for i in range(len(rows)):
            du = float(flat[i, 0]) - float(img[i][0])
            dv = float(flat[i, 1]) - float(img[i][1])
            err += (du * du + dv * dv) ** 0.5
        return err / float(len(rows))

    # -- solve --------------------------------------------------------------
    def _try_accept_kabsch(self, kabsch, tag_rows) -> bool:
        """Accept Umeyama only if its 2D tag reproj beats the current pose."""
        if kabsch is None or not tag_rows:
            return False
        Rk, tk = kabsch[0], kabsch[1]
        rvec_k, tvec_k = _cv_from_cell(
            [[float(Rk[i, j]) for j in range(3)] for i in range(3)],
            (float(tk[0]), float(tk[1]), float(tk[2])),
        )
        kabsch_err = self._mean_reproj(rvec_k, tvec_k, tag_rows)
        if not math.isfinite(kabsch_err):
            return False
        if self.have_pose:
            cur_err = self._mean_reproj(*self._cv_pose(), tag_rows)
            if math.isfinite(cur_err) and not (kabsch_err < cur_err):
                return False
        self.mean_px = kabsch_err
        return self._accept_pose(Rk, tk)

    def update(self, use_blue: Optional[bool] = None) -> bool:
        """Re-solve at most once per update period. Returns True if accepted."""
        now = time.monotonic()
        if now - self._last_update_t < self.update_period_s:
            return False
        self._last_update_t = now
        ub = self.use_blue if use_blue is None else bool(use_blue)
        rows = [pr for pr in self.buffer if ub or not pr["is_blue"]]
        rows3 = [pr for pr in rows if pr.get("cam_xyz") is not None]
        self.n_pairs = len(rows)
        self.n_poses = self._distinct_poses(ub)
        self.n_3d = len(rows3)
        self.last_accepted = False

        kabsch = None
        if len(rows3) >= self.min_3d_pairs:
            kabsch = umeyama_cell(
                [pr["cam_xyz"] for pr in rows3], [pr["obj"] for pr in rows3]
            )
            if kabsch is not None:
                self.mean_3d_mm = float(kabsch[2])

        tag_rows = [pr for pr in rows if not pr["is_blue"]]
        # Blues stay out of solvePnPRansac unless online_use_blue_in_pnp.
        pnp_rows = rows if self.use_blue_in_pnp else tag_rows
        pnp_poses = self._distinct_poses(bool(self.use_blue_in_pnp and ub))

        if len(pnp_rows) < self.min_pairs or pnp_poses < self.min_poses:
            if self._try_accept_kabsch(kabsch, tag_rows):
                return True
            if self.have_pose and tag_rows:
                self.mean_px = self._mean_reproj(*self._cv_pose(), tag_rows)
            return False

        obj = np.asarray([r["obj"] for r in pnp_rows], dtype=np.float32).reshape(-1, 3)
        img = np.asarray([r["img"] for r in pnp_rows], dtype=np.float32).reshape(-1, 2)

        guess = self.have_pose or kabsch is not None
        if self.have_pose:
            rvec0, tvec0 = self._cv_pose()
        elif kabsch is not None:
            rvec0, tvec0 = _cv_from_cell(
                [[float(kabsch[0][i, j]) for j in range(3)] for i in range(3)],
                (float(kabsch[1][0]), float(kabsch[1][1]), float(kabsch[1][2])),
            )
        else:
            rvec0 = np.zeros((3, 1), dtype=np.float64)
            tvec0 = np.zeros((3, 1), dtype=np.float64)

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                img,
                self.K,
                self.dist,
                useExtrinsicGuess=bool(guess),
                rvec=rvec0.copy(),
                tvec=tvec0.copy(),
                reprojectionError=8.0,
                iterationsCount=200,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            ok, rvec, tvec, inliers = False, None, None, None
        cmp_rows = tag_rows if tag_rows else pnp_rows
        if not ok or rvec is None or inliers is None or len(inliers) < self.min_pairs:
            if self._try_accept_kabsch(kabsch, tag_rows):
                return True
            if self.have_pose:
                self.mean_px = self._mean_reproj(rvec0, tvec0, cmp_rows)
            return False

        idx = [int(i) for i in inliers.reshape(-1)]
        obj_in = obj[idx].reshape(-1, 1, 3)
        img_in = img[idx].reshape(-1, 1, 2)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_in, img_in, self.K, self.dist, rvec, tvec
            )
        except Exception:
            pass

        new_err = self._mean_reproj(rvec, tvec, cmp_rows)
        old_err = self._mean_reproj(rvec0, tvec0, cmp_rows) if guess else float("inf")
        self.mean_px = new_err
        if not math.isfinite(new_err):
            return self._try_accept_kabsch(kabsch, tag_rows)
        # Accept only when the fit over the tag rows improves.
        if guess and new_err > old_err + 0.5:
            if self._try_accept_kabsch(kabsch, tag_rows):
                return True
            return False
        R_new, t_new = _cell_from_cv(rvec, tvec)
        return self._accept_pose(R_new, t_new)

    def _accept_pose(self, R_new, t_new) -> bool:
        """Keep the candidate if it does not jump too far from the current pose."""
        if self.have_pose and self._jump_armed:
            dt = (
                (float(t_new[0]) - float(self.t[0])) ** 2
                + (float(t_new[1]) - float(self.t[1])) ** 2
                + (float(t_new[2]) - float(self.t[2])) ** 2
            ) ** 0.5
            dang = _rot_angle_deg(
                [[float(self.R[i, j]) for j in range(3)] for i in range(3)],
                [[float(R_new[i, j]) for j in range(3)] for i in range(3)],
            )
            if dt > self.max_jump_mm or dang > self.max_jump_deg:
                return False
        self.R = np.asarray(R_new, dtype=np.float64).reshape(3, 3).copy()
        self.t = np.asarray(t_new, dtype=np.float64).reshape(3).copy()
        self.last_accepted = True
        if math.isfinite(float(self.mean_px)) and float(self.mean_px) < 10.0:
            self._jump_armed = True
        return True

    def _cv_pose(self):
        return _cv_from_cell(
            [[float(self.R[i, j]) for j in range(3)] for i in range(3)],
            (float(self.t[0]), float(self.t[1]), float(self.t[2])),
        )

    # -- projection ---------------------------------------------------------
    def project(self, fk_points_mm) -> list:
        """FK base points -> pixel list (None where behind the camera)."""
        fk = np.asarray(fk_points_mm, dtype=np.float32).reshape(-1, 3)
        if not self.have_pose:
            return [None] * int(fk.shape[0])
        rvec, tvec = self._cv_pose()
        proj, _ = cv2.projectPoints(fk, rvec, tvec, self.K, self.dist)
        flat = proj.reshape(-1, 2)
        out: list = []
        for i in range(fk.shape[0]):
            u, v = float(flat[i, 0]), float(flat[i, 1])
            if math.isfinite(u) and math.isfinite(v):
                out.append((int(round(u)), int(round(v))))
            else:
                out.append(None)
        return out

    # -- persist ------------------------------------------------------------
    def save(self, cell_path=None) -> bool:
        """Write this camera's pose into cell_frames.json (layout arm_tags)."""
        if not self.have_pose:
            return False
        from .cell_frames import LAYOUT_ARM_TAGS, load_cell_frames, save_cell_frames

        cf = load_cell_frames(cell_path)
        pose = cf.pose(self.cam)
        pose.R = self.R.copy()
        pose.t_mm = self.t.copy()
        pose.ready = True
        cf.layout = LAYOUT_ARM_TAGS
        cf.bootstrap_from_blue_base = False
        cf.fx_px = float(self.K[0, 0])
        save_cell_frames(cf)
        return True
