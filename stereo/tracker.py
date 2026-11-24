"""Multi-view stereo 3D pose tracker with foreshortening-aware kinematic fusion.

Triangulates directly from the two calibrated camera rays, learns each person's
rigid bone lengths from clean high-confidence dual-view frames, and uses
visibility + foreshortening weighting plus a single-camera ray + bone-sphere
handover whenever one camera looks straight down the arm or loses sight of a
joint while the other camera still has a clean side view.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

from arm.protocol import Joint
from stereo.charuco import CharucoHelper, paired_charuco_points
from stereo.config import StereoConfig
from stereo.filters import OneEuro1D, OneEuro3D
from stereo.geometry import (
    RigFrame,
    StereoCalibration,
    project_points,
    rig_frame_from_calibration,
    triangulate_with_diagnostics,
)
from stereo.kinematics import calculate_3d_angle

NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24

UPPER_BODY_INDICES = (
    NOSE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_PINKY,
    RIGHT_PINKY,
    LEFT_INDEX,
    RIGHT_INDEX,
    LEFT_THUMB,
    RIGHT_THUMB,
    LEFT_HIP,
    RIGHT_HIP,
)

LEFT_ARM_INDICES = (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX, LEFT_PINKY, LEFT_THUMB)

_LR_SWAP = {
    7: 8, 8: 7,
    11: 12, 12: 11,
    13: 14, 14: 13,
    15: 16, 16: 15,
    17: 18, 18: 17,
    19: 20, 20: 19,
    21: 22, 22: 21,
    23: 24, 24: 23,
}

DEPTH_MIN_M = 0.25
DEPTH_MAX_M = 3.50
LIMB_MIN_M = 0.14
LIMB_MAX_M = 0.48
SHOULDER_MIN_M = 0.20
SHOULDER_MAX_M = 0.65

MAX_SAMPSON_PX = 38.0
MAX_REPROJ_PX = 46.0
TORSO_MAX_SAMPSON_PX = 52.0
TORSO_MAX_REPROJ_PX = 60.0
TORSO_INDICES = frozenset((LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP))
HOLD_MAX_AGE_S = 0.28

_PARENT_JOINT: dict[int, int] = {
    LEFT_ELBOW: LEFT_SHOULDER,
    LEFT_WRIST: LEFT_ELBOW,
    LEFT_INDEX: LEFT_WRIST,
    LEFT_PINKY: LEFT_WRIST,
    LEFT_THUMB: LEFT_WRIST,
    RIGHT_ELBOW: RIGHT_SHOULDER,
    RIGHT_WRIST: RIGHT_ELBOW,
    RIGHT_INDEX: RIGHT_WRIST,
    RIGHT_PINKY: RIGHT_WRIST,
    RIGHT_THUMB: RIGHT_WRIST,
}


@dataclass(frozen=True)
class StereoJoint:
    index: int
    xyz_cam1: np.ndarray
    xyz_rig: np.ndarray
    pt1: tuple[float, float]
    pt2: tuple[float, float]
    reproj1_px: float
    reproj2_px: float
    sampson_px: float
    ray_gap_mm: float
    ray_angle_deg: float
    quality: str           # "good" | "fair" | "poor"
    source: str = "stereo"  # "stereo" | "ray+bone"

    @property
    def mean_reproj_px(self) -> float:
        vals = [v for v in (self.reproj1_px, self.reproj2_px) if math.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")


@dataclass(frozen=True)
class ArmKinematics:
    upper_arm_cm: float | None = None
    forearm_cm: float | None = None
    hip_yaw_deg: float | None = None
    elevation_deg: float | None = None
    forward_pitch_deg: float | None = None
    side_abduction_deg: float | None = None
    elbow_interior_deg: float | None = None
    elbow_flexion_deg: float | None = None
    wrist_angle_deg: float | None = None
    hand_open_deg: float | None = None

    def to_uart_angles(self) -> dict[Joint, float]:
        """Map 3D arm kinematics to the 5 STM32 joint commands in [0, 180] deg.

        Joint conventions match ``stm32/Core/Inc/main.h`` and ``pose-estimation.py``:
        - ``HIP`` (0): 1:1 torso twist around neutral facing angle (0° twisted -90°, 90° neutral, 180° twisted +90°)
        - ``HAND`` (1): hand openness (0° closed fist/pinch, 180° wide open)
        - ``ELBOW`` (2): elbow flexion angle (0° straight, 90° bent, 180° folded)
        - ``WRIST`` (3): wrist roll around forearm (0° palm down, 90° thumb up, 180° palm up)
        - ``SHOULDER`` (4): upper-arm elevation from hanging down (0° down, 90° level, 180° up)
        """
        out: dict[Joint, float] = {}
        mapping = (
            (Joint.HIP, self.hip_yaw_deg),
            (Joint.HAND, self.hand_open_deg),
            (Joint.ELBOW, self.elbow_flexion_deg),
            (Joint.WRIST, self.wrist_angle_deg),
            (Joint.SHOULDER, self.elevation_deg),
        )
        for joint, val in mapping:
            if val is not None and math.isfinite(val):
                out[joint] = float(np.clip(val, 0.0, 180.0))
        return out


@dataclass(frozen=True)
class BoardCheckResult:
    corners_rig: np.ndarray
    expected_square_mm: float
    measured_square_mm: float
    scale_error_pct: float
    planarity_rms_mm: float
    mean_ray_gap_mm: float
    mean_reproj_px: float
    depth_m: float
    n_corners: int


@dataclass(frozen=True)
class StereoFrame:
    joints: dict[int, StereoJoint]
    raw_joints: dict[int, StereoJoint]
    hypothesis: str
    left_arm: ArmKinematics
    right_arm: ArmKinematics
    median_ray_gap_mm: float
    median_reproj_px: float
    median_sampson_px: float
    board_check: BoardCheckResult | None = None


def _quality_tier(ray_gap_mm: float, mean_reproj_px: float, sampson_px: float) -> str:
    if not math.isfinite(ray_gap_mm):
        return "fair"
    if ray_gap_mm <= 15.0 and mean_reproj_px <= 6.0 and sampson_px <= 6.0:
        return "good"
    if ray_gap_mm <= 35.0 and mean_reproj_px <= 14.0 and sampson_px <= 12.0:
        return "fair"
    return "poor"


def _unpack_obs(obs: dict[int, tuple]) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    """Accept either (x, y) or (x, y, visibility) landmark maps."""
    xy: dict[int, tuple[float, float]] = {}
    vis: dict[int, float] = {}
    for idx, val in obs.items():
        xy[idx] = (float(val[0]), float(val[1]))
        vis[idx] = float(val[2]) if len(val) >= 3 else 1.0
    return xy, vis


def _camera_ray_in_rig(
    calib: StereoCalibration,
    rig: RigFrame,
    pixel: tuple[float, float],
    camera: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin_rig, unit_dir_rig) for a distorted pixel in camera 1 or 2."""
    px = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
    if camera == 1:
        norm = cv2.undistortPoints(px, calib.K1, calib.D1).reshape(2)
        d_cam1 = np.array([float(norm[0]), float(norm[1]), 1.0], dtype=np.float64)
        d_cam1 /= np.linalg.norm(d_cam1)
        return rig.cam1_pos_rig, rig.dir_to_rig(d_cam1)
    norm = cv2.undistortPoints(px, calib.K2, calib.D2).reshape(2)
    d_cam2 = np.array([float(norm[0]), float(norm[1]), 1.0], dtype=np.float64)
    d_cam2 /= np.linalg.norm(d_cam2)
    d_cam1 = np.asarray(calib.R, dtype=np.float64).T @ d_cam2
    return rig.cam2_pos_rig, rig.dir_to_rig(d_cam1)


def _point_to_ray_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    rel = point - origin
    along = float(np.dot(rel, direction))
    perp = rel - along * direction
    return float(np.linalg.norm(perp))


def _ray_sphere_candidates(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> list[np.ndarray]:
    """Intersect ray origin + t*direction (t > 0.25) with sphere (center, radius).

    If the ray narrowly misses the sphere (e.g. a few cm due to 2D pixel noise
    at full extension), returns the single closest point projected onto the
    sphere surface so tracking stays continuous instead of dropping.
    """
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - radius * radius)
    disc = b * b - c
    if disc >= 0.0:
        root = math.sqrt(disc)
        pts: list[np.ndarray] = []
        for t in (-b - root, -b + root):
            if t > 0.25:
                pts.append(origin + t * direction)
        if pts:
            return pts

    # Closest point on ray to sphere center, then clamp smoothly to sphere if within 12 cm.
    t_closest = max(0.25, -b)
    p_ray = origin + t_closest * direction
    diff = p_ray - center
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6 or (dist - radius) > 0.12:
        return []
    return [center + diff * (radius / dist)]


def _solve_weighted_joint(
    init_pt: np.ndarray,
    rays: list[tuple[np.ndarray, np.ndarray, float]],
    proximal: np.ndarray | None,
    bone_len: float | None,
    bone_weight: float,
    prev_pt: np.ndarray | None,
    smooth_weight: float,
    iters: int = 8,
) -> np.ndarray:
    """Gauss-Newton refinement combining weighted camera rays, rigid bone length, and smoothness.

    Minimizes:
        sum_k w_k * ||(I - d_k d_k^T)(P - C_k)||^2
      + w_bone * (||P - P_prox|| - L)^2
      + w_smooth * ||P - P_prev||^2
    """
    p = np.asarray(init_pt, dtype=np.float64).copy()
    eye3 = np.eye(3, dtype=np.float64)

    # Precompute constant linear normal equations from camera rays and smoothness prior.
    H_const = np.zeros((3, 3), dtype=np.float64)
    g_const = np.zeros(3, dtype=np.float64)
    for origin, direction, weight in rays:
        if weight <= 0.0:
            continue
        proj = eye3 - np.outer(direction, direction)
        H_const += weight * proj
        g_const += weight * (proj @ origin)

    if prev_pt is not None and smooth_weight > 0.0:
        H_const += smooth_weight * eye3
        g_const += smooth_weight * prev_pt

    if proximal is None or bone_len is None or bone_weight <= 0.0:
        H = H_const + 1e-9 * eye3
        rhs = g_const - H_const @ p
        return p + np.linalg.solve(H, rhs)

    for _ in range(iters):
        H = H_const.copy()
        rhs = g_const - H_const @ p

        vec = p - proximal
        dist = float(np.linalg.norm(vec))
        if dist > 1e-5:
            u = vec / dist
            r_bone = dist - bone_len
            H += bone_weight * np.outer(u, u)
            rhs -= bone_weight * r_bone * u

        H += 1e-6 * eye3
        step = np.linalg.solve(H, rhs)
        p += step
        if float(np.linalg.norm(step)) < 1e-5:
            break
    return p


class BoneLengthModel:
    """Learns the person's metric bone lengths from clean, high-confidence dual-view frames."""

    def __init__(self, max_samples: int = 120):
        self._samples: dict[tuple[int, int], deque[float]] = {
            (LEFT_SHOULDER, LEFT_ELBOW): deque(maxlen=max_samples),
            (LEFT_ELBOW, LEFT_WRIST): deque(maxlen=max_samples),
            (RIGHT_SHOULDER, RIGHT_ELBOW): deque(maxlen=max_samples),
            (RIGHT_ELBOW, RIGHT_WRIST): deque(maxlen=max_samples),
            (LEFT_SHOULDER, RIGHT_SHOULDER): deque(maxlen=max_samples),
        }
        self._mirror_pair = {
            (LEFT_SHOULDER, LEFT_ELBOW): (RIGHT_SHOULDER, RIGHT_ELBOW),
            (RIGHT_SHOULDER, RIGHT_ELBOW): (LEFT_SHOULDER, LEFT_ELBOW),
            (LEFT_ELBOW, LEFT_WRIST): (RIGHT_ELBOW, RIGHT_WRIST),
            (RIGHT_ELBOW, RIGHT_WRIST): (LEFT_ELBOW, LEFT_WRIST),
        }

    def update(
        self,
        raw_joints: dict[int, StereoJoint],
        vis1: dict[int, float],
        vis2: dict[int, float],
    ) -> None:
        for pair, buf in self._samples.items():
            a, b = pair
            if a not in raw_joints or b not in raw_joints:
                continue
            ja, jb = raw_joints[a], raw_joints[b]
            if ja.source != "stereo" or jb.source != "stereo":
                continue
            if min(vis1.get(a, 0.0), vis2.get(a, 0.0), vis1.get(b, 0.0), vis2.get(b, 0.0)) < 0.72:
                continue
            if max(ja.ray_gap_mm, jb.ray_gap_mm) > 15.0 or max(ja.sampson_px, jb.sampson_px) > 8.0:
                continue
            dist = float(np.linalg.norm(jb.xyz_rig - ja.xyz_rig))
            lo, hi = (SHOULDER_MIN_M, SHOULDER_MAX_M) if pair == (LEFT_SHOULDER, RIGHT_SHOULDER) else (0.16, 0.44)
            if lo <= dist <= hi:
                buf.append(dist)

    def get(self, proximal: int, distal: int) -> float | None:
        pair = (proximal, distal)
        buf = self._samples.get(pair)
        if buf and len(buf) >= 5:
            return float(np.median(buf))
        mirror = self._mirror_pair.get(pair)
        if mirror is not None:
            mbuf = self._samples.get(mirror)
            if mbuf and len(mbuf) >= 8:
                return float(np.median(mbuf))
        if buf and len(buf) >= 2:
            return float(np.median(buf))
        return None


def _wrap_deg_180(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def compute_arm_kinematics(
    points_rig: dict[int, np.ndarray],
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    side_sign: float,
    *,
    opp_shoulder_idx: int | None = None,
    index_idx: int | None = None,
    pinky_idx: int | None = None,
    thumb_idx: int | None = None,
    hip_yaw_override_deg: float | None = None,
    use_hip_yaw_override: bool = False,
    prev_hip_yaw_deg: float | None = None,
) -> ArmKinematics:
    """Compute intuitive 3D hip/shoulder/elbow/wrist/hand angles in the rig-aligned frame."""
    sh = points_rig.get(shoulder_idx)
    el = points_rig.get(elbow_idx)
    if sh is None or el is None:
        return ArmKinematics(hip_yaw_deg=hip_yaw_override_deg)

    v_up = el - sh
    upper_m = float(np.linalg.norm(v_up))
    if upper_m < 1e-4:
        return ArmKinematics(hip_yaw_deg=hip_yaw_override_deg)

    cos_el = float(np.clip(v_up[1] / upper_m, -1.0, 1.0))
    elevation_deg = float(np.degrees(np.arccos(cos_el)))
    forward_pitch_deg = float(np.degrees(math.atan2(-float(v_up[2]), float(v_up[1]))))
    side_abduction_deg = float(np.degrees(math.atan2(side_sign * float(v_up[0]), float(v_up[1]))))

    wr = points_rig.get(wrist_idx)

    if use_hip_yaw_override or hip_yaw_override_deg is not None:
        hip_yaw_deg: float | None = (
            float(np.clip(hip_yaw_override_deg, 0.0, 180.0)) if hip_yaw_override_deg is not None else None
        )
    else:
        # Fallback pure shoulder-line torso twist (0 deg = -90 deg twist, 90 deg = facing cameras, 180 deg = +90 deg twist).
        opp_sh = points_rig.get(opp_shoulder_idx) if opp_shoulder_idx is not None else None
        if opp_sh is not None:
            v_sh = (sh - opp_sh) * side_sign  # always LEFT_SHOULDER - RIGHT_SHOULDER
            span_m = float(np.linalg.norm(v_sh))
            horiz_span = math.hypot(float(v_sh[0]), float(v_sh[2]))
            if 0.24 <= span_m <= 0.54 and horiz_span >= 0.16:
                raw_yaw = float(np.degrees(math.atan2(float(v_sh[2]), float(v_sh[0]))))
                hip_yaw_deg = float(np.clip(90.0 + raw_yaw, 0.0, 180.0))
            else:
                hip_yaw_deg = prev_hip_yaw_deg
        else:
            hip_yaw_deg = prev_hip_yaw_deg

    forearm_cm: float | None = None
    elbow_interior_deg: float | None = None
    elbow_flexion_deg: float | None = None
    wrist_angle_deg: float | None = None
    hand_open_deg: float | None = None

    if wr is not None:
        forearm_m = float(np.linalg.norm(wr - el))
        if forearm_m > 1e-4:
            forearm_cm = forearm_m * 100.0
            interior = calculate_3d_angle(sh, el, wr)
            elbow_interior_deg = interior
            elbow_flexion_deg = 180.0 - interior

        # Wrist angle & hand openness from 3D Pose hand landmarks (gated to sane palm distances).
        idx_pt = points_rig.get(index_idx) if index_idx is not None else None
        pnk_pt = points_rig.get(pinky_idx) if pinky_idx is not None else None
        thm_pt = points_rig.get(thumb_idx) if thumb_idx is not None else None

        idx_ok = idx_pt is not None and 0.03 <= float(np.linalg.norm(idx_pt - wr)) <= 0.24
        pnk_ok = pnk_pt is not None and 0.03 <= float(np.linalg.norm(pnk_pt - wr)) <= 0.24
        thm_ok = thm_pt is not None and 0.02 <= float(np.linalg.norm(thm_pt - wr)) <= 0.22

        # Pose-only fallback for wrist roll (0 deg palm down, 90 deg thumb up, 180 deg palm up)
        # when MediaPipe Hands is unavailable; stereo_pose overrides with MediaPipe Hands when active.
        if idx_ok and pnk_ok and forearm_m > 0.05:
            u_forearm = (wr - el) / forearm_m
            v_knuckle = idx_pt - pnk_pt
            k_perp = v_knuckle - float(np.dot(v_knuckle, u_forearm)) * u_forearm
            kn = float(np.linalg.norm(k_perp))
            if kn >= 0.015:
                k_hat = k_perp / kn
                world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
                vert = abs(float(np.dot(world_up, u_forearm)))
                if vert > 0.80:
                    blend = float(np.clip((vert - 0.80) / 0.18, 0.0, 1.0))
                    ref_up = (1.0 - blend) * world_up + blend * np.array([0.0, 0.0, 1.0], dtype=np.float64)
                else:
                    ref_up = world_up
                up_proj = ref_up - float(np.dot(ref_up, u_forearm)) * u_forearm
                un = float(np.linalg.norm(up_proj))
                if un >= 1e-6:
                    u_up = up_proj / un
                    u_inward = -side_sign * np.cross(u_up, u_forearm)
                    in_n = float(np.linalg.norm(u_inward))
                    if in_n >= 1e-6:
                        u_inward /= in_n
                        cos_r = float(np.dot(k_hat, u_inward))
                        sin_r = float(np.dot(k_hat, u_up))
                        if sin_r >= 0.0:
                            wrist_angle_deg = float(np.clip(np.degrees(math.atan2(sin_r, cos_r)), 0.0, 180.0))
                        else:
                            wrist_angle_deg = 0.0 if cos_r >= 0.0 else 180.0

        if idx_ok and thm_ok:
            palm_len = max(float(np.linalg.norm(idx_pt - wr)), 0.04)
            pinch_dist = float(np.linalg.norm(idx_pt - thm_pt))
            if pinch_dist <= 0.24:
                ratio = pinch_dist / palm_len
                # Map thumb-index spread ratio [0.25..0.90] linearly to [0°..180°].
                open_frac = float(np.clip((ratio - 0.25) / 0.65, 0.0, 1.0))
                hand_open_deg = open_frac * 180.0

    return ArmKinematics(
        upper_arm_cm=upper_m * 100.0,
        forearm_cm=forearm_cm,
        hip_yaw_deg=hip_yaw_deg,
        elevation_deg=elevation_deg,
        forward_pitch_deg=forward_pitch_deg,
        side_abduction_deg=side_abduction_deg,
        elbow_interior_deg=elbow_interior_deg,
        elbow_flexion_deg=elbow_flexion_deg,
        wrist_angle_deg=wrist_angle_deg,
        hand_open_deg=hand_open_deg,
    )


def _swap_lr(obs: dict[int, tuple]) -> dict[int, tuple]:
    return {_LR_SWAP.get(idx, idx): val for idx, val in obs.items()}


def _triangulate_hypothesis(
    calib: StereoCalibration,
    rig: RigFrame,
    left_xy: dict[int, tuple[float, float]],
    right_xy: dict[int, tuple[float, float]],
    left_vis: dict[int, float],
    right_vis: dict[int, float],
    min_vis: float = 0.25,
    max_sampson_px: float = MAX_SAMPSON_PX,
    max_reproj_px: float = MAX_REPROJ_PX,
) -> dict[int, StereoJoint]:
    ids = [
        i
        for i in UPPER_BODY_INDICES
        if i in left_xy
        and i in right_xy
        and left_vis.get(i, 1.0) >= min_vis
        and right_vis.get(i, 1.0) >= min_vis
    ]
    if not ids:
        return {}
    pts1 = np.array([left_xy[i] for i in ids], dtype=np.float64)
    pts2 = np.array([right_xy[i] for i in ids], dtype=np.float64)
    tri = triangulate_with_diagnostics(calib, pts1, pts2)
    pts_rig = rig.to_rig(tri.xyz_cam1)

    out: dict[int, StereoJoint] = {}
    for k, idx in enumerate(ids):
        if not bool(tri.in_front[k]):
            continue
        z_rig = float(pts_rig[k, 2])
        if not (DEPTH_MIN_M <= z_rig <= DEPTH_MAX_M):
            continue
        sampson = float(tri.sampson_px[k])
        e1 = float(tri.err1_px[k])
        e2 = float(tri.err2_px[k])
        samp_lim = TORSO_MAX_SAMPSON_PX if idx in TORSO_INDICES else max_sampson_px
        rep_lim = TORSO_MAX_REPROJ_PX if idx in TORSO_INDICES else max_reproj_px
        if sampson > samp_lim or max(e1, e2) > rep_lim:
            continue
        gap_mm = float(tri.ray_gap_m[k]) * 1000.0
        out[idx] = StereoJoint(
            index=idx,
            xyz_cam1=tri.xyz_cam1[k].copy(),
            xyz_rig=pts_rig[k].copy(),
            pt1=left_xy[idx],
            pt2=right_xy[idx],
            reproj1_px=e1,
            reproj2_px=e2,
            sampson_px=sampson,
            ray_gap_mm=gap_mm,
            ray_angle_deg=float(tri.ray_angle_deg[k]),
            quality=_quality_tier(gap_mm, 0.5 * (e1 + e2), sampson),
            source="stereo",
        )
    return out


def _hypothesis_metrics(joints: dict[int, StereoJoint]) -> tuple[int, int, int, float, bool]:
    plausible = 0
    implausible = 0
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):
        if a in joints and b in joints:
            dist = float(np.linalg.norm(joints[b].xyz_rig - joints[a].xyz_rig))
            if LIMB_MIN_M <= dist <= LIMB_MAX_M:
                plausible += 1
            else:
                implausible += 1
    shoulder_ok = False
    if 11 in joints and 12 in joints:
        span = float(np.linalg.norm(joints[12].xyz_rig - joints[11].xyz_rig))
        if SHOULDER_MIN_M <= span <= SHOULDER_MAX_M:
            plausible += 1
            shoulder_ok = True
        else:
            implausible += 1
    med_sampson = float(np.median([j.sampson_px for j in joints.values()])) if joints else 999.0
    return plausible, implausible, len(joints), med_sampson, shoulder_ok


def _prefer_swap(id_joints: dict[int, StereoJoint], sw_joints: dict[int, StereoJoint]) -> bool:
    """Only flip to L/R swapped pairing when swap is overwhelmingly better than identity."""
    id_p, id_imp, id_n, id_s, _ = _hypothesis_metrics(id_joints)
    sw_p, sw_imp, sw_n, sw_s, sw_sh_ok = _hypothesis_metrics(sw_joints)
    if not sw_sh_ok or sw_n < 4 or sw_imp > id_imp or sw_s > 12.0:
        return False
    if id_n < 2 and sw_p >= 2:
        return True
    if sw_p >= id_p + 3 and sw_s <= id_s + 2.0:
        return True
    return sw_p >= id_p + 2 and (sw_s + 10.0 < id_s)


def _ray_to_sphere_excess(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> float:
    """Distance by which a ray misses the sphere (0 if it intersects)."""
    oc = origin - center
    t = max(0.25, -float(np.dot(oc, direction)))
    dist_to_center = float(np.linalg.norm((origin + t * direction) - center))
    return max(0.0, dist_to_center - radius)


def _ray_weight(
    vis: float,
    limb_dir: np.ndarray | None,
    ray_dir: np.ndarray,
    pixel_crowding_px: float | None,
    ray_origin: np.ndarray,
    prev_pt: np.ndarray | None,
    prox_pt: np.ndarray | None = None,
    bone_len: float | None = None,
) -> float:
    """Compute per-camera ray trust weight combining visibility, foreshortening, and outlier checks."""
    if vis < 0.15:
        return 0.0
    w = ((vis - 0.10) / 0.90) ** 2

    # Foreshortening penalty: when the limb points straight along this camera's viewing ray,
    # depth/2D localization degrades while the other camera (31 deg away) sees a clean profile.
    if limb_dir is not None:
        align = abs(float(np.dot(limb_dir, ray_dir)))
        if align > 0.80:
            frac = min(1.0, (align - 0.80) / 0.20)
            w *= max(0.08, 1.0 - 0.92 * frac * frac)

    # 2D self-occlusion penalty when distal & proximal joints overlap in this camera's image.
    if pixel_crowding_px is not None and pixel_crowding_px < 55.0:
        w *= max(0.10, pixel_crowding_px / 55.0)

    # Bone-sphere consistency: if a camera's ray misses the reachable bone sphere by >5 cm,
    # that camera is misdetecting the joint.
    if prox_pt is not None and bone_len is not None:
        excess = _ray_to_sphere_excess(ray_origin, ray_dir, prox_pt, bone_len)
        if excess > 0.05:
            w *= 1.0 / (1.0 + ((excess - 0.05) / 0.05) ** 2)

    # Robust Cauchy downweighting if one camera suddenly glitches far away from the previous 3D point.
    if prev_pt is not None:
        dist_m = _point_to_ray_distance(prev_pt, ray_origin, ray_dir)
        if dist_m > 0.10:
            w *= 1.0 / (1.0 + ((dist_m - 0.10) / 0.07) ** 2)

    return float(w)


def check_charuco_board_3d(
    calib: StereoCalibration,
    rig: RigFrame,
    helper: CharucoHelper,
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    config: StereoConfig,
) -> BoardCheckResult | None:
    """Triangulate a visible ChArUco board and compare its 3D size against truth."""
    corners_l, ids_l = helper.detect(frame_l)
    corners_r, ids_r = helper.detect(frame_r)
    paired = paired_charuco_points(
        helper.board, corners_l, ids_l, corners_r, ids_r, min_corners=config.min_corners
    )
    if paired is None:
        return None
    obj_pts, img_l, img_r = paired
    tri = triangulate_with_diagnostics(calib, img_l, img_r)
    ok = tri.in_front & (tri.sampson_px < 10.0)
    if int(np.sum(ok)) < 4:
        return None

    obj_ok = obj_pts[ok]
    xyz_cam1 = tri.xyz_cam1[ok]
    xyz_rig = rig.to_rig(xyz_cam1)

    sq_m = float(config.square_length_m)
    measured_dists: list[float] = []
    n = len(obj_ok)
    for i in range(n):
        for j in range(i + 1, n):
            true_d = float(np.linalg.norm(obj_ok[i] - obj_ok[j]))
            if abs(true_d - sq_m) < 0.002:
                measured_dists.append(float(np.linalg.norm(xyz_cam1[i] - xyz_cam1[j])) * 1000.0)
    if not measured_dists:
        for i in range(n):
            for j in range(i + 1, n):
                true_d = float(np.linalg.norm(obj_ok[i] - obj_ok[j]))
                if true_d > 0.005:
                    meas_d = float(np.linalg.norm(xyz_cam1[i] - xyz_cam1[j]))
                    measured_dists.append((meas_d / true_d) * sq_m * 1000.0)

    expected_mm = sq_m * 1000.0
    measured_mm = float(np.median(measured_dists))
    scale_err_pct = 100.0 * (measured_mm - expected_mm) / expected_mm

    centroid = np.mean(xyz_rig, axis=0)
    _, _, vt = np.linalg.svd(xyz_rig - centroid)
    normal = vt[-1]
    plane_resid = (xyz_rig - centroid) @ normal
    planarity_rms_mm = float(np.sqrt(np.mean(plane_resid**2))) * 1000.0

    reproj = 0.5 * (tri.err1_px[ok] + tri.err2_px[ok])
    return BoardCheckResult(
        corners_rig=xyz_rig,
        expected_square_mm=expected_mm,
        measured_square_mm=measured_mm,
        scale_error_pct=scale_err_pct,
        planarity_rms_mm=planarity_rms_mm,
        mean_ray_gap_mm=float(np.mean(tri.ray_gap_m[ok])) * 1000.0,
        mean_reproj_px=float(np.mean(reproj)),
        depth_m=float(centroid[2]),
        n_corners=n,
    )


class StereoPoseTracker:
    """Multi-view stereo 3D pose tracker with rigid bone calibration and single-camera handover."""

    def __init__(
        self,
        calib: StereoCalibration,
        config: StereoConfig | None = None,
        *,
        hip_gain: float = 1.0,
        hip_sign: float = 1.0,
    ):
        self.calib = calib
        self.config = config or StereoConfig()
        self.rig = rig_frame_from_calibration(calib)
        self.filter_3d = OneEuro3D(min_cutoff=1.1, beta=1.8, d_cutoff=1.0)
        self.bone_model = BoneLengthModel()
        self._prev_rig: dict[int, np.ndarray] = {}
        self._prev_ts: dict[int, float] = {}
        self._prev_joint: dict[int, StereoJoint] = {}
        self._prev_source: dict[int, str] = {}
        self._prev_hip_yaw: dict[str, float | None] = {"left": None, "right": None}
        self._torso_yaw_filter = OneEuro1D(min_cutoff=1.2, beta=0.02, d_cutoff=1.0, max_jump=120.0, max_gap_s=0.6)
        self._torso_yaw_unwrapped: float | None = None
        self._torso_yaw_ts: float = -1e9
        self._torso_neutral_yaw_deg: float = 0.0
        self._torso_neutral_set: bool = False
        self._torso_spike_streak: int = 0
        self._hip_gain = float(hip_gain)
        self._hip_sign = float(hip_sign)
        self._charuco: CharucoHelper | None = None

    def zero_torso_yaw(self) -> float | None:
        """Set the current filtered torso yaw as neutral (90 deg on Joint.HIP)."""
        if self._torso_yaw_unwrapped is None:
            return None
        self._torso_neutral_yaw_deg = _wrap_deg_180(self._torso_yaw_unwrapped)
        self._torso_neutral_set = True
        self._torso_yaw_unwrapped = self._torso_neutral_yaw_deg
        self._torso_yaw_filter.reset()
        self._torso_spike_streak = 0
        return self._torso_neutral_yaw_deg

    def _update_torso_twist(
        self,
        joints: dict[int, StereoJoint],
        timestamp: float,
        *,
        use_filter: bool,
    ) -> float | None:
        """Compute clean 1:1 torso twist angle in [0, 180] deg (90 deg = neutral)."""
        ls = joints.get(LEFT_SHOULDER)
        rs = joints.get(RIGHT_SHOULDER)
        measured_yaw: float | None = None
        if (
            ls is not None
            and rs is not None
            and ls.source == "stereo"
            and rs.source == "stereo"
            and max(ls.sampson_px, rs.sampson_px) <= 15.0
        ):
            v_sh = ls.xyz_rig - rs.xyz_rig
            span_m = float(np.linalg.norm(v_sh))
            horiz_span = math.hypot(float(v_sh[0]), float(v_sh[2]))
            learned_span = self.bone_model.get(LEFT_SHOULDER, RIGHT_SHOULDER)
            if learned_span is not None:
                span_ok = abs(span_m - learned_span) <= 0.20 * learned_span
            else:
                span_ok = 0.26 <= span_m <= 0.50
            if span_ok and horiz_span >= 0.18:
                cand_yaw = float(np.degrees(math.atan2(float(v_sh[2]), float(v_sh[0]))))
                # Once zeroed, ignore >100 deg backward-facing or 180-deg L/R flip measurements
                # so the angle never crosses the +-180 deg branch cut or winds up by 360 deg.
                if not self._torso_neutral_set or abs(_wrap_deg_180(cand_yaw - self._torso_neutral_yaw_deg)) <= 100.0:
                    measured_yaw = cand_yaw

        if measured_yaw is not None:
            if self._torso_neutral_set:
                target = self._torso_neutral_yaw_deg + _wrap_deg_180(measured_yaw - self._torso_neutral_yaw_deg)
            elif self._torso_yaw_unwrapped is None:
                target = measured_yaw
            else:
                target = self._torso_yaw_unwrapped + _wrap_deg_180(measured_yaw - self._torso_yaw_unwrapped)

            if self._torso_yaw_unwrapped is None:
                unwrapped = target
                self._torso_spike_streak = 0
            else:
                dt = timestamp - self._torso_yaw_ts
                step = target - self._torso_yaw_unwrapped
                # Reject 1-4 frame occlusion / L-R flip spikes, but allow real re-acquisition
                # if the new orientation holds consistently for 5+ consecutive frames.
                max_streak = 5 if abs(step) > 90.0 else 2
                if dt < 0.40 and abs(step) > 35.0 and self._torso_spike_streak < max_streak:
                    self._torso_spike_streak += 1
                    unwrapped = self._torso_yaw_unwrapped
                else:
                    self._torso_spike_streak = 0
                    unwrapped = target

            if use_filter:
                filtered = self._torso_yaw_filter.filter(unwrapped, timestamp)
            else:
                filtered = unwrapped
            self._torso_yaw_unwrapped = filtered
            self._torso_yaw_ts = timestamp
        elif self._torso_yaw_unwrapped is None or (timestamp - self._torso_yaw_ts) > 0.60:
            return None

        delta = self._torso_yaw_unwrapped - self._torso_neutral_yaw_deg
        return float(np.clip(90.0 + self._hip_sign * self._hip_gain * delta, 0.0, 180.0))

    def _recent_prev(self, idx: int, timestamp: float, max_age_s: float = 0.45) -> np.ndarray | None:
        if idx not in self._prev_rig:
            return None
        if (timestamp - self._prev_ts.get(idx, -1e9)) > max_age_s:
            return None
        return self._prev_rig[idx]

    def _refine_shoulders(
        self,
        joints: dict[int, StereoJoint],
        left_xy: dict[int, tuple[float, float]],
        right_xy: dict[int, tuple[float, float]],
        left_vis: dict[int, float],
        right_vis: dict[int, float],
        timestamp: float,
    ) -> None:
        """Keep shoulder anchors rock-solid across brief single-camera drops or Sampson spikes."""
        for sh_idx in (LEFT_SHOULDER, RIGHT_SHOULDER):
            prev_sh = self._recent_prev(sh_idx, timestamp)
            rays: list[tuple[np.ndarray, np.ndarray, float]] = []
            cam_info: dict[int, tuple[np.ndarray, np.ndarray, float, tuple[float, float]]] = {}
            for cam_num, xy_map, vis_map in ((1, left_xy, left_vis), (2, right_xy, right_vis)):
                if sh_idx not in xy_map:
                    continue
                v = vis_map.get(sh_idx, 1.0)
                origin, direction = _camera_ray_in_rig(self.calib, self.rig, xy_map[sh_idx], cam_num)
                w = _ray_weight(v, None, direction, None, origin, prev_sh)
                if w > 0.0:
                    rays.append((origin, direction, w))
                    cam_info[cam_num] = (origin, direction, w, xy_map[sh_idx])

            existing = joints.get(sh_idx)
            if existing is not None and len(rays) >= 2:
                # Leave clean, tight-epipolar stereo untouched; only damp noisy/drifting frames.
                excess_samp = max(0.0, existing.sampson_px - 6.0)
                if excess_samp <= 0.0:
                    continue
                w_max = max(r[2] for r in rays)
                smooth_w = (0.015 + 0.008 * excess_samp) * w_max if prev_sh is not None else 0.0
                refined_rig = _solve_weighted_joint(
                    existing.xyz_rig,
                    rays,
                    None,
                    None,
                    bone_weight=0.0,
                    prev_pt=prev_sh,
                    smooth_weight=smooth_w,
                )
                if DEPTH_MIN_M <= float(refined_rig[2]) <= DEPTH_MAX_M:
                    refined_cam1 = self.rig.to_cam1(refined_rig)
                    e1 = float(np.linalg.norm(project_points(self.calib, refined_cam1, 1)[0] - np.array(existing.pt1)))
                    e2 = float(np.linalg.norm(project_points(self.calib, refined_cam1, 2)[0] - np.array(existing.pt2)))
                    joints[sh_idx] = StereoJoint(
                        index=sh_idx,
                        xyz_cam1=refined_cam1,
                        xyz_rig=refined_rig,
                        pt1=existing.pt1,
                        pt2=existing.pt2,
                        reproj1_px=e1,
                        reproj2_px=e2,
                        sampson_px=existing.sampson_px,
                        ray_gap_mm=existing.ray_gap_mm,
                        ray_angle_deg=existing.ray_angle_deg,
                        quality=existing.quality,
                        source="stereo",
                    )
            elif existing is None and cam_info and prev_sh is not None:
                best_cam = max(cam_info.keys(), key=lambda c: cam_info[c][2])
                origin, direction, best_w, best_px = cam_info[best_cam]
                best_vis = (left_vis if best_cam == 1 else right_vis).get(sh_idx, 0.0)
                if best_vis < 0.35 or best_w < 0.10:
                    continue
                t_along = float(np.dot(prev_sh - origin, direction))
                if t_along <= 0.25:
                    continue
                init_rig = origin + t_along * direction
                refined_rig = _solve_weighted_joint(
                    init_rig,
                    [(origin, direction, best_w)],
                    None,
                    None,
                    bone_weight=0.0,
                    prev_pt=prev_sh,
                    smooth_weight=0.20 * best_w,
                )
                if not (DEPTH_MIN_M <= float(refined_rig[2]) <= DEPTH_MAX_M):
                    continue
                if float(np.linalg.norm(refined_rig - prev_sh)) > 0.20:
                    continue
                refined_cam1 = self.rig.to_cam1(refined_rig)
                p1_proj = project_points(self.calib, refined_cam1, 1)[0]
                p2_proj = project_points(self.calib, refined_cam1, 2)[0]
                pt1 = best_px if best_cam == 1 else (float(p1_proj[0]), float(p1_proj[1]))
                pt2 = best_px if best_cam == 2 else (float(p2_proj[0]), float(p2_proj[1]))
                e1 = float(np.linalg.norm(p1_proj - np.array(pt1))) if best_cam == 1 else float("nan")
                e2 = float(np.linalg.norm(p2_proj - np.array(pt2))) if best_cam == 2 else float("nan")
                joints[sh_idx] = StereoJoint(
                    index=sh_idx,
                    xyz_cam1=refined_cam1,
                    xyz_rig=refined_rig,
                    pt1=pt1,
                    pt2=pt2,
                    reproj1_px=e1,
                    reproj2_px=e2,
                    sampson_px=float("nan"),
                    ray_gap_mm=float("nan"),
                    ray_angle_deg=float("nan"),
                    quality="fair",
                    source="ray+bone",
                )

    def _refine_arm_chain(
        self,
        joints: dict[int, StereoJoint],
        left_xy: dict[int, tuple[float, float]],
        right_xy: dict[int, tuple[float, float]],
        left_vis: dict[int, float],
        right_vis: dict[int, float],
        timestamp: float,
        shoulder_idx: int,
        elbow_idx: int,
        wrist_idx: int,
    ) -> None:
        """Unified continuous weighted ray + rigid bone solver with seamless 1-camera handover."""
        for prox_idx, dist_idx in ((shoulder_idx, elbow_idx), (elbow_idx, wrist_idx)):
            prox_joint = joints.get(prox_idx)
            if prox_joint is None:
                continue
            prox_rig = prox_joint.xyz_rig
            bone_len = self.bone_model.get(prox_idx, dist_idx)
            prev_dist = self._recent_prev(dist_idx, timestamp)

            limb_dir: np.ndarray | None = None
            if prev_dist is not None:
                vec = prev_dist - prox_rig
                nrm = float(np.linalg.norm(vec))
                if nrm > 0.05:
                    limb_dir = vec / nrm

            # Build per-camera rays & continuous view-quality weights.
            rays: list[tuple[np.ndarray, np.ndarray, float]] = []
            cam_info: dict[int, tuple[np.ndarray, np.ndarray, float, tuple[float, float]]] = {}
            for cam_num, xy_map, vis_map in ((1, left_xy, left_vis), (2, right_xy, right_vis)):
                if dist_idx not in xy_map:
                    continue
                v = vis_map.get(dist_idx, 1.0)
                origin, direction = _camera_ray_in_rig(self.calib, self.rig, xy_map[dist_idx], cam_num)
                crowd = None
                if prox_idx in xy_map:
                    crowd = float(np.linalg.norm(np.array(xy_map[dist_idx]) - np.array(xy_map[prox_idx])))
                w = _ray_weight(v, limb_dir, direction, crowd, origin, prev_dist, prox_rig, bone_len)
                if w > 0.0:
                    rays.append((origin, direction, w))
                    cam_info[cam_num] = (origin, direction, w, xy_map[dist_idx])

            if not rays:
                continue

            existing = joints.get(dist_idx)
            w_max = max(r[2] for r in rays)
            w_min = min(r[2] for r in rays) if len(rays) >= 2 else 0.0
            ratio = w_min / max(w_max, 1e-6)
            asymmetry = 1.0 - ratio

            # Strong bone-length stiffness suppresses depth pumping from camera skew and foreshortening
            # while adding zero resistance to genuine rotational motion around the proximal joint.
            bone_w = (2.5 + 3.5 * asymmetry) * w_max if bone_len is not None else 0.0
            smooth_w = (0.025 + 0.03 * asymmetry) * w_max if prev_dist is not None else 0.0

            # Unified initialization: use dual-view triangulation when balanced, or ray-sphere root
            # when one camera is dropped/heavily degraded, then solve with ALL active weighted rays.
            use_handover_init = (existing is None) or (bone_len is not None and ratio < 0.18)

            if not use_handover_init and existing is not None:
                init_pt = existing.xyz_rig
                active_rays = rays
            else:
                if bone_len is None or not cam_info:
                    continue
                best_cam = max(cam_info.keys(), key=lambda c: cam_info[c][2])
                origin, direction, best_w, best_px = cam_info[best_cam]
                best_vis = (left_vis if best_cam == 1 else right_vis).get(dist_idx, 0.0)
                if best_vis < 0.30 or best_w < 0.08:
                    continue
                candidates = _ray_sphere_candidates(origin, direction, prox_rig, bone_len)
                if not candidates:
                    continue

                other_cam = 2 if best_cam == 1 else 1
                other_info = cam_info.get(other_cam)

                def cand_cost(pt: np.ndarray) -> float:
                    cost = 0.0
                    if prev_dist is not None:
                        cost += float(np.linalg.norm(pt - prev_dist))
                    if other_info is not None:
                        cost += 0.35 * _point_to_ray_distance(pt, other_info[0], other_info[1])
                    if prev_dist is None and other_info is None:
                        cost += 0.1 * abs(float(pt[2]) - float(prox_rig[2]))
                    return cost

                init_pt = min(candidates, key=cand_cost)
                if prev_dist is not None and float(np.linalg.norm(init_pt - prev_dist)) > 0.38:
                    continue
                # Keep both rays with their continuous weights (other ray is already smoothly downweighted)
                # unless stereo triangulation was completely rejected as an epipolar outlier.
                active_rays = rays if existing is not None else [(origin, direction, best_w)]

            refined_rig = _solve_weighted_joint(
                init_pt,
                active_rays,
                prox_rig,
                bone_len,
                bone_w,
                prev_dist,
                smooth_w,
            )
            if not (DEPTH_MIN_M <= float(refined_rig[2]) <= DEPTH_MAX_M):
                continue

            # Hysteresis on source badge ("stereo" vs "ray+bone") so HUD colors don't rapidly strobe.
            prev_src = self._prev_source.get(dist_idx, "stereo")
            if existing is None or len(rays) < 2:
                source = "ray+bone"
            elif prev_src == "ray+bone":
                source = "stereo" if ratio >= 0.28 else "ray+bone"
            else:
                source = "ray+bone" if ratio < 0.14 else "stereo"

            refined_cam1 = self.rig.to_cam1(refined_rig)
            p1_proj = project_points(self.calib, refined_cam1, 1)[0]
            p2_proj = project_points(self.calib, refined_cam1, 2)[0]
            if existing is not None:
                pt1, pt2 = existing.pt1, existing.pt2
                e1 = float(np.linalg.norm(p1_proj - np.array(pt1)))
                e2 = float(np.linalg.norm(p2_proj - np.array(pt2)))
                samp = existing.sampson_px
                gap_mm = existing.ray_gap_mm
                ray_ang = existing.ray_angle_deg
                qual = existing.quality
            else:
                best_cam = max(cam_info.keys(), key=lambda c: cam_info[c][2])
                best_px = cam_info[best_cam][3]
                pt1 = best_px if best_cam == 1 else (float(p1_proj[0]), float(p1_proj[1]))
                pt2 = best_px if best_cam == 2 else (float(p2_proj[0]), float(p2_proj[1]))
                e1 = float(np.linalg.norm(p1_proj - np.array(pt1))) if best_cam == 1 else float("nan")
                e2 = float(np.linalg.norm(p2_proj - np.array(pt2))) if best_cam == 2 else float("nan")
                samp = float("nan")
                gap_mm = float("nan")
                ray_ang = float("nan")
                qual = "fair"

            joints[dist_idx] = StereoJoint(
                index=dist_idx,
                xyz_cam1=refined_cam1,
                xyz_rig=refined_rig,
                pt1=pt1,
                pt2=pt2,
                reproj1_px=e1,
                reproj2_px=e2,
                sampson_px=samp,
                ray_gap_mm=gap_mm,
                ray_angle_deg=ray_ang,
                quality=qual,
                source=source,
            )

    def _fill_brief_dropouts(self, solved: dict[int, StereoJoint], timestamp: float) -> None:
        """Rigidly coast briefly missing joints (<= HOLD_MAX_AGE_S) to stop 1-3 frame blinks."""
        # Process in proximal-to-distal order so child joints inherit parent motion.
        order = (
            LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE, LEFT_EAR, RIGHT_EAR,
            LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
            LEFT_INDEX, LEFT_PINKY, LEFT_THUMB, RIGHT_INDEX, RIGHT_PINKY, RIGHT_THUMB,
        )
        for idx in order:
            if idx in solved:
                continue
            prev_j = self._prev_joint.get(idx)
            prev_t = self._prev_ts.get(idx, -1e9)
            if prev_j is None or (timestamp - prev_t) > HOLD_MAX_AGE_S:
                continue
            held_rig = prev_j.xyz_rig.copy()
            parent_idx = _PARENT_JOINT.get(idx)
            if parent_idx is not None and parent_idx in solved and parent_idx in self._prev_rig:
                # Translate rigidly with the parent joint so the limb stays attached.
                held_rig = held_rig + (solved[parent_idx].xyz_rig - self._prev_rig[parent_idx])
            if not (DEPTH_MIN_M <= float(held_rig[2]) <= DEPTH_MAX_M):
                continue
            held_cam1 = self.rig.to_cam1(held_rig)
            p1_proj = project_points(self.calib, held_cam1, 1)[0]
            p2_proj = project_points(self.calib, held_cam1, 2)[0]
            solved[idx] = StereoJoint(
                index=idx,
                xyz_cam1=held_cam1,
                xyz_rig=held_rig,
                pt1=(float(p1_proj[0]), float(p1_proj[1])),
                pt2=(float(p2_proj[0]), float(p2_proj[1])),
                reproj1_px=float("nan"),
                reproj2_px=float("nan"),
                sampson_px=float("nan"),
                ray_gap_mm=float("nan"),
                ray_angle_deg=float("nan"),
                quality="fair",
                source="hold",
            )

    def process(
        self,
        left_obs: dict[int, tuple],
        right_obs: dict[int, tuple],
        timestamp: float,
        *,
        use_filter: bool = True,
        frame_l: np.ndarray | None = None,
        frame_r: np.ndarray | None = None,
        check_board: bool = False,
    ) -> StereoFrame:
        left_xy, left_vis = _unpack_obs(left_obs)
        right_xy_id, right_vis_id = _unpack_obs(right_obs)
        right_xy_sw, right_vis_sw = _unpack_obs(_swap_lr(right_obs))

        id_joints = _triangulate_hypothesis(self.calib, self.rig, left_xy, right_xy_id, left_vis, right_vis_id)
        sw_joints = _triangulate_hypothesis(self.calib, self.rig, left_xy, right_xy_sw, left_vis, right_vis_sw)

        if _prefer_swap(id_joints, sw_joints):
            raw_joints = sw_joints
            right_xy, right_vis = right_xy_sw, right_vis_sw
            hypothesis = "swap"
        else:
            raw_joints = id_joints
            right_xy, right_vis = right_xy_id, right_vis_id
            hypothesis = "identity"

        # Learn rigid bone lengths only from clean, unoccluded dual-view stereo frames.
        self.bone_model.update(raw_joints, left_vis, right_vis)

        # Refine shoulders, then arm chains with continuous weighted rays + rigid bones.
        solved = dict(raw_joints)
        self._refine_shoulders(solved, left_xy, right_xy, left_vis, right_vis, timestamp)
        self._refine_arm_chain(
            solved, left_xy, right_xy, left_vis, right_vis, timestamp, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        )
        self._refine_arm_chain(
            solved, left_xy, right_xy, left_vis, right_vis, timestamp, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        )
        self._fill_brief_dropouts(solved, timestamp)

        if use_filter:
            joints: dict[int, StereoJoint] = {}
            for idx, j in solved.items():
                filt_rig = self.filter_3d.filter_point(idx, j.xyz_rig, timestamp)
                filt_cam1 = self.rig.to_cam1(filt_rig)
                joints[idx] = StereoJoint(
                    index=j.index,
                    xyz_cam1=filt_cam1,
                    xyz_rig=filt_rig,
                    pt1=j.pt1,
                    pt2=j.pt2,
                    reproj1_px=j.reproj1_px,
                    reproj2_px=j.reproj2_px,
                    sampson_px=j.sampson_px,
                    ray_gap_mm=j.ray_gap_mm,
                    ray_angle_deg=j.ray_angle_deg,
                    quality=j.quality,
                    source=j.source,
                )
        else:
            joints = solved

        for idx, j in joints.items():
            self._prev_rig[idx] = j.xyz_rig.copy()
            self._prev_joint[idx] = j
            if j.source != "hold":
                self._prev_ts[idx] = timestamp
                self._prev_source[idx] = j.source

        pts_rig = {idx: j.xyz_rig for idx, j in joints.items()}
        torso_hip_deg = self._update_torso_twist(joints, timestamp, use_filter=use_filter)
        left_arm = compute_arm_kinematics(
            pts_rig,
            LEFT_SHOULDER,
            LEFT_ELBOW,
            LEFT_WRIST,
            side_sign=1.0,
            opp_shoulder_idx=RIGHT_SHOULDER,
            index_idx=LEFT_INDEX,
            pinky_idx=LEFT_PINKY,
            thumb_idx=LEFT_THUMB,
            hip_yaw_override_deg=torso_hip_deg,
            use_hip_yaw_override=True,
            prev_hip_yaw_deg=self._prev_hip_yaw["left"],
        )
        right_arm = compute_arm_kinematics(
            pts_rig,
            RIGHT_SHOULDER,
            RIGHT_ELBOW,
            RIGHT_WRIST,
            side_sign=-1.0,
            opp_shoulder_idx=LEFT_SHOULDER,
            index_idx=RIGHT_INDEX,
            pinky_idx=RIGHT_PINKY,
            thumb_idx=RIGHT_THUMB,
            hip_yaw_override_deg=torso_hip_deg,
            use_hip_yaw_override=True,
            prev_hip_yaw_deg=self._prev_hip_yaw["right"],
        )
        if left_arm.hip_yaw_deg is not None:
            self._prev_hip_yaw["left"] = left_arm.hip_yaw_deg
        if right_arm.hip_yaw_deg is not None:
            self._prev_hip_yaw["right"] = right_arm.hip_yaw_deg

        stereo_only = [j for j in raw_joints.values() if j.source == "stereo"]
        if stereo_only:
            med_gap = float(np.median([j.ray_gap_mm for j in stereo_only]))
            med_rep = float(np.median([j.mean_reproj_px for j in stereo_only]))
            med_samp = float(np.median([j.sampson_px for j in stereo_only]))
        else:
            med_gap = float("nan")
            med_rep = float("nan")
            med_samp = float("nan")

        board_res: BoardCheckResult | None = None
        if check_board and frame_l is not None and frame_r is not None:
            if self._charuco is None:
                self._charuco = CharucoHelper(self.config)
            board_res = check_charuco_board_3d(
                self.calib, self.rig, self._charuco, frame_l, frame_r, self.config
            )

        return StereoFrame(
            joints=joints,
            raw_joints=raw_joints,
            hypothesis=hypothesis,
            left_arm=left_arm,
            right_arm=right_arm,
            median_ray_gap_mm=med_gap,
            median_reproj_px=med_rep,
            median_sampson_px=med_samp,
            board_check=board_res,
        )
