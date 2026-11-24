"""Dedicated MediaPipe Hands ROI tracker for crisp 3D wrist roll and hand open/close."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from stereo.filters import OneEuro1D
from stereo.geometry import RigFrame, StereoCalibration
from stereo.kinematics import calculate_3d_angle

# MediaPipe Hand landmark indices
HAND_WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

_FINGERS = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)

# MediaPipe Pose indices used to locate the active hand ROI
_POSE_ARM_INDICES = {
    "left": (13, 15, 19, 17, 21),   # elbow, wrist, index, pinky, thumb
    "right": (14, 16, 20, 18, 22),
}

HAND_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


@dataclass(frozen=True)
class HandObservation:
    camera_num: int  # 1 (cam0) or 2 (cam1)
    world_pts: np.ndarray  # (21, 3) meters in camera-aligned frame (+X right, +Y down, +Z away)
    image_xy: np.ndarray   # (21, 2) full-frame pixel coordinates
    score: float


@dataclass(frozen=True)
class HandMetrics:
    wrist_roll_deg: float | None = None
    hand_open_deg: float | None = None
    cam0_xy: np.ndarray | None = None
    cam1_xy: np.ndarray | None = None
    source: str = "none"  # "hands-2", "hands-1", "hold", or "none"


def compute_hand_roi(
    pose_obs: dict[int, tuple[float, float, float]],
    frame_shape: tuple[int, int],
    *,
    arm: str = "right",
    min_vis: float = 0.25,
) -> tuple[int, int, int, int] | None:
    """Compute a padded square pixel crop (x0, y0, x1, y1) around the active hand."""
    h, w = frame_shape
    el_idx, wr_idx, idx_idx, pnk_idx, thm_idx = _POSE_ARM_INDICES["left" if arm == "left" else "right"]
    wr = pose_obs.get(wr_idx)
    if wr is None or wr[2] < min_vis:
        return None
    wx, wy = float(wr[0]), float(wr[1])
    if not (0.0 <= wx <= w and 0.0 <= wy <= h):
        return None

    # Estimate palm center slightly past wrist using visible Pose hand hints or forearm direction.
    tip_pts = [
        (float(pose_obs[i][0]), float(pose_obs[i][1]))
        for i in (idx_idx, pnk_idx, thm_idx)
        if i in pose_obs and pose_obs[i][2] >= min_vis
    ]
    el = pose_obs.get(el_idx)
    forearm_px = 0.0
    if el is not None and el[2] >= min_vis:
        forearm_px = math.hypot(wx - float(el[0]), wy - float(el[1]))

    palm_px = 0.0
    if tip_pts:
        mx = float(np.mean([p[0] for p in tip_pts]))
        my = float(np.mean([p[1] for p in tip_pts]))
        cx = 0.35 * wx + 0.65 * mx
        cy = 0.35 * wy + 0.65 * my
        palm_px = max(math.hypot(p[0] - wx, p[1] - wy) for p in tip_pts)
    elif el is not None and el[2] >= min_vis and forearm_px >= 20.0:
        cx = wx + 0.28 * (wx - float(el[0]))
        cy = wy + 0.28 * (wy - float(el[1]))
    else:
        cx, cy = wx, wy

    half = int(round(max(130.0, 0.72 * forearm_px, 2.4 * palm_px)))
    half = max(120, min(260, half))

    x0 = max(0, int(round(cx)) - half)
    y0 = max(0, int(round(cy)) - half)
    x1 = min(w, int(round(cx)) + half)
    y1 = min(h, int(round(cy)) + half)
    if (x1 - x0) < 48 or (y1 - y0) < 48:
        return None
    return (x0, y0, x1, y1)


def extract_hand_in_roi(
    hands_model: Any,
    frame_bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    camera_num: int,
) -> HandObservation | None:
    """Run MediaPipe Hands on a cropped ROI and map landmarks back to full-frame pixels."""
    if roi is None:
        return None
    x0, y0, x1, y1 = roi
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands_model.process(rgb)
    if not res.multi_hand_landmarks or not res.multi_hand_world_landmarks:
        return None

    img_lms = res.multi_hand_landmarks[0].landmark
    wld_lms = res.multi_hand_world_landmarks[0].landmark
    cw = float(x1 - x0)
    ch = float(y1 - y0)

    image_xy = np.array(
        [[x0 + float(lm.x) * cw, y0 + float(lm.y) * ch] for lm in img_lms],
        dtype=np.float64,
    )
    world_pts = np.array(
        [[float(lm.x), float(lm.y), float(lm.z)] for lm in wld_lms],
        dtype=np.float64,
    )

    conf = 1.0
    if res.multi_handedness:
        try:
            conf = float(res.multi_handedness[0].classification[0].score)
        except Exception:
            conf = 1.0

    # Larger palm in pixels gives a cleaner visual estimate; weight gently by apparent palm size.
    palm_span_px = float(np.linalg.norm(image_xy[MIDDLE_MCP] - image_xy[HAND_WRIST]))
    score = max(0.05, conf * min(2.0, palm_span_px / 60.0))
    return HandObservation(
        camera_num=camera_num,
        world_pts=world_pts,
        image_xy=image_xy,
        score=score,
    )


def compute_hand_open_deg(world_pts: np.ndarray) -> float:
    """Map 21 3D hand world landmarks to gripper openness: 0 deg closed -> 180 deg wide open."""
    wrist = world_pts[HAND_WRIST]
    finger_scores: list[float] = []
    for mcp_i, pip_i, tip_i in _FINGERS:
        mcp = world_pts[mcp_i]
        pip = world_pts[pip_i]
        tip = world_pts[tip_i]
        mcp_d = max(float(np.linalg.norm(mcp - wrist)), 0.03)
        tip_d = float(np.linalg.norm(tip - wrist))
        ratio = tip_d / mcp_d
        # Relaxed open fingers sit near ratio ~1.60-1.65 and PIP ~145-155 deg;
        # saturate there so the gripper opens fully without hyper-extending fingers.
        ext = float(np.clip((ratio - 1.05) / 0.58, 0.0, 1.0))

        pip_deg = calculate_3d_angle(mcp, pip, tip)
        ang = float(np.clip((pip_deg - 80.0) / 70.0, 0.0, 1.0))
        finger_scores.append(0.55 * ext + 0.45 * ang)

    four_finger = float(np.median(finger_scores))

    palm_scale = max(float(np.linalg.norm(world_pts[MIDDLE_MCP] - wrist)), 0.04)
    pinch_ratio = float(np.linalg.norm(world_pts[THUMB_TIP] - world_pts[INDEX_TIP])) / palm_scale
    pinch_open = float(np.clip((pinch_ratio - 0.30) / 0.65, 0.0, 1.0))

    # Both a full fist and a deliberate thumb-index pinch close smoothly to 0 deg;
    # opening the hand biases toward full 180 deg travel for the mechanical gripper.
    base = min(four_finger, 0.75 * four_finger + 0.25 * pinch_open)
    w_pinch = float(np.clip((0.40 - pinch_open) / 0.25, 0.0, 1.0))
    raw_open = (1.0 - w_pinch) * base + w_pinch * min(base, pinch_open)
    # Boost mid-to-open range and deadzone the very bottom so closed stays 0 deg
    # while a comfortably open hand hits a solid 180 deg.
    rescaled = float(np.clip((raw_open - 0.05) / 0.78, 0.0, 1.0)) ** 0.75
    return float(np.clip(rescaled * 180.0, 0.0, 180.0))


def compute_wrist_roll_deg(
    world_pts: np.ndarray,
    R_cam_to_rig: np.ndarray,
    *,
    forearm_dir_rig: np.ndarray | None,
    side_sign: float,
) -> float | None:
    """Compute wrist pronation/supination roll around the forearm axis in [0, 180] deg.

    Uses the rigid knuckle line (pinky MCP -> index MCP) so opening or closing the
    fingers does not couple into wrist roll.

    Convention (symmetric for both arms):
    - 0 deg: palm facing down (pronated)
    - 90 deg: thumb facing up (neutral / handshake)
    - 180 deg: palm facing up (supinated)
    """
    pts_rig = world_pts @ R_cam_to_rig.T
    v_knuckle = pts_rig[INDEX_MCP] - pts_rig[PINKY_MCP]
    if float(np.linalg.norm(v_knuckle)) < 0.015:
        return None

    if forearm_dir_rig is not None and float(np.linalg.norm(forearm_dir_rig)) >= 0.05:
        u_forearm = forearm_dir_rig / float(np.linalg.norm(forearm_dir_rig))
    else:
        v_palm = pts_rig[MIDDLE_MCP] - pts_rig[HAND_WRIST]
        vn = float(np.linalg.norm(v_palm))
        if vn < 0.02:
            return None
        u_forearm = v_palm / vn

    # Project knuckle line perpendicular to the forearm axis.
    k_perp = v_knuckle - float(np.dot(v_knuckle, u_forearm)) * u_forearm
    kn = float(np.linalg.norm(k_perp))
    if kn < 0.01:
        return None
    k_hat = k_perp / kn

    # Build singularity-free (u_inward, u_up) basis perpendicular to u_forearm.
    # In rig coordinates (+X person's left, +Y down, +Z away from cameras):
    # world_up = (0, -1, 0). Blend smoothly toward +Z if forearm points nearly vertical.
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    vert = abs(float(np.dot(world_up, u_forearm)))
    if vert > 0.80:
        blend = float(np.clip((vert - 0.80) / 0.18, 0.0, 1.0))
        ref_up = (1.0 - blend) * world_up + blend * np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        ref_up = world_up

    up_proj = ref_up - float(np.dot(ref_up, u_forearm)) * u_forearm
    un = float(np.linalg.norm(up_proj))
    if un < 1e-6:
        return None
    u_up = up_proj / un

    # For right arm (side_sign=-1): cross(u_up, u_forearm) points inward (+X when forearm is -Z).
    # For left arm (side_sign=+1): -cross(u_up, u_forearm) points inward (-X when forearm is -Z).
    u_inward = -side_sign * np.cross(u_up, u_forearm)
    in_n = float(np.linalg.norm(u_inward))
    if in_n < 1e-6:
        return None
    u_inward /= in_n

    cos_r = float(np.dot(k_hat, u_inward))
    sin_r = float(np.dot(k_hat, u_up))
    if sin_r >= 0.0:
        return float(np.clip(np.degrees(math.atan2(sin_r, cos_r)), 0.0, 180.0))
    # Clamp cleanly to nearest endpoint when slightly over-pronated or over-supinated.
    return 0.0 if cos_r >= 0.0 else 180.0


class StereoHandTracker:
    """Fuses single- or dual-camera MediaPipe Hands detections with smooth filtering."""

    def __init__(
        self,
        calib: StereoCalibration,
        rig: RigFrame,
        *,
        hold_max_s: float = 0.35,
    ) -> None:
        self.R_cam1_to_rig = np.asarray(rig.R_cam1_to_rig, dtype=np.float64)
        self.R_cam2_to_rig = self.R_cam1_to_rig @ np.asarray(calib.R, dtype=np.float64).T
        self.hold_max_s = float(hold_max_s)

        self._roll_filter = OneEuro1D(min_cutoff=1.8, beta=0.06, d_cutoff=1.2)
        self._open_filter = OneEuro1D(min_cutoff=2.2, beta=0.08, d_cutoff=1.2)
        self._last_roll: float | None = None
        self._last_open: float | None = None
        self._last_ts: float = -1e9

    def update(
        self,
        obs_cam0: HandObservation | None,
        obs_cam1: HandObservation | None,
        *,
        forearm_dir_rig: np.ndarray | None,
        arm: str,
        timestamp: float,
        use_filter: bool = True,
    ) -> HandMetrics:
        side_sign = 1.0 if arm == "left" else -1.0
        rolls: list[tuple[float, float]] = []
        opens: list[tuple[float, float]] = []

        for obs, R_c2r in ((obs_cam0, self.R_cam1_to_rig), (obs_cam1, self.R_cam2_to_rig)):
            if obs is None:
                continue
            r_deg = compute_wrist_roll_deg(
                obs.world_pts,
                R_c2r,
                forearm_dir_rig=forearm_dir_rig,
                side_sign=side_sign,
            )
            if r_deg is not None:
                rolls.append((r_deg, obs.score))
            opens.append((compute_hand_open_deg(obs.world_pts), obs.score))

        n_views = (1 if obs_cam0 is not None else 0) + (1 if obs_cam1 is not None else 0)
        if n_views > 0:
            raw_roll = (
                sum(v * w for v, w in rolls) / sum(w for _, w in rolls)
                if rolls
                else None
            )
            raw_open = (
                sum(v * w for v, w in opens) / sum(w for _, w in opens)
                if opens
                else None
            )
            if use_filter:
                roll_out = (
                    float(np.clip(self._roll_filter.filter(raw_roll, timestamp), 0.0, 180.0))
                    if raw_roll is not None
                    else self._last_roll
                )
                open_out = (
                    float(np.clip(self._open_filter.filter(raw_open, timestamp), 0.0, 180.0))
                    if raw_open is not None
                    else self._last_open
                )
            else:
                roll_out = raw_roll
                open_out = raw_open

            if roll_out is not None:
                self._last_roll = roll_out
            if open_out is not None:
                self._last_open = open_out
            self._last_ts = timestamp
            return HandMetrics(
                wrist_roll_deg=roll_out,
                hand_open_deg=open_out,
                cam0_xy=None if obs_cam0 is None else obs_cam0.image_xy,
                cam1_xy=None if obs_cam1 is None else obs_cam1.image_xy,
                source="hands-2" if n_views == 2 else "hands-1",
            )

        if (timestamp - self._last_ts) <= self.hold_max_s and (
            self._last_roll is not None or self._last_open is not None
        ):
            return HandMetrics(
                wrist_roll_deg=self._last_roll,
                hand_open_deg=self._last_open,
                source="hold",
            )

        self._roll_filter.reset()
        self._open_filter.reset()
        return HandMetrics()
