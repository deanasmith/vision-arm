import math
import unittest

import numpy as np

from stereo.geometry import parallel_camera_calibration, rig_frame_from_calibration
from stereo.hand_tracker import (
    HAND_WRIST,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    HandObservation,
    StereoHandTracker,
    compute_hand_open_deg,
    compute_hand_roi,
    compute_wrist_roll_deg,
)


def _make_hand_world(
    *,
    curl: float = 0.0,
    pinch: float = 0.0,
    roll_deg: float = 0.0,
    side_sign: float = -1.0,
    forearm_dir: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Synthesize 21 3D hand world landmarks with exact roll_deg, curl [0..1], and pinch [0..1].

    forearm_dir defaults to (0, 0, -1) (pointing straight toward camera).
    roll_deg: 0 = palm down (pronated), 90 = thumb up, 180 = palm up (supinated).
    """
    u_f = np.array([0.0, 0.0, -1.0] if forearm_dir is None else forearm_dir, dtype=np.float64)
    u_f /= np.linalg.norm(u_f)
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    u_up = world_up - np.dot(world_up, u_f) * u_f
    u_up /= np.linalg.norm(u_up)
    u_in = -side_sign * np.cross(u_up, u_f)
    u_in /= np.linalg.norm(u_in)

    rad = math.radians(roll_deg)
    # Knuckle axis points from pinky MCP -> index MCP
    k_dir = math.cos(rad) * u_in + math.sin(rad) * u_up
    # Palm normal (points out of palm, away from back of hand)
    palm_down = -math.sin(rad) * u_in + math.cos(rad) * u_up
    # Wait: at roll=0 (palm down), palm faces -u_up (downward); at roll=90 (thumb up), palm faces +u_in (inward).
    palm_out = -math.cos(rad) * u_up + math.sin(rad) * u_in

    pts = np.zeros((21, 3), dtype=np.float64)
    pts[HAND_WRIST] = np.zeros(3)

    # 4 fingers spaced evenly across the knuckle line (index=+0.035 .. pinky=-0.035)
    finger_defs = [
        (INDEX_MCP, INDEX_PIP, INDEX_TIP, +0.035, 0.090),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, +0.012, 0.095),
        (RING_MCP, RING_PIP, RING_TIP, -0.012, 0.090),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP, -0.035, 0.080),
    ]
    for mcp_i, pip_i, tip_i, lateral, palm_len in finger_defs:
        mcp = (palm_len * u_f + lateral * k_dir) * scale
        if curl <= 0.0:
            pip = mcp + 0.045 * scale * u_f
            tip = mcp + 0.090 * scale * u_f
        elif curl >= 1.0:
            # Fully curled fist: PIP slightly forward, TIP folded back toward palm
            pip = mcp + 0.035 * scale * u_f + 0.020 * scale * palm_out
            tip = mcp - 0.015 * scale * u_f + 0.040 * scale * palm_out
        else:
            # Interpolate smoothly between open and fist
            pip = mcp + ((1.0 - curl) * 0.045 + curl * 0.035) * scale * u_f + (curl * 0.020) * scale * palm_out
            tip = mcp + ((1.0 - curl) * 0.090 - curl * 0.015) * scale * u_f + (curl * 0.040) * scale * palm_out
        pts[mcp_i] = mcp
        pts[pip_i] = pip
        pts[tip_i] = tip

    # Thumb tip: far from index tip when open (pinch=0), touching index tip when pinched (pinch=1)
    open_thumb = (0.11 * u_f + 0.13 * k_dir) * scale
    pts[THUMB_TIP] = (1.0 - pinch) * open_thumb + pinch * (pts[INDEX_TIP] + 0.008 * scale * k_dir)
    return pts


class HandTrackerTests(unittest.TestCase):
    def test_hand_open_close_and_pinch_and_scale_invariance(self):
        open_deg = compute_hand_open_deg(_make_hand_world(curl=0.0, pinch=0.0))
        relaxed_open_deg = compute_hand_open_deg(_make_hand_world(curl=0.22, pinch=0.20))
        fist_deg = compute_hand_open_deg(_make_hand_world(curl=1.0, pinch=0.8))
        pinch_deg = compute_hand_open_deg(_make_hand_world(curl=0.1, pinch=1.0))
        half_deg = compute_hand_open_deg(_make_hand_world(curl=0.55, pinch=0.40))
        small_open = compute_hand_open_deg(_make_hand_world(curl=0.0, pinch=0.0, scale=0.6))

        self.assertGreater(open_deg, 178.0)
        self.assertGreater(relaxed_open_deg, 160.0)
        self.assertLess(fist_deg, 10.0)
        self.assertLess(pinch_deg, 10.0)
        self.assertGreater(half_deg, 45.0)
        self.assertLess(half_deg, 165.0)
        self.assertAlmostEqual(open_deg, small_open, places=3)

    def test_wrist_roll_right_and_left_arms_across_orientations(self):
        eye = np.eye(3)
        for side_sign in (-1.0, 1.0):
            for forearm in (np.array([0.0, 0.0, -1.0]), np.array([side_sign, 0.0, 0.0])):
                for target_roll in (0.0, 45.0, 90.0, 135.0, 180.0):
                    for curl in (0.0, 1.0):
                        pts = _make_hand_world(
                            curl=curl,
                            roll_deg=target_roll,
                            side_sign=side_sign,
                            forearm_dir=forearm,
                        )
                        got = compute_wrist_roll_deg(
                            pts,
                            eye,
                            forearm_dir_rig=forearm,
                            side_sign=side_sign,
                        )
                        self.assertIsNotNone(got)
                        self.assertAlmostEqual(got, target_roll, delta=1.5)

    def test_stereo_hand_tracker_fuses_two_cameras_and_holds_brief_dropout(self):
        calib = parallel_camera_calibration((1280, 720), (1280, 720), baseline_m=0.36)
        rig = rig_frame_from_calibration(calib)
        tracker = StereoHandTracker(calib, rig, hold_max_s=0.35)

        pts_rig = _make_hand_world(curl=0.0, roll_deg=120.0, side_sign=-1.0)
        # Convert rig-aligned world_pts into cam1 and cam2 frames
        pts_cam1 = pts_rig @ rig.R_cam1_to_rig
        pts_cam2 = pts_cam1 @ calib.R.T
        obs0 = HandObservation(camera_num=1, world_pts=pts_cam1, image_xy=np.zeros((21, 2)), score=1.0)
        obs1 = HandObservation(camera_num=2, world_pts=pts_cam2, image_xy=np.zeros((21, 2)), score=1.0)

        m = tracker.update(obs0, obs1, forearm_dir_rig=np.array([0.0, 0.0, -1.0]), arm="right", timestamp=1.0, use_filter=False)
        self.assertEqual(m.source, "hands-2")
        self.assertAlmostEqual(m.wrist_roll_deg, 120.0, delta=1.5)
        self.assertGreater(m.hand_open_deg, 170.0)

        # Single-camera handover when cam0 loses sight of the hand
        m1 = tracker.update(None, obs1, forearm_dir_rig=np.array([0.0, 0.0, -1.0]), arm="right", timestamp=1.1, use_filter=False)
        self.assertEqual(m1.source, "hands-1")
        self.assertAlmostEqual(m1.wrist_roll_deg, 120.0, delta=1.5)

        # Brief 0.2s dropout holds smoothly
        mh = tracker.update(None, None, forearm_dir_rig=np.array([0.0, 0.0, -1.0]), arm="right", timestamp=1.3, use_filter=False)
        self.assertEqual(mh.source, "hold")
        self.assertAlmostEqual(mh.wrist_roll_deg, 120.0, delta=1.5)

        # Longer dropout clears cleanly
        me = tracker.update(None, None, forearm_dir_rig=np.array([0.0, 0.0, -1.0]), arm="right", timestamp=2.0, use_filter=False)
        self.assertEqual(me.source, "none")
        self.assertIsNone(me.wrist_roll_deg)
        self.assertIsNone(me.hand_open_deg)

    def test_compute_hand_roi_clamps_to_frame(self):
        obs = {14: (100.0, 200.0, 0.9), 16: (30.0, 40.0, 0.9)}
        roi = compute_hand_roi(obs, (720, 1280), arm="right")
        self.assertIsNotNone(roi)
        x0, y0, x1, y1 = roi
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 1280)
        self.assertLessEqual(y1, 720)
        self.assertGreater(x1 - x0, 100)


if __name__ == "__main__":
    unittest.main()
