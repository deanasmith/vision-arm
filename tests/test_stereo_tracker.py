import tempfile
import unittest
from pathlib import Path

import numpy as np

from arm.protocol import Joint
from stereo.filters import OneEuro3D
from stereo.geometry import (
    StereoCalibration,
    epipolar_curve_in_other,
    parallel_camera_calibration,
    project_points,
    rig_frame_from_calibration,
    triangulate_with_diagnostics,
)
from stereo.recorder import NUM_POSE_LANDMARKS, SessionRecorder
from stereo.tracker import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_INDEX,
    LEFT_PINKY,
    LEFT_SHOULDER,
    LEFT_THUMB,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_INDEX,
    RIGHT_PINKY,
    RIGHT_SHOULDER,
    RIGHT_THUMB,
    RIGHT_WRIST,
    StereoPoseTracker,
)
from stereo.viewer3d import DASH_H, DASH_W, DashboardState, compose_dashboard
from stereo_pose import reconnect_uart_live, setup_uart


def _project_obs(
    calib: StereoCalibration,
    pts_cam1: dict[int, np.ndarray],
    camera: int,
    vis: float = 0.98,
) -> dict[int, tuple[float, float, float]]:
    out = {}
    for idx, pt in pts_cam1.items():
        xy = project_points(calib, pt.reshape(1, 3), camera)[0]
        out[idx] = (float(xy[0]), float(xy[1]), vis)
    return out


class StereoTrackerTests(unittest.TestCase):
    def setUp(self):
        self.calib = parallel_camera_calibration((1280, 720), (1280, 720), baseline_m=0.36)
        self.rig = rig_frame_from_calibration(self.calib)

    def test_rig_frame_is_orthonormal_and_symmetric(self):
        R = self.rig.R_cam1_to_rig
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
        np.testing.assert_allclose(self.rig.cam1_pos_rig + self.rig.cam2_pos_rig, np.zeros(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(self.rig.cam2_pos_rig - self.rig.cam1_pos_rig)), 0.36, places=6)

        pt = np.array([0.12, -0.08, 1.35])
        np.testing.assert_allclose(self.rig.to_cam1(self.rig.to_rig(pt)), pt, atol=1e-9)

    def test_rig_frame_on_converged_calibration_if_present(self):
        path = Path(__file__).resolve().parents[1] / "calibration" / "stereo.npz"
        if not path.exists():
            self.skipTest("no calibration/stereo.npz")
        calib = StereoCalibration.load(path)
        rig = rig_frame_from_calibration(calib)
        np.testing.assert_allclose(rig.R_cam1_to_rig @ rig.R_cam1_to_rig.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rig.R_cam1_to_rig)), 1.0, places=9)
        left_pos = rig.cam1_pos_rig if rig.cam0_is_physical_left else rig.cam2_pos_rig
        right_pos = rig.cam2_pos_rig if rig.cam0_is_physical_left else rig.cam1_pos_rig
        self.assertLess(float(left_pos[0]), 0.0)
        self.assertGreater(float(right_pos[0]), 0.0)

    def test_ray_midpoint_triangulation_and_gap(self):
        pts = np.array([[0.10, -0.05, 1.25], [-0.12, 0.15, 1.60]], dtype=np.float64)
        p1 = project_points(self.calib, pts, 1)
        p2 = project_points(self.calib, pts, 2)
        tri = triangulate_with_diagnostics(self.calib, p1, p2)
        self.assertTrue(np.all(tri.in_front))
        np.testing.assert_allclose(tri.xyz_cam1, pts, atol=1e-6)
        np.testing.assert_allclose(tri.ray_gap_m, np.zeros(2), atol=1e-6)
        self.assertGreater(float(tri.ray_angle_deg[0]), 10.0)

        p2_bad = p2.copy()
        p2_bad[0, 1] += 20.0
        tri_bad = triangulate_with_diagnostics(self.calib, p1, p2_bad)
        self.assertGreater(float(tri_bad.ray_gap_m[0]) * 1000.0, 15.0)
        self.assertGreater(float(tri_bad.sampson_px[0]), 10.0)

    def test_epipolar_curve_passes_through_true_correspondence(self):
        pt = np.array([[0.15, -0.04, 1.30]], dtype=np.float64)
        p1 = project_points(self.calib, pt, 1)[0]
        p2 = project_points(self.calib, pt, 2)[0]
        curve = epipolar_curve_in_other(self.calib, (float(p1[0]), float(p1[1])), source_camera=1, n_samples=40)
        self.assertGreater(len(curve), 5)
        dists = []
        for a, b in zip(curve[:-1], curve[1:]):
            ab = b - a
            t = np.clip(np.dot(p2 - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0)
            dists.append(float(np.linalg.norm((a + t * ab) - p2)))
        self.assertLess(min(dists), 0.5)

    def test_one_euro_smooths_jitter_and_resets_on_jump(self):
        filt = OneEuro3D(min_cutoff=1.2, beta=0.5)
        rng = np.random.default_rng(7)
        true_pt = np.array([0.1, -0.1, 1.2])
        raw_err, filt_err = [], []
        for k in range(40):
            noisy = true_pt + rng.normal(0.0, 0.015, size=3)
            out = filt.filter_point(11, noisy, k * 0.05)
            if k >= 10:
                raw_err.append(float(np.linalg.norm(noisy - true_pt)))
                filt_err.append(float(np.linalg.norm(out - true_pt)))
        self.assertLess(float(np.mean(filt_err)), float(np.mean(raw_err)) * 0.6)

        jumped = np.array([0.9, 0.2, 1.8])
        out = filt.filter_point(11, jumped, 41 * 0.05)
        np.testing.assert_allclose(out, jumped, atol=1e-9)

    def test_tracker_recovers_exact_3d_kinematics_and_swap(self):
        body_rig = {
            LEFT_SHOULDER: np.array([0.20, -0.10, 1.30]),
            RIGHT_SHOULDER: np.array([-0.20, -0.10, 1.30]),
            LEFT_ELBOW: np.array([0.20, -0.10, 1.00]),
            LEFT_WRIST: np.array([0.20, -0.35, 1.00]),
            RIGHT_ELBOW: np.array([-0.20, 0.20, 1.30]),
            RIGHT_WRIST: np.array([-0.20, 0.45, 1.30]),
            LEFT_HIP: np.array([0.14, 0.35, 1.30]),
            RIGHT_HIP: np.array([-0.14, 0.35, 1.30]),
        }
        body_cam1 = {idx: self.rig.to_cam1(pt) for idx, pt in body_rig.items()}
        left_obs = _project_obs(self.calib, body_cam1, 1)
        right_obs = _project_obs(self.calib, body_cam1, 2)

        tracker = StereoPoseTracker(self.calib)
        res = tracker.process(left_obs, right_obs, timestamp=0.1, use_filter=False)
        self.assertEqual(res.hypothesis, "identity")
        self.assertAlmostEqual(res.left_arm.upper_arm_cm, 30.0, places=3)
        self.assertAlmostEqual(res.left_arm.forearm_cm, 25.0, places=3)
        self.assertAlmostEqual(res.left_arm.elevation_deg, 90.0, places=2)
        self.assertAlmostEqual(res.left_arm.forward_pitch_deg, 90.0, places=2)
        self.assertAlmostEqual(res.left_arm.elbow_flexion_deg, 90.0, places=2)
        self.assertLess(res.median_ray_gap_mm, 0.01)

        swap_map = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23}
        swapped_right = {swap_map[k]: v for k, v in right_obs.items()}
        res_sw = tracker.process(left_obs, swapped_right, timestamp=0.2, use_filter=False)
        self.assertEqual(res_sw.hypothesis, "swap")
        self.assertAlmostEqual(res_sw.left_arm.upper_arm_cm, 30.0, places=3)

    def test_single_camera_ray_bone_handover_when_one_view_occluded(self):
        # Warm up 6 clean frames so BoneLengthModel calibrates upper arm = 30 cm, forearm = 25 cm.
        body_rig = {
            LEFT_SHOULDER: np.array([0.20, -0.10, 1.30]),
            RIGHT_SHOULDER: np.array([-0.20, -0.10, 1.30]),
            LEFT_ELBOW: np.array([0.20, 0.08, 1.06]),     # 30 cm
            LEFT_WRIST: np.array([0.16, 0.05, 0.815]),    # ~25 cm
            LEFT_HIP: np.array([0.14, 0.35, 1.30]),
            RIGHT_HIP: np.array([-0.14, 0.35, 1.30]),
        }
        true_forearm = float(np.linalg.norm(body_rig[LEFT_WRIST] - body_rig[LEFT_ELBOW]))
        body_cam1 = {idx: self.rig.to_cam1(pt) for idx, pt in body_rig.items()}
        tracker = StereoPoseTracker(self.calib)
        for k in range(6):
            l_obs = _project_obs(self.calib, body_cam1, 1, vis=0.95)
            r_obs = _project_obs(self.calib, body_cam1, 2, vis=0.95)
            tracker.process(l_obs, r_obs, timestamp=0.05 * k, use_filter=False)

        # Now move wrist slightly and completely occlude wrist in camera 2 (arm pointing at cam 2).
        moved_rig = dict(body_rig)
        v = np.array([-0.03, -0.02, -0.245])
        v = v / np.linalg.norm(v) * true_forearm
        moved_rig[LEFT_WRIST] = moved_rig[LEFT_ELBOW] + v
        moved_cam1 = {idx: self.rig.to_cam1(pt) for idx, pt in moved_rig.items()}
        l_obs = _project_obs(self.calib, moved_cam1, 1, vis=0.95)
        r_obs = _project_obs(self.calib, moved_cam1, 2, vis=0.95)
        del r_obs[LEFT_WRIST]

        res = tracker.process(l_obs, r_obs, timestamp=0.35, use_filter=False)
        self.assertIn(LEFT_WRIST, res.joints)
        self.assertEqual(res.joints[LEFT_WRIST].source, "ray+bone")
        err_mm = float(np.linalg.norm(res.joints[LEFT_WRIST].xyz_rig - moved_rig[LEFT_WRIST])) * 1000.0
        self.assertLess(err_mm, 8.0)

        # Brief 1-frame dual-camera dropout holds rigidly, then expires after HOLD_MAX_AGE_S.
        del l_obs[LEFT_WRIST]
        res_hold = tracker.process(l_obs, r_obs, timestamp=0.42, use_filter=False)
        self.assertIn(LEFT_WRIST, res_hold.joints)
        self.assertEqual(res_hold.joints[LEFT_WRIST].source, "hold")
        res_expired = tracker.process(l_obs, r_obs, timestamp=0.90, use_filter=False)
        self.assertNotIn(LEFT_WRIST, res_expired.joints)

    def test_recorder_and_clean_dashboard(self):
        body_rig = {
            LEFT_SHOULDER: np.array([0.20, -0.10, 1.30]),
            RIGHT_SHOULDER: np.array([-0.20, -0.10, 1.30]),
            LEFT_ELBOW: np.array([0.28, 0.14, 1.15]),
            LEFT_WRIST: np.array([0.18, 0.05, 0.92]),
            LEFT_HIP: np.array([0.14, 0.35, 1.30]),
            RIGHT_HIP: np.array([-0.14, 0.35, 1.30]),
        }
        body_cam1 = {idx: self.rig.to_cam1(pt) for idx, pt in body_rig.items()}
        l_obs = _project_obs(self.calib, body_cam1, 1)
        r_obs = _project_obs(self.calib, body_cam1, 2)
        l_xy = {k: (v[0], v[1]) for k, v in l_obs.items()}
        r_xy = {k: (v[0], v[1]) for k, v in r_obs.items()}

        tracker = StereoPoseTracker(self.calib)
        stereo = tracker.process(l_obs, r_obs, timestamp=0.1, use_filter=False)

        with tempfile.TemporaryDirectory() as tmp:
            rec = SessionRecorder(self.calib, log_dir=tmp, flush_every=10)
            xyv = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)
            rec.append(0.1, 3.0, xyv, xyv, xyv, xyv, stereo)
            saved = rec.save()
            self.assertIsNotNone(saved)
            data = np.load(saved)
            self.assertEqual(data["cam0_xyv"].shape, (1, 33, 3))
            self.assertEqual(data["solved_rig"].shape[0], 1)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = DashboardState()
        for preset in (1, 2, 3, 4):
            state.orbit.set_preset(preset)
            img = compose_dashboard(
                frame,
                frame,
                l_xy,
                r_xy,
                stereo,
                self.calib,
                self.rig,
                state,
                fps=14.0,
                log_label="REC test.npz",
                uart_label="UART /dev/cu.usbserial-10",
            )
            self.assertEqual(img.shape, (DASH_H, DASH_W, 3))

    def test_all_five_uart_joint_angles_from_stereo_pose(self):
        # Left arm reaching straight forward (-Z) from shoulder, forearm straight up (-Y),
        # hand straight up with open thumb-index spread.
        body_rig = {
            LEFT_SHOULDER: np.array([0.20, -0.10, 1.30]),
            RIGHT_SHOULDER: np.array([-0.20, -0.10, 1.30]),
            LEFT_ELBOW: np.array([0.20, -0.10, 1.00]),       # 30 cm forward (-Z) -> hip=90°, shoulder=90°
            LEFT_WRIST: np.array([0.20, -0.35, 1.00]),       # 25 cm up (-Y) -> elbow interior=90°
            LEFT_INDEX: np.array([0.20, -0.45, 1.00]),       # 10 cm straight up -> wrist=180°
            LEFT_PINKY: np.array([0.20, -0.45, 1.00]),
            LEFT_THUMB: np.array([0.11, -0.43, 1.00]),       # ~9.2 cm spread on 10 cm palm -> open (~180°)
            LEFT_HIP: np.array([0.14, 0.35, 1.30]),
            RIGHT_HIP: np.array([-0.14, 0.35, 1.30]),
        }
        body_cam1 = {idx: self.rig.to_cam1(pt) for idx, pt in body_rig.items()}
        l_obs = _project_obs(self.calib, body_cam1, 1)
        r_obs = _project_obs(self.calib, body_cam1, 2)

        tracker = StereoPoseTracker(self.calib)
        res = tracker.process(l_obs, r_obs, timestamp=0.1, use_filter=False)
        angles = res.left_arm.to_uart_angles()
        self.assertEqual(
            set(angles.keys()),
            {Joint.HIP, Joint.HAND, Joint.ELBOW, Joint.WRIST, Joint.SHOULDER},
        )
        self.assertAlmostEqual(angles[Joint.HIP], 90.0, places=1)
        self.assertAlmostEqual(angles[Joint.SHOULDER], 90.0, places=1)
        self.assertAlmostEqual(angles[Joint.ELBOW], 90.0, places=1)
        self.assertAlmostEqual(angles[Joint.WRIST], 180.0, places=1)
        self.assertGreater(angles[Joint.HAND], 170.0)

    def test_stereo_uart_setup_handles_missing_adapter_and_no_uart(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(setup_uart(None, platform="darwin", directory=tmp))
            self.assertIsNone(setup_uart("/dev/cu.fake", no_uart=True, platform="darwin", directory=tmp))
            self.assertIsNone(reconnect_uart_live(None, None, platform="darwin", directory=tmp))
            # Multiple adapters should not raise during live I-key reconnect.
            (Path(tmp) / "cu.usbserial-A").touch()
            (Path(tmp) / "cu.usbserial-B").touch()
            self.assertIsNone(reconnect_uart_live(None, None, platform="darwin", directory=tmp))

    def test_dashboard_uart_hotkeys_default_disarmed(self):
        state = DashboardState()
        self.assertEqual(state.arm, "right")
        self.assertFalse(state.uart_enabled)
        self.assertFalse(state.uart_connect_requested)
        self.assertFalse(state.handle_key(ord("i")))
        self.assertTrue(state.uart_connect_requested)
        self.assertFalse(state.handle_key(ord("u")))
        self.assertTrue(state.uart_enabled)
        self.assertFalse(state.handle_key(ord("U")))
        self.assertFalse(state.uart_enabled)

    def test_right_arm_five_uart_joint_angles(self):
        tracker = StereoPoseTracker(self.calib)
        pts_rig = {
            LEFT_SHOULDER: np.array([0.18, -0.10, 1.50]),
            RIGHT_SHOULDER: np.array([-0.18, -0.10, 1.50]),
            LEFT_HIP: np.array([0.14, 0.35, 1.50]),
            RIGHT_HIP: np.array([-0.14, 0.35, 1.50]),
            # Right arm reaching forward toward cameras (-Z), 90 deg elevation, 90 deg elbow bend.
            RIGHT_ELBOW: np.array([-0.18, -0.10, 1.20]),
            RIGHT_WRIST: np.array([-0.18, -0.35, 1.20]),
            RIGHT_INDEX: np.array([-0.18, -0.45, 1.20]),
            RIGHT_PINKY: np.array([-0.14, -0.44, 1.20]),
            RIGHT_THUMB: np.array([-0.25, -0.41, 1.20]),
            LEFT_ELBOW: np.array([0.22, 0.20, 1.50]),
            LEFT_WRIST: np.array([0.22, 0.45, 1.50]),
        }
        pts_cam1 = {idx: self.rig.to_cam1(p) for idx, p in pts_rig.items()}
        obs0 = _project_obs(self.calib, pts_cam1, 1)
        obs1 = _project_obs(self.calib, pts_cam1, 2)
        frame = tracker.process(obs0, obs1, timestamp=0.0, use_filter=False)
        angles = frame.right_arm.to_uart_angles()
        self.assertEqual(set(angles.keys()), {Joint.HIP, Joint.HAND, Joint.ELBOW, Joint.WRIST, Joint.SHOULDER})
        self.assertAlmostEqual(angles[Joint.HIP], 90.0, delta=2.0)
        self.assertAlmostEqual(angles[Joint.SHOULDER], 90.0, delta=2.0)
        self.assertAlmostEqual(angles[Joint.ELBOW], 90.0, delta=2.0)
        self.assertGreater(angles[Joint.WRIST], 150.0)
        self.assertGreater(angles[Joint.HAND], 60.0)

        # Straight arm -> elbow flexion UART angle should be 0 deg (not 180 deg).
        pts_straight = dict(pts_rig)
        pts_straight[RIGHT_WRIST] = np.array([-0.18, -0.10, 0.95])
        pts_cam1_s = {idx: self.rig.to_cam1(p) for idx, p in pts_straight.items()}
        frame_s = StereoPoseTracker(self.calib).process(
            _project_obs(self.calib, pts_cam1_s, 1),
            _project_obs(self.calib, pts_cam1_s, 2),
            timestamp=0.0,
            use_filter=False,
        )
        self.assertAlmostEqual(frame_s.right_arm.to_uart_angles()[Joint.ELBOW], 0.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
