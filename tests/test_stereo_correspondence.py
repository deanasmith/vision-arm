"""Correspondence failures measured on calibration/stereo.npz."""

import unittest
from pathlib import Path

import cv2
import numpy as np

from stereo.correspondence import resolve_pose_correspondence, swap_left_right
from stereo.geometry import (
    StereoCalibration,
    fundamental_matrix,
    in_front_of_both,
    mean_reprojection_error,
    project_points,
    sampson_distances,
    triangulate_matched,
)
from stereo.kinematics import calculate_3d_angle

ROOT = Path(__file__).resolve().parents[1]
CALIB_PATH = ROOT / "calibration" / "stereo.npz"

# Traced reach from the stereo review. OpenCV camera frame, metres.
SHOULDER = np.array([0.180, -0.060, 1.650])
ELBOW = np.array([0.300, 0.100, 1.480])
WRIST = np.array([0.220, 0.260, 1.320])


def _pixels(calib, joints, camera):
    return {i: tuple(float(v) for v in project_points(calib, pt, camera)[0]) for i, pt in joints.items()}


def _shift_undistorted_right(calib, right_px, delta):
    und = cv2.undistortPoints(
        np.asarray(right_px, dtype=np.float64).reshape(1, 1, 2), calib.K2, calib.D2, P=calib.K2
    ).reshape(2)
    target = und + np.asarray(delta, dtype=np.float64)
    normalized = np.linalg.inv(calib.K2) @ np.array([target[0], target[1], 1.0])
    distorted, _ = cv2.projectPoints(
        normalized.reshape(1, 3), np.zeros(3), np.zeros(3), calib.K2, calib.D2
    )
    return tuple(float(v) for v in distorted.reshape(2))


def _epipolar_axes(calib, left_px):
    """Unit tangent and normal of the right epipolar line, in undistorted pixels."""
    und = cv2.undistortPoints(
        np.asarray(left_px, dtype=np.float64).reshape(1, 1, 2), calib.K1, calib.D1, P=calib.K1
    ).reshape(2)
    hom = np.array([und[0], und[1], 1.0])
    line = fundamental_matrix(calib) @ hom
    normal = line[:2] / np.linalg.norm(line[:2])
    tangent = np.array([-normal[1], normal[0]])
    return tangent, normal


def _reaching_body():
    """Both arms, with the left arm at the traced reach. Lengths are human."""
    body = {
        11: SHOULDER.copy(),
        13: ELBOW.copy(),
        15: WRIST.copy(),
        12: np.array([-0.180, -0.060, 1.650]),
        14: np.array([-0.300, 0.100, 1.480]),
        16: np.array([-0.220, 0.260, 1.320]),
        0: np.array([0.000, -0.280, 1.600]),
        23: np.array([0.110, 0.280, 1.560]),
        24: np.array([-0.110, 0.280, 1.560]),
        100: WRIST + np.array([-0.008, 0.012, -0.008]),
        200: np.array([-0.228, 0.272, 1.312]),
    }
    # Midline points the swap does not move. They keep the mean reprojection
    # of a bad shoulder assignment under a pixel.
    for index, x in enumerate(np.linspace(-0.12, 0.12, 12)):
        body[50 + index] = np.array([float(x), 0.05, 1.580])
    return body


def _image_z(joints):
    # Same order as camera Z. MediaPipe's image z uses that sense: smaller is closer.
    return {i: float(pt[2]) for i, pt in joints.items()}


class RealRigCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.calib = StereoCalibration.load(CALIB_PATH)

    def test_sampson_rejects_perpendicular_wrist_on_this_rig(self):
        left = _pixels(self.calib, {15: WRIST}, 1)
        right_px = _pixels(self.calib, {15: WRIST}, 2)[15]
        _, normal = _epipolar_axes(self.calib, left[15])
        right = {15: _shift_undistorted_right(self.calib, right_px, 12.0 * normal)}
        sampson = float(sampson_distances(self.calib, [left[15]], [right[15]])[0])
        self.assertGreater(sampson, 8.0)
        self.assertNotIn(15, triangulate_matched(self.calib, left, right))
        kept = triangulate_matched(self.calib, left, right, max_sampson_px=1e6)
        self.assertIn(15, kept)
        residual = max(
            np.linalg.norm(project_points(self.calib, kept[15], 1)[0] - np.array(left[15])),
            np.linalg.norm(project_points(self.calib, kept[15], 2)[0] - np.array(right[15])),
        )
        self.assertLess(residual, 15.0)
        self.assertLess(float(np.linalg.norm(kept[15] - WRIST)), 0.02)

    def test_along_epipolar_shift_is_not_a_sampson_failure(self):
        left = _pixels(self.calib, {15: WRIST}, 1)
        right_px = _pixels(self.calib, {15: WRIST}, 2)[15]
        tangent, _ = _epipolar_axes(self.calib, left[15])
        right = {15: _shift_undistorted_right(self.calib, right_px, -40.0 * tangent)}
        self.assertLess(float(sampson_distances(self.calib, [left[15]], [right[15]])[0]), 0.01)
        out = triangulate_matched(self.calib, left, right)
        self.assertIn(15, out)
        self.assertGreater(abs(float(out[15][2]) - WRIST[2]), 0.10)
        self.assertLess(
            float(np.linalg.norm(project_points(self.calib, out[15], 2)[0] - np.array(right[15]))),
            0.01,
        )

    def test_swapped_sides_do_not_publish_a_distant_shoulder(self):
        body = _reaching_body()
        left = _pixels(self.calib, body, 1)
        right = swap_left_right(_pixels(self.calib, body, 2))
        naive = triangulate_matched(self.calib, left, right)
        far = [pt for pt in naive.values() if float(pt[2]) > 100.0]
        self.assertTrue(far, "the swapped shoulder should triangulate past 100 m before the fix")
        ids = sorted(naive)
        mean = mean_reprojection_error(
            self.calib,
            np.stack([naive[i] for i in ids]),
            np.array([left[i] for i in ids]),
            np.array([right[i] for i in ids]),
        )
        self.assertLess(mean, 1.0)

        resolved = resolve_pose_correspondence(self.calib, left, right)
        self.assertEqual(resolved.hypothesis, "swap")
        for index, truth in body.items():
            self.assertIn(index, resolved.points)
            self.assertLess(float(np.linalg.norm(resolved.points[index] - truth)), 0.01)
        self.assertTrue(all(float(pt[2]) < 5.0 for pt in resolved.points.values()))

    def test_mirrored_right_camera_keeps_the_true_arm(self):
        body = _reaching_body()
        left = _pixels(self.calib, body, 1)
        width = self.calib.image_size2[0]
        right = {i: (width - x, y) for i, (x, y) in _pixels(self.calib, body, 2).items()}
        naive = triangulate_matched(self.calib, left, right)
        self.assertIn(11, naive)
        self.assertGreater(float(np.linalg.norm(naive[11] - body[11])), 1.0)

        resolved = resolve_pose_correspondence(self.calib, left, right)
        self.assertEqual(resolved.hypothesis, "mirror")
        for index, truth in body.items():
            self.assertIn(index, resolved.points)
            self.assertLess(float(np.linalg.norm(resolved.points[index] - truth)), 0.01)

    def test_consistent_frame_stays_on_identity(self):
        body = _reaching_body()
        resolved = resolve_pose_correspondence(
            self.calib, _pixels(self.calib, body, 1), _pixels(self.calib, body, 2)
        )
        self.assertEqual(resolved.hypothesis, "identity")
        for index, truth in body.items():
            self.assertLess(float(np.linalg.norm(resolved.points[index] - truth)), 0.01)

    def test_short_upper_arm_and_forearm_are_dropped(self):
        shoulder = SHOULDER.copy()
        direction = np.array([0.25, 0.85, -0.15])
        direction /= np.linalg.norm(direction)
        elbow = shoulder + direction * 0.08
        wrist = elbow + direction * 0.09
        joints = {11: shoulder, 13: elbow, 15: wrist}
        resolved = resolve_pose_correspondence(
            self.calib, _pixels(self.calib, joints, 1), _pixels(self.calib, joints, 2)
        )
        self.assertIn(11, resolved.points)
        self.assertNotIn(13, resolved.points)
        self.assertNotIn(15, resolved.points)
        self.assertEqual(resolved.hypothesis, "identity")

    def test_depth_error_inside_the_forearm_window_does_not_bend_the_elbow(self):
        # Twelve centimetres added to Z only. The forearm stays inside 15-45 cm
        # and the interior angle moves about 15°. Both views are reprojected,
        # so the reprojection and the Sampson distance are zero. Index 100 is
        # the hand wrist the overlay and the angle readout prefer.
        bad_wrist = np.array([WRIST[0], WRIST[1], WRIST[2] + 0.12])
        joints = {11: SHOULDER, 13: ELBOW, 15: bad_wrist, 100: bad_wrist.copy()}
        left = _pixels(self.calib, joints, 1)
        right = _pixels(self.calib, joints, 2)
        naive = triangulate_matched(self.calib, left, right)
        naive_angle = calculate_3d_angle(naive[11], naive[13], naive[15])
        true_angle = calculate_3d_angle(SHOULDER, ELBOW, WRIST)
        naive_forearm = float(np.linalg.norm(naive[15] - naive[13]))
        self.assertGreater(abs(naive_angle - true_angle), 14.0)
        self.assertTrue(0.15 <= naive_forearm <= 0.45)

        uncorrected = resolve_pose_correspondence(self.calib, left, right)
        self.assertIn(15, uncorrected.points)
        self.assertGreater(
            abs(calculate_3d_angle(uncorrected.points[11], uncorrected.points[13], uncorrected.points[15]) - true_angle),
            14.0,
        )

        world = {11: SHOULDER, 13: ELBOW, 15: WRIST}
        image_z = _image_z({11: SHOULDER, 13: ELBOW, 15: WRIST})
        resolved = resolve_pose_correspondence(
            self.calib,
            left,
            right,
            left_world=world,
            left_image_z=image_z,
        )
        self.assertIn(15, resolved.points)
        self.assertTrue(in_front_of_both(self.calib, resolved.points[15]))
        corrected = calculate_3d_angle(resolved.points[11], resolved.points[13], resolved.points[15])
        self.assertLess(abs(corrected - true_angle), 6.0)
        hand_angle = calculate_3d_angle(resolved.points[11], resolved.points[13], resolved.points[100])
        self.assertLess(abs(hand_angle - true_angle), 6.0)
        self.assertLess(
            abs(float(np.linalg.norm(resolved.points[15] - resolved.points[13])) - 0.24),
            0.01,
        )

    def test_epipolar_depth_error_follows_the_world_prior(self):
        left_px = _pixels(self.calib, {15: WRIST}, 1)[15]
        right_px = _pixels(self.calib, {15: WRIST}, 2)[15]
        tangent, _ = _epipolar_axes(self.calib, left_px)
        joints = {
            11: SHOULDER,
            13: ELBOW,
            15: WRIST,
        }
        left = _pixels(self.calib, joints, 1)
        right = _pixels(self.calib, joints, 2)
        right[15] = _shift_undistorted_right(self.calib, right_px, -40.0 * tangent)
        world = {i: pt.copy() for i, pt in joints.items()}
        image_z = _image_z(joints)
        resolved = resolve_pose_correspondence(
            self.calib, left, right, left_world=world, left_image_z=image_z
        )
        self.assertLess(float(np.linalg.norm(resolved.points[15] - WRIST)), 0.01)
        angle = calculate_3d_angle(resolved.points[11], resolved.points[13], resolved.points[15])
        self.assertAlmostEqual(angle, calculate_3d_angle(SHOULDER, ELBOW, WRIST), delta=0.5)
        self.assertTrue(all(in_front_of_both(self.calib, pt) for pt in resolved.points.values()))

        reversed_z = {11: 1.0, 13: 0.0, 15: 2.0}
        flipped = resolve_pose_correspondence(
            self.calib, left, right, left_world=world, left_image_z=reversed_z
        )
        self.assertGreater(float(flipped.points[15][2]), 1.55)

    def test_prior_that_agrees_with_stereo_does_not_replace_it(self):
        bad_wrist = np.array([WRIST[0], WRIST[1], WRIST[2] + 0.12])
        joints = {11: SHOULDER, 13: ELBOW, 15: bad_wrist}
        # The world prior describes the same (wrong) forearm the pixels do.
        world = {i: pt.copy() for i, pt in joints.items()}
        resolved = resolve_pose_correspondence(
            self.calib,
            _pixels(self.calib, joints, 1),
            _pixels(self.calib, joints, 2),
            left_world=world,
            left_image_z=_image_z(joints),
        )
        self.assertLess(float(np.linalg.norm(resolved.points[15] - bad_wrist)), 0.01)


if __name__ == "__main__":
    unittest.main()
