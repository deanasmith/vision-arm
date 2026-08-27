import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stereo.charuco import CharucoHelper
from stereo.config import StereoConfig
from stereo.geometry import (
    StereoCalibration,
    project_points,
    solve_stereo,
    triangulate_pixels,
)


def _board_views(obj: np.ndarray, K: np.ndarray, D: np.ndarray, T_stereo: np.ndarray):
    """Project a Charuco board into a pair of parallel cameras from several poses."""
    object_points = []
    img_left = []
    img_right = []
    for dx in (-0.06, 0.0, 0.06):
        for dy in (-0.05, 0.05):
            for yaw in (-0.25, 0.25):
                rvec = np.array([0.18, yaw, 0.05], dtype=np.float64)
                tvec = np.array([dx, dy, 0.95], dtype=np.float64)
                img1, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
                R1, _ = cv2.Rodrigues(rvec)
                x1 = (R1 @ obj.T + tvec.reshape(3, 1)).T
                x2 = x1 + T_stereo.reshape(1, 3)
                img2, _ = cv2.projectPoints(x2, np.zeros(3), np.zeros(3), K, D)
                object_points.append(obj.astype(np.float32))
                img_left.append(img1.reshape(-1, 2).astype(np.float32))
                img_right.append(img2.reshape(-1, 2).astype(np.float32))
    return object_points, img_left, img_right


class CalibrationTests(unittest.TestCase):
    def test_solve_stereo_recovers_geometry(self):
        config = StereoConfig(baseline_m=0.12)
        obj = CharucoHelper(config).board.getChessboardCorners().astype(np.float64)
        K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        D = np.zeros(5)
        T_true = np.array([-config.baseline_m, 0.0, 0.0])
        object_points, img_l, img_r = _board_views(obj, K, D, T_true)

        calib = solve_stereo(object_points, img_l, img_r, (640, 480), (640, 480), config.baseline_m)

        self.assertLess(calib.rms, 0.05)
        self.assertAlmostEqual(float(np.linalg.norm(calib.T)), config.baseline_m, places=3)
        self.assertAlmostEqual(calib.measured_baseline_m, config.baseline_m, delta=0.01)
        # Parallel cameras: rotation near identity, translation mostly -X.
        self.assertGreater(float(np.trace(calib.R)), 2.9)
        self.assertLess(float(calib.T[0, 0]), 0.0)

        world = np.array([[0.0, 0.0, 1.4], [0.08, -0.1, 1.8]])
        # Project with the solved cameras, then triangulate back.
        pts1 = project_points(calib, world, 1)
        pts2 = project_points(calib, world, 2)
        recovered = triangulate_pixels(calib, pts1, pts2)
        np.testing.assert_allclose(recovered, world, atol=1e-3)

    def test_save_and_load_round_trip(self):
        config = StereoConfig(baseline_m=0.12)
        obj = CharucoHelper(config).board.getChessboardCorners().astype(np.float64)
        K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
        object_points, img_l, img_r = _board_views(obj, K, np.zeros(5), np.array([-0.12, 0.0, 0.0]))
        calib = solve_stereo(object_points, img_l, img_r, (640, 480), (640, 480), 0.12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.npz"
            calib.save(path)
            loaded = StereoCalibration.load(path)
        np.testing.assert_allclose(loaded.K1, calib.K1)
        np.testing.assert_allclose(loaded.T, calib.T)
        self.assertEqual(loaded.image_size1, (640, 480))
        self.assertAlmostEqual(loaded.baseline_m, 0.12)


if __name__ == "__main__":
    unittest.main()
