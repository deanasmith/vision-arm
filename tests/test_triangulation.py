import unittest

import numpy as np

from stereo.geometry import (
    mean_reprojection_error,
    parallel_camera_calibration,
    project_points,
    scale_translation_to_baseline,
    triangulate_matched,
    triangulate_pixels,
)


class TriangulationTests(unittest.TestCase):
    def setUp(self):
        self.calib = parallel_camera_calibration((640, 480), (640, 480), baseline_m=0.15)
        self.world = np.array(
            [
                [0.00, 0.00, 1.50],
                [0.10, -0.20, 2.00],
                [-0.05, 0.12, 1.20],
                [0.20, 0.05, 2.50],
            ],
            dtype=np.float64,
        )

    def test_round_trip_pixels(self):
        pts1 = project_points(self.calib, self.world, 1)
        pts2 = project_points(self.calib, self.world, 2)
        recovered = triangulate_pixels(self.calib, pts1, pts2)
        np.testing.assert_allclose(recovered, self.world, atol=1e-6)

    def test_depth_from_disparity(self):
        point = np.array([[0.0, 0.0, 1.5]])
        pts1 = project_points(self.calib, point, 1)
        pts2 = project_points(self.calib, point, 2)
        disparity = float(pts1[0, 0] - pts2[0, 0])
        f = self.calib.K1[0, 0]
        z = f * self.calib.baseline_m / disparity
        self.assertAlmostEqual(z, 1.5, places=6)

    def test_baseline_scale_changes_metric_size(self):
        T, measured = scale_translation_to_baseline(np.array([2.0, 0.0, 0.0]), 0.15)
        self.assertAlmostEqual(measured, 2.0)
        self.assertAlmostEqual(float(np.linalg.norm(T)), 0.15, places=9)

    def test_matched_landmarks_and_cheirality(self):
        pts1 = {11: tuple(project_points(self.calib, self.world[0:1], 1)[0])}
        pts2 = {11: tuple(project_points(self.calib, self.world[0:1], 2)[0])}
        # Landmark only in left camera is ignored.
        pts1[12] = (100.0, 100.0)
        out = triangulate_matched(self.calib, pts1, pts2)
        self.assertIn(11, out)
        self.assertNotIn(12, out)
        np.testing.assert_allclose(out[11], self.world[0], atol=1e-6)

    def test_reprojection_error_near_zero(self):
        pts1 = project_points(self.calib, self.world, 1)
        pts2 = project_points(self.calib, self.world, 2)
        err = mean_reprojection_error(self.calib, self.world, pts1, pts2)
        self.assertLess(err, 1e-6)


if __name__ == "__main__":
    unittest.main()
