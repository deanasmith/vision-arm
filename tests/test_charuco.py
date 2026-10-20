import unittest

import numpy as np

from stereo.charuco import CharucoHelper, paired_charuco_points
from stereo.config import StereoConfig


class CharucoTests(unittest.TestCase):
    def setUp(self):
        self.config = StereoConfig()
        self.helper = CharucoHelper(self.config)

    def test_generated_board_is_detected(self):
        image = self.helper.generate_image((900, 1260))
        corners, ids = self.helper.detect(image)
        self.assertIsNotNone(ids)
        inner = (self.config.squares_x - 1) * (self.config.squares_y - 1)
        self.assertGreaterEqual(len(ids), inner - 2)

    def test_paired_points_match_shared_ids(self):
        image = self.helper.generate_image((900, 1260))
        corners, ids = self.helper.detect(image)
        paired = paired_charuco_points(self.helper.board, corners, ids, corners, ids, min_corners=6)
        self.assertIsNotNone(paired)
        obj, img_l, img_r = paired
        self.assertEqual(len(obj), len(ids))
        np.testing.assert_allclose(img_l, img_r)
        self.assertEqual(obj.shape[1], 3)

    def test_too_few_shared_corners_returns_none(self):
        image = self.helper.generate_image((900, 1260))
        corners, ids = self.helper.detect(image)
        paired = paired_charuco_points(
            self.helper.board, corners, ids, corners, ids[:2], min_corners=6
        )
        self.assertIsNone(paired)


if __name__ == "__main__":
    unittest.main()
