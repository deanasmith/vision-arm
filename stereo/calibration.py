from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from stereo.charuco import CharucoHelper, draw_detection, paired_charuco_points
from stereo.config import StereoConfig
from stereo.geometry import StereoCalibration, solve_stereo


@dataclass
class CaptureBuffer:
    object_points: list[np.ndarray] = field(default_factory=list)
    image_points_left: list[np.ndarray] = field(default_factory=list)
    image_points_right: list[np.ndarray] = field(default_factory=list)
    image_size1: tuple[int, int] | None = None
    image_size2: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.object_points)

    def add(
        self,
        obj: np.ndarray,
        img_l: np.ndarray,
        img_r: np.ndarray,
        size1: tuple[int, int],
        size2: tuple[int, int],
    ) -> None:
        if self.image_size1 is None:
            self.image_size1 = size1
            self.image_size2 = size2
        elif size1 != self.image_size1 or size2 != self.image_size2:
            raise ValueError("Camera resolution changed during calibration capture")
        self.object_points.append(obj)
        self.image_points_left.append(img_l)
        self.image_points_right.append(img_r)

    def solve(self, baseline_m: float) -> StereoCalibration:
        if self.image_size1 is None or self.image_size2 is None:
            raise ValueError("No Charuco stereo pairs captured")
        return solve_stereo(
            self.object_points,
            self.image_points_left,
            self.image_points_right,
            self.image_size1,
            self.image_size2,
            baseline_m,
        )


def hstack_frames(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = min(left.shape[0], right.shape[0])

    def resize(frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] == height:
            return frame
        scale = height / frame.shape[0]
        return cv2.resize(frame, (int(frame.shape[1] * scale), height))

    return np.hstack([resize(left), resize(right)])


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    vis = frame.copy()
    y = 28
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return vis


def try_pair(helper: CharucoHelper, frame_l: np.ndarray, frame_r: np.ndarray, config: StereoConfig):
    corners_l, ids_l = helper.detect(frame_l)
    corners_r, ids_r = helper.detect(frame_r)
    paired = paired_charuco_points(
        helper.board, corners_l, ids_l, corners_r, ids_r, min_corners=config.min_corners
    )
    vis_l = draw_detection(frame_l, corners_l, ids_l)
    vis_r = draw_detection(frame_r, corners_r, ids_r)
    n_l = 0 if ids_l is None else len(ids_l)
    n_r = 0 if ids_r is None else len(ids_r)
    n_common = 0 if paired is None else len(paired[0])
    return paired, vis_l, vis_r, n_l, n_r, n_common
