from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class StereoConfig:
    """Tunable stereo rig and Charuco board parameters.

    Distances are meters. The baseline is the physical distance between camera
    optical centers and is used to scale reconstruction into metric units.
    Default board (5x7 squares of 3 cm) fits on A4 / US Letter at 100% print.
    """

    left_camera: int = 0
    right_camera: int = 1
    baseline_m: float = 0.15
    squares_x: int = 5
    squares_y: int = 7
    square_length_m: float = 0.03
    marker_length_m: float = 0.022
    min_corners: int = 6
    min_stereo_pairs: int = 8
    visibility: float = 0.5
    calibration_path: str = "calibration/stereo.npz"
    board_image_path: str = "calibration/charuco_board.png"

    @property
    def board_width_m(self) -> float:
        return self.squares_x * self.square_length_m

    @property
    def board_height_m(self) -> float:
        return self.squares_y * self.square_length_m

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--left-camera", type=int, default=cls.left_camera)
        parser.add_argument("--right-camera", type=int, default=cls.right_camera)
        parser.add_argument(
            "--baseline",
            type=float,
            default=cls.baseline_m,
            help="Distance between camera optical centers in meters",
        )
        parser.add_argument("--squares-x", type=int, default=cls.squares_x)
        parser.add_argument("--squares-y", type=int, default=cls.squares_y)
        parser.add_argument("--square-length", type=float, default=cls.square_length_m)
        parser.add_argument("--marker-length", type=float, default=cls.marker_length_m)
        parser.add_argument("--calib", type=str, default=cls.calibration_path)
        parser.add_argument("--board-image", type=str, default=cls.board_image_path)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> StereoConfig:
        return cls(
            left_camera=args.left_camera,
            right_camera=args.right_camera,
            baseline_m=args.baseline,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length_m=args.square_length,
            marker_length_m=args.marker_length,
            calibration_path=args.calib,
            board_image_path=args.board_image,
        )
