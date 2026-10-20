#!/usr/bin/env python3
"""Generate a printable Charuco board or capture stereo calibration pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from stereo.calibration import CaptureBuffer, annotate, hstack_frames, try_pair
from stereo.cameras import DualCameras
from stereo.charuco import CharucoHelper
from stereo.config import StereoConfig


def generate_board(config: StereoConfig) -> Path:
    helper = CharucoHelper(config)
    path = helper.save_printable_board()
    width_cm = config.board_width_m * 100
    height_cm = config.board_height_m * 100
    print(f"Wrote Charuco board to {path}")
    print(f"Print at 100% scale (do not fit-to-page).")
    print(f"The board squares should measure {width_cm:.1f} cm x {height_cm:.1f} cm.")
    print(f"Square size {config.square_length_m * 100:.1f} cm, marker size {config.marker_length_m * 100:.1f} cm.")
    return path


def capture_calibration(config: StereoConfig) -> int:
    helper = CharucoHelper(config)
    buffer = CaptureBuffer()
    last_status = "Show the Charuco board to BOTH cameras, then press SPACE"

    with DualCameras(config.left_camera, config.right_camera) as cams:
        while True:
            frame_l, frame_r = cams.read()
            if frame_l is None or frame_r is None:
                print("Failed to read from one or both cameras", file=sys.stderr)
                return 1

            paired, vis_l, vis_r, n_l, n_r, n_common = try_pair(helper, frame_l, frame_r, config)
            ready = paired is not None
            hud = [
                f"L corners:{n_l}  R corners:{n_r}  shared:{n_common}  captured:{len(buffer)}/{config.min_stereo_pairs}",
                "SPACE capture pair   C compute+save   G show board   ESC quit",
                last_status,
            ]
            combo = annotate(hstack_frames(vis_l, vis_r), hud)
            cv2.imshow("Stereo Charuco calibration", combo)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("g"), ord("G")):
                board_img = helper.generate_image()
                cv2.imshow("Charuco board (print this at 100%)", board_img)
            if key == 32:
                if not ready:
                    last_status = f"Need at least {config.min_corners} shared Charuco corners in both views"
                    continue
                obj, img_l, img_r = paired
                try:
                    buffer.add(
                        obj,
                        img_l,
                        img_r,
                        (frame_l.shape[1], frame_l.shape[0]),
                        (frame_r.shape[1], frame_r.shape[0]),
                    )
                    last_status = f"Captured pair {len(buffer)} ({len(obj)} corners)"
                except ValueError as exc:
                    last_status = str(exc)
            if key in (ord("c"), ord("C")):
                if len(buffer) < config.min_stereo_pairs:
                    last_status = f"Need at least {config.min_stereo_pairs} pairs (have {len(buffer)})"
                    continue
                try:
                    calib = buffer.solve(config.baseline_m)
                    calib.save(config.calibration_path)
                    last_status = (
                        f"Saved {config.calibration_path}  RMS={calib.rms:.3f}px  "
                        f"measured baseline={calib.measured_baseline_m:.3f}m  "
                        f"scaled to {calib.baseline_m:.3f}m"
                    )
                    print(last_status)
                except Exception as exc:
                    last_status = f"Calibration failed: {exc}"
                    print(last_status, file=sys.stderr)

    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Charuco stereo calibration for dual-camera pose")
    parser.add_argument("--generate-board", action="store_true", help="Write a printable Charuco board image and exit")
    StereoConfig.add_args(parser)
    args = parser.parse_args(argv)
    config = StereoConfig.from_args(args)
    if args.generate_board:
        generate_board(config)
        return 0
    return capture_calibration(config)


if __name__ == "__main__":
    raise SystemExit(main())
