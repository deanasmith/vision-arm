#!/usr/bin/env python3
"""Dual-camera pose tracking with Charuco-calibrated triangulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from stereo.calibration import annotate, hstack_frames
from stereo.cameras import DualCameras
from stereo.config import StereoConfig
from stereo.geometry import (
    StereoCalibration,
    mean_reprojection_error,
    parallel_camera_calibration,
    triangulate_matched,
)
from stereo.overlay import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    draw_pose_overlay,
    draw_world_view,
    format_xyz,
    landmark_pixels,
)


def load_or_guess_calibration(config: StereoConfig, assume_parallel: bool, sizes) -> StereoCalibration:
    path = Path(config.calibration_path)
    if path.exists():
        calib = StereoCalibration.load(path)
        if abs(calib.baseline_m - config.baseline_m) > 1e-6 and config.baseline_m > 0:
            # Re-scale the saved translation to the baseline requested at runtime.
            from stereo.geometry import scale_translation_to_baseline

            calib.T, _ = scale_translation_to_baseline(calib.T, config.baseline_m)
            calib.baseline_m = config.baseline_m
        return calib
    if assume_parallel:
        print("No calibration file found; using parallel-camera approximation", file=sys.stderr)
        return parallel_camera_calibration(sizes[0], sizes[1], config.baseline_m)
    raise FileNotFoundError(
        f"Missing {path}. Run `python calibrate.py --generate-board`, print the board, "
        "then `python calibrate.py` to capture stereo pairs."
    )


def extract_pose_pixels(holistic, frame: np.ndarray, visibility: float):
    """Return (pose pixels for triangulation, overlay pixels, mediapipe results).

    Overlay pixels may substitute the hand-model wrist, matching the original
    single-camera drawing. Triangulation always uses pose-landmark indices so
    left/right cameras correspond.
    """
    h, w = frame.shape[:2]
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose_xy = landmark_pixels(results.pose_landmarks, w, h, visibility)
    draw_xy = dict(pose_xy)
    if results.left_hand_landmarks is not None:
        wrist = results.left_hand_landmarks.landmark[0]
        draw_xy[LEFT_WRIST] = (wrist.x * w, wrist.y * h)
    if results.right_hand_landmarks is not None:
        wrist = results.right_hand_landmarks.landmark[0]
        draw_xy[RIGHT_WRIST] = (wrist.x * w, wrist.y * h)
    return pose_xy, draw_xy, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stereo triangulated pose tracking")
    StereoConfig.add_args(parser)
    parser.add_argument(
        "--assume-parallel",
        action="store_true",
        help="Skip Charuco calibration and assume parallel cameras separated by --baseline",
    )
    args = parser.parse_args(argv)
    config = StereoConfig.from_args(args)

    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    with DualCameras(config.left_camera, config.right_camera) as cams:
        sizes = cams.image_sizes()
        try:
            calib = load_or_guess_calibration(config, args.assume_parallel, sizes)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1

        left_holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        right_holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        try:
            while True:
                frame_l, frame_r = cams.read()
                if frame_l is None or frame_r is None:
                    print("Failed to read from one or both cameras", file=sys.stderr)
                    return 1

                left_xy, left_draw, left_results = extract_pose_pixels(left_holistic, frame_l, config.visibility)
                right_xy, right_draw, right_results = extract_pose_pixels(right_holistic, frame_r, config.visibility)
                points_3d = triangulate_matched(calib, left_xy, right_xy)

                vis_l = draw_pose_overlay(frame_l, left_draw, points_3d)
                vis_r = draw_pose_overlay(frame_r, right_draw, points_3d)
                if left_results.left_hand_landmarks:
                    mp_draw.draw_landmarks(vis_l, left_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if left_results.right_hand_landmarks:
                    mp_draw.draw_landmarks(vis_l, left_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if right_results.left_hand_landmarks:
                    mp_draw.draw_landmarks(vis_r, right_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if right_results.right_hand_landmarks:
                    mp_draw.draw_landmarks(vis_r, right_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                ids = sorted(set(left_xy) & set(right_xy) & set(points_3d))
                if ids:
                    pts1 = np.array([left_xy[i] for i in ids])
                    pts2 = np.array([right_xy[i] for i in ids])
                    xyz = np.stack([points_3d[i] for i in ids])
                    reproj = mean_reprojection_error(calib, xyz, pts1, pts2)
                    reproj_txt = f"{reproj:.2f}px"
                else:
                    reproj_txt = "n/a"

                rms_txt = "n/a" if np.isnan(calib.rms) else f"{calib.rms:.3f}px"
                hud = [
                    f"baseline {calib.baseline_m:.3f}m  calib RMS {rms_txt}  reproj {reproj_txt}  joints {len(points_3d)}",
                    f"L-sh {format_xyz(points_3d.get(LEFT_SHOULDER))}   L-el {format_xyz(points_3d.get(LEFT_ELBOW))}   L-wr {format_xyz(points_3d.get(LEFT_WRIST))}",
                    f"R-sh {format_xyz(points_3d.get(RIGHT_SHOULDER))}   R-el {format_xyz(points_3d.get(RIGHT_ELBOW))}   R-wr {format_xyz(points_3d.get(RIGHT_WRIST))}",
                ]
                combo = annotate(hstack_frames(vis_l, vis_r), hud)
                world = draw_world_view(points_3d, width=combo.shape[1], height=280)
                cv2.imshow("Stereo pose (triangulated 3D)", np.vstack([combo, world]))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        finally:
            left_holistic.close()
            right_holistic.close()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
