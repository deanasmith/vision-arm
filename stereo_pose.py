#!/usr/bin/env python3
"""Dual-camera pose tracking with Charuco-calibrated triangulation."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from stereo.calibration import annotate, hstack_frames
from stereo.cameras import DualCameras
from stereo.config import StereoConfig
from stereo.correspondence import resolve_pose_correspondence
from stereo.geometry import (
    StereoCalibration,
    mean_reprojection_error,
    parallel_camera_calibration,
)
from stereo.kinematics import calculate_3d_angle, elevation_from_down, wrist_roll
from stereo.overlay import (
    LEFT_ELBOW,
    LEFT_HAND,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HAND,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    draw_pose_overlay,
    draw_world_view,
    format_xyz,
    landmark_pixels,
)


def _inside_elbow(flexion: str) -> str:
    """Inside elbow angle: 180 is straight, 90 is a right angle."""
    if not flexion:
        return "--"
    return f"{180.0 - float(flexion):.0f}"


def _arm_metrics(points, shoulder, elbow, wrist, hand_base) -> dict[str, str]:
    """Angles for one arm. Elbow flexion is 0 when straight and 90 at a right angle."""
    blank = {key: "" for key in ("elbow_flexion", "shoulder_elev", "wrist_roll", "upper_m", "forearm_m")}
    shoulder_pt = points.get(shoulder)
    elbow_pt = points.get(elbow)
    # The hand model's wrist is a tighter point than the body-pose wrist.
    wrist_pt = points.get(hand_base)
    if wrist_pt is None:
        wrist_pt = points.get(wrist)
    if shoulder_pt is None or elbow_pt is None:
        return blank
    blank["shoulder_elev"] = f"{elevation_from_down(shoulder_pt, elbow_pt):.1f}"
    blank["upper_m"] = f"{float(np.linalg.norm(elbow_pt - shoulder_pt)):.3f}"
    if wrist_pt is None:
        return blank
    forearm = float(np.linalg.norm(wrist_pt - elbow_pt))
    if not 0.15 <= forearm <= 0.45:
        return blank
    interior = calculate_3d_angle(shoulder_pt, elbow_pt, wrist_pt)
    blank["elbow_flexion"] = f"{180.0 - interior:.1f}"
    blank["forearm_m"] = f"{forearm:.3f}"
    index_mcp = points.get(hand_base + 5)
    pinky_mcp = points.get(hand_base + 17)
    if index_mcp is None or pinky_mcp is None:
        return blank
    knuckle = float(np.linalg.norm(index_mcp - pinky_mcp))
    if not 0.04 <= knuckle <= 0.12:
        return blank
    roll = wrist_roll(elbow_pt, wrist_pt, index_mcp, pinky_mcp)
    if np.isfinite(roll):
        blank["wrist_roll"] = f"{roll:.1f}"
    return blank


def load_or_guess_calibration(config: StereoConfig, assume_parallel: bool, sizes) -> StereoCalibration:
    path = Path(config.calibration_path)
    if path.exists():
        return StereoCalibration.load(path)
    if assume_parallel:
        print("No calibration file found; using parallel-camera approximation", file=sys.stderr)
        return parallel_camera_calibration(sizes[0], sizes[1], config.baseline_m)
    raise FileNotFoundError(
        f"Missing {path}. Run `python calibrate.py --generate-board`, print the board, "
        "then `python calibrate.py` to capture stereo pairs."
    )


def extract_pose_pixels(holistic, frame: np.ndarray, visibility: float):
    """Return (pixels for triangulation, overlay pixels, mediapipe results).

    Pose joints and both hands are triangulated. Hand indices start at
    LEFT_HAND and RIGHT_HAND so they do not collide with pose landmarks.
    """
    h, w = frame.shape[:2]
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose_xy = landmark_pixels(results.pose_landmarks, w, h, visibility)
    pose_xy.update(_hand_pixels(results.left_hand_landmarks, w, h, LEFT_HAND))
    pose_xy.update(_hand_pixels(results.right_hand_landmarks, w, h, RIGHT_HAND))
    draw_xy = dict(pose_xy)
    if results.left_hand_landmarks is not None:
        wrist = results.left_hand_landmarks.landmark[0]
        draw_xy[LEFT_WRIST] = (wrist.x * w, wrist.y * h)
    if results.right_hand_landmarks is not None:
        wrist = results.right_hand_landmarks.landmark[0]
        draw_xy[RIGHT_WRIST] = (wrist.x * w, wrist.y * h)
    return pose_xy, draw_xy, results


def _hand_pixels(landmarks, width: int, height: int, base: int) -> dict[int, tuple[float, float]]:
    if landmarks is None:
        return {}
    return {base + i: (lm.x * width, lm.y * height) for i, lm in enumerate(landmarks.landmark)}


def _pose_body_prior(results):
    """World-landmark positions (metres) and image-relative z for one camera.

    Image z is the pose landmark z MediaPipe already computed: smaller means
    closer to the camera. It is a depth cue, not a metric Z.
    """
    world = {}
    image_z = {}
    world_landmarks = getattr(results, "pose_world_landmarks", None)
    if world_landmarks is not None:
        world = {
            i: np.array([lm.x, lm.y, lm.z], dtype=np.float64)
            for i, lm in enumerate(world_landmarks.landmark)
        }
    pose = getattr(results, "pose_landmarks", None)
    if pose is not None:
        image_z = {i: float(lm.z) for i, lm in enumerate(pose.landmark)}
    return world, image_z


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
        log_path = Path("/tmp/joint_motion.csv")
        log_file = log_path.open("w", newline="")
        log = csv.writer(log_file)
        log.writerow(
            [
                "t",
                "elbow_flexion",
                "shoulder_elev",
                "wrist_roll",
                "upper_m",
                "forearm_m",
                "r_elbow_flexion",
                "r_shoulder_elev",
                "r_wrist_roll",
                "reproj_px",
            ]
        )
        started = time.perf_counter()
        try:
            while True:
                frame_l, frame_r = cams.read()
                if frame_l is None or frame_r is None:
                    print("Failed to read from one or both cameras", file=sys.stderr)
                    return 1

                left_xy, left_draw, left_results = extract_pose_pixels(left_holistic, frame_l, config.visibility)
                right_xy, right_draw, right_results = extract_pose_pixels(right_holistic, frame_r, config.visibility)
                left_world, left_z = _pose_body_prior(left_results)
                right_world, right_z = _pose_body_prior(right_results)
                resolved = resolve_pose_correspondence(
                    calib,
                    left_xy,
                    right_xy,
                    left_world=left_world,
                    right_world=right_world,
                    left_image_z=left_z,
                    right_image_z=right_z,
                )
                points_3d = resolved.points
                matched_right = resolved.right_xy

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

                ids = sorted(set(left_xy) & set(matched_right) & set(points_3d))
                if ids:
                    pts1 = np.array([left_xy[i] for i in ids])
                    pts2 = np.array([matched_right[i] for i in ids])
                    xyz = np.stack([points_3d[i] for i in ids])
                    reproj = mean_reprojection_error(calib, xyz, pts1, pts2)
                    reproj_txt = f"{reproj:.2f}px"
                else:
                    reproj_txt = "n/a"

                left_arm = _arm_metrics(points_3d, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HAND)
                right_arm = _arm_metrics(points_3d, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HAND)
                log.writerow(
                    [
                        f"{time.perf_counter() - started:.3f}",
                        left_arm["elbow_flexion"],
                        left_arm["shoulder_elev"],
                        left_arm["wrist_roll"],
                        left_arm["upper_m"],
                        left_arm["forearm_m"],
                        right_arm["elbow_flexion"],
                        right_arm["shoulder_elev"],
                        right_arm["wrist_roll"],
                        "" if reproj_txt == "n/a" else f"{reproj:.2f}",
                    ]
                )
                log_file.flush()

                rms_txt = "n/a" if np.isnan(calib.rms) else f"{calib.rms:.3f}px"
                hud = [
                    f"baseline {calib.baseline_m:.3f}m  calib RMS {rms_txt}  reproj {reproj_txt}  joints {len(points_3d)}",
                    f"L-sh {format_xyz(points_3d.get(LEFT_SHOULDER))}   L-el {format_xyz(points_3d.get(LEFT_ELBOW))}   L-wr {format_xyz(points_3d.get(LEFT_WRIST))}",
                    "L elbow {le}  elev {ls}  roll {lr}   R elbow {re}  elev {rs}  roll {rr}".format(
                        le=_inside_elbow(left_arm["elbow_flexion"]),
                        ls=left_arm["shoulder_elev"] or "--",
                        lr=left_arm["wrist_roll"] or "--",
                        re=_inside_elbow(right_arm["elbow_flexion"]),
                        rs=right_arm["shoulder_elev"] or "--",
                        rr=right_arm["wrist_roll"] or "--",
                    ),
                ]
                combo = annotate(hstack_frames(vis_l, vis_r), hud)
                world = draw_world_view(points_3d, width=combo.shape[1], height=combo.shape[0])
                cv2.imshow("Stereo pose (triangulated 3D)", np.vstack([combo, world]))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        finally:
            log_file.close()
            left_holistic.close()
            right_holistic.close()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
