#!/usr/bin/env python3
"""Dual-camera 3D stereo pose tracker with interactive 3D orbit view and session logging."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import replace
import math
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from arm.motion import JointMotionSmoother
from arm.protocol import Joint
from arm.uart import DEFAULT_BAUD, ArmUart, AsyncArmUart, UartSelectionError, resolve_uart_port
from stereo.cameras import DualCameras
from stereo.config import StereoConfig
from stereo.geometry import StereoCalibration, parallel_camera_calibration
from stereo.hand_tracker import StereoHandTracker, compute_hand_roi, extract_hand_in_roi
from stereo.recorder import NUM_POSE_LANDMARKS, SessionRecorder
from stereo.tracker import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    StereoPoseTracker,
    UPPER_BODY_INDICES,
)
from stereo.viewer3d import DASH_H, DASH_W, DashboardState, compose_dashboard

WINDOW_NAME = "Stereo 3D Pose"

NO_ADAPTER_WARNING = (
    "UART disabled: No USB-UART adapter found. "
    "Looked for /dev/cu.usb* and /dev/cu.wchusbserial*. "
    "Plug in the adapter or pass --port. "
    "Stereo tracking will keep running."
)


def setup_uart(
    port: str | None,
    baud: int = DEFAULT_BAUD,
    *,
    no_uart: bool = False,
    deadband_deg: float = 1.5,
    smooth: bool = True,
    speed_scale: float = 1.0,
    platform: str | None = None,
    directory: str = "/dev",
) -> AsyncArmUart | None:
    """Open the background UART sender, or return None when UART should stay off."""
    if no_uart:
        return None
    try:
        chosen = resolve_uart_port(port, platform=platform, directory=directory)
    except UartSelectionError as exc:
        print(exc, file=sys.stderr)
        raise
    if chosen is None:
        print(NO_ADAPTER_WARNING, file=sys.stderr)
        return None
    try:
        link = ArmUart(chosen, baud)
    except (OSError, ValueError) as exc:
        print(
            f"UART disabled: could not open {chosen} ({exc}). "
            "Pass --port to select the adapter. Stereo tracking will keep running.",
            file=sys.stderr,
        )
        return None
    smoother = JointMotionSmoother(speed_scale=speed_scale) if smooth else None
    mode_tag = f"smoothed x{speed_scale:.2g}" if smooth else "raw"
    print(
        f"UART connected: {chosen} @ {baud} 8N1 ({mode_tag}, all 5 joints: "
        f"HIP={int(Joint.HIP)}, HAND={int(Joint.HAND)}, ELBOW={int(Joint.ELBOW)}, "
        f"WRIST={int(Joint.WRIST)}, SHOULDER={int(Joint.SHOULDER)}) — press U to start sending",
        file=sys.stderr,
    )
    return AsyncArmUart(link, deadband_deg=deadband_deg, smoother=smoother)


def reconnect_uart_live(
    current: AsyncArmUart | None,
    port: str | None,
    baud: int = DEFAULT_BAUD,
    *,
    deadband_deg: float = 1.5,
    smooth: bool = True,
    speed_scale: float = 1.0,
    platform: str | None = None,
    directory: str = "/dev",
) -> AsyncArmUart | None:
    """Close any previous UART link and attempt a fresh connection while tracking runs."""
    if current is not None:
        current.close()
    try:
        return setup_uart(
            port,
            baud,
            no_uart=False,
            deadband_deg=deadband_deg,
            smooth=smooth,
            speed_scale=speed_scale,
            platform=platform,
            directory=directory,
        )
    except UartSelectionError:
        return None


def load_or_guess_calibration(config: StereoConfig, assume_parallel: bool, sizes) -> StereoCalibration:
    path = Path(config.calibration_path)
    if path.exists():
        return StereoCalibration.load(path)
    if assume_parallel:
        print("No calibration file found; using parallel-camera approximation", file=sys.stderr)
        return parallel_camera_calibration(sizes[0], sizes[1], config.baseline_m)
    raise FileNotFoundError(
        f"Missing {path}. Run `uv run python main.py calibrate` to capture stereo pairs."
    )


def _extract_pose(
    pose_model,
    frame: np.ndarray,
) -> tuple[dict[int, tuple[float, float, float]], np.ndarray, np.ndarray]:
    """Run MediaPipe Pose and return (upper-body obs map, full 33x3 xyv, full 33x3 world)."""
    h, w = frame.shape[:2]
    results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    xyv = np.full((NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
    world = np.full((NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
    obs: dict[int, tuple[float, float, float]] = {}

    if results.pose_landmarks is not None:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            x, y, v = float(lm.x * w), float(lm.y * h), float(getattr(lm, "visibility", 1.0))
            xyv[i] = (x, y, v)
            if i in UPPER_BODY_INDICES and v >= 0.10:
                obs[i] = (x, y, v)

    if results.pose_world_landmarks is not None:
        for i, lm in enumerate(results.pose_world_landmarks.landmark):
            world[i] = (float(lm.x), float(lm.y), float(lm.z))

    return obs, xyv, world


def _fmt(val: float | None, spec: str = ".1f") -> str:
    if val is None or not math.isfinite(val):
        return ""
    return format(val, spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive stereo 3D pose tracker")
    StereoConfig.add_args(parser)
    parser.add_argument(
        "--assume-parallel",
        action="store_true",
        help="Skip Charuco calibration and assume parallel cameras separated by --baseline",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="MediaPipe Pose complexity (2=most accurate, 1=faster)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable saving raw dual-camera session logs under logs/",
    )
    parser.add_argument(
        "--port",
        default=None,
        help=(
            "Serial device wired to STM32 USART3. "
            "Default: the only /dev/cu.usb* or /dev/cu.wchusbserial* adapter on macOS "
            "(UART stays off when none are listed), "
            "or the only /dev/ttyUSB* device on Linux (/dev/ttyUSB0 when none are listed)."
        ),
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"UART baud rate, 8N1 (default: {DEFAULT_BAUD}, matching the firmware)",
    )
    parser.add_argument(
        "--no-uart",
        action="store_true",
        help="Disable UART output even when a USB-UART adapter is connected",
    )
    parser.add_argument(
        "--arm",
        choices=("right", "left"),
        default="right",
        help="Which arm to highlight, log, and transmit over UART (default: right)",
    )
    parser.add_argument(
        "--uart-deadband",
        type=float,
        default=1.5,
        help="Minimum joint angle change in degrees before transmitting a new UART frame (default: 1.5)",
    )
    parser.add_argument(
        "--uart-speed",
        type=float,
        default=1.0,
        help="Multiplier for UART motion smoother speed/acceleration limits (default: 1.0; e.g. 0.7 gentler, 1.4 snappier)",
    )
    parser.add_argument(
        "--hip-gain",
        type=float,
        default=1.0,
        help="Scale factor for torso-twist to Joint.HIP angle (default: 1.0 = 1:1, +-90 deg -> 0..180)",
    )
    parser.add_argument(
        "--flip-hip",
        action="store_true",
        help="Invert torso-twist direction sent on Joint.HIP",
    )
    parser.add_argument(
        "--no-uart-smooth",
        action="store_true",
        help="Disable velocity/acceleration-limited UART motion smoothing and send raw angles directly",
    )
    args = parser.parse_args(argv)
    config = StereoConfig.from_args(args)

    try:
        uart = setup_uart(
            args.port,
            args.baud,
            no_uart=args.no_uart,
            deadband_deg=args.uart_deadband,
            smooth=not args.no_uart_smooth,
            speed_scale=args.uart_speed,
        )
    except UartSelectionError:
        return 1

    calib_path = Path(config.calibration_path)
    req_w, req_h = None, None
    preloaded_calib: StereoCalibration | None = None
    if calib_path.exists():
        preloaded_calib = StereoCalibration.load(calib_path)
        req_w, req_h = preloaded_calib.image_size1

    mp_pose = mp.solutions.pose

    try:
        with DualCameras(
            config.left_camera,
            config.right_camera,
            width=req_w,
            height=req_h,
            threaded=True,
        ) as cams:
            sizes = cams.image_sizes()
            if preloaded_calib is not None:
                calib = preloaded_calib
                if sizes[0] != calib.image_size1 or sizes[1] != calib.image_size2:
                    print(
                        f"Warning: camera sizes {sizes} differ from calibration "
                        f"{(calib.image_size1, calib.image_size2)}",
                        file=sys.stderr,
                    )
            else:
                try:
                    calib = load_or_guess_calibration(config, args.assume_parallel, sizes)
                except FileNotFoundError as exc:
                    print(exc, file=sys.stderr)
                    return 1

            tracker = StereoPoseTracker(
                calib,
                config,
                hip_gain=args.hip_gain,
                hip_sign=-1.0 if args.flip_hip else 1.0,
            )
            hip_zeroed = False
            hand_tracker = StereoHandTracker(calib, tracker.rig)
            state = DashboardState(arm=args.arm)
            arm_sh, arm_el, arm_wr = (
                (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
                if args.arm == "left"
                else (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
            )
            recorder = None if args.no_record else SessionRecorder(calib, log_dir="logs")
            log_label = f"REC {recorder.path.name}" if recorder is not None else ""

            pose_cam0 = mp_pose.Pose(
                model_complexity=args.model_complexity,
                smooth_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            pose_cam1 = mp_pose.Pose(
                model_complexity=args.model_complexity,
                smooth_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            mp_hands = mp.solutions.hands
            hands_cam0 = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            hands_cam1 = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, DASH_W, DASH_H)
            cv2.setMouseCallback(WINDOW_NAME, state.handle_mouse)

            log_path = Path("/tmp/joint_motion.csv")
            log_file = log_path.open("w", newline="")
            log = csv.writer(log_file)
            log.writerow(
                [
                    "t",
                    "l_upper_cm",
                    "l_forearm_cm",
                    "l_hip_yaw",
                    "l_sh_elev",
                    "l_sh_fwd",
                    "l_sh_side",
                    "l_elbow_int",
                    "l_elbow_bend",
                    "l_wrist",
                    "l_hand",
                    "l_sh_x",
                    "l_sh_y",
                    "l_sh_z",
                    "l_el_x",
                    "l_el_y",
                    "l_el_z",
                    "l_wr_x",
                    "l_wr_y",
                    "l_wr_z",
                    "l_el_src",
                    "l_wr_src",
                    "ray_gap_mm",
                    "reproj_px",
                    "sampson_px",
                    "hypothesis",
                ]
            )

            started = time.perf_counter()
            fps = 0.0
            last_frame_t = started

            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    while True:
                        frame_l, frame_r, skew_ms = cams.read_synced()
                        if frame_l is None or frame_r is None:
                            print("Failed to read from one or both cameras", file=sys.stderr)
                            return 1

                        now = time.perf_counter()
                        dt = now - last_frame_t
                        last_frame_t = now
                        if dt > 1e-4:
                            inst_fps = 1.0 / dt
                            fps = inst_fps if fps == 0.0 else 0.85 * fps + 0.15 * inst_fps

                        fut0 = pool.submit(_extract_pose, pose_cam0, frame_l)
                        fut1 = pool.submit(_extract_pose, pose_cam1, frame_r)
                        obs0, xyv0, world0 = fut0.result()
                        obs1, xyv1, world1 = fut1.result()

                        t_rel = now - started
                        stereo = tracker.process(obs0, obs1, t_rel, use_filter=state.use_filter)

                        roi0 = compute_hand_roi(obs0, frame_l.shape[:2], arm=args.arm)
                        roi1 = compute_hand_roi(obs1, frame_r.shape[:2], arm=args.arm)
                        hfut0 = pool.submit(extract_hand_in_roi, hands_cam0, frame_l, roi0, 1)
                        hfut1 = pool.submit(extract_hand_in_roi, hands_cam1, frame_r, roi1, 2)
                        hand0 = hfut0.result()
                        hand1 = hfut1.result()

                        sh = stereo.joints.get(arm_sh)
                        el = stereo.joints.get(arm_el)
                        wr = stereo.joints.get(arm_wr)
                        forearm_dir = (
                            (wr.xyz_rig - el.xyz_rig)
                            if (el is not None and wr is not None)
                            else None
                        )
                        hand_metrics = hand_tracker.update(
                            hand0,
                            hand1,
                            forearm_dir_rig=forearm_dir,
                            arm=args.arm,
                            timestamp=t_rel,
                            use_filter=state.use_filter,
                        )

                        if uart is not None and not uart.active:
                            uart.close()
                            uart = None
                            state.uart_enabled = False

                        base_arm = stereo.left_arm if args.arm == "left" else stereo.right_arm
                        active_arm = replace(
                            base_arm,
                            wrist_angle_deg=hand_metrics.wrist_roll_deg,
                            hand_open_deg=hand_metrics.hand_open_deg,
                        )

                        if uart is not None and uart.active and state.uart_enabled:
                            uart.submit(active_arm.to_uart_angles())

                        if recorder is not None:
                            recorder.append(t_rel, skew_ms, xyv0, xyv1, world0, world1, stereo)

                        log.writerow(
                            [
                                f"{t_rel:.3f}",
                                _fmt(active_arm.upper_arm_cm, ".2f"),
                                _fmt(active_arm.forearm_cm, ".2f"),
                                _fmt(active_arm.hip_yaw_deg, ".1f"),
                                _fmt(active_arm.elevation_deg, ".1f"),
                                _fmt(active_arm.forward_pitch_deg, ".1f"),
                                _fmt(active_arm.side_abduction_deg, ".1f"),
                                _fmt(active_arm.elbow_interior_deg, ".1f"),
                                _fmt(active_arm.elbow_flexion_deg, ".1f"),
                                _fmt(active_arm.wrist_angle_deg, ".1f"),
                                _fmt(active_arm.hand_open_deg, ".1f"),
                                _fmt(None if sh is None else float(sh.xyz_rig[0]), ".3f"),
                                _fmt(None if sh is None else float(sh.xyz_rig[1]), ".3f"),
                                _fmt(None if sh is None else float(sh.xyz_rig[2]), ".3f"),
                                _fmt(None if el is None else float(el.xyz_rig[0]), ".3f"),
                                _fmt(None if el is None else float(el.xyz_rig[1]), ".3f"),
                                _fmt(None if el is None else float(el.xyz_rig[2]), ".3f"),
                                _fmt(None if wr is None else float(wr.xyz_rig[0]), ".3f"),
                                _fmt(None if wr is None else float(wr.xyz_rig[1]), ".3f"),
                                _fmt(None if wr is None else float(wr.xyz_rig[2]), ".3f"),
                                "" if el is None else el.source,
                                "" if wr is None else wr.source,
                                _fmt(stereo.median_ray_gap_mm, ".2f"),
                                _fmt(stereo.median_reproj_px, ".2f"),
                                _fmt(stereo.median_sampson_px, ".2f"),
                                stereo.hypothesis,
                            ]
                        )
                        log_file.flush()

                        if uart is None or not uart.active:
                            uart_label = "UART off (I: connect)"
                        elif not state.uart_enabled:
                            uart_label = f"UART READY {Path(uart.port).name}"
                        else:
                            uart_label = f"UART TX {Path(uart.port).name}"

                        xy0_draw = {i: (v[0], v[1]) for i, v in obs0.items() if v[2] >= 0.45}
                        xy1_draw = {i: (v[0], v[1]) for i, v in obs1.items() if v[2] >= 0.45}
                        dashboard = compose_dashboard(
                            frame_l,
                            frame_r,
                            xy0_draw,
                            xy1_draw,
                            stereo,
                            calib,
                            tracker.rig,
                            state,
                            fps=fps,
                            hand_metrics=hand_metrics,
                            log_label=log_label,
                            uart_label=uart_label,
                        )
                        cv2.imshow(WINDOW_NAME, dashboard)

                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:
                            prev_uart_enabled = state.uart_enabled
                            if state.handle_key(key):
                                break
                            if state.uart_connect_requested:
                                state.uart_connect_requested = False
                                state.uart_enabled = False
                                uart = reconnect_uart_live(
                                    uart,
                                    args.port,
                                    args.baud,
                                    deadband_deg=args.uart_deadband,
                                    smooth=not args.no_uart_smooth,
                                    speed_scale=args.uart_speed,
                                )
                            elif state.zero_hip_requested:
                                state.zero_hip_requested = False
                                neutral = tracker.zero_torso_yaw()
                                if neutral is not None:
                                    hip_zeroed = True
                                    print(f"Torso hip zeroed at {neutral:.1f}° -> 90°", file=sys.stderr)
                                else:
                                    print("Torso hip zero skipped: shoulders not visible yet.", file=sys.stderr)
                            elif state.uart_enabled != prev_uart_enabled:
                                if uart is None or not uart.active:
                                    state.uart_enabled = False
                                    print("UART not connected — plug in cable and press I first.", file=sys.stderr)
                                elif state.uart_enabled:
                                    if not hip_zeroed:
                                        neutral = tracker.zero_torso_yaw()
                                        if neutral is not None:
                                            hip_zeroed = True
                                            print(f"Auto-zeroed torso hip at {neutral:.1f}° -> 90°", file=sys.stderr)
                                    uart.reset_deadband()
                                    print(f"UART TX enabled on {uart.port}", file=sys.stderr)
                                else:
                                    print(f"UART TX disabled on {uart.port}", file=sys.stderr)
            finally:
                if recorder is not None:
                    saved = recorder.save()
                    if saved is not None:
                        print(f"Saved stereo session log to {saved}", file=sys.stderr)
                log_file.close()
                pose_cam0.close()
                pose_cam1.close()
                hands_cam0.close()
                hands_cam1.close()
    finally:
        if uart is not None:
            uart.close()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
