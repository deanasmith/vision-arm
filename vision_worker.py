import argparse
from collections import deque
import datetime
import json
import time

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from stereo.config import StereoConfig
from stereo.geometry import triangulate_matched
from stereo.overlay import draw_pose_overlay, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
from stereo.kinematics import calculate_3d_angle, calculate_shoulder_flexion
from stereo_pose import extract_pose_pixels, load_or_guess_calibration
from angle_sender import AngleSender

# MediaPipe pose landmark indices (person's LEFT side)
LEFT_INDEX = 19
LEFT_THUMB = 21
LEFT_HIP = 23

# "mono": one webcam. 3D joints come from MediaPipe's own estimate (pose_world_landmarks)
# "stereo": two webcams + Charuco calibration (run `main.py calibrate` first)
CAMERA_MODE = "mono"
MONO_CAMERA_INDEX = 0 
STEREO_CAMERA_INDICES = (0, 1)  # (left, right)
# 0: logitec
# 1: mac's built-in cam

SMOOTH_WINDOW = 4

# Serial port for live streaming
DEFAULT_PORT = "/dev/cu.usbserial-A5069RR4"


class VisionWorker(QThread):
    # Signals to send images back to gui_main.py
    left_pixmap_signal = Signal(QImage)
    right_pixmap_signal = Signal(QImage)

    def __init__(self, port=DEFAULT_PORT):
        super().__init__()
        self._run_flag = True
        self._is_recording = False
        self._live_send_en = False
        self._recorded_frames = []
        self._port = port

        if CAMERA_MODE == "stereo":
            self.cap_left = cv2.VideoCapture(STEREO_CAMERA_INDICES[0])
            self.cap_right = cv2.VideoCapture(STEREO_CAMERA_INDICES[1])
        else:
            self.cap = cv2.VideoCapture(MONO_CAMERA_INDEX)

        # Keep serial port closed until Live Mode is toggled ON so Playback can use it freely
        self.sender = None

        # Last known smoothed angles (held when tracking is lost)
        self._angles = {"shoulder": 0.0, "wrist": 0.0, "elbow": 0.0, "hand": 0.0, "hip": 0.0}
        self._angle_history = {k: deque(maxlen=SMOOTH_WINDOW) for k in self._angles}

    # --- GUI control methods ------------------------------------------------
    def live_streaming(self, enabled):
        self._live_send_en = enabled
        if enabled:
            if self.sender is None:
                self.sender = AngleSender(port=self._port, debug=False)
        else:
            if self.sender is not None:
                self.sender.close()
                self.sender = None

    def start_recording(self):
        self._recorded_frames = []
        self._is_recording = True

    def stop_recording(self):
        self._is_recording = False

    def clear_recorded_frames(self):
        self._recorded_frames.clear()

    def save_data(self, filename):
        """Saves the recorded frames to the file chosen in the GUI."""
        if not self._recorded_frames:
            print("No frames to save.")
            return False
        with open(filename, "w") as f:
            json.dump(self._recorded_frames, f, indent=4)
        print(f"Saved {len(self._recorded_frames)} frames to {filename}!")
        self._recorded_frames.clear()
        return True

    def stop(self):
        self._run_flag = False
        self._live_send_en = False
        self.wait()
        if self.sender is not None:
            self.sender.close()
            self.sender = None

    # --- Main loop ----------------------------------------------------------
    def run(self):
        parser = argparse.ArgumentParser()
        StereoConfig.add_args(parser)
        config = StereoConfig.from_args(parser.parse_known_args([])[0])

        if CAMERA_MODE == "stereo":
            self._run_stereo(config)
        else:
            self._run_mono(config)

    def _run_mono(self, config):
        holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        try:
            while self._run_flag:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                _, draw_xy, results = extract_pose_pixels(holistic, frame, config.visibility)

                points_3d = {}
                if results.pose_world_landmarks is not None:
                    for i, lm in enumerate(results.pose_world_landmarks.landmark):
                        if lm.visibility > config.visibility:
                            points_3d[i] = np.array([lm.x, lm.y, lm.z])

                self._process(points_3d)

                vis = draw_pose_overlay(frame, draw_xy, points_3d)
                self._emit_frame(vis, self.left_pixmap_signal)
                #self._emit_frame(self._draw_angles(frame.copy()), self.right_pixmap_signal)
                self._emit_frame(vis, self.right_pixmap_signal)

        finally:
            holistic.close()
            self.cap.release()

    def _run_stereo(self, config):
        ret_l, f_l = self.cap_left.read()
        ret_r, f_r = self.cap_right.read()
        sizes = (
            (f_l.shape[1], f_l.shape[0]) if ret_l else (640, 480),
            (f_r.shape[1], f_r.shape[0]) if ret_r else (640, 480),
        )
        
        calib = load_or_guess_calibration(config, assume_parallel=True, sizes=sizes)

        left_h = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        right_h = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        try:
            while self._run_flag:
                ret_l, frame_l = self.cap_left.read()
                ret_r, frame_r = self.cap_right.read()
                if not (ret_l and ret_r):
                    time.sleep(0.01)
                    continue

                left_xy, left_draw, _ = extract_pose_pixels(left_h, frame_l, config.visibility)
                right_xy, right_draw, _ = extract_pose_pixels(right_h, frame_r, config.visibility)
                points_3d = triangulate_matched(calib, left_xy, right_xy)

                self._process(points_3d)

                self._emit_frame(
                    draw_pose_overlay(frame_l, left_draw, points_3d), self.left_pixmap_signal
                )
                self._emit_frame(
                    draw_pose_overlay(frame_r, right_draw, points_3d), self.right_pixmap_signal
                )
        finally:
            left_h.close()
            right_h.close()
            self.cap_left.release()
            self.cap_right.release()

    # --- Angles, smoothing, live send, recording ----------------------------
    def _smooth_angle(self, joint_name, raw_angle):
        history = self._angle_history[joint_name]
        history.append(float(raw_angle))
        return sum(history) / len(history)

    def _process(self, p):
        """Continuously update & smooth angles; stream live and/or record as enabled."""
        a = self._angles
        if LEFT_SHOULDER in p and LEFT_ELBOW in p:
            a["shoulder"] = self._smooth_angle(
                "shoulder", calculate_shoulder_flexion(p[LEFT_SHOULDER], p[LEFT_ELBOW])
            )
            if LEFT_HIP in p:
                a["hip"] = self._smooth_angle(
                    "hip", calculate_3d_angle(p[LEFT_HIP], p[LEFT_SHOULDER], p[LEFT_ELBOW])
                )
            if LEFT_WRIST in p:
                a["elbow"] = self._smooth_angle(
                    "elbow", calculate_3d_angle(p[LEFT_SHOULDER], p[LEFT_ELBOW], p[LEFT_WRIST])
                )
                if LEFT_INDEX in p:
                    a["wrist"] = self._smooth_angle(
                        "wrist", calculate_3d_angle(p[LEFT_ELBOW], p[LEFT_WRIST], p[LEFT_INDEX])
                    )
                    if LEFT_THUMB in p:
                        a["hand"] = self._smooth_angle(
                            "hand",
                            calculate_3d_angle(p[LEFT_WRIST], p[LEFT_INDEX], p[LEFT_THUMB]),
                        )

        # Non-blocking live teleoperation bypass
        if self._live_send_en and self.sender is not None:
            self.sender.send_live(
                a["shoulder"], a["wrist"], a["elbow"], a["hand"], a["hip"]
            )

        # Save to memory buffer only while recording
        if self._is_recording:
            self._recorded_frames.append({
                "time": datetime.datetime.now().isoformat(),
                **{k: float(v) for k, v in a.items()},
            })

    def _draw_angles(self, frame):
        """Right panel: live angle read-out."""
        y = 40
        if self._is_recording and self._live_send_en:
            label, color = "LIVE + REC", (0, 0, 255)
        elif self._is_recording:
            label, color = "REC", (0, 0, 255)
        elif self._live_send_en:
            label, color = "LIVE", (0, 200, 0)
        else:
            label, color = "IDLE", (200, 200, 200)

        cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        for name, val in self._angles.items():
            y += 40
            cv2.putText(
                frame,
                f"{name:9s} {val:6.1f} deg",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        return frame

    def _emit_frame(self, frame, signal):
        """Converts OpenCV images to PySide6 QImages."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        signal.emit(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())