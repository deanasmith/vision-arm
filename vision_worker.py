import argparse
import cv2
import json
import datetime
import mediapipe as mp
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

# Import CV existing logic ---
from stereo.config import StereoConfig
from stereo.cameras import DualCameras
from stereo.geometry import triangulate_matched
from stereo.overlay import draw_pose_overlay, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
from stereo.kinematics import calculate_3d_angle, calculate_shoulder_flexion
from stereo_pose import extract_pose_pixels, load_or_guess_calibration
from angle_sender import AngleSender

LEFT_INDEX = 19

class VisionWorker(QThread):
    # Signals to send images back to gui_main.py
    left_pixmap_signal = Signal(QImage)
    right_pixmap_signal = Signal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._is_recording = False
        self._recorded_frames = [] # List to hold our angle data
        # self.cap = cv2.VideoCapture(0)
        self.cap_left = cv2.VideoCapture(1)  # First ext cam
        self.cap_right = cv2.VideoCapture(2) # Second ext cam
        self.sender = AngleSender()

        # Memory for when the camera loses tracking
        self._last_sh = 0.0
        self._last_el = 0.0
        self._last_wr = 0.0

    # GUI Control Methods ---
    def start_recording(self):
        self._recorded_frames = [] # Clear old data
        self._is_recording = True  # Enable logging flag

    def stop_recording(self):
        self._is_recording = False # Stop logging flag
        # Dump all recorded frames to a JSON file
        # with open("motion_instructions.json", "w") as f:
        #     json.dump(self._recorded_frames, f, indent=4)
        # print(f"Saved {len(self._recorded_frames)} frames to motion_instructions.json!")

    def save_data(self, filename):
        """Saves the recorded frames to the file specified by the GUI."""
        if not self._recorded_frames:
            print("No frames to save.")
            return

        with open(filename, 'w') as f:
            json.dump(self._recorded_frames, f, indent=4)
        
        print(f"Saved {len(self._recorded_frames)} frames to {filename}!")
        
        # Clear the memory so it's fresh for the next recording
        self._recorded_frames.clear()

    # Main CV Loop ---
    ##
    def run(self):
        # Load config 
        parser = argparse.ArgumentParser()
        StereoConfig.add_args(parser)
        args = parser.parse_known_args([])[0]
        config = StereoConfig.from_args(args)

        mp_holistic = mp.solutions.holistic

        
        # Grab one test frame to get the image size for the calibration logic
        ## ret, test_frame = self.cap.read()
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()


        if not ret_left or not ret_right:
            print("Camera disconnected.")
            break

        if ret_left:
            h, w, _ = test_frame.shape
            sizes = ((w, h), (w, h))
        else:
            sizes = ((640, 480), (640, 480)) # Fallback if camera is slow to warm up

        # Load Charuco calibration using the mocked sizes
        calib = load_or_guess_calibration(config, assume_parallel=True, sizes=sizes) 

        left_holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        right_holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        try:
            while self._run_flag:
                # Read frame from local webcam
                # ret, frame = self.cap.read()
                ret_l, frame_l = self.cap_left.read()
                ret_r, frame_r = self.cap_right.read()

                if not ret_l or not ret_r: 
                    continue

                # Extract 2D pixels
                left_xy, left_draw, _ = extract_pose_pixels(left_holistic, frame_l, config.visibility)
                right_xy, right_draw, _ = extract_pose_pixels(right_holistic, frame_r, config.visibility)
                
                # Triangulate to find 3D XYZ coordinates
                points_3d = triangulate_matched(calib, left_xy, right_xy)

                # print(f"Visible joints: {list(points_3d.keys())}")
                
                
                # JSON Recording Logic
                if self._is_recording:
                    # Only calculate NEW angles if the joints are visible
                    if LEFT_SHOULDER in points_3d and LEFT_ELBOW in points_3d:
                        self._last_sh = calculate_shoulder_flexion(points_3d[LEFT_SHOULDER], points_3d[LEFT_ELBOW])
                        
                        if LEFT_WRIST in points_3d:
                            self._last_el = calculate_3d_angle(points_3d[LEFT_SHOULDER], points_3d[LEFT_ELBOW], points_3d[LEFT_WRIST])
                            
                            if LEFT_INDEX in points_3d:
                                self._last_wr = calculate_3d_angle(points_3d[LEFT_ELBOW], points_3d[LEFT_WRIST], points_3d[LEFT_INDEX])
                    
                    # ALWAYS send the packet (uses new angles if visible, or holds steady on old angles if lost)
                    self.sender.send_live(self._last_sh, self._last_el, self._last_wr)
                    
                    # ALWAYS save the frame
                    current_time = datetime.datetime.now().isoformat()
                    self._recorded_frames.append({
                        "time": current_time,
                        "shoulder": float(self._last_sh),
                        "elbow": float(self._last_el),
                        "wrist": float(self._last_wr)
                    })

                # Draw overlays (skeleton lines)
                vis_l = draw_pose_overlay(frame_l, left_draw, points_3d)
                vis_r = draw_pose_overlay(frame_r, right_draw, points_3d)

                # Send finished frames to GUI
                self._emit_frame(vis_l, self.left_pixmap_signal)
                self._emit_frame(vis_r, self.right_pixmap_signal)
        
        finally:
            # Clean up when the thread stops
            left_holistic.close()
            right_holistic.close()
            self.cap.release() # Turn off the Mac webcam green light
        #

    def _emit_frame(self, frame, signal):
        """Converts OpenCV images to PySide6 QImages."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        signal.emit(qt_image)

    def stop(self):
        self._run_flag = False
        self.wait()