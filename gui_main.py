import sys
import cv2
import os
import glob
import json
import time

from PySide6.QtCore import Qt, Slot, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QFileDialog, QScrollArea
from vision_worker import VisionWorker
from angle_sender import AngleSender

class PlaybackWorker(QThread):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        # print(f"\n--- SPooled up STM32 to play: {self.filepath} ---")
        # import time
        # time.sleep(2) # 2-second delay to test the UI
        # print("--- PLAYBACK COMPLETE ---\n")
        print(f"\n---Loading Motion Sequence: {self.filepath} ---")

        try:
            # Load recorded frames
            with open(self.filepath, 'r') as f: 
                frames = json.load(f)

            # Connect to STM32 (USB Port address missing)
            sender = AngleSender(port_name='/dev/cu.????', baudrate=115200)

            # Stream the packets 
            print(f"Streaming {len(frames)} frames to STM32...")
            for frame in frames: 
                # Extract the angles
                shoulder = int(frame.get("shoulder", 0))
                elbow = int(frame.get("elbow", 0))
                wrist = int(frame.get("wrist", 0))

                # Send the hex packet
                sender.send_angles(shoulder, elbow, wrist)

                # Pause slightly to match the original recording speed (~30 fps)
                time.sleep(0.033)
            print("PLAYBACK COMPLETE")

        except Exception as e: 
            print(f"Playback Erorr: {e}")

class VisionControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Arm")
        self.setFixedSize(1300, 700) # Window size

        # --- UI Elements ---
        self.left_label = QLabel("Left Camera")
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setStyleSheet("background-color: black; color: white;")
        self.left_label.setScaledContents(True)

        self.right_label = QLabel("Right Camera")
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setStyleSheet("background-color: black; color: white;")
        self.right_label.setScaledContents(True)

        self.status_label = QLabel("Status: IDLE")
        self.status_label.setAlignment(Qt.AlignCenter)

        # Create the Record buttons
        self.btn_start = QPushButton("Start Recording JSON")
        self.btn_stop = QPushButton("Stop & Save JSON")
        self.btn_stop.setEnabled(False) # Disabled by default until recording starts

        # --- Create the Scrollable Library ---
        self.playback_label = QLabel("Saved Motion Library:")
        self.playback_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(150)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop) # Keeps buttons stacked at the top
        self.scroll_area.setWidget(self.scroll_content)

        # --- Layouts ---
        # Put cameras side-by-side horizontally
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(self.left_label)
        cam_layout.addWidget(self.right_label)

        # Stack everything vertically (Cameras -> Status -> Buttons)
        main_layout = QVBoxLayout()
        main_layout.addLayout(cam_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.btn_start)
        main_layout.addWidget(self.btn_stop)
        main_layout.addWidget(self.playback_label)
        main_layout.addWidget(self.scroll_area)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Worker Thread Setup ---
        self.worker = VisionWorker()
        # Connect worker signals to our GUI update functions
        self.worker.left_pixmap_signal.connect(self.update_left_image)
        self.worker.right_pixmap_signal.connect(self.update_right_image)
        
        # Connect buttons to GUI functions
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        
        # Start the background CV processing
        self.worker.start()

        # Load all existing JSONs
        self.load_existing_motions()

    # --- Signal Slots (Updating the UI) ---
    @Slot(QImage)
    def update_left_image(self, image):
        self.left_label.setPixmap(QPixmap.fromImage(image))

    @Slot(QImage)
    def update_right_image(self, image):
        self.right_label.setPixmap(QPixmap.fromImage(image))

    # --- Button Actions ---
    def start_recording(self):
        self.worker.start_recording() # Tell the worker to start saving data
        self.status_label.setText("Status: RECORDING...")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    # def stop_recording(self):
    #     self.worker.stop_recording() # Tell the worker to dump data to JSON
    #     self.status_label.setText("Status: SAVED TO JSON")
    #     self.btn_start.setEnabled(True)
    #     self.btn_stop.setEnabled(False)

    def stop_recording(self):
        # Stop the background camera thread from saving more frames
        self.worker.stop_recording()
        print("Recording stopped. Opening save window...")
        
        self.status_label.setText("Status: SAVED TO JSON")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        # Open the Mac 'Save As' popup
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Motion Sequence", 
            "my_custom_motion.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        # Handle the saved file
        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            self.worker.save_data(filename)
            self.add_motion_button(filename)
        else:
            print("Save cancelled. Data was deleted.")
            self.worker._recorded_frames.clear() # Clear memory if they cancel

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()
    
    def load_existing_motions(self):
        # Scans the folder on startup and makes a button for every JSON file
        for filepath in glob.glob("*.json"):
            self.add_motion_button(filepath)

    def add_motion_button(self, filepath):
        # Creates a single playback button
        filename = os.path.basename(filepath)
        btn = QPushButton(f"Play: {filename}")

        #Connect the button to play this specific file
        btn.clicked.connect(lambda checked, f=filepath: self.play_motion(f))
        self.scroll_layout.addWidget(btn)
    
    def play_motion(self, filepath):
        # Fires up background thread to send hex data to STM32
        filename = os.path.basename(filepath)
        self.status_label.setText(f"Status: PLAYING {filename}...")

        self.playback_thread = PlaybackWorker(filepath)
        #Resets the status text when the robot finishes moving
        self.playback_thread.finished.connect(lambda: self.status_label.setText("Status: IDLE"))
        self.playback_thread.start()

def main(argv=None):
    app = QApplication(sys.argv)
    window = VisionControlPanel()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())