import glob
import json
import os
import sys
import time
from datetime import datetime

from PySide6.QtCore import Qt, Slot, QThread, QSize, Signal
from PySide6.QtGui import QImage, QPixmap, QAction, QFontDatabase, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QGraphicsDropShadowEffect,
    QMenu,
)

from vision_worker import VisionWorker
from angle_sender import (
    AngleSender,
    JOINT_IDS,
    REPLY_TIMEOUT_S,
    SLOW_MODE,
    SLOW_GAP_S,
    SLOW_MIN_CHANGE,
)

DEFAULT_PORT = "/dev/cu.usbserial-A5069RR4"


class PlaybackWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(str)

    def __init__(self, json_file, port=DEFAULT_PORT):
        super().__init__()
        self.json_file = json_file
        self.port = port
        self._run_flag = True

    def run(self):
        self.log_signal.emit(f"Loading {self.json_file}...")
        try:
            with open(self.json_file, "r") as f:
                frames = json.load(f)
        except Exception as exc:
            self.finished_signal.emit(f"Failed to load file: {exc}")
            return

        if not frames:
            self.finished_signal.emit("File is empty.")
            return

        sender = AngleSender(port=self.port, debug=False)
        try:
            start_time = time.monotonic()

            if SLOW_MODE:
                # Key poses only: send a joint when it moves >= SLOW_MIN_CHANGE deg,
                # spaced by SLOW_GAP_S seconds per command
                events = []
                last_angle = {}
                for frame in frames:
                    for name in JOINT_IDS:
                        angle = float(frame.get(name, 0))
                        if name not in last_angle or abs(angle - last_angle[name]) >= SLOW_MIN_CHANGE:
                            last_angle[name] = angle
                            events.append((len(events) * SLOW_GAP_S, name, angle))

                # Finish on the recording's final pose
                for name in JOINT_IDS:
                    angle = float(frames[-1].get(name, 0))
                    if name not in last_angle or abs(angle - last_angle[name]) >= 1.0:
                        events.append((len(events) * SLOW_GAP_S, name, angle))

                self.log_signal.emit(
                    f"SLOW playback: {len(frames)} frames -> {len(events)} key commands "
                    f"(~{events[-1][0] if events else 0:.0f} s)"
                )

                reply_timeout = max(REPLY_TIMEOUT_S, SLOW_GAP_S * 0.9)
                for t, name, angle in events:
                    if not self._run_flag:
                        break
                    delay = t - (time.monotonic() - start_time)
                    if delay > 0:
                        time.sleep(delay)
                    sender.send_joint(name, angle, timeout=reply_timeout)
            else:
                # Normal 1.0x speed playback matched to recorded timestamps
                t0 = None
                if "time" in frames[0]:
                    try:
                        t0 = datetime.fromisoformat(frames[0]["time"]).timestamp()
                    except ValueError:
                        t0 = None

                last_ccr = {}
                for i, frame in enumerate(frames):
                    if not self._run_flag:
                        break

                    if t0 is not None and "time" in frame:
                        try:
                            frame_t = datetime.fromisoformat(frame["time"]).timestamp()
                            delay = (frame_t - t0) - (time.monotonic() - start_time)
                        except ValueError:
                            delay = 0.033
                    else:
                        delay = (i * 0.033) - (time.monotonic() - start_time)

                    if delay > 0:
                        time.sleep(delay)

                    for name in JOINT_IDS:
                        if not self._run_flag:
                            break
                        angle = float(frame.get(name, 0))
                        ccr = AngleSender.angle_to_ccr(angle)
                        if last_ccr.get(name) != ccr:
                            if sender.send_joint(name, angle):
                                last_ccr[name] = ccr
                            else:
                                last_ccr.pop(name, None)

            summary = sender.summary()
            self.finished_signal.emit(f"PLAYBACK COMPLETE - {summary}")
        except Exception as exc:
            self.finished_signal.emit(f"Playback Error: {exc}")
        finally:
            sender.close()

    def stop(self):
        self._run_flag = False


class VisionControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Arm")
        self.setFixedSize(1300, 700)
        self.playback_thread = None

        # --- Top Cameras ----------------------------------------------------
        self.left_label = QLabel("Left Camera")
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setStyleSheet("background-color: black; color: white;")
        self.left_label.setScaledContents(True)

        self.right_label = QLabel("Right Camera")
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setStyleSheet("background-color: black; color: white;")
        self.right_label.setScaledContents(True)

        # --- Bottom Left (Logo & Saved Library) -----------------------------
        self.logo_label = QLabel("Vision Arm")
        self.logo_label.setStyleSheet(
            "font-size: 70px; letter-spacing: 15px; font-weight: bold; "
            "color: black; font-family: 'Doto';"
        )

        self.lib_header = QLabel("SAVED MOTION LIBRARY")
        self.lib_header.setStyleSheet("color: black; font-size: 14px; margin-top: 15px;")

        self.library_list = QListWidget()
        self.library_list.setMinimumHeight(100)
        self.library_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #4a86e8;
                border-radius: 2px;
                font-size: 14px;
                color: #555;
            }
            QListWidget::item {
                border-bottom: 1px solid #4a86e8;
                padding: 10px;
            }
            QListWidget::item:selected {
                background-color: #f0f5ff;
                color: black;
                font-weight: bold;
            }
        """)
        self.library_list.itemClicked.connect(self.play_motion_from_list)
        self.library_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.library_list.customContextMenuRequested.connect(self.show_context_menu)

        bottom_left_layout = QVBoxLayout()
        bottom_left_layout.addWidget(self.logo_label)
        bottom_left_layout.addWidget(self.lib_header)
        bottom_left_layout.addWidget(self.library_list)
        bottom_left_layout.setSpacing(5)

        # --- Bottom Right (Description, Status & Controls) ------------------
        welcome_text = (
            "Welcome to the capstone project { Vision-Arm } demo!\n\n"
            "Select a saved motion sequence from the library to play it back on the "
            "robotic arm, toggle LIVE for real-time control, or record your own custom "
            "motion and hit Save to keep it in the library."
        )
        self.desc_label = QLabel(welcome_text)
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("font-size: 20px; color: #666; line-height: 1.5;")

        self.status_box = QWidget()
        self.status_box.setFixedSize(220, 100)
        self.status_box.setStyleSheet("background-color: #afd9e4;")

        self.status_icon = QLabel()
        self.status_icon.setAlignment(Qt.AlignCenter)
        self._set_status_icon("Idle.png")

        status_layout = QVBoxLayout(self.status_box)
        status_layout.addWidget(self.status_icon)
        status_layout.setAlignment(Qt.AlignCenter)

        self.btn_style = """
            QPushButton {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                color: #555;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #f9f9f9; }
            QPushButton:disabled { color: #ccc; }
        """

        self.btn_live = QPushButton("LIVE")
        self.btn_live.setFixedSize(100, 80)
        self.btn_live.setCheckable(True)
        self.btn_live.setStyleSheet(self.btn_style)

        self.btn_start = QPushButton()
        self.btn_start.setFixedSize(100, 80)
        self.btn_start.setStyleSheet(self.btn_style)
        self.btn_start.setIcon(QIcon("recordB.png"))
        self.btn_start.setIconSize(QSize(100, 100))

        self.btn_stop = QPushButton()
        self.btn_stop.setFixedSize(100, 80)
        self.btn_stop.setStyleSheet(self.btn_style)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setIcon(QIcon("saveB.png"))
        self.btn_stop.setIconSize(QSize(80, 80))

        self.btn_start.setGraphicsEffect(self._shadow())
        self.btn_stop.setGraphicsEffect(self._shadow())
        self.btn_live.setGraphicsEffect(self._shadow())
        # self.status_box.setGraphicsEffect(self._shadow())

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.btn_live)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)

        dashboard_layout = QHBoxLayout()
        dashboard_layout.addWidget(self.status_box)
        dashboard_layout.addStretch()
        dashboard_layout.addLayout(controls_layout)

        bottom_right_layout = QVBoxLayout()
        bottom_right_layout.addWidget(self.desc_label)
        bottom_right_layout.addStretch()
        bottom_right_layout.addLayout(dashboard_layout)

        # --- Main Layout Assembly -------------------------------------------
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(self.left_label)
        cam_layout.addWidget(self.right_label)

        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(bottom_left_layout, stretch=1)
        bottom_layout.addSpacing(40)
        bottom_layout.addLayout(bottom_right_layout, stretch=1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(cam_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(bottom_layout)
        main_layout.setContentsMargins(30, 30, 30, 30)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Worker Thread Setup --------------------------------------------
        self.worker = VisionWorker(port=DEFAULT_PORT)
        self.worker.left_pixmap_signal.connect(self.update_left_image)
        self.worker.right_pixmap_signal.connect(self.update_right_image)

        self.btn_live.clicked.connect(self.live_mode)
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)

        self.worker.start()
        self.load_existing_motions()

    def _shadow(self):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(Qt.lightGray)
        shadow.setOffset(0, 4)
        return shadow

    def _set_status_icon(self, filename):
        pixmap = QPixmap(filename)
        if not pixmap.isNull():
            scaled = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.status_icon.setPixmap(scaled)

    # --- Signal Slots -------------------------------------------------------
    @Slot(QImage)
    def update_left_image(self, image):
        self.left_label.setPixmap(QPixmap.fromImage(image))

    @Slot(QImage)
    def update_right_image(self, image):
        self.right_label.setPixmap(QPixmap.fromImage(image))

    # --- Button Actions -----------------------------------------------------
    def live_mode(self, checked):
        if checked:
            # Stop any running playback first so both threads don't fight over UART
            if self.playback_thread and self.playback_thread.isRunning():
                print("Stopping playback to enable Live Mode...")
                self.playback_thread.stop()
                self.playback_thread.wait()
                time.sleep(0.2)

            self.btn_live.setStyleSheet(
                "background-color: #ffcccc; border: 2px solid red; "
                "border-radius: 8px; color: #900; font-size: 16px; font-weight: bold;"
            )
            print("Live Mode Enabled.")
        else:
            self.btn_live.setStyleSheet(self.btn_style)
            print("Live Mode Disabled: Serial port disconnected.")

        self.worker.live_streaming(checked)

    def start_recording(self):
        self.worker.start_recording()
        self._set_status_icon("Recording.png")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_recording(self):
        self.worker.stop_recording()
        print("Recording stopped. Opening save window...")
        self._set_status_icon("Saving.png")

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Motion Sequence",
            "my_custom_motion.json",
            "JSON Files (*.json);;All Files (*)",
        )

        if filename:
            if not filename.endswith(".json"):
                filename += ".json"
            if self.worker.save_data(filename):
                self.add_motion_button(filename)
        else:
            print("Save cancelled. Data was deleted.")
            self.worker.clear_recorded_frames()

        self._set_status_icon("Idle.png")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # --- Library & Playback -------------------------------------------------
    def load_existing_motions(self):
        self.library_list.clear()
        for filepath in sorted(glob.glob("*.json")):
            self.add_motion_button(filepath)

    def add_motion_button(self, filepath):
        filename = os.path.basename(filepath)
        display_name = filename.replace(".json", "")

        # Avoid duplicate entries if overwriting an existing file
        for i in range(self.library_list.count()):
            existing = self.library_list.item(i)
            if existing.text() == display_name:
                existing.setData(Qt.UserRole, filepath)
                return

        item = QListWidgetItem(display_name)
        item.setData(Qt.UserRole, filepath)
        self.library_list.addItem(item)

    def play_motion_from_list(self, item):
        filepath = item.data(Qt.UserRole)

        if self.playback_thread and self.playback_thread.isRunning():
            print("Playback already running - wait for it to finish.")
            return

        if self.btn_live.isChecked():
            print("Disabling Live Mode for playback...")
            self.btn_live.setChecked(False)
            self.live_mode(False)
            time.sleep(0.2)  # let OS release the serial port

        self._set_status_icon("Replaying.png")

        self.playback_thread = PlaybackWorker(filepath, port=DEFAULT_PORT)
        self.playback_thread.log_signal.connect(print)
        self.playback_thread.finished_signal.connect(print)
        self.playback_thread.finished.connect(self.reset_status)
        self.playback_thread.start()

    def reset_status(self):
        self._set_status_icon("Idle.png")
        self.library_list.clearSelection()

    def show_context_menu(self, pos):
        item = self.library_list.itemAt(pos)
        if item:
            menu = QMenu()
            delete_action = QAction("Delete Motion", self)
            delete_action.triggered.connect(
                lambda checked=False, i=item: self.delete_motion(i)
            )
            menu.addAction(delete_action)
            menu.exec(self.library_list.mapToGlobal(pos))

    def delete_motion(self, item):
        filepath = item.data(Qt.UserRole)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted {filepath}")
        self.library_list.takeItem(self.library_list.row(item))

    # --- Cleanup ------------------------------------------------------------
    def closeEvent(self, event):
        self.worker.stop()
        if self.playback_thread and self.playback_thread.isRunning():
            print("Stopping playback before closing...")
            self.playback_thread.stop()
            self.playback_thread.wait()
        event.accept()


def main(argv=None):
    app = QApplication(sys.argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, "Doto-Regular.ttf")
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
        print("Error: could not load Doto font.")
    else:
        family_name = QFontDatabase.applicationFontFamilies(font_id)[0]
        print(f"Successfully loaded custom font: {family_name}")

    window = VisionControlPanel()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())