import re
import time
import serial

JOINT_IDS = {"shoulder": 4, "wrist": 3, "elbow": 2, "hand": 1, "hip": 0}

# How long to wait for the STM32 to confirm one joint before giving up
# STM prints its debug lines after every 3-byte frame and is NOT
# listening while it prints, so we wait for that reply before sending the next frame
REPLY_TIMEOUT_S = 0.15

# Slow-mode pacing for STM32 firmware (~1 command every 2 s)
SLOW_MODE = False
SLOW_GAP_S = 2.0 # seconds of time gap between any UART packets
SLOW_MIN_CHANGE = 20.0 # minimum degree change before sending in slow mode

RE_JOINT = re.compile(rb"Joint ID is '(\d+)'")
RE_CCR = re.compile(rb"Setting CCR to '(\d+)'")


class AngleSender:
    def __init__(self, port="/dev/cu.usbserial-A5069RR4", baudrate=115200, debug=True):
        self.ser = None
        self.debug = debug
        self._last_ccr = {} # joint name -> last CCR sent (to skip unchanged joints)
        self._last_angle = {} # Remembers the last degree value actually sent for each joint: to measure how far it has moved since then
        self._last_send_time = 0.0 # Stores the exact clock time of the last transmission => can know when 2.0 seconds have passed
        self.stats = {"sent": 0, "ok": 0, "mismatch": 0, "no_reply": 0}

        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.01, write_timeout=0.5)
            time.sleep(0.2)
            self.ser.reset_input_buffer()
            print(f"Successfully connected to STM32 on {port}.")
        except serial.SerialException as exc:
            print(f"Error: STM32 not found ({exc}).")

    @staticmethod
    def angle_to_ccr(angle):  
        """Convert an angle in degrees to a CCR value for the STM32."""
        angle = int(max(0, min(180, angle)))  # [0, 180]
        return int((angle / 180.0) * 6400 + 1600)  # 0deg = 1600, 180deg = 8000

    def joint_frame(self, joint_id, ccr):
        return bytes([joint_id, (ccr >> 8) & 0xFF, ccr & 0xFF])

    def send_joint(self, name, angle, timeout=REPLY_TIMEOUT_S):
        """Send one joint and wait for the STM32 to confirm it. Returns True if confirmed."""
        joint_id = JOINT_IDS[name]
        ccr = AngleSender.angle_to_ccr(angle)
        print(f"[PLAYBACK] {name:8s} -> {angle:6.1f} deg (CCR {ccr})")

        if not (self.ser and self.ser.is_open):
            if self.debug:
                print(f"NO UART: would send {name} -> {angle:.1f} deg")
            return False

        try:
            self.ser.write(self.joint_frame(joint_id, ccr))
        except serial.SerialException as exc:
            print(f"UART write error on {name}: {exc}")
            return False

        self.stats["sent"] += 1

        buf = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            buf += self.ser.read(self.ser.in_waiting or 1)
            j, c = RE_JOINT.search(buf), RE_CCR.search(buf)
            if j and c:
                time.sleep(0.005)  # let STM finish its print
                self.ser.read(self.ser.in_waiting)
                rj, rc = int(j.group(1)), int(c.group(1))
                if rj == joint_id and rc == ccr:
                    self.stats["ok"] += 1
                    if self.debug:
                        print(f"STM32 OK: {name:8s} CCR {ccr}")
                    return True
                self.stats["mismatch"] += 1
                print(f"STM32 MISMATCH: sent {name} CCR {ccr}, board said joint {rj} CCR {rc}")
                return False

        self.stats["no_reply"] += 1
        print(f"STM32 NO REPLY: {name} CCR {ccr} (board may be stuck - press reset)")
        return False

    def stream_joint_fast(self, name, angle):
        """Sends one joint immediately without waiting for a reply."""
        ccr = AngleSender.angle_to_ccr(angle)
        if self._last_ccr.get(name) == ccr:
            return  # skip redundant frames when joint hasn't moved

        print(f"[LIVE]     {name:8s} -> {angle:6.1f} deg (CCR {ccr})")

        if not (self.ser and self.ser.is_open):
            self._last_ccr[name] = ccr
            return

        joint_id = JOINT_IDS[name]
        try:
            # Clear any unread STM32 debug output so the OS input buffer doesn't fill up
            if self.ser.in_waiting:
                self.ser.reset_input_buffer()
            self.ser.write(self.joint_frame(joint_id, ccr))
            self._last_ccr[name] = ccr
            self.stats["sent"] += 1
        except serial.SerialException as exc:
            print(f"Live stream UART error on {name}: {exc}")

    def sending_angles(self, shoulder, wrist, elbow, hand, hip):
        """Send with receipt verification."""
        angles = {"shoulder": shoulder, "wrist": wrist, "elbow": elbow, "hand": hand, "hip": hip}

        if not (self.ser and self.ser.is_open):
            pack = b"".join(
                self.joint_frame(JOINT_IDS[j], AngleSender.angle_to_ccr(a))
                for j, a in angles.items()
            )
            print(f"NO UART: STREAM -> {' '.join(f'{b:02X}' for b in pack)}")
            return

        for name, angle in angles.items():
            ccr = AngleSender.angle_to_ccr(angle)
            if self._last_ccr.get(name) == ccr:
                continue  # unchanged since last frame - don't resend
            if self.send_joint(name, angle):
                self._last_ccr[name] = ccr
            else:
                self._last_ccr.pop(name, None)  # force a resend next frame

    def send_live(self, shoulder, wrist, elbow, hand, hip):
        """Stream live angles. 1 command per SLOW_GAP_S when SLOW_MODE is True."""
        angles = {"shoulder": shoulder, "wrist": wrist, "elbow": elbow, "hand": hand, "hip": hip}

        if not SLOW_MODE:
            for name, angle in angles.items():
                self.stream_joint_fast(name, angle)
            return

        # Slow mode: time gap between commands
        now = time.monotonic() # current clk time
        if now - self._last_send_time < SLOW_GAP_S:
            return # exit func if less than 2 sec passed

        # Send whichever single joint moved the most (at least SLOW_MIN_CHANGE degrees)
        best_joint, best_angle, best_diff = None, None, 0.0
        for name, angle in angles.items():
            prev = self._last_angle.get(name)
            diff = abs(angle - prev) if prev is not None else float("inf")
            if diff >= SLOW_MIN_CHANGE and diff > best_diff:
                best_joint, best_angle, best_diff = name, angle, diff

        if best_joint is not None:
            self.stream_joint_fast(best_joint, best_angle) # send that best joint (single 3-byte packet)
            self._last_angle[best_joint] = best_angle # update the _last_angle for that joint
            self._last_send_time = now # resets to start next 2 sec slowing down

    def summary(self):
        s = self.stats
        verdict = "PASS" if s["mismatch"] == 0 and s["no_reply"] == 0 else "FAIL"
        return (
            f"{verdict}: sent {s['sent']}, confirmed {s['ok']}, "
            f"mismatched {s['mismatch']}, no reply {s['no_reply']}"
        )

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()