import json
import struct
import time
import serial

class AngleSender:
    def __init__(self, port='/dev/cu.????', baudrate=115200):
        # Attempt to open serial port to the STM32
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            print(f"Successfully connected to STM32 on {port}.")
        except serial.SerialException:
            print(f"Warning: STM32 not found.")

    def _create_packet(self, shoulder, elbow, wrist):
        """Packs angles into the 8-byte hardware payload"""
        sh = int(round(shoulder))
        el = int(round(elbow))
        wr = int(round(wrist))
        
        # Pack into 16-bit integers (Little Endian)
        packed_angles = struct.pack('<hhh', sh, el, wr)
        checksum = sum(packed_angles) & 0xFF
        
        # [0xFA][SH_L] [SH_H] [EL_L] [EL_H] [WR_L] [WR_H] [CHK]
        return bytes([0xFA]) + packed_angles + bytes([checksum])

    # A: Directly from CV
    def send_live(self, shoulder, elbow, wrist):
        """Call this every single frame from VisionWorker thread"""
        packet = self._create_packet(shoulder, elbow, wrist)
        
        if self.ser and self.ser.is_open:
            self.ser.write(packet)
        else:
            # no STM32 is plugged in => just print the hex to verify
            hex_str = " ".join(f"{b:02X}" for b in packet)
            print(f"LIVE STREAM -> {hex_str}")


    # B: Playback from GUI / JSON
    def send_from_json(self, json_filename, fps=20):
        """Loads a saved sequence and streams it to the STM32 at a controlled frame rate"""
        try:
            with open(json_filename, 'r') as f:
                frames = json.load(f)
        except FileNotFoundError:
            print(f"Error: {json_filename} not found.")
            return

        print(f"Starting playback of {len(frames)} frames...")
        delay = 1.0 / fps
        
        for frame in frames:
            packet = self._create_packet(
                frame.get('shoulder', 0),
                frame.get('elbow', 0),
                frame.get('wrist', 0)
            )
            
            if self.ser and self.ser.is_open:
                self.ser.write(packet)
            else:
                hex_str = " ".join(f"{b:02X}" for b in packet)
                print(f"PLAYBACK -> {hex_str}")
            
            # Pause briefly to avoid flooding the STM32 USART buffer
            time.sleep(delay)
            
        print("Playback complete.")

if __name__ == "__main__":
    sender = AngleSender(port='/dev/cu.????') 
    sender.send_from_json("motion_instructions.json")