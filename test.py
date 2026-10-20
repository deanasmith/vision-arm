import struct

def simulate_stm32(hex_file):
    with open(hex_file, 'r') as f:
        lines = f.readlines()
        
    print(f"Testing the first 3 packets from {hex_file}...\n")
    
    for i, line in enumerate(lines[:3]): 
        # 1. Convert the text hex string back into real raw bytes
        packet_bytes = bytes.fromhex(line.strip())
        
        # 2. Slice the bytes back into their components
        start_byte = packet_bytes[0]
        angle_bytes = packet_bytes[1:7]
        checksum = packet_bytes[7]
        
        # 3. Unpack the bytes back into three integers ('<hhh' means little-endian, 3 shorts)
        shoulder, elbow, wrist = struct.unpack('<hhh', angle_bytes)
        
        print(f"Packet {i+1}: {line.strip()}")
        print(f"  -> Start Marker : {hex(start_byte)}")
        print(f"  -> Decoded      : Shoulder={shoulder}°, Elbow={elbow}°, Wrist={wrist}°")
        print(f"  -> Checksum     : {checksum}\n")

if __name__ == "__main__":
    simulate_stm32("stm32_commands.hex")