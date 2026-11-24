# Vision Arm
A robotic arm that mimics the movements of a human's arm using computer vision.

## Getting Started
The project has several components that need to be running:

### Computer Vision
1. Clone the repository
2. Install `uv` by following [their documentation](https://github.com/astral-sh/uv#installation)
3. Run `uv run python main.py <args>`

`<args>` should be any of the following:
- `calibrate`: Calibrate the cameras with the charuco board
- `track` or `stereo`: Start dual-camera 3D stereo tracking and send all 5 joint angles (`HIP=0`, `HAND=1`, `ELBOW=2`, `WRIST=3`, `SHOULDER=4`) to the STM32 over UART
- `mono`: Start with a single camera (mostly for debugging). The left shoulder and elbow angles are sent to the STM32 over UART.

Both `stereo` and `mono` automatically discover the USB-UART adapter and open it at 115200 8N1 (3-byte big-endian frames `[joint_id, ccr_hi, ccr_lo]`, mapping `0° -> CCR 8000` to `180° -> CCR 1600`). On macOS that is the only `/dev/cu.usb*` or `/dev/cu.wchusbserial*` device. If none is listed, tracking warns on stderr and keeps running with UART off. On Linux it is the only `/dev/ttyUSB*` device, or `/dev/ttyUSB0` when that listing is empty. If several adapters match, the command prints them and exits. Pass `--port` to choose one (or `--no-uart` in stereo mode to force UART off):

```
uv run python main.py stereo --port /dev/cu.usbserial-10
```

Joint IDs match `stm32/Core/Inc/main.h`: Hip `0`, Hand `1`, Elbow `2`, Wrist `3`, Shoulder `4`. If the port cannot be opened, the camera loop still runs and UART stays off. In stereo mode, UART sending starts disabled for safety: plug in the cable at any time, press `I` to connect/reconnect to the UART device, and press `U` to enable or disable sending joint commands.

### STM32
The following STM32 toolchain setup is for Linux systems. A Linux system is recommended, but MacOS or Windows systems should work with some tinkering. (WSL is useful for Windows systems.)
1. Install the following packages (for a Debian-based system):
    - `gcc-arm-none-eabi`
    - `binutils-arm-none-eabi`
    - `cmake`
    - `make`
    - `openocd`
2. Navigate to the `stm32` directory
3. Create the build directory with `cmake -B build -S .`
4. Compile the microcontroller code with `cmake --build build` (this can be skipped if you also want to flash the STM32)
5. Flash the microcontroller with `cmake --build build --target flash`.
