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
- `track`: Start tracking with multiple cameras
- `mono`: Start with a single camera (mostly for debugging). The left shoulder and elbow angles are sent to the STM32 over UART.

`uv run python main.py mono` discovers the USB-UART adapter and opens it at 115200 8N1. On macOS that is the only `/dev/cu.usb*` or `/dev/cu.wchusbserial*` device. If none is listed, mono warns on stderr and keeps the camera running with UART off. On Linux it is the only `/dev/ttyUSB*` device, or `/dev/ttyUSB0` when that listing is empty. If several adapters match, the command prints them and exits. Pass `--port` to choose one:

```
uv run python main.py mono --port /dev/cu.usbserial-10
```

Shoulder is joint id 4 and elbow is joint id 2. If the port cannot be opened, the camera loop still runs and UART stays off.

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
