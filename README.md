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
- `mono`: Start with a single camera (mostly for debugging)

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
4. Compile the microcontroller code with `cmake --build build`
