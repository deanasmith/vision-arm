# Vision Arm
A robotic arm that mimics the movements of a human's arm using computer vision.

## Getting Started
The project has several components that need to be running:

### Computer Vision
1. Clone the repository
2. Install `uv` by following [their documentation](https://github.com/astral-sh/uv#installation)
3. Run `uv run python main.py <args>`

`<args>` should be any of the following:
- `calibrate`: Calibrate the cameras with the charuko board
- `track`: Start tracking with multiple cameras
- `mono`: Start with a single camera (mostly for debugging)
