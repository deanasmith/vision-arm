from __future__ import annotations

import sys

import cv2
import numpy as np


def _open_capture(index: int) -> cv2.VideoCapture:
    api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


class DualCameras:
    """Read a time-adjacent frame pair from two camera indices."""

    def __init__(self, left_index: int, right_index: int):
        if left_index == right_index:
            raise ValueError("Left and right camera indices must be different")
        self.left_index = left_index
        self.right_index = right_index
        self.left = _open_capture(left_index)
        self.right = _open_capture(right_index)
        if not self.left.isOpened():
            self.release()
            raise RuntimeError(f"Could not open left camera index {left_index}")
        if not self.right.isOpened():
            self.release()
            raise RuntimeError(f"Could not open right camera index {right_index}")

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        ok_l, frame_l = self.left.read()
        ok_r, frame_r = self.right.read()
        return (frame_l if ok_l else None, frame_r if ok_r else None)

    def image_sizes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return ((width, height), ...) from camera properties, falling back to a grab."""
        def size(cap: cv2.VideoCapture) -> tuple[int, int] | None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
            return None

        left_size = size(self.left)
        right_size = size(self.right)
        if left_size is None or right_size is None:
            frame_l, frame_r = self.read()
            if frame_l is None or frame_r is None:
                raise RuntimeError("Could not read frames to determine camera resolution")
            left_size = (frame_l.shape[1], frame_l.shape[0])
            right_size = (frame_r.shape[1], frame_r.shape[0])
        return left_size, right_size

    def release(self) -> None:
        if getattr(self, "left", None) is not None:
            self.left.release()
        if getattr(self, "right", None) is not None:
            self.right.release()

    def __enter__(self) -> DualCameras:
        return self

    def __exit__(self, *exc) -> None:
        self.release()
