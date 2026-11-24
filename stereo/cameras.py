from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


def _open_capture(
    index: int,
    width: int | None = None,
    height: int | None = None,
) -> cv2.VideoCapture:
    api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width is not None and height is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


class _CameraGrabber:
    """Continuously drains one camera and keeps a short timestamped ring buffer."""

    def __init__(self, cap: cv2.VideoCapture, name: str, history: int = 4):
        self._cap = cap
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._history: list[tuple[np.ndarray, float, int]] = []
        self._history_max = max(1, history)
        self._frame: np.ndarray | None = None
        self._ts: float = 0.0
        self._seq: int = 0
        self._failed: bool = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    with self._cond:
                        self._failed = True
                        self._cond.notify_all()
                    return
                time.sleep(0.005)
                continue
            consecutive_failures = 0
            now = time.monotonic()
            with self._cond:
                self._frame = frame
                self._ts = now
                self._seq += 1
                self._history.append((frame, now, self._seq))
                if len(self._history) > self._history_max:
                    self._history.pop(0)
                self._cond.notify_all()

    def latest(self, wait_for_first: bool = True, timeout: float = 2.0) -> tuple[np.ndarray | None, float, int]:
        with self._cond:
            if wait_for_first and self._frame is None and not self._failed:
                self._cond.wait(timeout=timeout)
            if self._failed or self._frame is None:
                return None, self._ts, self._seq
            return self._frame.copy(), self._ts, self._seq

    def wait_next(self, after_seq: int, timeout: float = 0.25) -> tuple[np.ndarray | None, float, int]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= after_seq and not self._failed and not self._stop.is_set():
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                self._cond.wait(timeout=rem)
            if self._failed or self._frame is None:
                return None, self._ts, self._seq
            return self._frame.copy(), self._ts, self._seq

    def recent_snapshot(self, min_seq: int = 0) -> list[tuple[np.ndarray, float, int]]:
        with self._cond:
            if self._failed:
                return []
            items = [(f, ts, seq) for (f, ts, seq) in self._history if seq > min_seq]
            if not items and self._frame is not None:
                items = [(self._frame, self._ts, self._seq)]
            return items

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class DualCameras:
    """Read a time-adjacent frame pair from two camera indices."""

    def __init__(
        self,
        left_index: int,
        right_index: int,
        *,
        width: int | None = None,
        height: int | None = None,
        threaded: bool = False,
    ):
        if left_index == right_index:
            raise ValueError("Left and right camera indices must be different")
        self.left_index = left_index
        self.right_index = right_index
        self._threaded = threaded
        self._grab_l: _CameraGrabber | None = None
        self._grab_r: _CameraGrabber | None = None
        self._last_seq_l = 0
        self._last_seq_r = 0

        self.left = _open_capture(left_index, width, height)
        self.right = _open_capture(right_index, width, height)
        if not self.left.isOpened():
            self.release()
            raise RuntimeError(f"Could not open left camera index {left_index}")
        if not self.right.isOpened():
            self.release()
            raise RuntimeError(f"Could not open right camera index {right_index}")

        if threaded:
            self._grab_l = _CameraGrabber(self.left, f"cam-{left_index}")
            self._grab_r = _CameraGrabber(self.right, f"cam-{right_index}")

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._threaded:
            frame_l, frame_r, _skew_ms = self.read_synced()
            return frame_l, frame_r
        ok_l = self.left.grab()
        ok_r = self.right.grab()
        if not ok_l or not ok_r:
            return None, None
        ok_l, frame_l = self.left.retrieve()
        ok_r, frame_r = self.right.retrieve()
        return (frame_l if ok_l else None, frame_r if ok_r else None)

    def read_synced(self, timeout: float = 2.0) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """Return (frame_l, frame_r, skew_ms) using the freshest synchronized pair."""
        if not self._threaded or self._grab_l is None or self._grab_r is None:
            t0 = time.monotonic()
            frame_l, frame_r = self.read()
            dt_ms = (time.monotonic() - t0) * 1000.0
            return frame_l, frame_r, dt_ms

        if self._last_seq_l == 0 or self._last_seq_r == 0:
            fl, tsl, seql = self._grab_l.latest(wait_for_first=True, timeout=timeout)
            fr, tsr, seqr = self._grab_r.latest(wait_for_first=True, timeout=timeout)
        else:
            # Ensure both cameras have produced at least one fresh frame since last call.
            fl, tsl, seql = self._grab_l.wait_next(self._last_seq_l, timeout=0.15)
            fr, tsr, seqr = self._grab_r.wait_next(self._last_seq_r, timeout=0.15)
            if fl is None or fr is None:
                return None, None, float("nan")
            # Pick the closest-in-time recent pair, preferring the freshest pair when tied.
            hist_l = self._grab_l.recent_snapshot(min_seq=self._last_seq_l)
            hist_r = self._grab_r.recent_snapshot(min_seq=self._last_seq_r)
            if hist_l and hist_r:
                newest_ts = max(hist_l[-1][1], hist_r[-1][1])
                best_score = float("inf")
                best_pair = (fl, tsl, seql, fr, tsr, seqr)
                for cl, ctl, csl in hist_l:
                    for cr, ctr, csr in hist_r:
                        skew = abs(ctl - ctr)
                        age = max(0.0, newest_ts - 0.5 * (ctl + ctr))
                        # Minimize skew while keeping latency low (1 ms skew ≈ 4 ms freshness).
                        score = skew + 0.25 * age
                        if score < best_score:
                            best_score = score
                            best_pair = (cl.copy(), ctl, csl, cr.copy(), ctr, csr)
                fl, tsl, seql, fr, tsr, seqr = best_pair
        self._last_seq_l = seql
        self._last_seq_r = seqr
        skew_ms = abs(tsl - tsr) * 1000.0 if fl is not None and fr is not None else float("nan")
        return fl, fr, skew_ms

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
        if getattr(self, "_grab_l", None) is not None:
            self._grab_l.stop()
            self._grab_l = None
        if getattr(self, "_grab_r", None) is not None:
            self._grab_r.stop()
            self._grab_r = None
        if getattr(self, "left", None) is not None:
            self.left.release()
        if getattr(self, "right", None) is not None:
            self.right.release()

    def __enter__(self) -> DualCameras:
        return self

    def __exit__(self, *exc) -> None:
        self.release()
