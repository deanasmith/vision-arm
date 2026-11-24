"""3D One-Euro temporal smoothing and rolling metric stability tracking."""

from __future__ import annotations

from collections import deque
import math

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class _OneEuroChannel:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float = 0.0

    def update(self, value: np.ndarray, timestamp: float, max_jump: float, max_gap: float) -> np.ndarray:
        val = np.asarray(value, dtype=np.float64)
        if self.x is None:
            self.x = val.copy()
            self.dx = np.zeros_like(val)
            self.t = timestamp
            return self.x.copy()

        dt = timestamp - self.t
        if dt <= 0.0:
            return self.x.copy()
        if dt > max_gap or float(np.linalg.norm(val - self.x)) > max_jump:
            self.x = val.copy()
            self.dx = np.zeros_like(val)
            self.t = timestamp
            return self.x.copy()

        a_d = _alpha(self.d_cutoff, dt)
        raw_dx = (val - self.x) / dt
        self.dx = a_d * raw_dx + (1.0 - a_d) * self.dx

        speed = float(np.linalg.norm(self.dx))
        cutoff = self.min_cutoff + self.beta * speed
        a_x = _alpha(cutoff, dt)
        self.x = a_x * val + (1.0 - a_x) * self.x
        self.t = timestamp
        return self.x.copy()


class OneEuro3D:
    """Per-landmark 3D One-Euro filter: steady when still, zero lag when moving."""

    def __init__(
        self,
        min_cutoff: float = 1.4,
        beta: float = 1.5,
        d_cutoff: float = 1.0,
        max_jump_m: float = 0.45,
        max_gap_s: float = 0.35,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_jump_m = max_jump_m
        self.max_gap_s = max_gap_s
        self._channels: dict[int, _OneEuroChannel] = {}

    def filter_point(self, index: int, xyz: np.ndarray, timestamp: float) -> np.ndarray:
        channel = self._channels.get(index)
        if channel is None:
            channel = _OneEuroChannel(self.min_cutoff, self.beta, self.d_cutoff)
            self._channels[index] = channel
        return channel.update(xyz, timestamp, self.max_jump_m, self.max_gap_s)

    def reset(self) -> None:
        self._channels.clear()


class BoneStabilityTracker:
    """Rolling bone-length and depth-history monitor to prove 3D metric consistency."""

    def __init__(self, window: int = 90):
        self.window = window
        self.bones: dict[str, deque[float]] = {
            "L Upper": deque(maxlen=window),
            "L Forearm": deque(maxlen=window),
            "Shoulders": deque(maxlen=window),
            "R Upper": deque(maxlen=window),
            "R Forearm": deque(maxlen=window),
        }
        self.depth_history: dict[str, deque[float]] = {
            "L-Sh": deque(maxlen=window),
            "L-El": deque(maxlen=window),
            "L-Wr": deque(maxlen=window),
        }

    def update(self, points_rig: dict[int, np.ndarray]) -> None:
        pairs = (
            ("L Upper", 11, 13),
            ("L Forearm", 13, 15),
            ("Shoulders", 11, 12),
            ("R Upper", 12, 14),
            ("R Forearm", 14, 16),
        )
        for name, a, b in pairs:
            if a in points_rig and b in points_rig:
                length_cm = float(np.linalg.norm(points_rig[b] - points_rig[a])) * 100.0
                if 5.0 <= length_cm <= 90.0:
                    self.bones[name].append(length_cm)
            else:
                self.bones[name].append(float("nan"))

        for label, idx in (("L-Sh", 11), ("L-El", 13), ("L-Wr", 15)):
            if idx in points_rig:
                self.depth_history[label].append(float(points_rig[idx][2]))
            else:
                self.depth_history[label].append(float("nan"))

    def stats(self, name: str) -> tuple[float, float, float, int] | None:
        """Return (latest_cm, mean_cm, std_cm, valid_count) or None."""
        buf = self.bones.get(name)
        if not buf:
            return None
        valid = [v for v in buf if math.isfinite(v)]
        if not valid:
            return None
        latest = buf[-1] if math.isfinite(buf[-1]) else valid[-1]
        arr = np.asarray(valid, dtype=np.float64)
        return float(latest), float(np.mean(arr)), float(np.std(arr)), len(valid)

    def reset(self) -> None:
        for buf in self.bones.values():
            buf.clear()
        for buf in self.depth_history.values():
            buf.clear()
