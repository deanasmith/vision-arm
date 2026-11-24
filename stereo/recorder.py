"""Compact per-frame stereo session recorder for offline replay and solver tuning."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from stereo.geometry import StereoCalibration
from stereo.tracker import StereoFrame, UPPER_BODY_INDICES

NUM_POSE_LANDMARKS = 33
_SOURCE_CODE = {"missing": 0, "stereo": 1, "ray+bone": 2, "hold": 3}


class SessionRecorder:
    """Buffers raw dual-camera MediaPipe landmarks and solved 3D poses into a single .npz."""

    def __init__(self, calib: StereoCalibration, log_dir: str | Path = "logs", flush_every: int = 150):
        self.calib = calib
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"session_{stamp}.npz"
        self.flush_every = flush_every

        self._ts: list[float] = []
        self._skew: list[float] = []
        self._cam0_xyv: list[np.ndarray] = []
        self._cam1_xyv: list[np.ndarray] = []
        self._cam0_world: list[np.ndarray] = []
        self._cam1_world: list[np.ndarray] = []
        self._solved_rig: list[np.ndarray] = []
        self._solved_src: list[np.ndarray] = []
        self._hypothesis: list[int] = []

    def append(
        self,
        timestamp: float,
        skew_ms: float,
        cam0_xyv: np.ndarray,
        cam1_xyv: np.ndarray,
        cam0_world: np.ndarray,
        cam1_world: np.ndarray,
        stereo: StereoFrame,
    ) -> None:
        self._ts.append(float(timestamp))
        self._skew.append(float(skew_ms))
        self._cam0_xyv.append(np.asarray(cam0_xyv, dtype=np.float32))
        self._cam1_xyv.append(np.asarray(cam1_xyv, dtype=np.float32))
        self._cam0_world.append(np.asarray(cam0_world, dtype=np.float32))
        self._cam1_world.append(np.asarray(cam1_world, dtype=np.float32))

        pts = np.full((len(UPPER_BODY_INDICES), 3), np.nan, dtype=np.float32)
        src = np.zeros(len(UPPER_BODY_INDICES), dtype=np.uint8)
        for k, idx in enumerate(UPPER_BODY_INDICES):
            j = stereo.joints.get(idx)
            if j is not None:
                pts[k] = j.xyz_rig.astype(np.float32)
                src[k] = _SOURCE_CODE.get(j.source, 1)
        self._solved_rig.append(pts)
        self._solved_src.append(src)
        self._hypothesis.append(1 if stereo.hypothesis == "swap" else 0)

        if self.flush_every > 0 and len(self._ts) % self.flush_every == 0:
            self.save()

    def save(self) -> Path | None:
        if not self._ts:
            return None
        tmp_path = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            timestamps=np.asarray(self._ts, dtype=np.float64),
            skew_ms=np.asarray(self._skew, dtype=np.float32),
            cam0_xyv=np.stack(self._cam0_xyv, axis=0),
            cam1_xyv=np.stack(self._cam1_xyv, axis=0),
            cam0_world=np.stack(self._cam0_world, axis=0),
            cam1_world=np.stack(self._cam1_world, axis=0),
            solved_rig=np.stack(self._solved_rig, axis=0),
            solved_source=np.stack(self._solved_src, axis=0),
            upper_body_indices=np.asarray(UPPER_BODY_INDICES, dtype=np.int32),
            hypothesis=np.asarray(self._hypothesis, dtype=np.uint8),
            K1=self.calib.K1,
            D1=self.calib.D1,
            K2=self.calib.K2,
            D2=self.calib.D2,
            R=self.calib.R,
            T=self.calib.T,
            image_size1=np.asarray(self.calib.image_size1, dtype=np.int32),
            image_size2=np.asarray(self.calib.image_size2, dtype=np.int32),
            baseline_m=np.asarray(self.calib.baseline_m, dtype=np.float64),
            rms=np.asarray(self.calib.rms, dtype=np.float64),
        )
        tmp_path.replace(self.path)
        return self.path
