from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StereoCalibration:
    """Intrinsics, stereo extrinsics, and metric baseline for a camera pair.

    World / camera-1 frame is OpenCV: X right, Y down, Z forward.
    A 3D point X1 in camera-1 coordinates maps to camera-2 as X2 = R @ X1 + T.
    """

    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size1: tuple[int, int]
    image_size2: tuple[int, int]
    baseline_m: float
    measured_baseline_m: float
    rms: float

    def save(self, path: str | Path) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dest,
            K1=self.K1,
            D1=self.D1,
            K2=self.K2,
            D2=self.D2,
            R=self.R,
            T=self.T,
            image_size1=np.array(self.image_size1),
            image_size2=np.array(self.image_size2),
            baseline_m=self.baseline_m,
            measured_baseline_m=self.measured_baseline_m,
            rms=self.rms,
        )

    @classmethod
    def load(cls, path: str | Path) -> StereoCalibration:
        data = np.load(path, allow_pickle=False)
        return cls(
            K1=data["K1"],
            D1=data["D1"],
            K2=data["K2"],
            D2=data["D2"],
            R=data["R"],
            T=data["T"].reshape(3, 1),
            image_size1=(int(data["image_size1"][0]), int(data["image_size1"][1])),
            image_size2=(int(data["image_size2"][0]), int(data["image_size2"][1])),
            baseline_m=float(data["baseline_m"]),
            measured_baseline_m=float(data["measured_baseline_m"]),
            rms=float(data["rms"]),
        )


def scale_translation_to_baseline(T: np.ndarray, baseline_m: float) -> tuple[np.ndarray, float]:
    """Scale stereo translation so ||T|| equals the configured camera separation."""
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    measured = float(np.linalg.norm(T))
    if measured < 1e-9:
        raise ValueError("Stereo translation is degenerate (zero length)")
    if baseline_m <= 0:
        return T, measured
    return T * (baseline_m / measured), measured


def parallel_camera_calibration(
    image_size1: tuple[int, int],
    image_size2: tuple[int, int],
    baseline_m: float,
    fov_deg: float = 70.0,
) -> StereoCalibration:
    """Approximate calibration when Charuco has not been run yet.

    Assumes parallel image planes with the right camera translated by -baseline
    along X in the left-camera frame (right camera sits to the right).
    """

    def guess_K(size: tuple[int, int]) -> np.ndarray:
        w, h = size
        f = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
        return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    T = np.array([[-baseline_m], [0.0], [0.0]], dtype=np.float64)
    return StereoCalibration(
        K1=guess_K(image_size1),
        D1=np.zeros(5, dtype=np.float64),
        K2=guess_K(image_size2),
        D2=np.zeros(5, dtype=np.float64),
        R=np.eye(3, dtype=np.float64),
        T=T,
        image_size1=image_size1,
        image_size2=image_size2,
        baseline_m=baseline_m,
        measured_baseline_m=baseline_m,
        rms=float("nan"),
    )


def _as_point_list(points: np.ndarray, dims: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    return arr.reshape(-1, 1, dims)


def solve_stereo(
    object_points: list[np.ndarray],
    image_points_left: list[np.ndarray],
    image_points_right: list[np.ndarray],
    image_size1: tuple[int, int],
    image_size2: tuple[int, int],
    baseline_m: float,
) -> StereoCalibration:
    """Estimate intrinsics and stereo extrinsics from paired Charuco observations."""
    if len(object_points) < 3:
        raise ValueError("Need at least 3 stereo Charuco captures to calibrate")

    obj = [_as_point_list(p, 3) for p in object_points]
    img_l = [_as_point_list(p, 2) for p in image_points_left]
    img_r = [_as_point_list(p, 2) for p in image_points_right]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    _rms1, K1, D1, _, _ = cv2.calibrateCamera(obj, img_l, image_size1, None, None, criteria=criteria)
    _rms2, K2, D2, _, _ = cv2.calibrateCamera(obj, img_r, image_size2, None, None, criteria=criteria)

    stereo_rms, K1, D1, K2, D2, R, T, _E, _F = cv2.stereoCalibrate(
        obj,
        img_l,
        img_r,
        K1,
        D1,
        K2,
        D2,
        image_size1,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria,
    )
    T, measured = scale_translation_to_baseline(T, baseline_m)
    return StereoCalibration(
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        image_size1=image_size1,
        image_size2=image_size2,
        baseline_m=baseline_m,
        measured_baseline_m=measured,
        rms=float(stereo_rms),
    )


def triangulate_pixels(calib: StereoCalibration, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Triangulate corresponding pixel coordinates into camera-1 XYZ (meters).

    pts1 / pts2 are (N, 2) pixel locations. Distortion is removed first, then
    points are triangulated with P1 = K1[I|0] and P2 = K2[R|T].
    """
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts1) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    und1 = cv2.undistortPoints(pts1, calib.K1, calib.D1, P=calib.K1).reshape(-1, 2).T
    und2 = cv2.undistortPoints(pts2, calib.K2, calib.D2, P=calib.K2).reshape(-1, 2).T
    P1 = calib.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = calib.K2 @ np.hstack([calib.R, calib.T.reshape(3, 1)])
    homog = cv2.triangulatePoints(P1, P2, und1, und2)
    w = homog[3]
    w = np.where(np.abs(w) < 1e-12, np.copysign(1e-12, w), w)
    return (homog[:3] / w).T


def triangulate_matched(
    calib: StereoCalibration,
    left_xy: dict[int, tuple[float, float]],
    right_xy: dict[int, tuple[float, float]],
) -> dict[int, np.ndarray]:
    """Triangulate landmarks observed in both cameras. Keys are landmark indices."""
    ids = sorted(set(left_xy) & set(right_xy))
    if not ids:
        return {}
    pts1 = np.array([left_xy[i] for i in ids], dtype=np.float64)
    pts2 = np.array([right_xy[i] for i in ids], dtype=np.float64)
    xyz = triangulate_pixels(calib, pts1, pts2)
    points: dict[int, np.ndarray] = {}
    for i, pt in zip(ids, xyz):
        if not np.all(np.isfinite(pt)):
            continue
        x2 = calib.R @ pt.reshape(3, 1) + calib.T.reshape(3, 1)
        if pt[2] <= 0 or float(x2[2, 0]) <= 0:
            continue
        points[i] = pt
    return points


def project_points(calib: StereoCalibration, xyz: np.ndarray, camera: int) -> np.ndarray:
    """Project camera-1-frame points into left (1) or right (2) pixels."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if camera == 1:
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        K, D = calib.K1, calib.D1
        pts = xyz
    else:
        rvec, _ = cv2.Rodrigues(calib.R)
        tvec = calib.T.reshape(3)
        K, D = calib.K2, calib.D2
        pts = xyz
    projected, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    return projected.reshape(-1, 2)


def mean_reprojection_error(
    calib: StereoCalibration,
    xyz: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> float:
    if len(xyz) == 0:
        return float("nan")
    err1 = np.linalg.norm(project_points(calib, xyz, 1) - np.asarray(pts1).reshape(-1, 2), axis=1)
    err2 = np.linalg.norm(project_points(calib, xyz, 2) - np.asarray(pts2).reshape(-1, 2), axis=1)
    return float(np.mean(np.concatenate([err1, err2])))
