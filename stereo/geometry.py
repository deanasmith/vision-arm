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
) -> StereoCalibration:
    """Estimate intrinsics and stereo extrinsics from paired Charuco observations.

    The metric scale is the baseline implied by the board, ||T|| from stereoCalibrate.
    """
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
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    measured = float(np.linalg.norm(T))
    if measured < 1e-9:
        raise ValueError("Stereo translation is degenerate (zero length)")
    return StereoCalibration(
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        image_size1=image_size1,
        image_size2=image_size2,
        baseline_m=measured,
        measured_baseline_m=measured,
        rms=float(stereo_rms),
    )


def fundamental_matrix(calib: StereoCalibration) -> np.ndarray:
    """Pixel fundamental matrix on undistorted coordinates, x2^T F x1 = 0."""
    translation = np.asarray(calib.T, dtype=np.float64).reshape(3)
    tx = np.array(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ],
        dtype=np.float64,
    )
    essential = tx @ np.asarray(calib.R, dtype=np.float64)
    return np.linalg.inv(calib.K2).T @ essential @ np.linalg.inv(calib.K1)


def sampson_distances(calib: StereoCalibration, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Symmetric epipolar error in undistorted pixels.

    A mismatch perpendicular to the epipolar line is large here. A pure
    disparity error runs along that line, so the rays still meet and this
    distance stays near zero while the depth is wrong.
    """
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    if len(pts1) == 0:
        return np.zeros(0, dtype=np.float64)
    und1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), calib.K1, calib.D1, P=calib.K1).reshape(-1, 2)
    und2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), calib.K2, calib.D2, P=calib.K2).reshape(-1, 2)
    matrix = fundamental_matrix(calib)
    hom1 = np.concatenate([und1, np.ones((len(und1), 1))], axis=1)
    hom2 = np.concatenate([und2, np.ones((len(und2), 1))], axis=1)
    line1 = hom1 @ matrix.T
    line2 = hom2 @ matrix
    residual = np.sum(hom2 * line1, axis=1)
    denom = line1[:, 0] ** 2 + line1[:, 1] ** 2 + line2[:, 0] ** 2 + line2[:, 1] ** 2
    return np.sqrt(residual**2 / np.maximum(denom, 1e-18))


def left_ray_direction(calib: StereoCalibration, pixel: np.ndarray) -> np.ndarray:
    """Unit direction, in camera 1, of the ray through a distorted left pixel."""
    normalized = cv2.undistortPoints(
        np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2), calib.K1, calib.D1
    ).reshape(2)
    direction = np.array([float(normalized[0]), float(normalized[1]), 1.0], dtype=np.float64)
    return direction / np.linalg.norm(direction)


def in_front_of_both(calib: StereoCalibration, point: np.ndarray) -> bool:
    """True when the point is in front of both cameras (positive Z)."""
    xyz = np.asarray(point, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(xyz)) or float(xyz[2]) <= 0.0:
        return False
    in_right = calib.R @ xyz.reshape(3, 1) + calib.T.reshape(3, 1)
    return float(in_right[2, 0]) > 0.0


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
    max_reproj_px: float = 15.0,
    max_sampson_px: float = 8.0,
) -> dict[int, np.ndarray]:
    """Triangulate landmarks observed in both cameras. Keys are landmark indices.

    Two gates both have to pass. Reprojection catches a triangulated point that
    misses the measured pixels. It stays near zero when the error is disparity
    along the epipolar line, and it also stays under max_reproj_px for a modest
    miss perpendicular to that line. The Sampson distance rejects that
    off-epipolar miss even when the reprojection residual is only a few pixels.
    """
    ids = sorted(set(left_xy) & set(right_xy))
    if not ids:
        return {}
    pts1 = np.array([left_xy[i] for i in ids], dtype=np.float64)
    pts2 = np.array([right_xy[i] for i in ids], dtype=np.float64)
    xyz = triangulate_pixels(calib, pts1, pts2)
    sampson = sampson_distances(calib, pts1, pts2)
    points: dict[int, np.ndarray] = {}
    for i, pt, p1, p2, epi in zip(ids, xyz, pts1, pts2, sampson):
        if not in_front_of_both(calib, pt):
            continue
        if float(epi) > max_sampson_px:
            continue
        err1 = float(np.linalg.norm(project_points(calib, pt, 1)[0] - p1))
        err2 = float(np.linalg.norm(project_points(calib, pt, 2)[0] - p2))
        if max(err1, err2) > max_reproj_px:
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


@dataclass(frozen=True)
class RigFrame:
    """Right-handed world frame aligned to the physical stereo rig.

    Origin sits halfway between the two camera centers.
    +X points from the physical-left camera toward the physical-right camera.
    +Z points forward along the average optical axis of the two cameras.
    +Y points down (OpenCV convention), so -Y is up.
    """

    origin_cam1: np.ndarray
    R_cam1_to_rig: np.ndarray
    cam1_pos_rig: np.ndarray
    cam2_pos_rig: np.ndarray
    cam1_R_rig: np.ndarray  # columns are cam1's X, Y, Z axes in rig coords
    cam2_R_rig: np.ndarray  # columns are cam2's X, Y, Z axes in rig coords
    cam0_is_physical_left: bool
    convergence_deg: float
    vergence_point_rig: np.ndarray

    def to_rig(self, pts_cam1: np.ndarray) -> np.ndarray:
        arr = np.asarray(pts_cam1, dtype=np.float64)
        single = arr.ndim == 1
        pts = arr.reshape(-1, 3)
        out = (pts - self.origin_cam1.reshape(1, 3)) @ self.R_cam1_to_rig.T
        return out.reshape(3) if single else out

    def to_cam1(self, pts_rig: np.ndarray) -> np.ndarray:
        arr = np.asarray(pts_rig, dtype=np.float64)
        single = arr.ndim == 1
        pts = arr.reshape(-1, 3)
        out = pts @ self.R_cam1_to_rig + self.origin_cam1.reshape(1, 3)
        return out.reshape(3) if single else out

    def dir_to_rig(self, vec_cam1: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec_cam1, dtype=np.float64)
        single = arr.ndim == 1
        vecs = arr.reshape(-1, 3)
        out = vecs @ self.R_cam1_to_rig.T
        return out.reshape(3) if single else out


def rig_frame_from_calibration(calib: StereoCalibration) -> RigFrame:
    """Build a symmetric rig-centered frame directly from calibrated R and T."""
    R = np.asarray(calib.R, dtype=np.float64)
    T = np.asarray(calib.T, dtype=np.float64).reshape(3)
    c1 = np.zeros(3, dtype=np.float64)
    c2 = -(R.T @ T)

    # In camera-1 coordinates (+X right), if c2[0] < 0 then cam2 (index 1) sits
    # physically to the left of cam1 (index 0).
    cam0_is_physical_left = bool(c2[0] >= 0.0)
    baseline_vec = (c2 - c1) if cam0_is_physical_left else (c1 - c2)
    b_norm = float(np.linalg.norm(baseline_vec))
    x_axis = baseline_vec / max(b_norm, 1e-9)

    z1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z2 = R.T @ z1
    z_avg = z1 + z2
    z_avg = z_avg - x_axis * float(np.dot(z_avg, x_axis))
    z_norm = float(np.linalg.norm(z_avg))
    z_axis = z_avg / max(z_norm, 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-9)
    # Ensure +Y points roughly in camera-1's down direction (+Y1).
    if float(np.dot(y_axis, np.array([0.0, 1.0, 0.0]))) < 0.0:
        y_axis = -y_axis
        z_axis = -z_axis

    R_cam1_to_rig = np.vstack([x_axis, y_axis, z_axis])
    origin_cam1 = 0.5 * (c1 + c2)

    cam1_pos_rig = R_cam1_to_rig @ (c1 - origin_cam1)
    cam2_pos_rig = R_cam1_to_rig @ (c2 - origin_cam1)
    cam1_R_rig = R_cam1_to_rig @ np.eye(3, dtype=np.float64)
    cam2_R_rig = R_cam1_to_rig @ R.T

    cos_conv = float(np.clip(np.dot(z1, z2), -1.0, 1.0))
    convergence_deg = float(np.degrees(np.arccos(cos_conv)))

    # Closest approach of the two optical axes (vergence point).
    A = np.column_stack([z1, -z2])
    if np.linalg.cond(A) < 1e6:
        st, *_ = np.linalg.lstsq(A, c2 - c1, rcond=None)
        if st[0] > 0.0 and st[1] > 0.0:
            v_cam1 = 0.5 * (c1 + st[0] * z1 + c2 + st[1] * z2)
        else:
            v_cam1 = origin_cam1 + np.array([0.0, 0.0, 1.2])
    else:
        v_cam1 = origin_cam1 + np.array([0.0, 0.0, 1.2])
    vergence_point_rig = R_cam1_to_rig @ (v_cam1 - origin_cam1)

    return RigFrame(
        origin_cam1=origin_cam1,
        R_cam1_to_rig=R_cam1_to_rig,
        cam1_pos_rig=cam1_pos_rig,
        cam2_pos_rig=cam2_pos_rig,
        cam1_R_rig=cam1_R_rig,
        cam2_R_rig=cam2_R_rig,
        cam0_is_physical_left=cam0_is_physical_left,
        convergence_deg=convergence_deg,
        vergence_point_rig=vergence_point_rig,
    )


@dataclass(frozen=True)
class RayTriangulation:
    """Exact 2-ray midpoint triangulation and per-point stereo diagnostics."""

    xyz_cam1: np.ndarray     # (N, 3)
    ray_gap_m: np.ndarray    # (N,) 3D closest-approach distance between left and right rays
    ray_angle_deg: np.ndarray  # (N,) triangulation parallax angle between the rays
    sampson_px: np.ndarray   # (N,) symmetric epipolar error
    err1_px: np.ndarray      # (N,) reprojection error in camera 1
    err2_px: np.ndarray      # (N,) reprojection error in camera 2
    in_front: np.ndarray     # (N,) bool, positive depth along both rays


def triangulate_with_diagnostics(
    calib: StereoCalibration,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> RayTriangulation:
    """Triangulate corresponding pixels via exact 3D ray closest-approach midpoint.

    Returns the 3D midpoint in camera-1 coordinates along with physical ray-miss
    distance (meters), triangulation parallax angle (degrees), Sampson epipolar
    error, and reprojection errors in both cameras.
    """
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    n = len(pts1)
    if n == 0:
        empty3 = np.zeros((0, 3), dtype=np.float64)
        empty = np.zeros(0, dtype=np.float64)
        return RayTriangulation(empty3, empty, empty, empty, empty, empty, np.zeros(0, dtype=bool))

    # Undistort to normalized camera rays.
    norm1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), calib.K1, calib.D1).reshape(-1, 2)
    norm2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), calib.K2, calib.D2).reshape(-1, 2)

    d1 = np.concatenate([norm1, np.ones((n, 1), dtype=np.float64)], axis=1)
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)

    d2_c2 = np.concatenate([norm2, np.ones((n, 1), dtype=np.float64)], axis=1)
    d2_c2 /= np.linalg.norm(d2_c2, axis=1, keepdims=True)

    R = np.asarray(calib.R, dtype=np.float64)
    T = np.asarray(calib.T, dtype=np.float64).reshape(3)
    c2 = -(R.T @ T)  # camera-2 optical center in camera-1 frame
    d2 = d2_c2 @ R   # (R.T @ d2_c2.T).T

    # Closest approach between ray1(s) = s*d1 and ray2(t) = c2 + t*d2.
    b = np.sum(d1 * d2, axis=1)
    d = d1 @ c2
    e = d2 @ c2
    denom = 1.0 - b * b
    parallel = denom < 1e-10
    safe_denom = np.where(parallel, 1.0, denom)

    s = (d - b * e) / safe_denom
    t = (b * d - e) / safe_denom

    p1_ray = s[:, None] * d1
    p2_ray = c2[None, :] + t[:, None] * d2
    xyz = 0.5 * (p1_ray + p2_ray)
    ray_gap_m = np.linalg.norm(p1_ray - p2_ray, axis=1)
    ray_angle_deg = np.degrees(np.arccos(np.clip(np.abs(b), 0.0, 1.0)))

    # Cheirality: positive distance along both rays and positive Z in both cameras.
    xyz_c2 = (xyz @ R.T) + T[None, :]
    in_front = (~parallel) & (s > 0.0) & (t > 0.0) & (xyz[:, 2] > 0.0) & (xyz_c2[:, 2] > 0.0) & np.all(np.isfinite(xyz), axis=1)

    sampson = sampson_distances(calib, pts1, pts2)
    proj1 = project_points(calib, xyz, 1)
    proj2 = project_points(calib, xyz, 2)
    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)

    return RayTriangulation(
        xyz_cam1=xyz,
        ray_gap_m=ray_gap_m,
        ray_angle_deg=ray_angle_deg,
        sampson_px=sampson,
        err1_px=err1,
        err2_px=err2,
        in_front=in_front,
    )


def epipolar_curve_in_other(
    calib: StereoCalibration,
    pt: tuple[float, float],
    source_camera: int,
    *,
    z_min: float = 0.35,
    z_max: float = 3.50,
    n_samples: int = 24,
) -> np.ndarray:
    """Distorted pixel polyline in the other camera for the ray through ``pt``.

    Sampling points along the 3D ray from z_min to z_max and projecting them
    into the other camera naturally includes lens distortion and clips to the
    realistic workspace depths.
    """
    px = np.asarray(pt, dtype=np.float64).reshape(1, 1, 2)
    R = np.asarray(calib.R, dtype=np.float64)
    T = np.asarray(calib.T, dtype=np.float64).reshape(3)
    zs = np.geomspace(z_min, z_max, n_samples)

    if source_camera == 1:
        norm = cv2.undistortPoints(px, calib.K1, calib.D1).reshape(2)
        pts_cam1 = np.column_stack([norm[0] * zs, norm[1] * zs, zs])
        # Keep only points in front of camera 2 and within its FOV cone.
        pts_c2 = (pts_cam1 @ R.T) + T[None, :]
        ok = (pts_c2[:, 2] > 0.1) & (np.abs(pts_c2[:, 0] / np.maximum(pts_c2[:, 2], 1e-6)) < 0.75) & (
            np.abs(pts_c2[:, 1] / np.maximum(pts_c2[:, 2], 1e-6)) < 0.6
        )
        if not np.any(ok):
            return np.zeros((0, 2), dtype=np.float64)
        return project_points(calib, pts_cam1[ok], 2)

    norm = cv2.undistortPoints(px, calib.K2, calib.D2).reshape(2)
    pts_c2 = np.column_stack([norm[0] * zs, norm[1] * zs, zs])
    pts_cam1 = (pts_c2 - T[None, :]) @ R
    ok = (pts_cam1[:, 2] > 0.1) & (np.abs(pts_cam1[:, 0] / np.maximum(pts_cam1[:, 2], 1e-6)) < 0.75) & (
        np.abs(pts_cam1[:, 1] / np.maximum(pts_cam1[:, 2], 1e-6)) < 0.6
    )
    if not np.any(ok):
        return np.zeros((0, 2), dtype=np.float64)
    return project_points(calib, pts_cam1[ok], 1)
