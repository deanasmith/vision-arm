"""Draw the original kinematic overlay, using triangulated 3D when available."""

from __future__ import annotations

import cv2
import numpy as np

from stereo.kinematics import calculate_3d_angle, calculate_shoulder_flexion

UPPER_ARM_CONNECTIONS = [(11, 12), (11, 13), (12, 14)]
SKELETON_CONNECTIONS = UPPER_ARM_CONNECTIONS + [(13, 15), (14, 16)]

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16


def landmark_pixels(landmarks, width: int, height: int, visibility: float) -> dict[int, tuple[float, float]]:
    pixels: dict[int, tuple[float, float]] = {}
    if landmarks is None:
        return pixels
    for i, lm in enumerate(landmarks.landmark):
        vis = getattr(lm, "visibility", 1.0)
        if vis < visibility:
            continue
        pixels[i] = (lm.x * width, lm.y * height)
    return pixels


def _pt(pixels: dict[int, tuple[float, float]], idx: int) -> tuple[int, int] | None:
    if idx not in pixels:
        return None
    x, y = pixels[idx]
    return int(x), int(y)


def draw_pose_overlay(
    frame: np.ndarray,
    pixels: dict[int, tuple[float, float]],
    points_3d: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    vis = frame.copy()
    for a, b in SKELETON_CONNECTIONS:
        pa, pb = _pt(pixels, a), _pt(pixels, b)
        if pa is None or pb is None:
            continue
        cv2.line(vis, pa, pb, (0, 255, 0), 3)
        cv2.circle(vis, pa, 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(vis, pb, 5, (0, 0, 255), cv2.FILLED)

    pts = points_3d or {}
    if LEFT_SHOULDER in pts and LEFT_ELBOW in pts:
        shoulder_angle = calculate_shoulder_flexion(pts[LEFT_SHOULDER], pts[LEFT_ELBOW])
        shoulder_px = _pt(pixels, LEFT_SHOULDER)
        if shoulder_px is not None:
            cv2.putText(
                vis,
                f"Shoulder: {int(shoulder_angle)}",
                (shoulder_px[0] + 15, shoulder_px[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        if LEFT_WRIST in pts:
            elbow_angle = calculate_3d_angle(pts[LEFT_SHOULDER], pts[LEFT_ELBOW], pts[LEFT_WRIST])
            elbow_px = _pt(pixels, LEFT_ELBOW)
            if elbow_px is not None:
                cv2.putText(
                    vis,
                    f"Elbow: {int(elbow_angle)}",
                    (elbow_px[0] + 15, elbow_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
    return vis


def draw_world_view(points_3d: dict[int, np.ndarray], width: int = 480, height: int = 360) -> np.ndarray:
    """Orthographic X-Z (top) view of triangulated skeleton, in meters."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(canvas, "World top view (X right, Z forward)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if len(points_3d) < 2:
        cv2.putText(canvas, "Need both cameras on the same joints", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        return canvas

    coords = np.stack(list(points_3d.values()))
    span = max(float(np.max(np.abs(coords[:, [0, 2]]))), 0.4)
    margin = 40

    def to_px(pt: np.ndarray) -> tuple[int, int]:
        x = int((pt[0] / span) * 0.45 * width + width / 2)
        z = int(height - margin - (pt[2] / (span * 2.2)) * (height - 2 * margin))
        return x, z

    origin = to_px(np.array([0.0, 0.0, 0.0]))
    cv2.circle(canvas, origin, 4, (255, 255, 0), cv2.FILLED)
    cv2.putText(canvas, "cam1", (origin[0] + 6, origin[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    for a, b in SKELETON_CONNECTIONS:
        if a not in points_3d or b not in points_3d:
            continue
        pa, pb = to_px(points_3d[a]), to_px(points_3d[b])
        cv2.line(canvas, pa, pb, (0, 255, 0), 2)
        cv2.circle(canvas, pa, 4, (0, 0, 255), cv2.FILLED)
        cv2.circle(canvas, pb, 4, (0, 0, 255), cv2.FILLED)

    labels = {11: "LS", 12: "RS", 13: "LE", 14: "RE", 15: "LW", 16: "RW"}
    for idx, name in labels.items():
        if idx not in points_3d:
            continue
        x, y = to_px(points_3d[idx])
        cv2.putText(canvas, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas


def format_xyz(pt: np.ndarray | None) -> str:
    if pt is None:
        return "---"
    return f"{pt[0]:+.3f}, {pt[1]:+.3f}, {pt[2]:+.3f}"
