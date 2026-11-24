"""Draw the original kinematic overlay, using triangulated 3D when available."""

from __future__ import annotations

import cv2
import numpy as np

from stereo.kinematics import calculate_3d_angle, elevation_from_down

UPPER_ARM_CONNECTIONS = [(11, 12), (11, 13), (12, 14)]
SKELETON_CONNECTIONS = UPPER_ARM_CONNECTIONS + [(13, 15), (14, 16)]

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

# Hand landmarks are stored past the pose indices so both can be triangulated together.
LEFT_HAND, RIGHT_HAND = 100, 200
_FINGER_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def hand_connections(base: int) -> list[tuple[int, int]]:
    return [(base + a, base + b) for a, b in _FINGER_CONNECTIONS]


LEFT_HAND_CONNECTIONS = hand_connections(LEFT_HAND) + [(LEFT_ELBOW, LEFT_HAND)]
RIGHT_HAND_CONNECTIONS = hand_connections(RIGHT_HAND) + [(RIGHT_ELBOW, RIGHT_HAND)]


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


def _chain_wrist(points_3d: dict[int, np.ndarray], wrist_idx: int, hand_base: int) -> np.ndarray | None:
    """Hand-model wrist when both cameras saw the hand. Pose wrist is the fallback."""
    if hand_base in points_3d:
        return points_3d[hand_base]
    return points_3d.get(wrist_idx)


def _label_arm(
    vis: np.ndarray,
    pixels: dict[int, tuple[float, float]],
    points_3d: dict[int, np.ndarray],
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    hand_base: int,
) -> None:
    if shoulder_idx not in points_3d or elbow_idx not in points_3d:
        return
    shoulder_px = _pt(pixels, shoulder_idx)
    if shoulder_px is not None:
        elev = elevation_from_down(points_3d[shoulder_idx], points_3d[elbow_idx])
        cv2.putText(
            vis,
            f"Shoulder: {int(round(elev))}",
            (shoulder_px[0] + 15, shoulder_px[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    wrist = _chain_wrist(points_3d, wrist_idx, hand_base)
    elbow_px = _pt(pixels, elbow_idx)
    if wrist is None or elbow_px is None:
        return
    forearm = float(np.linalg.norm(wrist - points_3d[elbow_idx]))
    if not 0.15 <= forearm <= 0.45:
        return
    flexion = 180.0 - calculate_3d_angle(points_3d[shoulder_idx], points_3d[elbow_idx], wrist)
    cv2.putText(
        vis,
        f"Elbow: {int(round(flexion))}",
        (elbow_px[0] + 15, elbow_px[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


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
    _label_arm(vis, pixels, pts, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HAND)
    _label_arm(vis, pixels, pts, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HAND)
    return vis


def _body_origin(points_3d: dict[int, np.ndarray]) -> np.ndarray:
    if LEFT_SHOULDER in points_3d and RIGHT_SHOULDER in points_3d:
        return 0.5 * (points_3d[LEFT_SHOULDER] + points_3d[RIGHT_SHOULDER])
    coords = np.stack(list(points_3d.values()))
    return np.mean(coords, axis=0)


_VIEW_INDICES = {
    idx
    for connections in (SKELETON_CONNECTIONS, LEFT_HAND_CONNECTIONS, RIGHT_HAND_CONNECTIONS)
    for pair in connections
    for idx in pair
}


def _view_points(points_3d: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Arms and hands only. Other pose points are not part of this drawing."""
    return {idx: pt for idx, pt in points_3d.items() if idx in _VIEW_INDICES}


def draw_world_view(points_3d: dict[int, np.ndarray], width: int = 480, height: int = 360) -> np.ndarray:
    """Top and side views of the triangulated arms and hands, in meters.

    Points are centered on the shoulders. The top view looks down, with the
    stereo cameras at the bottom of that panel. The side view uses the same
    depth axis: farther from the cameras is to the right, and up is up.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    half = width // 2
    cv2.line(canvas, (half, 0), (half, height), (45, 45, 45), 1)
    cv2.putText(canvas, "Top  (right / away from cameras)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "Side  (up / away from cameras)", (half + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    shown = _view_points(points_3d)
    if len(shown) < 2:
        cv2.putText(canvas, "Need both cameras on the same joints", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        return canvas

    origin = _body_origin(shown)
    # A real arm stays within about a meter of the shoulders. A mismatched
    # landmark can triangulate much farther away and shrink everything else.
    shown = {idx: pt for idx, pt in shown.items() if float(np.linalg.norm(pt - origin)) <= 1.0}
    if len(shown) < 2:
        cv2.putText(canvas, "Need both cameras on the same joints", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        return canvas
    centered = {idx: pt - origin for idx, pt in shown.items()}
    stack = np.stack(list(centered.values()))
    span_x = max(float(np.max(np.abs(stack[:, 0]))), 0.05)
    span_up = max(float(np.max(np.abs(stack[:, 1]))), 0.05)
    span_z = max(float(np.max(np.abs(stack[:, 2]))), 0.05)
    # One scale for both panels, large enough that the longer arm axis fills its panel.
    usable_w = half * 0.86
    usable_h = (height - 56) * 0.86
    scale = min(
        usable_w / (2 * span_x),
        usable_h / (2 * span_z),
        usable_w / (2 * span_z),
        usable_h / (2 * span_up),
    )

    def project(pt: np.ndarray, panel: int) -> tuple[int, int]:
        # Top: X right, +Z up the panel (away from the cameras).
        # Side: +Z to the right, -Y up.
        horizontal = pt[0] if panel == 0 else pt[2]
        vertical = pt[2] if panel == 0 else -pt[1]
        x = int(panel * half + half / 2 + horizontal * scale)
        y = int(height / 2 - vertical * scale)
        return x, y

    groups = (
        (SKELETON_CONNECTIONS, (0, 220, 0)),
        (LEFT_HAND_CONNECTIONS, (255, 200, 0)),
        (RIGHT_HAND_CONNECTIONS, (0, 180, 255)),
    )
    for connections, color in groups:
        for a, b in connections:
            if a not in centered or b not in centered:
                continue
            for panel in (0, 1):
                pa, pb = project(centered[a], panel), project(centered[b], panel)
                cv2.line(canvas, pa, pb, color, 2)
                cv2.circle(canvas, pa, 3, color, cv2.FILLED)
                cv2.circle(canvas, pb, 3, color, cv2.FILLED)

    labels = {11: "LS", 12: "RS", 13: "LE", 14: "RE", 15: "LW", 16: "RW", LEFT_HAND: "LH", RIGHT_HAND: "RH"}
    for idx, name in labels.items():
        if idx not in centered:
            continue
        x, y = project(centered[idx], 0)
        cv2.putText(canvas, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    tick = max(int(0.25 * scale), 1)
    for panel in (0, 1):
        x0 = panel * half + 16
        y0 = height - 28
        cv2.line(canvas, (x0, y0), (x0 + tick, y0), (180, 180, 180), 2)
        cv2.putText(canvas, "25 cm", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    return canvas


def format_xyz(pt: np.ndarray | None) -> str:
    if pt is None:
        return "---"
    return f"{pt[0]:+.3f}, {pt[1]:+.3f}, {pt[2]:+.3f}"
