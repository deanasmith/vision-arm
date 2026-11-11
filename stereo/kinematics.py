"""Joint-angle helpers. Same math as the original single-camera script."""

import numpy as np


def calculate_3d_angle(a, b, c) -> float:
    """Angle at vertex b formed by points a-b-c, in degrees."""
    a, b, c = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def elevation_from_down(proximal, distal) -> float:
    """Angle of a limb from straight down, in the camera frame. 0 hangs down, 90 is horizontal."""
    limb = np.asarray(distal, dtype=np.float64) - np.asarray(proximal, dtype=np.float64)
    down = np.array([0.0, 1.0, 0.0])
    if np.linalg.norm(limb) == 0:
        return 0.0
    cosine = np.dot(limb, down) / np.linalg.norm(limb)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def wrist_roll(elbow, wrist, index_mcp, pinky_mcp) -> float:
    """Signed roll of the palm around the forearm, in degrees.

    0 means the knuckle line (index to pinky) is perpendicular to the forearm
    and aligned with camera-down. Positive rotation follows the right-hand rule
    along elbow to wrist.
    """
    forearm = np.asarray(wrist, dtype=np.float64) - np.asarray(elbow, dtype=np.float64)
    axis_len = np.linalg.norm(forearm)
    if axis_len < 1e-6:
        return float("nan")
    axis = forearm / axis_len
    palm = np.asarray(index_mcp, dtype=np.float64) - np.asarray(pinky_mcp, dtype=np.float64)
    palm = palm - axis * np.dot(palm, axis)
    palm_len = np.linalg.norm(palm)
    if palm_len < 1e-6:
        return float("nan")
    palm = palm / palm_len
    down = np.array([0.0, 1.0, 0.0])
    ref = down - axis * np.dot(down, axis)
    if np.linalg.norm(ref) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0])
        ref = ref - axis * np.dot(ref, axis)
    ref = ref / np.linalg.norm(ref)
    sine = float(np.dot(axis, np.cross(ref, palm)))
    cosine = float(np.dot(ref, palm))
    return float(np.degrees(np.arctan2(sine, cosine)))


def calculate_shoulder_flexion(shoulder, elbow) -> float:
    """Front-raise angle ignoring X, using a static down vector in Y-Z."""
    sh = np.array([shoulder[1], shoulder[2]], dtype=np.float64)
    el = np.array([elbow[1], elbow[2]], dtype=np.float64)
    v_arm = el - sh
    v_down = np.array([1.0, 0.0])
    if np.linalg.norm(v_arm) == 0:
        return 0.0
    cosine_angle = np.dot(v_arm, v_down) / (np.linalg.norm(v_arm) * np.linalg.norm(v_down))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))
