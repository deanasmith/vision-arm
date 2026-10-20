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
