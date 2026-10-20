from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stereo.config import StereoConfig

ARUCO_DICT = cv2.aruco.DICT_5X5_50


def make_board(config: StereoConfig) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        config.square_length_m,
        config.marker_length_m,
        dictionary,
    )


class CharucoHelper:
    def __init__(self, config: StereoConfig):
        self.config = config
        self.board = make_board(config)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _marker_corners, _marker_ids = self.detector.detectBoard(gray)
        if ids is None or len(ids) == 0:
            return None, None
        return corners, ids

    def generate_image(self, out_size: tuple[int, int] | None = None) -> np.ndarray:
        if out_size is None:
            # ~150 px per square is sharp enough to print; keep it modest for previews.
            out_size = (self.config.squares_x * 150, self.config.squares_y * 150)
        margin = max(20, min(out_size) // 20)
        return self.board.generateImage(out_size, marginSize=margin)

    def save_printable_board(self, path: str | Path | None = None) -> Path:
        dest = Path(path or self.config.board_image_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        image = self.generate_image()
        if not cv2.imwrite(str(dest), image):
            raise RuntimeError(f"Failed to write Charuco board image to {dest}")
        return dest


def paired_charuco_points(
    board: cv2.aruco.CharucoBoard,
    corners_l: np.ndarray | None,
    ids_l: np.ndarray | None,
    corners_r: np.ndarray | None,
    ids_r: np.ndarray | None,
    min_corners: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Object points and corresponding left/right image points for shared Charuco IDs."""
    if corners_l is None or ids_l is None or corners_r is None or ids_r is None:
        return None
    ids_l = ids_l.flatten()
    ids_r = ids_r.flatten()
    common = np.intersect1d(ids_l, ids_r)
    if len(common) < min_corners:
        return None

    chess = board.getChessboardCorners()
    map_l = {int(i): corners_l[k].reshape(2) for k, i in enumerate(ids_l)}
    map_r = {int(i): corners_r[k].reshape(2) for k, i in enumerate(ids_r)}
    obj, img_l, img_r = [], [], []
    for cid in common:
        cid = int(cid)
        obj.append(chess[cid])
        img_l.append(map_l[cid])
        img_r.append(map_r[cid])
    return (
        np.asarray(obj, dtype=np.float32),
        np.asarray(img_l, dtype=np.float32),
        np.asarray(img_r, dtype=np.float32),
    )


def draw_detection(frame: np.ndarray, corners: np.ndarray | None, ids: np.ndarray | None) -> np.ndarray:
    vis = frame.copy()
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
    return vis
