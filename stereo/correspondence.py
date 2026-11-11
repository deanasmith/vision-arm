"""Choose a stereo correspondence that can belong to a human arm.

Index matching trusts each camera's own left/right labels. A full side swap,
or one mirrored camera, still triangulates with a small reprojection error and
can put a shoulder hundreds of meters out. The fix is to score the identity
assignment against that swap and against an x-mirror of the right camera, then
keep the assignment whose limbs have human lengths.

Depth error along an epipolar line is a second, separate miss: the rays meet,
so reprojection and the Sampson distance are both ~0, while a forearm that is
still inside a loose length window moves the elbow by about 15°. Pose world
landmarks supply the bone length, and the image-relative landmark z picks which
point on the stereo ray agrees with that prior. The published point stays on
the ray through the left pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereo.geometry import (
    StereoCalibration,
    in_front_of_both,
    left_ray_direction,
    triangulate_matched,
)

# MediaPipe pose left/right pairs, plus the two holistic hands stored at
# 100..120 and 200..220.
_POSE_LR_PAIRS = (
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
)
_POSE_SWAP = {left: right for left, right in _POSE_LR_PAIRS}
_POSE_SWAP.update({right: left for left, right in _POSE_LR_PAIRS})

# The symmetric arms-down pose used to check this rig is 16.9 cm / 15.4 cm,
# so the floor stays at 15 cm. Nine centimetres is not an arm. Eighteen is,
# which is why a length prior has to do more than this window.
LIMB_MIN_M = 0.15
LIMB_MAX_M = 0.45
SHOULDER_MIN_M = 0.18
SHOULDER_MAX_M = 0.70
# About the working distance in front of this 0.36 m rig.
DEPTH_MIN_M = 0.40
DEPTH_MAX_M = 2.50

_UPPER = ((11, 13), (12, 14))
_FOREARM = ((13, 15), (14, 16))
_HAND_FOREARM = ((13, 100), (14, 200))
_LIMB_BONES = _UPPER + _FOREARM
_HYPOTHESES = ("identity", "swap", "mirror", "mirror_swap")


@dataclass(frozen=True)
class ResolvedPose:
    """Landmarks in the left camera's index frame, and the right pixels used."""

    points: dict[int, np.ndarray]
    right_xy: dict[int, tuple[float, float]]
    hypothesis: str


def swapped_index(index: int) -> int:
    """Anatomical opposite of a pose or hand landmark index."""
    if 100 <= index < 121:
        return index + 100
    if 200 <= index < 221:
        return index - 100
    return _POSE_SWAP.get(index, index)


def swap_left_right(pixels: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """Exchange left and right pose landmarks and the two hands."""
    return {swapped_index(index): point for index, point in pixels.items()}


def resolve_pose_correspondence(
    calib: StereoCalibration,
    left_xy: dict[int, tuple[float, float]],
    right_xy: dict[int, tuple[float, float]],
    left_world: dict[int, np.ndarray] | None = None,
    right_world: dict[int, np.ndarray] | None = None,
    left_image_z: dict[int, float] | None = None,
    right_image_z: dict[int, float] | None = None,
    max_reproj_px: float = 15.0,
    max_sampson_px: float = 8.0,
) -> ResolvedPose:
    """Triangulate the geometrically consistent left/right assignment.

    ``left_world`` / ``right_world`` are MediaPipe ``pose_world_landmarks`` in
    metres (hip origin). Only their bone lengths are used. ``left_image_z`` /
    ``right_image_z`` are the image-relative landmark z values, where a smaller
    value is closer to the camera. Neither one replaces the stereo ray.
    """
    width = float(calib.image_size2[0])
    best: ResolvedPose | None = None
    best_rank: tuple | None = None
    for name in _HYPOTHESES:
        swapped = name in ("swap", "mirror_swap")
        matched_right = _hypothesis_pixels(right_xy, width, name)
        raw = triangulate_matched(
            calib,
            left_xy,
            matched_right,
            max_reproj_px=max_reproj_px,
            max_sampson_px=max_sampson_px,
        )
        lengths = _merged_lengths(left_world, _relabel(right_world, swapped))
        image_z = _merged_z(left_image_z, _relabel(right_image_z, swapped))
        rank = _rank(raw, lengths, image_z, name)
        if best_rank is None or rank < best_rank:
            points = _apply_body_prior(calib, raw, left_xy, lengths, image_z)
            points = {i: pt for i, pt in points.items() if _in_workspace(calib, pt)}
            best = ResolvedPose(points, matched_right, name)
            best_rank = rank
    assert best is not None
    return best


def _hypothesis_pixels(right_xy, width: float, name: str) -> dict[int, tuple[float, float]]:
    pixels = right_xy
    if name in ("mirror", "mirror_swap"):
        pixels = {i: (width - float(x), float(y)) for i, (x, y) in pixels.items()}
    if name in ("swap", "mirror_swap"):
        pixels = swap_left_right(pixels)
    return pixels


def _relabel(values, swapped: bool):
    if not values:
        return {}
    if not swapped:
        return dict(values)
    return {swapped_index(index): value for index, value in values.items()}


def _merged_lengths(left_world, right_world) -> dict[tuple[int, int], float]:
    left = _bone_lengths(left_world)
    right = _bone_lengths(right_world)
    merged: dict[tuple[int, int], float] = {}
    for key in set(left) | set(right):
        samples = [value for value in (left.get(key), right.get(key)) if value is not None]
        merged[key] = float(np.mean(samples))
    return merged


def _bone_lengths(world) -> dict[tuple[int, int], float]:
    if not world:
        return {}
    lengths = {}
    for proximal, distal in _LIMB_BONES:
        if proximal in world and distal in world:
            lengths[(proximal, distal)] = float(np.linalg.norm(world[distal] - world[proximal]))
    return lengths


def _merged_z(left_z, right_z) -> dict[int, float]:
    merged: dict[int, float] = {}
    for index in set(left_z or {}) | set(right_z or {}):
        samples = [table[index] for table in (left_z, right_z) if table and index in table]
        if samples:
            merged[index] = float(np.mean(samples))
    return merged


def _in_workspace(calib: StereoCalibration, point: np.ndarray) -> bool:
    return in_front_of_both(calib, point) and DEPTH_MIN_M <= float(point[2]) <= DEPTH_MAX_M


def _rank(points, lengths, image_z, name: str) -> tuple:
    plausible = 0
    implausible = 0
    errors: list[float] = []
    for proximal, distal in _LIMB_BONES:
        if proximal not in points or distal not in points:
            continue
        length = float(np.linalg.norm(points[distal] - points[proximal]))
        if LIMB_MIN_M <= length <= LIMB_MAX_M:
            plausible += 1
        else:
            implausible += 1
        if (proximal, distal) in lengths:
            errors.append(abs(length - lengths[(proximal, distal)]))
    if 11 in points and 12 in points:
        span = float(np.linalg.norm(points[11] - points[12]))
        if SHOULDER_MIN_M <= span <= SHOULDER_MAX_M:
            plausible += 1
        else:
            implausible += 1
    in_workspace = sum(1 for point in points.values() if DEPTH_MIN_M <= float(point[2]) <= DEPTH_MAX_M)
    absurd = len(points) - in_workspace
    mean_prior = float(np.mean(errors)) if errors else 0.0
    preference = _HYPOTHESES.index(name)
    return (-plausible, -in_workspace, implausible, absurd, mean_prior, _image_z_cost(points, image_z), preference)


def _image_z_cost(points, image_z) -> float:
    """0 when closer-to-camera image z lines up with a smaller stereo Z."""
    if not image_z:
        return 0.0
    indices = [
        index
        for index in points
        if index in image_z and DEPTH_MIN_M <= float(points[index][2]) <= DEPTH_MAX_M
    ]
    if len(indices) < 2:
        return 0.0
    relative_z = np.array([image_z[index] for index in indices], dtype=np.float64)
    depth = np.array([points[index][2] for index in indices], dtype=np.float64)
    if float(np.std(relative_z)) < 1e-6 or float(np.std(depth)) < 1e-6:
        return 0.0
    correlation = float(np.corrcoef(relative_z, depth)[0, 1])
    if not np.isfinite(correlation):
        return 0.0
    return 1.0 - correlation


def _length_tolerance(prior: float) -> float:
    # Wide enough for a few pixels of landmark noise and for MediaPipe's
    # own scale error, and tight enough that a 12 cm depth miss (forearm
    # 24 cm -> 18 cm) is not treated as agreement.
    return max(0.035, 0.12 * prior)


def _apply_body_prior(calib, points, left_xy, lengths, image_z) -> dict[int, np.ndarray]:
    kept = {index: np.array(point, dtype=np.float64, copy=True) for index, point in points.items()}
    for proximal, distal in _UPPER:
        _repair_or_drop(calib, kept, left_xy, lengths, image_z, proximal, distal)
    for proximal, distal in _FOREARM:
        _repair_or_drop(calib, kept, left_xy, lengths, image_z, proximal, distal)
    for proximal, distal in _HAND_FOREARM:
        _repair_or_drop(calib, kept, left_xy, lengths, image_z, proximal, distal)
    return kept


def _prior_for(lengths, proximal: int, distal: int) -> float | None:
    if (proximal, distal) in lengths:
        return lengths[(proximal, distal)]
    # The hand wrist has no world landmark. Its length should stay near the
    # pose forearm, which is the bone the model did measure.
    if distal == 100:
        return lengths.get((13, 15))
    if distal == 200:
        return lengths.get((14, 16))
    return None


def _repair_or_drop(calib, points, left_xy, lengths, image_z, proximal: int, distal: int) -> None:
    if proximal not in points or distal not in points:
        return
    length = float(np.linalg.norm(points[distal] - points[proximal]))
    prior = _prior_for(lengths, proximal, distal)
    if prior is not None and LIMB_MIN_M <= prior <= LIMB_MAX_M and abs(length - prior) > _length_tolerance(prior):
        repaired = None
        if distal in left_xy:
            repaired = _on_ray_at_length(
                calib,
                points[proximal],
                left_xy[distal],
                prior,
                _relative_z(image_z, proximal, distal),
                points[distal],
            )
        if repaired is not None:
            points[distal] = repaired
            return
        _drop_limb(points, distal)
        return
    if not LIMB_MIN_M <= length <= LIMB_MAX_M:
        _drop_limb(points, distal)


def _relative_z(image_z, proximal: int, distal: int) -> tuple[float | None, float | None]:
    if not image_z:
        return None, None
    distal_z = image_z.get(distal)
    if distal_z is None and distal == 100:
        distal_z = image_z.get(15)
    if distal_z is None and distal == 200:
        distal_z = image_z.get(16)
    return image_z.get(proximal), distal_z


def _on_ray_at_length(calib, proximal, pixel, target: float, relative_z, stereo_point) -> np.ndarray | None:
    """Point on the left ray whose distance to ``proximal`` is ``target``.

    The ray is the stereo measurement. The two intersections are the two
    depths that satisfy the world-landmark length. Image-relative z (smaller
    means closer to the camera) chooses between them.
    """
    direction = left_ray_direction(calib, pixel)
    along = float(np.dot(direction, proximal))
    offset = float(np.dot(proximal, proximal) - target * target)
    discriminant = along * along - offset
    if discriminant < 0.0:
        return None
    root = float(np.sqrt(discriminant))
    candidates = []
    for depth in (along + root, along - root):
        if depth <= 1e-4:
            continue
        point = depth * direction
        if in_front_of_both(calib, point):
            candidates.append(point)
    proximal_z, distal_z = relative_z
    if proximal_z is not None and distal_z is not None and abs(distal_z - proximal_z) > 1e-6:
        # MediaPipe image z: smaller is closer to the camera, same sense as Z.
        want_closer = distal_z < proximal_z
        agreed = [point for point in candidates if (float(point[2]) < float(proximal[2])) == want_closer]
        if not agreed:
            return None
        candidates = agreed
    if len(candidates) == 1:
        return candidates[0]
    if proximal_z is None or distal_z is None:
        return None
    return min(candidates, key=lambda point: float(np.linalg.norm(point - stereo_point)))


def _drop_limb(points, distal: int) -> None:
    points.pop(distal, None)
    if distal == 13:
        points.pop(15, None)
        _drop_hand(points, 100)
    elif distal == 14:
        points.pop(16, None)
        _drop_hand(points, 200)


def _drop_hand(points, base: int) -> None:
    for index in range(base, base + 21):
        points.pop(index, None)
