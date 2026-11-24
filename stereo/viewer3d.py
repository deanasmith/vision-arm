"""Clean, minimal dual-camera + interactive 3D orbit viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from stereo.geometry import RigFrame, StereoCalibration, epipolar_curve_in_other, project_points
from stereo.tracker import (
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_INDEX,
    LEFT_PINKY,
    LEFT_SHOULDER,
    LEFT_THUMB,
    LEFT_WRIST,
    NOSE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_INDEX,
    RIGHT_PINKY,
    RIGHT_SHOULDER,
    RIGHT_THUMB,
    RIGHT_WRIST,
    StereoFrame,
    StereoJoint,
)

DASH_W = 1280
CAM_W = 640
CAM_H = 360
VIEW3D_W = 1280
VIEW3D_H = 480
DASH_H = CAM_H + VIEW3D_H  # 840

_COLOR_STEREO = (90, 235, 130)    # green: locked by both cameras
_COLOR_HANDOVER = (245, 215, 70)  # cyan: single-camera ray + bone handover
_COLOR_OTHER = (180, 190, 200)

_UPPER_BODY_EDGES = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (RIGHT_WRIST, RIGHT_INDEX),
    (RIGHT_WRIST, RIGHT_PINKY),
    (RIGHT_WRIST, RIGHT_THUMB),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (LEFT_WRIST, LEFT_INDEX),
    (LEFT_WRIST, LEFT_PINKY),
    (LEFT_WRIST, LEFT_THUMB),
    (LEFT_EAR, NOSE),
    (RIGHT_EAR, NOSE),
)

def _arm_indices(arm: str) -> tuple[int, int, int]:
    if arm == "left":
        return (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    return (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)


def _epi_joints(arm: str) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    sh, el, wr = _arm_indices(arm)
    return (
        (sh, (255, 210, 60)),
        (el, (40, 175, 255)),
        (wr, (220, 90, 255)),
    )


@dataclass
class OrbitCamera:
    """Interactive 3D orbit camera in rig coordinates (+X right, +Y down, +Z forward)."""

    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.08, 0.85], dtype=np.float64))
    yaw_deg: float = -52.0
    pitch_deg: float = 22.0
    distance: float = 2.05
    fov_deg: float = 50.0
    preset_name: str = "3/4"

    def set_preset(self, key: int) -> None:
        if key == 1:
            self.target = np.array([0.0, 0.08, 0.85], dtype=np.float64)
            self.yaw_deg = -52.0
            self.pitch_deg = 22.0
            self.distance = 2.05
            self.preset_name = "3/4"
        elif key == 2:
            self.target = np.array([0.0, 0.0, 0.90], dtype=np.float64)
            self.yaw_deg = 0.0
            self.pitch_deg = 88.0
            self.distance = 2.15
            self.preset_name = "Top"
        elif key == 3:
            self.target = np.array([0.0, 0.08, 0.85], dtype=np.float64)
            self.yaw_deg = -90.0
            self.pitch_deg = 6.0
            self.distance = 2.10
            self.preset_name = "Side"
        elif key == 4:
            self.target = np.array([0.0, 0.05, 0.95], dtype=np.float64)
            self.yaw_deg = 0.0
            self.pitch_deg = 6.0
            self.distance = 2.05
            self.preset_name = "Front"

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        cp, sp = math.cos(pitch), math.sin(pitch)
        sy, cy = math.sin(yaw), math.cos(yaw)
        offset = np.array([-cp * sy, -sp, -cp * cy], dtype=np.float64) * self.distance
        pos = self.target + offset
        forward = self.target - pos
        forward /= max(float(np.linalg.norm(forward)), 1e-9)

        world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(forward, world_up))) > 0.995:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        down = np.cross(forward, right)
        down /= max(float(np.linalg.norm(down)), 1e-9)
        return pos, right, down, forward

    def orbit(self, dx_px: float, dy_px: float) -> None:
        self.yaw_deg = (self.yaw_deg - dx_px * 0.35 + 180.0) % 360.0 - 180.0
        self.pitch_deg = float(np.clip(self.pitch_deg + dy_px * 0.30, -85.0, 88.5))
        self.preset_name = "Orbit"

    def pan(self, dx_px: float, dy_px: float) -> None:
        _pos, right, down, _fwd = self.basis()
        scale = self.distance * 0.0018
        self.target = self.target - right * (dx_px * scale) - down * (dy_px * scale)

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 0.45, 7.0))


@dataclass
class DashboardState:
    orbit: OrbitCamera = field(default_factory=OrbitCamera)
    arm: str = "right"
    debug_overlay: bool = False
    use_filter: bool = True
    uart_enabled: bool = False
    uart_connect_requested: bool = False
    _drag_mode: str | None = None
    _last_x: int = 0
    _last_y: int = 0

    def handle_mouse(self, event: int, x: int, y: int, flags: int, _param=None) -> None:
        in_view3d = (0 <= x < VIEW3D_W) and (CAM_H <= y < DASH_H)
        if event == cv2.EVENT_LBUTTONDOWN and in_view3d:
            self._drag_mode = "pan" if (flags & cv2.EVENT_FLAG_SHIFTKEY) else "orbit"
            self._last_x, self._last_y = x, y
        elif event == cv2.EVENT_RBUTTONDOWN and in_view3d:
            self._drag_mode = "pan"
            self._last_x, self._last_y = x, y
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drag_mode = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_mode is not None:
            dx = x - self._last_x
            dy = y - self._last_y
            self._last_x, self._last_y = x, y
            if self._drag_mode == "orbit":
                self.orbit.orbit(dx, dy)
            else:
                self.orbit.pan(dx, dy)
        elif event == cv2.EVENT_LBUTTONDBLCLK and in_view3d:
            self.orbit.set_preset(1)
        elif event == cv2.EVENT_MOUSEWHEEL and in_view3d:
            delta = flags >> 16
            if delta == 0:
                delta = flags
            if delta > 0:
                self.orbit.zoom(0.90)
            elif delta < 0:
                self.orbit.zoom(1.11)

    def handle_key(self, key: int) -> bool:
        """Process keycode; return True to quit."""
        if key == 27 or key in (ord("q"), ord("Q")):
            return True
        if ord("1") <= key <= ord("4"):
            self.orbit.set_preset(key - ord("0"))
        elif key in (ord("r"), ord("R")):
            self.orbit.set_preset(1)
        elif key in (ord("d"), ord("D")):
            self.debug_overlay = not self.debug_overlay
        elif key in (ord("f"), ord("F")):
            self.use_filter = not self.use_filter
        elif key in (ord("i"), ord("I")):
            self.uart_connect_requested = True
        elif key in (ord("u"), ord("U")):
            self.uart_enabled = not self.uart_enabled
        elif key in (ord("+"), ord("=")):
            self.orbit.zoom(0.88)
        elif key in (ord("-"), ord("_")):
            self.orbit.zoom(1.14)
        return False


class _Projector3D:
    def __init__(self, orbit: OrbitCamera, width: int, height: int):
        self.width = width
        self.height = height
        self.pos, self.right, self.down, self.forward = orbit.basis()
        self.f = (height * 0.5) / math.tan(math.radians(orbit.fov_deg) * 0.5)
        self.cx = width * 0.5
        self.cy = height * 0.5

    def to_cam(self, pt: np.ndarray) -> np.ndarray:
        rel = np.asarray(pt, dtype=np.float64) - self.pos
        return np.array(
            [float(np.dot(rel, self.right)), float(np.dot(rel, self.down)), float(np.dot(rel, self.forward))],
            dtype=np.float64,
        )

    def project(self, pt: np.ndarray) -> tuple[int, int, float] | None:
        c = self.to_cam(pt)
        if c[2] < 0.05:
            return None
        u = self.cx + self.f * (c[0] / c[2])
        v = self.cy + self.f * (c[1] / c[2])
        if abs(u) > 8000 or abs(v) > 8000:
            return None
        return int(round(u)), int(round(v)), float(c[2])

    def line(self, canvas: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
        ca = self.to_cam(a)
        cb = self.to_cam(b)
        near = 0.06
        if ca[2] < near and cb[2] < near:
            return
        if ca[2] < near:
            t = (near - ca[2]) / max(cb[2] - ca[2], 1e-9)
            ca = ca + t * (cb - ca)
        elif cb[2] < near:
            t = (near - cb[2]) / max(ca[2] - cb[2], 1e-9)
            cb = cb + t * (ca - cb)
        ua = int(round(self.cx + self.f * (ca[0] / ca[2])))
        va = int(round(self.cy + self.f * (ca[1] / ca[2])))
        ub = int(round(self.cx + self.f * (cb[0] / cb[2])))
        vb = int(round(self.cy + self.f * (cb[1] / cb[2])))
        if max(abs(ua), abs(va), abs(ub), abs(vb)) > 10000:
            return
        cv2.line(canvas, (ua, va), (ub, vb), color, thickness, cv2.LINE_AA)


def _joint_color(joint: StereoJoint | None) -> tuple[int, int, int]:
    if joint is None:
        return (120, 125, 130)
    return _COLOR_HANDOVER if joint.source in ("ray+bone", "hold") else _COLOR_STEREO


def draw_camera_panel(
    frame: np.ndarray,
    camera_num: int,
    title: str,
    own_xy: dict[int, tuple[float, float]],
    other_xy: dict[int, tuple[float, float]],
    stereo: StereoFrame,
    calib: StereoCalibration,
    *,
    arm: str = "right",
    debug_overlay: bool = False,
) -> np.ndarray:
    """Clean camera view (CAM_W x CAM_H) with minimal 2D skeleton + optional debug overlays."""
    h, w = frame.shape[:2]
    vis = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    sx = CAM_W / float(w)
    sy = CAM_H / float(h)
    arm_sh, arm_el, arm_wr = _arm_indices(arm)
    arm_edges = ((arm_sh, arm_el), (arm_el, arm_wr))
    arm_joint_set = (arm_sh, arm_el, arm_wr)

    def sc(xy: tuple[float, float]) -> tuple[int, int]:
        return int(round(xy[0] * sx)), int(round(xy[1] * sy))

    if debug_overlay:
        other_cam = 2 if camera_num == 1 else 1
        for idx, color in _epi_joints(arm):
            j = stereo.joints.get(idx)
            if j is not None:
                src_pt = j.pt2 if camera_num == 1 else j.pt1
            elif idx in other_xy:
                src_pt = other_xy[idx]
            else:
                continue
            curve = epipolar_curve_in_other(calib, src_pt, source_camera=other_cam, n_samples=20)
            if len(curve) >= 2:
                pts = np.round(np.column_stack([curve[:, 0] * sx, curve[:, 1] * sy])).astype(np.int32)
                overlay = vis.copy()
                cv2.polylines(overlay, [pts], False, color, 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.5, vis, 0.5, 0.0, vis)

    for a, b in _UPPER_BODY_EDGES:
        if a not in own_xy or b not in own_xy:
            continue
        is_active_arm = (a, b) in arm_edges
        color = _COLOR_STEREO if is_active_arm else (160, 175, 165)
        thick = 2 if is_active_arm else 1
        cv2.line(vis, sc(own_xy[a]), sc(own_xy[b]), color, thick, cv2.LINE_AA)

    for idx, xy in own_xy.items():
        j = stereo.joints.get(idx)
        r = 4 if idx in arm_joint_set else 3
        cv2.circle(vis, sc(xy), r, _joint_color(j), cv2.FILLED, cv2.LINE_AA)

    if debug_overlay:
        for idx, j in stereo.joints.items():
            proj = project_points(calib, j.xyz_cam1, camera_num)[0]
            px = sc((float(proj[0]), float(proj[1])))
            if 0 <= px[0] < CAM_W and 0 <= px[1] < CAM_H:
                cv2.drawMarker(vis, px, (0, 255, 255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)

    # Subtle corner pill label
    cv2.rectangle(vis, (8, 8), (138, 30), (16, 18, 22), cv2.FILLED)
    cv2.putText(vis, title, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 228, 235), 1, cv2.LINE_AA)
    return vis


def _draw_camera_frustum(
    proj: _Projector3D,
    canvas: np.ndarray,
    pos: np.ndarray,
    R_rig: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    corners_local = np.array(
        [[-0.055, -0.032, 0.11], [0.055, -0.032, 0.11], [0.055, 0.032, 0.11], [-0.055, 0.032, 0.11]],
        dtype=np.float64,
    )
    corners = pos[None, :] + corners_local @ R_rig.T
    for c in corners:
        proj.line(canvas, pos, c, color, 1)
    for i in range(4):
        proj.line(canvas, corners[i], corners[(i + 1) % 4], color, 1)


def render_3d_viewport(
    rig: RigFrame,
    stereo: StereoFrame,
    state: DashboardState,
    *,
    fps: float,
    log_label: str = "",
    uart_label: str = "",
) -> np.ndarray:
    """Clean, full-width interactive 3D orbit viewport (VIEW3D_W x VIEW3D_H)."""
    canvas = np.full((VIEW3D_H, VIEW3D_W, 3), (18, 20, 24), dtype=np.uint8)
    proj = _Projector3D(state.orbit, VIEW3D_W, VIEW3D_H)

    floor_y = 0.65
    if stereo.joints:
        max_y = max(float(j.xyz_rig[1]) for j in stereo.joints.values())
        floor_y = max(0.55, min(1.30, max_y + 0.25))

    # Subtle metric floor grid (0.5 m spacing)
    for z in np.arange(0.0, 2.51, 0.5):
        proj.line(canvas, np.array([-1.25, floor_y, z]), np.array([1.25, floor_y, z]), (38, 43, 50), 1)
    for x in np.arange(-1.25, 1.26, 0.5):
        proj.line(canvas, np.array([x, floor_y, 0.0]), np.array([x, floor_y, 2.5]), (38, 43, 50), 1)

    # Calibrated 3D cameras + baseline bar
    c1_col = (195, 165, 80) if rig.cam0_is_physical_left else (180, 115, 195)
    c2_col = (180, 115, 195) if rig.cam0_is_physical_left else (195, 165, 80)
    proj.line(canvas, rig.cam1_pos_rig, rig.cam2_pos_rig, (120, 130, 145), 2)
    _draw_camera_frustum(proj, canvas, rig.cam1_pos_rig, rig.cam1_R_rig, c1_col)
    _draw_camera_frustum(proj, canvas, rig.cam2_pos_rig, rig.cam2_R_rig, c2_col)

    pts_rig = {idx: j.xyz_rig for idx, j in stereo.joints.items()}
    arm_sh, arm_el, arm_wr = _arm_indices(state.arm)
    arm_edges = ((arm_sh, arm_el), (arm_el, arm_wr))
    arm_joint_set = (arm_sh, arm_el, arm_wr)

    # Optional debug 3D rays from both cameras to the active arm
    if state.debug_overlay:
        for idx in arm_joint_set:
            if idx in pts_rig:
                pt = pts_rig[idx]
                proj.line(canvas, rig.cam1_pos_rig, pt, (75, 65, 45), 1)
                proj.line(canvas, rig.cam2_pos_rig, pt, (70, 50, 75), 1)

    # Floor shadow for immediate depth perception
    for a, b in _UPPER_BODY_EDGES:
        if a in pts_rig and b in pts_rig:
            sa = np.array([pts_rig[a][0], floor_y, pts_rig[a][2]])
            sb = np.array([pts_rig[b][0], floor_y, pts_rig[b][2]])
            proj.line(canvas, sa, sb, (40, 45, 52), 2)

    # Subtle shaded torso quad
    if all(k in pts_rig for k in (LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP)):
        quad = [proj.project(pts_rig[k]) for k in (LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP)]
        if all(p is not None for p in quad):
            poly = np.array([[p[0], p[1]] for p in quad], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillConvexPoly(overlay, poly, (48, 62, 54), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0, canvas)

    # Skeleton bones (depth-sorted)
    drawable: list[tuple[float, int, int]] = []
    for a, b in _UPPER_BODY_EDGES:
        if a in pts_rig and b in pts_rig:
            pm = proj.project(0.5 * (pts_rig[a] + pts_rig[b]))
            if pm is not None:
                drawable.append((pm[2], a, b))
    drawable.sort(key=lambda item: -item[0])

    for _d, a, b in drawable:
        is_active_arm = (a, b) in arm_edges
        if is_active_arm:
            handover = stereo.joints[a].source in ("ray+bone", "hold") or stereo.joints[b].source in ("ray+bone", "hold")
            color = _COLOR_HANDOVER if handover else _COLOR_STEREO
            thick = 4
        else:
            color, thick = _COLOR_OTHER, 2
        proj.line(canvas, pts_rig[a], pts_rig[b], color, thick)

    # Joint spheres (depth-sorted)
    proj_joints: list[tuple[float, StereoJoint, int, int]] = []
    for j in stereo.joints.values():
        pj = proj.project(j.xyz_rig)
        if pj is not None:
            proj_joints.append((pj[2], j, pj[0], pj[1]))
    proj_joints.sort(key=lambda item: -item[0])

    for depth_v, j, ux, uy in proj_joints:
        if not (0 <= ux < VIEW3D_W and 0 <= uy < VIEW3D_H):
            continue
        base_r = 6 if j.index in arm_joint_set else 4
        r = max(3, min(10, int(round(base_r * 2.0 / max(depth_v, 0.5)))))
        cv2.circle(canvas, (ux, uy), r + 1, (14, 16, 20), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(canvas, (ux, uy), r, _joint_color(j), cv2.FILLED, cv2.LINE_AA)

    # Minimal 1-line readout at top-left
    active_arm = stereo.left_arm if state.arm == "left" else stereo.right_arm
    arm_tag = "L" if state.arm == "left" else "R"
    parts = [f"[{state.orbit.preset_name} {arm_tag}]"]
    if active_arm.upper_arm_cm is not None:
        parts.append(f"Up {active_arm.upper_arm_cm:.1f}cm")
    if active_arm.forearm_cm is not None:
        parts.append(f"Fa {active_arm.forearm_cm:.1f}cm")
    if active_arm.hip_yaw_deg is not None:
        parts.append(f"Hip {active_arm.hip_yaw_deg:.0f}")
    if active_arm.elevation_deg is not None:
        parts.append(f"Sh {active_arm.elevation_deg:.0f}")
    if active_arm.elbow_interior_deg is not None:
        parts.append(f"El {active_arm.elbow_interior_deg:.0f}")
    if active_arm.wrist_angle_deg is not None:
        parts.append(f"Wr {active_arm.wrist_angle_deg:.0f}")
    if active_arm.hand_open_deg is not None:
        parts.append(f"Hnd {active_arm.hand_open_deg:.0f}")
    if math.isfinite(stereo.median_ray_gap_mm):
        parts.append(f"gap {stereo.median_ray_gap_mm:.1f}mm")
    parts.append(f"{fps:.0f} fps")
    if not state.use_filter:
        parts.append("RAW")
    if any(j.source == "ray+bone" for j in stereo.joints.values()):
        parts.append("1-CAM")

    hud_line = "  |  ".join(parts)
    cv2.putText(canvas, hud_line, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 228, 235), 1, cv2.LINE_AA)

    # Subtle 1-line bottom footer
    cv2.rectangle(canvas, (0, VIEW3D_H - 26), (VIEW3D_W, VIEW3D_H), (12, 14, 18), cv2.FILLED)
    footer = "Drag: Orbit  Shift-Drag: Pan  Scroll: Zoom  1-4: Views  R: Reset  D: Debug  F: Filter  I: Connect UART  U: Send UART  ESC: Quit"
    cv2.putText(canvas, footer, (14, VIEW3D_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (145, 158, 172), 1, cv2.LINE_AA)
    right_tags = [t for t in (uart_label, log_label) if t]
    if right_tags:
        right_text = "  |  ".join(right_tags)
        (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        if uart_label.startswith("UART TX"):
            color = (110, 235, 140)
        elif uart_label.startswith("UART READY"):
            color = (110, 210, 245)
        else:
            color = (145, 158, 172)
        cv2.putText(canvas, right_text, (VIEW3D_W - tw - 14, VIEW3D_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    return canvas


def compose_dashboard(
    frame_cam0: np.ndarray,
    frame_cam1: np.ndarray,
    xy_cam0: dict[int, tuple[float, float]],
    xy_cam1: dict[int, tuple[float, float]],
    stereo: StereoFrame,
    calib: StereoCalibration,
    rig: RigFrame,
    state: DashboardState,
    *,
    fps: float,
    log_label: str = "",
    uart_label: str = "",
) -> np.ndarray:
    """Assemble the clean 1280x840 dual-camera + 3D orbit window."""
    p0 = draw_camera_panel(
        frame_cam0,
        1,
        "CAM 0 (Left)" if rig.cam0_is_physical_left else "CAM 0 (Right)",
        xy_cam0,
        xy_cam1,
        stereo,
        calib,
        arm=state.arm,
        debug_overlay=state.debug_overlay,
    )
    p1 = draw_camera_panel(
        frame_cam1,
        2,
        "CAM 1 (Right)" if rig.cam0_is_physical_left else "CAM 1 (Left)",
        xy_cam1,
        xy_cam0,
        stereo,
        calib,
        arm=state.arm,
        debug_overlay=state.debug_overlay,
    )
    left_tile, right_tile = (p0, p1) if rig.cam0_is_physical_left else (p1, p0)
    top_row = np.hstack([left_tile, right_tile])
    cv2.line(top_row, (CAM_W, 0), (CAM_W, CAM_H), (35, 40, 48), 1)
    view3d = render_3d_viewport(
        rig,
        stereo,
        state,
        fps=fps,
        log_label=log_label,
        uart_label=uart_label,
    )
    return np.vstack([top_row, view3d])
