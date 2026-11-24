"""Critically-damped, velocity- and acceleration-limited joint motion smoother for UART servo control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping

from arm.protocol import Joint, MAX_ANGLE_DEG


@dataclass(frozen=True)
class JointMotionProfile:
    """Dynamic motion limits for a single robot joint."""

    max_vel_deg_s: float
    max_accel_deg_s2: float
    response_hz: float
    deadband_deg: float


DEFAULT_JOINT_PROFILES: dict[Joint, JointMotionProfile] = {
    Joint.HIP: JointMotionProfile(max_vel_deg_s=140.0, max_accel_deg_s2=650.0, response_hz=3.2, deadband_deg=1.5),
    Joint.SHOULDER: JointMotionProfile(max_vel_deg_s=150.0, max_accel_deg_s2=700.0, response_hz=3.5, deadband_deg=1.5),
    Joint.ELBOW: JointMotionProfile(max_vel_deg_s=200.0, max_accel_deg_s2=950.0, response_hz=4.2, deadband_deg=1.5),
    Joint.WRIST: JointMotionProfile(max_vel_deg_s=240.0, max_accel_deg_s2=1200.0, response_hz=4.5, deadband_deg=2.0),
    Joint.HAND: JointMotionProfile(max_vel_deg_s=280.0, max_accel_deg_s2=1500.0, response_hz=5.0, deadband_deg=2.0),
}


@dataclass
class _JointState:
    pos: float
    vel: float
    target: float
    window: deque[float]


class JointMotionSmoother:
    """Turns jittery vision angle targets into smooth, fast, non-whippy servo trajectories.

    Pipeline per joint:
    1. 3-sample median filter on incoming vision targets to reject 1-frame 180 deg spikes.
    2. Critically-damped 2nd-order follower with explicit max acceleration and max velocity
       clamps, stepped at high rate (~33 Hz) so servos glide instead of jerking at camera FPS.
    3. Zero overshoot and clean settling inside the deadband when holding still.
    """

    def __init__(
        self,
        profiles: Mapping[Joint, JointMotionProfile] | None = None,
        *,
        speed_scale: float = 1.0,
        median_window: int = 3,
    ) -> None:
        if speed_scale <= 0.0 or not math.isfinite(speed_scale):
            raise ValueError(f"speed_scale must be finite and positive, got {speed_scale!r}")
        base = DEFAULT_JOINT_PROFILES if profiles is None else profiles
        self.profiles: dict[Joint, JointMotionProfile] = {
            Joint(k): JointMotionProfile(
                max_vel_deg_s=p.max_vel_deg_s * speed_scale,
                max_accel_deg_s2=p.max_accel_deg_s2 * speed_scale,
                response_hz=p.response_hz * math.sqrt(speed_scale),
                deadband_deg=p.deadband_deg,
            )
            for k, p in base.items()
        }
        self._median_window = max(1, int(median_window))
        self._states: dict[Joint, _JointState] = {}

    def profile_for(self, joint: Joint) -> JointMotionProfile:
        return self.profiles.get(
            joint,
            JointMotionProfile(max_vel_deg_s=180.0, max_accel_deg_s2=800.0, response_hz=3.5, deadband_deg=1.5),
        )

    def update_targets(self, angles: Mapping[Joint | int, float]) -> None:
        """Ingest new vision targets (typically at camera frame rate)."""
        for joint_key, raw in angles.items():
            val = float(raw)
            if not math.isfinite(val):
                continue
            val = max(0.0, min(MAX_ANGLE_DEG, val))
            joint = Joint(int(joint_key))
            state = self._states.get(joint)
            if state is None:
                win: deque[float] = deque([val], maxlen=self._median_window)
                self._states[joint] = _JointState(pos=val, vel=0.0, target=val, window=win)
                continue
            state.window.append(val)
            ordered = sorted(state.window)
            state.target = ordered[len(ordered) // 2]

    def step(self, dt: float) -> dict[Joint, float]:
        """Advance all initialized joints by dt seconds and return smooth angles."""
        if dt <= 0.0 or not math.isfinite(dt):
            return {j: s.pos for j, s in self._states.items()}
        # Sub-step if a scheduling hiccup caused a large dt so integration stays unconditionally stable.
        dt_clamped = min(float(dt), 0.15)
        substeps = max(1, int(math.ceil(dt_clamped / 0.02)))
        sub_dt = dt_clamped / substeps

        out: dict[Joint, float] = {}
        for joint, state in self._states.items():
            prof = self.profile_for(joint)
            omega = 2.0 * math.pi * prof.response_hz
            for _ in range(substeps):
                err = state.target - state.pos
                if abs(err) < 0.08 and abs(state.vel) < 0.8:
                    state.pos = state.target
                    state.vel = 0.0
                    break
                # Smooth proportional velocity capped by both max_vel and exact zero-overshoot braking speed.
                v_brake = math.sqrt(2.0 * prof.max_accel_deg_s2 * abs(err))
                v_limit = min(prof.max_vel_deg_s, v_brake)
                v_target = max(-v_limit, min(v_limit, err * omega))
                dv = v_target - state.vel
                max_dv = prof.max_accel_deg_s2 * sub_dt
                vel = state.vel + max(-max_dv, min(max_dv, dv))
                vel = max(-prof.max_vel_deg_s, min(prof.max_vel_deg_s, vel))
                pos = state.pos + vel * sub_dt
                if (err > 0.0 and pos > state.target) or (err < 0.0 and pos < state.target):
                    pos = state.target
                    vel = 0.0
                pos = max(0.0, min(MAX_ANGLE_DEG, pos))
                state.pos = pos
                state.vel = vel
            out[joint] = state.pos
        return out

    def positions(self) -> dict[Joint, float]:
        return {j: s.pos for j, s in self._states.items()}

    def reset(self) -> None:
        self._states.clear()
