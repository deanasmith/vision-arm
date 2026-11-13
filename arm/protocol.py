"""Encode joint angles as the STM32 UART command.

The board (``stm32/Core/Src/main.c``) reads exactly three bytes per command:

* byte 0: joint id (``Joint`` in ``stm32/Core/Inc/main.h``)
* byte 1: CCR most-significant byte
* byte 2: CCR least-significant byte

``parse_new_ccr`` rebuilds the compare value as ``(byte1 << 8) | byte2`` and
writes it straight into the timer. USART3 is 115200 8N1 with no flow control.

0° stays on the firmware power-on pulse: CCR 8000, 2.5 ms, 12.5% of ARR.
The chart's other end (CCR 1600, 0.5 ms) is only 2.0 ms away. These servos
travel about 90° across that 2.0 ms, and other pulse widths already drive a
full 180° turn, so one arm degree has to move the pulse twice as far.

2.5 ms minus 4.0 ms is below 0. The 4.0 ms span therefore runs the other
way, to 6.5 ms (CCR 20800), which is still inside the 20 ms frame. The
firmware writes the CCR it is sent and does not clamp it to 1600..8000.
"""

from __future__ import annotations

import math
from enum import IntEnum

# stm32/Core/Inc/main.h — 48 MHz timer clock, PSC 14, ARR 64000, 50 Hz.
TIMER_ARR = 64000
MAX_ANGLE_DEG = 180.0
# 12.5% of ARR. Firmware TIMER_DEFAULT_CCR, the hanging / power-on pose.
SERVO_ZERO_CCR = 8000
# Chart window 8000→1600 is 6400 counts (2.0 ms) and about 90° of horn.
# 180° needs 12800 counts (4.0 ms). Added to the zero pulse: 6.5 ms.
SERVO_SPAN_CCR = 12800
SERVO_FULL_CCR = SERVO_ZERO_CCR + SERVO_SPAN_CCR  # 20800


class Joint(IntEnum):
    """Joint ids in PCB header order. Values match the firmware enum."""

    HIP = 0
    HAND = 1
    ELBOW = 2
    WRIST = 3
    SHOULDER = 4


def angle_to_ccr(angle_deg: float) -> int:
    """Map an absolute angle in degrees to a timer CCR.

    ``ccr = 8000 + (angle / 180) * 12800``. 0° is CCR 8000 (2.5 ms) and 180°
    is CCR 20800 (6.5 ms). Angles outside 0°–180° are clamped.
    """
    angle = float(angle_deg)
    if not math.isfinite(angle):
        raise ValueError(f"angle must be finite, got {angle_deg!r}")
    if angle < 0.0:
        angle = 0.0
    elif angle > MAX_ANGLE_DEG:
        angle = MAX_ANGLE_DEG
    delta = int(round(angle / MAX_ANGLE_DEG * SERVO_SPAN_CCR))
    return SERVO_ZERO_CCR + delta


def encode_joint_command(joint: Joint | int, angle_deg: float) -> bytes:
    """Return the three-byte UART command for one joint."""
    joint_id = int(joint)
    if joint_id < int(Joint.HIP) or joint_id > int(Joint.SHOULDER):
        raise ValueError(f"joint id {joint_id} is outside 0..4")
    ccr = angle_to_ccr(angle_deg)
    return bytes((joint_id, (ccr >> 8) & 0xFF, ccr & 0xFF))
