"""Encode joint angles as the STM32 UART command.

The board (``stm32/Core/Src/main.c``) reads exactly three bytes per command:

* byte 0: joint id (``Joint`` in ``stm32/Core/Inc/main.h``)
* byte 1: CCR most-significant byte
* byte 2: CCR least-significant byte

``parse_new_ccr`` rebuilds the compare value as ``(byte1 << 8) | byte2`` and
writes it straight into the timer. USART3 is 115200 8N1 with no flow control.

The servo window matches ``stm32/Core/Inc/main.h`` and the STM32 communication
protocol doc: ``SERVO_MAX_CCR`` (12.5% of ARR 64000 = 8000, 2.5 ms) is the
minimum servo angle (0°, firmware ``TIMER_DEFAULT_CCR``), and ``SERVO_MIN_CCR``
(2.5% of ARR 64000 = 1600, 0.5 ms) is the maximum servo angle (180°).
"""

from __future__ import annotations

import math
from enum import IntEnum

# stm32/Core/Inc/main.h — 48 MHz timer clock, PSC 14, ARR 64000, 50 Hz.
TIMER_ARR = 64000
DUTY_AT_MIN_ANGLE = 0.125
DUTY_AT_MAX_ANGLE = 0.025
MAX_ANGLE_DEG = 180.0
SERVO_MAX_CCR = int(DUTY_AT_MIN_ANGLE * TIMER_ARR)  # 8000, 0 deg (TIMER_DEFAULT_CCR)
SERVO_MIN_CCR = int(DUTY_AT_MAX_ANGLE * TIMER_ARR)  # 1600, 180 deg
SERVO_ZERO_CCR = SERVO_MAX_CCR
SERVO_FULL_CCR = SERVO_MIN_CCR


class Joint(IntEnum):
    """Joint ids in PCB header order. Values match the firmware enum."""

    HIP = 0
    HAND = 1
    ELBOW = 2
    WRIST = 3
    SHOULDER = 4


def angle_to_ccr(angle_deg: float) -> int:
    """Map an absolute angle in degrees to a timer CCR in ``[1600, 8000]``.

    ``ccr = 8000 - round((angle / 180) * (8000 - 1600))``. 0° maps to CCR 8000
    (12.5% duty, 2.5 ms) and 180° maps to CCR 1600 (2.5% duty, 0.5 ms). Angles
    outside 0°–180° are clamped.
    """
    angle = float(angle_deg)
    if not math.isfinite(angle):
        raise ValueError(f"angle must be finite, got {angle_deg!r}")
    if angle < 0.0:
        angle = 0.0
    elif angle > MAX_ANGLE_DEG:
        angle = MAX_ANGLE_DEG
    span = SERVO_MAX_CCR - SERVO_MIN_CCR
    delta = int(round(angle / MAX_ANGLE_DEG * span))
    return SERVO_MAX_CCR - delta


def encode_joint_command(joint: Joint | int, angle_deg: float) -> bytes:
    """Return the three-byte UART command for one joint."""
    joint_id = int(joint)
    if joint_id < int(Joint.HIP) or joint_id > int(Joint.SHOULDER):
        raise ValueError(f"joint id {joint_id} is outside 0..4")
    ccr = angle_to_ccr(angle_deg)
    return bytes((joint_id, (ccr >> 8) & 0xFF, ccr & 0xFF))
