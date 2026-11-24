"""Host-side UART link that drives the arm from computed joint angles."""

from arm.protocol import (
    SERVO_MAX_CCR,
    SERVO_MIN_CCR,
    Joint,
    angle_to_ccr,
    encode_joint_command,
)
from arm.uart import (
    DEFAULT_BAUD,
    DEFAULT_PORT,
    ArmUart,
    AsyncArmUart,
    UartSelectionError,
    resolve_uart_port,
)

__all__ = [
    "ArmUart",
    "AsyncArmUart",
    "DEFAULT_BAUD",
    "DEFAULT_PORT",
    "Joint",
    "SERVO_MAX_CCR",
    "SERVO_MIN_CCR",
    "UartSelectionError",
    "angle_to_ccr",
    "encode_joint_command",
    "resolve_uart_port",
]
