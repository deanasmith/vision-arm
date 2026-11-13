"""Host-side UART link that drives the arm from computed joint angles."""

from arm.protocol import Joint, angle_to_ccr, encode_joint_command
from arm.uart import DEFAULT_BAUD, DEFAULT_PORT, ArmUart, resolve_uart_port

__all__ = [
    "DEFAULT_BAUD",
    "DEFAULT_PORT",
    "ArmUart",
    "resolve_uart_port",
    "Joint",
    "angle_to_ccr",
    "encode_joint_command",
]
