"""Send joint commands to the STM32 over UART.

There is no host-side serial helper elsewhere in the repo. Settings are taken
from ``USART_Init`` in ``stm32/Core/Src/main.c``: 115200 baud, 8 data bits,
no parity, one stop bit, no flow control. The firmware listens on USART3
(PB10 TX, PB11 RX), which is an external UART rather than the ST-Link
virtual COM port.

The serial node is discovered. On macOS the single ``/dev/cu.usb*`` or
``/dev/cu.wchusbserial*`` callout device is used (never the matching
``tty.*`` node). On Linux the single ``/dev/ttyUSB*`` device is used, and
``/dev/ttyUSB0`` remains the default when that listing is empty. More than
one match is left for ``--port`` to choose.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Mapping

import serial

from arm.protocol import Joint, encode_joint_command

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200

_DARWIN_GLOBS = ("cu.usb*", "cu.wchusbserial*")
_LINUX_GLOBS = ("ttyUSB*",)


class UartSelectionError(Exception):
    """The USB-UART adapter could not be chosen without ``--port``."""


def list_uart_devices(platform: str | None = None, directory: str = "/dev") -> list[str]:
    """Return sorted callout paths for USB-UART adapters on this platform."""
    system = sys.platform if platform is None else platform
    patterns = _DARWIN_GLOBS if system == "darwin" else _LINUX_GLOBS
    found: set[str] = set()
    for pattern in patterns:
        found.update(glob.glob(os.path.join(directory, pattern)))
    return sorted(found)


def resolve_uart_port(
    port: str | None = None,
    *,
    platform: str | None = None,
    directory: str = "/dev",
) -> str:
    """Pick one serial device, or raise ``UartSelectionError``.

    An explicit ``port`` is returned unchanged. Otherwise the platform
    listing must contain exactly one adapter. Linux falls back to
    ``DEFAULT_PORT`` when ``/dev/ttyUSB*`` is empty.
    """
    if port is not None and port.strip():
        return port

    system = sys.platform if platform is None else platform
    found = list_uart_devices(system, directory)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        listing = "\n".join(f"  {path}" for path in found)
        raise UartSelectionError(
            "Several USB-UART adapters found. Pass --port to choose one:\n" + listing
        )
    if system == "darwin":
        raise UartSelectionError(
            "No USB-UART adapter found. "
            "Looked for /dev/cu.usb* and /dev/cu.wchusbserial*. "
            "Plug in the adapter or pass --port."
        )
    return DEFAULT_PORT

# set_servo_from_uart prints one of these before it reads the next command.
_ACK_MARKERS = (b"Setting CCR", b"Invalid joint", b"Unable to parse")


class ArmUart:
    """One open serial port. Each ``send_angle`` writes a single 3-byte frame.

    The STM32F0 USART has a one-byte receive register, and the firmware
    transmits a status line before it calls ``HAL_UART_Receive`` again. The
    next joint is held until that ack arrives, or until a short timeout when
    the adapter does not wire STM32 TX back to the host.
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        timeout: float = 0.02,
        serial_port: serial.Serial | None = None,
    ) -> None:
        if baud <= 0:
            raise ValueError(f"baud must be positive, got {baud}")
        if serial_port is not None:
            self._ser = serial_port
        else:
            self._ser = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
                write_timeout=0.2,
                rtscts=False,
                dsrdtr=False,
                xonxoff=False,
                exclusive=True,
            )
        self.port = port
        self.baud = baud

    def send_angle(self, joint: Joint | int, angle_deg: float) -> None:
        self._ser.write(encode_joint_command(joint, angle_deg))
        self._ser.flush()
        self._drain_ack()

    def send_angles(self, angles: Mapping[Joint | int, float]) -> None:
        for joint, angle in angles.items():
            if angle is None:
                continue
            self.send_angle(joint, angle)

    def close(self) -> None:
        self._ser.close()

    def _drain_ack(self) -> None:
        for _ in range(3):
            line = self._ser.readline()
            if not line:
                return
            if any(marker in line for marker in _ACK_MARKERS):
                return
