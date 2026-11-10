"""Send joint commands to the STM32 over UART.

There is no host-side serial helper elsewhere in the repo. Settings are taken
from ``USART_Init`` in ``stm32/Core/Src/main.c``: 115200 baud, 8 data bits,
no parity, one stop bit, no flow control. The firmware listens on USART3
(PB10 TX, PB11 RX), which is an external UART rather than the ST-Link
virtual COM port.

The serial node is discovered. On macOS the single ``/dev/cu.usb*`` or
``/dev/cu.wchusbserial*`` callout device is used (never the matching
``tty.*`` node). When none of those are present, discovery returns None
and mono mode warns and keeps running with UART disabled. On Linux the
single ``/dev/ttyUSB*`` device is used, and ``/dev/ttyUSB0`` remains the
default when that listing is empty. More than one match is left for
``--port`` to choose.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Mapping

import serial

from arm.protocol import Joint, encode_joint_command

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
# Long enough for the firmware's two status lines on a USB-UART adapter,
# short enough that a missing line cannot stall the camera.
ACK_TIMEOUT = 0.1

_DARWIN_GLOBS = ("cu.usb*", "cu.wchusbserial*")
_LINUX_GLOBS = ("ttyUSB*",)


class UartSelectionError(Exception):
    """Several USB-UART adapters matched and ``--port`` was not passed."""


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
) -> str | None:
    """Pick one serial device.

    An explicit ``port`` is returned unchanged. Otherwise the platform
    listing must contain exactly one adapter. macOS returns None when
    that listing is empty so the caller can run with UART disabled.
    Linux falls back to ``DEFAULT_PORT`` when ``/dev/ttyUSB*`` is empty.
    More than one match raises ``UartSelectionError``.
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
        return None
    return DEFAULT_PORT

# set_servo_from_uart prints one of these before it reads the next command.
_ACK_MARKERS = (b"Setting CCR", b"Invalid joint", b"Unable to parse")


class ArmUart:
    """One open serial port. Each ``send_angle`` writes a single 3-byte frame.

    The STM32F0 USART has a one-byte receive register, and the firmware
    transmits a status line before it calls ``HAL_UART_Receive`` again. The
    next joint waits for that line, but only until ``ACK_TIMEOUT``. A missing
    line or a failed read does not undo the write and does not close the port.
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        timeout: float = ACK_TIMEOUT,
        serial_port: serial.Serial | None = None,
    ) -> None:
        if baud <= 0:
            raise ValueError(f"baud must be positive, got {baud}")
        if timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        self._ack_timeout = timeout
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
        # The frame is committed before any ack read. A missing "Setting CCR"
        # line, or a read error while waiting for it, must not discard this
        # write or the next joint.
        self._ser.write(encode_joint_command(joint, angle_deg))
        try:
            self._ser.flush()
        except OSError:
            pass
        self._drain_ack()

    def send_angles(self, angles: Mapping[Joint | int, float]) -> None:
        for joint, angle in angles.items():
            if angle is None:
                continue
            self.send_angle(joint, angle)

    def close(self) -> None:
        self._ser.close()

    def _drain_ack(self) -> None:
        """Wait out the firmware's status text, then return either way.

        ``set_servo_from_uart`` prints ``Setting CCR`` and only then calls
        ``HAL_UART_Receive`` again. The read is best-effort: the line is not
        required for the command already written, and a serial error here is
        not a failed write.
        """
        deadline = time.monotonic() + self._ack_timeout
        while time.monotonic() < deadline:
            started = time.monotonic()
            try:
                line = self._ser.readline()
            except OSError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                time.sleep(min(0.01, remaining))
                continue
            if line and any(marker in line for marker in _ACK_MARKERS):
                return
            now = time.monotonic()
            if now >= deadline:
                return
            # An empty read that already blocked used up the port timeout.
            # An instant empty read has not actually waited.
            if not line and (now - started) >= 0.01:
                return
            if not line:
                time.sleep(min(0.01, deadline - now))
