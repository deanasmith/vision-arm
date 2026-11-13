import os
import tempfile
import threading
import time
import unittest
from typing import List

import serial

from arm.protocol import (
    SERVO_FULL_CCR,
    SERVO_ZERO_CCR,
    Joint,
    angle_to_ccr,
    encode_joint_command,
)
from arm.uart import DEFAULT_PORT, ArmUart, UartSelectionError, resolve_uart_port


class _FakeSerial:
    def __init__(self, lines: List[bytes] | None = None) -> None:
        self.written = b""
        self.lines = list(lines or [])
        self._rx = bytearray(b"".join(self.lines))
        self.closed = False
        self.timeout = 0.02
        self.flush_calls = 0

    def write(self, data: bytes) -> int:
        self.written += bytes(data)
        return len(data)

    def flush(self) -> None:
        self.flush_calls += 1
        raise AssertionError("flush/tcdrain must not gate the next joint write")

    @property
    def in_waiting(self) -> int:
        return len(self._rx)

    def read(self, size: int = 1) -> bytes:
        if size is None or size <= 0 or not self._rx:
            return b""
        chunk = bytes(self._rx[:size])
        del self._rx[:size]
        if not self._rx:
            self.lines.clear()
        return chunk

    def close(self) -> None:
        self.closed = True


class _EndlessNoise:
    """RX bytes that never contain a newline or an ack marker."""

    def __init__(self) -> None:
        self.written = b""
        self.timeout = None

    def write(self, data: bytes) -> int:
        self.written += bytes(data)
        return len(data)

    @property
    def in_waiting(self) -> int:
        return 8

    def read(self, size: int = 1) -> bytes:
        return b"X" * max(1, int(size))

    def flush(self) -> None:
        raise AssertionError("flush/tcdrain must not gate the next joint write")

    def close(self) -> None:
        return None


class AngleToCcrTests(unittest.TestCase):
    def test_endpoints(self):
        # 0° stays on the power-on pulse. 180° is 4.0 ms longer (12800 counts).
        self.assertEqual(SERVO_ZERO_CCR, 8000)
        self.assertEqual(SERVO_FULL_CCR, 20800)
        self.assertEqual(angle_to_ccr(0), 8000)
        self.assertEqual(angle_to_ccr(180), 20800)

    def test_nine_degrees_is_twice_the_old_chart_step(self):
        # 9/180 of the old 6400-count chart window is 320. The horn only
        # covered about half the requested angle across that window, so the
        # step is 640 counts and the pulse is 8000 + 640.
        self.assertEqual(angle_to_ccr(9), 8640)

    def test_midpoint(self):
        self.assertEqual(angle_to_ccr(90), 14400)

    def test_one_degree_moves_one_degree_of_the_4ms_span(self):
        # 12800/180 = 71.111 counts, which is 1/180 of the 4.0 ms span.
        self.assertEqual(angle_to_ccr(1), 8071)

    def test_clamps_to_servo_travel(self):
        self.assertEqual(angle_to_ccr(-10), 8000)
        self.assertEqual(angle_to_ccr(270), 20800)

    def test_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            angle_to_ccr(float("nan"))


class PacketTests(unittest.TestCase):
    def test_joint_ids_match_firmware_enum(self):
        self.assertEqual(
            [Joint.HIP, Joint.HAND, Joint.ELBOW, Joint.WRIST, Joint.SHOULDER],
            [0, 1, 2, 3, 4],
        )

    def test_elbow_nine_degrees_is_big_endian(self):
        # joint 2, CCR 8640 = 0x21C0
        self.assertEqual(encode_joint_command(Joint.ELBOW, 9), bytes((0x02, 0x21, 0xC0)))

    def test_shoulder_at_rest_is_startup_pulse(self):
        # joint 4, CCR 8000 = 0x1F40 (TIMER_DEFAULT_CCR)
        self.assertEqual(encode_joint_command(Joint.SHOULDER, 0), bytes((0x04, 0x1F, 0x40)))

    def test_straight_elbow(self):
        # joint 2, CCR 20800 = 0x5140
        self.assertEqual(encode_joint_command(2, 180), bytes((0x02, 0x51, 0x40)))

    def test_rejects_unknown_joint(self):
        with self.assertRaises(ValueError):
            encode_joint_command(5, 10)


class ArmUartTests(unittest.TestCase):
    def test_waits_for_ack_between_joints(self):
        fake = _FakeSerial(
            [
                b"Joint ID is '4' (SHOULDER)\n",
                b"Setting CCR to '8000'\n",
                b"Joint ID is '2' (ELBOW)\n",
                b"Setting CCR to '20800'\n",
            ]
        )
        link = ArmUart(serial_port=fake)
        link.send_angles({Joint.SHOULDER: 0, Joint.ELBOW: 180})
        self.assertEqual(
            fake.written,
            bytes((0x04, 0x1F, 0x40, 0x02, 0x51, 0x40)),
        )
        self.assertEqual(fake.lines, [])

    def test_stops_when_adapter_does_not_reply(self):
        fake = _FakeSerial()
        link = ArmUart(serial_port=fake, timeout=0.02)
        link.send_angle(Joint.ELBOW, 9)
        link.close()
        self.assertEqual(fake.written, bytes((0x02, 0x21, 0xC0)))
        self.assertEqual(fake.flush_calls, 0)
        self.assertTrue(fake.closed)

    def test_ack_read_error_still_writes_both_joints(self):
        # A macOS serial read can raise SerialException (an OSError) when the
        # firmware's "Setting CCR" line is not in the buffer yet. That must
        # not cancel the frame already written or skip the other joint.
        class _AckFails(_FakeSerial):
            def read(self, size: int = 1) -> bytes:
                raise serial.SerialException(
                    "device reports readiness to read but returned no data "
                    "(device disconnected or multiple access on port?)"
                )

        fake = _AckFails()
        link = ArmUart(serial_port=fake, timeout=0.02)
        link.send_angles({Joint.SHOULDER: 90, Joint.ELBOW: 180})
        self.assertEqual(
            fake.written,
            bytes((0x04, 0x38, 0x40, 0x02, 0x51, 0x40)),
        )

    def test_visible_shoulder_and_elbow_are_both_written(self):
        fake = _FakeSerial()
        link = ArmUart(serial_port=fake)
        link.send_angles({Joint.SHOULDER: 42.0, Joint.ELBOW: 90.0})
        self.assertEqual(fake.flush_calls, 0)
        self.assertEqual(
            fake.written,
            encode_joint_command(Joint.SHOULDER, 42)
            + encode_joint_command(Joint.ELBOW, 90),
        )

    def test_noise_without_newline_cannot_stall_the_next_joint(self):
        port = _EndlessNoise()
        link = ArmUart(timeout=0.05, serial_port=port)
        done = threading.Event()

        def run() -> None:
            link.send_angles({Joint.SHOULDER: 42.0, Joint.ELBOW: 90.0})
            done.set()

        thread = threading.Thread(target=run, daemon=True)
        started = time.monotonic()
        thread.start()
        self.assertTrue(done.wait(1.0), "ack wait stalled the elbow write")
        thread.join(0.2)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(
            port.written,
            encode_joint_command(Joint.SHOULDER, 42)
            + encode_joint_command(Joint.ELBOW, 90),
        )

    def test_pty_writes_both_joints_when_output_is_not_drained(self):
        # macOS tcdrain blocks until the tty output queue is empty. Nothing
        # is reading this PTY, which is the hang that dropped the elbow.
        master, slave = os.openpty()
        link = None
        try:
            link = ArmUart(os.ttyname(slave), timeout=0.02)
            done = threading.Event()
            errors: List[BaseException] = []

            def run() -> None:
                try:
                    link.send_angles({Joint.SHOULDER: 42.0, Joint.ELBOW: 90.0})
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    done.set()

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            self.assertTrue(done.wait(1.0), "tcdrain stalled the joint writes")
            thread.join(0.2)
            self.assertEqual(errors, [])
            import fcntl

            flags = fcntl.fcntl(master, fcntl.F_GETFL)
            fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            written = b""
            while True:
                try:
                    chunk = os.read(master, 64)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                written += chunk
            self.assertEqual(
                written,
                encode_joint_command(Joint.SHOULDER, 42)
                + encode_joint_command(Joint.ELBOW, 90),
            )
        finally:
            if link is not None:
                link.close()
            else:
                os.close(slave)
            os.close(master)

    def test_rejects_non_positive_baud(self):
        with self.assertRaises(ValueError):
            ArmUart(baud=0, serial_port=_FakeSerial())


class ResolveUartPortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def touch(self, name: str) -> str:
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8"):
            pass
        return path

    def test_darwin_picks_the_only_usb_callout(self):
        chosen = self.touch("cu.usbserial-10")
        self.touch("tty.usbserial-10")
        self.touch("cu.Bluetooth-Incoming-Port")
        self.assertEqual(
            resolve_uart_port(platform="darwin", directory=self.tmp.name),
            chosen,
        )

    def test_darwin_picks_a_wch_adapter(self):
        chosen = self.touch("cu.wchusbserial110")
        self.assertEqual(
            resolve_uart_port(platform="darwin", directory=self.tmp.name),
            chosen,
        )

    def test_darwin_lists_several_adapters(self):
        self.touch("cu.usbserial-A")
        self.touch("cu.wchusbserial110")
        with self.assertRaises(UartSelectionError) as caught:
            resolve_uart_port(platform="darwin", directory=self.tmp.name)
        message = str(caught.exception)
        self.assertIn("cu.usbserial-A", message)
        self.assertIn("cu.wchusbserial110", message)
        self.assertIn("--port", message)

    def test_darwin_missing_adapter_returns_none(self):
        self.touch("cu.debug-console")
        self.assertIsNone(
            resolve_uart_port(platform="darwin", directory=self.tmp.name)
        )

    def test_linux_uses_the_only_ttyusb_node(self):
        chosen = self.touch("ttyUSB1")
        self.assertEqual(
            resolve_uart_port(platform="linux", directory=self.tmp.name),
            chosen,
        )

    def test_linux_defaults_when_no_ttyusb_node_is_listed(self):
        self.assertEqual(
            resolve_uart_port(platform="linux", directory=self.tmp.name),
            DEFAULT_PORT,
        )

    def test_linux_lists_several_ttyusb_nodes(self):
        self.touch("ttyUSB0")
        self.touch("ttyUSB1")
        with self.assertRaises(UartSelectionError) as caught:
            resolve_uart_port(platform="linux", directory=self.tmp.name)
        self.assertIn("ttyUSB0", str(caught.exception))
        self.assertIn("ttyUSB1", str(caught.exception))

    def test_explicit_port_skips_discovery(self):
        self.touch("cu.usbserial-A")
        self.touch("cu.usbserial-B")
        self.assertEqual(
            resolve_uart_port("/dev/cu.custom", platform="darwin", directory=self.tmp.name),
            "/dev/cu.custom",
        )
