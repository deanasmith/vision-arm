import os
import tempfile
import unittest
from typing import List

from arm.protocol import (
    SERVO_MAX_CCR,
    SERVO_MIN_CCR,
    Joint,
    angle_to_ccr,
    encode_joint_command,
)
from arm.uart import DEFAULT_PORT, ArmUart, UartSelectionError, resolve_uart_port


class _FakeSerial:
    def __init__(self, lines: List[bytes] | None = None) -> None:
        self.written = b""
        self.lines = list(lines or [])
        self.closed = False

    def write(self, data: bytes) -> int:
        self.written += bytes(data)
        return len(data)

    def flush(self) -> None:
        return None

    def readline(self) -> bytes:
        if not self.lines:
            return b""
        return self.lines.pop(0)

    def close(self) -> None:
        self.closed = True


class AngleToCcrTests(unittest.TestCase):
    def test_endpoints_match_firmware_constants(self):
        self.assertEqual(SERVO_MAX_CCR, 8000)
        self.assertEqual(SERVO_MIN_CCR, 1600)
        self.assertEqual(angle_to_ccr(0), 8000)
        self.assertEqual(angle_to_ccr(180), 1600)

    def test_nine_degrees_subtracts_span_from_min_angle_pulse(self):
        # 9/180 of the 6400-count window is 320. The protocol worksheet
        # stops there (CCR 320). The firmware window is 1600..8000, and
        # 0° is the 8000-count end, so the pulse is 8000 - 320.
        self.assertEqual(angle_to_ccr(9), 7680)

    def test_midpoint(self):
        self.assertEqual(angle_to_ccr(90), 4800)

    def test_clamps_to_servo_travel(self):
        self.assertEqual(angle_to_ccr(-10), 8000)
        self.assertEqual(angle_to_ccr(270), 1600)

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
        # joint 2, CCR 7680 = 0x1E00
        self.assertEqual(encode_joint_command(Joint.ELBOW, 9), bytes((0x02, 0x1E, 0x00)))

    def test_shoulder_at_rest_is_startup_pulse(self):
        # joint 4, CCR 8000 = 0x1F40 (TIMER_DEFAULT_CCR)
        self.assertEqual(encode_joint_command(Joint.SHOULDER, 0), bytes((0x04, 0x1F, 0x40)))

    def test_straight_elbow(self):
        # joint 2, CCR 1600 = 0x0640
        self.assertEqual(encode_joint_command(2, 180), bytes((0x02, 0x06, 0x40)))

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
                b"Setting CCR to '1600'\n",
            ]
        )
        link = ArmUart(serial_port=fake)
        link.send_angles({Joint.SHOULDER: 0, Joint.ELBOW: 180})
        self.assertEqual(
            fake.written,
            bytes((0x04, 0x1F, 0x40, 0x02, 0x06, 0x40)),
        )
        self.assertEqual(fake.lines, [])

    def test_stops_when_adapter_does_not_reply(self):
        fake = _FakeSerial()
        link = ArmUart(serial_port=fake)
        link.send_angle(Joint.ELBOW, 9)
        link.close()
        self.assertEqual(fake.written, bytes((0x02, 0x1E, 0x00)))
        self.assertTrue(fake.closed)

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

    def test_darwin_missing_adapter(self):
        self.touch("cu.debug-console")
        with self.assertRaises(UartSelectionError) as caught:
            resolve_uart_port(platform="darwin", directory=self.tmp.name)
        self.assertIn("No USB-UART adapter found", str(caught.exception))

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
