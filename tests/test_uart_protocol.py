import unittest
from typing import List

from arm.protocol import (
    SERVO_MAX_CCR,
    SERVO_MIN_CCR,
    Joint,
    angle_to_ccr,
    encode_joint_command,
)
from arm.uart import ArmUart


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
