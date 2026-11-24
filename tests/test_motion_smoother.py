import time
import unittest

from arm.motion import JointMotionSmoother
from arm.protocol import Joint
from arm.uart import ArmUart, AsyncArmUart


class _RecordingSerial:
    def __init__(self) -> None:
        self.frames: list[tuple[int, int]] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        joint = data[0]
        ccr = (data[1] << 8) | data[2]
        self.frames.append((joint, ccr))
        return len(data)

    def read(self, n: int = 1) -> bytes:
        return b"Setting CCR to 'ok'\n"

    def close(self) -> None:
        self.closed = True


class MotionSmootherTests(unittest.TestCase):
    def test_single_frame_180_spike_is_rejected_without_twitch(self):
        smoother = JointMotionSmoother()
        smoother.update_targets({Joint.HIP: 60.0})
        smoother.update_targets({Joint.HIP: 60.0})
        # One bad frame teleports to 180, then immediately returns to 60.
        smoother.update_targets({Joint.HIP: 180.0})
        pos_during = smoother.step(0.03)[Joint.HIP]
        smoother.update_targets({Joint.HIP: 60.0})
        pos_after = smoother.step(0.03)[Joint.HIP]
        self.assertAlmostEqual(pos_during, 60.0, places=3)
        self.assertAlmostEqual(pos_after, 60.0, places=3)

    def test_step_response_respects_velocity_cap_and_converges_without_overshoot(self):
        smoother = JointMotionSmoother(speed_scale=1.0)
        smoother.update_targets({Joint.SHOULDER: 20.0})
        smoother.update_targets({Joint.SHOULDER: 20.0})
        # Sustained move to 110 deg (+90 deg)
        smoother.update_targets({Joint.SHOULDER: 110.0})
        smoother.update_targets({Joint.SHOULDER: 110.0})

        dt = 0.02
        prev = 20.0
        max_vel = 0.0
        peak = 20.0
        for _ in range(50):  # 1.0 second
            pos = smoother.step(dt)[Joint.SHOULDER]
            vel = abs(pos - prev) / dt
            max_vel = max(max_vel, vel)
            peak = max(peak, pos)
            prev = pos

        cap = smoother.profile_for(Joint.SHOULDER).max_vel_deg_s
        self.assertLessEqual(max_vel, cap + 1e-6)
        self.assertLessEqual(peak, 110.2)  # no overshoot
        self.assertAlmostEqual(prev, 110.0, delta=0.5)

    def test_async_uart_with_smoother_interpolates_smoothly(self):
        fake = _RecordingSerial()
        link = ArmUart("/dev/fake", serial_port=fake)
        smoother = JointMotionSmoother(speed_scale=1.5)
        with AsyncArmUart(link, deadband_deg=1.5, refresh_s=10.0, smoother=smoother, tick_hz=50.0) as sender:
            sender.submit({Joint.ELBOW: 30.0})
            time.sleep(0.06)
            sender.submit({Joint.ELBOW: 30.0})
            time.sleep(0.06)
            # Jump target to 120 deg (needs 2 frames to pass 3-sample median)
            sender.submit({Joint.ELBOW: 120.0})
            time.sleep(0.05)
            sender.submit({Joint.ELBOW: 120.0})
            time.sleep(0.55)

        elbow_ccrs = [ccr for j, ccr in fake.frames if j == int(Joint.ELBOW)]
        # Should have multiple smooth intermediate CCR steps descending from 30 deg toward 120 deg
        self.assertGreaterEqual(len(elbow_ccrs), 4)
        self.assertGreater(elbow_ccrs[0], elbow_ccrs[-1])
        # Monotonic decrease (no reversal / overshoot!)
        for a, b in zip(elbow_ccrs, elbow_ccrs[1:]):
            self.assertGreaterEqual(a, b)


if __name__ == "__main__":
    unittest.main()
