"""Mono UART startup without opening a camera."""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm.uart import UartSelectionError


def load_pose():
    spec = importlib.util.spec_from_file_location(
        "pose_estimation", ROOT / "pose-estimation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MonoStartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pose = load_pose()

    def touch(self, name: str) -> str:
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8"):
            pass
        return path

    def start(self, port=None, platform="darwin"):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            link = self.pose.setup_link(
                port,
                115200,
                platform=platform,
                directory=self.tmp.name,
            )
        if link is not None:
            link.close()
        return link, buf.getvalue()

    def test_missing_adapter_warns_and_disables_uart(self):
        self.touch("cu.debug-console")
        link, text = self.start()
        self.assertIsNone(link)
        self.assertEqual(text.strip(), self.pose.NO_ADAPTER_WARNING)
        self.assertIn("No USB-UART adapter found", text)
        self.assertIn("The camera loop will keep running.", text)

    def test_several_adapters_still_require_a_choice(self):
        self.touch("cu.usbserial-A")
        self.touch("cu.wchusbserial110")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            with self.assertRaises(UartSelectionError) as caught:
                self.pose.setup_link(
                    None,
                    115200,
                    platform="darwin",
                    directory=self.tmp.name,
                )
        message = str(caught.exception)
        printed = buf.getvalue()
        self.assertIn("cu.usbserial-A", message)
        self.assertIn("cu.wchusbserial110", message)
        self.assertIn("--port", message)
        self.assertIn(message, printed)
        self.assertNotIn("camera loop will keep running", printed)

    def test_explicit_port_is_not_treated_as_missing(self):
        link, text = self.start("/dev/cu.not-a-real-adapter")
        self.assertIsNone(link)
        self.assertIn("could not open /dev/cu.not-a-real-adapter", text)
        self.assertIn("The camera loop will keep running.", text)
        self.assertNotIn("No USB-UART adapter found", text)

    def test_single_adapter_is_selected_instead_of_disabled(self):
        chosen = self.touch("cu.usbserial-10")
        link, text = self.start()
        self.assertIsNone(link)
        self.assertIn(f"could not open {chosen}", text)
        self.assertNotIn("No USB-UART adapter found", text)
