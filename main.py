#!/usr/bin/env python3
"""Entry point for stereo pose tracking and Charuco calibration."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dual-camera pose tracking with Charuco stereo triangulation",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate-board", help="Write a printable Charuco board image")
    sub.add_parser("calibrate", help="Capture Charuco views and save stereo calibration")
    sub.add_parser("track", help="Run dual-camera triangulated pose tracking")
    sub.add_parser("mono", help="Run the original single-camera pose demo")

    args, rest = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        print("\nOriginal single-camera demo: python pose-estimation.py")
        return 0

    if args.command == "generate-board":
        from calibrate import main as calibrate_main
        return calibrate_main(["--generate-board", *rest])
    if args.command == "calibrate":
        from calibrate import main as calibrate_main
        return calibrate_main(rest)
    if args.command == "track":
        from stereo_pose import main as track_main
        return track_main(rest)
    if args.command == "mono":
        import runpy
        sys.argv = ["pose-estimation.py", *rest]
        runpy.run_path("pose-estimation.py", run_name="__main__")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
