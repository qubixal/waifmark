#!/usr/bin/env python3
"""Build waifmark distributions — wheel and standalone executable.

Usage:
    python build.py              # build both wheel and executable
    python build.py --wheel      # wheel only
    python build.py --exe        # executable only
    python build.py --clean      # remove build artifacts first
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"


def run(cmd: list[str], **kwargs) -> None:
    print(f"\n{'='*60}\n  $ {' '.join(cmd)}\n{'='*60}")
    subprocess.run(cmd, cwd=str(ROOT), check=True, **kwargs)


def clean() -> None:
    for d in [DIST, BUILD, ROOT / "waifmark.egg-info"]:
        if d.exists():
            print(f"Removing {d}")
            shutil.rmtree(d)


def build_wheel() -> Path:
    run([sys.executable, "-m", "build", "--sdist", "--wheel", "."])
    wheels = list(DIST.glob("*.whl"))
    if not wheels:
        raise RuntimeError("No wheel found in dist/")
    print(f"\nBuilt wheel: {wheels[0].name}")
    return wheels[0]


def build_executable() -> Path:
    run([sys.executable, "-m", "PyInstaller", "waifmark.spec", "--noconfirm"])
    exe = DIST / "waifmark" / "waifmark"
    if not exe.exists():
        raise RuntimeError("Executable not found at dist/waifmark/waifmark")
    print(f"\nBuilt executable: {exe}")
    return exe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build waifmark distributions")
    parser.add_argument("--wheel", action="store_true", help="Build wheel only")
    parser.add_argument("--exe", action="store_true", help="Build executable only")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts first")
    args = parser.parse_args()

    build_all = not args.wheel and not args.exe

    if args.clean:
        clean()

    # Ensure build tools are installed
    run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "build", "pyinstaller"])

    if build_all or args.wheel:
        build_wheel()

    if build_all or args.exe:
        build_executable()

    print("\nDone. Artifacts in dist/")


if __name__ == "__main__":
    main()
