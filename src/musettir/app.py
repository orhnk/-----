from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from PyQt5 import QtWidgets

from .cli import parse_args
from .window import FaceDetectionWindow


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]

    try:
        config = parse_args(args)
    except argparse.ArgumentTypeError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    # Check for display
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print("Error: No display detected. This is a GUI application that requires a graphical environment (X11 or Wayland).", file=sys.stderr)
        return 1

    app = QtWidgets.QApplication(sys.argv)

    try:
        window = FaceDetectionWindow(config)
    except RuntimeError as exc:
        QtWidgets.QMessageBox.critical(None, "Face Detector", str(exc))
        return 1

    window.resize(960, 720)
    window.show()
    return app.exec()


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
