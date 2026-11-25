"""Desktop entry point for both Python and the windowed Windows build."""

import sys


def main() -> int:
    if sys.platform == "win32" and getattr(sys, "frozen", False) and sys.argv[1:]:
        import ctypes

        # Preserve inherited redirection if present. A double-click has no
        # arguments and never attaches a terminal.
        if ctypes.windll.kernel32.AttachConsole(-1):
            if sys.stdout is None:
                sys.stdout = open("CONOUT$", "w", encoding="utf-8")
            if sys.stderr is None:
                sys.stderr = open("CONOUT$", "w", encoding="utf-8")

    from image_sharpening.cli import main as dispatch
    return dispatch(desktop_default=True)


if __name__ == "__main__":
    raise SystemExit(main())
