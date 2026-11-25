"""Locate source, external executable assets, or bundled assets independent of cwd."""

from pathlib import Path
import sys

INPUT_FOLDER = "HW2_test_image"
CONFIG_FILE = Path("configs/selected.json")
FROZEN = getattr(sys, "frozen", False)
PROJECT = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parents[2]


def resource_path(relative) -> Path:
    candidates = [PROJECT / relative]
    if FROZEN:
        if PROJECT.name.lower() == "dist":
            candidates.append(PROJECT.parent / relative)
        candidates.append(Path(sys._MEIPASS) / relative)
    return next((path for path in candidates if path.exists()), candidates[0])
