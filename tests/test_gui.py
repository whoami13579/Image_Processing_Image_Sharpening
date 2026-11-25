"""Exercise real Tk widgets and read-only background preview regeneration."""

from dataclasses import asdict
import json
from pathlib import Path
import time
import tkinter as tk

import numpy as np
from PIL import Image
import pytest

from image_sharpening.algorithms import LABELS
from image_sharpening.config import Parameters
from image_sharpening.gui import SharpeningViewer
from image_sharpening.pipeline import IMAGE_NAMES, IMAGE_SIZE, file_hash


@pytest.fixture(scope="module")
def desktop():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk desktop unavailable: {error}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def viewer(desktop, tmp_path):
    inputs, output = (tmp_path / "inputs"), (tmp_path / "outputs")
    config = tmp_path / "settings.json"
    config.write_text("{}", encoding="utf-8")
    inputs.mkdir()
    for key in LABELS:
        (output / key).mkdir(parents=True, exist_ok=True)
    records = {}
    for number, name in enumerate(IMAGE_NAMES):
        pixels = np.full(IMAGE_SIZE[::-1], 60 + 17 * number, dtype=np.uint8)
        pixels[:, 256:] += 20
        Image.fromarray(pixels).save(inputs / name)
        for index, key in enumerate(LABELS):
            Image.fromarray(pixels + index).save(output / key / name)
        records[name] = {"parameters": asdict(Parameters()), "metrics": {key: {} for key in LABELS},
                         "source_sha256": file_hash(inputs / name)}
    (output / "results.json").write_text(json.dumps({"images": records}), encoding="utf-8")
    root = tk.Toplevel(desktop)
    app = SharpeningViewer(root, inputs, output, config)
    root.update()
    yield app
    app.close()


def test_dropdown_wraparound_keyboard_handler_and_resizing(viewer):
    for index, name in enumerate(IMAGE_NAMES):
        viewer.selector.current(index)
        viewer.selector.event_generate("<<ComboboxSelected>>")
        viewer.root.update()
        assert viewer.name == name
        assert tuple(viewer.panes) == tuple(LABELS)
        assert all(pane.source.size == IMAGE_SIZE for pane in viewer.panes.values())
    assert viewer.keyboard_move(1) == "break"
    assert viewer.name == IMAGE_NAMES[0]
    viewer.move(-1)
    assert viewer.name == IMAGE_NAMES[-1]
    viewer.root.geometry("850x680")
    viewer.root.update()
    for pane in viewer.panes.values():
        width, height = pane.photo.width(), pane.photo.height()
        assert width > 100 and abs(width / height - 512 / 420) < 0.01
    assert "Homomorphic" in viewer.parameters.get()


def test_regenerate_while_navigating_preserves_files(viewer):
    parent = viewer.config.parent
    def snapshot():
        return {path: file_hash(path) for path in parent.rglob("*") if path.is_file()}

    before = snapshot()
    original_name = viewer.name
    viewer.regenerate_button.invoke()
    assert "Computing preview" in viewer.status.get()
    viewer.move(1)
    deadline = time.monotonic() + 20
    while viewer.poll_id is not None and time.monotonic() < deadline:
        viewer.root.update()
        time.sleep(0.02)
    assert viewer.name == IMAGE_NAMES[1]
    assert original_name in viewer.preview
    viewer.move(-1)
    assert "In-memory preview" in viewer.status.get()
    assert before == snapshot()


def test_native_inspection_window(viewer):
    viewer.inspect_tile("original")
    viewer.root.update()
    windows = [child for child in viewer.root.winfo_children() if isinstance(child, tk.Toplevel)]
    assert len(windows) == 1
    assert windows[0].title().endswith("Original")
    windows[0].destroy()
