"""Exercise real Tk widgets and read-only background preview regeneration."""

from dataclasses import asdict
import json
from threading import Event
import time
import tkinter as tk

import numpy as np
from PIL import Image
import pytest

from image_sharpening.algorithms import LABELS
from image_sharpening.config import Parameters
from image_sharpening.figures import save_spectrum
from image_sharpening import gui
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
    (output / "spectra").mkdir(parents=True)
    for key in LABELS:
        (output / key).mkdir(parents=True, exist_ok=True)
    records = {}
    hashes = {}
    for number, name in enumerate(IMAGE_NAMES):
        pixels = np.full(IMAGE_SIZE[::-1], 60 + 17 * number, dtype=np.uint8)
        pixels[:, 256:] += 20
        Image.fromarray(pixels).save(inputs / name)
        for index, key in enumerate(LABELS):
            Image.fromarray(pixels + index).save(output / key / name)
        relative = f"spectra/{name[:-4]}_spectrum.png"
        Image.new("RGB", (1590, 645), (number, 20, 30)).save(output / relative)
        hashes[relative] = file_hash(output / relative)
        records[name] = {"parameters": asdict(Parameters()), "metrics": {key: {} for key in LABELS},
                         "padded_shape": [1024, 1024],
                         "source_sha256": file_hash(inputs / name)}
    (output / "results.json").write_text(json.dumps({"images": records, "files": hashes}), encoding="utf-8")
    root = tk.Toplevel(desktop)
    app = SharpeningViewer(root, inputs, output, config)
    root.update()
    yield app
    app.close()


def assert_saved_view(viewer, index):
    name = IMAGE_NAMES[index]
    assert viewer.name == name
    assert tuple(viewer.panes) == tuple(LABELS)
    for key, pane in viewer.panes.items():
        with Image.open(viewer.output / key / name) as expected:
            assert pane.source.size == IMAGE_SIZE
            assert pane.source.tobytes() == expected.tobytes()
    assert viewer.spectrum.source.getpixel((0, 0)) == (index, 20, 30)


def finish_preview(viewer):
    deadline = time.monotonic() + 20
    while viewer.poll_id is not None and time.monotonic() < deadline:
        viewer.root.update()
        time.sleep(0.02)
    assert viewer.poll_id is None


def test_spectrum_is_inline_at_startup(viewer):
    assert_saved_view(viewer, 0)
    assert viewer.spectrum.winfo_ismapped()
    assert viewer.spectrum.master is viewer.image_grid.master
    assert viewer.spectrum.grid_info()["row"] > viewer.image_grid.grid_info()["row"]
    assert viewer.spectrum.photo.width() > 650
    assert not any(isinstance(child, tk.Toplevel) for child in viewer.root.winfo_children())


def test_dropdown_navigates_through_all_six_saved_views(viewer):
    for index, name in enumerate(IMAGE_NAMES):
        viewer.selector.current(index)
        viewer.selector.event_generate("<<ComboboxSelected>>")
        viewer.root.update()
        assert viewer.selector.get() == name
        assert_saved_view(viewer, index)


def test_previous_next_and_arrow_keys_wrap_with_plot_updates(viewer):
    viewer.previous_button.invoke()
    assert_saved_view(viewer, 5)
    viewer.next_button.invoke()
    assert_saved_view(viewer, 0)
    viewer.root.focus_force()
    viewer.root.update()
    viewer.root.event_generate("<Right>")
    viewer.root.update()
    assert_saved_view(viewer, 1)
    viewer.root.event_generate("<Left>")
    viewer.root.update()
    assert_saved_view(viewer, 0)


def test_resizing_and_vertical_scroll_keep_aspect_ratios(viewer):
    viewer.root.geometry("820x510")
    viewer.root.update()
    for pane in viewer.panes.values():
        width, height = pane.photo.width(), pane.photo.height()
        assert width > 100 and abs(width / height - 512 / 420) < 0.01
    width, height = viewer.spectrum.photo.width(), viewer.spectrum.photo.height()
    assert width > 720 and abs(width / height - 1590 / 645) < 0.02
    assert viewer.viewport.xview() == (0.0, 1.0)
    assert viewer.viewport.yview()[1] < 1
    viewer.root.event_generate("<MouseWheel>", delta=-120)
    viewer.root.update()
    assert viewer.viewport.yview()[0] > 0
    viewer.viewport.yview_moveto(1)
    viewer.root.update()
    assert viewer.viewport.yview()[1] == 1
    assert viewer.spectrum.winfo_rooty() >= viewer.viewport.winfo_rooty()
    assert "Homomorphic" in viewer.parameters.get()


def test_regenerate_synchronizes_images_and_spectrum_while_navigating(viewer, monkeypatch):
    parent = viewer.config.parent
    viewer.config.write_text('{"defaults":{"alpha":0.27,"frequency_gain":0.35}}', encoding="utf-8")
    (parent / "report.pdf").write_bytes(b"Read-only report fixture")

    def snapshot():
        return {path: file_hash(path) for path in parent.rglob("*") if path.is_file()}

    started, release = Event(), Event()
    run = {}
    real_sharpen = gui.sharpen

    def controlled_sharpen(pixels, parameters, methods):
        started.set()
        assert release.wait(10)
        result = real_sharpen(pixels, parameters, methods)
        run.update(parameters=parameters, result=result)
        return result

    monkeypatch.setattr(gui, "sharpen", controlled_sharpen)
    before = snapshot()
    original_name = viewer.name
    viewer.regenerate_button.invoke()
    try:
        assert started.wait(5)
        assert "Computing preview" in viewer.status.get()
        assert_saved_view(viewer, 0)
        viewer.next_button.invoke()
        viewer.root.update()
        assert_saved_view(viewer, 1)
        other_plot = viewer.spectrum.source
    finally:
        release.set()
    finish_preview(viewer)
    assert_saved_view(viewer, 1)
    assert viewer.spectrum.source is other_plot

    viewer.previous_button.invoke()
    assert "In-memory preview" in viewer.status.get()
    assert run["parameters"].frequency_gain == .35
    assert "0.35" in viewer.parameters.get()
    enhanced, _, diagnostics = run["result"]
    for key, pixels in enhanced.items():
        np.testing.assert_array_equal(np.asarray(viewer.panes[key].source), pixels)
    assert before == snapshot()
    reference = parent / "reference.png"
    save_spectrum(reference, original_name, diagnostics)
    with Image.open(reference) as expected:
        assert viewer.spectrum.source.tobytes() == expected.tobytes()
    assert viewer.spectrum.source.getpixel((0, 0)) != (0, 20, 30)


@pytest.mark.parametrize("problem", ["missing", "unreadable", "changed"])
def test_unavailable_spectrum_clears_the_previous_plot_only(viewer, problem):
    relative = "spectra/2-2_spectrum.png"
    path = viewer.output / relative
    if problem == "missing":
        path.unlink()
    elif problem == "unreadable":
        path.write_bytes(b"Corrupted PNG")
        viewer.saved_hashes[relative] = file_hash(path)
    else:
        Image.new("RGB", (1590, 645), "blue").save(path)
    viewer.next_button.invoke()
    viewer.root.update()
    assert all(pane.source is not None for pane in viewer.panes.values())
    assert viewer.spectrum.source is None
    assert viewer.spectrum.itemcget(viewer.spectrum.placeholder, "state") == "normal"
    assert "Spectrum unavailable" in viewer.status.get()
    viewer.previous_button.invoke()
    assert_saved_view(viewer, 0)


def test_failed_plot_generation_keeps_last_complete_view(viewer, monkeypatch):
    def fail(*args):
        raise OSError("unable to render preview plot")

    monkeypatch.setattr(gui, "sharpen", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(gui, "save_spectrum", fail)
    old_plot = viewer.spectrum.source
    viewer.regenerate_button.invoke()
    finish_preview(viewer)
    assert_saved_view(viewer, 0)
    assert viewer.spectrum.source is old_plot
    assert "Preview failed: unable to render preview plot" in viewer.status.get()
    viewer.next_button.invoke()
    assert_saved_view(viewer, 1)
    viewer.previous_button.invoke()
    assert "Preview failed: unable to render preview plot" in viewer.status.get()


def test_native_inspection_windows_for_image_and_spectrum(viewer):
    viewer.inspect_tile("original")
    viewer.inspect_spectrum()
    viewer.root.update()
    windows = [child for child in viewer.root.winfo_children() if isinstance(child, tk.Toplevel)]
    assert {window.title().split(" · ")[-1] for window in windows} == {"Original", "Spectrum and filter gains"}
    for window, size in zip(windows, (IMAGE_SIZE, (1590, 645)), strict=True):
        canvas = next(child for child in window.winfo_children() if isinstance(child, tk.Canvas))
        assert [int(value) for value in canvas.cget("scrollregion").split()] == [0, 0, *size]
        window.destroy()
