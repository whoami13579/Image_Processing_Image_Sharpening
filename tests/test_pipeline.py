"""Input, parameter, CLI, and manifest integration."""

import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from image_sharpening.algorithms import METHODS
from image_sharpening.config import Parameters, load_config
from image_sharpening.pipeline import IMAGE_NAMES, IMAGE_SIZE, file_hash, metrics, read_input, validate_inputs
from image_sharpening.resources import CONFIG_FILE, PROJECT, resource_path


def make_input(path, mode="L", size=IMAGE_SIZE):
    img = Image.new(mode, size, 57)
    info = PngImagePlugin.PngInfo()
    info.add_text("Fixture", "Original bytes must retain this metadata.")
    img.save(path, pnginfo=info)
    return path


@pytest.mark.parametrize("kwargs", [
    {"alpha": -1}, {"alpha": float("nan")}, {"alpha": True}, {"neighbors": 6},
    {"size": 4}, {"size": 3.0}, {"sigma": 0}, {"high_boost": 1},
    {"gamma_l": 0}, {"gamma_l": 1}, {"gamma_h": 1}, {"cutoff": 0},
    {"c": -1}, {"frequency_cutoff": 0}, {"frequency_gain": 0},
    {"preserve_dc": 1}, {"alpha": 10**400},
])
def test_invalid_parameter_types_and_ranges(kwargs):
    with pytest.raises(ValueError):
        replace(Parameters(), **kwargs)


@pytest.mark.parametrize("config", [
    '{"defaults":{"sigma":0}}', '{"extra":42}', '{"images":{"unknown.png":{}}}',
    '{"defaults":{"alpha":0.1,"alpha":0.2}}', '{"images":[]}', '{"images":{"2-1.png":3}}', "",
])
def test_bad_json_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(config, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid parameter file"):
        load_config(path, IMAGE_NAMES)


def test_selected_parameter_overrides():
    settings = load_config(resource_path(CONFIG_FILE), IMAGE_NAMES)
    assert settings["2-1.png"].size == 5
    assert settings["2-2.png"].high_boost == 1.3
    assert settings["2-3.png"] == settings["2-5.png"]
    assert settings["2-3.png"].sigma == 0.8
    assert settings["2-6.png"].gamma_l == 0.92
    assert all(params.neighbors == 4 and params.preserve_dc for params in settings.values())


@pytest.mark.parametrize("mode,size", [("L", (420, 512)), ("RGB", IMAGE_SIZE), ("1", IMAGE_SIZE),
                                         ("I;16", IMAGE_SIZE), ("LA", IMAGE_SIZE)])
def test_input_format_validation(tmp_path, mode, size):
    path = make_input(tmp_path / "2-1.png", mode, size)
    with pytest.raises(ValueError, match="Invalid input"):
        read_input(path)


def test_input_validation_rejects_missing_extra_and_corrupt_files(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        validate_inputs(missing)
    for name in IMAGE_NAMES[:-1]:
        make_input(tmp_path / name)
    extra = make_input(tmp_path / "extra.png")
    with pytest.raises(ValueError, match="Missing: 2-6.png. Unexpected: extra.png"):
        validate_inputs(tmp_path)
    extra.unlink()
    last = tmp_path / IMAGE_NAMES[-1]
    last.write_bytes(b"This is not a PNG")
    with pytest.raises(ValueError, match="Invalid input"):
        validate_inputs(tmp_path)


def test_manual_metrics_for_two_level_example():
    stats = metrics(np.array([[0, 255]], dtype=np.uint8))
    assert stats == {"mean": 127.5, "std": 127.5, "mean_gradient": 127.5,
                     "laplacian_variance": 65025.0, "black_pct": 50.0,
                     "white_pct": 50.0, "saturated_pct": 100.0}


def test_module_cli_all_methods_from_external_cwd(tmp_path):
    inputs, outputs = tmp_path / "inputs", tmp_path / "results"
    inputs.mkdir()
    name = "synthetic-step.png"
    pixels = np.tile(np.r_[np.full(256, 45), np.full(256, 170)], (420, 1)).astype(np.uint8)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Purpose", "Synthetic step-edge test fixture")
    Image.fromarray(pixels).save(inputs / name, pnginfo=metadata)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    cmd = [sys.executable, "-m", "image_sharpening", "--cli", "--synthetic",
           "--input", str(inputs), "--output", str(outputs), "--config", str(config)]
    original = (inputs / name).read_bytes()
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "1 synthetic images with 5 methods" in result.stdout
    manifest = json.loads((outputs / "results.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "synthetic"
    assert manifest["methods"] == list(METHODS)
    assert manifest["images"][name]["padded_shape"] == [1024, 1024]
    assert (outputs / "original" / name).read_bytes() == (inputs / name).read_bytes() == original
    assert all(file_hash(outputs / path) == checksum for path, checksum in manifest["files"].items())
    assert len(manifest["files"]) == 8
    for method in METHODS:
        with Image.open(outputs / method / name) as image:
            pixels = np.asarray(image)
            assert (image.format, image.size, image.mode) == ("PNG", IMAGE_SIZE, "L")
            assert pixels.dtype == np.uint8 and pixels.min() >= 0 and pixels.max() <= 255


def test_cli_help_and_input_output_overlap(tmp_path):
    help_text = subprocess.run([sys.executable, "-m", "image_sharpening", "--help"], cwd=tmp_path,
                               capture_output=True, text=True, timeout=30, check=True).stdout
    assert all(flag in help_text for flag in ("--input", "--output", "--method", "--config", "--gui", "--cli"))
    result = subprocess.run([sys.executable, "-m", "image_sharpening", "--input", str(tmp_path),
                             "--output", str(tmp_path / "outputs")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "separate" in result.stderr


def test_no_prohibited_dependency_or_transform_calls():
    banned_calls = {"convolve", "fftconvolve", "Laplacian", "GaussianBlur", "filter2D", "dft", "idft"}
    banned_module = ("cv2", "scipy", "skimage", "numpy.fft", "PIL.ImageFilter", "PIL.ImageEnhance")
    for path in (PROJECT / "src/image_sharpening").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(banned_module) for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert not node.module.startswith(banned_module)
                assert not {f"{node.module}.{alias.name}" for alias in node.names} & set(banned_module)
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_calls
                assert not (node.attr == "fft" and isinstance(node.value, ast.Name)
                            and node.value.id in {"np", "numpy"})
