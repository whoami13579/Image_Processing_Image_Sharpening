"""Validate source PNGs, run the experiment, and write its output manifest."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from PIL import Image

from .algorithms import LABELS, METHODS, laplacian, sharpen
from .config import load_config
from .convolution import symmetric_extend
from .figures import comparison_grid, save_spectrum

IMAGE_NAMES = tuple(f"2-{number}.png" for number in range(1, 7))
IMAGE_SIZE = (512, 420)
RESULTS_FILE = "results.json"


@dataclass(frozen=True)
class InputImage:
    name: str
    original_bytes: bytes
    pixels: np.ndarray


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_input(path: Path) -> InputImage:
    try:
        file_bytes = path.read_bytes()
        with Image.open(BytesIO(file_bytes)) as png:
            if png.format != "PNG" or png.size != IMAGE_SIZE or png.mode != "L" or "transparency" in png.info:
                raise ValueError("expected a 512 × 420 PNG in grayscale mode L")
            # PNG can decode low-bit-depth grayscale into mode L. Require
            # native 8-bit samples (IHDR bit depth 8 and color type 0).
            if len(file_bytes) < 29 or struct.unpack_from(">BB", file_bytes, 24) != (8, 0):
                raise ValueError("expected native 8-bit grayscale samples (PNG color type 0)")
            png.verify()
        with Image.open(BytesIO(file_bytes)) as png:
            pixels = np.array(png)
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid input {path}: {error}") from error
    return InputImage(path.name, file_bytes, pixels)


def validate_inputs(folder: Path, *, synthetic=False) -> list[InputImage]:
    if not folder.is_dir():
        raise ValueError(f"Input directory does not exist: {folder}")
    present = {path.name for path in folder.iterdir() if path.suffix.lower() == ".png"}
    expected = set(IMAGE_NAMES)
    if synthetic:
        if not present or any(not name.startswith("synthetic-") for name in present):
            raise ValueError("Synthetic inputs must be PNGs named synthetic-*.png.")
        expected = present
    if present != expected:
        missing = ", ".join(sorted(expected - present)) or "none"
        extra = ", ".join(sorted(present - expected)) or "none"
        raise ValueError(f"Expected exactly the six HW2 PNGs in {folder}. Missing: {missing}. Unexpected: {extra}.")
    return [read_input(folder / name) for name in sorted(expected)]


def metrics(image) -> dict[str, float]:
    source = image.astype(np.float64)
    height, width = source.shape
    ext = symmetric_extend(source, (1, 1), (1, 1))
    dy = (ext[2:, 1:width + 1] - ext[:height, 1:width + 1]) / 2
    dx = (ext[1:height + 1, 2:] - ext[1:height + 1, :width]) / 2
    black = 100.0 * np.count_nonzero(image == 0) / image.size
    white = 100.0 * np.count_nonzero(image == 255) / image.size
    return {"mean": float(source.mean()), "std": float(source.std()),
            "mean_gradient": float(np.sqrt(dx ** 2 + dy ** 2).mean()),
            "laplacian_variance": float(laplacian(source).var()),
            "black_pct": black, "white_pct": white, "saturated_pct": black + white}


def processing_source_hash(folder: Path) -> str | None:
    """Fingerprint the source involved in producing results, when available."""
    names = ("algorithms", "config", "convolution", "figures", "fourier", "frequency_filters", "pipeline")
    parts = [folder / f"{name}.py" for name in names]
    if all(part.is_file() for part in parts):
        return sha256(b"".join(bytes.fromhex(file_hash(part)) for part in parts))
    return None


@contextmanager
def output_lock(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".publish-lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise ValueError(f"Another process is publishing to {folder}.") from error
    try:
        yield
    finally:
        lock.rmdir()


def run(input_dir: Path, output_dir: Path, config: Path, *, methods=METHODS, synthetic=False) -> dict:
    start = perf_counter()
    input_dir, output_dir, config = map(Path, (input_dir, output_dir, config))
    incoming, outgoing = input_dir.resolve(), output_dir.resolve()
    if incoming.is_relative_to(outgoing) or outgoing.is_relative_to(incoming):
        raise ValueError("Choose an output directory separate from the input directory.")
    images = validate_inputs(input_dir, synthetic=synthetic)
    config_hash = file_hash(config)
    parameters = load_config(config, [image.name for image in images])
    methods = tuple(methods)
    records = {}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sharpening-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        for folder in ("original", *methods, "comparisons", "spectra"):
            (stage / folder).mkdir()
        for image in images:
            p = parameters[image.name]
            enhanced, timing, spectra = sharpen(image.pixels, p, methods)
            (stage / "original" / image.name).write_bytes(image.original_bytes)
            for method, pixels in enhanced.items():
                Image.fromarray(pixels).save(stage / method / image.name)
            stem = Path(image.name).stem
            absolute_pixels = {"original": image.pixels, **enhanced}
            comparison_grid([(LABELS[key], pixels) for key, pixels in absolute_pixels.items()],
                            columns=3).save(stage / "comparisons" / f"{stem}_comparison.png")
            if spectra:
                save_spectrum(stage / "spectra" / f"{stem}_spectrum.png", image.name, spectra)
            records[image.name] = {
                "source_sha256": sha256(image.original_bytes),
                "parameters": asdict(p),
                "metrics": {key: metrics(pixels) for key, pixels in absolute_pixels.items()},
                "runtime_ms": timing,
                "padded_shape": list(spectra["spectrum"].shape) if spectra else None,
            }
        checksums = {path.relative_to(stage).as_posix(): file_hash(path)
                     for path in sorted(stage.rglob("*.png"))}
        if file_hash(config) != config_hash:
            raise ValueError(f"Parameter file changed during processing: {config}")
        result = {"schema": 1, "dataset": "synthetic" if synthetic else "homework",
                  "image_size": list(IMAGE_SIZE), "methods": list(methods), "images": records,
                  "config_sha256": config_hash, "files": checksums,
                  "source_sha256": processing_source_hash(Path(__file__).parent),
                  "processing_seconds": perf_counter() - start}
        manifest = stage / RESULTS_FILE
        manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        # Validate every destination before publishing. A pre-existing symlink
        # or junction must never redirect generated files into another tree.
        paths = [Path(name) for name in checksums] + [Path(RESULTS_FILE)]
        with output_lock(outgoing):
            for relative in paths:
                target = outgoing / relative
                if target.resolve() != target:
                    raise ValueError(f"Output path must not be a symlink or junction: {target}")
                if target.exists() and not target.is_file():
                    raise ValueError(f"Expected a file destination: {target}")
            for relative in paths:
                target = outgoing / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / relative, target)
    return result
