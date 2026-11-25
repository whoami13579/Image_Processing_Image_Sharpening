"""Shared checks for the report, executable, and submission builders."""

from dataclasses import asdict
import json
from pathlib import Path
import re

from PIL import Image

from .algorithms import METHODS
from .config import load_config
from .pipeline import (IMAGE_NAMES, IMAGE_SIZE, RESULTS_FILE, file_hash, processing_source_hash,
                       sha256, validate_inputs)

PDF_NAME = "Image_Sharpening_Report.pdf"
EXE_NAME = "ImageSharpening.exe"
BUNDLE_INFO = "_build_manifest.json"


def verify_results(project: Path, inputs: Path, outputs: Path, config: Path) -> dict:
    manifest = json.loads((outputs / RESULTS_FILE).read_text(encoding="utf-8"))
    images = validate_inputs(inputs)
    parameters = load_config(config, IMAGE_NAMES)
    source = processing_source_hash(project / "src/image_sharpening")
    if (manifest.get("schema") != 1 or manifest.get("dataset") != "homework"
            or list(manifest.get("methods", [])) != list(METHODS)
            or manifest.get("image_size") != list(IMAGE_SIZE)
            or set(manifest.get("images", {})) != set(IMAGE_NAMES)
            or manifest.get("config_sha256") != file_hash(config)
            or not source or manifest.get("source_sha256") != source):
        raise ValueError("Current full homework results are required. Regenerate outputs with the selected JSON.")
    expected = set()
    for image in images:
        record = manifest["images"][image.name]
        if (record["source_sha256"] != sha256(image.original_bytes)
                or record["parameters"] != asdict(parameters[image.name])):
            raise ValueError(f"Input or parameter mismatch in results for {image.name}.")
        for kind in ("original", *METHODS):
            relative = f"{kind}/{image.name}"
            expected.add(relative)
            with Image.open(outputs / relative) as png:
                if (png.format, png.size, png.mode) != ("PNG", IMAGE_SIZE, "L"):
                    raise ValueError(f"Invalid saved image: {relative}")
                png.verify()
        if file_hash(outputs / "original" / image.name) != record["source_sha256"]:
            raise ValueError(f"Saved original differs for {image.name}.")
        stem = Path(image.name).stem
        expected.update({f"comparisons/{stem}_comparison.png", f"spectra/{stem}_spectrum.png"})
    if set(manifest.get("files", {})) != expected:
        raise ValueError("Result manifest must list all expected images and figures.")
    for relative in sorted(expected):
        if file_hash(outputs / relative) != manifest["files"][relative]:
            raise ValueError(f"Saved result differs from its manifest: {relative}")
    return manifest


def file_set_hash(paths) -> str:
    return sha256(b"".join(bytes.fromhex(file_hash(path)) for path in paths))


def pdf_subject(project: Path, outputs: Path, student_file: Path) -> str:
    paths = (project / "report/report.md", project / "report/build_report.py",
             outputs / RESULTS_FILE, student_file)
    return "HW2 " + file_set_hash(paths)


def executable_source_hash(project: Path) -> str:
    paths = [project / name for name in ("run_homework.py", "ImageSharpening.spec",
                                        "pyproject.toml", "uv.lock")]
    paths.extend(sorted((project / "src/image_sharpening").glob("*.py")))
    return file_set_hash(paths)


def read_student(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or not isinstance(data.get("name"), str) or not data["name"].strip()
            or not isinstance(data.get("student_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", data["student_id"])
            or not isinstance(data.get("submission_date", ""), str)):
        raise ValueError("Student metadata requires a name, a valid student_id, and an optional submission_date.")
    if data["student_id"].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)),
                                      *(f"LPT{n}" for n in range(1, 10))}:
        raise ValueError("Student ID cannot be a reserved Windows filename.")
    return data
