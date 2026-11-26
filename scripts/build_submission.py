"""Stage and validate a student-ID directory and ZIP from current local artifacts."""

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from pypdf import PdfReader
from PyInstaller.archive.readers import CArchiveReader

from image_sharpening.artifacts import (BUNDLE_INFO, EXE_NAME, PDF_NAME, executable_source_hash,
                                         pdf_subject, read_student, verify_results)
from image_sharpening.pipeline import IMAGE_NAMES, RESULTS_FILE, file_hash, sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT_FILES = (".gitattributes", ".gitignore", ".python-version", "pyproject.toml",
                     "uv.lock", "run_homework.py", "ImageSharpening.spec")


def verify_executable(root, exe, inputs, outputs, config, result):
    with exe.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError("Expected a Windows executable. Rebuild with ImageSharpening.spec.")
    archive = CArchiveReader(str(exe))

    def extract(path):
        key = path.replace("/", "\\")
        return archive.extract(key if key in archive.toc else path)

    embedded = json.loads(extract(BUNDLE_INFO))
    saved = file_hash(outputs / RESULTS_FILE)
    if embedded != {"source_sha256": executable_source_hash(root), "config_sha256": file_hash(config),
                    "results_sha256": saved}:
        raise ValueError("Executable was built from different source or results. Rebuild it.")
    hashes = {f"HW2_test_image/{name}": file_hash(inputs / name) for name in IMAGE_NAMES}
    hashes.update({f"outputs/{relative}": checksum for relative, checksum in result["files"].items()})
    hashes["outputs/" + RESULTS_FILE] = saved
    hashes["configs/selected.json"] = file_hash(config)
    if any(sha256(extract(path)) != expected for path, expected in hashes.items()):
        raise ValueError("Executable contains outdated bundled resources. Rebuild it.")


def file_map(root, inputs, outputs, student_file, result):
    files = {Path(name): root / name for name in ("README.md",)}
    files[Path(EXE_NAME)] = root / "dist" / EXE_NAME
    files[Path(PDF_NAME)] = root / "report" / PDF_NAME
    files[Path("source/report/student.json")] = student_file
    files.update({Path("source") / name: root / name for name in SOURCE_ROOT_FILES})
    for folder in ("src", "tests", "scripts", "report"):
        for path in (root / folder).rglob("*.py"):
            files[Path("source") / path.relative_to(root)] = path
    for relative in ("report/report.md", "configs/selected.json"):
        files[Path("source") / relative] = root / relative
    for name in IMAGE_NAMES:
        files[Path("HW2_test_image") / name] = inputs / name
    for relative in (*result["files"], RESULTS_FILE):
        files[Path("outputs") / relative] = outputs / relative
    return files


def distribution_hashes(folder: Path) -> dict:
    return {path.relative_to(folder).as_posix(): file_hash(path)
            for path in folder.rglob("*") if path.is_file()}


def verify_zip(zip_path: Path, student_id: str, expected: dict) -> None:
    with ZipFile(zip_path) as archive:
        names = archive.namelist()
        if archive.testzip() is not None or sorted(names) != sorted(f"{student_id}/{name}" for name in expected):
            raise ValueError("ZIP has a CRC error or does not have the expected student-ID layout.")
        for name, checksum in expected.items():
            if sha256(archive.read(f"{student_id}/{name}")) != checksum:
                raise ValueError(f"ZIP content differs from the staged file: {name}")


def stage_submission(destination: Path, student_id: str, files: dict) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    folder, archive = destination / student_id, destination / f"{student_id}.zip"
    with TemporaryDirectory(dir=destination, prefix=".submission-") as directory:
        temp = Path(directory)
        stage = temp / student_id
        stage.mkdir()
        for relative, source in files.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        hashes = distribution_hashes(stage)
        staged_zip = temp / archive.name
        with ZipFile(staged_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as target:
            for relative in sorted(hashes):
                target.write(stage / relative, f"{student_id}/{relative}")
        verify_zip(staged_zip, student_id, hashes)
        if folder.is_symlink() or folder.is_junction() or archive.is_symlink():
            raise ValueError(f"Choose a submission destination without symbolic links: {destination}")
        if folder.exists() and distribution_hashes(folder) != hashes:
            raise ValueError(f"Existing submission differs: {folder}. Choose a fresh --destination.")
        if archive.exists():
            verify_zip(archive, student_id, hashes)
        if not folder.exists():
            stage.rename(folder)
        if not archive.exists():
            os.replace(staged_zip, archive)
    verify_zip(archive, student_id, distribution_hashes(folder))
    return archive


def build(root: Path, destination: Path, student_file: Path, student_id: str | None = None) -> Path:
    student = read_student(student_file)
    if student_id is not None and student_id != student["student_id"]:
        raise ValueError("--student-id must match the cover metadata. Update that file and rebuild the PDF.")
    inputs, outputs, config = (root / name for name in ("HW2_test_image", "outputs", "configs/selected.json"))
    result = verify_results(root, inputs, outputs, config)
    pdf = root / "report" / PDF_NAME
    reader = PdfReader(pdf)
    cover = reader.pages[0].extract_text()
    if (reader.metadata.subject != pdf_subject(root, outputs, student_file)
            or student["student_id"] not in cover or student["name"] not in cover):
        raise ValueError("The PDF does not match the current report, results, and cover details. Rebuild it.")
    verify_executable(root, root / "dist" / EXE_NAME, inputs, outputs, config, result)
    return stage_submission(destination, student["student_id"],
                            file_map(root, inputs, outputs, student_file, result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id")
    parser.add_argument("--student-file", type=Path, default=ROOT / "report/student.json")
    parser.add_argument("--destination", type=Path, default=ROOT / "submission")
    args = parser.parse_args()
    try:
        print(f"Verified submission ZIP: {build(ROOT, args.destination, args.student_file, args.student_id)}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Submission unavailable: {error}\n")


if __name__ == "__main__":
    main()
