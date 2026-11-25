"""Archive layout, bytes, mismatch protection, and optional local artifacts."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from image_sharpening.artifacts import EXE_NAME, PDF_NAME, verify_results
from image_sharpening.pipeline import IMAGE_NAMES, file_hash
from image_sharpening.resources import PROJECT
from scripts.build_submission import (distribution_hashes, stage_submission, verify_executable,
                                      verify_zip)


def test_submission_archive_has_only_student_id_root_and_exact_files(tmp_path):
    fixtures = tmp_path / "files"
    fixtures.mkdir()
    files = {}
    members = [EXE_NAME, f"report/{PDF_NAME}", "source/uv.lock", "outputs/results.json"]
    members.extend(f"HW2_test_image/{name}" for name in IMAGE_NAMES)
    for index, member in enumerate(members):
        path = fixtures / f"fixture-{index}"
        path.write_bytes(bytes(range(index + 1)))
        files[Path(member)] = path
    destination = tmp_path / "submission"
    package = stage_submission(destination, "TEST123", files)
    assert package.name == "TEST123.zip"
    assert stage_submission(destination, "TEST123", files) == package
    original_sha256 = file_hash(package)
    with ZipFile(package) as archive:
        assert set(archive.namelist()) == {f"TEST123/{name}" for name in members}
    hashes = distribution_hashes(destination / "TEST123")
    verify_zip(package, "TEST123", hashes)
    files[Path(EXE_NAME)].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing submission differs"):
        stage_submission(destination, "TEST123", files)
    assert file_hash(package) == original_sha256


def test_real_local_outputs_and_executable_resources_when_available():
    outputs = PROJECT / "outputs"
    inputs = PROJECT / "HW2_test_image"
    config = PROJECT / "configs" / "selected.json"
    exe = PROJECT / "dist" / EXE_NAME
    if not ((outputs / "results.json").is_file() and exe.is_file() and inputs.is_dir()):
        pytest.skip("Local full homework artifacts are optional in a source checkout.")
    result = verify_results(PROJECT, inputs, outputs, config)
    verify_executable(PROJECT, exe, inputs, outputs, config, result)
    for image in IMAGE_NAMES:
        assert file_hash(outputs / "original" / image) == file_hash(inputs / image)
