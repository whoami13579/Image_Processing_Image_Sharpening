# -*- mode: python ; coding: utf-8 -*-

import json
from pathlib import Path
import sys

from PyInstaller.config import CONF

root = Path(SPECPATH)
sys.path.insert(0, str(root / "src"))

from image_sharpening.artifacts import BUNDLE_INFO, executable_source_hash, verify_results
from image_sharpening.pipeline import IMAGE_NAMES, RESULTS_FILE, file_hash, validate_inputs

inputs = root / "HW2_test_image"
outputs = root / "outputs"
config = root / "configs/selected.json"
datas = [(str(config), "configs")]
if inputs.exists():
    validate_inputs(inputs)
    datas.extend((str(inputs / name), "HW2_test_image") for name in IMAGE_NAMES)

info = {"source_sha256": executable_source_hash(root), "config_sha256": file_hash(config)}
if (outputs / RESULTS_FILE).is_file():
    results = verify_results(root, inputs, outputs, config)
    datas.append((str(outputs / RESULTS_FILE), "outputs"))
    for relative in results["files"]:
        datas.append((str(outputs / relative), (Path("outputs") / relative).parent.as_posix()))
    info["results_sha256"] = file_hash(outputs / RESULTS_FILE)

work = Path(CONF["workpath"])
work.mkdir(parents=True, exist_ok=True)
manifest = work / BUNDLE_INFO
manifest.write_text(json.dumps(info), encoding="utf-8")
datas.append((str(manifest), "."))

analysis = Analysis(
    [str(root / "run_homework.py")],
    pathex=[str(root / "src")],
    datas=datas,
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    excludes=["numpy.fft", "scipy", "cv2", "skimage"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, analysis.binaries, analysis.datas, [],
    name="ImageSharpening",
    console=False,
    upx=False,
)
