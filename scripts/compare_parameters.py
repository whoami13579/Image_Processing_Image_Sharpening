"""Produce per-image native-resolution comparison sheets for the parameter review."""

import argparse
from dataclasses import replace
from pathlib import Path

from image_sharpening.algorithms import LABELS, METHODS, sharpen
from image_sharpening.config import load_config
from image_sharpening.figures import comparison_grid
from image_sharpening.pipeline import validate_inputs

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "Conservative": dict(alpha=.10, size=3, sigma=.70, high_boost=1.20, gamma_l=.94,
                         gamma_h=1.05, cutoff=.018, frequency_gain=.06, frequency_cutoff=.12),
    "Moderate": dict(alpha=.18, size=5, sigma=.90, high_boost=1.40, gamma_l=.88,
                     gamma_h=1.10, cutoff=.025, frequency_gain=.12, frequency_cutoff=.10),
    "Strong": dict(alpha=.32, size=7, sigma=1.40, high_boost=1.85, gamma_l=.78,
                   gamma_h=1.20, cutoff=.038, frequency_gain=.25, frequency_cutoff=.075),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "HW2_test_image")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/parameter_trials")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/selected.json")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.input.resolve()):
        parser.error("Keep trial output outside the input folder.")
    images = validate_inputs(args.input)
    params = load_config(args.config, [image.name for image in images])
    args.output.mkdir(parents=True, exist_ok=True)
    for image in images:
        choices = {label: sharpen(image.pixels, replace(params[image.name], **profile))[0]
                   for label, profile in PROFILES.items()}
        choices["Selected"] = sharpen(image.pixels, params[image.name])[0]
        panels = [(f"{LABELS[method]} · {label}", pixels[method])
                  for method in METHODS for label, pixels in choices.items()]
        comparison_grid(panels, len(choices)).save(args.output / f"{Path(image.name).stem}_trials.png")
    print(f"Wrote {len(images)} trial sheets to {args.output}")


if __name__ == "__main__":
    main()
