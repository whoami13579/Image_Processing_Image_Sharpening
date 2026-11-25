"""Module CLI and launch-mode selection shared with the desktop entry point."""

import argparse
import json
from pathlib import Path
import sys

from .algorithms import METHODS
from .pipeline import run
from .resources import CONFIG_FILE, FROZEN, INPUT_FOLDER, PROJECT, resource_path


def main(argv=None, *, desktop_default=False) -> int:
    parser = argparse.ArgumentParser(description="Sharpen six grayscale PNGs using five manual methods.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="open the viewer")
    mode.add_argument("--cli", action="store_true", help="process and save results")
    parser.add_argument("--input", type=Path, help=f"input directory (default: {INPUT_FOLDER})")
    parser.add_argument("--output", type=Path, help="output directory (default: outputs)")
    parser.add_argument("--method", action="append", choices=METHODS,
                        help="run a specific method; repeat to select multiple (default: all five)")
    parser.add_argument("--config", type=Path, help="JSON settings (default: configs/selected.json)")
    parser.add_argument("--synthetic", action="store_true",
                        help="process only labeled synthetic-*.png fixtures")
    parser.add_argument("--summary", type=Path,
                        help="write a concise JSON runtime summary to a new file; useful with the windowed EXE")
    args = parser.parse_args(argv)
    arguments = sys.argv[1:] if argv is None else argv
    gui = args.gui or (desktop_default and not arguments)
    if gui and (args.method or args.synthetic or args.summary):
        parser.error("--method, --synthetic and --summary apply to CLI mode.")
    inputs = args.input or resource_path(INPUT_FOLDER)
    config = args.config or resource_path(CONFIG_FILE)
    if gui:
        from .gui import launch
        return launch(inputs, args.output or resource_path("outputs"), config)
    try:
        if args.summary and args.summary.exists():
            raise ValueError(f"Summary destination already exists: {args.summary}")
        outputs = args.output or (PROJECT / "outputs" if FROZEN else Path("outputs"))
        result = run(inputs, outputs, config, methods=args.method or METHODS, synthetic=args.synthetic)
        summary = {"dataset": result["dataset"], "image_count": len(result["images"]),
                   "methods": result["methods"], "processing_seconds": result["processing_seconds"],
                   "output": str(outputs.resolve())}
        if args.summary:
            with args.summary.open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2)
        print(f"Processed {summary['image_count']} {summary['dataset']} images with "
              f"{len(summary['methods'])} methods in {summary['processing_seconds']:.2f} s. "
              f"Results: {outputs}")
        return 0
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
