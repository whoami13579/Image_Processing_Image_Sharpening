# Image Sharpening

Image Processing Homework #2. A Python 3.12 project for sharpening six 512 × 420 grayscale PNGs. The program implements five methods, writes a reproducible experiment manifest, and opens a Tkinter viewer with all six panels.

## Run on Windows 11

Install [uv](https://docs.astral.sh/uv/) and run these commands from the repository in PowerShell:

```powershell
uv sync --locked --python 3.12
uv run --locked python -m pytest -q
uv run --locked python -m image_sharpening --input HW2_test_image --output outputs
uv run --locked python run_homework.py
```

Copy the instructor's six original files, `2-1.png` to `2-6.png`, into `HW2_test_image/`. The CLI validates file names, PNG readability, dimensions, and native 8-bit grayscale samples before it processes anything. The originals are copied byte for byte to `outputs/original/`. The public repository contains the code, experiment settings, and report source. The instructor PNGs, images derived from them, local cover metadata, and binary deliverables are kept locally.

The CLI runs all five methods by default. Use `--method laplacian`, `--method unsharp`, `--method high_boost`, `--method homomorphic`, or `--method combined` to select a subset. Repeat the flag to select more than one. Use a separate output directory for subset experiments. `--config` selects a JSON file with optional `defaults` and `images` overrides keyed by filename. See `configs/selected.json` for the settings used in the homework. For an explicitly labeled dry run, supply `--synthetic`, readable 512 × 420 `synthetic-*.png` fixtures, and a config containing `{}`. A synthetic manifest cannot be used to build a homework report or submission.

### Desktop viewer

Open the source viewer with `uv run --locked python run_homework.py`. With a runnable Windows build, double-click `dist/ImageSharpening.exe`. The main window shows the original and five results in a 2 × 3 grid, followed by the log-image spectrum and the three filter-response curves side by side. The dropdown, Previous/Next buttons, and Left/Right keys switch the images, plots, and parameters together.

The images and plots resize with the window and preserve their aspect ratios. Use the vertical scrollbar or mouse wheel when the window is shorter. Double-click an image or the plot pair for native-resolution inspection.

**Regenerate preview** computes all five methods and renders the corresponding plots in a background thread. The viewer updates the images, parameters, and plots together. You can navigate while it runs and return to the processed image to see its preview. Previews leave the saved files untouched. If a saved spectrum is missing or unreadable, the viewer keeps the saved images visible and prompts you to select **Regenerate preview**.

The module defaults to CLI mode. The desktop launcher defaults to GUI mode when given no arguments. Both accept `--gui`, `--cli`, `--input`, `--output`, `--config`, and `--help`. The GUI searches for data beside the executable, in the project when run from `dist/`, and in the executable bundle. CLI output defaults to `outputs` in the working directory for source mode and beside the executable for frozen mode. Pass absolute input and output paths for explicit resource selection.

## Methods

| Method | Definition |
| --- | --- |
| Laplacian | Apply the four-neighbor second-difference mask and compute `f - αL`. |
| Unsharp masking | Generate normalized separable Gaussian weights, compute the detail mask `M = f - G*f`, and return `f + M`. |
| High-boost | Use the same Gaussian detail mask and return `f + kM` with `k > 1`. |
| Homomorphic | Transform `log(1+f)` and multiply by `γL + (γH-γL)(1-exp(-cD²/D0²))`. Restore the DC gain to 1 when configured. |
| Combined | Multiply the homomorphic response by `1 + kb(1-exp(-D²/(2Db²)))` on the same log-image spectrum, then apply one inverse FFT. |

The spatial methods use symmetric boundary indexing. The frequency methods symmetrically extend the 420 rows to 512, mirror the resulting tile on both axes to form a 1024 × 1024 periodic array, and crop after inversion. Cutoffs use cycles per pixel. The project uses explicit butterfly stages, convolution weights, frequency coordinates, and output rounding. NumPy provides array operations and elementary mathematics.

The batch writes 30 8-bit enhanced images, six native-resolution comparison grids, and six spectrum and filter-response plots. The JSON records the resolved settings, output hashes, brightness, standard deviation, mean gradient, Laplacian variance, separate black and white saturation percentages, and stage runtimes. Generate the per-image Conservative, Moderate, Strong, and Selected experiment sheets with:

```powershell
uv run --locked python scripts/compare_parameters.py
```

## Build the report, executable, and ZIP

Create `report/student.json` locally with `name` and `student_id` strings for the report cover. Build after generating the full outputs:

```powershell
uv run --locked python report/build_report.py
uv run --locked python -m PyInstaller --noconfirm --clean ImageSharpening.spec
uv run --locked python scripts/build_submission.py
```

`report/Image_Sharpening_Report.pdf` is generated from the current Markdown, verified figures, and recorded parameters. The report builder embeds the Windows Microsoft JhengHei TrueType fonts by default. Use `--font` and `--bold-font` if these fonts are installed elsewhere. The report follows the section order in Chapter 0, page 11. The cover lists the homework title, name, and student ID.

The spec creates a one-file Windows GUI executable `dist/ImageSharpening.exe`. When the official files and results are present, it bundles the PNGs and selected settings. To test CLI mode from PowerShell on a Windows machine permitted to run the executable:

```powershell
$exe = (Resolve-Path .\dist\ImageSharpening.exe).Path
$cliOutput = Join-Path $env:TEMP 'HW2-frozen-check'
$summary = Join-Path $env:TEMP 'HW2-frozen-summary.json'
$argList = @('--cli', '--output', ('"' + $cliOutput + '"'), '--summary', ('"' + $summary + '"'))
$p = Start-Process -FilePath $exe -WorkingDirectory $env:TEMP -ArgumentList $argList -Wait -PassThru
$p.ExitCode
Get-Content -LiteralPath $summary
```

The supplied executable was built as a Windows x86-64 GUI binary. Its embedded images and settings were verified against the saved files. A Windows launch in `--cli` mode from a separate working directory reproduced all 48 saved PNG hashes and the recorded image metrics. The source viewer was exercised. Standalone desktop launch still needs a separate check.

The packaging script uses the local cover metadata, checks that the report and executable correspond to the current results, and verifies every archive member by SHA-256. It creates `submission/<student-id>/` and `submission/<student-id>.zip`. Inside the student-ID directory, `Image_Sharpening_Report.pdf` sits beside `ImageSharpening.exe` and `README.md`. The directory also contains `source/`, `HW2_test_image/`, and `outputs/`. The report Markdown and generator are included under `source/report/`.

The script preserves an existing package with different contents. To build a replacement for review, pass a fresh `--destination`. To regenerate the PDF from an extracted package, run this command from `source/`:

```powershell
uv run --locked python report/build_report.py --input ../HW2_test_image --outputs ../outputs --output ../Image_Sharpening_Report.pdf
```

The project follows the src layout, desktop navigation, uv, ReportLab, PyInstaller, and validated submission workflow from the previous assignment. Section 2.1 of the report documents the AI assistance used to implement and evaluate the sharpening methods. [Homework #1 at 6a15a6b](https://github.com/whoami13579/Image_Processing_Spatial_Image_Enhancement/tree/6a15a6b4aea3d27ae0190c685aaf9743388dd617)
