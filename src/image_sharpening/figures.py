"""Native-resolution comparison layouts and spectrum plots."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fourier import center_spectrum


def comparison_grid(panels, columns: int) -> Image.Image:
    """Paste each image at its exact pixel size with a readable header."""
    panels = list(panels)
    height, width = panels[0][1].shape
    label_height, gap = 36, 12
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (gap + columns * (width + gap),
                               gap + rows * (height + label_height + gap)), "#e8edf2")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=21)
    for index, (label, pixels) in enumerate(panels):
        row, col = divmod(index, columns)
        x = gap + col * (width + gap)
        y = gap + row * (height + label_height + gap)
        draw.rectangle((x, y, x + width - 1, y + label_height - 1), fill="#182c40")
        draw.text((x + 10, y + label_height // 2), label, fill="white", font=font, anchor="lm")
        canvas.paste(Image.fromarray(pixels), (x, y + label_height))
    return canvas


def save_spectrum(path: Path, name: str, diagnostics: dict) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    log_magnitude = np.log1p(np.abs(center_spectrum(diagnostics["spectrum"])))
    fig = Figure(figsize=(10.6, 4.3), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    fig.suptitle(f"{name}   |   Log-image spectrum and filter gains", fontweight="bold")
    # Extent is expressed at bin edges, so centered sample locations are exact.
    h, w = log_magnitude.shape
    image = axes[0].imshow(log_magnitude, cmap="magma", origin="lower",
                           extent=(-.5 - .5/w, .5 - .5/w, -.5 - .5/h, .5 - .5/h))
    axes[0].set(xlabel="Horizontal frequency (cycles/pixel)",
                ylabel="Vertical frequency (cycles/pixel)", title="log(1 + |Z|)")
    fig.colorbar(image, ax=axes[0], fraction=.047, pad=.035)
    response_ax = axes[1]
    styles = (("homomorphic", "#21659b", "Homomorphic"),
              ("high_boost", "#8b531d", "Gaussian high-boost"),
              ("combined", "#af3d58", "Combined"))
    for key, color, label in styles:
        if key in diagnostics:
            response = diagnostics[key][0]
            positive_axis = np.arange(response.size // 2 + 1)
            response_ax.plot(positive_axis / response.size, response[positive_axis], color=color,
                             label=label, linewidth=1.6)
            response_ax.scatter(0, response[0], color=color, s=13)
    response_ax.axhline(1, linewidth=.8, color="#999999", linestyle=":")
    response_ax.set(xlabel="Horizontal frequency (cycles/pixel)", ylabel="Amplitude gain",
                    title="Response along vertical frequency = 0", xlim=(-.008, .5))
    response_ax.grid(alpha=.2)
    response_ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=150)
