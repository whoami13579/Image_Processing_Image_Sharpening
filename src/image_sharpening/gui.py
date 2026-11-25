"""Responsive Tkinter viewer with read-only saved results and in-memory previews."""

from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
from queue import Empty, SimpleQueue
from tempfile import TemporaryDirectory
from threading import Thread
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .algorithms import LABELS, METHODS, sharpen
from .config import load_config
from .figures import save_spectrum
from .pipeline import IMAGE_NAMES, RESULTS_FILE, read_input, sha256


@dataclass(frozen=True)
class ViewerResult:
    images: dict[str, Image.Image]
    parameters: dict | None = None
    spectrum: Image.Image | None = None


class ImagePane(tk.Canvas):
    def __init__(self, parent, *, background="#101d2d", empty_text=""):
        super().__init__(parent, background=background, highlightthickness=0,
                         width=1, height=1, cursor="hand2")
        self.source = None
        self.photo = None
        self.item = self.create_image(0, 0)
        self.placeholder = self.create_text(0, 0, text=empty_text, fill="#243c53", justify="center")
        self.bind("<Configure>", self.draw)

    def set_image(self, source):
        self.source = source
        self.draw()

    def draw(self, _event=None):
        width, height = self.winfo_width(), self.winfo_height()
        self.coords(self.placeholder, width // 2, height // 2)
        self.itemconfigure(self.placeholder, width=max(1, width - 32),
                           state="normal" if self.source is None else "hidden")
        if self.source is None:
            self.itemconfigure(self.item, image="")
            self.photo = None
            return
        preview = self.source.copy()
        preview.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(preview, master=self)
        self.itemconfigure(self.item, image=self.photo)
        self.coords(self.item, width // 2, height // 2)


class SharpeningViewer:
    def __init__(self, root, inputs: Path, output: Path, config: Path):
        self.root, self.inputs, self.output, self.config = root, inputs, output, config
        self.index = 0
        self.preview = {}
        self.preview_errors = {}
        self.in_flight = SimpleQueue()
        self.poll_id = None
        self.auto_preview = False
        try:
            manifest = json.loads((output / RESULTS_FILE).read_text(encoding="utf-8"))
            self.manifest = manifest.get("images", {})
            self.saved_hashes = manifest.get("files", {})
        except (OSError, ValueError, AttributeError):
            self.manifest = {}
            self.saved_hashes = {}

        root.title("Image Sharpening")
        root.minsize(720, 500)
        width = max(720, min(1240, root.winfo_screenwidth() - 80))
        height = max(500, min(1020, root.winfo_screenheight() - 95))
        root.geometry(f"{width}x{height}")
        root.configure(background="#e8edf3")
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.configure("Tile.TLabel", background="#203751", foreground="white", padding=(10, 6))
        style.configure("Toolbar.TFrame", background="#e8edf3")
        style.configure("Status.TLabel", background="#e8edf3", foreground="#243c53", padding=(2, 3))

        body = ttk.Frame(root, style="Toolbar.TFrame", padding=(14, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        controls = ttk.Frame(body, style="Toolbar.TFrame")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(controls, text="Image", style="Status.TLabel").pack(side="left", padx=(0, 6))
        self.selector = ttk.Combobox(controls, values=IMAGE_NAMES, state="readonly", width=10)
        self.selector.pack(side="left")
        self.selector.bind("<<ComboboxSelected>>", lambda _event: self.show(self.selector.current()))
        self.previous_button = ttk.Button(controls, text="Previous", command=lambda: self.move(-1))
        self.previous_button.pack(side="left", padx=(8, 4))
        self.next_button = ttk.Button(controls, text="Next", command=lambda: self.move(1))
        self.next_button.pack(side="left")
        self.regenerate_button = ttk.Button(controls, text="Regenerate preview", command=self.regenerate)
        self.regenerate_button.pack(side="right", padx=6)
        root.bind("<Left>", lambda _event: self.keyboard_move(-1))
        root.bind("<Right>", lambda _event: self.keyboard_move(1))

        scroll_area = ttk.Frame(body, style="Toolbar.TFrame")
        scroll_area.grid(row=1, column=0, sticky="nsew")
        scroll_area.columnconfigure(0, weight=1)
        scroll_area.rowconfigure(0, weight=1)
        self.viewport = tk.Canvas(scroll_area, background="#e8edf3", highlightthickness=0,
                                  height=1, width=1, yscrollincrement=24)
        self.viewport.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(scroll_area, orient="vertical", command=self.viewport.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.viewport.configure(yscrollcommand=scrollbar.set)
        content = ttk.Frame(self.viewport, style="Toolbar.TFrame")
        content.columnconfigure(0, weight=1)
        self.content_item = self.viewport.create_window(0, 0, window=content, anchor="nw")
        content.bind("<Configure>", lambda event: self.viewport.configure(
            scrollregion=(0, 0, event.width, event.height)))
        self.viewport.bind("<Configure>", self.resize_content)
        root.bind("<MouseWheel>", self.scroll_content)
        root.bind("<Button-4>", self.scroll_content)
        root.bind("<Button-5>", self.scroll_content)

        self.image_grid = ttk.Frame(content, style="Toolbar.TFrame")
        self.image_grid.grid(row=0, column=0, sticky="ew")
        self.image_grid.grid_propagate(False)
        self.panes = {}
        for index, (key, label) in enumerate(LABELS.items()):
            row, col = divmod(index, 3)
            self.image_grid.rowconfigure(row, weight=1, uniform="images")
            self.image_grid.columnconfigure(col, weight=1, uniform="images")
            tile = ttk.Frame(self.image_grid)
            tile.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            ttk.Label(tile, text=label, style="Tile.TLabel").pack(fill="x")
            pane = ImagePane(tile)
            pane.pack(fill="both", expand=True)
            pane.bind("<Double-Button-1>", lambda _event, k=key: self.inspect_tile(k))
            self.panes[key] = pane

        self.spectrum = ImagePane(content, background="white",
                                  empty_text="Frequency plots unavailable. Select Regenerate preview to create them.")
        self.spectrum.grid(row=1, column=0, sticky="ew", padx=4, pady=(4, 0))
        self.spectrum.bind("<Double-Button-1>", lambda _event: self.inspect_spectrum())

        self.parameters = tk.StringVar(root)
        self.status = tk.StringVar(root)
        params_label = ttk.Label(body, textvariable=self.parameters, style="Status.TLabel")
        params_label.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        body.bind("<Configure>", lambda event: params_label.configure(wraplength=max(120, event.width - 32)))
        ttk.Label(body, textvariable=self.status, style="Status.TLabel").grid(row=3, column=0, sticky="ew")
        self.show(0)

    def resize_content(self, event):
        self.viewport.itemconfigure(self.content_item, width=event.width)
        # Scale the existing two-plot figure as a unit, then give the
        # image rows the remaining height. Scroll before shrinking tiles
        # below their minimum height.
        plot_width = min(1060, max(1, event.width - 8))
        plot_height = round(plot_width * 4.3 / 10.6)
        self.spectrum.configure(height=plot_height)
        self.image_grid.configure(height=max(374, min(620, event.height - plot_height - 8)))

    def scroll_content(self, event):
        if event.widget is self.selector:
            return
        if event.num in (4, 5):
            steps = -1 if event.num == 4 else 1
        elif event.delta:
            steps = (-1 if event.delta > 0 else 1) * max(1, abs(event.delta) // 120)
        else:
            return
        self.viewport.yview_scroll(3 * steps, "units")
        return "break"

    @property
    def name(self):
        return IMAGE_NAMES[self.index]

    def keyboard_move(self, delta):
        self.move(delta)
        return "break"

    def move(self, delta):
        self.show((self.index + delta) % len(IMAGE_NAMES))

    def show(self, index):
        self.index = index
        self.selector.current(index)
        name = self.name
        self.auto_preview = False
        try:
            if name in self.preview:
                view = self.preview[name]
                description = "In-memory preview"
            else:
                view, description = self.load_saved(name)
                self.auto_preview = not view.parameters
            self.display(view)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.display(ViewerResult({}))
            description = str(error)
        if name in self.preview_errors:
            description = f"{description}   ·   Preview failed: {self.preview_errors[name]}"
            self.auto_preview = False
        self.status.set(f"{index + 1} / {len(IMAGE_NAMES)}   ·   {name}   ·   {description}")
        if self.auto_preview and self.poll_id is None:
            self.regenerate()

    def read_original(self, name):
        source = self.inputs / name
        return read_input(source if source.exists() else self.output / "original" / name)

    def load_saved(self, name):
        original = self.read_original(name)
        images = {"original": Image.fromarray(original.pixels)}
        record = self.manifest.get(name, {})
        params = record.get("parameters")
        if not params or record.get("source_sha256") != sha256(original.original_bytes):
            return ViewerResult(images), "Waiting for preview"
        for key in METHODS:
            if key in record["metrics"]:
                with Image.open(self.output / key / name) as saved:
                    images[key] = saved.copy()
        if not record.get("padded_shape"):
            return ViewerResult(images, params), "Saved results · No saved frequency plots"
        relative = f"spectra/{Path(name).stem}_spectrum.png"
        try:
            data = (self.output / relative).read_bytes()
            expected = self.saved_hashes.get(relative)
            if expected and sha256(data) != expected:
                raise ValueError("figure differs from results.json")
            with Image.open(BytesIO(data)) as spectrum:
                if spectrum.format != "PNG":
                    raise ValueError("expected a PNG figure")
                return ViewerResult(images, params, spectrum.copy()), "Saved results"
        except (OSError, ValueError, SyntaxError) as error:
            detail = "figure is missing" if isinstance(error, FileNotFoundError) else "figure is unreadable or changed"
            return ViewerResult(images, params), f"Saved results · Spectrum unavailable ({detail})"

    def display(self, view):
        params = view.parameters
        parameter_text = ""
        if params:
            def compact(*names):
                return ", ".join(str(params[key]) for key in names)

            parameter_text = (
                f"Laplacian (α, neighbors) {compact('alpha', 'neighbors')}   |   "
                f"Gaussian (size, σ) {compact('size', 'sigma')}   |   High-boost k {params['high_boost']}\n"
                f"Homomorphic (γL, γH, D₀, c, DC preserve) "
                f"{compact('gamma_l', 'gamma_h', 'cutoff', 'c', 'preserve_dc')}   |   "
                f"Frequency boost (D₀, k) {compact('frequency_cutoff', 'frequency_gain')}"
            )
        for key, pane in self.panes.items():
            pane.set_image(view.images.get(key))
        self.spectrum.set_image(view.spectrum)
        self.parameters.set(parameter_text)

    def regenerate(self):
        if self.poll_id is not None:
            return
        name = self.name
        self.preview_errors.pop(name, None)
        self.regenerate_button.configure(state="disabled")
        self.status.set(f"{name}   ·   Computing preview…")
        Thread(target=self.generate_worker, args=(name,), daemon=True).start()
        self.poll_id = self.root.after(60, self.poll)

    def generate_worker(self, name):
        try:
            params = load_config(self.config, IMAGE_NAMES)[name]
            original = self.read_original(name)
            enhanced, _, diagnostics = sharpen(original.pixels, params, METHODS)
            images = {"original": Image.fromarray(original.pixels)}
            images.update({key: Image.fromarray(pixels) for key, pixels in enhanced.items()})
            with TemporaryDirectory(prefix="sharpening-preview-") as directory:
                path = Path(directory) / "spectrum.png"
                save_spectrum(path, name, diagnostics)
                with Image.open(path) as figure:
                    spectrum = figure.copy()
            self.in_flight.put((name, ViewerResult(images, asdict(params), spectrum), None))
        except Exception as error:
            # Transfer unexpected worker failures to the main thread as well.
            self.in_flight.put((name, None, str(error)))

    def poll(self):
        try:
            name, view, error = self.in_flight.get_nowait()
        except Empty:
            self.poll_id = self.root.after(60, self.poll)
            return
        self.poll_id = None
        self.regenerate_button.configure(state="normal")
        if error is None:
            self.preview[name] = view
        else:
            self.preview_errors[name] = error
        if name == self.name:
            if error is None:
                self.show(self.index)
            else:
                self.status.set(f"{name}   ·   Preview failed: {error}")
        elif self.auto_preview:
            self.regenerate()

    def inspect_tile(self, key):
        image = self.panes[key].source
        if image is not None:
            self.inspect(image, f"{self.name} · {LABELS[key]}")

    def inspect_spectrum(self):
        if self.spectrum.source is not None:
            self.inspect(self.spectrum.source, f"{self.name} · Spectrum and filter gains")

    def inspect(self, image, title):
        window = tk.Toplevel(self.root)
        window.title(title)
        width = min(image.width, max(1, self.root.winfo_screenwidth() - 120))
        height = min(image.height, max(1, self.root.winfo_screenheight() - 170))
        canvas = tk.Canvas(window, width=width, height=height,
                           scrollregion=(0, 0, image.width, image.height), highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        horizontal = ttk.Scrollbar(window, orient="horizontal", command=canvas.xview)
        vertical = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        horizontal.grid(row=1, column=0, sticky="ew")
        vertical.grid(row=0, column=1, sticky="ns")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        canvas.photo = ImageTk.PhotoImage(image, master=window)
        canvas.create_image(0, 0, image=canvas.photo, anchor="nw")

    def close(self):
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
        self.root.destroy()


def launch(inputs: Path, output: Path, config: Path) -> int:
    root = tk.Tk()
    SharpeningViewer(root, inputs, output, config)
    root.mainloop()
    return 0
