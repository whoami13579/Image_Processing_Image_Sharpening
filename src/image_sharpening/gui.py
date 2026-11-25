"""Responsive Tkinter viewer with read-only saved results and in-memory previews."""

import json
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Thread
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .algorithms import LABELS, METHODS, sharpen
from .config import load_config
from .pipeline import IMAGE_NAMES, RESULTS_FILE, read_input, sha256


class ImagePane(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent, background="#101d2d", highlightthickness=0,
                         width=1, height=1, cursor="hand2")
        self.source = None
        self.photo = None
        self.item = self.create_image(0, 0)
        self.bind("<Configure>", self.draw)

    def set_image(self, source):
        self.source = source
        self.draw()

    def draw(self, _event=None):
        width, height = self.winfo_width(), self.winfo_height()
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
        self.in_flight = SimpleQueue()
        self.poll_id = None
        try:
            manifest = json.loads((output / RESULTS_FILE).read_text(encoding="utf-8"))
            self.manifest = manifest.get("images", {})
        except (OSError, ValueError, AttributeError):
            self.manifest = {}

        root.title("Image Sharpening")
        root.geometry("1240x880")
        root.minsize(720, 570)
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
        ttk.Button(controls, text="Previous", command=lambda: self.move(-1)).pack(side="left", padx=(8, 4))
        ttk.Button(controls, text="Next", command=lambda: self.move(1)).pack(side="left")
        ttk.Button(controls, text="Spectrum", command=self.show_spectrum).pack(side="right")
        self.regenerate_button = ttk.Button(controls, text="Regenerate preview", command=self.regenerate)
        self.regenerate_button.pack(side="right", padx=6)
        root.bind("<Left>", lambda _event: self.keyboard_move(-1))
        root.bind("<Right>", lambda _event: self.keyboard_move(1))

        grid = ttk.Frame(body, style="Toolbar.TFrame")
        grid.grid(row=1, column=0, sticky="nsew")
        self.panes = {}
        for index, (key, label) in enumerate(LABELS.items()):
            row, col = divmod(index, 3)
            grid.rowconfigure(row, weight=1, uniform="images")
            grid.columnconfigure(col, weight=1, uniform="images")
            tile = ttk.Frame(grid)
            tile.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            ttk.Label(tile, text=label, style="Tile.TLabel").pack(fill="x")
            pane = ImagePane(tile)
            pane.pack(fill="both", expand=True)
            pane.bind("<Double-Button-1>", lambda _event, k=key: self.inspect_tile(k))
            self.panes[key] = pane

        self.parameters = tk.StringVar(root)
        self.status = tk.StringVar(root)
        params_label = ttk.Label(body, textvariable=self.parameters, style="Status.TLabel")
        params_label.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        body.bind("<Configure>", lambda event: params_label.configure(wraplength=max(120, event.width - 32)))
        ttk.Label(body, textvariable=self.status, style="Status.TLabel").grid(row=3, column=0, sticky="ew")
        self.show(0)
        if not any(pane.source is not None for key, pane in self.panes.items() if key != "original"):
            if self.panes["original"].source is not None:
                self.regenerate()

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
        self.parameters.set("")
        for pane in self.panes.values():
            pane.set_image(None)
        try:
            if name in self.preview:
                images, params = self.preview[name]
                description = "In-memory preview"
            else:
                record = self.manifest.get(name, {})
                source_path = self.inputs / name
                if not source_path.exists():
                    source_path = self.output / "original" / name
                original = read_input(source_path)
                images = {"original": Image.fromarray(original.pixels)}
                self.panes["original"].set_image(images["original"])
                params = record.get("parameters") if record.get("source_sha256") == sha256(original.original_bytes) else None
                if params:
                    for key in record["metrics"]:
                        if key != "original":
                            with Image.open(self.output / key / name) as saved:
                                images[key] = saved.copy()
                description = "Saved results" if params else "Select Regenerate preview to process this image"
            for key, image in images.items():
                self.panes[key].set_image(image)
            if params:
                def compact(*names):
                    return ", ".join(str(params[key]) for key in names)

                self.parameters.set(
                    f"Laplacian (α, neighbors) {compact('alpha', 'neighbors')}   |   "
                    f"Gaussian (size, σ) {compact('size', 'sigma')}   |   High-boost k {params['high_boost']}\n"
                    f"Homomorphic (γL, γH, D₀, c, DC preserve) "
                    f"{compact('gamma_l', 'gamma_h', 'cutoff', 'c', 'preserve_dc')}   |   "
                    f"Frequency boost (D₀, k) {compact('frequency_cutoff', 'frequency_gain')}"
                )
            self.status.set(f"{index + 1} / {len(IMAGE_NAMES)}   ·   {name}   ·   {description}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status.set(f"{name}   ·   {error}")

    def regenerate(self):
        if self.poll_id is not None:
            return
        name = self.name
        self.regenerate_button.configure(state="disabled")
        self.status.set(f"{name}   ·   Computing preview…")
        Thread(target=self.generate_worker, args=(name,), daemon=True).start()
        self.poll_id = self.root.after(60, self.poll)

    def generate_worker(self, name):
        from dataclasses import asdict

        try:
            params = load_config(self.config, IMAGE_NAMES)[name]
            original = read_input(self.inputs / name)
            enhanced, _, _ = sharpen(original.pixels, params, METHODS)
            images = {"original": Image.fromarray(original.pixels)}
            images.update({key: Image.fromarray(pixels) for key, pixels in enhanced.items()})
            self.in_flight.put((name, images, asdict(params), None))
        except Exception as error:
            # Transfer unexpected worker failures to the main thread as well.
            self.in_flight.put((name, None, None, str(error)))

    def poll(self):
        try:
            name, images, parameters, error = self.in_flight.get_nowait()
        except Empty:
            self.poll_id = self.root.after(60, self.poll)
            return
        self.poll_id = None
        self.regenerate_button.configure(state="normal")
        if error:
            self.status.set(f"{name}   ·   Preview failed: {error}")
            return
        self.preview[name] = images, parameters
        self.show(self.index)

    def inspect_tile(self, key):
        image = self.panes[key].source
        if image is not None:
            self.inspect(image, f"{self.name} · {LABELS[key]}")

    def show_spectrum(self):
        path = self.output / "spectra" / f"{Path(self.name).stem}_spectrum.png"
        record = self.manifest.get(self.name, {})
        if self.name in self.preview or not record.get("padded_shape"):
            self.status.set(f"{self.name}   ·   Spectrum available for saved frequency results")
            return
        try:
            with Image.open(path) as spectrum:
                self.inspect(spectrum.copy(), f"{self.name} · Saved spectrum and filter gains")
        except OSError as error:
            self.status.set(f"{self.name}   ·   Spectrum unavailable: {error}")

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
