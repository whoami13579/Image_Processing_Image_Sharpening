"""Typed sharpening parameters and JSON overrides."""

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Parameters:
    alpha: float = 0.18
    neighbors: int = 4
    size: int = 5
    sigma: float = 0.9
    high_boost: float = 1.4
    gamma_l: float = 0.88
    gamma_h: float = 1.10
    cutoff: float = 0.025
    c: float = 1.0
    preserve_dc: bool = True
    frequency_cutoff: float = 0.10
    frequency_gain: float = 0.12

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "preserve_dc":
                if type(value) is not bool:
                    raise ValueError("preserve_dc must be true or false.")
            elif type(value) not in (int, float) or abs(value) > 1e6 or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number within ±1,000,000.")
        if self.alpha < 0 or type(self.neighbors) is not int or self.neighbors not in (4, 8):
            raise ValueError("Laplacian requires alpha >= 0 and 4 or 8 neighbors.")
        if type(self.size) is not int or not (1 <= self.size <= 101) or self.size % 2 == 0:
            raise ValueError("Gaussian size must be an odd integer from 1 to 101.")
        if self.sigma <= 0 or self.high_boost <= 1:
            raise ValueError("Gaussian sigma must be > 0 and high_boost must be > 1.")
        if not (0 < self.gamma_l < 1 < self.gamma_h) or min(self.cutoff, self.c) <= 0:
            raise ValueError("Homomorphic filtering requires 0 < gamma_l < 1 < gamma_h, cutoff > 0 and c > 0.")
        if min(self.frequency_cutoff, self.frequency_gain) <= 0:
            raise ValueError("Frequency-domain cutoff and gain must be > 0.")


def load_config(path: Path, names) -> dict[str, Parameters]:
    """Read one defaults object and optional overrides keyed by input filename."""
    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate JSON key.")
        return result

    try:
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        if not isinstance(config, dict) or set(config) - {"defaults", "images"}:
            raise ValueError("Expected an object with defaults and/or images keys.")
        defaults = Parameters(**config.get("defaults", {}))
        overrides = config.get("images", {})
        names = tuple(names)
        if not isinstance(overrides, dict) or set(overrides) - set(names):
            raise ValueError("Image overrides must use validated input filenames.")
        return {name: replace(defaults, **overrides.get(name, {})) for name in names}
    except (OSError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid parameter file {path}: {error}") from error
