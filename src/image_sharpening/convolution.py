"""Explicit spatial convolution with symmetric boundaries."""

import numpy as np

LAPLACIAN_4 = ((0, 1, 0), (1, -4, 1), (0, 1, 0))
LAPLACIAN_8 = ((1, 1, 1), (1, -8, 1), (1, 1, 1))


def symmetric_indices(length: int, before: int, after: int) -> np.ndarray:
    """Mirror with duplicated edge samples: … b a | a b … y z | z y …"""
    positions = np.arange(-before, length + after) % (2 * length)
    return np.minimum(positions, 2 * length - 1 - positions)


def symmetric_extend(image, rows: tuple[int, int], cols: tuple[int, int]):
    y = symmetric_indices(image.shape[0], *rows)
    x = symmetric_indices(image.shape[1], *cols)
    return image[y[:, None], x]


def convolve2d(image, kernel) -> np.ndarray:
    """Convolve a nonempty 2-D array with an odd-sized kernel."""
    source = np.asarray(image, dtype=np.float64)
    weights = np.asarray(kernel, dtype=np.float64)
    if source.ndim != 2 or not source.size:
        raise ValueError("Convolution requires a nonempty two-dimensional image.")
    if weights.ndim != 2 or any(n % 2 != 1 for n in weights.shape):
        raise ValueError("Convolution kernel dimensions must be positive and odd.")
    height, width = source.shape
    y_radius, x_radius = (n // 2 for n in weights.shape)
    extended = symmetric_extend(source, (y_radius, y_radius), (x_radius, x_radius))
    filtered = np.zeros_like(source)
    # Reverse both kernel axes for convolution, including asymmetric kernels.
    for (y, x), weight in np.ndenumerate(weights[::-1, ::-1]):
        if weight:
            filtered += weight * extended[y:y + height, x:x + width]
    return filtered


def gaussian_weights(size: int, sigma: float) -> np.ndarray:
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1 or size % 2 != 1:
        raise ValueError("Gaussian size must be a positive odd integer.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive.")
    offsets = np.arange(size, dtype=np.float64) - size // 2
    with np.errstate(over="ignore"):
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    weights = gaussian_weights(size, sigma)
    return weights[:, None] * weights[None, :]


def gaussian_smooth(image, size: int, sigma: float) -> np.ndarray:
    weights = gaussian_weights(size, sigma)
    return convolve2d(convolve2d(image, weights[None, :]), weights[:, None])
