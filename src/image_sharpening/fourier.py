"""Radix-2 Cooley–Tukey FFT, mirrored extension, and spectrum indexing."""

from functools import lru_cache

import numpy as np

from .convolution import symmetric_extend


@lru_cache(maxsize=20)
def _plan(length: int):
    """Cache bit reversal and forward twiddle factors for each stage."""
    if length < 1 or length & (length - 1):
        raise ValueError("FFT axis length must be a positive power of two.")
    indices = np.arange(length)
    reversed_bits = np.zeros(length, dtype=np.intp)
    for _ in range(length.bit_length() - 1):
        reversed_bits = (reversed_bits << 1) | (indices & 1)
        indices >>= 1
    factors = []
    for stage in range(1, length.bit_length()):
        block = 1 << stage
        angle = -2 * np.pi * np.arange(block // 2) / block
        factors.append(np.cos(angle) + 1j * np.sin(angle))
    return reversed_bits, factors


def fft1(values, *, inverse: bool = False) -> np.ndarray:
    """Transform the last axis, treating leading dimensions as independent batches."""
    source = np.asarray(values, dtype=np.complex128)
    if source.ndim == 0:
        raise ValueError("FFT requires at least one axis.")
    length = source.shape[-1]
    indices, factors = _plan(length)
    result = np.ascontiguousarray(source[..., indices])
    for twiddles in factors:
        half = twiddles.size
        groups = result.reshape(*source.shape[:-1], length // (2 * half), 2 * half)
        even = groups[..., :half].copy()
        odd = groups[..., half:] * (twiddles.conj() if inverse else twiddles)
        groups[..., :half] = even + odd
        groups[..., half:] = even - odd
    if inverse:
        result /= length
    return result


def ifft1(values) -> np.ndarray:
    return fft1(values, inverse=True)


def fft2(values, *, inverse: bool = False) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 2:
        raise ValueError("2-D FFT requires a two-dimensional array.")
    rows = fft1(source, inverse=inverse)
    return fft1(rows.T, inverse=inverse).T


def ifft2(values) -> np.ndarray:
    return fft2(values, inverse=True)


def mirror_for_fft(image) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Symmetrically extend to powers of two, then mirror the whole tile on both axes."""
    padding = []
    crop = []
    for length in image.shape:
        remainder = (1 << (length - 1).bit_length()) - length
        before = remainder // 2
        padding.append((before, remainder - before))
        crop.append(slice(before, before + length))
    tile = symmetric_extend(image, *padding)
    extended = symmetric_extend(tile, (0, tile.shape[0]), (0, tile.shape[1]))
    return extended, (crop[0], crop[1])


def signed_frequencies(length: int) -> np.ndarray:
    """Return unshifted DFT bins in cycles/pixel, including negative Nyquist."""
    bins = np.arange(length)
    return np.where(bins < (length + 1) // 2, bins, bins - length) / length


def center_spectrum(spectrum) -> np.ndarray:
    """Move the zero-frequency bin to the middle for display only."""
    y, x = ((np.arange(n) - n // 2) % n for n in spectrum.shape)
    return spectrum[y[:, None], x]
