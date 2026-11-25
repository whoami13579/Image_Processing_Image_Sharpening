"""Five sharpening methods operating on a shared float64 grayscale source."""

from time import perf_counter

import numpy as np

from .config import Parameters
from .convolution import LAPLACIAN_4, LAPLACIAN_8, convolve2d, gaussian_smooth
from .fourier import fft2, ifft2, mirror_for_fft
from .frequency_filters import gaussian_high_boost_response, homomorphic_response, squared_distance

LABELS = {
    "original": "Original",
    "laplacian": "Laplacian",
    "unsharp": "Unsharp Masking",
    "high_boost": "High-boost",
    "homomorphic": "Homomorphic",
    "combined": "Combined",
}
METHODS = tuple(LABELS)[1:]


def to_uint8(values) -> np.ndarray:
    """Clamp explicitly, round half up, and convert to 8-bit grayscale."""
    return np.floor(np.minimum(255.0, np.maximum(0.0, values)) + 0.5).astype(np.uint8)


def laplacian(image, neighbors=4) -> np.ndarray:
    return convolve2d(image, {4: LAPLACIAN_4, 8: LAPLACIAN_8}[neighbors])


def prepare_log_spectrum(image):
    extended, crop = mirror_for_fft(np.log1p(image))
    return fft2(extended), crop


def reconstruct_intensity(spectrum, response, crop) -> np.ndarray:
    # A real, even response preserves Hermitian symmetry. Discard numerical
    # imaginary residue, and cap the log intensity before exp to avoid overflow.
    spatial_log = ifft2(spectrum * response).real[crop]
    return np.expm1(np.minimum(np.log1p(255.0), np.maximum(0.0, spatial_log)))


def sharpen(image, parameters: Parameters, methods=METHODS):
    """Return enhanced pixels, stage timings, and optional spectral diagnostics."""
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8 or not image.size:
        raise ValueError("Expected a nonempty two-dimensional uint8 grayscale image.")
    methods = tuple(methods)
    if not methods or set(methods) - set(METHODS) or len(set(methods)) != len(methods):
        raise ValueError("Choose distinct known sharpening methods.")
    source = image.astype(np.float64)
    pixels, runtimes = {}, {}
    detail = None
    spectral = {}
    p = parameters

    def save_result(name, array, start):
        pixels[name] = to_uint8(array)
        runtimes[name] = (perf_counter() - start) * 1000

    if "laplacian" in methods:
        start = perf_counter()
        save_result("laplacian", source - p.alpha * laplacian(source, p.neighbors), start)
    if {"unsharp", "high_boost"} & set(methods):
        start = perf_counter()
        detail = source - gaussian_smooth(source, p.size, p.sigma)
        runtimes["shared_gaussian"] = (perf_counter() - start) * 1000
    if "unsharp" in methods:
        start = perf_counter()
        save_result("unsharp", source + detail, start)
    if "high_boost" in methods:
        start = perf_counter()
        save_result("high_boost", source + p.high_boost * detail, start)
    if {"homomorphic", "combined"} & set(methods):
        start = perf_counter()
        spectrum, crop = prepare_log_spectrum(source)
        distance2 = squared_distance(spectrum.shape)
        homo = homomorphic_response(distance2, gamma_l=p.gamma_l, gamma_h=p.gamma_h,
                                    cutoff=p.cutoff, c=p.c, preserve_dc=p.preserve_dc)
        runtimes["shared_frequency"] = (perf_counter() - start) * 1000
        spectral = {"spectrum": spectrum, "homomorphic": homo}
        if "homomorphic" in methods:
            start = perf_counter()
            save_result("homomorphic", reconstruct_intensity(spectrum, homo, crop), start)
        if "combined" in methods:
            start = perf_counter()
            boost = gaussian_high_boost_response(distance2, cutoff=p.frequency_cutoff,
                                                 gain=p.frequency_gain)
            combined = homo * boost
            save_result("combined", reconstruct_intensity(spectrum, combined, crop), start)
            spectral.update(high_boost=boost, combined=combined)
    return pixels, runtimes, spectral
