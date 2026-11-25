"""Transfer functions sampled at the unshifted FFT bin coordinates."""

import numpy as np

from .fourier import signed_frequencies


def squared_distance(shape: tuple[int, int]) -> np.ndarray:
    v, u = (signed_frequencies(n) for n in shape)
    return v[:, None] ** 2 + u[None, :] ** 2


def homomorphic_response(distance2, *, gamma_l, gamma_h, cutoff, c, preserve_dc):
    with np.errstate(over="ignore"):
        transition = -np.expm1(-c * (np.sqrt(distance2) / cutoff) ** 2)
    response = gamma_l + (gamma_h - gamma_l) * transition
    if preserve_dc:
        response[0, 0] = 1.0
    return response


def gaussian_high_boost_response(distance2, *, cutoff, gain):
    # 1 - exp(-t), evaluated accurately near DC.
    with np.errstate(over="ignore"):
        high_pass = -np.expm1(-0.5 * (np.sqrt(distance2) / cutoff) ** 2)
    return 1.0 + gain * high_pass
