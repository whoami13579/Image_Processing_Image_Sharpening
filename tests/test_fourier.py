"""Check forward and inverse transforms independently against direct DFT sums."""

from dataclasses import replace
import cmath
import math

import numpy as np
import pytest

from image_sharpening.algorithms import (prepare_log_spectrum, reconstruct_intensity, sharpen,
                                         to_uint8)
from image_sharpening.config import Parameters
from image_sharpening.fourier import (center_spectrum, fft1, fft2, ifft1, ifft2, mirror_for_fft,
                                      signed_frequencies)
from image_sharpening.frequency_filters import (gaussian_high_boost_response, homomorphic_response,
                                                  squared_distance)


def direct_1d(x, inverse=False):
    n = len(x)
    sign = 1 if inverse else -1
    return [sum(value * cmath.exp(sign * 2j * math.pi * k * t / n)
                for t, value in enumerate(x)) / (n if inverse else 1) for k in range(n)]


def direct_2d(image, inverse=False):
    h, w = image.shape
    sign = 1 if inverse else -1
    out = np.zeros_like(image, dtype=complex)
    for v, u in np.ndindex(image.shape):
        out[v, u] = sum(image[y, x] * cmath.exp(sign * 2j * math.pi * (y*v/h + x*u/w))
                        for y, x in np.ndindex(image.shape)) / (h * w if inverse else 1)
    return out


@pytest.mark.parametrize("length", [1, 2, 4, 8, 16])
def test_batch_fft_and_inverse_match_direct_dft(length):
    data = np.random.default_rng(length).normal(size=(3, length)) + 1j * np.arange(length)
    np.testing.assert_allclose(fft1(data), [direct_1d(row) for row in data], atol=1e-11)
    np.testing.assert_allclose(ifft1(data), [direct_1d(row, True) for row in data], atol=1e-11)
    np.testing.assert_allclose(ifft1(fft1(data)), data, atol=1e-11)


@pytest.mark.parametrize("shape", [(1, 1), (2, 4), (4, 8), (1, 8), (8, 1)])
def test_2d_transform_and_inverse_against_full_double_sum(shape):
    data = np.random.default_rng(42).normal(size=shape) + 1j
    np.testing.assert_allclose(fft2(data), direct_2d(data), atol=1e-11)
    np.testing.assert_allclose(ifft2(data), direct_2d(data, True), atol=1e-11)
    np.testing.assert_allclose(ifft2(fft2(data)), data, atol=1e-11)


@pytest.mark.parametrize("shape", [(0,), (6,), (4, 3)])
def test_non_power_of_two_rejected(shape):
    with pytest.raises(ValueError, match="power of two"):
        fft1(np.zeros(shape))


def test_padding_crop_and_mirrored_periodic_seam():
    image = np.fromfunction(lambda y, x: y + x * 0.5, (420, 512))
    extended, crop = mirror_for_fft(image)
    assert extended.shape == (1024, 1024)
    assert crop == (slice(46, 466), slice(0, 512))
    np.testing.assert_array_equal(extended[crop], image)
    np.testing.assert_array_equal(extended[:, 0], extended[:, -1])
    np.testing.assert_array_equal(extended[0, :], extended[-1, :])
    np.testing.assert_array_equal(extended[:, 511], extended[:, 512])
    np.testing.assert_array_equal(extended[511, :], extended[512, :])


@pytest.mark.parametrize("shape", [(5, 7), (4, 8), (1, 1)])
def test_manual_center_index_permutation(shape):
    spectrum = np.arange(math.prod(shape)).reshape(shape)
    centered = center_spectrum(spectrum)
    h, w = shape
    assert centered[h // 2, w // 2] == spectrum[0, 0]
    for y, x in np.ndindex(shape):
        assert centered[(y + h // 2) % h, (x + w // 2) % w] == spectrum[y, x]
    dc_only = center_spectrum(fft2(np.ones((4, 8))))
    assert np.unravel_index(np.abs(dc_only).argmax(), dc_only.shape) == (2, 4)


def test_filter_gain_formula_and_signed_frequency_coordinates():
    np.testing.assert_array_equal(signed_frequencies(4), [0, .25, -.5, -.25])
    np.testing.assert_allclose(signed_frequencies(5), [0, .2, .4, -.4, -.2])
    distance = squared_distance((8, 16))
    p = Parameters()
    args = dict(gamma_l=p.gamma_l, gamma_h=p.gamma_h, cutoff=p.cutoff, c=p.c)
    response = homomorphic_response(distance, **args, preserve_dc=False)
    compensated = homomorphic_response(distance, **args, preserve_dc=True)
    hb = gaussian_high_boost_response(distance, cutoff=p.frequency_cutoff, gain=p.frequency_gain)
    assert response[0, 0] == p.gamma_l
    assert compensated[0, 0] == hb[0, 0] == 1.0
    assert response[0, 1] < response[0, 2] < response[0, 3]
    assert hb[0, 1] < hb[0, 2] < hb[0, 3] < 1 + p.frequency_gain
    for v, u in np.ndindex(distance.shape):
        d2 = min(v, 8-v)**2 / 8**2 + min(u, 16-u)**2 / 16**2
        assert response[v, u] == pytest.approx(p.gamma_h - (p.gamma_h - p.gamma_l) * math.exp(-p.c*d2/p.cutoff**2))
        assert hb[v, u] == pytest.approx(1 + p.frequency_gain * (1 - math.exp(-d2/(2*p.frequency_cutoff**2))))
        assert hb[v, u] == hb[-v % 8, -u % 16]


def test_identity_response_reconstructs_non_power_of_two_image_including_zero_white():
    data = np.array([[0, 23, 255], [250, 14, 109], [3, 9, 90]], dtype=float)
    spec, crop = prepare_log_spectrum(data)
    np.testing.assert_array_equal(to_uint8(reconstruct_intensity(spec, np.ones_like(spec), crop)), data)


def test_uncompensated_constant_uses_gamma_l_in_log_domain():
    p = replace(Parameters(), preserve_dc=False)
    data = np.full((4, 6), 90, np.uint8)
    results, _, _ = sharpen(data, p, ("homomorphic", "combined"))
    expected = math.floor(math.expm1(math.log1p(90) * p.gamma_l) + 0.5)
    for array in results.values():
        np.testing.assert_array_equal(array, expected)


def test_combined_matches_independent_direct_dft_product():
    data = np.array([[40, 150], [22, 72]], dtype=np.uint8)
    out, _, diagnostics = sharpen(data, Parameters(), ("homomorphic", "combined"))
    logged = np.log1p(data.astype(float))
    reflected = logged[np.ix_([0, 1, 1, 0], [0, 1, 1, 0])]
    reference = direct_2d(reflected)
    product = diagnostics["homomorphic"] * diagnostics["high_boost"]
    np.testing.assert_array_equal(diagnostics["combined"], product)
    filtered = direct_2d(reference * product, inverse=True).real[:2, :2]
    expected = np.floor(np.expm1(filtered) + 0.5)
    np.testing.assert_array_equal(out["combined"], expected)
    assert np.any(out["combined"] != out["homomorphic"])
