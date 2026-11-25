"""Independent spatial-domain checks, including asymmetric convolution."""

from dataclasses import replace
import math

import numpy as np
import pytest

from image_sharpening.algorithms import METHODS, laplacian, sharpen, to_uint8
from image_sharpening.config import Parameters
from image_sharpening.convolution import (LAPLACIAN_4, LAPLACIAN_8, convolve2d,
                                          gaussian_kernel, gaussian_smooth, symmetric_indices)


def reflect(index, length):
    while index < 0 or index >= length:
        index = -index - 1 if index < 0 else 2 * length - index - 1
    return index


@pytest.mark.parametrize("shape", [(2, 3), (1, 4), (5, 1), (1, 1)])
def test_asymmetric_convolution_matches_scalar_reference(shape):
    image = np.arange(math.prod(shape)).reshape(shape)
    kernel = [[-1, 0, 4], [0, 6, 1], [7, 0, 0]]
    expected = np.zeros(shape)
    for y, x in np.ndindex(shape):
        expected[y, x] = sum(kernel[i][j] * image[reflect(y + 1 - i, shape[0]),
                                                   reflect(x + 1 - j, shape[1])]
                             for i, j in np.ndindex((3, 3)))
    np.testing.assert_array_equal(convolve2d(image, kernel), expected)


@pytest.mark.parametrize("length", [1, 2, 3, 9])
def test_symmetric_boundary_extends_multiple_periods(length):
    np.testing.assert_array_equal(symmetric_indices(length, 17, 13),
                                  [reflect(i, length) for i in range(-17, length + 13)])


@pytest.mark.parametrize("size,sigma", [(1, 1.0), (3, 0.7), (5, 0.9), (7, 1.4)])
def test_gaussian_definition_and_separable_smoothing(size, sigma):
    offsets = range(-size // 2 + 1, size // 2 + 1)
    reference = np.array([[math.exp(-(x*x + y*y) / (2*sigma*sigma)) for x in offsets] for y in offsets])
    reference /= reference.sum()
    kernel = gaussian_kernel(size, sigma)
    np.testing.assert_allclose(kernel, reference, atol=1e-15)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1])
    assert kernel.sum() == pytest.approx(1)
    image = np.arange(15).reshape(3, 5)
    np.testing.assert_allclose(gaussian_smooth(image, size, sigma), convolve2d(image, kernel))


@pytest.mark.parametrize("neighbors,mask", [(4, LAPLACIAN_4), (8, LAPLACIAN_8)])
def test_laplacian_mask_and_impulse(neighbors, mask):
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 3] = 10
    expected = np.zeros_like(image, dtype=float)
    expected[2:5, 2:5] = 10 * np.array(mask)
    np.testing.assert_array_equal(laplacian(image, neighbors), expected)
    assert expected[3, 3] == -neighbors * 10
    assert expected.sum() == 0


@pytest.mark.parametrize("value", [0, 57, 128, 255])
def test_constant_is_preserved_for_all_methods_with_dc_option(value):
    image = np.full((7, 11), value, dtype=np.uint8)
    outputs, _, _ = sharpen(image, Parameters())
    assert tuple(outputs) == METHODS
    for result in outputs.values():
        np.testing.assert_array_equal(result, image)


def test_step_edge_prediction_and_amplification():
    image = np.array([[70, 70, 70, 130, 130, 130]], dtype=np.uint8)
    p = replace(Parameters(), alpha=0.25, size=3, sigma=1.0, high_boost=1.5)
    output, times, _ = sharpen(image, p, ("laplacian", "unsharp", "high_boost"))
    weight = math.exp(-0.5) / (1 + 2 * math.exp(-0.5))
    vals = (60 * weight, 60 * weight * p.high_boost)
    assert list(output["laplacian"][0]) == [70, 70, 55, 145, 130, 130]
    for method, val in zip(("unsharp", "high_boost"), vals):
        expected = [70, 70, math.floor(70 - val + 0.5), math.floor(130 + val + 0.5), 130, 130]
        assert list(output[method][0]) == expected
    assert output["high_boost"][0, 3] > output["unsharp"][0, 3] > image[0, 3]
    assert set(times) == {"laplacian", "unsharp", "high_boost", "shared_gaussian"}


def test_rounding_and_clamping():
    values = np.array([-1000.0, -0.6, 0.49, 0.5, 125.5, 254.499, 254.5, 1000.])
    output = to_uint8(values)
    assert output.dtype == np.uint8
    assert list(output) == [0, 0, 0, 1, 126, 254, 255, 255]


@pytest.mark.parametrize("image", [np.array(5), np.array([[2.0]]), np.zeros((0, 1), dtype=np.uint8)])
def test_bad_sharpen_source(image):
    with pytest.raises(ValueError, match="uint8"):
        sharpen(image, Parameters())
