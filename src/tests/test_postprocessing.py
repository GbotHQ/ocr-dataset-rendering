from random import seed

import cv2 as cv
import numpy as np

from tests.generate_test_data import make_random_gaussian_blur_sample
from bounding_box import QuadBoundingBox
from postprocess import (
    apply_random_gaussian_blur_to_sample,
    calculate_simple_bbox,
    calculate_precise_bbox,
    imread_coords,
)
from scipy.ndimage import map_coordinates


def test_apply_random_gaussian_blur_to_sample(tmp_path):
    seed(0)

    image_path = make_random_gaussian_blur_sample(tmp_path)

    gaussian_blur_radius = apply_random_gaussian_blur_to_sample(image_path, 9)

    img = cv.imread(str(image_path))

    assert np.all(
        img == cv.imread("tests/assets/test_apply_random_gaussian_blur_to_sample.png")
    )
    assert gaussian_blur_radius == 1


def test_line_by_line_bbox():
    document_bbox = QuadBoundingBox((10, 10), (400, 10), (400, 400), (10, 400))

    mask = cv.imread(
        "tests/assets/test_alpha_font_rendering_generate.png", cv.IMREAD_GRAYSCALE
    )

    src_shape = (512, 512)
    coords_relative, coords_absolute, _ = imread_coords(
        "tests/assets/coordinates0001.png", src_shape
    )

    mask_warped = map_coordinates(mask, coords_absolute, cval=255)

    bbox_line_by_line, bbox_line_by_line_relative_xxyy = calculate_precise_bbox(
        document_bbox, src_shape, mask_warped, coords_relative
    )

    ref_bbox_line_by_line = np.array(
        ((125, 134), (291, 136), (291, 440), (127, 439)), np.int32
    )
    ref_bbox_line_by_line_relative_xxyy = np.array(
        (
            (0.24414062, 0.26171875),
            (0.5683594, 0.265625),
            (0.5683594, 0.859375),
            (0.24804688, 0.8574219),
        ),
        np.float32,
    )

    np.testing.assert_array_equal(
        bbox_line_by_line.points, ref_bbox_line_by_line
    )
    np.testing.assert_array_equal(
        bbox_line_by_line_relative_xxyy.points,
        ref_bbox_line_by_line_relative_xxyy,
    )


def test_simple_bbox():
    mask = cv.imread(
        "tests/assets/test_alpha_font_rendering_generate.png", cv.IMREAD_GRAYSCALE
    )

    src_shape = (512, 512)
    coords_relative, coords_absolute, _ = imread_coords(
        "tests/assets/coordinates0001.png", src_shape
    )

    mask_warped = map_coordinates(mask, coords_absolute, cval=255)

    bbox, bbox_relative_xxyy = calculate_simple_bbox(mask_warped, coords_relative)

    ref_bbox = np.array(((124, 134), (293, 441)), np.int32)
    ref_bbox_relative_xxyy = (0.2421875, 0.5722656, 0.26171875, 0.8613281)

    np.testing.assert_array_equal(bbox.points, ref_bbox)
    np.testing.assert_array_equal(
        np.array(bbox_relative_xxyy, np.float32),
        np.array(ref_bbox_relative_xxyy, np.float32),
    )
