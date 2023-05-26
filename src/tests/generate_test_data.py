from pathlib import Path as pth
from random import seed

import numpy as np
import cv2 as cv
from PIL import ImageFont
from bounding_box import QuadBoundingBox

from postprocess import (
    apply_random_gaussian_blur_to_sample,
    imread_coords,
    calculate_precise_bbox,
    calculate_simple_bbox,
)
from font_rendering import generate
from scipy.ndimage import map_coordinates


def make_random_gaussian_blur_sample(path):
    path = pth(path)
    image_path = path / "test_apply_random_gaussian_blur_to_sample.png"

    img = cv.imread("tests/assets/test_img_font_rendering_generate.png")
    cv.imwrite(str(image_path), img)

    return str(image_path)


def generate_font_rendering_test_data():
    seed(0)

    original_text = "The quick brown fox"
    font = ImageFont.truetype("tests/assets/SilentReaction.ttf", 42)

    _, img, alpha, _, _, _, _ = generate(original_text, font)

    compression = [cv.IMWRITE_PNG_COMPRESSION, 7]
    cv.imwrite("tests/assets/test_img_font_rendering_generate.png", img, compression)
    cv.imwrite(
        "tests/assets/test_alpha_font_rendering_generate.png", alpha, compression
    )


def generate_random_gaussian_blur_test_data():
    seed(0)

    image_path = make_random_gaussian_blur_sample("tests/assets")
    gaussian_blur_radius = apply_random_gaussian_blur_to_sample(image_path, 9)

    print(gaussian_blur_radius)


def generate_line_by_line_bbox_test_data():
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

    print(bbox_line_by_line.points, bbox_line_by_line_relative_xxyy.points)


def generate_simple_bbox_test_data():
    mask = cv.imread(
        "tests/assets/test_alpha_font_rendering_generate.png", cv.IMREAD_GRAYSCALE
    )

    src_shape = (512, 512)
    coords_relative, coords_absolute, _ = imread_coords(
        "tests/assets/coordinates0001.png", src_shape
    )

    mask_warped = map_coordinates(mask, coords_absolute, cval=255)

    bbox, bbox_relative_xxyy = calculate_simple_bbox(mask_warped, coords_relative)

    print(bbox.points, bbox_relative_xxyy)


if __name__ == "__main__":
    generate_font_rendering_test_data()
    generate_random_gaussian_blur_test_data()
    generate_line_by_line_bbox_test_data()
    generate_simple_bbox_test_data()
