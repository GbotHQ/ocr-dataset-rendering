from typing import Iterable, Tuple, Union, List
import math
from random import random
import concurrent.futures
import multiprocessing
from pathlib import Path as pth

import numpy as np
import cv2 as cv

from bounding_box import (
    SimpleBoundingBox,
)
from bounding_box_math import bbox_from_mask
from calculate_bounding_boxes import calculate_bounding_boxes
from generate_samples_2d import SampleInfo


def random_gaussian_blur(img: np.ndarray) -> Tuple[np.ndarray, int]:
    radius = math.floor((random() ** 3) * 2)
    if radius > 0:
        img = cv.GaussianBlur(img, (radius * 2 + 1,) * 2, 0)
    return img, radius


def make_grid(images: Iterable[np.ndarray]) -> np.ndarray:
    resolution = images[0].shape[:2]
    n_images = len(images)
    grid_size = n_images**0.5
    grid_size = math.ceil(grid_size)

    img = np.full(
        (resolution[0] * grid_size, resolution[1] * grid_size, 3), 0, dtype=np.uint8
    )

    for x in range(grid_size):
        for y in range(grid_size):
            i = x * grid_size + y
            if i >= n_images:
                break
            img[
                x * resolution[0] : (x + 1) * resolution[0],
                y * resolution[1] : (y + 1) * resolution[1],
            ] = images[i]
        else:
            continue
        break

    return img


def calculate_mask(document_img: np.ndarray):
    mask = 1 - document_img.astype(np.float32) / 255
    mask = 1 - mask / np.max(mask, axis=(0, 1))[None, None]
    mask = np.clip(mask[:, :, 0], 0, 1)
    return (mask * 255).astype(np.uint8)


def apply_random_gaussian_blur_to_sample(
    image_path: Union[str, pth], compression_level: int
):
    img = cv.imread(str(image_path))
    img, gaussian_blur_radius = random_gaussian_blur(img)
    cv.imwrite(
        str(image_path),
        img,
        [cv.IMWRITE_PNG_COMPRESSION, compression_level],
    )
    return gaussian_blur_radius


def calculate_simple_bbox(
    mask_warped: np.ndarray, coords_relative: np.ndarray
) -> Tuple[SimpleBoundingBox, SimpleBoundingBox]:
    bbox = bbox_from_mask(mask_warped, 255)
    bbox_relative_xxyy = bbox.relative(coords_relative.shape[:2][::-1]).xxyy()

    return bbox, bbox_relative_xxyy


def postprocess_sample(sample: SampleInfo):
    bounding_boxes, _, _, _ = calculate_bounding_boxes(sample)

    sample.bounding_boxes = bounding_boxes

    # sample.gaussian_blur_radius = apply_random_gaussian_blur_to_sample(
    #     sample.output_image_path, sample.compression_level
    # )


def postprocess_samples(samples: List[SampleInfo]):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    ) as executor:
        list(executor.map(postprocess_sample, samples))
