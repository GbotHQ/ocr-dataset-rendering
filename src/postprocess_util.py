from typing import Iterable
import math
from random import random

import numpy as np
import cv2 as cv

from bounding_box import SimpleBoundingBox
from typing import Tuple


def to_uint(img, dtype=np.uint8):
    return (np.clip(img, 0, 1) * np.iinfo(dtype).max).astype(dtype)


def to_float(img, fdtype=np.float32):
    return img.astype(fdtype) / np.iinfo(img.dtype).max


def imread_coords(path):
    # unchanged to read as uint16
    coords = to_float(cv.imread(path, cv.IMREAD_UNCHANGED))
    alpha = coords[..., 0, None]
    # flip y to match opencv coordinates
    coords[..., 1] = 1 - coords[..., 1]
    coords = np.where(alpha < 1, -1, coords[..., 1:])
    coords = coords[..., ::-1]

    return coords, alpha


def calculate_bbox(
    img: np.ndarray, background_color: int, tolerance: int = 12
) -> SimpleBoundingBox:
    mask = np.abs(img.astype(np.int32) - background_color) > tolerance
    mask = (mask.any(0), mask.any(1))
    bbox = [(np.argmax(k), k.size - np.argmax(k[::-1])) for k in mask]
    bbox = np.swapaxes(np.array(bbox, np.int32), 0, 1)
    bbox = SimpleBoundingBox(bbox[0], bbox[1], bbox.dtype)
    # convert from yx to xy
    bbox.points = bbox.points[:, ::-1]
    return bbox


def apply_random_gauss_blur(img: np.ndarray) -> Tuple[np.ndarray, int]:
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
