from typing import Tuple
import copy

import numpy as np
import cv2 as cv


def remap_point(point, coords):
    point = np.array(point)

    # distance from point
    distance = np.amax(np.abs(point - coords), axis=-1)

    # take the pixel with the lowest distance
    return np.unravel_index(np.argmin(distance), distance.shape)


class BaseBoundingBox:
    def __init__(self, dtype=np.int32):
        self.dtype = dtype
        self.points = np.array([], self.dtype)

    def __getitem__(self, index: int):
        return self.points[index]

    def to_simple(self):
        return self

    def to_quad(self):
        return self

    def astype(self, dtype):
        bbox = copy.deepcopy(self)
        bbox.dtype = dtype
        bbox.points = self.points.astype(dtype)
        return bbox

    def relative(self, size, dtype=np.float32):
        bbox = copy.deepcopy(self)
        bbox.dtype = dtype
        bbox.points = bbox.points.astype(bbox.dtype) / np.array(size, dtype)[None]
        return bbox

    def absolute(self, size, dtype=np.int32):
        bbox = copy.deepcopy(self)
        bbox.dtype = dtype
        bbox.points = bbox.points.astype(bbox.dtype) * np.array(size, dtype)[None]
        return bbox

    def get_size(self):
        simple = self.to_simple()
        return simple.get_size()

    def draw(self, img):
        return self.to_quad().draw(img)


class SimpleBoundingBox(BaseBoundingBox):
    def __init__(self, p0: Tuple[int, int], p1: Tuple[int, int], dtype=np.int32):
        super().__init__(dtype)
        self.points = np.array((p0, p1), self.dtype)

    def xxyy(self):
        x0, y0, x1, y1 = self.points.ravel()
        return (x0, x1, y0, y1)

    def to_quad(self):
        x0, y0, x1, y1 = self.points.ravel()
        return QuadBoundingBox((x0, y0), (x1, y0), (x1, y1), (x0, y1), self.dtype)

    def get_size(self):
        return np.array(self.points[1] - self.points[0])


class QuadBoundingBox(BaseBoundingBox):
    def __init__(self, p0, p1, p2, p3, dtype=np.int32):
        super().__init__(dtype)
        self.points = np.array((p0, p1, p2, p3), self.dtype)

    def to_simple(self):
        return SimpleBoundingBox(
            np.amin(self.points, axis=0), np.amax(self.points, axis=0), self.dtype
        )

    def remap(self, coords):
        bbox = copy.deepcopy(self)
        for i in range(4):
            bbox.points[i] = remap_point(self.points[i][::-1], coords)
        return bbox

    def draw(self, img):
        points = self.points[:, ::-1]
        img = cv.polylines(img, (points,), True, (255, 0, 255), 1, cv.LINE_AA)
        for i in range(4):
            img = cv.circle(img, points[i], 4, (255, 0, 255), -1)
        return img
