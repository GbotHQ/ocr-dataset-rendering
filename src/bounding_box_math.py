from typing import Tuple

import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw

from bounding_box import SimpleBoundingBox, QuadBoundingBox


def bbox_from_mask(
    img: np.ndarray, background_color: int, tolerance: int = 12
) -> SimpleBoundingBox:
    mask = np.abs(img.astype(np.int32) - background_color) > tolerance
    mask = (mask.any(0), mask.any(1))

    bbox = [(np.argmax(k), k.size - np.argmax(k[::-1])) for k in mask]
    bbox = np.swapaxes(np.array(bbox, np.int32), 0, 1)
    bbox = SimpleBoundingBox(bbox[0], bbox[1], bbox.dtype)

    bbox.points = bbox.points[:, ::-1]
    return bbox


def remap_value(value, low1, high1, low2, high2):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)


def perspective_transform_points(array: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv.perspectiveTransform(array.reshape(-1, 1, 2), matrix).reshape(-1, 2)


def transform_quad_to_fit_points(quad, points):
    reference_rect = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    quad = quad.astype(np.float32)
    points = points.astype(np.float32)

    # Compute the perspective transform from the quad to a rectangle
    transform_matrix = cv.getPerspectiveTransform(quad, reference_rect)
    transform_matrix_inverse = cv.getPerspectiveTransform(reference_rect, quad)

    # Transform using the matrix
    quad = perspective_transform_points(quad, transform_matrix)
    points = perspective_transform_points(points, transform_matrix)

    quad = quad.astype(np.float32)
    points = points.astype(np.float32)

    quad_transformed_min = np.amin(quad, axis=0)
    quad_transformed_max = np.amax(quad, axis=0)
    points_transformed_min = np.amin(points, axis=0)
    points_transformed_max = np.amax(points, axis=0)

    quad = remap_value(
        quad,
        quad_transformed_min[None],
        quad_transformed_max[None],
        points_transformed_min[None],
        points_transformed_max[None],
    )

    return perspective_transform_points(quad, transform_matrix_inverse)


def get_tight_character_bbox(char, font):
    mask = font.getmask(char)
    mask = np.array(mask, dtype=np.uint8).reshape(mask.size[::-1])

    return mask, bbox_from_mask(mask, 0, 0)


def get_img_rotation_matrix(shape, angle):
    res = np.array(shape[:2], np.float32)[::-1]
    half = (res / 2).astype(np.int32)

    rotation_matrix = cv.getRotationMatrix2D(half.tolist(), angle, 1.0)

    cos, sin = np.abs(rotation_matrix[0, :2])
    new_res = (res * cos + res[::-1] * sin).astype(np.int32)
    rotation_matrix[:2, 2] += (new_res / 2) - half

    return rotation_matrix, new_res


def img_bbox_rotate(bbox, shape, angle):
    bbox = bbox.copy()

    rotation_matrix, _ = get_img_rotation_matrix(shape, angle)

    points = bbox.points[:, ::-1]
    points = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
    points = np.dot(points, rotation_matrix.T)

    bbox.points = points.astype(np.int32)[:, ::-1]

    return bbox


def bbox_union(bbox1, bbox2):
    bbox1.points[0] = np.minimum(bbox1.points[0], bbox2.points[0])
    bbox1.points[1] = np.maximum(bbox1.points[1], bbox2.points[1])

    return bbox1


def calculate_overall_bbox(line_bboxes):
    overall_bbox = line_bboxes[0].copy()
    for bbox in line_bboxes:
        overall_bbox = bbox_union(overall_bbox, bbox)
    return overall_bbox


def calculate_line_bboxes(char_bboxes):
    line_bboxes = []
    for k in char_bboxes:
        if len(line_bboxes) < k["line_index"] + 1:
            line_bboxes.append(k["bbox"].copy())

        line_bbox = line_bboxes[k["line_index"]]

        line_bbox = bbox_union(line_bbox, k["bbox"])

        line_bboxes[k["line_index"]] = line_bbox

    return line_bboxes


def calculate_char_bboxes(xy, text, font):
    draw = ImageDraw.Draw(Image.new("RGB", (0, 0)))

    char_bboxes = []
    for i, char in enumerate(text):
        char_bbox = draw.textbbox(xy, text[i], font=font)
        width = char_bbox[2] - char_bbox[0]
        height = char_bbox[3] - char_bbox[1]

        if width == 0 or height == 0:
            continue

        mask, offset_bbox = get_tight_character_bbox(char, font)
        offset_bbox_size = offset_bbox.get_size()

        line_index = text[: i + 1].count("\n")

        char_bbox = draw.textbbox(xy, "\n" * line_index + text[i], font=font)
        bottom = char_bbox[3]

        char_bbox = draw.textbbox(xy, text[: i + 1].split("\n")[-1], font=font)
        right = char_bbox[2]

        bottom -= mask.shape[0] - offset_bbox.points[1][0]
        right -= mask.shape[1] - offset_bbox.points[1][1]
        top = bottom - offset_bbox_size[0]
        left = right - offset_bbox_size[1]

        char_bboxes.append(
            {
                "bbox": SimpleBoundingBox((top, left), (bottom, right)),
                "mask": mask[
                    offset_bbox.points[0, 0] : offset_bbox.points[1, 0],
                    offset_bbox.points[0, 1] : offset_bbox.points[1, 1],
                ],
                "char_index": i,
                "line_index": line_index,
            }
        )
    return char_bboxes


def calculate_precise_bbox(
    document_bbox: QuadBoundingBox,
    src_shape: Tuple[int, int],
    mask_warped: np.ndarray,
    coords_relative: np.ndarray,
) -> Tuple[QuadBoundingBox, QuadBoundingBox]:
    bbox = document_bbox
    bbox = bbox.relative(src_shape[:2])
    bbox = bbox.remap(coords_relative[..., ::-1])
    bbox.points = bbox.points.astype(np.int32)
    points = np.stack(np.where(mask_warped), -1)

    bbox.points = transform_quad_to_fit_points(bbox.points, points)
    bbox.points = np.rint(bbox.points).astype(np.int32)

    bbox_precise = bbox
    bbox_precise_relative = bbox.relative(coords_relative.shape[:2][::-1])

    return bbox_precise, bbox_precise_relative


def bbox_from_binary_mask(mask: np.ndarray) -> SimpleBoundingBox:
    mask = (mask.any(0), mask.any(1))

    bbox = [(np.argmax(k), k.size - np.argmax(k[::-1])) for k in mask]
    bbox = np.swapaxes(np.array(bbox, np.int32), 0, 1)
    bbox = SimpleBoundingBox(bbox[0], bbox[1], bbox.dtype)

    bbox.points = bbox.points[:, ::-1]
    return bbox


def perspective_transform_bboxes(bboxes, transform_matrix):
    points = [k.to_quad().points for k in bboxes]
    points = np.concatenate(points).astype(np.float32)

    points = points[:, ::-1]
    points = perspective_transform_points(points, transform_matrix)
    points = np.rint(points[:, ::-1]).astype(np.int32).reshape(-1, 4, 2)

    return [QuadBoundingBox(k[0], k[1], k[2], k[3]) for k in points]
