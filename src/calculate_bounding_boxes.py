from typing import Union, Tuple
from pathlib import Path as pth

import numpy as np
import cv2 as cv
from scipy.ndimage import map_coordinates
from PIL import ImageFont

from bounding_box import SimpleBoundingBox
from bounding_box_math import (
    calculate_char_bboxes,
    calculate_line_bboxes,
    calculate_overall_bbox,
    img_bbox_rotate,
    get_tight_character_bbox,
    bbox_from_binary_mask,
    perspective_transform_bboxes,
    calculate_precise_bbox,
)


def to_uint(img: np.ndarray, dtype=np.uint8):
    return (np.clip(img, 0, 1) * np.iinfo(dtype).max).astype(dtype)


def to_float(img: np.ndarray, fdtype=np.float32):
    return img.astype(fdtype) / np.iinfo(img.dtype).max


def imread_coords(path: Union[str, pth], src_shape: Tuple[int, int]):
    # unchanged to read as uint16
    coords_relative = to_float(cv.imread(path, cv.IMREAD_UNCHANGED))
    alpha = coords_relative[..., 0, None]
    # flip y to match opencv coordinates
    coords_relative[..., 1] = 1 - coords_relative[..., 1]
    coords_relative = np.where(alpha < 1, -1, coords_relative[..., 1:])

    coords_absolute = np.moveaxis(coords_relative.copy(), -1, 0)
    coords_absolute *= np.array(
        (src_shape[0] - 1, src_shape[1] - 1), coords_absolute.dtype
    )[:, None, None]

    return coords_relative, coords_absolute, alpha


def calculate_document_bboxes(sample, font):
    char_bboxes = calculate_char_bboxes(sample.anchor, sample.text, font)

    for k in char_bboxes:
        left_offset = sample.line_offsets[k["line_index"]]
        points = k["bbox"].points.astype(np.float32)
        points[:, 1] += left_offset
        k["bbox"].points = points.astype(np.int32)

    overall_bbox = calculate_overall_bbox(calculate_line_bboxes(char_bboxes))

    shape = sample.resolution_before_rotation[::-1]

    for k in char_bboxes:
        k["bbox"] = img_bbox_rotate(
            k["bbox"].to_quad(), shape, sample.text_rotation_angle
        )

    overall_bbox = img_bbox_rotate(
        overall_bbox.to_quad(), shape, sample.text_rotation_angle
    )

    padding_array = np.array(sample.padding[:2], np.int32)[None]
    for k in char_bboxes:
        k["bbox"].points += padding_array

    overall_bbox.points += padding_array

    return char_bboxes, overall_bbox


def calculate_char_labels(sample, document_img_shape, char_bboxes, font):
    mask_combined = np.zeros(document_img_shape[:2], dtype=np.int32)

    for i, k in enumerate(char_bboxes):
        bbox = k["bbox"]
        char_index = k["char_index"]

        mask, mask_bbox = get_tight_character_bbox(sample.text[char_index], font)
        mask = mask[
            mask_bbox.points[0, 0] : mask_bbox.points[1, 0],
            mask_bbox.points[0, 1] : mask_bbox.points[1, 1],
        ]

        h, w = np.array(mask.shape[:2]) - 1
        reference_rect = np.array([[0, 0], [h, 0], [h, w], [0, w]], dtype=np.float32)
        quad = bbox.points.astype(np.float32)

        transform_matrix = cv.getPerspectiveTransform(
            reference_rect[:, ::-1], quad[:, ::-1]
        )
        mask = cv.warpPerspective(mask, transform_matrix, document_img_shape[:2][::-1])

        mask_combined[mask > 64] = i + 1

    return mask_combined


def calculate_render_bboxes(
    document_img_shape,
    labels,
    document_char_bboxes,
    document_overall_bbox,
    coords_relative,
    coords_absolute,
    resolution,
):
    labels_warped = map_coordinates(
        labels, np.rint(coords_absolute).astype(np.int32), cval=0
    )

    overall_bbox_remapped, _ = calculate_precise_bbox(
        document_overall_bbox, document_img_shape, labels_warped != 0, coords_relative
    )

    w, h = resolution
    reference_rect = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32) * (
        h - 1,
        w - 1,
    )
    reference_rect = reference_rect.astype(np.float32)[:, ::-1]
    quad = overall_bbox_remapped.points.astype(np.float32)[:, ::-1]

    transform_matrix = cv.getPerspectiveTransform(quad, reference_rect)
    transform_matrix_inverse = cv.getPerspectiveTransform(reference_rect, quad)
    labels_warped = cv.warpPerspective(
        labels_warped.astype(np.float32),
        transform_matrix,
        (w, h),
        flags=cv.INTER_NEAREST,
    )

    new_char_bboxes = []
    for i, k in enumerate(document_char_bboxes):
        mask = labels_warped == i + 1
        bbox = bbox_from_binary_mask(mask)
        bbox.points[1] -= 1
        k = k.copy()
        k["bbox"] = bbox
        new_char_bboxes.append(k)

    new_line_bboxes = calculate_line_bboxes(new_char_bboxes)

    # calculate overall bbox
    new_overall_bbox = np.concatenate([k["bbox"].points for k in new_char_bboxes], 0)
    new_overall_bbox = np.amin(new_overall_bbox, 0), np.amax(new_overall_bbox, 0)
    new_overall_bbox = SimpleBoundingBox(*new_overall_bbox).to_quad()

    # transform bboxes back
    new_overall_bbox = perspective_transform_bboxes(
        [new_overall_bbox], transform_matrix_inverse
    )[0]
    new_char_bboxes = perspective_transform_bboxes(
        [k["bbox"] for k in new_char_bboxes], transform_matrix_inverse
    )
    new_line_bboxes = perspective_transform_bboxes(
        new_line_bboxes, transform_matrix_inverse
    )

    return new_char_bboxes, new_line_bboxes, new_overall_bbox


def bboxes_to_dict(
    text,
    output_img_shape,
    new_overall_bbox,
    new_line_bboxes,
    new_char_bboxes,
    original_char_bboxes,
):
    # copy data
    new_overall_bbox = new_overall_bbox.copy()
    new_line_bboxes = [k.copy() for k in new_line_bboxes]
    new_char_bboxes = [k.copy() for k in new_char_bboxes]
    original_char_bboxes = original_char_bboxes.copy()
    for k in original_char_bboxes:
        k["bbox"] = k["bbox"].copy()

    res = np.array(output_img_shape[:2], np.int32)[None]
    for bbox in new_char_bboxes:
        bbox.points = (bbox.points / res)[:, ::-1]
    for bbox in new_line_bboxes:
        bbox.points = (bbox.points / res)[:, ::-1]
    new_overall_bbox.points = (new_overall_bbox.points / res)[:, ::-1]

    def make_tl_tr_br_bl(points):
        return np.stack((points[0], points[3], points[2], points[1]), 0)

    for bbox in new_char_bboxes:
        bbox.points = make_tl_tr_br_bl(bbox.points)
    for bbox in new_line_bboxes:
        bbox.points = make_tl_tr_br_bl(bbox.points)
    new_overall_bbox.points = make_tl_tr_br_bl(new_overall_bbox.points)

    axis_aligned_overall_bbox = np.stack(
        (np.amin(new_overall_bbox.points, 0), np.amax(new_overall_bbox.points, 0)), 0
    )
    axis_aligned_overall_bbox_xxyy = [
        axis_aligned_overall_bbox[0, 0],
        axis_aligned_overall_bbox[1, 0],
        axis_aligned_overall_bbox[0, 1],
        axis_aligned_overall_bbox[1, 1],
    ]

    lines = text.split("\n")
    bounding_boxes = {
        "overall_bbox": new_overall_bbox.points.tolist(),
        "axis_aligned_overall_bbox": axis_aligned_overall_bbox.tolist(),
        "axis_aligned_overall_bbox_xxyy": axis_aligned_overall_bbox_xxyy,
        "lines": {},
    }
    
    for i in {k["line_index"] for k in original_char_bboxes}:
        line_char_bboxes = [
            (k[0]["char_index"], k[1].points.tolist())
            for k in zip(original_char_bboxes, new_char_bboxes)
            if k[0]["line_index"] == i
        ]
        chars = {
            text[k[0]]: {"char_index": k[0], "char_bbox": k[1]}
            for k in line_char_bboxes
        }

        line = lines[i]
        bounding_boxes["lines"][line] = {
            "line_bbox": new_line_bboxes[i].points.tolist(),
            "chars": chars,
        }

    return bounding_boxes


def calculate_bounding_boxes(sample):
    document_img = cv.imread(str(sample.image_path))
    document_res = document_img.shape[:2]
    font = ImageFont.truetype(str(sample.font_path), sample.font_size)

    coords_relative, coords_absolute, _ = imread_coords(
        str(sample.output_coordinates_path), document_res
    )

    document_char_bboxes, document_overall_bbox = calculate_document_bboxes(
        sample, font
    )
    mask_combined = calculate_char_labels(
        sample, document_res, document_char_bboxes, font
    )
    new_char_bboxes, new_line_bboxes, new_overall_bbox = calculate_render_bboxes(
        document_res,
        mask_combined,
        document_char_bboxes,
        document_overall_bbox,
        coords_relative,
        coords_absolute,
        sample.output_image_resolution,
    )

    bounding_boxes = bboxes_to_dict(
        sample.text,
        coords_absolute.shape[1:],
        new_overall_bbox,
        new_line_bboxes,
        new_char_bboxes,
        document_char_bboxes,
    )

    return bounding_boxes, new_char_bboxes, new_line_bboxes, new_overall_bbox
