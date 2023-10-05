#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import pyclipper

from typing import Union
from shapely.geometry import Polygon


def read_image(image: Union[str, np.ndarray]) -> np.ndarray:
    # Check given image is a path or a numpy array
    if os.path.isfile(image):
        # Read Image
        image = cv2.imread(image)

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_with_aspect_ratio(image, new_width=None, new_height=None):
    # Get the current dimensions of the image
    (height, width) = image.shape[:2]

    if new_width is None and new_height is None:
        # If both new_width and new_height are None, return the original image
        return image

    if new_width is None:
        # Calculate the ratio based on the new height
        ratio = new_height / float(height)
        new_width = int(width * ratio)
    elif new_height is None:
        # Calculate the ratio based on the new width
        ratio = new_width / float(width)
        new_height = int(height * ratio)
    else:
        # Pick the best ratio that fits the image
        ratio = min(new_width / float(width), new_height / float(height))
        new_width = int(width * ratio)
        new_height = int(height * ratio)

    ratio_height = new_height / float(height)
    ratio_width = new_width / float(width)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return resized_image, ratio_height, ratio_width


def resize_with_side_limit(image: np.ndarray, max_side_len: int, multiple_of: int = None) -> np.ndarray:
    """Resize the image with the given side length limit."""
    height, width = image.shape[:2]

    if max(height, width) > max_side_len:
        ratio = max_side_len / max(height, width)
    else:
        ratio = 1.0

    new_height, new_width = int(height * ratio), int(width * ratio)

    if multiple_of is not None:
        new_height = int(np.ceil(new_height / multiple_of) * multiple_of)
        new_width = int(np.ceil(new_width / multiple_of) * multiple_of)

    ratio_height = new_height / float(height)
    ratio_width = new_width / float(width)

    image = cv2.resize(image, (new_width, new_height))

    return image, ratio_height, ratio_width


def pad_image(image: np.ndarray, min_side_len: int, pad_value: int) -> np.ndarray:
    """Pad the image to the given length."""
    height, width, c = image.shape

    new_w = max(min_side_len, width)
    new_h = max(min_side_len, height)

    padded_image = np.zeros((new_h, new_w, c), np.uint8) + pad_value
    padded_image[:height, :width, :] = image

    return padded_image


def box_score(bitmap, _box):
    '''
    box_score_fast: use bbox mean score as the mean score
    '''
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])


def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def quad_points_from_bitmap(
        pred,
        _bitmap,
        dest_width,
        dest_height,
        max_boxes,
        min_box_size,
        box_thresh,
        unclip_ratio
):
    '''
    pred: single map with shape (1, H, W),
    _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
    dest_width, dest_height: size of image before pad
    max_boxes: max boxes number
    min_box_size: min box size
    box_thresh: threshold for binarize
    unclip_ratio: unclip ratio for boxes
    '''

    bitmap = _bitmap
    height, width = bitmap.shape

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        _, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]

    num_contours = min(len(contours), max_boxes)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        points, sside = get_mini_boxes(contour)
        if sside < min_box_size:
            continue

        points = np.array(points)
        score = box_score(pred, points.reshape(-1, 2))

        if box_thresh > score:
            continue

        box = unclip(points, unclip_ratio).reshape(-1, 1, 2)

        box, sside = get_mini_boxes(box)
        if sside < min_box_size + 2:
            continue
        box = np.array(box)

        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)

        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)

        boxes.append(box.astype("int32"))
        scores.append(score)

    return np.array(boxes, dtype="int32"), scores
