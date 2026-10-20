"""Bounding-box geometry used for lightweight face-track association."""

import numpy as np


def box_iou(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    first_area = max(0.0, first[2] - first[0]) * max(
        0.0,
        first[3] - first[1],
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0,
        second[3] - second[1],
    )
    union = first_area + second_area - intersection

    return intersection / union if union > 0 else 0.0


def _center_and_size(box):
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2]), width, height
