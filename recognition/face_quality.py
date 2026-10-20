import cv2
import numpy as np

from config import (
    FACE_BLUR_THRESHOLD,
    FACE_FRAME_EDGE_MARGIN,
    FACE_MAX_OUTSIDE_RATIO,
    FACE_PADDING_MIN,
    FACE_PADDING_RATIO,
    MAX_FACE_ASPECT_RATIO,
    MIN_FACE_ASPECT_RATIO,
    MIN_FACE_SIZE,
)


def get_best_face(faces):
    if not faces:
        return None

    return max(
        faces,
        key=lambda face: (
            face.bbox[2] - face.bbox[0]
        ) * (
            face.bbox[3] - face.bbox[1]
        ),
    )


def assess_face_quality(
    frame,
    box,
    min_face_size=MIN_FACE_SIZE,
    blur_threshold=FACE_BLUR_THRESHOLD,
):
    """Reject malformed, clipped, small, or blurry face boxes."""
    height, width = frame.shape[:2]
    coordinates = np.asarray(box, dtype=np.float32)
    quality = {
        "accepted": False,
        "reason": "invalid_box",
        "blur": 0.0,
        "face_size": 0,
    }

    if coordinates.shape != (4,) or not np.all(np.isfinite(coordinates)):
        return quality

    raw_x1, raw_y1, raw_x2, raw_y2 = coordinates.tolist()
    raw_width = raw_x2 - raw_x1
    raw_height = raw_y2 - raw_y1

    if raw_width <= 0 or raw_height <= 0:
        return quality

    raw_area = raw_width * raw_height
    clipped_x1 = max(0.0, raw_x1)
    clipped_y1 = max(0.0, raw_y1)
    clipped_x2 = min(float(width), raw_x2)
    clipped_y2 = min(float(height), raw_y2)
    clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(
        0.0,
        clipped_y2 - clipped_y1,
    )
    outside_ratio = 1.0 - clipped_area / raw_area

    if outside_ratio > FACE_MAX_OUTSIDE_RATIO:
        quality["reason"] = "severely_cropped"
        return quality

    x1, y1, x2, y2 = map(
        int,
        (clipped_x1, clipped_y1, clipped_x2, clipped_y2),
    )
    face_width = x2 - x1
    face_height = y2 - y1
    face_size = min(face_width, face_height)
    quality["face_size"] = face_size

    if face_size < min_face_size:
        quality["reason"] = "too_small"
        return quality

    aspect_ratio = face_width / max(face_height, 1)
    if not MIN_FACE_ASPECT_RATIO <= aspect_ratio <= MAX_FACE_ASPECT_RATIO:
        quality["reason"] = "severely_cropped"
        return quality

    if (
        x1 <= FACE_FRAME_EDGE_MARGIN
        or y1 <= FACE_FRAME_EDGE_MARGIN
        or width - x2 <= FACE_FRAME_EDGE_MARGIN
        or height - y2 <= FACE_FRAME_EDGE_MARGIN
    ):
        quality["reason"] = "frame_edge"
        return quality

    face_region = frame[y1:y2, x1:x2]
    if face_region.size == 0:
        quality["reason"] = "empty_crop"
        return quality

    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    quality["blur"] = blur_score

    if blur_score < blur_threshold:
        quality["reason"] = "blurry"
        return quality

    quality.update(
        {
            "accepted": True,
            "reason": "ok",
            "bbox": [x1, y1, x2, y2],
        }
    )
    return quality


def prepare_face_crop(frame, box):
    """Validate a YOLO box and return a padded, unscaled face crop."""
    height, width = frame.shape[:2]
    quality = assess_face_quality(frame, box)
    if not quality["accepted"]:
        return None, quality

    x1, y1, x2, y2 = quality["bbox"]
    face_width = x2 - x1
    face_height = y2 - y1
    padding_x = max(FACE_PADDING_MIN, int(face_width * FACE_PADDING_RATIO))
    padding_y = max(FACE_PADDING_MIN, int(face_height * FACE_PADDING_RATIO))
    padded_x1 = max(0, x1 - padding_x)
    padded_y1 = max(0, y1 - padding_y)
    padded_x2 = min(width, x2 + padding_x)
    padded_y2 = min(height, y2 + padding_y)

    return frame[padded_y1:padded_y2, padded_x1:padded_x2], quality
