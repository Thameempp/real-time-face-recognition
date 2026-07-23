import cv2
import numpy as np

from .config import (
    FACE_PADDING_MIN,
    FACE_PADDING_RATIO,
    LIVE_INFERENCE_WIDTH,
    MIN_FACE_CROP_SIZE,
)


def resize_for_inference(frame):
    height, width = frame.shape[:2]

    if width <= LIVE_INFERENCE_WIDTH:
        return frame, 1.0, 1.0

    scale = LIVE_INFERENCE_WIDTH / width
    inference_height = int(height * scale)
    inference_frame = cv2.resize(
        frame,
        (LIVE_INFERENCE_WIDTH, inference_height),
        interpolation=cv2.INTER_AREA,
    )

    return (
        inference_frame,
        width / LIVE_INFERENCE_WIDTH,
        height / inference_height,
    )


def face_to_result(face, scale_x, scale_y, frame_id):
    bbox = face.bbox * np.array(
        [scale_x, scale_y, scale_x, scale_y],
        dtype=np.float32,
    )

    keypoints = None

    if face.kps is not None:
        keypoints = face.kps * np.array(
            [scale_x, scale_y],
            dtype=np.float32,
        )

    score = getattr(face, "det_score", 0.0)

    return {
        "frame_id": frame_id,
        "bbox": bbox.tolist(),
        "kps": keypoints.tolist() if keypoints is not None else None,
        "score": float(score),
    }


def crop_face(frame, bbox):
    """Crop the detected face with adaptive padding."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).astype(int)

    face_width = max(1, x2 - x1)
    face_height = max(1, y2 - y1)
    padding_x = max(FACE_PADDING_MIN, int(face_width * FACE_PADDING_RATIO))
    padding_y = max(FACE_PADDING_MIN, int(face_height * FACE_PADDING_RATIO))

    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(frame_width, x2 + padding_x)
    y2 = min(frame_height, y2 + padding_y)

    return frame[y1:y2, x1:x2]


def upscale_small_face(face_image):
    """Keep far-face crops usable for later embedding extraction."""
    height, width = face_image.shape[:2]
    min_side = min(height, width)

    if min_side <= 0 or min_side >= MIN_FACE_CROP_SIZE:
        return face_image

    scale = MIN_FACE_CROP_SIZE / min_side

    return cv2.resize(
        face_image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
