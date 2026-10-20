"""Draw detected face boxes and collector quality status."""

import cv2
import numpy as np


def draw_face_result(display_frame, face):
    frame_height, frame_width = display_frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(
        face["bbox"],
        dtype=np.float32,
    ).astype(int)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width - 1, x2)
    y2 = min(frame_height - 1, y2)

    cv2.rectangle(
        display_frame,
        (x1 + 4, y1 + 4),
        (x2 + 4, y2 + 4),
        (20, 20, 20),
        4,
    )
    cv2.rectangle(
        display_frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    keypoints = face.get("kps")
    if keypoints is not None:
        for point in keypoints:
            point_x, point_y = np.asarray(
                point,
                dtype=np.float32,
            ).astype(int)
            cv2.circle(
                display_frame,
                (point_x, point_y),
                4,
                (0, 255, 255),
                cv2.FILLED,
            )


def get_status_text(faces):
    if len(faces) == 1:
        quality = faces[0].get("quality", {})
        if not quality.get("accepted", False):
            reason = quality.get("reason", "poor quality")
            return (f"Face rejected: {reason}", (0, 0, 255))

        return "Face detected - Press S to save", (0, 255, 0)

    if len(faces) == 0:
        return "No face detected", (0, 0, 255)

    return "Multiple faces detected", (0, 165, 255)
