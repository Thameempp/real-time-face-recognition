"""Draw one detected face and its identity label."""

import cv2
import numpy as np


def draw_detection(frame, detection):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(
        detection["bbox"],
        dtype=np.float32,
    ).astype(int)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width - 1, x2)
    y2 = min(frame_height - 1, y2)

    if detection["name"] == "Unknown":
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    name = detection["name"].title()
    quality = detection.get("quality", {})
    if name == "Unknown" and not quality.get("accepted", False):
        name = f"Rejected: {quality.get('reason', 'quality')}"

    label = (
        f"#{detection.get('track_id', 0)} {name} "
        f"{detection['score']:.2f} yolo:{detection['det_conf']:.2f}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    label_width = text_size[0] + 16
    label_height = text_size[1] + 16
    label_y1 = max(0, y1 - label_height)

    cv2.rectangle(
        frame,
        (x1, label_y1),
        (min(frame_width - 1, x1 + label_width), y1),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 8, y1 - 7),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )
