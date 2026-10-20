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

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    label = (
        f"{detection['name'].title()} "
        f"{detection['score']:.2f} "
        f"yolo:{detection['det_conf']:.2f}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2

    text_size, _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

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


def draw_overlay(
    frame,
    known_people_count,
    face_count,
    display_fps,
    recognition_fps,
    inference_ms,
    result_age_ms,
    yolo_device,
):
    rows = [
        f"YOLO faces: {face_count}",
        f"Known people: {known_people_count}",
        f"Display FPS: {display_fps:.1f}",
        f"Recognition FPS: {recognition_fps:.1f}",
        f"Inference: {inference_ms:.0f} ms",
        f"Result age: {result_age_ms:.0f} ms",
        f"Device: {yolo_device}",
        "Press Q to quit",
    ]

    for index, text in enumerate(rows):
        cv2.putText(
            frame,
            text,
            (20, 35 + index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
        )
