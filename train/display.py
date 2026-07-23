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
        return (
            "Face detected - Press S to save",
            (0, 255, 0),
        )

    if len(faces) == 0:
        return (
            "No face detected",
            (0, 0, 255),
        )

    return (
        "Multiple faces detected",
        (0, 165, 255),
    )


def print_controls():
    print("\nControls")
    print("1 = Front")
    print("2 = Slight left")
    print("3 = Strong left")
    print("4 = Slight right")
    print("5 = Strong right")
    print("6 = Looking up")
    print("7 = Looking down")
    print("S = Save current face crop")
    print("Q = Quit")


def draw_overlay(
    display_frame,
    person_name,
    current_pose,
    image_count,
    faces,
    status_text,
    status_color,
    display_fps,
    detection_fps,
    result_age_ms,
):
    rows = [
        (f"Person: {person_name}", 0.8, (255, 255, 255)),
        (f"Pose: {current_pose}", 0.8, (255, 255, 255)),
        (f"Saved sets: {image_count - 1}", 0.8, (255, 255, 255)),
        (status_text, 0.7, status_color),
        (f"Faces: {len(faces)}", 0.65, (255, 255, 255)),
        (f"Display FPS: {display_fps:.1f}", 0.65, (255, 255, 255)),
        (f"Detection FPS: {detection_fps:.1f}", 0.65, (255, 255, 255)),
        (f"Result age: {result_age_ms:.0f} ms", 0.65, (255, 255, 255)),
        ("1-7: Pose | S: Save | Q: Quit", 0.65, (255, 255, 255)),
    ]

    for index, (text, font_scale, color) in enumerate(rows):
        cv2.putText(
            display_frame,
            text,
            (20, 35 + index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
        )
