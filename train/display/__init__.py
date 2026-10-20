"""Collector controls and status overlay."""

import cv2

from .face import draw_face_result, get_status_text

__all__ = ["draw_face_result", "draw_overlay", "get_status_text", "print_controls"]


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
