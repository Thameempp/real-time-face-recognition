"""Draw live recognition performance and freshness information."""

import cv2


def draw_overlay(
    frame,
    known_people_count,
    face_count,
    display_fps,
    recognition_fps,
    inference_ms,
    result_age_ms,
    yolo_device,
    stage_ms=None,
    queue_latency_ms=0.0,
    skipped_frames=0,
    inference_errors=0,
    worker_error=None,
):
    stage_ms = stage_ms or {}
    rows = [
        f"YOLO faces: {face_count}",
        f"Known people: {known_people_count}",
        f"Display FPS: {display_fps:.1f}",
        f"Recognition FPS: {recognition_fps:.1f}",
        f"Inference: {inference_ms:.0f} ms",
        f"Result age: {result_age_ms:.0f} ms",
        (
            "Stage ms Y/C/I/M: "
            f"{stage_ms.get('yolo_ms', 0.0):.0f}/"
            f"{stage_ms.get('crop_ms', 0.0):.0f}/"
            f"{stage_ms.get('insightface_ms', 0.0):.0f}/"
            f"{stage_ms.get('matching_ms', 0.0):.1f}"
        ),
        f"Tracking: {stage_ms.get('tracking_ms', 0.0):.1f} ms",
        (
            f"Result queue: {queue_latency_ms:.0f} ms  "
            f"Skipped frames: {skipped_frames}  "
            f"Errors: {inference_errors}"
        ),
        f"Device: {yolo_device}",
        f"Worker error: {worker_error}" if worker_error else "",
        "Press Q to quit",
    ]

    for index, text in enumerate(rows):
        if not text:
            continue
        cv2.putText(
            frame,
            text,
            (20, 35 + index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
        )
