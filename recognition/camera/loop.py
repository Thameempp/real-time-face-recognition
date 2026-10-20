"""Display-loop state and result freshness handling for live recognition."""

import queue
import time

import cv2
import numpy as np

from config import DISPLAY_FPS, MAX_RESULT_AGE_SECONDS
from recognition.drawing import draw_detection, draw_overlay


def run_camera_loop(
    camera,
    first_frame,
    frame_buffer,
    frame_id_value,
    frame_timestamp_value,
    frame_lock,
    result_queue,
    worker,
    known_people_count,
    yolo_device,
):
    frame_id = 0
    pending_first_frame = first_frame
    latest_faces = []
    recognition_fps = 0.0
    inference_ms = 0.0
    stage_ms = {}
    result_frame_timestamp = 0.0
    queue_latency_ms = 0.0
    skipped_frames = 0
    inference_errors = 0
    worker_error = None
    display_fps = 0.0
    display_count = 0
    display_started_at = time.perf_counter()

    while True:
        frame_started_at = time.perf_counter()

        if pending_first_frame is not None:
            frame = pending_first_frame
            pending_first_frame = None
            success = True
        else:
            success, frame = camera.read()

        if not success:
            print("Could not read webcam frame")
            break

        frame_captured_at = time.perf_counter()
        frame_id += 1
        now = time.perf_counter()
        display_count += 1

        if now - display_started_at >= 1.0:
            display_fps = display_count / (now - display_started_at)
            display_count = 0
            display_started_at = now

        with frame_lock:
            np.copyto(frame_buffer, frame)
            frame_id_value.value = frame_id
            frame_timestamp_value.value = frame_captured_at

        while True:
            try:
                update = result_queue.get_nowait()
            except queue.Empty:
                break

            update_age = time.perf_counter() - update["frame_timestamp"]
            if update_age > MAX_RESULT_AGE_SECONDS:
                latest_faces = []
                worker_error = "Stale recognition result discarded"
                continue

            latest_faces = update["faces"]
            result_frame_timestamp = update["frame_timestamp"]
            inference_ms = update["inference_ms"]
            stage_ms = update["stage_ms"]
            skipped_frames = update["skipped_frames"]
            inference_errors = update["inference_errors"]
            worker_error = update["error"]
            queue_latency_ms = max(
                0.0,
                (time.perf_counter() - update["publish_started_at"]) * 1000.0,
            )

            if update["recognition_fps"] > 0:
                recognition_fps = update["recognition_fps"]

        result_age_seconds = (
            time.perf_counter() - result_frame_timestamp
            if result_frame_timestamp > 0
            else 0.0
        )
        if result_age_seconds > MAX_RESULT_AGE_SECONDS:
            latest_faces = []

        for detection in latest_faces:
            draw_detection(frame, detection)

        result_age_ms = result_age_seconds * 1000.0
        if not worker.is_alive() and worker_error is None:
            worker_error = "Recognition worker stopped"

        draw_overlay(
            frame,
            known_people_count,
            len(latest_faces),
            display_fps,
            recognition_fps,
            inference_ms,
            result_age_ms,
            yolo_device,
            stage_ms,
            queue_latency_ms,
            skipped_frames,
            inference_errors,
            worker_error,
        )

        cv2.imshow("YOLO Face Recognition", frame)
        wait_ms = 1
        if DISPLAY_FPS is not None:
            frame_interval = 1.0 / DISPLAY_FPS
            elapsed = time.perf_counter() - frame_started_at
            remaining_ms = max(0.0, frame_interval - elapsed) * 1000
            wait_ms = max(1, int(remaining_ms))

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            break
