"""Webcam display loop and keyboard handling for dataset collection."""

import queue
import time

import cv2
import numpy as np

from .capture import remember_frame, save_current_face
from train.config import POSE_KEYS, SAVE_COOLDOWN_SECONDS
from train.dataset import find_next_count
from train.display import (
    draw_face_result,
    draw_overlay,
    get_status_text,
    print_controls,
)


def run_collection_loop(
    camera,
    first_frame,
    frame_buffer,
    frame_id_value,
    frame_lock,
    result_queue,
    folders,
    person_name,
):
    image_count = find_next_count(folders["originals"])
    current_pose = "front"
    last_saved_time = 0.0
    frame_id = 0
    recent_frames = {}
    latest_faces = []
    latest_detection_frame_id = None
    latest_result_timestamp = 0.0
    detection_fps = 0.0
    display_fps = 0.0
    display_count = 0
    display_started_at = time.perf_counter()
    pending_first_frame = first_frame

    print_controls()
    while True:
        if pending_first_frame is not None:
            frame = pending_first_frame
            pending_first_frame = None
            success = True
        else:
            success, frame = camera.read()

        if not success:
            print("Could not read webcam frame")
            break

        frame_id += 1
        display_frame = frame.copy()
        remember_frame(recent_frames, frame_id, frame.copy())

        with frame_lock:
            np.copyto(frame_buffer, frame)
            frame_id_value.value = frame_id

        now = time.perf_counter()
        display_count += 1
        if now - display_started_at >= 1.0:
            display_fps = display_count / (now - display_started_at)
            display_count = 0
            display_started_at = now

        while True:
            try:
                update = result_queue.get_nowait()
            except queue.Empty:
                break

            latest_faces = update["faces"]
            latest_detection_frame_id = update["frame_id"]
            latest_result_timestamp = update["timestamp"]
            if update["detection_fps"] > 0:
                detection_fps = update["detection_fps"]

        for face in latest_faces:
            draw_face_result(display_frame, face)

        status_text, status_color = get_status_text(latest_faces)
        result_age_ms = 0.0
        if latest_result_timestamp > 0:
            result_age_ms = (
                time.perf_counter() - latest_result_timestamp
            ) * 1000.0

        draw_overlay(
            display_frame,
            person_name,
            current_pose,
            image_count,
            latest_faces,
            status_text,
            status_color,
            display_fps,
            detection_fps,
            result_age_ms,
        )

        cv2.imshow("Face Dataset Collector", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in POSE_KEYS:
            current_pose = POSE_KEYS[key]
        elif key == ord("s"):
            current_time = time.time()
            if current_time - last_saved_time < SAVE_COOLDOWN_SECONDS:
                continue

            saved = save_current_face(
                latest_faces,
                latest_detection_frame_id,
                latest_result_timestamp,
                recent_frames,
                folders,
                person_name,
                image_count,
                current_pose,
            )
            if saved:
                image_count += 1
                last_saved_time = current_time
        elif key == ord("q"):
            break
