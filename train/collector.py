import multiprocessing as mp
import os
import queue
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from .config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MAX_SAVE_RESULT_AGE_SECONDS,
    POSE_KEYS,
    RECENT_FRAME_LIMIT,
    SAVE_COOLDOWN_SECONDS,
    TRAINED_DATA_FOLDER,
)
from .dataset import create_person_folders, find_next_count, save_face_forms
from .detector import detection_worker
from .display import (
    draw_face_result,
    draw_overlay,
    get_status_text,
    print_controls,
)
from .face_processing import crop_face, upscale_small_face


def open_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


def remember_frame(recent_frames, frame_id, frame):
    recent_frames[frame_id] = frame

    while len(recent_frames) > RECENT_FRAME_LIMIT:
        oldest_frame_id = min(recent_frames)
        del recent_frames[oldest_frame_id]


def save_current_face(
    latest_faces,
    latest_detection_frame_id,
    latest_result_timestamp,
    recent_frames,
    folders,
    person_name,
    image_count,
    current_pose,
):
    if len(latest_faces) == 0:
        print("No face detected. Image not saved.")
        return False

    if len(latest_faces) > 1:
        print("Multiple faces detected. Keep only one person visible.")
        return False

    result_age = time.perf_counter() - latest_result_timestamp

    if result_age > MAX_SAVE_RESULT_AGE_SECONDS:
        print("Detection result is stale. Wait for a fresh box.")
        return False

    source_frame = recent_frames.get(latest_detection_frame_id)

    if source_frame is None:
        print("Matching frame expired. Try saving again.")
        return False

    face_image = crop_face(source_frame, latest_faces[0]["bbox"])

    if face_image.size == 0:
        print("Invalid face crop")
        return False

    face_image = upscale_small_face(face_image)

    save_face_forms(
        face_image=face_image,
        folders=folders,
        person_name=person_name,
        image_count=image_count,
        pose_name=current_pose,
    )

    return True


def run_collector(person_name):
    folders = create_person_folders(person_name)
    camera = open_camera()
    success, first_frame = camera.read()

    if not success:
        camera.release()
        raise RuntimeError("Could not read initial webcam frame")

    frame_shape = first_frame.shape
    frame_dtype = first_frame.dtype
    shared_frame = shared_memory.SharedMemory(
        create=True,
        size=first_frame.nbytes,
    )
    frame_buffer = np.ndarray(
        frame_shape,
        dtype=frame_dtype,
        buffer=shared_frame.buf,
    )

    frame_id_value = mp.Value("i", 0)
    frame_lock = mp.Lock()
    result_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    worker = mp.Process(
        target=detection_worker,
        args=(
            shared_frame.name,
            frame_shape,
            str(frame_dtype),
            frame_id_value,
            frame_lock,
            result_queue,
            stop_event,
        ),
    )

    worker.start()

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

    try:
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

    finally:
        stop_event.set()
        worker.join(timeout=2.0)

        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)

        shared_frame.close()
        shared_frame.unlink()

        camera.release()
        cv2.destroyAllWindows()

    print("\nDataset collection completed.")
    print(f"Saved inside: {os.path.join(TRAINED_DATA_FOLDER, person_name)}")
