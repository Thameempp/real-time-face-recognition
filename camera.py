import multiprocessing as mp
import queue
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH
from drawing import draw_detection, draw_overlay
from workers import recognition_worker


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


def run_camera(
    known_embeddings,
    known_names,
    yolo_model_path,
    yolo_imgsz,
    yolo_conf,
    yolo_device,
):
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
        target=recognition_worker,
        args=(
            shared_frame.name,
            frame_shape,
            str(frame_dtype),
            frame_id_value,
            frame_lock,
            result_queue,
            stop_event,
            known_embeddings,
            known_names,
            yolo_model_path,
            yolo_imgsz,
            yolo_conf,
            yolo_device,
        ),
    )

    worker.start()

    frame_id = 0
    pending_first_frame = first_frame
    latest_faces = []
    recognition_fps = 0.0
    inference_ms = 0.0
    result_timestamp = 0.0
    display_fps = 0.0
    display_count = 0
    display_started_at = time.perf_counter()
    known_people_count = len(set(known_names))

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
            now = time.perf_counter()
            display_count += 1

            if now - display_started_at >= 1.0:
                display_fps = display_count / (now - display_started_at)
                display_count = 0
                display_started_at = now

            with frame_lock:
                np.copyto(frame_buffer, frame)
                frame_id_value.value = frame_id

            while True:
                try:
                    update = result_queue.get_nowait()
                except queue.Empty:
                    break

                latest_faces = update["faces"]
                result_timestamp = update["timestamp"]
                inference_ms = update["inference_ms"]

                if update["recognition_fps"] > 0:
                    recognition_fps = update["recognition_fps"]

            for detection in latest_faces:
                draw_detection(frame, detection)

            result_age_ms = 0.0

            if result_timestamp > 0:
                result_age_ms = (
                    time.perf_counter() - result_timestamp
                ) * 1000.0

            draw_overlay(
                frame,
                known_people_count,
                len(latest_faces),
                display_fps,
                recognition_fps,
                inference_ms,
                result_age_ms,
                yolo_device,
            )

            cv2.imshow("YOLO Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
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
