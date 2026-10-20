"""Camera and worker lifecycle for the live recognition window."""

import multiprocessing as mp
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH
from .loop import run_camera_loop
from recognition.workers import recognition_worker


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
    matching_strategy,
    similarity_threshold,
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
    np.copyto(frame_buffer, first_frame)

    frame_id_value = mp.Value("i", 0)
    frame_timestamp_value = mp.Value("d", time.perf_counter())
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
            frame_timestamp_value,
            frame_lock,
            result_queue,
            stop_event,
            known_embeddings,
            known_names,
            yolo_model_path,
            yolo_imgsz,
            yolo_conf,
            yolo_device,
            matching_strategy,
            similarity_threshold,
        ),
    )
    worker_started = False

    try:
        worker.start()
        worker_started = True
        run_camera_loop(
            camera,
            first_frame,
            frame_buffer,
            frame_id_value,
            frame_timestamp_value,
            frame_lock,
            result_queue,
            worker,
            len(set(known_names)),
            yolo_device,
        )
    finally:
        stop_event.set()
        if worker_started:
            worker.join(timeout=2.0)

            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)

        shared_frame.close()
        shared_frame.unlink()
        result_queue.close()
        result_queue.join_thread()
        camera.release()
        cv2.destroyAllWindows()
