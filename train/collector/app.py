"""Camera, shared-memory, and detection-worker lifecycle for enrollment."""

import multiprocessing as mp
import os
from multiprocessing import shared_memory

import cv2
import numpy as np

from .loop import run_collection_loop
from .capture import remember_frame, save_current_face
from train.config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH, TRAINED_DATA_FOLDER
from train.dataset import create_person_folders, find_next_count, save_face_forms
from train.detector import detection_worker


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

    worker_started = False
    try:
        worker.start()
        worker_started = True
        run_collection_loop(
            camera,
            first_frame,
            frame_buffer,
            frame_id_value,
            frame_lock,
            result_queue,
            folders,
            person_name,
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
        camera.release()
        cv2.destroyAllWindows()

    print("\nDataset collection completed.")
    print(f"Saved inside: {os.path.join(TRAINED_DATA_FOLDER, person_name)}")
