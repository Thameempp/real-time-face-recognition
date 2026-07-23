import queue
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from detector import YoloFaceRecognizer
from model_utils import create_embedder_app


def put_latest(target_queue, value):
    try:
        target_queue.put_nowait(value)
        return
    except queue.Full:
        pass

    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        target_queue.put_nowait(value)
    except queue.Full:
        pass


def read_latest_frame(
    frame_buffer,
    frame_id_value,
    frame_lock,
    last_seen_frame_id,
):
    with frame_lock:
        frame_id = frame_id_value.value

        if frame_id == last_seen_frame_id:
            return None, last_seen_frame_id

        frame = frame_buffer.copy()

    return (
        {
            "frame_id": frame_id,
            "frame": frame,
        },
        frame_id,
    )


def recognition_worker(
    shared_memory_name,
    frame_shape,
    frame_dtype,
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
):
    cv2.setUseOptimized(True)

    shared_frame = shared_memory.SharedMemory(name=shared_memory_name)
    frame_buffer = np.ndarray(
        frame_shape,
        dtype=np.dtype(frame_dtype),
        buffer=shared_frame.buf,
    )

    detector = YoloFaceRecognizer(
        yolo_model_path,
        create_embedder_app(),
        known_embeddings,
        known_names,
        yolo_imgsz,
        yolo_conf,
        yolo_device,
    )

    processed_count = 0
    started_at = time.perf_counter()
    last_seen_frame_id = -1

    try:
        while not stop_event.is_set():
            frame_item, last_seen_frame_id = read_latest_frame(
                frame_buffer,
                frame_id_value,
                frame_lock,
                last_seen_frame_id,
            )

            if frame_item is None:
                time.sleep(0.001)
                continue

            frame = frame_item["frame"]
            frame_id = frame_item["frame_id"]
            started_frame_at = time.perf_counter()
            detections = detector.detect_and_recognize(frame)

            processed_count += 1
            now = time.perf_counter()
            recognition_fps = 0.0

            if now - started_at >= 1.0:
                recognition_fps = processed_count / (now - started_at)
                processed_count = 0
                started_at = now

            put_latest(
                result_queue,
                {
                    "frame_id": frame_id,
                    "faces": detections,
                    "recognition_fps": recognition_fps,
                    "inference_ms": (now - started_frame_at) * 1000.0,
                    "timestamp": now,
                },
            )
    finally:
        shared_frame.close()
