import time
from multiprocessing import shared_memory

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from .config import (
    LIVE_DETECTION_SIZE,
    LIVE_DETECTION_THRESHOLD,
    MODEL_NAME,
)
from .face_processing import face_to_result, resize_for_inference
from .utils import put_latest


def get_execution_providers():
    try:
        import onnxruntime as ort
    except ImportError:
        return ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    preferred = [
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    providers = [
        provider
        for provider in preferred
        if provider in available
    ]

    return providers or ["CPUExecutionProvider"]


def create_app():
    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=get_execution_providers(),
    )

    app.prepare(
        ctx_id=-1,
        det_size=LIVE_DETECTION_SIZE,
        det_thresh=LIVE_DETECTION_THRESHOLD,
    )

    return app


def detection_worker(
    shared_memory_name,
    frame_shape,
    frame_dtype,
    frame_id_value,
    frame_lock,
    result_queue,
    stop_event,
):
    cv2.setUseOptimized(True)

    shared_frame = shared_memory.SharedMemory(name=shared_memory_name)
    frame_buffer = np.ndarray(
        frame_shape,
        dtype=np.dtype(frame_dtype),
        buffer=shared_frame.buf,
    )

    app = create_app()
    processed_count = 0
    last_seen_frame_id = -1
    started_at = time.perf_counter()

    try:
        while not stop_event.is_set():
            with frame_lock:
                frame_id = frame_id_value.value

                if frame_id == last_seen_frame_id:
                    frame = None
                else:
                    frame = frame_buffer.copy()
                    last_seen_frame_id = frame_id

            if frame is None:
                time.sleep(0.001)
                continue

            inference_frame, scale_x, scale_y = resize_for_inference(frame)
            faces = app.get(inference_frame)
            results = [
                face_to_result(face, scale_x, scale_y, frame_id)
                for face in faces
            ]

            processed_count += 1
            now = time.perf_counter()
            detection_fps = 0.0

            if now - started_at >= 1.0:
                detection_fps = processed_count / (now - started_at)
                processed_count = 0
                started_at = now

            put_latest(
                result_queue,
                {
                    "frame_id": frame_id,
                    "faces": results,
                    "detection_fps": detection_fps,
                    "timestamp": now,
                },
            )
    finally:
        shared_frame.close()
