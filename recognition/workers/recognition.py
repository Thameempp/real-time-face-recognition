import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from recognition.detector import YoloFaceRecognizer
from recognition.model_utils import create_embedder_app
from recognition.tracking import FaceTracker
from .buffers import put_latest, read_latest_frame


def recognition_worker(
    shared_memory_name,
    frame_shape,
    frame_dtype,
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
):
    cv2.setUseOptimized(True)

    shared_frame = shared_memory.SharedMemory(name=shared_memory_name)
    frame_buffer = np.ndarray(
        frame_shape,
        dtype=np.dtype(frame_dtype),
        buffer=shared_frame.buf,
    )

    processed_count = 0
    started_at = 0.0
    last_seen_frame_id = 0
    skipped_frames = 0
    inference_errors = 0
    tracker = FaceTracker(threshold=similarity_threshold)

    try:
        detector = YoloFaceRecognizer(
            yolo_model_path,
            create_embedder_app(),
            known_embeddings,
            known_names,
            yolo_imgsz,
            yolo_conf,
            yolo_device,
            matching_strategy,
            similarity_threshold,
        )
        with frame_lock:
            warmup_frame = frame_buffer.copy()
        detector.warmup(warmup_frame)
        started_at = time.perf_counter()

        while not stop_event.is_set():
            previous_frame_id = last_seen_frame_id
            frame_item, last_seen_frame_id = read_latest_frame(
                frame_buffer,
                frame_id_value,
                frame_timestamp_value,
                frame_lock,
                last_seen_frame_id,
            )

            if frame_item is None:
                time.sleep(0.001)
                continue

            frame = frame_item["frame"]
            frame_id = frame_item["frame_id"]
            frame_timestamp = frame_item["frame_timestamp"]
            skipped_frames += max(0, frame_id - previous_frame_id - 1)
            started_frame_at = time.perf_counter()

            error = None
            try:
                detections, stage_ms = detector.detect_and_recognize(
                    frame,
                    tracker,
                    started_frame_at,
                )
            except Exception as exc:
                inference_errors += 1
                error = f"{type(exc).__name__}: {exc}"
                detections = []
                stage_ms = {
                    "yolo_ms": 0.0,
                    "crop_ms": 0.0,
                    "insightface_ms": 0.0,
                    "matching_ms": 0.0,
                    "tracking_ms": 0.0,
                }

            processed_count += 1
            now = time.perf_counter()
            recognition_fps = 0.0

            if now - started_at >= 1.0:
                recognition_fps = processed_count / (now - started_at)
                processed_count = 0
                started_at = now

            result = {
                "frame_id": frame_id,
                "frame_timestamp": frame_timestamp,
                "faces": detections,
                "recognition_fps": recognition_fps,
                "inference_ms": (now - started_frame_at) * 1000.0,
                "stage_ms": stage_ms,
                "timestamp": now,
                "publish_started_at": time.perf_counter(),
                "skipped_frames": skipped_frames,
                "inference_errors": inference_errors,
                "error": error,
            }
            put_latest(result_queue, result)
    except Exception as exc:
        put_latest(
            result_queue,
            {
                "frame_id": frame_id_value.value,
                "frame_timestamp": frame_timestamp_value.value,
                "faces": [],
                "recognition_fps": 0.0,
                "inference_ms": 0.0,
                "stage_ms": {},
                "timestamp": time.perf_counter(),
                "publish_started_at": time.perf_counter(),
                "skipped_frames": skipped_frames,
                "inference_errors": inference_errors + 1,
                "error": f"Worker startup failed: {type(exc).__name__}: {exc}",
            },
        )
    finally:
        shared_frame.close()
