import time

import numpy as np
from ultralytics import YOLO

from config import MAX_FACES
from recognition.recognizer import (
    EmbeddingMatcher,
    embed_crop,
    embed_crops,
    prepare_face_crop,
)


class YoloFaceRecognizer:
    def __init__(
        self,
        model_path,
        embedder_app,
        known_embeddings,
        known_names,
        image_size,
        confidence,
        device,
        matching_strategy,
        similarity_threshold,
    ):
        self.model = YOLO(model_path)
        self.embedder_app = embedder_app
        self.known_embeddings = known_embeddings
        self.known_names = known_names
        self.matcher = EmbeddingMatcher(
            known_embeddings,
            known_names,
            strategy=matching_strategy,
        )
        self.similarity_threshold = similarity_threshold
        self.image_size = image_size
        self.confidence = confidence
        self.device = device

    def warmup(self, frame):
        results = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            max_det=MAX_FACES,
            verbose=False,
        )
        embedder_warmed = False
        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                crop, quality = prepare_face_crop(frame, box)
                if quality["accepted"] and crop is not None:
                    embed_crop(self.embedder_app, crop)
                    embedder_warmed = True
                    break

        if not embedder_warmed:
            detector_model = self.embedder_app.det_model
            detector_model.detect(frame, max_num=0, metric="default")
            recognition_model = self.embedder_app.models.get("recognition")
            if recognition_model is not None:
                input_width, input_height = recognition_model.input_size
                empty_aligned_face = np.zeros(
                    (input_height, input_width, 3),
                    dtype=np.uint8,
                )
                recognition_model.get_feat([empty_aligned_face])

        try:
            import torch

            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            elif str(self.device) == "mps" and torch.backends.mps.is_available():
                torch.mps.synchronize()
        except (ImportError, AttributeError):
            pass

    def detect_and_recognize(self, frame, tracker, timestamp):
        stages = {
            "yolo_ms": 0.0,
            "crop_ms": 0.0,
            "insightface_ms": 0.0,
            "matching_ms": 0.0,
            "tracking_ms": 0.0,
        }
        started_at = time.perf_counter()
        yolo_results = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            max_det=MAX_FACES,
            verbose=False,
        )
        stages["yolo_ms"] = (time.perf_counter() - started_at) * 1000.0

        detections = []

        if not yolo_results:
            tracker.update(detections, timestamp)
            return detections, stages

        boxes = yolo_results[0].boxes

        if boxes is None:
            tracker.update(detections, timestamp)
            return detections, stages

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        crop_started_at = time.perf_counter()
        for box, confidence in zip(xyxy, confidences):
            crop, quality = prepare_face_crop(frame, box)
            detections.append(
                {
                    "bbox": box.tolist(),
                    "name": "Unknown",
                    "score": 0.0,
                    "det_conf": float(confidence),
                    "quality": quality,
                    "_crop": crop,
                }
            )

        stages["crop_ms"] = (time.perf_counter() - crop_started_at) * 1000.0

        tracking_started_at = time.perf_counter()
        tracker.update(detections, timestamp)
        stages["tracking_ms"] = (
            time.perf_counter() - tracking_started_at
        ) * 1000.0

        pending = [
            detection
            for detection in detections
            if tracker.needs_embedding(detection, timestamp)
        ]
        if pending:
            embedding_started_at = time.perf_counter()
            pending_embeddings = embed_crops(
                self.embedder_app,
                [detection["_crop"] for detection in pending],
            )
            stages["insightface_ms"] = (
                time.perf_counter() - embedding_started_at
            ) * 1000.0

            valid_embeddings = [
                embedding
                for embedding in pending_embeddings
                if embedding is not None
            ]
            matching_started_at = time.perf_counter()
            valid_matches = iter(
                self.matcher.match_many(
                    valid_embeddings,
                    threshold=self.similarity_threshold,
                )
            )
            for detection, embedding in zip(pending, pending_embeddings):
                name, score = (
                    next(valid_matches)
                    if embedding is not None
                    else ("Unknown", 0.0)
                )
                tracker.update_identity(detection, name, score, timestamp)
            stages["matching_ms"] = (
                time.perf_counter() - matching_started_at
            ) * 1000.0

        for detection in detections:
            detection.pop("_crop", None)

        tracker.apply_identity(detections)
        return detections, stages
