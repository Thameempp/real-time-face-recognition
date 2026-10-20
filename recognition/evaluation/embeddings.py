"""Load InsightFace once and prepare validated embeddings for evaluation."""

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import (
    EMBEDDER_DETECTION_SIZE,
    EMBEDDER_DETECTION_THRESHOLD,
    MIN_FACE_SIZE,
    MODEL_NAME,
)
from recognition.recognizer import (
    assess_face_quality,
    get_best_face,
    normalize_embedding,
)
from runtime_utils import get_execution_providers


def create_evaluation_app():
    app = FaceAnalysis(name=MODEL_NAME, providers=get_execution_providers())
    app.prepare(
        ctx_id=-1,
        det_size=EMBEDDER_DETECTION_SIZE,
        det_thresh=EMBEDDER_DETECTION_THRESHOLD,
    )
    return app


def get_image_embedding(app, image_path, min_face_size=MIN_FACE_SIZE):
    image = cv2.imread(image_path)
    if image is None:
        return None

    face = get_best_face(app.get(image))
    if face is None:
        return None

    quality = assess_face_quality(
        image,
        face.bbox,
        min_face_size=min_face_size,
    )
    if not quality["accepted"]:
        return None

    embedding = normalize_embedding(face.embedding)
    if not np.all(np.isfinite(embedding)) or not np.isclose(
        np.linalg.norm(embedding),
        1.0,
        atol=1e-3,
    ):
        return None

    return embedding
