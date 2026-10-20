import os
import urllib.request

from insightface.app import FaceAnalysis

from config import (
    DEFAULT_YOLO_MODEL,
    DEFAULT_YOLO_MODEL_URL,
    EMBEDDER_DETECTION_SIZE,
    EMBEDDER_DETECTION_THRESHOLD,
    MODEL_NAME,
    YOLO_MODEL_FOLDER,
)
from recognition.embedding_cache import (
    load_face_recognition_module,
    load_known_embeddings,
)
from runtime_utils import get_execution_providers


def get_torch_device():
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return "mps"

    return "cpu"


def create_embedder_app():
    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=get_execution_providers(),
    )

    app.prepare(
        ctx_id=-1,
        det_size=EMBEDDER_DETECTION_SIZE,
        det_thresh=EMBEDDER_DETECTION_THRESHOLD,
    )

    return app


def ensure_yolo_model(model_path, allow_download=True):
    if os.path.exists(model_path):
        return model_path

    if model_path != DEFAULT_YOLO_MODEL:
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    if not allow_download:
        raise FileNotFoundError(f"YOLO face model not found: {model_path}")

    os.makedirs(YOLO_MODEL_FOLDER, exist_ok=True)
    temporary_path = f"{model_path}.download"

    print(f"Downloading YOLO face model to: {model_path}")

    urllib.request.urlretrieve(
        DEFAULT_YOLO_MODEL_URL,
        temporary_path,
    )
    os.replace(temporary_path, model_path)

    return model_path
