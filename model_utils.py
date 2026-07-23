import importlib.util
import os
import urllib.request

import numpy as np
from insightface.app import FaceAnalysis

from config import (
    DEFAULT_YOLO_MODEL,
    DEFAULT_YOLO_MODEL_URL,
    EMBEDDINGS_FILE,
    EMBEDDER_DETECTION_SIZE,
    EMBEDDER_DETECTION_THRESHOLD,
    FACE_RECOGNITION_FILE,
    MODEL_NAME,
    YOLO_MODEL_FOLDER,
)


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


def load_face_recognition_module():
    spec = importlib.util.spec_from_file_location(
        "face_recognition_cache",
        FACE_RECOGNITION_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load: {FACE_RECOGNITION_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_known_embeddings(rebuild=False):
    helper = load_face_recognition_module()

    if (
        not rebuild
        and os.path.exists(EMBEDDINGS_FILE)
        and not helper.database_needs_rebuild()
    ):
        data = np.load(EMBEDDINGS_FILE, allow_pickle=False)
        embeddings = data["embeddings"].astype(np.float32)
        names = data["names"].astype(str).tolist()

        print(f"Loaded {len(embeddings)} cached embeddings")

        return embeddings, names

    app = helper.create_app(
        (640, 640),
        det_thresh=0.35,
    )

    embeddings, names, _ = helper.load_embeddings(
        app,
        rebuild=True,
    )

    return embeddings, names


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
