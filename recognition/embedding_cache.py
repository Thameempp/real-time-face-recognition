"""Load the embedding database builder and its validated cache."""

import importlib.util

from config import FACE_RECOGNITION_FILE


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
    app = helper.create_app() if rebuild else None
    embeddings, names, _ = helper.load_embeddings(app, rebuild=rebuild)

    return embeddings, names
