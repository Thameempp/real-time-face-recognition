"""Build per-image or representative-prototype enrollment galleries."""

import os

import numpy as np

from recognition.recognizer import normalize_embedding

POSE_NAMES = {
    "front",
    "slight_left",
    "strong_left",
    "slight_right",
    "strong_right",
    "looking_up",
    "looking_down",
}


def pose_from_path(image_path):
    parts = os.path.splitext(os.path.basename(image_path))[0].rsplit("_", 2)
    if len(parts) == 3 and parts[-2] in POSE_NAMES:
        return parts[-2]
    return "all"


def build_gallery(embeddings, names, paths, strategy):
    if strategy in {"top3", "max"}:
        return embeddings, names, strategy

    grouped = {}
    for embedding, name, path in zip(embeddings, names, paths):
        group = pose_from_path(path) if strategy == "pose-prototypes" else "all"
        grouped.setdefault((name, group), []).append(embedding)

    prototype_embeddings = []
    prototype_names = []
    for (name, _), group_embeddings in grouped.items():
        prototype = normalize_embedding(np.mean(group_embeddings, axis=0))
        prototype_embeddings.append(prototype)
        prototype_names.append(name)

    return (
        np.asarray(prototype_embeddings, dtype=np.float32),
        prototype_names,
        "max",
    )
