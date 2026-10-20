"""Image and embedding loading shared by offline evaluation commands."""

import os

import numpy as np

from config import (
    ENROLLMENT_FACE_MIN_SIZE,
    EMBEDDING_FOLDERS,
)
from .embeddings import (
    create_evaluation_app,
    get_image_embedding,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def iter_images(root):
    if not os.path.isdir(root):
        return

    for current_root, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(current_root, filename)


def synchronize_device(device):
    try:
        import torch
    except ImportError:
        return

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif str(device) == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def load_enrollment(app, enrollment_dir, mode):
    embeddings = []
    names = []
    paths = []
    person_names = sorted(
        entry
        for entry in os.listdir(enrollment_dir)
        if os.path.isdir(os.path.join(enrollment_dir, entry))
    )

    selected_folders = ("originals",) if mode == "original" else EMBEDDING_FOLDERS
    for person_name in person_names:
        person_folder = os.path.join(enrollment_dir, person_name)
        for folder_name in selected_folders:
            image_folder = os.path.join(person_folder, folder_name)
            for image_path in iter_images(image_folder):
                embedding = get_image_embedding(
                    app,
                    image_path,
                    min_face_size=ENROLLMENT_FACE_MIN_SIZE,
                )
                if embedding is None:
                    continue
                embeddings.append(embedding)
                names.append(person_name)
                paths.append(image_path)

    if not embeddings:
        raise RuntimeError(
            f"No usable {mode} enrollment images found in {enrollment_dir}"
        )

    return np.asarray(embeddings, dtype=np.float32), names, paths


def load_labeled_images(app, image_root):
    samples = []
    for label in sorted(os.listdir(image_root)):
        label_folder = os.path.join(image_root, label)
        if not os.path.isdir(label_folder):
            continue

        for image_path in iter_images(label_folder):
            embedding = get_image_embedding(app, image_path)
            samples.append((label, embedding, image_path))

    if not samples:
        raise RuntimeError(f"No usable face images found in {image_root}")

    return samples
