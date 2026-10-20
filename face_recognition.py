import hashlib
import importlib.metadata
import inspect
import json
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from recognition.recognizer import (
    assess_face_quality,
    embed_crops,
    get_best_face,
    normalize_embedding,
)
from runtime_utils import get_execution_providers
from train.augmentations import (
    create_bright_version,
    create_dark_version,
    create_shadow_version,
    improve_lighting,
)
from config import (
    EMBEDDING_DETECTION_SIZE,
    EMBEDDING_DETECTION_THRESHOLD,
    EMBEDDING_DIMENSION,
    EMBEDDING_FOLDERS,
    ENROLLMENT_FACE_MIN_SIZE,
    FACE_BLUR_THRESHOLD,
    FACE_FRAME_EDGE_MARGIN,
    FACE_MAX_OUTSIDE_RATIO,
    MAX_FACE_ASPECT_RATIO,
    MIN_FACE_ASPECT_RATIO,
    MODEL_NAME,
)


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
TRAINED_DATA_FOLDER = os.environ.get(
    "FACE_TRAINED_DATA_FOLDER",
    os.path.join(BASE_FOLDER, "trained_data"),
)

EMBEDDINGS_FILE = os.path.join(BASE_FOLDER, "face_embeddings.npz")
MANIFEST_FILE = os.path.join(BASE_FOLDER, "face_embeddings_manifest.json")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SOURCE_FOLDER = "originals"
CACHE_SCHEMA_VERSION = 2
AUGMENTATION_VERSION = 1


def create_app(
    det_size=EMBEDDING_DETECTION_SIZE,
    det_thresh=EMBEDDING_DETECTION_THRESHOLD,
):
    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=get_execution_providers(),
    )

    app.prepare(
        ctx_id=-1,
        det_size=det_size,
        det_thresh=det_thresh,
    )

    return app


def get_transformed_images(image):
    return {
        "corrected": improve_lighting(image),
        "dark": create_dark_version(image),
        "bright": create_bright_version(image),
        "shadow": create_shadow_version(image),
    }


def iter_people():
    if not os.path.isdir(TRAINED_DATA_FOLDER):
        return

    for person_name in sorted(os.listdir(TRAINED_DATA_FOLDER)):
        person_folder = os.path.join(TRAINED_DATA_FOLDER, person_name)

        if os.path.isdir(person_folder):
            yield person_name, person_folder


def iter_images(folder):
    if not os.path.isdir(folder):
        return

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            yield os.path.join(folder, filename)


def _runtime_manifest(embedding_dimension=EMBEDDING_DIMENSION):
    try:
        insightface_version = importlib.metadata.version("insightface")
    except importlib.metadata.PackageNotFoundError:
        insightface_version = "unknown"

    preprocessing_functions = (
        create_app,
        get_transformed_images,
        transform_training_images,
        build_embeddings,
        create_bright_version,
        create_dark_version,
        create_shadow_version,
        improve_lighting,
        assess_face_quality,
        embed_crops,
        get_best_face,
        normalize_embedding,
    )
    preprocessing_source = "\n".join(
        inspect.getsource(function) for function in preprocessing_functions
    )
    preprocessing_fingerprint = hashlib.sha256(
        preprocessing_source.encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_version": insightface_version,
        "embedding_dimension": embedding_dimension,
        "preprocessing": {
            "detection_size": list(EMBEDDING_DETECTION_SIZE),
            "detection_threshold": EMBEDDING_DETECTION_THRESHOLD,
            "alignment": "InsightFace FaceAnalysis.get landmark alignment",
            "embedding_folders": list(EMBEDDING_FOLDERS),
            "augmentation_version": AUGMENTATION_VERSION,
            "code_fingerprint": preprocessing_fingerprint,
            "quality": {
                "minimum_face_size": ENROLLMENT_FACE_MIN_SIZE,
                "blur_threshold": FACE_BLUR_THRESHOLD,
                "frame_edge_margin": FACE_FRAME_EDGE_MARGIN,
                "maximum_outside_ratio": FACE_MAX_OUTSIDE_RATIO,
                "minimum_aspect_ratio": MIN_FACE_ASPECT_RATIO,
                "maximum_aspect_ratio": MAX_FACE_ASPECT_RATIO,
            },
        },
    }


def make_manifest(embedding_dimension=EMBEDDING_DIMENSION):
    files = []

    for person_name, person_folder in iter_people():
        for folder_name in EMBEDDING_FOLDERS:
            image_folder = os.path.join(person_folder, folder_name)

            for image_path in iter_images(image_folder):
                stat = os.stat(image_path)
                files.append(
                    {
                        "person": person_name,
                        "path": os.path.relpath(image_path, TRAINED_DATA_FOLDER),
                        "mtime_ns": stat.st_mtime_ns,
                        "size": stat.st_size,
                    }
                )

    canonical_files = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = _runtime_manifest(embedding_dimension)
    manifest.update(
        {
            "dataset_fingerprint": hashlib.sha256(canonical_files).hexdigest(),
            "files": files,
        }
    )

    return manifest


def load_manifest():
    if not os.path.exists(MANIFEST_FILE):
        return None

    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return None


def save_manifest(manifest):
    temporary_path = f"{MANIFEST_FILE}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, MANIFEST_FILE)


def _validate_embedding_cache(manifest):
    try:
        with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            names = np.asarray(data["names"], dtype=str)
            paths = np.asarray(data["paths"], dtype=str)
    except (EOFError, OSError, TypeError, ValueError, KeyError):
        return False

    if embeddings.ndim != 2 or embeddings.shape[1] != manifest.get(
        "embedding_dimension"
    ):
        return False
    if (
        len(embeddings) == 0
        or len(embeddings) != len(names)
        or len(embeddings) != len(paths)
    ):
        return False
    if (
        not np.all(np.isfinite(embeddings))
        or np.any(names == "")
        or np.any(paths == "")
    ):
        return False

    norms = np.linalg.norm(embeddings, axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= 1e-3))


def database_needs_rebuild():
    if not os.path.exists(EMBEDDINGS_FILE):
        return True

    manifest = load_manifest()
    if not isinstance(manifest, dict):
        return True

    runtime_manifest = _runtime_manifest()
    if any(manifest.get(key) != value for key, value in runtime_manifest.items()):
        return True

    if not _validate_embedding_cache(manifest):
        return True

    # Keep a valid cache usable when a custom training directory is offline.
    if not os.path.isdir(TRAINED_DATA_FOLDER):
        return False

    return manifest != make_manifest(manifest["embedding_dimension"])


def transform_training_images():
    if not os.path.isdir(TRAINED_DATA_FOLDER):
        raise RuntimeError(
            f"Training data folder not found: {TRAINED_DATA_FOLDER}"
        )

    transformed_count = 0

    for _, person_folder in iter_people():
        originals_folder = os.path.join(person_folder, SOURCE_FOLDER)

        for image_path in iter_images(originals_folder):
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read: {image_path}")
                continue

            filename = os.path.basename(image_path)

            for folder_name, transformed in get_transformed_images(
                image
            ).items():
                output_folder = os.path.join(person_folder, folder_name)
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, filename)

                if cv2.imwrite(output_path, transformed):
                    transformed_count += 1
                else:
                    print(f"Could not write: {output_path}")

    return transformed_count


def build_embeddings(app):
    embeddings = []
    names = []
    paths = []

    for person_name, person_folder in iter_people():
        loaded_for_person = 0
        original_paths = list(
            iter_images(os.path.join(person_folder, SOURCE_FOLDER))
        )
        if not original_paths:
            print(f"No original enrollment images for {person_name}; skipping")
            continue

        original_names = {
            os.path.basename(path)
            for path in original_paths
            if cv2.imread(path) is not None
        }

        for folder_name in EMBEDDING_FOLDERS:
            image_folder = os.path.join(person_folder, folder_name)

            for image_path in iter_images(image_folder):
                if (
                    folder_name != SOURCE_FOLDER
                    and os.path.basename(image_path) not in original_names
                ):
                    continue

                image = cv2.imread(image_path)

                if image is None:
                    print(f"Could not read: {image_path}")
                    continue

                face = get_best_face(app.get(image))

                if face is None:
                    print(f"No face detected: {image_path}")
                    continue

                quality = assess_face_quality(
                    image,
                    face.bbox,
                    min_face_size=ENROLLMENT_FACE_MIN_SIZE,
                )
                if not quality["accepted"]:
                    print(
                        f"Poor quality ({quality['reason']}): {image_path}"
                    )
                    continue

                normalized_embedding = normalize_embedding(face.embedding)
                if normalized_embedding.shape != (EMBEDDING_DIMENSION,):
                    print(f"Unexpected embedding dimension: {image_path}")
                    continue
                if not np.isclose(
                    np.linalg.norm(normalized_embedding),
                    1.0,
                    atol=1e-3,
                ):
                    print(f"Invalid embedding: {image_path}")
                    continue

                embeddings.append(normalized_embedding)
                names.append(person_name)
                paths.append(image_path)
                loaded_for_person += 1

        if loaded_for_person == 0:
            raise RuntimeError(
                f"No usable embeddings were found for {person_name}; "
                "check the images or quality thresholds."
            )

        print(f"Loaded {loaded_for_person} embeddings for {person_name}")

    if not embeddings:
        raise RuntimeError(
            f"No embeddings were built from: {TRAINED_DATA_FOLDER}"
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError("Built embeddings contain non-finite values")

    temporary_path = f"{EMBEDDINGS_FILE}.tmp"
    with open(temporary_path, "wb") as file:
        np.savez_compressed(
            file,
            embeddings=embeddings,
            names=np.asarray(names),
            paths=np.asarray(paths),
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, EMBEDDINGS_FILE)
    save_manifest(make_manifest(embeddings.shape[1]))

    return embeddings, names, paths


def load_embeddings(app, rebuild=False):
    if rebuild:
        transformed_count = transform_training_images()

        if transformed_count:
            print(f"Updated {transformed_count} transformed training images")

        print("Building face embedding database...")
        return build_embeddings(app)

    if database_needs_rebuild():
        raise RuntimeError(
            "The embedding cache is missing, stale, or invalid. "
            "Rebuild it separately with `python main.py --rebuild --build-only`."
        )

    manifest = load_manifest()
    if not isinstance(manifest, dict) or not _validate_embedding_cache(manifest):
        raise RuntimeError("The face embedding cache is missing or invalid")

    with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
        embeddings = data["embeddings"].astype(np.float32)
        names = data["names"].astype(str).tolist()
        paths = data["paths"].astype(str).tolist()

    print(f"Loaded {len(embeddings)} cached embeddings")

    return embeddings, names, paths
