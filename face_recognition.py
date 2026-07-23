import json
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
TRAINED_DATA_FOLDER = os.environ.get(
    "FACE_TRAINED_DATA_FOLDER",
    os.path.join(BASE_FOLDER, "trained_data"),
)

EMBEDDINGS_FILE = os.path.join(BASE_FOLDER, "face_embeddings.npz")
MANIFEST_FILE = os.path.join(BASE_FOLDER, "face_embeddings_manifest.json")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MODEL_NAME = "buffalo_s"
SOURCE_FOLDER = "originals"
EMBEDDING_FOLDERS = (
    "originals",
    "corrected",
    "dark",
    "bright",
    "shadow",
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


def create_app(det_size, det_thresh=0.5):
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


def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def improve_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    improved = clahe.apply(lightness)

    merged = cv2.merge((improved, channel_a, channel_b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def create_dark_version(image):
    return cv2.convertScaleAbs(image, alpha=0.55, beta=-25)


def create_bright_version(image):
    return cv2.convertScaleAbs(image, alpha=1.20, beta=35)


def create_shadow_version(image):
    height, width = image.shape[:2]
    gradient = np.linspace(0.35, 1.0, width, dtype=np.float32)
    shadow_mask = np.tile(gradient, (height, 1))
    shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 0)

    shadow_image = image.astype(np.float32)
    shadow_image *= shadow_mask[:, :, np.newaxis]

    return np.clip(shadow_image, 0, 255).astype(np.uint8)


def get_transformed_images(image):
    return {
        "corrected": improve_lighting(image),
        "dark": create_dark_version(image),
        "bright": create_bright_version(image),
        "shadow": create_shadow_version(image),
    }


def get_best_face(faces):
    if not faces:
        return None

    return max(
        faces,
        key=lambda face: (
            face.bbox[2] - face.bbox[0]
        ) * (
            face.bbox[3] - face.bbox[1]
        ),
    )


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


def make_manifest():
    files = []

    for person_name, person_folder in iter_people():
        for folder_name in EMBEDDING_FOLDERS:
            image_folder = os.path.join(person_folder, folder_name)

            for image_path in iter_images(image_folder):
                stat = os.stat(image_path)
                files.append(
                    {
                        "person": person_name,
                        "path": image_path,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                    }
                )

    return {
        "model": MODEL_NAME,
        "folders": list(EMBEDDING_FOLDERS),
        "files": files,
    }


def has_training_images():
    return bool(make_manifest()["files"])


def load_manifest():
    if not os.path.exists(MANIFEST_FILE):
        return None

    with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(manifest):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def database_needs_rebuild():
    if not os.path.exists(EMBEDDINGS_FILE):
        return True

    if not os.path.isdir(TRAINED_DATA_FOLDER) or not has_training_images():
        return False

    return load_manifest() != make_manifest()


def transform_training_images():
    if not os.path.isdir(TRAINED_DATA_FOLDER):
        raise RuntimeError(
            f"Training data folder not found: {TRAINED_DATA_FOLDER}"
        )

    created_count = 0

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

                if os.path.exists(output_path):
                    continue

                if cv2.imwrite(output_path, transformed):
                    created_count += 1

    return created_count


def build_embeddings(app):
    embeddings = []
    names = []
    paths = []

    for person_name, person_folder in iter_people():
        loaded_for_person = 0

        for folder_name in EMBEDDING_FOLDERS:
            image_folder = os.path.join(person_folder, folder_name)

            for image_path in iter_images(image_folder):
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Could not read: {image_path}")
                    continue

                face = get_best_face(app.get(image))

                if face is None:
                    print(f"No face detected: {image_path}")
                    continue

                embeddings.append(normalize_embedding(face.embedding))
                names.append(person_name)
                paths.append(image_path)
                loaded_for_person += 1

        print(f"Loaded {loaded_for_person} embeddings for {person_name}")

    if not embeddings:
        raise RuntimeError(
            f"No embeddings were built from: {TRAINED_DATA_FOLDER}"
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.savez_compressed(
        EMBEDDINGS_FILE,
        embeddings=embeddings,
        names=np.asarray(names),
        paths=np.asarray(paths),
    )
    save_manifest(make_manifest())

    return embeddings, names, paths


def load_embeddings(app, rebuild=False):
    if rebuild or database_needs_rebuild():
        created_count = transform_training_images()

        if created_count:
            print(f"Created {created_count} transformed training images")

        print("Building face embedding database...")
        return build_embeddings(app)

    data = np.load(EMBEDDINGS_FILE, allow_pickle=False)
    embeddings = data["embeddings"].astype(np.float32)
    names = data["names"].astype(str).tolist()
    paths = data["paths"].astype(str).tolist()

    print(f"Loaded {len(embeddings)} cached embeddings")

    return embeddings, names, paths
