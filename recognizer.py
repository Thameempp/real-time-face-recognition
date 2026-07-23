from collections import defaultdict

import cv2
import numpy as np

from config import (
    FACE_PADDING_MIN,
    FACE_PADDING_RATIO,
    MIN_FACE_CROP_SIZE,
    SIMILARITY_THRESHOLD,
)


def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


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


def recognize_embedding(embedding, known_embeddings, known_names):
    if len(known_embeddings) == 0:
        return "Unknown", 0.0

    embedding = normalize_embedding(embedding)
    similarities = np.dot(known_embeddings, embedding)
    scores_by_person = defaultdict(list)

    for person_name, score in zip(known_names, similarities):
        scores_by_person[person_name].append(float(score))

    best_name = "Unknown"
    best_score = -1.0

    for person_name, scores in scores_by_person.items():
        top_scores = sorted(scores, reverse=True)[:3]
        average_score = float(np.mean(top_scores))

        if average_score > best_score:
            best_name = person_name
            best_score = average_score

    if best_score < SIMILARITY_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


def crop_face(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).astype(int)

    face_width = max(1, x2 - x1)
    face_height = max(1, y2 - y1)

    padding_x = max(
        FACE_PADDING_MIN,
        int(face_width * FACE_PADDING_RATIO),
    )
    padding_y = max(
        FACE_PADDING_MIN,
        int(face_height * FACE_PADDING_RATIO),
    )

    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(width, x2 + padding_x)
    y2 = min(height, y2 + padding_y)

    return frame[y1:y2, x1:x2]


def upscale_small_crop(face_crop):
    height, width = face_crop.shape[:2]
    min_side = min(height, width)

    if min_side <= 0 or min_side >= MIN_FACE_CROP_SIZE:
        return face_crop

    scale = MIN_FACE_CROP_SIZE / min_side

    return cv2.resize(
        face_crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def embed_crop(embedder_app, crop):
    if crop.size == 0:
        return None

    crop = upscale_small_crop(crop)
    faces = embedder_app.get(crop)
    face = get_best_face(faces)

    if face is None:
        return None

    return face.embedding
