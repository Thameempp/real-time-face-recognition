import numpy as np
from insightface.utils import face_align


def embed_crop(embedder_app, crop):
    return embed_crops(embedder_app, [crop])[0]


def embed_crops(embedder_app, crops, return_batch_status=False):
    """Create aligned embeddings, batching the InsightFace recognition model."""
    recognition_model = embedder_app.models.get("recognition")
    if recognition_model is None:
        raise RuntimeError("InsightFace recognition model is unavailable")

    embeddings = [None] * len(crops)
    aligned_faces = []
    aligned_indices = []

    for index, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            continue

        boxes, keypoints = embedder_app.det_model.detect(
            crop,
            max_num=0,
            metric="default",
        )
        if len(boxes) == 0 or keypoints is None:
            continue

        face_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        face_index = int(np.argmax(face_areas))
        aligned = face_align.norm_crop(
            crop,
            landmark=keypoints[face_index],
            image_size=recognition_model.input_size[0],
        )
        aligned_faces.append(aligned)
        aligned_indices.append(index)

    if not aligned_faces:
        return (embeddings, False) if return_batch_status else embeddings

    # This is the same detect -> landmark norm_crop -> recognition sequence
    # used by FaceAnalysis.get, with only the recognition forward pass batched.
    used_batch = True
    try:
        features = np.asarray(recognition_model.get_feat(aligned_faces))
        if features.shape[0] != len(aligned_faces):
            raise ValueError("Batched InsightFace output has an unexpected size")
    except Exception:
        used_batch = False
        features = np.concatenate(
            [
                np.asarray(recognition_model.get_feat(aligned)).reshape(1, -1)
                for aligned in aligned_faces
            ],
            axis=0,
        )

    for index, feature in zip(aligned_indices, features):
        embeddings[index] = np.asarray(feature, dtype=np.float32).reshape(-1)

    if return_batch_status:
        return embeddings, used_batch
    return embeddings
