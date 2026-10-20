"""Public recognition helpers; implementations live in focused modules."""

from recognition.embeddings import embed_crop, embed_crops
from recognition.face_quality import (
    assess_face_quality,
    get_best_face,
    prepare_face_crop,
)
from recognition.matching import (
    EmbeddingMatcher,
    normalize_embedding,
    recognize_embedding,
)

__all__ = [
    "EmbeddingMatcher",
    "assess_face_quality",
    "embed_crop",
    "embed_crops",
    "get_best_face",
    "normalize_embedding",
    "prepare_face_crop",
    "recognize_embedding",
]
