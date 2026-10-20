"""Benchmark serial and batched InsightFace embedding paths."""

import random
import time

import cv2
import numpy as np

from .data import iter_images
from .embeddings import create_evaluation_app
from recognition.recognizer import embed_crops, get_best_face, normalize_embedding


def benchmark_embeddings(args):
    image_paths = list(iter_images(args.image_dir))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.image_dir}")
    if args.max_images > 0 and len(image_paths) > args.max_images:
        image_paths = random.Random(42).sample(image_paths, args.max_images)

    images = [cv2.imread(path) for path in image_paths]
    images = [image for image in images if image is not None]
    if not images:
        raise RuntimeError(f"No readable images found under: {args.image_dir}")

    app = create_evaluation_app()
    for _ in range(args.warmup):
        app.get(images[0])
        embed_crops(app, images[: min(args.batch_size, len(images))])

    single_durations = []
    single_embeddings = []
    for _ in range(args.repeats):
        for image in images:
            started_at = time.perf_counter()
            face = get_best_face(app.get(image))
            single_durations.append(time.perf_counter() - started_at)
            single_embeddings.append(
                None if face is None else np.asarray(face.embedding, dtype=np.float32)
            )

    batch_durations = []
    batch_embeddings = []
    batch_supported = True
    for _ in range(args.repeats):
        repeat_embeddings = []
        for start in range(0, len(images), args.batch_size):
            crop_batch = images[start : start + args.batch_size]
            started_at = time.perf_counter()
            embeddings, used_batch = embed_crops(
                app,
                crop_batch,
                return_batch_status=True,
            )
            batch_durations.append(time.perf_counter() - started_at)
            batch_supported = batch_supported and used_batch
            repeat_embeddings.extend(embeddings)
        batch_embeddings.append(repeat_embeddings)

    common_count = min(len(single_embeddings), len(batch_embeddings[-1]))
    cosine_scores = []
    for first, second in zip(
        single_embeddings[:common_count],
        batch_embeddings[-1][:common_count],
    ):
        if first is None or second is None:
            continue
        cosine_scores.append(
            float(np.dot(normalize_embedding(first), normalize_embedding(second)))
        )

    single_seconds = sum(single_durations) / args.repeats
    batch_seconds = sum(batch_durations) / args.repeats
    print(f"Images: {len(images)}; repeats: {args.repeats}; warmup: {args.warmup}")
    print(
        f"FaceAnalysis.get one at a time: "
        f"{single_seconds / len(images) * 1000.0:.1f} ms/face"
    )
    print(
        f"Aligned recognition in batches of {args.batch_size}: "
        f"{batch_seconds / len(images) * 1000.0:.1f} ms/face"
    )
    print(f"Batched ONNX recognition supported by this provider: {batch_supported}")
    if cosine_scores:
        print(f"Mean cosine agreement with FaceAnalysis.get: {np.mean(cosine_scores):.6f}")
