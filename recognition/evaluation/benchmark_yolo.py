"""Offline YOLO image-size latency benchmark."""

import random
import time

import cv2
import numpy as np
from ultralytics import YOLO

from config import DEFAULT_YOLO_MODEL
from .data import iter_images, synchronize_device


def benchmark_yolo(args):
    all_image_paths = list(iter_images(args.image_dir))
    if not all_image_paths:
        raise RuntimeError(f"No images found under: {args.image_dir}")

    if args.max_images > 0 and len(all_image_paths) > args.max_images:
        image_paths = random.Random(42).sample(all_image_paths, args.max_images)
    else:
        image_paths = all_image_paths

    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)

    if not images:
        raise RuntimeError(f"No readable images found under: {args.image_dir}")

    model = YOLO(args.model or DEFAULT_YOLO_MODEL)
    print(f"Benchmarking {len(images)} images with {args.model}")
    print(
        "Warmup runs are excluded from the timings; detection counts do not "
        "measure face-detection accuracy."
    )

    for image_size in args.sizes:
        for _ in range(args.warmup):
            model.predict(
                images[0],
                imgsz=image_size,
                device=args.device,
                conf=args.conf,
                verbose=False,
            )
        synchronize_device(args.device)

        durations = []
        detection_counts = []
        for _ in range(args.repeats):
            for image in images:
                synchronize_device(args.device)
                started_at = time.perf_counter()
                result = model.predict(
                    image,
                    imgsz=image_size,
                    device=args.device,
                    conf=args.conf,
                    verbose=False,
                )
                synchronize_device(args.device)
                durations.append((time.perf_counter() - started_at) * 1000.0)
                boxes = result[0].boxes if result else None
                detection_counts.append(0 if boxes is None else len(boxes))

        values = np.asarray(durations, dtype=np.float64)
        print(
            f"imgsz={image_size}: mean={values.mean():.1f} ms, "
            f"p50={np.percentile(values, 50):.1f} ms, "
            f"p95={np.percentile(values, 95):.1f} ms, "
            f"mean faces={np.mean(detection_counts):.2f}"
        )
