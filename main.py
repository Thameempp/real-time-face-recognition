import argparse
import math
import multiprocessing as mp

import cv2

from recognition.camera import run_camera
from config import (
    DEFAULT_YOLO_MODEL,
    MATCHING_STRATEGY,
    SIMILARITY_THRESHOLD,
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
)
from recognition.model_utils import (
    ensure_yolo_model,
    get_torch_device,
    load_known_embeddings,
)


cv2.setUseOptimized(True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use YOLO for face detection and InsightFace embeddings "
            "for identity recognition."
        )
    )

    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help="Path to a YOLO face detection .pt model.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=YOLO_IMAGE_SIZE,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=YOLO_CONFIDENCE,
        help="YOLO face confidence threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device, for example cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--matching-strategy",
        choices=("top3", "max"),
        default=MATCHING_STRATEGY,
        help="Compare each identity using its top-three mean or maximum score.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Minimum identity similarity score to accept a match.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the embedding cache and exit without opening the camera.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Validate the embedding cache and exit; use --rebuild to regenerate it.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not auto-download the default YOLO face model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not math.isfinite(args.threshold) or not -1.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be a finite value between -1 and 1")
    if not math.isfinite(args.conf) or not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be a finite value between 0 and 1")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero")

    known_embeddings, known_names = load_known_embeddings(
        rebuild=args.rebuild
    )

    print(f"Total embeddings: {len(known_embeddings)}")
    print(f"Total people: {len(set(known_names))}")

    if args.build_only or args.rebuild:
        return

    if len(known_embeddings) == 0:
        raise RuntimeError(
            "No embeddings found. Collect images with `python -m train`, "
            "then rebuild embeddings."
        )

    yolo_model_path = ensure_yolo_model(
        args.yolo_model,
        allow_download=not args.no_download,
    )
    yolo_device = args.device or get_torch_device()

    print(f"YOLO model: {yolo_model_path}")
    print(f"YOLO device: {yolo_device}")

    run_camera(
        known_embeddings,
        known_names,
        yolo_model_path,
        args.imgsz,
        args.conf,
        yolo_device,
        args.matching_strategy,
        args.threshold,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
