"""CLI for YOLO benchmarks, embedding benchmarks, and matching evaluation."""

import argparse

from config import DEFAULT_YOLO_MODEL
from .data import (
    IMAGE_EXTENSIONS,
    create_evaluation_app,
    get_image_embedding,
    iter_images,
    load_enrollment,
    load_labeled_images,
    synchronize_device,
)
from .benchmark_embeddings import benchmark_embeddings
from .metrics import (
    calibrate_threshold,
    rates_at_threshold,
    score_samples,
)
from .matching import (
    POSE_NAMES,
    build_gallery,
    evaluate_matching,
    pose_from_path,
)
from .benchmark_yolo import benchmark_yolo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark-yolo")
    benchmark.add_argument("--image-dir", required=True)
    benchmark.add_argument("--model", default=DEFAULT_YOLO_MODEL)
    benchmark.add_argument("--sizes", nargs="+", type=int, default=[416, 512, 640])
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--conf", type=float, default=0.25)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Deterministically sample this many benchmark images; 0 uses all.",
    )

    embedding_benchmark = subparsers.add_parser("benchmark-embeddings")
    embedding_benchmark.add_argument("--image-dir", required=True)
    embedding_benchmark.add_argument("--batch-size", type=int, default=8)
    embedding_benchmark.add_argument("--warmup", type=int, default=2)
    embedding_benchmark.add_argument("--repeats", type=int, default=3)
    embedding_benchmark.add_argument("--max-images", type=int, default=24)

    matching = subparsers.add_parser("evaluate-matching")
    matching.add_argument("--enrollment-dir", default="trained_data")
    matching.add_argument("--calibration-dir", required=True)
    matching.add_argument("--test-dir", required=True)
    matching.add_argument(
        "--enrollment-modes",
        nargs="+",
        choices=("original", "augmented"),
        default=["original", "augmented"],
    )
    matching.add_argument(
        "--strategies",
        nargs="+",
        choices=("top3", "max", "prototype", "pose-prototypes"),
        default=["top3", "max", "prototype", "pose-prototypes"],
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "benchmark-yolo":
        benchmark_yolo(args)
    elif args.command == "benchmark-embeddings":
        benchmark_embeddings(args)
    else:
        evaluate_matching(args)


if __name__ == "__main__":
    main()
