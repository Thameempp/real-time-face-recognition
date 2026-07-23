import argparse
import multiprocessing as mp

import cv2

from camera import run_camera
from config import DEFAULT_YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IMAGE_SIZE
from model_utils import (
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
        "--rebuild",
        action="store_true",
        help="Rebuild the InsightFace embedding cache first.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only verify/rebuild embeddings, then exit.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not auto-download the default YOLO face model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    known_embeddings, known_names = load_known_embeddings(
        rebuild=args.rebuild
    )

    print(f"Total embeddings: {len(known_embeddings)}")
    print(f"Total people: {len(set(known_names))}")

    if args.build_only:
        return

    if len(known_embeddings) == 0:
        raise RuntimeError(
            "No embeddings found. Run train.py, then rebuild embeddings."
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
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
