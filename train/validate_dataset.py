"""Command-line report for exact and near-duplicate enrollment images."""

import argparse

from train.config import TRAINED_DATA_FOLDER
from train.validation.dataset import (
    difference_hash,
    find_duplicates,
    hamming_distance,
    iter_originals,
)

__all__ = [
    "difference_hash",
    "find_duplicates",
    "hamming_distance",
    "iter_originals",
    "main",
    "parse_args",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report duplicate or near-duplicate enrollment originals."
    )
    parser.add_argument(
        "--data-dir",
        default=TRAINED_DATA_FOLDER,
        help="Training-data root (default: configured training-data folder).",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=5,
        help="Maximum 64-bit difference-hash distance for near duplicates.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.max_distance <= 64:
        raise ValueError("--max-distance must be between 0 and 64")

    pairs, unreadable = find_duplicates(args.data_dir, args.max_distance)
    if pairs:
        for person_name, duplicate_type, distance, first, second in pairs:
            print(
                f"{duplicate_type} ({person_name}, hash distance {distance}):\n"
                f"  {first}\n"
                f"  {second}"
            )
    else:
        print("No exact or near duplicates found.")

    for image_path in unreadable:
        print(f"Unreadable image: {image_path}")

    print(f"Duplicate pairs: {len(pairs)}; unreadable images: {len(unreadable)}")


if __name__ == "__main__":
    main()
