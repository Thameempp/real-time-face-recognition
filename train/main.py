import argparse
import multiprocessing as mp
import os
import sys

import cv2

if __package__ in (None, ""):
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    from train.collector import run_collector
    from train.utils import clean_name
else:
    from .collector import run_collector
    from .utils import clean_name


cv2.setUseOptimized(True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect transformed face-training images."
    )

    parser.add_argument(
        "--name",
        help="Person name. If omitted, the script asks interactively.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.name:
        person_name = clean_name(args.name)
    else:
        person_name = clean_name(input("Enter person's name: "))

    if not person_name:
        raise ValueError("Person name cannot be empty")

    run_collector(person_name)


if __name__ == "__main__":
    mp.freeze_support()
    main()
