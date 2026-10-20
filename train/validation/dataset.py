"""Duplicate detection helpers for original enrollment images."""

import hashlib
import os

import cv2

from train.config import IMAGE_EXTENSIONS


def difference_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0

    for bit in bits.ravel():
        value = (value << 1) | int(bit)

    return value


def hamming_distance(first, second):
    return (first ^ second).bit_count()


def iter_originals(data_root):
    if not os.path.isdir(data_root):
        return

    for person_name in sorted(os.listdir(data_root)):
        person_folder = os.path.join(data_root, person_name, "originals")
        if not os.path.isdir(person_folder):
            continue

        for filename in sorted(os.listdir(person_folder)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                yield person_name, os.path.join(person_folder, filename)


def find_duplicates(data_root, max_distance=5):
    images_by_person = {}
    unreadable = []

    for person_name, image_path in iter_originals(data_root):
        image = cv2.imread(image_path)
        if image is None:
            unreadable.append(image_path)
            continue

        with open(image_path, "rb") as file:
            exact_hash = hashlib.sha256(file.read()).hexdigest()

        images_by_person.setdefault(person_name, []).append(
            {
                "path": image_path,
                "exact_hash": exact_hash,
                "difference_hash": difference_hash(image),
            }
        )

    duplicate_pairs = []
    for person_name, images in images_by_person.items():
        for index, first in enumerate(images):
            for second in images[index + 1 :]:
                if first["exact_hash"] == second["exact_hash"]:
                    distance = 0
                    duplicate_type = "exact"
                else:
                    distance = hamming_distance(
                        first["difference_hash"],
                        second["difference_hash"],
                    )
                    if distance > max_distance:
                        continue
                    duplicate_type = "near"

                duplicate_pairs.append(
                    (
                        person_name,
                        duplicate_type,
                        distance,
                        first["path"],
                        second["path"],
                    )
                )

    return duplicate_pairs, unreadable
