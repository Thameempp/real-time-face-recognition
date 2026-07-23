import os

import cv2

from .augmentations import create_face_forms
from .config import (
    FOLDER_NAMES,
    IMAGE_EXTENSIONS,
    JPEG_PARAMS,
    TRAINED_DATA_FOLDER,
)


def create_person_folders(person_name):
    """Create a separate folder for every image form."""
    person_folder = os.path.join(TRAINED_DATA_FOLDER, person_name)
    folders = {}

    for folder_name in FOLDER_NAMES:
        folder_path = os.path.join(person_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        folders[folder_name] = folder_path

    return folders


def find_next_count(folder):
    """Continue numbering without overwriting old photos."""
    image_files = [
        filename
        for filename in os.listdir(folder)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ]

    return len(image_files) + 1


def save_image(path, image):
    if path.lower().endswith((".jpg", ".jpeg")):
        return cv2.imwrite(path, image, JPEG_PARAMS)

    return cv2.imwrite(path, image)


def save_face_forms(
    face_image,
    folders,
    person_name,
    image_count,
    pose_name,
):
    """Create and save every face variation."""
    image_name = f"{person_name}_{pose_name}_{image_count:03d}.jpg"

    for form_name, generated_image in create_face_forms(face_image).items():
        image_path = os.path.join(folders[form_name], image_name)
        saved = save_image(image_path, generated_image)

        if not saved:
            print(f"Could not save: {image_path}")

    print(f"Saved {pose_name} image set: {image_count}")
