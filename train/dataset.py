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
    highest_count = 0

    for filename in os.listdir(folder):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        suffix = os.path.splitext(filename)[0].rsplit("_", 1)[-1]
        if suffix.isdigit():
            highest_count = max(highest_count, int(suffix))

    return highest_count + 1


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
    all_saved = True

    for form_name, generated_image in create_face_forms(face_image).items():
        image_path = os.path.join(folders[form_name], image_name)
        saved = save_image(image_path, generated_image)

        if not saved:
            print(f"Could not save: {image_path}")
            all_saved = False

    if all_saved:
        print(f"Saved {pose_name} image set: {image_count}")
    else:
        print(f"Image set incomplete: {pose_name} {image_count}")

    return all_saved
