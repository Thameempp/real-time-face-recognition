import cv2
import numpy as np


def improve_lighting(image):
    """Improve local lighting and contrast using CLAHE."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab_image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    improved_lightness = clahe.apply(lightness)
    improved_lab = cv2.merge(
        (
            improved_lightness,
            channel_a,
            channel_b,
        )
    )

    return cv2.cvtColor(improved_lab, cv2.COLOR_LAB2BGR)


def create_dark_version(image):
    """Create a darker version of the face."""
    return cv2.convertScaleAbs(image, alpha=0.55, beta=-25)


def create_bright_version(image):
    """Create a brighter version of the face."""
    return cv2.convertScaleAbs(image, alpha=1.20, beta=35)


def create_shadow_version(image):
    """Add a soft artificial shadow over one side of the face."""
    height, width = image.shape[:2]
    gradient = np.linspace(0.35, 1.0, width, dtype=np.float32)
    shadow_mask = np.tile(gradient, (height, 1))
    shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 0)

    shadow_image = image.astype(np.float32)
    shadow_image *= shadow_mask[:, :, np.newaxis]

    return np.clip(shadow_image, 0, 255).astype(np.uint8)


def create_face_forms(face_image):
    corrected = improve_lighting(face_image)

    return {
        "originals": face_image,
        "corrected": corrected,
        "dark": create_dark_version(face_image),
        "bright": create_bright_version(face_image),
        "shadow": create_shadow_version(face_image),
    }
