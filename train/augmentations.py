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


def create_grayscale_version(image):
    """Convert the face into grayscale."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


def create_edge_version(image):
    """Detect strong facial edges using Canny."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, 70, 150)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def create_outline_version(image):
    """Create a smooth face-outline style image."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(grayscale, 9, 75, 75)
    edges = cv2.Canny(blurred, 40, 100)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    outline = np.full_like(image, 255)
    outline[edges > 0] = (0, 0, 0)

    return outline


def create_pattern_version(image):
    """Highlight facial texture patterns using local contrast."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(grayscale)
    pattern = cv2.Laplacian(enhanced, cv2.CV_64F, ksize=3)
    pattern = cv2.convertScaleAbs(pattern)

    return cv2.applyColorMap(pattern, cv2.COLORMAP_BONE)


def create_face_forms(face_image):
    corrected = improve_lighting(face_image)

    return {
        "originals": face_image,
        "corrected": corrected,
        "dark": create_dark_version(face_image),
        "bright": create_bright_version(face_image),
        "shadow": create_shadow_version(face_image),
        "grayscale": create_grayscale_version(face_image),
        "edges": create_edge_version(face_image),
        "outline": create_outline_version(face_image),
        "pattern": create_pattern_version(face_image),
    }
