import os

import cv2


PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINED_DATA_FOLDER = os.environ.get(
    "FACE_TRAINED_DATA_FOLDER",
    os.path.join(PROJECT_FOLDER, "trained_data"),
)

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

MODEL_NAME = "buffalo_s"
LIVE_INFERENCE_WIDTH = 960
LIVE_DETECTION_SIZE = (640, 640)
LIVE_DETECTION_THRESHOLD = 0.35

FACE_PADDING_MIN = 60
FACE_PADDING_RATIO = 0.35
MIN_FACE_CROP_SIZE = 320
SAVE_COOLDOWN_SECONDS = 0.5
MAX_SAVE_RESULT_AGE_SECONDS = 1.0
RECENT_FRAME_LIMIT = 12

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
JPEG_PARAMS = [
    cv2.IMWRITE_JPEG_QUALITY,
    95,
]

FOLDER_NAMES = [
    "originals",
    "corrected",
    "dark",
    "bright",
    "shadow",
    "grayscale",
    "edges",
    "outline",
    "pattern",
]

POSE_KEYS = {
    ord("1"): "front",
    ord("2"): "slight_left",
    ord("3"): "strong_left",
    ord("4"): "slight_right",
    ord("5"): "strong_right",
    ord("6"): "looking_up",
    ord("7"): "looking_down",
}
