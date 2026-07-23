import os


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

FACE_RECOGNITION_FILE = os.path.join(BASE_FOLDER, "face_recognition.py")
EMBEDDINGS_FILE = os.path.join(BASE_FOLDER, "face_embeddings.npz")
YOLO_MODEL_FOLDER = os.path.join(BASE_FOLDER, "models")
DEFAULT_YOLO_MODEL = os.path.join(YOLO_MODEL_FOLDER, "yolov8n-face.pt")

DEFAULT_YOLO_MODEL_URL = (
    "https://github.com/YapaLab/yolo-face/releases/download/"
    "1.0.0/yolov8n-face.pt"
)

MODEL_NAME = "buffalo_s"
SIMILARITY_THRESHOLD = 0.45

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

YOLO_IMAGE_SIZE = 960
YOLO_CONFIDENCE = 0.25
MAX_FACES = 20

FACE_PADDING_MIN = 24
FACE_PADDING_RATIO = 0.18
MIN_FACE_CROP_SIZE = 224

EMBEDDER_DETECTION_SIZE = (320, 320)
EMBEDDER_DETECTION_THRESHOLD = 0.4
