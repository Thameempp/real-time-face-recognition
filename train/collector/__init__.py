"""Enrollment capture lifecycle and helper functions."""

from .app import open_camera, run_collector
from .capture import remember_frame, save_current_face
from train.config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH, TRAINED_DATA_FOLDER
from train.dataset import create_person_folders, find_next_count, save_face_forms
from train.detector import detection_worker

__all__ = [
    "CAMERA_FPS",
    "CAMERA_HEIGHT",
    "CAMERA_WIDTH",
    "TRAINED_DATA_FOLDER",
    "create_person_folders",
    "detection_worker",
    "find_next_count",
    "open_camera",
    "remember_frame",
    "run_collector",
    "save_current_face",
    "save_face_forms",
]
