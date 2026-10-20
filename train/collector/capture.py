"""Frame retention and quality-checked enrollment image saving."""

import time

from train.config import MAX_SAVE_RESULT_AGE_SECONDS, RECENT_FRAME_LIMIT
from train.dataset import save_face_forms
from train.face_processing import crop_face


def remember_frame(recent_frames, frame_id, frame):
    recent_frames[frame_id] = frame

    while len(recent_frames) > RECENT_FRAME_LIMIT:
        oldest_frame_id = min(recent_frames)
        del recent_frames[oldest_frame_id]


def save_current_face(
    latest_faces,
    latest_detection_frame_id,
    latest_result_timestamp,
    recent_frames,
    folders,
    person_name,
    image_count,
    current_pose,
):
    if len(latest_faces) == 0:
        print("No face detected. Image not saved.")
        return False

    if len(latest_faces) > 1:
        print("Multiple faces detected. Keep only one person visible.")
        return False

    quality = latest_faces[0].get("quality", {})
    if not quality.get("accepted", False):
        print(
            "Face image quality is too low "
            f"({quality.get('reason', 'unavailable')}); image not saved."
        )
        return False

    result_age = time.perf_counter() - latest_result_timestamp
    if result_age > MAX_SAVE_RESULT_AGE_SECONDS:
        print("Detection result is stale. Wait for a fresh box.")
        return False

    source_frame = recent_frames.get(latest_detection_frame_id)
    if source_frame is None:
        print("Matching frame expired. Try saving again.")
        return False

    face_image = crop_face(source_frame, latest_faces[0]["bbox"])
    if face_image.size == 0:
        print("Invalid face crop")
        return False

    return save_face_forms(
        face_image=face_image,
        folders=folders,
        person_name=person_name,
        image_count=image_count,
        pose_name=current_pose,
    )
