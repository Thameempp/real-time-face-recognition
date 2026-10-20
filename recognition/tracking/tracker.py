from collections import deque

import numpy as np

from config import (
    SIMILARITY_THRESHOLD,
    TRACK_CENTER_MOTION_RATIO,
    TRACK_EXPIRY_SECONDS,
    TRACK_HISTORY_SIZE,
    TRACK_IOU_THRESHOLD,
    TRACK_MOTION_IOU_THRESHOLD,
    TRACK_RECOGNITION_INTERVAL,
    TRACK_SCALE_CHANGE_RATIO,
    TRACK_SWITCH_VOTES,
    TRACK_UNKNOWN_RETRY_INTERVAL,
)
from .geometry import _center_and_size, box_iou


class FaceTracker:
    """Greedy IoU tracker with cached embeddings and smoothed identity votes."""

    def __init__(self, threshold=SIMILARITY_THRESHOLD):
        self._tracks = {}
        self._next_track_id = 1
        self.threshold = threshold

    def update(self, detections, timestamp):
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if timestamp - track["last_seen"] <= TRACK_EXPIRY_SECONDS
        }

        possible_matches = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                overlap = box_iou(detection["bbox"], track["bbox"])
                if overlap >= TRACK_IOU_THRESHOLD:
                    possible_matches.append((overlap, detection_index, track_id))

        possible_matches.sort(reverse=True)
        matched_detections = set()
        matched_tracks = set()
        assigned = {}

        for _, detection_index, track_id in possible_matches:
            if detection_index in matched_detections or track_id in matched_tracks:
                continue
            matched_detections.add(detection_index)
            matched_tracks.add(track_id)
            assigned[detection_index] = track_id

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned:
                track_id = assigned[detection_index]
                track = self._tracks[track_id]
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                track = {
                    "bbox": detection["bbox"],
                    "last_seen": timestamp,
                    "last_embedding_at": None,
                    "last_embedding_bbox": None,
                    "name": "Unknown",
                    "score": 0.0,
                    "votes": deque(maxlen=TRACK_HISTORY_SIZE),
                    "needs_refresh": True,
                }
                self._tracks[track_id] = track

            track["bbox"] = detection["bbox"]
            track["last_seen"] = timestamp
            if not detection.get("quality", {}).get("accepted", False):
                track["needs_refresh"] = True
            detection["track_id"] = track_id

        return detections

    def needs_embedding(self, detection, timestamp):
        track = self._tracks[detection["track_id"]]

        if not detection.get("quality", {}).get("accepted", False):
            return False

        last_embedded_at = track["last_embedding_at"]
        if last_embedded_at is None or track["needs_refresh"]:
            return True

        retry_interval = (
            TRACK_UNKNOWN_RETRY_INTERVAL
            if track["name"] == "Unknown"
            or track["score"] < self.threshold + 0.05
            else TRACK_RECOGNITION_INTERVAL
        )
        if timestamp - last_embedded_at >= retry_interval:
            return True

        last_box = track["last_embedding_bbox"]
        current_box = detection["bbox"]
        center, width, height = _center_and_size(current_box)
        previous_center, previous_width, previous_height = _center_and_size(
            last_box
        )
        reference_size = max(previous_width, previous_height, 1.0)
        center_motion = (
            float(np.linalg.norm(center - previous_center)) / reference_size
        )
        scale_change = max(
            abs(width - previous_width) / previous_width,
            abs(height - previous_height) / previous_height,
        )

        return (
            box_iou(current_box, last_box) < TRACK_MOTION_IOU_THRESHOLD
            or center_motion >= TRACK_CENTER_MOTION_RATIO
            or scale_change >= TRACK_SCALE_CHANGE_RATIO
        )

    def update_identity(self, detection, name, score, timestamp):
        track = self._tracks[detection["track_id"]]
        name = str(name)
        score = float(score)
        track["votes"].append((name, score))
        track["last_embedding_at"] = timestamp
        track["last_embedding_bbox"] = detection["bbox"]
        track["needs_refresh"] = False

        current_name = track["name"]
        if current_name == "Unknown" and len(track["votes"]) == 1:
            track["name"] = name
        elif name == current_name:
            track["score"] = score
        else:
            matching_votes = sum(
                vote_name == name
                for vote_name, _ in track["votes"]
            )
            if matching_votes >= TRACK_SWITCH_VOTES:
                track["name"] = name
                track["score"] = score

        current_scores = [
            vote_score
            for vote_name, vote_score in track["votes"]
            if vote_name == track["name"]
        ]
        if current_scores:
            track["score"] = float(np.mean(current_scores))

    def apply_identity(self, detections):
        for detection in detections:
            track = self._tracks[detection["track_id"]]
            detection["name"] = track["name"]
            detection["score"] = float(track["score"])

        return detections
