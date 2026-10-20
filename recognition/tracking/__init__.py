"""Face tracking and bounding-box geometry."""

from .geometry import box_iou
from .tracker import FaceTracker

__all__ = ["FaceTracker", "box_iou"]
