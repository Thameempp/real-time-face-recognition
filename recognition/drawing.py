"""Compatibility exports for live detection and performance drawing."""

from recognition.rendering.face import draw_detection
from recognition.rendering.overlay import draw_overlay

__all__ = ["draw_detection", "draw_overlay"]
