"""
Phase Two detection prototype: RF-DETR object detection + pitch-area filtering
+ kit-colour team classification + temporal tracking with role stabilization.

This package is intentionally kept separate from `processing/worker.py`
(the live Modal pipeline). Nothing here is wired into the production
app/webhook flow. See processing/detection_v2/README.md for usage and for
the exact stub-vs-fine-tuned-model caveats.
"""

from .coco import BALL_CLASS_ID, PERSON_CLASS_ID
from .pitch import PitchMask
from .team import Role, TeamClassifier
from .tracker import Track, Tracker

__all__ = [
    "PERSON_CLASS_ID",
    "BALL_CLASS_ID",
    "PitchMask",
    "TeamClassifier",
    "Role",
    "Track",
    "Tracker",
]
