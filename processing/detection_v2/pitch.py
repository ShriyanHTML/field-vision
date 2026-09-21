"""
Playing-area filtering.

Two ways to get a pitch boundary polygon:
  - `PitchMask.from_frame(frame)`   — automatic HSV grass segmentation.
  - `PitchMask.from_manual(path)`   — a manually reviewed polygon, saved as
    JSON: {"points": [[x1, y1], [x2, y2], ...]} in source-video pixel coords.

Detections whose foot position (bbox bottom-center — a better proxy for
"where the player is standing" than the box center) falls outside the
polygon are dropped, to suppress off-pitch detections (crowd, technical
area, subs bench) as the camera pans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BBox = tuple[float, float, float, float]


@dataclass
class PitchMask:
    polygon: np.ndarray  # (N, 2) int32 points, in frame pixel coordinates
    frame_shape: tuple[int, int]  # (height, width) it was computed for

    @classmethod
    def from_manual(cls, path: str | Path, frame_shape: tuple[int, int]) -> "PitchMask":
        data = json.loads(Path(path).read_text())
        points = np.array(data["points"], dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"{path} must contain >=3 [x, y] points")
        return cls(polygon=points, frame_shape=frame_shape)

    @classmethod
    def from_frame(cls, frame: np.ndarray) -> "PitchMask":
        """
        Segment the largest green region in `frame` and take its convex hull
        as an approximate playing-area boundary. This is a stand-in for a
        real fine-tuned pitch/line-detection model — it works reasonably on
        a single stationary or slowly-panning broadcast-style camera, but
        will under/over-segment on very different grass tones or heavy
        shadow. Prefer `from_manual` for anything that matters.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Broad "grass green" band; deliberately generous since we clean up
        # with morphology + largest-contour rather than tight thresholds.
        lower = np.array([25, 30, 30])
        upper = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # No grass found (indoor / heavily cropped shot) — fall back to
            # "everything is in play" rather than dropping every detection.
            full_frame = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
            return cls(polygon=full_frame, frame_shape=(h, w))

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest).reshape(-1, 2)
        return cls(polygon=hull, frame_shape=(h, w))

    def contains(self, point: tuple[float, float]) -> bool:
        result = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)
        return result >= 0

    def contains_bbox_foot(self, bbox: BBox) -> bool:
        x1, y1, x2, y2 = bbox
        foot = ((x1 + x2) / 2.0, y2)
        return self.contains(foot)

    def filter_boxes(self, boxes: list[BBox]) -> list[bool]:
        """Returns a keep-mask (same length/order as `boxes`)."""
        return [self.contains_bbox_foot(b) for b in boxes]

    def draw(self, frame: np.ndarray, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        out = frame.copy()
        cv2.polylines(out, [self.polygon], isClosed=True, color=color, thickness=2)
        return out
