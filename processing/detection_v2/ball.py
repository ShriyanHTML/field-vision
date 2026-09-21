"""
Ball-detection fallback chain — ported from `processing/worker.py`'s
`run_tracking()` so detection_v2 has the same fallback depth for the ball
specifically. The ball is the weak point of a single detector pass: small,
fast, and easily missed at broadcast resolution. worker.py compensates with
three passes; this module replicates that chain against RF-DETR/supervision
instead of Ultralytics YOLO.

  1. Primary — best ball-shaped detection from the frame's main detection
     pass (already computed for player detection; no extra inference).
  2. Zoom retry — if nothing found and the ball's last known position is
     known, crop+upscale 2x around that position and re-run detection at a
     lower confidence threshold. Small/blurry balls are much easier to
     catch once magnified.
  3. Hough-circle fallback — if the detector still finds nothing, look for
     a bright circular blob. worker.py restricted this to a fixed
     horizontal "grass region" cutoff (`y > 0.35 * height`); here we use
     the real `PitchMask` polygon instead, which is a strict improvement —
     it also excludes the crowd/touchline areas to the left and right, not
     just everything above a horizontal line.

`BallTracker` holds the running "last known position" state and orchestrates
the three passes per frame; call `.update(...)` once per frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .coco import BALL_CLASS_ID
from .pitch import PitchMask

BBox = tuple[float, float, float, float]

ZOOM_CONF_THRESHOLD = 0.18   # worker.py's lower threshold for the zoomed retry pass
HOUGH_BRIGHTNESS_MIN = 160.0  # worker.py: ball must be very bright (white) to count
HOUGH_GLOBAL_SCAN_INTERVAL = 30  # frames between full-frame Hough scans when ball has never been seen


def is_ball_shaped(bbox: BBox) -> bool:
    """Reject detections that are not roughly circular (shoes, bags, etc.)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return False
    aspect = w / max(h, 1e-6)
    return 0.4 < aspect < 2.5  # round objects are ~1.0, bags/shoes are far off that


def _best_ball_box(detections, conf_threshold: float) -> BBox | None:
    best_bbox, best_conf = None, 0.0
    for xyxy, conf, cls_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        if int(cls_id) != BALL_CLASS_ID or conf < conf_threshold:
            continue
        bbox = tuple(float(v) for v in xyxy)
        if not is_ball_shaped(bbox):
            continue
        if conf > best_conf:
            best_bbox, best_conf = bbox, float(conf)
    return best_bbox


class BallTracker:
    def __init__(self, frame_w: int, frame_h: int, search_radius_frac: float = 0.18) -> None:
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.search_radius = int(frame_w * search_radius_frac)
        self.last_pos: tuple[int, int] | None = None
        self.method_counts = {"primary": 0, "zoom_retry": 0, "hough_fallback": 0, "not_found": 0}

    def update(
        self,
        model,
        frame: np.ndarray,
        main_detections,
        conf_threshold: float,
        pitch_mask: PitchMask | None,
        frame_idx: int,
    ) -> BBox | None:
        bbox = _best_ball_box(main_detections, conf_threshold)
        method = "primary"

        if bbox is None and self.last_pos is not None:
            bbox = self._zoom_retry(model, frame)
            method = "zoom_retry"

        if bbox is None:
            bbox = self._hough_fallback(frame, pitch_mask, frame_idx)
            method = "hough_fallback"

        if bbox is None:
            self.method_counts["not_found"] += 1
            return None

        self.method_counts[method] += 1
        x1, y1, x2, y2 = bbox
        self.last_pos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return bbox

    def _zoom_retry(self, model, frame: np.ndarray) -> BBox | None:
        cx, cy = self.last_pos
        r = self.search_radius
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(self.frame_w, cx + r)
        y2 = min(self.frame_h, cy + r)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        upscaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
        detections = model.predict(img, threshold=ZOOM_CONF_THRESHOLD)
        crop_bbox = _best_ball_box(detections, ZOOM_CONF_THRESHOLD)
        if crop_bbox is None:
            return None

        # Map back: upscaled-crop coords -> crop coords (÷2) -> full-frame coords (+x1,y1)
        cbx1, cby1, cbx2, cby2 = crop_bbox
        return (x1 + cbx1 / 2, y1 + cby1 / 2, x1 + cbx2 / 2, y1 + cby2 / 2)

    def _hough_fallback(self, frame: np.ndarray, pitch_mask: PitchMask | None, frame_idx: int) -> BBox | None:
        if self.last_pos is not None:
            cx, cy = self.last_pos
            r = self.search_radius
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(self.frame_w, cx + r), min(self.frame_h, cy + r)
        elif frame_idx % HOUGH_GLOBAL_SCAN_INTERVAL == 0:
            x1, y1, x2, y2 = 0, 0, self.frame_w, self.frame_h
        else:
            return None  # don't run an expensive full-frame scan every single frame

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=60, param2=32, minRadius=5, maxRadius=28,
        )
        if circles is None:
            return None

        best_c, best_score = None, -1.0
        for c in circles[0]:
            cx_c, cy_c, r_c = int(c[0]), int(c[1]), int(c[2])
            full_x, full_y = x1 + cx_c, y1 + cy_c

            if pitch_mask is not None and not pitch_mask.contains((full_x, full_y)):
                continue  # off the playing area — not the ball

            py1, py2 = max(0, cy_c - r_c), min(patch.shape[0], cy_c + r_c)
            px1, px2 = max(0, cx_c - r_c), min(patch.shape[1], cx_c + r_c)
            sub = patch[py1:py2, px1:px2]
            if sub.size == 0:
                continue
            brightness = float(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).mean())
            if brightness > HOUGH_BRIGHTNESS_MIN and brightness > best_score:
                best_score, best_c = brightness, (full_x, full_y, r_c)

        if best_c is None:
            return None
        fx, fy, r = best_c
        return (fx - r, fy - r, fx + r, fy + r)
