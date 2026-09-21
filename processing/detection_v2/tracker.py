"""
Greedy-optimal IOU tracker with kit-colour-assisted matching and temporal
role stabilization.

This is a lightweight SORT-style tracker (bbox IOU + linear assignment),
not an appearance-embedding tracker — good enough to bridge short
occlusions/missed detections on a single continuous shot, not to re-identify
a player who leaves and re-enters frame later.

Role stabilization: each track keeps a rolling window of per-frame role
guesses (team_a/team_b/goalkeeper/referee/unknown from `team.TeamClassifier`)
and reports the majority vote as its stabilized role, so a single
misclassified frame doesn't flip the on-screen label.

`Tracker.total_tracks_created` is the fragment-count proxy: the total number
of distinct track IDs ever created in a run. Lower, for a similar detection
volume, means fewer tracking interruptions (occlusions/missed frames
forcing a new ID) — the same metric referenced by the 127 -> 87 comparison
in the project write-up. It is a continuity metric, not player-identity
accuracy.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .team import Role

BBox = tuple[float, float, float, float]

MAX_COST = 1.5  # cost above which a match is refused outright
COLOUR_WEIGHT = 0.3
ROLE_WINDOW = 15


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _colour_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.5  # neutral — neither helps nor hurts the match
    return float(np.linalg.norm(a - b) / (255.0 * np.sqrt(3)))  # normalized to [0, 1]


@dataclass
class Track:
    id: int
    bbox: BBox
    colour: np.ndarray | None
    role_history: deque[Role] = field(default_factory=lambda: deque(maxlen=ROLE_WINDOW))
    age: int = 0
    hits: int = 1
    first_frame: int = 0
    last_frame: int = 0

    @property
    def role(self) -> Role:
        if not self.role_history:
            return Role.UNKNOWN
        return Counter(self.role_history).most_common(1)[0][0]


class Tracker:
    def __init__(self, max_age: int = 10, iou_threshold: float = 0.2) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.total_tracks_created = 0

    def update(
        self,
        frame_idx: int,
        detections: list[tuple[BBox, np.ndarray | None, Role]],
    ) -> list[Track]:
        """
        `detections` is a list of (bbox, kit_colour, role_guess) for the
        current frame's person detections (already pitch-filtered).
        Returns the tracks matched/created this frame.
        """
        active_ids = list(self.tracks.keys())
        matched_det_idx: set[int] = set()

        if active_ids and detections:
            cost = np.full((len(active_ids), len(detections)), MAX_COST + 1.0)
            for i, tid in enumerate(active_ids):
                t = self.tracks[tid]
                for j, (bbox, colour, _role) in enumerate(detections):
                    overlap = iou(t.bbox, bbox)
                    if overlap < self.iou_threshold:
                        continue
                    cost[i, j] = (1.0 - overlap) + COLOUR_WEIGHT * _colour_distance(t.colour, colour)

            row_idx, col_idx = linear_sum_assignment(cost)
            for i, j in zip(row_idx, col_idx):
                if cost[i, j] > MAX_COST:
                    continue
                tid = active_ids[i]
                bbox, colour, role_guess = detections[j]
                t = self.tracks[tid]
                t.bbox = bbox
                t.colour = colour if colour is not None else t.colour
                t.role_history.append(role_guess)
                t.age = 0
                t.hits += 1
                t.last_frame = frame_idx
                matched_det_idx.add(j)

        # Age out unmatched tracks.
        for tid in active_ids:
            if tid not in self.tracks:
                continue
            t = self.tracks[tid]
            if t.last_frame != frame_idx:
                t.age += 1
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.age <= self.max_age}

        # New tracks for unmatched detections.
        for j, (bbox, colour, role_guess) in enumerate(detections):
            if j in matched_det_idx:
                continue
            t = Track(id=self._next_id, bbox=bbox, colour=colour, first_frame=frame_idx, last_frame=frame_idx)
            t.role_history.append(role_guess)
            self.tracks[t.id] = t
            self.total_tracks_created += 1
            self._next_id += 1

        return [t for t in self.tracks.values() if t.last_frame == frame_idx]
