"""
Kit-colour clustering for team/role assignment.

The COCO-pretrained RF-DETR stub only emits a generic "person" class, so
there is no model-native player/goalkeeper/referee distinction to stabilize
— that has to be reconstructed from kit colour instead:

  1. Sample torso-region colour from a batch of calibration-frame person
     detections.
  2. K-means with k=4 splits those samples into 4 kit-colour groups.
  3. The two largest groups are assumed to be the two outfield teams
     (most detections on screen, most of the time, are outfield players).
     The two smaller groups are assumed to be goalkeeper and referee kits
     (by size: the bigger of the two minority groups -> goalkeeper, the
     smaller -> referee), since a single referee is typically on screen
     less often than a keeper.

That size-based split is a heuristic proxy, not a learned classifier — it
will mislabel a match with an unusually large/small refereeing crew, or
when a keeper's kit colour happens to closely match an outfield team's.
Once a fine-tuned RF-DETR checkpoint emits real player/goalkeeper/referee
classes directly, this module should be narrowed to just the team_a/team_b
split for outfield players, and role should come from the detector.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from sklearn.cluster import KMeans

BBox = tuple[float, float, float, float]


class Role(str, Enum):
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    UNKNOWN = "unknown"


def extract_kit_colour(frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
    """Median BGR colour of the torso region of a person bounding box."""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return None

    # Torso only: skip the head (dark hair / skin) and legs (socks/boots),
    # and inset from the left/right edges to avoid background bleed.
    top = y1 + int(h * 0.15)
    bottom = y1 + int(h * 0.55)
    left = x1 + int(w * 0.2)
    right = x1 + int(w * 0.8)
    crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
    if crop.size == 0:
        return None
    return np.median(crop.reshape(-1, 3).astype(np.float32), axis=0)


class TeamClassifier:
    def __init__(self, n_clusters: int = 4) -> None:
        self.n_clusters = n_clusters
        self._kmeans: KMeans | None = None
        self._cluster_role: dict[int, Role] = {}

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    def fit(self, colour_samples: list[np.ndarray]) -> None:
        samples = [c for c in colour_samples if c is not None]
        n_clusters = min(self.n_clusters, len(samples))
        if n_clusters < 2:
            self._kmeans = None
            self._cluster_role = {}
            return

        X = np.stack(samples)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X)

        counts = np.bincount(kmeans.labels_, minlength=n_clusters)
        order = np.argsort(-counts)  # cluster indices, largest group first

        role_by_rank = [Role.TEAM_A, Role.TEAM_B, Role.GOALKEEPER, Role.REFEREE]
        cluster_role = {
            int(cluster_idx): role_by_rank[rank] if rank < len(role_by_rank) else Role.UNKNOWN
            for rank, cluster_idx in enumerate(order)
        }

        self._kmeans = kmeans
        self._cluster_role = cluster_role

    def predict(self, colour: np.ndarray | None) -> Role:
        if colour is None or self._kmeans is None:
            return Role.UNKNOWN
        cluster_idx = int(self._kmeans.predict(colour.reshape(1, -1))[0])
        return self._cluster_role.get(cluster_idx, Role.UNKNOWN)
