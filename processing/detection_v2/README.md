# Detection v2 — Phase Two prototype

Standalone prototype for RF-DETR detection + pitch-area filtering +
kit-colour team classification + tracking with role stabilization. It does
**not** touch `processing/worker.py` or the live Modal/Vercel pipeline —
run it locally against a clip to try it out.

## Setup

```bash
python -m venv .venv-detection-v2
source .venv-detection-v2/bin/activate
pip install -r processing/requirements-v2.txt
```

## Run

```bash
python -m processing.detection_v2.pipeline \
  --input match_clip.mp4 \
  --output annotated.mp4 \
  --metrics-json stats.json
```

Optional flags:

- `--manual-pitch boundary.json` — use a manually reviewed pitch polygon
  instead of automatic grass segmentation. Format:
  `{"points": [[x1, y1], [x2, y2], [x3, y3], ...]}` in source-video pixel
  coordinates (>=3 points).
- `--calibration-frames` / `--calibration-stride` — how many sampled frames
  (default 40, every 3rd frame) are used to fit the kit-colour clusters
  before the main pass. Increase for clips with brief camera cuts.
- `--conf-threshold` — detection confidence cutoff (default 0.5).
- `--max-frames` — stop early, for quick local iteration.

Output: an annotated MP4 (boxes colour-coded by stabilized role, ball
marker, pitch-boundary overlay, track IDs) plus a metrics JSON with frame
count, resolution, per-role detection counts, ball detections,
`ball_detections_by_method` (see below), `total_tracks_created` (the
track-fragment-count proxy), and FPS — in the same shape as the Phase
One/Two write-up's reported numbers, so a run here is directly comparable.

## Ball-detection fallback chain

The ball is the hardest object here — small, fast, and the first thing a
single detector pass misses. `ball.py` ports the same three-pass fallback
chain `processing/worker.py`'s `run_tracking()` uses, adapted to RF-DETR:

1. **Primary** — best ball-shaped box from the frame's main detection pass
   (shape-filtered: aspect ratio 0.4–2.5, so shoes/bags aren't mistaken for
   the ball).
2. **Zoom retry** — if nothing found and the ball's last known position is
   known, crop+upscale 2x around that position and re-run detection there
   at a lower confidence threshold (0.18). A second, targeted inference
   call only happens on this fallback path, not every frame.
3. **Hough-circle fallback** — if the detector still finds nothing, look
   for a bright circular blob via `cv2.HoughCircles`. worker.py restricts
   this to a fixed horizontal "grass region" cutoff; here it's restricted
   to the real `PitchMask` polygon instead, which also excludes the
   crowd/touchline areas to the sides, not just above a horizontal line.

`ball_detections_by_method` in the metrics JSON (`primary` /
`zoom_retry` / `hough_fallback` / `not_found`) shows how often each stage
had to carry the detection — useful for judging whether the fallback chain
is pulling its weight on a given clip.

## What's real vs. stubbed right now

- **Object detector**: `RFDETRSmall()` from the `rfdetr` package, loaded
  with its **COCO-pretrained** checkpoint — no fine-tuning on football
  footage. It only emits a generic `person` class (COCO id 1) and
  `sports ball` (id 37); it does **not** natively distinguish player vs.
  goalkeeper vs. referee. Accuracy will not match the reported 83% mAP@50 /
  88% recall until a fine-tuned checkpoint is substituted — see below.
- **Team/role classification** (`team.py`): reconstructs the
  player/goalkeeper/referee/team-A/team-B split via k-means clustering on
  torso kit colour, since the detector doesn't provide it. This is a
  documented heuristic (largest two colour clusters = the two outfield
  teams; the two smaller clusters = goalkeeper/referee by size), not a
  learned classifier — it will mislabel unusual kit-colour combinations.
- **Pitch detection** (`pitch.py`): HSV grass-colour segmentation +
  convex hull, not a trained pitch/line-keypoint model. Works on a
  reasonably steady broadcast-style shot; pass `--manual-pitch` for
  anything that needs to be reliable, matching the "manually reviewed
  pitch boundaries" approach described in the write-up.
- **Tracking** (`tracker.py`): real, not stubbed — an IOU + kit-colour-cost
  linear-assignment tracker (SORT-style) with rolling-window majority-vote
  role stabilization. This part behaves the same regardless of which
  detector/checkpoint feeds it.

## Swapping in a fine-tuned checkpoint

Once real trained weights exist (e.g. a Roboflow-hosted RF-DETR checkpoint
fine-tuned on player/goalkeeper/referee/ball classes):

1. Replace the `RFDETRSmall()` construction in `pipeline.py` with your
   checkpoint (`RFDETRSmall(pretrain_weights="path/or/roboflow-url")`, or
   the loading call your training run documents).
2. Update `coco.py`'s class-id constants (or replace it entirely) to match
   your fine-tuned model's class map, which should already include
   `goalkeeper` and `referee` — at that point `team.py` only needs to do
   the team-A/team-B kit-colour split for `player`-class detections, not
   invent goalkeeper/referee from colour.
3. `pitch.py` and `tracker.py` need no changes.
