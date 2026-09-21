"""
End-to-end Phase Two prototype pipeline:

  RF-DETR-Small detection -> pitch-area filtering -> kit-colour team/role
  classification -> IOU tracking with role stabilization -> annotated video
  + JSON metrics.

Standalone by design — does not import or modify `processing/worker.py`,
the live Modal pipeline. Run it locally or in any Python 3.10+ environment
with `processing/requirements-v2.txt` installed:

    python -m processing.detection_v2.pipeline \\
        --input match_clip.mp4 --output annotated.mp4 --metrics-json stats.json

See README.md for the stub-vs-fine-tuned-model caveats (most importantly:
this uses a COCO-pretrained checkpoint, so "goalkeeper" / "referee" /
"team" labels come from kit-colour clustering, not from the detector
itself — swap in a fine-tuned RF-DETR checkpoint to change that).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .ball import BallTracker
from .coco import PERSON_CLASS_ID
from .pitch import PitchMask
from .team import Role, TeamClassifier, extract_kit_colour
from .tracker import Tracker

ROLE_COLOR = {
    Role.TEAM_A: (255, 160, 0),      # blue-ish (BGR)
    Role.TEAM_B: (0, 0, 255),        # red
    Role.GOALKEEPER: (0, 220, 0),    # green
    Role.REFEREE: (0, 220, 255),     # yellow
    Role.UNKNOWN: (200, 200, 200),   # grey
}


def _to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def _person_boxes(detections, conf_threshold: float) -> list[tuple[float, float, float, float]]:
    boxes = []
    for xyxy, conf, cls_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        if conf < conf_threshold or int(cls_id) != PERSON_CLASS_ID:
            continue
        boxes.append(tuple(float(v) for v in xyxy))
    return boxes


def calibrate_team_classifier(
    model,
    cap: cv2.VideoCapture,
    conf_threshold: float,
    calibration_frames: int,
    calibration_stride: int,
) -> TeamClassifier:
    samples: list[np.ndarray] = []
    frame_idx = 0
    frames_sampled = 0
    while frames_sampled < calibration_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % calibration_stride == 0:
            detections = model.predict(_to_pil(frame), threshold=conf_threshold)
            person_boxes = _person_boxes(detections, conf_threshold)
            for bbox in person_boxes:
                colour = extract_kit_colour(frame, bbox)
                if colour is not None:
                    samples.append(colour)
            frames_sampled += 1
        frame_idx += 1

    classifier = TeamClassifier(n_clusters=4)
    classifier.fit(samples)
    return classifier


def run_pipeline(
    input_path: str,
    output_path: str,
    metrics_json_path: str | None = None,
    manual_pitch_path: str | None = None,
    calibration_frames: int = 40,
    calibration_stride: int = 3,
    conf_threshold: float = 0.5,
    max_frames: int | None = None,
) -> dict:
    from rfdetr import RFDETRSmall

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Loading RF-DETR-Small (COCO-pretrained stub)…")
    model = RFDETRSmall()

    print(f"Calibrating team-colour clusters over ~{calibration_frames} sampled frames…")
    team_classifier = calibrate_team_classifier(
        model, cap, conf_threshold, calibration_frames, calibration_stride
    )
    if not team_classifier.is_fitted:
        print("  Warning: not enough person detections to fit team clusters — "
              "all detections will be labelled 'unknown'.")

    # Pitch mask: manual boundary if supplied, else auto grass segmentation
    # from a frame a quarter of the way through the clip.
    if manual_pitch_path:
        pitch_mask = PitchMask.from_manual(manual_pitch_path, (h, w))
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 4))
        ok, calib_frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read a frame for pitch segmentation")
        pitch_mask = PitchMask.from_frame(calib_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    tracker = Tracker()
    ball_tracker = BallTracker(frame_w=w, frame_h=h)
    counts = {"player": 0, "goalkeeper": 0, "referee": 0, "unknown": 0, "ball": 0}
    frame_idx = 0
    start = time.time()

    print("Running detection + tracking pass…")
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            detections = model.predict(_to_pil(frame), threshold=conf_threshold)
            person_boxes = _person_boxes(detections, conf_threshold)
            ball_box = ball_tracker.update(model, frame, detections, conf_threshold, pitch_mask, frame_idx)

            keep_mask = pitch_mask.filter_boxes(person_boxes)
            on_pitch = [b for b, keep in zip(person_boxes, keep_mask) if keep]

            det_tuples = []
            for bbox in on_pitch:
                colour = extract_kit_colour(frame, bbox)
                role_guess = team_classifier.predict(colour)
                det_tuples.append((bbox, colour, role_guess))

            active_tracks = tracker.update(frame_idx, det_tuples)

            annotated = pitch_mask.draw(frame)
            for t in active_tracks:
                role = t.role
                if role in (Role.TEAM_A, Role.TEAM_B):
                    counts["player"] += 1
                elif role == Role.GOALKEEPER:
                    counts["goalkeeper"] += 1
                elif role == Role.REFEREE:
                    counts["referee"] += 1
                else:
                    counts["unknown"] += 1

                x1, y1, x2, y2 = (int(v) for v in t.bbox)
                color = ROLE_COLOR[role]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"#{t.id} {role.value}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            if ball_box is not None:
                counts["ball"] += 1
                bx1, by1, bx2, by2 = ball_box
                bcx, bcy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
                cv2.circle(annotated, (bcx, bcy), 8, (0, 255, 80), 2)

            writer.write(annotated)
            frame_idx += 1

            if total_frames > 0 and frame_idx % max(1, total_frames // 20) == 0:
                print(f"  {frame_idx}/{total_frames} frames ({100 * frame_idx // total_frames}%)")
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start
    metrics = {
        "input": str(input_path),
        "output": str(output_path),
        "frames_processed": frame_idx,
        "resolution": f"{w}x{h}",
        "player_detections": counts["player"],
        "goalkeeper_detections": counts["goalkeeper"],
        "referee_detections": counts["referee"],
        "unknown_role_detections": counts["unknown"],
        "ball_detections": counts["ball"],
        "ball_detections_by_method": ball_tracker.method_counts,
        "total_tracks_created": tracker.total_tracks_created,
        "elapsed_seconds": round(elapsed, 2),
        "fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0,
    }

    print(json.dumps(metrics, indent=2))
    if metrics_json_path:
        Path(metrics_json_path).write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to source video")
    parser.add_argument("--output", default=None, help="Path for annotated output video (default: <input>_annotated.mp4)")
    parser.add_argument("--metrics-json", default=None, help="Optional path to write run metrics as JSON")
    parser.add_argument("--manual-pitch", default=None, help="Optional path to a manual pitch-boundary polygon JSON")
    parser.add_argument("--calibration-frames", type=int, default=40)
    parser.add_argument("--calibration-stride", type=int, default=3)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug: stop after N frames")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_annotated.mp4")

    run_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        metrics_json_path=args.metrics_json,
        manual_pitch_path=args.manual_pitch,
        calibration_frames=args.calibration_frames,
        calibration_stride=args.calibration_stride,
        conf_threshold=args.conf_threshold,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
