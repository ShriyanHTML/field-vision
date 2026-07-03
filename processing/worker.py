"""
FieldVision Processing Worker — Modal.com

Pipeline:
  1. Download left + right videos from R2
  2. Sync timestamps (audio cross-correlation)
  3. Stitch into panoramic video (FFmpeg side-by-side + seam blend)
  4. Run YOLOv8 ball + player detection on each frame
  5. Render tracking overlays, auto-zoom following ball
  6. Detect goal events (ball crosses goal-line region)
  7. Cut highlight clips
  8. Upload outputs to R2
  9. POST webhook to Vercel with results

Run locally:  modal run processing/worker.py
Deploy:       modal deploy processing/worker.py
"""

import os
import json
import time
import subprocess
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------

app = modal.App("field-vision")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]",
        "ultralytics==8.3.0",
        "opencv-python-headless==4.10.0.84",
        "boto3==1.35.0",
        "numpy==1.26.4",
        "scipy==1.14.0",
        "requests==2.32.3",
    )
)

# ---------------------------------------------------------------------------
# Secrets — set these in your Modal dashboard:
#   modal secret create field-vision-secrets \
#     R2_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
#     R2_BUCKET=... VERCEL_WEBHOOK_URL=... MODAL_AUTH_TOKEN=...
# ---------------------------------------------------------------------------

secrets = [modal.Secret.from_name("field-vision-secrets")]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _download(s3, key: str, dest: Path, url: str = "") -> None:
    if url:
        import requests as _req
        print(f"Downloading via URL: {url[:80]}...")
        r = _req.get(url, stream=True, timeout=600)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    else:
        s3.download_file(os.environ["R2_BUCKET"], key, str(dest))


def _upload(s3, src: Path, key: str) -> None:
    import boto3.s3.transfer as s3t
    config = s3t.TransferConfig(
        multipart_threshold=1024 * 1024 * 200,   # only use multipart above 200 MB
        multipart_chunksize=1024 * 1024 * 100,   # 100 MB chunks (R2 minimum is 5 MB)
        max_concurrency=4,
        use_threads=True,
    )
    s3.upload_file(
        str(src), os.environ["R2_BUCKET"], key,
        ExtraArgs={"ContentType": "video/mp4"},
        Config=config,
    )


def _report(session_id: str, payload: dict) -> None:
    import requests
    base = os.environ.get("NEXT_PUBLIC_SITE_URL", "").rstrip("/")
    if not base:
        print("_report: NEXT_PUBLIC_SITE_URL not set, skipping")
        return
    url = f"{base}/api/sessions/{session_id}/report"
    try:
        r = requests.post(url, json=payload, timeout=30)
        print(f"_report {payload} → {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"_report FAILED: {e}")


# ---------------------------------------------------------------------------
# Step 1 — Sync & Stitch  (pure FFmpeg — 10-50x faster than Python per-frame)
# ---------------------------------------------------------------------------

def _extract_frame(path: Path, pos: float = 0.25) -> "np.ndarray | None":
    """Extract a single frame at `pos` fraction through the video."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * pos) - 1))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _audio_sync_offset(left_path: Path, right_path: Path, fps: float, tmp_dir: Path) -> int:
    """
    Cross-correlate audio tracks to find how many frames the right video
    starts after the left video (positive = right starts later).
    Returns frame offset to skip on the earlier video.
    """
    import numpy as np
    from scipy.signal import correlate, butter, filtfilt

    SR = 8000  # downsample to 8 kHz for speed
    audio_files = {}
    for side, path in [("left", left_path), ("right", right_path)]:
        out = tmp_dir / f"sync_{side}.wav"
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(path),
            "-vn", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
            str(out),
        ], capture_output=True, timeout=120)
        if r.returncode != 0 or not out.exists():
            print(f"Audio extract failed for {side}: {r.stderr.decode()[:200]}")
            return 0
        audio_files[side] = out

    def load_wav(p):
        import wave, struct
        with wave.open(str(p), "rb") as w:
            raw = w.readframes(w.getnframes())
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        arr /= 32768.0
        return arr

    al = load_wav(audio_files["left"])
    ar = load_wav(audio_files["right"])

    # High-pass filter to remove DC / low-freq rumble
    b, a = butter(4, 80 / (SR / 2), btype="high")
    al = filtfilt(b, a, al)
    ar = filtfilt(b, a, ar)

    # Limit to first 60 s (sufficient for sync detection)
    cap = SR * 60
    al = al[:cap]
    ar = ar[:cap]

    corr = correlate(al, ar, mode="full")
    lag_samples = int(np.argmax(np.abs(corr))) - (len(ar) - 1)
    lag_sec = lag_samples / SR
    lag_frames = int(round(lag_sec * fps))
    print(f"Audio sync: lag={lag_sec:.3f}s → {lag_frames} frames "
          f"({'right starts later' if lag_frames > 0 else 'left starts later'})")
    return lag_frames


def _compute_homography(left_frame, right_frame, W: int, H: int):
    """
    Compute homography that warps right frame into left frame coordinate system.
    Uses ORB features across full overlap zone.
    Returns (H_matrix, canvas_w, canvas_h, tx, ty) or None on failure.
    tx/ty = translation applied so the left frame origin maps to (tx, ty) on canvas.
    """
    import cv2
    import numpy as np

    try:
        search_w = int(W * 0.50)
        left_roi  = left_frame[:,  W - search_w : W]
        right_roi = right_frame[:, 0 : search_w]

        gray_l = cv2.cvtColor(left_roi,  cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=3000)
        kp_l, des_l = orb.detectAndCompute(gray_l, None)
        kp_r, des_r = orb.detectAndCompute(gray_r, None)

        if des_l is None or des_r is None or len(kp_l) < 10 or len(kp_r) < 10:
            print("Homography: not enough keypoints")
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des_r, des_l, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        print(f"Homography: {len(good)} good matches")

        if len(good) < 10:
            print("Homography: too few matches")
            return None

        # Map ROI-local coords back to full-frame coords
        pts_r_full = np.float32([kp_r[m.queryIdx].pt for m in good])
        pts_l_full = np.float32([kp_l[m.trainIdx].pt for m in good])
        pts_l_full[:, 0] += W - search_w   # shift left ROI to full-frame x

        H_mat, mask = cv2.findHomography(pts_r_full, pts_l_full, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        print(f"Homography: {inliers} inliers, H=\n{H_mat}")

        if H_mat is None or inliers < 8:
            print("Homography: poor fit")
            return None

        # Find output canvas that contains both frames
        corners_r = np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2)
        corners_r_warped = cv2.perspectiveTransform(corners_r, H_mat)
        corners_l = np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2)
        all_corners = np.concatenate([corners_l, corners_r_warped], axis=0)

        x_min = float(all_corners[:, 0, 0].min())
        y_min = float(all_corners[:, 0, 1].min())
        x_max = float(all_corners[:, 0, 0].max())
        y_max = float(all_corners[:, 0, 1].max())

        tx = max(0, -x_min)
        ty = max(0, -y_min)
        canvas_w = int(x_max + tx + 0.5)
        canvas_h = int(y_max + ty + 0.5)

        # Clip to reasonable output width (not wider than 2.5x a single frame)
        canvas_w = min(canvas_w, int(W * 2.5))
        canvas_h = min(canvas_h, int(H * 1.2))

        print(f"Canvas: {canvas_w}x{canvas_h}, translation=({tx:.1f},{ty:.1f})")
        return H_mat, canvas_w, canvas_h, tx, ty

    except Exception as e:
        print(f"Homography failed ({e})")
        return None


def _colour_match(left_frame, right_frame, left_keep: int, right_skip: int, W: int) -> str:
    """
    Full per-channel histogram matching on the overlap zone.
    Returns an FFmpeg colorlevels filter string to apply to the right video.
    """
    import cv2
    import numpy as np

    try:
        sample_w = max(30, int(W * 0.10))
        left_sample  = left_frame[:, max(0, left_keep - sample_w): left_keep]
        right_sample = right_frame[:, right_skip: right_skip + sample_w]

        # Per-channel (BGR) mean+std matching — maps right distribution onto left
        channel_names = ["b", "g", "r"]
        in_ranges, out_ranges = [], []

        for ch in range(3):
            lvals = left_sample[:, :, ch].astype(float).flatten()
            rvals = right_sample[:, :, ch].astype(float).flatten()

            lmean, lstd = np.mean(lvals), np.std(lvals) + 1e-6
            rmean, rstd = np.mean(rvals), np.std(rvals) + 1e-6

            # Linear transform: r_new = (r - rmean) * (lstd/rstd) + lmean
            scale = float(np.clip(lstd / rstd, 0.4, 2.5))
            shift = float(lmean - rmean * scale)

            # Map to colorlevels input range [0,255] → output range
            # colorlevels: rimin/rimax defines input black/white point
            # We'll keep input full range and adjust output
            in_black  = 0
            in_white  = 255
            # Output: what 0 and 255 map to after correction
            out_black = int(np.clip(shift, -50, 50))
            out_white = int(np.clip(255 * scale + shift, 200, 310))

            in_ranges.append((in_black, in_white))
            out_ranges.append((out_black, out_white))
            print(f"  {channel_names[ch]}: mean {rmean:.1f}→{lmean:.1f}  "
                  f"std {rstd:.1f}→{lstd:.1f}  scale={scale:.3f} shift={shift:.1f}")

        b_in, g_in, r_in = in_ranges
        b_out, g_out, r_out = out_ranges

        # FFmpeg colorlevels filter — normalize output to 0-255
        filt = (
            f"colorlevels="
            f"rimin={r_in[0]/255:.4f}:rimax={r_in[1]/255:.4f}:"
            f"gimin={g_in[0]/255:.4f}:gimax={g_in[1]/255:.4f}:"
            f"bimin={b_in[0]/255:.4f}:bimax={b_in[1]/255:.4f}:"
            f"romin={max(0,r_out[0])/255:.4f}:romax={min(255,r_out[1])/255:.4f}:"
            f"gomin={max(0,g_out[0])/255:.4f}:gomax={min(255,g_out[1])/255:.4f}:"
            f"bomin={max(0,b_out[0])/255:.4f}:bomax={min(255,b_out[1])/255:.4f}"
        )
        print(f"Colour match filter: {filt}")
        return filt
    except Exception as e:
        print(f"Colour match failed ({e}), no correction")
        return ""


def _colour_correct_frame(frame, scales, shifts):
    """Apply per-channel linear colour correction to a single frame."""
    import numpy as np
    out = frame.astype(np.float32)
    for ch in range(3):
        out[:, :, ch] = np.clip(out[:, :, ch] * scales[ch] + shifts[ch], 0, 255)
    return out.astype(np.uint8)


def sync_and_stitch(left_path: Path, right_path: Path, out_path: Path, session_id: str = "") -> None:
    """
    Panoramic stitch using perspective homography (warpPerspective).
    Right frame is geometrically warped into the left frame's coordinate system,
    then blended with a smooth gradient — no MOG2 oscillation, no diamond shape.
    Falls back to fixed side-by-side if homography fails.
    """
    import cv2
    import numpy as np

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", str(left_path)],
        capture_output=True, text=True, timeout=30,
    )
    parts = probe.stdout.strip().split(",")
    W, H = int(parts[0]), int(parts[1])
    num, den = map(int, (parts[2] if len(parts) > 2 else "25/1").split("/"))
    fps = num / den
    print(f"Video: {W}x{H} @ {fps:.2f}fps")

    # Use a stable frame from the first third for geometry calibration
    left_frame  = _extract_frame(left_path,  0.15)
    right_frame = _extract_frame(right_path, 0.15)

    # ── Colour correction (right → match left at the seam region) ──────────
    scales = [1.0, 1.0, 1.0]
    shifts = [0.0, 0.0, 0.0]
    if left_frame is not None and right_frame is not None:
        try:
            # Sample the inner 20% of each frame near their respective edges
            sw = max(80, int(W * 0.20))
            ls = left_frame[:,  W - sw :]        # right edge of left frame
            rs = right_frame[:, : sw]             # left edge of right frame
            for ch in range(3):
                lm  = float(np.mean(ls[:, :, ch]))
                rm  = float(np.mean(rs[:, :, ch]))
                lstd = float(np.std(ls[:, :, ch])) + 1e-6
                rstd = float(np.std(rs[:, :, ch])) + 1e-6
                scales[ch] = float(np.clip(lstd / rstd, 0.5, 2.0))
                shifts[ch] = lm - rm * scales[ch]
            print(f"Colour: scales={[f'{s:.3f}' for s in scales]} shifts={[f'{s:.1f}' for s in shifts]}")
        except Exception as e:
            print(f"Colour correction failed ({e})")

    sc_arr = np.array(scales, dtype=np.float32)
    sh_arr = np.array(shifts, dtype=np.float32)

    def colour_correct(frame):
        return np.clip(frame.astype(np.float32) * sc_arr + sh_arr, 0, 255).astype(np.uint8)

    # ── Camera alignment via ORB: rotation angle + vertical shift ───────────
    dy_shift    = 0
    rot_angle   = 0.0   # degrees — right camera tilt relative to left
    if left_frame is not None and right_frame is not None:
        try:
            search_w = int(W * 0.40)
            gray_l = cv2.cvtColor(left_frame[:, W - search_w:], cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right_frame[:, :search_w],    cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=3000)
            kp_l, des_l = orb.detectAndCompute(gray_l, None)
            kp_r, des_r = orb.detectAndCompute(gray_r, None)
            if des_l is not None and des_r is not None and len(kp_l) >= 8 and len(kp_r) >= 8:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                good = [m for m, n in bf.knnMatch(des_l, des_r, k=2)
                        if m.distance < 0.75 * n.distance]
                if len(good) >= 8:
                    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good])
                    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])
                    dy_shift = int(round(float(np.median(pts_l[:, 1] - pts_r[:, 1]))))
                    # estimateAffinePartial2D gives rotation + scale + translation
                    M_aff, inliers = cv2.estimateAffinePartial2D(
                        pts_r, pts_l, method=cv2.RANSAC, ransacReprojThreshold=8)
                    if M_aff is not None:
                        rot_angle = float(np.degrees(np.arctan2(M_aff[1, 0], M_aff[0, 0])))
                        rot_angle = float(np.clip(rot_angle, -35.0, 35.0))
                    print(f"ORB dy_shift={dy_shift}px  rot={rot_angle:.1f}° from {len(good)} matches")
        except Exception as e:
            print(f"ORB failed ({e}), dy_shift=0 rot=0")

    # ── Audio sync ──────────────────────────────────────────────────────────
    tmp_dir = out_path.parent
    frame_offset = _audio_sync_offset(left_path, right_path, fps, tmp_dir)

    cap_l = cv2.VideoCapture(str(left_path))
    cap_r = cv2.VideoCapture(str(right_path))

    if frame_offset > 0:
        for _ in range(frame_offset):
            cap_l.read()
        print(f"Skipped {frame_offset} frames from left")
    elif frame_offset < 0:
        for _ in range(-frame_offset):
            cap_r.read()
        print(f"Skipped {-frame_offset} frames from right")

    total_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Build stitch parameters ─────────────────────────────────────────────
    if False and homo_result is not None:  # homography disabled — side-by-side is more reliable
        H_mat, canvas_w, canvas_h, tx, ty = homo_result
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        H_c = T @ H_mat.astype(np.float64)
        tx_i, ty_i = int(round(tx)), int(round(ty))
        # Guarantee the canvas is large enough to hold the left frame at (tx_i, ty_i)
        canvas_w = max(canvas_w, tx_i + W)
        canvas_h = max(canvas_h, ty_i + H)

        # Cap output width at 1.85x a single frame to keep file sizes sane
        MAX_W = int(W * 1.85)
        MAX_W -= MAX_W % 2
        if canvas_w > MAX_W:
            out_w = MAX_W
            out_h = int(canvas_h * MAX_W / canvas_w)
            out_h -= out_h % 2
        else:
            out_w = canvas_w - (canvas_w % 2)
            out_h = canvas_h - (canvas_h % 2)

        # Build right frame coverage from its convex hull after warping —
        # this avoids corner pixel artifacts that warpPerspective creates at frame edges.
        corners_r = np.float32([[0,0],[W,0],[W,H],[0,H]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners_r, H_c)
        right_poly = np.clip(
            np.int32(warped_corners).reshape(-1, 2),
            [0, 0], [canvas_w - 1, canvas_h - 1]
        )
        right_poly_img = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillConvexPoly(right_poly_img, right_poly, 255)
        right_coverage = right_poly_img.astype(np.float32) / 255.0

        # Also clip anything left of the left frame's midpoint — right camera
        # should never contribute to the far-left portion of the canvas.
        ghost_clip_x = tx_i + W // 2
        right_coverage[:, :ghost_clip_x] = 0.0

        left_coverage = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        left_coverage[ty_i: ty_i + H, tx_i: tx_i + W] = 1.0

        # Overlap: both cameras have content
        overlap_mask = (left_coverage > 0.5) & (right_coverage > 0.5)

        # Build smooth gradient weight for left frame across the overlap
        weight_l = left_coverage.copy()
        if overlap_mask.any():
            cols = np.where(overlap_mask.any(axis=0))[0]
            x0, x1 = int(cols[0]), int(cols[-1])
            n = max(x1 - x0 + 1, 2)
            grad = np.linspace(1.0, 0.0, n, dtype=np.float32)
            rows = np.where(overlap_mask.any(axis=1))[0]
            weight_l[np.ix_(rows, np.arange(x0, x1 + 1))] = grad
            right_only = (right_coverage > 0.5) & (left_coverage < 0.5)
            weight_l[right_only] = 0.0

        weight_l_3 = weight_l[:, :, np.newaxis]
        right_alpha = (right_coverage > 0.1)[:, :, np.newaxis]

        # Auto-crop: find the largest rectangle with full-width/height content.
        # Rows where ≥70% of columns have content from at least one camera.
        valid = (left_coverage + right_coverage) > 0.1
        row_fill = valid.mean(axis=1)
        col_fill = valid.mean(axis=0)
        valid_rows = np.where(row_fill > 0.70)[0]
        valid_cols = np.where(col_fill > 0.70)[0]
        if len(valid_rows) > 0 and len(valid_cols) > 0:
            crop_y0 = int(valid_rows[0]);  crop_y1 = int(valid_rows[-1]) + 1
            crop_x0 = int(valid_cols[0]);  crop_x1 = int(valid_cols[-1]) + 1
        else:
            crop_y0, crop_y1 = ty_i, ty_i + H
            crop_x0, crop_x1 = tx_i, canvas_w
        crop_h = crop_y1 - crop_y0
        crop_w = crop_x1 - crop_x0

        MAX_W = int(W * 1.85)
        if crop_w > MAX_W:
            scale  = MAX_W / crop_w
            out_w  = MAX_W - (MAX_W % 2)
            out_h  = int(crop_h * scale) - (int(crop_h * scale) % 2)
        else:
            out_w  = crop_w - (crop_w % 2)
            out_h  = crop_h - (crop_h % 2)

        print(f"Homography stitch: canvas={canvas_w}x{canvas_h} crop=[{crop_x0}:{crop_x1},{crop_y0}:{crop_y1}] → output={out_w}x{out_h}")
        use_homo = True
    else:
        # Side-by-side stitch: 88/12 split puts seam in open grass past center circle
        left_keep  = int(W * 0.88)
        right_skip = int(W * 0.12)
        right_keep = W - right_skip
        total_w    = left_keep + right_keep
        out_w, out_h = total_w - (total_w % 2), H
        FEATHER = 80   # narrow blend — fast transition prevents center-circle ghost
        blend_start = max(0, left_keep - FEATHER // 2)
        blend_end   = min(total_w, left_keep + FEATHER // 2)
        bw = blend_end - blend_start
        r_bs = blend_start - left_keep + right_skip
        r_be = blend_end   - left_keep + right_skip
        print(f"Stitch: left_keep={left_keep} right_skip={right_skip} total_w={total_w} dy={dy_shift}px")
        use_homo = False

    enc = subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-movflags", "+faststart",
        str(out_path),
    ], stdin=subprocess.PIPE)

    # Pre-build rotation matrix for right frame (applied every frame)
    rot_M = None
    if abs(rot_angle) > 0.3:
        cx, cy = W / 2.0, H / 2.0
        rot_M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        print(f"Applying rotation correction: {rot_angle:.1f}° to every right frame")

    frame_idx = 0
    try:
        while True:
            ok_l, fl = cap_l.read()
            ok_r, fr = cap_r.read()
            if not ok_l or not ok_r:
                break

            if fr.shape[0] != H or fr.shape[1] != W:
                fr = cv2.resize(fr, (W, H))

            # Correct physical camera tilt of right camera
            if rot_M is not None:
                fr = cv2.warpAffine(fr, rot_M, (W, H), borderMode=cv2.BORDER_REPLICATE)

            fr_cc = colour_correct(fr)

            if use_homo:
                # Place left frame on canvas
                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                canvas[ty_i: ty_i + H, tx_i: tx_i + W] = fl

                # Warp right frame into left coordinate system
                right_w = cv2.warpPerspective(fr_cc, H_c, (canvas_w, canvas_h),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                # Kill ghost pixels on left side (warp artifact from rotation)
                right_w[:, :ghost_clip_x] = 0

                # Gradient blend in overlap, right fills right-only area
                blended = (canvas.astype(np.float32) * weight_l_3 +
                           right_w.astype(np.float32) * (1.0 - weight_l_3))
                canvas_out = np.where(right_alpha, blended, canvas.astype(np.float32))
                canvas_out = np.clip(canvas_out, 0, 255).astype(np.uint8)

                # Crop to valid content region (removes parallax dead zones / black triangles)
                canvas_out = canvas_out[crop_y0:crop_y1, crop_x0:crop_x1]
                if canvas_out.shape[1] != out_w or canvas_out.shape[0] != out_h:
                    canvas_out = cv2.resize(canvas_out, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Vertical alignment
                if dy_shift != 0:
                    M = np.float32([[1, 0, 0], [0, 1, dy_shift]])
                    fr_cc = cv2.warpAffine(fr_cc, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

                canvas_out = np.empty((H, total_w, 3), dtype=np.uint8)
                canvas_out[:, :left_keep]        = fl[:, :left_keep]
                canvas_out[:, left_keep:total_w] = fr_cc[:, right_skip: right_skip + right_keep]

                # Laplacian pyramid blend at seam
                if bw > 0 and r_bs >= 0 and r_be <= W:
                    L_c = fl[:,    blend_start: blend_end]
                    R_c = fr_cc[:, r_bs: r_be]
                    ws  = min(L_c.shape[1], R_c.shape[1], bw)
                    def _lap_blend_zone(LL, RR, levels=4):
                        h, w = LL.shape[:2]
                        mask = np.tile(np.linspace(1., 0., w, dtype=np.float32)[np.newaxis,:], (h,1))
                        def gp(img, n):
                            p=[img]
                            for _ in range(n): p.append(cv2.pyrDown(p[-1]))
                            return p
                        def lp(img, n):
                            g=gp(img,n)
                            out=[g[i].astype(np.float32)-cv2.pyrUp(g[i+1],dstsize=(g[i].shape[1],g[i].shape[0])).astype(np.float32) for i in range(n)]
                            out.append(g[n].astype(np.float32)); return out
                        lL,lR,gm=lp(LL,levels),lp(RR,levels),gp(mask,levels)
                        bl=[lL[i]*cv2.resize(gm[i],(lL[i].shape[1],lL[i].shape[0]))[:,:,np.newaxis]+lR[i]*(1-cv2.resize(gm[i],(lL[i].shape[1],lL[i].shape[0]))[:,:,np.newaxis]) for i in range(levels+1)]
                        r=bl[-1]
                        for i in range(levels-1,-1,-1): r=cv2.pyrUp(r,dstsize=(bl[i].shape[1],bl[i].shape[0]))+bl[i]
                        return np.clip(r,0,255).astype(np.uint8)
                    canvas_out[:, blend_start: blend_start + ws] = _lap_blend_zone(
                        L_c[:, :ws].astype(np.float32), R_c[:, :ws].astype(np.float32))
                if out_w != total_w:
                    canvas_out = cv2.resize(canvas_out, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            enc.stdin.write(canvas_out.tobytes())
            frame_idx += 1
            if total_frames > 0 and frame_idx % max(1, total_frames // 20) == 0:
                pct = 15 + int((frame_idx / total_frames) * 22)
                _report(session_id, {"status": "processing", "progress": pct})
                print(f"  Stitch {frame_idx}/{total_frames} ({pct}%)")
    finally:
        cap_l.release()
        cap_r.release()
        enc.stdin.close()
        enc.wait()

    if not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError("Stitch produced empty output")
    print(f"Stitch done: {frame_idx} frames → {out_path.stat().st_size} bytes")


# ---------------------------------------------------------------------------
# Step 2 — Ball & Player tracking
# ---------------------------------------------------------------------------

def run_tracking(input_path: Path, output_path: Path, session_id: str) -> list[dict]:
    """
    Runs YOLOv8 detection on each frame, draws bounding boxes,
    applies auto-zoom window following the ball, and returns goal events.
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")  # small — better small-object detection than nano

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for tracking: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w == 0 or h == 0:
        raise RuntimeError(f"Invalid video dimensions {w}x{h} for {input_path}")

    # Pipe frames directly to FFmpeg — faststart so browsers can stream immediately
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-movflags", "+faststart",
        str(output_path),
    ]
    enc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Viewport for auto-zoom (follows ball or player cluster)
    zoom_x, zoom_y = w // 2, h // 2
    zoom_w, zoom_h = w, h
    ZOOM_W_BALL   = w * 3 // 5   # 60% — ball zone: less aggressive, better resolution
    ZOOM_W_PLAYER = w // 2
    ZOOM_LERP     = 0.08   # smooth pan
    ZOOM_IN_LERP  = 0.06
    ZOOM_OUT_LERP = 0.015  # very slow zoom-out — don't cut player bodies

    last_ball_cx: int | None = None
    last_ball_cy: int | None = None
    SEARCH_R = int(w * 0.18)

    # Rolling buffers for position smoothing
    from collections import deque
    pos_history: deque = deque(maxlen=12)  # ~0.4s at 30fps
    no_detect_frames = 0
    HOLD_FRAMES = 20  # frames to hold position before zooming out

    goal_events: list[dict] = []
    frame_idx = 0

    # Goal-line x-positions (approximate for side-by-side stitch)
    # Left camera goal: x ~ w*0.07, Right camera goal: x ~ w*0.93
    GOAL_LEFT_X = int(w * 0.07)
    GOAL_RIGHT_X = int(w * 0.93)
    GOAL_BAND = int(w * 0.03)
    last_goal_frame = -int(fps * 5)  # debounce 5s

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, classes=[0, 32], conf=0.20)
        annotated = frame.copy()

        ball_cx, ball_cy = None, None
        ball_conf_best = 0.0
        all_players = []

        def _is_ball_shaped(x1, y1, x2, y2):
            """Reject detections that are not roughly circular (shoes, bags, etc.)."""
            bw, bh = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                return False
            aspect = bw / max(bh, 1)
            return 0.4 < aspect < 2.5   # round objects are ~1.0, bags are very wide

        for box in results[0].boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if cls == 32 and _is_ball_shaped(x1, y1, x2, y2):
                if conf > ball_conf_best:
                    ball_cx, ball_cy = cx, cy
                    ball_conf_best = conf
                cv2.circle(annotated, (cx, cy), 14, (0, 255, 80), 3)
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 80), -1)

                if frame_idx - last_goal_frame > fps * 5:
                    if abs(cx - GOAL_LEFT_X) < GOAL_BAND:
                        ts = frame_idx / fps
                        goal_events.append({"label": "Goal (left net)", "start_sec": max(0, ts - 5), "end_sec": ts + 3, "frame": frame_idx})
                        last_goal_frame = frame_idx
                    elif abs(cx - GOAL_RIGHT_X) < GOAL_BAND:
                        ts = frame_idx / fps
                        goal_events.append({"label": "Goal (right net)", "start_sec": max(0, ts - 5), "end_sec": ts + 3, "frame": frame_idx})
                        last_goal_frame = frame_idx

            elif cls == 0 and conf > 0.4:
                all_players.append((cx, cy))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 1)

        # Second pass: zoom into last known area at 2x resolution
        if ball_cx is None and last_ball_cx is not None:
            sx1 = max(0, last_ball_cx - SEARCH_R)
            sy1 = max(0, last_ball_cy - SEARCH_R)
            sx2 = min(w, last_ball_cx + SEARCH_R)
            sy2 = min(h, last_ball_cy + SEARCH_R)
            crop_up = cv2.resize(frame[sy1:sy2, sx1:sx2], None, fx=2, fy=2)
            res2 = model(crop_up, verbose=False, classes=[32], conf=0.18)
            for box2 in res2[0].boxes:
                x1c, y1c, x2c, y2c = map(int, box2.xyxy[0])
                if _is_ball_shaped(x1c, y1c, x2c, y2c):
                    ball_cx = sx1 + (x1c + x2c) // 4
                    ball_cy = sy1 + (y1c + y2c) // 4
                    cv2.circle(annotated, (ball_cx, ball_cy), 14, (0, 200, 255), 2)
                    cv2.circle(annotated, (ball_cx, ball_cy), 4, (0, 200, 255), -1)
                    break

        # Hough circles fallback — only on the pitch area (bottom 65% of frame = grass zone)
        if ball_cx is None:
            pitch_top = int(h * 0.35)   # sky/stands are above this line
            if last_ball_cx is not None:
                sx1 = max(0, last_ball_cx - SEARCH_R)
                sy1 = max(pitch_top, last_ball_cy - SEARCH_R)
                sx2 = min(w, last_ball_cx + SEARCH_R)
                sy2 = min(h, last_ball_cy + SEARCH_R)
            elif frame_idx % 30 == 0:   # global scan every 30 frames only
                sx1, sy1, sx2, sy2 = 0, pitch_top, w, h
            else:
                sx1, sy1, sx2, sy2 = 0, 0, 0, 0

            if sx2 > sx1 and sy2 > sy1:
                patch = frame[sy1:sy2, sx1:sx2]
                if patch.size > 0:
                    gray_p = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    gray_p = cv2.GaussianBlur(gray_p, (7, 7), 1.5)
                    circles = cv2.HoughCircles(gray_p, cv2.HOUGH_GRADIENT, dp=1.2,
                                               minDist=30, param1=60, param2=32,
                                               minRadius=5, maxRadius=28)
                    if circles is not None:
                        best_c, best_score = None, -1
                        for c in circles[0]:
                            cx_c, cy_c, r_c = int(c[0]), int(c[1]), int(c[2])
                            py1 = max(0, cy_c - r_c); py2 = min(patch.shape[0], cy_c + r_c)
                            px1 = max(0, cx_c - r_c); px2 = min(patch.shape[1], cx_c + r_c)
                            sub = patch[py1:py2, px1:px2]
                            if sub.size == 0:
                                continue
                            gray_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
                            bright = float(gray_sub.mean())
                            # Check circle is on green grass (HSV S>40, H in green range)
                            hsv_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
                            green = ((hsv_sub[:,:,0] > 30) & (hsv_sub[:,:,0] < 90) & (hsv_sub[:,:,1] > 40))
                            # Ball is bright (white) sitting on green — surrounding area should be green
                            # but the ball itself is white, so check surrounding ring
                            ring_bright = bright > 160   # ball must be very bright (white)
                            if ring_bright and best_score < bright:
                                best_score, best_c = bright, c
                        if best_c is not None:
                            ball_cx = int(sx1 + best_c[0])
                            ball_cy = int(sy1 + best_c[1])
                            cv2.circle(annotated, (ball_cx, ball_cy), int(best_c[2]) + 4, (255, 165, 0), 2)
                            cv2.circle(annotated, (ball_cx, ball_cy), 4, (255, 165, 0), -1)

        if ball_cx is not None:
            last_ball_cx, last_ball_cy = ball_cx, ball_cy
            no_detect_frames = 0
            # Expand viewport to include players close to the ball (full body)
            nearby_px = [p[0] for p in all_players if abs(p[0] - ball_cx) < ZOOM_W_BALL]
            if nearby_px:
                span = max(nearby_px) - min(nearby_px)
                target_w = max(ZOOM_W_BALL, min(span + 250, w * 3 // 5))
            else:
                target_w = ZOOM_W_BALL
            pos_history.append((ball_cx, ball_cy, target_w))
        elif all_players and no_detect_frames > HOLD_FRAMES:
            # Ball lost for too long — zoom to player cluster near last known ball pos
            if last_ball_cx is not None:
                # Pick players closest to last known ball position
                nearby = sorted(all_players, key=lambda p: abs(p[0] - last_ball_cx))[:5]
            else:
                nearby = all_players[:5]
            cx = int(sum(p[0] for p in nearby) / len(nearby))
            cy = int(sum(p[1] for p in nearby) / len(nearby))
            pos_history.append((cx, cy, ZOOM_W_PLAYER))
            no_detect_frames += 1
        else:
            no_detect_frames += 1

        if pos_history:
            avg_cx = int(sum(p[0] for p in pos_history) / len(pos_history))
            avg_cy = int(sum(p[1] for p in pos_history) / len(pos_history))
            avg_tw = int(sum(p[2] for p in pos_history) / len(pos_history))
            target_h = int(avg_tw * h / w)
            zoom_x = int(zoom_x + (avg_cx - zoom_x) * ZOOM_LERP)
            zoom_y = int(zoom_y + (avg_cy - zoom_y) * ZOOM_LERP)
            zoom_w = int(zoom_w + (avg_tw - zoom_w) * ZOOM_IN_LERP)
            zoom_h = int(zoom_h + (target_h - zoom_h) * ZOOM_IN_LERP)
        elif no_detect_frames > HOLD_FRAMES:
            zoom_w = int(zoom_w + (w - zoom_w) * ZOOM_OUT_LERP)
            zoom_h = int(zoom_h + (h - zoom_h) * ZOOM_OUT_LERP)

        # Clamp viewport
        half_w, half_h = zoom_w // 2, zoom_h // 2
        x1v = max(0, min(zoom_x - half_w, w - zoom_w))
        y1v = max(0, min(zoom_y - half_h, h - zoom_h))
        crop = annotated[y1v:y1v + zoom_h, x1v:x1v + zoom_w]
        out_frame = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)

        try:
            enc.stdin.write(out_frame.tobytes())
        except (BrokenPipeError, OSError):
            print(f"Tracking pipe broken at frame {frame_idx}")
            break
        frame_idx += 1

        # Report progress every 5%
        if total_frames > 0 and frame_idx % max(1, total_frames // 20) == 0:
            pct = 20 + int((frame_idx / total_frames) * 60)  # 20-80% range
            _report(session_id, {"status": "processing", "progress": pct})

    cap.release()
    try:
        enc.stdin.close()
    except OSError:
        pass
    stderr_bytes = enc.stderr.read()
    enc.wait()
    if enc.returncode != 0:
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"FFmpeg tracking failed after {frame_idx} frames (rc={enc.returncode}):\n{stderr_text}")

    return goal_events


# ---------------------------------------------------------------------------
# Step 3 — Cut highlight clips
# ---------------------------------------------------------------------------

def cut_highlights(source: Path, events: list[dict], out_dir: Path) -> list[dict]:
    clips = []
    for i, ev in enumerate(events):
        clip_path = out_dir / f"highlight_{i}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(source),
            "-ss", str(ev["start_sec"]),
            "-to", str(ev["end_sec"]),
            "-c", "copy",
            str(clip_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        clips.append({**ev, "clip_path": str(clip_path)})
    return clips


# ---------------------------------------------------------------------------
# Step A — CPU: download, transcode, stitch  (~$0.03/hr vs $0.59/hr for GPU)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=secrets,
    cpu=4,
    memory=8192,
    timeout=10800,
)
def stitch_session(session_id: str, left_key: str, right_key: str, left_url: str = "", right_url: str = "") -> None:
    s3 = _r2_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        try:
            # -- Download
            _report(session_id, {"status": "processing", "progress": 5})
            left_path  = tmp / "left.mp4"
            right_path = tmp / "right.mp4"
            print(f"Downloading left:  {left_key} (url={'yes' if left_url else 'no'})")
            print(f"Downloading right: {right_key} (url={'yes' if right_url else 'no'})")
            _download(s3, left_key,  left_path,  left_url)
            _download(s3, right_key, right_path, right_url)
            print(f"Downloaded  left: {left_path.stat().st_size} bytes")
            print(f"Downloaded right: {right_path.stat().st_size} bytes")

            # -- Transcode to h264 so OpenCV can decode frames
            _report(session_id, {"status": "processing", "progress": 10})
            left_h264  = tmp / "left_h264.mp4"
            right_h264 = tmp / "right_h264.mp4"
            for i, (src, dst) in enumerate([(left_path, left_h264), (right_path, right_h264)]):
                print(f"Transcoding {src.name} ({src.stat().st_size} bytes)…")
                with open(src, "rb") as fh:
                    magic = fh.read(12)
                is_webm = magic[4:8] == b"webm" or magic[:4] == b"\x1a\x45\xdf\xa3"
                print(f"  format: {'webm/VP9' if is_webm else 'mp4/H264'}")
                base_cmd = ["ffmpeg", "-y", "-fflags", "+discardcorrupt+genpts", "-threads", "0"]
                enc_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
                            "-c:a", "aac", "-movflags", "+faststart", str(dst)]
                try:
                    if is_webm:
                        with open(src, "rb") as f_in:
                            r = subprocess.run(base_cmd + ["-f", "webm", "-i", "pipe:0"] + enc_args,
                                               stdin=f_in, capture_output=True, timeout=1800)
                    else:
                        r = subprocess.run(base_cmd + ["-i", str(src)] + enc_args,
                                           capture_output=True, timeout=1800)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(f"Transcode timed out for {src.name}")
                if r.returncode != 0:
                    raise RuntimeError(f"Transcode failed {src.name} (rc={r.returncode}):\n"
                                       + r.stderr.decode(errors="replace")[-3000:])
                print(f"Transcoded → {dst.name} ({dst.stat().st_size} bytes)")
                _report(session_id, {"status": "processing", "progress": 11 + i * 3})

            # -- Stitch
            _report(session_id, {"status": "processing", "progress": 15})
            stitched_path = tmp / "stitched.mp4"
            sync_and_stitch(left_h264, right_h264, stitched_path, session_id)

            # -- Upload stitched panorama
            _report(session_id, {"status": "processing", "progress": 40})
            stitched_key = f"sessions/{session_id}/output/stitched.mp4"
            _upload(s3, stitched_path, stitched_key)
            print(f"Uploaded stitched: {stitched_key}")

            # -- Mark done now so users can watch the panorama immediately
            _report(session_id, {
                "status": "done",
                "progress": 100,
                "stitched_video_key": stitched_key,
            })

            # -- Hand off to GPU for tracking (updates tracked_video_key when done)
            track_session.spawn(session_id, stitched_key)

        except Exception as exc:
            _report(session_id, {"status": "error", "error_message": str(exc)})
            raise


# ---------------------------------------------------------------------------
# Step B — GPU: download stitched, YOLO tracking, highlights, upload
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=secrets,
    gpu="T4",
    memory=8192,
    timeout=7200,
)
def track_session(session_id: str, stitched_key: str) -> None:
    s3 = _r2_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        try:
            # -- Download stitched video
            print(f"Downloading stitched: {stitched_key}")
            stitched_path = tmp / "stitched.mp4"
            _download(s3, stitched_key, stitched_path)

            # -- Resize to 1280px wide before YOLO (much faster on T4)
            small_path = tmp / "stitched_small.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(stitched_path),
                "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                str(small_path),
            ], check=True, capture_output=True, timeout=600)

            # -- Ball/player tracking + auto-zoom
            tracked_path = tmp / "tracked.mp4"
            goal_events = run_tracking(small_path, tracked_path, session_id)

            # -- Upload tracked video
            tracked_key = f"sessions/{session_id}/output/tracked.mp4"
            _upload(s3, tracked_path, tracked_key)

            # -- Cut & upload highlight clips
            highlights = []
            if goal_events:
                clips_dir = tmp / "clips"
                clips_dir.mkdir()
                clips = cut_highlights(stitched_path, goal_events, clips_dir)
                for c in clips:
                    clip_key = f"sessions/{session_id}/output/{Path(c['clip_path']).name}"
                    _upload(s3, Path(c["clip_path"]), clip_key)
                    highlights.append({
                        "label": c["label"],
                        "start_sec": c["start_sec"],
                        "end_sec": c["end_sec"],
                        "clip_key": clip_key,
                    })

            # Only update the tracking keys — session is already marked done after stitch
            _report(session_id, {
                "tracked_video_key": tracked_key,
                "highlights": highlights,
            })

        except Exception as exc:
            print(f"track_session error (non-fatal): {exc}")
            raise


# ---------------------------------------------------------------------------
# HTTP webhook endpoint (called by Vercel /api/process)
# ---------------------------------------------------------------------------

@app.function(image=image, secrets=secrets)
@modal.fastapi_endpoint(method="POST")
def process(body: dict) -> dict:
    session_id = body.get("session_id")
    if not session_id:
        return {"error": "Missing session_id"}

    left_key  = body.get("left_key",  f"sessions/{session_id}/raw/left.mp4")
    right_key = body.get("right_key", f"sessions/{session_id}/raw/right.mp4")
    left_url  = body.get("left_url",  "")
    right_url = body.get("right_url", "")

    _report(session_id, {"status": "processing", "progress": 1})
    stitch_session.spawn(session_id, left_key, right_key, left_url, right_url)
    return {"jobId": f"modal-{session_id}", "ok": True}
