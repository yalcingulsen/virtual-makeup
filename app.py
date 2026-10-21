"""Virtual makeup try-on - live application (week 3 version).

Run:
  python app.py                 # webcam
  python app.py --image face.jpg  # a photo instead of the webcam

Mouse: click a colour swatch (the "+" swatch opens the custom colour picker).
Keys:
  1-9  select product        M  LAB / RGB rendering    V  mask v2 / v1
  F    feathering on/off     J  smoothing on/off       L  show landmarks
  S    screenshot            C  save RGB-vs-LAB comparison image
  T    stillness test: hold your head still for 10 s (measures landmark jitter)
  Q    quit (a timing log is saved in logs/)
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from makeup.config import LIPS_OUTER
from makeup.landmarks import FaceLandmarkDetector, OneEuroFilter
from makeup.masks import lip_mask
from makeup.products import load_lipsticks
from makeup.render import render_lab, render_rgb
from makeup.ui import WINDOW, ControlPanel

LOG_FIELDS = ["timestamp_ms", "frame_interval_ms", "landmark_ms", "mask_ms", "render_ms",
              "total_ms", "mode", "mask_version", "feather", "smooth",
              "raw_jitter_px", "filtered_jitter_px", "still_test"]
STILL_TEST_S = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="use a photo instead of the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("screenshots").mkdir(exist_ok=True)

    detector = FaceLandmarkDetector(video=args.image is None)
    panel = ControlPanel(load_lipsticks())
    filt = OneEuroFilter()

    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            raise SystemExit(f"Cannot read {args.image}")
        cap = None
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)

    mode, version, feather, smooth, show_lm = "LAB", 2, True, True, False
    rows, prev_raw, prev_filt, shot = [], None, None, 0
    still_until, t_prev = 0.0, None
    t0 = time.perf_counter()

    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
        else:
            frame = still.copy()

        t_start = time.perf_counter()
        interval = (t_start - t_prev) * 1000 if t_prev is not None else float("nan")
        t_prev = t_start
        ts = int((t_start - t0) * 1000)
        raw = detector.detect(frame, ts)
        t_lm = time.perf_counter()

        # The filter always runs on the raw landmarks, so raw and filtered jitter
        # are measured on exactly the same frames (a fair comparison).
        out, mask, pts = frame, None, None
        raw_jit = filt_jit = float("nan")
        if raw is not None:
            smoothed = filt(raw, t_start) if cap is not None else raw
            if prev_raw is not None:
                raw_jit = float(np.linalg.norm(raw[LIPS_OUTER] - prev_raw[LIPS_OUTER], axis=1).mean())
                filt_jit = float(np.linalg.norm(smoothed[LIPS_OUTER] - prev_filt[LIPS_OUTER], axis=1).mean())
            prev_raw, prev_filt = raw, smoothed
            pts = smoothed if smooth else raw
            mask = lip_mask(frame.shape, pts, feather, version)
        else:
            prev_raw = prev_filt = None
            filt.reset()
        t_mask = time.perf_counter()

        if mask is not None:
            colour, inten = panel.colour_bgr, panel.intensity
            if mode == "LAB":
                out = render_lab(frame, mask, colour, inten, panel.coverage)
            else:
                out = render_rgb(frame, mask, colour, inten)
            if show_lm:
                for x, y in pts.astype(int):
                    cv2.circle(out, (int(x), int(y)), 1, (0, 255, 0), -1)
        t_end = time.perf_counter()

        total = (t_end - t_start) * 1000
        in_test = time.perf_counter() < still_until
        rows.append([ts, round(interval, 2), round((t_lm - t_start) * 1000, 2),
                     round((t_mask - t_lm) * 1000, 2), round((t_end - t_mask) * 1000, 2),
                     round(total, 2), mode, version, int(feather), int(smooth),
                     round(raw_jit, 3), round(filt_jit, 3), int(in_test)])
        status = (f"{panel.name} | {mode} | mask v{version} | feather {'on' if feather else 'off'} | "
                  f"smooth {'on' if smooth else 'off'} | {panel.intensity:.0%} | {1000 / max(total, 1e-3):.0f} FPS")
        if pts is None:
            status = "No face found - " + status
        if in_test:
            status = f"STILLNESS TEST: hold your head still... {still_until - time.perf_counter():.0f} s"
        cv2.imshow(WINDOW, panel.compose(out, status))

        key = cv2.waitKey(1 if cap is not None else 30) & 0xFF
        ch = chr(key).lower() if key < 128 else ""
        if ch.isdigit() and ch != "0":
            panel.select(int(ch) - 1)
        elif ch == "m":
            mode = "RGB" if mode == "LAB" else "LAB"
        elif ch == "v":
            version = 1 if version == 2 else 2
        elif ch == "f":
            feather = not feather
        elif ch == "j":
            smooth = not smooth
        elif ch == "t":
            still_until = time.perf_counter() + STILL_TEST_S
        elif ch == "l":
            show_lm = not show_lm
        elif ch == "s":
            shot += 1
            cv2.imwrite(f"screenshots/shot_{shot:02d}.png", out)
        elif ch == "c" and mask is not None:
            a = render_rgb(frame, mask, panel.colour_bgr, panel.intensity)
            b = render_lab(frame, mask, panel.colour_bgr, panel.intensity, panel.coverage)
            cv2.imwrite("screenshots/rgb_vs_lab.png", np.hstack([a, b]))
        elif ch == "q" or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    if cap is not None:
        cap.release()
    detector.close()
    cv2.destroyAllWindows()
    log = Path("logs") / f"timing_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_FIELDS)
        w.writerows(rows)
    print(f"Saved {len(rows)} frames to {log}")


if __name__ == "__main__":
    main()
