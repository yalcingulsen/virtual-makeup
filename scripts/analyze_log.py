"""Summarise a timing log written by app.py.

  python scripts/analyze_log.py                 # newest log in logs/
  python scripts/analyze_log.py logs/xyz.csv
"""
import platform
import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(Path("logs").glob("timing_*.csv"))
df = pd.read_csv(path).iloc[30:]          # skip the first 30 frames (warm-up)
print(f"Log: {path}  ({len(df)} frames after warm-up)")
print(f"System: {platform.platform()} | {platform.processor() or platform.machine()}\n")

cols = ["landmark_ms", "mask_ms", "render_ms", "total_ms"]
speed = df.groupby(["mode", "mask_version"])[cols].mean().round(2)
speed["processing_fps"] = (1000 / speed["total_ms"]).round(1)
print("Processing time per frame (ms) - landmarks + mask + rendering:")
print(speed.to_string(), "\n")

if "frame_interval_ms" in df:
    iv = df["frame_interval_ms"].dropna().median()
    print(f"Real application speed (camera + processing + display): {1000 / iv:.1f} FPS "
          f"(median frame interval {iv:.1f} ms)\n")

if "still_test" in df and df["still_test"].sum() > 0:
    st = df[(df["still_test"] == 1)].dropna(subset=["raw_jitter_px"])
    raw, filt = st["raw_jitter_px"].mean(), st["filtered_jitter_px"].mean()
    print(f"Stillness test ({len(st)} frames, same frames for both):")
    print(f"  raw landmarks      {raw:.3f} px per frame")
    print(f"  One Euro filtered  {filt:.3f} px per frame")
    print(f"  reduction          {(1 - filt / raw) * 100:.0f}%")
else:
    print("No stillness test in this log. Press T in the app and hold your head still for 10 s.")
