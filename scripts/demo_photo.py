"""Offline demo on a real face photo, without a camera or the MediaPipe model.

Uses skimage's astronaut photo (NASA, public domain) and a hand-annotated lip
contour (data/test_face_lips.json). Saves figures into docs/figures/.

  python scripts/demo_photo.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from makeup.config import LIPS_INNER, LIPS_OUTER, NUM_LANDMARKS   # noqa: E402
from makeup.masks import lip_mask_v1, lip_mask_v2                 # noqa: E402
from makeup.products import load_lipsticks                        # noqa: E402
from makeup.render import render_lab, render_rgb                  # noqa: E402

OUT = ROOT / "docs" / "figures"


def load_face():
    from skimage import data
    img = np.ascontiguousarray(data.astronaut()[..., ::-1])       # RGB -> BGR
    ann = json.loads((ROOT / "data" / "test_face_lips.json").read_text())
    pts = np.zeros((NUM_LANDMARKS, 2), np.float32)
    pts[LIPS_OUTER] = ann["outer"]
    pts[LIPS_INNER] = ann["inner"]
    # work at 3x resolution so the small mouth is closer to a webcam close-up
    s = 3
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), pts * s + (s - 1) / 2


def crop(img, pts, pad=30):
    x0, y0 = pts[LIPS_OUTER].min(0).astype(int) - pad
    x1, y1 = pts[LIPS_OUTER].max(0).astype(int) + pad
    return img[y0:y1, x0:x1]


def label(img, text):
    bar = np.full((34, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def hgap(imgs, gap=8):
    sep = np.full((imgs[0].shape[0], gap, 3), 255, np.uint8)
    out = [imgs[0]]
    for im in imgs[1:]:
        out += [sep, im]
    return np.hstack(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img, pts = load_face()
    red = (40, 30, 190)

    # 1. mask v1 vs v2 with a strong colour, zoomed, to see the edge
    m1, m2 = lip_mask_v1(img.shape, pts), lip_mask_v2(img.shape, pts)
    a = crop(render_lab(img, m1, red, 1.0, 0.8), pts)
    b = crop(render_lab(img, m2, red, 1.0, 0.8), pts)
    z = lambda x: cv2.resize(x, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "w3_mask_v1_vs_v2.png"),
                hgap([label(z(crop(img, pts)), "original"), label(z(a), "mask v1 (week 2)"),
                      label(z(b), "mask v2 (week 3)")]))

    # 2. RGB vs LAB on the real photo
    c = crop(render_rgb(img, m2, red, 0.8), pts)
    d = crop(render_lab(img, m2, red, 0.8, 0.8), pts)
    cv2.imwrite(str(OUT / "w3_rgb_vs_lab_photo.png"),
                hgap([label(z(c), "RGB blending"), label(z(d), "CIELAB rendering")]))

    # 3. the product palette
    tiles = []
    for p in load_lipsticks():
        t = crop(render_lab(img, m2, p.bgr, 0.9, p.coverage), pts)
        tiles.append(label(t, f"{p.id} {p.name} ({p.coverage})"))
    row1, row2 = hgap(tiles[:4]), hgap(tiles[4:8])
    sep = np.full((8, row1.shape[1], 3), 255, np.uint8)
    cv2.imwrite(str(OUT / "w3_palette.png"), np.vstack([row1, sep, row2]))
    print("Saved figures to", OUT)


if __name__ == "__main__":
    main()
