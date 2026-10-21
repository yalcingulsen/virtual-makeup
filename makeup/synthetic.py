"""Synthetic test data: a skin-coloured frame with lip-like texture and elliptical
lip landmarks. The true lip area (ring between two ellipses) is known exactly,
which lets us measure mask accuracy without a camera."""
import numpy as np

from .config import LIPS_INNER, LIPS_OUTER, NUM_LANDMARKS


def lip_landmarks(cx=320, cy=300, outer=(60, 25), inner=(40, 5)):
    pts = np.zeros((NUM_LANDMARKS, 2), np.float32)
    for idx_list, (ax, ay) in ((LIPS_OUTER, outer), (LIPS_INNER, inner)):
        n = len(idx_list)
        for i, idx in enumerate(idx_list):
            t = np.pi + 2 * np.pi * i / n          # start at the left mouth corner
            pts[idx] = [cx + ax * np.cos(t), cy - ay * np.sin(t)]
    return pts


def frame(h=480, w=640, skin=(150, 170, 200), seed=0):
    rng = np.random.default_rng(seed)
    f = np.full((h, w, 3), skin, np.float32)
    f += rng.normal(0, 6, f.shape)
    f += (8 * np.sin(np.arange(h)[:, None] / 2.0))[..., None]   # lip-line texture
    return np.clip(f, 0, 255).astype(np.uint8)


def true_lip_area(shape, pts, ss=8):
    """Exact lip coverage per pixel (0-1) of the landmark polygons, by supersampling."""
    import cv2
    h, w = shape[:2]
    big = np.zeros((h * ss, w * ss), np.uint8)
    cv2.fillPoly(big, [np.round(pts[LIPS_OUTER] * ss + (ss - 1) / 2).astype(np.int32)], 1)
    cv2.fillPoly(big, [np.round(pts[LIPS_INNER] * ss + (ss - 1) / 2).astype(np.int32)], 0)
    return big.reshape(h, ss, w, ss).mean(axis=(1, 3)).astype(np.float32)
