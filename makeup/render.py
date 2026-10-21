"""Colour rendering of a product inside a soft mask."""
import cv2
import numpy as np

from .masks import mask_roi


def render_rgb(frame, mask, colour_bgr, intensity=0.7):
    """Baseline: plain alpha blending in RGB. Covers the lip texture (flat look)."""
    box = mask_roi(mask)
    if box is None or intensity <= 0:
        return frame
    y0, y1, x0, x1 = box
    out = frame.copy()
    m = (mask[y0:y1, x0:x1] * intensity)[..., None]
    colour = np.array(colour_bgr, np.float32)
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(patch * (1 - m) + colour * m, 0, 255).astype(np.uint8)
    return out


def render_lab(frame, mask, colour_bgr, intensity=0.7, coverage=0.6):
    """Texture-preserving rendering in CIELAB.

    a, b (colour) move towards the product colour.
    L (brightness) is shifted by the same amount for every pixel, so lip lines and
    highlights stay visible. `coverage` controls how far the average brightness
    moves towards the product: 0.3 = sheer (tinted balm), 1.0 = full coverage.
    """
    box = mask_roi(mask)
    if box is None or intensity <= 0:
        return frame
    y0, y1, x0, x1 = box
    out = frame.copy()
    patch, pm = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[colour_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    m = pm * intensity
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    mean_L = float((L * pm).sum() / max(pm.sum(), 1e-6))
    lab[..., 0] = L + m * coverage * (target[0] - mean_L)
    lab[..., 1] = (1 - m) * a + m * target[1]
    lab[..., 2] = (1 - m) * b + m * target[2]
    new = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # BGR->LAB->BGR in 8 bits is not exactly lossless, so only write back masked pixels
    out[y0:y1, x0:x1] = np.where((pm > 0)[..., None], new, patch)
    return out


def hex_to_bgr(hex_colour):
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def delta_e76(bgr1, bgr2):
    """CIE76 colour difference between two BGR colours (uint8 tuples)."""
    def lab(c):
        px = np.array([[c]], np.float32) / 255.0
        return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]
    return float(np.linalg.norm(lab(bgr1) - lab(bgr2)))
