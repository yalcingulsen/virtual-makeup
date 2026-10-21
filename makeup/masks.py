"""Region masks built from facial landmarks.

Two versions of the lip mask are kept so they can be compared:
  v1 (week 2): polygon filled on whole pixels, then blurred on both sides of the edge.
  v2 (week 3): polygon filled with sub-pixel precision (supersampling), then
               feathered only towards the inside, so no colour leaks onto the skin.
"""
import cv2
import numpy as np

from .config import LIPS_INNER, LIPS_OUTER, MOUTH_LEFT, MOUTH_RIGHT


def mouth_width(pts):
    return float(np.linalg.norm(pts[MOUTH_LEFT] - pts[MOUTH_RIGHT]))


def _fill_subpixel(shape, polygons_add, polygons_sub, ss=4):
    """Exact-ish pixel coverage (0-1) by supersampling, computed only inside the
    bounding box of the polygons so it stays fast.

    (cv2.fillPoly with LINE_AA was tried first, but the tests showed it makes the
    polygon about half a pixel too large, so colour leaked onto the skin.)"""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    allpts = np.vstack(polygons_add)
    x0 = max(int(np.floor(allpts[:, 0].min())) - 2, 0)
    y0 = max(int(np.floor(allpts[:, 1].min())) - 2, 0)
    x1 = min(int(np.ceil(allpts[:, 0].max())) + 3, w)
    y1 = min(int(np.ceil(allpts[:, 1].max())) + 3, h)
    if x1 <= x0 or y1 <= y0:
        return m
    big = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    origin = np.array([x0, y0], np.float32)

    def to_big(p):
        return np.round((p - origin) * ss + (ss - 1) / 2).astype(np.int32)

    cv2.fillPoly(big, [to_big(p) for p in polygons_add], 1)
    if polygons_sub:
        cv2.fillPoly(big, [to_big(p) for p in polygons_sub], 0)
    m[y0:y1, x0:x1] = big.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))
    return m


def _fill(shape, polygons_add, polygons_sub, subpixel):
    if subpixel:
        return _fill_subpixel(shape, polygons_add, polygons_sub)
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [p.astype(np.int32) for p in polygons_add], 255)
    if polygons_sub:
        cv2.fillPoly(m, [p.astype(np.int32) for p in polygons_sub], 0)
    return m.astype(np.float32) / 255.0


def lip_mask_v1(shape, pts, feather=True):
    """Week-2 mask: integer polygon + symmetric Gaussian feathering."""
    mask = _fill(shape, [pts[LIPS_OUTER]], [pts[LIPS_INNER]], subpixel=False)
    if feather:
        k = int(max(3, mouth_width(pts) * 0.08)) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def lip_mask_v2(shape, pts, feather=True, feather_ratio=0.03):
    """Week-3 mask: sub-pixel polygon + inward-only feathering.

    feather_ratio: width of the soft edge as a fraction of the mouth width.
    """
    hard = _fill(shape, [pts[LIPS_OUTER]], [pts[LIPS_INNER]], subpixel=True)
    if not feather:
        return hard
    box = mask_roi(hard, pad=2)
    if box is None:
        return hard
    y0, y1, x0, x1 = box
    r = max(1.0, mouth_width(pts) * feather_ratio)
    inside = (hard[y0:y1, x0:x1] > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3)   # distance to the lip edge
    ramp = np.clip(dist / r, 0.0, 1.0)
    out = np.zeros_like(hard)
    out[y0:y1, x0:x1] = ramp * ramp * (3 - 2 * ramp)     # smoothstep: 0 at edge, 1 inside
    return out


def lip_mask(shape, pts, feather=True, version=2):
    return lip_mask_v2(shape, pts, feather) if version == 2 else lip_mask_v1(shape, pts, feather)


def mask_roi(mask, pad=4, thr=0.0):
    """Bounding box (y0, y1, x0, x1) of the non-zero part of a mask, or None."""
    rows = np.flatnonzero(mask.max(axis=1) > thr)
    if len(rows) == 0:
        return None
    cols = np.flatnonzero(mask[rows[0]:rows[-1] + 1].max(axis=0) > thr)
    h, w = mask.shape
    return (max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, h),
            max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, w))
