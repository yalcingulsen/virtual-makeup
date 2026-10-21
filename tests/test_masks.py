import numpy as np

from makeup import synthetic
from makeup.masks import lip_mask_v1, lip_mask_v2, mask_roi


def _setup():
    f = synthetic.frame()
    pts = synthetic.lip_landmarks()
    return f, pts, synthetic.true_lip_area(f.shape, pts)


def test_masks_are_between_0_and_1():
    f, pts, _ = _setup()
    for m in (lip_mask_v1(f.shape, pts), lip_mask_v2(f.shape, pts)):
        assert m.min() >= 0.0 and m.max() <= 1.0 and m.max() > 0.9


def test_mouth_opening_is_not_coloured():
    f, pts, _ = _setup()
    m = lip_mask_v2(f.shape, pts)
    assert m[300, 320] == 0.0          # centre of the mouth opening (teeth)


def test_v2_does_not_leak_onto_skin():
    f, pts, gt = _setup()
    m = lip_mask_v2(f.shape, pts)
    assert float((m * (gt == 0)).sum()) == 0.0


def test_v1_leaks_onto_skin():
    f, pts, gt = _setup()
    m = lip_mask_v1(f.shape, pts)
    assert float((m * (gt == 0)).sum()) > 0.0


def test_roi_contains_mask():
    f, pts, _ = _setup()
    m = lip_mask_v2(f.shape, pts)
    y0, y1, x0, x1 = mask_roi(m)
    assert m.sum() == m[y0:y1, x0:x1].sum()
    assert mask_roi(np.zeros_like(m)) is None
