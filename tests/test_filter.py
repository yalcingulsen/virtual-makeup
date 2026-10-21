import numpy as np

from makeup.landmarks import OneEuroFilter


def test_filter_reduces_jitter():
    rng = np.random.default_rng(1)
    true = np.zeros((40, 2), np.float32)
    f = OneEuroFilter()
    raw, smooth, prev_r, prev_s = [], [], None, None
    for k in range(200):
        x = true + rng.normal(0, 1, true.shape).astype(np.float32)
        y = f(x, k / 30)
        if prev_r is not None:
            raw.append(np.abs(x - prev_r).mean())
            smooth.append(np.abs(y - prev_s).mean())
        prev_r, prev_s = x, y
    assert np.mean(smooth) < 0.5 * np.mean(raw)


def test_filter_follows_a_moving_point():
    f = OneEuroFilter()
    for k in range(120):                      # point moves 2 px per frame
        y = f(np.array([[2.0 * k, 0.0]], np.float32), k / 30)
    assert abs(y[0, 0] - 2.0 * 119) < 10      # small lag only


def test_reset():
    f = OneEuroFilter()
    f(np.ones((1, 2)), 0.0)
    f.reset()
    assert f.x_prev is None
