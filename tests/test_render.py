import cv2
import numpy as np

from makeup import synthetic
from makeup.masks import lip_mask_v2
from makeup.render import delta_e76, hex_to_bgr, render_lab, render_rgb

RED = (40, 30, 190)


def _l_std(img, sel):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0][sel].astype(np.float32).std()


def _setup():
    f = synthetic.frame()
    pts = synthetic.lip_landmarks()
    return f, lip_mask_v2(f.shape, pts)


def test_empty_mask_returns_same_frame():
    f, m = _setup()
    assert np.array_equal(render_lab(f, np.zeros_like(m), RED), f)
    assert np.array_equal(render_rgb(f, np.zeros_like(m), RED), f)


def test_zero_intensity_changes_nothing():
    f, m = _setup()
    assert np.abs(render_lab(f, m, RED, intensity=0.0).astype(int) - f).max() <= 1


def test_lab_keeps_texture_rgb_loses_it():
    f, m = _setup()
    core = m > 0.99
    orig = _l_std(f, core)
    assert abs(_l_std(render_lab(f, m, RED, 0.7), core) - orig) < 0.15 * orig
    assert _l_std(render_rgb(f, m, RED, 0.7), core) < 0.5 * orig


def test_pixels_outside_mask_unchanged():
    f, m = _setup()
    out = render_lab(f, m, RED, 1.0)
    assert np.array_equal(out[m == 0], f[m == 0])


def test_hex_and_delta_e():
    assert hex_to_bgr("#BE1E28") == (40, 30, 190)
    assert delta_e76((10, 20, 30), (10, 20, 30)) == 0.0
    assert delta_e76((0, 0, 0), (255, 255, 255)) > 99
