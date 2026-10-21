"""Week-3 measurements on synthetic data (no camera or model needed).

  python scripts/benchmark_lipstick.py

Prints the tables used in the weekly report:
  A. lip mask accuracy, v1 (week 2) vs v2 (week 3), at three face sizes
  B. speed of masks and rendering
  C. colour accuracy (Delta E) for different coverage values
  D. texture preservation, RGB vs LAB
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from makeup import synthetic                                   # noqa: E402
from makeup.masks import lip_mask_v1, lip_mask_v2              # noqa: E402
from makeup.render import render_lab, render_rgb               # noqa: E402

RED = (40, 30, 190)


def timeit(fn, n=200):
    for _ in range(10):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) * 1000 / n


def mask_metrics(m, gt):
    leak = (m * (gt == 0)).sum() / gt.sum()            # colour put on skin / lip area
    full = (m[gt >= 0.999] >= 0.95).mean()            # share of lip pixels at full strength
    return leak, full


def iou(m, gt):
    pred, true = m > 0.5, gt > 0.5
    return (pred & true).sum() / max((pred | true).sum(), 1)


def table_a():
    print("A. Lip mask accuracy (synthetic lips, exact ground truth)")
    print("   IoU: shape of the mask without feathering. Leak and full colour: with feathering.")
    print(f"{'mouth width':>12} | {'mask':>4} | {'IoU':>6} | {'leak onto skin':>14} | {'lip at full colour':>18}")
    for scale in (1 / 3, 2 / 3, 4 / 3):
        pts = synthetic.lip_landmarks(outer=(60 * scale, 25 * scale), inner=(40 * scale, 5 * scale))
        f = synthetic.frame()
        gt = synthetic.true_lip_area(f.shape, pts)
        for name, fn in (("v1", lip_mask_v1), ("v2", lip_mask_v2)):
            leak, full = mask_metrics(fn(f.shape, pts), gt)
            j = iou(fn(f.shape, pts, False), gt)
            print(f"{round(120 * scale):>9} px | {name:>4} | {j:6.3f} | {leak * 100:13.1f}% | {full * 100:17.1f}%")
    print()


def table_b():
    f, pts = synthetic.frame(), synthetic.lip_landmarks()
    m = lip_mask_v2(f.shape, pts)
    print("B. Speed at 640x480 (ms per frame, mean of 200 runs)")
    print(f"  mask v1            {timeit(lambda: lip_mask_v1(f.shape, pts)):.2f}")
    print(f"  mask v2            {timeit(lambda: lip_mask_v2(f.shape, pts)):.2f}")
    print(f"  render RGB         {timeit(lambda: render_rgb(f, m, RED, 0.7)):.2f}")
    print(f"  render LAB         {timeit(lambda: render_lab(f, m, RED, 0.7)):.2f}")
    print()


def table_c():
    print("C. Colour accuracy: Delta E (CIE76) between mean lip colour and product colour,")
    print("   intensity 100%. Below ~2.3 is hard to see; 10+ is a clearly different colour.")
    bases = {"light pink lips": (150, 150, 210), "medium lips": (110, 110, 170), "dark lips": (70, 75, 110)}
    products = {"Classic Red": (40, 30, 190), "Nude Pink": (140, 140, 205), "Plum": (80, 40, 100)}
    print(f"{'':>16} | " + " | ".join(f"{p:>11}" for p in products) + "   (coverage)")
    for cov in (0.3, 0.6, 1.0):
        for bname, base in bases.items():
            f = synthetic.frame(skin=base)
            pts = synthetic.lip_landmarks()
            m = lip_mask_v2(f.shape, pts)
            core = m >= 0.999
            vals = []
            for prod in products.values():
                out = render_lab(f, m, prod, 1.0, coverage=cov)
                lab = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[core].mean(0)
                tgt = cv2.cvtColor(np.array([[prod]], np.float32) / 255, cv2.COLOR_BGR2LAB)[0, 0]
                vals.append(float(np.linalg.norm(lab - tgt)))
            print(f"{bname:>16} | " + " | ".join(f"{v:11.1f}" for v in vals) + f"   ({cov})")
    print()


def table_d():
    f, pts = synthetic.frame(), synthetic.lip_landmarks()
    m = lip_mask_v2(f.shape, pts)
    core = m >= 0.999

    def l_std(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0][core].astype(np.float32).std()

    print("D. Texture (std of L inside the lips, intensity 70%)")
    print(f"  original {l_std(f):.2f} | LAB {l_std(render_lab(f, m, RED, 0.7)):.2f} | "
          f"RGB {l_std(render_rgb(f, m, RED, 0.7)):.2f}")


if __name__ == "__main__":
    table_a()
    table_b()
    table_c()
    table_d()
