# Real-Time Virtual Makeup Try-On

Graduation project, Computer Engineering — Yalçın Gülşen (advisor: Prof. Dr. Gazi Erkan Bostancı).

A webcam application that detects facial landmarks with MediaPipe and applies lipstick
in the CIELAB colour space so that the natural lip texture stays visible.
Current version: **week 3** (lipstick module, product palette, custom colour picker).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download the face model and save it as `models/face_landmarker.task`:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

## Run

```bash
python app.py                    # webcam
python app.py --image photo.jpg  # a photo
```

Click a colour swatch to choose a lipstick; the `+` swatch opens a custom colour picker.
The *Intensity* slider sets how strong the colour is.

| Key | Action |
| --- | --- |
| 1–9 | select product |
| M | LAB / RGB rendering |
| V | mask v2 (week 3) / v1 (week 2) |
| F | feathering on/off |
| J | landmark smoothing on/off |
| L | show landmarks |
| S | screenshot, C: RGB-vs-LAB comparison |
| Q | quit (saves a timing log in `logs/`) |

## Tests and measurements (no camera needed)

```bash
python tests/run_tests.py              # or: pytest
python scripts/benchmark_lipstick.py   # mask accuracy, speed, colour accuracy, texture
python scripts/demo_photo.py           # figures on a test photo -> docs/figures/
python scripts/analyze_log.py          # after running app.py: speed and jitter from the log
```

## Project structure

```
app.py                  live application
makeup/
  config.py             landmark index sets
  landmarks.py          MediaPipe wrapper + One Euro filter
  masks.py              lip masks (v1 week 2, v2 week 3)
  render.py             RGB and CIELAB rendering, colour helpers
  products.py           product catalogue loader
  ui.py                 palette, colour picker, intensity slider
  synthetic.py          synthetic test data with exact ground truth
data/
  products.json         lipstick catalogue (colour, coverage, finish)
  test_face_lips.json   hand-annotated lip contour for the offline photo demo
scripts/                benchmark, photo demo, log analysis
tests/                  unit tests
docs/figures/           figures for the weekly reports
```

The offline demo uses the astronaut photo bundled with scikit-image (NASA, public domain)
with a hand-annotated lip contour, so it runs without the camera or the model.
