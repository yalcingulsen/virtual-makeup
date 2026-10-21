"""Facial landmark detection (MediaPipe Face Landmarker) and temporal smoothing."""
import math
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012), applied to every coordinate.

    min_cutoff: lower -> less jitter when still, but more lag.
    beta:       higher -> less lag when moving fast.
    """

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, np.float32)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class FaceLandmarkDetector:
    """Thin wrapper around MediaPipe Face Landmarker.

    detect(frame_bgr, timestamp_ms) -> (478, 2) array of pixel coordinates, or None.
    """

    def __init__(self, model_path=DEFAULT_MODEL, video=True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\nDownload it from:\n  {MODEL_URL}\n"
                f"and save it as models/face_landmarker.task")
        self._mp = mp
        self.video = video
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode, num_faces=1)
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms=0):
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        else:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], np.float32)

    def close(self):
        self._landmarker.close()
