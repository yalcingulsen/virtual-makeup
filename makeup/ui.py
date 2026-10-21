"""Simple OpenCV user interface: colour palette with mouse selection,
a custom colour picker (HSV sliders) and an intensity slider."""
import cv2
import numpy as np

WINDOW = "Virtual Makeup Try-On"
PICKER = "Custom colour"
BAR_H = 78          # height of the palette bar under the camera image
SWATCH = 46


class ControlPanel:
    def __init__(self, products, intensity=70):
        self.products = products
        self.selected = 0                 # index into products, len(products) = custom
        self.custom_bgr = (60, 40, 200)
        self._picker_open = False
        self._swatch_boxes = []
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Intensity %", WINDOW, intensity, 100, lambda v: None)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self._frame_h = 0

    # ------------------------------------------------------------ state
    @property
    def is_custom(self):
        return self.selected == len(self.products)

    @property
    def colour_bgr(self):
        if self.is_custom:
            self._read_picker()
            return self.custom_bgr
        return self.products[self.selected].bgr

    @property
    def coverage(self):
        return 0.6 if self.is_custom else self.products[self.selected].coverage

    @property
    def name(self):
        return "Custom" if self.is_custom else self.products[self.selected].name

    @property
    def intensity(self):
        return max(cv2.getTrackbarPos("Intensity %", WINDOW), 0) / 100.0

    def select(self, index):
        self.selected = int(np.clip(index, 0, len(self.products)))
        if self.is_custom:
            self._open_picker()

    # ------------------------------------------------------------ custom colour picker
    def _open_picker(self):
        if self._picker_open:
            return
        cv2.namedWindow(PICKER, cv2.WINDOW_NORMAL)
        hsv = cv2.cvtColor(np.uint8([[self.custom_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        cv2.createTrackbar("Hue", PICKER, int(hsv[0]), 179, lambda v: None)
        cv2.createTrackbar("Saturation", PICKER, int(hsv[1]), 255, lambda v: None)
        cv2.createTrackbar("Value", PICKER, int(hsv[2]), 255, lambda v: None)
        self._picker_open = True

    def _read_picker(self):
        if not self._picker_open:
            return
        try:
            h = cv2.getTrackbarPos("Hue", PICKER)
            s = cv2.getTrackbarPos("Saturation", PICKER)
            v = cv2.getTrackbarPos("Value", PICKER)
        except cv2.error:                 # window closed by the user
            self._picker_open = False
            return
        bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
        self.custom_bgr = tuple(int(c) for c in bgr)
        preview = np.full((60, 300, 3), self.custom_bgr, np.uint8)
        cv2.imshow(PICKER, preview)

    # ------------------------------------------------------------ drawing and mouse
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or y < self._frame_h:
            return
        for i, (x0, y0, x1, y1) in enumerate(self._swatch_boxes):
            if x0 <= x <= x1 and y0 <= y - self._frame_h <= y1:
                self.select(i)
                return

    def compose(self, frame, status=""):
        """Return the camera frame with the palette bar and status text."""
        h, w = frame.shape[:2]
        self._frame_h = h
        bar = np.full((BAR_H, w, 3), 245, np.uint8)
        self._swatch_boxes = []
        items = [(p.bgr, p.name) for p in self.products] + [(self.custom_bgr, "Custom")]
        gap = 10
        for i, (bgr, label) in enumerate(items):
            x0 = gap + i * (SWATCH + gap)
            if x0 + SWATCH > w:
                break
            y0 = 8
            box = (x0, y0, x0 + SWATCH, y0 + SWATCH)
            self._swatch_boxes.append(box)
            cv2.rectangle(bar, box[:2], box[2:], bgr, -1)
            if i == self.selected:
                cv2.rectangle(bar, (x0 - 3, y0 - 3), (x0 + SWATCH + 3, y0 + SWATCH + 3), (30, 30, 30), 2)
            if label == "Custom":
                cv2.putText(bar, "+", (x0 + 15, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(bar, str(i + 1) if i < 9 else "", (x0 + 2, y0 + SWATCH + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        out = np.vstack([frame, bar])
        cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(out, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out
