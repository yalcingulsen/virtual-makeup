"""Landmark index sets of the MediaPipe face mesh (478 points incl. iris)."""

# Ordered around each contour. The first 11 points of each list go along the
# lower lip from one mouth corner to the other, the last 9 come back along the upper lip.
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              415, 310, 311, 312, 13, 82, 81, 80, 191]

MOUTH_LEFT, MOUTH_RIGHT = 61, 291

NUM_LANDMARKS = 478
