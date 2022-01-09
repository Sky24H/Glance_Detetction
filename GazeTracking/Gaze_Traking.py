"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from models import GazeTracking


def init_gaze_tracking():
    gaze = GazeTracking()
    return gaze

def track_gaze(gaze, frame, annotate=False):
    gaze.refresh(frame)
    if annotate:
        frame = gaze.annotated_frame()
    text = None

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # left_pupil = gaze.pupil_left_coords()
    # right_pupil = gaze.pupil_right_coords()
    # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    return text, frame